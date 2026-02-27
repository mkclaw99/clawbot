"""
Correlation Engine — nightly pairwise + lag correlation analysis.

For each pair of symbols (A, B):
  - Contemporaneous Pearson r for 30 / 60 / 90-day windows
  - Lag correlations (A leads B by 1–3 days) to detect leading indicators

Results stored in market.db → correlation_results table.
Injected into strategy context as context["correlations"].

Key outputs used by strategies:
  - Positive correlation peers  → momentum confirmation (if peer is up, be more aggressive)
  - Negative correlation peers  → hedge signal (if anti-correlated peer spikes, be cautious)
  - Leading indicators          → early entry signals (A leads B by 1 day)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from core.market_db import MarketDatabase
from crawler.price_historian import FULL_UNIVERSE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERIODS        = [30, 60, 90]   # rolling windows in trading days
MAX_LAG        = 3              # days of lag to test for leading indicators
MIN_ABS_R      = 0.45           # minimum |r| to store (filters noise)
MAX_P_VALUE    = 0.05           # maximum p-value (statistical significance)
MIN_OVERLAP    = 20             # minimum overlapping observations for a pair


# ---------------------------------------------------------------------------
# CorrelationEngine
# ---------------------------------------------------------------------------

class CorrelationEngine:
    """
    Computes pairwise correlations from market.db price bars.
    Designed to run nightly after price data is refreshed.
    """

    def __init__(self, db: MarketDatabase | None = None) -> None:
        self._db = db or MarketDatabase()

    def run(self, universe: list[str] | None = None) -> dict:
        """
        Full nightly correlation run.
        Downloads returns matrix, computes all pairs for all periods and lags,
        saves to DB.

        Returns a summary dict.
        """
        symbols = universe or self._db.get_available_symbols()
        if not symbols:
            logger.warning("[CorrelationEngine] No symbols in DB — run update-prices first")
            return {"error": "no data"}

        logger.info(
            f"[CorrelationEngine] Computing correlations for "
            f"{len(symbols)} symbols across {PERIODS} day windows + lags 0–{MAX_LAG}"
        )
        t0 = time.time()
        total_pairs = 0

        for period in PERIODS:
            returns = self._db.get_returns_matrix(symbols, days=period)
            if returns.empty or returns.shape[1] < 2:
                logger.warning(f"[CorrelationEngine] Insufficient data for period={period}d")
                continue

            avail = list(returns.columns)
            logger.info(f"[CorrelationEngine] period={period}d | {len(avail)} symbols with data")

            # Contemporaneous (lag=0)
            pairs_lag0 = self._compute_pairs(returns, lag=0, period=period)
            self._db.save_correlations(pairs_lag0, period_days=period, lag_days=0)
            total_pairs += len(pairs_lag0)

        # Lag correlations — use 60-day window (best balance of data vs recency)
        returns_60 = self._db.get_returns_matrix(symbols, days=60)
        if not returns_60.empty and returns_60.shape[1] >= 2:
            for lag in range(1, MAX_LAG + 1):
                lag_pairs = self._compute_pairs(returns_60, lag=lag, period=60)
                self._db.save_correlations(lag_pairs, period_days=60, lag_days=lag)
                total_pairs += len(lag_pairs)

        elapsed = round(time.time() - t0, 1)
        logger.info(
            f"[CorrelationEngine] Done in {elapsed}s — "
            f"stored {total_pairs} significant pairs"
        )
        return {
            "symbols":           len(symbols),
            "periods":           PERIODS,
            "total_pairs_stored": total_pairs,
            "elapsed_seconds":   elapsed,
            "computed_at":       datetime.now(timezone.utc).isoformat(),
        }

    def _compute_pairs(
        self, returns: pd.DataFrame, lag: int, period: int
    ) -> list[dict]:
        """
        Compute Pearson r for every pair in `returns`.
        If lag > 0: shift column B backward by `lag` rows so A leads B.
        Only stores pairs with |r| >= MIN_ABS_R and p < MAX_P_VALUE.
        """
        cols    = list(returns.columns)
        results = []

        for sym_a, sym_b in combinations(cols, 2):
            a = returns[sym_a].dropna()
            b = returns[sym_b].dropna()

            if lag > 0:
                # A leads B: align so that a[t] pairs with b[t+lag]
                b = b.shift(-lag)

            # Align on common index
            aligned  = pd.concat([a, b], axis=1).dropna()
            n        = len(aligned)
            if n < MIN_OVERLAP:
                continue

            r, p = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

            if not np.isfinite(r):
                continue
            if abs(r) < MIN_ABS_R:
                continue
            if p > MAX_P_VALUE:
                continue

            results.append({
                "symbol_a":  sym_a,
                "symbol_b":  sym_b,
                "pearson_r": round(float(r), 4),
                "lag_days":  lag,
                "p_value":   round(float(p), 6),
            })

        return results

    def get_context_dict(self, universe: list[str], period_days: int = 60) -> dict:
        """
        Build the context["correlations"] dict consumed by strategies.

        Returns:
          {
            symbol: {
              "positive_peers":   [(peer, r), ...],   # r > 0.5, same-day
              "negative_peers":   [(peer, r), ...],   # r < -0.5, same-day
              "leading_for":      [(peer, lag, r), ...],  # this symbol leads `peer`
              "led_by":           [(leader, lag, r), ...], # `leader` leads this symbol
            }
          }
        """
        ctx: dict[str, dict] = {}

        for sym in universe:
            pos = self._db.get_top_correlations(sym, period_days, min_abs_r=0.5, lag_days=0)
            ctx[sym] = {
                "positive_peers": [(d["peer"], d["pearson_r"]) for d in pos if d["pearson_r"] > 0],
                "negative_peers": [(d["peer"], d["pearson_r"]) for d in pos if d["pearson_r"] < 0],
                "leading_for":    [],
                "led_by":         [],
            }

        # Leading indicator relationships
        for lag in range(1, MAX_LAG + 1):
            with self._db._engine.connect() as _:
                pass
            from sqlalchemy.orm import Session
            from core.market_db import CorrelationResult
            with Session(self._db._engine) as session:
                rows = (
                    session.query(CorrelationResult)
                           .filter(
                               CorrelationResult.period_days == period_days,
                               CorrelationResult.lag_days    == lag,
                               CorrelationResult.p_value     < MAX_P_VALUE,
                           )
                           .all()
                )
            for r in rows:
                if abs(r.pearson_r) < MIN_ABS_R:
                    continue
                # symbol_a leads symbol_b
                if r.symbol_a in ctx:
                    ctx[r.symbol_a]["leading_for"].append(
                        (r.symbol_b, lag, round(r.pearson_r, 3))
                    )
                if r.symbol_b in ctx:
                    ctx[r.symbol_b]["led_by"].append(
                        (r.symbol_a, lag, round(r.pearson_r, 3))
                    )

        return ctx

    def find_clusters(self, period_days: int = 60, threshold: float = 0.70) -> list[dict]:
        """
        Simple greedy clustering: group symbols where all intra-cluster
        correlations exceed `threshold`.  Returns list of cluster dicts.
        """
        symbols = self._db.get_available_symbols()
        if not symbols:
            return []

        pairs = self._db.get_all_correlation_pairs(
            period_days=period_days, min_abs_r=threshold, lag_days=0
        )
        # Build adjacency set
        adj: dict[str, set] = {s: set() for s in symbols}
        for p in pairs:
            if p["pearson_r"] >= threshold:
                adj[p["symbol_a"]].add(p["symbol_b"])
                adj[p["symbol_b"]].add(p["symbol_a"])

        visited: set[str] = set()
        clusters: list[dict] = []

        for seed in symbols:
            if seed in visited or not adj[seed]:
                continue
            cluster = {seed}
            queue   = list(adj[seed])
            while queue:
                node = queue.pop()
                if node in cluster:
                    continue
                # Only add if correlated with ALL current cluster members
                if all(node in adj[m] for m in cluster):
                    cluster.add(node)
                    queue.extend(adj[node] - cluster)
            if len(cluster) >= 3:
                # Compute average intra-cluster r
                cluster_pairs = [
                    p for p in pairs
                    if p["symbol_a"] in cluster and p["symbol_b"] in cluster
                ]
                avg_r = (
                    round(sum(p["pearson_r"] for p in cluster_pairs) / len(cluster_pairs), 3)
                    if cluster_pairs else threshold
                )
                clusters.append({
                    "symbols": sorted(cluster),
                    "size":    len(cluster),
                    "avg_r":   avg_r,
                })
                visited |= cluster

        return sorted(clusters, key=lambda c: c["size"], reverse=True)
