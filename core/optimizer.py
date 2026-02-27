"""
Self-Improvement Optimizer
===========================
Runs every 6 hours (after market hours) and continuously improves ClawBot by:

  1. PerformanceAnalyzer  — evaluates each strategy's rolling Sharpe, win rate, drawdown
  2. ParameterTuner       — adjusts strategy params within hard bounds based on performance
  3. WebResearcher        — searches web for new insights, trending tickers, market regime
  4. StrategyAllocator    — rebalances capital weights based on Sharpe ratio

HARD RULES (enforced in code, cannot be overridden by the optimizer itself):
  • safety_constants.py is NEVER imported or touched here
  • Each param has a (min, max, step) bound that cannot be exceeded
  • Max 1 param change per strategy per cycle
  • 24-hour cooling period between changes to the same param
  • No tuning if strategy has <MIN_TRADES_FOR_TUNING trades
  • Optimizer only runs outside market hours
  • All changes are written to the append-only audit log
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from config.settings import (
    DATABASE_URL, MIN_TRADES_FOR_TUNING, OPTIMIZER_COOLING_HOURS,
    OPTIMIZER_MAX_STEP_PCT,
)
from core.audit import audit_param_change, audit_system
from core.llm_client import LLMClient
from crawler.web_searcher import WebResearcher, ResearchFinding

# ---------------------------------------------------------------------------
# Parameter bounds — the ONLY things the optimizer is allowed to change
# Format: { strategy_name: { param_name: (min_val, max_val, step) } }
# The optimizer CANNOT touch anything not in this dict.
# ---------------------------------------------------------------------------

PARAM_BOUNDS: dict[str, dict[str, tuple]] = {
    "meme_momentum": {
        "MIN_MENTION_SPIKE": (1.5,  4.0,  0.25),
        "MIN_SENTIMENT":     (0.05, 0.40, 0.05),
    },
    "technical_trend": {
        "RSI_OVERSOLD":   (25.0, 45.0, 2.0),
        "RSI_OVERBOUGHT": (60.0, 80.0, 2.0),
    },
    "mean_reversion": {
        "ZSCORE_ENTRY": (-3.0, -1.5, 0.25),
        "BB_PERIOD":    (15.0, 30.0, 1.0),
    },
    "options_flow": {
        "MIN_PREMIUM_RATIO":  (2.0, 5.0, 0.25),
        "MIN_CALL_PUT_RATIO": (1.5, 3.5, 0.25),
    },
    "macro_news": {
        "BUY_SENTIMENT_THRESHOLD":  (0.40, 0.75, 0.05),
        "SELL_SENTIMENT_THRESHOLD": (-0.60, -0.25, 0.05),
    },
}

# Allocation weight bounds per strategy (must sum to 1.0)
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.40
WEIGHT_MAX_CHANGE_PER_CYCLE = 0.05

# Persistence path for optimizer state (not safety-critical — just bookkeeping)
_STATE_PATH = Path(__file__).parent.parent / "logs" / "optimizer_state.json"
_RESEARCH_LOG_PATH = Path(__file__).parent.parent / "logs" / "research_log.jsonl"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StrategyPerf:
    strategy: str
    total_trades: int
    win_rate: float           # 0–100
    recent_win_rate: float    # last 10 trades, 0–100
    prior_win_rate: float     # 10 trades before that, 0–100
    sharpe: float             # rolling Sharpe (last 30 trades), may be nan
    total_pnl: float
    is_improving: bool        # recent_win_rate > prior_win_rate


@dataclass
class OptimizerState:
    # When each param was last changed: {strategy: {param: iso_timestamp}}
    last_changed: dict[str, dict[str, str]] = field(default_factory=dict)
    # Win rate at time of last change (for reversion detection): {strategy: {param: float}}
    win_rate_at_change: dict[str, dict[str, float]] = field(default_factory=dict)
    # Value at time of last change (for reversion): {strategy: {param: float}}
    value_at_change: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Current allocation weights: {strategy: float}
    weights: dict[str, float] = field(default_factory=dict)
    # Run history: list of {timestamp, changes, regime, findings_count}
    run_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerState":
        s = cls()
        s.last_changed       = d.get("last_changed", {})
        s.win_rate_at_change = d.get("win_rate_at_change", {})
        s.value_at_change    = d.get("value_at_change", {})
        s.weights            = d.get("weights", {})
        s.run_history        = d.get("run_history", [])
        return s


# ---------------------------------------------------------------------------
# PerformanceAnalyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """Reads trade history from SQLite and computes per-strategy metrics."""

    def __init__(self) -> None:
        self._engine = create_engine(DATABASE_URL, echo=False)

    def analyze_all(self, strategies: list[str]) -> dict[str, StrategyPerf]:
        """Returns a StrategyPerf for each strategy name."""
        results = {}
        for strategy in strategies:
            results[strategy] = self._analyze_strategy(strategy)
        return results

    def _analyze_strategy(self, strategy: str) -> StrategyPerf:
        with Session(self._engine) as session:
            rows = session.execute(
                text(
                    "SELECT action, pnl, timestamp FROM trades "
                    "WHERE strategy = :s ORDER BY timestamp ASC"
                ),
                {"s": strategy},
            ).fetchall()

        trades = [dict(r._mapping) for r in rows]
        sells  = [t for t in trades if t["action"] == "SELL"]
        n      = len(sells)

        if n == 0:
            return StrategyPerf(strategy, 0, 0.0, 0.0, 0.0, float("nan"), 0.0, False)

        pnls     = [t["pnl"] for t in sells]
        wins     = [p for p in pnls if p > 0]
        win_rate = len(wins) / n * 100

        # Recent vs prior win rate (for trend detection)
        recent_trades = pnls[-10:] if n >= 10 else pnls
        prior_trades  = pnls[-20:-10] if n >= 20 else pnls[:max(1, n // 2)]
        recent_wr = len([p for p in recent_trades if p > 0]) / max(1, len(recent_trades)) * 100
        prior_wr  = len([p for p in prior_trades  if p > 0]) / max(1, len(prior_trades))  * 100

        # Rolling Sharpe on last 30 trades
        sharpe = self._sharpe(pnls[-30:])

        return StrategyPerf(
            strategy=strategy,
            total_trades=n,
            win_rate=round(win_rate, 1),
            recent_win_rate=round(recent_wr, 1),
            prior_win_rate=round(prior_wr, 1),
            sharpe=round(sharpe, 3),
            total_pnl=round(sum(pnls), 2),
            is_improving=recent_wr > prior_wr,
        )

    @staticmethod
    def _sharpe(pnls: list[float], risk_free: float = 0.0) -> float:
        """Compute Sharpe ratio from a list of trade P&Ls."""
        if len(pnls) < 3:
            return float("nan")
        n    = len(pnls)
        mean = sum(pnls) / n
        var  = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std  = math.sqrt(var)
        if std == 0:
            return float("nan")
        return (mean - risk_free) / std


# ---------------------------------------------------------------------------
# ParameterTuner
# ---------------------------------------------------------------------------

class ParameterTuner:
    """
    Adjusts strategy class attributes within PARAM_BOUNDS.
    Never touches safety_constants.py.

    When an LLMClient is supplied the tuner asks the LLM which parameter to
    change and in which direction.  The LLM recommendation is advisory: the
    final value is always computed by _compute_new_val() which enforces hard
    bounds, step limits, and the 10% OPTIMIZER_MAX_STEP_PCT cap.
    """

    def __init__(self, strategies: list, state: OptimizerState,
                 llm: LLMClient | None = None) -> None:
        self._strategy_map = {s.name: s for s in strategies}
        self._state        = state
        self._llm          = llm

    def tune(self, perfs: dict[str, StrategyPerf]) -> list[dict]:
        """
        Given performance data, decide which params to adjust.
        Returns a list of change records for logging.
        """
        changes = []
        for strategy_name, perf in perfs.items():
            if perf.total_trades < MIN_TRADES_FOR_TUNING:
                logger.debug(
                    f"[Tuner] {strategy_name}: only {perf.total_trades} trades "
                    f"(need {MIN_TRADES_FOR_TUNING}) — skipping"
                )
                continue

            strategy = self._strategy_map.get(strategy_name)
            if not strategy:
                continue

            change = self._tune_one(strategy, perf)
            if change:
                changes.append(change)

        return changes

    def _tune_one(self, strategy, perf: StrategyPerf) -> dict | None:
        """Tune at most ONE parameter for this strategy this cycle."""
        bounds_for_strategy = PARAM_BOUNDS.get(strategy.name, {})
        if not bounds_for_strategy:
            return None

        # First: check if any previous change performed badly → revert it
        reversion = self._check_revert(strategy, perf, bounds_for_strategy)
        if reversion:
            return reversion

        # Ask LLM for a recommendation; fall back to simple heuristic
        llm_rec   = self._llm_recommend_param(strategy, perf, bounds_for_strategy)
        direction = "tighten" if perf.is_improving else "loosen"

        # Build ordered param list — LLM-preferred param first
        params_ordered = list(bounds_for_strategy.items())
        if llm_rec and llm_rec["direction"] != "no_change":
            rec_param = llm_rec["param"]
            params_ordered = (
                [(rec_param, bounds_for_strategy[rec_param])] +
                [(p, b) for p, b in params_ordered if p != rec_param]
            )
            logger.info(
                f"[Tuner/LLM] {strategy.name}: recommends {rec_param} → "
                f"{llm_rec['direction']} | {llm_rec.get('reasoning', '')}"
            )

        for idx, (param, (lo, hi, step)) in enumerate(params_ordered):
            if not self._is_cooled(strategy.name, param):
                continue

            current = getattr(strategy, param, None)
            if current is None:
                continue

            # Use LLM direction for the recommended param; heuristic for the rest
            if llm_rec and llm_rec["direction"] != "no_change" and idx == 0:
                param_direction = "tighten" if llm_rec["direction"] == "increase" else "loosen"
            else:
                param_direction = direction

            new_val = self._compute_new_val(current, lo, hi, step, param_direction, perf)
            if new_val is None or new_val == current:
                continue

            old_val = current
            setattr(strategy, param, new_val)

            now_iso = datetime.now(timezone.utc).isoformat()
            self._state.last_changed.setdefault(strategy.name, {})[param]       = now_iso
            self._state.win_rate_at_change.setdefault(strategy.name, {})[param] = perf.win_rate
            self._state.value_at_change.setdefault(strategy.name, {})[param]    = old_val

            llm_note = (f" [LLM: {llm_rec['reasoning']}]"
                        if llm_rec and idx == 0 and llm_rec.get("reasoning") else "")
            reason = (
                f"win_rate={perf.win_rate:.1f}% (recent={perf.recent_win_rate:.1f}%, "
                f"prior={perf.prior_win_rate:.1f}%), sharpe={perf.sharpe:.3f}, "
                f"{'improving' if perf.is_improving else 'declining'} → {param_direction}"
                f"{llm_note}"
            )
            audit_param_change(strategy.name, param, old_val, new_val, reason)
            logger.info(f"[Tuner] {strategy.name}.{param}: {old_val} → {new_val} ({reason})")

            return {
                "strategy":  strategy.name,
                "param":     param,
                "old_val":   old_val,
                "new_val":   new_val,
                "reason":    reason,
                "timestamp": now_iso,
            }

        return None

    # ------------------------------------------------------------------
    # LLM advisory
    # ------------------------------------------------------------------

    def _llm_recommend_param(
        self, strategy, perf: StrategyPerf, bounds: dict
    ) -> dict | None:
        """
        Ask the LLM which parameter to tune and in which direction.
        Returns {"param": str, "direction": "increase"|"decrease"|"no_change",
                 "reasoning": str}  or None if LLM is unavailable / reply invalid.

        The LLM is ADVISORY ONLY — bounds and step limits are enforced afterwards.
        """
        if not self._llm:
            return None

        current_params = {
            param: getattr(strategy, param, None)
            for param in bounds
        }
        param_info = {
            p: {"current": current_params.get(p), "min": b[0], "max": b[1], "step": b[2]}
            for p, b in bounds.items()
        }

        prompt = (
            f"You are a quantitative trading parameter advisor for a German equity strategy.\n\n"
            f"Strategy: {strategy.name}\n"
            f"Performance metrics:\n"
            f"  total_trades={perf.total_trades}, win_rate={perf.win_rate:.1f}%, "
            f"  recent_win_rate={perf.recent_win_rate:.1f}%, "
            f"  prior_win_rate={perf.prior_win_rate:.1f}%, "
            f"  sharpe={perf.sharpe:.3f}, total_pnl={perf.total_pnl:.2f}, "
            f"  trend={'improving' if perf.is_improving else 'declining'}\n\n"
            f"Tunable parameters (with allowed bounds):\n"
            f"{json.dumps(param_info, indent=2)}\n\n"
            f"Which single parameter change would most improve this strategy's risk-adjusted "
            f"return? Respond with JSON only — no explanation outside the JSON:\n"
            f'{{"param": "<param_name>", "direction": "increase" | "decrease" | "no_change", '
            f'"reasoning": "<one sentence max 15 words>"}}'
        )

        result = self._llm.complete_json(
            [{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        if (
            result
            and result.get("param") in bounds
            and result.get("direction") in ("increase", "decrease", "no_change")
        ):
            return result

        logger.debug(f"[Tuner/LLM] Invalid or no response for {strategy.name}: {result}")
        return None

    def _check_revert(self, strategy, perf: StrategyPerf, bounds: dict) -> dict | None:
        """If win rate dropped >10pp since last change, revert that param."""
        last_wr  = self._state.win_rate_at_change.get(strategy.name, {})
        old_vals = self._state.value_at_change.get(strategy.name, {})

        for param, wr_at_change in last_wr.items():
            if perf.win_rate < wr_at_change - 10.0:   # win rate dropped 10pp+
                if param not in old_vals:
                    continue
                original = old_vals[param]
                current  = getattr(strategy, param, None)
                if current == original:
                    continue
                lo, hi, step = bounds.get(param, (None, None, None))
                if lo is None:
                    continue
                # Clamp reversion to bounds
                reverted = max(lo, min(hi, original))
                setattr(strategy, param, reverted)

                now_iso = datetime.now(timezone.utc).isoformat()
                reason  = (
                    f"REVERT: win_rate dropped from {wr_at_change:.1f}% → {perf.win_rate:.1f}% "
                    f"after previous change"
                )
                audit_param_change(strategy.name, param, current, reverted, reason)
                logger.warning(f"[Tuner] REVERT {strategy.name}.{param}: {current} → {reverted}")

                # Clear the change record so we don't revert again next cycle
                del self._state.win_rate_at_change[strategy.name][param]
                del self._state.value_at_change[strategy.name][param]

                return {
                    "strategy": strategy.name,
                    "param": param,
                    "old_val": current,
                    "new_val": reverted,
                    "reason": reason,
                    "timestamp": now_iso,
                    "type": "revert",
                }

        return None

    def _compute_new_val(
        self, current: float, lo: float, hi: float, step: float,
        direction: str, perf: StrategyPerf,
    ) -> float | None:
        """
        Compute a new parameter value.
        'tighten' = more selective (for high-threshold params, increase; for entry thresholds, tighten)
        'loosen'  = less selective (more trades)
        """
        # Simple hill-climbing: move one step in the direction that should help
        if direction == "tighten":
            # More selective thresholds → higher confidence per trade
            new_val = current + step
        else:
            # More permissive → more trades, wider net
            new_val = current - step

        # Clamp to bounds
        new_val = max(lo, min(hi, new_val))
        new_val = round(new_val, 4)

        # Don't exceed max step as a fraction of current value
        if current != 0:
            actual_change = abs(new_val - current) / abs(current)
            if actual_change > OPTIMIZER_MAX_STEP_PCT:
                # Scale back to max step
                sign    = 1 if new_val > current else -1
                new_val = round(current + sign * abs(current) * OPTIMIZER_MAX_STEP_PCT, 4)
                new_val = max(lo, min(hi, new_val))

        return new_val if new_val != current else None

    def _is_cooled(self, strategy_name: str, param: str) -> bool:
        """True if the cooling period has passed since this param was last changed."""
        changed_at_str = self._state.last_changed.get(strategy_name, {}).get(param)
        if not changed_at_str:
            return True  # never changed → always available
        changed_at = datetime.fromisoformat(changed_at_str)
        if changed_at.tzinfo is None:
            changed_at = changed_at.replace(tzinfo=timezone.utc)
        cooldown_until = changed_at + timedelta(hours=OPTIMIZER_COOLING_HOURS)
        return datetime.now(timezone.utc) >= cooldown_until


# ---------------------------------------------------------------------------
# StrategyAllocator
# ---------------------------------------------------------------------------

class StrategyAllocator:
    """
    Rebalances capital weights across strategies based on rolling Sharpe ratio.
    Better Sharpe → more capital, up to WEIGHT_MAX.
    All weights sum to 1.0 and stay within [WEIGHT_MIN, WEIGHT_MAX].
    """

    def __init__(self, strategy_names: list[str], state: OptimizerState) -> None:
        self._names = strategy_names
        self._state = state
        # Initialize equal weights if not yet set
        if not self._state.weights:
            equal = round(1.0 / len(strategy_names), 4)
            self._state.weights = {n: equal for n in strategy_names}

    def rebalance(self, perfs: dict[str, StrategyPerf]) -> dict[str, float]:
        """
        Rebalance weights based on Sharpe ratios.
        Returns the new weight dict.
        """
        current = dict(self._state.weights)

        # Compute target weights from Sharpe ratios
        sharpes = {}
        for name in self._names:
            perf = perfs.get(name)
            if perf and not math.isnan(perf.sharpe) and perf.total_trades >= MIN_TRADES_FOR_TUNING:
                sharpes[name] = max(0.01, perf.sharpe)
            else:
                # Not enough data — give it equal weight
                sharpes[name] = 1.0

        total_sharpe = sum(sharpes.values())
        raw_targets  = {n: s / total_sharpe for n, s in sharpes.items()}

        equal = 1.0 / len(self._names)

        # Per-strategy feasible bounds: intersection of cycle cap and absolute bounds
        per_bounds = {
            n: (
                max(WEIGHT_MIN, current.get(n, equal) - WEIGHT_MAX_CHANGE_PER_CYCLE),
                min(WEIGHT_MAX, current.get(n, equal) + WEIGHT_MAX_CHANGE_PER_CYCLE),
            )
            for n in self._names
        }

        # Blend current toward target, capped
        new_weights: dict[str, float] = {}
        for name in self._names:
            old   = current.get(name, equal)
            delta = raw_targets[name] - old
            delta = max(-WEIGHT_MAX_CHANGE_PER_CYCLE, min(WEIGHT_MAX_CHANGE_PER_CYCLE, delta))
            lo, hi = per_bounds[name]
            new_weights[name] = max(lo, min(hi, old + delta))

        # Iterative redistribution: make weights sum to 1.0 while respecting per_bounds
        # (up to 20 passes; converges in 2-3 for typical cases)
        for _ in range(20):
            residual = 1.0 - sum(new_weights.values())
            if abs(residual) < 1e-7:
                break
            available = [
                n for n in self._names
                if (residual > 0 and new_weights[n] < per_bounds[n][1] - 1e-8)
                or (residual < 0 and new_weights[n] > per_bounds[n][0] + 1e-8)
            ]
            if not available:
                break
            share = residual / len(available)
            for n in available:
                lo, hi = per_bounds[n]
                new_weights[n] = max(lo, min(hi, new_weights[n] + share))

        new_weights = {n: round(w, 4) for n, w in new_weights.items()}

        # Log changes
        for name, w in new_weights.items():
            old = current.get(name, 0)
            if abs(w - old) >= 0.005:
                reason = f"Sharpe={sharpes[name]:.3f}, rebalance cycle"
                audit_param_change(name, "allocation_weight", old, w, reason)
                logger.info(f"[Allocator] {name}: weight {old:.3f} → {w:.3f}")

        self._state.weights = new_weights
        return new_weights


# ---------------------------------------------------------------------------
# Main Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """
    Top-level self-improvement orchestrator.
    Called by the engine every 6 hours.
    """

    def __init__(self, strategies: list, aggregator) -> None:
        self._strategies  = strategies
        self._aggregator  = aggregator
        self._state       = self._load_state()
        self._analyzer    = PerformanceAnalyzer()

        # LLM client — used by tuner and researcher; gracefully absent if offline
        self._llm = LLMClient()
        if self._llm.available():
            logger.info(f"[Optimizer] LLM online: {self._llm._model} @ {self._llm._url}")
        else:
            logger.warning("[Optimizer] LLM offline — falling back to heuristics")
            self._llm = None

        self._tuner      = ParameterTuner(strategies, self._state, llm=self._llm)
        self._allocator  = StrategyAllocator([s.name for s in strategies], self._state)
        self._researcher = WebResearcher(llm=self._llm)
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RESEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[Optimizer] Initialized.")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run one full self-improvement cycle.
        Returns a summary dict (for dashboard + audit log).
        """
        now = datetime.now(timezone.utc)
        logger.info(f"[Optimizer] === Self-improvement cycle starting at {now.isoformat()} ===")
        audit_system("OPTIMIZER_START", "Self-improvement cycle started")

        # 1. Analyze performance
        strategy_names = [s.name for s in self._strategies]
        perfs = self._analyzer.analyze_all(strategy_names)
        for name, p in perfs.items():
            logger.info(
                f"[Optimizer] {name}: trades={p.total_trades}, "
                f"win_rate={p.win_rate:.1f}%, recent_wr={p.recent_win_rate:.1f}%, "
                f"sharpe={p.sharpe:.3f}, improving={p.is_improving}"
            )

        # 2. Tune parameters
        param_changes = self._tuner.tune(perfs)

        # 3. Rebalance strategy weights
        new_weights = self._allocator.rebalance(perfs)

        # 4. Web research (runs last — non-critical path)
        findings    = self._researcher.run_research()
        regime      = self._researcher.extract_market_regime(findings)
        new_tickers = self._researcher.extract_trending_tickers(findings)
        insights    = self._researcher.synthesize_insights(findings)
        self._update_universe(new_tickers)
        self._save_research(findings, insights)

        # 5. Persist state
        self._state.run_history.append({
            "timestamp":      now.isoformat(),
            "param_changes":  len(param_changes),
            "weight_changes": sum(1 for n in strategy_names
                                  if abs(new_weights.get(n, 0) - self._state.weights.get(n, 0)) > 0.001),
            "market_regime":  regime,
            "findings":       len(findings),
            "new_tickers":    new_tickers[:5],
        })
        # Keep last 100 run records
        self._state.run_history = self._state.run_history[-100:]
        self._save_state()

        summary = {
            "timestamp":      now.isoformat(),
            "param_changes":  param_changes,
            "weights":        new_weights,
            "market_regime":  regime,
            "findings_count": len(findings),
            "new_tickers":    new_tickers[:10],
            "insights":       insights,
            "perfs":          {k: {
                "total_trades": v.total_trades,
                "win_rate": v.win_rate,
                "sharpe": v.sharpe,
                "is_improving": v.is_improving,
            } for k, v in perfs.items()},
        }

        audit_system("OPTIMIZER_DONE", "Self-improvement cycle complete",
                     param_changes=len(param_changes),
                     regime=regime,
                     findings=len(findings))

        logger.info(
            f"[Optimizer] Cycle complete: {len(param_changes)} param changes, "
            f"regime={regime}, {len(findings)} web findings, "
            f"{len(new_tickers)} new tickers discovered"
        )
        return summary

    # ------------------------------------------------------------------
    # Universe management
    # ------------------------------------------------------------------

    def _update_universe(self, new_tickers: list[str]) -> None:
        """
        Add genuinely new, valid tickers to the aggregator's universe.
        Enforces a cap of 60 total tickers.
        """
        if not new_tickers:
            return

        current = set(self._aggregator.universe)
        added   = []
        for ticker in new_tickers:
            if ticker in current:
                continue
            if len(current) >= 60:
                break
            # Basic sanity check: 1–5 uppercase letters
            if not ticker.isalpha() or not ticker.isupper() or len(ticker) > 5:
                continue
            current.add(ticker)
            added.append(ticker)

        if added:
            self._aggregator.universe = list(current)
            audit_system("UNIVERSE_UPDATE", f"Added {len(added)} tickers from web research",
                         added=added)
            logger.info(f"[Optimizer] Universe expanded with: {added}")

    # ------------------------------------------------------------------
    # Research log
    # ------------------------------------------------------------------

    def _save_research(self, findings: list[ResearchFinding], insights: str = "") -> None:
        """Append research findings (and optional LLM insights) to the research log (JSONL)."""
        with open(_RESEARCH_LOG_PATH, "a", encoding="utf-8") as fh:
            for f in findings:
                fh.write(json.dumps({
                    "timestamp":       f.timestamp,
                    "query":           f.query,
                    "ticker":          f.ticker,
                    "snippet":         f.snippet,
                    "relevance_score": f.relevance_score,
                    "category":        f.category,
                    "source_url":      f.source_url,
                }) + "\n")
            if insights:
                fh.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query":     "_llm_synthesis",
                    "ticker":    None,
                    "snippet":   insights,
                    "relevance_score": 1.0,
                    "category":  "synthesis",
                    "source_url": "",
                }) + "\n")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> OptimizerState:
        if _STATE_PATH.exists():
            try:
                return OptimizerState.from_dict(json.loads(_STATE_PATH.read_text()))
            except Exception as e:
                logger.warning(f"[Optimizer] Failed to load state: {e} — starting fresh")
        return OptimizerState()

    def _save_state(self) -> None:
        _STATE_PATH.write_text(json.dumps(self._state.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "weights":         self._state.weights,
            "run_history":     self._state.run_history[-10:],
            "last_run":        self._state.run_history[-1]["timestamp"] if self._state.run_history else None,
            "param_bounds":    {s: list(p.keys()) for s, p in PARAM_BOUNDS.items()},
            "cooling_periods": self._state.last_changed,
        }

    def get_current_params(self) -> dict[str, dict[str, float]]:
        """Current live values of all tunable parameters."""
        result = {}
        for s in self._strategies:
            result[s.name] = {}
            for param in PARAM_BOUNDS.get(s.name, {}):
                result[s.name][param] = getattr(s, param, None)
        return result

    def get_research_log(self, limit: int = 50) -> list[dict]:
        """Read recent research findings from log."""
        if not _RESEARCH_LOG_PATH.exists():
            return []
        lines = _RESEARCH_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in reversed(lines[-limit:]):
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        return records[:limit]
