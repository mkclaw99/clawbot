"""
Google Trends crawler — pytrends (no API key, no account required).

Measures real-time search interest for stock tickers as a leading
indicator for retail attention and potential meme momentum.

Output fed into context["google_trends"]:
    {
        symbol: {
            "interest_score":  float,   # 0–100, current relative search interest
            "spike_ratio":     float,   # current vs 3-month avg interest
            "trend":           str,     # "rising" | "falling" | "flat"
            "peak_week":       str,     # ISO date of highest recent interest
        }
    }

Strategy consumption:
  - meme_momentum: boosts buy score when Google Trends spike_ratio > 2.0
  - Any strategy: if interest_score > 80, increase confidence slightly
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 3600   # 1 hour (Google Trends doesn't change that fast)
_BATCH_SIZE = 5     # pytrends supports up to 5 keywords per request


def _get_pytrends():
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise RuntimeError("pytrends not installed. Run: pip install pytrends")

    # urllib3 v2 renamed Retry's `method_whitelist` → `allowed_methods`.
    # pytrends passes the old name internally regardless of what we pass here,
    # so patch it away at the source before constructing TrendReq.
    try:
        import urllib3.util.retry as _r
        _orig_init = _r.Retry.__init__
        def _patched_init(self, *a, **kw):
            kw.pop("method_whitelist", None)   # strip the deprecated kwarg
            _orig_init(self, *a, **kw)
        _r.Retry.__init__ = _patched_init
    except Exception:
        pass

    # tz=60 = CET (UTC+1, Germany); hl="de-DE" returns German-context results
    return TrendReq(hl="de-DE", tz=60, timeout=(10, 25))


def fetch_trends_for_batch(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch Google Trends interest for a batch of ≤5 symbols.
    Returns partial results (may be empty if request fails).
    """
    if not symbols:
        return {}

    # Check cache first
    results = {}
    uncached = []
    for sym in symbols:
        if sym in _CACHE and time.time() - _CACHE[sym][0] < _CACHE_TTL:
            results[sym] = _CACHE[sym][1]
        else:
            uncached.append(sym)

    if not uncached:
        return results

    try:
        pt = _get_pytrends()
        # Strip .DE suffix — Google Trends uses company names/short tickers
        # Append " Aktie" (German for "stock") to distinguish from non-stock queries
        kw_map = {
            sym: sym.replace(".DE", "") if sym.endswith(".DE") else f"{sym} stock"
            for sym in uncached[:_BATCH_SIZE]
        }
        kw_list = list(kw_map.values())
        pt.build_payload(kw_list, cat=0, timeframe="today 3-m", geo="DE")
        df = pt.interest_over_time()

        if df.empty:
            return results

        for sym in uncached[:_BATCH_SIZE]:
            col = kw_map[sym]
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            # Current interest = last available week's value
            current = float(series.iloc[-1])
            avg_3m  = float(series.mean())
            spike   = current / max(1.0, avg_3m)

            # Trend: compare last 2 weeks vs 2 weeks before that
            if len(series) >= 4:
                recent = series.iloc[-2:].mean()
                prior  = series.iloc[-4:-2].mean()
                if recent > prior * 1.15:
                    trend = "rising"
                elif recent < prior * 0.85:
                    trend = "falling"
                else:
                    trend = "flat"
            else:
                trend = "flat"

            peak_idx = series.idxmax()
            peak_week = peak_idx.strftime("%Y-%m-%d") if hasattr(peak_idx, "strftime") else str(peak_idx)

            entry = {
                "interest_score": round(current, 1),
                "spike_ratio":    round(spike, 2),
                "trend":          trend,
                "peak_week":      peak_week,
            }
            results[sym]       = entry
            _CACHE[sym]        = (time.time(), entry)

    except Exception as e:
        logger.warning(f"[GoogleTrends] Batch {symbols}: {e}")

    return results


def fetch_google_trends(universe: list[str], meme_first: bool = True) -> dict[str, dict]:
    """
    Fetch Google Trends for symbols in the universe.
    Prioritises meme universe if meme_first=True.

    Returns context["google_trends"] dict.
    """
    from crawler.signal_aggregator import MEME_UNIVERSE

    # Order: meme stocks first, then others
    if meme_first:
        ordered = [s for s in universe if s in MEME_UNIVERSE]
        ordered += [s for s in universe if s not in MEME_UNIVERSE]
    else:
        ordered = list(universe)

    # Focus on a reasonable subset to avoid rate limits
    targets = ordered[:25]

    all_results: dict[str, dict] = {}

    for i in range(0, len(targets), _BATCH_SIZE):
        batch = targets[i : i + _BATCH_SIZE]
        batch_results = fetch_trends_for_batch(batch)
        all_results.update(batch_results)
        if i + _BATCH_SIZE < len(targets):
            time.sleep(1.5)   # polite delay between batches

    logger.info(
        f"[GoogleTrends] {len(all_results)}/{len(targets)} tickers with trend data. "
        f"Rising: {sum(1 for v in all_results.values() if v['trend'] == 'rising')}"
    )
    return all_results
