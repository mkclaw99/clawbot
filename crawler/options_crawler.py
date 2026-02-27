"""
Real options flow crawler — yfinance options chains.
No API key or account required.

Replaces the stub in news_crawler.py with genuine data:
  - Call/put volume and open interest from Yahoo Finance
  - Implied volatility distribution
  - Unusual activity detection: volume vs open interest ratio
  - Near-term focus: options expiring within MAX_DTE days

The options_flow strategy expects:
    {
        symbol: {
            "call_put_ratio":      float,   # call_volume / put_volume
            "premium_ratio":       float,   # today vol vs 30-day avg vol
            "oi_change":           float,   # OI change fraction
            "avg_days_to_expiry":  float,
            "block_trade_count":   int,     # proxy: contracts with vol > threshold
            "direction":           str,     # "bullish" | "bearish" | "neutral"
            "source":              str,
            "notable_trade":       str,
        }
    }
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import yfinance as yf
from loguru import logger

# Max days to expiry to consider (ignore LEAPS — likely hedges)
MAX_DTE   = 45
# Minimum total option volume to bother analyzing
MIN_VOL   = 50
# A single contract counts as a "block" if its volume exceeds this
BLOCK_VOL_THRESHOLD = 500
# Seconds between ticker fetches (polite)
FETCH_DELAY = 0.3

# ---------------------------------------------------------------------------
# Per-ticker cache so we don't hammer Yahoo on every refresh cycle
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 900  # 15 minutes


def _cached_flow(symbol: str) -> Optional[dict]:
    entry = _CACHE.get(symbol)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def fetch_options_flow_for_ticker(symbol: str) -> Optional[dict]:
    """
    Fetch real options chain data for a single ticker via yfinance.
    Returns a dict compatible with the options_flow strategy, or None on failure.
    """
    cached = _cached_flow(symbol)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options         # list of expiry date strings
        if not expirations:
            return None

        today = datetime.now(timezone.utc).date()
        total_call_vol = 0
        total_put_vol  = 0
        total_call_oi  = 0
        total_put_oi   = 0
        dte_list: list[float] = []
        block_count = 0
        max_iv = 0.0
        notable = ""

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < 0 or dte > MAX_DTE:
                continue

            try:
                chain = ticker.option_chain(exp_str)
            except Exception:
                continue

            calls = chain.calls
            puts  = chain.puts

            if calls.empty and puts.empty:
                continue

            c_vol = int(calls["volume"].fillna(0).sum())
            p_vol = int(puts["volume"].fillna(0).sum())
            c_oi  = int(calls["openInterest"].fillna(0).sum())
            p_oi  = int(puts["openInterest"].fillna(0).sum())

            total_call_vol += c_vol
            total_put_vol  += p_vol
            total_call_oi  += c_oi
            total_put_oi   += p_oi

            if c_vol + p_vol > 0:
                dte_list.append(dte)

            # Block trade proxy: single-contract rows with high volume
            for _, row in calls.iterrows():
                vol = row.get("volume", 0) or 0
                if vol >= BLOCK_VOL_THRESHOLD:
                    block_count += 1
                    strike = row.get("strike", 0)
                    iv     = row.get("impliedVolatility", 0) or 0
                    if iv > max_iv:
                        max_iv   = iv
                        notable  = f"Large call: {int(vol)} contracts @ ${strike:.0f} strike, IV={iv:.0%}"

            for _, row in puts.iterrows():
                vol = row.get("volume", 0) or 0
                if vol >= BLOCK_VOL_THRESHOLD:
                    block_count += 1

        total_vol = total_call_vol + total_put_vol
        if total_vol < MIN_VOL:
            return None

        cp_ratio = total_call_vol / max(1, total_put_vol)
        avg_dte  = sum(dte_list) / len(dte_list) if dte_list else 30

        # Premium ratio: vol / OI is a proxy for unusual activity
        # Higher vol-to-OI means lots of fresh contracts being opened today
        total_oi = total_call_oi + total_put_oi
        premium_ratio = (total_vol / max(1, total_oi)) * 10   # scale to ~1-10 range
        premium_ratio = min(10.0, max(0.1, round(premium_ratio, 2)))

        # OI change proxy: call OI vs put OI imbalance (normalized)
        oi_skew = (total_call_oi - total_put_oi) / max(1, total_call_oi + total_put_oi)
        oi_change = round(max(-1.0, min(1.0, oi_skew)), 3)

        if cp_ratio >= 2.0:
            direction = "bullish"
        elif cp_ratio <= 0.5:
            direction = "bearish"
        else:
            direction = "neutral"

        result = {
            "call_put_ratio":     round(cp_ratio, 2),
            "premium_ratio":      premium_ratio,
            "oi_change":          oi_change,
            "avg_days_to_expiry": round(avg_dte, 1),
            "block_trade_count":  block_count,
            "direction":          direction,
            "source":             "yfinance_options",
            "notable_trade":      notable,
            # Extra fields (not consumed by strategy but useful for dashboard)
            "call_volume":        total_call_vol,
            "put_volume":         total_put_vol,
            "call_oi":            total_call_oi,
            "put_oi":             total_put_oi,
            "max_iv":             round(max_iv, 3),
        }

        _CACHE[symbol] = (time.time(), result)
        return result

    except Exception as e:
        logger.debug(f"[Options] yfinance fetch failed for {symbol}: {e}")
        return None


def fetch_options_flow(universe: list[str], max_tickers: int = 20) -> dict[str, dict]:
    """
    Fetch real options data for up to max_tickers symbols in the universe.
    Prioritises symbols with highest recent volume (from bars if available).

    Returns dict compatible with context["options_flow"].
    """
    results: dict[str, dict] = {}
    checked = 0

    # Focus on a manageable subset — options chains are slow to fetch
    targets = universe[:max_tickers]

    for symbol in targets:
        if symbol in ("SPY", "QQQ", "IWM", "GLD"):
            # These have massive option chains — skip for performance
            continue

        flow = fetch_options_flow_for_ticker(symbol)
        if flow:
            results[symbol] = flow
            logger.debug(
                f"[Options] {symbol}: CP={flow['call_put_ratio']:.1f}, "
                f"dir={flow['direction']}, blocks={flow['block_trade_count']}"
            )
        checked += 1
        time.sleep(FETCH_DELAY)

    logger.info(f"[Options] Real options data: {len(results)}/{checked} tickers had activity")
    return results
