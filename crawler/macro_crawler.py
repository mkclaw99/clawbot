"""
Macro data crawler — FRED CSV API + CNN Fear & Greed Index + yfinance snapshots.
All data sources are completely free with no account or API key required.

FRED (Federal Reserve Economic Data):
  - Direct CSV download: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
  - No authentication, no rate limits beyond being polite
  - Key series used:
      ECBDFR    — ECB Deposit Facility Rate (European overnight rate)
      IRLTLT01DEM156N — German 10yr government bond yield
      VIXCLS    — CBOE VIX daily close (global market fear)
      CP0000EZ19M086NEST — Euro area HICP inflation (monthly)
      LRHUTTTTDEM156S — German unemployment rate (monthly)

yfinance snapshots (no key needed):
      ^GDAXI    — DAX 40 index level (German blue-chip benchmark)
      ^V2TX     — VSTOXX (European volatility index, ~= VIX for Euro Stoxx 50)

CNN Fear & Greed:
  - Endpoint: https://production.dataviz.cnn.io/index/fearandgreed/graphdata
  - Returns JSON with current score (0–100) + components
  - No auth, publicly accessible; used as a secondary global sentiment indicator

Output added to context:
    context["macro"] = {
        "ecb_rate":         float,     # ECB deposit facility rate (%)
        "de_yield_10y":     float,     # German 10yr Bund yield (%)
        "vix":              float,     # CBOE VIX (global fear gauge)
        "vstoxx":           float,     # VSTOXX (European fear gauge)
        "dax_level":        float,     # DAX 40 index level
        "cpi":              float,     # Euro area HICP index level
        "unemployment":     float,     # German unemployment rate (%)
        "regime":           str,       # "risk-on" | "risk-off" | "neutral"
        "fear_greed_score": float,     # 0–100 (0=extreme fear, 100=extreme greed)
        "fear_greed_label": str,       # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
        "source":           str,
        "timestamp":        str,
    }
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import requests
from loguru import logger

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 3600   # 1 hour — macro data doesn't change intraday

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS   = {
    "User-Agent": "clawbot-research-bot/1.0 (educational paper trading bot)",
    "Accept":     "text/csv,application/json,*/*",
}

FRED_SERIES = {
    "ecb_rate":    "ECBDFR",                  # ECB Deposit Facility Rate
    "de_yield_10y": "IRLTLT01DEM156N",        # German 10yr Bund yield
    "vix":         "VIXCLS",                  # CBOE VIX (global fear, kept as reference)
    "cpi":         "CP0000EZ19M086NEST",       # Euro area HICP inflation
    "unemployment": "LRHUTTTTDEM156S",         # German unemployment rate
}


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str, observations: int = 30) -> Optional[float]:
    """
    Fetch the most recent value of a FRED series via direct CSV download.
    Returns the latest non-null value, or None on failure.
    """
    url = f"{_FRED_BASE}?id={series_id}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        # Lines: DATE,VALUE  (header on first line)
        data_lines = [l for l in lines[1:] if not l.endswith(".")]
        # Pick last non-missing value
        for line in reversed(data_lines[-observations:]):
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
        return None
    except Exception as e:
        logger.debug(f"[FRED] Series {series_id} failed: {e}")
        return None


def fetch_fred_macro() -> dict[str, Optional[float]]:
    """Fetch all FRED macro indicators. Returns dict with latest values."""
    values: dict[str, Optional[float]] = {}
    for field, series_id in FRED_SERIES.items():
        values[field] = _fetch_fred_series(series_id)
        time.sleep(0.5)   # polite
    return values


# ---------------------------------------------------------------------------
# yfinance snapshots — DAX level + VSTOXX
# ---------------------------------------------------------------------------

def _fetch_yfinance_snapshot(ticker: str) -> Optional[float]:
    """Return the most recent closing price for a yfinance ticker."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        val = df["Close"].dropna()
        return float(val.iloc[-1].item() if hasattr(val.iloc[-1], "item") else val.iloc[-1])
    except Exception as e:
        logger.debug(f"[Macro] yfinance snapshot {ticker}: {e}")
        return None


def fetch_european_snapshots() -> dict[str, Optional[float]]:
    """Fetch DAX 40 level via yfinance. VSTOXX is not available on Yahoo Finance."""
    return {
        "dax_level": _fetch_yfinance_snapshot("^GDAXI"),
        "vstoxx":    None,   # ^V2TX / ^VSTOXX not carried by Yahoo Finance
    }


# ---------------------------------------------------------------------------
# CNN Fear & Greed
# ---------------------------------------------------------------------------

_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


def fetch_fear_greed() -> dict[str, object]:
    """
    Fetch the CNN Fear & Greed Index.
    Returns {'score': float, 'label': str} or defaults on failure.
    """
    try:
        resp = requests.get(_FNG_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Structure: {"fear_and_greed": {"score": 45.3, "rating": "Fear", ...}}
        fng  = data.get("fear_and_greed", {})
        score = float(fng.get("score", 50))
        label = str(fng.get("rating", "Neutral"))
        return {"score": round(score, 1), "label": label}

    except Exception as e:
        logger.debug(f"[FearGreed] Fetch failed: {e}")
        return {"score": 50.0, "label": "Neutral"}


# ---------------------------------------------------------------------------
# Regime inference
# ---------------------------------------------------------------------------

def _infer_regime(
    vix: Optional[float],
    vstoxx: Optional[float],
    de_yield_10y: Optional[float],
    fear_greed: float,
) -> str:
    """
    Classify the current macro regime as risk-on, risk-off, or neutral.
    Uses European indicators (VSTOXX, German Bund yield) as primary signals.
    VIX is kept as a secondary global risk gauge.
    """
    risk_off_signals = 0
    risk_on_signals  = 0

    # VSTOXX — European volatility (primary fear gauge for German stocks)
    vol = vstoxx if vstoxx is not None else vix
    if vol is not None:
        if vol > 35:
            risk_off_signals += 2   # high European fear
        elif vol > 25:
            risk_off_signals += 1
        else:
            risk_on_signals += 1    # low volatility = risk-on

    # German Bund yield — very low / negative = flight to safety = risk-off
    if de_yield_10y is not None:
        if de_yield_10y < 0:
            risk_off_signals += 2   # negative real rates = deep risk-off
        elif de_yield_10y > 3.0:
            risk_off_signals += 1   # rising rates = headwind for equities

    # CNN Fear & Greed (global sentiment, secondary weight)
    if fear_greed < 25:
        risk_off_signals += 2
    elif fear_greed < 40:
        risk_off_signals += 1
    elif fear_greed > 75:
        risk_on_signals += 1
    elif fear_greed > 60:
        risk_on_signals += 1

    if risk_off_signals >= 3:
        return "risk-off"
    if risk_on_signals >= 2:
        return "risk-on"
    return "neutral"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_macro_context() -> dict:
    """
    Fetch all macro data and return a unified context dict.
    Cached for 1 hour.
    """
    cache_key = "macro"
    if cache_key in _CACHE and time.time() - _CACHE[cache_key][0] < _CACHE_TTL:
        logger.debug("[Macro] Returning cached macro context")
        return _CACHE[cache_key][1]

    logger.info("[Macro] Fetching FRED + yfinance + CNN Fear & Greed data...")

    fred     = fetch_fred_macro()
    snap     = fetch_european_snapshots()
    fng      = fetch_fear_greed()

    vix_val      = fred.get("vix")
    vstoxx_val   = snap.get("vstoxx")
    dax_level    = snap.get("dax_level")
    ecb_rate     = fred.get("ecb_rate")
    de_yield_10y = fred.get("de_yield_10y")
    cpi_val      = fred.get("cpi")
    unemp_val    = fred.get("unemployment")

    regime = _infer_regime(vix_val, vstoxx_val, de_yield_10y, fng["score"])

    result = {
        "ecb_rate":         ecb_rate,
        "de_yield_10y":     de_yield_10y,
        "vix":              vix_val,
        "vstoxx":           vstoxx_val,
        "dax_level":        dax_level,
        "cpi":              cpi_val,
        "unemployment":     unemp_val,
        "fear_greed_score": fng["score"],
        "fear_greed_label": fng["label"],
        "regime":           regime,
        "source":           "fred+yfinance+cnn",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    _CACHE[cache_key] = (time.time(), result)

    logger.info(
        f"[Macro] VSTOXX={vstoxx_val}, VIX={vix_val}, DAX={dax_level}, "
        f"ECB_rate={ecb_rate}, DE_10y={de_yield_10y}, "
        f"fear_greed={fng['score']} ({fng['label']}), regime={regime}"
    )
    return result
