"""
Price Historian — downloads historical OHLCV data into market.db.

Fetches up to 2 years of daily bars for the German DAX 40 + MDAX universe.
All tickers use the XETRA .DE suffix recognised by yfinance.

Designed to run nightly (after market close) via `clawbot update-prices`,
and also as a one-time bootstrap with `clawbot update-prices --full`.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from core.market_db import MarketDatabase
from config.universe import (
    DAX40, MDAX_SELECTED, GERMAN_ETFS, FULL_UNIVERSE,
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_batch(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for a batch of symbols via yfinance.
    Returns {symbol: DataFrame} with columns Date, Open, High, Low, Close, Volume.
    """
    result: dict[str, pd.DataFrame] = {}
    if not symbols:
        return result

    tickers_str = " ".join(symbols)
    try:
        raw = yf.download(
            tickers_str,
            period=period,
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        for sym in symbols:
            try:
                df = raw[sym] if len(symbols) > 1 else raw
                if df is None or df.empty:
                    continue
                df = df.dropna(subset=["Close"]).copy()
                df.index = pd.to_datetime(df.index)
                result[sym] = df
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[PriceHistorian] Batch download failed: {e}")

    return result


def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily_return and log_return columns to an OHLCV DataFrame."""
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"]   = np.log(df["Close"] / df["Close"].shift(1))
    return df


def _to_db_records(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert a yfinance DataFrame to the schema expected by MarketDatabase.upsert_bars."""
    df = _compute_returns(df)
    records = pd.DataFrame({
        "date":         df.index.date,
        "open":         df["Open"].values,
        "high":         df["High"].values,
        "low":          df["Low"].values,
        "close":        df["Close"].values,
        "volume":       df["Volume"].values,
        "daily_return": df["daily_return"].values,
        "log_return":   df["log_return"].values,
    })
    return records.dropna(subset=["close"])


# ---------------------------------------------------------------------------
# PriceHistorian
# ---------------------------------------------------------------------------

_BATCH_SIZE = 20   # yfinance handles ~20 tickers well in one call


class PriceHistorian:
    """
    Downloads and persists daily OHLCV bars for the full universe.

    Usage:
        historian = PriceHistorian()
        historian.update(period="2y")   # full bootstrap or nightly refresh
    """

    def __init__(self, db: MarketDatabase | None = None) -> None:
        self._db = db or MarketDatabase()

    def update(
        self,
        universe: list[str] | None = None,
        period: str = "1y",
    ) -> dict:
        """
        Download `period` of daily bars for the universe and upsert into market.db.
        period: yfinance period string — "1y", "2y", "6mo", etc.

        Returns a summary dict.
        """
        symbols = universe or FULL_UNIVERSE
        logger.info(
            f"[PriceHistorian] Updating {len(symbols)} symbols | period={period}"
        )

        total_inserted = 0
        failed: list[str] = []
        t0 = time.time()

        for i in range(0, len(symbols), _BATCH_SIZE):
            batch = symbols[i : i + _BATCH_SIZE]
            raw   = _download_batch(batch, period=period)

            for sym in batch:
                if sym not in raw:
                    failed.append(sym)
                    continue
                try:
                    records = _to_db_records(sym, raw[sym])
                    n = self._db.upsert_bars(sym, records)
                    total_inserted += n
                except Exception as e:
                    logger.warning(f"[PriceHistorian] {sym}: {e}")
                    failed.append(sym)

            # polite delay between batches
            if i + _BATCH_SIZE < len(symbols):
                time.sleep(0.5)

        elapsed = round(time.time() - t0, 1)
        stats   = self._db.get_stats()
        logger.info(
            f"[PriceHistorian] Done in {elapsed}s — "
            f"inserted {total_inserted} new bars | failed={len(failed)}"
        )
        return {
            "symbols_attempted": len(symbols),
            "symbols_failed":    len(failed),
            "failed_symbols":    failed,
            "new_bars_inserted": total_inserted,
            "elapsed_seconds":   elapsed,
            "db_stats":          stats,
        }
