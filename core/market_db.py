"""
Market data database (market.db — separate from trading state clawbot.db).

Tables:
  price_bars          — daily OHLCV + returns for every tracked symbol
  correlation_results — pre-computed pairwise Pearson correlations (various windows & lags)

Designed to hold 1-2 years of daily bars for 150+ symbols (~100k rows) and
a correlation matrix of ~10k pairs.  SQLite handles this comfortably.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column, Float, Integer, String, Date, DateTime,
    UniqueConstraint, create_engine,
)
from sqlalchemy.orm import declarative_base, Session

from config.settings import MARKET_DB_URL

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class PriceBar(Base):
    __tablename__  = "price_bars"
    __table_args__ = (UniqueConstraint("symbol", "date", name="uq_price_bar"),)

    id           = Column(Integer, primary_key=True, autoincrement=True)
    symbol       = Column(String, index=True, nullable=False)
    date         = Column(Date,   index=True, nullable=False)
    open         = Column(Float)
    high         = Column(Float)
    low          = Column(Float)
    close        = Column(Float, nullable=False)
    volume       = Column(Float)
    daily_return = Column(Float)   # (close / prev_close) - 1
    log_return   = Column(Float)   # log(close / prev_close)
    fetched_at   = Column(DateTime)


class CorrelationResult(Base):
    __tablename__ = "correlation_results"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    computed_at = Column(DateTime, index=True)
    period_days = Column(Integer)           # 30 | 60 | 90
    symbol_a    = Column(String, index=True)
    symbol_b    = Column(String, index=True)
    pearson_r   = Column(Float)             # −1.0 … 1.0
    lag_days    = Column(Integer, default=0) # 0 = contemporaneous; >0 = a leads b
    p_value     = Column(Float)             # two-tailed significance


# ---------------------------------------------------------------------------
# MarketDatabase
# ---------------------------------------------------------------------------

class MarketDatabase:

    def __init__(self) -> None:
        self._engine = create_engine(MARKET_DB_URL, echo=False)
        Base.metadata.create_all(self._engine)
        logger.info(f"[MarketDB] Ready — {MARKET_DB_URL}")

    # ------------------------------------------------------------------
    # Price bars
    # ------------------------------------------------------------------

    def upsert_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Bulk-insert new daily bars for a symbol.
        df must have columns: date (date), open, high, low, close, volume,
                              daily_return, log_return
        Skips dates already present in the DB (append-only for history).
        Returns number of rows inserted.
        """
        if df.empty:
            return 0

        with Session(self._engine) as session:
            existing = {
                r.date for r in
                session.query(PriceBar.date)
                       .filter(PriceBar.symbol == symbol)
                       .all()
            }

        new_df = df[~df["date"].isin(existing)]
        if new_df.empty:
            return 0

        now = datetime.now(timezone.utc)
        records = [
            PriceBar(
                symbol=symbol,
                date=row["date"],
                open=row.get("open"),
                high=row.get("high"),
                low=row.get("low"),
                close=row["close"],
                volume=row.get("volume"),
                daily_return=row.get("daily_return"),
                log_return=row.get("log_return"),
                fetched_at=now,
            )
            for _, row in new_df.iterrows()
        ]

        with Session(self._engine) as session:
            session.add_all(records)
            session.commit()

        return len(records)

    def get_bars(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Return the most recent `days` bars for a symbol as a DataFrame."""
        with Session(self._engine) as session:
            rows = (
                session.query(PriceBar)
                       .filter(PriceBar.symbol == symbol)
                       .order_by(PriceBar.date.desc())
                       .limit(days)
                       .all()
            )
        if not rows:
            return pd.DataFrame()
        data = [
            {"date": r.date, "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume,
             "daily_return": r.daily_return, "log_return": r.log_return}
            for r in rows
        ]
        return pd.DataFrame(data).sort_values("date").reset_index(drop=True)

    def get_returns_matrix(self, symbols: list[str], days: int = 90) -> pd.DataFrame:
        """
        Returns a (days × n_symbols) DataFrame of daily returns.
        Drops symbols with >50 % missing data.
        """
        frames: dict[str, pd.Series] = {}
        with Session(self._engine) as session:
            for sym in symbols:
                rows = (
                    session.query(PriceBar.date, PriceBar.daily_return)
                           .filter(
                               PriceBar.symbol == sym,
                               PriceBar.daily_return.isnot(None),
                           )
                           .order_by(PriceBar.date.desc())
                           .limit(days)
                           .all()
                )
                if rows:
                    frames[sym] = pd.Series(
                        [r.daily_return for r in rows],
                        index=[r.date for r in rows],
                        name=sym,
                    )
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames).sort_index()
        return df.dropna(axis=1, thresh=max(1, int(days * 0.5)))

    def get_available_symbols(self) -> list[str]:
        with Session(self._engine) as session:
            return [r[0] for r in session.query(PriceBar.symbol).distinct().all()]

    def get_stats(self) -> dict:
        with Session(self._engine) as session:
            n_symbols = session.query(PriceBar.symbol).distinct().count()
            n_bars    = session.query(PriceBar).count()
            latest    = session.query(PriceBar.date).order_by(PriceBar.date.desc()).first()
            n_corr    = session.query(CorrelationResult).count()
        return {
            "symbols":           n_symbols,
            "bars":              n_bars,
            "latest_date":       str(latest[0]) if latest else None,
            "correlations":      n_corr,
        }

    # ------------------------------------------------------------------
    # Correlation results
    # ------------------------------------------------------------------

    def save_correlations(
        self, results: list[dict], period_days: int, lag_days: int = 0
    ) -> None:
        """
        Replace correlations for a specific (period_days, lag_days) bucket.
        Scoping by lag avoids clobbering contemporaneous results when saving lag data.
        """
        now = datetime.now(timezone.utc)
        with Session(self._engine) as session:
            (session.query(CorrelationResult)
                    .filter_by(period_days=period_days, lag_days=lag_days)
                    .delete())
            session.add_all([
                CorrelationResult(
                    computed_at=now,
                    period_days=period_days,
                    symbol_a=r["symbol_a"],
                    symbol_b=r["symbol_b"],
                    pearson_r=r["pearson_r"],
                    lag_days=lag_days,
                    p_value=r.get("p_value", 1.0),
                )
                for r in results
            ])
            session.commit()
        logger.info(
            f"[MarketDB] Saved {len(results)} correlations "
            f"(period={period_days}d, lag={lag_days}d)"
        )

    def get_top_correlations(
        self,
        symbol: str,
        period_days: int = 60,
        min_abs_r: float = 0.5,
        limit: int = 10,
        lag_days: int = 0,
    ) -> list[dict]:
        """Top correlated / anti-correlated peers for a symbol."""
        with Session(self._engine) as session:
            rows = (
                session.query(CorrelationResult)
                       .filter(
                           CorrelationResult.period_days == period_days,
                           CorrelationResult.lag_days    == lag_days,
                       )
                       .all()
            )
        results = []
        for r in rows:
            if r.symbol_a == symbol:
                peer = r.symbol_b
            elif r.symbol_b == symbol:
                peer = r.symbol_a
            else:
                continue
            if abs(r.pearson_r) >= min_abs_r:
                results.append({
                    "peer":      peer,
                    "pearson_r": round(r.pearson_r, 3),
                    "lag_days":  r.lag_days,
                    "p_value":   round(r.p_value, 4),
                })
        return sorted(results, key=lambda x: abs(x["pearson_r"]), reverse=True)[:limit]

    def get_correlation_matrix(
        self, symbols: list[str], period_days: int = 60
    ) -> pd.DataFrame:
        """Full n×n correlation DataFrame for a symbol list (dashboard heatmap)."""
        sym_set = set(symbols)
        with Session(self._engine) as session:
            rows = (
                session.query(CorrelationResult)
                       .filter(
                           CorrelationResult.period_days == period_days,
                           CorrelationResult.lag_days    == 0,
                           CorrelationResult.symbol_a.in_(sym_set),
                           CorrelationResult.symbol_b.in_(sym_set),
                       )
                       .all()
            )
        mat = {s: {s: 1.0} for s in symbols}
        for r in rows:
            if r.symbol_a in sym_set and r.symbol_b in sym_set:
                mat[r.symbol_a][r.symbol_b] = r.pearson_r
                mat[r.symbol_b][r.symbol_a] = r.pearson_r
        df = pd.DataFrame(mat, index=symbols, columns=symbols).fillna(0)
        # Ensure self-correlation = 1
        for s in symbols:
            if s in df.index and s in df.columns:
                df.loc[s, s] = 1.0
        return df

    def get_latest_computed_at(self) -> str | None:
        with Session(self._engine) as session:
            row = (
                session.query(CorrelationResult.computed_at)
                       .order_by(CorrelationResult.computed_at.desc())
                       .first()
            )
        return row[0].isoformat() if row else None

    def get_all_correlation_pairs(
        self, period_days: int = 60, min_abs_r: float = 0.6, lag_days: int = 0
    ) -> list[dict]:
        """All pairs above a correlation threshold — for the dashboard table."""
        with Session(self._engine) as session:
            rows = (
                session.query(CorrelationResult)
                       .filter(
                           CorrelationResult.period_days == period_days,
                           CorrelationResult.lag_days    == lag_days,
                           CorrelationResult.p_value     < 0.05,
                       )
                       .all()
            )
        results = [
            {
                "symbol_a":  r.symbol_a,
                "symbol_b":  r.symbol_b,
                "pearson_r": round(r.pearson_r, 3),
                "p_value":   round(r.p_value, 4),
            }
            for r in rows
            if abs(r.pearson_r) >= min_abs_r
        ]
        return sorted(results, key=lambda x: abs(x["pearson_r"]), reverse=True)
