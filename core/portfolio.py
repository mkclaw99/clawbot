"""
Portfolio state manager.
Tracks positions, P&L, trade history, and per-strategy exposure.
Persists to SQLite via SQLAlchemy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, Boolean, Text, text
)
from sqlalchemy.orm import declarative_base, Session

from config.settings import DATABASE_URL, STARTING_CAPITAL
from core.broker import Broker, Position, AccountInfo

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class TradeRecord(Base):
    __tablename__ = "trades"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    order_id     = Column(String, unique=True, index=True)
    symbol       = Column(String, index=True)
    action       = Column(String)          # BUY | SELL
    qty          = Column(Float)
    price        = Column(Float)
    strategy     = Column(String, index=True)
    timestamp    = Column(DateTime, index=True)
    pnl          = Column(Float, default=0.0)
    is_meme      = Column(Boolean, default=False)
    notes        = Column(Text, default="")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime, index=True)
    portfolio_id    = Column(String, index=True, default="default")   # which strategy portfolio
    portfolio_value = Column(Float)
    cash            = Column(Float)
    equity          = Column(Float)
    daily_pnl       = Column(Float, default=0.0)
    total_pnl       = Column(Float, default=0.0)


class StrategyStats(Base):
    __tablename__ = "strategy_stats"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    strategy        = Column(String, unique=True, index=True)
    allocation      = Column(Float, default=0.0)   # current exposure $
    total_trades    = Column(Integer, default=0)
    winning_trades  = Column(Integer, default=0)
    total_pnl       = Column(Float, default=0.0)
    last_updated    = Column(DateTime)


# ---------------------------------------------------------------------------
# Portfolio manager
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    portfolio_value: float = STARTING_CAPITAL
    cash: float            = STARTING_CAPITAL
    equity: float          = 0.0
    buying_power: float    = STARTING_CAPITAL
    positions: list[Position] = field(default_factory=list)
    daily_pnl: float       = 0.0
    total_pnl: float       = 0.0
    start_value: float     = STARTING_CAPITAL


class PortfolioManager:

    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._engine = create_engine(DATABASE_URL, echo=False)
        self._migrate_schema()
        Base.metadata.create_all(self._engine)
        self._state = PortfolioState()
        self._strategy_exposure: dict[str, float] = {}
        self._entry_prices: dict[str, float] = {}   # symbol → avg buy price
        logger.info("PortfolioManager initialized.")

    def _migrate_schema(self) -> None:
        """Add portfolio_id column to portfolio_snapshots if missing (schema v2)."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM pragma_table_info('portfolio_snapshots') "
                    "WHERE name='portfolio_id'"
                ))
                has_col = result.scalar() > 0
            if not has_col:
                with self._engine.connect() as conn:
                    conn.execute(text(
                        "ALTER TABLE portfolio_snapshots "
                        "ADD COLUMN portfolio_id TEXT DEFAULT 'default'"
                    ))
                    conn.commit()
                logger.info("Migrated portfolio_snapshots: added portfolio_id column.")
        except Exception:
            pass  # table doesn't exist yet — create_all handles it

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def sync(self) -> PortfolioState:
        """Fetch latest state from broker and persist a snapshot."""
        acct: AccountInfo = self._broker.get_account()
        positions         = self._broker.get_positions()

        self._state.portfolio_value = acct.portfolio_value
        self._state.cash            = acct.cash
        self._state.equity          = acct.equity
        self._state.buying_power    = acct.buying_power
        self._state.positions       = positions

        self._state.total_pnl = acct.portfolio_value - self._state.start_value
        self._save_snapshot()

        return self._state

    def _save_snapshot(self) -> None:
        with Session(self._engine) as session:
            snap = PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc),
                portfolio_id=self._broker._portfolio_id,
                portfolio_value=self._state.portfolio_value,
                cash=self._state.cash,
                equity=self._state.equity,
                total_pnl=self._state.total_pnl,
            )
            session.add(snap)
            session.commit()

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, order_id: str, symbol: str, action: str,
                     qty: float, price: float, strategy: str,
                     is_meme: bool = False) -> None:
        pnl = 0.0
        if action == "SELL":
            entry = self._entry_prices.get(symbol, price)
            pnl   = (price - entry) * qty

        if action == "BUY":
            self._entry_prices[symbol] = price

        with Session(self._engine) as session:
            # Upsert strategy stats
            stats = session.query(StrategyStats).filter_by(strategy=strategy).first()
            if not stats:
                stats = StrategyStats(
                    strategy=strategy,
                    total_trades=0,
                    winning_trades=0,
                    total_pnl=0.0,
                    allocation=0.0,
                    last_updated=datetime.now(timezone.utc),
                )
                session.add(stats)
            stats.total_trades   = (stats.total_trades  or 0) + 1
            stats.total_pnl      = (stats.total_pnl     or 0) + pnl
            if pnl > 0:
                stats.winning_trades = (stats.winning_trades or 0) + 1
            stats.last_updated = datetime.now(timezone.utc)

            trade = TradeRecord(
                order_id=order_id,
                symbol=symbol,
                action=action,
                qty=qty,
                price=price,
                strategy=strategy,
                timestamp=datetime.now(timezone.utc),
                pnl=pnl,
                is_meme=is_meme,
            )
            session.add(trade)
            session.commit()

        # Update in-memory exposure
        trade_value = qty * price
        if action == "BUY":
            self._strategy_exposure[strategy] = (
                self._strategy_exposure.get(strategy, 0.0) + trade_value
            )
        elif action == "SELL":
            self._strategy_exposure[strategy] = max(
                0.0, self._strategy_exposure.get(strategy, 0.0) - trade_value
            )

    def get_strategy_exposure(self, strategy: str) -> float:
        return self._strategy_exposure.get(strategy, 0.0)

    def get_position_value(self, symbol: str) -> float:
        for pos in self._state.positions:
            if pos.symbol == symbol:
                return pos.market_value
        return 0.0

    def get_state(self) -> PortfolioState:
        return self._state

    # ------------------------------------------------------------------
    # History queries (for dashboard)
    # ------------------------------------------------------------------

    def get_trade_history(self, limit: int = 200) -> list[dict]:
        with Session(self._engine) as session:
            rows = (session.query(TradeRecord)
                    .order_by(TradeRecord.timestamp.desc())
                    .limit(limit)
                    .all())
            return [
                {
                    "order_id": r.order_id,
                    "symbol": r.symbol,
                    "action": r.action,
                    "qty": r.qty,
                    "price": r.price,
                    "strategy": r.strategy,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "pnl": r.pnl,
                    "is_meme": r.is_meme,
                }
                for r in rows
            ]

    def get_equity_curve(self, limit: int = 500) -> list[dict]:
        with Session(self._engine) as session:
            rows = (session.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.asc())
                    .limit(limit)
                    .all())
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "portfolio_value": r.portfolio_value,
                    "total_pnl": r.total_pnl,
                    "portfolio_id": r.portfolio_id or "default",
                }
                for r in rows
            ]

    def get_equity_curves_by_strategy(self, limit_per_strategy: int = 500) -> dict[str, list[dict]]:
        """Return per-strategy equity curves keyed by portfolio_id."""
        with Session(self._engine) as session:
            rows = (session.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.asc())
                    .limit(limit_per_strategy * 10)
                    .all())
        result: dict[str, list[dict]] = {}
        for r in rows:
            pid = r.portfolio_id or "default"
            if pid not in result:
                result[pid] = []
            result[pid].append({
                "timestamp": r.timestamp.isoformat(),
                "portfolio_value": r.portfolio_value,
                "total_pnl": r.total_pnl,
            })
        # Trim each strategy to most recent limit_per_strategy points
        for pid in result:
            result[pid] = result[pid][-limit_per_strategy:]
        return result

    def get_strategy_stats(self) -> list[dict]:
        with Session(self._engine) as session:
            rows = session.query(StrategyStats).all()
            results = []
            for r in rows:
                win_rate = (r.winning_trades / r.total_trades * 100
                            if r.total_trades > 0 else 0.0)
                results.append({
                    "strategy": r.strategy,
                    "allocation": self._strategy_exposure.get(r.strategy, 0.0),
                    "total_trades": r.total_trades,
                    "winning_trades": r.winning_trades,
                    "win_rate": round(win_rate, 1),
                    "total_pnl": round(r.total_pnl, 2),
                })
            return results
