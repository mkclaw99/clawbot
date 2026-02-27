"""
Local paper broker — no external account needed.
Stores positions and cash in SQLite; prices from yfinance.
Each strategy gets its own isolated portfolio via portfolio_id.
Strategies NEVER import this directly; they go through core/safety.py.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import create_engine, Column, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, Session

from config.settings import DATABASE_URL, STARTING_CAPITAL

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class PaperAccount(Base):
    __tablename__ = "paper_account"
    id               = Column(String, primary_key=True)   # portfolio_id
    cash             = Column(Float, nullable=False)
    starting_capital = Column(Float, nullable=False)
    created_at       = Column(DateTime)
    updated_at       = Column(DateTime)


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    portfolio_id    = Column(String, primary_key=True)
    symbol          = Column(String, primary_key=True)
    qty             = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    created_at      = Column(DateTime)
    updated_at      = Column(DateTime)


# ---------------------------------------------------------------------------
# Data models (public interface)
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    order_id: str
    symbol: str
    action: str          # BUY | SELL
    qty: float
    filled_price: float
    status: str          # filled | rejected | error
    timestamp: str


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class AccountInfo:
    portfolio_value: float
    cash: float
    buying_power: float
    equity: float


# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------

_price_cache: dict[str, tuple[float, float]] = {}  # symbol → (price, ts)
_CACHE_TTL = 60.0


def _get_price(symbol: str) -> float:
    now = time.time()
    if symbol in _price_cache:
        price, ts = _price_cache[symbol]
        if now - ts < _CACHE_TTL:
            return price
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get("last_price") or ticker.fast_info.get("regularMarketPrice")
        if not price or price == 0:
            hist = ticker.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        price = float(price)
    except Exception as e:
        logger.warning(f"Price lookup failed for {symbol}: {e} — using 0.0")
        price = 0.0
    _price_cache[symbol] = (price, now)
    return price


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class Broker:
    """
    Local paper broker, one instance per portfolio_id (strategy name).
    All state persists in SQLite. Prices from yfinance (no API key needed).
    """

    def __init__(self, portfolio_id: str = "default") -> None:
        self._portfolio_id = portfolio_id
        self._engine = create_engine(DATABASE_URL, echo=False)
        self._migrate_schema()
        Base.metadata.create_all(self._engine)
        self._ensure_account()
        logger.info(f"LocalPaperBroker ready (portfolio={portfolio_id})")

    def _migrate_schema(self) -> None:
        """Drop old single-PK paper_positions table if it predates portfolio_id column."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM pragma_table_info('paper_positions') "
                    "WHERE name='portfolio_id'"
                ))
                has_col = result.scalar() > 0
            if not has_col:
                with self._engine.connect() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS paper_positions"))
                    conn.execute(text("DROP TABLE IF EXISTS paper_account"))
                    conn.commit()
                logger.info("Migrated paper_positions schema to per-portfolio layout.")
        except Exception:
            pass  # table doesn't exist yet — create_all will handle it

    def _ensure_account(self) -> None:
        with Session(self._engine) as session:
            acct = session.get(PaperAccount, self._portfolio_id)
            if acct is None:
                now = datetime.now(timezone.utc)
                session.add(PaperAccount(
                    id=self._portfolio_id,
                    cash=STARTING_CAPITAL,
                    starting_capital=STARTING_CAPITAL,
                    created_at=now,
                    updated_at=now,
                ))
                session.commit()
                logger.info(f"Portfolio '{self._portfolio_id}' initialised with ${STARTING_CAPITAL:,.0f}")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        try:
            with Session(self._engine) as session:
                acct = session.get(PaperAccount, self._portfolio_id)
                cash = acct.cash if acct else STARTING_CAPITAL
                positions = self._fetch_positions(session)
                equity = sum(p.market_value for p in positions)
            return AccountInfo(
                portfolio_value=cash + equity,
                cash=cash,
                buying_power=cash,
                equity=equity,
            )
        except Exception as e:
            logger.error(f"get_account [{self._portfolio_id}] failed: {e}")
            return AccountInfo(0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        try:
            with Session(self._engine) as session:
                return self._fetch_positions(session)
        except Exception as e:
            logger.error(f"get_positions [{self._portfolio_id}] failed: {e}")
            return []

    def _fetch_positions(self, session: Session) -> list[Position]:
        rows = (session.query(PaperPosition)
                .filter(PaperPosition.portfolio_id == self._portfolio_id)
                .all())
        out = []
        for row in rows:
            if row.qty <= 0:
                continue
            cp = _get_price(row.symbol)
            mv = row.qty * cp
            pnl = (cp - row.avg_entry_price) * row.qty
            pct = ((cp / row.avg_entry_price) - 1) if row.avg_entry_price else 0.0
            out.append(Position(
                symbol=row.symbol, qty=row.qty,
                avg_entry_price=row.avg_entry_price,
                current_price=cp, market_value=mv,
                unrealized_pnl=pnl, unrealized_pnl_pct=pct,
            ))
        return out

    def get_position(self, symbol: str) -> Optional[Position]:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def market_buy(self, symbol: str, qty: float) -> OrderResult:
        logger.info(f"[BROKER:{self._portfolio_id}] BUY {qty} {symbol}")
        qty = float(qty)
        if qty <= 0:
            return OrderResult("", symbol, "BUY", qty, 0, "rejected",
                               datetime.now(timezone.utc).isoformat())

        price = _get_price(symbol)
        if price <= 0:
            return OrderResult("", symbol, "BUY", qty, 0, "error",
                               datetime.now(timezone.utc).isoformat())

        cost = qty * price
        try:
            with Session(self._engine) as session:
                acct = session.get(PaperAccount, self._portfolio_id)
                if acct.cash < cost:
                    logger.warning(f"[{self._portfolio_id}] Insufficient cash: need ${cost:.2f}, have ${acct.cash:.2f}")
                    return OrderResult("", symbol, "BUY", qty, price, "rejected",
                                       datetime.now(timezone.utc).isoformat())

                acct.cash -= cost
                acct.updated_at = datetime.now(timezone.utc)

                pos = session.get(PaperPosition, (self._portfolio_id, symbol))
                now = datetime.now(timezone.utc)
                if pos:
                    total_qty = pos.qty + qty
                    avg       = (pos.avg_entry_price * pos.qty + price * qty) / total_qty
                    pos.qty             = total_qty
                    pos.avg_entry_price = avg
                    pos.updated_at      = now
                else:
                    session.add(PaperPosition(
                        portfolio_id=self._portfolio_id, symbol=symbol,
                        qty=qty, avg_entry_price=price,
                        created_at=now, updated_at=now,
                    ))
                session.commit()

            order_id = f"paper-B-{self._portfolio_id[:4]}-{symbol}-{uuid.uuid4().hex[:6]}"
            logger.info(f"[BROKER:{self._portfolio_id}] BUY filled: {qty} {symbol} @ ${price:.2f}")
            return OrderResult(order_id=order_id, symbol=symbol, action="BUY",
                               qty=qty, filled_price=price, status="filled",
                               timestamp=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            logger.error(f"market_buy [{self._portfolio_id}] {symbol}: {e}")
            return OrderResult("ERROR", symbol, "BUY", qty, 0, "error", "")

    def market_sell(self, symbol: str, qty: float) -> OrderResult:
        logger.info(f"[BROKER:{self._portfolio_id}] SELL {qty} {symbol}")
        qty = float(qty)
        if qty <= 0:
            return OrderResult("", symbol, "SELL", qty, 0, "rejected",
                               datetime.now(timezone.utc).isoformat())

        price = _get_price(symbol)
        if price <= 0:
            return OrderResult("", symbol, "SELL", qty, 0, "error",
                               datetime.now(timezone.utc).isoformat())

        try:
            with Session(self._engine) as session:
                pos = session.get(PaperPosition, (self._portfolio_id, symbol))
                if not pos or pos.qty < qty:
                    have = pos.qty if pos else 0
                    logger.warning(f"[{self._portfolio_id}] Can't sell {qty} {symbol}: have {have}")
                    return OrderResult("", symbol, "SELL", qty, price, "rejected",
                                       datetime.now(timezone.utc).isoformat())

                acct = session.get(PaperAccount, self._portfolio_id)
                acct.cash += qty * price
                acct.updated_at = datetime.now(timezone.utc)

                pos.qty -= qty
                pos.updated_at = datetime.now(timezone.utc)
                if pos.qty <= 0:
                    session.delete(pos)
                session.commit()

            order_id = f"paper-S-{self._portfolio_id[:4]}-{symbol}-{uuid.uuid4().hex[:6]}"
            logger.info(f"[BROKER:{self._portfolio_id}] SELL filled: {qty} {symbol} @ ${price:.2f}")
            return OrderResult(order_id=order_id, symbol=symbol, action="SELL",
                               qty=qty, filled_price=price, status="filled",
                               timestamp=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            logger.error(f"market_sell [{self._portfolio_id}] {symbol}: {e}")
            return OrderResult("ERROR", symbol, "SELL", qty, 0, "error", "")

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        pos = self.get_position(symbol)
        if pos and pos.qty > 0:
            return self.market_sell(symbol, pos.qty)
        return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_latest_price(self, symbol: str) -> float:
        return _get_price(symbol)

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> list[dict]:
        try:
            import yfinance as yf
            period_map = {"1Day": "1d", "1Hour": "1h", "5Min": "5m"}
            interval = period_map.get(timeframe, "1d")
            df = yf.download(symbol, period=f"{limit}d", interval=interval,
                             progress=False, auto_adjust=True)
            return [
                {"t": ts.isoformat(), "o": float(r["Open"]), "h": float(r["High"]),
                 "l": float(r["Low"]),  "c": float(r["Close"]), "v": float(r["Volume"])}
                for ts, r in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"get_bars {symbol}: {e}")
            return []
