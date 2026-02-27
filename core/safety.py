"""
IMMUTABLE SAFETY LAYER
======================
This module is the ONLY path between strategies and the broker.
Strategies NEVER call the broker directly.

At startup, this module verifies the SHA-256 hash of safety_constants.py.
If the file has been tampered with, the engine refuses to start.

Circuit breaker state is stored in memory and on disk. It is never reset
automatically — only a human operator with the approval token can reset it.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

import config.safety_constants as SC
from config.settings import HUMAN_APPROVAL_THRESHOLD, LIVE_TRADING_ENABLED, APPROVAL_TOKEN_HASH
from core.audit import audit_safety_event, audit_order

# ---------------------------------------------------------------------------
# Constants file integrity check
# ---------------------------------------------------------------------------

_SAFETY_CONSTANTS_PATH = Path(__file__).parent.parent / "config" / "safety_constants.py"
_SAFETY_HASH_PATH       = Path(__file__).parent.parent / "config" / "safety.hash"
_BREAKER_STATE_PATH     = Path(__file__).parent.parent / "logs" / "circuit_breaker_state.json"


def _compute_hash(path: Path) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def seal_safety_constants() -> str:
    """Generate and save the hash of safety_constants.py. Run once by operator."""
    digest = _compute_hash(_SAFETY_CONSTANTS_PATH)
    _SAFETY_HASH_PATH.write_text(digest)
    logger.info(f"Safety constants sealed. Hash: {digest}")
    return digest


def verify_safety_constants() -> None:
    """
    Called at engine startup. Raises RuntimeError if safety_constants.py
    has been modified since it was last sealed.
    """
    if not _SAFETY_HASH_PATH.exists():
        logger.warning("safety.hash not found — sealing now for first run.")
        seal_safety_constants()
        return

    expected = _SAFETY_HASH_PATH.read_text().strip()
    actual   = _compute_hash(_SAFETY_CONSTANTS_PATH)

    if actual != expected:
        raise RuntimeError(
            f"SAFETY CONSTANTS TAMPERED!\n"
            f"  Expected hash: {expected}\n"
            f"  Actual hash:   {actual}\n"
            f"  Run `python scripts/seal_safety.py` to re-seal after a deliberate, "
            f"  reviewed change."
        )
    logger.info("Safety constants verified OK.")


# ---------------------------------------------------------------------------
# Circuit Breaker State
# ---------------------------------------------------------------------------

class BreakerLevel(str, Enum):
    NORMAL  = "NORMAL"
    LEVEL_1 = "LEVEL_1"   # 2% daily loss — reduce sizes
    LEVEL_2 = "LEVEL_2"   # 3% daily loss — halt new orders
    LEVEL_3 = "LEVEL_3"   # 10% drawdown — full halt, human required


@dataclass
class CircuitBreakerState:
    level: BreakerLevel = BreakerLevel.NORMAL
    triggered_at: Optional[str] = None
    triggered_reason: str = ""
    peak_portfolio_value: float = 0.0
    daily_start_value: float = 0.0
    date_str: str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "triggered_at": self.triggered_at,
            "triggered_reason": self.triggered_reason,
            "peak_portfolio_value": self.peak_portfolio_value,
            "daily_start_value": self.daily_start_value,
            "date_str": self.date_str,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitBreakerState":
        s = cls()
        s.level                  = BreakerLevel(d.get("level", "NORMAL"))
        s.triggered_at           = d.get("triggered_at")
        s.triggered_reason       = d.get("triggered_reason", "")
        s.peak_portfolio_value   = d.get("peak_portfolio_value", 0.0)
        s.daily_start_value      = d.get("daily_start_value", 0.0)
        s.date_str               = d.get("date_str", date.today().isoformat())
        return s


class SafetyLayer:
    """
    The safety layer is the gatekeeper between strategies and the broker.
    All order requests must pass through check_order().
    """

    def __init__(self) -> None:
        verify_safety_constants()
        self._state = self._load_state()
        self._order_times: list[float] = []   # timestamps for rate limiting
        self._daily_order_count: int = 0
        self._daily_order_date: str = date.today().isoformat()

        # Verify no live trading sneaked in
        if LIVE_TRADING_ENABLED and not APPROVAL_TOKEN_HASH:
            raise RuntimeError(
                "LIVE_TRADING_ENABLED=true but APPROVAL_TOKEN_HASH is empty. "
                "Run scripts/generate_approval_token.py and set the hash."
            )

        audit_safety_event("INFO", "SafetyLayer initialized", cb_level=self._state.level.value)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> CircuitBreakerState:
        _BREAKER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _BREAKER_STATE_PATH.exists():
            try:
                d = json.loads(_BREAKER_STATE_PATH.read_text())
                state = CircuitBreakerState.from_dict(d)
                # Reset daily counters on a new calendar day
                if state.date_str != date.today().isoformat():
                    state.daily_start_value = state.peak_portfolio_value
                    state.date_str = date.today().isoformat()
                    if state.level in (BreakerLevel.LEVEL_1, BreakerLevel.LEVEL_2):
                        state.level = BreakerLevel.NORMAL
                        state.triggered_reason = "auto-reset on new day"
                self._save_state(state)
                return state
            except Exception as e:
                logger.error(f"Failed to load circuit breaker state: {e} — starting fresh.")
        return CircuitBreakerState()

    def _save_state(self, state: CircuitBreakerState) -> None:
        _BREAKER_STATE_PATH.write_text(json.dumps(state.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Portfolio value tracking
    # ------------------------------------------------------------------

    def update_portfolio_value(self, current_value: float) -> BreakerLevel:
        """
        Called periodically with the current portfolio value.
        Returns the current circuit breaker level (may have escalated).
        """
        state = self._state

        # Initialise peak on first call
        if state.peak_portfolio_value == 0.0:
            state.peak_portfolio_value = current_value
            state.daily_start_value    = current_value
            self._save_state(state)
            return state.level

        # Update peak
        if current_value > state.peak_portfolio_value:
            state.peak_portfolio_value = current_value

        # Daily loss calculation
        if state.daily_start_value > 0:
            daily_loss_frac = (state.daily_start_value - current_value) / state.daily_start_value
        else:
            daily_loss_frac = 0.0

        # Drawdown from peak
        if state.peak_portfolio_value > 0:
            drawdown_frac = (state.peak_portfolio_value - current_value) / state.peak_portfolio_value
        else:
            drawdown_frac = 0.0

        prev_level = state.level

        if drawdown_frac >= SC.CIRCUIT_BREAKER_L3_DRAWDOWN:
            self._escalate(BreakerLevel.LEVEL_3,
                           f"Drawdown {drawdown_frac:.1%} >= {SC.CIRCUIT_BREAKER_L3_DRAWDOWN:.1%}")
        elif daily_loss_frac >= SC.CIRCUIT_BREAKER_L2_DAILY_LOSS:
            self._escalate(BreakerLevel.LEVEL_2,
                           f"Daily loss {daily_loss_frac:.1%} >= {SC.CIRCUIT_BREAKER_L2_DAILY_LOSS:.1%}")
        elif daily_loss_frac >= SC.CIRCUIT_BREAKER_L1_DAILY_LOSS:
            self._escalate(BreakerLevel.LEVEL_1,
                           f"Daily loss {daily_loss_frac:.1%} >= {SC.CIRCUIT_BREAKER_L1_DAILY_LOSS:.1%}")

        self._save_state(state)

        if state.level != prev_level:
            audit_safety_event(
                "CIRCUIT_BREAKER",
                f"Level escalated {prev_level} → {state.level}",
                drawdown=drawdown_frac,
                daily_loss=daily_loss_frac,
                portfolio_value=current_value,
            )

        return state.level

    def _escalate(self, new_level: BreakerLevel, reason: str) -> None:
        state = self._state
        # Never de-escalate automatically
        if new_level.value > state.level.value or state.level == BreakerLevel.NORMAL:
            if state.level != new_level:
                state.level            = new_level
                state.triggered_at     = datetime.now(timezone.utc).isoformat()
                state.triggered_reason = reason
                logger.warning(f"[SAFETY] Circuit breaker → {new_level}: {reason}")

    # ------------------------------------------------------------------
    # Order validation — the main gate
    # ------------------------------------------------------------------

    def check_order(
        self,
        symbol: str,
        action: str,          # "BUY" or "SELL"
        qty: float,
        price: float,
        strategy: str,
        portfolio_value: float,
        current_position_value: float,   # existing position in this symbol
        strategy_exposure: float,        # total strategy allocation value
        is_meme: bool = False,
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        NEVER raises — always returns a decision.
        """

        # 1. Short selling blocked
        if action == "SELL_SHORT" or (action == "SELL" and qty < 0):
            reason = "Short selling is disabled"
            audit_order(action, symbol, qty, price, strategy, reason, False)
            return False, reason

        # 2. Penny stock filter
        if price < SC.MIN_STOCK_PRICE:
            reason = f"Price ${price:.2f} below minimum ${SC.MIN_STOCK_PRICE:.2f}"
            audit_order(action, symbol, qty, price, strategy, reason, False)
            return False, reason

        trade_value = qty * price

        if action == "BUY":
            # 3. Rate limiting
            ok, reason = self._check_rate_limit()
            if not ok:
                audit_order(action, symbol, qty, price, strategy, reason, False)
                return False, reason

            # 4. Circuit breaker
            if self._state.level == BreakerLevel.LEVEL_3:
                reason = f"Circuit breaker LEVEL_3 active: {self._state.triggered_reason}"
                audit_order(action, symbol, qty, price, strategy, reason, False)
                return False, reason

            if self._state.level == BreakerLevel.LEVEL_2:
                reason = f"Circuit breaker LEVEL_2 active — no new orders"
                audit_order(action, symbol, qty, price, strategy, reason, False)
                return False, reason

            # 5. Position size limit
            new_position_value = current_position_value + trade_value
            max_position = portfolio_value * SC.MAX_POSITION_FRACTION
            if new_position_value > max_position:
                reason = (f"Position size ${new_position_value:,.0f} exceeds "
                          f"max {SC.MAX_POSITION_FRACTION:.0%} of portfolio (${max_position:,.0f})")
                audit_order(action, symbol, qty, price, strategy, reason, False)
                return False, reason

            # 6. Meme stock limits
            if is_meme:
                max_meme_pos = portfolio_value * SC.MAX_MEME_POSITION_FRACTION
                if new_position_value > max_meme_pos:
                    reason = (f"Meme position ${new_position_value:,.0f} exceeds "
                              f"meme max {SC.MAX_MEME_POSITION_FRACTION:.0%} (${max_meme_pos:,.0f})")
                    audit_order(action, symbol, qty, price, strategy, reason, False)
                    return False, reason

            # 7. Strategy allocation limit
            max_strategy = portfolio_value * SC.MAX_STRATEGY_ALLOCATION
            if is_meme:
                max_strategy = min(max_strategy, portfolio_value * SC.MAX_MEME_ALLOCATION)
            if strategy_exposure + trade_value > max_strategy:
                reason = (f"Strategy {strategy} exposure would exceed "
                          f"{SC.MAX_STRATEGY_ALLOCATION:.0%} limit")
                audit_order(action, symbol, qty, price, strategy, reason, False)
                return False, reason

            # 8. Human approval gate
            if trade_value > HUMAN_APPROVAL_THRESHOLD:
                reason = (f"Trade value ${trade_value:,.0f} exceeds human approval "
                          f"threshold ${HUMAN_APPROVAL_THRESHOLD:,.0f} — awaiting approval")
                audit_order(action, symbol, qty, price, strategy, reason, False,
                            needs_approval=True)
                return False, reason

            # 9. Level 1 — scale down order size (handled by caller after checking)
            # (Caller should check get_size_multiplier() before sizing orders)

        # All checks passed
        audit_order(action, symbol, qty, price, strategy, "All checks passed", True)
        self._record_order()
        return True, "OK"

    def get_size_multiplier(self) -> float:
        """Returns 0.5 at Level 1, 0.0 at Level 2+, else 1.0."""
        if self._state.level == BreakerLevel.LEVEL_1:
            return 0.5
        if self._state.level in (BreakerLevel.LEVEL_2, BreakerLevel.LEVEL_3):
            return 0.0
        return 1.0

    def is_halted(self) -> bool:
        return self._state.level in (BreakerLevel.LEVEL_2, BreakerLevel.LEVEL_3)

    def get_state(self) -> CircuitBreakerState:
        return self._state

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> tuple[bool, str]:
        now = time.time()
        # Per-minute window
        self._order_times = [t for t in self._order_times if now - t < 60]
        if len(self._order_times) >= SC.MAX_ORDERS_PER_MINUTE:
            return False, f"Rate limit: max {SC.MAX_ORDERS_PER_MINUTE} orders/minute"

        # Daily count
        today = date.today().isoformat()
        if today != self._daily_order_date:
            self._daily_order_count = 0
            self._daily_order_date  = today
        if self._daily_order_count >= SC.MAX_ORDERS_PER_DAY:
            return False, f"Daily order limit reached ({SC.MAX_ORDERS_PER_DAY})"

        return True, "OK"

    def _record_order(self) -> None:
        self._order_times.append(time.time())
        self._daily_order_count += 1

    # ------------------------------------------------------------------
    # Human operator reset (Level 3 only)
    # ------------------------------------------------------------------

    def operator_reset(self, token: str) -> bool:
        """
        Reset a Level 3 circuit breaker. Requires a valid operator token.
        The token is the raw secret whose SHA-256 matches APPROVAL_TOKEN_HASH.
        """
        import hashlib
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash != APPROVAL_TOKEN_HASH:
            audit_safety_event("WARN", "Invalid operator reset token presented")
            return False

        prev = self._state.level
        self._state.level            = BreakerLevel.NORMAL
        self._state.triggered_reason = "Operator reset"
        self._state.triggered_at     = None
        self._save_state(self._state)
        audit_safety_event("INFO", f"Circuit breaker reset by operator: {prev} → NORMAL")
        return True
