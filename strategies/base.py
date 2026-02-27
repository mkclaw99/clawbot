"""
Abstract base class for all trading strategies.
Every strategy returns a list of SignalResult objects.
The engine applies safety checks before placing any orders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SignalResult:
    symbol: str
    action: str              # "BUY" | "SELL" | "HOLD"
    score: float             # -1.0 (strong sell) to +1.0 (strong buy)
    confidence: float        # 0.0 to 1.0
    reason: str
    strategy: str
    is_meme: bool = False
    suggested_qty: Optional[float] = None   # None = engine decides
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL") and self.confidence >= 0.5


class BaseStrategy(ABC):
    """
    Subclass this for each strategy.
    Subclasses MUST implement: name, description, generate_signals().
    Subclasses SHOULD NOT import core.broker or place orders directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short strategy identifier, e.g. 'technical_trend'"""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the dashboard."""

    @abstractmethod
    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """
        Given a list of ticker symbols and a context dict (containing
        web signals, portfolio state, etc.), return a list of SignalResults.
        This method must be safe to call every 60 seconds during market hours.
        """

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
