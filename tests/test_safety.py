"""
Tests for the safety layer — the most critical module.
Run with: pytest tests/test_safety.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch
from core.safety import SafetyLayer, BreakerLevel
import config.safety_constants as SC


@pytest.fixture
def safety(tmp_path, monkeypatch):
    """SafetyLayer with temp paths so we don't touch real state files."""
    monkeypatch.setattr("core.safety._BREAKER_STATE_PATH", tmp_path / "cb_state.json")
    monkeypatch.setattr("core.safety._SAFETY_HASH_PATH",   tmp_path / "safety.hash")
    monkeypatch.setattr("config.settings.APPROVAL_TOKEN_HASH", "")
    monkeypatch.setattr("config.settings.LIVE_TRADING_ENABLED", False)
    return SafetyLayer()


class TestPositionLimits:

    def test_blocks_order_exceeding_position_limit(self, safety):
        portfolio = 100_000
        # Try to buy 10% when max is 5%
        price = 100.0
        qty   = int(portfolio * 0.10 / price)
        allowed, reason = safety.check_order(
            symbol="AAPL", action="BUY", qty=qty, price=price,
            strategy="test", portfolio_value=portfolio,
            current_position_value=0.0, strategy_exposure=0.0,
        )
        assert not allowed
        assert "Position size" in reason

    def test_allows_order_within_position_limit(self, safety):
        portfolio = 100_000
        price = 100.0
        qty   = int(portfolio * 0.03 / price)  # 3% — within 5% limit
        allowed, reason = safety.check_order(
            symbol="AAPL", action="BUY", qty=qty, price=price,
            strategy="test", portfolio_value=portfolio,
            current_position_value=0.0, strategy_exposure=0.0,
        )
        assert allowed, f"Should be allowed but got: {reason}"

    def test_blocks_meme_stock_exceeding_meme_limit(self, safety):
        portfolio = 100_000
        price = 10.0
        qty   = int(portfolio * 0.05 / price)  # 5% — exceeds 2% meme limit
        allowed, reason = safety.check_order(
            symbol="GME", action="BUY", qty=qty, price=price,
            strategy="meme_momentum", portfolio_value=portfolio,
            current_position_value=0.0, strategy_exposure=0.0,
            is_meme=True,
        )
        assert not allowed
        assert "eme" in reason.lower()

    def test_blocks_penny_stocks(self, safety):
        allowed, reason = safety.check_order(
            symbol="SHIB", action="BUY", qty=1000, price=0.50,
            strategy="test", portfolio_value=100_000,
            current_position_value=0.0, strategy_exposure=0.0,
        )
        assert not allowed
        assert "minimum" in reason.lower() or "below" in reason.lower()

    def test_blocks_strategy_overallocation(self, safety):
        portfolio = 100_000
        price = 100.0
        qty   = 10
        # Current strategy already has 35% (over 30% limit)
        current_exposure = portfolio * 0.35
        allowed, reason = safety.check_order(
            symbol="AAPL", action="BUY", qty=qty, price=price,
            strategy="test", portfolio_value=portfolio,
            current_position_value=0.0, strategy_exposure=current_exposure,
        )
        assert not allowed
        assert "Strategy" in reason or "strategy" in reason


class TestCircuitBreakers:

    def test_level1_at_2pct_loss(self, safety):
        safety._state.peak_portfolio_value = 100_000
        safety._state.daily_start_value    = 100_000
        level = safety.update_portfolio_value(97_800)  # 2.2% loss
        assert level == BreakerLevel.LEVEL_1

    def test_level2_at_3pct_loss(self, safety):
        safety._state.peak_portfolio_value = 100_000
        safety._state.daily_start_value    = 100_000
        level = safety.update_portfolio_value(96_800)  # 3.2% loss
        assert level == BreakerLevel.LEVEL_2

    def test_level3_at_10pct_drawdown(self, safety):
        safety._state.peak_portfolio_value = 100_000
        safety._state.daily_start_value    = 95_000
        level = safety.update_portfolio_value(89_500)  # >10% from peak
        assert level == BreakerLevel.LEVEL_3

    def test_level2_blocks_new_orders(self, safety):
        safety._state.level = BreakerLevel.LEVEL_2
        allowed, reason = safety.check_order(
            symbol="AAPL", action="BUY", qty=10, price=100,
            strategy="test", portfolio_value=100_000,
            current_position_value=0.0, strategy_exposure=0.0,
        )
        assert not allowed
        assert "LEVEL_2" in reason or "Circuit" in reason

    def test_level3_blocks_new_orders(self, safety):
        safety._state.level = BreakerLevel.LEVEL_3
        allowed, reason = safety.check_order(
            symbol="AAPL", action="BUY", qty=10, price=100,
            strategy="test", portfolio_value=100_000,
            current_position_value=0.0, strategy_exposure=0.0,
        )
        assert not allowed
        assert "LEVEL_3" in reason or "Circuit" in reason

    def test_level1_halves_size_multiplier(self, safety):
        safety._state.level = BreakerLevel.LEVEL_1
        assert safety.get_size_multiplier() == 0.5

    def test_normal_gives_full_size(self, safety):
        safety._state.level = BreakerLevel.NORMAL
        assert safety.get_size_multiplier() == 1.0

    def test_never_deescalates_automatically(self, safety):
        safety._state.peak_portfolio_value = 100_000
        safety._state.daily_start_value    = 100_000
        # Hit level 2
        safety.update_portfolio_value(96_500)
        assert safety._state.level == BreakerLevel.LEVEL_2
        # Portfolio recovers slightly — should NOT deescalate
        safety.update_portfolio_value(97_500)
        assert safety._state.level == BreakerLevel.LEVEL_2


class TestRateLimiting:

    def test_blocks_after_max_orders_per_minute(self, safety):
        # Fill up the rate limiter
        for _ in range(SC.MAX_ORDERS_PER_MINUTE):
            safety._record_order()

        allowed, reason = safety.check_order(
            symbol="AAPL", action="BUY", qty=1, price=100,
            strategy="test", portfolio_value=100_000,
            current_position_value=0.0, strategy_exposure=0.0,
        )
        assert not allowed
        assert "Rate limit" in reason or "rate" in reason.lower()


class TestSafetyConstants:

    def test_no_shorting_allowed(self):
        assert SC.SHORT_SELLING is False

    def test_no_leverage(self):
        assert SC.MAX_LEVERAGE == 1.0

    def test_live_trading_requires_human(self):
        assert SC.LIVE_TRADING_REQUIRES_HUMAN is True

    def test_audit_append_only(self):
        assert SC.AUDIT_LOG_APPEND_ONLY is True
