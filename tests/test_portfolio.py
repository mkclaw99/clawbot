"""
Tests for core/portfolio.py — PortfolioManager.
Uses a temp SQLite DB and mocked broker prices.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

import core.broker as broker_module
from core.broker import Broker, Position, AccountInfo
from core.portfolio import PortfolioManager, PortfolioState


FAKE_PRICES = {"AAPL": 150.0, "TSLA": 200.0, "NVDA": 500.0}


@pytest.fixture(autouse=True)
def patch_price():
    with patch.object(broker_module, "_get_price", side_effect=lambda s: FAKE_PRICES.get(s, 100.0)):
        yield


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def broker(db_url, monkeypatch):
    monkeypatch.setattr("core.broker.DATABASE_URL", db_url)
    monkeypatch.setattr("core.broker.STARTING_CAPITAL", 100_000.0)
    return Broker(portfolio_id="test")


@pytest.fixture
def pm(broker, db_url, monkeypatch):
    monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
    monkeypatch.setattr("core.portfolio.STARTING_CAPITAL", 100_000.0)
    return PortfolioManager(broker)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

class TestPortfolioSync:

    def test_sync_returns_portfolio_state(self, pm):
        state = pm.sync()
        assert isinstance(state, PortfolioState)

    def test_initial_sync_has_starting_capital(self, pm):
        state = pm.sync()
        assert state.portfolio_value == 100_000.0
        assert state.cash == 100_000.0

    def test_initial_total_pnl_is_zero(self, pm):
        state = pm.sync()
        assert state.total_pnl == 0.0

    def test_sync_after_buy_reflects_positions(self, pm, broker):
        broker.market_buy("AAPL", 10)
        state = pm.sync()
        assert len(state.positions) == 1
        assert state.positions[0].symbol == "AAPL"

    def test_sync_saves_snapshot(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.sync()
        curve = pm.get_equity_curve(limit=10)
        assert len(curve) >= 1

    def test_total_pnl_negative_after_price_drop(self, pm, broker):
        broker.market_buy("AAPL", 10)   # bought @ 150 = $1500
        # Simulate price drop by patching
        with patch.object(broker_module, "_get_price", return_value=100.0):
            state = pm.sync()
        # portfolio_value = (100_000 - 1500) cash + 10*100 equity = 99_500
        assert state.total_pnl < 0.0

    def test_get_state_returns_last_synced(self, pm):
        state1 = pm.sync()
        state2 = pm.get_state()
        assert state1.portfolio_value == state2.portfolio_value


# ---------------------------------------------------------------------------
# record_trade
# ---------------------------------------------------------------------------

class TestRecordTrade:

    def test_buy_trade_recorded(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY", 10, 150.0, "technical_trend")
        history = pm.get_trade_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "AAPL"
        assert history[0]["action"] == "BUY"
        assert history[0]["qty"] == 10
        assert history[0]["price"] == 150.0
        assert history[0]["strategy"] == "technical_trend"

    def test_sell_trade_has_pnl(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY",  10, 100.0, "tech")
        pm.record_trade("ord-2", "AAPL", "SELL", 10, 150.0, "tech")
        history = pm.get_trade_history()
        sell = next(t for t in history if t["action"] == "SELL")
        assert sell["pnl"] == pytest.approx((150.0 - 100.0) * 10, abs=0.01)

    def test_sell_pnl_negative_when_price_down(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY",  10, 200.0, "tech")
        pm.record_trade("ord-2", "AAPL", "SELL", 10, 150.0, "tech")
        history = pm.get_trade_history()
        sell = next(t for t in history if t["action"] == "SELL")
        assert sell["pnl"] < 0.0

    def test_is_meme_flag_persisted(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "GME", "BUY", 5, 20.0, "meme_momentum", is_meme=True)
        history = pm.get_trade_history()
        assert history[0]["is_meme"] is True

    def test_multiple_strategies_recorded_independently(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY", 5,  150.0, "technical_trend")
        pm.record_trade("ord-2", "TSLA", "BUY", 3,  200.0, "meme_momentum")
        history = pm.get_trade_history()
        assert len(history) == 2
        strategies = {t["strategy"] for t in history}
        assert strategies == {"technical_trend", "meme_momentum"}


# ---------------------------------------------------------------------------
# Strategy exposure
# ---------------------------------------------------------------------------

class TestStrategyExposure:

    def test_exposure_zero_before_any_trades(self, pm):
        assert pm.get_strategy_exposure("technical_trend") == 0.0

    def test_exposure_increases_on_buy(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY", 10, 150.0, "technical_trend")
        exposure = pm.get_strategy_exposure("technical_trend")
        assert exposure == pytest.approx(1500.0, abs=0.01)

    def test_exposure_decreases_on_sell(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY",  10, 150.0, "technical_trend")
        pm.record_trade("ord-2", "AAPL", "SELL",  5, 150.0, "technical_trend")
        exposure = pm.get_strategy_exposure("technical_trend")
        assert exposure == pytest.approx(750.0, abs=0.01)

    def test_exposure_never_negative(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY",  5, 150.0, "technical_trend")
        pm.record_trade("ord-2", "AAPL", "SELL", 10, 150.0, "technical_trend")
        assert pm.get_strategy_exposure("technical_trend") >= 0.0

    def test_different_strategies_have_independent_exposure(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY", 10, 150.0, "technical_trend")
        pm.record_trade("ord-2", "TSLA", "BUY",  5, 200.0, "meme_momentum")
        assert pm.get_strategy_exposure("technical_trend") == pytest.approx(1500.0)
        assert pm.get_strategy_exposure("meme_momentum")  == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Position value
# ---------------------------------------------------------------------------

class TestPositionValue:

    def test_position_value_zero_when_no_position(self, pm, broker):
        pm.sync()
        assert pm.get_position_value("AAPL") == 0.0

    def test_position_value_after_buy(self, pm, broker):
        broker.market_buy("AAPL", 10)    # 10 * 150 = 1500
        pm.sync()
        assert pm.get_position_value("AAPL") == pytest.approx(1500.0, abs=1.0)


# ---------------------------------------------------------------------------
# History queries
# ---------------------------------------------------------------------------

class TestHistoryQueries:

    def test_trade_history_respects_limit(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        for i in range(10):
            pm.record_trade(f"ord-{i}", "AAPL", "BUY", 1, 150.0, "tech")
        history = pm.get_trade_history(limit=5)
        assert len(history) == 5

    def test_equity_curve_populated_after_sync(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.sync()
        pm.sync()
        curve = pm.get_equity_curve()
        assert len(curve) >= 2
        for point in curve:
            assert "timestamp" in point
            assert "portfolio_value" in point

    def test_strategy_stats_returned_after_trades(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY",  10, 100.0, "technical_trend")
        pm.record_trade("ord-2", "AAPL", "SELL", 10, 120.0, "technical_trend")
        stats = pm.get_strategy_stats()
        tech = next((s for s in stats if s["strategy"] == "technical_trend"), None)
        assert tech is not None
        assert tech["total_trades"] == 2
        assert tech["winning_trades"] == 1   # the SELL was profitable
        assert tech["win_rate"] == pytest.approx(50.0, abs=0.1)

    def test_trade_history_most_recent_first(self, pm, db_url, monkeypatch):
        monkeypatch.setattr("core.portfolio.DATABASE_URL", db_url)
        pm.record_trade("ord-1", "AAPL", "BUY", 1, 100.0, "tech")
        pm.record_trade("ord-2", "TSLA", "BUY", 1, 200.0, "tech")
        history = pm.get_trade_history()
        # Most recent first
        assert history[0]["symbol"] == "TSLA"
        assert history[1]["symbol"] == "AAPL"
