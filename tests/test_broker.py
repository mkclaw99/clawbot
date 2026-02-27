"""
Tests for core/broker.py — LocalPaperBroker.

All tests use an in-memory (temp) SQLite database and mock _get_price
so no network calls are made.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch

import core.broker as broker_module
from core.broker import Broker, AccountInfo, Position, OrderResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FAKE_PRICES = {"AAPL": 150.0, "TSLA": 200.0, "GME": 20.0, "NVDA": 500.0}


def fake_price(symbol: str) -> float:
    return FAKE_PRICES.get(symbol, 100.0)


@pytest.fixture(autouse=True)
def patch_price():
    """Globally replace _get_price with deterministic values for every test."""
    with patch.object(broker_module, "_get_price", side_effect=fake_price):
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
def broker_b(db_url, monkeypatch):
    """Second broker in the SAME db but different portfolio_id — for isolation tests."""
    monkeypatch.setattr("core.broker.DATABASE_URL", db_url)
    monkeypatch.setattr("core.broker.STARTING_CAPITAL", 100_000.0)
    return Broker(portfolio_id="other")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestBrokerInit:

    def test_account_created_with_starting_capital(self, broker):
        acct = broker.get_account()
        assert acct.portfolio_value == 100_000.0
        assert acct.cash == 100_000.0

    def test_no_positions_on_init(self, broker):
        assert broker.get_positions() == []

    def test_multiple_init_calls_do_not_duplicate_account(self, db_url, monkeypatch):
        monkeypatch.setattr("core.broker.DATABASE_URL", db_url)
        monkeypatch.setattr("core.broker.STARTING_CAPITAL", 100_000.0)
        b1 = Broker(portfolio_id="dup")
        b2 = Broker(portfolio_id="dup")   # second init should be a no-op
        assert b2.get_account().cash == 100_000.0

    def test_schema_migration_runs_without_error(self, db_url, monkeypatch):
        """Migration on a fresh DB should not raise."""
        monkeypatch.setattr("core.broker.DATABASE_URL", db_url)
        monkeypatch.setattr("core.broker.STARTING_CAPITAL", 100_000.0)
        b = Broker(portfolio_id="migrate_test")
        assert b.get_account().portfolio_value == 100_000.0


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class TestBrokerAccount:

    def test_portfolio_value_equals_cash_plus_equity(self, broker):
        broker.market_buy("AAPL", 10)   # $1500 worth
        acct = broker.get_account()
        assert abs(acct.portfolio_value - (acct.cash + acct.equity)) < 0.01

    def test_equity_reflects_positions(self, broker):
        broker.market_buy("AAPL", 10)   # 10 * 150 = $1500
        acct = broker.get_account()
        assert abs(acct.equity - 1500.0) < 0.01

    def test_cash_decreases_after_buy(self, broker):
        broker.market_buy("AAPL", 10)
        acct = broker.get_account()
        assert abs(acct.cash - (100_000.0 - 10 * 150.0)) < 0.01

    def test_cash_increases_after_sell(self, broker):
        broker.market_buy("AAPL", 10)
        cash_after_buy = broker.get_account().cash
        broker.market_sell("AAPL", 5)
        cash_after_sell = broker.get_account().cash
        assert cash_after_sell > cash_after_buy
        assert abs(cash_after_sell - (cash_after_buy + 5 * 150.0)) < 0.01

    def test_buying_power_equals_cash(self, broker):
        acct = broker.get_account()
        assert acct.buying_power == acct.cash


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

class TestBrokerPositions:

    def test_position_created_after_buy(self, broker):
        broker.market_buy("AAPL", 5)
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 5

    def test_position_removed_after_full_sell(self, broker):
        broker.market_buy("AAPL", 5)
        broker.market_sell("AAPL", 5)
        assert broker.get_positions() == []

    def test_position_qty_reduced_after_partial_sell(self, broker):
        broker.market_buy("AAPL", 10)
        broker.market_sell("AAPL", 4)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.qty == 6.0

    def test_avg_entry_price_on_single_buy(self, broker):
        broker.market_buy("AAPL", 10)
        pos = broker.get_position("AAPL")
        assert abs(pos.avg_entry_price - 150.0) < 0.01

    def test_avg_entry_price_weighted_on_multiple_buys(self, broker):
        # Buy 10 @ 150, then price changes and buy 10 more
        broker.market_buy("AAPL", 10)       # 10 @ 150
        FAKE_PRICES["AAPL"] = 200.0
        broker.market_buy("AAPL", 10)       # 10 @ 200
        pos = broker.get_position("AAPL")
        expected_avg = (10 * 150.0 + 10 * 200.0) / 20
        assert abs(pos.avg_entry_price - expected_avg) < 0.01
        FAKE_PRICES["AAPL"] = 150.0         # restore

    def test_unrealized_pnl_positive_when_price_up(self, broker):
        broker.market_buy("AAPL", 10)       # bought @ 150, current = 150
        pos = broker.get_position("AAPL")
        assert pos.unrealized_pnl == 0.0    # same price

    def test_get_position_returns_none_when_missing(self, broker):
        assert broker.get_position("NONEXISTENT") is None

    def test_multiple_positions(self, broker):
        broker.market_buy("AAPL", 5)
        broker.market_buy("TSLA", 3)
        positions = broker.get_positions()
        symbols = {p.symbol for p in positions}
        assert symbols == {"AAPL", "TSLA"}


# ---------------------------------------------------------------------------
# market_buy
# ---------------------------------------------------------------------------

class TestMarketBuy:

    def test_successful_buy_returns_filled_status(self, broker):
        result = broker.market_buy("AAPL", 10)
        assert result.status == "filled"
        assert result.action == "BUY"
        assert result.filled_price == 150.0
        assert result.qty == 10

    def test_buy_returns_unique_order_id(self, broker):
        r1 = broker.market_buy("AAPL", 1)
        r2 = broker.market_buy("TSLA", 1)
        assert r1.order_id != r2.order_id

    def test_buy_rejected_when_insufficient_cash(self, broker):
        # Try to buy 1000 shares @ $150 = $150k > $100k
        result = broker.market_buy("AAPL", 1000)
        assert result.status == "rejected"
        # Cash must be unchanged
        assert broker.get_account().cash == 100_000.0

    def test_buy_rejected_when_qty_zero(self, broker):
        result = broker.market_buy("AAPL", 0)
        assert result.status == "rejected"

    def test_buy_error_when_price_unavailable(self, broker):
        with patch.object(broker_module, "_get_price", return_value=0.0):
            result = broker.market_buy("AAPL", 1)
        assert result.status == "error"

    def test_buy_order_id_contains_symbol(self, broker):
        result = broker.market_buy("AAPL", 1)
        assert "AAPL" in result.order_id


# ---------------------------------------------------------------------------
# market_sell
# ---------------------------------------------------------------------------

class TestMarketSell:

    def test_successful_sell_returns_filled_status(self, broker):
        broker.market_buy("AAPL", 10)
        result = broker.market_sell("AAPL", 5)
        assert result.status == "filled"
        assert result.action == "SELL"
        assert result.filled_price == 150.0

    def test_sell_rejected_when_no_position(self, broker):
        result = broker.market_sell("AAPL", 1)
        assert result.status == "rejected"

    def test_sell_rejected_when_overselling(self, broker):
        broker.market_buy("AAPL", 5)
        result = broker.market_sell("AAPL", 10)
        assert result.status == "rejected"
        # Position must be unchanged
        assert broker.get_position("AAPL").qty == 5.0

    def test_sell_rejected_when_qty_zero(self, broker):
        broker.market_buy("AAPL", 5)
        result = broker.market_sell("AAPL", 0)
        assert result.status == "rejected"


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

class TestClosePosition:

    def test_closes_full_position(self, broker):
        broker.market_buy("AAPL", 10)
        result = broker.close_position("AAPL")
        assert result is not None
        assert result.status == "filled"
        assert broker.get_position("AAPL") is None

    def test_returns_none_when_no_position(self, broker):
        result = broker.close_position("AAPL")
        assert result is None


# ---------------------------------------------------------------------------
# Portfolio isolation
# ---------------------------------------------------------------------------

class TestPortfolioIsolation:

    def test_two_portfolios_have_independent_cash(self, broker, broker_b):
        broker.market_buy("AAPL", 100)        # spend $15k from "test"
        assert broker_b.get_account().cash == 100_000.0   # "other" unaffected

    def test_two_portfolios_have_independent_positions(self, broker, broker_b):
        broker.market_buy("AAPL", 10)
        assert broker_b.get_position("AAPL") is None

    def test_sell_in_one_portfolio_does_not_affect_other(self, broker, broker_b):
        broker.market_buy("AAPL", 10)
        broker_b.market_buy("AAPL", 10)
        broker.market_sell("AAPL", 10)        # close in "test"
        pos_b = broker_b.get_position("AAPL")
        assert pos_b is not None
        assert pos_b.qty == 10.0

    def test_five_strategy_portfolios_all_independent(self, db_url, monkeypatch):
        monkeypatch.setattr("core.broker.DATABASE_URL", db_url)
        monkeypatch.setattr("core.broker.STARTING_CAPITAL", 100_000.0)
        strategies = ["meme_momentum", "technical_trend", "mean_reversion",
                      "options_flow", "macro_news"]
        brokers = [Broker(portfolio_id=pid) for pid in strategies]

        # Buy AAPL only in meme_momentum
        brokers[0].market_buy("AAPL", 10)

        for b in brokers[1:]:
            assert b.get_position("AAPL") is None
            assert b.get_account().cash == 100_000.0


# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------

class TestPriceCache:

    def test_get_latest_price_returns_correct_value(self, broker):
        assert broker.get_latest_price("AAPL") == 150.0
        assert broker.get_latest_price("TSLA") == 200.0

    def test_unknown_symbol_returns_fallback(self, broker):
        price = broker.get_latest_price("UNKNOWNSYM")
        assert price == 100.0   # FAKE_PRICES default


# ---------------------------------------------------------------------------
# get_bars
# ---------------------------------------------------------------------------

class TestGetBars:

    def test_get_bars_returns_list(self, broker):
        import pandas as pd
        import numpy as np
        mock_df = pd.DataFrame({
            "Open": [100.0], "High": [110.0], "Low": [90.0],
            "Close": [105.0], "Volume": [1_000_000.0],
        }, index=pd.to_datetime(["2024-01-01"]))
        with patch("yfinance.download", return_value=mock_df):
            bars = broker.get_bars("AAPL", limit=1)
        assert isinstance(bars, list)
        assert len(bars) == 1
        assert set(bars[0].keys()) == {"t", "o", "h", "l", "c", "v"}

    def test_get_bars_returns_empty_on_error(self, broker):
        with patch("yfinance.download", side_effect=RuntimeError("fail")):
            bars = broker.get_bars("AAPL")
        assert bars == []
