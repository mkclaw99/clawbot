"""
Tests for clawbot_cli.py — all subcommand functions.

Dependencies are mocked so no network calls, DB writes, or filesystem I/O occur.
Each CLI command is called directly (not via subprocess); stdout is captured and
parsed as JSON, and sys.exit() is caught with pytest.raises(SystemExit).
"""
import sys
import io
import json
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import clawbot_cli
from clawbot_cli import (
    cmd_portfolio, cmd_safety_status, cmd_macro,
    cmd_scan, cmd_trade, cmd_audit, cmd_run_cycle,
    STRATEGY_PORTFOLIOS,
)
from strategies.base import SignalResult
from core.broker import OrderResult, Position, AccountInfo
from core.portfolio import PortfolioState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(func, args=None):
    """
    Invoke a CLI command function, capture its stdout, parse JSON, return
    (data: dict, exit_code: int).  Always catches SystemExit from _out/_err.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc_info:
            func(args)
    data = json.loads(buf.getvalue())
    return data, exc_info.value.code


def ns(**kwargs):
    """Build a SimpleNamespace (stand-in for argparse.Namespace)."""
    return SimpleNamespace(**kwargs)


# --- Object factories -------------------------------------------------------

def fake_state(portfolio_value=100_000.0, cash=100_000.0,
               equity=0.0, total_pnl=0.0, positions=None):
    """Return a MagicMock PortfolioState."""
    state = MagicMock(spec=PortfolioState)
    state.portfolio_value = portfolio_value
    state.cash = cash
    state.equity = equity
    state.total_pnl = total_pnl
    state.start_value = 100_000.0
    state.positions = positions or []
    return state


def fake_position(symbol="AAPL", qty=10, avg_entry_price=150.0,
                  current_price=155.0, market_value=1550.0,
                  unrealized_pnl=50.0, unrealized_pnl_pct=3.33):
    pos = MagicMock(spec=Position)
    pos.symbol = symbol
    pos.qty = qty
    pos.avg_entry_price = avg_entry_price
    pos.current_price = current_price
    pos.market_value = market_value
    pos.unrealized_pnl = unrealized_pnl
    pos.unrealized_pnl_pct = unrealized_pnl_pct
    return pos


def fake_order(symbol="AAPL", action="BUY", qty=10,
               filled_price=150.0, status="filled"):
    r = MagicMock(spec=OrderResult)
    r.order_id = f"ord-{symbol}-test"
    r.symbol = symbol
    r.action = action
    r.qty = qty
    r.filled_price = filled_price
    r.status = status
    r.timestamp = "2024-01-01T00:00:00+00:00"
    return r


def fake_signal(symbol="AAPL", action="BUY", score=0.8, confidence=0.75,
                strategy="technical_trend", reason="test signal",
                is_meme=False, is_actionable=True):
    sig = MagicMock(spec=SignalResult)
    sig.symbol = symbol
    sig.action = action
    sig.score = score
    sig.confidence = confidence
    sig.strategy = strategy
    sig.reason = reason
    sig.is_meme = is_meme
    sig.is_actionable = is_actionable
    return sig


class FakeStrategy:
    """Lightweight strategy stub; avoids MagicMock.name collision."""
    def __init__(self, name: str, signals=None):
        self.name = name
        self._signals = signals or []

    def generate_signals(self, universe, context):
        return self._signals


def make_safety_mock(level="NORMAL", is_halted=False, size_mult=1.0,
                     allowed=True, reject_reason=""):
    safety = MagicMock()
    state = MagicMock()
    state.level.value = level
    state.triggered_at = None
    state.triggered_reason = None
    state.daily_start_value = 100_000.0
    state.peak_portfolio_value = 100_000.0
    safety.get_state.return_value = state
    safety.get_size_multiplier.return_value = size_mult
    safety.is_halted.return_value = is_halted
    safety.check_order.return_value = (allowed, reject_reason)
    safety.update_portfolio_value = MagicMock()
    return safety


def make_pm_mock(state=None):
    pm = MagicMock()
    pm.sync.return_value = state or fake_state()
    pm.get_position_value.return_value = 0.0
    pm.get_strategy_exposure.return_value = 0.0
    pm.record_trade = MagicMock()
    return pm


# Shorthand lists for patching all 5 strategy modules (outer → inner order
# matches @patch decorator stacking: bottom-most decorator = first arg)
_STRATEGY_PATCHES = [
    "strategies.meme_momentum.MemeMomentumStrategy",
    "strategies.technical_trend.TechnicalTrendStrategy",
    "strategies.mean_reversion.MeanReversionStrategy",
    "strategies.options_flow.OptionsFlowStrategy",
    "strategies.macro_news.MacroNewsStrategy",
]


def _mock_strategies_with_names(mocks, signals_map=None):
    """
    Given the 5 strategy mocks (in STRATEGY_PORTFOLIOS order) configure
    each to return a FakeStrategy with the correct name.
    signals_map: {portfolio_id: [SignalResult, ...]} — defaults to empty.
    """
    signals_map = signals_map or {}
    for mock, pid in zip(mocks, STRATEGY_PORTFOLIOS):
        sigs = signals_map.get(pid, [])
        mock.return_value = FakeStrategy(pid, sigs)


# ---------------------------------------------------------------------------
# TestPortfolioCmd
# ---------------------------------------------------------------------------

class TestPortfolioCmd:

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_returns_all_five_strategies(self, MockBroker, MockPM):
        MockPM.return_value = make_pm_mock()
        data, _ = run_cmd(cmd_portfolio)
        assert len(data["portfolios"]) == len(STRATEGY_PORTFOLIOS)
        assert {p["strategy"] for p in data["portfolios"]} == set(STRATEGY_PORTFOLIOS)

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_exit_code_is_zero(self, MockBroker, MockPM):
        MockPM.return_value = make_pm_mock()
        _, code = run_cmd(cmd_portfolio)
        assert code == 0

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_combined_value_is_sum_of_portfolios(self, MockBroker, MockPM):
        MockPM.return_value = make_pm_mock(fake_state(portfolio_value=99_000.0))
        data, _ = run_cmd(cmd_portfolio)
        expected = sum(p["portfolio_value"] for p in data["portfolios"])
        assert abs(data["combined_value"] - expected) < 0.01

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_combined_pnl_is_sum_of_pnls(self, MockBroker, MockPM):
        MockPM.return_value = make_pm_mock(fake_state(total_pnl=500.0))
        data, _ = run_cmd(cmd_portfolio)
        expected_pnl = sum(p["total_pnl"] for p in data["portfolios"])
        assert abs(data["combined_pnl"] - expected_pnl) < 0.01

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_each_portfolio_has_required_fields(self, MockBroker, MockPM):
        MockPM.return_value = make_pm_mock()
        data, _ = run_cmd(cmd_portfolio)
        required = {
            "strategy", "portfolio_value", "cash", "equity",
            "total_pnl", "total_pnl_pct", "position_count", "positions",
        }
        for p in data["portfolios"]:
            assert required.issubset(p.keys()), f"Missing keys in {p}"

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_timestamp_present(self, MockBroker, MockPM):
        MockPM.return_value = make_pm_mock()
        data, _ = run_cmd(cmd_portfolio)
        assert "timestamp" in data

    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_positions_listed_in_portfolio(self, MockBroker, MockPM):
        pos = fake_position("AAPL", qty=5)
        MockPM.return_value = make_pm_mock(
            fake_state(portfolio_value=100_750.0, equity=750.0, positions=[pos])
        )
        data, _ = run_cmd(cmd_portfolio)
        p = data["portfolios"][0]
        assert p["position_count"] == 1
        assert len(p["positions"]) == 1
        assert p["positions"][0]["symbol"] == "AAPL"
        assert p["positions"][0]["qty"] == 5


# ---------------------------------------------------------------------------
# TestSafetyStatusCmd
# ---------------------------------------------------------------------------

class TestSafetyStatusCmd:

    @patch("core.audit.read_audit_log", return_value=[])
    @patch("core.safety.SafetyLayer")
    def test_required_fields_present(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock()
        data, _ = run_cmd(cmd_safety_status)
        required = {
            "level", "is_halted", "size_multiplier",
            "recent_safety_events", "timestamp",
            "triggered_at", "triggered_reason",
            "daily_start_value", "peak_portfolio_value",
        }
        assert required.issubset(data.keys())

    @patch("core.audit.read_audit_log", return_value=[])
    @patch("core.safety.SafetyLayer")
    def test_normal_level_reflected(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock(level="NORMAL")
        data, _ = run_cmd(cmd_safety_status)
        assert data["level"] == "NORMAL"

    @patch("core.audit.read_audit_log", return_value=[])
    @patch("core.safety.SafetyLayer")
    def test_not_halted_reflected(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock(is_halted=False)
        data, _ = run_cmd(cmd_safety_status)
        assert data["is_halted"] is False

    @patch("core.audit.read_audit_log", return_value=[])
    @patch("core.safety.SafetyLayer")
    def test_halted_reflected(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock(is_halted=True)
        data, _ = run_cmd(cmd_safety_status)
        assert data["is_halted"] is True

    @patch("core.audit.read_audit_log", return_value=[
        {"event": "SAFETY", "level": "L1", "message": "drawdown", "_ts": "x"}
    ])
    @patch("core.safety.SafetyLayer")
    def test_recent_safety_events_included(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock()
        data, _ = run_cmd(cmd_safety_status)
        assert len(data["recent_safety_events"]) == 1
        assert data["recent_safety_events"][0]["level"] == "L1"

    @patch("core.audit.read_audit_log", return_value=[])
    @patch("core.safety.SafetyLayer")
    def test_size_multiplier_present(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock(size_mult=0.5)
        data, _ = run_cmd(cmd_safety_status)
        assert data["size_multiplier"] == 0.5

    @patch("core.audit.read_audit_log", return_value=[])
    @patch("core.safety.SafetyLayer")
    def test_exit_code_is_zero(self, MockSafety, _log):
        MockSafety.return_value = make_safety_mock()
        _, code = run_cmd(cmd_safety_status)
        assert code == 0


# ---------------------------------------------------------------------------
# TestMacroCmd
# ---------------------------------------------------------------------------

FAKE_MACRO = {
    "regime": "risk_on",
    "vix": 15.2,
    "vstoxx": 17.4,
    "dax_level": 19800.0,
    "fear_greed_score": 65,
    "ecb_rate": 3.75,
    "de_yield_10y": 2.5,
}


class TestMacroCmd:

    @patch("crawler.macro_crawler.fetch_macro_context", return_value=FAKE_MACRO)
    def test_forwards_macro_context(self, _mock):
        data, code = run_cmd(cmd_macro)
        assert code == 0
        assert data["regime"] == "risk_on"
        assert data["vix"] == 15.2

    @patch("crawler.macro_crawler.fetch_macro_context", return_value=FAKE_MACRO)
    def test_all_macro_fields_present(self, _mock):
        data, _ = run_cmd(cmd_macro)
        for key in FAKE_MACRO:
            assert key in data, f"Missing key: {key}"

    @patch("crawler.macro_crawler.fetch_macro_context", return_value=FAKE_MACRO)
    def test_exit_code_is_zero(self, _mock):
        _, code = run_cmd(cmd_macro)
        assert code == 0


# ---------------------------------------------------------------------------
# TestScanCmd
# ---------------------------------------------------------------------------

class TestScanCmd:

    def _base_patches(self):
        """Return a dict of patch targets → values for cmd_scan."""
        return {
            "core.broker.Broker": MagicMock(return_value=MagicMock()),
            "core.portfolio.PortfolioManager": MagicMock(
                return_value=make_pm_mock()
            ),
            "crawler.signal_aggregator.SignalAggregator": MagicMock(
                return_value=MagicMock(
                    refresh=MagicMock(
                        return_value={"universe": ["AAPL", "TSLA"],
                                      "macro": {"regime": "risk_on"}}
                    )
                )
            ),
        }

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_returns_signals_array_and_exits_zero(
        self, MockBroker, MockPM, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        MockPM.return_value = make_pm_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": [], "macro": {"regime": "risk_on"},
        }
        _mock_strategies_with_names([MockMeme, MockTech, MockMR, MockOF, MockMN])

        data, code = run_cmd(cmd_scan)
        assert "signals" in data
        assert isinstance(data["signals"], list)
        assert code == 0

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_required_top_level_fields(
        self, MockBroker, MockPM, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        MockPM.return_value = make_pm_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": [], "macro": {"regime": "unknown"},
        }
        _mock_strategies_with_names([MockMeme, MockTech, MockMR, MockOF, MockMN])

        data, _ = run_cmd(cmd_scan)
        required = {"signals", "signal_count", "portfolio_value",
                    "macro_regime", "timestamp"}
        assert required.issubset(data.keys())

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_deduplicates_keeping_highest_confidence(
        self, MockBroker, MockPM, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        MockPM.return_value = make_pm_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": ["AAPL"], "macro": {"regime": "risk_on"},
        }
        sig_low  = fake_signal("AAPL", confidence=0.6, strategy="meme_momentum")
        sig_high = fake_signal("AAPL", confidence=0.9, strategy="technical_trend")
        _mock_strategies_with_names(
            [MockMeme, MockTech, MockMR, MockOF, MockMN],
            signals_map={
                "meme_momentum":   [sig_low],
                "technical_trend": [sig_high],
            },
        )

        data, _ = run_cmd(cmd_scan)
        aapl = [s for s in data["signals"] if s["symbol"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl[0]["confidence"] == 0.9
        assert aapl[0]["strategy"] == "technical_trend"

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_signal_has_all_required_fields(
        self, MockBroker, MockPM, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        MockPM.return_value = make_pm_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": ["AAPL"], "macro": {"regime": "risk_on"},
        }
        sig = fake_signal("AAPL", action="BUY", score=0.8, confidence=0.75,
                          strategy="meme_momentum")
        _mock_strategies_with_names(
            [MockMeme, MockTech, MockMR, MockOF, MockMN],
            signals_map={"meme_momentum": [sig]},
        )

        data, _ = run_cmd(cmd_scan)
        assert len(data["signals"]) >= 1
        s = data["signals"][0]
        for field in ("symbol", "action", "score", "confidence",
                      "strategy", "reason", "is_meme", "is_actionable"):
            assert field in s, f"Missing field: {field}"

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_signal_count_matches_signals_array_length(
        self, MockBroker, MockPM, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        MockPM.return_value = make_pm_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": ["AAPL", "TSLA"], "macro": {},
        }
        _mock_strategies_with_names(
            [MockMeme, MockTech, MockMR, MockOF, MockMN],
            signals_map={
                "meme_momentum": [
                    fake_signal("AAPL"),
                    fake_signal("TSLA"),
                ],
            },
        )

        data, _ = run_cmd(cmd_scan)
        assert data["signal_count"] == len(data["signals"])


# ---------------------------------------------------------------------------
# TestTradeCmd
# ---------------------------------------------------------------------------

class TestTradeCmd:

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_buy_approved_filled(self, MockBroker, MockPM, MockSafety):
        MockBroker.return_value.get_latest_price.return_value = 150.0
        MockBroker.return_value.market_buy.return_value = fake_order("AAPL", "BUY", 5, 150.0)
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock(allowed=True)

        a = ns(symbol="AAPL", action="BUY", reason="test", qty=5, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 0
        assert data["allowed"] is True
        assert data["symbol"] == "AAPL"
        assert data["action"] == "BUY"
        assert data["status"] == "filled"

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_buy_order_id_in_output(self, MockBroker, MockPM, MockSafety):
        MockBroker.return_value.get_latest_price.return_value = 150.0
        MockBroker.return_value.market_buy.return_value = fake_order()
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock()

        a = ns(symbol="AAPL", action="BUY", reason="test", qty=5, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 0
        assert "order_id" in data
        assert data["order_id"] == "ord-AAPL-test"

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_buy_rejected_by_safety_returns_allowed_false(
        self, MockBroker, MockPM, MockSafety
    ):
        MockBroker.return_value.get_latest_price.return_value = 150.0
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock(
            allowed=False, reject_reason="position limit exceeded"
        )

        a = ns(symbol="AAPL", action="BUY", reason="test", qty=5, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 0          # not an error — just a business-logic rejection
        assert data["allowed"] is False
        assert "rejection_reason" in data
        assert "position limit exceeded" in data["rejection_reason"]

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_buy_auto_qty_when_none_given(self, MockBroker, MockPM, MockSafety):
        MockBroker.return_value.get_latest_price.return_value = 100.0
        MockBroker.return_value.market_buy.return_value = fake_order()
        MockPM.return_value = make_pm_mock(fake_state(portfolio_value=100_000.0))
        MockSafety.return_value = make_safety_mock()

        # qty=None → auto-calc: max(1, int(100_000 * 0.02 * 1.0 / 100)) = 20
        a = ns(symbol="AAPL", action="BUY", reason="auto", qty=None, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 0
        assert data["allowed"] is True
        assert data["qty"] >= 1

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_sell_approved_filled(self, MockBroker, MockPM, MockSafety):
        pos = fake_position("AAPL", qty=10, current_price=160.0)
        MockBroker.return_value.get_position.return_value = pos
        MockBroker.return_value.market_sell.return_value = fake_order(
            "AAPL", "SELL", 10, 160.0
        )
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock()

        a = ns(symbol="AAPL", action="SELL", reason="take profit",
               qty=None, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 0
        assert data["allowed"] is True
        assert data["action"] == "SELL"
        assert data["filled_price"] == 160.0

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_sell_no_position_exits_one_with_error(
        self, MockBroker, MockPM, MockSafety
    ):
        MockBroker.return_value.get_position.return_value = None
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock()

        a = ns(symbol="AAPL", action="SELL", reason="close",
               qty=None, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 1
        assert "error" in data

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_invalid_action_exits_one_with_error(
        self, MockBroker, MockPM, MockSafety
    ):
        MockBroker.return_value.get_latest_price.return_value = 150.0
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock()

        a = ns(symbol="AAPL", action="HOLD", reason="test",
               qty=None, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 1
        assert "error" in data

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_halted_system_exits_one(self, MockBroker, MockPM, MockSafety):
        MockBroker.return_value.get_latest_price.return_value = 150.0
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock(is_halted=True, level="LEVEL_3")

        a = ns(symbol="AAPL", action="BUY", reason="try",
               qty=5, portfolio="manual")
        data, code = run_cmd(cmd_trade, a)
        assert code == 1
        assert "error" in data

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_buy_calls_record_trade(self, MockBroker, MockPM, MockSafety):
        MockBroker.return_value.get_latest_price.return_value = 150.0
        MockBroker.return_value.market_buy.return_value = fake_order()
        pm = make_pm_mock()
        MockPM.return_value = pm
        MockSafety.return_value = make_safety_mock()

        a = ns(symbol="AAPL", action="BUY", reason="test", qty=5, portfolio="manual")
        run_cmd(cmd_trade, a)
        pm.record_trade.assert_called_once()

    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_sell_calls_record_trade(self, MockBroker, MockPM, MockSafety):
        pos = fake_position("AAPL", qty=5)
        MockBroker.return_value.get_position.return_value = pos
        MockBroker.return_value.market_sell.return_value = fake_order(
            "AAPL", "SELL", 5, 155.0
        )
        pm = make_pm_mock()
        MockPM.return_value = pm
        MockSafety.return_value = make_safety_mock()

        a = ns(symbol="AAPL", action="SELL", reason="test",
               qty=None, portfolio="manual")
        run_cmd(cmd_trade, a)
        pm.record_trade.assert_called_once()


# ---------------------------------------------------------------------------
# TestAuditCmd
# ---------------------------------------------------------------------------

SAMPLE_ENTRIES = [
    {"event": "ORDER",  "action": "BUY", "symbol": "AAPL",
     "qty": 10, "price": 150.0, "_ts": "2024-01-01T00:00:00+00:00"},
    {"event": "SAFETY", "level": "INFO", "message": "init",
     "_ts": "2024-01-01T00:01:00+00:00"},
]


class TestAuditCmd:

    @patch("core.audit.read_audit_log", return_value=SAMPLE_ENTRIES)
    def test_returns_entries_and_count(self, _log):
        a = ns(tail=20)
        data, code = run_cmd(cmd_audit, a)
        assert code == 0
        assert "entries" in data
        assert "count" in data
        assert isinstance(data["entries"], list)

    @patch("core.audit.read_audit_log", return_value=SAMPLE_ENTRIES)
    def test_count_matches_entries_length(self, _log):
        a = ns(tail=20)
        data, _ = run_cmd(cmd_audit, a)
        assert data["count"] == len(data["entries"])

    @patch("core.audit.read_audit_log", return_value=[])
    def test_empty_entries_count_zero(self, _log):
        a = ns(tail=20)
        data, _ = run_cmd(cmd_audit, a)
        assert data["count"] == 0
        assert data["entries"] == []

    @patch("core.audit.read_audit_log")
    def test_tail_argument_passed_as_limit(self, mock_log):
        mock_log.return_value = []
        a = ns(tail=7)
        run_cmd(cmd_audit, a)
        mock_log.assert_called_once_with(limit=7)

    @patch("core.audit.read_audit_log", return_value=SAMPLE_ENTRIES)
    def test_entries_preserve_event_field(self, _log):
        a = ns(tail=20)
        data, _ = run_cmd(cmd_audit, a)
        events = {e["event"] for e in data["entries"]}
        assert "ORDER" in events
        assert "SAFETY" in events

    @patch("core.audit.read_audit_log", return_value=SAMPLE_ENTRIES)
    def test_exit_code_is_zero(self, _log):
        _, code = run_cmd(cmd_audit, ns(tail=20))
        assert code == 0


# ---------------------------------------------------------------------------
# TestRunCycleCmd
# ---------------------------------------------------------------------------

def _setup_run_cycle(MockBroker, MockPM, MockSafety, MockAgg, strategy_mocks,
                     is_halted=False, signals_map=None):
    """Wire up all mocks needed by cmd_run_cycle."""
    MockPM.return_value = make_pm_mock()
    MockSafety.return_value = make_safety_mock(
        level="NORMAL", is_halted=is_halted
    )
    MockAgg.return_value.refresh.return_value = {
        "universe": [], "macro": {"regime": "risk_on"},
    }
    _mock_strategies_with_names(strategy_mocks, signals_map)


class TestRunCycleCmd:

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_output_has_required_keys(
        self, MockBroker, MockPM, MockSafety, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        _setup_run_cycle(MockBroker, MockPM, MockSafety, MockAgg,
                         [MockMeme, MockTech, MockMR, MockOF, MockMN])
        data, code = run_cmd(cmd_run_cycle)
        assert code == 0
        required = {
            "cycle_timestamp", "is_market_hours", "circuit_breaker",
            "combined_value", "combined_pnl", "portfolios",
            "signals_generated", "trades_executed", "trades_blocked",
            "errors", "macro_regime",
        }
        assert required.issubset(data.keys())

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_portfolios_dict_has_all_five_strategies(
        self, MockBroker, MockPM, MockSafety, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        _setup_run_cycle(MockBroker, MockPM, MockSafety, MockAgg,
                         [MockMeme, MockTech, MockMR, MockOF, MockMN])
        data, _ = run_cmd(cmd_run_cycle)
        assert set(data["portfolios"].keys()) == set(STRATEGY_PORTFOLIOS)

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_portfolio_entry_has_required_fields(
        self, MockBroker, MockPM, MockSafety, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        _setup_run_cycle(MockBroker, MockPM, MockSafety, MockAgg,
                         [MockMeme, MockTech, MockMR, MockOF, MockMN])
        data, _ = run_cmd(cmd_run_cycle)
        for _pid, portfolio_data in data["portfolios"].items():
            assert "portfolio_value" in portfolio_data
            assert "total_pnl" in portfolio_data
            assert "position_count" in portfolio_data

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_halted_produces_no_executed_trades(
        self, MockBroker, MockPM, MockSafety, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        # Even with high-confidence signals, halted system must not execute
        high_sig = fake_signal("AAPL", confidence=0.95)
        _setup_run_cycle(
            MockBroker, MockPM, MockSafety, MockAgg,
            [MockMeme, MockTech, MockMR, MockOF, MockMN],
            is_halted=True,
            signals_map={pid: [high_sig] for pid in STRATEGY_PORTFOLIOS},
        )
        data, code = run_cmd(cmd_run_cycle)
        assert code == 0
        assert len(data["trades_executed"]) == 0

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_macro_regime_forwarded_from_aggregator(
        self, MockBroker, MockPM, MockSafety, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": [], "macro": {"regime": "risk_off"},
        }
        _mock_strategies_with_names([MockMeme, MockTech, MockMR, MockOF, MockMN])
        data, _ = run_cmd(cmd_run_cycle)
        assert data["macro_regime"] == "risk_off"

    @patch(_STRATEGY_PATCHES[4])
    @patch(_STRATEGY_PATCHES[3])
    @patch(_STRATEGY_PATCHES[2])
    @patch(_STRATEGY_PATCHES[1])
    @patch(_STRATEGY_PATCHES[0])
    @patch("crawler.signal_aggregator.SignalAggregator")
    @patch("core.safety.SafetyLayer")
    @patch("core.portfolio.PortfolioManager")
    @patch("core.broker.Broker")
    def test_strategy_errors_captured_not_raised(
        self, MockBroker, MockPM, MockSafety, MockAgg,
        MockMeme, MockTech, MockMR, MockOF, MockMN,
    ):
        """A crashing strategy should appear in errors[], not crash the cycle."""
        MockPM.return_value = make_pm_mock()
        MockSafety.return_value = make_safety_mock()
        MockAgg.return_value.refresh.return_value = {
            "universe": [], "macro": {},
        }
        # meme_momentum raises; others return empty
        bad_strat = FakeStrategy("meme_momentum", [])
        bad_strat.generate_signals = MagicMock(side_effect=RuntimeError("boom"))
        MockMeme.return_value = bad_strat
        for MockS, name in zip([MockTech, MockMR, MockOF, MockMN],
                                STRATEGY_PORTFOLIOS[1:]):
            MockS.return_value = FakeStrategy(name, [])

        data, code = run_cmd(cmd_run_cycle)
        assert code == 0
        assert any("meme_momentum" in e for e in data["errors"])
