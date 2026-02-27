"""
Tests for the self-improvement optimizer.
Run with: pytest tests/test_optimizer.py -v
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from core.optimizer import (
    PerformanceAnalyzer, ParameterTuner, StrategyAllocator,
    OptimizerState, StrategyPerf, PARAM_BOUNDS,
    WEIGHT_MIN, WEIGHT_MAX, WEIGHT_MAX_CHANGE_PER_CYCLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_strategy_mock(name: str, **attrs) -> MagicMock:
    """Create a mock strategy with the given attributes."""
    s = MagicMock()
    s.name = name
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def make_perf(strategy: str, win_rate: float = 55.0, sharpe: float = 0.8,
              total_trades: int = 20, is_improving: bool = True) -> StrategyPerf:
    return StrategyPerf(
        strategy=strategy,
        total_trades=total_trades,
        win_rate=win_rate,
        recent_win_rate=win_rate + (5 if is_improving else -5),
        prior_win_rate=win_rate - (5 if is_improving else -5),
        sharpe=sharpe,
        total_pnl=500.0,
        is_improving=is_improving,
    )


# ---------------------------------------------------------------------------
# PerformanceAnalyzer
# ---------------------------------------------------------------------------

class TestPerformanceAnalyzer:

    def test_sharpe_positive_returns(self):
        pnls = [10, 15, 8, 12, 9, 11, 14, 7, 13, 10]
        sharpe = PerformanceAnalyzer._sharpe(pnls)
        assert sharpe > 0

    def test_sharpe_negative_returns(self):
        pnls = [-10, -5, -8, -12, -3]
        sharpe = PerformanceAnalyzer._sharpe(pnls)
        assert sharpe < 0

    def test_sharpe_too_few_returns_nan(self):
        sharpe = PerformanceAnalyzer._sharpe([10, 15])   # need at least 3
        assert math.isnan(sharpe)

    def test_sharpe_constant_returns_nan(self):
        # Zero std → NaN
        sharpe = PerformanceAnalyzer._sharpe([10, 10, 10, 10, 10])
        assert math.isnan(sharpe)


# ---------------------------------------------------------------------------
# ParameterTuner
# ---------------------------------------------------------------------------

class TestParameterTuner:

    @pytest.fixture
    def state(self):
        return OptimizerState()

    @pytest.fixture
    def strategy(self):
        return make_strategy_mock(
            "technical_trend",
            RSI_OVERSOLD=35.0,
            RSI_OVERBOUGHT=70.0,
        )

    @pytest.fixture
    def tuner(self, strategy, state):
        return ParameterTuner([strategy], state)

    def test_param_stays_within_lower_bound(self, tuner, strategy, state):
        """Even if we try to go lower than min, it should be clamped."""
        # Force RSI_OVERSOLD to its minimum
        strategy.RSI_OVERSOLD = 25.0   # already at min
        perf = make_perf("technical_trend", win_rate=40.0, is_improving=False)
        tuner.tune({"technical_trend": perf})
        # Should not go below 25.0 (the minimum bound)
        assert strategy.RSI_OVERSOLD >= 25.0

    def test_param_stays_within_upper_bound(self, tuner, strategy, state):
        """Even if we try to go above max, it should be clamped."""
        strategy.RSI_OVERBOUGHT = 80.0   # already at max
        perf = make_perf("technical_trend", win_rate=80.0, is_improving=True)
        tuner.tune({"technical_trend": perf})
        assert strategy.RSI_OVERBOUGHT <= 80.0

    def test_no_tuning_with_insufficient_trades(self, tuner, strategy, state):
        """Strategies with fewer than MIN_TRADES_FOR_TUNING should not be touched."""
        original_oversold = strategy.RSI_OVERSOLD
        perf = make_perf("technical_trend", total_trades=5)   # below MIN (10)
        tuner.tune({"technical_trend": perf})
        assert strategy.RSI_OVERSOLD == original_oversold

    def test_max_one_param_changed_per_cycle(self, tuner, strategy, state):
        """Only one parameter should change per strategy per cycle."""
        original_oversold   = strategy.RSI_OVERSOLD
        original_overbought = strategy.RSI_OVERBOUGHT
        perf = make_perf("technical_trend", total_trades=20)
        tuner.tune({"technical_trend": perf})

        changed = (strategy.RSI_OVERSOLD != original_oversold,
                   strategy.RSI_OVERBOUGHT != original_overbought)
        # At most 1 should have changed
        assert sum(changed) <= 1

    def test_cooling_period_blocks_change(self, tuner, strategy, state):
        """A param changed recently should not be changed again."""
        # Mark RSI_OVERSOLD as changed 1 hour ago (cooling = 24h)
        recent_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        state.last_changed["technical_trend"] = {"RSI_OVERSOLD": recent_ts,
                                                   "RSI_OVERBOUGHT": recent_ts}
        original = strategy.RSI_OVERSOLD
        perf = make_perf("technical_trend", total_trades=20)
        tuner.tune({"technical_trend": perf})
        # Both params are in cooling — nothing should change
        assert strategy.RSI_OVERSOLD == original
        assert strategy.RSI_OVERBOUGHT == 70.0

    def test_revert_after_win_rate_drop(self, tuner, strategy, state):
        """If win rate dropped 10pp+ since last change, revert the param."""
        original_value = 37.0
        strategy.RSI_OVERSOLD = 39.0   # changed from 37 previously

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
        state.last_changed.setdefault("technical_trend", {})["RSI_OVERSOLD"] = old_ts
        state.win_rate_at_change.setdefault("technical_trend", {})["RSI_OVERSOLD"] = 70.0
        state.value_at_change.setdefault("technical_trend", {})["RSI_OVERSOLD"] = original_value

        # Win rate is now 55% (dropped from 70%)
        perf = make_perf("technical_trend", win_rate=55.0, total_trades=20)
        changes = tuner.tune({"technical_trend": perf})

        revert_changes = [c for c in changes if c.get("type") == "revert"]
        assert len(revert_changes) == 1
        assert strategy.RSI_OVERSOLD == pytest.approx(original_value, abs=0.01)

    def test_bounds_dict_is_not_safety_constants(self):
        """PARAM_BOUNDS must not reference safety_constants.py content."""
        # Safety constants: MAX_POSITION_FRACTION, CIRCUIT_BREAKER_*, MAX_LEVERAGE etc.
        forbidden = {
            "MAX_POSITION_FRACTION", "MAX_STRATEGY_ALLOCATION",
            "CIRCUIT_BREAKER_L1_DAILY_LOSS", "CIRCUIT_BREAKER_L2_DAILY_LOSS",
            "CIRCUIT_BREAKER_L3_DRAWDOWN", "MAX_LEVERAGE", "MAX_ORDERS_PER_DAY",
            "SHORT_SELLING", "MIN_STOCK_PRICE",
        }
        all_params = set()
        for strategy_params in PARAM_BOUNDS.values():
            all_params.update(strategy_params.keys())
        intersection = all_params & forbidden
        assert not intersection, f"PARAM_BOUNDS contains safety constant names: {intersection}"


# ---------------------------------------------------------------------------
# StrategyAllocator
# ---------------------------------------------------------------------------

class TestStrategyAllocator:

    STRATEGY_NAMES = ["meme_momentum", "technical_trend", "mean_reversion",
                      "options_flow", "macro_news"]

    @pytest.fixture
    def state(self):
        return OptimizerState()

    @pytest.fixture
    def allocator(self, state):
        return StrategyAllocator(self.STRATEGY_NAMES, state)

    def test_weights_sum_to_one(self, allocator):
        perfs = {n: make_perf(n, sharpe=0.5 + i * 0.1) for i, n in enumerate(self.STRATEGY_NAMES)}
        weights = allocator.rebalance(perfs)
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_all_weights_within_bounds(self, allocator):
        perfs = {n: make_perf(n, sharpe=abs(i - 2) * 0.5) for i, n in enumerate(self.STRATEGY_NAMES)}
        weights = allocator.rebalance(perfs)
        for name, w in weights.items():
            assert WEIGHT_MIN <= w <= WEIGHT_MAX, f"{name} weight {w} out of bounds"

    def test_better_sharpe_gets_more_weight(self, allocator):
        """Strategy with 3× Sharpe of another should get more weight."""
        perfs = {n: make_perf(n, sharpe=0.1) for n in self.STRATEGY_NAMES}
        perfs["technical_trend"] = make_perf("technical_trend", sharpe=2.0)  # much better
        weights = allocator.rebalance(perfs)
        # technical_trend should have the highest weight
        assert weights["technical_trend"] == max(weights.values())

    def test_weight_change_capped_per_cycle(self, allocator, state):
        """No single strategy's weight should change by more than WEIGHT_MAX_CHANGE_PER_CYCLE."""
        # Set initial equal weights
        equal = 1.0 / len(self.STRATEGY_NAMES)
        state.weights = {n: equal for n in self.STRATEGY_NAMES}

        # Extreme Sharpe disparity
        perfs = {n: make_perf(n, sharpe=0.01) for n in self.STRATEGY_NAMES}
        perfs["macro_news"] = make_perf("macro_news", sharpe=100.0)

        weights = allocator.rebalance(perfs)
        for name in self.STRATEGY_NAMES:
            change = abs(weights[name] - equal)
            assert change <= WEIGHT_MAX_CHANGE_PER_CYCLE + 0.001, \
                f"{name} changed by {change:.4f}, cap is {WEIGHT_MAX_CHANGE_PER_CYCLE}"

    def test_insufficient_data_gets_equal_weight(self, allocator):
        """Strategies with <MIN_TRADES_FOR_TUNING should get equal weight, not zero."""
        perfs = {n: make_perf(n, total_trades=5) for n in self.STRATEGY_NAMES}  # all below min
        weights = allocator.rebalance(perfs)
        # All should be roughly equal (within bounds)
        values = list(weights.values())
        assert max(values) - min(values) < 0.02


# ---------------------------------------------------------------------------
# Integration: PARAM_BOUNDS completeness
# ---------------------------------------------------------------------------

class TestParamBoundsIntegrity:

    def test_all_strategies_have_bounds(self):
        expected = {"meme_momentum", "technical_trend", "mean_reversion",
                    "options_flow", "macro_news"}
        assert set(PARAM_BOUNDS.keys()) == expected

    def test_all_bounds_are_valid_tuples(self):
        for strategy, params in PARAM_BOUNDS.items():
            for param, bounds in params.items():
                lo, hi, step = bounds
                assert lo < hi, f"{strategy}.{param}: lo >= hi ({lo} >= {hi})"
                assert step > 0, f"{strategy}.{param}: step must be positive"
                assert step <= (hi - lo), f"{strategy}.{param}: step larger than range"

    def test_bounds_do_not_include_safety_fields(self):
        """Sanity check that we're not accidentally adjusting safety-critical values."""
        safety_fields = {"MAX_LEVERAGE", "SHORT_SELLING", "OPTIONS_WRITING",
                         "MAX_ORDERS_PER_DAY", "MIN_STOCK_PRICE",
                         "CIRCUIT_BREAKER_L1_DAILY_LOSS"}
        all_params = {p for params in PARAM_BOUNDS.values() for p in params}
        assert not all_params & safety_fields
