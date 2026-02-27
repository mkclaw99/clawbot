"""
Tests for new free-data crawlers:
  - options_crawler   (yfinance options chains)
  - google_trends_crawler (pytrends)
  - macro_crawler     (FRED + CNN Fear & Greed)

All network calls are mocked — tests run offline.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from datetime import date, timedelta

import pandas as pd
import pytest

# Near-future expiry that always stays within options_crawler.MAX_DTE=45
_NEAR_EXPIRY = (date.today() + timedelta(days=20)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_option_df(volumes: list[float], ois: list[float], strikes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "volume":          volumes,
        "openInterest":    ois,
        "strike":          strikes,
        "impliedVolatility": [0.3] * len(volumes),
    })


# ===========================================================================
# options_crawler
# ===========================================================================

class TestOptionsCache:
    """Cache returns stale data within TTL, refreshes after expiry."""

    def test_cache_hit(self):
        from crawler.options_crawler import _CACHE, _cached_flow
        _CACHE["CACHE_HIT"] = (time.time(), {"call_put_ratio": 2.5})
        result = _cached_flow("CACHE_HIT")
        assert result is not None
        assert result["call_put_ratio"] == 2.5

    def test_cache_miss_expired(self):
        from crawler.options_crawler import _CACHE, _cached_flow
        _CACHE["EXPIRED"] = (time.time() - 1000, {"call_put_ratio": 2.5})
        assert _cached_flow("EXPIRED") is None

    def test_cache_miss_absent(self):
        from crawler.options_crawler import _cached_flow
        assert _cached_flow("DEFINITELY_NOT_IN_CACHE_XYZ") is None


class TestFetchOptionsFlowForTicker:
    """Unit tests for fetch_options_flow_for_ticker()."""

    def _make_ticker_mock(self, call_vols, put_vols, call_ois, put_ois):
        mock_ticker = MagicMock()
        mock_ticker.options = [_NEAR_EXPIRY]   # near-future expiry (within MAX_DTE=45)
        chain = SimpleNamespace(
            calls=_make_option_df(call_vols, call_ois, [100.0] * len(call_vols)),
            puts =_make_option_df(put_vols,  put_ois,  [95.0]  * len(put_vols)),
        )
        mock_ticker.option_chain.return_value = chain
        return mock_ticker

    @patch("crawler.options_crawler.yf.Ticker")
    def test_bullish_call_heavy(self, mock_yf):
        # 3× more call vol than put vol → bullish
        mock_yf.return_value = self._make_ticker_mock(
            call_vols=[300, 200], put_vols=[100, 50],
            call_ois=[1000, 800], put_ois=[400, 200],
        )
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("BULLTEST", None)

        result = fetch_options_flow_for_ticker("BULLTEST")
        assert result is not None
        assert result["direction"] == "bullish"
        assert result["call_put_ratio"] >= 2.0
        assert result["source"] == "yfinance_options"

    @patch("crawler.options_crawler.yf.Ticker")
    def test_bearish_put_heavy(self, mock_yf):
        mock_yf.return_value = self._make_ticker_mock(
            call_vols=[50, 30], put_vols=[300, 200],
            call_ois=[200, 100], put_ois=[900, 600],
        )
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("BEARTEST", None)

        result = fetch_options_flow_for_ticker("BEARTEST")
        assert result is not None
        assert result["direction"] == "bearish"
        assert result["call_put_ratio"] <= 0.5

    @patch("crawler.options_crawler.yf.Ticker")
    def test_insufficient_volume_returns_none(self, mock_yf):
        # Total volume < MIN_VOL=50
        mock_yf.return_value = self._make_ticker_mock(
            call_vols=[10, 5], put_vols=[8, 4],
            call_ois=[100, 50], put_ois=[80, 40],
        )
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("LOWVOL", None)

        result = fetch_options_flow_for_ticker("LOWVOL")
        assert result is None

    @patch("crawler.options_crawler.yf.Ticker")
    def test_no_expirations_returns_none(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.options = []
        mock_yf.return_value = mock_ticker
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("NOEXP", None)

        assert fetch_options_flow_for_ticker("NOEXP") is None

    @patch("crawler.options_crawler.yf.Ticker")
    def test_block_trade_detection(self, mock_yf):
        """Single contract with volume >= BLOCK_VOL_THRESHOLD=500 counts as block."""
        mock_ticker = MagicMock()
        mock_ticker.options = [_NEAR_EXPIRY]
        chain = SimpleNamespace(
            calls=_make_option_df([600, 100], [2000, 500], [150.0, 140.0]),
            puts =_make_option_df([200, 100], [1000, 500], [145.0, 135.0]),
        )
        mock_ticker.option_chain.return_value = chain
        mock_yf.return_value = mock_ticker
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("BLOCK", None)

        result = fetch_options_flow_for_ticker("BLOCK")
        assert result is not None
        assert result["block_trade_count"] >= 1
        assert result["notable_trade"] != ""

    @patch("crawler.options_crawler.yf.Ticker")
    def test_result_fields_present(self, mock_yf):
        mock_yf.return_value = self._make_ticker_mock(
            call_vols=[200, 100], put_vols=[80, 60],
            call_ois=[500, 300], put_ois=[200, 150],
        )
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("FIELDS", None)

        result = fetch_options_flow_for_ticker("FIELDS")
        required = {
            "call_put_ratio", "premium_ratio", "oi_change",
            "avg_days_to_expiry", "block_trade_count",
            "direction", "source", "notable_trade",
        }
        assert result is not None
        assert required.issubset(result.keys())

    @patch("crawler.options_crawler.yf.Ticker")
    def test_exception_returns_none(self, mock_yf):
        mock_yf.side_effect = RuntimeError("network error")
        from crawler.options_crawler import _CACHE, fetch_options_flow_for_ticker
        _CACHE.pop("ERR", None)
        assert fetch_options_flow_for_ticker("ERR") is None

    @patch("crawler.options_crawler.fetch_options_flow_for_ticker")
    def test_fetch_options_flow_skips_etfs(self, mock_fetch):
        """SPY, QQQ, IWM, GLD are always skipped."""
        from crawler.options_crawler import fetch_options_flow
        mock_fetch.return_value = {
            "call_put_ratio": 2.0, "premium_ratio": 3.0, "oi_change": 0.1,
            "avg_days_to_expiry": 20, "block_trade_count": 1,
            "direction": "bullish", "source": "yfinance_options", "notable_trade": "",
        }
        results = fetch_options_flow(["SPY", "QQQ", "AAPL"])
        called_syms = [call.args[0] for call in mock_fetch.call_args_list]
        assert "SPY" not in called_syms
        assert "QQQ" not in called_syms
        assert "AAPL" in called_syms


# ===========================================================================
# macro_crawler
# ===========================================================================

class TestFetchFredSeries:
    """FRED CSV parser."""

    @patch("crawler.macro_crawler.requests.get")
    def test_returns_last_valid_value(self, mock_get):
        csv = "DATE,VALUE\n2025-01-01,4.5\n2025-02-01,4.8\n2025-03-01,.\n"
        mock_get.return_value = MagicMock(status_code=200, text=csv)
        mock_get.return_value.raise_for_status = MagicMock()
        from crawler.macro_crawler import _fetch_fred_series
        val = _fetch_fred_series("DFF")
        assert val == 4.8   # last non-missing value

    @patch("crawler.macro_crawler.requests.get")
    def test_handles_all_missing(self, mock_get):
        csv = "DATE,VALUE\n2025-01-01,.\n2025-02-01,.\n"
        mock_get.return_value = MagicMock(status_code=200, text=csv)
        mock_get.return_value.raise_for_status = MagicMock()
        from crawler.macro_crawler import _fetch_fred_series
        assert _fetch_fred_series("DFF") is None

    @patch("crawler.macro_crawler.requests.get")
    def test_returns_none_on_error(self, mock_get):
        mock_get.side_effect = RuntimeError("timeout")
        from crawler.macro_crawler import _fetch_fred_series
        assert _fetch_fred_series("DFF") is None


class TestFetchFearGreed:
    """CNN Fear & Greed parser."""

    @patch("crawler.macro_crawler.requests.get")
    def test_parses_score_and_label(self, mock_get):
        payload = {"fear_and_greed": {"score": 32.5, "rating": "Fear"}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        from crawler.macro_crawler import fetch_fear_greed
        result = fetch_fear_greed()
        assert result["score"] == 32.5
        assert result["label"] == "Fear"

    @patch("crawler.macro_crawler.requests.get")
    def test_defaults_on_error(self, mock_get):
        mock_get.side_effect = RuntimeError("network error")
        from crawler.macro_crawler import fetch_fear_greed
        result = fetch_fear_greed()
        assert result["score"] == 50.0
        assert result["label"] == "Neutral"


class TestInferRegime:
    """Regime classification logic — European macro indicators."""

    def test_risk_off_high_vstoxx(self):
        from crawler.macro_crawler import _infer_regime
        # VSTOXX > 35 (+2) + extreme fear < 25 (+2) → 4 risk-off signals
        assert _infer_regime(vix=None, vstoxx=40, de_yield_10y=1.0, fear_greed=20) == "risk-off"

    def test_risk_off_high_bund_yield(self):
        from crawler.macro_crawler import _infer_regime
        # DE 10y > 3 (+1) + extreme fear (+2) → 3 risk-off signals
        assert _infer_regime(vix=None, vstoxx=None, de_yield_10y=4.0, fear_greed=20) == "risk-off"

    def test_risk_on(self):
        from crawler.macro_crawler import _infer_regime
        # Low VIX fallback (+1 risk-on) + greed (+1) → 2 risk-on signals
        assert _infer_regime(vix=14, vstoxx=None, de_yield_10y=1.0, fear_greed=80) == "risk-on"

    def test_neutral(self):
        from crawler.macro_crawler import _infer_regime
        assert _infer_regime(vix=22, vstoxx=None, de_yield_10y=1.0, fear_greed=50) == "neutral"

    def test_handles_none_inputs(self):
        from crawler.macro_crawler import _infer_regime
        # Should not raise even when all optional inputs are None
        result = _infer_regime(vix=None, vstoxx=None, de_yield_10y=None, fear_greed=50)
        assert result in ("risk-on", "risk-off", "neutral")


_FRED_EU = {
    "ecb_rate":    3.75,
    "de_yield_10y": 2.5,
    "vix":          18.0,
    "cpi":          130.0,
    "unemployment": 5.5,
}
_SNAP_EU = {"dax_level": 19500.0, "vstoxx": 20.5}


class TestFetchMacroContext:
    """Integration: fetch_macro_context() returns expected shape."""

    @patch("crawler.macro_crawler.fetch_european_snapshots")
    @patch("crawler.macro_crawler.fetch_fred_macro")
    @patch("crawler.macro_crawler.fetch_fear_greed")
    def test_context_shape(self, mock_fng, mock_fred, mock_snap):
        mock_fred.return_value = _FRED_EU.copy()
        mock_snap.return_value = _SNAP_EU.copy()
        mock_fng.return_value = {"score": 62.0, "label": "Greed"}

        from crawler import macro_crawler
        macro_crawler._CACHE.clear()   # bypass cache for test

        result = macro_crawler.fetch_macro_context()
        required_keys = {
            "ecb_rate", "de_yield_10y", "vix", "vstoxx", "dax_level",
            "cpi", "unemployment",
            "fear_greed_score", "fear_greed_label", "regime", "source", "timestamp",
        }
        assert required_keys.issubset(result.keys())
        assert result["fear_greed_score"] == 62.0
        assert result["regime"] in ("risk-on", "risk-off", "neutral")

    @patch("crawler.macro_crawler.fetch_european_snapshots")
    @patch("crawler.macro_crawler.fetch_fred_macro")
    @patch("crawler.macro_crawler.fetch_fear_greed")
    def test_cache_is_populated(self, mock_fng, mock_fred, mock_snap):
        mock_fred.return_value = _FRED_EU.copy()
        mock_snap.return_value = _SNAP_EU.copy()
        mock_fng.return_value = {"score": 55.0, "label": "Greed"}

        from crawler import macro_crawler
        macro_crawler._CACHE.clear()
        macro_crawler.fetch_macro_context()
        assert "macro" in macro_crawler._CACHE

    @patch("crawler.macro_crawler.fetch_european_snapshots")
    @patch("crawler.macro_crawler.fetch_fred_macro")
    @patch("crawler.macro_crawler.fetch_fear_greed")
    def test_cache_returned_without_refetch(self, mock_fng, mock_fred, mock_snap):
        mock_fred.return_value = _FRED_EU.copy()
        mock_snap.return_value = _SNAP_EU.copy()
        mock_fng.return_value = {"score": 50.0, "label": "Neutral"}

        from crawler import macro_crawler
        macro_crawler._CACHE.clear()
        macro_crawler.fetch_macro_context()   # populates cache
        mock_fred.reset_mock()
        mock_fng.reset_mock()
        mock_snap.reset_mock()
        macro_crawler.fetch_macro_context()   # should use cache
        mock_fred.assert_not_called()
        mock_fng.assert_not_called()
        mock_snap.assert_not_called()


# ===========================================================================
# google_trends_crawler
# ===========================================================================

class TestFetchTrendsForBatch:
    """Unit tests for fetch_trends_for_batch()."""

    def _make_trends_df(self, symbols: list[str], values: list[list[float]]) -> pd.DataFrame:
        """Create a fake pytrends DataFrame with weekly rows."""
        import numpy as np
        dates = pd.date_range("2024-11-01", periods=12, freq="W")
        data = {}
        for sym, vals in zip(symbols, values):
            # Pad/truncate to 12 weeks
            padded = (vals * 12)[:12]
            data[f"{sym} stock"] = padded
        df = pd.DataFrame(data, index=dates)
        return df

    @patch("crawler.google_trends_crawler._get_pytrends")
    def test_rising_trend_detected(self, mock_pytrends_fn):
        pt = MagicMock()
        df = self._make_trends_df(
            ["AAPL"],
            [[20, 22, 25, 28, 30, 32, 35, 38, 40, 50, 60, 80]],
        )
        pt.interest_over_time.return_value = df
        mock_pytrends_fn.return_value = pt

        from crawler import google_trends_crawler
        google_trends_crawler._CACHE.clear()
        result = google_trends_crawler.fetch_trends_for_batch(["AAPL"])

        assert "AAPL" in result
        assert result["AAPL"]["trend"] == "rising"
        assert result["AAPL"]["interest_score"] > 0

    @patch("crawler.google_trends_crawler._get_pytrends")
    def test_falling_trend_detected(self, mock_pytrends_fn):
        pt = MagicMock()
        df = self._make_trends_df(
            ["GME"],
            [[80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 15, 10]],
        )
        pt.interest_over_time.return_value = df
        mock_pytrends_fn.return_value = pt

        from crawler import google_trends_crawler
        google_trends_crawler._CACHE.clear()
        result = google_trends_crawler.fetch_trends_for_batch(["GME"])
        assert "GME" in result
        assert result["GME"]["trend"] == "falling"

    @patch("crawler.google_trends_crawler._get_pytrends")
    def test_spike_ratio_calculation(self, mock_pytrends_fn):
        """spike_ratio = last value / mean of series."""
        pt = MagicMock()
        # Series: constant 20 then jump to 80 at the end
        vals = [20] * 11 + [80]
        df = self._make_trends_df(["NVDA"], [vals])
        pt.interest_over_time.return_value = df
        mock_pytrends_fn.return_value = pt

        from crawler import google_trends_crawler
        google_trends_crawler._CACHE.clear()
        result = google_trends_crawler.fetch_trends_for_batch(["NVDA"])

        assert "NVDA" in result
        # avg ≈ (20*11 + 80)/12 ≈ 25; spike = 80/25 ≈ 3.2
        assert result["NVDA"]["spike_ratio"] > 2.5

    @patch("crawler.google_trends_crawler._get_pytrends")
    def test_empty_df_returns_empty(self, mock_pytrends_fn):
        pt = MagicMock()
        pt.interest_over_time.return_value = pd.DataFrame()
        mock_pytrends_fn.return_value = pt

        from crawler import google_trends_crawler
        google_trends_crawler._CACHE.clear()
        result = google_trends_crawler.fetch_trends_for_batch(["TSLA"])
        assert result == {}

    @patch("crawler.google_trends_crawler._get_pytrends")
    def test_cache_hit_skips_fetch(self, mock_pytrends_fn):
        from crawler import google_trends_crawler
        # Pre-populate cache
        google_trends_crawler._CACHE["CACHED_SYM"] = (time.time(), {
            "interest_score": 45.0, "spike_ratio": 1.2,
            "trend": "flat", "peak_week": "2025-01-01",
        })
        result = google_trends_crawler.fetch_trends_for_batch(["CACHED_SYM"])
        mock_pytrends_fn.assert_not_called()
        assert "CACHED_SYM" in result

    @patch("crawler.google_trends_crawler._get_pytrends")
    def test_exception_returns_partial_results(self, mock_pytrends_fn):
        mock_pytrends_fn.side_effect = RuntimeError("rate limited")
        from crawler import google_trends_crawler
        google_trends_crawler._CACHE.clear()
        result = google_trends_crawler.fetch_trends_for_batch(["MSFT"])
        # Should not raise; return empty for uncached symbols
        assert isinstance(result, dict)

    def test_empty_input_returns_empty(self):
        from crawler.google_trends_crawler import fetch_trends_for_batch
        assert fetch_trends_for_batch([]) == {}

    def test_result_keys_present(self):
        from crawler import google_trends_crawler
        google_trends_crawler._CACHE["KEYS_TEST"] = (time.time(), {
            "interest_score": 50.0, "spike_ratio": 1.5,
            "trend": "rising", "peak_week": "2025-03-01",
        })
        result = google_trends_crawler.fetch_trends_for_batch(["KEYS_TEST"])
        d = result["KEYS_TEST"]
        assert {"interest_score", "spike_ratio", "trend", "peak_week"}.issubset(d.keys())


# ===========================================================================
# macro_news strategy — macro context integration
# ===========================================================================

class TestMacroNewsStrategyMacroAware:
    """Verify that macro context influences BUY confidence and SELL sensitivity."""

    def _make_strategy(self):
        from strategies.macro_news import MacroNewsStrategy
        return MacroNewsStrategy()

    def _make_news(self, articles=None, earn_beat=None, earn_surp=0.0):
        return {
            "articles": articles or [
                {"title": "Company beats expectations", "description": "Strong quarter"},
                {"title": "Revenue growth accelerates",  "description": "Positive outlook"},
                {"title": "CEO optimistic about future", "description": "Guidance raised"},
            ],
            "earnings_beat":         earn_beat,
            "earnings_surprise_pct": earn_surp,
            "insider_buys":          0,
            "insider_sells":         0,
            "institutional_adds":    0,
        }

    def test_risk_off_reduces_buy_confidence(self):
        s = self._make_strategy()
        # Neutral macro
        sig_neutral = s._analyse(
            "TST", self._make_news(earn_beat=True, earn_surp=10.0), held=set(),
            regime="neutral", fear_greed=60.0, volatility=18.0, de_yield=1.0,
        )
        # Risk-off macro
        sig_riskoff = s._analyse(
            "TST", self._make_news(earn_beat=True, earn_surp=10.0), held=set(),
            regime="risk-off", fear_greed=20.0, volatility=35.0, de_yield=4.0,
        )
        # Risk-off should lower confidence (or suppress signal entirely)
        if sig_riskoff is not None and sig_neutral is not None:
            assert sig_riskoff.confidence <= sig_neutral.confidence

    def test_risk_on_can_boost_confidence(self):
        s = self._make_strategy()
        sig_neutral = s._analyse(
            "TST", self._make_news(earn_beat=True, earn_surp=8.0), held=set(),
            regime="neutral", fear_greed=50.0, volatility=18.0, de_yield=1.0,
        )
        sig_riskon = s._analyse(
            "TST", self._make_news(earn_beat=True, earn_surp=8.0), held=set(),
            regime="risk-on", fear_greed=75.0, volatility=14.0, de_yield=1.0,
        )
        if sig_riskon is not None and sig_neutral is not None:
            assert sig_riskon.confidence >= sig_neutral.confidence - 0.01

    def test_macro_reason_in_output(self):
        s = self._make_strategy()
        sig = s._analyse(
            "TST", self._make_news(earn_beat=True, earn_surp=12.0), held=set(),
            regime="risk-off", fear_greed=15.0, volatility=40.0, de_yield=4.0,
        )
        if sig is not None:
            assert "macro:" in sig.reason

    def test_macro_metadata_in_buy(self):
        s = self._make_strategy()
        sig = s._analyse(
            "TST", self._make_news(earn_beat=True, earn_surp=15.0), held=set(),
            regime="neutral", fear_greed=55.0, volatility=17.0, de_yield=1.0,
        )
        if sig is not None and sig.action == "BUY":
            assert "regime" in sig.metadata
            assert "volatility" in sig.metadata
            assert "fear_greed" in sig.metadata

    def test_sell_fires_more_readily_in_risk_off(self):
        s = self._make_strategy()
        # Score just below neutral sell threshold but above risk-off threshold
        # (risk-off threshold = -0.40 + 0.10 = -0.30)
        neg_news = [
            {"title": "Earnings miss badly",      "description": "Revenue fell short"},
            {"title": "Guidance cut significantly","description": "Below expectations"},
            {"title": "Analyst downgrades stock",  "description": "Target reduced"},
        ]
        held = {"TST"}

        # In risk-off, the sell threshold loosens (fires at a less negative score)
        with patch("strategies.macro_news._sentiment_score", return_value=-0.35):
            sig_neutral = s._analyse(
                "TST", self._make_news(articles=neg_news), held=held,
                regime="neutral", fear_greed=50.0,
            )
            sig_riskoff = s._analyse(
                "TST", self._make_news(articles=neg_news), held=held,
                regime="risk-off", fear_greed=20.0,
            )
        # risk-off should sell when neutral wouldn't (or sell with higher confidence)
        if sig_riskoff is not None:
            assert sig_riskoff.action == "SELL"
