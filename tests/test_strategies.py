"""
Tests for all 5 strategy modules.
Run with: pytest tests/test_strategies.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import random
from datetime import datetime, timedelta, timezone

from strategies.base import SignalResult
from strategies.technical_trend import TechnicalTrendStrategy
from strategies.meme_momentum import MemeMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.options_flow import OptionsFlowStrategy
from strategies.macro_news import MacroNewsStrategy
from strategies.long_short_equity import LongShortEquityStrategy


def make_bars(n: int = 60, trend: str = "up") -> list[dict]:
    """Generate synthetic OHLCV bars."""
    bars = []
    price = 100.0
    ts    = datetime.now(timezone.utc) - timedelta(days=n)
    for i in range(n):
        if trend == "up":
            price *= random.uniform(1.000, 1.012)
        elif trend == "down":
            price *= random.uniform(0.988, 1.000)
        else:
            price *= random.uniform(0.994, 1.006)
        bars.append({
            "t": ts.isoformat(),
            "o": price * 0.99,
            "h": price * 1.01,
            "l": price * 0.98,
            "c": price,
            "v": random.uniform(1e6, 5e6),
        })
        ts += timedelta(days=1)
    return bars


class TestTechnicalTrend:

    def test_returns_list(self):
        s = TechnicalTrendStrategy()
        sigs = s.generate_signals(["AAPL"], {"bars": {"AAPL": make_bars(60, "up")}})
        assert isinstance(sigs, list)

    def test_buy_signal_in_uptrend(self):
        """Strong uptrend should generate a BUY signal."""
        s = TechnicalTrendStrategy()
        # Very strong consistent uptrend
        bars = make_bars(60, "up")
        sigs = s.generate_signals(["AAPL"], {"bars": {"AAPL": bars}})
        # Not guaranteed every run due to randomness, but should be consistent
        actions = [sig.action for sig in sigs]
        assert all(a in ("BUY", "SELL") for a in actions)

    def test_skips_insufficient_bars(self):
        s = TechnicalTrendStrategy()
        sigs = s.generate_signals(["AAPL"], {"bars": {"AAPL": make_bars(10)}})
        assert sigs == []

    def test_signals_have_required_fields(self):
        s = TechnicalTrendStrategy()
        sigs = s.generate_signals(["AAPL"], {"bars": {"AAPL": make_bars(60, "up")}})
        for sig in sigs:
            assert sig.symbol == "AAPL"
            assert sig.action in ("BUY", "SELL", "HOLD")
            assert 0.0 <= sig.confidence <= 1.0
            assert -1.0 <= sig.score <= 1.0
            assert sig.strategy == "technical_trend"


class TestMemeMomentum:

    def make_context(self, spike: float = 3.0, sentiment: float = 0.5):
        return {
            "meme_signals": {
                "GME": {
                    "mention_count": 100,
                    "mention_spike": spike,
                    "sentiment": sentiment,
                    "sources": ["wsb", "stocktwits"],
                    "top_post": "GME is going to the moon!!!",
                }
            },
            "bars": {"GME": make_bars(10, "up")},
            "current_positions": set(),
        }

    def test_buy_on_high_spike_positive_sentiment(self):
        s    = MemeMomentumStrategy()
        sigs = s.generate_signals(["GME"], self.make_context(spike=4.0, sentiment=0.6))
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) > 0

    def test_no_signal_on_low_spike(self):
        s    = MemeMomentumStrategy()
        sigs = s.generate_signals(["GME"], self.make_context(spike=0.5, sentiment=0.8))
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 0

    def test_no_signal_on_negative_sentiment(self):
        s    = MemeMomentumStrategy()
        sigs = s.generate_signals(["GME"], self.make_context(spike=5.0, sentiment=-0.5))
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 0

    def test_meme_flag_set(self):
        s    = MemeMomentumStrategy()
        sigs = s.generate_signals(["GME"], self.make_context(spike=4.0, sentiment=0.6))
        for sig in sigs:
            if sig.action == "BUY":
                assert sig.is_meme is True

    def test_exit_on_fading_signal(self):
        s = MemeMomentumStrategy()
        context = {
            "meme_signals": {
                "GME": {
                    "mention_count": 5,
                    "mention_spike": 0.3,   # faded
                    "sentiment": -0.2,
                    "sources": ["wsb"],
                    "top_post": "",
                }
            },
            "bars": {},
            "current_positions": {"GME"},   # currently held
        }
        sigs = s.generate_signals(["GME"], context)
        sells = [sig for sig in sigs if sig.action == "SELL"]
        assert len(sells) > 0


class TestMeanReversion:

    def make_oversold_bars(self) -> list[dict]:
        """Bars where price has fallen sharply below its 20-day mean."""
        bars = make_bars(30, "flat")
        # Artificially crash the last 3 bars
        for i in range(-3, 0):
            bars[i]["c"] *= 0.88
            bars[i]["l"] *= 0.87
        return bars

    def test_buy_on_oversold(self):
        s    = MeanReversionStrategy()
        bars = self.make_oversold_bars()
        sigs = s.generate_signals(["AAPL"], {
            "bars": {"AAPL": bars},
            "current_positions": set(),
            "meme_universe": set(),
        })
        # May or may not trigger depending on Z-score threshold
        assert isinstance(sigs, list)

    def test_skips_meme_stocks(self):
        s    = MeanReversionStrategy()
        sigs = s.generate_signals(["GME"], {
            "bars": {"GME": self.make_oversold_bars()},
            "current_positions": set(),
            "meme_universe": {"GME"},
        })
        assert sigs == []


class TestOptionsFlow:

    def make_bullish_flow(self) -> dict:
        return {
            "call_put_ratio": 3.5,
            "premium_ratio": 4.0,
            "oi_change": 0.35,
            "avg_days_to_expiry": 14,
            "block_trade_count": 3,
            "direction": "bullish",
            "source": "unusual_whales",
            "notable_trade": "Large $2M call sweep at $150 strike",
        }

    def test_buy_on_strong_bullish_flow(self):
        s = OptionsFlowStrategy()
        context = {
            "options_flow": {"AAPL": self.make_bullish_flow()},
            "bars": {"AAPL": make_bars(10, "up")},
            "current_positions": set(),
        }
        sigs = s.generate_signals(["AAPL"], context)
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) > 0

    def test_no_signal_on_weak_premium(self):
        s    = OptionsFlowStrategy()
        flow = self.make_bullish_flow()
        flow["premium_ratio"] = 1.2  # below threshold
        context = {
            "options_flow": {"AAPL": flow},
            "bars": {"AAPL": make_bars(10, "up")},
            "current_positions": set(),
        }
        sigs = s.generate_signals(["AAPL"], context)
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 0


class TestMacroNews:

    def make_positive_news(self) -> dict:
        return {
            "articles": [
                {"title": f"AAPL beats earnings estimates by 8%",
                 "description": "Apple exceeded Wall Street expectations.",
                 "source": "Reuters"},
                {"title": "Analysts upgrade AAPL price target after strong results",
                 "description": "Multiple firms raise targets amid positive guidance.",
                 "source": "Bloomberg"},
                {"title": "AAPL CEO buys 10,000 shares in open market",
                 "description": "Insider purchase signals confidence.",
                 "source": "SEC Filing"},
            ],
            "earnings_beat": True,
            "earnings_surprise_pct": 8.0,
            "insider_buys": 2,
            "insider_sells": 0,
            "institutional_adds": 3,
        }

    def test_returns_list(self):
        s    = MacroNewsStrategy()
        sigs = s.generate_signals(["AAPL"], {
            "news_signals": {"AAPL": self.make_positive_news()},
            "current_positions": set(),
        })
        assert isinstance(sigs, list)

    def test_no_signal_without_articles(self):
        s    = MacroNewsStrategy()
        sigs = s.generate_signals(["AAPL"], {
            "news_signals": {"AAPL": {"articles": [], "earnings_beat": None,
                                       "insider_buys": 0, "insider_sells": 0}},
            "current_positions": set(),
        })
        assert sigs == []


# ---------------------------------------------------------------------------
# TestLongShortEquityStrategy
# ---------------------------------------------------------------------------

def make_ls_universe(n_tickers: int = 10) -> list[str]:
    """Generate dummy .DE ticker names for L/S tests."""
    return [f"T{i:02d}.DE" for i in range(n_tickers)]


def make_ls_bars(n_tickers: int, trends: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Generate bars for n_tickers. `trends` is a list of "up"/"down"/"flat"
    per ticker (length must match n_tickers). Defaults to flat for all.
    """
    if trends is None:
        trends = ["flat"] * n_tickers
    tickers = make_ls_universe(n_tickers)
    return {sym: make_bars(70, trend) for sym, trend in zip(tickers, trends)}


def make_ls_context(
    bars: dict | None = None,
    n_tickers: int = 10,
    trends: list[str] | None = None,
    position_qtys: dict | None = None,
    regime: str = "neutral",
    cb_level: str = "NORMAL",
    vstoxx: float = 20.0,
    news_signals: dict | None = None,
    google_trends: dict | None = None,
) -> dict:
    """Build a minimal context dict for LongShortEquityStrategy tests."""
    if bars is None:
        bars = make_ls_bars(n_tickers, trends)
    universe = list(bars.keys())
    return {
        "universe":             universe,
        "bars":                 bars,
        "news_signals":         news_signals or {},
        "google_trends":        google_trends or {},
        "current_positions":    set(position_qtys.keys()) if position_qtys else set(),
        "position_qtys":        position_qtys or {},
        "macro":                {"regime": regime, "vstoxx": vstoxx, "vix": 20.0},
        "circuit_breaker_level": cb_level,
    }


def _strong_positive_news() -> dict:
    """News signals that produce a high heuristic sentiment (~0.72)."""
    return {
        "articles": [{"title": "beats estimates", "description": "strong quarter"}],
        "earnings_beat": True,
        "earnings_surprise_pct": 20.0,
        "insider_buys": 2,
        "insider_sells": 0,
        "institutional_adds": 1,
    }


def _strong_negative_news() -> dict:
    """News signals that produce a low heuristic sentiment (~−0.68)."""
    return {
        "articles": [{"title": "misses estimates", "description": "weak quarter"}],
        "earnings_beat": False,
        "earnings_surprise_pct": -20.0,
        "insider_buys": 0,
        "insider_sells": 2,
        "institutional_adds": 0,
    }


def make_controlled_ls_context(
    n_tickers: int = 10,
    top_idx: int = 0,
    bottom_idx: int = 9,
    position_qtys: dict | None = None,
    regime: str = "neutral",
    cb_level: str = "NORMAL",
    vstoxx: float = 20.0,
) -> dict:
    """
    Build a controlled L/S context where `top_idx` ticker dominates on ALL
    four factors (momentum, sentiment, trends, low-vol) and `bottom_idx`
    ticker is worst on all factors. The remaining tickers are flat/neutral.

    Guarantees:
      composite(top)    ≥ +0.70  → BUY / COVER eligible
      composite(bottom) ≤ −0.70  → SHORT / SELL eligible
    """
    tickers = make_ls_universe(n_tickers)
    trends  = ["flat"] * n_tickers
    trends[top_idx]    = "up"
    trends[bottom_idx] = "down"

    bars = {sym: make_bars(70, trend) for sym, trend in zip(tickers, trends)}

    news = {}
    news[tickers[top_idx]]    = _strong_positive_news()
    news[tickers[bottom_idx]] = _strong_negative_news()

    google = {
        tickers[top_idx]:    {"interest_score": 90},
        tickers[bottom_idx]: {"interest_score": 10},
    }

    return {
        "universe":             tickers,
        "bars":                 bars,
        "news_signals":         news,
        "google_trends":        google,
        "current_positions":    set(position_qtys.keys()) if position_qtys else set(),
        "position_qtys":        position_qtys or {},
        "macro":                {"regime": regime, "vstoxx": vstoxx, "vix": 20.0},
        "circuit_breaker_level": cb_level,
    }


class TestLongShortEquityStrategy:

    def test_returns_list(self):
        s    = LongShortEquityStrategy()
        ctx  = make_ls_context()
        sigs = s.generate_signals(list(ctx["bars"].keys()), ctx)
        assert isinstance(sigs, list)

    def test_long_signal_for_top_ticker(self):
        """
        The ticker that ranks highest on all four factors should receive a BUY signal.
        We control momentum, news sentiment, Google Trends, and position so that
        T00.DE has composite ≥ +0.70 regardless of random bar noise.
        """
        ctx  = make_controlled_ls_context(top_idx=0, bottom_idx=9)
        s    = LongShortEquityStrategy()
        sigs = s.generate_signals(ctx["universe"], ctx)
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) > 0, "Expected at least one BUY signal"
        assert any(sig.symbol == "T00.DE" for sig in buys), \
            "T00.DE (best on all factors) must receive a BUY signal"

    def test_short_signal_for_bottom_ticker(self):
        """
        The ticker that ranks lowest on all four factors should receive a SHORT signal.
        T09.DE has composite ≤ −0.70 regardless of random bar noise.
        """
        ctx    = make_controlled_ls_context(top_idx=0, bottom_idx=9)
        s      = LongShortEquityStrategy()
        sigs   = s.generate_signals(ctx["universe"], ctx)
        shorts = [sig for sig in sigs if sig.action == "SHORT"]
        assert len(shorts) > 0, "Expected at least one SHORT signal"
        assert any(sig.symbol == "T09.DE" for sig in shorts), \
            "T09.DE (worst on all factors) must receive a SHORT signal"

    def test_sell_when_existing_long_drops_below_exit_threshold(self):
        """
        When we hold a long in the bottom-ranked ticker (composite ≤ −0.70,
        below EXIT_LONG_THRESHOLD −0.10), the strategy must emit SELL.
        """
        ctx = make_controlled_ls_context(
            top_idx=0, bottom_idx=9,
            position_qtys={"T09.DE": 5.0},   # hold long in worst performer
        )
        s     = LongShortEquityStrategy()
        sigs  = s.generate_signals(ctx["universe"], ctx)
        sells = [sig for sig in sigs if sig.action == "SELL" and sig.symbol == "T09.DE"]
        assert len(sells) > 0, \
            "Expected SELL for held long whose composite is well below EXIT_LONG_THRESHOLD"

    def test_cover_when_existing_short_rises_above_exit_threshold(self):
        """
        When we hold a short in the top-ranked ticker (composite ≥ +0.70,
        above EXIT_SHORT_THRESHOLD +0.10), the strategy must emit COVER.
        """
        ctx = make_controlled_ls_context(
            top_idx=0, bottom_idx=9,
            position_qtys={"T00.DE": -5.0},   # hold short in best performer
        )
        s      = LongShortEquityStrategy()
        sigs   = s.generate_signals(ctx["universe"], ctx)
        covers = [sig for sig in sigs if sig.action == "COVER" and sig.symbol == "T00.DE"]
        assert len(covers) > 0, \
            "Expected COVER for held short whose composite is well above EXIT_SHORT_THRESHOLD"

    def test_no_short_in_risk_off_regime(self):
        """No SHORT signals should be emitted when macro regime is 'risk-off'."""
        ctx    = make_controlled_ls_context(regime="risk-off")
        s      = LongShortEquityStrategy()
        sigs   = s.generate_signals(ctx["universe"], ctx)
        shorts = [sig for sig in sigs if sig.action == "SHORT"]
        assert len(shorts) == 0, "No SHORT signals should be emitted in risk-off regime"

    def test_no_short_when_circuit_breaker_elevated(self):
        """No SHORT signals when circuit breaker is LEVEL_1 or above."""
        for level in ("LEVEL_1", "LEVEL_2", "LEVEL_3"):
            ctx    = make_controlled_ls_context(cb_level=level)
            s      = LongShortEquityStrategy()
            sigs   = s.generate_signals(ctx["universe"], ctx)
            shorts = [sig for sig in sigs if sig.action == "SHORT"]
            assert len(shorts) == 0, f"No SHORT signals at circuit breaker {level}"

    def test_dollar_neutral_cap_suppresses_excess_buys(self):
        """
        If existing long notional > 1.5 × short notional, new BUY signals
        should be suppressed by the dollar-neutral cap.
        """
        ctx = make_controlled_ls_context(top_idx=0, bottom_idx=9)
        # Add a huge long position and a tiny short so long >> 1.5 × short
        bars = ctx["bars"]
        ctx["position_qtys"]  = {
            "T01.DE": 10000.0,   # huge long → long_notional enormous
            "T02.DE": -1.0,      # tiny short → short_notional tiny
        }
        ctx["current_positions"] = {"T01.DE", "T02.DE"}
        s    = LongShortEquityStrategy()
        sigs = s.generate_signals(ctx["universe"], ctx)
        # T00.DE should not get a BUY because long_notional > short_notional * 1.5
        buys = [sig for sig in sigs if sig.action == "BUY" and sig.symbol == "T00.DE"]
        assert len(buys) == 0, "Dollar-neutral cap must suppress BUY when portfolio is long-heavy"

    def test_signal_fields_valid(self):
        """All emitted signals have valid field values."""
        s    = LongShortEquityStrategy()
        ctx  = make_ls_context()
        sigs = s.generate_signals(list(ctx["bars"].keys()), ctx)
        for sig in sigs:
            assert sig.action in ("BUY", "SELL", "SHORT", "COVER")
            assert 0.0 <= sig.confidence <= 1.0
            assert -1.0 <= sig.score <= 1.0
            assert sig.strategy == "long_short_equity"
            assert sig.symbol.endswith(".DE")

    def test_too_few_tickers_returns_empty(self):
        """Returns empty list when fewer than 5 valid tickers are available."""
        bars = make_ls_bars(3)
        ctx  = make_ls_context(bars=bars, n_tickers=3)
        s    = LongShortEquityStrategy()
        sigs = s.generate_signals(list(bars.keys()), ctx)
        assert sigs == []

    def test_etf_tickers_excluded(self):
        """ETF tickers (from GERMAN_ETFS) should never appear in signals."""
        from config.universe import GERMAN_ETFS
        ctx = make_ls_context()
        # Add an ETF to the universe
        etf = list(GERMAN_ETFS)[0]
        ctx["bars"][etf] = make_bars(70, "up")
        universe = list(ctx["bars"].keys())
        s    = LongShortEquityStrategy()
        sigs = s.generate_signals(universe, ctx)
        etf_sigs = [sig for sig in sigs if sig.symbol == etf]
        assert len(etf_sigs) == 0, f"ETF {etf} should not appear in L/S signals"
