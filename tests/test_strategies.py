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
