"""
Strategy 1: Meme Stock Momentum
=================================
Monitors Reddit (r/wallstreetbets, r/stocks, r/investing), StockTwits
and news for social-media-driven momentum.

Edge: ride retail momentum surges EARLY — before mainstream coverage.
Risk: extremely volatile; hard position caps enforced in safety layer.

Signal pipeline:
  1. Crawler feeds mention counts + sentiment scores into context['meme_signals']
  2. This strategy filters by quality criteria and ranks tickers
  3. Buys only when social signal AND price confirm momentum
  4. Exits aggressively when momentum fades
"""
from __future__ import annotations

from loguru import logger
import pandas as pd

from strategies.base import BaseStrategy, SignalResult


class MemeMomentumStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "meme_momentum"

    @property
    def description(self) -> str:
        return (
            "Social-media-driven momentum. Monitors Reddit WSB, StockTwits, "
            "and financial news for unusual mention spikes. Buys momentum with "
            "price confirmation. Hard caps: 2% per position, 10% total."
        )

    # Thresholds — tunable by web optimizer (within bounds)
    MIN_MENTION_SPIKE  = 1.2    # 2.0 was US-centric; XETRA stocks get less Reddit chatter
    MIN_SENTIMENT      = 0.0    # neutral sentiment acceptable (was 0.15)
    MIN_PRICE          = 2.0    # ignore sub-$2 stocks
    MAX_PRICE          = 500.0  # ignore extremely expensive for sizing
    MOMENTUM_LOOKBACK  = 3      # bars of price momentum needed
    # Google Trends primary path (fires even without Reddit data)
    GTREND_SPIKE_MIN   = 2.5    # search interest must be 2.5× 3-month avg
    GTREND_SCORE_MIN   = 30     # absolute interest floor (0–100)

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """
        context['meme_signals']  — Reddit mention counts + VADER sentiment
        context['google_trends'] — Google Trends search interest per ticker
        context['macro']         — CNN Fear & Greed + FRED regime
        context['bars']          — OHLCV bars
        context['current_positions'] — set of currently held symbols
        """
        signals: list[SignalResult] = []
        meme_signals: dict   = context.get("meme_signals", {})
        bars_data: dict      = context.get("bars", {})
        held: set            = context.get("current_positions", set())
        google_trends: dict  = context.get("google_trends", {})
        macro: dict          = context.get("macro", {})

        # In extreme fear regimes, reduce max new meme positions
        regime      = macro.get("regime", "neutral")
        fear_greed  = macro.get("fear_greed_score", 50)
        max_new_pos = 1 if regime == "risk-off" or fear_greed < 25 else 3

        # Score and rank all meme candidates
        # Evaluate symbols with Reddit data first, then Trends-only symbols
        seen: set[str] = set()
        candidates = []
        for symbol, ms in meme_signals.items():
            if symbol not in universe:
                continue
            seen.add(symbol)
            trend = google_trends.get(symbol, {})
            score = self._score(symbol, ms, bars_data.get(symbol, []), trend)
            if score is not None:
                candidates.append((symbol, score, ms))

        # Google Trends-primary path: evaluate universe symbols with no Reddit data
        for symbol, trend in google_trends.items():
            if symbol in seen or symbol not in universe:
                continue
            gtrend_spike = trend.get("spike_ratio", 1.0)
            gtrend_dir   = trend.get("trend", "flat")
            gtrend_score = trend.get("interest_score", 0)
            if (gtrend_spike >= self.GTREND_SPIKE_MIN
                    and gtrend_dir == "rising"
                    and gtrend_score >= self.GTREND_SCORE_MIN):
                sig = self._score(symbol, {}, bars_data.get(symbol, []), trend)
                if sig is not None:
                    candidates.append((symbol, sig, {}))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1].score, reverse=True)

        # Generate BUY signals for top candidates
        buy_count = 0
        for symbol, sig, ms in candidates:
            if sig.action == "BUY" and buy_count < max_new_pos:
                signals.append(sig)
                buy_count += 1
            elif sig.action == "SELL":
                signals.append(sig)

        # Check exits for existing meme positions
        for symbol in held:
            if symbol not in meme_signals:
                continue
            ms = meme_signals[symbol]
            # Exit if sentiment turned negative or spike faded
            if ms.get("sentiment", 0) < -0.1 or ms.get("mention_spike", 1) < 0.5:
                signals.append(SignalResult(
                    symbol=symbol,
                    action="SELL",
                    score=-0.7,
                    confidence=0.8,
                    reason=f"Meme fading: sentiment={ms.get('sentiment', 0):.2f}, "
                           f"spike={ms.get('mention_spike', 1):.1f}x",
                    strategy=self.name,
                    is_meme=True,
                ))

        logger.info(f"[MemeMomentum] {len(signals)} signals ({buy_count} buys)")
        return signals

    def _score(
        self, symbol: str, ms: dict, bars: list[dict], trend: dict
    ) -> SignalResult | None:
        mention_spike = ms.get("mention_spike", 0)
        sentiment     = ms.get("sentiment", 0)
        sources       = ms.get("sources", [])

        # Google Trends data
        gtrend_score  = trend.get("interest_score", 0)     # 0–100
        gtrend_spike  = trend.get("spike_ratio", 1.0)      # vs 3-month avg
        gtrend_dir    = trend.get("trend", "flat")          # rising/falling/flat

        # Two entry paths:
        #   1. Reddit path — mention spike + positive sentiment
        #   2. Google Trends-primary path — strong search interest spike
        #      (covers XETRA stocks with little Reddit presence)
        reddit_ok  = mention_spike >= self.MIN_MENTION_SPIKE and sentiment >= self.MIN_SENTIMENT
        gtrend_ok  = (gtrend_spike >= self.GTREND_SPIKE_MIN
                      and gtrend_dir == "rising"
                      and gtrend_score >= self.GTREND_SCORE_MIN)

        if not reddit_ok and not gtrend_ok:
            return None

        # Price confirmation — required for both paths
        price_confirmed = False
        last_price = 0.0
        if bars and len(bars) >= self.MOMENTUM_LOOKBACK + 1:
            closes = [b["c"] for b in bars[-self.MOMENTUM_LOOKBACK - 1:]]
            if closes[-1] > closes[0]:  # price up over lookback
                price_confirmed = True
            last_price = closes[-1]

            if last_price < self.MIN_PRICE or last_price > self.MAX_PRICE:
                return None

        if not price_confirmed:
            return None

        # Volume spike check
        vol_spike = False
        if bars and len(bars) >= 20:
            recent_vol = bars[-1]["v"]
            avg_vol    = sum(b["v"] for b in bars[-20:]) / 20
            vol_spike  = recent_vol > avg_vol * 1.5

        if gtrend_ok and not reddit_ok:
            # Google Trends-primary path — no Reddit data required
            score = min(1.0,
                min(1.0, gtrend_spike / 5.0) * 0.50    # trends spike magnitude
                + (0.20 if gtrend_score > 50 else 0.10) # absolute interest level
                + (0.15 if vol_spike else 0.0)           # volume confirmation
                + 0.10                                   # price-confirmed bonus
            )
            reason = (
                f"Trends spike: {gtrend_spike:.1f}×avg, "
                f"interest={gtrend_score:.0f}/100 ({gtrend_dir})"
                f"{', vol surge' if vol_spike else ''}"
            )
        else:
            # Reddit path with optional Trends boost
            gtrend_bonus = 0.0
            if gtrend_spike > 2.0 and gtrend_dir == "rising":
                gtrend_bonus = 0.12
            elif gtrend_spike > 1.5 or gtrend_dir == "rising":
                gtrend_bonus = 0.06

            score = min(1.0,
                min(1.0, mention_spike / 5.0) * 0.38   # Reddit mention spike
                + max(0, min(1, sentiment))     * 0.32  # social sentiment
                + (0.13 if vol_spike else 0)            # volume confirmation
                + (0.05 if len(sources) > 1 else 0)    # multi-source bonus
                + gtrend_bonus
            )
            multi_source = ", ".join(sources)
            trend_str = (f"gtrend={gtrend_score:.0f}/100 ({gtrend_dir}, {gtrend_spike:.1f}×avg)"
                         if gtrend_score > 0 else "no trend data")
            reason = (
                f"Meme spike: {mention_spike:.1f}× mentions, "
                f"sentiment={sentiment:.2f}, sources=[{multi_source}], "
                f"{'vol spike, ' if vol_spike else ''}{trend_str}"
            )

        return SignalResult(
            symbol=symbol,
            action="BUY",
            score=round(score, 3),
            confidence=min(0.9, score),
            reason=reason,
            strategy=self.name,
            is_meme=True,
            metadata={
                "mention_count":  ms.get("mention_count", 0),
                "mention_spike":  mention_spike,
                "sentiment":      sentiment,
                "top_post":       ms.get("top_post", ""),
                "sources":        sources,
                "gtrend_score":   gtrend_score,
                "gtrend_spike":   gtrend_spike,
                "gtrend_trend":   gtrend_dir,
            },
        )
