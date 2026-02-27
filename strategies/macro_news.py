"""
Strategy 5: Macro / News NLP
==============================
Event-driven strategy using NLP sentiment analysis on:
  - Earnings releases (beat/miss + guidance)
  - Fed / FOMC statements
  - SEC filings (Form 4 insider buys, 13F institutional adds)
  - Breaking financial news (NewsAPI + Benzinga)

Uses FinBERT (financial-domain BERT) for news sentiment.
Falls back to VADER if FinBERT model is not available.
"""
from __future__ import annotations

from loguru import logger

from strategies.base import BaseStrategy, SignalResult

# Lazy imports for NLP (heavy models)
_vader = None
_finbert = None


def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def _get_finbert():
    global _finbert
    if _finbert is None:
        try:
            from transformers import pipeline
            _finbert = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
        except Exception as e:
            logger.warning(f"FinBERT unavailable: {e} — using VADER fallback")
    return _finbert


def _sentiment_score(text: str) -> float:
    """Returns -1.0 (very negative) to +1.0 (very positive)."""
    finbert = _get_finbert()
    if finbert:
        try:
            result = finbert(text)[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            return 0.0
        except Exception:
            pass

    # VADER fallback
    vader = _get_vader()
    return vader.polarity_scores(text)["compound"]


class MacroNewsStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "macro_news"

    @property
    def description(self) -> str:
        return (
            "Event-driven NLP strategy. Uses FinBERT to score earnings releases, "
            "insider filings, Fed statements, and breaking news. "
            "Buys strong positive surprises. Exits on negative news."
        )

    # Thresholds
    BUY_SENTIMENT_THRESHOLD  = 0.55   # strong positive news
    SELL_SENTIMENT_THRESHOLD = -0.40  # strong negative news
    MIN_ARTICLE_COUNT        = 2      # require multiple sources
    EARNINGS_BEAT_BONUS      = 0.20   # extra score for earnings beat
    INSIDER_BUY_BONUS        = 0.15   # extra score for Form 4 buy
    CONFIDENCE_BASE          = 0.60

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """
        context['news_signals'] — articles, earnings, insider filings
        context['macro']        — FRED rates, VIX, CNN Fear & Greed, regime
        """
        signals: list[SignalResult] = []
        news_data: dict = context.get("news_signals", {})
        held: set       = context.get("current_positions", set())
        macro: dict     = context.get("macro", {})

        # Extract macro context to pass into each analysis
        regime      = macro.get("regime", "neutral")
        fear_greed  = macro.get("fear_greed_score", 50.0)
        # VSTOXX is the primary European fear gauge; fall back to VIX if absent
        volatility  = macro.get("vstoxx") or macro.get("vix")
        de_yield    = macro.get("de_yield_10y")

        for symbol in universe:
            news = news_data.get(symbol)
            if not news:
                continue
            try:
                sig = self._analyse(symbol, news, held, regime, fear_greed, volatility, de_yield)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"[MacroNews] Error {symbol}: {e}")

        logger.info(
            f"[MacroNews] {len(signals)} signals | "
            f"regime={regime} fear_greed={fear_greed:.0f} vstoxx/vix={volatility}"
        )
        return signals

    def _analyse(
        self, symbol: str, news: dict, held: set,
        regime: str = "neutral", fear_greed: float = 50.0,
        volatility: float | None = None, de_yield: float | None = None,
    ) -> SignalResult | None:
        articles    = news.get("articles", [])
        earn_beat   = news.get("earnings_beat")
        earn_surp   = news.get("earnings_surprise_pct", 0.0) or 0.0
        ins_buys    = news.get("insider_buys", 0)
        ins_sells   = news.get("insider_sells", 0)
        inst_adds   = news.get("institutional_adds", 0)

        if len(articles) < self.MIN_ARTICLE_COUNT:
            return None

        # Score each article
        article_scores = []
        for art in articles[:10]:  # cap at 10 to avoid latency
            text  = f"{art.get('title', '')} {art.get('description', '')}"
            score = _sentiment_score(text)
            article_scores.append(score)

        if not article_scores:
            return None

        avg_sentiment = sum(article_scores) / len(article_scores)

        # Bonus adjustments
        bonus = 0.0
        reasons = [f"news={avg_sentiment:.2f}({len(articles)} articles)"]

        if earn_beat is True:
            bonus += self.EARNINGS_BEAT_BONUS
            reasons.append(f"earnings beat {earn_surp:+.1f}%")
        elif earn_beat is False:
            bonus -= self.EARNINGS_BEAT_BONUS
            reasons.append(f"earnings miss {earn_surp:+.1f}%")

        if ins_buys > 0:
            bonus += self.INSIDER_BUY_BONUS * min(ins_buys, 3)
            reasons.append(f"{ins_buys} insider buy(s)")
        if ins_sells > ins_buys:
            bonus -= self.INSIDER_BUY_BONUS
            reasons.append(f"{ins_sells} insider sell(s)")

        if inst_adds > 0:
            bonus += 0.05 * min(inst_adds, 3)
            reasons.append(f"{inst_adds} institutional add(s)")

        final_score = avg_sentiment + bonus

        # --- Macro context adjustments ---
        macro_reasons: list[str] = []
        macro_penalty  = 0.0
        confidence_adj = 0.0

        # Risk-off or extreme fear: be more conservative on new entries
        if regime == "risk-off" or fear_greed < 25:
            macro_penalty  -= 0.10
            confidence_adj -= 0.10
            macro_reasons.append(f"risk-off(fg={fear_greed:.0f})")
        elif fear_greed < 40:
            macro_penalty  -= 0.05
            confidence_adj -= 0.05
            macro_reasons.append(f"fear(fg={fear_greed:.0f})")

        # High VSTOXX/VIX: uncertainty penalty on confidence
        if volatility is not None and volatility > 35:
            confidence_adj -= 0.10
            macro_reasons.append(f"high-vol={volatility:.1f}")
        elif volatility is not None and volatility > 25:
            confidence_adj -= 0.05
            macro_reasons.append(f"vol={volatility:.1f}")

        # Rising German Bund yield: headwind for European equities
        if de_yield is not None and de_yield > 3.0:
            macro_penalty -= 0.05
            macro_reasons.append(f"de-bund={de_yield:.2f}%")

        # Risk-on boost
        if regime == "risk-on" and fear_greed > 60:
            macro_penalty  += 0.05
            confidence_adj += 0.05
            macro_reasons.append(f"risk-on(fg={fear_greed:.0f})")

        if macro_reasons:
            reasons.append("macro:[" + ",".join(macro_reasons) + "]")

        final_score += macro_penalty

        # --- EXIT ---
        # In risk-off regimes exits fire more readily (lower abs threshold)
        sell_threshold = self.SELL_SENTIMENT_THRESHOLD
        if regime == "risk-off" or fear_greed < 25:
            sell_threshold += 0.10   # e.g. -0.40 → -0.30 (fires sooner)

        if symbol in held and final_score < sell_threshold:
            sell_confidence = min(0.85, 0.70 - confidence_adj)  # higher conf to exit in bad macro
            return SignalResult(
                symbol=symbol,
                action="SELL",
                score=abs(final_score),
                confidence=round(max(0.55, sell_confidence), 3),
                reason="Negative news: " + ", ".join(reasons),
                strategy=self.name,
            )

        # --- BUY ---
        if symbol not in held and final_score >= self.BUY_SENTIMENT_THRESHOLD:
            confidence = min(0.90, max(0.30,
                self.CONFIDENCE_BASE + abs(final_score) * 0.3 + confidence_adj
            ))
            return SignalResult(
                symbol=symbol,
                action="BUY",
                score=min(1.0, final_score),
                confidence=round(confidence, 3),
                reason="Positive news: " + ", ".join(reasons),
                strategy=self.name,
                metadata={
                    "avg_sentiment":  avg_sentiment,
                    "bonus":          bonus,
                    "macro_penalty":  macro_penalty,
                    "article_count":  len(articles),
                    "regime":         regime,
                    "fear_greed":     fear_greed,
                    "volatility":     volatility,
                    "de_yield":       de_yield,
                },
            )

        return None
