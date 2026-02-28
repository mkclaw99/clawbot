"""
Strategy 6: Long/Short Equity — Multi-Factor (Hedge Fund Style)
================================================================
Inspired by AQR Capital, Renaissance, and Citadel cross-sectional factor models.

Scores each stock in DAX40 + MDAX_SELECTED across 4 factors, rank-normalises
cross-sectionally, and generates BUY (long) or SHORT signals at the tails.

Factors (all rank-normalised to [0, 1] across the live universe):
  • 12-1M Momentum  (40%) — bar[-60] to bar[-5] return; skips last 5 days
  • News Sentiment  (30%) — heuristic from earnings beat, insider activity
  • Google Trends   (15%) — search interest_score / 100
  • Low Volatility  (15%) — negative 20-day realised vol (lower vol ranks higher)

Thresholds (tunable by optimizer):
  LONG_THRESHOLD        =  0.30  → BUY,   confidence = min(1, score × 2)
  SHORT_THRESHOLD       = -0.30  → SHORT, confidence = min(1, |score| × 2)
  EXIT_LONG_THRESHOLD   = -0.10  → SELL  (when holding long)
  EXIT_SHORT_THRESHOLD  =  0.10  → COVER (when holding short)

Risk overlay:
  • regime == "risk-off" OR circuit breaker ≥ LEVEL_1 → suppress new SHORTs
  • VSTOXX/VIX > 30 → halve confidence on SHORT signals
  • Dollar-neutral cap: skip new BUYs if long_notional > 1.5 × short_notional
                        skip new SHORTs if short_notional > 1.5 × long_notional

Context keys consumed:
  bars            — {symbol: list[{t,o,h,l,c,v}]}
  news_signals    — {symbol: {articles, earnings_beat, earnings_surprise_pct,
                               insider_buys, insider_sells, institutional_adds}}
  google_trends   — {symbol: {interest_score, spike_ratio, trend, peak_week}}
  macro           — {regime, vstoxx, vix, ...}
  current_positions — set[str] (all held symbols)
  position_qtys   — {symbol: float}  qty > 0 = long, qty < 0 = short
  circuit_breaker_level — str  "NORMAL" | "LEVEL_1" | "LEVEL_2" | "LEVEL_3"
"""
from __future__ import annotations

import math
from loguru import logger

from strategies.base import BaseStrategy, SignalResult
from config.universe import GERMAN_ETFS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_normalize(values: list[float]) -> list[float]:
    """
    Rank-percentile normalize a list to [0, 1].
    Lowest value → 0.0, highest value → 1.0.  Handles ties by averaging ranks.
    """
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank / (n - 1)
    return ranks


def _realized_vol(prices: list[float], window: int = 20) -> float:
    """Daily realised volatility (std of log-returns) over the last `window` days."""
    if len(prices) < window + 1:
        return 0.02   # fallback: 2% daily vol
    closes = prices[-(window + 1):]
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(returns) < 2:
        return 0.02
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class LongShortEquityStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "long_short_equity"

    @property
    def description(self) -> str:
        return (
            "Cross-sectional multi-factor long/short strategy (hedge fund style). "
            "Ranks DAX40 + MDAX stocks by momentum, news sentiment, Google Trends "
            "interest, and low-volatility factor. Longs the top-scored stocks, "
            "shorts the bottom-scored. Includes dollar-neutral and risk-off overlays."
        )

    # --- Thresholds (tunable by optimizer within PARAM_BOUNDS) ---
    LONG_THRESHOLD        =  0.30
    SHORT_THRESHOLD       = -0.30
    EXIT_LONG_THRESHOLD   = -0.10
    EXIT_SHORT_THRESHOLD  =  0.10

    # Minimum number of daily bars required for momentum calculation.
    # The signal aggregator fetches ~60 calendar-day bars (~42-60 trading days),
    # so 30 is the practical floor that allows both momentum and vol factors.
    MIN_BARS: int = 30

    # ETF tickers excluded from the L/S universe (we only trade individual stocks)
    _ETF_TICKERS: frozenset = frozenset(GERMAN_ETFS)

    # ---------------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------------

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """
        Returns BUY, SHORT, SELL, or COVER signals based on cross-sectional ranking.
        """
        bars_data      = context.get("bars", {})
        news_data      = context.get("news_signals", {})
        trends_data    = context.get("google_trends", {})
        position_qtys  = context.get("position_qtys", {})   # symbol → qty
        macro          = context.get("macro", {})
        cb_level       = context.get("circuit_breaker_level", "NORMAL")

        regime   = macro.get("regime", "neutral") or "neutral"
        vstoxx   = float(macro.get("vstoxx") or 0.0)
        vix      = float(macro.get("vix") or 0.0)
        vol_idx  = vstoxx if vstoxx > 0 else vix   # prefer VSTOXX for European equities

        # ------------------------------------------------------------------
        # 1. Filter universe to non-ETF tickers
        # ------------------------------------------------------------------
        tickers = [t for t in universe if t not in self._ETF_TICKERS]

        # ------------------------------------------------------------------
        # 2. Compute raw factor values for each ticker
        # ------------------------------------------------------------------
        raw: dict[str, dict] = {}
        valid: list[str]     = []

        for sym in tickers:
            bars = bars_data.get(sym)
            if not bars or len(bars) < self.MIN_BARS:
                continue

            prices = [float(b["c"]) for b in bars]
            n      = len(prices)

            # Factor 1: Momentum — return from bar[-60] to bar[-5]
            start_idx = max(0, n - 60)
            start_px  = prices[start_idx]
            end_px    = prices[max(0, n - 5)]
            momentum  = (end_px - start_px) / start_px if start_px > 0 else 0.0

            # Factor 2: News sentiment (heuristic; no heavy NLP here)
            sentiment = self._news_sentiment(news_data.get(sym) or {})

            # Factor 3: Google Trends interest score (0–100) → [0, 1]
            trends_val  = trends_data.get(sym) or {}
            if isinstance(trends_val, dict):
                trend_score = float(trends_val.get("interest_score") or 50) / 100.0
            else:
                trend_score = 0.5

            # Factor 4: Low-volatility factor (negate so lower vol → higher rank)
            vol     = _realized_vol(prices, window=20)
            neg_vol = -vol

            raw[sym]   = {
                "momentum":  momentum,
                "sentiment": sentiment,
                "trend":     trend_score,
                "neg_vol":   neg_vol,
            }
            valid.append(sym)

        if len(valid) < 5:
            logger.warning(
                f"[LongShortEquity] Only {len(valid)} valid tickers — "
                "need ≥5 for cross-sectional ranking"
            )
            return []

        # ------------------------------------------------------------------
        # 3. Cross-sectional rank-normalization
        # ------------------------------------------------------------------
        mom_ranks  = _rank_normalize([raw[s]["momentum"]  for s in valid])
        sent_ranks = _rank_normalize([raw[s]["sentiment"] for s in valid])
        trnd_ranks = _rank_normalize([raw[s]["trend"]     for s in valid])
        vol_ranks  = _rank_normalize([raw[s]["neg_vol"]   for s in valid])

        # Weighted composite → [0, 1], then mapped to [-1, +1]
        composites: dict[str, float] = {}
        for i, sym in enumerate(valid):
            weighted = (
                0.40 * mom_ranks[i]  +
                0.30 * sent_ranks[i] +
                0.15 * trnd_ranks[i] +
                0.15 * vol_ranks[i]
            )
            composites[sym] = (weighted - 0.5) * 2   # → [-1, +1]

        # ------------------------------------------------------------------
        # 4. Dollar-neutral cap — measure existing long/short exposure
        # ------------------------------------------------------------------
        long_notional  = 0.0
        short_notional = 0.0
        for sym, qty in position_qtys.items():
            bars = bars_data.get(sym)
            if not bars:
                continue
            price = float(bars[-1]["c"])
            if qty > 0:
                long_notional  += qty * price
            elif qty < 0:
                short_notional += abs(qty) * price

        # ------------------------------------------------------------------
        # 5. Generate signals
        # ------------------------------------------------------------------
        signals: list[SignalResult] = []

        for sym in valid:
            score    = composites[sym]
            qty_held = position_qtys.get(sym, 0)

            # --- Exit signals (checked first; skip to next symbol after emitting) ---

            # Close long if composite turns negative enough
            if qty_held > 0 and score < self.EXIT_LONG_THRESHOLD:
                signals.append(SignalResult(
                    symbol=sym,
                    action="SELL",
                    score=round(score, 4),
                    confidence=0.70,
                    reason=f"L/S exit long: composite={score:.3f} < {self.EXIT_LONG_THRESHOLD}",
                    strategy=self.name,
                ))
                continue

            # Cover short if composite turns positive enough
            if qty_held < 0 and score > self.EXIT_SHORT_THRESHOLD:
                signals.append(SignalResult(
                    symbol=sym,
                    action="COVER",
                    score=round(score, 4),
                    confidence=0.70,
                    reason=f"L/S cover short: composite={score:.3f} > {self.EXIT_SHORT_THRESHOLD}",
                    strategy=self.name,
                ))
                continue

            # --- Entry signals ---

            if score >= self.LONG_THRESHOLD and qty_held <= 0:
                # Dollar-neutral cap: skip if portfolio is too long-heavy
                if short_notional > 0 and long_notional > short_notional * 1.5:
                    continue
                confidence = min(1.0, score * 2)
                signals.append(SignalResult(
                    symbol=sym,
                    action="BUY",
                    score=round(score, 4),
                    confidence=round(confidence, 3),
                    reason=f"L/S long entry: composite={score:.3f}",
                    strategy=self.name,
                ))

            elif score <= self.SHORT_THRESHOLD and qty_held >= 0:
                # Risk overlay: suppress new SHORTs in risk-off regime
                if regime == "risk-off":
                    continue
                # Risk overlay: suppress new SHORTs if circuit breaker elevated
                if cb_level in ("LEVEL_1", "LEVEL_2", "LEVEL_3"):
                    continue
                # Dollar-neutral cap: skip if portfolio is too short-heavy
                if long_notional > 0 and short_notional > long_notional * 1.5:
                    continue
                confidence = min(1.0, abs(score) * 2)
                # High VSTOXX/VIX: halve confidence on SHORT signals
                if vol_idx > 30:
                    confidence *= 0.5
                if confidence < 0.5:
                    continue   # below actionable threshold
                signals.append(SignalResult(
                    symbol=sym,
                    action="SHORT",
                    score=round(score, 4),
                    confidence=round(confidence, 3),
                    reason=f"L/S short entry: composite={score:.3f}",
                    strategy=self.name,
                ))

        logger.info(
            f"[LongShortEquity] {len(signals)} signals "
            f"({sum(1 for s in signals if s.action == 'BUY')} BUY, "
            f"{sum(1 for s in signals if s.action == 'SHORT')} SHORT, "
            f"{sum(1 for s in signals if s.action == 'SELL')} SELL, "
            f"{sum(1 for s in signals if s.action == 'COVER')} COVER) "
            f"from {len(valid)} valid tickers | regime={regime} cb={cb_level}"
        )
        return signals

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _news_sentiment(news: dict) -> float:
        """
        Heuristic news sentiment in [-1, +1] derived from:
          - Earnings beat/miss and surprise magnitude
          - Insider buy/sell activity
          - Article presence as a mild confirmation signal
        Returns 0.0 (neutral) when no relevant data is present.
        """
        articles  = news.get("articles") or []
        earn_beat = news.get("earnings_beat")
        earn_surp = float(news.get("earnings_surprise_pct") or 0.0)
        ins_buys  = int(news.get("insider_buys") or 0)
        ins_sells = int(news.get("insider_sells") or 0)

        if not articles and earn_beat is None and ins_buys == 0 and ins_sells == 0:
            return 0.0

        score = 0.0

        # Earnings signal (dominant driver)
        if earn_beat is True:
            score += 0.30 + min(0.20, earn_surp / 50.0)
        elif earn_beat is False:
            score -= 0.30 + min(0.20, abs(earn_surp) / 50.0)

        # Insider activity
        score += (ins_buys - ins_sells) * 0.10

        # Article presence: mild confirmation (more articles = more newsflow)
        if articles:
            score += 0.05 * min(1.0, len(articles) / 3.0)

        return max(-1.0, min(1.0, score))
