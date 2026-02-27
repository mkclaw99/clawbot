"""
Strategy 4: Options Flow / Unusual Activity
============================================
Follows unusual options activity — large block trades, high call/put skew,
and dark pool prints — as a proxy for informed/institutional money.

Data sources (web crawler feeds into context['options_flow']):
  - Unusual Whales (free tier / web scrape)
  - Barchart unusual options
  - Market Chameleon

The strategy does NOT trade options — it uses options flow as a
directional signal to trade the underlying equity.
"""
from __future__ import annotations

from loguru import logger

from strategies.base import BaseStrategy, SignalResult


class OptionsFlowStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "options_flow"

    @property
    def description(self) -> str:
        return (
            "Follows unusual options activity (large blocks, high call/put skew) "
            "as a signal for institutional / smart-money directional bets. "
            "Trades the underlying equity only."
        )

    # Thresholds
    MIN_PREMIUM_RATIO  = 3.0   # options premium must be 3× average
    MIN_CALL_PUT_RATIO = 2.0   # calls:puts ratio for bullish signal
    MAX_DAYS_TO_EXPIRY = 30    # ignore LEAPS (may be hedges)
    MIN_OI_CHANGE      = 0.20  # 20% open interest change
    CONFIDENCE_BASE    = 0.60

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """
        context['options_flow'] = {
            symbol: {
                'call_put_ratio': float,
                'premium_ratio': float,       # vs 30-day avg
                'oi_change': float,           # % change in OI
                'avg_days_to_expiry': float,
                'block_trade_count': int,
                'direction': 'bullish'|'bearish'|'neutral',
                'source': str,
                'notable_trade': str,         # human-readable description
            }
        }
        """
        signals: list[SignalResult] = []
        flow_data: dict = context.get("options_flow", {})
        bars_data: dict = context.get("bars", {})
        held: set       = context.get("current_positions", set())

        for symbol, flow in flow_data.items():
            if symbol not in universe:
                continue
            try:
                sig = self._analyse(symbol, flow, bars_data.get(symbol, []), held)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"[OptionsFlow] Error {symbol}: {e}")

        logger.info(f"[OptionsFlow] {len(signals)} signals")
        return signals

    def _analyse(self, symbol: str, flow: dict, bars: list[dict], held: set) -> SignalResult | None:
        direction     = flow.get("direction", "neutral")
        cp_ratio      = flow.get("call_put_ratio", 1.0)
        prem_ratio    = flow.get("premium_ratio", 1.0)
        oi_change     = flow.get("oi_change", 0.0)
        avg_dte       = flow.get("avg_days_to_expiry", 60)
        block_count   = flow.get("block_trade_count", 0)
        notable_trade = flow.get("notable_trade", "")

        # Filter: ignore if LEAPS-dominated or no unusual activity
        if avg_dte > self.MAX_DAYS_TO_EXPIRY:
            return None
        if prem_ratio < self.MIN_PREMIUM_RATIO:
            return None

        # Price confirmation from bars
        price_trend = "flat"
        if bars and len(bars) >= 5:
            closes = [b["c"] for b in bars[-5:]]
            if closes[-1] > closes[0] * 1.005:
                price_trend = "up"
            elif closes[-1] < closes[0] * 0.995:
                price_trend = "down"

        # --- EXIT for held positions ---
        if symbol in held:
            if direction == "bearish" and price_trend == "down":
                return SignalResult(
                    symbol=symbol,
                    action="SELL",
                    score=0.65,
                    confidence=0.70,
                    reason=f"Options flow turned bearish: CP={cp_ratio:.1f}, trend=down",
                    strategy=self.name,
                )
            return None

        # --- BUY signal ---
        if (direction == "bullish"
                and cp_ratio >= self.MIN_CALL_PUT_RATIO
                and prem_ratio >= self.MIN_PREMIUM_RATIO
                and oi_change >= self.MIN_OI_CHANGE
                and price_trend in ("up", "flat")
                and block_count >= 1):

            score = min(1.0,
                (cp_ratio / 5.0) * 0.3
                + min(1, prem_ratio / 10.0) * 0.3
                + min(1, oi_change) * 0.2
                + (0.2 if price_trend == "up" else 0.1)
            )

            return SignalResult(
                symbol=symbol,
                action="BUY",
                score=round(float(score), 3),
                confidence=min(0.85, self.CONFIDENCE_BASE + score * 0.25),
                reason=(
                    f"Unusual options: CP={cp_ratio:.1f}, premium {prem_ratio:.1f}×avg, "
                    f"OI+{oi_change:.0%}, {block_count} blocks. {notable_trade}"
                ),
                strategy=self.name,
                metadata=flow,
            )

        # --- SELL/short-avoidance signal ---
        if (direction == "bearish"
                and cp_ratio < 0.5
                and prem_ratio >= self.MIN_PREMIUM_RATIO
                and price_trend == "down"):
            # We don't short, but we want to exit if held
            return None  # handled in exit block above

        return None
