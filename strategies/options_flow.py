"""
Strategy 4: Volume & Price Anomaly
====================================
Detects unusual volume and gap-up activity in XETRA equities as a proxy
for institutional / smart-money accumulation.

Replaces the original options-flow strategy which required options chain data
not available for German stocks on yfinance.

Signal logic:
  BUY  — volume surge (≥ MIN_VOL_RATIO × 20-day avg) with price moving up
          Optional gap-up bonus when open > prior close by ≥ 0.3%
  SELL — held position reverses ≥ 1% after a volume-driven entry
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
            "Volume & price anomaly detection for XETRA. Detects unusual volume "
            "surges and gap-up moves as institutional activity proxies. "
            "Trades underlying equity only."
        )

    # Thresholds — tunable by optimizer
    MIN_VOL_RATIO  = 2.0    # volume must be ≥ 2× 20-day average
    MIN_PRICE_MOVE = 0.005  # price must be up ≥ 0.5% on the volume day
    VOL_LOOKBACK   = 20     # days for volume baseline
    EXIT_REVERSAL  = -0.01  # exit held position on ≥ 1% intraday reversal

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """
        context['bars']             — OHLCV bars per symbol
        context['current_positions'] — set of currently held symbols
        context['macro']            — regime; SHORTs suppressed in risk-off
        """
        bars_data: dict = context.get("bars", {})
        held: set       = context.get("current_positions", set())
        macro: dict     = context.get("macro", {})
        regime          = macro.get("regime", "neutral")

        signals: list[SignalResult] = []
        for symbol in universe:
            bars = bars_data.get(symbol, [])
            if not bars or len(bars) < self.VOL_LOOKBACK + 1:
                continue
            try:
                sig = self._analyse(symbol, bars, held, regime)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"[VolumeAnomaly] {symbol}: {e}")

        logger.info(f"[VolumeAnomaly] {len(signals)} signals")
        return signals

    def _analyse(
        self, symbol: str, bars: list[dict], held: set, regime: str
    ) -> SignalResult | None:
        recent = bars[-1]
        prev   = bars[-2]

        # Volume baseline — exclude most-recent bar to avoid look-ahead
        vol_window = bars[-(self.VOL_LOOKBACK + 1):-1]
        avg_vol = sum(b["v"] for b in vol_window) / self.VOL_LOOKBACK
        if avg_vol == 0:
            return None
        vol_ratio = recent["v"] / avg_vol

        price_change = (recent["c"] - prev["c"]) / prev["c"] if prev["c"] > 0 else 0.0
        gap_up       = (recent["o"] - prev["c"]) / prev["c"] if prev["c"] > 0 else 0.0

        # --- EXIT held positions on reversal ---
        if symbol in held:
            if price_change < self.EXIT_REVERSAL:
                return SignalResult(
                    symbol=symbol,
                    action="SELL",
                    score=0.60,
                    confidence=0.70,
                    reason=(
                        f"Volume anomaly reversal: {price_change:+.1%} "
                        f"on vol {vol_ratio:.1f}×avg"
                    ),
                    strategy=self.name,
                )
            return None

        # --- BUY signal ---
        if vol_ratio < self.MIN_VOL_RATIO:
            return None
        if price_change < self.MIN_PRICE_MOVE:
            return None
        if regime == "risk-off":
            return None

        score = min(1.0,
            min(1.0, vol_ratio / 5.0)          * 0.50   # volume surge magnitude
            + min(1.0, price_change / 0.03)    * 0.30   # price move
            + (0.20 if gap_up > 0.003 else 0.10)        # gap-up bonus vs flat open
        )

        return SignalResult(
            symbol=symbol,
            action="BUY",
            score=round(score, 3),
            confidence=min(0.85, 0.60 + score * 0.25),
            reason=(
                f"Volume surge: {vol_ratio:.1f}×avg, "
                f"price {price_change:+.1%}, gap={gap_up:+.1%}"
            ),
            strategy=self.name,
            metadata={
                "vol_ratio":    round(vol_ratio, 2),
                "price_change": round(price_change, 4),
                "gap_up":       round(gap_up, 4),
            },
        )
