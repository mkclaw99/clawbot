"""
Strategy 3: Mean Reversion
===========================
Buys statistically oversold stocks (Z-score < -2 vs 20-day rolling mean)
with Bollinger Band confirmation. Exits when price reverts to the mean.

Best suited for range-bound, large-cap stocks.
NOT applied to meme stocks (which can stay irrational a long time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, SignalResult


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return (
            "Statistical mean reversion using Bollinger Bands and Z-score. "
            "Buys oversold stocks (Z < -2, below lower BB). "
            "Exits at mean (middle BB). Avoids meme/high-volatility stocks."
        )

    # Parameters
    ZSCORE_ENTRY  = -1.5     # buy when Z < -1.5 (was -2.0; loosened to generate more signals)
    ZSCORE_EXIT   = 0.0      # exit when Z >= 0 (back to mean)
    BB_PERIOD     = 20
    BB_STD        = 2.0
    MIN_BARS      = 30
    MAX_VOLATILITY = 0.04    # skip stocks with >4% daily std dev (too volatile)

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        signals: list[SignalResult] = []
        bars_data: dict = context.get("bars", {})
        held: set       = context.get("current_positions", set())
        meme_set: set   = context.get("meme_universe", set())

        for symbol in universe:
            if symbol in meme_set:
                continue  # mean reversion doesn't apply to meme stocks

            bars = bars_data.get(symbol)
            if not bars or len(bars) < self.MIN_BARS:
                continue

            try:
                sig = self._analyse(symbol, bars, held)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"[MeanReversion] Error {symbol}: {e}")

        logger.info(f"[MeanReversion] {len(signals)} signals")
        return signals

    def _analyse(self, symbol: str, bars: list[dict], held: set) -> SignalResult | None:
        df     = pd.DataFrame(bars).sort_values("t").reset_index(drop=True)
        close  = df["c"].astype(float)

        # Bollinger Bands
        ma  = close.rolling(self.BB_PERIOD).mean()
        std = close.rolling(self.BB_PERIOD).std()
        upper_bb = ma + self.BB_STD * std
        lower_bb = ma - self.BB_STD * std

        # Z-score
        zscore = (close - ma) / std.replace(0, np.nan)

        last_close = close.iloc[-1]
        last_z     = zscore.iloc[-1]
        last_ma    = ma.iloc[-1]
        last_lower = lower_bb.iloc[-1]
        last_upper = upper_bb.iloc[-1]

        if pd.isna(last_z) or pd.isna(last_ma):
            return None

        # Skip high-volatility stocks (std as % of price)
        daily_vol = std.iloc[-1] / last_close if last_close > 0 else 0
        if daily_vol > self.MAX_VOLATILITY:
            return None

        # ---------- EXIT for held positions ----------
        if symbol in held:
            if last_z >= self.ZSCORE_EXIT:
                return SignalResult(
                    symbol=symbol,
                    action="SELL",
                    score=0.6,
                    confidence=0.75,
                    reason=f"Mean reversion complete: Z={last_z:.2f}, price={last_close:.2f}, MA={last_ma:.2f}",
                    strategy=self.name,
                )
            # Stop-loss: if Z goes even more negative (wrong-way trade)
            if last_z < -3.5:
                return SignalResult(
                    symbol=symbol,
                    action="SELL",
                    score=0.8,
                    confidence=0.85,
                    reason=f"Stop loss: Z={last_z:.2f} (extreme deviation, exiting)",
                    strategy=self.name,
                )
            return None

        # ---------- ENTRY ----------
        if last_z < self.ZSCORE_ENTRY and last_close < last_lower:
            # Additional filter: not in free-fall (price still above 52-week low region)
            min_close = close.tail(min(len(close), 50)).min()
            if last_close < min_close * 0.85:
                return None  # possible fundamental break, not mean reversion

            upside = (last_ma - last_close) / last_close if last_close > 0 else 0
            score  = min(1.0, abs(last_z) / 4.0)

            return SignalResult(
                symbol=symbol,
                action="BUY",
                score=round(float(score), 3),
                confidence=0.65,
                reason=(
                    f"Oversold: Z={last_z:.2f}, below BB({self.BB_PERIOD},{self.BB_STD}), "
                    f"upside to mean={upside:.1%}"
                ),
                strategy=self.name,
                metadata={"zscore": last_z, "upside_to_mean": upside},
            )

        return None
