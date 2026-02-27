"""
Strategy 2: Technical Trend Following
======================================
Uses moving averages (EMA 9/21/50), RSI, MACD, and volume to identify
trending stocks. Buys when price is above all EMAs with RSI not overbought.
Sells when trend reverses or RSI hits overbought territory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, SignalResult


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast  = _ema(series, 12)
    slow  = _ema(series, 26)
    macd  = fast - slow
    signal = _ema(macd, 9)
    return macd, signal


class TechnicalTrendStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "technical_trend"

    @property
    def description(self) -> str:
        return (
            "Trend following using EMA (9/21/50), RSI(14), and MACD. "
            "Buys uptrending stocks with momentum confirmation. "
            "Exits on trend reversal or overbought RSI."
        )

    # Parameters — these can be adjusted by the web optimizer within bounds
    RSI_OVERSOLD   = 35
    RSI_OVERBOUGHT = 70
    MIN_BARS       = 55   # need at least 55 bars for EMA-50

    def generate_signals(self, universe: list[str], context: dict) -> list[SignalResult]:
        """context['bars'] = {symbol: list[dict(t,o,h,l,c,v)]}"""
        signals: list[SignalResult] = []
        bars_data: dict = context.get("bars", {})

        for symbol in universe:
            bars = bars_data.get(symbol)
            if not bars or len(bars) < self.MIN_BARS:
                continue

            try:
                signal = self._analyse(symbol, bars)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"[TechnicalTrend] Error analysing {symbol}: {e}")

        logger.info(f"[TechnicalTrend] {len(signals)} signals from {len(universe)} symbols")
        return signals

    def _analyse(self, symbol: str, bars: list[dict]) -> SignalResult | None:
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.sort_values("t").reset_index(drop=True)

        close  = df["c"].astype(float)
        volume = df["v"].astype(float)

        ema9  = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        rsi   = _rsi(close)
        macd, macd_sig = _macd(close)
        vol_sma = volume.rolling(20).mean()

        last_close   = close.iloc[-1]
        last_rsi     = rsi.iloc[-1]
        last_macd    = macd.iloc[-1]
        last_macd_sig = macd_sig.iloc[-1]
        last_ema9    = ema9.iloc[-1]
        last_ema21   = ema21.iloc[-1]
        last_ema50   = ema50.iloc[-1]
        last_vol     = volume.iloc[-1]
        avg_vol      = vol_sma.iloc[-1]

        # Volume confirmation: current volume above average
        vol_confirmed = last_vol > avg_vol * 0.8

        # ---------- BUY conditions ----------
        # MACD bullish: crossover only (no longer require positive MACD line,
        # which blocked all entries during ranging/recovering markets)
        macd_bull  = last_macd > last_macd_sig
        rsi_ok     = self.RSI_OVERSOLD < last_rsi < self.RSI_OVERBOUGHT

        # Full uptrend: price above all 3 EMAs
        in_uptrend = (last_close > last_ema9 > last_ema21 > last_ema50)

        # Early-recovery: price above EMA9 and EMA21 (not yet above EMA50)
        early_recovery = (last_close > last_ema9 > last_ema21) and not in_uptrend

        if in_uptrend and rsi_ok and macd_bull and vol_confirmed:
            score = min(1.0, 0.4 + (last_rsi - 50) / 100 + (last_macd / last_close))
            return SignalResult(
                symbol=symbol,
                action="BUY",
                score=round(float(score), 3),
                confidence=0.7,
                reason=f"Uptrend: close>{last_ema50:.2f}, RSI={last_rsi:.1f}, MACD cross",
                strategy=self.name,
            )

        if early_recovery and rsi_ok and macd_bull and vol_confirmed:
            score = min(0.75, 0.3 + (last_rsi - 50) / 100 + (last_macd / last_close))
            return SignalResult(
                symbol=symbol,
                action="BUY",
                score=round(float(score), 3),
                confidence=0.65,
                reason=f"Early recovery: close>{last_ema21:.2f}, RSI={last_rsi:.1f}, MACD cross",
                strategy=self.name,
            )

        # ---------- SELL conditions ----------
        # Price drops below EMA9 with RSI overbought or MACD bearish cross
        below_ema9 = last_close < last_ema9
        rsi_over   = last_rsi > self.RSI_OVERBOUGHT
        macd_bear  = last_macd < last_macd_sig

        if below_ema9 and (rsi_over or macd_bear):
            score = max(-1.0, -0.4 - (last_rsi - 50) / 100)
            return SignalResult(
                symbol=symbol,
                action="SELL",
                score=round(float(score), 3),
                confidence=0.65,
                reason=f"Below EMA9={last_ema9:.2f}, RSI={last_rsi:.1f}, MACD={'bear' if macd_bear else 'ok'}",
                strategy=self.name,
            )

        return None
