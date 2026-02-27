"""
Signal aggregator — orchestrates all crawlers and builds the context dict
passed to each strategy's generate_signals() method.
Also runs the web optimization loop that adjusts strategy parameters.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from crawler.reddit_crawler import fetch_reddit_signals
from crawler.news_crawler import fetch_news_signals
from crawler.options_crawler import fetch_options_flow
from crawler.google_trends_crawler import fetch_google_trends
from crawler.macro_crawler import fetch_macro_context
from core.audit import audit_signal
from config.universe import DEFAULT_UNIVERSE, HIGH_VOLATILITY


# MEME_UNIVERSE repurposed as "high-volatility / speculative" German stocks
MEME_UNIVERSE: set[str] = HIGH_VOLATILITY

# How often each data source refreshes (seconds)
_BARS_TTL          = 300    # 5 min
_REDDIT_TTL        = 300    # 5 min
_NEWS_TTL          = 600    # 10 min (NewsAPI has rate limits)
_OPTIONS_TTL       = 900    # 15 min (yfinance options are slow)
_GOOGLE_TRENDS_TTL = 3600   # 1 hour (trends change slowly)
_MACRO_TTL         = 3600   # 1 hour (FRED + CNN data)
_CORRELATION_TTL   = 86400  # 24 hours (updated nightly)


class SignalAggregator:
    """
    Runs all crawlers on a schedule and builds a unified context dict.
    The engine calls refresh() periodically and passes the result to strategies.

    Context keys produced:
        bars            — OHLCV from yfinance (60-day daily)
        meme_signals    — Reddit mention counts + VADER sentiment
        news_signals    — NewsAPI articles + SEC insider filings
        options_flow    — Real yfinance options chains (call/put vol, OI, IV)
        google_trends   — Google Trends search interest per ticker
        macro           — FRED macro data + CNN Fear & Greed
    """

    def __init__(self, universe: list[str] | None = None) -> None:
        self.universe = universe or DEFAULT_UNIVERSE

        self._last_context: dict = {}
        self._last_refresh: float = 0.0

        # Per-source timestamps for independent TTLs
        self._ts: dict[str, float] = {}

        # Per-source caches
        self._cache: dict[str, Any] = {}

    def refresh(self, force: bool = False) -> dict[str, Any]:
        """
        Refresh all signals. Returns the context dict.
        Each source has its own TTL — only stale sources are re-fetched.
        Minimum 60s between full refresh cycles unless force=True.
        """
        now = time.time()
        if not force and (now - self._last_refresh) < 60:
            return self._last_context

        logger.info("[Aggregator] Refreshing signals...")

        context: dict[str, Any] = {
            "universe":     self.universe,
            "meme_universe": MEME_UNIVERSE,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

        # --- 1. Price bars (yfinance, 5-min TTL) ---
        context["bars"] = self._refresh_source(
            "bars", _BARS_TTL, force,
            lambda: self._fetch_bars(),
        )

        # --- 2. Reddit / meme signals (5-min TTL) ---
        context["meme_signals"] = self._refresh_source(
            "meme_signals", _REDDIT_TTL, force,
            lambda: fetch_reddit_signals(universe=self.universe),
        )

        # --- 3. News / earnings / insider signals (10-min TTL) ---
        context["news_signals"] = self._refresh_source(
            "news_signals", _NEWS_TTL, force,
            lambda: fetch_news_signals(self.universe),
        )

        # --- 4. Options flow — real yfinance options chains (15-min TTL) ---
        context["options_flow"] = self._refresh_source(
            "options_flow", _OPTIONS_TTL, force,
            lambda: fetch_options_flow(self.universe),
        )

        # --- 5. Google Trends — search interest per ticker (1-hour TTL) ---
        context["google_trends"] = self._refresh_source(
            "google_trends", _GOOGLE_TRENDS_TTL, force,
            lambda: fetch_google_trends(self.universe),
        )

        # --- 6. Macro — FRED rates + CNN Fear & Greed (1-hour TTL) ---
        context["macro"] = self._refresh_source(
            "macro", _MACRO_TTL, force,
            lambda: fetch_macro_context(),
        )

        # --- 7. Correlations — pre-computed nightly (24-hour TTL) ---
        context["correlations"] = self._refresh_source(
            "correlations", _CORRELATION_TTL, force,
            lambda: self._fetch_correlations(),
        )

        # --- Audit high-signal meme mentions ---
        for symbol, ms in context["meme_signals"].items():
            if ms.get("mention_spike", 0) > 2:
                audit_signal(
                    "meme_momentum", symbol,
                    "BUY" if ms["sentiment"] > 0.1 else "HOLD",
                    ms["mention_spike"], "reddit",
                    mention_count=ms["mention_count"],
                    sentiment=ms["sentiment"],
                )

        # --- Audit macro regime changes ---
        macro = context.get("macro", {})
        if macro.get("regime") == "risk-off":
            audit_signal(
                "macro_news", "MARKET", "HOLD",
                macro.get("fear_greed_score", 50), "macro",
                regime="risk-off",
                vix=macro.get("vix"),
                fear_greed=macro.get("fear_greed_score"),
            )

        self._last_context = context
        self._last_refresh = now

        regime     = macro.get("regime", "?")
        fear_greed = macro.get("fear_greed_score", "?")
        trend_up   = sum(1 for v in context["google_trends"].values() if v.get("trend") == "rising")
        real_opts  = sum(1 for v in context["options_flow"].values() if v.get("source") == "yfinance_options")
        logger.info(
            f"[Aggregator] Refresh complete. Universe={len(self.universe)} | "
            f"regime={regime} | fear_greed={fear_greed} | "
            f"trends_rising={trend_up} | real_options={real_opts}"
        )
        return context

    def _refresh_source(
        self, key: str, ttl: float, force: bool, fetcher
    ) -> Any:
        """
        Re-fetch a data source if its TTL has expired (or force=True).
        Returns cached value otherwise.
        Silently returns empty dict on failure.
        """
        now = time.time()
        if not force and key in self._cache and (now - self._ts.get(key, 0)) < ttl:
            return self._cache[key]

        try:
            result = fetcher()
            self._cache[key] = result
            self._ts[key]    = now
            return result
        except Exception as e:
            logger.error(f"[Aggregator] Source '{key}' failed: {e}")
            return self._cache.get(key, {})

    def _fetch_bars(self) -> dict[str, list[dict]]:
        """Download 60 days of daily bars for the universe via yfinance."""
        logger.info(f"[Aggregator] Fetching price bars for {len(self.universe)} tickers...")
        bars: dict[str, list[dict]] = {}

        try:
            tickers_str = " ".join(self.universe)
            df = yf.download(
                tickers_str,
                period="60d",
                interval="1d",
                group_by="ticker",
                progress=False,
                auto_adjust=True,
            )

            for symbol in self.universe:
                try:
                    if len(self.universe) == 1:
                        sym_df = df
                    else:
                        sym_df = df[symbol] if symbol in df.columns.get_level_values(0) else None

                    if sym_df is None or sym_df.empty:
                        continue

                    records = []
                    for ts, row in sym_df.dropna().iterrows():
                        records.append({
                            "t": ts.isoformat(),
                            "o": float(row["Open"]),
                            "h": float(row["High"]),
                            "l": float(row["Low"]),
                            "c": float(row["Close"]),
                            "v": float(row["Volume"]),
                        })
                    if records:
                        bars[symbol] = records
                except Exception as e:
                    logger.debug(f"[Aggregator] Bars error {symbol}: {e}")

        except Exception as e:
            logger.error(f"[Aggregator] Batch bars download failed: {e}")

        logger.info(f"[Aggregator] Got bars for {len(bars)}/{len(self.universe)} tickers")
        return bars

    def _fetch_correlations(self) -> dict:
        """
        Load pre-computed correlations from market.db.
        Returns context["correlations"] — a dict keyed by symbol.
        Returns empty dict gracefully if market.db has no data yet.
        """
        try:
            from core.market_db import MarketDatabase
            from core.correlation_engine import CorrelationEngine
            db     = MarketDatabase()
            engine = CorrelationEngine(db)
            result = engine.get_context_dict(self.universe, period_days=60)
            logger.info(
                f"[Aggregator] Correlation context loaded for "
                f"{len(result)} symbols"
            )
            return result
        except Exception as e:
            logger.debug(f"[Aggregator] Correlations unavailable: {e}")
            return {}

    def get_top_meme_signals(self, top_n: int = 10) -> list[dict]:
        """Helper for the dashboard: returns top N meme signals sorted by spike."""
        meme = self._last_context.get("meme_signals", {})
        sorted_meme = sorted(meme.items(), key=lambda x: x[1].get("mention_spike", 0), reverse=True)
        return [{"symbol": sym, **data} for sym, data in sorted_meme[:top_n]]
