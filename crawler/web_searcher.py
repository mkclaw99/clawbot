"""
Web Searcher — financial web research for the self-improvement loop.

Primary backend: Tavily (tavily-python) when TAVILY_API_KEY is set.
  - Structured JSON results; no HTML parsing; finance + news topic modes.
  - Free tier: 1 000 searches/month.

Fallback backend: DuckDuckGo HTML scrape (no API key needed).
  - Used when Tavily key is absent or the call fails.

Output: list[ResearchFinding] consumed by the Optimizer's WebResearcher.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import requests
from loguru import logger

from config.settings import WEB_SEARCH_MAX_REQUESTS, TAVILY_API_KEY

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResearchFinding:
    query: str
    ticker: str | None          # extracted ticker if applicable
    snippet: str                # relevant text snippet
    source_url: str
    relevance_score: float      # 0.0–1.0
    category: str               # "trending" | "earnings" | "regime" | "strategy"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r'\b([A-Z]{2,5})\b')
_STOPWORDS  = {
    "I", "A", "THE", "AND", "OR", "NOT", "IN", "ON", "AT", "TO",
    "FOR", "OF", "BY", "IS", "IT", "BE", "AS", "DO", "IF", "SO",
    "MY", "AM", "UP", "US", "WE", "DD", "CEO", "CFO", "ETF",
    "IPO", "GDP", "EPS", "ATH", "SEC", "FDA", "FED", "NYSE", "NASDAQ",
    "AI", "ML", "Q1", "Q2", "Q3", "Q4", "YTD", "QOQ", "YOY",
}


def _extract_tickers(text: str) -> list[str]:
    return [m for m in _TICKER_RE.findall(text) if m not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Backend 1 — Tavily
# ---------------------------------------------------------------------------

def _tavily_search(
    query: str,
    topic: Literal["general", "news", "finance"] = "finance",
    max_results: int = 8,
) -> list[dict]:
    """
    Search via Tavily API. Returns normalised result dicts.
    Returns [] if the key is absent, the package is missing, or the call fails.
    """
    if not TAVILY_API_KEY:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(
            query=query,
            topic=topic,
            search_depth="basic",
            max_results=max_results,
            time_range="week",
            include_answer=False,
        )
        results = []
        for r in resp.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "snippet": r.get("content", "")[:300],
                "url":     r.get("url", ""),
            })
        return results
    except Exception as e:
        logger.debug(f"[WebSearch/Tavily] '{query}': {e}")
        return []


# ---------------------------------------------------------------------------
# Backend 2 — DuckDuckGo (fallback)
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_DDG_URL = "https://html.duckduckgo.com/html/"


def _ddg_search(query: str) -> list[dict]:
    """DuckDuckGo HTML fallback. Returns [] on any error."""
    try:
        from bs4 import BeautifulSoup
        resp = requests.post(
            _DDG_URL,
            data={"q": query, "kl": "us-en"},
            headers=_HEADERS,
            timeout=12,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result"):
            title_el   = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            url_el     = result.select_one(".result__url")
            if title_el and snippet_el:
                results.append({
                    "title":   title_el.get_text(strip=True),
                    "snippet": snippet_el.get_text(strip=True),
                    "url":     url_el.get_text(strip=True) if url_el else "",
                })
        return results[:8]
    except Exception as e:
        logger.debug(f"[WebSearch/DDG] '{query}': {e}")
        return []


# ---------------------------------------------------------------------------
# Unified search — Tavily first, DDG fallback
# ---------------------------------------------------------------------------

# Map research category → best Tavily topic
_CATEGORY_TOPIC: dict[str, Literal["general", "news", "finance"]] = {
    "trending": "news",
    "earnings": "finance",
    "regime":   "finance",
    "strategy": "general",
}

_REQUEST_DELAY_DDG    = 2.0   # polite delay when using DDG
_REQUEST_DELAY_TAVILY = 0.5   # Tavily is an API, smaller delay is fine


def _search(query: str, category: str) -> list[dict]:
    """Try Tavily; fall back to DDG."""
    topic   = _CATEGORY_TOPIC.get(category, "general")
    results = _tavily_search(query, topic=topic)
    backend = "Tavily"
    if not results:
        results = _ddg_search(query)
        backend = "DDG"
    logger.debug(f"[WebSearch/{backend}] '{query}' → {len(results)} results")
    return results


# ---------------------------------------------------------------------------
# Research queries
# ---------------------------------------------------------------------------

RESEARCH_QUERIES: list[tuple[str, str]] = [
    ("stocks trending highest volume today",                "trending"),
    ("most active stocks unusual volume today",             "trending"),
    ("earnings beats surprises this week",                  "earnings"),
    ("stocks beat earnings estimates analyst upgrade",      "earnings"),
    ("stock market outlook bull bear indicator",            "regime"),
    ("S&P 500 market breadth advance decline ratio",        "regime"),
    ("mean reversion oversold stocks technical analysis",   "strategy"),
    ("momentum stocks breaking out high relative strength", "strategy"),
    ("insider buying unusual options activity stocks",      "strategy"),
    ("meme stocks trending reddit wallstreetbets",          "trending"),
]


# ---------------------------------------------------------------------------
# WebResearcher
# ---------------------------------------------------------------------------

class WebResearcher:
    """
    Runs financial web research queries to discover trading insights.
    Called by the Optimizer every 6 hours (after market close).

    Uses Tavily when TAVILY_API_KEY is set; otherwise falls back to DuckDuckGo.
    """

    def __init__(self) -> None:
        self._request_count = 0
        self._run_start     = time.time()
        backend = "Tavily" if TAVILY_API_KEY else "DuckDuckGo (fallback — set TAVILY_API_KEY)"
        logger.info(f"[WebResearcher] Backend: {backend}")

    @property
    def backend(self) -> str:
        return "tavily" if TAVILY_API_KEY else "duckduckgo"

    def run_research(self, max_requests: int = WEB_SEARCH_MAX_REQUESTS) -> list[ResearchFinding]:
        """
        Run all research queries and return ResearchFindings.
        Stops when max_requests is reached.
        """
        self._request_count = 0
        self._run_start     = time.time()
        findings: list[ResearchFinding] = []
        delay = _REQUEST_DELAY_TAVILY if TAVILY_API_KEY else _REQUEST_DELAY_DDG

        for query, category in RESEARCH_QUERIES:
            if self._request_count >= max_requests:
                logger.info(f"[WebSearch] Request cap ({max_requests}) reached — stopping")
                break

            logger.info(f"[WebSearch] Querying: '{query}'")
            results = _search(query, category)
            self._request_count += 1

            for r in results:
                full_text = f"{r['title']} {r['snippet']}"
                tickers   = _extract_tickers(full_text)
                score     = self._relevance_score(r["title"], r["snippet"], category)

                if score < 0.2:
                    continue

                findings.append(ResearchFinding(
                    query=query,
                    ticker=tickers[0] if tickers else None,
                    snippet=f"{r['title']} — {r['snippet'][:200]}",
                    source_url=r["url"],
                    relevance_score=score,
                    category=category,
                ))

            time.sleep(delay)

        logger.info(
            f"[WebSearch] Research complete: {len(findings)} findings "
            f"from {self._request_count} queries in "
            f"{time.time() - self._run_start:.1f}s"
        )
        return findings

    def search(self, query: str, category: str = "general", max_results: int = 5) -> list[ResearchFinding]:
        """
        One-off search for a specific query (e.g. from CLI or skill).
        Returns ResearchFindings sorted by relevance.
        """
        results  = _search(query, category)
        findings = []
        for r in results[:max_results]:
            full_text = f"{r['title']} {r['snippet']}"
            tickers   = _extract_tickers(full_text)
            score     = self._relevance_score(r["title"], r["snippet"], category)
            findings.append(ResearchFinding(
                query=query,
                ticker=tickers[0] if tickers else None,
                snippet=f"{r['title']} — {r['snippet'][:200]}",
                source_url=r["url"],
                relevance_score=score,
                category=category,
            ))
        return sorted(findings, key=lambda f: f.relevance_score, reverse=True)

    def extract_trending_tickers(self, findings: list[ResearchFinding]) -> list[str]:
        """
        From research findings, extract tickers appearing multiple times
        or with high relevance — candidates to add to the trading universe.
        """
        from collections import Counter
        ticker_counts: Counter = Counter()
        for f in findings:
            if f.ticker:
                ticker_counts[f.ticker] += f.relevance_score

        trending = [t for t, score in ticker_counts.most_common(20) if score >= 0.5]
        logger.info(f"[WebSearch] Trending tickers from web: {trending[:10]}")
        return trending

    def extract_market_regime(self, findings: list[ResearchFinding]) -> str:
        """
        Infer market regime from regime-category findings.
        Returns: "bull" | "bear" | "volatile" | "unknown"
        """
        regime_findings = [f for f in findings if f.category == "regime"]
        if not regime_findings:
            return "unknown"

        bull_words = ["bull", "rally", "breakout", "risk-on", "gains", "surge", "uptrend"]
        bear_words = ["bear", "selloff", "correction", "risk-off", "decline", "downturn"]
        vol_words  = ["volatile", "uncertainty", "turbulent", "choppy", "whipsaw"]

        bull_score = bear_score = vol_score = 0
        for f in regime_findings:
            text = f.snippet.lower()
            bull_score += sum(1 for w in bull_words if w in text)
            bear_score += sum(1 for w in bear_words if w in text)
            vol_score  += sum(1 for w in vol_words  if w in text)

        if vol_score > max(bull_score, bear_score):
            return "volatile"
        if bull_score > bear_score:
            return "bull"
        if bear_score > bull_score:
            return "bear"
        return "unknown"

    def _relevance_score(self, title: str, snippet: str, category: str) -> float:
        """Score a result's relevance to financial trading (0.0–1.0)."""
        text = f"{title} {snippet}".lower()
        finance_words = [
            "stock", "shares", "nasdaq", "nyse", "ticker", "earnings",
            "revenue", "market", "trading", "investor", "analyst", "options",
            "volume", "momentum", "technical", "chart", "sector",
        ]
        cat_words = {
            "trending":  ["volume", "trending", "surge", "spike", "momentum"],
            "earnings":  ["earnings", "eps", "beat", "miss", "guidance", "revenue"],
            "regime":    ["market", "bull", "bear", "s&p", "index", "breadth"],
            "strategy":  ["technical", "analysis", "indicator", "signal", "strategy"],
        }
        finance_hits = sum(1 for w in finance_words if w in text)
        cat_hits     = sum(1 for w in cat_words.get(category, []) if w in text)
        score        = min(1.0, (finance_hits * 0.08) + (cat_hits * 0.15))
        return round(score, 2)
