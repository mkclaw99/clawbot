"""
Web Searcher — financial web research for the self-improvement loop.

Primary backend: Tavily (tavily-python) when TAVILY_API_KEY is set.
  - Structured JSON results; no HTML parsing; finance + news topic modes.
  - Free tier: 1 000 searches/month.

Fallback backend: DuckDuckGo HTML scrape (no API key needed).
  - Used when Tavily key is absent or the call fails.

LLM analysis: when a local LLM client is injected, it replaces keyword-based
  regime detection and ticker extraction with model-backed reasoning.  Falls
  back to keyword heuristics transparently if the LLM is offline.

Output: list[ResearchFinding] consumed by the Optimizer's WebResearcher.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import requests
from loguru import logger

from config.settings import WEB_SEARCH_MAX_REQUESTS, TAVILY_API_KEY

if TYPE_CHECKING:
    from core.llm_client import LLMClient

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
    "DAX", "ECB", "DE", "EU", "AG", "SE", "NV",
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
        return [
            {
                "title":   r.get("title", ""),
                "snippet": r.get("content", "")[:300],
                "url":     r.get("url", ""),
            }
            for r in resp.get("results", [])
        ]
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
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}
_DDG_URL = "https://html.duckduckgo.com/html/"


def _ddg_search(query: str) -> list[dict]:
    try:
        from bs4 import BeautifulSoup
        resp = requests.post(
            _DDG_URL,
            data={"q": query, "kl": "de-de"},
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

_CATEGORY_TOPIC: dict[str, Literal["general", "news", "finance"]] = {
    "trending": "news",
    "earnings": "finance",
    "regime":   "finance",
    "strategy": "general",
}

_REQUEST_DELAY_DDG    = 2.0
_REQUEST_DELAY_TAVILY = 0.5


def _search(query: str, category: str) -> list[dict]:
    topic   = _CATEGORY_TOPIC.get(category, "general")
    results = _tavily_search(query, topic=topic)
    backend = "Tavily"
    if not results:
        results = _ddg_search(query)
        backend = "DDG"
    logger.debug(f"[WebSearch/{backend}] '{query}' → {len(results)} results")
    return results


# ---------------------------------------------------------------------------
# Research queries — German / European market focus
# ---------------------------------------------------------------------------

RESEARCH_QUERIES: list[tuple[str, str]] = [
    ("DAX 40 Aktien höchstes Handelsvolumen heute",               "trending"),
    ("MDAX Aktien ungewöhnliches Volumen Deutschland heute",      "trending"),
    ("German company earnings beats DAX MDAX this week",          "earnings"),
    ("DAX stocks beat earnings estimates analyst upgrade Germany", "earnings"),
    ("German stock market outlook DAX bull bear ECB",             "regime"),
    ("ECB interest rate decision impact German equities DAX",     "regime"),
    ("mean reversion oversold German stocks technical analysis",   "strategy"),
    ("momentum stocks DAX breaking out high relative strength",   "strategy"),
    ("insider buying German stocks unusual options activity",      "strategy"),
    ("German stocks trending Reddit Aktien mauerstrassenwetten",  "trending"),
]


# ---------------------------------------------------------------------------
# WebResearcher
# ---------------------------------------------------------------------------

class WebResearcher:
    """
    Runs financial web research queries to discover trading insights.
    Called by the Optimizer after market close.

    Uses Tavily when TAVILY_API_KEY is set; otherwise falls back to DuckDuckGo.
    When an LLMClient is provided, uses the LLM for regime detection and
    ticker extraction; otherwise falls back to keyword heuristics.
    """

    def __init__(self, llm: "LLMClient | None" = None) -> None:
        self._request_count = 0
        self._run_start     = time.time()
        self._llm           = llm
        backend = "Tavily" if TAVILY_API_KEY else "DuckDuckGo (fallback)"
        llm_tag = f" | LLM: {llm._model if llm else 'disabled'}"
        logger.info(f"[WebResearcher] Backend: {backend}{llm_tag}")

    @property
    def backend(self) -> str:
        return "tavily" if TAVILY_API_KEY else "duckduckgo"

    def run_research(self, max_requests: int = WEB_SEARCH_MAX_REQUESTS) -> list[ResearchFinding]:
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

    # ------------------------------------------------------------------
    # Regime detection — LLM primary, keyword fallback
    # ------------------------------------------------------------------

    def extract_market_regime(self, findings: list[ResearchFinding]) -> str:
        """
        Infer market regime from research findings.
        Returns: "bull" | "bear" | "volatile" | "neutral" | "unknown"
        Uses LLM if available; falls back to keyword heuristics.
        """
        regime_findings = [f for f in findings if f.category == "regime"]
        if not regime_findings:
            return "unknown"

        if self._llm:
            result = self._llm_extract_regime(regime_findings)
            if result:
                return result

        return self._keyword_regime(regime_findings)

    def _llm_extract_regime(self, findings: list[ResearchFinding]) -> str | None:
        snippets = "\n".join(f"- {f.snippet}" for f in findings[:6])
        prompt = (
            "You are a German equity market analyst. Based on the following recent news snippets "
            "about the German/European stock market, classify the current market regime.\n\n"
            f"News snippets:\n{snippets}\n\n"
            "Respond with a JSON object only:\n"
            '{"regime": "bull" | "bear" | "volatile" | "neutral", '
            '"reasoning": "<one sentence max>"}'
        )
        result = self._llm.complete_json(
            [{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        if result and result.get("regime") in ("bull", "bear", "volatile", "neutral"):
            reasoning = result.get("reasoning", "")
            logger.info(f"[WebResearcher/LLM] Regime: {result['regime']} — {reasoning}")
            return result["regime"]
        return None

    def _keyword_regime(self, findings: list[ResearchFinding]) -> str:
        bull_words = ["bull", "rally", "breakout", "risk-on", "gains", "surge", "uptrend",
                      "Aufwärtstrend", "Hausse", "Kursanstieg"]
        bear_words = ["bear", "selloff", "correction", "risk-off", "decline", "downturn",
                      "Abwärtstrend", "Baisse", "Kursrückgang"]
        vol_words  = ["volatile", "uncertainty", "turbulent", "choppy", "whipsaw",
                      "Volatilität", "Unsicherheit"]

        bull_score = bear_score = vol_score = 0
        for f in findings:
            text = f.snippet.lower()
            bull_score += sum(1 for w in bull_words if w.lower() in text)
            bear_score += sum(1 for w in bear_words if w.lower() in text)
            vol_score  += sum(1 for w in vol_words  if w.lower() in text)

        if vol_score > max(bull_score, bear_score):
            return "volatile"
        if bull_score > bear_score:
            return "bull"
        if bear_score > bull_score:
            return "bear"
        return "unknown"

    # ------------------------------------------------------------------
    # Ticker extraction — LLM primary, keyword fallback
    # ------------------------------------------------------------------

    def extract_trending_tickers(self, findings: list[ResearchFinding]) -> list[str]:
        """
        Extract tickers from research findings.
        Prefers XETRA .DE tickers. Uses LLM if available.
        """
        if self._llm and findings:
            result = self._llm_extract_tickers(findings)
            if result:
                return result

        return self._keyword_tickers(findings)

    def _llm_extract_tickers(self, findings: list[ResearchFinding]) -> list[str] | None:
        snippets = "\n".join(f"- {f.snippet}" for f in findings[:12])
        prompt = (
            "You are a German stock market analyst. Extract stock tickers mentioned in the "
            "following news snippets. Focus on German XETRA-listed stocks (DAX 40, MDAX). "
            "Return tickers with .DE suffix where applicable (e.g. SAP.DE, BMW.DE, ALV.DE).\n\n"
            f"Snippets:\n{snippets}\n\n"
            "Respond with a JSON object only:\n"
            '{"tickers": ["SAP.DE", "BMW.DE", ...], "reasoning": "<one sentence>"}\n'
            "List up to 10 tickers, most relevant first. Only include tickers you are confident about."
        )
        result = self._llm.complete_json(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        if result and isinstance(result.get("tickers"), list):
            tickers = [t for t in result["tickers"] if isinstance(t, str) and len(t) >= 2]
            logger.info(f"[WebResearcher/LLM] Extracted tickers: {tickers[:10]}")
            return tickers[:10]
        return None

    def _keyword_tickers(self, findings: list[ResearchFinding]) -> list[str]:
        from collections import Counter
        ticker_counts: Counter = Counter()
        for f in findings:
            if f.ticker:
                ticker_counts[f.ticker] += f.relevance_score
        trending = [t for t, score in ticker_counts.most_common(20) if score >= 0.5]
        logger.info(f"[WebSearch] Trending tickers (keyword): {trending[:10]}")
        return trending

    # ------------------------------------------------------------------
    # LLM synthesis — optional rich insight summary
    # ------------------------------------------------------------------

    def synthesize_insights(self, findings: list[ResearchFinding]) -> str:
        """
        Ask the LLM to summarise key actionable insights from all findings.
        Returns an empty string if LLM is unavailable.
        """
        if not self._llm or not findings:
            return ""

        snippets = "\n".join(f"- [{f.category}] {f.snippet}" for f in findings[:15])
        prompt = (
            "You are a quantitative analyst specialising in German equities (DAX, MDAX). "
            "Summarise the following research findings into 3–5 bullet points of actionable "
            "trading insights relevant to momentum, mean-reversion, and macro strategies.\n\n"
            f"Findings:\n{snippets}\n\n"
            "Be concise. Each bullet ≤ 20 words."
        )
        reply = self._llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        if reply:
            logger.info(f"[WebResearcher/LLM] Insights synthesised ({len(reply)} chars)")
        return reply or ""

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _relevance_score(self, title: str, snippet: str, category: str) -> float:
        text = f"{title} {snippet}".lower()
        finance_words = [
            "stock", "shares", "aktie", "dax", "mdax", "nasdaq", "nyse",
            "ticker", "earnings", "ergebnis", "revenue", "umsatz",
            "market", "markt", "trading", "investor", "analyst", "options",
            "volume", "volumen", "momentum", "technical", "chart", "sector",
        ]
        cat_words = {
            "trending":  ["volume", "volumen", "trending", "surge", "spike", "momentum"],
            "earnings":  ["earnings", "ergebnis", "eps", "beat", "miss", "guidance", "revenue"],
            "regime":    ["market", "markt", "bull", "bear", "dax", "index", "breadth", "ecb", "ezb"],
            "strategy":  ["technical", "analysis", "indicator", "signal", "strategy", "strategie"],
        }
        finance_hits = sum(1 for w in finance_words if w in text)
        cat_hits     = sum(1 for w in cat_words.get(category, []) if w in text)
        score        = min(1.0, (finance_hits * 0.08) + (cat_hits * 0.15))
        return round(score, 2)
