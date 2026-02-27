"""
News & fundamental signal crawler — no API key required.

Two free sources:
  1. yfinance .news   — Yahoo Finance news per ticker (already installed)
  2. Google News RSS  — https://news.google.com/rss/search?q=...
                        Parsed with stdlib xml.etree.ElementTree; no extra deps.

EDGAR insider-trade scraping is omitted: German stocks are regulated by BaFin
(not SEC), so Form 4 filings don't exist for XETRA-listed companies.

Output shape (unchanged from before):
    {
        symbol: {
            "articles":            [{"title", "description", "source", "published_at", "url"}, ...],
            "earnings_beat":       True | False | None,
            "earnings_surprise_pct": float | None,
            "insider_buys":        0,   # BaFin filings not yet scraped
            "insider_sells":       0,
            "institutional_adds":  0,
        }
    }
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import requests
from loguru import logger

from config.universe import TICKER_NAMES

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; clawbot/1.0; research bot)",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}

# Per-symbol article cache (5-minute TTL)
_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 300


# ---------------------------------------------------------------------------
# Source 1 — yfinance news
# ---------------------------------------------------------------------------

def _yf_news(symbol: str) -> list[dict]:
    """
    Fetch news via yfinance Ticker.news.
    Returns normalised article dicts.
    """
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol).news or []
        articles = []
        for item in raw:
            c = item.get("content", item)   # new yfinance wraps under "content"
            title   = c.get("title", "")
            summary = c.get("summary", c.get("description", ""))
            source  = (c.get("provider") or {}).get("displayName", "Yahoo Finance")
            pub     = c.get("pubDate", c.get("providerPublishTime", ""))
            url     = (c.get("canonicalUrl") or {}).get("url", c.get("link", ""))
            if title:
                articles.append({
                    "title":        title,
                    "description":  summary[:400],
                    "source":       source,
                    "published_at": str(pub),
                    "url":          url,
                })
        return articles
    except Exception as e:
        logger.debug(f"[News] yfinance news failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Source 2 — Google News RSS
# ---------------------------------------------------------------------------

_GNEWS_URL = "https://news.google.com/rss/search?q={query}&hl=de&gl=DE&ceid=DE:de"


def _google_news_rss(query: str, max_items: int = 10) -> list[dict]:
    """
    Fetch Google News RSS for a search query.
    Parses with stdlib xml; returns normalised article dicts.
    """
    url = _GNEWS_URL.format(query=quote_plus(query))
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        root    = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []
        articles = []
        for item in channel.findall("item")[:max_items]:
            title  = (item.findtext("title") or "").strip()
            desc   = (item.findtext("description") or "").strip()
            source = (item.findtext("source") or "Google News").strip()
            pub    = (item.findtext("pubDate") or "").strip()
            link   = (item.findtext("link") or "").strip()
            if title:
                articles.append({
                    "title":        title,
                    "description":  desc[:400],
                    "source":       source,
                    "published_at": pub,
                    "url":          link,
                })
        return articles
    except Exception as e:
        logger.debug(f"[News] Google News RSS failed for '{query}': {e}")
        return []


# ---------------------------------------------------------------------------
# Per-ticker fetch with caching
# ---------------------------------------------------------------------------

def fetch_news_for_ticker(symbol: str) -> list[dict]:
    """
    Fetch recent news for a single ticker.
    Tries yfinance first; falls back to Google News RSS if <3 articles returned.
    Results are cached for 5 minutes.
    """
    now = time.time()
    if symbol in _CACHE and now - _CACHE[symbol][0] < _CACHE_TTL:
        return _CACHE[symbol][1]

    articles = _yf_news(symbol)

    if len(articles) < 3:
        # Build a meaningful Google News query using the company name
        company = TICKER_NAMES.get(symbol, symbol.replace(".DE", ""))
        query   = f"{company} Aktie"
        gnews   = _google_news_rss(query, max_items=8)
        # Merge, deduplicate on title
        seen_titles = {a["title"] for a in articles}
        for a in gnews:
            if a["title"] not in seen_titles:
                articles.append(a)
                seen_titles.add(a["title"])
        time.sleep(0.3)   # polite gap between Google News requests

    _CACHE[symbol] = (now, articles)
    return articles


# ---------------------------------------------------------------------------
# Earnings heuristic
# ---------------------------------------------------------------------------

def _parse_earnings_from_articles(articles: list[dict]) -> dict:
    """Heuristic: look for beat/miss keywords in earnings-related articles."""
    beat_words = ["beat", "beats", "topped", "exceeded", "surpassed", "above estimates",
                  "übertroffen", "besser als erwartet", "schlägt erwartungen"]
    miss_words = ["miss", "missed", "below", "fell short", "disappointing", "cut guidance",
                  "verfehlt", "enttäuscht", "unter den erwartungen"]

    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".lower()
        if any(kw in text for kw in ["earnings", "eps", "quarterly", "ergebnis",
                                      "quartal", "gewinn", "umsatz"]):
            beat_count = sum(1 for w in beat_words if w in text)
            miss_count = sum(1 for w in miss_words if w in text)
            if beat_count > miss_count:
                m = re.search(r'(\d+\.?\d*)\s*%.*?(beat|above|übertroffen)', text)
                return {"beat": True,  "surprise_pct": float(m.group(1)) if m else None}
            elif miss_count > beat_count:
                m = re.search(r'(\d+\.?\d*)\s*%.*?(miss|below|verfehlt)', text)
                return {"beat": False, "surprise_pct": -float(m.group(1)) if m else None}

    return {"beat": None, "surprise_pct": None}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_news_signals(universe: list[str]) -> dict[str, dict]:
    """
    Fetch and score news signals for all tickers in universe.
    Returns context["news_signals"] dict.
    """
    results: dict[str, dict] = {}

    for symbol in universe:
        articles = fetch_news_for_ticker(symbol)
        if not articles:
            continue

        earnings = _parse_earnings_from_articles(articles)
        results[symbol] = {
            "articles":              articles,
            "earnings_beat":         earnings["beat"],
            "earnings_surprise_pct": earnings["surprise_pct"],
            "insider_buys":          0,   # BaFin filings not yet scraped
            "insider_sells":         0,
            "institutional_adds":    0,
        }

    logger.info(
        f"[News] Fetched signals for {len(results)}/{len(universe)} tickers "
        f"(yfinance + Google News RSS, no API key)"
    )
    return results
