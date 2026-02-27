"""
Reddit crawler — no API key required.

Uses Reddit's public JSON API (append .json to any Reddit URL).
Requires only a descriptive User-Agent; no OAuth, no PRAW, no account.

Rate limit: Reddit allows ~1 request/second per IP for unauthenticated reads.
We fetch each subreddit with a 1 s delay to stay polite.

Subreddits monitored:
  German-language: r/Aktien, r/mauerstrassenwetten, r/finanzen, r/de_investing
  English-language: r/wallstreetbets, r/stocks, r/investing, r/stockmarket

Ticker matching strips the .DE suffix so that "SAP.DE" is recognised from
raw posts that mention "SAP", "BMW", "BAYN", etc.
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from typing import Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_USER_AGENT = "clawbot/1.0 (paper trading research bot; contact via github)"
_HEADERS    = {"User-Agent": _USER_AGENT}
_BASE       = "https://www.reddit.com/r/{sub}/new.json"

SUBREDDITS = [
    # German-language
    "Aktien",
    "mauerstrassenwetten",  # German WSB equivalent
    "finanzen",
    "de_investing",
    # English-language (still relevant for European large-caps)
    "wallstreetbets",
    "stocks",
    "investing",
]

# Rolling baseline: tracks last 24 readings per ticker (base symbol, no .DE)
_BASELINE: dict[str, list[float]] = defaultdict(list)

_STOPWORDS = {
    "I", "A", "THE", "AND", "OR", "NOT", "IN", "ON", "AT", "TO",
    "FOR", "OF", "BY", "IS", "IT", "BE", "AS", "DO", "IF", "SO",
    "MY", "AM", "UP", "US", "WE", "DD", "OP", "OG", "CEO", "CFO",
    "ETF", "IPO", "GDP", "EPS", "ATH", "ATL", "DCA", "YOLO", "FUD",
    "FOMO", "IMO", "TBH", "TLDR", "SEC", "FDA", "FED", "NYSE", "NASDAQ",
    "EU", "ECB", "DAX", "DE", "AG", "SE", "KG", "GBP", "EUR", "USD",
    "MDAX", "SDAX", "RE", "FC",
}

# Upper-case words 2–6 chars (covers DAX tickers like SAP, BMW, BAYN, 1COV)
_TICKER_RE = re.compile(r'\b([A-Z0-9]{2,6})\b')


def _extract_tickers(text: str) -> list[str]:
    """Extract candidate ticker symbols from post text."""
    return [m for m in _TICKER_RE.findall(text) if m not in _STOPWORDS]


def _build_base_ticker_map(universe: list[str]) -> dict[str, str]:
    """
    Map base ticker (without .DE) → full ticker symbol.
    e.g. {"SAP": "SAP.DE", "BMW": "BMW.DE", "1COV": "1COV.DE"}
    """
    return {
        sym.replace(".DE", ""): sym
        for sym in universe
        if sym.endswith(".DE")
    }


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_subreddit(sub: str, limit: int = 100) -> list[dict]:
    """
    Fetch `limit` newest posts from a subreddit via the public JSON API.
    Returns list of {title, selftext} dicts, or [] on failure.
    """
    url = _BASE.format(sub=sub)
    try:
        resp = requests.get(
            url,
            params={"limit": limit, "raw_json": 1},
            headers=_HEADERS,
            timeout=12,
        )
        if resp.status_code == 404:
            logger.debug(f"[Reddit] r/{sub} not found (404)")
            return []
        if resp.status_code == 429:
            logger.warning(f"[Reddit] r/{sub} rate-limited (429) — backing off")
            time.sleep(5)
            return []
        resp.raise_for_status()
        data    = resp.json()
        posts   = data.get("data", {}).get("children", [])
        return [
            {
                "title":    p["data"].get("title", ""),
                "selftext": p["data"].get("selftext", "")[:500],
            }
            for p in posts
            if p.get("kind") == "t3"
        ]
    except Exception as e:
        logger.warning(f"[Reddit] r/{sub} fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_reddit_signals(
    universe: list[str] | None = None,
    post_limit: int = 100,
) -> dict[str, dict]:
    """
    Fetch recent Reddit posts and return meme/mention signals per ticker.

    universe: list of full ticker symbols (e.g. ["SAP.DE", "BMW.DE", ...])
              Used to build the base-ticker lookup so mentions of "SAP" map
              to "SAP.DE".  Falls back to raw base symbols if no universe given.

    Returns:
        {
            "SAP.DE": {
                "mention_count": 12,
                "mention_spike": 2.1,
                "sentiment":     0.35,
                "sources":       ["Aktien", "stocks"],
                "top_post":      "SAP earnings beat...",
            },
            ...
        }
    """
    from config.universe import DEFAULT_UNIVERSE
    symbols  = universe or DEFAULT_UNIVERSE
    base_map = _build_base_ticker_map(symbols)   # "SAP" → "SAP.DE"

    ticker_posts:   dict[str, list[str]] = defaultdict(list)
    ticker_sources: dict[str, set]       = defaultdict(set)

    for sub in SUBREDDITS:
        posts = _fetch_subreddit(sub, limit=post_limit)
        for post in posts:
            text    = f"{post['title']} {post['selftext']}"
            raw     = _extract_tickers(text.upper())
            matched = {base_map[b] for b in raw if b in base_map}
            for sym in matched:
                ticker_posts[sym].append(text)
                ticker_sources[sym].add(sub)
        time.sleep(1.0)   # polite delay between subreddits

    # Update rolling baseline and compute signals
    mention_counts = {sym: len(posts) for sym, posts in ticker_posts.items()}
    for sym, count in mention_counts.items():
        baseline = _BASELINE[sym]
        baseline.append(count)
        if len(baseline) > 24:
            baseline.pop(0)

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader: Optional[object] = SentimentIntensityAnalyzer()
    except ImportError:
        vader = None

    results: dict[str, dict] = {}
    for sym, posts in ticker_posts.items():
        if not posts:
            continue
        count    = mention_counts[sym]
        baseline = _BASELINE.get(sym, [count])
        avg_base = sum(baseline[:-1]) / max(1, len(baseline) - 1) if len(baseline) > 1 else count
        spike    = count / max(1, avg_base)

        sentiment = 0.0
        if vader and posts:
            scores    = [vader.polarity_scores(p)["compound"] for p in posts[:20]]  # type: ignore[attr-defined]
            sentiment = sum(scores) / len(scores)

        results[sym] = {
            "mention_count": count,
            "mention_spike": round(spike, 2),
            "sentiment":     round(sentiment, 3),
            "sources":       sorted(ticker_sources[sym]),
            "top_post":      posts[0][:200],
        }

    logger.info(
        f"[Reddit] {len(results)} tickers with mentions across "
        f"{len(SUBREDDITS)} subreddits (no API key)"
    )
    return results
