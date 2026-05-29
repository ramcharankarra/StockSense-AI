"""
News Service
=============
Fetches financial news headlines from NewsAPI (with fallback to RSS feeds).
Falls back to yfinance news if no API key is configured.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import urllib.request
import json

import yfinance as yf

from config import NEWS_API_KEY, MAX_NEWS_ARTICLES, NEWS_LOOKBACK_DAYS
from utils.logger import get_logger

logger = get_logger(__name__)


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def _fetch_newsapi(symbol: str, company_name: str = "") -> list[dict]:
    """Fetch from NewsAPI.org if API key available."""
    if not NEWS_API_KEY:
        return []
    query = company_name or symbol
    from_date = (datetime.today() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={urllib.parse.quote(query)}&from={from_date}"
        f"&sortBy=publishedAt&language=en&pageSize={MAX_NEWS_ARTICLES}"
        f"&apiKey={NEWS_API_KEY}"
    )
    try:
        import urllib.parse
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "description": a.get("description", ""),
            }
            for a in articles
        ]
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []


# ── yfinance news fallback ────────────────────────────────────────────────────

def _fetch_yfinance_news(symbol: str) -> list[dict]:
    """Use yfinance's built-in news feed as a fallback source."""
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
        articles = []
        for item in raw[:MAX_NEWS_ARTICLES]:
            content = item.get("content", {})
            articles.append({
                "title": content.get("title", item.get("title", "")),
                "source": content.get("provider", {}).get("displayName", "Yahoo Finance"),
                "url": content.get("canonicalUrl", {}).get("url", ""),
                "published_at": content.get("pubDate", ""),
                "description": content.get("summary", ""),
            })
        return articles
    except Exception as exc:
        logger.warning("yfinance news fetch failed for %s: %s", symbol, exc)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_news(symbol: str, company_name: str = "") -> list[dict]:
    """
    Fetch financial news for a stock symbol.
    Priority: NewsAPI → yfinance news
    Returns a list of article dicts with keys:
      title, source, url, published_at, description
    """
    symbol = symbol.strip().upper()
    articles = _fetch_newsapi(symbol, company_name)
    if not articles:
        articles = _fetch_yfinance_news(symbol)

    # Deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        key = a.get("title", "")[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    logger.info("Fetched %d news articles for %s", len(unique), symbol)
    return unique


def get_market_news(topics: list[str] = None) -> list[dict]:
    """
    Fetch general market news (not stock-specific).
    Uses NewsAPI with finance-related keywords.
    """
    query = " OR ".join(topics or ["stock market", "S&P 500", "Federal Reserve", "earnings"])
    return _fetch_newsapi("market", query) or _fetch_yfinance_news("SPY")
