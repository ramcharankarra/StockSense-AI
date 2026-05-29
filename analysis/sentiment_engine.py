"""
Sentiment Analysis Engine
==========================
Fetches financial news and scores each article using VADER and TextBlob.
Returns aggregated sentiment scores and trend labels.
"""

from __future__ import annotations

from typing import Optional
import statistics

import nltk
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from data_ingestion.news_service import fetch_news
from database.db_manager import execute_write, execute_query, get_stock_id, upsert_stock
from utils.logger import get_logger

logger = get_logger(__name__)

# ── NLTK setup ────────────────────────────────────────────────────────────────
try:
    _sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()


# ── Article scorer ────────────────────────────────────────────────────────────

def _score_article(text: str) -> dict:
    """Score a single text string with VADER and TextBlob."""
    text = (text or "").strip()
    if not text:
        return {"compound": 0.0, "textblob": 0.0, "label": "neutral", "score": 0.0}

    vader_scores = _sia.polarity_scores(text)
    compound = vader_scores["compound"]       # -1 to +1

    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity     # -1 to +1

    # Ensemble: 60 % VADER + 40 % TextBlob
    ensemble = 0.6 * compound + 0.4 * tb_polarity

    if ensemble >= 0.05:
        label = "positive"
    elif ensemble <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": compound,
        "textblob": tb_polarity,
        "label": label,
        "score": round(ensemble, 4),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_sentiment(symbol: str, company_name: str = "") -> dict:
    """
    Fetch news for `symbol` and compute aggregated sentiment.

    Returns:
        {
            articles: list[dict],
            overall_score: float,       # -1 to +1
            overall_label: str,
            positive_pct: float,
            neutral_pct: float,
            negative_pct: float,
            score_trend: list[float],   # daily rolling avg
        }
    """
    symbol = symbol.strip().upper()
    articles = fetch_news(symbol, company_name)

    if not articles:
        logger.warning("No news articles found for %s", symbol)
        return _empty_sentiment()

    scored = []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}"
        scores = _score_article(text)
        scored.append({**art, **scores})

    scores_list = [a["score"] for a in scored]
    labels = [a["label"] for a in scored]

    overall_score = statistics.mean(scores_list) if scores_list else 0.0
    n = len(labels)
    pos = labels.count("positive") / n * 100
    neu = labels.count("neutral") / n * 100
    neg = labels.count("negative") / n * 100

    if overall_score >= 0.05:
        overall_label = "Positive"
    elif overall_score <= -0.05:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    _cache_sentiments(symbol, scored)

    return {
        "articles": scored,
        "overall_score": round(overall_score, 4),
        "overall_label": overall_label,
        "positive_pct": round(pos, 1),
        "neutral_pct": round(neu, 1),
        "negative_pct": round(neg, 1),
        "score_trend": scores_list,
    }


def _empty_sentiment() -> dict:
    return {
        "articles": [],
        "overall_score": 0.0,
        "overall_label": "Neutral",
        "positive_pct": 0.0,
        "neutral_pct": 100.0,
        "negative_pct": 0.0,
        "score_trend": [],
    }


# ── DB caching ────────────────────────────────────────────────────────────────

def _cache_sentiments(symbol: str, scored: list[dict]) -> None:
    stock_id = get_stock_id(symbol)
    if not stock_id:
        return
    for art in scored:
        try:
            execute_write(
                """INSERT OR IGNORE INTO sentiments
                   (stock_id, source, headline, sentiment_score, sentiment_label,
                    compound_vader, published_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    stock_id,
                    art.get("source", ""),
                    art.get("title", "")[:500],
                    art.get("score", 0.0),
                    art.get("label", "neutral"),
                    art.get("compound", 0.0),
                    art.get("published_at", ""),
                ),
            )
        except Exception:
            pass


def get_cached_sentiment(symbol: str, limit: int = 20) -> list[dict]:
    """Load recent cached sentiment records from DB."""
    stock_id = get_stock_id(symbol)
    if not stock_id:
        return []
    return execute_query(
        "SELECT * FROM sentiments WHERE stock_id=? ORDER BY fetched_at DESC LIMIT ?",
        (stock_id, limit),
    )


# ── Sentiment signals ─────────────────────────────────────────────────────────

def sentiment_signal(overall_score: float) -> str:
    """Convert numeric score to trading signal."""
    if overall_score >= 0.2:
        return "Strong Positive"
    if overall_score >= 0.05:
        return "Positive"
    if overall_score <= -0.2:
        return "Strong Negative"
    if overall_score <= -0.05:
        return "Negative"
    return "Neutral"
