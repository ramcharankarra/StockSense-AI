"""
Trading Signal Engine
======================
Combines technical indicators, risk metrics, ML predictions, and sentiment
to generate actionable trading signals with confidence scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from analysis.technical_indicators import compute_all
from analysis.risk_analytics import volatility, sharpe_ratio, max_drawdown
from utils.logger import get_logger

logger = get_logger(__name__)

SIGNALS = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]


def _score_technical(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Score from -2 to +2 based on technical indicators. Returns (score, reasons)."""
    df = compute_all(df)
    last = df.iloc[-1]
    reasons = []
    score = 0.0

    # RSI
    rsi = last.get("rsi", 50)
    if rsi < 30:
        score += 1.5
        reasons.append(f"RSI={rsi:.1f} — Oversold (Bullish)")
    elif rsi > 70:
        score -= 1.5
        reasons.append(f"RSI={rsi:.1f} — Overbought (Bearish)")
    else:
        reasons.append(f"RSI={rsi:.1f} — Neutral")

    # MACD
    macd = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    if macd > macd_sig:
        score += 1.0
        reasons.append("MACD above signal — Bullish crossover")
    else:
        score -= 1.0
        reasons.append("MACD below signal — Bearish crossover")

    # Price vs. SMA 50
    close = last.get("close", last.get("Close", 0))
    sma50 = last.get("sma_50", close)
    if close > sma50:
        score += 0.5
        reasons.append("Price above 50-SMA — Uptrend")
    else:
        score -= 0.5
        reasons.append("Price below 50-SMA — Downtrend")

    # Bollinger Bands
    bb_upper = last.get("bb_upper", float("inf"))
    bb_lower = last.get("bb_lower", 0)
    if close < bb_lower:
        score += 0.5
        reasons.append("Price at lower Bollinger Band — Potential bounce")
    elif close > bb_upper:
        score -= 0.5
        reasons.append("Price at upper Bollinger Band — Potential reversal")

    # Stochastic
    stoch_k = last.get("stoch_k", 50)
    if stoch_k < 20:
        score += 0.5
        reasons.append(f"Stochastic %K={stoch_k:.1f} — Oversold")
    elif stoch_k > 80:
        score -= 0.5
        reasons.append(f"Stochastic %K={stoch_k:.1f} — Overbought")

    return score, reasons


def _score_ml(current_price: float, predicted_price: Optional[float]) -> tuple[float, list[str]]:
    """Score from -2 to +2 based on ML prediction direction and magnitude."""
    if predicted_price is None:
        return 0.0, ["ML prediction unavailable"]
    change_pct = (predicted_price - current_price) / current_price * 100
    if change_pct > 3:
        return 2.0, [f"ML projects +{change_pct:.2f}% gain — Strongly Bullish"]
    if change_pct > 1:
        return 1.0, [f"ML projects +{change_pct:.2f}% gain — Bullish"]
    if change_pct < -3:
        return -2.0, [f"ML projects {change_pct:.2f}% decline — Strongly Bearish"]
    if change_pct < -1:
        return -1.0, [f"ML projects {change_pct:.2f}% decline — Bearish"]
    return 0.0, [f"ML projects {change_pct:.2f}% — Neutral"]


def _score_sentiment(sentiment_score: float) -> tuple[float, list[str]]:
    """Score from -1 to +1 based on sentiment."""
    s = sentiment_score
    if s >= 0.2:
        return 1.0, [f"Sentiment strongly positive ({s:.3f})"]
    if s >= 0.05:
        return 0.5, [f"Sentiment mildly positive ({s:.3f})"]
    if s <= -0.2:
        return -1.0, [f"Sentiment strongly negative ({s:.3f})"]
    if s <= -0.05:
        return -0.5, [f"Sentiment mildly negative ({s:.3f})"]
    return 0.0, [f"Sentiment neutral ({s:.3f})"]


def _score_risk(prices: pd.Series) -> tuple[float, list[str]]:
    """Penalise signal strength for very high-risk stocks."""
    vol = volatility(prices)
    mdd = abs(max_drawdown(prices))
    reasons = []
    score = 0.0
    if vol > 0.5:
        score -= 0.5
        reasons.append(f"High volatility ({vol:.1%}) — Risk penalty")
    if mdd > 0.4:
        score -= 0.5
        reasons.append(f"Large max drawdown ({mdd:.1%}) — Risk penalty")
    return score, reasons


# ── Main signal generator ─────────────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    predicted_price: Optional[float] = None,
    sentiment_score: float = 0.0,
) -> dict:
    """
    Generate a composite trading signal.

    Returns:
        {
            signal: str,           # 'Strong Buy' | 'Buy' | 'Hold' | 'Sell' | 'Strong Sell'
            confidence: float,     # 0.0 – 1.0
            composite_score: float,
            reasons: list[str],
        }
    """
    current_price = float(df["close"].iloc[-1])

    tech_score, tech_reasons = _score_technical(df)
    ml_score, ml_reasons = _score_ml(current_price, predicted_price)
    sent_score, sent_reasons = _score_sentiment(sentiment_score)
    risk_score, risk_reasons = _score_risk(df["close"])

    # Weighted composite: Technical 40%, ML 35%, Sentiment 15%, Risk 10%
    composite = 0.40 * tech_score + 0.35 * ml_score + 0.15 * sent_score + 0.10 * risk_score

    # Map score → signal
    # composite range: approximately [-4, +4]
    if composite >= 1.5:
        signal = "Strong Buy"
    elif composite >= 0.5:
        signal = "Buy"
    elif composite <= -1.5:
        signal = "Strong Sell"
    elif composite <= -0.5:
        signal = "Sell"
    else:
        signal = "Hold"

    confidence = min(abs(composite) / 4.0, 1.0)

    all_reasons = tech_reasons + ml_reasons + sent_reasons + risk_reasons

    logger.info("Signal for score=%.2f → %s (conf=%.2f)", composite, signal, confidence)
    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "composite_score": round(composite, 4),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "reasons": all_reasons,
        "tech_score": tech_score,
        "ml_score": ml_score,
        "sentiment_score": sent_score,
        "risk_score": risk_score,
    }
