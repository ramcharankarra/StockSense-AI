"""
Risk Analytics Engine
======================
Computes all risk metrics for a given price series or portfolio.
All functions accept a pandas Series of closing prices or returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, VAR_CONFIDENCE
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Returns ───────────────────────────────────────────────────────────────────

def daily_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(prices: pd.Series) -> float:
    """Compound Annual Growth Rate (CAGR)."""
    rets = daily_returns(prices)
    if len(rets) == 0:
        return 0.0
    total = np.exp(rets.sum()) - 1
    years = len(rets) / TRADING_DAYS_PER_YEAR
    if years == 0:
        return 0.0
    return (1 + total) ** (1 / years) - 1


def cumulative_returns(prices: pd.Series) -> pd.Series:
    """Return indexed cumulative returns series (starts at 1.0)."""
    return (1 + daily_returns(prices)).cumprod()


# ── Volatility ────────────────────────────────────────────────────────────────

def volatility(prices: pd.Series) -> float:
    """Annualised daily-return standard deviation."""
    rets = daily_returns(prices)
    return float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


# ── Risk-adjusted returns ─────────────────────────────────────────────────────

def sharpe_ratio(prices: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe Ratio."""
    rets = daily_returns(prices)
    excess = rets.mean() * TRADING_DAYS_PER_YEAR - rf
    vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(excess / vol) if vol != 0 else 0.0


def sortino_ratio(prices: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Sortino Ratio using downside deviation."""
    rets = daily_returns(prices)
    excess = rets.mean() * TRADING_DAYS_PER_YEAR - rf
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(excess / downside) if downside != 0 else 0.0


# ── Drawdown ──────────────────────────────────────────────────────────────────

def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a negative fraction)."""
    cum = cumulative_returns(prices)
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(drawdown.min())


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum = cumulative_returns(prices)
    peak = cum.cummax()
    return (cum - peak) / peak


# ── Beta & Alpha ─────────────────────────────────────────────────────────────

def beta_alpha(
    stock_prices: pd.Series,
    market_prices: pd.Series,
    rf: float = RISK_FREE_RATE,
) -> tuple[float, float]:
    """
    Compute Beta and Jensen's Alpha relative to a market index.
    Returns (beta, alpha).
    """
    sr = daily_returns(stock_prices)
    mr = daily_returns(market_prices)
    common = sr.index.intersection(mr.index)
    sr, mr = sr.loc[common], mr.loc[common]
    if len(sr) < 10:
        return 1.0, 0.0
    cov = np.cov(sr, mr)
    b = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
    ann_stock = sr.mean() * TRADING_DAYS_PER_YEAR
    ann_market = mr.mean() * TRADING_DAYS_PER_YEAR
    a = ann_stock - rf - b * (ann_market - rf)
    return float(b), float(a)


# ── Value at Risk ─────────────────────────────────────────────────────────────

def value_at_risk(prices: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical VaR at given confidence level (negative value = loss)."""
    rets = daily_returns(prices)
    return float(np.percentile(rets, (1 - confidence) * 100))


def conditional_var(prices: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Conditional VaR / Expected Shortfall."""
    rets = daily_returns(prices)
    var = value_at_risk(prices, confidence)
    return float(rets[rets <= var].mean())


# ── Risk Score & Category ─────────────────────────────────────────────────────

def risk_score(prices: pd.Series, market_prices: Optional[pd.Series] = None) -> tuple[float, str]:
    """
    Compute a composite Risk Score (0 – 100).
    Higher = riskier.
    Returns (score: float, category: str).
    """
    vol = volatility(prices)
    mdd = abs(max_drawdown(prices))
    var = abs(value_at_risk(prices))
    sr = sharpe_ratio(prices)

    # Normalise each component to 0-25 range
    vol_score = min(vol / 0.8, 1.0) * 25          # 80 % vol → full score
    mdd_score = min(mdd / 0.5, 1.0) * 25          # 50 % drawdown → full score
    var_score = min(var / 0.05, 1.0) * 25         # 5 % daily VaR → full score
    sharpe_score = max(0, (2 - sr) / 2) * 25      # Sharpe 0 → 25, Sharpe 2+ → 0

    score = vol_score + mdd_score + var_score + sharpe_score

    if score < 33:
        category = "Low"
    elif score < 66:
        category = "Moderate"
    else:
        category = "High"

    return round(score, 1), category


# ── Full Summary ──────────────────────────────────────────────────────────────

def full_risk_report(
    prices: pd.Series,
    market_prices: Optional[pd.Series] = None,
) -> dict:
    """Return a dict with all risk metrics for display."""
    rets = daily_returns(prices)
    b, a = beta_alpha(prices, market_prices) if market_prices is not None else (None, None)
    score, category = risk_score(prices, market_prices)

    return {
        "daily_return_mean": float(rets.mean()),
        "annualized_return": annualized_return(prices),
        "volatility": volatility(prices),
        "sharpe_ratio": sharpe_ratio(prices),
        "sortino_ratio": sortino_ratio(prices),
        "max_drawdown": max_drawdown(prices),
        "beta": b,
        "alpha": a,
        "var_95": value_at_risk(prices),
        "cvar_95": conditional_var(prices),
        "risk_score": score,
        "risk_category": category,
    }
