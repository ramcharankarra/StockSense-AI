"""
Shared Helpers
==============
Utility functions used across multiple modules.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ── Number formatting ─────────────────────────────────────────────────────────

def fmt_currency(value: float, symbol: str = "$") -> str:
    """Format a float as a currency string."""
    if abs(value) >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.2f}K"
    return f"{symbol}{value:.2f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_large(value: float) -> str:
    """Format large numbers with B/M/K suffixes."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


# ── Date helpers ─────────────────────────────────────────────────────────────

def date_range(period: str) -> tuple[datetime, datetime]:
    """Convert a period string to (start, end) datetime pair."""
    end = datetime.today()
    mapping = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
    }
    days = mapping.get(period, 365)
    return end - timedelta(days=days), end


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names and drop NaN rows."""
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    return df


def safe_divide(a: float | np.ndarray, b: float | np.ndarray, fallback: float = 0.0):
    """Divide without ZeroDivisionError."""
    if isinstance(b, (int, float)):
        return a / b if b != 0 else fallback
    result = np.where(b != 0, a / b, fallback)
    return result


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid_ticker(ticker: str) -> bool:
    """Basic ticker symbol validation."""
    return bool(re.match(r"^[A-Za-z.\-^]{1,10}$", ticker.strip()))


def sanitize_ticker(ticker: str) -> str:
    """Strip whitespace and uppercase a ticker symbol."""
    return ticker.strip().upper()


# ── Hashing (non-auth, general purpose) ──────────────────────────────────────

def hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── Signal / color helpers ────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "Strong Buy": "#1a6b3c",
    "Buy": "#27ae60",
    "Hold": "#f39c12",
    "Sell": "#e74c3c",
    "Strong Sell": "#922b21",
}

RISK_COLORS = {
    "Low": "#27ae60",
    "Moderate": "#f39c12",
    "High": "#e74c3c",
}


def signal_color(signal: str) -> str:
    return SIGNAL_COLORS.get(signal, "#333333")


def risk_color(category: str) -> str:
    return RISK_COLORS.get(category, "#333333")


def returns_color(value: float) -> str:
    return "#27ae60" if value >= 0 else "#e74c3c"
