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


# ── Premium SaaS KPI & Chart Styling Helpers ─────────────────────────────────

def render_premium_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    status_color: str = "blue",
    icon: str = "",
) -> str:
    """Render a premium styled HTML card for FinTech KPIs with shadows and hover scale transitions."""
    border_color_map = {
        "blue": "#2563EB",
        "cyan": "#06B6D4",
        "green": "#10B981",
        "orange": "#F59E0B",
        "red": "#EF4444",
        "grey": "#e5e7eb",
    }
    
    border_color = border_color_map.get(status_color, "#e5e7eb")
    icon_html = f"<span style='font-size: 1.25rem;'>{icon}</span>" if icon else ""
    
    return f"""
    <div class="stat-card" style="
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 5px solid {border_color};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        margin-bottom: 0.8rem;
    ">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
        <span style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280;">{label}</span>
        {icon_html}
      </div>
      <div style="font-size: 1.65rem; font-weight: 800; color: #111827; margin: 0.25rem 0;">{value}</div>
      <div style="font-size: 0.8rem; color: #6b7280; font-weight: 500;">{subtext}</div>
    </div>
    """


def apply_modern_theme(fig, title: str = "", height: int = 400, show_legend: bool = True):
    """Apply modern light SaaS/Bloomberg theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#111827", family="Inter, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#374151"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        ) if show_legend else dict(visible=False),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        linecolor="#e5e7eb",
        tickfont=dict(size=10, color="#6b7280"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        linecolor="#e5e7eb",
        tickfont=dict(size=10, color="#6b7280"),
    )
    return fig
