"""
StockSense AI – Home Dashboard (00_home.py)
===========================================
Market overview hub: KPI cards, indices chart, top movers,
market news feed, and quick-access navigation grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from auth.session_manager import get_session, require_login
from config import DEFAULT_WATCHLIST, MARKET_INDICES
from data_ingestion.news_service import get_market_news

# ── Page configuration ────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="Home | StockSense AI",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ── Auth guard ────────────────────────────────────────────────────────────────

user = require_login()

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Base resets */
        [data-testid="stAppViewContainer"] { background: #ffffff; }
        [data-testid="stSidebar"] { background: #f8faff; }

        /* Page header */
        .ss-page-header {
            padding: 1.2rem 0 0.4rem 0;
            border-bottom: 2px solid #1a56db;
            margin-bottom: 1.5rem;
        }
        .ss-page-title {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin: 0;
        }
        .ss-page-subtitle {
            font-size: 0.95rem;
            color: #6b7280;
            margin-top: 0.2rem;
        }

        /* Welcome card */
        .ss-welcome-card {
            background: linear-gradient(135deg, #1a56db 0%, #1e40af 100%);
            border-radius: 12px;
            padding: 1.1rem 1.5rem;
            color: #ffffff;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .ss-welcome-name {
            font-size: 1.25rem;
            font-weight: 700;
        }
        .ss-welcome-sub {
            font-size: 0.85rem;
            opacity: 0.85;
        }

        /* Section headings */
        .ss-section-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: #111827;
            border-left: 4px solid #1a56db;
            padding-left: 0.6rem;
            margin: 1.4rem 0 0.8rem 0;
        }

        /* Quick-access nav cards */
        .ss-nav-card {
            background: #ffffff;
            border: 1.5px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.1rem 0.9rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 0.6rem;
        }
        .ss-nav-card:hover {
            border-color: #1a56db;
            box-shadow: 0 4px 12px rgba(26, 86, 219, 0.12);
            transform: translateY(-2px);
        }
        .ss-nav-card-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
        .ss-nav-card-title {
            font-size: 0.82rem;
            font-weight: 600;
            color: #111827;
        }
        .ss-nav-card-desc {
            font-size: 0.72rem;
            color: #6b7280;
            margin-top: 0.2rem;
        }

        /* Mover rows */
        .ss-mover-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.8rem;
            border-bottom: 1px solid #f3f4f6;
            font-size: 0.88rem;
        }
        .ss-mover-ticker { font-weight: 700; color: #111827; }
        .ss-mover-price  { color: #374151; }
        .ss-badge-up   { color: #059669; font-weight: 600; }
        .ss-badge-down { color: #dc2626; font-weight: 600; }

        /* News item */
        .ss-news-item {
            padding: 0.55rem 0.2rem;
            border-bottom: 1px solid #f3f4f6;
        }
        .ss-news-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #111827;
            line-height: 1.4;
        }
        .ss-news-meta {
            font-size: 0.72rem;
            color: #9ca3af;
            margin-top: 0.2rem;
        }

        /* Metric delta overrides */
        [data-testid="stMetricDelta"] svg { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_number(val, prefix="", suffix="", decimals=2) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e12:
            return f"{prefix}{v/1e12:.2f}T{suffix}"
        if abs(v) >= 1e9:
            return f"{prefix}{v/1e9:.2f}B{suffix}"
        if abs(v) >= 1e6:
            return f"{prefix}{v/1e6:.2f}M{suffix}"
        return f"{prefix}{v:,.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def _delta_str(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


# ── Cached data fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_kpi_quotes() -> dict:
    """Fetch fast_info for the 4 headline indices."""
    tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
    }
    result = {}
    for name, sym in tickers.items():
        try:
            fi = yf.Ticker(sym).fast_info
            prev_close = getattr(fi, "previous_close", None) or getattr(fi, "regular_market_previous_close", None)
            last = getattr(fi, "last_price", None)
            if last and prev_close:
                chg_pct = (last - prev_close) / prev_close * 100
            else:
                chg_pct = None
            result[name] = {"price": last, "change_pct": chg_pct, "symbol": sym}
        except Exception:
            result[name] = {"price": None, "change_pct": None, "symbol": sym}
    return result


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_indices_history() -> dict[str, pd.Series]:
    """Fetch 30-day closing price history for all 5 main indices."""
    end = datetime.today()
    start = end - timedelta(days=42)  # extra buffer for weekends/holidays
    result = {}
    for name, sym in MARKET_INDICES.items():
        try:
            df = yf.download(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                close = df["Close"].squeeze()
                # Normalise to % change from first day for comparability
                close_norm = (close / close.iloc[0] - 1) * 100
                result[name] = close_norm.tail(30)
        except Exception:
            pass
    return result


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_watchlist_movers() -> list[dict]:
    """Return top 5 movers (by abs % change) from DEFAULT_WATCHLIST."""
    movers = []
    for sym in DEFAULT_WATCHLIST:
        try:
            fi = yf.Ticker(sym).fast_info
            last = getattr(fi, "last_price", None)
            prev = getattr(fi, "previous_close", None)
            if last and prev and prev != 0:
                chg_pct = (last - prev) / prev * 100
                movers.append({"ticker": sym, "price": last, "change_pct": chg_pct})
        except Exception:
            pass
    # Sort by absolute % change, descending
    movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
    return movers[:5]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_news() -> list[dict]:
    try:
        return get_market_news() or []
    except Exception:
        return []


# ── Navigation pages catalogue ────────────────────────────────────────────────

NAV_PAGES = [
    {"icon": "", "title": "Stock Analysis",    "desc": "OHLCV + indicators",   "page": "Stock Analysis"},
    {"icon": "", "title": "Prediction Center", "desc": "LSTM / XGBoost",       "page": "Prediction Center"},
    {"icon": "", "title": "Risk Analytics",    "desc": "VaR, Sharpe, MDD",     "page": "Risk Analytics"},
    {"icon": "", "title": "Sentiment Center",   "desc": "News sentiment NLP",   "page": "Sentiment Center"},
    {"icon": "", "title": "Portfolio Manager",  "desc": "Holdings & P&L",       "page": "Portfolio Manager"},
    {"icon": "", "title": "Backtesting Lab",   "desc": "Strategy replay",      "page": "Backtesting Lab"},
    {"icon": "", "title": "Market Scanner",    "desc": "Screener & filters",   "page": "Market Scanner"},
    {"icon": "", "title": "Settings",          "desc": "Account & API keys",   "page": "Settings"},
]

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="ss-page-header">
        <div class="ss-page-title">Market Overview</div>
        <div class="ss-page-subtitle">
            Real-time indices, top movers, market news, and quick access to all platform sections.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Welcome card ──────────────────────────────────────────────────────────────

username = user.get("username", "Trader") if user else "Trader"
full_name = user.get("full_name", username) if user else username
now_str = datetime.now().strftime("%A, %B %d %Y · %I:%M %p")

st.markdown(
    f"""
    <div class="ss-welcome-card">
        <div style="width:48px;height:48px;background:rgba(255,255,255,0.18);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.4rem;font-weight:800;color:white;font-family:sans-serif;">{full_name[0].upper() if full_name else 'U'}</div>
        <div>
            <div class="ss-welcome-name">Welcome back, {full_name}!</div>
            <div class="ss-welcome-sub">{now_str} — Markets are live. Here's your daily snapshot.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI Metrics row ───────────────────────────────────────────────────────────

st.markdown('<div class="ss-section-heading">Key Indices</div>', unsafe_allow_html=True)

with st.spinner("Loading market quotes…"):
    kpi_data = _fetch_kpi_quotes()

kpi_cols = st.columns(4)
kpi_labels = ["S&P 500", "NASDAQ", "Dow Jones", "VIX"]
kpi_formats = {
    "S&P 500":   {"prefix": "", "decimals": 2},
    "NASDAQ":    {"prefix": "", "decimals": 2},
    "Dow Jones": {"prefix": "", "decimals": 2},
    "VIX":       {"prefix": "", "decimals": 2},
}

for col, label in zip(kpi_cols, kpi_labels):
    d = kpi_data.get(label, {})
    price = d.get("price")
    chg_pct = d.get("change_pct")
    fmt = kpi_formats[label]
    price_str = _fmt_number(price, prefix=fmt["prefix"], decimals=fmt["decimals"]) if price else "N/A"
    delta_str = _delta_str(chg_pct) if chg_pct is not None else None

    with col:
        st.metric(
            label=label,
            value=price_str,
            delta=delta_str,
            delta_color="normal" if label != "VIX" else "inverse",
        )

st.divider()

# ── Indices mini chart ────────────────────────────────────────────────────────

st.markdown('<div class="ss-section-heading">30-Day Relative Performance</div>', unsafe_allow_html=True)

with st.spinner("Loading indices history…"):
    indices_history = _fetch_indices_history()

if indices_history:
    PALETTE = ["#1a56db", "#059669", "#d97706", "#7c3aed", "#dc2626"]
    fig_indices = go.Figure()

    for (name, series), color in zip(indices_history.items(), PALETTE):
        fig_indices.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=name,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{name}</b><br>Date: %{{x|%b %d}}<br>Change: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig_indices.add_hline(
        y=0,
        line_dash="dash",
        line_color="#9ca3af",
        line_width=1,
    )

    fig_indices.update_layout(
        title=dict(text="Market Indices — % Change (30 Days)", font=dict(size=14, color="#111827")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#f0f0f0",
            tickformat="%b %d",
            tickfont=dict(size=10, color="#6b7280"),
        ),
        yaxis=dict(
            title="% Change",
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
            tickfont=dict(size=10, color="#6b7280"),
            ticksuffix="%",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_indices, use_container_width=True, config={"displayModeBar": False})
else:
    st.warning("Could not load indices history. Please check your internet connection.")

st.divider()

# ── Top Movers + Market News ──────────────────────────────────────────────────

col_movers, col_news = st.columns([1, 1], gap="large")

# Left: Top 5 Movers
with col_movers:
    st.markdown('<div class="ss-section-heading">Top Movers (Watchlist)</div>', unsafe_allow_html=True)

    with st.spinner("Loading movers…"):
        movers = _fetch_watchlist_movers()

    if movers:
        # Header row
        st.markdown(
            """
            <div style="display:flex;justify-content:space-between;padding:0.3rem 0.8rem;
                        font-size:0.75rem;font-weight:700;color:#9ca3af;
                        border-bottom:2px solid #e5e7eb;">
                <span>TICKER</span><span>PRICE</span><span>CHG %</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for m in movers:
            chg = m.get("change_pct", 0)
            badge_cls = "ss-badge-up" if chg >= 0 else "ss-badge-down"
            arrow = "▲" if chg >= 0 else "▼"
            st.markdown(
                f"""
                <div class="ss-mover-row">
                    <span class="ss-mover-ticker">{m['ticker']}</span>
                    <span class="ss-mover-price">${m['price']:,.2f}</span>
                    <span class="{badge_cls}">{arrow} {abs(chg):.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Mini sparkline chart for all movers
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            mover_syms = [m["ticker"] for m in movers]
            end_d = datetime.today()
            start_d = end_d - timedelta(days=35)
            mover_hist = yf.download(
                mover_syms,
                start=start_d.strftime("%Y-%m-%d"),
                end=end_d.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            close_df = mover_hist["Close"] if "Close" in mover_hist.columns else mover_hist

            if not close_df.empty:
                fig_movers = go.Figure()
                MOVER_COLORS = ["#1a56db", "#059669", "#d97706", "#7c3aed", "#dc2626"]
                for sym, color in zip(mover_syms, MOVER_COLORS):
                    if sym in close_df.columns:
                        series = close_df[sym].dropna()
                        fig_movers.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                name=sym,
                                mode="lines",
                                line=dict(color=color, width=1.5),
                                hovertemplate=f"<b>{sym}</b>: $%{{y:.2f}}<extra></extra>",
                            )
                        )
                fig_movers.update_layout(
                    title=dict(text="30-Day Price Trend", font=dict(size=12, color="#374151")),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    height=200,
                    margin=dict(l=0, r=0, t=35, b=5),
                    showlegend=True,
                    legend=dict(font=dict(size=9), orientation="h", y=1.15),
                    xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickformat="%b %d", tickfont=dict(size=9)),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9), tickprefix="$"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_movers, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.caption(f"Sparkline unavailable: {e}")
    else:
        st.warning("Could not load mover data.")

# Right: Market News Feed
with col_news:
    st.markdown('<div class="ss-section-heading">Market News</div>', unsafe_allow_html=True)

    with st.spinner("Loading news…"):
        news_articles = _fetch_market_news()

    if news_articles:
        for article in news_articles[:8]:
            title = article.get("title", "No title")
            source = article.get("source", "Unknown")
            pub = article.get("published_at", "")
            url = article.get("url", "#")

            # Format published_at nicely
            pub_str = ""
            if pub:
                try:
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    pub_str = dt.strftime("%b %d, %Y")
                except Exception:
                    pub_str = pub[:10] if len(pub) >= 10 else pub

            if url and url != "#":
                link_html = f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit;">{title}</a>'
            else:
                link_html = title

            st.markdown(
                f"""
                <div class="ss-news-item">
                    <div class="ss-news-title">{link_html}</div>
                    <div class="ss-news-meta">📡 {source} &nbsp;·&nbsp; 🕐 {pub_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No market news available at this time. Check your NEWS_API_KEY in .env.")

st.divider()

# ── Quick-access navigation grid ──────────────────────────────────────────────

st.markdown('<div class="ss-section-heading">Navigate Platform</div>', unsafe_allow_html=True)

# 3 columns × 3 rows = 9 cards
grid_cols = st.columns(3, gap="medium")

for idx, page_info in enumerate(NAV_PAGES):
    col = grid_cols[idx % 3]
    with col:
        # Use a button styled to look like a card
        clicked = st.button(
            label=f"{page_info['title']}\n{page_info['desc']}",
            key=f"nav_{page_info['page'].lower().replace(' ', '_')}",
            use_container_width=True,
            help=page_info["desc"],
        )
        if clicked:
            st.session_state["current_page"] = page_info["page"]
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:#9ca3af;font-size:0.75rem;padding:1rem 0;">
        StockSense AI © 2025 &nbsp;·&nbsp; Data via Yahoo Finance &nbsp;·&nbsp;
        <span style="color:#1a56db;">Not financial advice.</span>
        Refresh every 5 min (cached).
    </div>
    """,
    unsafe_allow_html=True,
)
