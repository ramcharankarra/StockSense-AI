"""
StockSense AI – Stock Analysis Page (01_stock_analysis.py)
===========================================================
Full OHLCV candlestick chart, technical indicator tabs, company info,
KPI metrics, and a composite trading signal badge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from auth.session_manager import require_login
from analysis.technical_indicators import compute_all
from config import SUPPORTED_PERIODS, SUPPORTED_INTERVALS, DEFAULT_PERIOD, DEFAULT_INTERVAL
from data_ingestion.market_data_service import fetch_ohlcv, get_company_info
from signals.signal_engine import generate_signal

# ── Page configuration ────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="Stock Analysis | StockSense AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ── Auth guard ────────────────────────────────────────────────────────────────

require_login()

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] { background: #ffffff; }
        [data-testid="stSidebar"] { background: #f8faff; }

        /* Page header */
        .ss-page-header {
            padding: 1.2rem 0 0.4rem 0;
            border-bottom: 2px solid #1a56db;
            margin-bottom: 1.5rem;
        }
        .ss-page-title { font-size: 2rem; font-weight: 700; color: #111827; margin: 0; }
        .ss-page-subtitle { font-size: 0.95rem; color: #6b7280; margin-top: 0.2rem; }

        /* Company card */
        .ss-company-card {
            background: #f8faff;
            border: 1.5px solid #dbeafe;
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 1.2rem;
        }
        .ss-company-name { font-size: 1.35rem; font-weight: 700; color: #111827; }
        .ss-company-sub  { font-size: 0.85rem; color: #6b7280; margin-top: 0.15rem; }
        .ss-company-meta { display: flex; flex-wrap: wrap; gap: 1.2rem; margin-top: 0.8rem; }
        .ss-meta-item    { font-size: 0.8rem; }
        .ss-meta-label   { color: #9ca3af; font-weight: 600; text-transform: uppercase;
                           font-size: 0.68rem; letter-spacing: 0.04em; }
        .ss-meta-value   { color: #111827; font-weight: 600; margin-top: 0.1rem; }

        /* Signal badge */
        .ss-signal-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .badge-strong-buy  { background:#d1fae5; color:#065f46; border:1.5px solid #6ee7b7; }
        .badge-buy         { background:#dbeafe; color:#1e3a8a; border:1.5px solid #93c5fd; }
        .badge-hold        { background:#fef3c7; color:#78350f; border:1.5px solid #fcd34d; }
        .badge-sell        { background:#fee2e2; color:#7f1d1d; border:1.5px solid #fca5a5; }
        .badge-strong-sell { background:#fce7f3; color:#831843; border:1.5px solid #f9a8d4; }

        /* Section headings */
        .ss-section-heading {
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
            border-left: 4px solid #1a56db;
            padding-left: 0.6rem;
            margin: 1.2rem 0 0.6rem 0;
        }

        /* Tab overrides */
        [data-baseweb="tab-list"] { gap: 8px; }
        [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colour constants ──────────────────────────────────────────────────────────

PRIMARY   = "#1a56db"
GRID_CLR  = "#f0f0f0"
UP_COLOR  = "#059669"
DOWN_COLOR = "#dc2626"

CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#374151"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
    ),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=55, b=10),
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_number(val, prefix="$", suffix="", decimals=2) -> str:
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


def _signal_badge_html(signal: str, confidence: float) -> str:
    cls_map = {
        "Strong Buy":  ("badge-strong-buy",  ""),
        "Buy":         ("badge-buy",          ""),
        "Hold":        ("badge-hold",         ""),
        "Sell":        ("badge-sell",         ""),
        "Strong Sell": ("badge-strong-sell",  ""),
    }
    cls, icon = cls_map.get(signal, ("badge-hold", ""))
    pct = int(confidence * 100)
    icon_str = f"{icon} " if icon else ""
    return (
        f'<span class="ss-signal-badge {cls}">'
        f'{icon_str}{signal} &nbsp;·&nbsp; {pct}% confidence'
        f'</span>'
    )


def _axis_style(title: str = "", suffix: str = "") -> dict:
    return dict(
        title=title,
        showgrid=True,
        gridcolor=GRID_CLR,
        zeroline=False,
        tickfont=dict(size=10, color="#6b7280"),
        ticksuffix=suffix,
    )


# ── Cached data fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker, period=period, interval=interval)


@st.cache_data(ttl=600, show_spinner=False)
def _load_company_info(ticker: str) -> dict:
    return get_company_info(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _load_signal(ticker: str, period: str, interval: str) -> dict:
    df = fetch_ohlcv(ticker, period=period, interval=interval)
    if df.empty:
        return {}
    try:
        return generate_signal(df)
    except Exception as e:
        return {"signal": "Hold", "confidence": 0.0, "reasons": [str(e)], "composite_score": 0.0}


# ── Sidebar inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Stock Analysis")
    st.markdown("---")

    ticker_input = st.text_input(
        "Ticker Symbol",
        value=st.session_state.get("analysis_ticker", "AAPL"),
        placeholder="e.g. AAPL, MSFT, TSLA",
        help="Enter a valid US stock ticker symbol.",
    ).strip().upper()

    period_sel = st.selectbox(
        "Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,  # default 1y
        help="Historical data window.",
    )

    interval_sel = st.selectbox(
        "Interval",
        options=["1d", "1wk", "1mo"],
        index=0,
        help="Candlestick bar interval.",
    )

    analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

    if analyze_btn and ticker_input:
        st.session_state["analysis_ticker"] = ticker_input
        st.session_state["analysis_period"]   = period_sel
        st.session_state["analysis_interval"] = interval_sel
        # Clear caches on new request
        _load_ohlcv.clear()
        _load_company_info.clear()
        _load_signal.clear()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem;color:#9ca3af;">
        Data sourced from Yahoo Finance.<br>
        Cached for 5 minutes.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Resolve active selection ──────────────────────────────────────────────────

active_ticker   = st.session_state.get("analysis_ticker", "AAPL")
active_period   = st.session_state.get("analysis_period", DEFAULT_PERIOD)
active_interval = st.session_state.get("analysis_interval", DEFAULT_INTERVAL)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="ss-page-header">
        <div class="ss-page-title">Stock Analysis</div>
        <div class="ss-page-subtitle">
            Deep-dive OHLCV charts, technical indicators, and AI-driven trading signals
            for <strong>{active_ticker}</strong>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner(f"Loading data for {active_ticker}…"):
    df_raw = _load_ohlcv(active_ticker, active_period, active_interval)
    info   = _load_company_info(active_ticker)

if df_raw.empty:
    st.error(
        f"❌ No data found for **{active_ticker}**. "
        "Check the ticker symbol and try again."
    )
    st.stop()

# Compute technical indicators
try:
    df = compute_all(df_raw)
except Exception as e:
    st.error(f"Failed to compute indicators: {e}")
    df = df_raw.copy()

# ── Company Info Card + Signal Badge ─────────────────────────────────────────

company_name  = info.get("longName") or info.get("shortName") or active_ticker
sector        = info.get("sector", "N/A")
industry      = info.get("industry", "N/A")
country       = info.get("country", "N/A")
market_cap    = info.get("marketCap")
pe_ratio      = info.get("trailingPE")
fwd_pe        = info.get("forwardPE")
high_52w      = info.get("fiftyTwoWeekHigh")
low_52w       = info.get("fiftyTwoWeekLow")
div_yield     = info.get("dividendYield")
beta          = info.get("beta")
avg_vol       = info.get("averageVolume")

# Trading signal (async-safe: use cached call)
with st.spinner("Generating trading signal…"):
    signal_result = _load_signal(active_ticker, active_period, active_interval)

signal_label = signal_result.get("signal", "Hold")
signal_conf  = signal_result.get("confidence", 0.0)

# Layout: company card left, signal badge right
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    meta_items = [
        ("Sector",      sector),
        ("Industry",    industry),
        ("Country",     country),
        ("Market Cap",  _fmt_number(market_cap, prefix="$")),
        ("P/E (TTM)",   f"{pe_ratio:.2f}" if pe_ratio else "N/A"),
        ("Fwd P/E",     f"{fwd_pe:.2f}" if fwd_pe else "N/A"),
        ("52W High",    f"${high_52w:,.2f}" if high_52w else "N/A"),
        ("52W Low",     f"${low_52w:,.2f}" if low_52w else "N/A"),
        ("Div Yield",   f"{div_yield*100:.2f}%" if div_yield else "N/A"),
        ("Beta",        f"{beta:.2f}" if beta else "N/A"),
        ("Avg Volume",  _fmt_number(avg_vol, prefix="")),
    ]
    meta_html = "".join(
        f"""<div class="ss-meta-item">
                <div class="ss-meta-label">{lbl}</div>
                <div class="ss-meta-value">{val}</div>
            </div>"""
        for lbl, val in meta_items
    )
    exchange = info.get("exchange", "")
    exch_str = f" · {exchange}" if exchange else ""
    st.markdown(
        f"""
        <div class="ss-company-card">
            <div class="ss-company-name">{company_name} ({active_ticker})</div>
            <div class="ss-company-sub">{sector}{exch_str}</div>
            <div class="ss-company-meta">{meta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align:center;padding-top:1.5rem;">
            <div style="font-size:0.72rem;font-weight:700;color:#9ca3af;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">
                AI Trading Signal
            </div>
            {_signal_badge_html(signal_label, signal_conf)}
            <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.6rem;">
                Composite score: {signal_result.get('composite_score', 0.0):+.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Expandable reasons
    if signal_result.get("reasons"):
        with st.expander("📋 Signal Reasons", expanded=False):
            for r in signal_result["reasons"]:
                st.markdown(f"• {r}")

# ── 5 KPI Metrics ─────────────────────────────────────────────────────────────

st.markdown('<div class="ss-section-heading">Key Metrics</div>', unsafe_allow_html=True)

last_close  = float(df["close"].iloc[-1])
prev_close  = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
day_chg_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
last_vol    = int(df["volume"].iloc[-1]) if "volume" in df.columns else 0
day_chg_val = last_close - prev_close

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Current Price",
        f"${last_close:,.2f}",
        delta=f"{day_chg_pct:+.2f}%",
        delta_color="normal",
    )
with kpi2:
    st.metric(
        "Price Change",
        f"${day_chg_val:+,.2f}",
        delta=f"{day_chg_pct:+.2f}%",
        delta_color="normal",
    )
with kpi3:
    st.metric("Daily Volume", _fmt_number(last_vol, prefix="", decimals=0))
with kpi4:
    st.metric("Average Volume", _fmt_number(avg_vol, prefix="", decimals=0) if avg_vol else "N/A")

st.divider()

# ── Main Candlestick + Volume Chart ───────────────────────────────────────────

st.markdown('<div class="ss-section-heading">OHLCV Chart</div>', unsafe_allow_html=True)

fig_candle = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.75, 0.25],
    subplot_titles=("", "Volume"),
)

# Candlestick
fig_candle.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
        increasing=dict(line=dict(color=UP_COLOR), fillcolor=UP_COLOR),
        decreasing=dict(line=dict(color=DOWN_COLOR), fillcolor=DOWN_COLOR),
        hovertext=df.apply(
            lambda r: (
                f"O: ${r['open']:,.2f} | H: ${r['high']:,.2f} | "
                f"L: ${r['low']:,.2f} | C: ${r['close']:,.2f}"
            ),
            axis=1,
        ),
        hoverinfo="text",
    ),
    row=1, col=1,
)

# Volume bars (colour-coded)
colors = [
    UP_COLOR if df["close"].iloc[i] >= df["open"].iloc[i] else DOWN_COLOR
    for i in range(len(df))
]
fig_candle.add_trace(
    go.Bar(
        x=df.index,
        y=df["volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.6,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ),
    row=2, col=1,
)

fig_candle.update_layout(
    **CHART_LAYOUT,
    title=dict(
        text=f"{company_name} ({active_ticker}) — {active_period} · {active_interval}",
        font=dict(size=14, color="#111827"),
    ),
    height=520,
    xaxis_rangeslider_visible=False,
    showlegend=False,
)
fig_candle.update_xaxes(showgrid=True, gridcolor=GRID_CLR, tickfont=dict(size=10, color="#6b7280"))
fig_candle.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=10, color="#6b7280"))
fig_candle.update_yaxes(tickprefix="$", row=1, col=1)

st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": True})

st.divider()

# ── Technical Indicators — 4 Tabs ─────────────────────────────────────────────

st.markdown('<div class="ss-section-heading">Technical Indicators</div>', unsafe_allow_html=True)

tab_ma, tab_rsi, tab_macd, tab_bb = st.tabs([
    "Moving Averages",
    "RSI",
    "MACD",
    "Bollinger Bands",
])

# ── Tab 1: Moving Averages ────────────────────────────────────────────────────

with tab_ma:
    fig_ma = go.Figure()

    # Base price line (light)
    fig_ma.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        name="Close",
        mode="lines",
        line=dict(color="#94a3b8", width=1.2),
        opacity=0.6,
        hovertemplate="Close: $%{y:.2f}<extra></extra>",
    ))

    ma_specs = [
        ("sma_20",  "SMA 20",  PRIMARY,   1.8),
        ("sma_50",  "SMA 50",  "#059669", 1.8),
        ("sma_200", "SMA 200", "#dc2626", 2.0),
        ("ema_12",  "EMA 12",  "#d97706", 1.5),
    ]
    for col_name, label, color, width in ma_specs:
        if col_name in df.columns:
            fig_ma.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                name=label,
                mode="lines",
                line=dict(color=color, width=width),
                hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>",
            ))

    fig_ma.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"{active_ticker} — Moving Averages", font=dict(size=13, color="#111827")),
        height=380,
        xaxis=_axis_style(),
        yaxis=_axis_style("Price (USD)", "$"),
    )
    st.plotly_chart(fig_ma, use_container_width=True, config={"displayModeBar": False})

    # Interpretation callout
    if "sma_20" in df.columns and "sma_50" in df.columns:
        last_sma20 = df["sma_20"].iloc[-1]
        last_sma50 = df["sma_50"].iloc[-1]
        if pd.notna(last_sma20) and pd.notna(last_sma50):
            if last_close > last_sma20 > last_sma50:
                st.success(f"Price ({last_close:.2f}) > SMA20 ({last_sma20:.2f}) > SMA50 ({last_sma50:.2f}) — Bullish alignment")
            elif last_close < last_sma20 < last_sma50:
                st.error(f"Price ({last_close:.2f}) < SMA20 ({last_sma20:.2f}) < SMA50 ({last_sma50:.2f}) — Bearish alignment")
            else:
                st.info(f"Mixed signals — SMA20: {last_sma20:.2f} | SMA50: {last_sma50:.2f}")

# ── Tab 2: RSI ────────────────────────────────────────────────────────────────

with tab_rsi:
    if "rsi" in df.columns:
        fig_rsi = go.Figure()

        # RSI line
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            name="RSI (14)",
            mode="lines",
            line=dict(color=PRIMARY, width=2),
            hovertemplate="RSI: %{y:.2f}<extra></extra>",
        ))

        # Filled zones
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#fee2e2", opacity=0.2, line_width=0, annotation_text="Overbought Zone", annotation_position="top left", annotation_font_size=10, annotation_font_color="#dc2626")
        fig_rsi.add_hrect(y0=0, y1=30,  fillcolor="#d1fae5", opacity=0.2, line_width=0, annotation_text="Oversold Zone",   annotation_position="bottom left", annotation_font_size=10, annotation_font_color="#059669")

        # Reference lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=DOWN_COLOR, line_width=1.5, annotation_text="70 — Overbought", annotation_position="right", annotation_font_size=10)
        fig_rsi.add_hline(y=50, line_dash="dot",  line_color="#9ca3af",   line_width=1)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=UP_COLOR,   line_width=1.5, annotation_text="30 — Oversold", annotation_position="right", annotation_font_size=10)

        fig_rsi.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"{active_ticker} — Relative Strength Index (RSI 14)", font=dict(size=13, color="#111827")),
            height=350,
            yaxis=dict(range=[0, 100], **_axis_style("RSI")),
            xaxis=_axis_style(),
            showlegend=False,
        )
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": False})

        # Last RSI interpretation
        last_rsi = df["rsi"].iloc[-1]
        if pd.notna(last_rsi):
            if last_rsi >= 70:
                st.error(f"RSI = **{last_rsi:.1f}** — Overbought: potential reversal or consolidation ahead.")
            elif last_rsi <= 30:
                st.success(f"RSI = **{last_rsi:.1f}** — Oversold: potential bounce or recovery expected.")
            else:
                st.info(f"RSI = **{last_rsi:.1f}** — Neutral momentum zone.")
    else:
        st.warning("RSI data not available.")

# ── Tab 3: MACD ───────────────────────────────────────────────────────────────

with tab_macd:
    if "macd" in df.columns and "macd_signal" in df.columns and "macd_hist" in df.columns:
        fig_macd = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("MACD Line & Signal", "Histogram"),
        )

        # MACD line
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["macd"],
            name="MACD",
            mode="lines",
            line=dict(color=PRIMARY, width=2),
            hovertemplate="MACD: %{y:.4f}<extra></extra>",
        ), row=1, col=1)

        # Signal line
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            name="Signal",
            mode="lines",
            line=dict(color="#d97706", width=1.8, dash="dot"),
            hovertemplate="Signal: %{y:.4f}<extra></extra>",
        ), row=1, col=1)

        fig_macd.add_hline(y=0, line_color="#9ca3af", line_width=0.8, row=1, col=1)

        # Histogram — colour-coded
        hist_vals   = df["macd_hist"].fillna(0)
        hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in hist_vals]
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=hist_vals,
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.7,
            hovertemplate="Hist: %{y:.4f}<extra></extra>",
        ), row=2, col=1)

        fig_macd.add_hline(y=0, line_color="#9ca3af", line_width=0.8, row=2, col=1)

        fig_macd.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"{active_ticker} — MACD (12, 26, 9)", font=dict(size=13, color="#111827")),
            height=420,
            showlegend=True,
        )
        fig_macd.update_xaxes(showgrid=True, gridcolor=GRID_CLR, tickfont=dict(size=10, color="#6b7280"))
        fig_macd.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=10, color="#6b7280"))

        st.plotly_chart(fig_macd, use_container_width=True, config={"displayModeBar": False})

        # Interpretation
        last_macd  = df["macd"].iloc[-1]
        last_sig   = df["macd_signal"].iloc[-1]
        if pd.notna(last_macd) and pd.notna(last_sig):
            if last_macd > last_sig:
                st.success(f"MACD ({last_macd:.4f}) > Signal ({last_sig:.4f}) — **Bullish crossover** momentum.")
            else:
                st.error(f"MACD ({last_macd:.4f}) < Signal ({last_sig:.4f}) — **Bearish crossover** momentum.")
    else:
        st.warning("MACD data not available.")

# ── Tab 4: Bollinger Bands ────────────────────────────────────────────────────

with tab_bb:
    if "bb_upper" in df.columns and "bb_mid" in df.columns and "bb_lower" in df.columns:
        fig_bb = go.Figure()

        # Fill between bands
        fig_bb.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]),
            y=pd.concat([df["bb_upper"], df["bb_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(26, 86, 219, 0.06)",
            line=dict(color="rgba(255,255,255,0)"),
            name="BB Band",
            hoverinfo="skip",
            showlegend=False,
        ))

        # Upper band
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            name="Upper Band",
            mode="lines",
            line=dict(color="#7c3aed", width=1.5, dash="dash"),
            hovertemplate="Upper: $%{y:.2f}<extra></extra>",
        ))

        # Middle band (SMA 20)
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df["bb_mid"],
            name="SMA 20 (Mid)",
            mode="lines",
            line=dict(color=PRIMARY, width=2),
            hovertemplate="SMA20: $%{y:.2f}<extra></extra>",
        ))

        # Lower band
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            name="Lower Band",
            mode="lines",
            line=dict(color="#059669", width=1.5, dash="dash"),
            hovertemplate="Lower: $%{y:.2f}<extra></extra>",
        ))

        # Close price
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            name="Close",
            mode="lines",
            line=dict(color="#374151", width=1.5),
            opacity=0.85,
            hovertemplate="Close: $%{y:.2f}<extra></extra>",
        ))

        fig_bb.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"{active_ticker} — Bollinger Bands (20, 2σ)", font=dict(size=13, color="#111827")),
            height=380,
            xaxis=_axis_style(),
            yaxis=_axis_style("Price (USD)", "$"),
        )
        st.plotly_chart(fig_bb, use_container_width=True, config={"displayModeBar": False})

        # Interpretation
        last_upper = df["bb_upper"].iloc[-1]
        last_lower = df["bb_lower"].iloc[-1]
        if pd.notna(last_upper) and pd.notna(last_lower):
            band_width = last_upper - last_lower
            pos_in_band = (last_close - last_lower) / band_width if band_width > 0 else 0.5
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Upper Band", f"${last_upper:,.2f}")
            col_b2.metric("Middle (SMA20)", f"${df['bb_mid'].iloc[-1]:,.2f}")
            col_b3.metric("Lower Band", f"${last_lower:,.2f}")

            if last_close > last_upper:
                st.error("Price above Upper Band — **Overbought** signal, watch for reversal.")
            elif last_close < last_lower:
                st.success("Price below Lower Band — **Oversold** signal, potential bounce.")
            else:
                pct = pos_in_band * 100
                st.info(f"Price is at **{pct:.0f}%** within the Bollinger Band range — Neutral.")
    else:
        st.warning("Bollinger Bands data not available.")

# ── Raw data expander ─────────────────────────────────────────────────────────

with st.expander("Raw OHLCV Data", expanded=False):
    display_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    st.dataframe(
        df[display_cols]
          .sort_index(ascending=False)
          .style.format({c: "${:,.2f}" for c in ["open", "high", "low", "close"]}
                        | ({"volume": "{:,.0f}"} if "volume" in display_cols else {})),
        use_container_width=True,
        height=300,
    )
    st.download_button(
        "Download CSV",
        data=df[display_cols].to_csv().encode("utf-8"),
        file_name=f"{active_ticker}_{active_period}_{active_interval}.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:#9ca3af;font-size:0.75rem;padding:1rem 0;">
        StockSense AI © 2025 &nbsp;·&nbsp; Data via Yahoo Finance &nbsp;·&nbsp;
        <span style="color:#1a56db;">Not financial advice.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
