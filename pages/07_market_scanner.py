import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import traceback
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Market Scanner | StockSense AI",
        page_icon="🔭",
        layout="wide",
    )

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4ff;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1a56db !important;
        color: #fff !important;
    }
    .summary-card {
        background: #f8faff;
        border: 1px solid #dce8ff;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
    }
    .filter-card {
        background: #fafbff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Market Scanner")
st.markdown(
    "<p style='color:#6b7280;font-size:1.05rem;margin-top:-8px;'>"
    "Scan the market in real-time for top movers, volume leaders, and RSI extremes.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_ts" not in st.session_state:
    st.session_state.scan_ts = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "current_page" not in st.session_state:
    st.session_state.current_page = None

# ── Fallback scanner (yfinance-based) ────────────────────────────────────────
_DEFAULT_UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","UNH","LLY",
    "JPM","V","XOM","AVGO","PG","MA","HD","COST","ABBV","MRK",
    "CVX","KO","PEP","ADBE","WMT","MCD","CRM","BAC","ACN","NFLX",
    "AMD","INTC","QCOM","TXN","ORCL","IBM","CSCO","TMO","DHR","ABT",
    "NEE","DUK","SO","LIN","SHW","APD","ECL","FCX","NEM","GLD",
    "SPY","QQQ","IWM","DIA","EEM","GDX","XLF","XLK","XLE","XLV",
]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_scanner_data(universe: tuple) -> pd.DataFrame:
    """
    Download 5-day OHLCV for the given universe and compute scanner metrics.
    Falls back gracefully if yfinance is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    tickers_str = " ".join(universe)
    try:
        raw = yf.download(
            tickers_str,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    records = []
    for ticker in universe:
        try:
            if len(universe) == 1:
                df = raw
            else:
                df = raw[ticker] if ticker in raw.columns.get_level_values(0) else pd.DataFrame()

            if df is None or df.empty or len(df) < 2:
                continue

            df = df.dropna(subset=["Close"])
            prev_close = float(df["Close"].iloc[-2])
            curr_close = float(df["Close"].iloc[-1])
            volume     = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
            avg_vol    = float(df["Volume"].mean())    if "Volume" in df.columns else 1
            change_pct = (curr_close - prev_close) / prev_close * 100 if prev_close else 0
            vol_ratio  = volume / avg_vol if avg_vol else 0

            # RSI (14-day)
            closes = df["Close"].dropna()
            rsi_val = np.nan
            if len(closes) >= 2:
                delta = closes.diff().dropna()
                gain  = delta.clip(lower=0).rolling(min(14, len(delta))).mean()
                loss  = (-delta.clip(upper=0)).rolling(min(14, len(delta))).mean()
                rs    = gain / loss.replace(0, np.nan)
                rsi_s = 100 - (100 / (1 + rs))
                rsi_val = float(rsi_s.iloc[-1]) if not rsi_s.empty else np.nan

            records.append({
                "Symbol":    ticker,
                "Price":     round(curr_close, 2),
                "Change %":  round(change_pct, 2),
                "Volume":    int(volume),
                "Vol Ratio": round(vol_ratio, 2),
                "RSI":       round(rsi_val, 1) if not np.isnan(rsi_val) else None,
            })
        except Exception:
            continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def _try_module_scanner():
    """Attempt to call the project's scanner.market_scanner module."""
    try:
        from scanner.market_scanner import MarketScanner  # noqa: F401
        scanner = MarketScanner()
        return scanner.scan()
    except ModuleNotFoundError:
        return None
    except Exception as exc:
        st.warning(f"scanner.market_scanner raised an error: {exc}. Using built-in scanner.")
        return None


def _categorise(df: pd.DataFrame) -> dict:
    """Split a full scan DataFrame into the six category DataFrames."""
    if df.empty:
        empty = pd.DataFrame(columns=["Symbol","Price","Change %","Volume","Vol Ratio","RSI"])
        return {k: empty for k in ["gainers","losers","active","volatile","oversold","overbought"]}

    gainers    = df.nlargest(20, "Change %")
    losers     = df.nsmallest(20, "Change %")
    active     = df.nlargest(20, "Volume")
    volatile   = df.nlargest(20, "Vol Ratio")
    rsi_df     = df.dropna(subset=["RSI"])
    oversold   = rsi_df[rsi_df["RSI"] < 35].nsmallest(20, "RSI")
    overbought = rsi_df[rsi_df["RSI"] > 65].nlargest(20, "RSI")

    return {
        "gainers":    gainers.reset_index(drop=True),
        "losers":     losers.reset_index(drop=True),
        "active":     active.reset_index(drop=True),
        "volatile":   volatile.reset_index(drop=True),
        "oversold":   oversold.reset_index(drop=True),
        "overbought": overbought.reset_index(drop=True),
    }

# ── Top controls ──────────────────────────────────────────────────────────────
ctrl_l, ctrl_r = st.columns([3, 1])
with ctrl_l:
    st.markdown("**Universe**: S&P 500 large-caps + major ETFs &nbsp;|&nbsp; "
                "**Data refresh**: every 5 minutes (cached)")
with ctrl_r:
    run_scan = st.button("Run Scanner", type="primary", use_container_width=True)

# ── Custom Filters ────────────────────────────────────────────────────────────
with st.expander("Custom Scan Filters", expanded=False):
    st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        rsi_range = st.slider("RSI Range", 0, 100, (20, 80), help="Filter by RSI value.")
    with fc2:
        change_range = st.slider(
            "Change % Range", -20.0, 20.0, (-20.0, 20.0), step=0.5,
            help="Filter by daily price change.",
        )
    with fc3:
        vol_threshold = st.number_input(
            "Min Volume", min_value=0, value=500_000, step=100_000,
            format="%d", help="Minimum daily trading volume.",
        )
    apply_filters = st.checkbox("Apply filters to all tabs", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Execute scan ──────────────────────────────────────────────────────────────
if run_scan:
    with st.spinner("Scanning market…"):
        module_result = _try_module_scanner()
        if module_result is not None:
            full_df = module_result if isinstance(module_result, pd.DataFrame) else pd.DataFrame(module_result)
        else:
            full_df = _fetch_scanner_data(tuple(_DEFAULT_UNIVERSE))

        if full_df.empty:
            st.error("Scanner returned no results. Check your internet connection or universe list.")
        else:
            st.session_state.scan_results = full_df
            st.session_state.scan_ts = datetime.now()

# ── Display scan results ──────────────────────────────────────────────────────
full_df = st.session_state.scan_results

if full_df is None:
    st.info("Click **Run Scanner** to begin scanning the market.")
    st.stop()

# Apply custom filters if requested
if apply_filters and not full_df.empty:
    filtered_df = full_df.copy()
    filtered_df = filtered_df[
        (filtered_df["Change %"] >= change_range[0]) &
        (filtered_df["Change %"] <= change_range[1]) &
        (filtered_df["Volume"]   >= vol_threshold)
    ]
    rsi_col = filtered_df["RSI"].dropna()
    filtered_df = filtered_df[
        (filtered_df["RSI"].isna()) |
        ((filtered_df["RSI"] >= rsi_range[0]) & (filtered_df["RSI"] <= rsi_range[1]))
    ]
else:
    filtered_df = full_df.copy()

categories = _categorise(filtered_df)

# ── Summary stats ─────────────────────────────────────────────────────────────
ts_str = st.session_state.scan_ts.strftime("%Y-%m-%d %H:%M:%S") if st.session_state.scan_ts else "—"
total_scanned = len(full_df)
gainers_cnt   = int((full_df["Change %"] > 0).sum())
losers_cnt    = int((full_df["Change %"] < 0).sum())
flat_cnt      = total_scanned - gainers_cnt - losers_cnt

st.markdown(f"<p style='color:#6b7280;font-size:0.85rem'>Last scan: {ts_str}</p>",
            unsafe_allow_html=True)

sm1, sm2, sm3, sm4 = st.columns(4)
with sm1:
    st.metric("Stocks Scanned", f"{total_scanned:,}")
with sm2:
    st.metric("Gainers", f"{gainers_cnt:,}",
              delta=f"{gainers_cnt/total_scanned*100:.1f}%" if total_scanned else None)
with sm3:
    st.metric("Losers", f"{losers_cnt:,}",
              delta=f"-{losers_cnt/total_scanned*100:.1f}%" if total_scanned else None,
              delta_color="inverse")
with sm4:
    st.metric("Flat", f"{flat_cnt:,}")

st.divider()

# ── Shared rendering helpers ──────────────────────────────────────────────────

def _style_change(val):
    if isinstance(val, (int, float)):
        return f"color: {'#16a34a' if val >= 0 else '#dc2626'}; font-weight: 700"
    return ""


def _style_rsi(val):
    if isinstance(val, (int, float)):
        if val <= 30:
            return "color: #16a34a; font-weight: 700"
        if val >= 70:
            return "color: #dc2626; font-weight: 700"
    return ""


def _render_tab(
    df: pd.DataFrame,
    tab_title: str,
    bar_col: str,
    bar_label: str,
    bar_color: str = "#1a56db",
    top_n: int = 10,
):
    """Render a scanner tab: styled table + bar chart + per-row Analyze buttons."""
    if df.empty:
        st.info(f"No stocks match the **{tab_title}** criteria in the current scan.")
        return

    # ── Styled DataFrame ──────────────────────────────────────────────────────
    display_df = df.copy()
    if "Volume" in display_df.columns:
        display_df["Volume"] = display_df["Volume"].apply(
            lambda v: f"{v/1e6:.2f}M" if v >= 1_000_000 else f"{v/1e3:.0f}K"
        )

    styled = display_df.style
    if "Change %" in display_df.columns:
        styled = styled.applymap(_style_change, subset=["Change %"])
    if "RSI" in display_df.columns:
        styled = styled.applymap(_style_rsi, subset=["RSI"])

    fmt_map = {}
    if "Price" in display_df.columns:
        fmt_map["Price"] = "${:.2f}"
    if "Change %" in display_df.columns:
        fmt_map["Change %"] = "{:+.2f}%"
    if "Vol Ratio" in display_df.columns:
        fmt_map["Vol Ratio"] = "{:.2f}x"
    if "RSI" in display_df.columns:
        fmt_map["RSI"] = lambda v: f"{v:.1f}" if pd.notna(v) else "—"
    if fmt_map:
        styled = styled.format(fmt_map)

    st.dataframe(styled, use_container_width=True, height=420)

    # ── Analyze buttons ───────────────────────────────────────────────────────
    st.markdown("**Quick-navigate to analysis:**")
    btn_cols = st.columns(min(8, len(df)))
    for i, (_, row) in enumerate(df.head(8).iterrows()):
        sym = row["Symbol"]
        with btn_cols[i % len(btn_cols)]:
            if st.button(sym, key=f"analyze_{tab_title}_{sym}_{i}"):
                st.session_state.selected_ticker = sym
                st.session_state.current_page    = "Stock Analysis"
                st.success(f"{sym} set as active ticker. Navigate to **Stock Analysis** page.")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if bar_col in df.columns:
        chart_df = df.head(top_n).copy()
        color_seq = [
            bar_color if (isinstance(v, float) and v >= 0) else "#ef4444"
            if bar_col == "Change %" else bar_color
            for v in chart_df[bar_col]
        ]

        fig = go.Figure(
            go.Bar(
                x=chart_df["Symbol"],
                y=chart_df[bar_col],
                marker_color=color_seq,
                text=[f"{v:+.2f}" if bar_col == "Change %" else f"{v:.2f}"
                      for v in chart_df[bar_col]],
                textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{bar_label}: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(
                text=f"Top {top_n} — {tab_title}",
                font=dict(size=14, color="#1e293b"),
            ),
            xaxis_title="Symbol",
            yaxis_title=bar_label,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", size=12, color="#374151"),
            height=340,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        fig.update_yaxes(gridcolor="#f0f0f0", gridwidth=1)
        fig.update_xaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)


# ── Six tabs ──────────────────────────────────────────────────────────────────
tab_gain, tab_lose, tab_active, tab_vol, tab_os, tab_ob = st.tabs([
    "Top Gainers",
    "Top Losers",
    "Most Active",
    "Most Volatile",
    "Oversold RSI",
    "Overbought RSI",
])

with tab_gain:
    _render_tab(
        categories["gainers"],
        tab_title="Top Gainers",
        bar_col="Change %",
        bar_label="Change %",
        bar_color="#16a34a",
    )

with tab_lose:
    _render_tab(
        categories["losers"],
        tab_title="Top Losers",
        bar_col="Change %",
        bar_label="Change %",
        bar_color="#ef4444",
    )

with tab_active:
    _render_tab(
        categories["active"],
        tab_title="Most Active",
        bar_col="Volume",
        bar_label="Volume",
        bar_color="#1a56db",
    )

with tab_vol:
    _render_tab(
        categories["volatile"],
        tab_title="Most Volatile",
        bar_col="Vol Ratio",
        bar_label="Volume Ratio",
        bar_color="#f59e0b",
    )

with tab_os:
    st.markdown(
        "<p style='color:#16a34a;font-weight:600'>RSI ≤ 35 — Potentially oversold / undervalued</p>",
        unsafe_allow_html=True,
    )
    _render_tab(
        categories["oversold"],
        tab_title="Oversold RSI",
        bar_col="RSI",
        bar_label="RSI",
        bar_color="#16a34a",
    )

with tab_ob:
    st.markdown(
        "<p style='color:#dc2626;font-weight:600'>RSI ≥ 65 — Potentially overbought / overvalued</p>",
        unsafe_allow_html=True,
    )
    _render_tab(
        categories["overbought"],
        tab_title="Overbought RSI",
        bar_col="RSI",
        bar_label="RSI",
        bar_color="#dc2626",
    )

# ── Active ticker callout ──────────────────────────────────────────────────────
if st.session_state.selected_ticker:
    st.divider()
    st.success(
        f"Active ticker set to **{st.session_state.selected_ticker}**. "
        "Head to the **Stock Analysis** or **Technical Analysis** page to deep-dive."
    )
