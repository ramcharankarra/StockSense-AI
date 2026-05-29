"""
⚠️ Risk Analytics
==================
Comprehensive risk analysis page covering:
  • Risk score gauge, category badge
  • 8 KPI metrics (Return, Vol, Sharpe, Sortino, Drawdown, Beta, Alpha, VaR)
  • Cumulative returns vs benchmark
  • Drawdown area chart
  • Daily-returns histogram with normal-distribution overlay
  • Rolling 30-day Sharpe ratio
  • Risk decomposition grid & CVaR section
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as scipy_stats

from config import (
    DEFAULT_WATCHLIST,
    MARKET_INDICES,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    VAR_CONFIDENCE,
)
from data_ingestion.market_data_service import fetch_ohlcv
from analysis.risk_analytics import (
    daily_returns,
    annualized_return,
    cumulative_returns,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown_series,
    beta_alpha,
    value_at_risk,
    conditional_var,
    risk_score,
    full_risk_report,
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BLUE        = "#1a56db"
BLUE_LIGHT  = "#e8f0fe"
GREEN       = "#16a34a"
GREEN_LIGHT = "#dcfce7"
RED         = "#dc2626"
RED_LIGHT   = "#fee2e2"
AMBER       = "#d97706"
AMBER_LIGHT = "#fef3c7"
BORDER      = "#e5e7eb"
GREY_BG     = "#f9fafb"

RISK_COLOURS = {
    "Low":      (GREEN, GREEN_LIGHT),
    "Moderate": (AMBER, AMBER_LIGHT),
    "High":     (RED,   RED_LIGHT),
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _chart_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#111827", family="Inter, sans-serif")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#374151"),
        height=height,
        margin=dict(l=20, r=20, t=52, b=40),
        legend=dict(bgcolor="white", bordercolor=BORDER, borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#f3f4f6", linecolor=BORDER, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#f3f4f6", linecolor=BORDER, showgrid=True, zeroline=False),
    )
    return fig


def _section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""<div style='margin:1.5rem 0 0.6rem;'>
              <div style='font-size:1.05rem;font-weight:700;color:#111827;'>{title}</div>
              {'<div style="font-size:0.82rem;color:#6b7280;">' + subtitle + '</div>' if subtitle else ''}
            </div>""",
        unsafe_allow_html=True,
    )


def _info_card(title: str, body: str, icon: str = "", border_colour: str = BLUE) -> None:
    st.markdown(
        f"""<div style='background:white;border:1px solid {BORDER};border-left:4px solid {border_colour};
                        border-radius:8px;padding:0.9rem 1rem;height:100%;'>
              <div style='font-size:0.85rem;font-weight:700;color:#111827;margin-bottom:0.3rem;'>
                {icon + " " if icon else ""}{title}
              </div>
              <div style='font-size:0.8rem;color:#4b5563;line-height:1.55;'>{body}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def _pct(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.{decimals}f}%"


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


# ══════════════════════════════════════════════════════════════════════════════
# Chart renderers
# ══════════════════════════════════════════════════════════════════════════════

def _render_gauge(score: float, category: str) -> None:
    """Plotly indicator gauge 0-100 with green/amber/red colour zones."""
    cat_colour, _ = RISK_COLOURS.get(category, (BLUE, BLUE_LIGHT))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(font=dict(size=40, color=cat_colour, family="Inter, sans-serif"), suffix=""),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#9ca3af",
                tickvals=[0, 33, 66, 100],
                ticktext=["0", "33", "66", "100"],
                tickfont=dict(size=11),
            ),
            bar=dict(color=cat_colour, thickness=0.28),
            bgcolor="white",
            borderwidth=1,
            bordercolor=BORDER,
            steps=[
                dict(range=[0,  33], color=GREEN_LIGHT),
                dict(range=[33, 66], color=AMBER_LIGHT),
                dict(range=[66, 100], color=RED_LIGHT),
            ],
            threshold=dict(
                line=dict(color=cat_colour, width=3),
                thickness=0.8,
                value=score,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="white",
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cumulative_returns(
    prices: pd.Series,
    benchmark_prices: pd.Series | None,
    ticker: str,
    benchmark_label: str,
) -> None:
    cum_stock = cumulative_returns(prices)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum_stock.index, y=(cum_stock - 1) * 100,
        name=ticker,
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(26,86,219,0.06)",
    ))

    if benchmark_prices is not None and not benchmark_prices.empty:
        cum_bench = cumulative_returns(benchmark_prices)
        common    = cum_stock.index.intersection(cum_bench.index)
        fig.add_trace(go.Scatter(
            x=common, y=(cum_bench.loc[common] - 1) * 100,
            name=benchmark_label,
            line=dict(color="#9ca3af", width=2, dash="dash"),
        ))

    fig.add_hline(y=0, line=dict(color=BORDER, width=1))
    _chart_layout(fig, f"Cumulative Returns — {ticker} vs {benchmark_label}", height=400)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Return (%)", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)


def _render_drawdown(prices: pd.Series, ticker: str) -> None:
    dd = drawdown_series(prices) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color=RED, width=1.5),
        fillcolor="rgba(220,38,38,0.12)",
    ))
    _chart_layout(fig, f"Drawdown — {ticker}", height=320)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)


def _render_returns_histogram(prices: pd.Series, ticker: str) -> None:
    rets = daily_returns(prices).dropna() * 100  # pct

    mu, sigma = float(rets.mean()), float(rets.std())
    x_range   = np.linspace(rets.min(), rets.max(), 300)
    normal_y  = scipy_stats.norm.pdf(x_range, mu, sigma) * len(rets) * (rets.max() - rets.min()) / 50

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets,
        nbinsx=50,
        name="Daily Returns",
        marker=dict(color=BLUE, opacity=0.7, line=dict(color=BLUE_LIGHT, width=0.5)),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_y,
        name="Normal Distribution",
        line=dict(color=RED, width=2.5, dash="dash"),
        mode="lines",
    ))

    # VaR line
    var_95 = float(np.percentile(rets, 5))
    fig.add_vline(
        x=var_95,
        line=dict(color=AMBER, width=2, dash="dot"),
        annotation_text=f"VaR 95%: {var_95:.2f}%",
        annotation_position="top left",
        annotation_font=dict(color=AMBER, size=11),
    )

    _chart_layout(fig, f"Daily Returns Distribution — {ticker}", height=360)
    fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def _render_rolling_sharpe(prices: pd.Series, ticker: str, window: int = 30) -> None:
    rets   = daily_returns(prices)
    ann    = TRADING_DAYS_PER_YEAR
    roll   = (
        (rets.rolling(window).mean() * ann - RISK_FREE_RATE)
        / (rets.rolling(window).std() * np.sqrt(ann))
    ).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll,
        name=f"Rolling {window}d Sharpe",
        line=dict(color=BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(26,86,219,0.07)",
    ))
    fig.add_hline(y=0,   line=dict(color="#9ca3af", width=1, dash="dot"))
    fig.add_hline(y=1.0, line=dict(color=GREEN,    width=1.2, dash="dash"),
                  annotation_text="Sharpe = 1.0", annotation_position="top right",
                  annotation_font=dict(color=GREEN, size=10))
    _chart_layout(fig, f"Rolling {window}-Day Sharpe Ratio — {ticker}", height=320)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sharpe Ratio")
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_decomposition(report: dict) -> None:
    """2×2 grid explaining each key metric with interpretation text."""
    _section_header(
        "Risk Decomposition",
        "Metric interpretation with contextual signals",
    )

    vol     = report.get("volatility", 0)
    sharpe  = report.get("sharpe_ratio", 0)
    mdd     = report.get("max_drawdown", 0)
    sortino = report.get("sortino_ratio", 0)

    # Volatility
    vol_cat  = "Low" if vol < 0.15 else ("Moderate" if vol < 0.35 else "High")
    vol_txt  = (
        f"Annualised volatility of <b>{vol*100:.1f}%</b>. "
        f"Categorised as <b>{vol_cat}</b>. "
        + ("Below-average risk profile." if vol < 0.15
           else "Elevated price swings — expect significant daily moves." if vol > 0.35
           else "Moderate day-to-day fluctuations.")
    )

    # Sharpe
    sharpe_sig = "Strong" if sharpe >= 1.5 else ("Acceptable" if sharpe >= 0.5 else "Weak")
    sharpe_txt = (
        f"Sharpe ratio of <b>{sharpe:.2f}</b>. Signal: <b>{sharpe_sig}</b>. "
        "Measures excess return per unit of total risk. "
        + ("Excellent risk-adjusted performance." if sharpe >= 2.0
           else "Good performance relative to risk." if sharpe >= 1.0
           else "Return may not compensate for risk taken.")
    )

    # Max Drawdown
    mdd_abs  = abs(mdd) * 100
    mdd_sig  = "Mild" if mdd_abs < 15 else ("Moderate" if mdd_abs < 40 else "Severe")
    mdd_txt  = (
        f"Peak-to-trough decline of <b>{mdd_abs:.1f}%</b>. Severity: <b>{mdd_sig}</b>. "
        "Represents the worst capital loss from a peak. "
        + ("Resilient — minimal capital erosion." if mdd_abs < 15
           else "Significant drawdown; recovery takes time." if mdd_abs > 40
           else "Moderate drawdown; within typical market range.")
    )

    # Sortino
    sortino_sig = "Strong" if sortino >= 1.5 else ("Acceptable" if sortino >= 0.5 else "Weak")
    sortino_txt = (
        f"Sortino ratio of <b>{sortino:.2f}</b>. Signal: <b>{sortino_sig}</b>. "
        "Like Sharpe but penalese only downside volatility — a more investor-friendly metric. "
        + ("Excellent downside-adjusted returns." if sortino >= 2.0
           else "Decent protection against downside." if sortino >= 1.0
           else "Downside risk appears elevated relative to returns.")
    )

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        _info_card("Volatility", vol_txt, "", BLUE)
    with r1c2:
        _info_card("Sharpe Ratio", sharpe_txt, "", GREEN)
    with r2c1:
        _info_card("Max Drawdown", mdd_txt, "", RED)
    with r2c2:
        _info_card("Sortino Ratio", sortino_txt, "", AMBER)


def _render_cvar_section(prices: pd.Series) -> None:
    """Expected shortfall / CVaR explanation and value."""
    var  = value_at_risk(prices) * 100
    cvar = conditional_var(prices) * 100

    _section_header(
        "Conditional Value at Risk (CVaR)",
        "Expected Shortfall — average loss beyond the VaR threshold",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("VaR 95%",  f"{var:.2f}%",  help="Worst loss expected 5% of the time")
    with c2:
        st.metric("CVaR 95%", f"{cvar:.2f}%", help="Average loss on the worst 5% of days")
    with c3:
        st.markdown(
            f"""<div style='background:{RED_LIGHT};border:1px solid {RED}44;border-radius:10px;
                            padding:0.9rem 1.1rem;'>
                  <div style='font-size:0.83rem;font-weight:700;color:{RED};margin-bottom:0.3rem;'>
                    What is CVaR?
                  </div>
                  <div style='font-size:0.8rem;color:#374151;line-height:1.55;'>
                    VaR tells you the <i>minimum</i> loss in the worst {int((1-VAR_CONFIDENCE)*100)}% of days:
                    <b>{var:.2f}%</b>.<br>
                    CVaR (Expected Shortfall) tells you the <i>average</i> loss on those bad days:
                    <b>{cvar:.2f}%</b>.<br>
                    CVaR is a more conservative and tail-risk-sensitive measure widely used
                    in institutional risk management.
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Page entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{RED}10,{BLUE}06);
                    border-bottom:1px solid {BORDER};padding:1.4rem 0 1rem;margin-bottom:1rem;'>
          <div style='display:flex;align-items:center;gap:0.8rem;'>
            <div>
              <h1 style='margin:0;font-size:1.6rem;font-weight:800;color:#111827;'>
                Risk Analytics
              </h1>
              <p style='margin:0;color:#6b7280;font-size:0.9rem;'>
                Comprehensive risk metrics, drawdown analysis, and tail-risk assessment
              </p>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuration")
        st.markdown("---")

        ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("risk_ticker", "AAPL"),
            placeholder="e.g. AAPL, MSFT",
        ).upper().strip()
        st.session_state["risk_ticker"] = ticker

        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Historical window for risk calculations",
        )

        benchmark_name = st.selectbox(
            "Benchmark",
            list(MARKET_INDICES.keys()),
            index=0,
        )
        benchmark_ticker = MARKET_INDICES[benchmark_name]

        rolling_window = st.slider(
            "Rolling Sharpe Window (days)",
            min_value=10, max_value=90, value=30, step=5,
        )

        st.markdown("---")
        refresh = st.button("Refresh Analysis", use_container_width=True, type="primary")

    if not ticker:
        st.info("👈 Enter a ticker symbol in the sidebar to begin.")
        return

    # ── Data Loading ──────────────────────────────────────────────────────────
    @st.cache_data(ttl=900, show_spinner=False)
    def _load(sym: str, prd: str) -> pd.DataFrame:
        return fetch_ohlcv(sym, period=prd)

    with st.spinner(f"📡 Loading data for {ticker} and {benchmark_name}…"):
        df        = _load(ticker, period)
        bench_df  = _load(benchmark_ticker, period)

    if df is None or df.empty:
        st.error(
            f"❌ No data for **{ticker}**. Verify the ticker and try again."
        )
        return

    if "close" not in df.columns:
        st.error("❌ Data format error: 'close' column missing.")
        return

    prices         = df["close"].dropna()
    bench_prices   = bench_df["close"].dropna() if (bench_df is not None and not bench_df.empty) else None

    # ── Compute report ────────────────────────────────────────────────────────
    try:
        report = full_risk_report(prices, bench_prices)
    except Exception as exc:
        st.error(f"❌ Risk calculation failed: {exc}")
        return

    score    = report["risk_score"]
    category = report["risk_category"]
    cat_c, cat_bg = RISK_COLOURS.get(category, (BLUE, BLUE_LIGHT))

    # ══════════════════════════════════════════════════════════════════════════
    # Top Section: Gauge + Badge + KPIs
    # ══════════════════════════════════════════════════════════════════════════
    gauge_col, kpi_col = st.columns([1, 2.8])

    with gauge_col:
        st.markdown(
            f"""<div style='text-align:center;margin-bottom:-1rem;'>
                  <div style='font-size:0.78rem;font-weight:600;color:#6b7280;
                              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem;'>
                    Composite Risk Score
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
        _render_gauge(score, category)

        # Risk category badge
        st.markdown(
            f"""<div style='text-align:center;margin-top:-0.5rem;'>
                  <span style='background:{cat_c};color:white;padding:4px 18px;
                               border-radius:20px;font-size:1rem;font-weight:800;
                               letter-spacing:0.04em;'>
                    {category} Risk
                  </span>
                  <div style='font-size:0.75rem;color:#6b7280;margin-top:0.5rem;'>
                    Score: {score:.1f} / 100
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    with kpi_col:
        _section_header(f"Key Risk Metrics — {ticker}")

        ann_ret  = report.get("annualized_return")
        vol      = report.get("volatility")
        sharpe   = report.get("sharpe_ratio")
        sortino  = report.get("sortino_ratio")
        mdd      = report.get("max_drawdown")
        beta     = report.get("beta")
        alpha    = report.get("alpha")
        var95    = report.get("var_95")

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)

        with r1c1:
            delta_c = "normal" if ann_ret is not None and ann_ret >= 0 else "inverse"
            st.metric(
                "Annualised Return",
                f"{ann_ret*100:.1f}%" if ann_ret is not None else "N/A",
                delta=("▲" if ann_ret and ann_ret >= 0 else "▼"),
                delta_color=delta_c,
            )
        with r1c2:
            st.metric("Volatility", f"{vol*100:.1f}%" if vol else "N/A")
        with r1c3:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A",
                      help="Higher = better risk-adjusted return")
        with r1c4:
            st.metric("Sortino Ratio", f"{sortino:.2f}" if sortino is not None else "N/A",
                      help="Like Sharpe but penalises downside only")
        with r2c1:
            st.metric(
                "Max Drawdown",
                f"{mdd*100:.1f}%" if mdd is not None else "N/A",
                help="Peak-to-trough loss",
            )
        with r2c2:
            st.metric("Beta",  f"{beta:.2f}"  if beta  is not None else "N/A",
                      help=f"vs {benchmark_name}")
        with r2c3:
            st.metric("Alpha", f"{alpha*100:.2f}%" if alpha is not None else "N/A",
                      help=f"Jensen's Alpha vs {benchmark_name}")
        with r2c4:
            st.metric("VaR 95%", f"{var95*100:.2f}%" if var95 is not None else "N/A",
                      help="Max daily loss at 95% confidence")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Cumulative Returns vs Benchmark
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(
        "Cumulative Returns",
        f"Performance of {ticker} vs {benchmark_name} over the selected period",
    )
    try:
        _render_cumulative_returns(prices, bench_prices, ticker, benchmark_name)
    except Exception as exc:
        st.error(f"Cumulative returns chart error: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # Drawdown + Rolling Sharpe  (side by side)
    # ══════════════════════════════════════════════════════════════════════════
    dd_col, sharpe_col = st.columns(2)

    with dd_col:
        _section_header("Drawdown Over Time")
        try:
            _render_drawdown(prices, ticker)
        except Exception as exc:
            st.error(f"Drawdown chart error: {exc}")

    with sharpe_col:
        _section_header(f"Rolling {rolling_window}-Day Sharpe Ratio")
        try:
            _render_rolling_sharpe(prices, ticker, window=rolling_window)
        except Exception as exc:
            st.error(f"Rolling Sharpe chart error: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Daily Returns Distribution
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(
        "Daily Returns Distribution",
        "Histogram with normal distribution overlay and VaR 95% threshold",
    )
    try:
        _render_returns_histogram(prices, ticker)
    except Exception as exc:
        st.error(f"Distribution chart error: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Risk Decomposition
    # ══════════════════════════════════════════════════════════════════════════
    try:
        _render_risk_decomposition(report)
    except Exception as exc:
        st.error(f"Risk decomposition error: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # CVaR / Expected Shortfall
    # ══════════════════════════════════════════════════════════════════════════
    try:
        _render_cvar_section(prices)
    except Exception as exc:
        st.error(f"CVaR section error: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Additional stats footer
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("Full Statistics Summary")

    rets = daily_returns(prices).dropna()

    stat_data = {
        "Metric": [
            "Total Data Points", "Date Range",
            "Mean Daily Return", "Median Daily Return",
            "Std Dev (Daily)", "Skewness", "Kurtosis",
            "Best Trading Day", "Worst Trading Day",
            "Positive Trading Days %", "Daily Expected Shortfall (CVaR 95%)",
        ],
        "Value": [
            f"{len(prices):,}",
            f"{prices.index.min().strftime('%Y-%m-%d')} → {prices.index.max().strftime('%Y-%m-%d')}",
            f"{rets.mean()*100:.4f}%",
            f"{rets.median()*100:.4f}%",
            f"{rets.std()*100:.4f}%",
            f"{float(rets.skew()):.4f}",
            f"{float(rets.kurtosis()):.4f}",
            f"{rets.max()*100:.2f}%",
            f"{rets.min()*100:.2f}%",
            f"{(rets > 0).mean()*100:.1f}%",
            f"{report.get('cvar_95', 0)*100:.2f}%",
        ],
    }

    stats_df = pd.DataFrame(stat_data)
    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width=220),
            "Value":  st.column_config.TextColumn("Value",  width=200),
        },
    )

    # Data freshness note
    st.markdown(
        f"""<div style='text-align:right;font-size:0.75rem;color:#9ca3af;margin-top:0.5rem;'>
              Data sourced via yfinance · Period: {period} · Risk-free rate: {RISK_FREE_RATE*100:.1f}%
            </div>""",
        unsafe_allow_html=True,
    )


# ── Run ───────────────────────────────────────────────────────────────────────
main()
