import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import traceback

# ── Page config ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Backtesting Lab | StockSense AI",
        page_icon="🔬",
        layout="wide",
    )

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8faff;
        border: 1px solid #dce8ff;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4ff;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1a56db !important;
        color: #fff !important;
    }
    .strategy-card {
        background: #f0f6ff;
        border-left: 4px solid #1a56db;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
    }
    .buy-badge  { color: #16a34a; font-weight: 700; }
    .sell-badge { color: #dc2626; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Strategy Backtesting Lab")
st.markdown(
    "<p style='color:#6b7280;font-size:1.05rem;margin-top:-8px;'>"
    "Simulate trading strategies on historical data and evaluate performance.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Strategy descriptions ─────────────────────────────────────────────────────
STRATEGY_INFO = {
    "RSI": {
        "name": "RSI (Relative Strength Index)",
        "description": (
            "Buys when RSI falls below the oversold threshold (asset is cheap) and sells when "
            "RSI rises above the overbought threshold (asset is expensive). "
            "A momentum oscillator ranging 0-100 that measures the speed and change of price movements."
        ),
        "params": ["rsi_period", "rsi_oversold", "rsi_overbought"],
    },
    "MACD": {
        "name": "MACD (Moving Average Convergence Divergence)",
        "description": (
            "Buys when the MACD line crosses above the Signal line (bullish crossover) and sells "
            "when it crosses below (bearish crossover). Captures trend momentum by comparing "
            "short-term and long-term exponential moving averages."
        ),
        "params": ["macd_fast", "macd_slow", "macd_signal"],
    },
    "SMA": {
        "name": "SMA Crossover (Simple Moving Average)",
        "description": (
            "Buys when the short-term SMA crosses above the long-term SMA (Golden Cross) and sells "
            "when it crosses below (Death Cross). A classic trend-following strategy that smooths "
            "price noise to identify trend direction."
        ),
        "params": ["sma_short", "sma_long"],
    },
    "Bollinger": {
        "name": "Bollinger Bands",
        "description": (
            "Buys when price touches or falls below the lower Bollinger Band (mean-reversion entry) "
            "and sells when price reaches the upper band. Bands are plotted at N standard deviations "
            "around a rolling mean, automatically widening in volatile markets."
        ),
        "params": ["bb_period", "bb_std"],
    },
}

# ── Session state init ────────────────────────────────────────────────────────
if "backtest_runs" not in st.session_state:
    st.session_state.backtest_runs = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Backtest Configuration")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. AAPL").upper().strip()
    strategy = st.selectbox("Strategy", list(STRATEGY_INFO.keys()), index=0)

    st.markdown("**Date Range**")
    col_sd, col_ed = st.columns(2)
    with col_sd:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=365 * 3))
    with col_ed:
        end_date = st.date_input("End", value=date.today())

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
        value=100_000, step=5_000, format="%d"
    )
    commission_rate = st.number_input(
        "Commission Rate (%)", min_value=0.0, max_value=5.0,
        value=0.1, step=0.05, format="%.2f"
    )

    st.markdown("---")
    st.markdown("### Strategy Parameters")

    params = {}
    if strategy == "RSI":
        params["rsi_period"]    = st.slider("RSI Period",    5,  30, 14)
        params["rsi_oversold"]  = st.slider("Oversold Threshold",  10, 45, 30)
        params["rsi_overbought"]= st.slider("Overbought Threshold",55, 90, 70)
    elif strategy == "MACD":
        params["macd_fast"]   = st.slider("Fast Period",   5,  30, 12)
        params["macd_slow"]   = st.slider("Slow Period",  15,  60, 26)
        params["macd_signal"] = st.slider("Signal Period",  5,  20,  9)
    elif strategy == "SMA":
        params["sma_short"] = st.slider("Short SMA Period",  5,  50, 20)
        params["sma_long"]  = st.slider("Long SMA Period",  20, 200, 50)
    elif strategy == "Bollinger":
        params["bb_period"] = st.slider("BB Period", 5,  60, 20)
        params["bb_std"]    = st.slider("Std Dev Multiplier", 1.0, 4.0, 2.0, step=0.1)

    st.markdown("---")
    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _annualized_return(total_return_pct: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    r = total_return_pct / 100
    return ((1 + r) ** (365 / n_days) - 1) * 100


def _sharpe(daily_returns: pd.Series, risk_free: float = 0.04) -> float:
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free / 252
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min() * 100)


def _color_metric(value: float, suffix: str = "%") -> str:
    color = "#16a34a" if value >= 0 else "#dc2626"
    sign  = "+" if value > 0 else ""
    return f"<span style='color:{color};font-weight:700'>{sign}{value:.2f}{suffix}</span>"

# ── Pure-Python fallback backtest engine ─────────────────────────────────────

def _fallback_backtest(
    ticker: str,
    start: date,
    end: date,
    strategy: str,
    initial_capital: float,
    commission_pct: float,
    params: dict,
) -> dict:
    """
    Minimal self-contained backtest engine used when the project's
    backtesting.backtest_engine module is unavailable.
    Downloads OHLCV data via yfinance and applies the chosen strategy.
    """
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance is not installed. Run `pip install yfinance` and retry.")
        return {}

    df = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=True)
    if df.empty:
        st.error(f"No data returned for **{ticker}** in the selected date range.")
        return {}

    df = df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index = pd.to_datetime(df.index)
    close = df["close"]

    # ── Signal generation ──────────────────────────────────────────────────
    signals = pd.Series(0, index=df.index)

    if strategy == "RSI":
        period      = params.get("rsi_period", 14)
        oversold    = params.get("rsi_oversold", 30)
        overbought  = params.get("rsi_overbought", 70)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        signals[rsi < oversold]  =  1
        signals[rsi > overbought] = -1

    elif strategy == "MACD":
        fast   = params.get("macd_fast", 12)
        slow   = params.get("macd_slow", 26)
        signal_p = params.get("macd_signal", 9)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
        crossover   = macd_line - signal_line
        signals[(crossover > 0) & (crossover.shift(1) <= 0)] =  1
        signals[(crossover < 0) & (crossover.shift(1) >= 0)] = -1

    elif strategy == "SMA":
        short = params.get("sma_short", 20)
        long_ = params.get("sma_long", 50)
        sma_s = close.rolling(short).mean()
        sma_l = close.rolling(long_).mean()
        diff  = sma_s - sma_l
        signals[(diff > 0) & (diff.shift(1) <= 0)] =  1
        signals[(diff < 0) & (diff.shift(1) >= 0)] = -1

    elif strategy == "Bollinger":
        period = params.get("bb_period", 20)
        std_m  = params.get("bb_std", 2.0)
        mid    = close.rolling(period).mean()
        std    = close.rolling(period).std()
        upper  = mid + std_m * std
        lower  = mid - std_m * std
        signals[close < lower] =  1
        signals[close > upper] = -1

    # ── Portfolio simulation ───────────────────────────────────────────────
    cash       = float(initial_capital)
    shares     = 0.0
    commission = commission_pct / 100
    trades     = []
    equity_list= []

    in_position = False
    for idx, row in df.iterrows():
        price = float(close.loc[idx])
        sig   = int(signals.loc[idx])

        if sig == 1 and not in_position:
            cost   = cash * (1 - commission)
            bought = cost / price
            comm_  = cash * commission
            shares = bought
            cash   = 0.0
            in_position = True
            trades.append({
                "Date":            idx.date(),
                "Action":          "BUY",
                "Price":           round(price, 2),
                "Shares":          round(shares, 4),
                "Commission ($)":  round(comm_, 2),
                "Portfolio Value": round(shares * price, 2),
            })

        elif sig == -1 and in_position:
            proceeds = shares * price
            comm_    = proceeds * commission
            cash     = proceeds - comm_
            in_position = False
            trades.append({
                "Date":            idx.date(),
                "Action":          "SELL",
                "Price":           round(price, 2),
                "Shares":          round(shares, 4),
                "Commission ($)":  round(comm_, 2),
                "Portfolio Value": round(cash, 2),
            })
            shares = 0.0

        equity_val = cash + shares * price
        equity_list.append({"Date": idx, "Equity": equity_val})

    # Close open position at last price
    if in_position:
        last_price = float(close.iloc[-1])
        cash       = shares * last_price * (1 - commission)
        shares     = 0.0

    equity_df = pd.DataFrame(equity_list).set_index("Date")
    bnh_equity = (close / close.iloc[0]) * initial_capital

    daily_ret    = equity_df["Equity"].pct_change().dropna()
    total_ret    = (equity_df["Equity"].iloc[-1] / initial_capital - 1) * 100
    n_days       = (df.index[-1] - df.index[0]).days
    ann_ret      = _annualized_return(total_ret, n_days)
    sharpe       = _sharpe(daily_ret)
    max_dd       = _max_drawdown(equity_df["Equity"])
    trades_df    = pd.DataFrame(trades) if trades else pd.DataFrame()
    win_trades   = 0
    if len(trades_df) >= 2:
        sell_rows = trades_df[trades_df["Action"] == "SELL"]
        buy_rows  = trades_df[trades_df["Action"] == "BUY"]
        n_pairs   = min(len(sell_rows), len(buy_rows))
        wins = sum(
            sell_rows.iloc[i]["Price"] > buy_rows.iloc[i]["Price"]
            for i in range(n_pairs)
        )
        win_trades = (wins / n_pairs * 100) if n_pairs else 0

    peak     = equity_df["Equity"].cummax()
    drawdown = (equity_df["Equity"] - peak) / peak * 100

    return {
        "equity":        equity_df,
        "bnh":           bnh_equity,
        "drawdown":      drawdown,
        "trades":        trades_df,
        "total_return":  round(total_ret, 2),
        "ann_return":    round(ann_ret, 2),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 2),
        "win_rate":      round(win_trades, 1),
        "total_trades":  len(trades_df),
        "final_value":   round(equity_df["Equity"].iloc[-1], 2),
        "ticker":        ticker,
        "strategy":      strategy,
        "params":        params,
        "start":         str(start),
        "end":           str(end),
        "initial_capital": initial_capital,
    }

# ── Run backtest ──────────────────────────────────────────────────────────────

def run_backtest_wrapper(ticker, start, end, strategy, capital, commission, params):
    try:
        from backtesting.backtest_engine import run_backtest, STRATEGIES  # noqa: F401
        result = run_backtest(
            ticker=ticker,
            start=str(start),
            end=str(end),
            strategy=strategy,
            initial_capital=capital,
            commission=commission / 100,
            **params,
        )
        return result
    except ModuleNotFoundError:
        return _fallback_backtest(ticker, start, end, strategy, capital, commission, params)
    except Exception as exc:
        st.error(f"Backtest engine error: {exc}")
        st.code(traceback.format_exc())
        return {}

# ── Strategy explanation card ─────────────────────────────────────────────────
info = STRATEGY_INFO[strategy]
st.markdown(
    f"""
    <div class='strategy-card'>
        <strong style='font-size:1.05rem;color:#1a56db'>📘 {info['name']}</strong><br>
        <span style='color:#374151;font-size:0.95rem'>{info['description']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_results, tab_compare = st.tabs(["Backtest Results", "Compare Runs"])

# ════════════════════════════════════════════════════════════════════════════
#  BACKTEST RESULTS TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_results:

    if run_button:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        elif not ticker:
            st.error("Please enter a valid ticker symbol.")
        else:
            with st.spinner(f"Running {strategy} backtest on {ticker}…"):
                result = run_backtest_wrapper(
                    ticker, start_date, end_date, strategy,
                    initial_capital, commission_rate, params
                )
            if result:
                st.session_state.last_result = result
                # Store run summary for compare tab
                run_summary = {
                    "Label":          f"{ticker} · {strategy}",
                    "Ticker":         ticker,
                    "Strategy":       strategy,
                    "Start":          str(start_date),
                    "End":            str(end_date),
                    "Total Return %": result.get("total_return", 0),
                    "Ann. Return %":  result.get("ann_return", 0),
                    "Sharpe":         result.get("sharpe", 0),
                    "Max DD %":       result.get("max_drawdown", 0),
                    "Win Rate %":     result.get("win_rate", 0),
                    "Total Trades":   result.get("total_trades", 0),
                    "Final Value $":  result.get("final_value", 0),
                    "Capital $":      initial_capital,
                    "Commission %":   commission_rate,
                    "result_obj":     result,
                }
                st.session_state.backtest_runs.append(run_summary)

    result = st.session_state.last_result
    if not result:
        st.info("👈 Configure your strategy in the sidebar and click **Run Backtest** to begin.")
        st.stop()

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    total_ret  = result.get("total_return",  0)
    ann_ret    = result.get("ann_return",    0)
    sharpe_r   = result.get("sharpe",        0)
    max_dd     = result.get("max_drawdown",  0)
    win_rate   = result.get("win_rate",      0)
    tot_trades = result.get("total_trades",  0)
    final_val  = result.get("final_value",   0)

    profit_color = "#16a34a" if total_ret >= 0 else "#dc2626"

    st.markdown("### Performance Summary")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def _delta_str(v, suffix="%"):
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.2f}{suffix}"

    with k1:
        st.metric("Total Return", f"{_delta_str(total_ret)}",
                  delta=None, help="Percentage gain/loss over the full period.")
        st.markdown(
            f"<div style='text-align:center;color:{profit_color};font-weight:700;font-size:1.3rem'>"
            f"{_delta_str(total_ret)}</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.metric("Ann. Return", f"{_delta_str(ann_ret)}", help="CAGR over the backtest period.")
    with k3:
        sharpe_col = "#16a34a" if sharpe_r >= 1 else ("#f59e0b" if sharpe_r >= 0 else "#dc2626")
        st.metric("Sharpe Ratio", f"{sharpe_r:.2f}", help="Risk-adjusted return (>1 is good).")
    with k4:
        st.metric("Max Drawdown", f"{max_dd:.2f}%", help="Largest peak-to-trough decline.")
    with k5:
        st.metric("Win Rate", f"{win_rate:.1f}%", help="Percentage of profitable trades.")
    with k6:
        st.metric("Total Trades", f"{tot_trades}", help="Number of completed round-trip trades.")

    st.markdown(
        f"<p style='color:#6b7280;font-size:0.9rem;margin-top:4px'>"
        f"Final Portfolio Value: <strong style='color:{profit_color}'>"
        f"${final_val:,.2f}</strong> &nbsp;|&nbsp; "
        f"Initial Capital: <strong>${initial_capital:,.2f}</strong></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Equity Curve + Drawdown ───────────────────────────────────────────────
    equity_df = result.get("equity", pd.DataFrame())
    bnh       = result.get("bnh",      pd.Series(dtype=float))
    drawdown  = result.get("drawdown", pd.Series(dtype=float))

    if not equity_df.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            subplot_titles=["Equity Curve vs Buy-and-Hold", "Drawdown (%)"],
            vertical_spacing=0.08,
        )

        # Strategy equity
        fig.add_trace(
            go.Scatter(
                x=equity_df.index, y=equity_df["Equity"],
                name=f"{strategy} Strategy",
                line=dict(color="#1a56db", width=2),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

        # Buy-and-Hold baseline
        if not bnh.empty:
            bnh_aligned = bnh.reindex(equity_df.index).ffill().bfill()
            fig.add_trace(
                go.Scatter(
                    x=bnh_aligned.index, y=bnh_aligned.values,
                    name="Buy & Hold",
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

        # Drawdown area
        if not drawdown.empty:
            dd_aligned = drawdown.reindex(equity_df.index).ffill().bfill()
            fig.add_trace(
                go.Scatter(
                    x=dd_aligned.index, y=dd_aligned.values,
                    name="Drawdown",
                    fill="tozeroy",
                    line=dict(color="#ef4444", width=1),
                    fillcolor="rgba(239,68,68,0.18)",
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>",
                ),
                row=2, col=1,
            )

        fig.update_layout(
            height=560,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", size=12, color="#374151"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=60, b=40),
            hovermode="x unified",
        )
        fig.update_yaxes(
            gridcolor="#f0f0f0", gridwidth=1,
            showgrid=True, zeroline=False,
        )
        fig.update_xaxes(gridcolor="#f0f0f0", showgrid=True)
        fig.update_yaxes(tickprefix="$", row=1, col=1)
        fig.update_yaxes(ticksuffix="%", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Trade Log ─────────────────────────────────────────────────────────────
    st.markdown("### Trade Log")
    trades_df = result.get("trades", pd.DataFrame())

    if trades_df is not None and not trades_df.empty:

        def _style_action(val):
            color = "#16a34a" if val == "BUY" else "#dc2626"
            return f"color: {color}; font-weight: 700"

        def _style_pv(val):
            return "color: #1a56db"

        styled = trades_df.style.applymap(_style_action, subset=["Action"])

        # Format numeric columns if present
        fmt_map = {}
        if "Price" in trades_df.columns:
            fmt_map["Price"] = "${:,.2f}"
        if "Shares" in trades_df.columns:
            fmt_map["Shares"] = "{:,.4f}"
        if "Commission ($)" in trades_df.columns:
            fmt_map["Commission ($)"] = "${:,.2f}"
        if "Portfolio Value" in trades_df.columns:
            fmt_map["Portfolio Value"] = "${:,.2f}"
        if fmt_map:
            styled = styled.format(fmt_map)

        st.dataframe(styled, use_container_width=True, height=300)
        st.caption(f"Total: **{len(trades_df)}** trades recorded.")
    else:
        st.info("No trades were executed in this period. Try adjusting the strategy parameters or date range.")

    st.divider()

    # ── Monthly Returns Heatmap ───────────────────────────────────────────────
    st.markdown("### Monthly Returns Heatmap")

    if not equity_df.empty:
        try:
            monthly = equity_df["Equity"].resample("ME").last()
            monthly_ret = monthly.pct_change().dropna() * 100

            monthly_df = pd.DataFrame({
                "Year":  monthly_ret.index.year,
                "Month": monthly_ret.index.month,
                "Return": monthly_ret.values,
            })

            if not monthly_df.empty:
                pivot = monthly_df.pivot_table(
                    index="Year", columns="Month", values="Return", aggfunc="first"
                )
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot.columns = [month_names[m - 1] for m in pivot.columns]

                fig_heat = go.Figure(
                    go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale=[
                            [0.0,  "#dc2626"],
                            [0.5,  "#ffffff"],
                            [1.0,  "#16a34a"],
                        ],
                        zmid=0,
                        text=np.round(pivot.values, 1),
                        texttemplate="%{text}%",
                        textfont=dict(size=11),
                        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
                        colorbar=dict(
                            title="Return %",
                            ticksuffix="%",
                            len=0.8,
                        ),
                    )
                )
                fig_heat.update_layout(
                    title=dict(text="Monthly Returns (%)", font=dict(size=15, color="#1e293b")),
                    xaxis_title="Month",
                    yaxis_title="Year",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", size=12),
                    height=max(250, len(pivot) * 55 + 100),
                    margin=dict(l=60, r=80, t=50, b=40),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render monthly heatmap: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  COMPARE RUNS TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    runs = st.session_state.backtest_runs

    if not runs:
        st.info("Run at least one backtest to populate the comparison table. "
                "Run multiple backtests with different settings to compare them side-by-side.")
    else:
        st.markdown("### All Backtest Runs")

        compare_cols = [
            "Label", "Total Return %", "Ann. Return %",
            "Sharpe", "Max DD %", "Win Rate %", "Total Trades", "Final Value $",
        ]
        compare_df = pd.DataFrame(runs)[compare_cols].copy()

        def _color_return(val):
            if isinstance(val, (int, float)):
                return f"color: {'#16a34a' if val >= 0 else '#dc2626'}; font-weight: 700"
            return ""

        styled_cmp = (
            compare_df.style
            .applymap(_color_return, subset=["Total Return %", "Ann. Return %", "Max DD %"])
            .format({
                "Total Return %": "{:+.2f}%",
                "Ann. Return %":  "{:+.2f}%",
                "Sharpe":         "{:.3f}",
                "Max DD %":       "{:.2f}%",
                "Win Rate %":     "{:.1f}%",
                "Final Value $":  "${:,.2f}",
            })
        )
        st.dataframe(styled_cmp, use_container_width=True, height=300)

        # ── Equity overlay chart ──────────────────────────────────────────────
        if len(runs) >= 2:
            st.markdown("### Equity Curve Overlay")
            palette = [
                "#1a56db", "#16a34a", "#f59e0b", "#ef4444",
                "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
            ]
            fig_cmp = go.Figure()
            for i, run in enumerate(runs):
                res = run.get("result_obj", {})
                eq  = res.get("equity", pd.DataFrame())
                if not eq.empty:
                    # Normalise to 100
                    normalised = eq["Equity"] / eq["Equity"].iloc[0] * 100
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=normalised.index,
                            y=normalised.values,
                            name=run["Label"],
                            line=dict(color=palette[i % len(palette)], width=2),
                            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.1f}<extra></extra>",
                        )
                    )

            fig_cmp.update_layout(
                title="Normalised Equity (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (normalised)",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", size=12, color="#374151"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                height=420,
                margin=dict(l=60, r=20, t=60, b=40),
            )
            fig_cmp.update_yaxes(gridcolor="#f0f0f0", gridwidth=1)
            fig_cmp.update_xaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig_cmp, use_container_width=True)

        if st.button("Clear All Runs", type="secondary"):
            st.session_state.backtest_runs = []
            st.session_state.last_result   = None
            st.rerun()
