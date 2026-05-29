"""
Portfolio Manager – StockSense AI
===================================
Page 05: Full portfolio CRUD, analytics, and visualisations.
Requires authenticated session via auth.session_manager.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Portfolio Manager | StockSense AI",
        page_icon="💼",
        layout="wide",
    )

# ── Auth guard ────────────────────────────────────────────────────────────────
from auth.session_manager import require_login
user = require_login()
user_id = user["id"]

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY  = "#1a56db"
POSITIVE = "#16a34a"
NEGATIVE = "#dc2626"
NEUTRAL  = "#d97706"
BG_WHITE = "#ffffff"
GRID     = "#f3f4f6"

BASE_LAYOUT = dict(
    paper_bgcolor=BG_WHITE,
    plot_bgcolor=BG_WHITE,
    font=dict(family="Inter, sans-serif", color="#111827"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def apply_base(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, **kwargs)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor="#e5e7eb")
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def pnl_color(val: float) -> str:
    return POSITIVE if val >= 0 else NEGATIVE


def fmt_currency(val: float) -> str:
    return f"${val:,.2f}"


def fmt_pct(val: float) -> str:
    return f"{val * 100:+.2f}%"


def color_pnl_html(val: float, display: str) -> str:
    color = pnl_color(val)
    return f'<span style="color:{color};font-weight:600;">{display}</span>'


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1a56db 0%,#1e40af 100%);
                border-radius:12px;padding:28px 32px 20px;margin-bottom:24px;">
        <h1 style="color:#fff;margin:0;font-size:2rem;">Portfolio Manager</h1>
        <p style="color:#bfdbfe;margin:6px 0 0;font-size:1rem;">
            Track holdings, P&amp;L, allocation, and risk — all in one place
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Import service ─────────────────────────────────────────────────────────────
try:
    from portfolio.portfolio_service import (
        create_portfolio, get_portfolios, get_portfolio, delete_portfolio,
        add_holding, remove_holding, get_holdings, portfolio_analytics,
    )
except ImportError as exc:
    st.error(f"❌ Failed to import portfolio service: {exc}")
    st.stop()

# ── Sidebar: Create + Select portfolio ────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#1a56db;margin-bottom:8px;'>"
        "My Portfolios</div>",
        unsafe_allow_html=True,
    )

    # Create portfolio form
    with st.expander("Create New Portfolio", expanded=False):
        with st.form("create_portfolio_form", clear_on_submit=True):
            new_name = st.text_input("Portfolio Name", max_chars=60, placeholder="e.g. Growth Portfolio")
            new_desc = st.text_area("Description (optional)", max_chars=200, height=70)
            submitted = st.form_submit_button("Create Portfolio", use_container_width=True, type="primary")
            if submitted:
                if not new_name.strip():
                    st.warning("Please enter a portfolio name.")
                else:
                    try:
                        pid = create_portfolio(user_id, new_name.strip(), new_desc.strip())
                        st.success(f'✅ "{new_name}" created!')
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    st.markdown("---")

    # Portfolio selector
    try:
        portfolios = get_portfolios(user_id)
    except Exception as exc:
        st.error(f"Could not load portfolios: {exc}")
        portfolios = []

    if portfolios:
        port_names = {p["name"]: p["id"] for p in portfolios}
        selected_name = st.selectbox(
            "Select Portfolio",
            options=list(port_names.keys()),
            key="selected_portfolio_name",
        )
        selected_id = port_names[selected_name]
    else:
        selected_id   = None
        selected_name = None

    st.markdown("---")
    if selected_id:
        st.caption(f"Logged in as **{user.get('username', 'User')}**")
        if st.button("Delete This Portfolio", use_container_width=True):
            try:
                delete_portfolio(selected_id, user_id)
                st.success("Portfolio deleted.")
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")

# ── No portfolio state ────────────────────────────────────────────────────────

if selected_id is None:
    st.markdown(
        """
        <div style="text-align:center;padding:80px 24px;background:#f9fafb;
                    border-radius:12px;border:2px dashed #d1d5db;">
            <div style="font-size:3rem;">📂</div>
            <h2 style="color:#374151;margin:16px 0 8px;">No Portfolio Selected</h2>
            <p style="color:#6b7280;font-size:1rem;">
                Create your first portfolio using the sidebar, then start adding holdings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load analytics ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _load_analytics(portfolio_id: int) -> dict:
    return portfolio_analytics(portfolio_id)


with st.spinner("Loading portfolio data …"):
    try:
        analytics = _load_analytics(selected_id)
    except Exception as exc:
        st.error(f"❌ Failed to load portfolio analytics: {exc}")
        st.stop()

has_error    = "error" in analytics
holdings_det = analytics.get("holdings_detail", [])

# ── Tabs layout ───────────────────────────────────────────────────────────────

tab_overview, tab_holdings, tab_manage = st.tabs(["Overview", "Holdings", "Add / Remove"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    if has_error:
        st.info(
            "📭 This portfolio is empty. Go to **Add / Remove** to add your first holding.",
            icon="ℹ️",
        )
    else:
        total_value   = analytics.get("total_value", 0.0)
        total_cost    = analytics.get("total_cost", 0.0)
        total_pnl     = analytics.get("total_pnl", 0.0)
        total_pnl_pct = analytics.get("total_pnl_pct", 0.0)
        port_return   = analytics.get("portfolio_return", 0.0)
        sharpe        = analytics.get("sharpe_ratio", 0.0)
        risk_sc       = analytics.get("risk_score", 50.0)
        risk_cat      = analytics.get("risk_category", "Moderate")
        allocation    = analytics.get("allocation", [])

        # ── 5 KPI tiles ───────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Value",     fmt_currency(total_value))
        k2.metric(
            "Total P&L",
            fmt_currency(total_pnl),
            delta=fmt_currency(total_pnl),
            delta_color="normal",
        )
        k3.metric("P&L %",           fmt_pct(total_pnl_pct))
        k4.metric("Ann. Return",     fmt_pct(port_return))
        k5.metric("Sharpe Ratio",    f"{sharpe:.2f}")

        st.markdown("---")

        # ── Risk Gauge + Allocation Pie ────────────────────────────────────
        col_risk, col_alloc = st.columns([1, 1], gap="large")

        with col_risk:
            st.subheader("Risk Score")

            risk_step_color = (
                "#dcfce7" if risk_sc < 30 else
                "#fef3c7" if risk_sc < 60 else
                "#fee2e2"
            )
            risk_bar_color = (
                POSITIVE if risk_sc < 30 else
                NEUTRAL  if risk_sc < 60 else
                NEGATIVE
            )
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_sc,
                number={"suffix": " / 100", "font": {"size": 26, "color": risk_bar_color}},
                gauge={
                    "axis":    {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6b7280"},
                    "bar":     {"color": risk_bar_color, "thickness": 0.25},
                    "bgcolor": BG_WHITE,
                    "borderwidth": 2,
                    "bordercolor": "#e5e7eb",
                    "steps": [
                        {"range": [0, 30],   "color": "#dcfce7"},
                        {"range": [30, 60],  "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                },
                title={"text": f"<b>Risk Category:</b> {risk_cat}", "font": {"size": 14}},
            ))
            fig_risk.update_layout(**BASE_LAYOUT, height=300)
            st.plotly_chart(fig_risk, use_container_width=True)

        with col_alloc:
            st.subheader("Allocation Breakdown")
            if allocation:
                alloc_df = pd.DataFrame(allocation)
                fig_pie = go.Figure(go.Pie(
                    labels=alloc_df["symbol"],
                    values=alloc_df["value"],
                    hole=0.5,
                    textinfo="label+percent",
                    textfont_size=12,
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Value: $%{value:,.2f}<br>"
                        "Share: %{percent}<extra></extra>"
                    ),
                ))
                fig_pie.update_layout(
                    **BASE_LAYOUT,
                    height=300,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No allocation data.")

        # ── Portfolio return vs S&P 500 ────────────────────────────────────
        st.markdown("---")
        st.subheader("Portfolio Return vs S&P 500")

        try:
            from data_ingestion.market_data_service import fetch_ohlcv
            port_daily = analytics.get("daily_returns")
            spx_df     = fetch_ohlcv("^GSPC", period="1y")

            if port_daily is not None and not (hasattr(port_daily, "empty") and port_daily.empty):
                port_cum = (1 + port_daily).cumprod()

                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(
                    x=port_cum.index,
                    y=port_cum.values,
                    name=f"{selected_name}",
                    mode="lines",
                    line=dict(color=PRIMARY, width=2.5),
                    hovertemplate="%{y:.3f}x<extra></extra>",
                ))

                if not spx_df.empty:
                    close_col = "close" if "close" in spx_df.columns else spx_df.columns[0]
                    spx_ret = spx_df[close_col].pct_change().dropna()
                    spx_cum = (1 + spx_ret).cumprod()
                    fig_ret.add_trace(go.Scatter(
                        x=spx_cum.index,
                        y=spx_cum.values,
                        name="S&P 500",
                        mode="lines",
                        line=dict(color="#9ca3af", width=2, dash="dash"),
                        hovertemplate="S&P 500: %{y:.3f}x<extra></extra>",
                    ))

                fig_ret.add_hline(y=1.0, line_color="#d1d5db", line_dash="dot", line_width=1)
                apply_base(
                    fig_ret,
                    title=f"<b>{selected_name}</b> vs S&P 500 — Cumulative Return",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (×)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=380,
                )
                st.plotly_chart(fig_ret, use_container_width=True)
            else:
                st.info("Insufficient return data to render comparison chart.")
        except Exception as exc:
            st.warning(f"Return chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_holdings:

    st.subheader(f"Holdings — {selected_name}")

    if has_error or not holdings_det:
        st.info("📭 No holdings yet. Add positions in the **Add / Remove** tab.")
    else:
        # Build styled HTML table
        total_value = analytics.get("total_value", 0.0)

        header_cells = (
            "Symbol", "Shares", "Avg Cost", "Current Price",
            "Market Value", "P&L", "P&L %",
        )
        header_html = "".join(
            f"<th style='padding:10px 12px;text-align:{'right' if i > 0 else 'left'};"
            f"font-size:0.85rem;color:#374151;background:#f9fafb;'>{h}</th>"
            for i, h in enumerate(header_cells)
        )

        rows_html = ""
        for h in holdings_det:
            pnl      = h.get("pnl", 0.0)
            pnl_pct  = h.get("pnl_pct", 0.0)
            pnl_c    = pnl_color(pnl)
            rows_html += (
                f"<tr style='border-bottom:1px solid #f3f4f6;'>"
                f"<td style='padding:9px 12px;font-weight:700;color:{PRIMARY};'>{h['symbol']}</td>"
                f"<td style='padding:9px 12px;text-align:right;'>{h['shares']:,.4f}</td>"
                f"<td style='padding:9px 12px;text-align:right;'>{fmt_currency(h['avg_cost'])}</td>"
                f"<td style='padding:9px 12px;text-align:right;'>{fmt_currency(h.get('current_price', h['avg_cost']))}</td>"
                f"<td style='padding:9px 12px;text-align:right;font-weight:600;'>{fmt_currency(h.get('market_value', 0))}</td>"
                f"<td style='padding:9px 12px;text-align:right;font-weight:600;color:{pnl_c};'>{fmt_currency(pnl)}</td>"
                f"<td style='padding:9px 12px;text-align:right;font-weight:600;color:{pnl_c};'>{fmt_pct(pnl_pct)}</td>"
                f"</tr>"
            )

        # Summary row
        total_pnl     = analytics.get("total_pnl", 0.0)
        total_pnl_pct = analytics.get("total_pnl_pct", 0.0)
        tc             = pnl_color(total_pnl)
        rows_html += (
            f"<tr style='border-top:2px solid #1a56db;background:#f9fafb;font-weight:700;'>"
            f"<td style='padding:10px 12px;'>TOTAL</td>"
            f"<td colspan='3' style='padding:10px 12px;'></td>"
            f"<td style='padding:10px 12px;text-align:right;'>{fmt_currency(total_value)}</td>"
            f"<td style='padding:10px 12px;text-align:right;color:{tc};'>{fmt_currency(total_pnl)}</td>"
            f"<td style='padding:10px 12px;text-align:right;color:{tc};'>{fmt_pct(total_pnl_pct)}</td>"
            f"</tr>"
        )

        table_html = f"""
        <div style="overflow-x:auto;border:1px solid #e5e7eb;border-radius:8px;margin-top:8px;">
        <table style="width:100%;border-collapse:collapse;background:{BG_WHITE};">
          <thead><tr style="border-bottom:2px solid #e5e7eb;">{header_html}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # ── Per-holding P&L bar chart ──────────────────────────────────────
        st.markdown("---")
        st.subheader("P&L by Holding")
        pnl_df = pd.DataFrame([
            {"Symbol": h["symbol"], "P&L": h.get("pnl", 0.0), "P&L %": h.get("pnl_pct", 0.0) * 100}
            for h in holdings_det
        ])
        fig_pnl = go.Figure(go.Bar(
            x=pnl_df["Symbol"],
            y=pnl_df["P&L"],
            marker_color=[pnl_color(v) for v in pnl_df["P&L"]],
            text=[fmt_currency(v) for v in pnl_df["P&L"]],
            textposition="outside",
            hovertemplate="%{x}<br>P&L: %{y:$,.2f}<extra></extra>",
        ))
        apply_base(
            fig_pnl,
            title="Unrealised P&L per Holding",
            xaxis_title="Symbol",
            yaxis_title="P&L (USD)",
            height=340,
        )
        fig_pnl.add_hline(y=0, line_color="#9ca3af", line_dash="dot", line_width=1)
        st.plotly_chart(fig_pnl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ADD / REMOVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_manage:

    st.subheader(f"Manage Holdings — {selected_name}")

    # ── Add Holding form ──────────────────────────────────────────────────────
    st.markdown("#### Add Holding")
    with st.form("add_holding_form", clear_on_submit=True):
        col_sym, col_shr, col_cost = st.columns(3)
        with col_sym:
            sym_input = st.text_input(
                "Ticker Symbol",
                max_chars=10,
                placeholder="e.g. MSFT",
            ).strip().upper()
        with col_shr:
            shares_input = st.number_input(
                "Number of Shares",
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                value=1.0,
            )
        with col_cost:
            cost_input = st.number_input(
                "Average Cost per Share ($)",
                min_value=0.01,
                step=0.01,
                format="%.2f",
                value=100.0,
            )
        add_btn = st.form_submit_button("Add Holding", use_container_width=True, type="primary")

        if add_btn:
            if not sym_input:
                st.warning("Please enter a ticker symbol.")
            elif shares_input <= 0:
                st.warning("Shares must be greater than zero.")
            elif cost_input <= 0:
                st.warning("Average cost must be greater than zero.")
            else:
                try:
                    ok, msg = add_holding(selected_id, sym_input, shares_input, cost_input)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                except Exception as exc:
                    st.error(f"❌ Failed to add holding: {exc}")

    st.markdown("---")

    # ── Remove Holding ─────────────────────────────────────────────────────────
    st.markdown("#### Remove Holding")

    try:
        current_holdings = get_holdings(selected_id)
    except Exception as exc:
        st.error(f"Could not load holdings: {exc}")
        current_holdings = []

    if not current_holdings:
        st.info("No holdings to remove.")
    else:
        # Render each holding as a row with a remove button
        for holding in current_holdings:
            sym     = holding["symbol"]
            shares  = holding["shares"]
            cost    = holding["avg_cost"]
            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"<div style='padding:10px 0;font-size:0.9rem;'>"
                    f"<b style='color:{PRIMARY};'>{sym}</b> — "
                    f"{shares:,.4f} shares @ {fmt_currency(cost)} avg cost"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Remove", key=f"remove_{sym}_{selected_id}", type="secondary"):
                    try:
                        success = remove_holding(selected_id, sym)
                        if success:
                            st.success(f"✅ {sym} removed.")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.warning(f"{sym} not found in portfolio.")
                    except Exception as exc:
                        st.error(f"Remove failed: {exc}")

    st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.78rem;margin-top:40px;'>"
    "Portfolio data is for informational purposes only. "
    "Past performance is not indicative of future results."
    "</div>",
    unsafe_allow_html=True,
)
