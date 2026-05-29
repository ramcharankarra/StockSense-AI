"""
Sentiment Analysis Center – StockSense AI
==========================================
Page 04: Full sentiment dashboard powered by analysis.sentiment_engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Sentiment Center | StockSense AI",
        page_icon="📰",
        layout="wide",
    )

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY   = "#1a56db"
POSITIVE  = "#16a34a"
NEGATIVE  = "#dc2626"
NEUTRAL   = "#d97706"
BG_WHITE  = "#ffffff"
GRID      = "#f3f4f6"

# ── Shared chart layout ───────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor=BG_WHITE,
    plot_bgcolor=BG_WHITE,
    font=dict(family="Inter, sans-serif", color="#111827"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def apply_base(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, **kwargs)
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#e5e7eb",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#e5e7eb",
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def sentiment_color(label: str) -> str:
    label = (label or "").lower()
    if label == "positive":
        return POSITIVE
    if label == "negative":
        return NEGATIVE
    return NEUTRAL


def sentiment_badge(label: str) -> str:
    colors = {"positive": "#dcfce7", "negative": "#fee2e2", "neutral": "#fef3c7"}
    text   = {"positive": "#166534", "negative": "#991b1b", "neutral": "#92400e"}
    l = (label or "neutral").lower()
    bg  = colors.get(l, colors["neutral"])
    fg  = text.get(l, text["neutral"])
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">'
        f'{label.capitalize()}</span>'
    )


def extract_keywords(articles: list[dict], top_n: int = 20) -> pd.DataFrame:
    """Naive stopword-filtered word frequency from article titles."""
    STOPWORDS = {
        "the","a","an","in","on","at","to","of","and","or","for","with",
        "is","are","was","were","be","been","by","its","it","this","that",
        "as","from","but","not","have","has","had","will","would","could",
        "should","may","can","their","they","we","our","he","she","his",
        "her","about","after","before","more","also","into","than","s","t",
    }
    all_words: list[str] = []
    for art in articles:
        text = (art.get("title", "") + " " + art.get("description", "")).lower()
        words = re.findall(r"\b[a-z]{4,}\b", text)
        all_words.extend(w for w in words if w not in STOPWORDS)
    freq = Counter(all_words).most_common(top_n)
    return pd.DataFrame(freq, columns=["Keyword", "Count"])


def trading_implication(score: float, label: str) -> tuple[str, str, str]:
    """Returns (signal, rationale, border_color)."""
    if score >= 0.25:
        return (
            "Strong Buy Signal",
            "Overwhelmingly positive media coverage suggests strong market confidence. "
            "Consider initiating or adding to a long position, subject to technical confirmation.",
            POSITIVE,
        )
    if score >= 0.05:
        return (
            "Mild Buy Signal",
            "Sentiment is moderately positive. News flow supports a cautious accumulation "
            "strategy. Watch for volume confirmation before committing full position size.",
            "#22c55e",
        )
    if score <= -0.25:
        return (
            "Strong Sell Signal",
            "Significant negative sentiment detected. Risk of further downside exists. "
            "Consider reducing exposure or implementing hedges.",
            NEGATIVE,
        )
    if score <= -0.05:
        return (
            "Mild Sell Signal",
            "Sentiment is cautiously negative. Monitor closely; avoid adding to positions "
            "until a sustained improvement in news tone is observed.",
            "#f97316",
        )
    return (
        "Neutral — Hold",
        "Mixed or subdued news flow provides no clear directional signal. "
        "Maintain current position sizing and await a clearer catalyst.",
        NEUTRAL,
    )


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1a56db 0%,#1e40af 100%);
                border-radius:12px;padding:28px 32px 20px;margin-bottom:24px;">
        <h1 style="color:#fff;margin:0;font-size:2rem;">Sentiment Analysis Center</h1>
        <p style="color:#bfdbfe;margin:6px 0 0;font-size:1rem;">
            Real-time news sentiment scoring powered by VADER + TextBlob ensemble
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#1a56db;margin-bottom:8px;'>"
        "Sentiment Analysis</div>",
        unsafe_allow_html=True,
    )
    ticker_input   = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).strip().upper()
    company_input  = st.text_input("Company Name (optional)", value="Apple Inc.", max_chars=80)
    period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
    price_period   = st.selectbox("Price History Period", list(period_options.keys()), index=3)
    analyze_btn    = st.button("Analyze Sentiment", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Scores are computed using a 60 % VADER / 40 % TextBlob ensemble on article "
        "titles and descriptions. Scores range from **-1** (very negative) to **+1** (very positive)."
    )

# ── Session-state for result caching ─────────────────────────────────────────

if "sentiment_result" not in st.session_state:
    st.session_state.sentiment_result  = None
if "sentiment_ticker" not in st.session_state:
    st.session_state.sentiment_ticker  = ""
if "sentiment_pricedf" not in st.session_state:
    st.session_state.sentiment_pricedf = None

# ── Analysis trigger ──────────────────────────────────────────────────────────

if analyze_btn and ticker_input:
    with st.spinner(f"Fetching & scoring news for **{ticker_input}** …"):
        try:
            from analysis.sentiment_engine import analyze_sentiment
            from data_ingestion.market_data_service import fetch_ohlcv

            result = analyze_sentiment(ticker_input, company_input)
            price_df = fetch_ohlcv(ticker_input, period=period_options[price_period])
            st.session_state.sentiment_result  = result
            st.session_state.sentiment_ticker  = ticker_input
            st.session_state.sentiment_pricedf = price_df

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            st.stop()

# ── Display results ───────────────────────────────────────────────────────────

result   = st.session_state.sentiment_result
ticker   = st.session_state.sentiment_ticker
price_df = st.session_state.sentiment_pricedf

if result is None:
    st.info("👈 Enter a ticker in the sidebar and click **Analyze Sentiment** to begin.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────

overall_score = result.get("overall_score", 0.0)
overall_label = result.get("overall_label", "Neutral")
articles      = result.get("articles", [])
pos_pct       = result.get("positive_pct", 0.0)
neu_pct       = result.get("neutral_pct", 0.0)
neg_pct       = result.get("negative_pct", 0.0)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Overall Sentiment Score",  f"{overall_score:+.4f}", help="Ensemble score: -1 to +1")
kpi2.metric("Sentiment Label", overall_label)
kpi3.metric("Articles Analyzed", len(articles))

st.markdown("---")

# ── Row 1: Gauge + Donut ──────────────────────────────────────────────────────

col_gauge, col_donut = st.columns([1, 1], gap="large")

with col_gauge:
    st.subheader("Sentiment Gauge")
    gauge_color = sentiment_color(overall_label)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        number={"valueformat": "+.3f", "font": {"size": 32, "color": gauge_color}},
        delta={"reference": 0, "valueformat": "+.3f"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#6b7280"},
            "bar":  {"color": gauge_color, "thickness": 0.25},
            "bgcolor": BG_WHITE,
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [-1.0, -0.2], "color": "#fee2e2"},
                {"range": [-0.2, -0.05], "color": "#fef3c7"},
                {"range": [-0.05, 0.05], "color": "#f3f4f6"},
                {"range": [0.05,  0.20], "color": "#dcfce7"},
                {"range": [0.20,  1.00], "color": "#bbf7d0"},
            ],
            "threshold": {
                "line":  {"color": PRIMARY, "width": 3},
                "thickness": 0.75,
                "value": overall_score,
            },
        },
        title={"text": f"<b>{ticker}</b> Sentiment Score", "font": {"size": 16}},
    ))
    fig_gauge.update_layout(**BASE_LAYOUT, height=320)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_donut:
    st.subheader("Sentiment Distribution")
    if pos_pct + neu_pct + neg_pct > 0:
        fig_donut = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos_pct, neu_pct, neg_pct],
            hole=0.55,
            marker_colors=[POSITIVE, NEUTRAL, NEGATIVE],
            textinfo="label+percent",
            textfont_size=13,
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig_donut.update_layout(
            **BASE_LAYOUT,
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            annotations=[dict(
                text=f"<b>{pos_pct:.0f}%</b><br>Positive",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font_color=POSITIVE,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.warning("No distribution data available.")

# ── Row 2: Sentiment Trend ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Sentiment Trend (Articles Over Time)")

if articles:
    # Build a DataFrame from articles; parse published_at
    art_df = pd.DataFrame(articles)
    date_col = None
    for cname in ("published_at", "publishedAt", "date", "pub_date"):
        if cname in art_df.columns:
            date_col = cname
            break

    if date_col:
        try:
            art_df["_date"] = pd.to_datetime(art_df[date_col], utc=True, errors="coerce")
            art_df = art_df.dropna(subset=["_date"]).sort_values("_date")
            art_df["_date_label"] = art_df["_date"].dt.strftime("%b %d, %Y")

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=art_df["_date"],
                y=art_df["score"],
                mode="lines+markers",
                name="Article Score",
                line=dict(color=PRIMARY, width=2),
                marker=dict(
                    color=[sentiment_color(l) for l in art_df.get("label", ["neutral"] * len(art_df))],
                    size=8, line=dict(width=1, color="#fff"),
                ),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Score: %{y:+.4f}<extra></extra>"
                ),
                customdata=art_df.get("title", art_df["_date_label"]),
            ))
            # Rolling 3-article average
            if len(art_df) >= 3:
                rolling = art_df["score"].rolling(3, min_periods=1).mean()
                fig_trend.add_trace(go.Scatter(
                    x=art_df["_date"],
                    y=rolling,
                    mode="lines",
                    name="3-Article Rolling Avg",
                    line=dict(color=NEUTRAL, width=2, dash="dash"),
                    hovertemplate="Rolling Avg: %{y:+.4f}<extra></extra>",
                ))
            # Zero line
            fig_trend.add_hline(y=0, line_color="#9ca3af", line_dash="dot", line_width=1)
            apply_base(
                fig_trend,
                title=f"<b>{ticker}</b> — Article Sentiment Over Time",
                xaxis_title="Publication Date",
                yaxis_title="Sentiment Score",
                yaxis_range=[-1.1, 1.1],
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render trend chart: {exc}")
    else:
        # Fallback: plot by article index
        scores = [a.get("score", 0.0) for a in articles]
        fig_trend = go.Figure(go.Scatter(
            x=list(range(1, len(scores) + 1)),
            y=scores,
            mode="lines+markers",
            line=dict(color=PRIMARY, width=2),
            marker=dict(color=[sentiment_color(a.get("label", "neutral")) for a in articles], size=8),
            name="Article Score",
        ))
        fig_trend.add_hline(y=0, line_color="#9ca3af", line_dash="dot", line_width=1)
        apply_base(
            fig_trend,
            title=f"<b>{ticker}</b> — Article Sentiment (by index)",
            xaxis_title="Article #",
            yaxis_title="Sentiment Score",
            yaxis_range=[-1.1, 1.1],
            height=340,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No articles available for trend chart.")

# ── Row 3: Articles Table ─────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Articles Table")

if articles:
    MAX_TITLE = 90

    def _safe_date(val: str) -> str:
        try:
            return pd.to_datetime(val, utc=True).strftime("%b %d, %Y")
        except Exception:
            return val or "—"

    rows_html = ""
    for art in articles:
        title  = (art.get("title") or "")[:MAX_TITLE] + ("…" if len(art.get("title", "")) > MAX_TITLE else "")
        source = art.get("source") or art.get("source_name") or "—"
        label  = art.get("label", "neutral")
        score  = art.get("score", 0.0)
        date   = _safe_date(art.get("published_at") or art.get("publishedAt") or "")
        badge  = sentiment_badge(label)
        score_color = sentiment_color(label)
        rows_html += (
            f"<tr>"
            f"<td style='padding:8px 10px;max-width:380px;font-size:0.85rem;'>{title}</td>"
            f"<td style='padding:8px 10px;font-size:0.85rem;white-space:nowrap;'>{source}</td>"
            f"<td style='padding:8px 10px;'>{badge}</td>"
            f"<td style='padding:8px 10px;font-weight:600;color:{score_color};text-align:right;'>{score:+.4f}</td>"
            f"<td style='padding:8px 10px;font-size:0.85rem;white-space:nowrap;'>{date}</td>"
            f"</tr>"
        )

    table_html = f"""
    <div style="overflow-x:auto;border:1px solid #e5e7eb;border-radius:8px;">
    <table style="width:100%;border-collapse:collapse;background:{BG_WHITE};">
      <thead>
        <tr style="background:#f9fafb;border-bottom:2px solid #e5e7eb;">
          <th style="padding:10px 10px;text-align:left;font-size:0.85rem;color:#374151;">Title</th>
          <th style="padding:10px 10px;text-align:left;font-size:0.85rem;color:#374151;">Source</th>
          <th style="padding:10px 10px;text-align:left;font-size:0.85rem;color:#374151;">Sentiment</th>
          <th style="padding:10px 10px;text-align:right;font-size:0.85rem;color:#374151;">Score</th>
          <th style="padding:10px 10px;text-align:left;font-size:0.85rem;color:#374151;">Date</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.info("No articles to display.")

# ── Row 4: Keyword Bar Chart ──────────────────────────────────────────────────

st.markdown("---")
st.subheader("Top Keywords (Word Cloud Alternative)")

if articles:
    kw_df = extract_keywords(articles, top_n=20)
    if not kw_df.empty:
        fig_kw = px.bar(
            kw_df,
            x="Count",
            y="Keyword",
            orientation="h",
            color="Count",
            color_continuous_scale=[[0, "#bfdbfe"], [1, PRIMARY]],
            text="Count",
            labels={"Count": "Frequency", "Keyword": ""},
        )
        fig_kw.update_traces(textposition="outside", cliponaxis=False)
        fig_kw.update_layout(
            **BASE_LAYOUT,
            title=f"<b>{ticker}</b> — Most Frequent News Keywords",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=480,
        )
        fig_kw.update_xaxes(title="Frequency")
        fig_kw.update_yaxes(showgrid=False)
        st.plotly_chart(fig_kw, use_container_width=True)
    else:
        st.info("No keywords extracted.")
else:
    st.info("Analyze a ticker to see keyword frequencies.")

# ── Row 5: Sentiment vs Price Correlation ─────────────────────────────────────

st.markdown("---")
st.subheader("Sentiment vs Price Correlation")

if price_df is not None and not price_df.empty and articles:
    try:
        art_df2 = pd.DataFrame(articles)
        date_col2 = next(
            (c for c in ("published_at", "publishedAt", "date") if c in art_df2.columns), None
        )

        fig_corr = go.Figure()

        # Price trace (primary y)
        close_col = "close" if "close" in price_df.columns else price_df.columns[0]
        fig_corr.add_trace(go.Scatter(
            x=price_df.index,
            y=price_df[close_col],
            name="Close Price",
            mode="lines",
            line=dict(color=PRIMARY, width=2),
            yaxis="y1",
        ))

        # Sentiment scatter (secondary y)
        if date_col2:
            art_df2["_d"] = pd.to_datetime(art_df2[date_col2], utc=True, errors="coerce")
            art_df2 = art_df2.dropna(subset=["_d"])
            fig_corr.add_trace(go.Scatter(
                x=art_df2["_d"],
                y=art_df2["score"],
                name="Article Sentiment",
                mode="markers",
                marker=dict(
                    color=[sentiment_color(l) for l in art_df2.get("label", ["neutral"] * len(art_df2))],
                    size=9, symbol="diamond",
                    line=dict(width=1, color="#fff"),
                ),
                yaxis="y2",
                hovertemplate="Score: %{y:+.4f}<extra></extra>",
            ))

        fig_corr.update_layout(
            **BASE_LAYOUT,
            title=f"<b>{ticker}</b> — Price vs Sentiment",
            xaxis_title="Date",
            yaxis=dict(
                title="Close Price (USD)", side="left",
                showgrid=True, gridcolor=GRID,
            ),
            yaxis2=dict(
                title="Sentiment Score",
                overlaying="y", side="right",
                range=[-1.2, 1.2], showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render correlation chart: {exc}")
else:
    if price_df is None or (hasattr(price_df, "empty") and price_df.empty):
        st.info("Price data unavailable for correlation chart.")
    else:
        st.info("Run sentiment analysis to see price correlation.")

# ── Row 6: Trading Implication Card ──────────────────────────────────────────

st.markdown("---")
st.subheader("Trading Implication")

signal, rationale, border_color = trading_implication(overall_score, overall_label)

try:
    from analysis.sentiment_engine import sentiment_signal
    signal_label = sentiment_signal(overall_score)
except Exception:
    signal_label = overall_label

st.markdown(
    f"""
    <div style="border-left:5px solid {border_color};background:#f9fafb;
                border-radius:0 10px 10px 0;padding:20px 24px;margin-top:4px;">
        <div style="font-size:1.25rem;font-weight:700;color:{border_color};margin-bottom:8px;">
            {signal}
        </div>
        <div style="font-size:0.95rem;color:#374151;line-height:1.6;">
            {rationale}
        </div>
        <div style="margin-top:12px;font-size:0.85rem;color:#6b7280;">
            <b>Signal category:</b> {signal_label} &nbsp;|&nbsp;
            <b>Ensemble score:</b> {overall_score:+.4f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.78rem;margin-top:32px;'>"
    "Sentiment analysis is for informational purposes only and does not constitute financial advice."
    "</div>",
    unsafe_allow_html=True,
)
