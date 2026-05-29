"""
🤖 AI Prediction Center
========================
Two-phase ML workflow:
  Phase 1 – Train models (Linear Regression, Random Forest, XGBoost, LSTM, GRU)
  Phase 2 – Display predictions, metrics, explainability, and horizon comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from config import PREDICTION_HORIZONS, DEFAULT_WATCHLIST, MODELS_DIR
from data_ingestion.market_data_service import fetch_ohlcv

# ── TensorFlow availability check ─────────────────────────────────────────────
_TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401
    _TF_AVAILABLE = True
except Exception:
    pass

# ── Design tokens ─────────────────────────────────────────────────────────────
BLUE        = "#1a56db"
BLUE_LIGHT  = "#e8f0fe"
GREEN       = "#16a34a"
GREEN_LIGHT = "#dcfce7"
RED         = "#dc2626"
RED_LIGHT   = "#fee2e2"
AMBER       = "#d97706"
AMBER_LIGHT = "#fef3c7"
GREY_BG     = "#f9fafb"
BORDER      = "#e5e7eb"

MODEL_COLOURS = {
    "Linear Regression": "#1a56db",
    "Random Forest":     "#16a34a",
    "XGBoost":           "#f59e0b",
    "LSTM":              "#8b5cf6",
    "GRU":               "#ec4899",
}

ALL_MODELS = ["Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU"]
TRADITIONAL_MODELS = ["Linear Regression", "Random Forest", "XGBoost"]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chart_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply institutional white-background chart theme."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#111827", family="Inter, sans-serif")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#374151"),
        height=height,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(
            bgcolor="white",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="#f3f4f6",
            linecolor=BORDER,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#f3f4f6",
            linecolor=BORDER,
            showgrid=True,
            zeroline=False,
        ),
    )
    return fig


def _badge(text: str, bg: str, fg: str = "white") -> str:
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 10px;"
        f"border-radius:12px;font-size:0.78rem;font-weight:700;'>{text}</span>"
    )


def _metric_card(label: str, value: str, sub: str = "", colour: str = BLUE) -> str:
    return f"""
    <div style='background:white;border:1px solid {BORDER};border-radius:10px;
                padding:1rem 1.2rem;border-top:3px solid {colour};'>
      <div style='font-size:0.78rem;color:#6b7280;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.05em;'>{label}</div>
      <div style='font-size:1.6rem;font-weight:800;color:#111827;margin:0.2rem 0;'>{value}</div>
      <div style='font-size:0.8rem;color:#6b7280;'>{sub}</div>
    </div>"""


def _section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""<div style='margin:1.5rem 0 0.75rem;'>
              <div style='font-size:1.05rem;font-weight:700;color:#111827;'>{title}</div>
              {'<div style="font-size:0.82rem;color:#6b7280;">' + subtitle + '</div>' if subtitle else ''}
            </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Training
# ══════════════════════════════════════════════════════════════════════════════

def _run_training(df: pd.DataFrame, symbol: str, horizon: int, selected_models: list[str]) -> dict:
    """Train selected models and return results dict {model_name: metrics_dict}."""
    from ml.model_trainer import (
        train_linear_regression,
        train_random_forest,
        train_xgboost,
    )

    trainers = {
        "Linear Regression": lambda: train_linear_regression(df, symbol, horizon),
        "Random Forest":     lambda: train_random_forest(df, symbol, horizon),
        "XGBoost":           lambda: train_xgboost(df, symbol, horizon),
    }

    if _TF_AVAILABLE:
        from ml.model_trainer import train_lstm, train_gru
        trainers["LSTM"] = lambda: train_lstm(df, symbol, horizon)
        trainers["GRU"]  = lambda: train_gru(df, symbol, horizon)

    results = {}
    total = len(selected_models)

    status_container = st.status("Training models…", expanded=True)
    progress_bar = st.progress(0)

    with status_container:
        for idx, model_name in enumerate(selected_models):
            if model_name not in trainers:
                st.warning(f"{model_name} requires TensorFlow (not installed). Skipping.")
                progress_bar.progress((idx + 1) / total)
                continue

            st.write(f"Training **{model_name}**…")
            t0 = time.time()
            try:
                result = trainers[model_name]()
                elapsed = time.time() - t0
                results[model_name] = result
                rmse  = result.get("rmse", float("nan"))
                r2    = result.get("r2", float("nan"))
                st.write(
                    f"  {model_name} — RMSE: **{rmse:.4f}** | R²: **{r2:.4f}** "
                    f"| {elapsed:.1f}s"
                )
            except Exception as exc:
                st.write(f"  {model_name} failed: {exc}")

            progress_bar.progress((idx + 1) / total)

        status_container.update(label="✅ Training complete!", state="complete", expanded=False)

    progress_bar.empty()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 – Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _render_model_comparison_table(results: dict, current_price: float) -> str:
    """Render coloured comparison table and return best model name."""
    from ml.model_evaluator import rank_models

    ranked = rank_models(results)
    best_name = ranked[0][0] if ranked else ""

    rows_html = ""
    for name, m in ranked:
        pred  = m.get("predicted_price", current_price)
        chg   = ((pred - current_price) / current_price * 100) if current_price else 0
        rmse  = m.get("rmse", float("nan"))
        mae   = m.get("mae", float("nan"))
        mape  = m.get("mape", float("nan"))
        r2    = m.get("r2", float("nan"))

        best_badge = (
            " " + _badge("★ Best", GREEN, "white")
            if name == best_name else ""
        )
        chg_colour = GREEN if chg >= 0 else RED
        arrow      = "▲" if chg >= 0 else "▼"

        rows_html += f"""
        <tr style='border-bottom:1px solid {BORDER};'>
          <td style='padding:0.6rem 0.8rem;font-weight:600;color:#111827;'>
            {name}{best_badge}
          </td>
          <td style='padding:0.6rem 0.8rem;text-align:right;'>{rmse:.4f}</td>
          <td style='padding:0.6rem 0.8rem;text-align:right;'>{mae:.4f}</td>
          <td style='padding:0.6rem 0.8rem;text-align:right;'>{mape:.2f}%</td>
          <td style='padding:0.6rem 0.8rem;text-align:right;'>{r2:.4f}</td>
          <td style='padding:0.6rem 0.8rem;text-align:right;font-weight:700;'>
            ${pred:,.2f}
          </td>
          <td style='padding:0.6rem 0.8rem;text-align:right;font-weight:700;color:{chg_colour};'>
            {arrow} {abs(chg):.2f}%
          </td>
        </tr>"""

    table_html = f"""
    <div style='overflow-x:auto;'>
      <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
        <thead>
          <tr style='background:{BLUE_LIGHT};'>
            <th style='padding:0.65rem 0.8rem;text-align:left;color:{BLUE};'>Model</th>
            <th style='padding:0.65rem 0.8rem;text-align:right;color:{BLUE};'>RMSE</th>
            <th style='padding:0.65rem 0.8rem;text-align:right;color:{BLUE};'>MAE</th>
            <th style='padding:0.65rem 0.8rem;text-align:right;color:{BLUE};'>MAPE</th>
            <th style='padding:0.65rem 0.8rem;text-align:right;color:{BLUE};'>R²</th>
            <th style='padding:0.65rem 0.8rem;text-align:right;color:{BLUE};'>Predicted Price</th>
            <th style='padding:0.65rem 0.8rem;text-align:right;color:{BLUE};'>Change %</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

    st.markdown(table_html, unsafe_allow_html=True)
    return best_name


def _render_actual_vs_predicted(results: dict) -> None:
    """Line chart: test-set actuals vs predictions for each model."""
    fig = go.Figure()
    first = True

    for model_name, m in results.items():
        actuals = m.get("actuals")
        preds   = m.get("predictions")
        if actuals is None or preds is None:
            continue

        n     = min(len(actuals), len(preds))
        x_idx = list(range(n))
        colour = MODEL_COLOURS.get(model_name, BLUE)

        if first:
            fig.add_trace(go.Scatter(
                x=x_idx, y=actuals[:n],
                name="Actual",
                line=dict(color="#111827", width=2),
                mode="lines",
            ))
            first = False

        fig.add_trace(go.Scatter(
            x=x_idx, y=preds[:n],
            name=model_name,
            line=dict(color=colour, width=1.5, dash="dash"),
            mode="lines",
        ))

    _chart_layout(fig, "Test-Set: Actual vs Predicted Price", height=420)
    fig.update_xaxes(title_text="Test Sample Index")
    fig.update_yaxes(title_text="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_chart(results: dict) -> None:
    """Radar chart comparing models across normalised RMSE, MAE, R² dimensions."""
    if len(results) < 2:
        st.info("Train at least 2 models to see the radar comparison.")
        return

    categories = ["RMSE (inv)", "MAE (inv)", "MAPE (inv)", "R²"]

    # Normalise: higher = better  → invert error metrics
    all_rmse = [m["rmse"] for m in results.values() if "rmse" in m]
    all_mae  = [m["mae"]  for m in results.values() if "mae"  in m]
    all_mape = [m["mape"] for m in results.values() if "mape" in m]

    max_rmse = max(all_rmse) if all_rmse else 1
    max_mae  = max(all_mae)  if all_mae  else 1
    max_mape = max(all_mape) if all_mape else 1

    fig = go.Figure()

    for model_name, m in results.items():
        rmse_n = 1 - (m.get("rmse", 0) / max_rmse) if max_rmse else 0
        mae_n  = 1 - (m.get("mae",  0) / max_mae)  if max_mae  else 0
        mape_n = 1 - (m.get("mape", 0) / max_mape) if max_mape else 0
        r2_n   = max(0, min(1, m.get("r2", 0)))

        vals = [rmse_n, mae_n, mape_n, r2_n]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]

        colour = MODEL_COLOURS.get(model_name, BLUE)
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            name=model_name,
            line=dict(color=colour, width=2),
            fillcolor=colour.replace(")", ",0.12)").replace("rgb(", "rgba("),
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e5e7eb"),
            angularaxis=dict(gridcolor="#e5e7eb"),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        showlegend=True,
        title=dict(text="Model Performance Radar", font=dict(size=15, color="#111827")),
        height=420,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(bgcolor="white", bordercolor=BORDER, borderwidth=1),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_prediction_summary(
    results: dict,
    best_name: str,
    current_price: float,
    horizon_label: str,
    horizon_days: int,
) -> None:
    """Summary card for the best model's prediction."""
    m     = results.get(best_name, {})
    pred  = m.get("predicted_price", current_price)
    chg   = ((pred - current_price) / current_price * 100) if current_price else 0
    arrow = "▲" if chg >= 0 else "▼"
    chg_c = GREEN if chg >= 0 else RED

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{BLUE_LIGHT},{BLUE}22);
                    border:1px solid {BLUE}44;border-radius:14px;padding:1.4rem 1.8rem;
                    margin-bottom:1rem;'>
          <div style='font-size:0.8rem;font-weight:600;color:{BLUE};
                      text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;'>
            Best Model Prediction — {best_name}
          </div>
          <div style='display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center;'>
            <div>
              <div style='font-size:0.75rem;color:#6b7280;'>Current Price</div>
              <div style='font-size:1.7rem;font-weight:800;color:#111827;'>${current_price:,.2f}</div>
            </div>
            <div style='font-size:2rem;color:#9ca3af;'>→</div>
            <div>
              <div style='font-size:0.75rem;color:#6b7280;'>Predicted ({horizon_label})</div>
              <div style='font-size:1.7rem;font-weight:800;color:{BLUE};'>${pred:,.2f}</div>
            </div>
            <div>
              <div style='font-size:0.75rem;color:#6b7280;'>Expected Change</div>
              <div style='font-size:1.7rem;font-weight:800;color:{chg_c};'>
                {arrow} {abs(chg):.2f}%
              </div>
            </div>
            <div>
              <div style='font-size:0.75rem;color:#6b7280;'>Horizon</div>
              <div style='font-size:1.1rem;font-weight:700;color:#374151;'>
                {horizon_days} trading day{'s' if horizon_days != 1 else ''}
              </div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_shap_section(df: pd.DataFrame, symbol: str, horizon: int) -> None:
    """SHAP feature importance horizontal bar chart."""
    _section_header("SHAP Explainability", "Top 10 features driving the XGBoost prediction")

    try:
        from ml.explainability import compute_shap, shap_summary_df

        with st.spinner("Computing SHAP values…"):
            result = compute_shap(df, symbol, "xgboost", horizon)

        if result is None:
            st.warning(
                "SHAP values unavailable — make sure the **SHAP** library is installed "
                "(`pip install shap`) and the XGBoost model has been trained."
            )
            return

        shap_vals, feature_names = result
        imp_df = shap_summary_df(shap_vals, feature_names).head(10)

        fig = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(
                color=imp_df["importance"],
                colorscale=[[0, BLUE_LIGHT], [1, BLUE]],
                showscale=False,
            ),
        ))
        _chart_layout(fig, "Top 10 SHAP Feature Importances (XGBoost)", height=380)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(title_text="Mean |SHAP Value|")
        fig.update_yaxes(title_text="Feature")
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Install `shap` to enable explainability: `pip install shap`")
    except Exception as exc:
        st.error(f"SHAP computation error: {exc}")


def _render_horizon_comparison(df: pd.DataFrame, symbol: str, best_name: str, current_price: float) -> None:
    """Compare the best model's prediction across 1d, 5d, 21d horizons."""
    _section_header(
        "Prediction Horizon Comparison",
        f"How {best_name} forecasts evolve across time horizons",
    )

    horizons = {"Next Day (1d)": 1, "Next Week (5d)": 5, "Next Month (21d)": 21}
    horizon_labels = list(horizons.keys())
    horizon_days   = list(horizons.values())

    try:
        from ml.model_trainer import (
            train_linear_regression,
            train_random_forest,
            train_xgboost,
        )

        trainer_map = {
            "Linear Regression": train_linear_regression,
            "Random Forest":     train_random_forest,
            "XGBoost":           train_xgboost,
        }

        if _TF_AVAILABLE:
            from ml.model_trainer import train_lstm, train_gru
            trainer_map["LSTM"] = train_lstm
            trainer_map["GRU"]  = train_gru

        trainer_fn = trainer_map.get(best_name)
        if trainer_fn is None:
            st.info(f"Horizon comparison not available for {best_name}.")
            return

        preds  = []
        labels = []
        cols   = st.columns(3)

        for i, (label, days) in enumerate(horizons.items()):
            try:
                if best_name in TRADITIONAL_MODELS:
                    res = trainer_fn(df, symbol, days)
                else:
                    res = trainer_fn(df, symbol, days)

                pred = res.get("predicted_price", current_price)
                chg  = ((pred - current_price) / current_price * 100) if current_price else 0
                preds.append(pred)
                labels.append(label)

                chg_c = GREEN if chg >= 0 else RED
                arrow = "▲" if chg >= 0 else "▼"
                with cols[i]:
                    st.markdown(
                        f"""<div style='background:white;border:1px solid {BORDER};border-radius:10px;
                                        padding:1rem;text-align:center;border-top:3px solid {BLUE};'>
                              <div style='font-size:0.78rem;color:#6b7280;font-weight:600;'>{label}</div>
                              <div style='font-size:1.4rem;font-weight:800;color:#111827;margin:0.3rem 0;'>
                                ${pred:,.2f}
                              </div>
                              <div style='font-size:0.9rem;font-weight:700;color:{chg_c};'>
                                {arrow} {abs(chg):.2f}%
                              </div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                preds.append(current_price)
                labels.append(label)
                with cols[i]:
                    st.warning(f"{label}: {exc}")

        # Bar chart comparison
        fig = go.Figure()
        bar_colours = [GREEN if p >= current_price else RED for p in preds]
        fig.add_trace(go.Bar(
            x=labels,
            y=preds,
            marker_color=bar_colours,
            text=[f"${p:,.2f}" for p in preds],
            textposition="outside",
        ))
        fig.add_hline(
            y=current_price,
            line=dict(color=BLUE, width=2, dash="dot"),
            annotation_text=f"Current: ${current_price:,.2f}",
            annotation_position="top right",
        )
        _chart_layout(fig, f"{best_name} — Price Forecast by Horizon", height=360)
        fig.update_yaxes(title_text="Predicted Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Horizon comparison failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Page entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{BLUE}12,{BLUE}06);
                    border-bottom:1px solid {BORDER};padding:1.4rem 0 1rem;margin-bottom:1rem;'>
          <div style='display:flex;align-items:center;gap:0.8rem;'>
            <div>
              <h1 style='margin:0;font-size:1.6rem;font-weight:800;color:#111827;'>
                AI Prediction Center
              </h1>
              <p style='margin:0;color:#6b7280;font-size:0.9rem;'>
                Train ML models and generate stock price forecasts with explainability
              </p>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── TF warning ────────────────────────────────────────────────────────────
    if not _TF_AVAILABLE:
        st.warning(
            "TensorFlow not detected — LSTM and GRU models are disabled. "
            "Install with `pip install tensorflow` to enable deep-learning models."
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuration")
        st.markdown("---")

        ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("pred_ticker", "AAPL"),
            placeholder="e.g. AAPL, MSFT",
            help="Enter a valid stock ticker symbol",
        ).upper().strip()
        st.session_state["pred_ticker"] = ticker

        period = st.selectbox(
            "Training Data Period",
            ["1y", "2y", "3y", "5y"],
            index=1,
            help="How much historical data to train on",
        )

        horizon_label = st.selectbox(
            "Prediction Horizon",
            list(PREDICTION_HORIZONS.keys()),
            index=0,
        )
        horizon_days = PREDICTION_HORIZONS[horizon_label]

        available_models = TRADITIONAL_MODELS + (
            ["LSTM", "GRU"] if _TF_AVAILABLE else []
        )
        model_choice = st.selectbox(
            "Model Selection",
            ["All Models"] + available_models,
            index=0,
        )

        selected_models = available_models if model_choice == "All Models" else [model_choice]

        st.markdown("---")
        train_btn   = st.button("Train Models",   use_container_width=True, type="primary")
        predict_btn = st.button("Predict",        use_container_width=True)

        st.markdown("---")
        st.markdown(
            f"""<div style='font-size:0.78rem;color:#6b7280;'>
              <b>Selected Models:</b><br>{'<br>'.join(f'• {m}' for m in selected_models)}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Validate ticker ───────────────────────────────────────────────────────
    if not ticker:
        st.info("👈 Enter a ticker symbol in the sidebar to get started.")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    @st.cache_data(ttl=900, show_spinner=False)
    def _load_data(sym: str, prd: str) -> pd.DataFrame:
        return fetch_ohlcv(sym, period=prd)

    with st.spinner(f"📡 Fetching {ticker} data…"):
        df = _load_data(ticker, period)

    if df is None or df.empty:
        st.error(
            f"❌ No data found for **{ticker}**. "
            "Please check the ticker symbol and try again."
        )
        return

    current_price = float(df["close"].iloc[-1])

    # ── Attach predicted_price to training results ─────────────────────────
    def _attach_prediction(results: dict, sym: str, hor: int) -> dict:
        """Run inference after training to get the forward prediction price."""
        try:
            from ml.predictor import predict_all
            preds = predict_all(df, sym, hor)
            for name in results:
                if name in preds and preds[name] is not None:
                    results[name]["predicted_price"] = preds[name]
                else:
                    results[name]["predicted_price"] = current_price
        except Exception:
            for name in results:
                results[name].setdefault("predicted_price", current_price)
        return results

    # ── Phase 1: Training ─────────────────────────────────────────────────────
    if train_btn:
        if len(df) < 120:
            st.error(
                f"❌ Insufficient data: {len(df)} rows. Need at least 120 rows. "
                "Try a longer period (2y or 5y)."
            )
            return

        st.markdown(f"### Training on **{ticker}** — {len(df)} days of data")
        try:
            results = _run_training(df, ticker, horizon_days, selected_models)
            results = _attach_prediction(results, ticker, horizon_days)
            st.session_state[f"train_results_{ticker}_{horizon_days}"] = results
            st.session_state[f"train_meta_{ticker}_{horizon_days}"] = {
                "horizon_label": horizon_label,
                "horizon_days":  horizon_days,
                "current_price": current_price,
            }
            st.success(f"✅ Training complete! {len(results)} model(s) trained successfully.")
        except Exception as exc:
            st.error(f"❌ Training failed: {exc}")
        return

    # ── Phase 2: Predictions & Analysis ──────────────────────────────────────
    results_key = f"train_results_{ticker}_{horizon_days}"
    meta_key    = f"train_meta_{ticker}_{horizon_days}"

    results = st.session_state.get(results_key)
    meta    = st.session_state.get(meta_key, {})

    if predict_btn and results is None:
        st.warning(
            "No trained models found for this configuration. "
            "Click **Train Models** first."
        )
        return

    if results is None:
        # Welcome state
        st.markdown(
            f"""
            <div style='background:{BLUE_LIGHT};border:1px dashed {BLUE};border-radius:12px;
                        padding:2rem;text-align:center;margin-top:2rem;'>
              <div style='font-size:1.1rem;font-weight:700;color:{BLUE};'>
                Ready to train on {ticker}
              </div>
              <div style='color:#6b7280;margin-top:0.4rem;'>
                Select your models and click <b>Train Models</b> in the sidebar to begin.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    if not results:
        st.error("❌ All models failed to train. Please check the logs.")
        return

    hl     = meta.get("horizon_label", horizon_label)
    hd     = meta.get("horizon_days",  horizon_days)
    cp     = meta.get("current_price", current_price)

    # ── Model Comparison Table ────────────────────────────────────────────────
    _section_header(
        "Model Performance Comparison",
        f"Ranked by RMSE (lower = better) · Horizon: {hl}",
    )
    best_name = _render_model_comparison_table(results, cp)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction Summary Card ───────────────────────────────────────────────
    _section_header("Prediction Summary")
    _render_prediction_summary(results, best_name, cp, hl, hd)

    st.markdown("---")

    # ── Charts: Actual vs Predicted + Radar ───────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        _section_header("Actual vs Predicted (Test Set)")
        _render_actual_vs_predicted(results)

    with chart_col2:
        _section_header("Model Radar Comparison")
        _render_radar_chart(results)

    st.markdown("---")

    # ── SHAP Explainability ───────────────────────────────────────────────────
    _render_shap_section(df, ticker, hd)

    st.markdown("---")

    # ── Horizon Comparison ────────────────────────────────────────────────────
    _render_horizon_comparison(df, ticker, best_name, cp)


# ── Run ───────────────────────────────────────────────────────────────────────
main()
