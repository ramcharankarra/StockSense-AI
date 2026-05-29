"""
Explainability (SHAP)
======================
Generates SHAP feature importance values for trained tree/linear models.
Supports Random Forest, XGBoost, and Linear Regression.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ml.feature_engineering import build_features
from ml.model_trainer import load_model, _model_path
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_shap(
    df: pd.DataFrame,
    symbol: str,
    model_name: str,   # 'random_forest' | 'xgboost' | 'linear'
    horizon: int = 1,
    sentiment: float = 0.0,
    n_samples: int = 100,
) -> Optional[tuple[np.ndarray, list[str]]]:
    """
    Compute SHAP values for the given model.

    Returns (shap_values: np.ndarray, feature_names: list[str]) or None.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping explainability.")
        return None

    path = _model_path(symbol, model_name, horizon)
    bundle = load_model(path)
    if bundle is None:
        return None

    model, scaler, feature_cols = bundle
    X, _ = build_features(df, target_horizon=horizon, include_sentiment=sentiment)

    if feature_cols:
        X = X[[c for c in feature_cols if c in X.columns]]

    # Use a background sample for Tree/Linear explainers
    bg = X.sample(min(n_samples, len(X)), random_state=42)

    if scaler is not None:
        bg_t = pd.DataFrame(scaler.transform(bg), columns=bg.columns)
    else:
        bg_t = bg

    try:
        if model_name in ("xgboost", "random_forest"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, bg_t, feature_perturbation="interventional")

        shap_vals = explainer.shap_values(bg_t)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        logger.info("SHAP computed for %s/%s", symbol, model_name)
        return shap_vals, list(bg.columns)

    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return None


def shap_summary_df(shap_vals: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of mean absolute SHAP values ranked by importance."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def get_prediction_explanation(
    shap_vals: np.ndarray,
    feature_names: list[str],
    sample_idx: int = -1,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-N features driving a single prediction."""
    row = shap_vals[sample_idx]
    df = pd.DataFrame({"feature": feature_names, "shap_value": row})
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n)
    df["direction"] = df["shap_value"].apply(lambda v: "↑ Bullish" if v > 0 else "↓ Bearish")
    return df.drop(columns=["abs_shap"])
