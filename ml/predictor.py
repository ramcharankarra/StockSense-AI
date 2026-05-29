"""
Predictor (Inference Engine)
=============================
Loads a persisted model and generates price predictions.
Supports all model types: Linear, RF, XGBoost, LSTM, GRU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import MODELS_DIR, LSTM_LOOKBACK
from ml.feature_engineering import build_features, build_lstm_sequences
from ml.model_trainer import load_model, _model_path
from analysis.technical_indicators import compute_all
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Traditional model inference ───────────────────────────────────────────────

def predict_traditional(
    df: pd.DataFrame,
    symbol: str,
    model_name: str,
    horizon: int = 1,
    sentiment: float = 0.0,
) -> Optional[float]:
    """
    Load a traditional model (linear/RF/XGBoost) and predict next price.
    Returns the predicted price as a float.
    """
    safe_name = model_name.lower().replace(" ", "_")
    path = _model_path(symbol, safe_name, horizon)
    bundle = load_model(path)
    if bundle is None:
        logger.warning("No saved model at %s", path)
        return None

    model, scaler, feature_cols = bundle

    X, _ = build_features(df, target_horizon=horizon, include_sentiment=sentiment)
    X = X[feature_cols] if feature_cols else X
    last_row = X.iloc[[-1]]

    if scaler is not None:
        last_row = pd.DataFrame(scaler.transform(last_row), columns=last_row.columns)

    pred = float(model.predict(last_row)[0])
    return pred


# ── LSTM / GRU inference ──────────────────────────────────────────────────────

def predict_deep(
    df: pd.DataFrame,
    symbol: str,
    model_name: str,   # 'lstm' or 'gru'
    horizon: int = 1,
) -> Optional[float]:
    """Load LSTM/GRU and predict next price."""
    path = _model_path(symbol, model_name.lower(), horizon)
    bundle = load_model(path)
    if bundle is None:
        logger.warning("No saved deep model at %s", path)
        return None

    model, scaler, lookback, n_feats = bundle

    # Prepare last `lookback` rows
    df_feat = compute_all(df.copy()).dropna()
    feature_cols = ["close", "volume", "rsi", "macd"]
    feature_cols = [c for c in feature_cols if c in df_feat.columns][:n_feats]
    data = df_feat[feature_cols].iloc[-lookback:].values

    if len(data) < lookback:
        return None

    scaled = scaler.transform(
        np.pad(data, ((0, 0), (0, n_feats - data.shape[1])), mode="constant")
    )
    X = scaled[np.newaxis, :, :]   # (1, lookback, n_feats)
    pred_scaled = model.predict(X, verbose=0)[0, 0]

    # Inverse-transform
    dummy = np.zeros((1, n_feats))
    dummy[0, 0] = pred_scaled
    pred = float(scaler.inverse_transform(dummy)[0, 0])
    return pred


# ── Unified prediction ────────────────────────────────────────────────────────

def predict_all(
    df: pd.DataFrame,
    symbol: str,
    horizon: int = 1,
    sentiment: float = 0.0,
) -> dict[str, Optional[float]]:
    """Run inference for all available persisted models."""
    current_price = float(df["close"].iloc[-1])
    results = {"current_price": current_price}

    trad_models = {
        "Linear Regression": "linear",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    for display_name, file_name in trad_models.items():
        pred = predict_traditional(df, symbol, file_name, horizon, sentiment)
        results[display_name] = pred

    for dl_name in ["lstm", "gru"]:
        try:
            pred = predict_deep(df, symbol, dl_name, horizon)
            results[dl_name.upper()] = pred
        except Exception as exc:
            logger.debug("Deep inference skipped (%s): %s", dl_name, exc)
            results[dl_name.upper()] = None

    return results
