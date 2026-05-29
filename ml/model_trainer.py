"""
Model Trainer
==============
Trains all ML models (Linear Regression, Random Forest, XGBoost, LSTM, GRU)
on engineered features and persists trained models to disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from config import MODELS_DIR, ML_RANDOM_STATE, TRAIN_TEST_SPLIT, LSTM_LOOKBACK
from ml.feature_engineering import build_features, build_lstm_sequences
from ml.model_evaluator import evaluate_regression
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Persistence helpers ───────────────────────────────────────────────────────

def _model_path(symbol: str, model_name: str, horizon: int) -> Path:
    return MODELS_DIR / f"{symbol}_{model_name}_{horizon}d.pkl"


def save_model(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Traditional Models ────────────────────────────────────────────────────────

def train_linear_regression(
    df: pd.DataFrame, symbol: str, horizon: int = 1, sentiment: float = 0.0
) -> dict:
    X, y = build_features(df, target_horizon=horizon, include_sentiment=sentiment)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=TRAIN_TEST_SPLIT, shuffle=False
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)

    metrics = evaluate_regression(y_te.values, preds)
    save_model((model, scaler, list(X.columns)), _model_path(symbol, "linear", horizon))
    logger.info("[%s] Linear Regression trained | RMSE=%.4f", symbol, metrics["rmse"])
    return {**metrics, "model_name": "Linear Regression", "predictions": preds, "actuals": y_te.values}


def train_random_forest(
    df: pd.DataFrame, symbol: str, horizon: int = 1, sentiment: float = 0.0
) -> dict:
    X, y = build_features(df, target_horizon=horizon, include_sentiment=sentiment)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=TRAIN_TEST_SPLIT, shuffle=False
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=ML_RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    metrics = evaluate_regression(y_te.values, preds)
    save_model((model, None, list(X.columns)), _model_path(symbol, "random_forest", horizon))
    logger.info("[%s] Random Forest trained | RMSE=%.4f", symbol, metrics["rmse"])
    return {**metrics, "model_name": "Random Forest", "predictions": preds, "actuals": y_te.values,
            "feature_names": list(X.columns), "feature_importances": model.feature_importances_}


def train_xgboost(
    df: pd.DataFrame, symbol: str, horizon: int = 1, sentiment: float = 0.0
) -> dict:
    X, y = build_features(df, target_horizon=horizon, include_sentiment=sentiment)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=TRAIN_TEST_SPLIT, shuffle=False
    )
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=ML_RANDOM_STATE, verbosity=0, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    preds = model.predict(X_te)

    metrics = evaluate_regression(y_te.values, preds)
    save_model((model, None, list(X.columns)), _model_path(symbol, "xgboost", horizon))
    logger.info("[%s] XGBoost trained | RMSE=%.4f", symbol, metrics["rmse"])
    return {**metrics, "model_name": "XGBoost", "predictions": preds, "actuals": y_te.values,
            "feature_names": list(X.columns), "feature_importances": model.feature_importances_}


# ── Deep Learning Models ──────────────────────────────────────────────────────

def _build_lstm_model(input_shape: tuple, gru: bool = False):
    """Build LSTM or GRU Keras model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping

    RNN = GRU if gru else LSTM
    model = Sequential([
        RNN(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        RNN(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def _train_rnn(df, symbol, horizon, model_name="lstm"):
    X, y, scaler = build_lstm_sequences(df, lookback=LSTM_LOOKBACK, target_horizon=horizon)
    if len(X) < 100:
        raise ValueError(f"Insufficient data for {model_name}: {len(X)} samples")

    split = int(len(X) * TRAIN_TEST_SPLIT)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    model = _build_lstm_model(X_tr.shape[1:], gru=(model_name == "gru"))
    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=50, batch_size=32,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5),
        ],
        verbose=0,
    )

    preds_scaled = model.predict(X_te, verbose=0).flatten()
    # Inverse-transform: scaler transforms all features, close is index 0
    n_feats = X.shape[2]
    def inv_transform(arr):
        dummy = np.zeros((len(arr), n_feats))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    preds = inv_transform(preds_scaled)
    actuals = inv_transform(y_te)

    metrics = evaluate_regression(actuals, preds)
    path = _model_path(symbol, model_name, horizon)
    save_model((model, scaler, LSTM_LOOKBACK, n_feats), path)
    logger.info("[%s] %s trained | RMSE=%.4f", symbol, model_name.upper(), metrics["rmse"])
    return {**metrics, "model_name": model_name.upper(), "predictions": preds, "actuals": actuals}


def train_lstm(df, symbol, horizon=1):
    return _train_rnn(df, symbol, horizon, "lstm")


def train_gru(df, symbol, horizon=1):
    return _train_rnn(df, symbol, horizon, "gru")


# ── Train All ─────────────────────────────────────────────────────────────────

def train_all_models(
    df: pd.DataFrame,
    symbol: str,
    horizon: int = 1,
    sentiment: float = 0.0,
    include_deep: bool = True,
) -> dict[str, dict]:
    """
    Train all available models and return results dict.
    Returns {model_name: metrics_dict}.
    """
    results = {}

    for name, fn in [
        ("Linear Regression", lambda: train_linear_regression(df, symbol, horizon, sentiment)),
        ("Random Forest",     lambda: train_random_forest(df, symbol, horizon, sentiment)),
        ("XGBoost",           lambda: train_xgboost(df, symbol, horizon, sentiment)),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            logger.error("Training %s failed: %s", name, exc)

    if include_deep:
        for name, fn in [
            ("LSTM", lambda: train_lstm(df, symbol, horizon)),
            ("GRU",  lambda: train_gru(df, symbol, horizon)),
        ]:
            try:
                results[name] = fn()
            except Exception as exc:
                logger.warning("Deep learning training skipped (%s): %s", name, exc)

    return results
