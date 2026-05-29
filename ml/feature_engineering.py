"""
Feature Engineering
====================
Automatically constructs ML-ready feature matrices from raw OHLCV data
and technical indicators. Returns (X, y) ready for model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.technical_indicators import compute_all
from utils.logger import get_logger

logger = get_logger(__name__)


def build_features(
    df: pd.DataFrame,
    target_horizon: int = 1,
    include_sentiment: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target series y.

    Parameters
    ----------
    df : OHLCV DataFrame (columns: open, high, low, close, volume)
    target_horizon : number of days ahead to predict (1, 5, or 21)
    include_sentiment : optional scalar sentiment score to append

    Returns
    -------
    X : pd.DataFrame of features (no NaN rows)
    y : pd.Series of future close prices
    """
    df = compute_all(df.copy())

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)

    # Price ratios
    df["hl_ratio"] = df["high"] / df["low"]
    df["oc_ratio"] = df["open"] / df["close"]

    # Volume features
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma5"].replace(0, np.nan)

    # Volatility features
    df["rolling_vol_5"] = df["close"].pct_change().rolling(5).std()
    df["rolling_vol_20"] = df["close"].pct_change().rolling(20).std()

    # Trend features
    df["above_sma50"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_sma200"] = (df["close"] > df["sma_200"]).astype(int)
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # Distance from Bollinger Bands
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # Price momentum
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # Target: future close price
    df["target"] = df["close"].shift(-target_horizon)

    # Sentiment (static scalar → broadcast)
    if include_sentiment is not None:
        df["sentiment"] = include_sentiment

    # Feature columns (exclude raw OHLCV and target)
    exclude = {"open", "high", "low", "close", "volume", "dividends",
               "stock_splits", "stock splits", "target"}
    feature_cols = [c for c in df.columns if c not in exclude]

    df = df.dropna(subset=feature_cols + ["target"])

    X = df[feature_cols].astype(float)
    y = df["target"].astype(float)

    logger.info("Feature matrix: %s rows × %s features", len(X), len(feature_cols))
    return X, y


def build_lstm_sequences(
    df: pd.DataFrame,
    lookback: int = 60,
    target_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Build 3-D (samples, timesteps, features) arrays for LSTM/GRU.
    Also returns the fitted MinMaxScaler for inverse-transform.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = compute_all(df.copy()).dropna()
    feature_cols = ["close", "volume", "rsi", "macd", "bb_pct"]
    # Keep only cols that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Bollinger % if not yet computed
    if "bb_pct" not in df.columns and "bb_upper" in df.columns:
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
            (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        )

    data = df[feature_cols].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled) - target_horizon):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i + target_horizon - 1, 0])  # close price index 0

    return np.array(X), np.array(y), scaler
