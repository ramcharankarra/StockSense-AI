"""
Model Evaluator
================
Regression metrics: RMSE, MAE, MAPE, R².
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MAPE, R² for regression predictions.

    Returns dict with keys: rmse, mae, mape, r2.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE — avoid divide-by-zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("inf")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def rank_models(results: dict[str, dict]) -> list[tuple[str, dict]]:
    """
    Rank models by RMSE (ascending).
    Returns list of (model_name, metrics) tuples.
    """
    return sorted(results.items(), key=lambda kv: kv[1].get("rmse", float("inf")))


def best_model(results: dict[str, dict]) -> tuple[str, dict]:
    """Return (name, metrics) for the best-performing model."""
    ranked = rank_models(results)
    return ranked[0] if ranked else ("", {})
