# StockSense AI — Machine Learning Pipeline Specification

StockSense AI implements a comprehensive, multi-model predictive framework (`ml/`) designed to engineer technical features, train supervised models, evaluate fitting accuracies, and visualize predictive weight parameters.

---

## 📈 Feature Engineering Pipeline

The feature engineering layer (`ml/feature_engineering.py`) automatically generates **over 30 analytical inputs** from raw historical OHLCV price series:

| Feature Category | Engineered Variables | Purpose |
|------------------|----------------------|---------|
| **Lags & Returns** | `Lag_1`, `Lag_2`, `Lag_3`, `Lag_5`, `Lag_10` | Captures immediate auto-regressive momentum patterns. |
| **Rolling Averages** | `SMA_10`, `SMA_20`, `SMA_50`, `SMA_100` | Smoothes short and medium term noise to isolate trend paths. |
| **Volatility Metrics** | `Vol_5`, `Vol_10`, `Vol_20` | Annualized rolling standard deviations capturing risk regimes. |
| **Technical Indicators** | `RSI`, `MACD`, `MACD_Sig`, `BB_Upper`, `BB_Lower` | Incorporates standard momentum and boundary thresholds into feature vectors. |
| **Relative Metrics** | `Price_to_SMA50`, `Price_to_SMA200` | Measures absolute stretch relative to major historical support levels. |
| **Range Spreads** | `High_Low_Spread`, `Close_Open_Spread` | Quantifies intraday volatility and directional commitment. |

---

## 🤖 Predictive Models Framework

StockSense AI orchestrates five separate architectures to forecast future prices across multiple horizons (1-Day, 5-Day, and 21-Day intervals):

### 1. Supervised Traditional Estimators
*   **Linear Regression:** Fits an ordinary least squares baseline to measure linear relationships.
*   **Random Forest Regressor:** An ensemble of decision trees that reduces variance through bagging and splits on random subsets of features.
*   **XGBoost (Extreme Gradient Boosting):** Fits sequential trees on negative gradient residuals, minimizing a customized squared error loss function with regularization to prevent overfitting.

### 2. Recurrent Deep Learning Architectures
If **TensorFlow** is present, StockSense AI compiles recurrent architectures designed for sequential time series forecasting:
*   **LSTM (Long Short-Term Memory):** Features input, forget, and output gating cells that regulate information flow, capturing long-term temporal dependencies.
*   **GRU (Gated Recurrent Unit):** Streamlines gating mechanisms by combining the cell state and hidden state, improving computational speed while retaining sequence properties.

---

## 📐 Evaluation & Accuracy Benchmarks

Models are ranked and selected based on regression performance metrics computed on an out-of-sample test set (using an 80/20 temporal train/test split to prevent lookahead bias):

### 1. Root Mean Squared Error (RMSE)
Penalizes larger outliers and errors heavily:
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 2. Mean Absolute Error (MAE)
Measures average absolute magnitude of deviations:
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 3. Mean Absolute Percentage Error (MAPE)
Quantifies error as a percentage of actual price values:
$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### 4. Coefficient of Determination ($R^2$)
Measures the proportion of variance in prices explained by the engineered features:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

---

## 🔍 Explainable AI (SHAP Framework)

To move beyond "black-box" predictions, the platform utilizes **SHAP (SHapley Additive exPlanations)** to compute game-theoretic Shapley attribution values for each engineered feature. 

When the `shap` package is present, the module generates local explanation values. When absent, the system falls back on computing normalized feature importances from Random Forest/XGBoost tree splits to ensure consistent explainability.
