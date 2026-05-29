# StockSense AI — Modules Technical Specification

StockSense AI is built as a highly decoupled, modular application consisting of 13 separate layers. Each module has strict design boundaries, standard interfaces, and clear responsibilities.

---

## 📋 Module Technical Specifications

### 1. User Authentication (`auth/`)
*   **Purpose:** Manages user state, password protections, and login walls.
*   **Capabilities:** Uses **bcrypt** with 12 computational cost factors to hash passwords securely. Handles Streamlit session persistence and redirects to the landing area upon successful login.

### 2. Market Data Engine (`data_ingestion/market_data_service.py`)
*   **Purpose:** Resilient wrapper around the `yfinance` library.
*   **Capabilities:** Retrieves OHLCV historical tables, institutional profiles, metadata ratios, and splits. Includes automatic SQLite caching with tunable Time-To-Live (TTL) thresholds.

### 3. Financial News Ingestion (`data_ingestion/news_service.py`)
*   **Purpose:** Gathers market feeds.
*   **Capabilities:** Queries the `yfinance` RSS news feeds and supports external API integrations (such as **NewsAPI.org**) with automatic mock fallbacks when API keys are absent.

### 4. Technical Indicators Engine (`analysis/technical_indicators.py`)
*   **Purpose:** Vectorized pandas-based quantitative technical calculations.
*   **Capabilities:** Computes standard mathematical formulations:
    *   **Simple & Exponential Moving Averages (SMA/EMA):** Overlaid on price channels.
    *   **Relative Strength Index (RSI):** Calculated over a standard 14-day window.
    *   **MACD:** Signal and histogram series.
    *   **Bollinger Bands:** Standard deviation margins.
    *   **Average True Range (ATR):** Measuring volatility.
    *   **VWAP & OBV:** Incorporating volume dimensions.

### 5. Machine Learning Pipeline (`ml/`)
*   **Purpose:** Formulates historical feature frames and trains forecasting models.
*   **Capabilities:**
    *   `feature_engineering.py`: Computes 30+ features (lags, volatility, momentum flags).
    *   `model_trainer.py`: Fits Linear Regression, Random Forests, XGBoost, LSTM, and GRU configurations.
    *   `predictor.py`: Generates next-day (1d), next-week (5d), and next-month (21d) predictions.

### 6. Quantitative Risk Engine (`analysis/risk_analytics.py`)
*   **Purpose:** Portfolio analytics and downside protection math.
*   **Capabilities:** Evaluates annualized return, volatility, Sharpe ratio, Sortino ratio, max drawdown, Beta, and Alpha vs. S&P 500. Computes advanced downside parameters: **Value-at-Risk (VaR 95%)** and **Conditional VaR (CVaR)**.

### 7. Sentiment Analytics Engine (`analysis/sentiment_engine.py`)
*   **Purpose:** Ensemble NLP scoring of financial texts.
*   **Capabilities:** Runs a hybrid ensemble scoring models combining **NLTK VADER** (60 % weight, tailored for market terminology) and **TextBlob** (40 % weight, measuring general polarity and subjectivity).

### 8. Consensus Signal Engine (`signals/signal_engine.py`)
*   **Purpose:** Multi-factor quantitative decision aggregator.
*   **Capabilities:** Aggregates findings from technical thresholds (RSI/MACD crossovers), ML directional predictions, risk metrics, and media sentiment. Emits unified signals: `Strong Buy`, `Buy`, `Hold`, `Sell`, or `Strong Sell` along with confidence ratings and qualitative rationales.

### 9. Portfolio Manager (`portfolio/`)
*   **Purpose:** Asset allocation tracking and P&L accounting.
*   **Capabilities:** Tracks holdings, calculates dynamic cost basis, handles transaction histories, and visualizes assets distribution and cost averages.

### 10. Strategy Backtesting Lab (`backtesting/`)
*   **Purpose:** Event-driven strategy performance simulator.
*   **Capabilities:** Simulates historical strategy returns (such as RSI mean reversion and SMA crossovers). Features transaction cost accounting, Buy & Hold benchmark comparisons, monthly returns heatmap charts, and equity curve visualizations.

### 11. Market Scanner screener (`scanner/`)
*   **Purpose:** Concurrent screener scans.
*   **Capabilities:** Implements `ThreadPoolExecutor` parallel worker threads to scan popular tickers, generating screener tables for top movers, oversold indicators, and active volumes.

### 12. Local Alert Service (`alerts/`)
*   **Purpose:** Custom parameter monitoring.
*   **Capabilities:** Monitors price, RSI, and volume thresholds. Connects to standard SMTP mail configurations to fire email notifications when criteria are triggered.

### 13. Streamlit Visual Layout (`pages/`, `app.py`, `assets/`)
*   **Purpose:** Modern Bloomberg Ice-White presentation layer.
*   **Capabilities:** Integrates custom stylesheets (`assets/style.css`), dynamic navigation layouts, responsive grids, and interactive, customized Plotly chart containers.
