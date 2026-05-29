# 📈 StockSense AI
### *Enterprise-Grade Stock Market Prediction & Quantitative Risk Analysis Platform*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Database](https://img.shields.io/badge/SQLite-3.0-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org/)
[![Ensemble NLP](https://img.shields.io/badge/NLTK--VADER--TextBlob-Ensemble-blue?style=flat-square)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 🎯 Platform Overview

**StockSense AI** is a production-grade, startup-quality financial analytics platform designed to provide retail investors and quantitative analysts with institutional-grade insights. By combining **Machine Learning models** (Regression, Tree Ensembles, and Recurrent Neural Networks), **VADER/TextBlob Sentiment ensembles**, and advanced **Risk Modeling** (Value-at-Risk, Conditional VaR), StockSense AI transforms raw market feeds into actionable, quantitative intelligence.

The application features a premium, **Bloomberg-inspired light-mode layout** designed to maximize data density and visual hierarchy. It is fully optimized for final year major project showcases, FinTech portfolio demonstrations, and Quant/ML engineering internship interviews.

---

## 🖼️ User Interface & Visuals
*(Click on the tabs below to preview the platform's high-density Bloomberg-inspired design panels)*

````carousel
```
================================================================================
                         [ HOME DASHBOARD PANEL ]
================================================================================
  S&P 500: $5,248.50 (+0.42%)  |  NASDAQ: $16,420.10 (+0.68%)  |  VIX: 13.42 (-2.1%)
--------------------------------------------------------------------------------
  [ Welcome back, Jane Doe! ] — Markets are live. Here is your daily snapshot.
--------------------------------------------------------------------------------
  30-Day Relative Performance (Plotly Chart)
  /\___/\___/\   S&P 500 (#1a56db)
  \___/\___/\_   NASDAQ (#059669)
--------------------------------------------------------------------------------
  [ Top Watchlist Movers ]                 [ Real-time Market News ]
  - AAPL: $189.50 (▲ 2.45%)                - FED signals rate pause ... (Reuters)
  - NVDA: $920.10 (▲ 4.12%)                - Tech shares rally on AI demand ... (BBG)
```
<!-- slide -->
```
================================================================================
                        [ AI PREDICTION LAB PANEL ]
================================================================================
  Selected Stock: AAPL  |  Horizon: 5 Days (Next Week)  |  Model: All Models
--------------------------------------------------------------------------------
  [ Best Performing Model: XGBoost ★ (RMSE: 1.4250 | R²: 0.9412) ]
--------------------------------------------------------------------------------
  [ Model Comparison Table ]
  Model               RMSE     MAE      MAPE     R²       Predicted    Change %
  * XGBoost ★         1.4250   1.0420   0.55%    0.9412   $194.20      ▲ +2.48%
  * Random Forest     1.8540   1.3500   0.72%    0.9102   $193.10      ▲ +1.90%
  * LSTM              2.1240   1.6210   0.88%    0.8874   $192.50      ▲ +1.58%
  * LinReg            2.4520   1.9540   1.05%    0.8520   $191.10      ▲ +0.84%
--------------------------------------------------------------------------------
  Actual vs Predicted Evaluation Plot  |  SHAP Feature Importance (Explainable AI)
```
<!-- slide -->
```
================================================================================
                        [ RISK ANALYTICS PANEL ]
================================================================================
  Selected Stock: AAPL  |  Benchmark: S&P 500  |  Risk-Free Rate: 5.0%
--------------------------------------------------------------------------------
  [ Risk Score Gauge ]                        [ Historical Drawdown Area Chart ]
   ______ 54 / 100 ______                     0% |-----------------------------
  [ Moderate Risk Category ]                     |   \__/\_   _/\_
  (Green <33 | Amber <66 | Red >66)          -8% |_________\_/____\___________/
--------------------------------------------------------------------------------
  - Annualised Return: +24.12%    - Sharpe Ratio: 1.68        - Beta: 1.15
  - Historical Volatility: 16.50% - Sortino Ratio: 2.14       - Alpha: +4.20%
  - Value at Risk (95%): -2.42%   - Conditional VaR: -3.85%   - Max Drawdown: -12.4%
```
````

---

## 🏗️ Technical Architecture & Directory Mapping

StockSense AI follows a highly decoupled, modular architectural layout where backend computations are entirely separated from the Streamlit UI components:

```
stocksense-ai/
├── app.py                      # Main entry point (auth wall, router, sidebar layout)
├── config.py                   # Centralized platform configs, watchlists, risk ratios
├── requirements.txt            # Pinned production-grade python dependencies
├── setup.py                    # One-shot environment bootstraper & NLTK downloader
├── .env.template               # Template config file for API keys and DB credentials
│
├── auth/                       # Module 1: Session Management & Authentication
│   ├── auth_service.py         # Bcrypt password hashing, registration, and user CRUD
│   └── session_manager.py      # Streamlit session wrappers & security walls
│
├── database/                   # Module 2: Data Persistence & Abstract Layer
│   ├── schema.sql              # Clean 10-table optimized SQLite database schema
│   └── db_manager.py           # Thread-safe connection context manager & helpers
│
├── data_ingestion/             # Module 3: Market Feed Ingestion
│   ├── market_data_service.py  # Resilient yfinance wrapper with local SQLite cache
│   └── news_service.py         # Financial news collector (Yahoo News + NewsAPI)
│
├── analysis/                   # Modules 4, 5, 6: Quantitative Engines
│   ├── technical_indicators.py # Vectorized pandas math (RSI, MACD, BB, SMA, EMA, VWAP)
│   ├── risk_analytics.py       # Portfolio & stock math (Sharpe, Sortino, VaR, CVaR)
│   └── sentiment_engine.py     # Ensemble VADER + TextBlob scoring NLP models
│
├── ml/                         # Module 7: Predictive Pipeline
│   ├── feature_engineering.py  # 30+ engineered metrics (lag, rolling vol, BB pct)
│   ├── model_trainer.py        # Estimator models (Linear, Forest, XGBoost, LSTM, GRU)
│   ├── model_evaluator.py      # Regression benchmarks (RMSE, MAE, MAPE, R²)
│   ├── predictor.py            # Multi-horizon inference coordinator (1d, 5d, 21d)
│   └── explainability.py       # SHAP feature importance plotters & fallback weights
│
├── signals/                    # Module 8: Consolidated Signal Engine
│   └── signal_engine.py        # Consensus quant vote engine (Signals + Rationale)
│
├── portfolio/                  # Module 9: Portfolio Tracking
│   └── portfolio_service.py    # Multi-asset holdings trackers, P&L, & allocations
│
├── backtesting/                # Module 10: Backtesting Lab
│   └── backtest_engine.py      # Event-driven historical simulation & performance heatmaps
│
├── scanner/                    # Module 11: Market Screener
│   └── market_scanner.py       # Multi-threaded parallel market scanner
│
├── pages/                      # Module 12: Streamlit Visual Pages
│   ├── 00_home.py              # Market dashboard, relative performance line plots
│   ├── 01_stock_analysis.py    # Candlestick overlays & indicator subplots
│   ├── 02_prediction_center.py # ML trainer, radar evaluation, actual vs forecast
│   ├── 03_risk_analytics.py    # Volatility gauges, returns distribution & drawdowns
│   ├── 04_sentiment_center.py  # News VADER indicator gauges & keyword bar charts
│   ├── 05_portfolio_manager.py # Holdings tracker, P&L columns, allocation pie charts
│   ├── 06_backtesting.py       # Backtest simulator, commissions, returns heatmaps
│   ├── 07_market_scanner.py    # Top Movers, Active, RSI Oversold/Overbought screeners
│   └── 08_settings.py          # Watchlists selectors, default period preferences
│
└── assets/                     # Custom Styling Stylesheets
    └── style.css               # Bloomberg White skin, button spacing overrides
```

---

## 📊 Complete Feature Walk-Through

### 1. Home Dashboard
*   **Real-time Index Tickers:** Instantly fetches the S&P 500 (`^GSPC`), NASDAQ (`^IXIC`), Dow Jones (`^DJI`), and VIX (`^VIX`) quotes.
*   **30-Day Relative Performance:** Visualizes market indexes relative to a 30-day baseline on a single normalized Plotly line chart.
*   **Watchlist Movers & News:** Highlights the top 5 movers from the user's watchlist alongside a dynamic real-time financial news stream.

### 2. Stock Analysis
*   **Premium Candlestick Charting:** Plotly-based candlestick charts with matching volume bar subplots.
*   **Dynamic Indicator Panels:** Interactive tabs for Overlay SMA/EMAs, Relative Strength Index (RSI), MACD (Signal + Histograms), and Bollinger Bands.
*   **Quant Trading Signal:** Computes a composite consensus buy/sell badge with real-time quantitative explanations.

### 3. AI Prediction Center
*   **Ensemble ML Training:** Train traditional models (Linear Regression, Random Forest, XGBoost) and deep recurrent networks (LSTM, GRU) on demand.
*   **Comprehensive Model Grid:** Ranks all models using standard regression metrics (RMSE, MAE, MAPE, R²).
*   **Explainable AI (SHAP):** Visualizes feature importance using SHAP values (or mathematical random forest weight fallbacks) so the user understands the predictive drivers.
*   **Prediction Horizons:** Compares forecasts across 1-Day, 5-Day (Next Week), and 21-Day (Next Month) horizons.

### 4. Risk Analytics
*   **Composite Risk Score:** A proprietary risk gauge (0 to 100) mapping stocks into Low, Moderate, or High risk categories.
*   **8 core Quant KPIs:** Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Beta, Alpha, and Value-at-Risk (VaR 95%).
*   **Quantitative Visualizations:** Includes cumulative returns vs. benchmark, historical drawdown area charts, daily returns distribution plots overlaid with standard normal bell curves, and rolling 30-day Sharpe charts.

### 5. Sentiment Center
*   **Hybrid Ensemble Sentiment:** Computes a consolidated score using VADER (lexicon-based) and TextBlob (subjectivity/polarity) weights.
*   **News Insights:** Displays news distribution donut charts, raw articles tables decorated with sentiment badges, and keyword frequency counts.
*   **Sentiment vs. Price Overlay:** Automatically maps the sentiment score trend over the historical price timeline to check media correlation.

### 6. Portfolio Manager
*   **Multi-Asset Portfolios:** Full support for creating, reading, updating, and deleting (CRUD) portfolios.
*   **Allocation Analytics:** Renders portfolio value distribution via Plotly pie charts and tracks cost-averaging.
*   **Metrics & Trackers:** Real-time portfolio volatility, Sharpe ratio, and individual holdings P&L color-coding.

### 7. Backtesting Lab
*   **Strategy Replay:** Simulate RSI Mean Reversion, MACD Crossover, SMA Double-Crossover, or Bollinger Band Breakout.
*   **Parameters & Commissions:** Adjust parameters, initial capital, and transaction commissions to model real-world slippage.
*   **Performance Metrics:** Evaluates equity growth curves, Buy & Hold comparisons, win rates, and drawdowns, accompanied by a Plotly monthly returns performance heatmap.

### 8. Market Scanner
*   **Parallel Scanning Screener:** Multi-threaded scan over the stock universe.
*   **Dynamic Screening Tabs:** Filters market assets into Top Gainers, Top Losers, Most Active, Most Volatile, RSI Oversold (<30), and RSI Overbought (>70) lists.
*   **Quick Analysis Link:** Click "Analyze" on any scanner row to automatically set the global ticker state and jump directly to technical or ML pages.

### 9. Settings & Profile
*   **Profile Management:** Update personal user details, change passwords, and track account creation times.
*   **Platform Preferences:** Modify your pre-selected default ticker, lookback periods, risk-free rates, and multiselect personal watchlist.
*   **System Telemetry:** Real-time database telemetry dashboards displaying database record counts and cache-clearing tools.

---

## 🛠️ Step-by-Step Onboarding & Installation

### 1. Prerequisites
Ensure you have **Python 3.10** or **Python 3.11** installed on your operating system. 

> [!NOTE]
> For macOS users, `XGBoost` requires the OpenMP runtime. Please install it using Homebrew before starting the virtual environment:
> ```bash
> brew install libomp
> ```

### 2. Setup the Repository
```bash
# 1. Clone the repository
git clone https://github.com/ramcharankarra/StockSense-AI.git
cd StockSense-AI

# 2. Initialize a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# On Windows: .venv\Scripts\activate

# 3. Install production-grade requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Automated One-Shot Bootstrapping
StockSense AI includes a dedicated `setup.py` onboarding script. It validates python packaging, creates your `.env` variables file from the template, initializes the thread-safe SQLite database schema, and installs the required unverified NLTK Lexicon directories automatically:

```bash
python setup.py
```

### 4. Running the Platform
Launch the Streamlit developer server:
```bash
streamlit run app.py
```
The server will boot headless and print the local address, typically **`http://localhost:8501`**. Open this URL in your web browser.

### 5. Create an Account & Sign In
1.  On the landing panel, navigate to the **Create Account** tab.
2.  Register a test username, email, and password.
3.  Upon submitting, the system will **automatically log you in**, set up your session cache, and redirect you to the main **Home Dashboard** workspace.

---

## 🗄️ Database Schema Blueprint

The database engine uses parameterized, thread-safe queries via a connection pool context manager. It is fully prepared for PostgreSQL by modifying the `DATABASE_URL` in `.env` without any Python code updates.

```
+------------------+          +--------------------+
|      users       | <------- |   user_settings    |
| (PK, hash_pass)  |          | (FK, watchlist, df)|
+------------------+          +--------------------+
         |
         | (1 : N)
         v
+------------------+          +--------------------+
|    portfolios    | <------- |      holdings      |
| (PK, user_id FK) |          | (FK, symbol, cost) |
+------------------+          +--------------------+

  ================= Runtime Cache Tables =================
  * stock_prices (PK: symbol + date, High, Low, Close, Volume)
  * indicators   (FK: symbol + date, RSI, MACD, Bollinger)
  * predictions  (FK: symbol, model_name, horizon, price, MAPE)
  * sentiments   (FK: symbol, overall_score, label, timestamp)
  * backtests    (FK: user_id, strategy, total_return, win_rate)
```

---

## 🛡️ Security & Best Practices

*   **Zero Exposed Secrets:** All credentials, SMTP details, and API keys are stored strictly in the `.env` file, which is fully ignored in `.gitignore`.
*   **Bcrypt Hashing:** User passwords are hashed securely using Bcrypt (12 work factors).
*   **Database Sanitization:** All database operations utilize SQL parameters to neutralize SQL Injection vulnerabilities.
*   **Server-Side Isolation:** Session configurations and credential states are isolated server-side inside Streamlit's `st.session_state` structure, shielding data from client side manipulation.

---

## 🚀 Future Enhancements

*   **PostgreSQL Scale:** Fully migrate the database backend to a production PostgreSQL container with index performance tuning.
*   **Live Brokerage Integration:** Add sandbox execution integrations using Alpaca API or Interactive Brokers for paper trading.
*   **Deep Learning Models Optimization:** Introduce hyperparameter tuning via Optuna for LSTM and GRU neural configurations.
*   **Advanced Sentiment Engine:** Upgrade from lexicon ensembles to a fine-tuned financial BERT transformer model (FinBERT) for news and earnings transcript analysis.

---

## 👨‍💻 Developed By

Developed with a commitment to clean code, professional architecture, and financial analytic rigor. 

*   **GitHub Repository:** [https://github.com/ramcharankarra/StockSense-AI](https://github.com/ramcharankarra/StockSense-AI)
*   **Technologies:** Python, Streamlit, yfinance, Pandas, NumPy, scikit-learn, XGBoost, TensorFlow, NLTK, VADER, TextBlob, SHAP, Plotly, SQLite, bcrypt.
