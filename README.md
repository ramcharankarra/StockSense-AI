# 📈 StockSense AI
### *Advanced AI-Powered FinTech Analytics Platform*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Database](https://img.shields.io/badge/SQLite-3.0-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org/)
[![NLP Ensemble](https://img.shields.io/badge/NLTK--VADER--TextBlob-Ensemble-blue?style=flat-square)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 🎯 Platform Overview

**StockSense AI** is an advanced AI-powered FinTech analytics platform that combines machine learning, quantitative risk analysis, NLP sentiment ensembles, and event-driven backtesting into a single Bloomberg-inspired dashboard. 

The application utilizes dynamic data-fetching services, thread-safe database caching pipelines, and modern visual controls, making it an excellent demonstration of full-stack financial engineering.

---

## 💡 Skills Demonstrated

*   **Machine Learning Engineering:** Designing time-series forecasting datasets using supervised tree algorithms (Random Forests, XGBoost) and recurrent deep learning networks (LSTM, GRU).
*   **Quantitative Risk Analytics:** Implementing asset risk metrics (annualized returns, Sharpe, Sortino, historical drawdown series, Alpha, and Beta sensitivity) and estimating capital losses via **Value-at-Risk (VaR 95%)** and **Conditional VaR (CVaR)**.
*   **Ensemble Natural Language Processing:** Mining financial media feeds using a weighted ensemble combining **NLTK VADER** and **TextBlob** sentiment scoring models.
*   **Financial Data Engineering:** Designing thread-safe database persistence caches (SQLite/PostgreSQL mappings), multi-threaded data ingestion feeds, and event-driven backtesting simulators.
*   **Institutional Product Design:** Building responsive, high-density light-mode user interfaces with customized styling sheets (`style.css`), rounded KPI elements, drop shadows, and interactive Plotly figures.

---

## ✨ Project Highlights

*   **10+ Analytical Modules:** Covers authentic auth walls, portfolio CRUD trackers, technical indicator tabs, ensemble NLP gauges, multi-model forecast labs, parallel screeners, and backtesting.
*   **5 Machine Learning Architectures:** Linear Regression, Random Forest, XGBoost, LSTM, and GRU networks.
*   **15+ Technical & Quantitative Indicators:** RSI, MACD, Bollinger Bands, Moving Averages (SMA/EMA), ATR, Sharpe, Sortino, VaR 95%, CVaR, Alpha, and Beta.
*   **Dynamic Data Feeds:** Real-time indices quotes, watchlist movers, and financial news fetched dynamically from Yahoo Finance via `yfinance`.
*   **High-Density Recruiter Layout:** Reorganized landing dashboard showing Portfolio simulated balances, model predictions, risk gauges, sentiment indices, movers, and news feeds on the very first screen.

---

## 🏗️ Core Architecture & Data Flow

To keep the codebase modular, StockSense AI decouples visual elements from computations. Detailed specification logs have been compiled inside the `docs/` folder:

*   **[Technical Architecture & Schema](file:///Users/charan/.gemini/antigravity/scratch/stocksense-ai/docs/architecture.md):** Details relational SQLite database schema mappings, cache TTL mechanisms, and the multi-threaded Market Scanner worker pipeline.
*   **[Codebase Modules Spec](file:///Users/charan/.gemini/antigravity/scratch/stocksense-ai/docs/modules.md):** Summarizes the responsibilities and interactions of all 13 structural layers.
*   **[Quantitative Risk Specifications](file:///Users/charan/.gemini/antigravity/scratch/stocksense-ai/docs/risk_engine.md):** Mathematical formulas and explanations for Sharpe, Sortino, Max Drawdown, VaR, CVaR, Alpha, and Beta calculations.
*   **[ML Predictive Framework](file:///Users/charan/.gemini/antigravity/scratch/stocksense-ai/docs/ml_pipeline.md):** Outlines engineered features, forecast models structure, regression errors evaluation, and SHAP explainability.

```
                    +-----------------------------+
                    |        yfinance API         |
                    +-----------------------------+
                                   | Ingestion
                                   v
+------------------+     +-------------------+     +--------------------+
|  analysis/       | <-> | database/         | <-> | ml/                |
|  - Technicals    |     | - schema.sql      |     | - Feature Eng.     |
|  - Risk Math     |     | - Caching Tables  |     | - Train/Predict    |
|  - Sentiment NLP |     +-------------------+     | - SHAP Attribution |
+------------------+                               +--------------------+
         |                                                   |
         +------------------------+--------------------------+
                                  |
                                  v
                    +-----------------------------+
                    | pages/   (Streamlit UI)     |
                    | assets/  (SaaS style.css)   |
                    +-----------------------------+
```

---

## 🖼️ User Interface Screenshots

*Visual dashboard captures showcasing the platform's high-density Bloomberg-inspired SaaS panels (stored under the `/screenshots` directory):*

*   **Home Dashboard Overview:** Portfolio Value metrics, next-day consensus forecasts, volatility risk indicators, and real-time market movers.
*   **Advanced Stock Charting:** Premium interactive candlestick charts, moving average crossovers, and volume histograms.
*   **Model Comparison Center:** Model evaluation matrices ranking XGBoost, LSTMs, and Random Forests across RMSE, MAE, and R² scores.
*   **Downside Risk Analytics:** Returns distribution charts with normal bell overlays, historical area drawdowns, and Expected Shortfall indicators.

---

## 🛠️ Step-by-Step Installation & Usage

### 1. Prerequisites
Ensure you have **Python 3.10** or **Python 3.11** installed. 

> [!NOTE]
> macOS users require the OpenMP runtime for `XGBoost`. Install it using Homebrew before compiling packages:
> ```bash
> brew install libomp
> ```

### 2. Standard Setup
```bash
# 1. Clone the repository
git clone https://github.com/ramcharankarra/StockSense-AI.git
cd StockSense-AI

# 2. Initialize a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# On Windows: .venv\Scripts\activate

# 3. Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Automated One-Shot Bootstrapping
StockSense AI includes a setup utility to prepare your system:
```bash
python setup.py
```
*This script copies the `.env.template` to `.env`, initializes the SQLite database tables, and securely downloads unverified NLTK sentiment dictionaries.*

### 4. Running the Platform
Launch the Streamlit dev server:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Navigate to the **Create Account** tab to register and explore the live workspace!

---

## 🛡️ Security & Safeguards

*   **Credential Hiding:** All personal API keys, database settings, and SMTP configurations are isolated strictly inside `.env` (fully excluded in `.gitignore`).
*   **Bcrypt Protections:** User passwords are encrypted server-side using Bcrypt (12 work factors).
*   **SQL Injection Shielding:** SQLite transactions utilize parameterized query parameters to prevent database manipulation exploits.

---

## 🚀 Future Roadmap

*   **Production PostgreSQL:** Transition caching and holdings storage to an external production PostgreSQL container.
*   **Brokerage Integrations:** Integrate paper-trading APIs (such as Alpaca or Interactive Brokers) to execute virtual trades.
*   **FinBERT Sentiment NLP:** Upgrade the TextBlob lexical VADER score to a fine-tuned financial BERT transformer network to improve NLP signal accuracies.

---

## 👨‍💻 Developed By

*   **GitHub Repository:** [https://github.com/ramcharankarra/StockSense-AI](https://github.com/ramcharankarra/StockSense-AI)
*   **Core Stack:** Python, Streamlit, yfinance, Pandas, NumPy, scikit-learn, XGBoost, TensorFlow, NLTK, VADER, TextBlob, SHAP, Plotly, SQLite, bcrypt.
*   **License:** MIT License (see [LICENSE](LICENSE)).
