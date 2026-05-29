"""
StockSense AI – Central Configuration
======================================
All environment variables, paths, and tunable knobs live here.
Sub-modules import from this file; nothing reads os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

# ── Application ─────────────────────────────────────────────────────────────
APP_NAME = "StockSense AI"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Stock Market Prediction & Risk Analysis Platform"

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/database/stocksense.db")
DB_PATH = BASE_DIR / "database" / "stocksense.db"

# ── Auth ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "stocksense-secret-key-change-in-production")
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))

# ── Data APIs ────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── yfinance defaults ────────────────────────────────────────────────────────
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
SUPPORTED_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
SUPPORTED_INTERVALS = ["1d", "1wk", "1mo"]

# ── ML ───────────────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_HORIZONS = {
    "Next Day": 1,
    "Next Week": 5,
    "Next Month": 21,
}

ML_RANDOM_STATE = 42
LSTM_LOOKBACK = 60          # days of history fed to LSTM/GRU
TRAIN_TEST_SPLIT = 0.8

# ── Risk ─────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))   # 5 % annual
TRADING_DAYS_PER_YEAR = 252
VAR_CONFIDENCE = 0.95       # 95 % VaR

# ── Sentiment ─────────────────────────────────────────────────────────────────
MAX_NEWS_ARTICLES = 50
NEWS_LOOKBACK_DAYS = 7

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "stocksense.log"

# ── Streamlit page config ────────────────────────────────────────────────────
PAGE_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ── Popular tickers for scanner / default search ─────────────────────────────
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "JNJ",
    "WMT", "BRK-B", "UNH", "XOM", "PG",
]

MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
}

# ── Backtesting ──────────────────────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL = 100_000.0
BACKTEST_COMMISSION = 0.001    # 0.1 % per trade
