-- ============================================================
-- StockSense AI — Database Schema
-- SQLite-compatible; PostgreSQL migration: replace
--   TEXT → VARCHAR(n), INTEGER PRIMARY KEY → SERIAL PRIMARY KEY
-- ============================================================

PRAGMA foreign_keys = ON;

-- ── Users & Auth ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    full_name       TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    last_login      TEXT,
    is_active       INTEGER DEFAULT 1,   -- 0 = disabled
    role            TEXT DEFAULT 'user'  -- 'user' | 'admin'
);

CREATE TABLE IF NOT EXISTS user_settings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    default_ticker      TEXT DEFAULT 'AAPL',
    default_period      TEXT DEFAULT '1y',
    risk_free_rate      REAL DEFAULT 0.05,
    email_alerts        INTEGER DEFAULT 0,
    watchlist           TEXT DEFAULT '[]',   -- JSON list
    theme               TEXT DEFAULT 'light',
    updated_at          TEXT DEFAULT (datetime('now'))
);

-- ── Market Data ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS stocks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT UNIQUE NOT NULL,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    market_cap  REAL,
    currency    TEXT DEFAULT 'USD',
    exchange    TEXT,
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id        INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date            TEXT NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    adj_close       REAL,
    volume          INTEGER,
    dividends       REAL DEFAULT 0,
    stock_splits    REAL DEFAULT 0,
    UNIQUE(stock_id, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_stock_date ON stock_prices(stock_id, date);

-- ── Technical Indicators ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS indicators (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id    INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date        TEXT NOT NULL,
    rsi         REAL,
    macd        REAL,
    macd_signal REAL,
    macd_hist   REAL,
    bb_upper    REAL,
    bb_mid      REAL,
    bb_lower    REAL,
    sma_20      REAL,
    sma_50      REAL,
    sma_200     REAL,
    ema_12      REAL,
    ema_26      REAL,
    atr         REAL,
    vwap        REAL,
    obv         REAL,
    stoch_k     REAL,
    stoch_d     REAL,
    UNIQUE(stock_id, date)
);

-- ── ML Predictions ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id        INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    model_name      TEXT NOT NULL,
    horizon         TEXT NOT NULL,   -- '1d' | '5d' | '21d'
    predicted_price REAL NOT NULL,
    confidence      REAL,
    mae             REAL,
    rmse            REAL,
    mape            REAL,
    r2              REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ── Sentiment ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sentiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id        INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    source          TEXT,             -- 'newsapi' | 'rss' | 'mock'
    headline        TEXT,
    sentiment_score REAL,             -- -1 (neg) to +1 (pos)
    sentiment_label TEXT,             -- 'positive' | 'neutral' | 'negative'
    compound_vader  REAL,
    published_at    TEXT,
    fetched_at      TEXT DEFAULT (datetime('now'))
);

-- ── Portfolios ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS portfolios (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS holdings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    stock_id        INTEGER NOT NULL REFERENCES stocks(id),
    symbol          TEXT NOT NULL,
    shares          REAL NOT NULL,
    avg_cost        REAL NOT NULL,
    added_at        TEXT DEFAULT (datetime('now')),
    UNIQUE(portfolio_id, symbol)
);

-- ── Backtests ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    symbol              TEXT NOT NULL,
    strategy_name       TEXT NOT NULL,
    strategy_params     TEXT,   -- JSON
    start_date          TEXT,
    end_date            TEXT,
    initial_capital     REAL,
    final_value         REAL,
    total_return_pct    REAL,
    annualized_return   REAL,
    sharpe_ratio        REAL,
    max_drawdown_pct    REAL,
    win_rate            REAL,
    total_trades        INTEGER,
    results_json        TEXT,   -- full trade log JSON
    created_at          TEXT DEFAULT (datetime('now'))
);

-- ── Alerts ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol          TEXT NOT NULL,
    alert_type      TEXT NOT NULL,   -- 'price' | 'rsi' | 'volume'
    condition       TEXT NOT NULL,   -- 'above' | 'below'
    threshold       REAL NOT NULL,
    is_active       INTEGER DEFAULT 1,
    triggered       INTEGER DEFAULT 0,
    triggered_at    TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    message         TEXT
);
