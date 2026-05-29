"""
Market Data Service
====================
Wraps yfinance to fetch OHLCV data, company info, and financials.
Caches data to SQLite to minimise redundant API calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from config import DEFAULT_INTERVAL, DEFAULT_PERIOD
from database.db_manager import execute_query, execute_write, execute_many, upsert_stock, get_stock_id
from utils.logger import get_logger
from utils.helpers import clean_ohlcv, sanitize_ticker

logger = get_logger(__name__)


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.
    If start/end are provided, they take precedence over period.
    Returns a cleaned DataFrame with DatetimeIndex.
    """
    symbol = sanitize_ticker(symbol)
    logger.info("Fetching %s | period=%s interval=%s", symbol, period, interval)
    try:
        ticker = yf.Ticker(symbol)
        if start:
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
        else:
            df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        df = clean_ohlcv(df)
        _cache_to_db(symbol, df, ticker.info)
        return df
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", symbol, exc)
        return pd.DataFrame()


def fetch_multiple(symbols: list[str], period: str = DEFAULT_PERIOD) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols. Returns {symbol: DataFrame}."""
    result = {}
    for sym in symbols:
        df = fetch_ohlcv(sym, period=period)
        if not df.empty:
            result[sym] = df
    return result


def get_company_info(symbol: str) -> dict:
    """Return ticker info dict from yfinance."""
    symbol = sanitize_ticker(symbol)
    try:
        info = yf.Ticker(symbol).info
        return info or {}
    except Exception as exc:
        logger.error("Info fetch failed for %s: %s", symbol, exc)
        return {}


def get_financials(symbol: str) -> dict[str, pd.DataFrame]:
    """Return income_stmt, balance_sheet, cashflow DataFrames."""
    symbol = sanitize_ticker(symbol)
    try:
        t = yf.Ticker(symbol)
        return {
            "income_stmt": t.income_stmt,
            "balance_sheet": t.balance_sheet,
            "cashflow": t.cashflow,
        }
    except Exception as exc:
        logger.error("Financials fetch failed for %s: %s", symbol, exc)
        return {}


def get_options_chain(symbol: str) -> dict:
    """Return basic options data (calls + puts) for nearest expiry."""
    symbol = sanitize_ticker(symbol)
    try:
        t = yf.Ticker(symbol)
        expirations = t.options
        if not expirations:
            return {}
        chain = t.option_chain(expirations[0])
        return {"expiry": expirations[0], "calls": chain.calls, "puts": chain.puts}
    except Exception as exc:
        logger.error("Options fetch failed for %s: %s", symbol, exc)
        return {}


# ── DB caching ────────────────────────────────────────────────────────────────

def _cache_to_db(symbol: str, df: pd.DataFrame, info: dict) -> None:
    """Upsert stock info and price rows into the database."""
    try:
        stock_id = upsert_stock(symbol, info)
        rows = []
        for date, row in df.iterrows():
            rows.append((
                stock_id,
                str(date.date()),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("close"),   # adj_close ≈ close after auto_adjust
                int(row.get("volume") or 0),
                row.get("dividends", 0),
                row.get("stock splits", 0),
            ))
        execute_many(
            """INSERT OR REPLACE INTO stock_prices
               (stock_id, date, open, high, low, close, adj_close, volume, dividends, stock_splits)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
    except Exception as exc:
        logger.warning("DB cache failed for %s: %s", symbol, exc)


def load_from_db(symbol: str, days: int = 365) -> pd.DataFrame:
    """Load cached price data from SQLite."""
    stock_id = get_stock_id(symbol)
    if not stock_id:
        return pd.DataFrame()
    since = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = execute_query(
        """SELECT date, open, high, low, close, volume, dividends, stock_splits
           FROM stock_prices WHERE stock_id=? AND date>=? ORDER BY date""",
        (stock_id, since),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


# ── Market overview helpers ────────────────────────────────────────────────────

def get_index_data(tickers: dict[str, str]) -> dict[str, dict]:
    """Fetch last price and daily change for a dict of {name: ticker}."""
    result = {}
    for name, sym in tickers.items():
        try:
            info = yf.Ticker(sym).fast_info
            result[name] = {
                "price": getattr(info, "last_price", None),
                "change_pct": getattr(info, "three_month_return", None),
            }
        except Exception:
            result[name] = {"price": None, "change_pct": None}
    return result


def get_realtime_quote(symbol: str) -> dict:
    """Return current price, day high/low, volume for a symbol."""
    symbol = sanitize_ticker(symbol)
    try:
        fi = yf.Ticker(symbol).fast_info
        return {
            "price": getattr(fi, "last_price", None),
            "open": getattr(fi, "open", None),
            "day_high": getattr(fi, "day_high", None),
            "day_low": getattr(fi, "day_low", None),
            "volume": getattr(fi, "last_volume", None),
            "market_cap": getattr(fi, "market_cap", None),
        }
    except Exception as exc:
        logger.error("Quote fetch failed for %s: %s", symbol, exc)
        return {}
