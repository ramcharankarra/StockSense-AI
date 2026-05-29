"""
Market Scanner
==============
Scans a universe of stocks to identify top gainers, losers,
most active, most volatile, oversold, and overbought stocks.
"""

from __future__ import annotations

from typing import Optional
import concurrent.futures

import pandas as pd
import yfinance as yf

from config import DEFAULT_WATCHLIST
from analysis.technical_indicators import rsi, compute_all
from utils.logger import get_logger

logger = get_logger(__name__)

# Extended scanner universe
SCANNER_UNIVERSE = DEFAULT_WATCHLIST + [
    "AMD", "INTC", "NFLX", "DIS", "BA", "GS", "MS", "C",
    "BAC", "WFC", "T", "VZ", "KO", "PEP", "MCD", "SBUX",
    "CVX", "COP", "MRK", "PFE", "ABBV", "LLY", "TMO",
    "COST", "TGT", "HD", "LOW", "AMGN", "GILD",
]


def _fetch_summary(symbol: str) -> Optional[dict]:
    """Fetch 1-day OHLCV and compute key metrics for a single symbol."""
    try:
        df = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
        if df is None or len(df) < 2:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        close_today = float(df["close"].iloc[-1])
        close_prev = float(df["close"].iloc[-2])
        change_pct = (close_today - close_prev) / close_prev * 100
        volume = int(df["volume"].iloc[-1])
        vol_avg = int(df["volume"].mean())
        vol_ratio = volume / vol_avg if vol_avg else 1.0

        # RSI on 5-day data
        rsi_val = None
        if len(df) >= 5:
            try:
                rsi_series = rsi(df, period=min(14, len(df) - 1))
                rsi_val = float(rsi_series.iloc[-1])
            except Exception:
                pass

        return {
            "symbol": symbol,
            "price": round(close_today, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "vol_ratio": round(vol_ratio, 2),
            "rsi": round(rsi_val, 1) if rsi_val else None,
        }
    except Exception as exc:
        logger.debug("Scanner skipped %s: %s", symbol, exc)
        return None


def scan_market(universe: Optional[list[str]] = None, max_workers: int = 10) -> pd.DataFrame:
    """
    Parallel scan of all symbols. Returns a DataFrame sorted by change_pct desc.
    """
    symbols = universe or SCANNER_UNIVERSE
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_summary, sym): sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures):
            data = fut.result()
            if data:
                results.append(data)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("change_pct", ascending=False)
    logger.info("Market scan complete: %d stocks", len(df))
    return df


def top_gainers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "change_pct")


def top_losers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nsmallest(n, "change_pct")


def most_active(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "volume")


def most_volatile(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "vol_ratio")


def oversold(df: pd.DataFrame, threshold: float = 30) -> pd.DataFrame:
    return df[df["rsi"].notna() & (df["rsi"] < threshold)].sort_values("rsi")


def overbought(df: pd.DataFrame, threshold: float = 70) -> pd.DataFrame:
    return df[df["rsi"].notna() & (df["rsi"] > threshold)].sort_values("rsi", ascending=False)
