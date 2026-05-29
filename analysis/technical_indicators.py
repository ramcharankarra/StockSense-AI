"""
Technical Indicators Engine
============================
All indicators are implemented as pure pandas functions that accept
an OHLCV DataFrame and return a new column or a sub-DataFrame.

No TA-Lib dependency — fully implemented in NumPy/Pandas for portability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ──────────────────────────────────────────────────────────────────────

def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}, index=df.index)


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    col: str = "close",
) -> pd.DataFrame:
    """Upper, Middle (SMA), and Lower Bollinger Bands."""
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    return pd.DataFrame(
        {"bb_upper": mid + std_dev * std, "bb_mid": mid, "bb_lower": mid - std_dev * std},
        index=df.index,
    )


# ── Simple & Exponential Moving Averages ──────────────────────────────────────

def sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """Simple Moving Average."""
    return df[col].rolling(period).mean()


def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """Exponential Moving Average."""
    return df[col].ewm(span=period, adjust=False).mean()


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── VWAP ──────────────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (cumulative, resets at data start)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ── OBV ───────────────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ── Stochastic Oscillator ─────────────────────────────────────────────────────

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic %K and %D."""
    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)


# ── Composite: compute all indicators ─────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all technical indicators to a copy of the input DataFrame.
    Expects columns: open, high, low, close, volume.
    Returns expanded DataFrame.
    """
    result = df.copy()

    result["rsi"] = rsi(df)

    _macd = macd(df)
    result["macd"] = _macd["macd"]
    result["macd_signal"] = _macd["macd_signal"]
    result["macd_hist"] = _macd["macd_hist"]

    _bb = bollinger_bands(df)
    result["bb_upper"] = _bb["bb_upper"]
    result["bb_mid"] = _bb["bb_mid"]
    result["bb_lower"] = _bb["bb_lower"]

    result["sma_20"] = sma(df, 20)
    result["sma_50"] = sma(df, 50)
    result["sma_200"] = sma(df, 200)

    result["ema_12"] = ema(df, 12)
    result["ema_26"] = ema(df, 26)

    result["atr"] = atr(df)
    result["vwap"] = vwap(df)
    result["obv"] = obv(df)

    _stoch = stochastic(df)
    result["stoch_k"] = _stoch["stoch_k"]
    result["stoch_d"] = _stoch["stoch_d"]

    return result


# ── Helpers: signal interpretation ────────────────────────────────────────────

def interpret_rsi(rsi_val: float) -> str:
    if rsi_val < 30:
        return "Oversold"
    if rsi_val > 70:
        return "Overbought"
    return "Neutral"


def interpret_macd(macd_val: float, signal_val: float) -> str:
    if macd_val > signal_val:
        return "Bullish"
    return "Bearish"


def interpret_bb(price: float, upper: float, lower: float) -> str:
    if price > upper:
        return "Overbought"
    if price < lower:
        return "Oversold"
    return "Neutral"
