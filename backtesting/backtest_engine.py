"""
Backtesting Engine
==================
Event-driven backtester supporting indicator-based strategies.
Strategies defined as callable rules on indicator DataFrames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from analysis.technical_indicators import compute_all
from analysis.risk_analytics import (
    daily_returns, annualized_return, volatility,
    sharpe_ratio, max_drawdown,
)
from config import BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION
from database.db_manager import execute_write
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date: str
    action: str          # 'BUY' | 'SELL'
    price: float
    shares: float
    commission: float
    portfolio_value: float


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)


# ── Built-in strategies ───────────────────────────────────────────────────────

def rsi_strategy(df: pd.DataFrame, buy_threshold: float = 30, sell_threshold: float = 70) -> pd.Series:
    """Buy when RSI < buy_threshold, sell when RSI > sell_threshold."""
    signals = pd.Series("HOLD", index=df.index)
    signals[df["rsi"] < buy_threshold] = "BUY"
    signals[df["rsi"] > sell_threshold] = "SELL"
    return signals


def macd_crossover_strategy(df: pd.DataFrame) -> pd.Series:
    """Buy on MACD bullish crossover, sell on bearish crossover."""
    signals = pd.Series("HOLD", index=df.index)
    prev_macd = df["macd"].shift(1)
    prev_sig = df["macd_signal"].shift(1)
    signals[(df["macd"] > df["macd_signal"]) & (prev_macd <= prev_sig)] = "BUY"
    signals[(df["macd"] < df["macd_signal"]) & (prev_macd >= prev_sig)] = "SELL"
    return signals


def sma_crossover_strategy(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Golden cross / death cross strategy."""
    signals = pd.Series("HOLD", index=df.index)
    fast_col = f"sma_{fast}" if f"sma_{fast}" in df.columns else "sma_20"
    slow_col = f"sma_{slow}" if f"sma_{slow}" in df.columns else "sma_50"
    prev_fast = df[fast_col].shift(1)
    prev_slow = df[slow_col].shift(1)
    signals[(df[fast_col] > df[slow_col]) & (prev_fast <= prev_slow)] = "BUY"
    signals[(df[fast_col] < df[slow_col]) & (prev_fast >= prev_slow)] = "SELL"
    return signals


def bollinger_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """Buy at lower band, sell at upper band."""
    signals = pd.Series("HOLD", index=df.index)
    signals[df["close"] < df["bb_lower"]] = "BUY"
    signals[df["close"] > df["bb_upper"]] = "SELL"
    return signals


STRATEGIES = {
    "RSI Oversold/Overbought": rsi_strategy,
    "MACD Crossover": macd_crossover_strategy,
    "SMA Golden/Death Cross": sma_crossover_strategy,
    "Bollinger Mean Reversion": bollinger_mean_reversion,
}


# ── Core backtesting loop ─────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy_name: str = "RSI Oversold/Overbought",
    strategy_params: Optional[dict] = None,
    initial_capital: float = BACKTEST_INITIAL_CAPITAL,
    commission_rate: float = BACKTEST_COMMISSION,
    user_id: Optional[int] = None,
) -> BacktestResult:
    """
    Run a full backtest simulation.

    Parameters
    ----------
    df           : OHLCV DataFrame
    symbol       : ticker symbol
    strategy_name: name in STRATEGIES dict
    strategy_params: kwargs forwarded to strategy function
    """
    df = compute_all(df.copy()).dropna()
    strategy_fn = STRATEGIES.get(strategy_name, rsi_strategy)
    params = strategy_params or {}

    try:
        signals = strategy_fn(df, **params)
    except TypeError:
        signals = strategy_fn(df)

    cash = initial_capital
    shares_held = 0.0
    trades: list[Trade] = []
    equity_curve: list[float] = []
    dates: list[str] = []

    for date, row in df.iterrows():
        signal = signals.loc[date]
        price = row["close"]
        portfolio_val = cash + shares_held * price

        if signal == "BUY" and cash > price:
            shares_to_buy = (cash * 0.95) / price   # invest 95 % of cash
            commission = shares_to_buy * price * commission_rate
            if cash >= shares_to_buy * price + commission:
                shares_held += shares_to_buy
                cash -= shares_to_buy * price + commission
                trades.append(Trade(
                    date=str(date.date()), action="BUY", price=price,
                    shares=shares_to_buy, commission=commission,
                    portfolio_value=cash + shares_held * price,
                ))

        elif signal == "SELL" and shares_held > 0:
            commission = shares_held * price * commission_rate
            cash += shares_held * price - commission
            trades.append(Trade(
                date=str(date.date()), action="SELL", price=price,
                shares=shares_held, commission=commission,
                portfolio_value=cash,
            ))
            shares_held = 0.0

        portfolio_val = cash + shares_held * price
        equity_curve.append(portfolio_val)
        dates.append(str(date.date()))

    # Final liquidation
    final_price = float(df["close"].iloc[-1])
    final_value = cash + shares_held * final_price

    # Metrics
    equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
    total_return_pct = (final_value - initial_capital) / initial_capital * 100
    ann_ret = annualized_return(equity_series) * 100
    sr = sharpe_ratio(equity_series)
    mdd = max_drawdown(equity_series) * 100

    # Win rate
    sell_trades = [t for t in trades if t.action == "SELL"]
    buy_trades = [t for t in trades if t.action == "BUY"]
    wins = 0
    for i, sell in enumerate(sell_trades):
        if i < len(buy_trades) and sell.price > buy_trades[i].price:
            wins += 1
    win_rate = wins / len(sell_trades) * 100 if sell_trades else 0.0

    result = BacktestResult(
        symbol=symbol,
        strategy_name=strategy_name,
        start_date=dates[0] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_value=final_value,
        total_return_pct=total_return_pct,
        annualized_return=ann_ret,
        sharpe_ratio=sr,
        max_drawdown_pct=mdd,
        win_rate=win_rate,
        total_trades=len(trades),
        trades=trades,
        equity_curve=equity_curve,
        dates=dates,
    )

    _save_backtest(result, strategy_params, user_id)
    logger.info(
        "Backtest [%s/%s] | Return=%.2f%% | Sharpe=%.2f | MaxDD=%.2f%%",
        symbol, strategy_name, total_return_pct, sr, mdd,
    )
    return result


def _save_backtest(result: BacktestResult, params: Optional[dict], user_id: Optional[int]) -> None:
    try:
        execute_write(
            """INSERT INTO backtests
               (user_id, symbol, strategy_name, strategy_params,
                start_date, end_date, initial_capital, final_value,
                total_return_pct, annualized_return, sharpe_ratio,
                max_drawdown_pct, win_rate, total_trades, results_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                result.symbol,
                result.strategy_name,
                json.dumps(params or {}),
                result.start_date,
                result.end_date,
                result.initial_capital,
                result.final_value,
                result.total_return_pct,
                result.annualized_return,
                result.sharpe_ratio,
                result.max_drawdown_pct,
                result.win_rate,
                result.total_trades,
                json.dumps([asdict(t) for t in result.trades]),
            ),
        )
    except Exception as exc:
        logger.warning("Could not save backtest to DB: %s", exc)
