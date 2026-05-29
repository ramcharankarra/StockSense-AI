"""
Portfolio Service
=================
Full portfolio CRUD, analytics, and reporting.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from database.db_manager import execute_query, execute_write, get_stock_id, upsert_stock
from data_ingestion.market_data_service import fetch_ohlcv, get_company_info
from analysis.risk_analytics import (
    daily_returns, annualized_return, volatility,
    sharpe_ratio, max_drawdown, risk_score,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_portfolio(user_id: int, name: str, description: str = "") -> int:
    """Create a new portfolio. Returns portfolio id."""
    pid = execute_write(
        "INSERT INTO portfolios (user_id, name, description) VALUES (?,?,?)",
        (user_id, name, description),
    )
    logger.info("Portfolio created: %s (id=%s, user=%s)", name, pid, user_id)
    return pid


def get_portfolios(user_id: int) -> list[dict]:
    return execute_query("SELECT * FROM portfolios WHERE user_id=? ORDER BY created_at DESC", (user_id,))


def get_portfolio(portfolio_id: int, user_id: int) -> Optional[dict]:
    rows = execute_query(
        "SELECT * FROM portfolios WHERE id=? AND user_id=?", (portfolio_id, user_id)
    )
    return rows[0] if rows else None


def delete_portfolio(portfolio_id: int, user_id: int) -> bool:
    n = execute_write(
        "DELETE FROM portfolios WHERE id=? AND user_id=?", (portfolio_id, user_id)
    )
    return bool(n)


# ── Holdings ──────────────────────────────────────────────────────────────────

def add_holding(portfolio_id: int, symbol: str, shares: float, avg_cost: float) -> tuple[bool, str]:
    symbol = symbol.upper()
    try:
        info = get_company_info(symbol) or {}
        stock_id = upsert_stock(symbol, info)
        existing = execute_query(
            "SELECT id, shares, avg_cost FROM holdings WHERE portfolio_id=? AND symbol=?",
            (portfolio_id, symbol),
        )
        if existing:
            # Average down / up cost basis
            old = existing[0]
            new_shares = old["shares"] + shares
            new_cost = (old["shares"] * old["avg_cost"] + shares * avg_cost) / new_shares
            execute_write(
                "UPDATE holdings SET shares=?, avg_cost=? WHERE id=?",
                (new_shares, new_cost, old["id"]),
            )
        else:
            execute_write(
                "INSERT INTO holdings (portfolio_id, stock_id, symbol, shares, avg_cost) VALUES (?,?,?,?,?)",
                (portfolio_id, stock_id, symbol, shares, avg_cost),
            )
        return True, f"{symbol} added to portfolio."
    except Exception as exc:
        logger.error("Add holding failed: %s", exc)
        return False, str(exc)


def remove_holding(portfolio_id: int, symbol: str) -> bool:
    n = execute_write(
        "DELETE FROM holdings WHERE portfolio_id=? AND symbol=?", (portfolio_id, symbol.upper())
    )
    return bool(n)


def get_holdings(portfolio_id: int) -> list[dict]:
    return execute_query(
        "SELECT * FROM holdings WHERE portfolio_id=? ORDER BY symbol", (portfolio_id,)
    )


# ── Analytics ─────────────────────────────────────────────────────────────────

def portfolio_analytics(portfolio_id: int, period: str = "1y") -> dict:
    """
    Compute portfolio-level analytics.

    Returns a dict with:
        holdings_detail, total_value, total_cost, total_pnl, total_pnl_pct,
        portfolio_return, volatility, sharpe_ratio, max_drawdown,
        allocation (pie data), risk_score, risk_category.
    """
    holdings = get_holdings(portfolio_id)
    if not holdings:
        return {"error": "Portfolio is empty."}

    symbols = [h["symbol"] for h in holdings]
    price_data: dict[str, pd.Series] = {}
    current_prices: dict[str, float] = {}

    for sym in symbols:
        df = fetch_ohlcv(sym, period=period)
        if not df.empty:
            price_data[sym] = df["close"]
            current_prices[sym] = float(df["close"].iloc[-1])

    # Build enriched holdings
    details = []
    total_cost = 0.0
    total_value = 0.0

    for h in holdings:
        sym = h["symbol"]
        cur = current_prices.get(sym, h["avg_cost"])
        cost = h["shares"] * h["avg_cost"]
        value = h["shares"] * cur
        pnl = value - cost
        pnl_pct = pnl / cost if cost else 0.0
        total_cost += cost
        total_value += value
        details.append({
            **h,
            "current_price": cur,
            "market_value": value,
            "cost_basis": cost,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = total_pnl / total_cost if total_cost else 0.0

    # Allocation
    allocation = [
        {"symbol": d["symbol"], "value": d["market_value"], "pct": d["market_value"] / total_value * 100}
        for d in details
        if total_value > 0
    ]

    # Portfolio weighted returns series
    weights = {h["symbol"]: current_prices.get(h["symbol"], 0) * h["shares"] / total_value
               for h in holdings if total_value > 0}

    # Build equal-length returns matrix
    if price_data:
        returns_df = pd.DataFrame({s: price_data[s].pct_change() for s in price_data}).dropna()
        port_returns = sum(returns_df[s] * weights.get(s, 0) for s in returns_df.columns if s in weights)
        port_prices = (1 + port_returns).cumprod()

        ann_ret = annualized_return(port_prices)
        vol = volatility(port_prices)
        sr = sharpe_ratio(port_prices)
        mdd = max_drawdown(port_prices)
        score, cat = risk_score(port_prices)
    else:
        ann_ret = vol = sr = mdd = 0.0
        score, cat = 50.0, "Moderate"
        port_returns = pd.Series(dtype=float)

    return {
        "holdings_detail": details,
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "portfolio_return": ann_ret,
        "volatility": vol,
        "sharpe_ratio": sr,
        "max_drawdown": mdd,
        "allocation": allocation,
        "risk_score": score,
        "risk_category": cat,
        "daily_returns": port_returns,
    }
