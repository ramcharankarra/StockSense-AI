"""
Database Manager
================
Abstraction layer over SQLite.
Swap DATABASE_URL for a postgres:// connection string in production to use
SQLAlchemy's PostgreSQL dialect — no other code changes required.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from config import DB_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


# ── Connection helper ─────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # dict-like rows
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and auto-commits / rolls back."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("DB error — rolled back: %s", exc)
        raise
    finally:
        conn.close()


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables from schema.sql if they do not exist."""
    sql = _SCHEMA_FILE.read_text()
    with get_db() as conn:
        conn.executescript(sql)
    logger.info("Database initialised at %s", DB_PATH)


# ── Generic CRUD helpers ──────────────────────────────────────────────────────

def execute_query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT query and return a list of row dicts."""
    with get_db() as conn:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def execute_write(sql: str, params: tuple = ()) -> int:
    """Run INSERT / UPDATE / DELETE and return lastrowid or rowcount."""
    with get_db() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid or cur.rowcount


def execute_many(sql: str, param_list: list[tuple]) -> None:
    """Batch INSERT / UPDATE."""
    with get_db() as conn:
        conn.executemany(sql, param_list)


# ── Stock helpers ─────────────────────────────────────────────────────────────

def upsert_stock(symbol: str, info: dict) -> int:
    """Insert or update a stock record; return its id."""
    existing = execute_query("SELECT id FROM stocks WHERE symbol = ?", (symbol,))
    if existing:
        execute_write(
            """UPDATE stocks SET name=?, sector=?, industry=?, market_cap=?,
               currency=?, exchange=?, updated_at=datetime('now') WHERE symbol=?""",
            (
                info.get("longName"),
                info.get("sector"),
                info.get("industry"),
                info.get("marketCap"),
                info.get("currency", "USD"),
                info.get("exchange"),
                symbol,
            ),
        )
        return existing[0]["id"]
    return execute_write(
        "INSERT INTO stocks (symbol, name, sector, industry, market_cap, currency, exchange) VALUES (?,?,?,?,?,?,?)",
        (
            symbol,
            info.get("longName"),
            info.get("sector"),
            info.get("industry"),
            info.get("marketCap"),
            info.get("currency", "USD"),
            info.get("exchange"),
        ),
    )


def get_stock_id(symbol: str) -> Optional[int]:
    rows = execute_query("SELECT id FROM stocks WHERE symbol = ?", (symbol,))
    return rows[0]["id"] if rows else None


# ── User settings helpers ─────────────────────────────────────────────────────

def get_user_settings(user_id: int) -> dict:
    rows = execute_query("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
    if not rows:
        return {}
    s = rows[0]
    s["watchlist"] = json.loads(s.get("watchlist") or "[]")
    return s


def save_user_settings(user_id: int, **kwargs) -> None:
    existing = execute_query("SELECT id FROM user_settings WHERE user_id = ?", (user_id,))
    if "watchlist" in kwargs and isinstance(kwargs["watchlist"], list):
        kwargs["watchlist"] = json.dumps(kwargs["watchlist"])
    if existing:
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [user_id]
        execute_write(f"UPDATE user_settings SET {sets}, updated_at=datetime('now') WHERE user_id=?", tuple(vals))
    else:
        kwargs["user_id"] = user_id
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" * len(kwargs))
        execute_write(f"INSERT INTO user_settings ({cols}) VALUES ({placeholders})", tuple(kwargs.values()))
