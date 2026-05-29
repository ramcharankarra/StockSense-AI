"""
Alert Service
=============
Price, RSI, and Volume alerts. Evaluates active alerts against current data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import yfinance as yf
import smtplib
from email.mime.text import MIMEText

from database.db_manager import execute_query, execute_write
from analysis.technical_indicators import rsi as compute_rsi
from data_ingestion.market_data_service import fetch_ohlcv
from utils.logger import get_logger

logger = get_logger(__name__)

ALERT_TYPES = ["price", "rsi", "volume"]
CONDITIONS = ["above", "below"]


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_alert(
    user_id: int,
    symbol: str,
    alert_type: str,
    condition: str,
    threshold: float,
    message: str = "",
) -> tuple[bool, str]:
    if alert_type not in ALERT_TYPES:
        return False, f"Invalid alert type. Choose from: {ALERT_TYPES}"
    if condition not in CONDITIONS:
        return False, f"Invalid condition. Choose from: {CONDITIONS}"
    try:
        aid = execute_write(
            """INSERT INTO alerts (user_id, symbol, alert_type, condition, threshold, message)
               VALUES (?,?,?,?,?,?)""",
            (user_id, symbol.upper(), alert_type, condition, threshold, message),
        )
        return True, f"Alert #{aid} created for {symbol.upper()}."
    except Exception as exc:
        return False, str(exc)


def get_alerts(user_id: int, active_only: bool = True) -> list[dict]:
    sql = "SELECT * FROM alerts WHERE user_id=?"
    params = [user_id]
    if active_only:
        sql += " AND is_active=1"
    sql += " ORDER BY created_at DESC"
    return execute_query(sql, tuple(params))


def delete_alert(alert_id: int, user_id: int) -> bool:
    n = execute_write("DELETE FROM alerts WHERE id=? AND user_id=?", (alert_id, user_id))
    return bool(n)


def toggle_alert(alert_id: int, user_id: int) -> bool:
    rows = execute_query("SELECT is_active FROM alerts WHERE id=? AND user_id=?", (alert_id, user_id))
    if not rows:
        return False
    new_state = 0 if rows[0]["is_active"] else 1
    execute_write("UPDATE alerts SET is_active=? WHERE id=?", (new_state, alert_id))
    return bool(new_state)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _check_condition(value: float, condition: str, threshold: float) -> bool:
    if condition == "above":
        return value > threshold
    if condition == "below":
        return value < threshold
    return False


def evaluate_alerts(user_id: int) -> list[dict]:
    """
    Check all active alerts for a user against live market data.
    Returns list of triggered alert dicts.
    """
    alerts = get_alerts(user_id, active_only=True)
    triggered = []

    for alert in alerts:
        sym = alert["symbol"]
        try:
            df = fetch_ohlcv(sym, period="5d")
            if df.empty:
                continue

            last = df.iloc[-1]
            atype = alert["alert_type"]

            if atype == "price":
                value = float(last["close"])
            elif atype == "rsi":
                rsi_series = compute_rsi(df)
                value = float(rsi_series.iloc[-1])
            elif atype == "volume":
                value = float(last["volume"])
            else:
                continue

            if _check_condition(value, alert["condition"], alert["threshold"]):
                execute_write(
                    "UPDATE alerts SET triggered=1, triggered_at=?, is_active=0 WHERE id=?",
                    (datetime.utcnow().isoformat(), alert["id"]),
                )
                alert["triggered_value"] = value
                triggered.append(alert)
                logger.info(
                    "Alert triggered: %s %s %s %.2f (actual=%.2f)",
                    sym, atype, alert["condition"], alert["threshold"], value,
                )
        except Exception as exc:
            logger.warning("Alert evaluation failed for %s: %s", sym, exc)

    return triggered


# ── Email notification ────────────────────────────────────────────────────────

def send_email_alert(to_email: str, subject: str, body: str) -> bool:
    """Send an email alert (requires SMTP env vars configured)."""
    import os
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not (smtp_host and smtp_user and smtp_pass):
        logger.warning("SMTP not configured — email alert skipped.")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False
