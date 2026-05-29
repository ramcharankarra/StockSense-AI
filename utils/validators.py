"""
Input Validators
================
Centralised validation functions used by API / UI layers.
"""

import re
from typing import Optional


# ── Auth validators ───────────────────────────────────────────────────────────

def validate_email(email: str) -> tuple[bool, str]:
    pattern = r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True, ""
    return False, "Invalid email address format."


def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    return True, ""


def validate_username(username: str) -> tuple[bool, str]:
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 32:
        return False, "Username must be 32 characters or fewer."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username may only contain letters, digits, and underscores."
    return True, ""


# ── Financial validators ──────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> tuple[bool, str]:
    ticker = ticker.strip().upper()
    if not ticker:
        return False, "Ticker symbol cannot be empty."
    if not re.match(r"^[A-Z0-9.\-^]{1,10}$", ticker):
        return False, f"'{ticker}' is not a valid ticker symbol."
    return True, ""


def validate_positive_number(value, name: str = "Value") -> tuple[bool, str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"{name} must be a number."
    if v <= 0:
        return False, f"{name} must be greater than zero."
    return True, ""


def validate_shares(shares) -> tuple[bool, str]:
    return validate_positive_number(shares, "Number of shares")


def validate_price(price) -> tuple[bool, str]:
    return validate_positive_number(price, "Price")


def validate_portfolio_name(name: str) -> tuple[bool, str]:
    if not name or not name.strip():
        return False, "Portfolio name cannot be empty."
    if len(name) > 64:
        return False, "Portfolio name must be 64 characters or fewer."
    return True, ""
