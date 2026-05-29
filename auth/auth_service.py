"""
Authentication Service
======================
Handles user registration, login, logout, and password management.
Uses bcrypt for hashing. Session tokens are SHA-256 UUIDs stored in
Streamlit session state (server-side) — no JWT overhead for MVP.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Optional

import bcrypt

from database.db_manager import execute_query, execute_write, get_db
from utils.logger import get_logger
from utils.validators import validate_email, validate_password, validate_username

logger = get_logger(__name__)


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Registration ──────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str, full_name: str = "") -> tuple[bool, str]:
    """
    Create a new user account.
    Returns (success: bool, message: str).
    """
    # Validate inputs
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
    ok, msg = validate_email(email)
    if not ok:
        return False, msg
    ok, msg = validate_password(password)
    if not ok:
        return False, msg

    # Duplicate check
    existing = execute_query(
        "SELECT id FROM users WHERE username=? OR email=?", (username, email)
    )
    if existing:
        return False, "Username or email already registered."

    pw_hash = hash_password(password)
    try:
        user_id = execute_write(
            "INSERT INTO users (username, email, password_hash, full_name) VALUES (?,?,?,?)",
            (username, email.lower(), pw_hash, full_name),
        )
        # Create default settings
        execute_write(
            "INSERT INTO user_settings (user_id) VALUES (?)", (user_id,)
        )
        logger.info("New user registered: %s (id=%s)", username, user_id)
        return True, "Account created successfully! You can now log in."
    except Exception as exc:
        logger.error("Registration failed: %s", exc)
        return False, "Registration failed due to an internal error."


# ── Login / Logout ────────────────────────────────────────────────────────────

def login_user(username_or_email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """
    Authenticate a user.
    Returns (success, message, user_dict | None).
    """
    rows = execute_query(
        "SELECT * FROM users WHERE (username=? OR email=?) AND is_active=1",
        (username_or_email, username_or_email.lower()),
    )
    if not rows:
        return False, "Invalid username/email or password.", None

    user = rows[0]
    if not verify_password(password, user["password_hash"]):
        return False, "Invalid username/email or password.", None

    # Update last_login
    execute_write(
        "UPDATE users SET last_login=? WHERE id=?",
        (datetime.utcnow().isoformat(), user["id"]),
    )
    logger.info("User logged in: %s", user["username"])
    return True, "Login successful!", dict(user)


def get_user_by_id(user_id: int) -> Optional[dict]:
    rows = execute_query("SELECT * FROM users WHERE id=?", (user_id,))
    return dict(rows[0]) if rows else None


# ── Profile ───────────────────────────────────────────────────────────────────

def update_profile(user_id: int, full_name: str = None, email: str = None) -> tuple[bool, str]:
    updates = {}
    if full_name:
        updates["full_name"] = full_name
    if email:
        ok, msg = validate_email(email)
        if not ok:
            return False, msg
        updates["email"] = email.lower()
    if not updates:
        return False, "Nothing to update."
    sets = ", ".join(f"{k}=?" for k in updates)
    execute_write(f"UPDATE users SET {sets} WHERE id=?", (*updates.values(), user_id))
    return True, "Profile updated."


def change_password(user_id: int, old_pw: str, new_pw: str) -> tuple[bool, str]:
    rows = execute_query("SELECT password_hash FROM users WHERE id=?", (user_id,))
    if not rows or not verify_password(old_pw, rows[0]["password_hash"]):
        return False, "Current password is incorrect."
    ok, msg = validate_password(new_pw)
    if not ok:
        return False, msg
    execute_write(
        "UPDATE users SET password_hash=? WHERE id=?",
        (hash_password(new_pw), user_id),
    )
    return True, "Password changed successfully."
