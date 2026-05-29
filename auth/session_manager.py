"""
Session Manager
===============
Thin wrapper around Streamlit's session_state to manage
authenticated user sessions across page reloads.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


SESSION_KEY = "ss_user"


def set_session(user: dict) -> None:
    """Persist authenticated user dict in Streamlit session state."""
    st.session_state[SESSION_KEY] = user


def get_session() -> Optional[dict]:
    """Return the current authenticated user dict, or None."""
    return st.session_state.get(SESSION_KEY)


def clear_session() -> None:
    """Log out the current user."""
    if SESSION_KEY in st.session_state:
        del st.session_state[SESSION_KEY]


def is_logged_in() -> bool:
    return get_session() is not None


def require_login() -> dict:
    """
    Guard decorator for Streamlit pages.
    If not logged in, shows a login prompt and stops execution.
    Returns the user dict if authenticated.
    """
    user = get_session()
    if user is None:
        st.warning("Please log in to access this page.")
        st.stop()
    return user


def get_user_id() -> Optional[int]:
    user = get_session()
    return user["id"] if user else None


def get_username() -> Optional[str]:
    user = get_session()
    return user["username"] if user else None
