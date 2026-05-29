"""
StockSense AI — Main Streamlit Application Entry Point
=======================================================
Handles: CSS injection, DB initialisation, auth wall, sidebar nav.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import PAGE_CONFIG, APP_NAME, APP_TAGLINE, APP_VERSION, MARKET_INDICES
from database.db_manager import init_db
from auth.auth_service import register_user, login_user, change_password
from auth.session_manager import set_session, get_session, clear_session, is_logged_in
from data_ingestion.market_data_service import get_index_data
from utils.logger import get_logger

logger = get_logger("app")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(**PAGE_CONFIG)


# ── CSS injection ─────────────────────────────────────────────────────────────
def _inject_css() -> None:
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


_inject_css()

# ── DB bootstrap (runs once) ──────────────────────────────────────────────────
@st.cache_resource
def _init_database():
    init_db()
    return True


_init_database()


# ── Auth wall ─────────────────────────────────────────────────────────────────

def _show_auth() -> None:
    """Render login / register form when no session exists."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align:center; padding: 2rem 0 1rem;'>
              <div style='width: 64px; height: 64px; background: #1a56db; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; color: white; font-weight: 900; font-size: 2.2rem; font-family: sans-serif; margin-bottom: 0.5rem;'>S</div>
              <h1 style='font-size:2rem; font-weight:800; color:#111827; margin:0.25rem 0;'>StockSense AI</h1>
              <p style='color:#6b7280; font-size:1rem; margin:0;'>Stock Market Prediction & Risk Analysis Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                st.markdown("#### Welcome back")
                identifier = st.text_input("Username or Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not identifier or not password:
                    st.error("Please fill in all fields.")
                else:
                    with st.spinner("Authenticating…"):
                        ok, msg, user = login_user(identifier, password)
                    if ok:
                        set_session(user)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_register:
            with st.form("register_form"):
                st.markdown("#### Create your account")
                col_a, col_b = st.columns(2)
                with col_a:
                    full_name = st.text_input("Full Name", placeholder="Jane Doe")
                    username = st.text_input("Username", placeholder="janedoe")
                with col_b:
                    email = st.text_input("Email", placeholder="jane@example.com")
                    pw = st.text_input("Password", type="password", placeholder="Min. 8 chars, 1 uppercase, 1 digit", help="Must be at least 8 characters, containing at least one uppercase letter and one digit.")
                pw2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                submitted_r = st.form_submit_button("Create Account", use_container_width=True)

            if submitted_r:
                if pw != pw2:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Creating account…"):
                        ok, msg = register_user(username, email, pw, full_name)
                    if ok:
                        # Log the newly registered user in automatically
                        with st.spinner("Logging you in automatically…"):
                            login_ok, login_msg, user = login_user(username, pw)
                        if login_ok:
                            set_session(user)
                            st.success(f"{msg} Auto-logged in! Redirecting...")
                            st.rerun()
                        else:
                            st.success(msg)
                    else:
                        st.error(msg)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(user: dict) -> None:
    with st.sidebar:
        # Logo + brand
        st.markdown(
            """
            <div style='padding: 0.5rem 0 1rem;'>
              <div style='display:flex; align-items:center; gap:0.6rem;'>
                <div style='width: 32px; height: 32px; background: #2563eb; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 900; font-size: 1.1rem; font-family: sans-serif;'>S</div>
                <div>
                  <div style='font-size:1.1rem; font-weight:800; color:#111827; line-height:1.2;'>StockSense AI</div>
                  <div style='font-size:0.7rem; color:#6b7280; font-weight:500;'>v1.0 · FinTech Platform</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # User info
        st.markdown(
            f"""
            <div style='background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px;
                        padding:0.75rem 1rem; margin-bottom:1rem;'>
              <div style='font-size:0.8rem; color:#4b5563; font-weight:500;'>Signed in as</div>
              <div style='font-size:0.95rem; font-weight:700; color:#111827;'>{user.get("full_name") or user["username"]}</div>
              <div style='font-size:0.75rem; color:#6b7280;'>{user.get("email","")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Market indices strip
        st.markdown("**Market Overview**")
        try:
            indices = get_index_data(MARKET_INDICES)
            for name, data in indices.items():
                price = data.get("price")
                if price:
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"font-size:0.8rem; padding:0.15rem 0; color:#374151;'>"
                        f"<span>{name}</span><strong>${price:,.0f}</strong></div>",
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

        st.markdown("---")

        # Navigation
        st.markdown("**Navigation**")
        pages = {
            "Home Dashboard":        "pages/00_home.py",
            "Stock Analysis":         "pages/01_stock_analysis.py",
            "Prediction Center":      "pages/02_prediction_center.py",
            "Risk Analytics":         "pages/03_risk_analytics.py",
            "Sentiment Center":       "pages/04_sentiment_center.py",
            "Portfolio Manager":      "pages/05_portfolio_manager.py",
            "Backtesting Lab":        "pages/06_backtesting.py",
            "Market Scanner":         "pages/07_market_scanner.py",
            "Settings":              "pages/08_settings.py",
        }

        if "current_page" not in st.session_state:
            st.session_state.current_page = "Home Dashboard"

        for page_name in pages:
            selected = st.session_state.current_page == page_name
            btn_style = "primary" if selected else "secondary"
            if st.button(page_name, key=f"nav_{page_name}", use_container_width=True,
                         type=btn_style if selected else "secondary"):
                st.session_state.current_page = page_name
                st.rerun()

        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            clear_session()
            st.rerun()


# ── Page router ───────────────────────────────────────────────────────────────

def _route_page(page_name: str) -> None:
    import importlib.util

    pages_map = {
        "Home Dashboard":        "pages/00_home.py",
        "Stock Analysis":         "pages/01_stock_analysis.py",
        "Prediction Center":      "pages/02_prediction_center.py",
        "Risk Analytics":         "pages/03_risk_analytics.py",
        "Sentiment Center":       "pages/04_sentiment_center.py",
        "Portfolio Manager":      "pages/05_portfolio_manager.py",
        "Backtesting Lab":        "pages/06_backtesting.py",
        "Market Scanner":         "pages/07_market_scanner.py",
        "Settings":              "pages/08_settings.py",
    }

    page_file = ROOT / pages_map.get(page_name, "pages/00_home.py")
    spec = importlib.util.spec_from_file_location("page_module", page_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        st.error(f"Page error: {exc}")
        logger.error("Page routing error [%s]: %s", page_name, exc, exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not is_logged_in():
        _show_auth()
        return

    user = get_session()
    _render_sidebar(user)
    current = st.session_state.get("current_page", "Home Dashboard")
    _route_page(current)


if __name__ == "__main__":
    main()
else:
    main()
