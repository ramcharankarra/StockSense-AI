"""
09_settings.py – ⚙️ Settings & Profile
========================================
Four-tab settings hub: Profile, Preferences, About, and Data Management.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import streamlit as st

from auth.session_manager import require_login, set_session
from auth.auth_service import update_profile, change_password, get_user_by_id
from database.db_manager import get_user_settings, save_user_settings, execute_query
from config import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    DEFAULT_WATCHLIST,
    SUPPORTED_PERIODS,
    RISK_FREE_RATE,
)

# ── Page config ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Settings – StockSense AI",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ── Auth guard ─────────────────────────────────────────────────────────────────
user = require_login()
user_id: int = user["id"]

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .settings-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
        padding:20px 24px; margin-bottom:16px;
    }
    .info-label { color:#64748b; font-size:0.8rem; font-weight:600; text-transform:uppercase; }
    .info-value { color:#0f172a; font-size:1rem; margin-bottom:12px; }
    .tech-badge {
        display:inline-block; background:#dbeafe; color:#1e40af;
        padding:3px 12px; border-radius:20px; font-size:0.8rem;
        font-weight:600; margin:3px;
    }
    .section-title {
        font-size:1.05rem; font-weight:700; color:#1a56db;
        border-bottom:2px solid #1a56db; padding-bottom:4px; margin:16px 0 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("# Settings & Profile")
st.markdown(
    "<p style='color:#64748b; margin-top:-12px;'>"
    "Manage your profile, preferences, and platform configuration."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_profile, tab_prefs, tab_about, tab_data = st.tabs(
    ["Profile", "Preferences", "About", "Data Management"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Profile
# ─────────────────────────────────────────────────────────────────────────────
with tab_profile:
    # Refresh user data from DB on every load for accuracy
    try:
        fresh_user = get_user_by_id(user_id) or user
    except Exception:
        fresh_user = user

    # ── Current user info ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Account Information</div>", unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        with st.container(border=True):
            st.markdown(
                f"<div class='info-label'>Username</div>"
                f"<div class='info-value'>{fresh_user.get('username', '—')}</div>"
                f"<div class='info-label'>Email</div>"
                f"<div class='info-value'>{fresh_user.get('email', '—')}</div>",
                unsafe_allow_html=True,
            )
    with info_col2:
        with st.container(border=True):
            full_name = fresh_user.get("full_name") or "Not set"
            created_at = str(fresh_user.get("created_at", ""))[:10] or "—"
            last_login = str(fresh_user.get("last_login", ""))[:16].replace("T", " ") or "—"
            st.markdown(
                f"<div class='info-label'>Full Name</div>"
                f"<div class='info-value'>{full_name}</div>"
                f"<div class='info-label'>Member Since</div>"
                f"<div class='info-value'>{created_at}</div>",
                unsafe_allow_html=True,
            )

    st.caption(f"Last login: **{last_login}**")

    st.divider()

    # ── Edit Profile form ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Edit Profile</div>", unsafe_allow_html=True)

    with st.form("edit_profile_form"):
        ep_col1, ep_col2 = st.columns(2)
        with ep_col1:
            new_full_name = st.text_input(
                "Full Name",
                value=fresh_user.get("full_name") or "",
                placeholder="Jane Doe",
            )
        with ep_col2:
            new_email = st.text_input(
                "Email Address",
                value=fresh_user.get("email") or "",
                placeholder="jane@example.com",
            )
        ep_submitted = st.form_submit_button("Save Profile", type="primary")

    if ep_submitted:
        if not new_full_name and not new_email:
            st.warning("Enter a new full name or email to update.")
        else:
            try:
                ok, msg = update_profile(
                    user_id,
                    full_name=new_full_name or None,
                    email=new_email or None,
                )
                if ok:
                    st.success(f"{msg}")
                    # Refresh session state with updated values
                    updated = get_user_by_id(user_id)
                    if updated:
                        set_session(updated)
                    st.rerun()
                else:
                    st.error(f"{msg}")
            except Exception as exc:
                st.error(f"Profile update failed: {exc}")

    st.divider()

    # ── Change Password form ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Change Password</div>", unsafe_allow_html=True)

    with st.form("change_password_form"):
        cp_col1, cp_col2, cp_col3 = st.columns(3)
        with cp_col1:
            old_pw = st.text_input("Current Password", type="password", placeholder="••••••••")
        with cp_col2:
            new_pw = st.text_input(
                "New Password",
                type="password",
                placeholder="Min 8 chars",
                help="Must be at least 8 characters.",
            )
        with cp_col3:
            confirm_pw = st.text_input(
                "Confirm New Password", type="password", placeholder="Re-enter new password"
            )
        cp_submitted = st.form_submit_button("Change Password", type="primary")

    if cp_submitted:
        if not old_pw or not new_pw or not confirm_pw:
            st.error("Please fill in all password fields.")
        elif new_pw != confirm_pw:
            st.error("New password and confirmation do not match.")
        else:
            try:
                ok, msg = change_password(user_id, old_pw, new_pw)
                if ok:
                    st.success(f"{msg}")
                else:
                    st.error(f"{msg}")
            except Exception as exc:
                st.error(f"Password change failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Preferences
# ─────────────────────────────────────────────────────────────────────────────
with tab_prefs:
    try:
        settings = get_user_settings(user_id)
    except Exception as exc:
        st.error(f"Failed to load preferences: {exc}")
        settings = {}

    st.markdown("<div class='section-title'>Platform Preferences</div>", unsafe_allow_html=True)

    with st.form("preferences_form"):
        pref_col1, pref_col2 = st.columns(2)

        with pref_col1:
            default_ticker = st.text_input(
                "Default Ticker Symbol",
                value=settings.get("default_ticker") or "AAPL",
                placeholder="e.g. AAPL",
                help="This ticker will be pre-selected on analysis pages.",
            ).strip().upper()

            default_period = st.selectbox(
                "Default Analysis Period",
                options=SUPPORTED_PERIODS,
                index=SUPPORTED_PERIODS.index(settings.get("default_period") or "1y")
                if (settings.get("default_period") or "1y") in SUPPORTED_PERIODS
                else 3,
                help="Default lookback period used across analysis pages.",
            )

            rfr_pct = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(settings.get("risk_free_rate") or RISK_FREE_RATE) * 100,
                step=0.25,
                format="%.2f",
                help="Annual risk-free rate used for Sharpe, Sortino calculations.",
            )

        with pref_col2:
            saved_watchlist: list = settings.get("watchlist") or []
            all_ticker_options = sorted(set(DEFAULT_WATCHLIST) | set(saved_watchlist))

            watchlist_selected = st.multiselect(
                "Watchlist",
                options=all_ticker_options,
                default=[t for t in saved_watchlist if t in all_ticker_options],
                help="Tickers to track in your personal watchlist.",
            )

            custom_ticker_raw = st.text_input(
                "Add Custom Tickers (comma-separated)",
                placeholder="e.g. TSM, BABA, SNOW",
                help="Add tickers not in the default list. They will be merged into your watchlist.",
            )

            email_alerts = st.toggle(
                "Email Alert Notifications",
                value=bool(settings.get("email_alerts")),
                help="Receive email notifications when alerts are triggered (requires SMTP config).",
            )

        pref_submitted = st.form_submit_button("Save Preferences", type="primary", use_container_width=True)

    if pref_submitted:
        # Merge custom tickers
        custom_tickers = [
            t.strip().upper() for t in custom_ticker_raw.split(",") if t.strip()
        ]
        final_watchlist = sorted(set(watchlist_selected) | set(custom_tickers))

        try:
            save_user_settings(
                user_id,
                default_ticker=default_ticker or "AAPL",
                default_period=default_period,
                risk_free_rate=rfr_pct / 100.0,
                watchlist=final_watchlist,
                email_alerts=int(email_alerts),
            )
            st.success(
                f"Preferences saved! "
                f"Watchlist: {len(final_watchlist)} tickers · "
                f"Default: **{default_ticker}** · "
                f"Period: **{default_period}** · "
                f"RFR: **{rfr_pct:.2f}%**"
            )
        except Exception as exc:
            st.error(f"Failed to save preferences: {exc}")

    # Watchlist preview
    try:
        fresh_settings = get_user_settings(user_id)
        wl = fresh_settings.get("watchlist") or []
        if wl:
            st.divider()
            st.markdown("**Current Watchlist**")
            badge_html = " ".join(f"<span class='tech-badge'>{t}</span>" for t in wl)
            st.markdown(badge_html, unsafe_allow_html=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – About
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    # ── App info card ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Application</div>", unsafe_allow_html=True)

    about_col1, about_col2 = st.columns([2, 1])
    with about_col1:
        with st.container(border=True):
            st.markdown(
                f"<h3 style='margin:0; color:#1a56db;'>{APP_NAME}</h3>"
                f"<p style='color:#64748b; margin:4px 0 12px;'>{APP_TAGLINE}</p>"
                f"<b>Version:</b> <code>{APP_VERSION}</code><br>"
                f"<b>License:</b> MIT &nbsp;|&nbsp; "
                f"<b>GitHub:</b> "
                f"<a href='https://github.com/your-org/stocksense-ai' target='_blank'>"
                f"github.com/your-org/stocksense-ai</a>",
                unsafe_allow_html=True,
            )

    with about_col2:
        st.metric("Version", APP_VERSION)
        st.metric("Database", "SQLite (WAL)")
        st.metric("Deployment", "Streamlit")

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Tech Stack</div>", unsafe_allow_html=True)

    TECH_STACK = [
        ("Python 3.11+", "Core runtime"),
        ("Streamlit", "Web UI framework"),
        ("yfinance", "Market data ingestion"),
        ("NLTK / VADER", "Sentiment analysis"),
        ("scikit-learn", "Classical ML (RF, SVM, LR)"),
        ("XGBoost", "Gradient boosted trees"),
        ("TensorFlow / Keras", "LSTM & GRU deep learning"),
        ("SQLite", "Local persistence (WAL mode)"),
        ("Plotly", "Interactive charts"),
        ("bcrypt", "Password hashing"),
        ("pandas / NumPy", "Data manipulation"),
    ]

    badge_html = " ".join(
        f"<span class='tech-badge'>{name}</span>" for name, _ in TECH_STACK
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("")  # spacer

    tech_df_data = [{"Component": n, "Purpose": p} for n, p in TECH_STACK]
    import pandas as pd
    st.dataframe(
        pd.DataFrame(tech_df_data),
        use_container_width=True,
        hide_index=True,
    )

    # ── Architecture diagram ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Architecture Overview</div>", unsafe_allow_html=True)

    st.markdown(
        """
| Layer | Modules | Responsibility |
|---|---|---|
| **Data Ingestion** | `data_ingestion/market_data_service.py` | Fetch OHLCV from yfinance, cache to SQLite |
| **Analysis** | `analysis/technical_indicators.py`, `analysis/risk_metrics.py` | RSI, MACD, Bollinger, VaR, Sharpe |
| **Sentiment** | `analysis/sentiment_analyzer.py` | News NLP via NLTK VADER |
| **ML / Forecasting** | `ml/` (Random Forest, LSTM, XGBoost) | Price direction & return predictions |
| **Portfolio** | `portfolio/portfolio_service.py` | Holdings, P&L, diversification |
| **Backtesting** | `backtesting/backtesting_engine.py` | Strategy simulation on historical data |
| **Auth** | `auth/auth_service.py`, `auth/session_manager.py` | Registration, login, bcrypt, sessions |
| **Database** | `database/db_manager.py`, `database/schema.sql` | SQLite abstraction, CRUD helpers |
| **UI** | `app.py`, `pages/` | Streamlit multi-page app |
        """,
        unsafe_allow_html=False,
    )

    # ── Credits ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Credits</div>", unsafe_allow_html=True)
    st.markdown(
        """
- **yfinance** by Ran Aroussi — free market data API wrapper
- **VADER Sentiment** by C.J. Hutto & E.E. Gilbert — social media sentiment lexicon
- **Plotly** — interactive charting
- **Streamlit** team — rapid data-app framework
- All open-source contributors whose packages power this platform
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Data Management
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("<div class='section-title'>Database Statistics</div>", unsafe_allow_html=True)

    # Gather stats
    @st.cache_data(ttl=30, show_spinner=False)
    def _db_stats() -> dict:
        try:
            stocks_count = execute_query("SELECT COUNT(*) AS n FROM stocks")[0]["n"]
        except Exception:
            stocks_count = 0
        try:
            prices_count = execute_query("SELECT COUNT(*) AS n FROM stock_prices")[0]["n"]
        except Exception:
            prices_count = 0
        try:
            preds_count = execute_query("SELECT COUNT(*) AS n FROM predictions")[0]["n"]
        except Exception:
            preds_count = 0
        try:
            users_count = execute_query("SELECT COUNT(*) AS n FROM users")[0]["n"]
        except Exception:
            users_count = 0
        try:
            backtests_count = execute_query("SELECT COUNT(*) AS n FROM backtests")[0]["n"]
        except Exception:
            backtests_count = 0
        return {
            "stocks_count": stocks_count,
            "prices_count": prices_count,
            "preds_count": preds_count,
            "users_count": users_count,
            "backtests_count": backtests_count,
        }

    try:
        stats = _db_stats()
        dm_c1, dm_c2, dm_c3, dm_c4, dm_c5 = st.columns(5)
        dm_c1.metric("Stocks Cached", f"{stats['stocks_count']:,}")
        dm_c2.metric("Price Records", f"{stats['prices_count']:,}")
        dm_c3.metric("Predictions", f"{stats['preds_count']:,}")
        dm_c4.metric("Registered Users", f"{stats['users_count']:,}")
        dm_c5.metric("Backtests Run", f"{stats['backtests_count']:,}")
    except Exception as exc:
        st.error(f"Could not load DB stats: {exc}")

    st.divider()

    # ── Cache management ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Cache Management</div>", unsafe_allow_html=True)

    cache_col1, cache_col2 = st.columns(2)

    with cache_col1:
        with st.container(border=True):
            st.markdown("**Clear Streamlit Data Cache**")
            st.caption(
                "Clears all `@st.cache_data` in-memory caches. "
                "Market data will be re-fetched on next load. "
                "This does **not** delete database records."
            )
            if st.button("Clear Data Cache", use_container_width=True):
                try:
                    st.cache_data.clear()
                    st.success("Streamlit data cache cleared successfully.")
                except Exception as exc:
                    st.error(f"Cache clear failed: {exc}")

    with cache_col2:
        with st.container(border=True):
            st.markdown("**Export Portfolio Data**")
            st.caption(
                "Download your portfolio holdings as a CSV file for external analysis."
            )
            try:
                holdings_raw = execute_query(
                    """
                    SELECT h.symbol, h.shares, h.avg_cost, h.added_at,
                           p.name AS portfolio_name
                    FROM holdings h
                    JOIN portfolios p ON h.portfolio_id = p.id
                    WHERE p.user_id = ?
                    ORDER BY p.name, h.symbol
                    """,
                    (user_id,),
                )
                import pandas as pd

                if holdings_raw:
                    df_holdings = pd.DataFrame(holdings_raw)
                    csv_bytes = df_holdings.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Portfolio CSV",
                        data=csv_bytes,
                        file_name="stocksense_portfolio.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No portfolio holdings to export.")
            except Exception as exc:
                st.warning(f"Portfolio export unavailable: {exc}")

    st.divider()

    # ── Danger zone ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title' style='color:#dc2626; border-color:#dc2626;'>"
        "Danger Zone</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Clear My Alert History", expanded=False):
        st.warning(
            "This will permanently delete all **triggered (inactive) alerts** "
            "from your account. Active alerts are preserved.",
            icon="⚠️",
        )
        confirm_clear = st.text_input(
            "Type **DELETE** to confirm",
            placeholder="DELETE",
            key="danger_confirm_input",
        )
        if st.button("Clear Alert History", type="primary"):
            if confirm_clear.strip().upper() == "DELETE":
                try:
                    from database.db_manager import execute_write as _ew
                    _ew(
                        "DELETE FROM alerts WHERE user_id=? AND is_active=0 AND triggered=1",
                        (user_id,),
                    )
                    st.success("Triggered alert history cleared.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to clear history: {exc}")
            else:
                st.error("Confirmation text does not match. Type DELETE to proceed.")
