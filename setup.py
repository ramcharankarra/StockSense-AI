#!/usr/bin/env python3
"""
StockSense AI — One-shot Setup Script
=======================================
Run: python setup.py
- Creates .env from template if not exists
- Initialises the database
- Downloads required NLTK data
- Validates key imports
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _banner(text: str) -> None:
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def check_python_version():
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required. Current:", sys.version)
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")


def create_env_file():
    env_template = ROOT / ".env.template"
    env_file = ROOT / ".env"
    if env_file.exists():
        print("✅ .env file already exists — skipping.")
    else:
        import shutil
        shutil.copy(env_template, env_file)
        print("✅ .env created from template. Edit it to add your API keys.")


def init_database():
    sys.path.insert(0, str(ROOT))
    try:
        from database.db_manager import init_db
        init_db()
        print("✅ Database initialised.")
    except Exception as e:
        print(f"⚠️  Database init warning: {e}")


def download_nltk():
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        import nltk
        for resource in ["vader_lexicon", "punkt", "stopwords"]:
            nltk.download(resource, quiet=True)
        print("✅ NLTK data downloaded.")
    except ImportError:
        print("⚠️  NLTK not installed yet — run pip install -r requirements.txt first.")


def validate_imports():
    packages = {
        "streamlit": "streamlit",
        "yfinance": "yfinance",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "plotly": "plotly",
        "bcrypt": "bcrypt",
        "textblob": "textblob",
        "nltk": "nltk",
    }
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} — run: pip install {package}")
            all_ok = False

    # Optional
    for module, package in [("tensorflow", "tensorflow"), ("shap", "shap")]:
        try:
            __import__(module)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️  {package} not installed (optional — deep learning / SHAP disabled)")

    return all_ok


def main():
    _banner("StockSense AI — Setup")

    _banner("1. Python Version")
    check_python_version()

    _banner("2. Environment File")
    create_env_file()

    _banner("3. Required Packages")
    ok = validate_imports()
    if not ok:
        print("\n❌ Missing packages. Install with:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)

    _banner("4. Database Initialisation")
    init_database()

    _banner("5. NLTK Data")
    download_nltk()

    _banner("✨ Setup Complete!")
    print("Run the app with:")
    print("  streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
