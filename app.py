"""
FOODFLOW - Food Demand Forecasting & Smart Preparation Planning
Entry point:  python -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT / "src" / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(
    page_title="FOODFLOW",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import ui  # noqa: E402

ui.setup_page()

pages = [
    st.Page("pages/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/forecast.py", title="Demand Forecasting", icon="📈"),
    st.Page("pages/analysis.py", title="Demand & Error Analysis", icon="🔍"),
    st.Page("pages/planner.py", title="Smart Preparation Planner", icon="🧑‍🍳"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("## 🍽️ FOODFLOW")
    st.caption("Food demand forecasting & smart preparation planning")
    st.divider()
    if ui.safe_load():
        _, source = ui.history()
        st.caption("**Model:** frozen reference XGBoost (no retraining in the app)")
        st.caption(f"**Forecast history:** {source}")
        st.caption("**Forecast window:** 2017-08-16 onwards (day after the last observed date)")
    else:
        st.stop()

nav.run()
