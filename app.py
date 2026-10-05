from pathlib import Path
import json
import pickle

import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="FOODFLOW",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "artifacts" / "model" / "xgb_reference.pkl"
FEATURE_PATH = ROOT / "artifacts" / "model" / "feature_columns.json"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_features():

    with open(FEATURE_PATH, "r") as f:
        return json.load(f)


try:

    model = load_model()
    feature_columns = load_features()

except Exception as e:

    st.error("FOODFLOW could not load the frozen model artifacts.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🍽️ FOODFLOW")

    st.caption(
        "Food Demand Forecasting & "
        "Smart Preparation Optimization"
    )

    st.divider()

    st.success("Frozen model loaded")

    st.metric(
        "Model features",
        len(feature_columns)
    )

    st.divider()

    st.caption(
        "FOODFLOW uses the frozen reference XGBoost "
        "model for inference. No retraining is performed "
        "inside the application."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🍽️ FOODFLOW")

st.subheader(
    "Food Demand Forecasting & Smart Preparation Optimization"
)

st.write(
    "Forecast food demand using historical sales patterns, "
    "calendar information, store characteristics and promotion status."
)

st.divider()


# ============================================================
# MAIN OVERVIEW
# ============================================================

st.header("Forecasting System")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Reference Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "Validation MAE",
        "57.55"
    )

with col3:
    st.metric(
        "Validation RMSE",
        "198.03"
    )


st.divider()


# ============================================================
# HOW IT WORKS
# ============================================================

st.header("How FOODFLOW Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.subheader("1")
    st.write("User Inputs")
    st.caption(
        "Store, product family, date and promotion."
    )

with step2:

    st.subheader("2")
    st.write("Feature Generation")
    st.caption(
        "Historical lag, rolling and calendar features."
    )

with step3:

    st.subheader("3")
    st.write("Frozen XGBoost")
    st.caption(
        "Existing trained model performs inference."
    )

with step4:

    st.subheader("4")
    st.write("Demand Forecast")
    st.caption(
        "Predicted demand is returned to the user."
    )


st.divider()


# ============================================================
# RESEARCH CONTRIBUTION
# ============================================================

st.header("Research Findings")

st.write(
    "The research component investigates where demand forecasting "
    "errors are concentrated and evaluates a targeted tail-weighting "
    "strategy for high-demand observations."
)

finding1, finding2, finding3 = st.columns(3)

with finding1:

    st.info(
        "Error concentration\n\n"
        "5% of observations account for approximately "
        "93.86% of total squared error."
    )

with finding2:

    st.info(
        "High-demand regime\n\n"
        "Forecast errors increase substantially as demand magnitude "
        "increases."
    )

with finding3:

    st.info(
        "Tail weighting\n\n"
        "Giving greater training importance to high-demand observations "
        "slightly improves overall and high-demand MAE."
    )


st.divider()


# ============================================================
# NAVIGATION
# ============================================================

st.header("Explore FOODFLOW")

st.write(
    "Use the pages in the sidebar to interact with the forecasting "
    "system and explore the research findings."
)

st.page_link(
    "pages/1_Forecast.py",
    label="📈 Open Demand Forecast",
    icon="📈"
)

st.caption(
    "Additional research-analysis pages can be added without "
    "changing the frozen model."
)