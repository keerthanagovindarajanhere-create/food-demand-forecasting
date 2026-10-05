from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts" / "model" / "xgb_reference.pkl"
FEATURE_PATH = ROOT / "artifacts" / "model" / "feature_columns.json"
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_feature_columns():

    with open(FEATURE_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_data():

    return pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"]
    )


model = load_model()
feature_columns = load_feature_columns()
data = load_data()


# ============================================================
# FEATURE GENERATION
# ============================================================

def create_forecast_features(
    historical_data,
    store_nbr,
    family,
    forecast_date,
    onpromotion
):

    # Import the SAME feature-generation function
    # used during model training.
    import sys

    src_models = ROOT / "src" / "models"

    if str(src_models) not in sys.path:
        sys.path.insert(0, str(src_models))

    from features import create_features

    # Historical data
    history = historical_data[
        ["date", "store_nbr", "family", "sales", "onpromotion"]
    ].copy()

    # Create the future observation.
    future_row = pd.DataFrame({
        "date": [pd.Timestamp(forecast_date)],
        "store_nbr": [store_nbr],
        "family": [family],
        "sales": [np.nan],
        "onpromotion": [int(onpromotion)],
    })

    # Append future observation.
    combined = pd.concat(
        [history, future_row],
        ignore_index=True
    )

    # Generate the exact original features.
    combined = create_features(combined)

    # Retrieve the future row.
    future_features = combined[
        (combined["date"] == pd.Timestamp(forecast_date))
        & (combined["store_nbr"] == store_nbr)
        & (combined["family"] == family)
    ].copy()

    if future_features.empty:
        raise ValueError(
            "Could not generate features for the selected forecast."
        )

    # One-hot encode family exactly as training did.
    future_features = pd.get_dummies(
        future_features,
        columns=["family"],
        dtype=int
    )

    # Ensure every model feature exists.
    for column in feature_columns:

        if column not in future_features.columns:
            future_features[column] = 0

    # Keep EXACT frozen-model feature order.
    X_future = future_features[feature_columns]

    return X_future


# ============================================================
# PAGE
# ============================================================

st.title("📈 Demand Forecast")

st.write(
    "Enter the forecast conditions below. FOODFLOW automatically "
    "constructs the historical lag and rolling features required "
    "by the frozen XGBoost model."
)

st.divider()


# ============================================================
# USER INPUTS
# ============================================================

st.subheader("Forecast Inputs")

col1, col2 = st.columns(2)

with col1:

    stores = sorted(
        data["store_nbr"].dropna().unique().tolist()
    )

    store = st.selectbox(
        "Store",
        stores
    )

    families = sorted(
        data["family"].dropna().unique().tolist()
    )

    family = st.selectbox(
        "Product Family",
        families
    )


with col2:

    min_date = data["date"].max() + pd.Timedelta(days=1)

    forecast_date = st.date_input(
        "Forecast Date",
        value=min_date.date(),
        min_value=min_date.date()
    )

    promotion = st.selectbox(
        "Promotion",
        ["No", "Yes"]
    )


onpromotion = 1 if promotion == "Yes" else 0


st.divider()


# ============================================================
# PREDICT
# ============================================================

if st.button(
    "🔮 Predict Demand",
    type="primary",
    use_container_width=True
):

    with st.spinner("Generating demand forecast..."):

        try:

            X_future = create_forecast_features(
                historical_data=data,
                store_nbr=store,
                family=family,
                forecast_date=forecast_date,
                onpromotion=onpromotion
            )

            # Check for missing features.
            missing = X_future.isna().sum().sum()

            if missing > 0:

                st.error(
                    "The model input contains missing features. "
                    "The forecast cannot be generated safely."
                )

                st.stop()

            # ==================================================
            # FROZEN MODEL INFERENCE
            # ==================================================

            prediction = model.predict(X_future)

            prediction = max(
                0,
                float(prediction[0])
            )

        except Exception as error:

            st.error(
                "Unable to generate the forecast."
            )

            st.exception(error)

            st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.success("Forecast generated successfully.")

    st.subheader("Predicted Demand")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Predicted Units",
            f"{prediction:,.0f}"
        )

    with result_col2:

        st.metric(
            "Store",
            str(store)
        )

    with result_col3:

        st.metric(
            "Promotion",
            promotion
        )


    st.markdown(
        f"""
        ### Forecast Summary

        FOODFLOW predicts approximately

        # {prediction:,.0f} units

        for **{family}** at **Store {store}**
        on **{pd.Timestamp(forecast_date).strftime("%d %B %Y")}**.

        Promotion status: **{promotion}**
        """
    )


    # ========================================================
    # MODEL INPUT PREVIEW
    # ========================================================

    with st.expander("View model input"):

        st.caption(
            "These are the automatically generated features "
            "passed to the frozen XGBoost model."
        )

        st.dataframe(
            X_future.T.rename(
                columns={X_future.index[0]: "value"}
            ),
            use_container_width=True
        )