import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

from prepare_xgb_data import prepare_data
from features import create_features


# =========================================================
# RMSLE
# =========================================================

def calculate_rmsle(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            np.log1p(y_true),
            np.log1p(np.maximum(y_pred, 0))
        )
    )


# =========================================================
# LOAD VALIDATION METADATA
# =========================================================

def get_validation_metadata():

    print("\n>>> Loading validation metadata...", flush=True)

    df = pd.read_csv(
        "data/raw/train.csv",
        parse_dates=["date"]
    )

    print(">>> Creating features for metadata...", flush=True)

    df = create_features(df)

    df = df.dropna(
        subset=[
            "lag_28",
            "rolling_mean_28"
        ]
    )

    validation = df[
        (df["date"] >= "2017-07-19") &
        (df["date"] <= "2017-08-15")
    ].copy()

    metadata = validation[
        [
            "date",
            "store_nbr",
            "family",
            "sales"
        ]
    ].reset_index(drop=True)

    print(
        f">>> Validation metadata loaded: {len(metadata):,} rows",
        flush=True
    )

    return metadata


# =========================================================
# START
# =========================================================

start_time = time.time()

print("=" * 60, flush=True)
print("FOODFLOW XGBOOST FEATURE IMPORTANCE + ERROR ANALYSIS", flush=True)
print("=" * 60, flush=True)


# =========================================================
# PREPARE DATA
# =========================================================

print("\n>>> Preparing XGBoost data...", flush=True)

X_train, y_train, X_val, y_val = prepare_data()

print(
    f">>> Training shape:   {X_train.shape}",
    flush=True
)

print(
    f">>> Validation shape: {X_val.shape}",
    flush=True
)


# =========================================================
# VALIDATION METADATA
# =========================================================

metadata = get_validation_metadata()

# Safety check
if len(metadata) != len(X_val):

    raise ValueError(
        f"Validation metadata rows ({len(metadata)}) "
        f"do not match X_val rows ({len(X_val)})."
    )

print(
    ">>> Metadata/X_val alignment verified.",
    flush=True
)


# =========================================================
# XGBOOST MODEL
# =========================================================

print("\n>>> Creating XGBoost model...", flush=True)

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


# =========================================================
# TRAIN
# =========================================================

print("\n>>> STARTING XGBOOST TRAINING", flush=True)

train_start = time.time()

model.fit(
    X_train,
    y_train,
    verbose=False
)

train_time = time.time() - train_start

print(
    f">>> XGBoost training complete: {train_time:.2f} seconds",
    flush=True
)


# =========================================================
# PREDICTIONS
# =========================================================

print("\n>>> Generating validation predictions...", flush=True)

predictions = model.predict(X_val)

predictions = np.maximum(predictions, 0)

print(">>> Predictions generated.", flush=True)


# =========================================================
# BASELINE METRICS
# =========================================================

print("\n>>> Calculating baseline metrics...", flush=True)

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

rmsle_value = calculate_rmsle(
    y_val,
    predictions
)


print("\n" + "=" * 60, flush=True)
print("XGBOOST RESULTS", flush=True)
print("=" * 60, flush=True)

print(f"MAE:   {mae:.4f}", flush=True)
print(f"RMSE:  {rmse:.4f}", flush=True)
print(f"RMSLE: {rmsle_value:.4f}", flush=True)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\n" + "=" * 60, flush=True)
print("FEATURE IMPORTANCE", flush=True)
print("=" * 60, flush=True)

print("\n>>> Calculating feature importance...", flush=True)

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nTop 20 features:\n", flush=True)

print(
    importance.head(20).to_string(index=False),
    flush=True
)

print(
    "\n>>> PASSED FEATURE IMPORTANCE",
    flush=True
)


# =========================================================
# FEATURE IMPORTANCE PLOT
# =========================================================

print(
    "\n>>> STARTING PLOT",
    flush=True
)

top_features = (
    importance
    .head(20)
    .sort_values("importance")
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    "FOODFLOW XGBoost — Top 20 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "xgb_feature_importance.png",
    dpi=300
)

# IMPORTANT:
# Do NOT use plt.show()
plt.close()

print(
    ">>> PLOT FINISHED",
    flush=True
)


# =========================================================
# ERROR DATAFRAME
# =========================================================

print(
    "\n>>> STARTING ERROR ANALYSIS",
    flush=True
)

error_df = metadata.copy()

error_df["predicted"] = predictions

error_df["error"] = (
    error_df["sales"] -
    error_df["predicted"]
)

error_df["absolute_error"] = (
    error_df["error"].abs()
)

error_df["squared_error"] = (
    error_df["error"] ** 2
)

print(
    f">>> Error dataframe created: {len(error_df):,} rows",
    flush=True
)


# =========================================================
# WORST PREDICTIONS
# =========================================================

print("\n" + "=" * 60, flush=True)
print("TOP 20 WORST PREDICTIONS", flush=True)
print("=" * 60, flush=True)

print(
    "\n>>> Sorting validation errors...",
    flush=True
)

worst_predictions = (
    error_df
    .nlargest(
        20,
        "absolute_error"
    )
)

print(
    worst_predictions[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "predicted",
            "absolute_error"
        ]
    ].to_string(index=False),
    flush=True
)


# =========================================================
# ERROR BY FOOD FAMILY
# =========================================================

print("\n" + "=" * 60, flush=True)
print("ERROR BY FOOD FAMILY", flush=True)
print("=" * 60, flush=True)

print(
    ">>> Calculating family errors...",
    flush=True
)

family_error = (
    error_df
    .groupby("family")
    .agg(
        observations=("sales", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", "mean"),
        actual_mean=("sales", "mean"),
        predicted_mean=("predicted", "mean")
    )
)

family_error["RMSE"] = np.sqrt(
    family_error["RMSE"]
)

family_error = family_error.sort_values(
    "MAE",
    ascending=False
)

print(
    family_error.to_string(),
    flush=True
)


# =========================================================
# ERROR BY STORE
# =========================================================

print("\n" + "=" * 60, flush=True)
print("ERROR BY STORE", flush=True)
print("=" * 60, flush=True)

print(
    ">>> Calculating store errors...",
    flush=True
)

store_error = (
    error_df
    .groupby("store_nbr")
    .agg(
        observations=("sales", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", "mean"),
        actual_mean=("sales", "mean"),
        predicted_mean=("predicted", "mean")
    )
)

store_error["RMSE"] = np.sqrt(
    store_error["RMSE"]
)

store_error = store_error.sort_values(
    "MAE",
    ascending=False
)

print(
    store_error.to_string(),
    flush=True
)


# =========================================================
# ERROR BY DATE
# =========================================================

print("\n" + "=" * 60, flush=True)
print("ERROR BY DATE", flush=True)
print("=" * 60, flush=True)

print(
    ">>> Calculating date errors...",
    flush=True
)

date_error = (
    error_df
    .groupby("date")
    .agg(
        observations=("sales", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", "mean"),
        actual_total=("sales", "sum"),
        predicted_total=("predicted", "sum")
    )
)

date_error["RMSE"] = np.sqrt(
    date_error["RMSE"]
)

date_error = date_error.sort_values(
    "MAE",
    ascending=False
)

print(
    "\nWorst 15 dates:\n",
    flush=True
)

print(
    date_error.head(15).to_string(),
    flush=True
)


# =========================================================
# ERROR BY DEMAND MAGNITUDE
# =========================================================

print("\n" + "=" * 60, flush=True)
print("ERROR BY DEMAND MAGNITUDE", flush=True)
print("=" * 60, flush=True)

print(
    ">>> Creating demand groups...",
    flush=True
)

error_df["demand_group"] = pd.qcut(
    error_df["sales"],
    q=5,
    duplicates="drop"
)

demand_error = (
    error_df
    .groupby(
        "demand_group",
        observed=True
    )
    .agg(
        observations=("sales", "size"),
        actual_mean=("sales", "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", "mean")
    )
)

demand_error["RMSE"] = np.sqrt(
    demand_error["RMSE"]
)

print(
    demand_error.to_string(),
    flush=True
)


# =========================================================
# EXTREME ERROR ANALYSIS
# =========================================================

print("\n" + "=" * 60, flush=True)
print("EXTREME ERROR ANALYSIS", flush=True)
print("=" * 60, flush=True)

print(
    ">>> Calculating extreme-error contribution...",
    flush=True
)

total_squared_error = (
    error_df["squared_error"].sum()
)

sorted_errors = (
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
)

top_1_percent_count = max(
    1,
    int(len(error_df) * 0.01)
)

top_5_percent_count = max(
    1,
    int(len(error_df) * 0.05)
)

top_1_percent_contribution = (
    sorted_errors
    .head(top_1_percent_count)["squared_error"]
    .sum()
    / total_squared_error
)

top_5_percent_contribution = (
    sorted_errors
    .head(top_5_percent_count)["squared_error"]
    .sum()
    / total_squared_error
)

print(
    f"\nTop 1% of observations contribute "
    f"{top_1_percent_contribution:.2%} "
    f"of total squared error.",
    flush=True
)

print(
    f"Top 5% of observations contribute "
    f"{top_5_percent_contribution:.2%} "
    f"of total squared error.",
    flush=True
)


# =========================================================
# SAVE RESULTS
# =========================================================

print(
    "\n>>> Saving analysis results...",
    flush=True
)

importance.to_csv(
    "xgb_feature_importance.csv",
    index=False
)

error_df.to_csv(
    "xgb_validation_errors.csv",
    index=False
)

family_error.to_csv(
    "xgb_error_by_family.csv"
)

store_error.to_csv(
    "xgb_error_by_store.csv"
)

date_error.to_csv(
    "xgb_error_by_date.csv"
)

demand_error.to_csv(
    "xgb_error_by_demand.csv"
)

print(
    ">>> Results saved.",
    flush=True
)


# =========================================================
# COMPLETE
# =========================================================

total_time = time.time() - start_time

print("\n" + "=" * 60, flush=True)
print("ANALYSIS COMPLETE", flush=True)
print("=" * 60, flush=True)

print(
    f"\nTotal runtime: {total_time:.2f} seconds",
    flush=True
)

print("\nGenerated files:", flush=True)

print(
    "  xgb_feature_importance.csv",
    flush=True
)

print(
    "  xgb_feature_importance.png",
    flush=True
)

print(
    "  xgb_validation_errors.csv",
    flush=True
)

print(
    "  xgb_error_by_family.csv",
    flush=True
)

print(
    "  xgb_error_by_store.csv",
    flush=True
)

print(
    "  xgb_error_by_date.csv",
    flush=True
)

print(
    "  xgb_error_by_demand.csv",
    flush=True
)