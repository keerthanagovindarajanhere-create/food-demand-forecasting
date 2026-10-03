import time
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "models")
    )
)

from prepare_xgb_data import prepare_data


# =========================================================
# CONFIGURATION
# =========================================================

HIGH_DEMAND_QUANTILE = 0.80
HIGH_DEMAND_WEIGHT = 2.0


# Same configuration as reference XGBoost model
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(y_true, y_pred):

    errors = y_true - y_pred
    absolute_errors = np.abs(errors)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    bias = np.mean(errors)

    underprediction_rate = np.mean(
        y_pred < y_true
    )

    overprediction_rate = np.mean(
        y_pred > y_true
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Underprediction_Rate": underprediction_rate,
        "Overprediction_Rate": overprediction_rate,
    }


# =========================================================
# START
# =========================================================

start_time = time.time()

print("=" * 70)
print("FOODFLOW TAIL-WEIGHTED XGBOOST EXPERIMENT")
print("=" * 70)


# =========================================================
# PREPARE DATA
# =========================================================

print("\n>>> Preparing data...")

X_train, y_train, X_val, y_val = prepare_data()

print(f">>> Training shape:   {X_train.shape}")
print(f">>> Validation shape: {X_val.shape}")


# =========================================================
# DEFINE HIGH-DEMAND THRESHOLD
# TRAINING DATA ONLY — NO VALIDATION LEAKAGE
# =========================================================

high_threshold = y_train.quantile(
    HIGH_DEMAND_QUANTILE
)

print("\n" + "=" * 70)
print("TAIL DEFINITION")
print("=" * 70)

print(
    f"High-demand threshold "
    f"(training 80th percentile): {high_threshold:.4f}"
)

high_train = y_train >= high_threshold

print(
    f"High-demand training observations: "
    f"{high_train.sum():,} "
    f"({high_train.mean():.2%})"
)


# =========================================================
# SAMPLE WEIGHTS
# =========================================================

sample_weight = np.where(
    high_train,
    HIGH_DEMAND_WEIGHT,
    1.0
)

print(
    f"Normal-demand weight: 1.0\n"
    f"High-demand weight:   {HIGH_DEMAND_WEIGHT}"
)


# =========================================================
# MODEL
# =========================================================

print("\n>>> Creating XGBoost model...")

model = XGBRegressor(**XGB_PARAMS)


# =========================================================
# TRAIN
# =========================================================

print("\n>>> STARTING TAIL-WEIGHTED TRAINING")

train_start = time.time()

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
    verbose=False
)

train_time = time.time() - train_start

print(
    f">>> Training complete: {train_time:.2f} seconds"
)


# =========================================================
# PREDICTIONS
# =========================================================

print("\n>>> Generating validation predictions...")

predictions = model.predict(X_val)

predictions = np.maximum(
    predictions,
    0
)


# =========================================================
# VALIDATION RESULTS
# =========================================================

results = pd.DataFrame({
    "actual": y_val.reset_index(drop=True),
    "predicted": predictions
})

results["error"] = (
    results["actual"] -
    results["predicted"]
)

results["absolute_error"] = (
    results["error"].abs()
)


# =========================================================
# VALIDATION DEMAND REGIMES
# =========================================================
#
# IMPORTANT:
# Use the TRAINING threshold to define validation
# high-demand observations.
#
# This avoids using validation distribution to define
# the regime.
# =========================================================

results["demand_regime"] = np.select(
    [
        results["actual"] < high_threshold,
        results["actual"] >= high_threshold,
    ],
    [
        "normal",
        "high",
    ],
    default="normal"
)


# =========================================================
# OVERALL METRICS
# =========================================================

overall_metrics = calculate_metrics(
    results["actual"],
    results["predicted"]
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

for metric, value in overall_metrics.items():
    print(f"{metric}: {value:.6f}")


# =========================================================
# REGIME METRICS
# =========================================================

regime_rows = []

for regime in ["normal", "high"]:

    subset = results[
        results["demand_regime"] == regime
    ]

    metrics = calculate_metrics(
        subset["actual"],
        subset["predicted"]
    )

    row = {
        "regime": regime,
        "observations": len(subset),
        "actual_mean": subset["actual"].mean(),
        **metrics
    }

    regime_rows.append(row)


regime_metrics = pd.DataFrame(
    regime_rows
)


print("\n" + "=" * 70)
print("DEMAND REGIME RESULTS")
print("=" * 70)

print(
    regime_metrics.to_string(index=False)
)


# =========================================================
# SAVE RESULTS
# =========================================================

output_dir = (
    "reports/experiments/"
    "experiment_tail_weighted_xgb"
)

import os

os.makedirs(
    output_dir,
    exist_ok=True
)


regime_metrics.to_csv(
    f"{output_dir}/regime_metrics.csv",
    index=False
)

pd.DataFrame([
    overall_metrics
]).to_csv(
    f"{output_dir}/overall_metrics.csv",
    index=False
)

results.to_csv(
    f"{output_dir}/validation_predictions.csv",
    index=False
)


# =========================================================
# EXPERIMENT SUMMARY
# =========================================================

with open(
    f"{output_dir}/experiment_summary.md",
    "w",
    encoding="utf-8"
) as f:

    f.write("# Tail-Weighted XGBoost Experiment\n\n")

    f.write(
        "## Hypothesis\n\n"
        "Because large forecasting errors are concentrated in "
        "high-demand observations, increasing the training "
        "importance of high-demand observations may improve "
        "high-demand forecasting performance.\n\n"
    )

    f.write(
        "## Controlled Setup\n\n"
        "- Same dataset\n"
        "- Same temporal split\n"
        "- Same features\n"
        "- Same XGBoost hyperparameters\n"
        "- Same random seed\n"
        "- Only training sample weights were changed\n"
        "- High demand defined using the training 80th percentile\n"
        "- High-demand observations receive weight 2.0\n\n"
    )

    f.write(
        f"Training high-demand threshold: "
        f"{high_threshold:.4f}\n\n"
    )

    f.write("## Overall Results\n\n")

    for metric, value in overall_metrics.items():
        f.write(
            f"- {metric}: {value:.6f}\n"
        )

    f.write("\n## Regime Results\n\n")

    f.write(
        regime_metrics.to_string(index=False)
    )

    f.write("\n")


# =========================================================
# COMPLETE
# =========================================================

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"\nTotal runtime: {total_time:.2f} seconds"
)

print("\nGenerated files:")

print(
    f"  {output_dir}/regime_metrics.csv"
)

print(
    f"  {output_dir}/overall_metrics.csv"
)

print(
    f"  {output_dir}/validation_predictions.csv"
)

print(
    f"  {output_dir}/experiment_summary.md"
)