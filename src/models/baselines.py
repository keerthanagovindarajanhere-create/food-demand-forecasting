import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


TRAIN_END = "2017-07-18"
VAL_START = "2017-07-19"
VAL_END = "2017-08-15"


def rmsle(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            np.log1p(y_true),
            np.log1p(np.maximum(y_pred, 0))
        )
    )


# Load data
df = pd.read_csv(
    "data/raw/train.csv",
    parse_dates=["date"]
)

df = df.sort_values(["store_nbr", "family", "date"])


# Split
train = df[df["date"] <= TRAIN_END].copy()
val = df[
    (df["date"] >= VAL_START) &
    (df["date"] <= VAL_END)
].copy()


print("=" * 60)
print("FOODFLOW BASELINE EXPERIMENT")
print("=" * 60)

print(f"Training:   {train['date'].min().date()} -> {train['date'].max().date()}")
print(f"Validation: {val['date'].min().date()} -> {val['date'].max().date()}")
print(f"Validation days: {val['date'].nunique()}")


# Combine train + validation so lag features can be generated
combined = pd.concat(
    [train[["date", "store_nbr", "family", "sales"]],
     val[["date", "store_nbr", "family", "sales"]]],
    ignore_index=True
)

combined = combined.sort_values(
    ["store_nbr", "family", "date"]
)


group_cols = ["store_nbr", "family"]


# ---------------------------------------------------------
# 1. Naive
# ---------------------------------------------------------

combined["naive_pred"] = (
    combined.groupby(group_cols)["sales"]
    .shift(1)
)


# ---------------------------------------------------------
# 2. Seasonal Naive (7 days)
# ---------------------------------------------------------

combined["seasonal_naive_pred"] = (
    combined.groupby(group_cols)["sales"]
    .shift(7)
)


# ---------------------------------------------------------
# 3. 7-Day Moving Average
# ---------------------------------------------------------

combined["moving_average_pred"] = (
    combined.groupby(group_cols)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)


# Select validation period
val_pred = combined[
    (combined["date"] >= VAL_START) &
    (combined["date"] <= VAL_END)
].copy()


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

models = {
    "Naive": "naive_pred",
    "Seasonal Naive (7d)": "seasonal_naive_pred",
    "7-Day Moving Average": "moving_average_pred",
}


results = []

for model_name, pred_col in models.items():

    valid = val_pred.dropna(subset=[pred_col]).copy()

    y_true = valid["sales"]
    y_pred = valid[pred_col]

    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "RMSLE": rmsle(y_true, y_pred),
        "Observations": len(valid),
    })


results_df = pd.DataFrame(results)


print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Save results
results_df.to_csv(
    "reports/baseline_results.csv",
    index=False
)

print("\nSaved: reports/baseline_results.csv")