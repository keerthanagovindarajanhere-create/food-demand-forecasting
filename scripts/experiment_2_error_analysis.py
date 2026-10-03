import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FOODFLOW — EXPERIMENT 2: XGBOOST ERROR ANALYSIS
# ============================================================
INPUT_FILE = "xgb_validation_errors.csv"
OUTPUT_DIR = Path("reports/experiments/experiment_2_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("FOODFLOW EXPERIMENT 2 — XGBOOST ERROR ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD EXISTING VALIDATION ERRORS
# ------------------------------------------------------------

print("\n[1] Loading validation errors...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# ------------------------------------------------------------
# 3. IDENTIFY IMPORTANT COLUMNS
# ------------------------------------------------------------

def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


date_col = find_column([
    "date",
    "ds",
    "datetime",
    "timestamp"
])

weekday_col = find_column([
    "weekday",
    "day_of_week",
    "dow"
])

promotion_col = find_column([
    "onpromotion",
    "on_promotion",
    "promotion",
    "promotions"
])

family_col = find_column([
    "family",
    "product_family"
])

store_col = find_column([
    "store_nbr",
    "store",
    "store_id"
])

actual_col = find_column([
    "actual",
    "y_true",
    "true",
    "target",
    "sales"
])

pred_col = find_column([
    "prediction",
    "predicted",
    "y_pred",
    "forecast"
])

error_col = find_column([
    "error",
    "residual"
])

abs_error_col = find_column([
    "absolute_error",
    "abs_error"
])


print("\nDetected columns:")
print(f"Date       : {date_col}")
print(f"Weekday    : {weekday_col}")
print(f"Promotion  : {promotion_col}")
print(f"Family     : {family_col}")
print(f"Store      : {store_col}")
print(f"Actual     : {actual_col}")
print(f"Prediction : {pred_col}")
print(f"Error      : {error_col}")
print(f"Abs Error  : {abs_error_col}")


# ------------------------------------------------------------
# 4. CREATE DATE / WEEKDAY IF NECESSARY
# ------------------------------------------------------------

if date_col is not None:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    if weekday_col is None:

        df["weekday"] = df[date_col].dt.day_name()
        weekday_col = "weekday"

        print("\nCreated weekday from date.")

# ------------------------------------------------------------
# 5. CREATE ERROR COLUMNS IF NECESSARY
# ------------------------------------------------------------

if error_col is None and actual_col is not None and pred_col is not None:

    df["error"] = df[pred_col] - df[actual_col]
    error_col = "error"

    print("\nCreated error column.")

if abs_error_col is None and error_col is not None:

    df["absolute_error"] = df[error_col].abs()
    abs_error_col = "absolute_error"

    print("Created absolute error column.")


# ------------------------------------------------------------
# 6. BASIC ERROR SUMMARY
# ------------------------------------------------------------

if abs_error_col is not None:

    print("\n" + "=" * 70)
    print("[2] OVERALL ERROR DISTRIBUTION")
    print("=" * 70)

    print(
        df[abs_error_col].describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    overall_summary = pd.DataFrame({
        "metric": [
            "mean_absolute_error",
            "median_absolute_error",
            "90th_percentile",
            "95th_percentile",
            "99th_percentile",
            "maximum_absolute_error"
        ],
        "value": [
            df[abs_error_col].mean(),
            df[abs_error_col].median(),
            df[abs_error_col].quantile(0.90),
            df[abs_error_col].quantile(0.95),
            df[abs_error_col].quantile(0.99),
            df[abs_error_col].max()
        ]
    })

    overall_summary.to_csv(
        OUTPUT_DIR / "overall_error_summary.csv",
        index=False
    )


# ------------------------------------------------------------
# 7. ERROR BY WEEKDAY
# ------------------------------------------------------------

if weekday_col is not None and abs_error_col is not None:

    print("\n" + "=" * 70)
    print("[3] ERROR BY WEEKDAY")
    print("=" * 70)

    weekday_analysis = (
        df.groupby(weekday_col)
        .agg(
            observations=(abs_error_col, "size"),
            mean_absolute_error=(abs_error_col, "mean"),
            median_absolute_error=(abs_error_col, "median"),
            p95_absolute_error=(
                abs_error_col,
                lambda x: x.quantile(0.95)
            ),
            max_absolute_error=(abs_error_col, "max")
        )
        .reset_index()
    )

    weekday_analysis = weekday_analysis.sort_values(
        "mean_absolute_error",
        ascending=False
    )

    print(weekday_analysis.to_string(index=False))

    weekday_analysis.to_csv(
        OUTPUT_DIR / "error_by_weekday.csv",
        index=False
    )


# ------------------------------------------------------------
# 8. ERROR BY PROMOTION
# ------------------------------------------------------------

if promotion_col is not None and abs_error_col is not None:

    print("\n" + "=" * 70)
    print("[4] ERROR BY PROMOTION")
    print("=" * 70)

    promotion_analysis = (
        df.groupby(promotion_col)
        .agg(
            observations=(abs_error_col, "size"),
            mean_absolute_error=(abs_error_col, "mean"),
            median_absolute_error=(abs_error_col, "median"),
            p95_absolute_error=(
                abs_error_col,
                lambda x: x.quantile(0.95)
            ),
            max_absolute_error=(abs_error_col, "max")
        )
        .reset_index()
    )

    promotion_analysis = promotion_analysis.sort_values(
        "mean_absolute_error",
        ascending=False
    )

    print(promotion_analysis.to_string(index=False))

    promotion_analysis.to_csv(
        OUTPUT_DIR / "error_by_promotion.csv",
        index=False
    )


# ------------------------------------------------------------
# 9. ERROR BY PRODUCT FAMILY
# ------------------------------------------------------------

if family_col is not None and abs_error_col is not None:

    print("\n" + "=" * 70)
    print("[5] ERROR BY PRODUCT FAMILY")
    print("=" * 70)

    family_analysis = (
        df.groupby(family_col)
        .agg(
            observations=(abs_error_col, "size"),
            mean_absolute_error=(abs_error_col, "mean"),
            median_absolute_error=(abs_error_col, "median"),
            p95_absolute_error=(
                abs_error_col,
                lambda x: x.quantile(0.95)
            ),
            max_absolute_error=(abs_error_col, "max")
        )
        .reset_index()
    )

    family_analysis = family_analysis.sort_values(
        "mean_absolute_error",
        ascending=False
    )

    print("\nTop 20 families by MAE:\n")

    print(
        family_analysis
        .head(20)
        .to_string(index=False)
    )

    family_analysis.to_csv(
        OUTPUT_DIR / "error_by_family.csv",
        index=False
    )


# ------------------------------------------------------------
# 10. ERROR BY STORE
# ------------------------------------------------------------

if store_col is not None and abs_error_col is not None:

    print("\n" + "=" * 70)
    print("[6] ERROR BY STORE")
    print("=" * 70)

    store_analysis = (
        df.groupby(store_col)
        .agg(
            observations=(abs_error_col, "size"),
            mean_absolute_error=(abs_error_col, "mean"),
            median_absolute_error=(abs_error_col, "median"),
            p95_absolute_error=(
                abs_error_col,
                lambda x: x.quantile(0.95)
            ),
            max_absolute_error=(abs_error_col, "max")
        )
        .reset_index()
    )

    store_analysis = store_analysis.sort_values(
        "mean_absolute_error",
        ascending=False
    )

    print("\nTop 20 stores by MAE:\n")

    print(
        store_analysis
        .head(20)
        .to_string(index=False)
    )

    store_analysis.to_csv(
        OUTPUT_DIR / "error_by_store.csv",
        index=False
    )


# ------------------------------------------------------------
# 11. TOP ERROR SPIKES
# ------------------------------------------------------------

if abs_error_col is not None:

    print("\n" + "=" * 70)
    print("[7] TOP 50 ERROR SPIKES")
    print("=" * 70)

    top_errors = (
        df.sort_values(
            abs_error_col,
            ascending=False
        )
        .head(50)
        .copy()
    )

    # Keep useful columns if they exist
    preferred_columns = [
        date_col,
        store_col,
        family_col,
        weekday_col,
        promotion_col,
        actual_col,
        pred_col,
        error_col,
        abs_error_col
    ]

    preferred_columns = [
        c for c in preferred_columns
        if c is not None and c in top_errors.columns
    ]

    # Add any remaining columns after important ones
    remaining_columns = [
        c for c in top_errors.columns
        if c not in preferred_columns
    ]

    top_errors = top_errors[
        preferred_columns + remaining_columns
    ]

    print(
        top_errors[
            preferred_columns
        ].to_string(index=False)
    )

    top_errors.to_csv(
        OUTPUT_DIR / "top_50_error_spikes.csv",
        index=False
    )


# ------------------------------------------------------------
# 12. HIGH-ERROR OBSERVATIONS
# ------------------------------------------------------------

if abs_error_col is not None:

    p95 = df[abs_error_col].quantile(0.95)
    p99 = df[abs_error_col].quantile(0.99)

    high_error_summary = pd.DataFrame({
        "threshold": [
            "95th_percentile",
            "99th_percentile"
        ],
        "absolute_error_threshold": [
            p95,
            p99
        ],
        "number_of_observations": [
            (df[abs_error_col] >= p95).sum(),
            (df[abs_error_col] >= p99).sum()
        ],
        "percentage_of_validation_data": [
            (df[abs_error_col] >= p95).mean() * 100,
            (df[abs_error_col] >= p99).mean() * 100
        ]
    })

    high_error_summary.to_csv(
        OUTPUT_DIR / "high_error_summary.csv",
        index=False
    )


# ------------------------------------------------------------
# 13. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)

print(f"""
Results saved to:

{OUTPUT_DIR.resolve()}

Generated files:
- overall_error_summary.csv
- error_by_weekday.csv
- error_by_promotion.csv
- error_by_family.csv
- error_by_store.csv
- top_50_error_spikes.csv
- high_error_summary.csv

No XGBoost model was retrained.
""")