import os
import numpy as np
import pandas as pd

RAW = "data/raw"

TRAIN = os.path.join(RAW, "train.csv")
TEST = os.path.join(RAW, "test.csv")
STORES = os.path.join(RAW, "stores.csv")
TRANSACTIONS = os.path.join(RAW, "transactions.csv")
OIL = os.path.join(RAW, "oil.csv")
HOLIDAYS = os.path.join(RAW, "holidays_events.csv")

CHUNK_SIZE = 200_000


def inspect_file(path):
    print("\n" + "=" * 70)
    print(os.path.basename(path))
    print("=" * 70)

    df = pd.read_csv(path, nrows=5)

    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nSample:")
    print(df.head().to_string(index=False))

    return df.columns.tolist()


# ============================================================
# TRAIN PROFILE
# ============================================================

print("\n" + "=" * 70)
print("TRAIN DATASET PROFILE")
print("=" * 70)

train = pd.read_csv(
    TRAIN,
    parse_dates=["date"]
)

print(f"\nRows: {len(train):,}")
print(f"Columns: {len(train.columns)}")

print("\nSchema:")
print(train.dtypes.to_string())

print("\nMissing values:")
missing = train.isna().sum()

for col in train.columns:
    pct = missing[col] / len(train) * 100
    print(f"  {col:15s}: {missing[col]:12,} ({pct:.4f}%)")

print("\nTemporal coverage:")
print(f"  Start: {train['date'].min().date()}")
print(f"  End:   {train['date'].max().date()}")

print("\nCardinality:")
print(f"  Stores:              {train['store_nbr'].nunique():,}")
print(f"  Product families:    {train['family'].nunique():,}")
print(
    f"  Store-family series: "
    f"{train.groupby(['store_nbr', 'family']).ngroups:,}"
)

# ============================================================
# DEMAND
# ============================================================

sales = train["sales"]

print("\n" + "=" * 70)
print("DEMAND DISTRIBUTION")
print("=" * 70)

quantiles = sales.quantile(
    [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]
)

for q, value in quantiles.items():
    print(f"  Q{q:g}: {value:,.4f}")

zero_count = (sales == 0).sum()
negative_count = (sales < 0).sum()

print(f"\nZero-demand observations: {zero_count:,}")
print(f"Zero-demand percentage:   {zero_count / len(sales) * 100:.2f}%")
print(f"Negative observations:    {negative_count:,}")

q1 = quantiles.loc[0.25]
q3 = quantiles.loc[0.75]
iqr = q3 - q1
upper_threshold = q3 + 1.5 * iqr

outliers = sales > upper_threshold

print("\nIQR outlier analysis:")
print(f"  Q1:                 {q1:,.4f}")
print(f"  Q3:                 {q3:,.4f}")
print(f"  IQR:                {iqr:,.4f}")
print(f"  Upper threshold:    {upper_threshold:,.4f}")
print(f"  High-sales rows:    {outliers.sum():,}")
print(
    f"  High-sales %:       "
    f"{outliers.mean() * 100:.2f}%"
)

# ============================================================
# PROMOTIONS
# ============================================================

promotion = train["onpromotion"]

print("\n" + "=" * 70)
print("PROMOTION PROFILE")
print("=" * 70)

positive_promo = promotion > 0

print(f"Rows with promotion:      {positive_promo.sum():,}")
print(
    f"Promotion coverage:       "
    f"{positive_promo.mean() * 100:.2f}%"
)
print(f"Mean promoted items:      {promotion.mean():.4f}")
print(f"Maximum promoted items:   {promotion.max():,}")

print("\nPromotion quantiles:")
print(
    promotion.quantile(
        [0, .25, .50, .75, .90, .95, .99, 1]
    ).to_string()
)

# ============================================================
# SERIES COMPLETENESS
# ============================================================

print("\n" + "=" * 70)
print("SERIES COMPLETENESS")
print("=" * 70)

expected_days = (
    train["date"].max() - train["date"].min()
).days + 1

series_counts = (
    train.groupby(["store_nbr", "family"])
    .size()
)

print(f"Expected calendar days:       {expected_days:,}")
print(f"Total series:                  {len(series_counts):,}")

print("\nObservations per series:")
print(series_counts.describe().to_string())

complete_series = (series_counts == expected_days).sum()

print(
    f"\nSeries with complete daily coverage: "
    f"{complete_series:,} / {len(series_counts):,}"
)

print(
    f"Series with incomplete coverage: "
    f"{(series_counts < expected_days).sum():,}"
)

# ============================================================
# DATE COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("DAILY COVERAGE")
print("=" * 70)

daily_counts = train.groupby("date").size()

expected_rows_per_day = 54 * 33

print(f"Expected rows per complete day: {expected_rows_per_day:,}")

print("\nRows per day:")
print(daily_counts.describe().to_string())

incomplete_days = (
    daily_counts[daily_counts != expected_rows_per_day]
)

print(
    f"\nDays with incomplete store-family coverage: "
    f"{len(incomplete_days):,}"
)

if len(incomplete_days) > 0:
    print("\nFirst incomplete dates:")
    print(incomplete_days.head(20).to_string())

# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_keys = train.duplicated(
    subset=["date", "store_nbr", "family"]
).sum()

duplicate_ids = train["id"].duplicated().sum()

print(
    f"Duplicate date-store-family keys: {duplicate_keys:,}"
)
print(f"Duplicate IDs:                    {duplicate_ids:,}")

# ============================================================
# SUPPORTING DATASETS
# ============================================================

for path in [
    STORES,
    TRANSACTIONS,
    OIL,
    HOLIDAYS,
    TEST
]:
    inspect_file(path)

# ============================================================
# SUPPORTING DATASET STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SUPPORTING DATASET STATISTICS")
print("=" * 70)

# Stores
stores = pd.read_csv(STORES)

print("\nSTORES")
print(f"Rows: {len(stores):,}")
print(f"Columns: {len(stores.columns):,}")
print(f"Missing values:\n{stores.isna().sum().to_string()}")

# Transactions
transactions = pd.read_csv(
    TRANSACTIONS,
    parse_dates=["date"]
)

print("\nTRANSACTIONS")
print(f"Rows: {len(transactions):,}")
print(
    f"Date range: "
    f"{transactions['date'].min().date()} -> "
    f"{transactions['date'].max().date()}"
)
print(
    f"Stores covered: "
    f"{transactions['store_nbr'].nunique():,}"
)
print("\nTransaction statistics:")
print(transactions["transactions"].describe().to_string())

print(
    f"\nMissing transaction values: "
    f"{transactions['transactions'].isna().sum():,}"
)

# Oil
oil = pd.read_csv(
    OIL,
    parse_dates=["date"]
)

print("\nOIL")
print(f"Rows: {len(oil):,}")
print(
    f"Date range: "
    f"{oil['date'].min().date()} -> "
    f"{oil['date'].max().date()}"
)
print("\nMissing values:")
print(oil.isna().sum().to_string())

print("\nOil statistics:")
print(oil["dcoilwtico"].describe().to_string())

# Holidays
holidays = pd.read_csv(HOLIDAYS)

print("\nHOLIDAYS / EVENTS")
print(f"Rows: {len(holidays):,}")
print(f"Columns: {len(holidays.columns):,}")

print("\nMissing values:")
print(holidays.isna().sum().to_string())

for column in holidays.columns:
    if holidays[column].dtype == "object":
        print(f"\n{column} unique values:")
        print(holidays[column].value_counts().head(15).to_string())

# ============================================================
# TEST HORIZON
# ============================================================

test = pd.read_csv(
    TEST,
    parse_dates=["date"]
)

print("\n" + "=" * 70)
print("TEST SET")
print("=" * 70)

print(f"Rows: {len(test):,}")
print(
    f"Date range: "
    f"{test['date'].min().date()} -> "
    f"{test['date'].max().date()}"
)

print(
    f"Stores: {test['store_nbr'].nunique():,}"
)

print(
    f"Families: {test['family'].nunique():,}"
)

print(
    f"Store-family series: "
    f"{test.groupby(['store_nbr', 'family']).ngroups:,}"
)

print("\n" + "=" * 70)
print("PROFILE COMPLETE")
print("=" * 70)