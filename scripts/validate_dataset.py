import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"


def main():
    print("=" * 60)
    print("FOODFLOW DATASET VALIDATION")
    print("=" * 60)

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"]
    )

    # 1. Schema
    print("\n[1] Schema")
    print(train.dtypes)

    required_columns = {
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
    }

    missing_columns = required_columns - set(train.columns)

    if missing_columns:
        print(f"FAIL: Missing columns: {missing_columns}")
    else:
        print("PASS: Required columns present")

    # 2. Date range
    print("\n[2] Date coverage")
    print(f"Start: {train['date'].min().date()}")
    print(f"End:   {train['date'].max().date()}")
    print(f"Rows:  {len(train):,}")

    # 3. Duplicate observations
    print("\n[3] Duplicate observations")

    duplicates = train.duplicated(
        subset=["date", "store_nbr", "family"]
    ).sum()

    print(f"Duplicate date-store-family rows: {duplicates:,}")

    if duplicates == 0:
        print("PASS")
    else:
        print("FAIL")

    # 4. Missing values
    print("\n[4] Missing values")

    missing = train.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("PASS: No missing values")
    else:
        print(missing)

    # 5. Unique stores and product families
    print("\n[5] Entity coverage")

    print(f"Stores:   {train['store_nbr'].nunique()}")
    print(f"Families: {train['family'].nunique()}")

    # 6. Expected combinations
    print("\n[6] Store-family combinations")

    combinations = (
        train[["store_nbr", "family"]]
        .drop_duplicates()
    )

    print(f"Unique store-family combinations: {len(combinations):,}")

    # 7. Date continuity
    print("\n[7] Overall date continuity")

    expected_dates = pd.date_range(
        train["date"].min(),
        train["date"].max(),
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(
        train["date"].drop_duplicates()
    ).sort_values()

    missing_dates = expected_dates.difference(actual_dates)

    print(f"Expected dates: {len(expected_dates):,}")
    print(f"Actual dates:   {len(actual_dates):,}")
    print(f"Missing dates:  {len(missing_dates):,}")

    if len(missing_dates) == 0:
        print("PASS")
    else:
        print("WARNING: Missing dates detected")

    # 8. Target statistics
    print("\n[8] Target statistics")

    print(train["sales"].describe())

    negative_sales = (train["sales"] < 0).sum()

    print(f"Negative sales rows: {negative_sales:,}")

    if negative_sales == 0:
        print("PASS")
    else:
        print("WARNING")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()