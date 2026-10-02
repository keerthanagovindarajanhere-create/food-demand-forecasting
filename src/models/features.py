import pandas as pd


def create_features(df):
    """
    Create historical, rolling, calendar, and promotion features.

    All lag and rolling features use only observations
    available before the prediction date.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["store_nbr", "family", "date"]
    )

    group_cols = ["store_nbr", "family"]

    # -----------------------------
    # Lag features
    # -----------------------------

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            df.groupby(group_cols)["sales"]
            .shift(lag)
        )

    # -----------------------------
    # Rolling demand features
    # -----------------------------

    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_cols)["sales"]
            .transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
        )

    # -----------------------------
    # Calendar features
    # -----------------------------

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # -----------------------------
    # Promotion feature
    # -----------------------------

    df["onpromotion"] = df["onpromotion"].fillna(0)

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/train.csv",
        parse_dates=["date"]
    )

    df = create_features(df)

    print("=" * 60)
    print("FOODFLOW FEATURE CHECK")
    print("=" * 60)

    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "onpromotion",
    ]

    print("\nFeature columns:")
    print(feature_cols)

    print("\nMissing values:")
    print(df[feature_cols].isna().sum())

    print("\nSample:")
    print(
        df[
            [
                "date",
                "store_nbr",
                "family",
                "sales",
                *feature_cols,
            ]
        ].tail(10)
    )