import pandas as pd

from features import create_features


TRAIN_END = "2017-07-18"
VAL_START = "2017-07-19"
VAL_END = "2017-08-15"


FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "onpromotion",
]


def prepare_data():

    # Load dataset
    df = pd.read_csv(
        "data/raw/train.csv",
        parse_dates=["date"]
    )

    # Create features
    df = create_features(df)

    # Remove rows without sufficient historical information
    df = df.dropna(
        subset=[
            "lag_28",
            "rolling_mean_28"
        ]
    )

    # Temporal split
    train = df[
        df["date"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["date"] >= VAL_START) &
        (df["date"] <= VAL_END)
    ].copy()

    # One-hot encode food family
    train = pd.get_dummies(
        train,
        columns=["family"],
        dtype=int
    )

    validation = pd.get_dummies(
        validation,
        columns=["family"],
        dtype=int
    )

    # Align validation columns with training columns
    train, validation = train.align(
        validation,
        join="left",
        axis=1,
        fill_value=0
    )

    # Identify family columns
    family_cols = [
        col for col in train.columns
        if col.startswith("family_")
    ]

    final_features = FEATURE_COLS + [
        "store_nbr"
    ] + family_cols

    X_train = train[final_features]
    y_train = train["sales"]

    X_val = validation[final_features]
    y_val = validation["sales"]

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":

    X_train, y_train, X_val, y_val = prepare_data()

    print("=" * 60)
    print("FOODFLOW XGBOOST DATASET")
    print("=" * 60)

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nValidation shape:")
    print(X_val.shape)

    print("\nNumber of features:")
    print(X_train.shape[1])

    print("\nTraining target:")
    print(y_train.shape)

    print("\nValidation target:")
    print(y_val.shape)

    print("\nMissing values in training features:")
    print(X_train.isna().sum().sum())

    print("\nMissing values in validation features:")
    print(X_val.isna().sum().sum())

    print("\nTraining date range:")
    print(
        train_date := (
            pd.read_csv(
                "data/raw/train.csv",
                usecols=["date"],
                parse_dates=["date"]
            )["date"].min()
        )
    )