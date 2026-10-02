import pandas as pd

TRAIN_PATH = "data/raw/train.csv"

VALIDATION_DAYS = 28

print("=" * 60)
print("FOODFLOW TEMPORAL SPLIT ANALYSIS")
print("=" * 60)

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["date", "store_nbr", "family", "sales"],
    parse_dates=["date"]
)

dates = pd.DatetimeIndex(
    train["date"].unique()
).sort_values()

validation_dates = dates[-VALIDATION_DAYS:]
training_dates = dates[:-VALIDATION_DAYS]

print("\n[1] Dataset")
print(f"First date: {dates.min().date()}")
print(f"Last date:  {dates.max().date()}")
print(f"Observed dates: {len(dates)}")

print("\n[2] Training period")
print(f"Start: {training_dates.min().date()}")
print(f"End:   {training_dates.max().date()}")
print(f"Observed days: {len(training_dates)}")

print("\n[3] Validation period")
print(f"Start: {validation_dates.min().date()}")
print(f"End:   {validation_dates.max().date()}")
print(f"Observed days: {len(validation_dates)}")

print("\n[4] Row counts")

training = train[train["date"].isin(training_dates)]
validation = train[train["date"].isin(validation_dates)]

print(f"Training rows:   {len(training):,}")
print(f"Validation rows: {len(validation):,}")

print("\n[5] Series coverage")

train_series = training.groupby(
    ["store_nbr", "family"]
).size()

validation_series = validation.groupby(
    ["store_nbr", "family"]
).size()

print(f"Training series:   {len(train_series):,}")
print(f"Validation series: {len(validation_series):,}")

print("\n[6] Validation observations per series")

print(
    validation_series.describe()
)

print("\n" + "=" * 60)
print("TEMPORAL SPLIT ANALYSIS COMPLETE")
print("=" * 60)