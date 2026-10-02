import pandas as pd

TRAIN_PATH = "data/raw/train.csv"

print("=" * 60)
print("FOODFLOW PANEL COMPLETENESS ANALYSIS")
print("=" * 60)

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["date", "store_nbr", "family"],
    parse_dates=["date"]
)

expected_dates = pd.date_range(
    train["date"].min(),
    train["date"].max(),
    freq="D"
)

known_holiday_closures = pd.to_datetime([
    "2013-12-25",
    "2014-12-25",
    "2015-12-25",
    "2016-12-25"
])

expected_observed_dates = expected_dates.difference(
    known_holiday_closures
)

expected_days = len(expected_observed_dates)

print(f"\nCalendar days: {len(expected_dates)}")
print(f"Known holiday closures: {len(known_holiday_closures)}")
print(f"Expected observed days: {expected_days}")
print(f"Actual observed days: {train['date'].nunique()}")

series_counts = (
    train.groupby(["store_nbr", "family"])["date"]
    .nunique()
    .reset_index(name="observed_days")
)

series_counts["missing_unexpected_days"] = (
    expected_days - series_counts["observed_days"]
)

print("\n[1] Series coverage")
print(f"Total series: {len(series_counts)}")

complete = (
    series_counts["missing_unexpected_days"] == 0
).sum()

incomplete = (
    series_counts["missing_unexpected_days"] > 0
).sum()

print(f"Complete after accounting for known closures: {complete}")
print(f"Unexpectedly incomplete: {incomplete}")

print("\n[2] Unexpected missing-day distribution")

print(
    series_counts["missing_unexpected_days"]
    .value_counts()
    .sort_index()
)

print("\n[3] Unexpected gaps")

gaps = []

for (store, family), group in train.groupby(
    ["store_nbr", "family"]
):
    observed = pd.DatetimeIndex(group["date"].unique())

    missing = expected_observed_dates.difference(observed)

    for date in missing:
        gaps.append({
            "store_nbr": store,
            "family": family,
            "date": date
        })

gaps_df = pd.DataFrame(gaps)

print(
    f"Total unexpected missing "
    f"series-date observations: {len(gaps_df)}"
)

if not gaps_df.empty:
    print("\nUnexpected missing dates:")
    print(
        gaps_df["date"]
        .value_counts()
        .sort_index()
        .to_string()
    )

print("\n" + "=" * 60)
print("PANEL ANALYSIS COMPLETE")
print("=" * 60)