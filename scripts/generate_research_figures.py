from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "research_analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DEMAND MAGNITUDE -> MAE / RMSE
# ============================================================

df = pd.read_csv(ROOT / "xgb_error_by_demand.csv")

x = np.arange(len(df))
labels = df["demand_group"].tolist()

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.36
ax.bar(x - width / 2, df["MAE"], width, label="MAE")
ax.bar(x + width / 2, df["RMSE"], width, label="RMSE")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_xlabel("Demand magnitude group")
ax.set_ylabel("Forecasting error")
ax.set_title("Forecasting Error Increases with Demand Magnitude")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "01_demand_magnitude_error.png", dpi=300)
plt.close()


# ============================================================
# 2. CUMULATIVE SQUARED-ERROR CONTRIBUTION
# ============================================================

df = pd.read_csv(ROOT / "xgb_validation_errors.csv")

sq = np.sort(df["squared_error"].to_numpy())[::-1]

cumulative = np.cumsum(sq)
cumulative_pct = cumulative / cumulative[-1] * 100
observation_pct = np.arange(1, len(sq) + 1) / len(sq) * 100

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(observation_pct, cumulative_pct, linewidth=2)

ax.axhline(80, linestyle="--", linewidth=1)
ax.axhline(90, linestyle="--", linewidth=1)

ax.set_xlabel("Largest-error observations included (%)")
ax.set_ylabel("Cumulative contribution to squared error (%)")
ax.set_title("Concentration of Total Squared Forecasting Error")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "02_cumulative_squared_error.png", dpi=300)
plt.close()


# ============================================================
# 3. FAMILY-LEVEL MAE
# ============================================================

df = pd.read_csv(ROOT / "xgb_error_by_family.csv")
df = df.sort_values("MAE", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))

ax.barh(df["family"], df["MAE"])

ax.set_xlabel("MAE")
ax.set_ylabel("Product family")
ax.set_title("Forecasting Error Across Product Families")
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "03_family_mae.png", dpi=300)
plt.close()


# ============================================================
# 4. EXTREME-ERROR TIMELINE
# ============================================================

df = pd.read_csv(ROOT / "xgb_extreme_error_dates.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    df["date"],
    df["abs_error"],
    marker="o",
    linewidth=1.5
)

ax.set_xlabel("Date")
ax.set_ylabel("Absolute forecasting error")
ax.set_title("Temporal Distribution of Extreme Forecasting Errors")
ax.grid(alpha=0.25)

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT / "04_extreme_error_timeline.png", dpi=300)
plt.close()


# ============================================================
# 5. REFERENCE VS TAIL-WEIGHTED XGBOOST
# ============================================================

df = pd.read_csv(
    ROOT / "reports" /
    "experiments" /
    "experiment_tail_weighted_xgb" /
    "reference_vs_tail.csv"
)

# Only MAE and RMSE are used for the primary model comparison.
plot_df = df[
    (df["metric"].isin(["MAE", "RMSE"])) &
    (df["regime"].isin(["overall", "high", "normal"]))
].copy()

regimes = ["overall", "normal", "high"]
metrics = ["MAE", "RMSE"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, metric in zip(axes, metrics):

    temp = plot_df[plot_df["metric"] == metric]

    ref = [
        temp.loc[temp["regime"] == r, "reference"].iloc[0]
        for r in regimes
    ]

    tail = [
        temp.loc[temp["regime"] == r, "tail_weighted"].iloc[0]
        for r in regimes
    ]

    x = np.arange(len(regimes))
    width = 0.35

    ax.bar(x - width / 2, ref, width, label="Reference")
    ax.bar(x + width / 2, tail, width, label="Tail-weighted")

    ax.set_xticks(x)
    ax.set_xticklabels(["Overall", "Normal", "High"])
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(axis="y", alpha=0.25)

    if metric == "MAE":
        ax.legend()

fig.suptitle(
    "Reference vs Tail-Weighted XGBoost",
    fontsize=14
)

plt.tight_layout()
plt.savefig(OUT / "05_reference_vs_tail_weighted.png", dpi=300)
plt.close()


print("\nGenerated five research figures:")
for path in sorted(OUT.glob("*.png")):
    print(path)