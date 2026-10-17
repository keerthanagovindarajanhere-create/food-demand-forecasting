"""
FOODFLOW core logic (no Streamlit dependency, so it can be unit-tested).

Contents
--------
1. Paths and artifact loading
2. History source for forecasting
3. Forecasting with the frozen reference XGBoost model
4. Metrics helpers
5. Preparation-planning helpers

Forecasting design
------------------
The frozen model was trained on lag / rolling features built by
`src/models/features.py::create_features`. That function uses row-based
`shift`, so the history passed to it must contain every consecutive day
for a series. We therefore:

* use the real observed sales up to 2017-08-15 (the last training date),
* forecast 2017-08-16 directly (all features come from real data), and
* forecast later days recursively, feeding each prediction back in as the
  next day's "sales". Those later forecasts are clearly labelled as
  recursive, because their lag features partly depend on predictions.

The history source is `data/raw/train.csv` when it is available locally.
Otherwise the app falls back to the actual sales stored in
`reports/research_analysis/results/xgb_validation_errors.csv`, which covers
every store-family series from 2017-07-19 to 2017-08-15 (28 consecutive
days). 28 days is exactly what the longest feature (lag_28 / 28-day rolling
mean) needs, so the next-day forecast is computed from the same inputs the
full dataset would give.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts" / "model" / "xgb_reference.pkl"
FEATURE_PATH = ROOT / "artifacts" / "model" / "feature_columns.json"
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

RESULTS = ROOT / "reports" / "research_analysis" / "results"
FIGURES = ROOT / "reports" / "research_analysis" / "figures"
TAIL_DIR = ROOT / "reports" / "experiments" / "experiment_tail_weighted_xgb"

VALIDATION_ERRORS_PATH = RESULTS / "xgb_validation_errors.csv"
BASELINE_PATH = ROOT / "reports" / "baseline_results.csv"

LAST_OBSERVED_DATE = pd.Timestamp("2017-08-15")
VAL_START = pd.Timestamp("2017-07-19")
MAX_HORIZON_DAYS = 16  # same length as the Kaggle test window (Aug 16-31)
HISTORY_DAYS_NEEDED = 28

SRC_MODELS = ROOT / "src" / "models"
if str(SRC_MODELS) not in sys.path:
    sys.path.insert(0, str(SRC_MODELS))

from features import create_features  # noqa: E402  (same function used in training)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_feature_columns() -> list[str]:
    with open(FEATURE_PATH, "r") as f:
        return json.load(f)


def load_validation_errors() -> pd.DataFrame:
    """Per-observation validation results of the reference model."""
    return pd.read_csv(VALIDATION_ERRORS_PATH, parse_dates=["date"])


def load_tail_predictions() -> pd.DataFrame:
    """Tail-weighted model predictions (same row order as validation errors)."""
    return pd.read_csv(TAIL_DIR / "validation_predictions.csv")


# ============================================================
# 2. HISTORY SOURCE
# ============================================================

def load_history() -> tuple[pd.DataFrame, str]:
    """
    Return (history, source_label) with columns
    date, store_nbr, family, sales, onpromotion (onpromotion may be NaN).
    """
    if TRAIN_PATH.exists():
        df = pd.read_csv(
            TRAIN_PATH,
            usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
            parse_dates=["date"],
        )
        # Keep a recent window only: enough for features + a readable chart.
        df = df[df["date"] >= LAST_OBSERVED_DATE - pd.Timedelta(days=89)]
        return df.reset_index(drop=True), "data/raw/train.csv (last 90 days)"

    v = load_validation_errors()
    hist = v[["date", "store_nbr", "family", "sales"]].copy()
    hist["onpromotion"] = np.nan  # not stored in the validation artifact
    return hist, "validation actuals (2017-07-19 to 2017-08-15)"


def series_history(history: pd.DataFrame, store_nbr: int, family: str) -> pd.DataFrame:
    s = history[(history["store_nbr"] == store_nbr) & (history["family"] == family)]
    return s.sort_values("date").reset_index(drop=True)


# ============================================================
# 3. FORECASTING
# ============================================================

def _feature_row(series: pd.DataFrame, target_date, store_nbr, family, onpromotion,
                 feature_columns):
    """Build the single model-input row for `target_date` using create_features."""
    future = pd.DataFrame({
        "date": [pd.Timestamp(target_date)],
        "store_nbr": [store_nbr],
        "family": [family],
        "sales": [np.nan],
        "onpromotion": [float(onpromotion)],
    })
    base = series[["date", "store_nbr", "family", "sales", "onpromotion"]]
    combined = pd.concat([base, future], ignore_index=True)
    feats = create_features(combined)
    row = feats[feats["date"] == pd.Timestamp(target_date)].copy()

    # One-hot family exactly as in training (get_dummies on 'family').
    for col in feature_columns:
        if col.startswith("family_"):
            row[col] = int(col == f"family_{family}")
    missing = [c for c in feature_columns if c not in row.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    return row[feature_columns].astype(float)


def check_series_history(series: pd.DataFrame) -> None:
    """Raise ValueError if the history cannot support valid lag features."""
    if series.empty:
        raise ValueError("No history found for this store and product family.")
    if series["date"].max() != LAST_OBSERVED_DATE:
        raise ValueError("History does not end on the last observed date (2017-08-15).")
    tail = series.tail(HISTORY_DAYS_NEEDED)
    expected = pd.date_range(end=LAST_OBSERVED_DATE, periods=HISTORY_DAYS_NEEDED)
    if len(tail) < HISTORY_DAYS_NEEDED or [pd.Timestamp(d) for d in tail["date"]] != list(expected):
        raise ValueError(
            "The last 28 days of history are incomplete, so lag features would be "
            "misaligned. Forecast not generated."
        )


def forecast_series(model, feature_columns, series: pd.DataFrame, store_nbr: int,
                    family: str, horizon_days: int, onpromotion: float) -> pd.DataFrame:
    """
    Forecast day 1..horizon_days after 2017-08-15.

    Day 1 is a direct one-step forecast (every feature from real data).
    Days 2+ are recursive: the previous predictions are used as lag values.
    `onpromotion` is applied to every forecast day.
    """
    if not 1 <= horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(f"Horizon must be between 1 and {MAX_HORIZON_DAYS} days.")
    if onpromotion < 0:
        raise ValueError("Promotion count cannot be negative.")
    check_series_history(series)

    work = series[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    work = work.tail(60)  # plenty for 28-day features; keeps it fast
    rows = []
    for step in range(1, horizon_days + 1):
        target = LAST_OBSERVED_DATE + pd.Timedelta(days=step)
        X = _feature_row(work, target, store_nbr, family, onpromotion, feature_columns)
        if X.isna().any().any():
            raise ValueError(f"Model input for {target.date()} contains missing values.")
        pred = max(0.0, float(model.predict(X)[0]))  # training also clipped at 0
        rows.append({
            "date": target,
            "step": step,
            "prediction": pred,
            "method": "direct (real history)" if step == 1 else "recursive",
            "real_lag_share": _real_lag_share(step),
        })
        work = pd.concat([work, pd.DataFrame({
            "date": [target], "store_nbr": [store_nbr], "family": [family],
            "sales": [pred], "onpromotion": [float(onpromotion)],
        })], ignore_index=True)
    return pd.DataFrame(rows)


def _real_lag_share(step: int) -> float:
    """Share of the 7 lag/rolling inputs that use only real (not predicted) sales."""
    real = 0
    for lag in (1, 7, 14, 28):
        real += lag >= step
    for window in (7, 14, 28):
        real += step == 1  # any rolling window after day 1 includes a prediction
    return real / 7


def feature_vector(series, store_nbr, family, onpromotion, feature_columns):
    """Model input for the direct next-day forecast (for display)."""
    check_series_history(series)
    target = LAST_OBSERVED_DATE + pd.Timedelta(days=1)
    return _feature_row(series.tail(60), target, store_nbr, family, onpromotion,
                        feature_columns)


def promotion_sensitivity(model, feature_columns, series, store_nbr, family,
                          promo_values=(0, 1, 5, 10, 25, 50)) -> pd.DataFrame:
    """Next-day prediction for several promotion counts (everything else fixed)."""
    X0 = feature_vector(series, store_nbr, family, 0, feature_columns)
    out = []
    for p in promo_values:
        X = X0.copy()
        X["onpromotion"] = float(p)
        out.append({"onpromotion": p, "prediction": max(0.0, float(model.predict(X)[0]))})
    return pd.DataFrame(out)


# ============================================================
# 4. METRICS
# ============================================================

def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def rmsle(y, p):
    y = np.maximum(np.asarray(y, dtype=float), 0)
    p = np.maximum(np.asarray(p, dtype=float), 0)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def cumulative_squared_error_share(actual, predicted) -> pd.DataFrame:
    """Curve of 'top x% of observations' -> share of total squared error."""
    se = np.sort((np.asarray(actual) - np.asarray(predicted)) ** 2)[::-1]
    share = np.cumsum(se) / se.sum()
    n = len(se)
    return pd.DataFrame({
        "obs_pct": np.arange(1, n + 1) / n * 100,
        "error_share_pct": share * 100,
    })


def share_at(curve: pd.DataFrame, pct: float) -> float:
    n = len(curve)
    idx = int(np.ceil(pct / 100 * n)) - 1
    return float(curve["error_share_pct"].iloc[idx])


# ============================================================
# 5. PREPARATION PLANNING  (illustrative simulation)
# ============================================================

def buffered_quantity(forecast: float, buffer_pct: float) -> float:
    """Preparation quantity = forecast x (1 + buffer%)."""
    return max(0.0, forecast * (1 + buffer_pct / 100))


def critical_ratio(shortage_cost: float, waste_cost: float) -> float:
    """Newsvendor critical ratio Cu / (Cu + Co)."""
    if shortage_cost <= 0 or waste_cost <= 0:
        raise ValueError("Costs must be positive.")
    return shortage_cost / (shortage_cost + waste_cost)


def simulate_outcomes(quantity: float, forecast: float, residuals,
                      shortage_cost: float, waste_cost: float) -> dict:
    """
    Simulate demand as forecast + historical residual (actual - predicted),
    floored at 0, and evaluate a preparation quantity against it.
    """
    residuals = np.asarray(residuals, dtype=float)
    demand = np.maximum(forecast + residuals, 0)
    short = np.maximum(demand - quantity, 0)
    waste = np.maximum(quantity - demand, 0)
    return {
        "quantity": quantity,
        "expected_shortage": float(short.mean()),
        "expected_waste": float(waste.mean()),
        "stockout_prob": float((short > 0).mean()),
        "expected_cost": float(shortage_cost * short.mean() + waste_cost * waste.mean()),
    }


def optimal_quantity(forecast: float, residuals, shortage_cost: float,
                     waste_cost: float) -> float:
    """Newsvendor solution: quantile of simulated demand at the critical ratio."""
    demand = np.maximum(forecast + np.asarray(residuals, dtype=float), 0)
    return float(np.quantile(demand, critical_ratio(shortage_cost, waste_cost)))
