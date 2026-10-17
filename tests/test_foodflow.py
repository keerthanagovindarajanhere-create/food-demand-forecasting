"""
Targeted tests for the FOODFLOW app logic.

Run:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import foodflow_core as core  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return core.load_model()


@pytest.fixture(scope="module")
def fcols():
    return core.load_feature_columns()


@pytest.fixture(scope="module")
def history():
    return core.load_history()[0]


@pytest.fixture(scope="module")
def validation():
    return core.load_validation_errors()


# ---------------------------------------------------------------- artifacts

def test_feature_schema_matches_model(model, fcols):
    assert fcols == list(model.get_booster().feature_names)
    assert len(fcols) == 51


def test_reported_metrics_reproduce(validation):
    assert core.mae(validation["sales"], validation["predicted"]) == pytest.approx(57.5538, abs=1e-3)
    assert core.rmse(validation["sales"], validation["predicted"]) == pytest.approx(198.0338, abs=1e-3)
    curve = core.cumulative_squared_error_share(validation["sales"], validation["predicted"])
    assert core.share_at(curve, 1) == pytest.approx(69.04, abs=0.01)
    assert core.share_at(curve, 5) == pytest.approx(93.86, abs=0.01)


def test_tail_weighted_metrics_reproduce(validation):
    tail = core.load_tail_predictions()
    assert np.array_equal(tail["actual"].values, validation["sales"].values)  # same row order
    assert core.mae(tail["actual"], tail["predicted"]) == pytest.approx(56.9090, abs=1e-3)


def test_seasonal_naive_baseline():
    base = pd.read_csv(core.BASELINE_PATH)
    seasonal = base.loc[base["Model"].str.startswith("Seasonal"), "MAE"].iloc[0]
    assert seasonal == pytest.approx(86.93, abs=0.01)


# ---------------------------------------------------------------- forecasting

def test_next_day_features_use_real_history(history, fcols):
    s = core.series_history(history, 44, "BEVERAGES")
    X = core.feature_vector(s, 44, "BEVERAGES", 3, fcols).iloc[0]
    sales = s.set_index("date")["sales"]
    d = core.LAST_OBSERVED_DATE
    assert X["lag_1"] == sales[d]
    assert X["lag_7"] == sales[d - pd.Timedelta(days=6)]
    assert X["lag_28"] == sales[d - pd.Timedelta(days=27)]
    assert X["rolling_mean_7"] == pytest.approx(sales.tail(7).mean())
    assert X["rolling_mean_28"] == pytest.approx(sales.tail(28).mean())
    assert X["day_of_week"] == 2  # 2017-08-16 was a Wednesday
    assert X["onpromotion"] == 3 and X["store_nbr"] == 44
    fam = [c for c in fcols if c.startswith("family_")]
    assert X[fam].sum() == 1 and X["family_BEVERAGES"] == 1


def test_recursive_forecast_feeds_back_predictions(model, fcols, history):
    s = core.series_history(history, 1, "BREAD/BAKERY")
    fc = core.forecast_series(model, fcols, s, 1, "BREAD/BAKERY", 3, 0)
    assert list(fc["method"]) == ["direct (real history)", "recursive", "recursive"]
    assert (fc["prediction"] >= 0).all() and np.isfinite(fc["prediction"]).all()
    # Day-2 lag_1 must be the day-1 prediction, not a real or stale value.
    work = pd.concat([s, pd.DataFrame({"date": [fc["date"][0]], "store_nbr": [1],
                                       "family": ["BREAD/BAKERY"],
                                       "sales": [fc["prediction"][0]], "onpromotion": [0]})])
    X2 = core._feature_row(work, fc["date"][1], 1, "BREAD/BAKERY", 0, fcols)
    assert X2["lag_1"].iloc[0] == pytest.approx(fc["prediction"][0])
    assert model.predict(X2)[0] == pytest.approx(fc["prediction"][1], rel=1e-5)


def test_invalid_inputs_are_rejected(model, fcols, history):
    s = core.series_history(history, 1, "DAIRY")
    with pytest.raises(ValueError):
        core.forecast_series(model, fcols, s, 1, "DAIRY", 0, 0)
    with pytest.raises(ValueError):
        core.forecast_series(model, fcols, s, 1, "DAIRY", core.MAX_HORIZON_DAYS + 1, 0)
    with pytest.raises(ValueError):
        core.forecast_series(model, fcols, s, 1, "DAIRY", 1, -1)
    with pytest.raises(ValueError):  # gap in history -> would misalign lags
        core.forecast_series(model, fcols, s.drop(index=s.index[-5]), 1, "DAIRY", 1, 0)


def test_matches_saved_validation_predictions_when_raw_data_present(model, fcols):
    """Only runs if data/raw/train.csv exists locally (it is not in the repo)."""
    if not core.TRAIN_PATH.exists():
        pytest.skip("data/raw/train.csv not available")
    raw = pd.read_csv(core.TRAIN_PATH, parse_dates=["date"])
    val = core.load_validation_errors()
    target = pd.Timestamp("2017-08-15")
    for store, fam in [(44, "BEVERAGES"), (1, "DAIRY"), (3, "GROCERY I")]:
        s = raw[(raw.store_nbr == store) & (raw.family == fam) & (raw.date < target)]
        promo = raw[(raw.store_nbr == store) & (raw.family == fam) & (raw.date == target)]["onpromotion"].iloc[0]
        X = core._feature_row(s.tail(60), target, store, fam, promo, fcols)
        pred = max(0.0, float(model.predict(X)[0]))
        saved = val[(val.store_nbr == store) & (val.family == fam) & (val.date == target)]["predicted"].iloc[0]
        assert pred == pytest.approx(saved, rel=1e-4, abs=1e-3)


# ---------------------------------------------------------------- planner

def test_planner_formulas():
    assert core.buffered_quantity(100, 10) == pytest.approx(110)
    assert core.critical_ratio(3, 1) == pytest.approx(0.75)
    r = core.simulate_outcomes(100, 100, [-20, 0, 20], shortage_cost=3, waste_cost=1)
    assert r["expected_shortage"] == pytest.approx(20 / 3)
    assert r["expected_waste"] == pytest.approx(20 / 3)
    assert r["expected_cost"] == pytest.approx(3 * 20 / 3 + 1 * 20 / 3)
    # Higher shortage cost should never lower the cost-balanced quantity.
    res = np.random.default_rng(0).normal(0, 30, 200)
    assert core.optimal_quantity(100, res, 5, 1) >= core.optimal_quantity(100, res, 1, 5)
