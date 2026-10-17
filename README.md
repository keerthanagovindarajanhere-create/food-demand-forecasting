# FOODFLOW

## Food Demand Forecasting & Smart Preparation Planning

FOODFLOW forecasts daily demand for each store and product family, studies where the
forecasting model fails, and turns a forecast into an illustrative preparation quantity.
It ships as a four-page Streamlit dashboard built on a frozen XGBoost model.

---

## Problem

Food retailers decide every day how much to stock or prepare. Too little means empty shelves
and lost sales; too much means waste. Demand varies by store, product family, weekday, season,
promotions and recent sales history, so fixed rules of thumb perform poorly.

FOODFLOW forecasts demand from historical sales patterns and operational information, and
examines whether good *average* accuracy hides large failures in specific demand regimes.

---

## Dataset

Kaggle **Store Sales – Time Series Forecasting** (Corporación Favorita, Ecuador).

| Property | Value |
|---|---:|
| Training observations | 3,000,888 |
| Stores × product families | 54 × 33 = 1,782 series |
| Period | 2013-01-01 to 2017-08-15 |
| Zero-sales observations | 31.30% |
| Rows with any promotion | 20.37% |

`sales` is strongly right-skewed (median 11, 99th percentile 5,507). `onpromotion` is the
*number* of items in the family on promotion. Full profile: `docs/DATASET_PROFILE.md`.

The raw CSV is not committed (see `.gitignore`). The dashboard does **not** need it: it uses
the saved validation results, which contain actual sales for every series from 2017-07-19 to
2017-08-15. If `data/raw/train.csv` is present, the forecast page uses it as its history source
instead.

---

## Methodology

- **Temporal split:** train 2013-01-01 → 2017-07-18; validate 2017-07-19 → 2017-08-15
  (28 days, 49,896 observations).
- **Features (51):** lags 1/7/14/28; 7/14/28-day rolling means (shifted, past-only); day of
  week, day of month, week of year, month, quarter, year, weekend / month-start / month-end
  flags; promotion count; store number; one-hot product family (`src/models/features.py`).
- **Model:** XGBoost, 500 trees, learning rate 0.05, max depth 8, subsample 0.8, colsample 0.8,
  min child weight 5; predictions clipped at 0 (`src/models/train_xgb.py`).
- **Research analysis:** error by demand magnitude, cumulative squared-error concentration,
  family / store / date breakdowns, and a controlled tail-weighted retraining experiment.

---

## Results

All values below are recomputed by the app and by `tests/test_foodflow.py` from the saved
artifacts in `reports/`.

| Model | MAE | RMSE | RMSLE |
|---|---:|---:|---:|
| Naive (yesterday) | 114.01 | 390.21 | 0.547 |
| Seasonal naive (same weekday last week) | 86.93 | 313.83 | 0.547 |
| 7-day moving average | 96.30 | 326.18 | 0.450 |
| **Reference XGBoost** | **57.55** | **198.03** | 0.765 |

XGBoost reduces MAE by 33.8% versus the seasonal naive baseline. It does not win on RMSLE,
which heavily weights relative errors on low and zero-demand series; the model was optimised
for absolute error.

### Error concentration

Sorting validation observations by squared error:

| Worst observations | Share of total squared error |
|---:|---:|
| Top 1% (499 rows) | 69.04% |
| Top 2% | 81.89% |
| Top 5% | 93.86% |
| Top 10% | 98.01% |
| Top 20% | 99.55% |

The top 1% is dominated by BEVERAGES, GROCERY I, CLEANING and PRODUCE. MAE rises from 0.58 in
the lowest demand quintile to 234.62 in the highest.

### Tail-weighted experiment

Same data, split, features, hyperparameters and seed; only sample weights changed (weight 2.0
for training rows above the 80th percentile, 298 units).

| Metric | Reference | Tail-weighted | Change |
|---|---:|---:|---:|
| Overall MAE | 57.5538 | 56.9090 | −0.64 |
| Overall RMSE | 198.0338 | 197.4767 | −0.56 |
| High-demand MAE | 204.1761 | 201.4481 | −2.73 |
| Normal-demand RMSE | 21.9526 | 23.4246 | +1.47 |
| High-demand bias | +33.00 | +41.65 | +8.65 |

A small, consistent gain where the error is concentrated, with trade-offs in the normal regime
and in over-prediction bias. The app uses the frozen reference model.

Details: `reports/research_analysis/research_findings.md`.

---

## Dashboard

| Page | What it shows |
|---|---|
| **Overview** | Problem, objective, verified KPI cards, baselines vs XGBoost, key finding, dataset stats |
| **Demand Forecasting** | Store / family / date / promotion inputs → frozen-model forecast, recent demand chart, validation fit, feature-based explanation, promotion sensitivity |
| **Demand & Error Analysis** | Demand trends, family comparisons, actual vs predicted with filters, error distribution, cumulative squared-error curve, demand-regime and tail-weighted comparison, feature importance |
| **Smart Preparation Planner** | Safety-buffer quantity, shortage-vs-waste what-if table, cost-balanced (newsvendor) quantity with configurable costs |

### How forecasts are produced

The data ends on 2017-08-15, so the supported forecast dates are **2017-08-16 to 2017-08-31**.

- **16 Aug — direct next-day forecast.** Every lag and rolling feature comes from real sales,
  exactly as in validation.
- **17–31 Aug — recursive forecast.** Earlier predictions are fed back as lag values. These are
  clearly labelled in the app and are less reliable.

Features are built with the same `create_features` function used in training, and the app
checks that the last 28 days of history are complete before predicting.

### Preparation planner formulas

- Buffered quantity = forecast × (1 + buffer%).
- Demand scenarios: `D_i = max(0, forecast + r_i)`, where `r_i` are the model's 28 validation
  errors (actual − predicted) for that store-family.
- Cost of preparing Q = `C_short × max(D − Q, 0) + C_waste × max(Q − D, 0)`, averaged over scenarios.
- Cost-balanced quantity = quantile of `D_i` at the critical ratio `C_short / (C_short + C_waste)`.

All planner values are simulations based on user-entered costs. No real inventory, savings or
waste reduction is measured.

---

## Installation and running

```bash
git clone https://github.com/keerthanagovindarajanhere-create/food-demand-forecasting.git
cd food-demand-forecasting
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m streamlit run app.py
```

The app opens at http://localhost:8501.

Run the tests:

```bash
python -m pytest tests -q
```

One test (`test_matches_saved_validation_predictions_when_raw_data_present`) is skipped unless
`data/raw/train.csv` exists; with the raw data present it checks that the app's feature
pipeline reproduces the saved validation predictions.

---

## Project structure

```
app.py                      Streamlit entry point and navigation
pages/                      overview, forecast, analysis, planner
src/foodflow_core.py        forecasting, metrics and planner logic (no Streamlit)
src/ui.py                   cached loaders, styling, KPI cards
src/models/                 features, data preparation, training, baselines
scripts/                    profiling, validation, experiments, figure generation
artifacts/model/            frozen XGBoost model and feature schema
reports/                    baseline, error-analysis and tail-weighted results
docs/                       dataset profile, evaluation and methodology notes
tests/                      targeted tests and page smoke tests
```

---

## Known limitations

- Forecasts are only available for 2017-08-16 to 2017-08-31. Beyond 16 Aug they are recursive,
  and the model was only validated on next-day forecasts.
- The promotion value entered is applied to every day of a recursive forecast.
- Holidays, oil prices, transactions and store metadata are not model features.
- The ± band on the forecast is the series' validation MAE, not a calibrated prediction interval.
- Planner scenarios use only 28 past errors per series, and the costs are user assumptions.
- The error-concentration and tail-weighting results come from one 28-day validation window.
