# FOODFLOW

## Food Demand Forecasting & Smart Preparation Optimization

FOODFLOW is a food demand forecasting system designed to estimate
product demand for individual stores and product families.

The project combines time-series feature engineering, machine learning,
error analysis and a targeted high-demand tail experiment.

---

## Problem

Food businesses need reliable demand estimates to prepare sufficient
inventory while avoiding unnecessary over-preparation.

Demand varies across:

- stores
- product families
- weekdays
- seasons
- promotions
- recent demand history

The objective of FOODFLOW is to forecast demand using historical sales
patterns and operational information.

---

## Dataset

The project uses the Ecuador store sales dataset from Kaggle.

The training data contains:

- 54 stores
- 33 product families
- approximately 3 million observations
- daily sales observations
- promotion information
- historical demand patterns

The observed training period spans:

2013-01-01 to 2017-08-15

---

## Methodology

The project follows a temporal forecasting setup.

### Feature Engineering

The model uses:

- lag-1 demand
- lag-7 demand
- lag-14 demand
- lag-28 demand
- 7-day rolling mean
- 14-day rolling mean
- 28-day rolling mean
- day of week
- day of month
- week of year
- month
- quarter
- year
- weekend indicator
- month-start indicator
- month-end indicator
- promotion status
- store number
- product-family encoding

---

## Baselines

Three simple forecasting baselines were evaluated:

| Model | MAE | RMSE |
|---|---:|---:|
| Naive | 114.01 | 390.21 |
| Seasonal Naive | 86.93 | 313.83 |
| 7-Day Moving Average | 96.30 | 326.18 |

These baselines establish the difficulty of the forecasting problem.

---

## Reference XGBoost Model

The final reference model uses XGBoost with:

- 500 estimators
- learning rate = 0.05
- maximum depth = 8
- subsample = 0.8
- column sampling = 0.8
- minimum child weight = 5

The frozen reference model achieved:

- MAE = 57.5538
- RMSE = 198.0338
- RMSLE = 0.7651

The model is stored in:

`artifacts/model/xgb_reference.pkl`

The feature schema is stored in:

`artifacts/model/feature_columns.json`

---

## Error Analysis

The research analysis showed that forecast error is highly concentrated.

On the validation set:

- 1% of observations account for approximately 69.04% of squared error
- 2% account for approximately 81.89%
- 5% account for approximately 93.86%
- 10% account for approximately 98.01%
- 20% account for approximately 99.55%

This indicates that a relatively small number of observations dominate
the total squared forecasting error.

The largest errors are concentrated in high-demand observations and
specific product-family/store combinations.

---

## Tail-Weighted Experiment

A targeted experiment was conducted after identifying the high-demand
tail as an important failure regime.

High-demand observations were defined using the 80th percentile of
training demand.

The experiment assigned:

- weight = 1.0 to normal-demand observations
- weight = 2.0 to high-demand observations

Results:

| Metric | Reference | Tail-Weighted |
|---|---:|---:|
| Overall MAE | 57.5538 | 56.9090 |
| Overall RMSE | 198.0338 | 197.4767 |
| High-demand MAE | 204.1761 | 201.4481 |

The experiment produced a small improvement in overall error and a
larger improvement within the high-demand regime.

This provides an empirical finding rather than claiming a universally
optimal forecasting method.

---

## Research Contribution

The main contribution of FOODFLOW is an empirical investigation of
forecasting failure concentration in a large retail demand dataset.

The analysis demonstrates that:

1. Strong overall forecasting performance does not imply uniform
   performance across demand regimes.

2. A small fraction of high-error observations can dominate total
   squared error.

3. High-demand observations represent an important forecasting
   failure regime.

4. A simple tail-weighting strategy can slightly improve forecasting
   performance in this regime.

The project therefore focuses on diagnosing and addressing a specific
forecasting failure mode rather than performing unrestricted feature
engineering or hyperparameter tuning.

---

## Streamlit Application

FOODFLOW provides an interactive Streamlit interface.

The user can provide:

- store
- product family
- forecast date
- promotion status

The application automatically constructs the required historical and
calendar features and sends them to the frozen XGBoost model.

The application returns the predicted demand.

### Run locally

```bash
python -m streamlit run app.py
