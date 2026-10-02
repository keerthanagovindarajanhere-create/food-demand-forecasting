\# Baseline Forecasting Experiment



\## Objective



Establish simple forecasting benchmarks before evaluating machine-learning models.



The validation period is the exact 28-day temporal holdout used throughout the project.



\## Dataset Split



\- Training period: 2013-01-01 to 2017-07-18

\- Validation period: 2017-07-19 to 2017-08-15

\- Validation days: 28

\- Unique store-family series: 1,782

\- Validation observations: 49,896

\- Duplicate observations: 0



\## Baselines



\### 1. Naive



Predict the previous observed sales value for each store-family series.



\### 2. Seasonal Naive



Predict sales using the value from 7 days earlier.



This provides a simple benchmark for weekly seasonality.



\### 3. 7-Day Moving Average



Predict using the mean sales of the previous seven observations.



Only historical observations available before the prediction date are used.



\## Results



| Model | MAE | RMSE | RMSLE |

|---|---:|---:|---:|

| Naive | 114.0103 | 390.2131 | 0.5473 |

| Seasonal Naive (7d) | 86.9272 | 313.8271 | 0.5468 |

| 7-Day Moving Average | 96.3021 | 326.1776 | 0.4497 |



\## Interpretation



The baselines establish the minimum performance level that subsequent machine-learning models must be compared against.



The seasonal naive model improves substantially over the basic naive forecast on MAE and RMSE, indicating that weekly temporal structure is important in the demand series.



The 7-day moving average produces the lowest RMSLE among the tested baselines, while its MAE and RMSE remain higher than the seasonal naive model.



These results motivate the inclusion of lagged and rolling historical-demand features in subsequent machine-learning experiments.



\## Validation Integrity



The validation panel contains all 1,782 store-family series for all 28 validation days:



\- Expected observations: 49,896

\- Actual observations: 49,896

\- Duplicate rows: 0



No validation observations were used to construct historical features or train the baseline forecasts.

