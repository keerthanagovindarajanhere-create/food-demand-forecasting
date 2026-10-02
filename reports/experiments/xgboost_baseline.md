\# Initial XGBoost Experiment



\## Objective



Evaluate an initial XGBoost regression model using historical demand, rolling, calendar, promotion, store, and family features.



\## Dataset



\- Training period: 2013-01-01 to 2017-07-18

\- Validation period: 2017-07-19 to 2017-08-15

\- Training observations: 2,901,096

\- Validation observations: 49,896

\- Features: 46



\## Features



\### Historical demand

\- lag\_1

\- lag\_7

\- lag\_14

\- lag\_28



\### Rolling demand

\- rolling\_mean\_7

\- rolling\_mean\_14

\- rolling\_mean\_28



\### Calendar

\- day\_of\_week

\- day\_of\_month

\- month

\- year



\### Other

\- onpromotion

\- store\_nbr

\- one-hot encoded family



All historical features are constructed using only observations available before the prediction date.



\## Model



XGBRegressor:



\- objective: reg:squarederror

\- n\_estimators: 500

\- learning\_rate: 0.05

\- max\_depth: 8

\- subsample: 0.8

\- colsample\_bytree: 0.8

\- min\_child\_weight: 5

\- reg\_alpha: 0

\- reg\_lambda: 1

\- random\_state: 42



\## Results



| Model | MAE | RMSE | RMSLE |

|---|---:|---:|---:|

| Naive | 114.0103 | 390.2131 | 0.5473 |

| Seasonal Naive (7d) | 86.9272 | 313.8271 | 0.5468 |

| 7-Day Moving Average | 96.3021 | 326.1776 | 0.4497 |

| Initial XGBoost | 59.0372 | 206.3915 | 0.4563 |



\## Findings



The initial XGBoost model substantially improves MAE and RMSE compared with the simple forecasting baselines.



However, the 7-day moving-average baseline achieves a lower RMSLE than the initial XGBoost model. Therefore, XGBoost does not outperform every baseline on every metric.



This establishes the initial XGBoost model as a strong candidate for further experimentation rather than a final model.



\## Next Experiment



Investigate XGBoost improvements through controlled experiments, beginning with model configuration and feature importance rather than unrestricted hyperparameter tuning.

