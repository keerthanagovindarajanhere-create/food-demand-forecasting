# Tail-Weighted XGBoost Experiment

## Hypothesis

Because large forecasting errors are concentrated in high-demand observations, increasing the training importance of high-demand observations may improve high-demand forecasting performance.

## Controlled Setup

- Same dataset
- Same temporal split
- Same features
- Same XGBoost hyperparameters
- Same random seed
- Only training sample weights were changed
- High demand defined using the training 80th percentile
- High-demand observations receive weight 2.0

Training high-demand threshold: 298.0000

## Overall Results

- MAE: 56.908995
- RMSE: 197.476707
- Bias: 9.598700
- Underprediction_Rate: 0.528660
- Overprediction_Rate: 0.347643

## Regime Results

regime  observations  actual_mean        MAE       RMSE      Bias  Underprediction_Rate  Overprediction_Rate
normal         37957    48.463858  11.445634  23.424649 -0.483491              0.517981             0.319414
  high         11939  1808.972688 201.448136 401.539175 41.652453              0.562610             0.437390
