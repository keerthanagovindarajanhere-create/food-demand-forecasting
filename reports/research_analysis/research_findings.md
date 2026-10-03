# FOODFLOW — Research Analysis Findings

## 1. Research Status

**Modeling status:** Frozen

This document contains the final quantitative evidence from the FOODFLOW research analysis.

The analysis investigates whether food-demand forecasting errors are uniformly distributed or concentrated in specific demand regimes, product families, and time periods.

A final controlled experiment evaluates whether explicitly increasing the training importance of high-demand observations changes forecasting performance.

No additional model training is part of this frozen analysis.

---

# 2. Research Question

The central research question is:

> How does demand magnitude affect forecasting error, and can explicitly increasing the importance of high-demand observations alter forecasting performance in the difficult demand tail?

The analysis examines:

1. Forecasting error across demand magnitudes.
2. Concentration of squared forecasting error.
3. Error heterogeneity across product families.
4. Temporal concentration of extreme errors.
5. The effect of tail-weighted XGBoost relative to the reference model.

---

# 3. Reference Forecasting Model

The reference model is an XGBoost forecasting model evaluated on a held-out temporal validation set.

The validation set contains:

**49,896 observations.**

The reference model's overall performance is:

| Metric | Reference |
|---|---:|
| MAE | 57.5538 |
| RMSE | 198.0338 |

The regime analysis divides observations into normal- and high-demand regimes.

| Regime | Observations | MAE | RMSE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|---:|
| Normal | 37,957 | 11.4352 | 21.9526 | 1.9633 | 55.42% |
| High | 11,939 | 204.1761 | 402.9477 | 32.9980 | 55.42% |

The high-demand regime therefore represents approximately 23.93% of the validation observations while exhibiting substantially larger forecasting error.

---

# 4. Figure 1 — Demand Magnitude → Forecasting Error

## Observation

Forecasting error increases sharply as demand magnitude increases.

The demand distribution was divided into five groups using the observed demand values.

| Demand group | Observations | Actual mean | MAE | RMSE |
|---|---:|---:|---:|---:|
| (-0.001, 2.0] | 11,033 | 0.48 | 0.58 | 1.69 |
| (2.0, 13.0] | 9,045 | 7.03 | 5.25 | 6.70 |
| (13.0, 83.0] | 9,868 | 35.68 | 13.40 | 18.46 |
| (83.0, 416.108] | 9,971 | 211.31 | 34.54 | 49.13 |
| (416.108, 18340.0] | 9,979 | 2,095.29 | 234.62 | 439.66 |

## Quantitative Evidence

The lowest-demand group has:

- MAE = 0.58
- RMSE = 1.69

The highest-demand group has:

- MAE = 234.62
- RMSE = 439.66

Therefore, the highest-demand group has approximately 406× the MAE of the lowest-demand group.

## Interpretation

Forecasting difficulty is strongly dependent on demand magnitude.

The error does not increase uniformly with demand. Instead, the highest-demand observations form a substantially more difficult portion of the forecasting problem.

This supports evaluating forecasting systems across demand regimes rather than relying exclusively on an aggregate metric.

## Limitation

The demand groups are dataset-specific and describe an empirical relationship rather than establishing causality.

---

# 5. Figure 2 — Cumulative Squared-Error Contribution

## Observation

Squared forecasting error is extremely concentrated in a small fraction of validation observations.

## Quantitative Evidence

There are 49,896 validation observations.

When observations are sorted from largest to smallest squared error:

| Largest-error observations | Share of total squared error |
|---:|---:|
| Top 1% | 69.04% |
| Top 2% | 81.89% |
| Top 5% | 93.86% |
| Top 10% | 98.01% |
| Top 20% | 99.55% |

## Interpretation

The top 1% of observations contribute approximately 69% of the total squared forecasting error.

The top 5% contribute approximately 94%.

This demonstrates that aggregate squared-error metrics are dominated by a relatively small subset of difficult observations.

The result provides strong empirical motivation for analysing the forecasting tail separately from average performance.

## Limitation

This concentration is specific to the evaluated validation set and the squared-error metric. It should not be assumed to occur at the same magnitude in other datasets.

---

# 6. Figure 3 — Family-Level MAE

## Observation

Forecasting error varies substantially across product families.

## Quantitative Evidence

The highest observed family-level MAE values include:

| Product family | Observations | MAE | RMSE |
|---|---:|---:|---:|
| BEVERAGES | 1,512 | 448.73 | 685.23 |
| GROCERY I | 1,512 | 440.78 | 656.65 |
| CLEANING | 1,512 | 243.96 | 489.69 |
| PRODUCE | 1,512 | 183.40 | 291.95 |
| DAIRY | 1,512 | 78.41 | 121.53 |
| BREAD/BAKERY | 1,512 | 56.72 | 92.00 |
| MEATS | 1,512 | 50.70 | 81.02 |

The complete family-level distribution is shown in Figure 3.

## Interpretation

Forecasting difficulty is heterogeneous across product categories.

BEVERAGES and GROCERY I exhibit substantially larger MAE than many other product families.

This indicates that demand magnitude alone does not fully describe forecasting difficulty; product-family structure also corresponds to substantial differences in observed error.

## Limitation

Family-level differences do not establish why some categories are harder to forecast.

Potential explanations include demand variability, product characteristics, temporal patterns, event effects, or differences in the distribution of high-demand observations.

---

# 7. Figure 4 — Extreme-Error Timeline

## Observation

The identified extreme-error observations show substantial variation across dates.

## Quantitative Evidence

The largest identified daily absolute error occurs on:

**2017-08-12: 154,953.86 units**

Other large daily absolute errors include:

- 2017-08-13: 113,247.26
- 2017-08-05: 90,316.45
- 2017-07-28: 71,209.93
- 2017-08-06: 62,312.74
- 2017-08-03: 59,911.01
- 2017-08-11: 53,018.72

## Interpretation

The identified extreme-error dates demonstrate that the largest forecasting failures can be concentrated around particular time periods.

The magnitude of the largest daily errors is substantially greater than the typical observation-level errors.

This motivates temporal analysis of forecasting failures rather than treating all validation observations as equally difficult.

## Limitation

The analysis identifies when extreme errors occurred but does not establish their external cause.

No causal attribution is made to promotions, holidays, weather, events, or other external factors without additional evidence.

---

# 8. Figure 5 — Reference vs Tail-Weighted XGBoost

## Hypothesis

Because large forecasting errors are concentrated in high-demand observations, increasing the training importance of high-demand observations may alter high-demand forecasting performance.

## Controlled Setup

The tail-weighted experiment used:

- the same dataset,
- the same temporal split,
- the same features,
- the same XGBoost hyperparameters,
- the same random seed,
- and the same model architecture.

The only change was the training sample weighting.

High-demand observations were defined using the training 80th percentile.

Training high-demand threshold:

**298 units**

High-demand observations received a training weight of:

**2.0**

## Quantitative Evidence

| Regime | Metric | Reference | Tail-weighted | Change |
|---|---|---:|---:|---:|
| Overall | MAE | 57.5538 | 56.9090 | -0.6448 |
| Overall | RMSE | 198.0338 | 197.4767 | -0.5571 |
| High | MAE | 204.1761 | 201.4481 | -2.7280 |
| High | RMSE | 402.9477 | 401.5392 | -1.4085 |
| High | Bias | 33.00 | 41.65 | +8.65 |
| High | Underprediction rate | 55.42% | 56.26% | +0.85 pp |
| Normal | MAE | 11.4352 | 11.4456 | +0.0104 |
| Normal | RMSE | 21.9526 | 23.4246 | +1.4720 |

Tail-weighted regime results:

| Regime | Observations | Actual mean | MAE | RMSE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 37,957 | 48.46 | 11.45 | 23.42 | -0.48 | 51.80% |
| High | 11,939 | 1,808.97 | 201.45 | 401.54 | 41.65 | 56.26% |

## Interpretation

Tail weighting produced a modest reduction in overall and high-demand error.

High-demand MAE decreased from 204.18 to 201.45, while high-demand RMSE decreased from 402.95 to 401.54.

However, the improvement was not uniform.

High-demand bias increased from 33.00 to 41.65, and the high-demand underprediction rate increased from 55.42% to 56.26%.

Normal-regime RMSE also increased from 21.95 to 23.42.

Therefore, the experiment demonstrates a trade-off rather than a uniformly superior forecasting model.

## Limitation

Only one tail-weighting configuration was evaluated in the frozen analysis.

The experiment therefore demonstrates the empirical effect of this controlled weighting strategy rather than establishing that tail weighting is generally optimal.

---

# 9. Consolidated Research Findings

## Finding 1 — Forecasting difficulty increases strongly with demand magnitude

The highest demand group has MAE of 234.62 and RMSE of 439.66, compared with MAE of 0.58 and RMSE of 1.69 for the lowest demand group.

This establishes strong demand-dependent heterogeneity in forecasting error.

---

## Finding 2 — Squared forecasting error is dominated by a small error tail

The top 1% of validation observations contribute 69.04% of total squared error.

The top 5% contribute 93.86%.

This demonstrates that average forecasting performance does not adequately describe the distribution of the largest failures.

---

## Finding 3 — Forecasting difficulty differs across product families

BEVERAGES and GROCERY I have MAE values of 448.73 and 440.78 respectively, substantially exceeding many other product families.

This indicates substantial category-level heterogeneity.

---

## Finding 4 — Extreme errors vary substantially across time

The largest identified daily absolute error is 154,953.86 units on 2017-08-12.

The extreme-error timeline demonstrates that large forecasting failures can be concentrated in particular periods.

---

## Finding 5 — Tail weighting changes the error profile but involves trade-offs

Tail-weighted XGBoost reduced:

- overall MAE by 0.6448,
- overall RMSE by 0.5571,
- high-demand MAE by 2.7280,
- high-demand RMSE by 1.4085.

However, it also increased:

- high-demand bias by 8.65,
- high-demand underprediction rate by 0.85 percentage points,
- normal-regime RMSE by 1.47.

The result therefore supports a nuanced conclusion:

> Tail weighting can modestly reduce error in the high-demand regime, but improvements are accompanied by changes in bias and normal-regime performance.

---

# 10. Research Contribution

The empirical contribution of the analysis is the characterization of food-demand forecasting error as a heterogeneous and strongly tail-concentrated problem.

The evidence demonstrates that:

1. Forecasting error increases sharply with demand magnitude.
2. A very small proportion of observations dominates squared forecasting loss.
3. Error varies substantially across product families.
4. Extreme forecasting failures exhibit substantial temporal variation.
5. Explicit tail weighting changes high-demand performance, but introduces measurable trade-offs.

The study therefore argues for evaluating food-demand forecasting systems using demand-regime and tail-sensitive analyses in addition to aggregate forecasting metrics.

The contribution is empirical rather than a claim of a universally superior forecasting algorithm.

---

# 11. Limitations

1. The analysis is based on a single forecasting dataset.
2. Demand-group boundaries are dataset-specific.
3. The high-demand regime definition uses the training 80th percentile.
4. Extreme-error dates do not establish causal explanations.
5. External factors such as promotions, weather, events, and local disruptions are not fully represented.
6. Only one tail-weighting configuration is included in the frozen experiment.
7. The results should not be generalized to other datasets without additional evaluation.
8. No causal interpretation is claimed.

---

# 12. Final Figures

The final research analysis contains exactly five figures:

1. `01_demand_magnitude_error.png`
2. `02_cumulative_squared_error.png`
3. `03_family_mae.png`
4. `04_extreme_error_timeline.png`
5. `05_reference_vs_tail_weighted.png`

These figures constitute the primary visual evidence for the research presentation.

---

# 13. Modeling Freeze

Modeling is frozen at this stage.

No additional model architectures, hyperparameter searches, feature-engineering experiments, or weighting strategies are required for the primary research analysis.

Remaining work:

1. Finalize and commit this research-analysis report.
2. Commit the five final figures.
3. Build the STRIDE presentation around the frozen evidence.
4. Use the same evidence as the basis for the research paper.