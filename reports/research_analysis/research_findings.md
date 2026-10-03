# FOODFLOW — Research Analysis Findings

## Status

**Modeling status:** Frozen  
**Purpose:** Consolidated evidence from the completed research-analysis experiments.

This document is the source of truth for the final research figures, presentation, and subsequent paper writing.

No additional model training or model-selection experiments are included after this point unless explicitly stated as a separate future experiment.

---

# 1. Research Question

The central research question investigated in this analysis is:

> How does demand magnitude affect forecasting error, and can explicitly increasing the importance of high-demand observations improve performance in the extreme-demand regime?

The analysis focuses not only on aggregate forecasting accuracy but also on the distribution and concentration of forecasting errors.

The investigation therefore examines:

1. Overall baseline forecasting performance.
2. Error behaviour across demand regimes.
3. Concentration of extreme forecasting errors.
4. Differences in error across product families.
5. Temporal concentration of extreme errors.
6. The effect of tail-weighted training relative to the reference XGBoost model.

---

# 2. Baseline Forecasting Performance

## Observation

The reference XGBoost model performs substantially differently across demand regimes.

## Quantitative Evidence

For the reference model:

| Demand regime | Observations | MAE | RMSE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|---:|
| Normal | 37,957 | 11.44 | 21.95 | 1.96 | 58.32% |
| High | 11,939 | 204.18 | 402.95 | 33.00 | 55.42% |

The high-demand regime therefore produces dramatically larger forecasting errors than the normal-demand regime.

## Interpretation

Aggregate forecasting metrics conceal substantial heterogeneity in forecasting difficulty.

The error increase in the high-demand regime indicates that the forecasting problem is strongly affected by demand magnitude. High-demand observations contribute disproportionately to the total forecasting loss.

This motivates analysing the forecasting problem through the lens of error concentration rather than relying only on overall MAE.

## Limitation

The regime analysis is observational. The results establish an empirical relationship between demand magnitude and forecasting error but do not, by themselves, establish causality.

---

# 3. Extreme-Error Concentration

## Observation

A relatively small subset of observations produces very large absolute forecasting errors.

The tail of the error distribution is therefore substantially more severe than the central portion of the distribution.

## Quantitative Evidence

The previously measured absolute-error statistics include:

- Median absolute error: substantially lower than the upper tail.
- 95th-percentile absolute error: substantially larger than the median.
- Maximum absolute errors: several thousand units.

For the analysed product families:

| Family | Observations | MAE | Median AE | P95 AE | Maximum AE |
|---|---:|---:|---:|---:|---:|
| BEVERAGES | 1,512 | 475.30 | 304.76 | 1,486.37 | 4,021.91 |
| GROCERY I | 1,512 | 463.83 | 317.07 | 1,380.46 | 4,268.63 |

## Interpretation

The large separation between typical error and extreme error indicates a heavy-tailed forecasting-error distribution.

This means that improving average performance alone may not adequately address the operational impact of the forecasting system.

A small number of high-error observations can have a disproportionately large influence on squared-error-based objectives.

## Limitation

The reported extreme-error statistics are sensitive to the dataset and evaluation period. They should therefore be interpreted as evidence for this forecasting dataset rather than as a universal property of food-demand forecasting.

---

# 4. Demand Magnitude and Forecasting Error

## Observation

Forecasting error increases sharply with demand magnitude.

## Quantitative Evidence

The demand-regime comparison shows:

- Normal-demand MAE: **11.44**
- High-demand MAE: **204.18**

The corresponding RMSE values are:

- Normal-demand RMSE: **21.95**
- High-demand RMSE: **402.95**

Thus, the high-demand regime has substantially larger absolute and squared-error penalties.

## Interpretation

The relationship between demand magnitude and error suggests that forecasting difficulty is not uniformly distributed across the target space.

The problem can therefore be viewed as having a difficult high-demand tail rather than simply having uniformly noisy predictions.

This provides the motivation for explicitly examining tail-sensitive learning strategies.

## Limitation

The regime boundary and definition of "high demand" depend on the experimental setup. Different thresholds may produce different quantitative results.

---

# 5. Product-Family Error Differences

## Observation

Forecasting error is not uniform across product families.

The available family-level analysis shows substantial differences in MAE and extreme-error behaviour.

## Quantitative Evidence

Two representative family-level results are:

| Family | Observations | MAE | Median AE | P95 AE | Maximum AE |
|---|---:|---:|---:|---:|---:|
| BEVERAGES | 1,512 | 475.30 | 304.76 | 1,486.37 | 4,021.91 |
| GROCERY I | 1,512 | 463.83 | 317.07 | 1,380.46 | 4,268.63 |

The family-level analysis demonstrates that both typical error and extreme error vary across product categories.

## Interpretation

The forecasting challenge is therefore heterogeneous not only across demand magnitudes but also across product families.

This suggests that some product categories may contain different demand dynamics, variability, or susceptibility to extreme demand events.

The final family-level figure will provide a compact visual representation of this heterogeneity.

## Limitation

Family-level error differences alone cannot identify the underlying cause. Possible explanations include demand variability, product characteristics, event effects, data sparsity, or differences in temporal demand patterns.

---

# 6. Temporal Concentration of Extreme Errors

## Observation

Extreme forecasting errors are not necessarily distributed uniformly over time.

The analysis identified periods in which unusually large errors are concentrated.

## Quantitative Evidence

The extreme-error timeline analysis identifies observations with unusually high absolute forecasting error and examines their temporal distribution.

The largest observed family-level errors include:

- BEVERAGES: maximum absolute error ≈ **4,021.91**
- GROCERY I: maximum absolute error ≈ **4,268.63**

## Interpretation

Temporal clustering of extreme errors suggests that forecasting difficulty may be associated with specific periods rather than being purely random observation-level noise.

This is important because large forecasting failures may correspond to demand shocks, events, seasonal effects, or other temporal conditions that are not completely captured by the current feature representation.

## Limitation

The current analysis identifies temporal concentration but does not establish the specific external cause of each spike. Attribution to holidays, events, promotions, weather, or other external factors would require additional data.

---

# 7. Tail-Weighted XGBoost Experiment

## Observation

A tail-weighted training strategy was evaluated against the reference XGBoost model to investigate whether increasing the importance of high-demand observations changes performance in the difficult demand tail.

## Quantitative Evidence

The reference model produced:

| Regime | MAE | RMSE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|
| Normal | 11.44 | 21.95 | 1.96 | 58.32% |
| High | 204.18 | 402.95 | 33.00 | 55.42% |

The tail-weighted experiment is evaluated against this reference using the same regime-level metrics and the final comparison figure.

The purpose of the comparison is not simply to identify a model with a lower aggregate error, but to determine whether explicit tail emphasis changes the error profile in the high-demand regime.

## Interpretation

The tail-weighting experiment tests a specific hypothesis:

> If high-demand observations are systematically more difficult and contribute disproportionately to forecasting loss, increasing their training importance may alter performance in the high-demand regime.

This constitutes an empirical investigation of tail-sensitive learning for food-demand forecasting.

## Limitation

Tail weighting does not establish that the underlying causes of extreme demand have been modelled. Improvements or changes in tail performance may depend on the weighting strategy and selected definition of the high-demand regime.

---

# 8. Research Finding: Error Is Highly Unevenly Distributed

## Core Finding

Forecasting error is highly heterogeneous.

It varies across:

- demand magnitude,
- product family,
- and time.

The high-demand regime is particularly difficult, with substantially larger MAE and RMSE than the normal-demand regime.

## Evidence

High-demand observations:

- MAE = **204.18**
- RMSE = **402.95**

Normal-demand observations:

- MAE = **11.44**
- RMSE = **21.95**

The error distribution also contains extreme observations with absolute errors exceeding **4,000 units** in the analysed family-level results.

## Research Significance

This provides evidence that aggregate forecasting metrics alone do not fully describe the behaviour of the forecasting system.

A forecasting model may appear reasonable under average-error metrics while still exhibiting substantial failures in specific demand regimes.

---

# 9. Research Finding: The High-Demand Tail Is the Main Difficulty

## Core Finding

The largest forecasting errors are concentrated in the high-demand portion of the target distribution.

## Evidence

The high-demand regime has:

- approximately **18%** of the analysed observations,
- but an MAE of **204.18** compared with **11.44** in the normal regime,
- and an RMSE of **402.95** compared with **21.95**.

## Interpretation

The disproportionate error magnitude indicates that forecasting difficulty increases sharply in the demand tail.

This motivates evaluating forecasting systems using regime-specific and tail-sensitive metrics rather than relying exclusively on global averages.

## Limitation

The proportion and magnitude of the high-demand regime are dataset-specific.

---

# 10. Research Finding: Tail-Sensitive Evaluation Is Necessary

## Core Finding

Because extreme errors have a disproportionate effect on squared-error metrics, evaluating only average error can hide important forecasting failures.

## Evidence

The observed error distributions contain a substantial difference between median error, 95th-percentile error, and maximum error.

For example:

- BEVERAGES median AE = **304.76**
- BEVERAGES P95 AE = **1,486.37**
- BEVERAGES maximum AE = **4,021.91**

For GROCERY I:

- median AE = **317.07**
- P95 AE = **1,380.46**
- maximum AE = **4,268.63**

## Interpretation

The wide spread between central and extreme error statistics demonstrates that forecasting performance should be evaluated beyond a single average metric.

This supports the use of:

- MAE,
- RMSE,
- regime-specific MAE/RMSE,
- percentile error statistics,
- extreme-error concentration,
- and tail-weighted comparisons.

---

# 11. Final Evidence Set

The final research analysis will contain exactly five figures.

### Figure 1 — Demand Magnitude → MAE/RMSE

Purpose:

Show how forecasting error changes as demand magnitude increases.

Expected takeaway:

> Forecasting error increases sharply with demand magnitude.

---

### Figure 2 — Cumulative Squared-Error Contribution

Purpose:

Show how much of the total squared forecasting error is contributed by the largest-error observations.

Expected takeaway:

> A relatively small subset of extreme observations contributes disproportionately to total squared error.

---

### Figure 3 — Family-Level MAE

Purpose:

Show heterogeneity in forecasting error across product families.

Expected takeaway:

> Forecasting difficulty differs substantially across product categories.

---

### Figure 4 — Extreme-Error Timeline

Purpose:

Show when the largest forecasting failures occur.

Expected takeaway:

> Extreme errors exhibit temporal concentration rather than being uniformly distributed.

---

### Figure 5 — Reference vs Tail-Weighted XGBoost

Purpose:

Compare the reference and tail-weighted models across the relevant demand regimes.

Expected takeaway:

> Tail weighting provides an empirical test of whether explicitly emphasizing high-demand observations changes performance in the difficult demand regime.

---

# 12. Overall Research Contribution

The current evidence supports an empirical contribution centred on the following observation:

> Food-demand forecasting errors are strongly heterogeneous and concentrated in the high-demand tail, with extreme observations contributing disproportionately to forecasting loss.

The analysis further investigates whether this structure can be addressed through explicit tail-weighted training.

The contribution is therefore not presented as a claim of a universally superior forecasting algorithm.

Instead, the study provides an empirical analysis of:

1. demand-dependent forecasting difficulty,
2. extreme-error concentration,
3. product-family heterogeneity,
4. temporal concentration of failures,
5. and the effect of tail-weighted learning.

---

# 13. Current Limitations

The following limitations must remain explicit in the final presentation and paper:

1. The analysis is based on a single forecasting dataset.
2. The identified high-demand regime depends on the selected threshold.
3. Extreme-error observations are not automatically attributable to specific real-world causes.
4. External variables such as promotions, holidays, weather, and local events are not fully represented in the current analysis.
5. Tail weighting changes the training objective but does not directly model the causes of demand spikes.
6. The empirical findings should not be generalized beyond the evaluated data without additional datasets or experiments.
7. No causal interpretation is claimed from the observed error patterns.

---

# 14. Modeling Freeze

As of this analysis:

**Modeling is frozen.**

No additional model architectures, hyperparameter searches, feature-engineering experiments, or alternative weighting strategies should be introduced into the primary research analysis.

The remaining work is:

1. finalize the five figures,
2. verify numerical consistency,
3. document findings,
4. construct the presentation,
5. and use the frozen evidence as the basis for the research paper.