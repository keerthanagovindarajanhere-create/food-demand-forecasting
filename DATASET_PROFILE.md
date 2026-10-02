\# Empirical Dataset Profile



\## Dataset



\*\*Store Sales - Time Series Forecasting\*\*



Source: Corporación Favorita / Kaggle Store Sales Time Series Forecasting



The dataset contains historical sales for Favorita stores in Ecuador together

with promotion information and supporting contextual datasets including store

metadata, transactions, oil prices, and holidays/events.



\---



\## 1. Training Dataset



\### Schema



| Column | Type | Description |

|---|---|---|

| `id` | int64 | Observation identifier |

| `date` | datetime | Observation date |

| `store\_nbr` | int64 | Store identifier |

| `family` | string | Product family |

| `sales` | float64 | Sales demand |

| `onpromotion` | int64 | Number of products in the family on promotion |



\### Dimensions



\- Training observations: \*\*3,000,888\*\*

\- Columns: \*\*6\*\*

\- Stores: \*\*54\*\*

\- Product families: \*\*33\*\*

\- Store-family series: \*\*1,782\*\*



The 1,782 series correspond to:



`54 stores × 33 product families = 1,782`



\---



\## 2. Temporal Coverage



Training data:



\- Start: \*\*2013-01-01\*\*

\- End: \*\*2017-08-15\*\*

\- Calendar span: \*\*1,688 days\*\*

\- Observed dates: \*\*1,684\*\*



The four absent dates are:



\- 2013-12-25

\- 2014-12-25

\- 2015-12-25

\- 2016-12-25



These correspond to Christmas Day.



All 1,684 observed dates contain exactly 1,782

store-family observations.



Therefore:



\- No partial store-family coverage was observed on an existing training date.

\- The four missing calendar dates represent complete date-level gaps rather

&#x20; than random missing observations within individual series.



\---



\## 3. Data Completeness



No missing values were found in the six training columns.



| Column | Missing | Missing % |

|---|---:|---:|

| `id` | 0 | 0% |

| `date` | 0 | 0% |

| `store\_nbr` | 0 | 0% |

| `family` | 0 | 0% |

| `sales` | 0 | 0% |

| `onpromotion` | 0 | 0% |



Duplicate checks:



\- Duplicate `(date, store\_nbr, family)` keys: \*\*0\*\*

\- Duplicate `id` values: \*\*0\*\*



Negative sales observations: \*\*0\*\*



\---



\## 4. Demand Distribution



The target variable `sales` is strongly right-skewed.



| Statistic | Sales |

|---|---:|

| Minimum | 0 |

| Q1 | 0 |

| Median | 11 |

| Q3 | 195.8473 |

| 90th percentile | 867 |

| 95th percentile | 1,965 |

| 99th percentile | 5,507 |

| 99.9th percentile | 12,076.2255 |

| Maximum | 124,717 |



\### Demand sparsity



Zero-demand observations:



\- Count: \*\*939,130\*\*

\- Percentage: \*\*31.30%\*\*



Thus, nearly one-third of all store-family-day observations have zero

recorded sales.



This indicates substantial demand sparsity and should be considered when

designing forecasting models and evaluation procedures.



\### IQR-based extreme observations



Using the conventional upper IQR threshold:



`Q3 + 1.5 × IQR`



the calculated threshold is:



\*\*489.6181\*\*



Observations above this threshold:



\- Count: \*\*447,105\*\*

\- Percentage: \*\*14.90%\*\*



This is a statistical outlier flag, not a conclusion that these observations

are erroneous. High sales may represent legitimate differences in store,

product-family, promotion, seasonality, or event-driven demand.



\---



\## 5. Promotion Characteristics



`onpromotion` represents the number of products within the product family

that were on promotion.



| Statistic | Value |

|---|---:|

| Observations with promotion > 0 | 611,329 |

| Promotion coverage | 20.37% |

| Mean promoted items | 2.6028 |

| Maximum promoted items | 741 |



Promotion distribution:



| Quantile | Value |

|---|---:|

| 0% | 0 |

| 25% | 0 |

| 50% | 0 |

| 75% | 0 |

| 90% | 4 |

| 95% | 13 |

| 99% | 55 |

| 100% | 741 |



Promotion is therefore sparse and highly concentrated rather than uniformly

distributed across observations.



\---



\## 6. Series Completeness



There are \*\*1,782 store-family series\*\*.



Expected observations per series across the 1,688-day calendar span:



\*\*1,688\*\*



Observed observations per series:



\*\*1,684\*\*



All 1,782 series have exactly 1,684 observations.



Therefore, the missing dates occur consistently across all series rather

than affecting only particular store-family combinations.



\---



\## 7. Test Dataset



The test dataset contains:



\- Rows: \*\*28,512\*\*

\- Stores: \*\*54\*\*

\- Product families: \*\*33\*\*

\- Store-family series: \*\*1,782\*\*



Test period:



\*\*2017-08-16 → 2017-08-31\*\*



The test set therefore contains 16 forecast dates for every store-family

series.



The test set shares the following structural fields with training:



\- `id`

\- `date`

\- `store\_nbr`

\- `family`

\- `onpromotion`



The target `sales` is absent from the test set.



\---



\## 8. Store Metadata



`stores.csv` contains:



\- 54 stores

\- `store\_nbr`

\- `city`

\- `state`

\- `type`

\- `cluster`



Missing values: \*\*0\*\*



The metadata provides geographic and store-segmentation information that can

potentially be used as static covariates.



\---



\## 9. Transactions



`transactions.csv` contains:



\- `date`

\- `store\_nbr`

\- `transactions`



Rows: \*\*83,488\*\*



Date coverage:



\*\*2013-01-01 → 2017-08-15\*\*



Unique transaction dates: \*\*1,682\*\*



Stores represented: \*\*54\*\*



Transaction statistics:



| Statistic | Value |

|---|---:|

| Mean | 1,694.60 |

| Std | 963.29 |

| Minimum | 5 |

| Q1 | 1,046 |

| Median | 1,393 |

| Q3 | 2,079 |

| Maximum | 8,359 |



No missing transaction values were observed.



However, transaction coverage is not complete at the

date-store level:



\- Dates with all 54 stores: \*\*118\*\*

\- Dates with incomplete store coverage: \*\*1,564\*\*



Therefore, transactions should not automatically be treated as a complete

daily exogenous variable without an explicit missing/coverage strategy.



\---



\## 10. Oil Price Data



`oil.csv` contains:



\- `date`

\- `dcoilwtico`



Rows: \*\*1,218\*\*



Date coverage:



\*\*2013-01-01 → 2017-08-31\*\*



Missing oil values:



\*\*43\*\*



Non-missing observations:



\*\*1,175\*\*



Oil statistics:



| Statistic | Value |

|---|---:|

| Mean | 67.7144 |

| Std | 25.6305 |

| Minimum | 26.19 |

| Q1 | 46.405 |

| Median | 53.19 |

| Q3 | 95.66 |

| Maximum | 110.62 |



The oil dataset extends beyond the training period and covers the complete

test horizon.



\---



\## 11. Holidays and Events



`holidays\_events.csv` contains:



\- 350 records

\- 6 columns

\- No missing values



Event types:



| Type | Count |

|---|---:|

| Holiday | 221 |

| Event | 56 |

| Additional | 51 |

| Transfer | 12 |

| Bridge | 5 |

| Work Day | 5 |



Locales:



| Locale | Count |

|---|---:|

| National | 174 |

| Local | 152 |

| Regional | 24 |



Transferred events:



\- False: \*\*338\*\*

\- True: \*\*12\*\*



This provides multiple levels of calendar/event information that may be

relevant to demand variation.



\---



\## 12. Price Availability



No explicit price-related column was found in:



\- `train.csv`

\- `test.csv`



The selected dataset therefore does not provide a direct product-price

variable in the primary forecasting tables.



Price effects cannot consequently be modelled directly from an observed

product-price field unless an additional compatible data source is introduced.



\---



\# Empirical Findings Relevant to Forecasting



The profiling establishes the following characteristics:



1\. The forecasting problem contains \*\*1,782 parallel store-family time series\*\*.

2\. The target contains substantial \*\*zero-demand observations (31.30%)\*\*.

3\. Demand is \*\*strongly right-skewed\*\*, with a maximum of 124,717 compared

&#x20;  with a median of 11.

4\. Promotion is present in \*\*20.37%\*\* of observations and is itself highly

&#x20;  skewed.

5\. Training observations are structurally complete on every observed date,

&#x20;  with the four missing calendar dates corresponding to Christmas Day.

6\. The primary training table contains \*\*no missing values\*\* and no duplicate

&#x20;  series keys.

7\. Store metadata, transactions, oil prices, and holiday/event information

&#x20;  provide potential exogenous features.

8\. Transaction coverage is substantially incomplete at the

&#x20;  date-store level and therefore requires explicit treatment.

9\. No direct price variable is available in the primary train/test tables.

10\. The test period consists of \*\*16 future dates\*\* across all 1,782

&#x20;   store-family series.



These findings will guide subsequent feature engineering, validation design,

and model selection. They do not by themselves determine which forecasting

model should be used.

