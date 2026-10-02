\# FOODFLOW — Methodology



\## 1. Overview



FOODFLOW is a food demand forecasting system designed to predict future product-level demand using historical sales, temporal patterns, store information, product-family information, and relevant external factors.



The project is structured as a reproducible forecasting pipeline rather than a single-model prediction task.



The methodology follows this progression:



1\. Dataset evaluation

2\. Data validation and preprocessing

3\. Exploratory time-series analysis

4\. Temporal train-validation-test design

5\. Feature engineering

6\. Baseline forecasting

7\. Machine learning forecasting

8\. Advanced temporal modelling

9\. Model evaluation and comparison

10\. Error analysis

11\. Forecast generation and interpretation



The primary objective is not only to obtain a low forecasting error, but to determine which modelling choices provide reliable and generalizable demand predictions without introducing temporal data leakage.



\---



\## 2. Research Objective



The central objective of FOODFLOW is:



> To develop and evaluate a reproducible machine-learning-based framework for forecasting food demand at store-product level while explicitly accounting for temporal structure, demand variability, seasonality, and forecasting data leakage.



The project investigates whether carefully engineered temporal and demand-related features can improve forecasting performance over simple historical baselines.



The methodology also emphasizes:



\- realistic forecasting conditions

\- chronological validation

\- reproducibility

\- interpretable feature engineering

\- comparison across model families

\- systematic error analysis



\---



\## 3. Dataset



The primary dataset used for the project is the Corporación Favorita Grocery Sales Forecasting dataset.



The main training data contains historical observations with fields including:



\- `date`

\- `store\_nbr`

\- `family`

\- `sales`

\- `onpromotion`



Additional datasets may be incorporated when justified by the forecasting methodology, including:



\- store metadata

\- oil prices

\- holidays/events

\- transaction information



Before modelling, all candidate datasets are evaluated for:



\- temporal coverage

\- granularity

\- missingness

\- duplicate records

\- compatibility with the target variable

\- potential data leakage

\- usefulness for forecasting



Dataset-selection decisions are documented separately in:



`DATASET\_EVALUATION.md`



\---



\## 4. Forecasting Unit



The fundamental forecasting unit is:



> Store × Product Family × Date



Each observation represents the demand for a particular product family at a particular store on a particular date.



The target variable is:



`y = sales`



The forecasting problem is therefore formulated as a supervised temporal regression problem.



For a given store-product combination, the model uses information available up to time `t` to predict demand at a future time `t+h`.



\---



\## 5. Data Validation



Before modelling, the dataset is checked for structural and temporal consistency.



\### 5.1 Schema validation



The following are verified:



\- expected columns exist

\- numerical variables have appropriate types

\- date columns are correctly parsed

\- categorical identifiers are consistent

\- target variable is numeric



\### 5.2 Duplicate detection



Duplicate observations are checked at the expected forecasting grain:



`date × store\_nbr × family`



Unexpected duplicates are investigated before modelling.



\### 5.3 Missing-value analysis



Missing values are profiled by:



\- column

\- time period

\- store

\- product family



Missingness is not automatically imputed without determining whether it represents:



\- genuinely missing information

\- zero demand

\- unavailable observations

\- structural absence



\### 5.4 Temporal continuity



The date range and temporal continuity of the dataset are examined.



For each store-product combination, the project evaluates:



\- earliest observation

\- latest observation

\- number of observations

\- missing dates

\- active forecasting period



This is important because forecasting models assume that temporal ordering is meaningful.



\---



\## 6. Exploratory Time-Series Analysis



Exploratory analysis is performed before model training to understand the demand-generating process.



The analysis investigates:



\### 6.1 Overall demand



\- daily total sales

\- weekly demand patterns

\- monthly demand patterns

\- long-term trends



\### 6.2 Store-level behaviour



Demand is compared across stores to identify:



\- high-demand stores

\- low-demand stores

\- differences in demand scale

\- temporal differences



\### 6.3 Product-family behaviour



Product families are analysed for:



\- average demand

\- demand variability

\- intermittency

\- seasonality

\- trend behaviour



\### 6.4 Seasonality



Potential periodic patterns are investigated at:



\- day-of-week level

\- weekly level

\- monthly level

\- yearly level



\### 6.5 Promotion effects



The relationship between promotions and sales is investigated.



Promotion features are treated as potentially predictive variables rather than assuming a causal relationship.



\---



\## 7. Temporal Data Splitting



Random train-test splitting is not used for the forecasting task.



The data is divided chronologically so that future observations are never used to train a model that predicts the past.



The general structure is:



```text

Past -------------------------> Future



|--------- Training ----------|---- Validation ----|---- Test ----|

