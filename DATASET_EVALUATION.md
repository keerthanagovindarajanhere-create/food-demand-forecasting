# FOODFLOW — Dataset Evaluation

## 1. Purpose

FOODFLOW requires a dataset that supports realistic food-demand forecasting while remaining practical for experimentation, reproducibility, and model comparison.

This document records the dataset evaluation process, selection criteria, candidate datasets, limitations, and the resulting dataset strategy.

The evaluation considers both forecasting suitability and engineering feasibility. Dataset selection is therefore treated as a methodological decision rather than simply choosing the largest available dataset.

---

## 2. Dataset Selection Criteria

Candidate datasets are evaluated using the following criteria:

| Criterion | Description |
|---|---|
| Direct demand target | Whether the dataset directly provides a quantity suitable for forecasting |
| Food/grocery relevance | Relevance to food, meals, grocery products, or retail demand |
| Temporal granularity | Daily or weekly observations and suitability for temporal forecasting |
| Historical depth | Length of available historical observations |
| Number of demand series | Number of products, stores, or product-location combinations |
| Product information | Availability of product/category attributes |
| Location information | Availability of store, region, or fulfillment-center information |
| Price information | Availability of price-related variables |
| Promotion information | Availability of promotional variables |
| Calendar information | Availability of holidays, events, or calendar variables |
| External variables | Availability of contextual variables such as weather, oil price, or events |
| Hierarchical structure | Support for product/store/category hierarchies |
| Computational feasibility | Practicality for local development and experimentation |
| Benchmark maturity | Availability of established forecasting literature or benchmarks |
| Reproducibility | Ease of obtaining and processing the dataset |
| FOODFLOW alignment | Overall compatibility with the project's objectives |

---

# 3. Candidate Dataset Pool

The initial candidate pool consists of:

1. Genpact Food Demand Forecasting
2. M5 Forecasting — Accuracy
3. Corporación Favorita Grocery Sales Forecasting
4. Store Sales — Time Series Forecasting
5. dunnhumby Complete Journey
6. Instacart Market Basket Analysis

These datasets are not treated as interchangeable. Some provide direct demand targets, while others are primarily useful for customer-behavior or supplementary research.

---

# 4. Dataset Evaluation

## 4.1 Genpact Food Demand Forecasting

### Overview

The Genpact Food Demand Forecasting dataset is a food-service demand forecasting dataset containing weekly demand observations for meals across fulfillment centers.

The commonly distributed version contains:

- Historical observations across 145 weeks
- 77 fulfillment centers
- 51 meals
- Approximately 456K training observations
- Approximately 33K test observations
- Weekly demand observations

### Main files

| File | Purpose |
|---|---|
| `train.csv` | Historical demand observations |
| `test.csv` | Forecasting instances |
| `meal_info.csv` | Meal category and cuisine information |
| `fulfilment_center_info.csv` | Fulfillment-center information |

### Important variables

`train.csv` contains variables such as:

- `week`
- `center_id`
- `meal_id`
- `checkout_price`
- `base_price`
- `emailer_for_promotion`
- `homepage_featured`
- `num_orders`

The primary forecasting target is `num_orders`.

### Strengths

- Direct food-demand target
- Clear forecasting formulation
- Manageable computational size
- Product/meal information
- Fulfillment-center information
- Price and promotion variables
- Suitable for rapid experimentation

### Limitations

- Weekly rather than daily granularity
- Relatively short historical period
- Limited external contextual variables
- Anonymized operational geography
- Competition-oriented dataset formulation

### FOODFLOW relevance

This dataset is useful as a lightweight food-demand benchmark and for validating whether the forecasting pipeline generalizes beyond grocery retail.

---

## 4.2 M5 Forecasting — Accuracy

### Overview

The M5 dataset is a large-scale retail forecasting benchmark based on Walmart sales.

It contains daily sales for thousands of item-store combinations across stores in California, Texas, and Wisconsin.

### Main data components

| File | Information |
|---|---|
| `sales_train_validation.csv` | Historical unit sales |
| `sales_train_evaluation.csv` | Extended historical sales |
| `calendar.csv` | Calendar, event, and SNAP information |
| `sell_prices.csv` | Product-store weekly prices |
| `sample_submission.csv` | Competition submission structure |

### Characteristics

The dataset contains:

- 30,490 item-store time series
- 10 stores
- 3 states
- 3,049 items
- Daily observations
- Approximately five years of historical sales
- Product/category/store hierarchy

The original forecasting task used a 28-day forecast horizon.

### Strengths

- Large number of demand series
- Daily forecasting
- Long historical period
- Strong hierarchical structure
- Product and store information
- Calendar and event variables
- SNAP information
- Price information
- Highly established forecasting benchmark

### Limitations

- General retail rather than food-only
- Large computational requirements
- Requires careful reshaping from wide to temporal format
- Hierarchical structure increases implementation complexity
- Competition-specific dataset formulation

### FOODFLOW relevance

M5 provides an important external benchmark for testing whether FOODFLOW's methodology works on large-scale hierarchical retail demand data.

---

## 4.3 Corporación Favorita Grocery Sales Forecasting

### Overview

The Corporación Favorita dataset contains grocery sales data from an Ecuadorian retailer.

The original competition dataset provides daily sales information across multiple stores and products together with several contextual data sources.

### Main files

| File | Information |
|---|---|
| `train.csv` | Historical sales |
| `test.csv` | Forecasting instances |
| `stores.csv` | Store metadata |
| `items.csv` | Product metadata |
| `transactions.csv` | Store-level transactions |
| `oil.csv` | Daily oil prices |
| `holidays_events.csv` | Holidays and events |

### Characteristics

The original dataset contains:

- More than 100 million historical observations
- 54 stores
- Thousands of products
- Daily observations
- Approximately four years of historical data
- Product and store metadata

The forecasting target is `unit_sales`.

### Important characteristics

Product information includes attributes such as:

- Product family
- Product class
- Perishability

Store information includes:

- City
- State
- Store type
- Cluster

Additional variables include:

- Promotions
- Transactions
- Oil prices
- Holidays and events

Sales may also contain returns, meaning negative values require appropriate treatment during preprocessing.

### Strengths

- Strong grocery relevance
- Daily demand forecasting
- Large number of demand series
- Long historical period
- Product hierarchy
- Store hierarchy
- Promotion information
- Calendar and event information
- Additional external variables
- Perishability information

### Limitations

- Extremely large original dataset
- Requires memory-efficient processing
- Sparse store-product-date combinations require careful interpretation
- Missing promotion values require investigation
- Negative sales require treatment
- Feature generation can become computationally expensive

### FOODFLOW relevance

Favorita provides a strong environment for evaluating contextual demand forecasting because demand can be modeled together with product, store, promotion, calendar, and external variables.

---

## 4.4 Store Sales — Time Series Forecasting

### Overview

Store Sales — Time Series Forecasting is a later Favorita-based forecasting dataset designed around the same Ecuadorian grocery-retail environment.

It provides a more practical development-scale alternative to working immediately with the full original Favorita dataset.

### Important characteristics

The dataset contains daily sales at store-product-family level and includes supporting information related to:

- Stores
- Products
- Promotions
- Transactions
- Holidays and events
- Oil prices

### Strengths

- Grocery-specific
- Daily forecasting
- Direct sales target
- Store and product structure
- Promotional information
- Calendar/event information
- External contextual variables
- More practical for initial development than the original Favorita dataset

### Limitations

- Still requires temporal feature engineering
- Competition-derived formulation
- Large enough to require efficient preprocessing
- Dataset structure differs from the original Favorita formulation

### FOODFLOW relevance

This dataset is selected as the primary development environment because it provides a strong balance between realistic grocery demand forecasting and manageable experimentation.

---

## 4.5 dunnhumby Complete Journey

### Overview

The Complete Journey dataset from dunnhumby is primarily a customer-behavior and grocery-transaction dataset.

It contains information related to:

- Households
- Transactions
- Products
- Customer attributes
- Marketing contacts
- Campaigns

The dataset covers approximately two years of customer activity.

### Strengths

- Grocery-domain data
- Customer-level information
- Transaction history
- Marketing information
- Product information
- Useful for studying demand behavior and promotion response

### Limitations

Unlike the main forecasting datasets, it does not provide a predefined operational demand-forecasting target and horizon.

A forecasting task would therefore need to be constructed through aggregation, for example:

- Daily product demand
- Weekly product demand
- Category demand
- Household-product demand

### FOODFLOW relevance

Useful for future research into customer-aware and promotion-aware demand forecasting, but not selected for the initial core pipeline because substantial target construction is required.

---

## 4.6 Instacart Market Basket Analysis

### Overview

The Instacart Market Basket dataset contains online grocery ordering behavior.

Its main components include:

- Orders
- Products
- Aisles
- Departments
- Products included in orders

Temporal variables include:

- Order sequence
- Day of week
- Order hour
- Days since prior order

### Strengths

- Large grocery behavior dataset
- Product hierarchy
- Customer ordering behavior
- Reorder information
- Useful temporal behavioral variables

### Limitations

The dataset does not directly provide physical unit-sales demand.

It also lacks:

- Store-level operational demand
- Direct price information
- Direct promotion information
- A predefined demand-forecasting target

A demand proxy would need to be constructed from order behavior.

### FOODFLOW relevance

Potentially valuable for future research into behavioral demand and reorder prediction, but not suitable as the initial core forecasting dataset.

---

# 5. Comparative Evaluation

| Criterion | Genpact | M5 | Favorita Store Sales | Favorita Original | dunnhumby | Instacart |
|---|---|---|---|---|---|---|
| Direct demand target | Strong | Strong | Strong | Strong | Partial | Weak |
| Food/grocery relevance | Strong | Partial | Strong | Strong | Strong | Strong |
| Temporal granularity | Weekly | Daily | Daily | Daily | Transaction | Transaction |
| Historical depth | Moderate | Strong | Strong | Strong | Moderate | Limited |
| Multiple demand series | Strong | Very strong | Strong | Very strong | Flexible | Flexible |
| Product information | Strong | Strong | Strong | Strong | Strong | Strong |
| Store/location information | Strong | Strong | Strong | Strong | Limited | Weak |
| Price information | Strong | Strong | Available/limited | Available | Available | Weak |
| Promotions | Strong | Strong | Strong | Strong | Strong | Limited |
| Calendar/events | Limited | Strong | Strong | Strong | Limited | Limited |
| External variables | Limited | Moderate | Strong | Strong | Limited | Limited |
| Perishability | Weak | Weak | Available | Strong | Not central | Weak |
| Hierarchical forecasting | Moderate | Strong | Strong | Strong | Possible | Limited |
| Computational feasibility | Strong | Moderate | Strong | Weak | Moderate | Moderate |
| Target-construction effort | Low | Low | Low | Low | High | High |
| Benchmark maturity | Strong | Very strong | Strong | Strong | Weak | Weak |
| FOODFLOW alignment | Strong | Strong | Very strong | Very strong | Moderate | Moderate |

The comparison is intended to identify dataset roles rather than produce a universal ranking.

---

# 6. Dataset-Specific Risks

## Genpact

- Limited historical duration
- Weekly temporal resolution
- Limited external context
- Potential sparsity across center-meal combinations
- Forecasting formulation is competition-oriented

## M5

- Large computational footprint
- General retail rather than food-only
- Hierarchical forecasting complexity
- Wide-format historical sales require transformation
- Large number of time series increases experiment cost

## Favorita Store Sales

- Requires temporal feature engineering
- Large enough to require efficient preprocessing
- Competition-derived formulation
- Store/product structure must be preserved during splitting

## Favorita Original

- Very large dataset
- Memory-intensive preprocessing
- Sparse store-product-date observations
- Negative sales require treatment
- Missing values require explicit investigation
- Large-scale feature generation can be expensive

## dunnhumby

- No predefined demand target
- Aggregation strategy must be designed
- Forecasting horizon must be defined
- Customer-level modeling can expand project scope

## Instacart

- No direct unit-sales target
- Reorders are not equivalent to physical demand
- No store-level operational demand
- Target construction is required
- Primarily behavioral rather than operational forecasting data

---

# 7. Dataset Strategy

FOODFLOW will use a staged dataset strategy rather than depending on a single dataset.

### Primary development dataset

**Store Sales — Time Series Forecasting**

Used for:

- Pipeline development
- Feature engineering
- Baseline models
- Model comparison
- Time-series validation
- Error analysis

### Large-scale validation

**Original Corporación Favorita Grocery Sales Forecasting**

Used after the core pipeline is stable to evaluate:

- Scalability
- Memory-efficient preprocessing
- Large-scale model training
- Generalization across a larger demand space

### External benchmark

**M5 Forecasting — Accuracy**

Used to evaluate whether the developed methodology transfers to a well-established retail forecasting benchmark.

### Lightweight food-demand comparison

**Genpact Food Demand Forecasting**

Used to test the pipeline on a different food-demand formulation with weekly observations and fulfillment-center/meal structure.

### Future research datasets

**dunnhumby Complete Journey** and **Instacart Market Basket Analysis** may be incorporated later for customer-aware or behavioral demand research.

---

# 8. Recommended Initial Dataset

The initial FOODFLOW pipeline will be developed using:

> **Store Sales — Time Series Forecasting**

The decision is based on the combination of:

- Direct sales target
- Grocery-specific domain
- Daily temporal resolution
- Multiple stores
- Product information
- Promotions
- Calendar/events
- External contextual variables
- Practical development scale
- Compatibility with time-series feature engineering
- Ability to scale the methodology to the original Favorita dataset

This dataset provides a practical environment for developing the forecasting methodology before moving to larger-scale validation.

---

# 9. Planned Dataset Progression

The project will follow the progression:

```text
Store Sales
    ↓
Baseline forecasting pipeline
    ↓
Feature engineering
    ↓
Time-series validation
    ↓
Model comparison
    ↓
Error analysis
    ↓
Original Favorita
    ↓
Large-scale validation
    ↓
M5
    ↓
External benchmark
    ↓
Genpact
    ↓
Cross-domain food-demand validation