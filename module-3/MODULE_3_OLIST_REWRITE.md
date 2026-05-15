---
title: "Module 3 — Data Preparation & Feature Engineering (Olist Edition)"
subtitle: "Cleaning real-world e-commerce data, end-to-end"
author: "BePro AI/ML Mentorship Program"
date: "2026-05-04"
---

# Module 3 — Data Preparation & Feature Engineering

| | |
|---|---|
| Module duration | 12 hours (6 classes × 2 h) |
| Weeks | 5–6 |
| Pre-requisite | Module 2 — Python and Data Foundations |
| Dataset | **Olist Brazilian E-Commerce** (raw, 9 CSVs) — locked for M3-M8 |
| Module deliverable | One Parquet file: `olist_clean.parquet` (used by M4-M8) |

---

## Module overview

The 80% of every ML project that nobody writes blog posts about. This module turns 9 messy CSVs into one clean analytical table. By the end, every student has produced **`olist_clean.parquet`** — the file that every module from M4 onwards will load as input.

This is the first module where every student is working on the **same final project**. The work compounds. Don't lose your output file.

## Learning outcomes

By the end of Module 3 a student can:

| LO | Statement | Cognitive level |
|---|---|---|
| LO1 | Identify and handle missing values, duplicates, and outliers in messy real-world data | Apply |
| LO2 | Convert categorical, datetime, and continuous columns into model-ready features | Apply |
| LO3 | Engineer informative features from raw fields (dates, ratios, geographic distance) | Create |
| LO4 | Select features using statistical and model-based criteria | Analyse |
| LO5 | Build reproducible preprocessing pipelines with `scikit-learn` | Apply |
| LO6 | Avoid data leakage between training and test data | Analyse |

## Class structure (locked)

| # | Class title | Olist focus |
|---|---|---|
| 3-1 | Data Quality and Cleaning | Inspect raw `olist_orders_dataset`, handle missing delivery dates, fix mixed timestamp formats |
| 3-2 | Data Encoding and Transformation | Encode `customer_state`, `payment_type`, scale `freight_value`, log-transform skewed prices |
| 3-3 | Feature Engineering | Build `delivery_days`, `is_late`, `distance_km` (Haversine), date-derived features |
| 3-4 | Feature Selection | Drop redundant time columns, correlation analysis on engineered features |
| 3-5 | Data Pipelines | Wrap everything in a `sklearn.Pipeline + ColumnTransformer` |
| 3-6 | Lab — End-to-End | Run the pipeline on full data, ship `olist_clean.parquet` |

---

# Class 3-1 — Data Quality and Cleaning

> **Today: load the raw Olist orders + customers tables, find every kind of dirt, fix it.**

## Why this matters today

The classifier you'll train in Module 4 (predicting late deliveries) is only as good as the cleaning you do today. Garbage in → garbage out. The dataset you'll work with for the next 6 weeks lands on your laptop today.

## Section 1 — The 80% rule

You will spend **80% of your time as an ML engineer** cleaning, joining, and shaping data. The remaining 20% is the modelling people brag about. Today is the 80% — and it's the part that decides whether your model actually works.

A clean dataset is one where:
- Every column has the right `dtype`
- Missing values are explicit (`NaN`), not hidden as `'?'` or empty strings
- Each row represents one well-defined unit (here: one order)
- Duplicates are gone
- Outliers are identified and either kept, capped, or removed — *with a reason written down*

## Section 2 — Loading the raw Olist data

Download the dataset from Kaggle ([https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)). You get 9 CSVs. Today we touch only 2: orders and customers.

```python
import pandas as pd

orders = pd.read_csv('olist_orders_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')

print(orders.shape)         # (99441, 8)
print(customers.shape)      # (99441, 5)
```

### What to check first (the 5 commands from Module 2)

```python
orders.head()
orders.shape
orders.dtypes
orders.isna().sum()
orders.describe(include='all')
```

Run each. **Read the output before moving on.** Notes you should make:

- `order_purchase_timestamp` is `object` (string), not `datetime64`.
- `order_delivered_customer_date` has roughly 3,000 missing values. **These are interesting** — they're probably orders that never arrived.
- `order_status` has 8 unique values; you'll need to decide which ones to keep.

## Section 3 — Fix the dtypes (parse the dates)

```python
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c], errors='coerce')
```

`errors='coerce'` is critical: any unparseable date becomes `NaT` (the datetime equivalent of `NaN`) instead of crashing. **Always use it on real-world dates.**

## Section 4 — Missing values: investigate before fixing

```python
orders.isna().mean().sort_values(ascending=False) * 100
```

You'll see roughly:
- `order_delivered_customer_date` — 2.98% missing
- `order_delivered_carrier_date` — 1.78% missing
- `order_approved_at` — 0.16% missing

The lecture's number-one rule applies: **investigate why before deciding what to do.**

```python
orders[orders['order_delivered_customer_date'].isna()]['order_status'].value_counts()
```

You'll find the missing-delivery rows are mostly `canceled`, `unavailable`, or `shipped` (still in transit). Those rows should NOT be filled with mean/median — they're real signal.

**Decision (write this in your notebook):**
> "Rows with missing `order_delivered_customer_date` are kept. They represent orders that never reached the customer. We'll filter them out of the M4 modelling step (`is_late` target), but keep them in the analytical table for the segmentation work in M5."

## Section 5 — Duplicates

```python
orders.duplicated().sum()      # should be 0 in Olist orders, but always check
orders['order_id'].nunique() == len(orders)   # should be True
```

Olist is well-curated, so duplicates are rare. But the **habit** of checking is what we're building.

## Section 6 — Outliers (preview)

We don't drop outliers in M3-1 yet — first we need to merge in `order_items` to see prices and freight. That's M3-3 work. But you should already be suspicious of:

- Extremely long delivery times (some orders show 200+ days)
- `order_estimated_delivery_date` before `order_purchase_timestamp` (data entry error)

Note them down. We'll handle them in 3-3.

## Quick Check

1. What happens if you call `pd.to_datetime` on a malformed date string without `errors='coerce'`?
2. Why is `order_delivered_customer_date` being NaN sometimes a *signal*, not a bug to impute away?
3. The Olist dataset has 99,441 orders. After today's cleaning, how many rows do you expect to keep for the M4 modelling step?

## Today's deliverable (homework)

### Bronze (Type 2) version
- Open the provided notebook `m3c1_bronze.ipynb` (in the cohort repo at `module-3/class_1/m3c1_bronze.ipynb`).
- 5 TODO cells: parse dates, identify missing values, investigate the canceled-orders pattern, decide your strategy in markdown, save intermediate `orders_step1.parquet`.
- Each TODO has a hint comment in plain English.
- Submit notebook to `module-3/class_1/submissions/<YourName>/`.

### Gold (Type 1) version
- Open `m3c1_gold.ipynb` (skeleton only — no code).
- Same business goal: produce `orders_step1.parquet`.
- You decide the order of operations, the missing-value strategy, and write a **half-page rationale** of every decision.
- Bonus: also clean `customers_step1.parquet` and merge them on `customer_id`.

---

# Class 3-2 — Data Encoding and Transformation

> **Today: turn the categories into numbers, the skewed columns into model-friendly shapes.**

## Why this matters today

Models don't read text. `payment_type='credit_card'` is a string — useless to logistic regression unless we convert it. Similarly, `freight_value` ranges from 0 to 400+ Brazilian reais — heavy right tail — log scale lifts the shape into something a linear model can fit.

## Section 1 — Categorical encoding strategies

Three encoding choices, three trade-offs:

| Method | When | Olist example |
|---|---|---|
| **Label encoding** | Ordinal categories (rank order matters) | `review_score` 1-5 — already numeric, leave alone |
| **One-hot** | Nominal categories, low cardinality (≤ ~20 levels) | `payment_type` (5 levels: credit_card, boleto, voucher, debit_card, not_defined) |
| **Target / mean encoding** | Nominal, high cardinality (>20 levels) | `customer_state` (27 levels — borderline; you can do either) |

```python
# One-hot
payment_dummies = pd.get_dummies(items_payments['payment_type'], prefix='pay')

# Target encoding (CAREFUL — leakage risk; use only on training fold)
state_late_rate = train_df.groupby('customer_state')['is_late'].mean()
df['state_target_enc'] = df['customer_state'].map(state_late_rate)
```

**Trap:** target encoding done on the full dataset before train/test split = leakage. Always inside cross-validation folds.

## Section 2 — Scaling numerical columns

Tree models don't need scaling. Linear models, neural nets, and distance-based models do.

| Scaler | What it does | When |
|---|---|---|
| `StandardScaler` | mean 0, std 1 | Most numeric, near-Gaussian |
| `MinMaxScaler` | rescale to [0, 1] | Bounded inputs, neural nets with sigmoid |
| `RobustScaler` | uses median/IQR | Heavy outliers present |

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['freight_value', 'price']] = scaler.fit_transform(df[['freight_value', 'price']])
```

## Section 3 — Non-linear transforms

Olist's `freight_value` is right-skewed (most freight is 5-30 reais; a long tail goes to 400+). Linear models struggle to fit a heavy tail.

```python
import numpy as np
df['log_freight'] = np.log1p(df['freight_value'])
```

`log1p(x) = log(1 + x)` — handles `x = 0` cleanly. Use it whenever the column is right-skewed and non-negative.

After log-transform, plot a histogram. The shape should be much closer to a bell curve.

## Section 4 — Putting it together for Olist

We're combining `orders + order_items + payments` for today's exercise.

```python
items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')

# Per-order totals (one item-row per order line — collapse to one row per order)
order_totals = items.groupby('order_id').agg(
    num_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
).reset_index()

# Primary payment method per order (use the largest payment value)
primary_payment = (payments.sort_values('payment_value', ascending=False)
                            .drop_duplicates('order_id')[['order_id', 'payment_type']])
```

Now merge into the orders frame from Class 3-1. (Saving the merged frame as `orders_step2.parquet` is today's deliverable.)

## Quick Check

1. You have a column `customer_state` with 27 unique Brazilian states. One-hot encoding produces 27 new columns. Why might that be too many?
2. Why does log-transforming `freight_value` help linear regression but make no difference to a Random Forest?
3. What's the leakage trap with target encoding, and how does cross-validation prevent it?

## Today's deliverable (homework)

- Bronze: notebook `m3c2_bronze.ipynb` with TODOs for one-hot encoding payment_type and log-transforming freight.
- Gold: same dataset, decide your encoding strategy for state and justify in markdown. Output `orders_step2.parquet`.

---

# Class 3-3 — Feature Engineering

> **Today: build the features that actually predict late deliveries.**

## Why this matters today

The columns you have right now (price, freight, payment_type) are necessary but not sufficient. The features that predict `is_late` are *derived*: how far did the parcel travel? How many days from purchase to estimate? Was the order placed on a weekend?

This is the class where students who've been quiet start having ideas. Good feature engineering beats fancy models on tabular data, every time.

## Section 1 — The features we need

**Target feature (the headline of the whole compound project):**

```python
df['is_late'] = (
    df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
).astype(int)
```

Filter out rows where `order_delivered_customer_date` is NaN — those orders never arrived. They're not "late," they're "missing." We model them separately or drop them; for M4 we drop.

**Time-based features:**

```python
df['delivery_days'] = (
    df['order_delivered_customer_date'] - df['order_purchase_timestamp']
).dt.days
df['estimate_days'] = (
    df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
).dt.days
df['days_buffer'] = df['estimate_days'] - df['delivery_days']  # negative = late
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['is_weekend'] = (df['purchase_dayofweek'] >= 5).astype(int)
```

**Geographic feature (the Haversine):**

```python
import numpy as np

R = 6371  # Earth radius in km

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# Aggregate geolocation by zip prefix (faster than per-row joins)
geo = pd.read_csv('olist_geolocation_dataset.csv')
geo_lookup = geo.groupby('geolocation_zip_code_prefix').agg(
    lat=('geolocation_lat', 'mean'),
    lon=('geolocation_lng', 'mean'),
).reset_index()

# Join customer + seller lat/lon, compute distance, all vectorized
# (...full merge code in the notebook...)

df['distance_km'] = haversine(
    df['cust_lat'].values, df['cust_lon'].values,
    df['seller_lat'].values, df['seller_lon'].values
)
```

This is the most powerful single feature in the dataset. Long Brazilian distances + late deliveries = a real signal.

## Section 2 — Real-world feature patterns

| Pattern | Olist application | Why it works |
|---|---|---|
| **Date decomposition** | hour, day-of-week, month, is_weekend | Captures cyclical demand & shipping cycles |
| **Ratios** | `freight / price` (freight share) | Normalises across product price ranges |
| **Aggregates** | `seller_avg_delivery_days` | Sellers have habits — past behaviour predicts future |
| **Distance** | Haversine customer ↔ seller | The biggest single predictor in shipping data |
| **Interactions** | `distance × is_weekend` | Joint conditions sometimes matter more than either alone |

## Section 3 — Feature engineering hierarchy

1. **Direct features** — already in the data (price, freight)
2. **Derived from one column** — `log_freight`, `purchase_hour`
3. **Derived from two columns** — `delivery_days`, `freight_per_item`
4. **Derived from joins** — `distance_km`, `seller_avg_delivery_days`
5. **Domain-knowledge features** — `is_holiday_brazil`, `is_pre_christmas`

Type 1 students should aim for 4-5. Type 2 students should be able to explain levels 1-3 and have at least one level-4 feature in their notebook.

## Quick Check

1. Why do we filter out rows where `order_delivered_customer_date` is NaN before computing `is_late`?
2. Haversine vs Euclidean — why is Haversine the right choice for Olist's distances?
3. Name two features you could engineer from `seller_id` alone (joining order history).

## Today's deliverable

- Bronze: notebook with date features + simple `is_late` target. Output `orders_step3.parquet`.
- Gold: include Haversine distance_km + at least 2 self-engineered features. Justify each in markdown.

---

# Class 3-4 — Feature Selection

> **Today: not every feature you engineered is worth keeping.**

## Why this matters today

You probably engineered 20+ features in 3-3. Some are redundant (`delivery_days` and `days_buffer` are perfectly anti-correlated). Some are useless (`purchase_month` may have near-zero correlation with `is_late`). Models trained on 50 features overfit and run slower. **Pick the 8-12 features that actually carry signal.**

## Section 1 — Three families of selection

| Family | Methods | When |
|---|---|---|
| **Filter** | Variance threshold, correlation matrix, mutual information | Fast, no model needed |
| **Wrapper** | RFE (Recursive Feature Elimination), forward/backward search | Slower but accurate; needs a model |
| **Embedded** | L1 regression, tree feature importance | Free output of the model itself |

For tabular ML on Olist, **start with filter, end with embedded**.

## Section 2 — Filter methods

```python
# 1. Drop near-zero-variance columns
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X_filtered = sel.fit_transform(X_numeric)

# 2. Drop highly-correlated pairs (keep one of any pair with |r| > 0.9)
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X = X.drop(columns=to_drop)

# 3. Mutual information vs target (works for non-linear relationships)
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X, y)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
```

## Section 3 — Embedded selection (free with your model)

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(15))
```

The top 15 features the random forest uses *are* your selected features. **Drop the rest** unless you have domain reason to keep them.

## Section 4 — Dimensionality reduction (introduction)

When you genuinely have hundreds of features (e.g. one-hot-encoded high-cardinality categoricals), use **PCA** to compress them into ~20 components.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_.cumsum())
```

For Olist with ~15 hand-engineered features, you don't need PCA. We'll see it again in M5.

## Quick Check

1. Two features have correlation 0.97. Why drop one?
2. Filter selection vs embedded selection — name one advantage of each.
3. What does PCA optimise for, and why is that not the same as "best for prediction"?

## Today's deliverable

- Bronze: drop columns with correlation > 0.9 + variance < 0.01. Save `orders_step4.parquet`.
- Gold: full pipeline (filter → mutual info → embedded). Provide a final feature list of 8-12 features with one-line justification each.

---

# Class 3-5 — Data Pipelines

> **Today: turn your messy notebook code into a reproducible pipeline.**

## Why this matters today

So far your cleaning lives in scattered notebook cells. When M4 starts and you load `olist_clean.parquet`, you'll never re-run that mess. Worse: in production, when **new** orders arrive, you need to apply the *exact same* transformations. A pipeline guarantees that.

## Section 1 — The two killers a pipeline prevents

1. **Data leakage** — fitting a scaler on the full dataset before splitting → test data has seen training data's mean/std. Pipelines fit only on training inside the right CV fold.
2. **Drift between train and serve** — manual cleaning produces inconsistent results between training time and inference time.

## Section 2 — `Pipeline` and `ColumnTransformer`

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ['delivery_days', 'distance_km', 'log_freight', 'num_items']
categorical_features = ['payment_type', 'customer_state']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# Use it
preprocessor.fit(X_train)
X_train_clean = preprocessor.transform(X_train)
X_test_clean = preprocessor.transform(X_test)
```

The pipeline is now a single object you can `pickle` and ship to production.

## Section 3 — Saving artifacts properly

```python
import joblib
joblib.dump(preprocessor, 'olist_preprocessor.pkl')

# Reload later
preprocessor = joblib.load('olist_preprocessor.pkl')
```

Save the cleaned dataset as Parquet (much faster than CSV, preserves dtypes):

```python
df_clean.to_parquet('olist_clean.parquet', index=False)
```

## Section 4 — Reproducibility checklist

Before you submit, check:

- [ ] One file produces `olist_clean.parquet` from the 9 raw CSVs (no manual edits)
- [ ] All random seeds set (np.random.seed, random_state in scikit-learn objects)
- [ ] Cleaning decisions documented in markdown cells
- [ ] Train/test split is deterministic (random_state=42)
- [ ] `requirements.txt` lists pandas, numpy, scikit-learn, pyarrow versions

## Quick Check

1. Why fit the scaler on training data only, not the full dataset?
2. What does `OneHotEncoder(handle_unknown='ignore')` do, and why is it important for production?
3. Why save as Parquet rather than CSV?

## Today's deliverable

- Bronze: complete the pipeline scaffold provided. Test that `pipeline.fit_transform(X_train)` works.
- Gold: end-to-end script `module_3_clean.py` that takes the 9 raw CSVs and produces `olist_clean.parquet`. Run it from the command line.

---

# Class 3-6 — Lab: End-to-End Olist Data Prep

> **The Module 3 capstone. 90 minutes. You produce the file every other module will use.**

## Brief

Take the 9 raw Olist CSVs as input. Produce one file: **`olist_clean.parquet`**, one row per order, with the columns specified below. This file is the input to Module 4.

## Required output schema

| Column | Type | Source |
|---|---|---|
| `order_id` | string | orders |
| `customer_unique_id` | string | customers |
| `customer_state` | string | customers |
| `seller_state` | string | sellers (most-frequent if multiple) |
| `purchase_year` | int | orders |
| `purchase_month` | int | orders |
| `purchase_dayofweek` | int | orders |
| `purchase_hour` | int | orders |
| `is_weekend` | int (0/1) | engineered |
| `num_items` | int | order_items aggregate |
| `total_price` | float | order_items aggregate |
| `total_freight` | float | order_items aggregate |
| `log_freight` | float | engineered |
| `payment_type` | string | payments (primary) |
| `payment_installments` | int | payments (primary) |
| `distance_km` | float | Haversine on geolocation |
| `delivery_days` | float | engineered |
| `estimate_days` | float | engineered |
| `is_late` | int (0/1) | TARGET |
| `review_score` | int 1-5 | reviews |
| `review_comment_message` | string | reviews (kept for M7) |

**Filter to:** orders with `order_status` in `['delivered', 'invoiced', 'shipped', 'approved']`. Drop canceled and unavailable for the M4 modelling task. Keep rows with NaN delivered date — they go in a separate `orders_undelivered.parquet`.

## Lab structure (90 min)

### Stage 1 — Load (10 min)
Load all 9 CSVs. Print shapes and confirm row counts match the dataset README.

### Stage 2 — Clean orders (15 min)
Parse dates with `errors='coerce'`. Filter to the 4 status values above. Investigate and document the missing-delivery-date rows.

### Stage 3 — Aggregate items + payments (15 min)
Collapse `order_items` to one row per order (sum prices, sum freights, count items). Pick primary payment method per order.

### Stage 4 — Geographic merge + Haversine (20 min)
Aggregate geolocation by zip prefix. Merge customer + seller lat/lon. Compute `distance_km` vectorized.

### Stage 5 — Date features + target (10 min)
Engineer `purchase_*` columns. Compute `delivery_days`, `estimate_days`, `is_late`.

### Stage 6 — Final merge + save (10 min)
Join everything on `order_id`. Validate the schema matches the table above. Save `olist_clean.parquet`.

### Stage 7 — Findings (10 min)
Markdown cell at the bottom:
- Total rows in the final table
- `is_late` rate (should be roughly 7%)
- 3 things you noticed during cleaning
- 1 question for the M4 mentor

## Submission

Push your notebook AND the produced `olist_clean.parquet` to:
```
module-3/class_6/submissions/<TeamName>/
```

Mentor will use the *largest cohort's* most-correct submission as the canonical M4 input.

## Grading rubric (100 points)

| Component | Weight |
|---|---|
| Schema compliance (all 21 columns, correct dtypes) | 25 % |
| Cleaning decisions documented in markdown | 15 % |
| Pipeline / script is one-shot reproducible | 20 % |
| Haversine implementation correctness (within 1km of `geopy`) | 10 % |
| `is_late` rate within 5-10% (sanity check) | 10 % |
| Findings cell substantive | 10 % |
| Code quality + comments | 10 % |
