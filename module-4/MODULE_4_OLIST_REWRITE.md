---
title: "Module 4 — Supervised Learning (Olist Edition)"
subtitle: "Train your first models on the cleaned Olist data"
author: "BePro AI/ML Mentorship Program"
date: "2026-05-04"
---

# Module 4 — Supervised Learning Algorithms

| | |
|---|---|
| Module duration | 12 hours (6 classes × 2 h) |
| Weeks | 7–8 |
| Pre-requisite | Module 3 — produces `olist_clean.parquet` (every class loads it) |
| Module deliverable | `model_v1.pkl` — trained classifier predicting `is_late` + 1-page evaluation report |
| Headline target | `is_late` (binary, ~7% positive) |

---

## Module overview

Now that the data is clean, we predict. This module teaches the four classical supervised algorithms students will see in every interview: linear regression, logistic regression, decision trees, and random forests. Then we evaluate honestly (accuracy is a trap on imbalanced data) and avoid overfitting.

By the end of M4, every student has trained a real classifier on real data and shipped a `.pkl` file that the M5 segmentation step will load.

## Learning outcomes

| LO | Statement | Cognitive level |
|---|---|---|
| LO1 | Train and interpret linear and logistic regression models | Apply |
| LO2 | Train and tune tree-based ensembles (Decision Tree, Random Forest) | Apply |
| LO3 | Choose the correct evaluation metric for an imbalanced classification problem | Analyse |
| LO4 | Diagnose overfitting with cross-validation and combat it with regularisation | Apply |
| LO5 | Compare multiple models on the same dataset and recommend the right one | Evaluate |

## Class structure

| # | Title | Olist focus |
|---|---|---|
| 4-1 | Linear Regression | Predict `freight_value` from `distance_km` |
| 4-2 | Logistic Regression | First model for `is_late` |
| 4-3 | Decision Trees & Random Forests | Beat the logistic baseline |
| 4-4 | Model Evaluation | Why accuracy lies on Olist (~93% baseline by predicting "all on time") |
| 4-5 | Overfitting & Regularisation | Cross-validation, L1/L2, max_depth, early stopping |
| 4-6 | Lab — Build a Classifier | Ship `model_v1.pkl` |

---

# Class 4-1 — Linear Regression

> **Today: predict the freight cost of an order from the distance it has to travel. Why do we start here? Because freight follows distance almost linearly — it's the perfect intuition-builder.**

## Why this matters today

Linear regression is the simplest possible model. If you can't beat it on a dataset, you don't understand the dataset. We use it on Olist to set a baseline for *price prediction*, but the same logic powers every regression problem in your career.

## Section 1 — The simplest possible model

A linear model fits the equation:

```
freight = w₀ + w₁ · distance_km
```

We learn `w₀` (intercept) and `w₁` (slope) from data. No fancy math today — just two numbers per feature.

## Section 2 — The cost function

How does the model "learn"? It picks `w` values that **minimise the squared error**:

```
Loss = (1/N) · Σ (predicted_freight - actual_freight)²
```

Squaring penalises big mistakes more than small ones. This is the "OLS" (Ordinary Least Squares) cost function.

## Section 3 — Code on Olist

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_parquet('olist_clean.parquet')

# Filter to delivered rows with non-null freight + distance
df = df.dropna(subset=['total_freight', 'distance_km'])

X = df[['distance_km']]
y = df['total_freight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Slope (w1): {model.coef_[0]:.3f} R$/km")
print(f"Intercept (w0): {model.intercept_:.2f} R$")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f} R$")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

You'll see:
- Slope around 0.005 R$/km (5 reais per 1000 km)
- Intercept around 13 R$ (the base freight even for short distances)
- R² roughly 0.4 — distance explains 40% of freight variance. Not bad. The remaining 60% is package weight, seller policy, etc.

## Section 4 — Multi-feature regression

```python
X = df[['distance_km', 'total_price', 'num_items', 'log_freight']]
# (drop log_freight if predicting freight — circular!)
X = df[['distance_km', 'total_price', 'num_items']]

model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients tell you what each feature "buys" you in freight cost
for feat, w in zip(X.columns, model.coef_):
    print(f"{feat}: {w:+.4f}")
```

The signs of the coefficients tell a story. `distance_km` is positive (further = more freight). `num_items` is positive (more stuff = more freight). `total_price` is small (price doesn't strongly drive freight on Olist).

## Quick Check

1. The slope on `distance_km` is 0.005 R$/km. What does that mean in plain English?
2. Why is R² of 0.4 actually OK for freight prediction? When would 0.4 be terrible?
3. If you regressed `freight` on `[distance, log_freight]`, what would happen?

## Today's deliverable

### Bronze
- `m4c1_bronze.ipynb` provided. 5 TODOs: load parquet, plot freight vs distance, fit `LinearRegression`, print coefficients, write a 2-sentence interpretation.
- Submit notebook + a screenshot of your scatter plot to `module-4/class_1/submissions/<YourName>/`.

### Gold
- Same dataset, no code skeleton.
- Add 3 more numeric features. Compare the multi-feature R² to the single-feature R².
- Write a half-page rationale: which features add real signal, which don't?

---

# Class 4-2 — Logistic Regression

> **Today: predict `is_late` for every order. Yes/no, not a number.**

## Why this matters today

90% of business ML is binary classification: will this customer churn, will this email be spam, will this order be late. Logistic regression is the simplest model for this and the *first* thing you should try on any classification problem. If a fancy model can't beat logistic, your data isn't rich enough — fix the data, not the model.

## Section 1 — From regression to classification

Linear regression outputs any number. We need a probability between 0 and 1. The fix is the **sigmoid function**:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

It squashes any input into [0, 1]. Below 0.5 → predict 0 (on time). Above 0.5 → predict 1 (late). The threshold can be tuned (more in Class 4-4).

## Section 2 — On Olist

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_parquet('olist_clean.parquet').dropna(subset=['is_late'])

features = ['distance_km', 'log_freight', 'num_items', 'total_price',
            'estimate_days', 'is_weekend', 'purchase_hour']
X = df[features]
y = df['is_late']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

**Notice:**
- `stratify=y` keeps the same is_late rate in train and test splits. Always do this for imbalanced classification.
- `class_weight='balanced'` makes the model care about the rare class (late orders). Without this it would just predict "on time" for everything and "achieve" 93% accuracy — useless.

## Section 3 — Reading the coefficients

```python
coefs = pd.Series(model.coef_[0], index=features).sort_values(key=abs, ascending=False)
print(coefs)
```

Positive coefficient = pushes toward "late." Negative = pushes toward "on time." On Olist you'll see:
- `distance_km` positive (further → more late risk)
- `is_weekend` positive (weekends ship slower)
- `estimate_days` negative (longer estimates → less likely to miss them)

This is **interpretable AI**. You can hand these coefs to a business stakeholder.

## Section 4 — The Decision Boundary

For a 2-feature logistic model you can plot the decision boundary:

```python
import matplotlib.pyplot as plt
# Plot is_late on (distance_km, estimate_days) plane, colour by class
# The model's decision boundary is a straight line through this 2D space.
```

This is the geometric interpretation of "logistic regression draws a line."

## Quick Check

1. Why use `class_weight='balanced'` on Olist's `is_late`?
2. The coefficient for `distance_km` is +0.8 (after standardisation). What does the magnitude tell you?
3. If you didn't `stratify=y`, what could go wrong?

## Today's deliverable

### Bronze
- Skeleton notebook. 5 TODOs: load parquet, split with stratify, scale, fit, print classification report.
- Output: trained model object pickled.

### Gold
- Same task. Compare 3 feature sets:
  1. Just `distance_km` and `estimate_days`
  2. The 7-feature set above
  3. All numeric features in the parquet
- Report which gives best F1 on the late class. 1-paragraph explanation of why.

---

# Class 4-3 — Decision Trees and Random Forests

> **Today: forget straight lines. Trees ask yes/no questions in sequence.**

## Why this matters today

Linear models draw a single boundary. Real Olist data isn't linearly separable — you need *if-then* logic. Trees encode that logic naturally and they're the workhorses of tabular ML. Random forests = many trees voting. Random forests routinely beat logistic regression on Olist's `is_late` task and are the right baseline for any tabular problem.

## Section 1 — How a Decision Tree splits

At each node, the tree asks: *"of all features and all thresholds, which split makes the two child nodes most pure?"*

Purity = "as many of one class as possible in each child." Measured by **Gini impurity** or **entropy**.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)

# Visualise — actually see what the tree decided
plot_tree(dt, feature_names=features, class_names=['ontime', 'late'], filled=True)
```

A 4-deep tree on Olist might look like:
```
distance_km > 1500?
├── yes: estimate_days < 20?
│   ├── yes: is_late=1
│   └── no: ...
└── no: ...
```

It's basically a flowchart. **Tree models are the most interpretable ML you'll ever ship.**

## Section 2 — Why one tree isn't enough

A single deep tree memorises the training data (overfits). The fix: train **many** shallow trees on bootstrapped data subsets, average their votes. This is **bagging**, and the model is called **Random Forest**.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
```

Three knobs that matter:
- `n_estimators` — more trees = better, slower. 200-500 is the sweet spot.
- `max_depth` — controls overfitting. 8-15 for Olist.
- `min_samples_leaf` — prevents tiny leaves that memorise noise.

## Section 3 — Feature importance

```python
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(imp)
```

On Olist `is_late` you should see `distance_km` and `estimate_days` dominate. This is **embedded feature selection** (Class 3-4) — you got it for free with the model.

## Section 4 — Random Forest vs Logistic — bake-off

```python
from sklearn.metrics import f1_score

models = {
    'Logistic': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Tree (d=4)': DecisionTreeClassifier(max_depth=4, class_weight='balanced'),
    'Tree (d=12)': DecisionTreeClassifier(max_depth=12, class_weight='balanced'),
    'RF (300×12)': RandomForestClassifier(n_estimators=300, max_depth=12,
                                          class_weight='balanced', n_jobs=-1),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    f1 = f1_score(y_test, m.predict(X_test))
    print(f"{name:15s}  F1 (late class) = {f1:.3f}")
```

Expected ordering on Olist (approximate):
1. RF — best
2. Tree d=12 — slightly worse, single-tree variance
3. Logistic — solid baseline
4. Tree d=4 — too shallow, underfits

## Quick Check

1. Why does a random forest with 300 trees beat a single tree with the same `max_depth`?
2. Tree models don't need feature scaling. Why?
3. `min_samples_leaf=20` — what would happen if you set it to 1?

## Today's deliverable

### Bronze
- Provided notebook. Train DT (depth=4), DT (depth=12), RF. Compare F1.

### Gold
- Same. Plus tune `max_depth` and `n_estimators` via grid search. Plot the validation curve. Pick the operating point and justify in a paragraph.

---

# Class 4-4 — Model Evaluation

> **Today: stop celebrating accuracy. The model that always predicts "on time" gets 93% accuracy on Olist and is completely useless.**

## Why this matters today

Imbalanced classification is the #1 trap students fall into. Olist `is_late` is 7% positive. A model that never predicts "late" gets 93% accuracy and catches zero late orders. The whole point of the exercise is catching those 7%. We need metrics that respect that.

## Section 1 — The confusion matrix

For binary classification:

```
                Predicted on-time   Predicted late
Actual on-time     TN                FP
Actual late        FN                TP
```

- **TP** — late, predicted late ✓
- **TN** — on-time, predicted on-time ✓
- **FP** — on-time, predicted late ✗ (we falsely alarmed the customer)
- **FN** — late, predicted on-time ✗ (we missed a real late order — usually the costlier mistake)

## Section 2 — Metrics derived from it

| Metric | Formula | Reads as |
|---|---|---|
| **Accuracy** | (TP+TN)/N | "fraction we got right" — useless on imbalance |
| **Precision** | TP/(TP+FP) | "of what we flagged late, how many actually were?" |
| **Recall** | TP/(TP+FN) | "of all the late orders, how many did we catch?" |
| **F1** | 2·P·R/(P+R) | balanced harmonic mean of precision + recall |
| **ROC-AUC** | area under the ROC curve | overall ranking quality (threshold-independent) |

For Olist's late-detection problem, **recall** and **F1** matter most. Missing a late order (FN) is costly — the customer is angry. False alarms (FP) cost a notification. So we trade some precision for high recall.

## Section 3 — Code

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'ROC — AUC = {roc_auc_score(y_test, y_proba):.3f}')
plt.show()
```

## Section 4 — Choosing the threshold

`predict_proba` gives a probability. Default threshold is 0.5, but for Olist we want higher recall — lower the threshold:

```python
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)
print(classification_report(y_test, y_pred))
```

The **precision-recall curve** lets you pick the threshold that balances them for your business cost.

## Section 5 — When to care about which metric

| Scenario | Metric to optimise |
|---|---|
| Cancer screening (catching disease matters most) | Recall |
| Spam filter (don't put real email in spam) | Precision |
| Olist late-delivery (balanced; missing is costly) | F1 / ROC-AUC |
| Highly skewed (1 positive in 10,000) | PR-AUC, not ROC-AUC |

## Quick Check

1. A model has 99% accuracy on Olist `is_late`. Should you celebrate?
2. Precision = 0.4, Recall = 0.85 on the late class. What does that mean for the business?
3. Why does lowering the threshold from 0.5 to 0.3 trade precision for recall?

## Today's deliverable

### Bronze
- Take your RF from 4-3. Print the confusion matrix and classification report. Identify which row of the confusion matrix is "missed late orders" and write its count.

### Gold
- Plot ROC curve and PR curve. Find the threshold that maximises F1. Write a 1-paragraph recommendation: which threshold should the business team use, and why?

---

# Class 4-5 — Overfitting and Regularisation

> **Today: a model that's perfect on training and bad on test is useless. We diagnose, then prevent.**

## Why this matters today

Your RF in Class 4-3 might score F1 = 0.95 on training and F1 = 0.55 on test. That's overfitting. Production data will look like the test set, not the training set. Today we learn the three weapons against overfitting: train/test splits, cross-validation, and regularisation.

## Section 1 — Train / Test / Validation split

```
Original data
    ├── 80% Training        — fit model
    └── 20% Test            — final score (touch ONCE at the end)
```

For tuning hyperparameters, split training further:
```
Training data
    ├── 64% Train (sub)     — fit candidate models
    └── 16% Validation      — pick the best
```

Or use **k-fold cross-validation**: split training into 5 folds, train on 4, validate on 1, rotate, average.

## Section 2 — Cross-validation in code

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1')
print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

If `mean` is high but `std` is also high, your model is unstable. If `mean` is high and `std` is low, you have a robust estimate.

## Section 3 — Regularisation

For logistic regression: shrink coefficients toward zero.

| Penalty | What it does | When |
|---|---|---|
| **L2 (Ridge)** | shrinks all coefs smoothly | Default. Good general regulariser. |
| **L1 (Lasso)** | drives some coefs to *exactly zero* | Built-in feature selection. Use when you have many redundant features. |
| **Elastic Net** | mix of L1 + L2 | Best of both, more knobs to tune. |

```python
LogisticRegression(C=0.1, penalty='l2')   # smaller C = stronger regularisation
LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
```

`C` is the *inverse* of regularisation strength. Lower C = simpler model.

## Section 4 — Bias-variance trade-off (the mental model)

- **High bias** = underfitting. Model too simple. Both train and test scores are bad.
- **High variance** = overfitting. Model too complex. Train great, test bad.
- The sweet spot is right between.

Tools to spot it:
- Plot **learning curve** (train vs test score as you grow training data).
- Plot **validation curve** (train vs validation score as you grow model complexity).

## Section 5 — On Olist

```python
from sklearn.model_selection import validation_curve

depths = [3, 5, 8, 12, 16, 20]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    X_train, y_train, param_name='max_depth', param_range=depths,
    cv=5, scoring='f1', n_jobs=-1,
)
# Plot mean of train_scores and val_scores vs depths.
# The depth where val_score starts dropping = your overfitting point.
```

For Olist `is_late`, validation usually peaks around depth 12–14 and starts to drop after.

## Quick Check

1. Your training F1 is 0.95 and test F1 is 0.60. Diagnosis?
2. L1 vs L2 — when would you reach for L1?
3. Why is k-fold CV more reliable than a single 80/20 split for hyperparameter tuning?

## Today's deliverable

### Bronze
- Run `cross_val_score` on your M4C3 RF with 5 folds. Print mean ± std.

### Gold
- Plot a validation curve over `max_depth`. Identify the overfitting elbow. Train the final RF at that depth. Compare its test F1 to the non-cross-validated version.

---

# Class 4-6 — Lab: Build a Classifier (Module 4 capstone)

> **The 90-minute capstone. You ship `model_v1.pkl` — the file M5 will load.**

## Brief

Train the best classifier you can on `olist_clean.parquet` predicting `is_late`. Pickle it. Ship a 1-page report.

## Lab structure (90 min)

### Stage 1 — Load & prepare (10 min)
- Load `olist_clean.parquet` from your group repo's M3 output.
- Drop rows with NaN in `is_late`.
- Stratified 80/20 train/test split, `random_state=42`.

### Stage 2 — Baseline (10 min)
- Logistic regression with `class_weight='balanced'`, scaled features.
- Print classification report. **This is your floor.**

### Stage 3 — Random Forest (15 min)
- `n_estimators=300`, `max_depth=12`, `min_samples_leaf=20`.
- Print classification report. Should beat baseline F1.

### Stage 4 — Hyperparameter tuning (20 min)
- 5-fold CV on `max_depth ∈ [8, 12, 16]` and `n_estimators ∈ [200, 300, 500]`.
- Refit best model on full training set.

### Stage 5 — Threshold tuning (15 min)
- Plot precision-recall curve.
- Pick threshold that maximises F1 on validation fold.
- Re-evaluate on test with that threshold.

### Stage 6 — Save (5 min)
```python
import joblib
joblib.dump({'model': best_rf, 'threshold': 0.32, 'features': features}, 'model_v1.pkl')
```

### Stage 7 — Report (15 min)
1-page markdown:
- Final test F1, ROC-AUC, precision, recall.
- Top 5 features by importance.
- 3 things you learned.
- 1 thing you'd improve given another week.

## Submission

```
module-4/class_6/submissions/<TeamName>/
    ├── notebook.ipynb
    ├── model_v1.pkl
    └── report.md
```

## Grading rubric (100 points)

| Component | Weight |
|---|---|
| Notebook runs end-to-end without errors | 15 |
| `model_v1.pkl` correctly loads & predicts | 15 |
| Test F1 (late class) ≥ 0.55 | 20 |
| Threshold tuning + PR curve | 15 |
| Feature importance interpretation | 10 |
| Report quality (clear, honest, actionable) | 15 |
| Code quality / comments | 10 |

> **Reminder:** the `model_v1.pkl` you ship today is loaded by Module 5. Don't lose it.
