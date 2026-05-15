---
title: "Module 5 — Unsupervised Learning (Olist Edition)"
subtitle: "Find structure in the Olist customer base without labels"
author: "BePro AI/ML Mentorship Program"
date: "2026-05-04"
---

# Module 5 — Unsupervised Learning

| | |
|---|---|
| Module duration | 12 hours (6 classes × 2 h) |
| Weeks | 9–10 (after midterm) |
| Pre-requisite | Module 3 — `olist_clean.parquet`. (Module 4 is helpful context but not strictly required for this module's deliverable.) |
| Module deliverable | `customer_clusters.csv` — every Olist `customer_unique_id` mapped to a cluster, plus a 1-page cohort write-up |

---

## Module overview

Module 4 had labels. The model learned to predict `is_late` because we told it which orders were late. **Module 5 has no labels.** We give the model the customer behaviour and ask it: *what kinds of customers are there?* The model groups them. We name the groups. The marketing team uses the groups.

By the end, every student has segmented the Olist customer base into 4-6 named cohorts and produced a CSV that joins back to the M3 spine.

## Learning outcomes

| LO | Statement | Cognitive level |
|---|---|---|
| LO1 | Distinguish supervised vs unsupervised learning and when to use each | Understand |
| LO2 | Apply K-Means and Hierarchical clustering and pick the number of clusters | Apply |
| LO3 | Reduce dimensionality with PCA and t-SNE and visualise high-dim data in 2D | Apply |
| LO4 | Detect anomalies (extreme orders) using statistical and model-based methods | Apply |
| LO5 | Translate cluster output into named, actionable cohorts | Evaluate |

## Class structure

| # | Title | Olist focus |
|---|---|---|
| 5-1 | Introduction to Clustering | Why segment Olist customers; distance metrics |
| 5-2 | K-Means Clustering | RFM features → 4 named cohorts |
| 5-3 | Hierarchical Clustering | Compare with K-Means; dendrograms |
| 5-4 | Dimensionality Reduction | PCA + t-SNE on the customer-feature matrix |
| 5-5 | Anomaly Detection | Find unusual Olist orders (extreme freight, very high price) |
| 5-6 | Lab — Customer Segmentation | Output `customer_clusters.csv` + cohort write-up |

---

# Class 5-1 — Introduction to Clustering

> **Today: stop labelling, start grouping. We turn 99k Olist customers into 4 named cohorts.**

## Why this matters today

Marketing doesn't have time to email each customer individually. They need 4-6 buckets they can run different campaigns against. *Clustering* finds those buckets *for them* from behaviour alone. Same data we used for `is_late` — completely different lens.

## Section 1 — Supervised vs Unsupervised — picture in your head

| | Supervised | Unsupervised |
|---|---|---|
| Data | Inputs **+ labels** | Inputs only |
| Question | "Predict the label" | "Find the structure" |
| Olist example | Predict `is_late` | Group customers by behaviour |
| Evaluation | Accuracy, F1 against truth | Cluster cohesion, business interpretability |

Unsupervised problems are *open*. Two analysts can produce different cluster sets on the same data and **both can be valid** — different lenses on the same reality.

## Section 2 — Distance metrics (the core building block)

Clustering = grouping points that are "close." Close means a distance:

| Distance | Formula | When |
|---|---|---|
| **Euclidean** | √Σ(xᵢ - yᵢ)² | Default for numeric, scaled features |
| **Manhattan** | Σ|xᵢ - yᵢ| | Robust to outliers |
| **Cosine** | 1 - x·y/(‖x‖·‖y‖) | Direction matters more than magnitude (text, embeddings) |
| **Hamming** | count(xᵢ ≠ yᵢ) | Binary / categorical |

For Olist customer features (price, freight, recency) → **Euclidean after scaling**.

## Section 3 — Why scaling matters here even more than before

Customer A spent 30 R$ once, 10 days ago.
Customer B spent 800 R$ once, 200 days ago.

Without scaling, the distance between them is dominated by `total_price` (range 0-7000) and ignores `recency` (range 0-700). Always `StandardScaler` numeric features before clustering.

## Section 4 — Evaluating clusters without labels

With no truth, we use:

- **Inertia / WCSS** — within-cluster sum of squares. Lower = tighter clusters.
- **Silhouette score** — [-1, +1]. Higher = better separation between clusters.
- **The eyeball test** — does the cluster mean look like a *real human group* a marketer would recognise?

The eyeball test is the most important. A statistically tight cluster of customers nobody can describe is a useless cluster.

## Quick Check

1. Why doesn't unsupervised learning need labels?
2. Customer A: high price, recent. Customer B: low price, old. Why is `StandardScaler` mandatory before clustering them?
3. You get silhouette = 0.20. Good or bad?

## Today's deliverable

### Bronze
- Read `olist_clean.parquet`. Compute per-customer aggregates: total_orders, total_spend, days_since_last_order. Plot a 2D scatter `total_spend × days_since_last_order`. Eyeball: do you see groups?

### Gold
- Same. Add 2 more aggregates of your choice. Run a quick `KMeans(n_clusters=4)` on scaled features. Plot the result coloured by cluster. Don't worry about quality yet — that's 5-2.

---

# Class 5-2 — K-Means Clustering

> **Today: the most-used clustering algorithm in industry. We segment Olist customers by RFM and name the cohorts.**

## Why this matters today

K-Means is the default. It's fast, simple, and good enough for 80% of segmentation tasks in production. Today we run it properly: choose K, scale, evaluate, name the clusters in business language.

## Section 1 — The K-Means algorithm in plain English

```
1. Pick K (number of clusters).
2. Place K random centroids in the feature space.
3. Assign each point to its nearest centroid.
4. Move each centroid to the mean of its assigned points.
5. Repeat 3-4 until centroids stop moving.
```

That's it. Five steps. Memorise this — interview question.

## Section 2 — RFM features for Olist

**RFM** = Recency, Frequency, Monetary value. It's the universal e-commerce segmentation language.

```python
import pandas as pd
import numpy as np

df = pd.read_parquet('olist_clean.parquet')

# Take "today" as the day after the last order in the data
SNAPSHOT = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_unique_id').agg(
    recency=('order_purchase_timestamp', lambda s: (SNAPSHOT - s.max()).days),
    frequency=('order_id', 'count'),
    monetary=('total_price', 'sum'),
).reset_index()

# Log-transform skewed M (most customers spend little, a few spend a lot)
rfm['monetary_log'] = np.log1p(rfm['monetary'])
```

Note: in Olist, most customers have `frequency=1` (one-time buyers). That's a common e-commerce reality. We can still cluster — the differentiation comes from R and M.

## Section 3 — Choosing K (the elbow method)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = rfm[['recency', 'frequency', 'monetary_log']]
X = StandardScaler().fit_transform(X)

inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

# Plot k vs inertia. The "elbow" — where the curve bends — is your K.
```

For Olist RFM you'll typically see the elbow at **K=4 or K=5**.

## Section 4 — Fit, assign, name

```python
km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['cluster'] = km.fit_predict(X)

# Profile each cluster
profile = rfm.groupby('cluster').agg(
    n=('customer_unique_id', 'count'),
    avg_recency=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_monetary=('monetary', 'mean'),
)
print(profile.sort_values('avg_monetary', ascending=False))
```

Now **name them based on the profile**. Examples:

| Cluster | Profile | Name |
|---|---|---|
| 0 | recency=180, freq=1, monetary=85 | "Lost casuals" |
| 1 | recency=45, freq=3, monetary=320 | "Loyal regulars" |
| 2 | recency=30, freq=1, monetary=950 | "High-value newcomers" |
| 3 | recency=15, freq=1, monetary=70 | "Recent low-spend" |

The names tell a marketing story. *That's* the deliverable.

## Section 5 — Common K-Means pitfalls

1. **Forgetting to scale** → cluster dominated by the largest-range feature.
2. **Only running once** → K-Means depends on random init. Use `n_init=10` (or higher).
3. **Picking K by accuracy** → there's no accuracy. Use elbow + silhouette + business sense.
4. **K-Means assumes spherical clusters** → if your data has long curvy clusters, K-Means won't find them. Use DBSCAN or hierarchical instead.

## Quick Check

1. K-Means assumes clusters are roughly spherical. What kind of cluster shape would break it?
2. Why log-transform `monetary` before clustering?
3. The elbow plot is "smooth" with no clear elbow. What does that mean about the data?

## Today's deliverable

### Bronze
- Compute RFM. Run K-Means with K=4. Profile the 4 clusters. Name them.

### Gold
- Same. Add 2 more features (e.g. avg freight share, payment instalment median). Run elbow analysis K=2..10. Pick K with justification. Profile and name.

---

# Class 5-3 — Hierarchical Clustering

> **Today: trees of clusters. K-Means' main rival.**

## Why this matters today

K-Means makes you commit to K up front. Hierarchical clustering builds a tree and lets you pick the cut later. It's slower but gives more visual insight into how customers relate to each other.

## Section 1 — The algorithm (agglomerative)

```
1. Start with every point as its own cluster.
2. Find the two closest clusters; merge.
3. Repeat until one cluster remains.
4. The order of merges = a tree (dendrogram).
```

Reading the dendrogram height = the distance at which clusters merged. Cut horizontally at any height → that's your K.

## Section 2 — Linkage criteria

When merging clusters, "closest" can mean different things:

| Linkage | Distance to closest pair | When |
|---|---|---|
| **Single** | minimum distance between any two points | Long, chained clusters |
| **Complete** | maximum distance | Compact, spherical (like K-Means) |
| **Average** | average of all pair distances | Balanced default |
| **Ward** | minimises within-cluster variance | Best general default for numeric |

For Olist RFM features → **Ward**.

## Section 3 — On Olist (subsample for speed)

Hierarchical is O(n²) memory — at 99k customers it crashes. Sample 5000.

```python
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample for tractability
sample = rfm.sample(n=5000, random_state=42)
X = StandardScaler().fit_transform(sample[['recency', 'frequency', 'monetary_log']])

# Build the linkage matrix
Z = sch.linkage(X, method='ward')

# Plot the dendrogram (top 30 merges)
plt.figure(figsize=(12, 5))
sch.dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Olist customer dendrogram (Ward linkage, 5000 sample)')
plt.show()

# Cut into 4 clusters
clusters = sch.fcluster(Z, t=4, criterion='maxclust')
```

## Section 4 — When to choose hierarchical over K-Means

| Use hierarchical | Use K-Means |
|---|---|
| Small dataset (<10k rows) | Large dataset (>100k) |
| You want to *see* the cluster tree | You just need labels |
| You aren't sure about K | You know K from business |
| Clusters have nested structure | Clusters are roughly spherical |

## Quick Check

1. Why does hierarchical clustering crash on 99k customers but K-Means is fine?
2. Single vs Complete linkage — which gives you more spherical clusters?
3. Reading a dendrogram — what does "cut at height 2.5" mean?

## Today's deliverable

### Bronze
- Sample 3000 Olist customers. Build dendrogram. Cut into 4. Compare cluster assignments to your K-Means from 5-2.

### Gold
- Repeat with three linkage methods (ward, complete, average). Compare which gives the most interpretable cohorts. Write a paragraph defending your choice.

---

# Class 5-4 — Dimensionality Reduction

> **Today: when you have 30 features, you can't see them all. Compress to 2D and look.**

## Why this matters today

Olist customer data has many derived features. Plotting `recency vs monetary` is fine but loses everything else. Dimensionality reduction projects high-dimensional data into 2-3 axes you can plot. Two flavours: **PCA** (linear, fast, interpretable) and **t-SNE** (non-linear, slow, beautiful clusters).

## Section 1 — PCA (Principal Component Analysis)

PCA finds the directions of maximum variance and projects data onto them.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

# Plot, colored by cluster
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=rfm['cluster'], s=8, alpha=0.5)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('Olist customers in PCA space')
plt.show()
```

The two PCA axes are **linear combinations of the original features**. You can read what each axis "means" by looking at `pca.components_`.

## Section 2 — t-SNE (the pretty cluster visualisation)

t-SNE preserves *local* structure (similar customers stay near each other) at the cost of distortion of global geometry.

```python
from sklearn.manifold import TSNE

# Subsample — t-SNE is O(n²)
sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
X_sub = X_scaled[sample_idx]

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sub)
```

Use t-SNE for **plots that go in slides**. Don't use t-SNE coordinates as features for downstream models — the absolute positions are meaningless.

## Section 3 — When PCA, when t-SNE

| | PCA | t-SNE |
|---|---|---|
| Speed | Fast (seconds on 100k) | Slow (minutes on 5k) |
| Determinism | Same output every time | Different each run |
| Interpretability | Each axis has meaning | Axes are arbitrary |
| Use as feature | Yes | No |
| Use for visualisation | Yes (basic) | Yes (publication) |

## Section 4 — UMAP (briefly)

Modern alternative to t-SNE. Faster, preserves more global structure. `pip install umap-learn`. Same API. For Olist work it's a fine choice.

## Quick Check

1. PCA finds directions of maximum variance. Why is that "useful"?
2. Why can't t-SNE coordinates be used as features for a downstream model?
3. PC1 explains 35%, PC2 explains 18%. Together = 53%. Is that good or bad?

## Today's deliverable

### Bronze
- Run PCA on your scaled RFM features. Plot 2D coloured by your K-Means cluster from 5-2.

### Gold
- Repeat with t-SNE. Compare visual quality. Write a paragraph: which is more useful for the marketing slide deck and why?

---

# Class 5-5 — Anomaly Detection

> **Today: the orders that don't look like the rest. Often where fraud or system bugs hide.**

## Why this matters today

Most rows in Olist look normal. But 0.5% of orders have freight 10× higher than peers, or were placed at 3am by a customer who's never bought before, or have a single product priced at 5000 R$. These are **anomalies** — possibly fraud, possibly data errors, possibly genuinely unusual customers. Either way, the business wants to know about them.

## Section 1 — What "anomaly" means

| Type | Olist example |
|---|---|
| **Point anomaly** | One order with unusually high freight relative to value |
| **Contextual anomaly** | A high-value order placed at an unusual time |
| **Collective anomaly** | A burst of 50 orders from one address in 1 hour |

For M5 we focus on **point anomalies** — easiest to detect, most useful starter skill.

## Section 2 — Statistical methods

**Z-score:**
```python
df['freight_z'] = (df['total_freight'] - df['total_freight'].mean()) / df['total_freight'].std()
anomalies = df[df['freight_z'].abs() > 3]
```

`|z| > 3` = beyond 3 standard deviations = roughly 0.3% of normal data.

**IQR method:**
```python
q1, q3 = df['total_price'].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
df_outliers = df[df['total_price'] > upper]
```

## Section 3 — Model-based: Isolation Forest

```python
from sklearn.ensemble import IsolationForest

features = ['total_price', 'total_freight', 'distance_km', 'num_items']
iso = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = iso.fit_predict(df[features].dropna())
# -1 = anomaly, +1 = normal
print(df['anomaly'].value_counts())
print(df[df['anomaly'] == -1].sample(5))
```

`contamination=0.01` — tell the model that ~1% of rows are anomalous. Tune based on business intuition.

## Section 4 — Local Outlier Factor (LOF)

LOF compares each point's local density to its neighbours' density. Points in low-density regions = anomalies. Better than Isolation Forest when anomalies are *contextual* (close to a different cluster).

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
df['lof'] = lof.fit_predict(df[features].dropna())
```

## Section 5 — On Olist

Useful anomalies to surface:
- **Freight cost > 5x order value** — possible system bug or fraud
- **Single order with > 50 items** — bulk buyer or wholesaler
- **Customer's first order > 2000 R$** — high-value newcomer worth a personal touch
- **Order placed at 3am from a state with no prior orders** — fraud signal

The output is a CSV the fraud / customer success team would use.

## Quick Check

1. `|z| > 3` — what % of a normal distribution is beyond this?
2. Isolation Forest's `contamination` parameter — what does it tell the model?
3. Why might LOF detect anomalies that Isolation Forest misses on Olist?

## Today's deliverable

### Bronze
- Compute z-score for `total_freight`. Print the top 20 most extreme orders.

### Gold
- Run Isolation Forest with `contamination=0.01`. Compare overlap with z-score-flagged orders. Write 1 paragraph: which method finds business-meaningful anomalies on Olist?

---

# Class 5-6 — Lab: Customer Segmentation (Module 5 capstone)

> **The 90-minute capstone. Output: `customer_clusters.csv` joining cluster ID to every Olist customer, plus a 1-page named-cohort write-up.**

## Brief

Take `olist_clean.parquet`. Aggregate to per-customer features. Cluster. Name the clusters. Save.

## Output schema

`customer_clusters.csv`:

| Column | Type |
|---|---|
| customer_unique_id | string |
| cluster_id | int |
| cluster_name | string ("Lost casuals", "Loyal regulars", etc.) |
| recency | int |
| frequency | int |
| monetary | float |

## Lab structure (90 min)

### Stage 1 — Aggregate to customer level (15 min)
- Compute RFM + at least 2 additional features (avg freight share, modal payment type, dominant state...).
- Save `customer_features.parquet` as an intermediate.

### Stage 2 — Scale + elbow (10 min)
- StandardScaler. Run K-Means for K=2..10. Plot inertia curve. Pick K.

### Stage 3 — Fit final K-Means (10 min)
- `random_state=42`, `n_init=10`. Compute silhouette score.

### Stage 4 — Profile + name (15 min)
- Group by cluster, print mean of each feature.
- Name each cluster in 2-4 words. Stick the name on the row.

### Stage 5 — Visualise (15 min)
- PCA 2D scatter, coloured by cluster.
- Bar chart of cluster sizes.
- Save PNGs.

### Stage 6 — Anomaly side-quest (10 min)
- Run Isolation Forest. Add `is_anomaly` column to the CSV. (Bonus only, not required.)

### Stage 7 — Write the cohort report (15 min)
1-page markdown:
- Final K and why.
- Each cluster's name + 2-sentence description + 1 marketing recommendation.
- 3 things you learned from the data.

## Submission

```
module-5/class_6/submissions/<TeamName>/
    ├── notebook.ipynb
    ├── customer_clusters.csv
    ├── pca_scatter.png
    └── cohort_report.md
```

## Grading rubric (100 points)

| Component | Weight |
|---|---|
| `customer_clusters.csv` schema correct | 20 |
| K choice justified (elbow + silhouette) | 15 |
| Cluster profiles sensible (cohort means differ meaningfully) | 20 |
| Cluster names + marketing recs make business sense | 20 |
| Visualisations clear and labelled | 10 |
| Code quality / reproducibility | 15 |

> **Reminder:** `customer_clusters.csv` is loaded by Module 8 (capstone) as a feature for the deployed model. Don't lose it.
