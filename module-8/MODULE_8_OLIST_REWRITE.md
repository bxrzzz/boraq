---
title: "Module 8 — Capstone Project & Deployment (Olist Edition)"
subtitle: "Ship the compound project as a deployed FastAPI service and demo it"
author: "BePro AI/ML Mentorship Program"
date: "2026-05-04"
---

# Module 8 — Capstone Project & Deployment

| | |
|---|---|
| Module duration | 12 hours (6 classes × 2 h) + Final exam |
| Weeks | 15–16 + 17 (final) |
| Pre-requisite | Modules 3-7 — clean data, model_v3 with sentiment feature |
| Module deliverable | A **public GitHub repo + deployed URL + 3-minute video** showing the compound Olist project end-to-end |
| Grading | 50% deployed system works · 30% code quality · 20% demo presentation |

---

## Module overview

The capstone. Six modules of work compound into one shippable system. Each student leaves with:

1. A **public GitHub repository** containing the full pipeline — raw data → cleaned Parquet → model → sentiment-enriched model → deployed API.
2. A **live HTTPS URL** they can paste in a CV that hits a `/predict` endpoint and returns "this Olist order has X% chance of being late, top reasons: A, B, C."
3. A **3-minute walkthrough video** explaining the project to a non-technical hiring manager.

These three artefacts are the resume bullet from §4 of the master plan.

## Learning outcomes

| LO | Statement | Cognitive level |
|---|---|---|
| LO1 | Plan an end-to-end ML project: data → model → API → monitoring | Apply |
| LO2 | Iterate on a model with structured error analysis | Apply |
| LO3 | Wrap a trained model in a Flask / FastAPI endpoint | Apply |
| LO4 | Containerise the application with Docker | Apply |
| LO5 | Deploy to a free cloud host (Render / Railway / Fly.io) | Apply |
| LO6 | Identify model drift and write a re-training script | Understand |
| LO7 | Present a technical project to a non-technical audience | Create |

## Class structure

| # | Title | Olist focus |
|---|---|---|
| 8-1 | Project Planning | 1-page architecture for the deployed Olist system |
| 8-2 | Model Development Sprint | Consolidate M4-M7 outputs into one polished model |
| 8-3 | Deployment | Wrap as FastAPI; Docker; deploy to free cloud |
| 8-4 | MLOps Basics | Logging, `/health`, drift, re-train script |
| 8-5 | Demo Day Prep | Record 3-min video; rehearse |
| 8-6 | Final Presentations | Live demo |

---

# Class 8-1 — Project Planning

> **Today: stop coding. Draw the system. One A4 page that shows everything.**

## Why this matters today

Engineering without a plan is shipping the first thing that compiles. We start the capstone by drawing the architecture: where does data enter, where does the model run, where does the answer come out, what gets logged. The plan is your contract for the next 5 classes.

## Section 1 — The 1-page architecture template

Every team produces one page with **5 boxes and the arrows between them**:

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────┐
│  Client  │ →  │  FastAPI    │ →  │ Model.pkl │ →  │  Response   │ →  │ Client   │
│  POSTs   │    │  /predict   │    │ + scaler  │    │  JSON       │    │ shows    │
│  order   │    │  endpoint   │    │ + sentim. │    │             │    │ result   │
└──────────┘    └─────────────┘    └───────────┘    └─────────────┘    └──────────┘
                       │
                       ▼
                ┌─────────────┐
                │  Log file   │
                │  /logs.csv  │
                └─────────────┘
```

5 boxes max. If your architecture needs more, simplify.

## Section 2 — Define the API contract

Before any code, write down:

```
POST /predict
Body:
{
  "distance_km": 1500,
  "log_freight": 2.8,
  "num_items": 1,
  "estimate_days": 22,
  "is_weekend": 0,
  "purchase_hour": 14,
  "review_text": "produto excelente"   // optional
}

Response:
{
  "is_late_probability": 0.34,
  "is_late_label": 0,
  "top_reasons": [
    "distance_km is high (1500 km)",
    "estimate_days is generous (22)"
  ],
  "model_version": "v3_with_text",
  "served_at": "2026-08-15T12:34:56Z"
}
```

This contract is the source of truth. Frontend, backend, and model all conform to it.

## Section 3 — Scope the work — what fits in 5 classes

In: predict `is_late` for one order at a time, return reasons, log every prediction.
Out: re-training UI, multi-tenant auth, batch inference, dashboards. (Time-box ruthlessly.)

## Section 4 — Common pitfalls

- **Trying to deploy 3 models** — pick one, ship it, then iterate.
- **Worrying about scale before traffic** — your demo will get 50 requests, not 50k. Don't over-engineer.
- **Skipping logs** — without logs you can't debug or measure drift. 3 lines of code, never skip.

## Quick Check

1. What goes in your 5-box architecture?
2. The API contract is fixed. Why does that matter for parallel teamwork?
3. You're tempted to add a real-time dashboard. Is it in scope for M8? Why or why not?

## Today's deliverable

### Both tiers
- Submit `architecture.md` with the 5-box diagram (ASCII or PNG sketch) + the API contract.
- 1-page max. Mentor signs off before Class 8-2.

---

# Class 8-2 — Model Development Sprint

> **Today: take M4-M7 outputs and consolidate into one polished model file.**

## Why this matters today

By M8 each student has many model files. We need ONE file the API will load. Pick the best, package it correctly, write the inference function.

## Section 1 — Consolidating

Pick the winning model — usually `model_v3_with_text.pkl` from M7. Rebuild it with cleaner code:

```python
# train_final_model.py
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet('olist_clean.parquet').dropna(subset=['is_late'])
sent = pd.read_parquet('review_sentiment.parquet')
df = df.merge(sent.groupby('order_id')['nlp_sentiment'].mean().reset_index(),
              on='order_id', how='left')
df['nlp_sentiment'] = df['nlp_sentiment'].fillna(3)
df['has_review'] = (df['nlp_sentiment'] != 3).astype(int)

NUM = ['distance_km', 'log_freight', 'num_items', 'estimate_days', 'purchase_hour', 'nlp_sentiment']
CAT = ['payment_type']
TARGET = 'is_late'

pre = ColumnTransformer([
    ('num', StandardScaler(), NUM),
    ('cat', 'passthrough', CAT),  # one-hot inline if needed
])
pipe = Pipeline([
    ('pre', pre),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=12,
                                  class_weight='balanced', n_jobs=-1, random_state=42)),
])

pipe.fit(df[NUM + CAT], df[TARGET])
joblib.dump({'model': pipe, 'features': NUM + CAT, 'version': 'v3_final'}, 'final_model.pkl')
```

One file. No notebooks. The API will load this.

## Section 2 — Inference function

```python
# inference.py
import joblib, pandas as pd

bundle = joblib.load('final_model.pkl')
MODEL = bundle['model']
FEATURES = bundle['features']

def predict(payload: dict) -> dict:
    """Take the API payload, return structured prediction."""
    X = pd.DataFrame([payload])[FEATURES]
    proba = float(MODEL.predict_proba(X)[0, 1])
    label = int(proba >= 0.4)
    reasons = top_reasons(payload, proba)
    return {
        'is_late_probability': round(proba, 3),
        'is_late_label': label,
        'top_reasons': reasons,
        'model_version': bundle['version'],
    }
```

Keep `inference.py` framework-agnostic. The API just calls `predict()`.

## Section 3 — Reason generation

A simple rule-based explanation pulls the user-visible top features:

```python
THRESHOLDS = {
    'distance_km': 1000,
    'estimate_days': 25,
    'is_weekend': 0.5,
    'nlp_sentiment': 2.5,
}
def top_reasons(payload, proba):
    if proba < 0.2:
        return ['Order looks on track']
    reasons = []
    if payload.get('distance_km', 0) > 1000:
        reasons.append(f"Long distance: {payload['distance_km']} km")
    if payload.get('estimate_days', 0) > 25:
        reasons.append(f"Long delivery estimate ({payload['estimate_days']} days)")
    if payload.get('nlp_sentiment', 3) < 2.5:
        reasons.append("Negative review history")
    return reasons[:3] or ['Some risk; see model.']
```

For sophistication: use SHAP. For simplicity: rules. Both are acceptable for the capstone.

## Section 4 — Acceptance criteria for the model

Before moving to deployment, verify:

- `joblib.load('final_model.pkl')` works on a fresh Python kernel.
- `predict(sample_payload)` returns a dict with the right keys.
- Test ROC-AUC on held-out set ≥ 0.70.

## Quick Check

1. Why ship a Pipeline instead of separate scaler + model files?
2. Why is `inference.py` framework-agnostic important?
3. What's a sane fallback when the user sends a payload missing one feature?

## Today's deliverable

- `final_model.pkl` and `inference.py` committed. `python -c "import inference; print(inference.predict({...}))"` works.

---

# Class 8-3 — Deployment

> **Today: the model is on your laptop. Get it on the internet.**

## Why this matters today

A model that lives only on your laptop is invisible. Today we wrap it in FastAPI, containerise with Docker, and deploy to a free cloud. Anyone with a URL can hit it.

## Section 1 — FastAPI app

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="Olist Late-Delivery Predictor")

class Order(BaseModel):
    distance_km: float
    log_freight: float
    num_items: int
    estimate_days: int
    purchase_hour: int
    nlp_sentiment: float = 3.0
    payment_type: str = 'credit_card'

@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': 'v3_final'}

@app.post('/predict')
def predict_endpoint(order: Order):
    return predict(order.dict())
```

Test locally:
```bash
pip install fastapi uvicorn
uvicorn app:app --reload
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km":1500,"log_freight":2.8,"num_items":1,"estimate_days":22,"purchase_hour":14}'
```

## Section 2 — Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# requirements.txt
fastapi==0.115.0
uvicorn==0.32.0
pandas==2.2.2
scikit-learn==1.5.0
joblib==1.4.2
pydantic==2.9.0
```

Build & test:
```bash
docker build -t olist-api .
docker run -p 8000:8000 olist-api
```

## Section 3 — Deploy to a free cloud

**Option 1: Render** — push your GitHub repo, set start command `uvicorn app:app --host 0.0.0.0 --port $PORT`. Free tier, sleeps after 15 min idle.

**Option 2: Railway** — connect GitHub, click deploy. Free $5/mo credit.

**Option 3: Fly.io** — `fly launch`, free 3 small VMs.

Whichever you pick, the goal is: a public HTTPS URL that responds to `POST /predict`.

## Section 4 — Test the live URL

```bash
curl -X POST https://<your-app>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km":1500,...}'
```

Save this curl command — you'll demo it in class 8-6.

## Quick Check

1. Why use FastAPI over Flask for this project?
2. The Dockerfile copies requirements first then code. Why?
3. Free-tier hosts sleep idle apps. What does that mean for your live demo?

## Today's deliverable

- Public live URL responds to `/health` and `/predict`.
- GitHub repo includes Dockerfile, app.py, requirements.txt.
- README has the URL + a sample curl command.

---

# Class 8-4 — MLOps Basics

> **Today: production-grade hygiene. Logs, drift, re-train.**

## Why this matters today

Models silently degrade. Distance distributions shift, customer behaviour changes, the model trained 6 months ago is worse than it was at launch. MLOps = the ops practices that catch and fix this.

## Section 1 — Logging every prediction

```python
import csv, datetime, os

LOG_PATH = '/data/predictions.csv'

@app.post('/predict')
def predict_endpoint(order: Order):
    result = predict(order.dict())
    log_row = {**order.dict(), **result, 'ts': datetime.datetime.utcnow().isoformat()}
    write_log(LOG_PATH, log_row)
    return result
```

The log gives you:
- What inputs are coming in (sanity check).
- What predictions are going out (drift check).
- Latency (`served_at - received_at`).

## Section 2 — Drift detection

A simple rule: weekly job that checks if input feature means / quantiles have shifted vs training distribution by > X standard deviations.

```python
# weekly_drift.py
import pandas as pd
log = pd.read_csv('/data/predictions.csv')
training_means = pd.read_csv('training_distribution.csv')

for feat in ['distance_km', 'estimate_days']:
    live_mean = log[feat].mean()
    train_mean = training_means[feat].iloc[0]
    train_std = training_means[feat].iloc[1]
    z = abs(live_mean - train_mean) / train_std
    if z > 2:
        print(f"DRIFT: {feat}: live={live_mean:.2f}, train={train_mean:.2f}, z={z:.2f}")
```

## Section 3 — Re-training script

`retrain.py` should be one command that takes the latest data, trains a fresh model, evaluates against the old one, and ships the better of the two.

```python
# retrain.py
def retrain():
    df = load_latest()
    new_model = train(df)
    new_auc = eval_on_holdout(new_model)
    old_auc = load_metric('current')
    if new_auc > old_auc + 0.01:    # require 1pt AUC improvement
        save_model(new_model, 'final_model.pkl')
        log_metric('current', new_auc)
        print(f"Promoted new model: {old_auc:.3f} → {new_auc:.3f}")
    else:
        print(f"Keeping current. {new_auc:.3f} not enough lift over {old_auc:.3f}")
```

## Section 4 — CI/CD (optional but neat)

A GitHub Action that re-runs tests + redeploys on every push:

```yaml
# .github/workflows/deploy.yml
name: Deploy
on: { push: { branches: [main] } }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest
      - run: curl -X POST $RENDER_DEPLOY_HOOK
```

## Quick Check

1. Why is logging predictions essential for MLOps?
2. Drift score = 2.5 standard deviations. Action?
3. Re-training script promotes a new model only if AUC improves by 0.01. Why a threshold instead of any improvement?

## Today's deliverable

- `/predict` writes a row to a logs file.
- `retrain.py` exists and prints either "promoted" or "keeping current."

---

# Class 8-5 — Demo Day Prep

> **Today: rehearse. The model is ready. The story isn't.**

## Why this matters today

Hiring managers won't read your code. They'll watch your 3-minute video. Good demos win interviews. Bad demos make even great projects forgettable.

## Section 1 — The 3-minute structure

```
0:00 – 0:30   The problem (1 sentence: "Olist customers don't know which orders will be late...")
0:30 – 1:00   The data (1 slide: 100k orders, 9 tables, target imbalance)
1:00 – 2:00   The system (architecture diagram + live curl demo)
2:00 – 2:30   The numbers (test AUC, F1, lift from sentiment, cost)
2:30 – 3:00   What's next (re-train cadence, drift monitoring, future features)
```

Time it. 3 minutes feels short — and it's exactly enough.

## Section 2 — The slide deck (5 slides max)

1. **Title** — name + project + your name
2. **The problem** — one screenshot of "Olist customer-service ticket complaining about late delivery"
3. **The system** — architecture diagram from Class 8-1
4. **The numbers** — bar chart of AUC across M4 / M6 / M7 models
5. **The future** — re-train + drift monitoring summary

## Section 3 — The live demo

Don't trust your free-tier app to wake up live. **Pre-warm it 5 min before** by hitting `/health`. Have a backup recording in case the network fails.

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_order.json
```

Show the JSON response on screen. Highlight the `top_reasons` — that's the magic moment.

## Section 4 — The recording

Use Loom, Zoom-record, or QuickTime. Camera small, screen big. Record once, watch it back, re-record if needed.

## Quick Check

1. The 3-minute video — what goes in 1:00 - 2:00?
2. Free-tier apps sleep. How do you avoid the 30-second wake-up dead air on stage?
3. Why include "what's next" in the demo?

## Today's deliverable

- 3-minute video recorded and shared (YouTube unlisted, Loom, or Drive link).
- 5-slide deck (PDF or Google Slides public link).

---

# Class 8-6 — Final Presentations & Course Graduation

> **The capstone. Each team presents the live system in 5 minutes (3 min video + 2 min Q&A) to the cohort + invited program partners.**

## Format

- 5 minutes per team. Strict.
- Mentor + assistant + invited partner panel.
- Questions afterwards from peers and panel.

## Submission (final, due before the live presentation)

```
Public GitHub repo: github.com/<student>/olist-delivery-intelligence
  ├── README.md                       ← project overview, how to run, link to live URL
  ├── data/                           ← (optional, if dataset is small enough to commit)
  ├── notebooks/                      ← M3-M7 work (clean, headed up)
  │   ├── 01_module3_clean.ipynb
  │   ├── 02_module4_classifier.ipynb
  │   ├── 03_module5_segments.ipynb
  │   ├── 04_module6_nn.ipynb
  │   └── 05_module7_sentiment.ipynb
  ├── pipeline/
  │   ├── train_final_model.py
  │   ├── inference.py
  │   ├── retrain.py
  │   └── final_model.pkl
  ├── app.py                          ← FastAPI service
  ├── Dockerfile
  ├── requirements.txt
  ├── .github/workflows/deploy.yml    ← optional CI
  └── docs/
      ├── architecture.md
      ├── 3min_video_link.md
      └── slides.pdf
```

## Grading rubric — Final (200 points)

| Component | Weight |
|---|---|
| `/predict` endpoint live and correct | 30 |
| GitHub repo runs end-to-end after `git clone + docker build + docker run` | 30 |
| Test AUC on the M3 holdout ≥ 0.70 | 25 |
| Comparison report shows lift from each module's contribution | 20 |
| 3-minute video clear, structured, on time | 25 |
| Live demo + Q&A | 20 |
| Code quality (modular, commented, tested) | 20 |
| README quality (clear setup + usage) | 15 |
| Documentation (architecture.md + reports) | 15 |

## Resume bullet template

(Take this directly to your CV.)

> **Olist E-commerce Delivery & Reviews Intelligence Platform** (BePro AI/ML, 2026)
> End-to-end ML system on the Olist Brazilian e-commerce dataset (100k+ orders).
> Engineered a relational analytical table from 9 raw CSVs · trained a Random-Forest classifier predicting late deliveries (ROC-AUC 0.7X) · segmented customers into 4 RFM cohorts · replaced the classifier with a 3-layer neural net for a Y% lift in F1 · enriched the model with review-text sentiment from a multilingual transformer for an additional Z% gain · deployed the final model as a FastAPI endpoint on Render with logging and re-train pipeline.
> **Stack:** Python · Pandas · NumPy · scikit-learn · TensorFlow · Hugging Face · FastAPI · Docker.
> **Repo:** github.com/<student>/olist-delivery-intelligence

## Course graduation

After the final presentations:

- Mentor announces grades within 1 week.
- Top 3 teams are highlighted on the program's homepage as "Cohort N graduates."
- Each student's repo URL is collected for the program partner report.
- LinkedIn-share template provided (optional).
