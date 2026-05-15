# Smart City Air Quality & Traffic

| | |
|---|---|
| **Industry** | IoT / Public Sector |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

A city deploys 50 air-quality sensors and 30 traffic counters. Find which roads' traffic best predicts which sensors' air quality, lagged by minutes — and the best lag value.

---

## 2. Dataset

- **Source:** Kaggle [*Beijing PM2.5 Multi-Site*](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set) (420k+ rows × 12 sites)
- **Volume:** ~420k+ multi-site readings

### 2.1 Columns to keep
`datetime`, `station`, `PM2.5`, `PM10`, `NO2`, `CO`, `temp`, `wind_speed`, `traffic_count` (synthetic if missing)

### 2.2 Columns to drop
Station metadata, columns with >50% null

### 2.3 Data hygiene — noise to handle
- Sensor outages → long NaN runs
- One station has time in UTC, others in local
- Wind direction in degrees needs sin/cos encoding

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Identify the road most predictive of which sensor's PM2.5, and the time lag in minutes. One paragraph for the city traffic department.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
