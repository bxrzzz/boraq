# Agriculture — Crop Yield from Soil & Weather

| | |
|---|---|
| **Industry** | AgriTech / Food Security |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

10 years of monthly soil-moisture, temperature, and precipitation across 1,000 districts (~1.2M rows) plus annual yield data. Surface the climate factors most predictive of yield. Report the top 3 climate features driving yield in each region.

---

## 2. Dataset

- **Source:** Kaggle [*FAOSTAT Crop & Climate*](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) + NASA POWER monthly time-series
- **Volume:** ~1.2M (1,000 districts × 120 months × multiple crops)

### 2.1 Columns to keep
`Year`, `Country`, `District`, `Crop`, `yield_hg_per_ha`, `avg_temp`, `precipitation`, `soil_moisture`, `pesticides_tonnes`

### 2.2 Columns to drop
Crops with <100 observations; non-cereal exotic crops

### 2.3 Data hygiene — noise to handle
- Units mixed (kg vs tonnes)
- Some districts have 2 names (admin merger)
- Precipitation has clear sensor-error spikes

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Top 3 climate features per region driving yield, with charts and one-paragraph policy implications.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
