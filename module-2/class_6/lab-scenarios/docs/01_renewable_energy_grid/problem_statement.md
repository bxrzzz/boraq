# Renewable Energy Grid Optimization

| | |
|---|---|
| **Industry** | Energy / Smart Grid |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

A national grid operator runs 200+ wind and solar plants. Power output is volatile (clouds, wind gusts), and the operator must pre-dispatch coal capacity to balance shortfalls. Analyse 1 year of 15-minute-interval generation data to find WHEN and WHERE renewables fall short, and recommend which 3 regions need backup capacity.

---

## 2. Dataset

- **Source:** [Open Power System Data — 15-minute time series](https://data.open-power-system-data.org/time_series/) or Kaggle's *Wind Turbine SCADA Dataset*
- **Volume:** ~7M (200 plants × 35,000 timesteps)

### 2.1 Columns to keep
`timestamp`, `plant_id`, `region`, `tech` (`solar`/`wind`), `generation_mwh`, `capacity_mw`, `wind_speed`, `irradiance`, `temperature_c`

### 2.2 Columns to drop
Forecast columns, telemetry health flags, plant metadata strings

### 2.3 Data hygiene — noise to handle
- Timestamps in mixed formats (`'2026-03-12 14:00'` vs `'12/03/2026 14:00:00.000'`)
- Negative generation values from sensor faults
- `NaN` blocks during maintenance windows

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Recommend 3 regions that need backup capacity, with evidence (charts + tables) showing when shortfalls occur.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
