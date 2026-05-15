# Aviation — U.S. Flight Delay Hotspots

| | |
|---|---|
| **Industry** | Aviation / Operations Research |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

5.8M+ US flights for 1 year. The FAA wants to know **which 5 origin airports contribute the most delay to the U.S. flight system**, and how that delay spreads downstream to their busiest destinations.

Your job:
1. Rank origin airports by their mean `DEPARTURE_DELAY`.
2. For each of the top 5, look at where their flights go — which destinations are most affected?
3. Recommend where the FAA should focus operational improvements.

---

## 2. Dataset

- **Source:** Kaggle [*US Flights 2015*](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Volume:** 5,819,079 flights

### 2.1 Columns to keep
`YEAR`, `MONTH`, `DAY`, `AIRLINE`, `FLIGHT_NUMBER`, `TAIL_NUMBER`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `SCHEDULED_DEPARTURE`, `DEPARTURE_DELAY`, `ARRIVAL_DELAY`, `DISTANCE`, `LATE_AIRCRAFT_DELAY`

### 2.2 Columns to drop
Any column >40% null (`CANCELLATION_REASON` etc.)

### 2.3 Data hygiene — noise to handle
- Times stored as `HHMM` integer (`'845'` = 08:45)
- Cancelled flights have negative delays
- Tail numbers occasionally null
- Dataset is sorted by date — random-sample if you reduce row count, don't take the first N

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. Use lowercase column names, ISO timestamps, explicit `NaN` for missing values.

---

## 3. Methods (what's expected — Module 2 toolkit only)

You only need what you've learned in Module 2:
- **pandas** — `read_csv`, `merge`, `groupby`, `mean`, `sort_values`, `sample`
- **matplotlib / seaborn** — bar charts, optionally a geographic scatter

You do NOT need graph libraries (NetworkX, igraph) or machine learning. The whole analysis is pandas groupby + ranking + matplotlib.

---

## 4. Deliverable

1. **Top 5 origin airports** by mean `DEPARTURE_DELAY` — printed table + bar chart.
2. **Spillover view** for each of the top 5 — for each origin, a chart (or table) of its top 10 destinations by mean `ARRIVAL_DELAY`. This shows where each "delay-heavy" airport propagates its problem.
3. **One-paragraph recommendation** tied to your specific top 5 — what should the FAA prioritize, and why?
4. **Cleaning log** — row counts before/after every cleaning step, visible in the notebook output.

Bonus (optional):
- Break the top 5 down by month — are some airports worse in winter vs summer?
- Break the top 5 down by airline — is one carrier driving the delay at one airport?

---

## 5. Today's Two Documents

For the first session, you are **designing** the solution, not coding it (unless your mentor asks you to clean the data).

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable. Mentor reviews and signs off before code begins.
