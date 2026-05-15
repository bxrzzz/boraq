# E-commerce Customer Lifetime Value (RFM)

| | |
|---|---|
| **Industry** | Retail / E-commerce |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

An online retailer wants to segment its 1M+ orders into RFM cohorts (Recency, Frequency, Monetary value) and visualise lifetime value by cohort. Pick the top 3 cohorts to target with a reactivation campaign.

---

## 2. Dataset

- **Source:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Volume:** 1,067,371 transactions across 2 years

### 2.1 Columns to keep
`Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

### 2.2 Columns to drop
Rows with `Customer ID` NaN, refunds (negative quantity) — but report counts dropped

### 2.3 Data hygiene — noise to handle
- Thousands of free-text product descriptions
- Some prices are 0
- UK/EU date formats inconsistent

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Top 3 RFM cohorts ranked by reactivation potential, with a recommended marketing offer per cohort.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
