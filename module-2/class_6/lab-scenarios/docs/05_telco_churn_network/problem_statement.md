# Telecommunications Churn & Network Quality

| | |
|---|---|
| **Industry** | Telecommunications |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

A national telecom (think Uztelecom) wants to find the network conditions that best predict churn. Combine call-detail records (CDR) with daily customer status to score 250k subscribers across 6 months. Recommend the top 3 actionable network features for the retention team.

---

## 2. Dataset

- **Source:** Kaggle [*Telco Customer Churn (IBM)*](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) merged with [*CDR Synthetic Dataset*](https://www.kaggle.com/datasets/sergiosacj/telecom-cdr-synthetic)
- **Volume:** ~250k subscriber-days

### 2.1 Columns to keep
`customer_id`, `tenure`, `monthly_charges`, `total_charges`, `contract`, `payment_method`, `dropped_calls_per_day`, `data_throughput_mbps`, `avg_call_duration_sec`, `cell_handovers`, `Churn`

### 2.2 Columns to drop
Address strings, social ID, raw timestamps not needed for rolling features

### 2.3 Data hygiene — noise to handle
- `total_charges` is a string with whitespace and a few `' '` blanks
- Some tenures are 0 (new customers)
- Duplicates from system retries

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Top 3 network features the retention team can act on, with quantified evidence and a one-line action per feature.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
