# Health Informatics — Hospital Readmission Risk

| | |
|---|---|
| **Industry** | Healthcare / Public Health |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

A hospital network wants to know which diabetes patients return within 30 days and what their treatment patterns look like. 101k anonymised admissions across 130 US hospitals over 10 years. Identify the 5 features most associated with 30-day readmission.

---

## 2. Dataset

- **Source:** [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Volume:** 101,766 rows × 50 columns

### 2.1 Columns to keep
`encounter_id`, `patient_nbr`, `age`, `gender`, `admission_type_id`, `time_in_hospital`, `num_lab_procedures`, `num_medications`, `diabetesMed`, `A1Cresult`, `readmitted`

### 2.2 Columns to drop
`weight` (97% missing), `payer_code` (40% missing), free-text fields

### 2.3 Data hygiene — noise to handle
- `?` placeholders for missing categorical values
- `age` stored as ranges (`'[70-80)'`)
- ICD-9 diagnosis codes need decoding to a disease group

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Ranked list of 5 features most associated with readmission, with a one-line clinical interpretation each.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
