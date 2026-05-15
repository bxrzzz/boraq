# Sports Analytics — Football Match Performance

| | |
|---|---|
| **Industry** | Sports Analytics |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

Match-level events from a top European football league across 5 seasons. Build a 'form curve' per team and identify the events most correlated with winning streaks. Surface the team most over-performing its expected level this season.

---

## 2. Dataset

- **Source:** Kaggle [*European Soccer Database*](https://www.kaggle.com/datasets/hugomathien/soccer)
- **Volume:** 25k matches + 11k players (SQLite)

### 2.1 Columns to keep
`match_api_id`, `date`, `home_team_api_id`, `away_team_api_id`, `home_team_goal`, `away_team_goal`, `B365H`, `B365D`, `B365A`, plus the 50+ in-match event columns

### 2.2 Columns to drop
Raw XML player attributes (parse separately), unused odds providers

### 2.3 Data hygiene — noise to handle
- SQL date strings
- A few seasons missing odds
- Team IDs change in some rows

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

Name the team most over-performing this season, with a one-paragraph explanation backed by 2 charts.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
