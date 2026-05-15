# Algorithmic Trading Backtest (Equities)

| | |
|---|---|
| **Industry** | Financial Markets |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

Build the analytical layer of a momentum-trading backtester. Use 10 years of daily OHLCV for the S&P 500 to characterise factor returns and drawdowns. Report the best sector regime, worst drawdown, and the strategy's Sharpe ratio.

---

## 2. Dataset

- **Source:** Kaggle [*S&P 500 Stocks (Daily Updated)*](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks)
- **Volume:** ~1.2M (500 tickers × 10 years of daily data)

### 2.1 Columns to keep
`Date`, `Symbol`, `Open`, `High`, `Low`, `Close`, `Volume`, `Sector`

### 2.2 Columns to drop
Adjusted-close (compute it yourself), share splits, dividends file

### 2.3 Data hygiene — noise to handle
- Trading-day gaps (weekends, holidays)
- A few tickers de-listed mid-period — drop survivorship bias cleanly

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

A trading-strategy report: sector regime, drawdown chart, Sharpe ratio, and 3 sentences on what worked and what didn't.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
