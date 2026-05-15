# Logistics & Last-Mile Delivery Chain

| | |
|---|---|
| **Industry** | E-commerce / Logistics |
| **Module** | 2 — Python and Data Foundations |
| **Lab type** | Final lab — student-defined approach |

---

## 1. Business Problem

A regional courier handles 1.5M parcel events/year across 12 warehouses. Late deliveries cost reputation and refunds. Find the routes, hubs, and times of day that drive late delivery. Recommend the 3 routes most worth re-engineering.

---

## 2. Dataset

- **Source:** Kaggle [*Olist Brazilian E-Commerce*](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 6 tables, 100k+ orders
- **Volume:** 100k+ orders across 6 joined tables

### 2.1 Columns to keep
`order_id`, `order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`, `customer_state`, `seller_state`, `freight_value`, `product_weight_g`, `product_volume_cm3`

### 2.2 Columns to drop
Review text columns, raw lat/lon (use the geo file separately), product photo URLs

### 2.3 Data hygiene — noise to handle
- ~3% of `order_delivered_customer_date` is `NaN` (parcel lost in transit)
- Date columns are strings
- Some seller states are uppercase, some lowercase — must standardise

> **Hygiene rule:** clean once, document the rule in code, never silently drop rows. Print row-count before and after every cleaning step. The cleaned dataset must follow the same column conventions as the platform's standard CSVs (lowercase column names, ISO timestamps, explicit `NaN` for missing values).

---

## 3. Deliverable

List the 3 worst routes with quantified late-rate, and a written recommendation for each.

---

## 4. Today's Two Documents

You are NOT writing code today. You are designing the solution.

1. Open `documentation_phase_1_what.md` — describe **WHAT** tools, libraries, and methods you propose to use.
2. Open `documentation_phase_2_why.md` — explain **WHY** each of those tools and methods is the right choice for this specific scenario.

Both docs are submitted as part of the team's first-week deliverable.
