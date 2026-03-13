# Recession Google Trends & Gold Price Analysis

## Overview

This project investigates whether rising global online concern about potential
stock market crashes — captured through Google Trends search intensity for the
term **"Recession"** — is associated with subsequent increases in gold prices,
and whether this signal can predict the direction of gold price movements in the
following month.

Gold has historically been regarded as a safe-haven asset that attracts
investors during periods of financial uncertainty. If heightened public anxiety
about market crashes (as expressed through search behaviour) reliably *precedes*
upward movements in gold prices, this offers investors a simple, freely
available leading indicator.

---

## Data

| File | Description |
|------|-------------|
| `data/gold_prices.csv` | Monthly average gold prices (USD/troy oz) from the **Bank of England** database.<br>Columns: `Date` (YYYY-MM-DD), `Price_USD` |
| `data/google_trends_recession.csv` | Global monthly Google Trends index for the search term **"Recession"** (scale 0–100).<br>Columns: `Month` (YYYY-MM), `Recession_Trend` |

> **Note:** The repository ships with **sample data** (Jan 2004 – Dec 2023)
> generated for demonstration purposes.  
> To run the analysis on real data, replace the CSV files with:
> - Gold prices downloaded from the [Bank of England Statistical Interactive Dataset](https://www.bankofengland.co.uk/boeapps/database/)
> - Google Trends data exported from [trends.google.com](https://trends.google.com) for the search term "Recession" (worldwide, monthly)

---

## Analysis (`analysis.R`)

The script performs the following steps using **base R only** (no external
packages required):

| Step | Method |
|------|--------|
| 1 | Load and preprocess gold prices and Google Trends data |
| 2 | Compute monthly gold **percentage returns**: `(P_t / P_{t-1} − 1) × 100` |
| 3 | Plot both time series and monthly returns |
| 4 | **Hypothesis Test 1** – Pearson correlation between contemporaneous Trends and gold return |
| 5 | **Lagged Cross-Correlation (CCF)** – identify whether Trends leads or lags gold returns up to ±12 months |
| 6 | **Hypothesis Test 2** – OLS regression: `Gold_Return[t] ~ Recession_Trend[t−1]` with one-sided test |
| 7 | **Granger Causality** – F-test for whether past Trends values improve prediction of gold returns (lags 1–3) |

### Hypotheses

**H₀ (contemporaneous):** `cor(Recession_Trend[t], Gold_Return[t]) = 0`  
**H₁:** The correlation is non-zero (two-sided)

**H₀ (predictive):** `β₁ = 0` — Trends at `t−1` has no effect on Gold Return at `t`  
**H₁:** `β₁ > 0` — Higher Trends at `t−1` predicts a positive Gold Return at `t`

---

## Running the Analysis

```bash
# From the repository root:
Rscript analysis.R
```

Results are printed to the console and four plots are saved to `output/`:

| Plot | Description |
|------|-------------|
| `output/time_series.png` | Three-panel time-series: gold price level, monthly return, and Trends index |
| `output/scatter_contemporaneous.png` | Scatter of Trends vs. same-month gold return with regression line |
| `output/ccf_plot.png` | Cross-correlation function for lags −12 to +12 months |
| `output/scatter_lag1.png` | Scatter of Trends[t−1] vs. Gold Return[t] (predictive direction) |

---

## Interpreting the Results

- A statistically significant **positive** contemporaneous correlation (p < 0.05)
  would suggest that months with high recession anxiety coincide with rising gold prices.
- Significant **negative** lags in the CCF (e.g., lag −1) would indicate that
  elevated Trends *precede* gold price increases — a potential leading indicator.
- A significant positive `β₁` in the lag-1 regression provides actionable evidence
  that last month's search intensity predicts a higher gold return this month.
- Significant Granger causality (p < 0.05 for any lag) strengthens the case that
  the Trends signal contains information beyond gold's own price history.