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
| `gold-300.xls` | Monthly average gold prices (USD/troy oz) from the **Bank of England** database (Feb 2001 – Jun 2025). HTML table format. |
| `multiTimeline (3).csv` | Global monthly Google Trends index (0–100) for the search term **"Recession"** exported from [trends.google.com](https://trends.google.com) (Jan 2004 – Mar 2026). |

The two datasets are merged on the overlapping period **Jan 2004 – Jun 2025**, giving **258 monthly observations**.

---

## Analysis (`analysis.R`)

The script performs the following steps using **base R only** (no external
packages required):

| Step | Method |
|------|--------|
| 1 | Parse `gold-300.xls` (HTML table) and `multiTimeline (3).csv` |
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
# From the repository root (R must be installed):
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

## Results (Real Data, Jan 2004 – Jun 2025, n = 258)

| Test | Result |
|------|--------|
| Contemporaneous Pearson r | r = **0.106**, p = 0.090 — *not significant* at α = 0.05 |
| Best CCF lag (Trends leads gold) | lag −11 months: r = 0.091 — *not significant* |
| 1-month lead regression β₁ | β = 0.011, p (one-sided) = 0.186 — *not significant* |
| Granger causality (lags 1–3) | p ≥ 0.44 at all lags — *no Granger causality detected* |

### Interpretation

The analysis finds **no statistically significant correlation** between Google Trends search intensity for "Recession" and monthly gold price returns at any tested lead or lag. While the contemporaneous correlation (r ≈ 0.11) is positive — consistent with the hypothesis that recession anxiety and gold price rises tend to coincide — it does not reach the conventional 5% significance threshold.

This does not rule out a relationship under specific market stress regimes (e.g., the 2008 GFC or 2020 COVID crash), but over the full 2004–2025 period the "Recession" search signal alone is not a reliable linear predictor of gold return direction.