# Recession & Inflation Google Trends vs. Gold Price Analysis

## Overview

This project investigates whether rising global online concern about economic
distress — captured through Google Trends search intensity for **"Recession"**
and **"Inflation"** — is associated with subsequent increases in gold prices.

Gold is a classic safe-haven and inflation-hedge asset. If heightened public
anxiety about economic instability reliably *precedes* upward movements in gold
prices, this offers investors a simple, freely available leading indicator.

---

## Data

| File | Description |
|------|-------------|
| `gold-300.xls` | Monthly average gold prices (USD/troy oz) from the **Bank of England** database (Feb 2001 – Jun 2025). HTML table format. |
| `multiTimeline (3).csv` | Global monthly Google Trends index (0–100) for **"Recession"** (Jan 2004 – Mar 2026). |
| `multiTimeline (4).csv` | Global monthly Google Trends index (0–100) for **"Inflation"** (Jan 2004 – Mar 2026). |

Both Trends datasets begin in Jan 2004 (the earliest month Google Trends provides for these terms), which sets the start of the analysis window. The gold price data also ends in Jun 2025, so both Trends datasets are trimmed to match. This gives **258 monthly observations** (Jan 2004 – Jun 2025) for each analysis. The gold data prior to Jan 2004 and the Trends data after Jun 2025 fall outside the overlapping window and are not used.

---

## Analysis Scripts

### `analysis.R` — Recession Trends vs. Gold

### `analysis_inflation.R` — Inflation Trends vs. Gold

Both scripts use **base R only** (no external packages) and perform:

| Step | Method |
|------|--------|
| 1 | Parse `gold-300.xls` (HTML table) and the relevant Trends CSV |
| 2 | Compute monthly gold **percentage returns**: `(P_t / P_{t-1} − 1) × 100` |
| 3 | Plot both time series and monthly returns |
| 4 | **Hypothesis Test 1** – Pearson correlation (contemporaneous) |
| 5 | **Lagged Cross-Correlation (CCF)** – lags −12 to +12 months |
| 6 | **Hypothesis Test 2** – OLS regression: `Gold_Return[t] ~ Trends[t−1]` (one-sided) |
| 7 | **Granger Causality** – F-test at lags 1–3 |

---

## Running the Analysis

```bash
# From the repository root (R must be installed):
Rscript analysis.R              # Recession Trends vs. Gold
Rscript analysis_inflation.R    # Inflation Trends vs. Gold
```

---

## Results

### Analysis 1: "Recession" Trends vs. Gold (Jan 2004 – Jun 2025, n = 258)

| Test | Result |
|------|--------|
| Contemporaneous Pearson r | r = **0.106**, p = 0.090 — *not significant* |
| Best CCF lag (Trends leads gold) | lag −11 months: r = 0.091 — *not significant* |
| 1-month lead regression β₁ | β = 0.011, p (one-sided) = 0.186 — *not significant* |
| Granger causality (lags 1–3) | p ≥ 0.44 at all lags — *no Granger causality* |

**Interpretation:** No statistically significant correlation found. The slight positive contemporaneous correlation (r ≈ 0.11) is consistent with safe-haven theory but does not reach the 5% significance threshold over the full period.

---

### Analysis 2: "Inflation" Trends vs. Gold (Jan 2004 – Jun 2025, n = 258)

| Test | Result |
|------|--------|
| Contemporaneous Pearson r | r = **0.050**, p = 0.428 — *not significant* |
| Best CCF lag (Trends leads gold) | lag −10 months: r = **0.129** — ✅ *significant* (|r| > 0.122) |
| lag −11 months | r = **0.124** — ✅ *significant* |
| 1-month lead regression β₁ | β = 0.006, p (one-sided) = 0.394 — *not significant* |
| Granger causality (lag 2) | F = **3.26**, p = **0.040** — ✅ *significant* |

**Interpretation:** The "Inflation" search signal shows a **weak but statistically significant lagged relationship** with gold returns. The CCF reveals that elevated inflation search interest tends to be followed by modestly positive gold returns roughly 10–11 months later. Granger causality is also detected at 2 lags (p = 0.040), suggesting that past inflation search behaviour contains some predictive information beyond gold's own return history. However, no simple 1-month lead relationship exists, and contemporaneous correlation is negligible.

#### Output plots (Inflation analysis)

| Plot | Description |
|------|-------------|
| `output/inflation_time_series.png` | Three-panel time-series: gold price, monthly return, Inflation Trends |
| `output/inflation_scatter_contemporaneous.png` | Scatter: Inflation Trends vs. same-month gold return |
| `output/inflation_ccf_plot.png` | CCF for lags −12 to +12 months (significant bars in gold) |
| `output/inflation_scatter_lag1.png` | Scatter: Inflation Trends[t−1] vs. Gold Return[t] |

---

## Comparison Summary

| Signal | Contemp. r | Best lag r | Granger (best p) |
|--------|-----------|------------|-----------------|
| "Recession" Trends | 0.106 (p=0.090) | 0.091 @ lag−11 (ns) | p ≥ 0.44 |
| "Inflation" Trends | 0.050 (p=0.428) | **0.129 @ lag−10 ✅** | **p = 0.040 ✅** |

The **"Inflation"** signal provides marginally stronger evidence of a lagged association with gold returns compared to the "Recession" signal, though neither offers a reliable short-term trading signal on its own.