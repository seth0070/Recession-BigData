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

> **🧒 ELI5 — What does "Granger causality at lag 2" actually mean?**
>
> Imagine you are trying to guess how much gold will gain or lose *this month*.
> One approach is to look at how gold itself moved in the last couple of months —
> that is already useful.
>
> The **Granger causality test** asks a different question:
> *"If I also peek at how many people were Googling the word 'inflation' over the
> last two months, can I make a better prediction than if I only used gold's own
> history?"*
>
> At **lag 2** the answer is **yes** (p = 0.040, which is just below the standard
> 5% threshold). In plain English:
>
> > *Knowing the inflation search trend from **two months ago** and **one month ago**
> > together gives you a small but statistically real improvement in predicting
> > gold's return this month — on top of what gold's own past returns already tell you.*
>
> **Why does this matter?**
> Google searches are free, public, and available in real time.
> If past search behaviour genuinely contains information about future gold moves,
> it could act as a low-cost early-warning signal.
>
> **But don't rush to trade on it.** A few important caveats:
> - The effect is *weak* — the R² of the regression barely budges.
> - "Granger causality" is a statistical term; it does **not** mean inflation
>   searches *cause* gold prices to rise in the everyday sense of the word.
>   It just means the search data is not redundant — it adds a little signal.
> - The test is significant at lag 2 but not at lags 1 or 3, which suggests the
>   pattern could be a fluke of this particular dataset rather than a robust law.
> - Over a 21-year period with many different economic regimes, the relationship
>   may be driven mainly by a few intense episodes (e.g. the 2022 inflation spike)
>   rather than holding consistently month to month.

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

---

## 🧒 Plain-English Glossary

For readers who are not statisticians, here is what each test in the analysis actually measures:

| Term | What it means in plain English |
|------|-------------------------------|
| **Pearson correlation (r)** | A number between −1 and +1 that measures how closely two things move together *at the same time*. r = 1 means they rise and fall in perfect lockstep; r = 0 means no relationship at all. |
| **p-value** | The probability that you would see a result this extreme *purely by chance* if there were really no relationship. A p-value below 0.05 is the conventional threshold for calling a result "statistically significant". |
| **Lag** | A time offset. "Lag −10" means we compare Inflation Trends from 10 months *earlier* with gold returns *today*. Negative lags test whether Trends *predicts* future gold moves. |
| **CCF (Cross-Correlation Function)** | Like Pearson correlation, but tested at every possible time offset (lag). The CCF plot shows which lag, if any, gives the strongest link between the two variables. |
| **OLS regression (β)** | Fits a straight line through the data. β (beta) is the slope — how much gold return we expect to change for each one-unit increase in the Trends index. If β > 0, higher search interest is associated with higher gold returns. |
| **Granger causality** | A statistical test that asks: "Does knowing the past values of variable X help me predict variable Y *better* than using Y's own past alone?" It does **not** mean X truly causes Y in a physical sense — just that X contains extra useful information. |
| **F-statistic** | The score produced by the Granger test. A larger F means the extra variable (Trends) adds more predictive power. The p-value tells you whether that improvement is too large to be a fluke. |
| **Lag 2 (in Granger context)** | The test uses the last **2 months** of Trends data to try to predict this month's gold return. Significant at lag 2 (p = 0.040) means the two-month history of inflation searches adds real (if small) predictive value beyond gold's own history. |
