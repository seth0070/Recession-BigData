# Recession-BigData

## Time-Series Analysis: Google Trends Inflation Search Interest vs Gold Returns

### Research Question

Does rising public concern about inflation, captured through Google Trends search intensity for the term "inflation", predict subsequent increases in monthly gold returns over a medium-term horizon of 10–12 months?

### Repository Contents

| File | Description |
|------|-------------|
| `analysis.R` | Complete R analysis script (Steps 1–9) |
| `multiTimeline.csv` | Google Trends monthly search index for "inflation" (Jan 2004–Mar 2026) |
| `gold-300.xls` | Monthly gold prices in USD/troy oz, HTML table format (Jan 2001–Feb 2026) |

### Required R Packages

```r
install.packages(c("tidyverse", "lubridate", "rvest", "tseries", "lmtest", "forecast", "ggplot2"))
```

### How to Run

```bash
Rscript analysis.R
```

This will run all 9 analysis steps and generate:

- `analysis_dataset.csv` — Clean merged dataset (254 observations)
- `qq_plots.png` — Q-Q normality plots
- `ccf_base_plot.png` — Base R CCF plot
- `arimax_residuals.png` — ARIMAX residual diagnostics
- `figure1_timeseries.png` — Dual-axis time series (search index + gold returns)
- `figure2_ccf.png` — ggplot2 styled CCF plot with significance highlights
- `figure3_scatter.png` — Scatter plot at best lag with regression line
- `figure4_terciles.png` — Bar chart of gold returns by search interest tercile

### Analysis Steps

1. **Data Loading & Cleaning** — Load Google Trends CSV and gold price HTML table, parse dates, merge by month
2. **Feature Engineering** — Compute monthly gold % returns, create lagged search index (lags 1–12)
3. **Assumption Checking** — Shapiro-Wilk normality, ADF stationarity, Spearman/Pearson correlation
4. **Cross-Correlation (CCF)** — Test lags −12 to +12, identify significant predictive lags
5. **Lagged Regression** — Simple OLS at lags 1, 6, 10, 11, 12; multiple regression with controls
6. **Granger Causality** — Test whether search index Granger-causes gold returns (lags 1–12)
7. **ARIMAX Model** — Compare ARIMA with and without search index as external regressor
8. **Visualisations** — 4 publication-quality ggplot2 figures
9. **Results Summary** — Comprehensive statistics table and plain English interpretation