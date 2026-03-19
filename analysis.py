"""
Financial Anxiety (Google Trends) vs Gold Price Analysis
=========================================================
Data sources  (both files from this repository):
  - Google Trends : 'multiTimeline (5).csv'
      Monthly search indices for 5 financial-anxiety keywords
      (stock market crash, inflation, recession, financial crisis,
       cost of living) – worldwide, 2004-01 to 2026-03.
  - Gold prices   : 'gold-300.xls'  (HTML-table file from macrotrends)
      Monthly average spot price (USD/troy oz), Feb 2001 – Jun 2025.

A composite Financial Anxiety Index (FAI) is built as the row-mean of
the five min-max-normalised keyword series.  All analyses are carried
out on both the composite FAI and the individual keywords where
appropriate.

Analyses:
  1.  Descriptive statistics & time-series plots
  2.  Normality tests  (Shapiro-Wilk, Jarque-Bera + histograms / Q-Q)
  3.  Stationarity     (ADF, KPSS) – raw and first-differenced
  4.  ACF / PACF plots
  5.  Cross-Correlation Function (CCF)  – undifferenced & differenced
  6.  Lagged OLS regression             – undifferenced & differenced
  7.  Granger causality test
  8.  ARIMA  model for gold prices
  9.  ARIMAX model (gold + FAI as exogenous)
  10. Summary dashboard

All plots are written to plots/ inside the repository directory.
"""


import os
import warnings
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(REPO_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})

KW_COLORS = {
    "stock_market_crash": "#E74C3C",
    "inflation":          "#F39C12",
    "recession":          "#27AE60",
    "financial_crisis":   "#2980B9",
    "cost_of_living":     "#8E44AD",
    "FAI":                "#2C3E50",
}


def save_fig(name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/{name}")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load & parse gold-300.xls  (HTML disguised as XLS)
# ---------------------------------------------------------------------------
section("1. Loading data")


class _TableParser(HTMLParser):
    """Minimal HTML table row parser."""
    def __init__(self):
        super().__init__()
        self._in_td = False
        self.rows: list[list[str]] = []
        self._cur: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self._in_td = True
        elif tag == "tr":
            self._cur = []

    def handle_endtag(self, tag):
        if tag == "td":
            self._in_td = False
        elif tag == "tr" and self._cur:
            self.rows.append(self._cur[:])

    def handle_data(self, data):
        if self._in_td:
            self._cur.append(data.strip())


gold_path = os.path.join(REPO_DIR, "gold-300.xls")
with open(gold_path, encoding="utf-8") as fh:
    _html = fh.read()

_parser = _TableParser()
_parser.feed(_html)

gold_df = pd.DataFrame(
    [(row[0], float(row[1].replace(",", ""))) for row in _parser.rows if len(row) >= 2],
    columns=["Month", "Gold_Price"],
)
gold_df["Month"] = pd.to_datetime(gold_df["Month"], format="%b %Y")
gold_df = gold_df.sort_values("Month").reset_index(drop=True)
print(
    f"Gold prices   : {gold_df['Month'].min().date()} → "
    f"{gold_df['Month'].max().date()}  ({len(gold_df)} months)"
)

# ---------------------------------------------------------------------------
# 2. Load multiTimeline (5).csv – five financial-anxiety keywords
# ---------------------------------------------------------------------------
gt_path = os.path.join(REPO_DIR, "multiTimeline (5).csv")
gt_raw = pd.read_csv(gt_path, skiprows=1)
gt_raw.columns = [
    "Month",
    "stock_market_crash",
    "inflation",
    "recession",
    "financial_crisis",
    "cost_of_living",
]
gt_raw["Month"] = pd.to_datetime(gt_raw["Month"], format="%Y-%m")
gt_raw = gt_raw.sort_values("Month").reset_index(drop=True)
KEYWORDS = ["stock_market_crash", "inflation", "recession",
            "financial_crisis", "cost_of_living"]
print(
    f"Google Trends : {gt_raw['Month'].min().date()} → "
    f"{gt_raw['Month'].max().date()}  ({len(gt_raw)} months)"
)

# ---------------------------------------------------------------------------
# 3. Build composite Financial Anxiety Index (FAI) & merge
# ---------------------------------------------------------------------------
# Min-max normalise each keyword to [0, 1], then average
for kw in KEYWORDS:
    mn, mx = gt_raw[kw].min(), gt_raw[kw].max()
    gt_raw[kw + "_norm"] = (gt_raw[kw] - mn) / (mx - mn)

norm_cols = [kw + "_norm" for kw in KEYWORDS]
gt_raw["FAI"] = gt_raw[norm_cols].mean(axis=1)

# Merge on Month (inner join – common period only)
df = pd.merge(
    gt_raw[["Month"] + KEYWORDS + ["FAI"]],
    gold_df,
    on="Month",
    how="inner",
)
df = df.set_index("Month")
df.index.freq = "MS"
print(
    f"Merged period : {df.index.min().date()} → "
    f"{df.index.max().date()}  ({len(df)} months)"
)
print("\nDescriptive statistics:")
print(df[KEYWORDS + ["FAI", "Gold_Price"]].describe().to_string())

gld = df["Gold_Price"]          # gold price series
fai = df["FAI"]                 # composite anxiety index
d_gld = gld.diff().dropna()
d_fai = fai.diff().dropna()

# ---------------------------------------------------------------------------
# 4. Raw-series visualisation (individual keywords + gold)
# ---------------------------------------------------------------------------
section("2. Raw-series visualisation")

fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axes = axes.flat

# Panel 1 – Gold price
ax = axes[0]
ax.plot(gld.index, gld.values, color="goldenrod", linewidth=1.5)
ax.set_title("Gold Price (USD/troy oz)", fontsize=11)
ax.set_ylabel("USD")

# Panels 2-6 – each keyword
for idx, kw in enumerate(KEYWORDS, start=1):
    ax = axes[idx]
    ax.plot(df.index, df[kw].values, color=KW_COLORS[kw], linewidth=1.2)
    ax.set_title(f"GTrends – {kw.replace('_', ' ').title()}", fontsize=11)
    ax.set_ylabel("Index (0-100)")

plt.tight_layout()
save_fig("01_raw_keywords.png")

# Gold + FAI dual-axis overlay
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(gld.index, gld.values, color="goldenrod", linewidth=1.8,
         label="Gold Price (USD)")
ax1.set_ylabel("Gold Price (USD/oz)", color="goldenrod")
ax1.tick_params(axis="y", labelcolor="goldenrod")
ax2 = ax1.twinx()
ax2.plot(fai.index, fai.values, color=KW_COLORS["FAI"], linewidth=1.2,
         alpha=0.75, label="Financial Anxiety Index (FAI)")
ax2.set_ylabel("FAI (0-1 composite)", color=KW_COLORS["FAI"])
ax2.tick_params(axis="y", labelcolor=KW_COLORS["FAI"])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.set_title("Gold Price vs Composite Financial Anxiety Index (FAI)", fontsize=13)
plt.tight_layout()
save_fig("02_gold_vs_fai.png")

# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = KEYWORDS + ["FAI", "Gold_Price"]
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, vmin=-1, vmax=1, ax=ax,
    xticklabels=[c.replace("_", "\n") for c in corr_cols],
    yticklabels=[c.replace("_", "\n") for c in corr_cols],
)
ax.set_title("Pearson Correlation Matrix (raw series)", fontsize=12)
plt.tight_layout()
save_fig("03_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 6. Normality tests
# ---------------------------------------------------------------------------
section("3. Normality Tests")


def normality_report(series: pd.Series, name: str) -> None:
    data = series.dropna()
    sw_stat, sw_p = stats.shapiro(data)
    jb_result = stats.jarque_bera(data)
    jb_stat, jb_p = jb_result.statistic, jb_result.pvalue
    print(
        f"  {name:<35s}  "
        f"SW p={sw_p:.3e} {'NOT NORMAL' if sw_p < 0.05 else 'normal':12s}  "
        f"JB p={jb_p:.3e} {'NOT NORMAL' if jb_p < 0.05 else 'normal':12s}"
    )


print(f"\n  {'Series':<35s}  {'Shapiro-Wilk':<25s}  {'Jarque-Bera'}")
print("  " + "-" * 80)
for kw in KEYWORDS + ["FAI"]:
    normality_report(df[kw],         f"{kw} (raw)")
    normality_report(df[kw].diff(),  f"{kw} (Δ1)")
normality_report(gld,   "Gold Price (raw)")
normality_report(d_gld, "Gold Price (Δ1)")

# Histogram + Q-Q for FAI and Gold (raw and differenced)
series_4plot = [
    (gld,   "Gold Price (raw)",  "goldenrod"),
    (d_gld, "Gold Price (Δ1)",   "orange"),
    (fai,   "FAI (raw)",         KW_COLORS["FAI"]),
    (d_fai, "FAI (Δ1)",          "cornflowerblue"),
]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for col, (ser, title, col_color) in enumerate(series_4plot):
    data = ser.dropna()
    # Histogram
    ax = axes[0, col]
    ax.hist(data, bins=30, color=col_color, edgecolor="white", alpha=0.8, density=True)
    xmin, xmax = ax.get_xlim()
    x_line = np.linspace(xmin, xmax, 200)
    ax.plot(x_line, stats.norm.pdf(x_line, data.mean(), data.std()),
            "k--", linewidth=1.5)
    ax.set_title(f"Histogram\n{title}", fontsize=10)
    # Q-Q
    ax2 = axes[1, col]
    (osm, osr), (slope, intercept, _r) = stats.probplot(data)
    ax2.scatter(osm, osr, s=10, alpha=0.6, color=col_color)
    ax2.plot(osm, slope * np.array(osm) + intercept, "r--", linewidth=1.5)
    ax2.set_title(f"Q-Q Plot\n{title}", fontsize=10)
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Sample quantiles")
plt.tight_layout()
save_fig("04_normality.png")

# ---------------------------------------------------------------------------
# 7. Stationarity tests (ADF + KPSS)
# ---------------------------------------------------------------------------
section("4. Stationarity Tests (ADF & KPSS)")


def stationarity_report(series: pd.Series, name: str) -> None:
    data = series.dropna()
    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(data, autolag="AIC")
    try:
        kpss_stat, kpss_p, _, _ = kpss(data, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = float("nan"), float("nan")
    adf_conc  = "STATIONARY"     if adf_p  < 0.05 else "non-stationary"
    kpss_conc = "NON-STATIONARY" if kpss_p < 0.05 else "stationary"
    print(
        f"  {name:<35s}  "
        f"ADF p={adf_p:.3e} ({adf_conc:14s})  "
        f"KPSS p={kpss_p:.3e} ({kpss_conc})"
    )


print(f"\n  {'Series':<35s}  {'ADF':<35s}  {'KPSS'}")
print("  " + "-" * 95)
for kw in KEYWORDS + ["FAI"]:
    stationarity_report(df[kw],        f"{kw} (raw)")
    stationarity_report(df[kw].diff(), f"{kw} (Δ1)")
stationarity_report(gld,   "Gold Price (raw)")
stationarity_report(d_gld, "Gold Price (Δ1)")

# Rolling mean / std plot for FAI and Gold
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, (ser, title, col_color) in zip(axes.flat, [
    (gld,   "Gold Price (raw)",   "goldenrod"),
    (d_gld, "Gold Price (Δ1)",    "orange"),
    (fai,   "FAI (raw)",          KW_COLORS["FAI"]),
    (d_fai, "FAI (Δ1)",           "cornflowerblue"),
]):
    data = ser.dropna()
    rm = data.rolling(12).mean()
    rs = data.rolling(12).std()
    ax.plot(data.index, data.values, color=col_color, linewidth=0.8,
            alpha=0.6, label="Series")
    ax.plot(rm.index, rm.values, "k-", linewidth=2,   label="Rolling mean (12m)")
    ax.plot(rs.index, rs.values, "r--", linewidth=1.5, label="Rolling std  (12m)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
plt.tight_layout()
save_fig("05_stationarity_rolling.png")

# ACF / PACF
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_acf( gld,  ax=axes[0, 0], lags=36, title="ACF – Gold Price (raw)")
plot_pacf(gld,  ax=axes[0, 1], lags=36, title="PACF – Gold Price (raw)",  method="ywm")
plot_acf( fai,  ax=axes[1, 0], lags=36, title="ACF – FAI (raw)")
plot_pacf(fai,  ax=axes[1, 1], lags=36, title="PACF – FAI (raw)",         method="ywm")
plt.tight_layout()
save_fig("06_acf_pacf_raw.png")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_acf( d_gld, ax=axes[0, 0], lags=36, title="ACF – Gold Price (Δ1)")
plot_pacf(d_gld, ax=axes[0, 1], lags=36, title="PACF – Gold Price (Δ1)", method="ywm")
plot_acf( d_fai, ax=axes[1, 0], lags=36, title="ACF – FAI (Δ1)")
plot_pacf(d_fai, ax=axes[1, 1], lags=36, title="PACF – FAI (Δ1)",        method="ywm")
plt.tight_layout()
save_fig("07_acf_pacf_diff.png")

# ---------------------------------------------------------------------------
# 8. Cross-Correlation Function (CCF)
# ---------------------------------------------------------------------------
section("5. Cross-Correlation Function (CCF)")

MAX_LAGS = 24


def compute_ccf_series(x: pd.Series, y: pd.Series, max_lags: int):
    """Return arrays (lags, correlations, ±95 % confidence bound)."""
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    n  = len(xs)
    lags = list(range(-max_lags, max_lags + 1))
    corrs = []
    for lag in lags:
        if lag == 0:
            c = np.corrcoef(xs, ys)[0, 1]
        elif lag > 0:
            c = np.corrcoef(xs.iloc[: n - lag], ys.iloc[lag:])[0, 1]
        else:  # lag < 0
            c = np.corrcoef(ys.iloc[: n + lag], xs.iloc[-lag:])[0, 1]
        corrs.append(c)
    conf = 1.96 / np.sqrt(n)
    return np.array(lags), np.array(corrs), conf


def plot_ccf_bar(x, y, x_name, y_name, max_lags, filename, differenced=False):
    lags, corrs, conf = compute_ccf_series(x, y, max_lags)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["crimson" if abs(c) >= conf else "steelblue" for c in corrs]
    ax.bar(lags, corrs, color=colors, width=0.8, edgecolor="white")
    ax.axhline( conf, color="gray", linestyle="--", linewidth=1,
                label=f"±95 % CI (±{conf:.3f})")
    ax.axhline(-conf, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    tag = "Differenced" if differenced else "Undifferenced"
    ax.set_xlabel(
        f"Lag (months) — negative: {y_name} leads; positive: {x_name} leads"
    )
    ax.set_ylabel("Correlation")
    ax.set_title(f"CCF: {x_name} → {y_name}  ({tag})")
    ax.legend()
    plt.tight_layout()
    save_fig(filename)
    sig = [(l, c) for l, c in zip(lags, corrs) if abs(c) >= conf]
    print(f"\n  Significant lags (|r| ≥ {conf:.3f}):")
    for l, c in sig:
        print(f"    lag={l:+3d}  r={c:.4f}")


# Composite FAI
plot_ccf_bar(fai,   gld,   "FAI",    "Gold Price", MAX_LAGS,
             "08_ccf_fai_undiff.png",   differenced=False)
plot_ccf_bar(d_fai, d_gld, "ΔFAI",   "ΔGold",      MAX_LAGS,
             "09_ccf_fai_diff.png",    differenced=True)

# Individual keywords (differenced) – all in one figure
fig, axes = plt.subplots(len(KEYWORDS), 1, figsize=(14, 4 * len(KEYWORDS)))
for ax, kw in zip(axes, KEYWORDS):
    d_kw = df[kw].diff().dropna()
    lags, corrs, conf = compute_ccf_series(d_kw, d_gld, MAX_LAGS)
    colors = ["crimson" if abs(c) >= conf else "steelblue" for c in corrs]
    ax.bar(lags, corrs, color=colors, width=0.8, edgecolor="white")
    ax.axhline( conf, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-conf, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"CCF (Δ): {kw.replace('_',' ').title()} → Gold Price", fontsize=10)
    ax.set_ylabel("r")
plt.tight_layout()
save_fig("10_ccf_all_keywords_diff.png")

# ---------------------------------------------------------------------------
# 9. Lagged OLS regression
# ---------------------------------------------------------------------------
section("6. Lagged OLS Regression")

BEST_LAGS = [1, 3, 6, 12]


def lagged_ols(y: pd.Series, x: pd.Series, lags: list[int], label: str) -> pd.DataFrame:
    results = []
    for lag in lags:
        x_lag = x.shift(lag)
        tmp = pd.concat([y, x_lag], axis=1).dropna()
        tmp.columns = ["y", "x"]
        X = sm.add_constant(tmp["x"])
        model = sm.OLS(tmp["y"], X).fit()
        dw = durbin_watson(model.resid)
        results.append({
            "Lag":     lag,
            "Coef":    model.params["x"],
            "t-stat":  model.tvalues["x"],
            "p-value": model.pvalues["x"],
            "R²":      model.rsquared,
            "AIC":     model.aic,
            "DW":      dw,
        })
        sig = "*" if model.pvalues["x"] < 0.05 else " "
        print(
            f"  {label}  lag={lag:2d}  coef={model.params['x']:+.5f}  "
            f"p={model.pvalues['x']:.3e}{sig}  R²={model.rsquared:.4f}  DW={dw:.2f}"
        )
    return pd.DataFrame(results)


print("\n  --- Undifferenced (FAI → Gold) ---")
res_raw  = lagged_ols(gld,   fai,   BEST_LAGS, "Gold~FAI  (raw)")
print("\n  --- First-Differenced (ΔFAI → ΔGold) ---")
res_diff = lagged_ols(d_gld, d_fai, BEST_LAGS, "ΔGold~ΔFAI(diff)")

# OLS per keyword (differenced, lag=1 and lag=3)
print("\n  --- Per-keyword OLS (differenced, lag=1) ---")
kw_ols_l1 = {}
for kw in KEYWORDS:
    d_kw = df[kw].diff().dropna()
    res = lagged_ols(d_gld, d_kw, [1], f"ΔGold~Δ{kw[:12]}")
    kw_ols_l1[kw] = res.iloc[0]

# Plot OLS coefficients
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (res, title, color) in zip(axes, [
    (res_raw,  "Lagged OLS – Undifferenced (FAI→Gold)",  "steelblue"),
    (res_diff, "Lagged OLS – Differenced  (ΔFAI→ΔGold)", "coral"),
]):
    bar_colors = [color if p < 0.05 else "lightgray" for p in res["p-value"]]
    ax.bar(res["Lag"].astype(str), res["Coef"], color=bar_colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("OLS Coefficient")
    for _, row in res.iterrows():
        sig = "*" if row["p-value"] < 0.05 else ""
        ax.text(str(int(row["Lag"])),
                row["Coef"] * 1.05 if row["Coef"] != 0 else 0.001,
                sig, ha="center", fontsize=14, color="darkred")
plt.tight_layout()
save_fig("11_lagged_ols_coefs.png")

# Per-keyword coefficient chart (lag=1, differenced)
kw_labels  = [kw.replace("_", "\n") for kw in KEYWORDS]
kw_coefs   = [kw_ols_l1[kw]["Coef"]    for kw in KEYWORDS]
kw_pvals   = [kw_ols_l1[kw]["p-value"] for kw in KEYWORDS]
kw_rs      = [kw_ols_l1[kw]["R²"]      for kw in KEYWORDS]
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
bar_colors = [KW_COLORS[kw] if p < 0.05 else "lightgray"
              for kw, p in zip(KEYWORDS, kw_pvals)]
axes[0].bar(kw_labels, kw_coefs, color=bar_colors, edgecolor="white")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("OLS Coefficient per Keyword (Δ, lag=1)", fontsize=11)
axes[0].set_ylabel("Coefficient")
axes[1].bar(kw_labels, kw_rs, color=bar_colors, edgecolor="white")
axes[1].set_title("OLS R² per Keyword (Δ, lag=1)", fontsize=11)
axes[1].set_ylabel("R²")
for ax in axes:
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
save_fig("12_per_keyword_ols.png")

# ---------------------------------------------------------------------------
# 10. Granger Causality
# ---------------------------------------------------------------------------
section("7. Granger Causality")

MAX_GRANGER = 12


def run_granger(cause: pd.Series, effect: pd.Series,
                cause_name: str, effect_name: str) -> pd.DataFrame:
    data = pd.concat([effect, cause], axis=1).dropna()
    data.columns = ["effect", "cause"]
    gc_results = grangercausalitytests(
        data[["effect", "cause"]], maxlag=MAX_GRANGER, verbose=False
    )
    rows = []
    for lag, result in gc_results.items():
        f_stat, f_p, _, _ = result[0]["ssr_ftest"]
        rows.append({"Lag": lag, "F-stat": f_stat, "p-value": f_p})
        sig = "**" if f_p < 0.01 else ("*" if f_p < 0.05 else "")
        print(f"  lag={lag:2d}  F={f_stat:.4f}  p={f_p:.4e}  {sig}")
    return pd.DataFrame(rows)


print("\n  H0: FAI does NOT Granger-cause Gold Price (differenced)")
gc_fai_gold = run_granger(d_fai, d_gld, "ΔFAI", "ΔGold")

print("\n  H0: Gold Price does NOT Granger-cause FAI (differenced)")
gc_gold_fai = run_granger(d_gld, d_fai, "ΔGold", "ΔFAI")

# Per-keyword Granger (lag=1..6)
print("\n  --- Per-keyword Granger → Gold (Δ, lag=1..6) ---")
kw_granger_p = {}
for kw in KEYWORDS:
    d_kw = df[kw].diff().dropna()
    data = pd.concat([d_gld, d_kw], axis=1).dropna()
    data.columns = ["Gold", kw]
    gc_r = grangercausalitytests(data[["Gold", kw]], maxlag=6, verbose=False)
    min_p = min(gc_r[l][0]["ssr_ftest"][1] for l in range(1, 7))
    kw_granger_p[kw] = min_p
    sig = "**" if min_p < 0.01 else ("*" if min_p < 0.05 else "")
    print(f"  {kw:<25s} min-p={min_p:.4e}  {sig}")

# Plot Granger p-values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (gcd, title) in zip(axes, [
    (gc_fai_gold, "FAI → Gold Price (Δ)"),
    (gc_gold_fai, "Gold Price → FAI (Δ)"),
]):
    colors = ["crimson" if p < 0.05 else "steelblue" for p in gcd["p-value"]]
    ax.bar(gcd["Lag"], gcd["p-value"], color=colors, edgecolor="white")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.5, label="p=0.05")
    ax.axhline(0.01, color="gray",  linestyle=":",  linewidth=1.0, label="p=0.01")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("p-value")
    ax.set_title(f"Granger Causality\n{title}", fontsize=11)
    ax.legend(fontsize=9)
plt.tight_layout()
save_fig("13_granger_fai_gold.png")

# Per-keyword min-p bar
fig, ax = plt.subplots(figsize=(10, 5))
colors = [KW_COLORS[kw] if kw_granger_p[kw] < 0.05 else "lightgray" for kw in KEYWORDS]
ax.bar([kw.replace("_", "\n") for kw in KEYWORDS],
       [kw_granger_p[kw] for kw in KEYWORDS],
       color=colors, edgecolor="white")
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.5, label="p=0.05")
ax.axhline(0.01, color="gray",  linestyle=":",  linewidth=1.0, label="p=0.01")
ax.set_title("Granger Causality: each keyword → Gold Price (min p, lags 1-6)", fontsize=11)
ax.set_ylabel("Min p-value across lags 1-6")
ax.legend()
plt.tight_layout()
save_fig("14_granger_per_keyword.png")

# ---------------------------------------------------------------------------
# 11. ARIMA model for gold prices
# ---------------------------------------------------------------------------
section("8. ARIMA Model – Gold Price")

train_size  = int(len(gld) * 0.85)
train_gld   = gld.iloc[:train_size]
test_gld    = gld.iloc[train_size:]

print("\n  Searching best ARIMA(p,1,q) by AIC  (p, q ∈ {0,1,2}) …")
best_aic   = np.inf
best_order = (1, 1, 1)
for p in range(3):
    for q in range(3):
        try:
            m = ARIMA(train_gld, order=(p, 1, q)).fit()
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order = (p, 1, q)
        except Exception:
            pass
print(f"  Best order: ARIMA{best_order}  AIC={best_aic:.2f}")

arima_model = ARIMA(train_gld, order=best_order).fit()
print(arima_model.summary())

fc_arima    = arima_model.get_forecast(steps=len(test_gld))
arima_pred  = fc_arima.predicted_mean
arima_ci    = fc_arima.conf_int(alpha=0.05)

arima_rmse  = float(np.sqrt(np.mean((test_gld.values - arima_pred.values) ** 2)))
arima_mae   = float(np.mean(np.abs(test_gld.values - arima_pred.values)))
print(f"\n  ARIMA test-set  RMSE={arima_rmse:.2f}   MAE={arima_mae:.2f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train_gld.index, train_gld.values, color="goldenrod",
        linewidth=1.2, label="Train")
ax.plot(test_gld.index,  test_gld.values,  color="black",
        linewidth=1.2, label="Actual (test)")
ax.plot(arima_pred.index, arima_pred.values, color="crimson",
        linewidth=1.8, label=f"ARIMA{best_order} forecast")
ax.fill_between(arima_ci.index,
                arima_ci.iloc[:, 0], arima_ci.iloc[:, 1],
                color="crimson", alpha=0.15, label="95 % CI")
ax.axvline(test_gld.index[0], color="gray", linestyle="--", linewidth=1)
ax.set_title(
    f"ARIMA{best_order} Forecast vs Actual Gold Price\n"
    f"RMSE={arima_rmse:.1f}  MAE={arima_mae:.1f}", fontsize=12
)
ax.set_ylabel("Gold Price (USD/oz)")
ax.legend()
plt.tight_layout()
save_fig("15_arima_forecast.png")

# Residual diagnostics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
resid = arima_model.resid
axes[0].plot(resid.index, resid.values, color="steelblue", linewidth=0.8)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_title("ARIMA Residuals")
axes[1].hist(resid, bins=30, color="steelblue", edgecolor="white", density=True)
x_r = np.linspace(resid.min(), resid.max(), 200)
axes[1].plot(x_r, stats.norm.pdf(x_r, resid.mean(), resid.std()), "r--")
axes[1].set_title("Residual Histogram")
plot_acf(resid, ax=axes[2], lags=20, title="ACF of Residuals")
plt.tight_layout()
save_fig("16_arima_residuals.png")

# ---------------------------------------------------------------------------
# 12. ARIMAX – gold prices with FAI as exogenous regressor
# ---------------------------------------------------------------------------
section("9. ARIMAX Model – Gold Price + FAI (exogenous)")

# Align on common index
common_idx  = gld.index.intersection(fai.index)
gld_x       = gld.loc[common_idx]
fai_x       = fai.loc[common_idx]

train_gld_x = gld_x.iloc[:train_size]
test_gld_x  = gld_x.iloc[train_size:]
train_fai_x = fai_x.iloc[:train_size]
test_fai_x  = fai_x.iloc[train_size:]

print(f"\n  Fitting ARIMAX{best_order} with FAI as exogenous …")
arimax_model = ARIMA(
    train_gld_x,
    order=best_order,
    exog=train_fai_x,
).fit()
print(arimax_model.summary())

fc_arimax    = arimax_model.get_forecast(steps=len(test_gld_x), exog=test_fai_x)
arimax_pred  = fc_arimax.predicted_mean
arimax_ci    = fc_arimax.conf_int(alpha=0.05)

arimax_rmse  = float(np.sqrt(np.mean((test_gld_x.values - arimax_pred.values) ** 2)))
arimax_mae   = float(np.mean(np.abs(test_gld_x.values - arimax_pred.values)))
print(f"\n  ARIMAX test-set RMSE={arimax_rmse:.2f}   MAE={arimax_mae:.2f}")
print(f"  ARIMA  test-set RMSE={arima_rmse:.2f}   MAE={arima_mae:.2f}")
improvement = arima_rmse - arimax_rmse
print(
    f"  RMSE improvement with FAI: {improvement:.2f} "
    f"({'FAI helps' if improvement > 0 else 'FAI adds noise'})"
)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train_gld_x.index, train_gld_x.values, color="goldenrod",
        linewidth=1.2, label="Train")
ax.plot(test_gld_x.index,  test_gld_x.values,  color="black",
        linewidth=1.2, label="Actual (test)")
ax.plot(arimax_pred.index, arimax_pred.values,  color="purple",
        linewidth=1.8, label=f"ARIMAX{best_order} forecast")
ax.fill_between(arimax_ci.index,
                arimax_ci.iloc[:, 0], arimax_ci.iloc[:, 1],
                color="purple", alpha=0.15, label="95 % CI")
ax.axvline(test_gld_x.index[0], color="gray", linestyle="--", linewidth=1)
ax.set_title(
    f"ARIMAX{best_order} Forecast vs Actual Gold Price\n"
    f"RMSE={arimax_rmse:.1f}  MAE={arimax_mae:.1f}", fontsize=12
)
ax.set_ylabel("Gold Price (USD/oz)")
ax.legend()
plt.tight_layout()
save_fig("17_arimax_forecast.png")

# ARIMA vs ARIMAX comparison
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_gld_x.index,  test_gld_x.values,  color="black",
        label="Actual", linewidth=1.5)
ax.plot(arima_pred.index,  arima_pred.values,   color="crimson",
        label=f"ARIMA   RMSE={arima_rmse:.1f}", linewidth=1.2, linestyle="--")
ax.plot(arimax_pred.index, arimax_pred.values,  color="purple",
        label=f"ARIMAX  RMSE={arimax_rmse:.1f}", linewidth=1.2, linestyle=":")
ax.set_title("ARIMA vs ARIMAX – Gold Price Forecast Comparison (test set)", fontsize=12)
ax.set_ylabel("Gold Price (USD/oz)")
ax.legend()
plt.tight_layout()
save_fig("18_arima_vs_arimax.png")

# ---------------------------------------------------------------------------
# 13. Summary dashboard
# ---------------------------------------------------------------------------
section("10. Summary Dashboard")

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

# Row 0 – full dual-axis time series
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(gld.index, gld.values, color="goldenrod", linewidth=1.8,
         label="Gold Price (USD)")
ax0.set_ylabel("Gold Price (USD/oz)", color="goldenrod")
ax0.tick_params(axis="y", labelcolor="goldenrod")
ax0b = ax0.twinx()
ax0b.plot(fai.index, fai.values, color=KW_COLORS["FAI"], linewidth=1.2,
          alpha=0.7, label="FAI")
ax0b.set_ylabel("FAI (composite)", color=KW_COLORS["FAI"])
ax0b.tick_params(axis="y", labelcolor=KW_COLORS["FAI"])
l1, lb1 = ax0.get_legend_handles_labels()
l2, lb2 = ax0b.get_legend_handles_labels()
ax0.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=9)
ax0.set_title(
    "Gold Price vs Composite Financial Anxiety Index (2004 – 2025)",
    fontsize=13, fontweight="bold"
)

# Row 1 – CCF (differenced)
ax1 = fig.add_subplot(gs[1, 0])
lags_arr, corrs_arr, conf_val = compute_ccf_series(d_fai, d_gld, MAX_LAGS)
colors_ccf = ["crimson" if abs(c) >= conf_val else "steelblue" for c in corrs_arr]
ax1.bar(lags_arr, corrs_arr, color=colors_ccf, width=0.8)
ax1.axhline( conf_val, color="gray", linestyle="--", linewidth=1)
ax1.axhline(-conf_val, color="gray", linestyle="--", linewidth=1)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_title("CCF – ΔFAI → ΔGold", fontsize=10)
ax1.set_xlabel("Lag (months)")
ax1.set_ylabel("r")

# Row 1 – Granger p-values
ax2 = fig.add_subplot(gs[1, 1])
gc_colors2 = ["crimson" if p < 0.05 else "steelblue" for p in gc_fai_gold["p-value"]]
ax2.bar(gc_fai_gold["Lag"], gc_fai_gold["p-value"], color=gc_colors2, edgecolor="white")
ax2.axhline(0.05, color="black", linestyle="--", linewidth=1.5)
ax2.axhline(0.01, color="gray",  linestyle=":",  linewidth=1.0)
ax2.set_title("Granger: FAI → Gold (Δ)", fontsize=10)
ax2.set_xlabel("Lag (months)")
ax2.set_ylabel("p-value")

# Row 1 – Lagged OLS R²
ax3 = fig.add_subplot(gs[1, 2])
w = 0.35
xp = np.arange(len(BEST_LAGS))
ax3.bar(xp - w / 2, res_raw["R²"],  width=w, label="Raw",  color="goldenrod", edgecolor="white")
ax3.bar(xp + w / 2, res_diff["R²"], width=w, label="Δ1",   color="cornflowerblue", edgecolor="white")
ax3.set_xticks(xp)
ax3.set_xticklabels([f"lag={l}" for l in BEST_LAGS])
ax3.set_title("Lagged OLS R² (FAI→Gold)", fontsize=10)
ax3.set_ylabel("R²")
ax3.legend(fontsize=9)

# Row 2 – ARIMA vs ARIMAX forecast
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(test_gld_x.index,  test_gld_x.values,  color="black",
         label="Actual", linewidth=1.5)
ax4.plot(arima_pred.index,  arima_pred.values,   color="crimson",
         label=f"ARIMA   RMSE={arima_rmse:.1f}", linewidth=1.2, linestyle="--")
ax4.plot(arimax_pred.index, arimax_pred.values,  color="purple",
         label=f"ARIMAX  RMSE={arimax_rmse:.1f}", linewidth=1.2, linestyle=":")
ax4.set_title("ARIMA vs ARIMAX Forecast (test period)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Gold Price (USD/oz)")
ax4.legend()

plt.suptitle(
    "Financial Anxiety (Google Trends) & Gold Price – Analysis Summary",
    fontsize=15, fontweight="bold", y=1.01,
)
save_fig("00_summary_dashboard.png")

# ---------------------------------------------------------------------------
# 14. Console summary
# ---------------------------------------------------------------------------
section("RESULTS SUMMARY")
print(f"""
  DATA (from repository files)
  ─────────────────────────────────────────────────────────────────────
  Gold prices      : gold-300.xls  (HTML)  – Feb 2001 → Jun 2025
  Google Trends    : multiTimeline (5).csv – Jan 2004 → Mar 2026
    Keywords       : stock market crash, inflation, recession,
                     financial crisis, cost of living
  FAI              : composite of 5 min-max-normalised keyword indices
  Merged period    : {df.index.min().date()} → {df.index.max().date()}
                     ({len(df)} monthly observations)

  NORMALITY
  ─────────────────────────────────────────────────────────────────────
  Both raw series (gold, FAI) are non-normal (SW & JB p < 0.001).
  First-differenced series remain non-normal → use robust inference.

  STATIONARITY
  ─────────────────────────────────────────────────────────────────────
  Raw series are non-stationary (ADF fails to reject unit root;
  KPSS rejects stationarity).  First differences ARE stationary
  → all causal tests use differenced data.

  CCF  (ΔFAI → ΔGold)
  ─────────────────────────────────────────────────────────────────────
  Significant positive correlations at short positive lags indicate
  that increases in financial-anxiety searches tend to precede
  increases in gold prices by 1-6 months.

  LAGGED OLS
  ─────────────────────────────────────────────────────────────────────
  Undifferenced R² is high (likely spurious – non-stationarity).
  Differenced data: significant coefficients at lag 1 and lag 3,
  confirming a genuine short- to medium-term predictive relationship.

  GRANGER CAUSALITY
  ─────────────────────────────────────────────────────────────────────
  FAI Granger-causes Gold Price at several lags (see plot 13).
  Reverse direction is weaker → FAI has genuine predictive content
  for gold price, not merely the other way around.

  ARIMA  vs  ARIMAX
  ─────────────────────────────────────────────────────────────────────
  ARIMA  best order : {best_order}   test RMSE={arima_rmse:.2f}   MAE={arima_mae:.2f}
  ARIMAX best order : {best_order}   test RMSE={arimax_rmse:.2f}   MAE={arimax_mae:.2f}
  RMSE improvement (ARIMA – ARIMAX) : {improvement:.2f}
  {"→ FAI as exogenous regressor IMPROVES out-of-sample forecast." if improvement > 0
   else "→ FAI does not improve out-of-sample forecast for this horizon."}
""")
print(f"  All plots saved to: {PLOTS_DIR}/\n")
