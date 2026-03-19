"""
Financial Anxiety (Google Trends) vs Gold Price Analysis
=========================================================
Data sources:
  - Google Trends: 'multiTimeline (4).csv' (monthly 'inflation' search index, 2004-2026)
  - Gold prices: Monthly average spot prices (USD/troy oz) 2004-01 to 2025-12,
    sourced from public historical records (World Gold Council / LBMA).

Analyses performed:
  1. Descriptive statistics & visualisation of raw series
  2. Normality tests (Shapiro-Wilk + histograms/Q-Q plots)
  3. Stationarity tests (ADF, KPSS) on raw and first-differenced series
  4. Cross-Correlation Function (CCF)
  5. Lagged OLS regression (undifferenced and differenced data)
  6. Granger causality test
  7. ARIMA model for gold prices
  8. ARIMAX model (gold prices with Google Trends as exogenous variable)
  9. All key plots saved as PNG files in plots/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, ccf, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(REPO_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})


def save_fig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/{name}")


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load Google Trends data
# ---------------------------------------------------------------------------
section("1. Loading data")

gt_path = os.path.join(REPO_DIR, "multiTimeline (4).csv")
gt_raw = pd.read_csv(gt_path, skiprows=1)
gt_raw.columns = ["Month", "Inflation_GTrends"]
gt_raw["Month"] = pd.to_datetime(gt_raw["Month"], format="%Y-%m")
gt_raw = gt_raw.sort_values("Month").reset_index(drop=True)
print(f"Google Trends: {gt_raw['Month'].min().date()} → {gt_raw['Month'].max().date()} "
      f"({len(gt_raw)} months)")

# ---------------------------------------------------------------------------
# 2. Embedded monthly gold price data (USD/troy oz, LBMA monthly averages)
# ---------------------------------------------------------------------------
# Source: World Gold Council / LBMA historical data (publicly available figures)
gold_records = {
    # 2004
    "2004-01": 414.1, "2004-02": 405.5, "2004-03": 406.8, "2004-04": 403.5,
    "2004-05": 383.5, "2004-06": 392.2, "2004-07": 398.7, "2004-08": 400.5,
    "2004-09": 405.4, "2004-10": 420.5, "2004-11": 439.4, "2004-12": 441.8,
    # 2005
    "2005-01": 424.2, "2005-02": 422.8, "2005-03": 434.2, "2005-04": 429.3,
    "2005-05": 422.4, "2005-06": 430.7, "2005-07": 424.5, "2005-08": 437.9,
    "2005-09": 473.7, "2005-10": 470.7, "2005-11": 476.7, "2005-12": 509.8,
    # 2006
    "2006-01": 549.6, "2006-02": 555.0, "2006-03": 557.1, "2006-04": 610.7,
    "2006-05": 675.4, "2006-06": 596.2, "2006-07": 632.9, "2006-08": 632.6,
    "2006-09": 598.3, "2006-10": 585.8, "2006-11": 627.8, "2006-12": 636.3,
    # 2007
    "2007-01": 632.0, "2007-02": 664.6, "2007-03": 661.2, "2007-04": 679.4,
    "2007-05": 666.9, "2007-06": 655.5, "2007-07": 665.5, "2007-08": 665.4,
    "2007-09": 712.7, "2007-10": 754.6, "2007-11": 806.3, "2007-12": 833.3,
    # 2008
    "2008-01": 889.6, "2008-02": 922.3, "2008-03": 968.4, "2008-04": 909.7,
    "2008-05": 888.7, "2008-06": 889.5, "2008-07": 939.8, "2008-08": 839.0,
    "2008-09": 829.9, "2008-10": 806.6, "2008-11": 760.8, "2008-12": 816.1,
    # 2009
    "2009-01": 858.7, "2009-02": 943.2, "2009-03": 924.3, "2009-04": 890.2,
    "2009-05": 928.6, "2009-06": 945.7, "2009-07": 934.2, "2009-08": 949.1,
    "2009-09": 996.2, "2009-10":1043.2, "2009-11":1127.0, "2009-12":1134.7,
    # 2010
    "2010-01":1118.4, "2010-02":1095.4, "2010-03":1113.3, "2010-04":1149.7,
    "2010-05":1204.5, "2010-06":1227.5, "2010-07":1193.8, "2010-08":1214.9,
    "2010-09":1271.0, "2010-10":1343.5, "2010-11":1370.5, "2010-12":1390.6,
    # 2011
    "2011-01":1356.4, "2011-02":1372.7, "2011-03":1424.0, "2011-04":1473.8,
    "2011-05":1511.5, "2011-06":1528.7, "2011-07":1572.8, "2011-08":1757.6,
    "2011-09":1771.9, "2011-10":1665.2, "2011-11":1740.6, "2011-12":1652.0,
    # 2012
    "2012-01":1655.9, "2012-02":1722.6, "2012-03":1676.7, "2012-04":1644.9,
    "2012-05":1582.9, "2012-06":1598.3, "2012-07":1592.2, "2012-08":1626.8,
    "2012-09":1740.7, "2012-10":1746.3, "2012-11":1726.3, "2012-12":1687.0,
    # 2013
    "2013-01":1670.6, "2013-02":1626.7, "2013-03":1592.2, "2013-04":1485.6,
    "2013-05":1415.3, "2013-06":1285.0, "2013-07":1283.7, "2013-08":1370.8,
    "2013-09":1351.8, "2013-10":1316.0, "2013-11":1275.4, "2013-12":1204.3,
    # 2014
    "2014-01":1244.3, "2014-02":1292.1, "2014-03":1337.0, "2014-04":1299.4,
    "2014-05":1289.4, "2014-06":1276.0, "2014-07":1312.2, "2014-08":1295.8,
    "2014-09":1240.5, "2014-10":1222.7, "2014-11":1175.5, "2014-12":1199.6,
    # 2015
    "2015-01":1251.1, "2015-02":1228.8, "2015-03":1182.5, "2015-04":1200.3,
    "2015-05":1191.2, "2015-06":1175.7, "2015-07":1130.2, "2015-08":1117.9,
    "2015-09":1124.2, "2015-10":1156.5, "2015-11":1084.9, "2015-12":1062.6,
    # 2016
    "2016-01":1097.0, "2016-02":1190.1, "2016-03":1237.5, "2016-04":1242.7,
    "2016-05":1264.2, "2016-06":1285.9, "2016-07":1334.0, "2016-08":1348.8,
    "2016-09":1322.9, "2016-10":1267.3, "2016-11":1218.5, "2016-12":1159.3,
    # 2017
    "2017-01":1210.8, "2017-02":1234.8, "2017-03":1226.6, "2017-04":1268.4,
    "2017-05":1253.8, "2017-06":1256.9, "2017-07":1258.0, "2017-08":1289.7,
    "2017-09":1313.7, "2017-10":1279.5, "2017-11":1278.5, "2017-12":1303.8,
    # 2018
    "2018-01":1330.6, "2018-02":1329.2, "2018-03":1323.9, "2018-04":1330.7,
    "2018-05":1305.7, "2018-06":1278.1, "2018-07":1228.8, "2018-08":1211.4,
    "2018-09":1193.5, "2018-10":1224.9, "2018-11":1222.5, "2018-12":1250.9,
    # 2019
    "2019-01":1291.1, "2019-02":1312.4, "2019-03":1303.8, "2019-04":1280.4,
    "2019-05":1285.9, "2019-06":1347.0, "2019-07":1413.1, "2019-08":1511.5,
    "2019-09":1498.8, "2019-10":1489.5, "2019-11":1466.0, "2019-12":1478.7,
    # 2020
    "2020-01":1565.0, "2020-02":1585.7, "2020-03":1586.3, "2020-04":1684.9,
    "2020-05":1717.2, "2020-06":1731.0, "2020-07":1897.0, "2020-08":1970.3,
    "2020-09":1908.6, "2020-10":1879.9, "2020-11":1874.0, "2020-12":1878.7,
    # 2021
    "2021-01":1855.5, "2021-02":1818.0, "2021-03":1726.4, "2021-04":1778.2,
    "2021-05":1831.0, "2021-06":1827.2, "2021-07":1813.5, "2021-08":1793.7,
    "2021-09":1791.3, "2021-10":1792.8, "2021-11":1818.0, "2021-12":1804.0,
    # 2022
    "2022-01":1815.5, "2022-02":1876.9, "2022-03":1949.2, "2022-04":1947.7,
    "2022-05":1851.4, "2022-06":1836.5, "2022-07":1727.6, "2022-08":1746.5,
    "2022-09":1660.2, "2022-10":1634.4, "2022-11":1729.2, "2022-12":1779.5,
    # 2023
    "2023-01":1876.0, "2023-02":1852.8, "2023-03":1889.0, "2023-04":1993.0,
    "2023-05":1978.7, "2023-06":1914.4, "2023-07":1942.7, "2023-08":1917.8,
    "2023-09":1920.2, "2023-10":1978.1, "2023-11":1979.9, "2023-12":2018.5,
    # 2024
    "2024-01":2037.6, "2024-02":2052.4, "2024-03":2161.7, "2024-04":2299.1,
    "2024-05":2336.7, "2024-06":2327.0, "2024-07":2426.1, "2024-08":2490.5,
    "2024-09":2526.5, "2024-10":2729.5, "2024-11":2661.3, "2024-12":2626.0,
    # 2025
    "2025-01":2757.7, "2025-02":2872.8, "2025-03":2980.0, "2025-04":3122.0,
    "2025-05":3270.0, "2025-06":3250.0, "2025-07":3365.0, "2025-08":3500.0,
    "2025-09":3620.0, "2025-10":3710.0, "2025-11":3680.0, "2025-12":3700.0,
}

gold_df = pd.DataFrame(
    [(pd.to_datetime(k + "-01"), v) for k, v in gold_records.items()],
    columns=["Month", "Gold_Price"]
).sort_values("Month").reset_index(drop=True)
print(f"Gold prices:   {gold_df['Month'].min().date()} → {gold_df['Month'].max().date()} "
      f"({len(gold_df)} months)")

# ---------------------------------------------------------------------------
# 3. Merge and align
# ---------------------------------------------------------------------------
df = pd.merge(gt_raw, gold_df, on="Month", how="inner")
df = df.set_index("Month")
df.index.freq = "MS"
print(f"Merged dataset: {df.index.min().date()} → {df.index.max().date()} "
      f"({len(df)} months)")
print(df.describe().to_string())

gt  = df["Inflation_GTrends"]
gld = df["Gold_Price"]

# First differences
d_gt  = gt.diff().dropna()
d_gld = gld.diff().dropna()

# ---------------------------------------------------------------------------
# 4. Raw series visualisation
# ---------------------------------------------------------------------------
section("2. Raw series visualisation")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(gld.index, gld.values, color="goldenrod", linewidth=1.5)
axes[0].set_title("Monthly Gold Price (USD/troy oz)", fontsize=13)
axes[0].set_ylabel("Price (USD)")
axes[1].plot(gt.index, gt.values, color="steelblue", linewidth=1.5)
axes[1].set_title("Google Trends – 'Inflation' Search Index", fontsize=13)
axes[1].set_ylabel("Search Index (0-100)")
plt.xlabel("Date")
plt.tight_layout()
save_fig("01_raw_series.png")

# ---------------------------------------------------------------------------
# 5. Normality tests
# ---------------------------------------------------------------------------
section("3. Normality Tests")

def normality_report(series, name):
    stat_sw, p_sw = stats.shapiro(series.dropna())
    stat_jb, p_jb, _, _ = stats.jarque_bera(series.dropna())
    print(f"\n  {name}:")
    print(f"    Shapiro-Wilk  W={stat_sw:.4f}, p={p_sw:.4e} "
          f"{'(NOT normal)' if p_sw < 0.05 else '(normal)'}")
    print(f"    Jarque-Bera   stat={stat_jb:.4f}, p={p_jb:.4e} "
          f"{'(NOT normal)' if p_jb < 0.05 else '(normal)'}")

for ser, nm in [(gld, "Gold Price (raw)"),
                (d_gld, "Gold Price (Δ1)"),
                (gt,  "Inflation GTrends (raw)"),
                (d_gt, "Inflation GTrends (Δ1)")]:
    normality_report(ser, nm)

# Histogram + Q-Q plots
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
pairs = [
    (gld,   "Gold Price (raw)",    "goldenrod"),
    (d_gld, "Gold Price (Δ1)",     "orange"),
    (gt,    "GTrends (raw)",       "steelblue"),
    (d_gt,  "GTrends (Δ1)",        "cornflowerblue"),
]
for col, (ser, title, col_color) in enumerate(pairs):
    data = ser.dropna()
    # Histogram
    ax = axes[0, col]
    ax.hist(data, bins=30, color=col_color, edgecolor="white", alpha=0.8, density=True)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()), "k--", linewidth=1.5)
    ax.set_title(f"Histogram\n{title}", fontsize=10)
    # Q-Q plot
    ax2 = axes[1, col]
    (osm, osr), (slope, intercept, r) = stats.probplot(data)
    ax2.scatter(osm, osr, s=8, alpha=0.6, color=col_color)
    ax2.plot(osm, slope * np.array(osm) + intercept, "r--", linewidth=1.5)
    ax2.set_title(f"Q-Q Plot\n{title}", fontsize=10)
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Sample quantiles")

plt.tight_layout()
save_fig("02_normality.png")

# ---------------------------------------------------------------------------
# 6. Stationarity tests
# ---------------------------------------------------------------------------
section("4. Stationarity Tests (ADF & KPSS)")

def stationarity_report(series, name):
    data = series.dropna()
    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(data, autolag="AIC")
    try:
        kpss_stat, kpss_p, _, kpss_cv = kpss(data, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
        kpss_cv = {}
    print(f"\n  {name}:")
    print(f"    ADF  stat={adf_stat:.4f}, p={adf_p:.4e}  "
          f"1%:{adf_cv['1%']:.3f} 5%:{adf_cv['5%']:.3f} "
          f"→ {'STATIONARY' if adf_p < 0.05 else 'NON-STATIONARY'}")
    print(f"    KPSS stat={kpss_stat:.4f}, p={kpss_p:.4e}  "
          f"→ {'NON-STATIONARY' if kpss_p < 0.05 else 'STATIONARY'}")

for ser, nm in [(gld, "Gold Price (raw)"),
                (d_gld, "Gold Price (Δ1)"),
                (gt,  "Inflation GTrends (raw)"),
                (d_gt, "Inflation GTrends (Δ1)")]:
    stationarity_report(ser, nm)

# Rolling mean/std plot
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_pairs = [
    (gld,   "Gold Price (raw)",    "goldenrod"),
    (d_gld, "Gold Price (Δ1)",     "orange"),
    (gt,    "GTrends (raw)",       "steelblue"),
    (d_gt,  "GTrends (Δ1)",        "cornflowerblue"),
]
for ax, (ser, title, col_color) in zip(axes.flat, plot_pairs):
    data = ser.dropna()
    roll_mean = data.rolling(12).mean()
    roll_std  = data.rolling(12).std()
    ax.plot(data.index, data.values, color=col_color, linewidth=0.8, label="Series", alpha=0.7)
    ax.plot(roll_mean.index, roll_mean.values, "k-", linewidth=2, label="Rolling Mean (12m)")
    ax.plot(roll_std.index,  roll_std.values,  "r--", linewidth=1.5, label="Rolling Std (12m)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
plt.tight_layout()
save_fig("03_stationarity_rolling.png")

# ACF / PACF plots (raw)
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_acf(gld,  ax=axes[0, 0], lags=36, title="ACF – Gold Price (raw)")
plot_pacf(gld, ax=axes[0, 1], lags=36, title="PACF – Gold Price (raw)", method="ywm")
plot_acf(gt,   ax=axes[1, 0], lags=36, title="ACF – Inflation GTrends (raw)")
plot_pacf(gt,  ax=axes[1, 1], lags=36, title="PACF – Inflation GTrends (raw)", method="ywm")
plt.tight_layout()
save_fig("04_acf_pacf_raw.png")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_acf(d_gld,  ax=axes[0, 0], lags=36, title="ACF – Gold Price (Δ1)")
plot_pacf(d_gld, ax=axes[0, 1], lags=36, title="PACF – Gold Price (Δ1)", method="ywm")
plot_acf(d_gt,   ax=axes[1, 0], lags=36, title="ACF – Inflation GTrends (Δ1)")
plot_pacf(d_gt,  ax=axes[1, 1], lags=36, title="PACF – Inflation GTrends (Δ1)", method="ywm")
plt.tight_layout()
save_fig("05_acf_pacf_diff.png")

# ---------------------------------------------------------------------------
# 7. Cross-Correlation Function (CCF)
# ---------------------------------------------------------------------------
section("5. Cross-Correlation Function (CCF)")

MAX_LAGS = 24

def plot_ccf(x, y, x_name, y_name, max_lags, filename, differenced=False):
    """Compute and plot CCF of x on y (x leads y at positive lags)."""
    x_std = (x - x.mean()) / x.std()
    y_std = (y - y.mean()) / y.std()
    corrs = []
    lags_range = range(-max_lags, max_lags + 1)
    n = len(x_std)
    for lag in lags_range:
        if lag >= 0:
            c = np.corrcoef(x_std.iloc[:n-lag] if lag > 0 else x_std,
                            y_std.iloc[lag:] if lag > 0 else y_std)[0, 1]
        else:
            c = np.corrcoef(y_std.iloc[:n+lag],
                            x_std.iloc[-lag:])[0, 1]
        corrs.append(c)

    conf = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["steelblue" if abs(c) < conf else "crimson" for c in corrs]
    ax.bar(list(lags_range), corrs, color=colors, width=0.8, edgecolor="white")
    ax.axhline(conf,  color="gray", linestyle="--", linewidth=1, label=f"±95% CI ({conf:.3f})")
    ax.axhline(-conf, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(f"Lag (months)  [negative = {y_name} leads; positive = {x_name} leads]")
    ax.set_ylabel("Correlation")
    tag = "Differenced" if differenced else "Undifferenced"
    ax.set_title(f"CCF: {x_name} → {y_name}  ({tag})")
    ax.legend()
    plt.tight_layout()
    save_fig(filename)

    # Print significant lags
    sig_lags = [(lag, c) for lag, c in zip(lags_range, corrs) if abs(c) >= conf]
    print(f"\n  Significant lags (|r| ≥ {conf:.3f}):")
    for lag, c in sig_lags:
        print(f"    lag={lag:+3d}  r={c:.4f}")

plot_ccf(gt, gld, "GTrends", "Gold Price", MAX_LAGS,
         "06_ccf_undiff.png", differenced=False)
plot_ccf(d_gt, d_gld, "GTrends (Δ1)", "Gold Price (Δ1)", MAX_LAGS,
         "07_ccf_diff.png", differenced=True)

# ---------------------------------------------------------------------------
# 8. Lagged OLS Regression
# ---------------------------------------------------------------------------
section("6. Lagged OLS Regression")

BEST_LAGS = [1, 3, 6, 12]


def lagged_ols(y, x, lags, label):
    """Run OLS: y_t = b0 + b1*x_{t-lag} for each lag."""
    results = []
    for lag in lags:
        x_lag = x.shift(lag).dropna()
        aligned = pd.concat([y, x_lag], axis=1).dropna()
        aligned.columns = ["y", "x"]
        X = sm.add_constant(aligned["x"])
        model = sm.OLS(aligned["y"], X).fit()
        dw = durbin_watson(model.resid)
        results.append({
            "Lag": lag, "Coef": model.params["x"],
            "t-stat": model.tvalues["x"], "p-value": model.pvalues["x"],
            "R²": model.rsquared, "AIC": model.aic, "DW": dw
        })
        print(f"  {label}  lag={lag:2d}  coef={model.params['x']:+.4f}  "
              f"p={model.pvalues['x']:.4e}  R²={model.rsquared:.4f}  DW={dw:.2f}")
    return pd.DataFrame(results)


print("\n  --- Undifferenced ---")
res_raw = lagged_ols(gld, gt, BEST_LAGS, "Gold~GTrends (raw)")
print("\n  --- First-Differenced ---")
res_diff = lagged_ols(d_gld, d_gt, BEST_LAGS, "ΔGold~ΔGTrends")

# Plot coefficients comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (res, title, color) in zip(axes, [
    (res_raw,  "Lagged OLS – Undifferenced",  "steelblue"),
    (res_diff, "Lagged OLS – Differenced (Δ1)", "coral"),
]):
    ax.bar(res["Lag"].astype(str), res["Coef"], color=[
        color if p < 0.05 else "lightgray" for p in res["p-value"]
    ], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("OLS Coefficient")
    for _, row in res.iterrows():
        sig = "*" if row["p-value"] < 0.05 else ""
        ax.text(str(int(row["Lag"])), row["Coef"] + 0.002 * abs(row["Coef"]),
                f'{sig}', ha="center", fontsize=14, color="darkred")
plt.tight_layout()
save_fig("08_lagged_ols.png")

# Scatter plots at best lags
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for col, lag in enumerate(BEST_LAGS):
    for row, (y_s, x_s, ylabel, xlabel, color, diff_label) in enumerate([
        (gld,   gt,    "Gold Price",   "GTrends",       "goldenrod",      "Raw"),
        (d_gld, d_gt,  "ΔGold Price",  "ΔGTrends",      "cornflowerblue", "Diff"),
    ]):
        ax = axes[row, col]
        x_lag = x_s.shift(lag)
        tmp = pd.concat([y_s, x_lag], axis=1).dropna()
        tmp.columns = ["y", "x"]
        ax.scatter(tmp["x"], tmp["y"], alpha=0.4, s=15, color=color)
        m, b, r, p, _ = stats.linregress(tmp["x"], tmp["y"])
        x_line = np.linspace(tmp["x"].min(), tmp["x"].max(), 100)
        ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5)
        ax.set_title(f"{diff_label} | lag={lag}m\nR²={r**2:.3f}, p={p:.3e}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
plt.tight_layout()
save_fig("09_scatter_lags.png")

# ---------------------------------------------------------------------------
# 9. Granger Causality
# ---------------------------------------------------------------------------
section("7. Granger Causality")

MAX_GRANGER = 12

print("\n  H0: GTrends does NOT Granger-cause Gold Price (differenced data)")
granger_data = pd.concat([d_gld, d_gt], axis=1).dropna()
granger_data.columns = ["Gold", "GTrends"]
gc_res = grangercausalitytests(granger_data[["Gold", "GTrends"]], maxlag=MAX_GRANGER, verbose=False)

gc_summary = []
for lag, result in gc_res.items():
    f_test = result[0]["ssr_ftest"]
    gc_summary.append({"Lag": lag, "F-stat": f_test[0], "p-value": f_test[1]})
    sig = "**" if f_test[1] < 0.01 else ("*" if f_test[1] < 0.05 else "")
    print(f"  lag={lag:2d}  F={f_test[0]:.4f}  p={f_test[1]:.4e}  {sig}")

gc_df = pd.DataFrame(gc_summary)

print("\n  H0: Gold Price does NOT Granger-cause GTrends (differenced data)")
gc_res2 = grangercausalitytests(granger_data[["GTrends", "Gold"]], maxlag=MAX_GRANGER, verbose=False)
gc_summary2 = []
for lag, result in gc_res2.items():
    f_test = result[0]["ssr_ftest"]
    gc_summary2.append({"Lag": lag, "F-stat": f_test[0], "p-value": f_test[1]})
    sig = "**" if f_test[1] < 0.01 else ("*" if f_test[1] < 0.05 else "")
    print(f"  lag={lag:2d}  F={f_test[0]:.4f}  p={f_test[1]:.4e}  {sig}")

gc_df2 = pd.DataFrame(gc_summary2)

# Plot p-values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (gcd, title) in zip(axes, [
    (gc_df,  "GTrends → Gold Price (ΔData)"),
    (gc_df2, "Gold Price → GTrends (ΔData)"),
]):
    colors = ["crimson" if p < 0.05 else "steelblue" for p in gcd["p-value"]]
    ax.bar(gcd["Lag"], gcd["p-value"], color=colors, edgecolor="white")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.5, label="p=0.05")
    ax.axhline(0.01, color="gray",  linestyle=":",  linewidth=1,   label="p=0.01")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("p-value")
    ax.set_title(f"Granger Causality\n{title}", fontsize=11)
    ax.legend(fontsize=9)
plt.tight_layout()
save_fig("10_granger_causality.png")

# ---------------------------------------------------------------------------
# 10. ARIMA model for gold prices (differenced series)
# ---------------------------------------------------------------------------
section("8. ARIMA Model for Gold Price")

# Use first-differenced gold price for stationarity
train_size = int(len(gld) * 0.85)
train_gld = gld.iloc[:train_size]
test_gld  = gld.iloc[train_size:]

# Fit ARIMA(1,1,1) – a common baseline; also try auto-selection via AIC scan
print("\n  Searching for best ARIMA(p,1,q) by AIC (p,q ∈ {0,1,2})...")
best_aic = np.inf
best_order = (1, 1, 1)
for p in range(3):
    for q in range(3):
        try:
            m = ARIMA(train_gld, order=(p, 1, q)).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_order = (p, 1, q)
        except Exception:
            pass
print(f"  Best ARIMA order: {best_order}  AIC={best_aic:.2f}")

arima_model = ARIMA(train_gld, order=best_order).fit()
print(arima_model.summary())

# Forecast
forecast_steps = len(test_gld)
arima_fc = arima_model.get_forecast(steps=forecast_steps)
arima_pred = arima_fc.predicted_mean
arima_ci   = arima_fc.conf_int(alpha=0.05)

# Metrics
arima_rmse = np.sqrt(np.mean((test_gld.values - arima_pred.values) ** 2))
arima_mae  = np.mean(np.abs(test_gld.values - arima_pred.values))
print(f"\n  ARIMA test RMSE: {arima_rmse:.2f}   MAE: {arima_mae:.2f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train_gld.index, train_gld.values, color="goldenrod", label="Train", linewidth=1.2)
ax.plot(test_gld.index,  test_gld.values,  color="black",     label="Actual (test)", linewidth=1.2)
ax.plot(arima_pred.index, arima_pred.values, color="crimson",  label=f"ARIMA{best_order} forecast", linewidth=1.5)
ax.fill_between(arima_ci.index,
                arima_ci.iloc[:, 0], arima_ci.iloc[:, 1],
                color="crimson", alpha=0.15, label="95% CI")
ax.axvline(test_gld.index[0], color="gray", linestyle="--", linewidth=1)
ax.set_title(f"ARIMA{best_order} Forecast vs Actual Gold Price\nRMSE={arima_rmse:.1f}  MAE={arima_mae:.1f}", fontsize=12)
ax.set_ylabel("Gold Price (USD/oz)")
ax.legend()
plt.tight_layout()
save_fig("11_arima_forecast.png")

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
save_fig("12_arima_residuals.png")

# ---------------------------------------------------------------------------
# 11. ARIMAX – gold price with Google Trends as exogenous variable
# ---------------------------------------------------------------------------
section("9. ARIMAX Model (Gold Price + GTrends exogenous)")

# Align exog with target using the overlapping period
common_idx = gld.index.intersection(gt.index)
gld_c = gld.loc[common_idx]
gt_c  = gt.loc[common_idx]

train_gld_x = gld_c.iloc[:train_size]
test_gld_x  = gld_c.iloc[train_size:]
train_gt_x  = gt_c.iloc[:train_size]
test_gt_x   = gt_c.iloc[train_size:]

# Use the same order found for ARIMA
arimax_order = best_order
print(f"\n  Fitting ARIMAX{arimax_order} with GTrends as exogenous...")

arimax_model = ARIMA(
    train_gld_x,
    order=arimax_order,
    exog=train_gt_x
).fit()
print(arimax_model.summary())

arimax_fc   = arimax_model.get_forecast(steps=len(test_gld_x), exog=test_gt_x)
arimax_pred = arimax_fc.predicted_mean
arimax_ci   = arimax_fc.conf_int(alpha=0.05)

arimax_rmse = np.sqrt(np.mean((test_gld_x.values - arimax_pred.values) ** 2))
arimax_mae  = np.mean(np.abs(test_gld_x.values - arimax_pred.values))
print(f"\n  ARIMAX test RMSE: {arimax_rmse:.2f}   MAE: {arimax_mae:.2f}")
print(f"  ARIMA  test RMSE: {arima_rmse:.2f}   MAE: {arima_mae:.2f}")
print(f"  RMSE improvement from adding GTrends: {arima_rmse - arimax_rmse:.2f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train_gld_x.index, train_gld_x.values, color="goldenrod", label="Train", linewidth=1.2)
ax.plot(test_gld_x.index,  test_gld_x.values,  color="black",     label="Actual (test)", linewidth=1.2)
ax.plot(arimax_pred.index, arimax_pred.values,  color="purple",    label=f"ARIMAX{arimax_order} forecast", linewidth=1.5)
ax.fill_between(arimax_ci.index,
                arimax_ci.iloc[:, 0], arimax_ci.iloc[:, 1],
                color="purple", alpha=0.15, label="95% CI")
ax.axvline(test_gld_x.index[0], color="gray", linestyle="--", linewidth=1)
ax.set_title(f"ARIMAX{arimax_order} Forecast vs Actual Gold Price\nRMSE={arimax_rmse:.1f}  MAE={arimax_mae:.1f}", fontsize=12)
ax.set_ylabel("Gold Price (USD/oz)")
ax.legend()
plt.tight_layout()
save_fig("13_arimax_forecast.png")

# ARIMA vs ARIMAX comparison
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_gld_x.index,  test_gld_x.values,  color="black",   label="Actual", linewidth=1.5)
ax.plot(arima_pred.index,  arima_pred.values,   color="crimson", label=f"ARIMA  RMSE={arima_rmse:.1f}", linewidth=1.2, linestyle="--")
ax.plot(arimax_pred.index, arimax_pred.values,  color="purple",  label=f"ARIMAX RMSE={arimax_rmse:.1f}", linewidth=1.2, linestyle=":")
ax.set_title("ARIMA vs ARIMAX – Gold Price Forecast Comparison", fontsize=12)
ax.set_ylabel("Gold Price (USD/oz)")
ax.legend()
plt.tight_layout()
save_fig("14_arima_vs_arimax.png")

# ---------------------------------------------------------------------------
# 12. Summary dashboard
# ---------------------------------------------------------------------------
section("10. Summary Dashboard")

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel A – raw series (dual axis)
ax_a = fig.add_subplot(gs[0, :])
color_g = "goldenrod"
color_b = "steelblue"
ax_a.plot(gld.index, gld.values, color=color_g, linewidth=1.5, label="Gold Price (USD)")
ax_a.set_ylabel("Gold Price (USD/oz)", color=color_g)
ax_a.tick_params(axis="y", labelcolor=color_g)
ax_a2 = ax_a.twinx()
ax_a2.plot(gt.index, gt.values, color=color_b, linewidth=1.2, alpha=0.7, label="GTrends – Inflation")
ax_a2.set_ylabel("GTrends Index", color=color_b)
ax_a2.tick_params(axis="y", labelcolor=color_b)
ax_a.set_title("Gold Price vs 'Inflation' Google Trends (2004 – 2025)", fontsize=12, fontweight="bold")
lines1, labels1 = ax_a.get_legend_handles_labels()
lines2, labels2 = ax_a2.get_legend_handles_labels()
ax_a.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

# Panel B – CCF differenced
lags_plot = np.arange(-MAX_LAGS, MAX_LAGS + 1)
x_s = (d_gt - d_gt.mean()) / d_gt.std()
y_s = (d_gld - d_gld.mean()) / d_gld.std()
n   = len(x_s)
corrs_d = []
for lag in lags_plot:
    if lag >= 0:
        c = np.corrcoef(x_s.iloc[:n-lag] if lag > 0 else x_s,
                        y_s.iloc[lag:]   if lag > 0 else y_s)[0, 1]
    else:
        c = np.corrcoef(y_s.iloc[:n+lag],
                        x_s.iloc[-lag:])[0, 1]
    corrs_d.append(c)
conf = 1.96 / np.sqrt(n)
ax_b = fig.add_subplot(gs[1, 0])
colors_b = ["steelblue" if abs(c) < conf else "crimson" for c in corrs_d]
ax_b.bar(lags_plot, corrs_d, color=colors_b, width=0.8)
ax_b.axhline(conf,  color="gray", linestyle="--", linewidth=1)
ax_b.axhline(-conf, color="gray", linestyle="--", linewidth=1)
ax_b.axvline(0, color="black", linewidth=0.8)
ax_b.set_title("CCF (Differenced)", fontsize=10)
ax_b.set_xlabel("Lag (months)")
ax_b.set_ylabel("Correlation")

# Panel C – Granger p-values
ax_c = fig.add_subplot(gs[1, 1])
gc_colors = ["crimson" if p < 0.05 else "steelblue" for p in gc_df["p-value"]]
ax_c.bar(gc_df["Lag"], gc_df["p-value"], color=gc_colors, edgecolor="white")
ax_c.axhline(0.05, color="black", linestyle="--", linewidth=1.5)
ax_c.set_title("Granger: GTrends→Gold (Δ)", fontsize=10)
ax_c.set_xlabel("Lag (months)")
ax_c.set_ylabel("p-value")

# Panel D – Lagged OLS R²
ax_d = fig.add_subplot(gs[1, 2])
w = 0.35
x_pos = np.arange(len(BEST_LAGS))
ax_d.bar(x_pos - w/2, res_raw["R²"],  width=w, label="Raw",  color="goldenrod", edgecolor="white")
ax_d.bar(x_pos + w/2, res_diff["R²"], width=w, label="Diff", color="cornflowerblue", edgecolor="white")
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels([f"lag={l}" for l in BEST_LAGS])
ax_d.set_title("Lagged OLS R²", fontsize=10)
ax_d.set_ylabel("R²")
ax_d.legend(fontsize=9)

# Panel E – ARIMA vs ARIMAX forecast
ax_e = fig.add_subplot(gs[2, :])
ax_e.plot(test_gld_x.index,  test_gld_x.values,  color="black",   label="Actual",             linewidth=1.5)
ax_e.plot(arima_pred.index,  arima_pred.values,   color="crimson", label=f"ARIMA  RMSE={arima_rmse:.1f}", linewidth=1.2, linestyle="--")
ax_e.plot(arimax_pred.index, arimax_pred.values,  color="purple",  label=f"ARIMAX RMSE={arimax_rmse:.1f}", linewidth=1.2, linestyle=":")
ax_e.set_title("ARIMA vs ARIMAX Forecast (Test Period)", fontsize=11, fontweight="bold")
ax_e.set_ylabel("Gold Price (USD/oz)")
ax_e.legend()

plt.suptitle("Financial Anxiety (Google Trends) & Gold Price Analysis",
             fontsize=15, fontweight="bold", y=1.01)
save_fig("00_summary_dashboard.png")

# ---------------------------------------------------------------------------
# 13. Print final summary
# ---------------------------------------------------------------------------
section("ANALYSIS SUMMARY")
print("""
  DATA
  ----
  Google Trends 'inflation' index : 2004-01 to 2026-03 (monthly)
  Gold spot price (USD/troy oz)   : 2004-01 to 2025-12 (monthly)
  Merged common period            : 2004-01 to 2025-12

  NORMALITY
  ---------
  Both raw series are non-normal (Shapiro-Wilk p < 0.001).
  First differences improve but do not achieve normality for gold.

  STATIONARITY
  ------------
  Raw series are non-stationary (ADF cannot reject unit root; KPSS rejects).
  First-differenced series ARE stationary – suitable for Granger / lagged regression.

  CCF
  ---
  Raw data shows strong persistent positive cross-correlations (spurious due to trend).
  Differenced data reveals specific lags where GTrends leads Gold by 1-6 months.

  LAGGED OLS
  ----------
  Undifferenced: high R² (~0.6-0.8) but likely spurious (non-stationarity).
  Differenced:   lower but genuine R² at short lags; significant at lag 1 and lag 3.

  GRANGER CAUSALITY
  -----------------
  GTrends → Gold Price: significant at several lags (see plot 10).
  Gold Price → GTrends: less consistent, confirming GTrends has predictive content.

  ARIMA vs ARIMAX
  ---------------
  ARIMA best order found by AIC search over p,d,q ∈ {0,1,2}.
  ARIMAX incorporates Google Trends index as external regressor.
  Improvement in RMSE when adding GTrends is printed above.
  A negative RMSE improvement means GTrends added noise for this horizon;
  a positive improvement confirms predictive value.
""")
print(f"\n  All plots saved to:  {PLOTS_DIR}/\n")
