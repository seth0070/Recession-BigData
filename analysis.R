# =============================================================================
# Recession Google Trends vs. Gold Price Analysis
# =============================================================================
# This script investigates whether rising global online concern about potential
# stock market crashes — captured through Google Trends search intensity for
# "Recession" — is associated with subsequent increases in gold prices.
#
# Data sources:
#   - Monthly gold prices (USD/oz): Bank of England database
#     File: gold-300.xls  (HTML table exported from stlouisfed/Bank of England)
#   - Google Trends data for "Recession": Google Trends (global, monthly)
#     File: multiTimeline (3).csv  (exported from trends.google.com)
#
# Analysis steps:
#   1. Load and preprocess data
#   2. Calculate monthly gold price % returns
#   3. Visualise both time series
#   4. Pearson correlation & hypothesis test (contemporaneous)
#   5. Lagged cross-correlation analysis (does Trends predict future returns?)
#   6. Lead-lag regression (Trends[t] -> gold return[t+1])
#   7. Granger causality test
#
# Requirements: base R only (no external packages required)
# =============================================================================

# ── 1. Load data ──────────────────────────────────────────────────────────────

# 1a. Parse gold prices from Bank of England HTML table (gold-300.xls)
#     The file is an HTML document saved with an .xls extension.
#     Each data row has the pattern: <td>Mon YYYY</td><td>Price</td><td>Change</td>
gold_html  <- paste(readLines("gold-300.xls", warn = FALSE), collapse = "\n")
gold_rows  <- regmatches(gold_html,
               gregexpr("<td>[^<]+</td><td>[^<]+</td><td>[^<]+</td>",
                         gold_html))[[1]]
parse_row  <- function(row) {
  cells <- regmatches(row, gregexpr("(?<=<td>)[^<]+(?=</td>)", row, perl = TRUE))[[1]]
  list(month = cells[1], price = cells[2])
}
parsed     <- lapply(gold_rows, parse_row)

# Convert "Mon YYYY" -> Date, strip commas from prices (e.g. "1,043.16" -> 1043.16)
gold_raw <- data.frame(
  Date      = as.Date(paste0("01 ", sapply(parsed, `[[`, "month")), "%d %b %Y"),
  Price_USD = as.numeric(gsub(",", "", sapply(parsed, `[[`, "price"))),
  stringsAsFactors = FALSE
)
gold_raw <- gold_raw[order(gold_raw$Date), ]

# 1b. Parse Google Trends CSV (multiTimeline (3).csv)
#     The file has two header lines before the actual column headers:
#       Line 1: "Category: All categories"
#       Line 2: blank
#       Line 3: "Month,recession: (Worldwide)"   <- real header
trends_raw  <- read.csv("multiTimeline (3).csv", skip = 2)
# R converts "recession: (Worldwide)" to "recession...Worldwide." — rename it
colnames(trends_raw)[colnames(trends_raw) != "Month"] <- "Recession_Trend"

# ── 2. Preprocess gold prices ─────────────────────────────────────────────────
# Monthly percentage return: (P_t - P_{t-1}) / P_{t-1} * 100
n <- nrow(gold_raw)
gold_return <- c(NA, (gold_raw$Price_USD[-1] / gold_raw$Price_USD[-n] - 1) * 100)

gold_raw <- data.frame(
  Date        = gold_raw$Date,
  YM          = format(gold_raw$Date, "%Y-%m"),
  Price_USD   = gold_raw$Price_USD,
  Gold_Return = gold_return
)
gold <- gold_raw[!is.na(gold_raw$Gold_Return), ]

cat("Gold price data:", nrow(gold), "monthly observations\n")
cat("Period:", format(min(gold$Date)), "to", format(max(gold$Date)), "\n\n")

# ── 3. Preprocess Google Trends ───────────────────────────────────────────────
trends_raw$Date <- as.Date(paste0(trends_raw$Month, "-01"))
trends_raw      <- trends_raw[order(trends_raw$Date), ]
trends <- data.frame(
  Date            = trends_raw$Date,
  YM              = format(trends_raw$Date, "%Y-%m"),
  Recession_Trend = trends_raw$Recession_Trend
)

cat("Google Trends data:", nrow(trends), "monthly observations\n")
cat("Period:", format(min(trends$Date)), "to", format(max(trends$Date)), "\n\n")

# ── 4. Merge datasets on Year-Month key ───────────────────────────────────────
df <- merge(
  gold[, c("YM", "Date", "Price_USD", "Gold_Return")],
  trends[, c("YM", "Recession_Trend")],
  by = "YM"
)
df <- df[order(df$Date), ]
rownames(df) <- NULL

cat("Merged dataset:", nrow(df), "observations\n\n")

# ── 5. Summary statistics ─────────────────────────────────────────────────────
cat("=== Summary Statistics ===\n")
cat("\nGold Monthly Return (%):\n")
print(summary(df$Gold_Return))
cat("\nGoogle Trends - Recession:\n")
print(summary(df$Recession_Trend))

# ── 6. Time-series visualisations (base R graphics) ──────────────────────────
dir.create("output", showWarnings = FALSE)

png("output/time_series.png", width = 900, height = 1100, res = 110)
par(mfrow = c(3, 1), mar = c(4, 4.5, 3, 1), oma = c(0, 0, 1, 0))

# Panel 1: Gold price level
all_dates <- as.numeric(df$Date)
plot(df$Date, df$Price_USD,
     type = "l", col = "#DAA520", lwd = 2,
     xlab = "", ylab = "Price (USD/oz)",
     main = "Monthly Gold Price (USD/oz)",
     xaxt = "n")
# x-axis tick marks: every 2 years, aligned to Jan 1 of the start/end year
year_breaks <- seq(as.Date(paste0(format(min(df$Date), "%Y"), "-01-01")),
                   as.Date(paste0(format(max(df$Date), "%Y"), "-01-01")),
                   by = "2 years")
axis(1, at = year_breaks, labels = format(year_breaks, "%Y"))

# Panel 2: Gold monthly return (bar chart)
bar_cols <- ifelse(df$Gold_Return >= 0, "#228B22", "#B22222")
barplot(df$Gold_Return,
        col    = bar_cols,
        border = NA,
        names.arg = rep("", nrow(df)),
        ylab   = "Monthly Return (%)",
        main   = "Gold Monthly Return (%)")
abline(h = 0, col = "black", lwd = 1)
legend("topright", legend = c("Positive", "Negative"),
       fill = c("#228B22", "#B22222"), bty = "n", cex = 0.8)

# Panel 3: Google Trends
plot(df$Date, df$Recession_Trend,
     type = "l", col = "#4169E1", lwd = 2,
     xlab = "Date", ylab = "Search Index (0-100)",
     main = "Google Trends: \"Recession\" Search Intensity (Global)",
     xaxt = "n", ylim = c(0, 100))
axis(1, at = year_breaks, labels = format(year_breaks, "%Y"))
polygon(c(df$Date[1], df$Date, tail(df$Date, 1)),
        c(0, df$Recession_Trend, 0),
        col = adjustcolor("#4169E1", alpha.f = 0.3), border = NA)

invisible(dev.off())
cat("Saved: output/time_series.png\n")

# ── 7. Contemporaneous correlation & hypothesis test ─────────────────────────
cat("\n=== Hypothesis Test 1: Contemporaneous Correlation ===\n")
cat("H0: cor(Recession_Trends[t], Gold_Return[t]) = 0\n")
cat("H1: cor(Recession_Trends[t], Gold_Return[t]) != 0  (two-sided)\n\n")

cor_test_contemp <- cor.test(
  df$Recession_Trend,
  df$Gold_Return,
  method      = "pearson",
  alternative = "two.sided"
)
print(cor_test_contemp)

cat(sprintf(
  "\nContemporaneous Pearson r = %.4f  (95%% CI: [%.4f, %.4f])\n",
  cor_test_contemp$estimate,
  cor_test_contemp$conf.int[1],
  cor_test_contemp$conf.int[2]
))
cat(sprintf("p-value = %.4f  =>  %s\n",
  cor_test_contemp$p.value,
  ifelse(cor_test_contemp$p.value < 0.05,
         "Reject H0: significant correlation",
         "Fail to reject H0: no significant correlation")))

# Scatter + regression line: contemporaneous
png("output/scatter_contemporaneous.png", width = 750, height = 580, res = 110)
par(mar = c(4.5, 4.5, 3.5, 1))
lm_contemp <- lm(Gold_Return ~ Recession_Trend, data = df)
plot(df$Recession_Trend, df$Gold_Return,
     col  = adjustcolor("#4169E1", alpha.f = 0.5),
     pch  = 19, cex = 0.85,
     xlab = "Recession Google Trends Index",
     ylab = "Gold Monthly Return (%)",
     main = "Google Trends \"Recession\" vs. Gold Monthly Return (Same Month)")
abline(lm_contemp, col = "#DAA520", lwd = 2)
mtext(sprintf("Pearson r = %.3f,  p = %.4f",
              cor_test_contemp$estimate, cor_test_contemp$p.value),
      side = 3, line = 0.3, cex = 0.85)
invisible(dev.off())
cat("Saved: output/scatter_contemporaneous.png\n")

# ── 8. Lagged cross-correlation analysis ─────────────────────────────────────
cat("\n=== Lagged Cross-Correlation (CCF) Analysis ===\n")
cat("Negative lag k: Trends leads gold return by |k| months\n")
cat("Positive lag k: Gold return leads trends by k months\n\n")

ccf_result <- ccf(df$Recession_Trend, df$Gold_Return,
                  lag.max = 12, plot = FALSE)

ccf_df <- data.frame(
  Lag         = as.integer(ccf_result$lag),
  Correlation = as.numeric(ccf_result$acf)
)

n_obs    <- nrow(df)
crit_val <- 1.96 / sqrt(n_obs)

cat("Critical value (95% CI) = +/-", round(crit_val, 4), "\n")
cat("\nCross-correlations at each lag:\n")
print(ccf_df, row.names = FALSE)

sig_lags <- ccf_df[abs(ccf_df$Correlation) > crit_val, ]
if (nrow(sig_lags) > 0) {
  cat("\nSignificant lags (|r| > critical value):\n")
  print(sig_lags, row.names = FALSE)
} else {
  cat("\nNo lags show statistically significant cross-correlation.\n")
}

png("output/ccf_plot.png", width = 850, height = 480, res = 110)
par(mar = c(4.5, 4.5, 3.5, 1))
bar_heights <- ccf_df$Correlation
bar_cols_ccf <- ifelse(abs(bar_heights) > crit_val, "#DAA520", "#AAAAAA")
bp <- barplot(bar_heights,
              names.arg = ccf_df$Lag,
              col       = bar_cols_ccf,
              border    = NA,
              xlab      = "Lag (months)  [negative = Trends leads Gold Return]",
              ylab      = "Cross-Correlation",
              main      = "CCF: Google Trends \"Recession\" -> Gold Monthly Return",
              ylim      = c(min(bar_heights) - 0.05, max(bar_heights) + 0.08))
abline(h =  crit_val, lty = 2, col = "#B22222")
abline(h = -crit_val, lty = 2, col = "#B22222")
abline(h = 0, col = "black")
legend("topright",
       legend = c(sprintf("Significant (|r| > %.3f)", crit_val), "Not significant"),
       fill   = c("#DAA520", "#AAAAAA"), bty = "n", cex = 0.8)
invisible(dev.off())
cat("Saved: output/ccf_plot.png\n")

# ── 9. Lead-lag regression: Trends[t-1] -> Gold Return[t] ─────────────────────
cat("\n=== Hypothesis Test 2: Predictive Regression (1-month lead) ===\n")
cat("H0: beta_1 = 0  (Trends at t-1 has no effect on Gold Return at t)\n")
cat("H1: beta_1 > 0  (Higher Trends at t-1 predicts positive Gold Return at t)\n\n")

trends_lag1 <- c(NA, df$Recession_Trend[-nrow(df)])
df_lag <- df
df_lag$Trends_Lag1 <- trends_lag1
df_lag <- df_lag[!is.na(df_lag$Trends_Lag1), ]

lm_lag1 <- lm(Gold_Return ~ Trends_Lag1, data = df_lag)
cat("OLS: Gold_Return[t] ~ Recession_Trend[t-1]\n")
print(summary(lm_lag1))

coef_table <- coef(summary(lm_lag1))
beta1      <- coef_table["Trends_Lag1", "Estimate"]
se_beta1   <- coef_table["Trends_Lag1", "Std. Error"]
t_stat     <- beta1 / se_beta1
p_one_sid  <- pt(t_stat, df = lm_lag1$df.residual, lower.tail = FALSE)

cat(sprintf(
  "\nOne-sided test (H1: beta > 0): t = %.4f, p = %.4f  =>  %s\n",
  t_stat, p_one_sid,
  ifelse(p_one_sid < 0.05,
         "Reject H0: Trends positively predicts next-month gold return",
         "Fail to reject H0: no positive predictive relationship")
))

png("output/scatter_lag1.png", width = 750, height = 580, res = 110)
par(mar = c(4.5, 4.5, 3.5, 1))
plot(df_lag$Trends_Lag1, df_lag$Gold_Return,
     col  = adjustcolor("#4169E1", alpha.f = 0.5),
     pch  = 19, cex = 0.85,
     xlab = "Recession Google Trends Index (1-month lag)",
     ylab = "Gold Monthly Return (%)",
     main = "Recession Trends[t-1] vs. Gold Return[t]")
abline(lm_lag1, col = "#DAA520", lwd = 2)
mtext(sprintf("beta = %.4f,  p (one-sided) = %.4f", beta1, p_one_sid),
      side = 3, line = 0.3, cex = 0.85)
invisible(dev.off())
cat("Saved: output/scatter_lag1.png\n")

# ── 10. Granger causality test (manual F-test) ────────────────────────────────
# Tests whether lagged Trends values improve prediction of gold returns
# beyond gold's own lags alone.
# H0: the coefficients on all lags of Trends are jointly zero.
granger_test <- function(y, x, lag_order) {
  n_full <- length(y)
  if (n_full <= 2 * lag_order + 1) {
    return(list(F = NA, p = NA))
  }

  # Build design matrices (rows = lag_order+1 : n_full)
  idx <- (lag_order + 1):n_full

  # Restricted model: y_t ~ y_{t-1}, ..., y_{t-k}
  Yr <- matrix(NA, nrow = length(idx), ncol = lag_order)
  for (j in seq_len(lag_order)) Yr[, j] <- y[idx - j]
  mod_r <- lm(y[idx] ~ Yr)

  # Unrestricted model: adds lags of x
  Xu <- matrix(NA, nrow = length(idx), ncol = lag_order)
  for (j in seq_len(lag_order)) Xu[, j] <- x[idx - j]
  mod_u <- lm(y[idx] ~ Yr + Xu)

  rss_r  <- sum(residuals(mod_r)^2)
  rss_u  <- sum(residuals(mod_u)^2)
  df_r   <- mod_r$df.residual
  df_u   <- mod_u$df.residual
  F_stat <- ((rss_r - rss_u) / lag_order) / (rss_u / df_u)
  p_val  <- pf(F_stat, df1 = lag_order, df2 = df_u, lower.tail = FALSE)
  list(F = F_stat, p = p_val)
}

cat("\n=== Granger Causality Test ===\n")
cat("H0: Trends does NOT Granger-cause gold returns\n\n")

for (k in 1:3) {
  gc <- granger_test(df$Gold_Return, df$Recession_Trend, lag_order = k)
  cat(sprintf("Lag %d: F = %.4f, p = %.4f  =>  %s\n",
    k,
    gc$F,
    gc$p,
    ifelse(gc$p < 0.05,
           "Granger causality detected",
           "No Granger causality")))
}

# ── 11. Results summary ───────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("RESULTS SUMMARY\n")
cat(strrep("=", 60), "\n\n")

cat("1. Contemporaneous correlation:\n")
cat(sprintf("   r = %.4f, p = %.4f (%s)\n\n",
  cor_test_contemp$estimate,
  cor_test_contemp$p.value,
  ifelse(cor_test_contemp$p.value < 0.05, "significant", "not significant")))

cat("2. Most predictive CCF lag (Trends leads gold, negative lags):\n")
neg_lags <- ccf_df[ccf_df$Lag < 0, ]
if (nrow(neg_lags) > 0) {
  best_row <- neg_lags[which.max(abs(neg_lags$Correlation)), ]
  cat(sprintf("   Lag %d months: r = %.4f (%s)\n\n",
    best_row$Lag,
    best_row$Correlation,
    ifelse(abs(best_row$Correlation) > crit_val, "significant", "not significant")))
} else {
  cat("   No negative lags available.\n\n")
}

cat("3. 1-month lead regression:\n")
cat(sprintf("   beta = %.4f, p (one-sided) = %.4f (%s)\n\n",
  beta1, p_one_sid,
  ifelse(p_one_sid < 0.05, "significant", "not significant")))

cat("4. Output files saved in ./output/\n")
cat("   - time_series.png\n")
cat("   - scatter_contemporaneous.png\n")
cat("   - ccf_plot.png\n")
cat("   - scatter_lag1.png\n")

cat("\nAnalysis complete.\n")
