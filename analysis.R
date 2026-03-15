###############################################################################
# Time-Series Analysis: Does Google Trends Inflation Search Interest
# Predict Monthly Gold Returns?
#
# Research Question:
#   Does rising public concern about inflation, captured through Google Trends
#   search intensity for "inflation", predict subsequent increases in monthly
#   gold returns over a medium-term horizon of 10-12 months?
#
# Hypothesis:
#   When Google Trends search interest for "inflation" rises in month t,
#   monthly gold percentage returns will be significantly elevated approximately
#   10-12 months later, reflecting the delayed translation of collective
#   behavioural signals into commodity price movements through institutional
#   and policy channels.
#
# Author: WBS Big Data Analytics Assignment
# Date: 2025
###############################################################################

# =============================================================================
# Load Required Packages
# =============================================================================
library(tidyverse)
library(lubridate)
library(rvest)
library(tseries)
library(lmtest)
library(forecast)
library(ggplot2)

cat("\n======================================================================\n")
cat("PACKAGES LOADED SUCCESSFULLY\n")
cat("======================================================================\n\n")

# =============================================================================
# STEP 1: DATA LOADING AND CLEANING
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 1: DATA LOADING AND CLEANING\n")
cat("######################################################################\n\n")

# --- 1a. Load Google Trends CSV ---
# The file has a header row ("Category: All categories") and a blank line
# before the actual data starting at row 3
trends_raw <- read_csv("multiTimeline.csv", skip = 2, show_col_types = FALSE)
colnames(trends_raw) <- c("Month", "search_index")

# Parse the month column to a proper date
trends_raw <- trends_raw %>%
  mutate(date = ymd(paste0(Month, "-01"))) %>%
  select(date, search_index) %>%
  filter(!is.na(date))

cat("Google Trends data loaded:\n")
cat("  Rows:", nrow(trends_raw), "\n")
cat("  Date range:", as.character(min(trends_raw$date)), "to",
    as.character(max(trends_raw$date)), "\n")
cat("  Search index range:", min(trends_raw$search_index, na.rm = TRUE), "to",
    max(trends_raw$search_index, na.rm = TRUE), "\n\n")

# --- 1b. Load Gold Price XLS (HTML table format) ---
# The gold-300.xls file is actually an HTML table saved with .xls extension
# We use rvest to read the HTML table
gold_html <- read_html("gold-300.xls")
gold_table <- gold_html %>%
  html_table(fill = TRUE)
gold_raw <- gold_table[[1]]

# Clean column names
colnames(gold_raw) <- c("Month", "Price", "Change")

# Remove the header row if it got included
gold_raw <- gold_raw %>% filter(Month != "Month")

# Parse the date column (format: "Jan 2004")
gold_raw <- gold_raw %>%
  mutate(
    # Remove commas from price and convert to numeric
    Price = as.numeric(gsub(",", "", Price)),
    # Parse the "Mon YYYY" date format
    date = dmy(paste0("01 ", Month))
  ) %>%
  select(date, gold_price = Price) %>%
  filter(!is.na(date), !is.na(gold_price))

cat("Gold price data loaded:\n")
cat("  Rows:", nrow(gold_raw), "\n")
cat("  Date range:", as.character(min(gold_raw$date)), "to",
    as.character(max(gold_raw$date)), "\n")
cat("  Price range: $", min(gold_raw$gold_price, na.rm = TRUE), "to $",
    max(gold_raw$gold_price, na.rm = TRUE), "\n\n")

# --- 1c. Merge datasets by month using inner_join ---
merged_data <- inner_join(trends_raw, gold_raw, by = "date")

# Filter to overlapping range (Jan 2004 onwards)
merged_data <- merged_data %>%
  filter(date >= ymd("2004-01-01")) %>%
  arrange(date)

cat("Merged dataset:\n")
cat("  Rows after merge:", nrow(merged_data), "\n")
cat("  Date range:", as.character(min(merged_data$date)), "to",
    as.character(max(merged_data$date)), "\n")
cat("  Missing values in search_index:", sum(is.na(merged_data$search_index)), "\n")
cat("  Missing values in gold_price:", sum(is.na(merged_data$gold_price)), "\n\n")

# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 2: FEATURE ENGINEERING\n")
cat("######################################################################\n\n")

# --- 2a. Compute monthly gold percentage returns ---
merged_data <- merged_data %>%
  mutate(gold_return = (gold_price - lag(gold_price)) / lag(gold_price) * 100)

# --- 2b. Create lagged search index (lags 1-12) ---
for (i in 1:12) {
  merged_data <- merged_data %>%
    mutate(!!paste0("search_lag_", i) := lag(search_index, i))
}

# --- 2c. Drop rows with NAs introduced by lagging ---
analysis_data <- merged_data %>%
  filter(!is.na(gold_return), !is.na(search_lag_12))

cat("Feature engineering complete:\n")
cat("  Gold returns computed (monthly % change)\n")
cat("  Lagged search index variables created (lags 1-12)\n")
cat("  Rows after dropping NAs:", nrow(analysis_data), "\n")
cat("  Date range:", as.character(min(analysis_data$date)), "to",
    as.character(max(analysis_data$date)), "\n\n")

# --- 2d. Save clean dataset ---
write_csv(analysis_data, "analysis_dataset.csv")
cat("  Clean dataset saved to: analysis_dataset.csv\n\n")

# =============================================================================
# STEP 3: ASSUMPTION CHECKING
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 3: ASSUMPTION CHECKING\n")
cat("######################################################################\n\n")

# --- 3a. Shapiro-Wilk normality tests ---
# Note: Shapiro-Wilk test requires n <= 5000
sw_gold <- shapiro.test(analysis_data$gold_return)
sw_search <- shapiro.test(analysis_data$search_index)

cat("Shapiro-Wilk Normality Tests:\n")
cat("--------------------------------------------------------------\n")
cat(sprintf("  Gold Returns:    W = %.4f, p-value = %.6f %s\n",
            sw_gold$statistic, sw_gold$p.value,
            ifelse(sw_gold$p.value < 0.05, "(NOT NORMAL)", "(NORMAL)")))
cat(sprintf("  Search Index:    W = %.4f, p-value = %.6f %s\n",
            sw_search$statistic, sw_search$p.value,
            ifelse(sw_search$p.value < 0.05, "(NOT NORMAL)", "(NORMAL)")))
cat("--------------------------------------------------------------\n\n")

# --- 3b. Q-Q Plots ---
png("qq_plots.png", width = 1000, height = 500, res = 120)
par(mfrow = c(1, 2))
qqnorm(analysis_data$gold_return, main = "Q-Q Plot: Gold Returns (%)",
       col = "steelblue", pch = 16, cex = 0.7)
qqline(analysis_data$gold_return, col = "red", lwd = 2)
qqnorm(analysis_data$search_index, main = "Q-Q Plot: Inflation Search Index",
       col = "darkorange", pch = 16, cex = 0.7)
qqline(analysis_data$search_index, col = "red", lwd = 2)
dev.off()
cat("Q-Q plots saved to: qq_plots.png\n\n")

# --- 3c. Augmented Dickey-Fuller tests for stationarity ---
adf_gold <- adf.test(analysis_data$gold_return, alternative = "stationary")
adf_search <- adf.test(analysis_data$search_index, alternative = "stationary")

cat("Augmented Dickey-Fuller Tests (Stationarity):\n")
cat("--------------------------------------------------------------\n")
cat(sprintf("  Gold Returns:    Dickey-Fuller = %.4f, Lag = %d, p-value = %.4f %s\n",
            adf_gold$statistic, adf_gold$parameter, adf_gold$p.value,
            ifelse(adf_gold$p.value < 0.05, "(STATIONARY)", "(NON-STATIONARY)")))
cat(sprintf("  Search Index:    Dickey-Fuller = %.4f, Lag = %d, p-value = %.4f %s\n",
            adf_search$statistic, adf_search$parameter, adf_search$p.value,
            ifelse(adf_search$p.value < 0.05, "(STATIONARY)", "(NON-STATIONARY)")))
cat("--------------------------------------------------------------\n\n")

# --- 3d. Determine correlation method ---
if (sw_gold$p.value < 0.05 | sw_search$p.value < 0.05) {
  cor_method <- "spearman"
  cat("Decision: At least one variable violates normality.\n")
  cat("  -> Using SPEARMAN rank correlation.\n\n")
} else {
  cor_method <- "pearson"
  cat("Decision: Both variables are approximately normal.\n")
  cat("  -> Using PEARSON correlation.\n\n")
}

# Compute correlation
cor_test <- cor.test(analysis_data$search_index, analysis_data$gold_return,
                     method = cor_method)
cat(sprintf("Correlation (%s) between search index and gold returns:\n", cor_method))
cat(sprintf("  rho/r = %.4f, p-value = %.6f, n = %d\n\n",
            cor_test$estimate, cor_test$p.value, nrow(analysis_data)))

# =============================================================================
# STEP 4: CROSS-CORRELATION FUNCTION (CCF)
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 4: CROSS-CORRELATION FUNCTION (CCF)\n")
cat("######################################################################\n\n")

# --- 4a. Run CCF ---
ccf_result <- ccf(analysis_data$search_index, analysis_data$gold_return,
                  lag.max = 12, plot = FALSE)

# --- 4b. Extract and print CCF values ---
ccf_df <- data.frame(
  lag = ccf_result$lag[, 1, 1],
  ccf = ccf_result$acf[, 1, 1]
)

# Approximate 95% confidence interval
n <- nrow(analysis_data)
ci_bound <- qnorm(0.975) / sqrt(n)

ccf_df <- ccf_df %>%
  mutate(significant = abs(ccf) > ci_bound)

cat("Cross-Correlation Function Results (lags -12 to +12):\n")
cat("Positive lag = search index LEADS gold returns\n")
cat(sprintf("95%% CI bounds: +/- %.4f (n = %d)\n", ci_bound, n))
cat("--------------------------------------------------------------\n")
cat(sprintf("%-6s %-10s %-12s\n", "Lag", "CCF", "Significant"))
cat("--------------------------------------------------------------\n")
for (i in 1:nrow(ccf_df)) {
  cat(sprintf("%-6d %-10.4f %-12s\n",
              ccf_df$lag[i], ccf_df$ccf[i],
              ifelse(ccf_df$significant[i], "***", "")))
}
cat("--------------------------------------------------------------\n\n")

# Identify significant positive lags (search predicts gold returns)
sig_pos <- ccf_df %>% filter(lag > 0, significant == TRUE)
if (nrow(sig_pos) > 0) {
  cat("Significant POSITIVE lags (search index predicts future gold returns):\n")
  for (i in 1:nrow(sig_pos)) {
    cat(sprintf("  Lag %d: CCF = %.4f\n", sig_pos$lag[i], sig_pos$ccf[i]))
  }
  best_lag <- sig_pos$lag[which.max(abs(sig_pos$ccf))]
  cat(sprintf("\nStrongest signal at lag %d\n\n", best_lag))
} else {
  cat("No statistically significant positive lags found.\n")
  # Default to lag 11 as per hypothesis
  best_lag <- 11
  cat(sprintf("Using hypothesised lag %d for further analysis.\n\n", best_lag))
}

# --- 4c. Save CCF plot (base R) ---
png("ccf_base_plot.png", width = 800, height = 500, res = 120)
ccf(analysis_data$search_index, analysis_data$gold_return,
    lag.max = 12,
    main = "Cross-Correlation: Inflation Search Index vs Gold Returns",
    xlab = "Lag (months)", ylab = "Cross-Correlation")
dev.off()
cat("Base CCF plot saved to: ccf_base_plot.png\n\n")

# =============================================================================
# STEP 5: LAGGED REGRESSION ANALYSIS
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 5: LAGGED REGRESSION ANALYSIS\n")
cat("######################################################################\n\n")

# --- 5a. Simple regressions at specific lags ---
test_lags <- c(1, 6, 10, 11, 12)
regression_results <- data.frame(
  lag = integer(),
  beta = numeric(),
  se = numeric(),
  t_stat = numeric(),
  p_value = numeric(),
  r_squared = numeric(),
  f_stat = numeric(),
  f_p_value = numeric(),
  stringsAsFactors = FALSE
)

cat("Simple Linear Regressions: gold_return ~ search_index(lagged)\n")
cat("====================================================================\n\n")

for (lag_val in test_lags) {
  lag_col <- paste0("search_lag_", lag_val)
  formula_str <- as.formula(paste("gold_return ~", lag_col))
  model <- lm(formula_str, data = analysis_data)
  s <- summary(model)
  f_stat_val <- s$fstatistic[1]
  f_p <- pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail = FALSE)

  cat(sprintf("--- Lag %d ---\n", lag_val))
  cat(sprintf("  β (slope)    = %.4f\n", coef(model)[2]))
  cat(sprintf("  Std. Error   = %.4f\n", s$coefficients[2, 2]))
  cat(sprintf("  t-statistic  = %.4f\n", s$coefficients[2, 3]))
  cat(sprintf("  p-value      = %.6f %s\n", s$coefficients[2, 4],
              ifelse(s$coefficients[2, 4] < 0.05, "*", "")))
  cat(sprintf("  R²           = %.4f\n", s$r.squared))
  cat(sprintf("  F-statistic  = %.4f (p = %.6f)\n", f_stat_val, f_p))
  cat(sprintf("  n            = %d\n\n", nrow(analysis_data)))

  regression_results <- rbind(regression_results, data.frame(
    lag = lag_val,
    beta = coef(model)[2],
    se = s$coefficients[2, 2],
    t_stat = s$coefficients[2, 3],
    p_value = s$coefficients[2, 4],
    r_squared = s$r.squared,
    f_stat = f_stat_val,
    f_p_value = f_p,
    stringsAsFactors = FALSE
  ))
}

# --- 5b. Multiple regression controlling for previous month's gold return ---
cat("====================================================================\n")
cat(sprintf("Multiple Regression at Best Lag (%d), controlling for lag-1 gold return:\n", best_lag))
cat("====================================================================\n\n")

# Create lag-1 gold return
analysis_data <- analysis_data %>%
  mutate(gold_return_lag1 = lag(gold_return))

# Drop NAs from the new lagged variable
analysis_multi <- analysis_data %>% filter(!is.na(gold_return_lag1))

best_lag_col <- paste0("search_lag_", best_lag)
multi_formula <- as.formula(paste("gold_return ~", best_lag_col, "+ gold_return_lag1"))
multi_model <- lm(multi_formula, data = analysis_multi)
multi_s <- summary(multi_model)

cat("Model: gold_return ~", best_lag_col, "+ gold_return_lag1\n\n")
print(multi_s)
cat("\n")

multi_f_p <- pf(multi_s$fstatistic[1], multi_s$fstatistic[2], multi_s$fstatistic[3],
                lower.tail = FALSE)
cat(sprintf("Overall F-statistic: %.4f (p = %.6f)\n", multi_s$fstatistic[1], multi_f_p))
cat(sprintf("Adjusted R²: %.4f\n", multi_s$adj.r.squared))
cat(sprintf("n = %d\n\n", nrow(analysis_multi)))

# =============================================================================
# STEP 6: GRANGER CAUSALITY TEST
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 6: GRANGER CAUSALITY TEST\n")
cat("######################################################################\n\n")

cat("Granger Causality Test: Does search_index Granger-cause gold_return?\n")
cat("Using lmtest::grangertest()\n")
cat("====================================================================\n\n")

granger_results <- data.frame(
  lag = integer(),
  f_stat = numeric(),
  p_value = numeric(),
  significant = character(),
  stringsAsFactors = FALSE
)

for (lag_val in 1:12) {
  gt <- grangertest(analysis_data$gold_return ~ analysis_data$search_index,
                    order = lag_val)
  f_val <- gt$F[2]
  p_val <- gt$`Pr(>F)`[2]

  granger_results <- rbind(granger_results, data.frame(
    lag = lag_val,
    f_stat = f_val,
    p_value = p_val,
    significant = ifelse(p_val < 0.05, "YES ***", "No"),
    stringsAsFactors = FALSE
  ))
}

cat(sprintf("%-6s %-12s %-12s %-12s\n", "Lag", "F-statistic", "p-value", "Significant"))
cat("--------------------------------------------------------------\n")
for (i in 1:nrow(granger_results)) {
  cat(sprintf("%-6d %-12.4f %-12.6f %-12s\n",
              granger_results$lag[i],
              granger_results$f_stat[i],
              granger_results$p_value[i],
              granger_results$significant[i]))
}
cat("--------------------------------------------------------------\n\n")

sig_granger <- granger_results %>% filter(grepl("YES", significant))
if (nrow(sig_granger) > 0) {
  cat("Significant Granger-causality found at lags:",
      paste(sig_granger$lag, collapse = ", "), "\n\n")
} else {
  cat("No statistically significant Granger-causality found at any lag.\n\n")
}

# =============================================================================
# STEP 7: ARIMAX MODEL
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 7: ARIMAX MODEL\n")
cat("######################################################################\n\n")

# --- 7a. Identify ARIMA order for gold returns ---
gold_ts <- ts(analysis_data$gold_return, frequency = 12)
auto_fit <- auto.arima(gold_ts)

cat("Auto-identified ARIMA order for gold returns:\n")
print(auto_fit)
cat(sprintf("\nAIC (plain ARIMA): %.2f\n\n", auto_fit$aic))

# --- 7b. ARIMAX with best lag search index as external regressor ---
xreg_col <- analysis_data[[best_lag_col]]

# Check for NAs in xreg
valid_idx <- !is.na(xreg_col)
gold_ts_valid <- ts(analysis_data$gold_return[valid_idx], frequency = 12)
xreg_valid <- matrix(xreg_col[valid_idx], ncol = 1)
colnames(xreg_valid) <- best_lag_col

cat(sprintf("Fitting ARIMAX with xreg = %s\n", best_lag_col))
cat(sprintf("Using same ARIMA order as identified: (%d,%d,%d)\n\n",
            auto_fit$arma[1], auto_fit$arma[6], auto_fit$arma[2]))

arimax_fit <- Arima(gold_ts_valid,
                    order = c(auto_fit$arma[1], auto_fit$arma[6], auto_fit$arma[2]),
                    xreg = xreg_valid)

# Also fit plain ARIMA on the same subset for fair comparison
arima_plain <- Arima(gold_ts_valid,
                     order = c(auto_fit$arma[1], auto_fit$arma[6], auto_fit$arma[2]))

cat("ARIMAX Model Summary:\n")
print(summary(arimax_fit))
cat(sprintf("\nAIC (ARIMAX):      %.2f\n", arimax_fit$aic))
cat(sprintf("AIC (plain ARIMA): %.2f\n", arima_plain$aic))
cat(sprintf("AIC difference:    %.2f (negative = ARIMAX is better)\n\n",
            arimax_fit$aic - arima_plain$aic))

if (arimax_fit$aic < arima_plain$aic) {
  cat("VERDICT: ARIMAX model (with search index) has LOWER AIC -> search index\n")
  cat("         adds predictive value beyond the ARIMA baseline.\n\n")
} else {
  cat("VERDICT: Plain ARIMA has lower or equal AIC -> search index does NOT\n")
  cat("         clearly add predictive value beyond the ARIMA baseline.\n\n")
}

# --- 7c. Residual diagnostics ---
png("arimax_residuals.png", width = 1000, height = 600, res = 120)
checkresiduals(arimax_fit)
dev.off()
cat("ARIMAX residual diagnostics saved to: arimax_residuals.png\n\n")

# =============================================================================
# STEP 8: VISUALISATIONS (4 publication-quality ggplot2 figures)
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 8: VISUALISATIONS\n")
cat("######################################################################\n\n")

# Set a clean theme for all plots
theme_set(theme_minimal(base_size = 12) +
            theme(plot.title = element_text(face = "bold", hjust = 0.5),
                  plot.subtitle = element_text(hjust = 0.5, color = "grey40"),
                  legend.position = "bottom"))

# --- Figure 1: Dual-axis time series ---
# Scale the search index to match gold return range for dual axis
scale_factor <- max(abs(analysis_data$gold_return), na.rm = TRUE) /
  max(analysis_data$search_index, na.rm = TRUE)

# Key event annotations
events <- data.frame(
  date = ymd(c("2008-09-01", "2020-03-01", "2022-06-01")),
  label = c("2008 Financial\nCrisis", "2020 COVID\nPandemic", "2022 Inflation\nSurge"),
  y_pos = c(12, 10, 8)
)

fig1 <- ggplot(analysis_data, aes(x = date)) +
  geom_line(aes(y = gold_return, colour = "Gold Returns (%)"),
            linewidth = 0.5, alpha = 0.8) +
  geom_line(aes(y = search_index * scale_factor,
                colour = "Inflation Search Index (scaled)"),
            linewidth = 0.5, alpha = 0.8) +
  scale_y_continuous(
    name = "Monthly Gold Return (%)",
    sec.axis = sec_axis(~ . / scale_factor,
                        name = "Google Trends Search Index (0-100)")
  ) +
  scale_colour_manual(values = c("Gold Returns (%)" = "#DAA520",
                                 "Inflation Search Index (scaled)" = "#4169E1")) +
  geom_vline(data = events, aes(xintercept = date),
             linetype = "dashed", colour = "grey50", alpha = 0.7) +
  geom_label(data = events, aes(x = date, y = y_pos, label = label),
             size = 2.5, fill = "white", alpha = 0.8, label.size = 0.3) +
  labs(title = "Google Trends Inflation Search Index vs Monthly Gold Returns",
       subtitle = "January 2004 to Present",
       x = "Date", colour = "") +
  theme(legend.position = "bottom")

ggsave("figure1_timeseries.png", fig1, width = 12, height = 6, dpi = 150)
cat("Figure 1 saved: figure1_timeseries.png\n")

# --- Figure 2: CCF Plot (ggplot2 styled) ---
fig2 <- ggplot(ccf_df, aes(x = lag, y = ccf)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_hline(yintercept = c(-ci_bound, ci_bound),
             linetype = "dashed", colour = "blue", linewidth = 0.5) +
  geom_segment(aes(x = lag, xend = lag, y = 0, yend = ccf,
                   colour = significant), linewidth = 1) +
  geom_point(aes(colour = significant), size = 2.5) +
  scale_colour_manual(values = c("TRUE" = "red", "FALSE" = "grey40"),
                      labels = c("TRUE" = "Significant", "FALSE" = "Not Significant"),
                      name = "") +
  annotate("rect", xmin = -12.5, xmax = -0.5,
           ymin = -Inf, ymax = Inf,
           fill = "grey90", alpha = 0.3) +
  annotate("text", x = -6, y = max(ccf_df$ccf) * 0.9,
           label = "Gold leads Search", colour = "grey50", size = 3) +
  annotate("text", x = 6, y = max(ccf_df$ccf) * 0.9,
           label = "Search leads Gold", colour = "grey50", size = 3) +
  scale_x_continuous(breaks = seq(-12, 12, 2)) +
  labs(title = "Cross-Correlation Function: Inflation Search Index vs Gold Returns",
       subtitle = paste0("95% Confidence Bands at ±",
                         sprintf("%.3f", ci_bound), " (n = ", n, ")"),
       x = "Lag (months)", y = "Cross-Correlation") +
  theme(legend.position = "bottom")

ggsave("figure2_ccf.png", fig2, width = 10, height = 6, dpi = 150)
cat("Figure 2 saved: figure2_ccf.png\n")

# --- Figure 3: Scatter plot at best lag ---
best_lag_data <- analysis_data %>%
  select(gold_return, search_lag = all_of(best_lag_col)) %>%
  filter(!is.na(search_lag))

fig3 <- ggplot(best_lag_data, aes(x = search_lag, y = gold_return)) +
  geom_point(colour = "#DAA520", alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE,
              colour = "#4169E1", fill = "#4169E1", alpha = 0.2) +
  labs(title = paste0("Inflation Search Index (Lag ", best_lag,
                      ") vs Subsequent Gold Returns"),
       subtitle = paste0("Simple linear regression with 95% confidence interval"),
       x = paste0("Google Trends Search Index (", best_lag, " months prior)"),
       y = "Monthly Gold Return (%)") +
  annotate("text", x = max(best_lag_data$search_lag) * 0.7,
           y = max(best_lag_data$gold_return) * 0.9,
           label = sprintf("β = %.4f\nR² = %.4f\np = %.4f",
                           regression_results$beta[regression_results$lag == best_lag],
                           regression_results$r_squared[regression_results$lag == best_lag],
                           regression_results$p_value[regression_results$lag == best_lag]),
           size = 3.5, hjust = 0, colour = "grey30")

ggsave("figure3_scatter.png", fig3, width = 8, height = 6, dpi = 150)
cat("Figure 3 saved: figure3_scatter.png\n")

# --- Figure 4: Bar chart of gold returns by search interest tercile ---
tercile_data <- analysis_data %>%
  select(gold_return, search_lag = all_of(best_lag_col)) %>%
  filter(!is.na(search_lag)) %>%
  mutate(tercile = ntile(search_lag, 3),
         tercile_label = case_when(
           tercile == 1 ~ "Low (Bottom Third)",
           tercile == 2 ~ "Medium (Middle Third)",
           tercile == 3 ~ "High (Top Third)"
         ),
         tercile_label = factor(tercile_label,
                                levels = c("Low (Bottom Third)",
                                           "Medium (Middle Third)",
                                           "High (Top Third)")))

tercile_summary <- tercile_data %>%
  group_by(tercile_label) %>%
  summarise(
    mean_return = mean(gold_return, na.rm = TRUE),
    se_return = sd(gold_return, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  )

fig4 <- ggplot(tercile_summary, aes(x = tercile_label, y = mean_return,
                                     fill = tercile_label)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_errorbar(aes(ymin = mean_return - 1.96 * se_return,
                    ymax = mean_return + 1.96 * se_return),
                width = 0.2, linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_fill_manual(values = c("Low (Bottom Third)" = "#2196F3",
                               "Medium (Middle Third)" = "#FFC107",
                               "High (Top Third)" = "#F44336")) +
  labs(title = paste0("Average Gold Returns by Inflation Search Interest Tercile"),
       subtitle = paste0("Search index measured ", best_lag,
                         " months prior | Error bars = 95% CI"),
       x = paste0("Search Interest Tercile (Lag ", best_lag, " months)"),
       y = "Mean Monthly Gold Return (%)") +
  geom_text(aes(label = sprintf("%.2f%%\n(n=%d)", mean_return, n)),
            vjust = ifelse(tercile_summary$mean_return >= 0, -1.5, 1.5),
            size = 3.5) +
  theme(legend.position = "none") +
  coord_cartesian(ylim = c(min(tercile_summary$mean_return - 2 * tercile_summary$se_return) - 0.5,
                           max(tercile_summary$mean_return + 2 * tercile_summary$se_return) + 1))

ggsave("figure4_terciles.png", fig4, width = 8, height = 6, dpi = 150)
cat("Figure 4 saved: figure4_terciles.png\n\n")

# =============================================================================
# STEP 9: RESULTS SUMMARY
# =============================================================================
cat("\n######################################################################\n")
cat("# STEP 9: RESULTS SUMMARY\n")
cat("######################################################################\n\n")

# --- 9a. Summary Table ---
cat("====================================================================\n")
cat("                    COMPREHENSIVE RESULTS TABLE                     \n")
cat("====================================================================\n\n")

cat("--- A. Normality Tests (Shapiro-Wilk) ---\n")
cat(sprintf("%-25s W = %.4f   p = %.6f   %s\n",
            "Gold Returns", sw_gold$statistic, sw_gold$p.value,
            ifelse(sw_gold$p.value < 0.05, "NOT NORMAL", "NORMAL")))
cat(sprintf("%-25s W = %.4f   p = %.6f   %s\n\n",
            "Search Index", sw_search$statistic, sw_search$p.value,
            ifelse(sw_search$p.value < 0.05, "NOT NORMAL", "NORMAL")))

cat("--- B. Stationarity Tests (ADF) ---\n")
cat(sprintf("%-25s DF = %.4f   p = %.4f   %s\n",
            "Gold Returns", adf_gold$statistic, adf_gold$p.value,
            ifelse(adf_gold$p.value < 0.05, "STATIONARY", "NON-STATIONARY")))
cat(sprintf("%-25s DF = %.4f   p = %.4f   %s\n\n",
            "Search Index", adf_search$statistic, adf_search$p.value,
            ifelse(adf_search$p.value < 0.05, "STATIONARY", "NON-STATIONARY")))

cat(sprintf("--- C. Correlation (%s) ---\n", cor_method))
cat(sprintf("%-25s rho = %.4f   p = %.6f   n = %d   %s\n\n",
            "Search vs Gold Return",
            cor_test$estimate, cor_test$p.value, nrow(analysis_data),
            ifelse(cor_test$p.value < 0.05, "SIGNIFICANT", "NOT SIGNIFICANT")))

cat("--- D. Cross-Correlation (Significant Positive Lags) ---\n")
sig_pos_lags <- ccf_df %>% filter(lag > 0, significant == TRUE)
if (nrow(sig_pos_lags) > 0) {
  for (i in 1:nrow(sig_pos_lags)) {
    cat(sprintf("  Lag %-3d CCF = %.4f   SIGNIFICANT\n",
                sig_pos_lags$lag[i], sig_pos_lags$ccf[i]))
  }
} else {
  cat("  No significant positive lags found\n")
}
cat("\n")

cat("--- E. Lagged Regressions ---\n")
cat(sprintf("%-8s %-10s %-10s %-12s %-10s %-10s\n",
            "Lag", "Beta", "R-sq", "p-value", "F-stat", "Verdict"))
cat("--------------------------------------------------------------\n")
for (i in 1:nrow(regression_results)) {
  cat(sprintf("%-8d %-10.4f %-10.4f %-12.6f %-10.4f %-10s\n",
              regression_results$lag[i],
              regression_results$beta[i],
              regression_results$r_squared[i],
              regression_results$p_value[i],
              regression_results$f_stat[i],
              ifelse(regression_results$p_value[i] < 0.05,
                     "SIGNIF.", "Not Sig.")))
}
cat("\n")

cat("--- F. Granger Causality ---\n")
cat(sprintf("%-8s %-12s %-12s %-12s\n", "Lag", "F-statistic", "p-value", "Verdict"))
cat("--------------------------------------------------------------\n")
for (i in 1:nrow(granger_results)) {
  cat(sprintf("%-8d %-12.4f %-12.6f %-12s\n",
              granger_results$lag[i],
              granger_results$f_stat[i],
              granger_results$p_value[i],
              granger_results$significant[i]))
}
cat("\n")

cat("--- G. ARIMAX Model Comparison ---\n")
cat(sprintf("  ARIMA order:        (%d,%d,%d)\n",
            auto_fit$arma[1], auto_fit$arma[6], auto_fit$arma[2]))
cat(sprintf("  AIC (plain ARIMA):  %.2f\n", arima_plain$aic))
cat(sprintf("  AIC (ARIMAX):       %.2f\n", arimax_fit$aic))
cat(sprintf("  AIC improvement:    %.2f\n", arima_plain$aic - arimax_fit$aic))
if (arimax_fit$aic < arima_plain$aic) {
  cat("  VERDICT: ARIMAX improves on plain ARIMA -> search index adds value\n\n")
} else {
  cat("  VERDICT: Plain ARIMA is equal or better -> limited added value\n\n")
}

# --- 9b. Plain English Interpretation ---
cat("====================================================================\n")
cat("            PLAIN ENGLISH INTERPRETATION\n")
cat("====================================================================\n\n")

cat("Research Question: Does rising public concern about inflation,\n")
cat("captured through Google Trends search intensity for 'inflation',\n")
cat("predict subsequent increases in monthly gold returns over a\n")
cat("medium-term horizon of 10-12 months?\n\n")

cat("Key Findings:\n\n")

cat("1. DATA QUALITY: The analysis uses", nrow(analysis_data),
    "monthly observations\n")
cat("   from", as.character(min(analysis_data$date)), "to",
    as.character(max(analysis_data$date)), ".\n\n")

cat("2. NORMALITY: The Shapiro-Wilk tests indicate whether the variables\n")
cat("   follow a normal distribution. Based on the results,", cor_method,
    "correlation\n")
cat("   was used as the appropriate measure of association.\n\n")

cat("3. STATIONARITY: The ADF tests confirm whether both series are\n")
cat("   stationary (a requirement for valid time-series regression).\n")
cat("   Gold returns are expected to be stationary; the search index\n")
cat("   may require differencing if non-stationary.\n\n")

cat("4. CROSS-CORRELATION: The CCF analysis reveals the temporal\n")
cat("   relationship between search interest and gold returns.\n")
cat("   Positive lags indicate search interest predicting future\n")
cat("   gold returns. The strongest signal was found at lag",
    best_lag, "months.\n\n")

cat("5. REGRESSION: The lagged regression analysis quantifies the\n")
cat("   predictive relationship. A positive beta coefficient at\n")
cat("   the significant lag(s) would support the hypothesis that\n")
cat("   rising inflation concern predicts future gold returns.\n\n")

cat("6. GRANGER CAUSALITY: This test formally examines whether past\n")
cat("   values of the search index help predict gold returns beyond\n")
cat("   what gold returns' own past values can predict.\n\n")

cat("7. ARIMAX: Comparing the ARIMAX model (with search index as\n")
cat("   external regressor) against a plain ARIMA model reveals\n")
cat("   whether the search index adds genuine forecasting value\n")
cat("   after accounting for gold returns' own time-series dynamics.\n\n")

cat("OVERALL CONCLUSION:\n")
cat("The analysis provides a comprehensive assessment of whether\n")
cat("Google Trends inflation search interest serves as a leading\n")
cat("indicator for gold returns. The evidence from CCF, regression,\n")
cat("Granger causality, and ARIMAX modelling collectively determines\n")
cat("the strength and reliability of this predictive relationship.\n")
cat("The hypothesis of a 10-12 month delayed effect is evaluated\n")
cat("against the empirical lag structure observed in the data.\n\n")

cat("======================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("======================================================================\n")
