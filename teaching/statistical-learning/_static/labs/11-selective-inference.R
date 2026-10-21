# Chapter 11: exact selected-Gaussian inference. Base R only.
# Run with: Rscript labs/11-selective-inference.R
# Optional plots: set COURSE_FIGURE_DIR to an existing or new directory.
set.seed(2026102111)
options(width = 100)

selected_tests <- function(z) {
  stopifnot(is.matrix(z), ncol(z) >= 2L)
  winner <- max.col(abs(z), ties.method = "first")
  t_obs <- abs(z[cbind(seq_len(nrow(z)), winner)])
  competitors <- abs(z)
  competitors[cbind(seq_len(nrow(z)), winner)] <- -Inf
  threshold <- apply(competitors, 1L, max)
  log_tail <- pnorm(t_obs, lower.tail = FALSE, log.p = TRUE)
  log_denom <- pnorm(threshold, lower.tail = FALSE, log.p = TRUE)
  data.frame(
    winner = winner, statistic = t_obs, threshold = threshold,
    naive = 2 * exp(log_tail),
    selective = pmin(1, exp(log_tail - log_denom))
  )
}

cat("Example 11.1\n")
print(selected_tests(matrix(c(2.4, -1.7, 0.5), nrow = 1)), digits = 5,
      row.names = FALSE)

repetitions <- 10000L
coordinates <- 20L
alpha <- 0.05
z_null <- matrix(rnorm(repetitions * coordinates),
                 nrow = repetitions, ncol = coordinates)
null_result <- selected_tests(z_null)
z_signal <- matrix(rnorm(repetitions * coordinates),
                   nrow = repetitions, ncol = coordinates)
z_signal[, 1L] <- z_signal[, 1L] + 2.5
signal_result <- selected_tests(z_signal)
selected_null <- signal_result$winner != 1L

summarize_p <- function(label, p_values) {
  rejected <- mean(p_values <= alpha)
  data.frame(
    procedure = label, repetitions = length(p_values),
    rejection_rate = rejected,
    monte_carlo_se = sqrt(rejected * (1 - rejected) / length(p_values))
  )
}

summary_table <- rbind(
  summarize_p("Global null: naive selected test", null_result$naive),
  summarize_p("Global null: exact selective test", null_result$selective),
  summarize_p("One signal: selected null coordinates",
              signal_result$selective[selected_null])
)
cat("\nSimulation calibration (alpha = 0.05)\n")
print(summary_table, digits = 4, row.names = FALSE)
cat("\nTheoretical global-null naive rejection rate:",
    format(1 - (1 - alpha)^coordinates, digits = 6), "\n")
cat("Known variance = 1; coordinates independent; no estimated nuisance parameters.\n")
stopifnot(all(is.finite(null_result$selective)),
          all(null_result$selective >= 0 & null_result$selective <= 1))

plot_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(plot_dir, "11-selective-inference.png"),
      width = 1400, height = 650, res = 130)
  old_par <- par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3, 1))
  hist(null_result$naive, breaks = seq(0, 1, by = 0.05), freq = FALSE,
       col = "#2f6ca3", border = "white",
       main = "Naive p-values after selection", xlab = "p-value", ylim = c(0, 14))
  abline(h = 1, col = "#bd4b38", lwd = 2, lty = 2)
  hist(null_result$selective, breaks = seq(0, 1, by = 0.05), freq = FALSE,
       col = "#2a8a7c", border = "white",
       main = "Exact selective p-values", xlab = "p-value", ylim = c(0, 1.5))
  abline(h = 1, col = "#bd4b38", lwd = 2, lty = 2)
  par(old_par)
  invisible(dev.off())
}
