# Topics of Statistical Learning
# Lab 05: Two-unlabelled-mixture risk estimation and instability
# Base R only. Run with: Rscript 05-uu-learning.R
set.seed(2026102105)

target_prior <- 0.5
a <- 0.7
b_values <- c(0.3, 0.5, 0.65, 0.69)
n_each <- 500L
n_rep <- 1000L
threshold <- 0.2
positive_mean <- 1
negative_mean <- -1

sample_mixture <- function(n, proportion, positive_location = positive_mean) {
  # Labels are internal to simulation and never passed to the risk estimator.
  latent_y <- rbinom(n, 1L, proportion)
  rnorm(n, mean = ifelse(latent_y == 1L, positive_location, negative_mean))
}
mixture_cdf <- function(proportion, positive_location = positive_mean) {
  proportion * pnorm(threshold, positive_location) +
    (1 - proportion) * pnorm(threshold, negative_mean)
}
risk_from_features <- function(x1, x2, a, b, target_prior) {
  if (abs(a - b) < 1e-12) stop("Distinct known mixture priors are required.")
  loss1_positive <- mean(x1 <= threshold)
  loss2_positive <- mean(x2 <= threshold)
  loss1_negative <- 1 - loss1_positive
  loss2_negative <- 1 - loss2_positive
  target_prior * ((1 - b) * loss1_positive - (1 - a) * loss2_positive) / (a - b) +
    (1 - target_prior) * (a * loss2_negative - b * loss1_negative) / (a - b)
}

true_risk <- target_prior * pnorm(threshold, positive_mean) +
  (1 - target_prior) * (1 - pnorm(threshold, negative_mean))
estimates <- matrix(NA_real_, n_rep, length(b_values))
summary_table <- do.call(rbind, lapply(seq_along(b_values), function(j) {
  b <- b_values[j]
  for (r in seq_len(n_rep)) {
    estimates[r, j] <<- risk_from_features(sample_mixture(n_each, a),
                                           sample_mixture(n_each, b),
                                           a, b, target_prior)
  }
  value <- estimates[, j]
  f1 <- mixture_cdf(a)
  f2 <- mixture_cdf(b)
  coefficient1 <- (target_prior + b - 2 * target_prior * b) / (a - b)
  coefficient2 <- (target_prior + a - 2 * target_prior * a) / (a - b)
  exact_sd <- sqrt(coefficient1^2 * f1 * (1 - f1) / n_each +
                    coefficient2^2 * f2 * (1 - f2) / n_each)
  data.frame(a = a, b = b, prior_gap = a - b, mean_risk = mean(value),
             bias = mean(value) - true_risk, empirical_SD = sd(value),
             exact_SD = exact_sd, mean_MCSE = sd(value) / sqrt(n_rep),
             fraction_outside_0_1 = mean(value < 0 | value > 1))
}))

cat("Target prior:", target_prior, "; threshold:", threshold, "\n")
cat("Independent samples per replication:", n_each, "from each mixture\n")
cat("Replications:", n_rep, "; analytic target risk:", true_risk, "\n\n")
print(summary_table, digits = 5, row.names = FALSE)

# Misspecification: only the positive class in mixture 2 is shifted.
b_shift <- 0.3
shifted_positive_mean <- 1.75
f1 <- mixture_cdf(a)
f2_shift <- mixture_cdf(b_shift, shifted_positive_mean)
correction_limit <- target_prior * ((1 - b_shift) * f1 -
                                      (1 - a) * f2_shift) / (a - b_shift) +
  (1 - target_prior) * (a * (1 - f2_shift) -
                         b_shift * (1 - f1)) / (a - b_shift)
shift_estimates <- replicate(300L, {
  risk_from_features(sample_mixture(2000L, a),
                     sample_mixture(2000L, b_shift, shifted_positive_mean),
                     a, b_shift, target_prior)
})
cat("\nCommon-class-conditional assumption violated in mixture 2:\n")
print(data.frame(target_risk = true_risk, misspecified_limit = correction_limit,
                 simulated_mean = mean(shift_estimates),
                 mean_MCSE = sd(shift_estimates) / sqrt(length(shift_estimates)),
                 asymptotic_bias = correction_limit - true_risk),
      digits = 5, row.names = FALSE)
cat("\nThe target population still uses positive mean 1 and negative mean -1.\n")
cat("Larger samples reduce variability around the misspecified limit, not its bias.\n")

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "05-uu-learning.png"), width = 1500,
      height = 700, res = 140, type = "cairo")
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3.2, 1.2))
  plot(summary_table$prior_gap, summary_table$exact_SD, type = "l", log = "xy",
       lwd = 2, col = "#1B4965", xlab = "Mixture-prior separation |a - b| (log scale)",
       ylab = "Risk estimator SD (log scale)", main = "Small separation reduces precision")
  points(summary_table$prior_gap, summary_table$empirical_SD,
         pch = 19, col = "#2D8063", cex = 1.1)
  legend("topright", c("Exact fixed-rule SD", "Simulation SD"),
         col = c("#1B4965", "#2D8063"), lty = c(1, NA), pch = c(NA, 19),
         bty = "n", cex = 0.8)
  hist(shift_estimates, breaks = 24L, col = "#DDE9F1", border = "white",
       xlim = range(c(true_risk, shift_estimates)), xlab = "Estimated target risk",
       main = "Violation of shared class-conditionals")
  abline(v = true_risk, lwd = 2, lty = 2, col = "#B44A3D")
  abline(v = correction_limit, lwd = 2, col = "#1B4965")
  legend("topleft", c("True target risk", "Misspecified limit"),
         col = c("#B44A3D", "#1B4965"), lty = c(2, 1), lwd = 2, bty = "n", cex = 0.8)
  invisible(dev.off())
}

stopifnot(all(is.finite(estimates)),
          all(is.finite(shift_estimates)),
          all(summary_table$exact_SD > 0))
