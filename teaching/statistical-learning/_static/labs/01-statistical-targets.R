# Topics of Statistical Learning
# Lab 01: Training error and future prediction risk
# Base R only. Run with: Rscript 01-statistical-targets.R
set.seed(2026102101)

n_train <- 50L
n_rep <- 200L
degrees <- c(1L, 2L, 8L)
noise_sd <- 0.7
mean_function <- function(x) 1 + 2 * x + 0.5 * x^2
evaluation_x <- -1 + (seq_len(801L) - 0.5) * 2 / 801
evaluation_data <- data.frame(x = evaluation_x)
evaluation_mean <- mean_function(evaluation_x)

results <- array(NA_real_, dim = c(n_rep, length(degrees), 2L),
                 dimnames = list(NULL, paste0("degree_", degrees),
                                 c("training_MSE", "target_risk")))
for (b in seq_len(n_rep)) {
  training <- data.frame(x = runif(n_train, -1, 1))
  training$y <- mean_function(training$x) + rnorm(n_train, sd = noise_sd)
  for (j in seq_along(degrees)) {
    degree <- degrees[j]
    fit <- lm(y ~ poly(x, degree = degree), data = training)
    train_prediction <- fitted(fit)
    test_prediction <- predict(fit, newdata = evaluation_data)
    results[b, j, "training_MSE"] <- mean((training$y - train_prediction)^2)
    # Midpoint approximation to E_X[(g_hat(X) - m(X))^2] + sigma^2.
    results[b, j, "target_risk"] <-
      mean((test_prediction - evaluation_mean)^2) + noise_sd^2
  }
}

summary_table <- do.call(rbind, lapply(seq_along(degrees), function(j) {
  train <- results[, j, "training_MSE"]
  risk <- results[, j, "target_risk"]
  difference <- risk - train
  data.frame(degree = degrees[j], mean_training_MSE = mean(train),
             mean_target_risk = mean(risk), risk_MCSE = sd(risk) / sqrt(n_rep),
             mean_optimism = mean(difference),
             optimism_MCSE = sd(difference) / sqrt(n_rep))
}))

cat("Independent training replications:", n_rep, "\n")
cat("Training observations per replication:", n_train, "\n")
cat("Irreducible variance:", noise_sd^2, "\n")
cat("Target: X uniform on [-1, 1]; squared prediction error.\n\n")
print(summary_table, digits = 5, row.names = FALSE)
cat("\nTarget risks use the known simulation mean and a midpoint integration grid.\n")
cat("MCSE describes uncertainty across repeated training datasets.\n")
cat("It is not a confidence interval for an individual fitted model's risk.\n")

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "01-statistical-targets.png"), width = 1500,
      height = 700, res = 140, type = "cairo")
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3.2, 1.2))
  means <- rbind(summary_table$mean_training_MSE, summary_table$mean_target_risk)
  positions <- barplot(means, beside = TRUE, col = c("#AFC3D4", "#1B4965"),
                       names.arg = paste("Degree", degrees), ylim = c(0, 1.1),
                       ylab = "Mean squared loss", main = "Training error and target risk")
  arrows(positions[2, ], summary_table$mean_target_risk - summary_table$risk_MCSE,
         positions[2, ], summary_table$mean_target_risk + summary_table$risk_MCSE,
         angle = 90, code = 3, length = 0.04)
  abline(h = noise_sd^2, lty = 2, col = "#B44A3D")
  legend("topleft", c("Training error", "Target risk (+/- 1 MCSE)", "Noise variance"),
         fill = c("#AFC3D4", "#1B4965", NA), border = NA,
         lty = c(NA, NA, 2), col = c(NA, NA, "#B44A3D"), bty = "n", cex = 0.8)
  boxplot(lapply(seq_along(degrees), function(j) results[, j, "target_risk"]),
          names = paste("Degree", degrees), log = "y", col = "#E5EDF3",
          ylab = "Target risk (log scale)", main = "Variation across training samples")
  abline(h = noise_sd^2, lty = 2, col = "#B44A3D")
  invisible(dev.off())
}

stopifnot(all(is.finite(results)),
          all(results[, , "target_risk"] >= noise_sd^2 - 1e-12))
