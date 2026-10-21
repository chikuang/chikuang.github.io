# Topics of Statistical Learning
# Lab 02: Class-conditional noise and corrected empirical risk
# Base R only. Run with: Rscript 02-weak-supervision.R
set.seed(2026102102)

n_train <- 2000L
n_test <- 100000L
rho_positive <- 0.30
rho_negative <- 0.10
d <- 1 - rho_positive - rho_negative
stopifnot(d > 0)
transition <- rbind(c(1 - rho_positive, rho_positive),
                    c(rho_negative, 1 - rho_negative))
dimnames(transition) <- list(true = c("positive", "negative"),
                            recorded = c("positive", "negative"))

x <- rnorm(n_train)
y <- rbinom(n_train, 1L, plogis(1.2 * x))
flip_probability <- ifelse(y == 1L, rho_positive, rho_negative)
flipped <- runif(n_train) < flip_probability
y_recorded <- ifelse(flipped, 1L - y, y)

thresholds <- seq(-1.5, 1.5, length.out = 151L)
curves <- t(vapply(thresholds, function(threshold) {
  prediction <- as.integer(x > threshold)
  loss_positive <- 1 - prediction
  loss_negative <- prediction
  corrected_positive <- ((1 - rho_negative) * loss_positive -
                           rho_positive * loss_negative) / d
  corrected_negative <- ((1 - rho_positive) * loss_negative -
                           rho_negative * loss_positive) / d
  corrected <- ifelse(y_recorded == 1L, corrected_positive, corrected_negative)
  c(noisy_risk = mean(prediction != y_recorded),
    corrected_risk = mean(corrected),
    oracle_clean_risk = mean(prediction != y))
}, numeric(3L)))

selected <- apply(curves, 2L, which.min)
x_test <- rnorm(n_test)
y_test <- rbinom(n_test, 1L, plogis(1.2 * x_test))
assessment <- do.call(rbind, lapply(seq_along(selected), function(j) {
  threshold <- thresholds[selected[j]]
  errors <- as.numeric((x_test > threshold) != y_test)
  data.frame(method = colnames(curves)[j], threshold = threshold,
             held_out_error = mean(errors),
             held_out_MCSE = sd(errors) / sqrt(n_test))
}))

cat("Known class-conditional transition matrix:\n")
print(transition)
cat("\nThresholds selected using each training criterion:\n")
print(assessment, digits = 5, row.names = FALSE)
cat("\nThe oracle clean criterion uses hidden training labels only as a benchmark.\n")
cat("Held-out labels are used for assessment, never threshold selection.\n")

# Identity check for the rule that always predicts positive.
clean_losses <- c(0, 1)
corrected_losses <- drop(solve(transition, clean_losses))
recovered <- drop(transition %*% corrected_losses)
cat("\nCorrected losses for a positive decision:\n")
print(data.frame(recorded_label = c("positive", "negative"),
                 corrected_loss = corrected_losses), row.names = FALSE)
cat("Recovered expected losses conditional on the true class:\n")
print(recovered)

# Three-class example: uniform complementary labels and prediction of class A.
classes <- c("A", "B", "C")
complement_result <- vapply(classes, function(true_class) {
  possible_complements <- setdiff(classes, true_class)
  mean(2 * (possible_complements == "A"))
}, numeric(1L))
cat("\nComplementary-label corrected risk for a prediction of A:\n")
print(data.frame(true_class = classes,
                 expected_corrected_loss = complement_result), row.names = FALSE)

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "02-weak-supervision.png"), width = 1500,
      height = 700, res = 140, type = "cairo")
  par(mfrow = c(1, 2), mar = c(5.2, 4.5, 4.0, 1.2))
  colours <- c("#B44A3D", "#1B4965", "#2D8063")
  matplot(thresholds, curves, type = "l", lty = c(1, 1, 2), lwd = 2,
          col = colours, xlab = "Classification threshold", ylab = "Empirical risk",
          main = "Different recorded and target risks")
  legend("top", c("Noisy labels", "Corrected target risk", "Clean-label oracle"),
         col = colours, lty = c(1, 1, 2), lwd = 2, bty = "n", cex = 0.8)
  positions <- barplot(assessment$held_out_error, names.arg = c("Naive", "Corrected", "Oracle"),
                       col = colours, ylim = c(0, 0.45), ylab = "Clean-label error rate",
                       main = "Independent final assessment")
  arrows(positions, assessment$held_out_error - 2 * assessment$held_out_MCSE,
         positions, assessment$held_out_error + 2 * assessment$held_out_MCSE,
         angle = 90, code = 3, length = 0.05)
  mtext("Error bars show +/- 2 Monte Carlo SE", side = 3, line = 0.2, cex = 0.75)
  invisible(dev.off())
}

stopifnot(max(abs(recovered - clean_losses)) < 1e-12,
          identical(as.numeric(complement_result), c(0, 1, 1)),
          all(is.finite(curves)))
