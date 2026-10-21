# Topics of Statistical Learning
# Lab 03: Logistic PU learning under an explicit two-sample design
# Base R only. Run with: Rscript 03-pu-learning.R
set.seed(2026102103)

prior <- 0.30
n_positive <- 250L
n_unlabelled <- 1500L
n_test <- 50000L
lambda <- 0.01
softplus <- function(z) pmax(z, 0) + log1p(exp(-abs(z)))
design <- function(x) cbind(intercept = 1, x = x)
sample_population <- function(n) {
  y <- rbinom(n, 1L, prior)
  list(x = rnorm(n, mean = ifelse(y == 1L, 1, -1)), y = y)
}

x_positive <- rnorm(n_positive, mean = 1)
unlabelled <- sample_population(n_unlabelled)
z_positive <- design(x_positive)
z_unlabelled <- design(unlabelled$x)

pu_objective <- function(beta) {
  gu <- drop(z_unlabelled %*% beta)
  gp <- drop(z_positive %*% beta)
  mean(softplus(gu)) - prior * mean(gp) + lambda * sum(beta^2)
}
pu_gradient <- function(beta) {
  gu <- drop(z_unlabelled %*% beta)
  drop(crossprod(z_unlabelled, plogis(gu))) / n_unlabelled -
    prior * colMeans(z_positive) + 2 * lambda * beta
}
pu_fit <- optim(c(0, 0), pu_objective, pu_gradient, method = "BFGS",
                control = list(reltol = 1e-10, maxit = 1000L))

fit_labelled <- function(z, y) {
  objective <- function(beta) {
    g <- drop(z %*% beta)
    mean(softplus(g) - y * g) + lambda * sum(beta^2)
  }
  gradient <- function(beta) {
    drop(crossprod(z, plogis(drop(z %*% beta)) - y)) / nrow(z) +
      2 * lambda * beta
  }
  optim(c(0, 0), objective, gradient, method = "BFGS",
        control = list(reltol = 1e-10, maxit = 1000L))
}

# Naive benchmark: recorded positives = 1, all marginal U observations = 0.
naive_fit <- fit_labelled(rbind(z_positive, z_unlabelled),
                          c(rep(1, n_positive), rep(0, n_unlabelled)))
# Oracle benchmark only: uses latent labels that a PU learner cannot observe.
oracle_fit <- fit_labelled(z_unlabelled, unlabelled$y)
beta_bayes <- c(qlogis(prior), 2)

test <- sample_population(n_test)
z_test <- design(test$x)
coefficients <- rbind(naive = naive_fit$par, PU = pu_fit$par,
                      labelled_oracle = oracle_fit$par, Bayes = beta_bayes)
assessment <- do.call(rbind, lapply(seq_len(nrow(coefficients)), function(j) {
  score <- drop(z_test %*% coefficients[j, ])
  probability <- plogis(score)
  errors <- as.numeric((score > 0) != test$y)
  data.frame(method = rownames(coefficients)[j],
             intercept = coefficients[j, 1], slope = coefficients[j, 2],
             test_error = mean(errors), test_error_MCSE = sd(errors) / sqrt(n_test),
             test_log_loss = mean(softplus(score) - test$y * score),
             test_Brier = mean((probability - test$y)^2))
}))
cat("Target prior:", prior, "; sample sizes P =", n_positive,
    ", U =", n_unlabelled, "\n")
cat("Positive fraction in the assembled training data:",
    n_positive / (n_positive + n_unlabelled), "(not the target prior)\n\n")
print(assessment, digits = 5, row.names = FALSE)

# Separate single-sample SCAR experiment.
population <- sample_population(50000L)
labelling_probability <- 0.4
s <- population$y == 1L & runif(length(population$y)) < labelling_probability
scar_table <- data.frame(
  quantity = c("Observed-label fraction", "Positive fraction among S=0",
               "Feature mean in full population", "Feature mean among S=0"),
  empirical = c(mean(s), mean(population$y[!s]), mean(population$x),
                mean(population$x[!s])),
  theoretical = c(prior * labelling_probability,
                  prior * (1 - labelling_probability) /
                    (1 - prior * labelling_probability),
                  2 * prior - 1,
                  2 * prior * (1 - labelling_probability) /
                    (1 - prior * labelling_probability) - 1))
cat("\nSeparate SCAR population sample:\n")
print(scar_table, digits = 5, row.names = FALSE)
cat("\nThe S=0 subset is not the full target feature marginal.\n")

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "03-pu-learning.png"), width = 1500,
      height = 700, res = 140, type = "cairo")
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3.2, 1.2))
  grid <- seq(-4, 4, length.out = 401L)
  positive_density <- dnorm(grid, 1)
  marginal_density <- prior * positive_density + (1 - prior) * dnorm(grid, -1)
  matplot(grid, cbind(positive_density, marginal_density), type = "l", lwd = 2,
          lty = c(1, 2), col = c("#2D8063", "#1B4965"), xlab = "Feature x",
          ylab = "Density", main = "Observed sampling distributions")
  legend("topright", c("Representative positives", "Full marginal U"),
         lty = c(1, 2), col = c("#2D8063", "#1B4965"), lwd = 2, bty = "n", cex = 0.8)
  probabilities <- plogis(design(grid) %*% t(coefficients))
  matplot(grid, probabilities, type = "l", lwd = 2, lty = c(1, 1, 2, 3),
          col = c("#B44A3D", "#1B4965", "#2D8063", "#333333"),
          xlab = "Feature x", ylab = "Predicted positive probability",
          main = "Independent models and Bayes reference")
  abline(h = 0.5, lty = 3, col = "#BBBBBB")
  legend("topleft", c("Naive", "PU", "Labelled oracle", "Bayes"),
         lty = c(1, 1, 2, 3), col = c("#B44A3D", "#1B4965", "#2D8063", "#333333"),
         lwd = 2, bty = "n", cex = 0.8)
  invisible(dev.off())
}

stopifnot(pu_fit$convergence == 0L, naive_fit$convergence == 0L,
          oracle_fit$convergence == 0L, all(is.finite(coefficients)))
