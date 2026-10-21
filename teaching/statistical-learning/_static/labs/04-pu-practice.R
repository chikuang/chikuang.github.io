# Topics of Statistical Learning
# Lab 04: nnPU bias, variability, and flexible-model fitting
# Base R only. Run with: Rscript 04-pu-practice.R
set.seed(2026102104)

prior <- 0.4
positive_mean <- 1.5
negative_mean <- -1.5
softplus <- function(z) pmax(z, 0) + log1p(exp(-abs(z)))
sample_population <- function(n) {
  y <- rbinom(n, 1L, prior)
  list(x = rnorm(n, ifelse(y == 1L, positive_mean, negative_mean)), y = y)
}
bayes_score <- function(x) qlogis(prior) + 3 * x
true_fixed_risk <- prior * integrate(function(x) {
  softplus(-bayes_score(x)) * dnorm(x, positive_mean)
}, -Inf, Inf)$value + (1 - prior) * integrate(function(x) {
  softplus(bayes_score(x)) * dnorm(x, negative_mean)
}, -Inf, Inf)$value

# Part A: the score is fixed before every sample is generated.
n_rep <- 1000L
fixed_estimates <- matrix(NA_real_, n_rep, 3L,
                          dimnames = list(NULL, c("uPU", "nnPU", "negative_part")))
for (b in seq_len(n_rep)) {
  gp <- bayes_score(rnorm(18L, positive_mean))
  gu <- bayes_score(sample_population(60L)$x)
  positive_part <- prior * mean(softplus(-gp))
  negative_part <- mean(softplus(gu)) - prior * mean(softplus(gp))
  fixed_estimates[b, ] <- c(positive_part + negative_part,
                            positive_part + max(0, negative_part), negative_part)
}
fixed_summary <- do.call(rbind, lapply(1:2, function(j) {
  value <- fixed_estimates[, j]
  data.frame(estimator = colnames(fixed_estimates)[j], mean = mean(value),
             bias = mean(value) - true_fixed_risk, SD = sd(value),
             bias_MCSE = sd(value) / sqrt(n_rep),
             MSE = mean((value - true_fixed_risk)^2))
}))
cat("PART A: fixed-rule logistic risk\n")
cat("Numerically integrated reference risk:", true_fixed_risk, "\n")
print(fixed_summary, digits = 5, row.names = FALSE)
cat("Fraction with negative estimated negative-class contribution:",
    mean(fixed_estimates[, "negative_part"] < 0), "\n")

# Part B: fit scores using only P and marginal U data.
x_positive <- rnorm(30L, positive_mean)
unlabelled <- sample_population(300L)
centres <- seq(-3.5, 4, length.out = 18L)
width <- 0.30
basis <- function(x) cbind(1, exp(-outer(x, centres, "-")^2 / (2 * width^2)))
zp <- basis(x_positive)
zu <- basis(unlabelled$x)
ridge <- 1e-4

components <- function(beta) {
  gp <- drop(zp %*% beta)
  gu <- drop(zu %*% beta)
  c(positive = prior * mean(softplus(-gp)),
    negative = mean(softplus(gu)) - prior * mean(softplus(gp)))
}
fit_objective <- function(nonnegative) {
  objective <- function(beta) {
    parts <- components(beta)
    parts[1] + if (nonnegative) max(0, parts[2]) else parts[2]
  }
  penalized <- function(beta) objective(beta) + ridge * sum(beta^2)
  gradient <- function(beta) {
    pp <- plogis(drop(zp %*% beta))
    pu <- plogis(drop(zu %*% beta))
    grad_positive <- prior * drop(crossprod(zp, pp - 1)) / nrow(zp)
    grad_negative <- drop(crossprod(zu, pu)) / nrow(zu) -
      prior * drop(crossprod(zp, pp)) / nrow(zp)
    use_negative <- !nonnegative || components(beta)[2] > 0
    grad_positive + if (use_negative) grad_negative else rep(0, ncol(zp))
  }
  penalized_gradient <- function(beta) gradient(beta) + 2 * ridge * beta
  starts <- list(rep(0, ncol(zp)), rep(0.1, ncol(zp)), rep(-0.1, ncol(zp)))
  fits <- lapply(starts, function(start) {
    optim(start, penalized, penalized_gradient, method = "L-BFGS-B",
          lower = -12, upper = 12,
          control = list(maxit = 2000L, factr = 1e7, pgtol = 1e-7))
  })
  fits[[which.min(vapply(fits, function(fit) fit$value, numeric(1L)))]]
}

fits <- list(uPU = fit_objective(FALSE), nnPU = fit_objective(TRUE))
test <- sample_population(30000L)
zt <- basis(test$x)
fitted_summary <- do.call(rbind, lapply(names(fits), function(method) {
  fit <- fits[[method]]
  parts <- components(fit$par)
  score <- drop(zt %*% fit$par)
  errors <- as.numeric((score > 0) != test$y)
  data.frame(method = method, training_positive_part = parts[1],
             training_negative_part = parts[2],
             test_log_loss = mean(softplus(score) - test$y * score),
             test_error = mean(errors), test_error_MCSE = sd(errors) / sqrt(length(errors)),
             optimizer_code = fit$convergence)
}))
cat("\nPART B: flexible fitted rules, assessed on independent labelled data\n")
print(fitted_summary, digits = 5, row.names = FALSE)
cat("\nOptimizer code 0 indicates numerical convergence, not global optimality.\n")
cat("The score bounds, regularizer, and three starts are shared by both methods.\n")
cat("A single fitted-rule comparison is illustrative; repeat training for a method comparison.\n")

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "04-pu-practice.png"), width = 1500,
      height = 700, res = 140, type = "cairo")
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3.2, 1.2))
  boxplot(fixed_estimates[, "uPU"], fixed_estimates[, "nnPU"],
          names = c("Unbiased PU", "Non-negative PU"), col = c("#AFC3D4", "#BBDACD"),
          ylab = "Estimated logistic risk", main = "The same fixed score, repeated samples")
  abline(h = true_fixed_risk, col = "#B44A3D", lty = 2, lwd = 2)
  legend("topleft", "True fixed-score risk", col = "#B44A3D", lty = 2, bty = "n", cex = 0.8)
  grid <- seq(-4, 4, length.out = 401L)
  predictions <- cbind(plogis(drop(basis(grid) %*% fits$uPU$par)),
                        plogis(drop(basis(grid) %*% fits$nnPU$par)),
                        plogis(bayes_score(grid)))
  matplot(grid, predictions, type = "l", lwd = 2, lty = c(1, 1, 2),
          col = c("#1B4965", "#2D8063", "#B44A3D"), xlab = "Feature x",
          ylab = "Predicted positive probability", main = "Flexible scores fitted to one sample")
  legend("topleft", c("Unbiased PU", "Non-negative PU", "Bayes reference"),
         col = c("#1B4965", "#2D8063", "#B44A3D"), lty = c(1, 1, 2),
         lwd = 2, bty = "n", cex = 0.8)
  invisible(dev.off())
}

stopifnot(all(is.finite(fixed_estimates)),
          all(fixed_estimates[, "nnPU"] >= fixed_estimates[, "uPU"] - 1e-12),
          all(vapply(fits, function(fit) is.finite(fit$value), logical(1L))))
