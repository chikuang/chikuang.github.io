# Chapter 7: uncertainty with repeated judges, using only base R.
set.seed(2026102107)
options(width = 100)
J <- 40L
m <- 20L
n <- J * m
judge <- rep(seq_len(J), each = m)
relevance <- rnorm(n)
conciseness <- rnorm(n)
order <- sample(c(-1, 1), n, replace = TRUE)
judge_order_bias <- rnorm(J, 0, 1.1)
eta <- 1.1 * relevance + 0.35 * conciseness +
  (0.45 + judge_order_bias[judge]) * order
dat <- data.frame(y = rbinom(n, 1, plogis(eta)), relevance, conciseness, order, judge)
fit <- glm(y ~ 0 + relevance + conciseness + order, family = binomial(), data = dat)
X <- model.matrix(fit)
mu <- fitted(fit)
residual_score <- X * (dat$y - mu)
cluster_score <- rowsum(residual_score, dat$judge, reorder = FALSE)
bread <- solve(crossprod(X, X * as.vector(mu * (1 - mu))))
p <- ncol(X)
correction <- (J / (J - 1)) * ((n - 1) / (n - p))
cluster_cov <- correction * bread %*% crossprod(cluster_score) %*% bread

B <- 150L
boot <- matrix(NA_real_, B, length(coef(fit)))
for (b in seq_len(B)) {
  sampled_judges <- sample.int(J, J, replace = TRUE)
  idx <- unlist(lapply(sampled_judges, function(j) which(dat$judge == j)),
                use.names = FALSE)
  f <- suppressWarnings(glm(y ~ 0 + relevance + conciseness + order,
                            family = binomial(), data = dat[idx, ]))
  if (f$converged && all(is.finite(coef(f)))) boot[b, ] <- coef(f)
}
ok <- complete.cases(boot)
out <- cbind(estimate = coef(fit), iid_se = sqrt(diag(vcov(fit))),
             judge_sandwich_se = sqrt(diag(cluster_cov)),
             judge_bootstrap_se = apply(boot[ok, , drop = FALSE], 2, sd))
cat("Working logistic model with repeated observations from each judge:\n")
print(round(out, 4))
cat("\nBootstrap usable/flagged replicates:", sum(ok), "/", sum(!ok), "\n")
cat("The working coefficients summarize a marginal logistic approximation.\n")
cat("They need not equal the conditional coefficients used to generate the data.\n")
cat("Judge-level resampling assumes independent judges; it is not a two-way bootstrap.\n")

# A finite-action illustration of the KL-regularized preference identity.
ref <- c(.5, .3, .2)
reward <- c(0, .4, 1)
tau <- .7
policy <- ref * exp(reward / tau)
policy <- policy / sum(policy)
pair <- c(3L, 1L)
reward_probability <- plogis(reward[pair[1]] - reward[pair[2]])
ratio_probability <- plogis(tau *
  (log(policy[pair[1]] / ref[pair[1]]) - log(policy[pair[2]] / ref[pair[2]])))
cat("\nFinite-action policy and reference probabilities:\n")
print(round(data.frame(action = 1:3, reference = ref, policy), 4))
cat("Preference probability from rewards / policy log ratios:\n")
print(round(c(reward = reward_probability, log_ratio = ratio_probability), 6))
stopifnot(abs(reward_probability - ratio_probability) < 1e-12)

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "07-contextual-preferences.png"),
      width = 1000, height = 550, res = 110)
  par(mar = c(5, 4.5, 4, 1))
  se_values <- t(out[, c("iid_se", "judge_sandwich_se", "judge_bootstrap_se")])
  barplot(se_values, beside = TRUE, col = c("#B88A00", "#007A87", "#A2006E"),
          ylim = c(0, 1.25 * max(se_values)), ylab = "Estimated standard error",
          main = "Repeated judges change the uncertainty calculation",
          legend.text = c("Independent rows", "Judge sandwich", "Judge bootstrap"),
          args.legend = list(x = "topleft", bty = "n", cex = .85))
  invisible(dev.off())
}
