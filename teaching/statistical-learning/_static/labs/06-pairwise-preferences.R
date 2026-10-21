# Chapter 6: fixed comparison schedules and Bradley-Terry information.
# Base R only. Repeated runs with the same R version and seed reproduce the draws.
set.seed(2026102106)
options(width = 100)
K <- 6L
theta <- c(-0.9, -0.5, -0.1, 0.1, 0.5, 0.9)
all_pairs <- t(combn(seq_len(K), 2L))
chain_pairs <- cbind(seq_len(K - 1L), 2:K)
disconnected_pairs <- rbind(t(combn(1:3, 2)), t(combn(4:6, 2)))
comparison_matrix <- function(pairs) {
  X <- matrix(0, nrow(pairs), K)
  X[cbind(seq_len(nrow(pairs)), pairs[, 1])] <- 1
  X[cbind(seq_len(nrow(pairs)), pairs[, 2])] <- -1
  X
}
cat("Comparison-matrix ranks; a connected K-item design needs rank K-1:\n")
print(c(complete = qr(comparison_matrix(all_pairs))$rank,
        chain = qr(comparison_matrix(chain_pairs))$rank,
        disconnected = qr(comparison_matrix(disconnected_pairs))$rank))
budget <- 600L
a <- c(1, rep(0, K - 2L)) # theta_1 - theta_K after fixing theta_K=0.
local_se <- function(pairs) {
  X <- comparison_matrix(pairs)[, -K, drop = FALSE]
  n <- rep(budget / nrow(pairs), nrow(pairs))
  p <- plogis(comparison_matrix(pairs) %*% theta)
  information <- crossprod(X, X * as.vector(n * p * (1 - p)))
  sqrt(drop(crossprod(a, solve(information, a))))
}
cat("\nLocal standard error for theta_1 - theta_6 at the generating strengths:\n")
print(round(c(complete = local_se(all_pairs), chain = local_se(chain_pairs)), 4))

run_design <- function(pairs, B = 150L) {
  X <- comparison_matrix(pairs)[, -K, drop = FALSE]
  n <- rep(as.integer(budget / nrow(pairs)), nrow(pairs))
  p <- as.vector(plogis(comparison_matrix(pairs) %*% theta))
  target <- theta[1] - theta[K]
  estimates <- ses <- rep(NA_real_, B)
  for (b in seq_len(B)) {
    wins <- rbinom(length(n), n, p)
    fit <- suppressWarnings(glm.fit(X, cbind(wins, n - wins),
                                    family = binomial(), intercept = FALSE))
    if (!fit$converged || any(!is.finite(fit$coefficients)) ||
        any(abs(fit$coefficients) > 20)) next
    phat <- fit$fitted.values
    information <- crossprod(X, X * as.vector(n * phat * (1 - phat)))
    inv <- tryCatch(solve(information), error = function(e) NULL)
    if (is.null(inv)) next
    estimates[b] <- fit$coefficients[1]
    ses[b] <- sqrt(inv[1, 1])
  }
  ok <- is.finite(estimates) & is.finite(ses)
  covered <- abs(estimates[ok] - target) <= qnorm(.975) * ses[ok]
  c(usable = sum(ok), flagged = sum(!ok),
    bias = mean(estimates[ok]) - target,
    rmse = sqrt(mean((estimates[ok] - target)^2)),
    mean_se = mean(ses[ok]), coverage = mean(covered),
    coverage_mcse = sqrt(mean(covered) * (1 - mean(covered)) / sum(ok)))
}
results <- rbind(complete = run_design(all_pairs), chain = run_design(chain_pairs))
cat("\nFixed-design simulation; coverage is estimated, not guaranteed:\n")
print(round(results, 4))
cat("\nThe disconnected design is not fitted: the endpoint contrast is unidentified.\n")
cat("Flagged fits are disclosed; reported simulation summaries use the usable fits.\n")

# A connected comparison graph can still have a separated win pattern.
win_graph <- matrix(FALSE, 3, 3)
win_graph[1, 2] <- win_graph[2, 3] <- TRUE
reach <- win_graph
diag(reach) <- TRUE
for (k in 1:3) reach <- reach | outer(reach[, k], reach[k, ], "&")
cat("\nA beats B, B beats C: undirected graph connected, directed wins strongly connected?\n")
print(all(reach))

# Optional static figure; the course build supplies this directory.
figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "06-pairwise-preferences.png"),
      width = 1100, height = 500, res = 110)
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3.5, 1))
  values <- t(results[, c("rmse", "mean_se")])
  barplot(values, beside = TRUE, col = c("#007A87", "#A2006E"),
          ylim = c(0, 1.25 * max(values)), ylab = "Log-strength scale",
          main = "Precision of the endpoint contrast",
          legend.text = c("Empirical RMSE", "Mean estimated SE"),
          args.legend = list(x = "topleft", bty = "n", cex = .8))
  coverage <- results[, "coverage"]
  mcse <- results[, "coverage_mcse"]
  limits <- range(c(.95, pmax(0, coverage - 2 * mcse), pmin(1, coverage + 2 * mcse)))
  plot(1:2, coverage, ylim = limits + c(-.01, .01), xlim = c(.6, 2.4),
       xaxt = "n", xlab = "Comparison schedule", ylab = "Estimated coverage",
       pch = 19, col = "#007A87", main = "Nominal 95% Wald intervals")
  axis(1, at = 1:2, labels = rownames(results))
  abline(h = .95, lty = 2, col = "#A2006E")
  arrows(1:2, pmax(0, coverage - 2 * mcse), 1:2, pmin(1, coverage + 2 * mcse),
         angle = 90, code = 3, length = .05, col = "#007A87")
  mtext("Bars show approximately two Monte Carlo SEs", side = 3, cex = .7)
  invisible(dev.off())
}
