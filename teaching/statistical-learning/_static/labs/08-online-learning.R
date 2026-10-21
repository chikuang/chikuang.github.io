# Chapter 8: full-information Hedge and projected online gradient descent.
set.seed(2026102108)
options(width = 100)
T <- 800L
K <- 3L
losses <- matrix(0, T, K)
for (t in seq_len(T)) {
  error_probability <- if (t <= T / 2) c(.15, .3, .45) else c(.45, .3, .15)
  losses[t, ] <- rbinom(K, 1, error_probability)
}
hedge <- function(L, reset_at = integer()) {
  horizon <- nrow(L)
  eta <- sqrt(8 * log(ncol(L)) / horizon)
  score <- rep(0, ncol(L))
  incurred <- numeric(horizon)
  weights <- matrix(NA_real_, horizon, ncol(L))
  for (t in seq_len(horizon)) {
    if (t %in% reset_at) score[] <- 0
    z <- -eta * score
    w <- exp(z - max(z))
    w <- w / sum(w)
    weights[t, ] <- w
    incurred[t] <- sum(w * L[t, ])
    score <- score + L[t, ] # Update only after the round's loss is incurred.
  }
  list(loss = incurred, weights = weights)
}
ordinary <- hedge(losses)
restart <- hedge(losses, reset_at = T / 2 + 1L)
best_fixed <- min(colSums(losses))
best_two_segments <- min(colSums(losses[1:(T / 2), ])) +
  min(colSums(losses[(T / 2 + 1):T, ]))
summarize <- function(v) c(total_loss = sum(v),
                           fixed_regret = sum(v) - best_fixed,
                           two_segment_regret = sum(v) - best_two_segments)
cat("Full-information losses; mixture loss is expected loss under sampled experts:\n")
print(round(rbind(uniform = summarize(rowMeans(losses)),
                  hedge = summarize(ordinary$loss),
                  known_boundary_restart = summarize(restart$loss)), 3))
cat("\nThe restart policy is given the true change boundary; it is an oracle illustration.\n")
cat("A negative fixed-comparator regret is possible because the learner can change weights.\n")
cat("Hedge weights before rounds 1, 400, 401 and 800:\n")
print(round(ordinary$weights[c(1, 400, 401, 800), ], 4))

# Linear convex losses ell_t(w)=g_t*w on [-1,1]; diameter D=2 and gradient bound G=1.
horizon <- 400L
g <- sample(c(-1, 1), horizon, replace = TRUE, prob = c(.55, .45))
D <- 2
G <- 1
eta <- D / (G * sqrt(horizon))
w <- 0
ogd_loss <- numeric(horizon)
for (t in seq_len(horizon)) {
  ogd_loss[t] <- g[t] * w
  w <- max(-1, min(1, w - eta * g[t]))
}
best_constant_loss <- -abs(sum(g))
regret <- sum(ogd_loss) - best_constant_loss
bound <- D^2 / (2 * eta) + eta * G^2 * horizon / 2
cat("\nProjected OGD, fixed-comparator regret and the deterministic upper bound:\n")
print(c(regret = regret, bound = bound, step_size = eta))
stopifnot(regret <= bound + 1e-10)

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "08-online-learning.png"),
      width = 1100, height = 850, res = 110)
  par(mfrow = c(2, 1), mar = c(4, 4.5, 3, 1))
  colours <- c("#007A87", "#A2006E", "#B88A00")
  matplot(seq_len(T), ordinary$weights, type = "l", lty = 1, lwd = 2,
          col = colours, xlab = "Round", ylab = "Weight before observing losses",
          main = "Ordinary Hedge retains evidence from the first regime")
  abline(v = T / 2, lty = 3)
  legend("topright", legend = paste("Expert", 1:K), col = colours,
         lty = 1, lwd = 2, bty = "n", cex = .8)
  comparator_loss <- losses[, which.min(colSums(losses))]
  loss_differences <- cbind(rowMeans(losses), ordinary$loss, restart$loss) - comparator_loss
  cumulative_difference <- apply(loss_differences, 2, cumsum)
  matplot(seq_len(T), cumulative_difference, type = "l", lty = 1, lwd = 2,
          col = colours, xlab = "Round", ylab = "Cumulative loss difference",
          main = "Comparator: the best fixed expert over the complete horizon")
  abline(h = 0, v = T / 2, lty = 3)
  legend("bottomleft", legend = c("Uniform", "Hedge", "Known-boundary restart"),
         col = colours, lty = 1, lwd = 2, bty = "n", cex = .8)
  invisible(dev.off())
}
