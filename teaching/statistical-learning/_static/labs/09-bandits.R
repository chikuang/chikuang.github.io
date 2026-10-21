# Chapter 9: stationary bounded rewards. Base R; no reward stream is shown to a policy.
set.seed(2026102109)
options(width = 100)
means <- c(.25, .40, .55, .65)
K <- length(means)
T <- 1200L
B <- 100L
policies <- c("uniform", "epsilon_0.1", "ucb")
one_run <- function(reward_table, policy) {
  counts <- reward_sum <- numeric(K)
  regret <- total_reward <- optimal_choices <- 0
  for (t in seq_len(nrow(reward_table))) {
    if (t <= K) {
      a <- t
    } else if (policy == "uniform") {
      a <- sample.int(K, 1)
    } else if (policy == "epsilon_0.1") {
      a <- if (runif(1) < .1) sample.int(K, 1) else which.max(reward_sum / counts)
    } else {
      index <- reward_sum / counts + sqrt(2 * log(t) / counts)
      a <- which.max(index)
    }
    y <- reward_table[t, a] # Only this chosen reward is supplied to the update.
    counts[a] <- counts[a] + 1
    reward_sum[a] <- reward_sum[a] + y
    regret <- regret + max(means) - means[a]
    total_reward <- total_reward + y
    optimal_choices <- optimal_choices + (a == which.max(means))
  }
  c(pseudo_regret = regret, reward_rate = total_reward / nrow(reward_table),
    optimal_fraction = optimal_choices / nrow(reward_table))
}
ans <- array(NA_real_, dim = c(B, length(policies), 3),
             dimnames = list(NULL, policies, c("pseudo_regret", "reward_rate", "optimal_fraction")))
for (b in seq_len(B)) {
  # Common potential outcomes couple policies for paired comparisons.
  reward_table <- sapply(means, function(p) rbinom(T, 1, p))
  for (j in seq_along(policies)) ans[b, j, ] <- one_run(reward_table, policies[j])
}
out <- cbind(mean_regret = colMeans(ans[, , 1]),
             regret_mcse = apply(ans[, , 1], 2, sd) / sqrt(B),
             mean_reward = colMeans(ans[, , 2]),
             optimal_fraction = colMeans(ans[, , 3]))
cat("Stationary Bernoulli bandits; fixed horizon, all arms sampled once first:\n")
print(round(out, 4))
paired <- ans[, "ucb", "pseudo_regret"] - ans[, "epsilon_0.1", "pseudo_regret"]
cat("\nPaired UCB minus epsilon-greedy pseudo-regret (negative favours UCB):\n")
print(round(c(mean_difference = mean(paired), mcse = sd(paired) / sqrt(B)), 4))
cat("Finite-horizon rankings can depend on gaps, tuning and horizon.\n")
cat("The simulator knows means for scoring; the policies do not.\n")

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "09-bandits.png"), width = 1000, height = 600, res = 110)
  par(mar = c(5, 4.5, 4, 1))
  boxplot(as.data.frame(ans[, , "pseudo_regret"]),
          names = c("Uniform", "Epsilon = 0.1", "UCB"),
          col = c("#E9D28B", "#A6D4D7", "#DBAACB"),
          ylab = "Cumulative pseudo-regret", main = "Finite-horizon regret across simulations",
          sub = paste(B, "replications; horizon", T, "; stationary Bernoulli rewards"))
  points(seq_along(policies), out[, "mean_regret"], pch = 18, cex = 1.3)
  legend("topright", legend = "Monte Carlo mean", pch = 18, bty = "n")
  invisible(dev.off())
}
