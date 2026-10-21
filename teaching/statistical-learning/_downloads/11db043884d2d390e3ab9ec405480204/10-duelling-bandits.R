# Chapter 10: winner definitions, cycles, and a transparent duelling policy.
set.seed(2026102110)
options(width = 100)
theta <- c(-.8, -.3, 0, .4, .9)
K <- length(theta)
P <- plogis(outer(theta, theta, "-"))
condorcet <- function(P) {
  which(vapply(seq_len(nrow(P)), function(i) all(P[i, -i] > .5), logical(1)))
}
borda <- function(P) (rowSums(P) - diag(P)) / (ncol(P) - 1)
winner <- condorcet(P)
stopifnot(length(winner) == 1L)
delta <- P[winner, ] - .5
pairs <- t(combn(seq_len(K), 2))
T <- 600L
explore <- 150L # Multiple of the number of pairs, so exploration is balanced.
B <- 150L
out <- matrix(NA_real_, B, 3, dimnames = list(NULL,
       c("balanced_regret", "explore_commit_regret", "selected_correctly")))
for (b in seq_len(B)) {
  wins <- matrix(0, K, K)
  trials <- matrix(0, K, K)
  balanced_regret <- explore_regret <- 0
  selected <- NA_integer_
  for (t in seq_len(T)) {
    pair <- pairs[(t - 1L) %% nrow(pairs) + 1L, ]
    i <- pair[1]; j <- pair[2]
    round_regret <- (delta[i] + delta[j]) / 2
    balanced_regret <- balanced_regret + round_regret
    if (t <= explore) {
      y <- rbinom(1, 1, P[i, j])
      wins[i, j] <- wins[i, j] + y
      wins[j, i] <- wins[j, i] + 1 - y
      trials[i, j] <- trials[i, j] + 1
      trials[j, i] <- trials[j, i] + 1
      explore_regret <- explore_regret + round_regret
      if (t == explore) selected <- which.max(rowSums(wins) / rowSums(trials))
    } else {
      # Commit to (selected, selected): no informative comparison is collected.
      explore_regret <- explore_regret + delta[selected]
    }
  }
  out[b, ] <- c(balanced_regret, explore_regret, selected == winner)
}
cat("Bradley-Terry preference matrix:\n")
print(round(P, 3))
cat("Condorcet winner:", winner, "\n")
cat("Borda scores:\n"); print(round(borda(P), 3))
cat("\nStrong pseudo-regret uses half the sum of the two winner gaps:\n")
print(round(rbind(mean = colMeans(out), mcse = apply(out, 2, sd) / sqrt(B)), 4))
cat("Explore-then-commit is a teaching baseline, not a claim of optimal regret.\n")

cycle <- matrix(.5, 3, 3)
cycle[1, 2] <- .75; cycle[2, 1] <- .25
cycle[2, 3] <- .65; cycle[3, 2] <- .35
cycle[3, 1] <- .60; cycle[1, 3] <- .40
cat("\nA cyclic matrix (row item beats column item):\n")
print(cycle)
cat("Condorcet winners:", if (length(condorcet(cycle))) condorcet(cycle) else "none", "\n")
cat("Borda scores and Borda winner:\n")
print(borda(cycle)); print(which.max(borda(cycle)))
cat("A Condorcet-regret target is unavailable for this cyclic example.\n")

figure_dir <- Sys.getenv("COURSE_FIGURE_DIR", "")
if (nzchar(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(figure_dir, "10-duelling-bandits.png"),
      width = 1100, height = 550, res = 110)
  par(mfrow = c(1, 2), mar = c(5, 4.5, 4, 1))
  boxplot(as.data.frame(out[, 1:2]), names = c("Balanced", "Explore then commit"),
          col = c("#A6D4D7", "#DBAACB"), ylab = "Strong cumulative pseudo-regret",
          main = "A finite-horizon policy comparison", cex.axis = .85)
  mtext(paste("Correct commitment:", round(mean(out[, 3]), 3)), side = 3, cex = .75)
  barplot(borda(cycle), names.arg = c("A", "B", "C"),
          col = c("#007A87", "#B9C7CA", "#B9C7CA"), ylim = c(0, .7),
          ylab = "Uniform-opponent win probability", main = "A cycle can still have a Borda winner")
  mtext("There is no Condorcet winner in this matrix", side = 3, cex = .75)
  invisible(dev.off())
}
