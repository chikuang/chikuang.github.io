# Chapter 12: Gaussian cluster comparisons.
# Required experiment uses base R. The authors' exact method runs when installed.
set.seed(2026102112)
options(width = 145)

n <- 40L
q <- 2L
sigma <- 1
repetitions <- 500L
alpha <- 0.05

wald_p <- function(x, groups, known_sigma = 1) {
  stopifnot(length(unique(groups)) == 2L)
  g1 <- groups == sort(unique(groups))[1L]
  difference <- colMeans(x[g1, , drop = FALSE]) -
    colMeans(x[!g1, , drop = FALSE])
  variance <- known_sigma^2 * (1 / sum(g1) + 1 / sum(!g1))
  pchisq(sum(difference^2) / variance, df = ncol(x), lower.tail = FALSE)
}

samples <- lapply(seq_len(repetitions), function(i) matrix(rnorm(n * q), n, q))
fixed_groups <- rep(1:2, each = n / 2L)
fixed_p <- learned_p <- numeric(repetitions)
for (b in seq_len(repetitions)) {
  x <- samples[[b]]
  tree <- hclust(dist(x)^2, method = "average")
  learned_groups <- cutree(tree, k = 2L)
  fixed_p[b] <- wald_p(x, fixed_groups, sigma)
  learned_p[b] <- wald_p(x, learned_groups, sigma)
}

summarize_p <- function(method, values) {
  rate <- mean(values <= alpha)
  mc_interval <- binom.test(sum(values <= alpha), length(values))$conf.int
  data.frame(method = method, runs = length(values), rejection_rate = rate,
             monte_carlo_se = sqrt(rate * (1 - rate) / length(values)),
             mc_lower = mc_interval[1L], mc_upper = mc_interval[2L])
}

cat("Global null: independent N_2(0, I) observations; fixed K = 2\n")
print(rbind(summarize_p("Fixed groups: ordinary Wald", fixed_p),
            summarize_p("Learned groups: naive Wald", learned_p)),
      row.names = FALSE, digits = 4)
cat("\nExample 12.1: naive p-value for squared standardized contrast 7.2 =",
    format(pchisq(7.2, df = 2, lower.tail = FALSE), digits = 6), "\n")
cat("mc_lower/mc_upper: exact 95% binomial intervals for simulation rejection probabilities.\n")

selective_plot <- NULL
if (requireNamespace("clusterpval", quietly = TRUE)) {
  selective_reps <- min(100L, repetitions)
  selective_p <- rep(NA_real_, selective_reps)
  errors <- character()
  for (b in seq_len(selective_reps)) {
    x <- samples[[b]]
    tree <- hclust(dist(x)^2, method = "average")
    fit <- tryCatch(
      clusterpval::test_hier_clusters_exact(
        x, link = "average", hcl = tree, K = 2L,
        k1 = 1L, k2 = 2L, sig = sigma
      ),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      errors <- c(errors, conditionMessage(fit))
    } else {
      selective_p[b] <- fit$pval
    }
  }
  successful <- is.finite(selective_p) & selective_p >= 0 & selective_p <= 1
  cat("\nAuthors' exact procedure EXECUTED; clusterpval version",
      as.character(utils::packageVersion("clusterpval")), "\n")
  cat("Successful finite p-values:", sum(successful), "of", selective_reps, "\n")
  if (all(successful)) {
    selective_plot <- selective_p
    print(summarize_p("Known-variance selective test", selective_p),
          row.names = FALSE, digits = 4)
  } else {
    cat("Calibration summary withheld because some selective runs failed.\n")
    if (length(errors)) print(unique(errors))
  }
  cat("The selective Monte Carlo check is small; its standard error matters.\n")
} else {
  cat("\nOptional exact selective procedure NOT EXECUTED: clusterpval is not installed.\n")
  cat("Author installation instructions: https://lucylgao.github.io/clusterpval/\n")
  cat("The base-R results above are ordinary/naive tests, not selective p-values.\n")
}

plot_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(plot_dir, "12-hierarchical-clustering.png"),
      width = 1400, height = 650, res = 130)
  old_par <- par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3, 1))
  x <- samples[[1L]]
  groups <- cutree(hclust(dist(x)^2, method = "average"), k = 2L)
  plot(x, col = c("#2f6ca3", "#bd4b38")[groups], pch = 19,
       xlab = "Feature 1", ylab = "Feature 2", main = "One Gaussian cloud; two learned groups")
  plot(ecdf(fixed_p), col = "#2a8a7c", lwd = 2, do.points = FALSE,
       xlab = "p-value threshold", ylab = "Empirical rejection probability",
       main = "Calibration under the global null")
  lines(ecdf(learned_p), col = "#bd4b38", lwd = 2, do.points = FALSE)
  legend_labels <- c("Fixed groups", "Learned: naive")
  legend_colors <- c("#2a8a7c", "#bd4b38")
  if (!is.null(selective_plot)) {
    lines(ecdf(selective_plot), col = "#2f6ca3", lwd = 2, do.points = FALSE)
    legend_labels <- c(legend_labels, "Learned: selective")
    legend_colors <- c(legend_colors, "#2f6ca3")
  }
  abline(0, 1, lty = 2, col = "grey40")
  legend("bottomright", legend_labels,
         col = legend_colors, lty = 1, lwd = 2, bty = "n")
  par(old_par)
  invisible(dev.off())
}
