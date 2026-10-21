# Chapter 13: transparent Lloyd algorithm and optional authors' inference.
set.seed(2026102113)
options(width = 145)

lloyd_trace <- function(x, initial_indices = c(1L, nrow(x)), max_iter = 30L) {
  centers <- x[initial_indices, , drop = FALSE]
  previous <- rep(0L, nrow(x))
  history <- list()
  objective <- numeric()
  for (iteration in seq_len(max_iter)) {
    distances <- sapply(seq_len(nrow(centers)), function(k)
      rowSums(sweep(x, 2L, centers[k, ], "-")^2))
    groups <- max.col(-distances, ties.method = "first")
    if (length(unique(groups)) != nrow(centers))
      stop("Empty cluster: this demonstration does not silently repair it.")
    centers <- t(vapply(seq_len(nrow(centers)), function(k)
      colMeans(x[groups == k, , drop = FALSE]), numeric(ncol(x))))
    history[[iteration]] <- groups
    objective[iteration] <- sum((x - centers[groups, , drop = FALSE])^2)
    if (identical(groups, previous)) break
    previous <- groups
  }
  list(groups = groups, centers = centers, history = history,
       objective = objective,
       converged = length(history) >= 2L &&
         identical(history[[length(history)]], history[[length(history) - 1L]]))
}

pair_agreement <- function(a, b) {
  keep <- upper.tri(matrix(FALSE, length(a), length(a)))
  mean((outer(a, a, "==") == outer(b, b, "=="))[keep])
}

wald_p <- function(x, groups, sigma = 1) {
  g <- groups == 1L
  d <- colMeans(x[g, , drop = FALSE]) - colMeans(x[!g, , drop = FALSE])
  pchisq(sum(d^2) / (sigma^2 * (1 / sum(g) + 1 / sum(!g))),
         df = ncol(x), lower.tail = FALSE)
}

n <- 40L
q <- 2L
repetitions <- 400L
samples <- lapply(seq_len(repetitions), function(i) matrix(rnorm(n * q), n, q))
x <- samples[[2L]] # Fixed illustration; the calibration study uses every replicate.
first <- lloyd_trace(x, c(1L, n))
second <- lloyd_trace(x, c(10L, 30L))
cat("Lloyd trace for one Gaussian cloud\n")
print(data.frame(iteration = seq_along(first$objective),
                 within_sum_squares = first$objective),
      row.names = FALSE, digits = 5)
cat("Objective never increases:",
    all(diff(first$objective) <= 1e-8), "\n")
cat("Converged:", first$converged, "\n")
cat("Pairwise co-membership agreement across two fixed initializations:",
    format(pair_agreement(first$groups, second$groups), digits = 4), "\n")

naive_p <- vapply(samples, function(dat) {
  fit <- lloyd_trace(dat)
  wald_p(dat, fit$groups)
}, numeric(1))

summarize_p <- function(method, p) {
  rate <- mean(p <= 0.05)
  mc_interval <- binom.test(sum(p <= 0.05), length(p))$conf.int
  data.frame(method = method, runs = length(p), rejection_rate = rate,
             monte_carlo_se = sqrt(rate * (1 - rate) / length(p)),
             mc_lower = mc_interval[1L], mc_upper = mc_interval[2L])
}
cat("\nBase-R fixed-initialization Lloyd: global-null diagnostic\n")
print(summarize_p("Naive Wald after Lloyd clustering", naive_p),
      row.names = FALSE, digits = 4)
cat("This diagnostic is not a selective-inference procedure.\n")
cat("mc_lower/mc_upper are exact 95% binomial Monte Carlo intervals.\n")
cat("An observed rate of 1 has plug-in MCSE 0 but still has binomial uncertainty.\n")

if (requireNamespace("KmeansInference", quietly = TRUE)) {
  exact_reps <- 60L
  p_selective <- p_naive_author <- rep(NA_real_, exact_reps)
  errors <- character()
  for (b in seq_len(exact_reps)) {
    fit <- tryCatch(
      KmeansInference::kmeans_inference(
        samples[[b]], 2L, 1L, 2L, sig = 1,
        iter.max = 30L, seed = 2026L, verbose = FALSE
      ),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      errors <- c(errors, conditionMessage(fit))
    } else {
      tab <- summary(fit)
      p_selective[b] <- tab$p_selective[1L]
      p_naive_author[b] <- tab$p_naive[1L]
    }
  }
  valid <- is.finite(p_selective) & p_selective >= 0 & p_selective <= 1 &
    is.finite(p_naive_author)
  cat("\nAuthors' path-conditioned method EXECUTED; KmeansInference version",
      as.character(utils::packageVersion("KmeansInference")), "\n")
  cat("Successful finite p-values:", sum(valid), "of", exact_reps, "\n")
  if (all(valid)) {
    print(rbind(
      summarize_p("Authors' algorithm: naive", p_naive_author),
      summarize_p("Authors' algorithm: selective", p_selective)
    ), row.names = FALSE, digits = 4)
  } else {
    cat("Calibration summary withheld because some runs failed.\n")
    if (length(errors)) print(unique(errors))
  }
  cat("The paired comparison uses the same authors' algorithm and seed.\n")
  cat("Sixty repetitions are a demonstration, not a high-precision size study.\n")
} else {
  cat("\nOptional selective procedure NOT EXECUTED: KmeansInference is not installed.\n")
  cat("Author instructions: https://yiqunchen.github.io/KmeansInference/\n")
}

plot_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(plot_dir, "13-kmeans-inference.png"),
      width = 1400, height = 650, res = 130)
  old_par <- par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3, 1))
  plot(x, col = c("#2f6ca3", "#bd4b38")[first$groups], pch = 19,
       xlab = "Feature 1", ylab = "Feature 2",
       main = "Lloyd clusters in a Gaussian cloud")
  points(first$centers, pch = 4, cex = 2, lwd = 3)
  plot(seq_along(first$objective), first$objective, type = "b", pch = 19,
       col = "#2a8a7c", xlab = "Iteration", ylab = "Within-cluster sum of squares",
       main = "Observed optimization path")
  par(old_par)
  invisible(dev.off())
}
