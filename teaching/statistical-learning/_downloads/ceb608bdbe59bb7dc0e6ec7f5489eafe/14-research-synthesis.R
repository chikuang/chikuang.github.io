# Chapter 14: Gaussian data thinning and sensitivity. Base R only.
set.seed(2026102114)
options(width = 145)

n <- 60L
q <- 2L
B <- 400L
sigma <- 1
taus <- c(0.2, 0.5, 0.8)
reference_groups <- rep(1:2, each = n / 2L)
noise <- lapply(seq_len(B), function(b) matrix(rnorm(n * q), n, q))
randomization <- lapply(seq_len(B), function(b) matrix(rnorm(n * q), n, q))

pair_agreement <- function(a, b) {
  keep <- upper.tri(matrix(FALSE, length(a), length(a)))
  mean((outer(a, a, "==") == outer(b, b, "=="))[keep])
}

one_run <- function(mu, z, e, tau, assumed_sigma = 1) {
  x <- mu + sigma * z
  discovery <- x + sqrt((1 - tau) / tau) * assumed_sigma * e
  inference <- x - sqrt(tau / (1 - tau)) * assumed_sigma * e
  groups <- cutree(hclust(dist(discovery), method = "ward.D2"), k = 2L)
  # Fix label orientation using discovery information only.
  if (mean(discovery[groups == 1L, 1L]) >
      mean(discovery[groups == 2L, 1L])) groups <- 3L - groups
  left <- groups == 1L
  estimate <- mean(inference[!left, 1L]) - mean(inference[left, 1L])
  target <- mean(mu[!left, 1L]) - mean(mu[left, 1L])
  standard_error <- assumed_sigma / sqrt(1 - tau) *
    sqrt(1 / sum(left) + 1 / sum(!left))
  half_width <- qnorm(0.975) * standard_error
  c(covered = abs(estimate - target) <= half_width,
    length = 2 * half_width, squared_error = (estimate - target)^2,
    reject_zero = abs(estimate) > half_width,
    agreement = pair_agreement(groups, reference_groups))
}

mu_null <- matrix(0, n, q)
mu_separated <- mu_null
mu_separated[, 1L] <- rep(c(-1.25, 1.25), each = n / 2L)
summaries <- list()
counter <- 1L
for (scenario in c("Global null", "Separated means")) {
  mu <- if (scenario == "Global null") mu_null else mu_separated
  for (tau in taus) {
    values <- vapply(seq_len(B), function(b)
      one_run(mu, noise[[b]], randomization[[b]], tau), numeric(5L))
    coverage <- mean(values["covered", ])
    summaries[[counter]] <- data.frame(
      scenario = scenario, discovery_fraction = tau,
      coverage = coverage,
      coverage_mcse = sqrt(coverage * (1 - coverage) / B),
      mean_interval_length = mean(values["length", ]),
      rmse_for_selected_target = sqrt(mean(values["squared_error", ])),
      null_rejection = if (scenario == "Global null")
        mean(values["reject_zero", ]) else NA_real_,
      grouping_agreement = if (scenario == "Separated means")
        mean(values["agreement", ]) else NA_real_
    )
    counter <- counter + 1L
  }
}

cat("Known sigma = 1; Ward discovery clustering; inference feature = 1\n")
cat("Coverage target: the actual selected groups' average underlying means.\n")
display_table <- do.call(rbind, summaries)
names(display_table) <- c("scenario", "tau", "coverage", "coverage_mcse",
                         "mean_CI_length", "selected_RMSE",
                         "null_rejection", "group_agreement")
print(display_table, row.names = FALSE, digits = 4)
cat("NA entries mark quantities deliberately not interpreted in that scenario.\n")

scales <- c(0.7, 1, 1.3)
sensitivity <- lapply(scales, function(assumed_sigma) {
  values <- vapply(seq_len(B), function(b)
    one_run(mu_null, noise[[b]], randomization[[b]],
            tau = 0.5, assumed_sigma = assumed_sigma), numeric(5L))
  rate <- mean(values["reject_zero", ])
  data.frame(
    assumed_sigma = assumed_sigma,
    component_covariance = sigma^2 - assumed_sigma^2,
    null_rejection = rate,
    monte_carlo_se = sqrt(rate * (1 - rate) / B)
  )
})
cat("\nScale sensitivity at discovery fraction 0.5; true sigma remains 1\n")
print(do.call(rbind, sensitivity), row.names = FALSE, digits = 4)
cat("Only assumed_sigma = 1 is covered by the independence derivation.\n")

plot_dir <- Sys.getenv("COURSE_FIGURE_DIR", unset = "")
if (nzchar(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  tab <- do.call(rbind, summaries)
  selected <- tab[tab$scenario == "Separated means", ]
  png(file.path(plot_dir, "14-research-synthesis.png"),
      width = 1400, height = 650, res = 130)
  old_par <- par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3, 1))
  plot(selected$discovery_fraction, selected$grouping_agreement,
       type = "b", pch = 19, col = "#2a8a7c", ylim = c(0.5, 1),
       xlab = "Discovery information fraction", ylab = "Co-membership agreement",
       main = "Discovery improves with more information")
  plot(selected$discovery_fraction, selected$mean_interval_length,
       type = "b", pch = 19, col = "#bd4b38",
       xlab = "Discovery information fraction", ylab = "Mean 95% interval length",
       main = "Inference loses precision")
  par(old_par)
  invisible(dev.off())
}
