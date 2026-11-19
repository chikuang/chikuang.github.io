## -----------------------------------------------------------------------------
#| label: a5-tree-fits
#| results: hide
a5_trees <- transform(datasets::trees, Diameter = Girth)
a5_raw <- lm(Volume ~ Diameter + Height, data = a5_trees)
a5_log <- lm(log(Volume) ~ log(Diameter) + log(Height),
             data = a5_trees)
a5_new <- data.frame(Diameter = 12, Height = 75)
a5_eta <- as.numeric(predict(a5_log, newdata = a5_new))
a5_s <- sigma(a5_log)


## -----------------------------------------------------------------------------
#| label: a5-tree-output
#| echo: false
knitr::kable(data.frame(Term = names(coef(a5_log)),
  Estimate = unname(coef(a5_log))), digits = 6,
  row.names = FALSE, align = c("l", "r"))
knitr::kable(data.frame(
  Quantity = c("Predicted log volume at diameter 12, height 75",
    "Residual SD of the log model", "SSE of the raw-volume model",
    "SSE of the log-volume model"),
  Value = c(a5_eta, a5_s, deviance(a5_raw), deviance(a5_log))),
  digits = 6, row.names = FALSE, align = c("l", "r"))


## -----------------------------------------------------------------------------
#| label: fig-a5-tree-diagnostics
#| echo: false
#| fig-pos: "H"
#| fig-width: 6.5
#| fig-height: 3.6
#| fig-cap: "Top: standardized residuals versus fitted values, with a smooth trend. Bottom: normal Q-Q plots with quartile reference lines. Left: raw-volume fit. Right: log-volume fit."
#| fig-alt: "Four diagnostic panels compare raw-volume and log-volume fits for 31 trees. The raw fit has a curved residual smooth; the log fit has a less pronounced mean pattern but some tail departures remain."
a5_par <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1),
              mgp = c(2.4, 0.7, 0), cex = 0.8)
for (j in 1:2) {
  m <- list(a5_raw, a5_log)[[j]]
  plot(fitted(m), rstandard(m), pch = 19, col = "#247D77",
       xlab = c("Fitted volume (cubic feet)", "Fitted log volume")[j],
       ylab = "Standardized residual", main = c("Raw-volume fit", "Log-volume fit")[j])
  abline(h = 0, lty = 2, col = "grey60")
  lines(lowess(fitted(m), rstandard(m)), col = "#B45F06", lwd = 2)
}
for (j in 1:2) {
  m <- list(a5_raw, a5_log)[[j]]
  qqnorm(rstandard(m), pch = 19, col = "#247D77",
         main = c("Raw fit: normal Q-Q", "Log fit: normal Q-Q")[j],
         ylab = "Standardized residual")
  qqline(rstandard(m), col = "#593196", lwd = 2)
}
par(a5_par)


## -----------------------------------------------------------------------------
#| label: a5-swiss-search
#| results: hide
a5_swiss <- datasets::swiss
a5_null <- lm(Fertility ~ 1, data = a5_swiss, na.action = na.fail)
a5_full <- lm(Fertility ~ ., data = a5_swiss, na.action = na.fail)
a5_selected <- step(a5_null,
  scope = list(lower = formula(a5_null), upper = formula(a5_full)),
  direction = "forward", k = 2, trace = 0)


## -----------------------------------------------------------------------------
#| label: a5-swiss-scores
#| echo: false
a5_candidates <- lapply(names(a5_swiss)[-1], function(term)
  lm(reformulate(term, response = "Fertility"), data = a5_swiss))
knitr::kable(data.frame(
  `First-step candidate model` = c("Intercept only",
    paste("Intercept +", names(a5_swiss)[-1])),
  AIC = c(AIC(a5_null), vapply(a5_candidates, AIC, numeric(1))),
  check.names = FALSE), digits = 3, row.names = FALSE)
knitr::kable(data.frame(
  Model = c("Final forward model", "All five predictors"),
  SSE = c(deviance(a5_selected), deviance(a5_full)),
  AIC = c(AIC(a5_selected), AIC(a5_full))),
  digits = 3, row.names = FALSE)


## -----------------------------------------------------------------------------
#| label: a5-sensor-data
#| include: false
set.seed(8561)
a5_n <- 120
a5_x <- matrix(rnorm(a5_n * 6), nrow = a5_n)
a5_x[, 2] <- 0.75 * a5_x[, 1] + sqrt(1 - 0.75^2) * a5_x[, 2]
colnames(a5_x) <- paste0("x", 1:6)
a5_dat <- data.frame(y = 2 + 2 * a5_x[, 1] - a5_x[, 2] +
  a5_x[, 3] + rnorm(a5_n, sd = 1.5), a5_x)
a5_id <- sample(seq_len(a5_n), 90)
a5_train <- a5_dat[a5_id, ]
a5_test <- a5_dat[-a5_id, ]
a5_folds <- sample(rep(1:5, each = 18))


## -----------------------------------------------------------------------------
#| label: a5-sensor-cv
#| results: hide
library(glmnet)
a5_xtrain <- model.matrix(y ~ ., data = a5_train)[, -1]
a5_xtest <- model.matrix(y ~ ., data = a5_test)[, -1]
a5_lasso <- cv.glmnet(a5_xtrain, a5_train$y,
  family = "gaussian", alpha = 1, intercept = TRUE,
  foldid = a5_folds,
  lambda = c(0.80, 0.40, 0.20, 0.10, 0.05),
  standardize = TRUE, type.measure = "mse")
a5_ridge <- cv.glmnet(a5_xtrain, a5_train$y,
  family = "gaussian", alpha = 0, intercept = TRUE,
  foldid = a5_folds,
  standardize = TRUE, type.measure = "mse")


## -----------------------------------------------------------------------------
#| label: a5-cv-table
#| echo: false
knitr::kable(data.frame(Lambda = a5_lasso$lambda,
  `CV MSE` = a5_lasso$cvm, `CV SE` = a5_lasso$cvsd,
  `Nonzero slopes` = a5_lasso$nzero, check.names = FALSE),
  digits = c(2, 3, 3, 0), row.names = FALSE)


## -----------------------------------------------------------------------------
#| label: a5-coefficient-table
#| echo: false
a5_coef_table <- data.frame(Term = rownames(coef(a5_lasso)),
  `Ridge: one-SE` = as.numeric(coef(a5_ridge, s = "lambda.1se")),
  `Lasso: minimum` = as.numeric(coef(a5_lasso, s = "lambda.min")),
  `Lasso: one-SE` = as.numeric(coef(a5_lasso, s = "lambda.1se")),
  check.names = FALSE)
knitr::kable(a5_coef_table, digits = 3, row.names = FALSE)


## -----------------------------------------------------------------------------
#| label: a5-test-assessment
#| results: hide
a5_pred <- as.numeric(predict(a5_lasso,
  newx = a5_xtest, s = "lambda.1se"))
a5_rmse <- sqrt(mean((a5_test$y - a5_pred)^2))
a5_baseline <- sqrt(mean((a5_test$y - mean(a5_train$y))^2))


## -----------------------------------------------------------------------------
#| label: a5-test-table
#| echo: false
knitr::kable(data.frame(Procedure = c("Prespecified one-SE lasso",
  "Training-mean baseline"), `Test RMSE` = c(a5_rmse, a5_baseline),
  check.names = FALSE), digits = 3, row.names = FALSE)

