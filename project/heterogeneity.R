d <- read.csv("data/derived/analysis_sample.csv")
fit <- lm(change ~ treatment * prior_use +
          factor(occupation) * factor(a_first) +
          factor(incentive_arm), data = d)
X <- model.matrix(fit)
u <- residuals(fit)
bread <- solve(crossprod(X))
V <- nrow(X) / df.residual(fit) *
     bread %*% crossprod(X * u) %*% bread
result <- data.frame(term = names(coef(fit)), estimate = coef(fit),
                     se = sqrt(diag(V)))
write.csv(result, "outputs/heterogeneity.csv", row.names = FALSE)
print(result[result$term %in% c("treatment", "treatment:prior_use"), ],
      row.names = FALSE)
