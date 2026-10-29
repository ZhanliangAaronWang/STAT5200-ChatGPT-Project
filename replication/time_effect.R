d <- read.csv("data/derived/analysis_sample.csv")
fit <- lm(change ~ treatment + factor(occupation) * factor(a_first) +
          factor(incentive_arm), data = d)
X <- model.matrix(fit)
u <- residuals(fit)
bread <- solve(crossprod(X))
V <- nrow(X) / df.residual(fit) *
     bread %*% crossprod(X * u) %*% bread
result <- data.frame(term = names(coef(fit)), estimate = coef(fit),
                     se = sqrt(diag(V)))
write.csv(result, "outputs/replication.csv", row.names = FALSE)
print(result[result$term == "treatment", ], row.names = FALSE)
