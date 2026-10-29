d <- read.csv("data/derived/analysis_sample.csv")
variables <- c("time1", "time2", "change", "prior_use")
stats <- do.call(rbind, lapply(0:1, function(group) {
  do.call(rbind, lapply(variables, function(variable) {
    z <- d[d$treatment == group, variable]
    z <- z[!is.na(z)]
    data.frame(group, variable, n = length(z),
               mean = mean(z), sd = sd(z), min = min(z), max = max(z))
  }))
}))
write.csv(stats, "outputs/descriptive.csv", row.names = FALSE)
print(stats, row.names = FALSE, digits = 5)
print(with(d, table(treatment, prior_use, useNA = "ifany")))
