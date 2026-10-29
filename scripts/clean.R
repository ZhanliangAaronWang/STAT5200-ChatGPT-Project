x <- read.csv("data/raw/fullsurvey.csv", fileEncoding = "latin1",
              na.strings = c("", "."), stringsAsFactors = FALSE)
x <- x[x$finished == 1 & !(x$consent %in% 2) &
       !(x$consent_exact %in% 2), ]
x <- x[order(x$prolific_pid, x$startdate), ]
x <- x[!duplicated(x$prolific_pid), ]
excluded <- readLines("data/derived/excluded_ids.txt")
x <- x[!(x$prolific_pid %in% excluded), ]
q <- read.csv("data/derived/quality_checks.csv")
x <- merge(x, q, by = "prolific_pid", all.x = TRUE)
x <- x[(is.na(x$maxqcheck) | x$maxqcheck <= 1) &
       (is.na(x$minqcheck) | x$minqcheck >= 1), ]
g <- read.csv("data/raw/grading_long_complete.csv",
              fileEncoding = "latin1", stringsAsFactors = FALSE)
names(g) <- tolower(names(g))
g$dategraded <- as.Date(substr(g$startdate, 1, 10))
g <- aggregate(dategraded ~ grader_id, g, min)
x$dategraded <- g$dategraded[match(x$prolific_pid, g$grader_id)]
x$datetasked <- as.Date(substr(x$startdate, 1, 10))
x <- x[is.na(x$dategraded) | is.na(x$datetasked) |
       x$datetasked <= x$dategraded, ]
cat("Full cleaned sample:", nrow(x), "\n")
x <- x[x$incentive_arm %in% c("linear", "convex"), ]
x$time1 <- rowSums(x[paste0("time1_", 4:7)])
x$time2 <- rowSums(x[paste0("time2_", 4:7)])
x$prior_use <- ifelse(is.na(x$usage), NA,
                     as.integer(grepl("(^|,)3(,|$)", x$usage)))
x$change <- x$time2 - x$time1
keep <- c("prolific_pid", "treatment", "occupation", "a_first",
          "incentive_arm", "time1", "time2", "change", "prior_use")
x <- x[keep]
write.csv(x, "data/derived/analysis_sample.csv", row.names = FALSE)
