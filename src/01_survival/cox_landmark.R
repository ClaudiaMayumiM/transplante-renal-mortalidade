#!/usr/bin/env Rscript
# Sensibilidade landmark com o relógio reiniciado no dia 7.

suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: cox_landmark.R INPUT_CSV OUTPUT_DIR")
df <- read.csv(normalizePath(args[[1]], mustWork = TRUE), check.names = FALSE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
required <- c("landmark_followup_days", "event_after_landmark", "age_gt40_main", "dgf_main")
# Aplica complete case à população elegível no landmark.
cc <- df[complete.cases(df[, required]), required, drop = FALSE]
stopifnot(nrow(cc) == 180L, sum(cc$event_after_landmark) == 18L)
# Ajusta o Cox para tempo e evento após o marco temporal.
fit <- coxph(
  Surv(landmark_followup_days, event_after_landmark) ~ age_gt40_main + dgf_main,
  data = cc, ties = "efron", x = TRUE, y = TRUE, model = TRUE
)
s <- summary(fit)
results <- data.frame(
  term = rownames(s$coefficients), n = fit$n, events = fit$nevent,
  coefficient = s$coefficients[, "coef"], hr = s$coefficients[, "exp(coef)"],
  ci95_lower = s$conf.int[, "lower .95"], ci95_upper = s$conf.int[, "upper .95"],
  p_value = s$coefficients[, "Pr(>|z|)"], ties = "efron", row.names = NULL
)
write.csv(results, file.path(output_dir, "cox_landmark_results.csv"), row.names = FALSE)
# Verifica a hipótese de riscos proporcionais.
z <- cox.zph(fit)
ph <- as.data.frame(z$table)
ph$term <- rownames(ph)
rownames(ph) <- NULL
write.csv(ph, file.path(output_dir, "cox_landmark_ph.csv"), row.names = FALSE)
