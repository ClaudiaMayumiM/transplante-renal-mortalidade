#!/usr/bin/env Rscript
# Modelo principal de Cox com seguimento completo disponível.

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: cox_primary.R INPUT_CSV OUTPUT_DIR")
input_csv <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_csv, check.names = FALSE)
required <- c("followup_days_fullfu", "event_death_fullfu", "age_gt40_main", "dgf_main")
stopifnot(all(required %in% names(df)))
# A população analítica usa complete case para tempo, evento e preditores.
dat <- df[complete.cases(df[, required]), required, drop = FALSE]
stopifnot(nrow(dat) == 192L, sum(dat$event_death_fullfu) == 22L)

# Ajusta tempo até óbito ou censura com empates pelo método de Efron.
fit <- coxph(
  Surv(followup_days_fullfu, event_death_fullfu) ~ age_gt40_main + dgf_main,
  data = dat, ties = "efron", x = TRUE, y = TRUE, model = TRUE
)
s <- summary(fit)
ci <- confint(fit)
results <- data.frame(
  term = rownames(s$coefficients), n = fit$n, events = fit$nevent,
  coefficient = s$coefficients[, "coef"], hr = s$coefficients[, "exp(coef)"],
  ci95_lower = exp(ci[, 1]), ci95_upper = exp(ci[, 2]),
  p_value = s$coefficients[, "Pr(>|z|)"], ties = "efron",
  row.names = NULL
)
write.csv(results, file.path(output_dir, "cox_primary_results.csv"), row.names = FALSE)

# Avalia a hipótese de riscos proporcionais.
z <- cox.zph(fit, transform = "km")
ph <- as.data.frame(z$table)
ph$term <- rownames(ph)
rownames(ph) <- NULL
write.csv(ph, file.path(output_dir, "cox_primary_ph.csv"), row.names = FALSE)
