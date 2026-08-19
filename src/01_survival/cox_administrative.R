#!/usr/bin/env Rscript
# Sensibilidade de Cox com encerramento administrativo do acompanhamento.

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: cox_administrative.R INPUT_CSV OUTPUT_DIR")
input_csv <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_csv, check.names = FALSE)
required <- c("time_to_death_or_censor_days", "event_death", "age_gt40_main", "dgf_main")
stopifnot(all(required %in% names(df)))
# Aplica complete case às variáveis necessárias ao modelo.
dat <- df[complete.cases(df[, required]), required, drop = FALSE]
stopifnot(nrow(dat) == 192L, sum(dat$event_death) == 21L)

# Ajusta o modelo e preserva a definição administrativa de tempo e evento.
fit <- coxph(
  Surv(time_to_death_or_censor_days, event_death) ~ age_gt40_main + dgf_main,
  data = dat, ties = "efron", x = TRUE, y = TRUE, model = TRUE
)
s <- summary(fit)
ci <- confint(fit)
results <- data.frame(
  term = rownames(s$coefficients), n = fit$n, events = fit$nevent,
  coefficient = s$coefficients[, "coef"], hr = s$coefficients[, "exp(coef)"],
  ci95_lower = exp(ci[, 1]), ci95_upper = exp(ci[, 2]),
  p_value = s$coefficients[, "Pr(>|z|)"], ties = "efron",
  administrative_censor_date = "2015-06-30", row.names = NULL
)
write.csv(results, file.path(output_dir, "cox_administrative_results.csv"), row.names = FALSE)
# Verifica a hipótese de riscos proporcionais.
z <- cox.zph(fit, transform = "km")
ph <- as.data.frame(z$table)
ph$term <- rownames(ph)
rownames(ph) <- NULL
write.csv(ph, file.path(output_dir, "cox_administrative_ph.csv"), row.names = FALSE)
diagnostics <- data.frame(
  n = fit$n, events = fit$nevent, concordance = s$concordance[[1]],
  ties = "efron", optimism_correction = "NOT_PERFORMED"
)
write.csv(diagnostics, file.path(output_dir, "cox_administrative_diagnostics.csv"), row.names = FALSE)
