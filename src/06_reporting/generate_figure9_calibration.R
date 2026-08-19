#!/usr/bin/env Rscript
# Transforma resultados agregados de calibração na figura para publicação.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ragg)
  library(svglite)
  library(digest)
})

root <- normalizePath(Sys.getenv("TCC_PROJECT_ROOT", unset = "."))
reference_dir <- file.path(root, "outputs/reference/metrics")
source_file <- file.path(reference_dir, "figura_calibracao_oof_2_anos_logistica_ridge_source.csv")
table_source <- file.path(reference_dir, "table3_reporting_source.csv")
out_dir <- file.path(root, "outputs/generated/figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Confere a integridade das fontes agregadas antes de construir a figura.
stopifnot(
  digest(file = source_file, algo = "sha256", serialize = FALSE) ==
    "d91e4db80a89d1617e16de512dfa49988a88055bed65cbf720091841b261e5ac",
  digest(file = table_source, algo = "sha256", serialize = FALSE) ==
    "05b2dc29babbbf361017e41dede55a85dbdc42a23c4a96f35ff2a11d46a08b5b"
)
cal <- read.csv(source_file, check.names = FALSE, stringsAsFactors = FALSE)
tab <- read.csv(table_source, check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(nrow(cal) == 8L, all(table(cal$model_name) == 4L))
expected <- data.frame(
  model_id = c("logistic_parsimonious", "logistic_core_ridge"),
  n = c(115L, 114L), events = c(18L, 18L),
  intercept = c(-0.01168643299805537, 0.003968644026456433),
  slope = c(0.7329346991068176, 0.9938584077711768)
)
for (i in seq_len(nrow(expected))) {
  points <- cal[cal$model_name == expected$model_id[i], , drop = FALSE]
  stats <- tab[tab$model_id == expected$model_id[i], , drop = FALSE]
  stopifnot(
    sum(points$n) == expected$n[i], sum(points$events) == expected$events[i],
    nrow(stats) == 1L,
    abs(stats$calibration_intercept - expected$intercept[i]) < 1e-12,
    abs(stats$calibration_slope - expected$slope[i]) < 1e-12
  )
}

labels <- c(
  "logistic_parsimonious" = "Logística parcimoniosa",
  "logistic_core_ridge" = "Regressão ridge"
)
cal$model <- factor(unname(labels[cal$model_name]), levels = unname(labels))
cal <- cal[order(cal$model, cal$mean_predicted_probability), , drop = FALSE]
palette <- c("Logística parcimoniosa" = "#0072B2", "Regressão ridge" = "#D55E00")
shapes <- c("Logística parcimoniosa" = 21, "Regressão ridge" = 22)
linetypes <- c("Logística parcimoniosa" = "solid", "Regressão ridge" = "dashed")

p <- ggplot(cal, aes(
  x = mean_predicted_probability, y = observed_proportion,
  colour = model, shape = model, linetype = model, group = model
)) +
  geom_abline(intercept = 0, slope = 1, colour = "#555555", linewidth = 0.7, linetype = "longdash") +
  geom_errorbar(aes(ymin = wilson_lower_95, ymax = wilson_upper_95), width = 0.012, linewidth = 0.7) +
  geom_line(linewidth = 0.85) +
  geom_point(fill = "white", size = 3.1, stroke = 0.9) +
  annotate("text", x = 0.81, y = 0.835, label = "Calibração ideal", angle = 45,
           family = "sans", size = 3.2, colour = "#555555") +
  scale_colour_manual(values = palette, drop = FALSE) +
  scale_shape_manual(values = shapes, drop = FALSE) +
  scale_linetype_manual(values = linetypes, drop = FALSE) +
  scale_x_continuous(breaks = seq(0, 1, 0.2), labels = function(x) sprintf("%.1f", x), expand = expansion(mult = 0)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), labels = function(x) sprintf("%.1f", x), expand = expansion(mult = 0)) +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1), ratio = 1, clip = "off") +
  labs(
    title = "Calibração out-of-fold em dois anos",
    x = "Probabilidade prevista",
    y = "Proporção observada de óbito em dois anos",
    colour = NULL, shape = NULL, linetype = NULL,
    caption = paste0(
      "Quatro grupos definidos por quantis das probabilidades OOF médias por paciente; ",
      "barras verticais: IC95% de Wilson."
    )
  ) +
  theme_classic(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0),
    axis.title = element_text(size = 10.5), axis.text = element_text(colour = "#222222", size = 9.5),
    axis.line = element_line(colour = "#333333", linewidth = 0.45),
    panel.grid.major = element_line(colour = "#E7E7E7", linewidth = 0.35),
    panel.grid.minor = element_blank(), legend.position = "top", legend.justification = "left",
    plot.caption = element_text(hjust = 0, size = 8.3, colour = "#444444", margin = margin(t = 8)),
    plot.margin = margin(10, 18, 10, 14)
  )

base <- file.path(out_dir, "figura_9_calibracao_oof_2_anos")
ggsave(paste0(base, ".png"), p, width = 8.3, height = 7.2, units = "in", dpi = 400, device = ragg::agg_png, bg = "white")
ggsave(paste0(base, ".svg"), p, width = 8.3, height = 7.2, units = "in", device = svglite::svglite, bg = "white")
ggsave(paste0(base, ".pdf"), p, width = 8.3, height = 7.2, units = "in", device = grDevices::pdf, bg = "white", useDingbats = FALSE)
