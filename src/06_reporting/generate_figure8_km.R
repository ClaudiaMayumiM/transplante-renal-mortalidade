#!/usr/bin/env Rscript
# Transforma coordenadas e horizontes agregados nas figuras de Kaplan-Meier.
# Este gerador cuida da apresentação gráfica e não recalcula as estimativas.

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(ragg)
  library(svglite)
})

root <- normalizePath(Sys.getenv("TCC_PROJECT_ROOT", unset = "."))
source_dir <- file.path(root, "outputs/reference/metrics")
out_dir <- file.path(root, "outputs/generated/figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

horizon_days <- c(0, 365, 730, 1096, 1461, 1826)
plot_xlim <- c(0, 1826)
palette <- c("#0072B2", "#D55E00")
label_days <- function(x) format(x, big.mark = ".", decimal.mark = ",", scientific = FALSE)

specs <- list(
  global = list(
    curve = "km_global_curva_completa_APPROVED.csv",
    horizons = "km_global_horizontes_APPROVED.csv",
    file = "figura_8a_km_global",
    panel = "A",
    title = "Sobrevida estimada por Kaplan-Meier na coorte global",
    labels = c("Global" = "Coorte global"),
    colors = c("Coorte global" = "#0072B2"),
    show_ci = TRUE,
    legend_title = NULL,
    note = "IC95% log-log; marcas verticais indicam censuras."
  ),
  idade = list(
    curve = "km_idade_curva_completa_APPROVED.csv",
    horizons = "km_idade_horizontes_APPROVED.csv",
    file = "figura_8b_km_categoria_etaria",
    panel = "B",
    title = "Sobrevida estimada por Kaplan-Meier segundo categoria etária",
    labels = c(
      "Categoria etária de menor idade" = "Categoria etária de referência",
      "Categoria etária de maior idade, rotulada como '>40 anos' no dataset original" =
        "Categoria etária de maior idade*"
    ),
    colors = c(
      "Categoria etária de referência" = palette[1],
      "Categoria etária de maior idade*" = palette[2]
    ),
    show_ci = FALSE,
    legend_title = "Categoria etária",
    note = paste0(
      "Análise descritiva; teste de log-rank não realizado. ",
      "*Categoria originalmente rotulada como ‘>40 anos’; inclui 12 participantes com exatamente 40 anos."
    )
  ),
  dgf = list(
    curve = "km_dgf_curva_completa_APPROVED.csv",
    horizons = "km_dgf_horizontes_APPROVED.csv",
    file = "figura_8c_km_dgf",
    panel = "C",
    title = "Sobrevida estimada por Kaplan-Meier segundo DGF",
    labels = c(
      "Sem DGF registrada na primeira semana" = "Sem DGF",
      "Com DGF registrada na primeira semana" = "Com DGF"
    ),
    colors = c("Sem DGF" = palette[1], "Com DGF" = palette[2]),
    show_ci = FALSE,
    legend_title = "DGF",
    note = paste0(
      "Análise descritiva; teste de log-rank não realizado. ",
      "Seis participantes com DGF ausente foram excluídos."
    )
  )
)

read_approved <- function(spec) {
  curve <- read.csv(file.path(source_dir, spec$curve), check.names = FALSE)
  horizons <- read.csv(file.path(source_dir, spec$horizons), check.names = FALSE)
  stopifnot(isTRUE(all.equal(sort(unique(horizons$horizon_days)), horizon_days)))
  stopifnot(all(horizons$n_risk >= 0), all(curve$survival >= 0 & curve$survival <= 1))

  # Corte editorial estrito. Para estratos sem observação exatamente no dia
  # 1.826, acrescenta-se somente o ponto terminal registrado na tabela de
  # horizontes (sem criar evento ou censura).
  curve <- curve[curve$time_days <= max(horizon_days), , drop = FALSE]
  endpoint <- horizons[horizons$horizon_days == max(horizon_days), , drop = FALSE]
  missing_endpoint <- !endpoint$stratum %in% curve$stratum[curve$time_days == max(horizon_days)]
  if (any(missing_endpoint)) {
    ep <- endpoint[missing_endpoint, , drop = FALSE]
    curve <- rbind(
      curve,
      data.frame(
        time_days = ep$horizon_days,
        time_years = ep$horizon_years,
        n_risk = ep$n_risk,
        n_event = 0,
        n_censor = 0,
        survival = ep$survival,
        std_error = ep$std_error,
        lower_95 = ep$lower_95,
        upper_95 = ep$upper_95,
        stratum = ep$stratum,
        check.names = FALSE
      )
    )
  }
  curve <- curve[order(curve$stratum, curve$time_days), , drop = FALSE]
  stopifnot(max(curve$time_days) == 1826)
  stopifnot(all(vapply(split(curve$time_days, curve$stratum), max, numeric(1)) == 1826))

  curve$label <- factor(
    unname(spec$labels[curve$stratum]),
    levels = unname(spec$labels)
  )
  horizons$label <- factor(
    unname(spec$labels[horizons$stratum]),
    levels = rev(unname(spec$labels))
  )
  list(curve = curve, horizons = horizons)
}

make_grobs <- function(spec, compact = FALSE) {
  dat <- read_approved(spec)
  curve <- dat$curve
  horizons <- dat$horizons
  censored <- curve[curve$n_censor > 0, , drop = FALSE]
  base_size <- if (compact) 8.8 else 11
  title_size <- if (compact) 9.7 else 10.5
  title_text <- paste0(spec$panel, ". ", spec$title)

  p <- ggplot(curve, aes(x = time_days, y = survival, colour = label, group = label))
  if (isTRUE(spec$show_ci)) {
    p <- p + geom_ribbon(
      aes(ymin = lower_95, ymax = upper_95, fill = label),
      alpha = 0.14, colour = NA, show.legend = FALSE
    )
  }
  p <- p +
    geom_step(linewidth = if (compact) 0.7 else 0.9, direction = "hv") +
    geom_point(
      data = censored, shape = 124,
      size = if (compact) 1.7 else 2.2,
      stroke = 0.45, alpha = 0.72, show.legend = FALSE
    ) +
    scale_colour_manual(values = spec$colors, drop = FALSE) +
    scale_x_continuous(
      breaks = horizon_days,
      labels = label_days,
      expand = expansion(mult = 0)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x),
      expand = expansion(mult = c(0, 0.015))
    ) +
    coord_cartesian(xlim = plot_xlim, ylim = c(0, 1), expand = FALSE, clip = "on") +
    labs(
      title = title_text,
      x = NULL,
      y = "Probabilidade de sobrevida",
      colour = spec$legend_title
    ) +
    theme_classic(base_family = "sans", base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = title_size, hjust = 0),
      axis.line = element_line(colour = "#333333", linewidth = 0.45),
      axis.ticks = element_line(colour = "#333333", linewidth = 0.4),
      axis.text = element_text(colour = "#222222"),
      panel.grid.major.y = element_line(colour = "#E7E7E7", linewidth = 0.35),
      legend.position = if (length(spec$labels) == 1) "none" else "top",
      legend.justification = "left",
      legend.box.just = "left",
      legend.title = element_text(size = if (compact) 7.7 else 9.2),
      legend.text = element_text(size = if (compact) 7.7 else 9.2),
      legend.key.width = unit(if (compact) 0.9 else 1.05, "cm"),
      legend.margin = margin(0, 0, 2, 0),
      plot.margin = margin(7, 18, 2, 18)
    )
  if (isTRUE(spec$show_ci)) {
    p <- p + scale_fill_manual(values = spec$colors, drop = FALSE)
  }

  horizons$text_hjust <- ifelse(
    horizons$horizon_days == min(horizon_days), 0,
    ifelse(horizons$horizon_days == max(horizon_days), 1, 0.5)
  )
  risk <- ggplot(horizons, aes(x = horizon_days, y = label, label = n_risk, colour = label)) +
    geom_text(aes(hjust = text_hjust), size = if (compact) 2.65 else 3.35, show.legend = FALSE) +
    scale_colour_manual(values = spec$colors, drop = FALSE) +
    scale_x_continuous(
      limits = plot_xlim, breaks = horizon_days,
      labels = label_days,
      expand = expansion(mult = 0)
    ) +
    labs(x = "Tempo de seguimento (dias)", y = NULL, title = "Número em risco") +
    theme_classic(base_family = "sans", base_size = if (compact) 7.8 else 9.5) +
    theme(
      plot.title = element_text(face = "bold", size = if (compact) 8.5 else 10, hjust = 0),
      axis.line.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_text(colour = "#222222", hjust = 1),
      axis.text.x = element_text(colour = "#222222"),
      plot.margin = margin(0, 18, 2, 18)
    )

  note <- textGrob(
    spec$note,
    x = unit(0.01, "npc"), y = unit(0.55, "npc"), just = c("left", "center"),
    gp = gpar(fontfamily = "sans", fontsize = if (compact) 6.8 else 8.2, col = "#444444")
  )
  list(plot = p, risk = risk, note = note)
}

assemble <- function(spec, compact = FALSE) {
  g <- make_grobs(spec, compact)
  arrangeGrob(
    g$plot, g$risk, g$note, ncol = 1,
    heights = if (compact) c(4.2, 1.4, 0.42) else c(4.35, 1.35, 0.38)
  )
}

save_all <- function(grob, basename, width, height) {
  png_file <- file.path(out_dir, paste0(basename, ".png"))
  svg_file <- file.path(out_dir, paste0(basename, ".svg"))
  pdf_file <- file.path(out_dir, paste0(basename, ".pdf"))
  ggsave(png_file, grob, width = width, height = height, units = "in", dpi = 400,
         device = ragg::agg_png, bg = "white")
  ggsave(svg_file, grob, width = width, height = height, units = "in",
         device = svglite::svglite, bg = "white")
  ggsave(pdf_file, grob, width = width, height = height, units = "in",
         device = grDevices::pdf, bg = "white", useDingbats = FALSE)
}

# Validações factuais e de identidade das fontes agregadas. As verificações
# antecedem a exportacao para impedir a producao a partir de entradas divergentes.
expected_n_risk <- list(
  global = c(`0` = 198, `365` = 140, `730` = 99, `1096` = 63, `1461` = 42, `1826` = 9),
  idade_ref = c(`0` = 104, `365` = 77, `730` = 58, `1096` = 39, `1461` = 31, `1826` = 8),
  idade_maior = c(`0` = 94, `365` = 63, `730` = 41, `1096` = 24, `1461` = 11, `1826` = 1),
  dgf_sem = c(`0` = 151, `365` = 120, `730` = 84, `1096` = 57, `1461` = 37, `1826` = 8),
  dgf_com = c(`0` = 41, `365` = 18, `730` = 13, `1096` = 5, `1461` = 4, `1826` = 1)
)
expected_survival <- list(
  global = c(`365` = 0.903617719425604, `730` = 0.879830591193675, `1826` = 0.832810390154615),
  idade_ref = c(`365` = 0.942694791525800, `730` = 0.926716913703329, `1826` = 0.910739035880858),
  idade_maior = c(`365` = 0.861800939989142, `730` = 0.829245380602524, `1826` = 0.736545185616470),
  dgf_sem = c(`365` = 0.942317532884026, `730` = 0.923524875533063, `1826` = 0.867539796591257),
  dgf_com = c(`365` = 0.778225244721408, `730` = 0.722637727241308, `1826` = 0.722637727241308)
)
expected_hashes <- c(
  "km_global_curva_completa_APPROVED.csv" = "c00af435e412d77844b140f923bf9dc16514bd5194f2f9de988a23225bde83ad",
  "km_global_horizontes_APPROVED.csv" = "456ca8e8d8de779ebcd6d24ad4796fb6c0ed272be35a02b98172b975c46bf8db",
  "km_idade_curva_completa_APPROVED.csv" = "28ddde8a4aad434820ae60be52e9d7780304b76b87037840aba04f642b82f1fa",
  "km_idade_horizontes_APPROVED.csv" = "cd693946957da26f3e1e55e4263b21b1ecba5aa90c197ccf977407a59bbccfcb",
  "km_dgf_curva_completa_APPROVED.csv" = "8840ee0a4840dd4c655f033b1554f52c8984aec7d0f6d9d830126cc9f4baafa2",
  "km_dgf_horizontes_APPROVED.csv" = "8eba33feaff350048501a470fac53b8e8bb1c19f95dc5d4448f912274765f39f"
)

input_paths <- file.path(source_dir, names(expected_hashes))
stopifnot(all(file.exists(input_paths)))
observed_hashes <- vapply(
  input_paths,
  function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE),
  character(1)
)
stopifnot(identical(unname(observed_hashes), unname(expected_hashes)))
write.csv(
  data.frame(
    arquivo = names(expected_hashes),
    sha256 = unname(observed_hashes),
    status = "MATCH_APPROVED",
    stringsAsFactors = FALSE
  ),
  file.path(out_dir, "inputs_sha256.csv"), row.names = FALSE, fileEncoding = "UTF-8"
)

hg <- read.csv(file.path(source_dir, specs$global$horizons))
hi <- read.csv(file.path(source_dir, specs$idade$horizons))
hd <- read.csv(file.path(source_dir, specs$dgf$horizons))
stopifnot(all(hg$n_risk == unname(expected_n_risk$global)))
stopifnot(all(hi$n_risk[hi$stratum == names(specs$idade$labels)[1]] == unname(expected_n_risk$idade_ref)))
stopifnot(all(hi$n_risk[hi$stratum == names(specs$idade$labels)[2]] == unname(expected_n_risk$idade_maior)))
stopifnot(all(hd$n_risk[hd$stratum == names(specs$dgf$labels)[1]] == unname(expected_n_risk$dgf_sem)))
stopifnot(all(hd$n_risk[hd$stratum == names(specs$dgf$labels)[2]] == unname(expected_n_risk$dgf_com)))

assert_survival <- function(data, stratum, expected, tolerance = 1e-10) {
  for (day in names(expected)) {
    observed <- data$survival[
      data$stratum == stratum & data$horizon_days == as.numeric(day)
    ]
    stopifnot(length(observed) == 1L)
    stopifnot(abs(observed - unname(expected[[day]])) < tolerance)
  }
}
assert_survival(hg, "Global", expected_survival$global)
assert_survival(hi, names(specs$idade$labels)[1], expected_survival$idade_ref)
assert_survival(hi, names(specs$idade$labels)[2], expected_survival$idade_maior)
assert_survival(hd, names(specs$dgf$labels)[1], expected_survival$dgf_sem)
assert_survival(hd, names(specs$dgf$labels)[2], expected_survival$dgf_com)

individual <- lapply(specs, assemble, compact = FALSE)
for (name in names(specs)) {
  save_all(individual[[name]], specs[[name]]$file, width = 9.2, height = 6.7)
}

# Mantido como opção suplementar; a apresentação principal usa três figuras individuais.
composite <- arrangeGrob(
  assemble(specs$global, compact = TRUE),
  arrangeGrob(
    assemble(specs$idade, compact = TRUE),
    assemble(specs$dgf, compact = TRUE),
    ncol = 2, widths = c(1, 1)
  ),
  ncol = 1, heights = c(1.03, 1)
)
save_all(composite, "figura_8_km_painel_composto", width = 13.2, height = 11.3)

message("Figuras exportadas e validacoes factuais concluidas com sucesso em: ", out_dir)
