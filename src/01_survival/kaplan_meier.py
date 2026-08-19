"""Calcula estimativas descritivas de Kaplan-Meier e resumos para reporting."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(os.environ.get("TCC_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "base_s_survival_fullfu.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "km_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TIME_COL = "followup_days_fullfu"
EVENT_COL = "event_death_fullfu"
TIMEPOINTS = [365, 730, 1096, 1826]
STRATA = [
    ("global", None),
    ("age_gt40_main", {"0": "Categoria etária de referência", "1": "Categoria etária de maior idade"}),
    ("dgf_main", {"0": "Sem DGF", "1": "Com DGF"}),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_missing(value: str | None) -> bool:
    return value is None or value.strip() == ""


def to_float(value: str | None) -> float | None:
    if is_missing(value):
        return None
    return float(value)


def to_int(value: str | None) -> int | None:
    numeric = to_float(value)
    if numeric is None:
        return None
    return int(numeric)


def valid_survival_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        time_value = to_float(row.get(TIME_COL))
        event_value = to_int(row.get(EVENT_COL))
        if time_value is None or event_value is None:
            continue
        filtered.append(row)
    return filtered


def km_curve(rows: list[dict[str, str]]) -> dict[str, object]:
    """Calcula a curva de sobrevivência e os números em risco."""
    records = []
    for row in rows:
        time_value = to_float(row[TIME_COL])
        event_value = to_int(row[EVENT_COL])
        records.append((time_value, event_value))
    records.sort(key=lambda item: (item[0], -item[1]))

    n_total = len(records)
    n_events = sum(event for _, event in records)
    n_censored = n_total - n_events
    followup_values = [time for time, _ in records]

    points = [(0.0, 1.0)]
    event_table: list[dict[str, float]] = []
    surv = 1.0
    idx = 0

    while idx < n_total:
        current_time = records[idx][0]
        d_i = 0
        c_i = 0
        while idx < n_total and records[idx][0] == current_time:
            if records[idx][1] == 1:
                d_i += 1
            else:
                c_i += 1
            idx += 1
        n_risk = sum(1 for time, _ in records if time >= current_time)
        if d_i > 0 and n_risk > 0:
            surv *= (1.0 - (d_i / n_risk))
            points.append((current_time, points[-1][1]))
            points.append((current_time, surv))
        event_table.append(
            {
                "time": current_time,
                "n_risk": n_risk,
                "events": d_i,
                "censored": c_i,
                "survival": surv,
            }
        )

    median_survival = ""
    for time, survival in points:
        if survival <= 0.5:
            median_survival = f"{time:.0f}"
            break

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_censored": n_censored,
        "followup_min": min(followup_values) if followup_values else math.nan,
        "followup_median": median(followup_values) if followup_values else math.nan,
        "followup_max": max(followup_values) if followup_values else math.nan,
        "followup_values": followup_values,
        "median_survival_days": median_survival if median_survival else "not_reached",
        "points": points,
        "event_table": event_table,
    }


def survival_at(curve: dict[str, object], timepoint: float) -> float:
    last_survival = 1.0
    for time, survival in curve["points"]:  # type: ignore[index]
        if time <= timepoint:
            last_survival = survival
        else:
            break
    return last_survival


def n_at_risk_at(curve: dict[str, object], timepoint: float) -> int:
    return sum(1 for value in curve["followup_values"] if value >= timepoint)  # type: ignore[index]


def tail_note(curve: dict[str, object]) -> str:
    n_risk_5y = n_at_risk_at(curve, 1826)
    if n_risk_5y < 10:
        return (
            "Curva Kaplan-Meier usada apenas para descrição de sobrevida; não constitui "
            "modelo preditivo. Interpretação cautelosa em 5 anos por baixo número de "
            "pacientes em risco."
        )
    return "Curva Kaplan-Meier usada apenas para descrição de sobrevida; não constitui modelo preditivo."


def label_for_stratum(variable: str, raw_value: str) -> str:
    if variable == "age_gt40_main":
        return "Categoria etária de maior idade" if raw_value == "1" else "Categoria etária de referência"
    if variable == "dgf_main":
        return "Com DGF" if raw_value == "1" else "Sem DGF"
    return raw_value


def stratified_groups(rows: list[dict[str, str]], variable: str) -> tuple[dict[str, list[dict[str, str]]], int]:
    """Organiza estratos usados somente para descrição da sobrevivência."""
    groups: dict[str, list[dict[str, str]]] = {}
    excluded_missing = 0
    for row in rows:
        value = row.get(variable, "")
        if is_missing(value):
            excluded_missing += 1
            continue
        groups.setdefault(value, []).append(row)
    return dict(sorted(groups.items())), excluded_missing


def svg_step_path(points: list[tuple[float, float]], x_scale, y_scale) -> str:
    commands = []
    for idx, (x, y) in enumerate(points):
        px = x_scale(x)
        py = y_scale(y)
        cmd = "M" if idx == 0 else "L"
        commands.append(f"{cmd} {px:.2f} {py:.2f}")
    return " ".join(commands)


def render_svg(
    curves: list[tuple[str, dict[str, object]]],
    title: str,
    output_path: Path,
) -> None:
    width = 900
    height = 620
    left = 90
    right = 30
    top = 70
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_time = max(
        max(time for time, _ in curve["points"])  # type: ignore[index]
        for _, curve in curves
    )
    x_max = max(max_time, 1.0)

    def x_scale(value: float) -> float:
        return left + (value / x_max) * plot_width

    def y_scale(value: float) -> float:
        return top + ((1.0 - value) * plot_height)

    colors = ["#0f766e", "#b45309", "#1d4ed8", "#b91c1c"]
    x_ticks = 6
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    any_unstable_tail = any(n_at_risk_at(curve, 1826) < 10 for _, curve in curves)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{title}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="36" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700">{title}</text>',
        f'<text x="{left}" y="58" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#444">Curva Kaplan-Meier usada apenas para descrição de sobrevida; não constitui modelo preditivo.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111" stroke-width="1.5"/>',
    ]
    if any_unstable_tail:
        parts.append(
            f'<text x="{left}" y="76" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#7c2d12">Interpretação cautelosa em 5 anos por baixo número de pacientes em risco.</text>'
        )

    for tick_idx in range(x_ticks + 1):
        tick_value = (x_max / x_ticks) * tick_idx
        x = x_scale(tick_value)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{tick_value:.0f}</text>')
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>')

    for tick_value in y_ticks:
        y = y_scale(tick_value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12">{tick_value:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')

    for idx, (label, curve) in enumerate(curves):
        color = colors[idx % len(colors)]
        path = svg_step_path(curve["points"], x_scale, y_scale)  # type: ignore[arg-type]
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in curve["event_table"]:  # type: ignore[index]
            if row["events"] > 0:
                cx = x_scale(row["time"])
                cy = y_scale(row["survival"])
                parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.8" fill="{color}"/>')

    legend_x = left + plot_width - 180
    legend_y = top + 16
    for idx, (label, _) in enumerate(curves):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 26}" y="{y + 4}" font-family="Arial, Helvetica, sans-serif" font-size="12">{label}</text>')

    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="14">Follow-up (dias)</text>',
            f'<text x="24" y="{top + plot_height / 2:.2f}" transform="rotate(-90 24 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="14">Sobrevida estimada</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def build_summary_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[Path]]:
    """Transforma curvas e horizontes em resumos tabulares e figuras intermediárias."""
    summary_rows: list[dict[str, object]] = []
    figure_paths: list[Path] = []

    overall_curve = km_curve(rows)
    summary_rows.append(
        {
            "analysis": "global",
            "analysis_type": "análise descritiva",
            "stratum": "overall",
            "n_included": overall_curve["n_total"],
            "n_excluded_missing_stratum": 0,
            "events": overall_curve["n_events"],
            "censored": overall_curve["n_censored"],
            "followup_min_days": f'{overall_curve["followup_min"]:.0f}',
            "followup_median_days": f'{overall_curve["followup_median"]:.0f}',
            "followup_max_days": f'{overall_curve["followup_max"]:.0f}',
            "km_median_survival_days": overall_curve["median_survival_days"],
            "survival_365d": f'{survival_at(overall_curve, 365):.4f}',
            "survival_730d": f'{survival_at(overall_curve, 730):.4f}',
            "survival_1096d": f'{survival_at(overall_curve, 1096):.4f}',
            "survival_1826d": f'{survival_at(overall_curve, 1826):.4f}',
            "n_risk_365d": n_at_risk_at(overall_curve, 365),
            "n_risk_730d": n_at_risk_at(overall_curve, 730),
            "n_risk_1096d": n_at_risk_at(overall_curve, 1096),
            "n_risk_1826d": n_at_risk_at(overall_curve, 1826),
            "note": tail_note(overall_curve),
        }
    )
    global_path = FIGURES_DIR / "km_global.svg"
    render_svg(
        [("Overall", overall_curve)],
        "Kaplan-Meier - análise descritiva - Global - Base S Full Follow-Up",
        global_path,
    )
    figure_paths.append(global_path)

    for variable, _ in STRATA[1:]:
        groups, excluded_missing = stratified_groups(rows, variable)
        curves_for_plot: list[tuple[str, dict[str, object]]] = []
        for raw_value, group_rows in groups.items():
            curve = km_curve(group_rows)
            label = label_for_stratum(variable, raw_value)
            curves_for_plot.append((label, curve))
            summary_rows.append(
                {
                    "analysis": variable,
                    "analysis_type": "análise descritiva",
                    "stratum": label,
                    "n_included": curve["n_total"],
                    "n_excluded_missing_stratum": excluded_missing,
                    "events": curve["n_events"],
                    "censored": curve["n_censored"],
                    "followup_min_days": f'{curve["followup_min"]:.0f}',
                    "followup_median_days": f'{curve["followup_median"]:.0f}',
                    "followup_max_days": f'{curve["followup_max"]:.0f}',
                    "km_median_survival_days": curve["median_survival_days"],
                    "survival_365d": f'{survival_at(curve, 365):.4f}',
                    "survival_730d": f'{survival_at(curve, 730):.4f}',
                    "survival_1096d": f'{survival_at(curve, 1096):.4f}',
                    "survival_1826d": f'{survival_at(curve, 1826):.4f}',
                    "n_risk_365d": n_at_risk_at(curve, 365),
                    "n_risk_730d": n_at_risk_at(curve, 730),
                    "n_risk_1096d": n_at_risk_at(curve, 1096),
                    "n_risk_1826d": n_at_risk_at(curve, 1826),
                    "note": tail_note(curve),
                }
            )
        figure_path = FIGURES_DIR / (
            "km_by_age.svg" if variable == "age_gt40_main" else "km_by_dgf.svg"
        )
        render_svg(
            curves_for_plot,
            f"Kaplan-Meier - análise descritiva - por {variable} - Base S Full Follow-Up",
            figure_path,
        )
        figure_paths.append(figure_path)

    return summary_rows, figure_paths


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "analysis",
                "analysis_type",
                "stratum",
                "n_included",
                "n_excluded_missing_stratum",
                "events",
                "censored",
                "followup_min_days",
                "followup_median_days",
                "followup_max_days",
                "km_median_survival_days",
                "survival_365d",
                "survival_730d",
                "survival_1096d",
                "survival_1826d",
                "n_risk_365d",
                "n_risk_730d",
                "n_risk_1096d",
                "n_risk_1826d",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = valid_survival_rows(load_rows(BASE_PATH))
    summary_rows, figure_paths = build_summary_rows(rows)
    write_summary(summary_rows)
    print(SUMMARY_PATH)
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
