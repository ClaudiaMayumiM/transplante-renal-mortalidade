#!/usr/bin/env python3
"""Transforma resultados analíticos agregados na Tabela 3 para publicação."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PACKAGE_ROOT / "outputs" / "reference" / "metrics" / "table3_reporting_source.csv"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "generated" / "tables"


def metric(value: str, lower: str, upper: str) -> str:
    return f"{float(value):.3f} ({float(lower):.3f}-{float(upper):.3f})"


def main() -> None:
    # Mantém a ordem predefinida dos modelos ao formatar a tabela final.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.source.open(encoding="utf-8")))
    if [row["model_id"] for row in rows] != [
        "logistic_parsimonious",
        "logistic_core_ridge",
        "decision_tree_shallow",
    ]:
        raise ValueError("Unexpected model set or order in Table 3 source")
    tree = rows[2]
    if tree["calibration_summary"] != "median" or abs(float(tree["calibration_slope"]) - 0.09176617740118166) > 1e-12:
        raise ValueError("A inclinação de calibração da árvore difere do valor esperado para o reporting")
    if tree["calibration_warning_repetitions"] != "41":
        raise ValueError("Tree calibration warning count must be 41")

    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "Modelo": row["model_label"],
                "N/eventos": f'{row["n"]}/{row["events"]}',
                "AUC média (P2,5-P97,5)": metric(row["auc"], row["auc_p2_5"], row["auc_p97_5"]),
                "AP média (P2,5-P97,5)": metric(row["average_precision"], row["ap_p2_5"], row["ap_p97_5"]),
                "Brier médio (P2,5-P97,5)": metric(row["brier"], row["brier_p2_5"], row["brier_p97_5"]),
                "Intercepto": f'{float(row["calibration_intercept"]):.3f}',
                "Inclinação": f'{float(row["calibration_slope"]):.3f}',
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "table3_oof_performance_2y.csv"
    md_path = args.output_dir / "table3_oof_performance_2y.md"
    fields = list(output_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    lines = [
        "<!-- Gera a tabela de desempenho a partir dos resultados agregados disponíveis. -->",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(fields) - 1)) + " |",
    ]
    lines.extend("| " + " | ".join(row[field] for field in fields) + " |" for row in output_rows)
    lines.extend(
        [
            "",
            "Fonte: resultados OOF congelados. Para a árvore rasa, intercepto e inclinação são medianas entre 100 repetições; para logística e ridge, são médias. Houve 41 avisos diagnósticos de calibração na árvore.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
