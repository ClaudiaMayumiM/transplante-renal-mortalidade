#!/usr/bin/env python3
"""Calcula métricas de classificação a partir de previsões OOF existentes.

As previsões dos folds são reunidas dentro de cada repetição, os limiares
predefinidos são aplicados e as métricas correspondentes são calculadas. O
script não ajusta modelos nem gera novas previsões.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


AUTHORIZED: frozenset[tuple[str, str]] = frozenset(
    {
        ("2y", "logistic_parsimonious"),
        ("2y", "logistic_core_ridge"),
        ("2y", "decision_tree_shallow"),
        ("1y", "logistic_parsimonious"),
        ("1y", "logistic_core_ridge"),
        ("2y_landmark", "logistic_parsimonious"),
        ("2y_landmark", "logistic_core_ridge"),
    }
)
THRESHOLDS: tuple[str, str] = ("FIXED_0_5", "TRAIN_EVENT_RATE")
METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "positive_predictive_value",
    "negative_predictive_value",
    "accuracy",
    "balanced_accuracy",
    "f1",
)
KEY_COLUMNS: tuple[str, ...] = (
    "analysis_key",
    "model_name",
    "repeat_index",
    "threshold_name",
)


def read_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV preservando a representação dos valores de ponto flutuante."""
    return pd.read_csv(path, float_precision="round_trip")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Escreve linhas conforme o contrato de serialização do reporting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Normaliza aliases de colunas e mantém as combinações previstas."""
    data = frame.copy()
    if "analysis_key" not in data and "horizon_key" in data:
        data["analysis_key"] = data["horizon_key"]
    if "outcome_true" not in data and "y_true" in data:
        data["outcome_true"] = data["y_true"]

    required = {
        "analysis_key",
        "model_name",
        "repeat_index",
        "fold_index",
        "id_original",
        "outcome_true",
        "predicted_probability",
        "train_event_rate",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required OOF columns: {missing}")

    combinations = data[["analysis_key", "model_name"]].apply(tuple, axis=1)
    data = data[combinations.isin(AUTHORIZED)].copy()
    if data.empty:
        raise ValueError("No authorized analysis/model combinations found")
    return data


def load_oof(paths: Iterable[Path]) -> pd.DataFrame:
    """Carrega e combina um ou mais arquivos de previsões OOF existentes."""
    frames = [normalize_input(read_csv(path)) for path in paths]
    if not frames:
        raise ValueError("At least one OOF input is required")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["outcome_true"] = combined["outcome_true"].astype(int)
    return combined


def validate_oof(
    oof: pd.DataFrame,
    *,
    expected_repetitions: int | None = 100,
    expected_combinations: frozenset[tuple[str, str]] | None = AUTHORIZED,
) -> None:
    """Valida uma previsão OOF por participante em cada repetição completa."""
    data = normalize_input(oof)
    if data[["outcome_true", "predicted_probability", "train_event_rate"]].isna().any().any():
        raise ValueError("Missing outcome, prediction, or training-fold event rate")
    if not data["outcome_true"].isin([0, 1]).all():
        raise ValueError("outcome_true must be binary")
    if not data["predicted_probability"].between(0, 1, inclusive="both").all():
        raise ValueError("predicted_probability must be within [0, 1]")
    if not data["train_event_rate"].between(0, 1, inclusive="both").all():
        raise ValueError("train_event_rate must be within [0, 1]")

    patient_key = ["analysis_key", "model_name", "repeat_index", "id_original"]
    if data.duplicated(patient_key).any():
        raise ValueError("A participant has more than one OOF prediction in a repetition")

    fold_rates = data.groupby(
        ["analysis_key", "model_name", "repeat_index", "fold_index"], sort=True
    )["train_event_rate"].nunique(dropna=False)
    if not fold_rates.eq(1).all():
        raise ValueError("train_event_rate is not constant within an originating fold")

    observed = set(map(tuple, data[["analysis_key", "model_name"]].drop_duplicates().to_numpy()))
    if expected_combinations is not None and observed != set(expected_combinations):
        raise ValueError(
            f"Analysis/model combinations differ: observed={sorted(observed)}"
        )
    if expected_repetitions is not None:
        repetitions = data.groupby(["analysis_key", "model_name"])["repeat_index"].nunique()
        if not repetitions.eq(expected_repetitions).all():
            raise ValueError("Incomplete repetition coverage")


def confusion_metrics(y: np.ndarray, predicted_class: np.ndarray) -> dict[str, Any]:
    """Calcula a matriz de confusão e suas métricas, preservando valores NaN."""
    tp = int(((y == 1) & (predicted_class == 1)).sum())
    fp = int(((y == 0) & (predicted_class == 1)).sum())
    tn = int(((y == 0) & (predicted_class == 0)).sum())
    fn = int(((y == 1) & (predicted_class == 0)).sum())

    def divide(numerator: int, denominator: int) -> float:
        return np.nan if denominator == 0 else numerator / denominator

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    ppv = divide(tp, tp + fp)
    npv = divide(tn, tn + fn)
    accuracy = divide(tp + tn, len(y))
    balanced_accuracy = (
        np.nan
        if np.isnan(sensitivity) or np.isnan(specificity)
        else (sensitivity + specificity) / 2
    )
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


def classification(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula uma matriz por repetição para 0,5 e prevalência do fold de treino, sem otimização."""
    data = normalize_input(oof)
    rows: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for (analysis, model, repetition), group in data.groupby(
        ["analysis_key", "model_name", "repeat_index"], sort=True
    ):
        y = group["outcome_true"].to_numpy(int)
        for threshold_name in THRESHOLDS:
            if threshold_name == "FIXED_0_5":
                threshold = np.repeat(0.5, len(group))
            else:
                threshold = group["train_event_rate"].to_numpy(float)
            predicted_class = (
                group["predicted_probability"].to_numpy(float) >= threshold
            ).astype(int)
            metrics = confusion_metrics(y, predicted_class)
            undefined = [
                name
                for name, value in metrics.items()
                if isinstance(value, float) and np.isnan(value)
            ]
            row = {
                "analysis_key": analysis,
                "model_name": model,
                "repeat_index": repetition,
                "threshold_name": threshold_name,
                "n": len(group),
                "events": int(y.sum()),
                "metric_status": "METRIC_UNDEFINED" if undefined else "VALID",
                "undefined_metrics": "|".join(undefined),
                **metrics,
            }
            rows.append(row)
            counts.append(
                {
                    key: row[key]
                    for key in (
                        "analysis_key",
                        "model_name",
                        "repeat_index",
                        "threshold_name",
                        "true_positives",
                        "false_positives",
                        "true_negatives",
                        "false_negatives",
                    )
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(counts)


def summarize(metrics_by_repetition: pd.DataFrame) -> pd.DataFrame:
    """Resume as métricas entre repetições conforme o contrato de reporting."""
    summary: list[dict[str, Any]] = []
    melted = metrics_by_repetition.melt(
        id_vars=list(KEY_COLUMNS),
        value_vars=list(METRICS),
        var_name="metric_name",
        value_name="metric_value",
    )
    grouped = melted.groupby(
        ["analysis_key", "model_name", "threshold_name", "metric_name"],
        sort=True,
    )
    for (analysis, model, threshold, metric), group in grouped:
        values = pd.to_numeric(group.metric_value, errors="coerce").dropna().to_numpy()
        summary.append(
            {
                "analysis_key": analysis,
                "model_name": model,
                "threshold_name": threshold,
                "metric_name": metric,
                "repetitions_expected": 100,
                "repetitions_valid": len(values),
                "repetitions_invalid": 100 - len(values),
                "mean": float(np.mean(values)) if len(values) else "",
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else "",
                "median": float(np.median(values)) if len(values) else "",
                "p2_5": float(np.percentile(values, 2.5)) if len(values) else "",
                "p97_5": float(np.percentile(values, 97.5)) if len(values) else "",
                "minimum": float(np.min(values)) if len(values) else "",
                "maximum": float(np.max(values)) if len(values) else "",
                "interval_label": "distribuicao descritiva entre repeticoes",
            }
        )
    return pd.DataFrame(summary)


def write_outputs(output_dir: Path, metrics: pd.DataFrame, counts: pd.DataFrame) -> None:
    """Escreve resultados por repetição e o resumo público agregado."""
    summary = summarize(metrics)
    write_csv(
        output_dir / "classification_metrics_by_repetition.csv",
        metrics.to_dict("records"),
        list(metrics.columns),
    )
    write_csv(
        output_dir / "classification_confusion_counts_by_repetition.csv",
        counts.to_dict("records"),
        list(counts.columns),
    )
    write_csv(
        output_dir / "classification_metrics_summary.csv",
        summary.to_dict("records"),
        list(summary.columns),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Frozen OOF CSV; repeat this argument to combine inputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    oof = load_oof(args.input)
    validate_oof(oof)
    metrics, counts = classification(oof)
    if len(counts) != 1400:
        raise ValueError(f"Expected 1400 confusion matrices, found {len(counts)}")
    write_outputs(args.output_dir, metrics, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
