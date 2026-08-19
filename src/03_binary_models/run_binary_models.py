#!/usr/bin/env python3
"""Ajusta as especificações binárias predefinidas e produz previsões OOF."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src" / "04_internal_validation"))
from oof_aggregation_utils import (  # noqa: E402
    compute_metrics_by_repetition,
    summarize_metrics_across_repetitions,
)

CV_FOLDS = 3
CV_REPEATS = 100
RANDOM_STATE = 42
EPSILON = 1e-6
CORE = ("age_gt40_main", "dgf_main", "sex_male", "race3_clean", "donor_deceased")


@dataclass(frozen=True)
class ModelSpec:
    analysis: str
    input_file: str
    outcome: str
    model: str
    predictors: tuple[str, ...]
    expected_n: int
    expected_events: int


SPECS = (
    ModelSpec("2y", "base_b_2y_observed.csv", "y2_status", "logistic_parsimonious", ("age_gt40_main", "dgf_main"), 115, 18),
    ModelSpec("2y", "base_b_2y_observed.csv", "y2_status", "logistic_core_ridge", CORE, 114, 18),
    ModelSpec("2y", "base_b_2y_observed.csv", "y2_status", "decision_tree_shallow", CORE, 114, 18),
    ModelSpec("1y", "base_b_1y_observed.csv", "y1_status", "logistic_parsimonious", ("age_gt40_main", "dgf_main"), 153, 15),
    ModelSpec("1y", "base_b_1y_observed.csv", "y1_status", "logistic_core_ridge", CORE, 151, 15),
    ModelSpec("2y_landmark", "base_b_2y_landmark_day7_observed.csv", "y2_landmark_status", "logistic_parsimonious", ("age_gt40_main", "dgf_main"), 111, 14),
    ModelSpec("2y_landmark", "base_b_2y_landmark_day7_observed.csv", "y2_landmark_status", "logistic_core_ridge", CORE, 110, 14),
)


def frame(data_dir: Path, spec: ModelSpec) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(data_dir / spec.input_file)
    cols = [spec.outcome, "id_original", *spec.predictors]
    complete = df[cols].dropna().copy()
    y = pd.to_numeric(complete[spec.outcome]).astype(int)
    if len(complete) != spec.expected_n or int(y.sum()) != spec.expected_events:
        raise ValueError(f"Population mismatch for {spec.analysis}/{spec.model}")
    return complete[list(spec.predictors)], y, complete["id_original"]


def preprocessor(spec: ModelSpec) -> ColumnTransformer:
    numeric = [p for p in spec.predictors if p != "race3_clean"]
    transformers: list[tuple[str, object, list[str]]] = [("num", "passthrough", numeric)]
    if "race3_clean" in spec.predictors:
        drop = "first" if spec.model == "logistic_core_ridge" else None
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False), ["race3_clean"]))
    return ColumnTransformer(transformers, remainder="drop")


def estimator(spec: ModelSpec) -> object:
    """Constrói logística parcimoniosa, ridge fixa ou árvore rasa sem tuning."""
    if spec.model == "logistic_parsimonious":
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    if spec.model == "logistic_core_ridge":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000)
    if spec.model == "decision_tree_shallow" and spec.analysis == "2y":
        return DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42,
        )
    raise ValueError(f"Noncanonical model request: {spec.analysis}/{spec.model}")


def pipeline(spec: ModelSpec) -> Pipeline:
    return Pipeline([("preprocess", preprocessor(spec)), ("model", estimator(spec))])


def inferential_logit_rows(spec: ModelSpec, x: pd.DataFrame, y: pd.Series) -> list[dict[str, object]]:
    if spec.model != "logistic_parsimonious":
        return []
    design = sm.add_constant(x.astype(float), has_constant="add")
    fit = sm.Logit(y, design).fit(disp=False, maxiter=5000)
    ci = fit.conf_int()
    rows = []
    for term in fit.params.index:
        rows.append(
            {
                "analysis": spec.analysis,
                "model": spec.model,
                "term": "Intercept" if term == "const" else term,
                "n": len(y),
                "events": int(y.sum()),
                "coefficient": float(fit.params[term]),
                "odds_ratio": float(np.exp(fit.params[term])),
                "ci95_lower": float(np.exp(ci.loc[term, 0])),
                "ci95_upper": float(np.exp(ci.loc[term, 1])),
                "p_value": float(fit.pvalues[term]),
            }
        )
    return rows


def run_oof(spec: ModelSpec, x: pd.DataFrame, y: pd.Series, ids: pd.Series) -> list[dict[str, object]]:
    """Gera previsões fora da amostra conforme folds e repetições predefinidos."""
    splitter = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    rows: list[dict[str, object]] = []
    template = pipeline(spec)
    for split_index, (train, test) in enumerate(splitter.split(x, y), start=1):
        repeat = ((split_index - 1) // CV_FOLDS) + 1
        fold = ((split_index - 1) % CV_FOLDS) + 1
        # Preserva em cada previsão a prevalência do fold que treinou o modelo.
        train_event_rate = float(y.iloc[train].mean())
        fitted = clone(template).fit(x.iloc[train], y.iloc[train])
        probability = np.clip(fitted.predict_proba(x.iloc[test])[:, 1], EPSILON, 1 - EPSILON)
        for position, row_index in enumerate(test):
            rows.append(
                {
                    "horizon_key": spec.analysis,
                    "model_name": spec.model,
                    "repeat_index": repeat,
                    "fold_index": fold,
                    "id_original": ids.iloc[row_index],
                    "outcome_true": int(y.iloc[row_index]),
                    "predicted_probability": float(probability[position]),
                    "train_event_rate": train_event_rate,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "outputs" / "generated" / "binary_models")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coefficient_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for spec in SPECS:
        x, y, ids = frame(args.data_dir, spec)
        coefficient_rows.extend(inferential_logit_rows(spec, x, y))
        oof_rows.extend(run_oof(spec, x, y, ids))

    coefficients = pd.DataFrame(coefficient_rows)
    oof = pd.DataFrame(oof_rows)
    by_repetition = compute_metrics_by_repetition(oof, expected_repeats=CV_REPEATS)
    summary = summarize_metrics_across_repetitions(by_repetition, expected_repeats=CV_REPEATS)
    coefficients.to_csv(args.output_dir / "logistic_coefficients.csv", index=False)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    by_repetition.to_csv(args.output_dir / "oof_metrics_by_repetition.csv", index=False)
    summary.to_csv(args.output_dir / "oof_metrics_summary_repetition.csv", index=False)


if __name__ == "__main__":
    main()
