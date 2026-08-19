"""Utilitários para registrar e validar diagnósticos dos ajustes de modelos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning as StatsmodelsConvergenceWarning,
        PerfectSeparationError,
        PerfectSeparationWarning,
    )
except Exception:  # pragma: no cover
    StatsmodelsConvergenceWarning = None
    PerfectSeparationError = None
    PerfectSeparationWarning = None


DIAGNOSTICS_SCHEMA_VERSION = "model_fit_diagnostics_v1"
ALLOWED_FIT_CONTEXTS = {
    "cv_fold",
    "full_sample_parsimonious_coefficients",
    "full_sample_ridge_coefficients",
    "full_sample_tree_rules",
    "fold_level_calibration_diagnostic",
}
ALLOWED_FIT_STATUS = {"VALID", "VALID_WITH_WARNING", "NONCONVERGED", "FAILED", "NOT_APPLICABLE"}
BLOCKING_WARNING_CODES = {"perfect_separation_warning", "perfect_separation_error"}
PREDICTIVE_FIT_CONTEXTS = {
    "cv_fold",
    "full_sample_parsimonious_coefficients",
    "full_sample_ridge_coefficients",
    "full_sample_tree_rules",
}
DIAGNOSTIC_UNIT_COLUMNS = ["horizon_key", "model_name", "fit_context", "repeat_index", "fold_index"]


@dataclass(frozen=True)
class FitDiagnostic:
    diagnostics_schema_version: str
    horizon_key: str
    horizon_label: str
    status_col: str
    model_name: str
    model_family: str
    fit_context: str
    repeat_index: object
    fold_index: object
    n_train: object
    n_events_train: object
    n_evaluation: object
    n_events_evaluation: object
    estimator_class: str
    fit_status: str
    converged: object
    usable_for_scientific_reporting: bool
    warning_count: int
    warning_codes: str
    warning_categories: str
    warning_text: str
    exception_type: str
    exception_text: str
    n_iter_observed: object
    max_iter_configured: object


class ModelFitDiagnosticError(RuntimeError):
    pass


def warning_code(category: type[Warning], message: str) -> str:
    text = str(message).lower()
    if issubclass(category, ConvergenceWarning):
        return "sklearn_convergence_warning"
    if StatsmodelsConvergenceWarning is not None and issubclass(category, StatsmodelsConvergenceWarning):
        return "statsmodels_convergence_warning"
    if PerfectSeparationWarning is not None and issubclass(category, PerfectSeparationWarning):
        return "perfect_separation_warning"
    if "perfect separation" in text:
        return "perfect_separation_warning"
    if issubclass(category, RuntimeWarning):
        if "overflow" in text:
            return "overflow_warning"
        return "runtime_warning"
    return f"other_warning:{category.__name__}"


def normalize_warnings(caught: Iterable[warnings.WarningMessage]) -> tuple[list[str], list[str], list[str]]:
    codes: list[str] = []
    categories: list[str] = []
    texts: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in caught:
        code = warning_code(item.category, str(item.message))
        category = item.category.__name__
        text = str(item.message)
        key = (code, category, text)
        if key not in seen:
            seen.add(key)
            codes.append(code)
            categories.append(category)
            texts.append(text)
    return codes, categories, texts


def _max_n_iter(estimator: Any) -> object:
    n_iter = getattr(estimator, "n_iter_", pd.NA)
    if n_iter is pd.NA:
        return pd.NA
    values = np.asarray(n_iter).ravel()
    if values.size == 0:
        return pd.NA
    return int(np.nanmax(values))


def _max_iter(estimator: Any) -> object:
    return getattr(estimator, "max_iter", pd.NA)


def _has_nonfinite(values: Sequence[Any]) -> bool:
    arr = np.asarray(values, dtype=float)
    return not np.all(np.isfinite(arr))


def safe_exponentiate_estimates(
    coefficients: Sequence[Any],
    lower_bounds: Sequence[Any] | None = None,
    upper_bounds: Sequence[Any] | None = None,
) -> dict[str, object]:
    """Exponentiate model estimates without allowing infinite scientific outputs."""

    inputs = {
        "odds_ratio": coefficients,
        "ci95_lower": lower_bounds,
        "ci95_upper": upper_bounds,
    }
    values: dict[str, list[object] | None] = {}
    warning_codes: list[str] = []
    warning_categories: list[str] = []
    warning_texts: list[str] = []

    for output_name, source_values in inputs.items():
        if source_values is None:
            values[output_name] = None
            continue
        arr = np.asarray(source_values, dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = np.exp(arr)
        codes, categories, texts = normalize_warnings(caught)
        warning_codes.extend(codes)
        warning_categories.extend(categories)
        warning_texts.extend(texts)
        if not np.all(np.isfinite(result)):
            warning_codes.append("nonfinite_derived_estimate")
            warning_texts.append(
                "Odds ratio or confidence interval became non-finite after exponentiation."
            )
            values[output_name] = [pd.NA for _ in result.ravel()]
        else:
            values[output_name] = [float(value) for value in result.ravel()]

    failed = "nonfinite_derived_estimate" in warning_codes
    diagnostic = {
        "fit_status": "FAILED" if failed else ("VALID_WITH_WARNING" if warning_codes else "VALID"),
        "converged": False if failed else True,
        "usable_for_scientific_reporting": not failed,
        "warning_codes": "|".join(dict.fromkeys(warning_codes)),
        "warning_categories": "|".join(dict.fromkeys(warning_categories)),
        "warning_text": " | ".join(dict.fromkeys(warning_texts)),
        "exception_type": "NonFiniteDerivedEstimate" if failed else "",
        "exception_text": (
            "Odds ratio or confidence interval became non-finite after exponentiation."
            if failed
            else ""
        ),
        "n_iter_observed": pd.NA,
        "max_iter_configured": pd.NA,
    }
    return {"values": values, "diagnostic": diagnostic}


# Classifica convergência, avisos e valores não finitos de cada ajuste.
def assess_estimator_fit(
    estimator: Any,
    caught_warnings: Sequence[warnings.WarningMessage],
    *,
    fit_context: str,
    estimator_class: str | None = None,
    exception: Exception | None = None,
    convergence_applicable: bool = True,
    parameter_values: Sequence[Any] | None = None,
) -> dict[str, object]:
    codes, categories, texts = normalize_warnings(caught_warnings)
    if exception is not None:
        exc_name = type(exception).__name__
        if PerfectSeparationError is not None and isinstance(exception, PerfectSeparationError):
            codes.append("perfect_separation_error")
            exc_name = "PerfectSeparationError"
        elif "singular" in str(exception).lower():
            codes.append("singular_matrix")
            exc_name = "SingularEstimate"
        else:
            codes.append("fit_exception")
        return {
            "fit_status": "FAILED",
            "converged": False if convergence_applicable else pd.NA,
            "usable_for_scientific_reporting": False,
            "warning_codes": "|".join(dict.fromkeys(codes)),
            "warning_categories": "|".join(dict.fromkeys(categories)),
            "warning_text": " | ".join(dict.fromkeys(texts)),
            "exception_type": exc_name,
            "exception_text": str(exception),
            "n_iter_observed": _max_n_iter(estimator) if estimator is not None else pd.NA,
            "max_iter_configured": _max_iter(estimator) if estimator is not None else pd.NA,
        }
    n_iter_observed = _max_n_iter(estimator)
    max_iter_configured = _max_iter(estimator)
    if any(code in BLOCKING_WARNING_CODES for code in codes):
        return {
            "fit_status": "FAILED",
            "converged": False if convergence_applicable else pd.NA,
            "usable_for_scientific_reporting": False,
            "warning_codes": "|".join(dict.fromkeys(codes)),
            "warning_categories": "|".join(dict.fromkeys(categories)),
            "warning_text": " | ".join(dict.fromkeys(texts)),
            "exception_type": "PerfectSeparationWarning",
            "exception_text": "Perfect separation warning captured during model fit.",
            "n_iter_observed": n_iter_observed,
            "max_iter_configured": max_iter_configured,
        }
    if parameter_values is not None and _has_nonfinite(parameter_values):
        codes.append("nonfinite_parameter")
        return {
            "fit_status": "FAILED",
            "converged": False if convergence_applicable else pd.NA,
            "usable_for_scientific_reporting": False,
            "warning_codes": "|".join(dict.fromkeys(codes)),
            "warning_categories": "|".join(dict.fromkeys(categories)),
            "warning_text": " | ".join(dict.fromkeys(texts)),
            "exception_type": "NonFiniteParameter",
            "exception_text": "At least one fitted parameter or inferential estimate was non-finite.",
            "n_iter_observed": n_iter_observed,
            "max_iter_configured": max_iter_configured,
        }
    nonconverged = any(code in {"sklearn_convergence_warning", "statsmodels_convergence_warning"} for code in codes)
    converged_attr = getattr(estimator, "converged", None)
    if converged_attr is False:
        nonconverged = True
    if max_iter_configured is not pd.NA and n_iter_observed is not pd.NA:
        try:
            if int(n_iter_observed) >= int(max_iter_configured):
                codes.append("max_iter_reached")
                nonconverged = True
        except Exception:
            pass
    if not convergence_applicable:
        status = "VALID_WITH_WARNING" if codes else "VALID"
        converged = pd.NA
    elif nonconverged:
        status = "NONCONVERGED"
        converged = False
    else:
        status = "VALID_WITH_WARNING" if codes else "VALID"
        converged = True
    return {
        "fit_status": status,
        "converged": converged,
        "usable_for_scientific_reporting": status in {"VALID", "VALID_WITH_WARNING"},
        "warning_codes": "|".join(dict.fromkeys(codes)),
        "warning_categories": "|".join(dict.fromkeys(categories)),
        "warning_text": " | ".join(dict.fromkeys(texts)),
        "exception_type": "",
        "exception_text": "",
        "n_iter_observed": n_iter_observed,
        "max_iter_configured": max_iter_configured,
    }


def build_fit_diagnostic(
    *,
    horizon_key: str,
    horizon_label: str,
    status_col: str,
    model_name: str,
    model_family: str,
    fit_context: str,
    repeat_index: object = pd.NA,
    fold_index: object = pd.NA,
    n_train: object = pd.NA,
    n_events_train: object = pd.NA,
    n_evaluation: object = pd.NA,
    n_events_evaluation: object = pd.NA,
    estimator_class: str,
    assessment: dict[str, object],
) -> dict[str, object]:
    warning_codes = str(assessment.get("warning_codes", ""))
    warning_count = 0 if not warning_codes else len(warning_codes.split("|"))
    diagnostic = FitDiagnostic(
        diagnostics_schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        horizon_key=horizon_key,
        horizon_label=horizon_label,
        status_col=status_col,
        model_name=model_name,
        model_family=model_family,
        fit_context=fit_context,
        repeat_index=repeat_index,
        fold_index=fold_index,
        n_train=n_train,
        n_events_train=n_events_train,
        n_evaluation=n_evaluation,
        n_events_evaluation=n_events_evaluation,
        estimator_class=estimator_class,
        fit_status=str(assessment["fit_status"]),
        converged=assessment["converged"],
        usable_for_scientific_reporting=bool(assessment["usable_for_scientific_reporting"]),
        warning_count=warning_count,
        warning_codes=warning_codes,
        warning_categories=str(assessment.get("warning_categories", "")),
        warning_text=str(assessment.get("warning_text", "")),
        exception_type=str(assessment.get("exception_type", "")),
        exception_text=str(assessment.get("exception_text", "")),
        n_iter_observed=assessment.get("n_iter_observed", pd.NA),
        max_iter_configured=assessment.get("max_iter_configured", pd.NA),
    )
    return asdict(diagnostic)


def _normalize_unit_value(value: object) -> object:
    if pd.isna(value):
        return "__NA__"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    except Exception:
        pass
    return str(value)


def _unit_set(df: pd.DataFrame) -> set[tuple[object, ...]]:
    return {
        tuple(_normalize_unit_value(row[column]) for column in DIAGNOSTIC_UNIT_COLUMNS)
        for _, row in df[DIAGNOSTIC_UNIT_COLUMNS].iterrows()
    }


def build_expected_diagnostic_units(
    *,
    horizon_config: Mapping[str, Mapping[str, object]],
    model_config: Mapping[str, Mapping[str, object]],
    expected_repeats: int,
    cv_n_splits_used: Mapping[tuple[str, str], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon_key in horizon_config:
        for model_name in model_config:
            n_splits = int(cv_n_splits_used[(horizon_key, model_name)])
            for repeat_index in range(1, expected_repeats + 1):
                for fold_index in range(1, n_splits + 1):
                    rows.append(
                        {
                            "horizon_key": horizon_key,
                            "model_name": model_name,
                            "fit_context": "cv_fold",
                            "repeat_index": repeat_index,
                            "fold_index": fold_index,
                        }
                    )
                    rows.append(
                        {
                            "horizon_key": horizon_key,
                            "model_name": model_name,
                            "fit_context": "fold_level_calibration_diagnostic",
                            "repeat_index": repeat_index,
                            "fold_index": fold_index,
                        }
                    )
    full_contexts = {
        "logistic_parsimonious": "full_sample_parsimonious_coefficients",
        "logistic_core_ridge": "full_sample_ridge_coefficients",
        "decision_tree_shallow": "full_sample_tree_rules",
    }
    for horizon_key in horizon_config:
        for model_name, fit_context in full_contexts.items():
            rows.append(
                {
                    "horizon_key": horizon_key,
                    "model_name": model_name,
                    "fit_context": fit_context,
                    "repeat_index": pd.NA,
                    "fold_index": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def _is_missing(value: object) -> bool:
    return bool(pd.isna(value))


def _validate_count_pair(row: pd.Series, n_col: str, events_col: str, context: str) -> None:
    if _is_missing(row[n_col]) or _is_missing(row[events_col]):
        raise ModelFitDiagnosticError(f"{context} requires {n_col} and {events_col}.")
    for column in [n_col, events_col]:
        value = row[column]
        try:
            numeric = float(value)
        except Exception as exc:
            raise ModelFitDiagnosticError(f"{column} must be numeric.") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ModelFitDiagnosticError(f"{column} must be an integer.")
        if column == n_col and numeric <= 0:
            raise ModelFitDiagnosticError(f"{column} must be positive.")
        if column == events_col and numeric < 0:
            raise ModelFitDiagnosticError(f"{column} cannot be negative.")
    if int(float(row[events_col])) > int(float(row[n_col])):
        raise ModelFitDiagnosticError(f"{events_col} cannot exceed {n_col}.")


def _validate_diagnostic_sample_metadata(diagnostics_df: pd.DataFrame) -> None:
    for _, row in diagnostics_df.iterrows():
        context = str(row["fit_context"])
        if context == "cv_fold":
            _validate_count_pair(row, "n_train", "n_events_train", context)
            _validate_count_pair(row, "n_evaluation", "n_events_evaluation", context)
        elif context == "fold_level_calibration_diagnostic":
            if not _is_missing(row["n_train"]) or not _is_missing(row["n_events_train"]):
                raise ModelFitDiagnosticError("Calibration diagnostics must not populate training counts.")
            _validate_count_pair(row, "n_evaluation", "n_events_evaluation", context)
        elif context in {
            "full_sample_parsimonious_coefficients",
            "full_sample_ridge_coefficients",
            "full_sample_tree_rules",
        }:
            _validate_count_pair(row, "n_train", "n_events_train", context)
            if not _is_missing(row["n_evaluation"]) or not _is_missing(row["n_events_evaluation"]):
                raise ModelFitDiagnosticError("Full-sample diagnostics must not populate evaluation counts.")


CALIBRATION_QC_REQUIRED_COLUMNS = [
    "diagnostics_schema_version",
    "horizon_key",
    "model_name",
    "n_calibration_fits_expected",
    "n_calibration_fits_observed",
    "n_calibration_valid",
    "n_calibration_valid_with_warning",
    "n_calibration_nonconverged",
    "n_calibration_failed",
    "n_intercepts_available",
    "n_intercepts_missing",
    "n_slopes_available",
    "n_slopes_missing",
    "coverage_status",
    "qc_status",
    "warning_text",
]
CALIBRATION_QC_STATUSES = {"PASS", "PASS_WITH_WARNING", "FAIL"}
EXPECTED_CALIBRATION_THRESHOLDS = {"threshold_0_5", "threshold_event_rate"}
INTERCEPT_MISSING_WARNING_CODES = {
    "calibration_intercept_failed",
    "calibration_intercept_nonfinite",
    "calibration_intercept_not_converged",
    "calibration_intercept_separation",
}
SLOPE_MISSING_WARNING_CODES = {
    "calibration_slope_failed",
    "calibration_slope_nonfinite",
    "calibration_slope_not_converged",
    "calibration_slope_separation",
}


def _values_identical_or_missing(series: pd.Series) -> bool:
    normalized = []
    for value in series.tolist():
        if pd.isna(value):
            normalized.append("__NA__")
        else:
            normalized.append(float(value))
    return len(set(normalized)) <= 1


def _is_available(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _deduplicate_calibration_metrics(cv_metrics_by_fold_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "horizon_key",
        "model_name",
        "repeat_index",
        "fold_index",
        "threshold_name",
        "calibration_intercept",
        "calibration_slope",
    ]
    missing = [column for column in required if column not in cv_metrics_by_fold_df.columns]
    if missing:
        raise ModelFitDiagnosticError("Missing calibration metric columns: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    warnings_found: list[str] = []
    for keys, group in cv_metrics_by_fold_df.groupby(
        ["horizon_key", "model_name", "repeat_index", "fold_index"],
        dropna=False,
    ):
        horizon_key, model_name, repeat_index, fold_index = keys
        threshold_names = set(group["threshold_name"].astype(str).tolist())
        intercept_identical = _values_identical_or_missing(group["calibration_intercept"])
        slope_identical = _values_identical_or_missing(group["calibration_slope"])
        if threshold_names != EXPECTED_CALIBRATION_THRESHOLDS:
            warnings_found.append(
                f"threshold_set_mismatch:{horizon_key}/{model_name}/repeat={repeat_index}/fold={fold_index}:"
                f"observed={sorted(threshold_names)}"
            )
        if len(group) != len(EXPECTED_CALIBRATION_THRESHOLDS):
            warnings_found.append(
                f"threshold_row_count_mismatch:{horizon_key}/{model_name}/repeat={repeat_index}/fold={fold_index}:"
                f"observed={len(group)}"
            )
        if not intercept_identical or not slope_identical:
            warnings_found.append(
                f"threshold_calibration_divergence:{horizon_key}/{model_name}/repeat={repeat_index}/fold={fold_index}"
            )
        rows.append(
            {
                "horizon_key": horizon_key,
                "model_name": model_name,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "calibration_intercept": group["calibration_intercept"].iloc[0],
                "calibration_slope": group["calibration_slope"].iloc[0],
                "threshold_rows": int(len(group)),
                "threshold_names": "|".join(sorted(threshold_names)),
                "threshold_values_identical": bool(intercept_identical and slope_identical),
            }
        )
    return pd.DataFrame(rows), warnings_found


def _calibration_metric_unit_set(df: pd.DataFrame) -> set[tuple[object, ...]]:
    return {
        (
            _normalize_unit_value(row["horizon_key"]),
            _normalize_unit_value(row["model_name"]),
            _normalize_unit_value(row["repeat_index"]),
            _normalize_unit_value(row["fold_index"]),
        )
        for _, row in df.iterrows()
    }


def _calibration_expected_metric_unit_set(expected: pd.DataFrame) -> set[tuple[object, ...]]:
    return _calibration_metric_unit_set(expected)


def _warning_code_set(row: pd.Series) -> set[str]:
    codes: set[str] = set()
    for column in ["warning_codes", "warning_text", "exception_type", "exception_text"]:
        value = str(row.get(column, ""))
        if not value:
            continue
        for separator in ["|", " "]:
            value = value.replace(separator, "|")
        codes.update(part.strip() for part in value.split("|") if part.strip())
    return codes


# Organiza o controle de qualidade dos diagnósticos de calibração.
def build_calibration_diagnostics_qc(
    model_fit_diagnostics_df: pd.DataFrame,
    cv_metrics_by_fold_df: pd.DataFrame,
    expected_diagnostic_units: pd.DataFrame,
) -> pd.DataFrame:
    calibration_units = ["horizon_key", "model_name", "repeat_index", "fold_index"]
    expected = expected_diagnostic_units[
        expected_diagnostic_units["fit_context"].eq("fold_level_calibration_diagnostic")
    ].copy()
    observed = model_fit_diagnostics_df[
        model_fit_diagnostics_df["fit_context"].eq("fold_level_calibration_diagnostic")
    ].copy()
    metrics, threshold_warnings = _deduplicate_calibration_metrics(cv_metrics_by_fold_df)
    expected_metric_units = _calibration_expected_metric_unit_set(expected[calibration_units])
    observed_metric_units = _calibration_metric_unit_set(metrics)
    missing_metric_units = expected_metric_units - observed_metric_units
    extra_metric_units = observed_metric_units - expected_metric_units

    rows: list[dict[str, object]] = []
    keys = sorted(
        set(expected[["horizon_key", "model_name"]].itertuples(index=False, name=None))
        | set(observed[["horizon_key", "model_name"]].itertuples(index=False, name=None))
        | set(metrics[["horizon_key", "model_name"]].itertuples(index=False, name=None)),
        key=lambda value: str(value),
    )
    for horizon_key, model_name in keys:
        expected_group = expected[(expected["horizon_key"] == horizon_key) & (expected["model_name"] == model_name)]
        observed_group = observed[(observed["horizon_key"] == horizon_key) & (observed["model_name"] == model_name)]
        metrics_group = metrics[(metrics["horizon_key"] == horizon_key) & (metrics["model_name"] == model_name)]
        expected_unit_set = _unit_set(expected_group.assign(fit_context="fold_level_calibration_diagnostic"))
        observed_unit_set = _unit_set(observed_group.assign(fit_context="fold_level_calibration_diagnostic"))
        missing_units = expected_unit_set - observed_unit_set
        extra_units = observed_unit_set - expected_unit_set
        warning_texts: list[str] = []
        if missing_units:
            warning_texts.append(f"missing_calibration_units={sorted(map(str, missing_units))[:5]}")
        if extra_units:
            warning_texts.append(f"extra_calibration_units={sorted(map(str, extra_units))[:5]}")
        group_missing_metric_units = [
            unit for unit in missing_metric_units if unit[0] == _normalize_unit_value(horizon_key) and unit[1] == _normalize_unit_value(model_name)
        ]
        group_extra_metric_units = [
            unit for unit in extra_metric_units if unit[0] == _normalize_unit_value(horizon_key) and unit[1] == _normalize_unit_value(model_name)
        ]
        if group_missing_metric_units:
            warning_texts.append(f"missing_metric_units={sorted(map(str, group_missing_metric_units))[:5]}")
        if group_extra_metric_units:
            warning_texts.append(f"extra_metric_units={sorted(map(str, group_extra_metric_units))[:5]}")
        unit_threshold_warnings = [
            warning
            for warning in threshold_warnings
            if f":{horizon_key}/{model_name}/" in warning
        ]
        warning_texts.extend(unit_threshold_warnings)

        merged = observed_group.merge(
            metrics_group,
            on=calibration_units,
            how="left",
            suffixes=("_diagnostic", "_metric"),
        )
        intercept_available = int(merged["calibration_intercept"].map(_is_available).sum()) if "calibration_intercept" in merged else 0
        slope_available = int(merged["calibration_slope"].map(_is_available).sum()) if "calibration_slope" in merged else 0
        observed_n = int(len(observed_group))
        intercept_missing = int(observed_n - intercept_available)
        slope_missing = int(observed_n - slope_available)

        for _, row in merged.iterrows():
            status = str(row["fit_status"])
            intercept_ok = _is_available(row.get("calibration_intercept", pd.NA))
            slope_ok = _is_available(row.get("calibration_slope", pd.NA))
            warning_codes = _warning_code_set(row)
            unit = f"{row['horizon_key']}/{row['model_name']}/repeat={row['repeat_index']}/fold={row['fold_index']}"
            if status == "VALID" and (not intercept_ok or not slope_ok):
                warning_texts.append(f"valid_calibration_missing_component:{unit}")
            if not intercept_ok and not (warning_codes & INTERCEPT_MISSING_WARNING_CODES):
                warning_texts.append(f"intercept_missing_without_component_diagnostic:{unit}")
            if not slope_ok and not (warning_codes & SLOPE_MISSING_WARNING_CODES):
                warning_texts.append(f"slope_missing_without_component_diagnostic:{unit}")
            if status in {"NONCONVERGED", "FAILED"} and intercept_ok and slope_ok:
                warning_texts.append(f"blocked_calibration_has_all_components:{unit}")

        status_counts = {
            "n_calibration_valid": int(observed_group["fit_status"].eq("VALID").sum()),
            "n_calibration_valid_with_warning": int(observed_group["fit_status"].eq("VALID_WITH_WARNING").sum()),
            "n_calibration_nonconverged": int(observed_group["fit_status"].eq("NONCONVERGED").sum()),
            "n_calibration_failed": int(observed_group["fit_status"].eq("FAILED").sum()),
        }
        coverage_status = "PASS" if not missing_units and not extra_units and observed_n == int(len(expected_group)) else "FAIL"
        arithmetic_ok = (
            sum(status_counts.values()) == observed_n
            and intercept_available + intercept_missing == observed_n
            and slope_available + slope_missing == observed_n
        )
        if not arithmetic_ok:
            warning_texts.append("calibration_qc_arithmetic_identity_failed")
        has_problem = bool(warning_texts) or not arithmetic_ok
        has_calibration_issue = any(value > 0 for key, value in status_counts.items() if key != "n_calibration_valid") or intercept_missing or slope_missing
        if coverage_status != "PASS" or has_problem:
            qc_status = "FAIL"
        elif has_calibration_issue:
            qc_status = "PASS_WITH_WARNING"
        else:
            qc_status = "PASS"
        rows.append(
            {
                "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "horizon_key": horizon_key,
                "model_name": model_name,
                "n_calibration_fits_expected": int(len(expected_group)),
                "n_calibration_fits_observed": observed_n,
                **status_counts,
                "n_intercepts_available": intercept_available,
                "n_intercepts_missing": intercept_missing,
                "n_slopes_available": slope_available,
                "n_slopes_missing": slope_missing,
                "coverage_status": coverage_status,
                "qc_status": qc_status,
                "warning_text": " | ".join(dict.fromkeys(warning_texts)),
            }
        )
    return pd.DataFrame(rows, columns=CALIBRATION_QC_REQUIRED_COLUMNS)


def validate_calibration_diagnostics_qc(calibration_qc_df: pd.DataFrame) -> None:
    missing = [column for column in CALIBRATION_QC_REQUIRED_COLUMNS if column not in calibration_qc_df.columns]
    if missing:
        raise ModelFitDiagnosticError("Missing calibration QC columns: " + ", ".join(missing))
    if calibration_qc_df.empty:
        raise ModelFitDiagnosticError("Calibration QC is empty.")
    if calibration_qc_df.duplicated(["horizon_key", "model_name"]).any():
        raise ModelFitDiagnosticError("Duplicate calibration QC horizon/model row.")
    if calibration_qc_df[["horizon_key", "model_name"]].isna().any().any():
        raise ModelFitDiagnosticError("Calibration QC horizon/model cannot be missing.")
    if ~calibration_qc_df["coverage_status"].isin({"PASS", "FAIL"}).all():
        raise ModelFitDiagnosticError("Unknown calibration coverage_status.")
    if ~calibration_qc_df["qc_status"].isin(CALIBRATION_QC_STATUSES).all():
        raise ModelFitDiagnosticError("Unknown calibration qc_status.")
    if calibration_qc_df["coverage_status"].ne("PASS").any():
        raise ModelFitDiagnosticError("Calibration QC coverage_status is not PASS.")
    if calibration_qc_df["qc_status"].eq("FAIL").any():
        raise ModelFitDiagnosticError("Calibration QC status is FAIL.")
    count_columns = [
        "n_calibration_fits_expected",
        "n_calibration_fits_observed",
        "n_calibration_valid",
        "n_calibration_valid_with_warning",
        "n_calibration_nonconverged",
        "n_calibration_failed",
        "n_intercepts_available",
        "n_intercepts_missing",
        "n_slopes_available",
        "n_slopes_missing",
    ]
    for column in count_columns:
        values = pd.to_numeric(calibration_qc_df[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values.to_numpy(dtype=float))).any():
            raise ModelFitDiagnosticError(f"{column} must be finite.")
        if (values < 0).any() or (values % 1 != 0).any():
            raise ModelFitDiagnosticError(f"{column} must be a non-negative integer.")
    for _, row in calibration_qc_df.iterrows():
        expected = int(row["n_calibration_fits_expected"])
        observed = int(row["n_calibration_fits_observed"])
        if expected != observed:
            raise ModelFitDiagnosticError("Calibration QC expected and observed counts differ.")
        status_sum = (
            int(row["n_calibration_valid"])
            + int(row["n_calibration_valid_with_warning"])
            + int(row["n_calibration_nonconverged"])
            + int(row["n_calibration_failed"])
        )
        if status_sum != observed:
            raise ModelFitDiagnosticError("Calibration QC status counts do not sum to observed.")
        if int(row["n_intercepts_available"]) + int(row["n_intercepts_missing"]) != observed:
            raise ModelFitDiagnosticError("Calibration QC intercept counts do not sum to observed.")
        if int(row["n_slopes_available"]) + int(row["n_slopes_missing"]) != observed:
            raise ModelFitDiagnosticError("Calibration QC slope counts do not sum to observed.")


def validate_model_fit_diagnostics(
    diagnostics_df: pd.DataFrame,
    *,
    expected_diagnostic_units: pd.DataFrame | None = None,
    allow_calibration_failures: bool = False,
) -> None:
    required = list(FitDiagnostic.__dataclass_fields__)
    missing = [column for column in required if column not in diagnostics_df.columns]
    if missing:
        raise ModelFitDiagnosticError("Missing diagnostic columns: " + ", ".join(missing))
    if ~diagnostics_df["fit_context"].isin(ALLOWED_FIT_CONTEXTS).all():
        raise ModelFitDiagnosticError("Invalid fit_context value.")
    if ~diagnostics_df["fit_status"].isin(ALLOWED_FIT_STATUS).all():
        raise ModelFitDiagnosticError("Invalid fit_status value.")
    _validate_diagnostic_sample_metadata(diagnostics_df)
    duplicate_cols = ["horizon_key", "model_name", "fit_context", "repeat_index", "fold_index"]
    if diagnostics_df.duplicated(duplicate_cols).any():
        raise ModelFitDiagnosticError("Duplicate fit diagnostic unit detected.")
    if expected_diagnostic_units is not None:
        missing_expected_cols = [
            column for column in DIAGNOSTIC_UNIT_COLUMNS if column not in expected_diagnostic_units.columns
        ]
        if missing_expected_cols:
            raise ModelFitDiagnosticError(
                "Missing expected diagnostic unit columns: " + ", ".join(missing_expected_cols)
            )
        expected_units = _unit_set(expected_diagnostic_units)
        observed_units = _unit_set(diagnostics_df)
        missing_units = expected_units - observed_units
        extra_units = observed_units - expected_units
        if missing_units or extra_units:
            missing_text = sorted(map(str, missing_units))[:5]
            extra_text = sorted(map(str, extra_units))[:5]
            raise ModelFitDiagnosticError(
                "Diagnostic coverage mismatch. "
                f"missing_first5={missing_text}; extra_first5={extra_text}."
            )
    predictive_mask = diagnostics_df["fit_context"].isin(PREDICTIVE_FIT_CONTEXTS)
    failed_mask = diagnostics_df["fit_status"].eq("FAILED")
    if failed_mask.any() and not allow_calibration_failures:
        raise ModelFitDiagnosticError("FAILED fit diagnostic present.")
    if (failed_mask & predictive_mask).any():
        raise ModelFitDiagnosticError("FAILED predictive fit diagnostic present.")
    logistic_mask = diagnostics_df["model_name"].isin(["logistic_parsimonious", "logistic_core_ridge"])
    if (diagnostics_df.loc[logistic_mask & predictive_mask, "fit_status"] == "NONCONVERGED").any():
        raise ModelFitDiagnosticError("NONCONVERGED logistic fit diagnostic present.")
    bad_usable = diagnostics_df["fit_status"].isin(["FAILED", "NONCONVERGED"]) & diagnostics_df[
        "usable_for_scientific_reporting"
    ].astype(bool)
    if bad_usable.any():
        raise ModelFitDiagnosticError("Non-usable status marked usable.")
    for _, row in diagnostics_df.iterrows():
        text = str(row["warning_text"])
        codes = str(row["warning_codes"])
        count = int(row["warning_count"])
        expected_count = 0 if not codes else len(codes.split("|"))
        if count != expected_count:
            raise ModelFitDiagnosticError("warning_count is inconsistent with warning_codes.")
        if count and not codes:
            raise ModelFitDiagnosticError("warning_count is positive but warning_codes is empty.")
        if row["fit_status"] == "FAILED" and not str(row["exception_type"]):
            raise ModelFitDiagnosticError("FAILED diagnostic lacks exception_type.")
        if row["fit_status"] != "FAILED" and str(row["exception_type"]):
            raise ModelFitDiagnosticError("Non-FAILED diagnostic has exception_type.")
        if row["fit_status"] == "VALID" and (count or codes or text):
            raise ModelFitDiagnosticError("VALID diagnostic contains warning evidence.")
        if row["fit_status"] in {"FAILED", "NONCONVERGED"} and bool(row["usable_for_scientific_reporting"]):
            raise ModelFitDiagnosticError("Blocked diagnostic marked usable.")
        if row["fit_status"] == "NONCONVERGED" and bool(row["converged"]) is not False:
            raise ModelFitDiagnosticError("NONCONVERGED diagnostic must have converged=False.")
