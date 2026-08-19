"""Valida e agrega previsões fora da amostra de validação cruzada repetida."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - usado somente se statsmodels não estiver disponível
    sm = None

try:
    from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning
except Exception:  # pragma: no cover - compatibilidade com versões anteriores do statsmodels
    PerfectSeparationError = None
    PerfectSeparationWarning = None


OOF_AGGREGATION_SCHEMA_VERSION = "oof_aggregation_v2"
PRIMARY_BY_REPETITION = "PRIMARY_BY_REPETITION"
SECONDARY_PATIENT_AVERAGED_OOF = "SECONDARY_PATIENT_AVERAGED_OOF"
DEPRECATED_INVALID_FOR_FINAL_REPORTING = "DEPRECATED_INVALID_FOR_FINAL_REPORTING"
EPSILON = 1e-6
CALIBRATION_ABS_COEFFICIENT_WARNING_THRESHOLD = 10.0


class OOFStructureError(ValueError):
    """Indica violação da estrutura por participante nas previsões OOF repetidas."""


@dataclass(frozen=True)
class OOFColumnConfig:
    """Nomes de colunas usados pelos utilitários de agregação OOF."""

    horizon: str = "horizon_key"
    model: str = "model_name"
    repeat: str = "repeat_index"
    fold: str = "fold_index"
    patient_id: str = "id_original"
    target: str = "outcome_true"
    probability: str = "predicted_probability"


def _required_columns(config: OOFColumnConfig) -> list[str]:
    return [
        config.horizon,
        config.model,
        config.repeat,
        config.fold,
        config.patient_id,
        config.target,
        config.probability,
    ]


def _as_expected_id_mapping(
    expected_patient_ids: Mapping[tuple[object, object], Sequence[object]] | None,
) -> dict[tuple[object, object], set[object]]:
    if expected_patient_ids is None:
        return {}
    normalized: dict[tuple[object, object], set[object]] = {}
    for key, value in expected_patient_ids.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise OOFStructureError("Expected patient ID map keys must be (horizon_key, model_name).")
        ids = set(value)
        if not ids:
            raise OOFStructureError(f"Expected patient ID map has empty ID set for {key[0]}/{key[1]}.")
        normalized[key] = ids
    return normalized


def _metric_warning(existing: list[str], message: str) -> None:
    if message not in existing:
        existing.append(message)


def _first_five(values: set[object]) -> list[object]:
    return sorted(values, key=lambda value: str(value))[:5]


def _warning_code(prefix: str, warning_obj: warnings.WarningMessage) -> str:
    category_name = warning_obj.category.__name__
    message = str(warning_obj.message).lower()
    if "perfect separation" in message or (
        PerfectSeparationWarning is not None
        and issubclass(warning_obj.category, PerfectSeparationWarning)
    ):
        return f"{prefix}_separation"
    return f"{prefix}_warning:{category_name}"


def _is_singular_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return "singular" in message or "singular matrix" in message


def _fit_calibration_glm(
    *,
    y_true: np.ndarray,
    design: np.ndarray,
    prefix: str,
    offset: np.ndarray | None = None,
    coefficient_index: int = 0,
) -> tuple[float, list[str]]:
    warnings_list: list[str] = []
    if sm is None:
        return np.nan, ["statsmodels_unavailable"]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.GLM(
                y_true,
                design,
                family=sm.families.Binomial(),
                offset=offset,
            ).fit(disp=False)
        for warning_obj in caught:
            _metric_warning(warnings_list, _warning_code(prefix, warning_obj))
        if any(code == f"{prefix}_separation" for code in warnings_list):
            return np.nan, warnings_list
        converged = getattr(model, "converged", None)
        if converged is False:
            _metric_warning(warnings_list, f"{prefix}_not_converged")
            return np.nan, warnings_list
        coefficient = float(model.params[coefficient_index])
        if not np.isfinite(coefficient):
            _metric_warning(warnings_list, f"{prefix}_nonfinite")
            return np.nan, warnings_list
        if abs(coefficient) > CALIBRATION_ABS_COEFFICIENT_WARNING_THRESHOLD:
            _metric_warning(warnings_list, f"{prefix}_extreme")
        return coefficient, warnings_list
    except Exception as exc:  # pragma: no cover - o caminho numérico exato varia conforme statsmodels
        if PerfectSeparationError is not None and isinstance(exc, PerfectSeparationError):
            _metric_warning(warnings_list, f"{prefix}_separation")
        elif _is_singular_exception(exc):
            _metric_warning(warnings_list, f"{prefix}_singular")
        else:
            _metric_warning(warnings_list, f"{prefix}_failed:{type(exc).__name__}")
        return np.nan, warnings_list


def validate_oof_structure(
    oof_df: pd.DataFrame,
    *,
    expected_repeats: int,
    expected_patient_ids: Mapping[tuple[object, object], Sequence[object]] | None = None,
    config: OOFColumnConfig = OOFColumnConfig(),
) -> None:
    """Valida previsões OOF de validação cruzada repetida sem modificá-las.

    A unidade é horizonte, modelo e repetição. Cada participante deve aparecer
    exatamente uma vez por repetição após a reunião dos folds.
    """

    missing = [column for column in _required_columns(config) if column not in oof_df.columns]
    if missing:
        raise OOFStructureError("Missing required OOF columns: " + ", ".join(missing))
    if oof_df.empty:
        raise OOFStructureError("OOF predictions are empty.")

    target = pd.to_numeric(oof_df[config.target], errors="coerce")
    if target.isna().any() or ~target.isin([0, 1]).all():
        raise OOFStructureError("OOF target must be binary 0/1 without missing values.")

    probability = pd.to_numeric(oof_df[config.probability], errors="coerce")
    if probability.isna().any() or ~np.isfinite(probability.to_numpy(dtype=float)).all():
        raise OOFStructureError("OOF probabilities must be finite and non-missing.")
    if ((probability < 0) | (probability > 1)).any():
        raise OOFStructureError("OOF probabilities must be between 0 and 1.")

    duplicate_cols = [config.horizon, config.model, config.repeat, config.patient_id]
    if oof_df.duplicated(duplicate_cols).any():
        raise OOFStructureError("A patient has more than one OOF prediction in the same repeat.")

    id_mapping = _as_expected_id_mapping(expected_patient_ids)
    group_cols = [config.horizon, config.model]
    observed_keys = set(
        oof_df[group_cols].drop_duplicates().itertuples(index=False, name=None)
    )
    if id_mapping:
        missing_map_keys = observed_keys - set(id_mapping)
        if missing_map_keys:
            key = sorted(missing_map_keys, key=lambda value: str(value))[0]
            raise OOFStructureError(
                f"Observed OOF key {key[0]}/{key[1]} is missing from expected patient ID map."
            )
        missing_oof_keys = set(id_mapping) - observed_keys
        if missing_oof_keys:
            key = sorted(missing_oof_keys, key=lambda value: str(value))[0]
            raise OOFStructureError(
                f"Expected patient ID map key {key[0]}/{key[1]} has no OOF predictions."
            )

    for (horizon, model), hm_group in oof_df.groupby(group_cols, dropna=False):
        observed_repeats = set(pd.to_numeric(hm_group[config.repeat], errors="coerce").astype(int).tolist())
        expected_repeat_labels = set(range(1, expected_repeats + 1))
        if observed_repeats != expected_repeat_labels:
            raise OOFStructureError(
                f"{horizon}/{model} has repeat labels {sorted(observed_repeats)}; "
                f"expected 1..{expected_repeats}."
            )

        repeat_id_sets: list[set[object]] = []
        if id_mapping:
            expected_ids = id_mapping[(horizon, model)]
        else:
            expected_ids = set(hm_group[config.patient_id].unique().tolist())

        for repeat_index, repeat_group in hm_group.groupby(config.repeat, dropna=False):
            ids = set(repeat_group[config.patient_id].tolist())
            repeat_id_sets.append(ids)
            if ids != expected_ids:
                missing_ids = expected_ids - ids
                extra_ids = ids - expected_ids
                raise OOFStructureError(
                    f"{horizon}/{model}/repeat {repeat_index} has incomplete patient coverage; "
                    f"missing={_first_five(missing_ids)}, extra={_first_five(extra_ids)}."
                )
            fold_patient_dupes = repeat_group.duplicated([config.fold, config.patient_id]).any()
            if fold_patient_dupes:
                raise OOFStructureError(
                    f"{horizon}/{model}/repeat {repeat_index} has duplicate patient within a fold."
                )

        if len({frozenset(ids) for ids in repeat_id_sets}) != 1:
            raise OOFStructureError(f"{horizon}/{model} has non-constant patient IDs across repeats.")

    target_counts = (
        oof_df.groupby([config.horizon, config.model, config.patient_id], dropna=False)[config.target]
        .nunique(dropna=False)
    )
    if (target_counts > 1).any():
        raise OOFStructureError("Target is not constant for at least one patient across repeats.")


# Estima separadamente o intercepto e a inclinação de calibração.
def _calibration_metrics(y_true: np.ndarray, y_probability: np.ndarray) -> tuple[float, float, list[str]]:
    warnings_list: list[str] = []
    if sm is None:
        return np.nan, np.nan, ["statsmodels_unavailable"]
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, ["single_class_calibration_undefined"]

    p_clipped = np.clip(y_probability.astype(float), EPSILON, 1.0 - EPSILON)
    logits = np.log(p_clipped / (1.0 - p_clipped))

    intercept, intercept_warnings = _fit_calibration_glm(
        y_true=y_true,
        design=np.ones((len(y_true), 1)),
        offset=logits,
        prefix="calibration_intercept",
        coefficient_index=0,
    )
    slope, slope_warnings = _fit_calibration_glm(
        y_true=y_true,
        design=sm.add_constant(logits, has_constant="add"),
        offset=None,
        prefix="calibration_slope",
        coefficient_index=1,
    )
    warnings_list.extend(intercept_warnings)
    warnings_list.extend(slope_warnings)

    return intercept, slope, warnings_list


def compute_binary_oof_metrics(y_true: Sequence[object], y_probability: Sequence[object]) -> dict[str, object]:
    """Calcula métricas OOF binárias para uma unidade independente de análise."""

    y = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pd.Series(y_probability), errors="coerce").to_numpy(dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true and y_probability must have the same length.")
    if len(y) == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if np.isnan(y).any() or ~np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary 0/1 without missing values.")
    if np.isnan(p).any() or ~np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("y_probability must be finite, non-missing, and between 0 and 1.")

    y_int = y.astype(int)
    warnings_list: list[str] = []
    n = int(len(y_int))
    events = int(np.sum(y_int == 1))
    event_rate = float(events / n)

    roc_auc = np.nan
    average_precision = np.nan
    if len(np.unique(y_int)) < 2:
        _metric_warning(warnings_list, "single_class_discrimination_undefined")
    else:
        roc_auc = float(roc_auc_score(y_int, p))
        average_precision = float(average_precision_score(y_int, p))

    brier = float(brier_score_loss(y_int, p))
    calibration_intercept, calibration_slope, calibration_warnings = _calibration_metrics(y_int, p)
    warnings_list.extend(calibration_warnings)

    metric_status = "VALID" if not warnings_list else "VALID_WITH_WARNING"
    return {
        "n": n,
        "events": events,
        "event_rate": event_rate,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": brier,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "metric_status": metric_status,
        "warning_text": " | ".join(dict.fromkeys(warnings_list)),
    }


# Reúne os três folds para que a repetição completa seja a unidade de avaliação.
def compute_metrics_by_repetition(
    oof_df: pd.DataFrame,
    *,
    expected_repeats: int,
    expected_patient_ids: Mapping[tuple[object, object], Sequence[object]] | None = None,
    config: OOFColumnConfig = OOFColumnConfig(),
) -> pd.DataFrame:
    """Calcula as métricas principais uma vez por horizonte, modelo e repetição."""

    validate_oof_structure(
        oof_df,
        expected_repeats=expected_repeats,
        expected_patient_ids=expected_patient_ids,
        config=config,
    )
    rows: list[dict[str, object]] = []
    group_cols = [config.horizon, config.model, config.repeat]
    for keys, group in oof_df.groupby(group_cols, dropna=False):
        horizon, model, repeat_index = keys
        metrics = compute_binary_oof_metrics(group[config.target], group[config.probability])
        rows.append(
            {
                "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                "analysis_level": PRIMARY_BY_REPETITION,
                "horizon_key": horizon,
                "model_name": model,
                "repeat_index": repeat_index,
                "fold_count": int(group[config.fold].nunique()),
                "unique_patient_count": int(group[config.patient_id].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_key", "model_name", "repeat_index"]).reset_index(drop=True)


def summarize_metrics_across_repetitions(
    metrics_by_repetition: pd.DataFrame,
    *,
    expected_repeats: int,
) -> pd.DataFrame:
    """Resume a distribuição das métricas entre repetições da validação cruzada."""

    if expected_repeats <= 0:
        raise ValueError("expected_repeats must be positive.")
    duplicate_cols = ["horizon_key", "model_name", "repeat_index"]
    if metrics_by_repetition.duplicated(duplicate_cols).any():
        raise ValueError("Duplicate repeat_index detected for at least one horizon/model.")
    repeat_values = pd.to_numeric(metrics_by_repetition["repeat_index"], errors="coerce")
    if repeat_values.isna().any() or (repeat_values < 1).any() or (repeat_values > expected_repeats).any():
        raise ValueError(f"repeat_index must be within 1..{expected_repeats}.")

    metric_columns = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
    ]
    rows: list[dict[str, object]] = []
    for (horizon, model), group in metrics_by_repetition.groupby(["horizon_key", "model_name"], dropna=False):
        observed = int(group["repeat_index"].nunique())
        if observed > expected_repeats:
            raise ValueError(f"{horizon}/{model} has more repeats than expected.")
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.dropna()
            rows.append(
                {
                    "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                    "analysis_level": PRIMARY_BY_REPETITION,
                    "horizon_key": horizon,
                    "model_name": model,
                    "metric_name": metric,
                    "n_repeats_expected": int(expected_repeats),
                    "n_repeats_observed": observed,
                    "n_repeats_valid": int(valid.shape[0]),
                    "n_repeats_invalid": int(expected_repeats - valid.shape[0]),
                    "mean": float(valid.mean()) if not valid.empty else np.nan,
                    "standard_deviation": float(valid.std(ddof=1)) if valid.shape[0] > 1 else np.nan,
                    "median": float(valid.median()) if not valid.empty else np.nan,
                    "percentile_2_5": float(valid.quantile(0.025)) if not valid.empty else np.nan,
                    "percentile_97_5": float(valid.quantile(0.975)) if not valid.empty else np.nan,
                    "minimum": float(valid.min()) if not valid.empty else np.nan,
                    "maximum": float(valid.max()) if not valid.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_oof_by_patient(
    oof_df: pd.DataFrame,
    *,
    expected_repeats: int,
    expected_patient_ids: Mapping[tuple[object, object], Sequence[object]] | None = None,
    config: OOFColumnConfig = OOFColumnConfig(),
) -> pd.DataFrame:
    """Obtém a média das probabilidades OOF em uma linha secundária por participante."""

    validate_oof_structure(
        oof_df,
        expected_repeats=expected_repeats,
        expected_patient_ids=expected_patient_ids,
        config=config,
    )
    rows: list[dict[str, object]] = []
    group_cols = [config.horizon, config.model, config.patient_id]
    for (horizon, model, patient_id), group in oof_df.groupby(group_cols, dropna=False):
        if group[config.target].nunique(dropna=False) != 1:
            raise OOFStructureError(f"Target is inconsistent for patient {patient_id}.")
        probabilities = pd.to_numeric(group[config.probability], errors="coerce")
        rows.append(
            {
                "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                "analysis_level": SECONDARY_PATIENT_AVERAGED_OOF,
                "horizon_key": horizon,
                "model_name": model,
                "id_original": patient_id,
                "y_true": int(group[config.target].iloc[0]),
                "prediction_count": int(len(group)),
                "repeat_count": int(group[config.repeat].nunique()),
                "probability_mean": float(probabilities.mean()),
                "probability_standard_deviation": float(probabilities.std(ddof=1)),
                "probability_minimum": float(probabilities.min()),
                "probability_maximum": float(probabilities.max()),
                "target_consistent": True,
                "aggregation_status": "VALID",
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_key", "model_name", "id_original"]).reset_index(drop=True)


def compute_patient_average_metrics(patient_average_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas secundárias por horizonte e modelo com médias por participante."""

    required = ["horizon_key", "model_name", "y_true", "probability_mean"]
    missing = [column for column in required if column not in patient_average_df.columns]
    if missing:
        raise ValueError("Missing patient-average columns: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    for (horizon, model), group in patient_average_df.groupby(["horizon_key", "model_name"], dropna=False):
        metrics = compute_binary_oof_metrics(group["y_true"], group["probability_mean"])
        rows.append(
            {
                "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                "analysis_level": SECONDARY_PATIENT_AVERAGED_OOF,
                "horizon_key": horizon,
                "model_name": model,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_key", "model_name"]).reset_index(drop=True)


def build_oof_aggregation_qc(
    oof_df: pd.DataFrame,
    *,
    expected_repeats: int,
    expected_patient_ids: Mapping[tuple[object, object], Sequence[object]] | None = None,
    config: OOFColumnConfig = OOFColumnConfig(),
) -> pd.DataFrame:
    """Constrói o controle de qualidade da estrutura de agregação OOF repetida."""

    rows: list[dict[str, object]] = []
    id_mapping = _as_expected_id_mapping(expected_patient_ids)
    required_missing = [column for column in _required_columns(config) if column not in oof_df.columns]
    if required_missing:
        return pd.DataFrame(
            [
                {
                    "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                    "qc_level": "global",
                    "horizon_key": pd.NA,
                    "model_name": pd.NA,
                    "repeat_index": pd.NA,
                    "expected_patient_count": pd.NA,
                    "observed_patient_count": pd.NA,
                    "missing_expected_patient_count": pd.NA,
                    "extra_patient_count": pd.NA,
                    "missing_expected_patient_examples": pd.NA,
                    "extra_patient_examples": pd.NA,
                    "expected_repeat_count": expected_repeats,
                    "observed_repeat_count": pd.NA,
                    "repeat_labels_valid": False,
                    "duplicate_predictions": pd.NA,
                    "folds_observed": pd.NA,
                    "target_discordant_patients": pd.NA,
                    "invalid_probabilities": pd.NA,
                    "status": "FAIL",
                    "note": "Missing required columns: " + ", ".join(required_missing),
                }
            ]
        )

    observed_keys = set(
        oof_df[[config.horizon, config.model]].drop_duplicates().itertuples(index=False, name=None)
    )
    keys_to_report = set(id_mapping) | observed_keys if id_mapping else observed_keys
    for horizon, model in sorted(keys_to_report, key=lambda value: str(value)):
        hm_group = oof_df[(oof_df[config.horizon] == horizon) & (oof_df[config.model] == model)]
        expected_ids = id_mapping.get((horizon, model), set(hm_group[config.patient_id].unique().tolist()))
        if hm_group.empty:
            rows.append(
                {
                    "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                    "qc_level": "horizon_model",
                    "horizon_key": horizon,
                    "model_name": model,
                    "repeat_index": pd.NA,
                    "expected_patient_count": int(len(expected_ids)),
                    "observed_patient_count": 0,
                    "missing_expected_patient_count": int(len(expected_ids)),
                    "extra_patient_count": 0,
                    "missing_expected_patient_examples": "|".join(map(str, _first_five(expected_ids))),
                    "extra_patient_examples": "",
                    "expected_repeat_count": expected_repeats,
                    "observed_repeat_count": 0,
                    "repeat_labels_valid": False,
                    "duplicate_predictions": 0,
                    "folds_observed": 0,
                    "target_discordant_patients": 0,
                    "invalid_probabilities": 0,
                    "status": "FAIL",
                    "note": "expected_key_without_oof",
                }
            )
            continue
        repeats_observed = int(hm_group[config.repeat].nunique())
        expected_repeat_labels = set(range(1, expected_repeats + 1))
        observed_repeat_labels = set(pd.to_numeric(hm_group[config.repeat], errors="coerce").dropna().astype(int))
        repeat_labels_valid = observed_repeat_labels == expected_repeat_labels
        for repeat_index, repeat_group in hm_group.groupby(config.repeat, dropna=False):
            observed_ids = set(repeat_group[config.patient_id].tolist())
            duplicate_predictions = int(repeat_group.duplicated([config.patient_id]).sum())
            target_discordant = int(
                (hm_group.groupby(config.patient_id, dropna=False)[config.target].nunique(dropna=False) > 1).sum()
            )
            probabilities = pd.to_numeric(repeat_group[config.probability], errors="coerce")
            invalid_probabilities = int(
                probabilities.isna().sum()
                + (~np.isfinite(probabilities.fillna(0).to_numpy(dtype=float))).sum()
                + ((probabilities < 0) | (probabilities > 1)).sum()
            )
            missing_ids = expected_ids - observed_ids
            extra_ids = observed_ids - expected_ids
            status = "PASS"
            notes: list[str] = []
            if not repeat_labels_valid:
                status = "FAIL"
                notes.append("unexpected_repeat_count")
            if duplicate_predictions or target_discordant or invalid_probabilities or missing_ids or extra_ids:
                status = "FAIL"
            rows.append(
                {
                    "aggregation_schema_version": OOF_AGGREGATION_SCHEMA_VERSION,
                    "qc_level": "horizon_model_repeat",
                    "horizon_key": horizon,
                    "model_name": model,
                    "repeat_index": repeat_index,
                    "expected_patient_count": int(len(expected_ids)),
                    "observed_patient_count": int(len(observed_ids)),
                    "missing_expected_patient_count": int(len(missing_ids)),
                    "extra_patient_count": int(len(extra_ids)),
                    "missing_expected_patient_examples": "|".join(map(str, _first_five(missing_ids))),
                    "extra_patient_examples": "|".join(map(str, _first_five(extra_ids))),
                    "expected_repeat_count": expected_repeats,
                    "observed_repeat_count": repeats_observed,
                    "repeat_labels_valid": repeat_labels_valid,
                    "duplicate_predictions": duplicate_predictions,
                    "folds_observed": int(repeat_group[config.fold].nunique()),
                    "target_discordant_patients": target_discordant,
                    "invalid_probabilities": invalid_probabilities,
                    "status": status,
                    "note": " | ".join(notes),
                }
            )

    return pd.DataFrame(rows).sort_values(["horizon_key", "model_name", "repeat_index"]).reset_index(drop=True)
