"""Executa a correção de otimismo por bootstrap para as análises predefinidas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MASTER_SEED = 42
CLIP_EPS = 1e-6
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load_analytical_dependencies() -> None:
    """Importa as dependências científicas somente nos modos analíticos."""
    global np, pd, sm
    global ColumnTransformer, ConvergenceWarning, LogisticRegression
    global average_precision_score, brier_score_loss, roc_auc_score
    global Pipeline, OneHotEncoder, DecisionTreeClassifier, export_text

    import numpy as np_module
    import pandas as pd_module
    import statsmodels.api as sm_module
    from sklearn.compose import ColumnTransformer as ColumnTransformerClass
    from sklearn.exceptions import ConvergenceWarning as ConvergenceWarningClass
    from sklearn.linear_model import LogisticRegression as LogisticRegressionClass
    from sklearn.metrics import average_precision_score as average_precision_score_fn
    from sklearn.metrics import brier_score_loss as brier_score_loss_fn
    from sklearn.metrics import roc_auc_score as roc_auc_score_fn
    from sklearn.pipeline import Pipeline as PipelineClass
    from sklearn.preprocessing import OneHotEncoder as OneHotEncoderClass
    from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierClass
    from sklearn.tree import export_text as export_text_fn

    np = np_module
    pd = pd_module
    sm = sm_module
    ColumnTransformer = ColumnTransformerClass
    ConvergenceWarning = ConvergenceWarningClass
    LogisticRegression = LogisticRegressionClass
    average_precision_score = average_precision_score_fn
    brier_score_loss = brier_score_loss_fn
    roc_auc_score = roc_auc_score_fn
    Pipeline = PipelineClass
    OneHotEncoder = OneHotEncoderClass
    DecisionTreeClassifier = DecisionTreeClassifierClass
    export_text = export_text_fn


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    order: int
    family: str
    model_key: str
    source_file: str
    outcome_col: str
    time_col: Optional[str]
    predictors: Tuple[str, ...]
    expected_n: int
    expected_events: int
    horizon_key: str = ""


ANALYSES: Tuple[AnalysisSpec, ...] = (
    AnalysisSpec("cox_fullfu_transplant", 1, "cox", "cox_parsimonious", "data/processed/base_s_survival_fullfu.csv", "event_death_fullfu", "followup_days_fullfu", ("age_gt40_main", "dgf_main"), 192, 22),
    AnalysisSpec("logistic_2y_current", 2, "logistic", "logistic_parsimonious", "data/processed/base_b_2y_observed.csv", "y2_status", None, ("age_gt40_main", "dgf_main"), 115, 18, "2y"),
    AnalysisSpec("ridge_2y_current", 3, "ridge", "logistic_core_ridge", "data/processed/base_b_2y_observed.csv", "y2_status", None, ("age_gt40_main", "dgf_main", "sex_male", "race3_clean", "donor_deceased"), 114, 18, "2y"),
    AnalysisSpec("logistic_1y_current", 4, "logistic", "logistic_parsimonious", "data/processed/base_b_1y_observed.csv", "y1_status", None, ("age_gt40_main", "dgf_main"), 153, 15, "1y"),
    AnalysisSpec("ridge_1y_current", 5, "ridge", "logistic_core_ridge", "data/processed/base_b_1y_observed.csv", "y1_status", None, ("age_gt40_main", "dgf_main", "sex_male", "race3_clean", "donor_deceased"), 151, 15, "1y"),
    AnalysisSpec("tree_2y_current_structure", 6, "tree", "decision_tree_shallow", "data/processed/base_b_2y_observed.csv", "y2_status", None, ("age_gt40_main", "dgf_main", "sex_male", "race3_clean", "donor_deceased"), 114, 18, "2y"),
    AnalysisSpec("cox_landmark_day7_fullfu", 7, "cox", "cox_parsimonious_landmark", "data/processed/base_s_landmark_day7_fullfu.csv", "event_after_landmark", "landmark_followup_days", ("age_gt40_main", "dgf_main"), 180, 18),
    AnalysisSpec("logistic_2y_landmark", 8, "logistic", "logistic_parsimonious_landmark", "data/processed/base_b_2y_landmark_day7_observed.csv", "y2_landmark_status", None, ("age_gt40_main", "dgf_main"), 111, 14),
    AnalysisSpec("ridge_2y_landmark", 9, "ridge", "logistic_core_ridge_landmark", "data/processed/base_b_2y_landmark_day7_observed.csv", "y2_landmark_status", None, ("age_gt40_main", "dgf_main", "sex_male", "race3_clean", "donor_deceased"), 110, 14),
)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve_under(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path must be relative and traversal-free: {relative_path}")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if not _is_within(candidate, root_resolved):
        raise ValueError(f"Resolved path escapes its declared root: {relative_path}")
    return candidate


def _validated_roots(data_root: Path, reference_root: Path, output_dir: Path, package_root: Path = PACKAGE_ROOT) -> Tuple[Path, Path, Path]:
    data = data_root.resolve()
    reference = reference_root.resolve()
    output = output_dir.resolve()
    release = package_root.resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"DATA_ROOT is not a directory: {data}")
    if not reference.is_dir():
        raise FileNotFoundError(f"REFERENCE_ROOT is not a directory: {reference}")
    if reference != (release / "outputs/reference").resolve():
        raise ValueError("REFERENCE_ROOT must be this release's outputs/reference directory")
    if _is_within(data, release) or _is_within(release, data):
        raise ValueError("DATA_ROOT must be outside the clean release")
    protected = (data, reference, release)
    for root in protected:
        if _is_within(output, root) or _is_within(root, output):
            raise ValueError(f"OUTPUT_DIR must be separate from protected root: {root}")
    if _is_within(data, reference) or _is_within(reference, data):
        raise ValueError("DATA_ROOT and REFERENCE_ROOT must be separate")
    return data, reference, output


def _read_csv_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(next(csv.reader(handle)))


def _read_manifest_hashes(package_root: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    manifest = package_root / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        rows[relative.strip().removeprefix("./")] = digest
    return rows


def _validate_resample_reference(path: Path) -> Dict[str, Any]:
    allowed = ["analysis_id", "attempt_id", "resample_index_sha256"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != allowed:
            raise ValueError(f"Unexpected resample reference schema: {reader.fieldnames}")
        rows = list(reader)
    canonical_ids = {spec.analysis_id for spec in ANALYSES}
    keys = {(row["analysis_id"], row["attempt_id"]) for row in rows}
    if len(rows) != 900 or len(keys) != 900:
        raise ValueError("Resample reference must contain 900 rows and 900 unique keys")
    if {row["analysis_id"] for row in rows} != canonical_ids:
        raise ValueError("Resample reference analysis IDs do not match ANALYSES")
    for analysis_id in canonical_ids:
        attempts = sorted(int(row["attempt_id"]) for row in rows if row["analysis_id"] == analysis_id)
        if attempts != list(range(1, 101)):
            raise ValueError(f"Resample attempts must be exactly 1..100 for {analysis_id}")
    if not all(re.fullmatch(r"[0-9a-f]{64}", row["resample_index_sha256"]) for row in rows):
        raise ValueError("Invalid SHA-256 fingerprint in resample reference")
    return {"rows": len(rows), "unique_keys": len(keys), "analysis_count": len(canonical_ids)}


def _validate_coefficient_coverage(path: Path) -> Dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"analysis_id", "term", "original_estimate"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Canonical coefficient reference lacks required columns")
        keys = {(row["analysis_id"], row["term"]) for row in reader}
    expected = set()
    for spec in ANALYSES:
        if spec.family == "tree":
            continue
        terms = list(spec.predictors)
        if spec.family in {"logistic", "ridge"}:
            terms = ["Intercept", *terms]
            if "race3_clean" in terms:
                terms.remove("race3_clean")
                terms.extend(["race3_clean=Mixed ethnicity/Asian", "race3_clean=White"])
        expected.update((spec.analysis_id, term) for term in terms)
    missing = sorted(expected - keys)
    if missing:
        raise ValueError(f"Canonical coefficient reference coverage is incomplete: {missing}")
    return {"expected_keys": len(expected), "covered_keys": len(expected)}


def run_preflight(data_root: Path, reference_root: Path, output_dir: Path, package_root: Path = PACKAGE_ROOT) -> Dict[str, Any]:
    """Pure dependency preflight: no model fitting, metrics, RNG, or R calls."""
    data, reference, output = _validated_roots(data_root, reference_root, output_dir, package_root=package_root)
    contract_path = package_root / "data" / "expected_files.csv"
    with contract_path.open(newline="", encoding="utf-8") as handle:
        contract = list(csv.DictReader(handle))
    checked_data = []
    for row in contract:
        candidate = _resolve_under(data, row["path"])
        if not candidate.is_file():
            raise FileNotFoundError(f"Expected data file is missing: {row['path']}")
        observed = sha256_file(candidate)
        if observed != row["expected_sha256"]:
            raise ValueError(f"Data hash mismatch: {row['path']}")
        checked_data.append(row["path"])

    schema_by_file: Dict[str, set[str]] = {}
    for spec in ANALYSES:
        required = {"id", spec.outcome_col, *spec.predictors}
        if spec.time_col:
            required.add(spec.time_col)
        schema_by_file.setdefault(spec.source_file, set()).update(required)
    declared = {row["path"] for row in contract}
    for relative, required in schema_by_file.items():
        if relative not in declared:
            continue
        header = set(_read_csv_header(_resolve_under(data, relative)))
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"Missing required columns in {relative}: {missing}")

    coefficient_path = _resolve_under(reference, "metrics/original_apparent_coefficients.csv")
    resample_path = _resolve_under(reference, "diagnostics/bootstrap_resample_hash_reference_1_100.csv")
    manifest_hashes = _read_manifest_hashes(package_root)
    for candidate in (coefficient_path, resample_path):
        relative = candidate.relative_to(package_root.resolve()).as_posix()
        if relative not in manifest_hashes or sha256_file(candidate) != manifest_hashes[relative]:
            raise ValueError(f"Reference is absent from or mismatched with MANIFEST.sha256: {relative}")
    coefficient = _validate_coefficient_coverage(coefficient_path)
    resample = _validate_resample_reference(resample_path)

    output.mkdir(parents=True, exist_ok=True)
    probe = output / ".phase4_preflight_write_probe"
    probe.write_text("preflight\n", encoding="utf-8")
    probe.unlink()
    return {
        "status": "PASS",
        "data_files_checked": len(checked_data),
        "coefficient_reference": coefficient,
        "resample_reference": resample,
        "analytical_calls": 0,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def rng_for(spec: AnalysisSpec, attempt_id: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER_SEED, spec.order, attempt_id])))


def load_population(data_root: Path, spec: AnalysisSpec) -> pd.DataFrame:
    cols = ["id", spec.outcome_col, *spec.predictors]
    if spec.time_col:
        cols.append(spec.time_col)
    return pd.read_csv(_resolve_under(data_root, spec.source_file))[cols].dropna().sort_values("id").reset_index(drop=True)


def population_registries(data_root: Path) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    pops: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for spec in ANALYSES:
        pop = load_population(data_root, spec)
        pops[spec.analysis_id] = pop
        events = int(pd.to_numeric(pop[spec.outcome_col]).sum())
        ids_hash = sha256_bytes("\n".join(map(str, pop["id"].tolist())).encode("utf-8"))
        status = "VALID" if len(pop) == spec.expected_n and events == spec.expected_events else "POPULATION_MISMATCH"
        rows.append({
            "analysis_id": spec.analysis_id,
            "source_file": spec.source_file,
            "n": len(pop),
            "events": events,
            "expected_n": spec.expected_n,
            "expected_events": spec.expected_events,
            "id_set_sha256": ids_hash,
            "previous_id_set_sha256": "NOT_REQUIRED_IN_CLEAN_PACKAGE",
            "status": status,
        })
    return pops, rows


def race_levels(pop: pd.DataFrame) -> List[str]:
    return sorted(pop["race3_clean"].astype(str).unique().tolist()) if "race3_clean" in pop else []


def preprocessor(spec: AnalysisSpec, levels: List[str], tree: bool = False) -> ColumnTransformer:
    numeric = [p for p in spec.predictors if p != "race3_clean"]
    transformers: List[Tuple[str, Any, List[str]]] = []
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    if "race3_clean" in spec.predictors:
        transformers.append(("cat", OneHotEncoder(categories=[levels], drop=None if tree else "first", handle_unknown="ignore", sparse_output=False), ["race3_clean"]))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def feature_names(pipe: Pipeline) -> List[str]:
    return [name.replace("num__", "").replace("cat__race3_clean_", "race3_clean=") for name in pipe.named_steps["preprocess"].get_feature_names_out()]


def fit_logit(pop: pd.DataFrame, spec: AnalysisSpec) -> Dict[str, Any]:
    x = sm.add_constant(pop[list(spec.predictors)], has_constant="add")
    y = pd.to_numeric(pop[spec.outcome_col]).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sm.Logit(y, x).fit(disp=False, maxiter=5000)
    params = {"Intercept" if k == "const" else k: float(v) for k, v in res.params.to_dict().items()}
    return {"status": "VALID" if res.mle_retvals.get("converged") else "NONCONVERGENCE", "params": params, "warnings": [str(w.message) for w in caught], "pred": np.asarray(res.predict(x))}


def fit_ridge(pop: pd.DataFrame, spec: AnalysisSpec, levels: Optional[List[str]] = None) -> Dict[str, Any]:
    levels = levels or race_levels(pop)
    pipe = Pipeline([("preprocess", preprocessor(spec, levels, tree=False)), ("model", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(pop[list(spec.predictors)], pd.to_numeric(pop[spec.outcome_col]).astype(int))
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    params = {"Intercept": float(model.intercept_[0])}
    params.update({n: float(v) for n, v in zip(names, model.coef_[0])})
    return {"status": "VALID_WITH_WARNING" if caught else "VALID", "params": params, "warnings": [str(w.message) for w in caught], "pred": pipe.predict_proba(pop[list(spec.predictors)])[:, 1], "pipe": pipe, "features": names}


def fit_tree(pop: pd.DataFrame, spec: AnalysisSpec, levels: Optional[List[str]] = None) -> Dict[str, Any]:
    levels = levels or race_levels(pop)
    pipe = Pipeline([("preprocess", preprocessor(spec, levels, tree=True)), ("model", DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=20, class_weight="balanced", random_state=42))])
    pipe.fit(pop[list(spec.predictors)], pd.to_numeric(pop[spec.outcome_col]).astype(int))
    names = feature_names(pipe)
    model = pipe.named_steps["model"]
    rules = export_text(model, feature_names=names)
    return {"status": "VALID", "pred": pipe.predict_proba(pop[list(spec.predictors)])[:, 1], "pipe": pipe, "features": names, "rules": rules, "root": names[model.tree_.feature[0]] if model.tree_.feature[0] >= 0 else "", "depth": model.get_depth(), "leaves": model.get_n_leaves(), "nodes": model.tree_.node_count}


def cox_fit_via_r(pop: pd.DataFrame, spec: AnalysisSpec, output_dir: Path, label: str) -> Dict[str, Any]:
    tmpdir = output_dir / "sensitive" / "cox_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    data_path = tmpdir / f"{label}.csv"
    out_path = tmpdir / f"{label}_out.json"
    pop[[spec.time_col, spec.outcome_col, *spec.predictors]].to_csv(data_path, index=False)
    r = f"""
    suppressPackageStartupMessages(library(survival))
    d <- read.csv({json.dumps(str(data_path))})
    fit <- tryCatch(coxph(Surv({spec.time_col}, {spec.outcome_col}) ~ age_gt40_main + dgf_main, ties='efron', data=d, x=TRUE, model=TRUE), warning=function(w) {{assign('ww', conditionMessage(w), envir=.GlobalEnv); invokeRestart('muffleWarning')}}, error=function(e) e)
    if (inherits(fit, 'error')) {{
      cat(jsonlite::toJSON(list(status='FAILED', error=conditionMessage(fit)), auto_unbox=TRUE), file={json.dumps(str(out_path))})
    }} else {{
      lp <- as.numeric(predict(fit, type='lp'))
      ci <- concordance(Surv(d${spec.time_col}, d${spec.outcome_col}) ~ lp, reverse=TRUE)$concordance
      co <- coef(fit)
      cat(jsonlite::toJSON(list(status='VALID', coef=as.list(co), c_index=ci, iter=fit$iter, warning=if (exists('ww')) ww else ''), auto_unbox=TRUE, digits=16), file={json.dumps(str(out_path))})
    }}
    """
    proc = subprocess.run(["Rscript", "-e", r], capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        return {"status": "FAILED", "error": proc.stderr.strip()}
    return json.loads(out_path.read_text())


def cox_fit_eval_via_r(boot: pd.DataFrame, original: pd.DataFrame, spec: AnalysisSpec, output_dir: Path, label: str) -> Dict[str, Any]:
    tmpdir = output_dir / "sensitive" / "cox_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    boot_path = tmpdir / f"{label}_boot.csv"
    orig_path = tmpdir / f"{label}_orig.csv"
    out_path = tmpdir / f"{label}_out.json"
    cols = [spec.time_col, spec.outcome_col, *spec.predictors]
    boot[cols].to_csv(boot_path, index=False)
    original[cols].to_csv(orig_path, index=False)
    r = f"""
    suppressPackageStartupMessages(library(survival))
    b <- read.csv({json.dumps(str(boot_path))})
    o <- read.csv({json.dumps(str(orig_path))})
    out <- tryCatch({{
      fit <- coxph(Surv({spec.time_col}, {spec.outcome_col}) ~ age_gt40_main + dgf_main, ties='efron', data=b, x=TRUE, model=TRUE)
      co <- coef(fit)
      lp_b <- as.numeric(predict(fit, type='lp'))
      lp_o <- as.numeric(as.matrix(o[, c('age_gt40_main','dgf_main')]) %*% co)
      cb <- concordance(Surv(b${spec.time_col}, b${spec.outcome_col}) ~ lp_b, reverse=TRUE)$concordance
      coo <- concordance(Surv(o${spec.time_col}, o${spec.outcome_col}) ~ lp_o, reverse=TRUE)$concordance
      list(status='VALID', coef=as.list(co), bootstrap_apparent=cb, original_test=coo, iter=fit$iter, warning='')
    }}, warning=function(w) {{
      list(status='VALID_WITH_WARNING', warning=conditionMessage(w))
    }}, error=function(e) {{
      list(status='FAILED', error=conditionMessage(e))
    }})
    cat(jsonlite::toJSON(out, auto_unbox=TRUE, digits=16), file={json.dumps(str(out_path))})
    """
    proc = subprocess.run(["Rscript", "-e", r], capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        return {"status": "FAILED", "error": proc.stderr.strip()}
    return json.loads(out_path.read_text())


def binary_metrics(y: Sequence[int], p: Sequence[float], include_calibration: bool = True) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    out = {
        "auc": float(roc_auc_score(y_arr, p_arr)),
        "average_precision": float(average_precision_score(y_arr, p_arr)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
    }
    if include_calibration:
        p_clip = np.clip(p_arr, CLIP_EPS, 1 - CLIP_EPS)
        lp = np.log(p_clip / (1 - p_clip))
        intercept = sm.GLM(y_arr, np.ones((len(y_arr), 1)), family=sm.families.Binomial(), offset=lp).fit()
        slope = sm.GLM(y_arr, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        out["calibration_intercept"] = float(intercept.params[0])
        out["calibration_slope"] = float(slope.params[1])
    return out


def metric_rows_for_binary(analysis_id: str, attempt_id: int, role: str, y: Sequence[int], p: Sequence[float], include_calibration: bool, invalid_fit: Optional[str] = None) -> List[Dict[str, Any]]:
    names = ["auc", "average_precision", "brier"] + (["calibration_intercept", "calibration_slope"] if include_calibration else [])
    if invalid_fit:
        return [{"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": n, "metric_value": "", "metric_status": "NOT_APPLICABLE", "failure_code": f"INVALID_PRIMARY_FIT:{invalid_fit}"} for n in names]
    rows = []
    base: Dict[str, float] = {}
    try:
        base.update({k: v for k, v in binary_metrics(y, p, include_calibration=False).items()})
    except Exception as exc:
        for n in ["auc", "average_precision", "brier"]:
            rows.append({"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": n, "metric_value": "", "metric_status": "METRIC_EXCEPTION", "failure_code": type(exc).__name__})
    else:
        for n in ["auc", "average_precision", "brier"]:
            rows.append({"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": n, "metric_value": base[n], "metric_status": "VALID", "failure_code": ""})
    if include_calibration:
        p_arr = np.asarray(p, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        p_clip = np.clip(p_arr, CLIP_EPS, 1 - CLIP_EPS)
        lp = np.log(p_clip / (1 - p_clip))
        try:
            intercept = sm.GLM(y_arr, np.ones((len(y_arr), 1)), family=sm.families.Binomial(), offset=lp).fit()
            rows.append({"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": "calibration_intercept", "metric_value": float(intercept.params[0]), "metric_status": "VALID", "failure_code": ""})
        except Exception as exc:
            rows.append({"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": "calibration_intercept", "metric_value": "", "metric_status": "CALIBRATION_EXCEPTION", "failure_code": type(exc).__name__})
        try:
            slope = sm.GLM(y_arr, sm.add_constant(lp), family=sm.families.Binomial()).fit()
            rows.append({"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": "calibration_slope", "metric_value": float(slope.params[1]), "metric_status": "VALID", "failure_code": ""})
        except Exception as exc:
            rows.append({"analysis_id": analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": "calibration_slope", "metric_value": "", "metric_status": "CALIBRATION_EXCEPTION", "failure_code": type(exc).__name__})
    return rows


def reference_equivalence(reference_root: Path, output_dir: Path, pops: Dict[str, pd.DataFrame], pop_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    coefficients_path = _resolve_under(reference_root, "metrics/original_apparent_coefficients.csv")
    coefficients = pd.read_csv(coefficients_path)
    for row in pop_rows:
        rows.append({"gate": "population", "analysis_id": row["analysis_id"], "component": "counts_and_ids", "observed": f"{row['n']}|{row['events']}", "reference": f"{row['expected_n']}|{row['expected_events']}", "abs_diff": "", "status": "PASS" if row["status"] == "VALID" else "FAIL"})
    for spec in ANALYSES:
        pop = pops[spec.analysis_id]
        try:
            if spec.family == "logistic":
                fit = fit_logit(pop, spec)
                tolerance = lambda rv: 1e-6 + 1e-6 * abs(rv)
            elif spec.family == "ridge":
                fit = fit_ridge(pop, spec)
                tolerance = lambda rv: 1e-8 + 1e-6 * abs(rv)
            elif spec.family == "tree":
                fit = fit_tree(pop, spec)
                rows.append({"gate": "reference_model", "analysis_id": spec.analysis_id, "component": "root_variable", "observed": fit["root"], "reference": "race3_clean=Mixed ethnicity/Asian", "abs_diff": "", "status": "PASS" if fit["root"] == "race3_clean=Mixed ethnicity/Asian" else "FAIL"})
                rows.append({"gate": "reference_model", "analysis_id": spec.analysis_id, "component": "depth_leaves", "observed": f"{fit['depth']}|{fit['leaves']}", "reference": "2|4", "abs_diff": "", "status": "PASS" if fit["depth"] == 2 and fit["leaves"] == 4 else "FAIL"})
                continue
            elif spec.family == "cox":
                fit = cox_fit_via_r(pop, spec, output_dir, f"ref_{spec.analysis_id}")
                tolerance = lambda rv: 1e-6 + 1e-6 * abs(rv)
            reference = coefficients[coefficients.analysis_id == spec.analysis_id]
            for term, val in fit.get("params", fit.get("coef", {})).items():
                matched = reference[reference.term == term]
                if len(matched) != 1:
                    raise ValueError(f"Missing or ambiguous canonical coefficient: {spec.analysis_id}/{term}")
                rv = float(matched.original_estimate.iloc[0])
                diff = abs(float(val) - rv)
                rows.append({"gate": "reference_model", "analysis_id": spec.analysis_id, "component": term, "observed": val, "reference": rv, "abs_diff": diff, "status": "PASS" if diff <= tolerance(rv) else "FAIL"})
        except Exception as exc:
            rows.append({"gate": "reference_model", "analysis_id": spec.analysis_id, "component": "exception", "observed": type(exc).__name__, "reference": str(exc), "abs_diff": "", "status": "FAIL"})
    return rows


def resample_hash_equivalence(reference_root: Path, pops: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    reference_path = _resolve_under(reference_root, "diagnostics/bootstrap_resample_hash_reference_1_100.csv")
    _validate_resample_reference(reference_path)
    previous = pd.read_csv(reference_path)
    prev = previous.set_index(["analysis_id", "attempt_id"])
    rows: List[Dict[str, Any]] = []
    for spec in ANALYSES:
        n = len(pops[spec.analysis_id])
        for attempt in range(1, 101):
            idx = rng_for(spec, attempt).integers(0, n, size=n, endpoint=False)
            observed = sha256_bytes(idx.astype(np.int64).tobytes())
            reference = prev.loc[(spec.analysis_id, attempt), "resample_index_sha256"]
            rows.append({"analysis_id": spec.analysis_id, "attempt_id": attempt, "observed_sha256": observed, "reference_sha256": reference, "status": "PASS" if observed == reference else "FAIL"})
    return rows


def original_apparent(data_root: Path, output_dir: Path) -> None:
    pops, _ = population_registries(data_root)
    fit_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    coef_rows: List[Dict[str, Any]] = []
    for spec in ANALYSES:
        pop = pops[spec.analysis_id]
        y = pd.to_numeric(pop[spec.outcome_col]).astype(int).to_numpy()
        try:
            if spec.family == "logistic":
                fit = fit_logit(pop, spec)
                metrics = metric_rows_for_binary(spec.analysis_id, 0, "apparent_original", y, fit["pred"], True, None if fit["status"] == "VALID" else fit["status"])
                params = fit["params"]
            elif spec.family == "ridge":
                fit = fit_ridge(pop, spec)
                metrics = metric_rows_for_binary(spec.analysis_id, 0, "apparent_original", y, fit["pred"], True, None if fit["status"] in {"VALID", "VALID_WITH_WARNING"} else fit["status"])
                params = fit["params"]
            elif spec.family == "tree":
                fit = fit_tree(pop, spec)
                metrics = metric_rows_for_binary(spec.analysis_id, 0, "apparent_original", y, fit["pred"], False, None)
                params = {}
            else:
                fit = cox_fit_via_r(pop, spec, output_dir, f"original_{spec.analysis_id}")
                metrics = [{
                    "analysis_id": spec.analysis_id,
                    "attempt_id": 0,
                    "sample_role": "apparent_original",
                    "metric_name": "c_index",
                    "metric_value": fit.get("c_index", ""),
                    "metric_status": "VALID" if fit.get("status") == "VALID" else "NOT_APPLICABLE",
                    "failure_code": "" if fit.get("status") == "VALID" else fit.get("status", "FAILED"),
                }]
                params = fit.get("coef", {})
            fit_rows.append({"analysis_id": spec.analysis_id, "model_family": spec.family, "fit_status": fit.get("status", "VALID"), "n": len(pop), "events": int(y.sum()), "warning_count": len(fit.get("warnings", [])) if isinstance(fit.get("warnings", []), list) else int(bool(fit.get("warning", "")))})
            for row in metrics:
                metric_rows.append({"analysis_id": row["analysis_id"], "metric_name": row["metric_name"], "apparent_original": row["metric_value"], "validity_status": row["metric_status"], "failure_code": row["failure_code"]})
            for term, val in params.items():
                coef_rows.append({"analysis_id": spec.analysis_id, "term": term, "original_estimate": val, "transformed_estimate": math.exp(float(val)) if spec.family in {"cox", "logistic"} else val, "original_sign": "positive" if float(val) > 0 else ("negative" if float(val) < 0 else "zero"), "coefficient_status": "VALID", "failure_code": ""})
        except Exception as exc:
            fit_rows.append({"analysis_id": spec.analysis_id, "model_family": spec.family, "fit_status": "FAILED", "n": len(pop), "events": int(y.sum()), "warning_count": ""})
            metric_rows.append({"analysis_id": spec.analysis_id, "metric_name": "all", "apparent_original": "", "validity_status": "FAILED", "failure_code": type(exc).__name__})
    write_csv(output_dir / "definitive_results/original_apparent_fits.csv", fit_rows, ["analysis_id", "model_family", "fit_status", "n", "events", "warning_count"])
    write_csv(output_dir / "definitive_results/original_apparent_metrics.csv", metric_rows, ["analysis_id", "metric_name", "apparent_original", "validity_status", "failure_code"])
    write_csv(output_dir / "definitive_results/original_apparent_coefficients.csv", coef_rows, ["analysis_id", "term", "original_estimate", "transformed_estimate", "original_sign", "coefficient_status", "failure_code"])


def run_gates(data_root: Path, reference_root: Path, output_dir: Path) -> int:
    pops, pop_rows = population_registries(data_root)
    write_csv(output_dir / "pilot_results/bootstrap_population_registry.csv", pop_rows, ["analysis_id", "source_file", "n", "events", "expected_n", "expected_events", "id_set_sha256", "previous_id_set_sha256", "status"])
    ref_rows = reference_equivalence(reference_root, output_dir, pops, pop_rows)
    write_csv(output_dir / "reference_model_equivalence.csv", ref_rows, ["gate", "analysis_id", "component", "observed", "reference", "abs_diff", "status"])
    hash_rows = resample_hash_equivalence(reference_root, pops)
    write_csv(output_dir / "resample_hash_equivalence.csv", hash_rows, ["analysis_id", "attempt_id", "observed_sha256", "reference_sha256", "status"])
    ok = all(r["status"] == "PASS" for r in ref_rows) and all(r["status"] == "PASS" for r in hash_rows)
    write_csv(output_dir / "qualification_checks.csv", [
        {"check": "reference_model_equivalence", "status": "PASS" if all(r["status"] == "PASS" for r in ref_rows) else "FAIL", "evidence": "reference_model_equivalence.csv"},
        {"check": "resample_hash_equivalence_900", "status": "PASS" if all(r["status"] == "PASS" for r in hash_rows) else "FAIL", "evidence": "resample_hash_equivalence.csv"},
    ], ["check", "status", "evidence"])
    return 0 if ok else 2


def fit_attempt(spec: AnalysisSpec, boot: pd.DataFrame, original: pd.DataFrame, output_dir: Path, attempt_id: int) -> Dict[str, Any]:
    """Compara o desempenho na reamostra com o da população analítica original."""
    start = time.time()
    y_boot = pd.to_numeric(boot[spec.outcome_col]).astype(int).to_numpy()
    y_orig = pd.to_numeric(original[spec.outcome_col]).astype(int).to_numpy()
    result: Dict[str, Any] = {"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "metrics": [], "coefficients": [], "tree": {}, "fit_status": "VALID", "warnings": []}
    if len(np.unique(y_boot)) < 2:
        result["fit_status"] = "SINGLE_CLASS" if spec.family != "cox" else "NO_EVENT"
    try:
        if result["fit_status"] == "VALID" and spec.family == "logistic":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x_boot = sm.add_constant(boot[list(spec.predictors)], has_constant="add")
                model = sm.Logit(y_boot, x_boot).fit(disp=False, maxiter=5000)
            result["warnings"] = [str(w.message) for w in caught]
            if not model.mle_retvals.get("converged"):
                result["fit_status"] = "NONCONVERGENCE"
            params = {"Intercept" if k == "const" else k: float(v) for k, v in model.params.to_dict().items()}
            finite = all(np.isfinite(list(params.values())))
            if not finite:
                result["fit_status"] = "NONFINITE_COEFFICIENT"
            p_boot = np.asarray(model.predict(x_boot))
            x_orig = sm.add_constant(original[list(spec.predictors)], has_constant="add")
            p_orig = np.asarray(model.predict(x_orig))
            invalid = None if result["fit_status"] == "VALID" else result["fit_status"]
            result["metrics"] += metric_rows_for_binary(spec.analysis_id, attempt_id, "bootstrap_apparent", y_boot, p_boot, True, invalid)
            result["metrics"] += metric_rows_for_binary(spec.analysis_id, attempt_id, "original_test", y_orig, p_orig, True, invalid)
            if invalid is None:
                for term, val in params.items():
                    result["coefficients"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "term": term, "estimate": val, "coefficient_status": "VALID"})
        elif result["fit_status"] == "VALID" and spec.family == "ridge":
            levels = race_levels(original)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipe = Pipeline([("preprocess", preprocessor(spec, levels, tree=False)), ("model", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000))])
                pipe.fit(boot[list(spec.predictors)], y_boot)
            result["warnings"] = [str(w.message) for w in caught]
            model = pipe.named_steps["model"]
            if int(model.n_iter_[0]) >= 5000:
                result["fit_status"] = "NONCONVERGENCE"
            names = feature_names(pipe)
            params = {"Intercept": float(model.intercept_[0])}
            params.update({n: float(v) for n, v in zip(names, model.coef_[0])})
            if not all(np.isfinite(list(params.values()))):
                result["fit_status"] = "NONFINITE_COEFFICIENT"
            p_boot = pipe.predict_proba(boot[list(spec.predictors)])[:, 1]
            p_orig = pipe.predict_proba(original[list(spec.predictors)])[:, 1]
            invalid = None if result["fit_status"] in {"VALID", "VALID_WITH_WARNING"} else result["fit_status"]
            result["metrics"] += metric_rows_for_binary(spec.analysis_id, attempt_id, "bootstrap_apparent", y_boot, p_boot, True, invalid)
            result["metrics"] += metric_rows_for_binary(spec.analysis_id, attempt_id, "original_test", y_orig, p_orig, True, invalid)
            if invalid is None:
                for term, val in params.items():
                    result["coefficients"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "term": term, "estimate": val, "coefficient_status": "VALID"})
        elif result["fit_status"] == "VALID" and spec.family == "tree":
            levels = race_levels(original)
            pipe = Pipeline([("preprocess", preprocessor(spec, levels, tree=True)), ("model", DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=20, class_weight="balanced", random_state=42))])
            pipe.fit(boot[list(spec.predictors)], y_boot)
            model = pipe.named_steps["model"]
            names = feature_names(pipe)
            p_boot = pipe.predict_proba(boot[list(spec.predictors)])[:, 1]
            p_orig = pipe.predict_proba(original[list(spec.predictors)])[:, 1]
            result["metrics"] += metric_rows_for_binary(spec.analysis_id, attempt_id, "bootstrap_apparent", y_boot, p_boot, False, None)
            result["metrics"] += metric_rows_for_binary(spec.analysis_id, attempt_id, "original_test", y_orig, p_orig, False, None)
            used = [names[i] for i in model.tree_.feature if i >= 0]
            result["tree"] = {"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "root_variable": used[0] if used else "", "variables_any_node": "|".join(used), "node_count": int(model.tree_.node_count), "leaf_count": int(model.get_n_leaves()), "depth": int(model.get_depth()), "rules": export_text(model, feature_names=names)}
        elif result["fit_status"] == "VALID" and spec.family == "cox":
            cox = cox_fit_eval_via_r(boot, original, spec, output_dir, f"{spec.analysis_id}_{attempt_id:03d}")
            result["fit_status"] = cox.get("status", "FAILED")
            result["warnings"] = [cox.get("warning", "")] if cox.get("warning") else []
            if result["fit_status"] == "VALID":
                result["metrics"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "sample_role": "bootstrap_apparent", "metric_name": "c_index", "metric_value": cox["bootstrap_apparent"], "metric_status": "VALID", "failure_code": ""})
                result["metrics"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "sample_role": "original_test", "metric_name": "c_index", "metric_value": cox["original_test"], "metric_status": "VALID", "failure_code": ""})
                for term, val in cox.get("coef", {}).items():
                    result["coefficients"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "term": term, "estimate": val, "coefficient_status": "VALID"})
            else:
                for role in ["bootstrap_apparent", "original_test"]:
                    result["metrics"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": "c_index", "metric_value": "", "metric_status": "NOT_APPLICABLE", "failure_code": f"INVALID_PRIMARY_FIT:{result['fit_status']}"})
        else:
            include_cal = spec.family in {"logistic", "ridge"}
            names = ["c_index"] if spec.family == "cox" else ["auc", "average_precision", "brier"] + (["calibration_intercept", "calibration_slope"] if include_cal else [])
            for role in ["bootstrap_apparent", "original_test"]:
                for name in names:
                    result["metrics"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": name, "metric_value": "", "metric_status": "NOT_APPLICABLE", "failure_code": f"INVALID_PRIMARY_FIT:{result['fit_status']}"})
    except Exception as exc:
        result["fit_status"] = "RUNTIME_EXCEPTION"
        result["exception_type"] = type(exc).__name__
        result["exception_text"] = str(exc)
        if spec.family == "cox":
            names = ["c_index"]
        else:
            names = ["auc", "average_precision", "brier"]
            if spec.family in {"logistic", "ridge"}:
                names += ["calibration_intercept", "calibration_slope"]
        for role in ["bootstrap_apparent", "original_test"]:
            for name in names:
                result["metrics"].append({"analysis_id": spec.analysis_id, "attempt_id": attempt_id, "sample_role": role, "metric_name": name, "metric_value": "", "metric_status": "NOT_APPLICABLE", "failure_code": "INVALID_PRIMARY_FIT:RUNTIME_EXCEPTION"})
    result["elapsed_seconds"] = time.time() - start
    return result


def run_attempt_range(data_root: Path, output_dir: Path, start_id: int, end_id: int, resume: bool = False) -> None:
    pops, _ = population_registries(data_root)
    for spec in ANALYSES:
        original = pops[spec.analysis_id]
        n = len(original)
        for attempt in range(start_id, end_id + 1):
            out_path = output_dir / "attempt_results" / spec.analysis_id / f"attempt_{attempt:03d}.json"
            if resume and out_path.exists():
                continue
            idx = rng_for(spec, attempt).integers(0, n, size=n, endpoint=False)
            boot = original.iloc[idx].reset_index(drop=True)
            result = fit_attempt(spec, boot, original, output_dir, attempt)
            result["n_original"] = n
            result["n_bootstrap"] = len(boot)
            result["events_bootstrap"] = int(pd.to_numeric(boot[spec.outcome_col]).sum())
            result["n_unique_original_ids"] = int(len(set(idx.tolist())))
            result["resample_index_sha256"] = sha256_bytes(idx.astype(np.int64).tobytes())
            write_text(out_path, json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")


def aggregate_results(output_dir: Path) -> None:
    """Agrega estimativas de otimismo e estabilidade entre tentativas válidas."""
    attempts: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []
    coefs: List[Dict[str, Any]] = []
    trees: List[Dict[str, Any]] = []
    for p in sorted((output_dir / "attempt_results").glob("*/*.json")):
        r = json.loads(p.read_text())
        attempts.append({"analysis_id": r["analysis_id"], "attempt_id": r["attempt_id"], "fit_status": r["fit_status"], "warning_count": len(r.get("warnings", [])), "elapsed_seconds": r.get("elapsed_seconds", ""), "n_original": r.get("n_original", ""), "n_bootstrap": r.get("n_bootstrap", ""), "events_bootstrap": r.get("events_bootstrap", ""), "n_unique_original_ids": r.get("n_unique_original_ids", ""), "resample_index_sha256": r.get("resample_index_sha256", "")})
        metrics.extend(r.get("metrics", []))
        coefs.extend(r.get("coefficients", []))
        if r.get("tree"):
            trees.append(r["tree"])
    results_dir = output_dir / ("definitive_results" if max([a["attempt_id"] for a in attempts], default=0) > 100 else "qualification_replay")
    write_csv(results_dir / "bootstrap_attempt_registry.csv", attempts, ["analysis_id", "attempt_id", "fit_status", "warning_count", "elapsed_seconds", "n_original", "n_bootstrap", "events_bootstrap", "n_unique_original_ids", "resample_index_sha256"])
    write_csv(results_dir / "bootstrap_metrics_long.csv", metrics, ["analysis_id", "attempt_id", "sample_role", "metric_name", "metric_value", "metric_status", "failure_code"])
    write_csv(results_dir / "bootstrap_coefficients_long.csv", coefs, ["analysis_id", "attempt_id", "term", "estimate", "coefficient_status"])
    write_csv(results_dir / "bootstrap_tree_attempts.csv", trees, ["analysis_id", "attempt_id", "root_variable", "variables_any_node", "node_count", "leaf_count", "depth", "rules"])
    # optimism and summaries
    metric_df = pd.DataFrame(metrics)
    opt_rows: List[Dict[str, Any]] = []
    if not metric_df.empty:
        for (aid, att, metric), g in metric_df.groupby(["analysis_id", "attempt_id", "metric_name"], sort=True):
            b = g[(g.sample_role == "bootstrap_apparent") & (g.metric_status == "VALID")]
            o = g[(g.sample_role == "original_test") & (g.metric_status == "VALID")]
            if len(b) == 1 and len(o) == 1:
                raw = float(b.metric_value.iloc[0]) - float(o.metric_value.iloc[0])
                opt_rows.append({"analysis_id": aid, "attempt_id": att, "metric_name": metric, "bootstrap_apparent": b.metric_value.iloc[0], "original_test": o.metric_value.iloc[0], "raw_optimism": raw, "pair_status": "VALID", "failure_code": ""})
            else:
                opt_rows.append({"analysis_id": aid, "attempt_id": att, "metric_name": metric, "bootstrap_apparent": "", "original_test": "", "raw_optimism": "", "pair_status": "NOT_APPLICABLE", "failure_code": "INVALID_OR_MISSING_PAIR"})
    write_csv(results_dir / "bootstrap_optimism_attempt_long.csv", opt_rows, ["analysis_id", "attempt_id", "metric_name", "bootstrap_apparent", "original_test", "raw_optimism", "pair_status", "failure_code"])
    opt_df = pd.DataFrame(opt_rows)
    summary: List[Dict[str, Any]] = []
    apparent_path = output_dir / "definitive_results/original_apparent_metrics.csv"
    apparent_df = pd.read_csv(apparent_path) if apparent_path.exists() else pd.DataFrame()
    if not opt_df.empty:
        for (aid, metric), g in opt_df.groupby(["analysis_id", "metric_name"], sort=True):
            vals = pd.to_numeric(g[g.pair_status == "VALID"].raw_optimism, errors="coerce").dropna().to_numpy()
            ap = ""
            if not apparent_df.empty:
                m = apparent_df[(apparent_df.analysis_id == aid) & (apparent_df.metric_name == metric) & (apparent_df.validity_status == "VALID")]
                ap = float(m.apparent_original.iloc[0]) if len(m) else ""
            mean_opt = float(np.mean(vals)) if len(vals) else ""
            summary.append({"analysis_id": aid, "metric_name": metric, "apparent_original": ap, "valid_pair_count": len(vals), "invalid_pair_count": int(len(g) - len(vals)), "mean_optimism": mean_opt, "median_optimism": float(np.median(vals)) if len(vals) else "", "sd_optimism": float(np.std(vals, ddof=1)) if len(vals) > 1 else "", "p2_5_optimism": float(np.percentile(vals, 2.5)) if len(vals) else "", "p97_5_optimism": float(np.percentile(vals, 97.5)) if len(vals) else "", "optimism_corrected": float(ap - mean_opt) if ap != "" and mean_opt != "" else "", "interpretation_status": "EXPLORATORIA_CALIBRACAO" if metric in {"calibration_intercept", "calibration_slope"} else "CORRECAO_PRIMARIA_DE_OTIMISMO"})
    write_csv(results_dir / "bootstrap_optimism_summary.csv", summary, ["analysis_id", "metric_name", "apparent_original", "valid_pair_count", "invalid_pair_count", "mean_optimism", "median_optimism", "sd_optimism", "p2_5_optimism", "p97_5_optimism", "optimism_corrected", "interpretation_status"])
    coef_df = pd.DataFrame(coefs)
    csum: List[Dict[str, Any]] = []
    if not coef_df.empty:
        for (aid, term), g in coef_df.groupby(["analysis_id", "term"], sort=True):
            vals = pd.to_numeric(g.estimate, errors="coerce").dropna().to_numpy()
            csum.append({"analysis_id": aid, "term": term, "valid_count": len(vals), "positive_sign_frequency": float(np.mean(vals > 0)) if len(vals) else "", "p2_5": float(np.percentile(vals, 2.5)) if len(vals) else "", "p50": float(np.percentile(vals, 50)) if len(vals) else "", "p97_5": float(np.percentile(vals, 97.5)) if len(vals) else ""})
    write_csv(results_dir / "bootstrap_coefficient_stability_summary.csv", csum, ["analysis_id", "term", "valid_count", "positive_sign_frequency", "p2_5", "p50", "p97_5"])
    # validity and decisions
    attempts_df = pd.DataFrame(attempts)
    validity: List[Dict[str, Any]] = []
    if not metric_df.empty:
        for (aid, metric), g in metric_df[metric_df.sample_role == "original_test"].groupby(["analysis_id", "metric_name"], sort=True):
            validity.append({"analysis_id": aid, "metric_name": metric, "valid_count": int((g.metric_status == "VALID").sum()), "invalid_count": int((g.metric_status != "VALID").sum())})
    write_csv(results_dir / "bootstrap_metric_validity_counts.csv", validity, ["analysis_id", "metric_name", "valid_count", "invalid_count"])
    decisions: List[Dict[str, Any]] = []
    for spec in ANALYSES:
        a = attempts_df[attempts_df.analysis_id == spec.analysis_id]
        valid_fits = int(a.fit_status.isin(["VALID", "VALID_WITH_WARNING"]).sum()) if not a.empty else 0
        required_attempts = 1000 if results_dir.name == "definitive_results" else 100
        go_valid = 950 if required_attempts == 1000 else 95
        cond_valid = 900 if required_attempts == 1000 else 90
        decision = "GO" if len(a) == required_attempts and valid_fits >= go_valid else ("GO_WITH_CONDITIONS" if valid_fits >= cond_valid else "NO_GO")
        decisions.append({"analysis_id": spec.analysis_id, "attempts": len(a), "valid_fits": valid_fits, "decision": decision})
    write_csv(results_dir / "definitive_bootstrap_go_no_go.csv", decisions, ["analysis_id", "attempts", "valid_fits", "decision"])


def run_reproducibility(data_root: Path, output_dir: Path) -> None:
    repro_dir = output_dir / "sensitive" / "repro_1_5"
    # Recompute to isolated directory, then compare JSON hashes after removing elapsed seconds.
    temp_out = output_dir / "sensitive" / "repro_tmp"
    if temp_out.exists():
        import shutil
        shutil.rmtree(temp_out)
    temp_out.mkdir(parents=True)
    run_attempt_range(data_root, temp_out, 1, 5, resume=False)
    rows = []
    for spec in ANALYSES:
        for attempt in range(1, 6):
            orig = json.loads((output_dir / "attempt_results" / spec.analysis_id / f"attempt_{attempt:03d}.json").read_text())
            rep = json.loads((temp_out / "attempt_results" / spec.analysis_id / f"attempt_{attempt:03d}.json").read_text())
            for obj in (orig, rep):
                obj.pop("elapsed_seconds", None)
            rows.append({"analysis_id": spec.analysis_id, "attempt_id": attempt, "status_match": orig.get("fit_status") == rep.get("fit_status"), "resample_hash_match": orig.get("resample_index_sha256") == rep.get("resample_index_sha256"), "content_match_excluding_runtime": orig == rep, "reproducibility_status": "PASS" if orig == rep and orig.get("resample_index_sha256") == rep.get("resample_index_sha256") else "FAIL"})
    write_csv(output_dir / "pilot_results/bootstrap_reproducibility_check.csv", rows, ["analysis_id", "attempt_id", "status_match", "resample_hash_match", "content_match_excluding_runtime", "reproducibility_status"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--attempt-start", type=int)
    parser.add_argument("--attempt-end", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--original-apparent", action="store_true")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    reference_root = Path(args.reference_root)
    output_dir = Path(args.output_dir)
    if args.preflight_only:
        print(json.dumps(run_preflight(data_root, reference_root, output_dir), sort_keys=True))
        return 0

    data_root, reference_root, output_dir = _validated_roots(data_root, reference_root, output_dir)
    _load_analytical_dependencies()
    if args.gate_only:
        return run_gates(data_root, reference_root, output_dir)
    if args.original_apparent:
        original_apparent(data_root, output_dir)
        return 0
    if args.attempt_start is not None or args.attempt_end is not None:
        if args.attempt_start is None or args.attempt_end is None or args.attempt_start < 1 or args.attempt_end < args.attempt_start:
            raise SystemExit("--attempt-start and --attempt-end must define a valid positive range")
        run_attempt_range(data_root, output_dir, args.attempt_start, args.attempt_end, args.resume)
        return 0
    if args.reproduce:
        run_reproducibility(data_root, output_dir)
        return 0
    if args.aggregate:
        aggregate_results(Path(args.output_dir).resolve())
        return 0
    raise SystemExit("pilot execution not reached before gate validation")


if __name__ == "__main__":
    raise SystemExit(main())
