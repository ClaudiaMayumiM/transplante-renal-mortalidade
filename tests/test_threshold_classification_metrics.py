from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/04_internal_validation/threshold_classification_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("threshold_classification_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class ThresholdClassificationTests(unittest.TestCase):
    def test_confusion_matrix_known_counts(self) -> None:
        result = MODULE.confusion_metrics(
            np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
        )
        self.assertEqual(
            {key: result[key] for key in (
                "true_positives", "false_positives", "true_negatives", "false_negatives"
            )},
            {"true_positives": 1, "false_positives": 1, "true_negatives": 1, "false_negatives": 1},
        )

    def test_fixed_threshold_is_inclusive_at_point_five(self) -> None:
        data = frame([
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 1, "id_original": 1, "outcome_true": 1, "predicted_probability": 0.5, "train_event_rate": 0.8},
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 1, "id_original": 2, "outcome_true": 0, "predicted_probability": 0.499, "train_event_rate": 0.8},
        ])
        metrics, _ = MODULE.classification(data)
        row = metrics[metrics.threshold_name.eq("FIXED_0_5")].iloc[0]
        self.assertEqual((row.true_positives, row.true_negatives), (1, 1))

    def test_training_prevalence_is_specific_to_originating_fold(self) -> None:
        data = frame([
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 1, "id_original": 1, "outcome_true": 1, "predicted_probability": 0.3, "train_event_rate": 0.2},
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 2, "id_original": 2, "outcome_true": 0, "predicted_probability": 0.3, "train_event_rate": 0.4},
        ])
        metrics, _ = MODULE.classification(data)
        row = metrics[metrics.threshold_name.eq("TRAIN_EVENT_RATE")].iloc[0]
        self.assertEqual((row.true_positives, row.true_negatives), (1, 1))

    def test_zero_denominator_preserves_nan_and_status(self) -> None:
        result = MODULE.confusion_metrics(np.array([1, 1]), np.array([0, 0]))
        self.assertTrue(np.isnan(result["specificity"]))
        self.assertTrue(np.isnan(result["balanced_accuracy"]))
        data = frame([
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 1, "id_original": 1, "outcome_true": 1, "predicted_probability": 0.1, "train_event_rate": 0.2},
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 2, "id_original": 2, "outcome_true": 1, "predicted_probability": 0.1, "train_event_rate": 0.2},
        ])
        metrics, _ = MODULE.classification(data)
        fixed = metrics[metrics.threshold_name.eq("FIXED_0_5")].iloc[0]
        self.assertEqual(fixed.metric_status, "METRIC_UNDEFINED")
        self.assertIn("specificity", fixed.undefined_metrics)

    def test_balanced_accuracy_formula(self) -> None:
        result = MODULE.confusion_metrics(
            np.array([1, 1, 0, 0, 0, 0]), np.array([1, 0, 1, 0, 0, 0])
        )
        self.assertAlmostEqual(result["balanced_accuracy"], (0.5 + 0.75) / 2)

    def test_f1_formula(self) -> None:
        result = MODULE.confusion_metrics(
            np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
        )
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_folds_are_combined_before_confusion_matrix(self) -> None:
        data = frame([
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 1, "id_original": 1, "outcome_true": 1, "predicted_probability": 0.9, "train_event_rate": 0.2},
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 2, "id_original": 2, "outcome_true": 0, "predicted_probability": 0.1, "train_event_rate": 0.3},
            {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": 1, "fold_index": 3, "id_original": 3, "outcome_true": 1, "predicted_probability": 0.8, "train_event_rate": 0.4},
        ])
        _, counts = MODULE.classification(data)
        fixed = counts[counts.threshold_name.eq("FIXED_0_5")]
        self.assertEqual(len(fixed), 1)
        self.assertEqual(int(fixed.iloc[0].true_positives), 2)

    def test_summary_uses_median_and_empirical_percentiles(self) -> None:
        rows = []
        for repetition, value in enumerate([0.0, 0.25, 0.75, 1.0], start=1):
            row = {"analysis_key": "2y", "model_name": "logistic_parsimonious", "repeat_index": repetition, "threshold_name": "FIXED_0_5"}
            row.update({metric: value for metric in MODULE.METRICS})
            rows.append(row)
        summary = MODULE.summarize(pd.DataFrame(rows))
        accuracy = summary[summary.metric_name.eq("accuracy")].iloc[0]
        self.assertEqual(accuracy.repetitions_valid, 4)
        self.assertAlmostEqual(accuracy["median"], 0.5)
        self.assertAlmostEqual(accuracy.p2_5, np.percentile([0.0, 0.25, 0.75, 1.0], 2.5))
        self.assertAlmostEqual(accuracy.p97_5, np.percentile([0.0, 0.25, 0.75, 1.0], 97.5))


if __name__ == "__main__":
    unittest.main()
