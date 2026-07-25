from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v2_rolling_cpu_baselines as cpu  # noqa: E402


class _ScoreColumnPipeline:
    def __init__(self, c_value: float):
        self.c_value = c_value

    def fit(self, features: pd.DataFrame, y: np.ndarray) -> "_ScoreColumnPipeline":
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        column = "candidate_a" if self.c_value == 0.01 else "candidate_b"
        score = features[column].to_numpy(float)
        return np.column_stack([1.0 - score, score])


class V2RollingCPUSelectionTest(unittest.TestCase):
    def test_b2_macro_recall_objective_is_not_lane_average_precision(self) -> None:
        # Candidate A wins lane AP by ranking the ten positives in the large
        # entry first. Candidate B wins the required per-entry macro recall@3
        # by retrieving the positive in the small entry.
        y = np.asarray([1] * 10 + [0] * 10 + [1] + [0] * 9, dtype=np.int8)
        groups = np.asarray(["large"] * 20 + ["small"] * 10)
        keys = np.asarray([f"j{index:02d}" for index in range(len(y))])
        score_a = np.asarray(
            list(np.linspace(1.0, 0.8, 10))
            + list(np.linspace(0.6, 0.4, 10))
            + [0.0]
            + list(np.linspace(0.7, 0.61, 9))
        )
        score_b = np.asarray(
            list(np.linspace(0.3, 0.21, 10))
            + list(np.linspace(0.9, 0.71, 10))
            + [1.0]
            + list(np.linspace(0.7, 0.61, 9))
        )

        self.assertGreater(
            average_precision_score(y, score_a),
            average_precision_score(y, score_b),
        )
        self.assertLess(
            cpu._positive_entry_macro_recall_at_k(y, score_a, groups, keys, k=3),
            cpu._positive_entry_macro_recall_at_k(y, score_b, groups, keys, k=3),
        )

    def test_b2_select_and_fit_records_aligned_group_safe_objective(self) -> None:
        groups = np.repeat(np.asarray(["g0", "g1", "g2", "g3", "g4", "g5"]), 4)
        y = np.tile(np.asarray([1, 0, 0, 0], dtype=np.int8), 6)
        hit = np.asarray([0.9, 0.3, 0.2, 0.1])
        miss = np.asarray([0.0, 0.3, 0.2, 0.1])
        # The folds deliberately have one versus five entries. Candidate A has
        # fold means (1.0, 0.2), while B has (0.0, 0.6). An incorrect unweighted
        # mean over folds selects A; entry-weighted OOF macro recall selects B.
        candidate_a = np.concatenate([hit, hit, miss, miss, miss, miss])
        candidate_b = np.concatenate([miss, hit, hit, hit, miss, miss])
        features = pd.DataFrame(
            {"candidate_a": candidate_a, "candidate_b": candidate_b}
        )
        tie_keys = np.tile(np.asarray(["a", "b", "c", "d"]), 6)
        fold0 = np.arange(0, 4)
        fold1 = np.arange(4, 24)
        splits = [(fold1, fold0), (fold0, fold1)]

        with (
            patch.object(cpu, "C_GRID", (0.01, 1.0)),
            patch.object(cpu, "_valid_group_splits", return_value=(splits, 2)),
            patch.object(
                cpu,
                "_pipeline",
                side_effect=lambda c_value, seed: _ScoreColumnPipeline(c_value),
            ),
        ):
            frozen = cpu._select_and_fit(
                features,
                y,
                groups,
                feature_names=("candidate_a", "candidate_b"),
                group_unit="exporter_stage_entry",
                seed=7,
                objective="positive_entry_macro_recall_at_3",
                objective_groups=groups,
                objective_tie_break_keys=tie_keys,
            )

        self.assertEqual(frozen.selection["selected_C"], 1.0)
        self.assertEqual(
            frozen.selection["objective"],
            "historical_group_cv_per_positive_entry_macro_recall_at_3",
        )
        self.assertTrue(frozen.selection["train_validation_group_overlap_checked"])
        self.assertEqual(frozen.selection["candidates"][0]["fold_objective_units"], [1, 5])
        self.assertAlmostEqual(frozen.selection["selected_mean_objective"], 0.5)
        self.assertIn("validation-entry count", frozen.selection["objective_aggregation"])
        self.assertEqual(
            frozen.selection["hyperparameter_tie_break"],
            "maximize_mean_objective_then_smaller_C",
        )

    def test_b2_exact_score_ties_use_destination_key(self) -> None:
        y = np.asarray([1, 0, 0, 1], dtype=np.int8)
        groups = np.asarray(["entry", "entry", "entry", "entry"])
        score = np.ones(4)
        keys = np.asarray(["D", "A", "B", "C"])
        observed = cpu._positive_entry_macro_recall_at_k(
            y, score, groups, keys, k=3
        )
        shuffled = np.asarray([2, 0, 3, 1])
        repeated = cpu._positive_entry_macro_recall_at_k(
            y[shuffled], score[shuffled], groups[shuffled], keys[shuffled], k=3
        )
        self.assertEqual(observed, 0.5)
        self.assertEqual(repeated, observed)

    def test_fit_chain_uses_exporters_for_a_and_entries_for_b2(self) -> None:
        history_a = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B"],
                "entry_id": ["A|s1", "A|s2", "B|s1", "B|s2"],
                "y": [1, 0, 1, 0],
                "size": [1.0, 2.0, 3.0, 4.0],
                "grav": [0.1, 0.2, 0.3, 0.4],
            }
        )
        history_b = pd.DataFrame(
            {
                "entry_id": ["A|s", "A|s", "B|s", "B|s"],
                "j_iso": ["X", "Y", "X", "Y"],
                "y": [1, 0, 0, 0],
                "log_importer_demand": [1.0, 2.0, 1.5, 2.5],
                "grav": [0.1, 0.2, 0.3, 0.4],
            }
        )
        entries = pd.DataFrame(
            {
                "i_iso": ["A", "B"],
                "entry_id": ["A|s", "B|s"],
                "z": [1, 0],
                "log_upstream_capacity": [2.0, 3.0],
            }
        )
        audit_a = {"rows": 4, "positive_lanes": 2}
        audit_b = {"rows": 4, "positive_lanes": 1}
        dummy = cpu.FrozenClassifier(pipeline=None, selection={})  # type: ignore[arg-type]

        with (
            patch.object(
                cpu,
                "_read_candidate",
                side_effect=[(history_a, audit_a), (history_b, audit_b)],
            ),
            patch.object(cpu, "_derive_entry_table", return_value=entries),
            patch.object(cpu, "_select_and_fit", return_value=dummy) as select,
        ):
            cpu._fit_chain(Path("synthetic"), "sheep", seed=11)

        a_call, b1_call, b2_call = select.call_args_list
        np.testing.assert_array_equal(
            a_call.args[2], np.asarray(["A", "A", "B", "B"])
        )
        self.assertEqual(a_call.kwargs["group_unit"], "exporter")
        self.assertEqual(b1_call.kwargs["group_unit"], "exporter")
        np.testing.assert_array_equal(
            b2_call.args[2], np.asarray(["A|s", "A|s"])
        )
        self.assertEqual(b2_call.kwargs["group_unit"], "exporter_stage_entry")
        self.assertEqual(
            b2_call.kwargs["objective"], "positive_entry_macro_recall_at_3"
        )


def _track_models(
    names: tuple[str, ...],
    *,
    headline_base: float,
    value_base: float,
    track: str,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for index, name in enumerate(names):
        headline = headline_base + index * 0.1
        realized = value_base + index * 0.05
        if track == "a":
            metrics = {
                "average_precision": headline,
                "budgets": {"k_500": {"value_capture": realized}},
            }
        elif track == "b1":
            metrics = {
                "average_precision": headline,
                "budgets": {"k_50": {"value_capture": realized}},
            }
        else:
            metrics = {
                "at_k": {
                    "k_3": {
                        "macro_recall": headline,
                        "macro_value_capture": realized,
                    }
                }
            }
        models[name] = {"metrics": metrics}
    return models


def _chain(headline_base: float, value_base: float) -> dict[str, object]:
    return {
        "track_a_destination_extension": {
            "models": _track_models(
                cpu.TRACK_MODEL_ORDER["track_a_destination_extension"],
                headline_base=headline_base,
                value_base=value_base,
                track="a",
            )
        },
        "track_b1_processed_export_stage_entry": {
            "models": _track_models(
                cpu.TRACK_MODEL_ORDER["track_b1_processed_export_stage_entry"],
                headline_base=headline_base + 0.01,
                value_base=value_base + 0.01,
                track="b1",
            )
        },
        "track_b2_conditional_destination_ranking": {
            "models": _track_models(
                cpu.TRACK_MODEL_ORDER["track_b2_conditional_destination_ranking"],
                headline_base=headline_base + 0.02,
                value_base=value_base + 0.02,
                track="b2",
            )
        },
    }


class V2RollingCPUSummaryTest(unittest.TestCase):
    def test_summary_adds_frozen_realized_value_and_descriptive_deltas(self) -> None:
        summary = cpu._macro_summary(
            {"chain_one": _chain(0.1, 0.2), "chain_two": _chain(0.3, 0.4)}
        )
        track_a = summary["track_a_destination_extension"]
        self.assertEqual(track_a["budget_definition"]["requested_k"], 500)
        size = track_a["models"]["size"]
        self.assertEqual(
            size["realized_value"]["per_chain"],
            {"chain_one": 0.2, "chain_two": 0.4},
        )
        self.assertAlmostEqual(size["realized_value"]["macro_mean"], 0.3)

        b2 = summary["track_b2_conditional_destination_ranking"]
        self.assertEqual(b2["budget_definition"]["requested_k_per_entry"], 3)
        self.assertEqual(
            b2["realized_value_metric"],
            "per_positive_entry_macro_value_capture_at_3",
        )

        pairwise = track_a["pairwise_deltas"]
        self.assertTrue(pairwise["frozen_before_main"])
        self.assertFalse(pairwise["post_hoc_champion_selection"])
        self.assertEqual(len(pairwise["comparisons"]), 3)
        first = pairwise["comparisons"][0]
        self.assertIsNone(first["headline"]["chain_level_ci95"])
        self.assertEqual(
            first["headline"]["per_chain"],
            {"chain_one": -0.1, "chain_two": -0.10000000000000003},
        )

    def test_classification_uncertainty_records_explicit_cluster_unit(self) -> None:
        frame = pd.DataFrame(
            {"y": [1, 0, 1, 0], "lateval": [10.0, 0.0, 20.0, 0.0]}
        )
        with patch.object(cpu, "_cluster_ap_ci", return_value=[0.1, 0.9]) as bootstrap:
            metrics = cpu._classification_metrics(
                frame,
                label="y",
                score=np.asarray([0.9, 0.2, 0.8, 0.1]),
                cluster=np.asarray(["A", "A", "B", "B"]),
                cluster_unit="exporter",
                budgets=(1,),
                bootstrap_draws=10,
                seed=5,
            )
        self.assertEqual(metrics["uncertainty"]["cluster_unit"], "exporter")
        np.testing.assert_array_equal(
            bootstrap.call_args.args[2], np.asarray(["A", "A", "B", "B"])
        )

    def test_v1_saved_result_is_rejected_before_any_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.json"
            path.write_text(
                json.dumps(
                    {"schema_version": "upgrade-bench-v2-rolling-cpu-baselines-1"}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pre-audit result is invalidated"):
                cpu.verify_existing_output(path)


if __name__ == "__main__":
    unittest.main()
