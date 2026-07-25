from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v2_gbdt_baselines as gbdt  # noqa: E402


class _FeaturePipeline:
    def __init__(self, column: str):
        self.column = column
        self.fit_rows: list[int] = []

    def fit(self, features: pd.DataFrame, y: np.ndarray) -> "_FeaturePipeline":
        self.fit_rows.append(len(features))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = features[self.column].to_numpy(float)
        return np.column_stack([1.0 - score, score])


class _RecordingPipeline:
    def __init__(self) -> None:
        self.prediction_rows: list[int] = []

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.prediction_rows.append(len(features))
        score = np.linspace(0.2, 0.8, len(features), dtype=float)
        return np.column_stack([1.0 - score, score])


class V2GBDTConfigAndSelectionTest(unittest.TestCase):
    def test_frozen_config_drives_the_hist_gradient_boosting_pipeline(self) -> None:
        config = gbdt.load_frozen_config()
        pipeline = gbdt._pipeline(config, "a", config["grid"][0])
        classifier = pipeline.named_steps["classifier"]

        self.assertEqual(config["features"]["a"], list(gbdt.FEATURES["a"]))
        self.assertTrue(pipeline.named_steps["imputer"].add_indicator)
        self.assertEqual(classifier.loss, "log_loss")
        self.assertEqual(classifier.learning_rate, 0.05)
        self.assertEqual(classifier.max_leaf_nodes, 3)
        self.assertEqual(classifier.max_iter, 100)
        self.assertEqual(classifier.min_samples_leaf, 20)
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertFalse(classifier.early_stopping)

    def test_frozen_config_tamper_is_rejected(self) -> None:
        config = copy.deepcopy(gbdt.load_frozen_config())
        config["main_evaluation"]["a_value_budget"] = 501
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(gbdt.GBDTProtocolError, "main-evaluation"):
                gbdt.load_frozen_config(path)

    def test_exact_grid_tie_selects_first_declared_configuration(self) -> None:
        config = gbdt.load_frozen_config()
        features = pd.DataFrame(
            {
                "size": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
                "log_gravity": [0.5] * 8,
            }
        )
        y = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        groups = np.repeat(np.asarray(["g0", "g1", "g2", "g3"]), 2)
        left = np.arange(0, 4)
        right = np.arange(4, 8)
        splits = [(left, right), (right, left)]
        created: list[_FeaturePipeline] = []

        def factory(*_args: object, **_kwargs: object) -> _FeaturePipeline:
            model = _FeaturePipeline("size")
            created.append(model)
            return model

        with (
            patch.object(gbdt.cpu, "_valid_group_splits", return_value=(splits, 2)),
            patch.object(gbdt, "_pipeline", side_effect=factory),
        ):
            frozen = gbdt._select_and_fit(config, "a", features, y, groups)

        self.assertEqual(frozen.selection["selected_grid_index"], 0)
        self.assertEqual(frozen.selection["selected_config_id"], "leaf3_iter100")
        self.assertTrue(
            frozen.selection["train_validation_group_overlap_checked"]
        )
        self.assertEqual(len(created), len(config["grid"]) * 2 + 1)
        gbdt._validate_selection(
            frozen.selection,
            config,
            "a",
            history_rows=len(y),
            history_positives=int(y.sum()),
            where="synthetic.selection",
        )

        tampered = copy.deepcopy(frozen.selection)
        tampered["candidates"][0]["mean_objective"] += 0.01
        with self.assertRaisesRegex(gbdt.GBDTProtocolError, "candidate mean"):
            gbdt._validate_selection(
                tampered,
                config,
                "a",
                history_rows=len(y),
                history_positives=int(y.sum()),
                where="synthetic.selection",
            )

    def test_b2_selection_uses_entry_macro_recall_and_entry_weighting(self) -> None:
        config = gbdt.load_frozen_config()
        groups = np.repeat(np.asarray(["g0", "g1", "g2", "g3", "g4", "g5"]), 4)
        y = np.tile(np.asarray([1, 0, 0, 0], dtype=np.int8), 6)
        hit = np.asarray([0.9, 0.3, 0.2, 0.1])
        miss = np.asarray([0.0, 0.3, 0.2, 0.1])
        candidate_a = np.concatenate([hit, hit, miss, miss, miss, miss])
        candidate_b = np.concatenate([miss, hit, hit, hit, miss, miss])
        features = pd.DataFrame(
            {
                "log_importer_demand": candidate_a,
                "log_gravity": candidate_b,
            }
        )
        keys = np.tile(np.asarray(["a", "b", "c", "d"]), 6)
        fold0 = np.arange(0, 4)
        fold1 = np.arange(4, 24)
        splits = [(fold1, fold0), (fold0, fold1)]

        def factory(
            _config: object, _track: object, grid_row: dict[str, object]
        ) -> _FeaturePipeline:
            column = (
                "log_importer_demand"
                if grid_row["config_id"] == "leaf3_iter100"
                else "log_gravity"
            )
            return _FeaturePipeline(column)

        with (
            patch.object(gbdt.cpu, "_valid_group_splits", return_value=(splits, 2)),
            patch.object(gbdt, "_pipeline", side_effect=factory),
        ):
            frozen = gbdt._select_and_fit(
                config,
                "b2",
                features,
                y,
                groups,
                objective_groups=groups,
                tie_break_keys=keys,
            )

        self.assertEqual(frozen.selection["selected_grid_index"], 1)
        self.assertEqual(frozen.selection["selected_config_id"], "leaf7_iter100")
        self.assertEqual(
            frozen.selection["candidates"][0]["fold_objective_units"], [1, 5]
        )
        self.assertEqual(
            frozen.selection["objective_aggregation"],
            "validation-entry-count-weighted-fold-mean",
        )

    def test_fit_chain_uses_exporters_for_a_b1_and_entries_for_b2(self) -> None:
        config = gbdt.load_frozen_config()
        history_a = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B"],
                "y": [1, 0, 1, 0],
                "size": [0.9, 0.1, 0.8, 0.2],
                "grav": [1.0, 2.0, 3.0, 4.0],
            }
        )
        history_b = pd.DataFrame(
            {
                "entry_id": ["A|s", "A|s", "B|s", "B|s"],
                "j_iso": ["X", "Y", "X", "Y"],
                "y": [1, 0, 0, 0],
                "log_importer_demand": [1.0, 2.0, 1.5, 2.5],
                "grav": [1.0, 2.0, 3.0, 4.0],
            }
        )
        entries = pd.DataFrame(
            {
                "i_iso": ["A", "B"],
                "entry_id": ["A|s", "B|s"],
                "z": [1, 0],
            }
        )
        audit_a = {"rows": 4, "positive_lanes": 2}
        audit_b = {"rows": 4, "positive_lanes": 1}
        dummy = gbdt.FrozenGBDT(pipeline=None, selection={})  # type: ignore[arg-type]

        with (
            patch.object(
                gbdt.cpu,
                "_read_candidate",
                side_effect=[(history_a, audit_a), (history_b, audit_b)],
            ),
            patch.object(gbdt.cpu, "_derive_entry_table", return_value=entries),
            patch.object(gbdt, "_select_and_fit", return_value=dummy) as select,
        ):
            gbdt._fit_chain(Path("synthetic"), "sheep", config)

        a_call, b1_call, b2_call = select.call_args_list
        np.testing.assert_array_equal(
            a_call.args[4], np.asarray(["A", "A", "B", "B"])
        )
        np.testing.assert_array_equal(b1_call.args[4], np.asarray(["A", "B"]))
        np.testing.assert_array_equal(
            b2_call.args[4], np.asarray(["A|s", "A|s"])
        )
        np.testing.assert_array_equal(
            b2_call.kwargs["objective_groups"], np.asarray(["A|s", "A|s"])
        )


class V2GBDTReadGateAndEvaluationTest(unittest.TestCase):
    def test_all_six_chains_fit_before_the_first_target_evaluation(self) -> None:
        events: list[tuple[str, str]] = []

        def fit(
            _data: Path, chain: str, _config: object
        ) -> object:
            events.append(("fit", chain))
            return object()

        def evaluate(
            _data: Path, chain: str, _model: object, _config: object
        ) -> dict[str, object]:
            events.append(("evaluate", chain))
            return {}

        with (
            patch.object(gbdt, "_candidate_inventory", return_value=[]),
            patch.object(gbdt, "_macro_summary", return_value={}),
            patch.object(gbdt.platform, "processor", return_value="synthetic-cpu"),
            patch.object(gbdt.os, "cpu_count", return_value=8),
            patch.object(gbdt.time, "perf_counter", side_effect=[10.0, 11.25]),
        ):
            payload = gbdt.run(
                Path("synthetic"),
                chains=gbdt.cpu.CHAINS,
                fit_chain=fit,  # type: ignore[arg-type]
                evaluate_chain=evaluate,  # type: ignore[arg-type]
            )

        self.assertEqual(
            events[: len(gbdt.cpu.CHAINS)],
            [("fit", chain) for chain in gbdt.cpu.CHAINS],
        )
        self.assertEqual(
            events[len(gbdt.cpu.CHAINS) :],
            [("evaluate", chain) for chain in gbdt.cpu.CHAINS],
        )
        self.assertEqual(payload["runtime"]["wall_elapsed_seconds"], 1.25)
        self.assertEqual(payload["runtime"]["cpu_model"], "synthetic-cpu")
        self.assertEqual(payload["runtime"]["logical_cpu_cores"], 8)

    def test_evaluation_uses_registered_bootstrap_units_and_scores_all_b2_lanes(self) -> None:
        config = gbdt.load_frozen_config()
        target_a = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B"],
                "stage": ["s", "s", "s", "s"],
                "j_iso": ["X", "Y", "X", "Y"],
                "entry_id": ["A|s", "A|s", "B|s", "B|s"],
                "y": [1, 0, 1, 0],
                "size": [1.0, 2.0, 3.0, 4.0],
                "grav": [1.0, 2.0, 3.0, 4.0],
                "lateval": [10.0, 0.0, 20.0, 0.0],
            }
        )
        target_b = pd.DataFrame(
            {
                "i_iso": ["A", "A", "A", "B", "B", "B"],
                "stage": ["s"] * 6,
                "j_iso": ["X", "Y", "Z", "X", "Y", "Z"],
                "entry_id": ["A|s"] * 3 + ["B|s"] * 3,
                "y": [1, 0, 0, 0, 0, 0],
                "log_importer_demand": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                "log_exporter_capacity": [2.0] * 3 + [3.0] * 3,
                "grav": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                "lateval": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        audit_a = {"rows": 4, "positive_lanes": 2}
        audit_b = {"rows": 6, "positive_lanes": 1}
        a_pipeline = _RecordingPipeline()
        b1_pipeline = _RecordingPipeline()
        b2_pipeline = _RecordingPipeline()
        models = gbdt.ChainModels(
            track_a=gbdt.FrozenGBDT(a_pipeline, {}),  # type: ignore[arg-type]
            track_b1=gbdt.FrozenGBDT(b1_pipeline, {}),  # type: ignore[arg-type]
            track_b2=gbdt.FrozenGBDT(b2_pipeline, {}),  # type: ignore[arg-type]
            history_a_audit=audit_a,
            history_b_audit=audit_b,
            history_b1_entries=2,
            history_b1_positives=1,
            history_b2_lanes=3,
            history_b2_positive_lanes=1,
            history_b2_entry_groups=1,
        )

        with (
            patch.object(
                gbdt.cpu,
                "_read_candidate",
                side_effect=[(target_a, audit_a), (target_b, audit_b)],
            ),
            patch.object(
                gbdt.cpu, "_classification_metrics", return_value={}
            ) as classify,
            patch.object(gbdt.cpu, "_conditional_metrics", return_value={}) as conditional,
        ):
            gbdt._evaluate_chain(Path("synthetic"), "sheep", models, config)

        self.assertEqual(a_pipeline.prediction_rows, [4])
        self.assertEqual(b1_pipeline.prediction_rows, [2])
        self.assertEqual(b2_pipeline.prediction_rows, [6])
        self.assertEqual(len(classify.call_args_list), 2)
        a_call, b1_call = classify.call_args_list
        self.assertEqual(a_call.kwargs["bootstrap_draws"], 200)
        self.assertEqual(a_call.kwargs["cluster_unit"], "exporter")
        np.testing.assert_array_equal(
            a_call.kwargs["cluster"], np.asarray(["A", "A", "B", "B"])
        )
        self.assertEqual(b1_call.kwargs["bootstrap_draws"], 200)
        self.assertEqual(b1_call.kwargs["cluster_unit"], "exporter")
        np.testing.assert_array_equal(
            b1_call.kwargs["cluster"], np.asarray(["A", "B"])
        )
        self.assertEqual(conditional.call_args.args[0]["entry_id"].nunique(), 1)
        self.assertEqual(len(conditional.call_args.args[0]), 3)
        self.assertEqual(len(conditional.call_args.args[1]), 3)
        self.assertEqual(conditional.call_args.kwargs["bootstrap_draws"], 200)
        self.assertEqual(conditional.call_args.kwargs["ks"], (1, 3, 5))


class V2GBDTArtifactGuardTest(unittest.TestCase):
    def test_macro_summary_canonicalizes_chain_order(self) -> None:
        config = gbdt.load_frozen_config()
        chain_payloads: dict[str, object] = {}
        for index, chain in enumerate(reversed(gbdt.cpu.CHAINS)):
            headline = 0.1 + index * 0.01
            chain_payloads[chain] = {
                gbdt.TRACK_ORDER[0]: {
                    "models": {
                        gbdt.MODEL_KEY: {
                            "metrics": {
                                "average_precision": headline,
                                "budgets": {"k_500": {"value_capture": 0.2}},
                            }
                        }
                    }
                },
                gbdt.TRACK_ORDER[1]: {
                    "models": {
                        gbdt.MODEL_KEY: {
                            "metrics": {
                                "average_precision": headline,
                                "budgets": {"k_50": {"value_capture": 0.2}},
                            }
                        }
                    }
                },
                gbdt.TRACK_ORDER[2]: {
                    "models": {
                        gbdt.MODEL_KEY: {
                            "metrics": {
                                "at_k": {
                                    "k_3": {
                                        "macro_recall": headline,
                                        "macro_value_capture": 0.2,
                                    }
                                }
                            }
                        }
                    }
                },
            }

        summary = gbdt._macro_summary(chain_payloads, config)
        for track in gbdt.TRACK_ORDER:
            self.assertEqual(
                summary[track]["chain_registry"], list(gbdt.cpu.CHAINS)
            )
            self.assertEqual(
                list(summary[track]["headline"]["per_chain"]),
                list(gbdt.cpu.CHAINS),
            )

    def test_source_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.txt"
            path.write_text("audited", encoding="utf-8")
            digest = gbdt._sha256(path)
            gbdt._verify_file_hash(path, digest, "synthetic source")
            with self.assertRaisesRegex(gbdt.GBDTProtocolError, "hash mismatch"):
                gbdt._verify_file_hash(path, "0" * 64, "synthetic source")

    def test_csv_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            json_path = Path(temporary) / "result.json"
            csv_path = Path(temporary) / "result.csv"
            payload = {"synthetic": True}
            json_path.write_bytes(gbdt._strict_json_bytes(payload))
            csv_path.write_bytes(b"tampered\n")
            with (
                patch.object(gbdt, "validate_payload"),
                patch.object(gbdt, "_csv_bytes", return_value=b"canonical\n"),
            ):
                with self.assertRaisesRegex(gbdt.GBDTProtocolError, "stale or noncanonical"):
                    gbdt.verify_existing_output(json_path, csv_path)

    def test_private_paths_and_cluster_names_are_rejected(self) -> None:
        separator = "\\"
        for value in (
            "C:" + separator + "Users" + separator + "person" + separator + "result.json",
            "/" + "home" + "/person/result.json",
            "compute-node" + ":" + "/scratch/person/result.json",
            separator + separator + "file-server" + separator + "share" + separator + "result.json",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(gbdt.GBDTProtocolError, "private/absolute"):
                    gbdt._assert_privacy({"value": value})


if __name__ == "__main__":
    unittest.main()
