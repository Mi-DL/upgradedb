import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"
sys.path.insert(0, str(ROOT / "src"))

from v2_gpu_protocol import ProtocolError  # noqa: E402
from v2_gpu_rolling import (  # noqa: E402
    DEFAULT_CHAINS,
    _assert_complete_global_freeze,
    _assert_frozen_seeds,
    _assert_exporter_stage_partition,
    _entry_arrays,
    _deterministic_score_order,
    _ranking_metrics,
    _selection_metric,
    _selection_metric_name,
    _cluster_bootstrap,
    _claim_main_start,
    _cached_score,
)


class V2GpuMetricTest(unittest.TestCase):
    def setUp(self):
        self.identities = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B", "C", "C"],
                "j_iso": ["X", "Y", "X", "Y", "X", "Y"],
                "stage": ["s", "s", "s", "s", "s", "s"],
            }
        )
        self.y = np.array([1, 0, 0, 0, 1, 1])
        self.score = np.array([0.9, 0.1, 0.4, 0.3, 0.8, 0.7])

    def test_track_b_uses_one_max_aggregated_score_per_entry(self):
        entry = _entry_arrays(self.identities, self.y, self.score)
        self.assertEqual(len(entry), 3)
        self.assertEqual(entry["y"].tolist(), [1, 0, 1])
        self.assertEqual(entry["score"].tolist(), [0.9, 0.4, 0.8])

    def test_finite_budget_ties_use_explicit_identity_order(self):
        identities = pd.DataFrame(
            {
                "i_iso": ["B", "A", "A", "A"],
                "stage": ["s", "t", "s", "s"],
                "j_iso": ["A", "A", "C", "B"],
            }
        )
        order = _deterministic_score_order(
            identities,
            np.ones(len(identities)),
            ("i_iso", "stage", "j_iso"),
        )
        self.assertEqual(order.tolist(), [3, 2, 1, 0])

        labels = pd.DataFrame(
            {
                "y": [0] * 49 + [1] + [0] * 9 + [1],
                "size": np.arange(60.0),
                "lateval": [0.0] * 49 + [2.0] + [0.0] * 9 + [3.0],
            }
        )
        lanes = pd.DataFrame(
            {
                "i_iso": ["A"] * 60,
                "stage": ["s"] * 60,
                "j_iso": [f"J{index:02d}" for index in range(60)],
            }
        )
        scores = np.ones(60)
        first = _ranking_metrics("a", lanes, labels, scores)
        permutation = np.arange(59, -1, -1)
        second = _ranking_metrics(
            "a",
            lanes.iloc[permutation].reset_index(drop=True),
            labels.iloc[permutation].reset_index(drop=True),
            scores[permutation],
        )
        for key in ("precision_at_50", "recall_at_50", "value_capture_at_50"):
            self.assertEqual(first[key], second[key])

    def test_track_b_selection_metric_is_entry_average_precision(self):
        metric = _selection_metric(
            "b1", self.identities, self.y, self.score, np.ones(len(self.y), dtype=bool)
        )
        self.assertEqual(metric, 1.0)
        self.assertEqual(
            _selection_metric_name("b1"),
            "track_b1_entry_average_precision_max_lane_score",
        )

    def test_track_b2_has_independent_historical_recall_selection(self):
        metric = _selection_metric(
            "b2", self.identities, self.y, self.score, np.ones(len(self.y), dtype=bool)
        )
        self.assertEqual(metric, 1.0)
        self.assertEqual(_selection_metric_name("b2"), "track_b2_positive_entry_macro_recall_at_3")

    def test_history_partition_rejects_exporter_stage_crossing(self):
        with self.assertRaisesRegex(ProtocolError, "leaks an exporter-stage"):
            _assert_exporter_stage_partition(
                self.identities,
                np.array([False, True, False, False, True, True]),
            )

    def test_default_chain_ids_match_registry_and_v2_artifact_names(self):
        registry = {path.stem for path in (ROOT / "chains").glob("*.json")}
        self.assertEqual(set(DEFAULT_CHAINS), registry)
        expected = {f"data/processed_v2/candidates_{chain}.csv" for chain in DEFAULT_CHAINS}
        self.assertEqual(
            expected,
            {f"data/processed_v2/candidates_{chain}.csv" for chain in registry},
        )

    @unittest.skipUnless(
        FULL_PAYLOAD_TESTS_ENABLED,
        "external processed tables are checked only in the explicit full-profile stage",
    )
    def test_full_profile_has_all_default_chain_candidate_tables(self):
        for chain in DEFAULT_CHAINS:
            self.assertTrue((ROOT / "data" / "processed_v2" / f"candidates_{chain}.csv").is_file())

    def test_main_gate_rejects_partial_freeze(self):
        with self.assertRaisesRegex(ProtocolError, "complete 6-chain"):
            _assert_complete_global_freeze({"sheep|a|kge": object()})

    def test_task_specific_metrics_and_cluster_units(self):
        labels = pd.DataFrame({"y": self.y, "size": np.arange(6.0), "lateval": [2, 0, 0, 0, 3, 4]})
        a = _ranking_metrics("a", self.identities, labels, self.score)
        self.assertIn("exporter_macro_recall_at_5", a)
        b1 = _ranking_metrics("b1", self.identities, labels, self.score)
        self.assertIn("entry_precision_at_25", b1)
        self.assertNotIn("conditional_recall_at_3", b1)
        b2 = _ranking_metrics("b2", self.identities, labels, self.score)
        self.assertIn("conditional_recall_at_3", b2)
        self.assertNotIn("entry_precision_at_25", b2)
        self.assertEqual(
            _cluster_bootstrap("a", self.identities, labels, self.score, iters=5, seed=1)["cluster_unit"],
            "exporter",
        )
        self.assertEqual(
            _cluster_bootstrap("b1", self.identities, labels, self.score, iters=5, seed=1)["cluster_unit"],
            "exporter",
        )
        self.assertEqual(
            _cluster_bootstrap("b2", self.identities, labels, self.score, iters=5, seed=1)["cluster_unit"],
            "exporter_stage",
        )

    def test_evaluation_seeds_are_locked(self):
        selection = {"selection_design": {"evaluation_seeds": [0, 1, 2]}}
        self.assertEqual(_assert_frozen_seeds(selection, [0, 1, 2]), [0, 1, 2])
        with self.assertRaisesRegex(ProtocolError, "do not equal frozen"):
            _assert_frozen_seeds(selection, [0, 1])

    def test_main_start_marker_is_immutable_per_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "MAIN_EVALUATION_STARTED.json"
            first = _claim_main_start(marker, run_id="run", manifest_sha256="a" * 64)
            second = _claim_main_start(marker, run_id="run", manifest_sha256="a" * 64)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ProtocolError, "different run/manifest"):
                _claim_main_start(marker, run_id="run", manifest_sha256="b" * 64)

    def test_score_cache_trains_once_per_hash_locked_config_seed(self):
        calls = []

        def scorer(model, hp, seed):
            calls.append((model, hp["x"], seed))
            return np.array([0.1, 0.2, 0.3]) + seed

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kwargs = {
                "scorer": scorer,
                "model": "M",
                "hyperparameters": {"x": 1},
                "cache_root": root,
                "cache_context": {"graph_sha256": "a" * 64, "input_sha256": "b" * 64},
                "expected_rows": 3,
            }
            first, first_meta = _cached_score(seed=0, **kwargs)
            second, second_meta = _cached_score(seed=0, **kwargs)
            self.assertEqual(calls, [("M", 1, 0)])
            np.testing.assert_array_equal(first, second)
            self.assertFalse(first_meta["hit"])
            self.assertTrue(second_meta["hit"])
            _cached_score(seed=1, **kwargs)
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
