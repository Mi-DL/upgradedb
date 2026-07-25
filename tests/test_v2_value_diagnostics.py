from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v2_value_diagnostics as value  # noqa: E402


class ValuePointTest(unittest.TestCase):
    def test_global_oracle_gap_is_same_budget_and_same_cohort(self) -> None:
        frame = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B"],
                "stage": ["s", "s", "s", "s"],
                "j_iso": ["W", "X", "Y", "Z"],
                "y": [1, 1, 0, 0],
                "lateval": [10.0, 30.0, 0.0, 0.0],
            }
        )
        point = value._global_point(
            frame,
            np.asarray([0.9, 0.1, 0.8, 0.7]),
            label="y",
            value="lateval",
            budget=1,
            keys=("i_iso", "stage", "j_iso"),
        )
        self.assertEqual(point["model_selected_observed_late_value_kusd"], 10.0)
        self.assertEqual(point["oracle_selected_observed_late_value_kusd"], 30.0)
        self.assertEqual(point["model_value_capture"], 0.25)
        self.assertEqual(point["oracle_value_capture"], 0.75)
        self.assertEqual(point["oracle_gap_value_capture"], 0.5)
        self.assertEqual(point["headroom_kusd"], 20.0)

    def test_paired_global_bootstrap_is_clustered_and_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B"],
                "stage": ["s"] * 4,
                "j_iso": ["W", "X", "Y", "Z"],
                "lateval": [10.0, 0.0, 20.0, 0.0],
            }
        )
        kwargs = dict(
            cluster="i_iso",
            value="lateval",
            budget=1,
            keys=("i_iso", "stage", "j_iso"),
            draws=50,
            seed=17,
        )
        first = value._bootstrap_global(frame, np.asarray([0.9, 0.8, 0.1, 0.7]), **kwargs)
        second = value._bootstrap_global(frame, np.asarray([0.9, 0.8, 0.1, 0.7]), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_unit"], "i_iso")
        self.assertEqual(first["n_clusters"], 2)
        self.assertEqual(first["effective_draws"], 50)
        self.assertIsNotNone(first["oracle_gap_value_capture_ci95"])


class GroupDiagnosticTest(unittest.TestCase):
    def test_b2_oracle_and_model_are_macroed_over_complete_entries(self) -> None:
        frame = pd.DataFrame(
            {
                "entry_id": ["A|s"] * 4 + ["B|s"] * 4,
                "j_iso": ["W", "X", "Y", "Z"] * 2,
                "y": [1, 0, 0, 0, 1, 0, 0, 0],
                "lateval": [10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            }
        )
        score = np.asarray([0.1, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.6])
        stats = value._group_statistics(
            frame,
            score,
            group="entry_id",
            label="y",
            value="lateval",
            budget=3,
            keys=("j_iso",),
        )
        point = value._group_point(stats, budget=3, unit="exporter_stage_entry")
        self.assertEqual(point["groups"], 2)
        self.assertEqual(point["model_macro_value_capture"], 0.5)
        self.assertEqual(point["oracle_macro_value_capture"], 1.0)
        self.assertAlmostEqual(point["model_pooled_value_capture"], 2 / 3)
        self.assertEqual(point["headroom_kusd"], 10.0)

    def test_score_alignment_fails_closed_on_missing_key(self) -> None:
        source = pd.DataFrame(
            {"i_iso": ["A"], "j_iso": ["X"], "stage": ["s"]}
        )
        target = pd.DataFrame(
            {
                "i_iso": ["A", "B"],
                "j_iso": ["X", "Y"],
                "stage": ["s", "s"],
            }
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            value._align_values(
                source,
                target,
                np.asarray([0.1]),
                keys=("i_iso", "j_iso", "stage"),
                role="synthetic",
            )


class DollarAccountingTest(unittest.TestCase):
    def test_b2_is_a_nested_view_and_is_not_added_to_b1(self) -> None:
        track_a = pd.DataFrame({"lateval": [100.0]})
        track_b1 = pd.DataFrame({"entry_lateval": [30.0]})
        track_b2 = pd.DataFrame({"lateval": [10.0, 20.0]})
        formal = value.FormalTables(
            track_a=track_a,
            track_b_lanes=pd.DataFrame(),
            track_b1=track_b1,
            track_b2=track_b2,
            audits={},
        )
        accounting = value._accounting({chain: formal for chain in value.CHAINS})
        n = len(value.CHAINS)
        self.assertEqual(
            accounting["totals"]["unique_project_observed_late_value_kusd"],
            n * 130.0,
        )
        self.assertEqual(
            accounting["totals"]["forbidden_naive_a_plus_b1_plus_b2_kusd"],
            n * 160.0,
        )
        self.assertTrue(accounting["totals"]["b2_excluded_from_unique_sum"])


class VerificationTest(unittest.TestCase):
    def test_generator_hash_must_match_current_script(self) -> None:
        current = value._sha256(Path(value.__file__).resolve())
        value._verify_generator_hash(
            {"runtime": {"script_sha256": current}}, Path("matching.json")
        )
        with self.assertRaisesRegex(ValueError, "generator hash is stale"):
            value._verify_generator_hash(
                {"runtime": {"script_sha256": "0" * 64}}, Path("stale.json")
            )


class CandidateBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = value.FormalTables(
            track_a=pd.DataFrame(),
            track_b_lanes=pd.DataFrame(),
            track_b1=pd.DataFrame(),
            track_b2=pd.DataFrame(),
            audits={
                "track_a": {
                    "path": "data/processed_v2/candidates_sheep.csv",
                    "sha256": "a" * 64,
                    "rows": 10,
                    "positive_lanes": 2,
                },
                "track_b_lane_pool": {
                    "path": "data/processed_v2/candidates_firsttime_sheep.csv",
                    "sha256": "b" * 64,
                    "rows": 20,
                    "positive_lanes": 3,
                },
            },
        )
        self.cpu_chain = {
            "protocol_audit": {
                "history_track_a": {
                    "path": "data/processed_v2/candidates_sheep_fold2.csv",
                    "sha256": "c" * 64,
                },
                "history_track_b": {
                    "path": "data/processed_v2/candidates_firsttime_sheep_fold2.csv",
                    "sha256": "d" * 64,
                },
            }
        }

    def _record(self, track: str) -> dict:
        is_a = track == "a"
        return {
            "target_candidate_role": (
                "data/processed_v2/candidates_sheep.csv"
                if is_a
                else "data/processed_v2/candidates_firsttime_sheep.csv"
            ),
            "target_candidate_sha256": ("a" if is_a else "b") * 64,
            "target_rows": 10 if is_a else 20,
            "target_positive_lanes": 2 if is_a else 3,
            "history_candidate_role": (
                "data/processed_v2/candidates_sheep_fold2.csv"
                if is_a
                else "data/processed_v2/candidates_firsttime_sheep_fold2.csv"
            ),
            "history_candidate_sha256": ("c" if is_a else "d") * 64,
        }

    def test_a_and_shared_b_lane_receipts_are_cross_bound(self) -> None:
        for track in ("a", "b1", "b2"):
            value._validate_gpu_candidate_binding(
                self._record(track),
                self.tables,
                self.cpu_chain,
                track,
                role=f"synthetic/{track}",
            )

    def test_stale_target_or_history_receipt_is_rejected(self) -> None:
        for field in ("target_candidate_sha256", "target_positive_lanes", "history_candidate_sha256"):
            with self.subTest(field=field):
                record = self._record("b1")
                record[field] = 0 if field == "target_positive_lanes" else "0" * 64
                with self.assertRaisesRegex(ValueError, field):
                    value._validate_gpu_candidate_binding(
                        record,
                        self.tables,
                        self.cpu_chain,
                        "b1",
                        role="synthetic/b1",
                    )


if __name__ == "__main__":
    unittest.main()
