from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "benchmark" / "upgrade-bench-v2"
sys.path.insert(0, str(PACKAGE))

import loader as v2_loader  # noqa: E402
import eval as v2_eval  # noqa: E402


def _metadata(chain: str, snapshot: str, i_iso: str, stage: str) -> dict[str, object]:
    window = v2_loader.SNAPSHOT_METADATA[snapshot]
    return {
        "benchmark_version": v2_loader.VERSION,
        "aggregation": "calendar_mean",
        "early_window": window["early_window"],
        "late_window": window["late_window"],
        "transductive_split_unit": "exporter_stage",
        "transductive_split": v2_loader._official_split(chain, i_iso, stage),
        "temporal_role": window["temporal_role"],
    }


def _track_a(chain: str = "sheep", snapshot: str = "main") -> pd.DataFrame:
    rows = []
    for j_iso, y, lateval in (("BBB", 1, 150.0), ("CCC", 0, 0.0)):
        row = {
            "i_iso": "AAA", "j_iso": j_iso, "stage": "processed", "y": y,
            "size": 5.0, "log_exporter_capacity": 2.0, "log_importer_demand": 3.0,
            "size_basis": "processed_exporter_plus_processed_importer", "grav": 0.5,
            "gnn": -0.25, "lateval": lateval, "group_id": "AAA|processed",
            "task": "destination_extension", "task_unit": "exporter_stage_destination",
            **_metadata(chain, snapshot, "AAA", "processed"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=list(v2_loader.LANE_COLUMNS))


def _track_b1(chain: str = "sheep", snapshot: str = "fold2") -> pd.DataFrame:
    rows = []
    for i_iso, z, value, materialized in (("AAA", 1, 170.0, 1), ("DDD", 0, 0.0, 0)):
        stage = "processed"
        rows.append(
            {
                "i_iso": i_iso, "stage": stage, "z": z, "size": 4.0,
                "log_upstream_capacity": 4.0, "entry_lateval": value,
                "n_candidate_destinations": 3, "n_materialized_destinations": materialized,
                "entry_id": f"{i_iso}|{stage}", "task": "processed_export_stage_entry",
                "task_unit": "exporter_stage", **_metadata(chain, snapshot, i_iso, stage),
            }
        )
    return pd.DataFrame(rows, columns=list(v2_loader.ENTRY_COLUMNS))


def _track_b2(chain: str = "sheep", snapshot: str = "main") -> pd.DataFrame:
    rows = []
    for j_iso, y, lateval in (("BBB", 1, 150.0), ("CCC", 0, 0.0)):
        row = {
            "i_iso": "AAA", "j_iso": j_iso, "stage": "processed", "y": y,
            "size": 5.0, "log_exporter_capacity": 2.0, "log_importer_demand": 3.0,
            "size_basis": "registered_upstream_exporter_plus_processed_importer", "grav": np.nan,
            "gnn": 0.0, "lateval": lateval, "group_id": "AAA|processed",
            "entry_id": "AAA|processed", "entry_y": 1,
            "task": "conditional_destination_given_entry", "task_unit": "exporter_stage_destination",
            **_metadata(chain, snapshot, "AAA", "processed"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=list(v2_loader.TRACK_B_LANE_COLUMNS))


class LoaderTests(unittest.TestCase):
    def test_load_all_tracks_and_snapshots_from_environment_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data" / "processed_v2"
            root.mkdir(parents=True)
            fixtures = (("A", "main", _track_a()), ("B1", "fold2", _track_b1()), ("B2", "main", _track_b2()))
            for track, snapshot, frame in fixtures:
                frame.to_csv(root / v2_loader.filename_for(track, "sheep", snapshot), index=False)
            with mock.patch.dict(os.environ, {"UPGRADE_BENCH_V2_DATA": temp}):
                for track, snapshot, frame in fixtures:
                    loaded = v2_loader.load(track, "sheep", snapshot)
                    self.assertEqual(len(loaded), len(frame))
                    self.assertEqual(loaded.attrs["benchmark_track"], track)

    def test_duplicate_key_and_official_split_are_strict(self) -> None:
        duplicate = pd.concat([_track_a(), _track_a().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(v2_loader.BenchmarkDataError, "duplicate"):
            v2_loader.validate_table(duplicate, "A", "sheep")
        bad_split = _track_b1()
        bad_split.loc[0, "transductive_split"] = "test" if bad_split.loc[0, "transductive_split"] == "train" else "train"
        with self.assertRaisesRegex(v2_loader.BenchmarkDataError, "SHA-256"):
            v2_loader.validate_table(bad_split, "B1", "sheep", "fold2")

    def test_exact_schema_rejects_unexpected_outcome_like_fields(self) -> None:
        bad = _track_a().assign(active_years=5)
        with self.assertRaisesRegex(v2_loader.BenchmarkDataError, "schema mismatch"):
            v2_loader.validate_table(bad, "A", "sheep")


class ScorerTests(unittest.TestCase):
    def test_average_precision_is_tie_block_invariant(self) -> None:
        self.assertEqual(v2_eval.average_precision([1, 0], [0.0, 0.0]), 0.5)
        self.assertIsNone(v2_eval.average_precision([0, 0], [0.2, 0.1]))

    def test_budget_ties_use_canonical_keys_not_row_order(self) -> None:
        frame = _track_a()
        # BBB is the positive and sorts before CCC, so an all-tied top-1 is deterministic.
        first = v2_eval.evaluate_track_a(frame, [0.0, 0.0], budgets=(1,))
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        second = v2_eval.evaluate_track_a(shuffled, [0.0, 0.0], budgets=(1,))
        self.assertEqual(first["budgets"], second["budgets"])
        self.assertEqual(first["budgets"]["k_1"]["precision"], 1.0)

    def test_track_a_per_exporter_shortlists_match_cpu_macro_definitions(self) -> None:
        frame = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B", "C"],
                "stage": ["s", "s", "s", "s", "s"],
                "j_iso": ["a", "b", "a", "b", "a"],
                "y": [1, 0, 0, 1, 0],
                "lateval": [10.0, 0.0, 0.0, 20.0, 0.0],
            }
        )
        # All scores tie. Lexical j_iso makes A's top-1 a hit and B's a miss.
        first = v2_eval._exporter_shortlists(frame, np.zeros(len(frame)), ks=(1,))
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        second = v2_eval._exporter_shortlists(shuffled, np.zeros(len(frame)), ks=(1,))
        self.assertEqual(first, second)
        metric = first["k_1_per_exporter"]
        self.assertEqual(metric["exporters"], 3)
        self.assertEqual(metric["exporters_with_positive"], 2)
        self.assertAlmostEqual(metric["macro_precision"], 1 / 3)
        self.assertAlmostEqual(metric["macro_recall_positive_exporters"], 0.5)
        self.assertAlmostEqual(metric["micro_recall"], 0.5)
        self.assertAlmostEqual(metric["macro_value_capture_positive_exporters"], 0.5)
        self.assertAlmostEqual(metric["micro_value_capture"], 1 / 3)

        no_positive = frame.assign(y=0, lateval=0.0)
        empty_denominator = v2_eval._exporter_shortlists(
            no_positive, np.zeros(len(no_positive)), ks=(1,)
        )["k_1_per_exporter"]
        self.assertEqual(empty_denominator["macro_precision"], 0.0)
        self.assertIsNone(empty_denominator["macro_recall_positive_exporters"])
        self.assertIsNone(empty_denominator["micro_recall"])
        self.assertIsNone(empty_denominator["macro_value_capture_positive_exporters"])
        self.assertIsNone(empty_denominator["micro_value_capture"])

    def test_track_a_official_output_contains_exporter_cutoffs_5_and_10(self) -> None:
        metrics = v2_eval.evaluate_track_a(_track_a(), [1.0, 0.0], budgets=(1,))
        self.assertEqual(
            set(metrics["per_exporter_shortlists"]),
            {"k_5_per_exporter", "k_10_per_exporter"},
        )

    def test_external_scores_require_exact_keys_and_no_outcomes(self) -> None:
        candidates = _track_a()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.csv"
            scores = candidates[["i_iso", "j_iso", "stage"]].iloc[::-1].copy()
            scores["score"] = [0.1, 0.9]
            scores.to_csv(path, index=False)
            aligned = v2_eval.load_external_scores(path, candidates, "A")
            self.assertTrue(np.allclose(aligned, [0.9, 0.1]))

            scores.iloc[:1].to_csv(path, index=False)
            with self.assertRaisesRegex(v2_eval.ScoreError, "exactly equal"):
                v2_eval.load_external_scores(path, candidates, "A")

            leaked = candidates[["i_iso", "j_iso", "stage", "y"]].copy()
            leaked["score"] = 0.0
            leaked.to_csv(path, index=False)
            with self.assertRaisesRegex(v2_eval.ScoreError, "forbidden"):
                v2_eval.load_external_scores(path, candidates, "A")

    def test_forbidden_builtin_outcomes(self) -> None:
        with self.assertRaisesRegex(v2_eval.ScoreError, "forbidden"):
            v2_eval.builtin_scores(_track_a(), "A", "lateval")

    def test_track_b2_reports_per_entry_macro_recall_and_value(self) -> None:
        frame = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B", "B"],
                "stage": ["s", "s", "s", "s", "s"],
                "j_iso": ["a", "b", "a", "b", "c"],
                "y": [1, 0, 0, 1, 1],
                "lateval": [10.0, 0.0, 0.0, 4.0, 6.0],
            }
        )
        metrics = v2_eval.evaluate_track_b2(frame, [2, 1, 3, 2, 1])
        self.assertEqual(metrics["entries"], 2)
        self.assertAlmostEqual(metrics["at_k"]["k_1"]["macro_recall"], 0.5)
        self.assertAlmostEqual(metrics["at_k"]["k_1"]["macro_value_captured"], 0.5)
        self.assertEqual(metrics["at_k"]["k_5"]["macro_recall"], 1.0)


class AttestationTests(unittest.TestCase):
    def _document(
        self,
        *,
        benchmark_data_sha256: str = "0" * 64,
        score_csv_sha256: str = "1" * 64,
        selection_config_sha256: str = "2" * 64,
    ) -> dict[str, object]:
        return {
            "schema_version": v2_eval.ATTESTATION_SCHEMA_VERSION,
            "attestation_type": v2_eval.ATTESTATION_TYPE,
            "benchmark_version": v2_loader.VERSION,
            "protocol": v2_eval.PROTOCOL,
            "submission_name": "fixture",
            "run_id": "fixture-run-001",
            "exact_seed_list": [0, 17, 2026],
            "selected_on_snapshot": "fold2",
            "choices_frozen_before_main_evaluation": True,
            "main_labels_used_for_selection": False,
            "main_labels_used_for_feature_fitting": False,
            "main_labels_used_for_imputation_or_calibration": False,
            "benchmark_data_sha256": benchmark_data_sha256,
            "score_csv_sha256": score_csv_sha256,
            "selection_config_sha256": selection_config_sha256,
            "attested_by": "test",
        }

    def test_main_external_requires_valid_attestation_or_explicit_override(self) -> None:
        with self.assertRaisesRegex(v2_eval.ProtocolAttestationError, "require.*--attestation"):
            v2_eval._protocol_record(snapshot="main", external=True, attestation=None, diagnostic_override=None)
        diagnostic = v2_eval._protocol_record(
            snapshot="main", external=True, attestation=None, diagnostic_override="same-window ablation"
        )
        self.assertFalse(diagnostic["official"])

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attestation.json"
            data_path = Path(temp) / "benchmark.csv"
            score_path = Path(temp) / "scores.csv"
            config_path = Path(temp) / "selection.json"
            data_path.write_bytes(b"frozen benchmark bytes\n")
            score_path.write_bytes(b"frozen score bytes\n")
            config_path.write_bytes(b'{"model":"fixture"}\n')
            document = self._document(
                benchmark_data_sha256=v2_eval.sha256_file(data_path),
                score_csv_sha256=v2_eval.sha256_file(score_path),
                selection_config_sha256=v2_eval.sha256_file(config_path),
            )
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                v2_eval.ProtocolAttestationError, "requires --selection-config"
            ):
                v2_eval._protocol_record(
                    snapshot="main",
                    external=True,
                    attestation=path,
                    diagnostic_override=None,
                    benchmark_data_path=data_path,
                    score_csv_path=score_path,
                )
            official = v2_eval._protocol_record(
                snapshot="main",
                external=True,
                attestation=path,
                diagnostic_override=None,
                benchmark_data_path=data_path,
                score_csv_path=score_path,
                selection_config_path=config_path,
            )
            self.assertTrue(official["official"])
            self.assertEqual(official["attestation_type"], "schema_checked_self_attestation")
            self.assertIn("not independently", official["verification_scope"])

            score_path.write_bytes(b"changed score bytes\n")
            with self.assertRaisesRegex(v2_eval.ProtocolAttestationError, "score_csv_sha256"):
                v2_eval._protocol_record(
                    snapshot="main",
                    external=True,
                    attestation=path,
                    diagnostic_override=None,
                    benchmark_data_path=data_path,
                    score_csv_path=score_path,
                    selection_config_path=config_path,
                )

            document = self._document()
            document["main_labels_used_for_selection"] = True
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(v2_eval.ProtocolAttestationError, "main_labels_used_for_selection"):
                v2_eval.validate_attestation(path)

    def test_main_score_request_binds_loaded_table_score_and_config_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / v2_loader.filename_for("A", "sheep", "main")
            candidates = _track_a()
            candidates.to_csv(data_path, index=False)
            score_path = root / "scores.csv"
            scores = candidates[["i_iso", "j_iso", "stage"]].copy()
            scores["score"] = [0.9, 0.1]
            scores.to_csv(score_path, index=False)
            config_path = root / "selection.json"
            config_path.write_text(
                json.dumps({"run_id": "fixture-run-001", "exact_seed_list": [0, 17, 2026]}),
                encoding="utf-8",
            )
            document = self._document(
                benchmark_data_sha256=v2_eval.sha256_file(data_path),
                score_csv_sha256=v2_eval.sha256_file(score_path),
                selection_config_sha256=v2_eval.sha256_file(config_path),
            )
            attestation_path = root / "attestation.json"
            attestation_path.write_text(json.dumps(document), encoding="utf-8")

            result = v2_eval.score_request(
                track="A",
                chain="sheep",
                snapshot="main",
                data_root=root,
                scores_path=score_path,
                attestation=attestation_path,
                selection_config_path=config_path,
                budgets=(1,),
            )
            protocol = result["protocol_attestation"]
            self.assertTrue(protocol["official"])
            self.assertEqual(protocol["run_id"], "fixture-run-001")
            self.assertEqual(protocol["exact_seed_list"], [0, 17, 2026])
            self.assertEqual(protocol["benchmark_data_sha256"], v2_eval.sha256_file(data_path))

    def test_self_attestation_schema_requires_run_id_exact_seeds_and_no_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attestation.json"
            document = self._document()
            document["exact_seed_list"] = []
            document["unrecognized_claim"] = True
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                v2_eval.ProtocolAttestationError,
                "unexpected=.*unrecognized_claim.*exact_seed_list",
            ):
                v2_eval.validate_attestation(path)

    def test_shipped_self_attestation_example_matches_schema_v2(self) -> None:
        document = v2_eval.validate_attestation(
            PACKAGE / "protocol_attestation.example.json"
        )
        self.assertEqual(document["attestation_type"], v2_eval.ATTESTATION_TYPE)
        self.assertEqual(document["exact_seed_list"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
