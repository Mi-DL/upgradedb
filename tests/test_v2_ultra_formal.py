import copy
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import v2_ultra_formal as formal  # noqa: E402


PRIVATE_PROVENANCE_TESTS_ENABLED = (
    os.environ.get("UPGRADE_BENCH_PRIVATE_PROVENANCE_TESTS") == "1"
)
_REAL_FORMAL_SHA256_FILE = formal.sha256_file


def _identity_frame():
    return pd.DataFrame(
        {
            "i_iso": ["AAA", "AAA", "BBB"],
            "j_iso": ["BBB", "CCC", "AAA"],
            "stage": ["exp_x", "exp_x", "exp_x"],
        }
    )


def _write_candidate(path: Path, source: str) -> None:
    frame = _identity_frame()
    frame["y"] = ["TARGET", "TARGET", "TARGET"]
    frame["entry_y"] = ["TARGET", "TARGET", "TARGET"]
    frame["lateval"] = ["TARGET", "TARGET", "TARGET"]
    frame["size"] = ["TARGET_DERIVED", "TARGET_DERIVED", "TARGET_DERIVED"]
    frame["grav"] = ["FEATURE", "FEATURE", "FEATURE"]
    frame["benchmark_version"] = "2.1-dev"
    frame["aggregation"] = "calendar_mean"
    frame["early_window"] = "2008-2012"
    frame["late_window"] = "2018-2022"
    frame["task"] = formal.SOURCE_SPECS[source]["task"]
    frame["task_unit"] = formal.SOURCE_SPECS[source]["task_unit"]
    frame.to_csv(path, index=False)


def _load_config_against_frozen_trained_reference() -> dict:
    """Validate the historical controller against its exact frozen input hash.

    The live trained-reference summary now has prospectively hardened
    provenance bytes.  The formal r4 controller is intentionally byte-frozen
    and must not be rewritten to claim that the historical run consumed those
    later bytes.  Unit tests for the controller's static contract therefore
    reproduce the hash it actually consumed; a separate test below confirms
    that the live controller still fails closed on the changed artifact.
    """

    payload = formal._load_json(formal.CANONICAL_CONFIG, "formal ULTRA config")
    with mock.patch.object(
        formal, "sha256_file", side_effect=_sha256_with_frozen_trained_reference
    ):
        return formal.validate_config_payload(payload)


def _sha256_with_frozen_trained_reference(path: Path) -> str:
    reference = formal._resolve(formal.REPORTING_CONTRACT["trained_reference_artifact"])
    if Path(path).resolve() == reference.resolve():
        return formal.REPORTING_CONTRACT["trained_reference_artifact_sha256"]
    return _REAL_FORMAL_SHA256_FILE(path)


class V2UltraFormalProtocolTest(unittest.TestCase):
    def test_config_locks_disclosed_4g_checkpoint_metadata(self):
        config = _load_config_against_frozen_trained_reference()
        self.assertEqual(config["chains"], list(formal.CHAINS))
        self.assertEqual(config["checkpoint"]["name"], "ultra_4g")
        self.assertEqual(config["checkpoint"]["bytes"], 2127350)
        self.assertEqual(
            config["checkpoint"]["sha256"],
            "48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0",
        )
        self.assertEqual(
            config["provenance"]["upstream_commit"],
            "427966ad8ed60420eef034063d44f3153addff90",
        )
        self.assertEqual(
            config["provenance"]["checkpoint_release_commit"],
            "68433c19a465735ae59f7f947e6dd062bab7b445",
        )
        self.assertEqual(
            config["provenance"]["checkpoint_git_blob_sha1"],
            "b10ca4f3a26c5084a6ec72e77dfc756b04acef4c",
        )
        self.assertEqual(
            config["provenance"]["checkpoint_training_config_git_blob"],
            "073ce8a72c20474006a0adc861b1f80bda42f7c7",
        )
        self.assertEqual(
            config["training_disclosure"]["pretraining_graphs"],
            ["FB15k237", "WN18RR", "CoDExMedium", "NELL995"],
        )
        self.assertFalse(
            config["training_disclosure"]["checkpoint_training_seed_disclosed"]
        )
        self.assertEqual(config["training_disclosure"]["reference_cli_default_seed"], 1024)
        self.assertIn(
            "not evidence",
            config["training_disclosure"]["reference_cli_default_seed_scope"],
        )
        self.assertIn(
            "not asserted identical",
            config["training_disclosure"]["vendored_reference_config_status"],
        )
        self.assertFalse(
            config["overlap_policy"]["disclosed_pretraining_list_contains_benchmark_sources"]
        )
        self.assertIn("forbidden", config["overlap_policy"]["ultra_50g_formal_use"])
        self.assertEqual(config["provenance"]["source_tree_receipt"]["status"], "PASS")
        self.assertIn(
            "no current-protocol ULTRA outcome",
            config["provenance"]["selection_basis"],
        )
        self.assertFalse(config["model_policy"]["checkpoint_search"])
        self.assertFalse(config["model_policy"]["fine_tuning"])
        self.assertEqual(config["repeatability_contract"]["inference_seed"], 1024)
        self.assertIn(
            "not the undisclosed checkpoint training seed",
            config["repeatability_contract"]["inference_seed_scope"],
        )
        self.assertEqual(
            config["raw_source_policy"]["archive_sha256"],
            "1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a",
        )
        self.assertEqual(config["reporting_contract"]["report_all_chain_task_headlines"], 18)
        self.assertEqual(
            config["reporting_contract"]["trained_reference_artifact_sha256"],
            "5978ca62462f68ffc93054fd1c448ee768646ec82a8b61b5d843d56559193acd",
        )

    def test_coordinator_scoring_marker_is_strict_canonical_and_byte_hashed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "frozen_manifest.json"
            manifest.write_bytes(formal.canonical_json_bytes({}))
            marker = root / "SCORING_STARTED.json"
            config = {"run_id": formal.CANONICAL_RUN_ID}
            payload = {
                "schema_version": formal.SCORE_START_SCHEMA,
                "protocol": formal.PROTOCOL,
                "run_id": formal.CANONICAL_RUN_ID,
                "manifest_sha256": formal.sha256_file(manifest),
                "started_at_utc": "2026-07-17T01:02:03+00:00",
                "policy": formal.SCORING_START_POLICY,
            }

            with self.assertRaisesRegex(formal.ProtocolError, "cannot read strict"):
                formal.verify_scoring_started(marker, manifest, config)

            marker.write_bytes(formal.canonical_json_bytes(payload))
            observed, digest = formal.verify_scoring_started(marker, manifest, config)
            self.assertEqual(observed, payload)
            self.assertEqual(digest, formal.sha256_file(marker))

            marker.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(formal.ProtocolError, "canonical compact JSON"):
                formal.verify_scoring_started(marker, manifest, config)

            marker.write_bytes(b'{"x":1,"x":2}\n')
            with self.assertRaisesRegex(formal.ProtocolError, "duplicate key"):
                formal.verify_scoring_started(marker, manifest, config)

            changed = dict(payload)
            changed["unexpected"] = True
            marker.write_bytes(formal.canonical_json_bytes(changed))
            with self.assertRaisesRegex(formal.ProtocolError, "fields are not exact"):
                formal.verify_scoring_started(marker, manifest, config)

            changed = dict(payload)
            changed["policy"] = "worker-created marker"
            marker.write_bytes(formal.canonical_json_bytes(changed))
            with self.assertRaisesRegex(formal.ProtocolError, "marker policy"):
                formal.verify_scoring_started(marker, manifest, config)

    def test_score_worker_only_verifies_the_preexisting_coordinator_marker(self):
        source = inspect.getsource(formal._score_chain)
        self.assertIn("verify_scoring_started", source)
        self.assertNotIn("_claim_marker", source)
        self.assertEqual(formal.SCORE_START_SCHEMA.rsplit("/", 1)[-1], "2")
        for schema in (
            formal.SCORE_SEAL_SCHEMA,
            formal.EVALUATION_START_SCHEMA,
            formal.METRIC_SCHEMA,
            formal.EVALUATION_SCHEMA,
        ):
            self.assertEqual(schema.rsplit("/", 1)[-1], "3")

    @unittest.skipUnless(
        PRIVATE_PROVENANCE_TESTS_ENABLED,
        "requires the external ULTRA source and checkpoint provenance",
    )
    def test_checkpoint_and_source_tree_provenance_files(self):
        config = _load_config_against_frozen_trained_reference()
        formal._validate_provenance_files()
        checkpoint = ROOT / config["checkpoint"]["path"]
        self.assertEqual(checkpoint.stat().st_size, config["checkpoint"]["bytes"])
        self.assertEqual(formal.sha256_file(checkpoint), config["checkpoint"]["sha256"])
        self.assertEqual(
            formal.git_blob_sha1(checkpoint),
            config["provenance"]["checkpoint_git_blob_sha1"],
        )

    def test_config_rejects_50g_or_any_checkpoint_search(self):
        config = _load_config_against_frozen_trained_reference()
        changed = copy.deepcopy(config)
        changed["checkpoint"]["name"] = "ultra_50g"
        with self.assertRaisesRegex(formal.ProtocolError, "fixed checkpoint"):
            formal.validate_config_payload(changed)
        changed = copy.deepcopy(config)
        changed["model_policy"]["checkpoint_search"] = True
        with self.assertRaisesRegex(formal.ProtocolError, "model policy"):
            formal.validate_config_payload(changed)
        changed = copy.deepcopy(config)
        changed["output_root"] = "tmp/ultra"
        with mock.patch.object(
            formal, "sha256_file", side_effect=_sha256_with_frozen_trained_reference
        ), self.assertRaisesRegex(formal.ProtocolError, "canonical output_root"):
            formal.validate_config_payload(changed)

    def test_historical_controller_fails_closed_on_later_reference_bytes(self):
        with self.assertRaisesRegex(
            formal.ProtocolError, "trained-reference artifact hash mismatch"
        ):
            formal.load_and_validate_config(formal.CANONICAL_CONFIG)

    def test_full_candidate_bytes_are_precommitted_without_target_parse(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidate.csv"
            _write_candidate(path, "A")
            identities, metadata = formal.read_candidate_identities(path, "A")
            before = formal.stable_file_sha256(path)
            record = formal._cohort_record(
                path,
                identities,
                metadata,
                {"candidate_rows": len(identities)},
                {"candidate_rows": len(identities), "early_forward_trade_triples": 1, "overlap_rows": 0},
            )
            self.assertEqual(record["full_file_sha256"], before)
            self.assertFalse(record["target_columns_semantically_accessed"])
            payload = path.read_text(encoding="utf-8").replace("TARGET", "ALTERED", 1)
            path.write_text(payload, encoding="utf-8")
            self.assertNotEqual(formal.stable_file_sha256(path), before)

    def test_raw_baci_attestation_is_path_independent_and_hash_bound(self):
        archive = mock.Mock()
        archive.is_file.return_value = True
        archive.name = formal.RAW_SOURCE_POLICY["archive_name"]
        archive.stat.return_value = SimpleNamespace(
            st_size=formal.RAW_SOURCE_POLICY["archive_bytes"]
        )
        with mock.patch.object(
            formal, "_resolved_raw_baci_path", return_value=archive
        ), mock.patch.object(
            formal,
            "stable_file_sha256",
            return_value=formal.RAW_SOURCE_POLICY["archive_sha256"],
        ):
            receipt = formal.verify_raw_source()
        self.assertEqual(receipt["archive_bytes"], 2450783074)
        self.assertFalse(receipt["host_path_serialized"])
        self.assertFalse(receipt["gravity_opened_by_formal_graph_builder"])

    def test_candidate_early_forward_trade_overlap_is_forbidden(self):
        identities = _identity_frame()
        triples = np.asarray(
            [["CCC", "exp_x", "BBB"], ["AAA", "other_relation", "BBB"]],
            dtype=str,
        )
        receipt = formal.validate_zero_candidate_early_trade_overlap(
            identities, triples, ["exp_x"]
        )
        self.assertEqual(receipt["overlap_rows"], 0)
        bad = np.asarray([["AAA", "exp_x", "BBB"]], dtype=str)
        with self.assertRaisesRegex(formal.ProtocolError, "overlaps 1"):
            formal.validate_zero_candidate_early_trade_overlap(identities, bad, ["exp_x"])

    def test_identity_reader_is_label_blind_for_both_lane_sources(self):
        real_read_csv = pd.read_csv
        for source in formal.SOURCE_SPECS:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "candidate.csv"
                _write_candidate(path, source)
                requested = []

                def audited_read_csv(*args, **kwargs):
                    requested.append(tuple(kwargs.get("usecols") or ()))
                    return real_read_csv(*args, **kwargs)

                with mock.patch("pandas.read_csv", side_effect=audited_read_csv):
                    identities, metadata = formal.read_candidate_identities(path, source)
                pd.testing.assert_frame_equal(identities, _identity_frame())
                self.assertEqual(metadata["task"], formal.SOURCE_SPECS[source]["task"])
                self.assertTrue(requested)
                for columns in requested:
                    self.assertTrue(set(columns).isdisjoint(formal.FORBIDDEN_TARGET_COLUMNS))

    def test_union_projection_reuses_one_score_for_overlapping_a_and_b_keys(self):
        union = pd.DataFrame(
            {
                "i_iso": ["AAA", "AAA", "BBB", "CCC"],
                "j_iso": ["BBB", "CCC", "AAA", "AAA"],
                "stage": ["exp_x"] * 4,
                "ultra_score": [0.1, 0.2, 0.3, 0.4],
            }
        )
        target = union.iloc[[0, 2]].loc[:, list(formal.KEYS)].reset_index(drop=True)
        projected = formal.project_union_scores(target, union.iloc[[3, 2, 1, 0]])
        pd.testing.assert_frame_equal(projected.loc[:, list(formal.KEYS)], target)
        np.testing.assert_allclose(projected["ultra_score"], [0.1, 0.3])
        missing = target.copy()
        missing.loc[0, "j_iso"] = "ZZZ"
        with self.assertRaisesRegex(formal.ProtocolError, "missing 1"):
            formal.project_union_scores(missing, union)

    def test_a_direct_b1_max_lane_and_b2_conditional_share_b_scores(self):
        a_ids = pd.DataFrame(
            {
                "i_iso": ["AAA", "AAA", "BBB", "BBB"],
                "j_iso": ["J1", "J2", "J1", "J2"],
                "stage": ["exp_a"] * 4,
            }
        )
        a_labels = a_ids.copy()
        a_labels["y"] = [1, 0, 0, 1]
        a_labels["size"] = [1.0, 2.0, 3.0, 4.0]
        a_labels["lateval"] = [10.0, 0.0, 0.0, 20.0]
        a_score = np.asarray([0.9, 0.1, 0.2, 0.8])

        b_ids = pd.DataFrame(
            {
                "i_iso": ["AAA"] * 4 + ["BBB"] * 4,
                "j_iso": ["J1", "J2", "J3", "J4"] * 2,
                "stage": ["exp_b"] * 8,
            }
        )
        b_labels = b_ids.copy()
        # AAA is a positive entry, but its only positive destination ranks 4th.
        # BBB is a negative entry.  B1 therefore compares max scores .9 vs .8.
        b_labels["y"] = [1, 0, 0, 0, 0, 0, 0, 0]
        b_labels["size"] = np.arange(1.0, 9.0)
        b_labels["lateval"] = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b_score = np.asarray([0.1, 0.9, 0.7, 0.6, 0.8, 0.5, 0.4, 0.3])
        metrics = formal.derive_task_metrics(
            a_ids, a_labels, a_score, b_ids, b_labels, b_score
        )
        self.assertEqual(metrics["A"]["lane_average_precision"], 1.0)
        self.assertEqual(metrics["B1"]["entry_average_precision"], 1.0)
        self.assertEqual(metrics["B1"]["entry_groups"], 2)
        self.assertEqual(metrics["B1"]["entry_value_capture_at_50"], 1.0)
        self.assertEqual(metrics["B2"]["positive_entry_groups"], 1)
        self.assertEqual(metrics["B2"]["conditional_recall_at_3"], 0.0)

    def test_b1_value_budget_uses_sum_value_max_score_and_canonical_ties(self):
        ids = pd.DataFrame(
            {
                "i_iso": [f"E{index:03d}" for index in range(60)],
                "j_iso": ["J"] * 60,
                "stage": ["exp_b"] * 60,
            }
        )
        labels = ids.copy()
        labels["y"] = 1
        labels["size"] = 1.0
        labels["lateval"] = 0.0
        labels.loc[0, "lateval"] = 10.0
        labels.loc[59, "lateval"] = 90.0
        scores = np.full(60, 0.5)
        values = formal.b1_entry_value_metrics(ids, labels, scores)
        self.assertEqual(values["entry_observed_value_kusd"], 100.0)
        self.assertEqual(values["entry_value_capture_at_25"], 0.1)
        self.assertEqual(values["entry_value_capture_at_50"], 0.1)
        self.assertEqual(values["entry_value_capture_at_100"], 1.0)

    def test_score_csv_round_trip_is_bit_exact_and_preserves_tie_order(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            {
                "i_iso": [f"E{index:04d}" for index in range(1000)],
                "j_iso": ["J"] * 1000,
                "stage": ["exp_x"] * 1000,
                "ultra_score": rng.standard_normal(1000).astype(np.float32).astype(np.float64),
            }
        )
        # Exact ties exercise the stable identity tie-break after serialization.
        frame.loc[:9, "ultra_score"] = 0.25
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "scores.csv"
            formal._write_csv_exclusive(path, frame)
            loaded = formal._read_score_file(path, frame.loc[:, list(formal.KEYS)])
        self.assertEqual(
            formal.stable_score_vector_hash(loaded), formal.stable_score_vector_hash(frame)
        )
        expected = np.lexsort(
            (
                frame["j_iso"].to_numpy(),
                frame["stage"].to_numpy(),
                frame["i_iso"].to_numpy(),
                -frame["ultra_score"].to_numpy(),
            )
        )
        observed = np.lexsort(
            (
                loaded["j_iso"].to_numpy(),
                loaded["stage"].to_numpy(),
                loaded["i_iso"].to_numpy(),
                -loaded["ultra_score"].to_numpy(),
            )
        )
        np.testing.assert_array_equal(observed, expected)

    def test_repeatability_score_and_metric_gates_are_predeclared(self):
        primary = _identity_frame()
        primary["ultra_score"] = [0.1, 0.2, 0.3]
        repeat = primary.copy()
        repeat["ultra_score"] += 1e-8
        gate = formal.score_repeatability(primary, repeat)
        self.assertTrue(gate["numeric_allclose"])
        self.assertFalse(gate["exact_hash_equality"])
        metrics = {
            "A": {"lane_average_precision": 0.1, "value_capture_at_500": 0.2},
            "B1": {"entry_average_precision": 0.3, "entry_value_capture_at_50": 0.4},
            "B2": {"conditional_recall_at_3": 0.5, "conditional_value_capture_at_3": 0.6},
        }
        metric_gate = formal.repeat_metric_gate(metrics, copy.deepcopy(metrics))
        self.assertTrue(metric_gate["all_metrics_pass"])

    def test_target_reader_rejects_null_nonbinary_and_negative_values(self):
        base = _identity_frame()
        base["y"] = [0, 1, 0]
        base["size"] = [1.0, 2.0, 3.0]
        base["lateval"] = [0.0, 4.0, 0.0]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "target.csv"
            base.to_csv(path, index=False)
            clean = formal._read_target_labels(path, _identity_frame())
            self.assertEqual(clean["y"].tolist(), [0, 1, 0])
            for column, value, message in (
                ("y", np.nan, "contains nulls"),
                ("y", 2, "binary"),
                ("size", -1, "finite and nonnegative"),
                ("lateval", np.inf, "finite and nonnegative"),
            ):
                bad = base.copy()
                bad.loc[1, column] = value
                bad.to_csv(path, index=False)
                with self.assertRaisesRegex(formal.ProtocolError, message):
                    formal._read_target_labels(path, _identity_frame())

    def test_exact_six_chain_completion_is_required(self):
        formal.require_complete_component_set(list(formal.CHAINS))
        with self.assertRaisesRegex(formal.ProtocolError, "labels remain locked"):
            formal.require_complete_component_set(list(formal.CHAINS[:-1]))
        with self.assertRaisesRegex(formal.ProtocolError, "labels remain locked"):
            formal.require_complete_component_set(list(reversed(formal.CHAINS)))

    def test_freeze_chain_mapping_accepts_canonical_json_key_sorting(self):
        formal.require_exact_chain_mapping({chain: {} for chain in sorted(formal.CHAINS)})
        with self.assertRaisesRegex(formal.ProtocolError, "six-chain matrix"):
            formal.require_exact_chain_mapping({chain: {} for chain in formal.CHAINS[:-1]})
        with self.assertRaisesRegex(formal.ProtocolError, "six-chain matrix"):
            formal.require_exact_chain_mapping(
                {**{chain: {} for chain in formal.CHAINS}, "extra": {}}
            )

    def test_score_seal_structure_and_unlock_policy_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            config = {"run_id": formal.CANONICAL_RUN_ID, "output_root": str(root)}
            scoring_started_sha256 = "a" * 64
            seal = {
                "schema_version": formal.SCORE_SEAL_SCHEMA,
                "protocol": formal.PROTOCOL,
                "status": "all_six_chains_scored_labels_unlocked",
                "run_id": formal.CANONICAL_RUN_ID,
                "manifest_sha256": formal.sha256_file(manifest),
                "scoring_started_sha256": scoring_started_sha256,
                "main_target_labels_accessed_before_seal": False,
                "repeatability_contract": formal.REPEATABILITY_CONTRACT,
                "sentinel_repeat_verified_before_label_unlock": True,
                "component_count": len(formal.CHAINS),
                "unlock_policy": formal.SCORE_UNLOCK_POLICY,
                "components": [{} for _ in formal.CHAINS],
            }
            patches = (
                mock.patch.object(formal, "load_and_validate_config", return_value=config),
                mock.patch.object(formal, "verify_freeze_manifest"),
                mock.patch.object(
                    formal,
                    "verify_scoring_started",
                    return_value=({}, scoring_started_sha256),
                ),
                mock.patch.object(formal, "_load_json", return_value=seal),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                seal["schema_version"] = "upgrade-bench-v2/ultra-formal-score-seal/2"
                with self.assertRaisesRegex(formal.ProtocolError, "score seal schema"):
                    formal.verify_score_seal(manifest, manifest, manifest)
                seal["schema_version"] = formal.SCORE_SEAL_SCHEMA
                seal["scoring_started_sha256"] = "b" * 64
                with self.assertRaisesRegex(formal.ProtocolError, "scoring-start hash"):
                    formal.verify_score_seal(manifest, manifest, manifest)
                seal["scoring_started_sha256"] = scoring_started_sha256
                seal["component_count"] = len(formal.CHAINS) - 1
                with self.assertRaisesRegex(formal.ProtocolError, "component count"):
                    formal.verify_score_seal(manifest, manifest, manifest)
                seal["component_count"] = len(formal.CHAINS)
                seal["unlock_policy"] = "labels are already available"
                with self.assertRaisesRegex(formal.ProtocolError, "unlock policy"):
                    formal.verify_score_seal(manifest, manifest, manifest)
                seal["unlock_policy"] = formal.SCORE_UNLOCK_POLICY
                seal["components"][-1] = "not-a-mapping"
                with self.assertRaisesRegex(formal.ProtocolError, "six mapping components"):
                    formal.verify_score_seal(manifest, manifest, manifest)

    def test_vendored_module_names_are_bound_to_canonical_paths(self):
        self.assertEqual(
            formal.CANONICAL_VENDORED_MODULE_PATHS["ultra.rspmm.rspmm"],
            "ultra/rspmm/rspmm.py",
        )
        with self.assertRaisesRegex(formal.ProtocolError, "unrecognized vendored ULTRA module"):
            formal._vendored_ultra_module_receipt("ultra.unlocked_module")

    def test_reporting_lock_emits_all_18_values_and_applies_abstract_rule(self):
        metrics = {
            chain: {
                "A": {"lane_average_precision": 0.9, "value_capture_at_500": 0.4},
                "B1": {"entry_average_precision": 0.8, "entry_value_capture_at_50": 0.5},
                "B2": {
                    "conditional_recall_at_3": 0.7,
                    "conditional_value_capture_at_3": 0.6,
                },
            }
            for chain in formal.CHAINS
        }
        references = {
            family: {
                task: {chain: value for chain in formal.CHAINS}
                for task, value in (("A", 0.2), ("B1", 0.3), ("B2", 0.4))
            }
            for family in ("kge", "nbfnet")
        }
        with mock.patch.object(formal, "_trained_reference_values", return_value=references):
            summary = formal.build_reporting_summary(metrics)
        self.assertEqual(len(summary["all_18_chain_task_headlines"]), 18)
        self.assertEqual(summary["unweighted_six_chain_value_means"]["B1"], 0.5)
        self.assertEqual(
            summary["trained_reference_comparisons"]["A"]["kge"]["counts"]["higher"],
            6,
        )
        self.assertTrue(summary["abstract_should_mention_ultra"])

    def test_evaluate_never_reads_labels_when_score_seal_fails(self):
        config = _load_config_against_frozen_trained_reference()
        args = SimpleNamespace(config=formal.CANONICAL_CONFIG)
        with mock.patch.object(
            formal, "load_and_validate_config", return_value=config
        ), mock.patch.object(
            formal, "verify_score_seal", side_effect=formal.ProtocolError("incomplete scores")
        ), mock.patch.object(formal, "_read_target_labels") as target_reader:
            with self.assertRaisesRegex(formal.ProtocolError, "incomplete scores"):
                formal._evaluate(args)
        target_reader.assert_not_called()
        self.assertEqual(config["phase_order"][-1], formal.PHASE_ORDER[-1])

    def test_evaluate_rechecks_scoring_marker_before_any_target_label_read(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = {"run_id": formal.CANONICAL_RUN_ID, "output_root": str(root)}
            args = SimpleNamespace(config=root / "config.json")
            with mock.patch.object(
                formal, "load_and_validate_config", return_value=config
            ), mock.patch.object(
                formal,
                "verify_score_seal",
                return_value={"scoring_started_sha256": "a" * 64},
            ), mock.patch.object(
                formal, "verify_freeze_manifest", return_value={}
            ), mock.patch.object(
                formal, "sha256_file", return_value="b" * 64
            ), mock.patch.object(
                formal,
                "verify_scoring_started",
                side_effect=formal.ProtocolError("replaced scoring marker"),
            ), mock.patch.object(formal, "_read_target_labels") as target_reader:
                with self.assertRaisesRegex(formal.ProtocolError, "replaced scoring marker"):
                    formal._evaluate(args)
            target_reader.assert_not_called()

    def test_synthetic_torch_scatter_shim_is_rejected(self):
        previous = sys.modules.get("torch_scatter")
        shim = types.ModuleType("torch_scatter")
        shim.scatter = lambda *args, **kwargs: None
        try:
            sys.modules["torch_scatter"] = shim
            with self.assertRaisesRegex(formal.ProtocolError, "compatibility shim"):
                formal._require_native_torch_scatter()
        finally:
            if previous is None:
                sys.modules.pop("torch_scatter", None)
            else:
                sys.modules["torch_scatter"] = previous

    def test_cli_has_no_checkpoint_selection_or_finetuning_override(self):
        parser = formal.build_parser()
        option_strings = set()
        pending = [parser]
        while pending:
            current = pending.pop()
            for action in current._actions:
                option_strings.update(action.option_strings)
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict):
                    pending.extend(choices.values())
        self.assertIn("--device", option_strings)
        self.assertNotIn("--checkpoint", option_strings)
        self.assertNotIn("--checkpoints", option_strings)
        self.assertNotIn("--selection", option_strings)
        self.assertNotIn("--finetune", option_strings)
        parsed = parser.parse_args(["score-chain", "--chain", "sheep", "--device", "cuda:1"])
        self.assertEqual(parsed.chain, "sheep")
        self.assertEqual(parsed.device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
