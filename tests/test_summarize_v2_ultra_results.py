import copy
import csv
import io
import inspect
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import summarize_v2_ultra_results as summary_module  # noqa: E402
from summarize_v2_ultra_results import (  # noqa: E402
    CHAINS,
    CLAIM_BOUNDARIES,
    DEFAULT_CSV_OUT,
    DEFAULT_JSON_OUT,
    DESIGN,
    FAMILIES,
    HEADLINE_METRICS,
    MODEL,
    PUBLIC_SUMMARY_SCHEMA,
    REPEAT_METRICS,
    TASKS,
    VALUE_METRICS,
    ResultValidationError,
    build_summary,
    render_csv,
    render_json,
    validate_public_summary,
    verify_outputs,
    write_outputs,
)


class UltraR4ScoringMarkerContractTest(unittest.TestCase):
    def test_scoring_marker_is_canonical_exact_and_bound_to_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker_path = root / "SCORING_STARTED.json"
            manifest_sha256 = "a" * 64
            marker = {
                "schema_version": summary_module.SCORING_START_SCHEMA,
                "protocol": summary_module.PROTOCOL,
                "run_id": summary_module.RUN_ID,
                "manifest_sha256": manifest_sha256,
                "started_at_utc": "2026-07-17T01:02:03+00:00",
                "policy": summary_module.SCORING_START_POLICY,
            }

            with self.assertRaisesRegex(ResultValidationError, "cannot read"):
                summary_module._validate_scoring_started(root, manifest_sha256)

            marker_path.write_bytes(summary_module._canonical_json_bytes(marker))
            observed, digest = summary_module._validate_scoring_started(
                root, manifest_sha256
            )
            self.assertEqual(observed, marker)
            self.assertEqual(digest, summary_module._sha256_file(marker_path))

            changed = dict(marker)
            changed["unexpected"] = True
            marker_path.write_bytes(summary_module._canonical_json_bytes(changed))
            with self.assertRaisesRegex(ResultValidationError, "fields are not exact"):
                summary_module._validate_scoring_started(root, manifest_sha256)

            marker_path.write_bytes(
                (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode("utf-8")
            )
            with self.assertRaisesRegex(ResultValidationError, "canonical compact JSON"):
                summary_module._validate_scoring_started(root, manifest_sha256)

            marker_path.write_bytes(summary_module._canonical_json_bytes(marker))
            with self.assertRaisesRegex(ResultValidationError, "manifest_sha256"):
                summary_module._validate_scoring_started(root, "b" * 64)

    def test_r4_private_schemas_and_public_receipt_bind_the_marker(self):
        for schema in (
            summary_module.SCORE_SEAL_SCHEMA,
            summary_module.EVALUATION_START_SCHEMA,
            summary_module.METRIC_SCHEMA,
            summary_module.EVALUATION_SCHEMA,
        ):
            self.assertEqual(schema.rsplit("/", 1)[-1], "3")
        self.assertEqual(summary_module.SCORING_START_SCHEMA.rsplit("/", 1)[-1], "2")
        self.assertIn("scoring_started_sha256", inspect.getsource(build_summary))
        self.assertIn(
            "scoring_started_sha256", inspect.getsource(validate_public_summary)
        )


class UltraTrainedReferenceBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trained, cls.trained_raw = summary_module._strict_json_file(
            summary_module.CANONICAL_TRAINED_SUMMARY,
            "test trained reference summary",
        )
        cls.observed_sha256 = summary_module._sha256_bytes(cls.trained_raw)
        config, _raw = summary_module._strict_json_file(
            summary_module.CANONICAL_CONFIG,
            "test formal config",
        )
        cls.frozen_sha256 = config["reporting_contract"][
            "trained_reference_artifact_sha256"
        ]

    def test_exact_reverse_projection_uses_strict_current_attestation(self):
        with mock.patch.object(
            summary_module.postfreeze,
            "verify_summary_binding",
            return_value={"status": "PASS"},
        ) as verify:
            summary_module._verify_current_trained_reference_bridge(
                copy.deepcopy(self.trained),
                self.trained_raw,
                self.observed_sha256,
                self.frozen_sha256,
            )
        verify.assert_called_once()
        self.assertTrue(verify.call_args.kwargs["require_full_inventory"])
        self.assertEqual(
            verify.call_args.kwargs["artifact_path"],
            ROOT / summary_module.postfreeze.ARTIFACT_ROLE,
        )

    def test_loader_preserves_frozen_formal_reference_provenance(self):
        with mock.patch.object(
            summary_module,
            "_verify_current_trained_reference_bridge",
        ) as bridge:
            _config, _config_sha256, trained, provenance_sha256 = (
                summary_module._load_config_and_references()
            )
        bridge.assert_called_once()
        self.assertEqual(trained, self.trained)
        self.assertNotEqual(self.observed_sha256, self.frozen_sha256)
        self.assertEqual(provenance_sha256, self.frozen_sha256)

    def test_nonallowlisted_scientific_change_is_rejected(self):
        tampered = copy.deepcopy(self.trained)
        tampered["records"][0]["primary_mean"] += 0.001
        raw = summary_module._gpu_summary_json_bytes(tampered)
        with mock.patch.object(
            summary_module.postfreeze,
            "verify_summary_binding",
            return_value={"status": "PASS"},
        ), self.assertRaisesRegex(ResultValidationError, "reverse projection"):
            summary_module._verify_current_trained_reference_bridge(
                tampered,
                raw,
                summary_module._sha256_bytes(raw),
                self.frozen_sha256,
            )

    def test_current_postfreeze_binding_error_is_rejected(self):
        with mock.patch.object(
            summary_module.postfreeze,
            "verify_summary_binding",
            side_effect=summary_module.postfreeze.AttestationError("binding mismatch"),
        ), self.assertRaisesRegex(
            ResultValidationError, "current post-freeze attestation failed"
        ):
            summary_module._verify_current_trained_reference_bridge(
                copy.deepcopy(self.trained),
                self.trained_raw,
                self.observed_sha256,
                self.frozen_sha256,
            )


@unittest.skipUnless(
    os.environ.get("UPGRADE_BENCH_PRIVATE_PROVENANCE_TESTS") == "1",
    "requires private formal ULTRA provenance",
)
class SummarizeV2UltraResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = build_summary()

    def test_real_r4_extraction_promotes_exact_complete_matrix(self):
        summary = self.summary
        self.assertEqual(summary["schema_version"], PUBLIC_SUMMARY_SCHEMA)
        self.assertEqual(summary["model"], MODEL)
        self.assertEqual(summary["design"], DESIGN)
        self.assertEqual(summary["claim_boundaries"], CLAIM_BOUNDARIES)
        self.assertEqual(len(summary["metric_records"]), 18)
        expected = [(task, chain) for task in TASKS for chain in CHAINS]
        self.assertEqual(
            [(row["task"], row["chain"]) for row in summary["metric_records"]],
            expected,
        )
        for row in summary["metric_records"]:
            self.assertEqual(row["headline_metric"], HEADLINE_METRICS[row["task"]])
            self.assertEqual(row["value_metric"], VALUE_METRICS[row["task"]])

    def test_unrounded_means_counts_and_abstract_rule_match_formal_result(self):
        summary = self.summary
        self.assertEqual(
            summary["task_summaries"]["A"]["unweighted_six_chain_headline_mean"],
            0.059251727441747525,
        )
        self.assertEqual(
            summary["task_summaries"]["B1"]["unweighted_six_chain_value_mean"],
            0.49645887628510699,
        )
        self.assertEqual(
            summary["task_summaries"]["B2"]["unweighted_six_chain_headline_mean"],
            0.049525949690756993,
        )
        self.assertEqual(
            summary["reference_comparisons"]["A"]["kge"]["counts"],
            {"higher": 0, "equal": 0, "lower": 6},
        )
        self.assertEqual(
            summary["reference_comparisons"]["B1"]["nbfnet"]["counts"],
            {"higher": 4, "equal": 0, "lower": 2},
        )
        rule = summary["abstract_rule"]
        self.assertTrue(rule["abstract_should_mention_ultra"])
        self.assertTrue(rule["tasks"]["A"]["eligible_for_abstract_mention"])
        self.assertFalse(rule["tasks"]["B1"]["eligible_for_abstract_mention"])
        self.assertTrue(rule["tasks"]["B2"]["eligible_for_abstract_mention"])
        self.assertEqual(rule["tasks"]["A"]["same_side_of_both_chain_count"], 6)
        self.assertEqual(rule["tasks"]["B2"]["side"], "lower")

    def test_sheep_repeat_is_exact_without_publishing_score_hashes(self):
        repeat = self.summary["sheep_exact_repeat"]
        self.assertTrue(repeat["score_gate_pass"])
        self.assertTrue(repeat["metric_gate_pass"])
        self.assertTrue(repeat["A_score_file_bytes_equal"])
        self.assertTrue(repeat["B_score_file_bytes_equal"])
        self.assertTrue(repeat["score_vector_hash_equal"])
        self.assertEqual(repeat["max_absolute_score_delta"], 0.0)
        self.assertEqual(repeat["mean_absolute_score_delta"], 0.0)
        self.assertEqual(set(repeat["metric_absolute_deltas"]), set(REPEAT_METRICS))
        self.assertTrue(all(value == 0.0 for value in repeat["metric_absolute_deltas"].values()))

    def test_public_payload_excludes_locators_target_hashes_and_raw_score_receipts(self):
        rendered = render_json(self.summary).decode("utf-8")
        lowered = rendered.lower()
        self.assertNotIn('"path"', lowered)
        self.assertNotIn('"host"', lowered)
        self.assertNotIn('"candidate', lowered)
        self.assertNotIn("score_vector_sha256", lowered)
        self.assertNotIn("score_artifact_sha256", lowered)
        self.assertIsNone(re.search(r"/(?:home|users)/[^/\s]+/", rendered, re.I))
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]+Users[\\/]+", rendered, re.I))
        self.assertIsNone(re.search(r"\bmars\d+\b", rendered, re.I))

        private_root = summary_module.CANONICAL_PRIVATE_ROOT
        target_hashes = set()
        for chain in CHAINS:
            metric = json.loads(
                (private_root / "metrics" / f"metrics_{chain}.json").read_text(encoding="utf-8")
            )
            for source in ("A", "B"):
                target_hashes.add(metric["target_sources"][source]["sha256"])
                target_hashes.add(metric["target_sources"][source]["precommitted_sha256"])
        self.assertTrue(target_hashes)
        self.assertTrue(all(value not in rendered for value in target_hashes))

    def test_provenance_binds_public_dependencies_and_formal_receipts(self):
        provenance = self.summary["provenance"]
        self.assertEqual(set(provenance["chain_metric_artifact_sha256"]), set(CHAINS))
        for field in (
            "config_sha256",
            "formal_controller_sha256",
            "generator_tool_sha256",
            "trained_reference_summary_sha256",
            "frozen_manifest_sha256",
            "scoring_started_sha256",
            "score_seal_sha256",
            "evaluation_start_marker_sha256",
            "evaluation_sha256",
            "checkpoint_sha256",
            "native_runtime_sha256",
            "formal_receipt_set_sha256",
        ):
            self.assertRegex(provenance[field], r"^[0-9a-f]{64}$")

    def test_validate_public_summary_recomputes_and_rejects_tampering(self):
        clean = validate_public_summary(copy.deepcopy(self.summary))
        self.assertEqual(clean, self.summary)

        tampered = copy.deepcopy(self.summary)
        tampered["metric_records"][0]["headline_value"] += 0.01
        with self.assertRaisesRegex(ResultValidationError, "numeric mismatch"):
            validate_public_summary(tampered)

        tampered = copy.deepcopy(self.summary)
        tampered["provenance"]["formal_controller_sha256"] = "0" * 64
        with self.assertRaisesRegex(ResultValidationError, "dependency hash"):
            validate_public_summary(tampered)

        tampered = copy.deepcopy(self.summary)
        tampered["model"]["checkpoint_training_seed_disclosed"] = True
        with self.assertRaisesRegex(ResultValidationError, "fixed public contract"):
            validate_public_summary(tampered)

    def test_csv_is_canonical_allowlist_with_30_rows(self):
        raw = render_csv(self.summary)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(row["row_type"] == "chain_task" for row in rows), 18)
        self.assertEqual(sum(row["row_type"] == "task_summary" for row in rows), 3)
        self.assertEqual(sum(row["row_type"] == "reference_comparison" for row in rows), 6)
        self.assertEqual(sum(row["row_type"] == "abstract_rule" for row in rows), 3)
        self.assertNotIn("path", rows[0])
        self.assertNotIn("score_artifact_sha256", rows[0])

    def test_verify_output_uses_no_extracted_formal_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            json_out = root / "summary.json"
            csv_out = root / "summary.csv"
            json_out.write_bytes(render_json(self.summary))
            csv_out.write_bytes(render_csv(self.summary))
            with mock.patch.object(
                summary_module,
                "build_summary",
                side_effect=AssertionError("public verifier touched extracted state"),
            ), mock.patch.object(
                summary_module,
                "CANONICAL_PRIVATE_ROOT",
                root / "does-not-exist",
            ):
                verified = verify_outputs(json_out, csv_out)
            self.assertEqual(verified, self.summary)

            csv_out.write_bytes(csv_out.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ResultValidationError, "stale"):
                verify_outputs(json_out, csv_out)

    def test_write_outputs_resumes_identical_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            json_out = root / "summary.json"
            csv_out = root / "summary.csv"
            written = write_outputs(
                summary_module.CANONICAL_PRIVATE_ROOT, json_out, csv_out
            )
            self.assertEqual(written, self.summary)
            resumed = write_outputs(
                summary_module.CANONICAL_PRIVATE_ROOT, json_out, csv_out
            )
            self.assertEqual(resumed, self.summary)
            json_out.write_text("different", encoding="utf-8")
            with self.assertRaisesRegex(ResultValidationError, "refusing to overwrite"):
                write_outputs(summary_module.CANONICAL_PRIVATE_ROOT, json_out, csv_out)

    def test_canonical_output_constants_are_public_metrics_only(self):
        self.assertEqual(DEFAULT_JSON_OUT.name, "v2_ultra_zero_shot_summary.json")
        self.assertEqual(DEFAULT_CSV_OUT.name, "v2_ultra_zero_shot_summary.csv")
        self.assertEqual(DEFAULT_JSON_OUT.parent, ROOT / "results_v2" / "metrics")
        self.assertEqual(DEFAULT_CSV_OUT.parent, ROOT / "results_v2" / "metrics")
        self.assertEqual(set(self.summary["reference_comparisons"]["A"]), set(FAMILIES))


if __name__ == "__main__":
    unittest.main()
