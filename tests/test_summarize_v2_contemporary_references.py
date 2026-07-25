from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import summarize_v2_contemporary_references as contemporary  # noqa: E402


class ContemporaryRepositoryVerificationTest(unittest.TestCase):
    def test_repository_profile_is_self_contained_and_returns_summary(self):
        with mock.patch.object(
            contemporary,
            "_validate_formal_envelope",
            side_effect=AssertionError("repository verification touched tmp"),
        ):
            summary = contemporary.verify_outputs(profile="repository")
        self.assertEqual(summary["schema_version"], contemporary.PUBLIC_SCHEMA)
        self.assertEqual(summary["status"], contemporary.PUBLIC_STATUS)
        self.assertEqual(summary["record_count"], 90)
        self.assertEqual(summary["aggregate_count"], 15)

    def test_public_matrix_is_exactly_five_by_six_by_three_with_five_seed_sd(self):
        summary = contemporary.verify_outputs(profile="repository")
        expected = [
            (method, chain, task)
            for method in contemporary.METHODS
            for chain in contemporary.CHAINS
            for task in contemporary.TASKS
        ]
        self.assertEqual(
            [(row["method"], row["chain"], row["task"]) for row in summary["records"]],
            expected,
        )
        self.assertTrue(all(row["seed_count"] == 5 for row in summary["records"]))
        self.assertTrue(
            all(
                0.0 <= row[field] <= 1.0
                for row in summary["records"]
                for field in (
                    "headline_mean",
                    "headline_population_sd",
                    "value_capture_mean",
                    "value_capture_population_sd",
                )
            )
        )

    def test_six_chain_means_sd_and_required_tex_macros_are_exact(self):
        summary = contemporary.verify_outputs(profile="repository")
        aggregates = {
            (row["method"], row["task"]): row for row in summary["aggregates"]
        }
        self.assertAlmostEqual(
            aggregates[("motif", "B1")]["value_capture_unweighted_six_chain_mean"],
            0.6042033369883602,
            places=15,
        )
        self.assertEqual(
            aggregates[("motif", "B1")][
                "value_capture_population_sd_across_seed_macros"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            aggregates[("flock", "B1")][
                "value_capture_population_sd_across_seed_macros"
            ],
            0.04075401588295299,
            places=15,
        )
        self.assertAlmostEqual(
            aggregates[("tabm", "B2")]["headline_unweighted_six_chain_mean"],
            0.22169533063056276,
            places=15,
        )
        self.assertAlmostEqual(
            aggregates[("tabm", "B2")][
                "headline_population_sd_across_seed_macros"
            ],
            0.008122985436901582,
            places=15,
        )
        self.assertAlmostEqual(
            aggregates[("tabiclv2", "A")][
                "value_capture_unweighted_six_chain_mean"
            ],
            0.23433392123283836,
            places=15,
        )
        self.assertAlmostEqual(
            aggregates[("tabiclv2", "A")][
                "value_capture_population_sd_across_seed_macros"
            ],
            0.004841018387727444,
            places=15,
        )

        macros = contemporary.parse_tex_macros(contemporary.DEFAULT_TEX_OUT.read_bytes())
        self.assertEqual(macros["VTwoContemporaryMethodCount"], "5")
        self.assertEqual(macros["VTwoContemporarySeedCount"], "5")
        self.assertEqual(macros["VTwoContemporaryRecordCount"], "90")
        self.assertEqual(macros["VTwoContemporaryAggregateCount"], "15")
        self.assertEqual(macros["VTwoContemporaryMOTIFTrackBOneValue"], "0.6042")
        self.assertEqual(macros["VTwoContemporaryMOTIFTrackBOneValueSD"], "0.0000")
        self.assertEqual(macros["VTwoContemporaryFlockTrackBOneValueSD"], "0.0408")
        self.assertEqual(macros["VTwoContemporaryTabICLvTwoTrackAValue"], "0.2343")
        self.assertEqual(macros["VTwoContemporaryTabICLvTwoTrackAValueSD"], "0.0048")
        expected_metric_macros = {
            f"VTwoContemporary{contemporary.METHOD_TEX[method]}"
            f"Track{contemporary.TASK_TEX[task]}{suffix}"
            for method in contemporary.METHODS
            for task in contemporary.TASKS
            for suffix in ("Headline", "HeadlineSD", "Value", "ValueSD")
        }
        self.assertTrue(expected_metric_macros.issubset(macros))

    def test_config_declares_fixed_five_seed_scope_without_common_uncertainty_claim(self):
        config, _raw = contemporary._load_config()
        self.assertEqual(config["benchmark_version"], "2.1-dev")
        self.assertEqual(config["benchmark"]["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(
            config["benchmark"]["aggregation_contract"],
            contemporary.AGGREGATION_CONTRACT,
        )
        claim = config["claim_scope"]
        self.assertEqual(
            claim["classification"], "contemporary fixed-configuration references"
        )
        self.assertTrue(claim["historical_labels_used_for_model_fitting"])
        self.assertFalse(
            claim[
                "main_window_target_outcomes_used_for_training_or_configuration_selection"
            ]
        )
        self.assertFalse(
            claim["historical_outcome_metrics_used_for_configuration_selection"]
        )
        self.assertFalse(claim["training_uncertainty_comparable_across_methods"])
        self.assertIn("scores were sealed", claim["evaluation_ordering"])
        self.assertNotIn("original_prespecified_reference_set", claim)
        serialized = json.dumps(config).lower()
        self.assertNotIn("preregister", serialized)
        self.assertNotIn("prespecified", serialized)
        for stale in ("post-freeze", "post_freeze", "exploratory", "single-point"):
            self.assertNotIn(stale, serialized)
        self.assertEqual(
            config["formal_evidence"]["final_reporting_receipt"]["sha256"],
            "e9f6b09897974c12b08baac00cf07ad59ae74756e824b5480514ca4ebb0c4e04",
        )
        self.assertEqual(
            config["formal_evidence"]["sanitized_record_matrix_sha256"],
            contemporary.EXPECTED_RECORD_MATRIX_SHA256,
        )
        self.assertEqual(
            config["formal_evidence"]["sanitized_aggregate_matrix_sha256"],
            contemporary.EXPECTED_AGGREGATE_MATRIX_SHA256,
        )
        contemporary._privacy_audit(config)

    def _copy_outputs(self, root: Path) -> tuple[Path, Path, Path]:
        json_path = root / "summary.json"
        csv_path = root / "summary.csv"
        tex_path = root / "summary.tex"
        shutil.copyfile(contemporary.DEFAULT_JSON_OUT, json_path)
        shutil.copyfile(contemporary.DEFAULT_CSV_OUT, csv_path)
        shutil.copyfile(contemporary.DEFAULT_TEX_OUT, tex_path)
        return json_path, csv_path, tex_path

    def test_tampered_json_record_mean_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            json_path, csv_path, tex_path = self._copy_outputs(Path(raw))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["records"][0]["headline_mean"] += 1e-6
            json_path.write_bytes(contemporary._canonical_json_bytes(payload))
            with self.assertRaisesRegex(
                contemporary.ContemporaryReferenceError,
                "record matrix differs from the fixed public digest",
            ):
                contemporary.verify_outputs(json_path, csv_path, tex_path)

    def test_tampered_json_record_sd_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            json_path, csv_path, tex_path = self._copy_outputs(Path(raw))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["records"][18]["headline_population_sd"] += 1e-6
            json_path.write_bytes(contemporary._canonical_json_bytes(payload))
            with self.assertRaisesRegex(
                contemporary.ContemporaryReferenceError,
                "record matrix differs from the fixed public digest",
            ):
                contemporary.verify_outputs(json_path, csv_path, tex_path)

    def test_tampered_aggregate_sd_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            json_path, csv_path, tex_path = self._copy_outputs(Path(raw))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["aggregates"][3][
                "headline_population_sd_across_seed_macros"
            ] += 1e-6
            json_path.write_bytes(contemporary._canonical_json_bytes(payload))
            with self.assertRaisesRegex(
                contemporary.ContemporaryReferenceError,
                "aggregate matrix differs from the fixed public digest",
            ):
                contemporary.verify_outputs(json_path, csv_path, tex_path)

    def test_tampered_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            json_path, csv_path, tex_path = self._copy_outputs(Path(raw))
            csv_path.write_bytes(csv_path.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(
                contemporary.ContemporaryReferenceError, "public CSV"
            ):
                contemporary.verify_outputs(json_path, csv_path, tex_path)

    def test_tampered_tex_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            json_path, csv_path, tex_path = self._copy_outputs(Path(raw))
            tex_path.write_text(
                tex_path.read_text(encoding="utf-8").replace("0.6042", "0.6043", 1),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                contemporary.ContemporaryReferenceError, "public TeX"
            ):
                contemporary.verify_outputs(json_path, csv_path, tex_path)

    def test_tampered_config_evidence_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            config_path = Path(raw) / "config.json"
            payload = json.loads(contemporary.DEFAULT_CONFIG.read_text(encoding="utf-8"))
            payload["formal_evidence"]["final_reporting_receipt"]["sha256"] = "f" * 64
            config_path.write_bytes(contemporary._canonical_json_bytes(payload))
            with self.assertRaisesRegex(
                contemporary.ContemporaryReferenceError,
                "final_reporting_receipt digest changed",
            ):
                contemporary._load_config(config_path)


@unittest.skipUnless(
    os.environ.get("UPGRADE_BENCH_PRIVATE_PROVENANCE_TESTS") == "1",
    "requires the ignored formal contemporary-reference extraction",
)
class ContemporaryFullProvenanceTest(unittest.TestCase):
    def test_full_profile_recomputes_receipt_and_repromotes_exact_public_bytes(self):
        summary = contemporary.verify_outputs(profile="full")
        self.assertEqual(summary["record_count"], 90)
        self.assertEqual(summary["aggregate_count"], 15)
        self.assertEqual(summary["benchmark"]["seeds"], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
