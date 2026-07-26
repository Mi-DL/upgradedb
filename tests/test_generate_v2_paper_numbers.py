import copy
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_v2_paper_numbers as generator  # noqa: E402
import public_release_policy as public_policy  # noqa: E402
import summarize_v2_loco_results as loco_public  # noqa: E402
from generate_v2_paper_numbers import (  # noqa: E402
    DEFAULT_PATHS,
    GPU_PROTOCOL,
    GPU_SUMMARY_SCHEMA,
    PAPER_NUMBERS_SCHEMA,
    PaperNumberValidationError,
    _add_gbdt_numbers,
    _assert_claimable_sources,
    _billions,
    _commas,
    _decimal,
    _percent,
    _validate_gpu_summary,
    _validate_gbdt_summary,
    collect_numbers,
    render_json,
    render_tex,
    verify_outputs,
    write_outputs,
)


CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
TRACKS = ("a", "b1", "b2")
FAMILIES = ("kge", "nbfnet")
FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"
PRIVATE_PROVENANCE_TESTS_ENABLED = (
    os.environ.get("UPGRADE_BENCH_PRIVATE_PROVENANCE_TESTS") == "1"
)
requires_full_payload = unittest.skipUnless(
    FULL_PAYLOAD_TESTS_ENABLED,
    "full processed-data payload tests run only in the explicit full-profile stage",
)
requires_private_provenance = unittest.skipUnless(
    PRIVATE_PROVENANCE_TESTS_ENABLED,
    "paper-source/private-provenance checks run only in the internal staging checkout",
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_loco_public_summary() -> dict:
    records = []
    for chain_index, chain in enumerate(loco_public.CHAINS):
        for mode in loco_public.MODES:
            for seed in loco_public.SEEDS:
                mode_delta = 0.05 if mode == "in_domain" else 0.0
                record = {
                    "component_id": f"{chain}|{mode}|seed{seed}",
                    "chain": chain,
                    "mode": mode,
                    "seed": seed,
                }
                for metric_index, metric in enumerate(loco_public.METRICS):
                    record[metric] = (
                        0.10
                        + 0.10 * metric_index
                        + 0.01 * chain_index
                        + 0.001 * seed
                        + mode_delta
                    )
                records.append(record)
    metrics, _ = loco_public._derive_metrics(records)
    payload = {
        "schema_version": loco_public.PUBLIC_SUMMARY_SCHEMA,
        "protocol": loco_public.PROTOCOL,
        "status": "complete",
        "run_id": loco_public.RUN_ID,
        "claim_scope": loco_public.CLAIM_SCOPE,
        "paper_eligible": True,
        "paper_eligibility_scope": loco_public.PAPER_SCOPE,
        "promotion_attestation": dict(loco_public.PROMOTION_ATTESTATION),
        "source_artifacts": {
            "formal_summary": {
                "artifact_role": loco_public.SUMMARY_ROLE,
                "sha256": digest("synthetic formal summary"),
            },
            "verification_receipt": {
                "artifact_role": loco_public.RECEIPT_ROLE,
                "sha256": digest("synthetic verification receipt"),
            },
        },
        "provenance": {
            "freeze_manifest_artifact_role": loco_public.FREEZE_ROLE,
            "freeze_manifest_file_sha256": digest("synthetic manifest file"),
            "freeze_sha256": digest("synthetic freeze"),
            "config_sha256": file_digest(generator.DEFAULT_PATHS.loco_config),
            "main_start_marker_artifact_role": loco_public.MARKER_ROLE,
            "main_start_marker_sha256": digest("synthetic main marker"),
        },
        "design": {
            "chains": list(loco_public.CHAINS),
            "modes": list(loco_public.MODES),
            "seeds": list(loco_public.SEEDS),
            "expected_component_count": loco_public.EXPECTED_COMPONENT_COUNT,
            "verified_component_count": loco_public.EXPECTED_COMPONENT_COUNT,
            "comparison_definition": "in_domain_minus_loco",
        },
        "metric_records": records,
        "metrics": metrics,
        "limitations": list(loco_public.PUBLIC_LIMITATIONS),
    }
    return loco_public.validate_public_summary(payload)


def synthetic_gpu_summary() -> dict:
    records = []
    for chain_index, chain in enumerate(CHAINS):
        for track_index, track in enumerate(TRACKS):
            for family_index, family in enumerate(FAMILIES):
                base = 0.1 + chain_index * 0.01 + track_index * 0.03 + family_index * 0.1
                values = [base + seed * 0.001 for seed in range(5)]
                stem = "candidates" if track == "a" else "candidates_firsttime"
                key = f"{chain}|{track}|{family}"
                records.append(
                    {
                        "chain": chain,
                        "track": track,
                        "family": family,
                        "primary_metric": {
                            "a": "lane_average_precision",
                            "b1": "entry_average_precision",
                            "b2": "conditional_recall_at_3",
                        }[track],
                        "primary_values_by_seed": values,
                        "primary_mean": statistics.fmean(values),
                        "primary_std_across_seeds": statistics.pstdev(values),
                        "history_candidate_role": f"data/processed_v2/{stem}_{chain}_fold2.csv",
                        "history_candidate_sha256": digest(f"history|{chain}|{stem}"),
                        "target_candidate_role": f"data/processed_v2/{stem}_{chain}.csv",
                        "target_candidate_sha256": digest(f"target|{chain}|{stem}"),
                        "selection_sha256": digest(f"selection|{key}"),
                        "metric_artifact_sha256": digest(f"metric|{key}"),
                        "score_artifact_sha256": digest(f"score|{key}"),
                    }
                )

    macros = []
    for track in TRACKS:
        for family in FAMILIES:
            rows = [row for row in records if row["track"] == track and row["family"] == family]
            chain_means = [row["primary_mean"] for row in rows]
            macros.append(
                {
                    "track": track,
                    "family": family,
                    "primary_metric": rows[0]["primary_metric"],
                    "mean_across_six_chain_means": statistics.fmean(chain_means),
                    "std_across_six_chain_means": statistics.pstdev(chain_means),
                    "n_chains": 6,
                    "per_seed_macro": [
                        {
                            "seed": seed,
                            "mean_across_six_chains": statistics.fmean(
                                row["primary_values_by_seed"][seed] for row in rows
                            ),
                        }
                        for seed in range(5)
                    ],
                }
            )
    return {
        "schema_version": GPU_SUMMARY_SCHEMA,
        "protocol": GPU_PROTOCOL,
        "status": "complete",
        "run_id": "synthetic-test-only",
        "target_fold": "main",
        "aggregation": "calendar_mean",
        "manifest_artifact_role": "results_v2/gpu_rolling/frozen_manifest.json",
        "manifest_sha256": digest("manifest"),
        "run_config_artifact_role": "configs/v2_gpu_rolling.json",
        "run_config_sha256": digest("config"),
        "seeds": [0, 1, 2, 3, 4],
        "complete_chain_family_jobs": 12,
        "complete_task_evaluations": 36,
        "reporting_policy": {
            "all_frozen_families_reported_separately": True,
            "all_six_chains_in_every_macro": True,
            "main_test_champion_selected": False,
            "macro_weighting": "unweighted arithmetic mean of six preregistered chain means",
            "raw_score_cross_seed_averaging": False,
            "bootstrap_rng_seed_mechanically_verified": True,
        },
        "records": records,
        "macro_summary": macros,
    }


def synthetic_verified_gbdt_summary() -> dict:
    track_specs = generator.GBDT_TRACK_SPECS
    public_sources = {
        "tools/v2_gbdt_baselines.py": file_digest(
            ROOT / "tools/v2_gbdt_baselines.py"
        ),
        generator.GBDT_SHARED_SOURCE_ROLE: file_digest(
            ROOT / generator.GBDT_SHARED_SOURCE_ROLE
        ),
        "configs/v2_gbdt_baselines.json": file_digest(
            ROOT / "configs/v2_gbdt_baselines.json"
        ),
    }
    chains: dict[str, dict] = {}
    per_track_headline: dict[str, dict[str, float]] = {
        track: {} for track in track_specs
    }
    per_track_value: dict[str, dict[str, float]] = {
        track: {} for track in track_specs
    }
    for chain_index, chain in enumerate(CHAINS):
        chain_payload: dict[str, dict] = {}
        for track_index, track in enumerate(track_specs):
            point = 0.10 + 0.02 * track_index + 0.01 * chain_index
            value = 0.20 + 0.02 * track_index + 0.01 * chain_index
            if track == "track_b2_conditional_destination_ranking":
                metrics = {
                    "at_k": {
                        "k_3": {
                            "macro_recall": point,
                            "macro_recall_ci95": [point - 0.01, point + 0.01],
                        }
                    }
                }
            else:
                metrics = {
                    "average_precision": point,
                    "average_precision_ci95": [point - 0.01, point + 0.01],
                }
            chain_payload[track] = {
                "models": {
                    generator.GBDT_MODEL_KEY: {
                        "metrics": metrics,
                    }
                }
            }
            per_track_headline[track][chain] = point
            per_track_value[track][chain] = value
        chains[chain] = chain_payload

    macro_summary = {}
    for track, spec in track_specs.items():
        headline_values = list(per_track_headline[track].values())
        value_values = list(per_track_value[track].values())
        macro_summary[track] = {
            "headline_metric": spec["headline_metric"],
            "realized_value_metric": spec["value_metric"],
            "aggregation": "unweighted_mean_over_chains",
            "chain_registry": list(CHAINS),
            "model": generator.GBDT_MODEL_KEY,
            "headline": {
                "per_chain": per_track_headline[track],
                "macro_mean": statistics.fmean(headline_values),
                "std_across_chains": statistics.pstdev(headline_values),
            },
            "realized_value": {
                "per_chain": per_track_value[track],
                "macro_mean": statistics.fmean(value_values),
                "std_across_chains": statistics.pstdev(value_values),
            },
        }
    return {
        "schema_version": generator.GBDT_PUBLIC_SUMMARY_SCHEMA,
        "benchmark_version": generator.BENCHMARK_VERSION,
        "status": generator.GBDT_PUBLIC_STATUS,
        "config": {
            "path": "configs/v2_gbdt_baselines.json",
            "sha256": public_sources["configs/v2_gbdt_baselines.json"],
        },
        "inputs": {
            "candidate_files": [{} for _ in range(24)],
            "public_sources": public_sources,
        },
        "protocol": {"bootstrap": {"draws": 200}},
        "runtime": {
            "python": "3.12.13",
            "platform": "synthetic-test-platform",
            "cpu_model": "Synthetic CPU Model 1",
            "logical_cpu_cores": 16,
            "numpy": "2.3.5",
            "pandas": "3.0.1",
            "scikit_learn": "1.9.0",
            "wall_elapsed_seconds": 98.8206877,
            "fit_count_upper_bound": 378,
        },
        "chains": chains,
        "macro_summary": macro_summary,
    }


class PaperNumberFormatTests(unittest.TestCase):
    def test_tex_safe_integer(self):
        self.assertEqual(_commas(385730), "385{,}730")

    def test_metrics_have_fixed_precision(self):
        self.assertEqual(_decimal(0.1712319), "0.1712")
        self.assertEqual(_percent(0.0408731), "4.09\\%")
        self.assertEqual(_billions(19_799_099.2384), "19.80")

    def test_cpu_model_is_the_only_strict_text_valued_macro(self):
        numbers = {"VTwoGBDTCPUModel": "AMD64 Family 26, AuthenticAMD"}
        with mock.patch.object(
            public_policy, "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT", 1
        ), mock.patch.object(
            public_policy,
            "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256",
            public_policy._paper_number_key_digest(numbers),
        ):
            self.assertEqual(public_policy._validate_paper_numbers(numbers), numbers)
            with self.assertRaisesRegex(ValueError, "malformed value"):
                public_policy._validate_paper_numbers(
                    {"VTwoUnreviewedText": "AMD64 Family 26, AuthenticAMD"}
                )


class SyntheticGBDTInterfaceTests(unittest.TestCase):
    def test_verified_reduction_exports_exact_gbdt_macro_shape(self):
        reduced = _validate_gbdt_summary(
            DEFAULT_PATHS,
            synthetic_verified_gbdt_summary(),
            {},
        )
        numbers: dict[str, str] = {}
        _add_gbdt_numbers(numbers, reduced)
        self.assertEqual(len(numbers), 69)
        self.assertEqual(numbers["VTwoGBDTStatus"], "COMPLETE")
        self.assertEqual(numbers["VTwoGBDTChainCount"], "6")
        self.assertEqual(numbers["VTwoGBDTTaskCount"], "3")
        self.assertEqual(numbers["VTwoGBDTHeadlineRecordCount"], "18")
        self.assertEqual(numbers["VTwoGBDTCPUModel"], "Synthetic CPU Model 1")
        self.assertEqual(numbers["VTwoGBDTWallSeconds"], "98.8207")
        self.assertEqual(numbers["VTwoGBDTLogicalCores"], "16")
        self.assertEqual(numbers["VTwoGBDTFitCountUpperBound"], "378")
        self.assertEqual(numbers["VTwoGBDTBootstrapDraws"], "200")
        self.assertEqual(numbers["VTwoSheepGBDTTrackAAP"], "0.1000")
        self.assertEqual(numbers["VTwoSheepGBDTTrackAAPCILower"], "0.0900")
        self.assertEqual(numbers["VTwoSheepGBDTTrackAAPCIUpper"], "0.1100")
        self.assertEqual(numbers["VTwoGBDTTrackAAPSixChainMean"], "0.1250")
        self.assertEqual(
            numbers["VTwoGBDTTrackAValueCaptureFiveHundredSixChainMean"],
            "0.2250",
        )

    def test_pending_reduction_exports_status_only(self):
        numbers: dict[str, str] = {}
        _add_gbdt_numbers(numbers, None)
        self.assertEqual(numbers, {"VTwoGBDTStatus": "PENDING"})

    def test_status_config_and_public_source_hashes_fail_closed(self):
        mutations = (
            (
                lambda payload: payload.__setitem__("status", "pending"),
                "status",
            ),
            (
                lambda payload: payload["config"].__setitem__("path", "configs/other.json"),
                "config path",
            ),
            (
                lambda payload: payload["inputs"]["public_sources"].__setitem__(
                    "tools/v2_gbdt_baselines.py", digest("stale runner")
                ),
                "stale hash",
            ),
            (
                lambda payload: payload["runtime"].__setitem__(
                    "cpu_model", "unsafe & TeX"
                ),
                "TeX-safe",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                payload = synthetic_verified_gbdt_summary()
                mutate(payload)
                with self.assertRaisesRegex(PaperNumberValidationError, message):
                    _validate_gbdt_summary(DEFAULT_PATHS, payload, {})


class SchemaEightDiagnosticReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.product_payload = generator._load(generator.PRODUCT_SPACE_SUMMARY)
        cls.r5_payload = generator._load(generator.SCORE_ROBUSTNESS_R5)
        cls.geometry_payload = generator._load(
            generator.ELIGIBILITY_THRESHOLD_GEOMETRY
        )
        cls.product = generator._validate_product_space_summary(
            DEFAULT_PATHS, cls.product_payload, {}
        )
        cls.r5 = generator._validate_score_robustness_r5(
            DEFAULT_PATHS, cls.r5_payload, {}
        )
        cls.geometry = generator._validate_eligibility_threshold_geometry(
            DEFAULT_PATHS, cls.geometry_payload, {}
        )
        cls.numbers: dict[str, str] = {}
        generator._add_product_space_numbers(cls.numbers, cls.product)
        generator._add_score_robustness_r5_numbers(cls.numbers, cls.r5)
        generator._add_eligibility_threshold_geometry_numbers(
            cls.numbers, cls.geometry
        )

    def test_product_space_main_history_chain_ci_and_universe_macros(self):
        expected = {
            "VTwoProductSpaceBOneAPSixChainMean": "0.3862",
            "VTwoProductSpaceBOneValueCaptureFiftySixChainMean": "0.4578",
            "VTwoProductSpaceHistoricalBOneAPSixChainMean": "0.4073",
            "VTwoProductSpaceHistoricalBOneValueCaptureFiftySixChainMean": "0.4162",
            "VTwoSheepProductSpaceMainBOneAP": "0.2425",
            "VTwoSheepProductSpaceMainBOneAPCILower": "0.1088",
            "VTwoSheepProductSpaceMainBOneAPCIUpper": "0.4622",
            "VTwoSheepProductSpaceMainBOneValueCaptureFifty": "0.6613",
            "VTwoSheepProductSpaceHistoricalBOneAP": "0.3407",
            "VTwoSheepProductSpaceHistoricalBOneValueCaptureFifty": "0.5605",
            "VTwoProductSpaceCountryUniverse": "235",
            "VTwoProductSpaceProductUniverse": "5{,}022",
            "VTwoProductSpaceTargetProductUniverse": "262",
            "VTwoProductSpaceMainCandidatesWithTargetMembership": "58",
            "VTwoProductSpaceTargetDiagonalNonzeroBeforeExclusion": "262",
            "VTwoProductSpaceTargetDiagonalMaxAfterExclusion": "0.0000",
        }
        for macro, value in expected.items():
            self.assertEqual(self.numbers[macro], value)

    def test_paired_pooling_budget_and_end_to_end_macros(self):
        expected = {
            "VTwoPairedTrackADelta": "0.0296",
            "VTwoPairedTrackACILower": "0.0196",
            "VTwoPairedTrackACIUpper": "0.0391",
            "VTwoPairedTrackBOneDelta": "0.1123",
            "VTwoPairedTrackBTwoDelta": "-0.1611",
            "VTwoPoolingKGERawMaxBOneAP": "0.3952",
            "VTwoPoolingKGERawMaxBOneValueCaptureFifty": "0.5209",
            "VTwoPoolingKGEECDFMeanBOneAP": "0.3490",
            "VTwoPoolingKGEECDFMeanBOneValueCaptureFifty": "0.4164",
            "VTwoPoolingKGEECDFTopThreeBOneAP": "0.4073",
            "VTwoPoolingKGEECDFTopThreeBOneValueCaptureFifty": "0.5068",
            "VTwoPoolingNBFNetRawMaxBOneAP": "0.2829",
            "VTwoPoolingNBFNetRawMaxBOneValueCaptureFifty": "0.2696",
            "VTwoPoolingNBFNetECDFMeanBOneValueCaptureFifty": "0.3176",
            "VTwoPoolingNBFNetECDFTopThreeBOneValueCaptureFifty": "0.4345",
            "VTwoBudgetTrackAKGERecallHeadline": "0.0923",
            "VTwoBudgetTrackANBFNetValueHeadline": "0.1027",
            "VTwoBudgetTrackBOneKGEValueHeadline": "0.5209",
            "VTwoBudgetTrackBTwoNBFNetRecallHeadline": "0.3757",
            "VTwoEndToEndKGEGateRecall": "0.4641",
            "VTwoEndToEndKGEDestinationRecall": "0.0948",
            "VTwoEndToEndKGEValueCapture": "0.0773",
            "VTwoEndToEndNBFNetValueCapture": "0.1451",
        }
        for macro, value in expected.items():
            self.assertEqual(self.numbers[macro], value)

    def test_threshold_counts_and_reference_retention_macros(self):
        expected = {
            "VTwoEligibilityThresholdExactHundredGatePass": "TRUE",
            "VTwoEligibilityThresholdFiftyTrackACandidates": "390{,}804",
            "VTwoEligibilityThresholdFiftyTrackAPositives": "15{,}218",
            "VTwoEligibilityThresholdFiftyTrackACandidateRetention": "0.9706",
            "VTwoEligibilityThresholdFiftyTrackAPositiveRetention": "0.7519",
            "VTwoEligibilityThresholdHundredTrackBOneCandidates": "1{,}518",
            "VTwoEligibilityThresholdHundredTrackBOnePositives": "270",
            "VTwoEligibilityThresholdHundredTrackBOneCandidateRetention": "1.0000",
            "VTwoEligibilityThresholdTwoFiftyTrackBTwoCandidates": "24{,}508",
            "VTwoEligibilityThresholdTwoFiftyTrackBTwoPositives": "492",
            "VTwoEligibilityThresholdTwoFiftyTrackBTwoPositiveRetention": "0.4820",
        }
        for macro, value in expected.items():
            self.assertEqual(self.numbers[macro], value)

    def test_reducers_fail_closed_on_semantic_mutation(self):
        product = copy.deepcopy(self.product_payload)
        product["protocol"]["main_labels_used_for_selection_or_calibration"] = True
        with self.assertRaisesRegex(PaperNumberValidationError, "expected False"):
            generator._validate_product_space_summary(DEFAULT_PATHS, product, {})

        r5 = copy.deepcopy(self.r5_payload)
        r5["analysis"]["paired_family_comparison"]["fixed_six_chain"][0][
            "direction"
        ] = "nbfnet_minus_kge"
        with self.assertRaisesRegex(PaperNumberValidationError, "paired protocol"):
            generator._validate_score_robustness_r5(DEFAULT_PATHS, r5, {})

        geometry = copy.deepcopy(self.geometry_payload)
        geometry["canonical_100kusd_gate"]["status"] = "FAIL"
        with self.assertRaisesRegex(PaperNumberValidationError, "did not pass"):
            generator._validate_eligibility_threshold_geometry(
                DEFAULT_PATHS, geometry, {}
            )


class CurrentArtifactGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interface, _ = public_policy._strict_canonical_json_file(
            generator.DEFAULT_JSON,
            "committed paper-number interface",
        )
        cls.numbers = dict(cls.interface["numbers"])
        cls.sources = dict(cls.interface["sources"])

    def test_committed_current_contract_needs_no_external_payload_reads(self):
        self.assertEqual(PAPER_NUMBERS_SCHEMA, "upgrade-bench-v2-paper-numbers-8")
        self.assertEqual(
            self.interface["schema_version"],
            public_policy.V2_PAPER_CURRENT_NUMBERS_SCHEMA,
        )
        self.assertEqual(
            len(self.numbers), public_policy.V2_PAPER_CURRENT_NUMBER_KEY_COUNT
        )
        self.assertEqual(
            public_policy._paper_number_key_digest(self.numbers),
            public_policy.V2_PAPER_CURRENT_NUMBER_KEYS_SHA256,
        )
        self.assertEqual(
            public_policy._paper_number_value_digest(self.numbers),
            public_policy.V2_PAPER_CURRENT_NUMBER_VALUES_SHA256,
        )
        self.assertEqual(self.numbers["VTwoLOCOStatus"], "COMPLETE")
        self.assertEqual(self.numbers["VTwoULTRAStatus"], "COMPLETE")
        self.assertEqual(self.numbers["VTwoGBDTStatus"], "COMPLETE")
        self.assertEqual(self.numbers["VTwoProductSpaceStatus"], "COMPLETE")
        self.assertEqual(
            self.numbers["VTwoScoreRobustnessRFiveStatus"], "COMPLETE"
        )
        self.assertEqual(
            self.numbers["VTwoEligibilityThresholdStatus"], "COMPLETE"
        )
        self.assertEqual(self.interface["ultra_status"], "COMPLETE")
        self.assertEqual(self.interface["gbdt_status"], "COMPLETE")
        self.assertEqual(len(self.sources), 37)
        self.assertEqual(self.numbers["VTwoGPUStatus"], "COMPLETE")
        self.assertEqual(self.numbers["VTwoGPUSeedCount"], "5")
        self.assertEqual(self.numbers["VTwoGPUCompleteTaskEvaluations"], "36")
        self.assertTrue(any(key.startswith("VTwoGPUTrack") for key in self.numbers))
        self.assertIn("results_v2/metrics/v2_gpu_rolling_summary.json", self.sources)
        self.assertIn("results_v2/metrics/v2_value_diagnostics.json", self.sources)
        self.assertIn("results_v2/metrics/v2_value_diagnostics.csv", self.sources)
        self.assertIn("results_v2/metrics/v2_ultra_zero_shot_summary.json", self.sources)
        self.assertIn("results_v2/metrics/v2_ultra_zero_shot_summary.csv", self.sources)
        self.assertIn("tools/summarize_v2_ultra_results.py", self.sources)
        self.assertIn("configs/v2_ultra_formal.json", self.sources)
        self.assertIn("tools/v2_ultra_formal.py", self.sources)
        self.assertIn("results_v2/metrics/v2_gbdt_baselines.json", self.sources)
        self.assertIn("results_v2/metrics/v2_gbdt_baselines.csv", self.sources)
        self.assertIn("tools/v2_gbdt_baselines.py", self.sources)
        self.assertIn("configs/v2_gbdt_baselines.json", self.sources)
        self.assertIn("results_v2/metrics/v2_product_space_density.json", self.sources)
        self.assertIn("results_v2/metrics/v2_product_space_density.csv", self.sources)
        self.assertIn("results_v2/scores/v2_product_space_density_scores.csv", self.sources)
        self.assertIn("tools/v2_product_space_density.py", self.sources)
        self.assertIn("configs/v2_product_space_density.json", self.sources)
        self.assertIn("results_v2/metrics/v2_score_robustness_r5.json", self.sources)
        self.assertIn("results_v2/metrics/v2_score_robustness_r5.csv", self.sources)
        self.assertIn("tools/v2_score_robustness_r5.py", self.sources)
        self.assertIn("configs/v2_score_robustness_r5.json", self.sources)
        self.assertIn(
            "results_v2/metrics/v2_eligibility_threshold_geometry.json",
            self.sources,
        )
        self.assertIn(
            "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
            self.sources,
        )
        self.assertIn("tools/v2_eligibility_threshold_geometry.py", self.sources)
        self.assertIn("configs/v2_eligibility_threshold_geometry.json", self.sources)
        self.assertEqual(
            self.sources["tools/v2_value_diagnostics.py"],
            hashlib.sha256((ROOT / "tools/v2_value_diagnostics.py").read_bytes()).hexdigest(),
        )

    def test_repository_contract_loader_never_calls_full_collection(self):
        with mock.patch.object(
            generator,
            "collect_numbers",
            side_effect=AssertionError("repository contract must not collect full inputs"),
        ):
            interface, _ = public_policy._strict_canonical_json_file(
                generator.DEFAULT_JSON,
                "committed paper-number interface",
            )
        self.assertEqual(interface["numbers"], self.numbers)

    @requires_full_payload
    def test_full_payload_recomputation_matches_committed_current_contract(self):
        numbers, sources = collect_numbers(
            DEFAULT_PATHS,
            require_gpu=True,
            require_loco=True,
            require_ultra=True,
            require_gbdt=True,
        )
        self.assertEqual(numbers["VTwoGBDTStatus"], "COMPLETE")
        self.assertEqual(set(sources), set(public_policy.V2_PAPER_SOURCE_PATHS))
        if self.interface["schema_version"] == PAPER_NUMBERS_SCHEMA:
            self.assertEqual(numbers, self.numbers)
            self.assertEqual(sources, self.sources)

    def test_ultra_reference_interface_is_complete_and_descriptive(self):
        expected = {
            "VTwoULTRAChainCount": "6",
            "VTwoULTRATaskCount": "3",
            "VTwoULTRAHeadlineRecordCount": "18",
            "VTwoULTRACheckpointCount": "1",
            "VTwoULTRATargetLabelsUsedForTrainingSelection": "FALSE",
            "VTwoULTRATargetEarlyGraphUsed": "TRUE",
            "VTwoULTRACheckpointTrainingSeedDisclosed": "FALSE",
            "VTwoULTRARepeatabilityScoreGatePass": "TRUE",
            "VTwoULTRARepeatabilityMetricGatePass": "TRUE",
            "VTwoULTRAAbstractMentionEligible": "TRUE",
            "VTwoULTRATrackALaneAPSixChainMean": "0.0593",
            "VTwoULTRATrackBOneEntryAPSixChainMean": "0.3511",
            "VTwoULTRATrackBTwoConditionalRecallThreeSixChainMean": "0.0495",
            "VTwoULTRATrackAValueCaptureFiveHundredSixChainMean": "0.0282",
            "VTwoULTRATrackBOneEntryValueCaptureFiftySixChainMean": "0.4965",
            "VTwoULTRATrackBTwoConditionalValueCaptureThreeSixChainMean": "0.0484",
        }
        for macro, value in expected.items():
            self.assertEqual(self.numbers[macro], value)
        self.assertFalse(any("ULTRA" in macro and "Std" in macro for macro in self.numbers))

    def test_neutral_value_diagnostic_interface_and_deduplicated_dollars(self):
        expected = {
            "VTwoValueDiagnosticTrackAKGESixChainMeanValueCaptureFiveHundred": "0.0769",
            "VTwoValueDiagnosticTrackANBFNetSixChainMeanValueCaptureFiveHundred": "0.1027",
            "VTwoValueDiagnosticTrackAOracleSixChainMeanValueCaptureFiveHundred": "0.8474",
            "VTwoValueDiagnosticTrackAOraclePooledValueCaptureFiveHundred": "0.8442",
            "VTwoValueDiagnosticTrackBOneKGESixChainMeanValueCaptureFifty": "0.5209",
            "VTwoValueDiagnosticTrackBOneNBFNetSixChainMeanValueCaptureFifty": "0.2696",
            "VTwoValueDiagnosticTrackBOneOracleSixChainMeanValueCaptureFifty": "0.9985",
            "VTwoValueDiagnosticTrackBOneOraclePooledValueCaptureFifty": "0.9964",
            "VTwoValueDiagnosticTrackBTwoKGESixChainMeanMacroValueCaptureThree": "0.2181",
            "VTwoValueDiagnosticTrackBTwoNBFNetSixChainMeanMacroValueCaptureThree": "0.3883",
            "VTwoValueDiagnosticTrackBTwoOracleSixChainMeanMacroValueCaptureThree": "0.9850",
            "VTwoValueDiagnosticTrackBTwoOraclePooledValueCaptureThree": "0.8743",
            "VTwoValueDiagnosticUniqueObservedValueB": "18.4856",
            "VTwoValueDiagnosticBTwoNestedObservedValueB": "1.6235",
            "VTwoValueDiagnosticBOneBTwoObservedValueDifferenceKUSD": "0.0000",
            "VTwoValueDiagnosticBTwoExcludedFromUniqueSum": "TRUE",
            "VTwoValueDiagnosticTargetLabelsUsedForSelection": "FALSE",
            "VTwoValueDiagnosticPostHocMainChampionSelected": "FALSE",
        }
        for macro, value in expected.items():
            self.assertEqual(self.numbers[macro], value)
        self.assertFalse(any("Winner" in macro for macro in self.numbers))

    @requires_private_provenance
    def test_all_paper_macros_resolve_after_formal_gpu_integration(self):
        paper_text = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "paper/abstract.tex",
                "paper/body.tex",
                "paper/appendix.tex",
            )
        )
        profile_tex = (ROOT / "paper/generated/v2_benchmark_profile.tex").read_text(
            encoding="utf-8"
        )
        profile_macros = set(
            re.findall(
                r"\\newcommand\{\\(VTwoProfile[A-Za-z0-9]+)\}",
                profile_tex,
            )
        )
        references = set(re.findall(r"\\(VTwo[A-Za-z0-9]+)", paper_text))
        self.assertEqual(references - set(self.numbers) - profile_macros, set())

    def test_registry_coverage_value_and_pairwise_interfaces(self):
        self.assertEqual(self.numbers["VTwoRegistryReviewedCodes"], "610")
        self.assertEqual(self.numbers["VTwoRegistryPurityCheckPassRate"], "100.00\\%")
        self.assertEqual(self.numbers["VTwoRegistryTargetStages"], "41")
        self.assertEqual(self.numbers["VTwoRegistrySemanticStageGates"], "35")
        expected_registry_counts = {
            "Sheep": ("7", "4", "3"),
            "Cotton": ("10", "7", "3"),
            "Aluminium": ("11", "9", "3"),
            "Nickel": ("11", "10", "4"),
            "Cocoa": ("9", "7", "4"),
            "OilseedSoy": ("5", "4", "2"),
        }
        for tag, (active_stages, target_stages, reassigned) in expected_registry_counts.items():
            self.assertEqual(
                self.numbers[f"VTwoRegistry{tag}ActiveStages"], active_stages
            )
            self.assertEqual(
                self.numbers[f"VTwoRegistry{tag}TargetStages"], target_stages
            )
            self.assertEqual(
                self.numbers[f"VTwoRegistry{tag}ReassignedCodes"], reassigned
            )
        self.assertEqual(self.numbers["VTwoBOneCoverageMainRealizedEntryCoverage"], "96.43\\%")
        self.assertEqual(self.numbers["VTwoBOneCoverageMainLateStartValueCoverage"], "99.68\\%")
        self.assertEqual(self.numbers["VTwoTrackASizeValueCaptureFiveHundred"], "0.2307")
        self.assertEqual(self.numbers["VTwoTrackASizeMinusGravityAPDeltaMean"], "0.0312")
        self.assertIn("VTwoSheepTrackASizeAP", self.numbers)
        self.assertIn("VTwoSheepTrackASizeMinusGravityAPDelta", self.numbers)

    def test_importer_unseen_groups_use_exact_slice_reconditioning(self):
        self.assertEqual(self.numbers["VTwoRobustBTwoImporterUnseenChains"], "5")
        self.assertEqual(self.numbers["VTwoRobustBTwoImporterUnseenPositiveEntryGroups"], "10")
        self.assertEqual(self.numbers["VTwoRobustBTwoImporterUnseenEntryGroupsBeforeConditioning"], "1{,}447")
        self.assertEqual(self.numbers["VTwoRobustBTwoImporterUnseenDroppedZeroPositiveEntryGroups"], "1{,}437")
        self.assertEqual(self.numbers["VTwoRobustBTwoImporterUnseenPositiveLanes"], "11")

    @requires_full_payload
    def test_formal_collection_refuses_missing_gpu(self):
        paths = replace(
            DEFAULT_PATHS,
            gpu_summary=ROOT / "results_v2" / "metrics" / "__missing_gpu_for_test__.json",
        )
        with self.assertRaisesRegex(RuntimeError, "GPU summary is PENDING"):
            collect_numbers(paths, require_gpu=True)

    def test_unresolved_invalidation_blocks_even_custom_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            invalidation = root / public_policy.PUBLIC_V2_INVALIDATION_NOTICE
            invalidation.parent.mkdir(parents=True)
            invalidation.write_text(
                json.dumps({"status": "INVALIDATED_REGISTRY_AUDIT"}) + "\n",
                encoding="utf-8",
            )
            paths = replace(DEFAULT_PATHS, root=root, invalidation=invalidation)
            with self.assertRaisesRegex(RuntimeError, "invalidation marker exists"):
                _assert_claimable_sources(paths)
            tex = Path(directory) / "numbers.tex"
            js = Path(directory) / "numbers.json"
            with self.assertRaisesRegex(RuntimeError, "invalidation marker exists"):
                write_outputs(tex, js, paths=paths)
            self.assertFalse(tex.exists())
            self.assertFalse(js.exists())

    def _tampered_paths(self, source: Path, mutate):
        payload = json.loads(source.read_text(encoding="utf-8"))
        mutate(payload)
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        with temporary:
            json.dump(payload, temporary)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        field = {
            DEFAULT_PATHS.rolling: "rolling",
            DEFAULT_PATHS.robustness: "robustness",
            DEFAULT_PATHS.raw_audit: "raw_audit",
            DEFAULT_PATHS.b1_coverage: "b1_coverage",
            DEFAULT_PATHS.registry_audit: "registry_audit",
            DEFAULT_PATHS.value_diagnostics: "value_diagnostics",
        }[source]
        return replace(DEFAULT_PATHS, **{field: Path(temporary.name)})

    @requires_full_payload
    def test_tampered_value_schema_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.value_diagnostics,
            lambda payload: payload.__setitem__("schema_version", "stale"),
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "stale schema"):
            collect_numbers(paths)

    @requires_full_payload
    def test_incomplete_value_source_inventory_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.value_diagnostics,
            lambda payload: payload["inputs"]["gpu_score_artifacts"].pop(),
        )
        with self.assertRaisesRegex(
            PaperNumberValidationError, "incomplete GPU score audit inventory"
        ):
            collect_numbers(paths)

    @requires_full_payload
    def test_tampered_value_generator_hash_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.value_diagnostics,
            lambda payload: payload["runtime"].__setitem__("script_sha256", "0" * 64),
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "generator hash is stale"):
            collect_numbers(paths)

    @requires_full_payload
    def test_post_hoc_value_champion_flag_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.value_diagnostics,
            lambda payload: payload["protocol"].__setitem__(
                "post_hoc_main_champion_selected", True
            ),
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "post-hoc main champion"):
            collect_numbers(paths)

    @requires_full_payload
    def test_tampered_value_summary_is_recomputed_and_rejected(self):
        def mutate(payload):
            payload["chains"]["sheep"]["tracks"]["a"]["gpu_families"]["kge"][
                "summary_across_seeds"
            ]["model_value_capture"]["mean"] += 0.01

        paths = self._tampered_paths(DEFAULT_PATHS.value_diagnostics, mutate)
        with self.assertRaisesRegex(
            PaperNumberValidationError, "model_value_capture/mean"
        ):
            collect_numbers(paths)

    @requires_full_payload
    def test_tampered_cpu_protocol_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.rolling,
            lambda payload: payload["protocol"].__setitem__(
                "target_labels_used_for_model_selection", True
            ),
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "protocol field"):
            collect_numbers(paths)

    @requires_full_payload
    def test_tampered_robustness_schema_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.robustness,
            lambda payload: payload.__setitem__("schema_version", "stale"),
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "robustness schema 2"):
            collect_numbers(paths)

    @requires_full_payload
    def test_tampered_b1_registry_hash_fails_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.b1_coverage,
            lambda payload: payload["registry"]["audit"].__setitem__(
                "sha256", "0" * 64
            ),
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "stale hash"):
            collect_numbers(paths)

    @requires_full_payload
    def test_tampered_registry_target_stages_fail_closed(self):
        paths = self._tampered_paths(
            DEFAULT_PATHS.registry_audit,
            lambda payload: payload["chains"]["sheep"][
                "capacity_from_stages"
            ].pop("exp_meat"),
        )
        with self.assertRaisesRegex(
            PaperNumberValidationError, "target stages disagree with the hashed registry"
        ):
            collect_numbers(paths)

    def test_registry_decisions_uniquely_cover_hashed_selected_metadata(self):
        audit = json.loads(DEFAULT_PATHS.registry_audit.read_text(encoding="utf-8"))
        evidence = json.loads(
            DEFAULT_PATHS.registry_evidence.read_text(encoding="utf-8")
        )
        code_frequency = {}
        for chain_payload in audit["chains"].values():
            for row in chain_payload["decisions"]:
                code_frequency[row["code"]] = code_frequency.get(row["code"], 0) + 1
        excluded = next(
            row
            for row in audit["chains"]["sheep"]["decisions"]
            if row["decision"] == "exclude" and code_frequency[row["code"]] == 1
        )
        excluded_code = excluded["code"]
        for payload in (audit, evidence):
            payload["chains"]["sheep"]["decisions"] = [
                row
                for row in payload["chains"]["sheep"]["decisions"]
                if row["code"] != excluded_code
            ]
            payload["summary"]["excluded_codes"] -= 1
            payload["summary"]["reviewed_codes"] -= 1
            payload["summary"]["decision_records"] -= 1
            payload["summary"]["unique_reviewed_hs6"] -= 1
            source_count = (
                "observable_candidate_records"
                if excluded["candidate_source"] == "observable_regex"
                else "legacy_only_records"
            )
            payload["summary"][source_count] -= 1
        audit["chains"]["sheep"]["removed_codes"] -= 1
        audit["chains"]["sheep"]["before_review_codes"] -= 1
        evidence["chains"]["sheep"]["excluded_count"] -= 1
        if excluded["candidate_source"] == "observable_regex":
            evidence["chains"]["sheep"]["observable_candidate_count"] -= 1
        else:
            evidence["chains"]["sheep"]["legacy_only_count"] -= 1

        with self.assertRaisesRegex(
            PaperNumberValidationError,
            "union of decision HS6 codes does not exactly cover selected metadata",
        ):
            generator._validate_registry(DEFAULT_PATHS, audit, evidence, {})

    def test_current_registry_schema3_three_way_ledger_validates(self):
        audit = json.loads(DEFAULT_PATHS.registry_audit.read_text(encoding="utf-8"))
        evidence = json.loads(
            DEFAULT_PATHS.registry_evidence.read_text(encoding="utf-8")
        )
        validated = generator._validate_registry(DEFAULT_PATHS, audit, evidence, {})
        self.assertEqual(validated["summary"], audit["summary"])
        self.assertEqual(
            sum(row["out_of_stage"] for row in validated["per_chain"].values()),
            audit["summary"]["out_of_stage_codes"],
        )

    def test_registry_duplicate_decision_fails_closed(self):
        audit = json.loads(DEFAULT_PATHS.registry_audit.read_text(encoding="utf-8"))
        evidence = json.loads(
            DEFAULT_PATHS.registry_evidence.read_text(encoding="utf-8")
        )
        for payload in (audit, evidence):
            payload["chains"]["sheep"]["decisions"].append(
                copy.deepcopy(payload["chains"]["sheep"]["decisions"][0])
            )
        with self.assertRaisesRegex(PaperNumberValidationError, "duplicate decision"):
            generator._validate_registry(DEFAULT_PATHS, audit, evidence, {})

    def test_registry_reassignment_count_is_recomputed_from_decisions(self):
        audit = json.loads(DEFAULT_PATHS.registry_audit.read_text(encoding="utf-8"))
        evidence = json.loads(
            DEFAULT_PATHS.registry_evidence.read_text(encoding="utf-8")
        )
        audit["chains"]["sheep"]["reassigned_codes"].pop()
        with self.assertRaisesRegex(
            PaperNumberValidationError,
            "reassigned-code record disagrees with decisions",
        ):
            generator._validate_registry(DEFAULT_PATHS, audit, evidence, {})

    @requires_full_payload
    def test_synchronized_registry_count_tamper_fails_complete_collection(self):
        """Cross-file count/hash edits cannot override the hashed chain registries."""

        with tempfile.TemporaryDirectory(
            dir=ROOT, prefix=".paper-number-registry-tamper-"
        ) as directory:
            isolated = Path(directory)
            for relative in (
                "configs",
                "docs",
                "chains",
                "requirements",
                "src",
                "tools",
                "results_v2/metrics",
                "results_v2/scores",
            ):
                shutil.copytree(ROOT / relative, isolated / relative)
            shutil.copytree(
                ROOT / "data/processed_v2",
                isolated / "data/processed_v2",
                copy_function=os.link,
            )
            shutil.copytree(
                ROOT / "results_v2/gpu_rolling",
                isolated / "results_v2/gpu_rolling",
                copy_function=os.link,
            )
            paths = generator.ArtifactPaths.under(isolated)

            audit = json.loads(paths.registry_audit.read_text(encoding="utf-8"))
            evidence = json.loads(paths.registry_evidence.read_text(encoding="utf-8"))
            for payload in (audit, evidence):
                payload["summary"]["included_codes"] += 1
                payload["summary"]["excluded_codes"] -= 1
            audit["chains"]["sheep"]["active_codes"] += 1
            audit["chains"]["sheep"]["removed_codes"] -= 1
            evidence["chains"]["sheep"]["included_count"] += 1
            evidence["chains"]["sheep"]["excluded_count"] -= 1
            paths.registry_audit.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths.registry_evidence.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            coverage = json.loads(paths.b1_coverage.read_text(encoding="utf-8"))
            coverage["registry"]["audit"]["sha256"] = hashlib.sha256(
                paths.registry_audit.read_bytes()
            ).hexdigest()
            coverage["registry"]["evidence"]["sha256"] = hashlib.sha256(
                paths.registry_evidence.read_bytes()
            ).hexdigest()
            coverage["registry"]["active_hs6_count"] += 1
            paths.b1_coverage.write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                PaperNumberValidationError,
                "self-reported active-code count disagrees with the hashed registry",
            ):
                collect_numbers(paths)


class SyntheticLOCOInterfaceTests(unittest.TestCase):
    """Exercise the paper interface without creating canonical result files."""

    def setUp(self):
        # collect_numbers records repository-relative source roles. Keep this
        # disposable fixture below whichever checkout is under test (including
        # a clean public export), then remove it in tearDown.
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".paper-loco-synthetic-",
            dir=ROOT,
        )
        self.directory = Path(self.temporary.name).resolve()
        self.json_path = self.directory / "v2_loco_transfer_summary.json"
        self.csv_path = self.directory / "v2_loco_transfer_summary.csv"
        self.payload = synthetic_loco_public_summary()
        self.paths = replace(
            DEFAULT_PATHS,
            loco_summary=self.json_path,
            loco_summary_csv=self.csv_path,
        )
        self._write(self.payload)

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, payload: dict) -> None:
        self.json_path.write_bytes(loco_public.render_json(payload))
        self.csv_path.write_bytes(loco_public.render_csv(payload))

    def _verified_reduced(self):
        hashes = {}
        verified = generator._verify_loco_summary_first(self.paths, hashes)
        reduced = generator._validate_loco_summary(self.paths, verified, hashes)
        return verified, reduced, hashes

    def test_valid_public_summary_exports_exact_descriptive_macro_interface(self):
        verified, reduced, hashes = self._verified_reduced()
        numbers = {}
        generator._add_loco_numbers(numbers, reduced)

        expected_names = {
            "VTwoLOCOStatus",
            "VTwoLOCOChainCount",
            "VTwoLOCOSeedCount",
            "VTwoLOCOVerifiedComponentCount",
            "VTwoLOCOMatchedPairCountPerMetric",
        }
        for _, track_tex in generator.LOCO_METRIC_SPECS:
            prefix = f"VTwoLOCO{track_tex}"
            expected_names.update(
                {
                    f"{prefix}InDomainMean",
                    f"{prefix}InDomainPopulationStd",
                    f"{prefix}LOCOMean",
                    f"{prefix}LOCOPopulationStd",
                    f"{prefix}MatchedGapMean",
                    f"{prefix}MatchedGapPopulationStd",
                }
            )
        self.assertEqual(set(numbers), expected_names)
        self.assertEqual(numbers["VTwoLOCOStatus"], "COMPLETE")
        self.assertEqual(numbers["VTwoLOCOVerifiedComponentCount"], "60")
        self.assertEqual(numbers["VTwoLOCOMatchedPairCountPerMetric"], "30")

        for metric, track_tex in generator.LOCO_METRIC_SPECS:
            prefix = f"VTwoLOCO{track_tex}"
            row = verified["metrics"][metric]
            self.assertEqual(
                numbers[f"{prefix}InDomainMean"],
                _decimal(row["by_mode"]["in_domain"]["mean"]),
            )
            self.assertEqual(
                numbers[f"{prefix}LOCOMean"],
                _decimal(row["by_mode"]["loco"]["mean"]),
            )
            self.assertEqual(
                numbers[f"{prefix}MatchedGapMean"],
                _decimal(row["matched_gap"]["mean"]),
            )
        self.assertFalse(
            any(
                token in name
                for name in numbers
                for token in ("Confidence", "Interval", "StdError", "StandardError", "CI")
            )
        )

        expected_bound_paths = {
            self.json_path.resolve(),
            self.csv_path.resolve(),
            DEFAULT_PATHS.loco_summary_generator.resolve(),
            DEFAULT_PATHS.loco_config.resolve(),
        }
        self.assertEqual(set(hashes), expected_bound_paths)
        for path in expected_bound_paths:
            self.assertEqual(hashes[path], file_digest(path))

    @requires_full_payload
    def test_complete_synthetic_pair_flows_through_collect_and_binds_four_sources(self):
        numbers, sources = collect_numbers(self.paths, require_loco=True)
        self.assertEqual(numbers["VTwoLOCOStatus"], "COMPLETE")
        expected_roles = {
            self.json_path.relative_to(ROOT).as_posix(),
            self.csv_path.relative_to(ROOT).as_posix(),
            "tools/summarize_v2_loco_results.py",
            "configs/v2_loco_formal.json",
        }
        self.assertTrue(expected_roles <= set(sources))
        for role in expected_roles:
            self.assertEqual(sources[role], file_digest(ROOT / role))

    def test_canonical_write_requires_loco_and_writes_no_placeholder(self):
        missing_paths = replace(
            DEFAULT_PATHS,
            root=self.directory,
            loco_summary=self.directory / "missing.json",
            loco_summary_csv=self.directory / "missing.csv",
        )
        tex = self.directory / "paper/generated/v2_numbers.tex"
        js = self.directory / "results_v2/paper_numbers.json"
        current_value = json.loads(
            DEFAULT_PATHS.value_diagnostics.read_text(encoding="utf-8")
        )
        with mock.patch.object(
            generator, "_assert_claimable_sources", return_value=None
        ), mock.patch.object(
            generator,
            "_verify_value_diagnostics_first",
            return_value=current_value,
        ), self.assertRaisesRegex(RuntimeError, "LOCO summary is PENDING"):
            write_outputs(tex, js, paths=missing_paths)
        self.assertFalse(tex.exists())
        self.assertFalse(js.exists())

    @requires_full_payload
    def test_partial_json_csv_pair_fails_closed(self):
        paths = replace(self.paths, loco_summary_csv=self.directory / "missing.csv")
        with self.assertRaisesRegex(PaperNumberValidationError, "both exist or both be absent"):
            collect_numbers(paths)

    def test_aggregate_drift_is_recomputed_and_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["A.lane_average_precision"]["matched_gap"]["mean"] += 0.01
        self._write(payload)
        with self.assertRaisesRegex(PaperNumberValidationError, "mechanically recomputed"):
            generator._verify_loco_summary_first(self.paths, {})

    def test_csv_drift_is_rejected(self):
        self.csv_path.write_bytes(self.csv_path.read_bytes() + b"drift\n")
        with self.assertRaisesRegex(PaperNumberValidationError, "stale or non-deterministic"):
            generator._verify_loco_summary_first(self.paths, {})

    def test_malformed_public_provenance_hash_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["source_artifacts"]["formal_summary"]["sha256"] = "not-a-hash"
        self._write(payload)
        with self.assertRaisesRegex(PaperNumberValidationError, "SHA-256"):
            generator._verify_loco_summary_first(self.paths, {})

    def test_current_config_hash_is_bound_not_merely_well_formed(self):
        payload = copy.deepcopy(self.payload)
        payload["provenance"]["config_sha256"] = "0" * 64
        self._write(payload)
        hashes = {}
        verified = generator._verify_loco_summary_first(self.paths, hashes)
        with self.assertRaisesRegex(PaperNumberValidationError, "stale hash"):
            generator._validate_loco_summary(self.paths, verified, hashes)

    def test_non_finite_metric_is_rejected_by_public_verifier(self):
        payload = copy.deepcopy(self.payload)
        payload["metric_records"][0]["A.lane_average_precision"] = float("nan")
        self.json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PaperNumberValidationError, "non-finite"):
            generator._verify_loco_summary_first(self.paths, {})

    def test_duplicate_json_key_is_rejected(self):
        raw = self.json_path.read_bytes()
        needle = b'  "status": "complete"\n'
        self.assertIn(needle, raw)
        duplicate = b'  "status": "complete",\n  "status": "complete"\n'
        self.json_path.write_bytes(raw.replace(needle, duplicate, 1))
        with self.assertRaisesRegex(PaperNumberValidationError, "duplicate key"):
            generator._verify_loco_summary_first(self.paths, {})

    def test_noncanonical_generator_substitution_is_rejected(self):
        substitute = self.directory / "summarize_v2_loco_results.py"
        substitute.write_bytes(
            DEFAULT_PATHS.loco_summary_generator.read_bytes() + b"\n# drift\n"
        )
        paths = replace(self.paths, loco_summary_generator=substitute)
        with self.assertRaisesRegex(PaperNumberValidationError, "not the canonical"):
            generator._verify_loco_summary_first(paths, {})

    def test_schema8_json_renderer_requires_all_formal_statuses_and_is_canonical(self):
        numbers = {
            "VTwoGPUStatus": "COMPLETE",
            "VTwoLOCOStatus": "COMPLETE",
            "VTwoULTRAStatus": "COMPLETE",
            "VTwoGBDTStatus": "COMPLETE",
            "VTwoProductSpaceStatus": "COMPLETE",
            "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
            "VTwoEligibilityThresholdStatus": "COMPLETE",
        }
        rendered = render_json(numbers, {})
        payload = json.loads(rendered)
        self.assertEqual(payload["schema_version"], "upgrade-bench-v2-paper-numbers-8")
        self.assertEqual(payload["loco_status"], "COMPLETE")
        self.assertEqual(payload["ultra_status"], "COMPLETE")
        self.assertEqual(payload["gbdt_status"], "COMPLETE")
        with self.assertRaisesRegex(ValueError, "LOCO status pending"):
            render_json(
                {
                    "VTwoGPUStatus": "COMPLETE",
                    "VTwoLOCOStatus": "PENDING",
                    "VTwoULTRAStatus": "COMPLETE",
                    "VTwoGBDTStatus": "COMPLETE",
                    "VTwoProductSpaceStatus": "COMPLETE",
                    "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
                    "VTwoEligibilityThresholdStatus": "COMPLETE",
                },
                {},
            )
        with self.assertRaisesRegex(ValueError, "ULTRA status pending"):
            render_json(
                {
                    "VTwoGPUStatus": "COMPLETE",
                    "VTwoLOCOStatus": "COMPLETE",
                    "VTwoULTRAStatus": "PENDING",
                    "VTwoGBDTStatus": "COMPLETE",
                    "VTwoProductSpaceStatus": "COMPLETE",
                    "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
                    "VTwoEligibilityThresholdStatus": "COMPLETE",
                },
                {},
            )
        with self.assertRaisesRegex(ValueError, "GBDT status pending"):
            render_json(
                {
                    "VTwoGPUStatus": "COMPLETE",
                    "VTwoLOCOStatus": "COMPLETE",
                    "VTwoULTRAStatus": "COMPLETE",
                    "VTwoGBDTStatus": "PENDING",
                    "VTwoProductSpaceStatus": "COMPLETE",
                    "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
                    "VTwoEligibilityThresholdStatus": "COMPLETE",
                },
                {},
            )
        incomplete = dict(numbers)
        incomplete["VTwoProductSpaceStatus"] = "PENDING"
        with self.assertRaisesRegex(ValueError, "product-space status pending"):
            render_json(incomplete, {})
        incomplete = dict(numbers)
        incomplete["VTwoScoreRobustnessRFiveStatus"] = "PENDING"
        with self.assertRaisesRegex(ValueError, "r5 robustness status pending"):
            render_json(incomplete, {})
        incomplete = dict(numbers)
        incomplete["VTwoEligibilityThresholdStatus"] = "PENDING"
        with self.assertRaisesRegex(ValueError, "threshold-geometry status pending"):
            render_json(incomplete, {})

class ResolvedMarkerGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.paths = generator.ArtifactPaths.under(self.root)
        self.tex = self.root / "paper/generated/v2_numbers.tex"
        self.js = self.root / "results_v2/paper_numbers.json"
        self.numbers = {
            "VTwoGPUStatus": "COMPLETE",
            "VTwoLOCOStatus": "COMPLETE",
            "VTwoULTRAStatus": "COMPLETE",
            "VTwoGBDTStatus": "COMPLETE",
            "VTwoProductSpaceStatus": "COMPLETE",
            "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
            "VTwoEligibilityThresholdStatus": "COMPLETE",
            "VTwoFixture": "1.0000",
        }
        number_contract_patchers = (
            mock.patch.object(
                public_policy,
                "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT",
                len(self.numbers),
            ),
            mock.patch.object(
                public_policy,
                "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256",
                public_policy._paper_number_key_digest(self.numbers),
            ),
            mock.patch.object(
                public_policy,
                "V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256",
                public_policy._paper_number_value_digest(self.numbers),
            ),
        )
        for patcher in number_contract_patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        for relative in sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"replacement:{relative}\n".encode("utf-8"))
        bound_paths = set(public_policy.V2_PAPER_SOURCE_PATHS) | set(
            public_policy.V2_RESOLUTION_VERIFIER_SOURCE_PATHS
        )
        for relative in sorted(bound_paths):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(f"bound:{relative}\n".encode("utf-8"))
        self.sources = {
            relative: file_digest(self.root / relative)
            for relative in sorted(public_policy.V2_PAPER_SOURCE_PATHS)
        }
        self.tex.write_bytes(render_tex(self.numbers, self.sources).encode("utf-8"))
        self.js.write_bytes(render_json(self.numbers, self.sources).encode("utf-8"))
        replacements = {
            relative: hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
            for relative in sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS)
        }
        verifiers = {
            relative: file_digest(self.root / relative)
            for relative in public_policy.V2_RESOLUTION_VERIFIER_SOURCE_PATHS
        }
        review_paths = {
            "receipt": public_policy.REGISTRY_HUMAN_REVIEW_RECEIPT,
            "protocol": public_policy.REGISTRY_HUMAN_REVIEW_PROTOCOL,
            "freeze": public_policy.REGISTRY_HUMAN_REVIEW_FREEZE,
        }
        # This fixture exercises the resolved marker's exact byte binding only;
        # semantic human-review eligibility is covered by its dedicated tests.
        for role, relative in review_paths.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                public_policy._canonical_json_bytes(
                    {"fixture_role": role, "synthetic": True}
                )
            )
        self.notice = self.root / public_policy.PUBLIC_V2_INVALIDATION_NOTICE
        self.notice.parent.mkdir(parents=True, exist_ok=True)
        self.payload = {
            "schema_version": public_policy.V2_INVALIDATION_SCHEMA,
            "status": public_policy.V2_INVALIDATION_RESOLVED_STATUS,
            "original_status": public_policy.V2_INVALIDATION_ACTIVE_STATUS,
            "invalidated_at": public_policy.V2_INVALIDATION_DATE,
            "resolved_at": "2026-07-13T00:00:00Z",
            "scope": sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS),
            "reason": public_policy.V2_INVALIDATION_REASON,
            "claim_policy": "Do not cite until every replacement is verified.",
            "resolution": "Rebuild and verify the complete normative scope.",
            "replacement_sha256": replacements,
            "resolution_gate_sha256": verifiers["tools/resolve_v2_invalidation.py"],
            "resolution_source_sha256": dict(self.sources),
            "resolution_verifier_sha256": verifiers,
            public_policy.REGISTRY_HUMAN_REVIEW_BINDING_FIELD: {
                "audit_id": "FIXTURE-REVIEW-0001",
                "disposition": "NO_CONSTRUCT_CHANGE",
                **{
                    f"{role}_{field}": (
                        relative
                        if field == "path"
                        else file_digest(self.root / relative)
                    )
                    for role, relative in review_paths.items()
                    for field in ("path", "sha256")
                },
            },
        }
        self._write_notice()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_notice(self):
        self.notice.write_bytes(public_policy._canonical_json_bytes(self.payload))

    def test_valid_resolved_marker_permits_verify_and_byte_identical_write(self):
        before = {self.tex: self.tex.read_bytes(), self.js: self.js.read_bytes()}
        with mock.patch.object(
            generator,
            "collect_numbers",
            return_value=(dict(self.numbers), dict(self.sources)),
        ) as collect:
            verify_outputs(self.tex, self.js, paths=self.paths)
            write_outputs(self.tex, self.js, paths=self.paths)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(collect.call_count, 2)
        for call in collect.call_args_list:
            self.assertTrue(call.kwargs["require_gpu"])
            self.assertTrue(call.kwargs["require_loco"])
            self.assertTrue(call.kwargs["require_ultra"])
            self.assertTrue(call.kwargs["require_gbdt"])

    def test_extra_paper_json_field_is_not_accepted_as_canonical(self):
        payload = json.loads(self.js.read_text(encoding="utf-8"))
        payload["unreviewed_claim"] = "must not pass canonical verification"
        self.js.write_bytes(public_policy._canonical_json_bytes(payload))
        relative = "results_v2/paper_numbers.json"
        self.payload["replacement_sha256"][relative] = file_digest(self.js)
        self._write_notice()
        with mock.patch.object(generator, "collect_numbers") as collect, self.assertRaisesRegex(
            RuntimeError, "proof is invalid.*paper-number field inventory mismatch"
        ):
            verify_outputs(self.tex, self.js, paths=self.paths)
        collect.assert_not_called()

    def test_bad_resolved_proof_is_rejected_before_collection(self):
        first = sorted(self.payload["replacement_sha256"])[0]
        self.payload["replacement_sha256"][first] = "0" * 64
        self._write_notice()
        with mock.patch.object(generator, "collect_numbers") as collect, self.assertRaisesRegex(
            RuntimeError, "proof is invalid.*hash mismatch"
        ):
            write_outputs(self.tex, self.js, paths=self.paths)
        collect.assert_not_called()

    def test_replacement_source_drift_is_rejected_before_collection(self):
        source = self.root / "results_v2/metrics/rolling_cpu_baselines.csv"
        source.write_bytes(source.read_bytes() + b"drift\n")
        before = {self.tex: self.tex.read_bytes(), self.js: self.js.read_bytes()}
        with mock.patch.object(generator, "collect_numbers") as collect, self.assertRaisesRegex(
            RuntimeError, "proof is invalid.*hash mismatch"
        ):
            write_outputs(self.tex, self.js, paths=self.paths)
        collect.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in before}, before)


class GPUInterfaceTests(unittest.TestCase):
    def test_complete_six_by_three_by_two_summary_is_accepted(self):
        _validate_gpu_summary(synthetic_gpu_summary())

    def test_decision_weighted_gpu_macros_use_declared_task_units(self):
        summary_by_chain = {
            chain: {
                "track_a_candidates": chain_index + 1,
                "track_b_unique_entries": (chain_index + 1) * 2,
                "track_b_positive_entries": (chain_index + 1) * 3,
            }
            for chain_index, chain in enumerate(CHAINS)
        }
        gpu = synthetic_gpu_summary()
        records = {
            (row["chain"], row["track"], row["family"]): row["primary_mean"]
            for row in gpu["records"]
        }
        numbers: dict[str, str] = {}
        generator._add_gpu_numbers(numbers, gpu, summary_by_chain)
        expected = sum(
            records[(chain, "a", "kge")] * (chain_index + 1)
            for chain_index, chain in enumerate(CHAINS)
        ) / sum(range(1, len(CHAINS) + 1))
        self.assertEqual(
            numbers["VTwoDecisionWeightedAKGE"],
            _decimal(expected),
        )
        self.assertEqual(
            {
                key
                for key in numbers
                if key.startswith("VTwoDecisionWeighted")
            },
            {
                "VTwoDecisionWeightedAKGE",
                "VTwoDecisionWeightedANBFNet",
                "VTwoDecisionWeightedBOneKGE",
                "VTwoDecisionWeightedBOneNBFNet",
                "VTwoDecisionWeightedBTwoKGE",
                "VTwoDecisionWeightedBTwoNBFNet",
            },
        )

    def test_incomplete_gpu_summary_fails_closed(self):
        payload = synthetic_gpu_summary()
        payload["records"].pop()
        with self.assertRaisesRegex(PaperNumberValidationError, "36 chain-track-family"):
            _validate_gpu_summary(payload)

    def test_gpu_summary_hash_fields_are_required(self):
        payload = synthetic_gpu_summary()
        payload["records"][0]["score_artifact_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(PaperNumberValidationError, "SHA-256"):
            _validate_gpu_summary(payload)

    def test_gpu_summary_must_bind_current_postfreeze_attestation(self):
        import build_gpu_step3_postfreeze_attestation as postfreeze

        payload = {
            "run_id": postfreeze.RUN_ID,
            "post_freeze_semantic_attestation": postfreeze.summary_binding(),
        }
        generator._validate_gpu_postfreeze_binding(DEFAULT_PATHS, payload, {})
        payload["post_freeze_semantic_attestation"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PaperNumberValidationError, "binding is stale"):
            generator._validate_gpu_postfreeze_binding(DEFAULT_PATHS, payload, {})


if __name__ == "__main__":
    unittest.main()
