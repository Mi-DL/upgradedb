from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import artifact_bundles as bundles  # noqa: E402
import public_release_policy as public_policy  # noqa: E402
import public_release_audit  # noqa: E402
import release_clean_clone  # noqa: E402
import release_manifest  # noqa: E402
import release_smoke  # noqa: E402
import repository_size_gate as size_gate  # noqa: E402


FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"


def fixture_index(path: str, size: int, digest: str) -> dict[str, object]:
    files = [{"path": path, "bytes": size, "sha256": digest}]
    return {
        "schema_version": bundles.SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "archive_format": "zip",
        "archive_member_paths": "repository-relative",
        "github_release_asset_limit_bytes": bundles.GITHUB_RELEASE_ASSET_LIMIT,
        "planned_uncompressed_limit_bytes": bundles.MAX_PLANNED_UNCOMPRESSED_BYTES,
        "public_distribution_policy": public_policy.index_policy(),
        "bundles": [
            {
                "id": "fixture",
                "archive": "fixture.zip",
                "description": "test fixture",
                "file_count": 1,
                "uncompressed_bytes": size,
                "payload_sha256": bundles._payload_digest(files),
                "files": files,
            }
        ],
    }


class ArtifactBundleTests(unittest.TestCase):
    def _build_single_fixture(self, root: Path) -> tuple[Path, Path]:
        payload = root / "data" / "example.txt"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"stable payload\n")
        index_path = root / "release" / "DATA_ARTIFACT_INDEX.json"
        index_path.parent.mkdir(parents=True)
        index = fixture_index(
            "data/example.txt",
            payload.stat().st_size,
            bundles.sha256_file(payload),
        )
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output = root / "dist"
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value=None,
        ):
            bundles.build_archives(
                ["fixture"], output_dir=output, index_path=index_path, root=root
            )
        return output, index_path

    def test_clean_repository_recovers_current_external_selectors_from_frozen_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path = root / "release" / "DATA_ARTIFACT_INDEX.json"
            index_path.parent.mkdir(parents=True)
            # Keep this selector test self-contained.  The live checkout may
            # intentionally omit its stale final index while an invalidation
            # hold is active.
            index_path.write_text(
                json.dumps(
                    {
                        "bundles": [
                            {
                                "id": "v2-main",
                                "files": [
                                    {"path": "data/processed_v2/candidates_sheep.csv"}
                                ],
                            },
                            {
                                "id": "v2-history",
                                "files": [
                                    {"path": "data/processed_v2/candidates_sheep_fold2.csv"}
                                ],
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ):
                specs = {spec.bundle_id: spec for spec in bundles.bundle_specs(root)}
        self.assertFalse(any(bundle_id.startswith("v1-") for bundle_id in specs))
        self.assertTrue(
            any(path.startswith("data/processed_v2/") for path in specs["v2-main"].paths)
        )
        self.assertTrue(
            any(path.startswith("data/processed_v2/") for path in specs["v2-history"].paths)
        )
        planned = {path for spec in specs.values() for path in spec.paths}
        self.assertTrue(public_policy.PERMISSION_GATED_PUBLIC_PATHS.isdisjoint(planned))

    def test_public_bundle_plan_excludes_permission_gated_extracts(self) -> None:
        # Inventory semantics are tested against an isolated resolved state;
        # the development checkout intentionally keeps a stale final receipt.
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value=None,
        ):
            specs = bundles.bundle_specs(ROOT)
        planned = {path for spec in specs for path in spec.paths}
        self.assertTrue(bundles.PERMISSION_GATED_PUBLIC_PATHS.isdisjoint(planned))
        self.assertNotIn("benchmark/upgrade-bench-v1/MANIFEST.sha256", planned)
        self.assertFalse(any(path.startswith("results_v2/gpu_rolling/runs/") for path in planned))
        self.assertFalse(any(path.startswith("results_v2/gpu_smoke/") for path in planned))
        self.assertNotIn("results_v2/gpu_rolling/RUN_STATUS.json", planned)
        self.assertNotIn("results_v2/gpu_rolling/PILOT_INVALIDATED.json", planned)
        self.assertNotIn("PROJECT_CHECKLIST.md", planned)
        self.assertIn("paper/generated/v2_numbers.tex", planned)
        self.assertIn("paper/generated/v2_benchmark_profile.tex", planned)
        self.assertIn("paper/generated/v2_contemporary_references.tex", planned)
        self.assertTrue(any(path.startswith("data/processed_v2/") for path in planned))
        self.assertFalse(any(path.startswith("data/raw/") for path in planned))
        v2_package = next(spec for spec in specs if spec.bundle_id == "v2-standalone")
        self.assertIn("benchmark/upgrade-bench-v2/MANIFEST.sha256", v2_package.paths)
        self.assertIn("benchmark/upgrade-bench-v2/loader.py", v2_package.paths)
        self.assertFalse(any(spec.bundle_id.startswith("v1-") for spec in specs))
        v2_results = next(spec for spec in specs if spec.bundle_id == "v2-results")
        selected_results = {path for path in v2_results.paths if path.startswith("results_v2/")}
        self.assertTrue(
            {
                "results_v2/metrics/v2_gbdt_baselines.json",
                "results_v2/metrics/v2_gbdt_baselines.csv",
            }
            <= selected_results
        )
        self.assertLessEqual(selected_results, public_policy.PUBLIC_V2_RESULT_ALLOWLIST)
        # Reviewed outputs become public only after they exist; an allowlisted
        # future formal summary is not fabricated while its run is in flight.
        present_allowlisted = {
            path
            for path in public_policy.PUBLIC_V2_RESULT_ALLOWLIST
            if (ROOT / path).is_file()
        }
        self.assertEqual(
            selected_results,
            present_allowlisted,
        )

    def test_v2_results_are_exact_allowlist_and_new_files_default_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed_names = {
                "results_v2/metrics/v2_gpu_rolling_summary.json",
                "results_v2/metrics/v2_gbdt_baselines.json",
                "results_v2/metrics/v2_gbdt_baselines.csv",
                "results_v2/metrics/v2_value_diagnostics.json",
                "results_v2/metrics/v2_value_diagnostics.csv",
                "results_v2/metrics/v2_loco_transfer_summary.json",
                "results_v2/metrics/v2_loco_transfer_summary.csv",
                "results_v2/metrics/v2_ultra_zero_shot_summary.json",
                "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
                "results_v2/metrics/v2_benchmark_profile.json",
                "results_v2/metrics/v2_contemporary_references.json",
                "results_v2/metrics/v2_contemporary_references.csv",
                "results_v2/metrics/v2_product_space_density.json",
                "results_v2/metrics/v2_product_space_density.csv",
                "results_v2/scores/v2_product_space_density_scores.csv",
                "results_v2/metrics/v2_score_robustness_r5.json",
                "results_v2/metrics/v2_score_robustness_r5.csv",
                "results_v2/metrics/v2_eligibility_threshold_geometry.json",
                "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
            }
            unreviewed = root / "results_v2" / "metrics" / "future_debug.json"
            unreviewed.parent.mkdir(parents=True)
            for name in allowed_names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            unreviewed.write_text("{}\n", encoding="utf-8")
            # This fixture isolates exact-name selection; invalidation behavior
            # is covered separately and is deliberately fail-closed when the
            # notice is missing from a governed result checkout.
            with mock.patch.object(public_policy, "unresolved_v2_invalidation", return_value=None):
                self.assertEqual(
                    bundles._public_v2_results(root),
                    sorted(allowed_names),
                )

    def test_gbdt_json_csv_are_atomic_in_the_public_result_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            json_path = root / "results_v2" / "metrics" / "v2_gbdt_baselines.json"
            json_path.parent.mkdir(parents=True)
            json_path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ):
                with self.assertRaisesRegex(ValueError, "incomplete atomic public result"):
                    bundles._public_v2_results(root)

    def test_contemporary_json_csv_are_atomic_in_the_public_result_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            json_path = (
                root / "results_v2" / "metrics" / "v2_contemporary_references.json"
            )
            json_path.parent.mkdir(parents=True)
            json_path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ):
                with self.assertRaisesRegex(ValueError, "incomplete atomic public result"):
                    bundles._public_v2_results(root)

    def test_r5_result_groups_are_atomic_in_the_public_result_bundle(self) -> None:
        groups = (
            (
                "results_v2/metrics/v2_product_space_density.json",
                "results_v2/metrics/v2_product_space_density.csv",
                "results_v2/scores/v2_product_space_density_scores.csv",
            ),
            (
                "results_v2/metrics/v2_score_robustness_r5.json",
                "results_v2/metrics/v2_score_robustness_r5.csv",
            ),
            (
                "results_v2/metrics/v2_eligibility_threshold_geometry.json",
                "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
            ),
        )
        for group in groups:
            with self.subTest(group=group), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / group[0]
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")
                with mock.patch.object(
                    public_policy,
                    "unresolved_v2_invalidation",
                    return_value=None,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "incomplete atomic public result"
                    ):
                        bundles._public_v2_results(root)

    def test_new_public_summaries_are_allowlisted_but_formal_evidence_is_private(self) -> None:
        for path in (
            "results_v2/metrics/v2_value_diagnostics.json",
            "results_v2/metrics/v2_value_diagnostics.csv",
            "results_v2/metrics/v2_loco_transfer_summary.json",
            "results_v2/metrics/v2_loco_transfer_summary.csv",
            "results_v2/metrics/v2_ultra_zero_shot_summary.json",
            "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
            "results_v2/metrics/v2_gbdt_baselines.json",
            "results_v2/metrics/v2_gbdt_baselines.csv",
            "results_v2/metrics/v2_benchmark_profile.json",
            "results_v2/metrics/v2_contemporary_references.json",
            "results_v2/metrics/v2_contemporary_references.csv",
            "results_v2/metrics/v2_product_space_density.json",
            "results_v2/metrics/v2_product_space_density.csv",
            "results_v2/scores/v2_product_space_density_scores.csv",
            "results_v2/metrics/v2_score_robustness_r5.json",
            "results_v2/metrics/v2_score_robustness_r5.csv",
            "results_v2/metrics/v2_eligibility_threshold_geometry.json",
            "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
        ):
            with self.subTest(path=path):
                self.assertIn(path, public_policy.PUBLIC_V2_RESULT_ALLOWLIST)
                self.assertIsNone(public_policy.exclusion_reason(path))

        for path in (
            "tools/v2_loco_formal.py",
            "tests/test_v2_loco_formal.py",
            "tests/test_summarize_v2_loco_results.py",
            "jobs/v2_loco_formal_worker.sh",
            "jobs/v2_loco_formal_launch.sh",
            "results_v2/loco_formal/frozen_manifest.json",
            "results_v2/loco_formal/components/sheep/loco/seed_0/component.json",
            "results_v2/loco_formal/job_claims/sheep_loco_seed0/claim.txt",
            "results_v2/loco_formal/logs/worker.log",
            "results_v2/loco_formal/scores/score_A.csv",
            "results_v2/ultra_formal/frozen_manifest.json",
            "results_v2/ultra_formal/scores/score_A.csv",
            "results_v2/gpu_rolling/nbfnet_attestation/selection-host-a.json",
            "results_v2/gpu_rolling/nbfnet_attestation/formal_gate_main-host-a.json",
            "third_party/ULTRA/ckpts/ultra_4g.pth",
            "third_party/ULTRA/ultra/models.py",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(public_policy.exclusion_reason(path))

        expected_public = (
            "src/v2_ultra.py",
            "tools/v2_ultra_formal.py",
            "tools/summarize_v2_ultra_results.py",
            "configs/v2_ultra_formal.json",
            "requirements/ultra-formal.md",
            "tests/test_v2_ultra_formal.py",
            "tests/test_v2_ultra_protocol.py",
            "tests/test_summarize_v2_ultra_results.py",
            "tools/generate_v2_benchmark_profile.py",
            "tests/test_v2_benchmark_profile.py",
            "tools/summarize_v2_contemporary_references.py",
            "tests/test_summarize_v2_contemporary_references.py",
            "configs/v2_contemporary_references.json",
            "tools/v2_gbdt_baselines.py",
            "tests/test_v2_gbdt_baselines.py",
            "configs/v2_gbdt_baselines.json",
            "tools/v2_product_space_density.py",
            "tests/test_v2_product_space_density.py",
            "configs/v2_product_space_density.json",
            "tools/v2_score_robustness_r5.py",
            "tests/test_v2_score_robustness_r5.py",
            "configs/v2_score_robustness_r5.json",
            "tools/v2_eligibility_threshold_geometry.py",
            "tests/test_v2_eligibility_threshold_geometry.py",
            "configs/v2_eligibility_threshold_geometry.json",
            "tools/build_gpu_step3_postfreeze_attestation.py",
            "tests/test_gpu_step3_postfreeze_attestation.py",
            "chains/evidence/gpu_step3_postfreeze_semantic_attestation.json",
            "tools/build_nbfnet_source_attestation.py",
            "tests/test_nbfnet_source_attestation.py",
            "chains/evidence/nbfnet_selection-host-a.public.json",
            "chains/evidence/nbfnet_selection-host-b.public.json",
            "chains/evidence/nbfnet_main-host-a.public.json",
            "chains/evidence/nbfnet_main-host-b.public.json",
            "chains/evidence/nbfnet_source_comparison.json",
            "chains/evidence/nbfnet_runtime_comparison.json",
            "tools/verify_registry_curation_protocol.py",
            "tests/test_registry_curation_protocol.py",
            "chains/evidence/registry_curation_protocol.json",
            "tools/build_registry_human_validation_sample.py",
            "tests/test_registry_human_validation_sample.py",
            "chains/evidence/registry_human_validation_sample.json",
            "tools/prepare_registry_human_review_receipt.py",
            "tests/test_prepare_registry_human_review_receipt.py",
            "chains/evidence/registry_human_review_freeze.json",
        )
        for path in expected_public:
            with self.subTest(public_path=path):
                self.assertIsNone(public_policy.exclusion_reason(path))

    def test_post_resolution_summaries_do_not_rewrite_historical_receipt_scope(self) -> None:
        self.assertTrue(
            public_policy.POST_RESOLUTION_V2_RESULT_PATHS.isdisjoint(
                public_policy.V2_INVALIDATION_DERIVED_PATHS
            )
        )
        self.assertLessEqual(
            public_policy.POST_RESOLUTION_V2_RESULT_PATHS,
            public_policy.V2_INVALIDATION_HOLD_PATHS,
        )
        self.assertTrue(
            {
                "results_v2/metrics/v2_ultra_zero_shot_summary.json",
                "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
                "results_v2/metrics/v2_gbdt_baselines.json",
                "results_v2/metrics/v2_gbdt_baselines.csv",
                "results_v2/metrics/v2_benchmark_profile.json",
                "results_v2/metrics/v2_contemporary_references.json",
                "results_v2/metrics/v2_contemporary_references.csv",
                "results_v2/metrics/v2_product_space_density.json",
                "results_v2/metrics/v2_product_space_density.csv",
                "results_v2/scores/v2_product_space_density_scores.csv",
                "results_v2/metrics/v2_score_robustness_r5.json",
                "results_v2/metrics/v2_score_robustness_r5.csv",
                "results_v2/metrics/v2_eligibility_threshold_geometry.json",
                "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
            }
            <= public_policy.POST_RESOLUTION_V2_RESULT_PATHS
        )
        self.assertIn(
            "paper/generated/v2_benchmark_profile.tex",
            public_policy.V2_INVALIDATION_HOLD_PATHS,
        )
        self.assertIn(
            "paper/generated/v2_contemporary_references.tex",
            public_policy.V2_INVALIDATION_HOLD_PATHS,
        )
        self.assertNotIn(
            "paper/generated/v2_benchmark_profile.tex",
            public_policy.V2_INVALIDATION_DERIVED_PATHS,
        )
        self.assertNotIn(
            "paper/generated/v2_contemporary_references.tex",
            public_policy.V2_INVALIDATION_DERIVED_PATHS,
        )

    def test_schema7_extends_frozen_schema6_by_exact_gbdt_sources(self) -> None:
        additions = {
            "results_v2/metrics/v2_gbdt_baselines.json",
            "results_v2/metrics/v2_gbdt_baselines.csv",
            "tools/v2_gbdt_baselines.py",
            "configs/v2_gbdt_baselines.json",
        }
        self.assertEqual(len(public_policy.V2_PAPER_SCHEMA6_SOURCE_PATHS), 20)
        self.assertEqual(len(public_policy.V2_PAPER_SCHEMA7_SOURCE_PATHS), 24)
        self.assertEqual(
            public_policy.V2_PAPER_SCHEMA7_SOURCE_PATHS
            - public_policy.V2_PAPER_SCHEMA6_SOURCE_PATHS,
            additions,
        )
        self.assertEqual(public_policy.V2_PAPER_SCHEMA6_FINAL_NUMBER_KEY_COUNT, 625)
        self.assertEqual(public_policy.V2_PAPER_SCHEMA7_FINAL_NUMBER_KEY_COUNT, 694)
        self.assertEqual(
            public_policy.V2_PAPER_SCHEMA7_FINAL_NUMBER_KEYS_SHA256,
            "fdbd7453ac32c2719d62f6de3594b52810dd7b54b49ad3dd1c8aee3555a3d84c",
        )
        self.assertEqual(
            public_policy.V2_PAPER_SCHEMA7_FINAL_NUMBER_VALUES_SHA256,
            "d618c970e9caa547563879cbec64fc9ee259f50a36931e8d3d741941692aab43",
        )

    def test_schema8_extends_frozen_schema7_by_exact_r5_sources(self) -> None:
        additions = {
            "results_v2/metrics/v2_product_space_density.json",
            "results_v2/metrics/v2_product_space_density.csv",
            "results_v2/scores/v2_product_space_density_scores.csv",
            "tools/v2_product_space_density.py",
            "configs/v2_product_space_density.json",
            "results_v2/metrics/v2_score_robustness_r5.json",
            "results_v2/metrics/v2_score_robustness_r5.csv",
            "tools/v2_score_robustness_r5.py",
            "configs/v2_score_robustness_r5.json",
            "results_v2/metrics/v2_eligibility_threshold_geometry.json",
            "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
            "tools/v2_eligibility_threshold_geometry.py",
            "configs/v2_eligibility_threshold_geometry.json",
        }
        self.assertEqual(len(public_policy.V2_PAPER_SCHEMA7_SOURCE_PATHS), 24)
        self.assertEqual(len(public_policy.V2_PAPER_SOURCE_PATHS), 37)
        self.assertEqual(
            public_policy.V2_PAPER_SOURCE_PATHS
            - public_policy.V2_PAPER_SCHEMA7_SOURCE_PATHS,
            additions,
        )
        self.assertEqual(public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT, 857)
        self.assertEqual(
            public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256,
            "bcd64e94804bea64fb66d5e73e8d463597d68a8d88a46b7d7230c42a2dfd4dda",
        )
        self.assertEqual(
            public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256,
            "152048039fd1482e069139f113745c2987796fd15b74fa4822e65f4dd357ef04",
        )

    def test_unresolved_v2_invalidation_blocks_final_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notice = root / public_policy.PUBLIC_V2_INVALIDATION_NOTICE
            notice.parent.mkdir(parents=True)
            active = {
                "schema_version": public_policy.V2_INVALIDATION_SCHEMA,
                "status": public_policy.V2_INVALIDATION_ACTIVE_STATUS,
                "invalidated_at": public_policy.V2_INVALIDATION_DATE,
                "scope": sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS),
                "reason": public_policy.V2_INVALIDATION_REASON,
                "claim_policy": "Do not cite while this hold is active.",
                "resolution": "Rebuild and verify the normative scope.",
            }
            notice.write_bytes(public_policy._canonical_json_bytes(active))
            self.assertIn(
                public_policy.V2_INVALIDATION_ACTIVE_STATUS,
                public_policy.unresolved_v2_invalidation(root),
            )
            with self.assertRaisesRegex(ValueError, "unresolved"):
                bundles.make_index(root)

    def test_missing_invalidation_notice_is_fail_closed_for_governed_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(public_policy.unresolved_v2_invalidation(root))

            governed_result = root / "results_v2" / "metrics" / "rolling_cpu_baselines.json"
            governed_result.parent.mkdir(parents=True)
            governed_result.write_text("{}\n", encoding="utf-8")
            self.assertIn("notice is missing", public_policy.unresolved_v2_invalidation(root))
            with self.assertRaisesRegex(ValueError, "notice is missing"):
                bundles.make_index(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            governed_data = root / "data" / "processed_v2" / "candidates_sheep.csv"
            governed_data.parent.mkdir(parents=True)
            governed_data.write_text("i_iso,j_iso,stage\n", encoding="utf-8")
            self.assertIn("notice is missing", public_policy.unresolved_v2_invalidation(root))
            with self.assertRaisesRegex(ValueError, "notice is missing"):
                bundles.make_index(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "release" / "DATA_ARTIFACT_INDEX.json"
            index.parent.mkdir(parents=True)
            index.write_text("{}\n", encoding="utf-8")
            self.assertIn("notice is missing", public_policy.unresolved_v2_invalidation(root))
            with self.assertRaisesRegex(ValueError, "notice is missing"):
                bundles.make_index(root)

    def test_minimal_resolved_invalidation_marker_is_not_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notice = root / public_policy.PUBLIC_V2_INVALIDATION_NOTICE
            notice.parent.mkdir(parents=True)
            payload = {
                "status": "RESOLVED",
                "resolved_at": "2026-07-13T00:00:00Z",
            }
            notice.write_bytes(public_policy._canonical_json_bytes(payload))
            blocker = public_policy.unresolved_v2_invalidation(root)
            self.assertIn("field inventory", blocker)

    def test_file_inventory_excludes_runtime_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pkg" / "__pycache__").mkdir(parents=True)
            (root / "pkg" / "module.py").write_text("pass\n", encoding="utf-8")
            (root / "pkg" / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"local")
            (root / "pkg" / "scratch.tmp").write_bytes(b"local")
            self.assertEqual(bundles._files_under(root / "pkg", root), ["pkg/module.py"])

    def test_deterministic_archive_and_checksum_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "data" / "example.txt"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"stable payload\n")
            index_path = root / "release" / "DATA_ARTIFACT_INDEX.json"
            index_path.parent.mkdir(parents=True)
            index = fixture_index(
                "data/example.txt",
                payload.stat().st_size,
                bundles.sha256_file(payload),
            )
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ):
                self.assertTrue(bundles.verify_index(index_path, root, check_inventory=False))
                first = root / "out-a"
                second = root / "out-b"
                built_a = bundles.build_archives(
                    ["fixture"], output_dir=first, index_path=index_path, root=root
                )
                built_b = bundles.build_archives(
                    ["fixture"], output_dir=second, index_path=index_path, root=root
                )
                self.assertEqual(built_a[0][1], built_b[0][1])
                self.assertEqual(
                    (first / "fixture.zip").read_bytes(),
                    (second / "fixture.zip").read_bytes(),
                )
                self.assertTrue(bundles.verify_archives(first, index_path))

    def test_archive_verifier_rejects_partial_planned_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "data" / "example.txt"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"stable payload\n")
            index_path = root / "release" / "DATA_ARTIFACT_INDEX.json"
            index_path.parent.mkdir(parents=True)
            index = fixture_index(
                "data/example.txt",
                payload.stat().st_size,
                bundles.sha256_file(payload),
            )
            second = dict(index["bundles"][0])
            second.update({"id": "fixture-two", "archive": "fixture-two.zip"})
            index["bundles"].append(second)
            index_path.write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output = root / "dist"
            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ):
                bundles.build_archives(
                    ["fixture"], output_dir=output, index_path=index_path, root=root
                )
            errors = io.StringIO()
            with mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, index_path))
            self.assertIn("fixture-two.zip", errors.getvalue())

    def test_archive_verifier_rejects_extra_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output, index_path = self._build_single_fixture(Path(temp))
            (output / "unplanned.zip").write_bytes(b"not a release asset")
            errors = io.StringIO()
            with mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, index_path))
            self.assertIn("unexpected archive output: unplanned.zip", errors.getvalue())

    def test_archive_verifier_rejects_extra_nonarchive_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output, index_path = self._build_single_fixture(Path(temp))
            (output / "build.log").write_text("local log\n", encoding="utf-8")
            errors = io.StringIO()
            with mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, index_path))
            self.assertIn("unexpected archive output: build.log", errors.getvalue())

    def test_archive_verifier_rejects_tampered_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output, index_path = self._build_single_fixture(Path(temp))
            (output / "fixture.zip.sha256").write_text(
                "0" * 64 + "  fixture.zip\n",
                encoding="utf-8",
                newline="\n",
            )
            errors = io.StringIO()
            with mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, index_path))
            self.assertIn("archive checksum mismatch: fixture.zip", errors.getvalue())

    def test_archive_verifier_requires_exact_sha256sums(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output, index_path = self._build_single_fixture(Path(temp))
            (output / "SHA256SUMS").write_text(
                "0" * 64 + "  fixture.zip\n",
                encoding="utf-8",
                newline="\n",
            )
            errors = io.StringIO()
            with mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, index_path))
            self.assertIn("SHA256SUMS inventory or digest mismatch", errors.getvalue())

    def test_archive_verifier_rechecks_archive_after_member_walk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, index_path = self._build_single_fixture(root)
            archive = output / "fixture.zip"
            replacement = root / "replacement.zip"
            replacement.write_bytes(archive.read_bytes())
            real_zip_file = bundles.zipfile.ZipFile
            with real_zip_file(replacement, "a") as replacement_zip:
                replacement_zip.comment = b"concurrent replacement"
            replacement_bytes = replacement.read_bytes()
            self.assertNotEqual(
                bundles.sha256_file(archive), bundles.sha256_file(replacement)
            )
            swapped = False

            def racing_zip_file(path, *args, **kwargs):
                nonlocal swapped
                if Path(path) == archive and not swapped:
                    archive.write_bytes(replacement_bytes)
                    swapped = True
                return real_zip_file(path, *args, **kwargs)

            errors = io.StringIO()
            with mock.patch.object(
                bundles.zipfile, "ZipFile", side_effect=racing_zip_file
            ), mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, index_path))
            self.assertTrue(swapped)
            self.assertIn("archive changed during verification", errors.getvalue())

    def test_archive_verifier_rejects_symlink_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, index_path = self._build_single_fixture(root)
            link = root / "linked-index.json"
            try:
                os.symlink(index_path, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            errors = io.StringIO()
            with mock.patch("sys.stderr", errors):
                self.assertFalse(bundles.verify_archives(output, link))
            self.assertIn("index is not a physical regular file", errors.getvalue())

    def test_missing_payload_requires_explicit_repository_mode(self) -> None:
        for missing in ("data/processed_v2/missing.csv",):
            with self.subTest(path=missing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                index_path = root / "index.json"
                index = fixture_index(missing, 7, "0" * 64)
                index_path.write_text(json.dumps(index), encoding="utf-8")
                self.assertFalse(bundles.verify_index(index_path, root, check_inventory=False))
                self.assertTrue(
                    bundles.verify_index(
                        index_path,
                        root,
                        allow_missing=True,
                        check_inventory=False,
                    )
                )

    def test_superseded_payload_is_not_an_allow_missing_external_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path = root / "index.json"
            missing = "data/processed/superseded_missing.csv"
            index_path.write_text(
                json.dumps(fixture_index(missing, 7, "0" * 64)),
                encoding="utf-8",
            )
            self.assertFalse(
                bundles.verify_index(
                    index_path,
                    root,
                    allow_missing=True,
                    check_inventory=False,
                )
            )

    def test_partial_external_checkout_still_rejects_present_unlisted_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external = root / "data" / "processed_v2"
            external.mkdir(parents=True)
            unlisted = external / "new_table.csv"
            unlisted.write_text("x\n1\n", encoding="utf-8")
            index_path = root / "index.json"
            index = fixture_index("data/processed_v2/frozen_missing.csv", 7, "0" * 64)
            index_path.write_text(json.dumps(index), encoding="utf-8")
            current = bundles.BundleSpec(
                "fixture",
                "fixture.zip",
                "fixture",
                ("data/processed_v2/new_table.csv",),
            )
            errors = io.StringIO()
            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ), mock.patch.object(
                bundles,
                "bundle_specs",
                return_value=(current,),
            ), mock.patch("sys.stderr", errors):
                self.assertFalse(
                    bundles.verify_index(index_path, root, allow_missing=True, check_inventory=True)
                )
            self.assertIn("INDEX INVENTORY UNLISTED", errors.getvalue())

    def test_partial_external_checkout_accepts_exact_present_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path = root / "index.json"
            index = fixture_index("data/processed_v2/frozen_missing.csv", 7, "0" * 64)
            index_path.write_text(json.dumps(index), encoding="utf-8")
            current = bundles.BundleSpec("fixture", "fixture.zip", "fixture", ())
            with mock.patch.object(
                public_policy,
                "unresolved_v2_invalidation",
                return_value=None,
            ), mock.patch.object(bundles, "bundle_specs", return_value=(current,)):
                self.assertTrue(
                    bundles.verify_index(index_path, root, allow_missing=True, check_inventory=True)
                )

    def test_index_rejects_unsafe_member_path(self) -> None:
        for path in ("../escape.txt", "/absolute.txt", "C:/absolute.txt", "a\\b.txt"):
            with self.subTest(path=path):
                index = fixture_index(path, 1, "0" * 64)
                with self.assertRaises(ValueError):
                    bundles.validate_index_structure(index)

    def test_index_rejects_noncanonical_or_nested_archive_name(self) -> None:
        for archive in (
            "../escape.zip",
            "..\\escape.zip",
            "/absolute.zip",
            "C:/absolute.zip",
            "C:\\absolute.zip",
            "C:escape.zip",
            "nested/escape.zip",
            "nested\\escape.zip",
        ):
            with self.subTest(archive=archive):
                index = fixture_index("docs/example.txt", 1, "0" * 64)
                index["bundles"][0]["archive"] = archive
                with self.assertRaisesRegex(ValueError, "archive name"):
                    bundles.validate_index_structure(index)

    def test_output_path_containment_rejects_resolved_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "artifacts"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "outside output directory"):
                bundles._contained_output_path(output, "../escape.zip")
            self.assertEqual(
                bundles._contained_output_path(output, "safe.zip"),
                output / "safe.zip",
            )

    def test_index_and_manifest_reject_symbolic_link_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "repo"
            root.mkdir()
            outside = base / "private.txt"
            outside.write_text("private bytes\n", encoding="utf-8")
            link = root / "docs" / "public.txt"
            link.parent.mkdir()
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            index = fixture_index("docs/public.txt", outside.stat().st_size, digest)
            with self.assertRaisesRegex(ValueError, "symbolic-link"):
                bundles.validate_index_structure(index, root)

            manifest = root / "RELEASE_MANIFEST.sha256"
            manifest.write_text(f"{digest}  docs/public.txt\n", encoding="utf-8")
            with mock.patch.object(release_manifest, "ROOT", root):
                self.assertFalse(release_manifest.verify_manifest(manifest))

    def test_index_rejects_permission_gated_member_even_if_policy_is_declared(self) -> None:
        for path in (
            "data/processed/tariffs_all.csv",
            "data/processed/sps_ntm_sheep_raw.csv",
        ):
            with self.subTest(path=path):
                index = fixture_index(path, 1, "0" * 64)
                with self.assertRaisesRegex(ValueError, "public-policy-excluded"):
                    bundles.validate_index_structure(index)

    def test_index_rejects_gpu_internal_and_unallowlisted_v2_result(self) -> None:
        for path in (
            "results_v2/gpu_smoke/worker/cache.npy",
            "results_v2/gpu_rolling/logs/main.log",
            "results_v2/ultra_formal/scores/sheep_A.csv",
            "third_party/ULTRA/ckpts/ultra_4g.pth",
            "results_v2/metrics/unreviewed.json",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "public-policy-excluded"):
                    bundles.validate_index_structure(fixture_index(path, 1, "0" * 64))


class RepositorySizeGateTests(unittest.TestCase):
    def test_collection_fails_hard_file_and_total_limits(self) -> None:
        files = [
            size_gate.SizedPath("small", 4),
            size_gate.SizedPath("large", 12),
        ]
        failures = size_gate._check_collection(
            "fixture",
            files,
            max_file=10,
            warn_file=8,
            max_total=15,
            warn_total=14,
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("large" in item for item in failures))
        self.assertTrue(any("total" in item for item in failures))


class CleanClonePreflightTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(root: Path, entries: dict[str, str]) -> None:
        lines = release_manifest.manifest_header("public-repository-scope")
        lines += "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items()))
        (root / "RELEASE_MANIFEST.sha256").write_text(lines, encoding="utf-8")

    def test_copy_public_tree_is_exact_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            source.mkdir()
            payload = source / "README.md"
            payload.write_bytes(b"public\r\n")
            digest = hashlib.sha256(b"public\n").hexdigest()
            self._write_manifest(source, {"README.md": digest})
            destination = base / "export"
            entries = release_clean_clone.copy_public_tree(source, destination)
            self.assertEqual(entries, {"README.md": digest})
            self.assertEqual(
                release_clean_clone._tree_files(destination),
                {"README.md", "RELEASE_MANIFEST.sha256"},
            )

    def test_copy_public_tree_rejects_private_manifest_member(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            private = source / "private" / "cache.csv"
            private.parent.mkdir(parents=True)
            private.write_text("secret\n", encoding="utf-8")
            digest = release_manifest.sha256(private)
            self._write_manifest(source, {"private/cache.csv": digest})
            with self.assertRaisesRegex(release_clean_clone.CleanCloneError, "private path"):
                release_clean_clone.copy_public_tree(source, base / "export")

    def test_full_clean_clone_requires_frozen_archives(self) -> None:
        with self.assertRaisesRegex(release_clean_clone.CleanCloneError, "artifacts-dir"):
            release_clean_clone.run_clean_clone(profile="full", artifacts_dir=None)

    def test_active_hold_selects_planned_audit_only(self) -> None:
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value=release_clean_clone.ACTIVE_HOLD_BLOCKER,
        ):
            self.assertEqual(release_clean_clone._audit_mode(ROOT), "planned")

    def test_unknown_release_blocker_cannot_use_planned_audit(self) -> None:
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value="invalid receipt fixture",
        ):
            with self.assertRaisesRegex(
                release_clean_clone.CleanCloneError, "invalid or unsupported release state"
            ):
                release_clean_clone._audit_mode(ROOT)


class PublicReleaseAuditTests(unittest.TestCase):
    def test_public_git_and_bundles_exclude_superseded_benchmark(self) -> None:
        scope = set(release_manifest.release_scope())
        self.assertFalse(any(path.startswith("data/processed/") for path in scope))
        self.assertFalse(any(path.startswith("results/") for path in scope))
        self.assertFalse(
            any(path.startswith("benchmark/upgrade-bench-v1/") for path in scope)
        )
        if public_policy.unresolved_v2_invalidation(ROOT) is not None:
            self.assertNotIn("release/DATA_ARTIFACT_INDEX.json", scope)
        self.assertIn("configs/v2_loco_formal.json", scope)
        self.assertIn("configs/v2_gbdt_baselines.json", scope)
        self.assertIn("configs/v2_product_space_density.json", scope)
        self.assertIn("configs/v2_score_robustness_r5.json", scope)
        self.assertIn("configs/v2_eligibility_threshold_geometry.json", scope)
        self.assertIn("configs/v2_contemporary_references.json", scope)
        for registry_reproduction_path in (
            "tools/build_registry_revision.py",
            "tools/build_registry_lexicon_negative_control.py",
            "tests/test_registry_revision.py",
            "tests/test_registry_lexicon_negative_control.py",
        ):
            self.assertIn(registry_reproduction_path, scope)
        self.assertNotIn("data/raw/BACI_HS92_V202401b.zip", scope)
        total_bytes = sum((ROOT / path).stat().st_size for path in scope)
        self.assertLess(total_bytes, 50 * 1024**2)
        self.assertFalse(
            any(spec.bundle_id.startswith("v1-") for spec in bundles.bundle_specs(ROOT))
        )
        self.assertEqual(
            {path for path in scope if path.startswith("src/")},
            set(public_policy.PUBLIC_CURRENT_SOURCE_ALLOWLIST),
        )
        self.assertEqual(
            {path for path in scope if path.startswith("tools/")},
            set(public_policy.PUBLIC_CURRENT_TOOL_ALLOWLIST),
        )
        self.assertEqual(
            {path for path in scope if path.startswith("tests/")},
            set(public_policy.PUBLIC_CURRENT_TEST_ALLOWLIST),
        )

    def test_superseded_analysis_code_is_not_public(self) -> None:
        for relative in (
            "src/loco_transfer.py",
            "src/wits_tariffs.py",
            "src/natural_experiment.py",
            "tools/llm_audit_score.py",
            "tools/value_headroom.py",
            "tests/test_window_consumers.py",
            "requirements/ultra-smoke.md",
        ):
            with self.subTest(path=relative):
                self.assertIsNotNone(public_policy.exclusion_reason(relative, ROOT))

    def test_raw_hash_bound_crlf_results_disable_git_eol_conversion(self) -> None:
        marker = json.loads(
            (ROOT / public_policy.PUBLIC_V2_INVALIDATION_NOTICE).read_text(encoding="utf-8")
        )
        if marker.get("status") != public_policy.V2_INVALIDATION_RESOLVED_STATUS:
            self.skipTest("live invalidation receipt is intentionally ACTIVE during schema migration")
        proof_paths = set(marker["replacement_sha256"])
        proof_paths.update(marker["resolution_source_sha256"])
        proof_paths.update(marker["resolution_verifier_sha256"])
        missing = {path for path in proof_paths if not (ROOT / path).is_file()}
        self.assertTrue(
            all(path.startswith(public_policy.V2_EXTERNAL_SOURCE_PREFIXES) for path in missing),
            f"only allowlisted external receipt sources may be absent: {sorted(missing)}",
        )
        crlf_paths = {
            path
            for path in proof_paths - missing
            if b"\r\n" in (ROOT / path).read_bytes()
        }
        attributes = {
            line.split()[0]
            for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
            if line.strip()
            and not line.lstrip().startswith("#")
            and line.split()[-1] == "-text"
        }
        self.assertIn("*.csv", attributes)
        self.assertLessEqual(crlf_paths, attributes)

    @unittest.skipUnless(
        FULL_PAYLOAD_TESTS_ENABLED,
        "all external receipt sources are read only in the explicit full-profile stage",
    )
    def test_full_profile_crlf_check_reads_every_receipt_source(self) -> None:
        marker = json.loads(
            (ROOT / public_policy.PUBLIC_V2_INVALIDATION_NOTICE).read_text(encoding="utf-8")
        )
        if marker.get("status") != public_policy.V2_INVALIDATION_RESOLVED_STATUS:
            self.skipTest("live invalidation receipt is intentionally ACTIVE during schema migration")
        proof_paths = set(marker["replacement_sha256"])
        proof_paths.update(marker["resolution_source_sha256"])
        proof_paths.update(marker["resolution_verifier_sha256"])
        missing = sorted(path for path in proof_paths if not (ROOT / path).is_file())
        self.assertEqual(missing, [])
        for path in proof_paths:
            (ROOT / path).read_bytes()

    def test_sensitive_path_detector_ignores_web_url_but_catches_host_path(self) -> None:
        private_path = "ROOT=" + "/" + "home/account/project"
        self.assertIn("absolute Unix home path", public_release_audit._sensitive_labels(private_path))
        self.assertEqual(public_release_audit._sensitive_labels("https://example.org/home/article"), [])

    def test_sensitive_detector_catches_json_escaped_windows_path_and_bare_host(self) -> None:
        windows_path = "C:" + ("\\" * 2) + "Users" + ("\\" * 2) + "account" + ("\\" * 2) + "repo"
        self.assertIn(
            "absolute Windows user path",
            public_release_audit._sensitive_labels(windows_path),
        )
        host_alias = "mars" + "42"
        self.assertIn(
            "bare institutional host alias",
            public_release_audit._sensitive_labels(host_alias),
        )
        self.assertIn(
            "absolute Windows user path",
            public_release_audit._sensitive_labels("C:" + "/" + "Users/account/private/repo"),
        )

    def test_sensitive_detector_rejects_generic_cluster_account_family(self) -> None:
        synthetic_account = "s" + "li" + str(123)
        self.assertIn(
            "cluster account family",
            public_release_audit._sensitive_labels(f"cluster user {synthetic_account}"),
        )
        self.assertNotIn(
            "cluster account family",
            public_release_audit._sensitive_labels("public benchmark user"),
        )
        self.assertIn(
            "absolute macOS user path",
            public_release_audit._sensitive_labels("ROOT=" + "/" + "Users/account/private/repo"),
        )

    def test_bundle_payload_content_is_scanned_not_only_member_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "docs" / "fixture.json"
            path.parent.mkdir(parents=True)
            private = "C:" + "\\" + "Users" + "\\" + "account" + "\\" + "project"
            path.write_text(json.dumps({"workspace": private}), encoding="utf-8")
            failures = public_release_audit.audit_selected_files(
                {"docs/fixture.json"}, root, surface="fixture bundle"
            )
            self.assertTrue(any("absolute Windows user path" in item for item in failures))

    def test_selected_text_templates_manifests_and_version_files_are_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = "ROOT=" + "/" + "home/account/project"
            selected = {
                "env.sh.example",
                "pkg/MANIFEST",
                "pkg/MANIFEST.sha256",
                "pkg/VERSION",
                "requirements/tool.lock",
                "pkg/model.bin",
            }
            for name in selected - {"pkg/model.bin"}:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(private + "\n", encoding="utf-8")
            binary = root / "pkg/model.bin"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"\xff\x00binary")

            failures = public_release_audit.audit_selected_files(
                selected, root, surface="fixture bundle"
            )
            for name in selected - {"pkg/model.bin"}:
                self.assertTrue(
                    any(name in failure and "absolute Unix home path" in failure for failure in failures),
                    name,
                )
            self.assertFalse(any("pkg/model.bin" in failure for failure in failures))

    def test_release_scope_keeps_only_portable_jobs_and_sanitized_provenance(self) -> None:
        # Test the intended frozen public inventory independently of the live
        # development hold, which is expected to remain fail-closed.
        resolved_result_candidates = {
            "results_v2/metrics/rolling_cpu_baselines.json",
            "results_v2/metrics/v2_gpu_rolling_summary.json",
            "results_v2/metrics/v2_value_diagnostics.json",
            "results_v2/metrics/v2_value_diagnostics.csv",
            "results_v2/metrics/v2_ultra_zero_shot_summary.json",
            "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
            "results_v2/metrics/v2_benchmark_profile.json",
            "results_v2/metrics/v2_contemporary_references.json",
            "results_v2/metrics/v2_contemporary_references.csv",
            "results_v2/metrics/v2_gbdt_baselines.json",
            "results_v2/metrics/v2_gbdt_baselines.csv",
            "results_v2/metrics/v2_product_space_density.json",
            "results_v2/metrics/v2_product_space_density.csv",
            "results_v2/scores/v2_product_space_density_scores.csv",
            "results_v2/metrics/v2_score_robustness_r5.json",
            "results_v2/metrics/v2_score_robustness_r5.csv",
            "results_v2/metrics/v2_eligibility_threshold_geometry.json",
            "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
            "paper/generated/v2_numbers.tex",
            "paper/generated/v2_benchmark_profile.tex",
            "paper/generated/v2_contemporary_references.tex",
        }
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value=None,
        ):
            scope = set(release_manifest.release_scope())
            resolved_result_scope = set(
                release_manifest.public_release_scope(resolved_result_candidates)
            )
        self.assertIn("jobs/v2_gpu_select.pbs", scope)
        self.assertIn("jobs/v2_gpu_evaluate.pbs", scope)
        self.assertIn("jobs/v2_gpu_main_worker.sh", scope)
        self.assertIn("jobs/v2_gpu_nohup_worker.sh", scope)
        self.assertIn("configs/v2_gpu_rolling.json", scope)
        self.assertIn("configs/v2_loco_formal.json", scope)
        self.assertIn("configs/v2_ultra_formal.json", scope)
        self.assertIn("configs/v2_gbdt_baselines.json", scope)
        self.assertIn("configs/v2_product_space_density.json", scope)
        self.assertIn("configs/v2_score_robustness_r5.json", scope)
        self.assertIn("configs/v2_eligibility_threshold_geometry.json", scope)
        self.assertIn("benchmark/upgrade-bench-v2/MANIFEST.sha256", scope)
        self.assertNotIn("benchmark/upgrade-bench-v1/MANIFEST.sha256", scope)
        self.assertNotIn("jobs/repro_backtest.pbs", scope)
        for obsolete_launcher in (
            "run_backtest.sh",
            "run_benchmark.sh",
            "run_hp.sh",
            "run_train.sh",
        ):
            self.assertNotIn(obsolete_launcher, scope)
        self.assertNotIn("results/metrics/taxonomy_coder2_raw.json", scope)
        self.assertFalse(any(path.startswith("results/logs/") for path in scope))
        self.assertFalse(any(path.startswith("results_v2/gpu_rolling/") for path in scope))
        self.assertFalse(any(path.startswith("results_v2/gpu_smoke/") for path in scope))
        self.assertFalse(any(path.startswith("data/processed/") for path in scope))
        self.assertFalse(any(path.startswith("results/") for path in scope))
        self.assertNotIn("PROJECT_CHECKLIST.md", scope)
        self.assertEqual(resolved_result_candidates, resolved_result_scope)
        self.assertNotIn("results_v2/loco_formal/summary.json", scope)
        self.assertFalse(any(path.startswith("results_v2/ultra_formal/") for path in scope))
        self.assertFalse(any(path.startswith("third_party/ULTRA/") for path in scope))
        self.assertNotIn("tools/v2_loco_formal.py", scope)
        self.assertNotIn("tests/test_v2_loco_formal.py", scope)
        self.assertNotIn("jobs/v2_loco_formal_worker.sh", scope)
        self.assertIn("src/v2_ultra.py", scope)
        self.assertIn("tools/v2_ultra_formal.py", scope)
        if (ROOT / "tools/summarize_v2_ultra_results.py").is_file():
            self.assertIn("tools/summarize_v2_ultra_results.py", scope)
        if (ROOT / "requirements/ultra-formal.md").is_file():
            self.assertIn("requirements/ultra-formal.md", scope)
        self.assertIn("tools/generate_v2_benchmark_profile.py", scope)
        self.assertIn("tests/test_v2_benchmark_profile.py", scope)
        self.assertIn("tools/summarize_v2_contemporary_references.py", scope)
        self.assertIn("tests/test_summarize_v2_contemporary_references.py", scope)
        self.assertIn("tools/v2_gbdt_baselines.py", scope)
        self.assertIn("tests/test_v2_gbdt_baselines.py", scope)
        self.assertIn("tools/v2_product_space_density.py", scope)
        self.assertIn("tests/test_v2_product_space_density.py", scope)
        self.assertIn("tools/v2_score_robustness_r5.py", scope)
        self.assertIn("tests/test_v2_score_robustness_r5.py", scope)
        self.assertIn("tools/v2_eligibility_threshold_geometry.py", scope)
        self.assertIn("tests/test_v2_eligibility_threshold_geometry.py", scope)
        self.assertIn("tools/verify_registry_curation_protocol.py", scope)
        self.assertIn("tests/test_registry_curation_protocol.py", scope)
        self.assertIn("chains/evidence/registry_curation_protocol.json", scope)
        self.assertIn("tools/build_registry_human_validation_sample.py", scope)
        self.assertIn("tests/test_registry_human_validation_sample.py", scope)
        self.assertIn("chains/evidence/registry_human_validation_sample.json", scope)
        self.assertIn("tools/prepare_registry_human_review_receipt.py", scope)
        self.assertIn("tests/test_prepare_registry_human_review_receipt.py", scope)
        self.assertIn("chains/evidence/registry_human_review_freeze.json", scope)
        self.assertIn(public_policy.PUBLIC_V2_INVALIDATION_NOTICE, scope)
        self.assertTrue(public_policy.PERMISSION_GATED_PUBLIC_PATHS.isdisjoint(scope))

    def test_final_artifact_index_is_public_only_after_hold_resolution(self) -> None:
        candidate = {"release/DATA_ARTIFACT_INDEX.json"}
        self.assertIn(
            "release/DATA_ARTIFACT_INDEX.json",
            release_manifest.public_release_scope(candidate),
        )
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value="fixture active invalidation",
        ):
            self.assertNotIn(
                "release/DATA_ARTIFACT_INDEX.json",
                release_manifest.public_release_scope(candidate),
            )
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value=None,
        ):
            self.assertIn(
                "release/DATA_ARTIFACT_INDEX.json",
                release_manifest.public_release_scope(candidate),
            )

    def test_active_hold_suppresses_all_contemporary_numeric_interfaces(self) -> None:
        candidates = {
            "results_v2/metrics/v2_contemporary_references.json",
            "results_v2/metrics/v2_contemporary_references.csv",
            "paper/generated/v2_contemporary_references.tex",
        }
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value="fixture active invalidation",
        ):
            self.assertEqual(release_manifest.public_release_scope(candidates), [])
        with mock.patch.object(
            public_policy,
            "unresolved_v2_invalidation",
            return_value=None,
        ):
            self.assertEqual(
                set(release_manifest.public_release_scope(candidates)),
                candidates,
            )

    def test_public_and_internal_manifest_scopes_are_mechanically_distinct(self) -> None:
        files = {
            "README.md",
            "results_v2/gpu_rolling/RUN_STATUS.json",
            "results_v2/metrics/INVALIDATED.json",
        }
        internal = set(release_manifest.internal_release_scope(files))
        public = set(release_manifest.public_release_scope(files))
        self.assertLess(public, internal)
        self.assertIn("results_v2/gpu_rolling/RUN_STATUS.json", internal)
        self.assertNotIn("results_v2/gpu_rolling/RUN_STATUS.json", public)
        self.assertIn("results_v2/metrics/INVALIDATED.json", public)

    def test_numpy_hashes_are_raw_binary_while_text_normalizes_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b"header\r\npayload\r\n"
            npy = root / "scores.npy"
            npz = root / "scores.npz"
            txt = root / "scores.txt"
            for path in (npy, npz, txt):
                path.write_bytes(raw)
            raw_digest = hashlib.sha256(raw).hexdigest()
            normalized_digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
            self.assertEqual(release_manifest.sha256(npy), raw_digest)
            self.assertEqual(release_manifest.sha256(npz), raw_digest)
            self.assertEqual(release_manifest.sha256(txt), normalized_digest)

    def test_gitignore_keeps_external_v2_payload_visible_and_blocks_private_outputs(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertNotIn("data/processed_v2/\n", ignore)
        for entry in (
            "results_v2/gpu_rolling/",
            "results_v2/gpu_smoke/",
            "results_v2/loco_formal/",
            "configs/v2_gpu_hosts.json",
            "paper/*.fls",
            "paper/main-acm.pdf",
            "output/",
            "outputs/",
            "**/private/",
            "**/baci-filtered-cache*/",
            "data/processed/sps_ntm_sheep_raw.csv",
            "benchmark/upgrade-bench-v1/data/tariffs.csv",
            "benchmark/upgrade-bench-v1/data/sps_measures.csv",
        ):
            self.assertIn(entry, ignore)
        self.assertNotIn("!data/processed/tariffs_all.csv", ignore)
        for portable in (
            "docs/V2_GPU_MAIN_OPERATIONS.md",
            "jobs/v2_gpu_main_worker.sh",
            "jobs/v2_gpu_nohup_worker.sh",
            "requirements/v2-gpu-nodeps-lock.txt",
            "tools/step3_sync_manifest.py",
            "tools/v2_gpu_env_check.py",
        ):
            self.assertNotIn(portable + "\n", ignore)

    def test_private_cache_components_are_never_public(self) -> None:
        for path in (
            "private/baci-filtered-cache/manifest.json",
            "output/review-session/private-human-review.xlsx",
            "outputs/review-session/private-human-review.xlsx",
            "data/.private/baci_filtered_cache/years/baci_hs92_2010.csv.gz",
            "docs/tmp/baci-cache/manifest.json",
            "benchmark/upgrade-bench-v2/raw/BACI_HS92_Y2010.csv",
            "benchmark/upgrade-bench-v2/cache/manifest.json",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(public_policy.exclusion_reason(path))

    def test_claim_log_score_cache_and_raw_directories_are_never_public(self) -> None:
        for path in (
            "docs/claims/formal.json",
            "release/job_claims/worker.txt",
            "docs/logs/worker.txt",
            "release/scores/predictions.csv",
            "release/score_artifacts/score_A.csv",
            "release/cache/model.bin",
            "release/raw/source.csv",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(public_policy.exclusion_reason(path))

    def test_raw_baci_and_gpu_run_products_are_private_everywhere(self) -> None:
        for path in (
            "benchmark/upgrade-bench-v2/data/BACI_HS92_V202401b.zip",
            "data/processed_v2/gpu_scores_cocoa.csv",
            "docs/logs/worker.txt",
            "release/checkpoints/model.pt",
            "release/formal-worker.log",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(public_policy.exclusion_reason(path))

    def test_product_space_keyed_scores_are_the_only_public_score_exception(self) -> None:
        public_score = "results_v2/scores/v2_product_space_density_scores.csv"
        self.assertEqual(
            public_policy.PUBLIC_V2_DERIVED_SCORE_ALLOWLIST,
            frozenset({public_score}),
        )
        self.assertIsNone(public_policy.exclusion_reason(public_score))
        for path in (
            "results_v2/scores/another_density.csv",
            "results_v2/scores/formal_graph_scores.csv",
            "release/scores/predictions.csv",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(public_policy.exclusion_reason(path))

    def test_shared_policy_rejects_noncanonical_repository_paths(self) -> None:
        paths = (
            "../README.md",
            "/etc/passwd",
            "C:" + "/" + "Users/name/private.txt",
            "docs\\x.md",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertIn("unsafe/non-canonical", public_policy.exclusion_reason(path))

    def test_ci_declares_unit_split_v2_and_paper_number_gates(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release-artifact.yml").read_text(encoding="utf-8")
        for command in (
            "id: release_state",
            "python -m unittest discover",
            "Run active-hold DB snapshot tests",
            "python tools/test_split.py",
            "python tools/validate_v2.py",
            "python tools/generate_v2_paper_numbers.py --verify",
            "python tools/generate_v2_benchmark_profile.py --verify --profile repository",
            "python tools/generate_v2_benchmark_profile.py --verify --profile full",
            "python tools/v2_gbdt_baselines.py --verify-output",
            "tests.test_v2_gbdt_baselines",
            "python benchmark/upgrade-bench-v2/loader.py",
            "python tools/public_release_audit.py",
            "python tools/public_release_audit.py --planned-only",
            "python tools/release_manifest.py --verify --scope all",
        ):
            self.assertIn(command, workflow)
        for private_marker in (
            "tmp/ultra_formal_r",
            "UPGRADE_BENCH_PRIVATE_PROVENANCE_TESTS",
            "tools/v2_loco_formal.py",
        ):
            self.assertNotIn(private_marker, workflow)

        release_workflow = (ROOT / "docs" / "V2_RELEASE_WORKFLOW.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Private maintainer finalization", release_workflow)
        self.assertIn("not shipped in the public repository", release_workflow)
        self.assertIn("Public aggregate verification", release_workflow)

    def test_manual_full_payload_dispatch_is_boolean_and_fail_closed(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release-artifact.yml").read_text(
            encoding="utf-8"
        )
        for declaration in (
            "workflow_dispatch:\n    inputs:\n      require_full_payload:",
            "required: false",
            "default: false",
            "type: boolean",
        ):
            self.assertIn(declaration, workflow)

        hold_gate = """      - name: Reject requested full-payload mode during an active hold
        if: github.event_name == 'workflow_dispatch' && inputs.require_full_payload && steps.release_state.outputs.active == 'true'
        shell: bash
        run: |
          echo "Full-payload verification was requested, but the invalidation hold is active." >&2
          exit 1
"""
        absent_gate = """      - name: Reject requested full-payload mode when payload is absent
        if: github.event_name == 'workflow_dispatch' && inputs.require_full_payload && steps.v2_payload.outputs.present != 'true'
        shell: bash
        run: |
          echo "Full-payload verification was requested, but data/processed_v2/dataset_summary.json is absent." >&2
          echo "The workflow input is an assertion and does not download or install external payloads." >&2
          exit 1
"""
        self.assertIn(hold_gate, workflow)
        self.assertIn(absent_gate, workflow)
        self.assertLess(
            workflow.index(hold_gate),
            workflow.index("      - name: Install minimal benchmark dependencies"),
        )
        self.assertLess(
            workflow.index(absent_gate),
            workflow.index("      - name: Install minimal benchmark dependencies"),
        )

    def test_release_smoke_profiles_use_public_receipt_without_private_gpu_provenance(self) -> None:
        for profile in ("repository", "full"):
            with self.subTest(profile=profile):
                receipt = mock.Mock(
                    return_value={"replacement_sha256": {"results_v2/paper_numbers.json": "0" * 64}}
                )
                value = mock.Mock()
                loco = mock.Mock()
                ultra = mock.Mock()
                gbdt_validate = mock.Mock()
                gbdt_verify = mock.Mock()
                gbdt_payload = {"fixture": True}
                # Use always-public files as byte fixtures so this contract
                # test also runs in an active-hold development snapshot where
                # claim-bearing GBDT outputs are intentionally absent.
                gbdt_json = (
                    ROOT / "benchmark/upgrade-bench-v2/protocol_attestation.example.json"
                )
                gbdt_csv = ROOT / "requirements/baci_country_codes_V202401b.csv"
                benchmark_profile = mock.Mock()
                nbfnet_binding = mock.Mock()
                human_review_gate = mock.Mock(
                    return_value={
                        "audit_id": "fixture-review",
                        "disposition": {"kind": "NO_CONSTRUCT_CHANGE"},
                    }
                )
                fake_modules = {
                    "numpy": SimpleNamespace(__version__="test-numpy"),
                    "pandas": SimpleNamespace(__version__="test-pandas"),
                    "sklearn": SimpleNamespace(__version__="test-sklearn"),
                    "artifact_bundles": SimpleNamespace(verify_index=mock.Mock(return_value=True)),
                    "audit_chain_registry": SimpleNamespace(
                        verify_outputs=mock.Mock(
                            return_value={
                                "summary": {"included_codes": 1, "excluded_codes": 1}
                            }
                        )
                    ),
                    "build_gpu_step3_postfreeze_attestation": SimpleNamespace(
                        ARTIFACT_ROLE="chains/evidence/gpu_step3_postfreeze_semantic_attestation.json",
                        verify_summary_binding=mock.Mock(),
                    ),
                    "summarize_v2_gpu_results": SimpleNamespace(
                        verify_nbfnet_public_binding=nbfnet_binding,
                    ),
                    "generate_v2_benchmark_profile": SimpleNamespace(
                        verify_outputs=benchmark_profile
                    ),
                    "public_release_audit": SimpleNamespace(audit=mock.Mock(return_value=True)),
                    "registry_human_review_receipt": SimpleNamespace(
                        verify_release_gate=human_review_gate
                    ),
                    "release_manifest": SimpleNamespace(verify_all=mock.Mock(return_value=True)),
                    "resolve_v2_invalidation": SimpleNamespace(
                        verify_public_receipt=receipt
                    ),
                    "audit_v2": SimpleNamespace(verify_existing_output=mock.Mock()),
                    "generate_v2_paper_numbers": SimpleNamespace(
                        DEFAULT_TEX=ROOT / "paper/generated/v2_numbers.tex",
                        DEFAULT_JSON=ROOT / "results_v2/paper_numbers.json",
                        verify_outputs=mock.Mock(),
                    ),
                    "summarize_v2_loco_results": SimpleNamespace(
                        DEFAULT_JSON_OUT=ROOT / "results_v2/metrics/v2_loco_transfer_summary.json",
                        DEFAULT_CSV_OUT=ROOT / "results_v2/metrics/v2_loco_transfer_summary.csv",
                        verify_outputs=loco,
                    ),
                    "summarize_v2_ultra_results": SimpleNamespace(
                        DEFAULT_JSON_OUT=ROOT
                        / "results_v2/metrics/v2_ultra_zero_shot_summary.json",
                        DEFAULT_CSV_OUT=ROOT
                        / "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
                        verify_outputs=ultra,
                    ),
                    "v2_gbdt_baselines": SimpleNamespace(
                        DEFAULT_JSON=gbdt_json,
                        DEFAULT_CSV=gbdt_csv,
                        _strict_json_load=mock.Mock(return_value=gbdt_payload),
                        _strict_json_bytes=mock.Mock(return_value=gbdt_json.read_bytes()),
                        _csv_bytes=mock.Mock(return_value=gbdt_csv.read_bytes()),
                        validate_payload=gbdt_validate,
                        verify_existing_output=gbdt_verify,
                    ),
                    "v2_rolling_cpu_baselines": SimpleNamespace(
                        verify_existing_output=mock.Mock()
                    ),
                    "v2_value_diagnostics": SimpleNamespace(
                        verify_existing_output=value
                    ),
                    "validate_v2": SimpleNamespace(validate_release=mock.Mock()),
                }
                argv = ["release_smoke.py", "--skip-manifest", "--profile", profile]
                with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                    sys,
                    "argv",
                    argv,
                ), mock.patch.object(
                    release_smoke,
                    "check_v2_package",
                    return_value=None,
                ), mock.patch.object(
                    Path,
                    "read_text",
                    return_value="{}",
                ):
                    self.assertEqual(release_smoke.main(), 0)

                receipt.assert_called_once_with(release_smoke.ROOT, profile=profile)
                loco.assert_called_once_with(
                    fake_modules["summarize_v2_loco_results"].DEFAULT_JSON_OUT,
                    fake_modules["summarize_v2_loco_results"].DEFAULT_CSV_OUT,
                )
                if profile == "repository":
                    ultra.assert_not_called()
                    gbdt_validate.assert_called_once_with(
                        gbdt_payload,
                        verify_sources=False,
                    )
                    gbdt_verify.assert_not_called()
                else:
                    ultra.assert_called_once_with(
                        fake_modules["summarize_v2_ultra_results"].DEFAULT_JSON_OUT,
                        fake_modules["summarize_v2_ultra_results"].DEFAULT_CSV_OUT,
                    )
                    gbdt_validate.assert_not_called()
                    gbdt_verify.assert_called_once_with(
                        gbdt_json.resolve(),
                        gbdt_csv.resolve(),
                    )
                benchmark_profile.assert_called_once_with(mode=profile)
                nbfnet_binding.assert_called_once()
                human_review_gate.assert_called_once_with(release_smoke.ROOT)
                value.assert_not_called()
                fake_modules["generate_v2_paper_numbers"].verify_outputs.assert_not_called()

    def test_release_smoke_reports_early_gate_exceptions_cleanly(self) -> None:
        fake_modules = {
            "numpy": SimpleNamespace(__version__="test-numpy"),
            "pandas": SimpleNamespace(__version__="test-pandas"),
            "sklearn": SimpleNamespace(__version__="test-sklearn"),
            "artifact_bundles": SimpleNamespace(
                verify_index=mock.Mock(side_effect=OSError("broken public index"))
            ),
            "audit_chain_registry": SimpleNamespace(),
            "build_gpu_step3_postfreeze_attestation": SimpleNamespace(
                ARTIFACT_ROLE="chains/evidence/gpu_step3_postfreeze_semantic_attestation.json",
                verify_summary_binding=mock.Mock(),
            ),
            "public_release_audit": SimpleNamespace(audit=mock.Mock(return_value=True)),
            "registry_human_review_receipt": SimpleNamespace(
                verify_release_gate=mock.Mock()
            ),
            "release_manifest": SimpleNamespace(verify_all=mock.Mock(return_value=True)),
            "resolve_v2_invalidation": SimpleNamespace(),
        }
        argv = ["release_smoke.py", "--skip-manifest", "--profile", "repository"]
        with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
            sys,
            "argv",
            argv,
        ), mock.patch("builtins.print") as printed:
            self.assertEqual(release_smoke.main(), 1)
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "SMOKE FAILED: broken public index"
                and call.kwargs.get("file") is sys.stderr
                for call in printed.call_args_list
            )
        )

    def test_public_smoke_never_requires_internal_v1_manifest(self) -> None:
        smoke = (ROOT / "tools" / "release_smoke.py").read_text(encoding="utf-8")
        self.assertIn('verify_all(scope="release", strict=True)', smoke)
        self.assertNotIn('verify_all(scope="all", strict=True)', smoke)


if __name__ == "__main__":
    unittest.main()
