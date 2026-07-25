import contextlib
import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_v2_paper_numbers as paper_numbers  # noqa: E402
import public_release_policy as public_policy  # noqa: E402
import resolve_v2_invalidation as gate  # noqa: E402


def digest(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class ResolveV2InvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        review_paths = {
            "receipt": self.root / public_policy.REGISTRY_HUMAN_REVIEW_RECEIPT,
            "protocol": self.root / public_policy.REGISTRY_HUMAN_REVIEW_PROTOCOL,
            "freeze": self.root / public_policy.REGISTRY_HUMAN_REVIEW_FREEZE,
        }
        for path in review_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        review_paths["receipt"].write_bytes(b'{"fixture":"receipt"}\n')
        review_paths["protocol"].write_bytes(b'{"fixture":"protocol"}\n')
        review_paths["freeze"].write_bytes(b'{"fixture":"freeze"}\n')
        self.review_paths = review_paths
        self.review_receipt = {
            "audit_id": "fixture-review",
            "disposition": {"kind": "NO_CONSTRUCT_CHANGE"},
            "pre_review_freeze": {
                "path": public_policy.REGISTRY_HUMAN_REVIEW_FREEZE,
                "sha256": digest(review_paths["freeze"].read_bytes()),
            },
        }
        review_gate_patcher = mock.patch.object(
            gate,
            "_verify_human_review_release",
            return_value=self.review_receipt,
        )
        self.review_gate = review_gate_patcher.start()
        self.addCleanup(review_gate_patcher.stop)
        self.notice = self.root / gate.NOTICE_PATH
        self.notice.parent.mkdir(parents=True, exist_ok=True)
        self.active = {
            "schema_version": gate.ACTIVE_SCHEMA,
            "status": gate.ACTIVE_STATUS,
            "invalidated_at": "2026-07-12",
            "scope": sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS),
            "reason": gate.ACTIVE_REASON,
            "claim_policy": "Do not cite until every replacement is verified.",
            "resolution": "Rebuild and verify the complete normative scope.",
        }
        self._write_notice(self.active)

        for relative in sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative == gate.PAPER_TEX_PATH:
                path.write_bytes(b"old tex\n")
            elif relative == gate.PAPER_JSON_PATH:
                path.write_bytes(b'{"old":true}\n')
            else:
                path.write_bytes(f"verified:{relative}\n".encode("utf-8"))

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
        self.production_final_number_contract = (
            public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT,
            public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256,
            public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256,
        )
        self.production_current_number_contract = (
            public_policy.V2_PAPER_CURRENT_NUMBERS_SCHEMA,
            public_policy.V2_PAPER_CURRENT_NUMBER_KEY_COUNT,
            public_policy.V2_PAPER_CURRENT_NUMBER_KEYS_SHA256,
            public_policy.V2_PAPER_CURRENT_NUMBER_VALUES_SHA256,
        )
        number_contract_patches = (
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
        for patcher in number_contract_patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.verifier_hashes = {}
        for relative in gate.VERIFIER_SOURCE_PATHS:
            verifier = self.root / relative
            verifier.parent.mkdir(parents=True, exist_ok=True)
            verifier.write_bytes(f"verifier:{relative}\n".encode("utf-8"))
            self.verifier_hashes[relative] = digest(verifier.read_bytes())

        self.sources = {}
        for relative in sorted(public_policy.V2_PAPER_SOURCE_PATHS):
            source = self.root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                source.write_bytes(f"source:{relative}\n".encode("utf-8"))
            self.sources[relative] = digest(source.read_bytes())
        self.source = self.root / "docs/registry_audit.json"
        self.external_source = self.root / "data/processed_v2/dataset_summary.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_notice(self, payload: dict) -> None:
        self.notice.write_bytes(gate._json_bytes(payload))

    @contextlib.contextmanager
    def _fixture_verifiers(
        self,
        *,
        gpu_complete: bool = True,
        loco_complete: bool = True,
        ultra_complete: bool = True,
        gbdt_complete: bool = True,
    ):
        numbers = dict(self.numbers)
        if not gpu_complete:
            numbers["VTwoGPUStatus"] = "PENDING"
        if not loco_complete:
            numbers["VTwoLOCOStatus"] = "PENDING"
        if not ultra_complete:
            numbers["VTwoULTRAStatus"] = "PENDING"
        if not gbdt_complete:
            numbers["VTwoGBDTStatus"] = "PENDING"
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(gate, "_run_authoritative_verifiers", return_value=None)
            )
            stack.enter_context(
                mock.patch.object(
                    gate,
                    "_current_verifier_hashes",
                    return_value=dict(self.verifier_hashes),
                )
            )
            collect = stack.enter_context(
                mock.patch.object(
                    gate.paper_numbers,
                    "collect_numbers",
                    return_value=(numbers, dict(self.sources)),
                )
            )
            stack.enter_context(
                mock.patch.object(gate.paper_numbers, "verify_outputs", return_value=None)
            )
            yield collect

    def _canonical_snapshot(self) -> dict[str, bytes]:
        paths = list(gate.GENERATED_PATHS) + [gate.NOTICE_PATH]
        return {relative: (self.root / relative).read_bytes() for relative in paths}

    def _resolve_fixture(self) -> gate.ResolutionPlan:
        with self._fixture_verifiers():
            return gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)

    def _paper_payload(self) -> dict:
        return json.loads((self.root / gate.PAPER_JSON_PATH).read_text(encoding="utf-8"))

    def _write_paper_payload(self, payload: dict) -> None:
        (self.root / gate.PAPER_JSON_PATH).write_bytes(gate._json_bytes(payload))

    def _write_tex_payload(self, payload: dict) -> None:
        (self.root / gate.PAPER_TEX_PATH).write_bytes(
            gate.paper_numbers.render_tex(
                payload["numbers"],
                payload["sources"],
            ).encode("utf-8")
        )

    def _write_and_bind_paper_interfaces(self, receipt: dict, paper: dict) -> None:
        self._write_paper_payload(paper)
        self._write_tex_payload(paper)
        for relative in (gate.PAPER_JSON_PATH, gate.PAPER_TEX_PATH):
            receipt["replacement_sha256"][relative] = digest(
                (self.root / relative).read_bytes()
            )
        self._write_notice(receipt)

    def test_circular_dependency_is_broken_only_inside_read_only_plan(self) -> None:
        paths = paper_numbers.ArtifactPaths.under(self.root)
        with self.assertRaisesRegex(RuntimeError, "invalidation marker exists"):
            paper_numbers.write_outputs(
                self.root / gate.PAPER_TEX_PATH,
                self.root / gate.PAPER_JSON_PATH,
                paths=paths,
            )
        before = self._canonical_snapshot()
        with self._fixture_verifiers() as collect:
            plan = gate.prepare_resolution(
                self.root,
                now=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )
        self.assertEqual(before, self._canonical_snapshot())
        payload = json.loads(plan.generated_bytes[gate.PAPER_JSON_PATH])
        self.assertEqual(payload["schema_version"], gate.PAPER_SCHEMA)
        self.assertEqual(payload["gpu_status"], "COMPLETE")
        self.assertEqual(payload["loco_status"], "COMPLETE")
        self.assertEqual(payload["ultra_status"], "COMPLETE")
        self.assertEqual(payload["gbdt_status"], "COMPLETE")
        self.assertGreaterEqual(collect.call_count, 2)
        for call in collect.call_args_list:
            self.assertTrue(call.kwargs["require_gpu"])
            self.assertTrue(call.kwargs["require_loco"])
            self.assertTrue(call.kwargs["require_ultra"])
            self.assertTrue(call.kwargs["require_gbdt"])

    def test_resolution_requires_review_gate_but_noncanonical_preview_remains_available(self) -> None:
        before = self._canonical_snapshot()
        self.review_gate.side_effect = gate.ResolutionError("review receipt missing")
        with self._fixture_verifiers(), self.assertRaisesRegex(
            gate.ResolutionError, "review receipt missing"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

        preview_root = self.root / "review-only-preview"
        with self._fixture_verifiers():
            preview = gate.write_paper_preview(
                self.root,
                preview_root=preview_root,
            )
        self.assertEqual(preview.preview_root, preview_root.resolve())
        self.assertEqual(before, self._canonical_snapshot())

    def test_resolution_plan_seals_exact_canonical_review_evidence(self) -> None:
        with self._fixture_verifiers():
            plan = gate.prepare_resolution(self.root)
        marker = json.loads(plan.resolved_notice_bytes)
        binding = marker[public_policy.REGISTRY_HUMAN_REVIEW_BINDING_FIELD]
        self.assertEqual(binding["audit_id"], "fixture-review")
        self.assertEqual(binding["disposition"], "NO_CONSTRUCT_CHANGE")
        for role, relative in (
            ("receipt", public_policy.REGISTRY_HUMAN_REVIEW_RECEIPT),
            ("protocol", public_policy.REGISTRY_HUMAN_REVIEW_PROTOCOL),
            ("freeze", public_policy.REGISTRY_HUMAN_REVIEW_FREEZE),
        ):
            self.assertEqual(binding[f"{role}_path"], relative)
            self.assertEqual(
                binding[f"{role}_sha256"],
                digest((self.root / relative).read_bytes()),
            )

    def test_review_artifact_drift_after_prepare_fails_precommit_without_writes(self) -> None:
        before = self._canonical_snapshot()
        with self._fixture_verifiers():
            plan = gate.prepare_resolution(self.root)
            self.review_paths["protocol"].write_bytes(b"drifted review protocol\n")
            with self.assertRaisesRegex(
                gate.ResolutionError,
                "changed during resolution",
            ):
                gate._precommit_guard(plan)
        self.assertEqual(before, self._canonical_snapshot())

    def test_scope_shrink_and_extra_both_fail_without_writes(self) -> None:
        for changed_scope in (
            self.active["scope"][:-1],
            self.active["scope"] + ["results_v2/metrics/unreviewed.json"],
        ):
            payload = dict(self.active)
            payload["scope"] = changed_scope
            self._write_notice(payload)
            before = self._canonical_snapshot()
            with self._fixture_verifiers(), self.assertRaisesRegex(
                gate.ResolutionError, "normative scope"
            ):
                gate.prepare_resolution(self.root)
            self.assertEqual(before, self._canonical_snapshot())
        self._write_notice(self.active)

    def test_scope_path_escape_is_rejected_without_writes(self) -> None:
        payload = dict(self.active)
        payload["scope"] = list(payload["scope"])
        payload["scope"][0] = "../outside.json"
        self._write_notice(payload)
        before = self._canonical_snapshot()
        with self._fixture_verifiers(), self.assertRaisesRegex(
            gate.ResolutionError, "unsafe invalidation scope"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_prepare_rejects_noncanonical_active_notice_without_writes(self) -> None:
        self.notice.write_text(json.dumps(self.active), encoding="utf-8")
        before = self._canonical_snapshot()
        with self._fixture_verifiers(), self.assertRaisesRegex(
            gate.ResolutionError, "active invalidation notice: bytes are not canonical JSON"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_formal_gpu_pending_fails_before_any_canonical_write(self) -> None:
        before = self._canonical_snapshot()
        with self._fixture_verifiers(gpu_complete=False), self.assertRaisesRegex(
            gate.ResolutionError, "GPU summary is not complete"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_formal_loco_pending_fails_before_any_canonical_write(self) -> None:
        before = self._canonical_snapshot()
        with self._fixture_verifiers(loco_complete=False), self.assertRaisesRegex(
            gate.ResolutionError, "LOCO summary is not complete"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_formal_ultra_pending_fails_before_any_canonical_write(self) -> None:
        before = self._canonical_snapshot()
        with self._fixture_verifiers(ultra_complete=False), self.assertRaisesRegex(
            gate.ResolutionError, "ULTRA summary is not complete"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_formal_gbdt_pending_fails_before_any_canonical_write(self) -> None:
        before = self._canonical_snapshot()
        with self._fixture_verifiers(gbdt_complete=False), self.assertRaisesRegex(
            gate.ResolutionError, "GBDT summary is not complete"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_unfrozen_final_value_contract_blocks_resolution_without_writes(self) -> None:
        before = self._canonical_snapshot()
        with self._fixture_verifiers(), mock.patch.object(
            public_policy,
            "V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256",
            None,
        ), self.assertRaisesRegex(
            gate.ResolutionError,
            "final paper-number contract failed.*value digest is not frozen",
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())

    def test_unfrozen_contract_exports_reviewable_noncanonical_preview(self) -> None:
        before = self._canonical_snapshot()
        preview_root = self.root / "current-schema-review"
        with self._fixture_verifiers(), mock.patch.object(
            public_policy, "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT", None
        ), mock.patch.object(
            public_policy, "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256", None
        ), mock.patch.object(
            public_policy, "V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256", None
        ):
            preview = gate.write_paper_preview(
                self.root,
                preview_root=preview_root,
            )
        self.assertEqual(before, self._canonical_snapshot())
        self.assertEqual(
            preview.number_values_sha256,
            public_policy._paper_number_value_digest(self.numbers),
        )
        self.assertEqual(preview.number_key_count, len(self.numbers))
        self.assertEqual(
            preview.number_keys_sha256,
            public_policy._paper_number_key_digest(self.numbers),
        )
        for relative in gate.GENERATED_PATHS:
            self.assertEqual(
                (preview_root / relative).read_bytes(),
                preview.generated_bytes[relative],
            )

        with self._fixture_verifiers(), mock.patch.object(
            public_policy,
            "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT",
            preview.number_key_count,
        ), mock.patch.object(
            public_policy,
            "V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256",
            preview.number_keys_sha256,
        ), mock.patch.object(
            public_policy,
            "V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256",
            preview.number_values_sha256,
        ):
            plan = gate.prepare_resolution(self.root)
        self.assertEqual(dict(preview.generated_bytes), dict(plan.generated_bytes))
        self.assertEqual(before, self._canonical_snapshot())

    def test_preview_refuses_canonical_root_before_running_verifiers(self) -> None:
        before = self._canonical_snapshot()
        with mock.patch.object(
            gate,
            "_run_authoritative_verifiers",
            side_effect=AssertionError("must reject before verification"),
        ) as verifier, self.assertRaisesRegex(
            gate.ResolutionError,
            "must not be the repository root",
        ):
            gate.write_paper_preview(self.root, preview_root=self.root)
        verifier.assert_not_called()
        self.assertEqual(before, self._canonical_snapshot())

    def test_current_paper_payload_requires_all_formal_statuses(self) -> None:
        payload = gate._paper_payload(dict(self.numbers), dict(self.sources))
        self.assertEqual(payload["schema_version"], "upgrade-bench-v2-paper-numbers-8")
        self.assertEqual(payload["gpu_status"], "COMPLETE")
        self.assertEqual(payload["loco_status"], "COMPLETE")
        self.assertEqual(payload["ultra_status"], "COMPLETE")
        self.assertEqual(payload["gbdt_status"], "COMPLETE")
        self.assertEqual(
            gate._json_bytes(payload),
            paper_numbers.render_json(self.numbers, self.sources).encode("utf-8"),
        )
        incomplete = dict(self.numbers)
        incomplete["VTwoLOCOStatus"] = "PENDING"
        with self.assertRaisesRegex(gate.ResolutionError, "LOCO summary is not complete"):
            gate._paper_payload(incomplete, dict(self.sources))
        incomplete = dict(self.numbers)
        incomplete["VTwoULTRAStatus"] = "PENDING"
        with self.assertRaisesRegex(gate.ResolutionError, "ULTRA summary is not complete"):
            gate._paper_payload(incomplete, dict(self.sources))
        incomplete = dict(self.numbers)
        incomplete["VTwoGBDTStatus"] = "PENDING"
        with self.assertRaisesRegex(gate.ResolutionError, "GBDT summary is not complete"):
            gate._paper_payload(incomplete, dict(self.sources))
        for key, message in (
            ("VTwoProductSpaceStatus", "product-space summary is not complete"),
            (
                "VTwoScoreRobustnessRFiveStatus",
                "r5 score-robustness summary is not complete",
            ),
            (
                "VTwoEligibilityThresholdStatus",
                "eligibility-threshold geometry is not complete",
            ),
        ):
            with self.subTest(key=key):
                incomplete = dict(self.numbers)
                incomplete[key] = "PENDING"
                with self.assertRaisesRegex(gate.ResolutionError, message):
                    gate._paper_payload(incomplete, dict(self.sources))

    def test_committed_current_and_declared_final_number_contracts_are_static(self) -> None:
        paper, _ = public_policy._strict_canonical_json_file(
            ROOT / public_policy.V2_PAPER_NUMBERS_PATH,
            "committed paper-number interface",
        )
        numbers = paper["numbers"]
        current_schema, current_count, current_keys_digest, current_values_digest = (
            self.production_current_number_contract
        )
        final_count, final_keys_digest, final_values_digest = (
            self.production_final_number_contract
        )
        if None in (final_count, final_keys_digest, final_values_digest):
            self.assertEqual(paper["schema_version"], current_schema)
            self.assertEqual(len(numbers), current_count)
            self.assertEqual(
                public_policy._paper_number_key_digest(numbers),
                current_keys_digest,
            )
            self.assertEqual(
                public_policy._paper_number_value_digest(numbers),
                current_values_digest,
            )
        else:
            self.assertEqual(paper["schema_version"], public_policy.V2_PAPER_FINAL_NUMBERS_SCHEMA)
            self.assertEqual(len(numbers), final_count)
            self.assertEqual(
                public_policy._paper_number_key_digest(numbers),
                final_keys_digest,
            )
            self.assertEqual(
                public_policy._paper_number_value_digest(numbers),
                final_values_digest,
            )

    def test_resolution_proof_binds_public_summary_verifiers_not_private_run_trees(self) -> None:
        self.assertIs(
            gate.VERIFIER_SOURCE_PATHS,
            public_policy.V2_RESOLUTION_VERIFIER_SOURCE_PATHS,
        )
        self.assertIn(
            "tools/summarize_v2_loco_results.py",
            gate.VERIFIER_SOURCE_PATHS,
        )
        self.assertIn(
            "tools/build_gpu_step3_postfreeze_attestation.py",
            gate.VERIFIER_SOURCE_PATHS,
        )
        self.assertIn(
            "tools/summarize_v2_ultra_results.py",
            gate.VERIFIER_SOURCE_PATHS,
        )
        self.assertIn("tools/v2_gbdt_baselines.py", gate.VERIFIER_SOURCE_PATHS)
        self.assertNotIn("tools/v2_loco_formal.py", gate.VERIFIER_SOURCE_PATHS)
        self.assertNotIn("results_v2/ultra_formal", gate.VERIFIER_SOURCE_PATHS)

    def test_symlink_in_fixed_scope_is_rejected(self) -> None:
        target = self.root / "results_v2/metrics/raw_label_audit.json"
        external = self.root / "outside.json"
        external.write_bytes(b"not a release replacement\n")
        target.unlink()
        try:
            os.symlink(external, target)
        except OSError as exc:
            self.skipTest(f"symbolic links unavailable: {exc}")
        before_tex = (self.root / gate.PAPER_TEX_PATH).read_bytes()
        with self._fixture_verifiers(), self.assertRaisesRegex(
            gate.ResolutionError, "symbolic-link"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual((self.root / gate.PAPER_TEX_PATH).read_bytes(), before_tex)

    def test_partial_atomic_replace_failure_restores_every_original_byte(self) -> None:
        before = self._canonical_snapshot()

        def fail_on_marker(source: Path, target: Path) -> None:
            if Path(target) == self.notice:
                raise OSError("injected marker replacement failure")
            os.replace(source, target)

        with self._fixture_verifiers(), mock.patch.object(
            gate, "_replace_path", side_effect=fail_on_marker
        ), self.assertRaisesRegex(OSError, "injected"):
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
        self.assertEqual(before, self._canonical_snapshot())
        self.assertFalse(
            (self.root / "results_v2/metrics/.resolve_v2_invalidation.lock").exists()
        )

    def test_review_toctou_before_marker_promotion_rolls_back_generated_files(self) -> None:
        before = self._canonical_snapshot()

        def replace_then_drift(source: Path, target: Path) -> None:
            os.replace(source, target)
            if Path(target) == self.root / gate.PAPER_JSON_PATH:
                self.review_paths["protocol"].write_bytes(b"late protocol drift\n")

        with self._fixture_verifiers(), mock.patch.object(
            gate,
            "_replace_path",
            side_effect=replace_then_drift,
        ), self.assertRaisesRegex(gate.ResolutionError, "changed during resolution"):
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
        self.assertEqual(before, self._canonical_snapshot())

    def test_mutating_mode_requires_exact_confirmation_token(self) -> None:
        before = self._canonical_snapshot()
        with self.assertRaisesRegex(gate.ResolutionError, gate.CONFIRMATION_TOKEN):
            gate.resolve(self.root, confirmation="yes")
        self.assertEqual(before, self._canonical_snapshot())

    def test_freshen_refuses_an_already_active_hold_without_writes(self) -> None:
        before = self._canonical_snapshot()
        with self.assertRaisesRegex(gate.ResolutionError, "prior RESOLVED receipt"):
            gate.freshen_resolution(
                self.root,
                confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
            )
        self.assertEqual(before, self._canonical_snapshot())

    def test_freshen_rejects_noncanonical_resolved_notice_without_writes(self) -> None:
        self._resolve_fixture()
        resolved = json.loads(self.notice.read_text(encoding="utf-8"))
        self.notice.write_text(json.dumps(resolved), encoding="utf-8")
        before = self._canonical_snapshot()
        with self.assertRaisesRegex(
            gate.ResolutionError,
            "prior resolved invalidation notice: bytes are not canonical JSON",
        ):
            gate.freshen_resolution(
                self.root,
                confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
            )
        self.assertEqual(before, self._canonical_snapshot())

    def test_success_fixture_resolves_exact_scope_and_verifies(self) -> None:
        with self._fixture_verifiers():
            plan = gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
            verified = gate.verify_resolved(self.root)
        self.assertEqual(verified["status"], gate.RESOLVED_STATUS)
        self.assertEqual(verified["original_status"], gate.ACTIVE_STATUS)
        self.assertEqual(
            set(verified["replacement_sha256"]),
            set(public_policy.V2_INVALIDATION_DERIVED_PATHS),
        )
        self.assertEqual(
            verified["resolution_gate_sha256"],
            self.verifier_hashes["tools/resolve_v2_invalidation.py"],
        )
        self.assertEqual(verified["resolution_source_sha256"], self.sources)
        self.assertIsNone(public_policy.unresolved_v2_invalidation(self.root))
        self.assertEqual(
            digest((self.root / gate.PAPER_JSON_PATH).read_bytes()),
            plan.replacement_sha256[gate.PAPER_JSON_PATH],
        )

    def test_public_receipt_verifier_never_runs_private_authoritative_verifiers(self) -> None:
        self._resolve_fixture()
        with mock.patch.object(
            gate,
            "_run_authoritative_verifiers",
            side_effect=AssertionError("private verifier must not run"),
        ) as private, mock.patch.object(
            gate.paper_numbers,
            "collect_numbers",
            side_effect=AssertionError("paper collector must not open private-bound inputs"),
        ) as collector:
            verified = gate.verify_public_receipt(self.root, profile="full")
        self.assertEqual(verified["status"], gate.RESOLVED_STATUS)
        private.assert_not_called()
        collector.assert_not_called()

    def test_public_receipt_rejects_later_review_artifact_drift(self) -> None:
        self._resolve_fixture()
        self.review_paths["receipt"].write_bytes(b"later receipt drift\n")
        with self.assertRaisesRegex(
            gate.ResolutionError,
            "registry human-review binding hash mismatch",
        ):
            gate.verify_public_receipt(self.root, profile="repository")
        self.assertIsNotNone(public_policy.unresolved_v2_invalidation(self.root))

    def test_public_receipt_rejects_review_binding_identifier_or_hash_tamper(self) -> None:
        self._resolve_fixture()
        original = json.loads(self.notice.read_text(encoding="utf-8"))
        variants = []
        audit_tamper = json.loads(json.dumps(original))
        audit_tamper[gate.REVIEW_BINDING_FIELD]["audit_id"] = "different-audit"
        variants.append((audit_tamper, "changed during resolution"))
        disposition_tamper = json.loads(json.dumps(original))
        disposition_tamper[gate.REVIEW_BINDING_FIELD]["disposition"] = "CHANGES_REQUIRED"
        variants.append((disposition_tamper, "not release-eligible"))
        hash_tamper = json.loads(json.dumps(original))
        hash_tamper[gate.REVIEW_BINDING_FIELD]["freeze_sha256"] = "0" * 64
        variants.append((hash_tamper, "registry human-review binding hash mismatch"))
        for payload, message in variants:
            with self.subTest(message=message):
                self._write_notice(payload)
                with self.assertRaisesRegex(gate.ResolutionError, message):
                    gate.verify_public_receipt(self.root, profile="repository")

    def test_verify_resolved_rechecks_semantic_review_gate_after_slow_verifiers(self) -> None:
        self._resolve_fixture()
        self.review_gate.reset_mock()
        self.review_gate.side_effect = [
            self.review_receipt,
            gate.ResolutionError("late semantic review drift"),
        ]
        with self._fixture_verifiers(), self.assertRaisesRegex(
            gate.ResolutionError,
            "late semantic review drift",
        ):
            gate.verify_resolved(self.root)
        self.assertEqual(self.review_gate.call_count, 2)

    def test_public_receipt_rejects_minimal_resolved_marker(self) -> None:
        self._resolve_fixture()
        self._write_notice(
            {
                "status": "RESOLVED",
                "resolved_at": "2026-07-14T00:00:00Z",
            }
        )
        with self.assertRaisesRegex(gate.ResolutionError, "field inventory"):
            gate.verify_public_receipt(self.root, profile="repository")
        self.assertIn("field inventory", public_policy.unresolved_v2_invalidation(self.root))

    def test_public_receipt_rejects_superseded_status(self) -> None:
        self._resolve_fixture()
        payload = json.loads(self.notice.read_text(encoding="utf-8"))
        payload["status"] = "SUPERSEDED"
        self._write_notice(payload)
        with self.assertRaisesRegex(gate.ResolutionError, "status transition"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_extra_and_missing_fields(self) -> None:
        self._resolve_fixture()
        original = json.loads(self.notice.read_text(encoding="utf-8"))
        variants = []
        missing = dict(original)
        missing.pop("resolution_source_sha256")
        variants.append(missing)
        extra = dict(original)
        extra["unreviewed"] = True
        variants.append(extra)
        for payload in variants:
            with self.subTest(fields=sorted(payload)):
                self._write_notice(payload)
                with self.assertRaisesRegex(gate.ResolutionError, "field inventory"):
                    gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_duplicate_nonfinite_and_noncanonical_json(self) -> None:
        self._resolve_fixture()
        canonical = self.notice.read_bytes()

        self.notice.write_bytes(b'{"status":"RESOLVED",' + canonical[1:])
        with self.assertRaisesRegex(gate.ResolutionError, "duplicate JSON object key"):
            gate.verify_public_receipt(self.root, profile="repository")

        payload = json.loads(canonical)
        payload["resolved_at"] = float("nan")
        self.notice.write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        with self.assertRaisesRegex(gate.ResolutionError, "strict JSON"):
            gate.verify_public_receipt(self.root, profile="repository")

        payload = json.loads(canonical)
        self.notice.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(gate.ResolutionError, "not canonical JSON"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_duplicate_normalized_scope(self) -> None:
        self._resolve_fixture()
        payload = json.loads(self.notice.read_text(encoding="utf-8"))
        first = next(
            path
            for path in sorted(public_policy.V2_INVALIDATION_DERIVED_PATHS)
            if path.startswith("results_v2/metrics/")
        )
        payload["scope"].append(Path(first).name)
        self._write_notice(payload)
        with self.assertRaisesRegex(gate.ResolutionError, "duplicate normalized"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_wrong_gate_and_verifier_hashes(self) -> None:
        self._resolve_fixture()
        original = json.loads(self.notice.read_text(encoding="utf-8"))

        payload = json.loads(json.dumps(original))
        payload["resolution_gate_sha256"] = "0" * 64
        self._write_notice(payload)
        with self.assertRaisesRegex(gate.ResolutionError, "gate hash differs"):
            gate.verify_public_receipt(self.root, profile="repository")

        payload = json.loads(json.dumps(original))
        payload["resolution_verifier_sha256"]["tools/public_release_policy.py"] = "0" * 64
        self._write_notice(payload)
        with self.assertRaisesRegex(gate.ResolutionError, "resolution verifier hash mismatch"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_paper_source_map_mismatch(self) -> None:
        self._resolve_fixture()
        payload = json.loads(self.notice.read_text(encoding="utf-8"))
        payload["resolution_source_sha256"]["docs/registry_audit.json"] = "0" * 64
        self._write_notice(payload)
        with self.assertRaisesRegex(gate.ResolutionError, "differs from the current"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_requires_exact_current_source_inventory(self) -> None:
        self._resolve_fixture()
        receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        paper = self._paper_payload()
        omitted = "configs/v2_ultra_formal.json"
        paper["sources"].pop(omitted)
        receipt["resolution_source_sha256"].pop(omitted)
        self._write_paper_payload(paper)
        receipt["replacement_sha256"][gate.PAPER_JSON_PATH] = digest(
            (self.root / gate.PAPER_JSON_PATH).read_bytes()
        )
        self._write_notice(receipt)
        with self.assertRaisesRegex(gate.ResolutionError, "source_sha256 inventory mismatch"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_requires_frozen_date_and_canonical_utc_timestamp(self) -> None:
        self._resolve_fixture()
        original = json.loads(self.notice.read_text(encoding="utf-8"))
        variants = (
            ("invalidated_at", "2026-07-13", "original invalidation date"),
            ("resolved_at", "2026-07-14T00:00:00+00:00", "canonical UTC timestamp"),
            ("resolved_at", "2026-07-14T00:00:00.123Z", "canonical UTC timestamp"),
        )
        for key, value, message in variants:
            with self.subTest(key=key, value=value):
                payload = json.loads(json.dumps(original))
                payload[key] = value
                self._write_notice(payload)
                with self.assertRaisesRegex(gate.ResolutionError, message):
                    gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_wrong_source_hash_even_when_maps_agree(self) -> None:
        self._resolve_fixture()
        receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        paper = self._paper_payload()
        paper["sources"]["docs/registry_audit.json"] = "0" * 64
        receipt["resolution_source_sha256"]["docs/registry_audit.json"] = "0" * 64
        self._write_and_bind_paper_interfaces(receipt, paper)
        with self.assertRaisesRegex(gate.ResolutionError, "paper-number source hash mismatch"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_missing_repository_source(self) -> None:
        self._resolve_fixture()
        self.source.unlink()
        with self.assertRaisesRegex(
            gate.ResolutionError,
            "file is missing: docs/registry_audit.json",
        ):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_repository_profile_allows_only_missing_external_source(self) -> None:
        self._resolve_fixture()
        self.external_source.unlink()
        verified = gate.verify_public_receipt(self.root, profile="repository")
        self.assertEqual(verified["status"], gate.RESOLVED_STATUS)
        self.assertIsNone(public_policy.unresolved_v2_invalidation(self.root))
        with self.assertRaisesRegex(
            gate.ResolutionError,
            "file is missing: data/processed_v2/dataset_summary.json",
        ):
            gate.verify_public_receipt(self.root, profile="full")

    def test_repository_profile_does_not_open_external_source_bytes(self) -> None:
        self._resolve_fixture()
        self.external_source.write_bytes(b"tampered external payload\n")
        verified = gate.verify_public_receipt(self.root, profile="repository")
        self.assertEqual(verified["status"], gate.RESOLVED_STATUS)
        with self.assertRaisesRegex(
            gate.ResolutionError,
            "paper-number source hash mismatch: data/processed_v2/dataset_summary.json",
        ):
            gate.verify_public_receipt(self.root, profile="full")

    def test_public_receipt_requires_exact_complete_paper_interface(self) -> None:
        self._resolve_fixture()
        original_receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        original_paper = self._paper_payload()
        variants = []
        extra = json.loads(json.dumps(original_paper))
        extra["unexpected"] = True
        variants.append((extra, "field inventory"))
        pending = json.loads(json.dumps(original_paper))
        pending["ultra_status"] = "PENDING"
        variants.append((pending, "not complete canonical schema 8"))
        for paper, message in variants:
            with self.subTest(message=message):
                self._write_paper_payload(paper)
                receipt = json.loads(json.dumps(original_receipt))
                receipt["replacement_sha256"][gate.PAPER_JSON_PATH] = digest(
                    (self.root / gate.PAPER_JSON_PATH).read_bytes()
                )
                self._write_notice(receipt)
                with self.assertRaisesRegex(gate.ResolutionError, message):
                    gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_requires_exact_paper_macro_inventory(self) -> None:
        self._resolve_fixture()
        original_receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        original_paper = self._paper_payload()
        variants = []
        missing = json.loads(json.dumps(original_paper))
        missing["numbers"].pop("VTwoFixture")
        variants.append(missing)
        extra = json.loads(json.dumps(original_paper))
        extra["numbers"]["VTwoUnexpected"] = "1"
        variants.append(extra)
        for paper in variants:
            with self.subTest(keys=sorted(paper["numbers"])):
                receipt = json.loads(json.dumps(original_receipt))
                self._write_and_bind_paper_interfaces(receipt, paper)
                with self.assertRaisesRegex(gate.ResolutionError, "macro inventory mismatch"):
                    gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_malformed_or_json_tex_divergent_values(self) -> None:
        self._resolve_fixture()
        receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        paper = self._paper_payload()
        paper["numbers"]["VTwoFixture"] = "٩.٩٩٩٩"
        self._write_and_bind_paper_interfaces(receipt, paper)
        with self.assertRaisesRegex(gate.ResolutionError, "malformed value"):
            gate.verify_public_receipt(self.root, profile="repository")

        self._write_notice(self.active)
        self._resolve_fixture()
        receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        tex = self.root / gate.PAPER_TEX_PATH
        tex.write_bytes(
            tex.read_bytes().replace(
                b"\\newcommand{\\VTwoFixture}{1.0000}",
                b"\\newcommand{\\VTwoFixture}{2.0000}",
            )
        )
        receipt["replacement_sha256"][gate.PAPER_TEX_PATH] = digest(tex.read_bytes())
        self._write_notice(receipt)
        with self.assertRaisesRegex(gate.ResolutionError, "JSON/TeX macro maps differ"):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_public_receipt_rejects_synchronized_forged_number_values(self) -> None:
        self._resolve_fixture()
        receipt = json.loads(self.notice.read_text(encoding="utf-8"))
        paper = self._paper_payload()
        paper["numbers"]["VTwoFixture"] = "9.9999"
        self._write_and_bind_paper_interfaces(receipt, paper)
        with self.assertRaisesRegex(
            gate.ResolutionError,
            "paper-number frozen value digest mismatch",
        ):
            gate.verify_public_receipt(self.root, profile="repository")

    def test_legacy_schema4_resolved_receipt_refreshes_atomically_to_current(self) -> None:
        with self._fixture_verifiers():
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)

        prior = json.loads(self.notice.read_text(encoding="utf-8"))
        prior["resolution_verifier_sha256"] = {
            path: sha256
            for path, sha256 in prior["resolution_verifier_sha256"].items()
            if path in gate.LEGACY_SCHEMA4_VERIFIER_SOURCE_PATHS
        }
        self._write_notice(prior)
        paper_tex = self.root / gate.PAPER_TEX_PATH
        paper_tex.write_bytes(b"stale legacy paper bytes\n")
        before = self._canonical_snapshot()

        with self.assertRaisesRegex(
            gate.ResolutionError, "active invalidation notice"
        ):
            gate.prepare_resolution(self.root)
        self.assertEqual(before, self._canonical_snapshot())
        with self.assertRaisesRegex(
            gate.ResolutionError, gate.FRESHEN_CONFIRMATION_TOKEN
        ):
            gate.freshen_resolution(self.root, confirmation="yes")
        self.assertEqual(before, self._canonical_snapshot())

        generated_before = {
            relative: (self.root / relative).read_bytes()
            for relative in gate.GENERATED_PATHS
        }
        active = gate.freshen_resolution(
            self.root,
            confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
        )
        self.assertEqual(active["status"], gate.ACTIVE_STATUS)
        self.assertEqual(set(active), set(gate.ACTIVE_FIELDS))
        self.assertEqual(
            generated_before,
            {
                relative: (self.root / relative).read_bytes()
                for relative in gate.GENERATED_PATHS
            },
        )
        self.assertIsNotNone(public_policy.unresolved_v2_invalidation(self.root))

        after_freshen = self._canonical_snapshot()
        with self._fixture_verifiers():
            plan = gate.prepare_resolution(
                self.root,
                now=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        self.assertEqual(after_freshen, self._canonical_snapshot())
        planned = json.loads(plan.resolved_notice_bytes)
        self.assertEqual(
            set(planned["resolution_verifier_sha256"]),
            set(gate.VERIFIER_SOURCE_PATHS),
        )

        with self._fixture_verifiers():
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
            verified = gate.verify_resolved(self.root)
        self.assertEqual(verified["status"], gate.RESOLVED_STATUS)
        self.assertEqual(
            set(verified["resolution_verifier_sha256"]),
            set(gate.VERIFIER_SOURCE_PATHS),
        )
        self.assertNotEqual(paper_tex.read_bytes(), b"stale legacy paper bytes\n")

    def test_legacy_schema5_resolved_receipt_can_be_freshened(self) -> None:
        self._resolve_fixture()
        prior = json.loads(self.notice.read_text(encoding="utf-8"))
        prior["resolution_verifier_sha256"] = {
            path: sha256
            for path, sha256 in prior["resolution_verifier_sha256"].items()
            if path in gate.LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS
        }
        self._write_notice(prior)

        active = gate.freshen_resolution(
            self.root,
            confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
        )
        self.assertEqual(active["status"], gate.ACTIVE_STATUS)
        self.assertEqual(set(active), set(gate.ACTIVE_FIELDS))

    def test_legacy_schema6_resolved_receipt_can_be_freshened(self) -> None:
        self._resolve_fixture()
        prior = json.loads(self.notice.read_text(encoding="utf-8"))
        prior["resolution_verifier_sha256"] = {
            path: sha256
            for path, sha256 in prior["resolution_verifier_sha256"].items()
            if path in gate.LEGACY_SCHEMA6_VERIFIER_SOURCE_PATHS
        }
        self._write_notice(prior)
        active = gate.freshen_resolution(
            self.root,
            confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
        )
        self.assertEqual(active["status"], gate.ACTIVE_STATUS)
        self.assertEqual(set(active), set(gate.ACTIVE_FIELDS))

    def test_legacy_schema7_resolved_receipt_can_be_freshened(self) -> None:
        self._resolve_fixture()
        prior = json.loads(self.notice.read_text(encoding="utf-8"))
        prior["resolution_verifier_sha256"] = {
            path: sha256
            for path, sha256 in prior["resolution_verifier_sha256"].items()
            if path in gate.LEGACY_SCHEMA7_VERIFIER_SOURCE_PATHS
        }
        self._write_notice(prior)
        active = gate.freshen_resolution(
            self.root,
            confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
        )
        self.assertEqual(active["status"], gate.ACTIVE_STATUS)
        self.assertEqual(set(active), set(gate.ACTIVE_FIELDS))

    def test_current_schema8_resolved_receipt_can_be_freshened(self) -> None:
        self._resolve_fixture()
        active = gate.freshen_resolution(
            self.root,
            confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
        )
        self.assertEqual(active["status"], gate.ACTIVE_STATUS)
        self.assertEqual(set(active), set(gate.ACTIVE_FIELDS))

    def test_resolved_receipt_with_unknown_verifier_inventory_cannot_refresh(self) -> None:
        with self._fixture_verifiers():
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
        payload = json.loads(self.notice.read_text(encoding="utf-8"))
        payload["resolution_verifier_sha256"]["tools/unrecognized_verifier.py"] = digest(
            "unrecognized"
        )
        self._write_notice(payload)
        before = self._canonical_snapshot()
        with self.assertRaisesRegex(
            gate.ResolutionError, "unrecognized verifier inventory"
        ):
            gate.freshen_resolution(
                self.root,
                confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
            )
        self.assertEqual(before, self._canonical_snapshot())

    def test_freshen_post_replace_failure_restores_prior_receipt(self) -> None:
        with self._fixture_verifiers():
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
        before = self._canonical_snapshot()

        def replace_then_fail(source: Path, target: Path) -> None:
            os.replace(source, target)
            raise OSError("injected post-replace freshen failure")

        with mock.patch.object(
            gate,
            "_replace_path",
            side_effect=replace_then_fail,
        ), self.assertRaisesRegex(OSError, "post-replace freshen failure"):
            gate.freshen_resolution(
                self.root,
                confirmation=gate.FRESHEN_CONFIRMATION_TOKEN,
            )
        self.assertEqual(before, self._canonical_snapshot())
        self.assertFalse(
            (self.root / "results_v2/metrics/.resolve_v2_invalidation.lock").exists()
        )

    def test_freshen_cli_is_explicit_and_mutually_exclusive(self) -> None:
        args = gate.build_parser().parse_args(
            [
                "--freshen",
                "--confirm",
                gate.FRESHEN_CONFIRMATION_TOKEN,
            ]
        )
        self.assertTrue(args.freshen)
        self.assertEqual(args.confirm, gate.FRESHEN_CONFIRMATION_TOKEN)

    def test_public_receipt_cli_exposes_repository_and_full_profiles(self) -> None:
        for profile in ("repository", "full"):
            args = gate.build_parser().parse_args(
                ["--verify-public-receipt", "--profile", profile]
            )
            self.assertTrue(args.verify_public_receipt)
            self.assertEqual(args.profile, profile)

    def test_preview_cli_is_explicit_and_exclusive(self) -> None:
        preview = self.root / "review"
        args = gate.build_parser().parse_args(["--preview-dir", str(preview)])
        self.assertEqual(args.preview_dir, preview)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            gate.build_parser().parse_args(
                ["--preview-dir", str(preview), "--dry-run"]
            )

    def test_tampered_replacement_and_notice_hash_are_rejected(self) -> None:
        with self._fixture_verifiers():
            gate.resolve(self.root, confirmation=gate.CONFIRMATION_TOKEN)
            target = self.root / "results_v2/metrics/v2_robustness.csv"
            target.write_bytes(target.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(gate.ResolutionError, "replacement hash mismatch"):
                gate.verify_resolved(self.root)

        # Restore the file and then corrupt only its proof in the marker.
        target.write_bytes(b"verified:results_v2/metrics/v2_robustness.csv\n")
        payload = json.loads(self.notice.read_text(encoding="utf-8"))
        payload["replacement_sha256"]["results_v2/metrics/v2_robustness.csv"] = "0" * 64
        self._write_notice(payload)
        with self._fixture_verifiers(), self.assertRaisesRegex(
            gate.ResolutionError, "replacement hash mismatch"
        ):
            gate.verify_resolved(self.root)


if __name__ == "__main__":
    unittest.main()
