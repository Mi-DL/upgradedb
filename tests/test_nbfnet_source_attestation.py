import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]

from tools import build_nbfnet_source_attestation as attestation


OBSERVED = "2026-07-16T15:00:00+00:00"
SELECTION_STARTED = "2026-07-16T14:00:00+00:00"


def all_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_strings(child)
    elif isinstance(value, str):
        yield value


class NbfnetSourceAttestationTest(unittest.TestCase):
    def setUp(self):
        # This fixture does not need a repository-relative path.  Using the
        # platform temporary directory keeps it hermetic in a clean checkout,
        # where the ignored ROOT/tmp directory is intentionally absent.
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "external" / "NBFNet-PyG"
        self.source.mkdir(parents=True)
        (self.source / "z.py").write_text("z = 1\n", encoding="utf-8")
        (self.source / "a.txt").write_text("alpha\n", encoding="utf-8")
        (self.source / "nbfnet").mkdir()
        (self.source / "nbfnet" / "model.py").write_text(
            "class Model: pass\n", encoding="utf-8"
        )
        (self.source / "__pycache__").mkdir()
        (self.source / "__pycache__" / "model.pyc").write_bytes(b"cache")
        (self.source / "data").mkdir()
        (self.source / "data" / "runtime.csv").write_text(
            "not,source\n", encoding="utf-8"
        )
        (self.source / "checkpoint.pt").write_bytes(b"weights")

        # Keep the retrospective-mtime fixture independent of the wall clock.
        # The selected start timestamp above is intentionally fixed, so source
        # files created by this test must also receive a fixed earlier mtime.
        for source_file in (
            self.source / "z.py",
            self.source / "a.txt",
            self.source / "nbfnet" / "model.py",
        ):
            os.utime(source_file, (1_700_000_000, 1_700_000_000))

        self.runtime = self.root / "runtime" / "rspmm.so"
        self.runtime.parent.mkdir()
        self.runtime.write_bytes(b"compiled-rspmm-v1")

        self.frozen = self.root / "frozen_manifest.json"
        self.frozen.write_text(
            json.dumps(
                {
                    "schema_version": "upgrade-bench-v2/gpu-freeze/1",
                    "run_id": "formal-test-r1",
                    "status": "frozen",
                    "all_selections_frozen_before_main": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.step3 = self.root / "STEP3_SYNC_MANIFEST.sha256"
        self.step3.write_text("a" * 64 + "  src/runner.py\n", encoding="utf-8")
        self.identity = attestation.RunIdentityInputs(
            run_id="formal-test-r1",
            frozen_manifest=self.frozen,
            frozen_manifest_sha256=attestation.sha256_file(self.frozen),
            step3_manifest=self.step3,
            step3_manifest_sha256=attestation.sha256_file(self.step3),
        )
        self.environment = mock.patch.dict(
            os.environ,
            {attestation.SOURCE_ENVIRONMENT_VARIABLE: str(self.source)},
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def receipt(self, *, host_role="gpu-host-a", runtime=True):
        expected_runtime = (
            {"rspmm-extension": attestation.sha256_file(self.runtime)}
            if runtime
            else None
        )
        return attestation.build_private_receipt(
            self.identity,
            source_root=self.source,
            observed_at_utc=OBSERVED,
            runtime_artifacts=(
                [("rspmm-extension", self.runtime)] if runtime else []
            ),
            expected_runtime_sha256=expected_runtime,
            host_role=host_role if runtime else None,
            selection_started_at_utc=SELECTION_STARTED,
        )

    def test_sorted_complete_inventory_and_documented_exclusions(self):
        receipt = self.receipt()
        inventory = receipt["source"]["inventory"]
        paths = [row["path"] for row in inventory["files"]]
        self.assertEqual(paths, ["a.txt", "nbfnet/model.py", "z.py"])
        self.assertEqual(paths, sorted(paths, key=lambda value: value.encode("utf-8")))
        self.assertEqual(inventory["file_count"], 3)
        self.assertRegex(inventory["tree_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(inventory["inventory_sha256"], r"^[0-9a-f]{64}$")
        policy = inventory["exclusion_policy"]
        self.assertIn(".git", policy["directory_basenames"])
        self.assertIn("__pycache__", policy["directory_basenames"])
        self.assertIn("data", policy["directory_basenames"])
        self.assertIn(".pyc", policy["file_suffixes_case_insensitive"])
        self.assertIn(".pt", policy["file_suffixes_case_insensitive"])
        self.assertTrue(policy["all_other_regular_files_hashed"])
        self.assertTrue(
            receipt["selection_timing"][
                "latest_source_mtime_not_after_selection_start"
            ]
        )
        self.assertIn("not proof", receipt["selection_timing"]["interpretation"])

    def test_atomic_no_overwrite_verify_and_tree_tamper_fail_closed(self):
        receipt = self.receipt()
        path = self.root / "private_receipt.json"
        original = attestation.render_json(receipt)
        attestation.atomic_create(path, original)
        self.assertEqual(path.read_bytes(), original)
        verified = attestation.verify_private_receipt(
            path,
            self.identity,
            source_root=self.source,
            runtime_artifacts=[("rspmm-extension", self.runtime)],
            host_role="gpu-host-a",
            expected_runtime_sha256={
                "rspmm-extension": attestation.sha256_file(self.runtime)
            },
        )
        self.assertEqual(
            verified["source"]["inventory"]["tree_sha256"],
            receipt["source"]["inventory"]["tree_sha256"],
        )
        with self.assertRaisesRegex(
            attestation.SourceAttestationError, "refusing to overwrite"
        ):
            attestation.atomic_create(path, b"replacement")
        self.assertEqual(path.read_bytes(), original)

        (self.source / "a.txt").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(
            attestation.SourceAttestationError, "current NBFNET_PATH tree differs"
        ):
            attestation.verify_private_receipt(
                path, self.identity, source_root=self.source
            )

    def test_run_identity_and_environment_binding_fail_closed(self):
        wrong_identity = attestation.RunIdentityInputs(
            run_id="different-run",
            frozen_manifest=self.frozen,
            frozen_manifest_sha256=self.identity.frozen_manifest_sha256,
            step3_manifest=self.step3,
            step3_manifest_sha256=self.identity.step3_manifest_sha256,
        )
        with self.assertRaisesRegex(
            attestation.SourceAttestationError, "run_id differs"
        ):
            attestation.build_private_receipt(
                wrong_identity, source_root=self.source, observed_at_utc=OBSERVED
            )
        other = self.root / "other-source"
        other.mkdir()
        (other / "model.py").write_text("pass\n", encoding="utf-8")
        with self.assertRaisesRegex(
            attestation.SourceAttestationError, "differs from NBFNET_PATH"
        ):
            attestation.build_private_receipt(
                self.identity, source_root=other, observed_at_utc=OBSERVED
            )

    def test_public_projection_redacts_private_paths_and_binds_private_bytes(self):
        private = self.receipt()
        private_path = self.root / "private.json"
        private_bytes = attestation.render_json(private)
        attestation.atomic_create(private_path, private_bytes)
        public = attestation.project_public_receipt(private_path)
        self.assertEqual(public["schema_version"], attestation.PUBLIC_SCHEMA)
        self.assertTrue(public["source"]["resolved_path_redacted"])
        self.assertTrue(public["runtime"]["observed_hostname_redacted"])
        self.assertTrue(
            public["runtime"]["artifacts"][0]["resolved_path_redacted"]
        )
        self.assertEqual(
            public["private_evidence"]["receipt_sha256"],
            attestation.sha256_bytes(private_bytes),
        )
        private_source = str(self.source).lower()
        private_runtime = str(self.runtime).lower()
        for value in all_strings(public):
            self.assertFalse(PurePosixPath(value).is_absolute(), value)
            self.assertFalse(PureWindowsPath(value).is_absolute(), value)
            self.assertNotIn(private_source, value.lower())
            self.assertNotIn(private_runtime, value.lower())

        public_path = self.root / "public.json"
        attestation.atomic_create(public_path, attestation.render_json(public))
        attestation.verify_public_projection(public_path, private_path)

    def test_cross_host_runtime_comparison_requires_and_compares_explicit_binary(self):
        first = self.receipt(host_role="gpu-host-a")
        first_path = self.root / "host_a.json"
        first_path.write_bytes(attestation.render_json(first))

        second = copy.deepcopy(first)
        second["runtime"]["host_role"] = "gpu-host-b"
        second["runtime"]["observed_hostname"] = "private-host-b"
        second_path = self.root / "host_b.json"
        second_path.write_bytes(attestation.render_json(second))
        comparison = attestation.compare_runtime_receipts(
            [second_path, first_path]
        )
        self.assertEqual(comparison["status"], "PASS")
        self.assertTrue(comparison["all_source_trees_match"])
        self.assertTrue(comparison["all_runtime_artifacts_match"])
        self.assertEqual(comparison["host_roles"], ["gpu-host-a", "gpu-host-b"])
        self.assertEqual(comparison["runtime_artifact_roles"], ["rspmm-extension"])

        mismatched = copy.deepcopy(second)
        mismatched["runtime"]["artifacts"][0]["sha256"] = "0" * 64
        mismatched["runtime"]["artifacts"][0]["expected_sha256"] = "0" * 64
        mismatch_path = self.root / "host_b_mismatch.json"
        mismatch_path.write_bytes(attestation.render_json(mismatched))
        mismatch = attestation.compare_runtime_receipts(
            [first_path, mismatch_path]
        )
        self.assertEqual(mismatch["status"], "MISMATCH")
        self.assertFalse(mismatch["all_runtime_artifacts_match"])

    def test_external_and_frozen_main_source_receipts_compare_without_runtime(self):
        external = self.receipt(runtime=False)
        external["runtime"]["host_role"] = "selection-external"
        external_path = self.root / "selection_external.json"
        external_path.write_bytes(attestation.render_json(external))

        frozen = copy.deepcopy(external)
        frozen["runtime"]["host_role"] = "main-frozen-snapshot"
        frozen_path = self.root / "main_frozen.json"
        frozen_path.write_bytes(attestation.render_json(frozen))
        comparison = attestation.compare_source_receipts(
            [external_path, frozen_path]
        )
        self.assertEqual(comparison["status"], "PASS")
        self.assertTrue(comparison["all_source_trees_match"])
        self.assertEqual(comparison["receipt_count"], 2)

        changed = copy.deepcopy(frozen)
        changed["source"]["inventory"]["files"][0]["sha256"] = "0" * 64
        # Recompute the internal inventory digests so the receipt is structurally
        # valid but attests genuinely different bytes.
        files = changed["source"]["inventory"]["files"]
        canonical = json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        changed["source"]["inventory"]["inventory_sha256"] = (
            attestation.sha256_bytes(canonical)
        )
        tree = __import__("hashlib").sha256()
        for row in files:
            tree.update(row["path"].encode("utf-8"))
            tree.update(b"\0")
            tree.update(str(row["size_bytes"]).encode("ascii"))
            tree.update(b"\0")
            tree.update(row["sha256"].encode("ascii"))
            tree.update(b"\n")
        changed["source"]["inventory"]["tree_sha256"] = tree.hexdigest()
        changed_path = self.root / "main_frozen_changed.json"
        changed_path.write_bytes(attestation.render_json(changed))
        mismatch = attestation.compare_source_receipts(
            [external_path, changed_path]
        )
        self.assertEqual(mismatch["status"], "MISMATCH")
        self.assertFalse(mismatch["all_source_trees_match"])

    def test_formal_gate_binds_read_only_main_snapshot_and_four_receipts(self):
        run_root = self.root / "isolated-run"
        frozen_source = run_root / "private" / "nbfnet_source_frozen"
        shutil.copytree(self.source, frozen_source)
        frozen_manifest = run_root / "results_v2" / "gpu_rolling" / "frozen_manifest.json"
        frozen_manifest.parent.mkdir(parents=True)
        shutil.copy2(self.frozen, frozen_manifest)
        step3 = (
            run_root
            / "results_v2"
            / "gpu_rolling"
            / "runs"
            / "formal-test-r1"
            / "STEP3_SYNC_MANIFEST.sha256"
        )
        step3.parent.mkdir(parents=True)
        shutil.copy2(self.step3, step3)
        identity = attestation.RunIdentityInputs(
            run_id="formal-test-r1",
            frozen_manifest=frozen_manifest,
            frozen_manifest_sha256=attestation.sha256_file(frozen_manifest),
            step3_manifest=step3,
            step3_manifest_sha256=attestation.sha256_file(step3),
        )
        runtime = run_root / "torch_extensions" / "rspmm" / "rspmm.so"
        runtime.parent.mkdir(parents=True)
        shutil.copy2(self.runtime, runtime)
        runtime_sha = attestation.sha256_file(runtime)
        receipts = run_root / "private" / "nbfnet_receipts"
        receipts.mkdir()

        with mock.patch.dict(
            os.environ,
            {attestation.SOURCE_ENVIRONMENT_VARIABLE: str(self.source)},
        ):
            selection_a = attestation.build_private_receipt(
                identity,
                source_root=self.source,
                observed_at_utc=OBSERVED,
                host_role="selection-host-a",
            )
        selection_b = copy.deepcopy(selection_a)
        selection_b["runtime"]["host_role"] = "selection-host-b"
        selection_a_path = receipts / "selection_a.json"
        selection_b_path = receipts / "selection_b.json"
        selection_a_path.write_bytes(attestation.render_json(selection_a))
        selection_b_path.write_bytes(attestation.render_json(selection_b))

        with mock.patch.dict(
            os.environ,
            {attestation.SOURCE_ENVIRONMENT_VARIABLE: str(frozen_source)},
        ):
            main_a = attestation.build_private_receipt(
                identity,
                source_root=frozen_source,
                observed_at_utc=OBSERVED,
                runtime_artifacts=[("rspmm-extension", runtime)],
                expected_runtime_sha256={"rspmm-extension": runtime_sha},
                host_role="main-host-a",
            )
            main_b = copy.deepcopy(main_a)
            main_b["runtime"]["host_role"] = "main-host-b"
            main_a_path = receipts / "main_a.json"
            main_b_path = receipts / "main_b.json"
            main_a_path.write_bytes(attestation.render_json(main_a))
            main_b_path.write_bytes(attestation.render_json(main_b))
            with mock.patch.object(
                attestation, "verify_read_only_source_snapshot"
            ) as read_only:
                result = attestation.formal_gate(
                    receipt_path=main_a_path,
                    identity_inputs=identity,
                    run_root=run_root,
                    source_root=frozen_source,
                    runtime_artifacts=[("rspmm-extension", runtime)],
                    expected_runtime_sha256={"rspmm-extension": runtime_sha},
                    host_role="main-host-a",
                    source_peer_receipts=[
                        selection_a_path,
                        selection_b_path,
                        main_b_path,
                    ],
                    runtime_peer_receipts=[main_b_path],
                )
        read_only.assert_called_once_with(frozen_source.resolve())
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["all_source_trees_match"])
        self.assertTrue(result["all_runtime_artifacts_match"])
        self.assertEqual(result["source_peer_count"], 3)

    def test_formal_snapshot_rejects_unattested_python_bytecode(self):
        snapshot = self.root / "bytecode-snapshot"
        snapshot.mkdir()
        (snapshot / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        cache = snapshot / "__pycache__"
        cache.mkdir()
        (cache / "model.cpython-312.pyc").write_bytes(b"unattested-bytecode")
        with self.assertRaisesRegex(
            attestation.SourceAttestationError,
            "executable bytecode outside the source inventory",
        ):
            attestation.verify_read_only_source_snapshot(snapshot)

    def test_formal_gate_cli_returns_pending_for_missing_receipt(self):
        missing = self.root / "missing.json"
        existing_peer = self.root / "existing-peer.json"
        existing_peer.write_text("{}", encoding="utf-8")
        result = attestation.main(
            [
                "formal-gate",
                "--receipt",
                str(missing),
                "--run-root",
                str(self.root),
                "--source-peer-receipt",
                str(existing_peer),
                "--run-id",
                self.identity.run_id,
                "--frozen-manifest",
                str(self.frozen),
                "--frozen-manifest-sha256",
                self.identity.frozen_manifest_sha256,
                "--step3-manifest",
                str(self.step3),
                "--step3-manifest-sha256",
                self.identity.step3_manifest_sha256,
                "--source-root",
                str(self.source),
                "--host-role",
                "main-host-a",
            ]
        )
        self.assertEqual(result, 75)

    @unittest.skipUnless(shutil.which("git"), "Git is required for provenance test")
    def test_git_head_dirty_tracked_changes_and_commit_timestamps_are_recorded(self):
        repository = self.root / "git-source"
        repository.mkdir()
        source_file = repository / "model.py"
        source_file.write_text("value = 1\n", encoding="utf-8")
        commands = [
            ["git", "init", "-q"],
            ["git", "add", "model.py"],
            [
                "git",
                "-c",
                "user.name=Attestation Test",
                "-c",
                "user.email=attestation@example.invalid",
                "commit",
                "-q",
                "-m",
                "initial",
            ],
        ]
        for command in commands:
            subprocess.run(command, cwd=repository, check=True, capture_output=True)
        source_file.write_text("value = 2\n", encoding="utf-8")
        evidence = attestation.git_provenance(repository)
        self.assertTrue(evidence["repository_detected"])
        self.assertRegex(evidence["head"], r"^[0-9a-f]{40,64}$")
        self.assertTrue(evidence["tracked_dirty"])
        self.assertEqual(evidence["dirty_tracked_paths"], ["model.py"])
        self.assertEqual(evidence["source_tracked_file_count"], 1)
        self.assertRegex(
            evidence["head_committer_timestamp_utc"], r"\+00:00$"
        )


if __name__ == "__main__":
    unittest.main()
