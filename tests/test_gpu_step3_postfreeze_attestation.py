import json
import shutil
import tempfile
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]

from tools import build_gpu_step3_postfreeze_attestation as attestation


def strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, str):
        yield value


class GpuStep3PostfreezeAttestationTest(unittest.TestCase):
    def test_full_current_tree_and_published_artifact_verify(self):
        if not (ROOT / "data/processed_v2/candidates_aluminium.csv").is_file():
            self.skipTest(
                "full attestation inventory is distributed as an external payload"
            )
        payload = attestation.verify_output(root=ROOT, require_full_inventory=True)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["comparison"]["entry_count"], 50)
        self.assertEqual(payload["comparison"]["exact_byte_match_count"], 44)
        self.assertEqual(payload["comparison"]["allowed_changed_file_count"], 6)
        self.assertEqual(payload["candidate_inputs"]["count"], 24)
        self.assertTrue(payload["candidate_inputs"]["all_exact_bytes_unchanged"])
        self.assertTrue(
            payload["machine_semantic_projection"][
                "membership_stage_relation_unchanged"
            ]
        )
        self.assertEqual(payload["machine_semantic_projection"]["chain_count"], 6)
        self.assertEqual(
            payload["machine_semantic_projection"]["active_assignment_count"], 283
        )
        self.assertTrue(
            payload["machine_semantic_projection"][
                "evidence_active_membership_stage_matches_chain_files"
            ]
        )
        self.assertEqual(
            len(payload["machine_semantic_projection"]["chains"]), 6
        )
        self.assertEqual(
            payload["machine_semantic_projection"]["frozen_sha256"],
            payload["machine_semantic_projection"]["current_sha256"],
        )

    def test_public_mode_is_deterministic_and_contains_no_private_path(self):
        payload = attestation.verify_output(root=ROOT, require_full_inventory=False)
        known_pointers = {
            pointer
            for proof in payload["changed_json_proofs"]
            for pointer in proof["allowed_json_pointers"]
        }
        for value in strings(payload):
            if value in known_pointers:
                continue
            self.assertFalse(PurePosixPath(value).is_absolute(), value)
            self.assertFalse(PureWindowsPath(value).is_absolute(), value)
            lowered = value.lower()
            for token in (
                "mars" + "10",
                "mars" + "29",
                "sli" + "6@",
                "/ho" + "me/",
                "c:\\" + "users\\",
            ):
                self.assertNotIn(token, lowered)

    def test_exact_changed_file_and_pointer_allowlists_are_frozen(self):
        payload = attestation.verify_output(root=ROOT, require_full_inventory=False)
        self.assertEqual(
            set(payload["comparison"]["allowed_changed_files"]),
            {
                "chains/sheep.json",
                "chains/evidence/registry_evidence.json",
                "docs/registry_audit.json",
                "results_v2/metrics/b1_candidate_coverage.json",
                "src/universe.py",
                "src/v2_gpu_rolling.py",
            },
        )
        proof_by_path = {row["path"]: row for row in payload["changed_json_proofs"]}
        self.assertEqual(
            len(
                proof_by_path["chains/evidence/registry_evidence.json"][
                    "allowed_json_pointers"
                ]
            ),
            238,
        )
        self.assertEqual(
            len(
                proof_by_path["docs/registry_audit.json"][
                    "allowed_json_pointers"
                ]
            ),
            239,
        )
        for relative in (
            "chains/evidence/registry_evidence.json",
            "docs/registry_audit.json",
        ):
            for pointer in proof_by_path[relative]["allowed_json_pointers"]:
                self.assertNotRegex(pointer, r"/(code|decision|stage)$")
        self.assertEqual(
            set(proof_by_path["results_v2/metrics/b1_candidate_coverage.json"]["allowed_json_pointers"]),
            {
                "/generated_at",
                "/source/cache_manifest_sha256",
                "/registry/audit/sha256",
                "/registry/evidence/sha256",
                "/protocol_sha256/registry_loader/sha256",
            },
        )
        source_proofs = payload["changed_source_proofs"]
        self.assertEqual(len(source_proofs), 2)
        source_by_path = {row["path"]: row for row in source_proofs}
        universe_proof = source_by_path["src/universe.py"]
        self.assertEqual(
            [row["id"] for row in universe_proof["allowed_text_changes"]],
            [row["id"] for row in attestation.UNIVERSE_TEXT_CHANGES],
        )
        self.assertTrue(universe_proof["reverse_patch_reconstructs_frozen_bytes"])
        self.assertTrue(universe_proof["non_allowlisted_source_content_unchanged"])
        runner_proof = source_by_path["src/v2_gpu_rolling.py"]
        self.assertEqual(
            [row["id"] for row in runner_proof["allowed_text_changes"]],
            [row["id"] for row in attestation.RUNNER_TEXT_CHANGES],
        )
        self.assertTrue(runner_proof["reverse_patch_reconstructs_frozen_bytes"])
        self.assertTrue(runner_proof["non_allowlisted_source_content_unchanged"])
        self.assertTrue(
            payload["coverage_invariants"]["snapshots_and_all_quantities_unchanged"]
        )

    def test_changed_current_json_or_artifact_bytes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in attestation.ALLOWED_CHANGES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            for chain_id in attestation.CHAIN_IDS:
                target = root / f"chains/{chain_id}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / f"chains/{chain_id}.json", target)
            artifact = root / attestation.ARTIFACT_ROLE
            artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attestation.DEFAULT_OUTPUT, artifact)

            attestation.verify_output(
                artifact, root=root, require_full_inventory=False
            )
            sheep = root / "chains/sheep.json"
            payload = json.loads(sheep.read_text(encoding="utf-8"))
            payload["description"] += " tampered"
            sheep.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(attestation.AttestationError, "current byte hash"):
                attestation.verify_output(
                    artifact, root=root, require_full_inventory=False
                )

            shutil.copy2(ROOT / "chains/sheep.json", sheep)
            artifact.write_bytes(artifact.read_bytes() + b"\n")
            with self.assertRaisesRegex(attestation.AttestationError, "stale or non-deterministic"):
                attestation.verify_output(
                    artifact, root=root, require_full_inventory=False
                )

    def test_runner_reverse_patch_reconstructs_exact_step3_bytes(self):
        payload = attestation.verify_output(root=ROOT, require_full_inventory=False)
        proof = next(
            row
            for row in payload["changed_source_proofs"]
            if row["path"] == "src/v2_gpu_rolling.py"
        )
        self.assertEqual(
            proof["frozen_sha256"],
            "c821c4027b199c2a115ba6abe9dfd2361bdd70a61cf812e75d014c7e786b6645",
        )
        self.assertEqual(len(proof["allowed_text_changes"]), 10)
        self.assertTrue(proof["reverse_patch_reconstructs_frozen_bytes"])
        self.assertTrue(proof["non_allowlisted_source_content_unchanged"])
        self.assertIn("frozen run was executed under the later CLI/config gates", payload["claim_boundary"]["not_supported"])

    def test_gpu_summary_binding_is_exact(self):
        binding = attestation.summary_binding()
        summary = {"run_id": attestation.RUN_ID, "post_freeze_semantic_attestation": binding}
        attestation.verify_summary_binding(summary, root=ROOT)
        summary["post_freeze_semantic_attestation"] = {**binding, "sha256": "0" * 64}
        with self.assertRaisesRegex(attestation.AttestationError, "binding is stale"):
            attestation.verify_summary_binding(summary, root=ROOT)


if __name__ == "__main__":
    unittest.main()
