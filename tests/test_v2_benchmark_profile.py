from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_v2_benchmark_profile as profile  # noqa: E402


FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(profile._canonical_json_bytes(payload))


class UltraReceiptFixture:
    def __init__(self, root: Path) -> None:
        self.formal = root / "formal"
        self.receipts = root / "receipts"
        self.formal.mkdir()
        self.receipts.mkdir()
        self.run_id = "ultra-4g-zero-shot-fixed-20260718-r4"
        self.attempt_id = self.run_id + "-attempt3"
        self.trust_sha256 = "f" * 64
        self.slot_plan = [
            {
                "slot_id": "slot0",
                "host": "private-alpha",
                "device": "cuda:0",
                "ordered_chains": ["cotton"],
            },
            {
                "slot_id": "slot1",
                "host": "private-alpha",
                "device": "cuda:1",
                "ordered_chains": ["aluminium"],
            },
            {
                "slot_id": "slot2",
                "host": "private-beta",
                "device": "cuda:0",
                "ordered_chains": ["sheep", "cocoa"],
            },
            {
                "slot_id": "slot3",
                "host": "private-beta",
                "device": "cuda:1",
                "ordered_chains": ["nickel", "oilseed-soy"],
            },
        ]
        self._build()

    @staticmethod
    def _score_ref(relative: str, file_hash: str) -> dict[str, object]:
        return {
            "path": "/private/formal/" + relative,
            "sha256": file_hash,
            "rows": 2,
            "identity_sha256": "1" * 64,
            "score_vector_sha256": "2" * 64,
            "column": "ultra_score",
        }

    def _build(self) -> None:
        for relative in profile.ULTRA_PRIVATE_FILES:
            path = self.formal / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".csv":
                path.write_text("id,ultra_score\n1,0.5\n", encoding="utf-8", newline="\n")

        manifest = {
            "schema_version": "upgrade-bench-v2/ultra-formal-freeze/2",
            "protocol": profile.ULTRA_PROTOCOL,
            "status": "frozen_before_target_scoring",
            "run_id": self.run_id,
            "main_target_labels_accessed": False,
        }
        _write_canonical_json(self.formal / "frozen_manifest.json", manifest)
        manifest_sha256 = profile._sha256(self.formal / "frozen_manifest.json")
        scoring_start = {
            "schema_version": "upgrade-bench-v2/ultra-formal-score-start/2",
            "protocol": profile.ULTRA_PROTOCOL,
            "run_id": self.run_id,
            "manifest_sha256": manifest_sha256,
            "started_at_utc": "2026-07-18T01:00:00+00:00",
            "policy": {"labels_locked": True},
        }
        _write_canonical_json(self.formal / "SCORING_STARTED.json", scoring_start)
        scoring_started_sha256 = profile._sha256(self.formal / "SCORING_STARTED.json")

        component_hashes: dict[str, str] = {}
        components: dict[str, dict[str, object]] = {}
        for chain in profile.CHAINS:
            score_refs = {}
            for source in ("A", "B"):
                relative = f"components/{chain}/scores_{source}.csv"
                score_refs[source] = self._score_ref(
                    relative, profile._sha256(self.formal / relative)
                )
            repeat_refs = None
            repeatability: dict[str, object] = {
                "required": False,
                "sentinel_chain": "sheep",
            }
            repeat_seconds: float | None = None
            if chain == "sheep":
                repeat_refs = {}
                for source in ("A", "B"):
                    relative = f"components/sheep/scores_{source}_repeat.csv"
                    repeat_refs[source] = self._score_ref(
                        relative, profile._sha256(self.formal / relative)
                    )
                repeatability = {"required": True, "numeric_allclose": True}
                repeat_seconds = 0.75
            component: dict[str, object] = {
                "schema_version": "upgrade-bench-v2/ultra-formal-score-component/2",
                "protocol": profile.ULTRA_PROTOCOL,
                "status": "complete_label_blind_scores",
                "run_id": self.run_id,
                "chain": chain,
                "created_at_utc": "2026-07-18T01:10:00+00:00",
                "manifest_sha256": manifest_sha256,
                "config_sha256": "3" * 64,
                "checkpoint_sha256": "4" * 64,
                "graph_sha256": "5" * 64,
                "candidate_precommit": {},
                "main_target_labels_accessed": False,
                "main_label_derived_columns_accessed": False,
                "training_or_fine_tuning_performed": False,
                "selection_performed": False,
                "combined_unique_lane_rows": 2,
                "scores": score_refs,
                "repeat_scores": repeat_refs,
                "repeatability": repeatability,
                "native_backend": {
                    "device": "cuda:0",
                    "device_name": "NVIDIA L4",
                },
                "peak_memory_allocated_bytes": 1024,
                "elapsed_seconds": 2.0,
                "scoring_seconds": {
                    "primary_run1": 1.25,
                    "repeat_run2": repeat_seconds,
                },
                "source_sha256": {},
            }
            relative = f"components/{chain}/component.json"
            _write_canonical_json(self.formal / relative, component)
            component_hashes[chain] = profile._sha256(self.formal / relative)
            components[chain] = component

        native_runtime_sha256 = "6" * 64
        sealed_components = []
        for chain in profile.CHAINS:
            component = components[chain]
            repeat = component["repeat_scores"]
            sealed_components.append(
                {
                    "chain": chain,
                    "path": f"/private/formal/components/{chain}/component.json",
                    "sha256": component_hashes[chain],
                    "A_score_sha256": component["scores"]["A"]["sha256"],
                    "B_score_sha256": component["scores"]["B"]["sha256"],
                    "A_repeat_score_sha256": None if repeat is None else repeat["A"]["sha256"],
                    "B_repeat_score_sha256": None if repeat is None else repeat["B"]["sha256"],
                    "repeatability": component["repeatability"],
                    "native_runtime_sha256": native_runtime_sha256,
                }
            )
        score_seal = {
            "schema_version": "upgrade-bench-v2/ultra-formal-score-seal/3",
            "protocol": profile.ULTRA_PROTOCOL,
            "status": "all_six_chains_scored_labels_unlocked",
            "run_id": self.run_id,
            "created_at_utc": "2026-07-18T01:20:00+00:00",
            "manifest_sha256": manifest_sha256,
            "scoring_started_sha256": scoring_started_sha256,
            "component_count": 6,
            "components": sealed_components,
            "native_runtime_sha256": native_runtime_sha256,
            "repeatability_contract": {"sentinel_chain": "sheep"},
            "sentinel_repeat_verified_before_label_unlock": True,
            "main_target_labels_accessed_before_seal": False,
            "unlock_policy": {"component_count": 6},
        }
        _write_canonical_json(self.formal / "SCORES_COMPLETE.json", score_seal)
        score_seal_sha256 = profile._sha256(self.formal / "SCORES_COMPLETE.json")
        evaluation_start = {
            "schema_version": "upgrade-bench-v2/ultra-formal-evaluation-start/3",
            "protocol": profile.ULTRA_PROTOCOL,
            "run_id": self.run_id,
            "score_seal_sha256": score_seal_sha256,
            "scoring_started_sha256": scoring_started_sha256,
            "started_at_utc": "2026-07-18T01:21:00+00:00",
            "ordering_attestation": "all six components preceded label access",
        }
        _write_canonical_json(
            self.formal / "LABEL_EVALUATION_STARTED.json", evaluation_start
        )
        for chain in profile.CHAINS:
            metric = {
                "schema_version": "upgrade-bench-v2/ultra-formal-chain-metrics/3",
                "protocol": profile.ULTRA_PROTOCOL,
                "status": "complete",
                "run_id": self.run_id,
                "chain": chain,
                "score_seal_sha256": score_seal_sha256,
                "scoring_started_sha256": scoring_started_sha256,
            }
            _write_canonical_json(self.formal / f"metrics/metrics_{chain}.json", metric)
        evaluation = {
            "schema_version": "upgrade-bench-v2/ultra-formal-evaluation/3",
            "protocol": profile.ULTRA_PROTOCOL,
            "status": "complete",
            "run_id": self.run_id,
            "manifest_sha256": manifest_sha256,
            "score_seal_sha256": score_seal_sha256,
            "scoring_started_sha256": scoring_started_sha256,
        }
        _write_canonical_json(self.formal / "evaluation.json", evaluation)

        terminals = []
        for plan in self.slot_plan:
            terminal = {
                "schema_version": "upgrade-bench-v2/ultra-r4-slot-terminal/1",
                "status": "success",
                "slot_id": plan["slot_id"],
                "host": plan["host"],
                "physical_device": plan["device"],
                "ordered_chains": plan["ordered_chains"],
                "components": [
                    {"chain": chain, "component_sha256": component_hashes[chain]}
                    for chain in plan["ordered_chains"]
                ],
                "manifest_sha256": manifest_sha256,
                "scoring_started_sha256": scoring_started_sha256,
                "trust_manifest_sha256": self.trust_sha256,
                "completed_at_utc": "2026-07-18T01:19:00+00:00",
            }
            name = f"slot-terminal-{plan['slot_id']}.json"
            _write_canonical_json(self.receipts / name, terminal)
            terminals.append(
                {"slot_id": plan["slot_id"], "sha256": profile._sha256(self.receipts / name)}
            )
        slot_set = {
            "schema_version": "upgrade-bench-v2/ultra-r4-slot-set/1",
            "status": "complete",
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "slot_plan": self.slot_plan,
            "terminals": terminals,
            "manifest_sha256": manifest_sha256,
            "scoring_started_sha256": scoring_started_sha256,
            "trust_manifest_sha256": self.trust_sha256,
            "created_at_utc": "2026-07-18T01:19:30+00:00",
        }
        _write_canonical_json(self.receipts / "slot-set.json", slot_set)
        actual_files, actual_dirs = profile._inventory_tree(self.formal)
        self.asserted_private_dirs = actual_dirs
        extraction = {
            "schema_version": "upgrade-bench-v2/ultra-r4-extraction/1",
            "status": "complete",
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "trust_manifest_sha256": self.trust_sha256,
            "source_root": "/private/formal-source",
            "destination_root": "/private/formal-extraction",
            "file_count": 31,
            "directory_count": 8,
            "inventory_sha256": profile._private_inventory_digest(actual_files),
            "files": actual_files,
            "manifest_sha256": actual_files["frozen_manifest.json"],
            "scoring_started_sha256": actual_files["SCORING_STARTED.json"],
            "score_seal_sha256": actual_files["SCORES_COMPLETE.json"],
            "evaluation_start_sha256": actual_files["LABEL_EVALUATION_STARTED.json"],
            "evaluation_sha256": actual_files["evaluation.json"],
            "slot_set_sha256": profile._sha256(self.receipts / "slot-set.json"),
            "created_at_utc": "2026-07-18T01:30:00+00:00",
        }
        _write_canonical_json(self.receipts / "extraction.json", extraction)

    def load_receipt(self, name: str) -> dict[str, object]:
        return json.loads((self.receipts / name).read_text(encoding="utf-8"))

    def write_receipt(self, name: str, payload: object) -> None:
        _write_canonical_json(self.receipts / name, payload)

    def refresh_terminal_and_extraction_hashes(self, slot_id: str) -> None:
        slot_set = self.load_receipt("slot-set.json")
        terminal_name = f"slot-terminal-{slot_id}.json"
        for row in slot_set["terminals"]:
            if row["slot_id"] == slot_id:
                row["sha256"] = profile._sha256(self.receipts / terminal_name)
        self.write_receipt("slot-set.json", slot_set)
        extraction = self.load_receipt("extraction.json")
        extraction["slot_set_sha256"] = profile._sha256(self.receipts / "slot-set.json")
        self.write_receipt("extraction.json", extraction)


class UltraReceiptEvidenceTests(unittest.TestCase):
    def test_profile_schema_and_evidence_role_are_version_two(self) -> None:
        self.assertEqual(profile.SCHEMA, "upgrade-bench-v2/benchmark-profile/2")
        self.assertIn("ultra_receipt_set_sha256", profile.FORMAL_EVIDENCE_ROLES)

    def test_build_requires_private_ultra_receipt_directory(self) -> None:
        args = profile.argparse.Namespace(
            freeze_manifest=Path("freeze.json"),
            claims_dir=Path("claims"),
            gpu_inventory=Path("gpus.json"),
            ultra_dir=Path("ultra"),
            ultra_receipts_dir=None,
        )
        with self.assertRaisesRegex(
            profile.ProfileError, "ULTRA orchestration receipt directory"
        ):
            profile.build_profile(args)

    def test_exact_receipt_set_reports_four_gpus_and_stable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = UltraReceiptFixture(Path(directory))
            count, digest = profile._validate_ultra_receipts(
                fixture.receipts, fixture.formal
            )
            self.assertEqual(count, 4)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(
                digest,
                profile._canonical_digest(
                    {
                        path.name: profile._sha256(path)
                        for path in fixture.receipts.iterdir()
                    }
                ),
            )
            sanitized = json.dumps(
                {
                    "physical_gpu_count": count,
                    "ultra_receipt_set_sha256": digest,
                },
                sort_keys=True,
            )
            self.assertNotIn("private-alpha", sanitized)
            self.assertNotIn("private-beta", sanitized)
            self.assertNotIn("cuda:", sanitized)

    def test_duplicate_host_physical_device_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = UltraReceiptFixture(Path(directory))
            slot_set = fixture.load_receipt("slot-set.json")
            slot_set["slot_plan"][1]["device"] = "cuda:0"
            terminal = fixture.load_receipt("slot-terminal-slot1.json")
            terminal["physical_device"] = "cuda:0"
            fixture.write_receipt("slot-terminal-slot1.json", terminal)
            fixture.write_receipt("slot-set.json", slot_set)
            fixture.refresh_terminal_and_extraction_hashes("slot1")
            with self.assertRaises(profile.ProfileError):
                profile._validate_ultra_receipts(fixture.receipts, fixture.formal)

    def test_terminal_component_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = UltraReceiptFixture(Path(directory))
            terminal = fixture.load_receipt("slot-terminal-slot0.json")
            terminal["components"][0]["component_sha256"] = "0" * 64
            fixture.write_receipt("slot-terminal-slot0.json", terminal)
            fixture.refresh_terminal_and_extraction_hashes("slot0")
            with self.assertRaises(profile.ProfileError):
                profile._validate_ultra_receipts(fixture.receipts, fixture.formal)

    def test_extraction_score_seal_cross_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = UltraReceiptFixture(Path(directory))
            extraction = fixture.load_receipt("extraction.json")
            extraction["score_seal_sha256"] = "0" * 64
            fixture.write_receipt("extraction.json", extraction)
            with self.assertRaises(profile.ProfileError):
                profile._validate_ultra_receipts(fixture.receipts, fixture.formal)

    def test_extra_receipt_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = UltraReceiptFixture(Path(directory))
            (fixture.receipts / "private-hosts.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(profile.ProfileError):
                profile._validate_ultra_receipts(fixture.receipts, fixture.formal)


class BenchmarkProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(profile.DEFAULT_PROFILE.read_text(encoding="utf-8"))

    def test_committed_profile_and_tex_verify(self) -> None:
        profile.verify_outputs(mode="repository")
        if FULL_PAYLOAD_TESTS_ENABLED:
            profile.verify_outputs(mode="full")

    def test_graph_and_sample_counts_reconcile(self) -> None:
        self.assertEqual(profile.BENCHMARK_VERSION, "2.1-dev")
        rows = {row["chain"]: row for row in self.payload["chains"]}
        self.assertEqual(set(rows), set(profile.CHAINS))
        for chain, row in rows.items():
            graph = row["graph"]
            hs6, hs4, hs2, products = profile._registry_products(
                ROOT / "chains" / f"{chain}.json"
            )
            self.assertEqual(
                (
                    graph["hs6_products"],
                    graph["hs4_products"],
                    graph["hs2_products"],
                    graph["products"],
                ),
                (hs6, hs4, hs2, products),
            )
            self.assertEqual(graph["nodes"], graph["countries"] + graph["products"])
        for field in (
            "b1_candidate_entries",
            "b1_positive_entries",
            "b2_positive_entry_groups",
            "b2_candidate_lanes",
            "b2_positive_lanes",
        ):
            self.assertEqual(
                self.payload["totals"][field],
                sum(row["samples"][field] for row in rows.values()),
            )

    def test_compute_claim_is_phase_specific_and_reconciles(self) -> None:
        compute = self.payload["compute"]
        main = compute["fitted_references"]["main_refit_and_evaluation"]
        self.assertEqual(main["duration_semantics"], "summed one-GPU worker wall-clock")
        self.assertEqual(
            main["total_worker_seconds"],
            main["pykeen_global_graph"]["worker_seconds"]
            + main["nbfnet"]["worker_seconds"],
        )
        self.assertAlmostEqual(
            main["total_worker_hours"],
            round(main["total_worker_seconds"] / 3600.0, 2),
            places=2,
        )
        self.assertFalse(
            compute["fitted_references"]["historical_selection"]["wall_time_retained"]
        )
        self.assertFalse(
            compute["ultra_zero_shot"]["benchmark_training_or_fine_tuning"]
        )

    def test_generated_tex_defines_every_profile_macro_used_by_paper(self) -> None:
        paper_sources = (
            ROOT / "paper/body.tex",
            ROOT / "paper/appendix.tex",
            ROOT / "paper/abstract.tex",
        )
        if not all(path.is_file() for path in paper_sources):
            self.skipTest("requires the maintainer paper-source checkout")
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in paper_sources
        )
        used = set(re.findall(r"\\(VTwoProfile[A-Za-z0-9]+)", source))
        generated = profile.DEFAULT_TEX.read_text(encoding="utf-8")
        defined = set(
            re.findall(r"\\newcommand\{\\(VTwoProfile[A-Za-z0-9]+)\}", generated)
        )
        self.assertTrue(used)
        self.assertLessEqual(used, defined)

    def test_mutated_b2_group_count_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["chains"][0]["samples"]["b2_positive_entry_groups"] += 1
        with self.assertRaises(profile.ProfileError):
            profile.validate_profile(mutated, mode="full")

    def test_mutated_graph_reconciliation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["chains"][0]["graph"]["nodes"] += 1
        with self.assertRaises(profile.ProfileError):
            profile.validate_profile(mutated, mode="full")

    def test_malformed_formal_receipt_digest_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["provenance"]["formal_evidence_sha256"][
            "early_graph_freeze_sha256"
        ] = "not-a-sha256"
        with self.assertRaises(profile.ProfileError):
            profile.validate_profile(mutated, mode="full")

    def test_mutated_compute_accounting_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["compute"]["fitted_references"]["main_refit_and_evaluation"][
            "total_worker_seconds"
        ] += 1
        with self.assertRaises(profile.ProfileError):
            profile.validate_profile(mutated, mode="full")

    def test_stale_tex_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            tex_path = root / "profile.tex"
            profile_path.write_bytes(profile.DEFAULT_PROFILE.read_bytes())
            tex_path.write_text("% stale\n", encoding="utf-8")
            with self.assertRaises(profile.ProfileError):
                profile.verify_outputs(profile_path, tex_path, mode="full")


if __name__ == "__main__":
    unittest.main()
