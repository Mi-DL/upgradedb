import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_chain_registry import (  # noqa: E402
    EVIDENCE,
    EXPECTED_CHAINS,
    JSON_OUTPUT,
    MARKDOWN_OUTPUT,
    METADATA,
    RegistryAuditError,
    _render_json,
    audit_registry,
    render_markdown,
)


class StrictRegistryAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = audit_registry()

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _copy_chains(self, destination: Path) -> Path:
        chains = destination / "chains"
        chains.mkdir()
        for chain_id in EXPECTED_CHAINS:
            shutil.copy2(ROOT / "chains" / f"{chain_id}.json", chains / f"{chain_id}.json")
        return chains

    def test_canonical_audit_passes_and_marks_old_outputs_stale(self) -> None:
        self.assertEqual(self.report["status"], "PASS")
        self.assertEqual(
            self.report["summary"],
            {
                "chain_count": 6,
                "active_stages": 53,
                "included_codes": 283,
                "excluded_codes": 228,
                "out_of_stage_codes": 99,
                "reviewed_codes": 610,
                "decision_records": 610,
                "unique_reviewed_hs6": 588,
                "observable_candidate_records": 576,
                "legacy_only_records": 34,
                "reassigned_included_codes": 19,
                "human_reviewed_records": 0,
                "historical_active_codes": 131,
                "historical_active_retained": 131,
                "historical_active_removed": 0,
                "new_active_added": 152,
            },
        )
        self.assertFalse(self.report["scientific_implications"]["existing_v2_numbers_valid"])
        self.assertTrue(self.report["scientific_implications"]["candidate_rebuild_required"])
        self.assertEqual(self.report["chains"]["sheep"]["active_codes"], 25)
        self.assertEqual(self.report["chains"]["cotton"]["active_codes"], 194)
        self.assertEqual(self.report["chains"]["nickel"]["active_codes"], 24)
        self.assertEqual(self.report["chains"]["oilseed-soy"]["out_of_stage_codes"], 1)
        self.assertEqual(self.report["checks"]["canonical_stage_definitions_complete"], "PASS")
        self.assertEqual(
            self.report["checks"]["per_code_stage_fit_supported_excluded_or_out_of_stage"],
            "PASS",
        )

    def test_human_facing_scope_disclosures_remain_unambiguous(self) -> None:
        sheep_description = self.report["chains"]["sheep"]["display_description"]
        self.assertIn("unresolved-material", sheep_description)
        self.assertIn("commodity-explicit blends remain eligible", sheep_description)
        rendered = render_markdown(self.report)
        self.assertIn("not a new or live UNSD API attestation", rendered)

    def test_high_risk_codes_have_exact_semantic_stages_and_supported_fit(self) -> None:
        expected = {
            ("sheep", "510521"): "exp_wooltop",
            ("sheep", "510529"): "exp_wooltop",
            ("sheep", "510620"): "exp_woolyarn",
            ("sheep", "510610"): "exp_woolyarn",
            ("sheep", "430130"): "exp_rawskin",
            ("cotton", "520210"): "exp_cottonwaste",
            ("cotton", "520300"): "exp_cottonprepared",
            ("cotton", "550953"): "exp_cottonyarn",
            ("cotton", "620520"): "exp_cottonapparel_woven",
            ("aluminium", "260600"): "exp_aluminium_ore",
            ("aluminium", "281830"): "exp_aluminium_hydroxide",
            ("nickel", "750120"): "exp_nickel_intermediate",
            ("nickel", "283324"): "exp_nickel_salts",
            ("nickel", "740323"): "exp_unwrought",
            ("cocoa", "180620"): "exp_cocoa_prep_bulk",
            ("cocoa", "180690"): "exp_cocoa_prep_other",
            ("oilseed-soy", "150790"): "exp_soyoil_noncrude",
            ("oilseed-soy", "120810"): "exp_soyflour_meal",
        }
        for (chain_id, code), stage in expected.items():
            decision = next(
                row for row in self.report["chains"][chain_id]["decisions"] if row["code"] == code
            )
            self.assertEqual(decision["stage"], stage)
            self.assertEqual(decision["stage_fit"]["status"], "supported")
            self.assertEqual(decision["stage_fit"]["evidence"], decision["description"])
            self.assertTrue(decision["stage_fit"]["canonical_definition"])
            self.assertTrue(decision["stage_fit"]["rationale"])

    def test_generated_machine_and_markdown_reports_are_current(self) -> None:
        self.assertEqual(JSON_OUTPUT.read_text(encoding="utf-8"), _render_json(self.report))
        self.assertEqual(MARKDOWN_OUTPUT.read_text(encoding="utf-8"), render_markdown(self.report))
        machine = JSON_OUTPUT.read_text(encoding="utf-8")
        markdown = MARKDOWN_OUTPUT.read_text(encoding="utf-8")
        # Build sensitive sentinels at runtime so the public privacy audit can
        # scan this test source without mistaking its denylist fixtures for a
        # leaked real path/account.
        for forbidden in (
            "/" + "home" + "/",
            "\\" + "Users" + "\\",
            "C:" + "\\" + "Users",
            "sli" + "6",
        ):
            self.assertNotIn(forbidden, machine)
            self.assertNotIn(forbidden, markdown)

    def test_missing_include_evidence_fails_closed(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        evidence["chains"]["cocoa"]["decisions"] = [
            row
            for row in evidence["chains"]["cocoa"]["decisions"]
            if row["code"] != "180100"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write_json(path, evidence)
            with self.assertRaisesRegex(RegistryAuditError, "active/include evidence mismatch"):
                audit_registry(evidence_path=path)

    def test_empty_source_or_rationale_fails_closed(self) -> None:
        base = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        for field in ("source_version", "rationale"):
            evidence = copy.deepcopy(base)
            evidence["chains"]["aluminium"]["decisions"][0][field] = ""
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "evidence.json"
                self._write_json(path, evidence)
                with self.assertRaisesRegex(RegistryAuditError, f"empty {field}"):
                    audit_registry(evidence_path=path)

    def test_reactivating_an_excluded_code_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chains = self._copy_chains(Path(tmp))
            sheep_path = chains / "sheep.json"
            sheep = json.loads(sheep_path.read_text(encoding="utf-8"))
            sheep["stages"]["exp_live"].append("010420")
            self._write_json(sheep_path, sheep)
            with self.assertRaisesRegex(RegistryAuditError, "both active and excluded|excluded HS6 remains active"):
                audit_registry(chains_dir=chains)

    def test_excluded_code_in_derived_map_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chains = self._copy_chains(Path(tmp))
            sheep_path = chains / "sheep.json"
            sheep = json.loads(sheep_path.read_text(encoding="utf-8"))
            sheep["derived_from"]["exp_wooltop"].append("510910")
            self._write_json(sheep_path, sheep)
            with self.assertRaisesRegex(RegistryAuditError, "excluded HS6 remain|inactive HS6"):
                audit_registry(chains_dir=chains)

    def test_invalid_upstream_stage_reference_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chains = self._copy_chains(Path(tmp))
            cotton_path = chains / "cotton.json"
            cotton = json.loads(cotton_path.read_text(encoding="utf-8"))
            cotton["upstream_map"]["exp_cottonyarn"].append("exp_missing")
            self._write_json(cotton_path, cotton)
            with self.assertRaisesRegex(RegistryAuditError, "references missing stages"):
                audit_registry(chains_dir=chains)

    def test_private_source_path_fails_closed(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        evidence["source"]["selected_metadata_file"] = (
            "C:" + "\\" + "Users" + "\\" + "person" + "\\" + "private.csv"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write_json(path, evidence)
            with self.assertRaisesRegex(RegistryAuditError, "private/host-specific path"):
                audit_registry(evidence_path=path)

    def test_selected_description_drift_fails_closed(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        evidence["chains"]["oilseed-soy"]["decisions"][0]["description"] = "renamed broad basket"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write_json(path, evidence)
            with self.assertRaisesRegex(RegistryAuditError, "differs from selected BACI metadata"):
                audit_registry(evidence_path=path, metadata_path=METADATA)

    def test_missing_stage_definition_fails_closed(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        del evidence["chains"]["sheep"]["stage_definitions"]["exp_wooltop"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write_json(path, evidence)
            with self.assertRaisesRegex(RegistryAuditError, "canonical stage definitions must exactly match"):
                audit_registry(evidence_path=path)

    def test_included_code_requires_supported_stage_fit(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        decision = next(
            row for row in evidence["chains"]["sheep"]["decisions"] if row["code"] == "510521"
        )
        decision["stage_fit"]["status"] = "unsupported"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write_json(path, evidence)
            with self.assertRaisesRegex(RegistryAuditError, "must have supported stage_fit"):
                audit_registry(evidence_path=path)

    def test_stage_fit_evidence_must_be_official_description(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        decision = next(
            row for row in evidence["chains"]["nickel"]["decisions"] if row["code"] == "283324"
        )
        decision["stage_fit"]["evidence"] = "battery material"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write_json(path, evidence)
            with self.assertRaisesRegex(RegistryAuditError, "evidence must equal the official description"):
                audit_registry(evidence_path=path)

    def test_high_risk_stage_assignment_regression_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chains = self._copy_chains(Path(tmp))
            sheep_path = chains / "sheep.json"
            sheep = json.loads(sheep_path.read_text(encoding="utf-8"))
            sheep["stages"]["exp_wooltop"].remove("510521")
            sheep["stages"]["exp_woolyarn"].append("510521")
            self._write_json(sheep_path, sheep)
            with self.assertRaisesRegex(RegistryAuditError, "high-risk stage-semantic assignments regressed"):
                audit_registry(chains_dir=chains)


if __name__ == "__main__":
    unittest.main()
