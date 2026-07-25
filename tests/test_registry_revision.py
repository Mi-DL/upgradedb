import csv
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_registry_revision import DEFAULT_BACI_ZIP, build  # noqa: E402


class RegistryRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.built = build(DEFAULT_BACI_ZIP) if DEFAULT_BACI_ZIP.is_file() else None
        cls.evidence = cls.built["evidence"] if cls.built is not None else None

    def require_raw_build(self) -> None:
        if self.built is None:
            self.skipTest("pinned BACI archive is not installed in this public checkout")

    def test_full_ledger_counts_and_historical_delta(self) -> None:
        self.require_raw_build()
        self.assertEqual(len(self.built["ledger"]), 610)
        self.assertEqual(len(self.built["unique_codes"]), 588)
        summary = self.evidence["summary"]
        self.assertEqual(summary["included_codes"], 283)
        self.assertEqual(summary["excluded_codes"], 228)
        self.assertEqual(summary["out_of_stage_codes"], 99)
        self.assertEqual(summary["historical_active_codes"], 131)
        self.assertEqual(summary["historical_active_retained"], 131)
        self.assertEqual(summary["historical_active_removed"], 0)
        self.assertEqual(summary["new_active_added"], 152)
        self.assertEqual(summary["human_reviewed_records"], 0)

    def test_double_check_boundary_decisions_are_frozen(self) -> None:
        self.require_raw_build()
        expected = {
            ("sheep", "430130"): ("include", "exp_rawskin"),
            ("cotton", "580126"): ("out_of_stage", None),
            ("cotton", "580310"): ("out_of_stage", None),
            ("aluminium", "282690"): ("exclude", None),
            ("aluminium", "690320"): ("out_of_stage", None),
            ("cocoa", "843820"): ("exclude", None),
        }
        records = {
            (row["chain_id"], row["code"]): (row["decision"], row["stage"])
            for row in self.built["ledger"]
        }
        for key, value in expected.items():
            self.assertEqual(records[key], value, key)

    def test_observable_and_full_ledger_counts_match_final_rule_artifact(self) -> None:
        rule = json.loads(
            (ROOT / "chains/evidence/registry_candidate_recall_rule.json").read_text(
                encoding="utf-8"
            )
        )
        observable = rule["observable_candidate_counts"]
        self.assertEqual(
            observable["status"],
            "observed_and_reproduced_after_ai_assisted_semantic_double_check",
        )
        self.assertIn("576", observable["scope"])
        self.assertEqual(observable["include"], 283)
        self.assertEqual(observable["exclude"], 194)
        self.assertEqual(observable["out_of_stage"], 99)
        self.assertEqual(observable["human_review_status"], "not_performed")

        full = rule["full_ledger_counts"]
        self.assertIn("610", full["scope"])
        self.assertEqual(full["decision_records"], 610)
        self.assertEqual(full["unique_hs6"], 588)
        self.assertEqual(full["observable_candidate_records"], 576)
        self.assertEqual(full["legacy_only_records"], 34)
        self.assertEqual(full["include"], 283)
        self.assertEqual(full["exclude"], 228)
        self.assertEqual(full["out_of_stage"], 99)
        self.assertEqual(full["human_review_status"], "not_performed")
        for counts in full["per_chain"].values():
            self.assertEqual(
                counts["decision_records"],
                counts["include"] + counts["exclude"] + counts["out_of_stage"],
            )

    def test_stage_wording_discloses_basket_and_form_scope(self) -> None:
        self.require_raw_build()
        evidence = self.evidence["chains"]

        nickel = evidence["nickel"]
        for code in ("740323", "740722", "740822", "740940", "741122"):
            record = next(row for row in nickel["decisions"] if row["code"] == code)
            self.assertEqual(record["decision"], "include")
            self.assertIn("copper-nickel", record["stage_fit"]["canonical_definition"])
            self.assertIn("copper-nickel", record["rationale"])

        cotton = evidence["cotton"]
        for code in ("550953", "550962", "550992", "551030"):
            record = next(row for row in cotton["decisions"] if row["code"] == code)
            self.assertEqual((record["decision"], record["stage"]), ("include", "exp_cottonyarn"))
            self.assertIn("cross-material blends", record["stage_fit"]["canonical_definition"])

        sheep_stages = evidence["sheep"]["stage_definitions"]
        self.assertIn("excludes crude wool grease", sheep_stages["exp_woolgrease"]["canonical_definition"])
        self.assertIn("excludes carded-wool fibre", sheep_stages["exp_wooltop"]["canonical_definition"])

    def test_recall_rule_records_frozen_current_revision_application(self) -> None:
        rule = json.loads(
            (ROOT / "chains/evidence/registry_candidate_recall_rule.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            rule["status"], "frozen_and_applied_to_current_registry_revision"
        )
        self.assertIn("has now been applied", rule["chronology_note"])

    def test_ai_semantic_check_is_not_independent_replication(self) -> None:
        self.require_raw_build()
        review_type = self.built["receipt"]["semantic_double_check"]["review_type"]
        self.assertIn("Separate partitioned AI-assisted", review_type)
        self.assertIn("not independent replication", review_type)
        self.assertIn("not human review", review_type)

    def test_committed_csv_has_exactly_610_unique_chain_code_rows(self) -> None:
        path = ROOT / "chains/evidence/registry_full_audit_ledger.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 610)
        self.assertEqual(len({(row["chain_id"], row["code"]) for row in rows}), 610)


if __name__ == "__main__":
    unittest.main()
