import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_registry_lexicon_negative_control import (  # noqa: E402
    CHALLENGES,
    DEFAULT_ARCHIVE,
    DEFAULT_OUTPUT,
    DEFAULT_RULE,
    build_artifact,
    evaluate_challenges,
    render_json,
)


class RegistryLexiconNegativeControlTests(unittest.TestCase):
    def test_committed_artifact_has_honest_claim_boundary(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(artifact["status"], "PASS_TESTED_VARIANTS_ONLY")
        self.assertEqual(artifact["source"]["source_rows_automatically_regex_scanned"], 5022)
        self.assertEqual(artifact["source"]["manual_row_adjudications_performed_by_this_artifact"], 0)
        self.assertEqual(artifact["summary"]["newly_unrecalled_hit_events"], 0)
        self.assertIn("does not prove lexicon completeness", artifact["claim_boundary"]["not_supported"])
        self.assertIn("not 5,022 manual reviews", artifact["execution_scope"]["main_pipeline_boundary"])
        self.assertIn("no include", artifact["execution_scope"]["ledger_boundary"])

    def test_every_frozen_challenge_is_reported_and_fail_closed_statuses_are_explicit(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        expected = {(chain_id, item["id"]) for chain_id, items in CHALLENGES.items() for item in items}
        observed = {
            (chain_id, item["id"])
            for chain_id, chain in artifact["chains"].items()
            for item in chain["challenges"]
        }
        self.assertEqual(observed, expected)
        for chain in artifact["chains"].values():
            for challenge in chain["challenges"]:
                self.assertIn(
                    challenge["main_lexicon_recall_status"],
                    {"no_source_hits", "all_source_hits_recalled"},
                )
                self.assertEqual(challenge["newly_unrecalled_records"], [])

    def test_unrecalled_tested_variant_emits_official_description(self) -> None:
        rows = [{"code": "000001", "description": "Mutton: chilled"}]
        lexicons = {"sheep": {"regex": r"\b(?:sheep|lambs?|wool)\b"}}
        challenges = {
            "sheep": (
                {
                    "id": "mutton_word",
                    "class": "product_term",
                    "regex": r"\bmutton\b",
                    "note": "synthetic regression probe",
                },
            )
        }
        result = evaluate_challenges(rows, lexicons, challenges)
        challenge = result["chains"]["sheep"]["challenges"][0]
        self.assertEqual(challenge["main_lexicon_recall_status"], "unrecalled_hits_present")
        self.assertEqual(challenge["newly_unrecalled_hits"], 1)
        self.assertEqual(
            challenge["newly_unrecalled_records"],
            [{"code": "000001", "description": "Mutton: chilled"}],
        )

    @unittest.skipUnless(DEFAULT_ARCHIVE.is_file(), "pinned BACI raw archive is not available")
    def test_canonical_artifact_reproduces_byte_for_byte_from_pinned_source(self) -> None:
        generated = render_json(build_artifact(DEFAULT_ARCHIVE, DEFAULT_RULE))
        self.assertEqual(generated, DEFAULT_OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
