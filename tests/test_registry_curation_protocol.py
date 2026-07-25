import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_registry_curation_protocol import (  # noqa: E402
    CurationProtocolError,
    validate_protocol,
)


PROTOCOL = ROOT / "chains" / "evidence" / "registry_curation_protocol.json"
EVIDENCE = ROOT / "chains" / "evidence" / "registry_evidence.json"


class RegistryCurationProtocolTests(unittest.TestCase):
    def test_canonical_protocol_is_hash_bound_and_explicit(self) -> None:
        payload = validate_protocol(PROTOCOL, EVIDENCE)
        self.assertIn("588/588", payload["quality_controls"]["pinned_baci_dictionary_membership"])
        self.assertEqual(payload["quality_controls"]["source_rows_automatically_regex_scanned"], 5022)
        self.assertEqual(payload["quality_controls"]["source_rows_manually_reviewed"], 0)
        self.assertEqual(payload["curation"]["completion_status"], "sampled_complete")
        self.assertEqual(payload["curation"]["curator_count"], 2)
        self.assertTrue(payload["curation"]["independent_second_review"])
        self.assertTrue(payload["curation"]["inter_annotator_agreement_available"])
        self.assertEqual(payload["quality_controls"]["completed_human_code_reviews"], 212)
        self.assertEqual(
            payload["quality_controls"]["completed_human_stage_definition_reviews"], 53
        )
        self.assertEqual(payload["quality_controls"]["planned_human_code_reviews"], 212)
        self.assertEqual(
            payload["quality_controls"]["planned_human_stage_definition_reviews"], 53
        )
        self.assertEqual(payload["quality_controls"]["unsampled_human_code_records"], 398)
        self.assertEqual(
            payload["curation"]["decision_validation_sampling"]["sampled_decision_records"],
            212,
        )

    def test_completed_protocol_cannot_hide_independent_review(self) -> None:
        payload = copy.deepcopy(validate_protocol(PROTOCOL, EVIDENCE))
        payload["curation"]["independent_second_review"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                CurationProtocolError, "without independent second review"
            ):
                validate_protocol(path, EVIDENCE)

    def test_stale_evidence_hash_fails(self) -> None:
        payload = copy.deepcopy(validate_protocol(PROTOCOL, EVIDENCE))
        payload["registry_evidence_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CurationProtocolError, "evidence hash mismatch"):
                validate_protocol(path, EVIDENCE)

    def test_completed_protocol_cannot_report_incomplete_sample(self) -> None:
        payload = copy.deepcopy(validate_protocol(PROTOCOL, EVIDENCE))
        payload["quality_controls"]["completed_human_code_reviews"] = 0
        payload["quality_controls"]["completed_human_stage_definition_reviews"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CurationProtocolError, "inconsistent with curation status"):
                validate_protocol(path, EVIDENCE)

    def test_complete_schema_rejects_noncanonical_receipt_path(self) -> None:
        payload = copy.deepcopy(validate_protocol(PROTOCOL, EVIDENCE))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CurationProtocolError, "canonical review receipt path"):
                validate_protocol(
                    path,
                    EVIDENCE,
                    receipt_path=Path(tmp) / "missing-receipt.json",
                )

    def test_sampling_disclosure_is_hash_bound_and_exact(self) -> None:
        payload = copy.deepcopy(validate_protocol(PROTOCOL, EVIDENCE))
        payload["curation"]["decision_validation_sampling"][
            "sampled_decision_records"
        ] = 153
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CurationProtocolError, "sampling disclosure"):
                validate_protocol(path, EVIDENCE)

    def test_pending_schema_does_not_require_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = copy.deepcopy(validate_protocol(PROTOCOL, EVIDENCE))
            payload.pop("human_review_receipt_file")
            payload.pop("human_review_receipt_sha256")
            payload["curation"].update(
                {
                    "curator_count": 1,
                    "independent_second_review": False,
                    "inter_annotator_agreement_available": False,
                    "completion_status": "pending",
                    "retained_row_level_review_record": False,
                    "scope": (
                        "The machine-assigned population contains 610 records. The frozen sample "
                        "contains 212 decisions and all 53 stage definitions; no completed human "
                        "validation is claimed."
                    ),
                }
            )
            payload["quality_controls"]["completed_human_code_reviews"] = 0
            payload["quality_controls"]["completed_human_stage_definition_reviews"] = 0
            payload["known_limitation"] = (
                "Validation remains pending; 398 unsampled decisions are not individually "
                "human-verified."
            )
            path = Path(tmp) / "pending-protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            payload = validate_protocol(
                path,
                EVIDENCE,
                receipt_path=Path(tmp) / "noncanonical-missing-receipt.json",
            )
        self.assertEqual(payload["curation"]["completion_status"], "pending")


if __name__ == "__main__":
    unittest.main()
