import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import registry_human_review_receipt as review  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RegistryHumanReviewReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _payload(self) -> dict:
        evidence = json.loads(
            (ROOT / "chains/evidence/registry_evidence.json").read_text(encoding="utf-8")
        )
        scan = json.loads(
            (ROOT / "chains/evidence/registry_full_scan_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        projection = review.construct_projection(ROOT)
        construct_hash = review.construct_sha256(projection)
        changes: list[dict] = []
        freeze = json.loads(review.DEFAULT_FREEZE.read_text(encoding="utf-8"))
        sample = json.loads(review.DEFAULT_SAMPLE.read_text(encoding="utf-8"))
        sampling_binding = {
            "path": review.DEFAULT_SAMPLE.relative_to(review.ROOT).as_posix(),
            "sha256": sha256(review.DEFAULT_SAMPLE),
            "plan_id": sample["plan_id"],
            "record_ids_sha256": sample["sample"]["record_ids_sha256"],
        }
        return {
            "schema_version": review.SCHEMA_VERSION,
            "status": review.STATUS_SAMPLED_COMPLETE,
            "audit_id": freeze["audit_id"],
            "benchmark_identity": {
                "benchmark_version": scan["benchmark_identity"]["benchmark_version"],
                "registry_snapshot": scan["benchmark_identity"]["data_snapshot"],
                "rule_id": evidence["rule_id"],
                "source_dictionary_member_sha256": evidence["source"][
                    "source_metadata_member_sha256"
                ],
            },
            "pre_review_freeze": {
                "path": review.DEFAULT_FREEZE.relative_to(review.ROOT).as_posix(),
                "sha256": review.EXPECTED_FREEZE_SHA256,
            },
            "public_inputs_sha256": review.current_public_input_hashes(ROOT),
            "private_record": {
                "workbook_sha256": "1" * 64,
                "normalized_review_sha256": "2" * 64,
                "adjudication_sha256": None,
            },
            "scope": {
                "decision_frame_records": 610,
                "sampled_decision_records": 212,
                "sampled_unique_hs6": sample["sample"]["unique_hs6"],
                "unique_hs6": 588,
                "stage_definition_records": 53,
            },
            "sampling_plan": sampling_binding,
            "completion": {
                "sampled_decision_records_complete": 212,
                "sampled_decision_records_not_started": 0,
                "sampled_decision_records_incomplete": 0,
                "stage_definitions_complete": 53,
                "stage_definitions_not_started": 0,
                "stage_definitions_incomplete": 0,
                "reviewer_count": 2,
                "row_outcome_blind_declarations_yes": 265,
                "audit_outcome_access_declaration_present": True,
                "reviewer_change_count": 0,
                "reviewer_uncertain_count": 0,
                "adjudication_required_count": 0,
                "adjudication_complete_count": 0,
                "unresolved_count": 0,
                "documentation_only_resolution_count": 0,
            },
            "outcome_blindness": {
                "declared": True,
                "instrument_forbidden_content_scan": "PASS",
                "prohibited_inputs": list(review.PROHIBITED_REVIEW_INPUTS),
                "claim_limit": review.OUTCOME_BLIND_CLAIM_LIMIT,
            },
            "construct": {
                "projection_schema": review.CONSTRUCT_SCHEMA,
                "fields": list(review.CONSTRUCT_FIELDS),
                "baseline_sha256": construct_hash,
                "reviewed_sha256": construct_hash,
                "accepted_changes": changes,
                "accepted_changes_sha256": hashlib.sha256(
                    review._canonical_json_bytes(changes)
                ).hexdigest(),
                "accepted_change_count": 0,
            },
            "disposition": {
                "kind": review.NO_CONSTRUCT_CHANGE,
                "release_eligible": True,
                "registry_dependent_rerun_required": False,
                "required_new_registry_id": None,
                "required_new_benchmark_version": None,
            },
            "review_completed_date": "2026-07-19",
            "receipt_issued_date": "2026-07-19",
        }

    def _write(self, payload: dict, name: str = "receipt.json") -> Path:
        path = self.temp / name
        path.write_bytes(review._canonical_json_bytes(payload))
        return path

    def _complete_protocol(self, receipt: Path, path: Path | None = None) -> Path:
        protocol = json.loads(
            (ROOT / "chains/evidence/registry_curation_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        protocol["schema_version"] = "upgrade-bench/registry-curation-protocol/4"
        protocol["human_review_receipt_file"] = (
            "chains/evidence/registry_human_review_receipt.json"
        )
        protocol["human_review_receipt_sha256"] = sha256(receipt)
        protocol["curation"]["completion_status"] = "sampled_complete"
        protocol["curation"]["retained_row_level_review_record"] = True
        protocol["curation"]["curator_count"] = 2
        protocol["curation"]["independent_second_review"] = True
        protocol["curation"]["inter_annotator_agreement_available"] = True
        protocol["curation"]["scope"] = (
            "Completed outcome-blind human validation used two independent reviewers on the "
            "frozen stratified probability sample of 212 decision records from the 610-record "
            "machine-assigned frame and the census of all 53 stage definitions. The retained "
            "private row-level record is bound by the public receipt; the other 398 decisions "
            "are not claimed as human-reviewed."
        )
        protocol["quality_controls"]["completed_human_code_reviews"] = 212
        protocol["quality_controls"]["completed_human_stage_definition_reviews"] = 53
        protocol["known_limitation"] = (
            "The completed sampled validation has two independent reviewers and available "
            "inter-annotator agreement; 398 machine-assigned decisions were not individually "
            "human-verified, and finite negative controls support but do not prove lexicon "
            "completeness."
        )
        path = path or self.temp / "protocol.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol), encoding="utf-8")
        return path

    def test_canonical_no_change_receipt_validates(self) -> None:
        receipt = self._write(self._payload())
        observed = review.validate_receipt(receipt, root=ROOT, require_release_eligible=True)
        self.assertEqual(observed["scope"]["decision_frame_records"], 610)
        self.assertEqual(observed["scope"]["sampled_decision_records"], 212)
        self.assertTrue(observed["disposition"]["release_eligible"])

    def test_missing_noncanonical_duplicate_and_extra_key_receipts_fail(self) -> None:
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "missing"):
            review.validate_receipt(self.temp / "missing.json", root=ROOT)

        payload = self._payload()
        noncanonical = self.temp / "noncanonical.json"
        noncanonical.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "not canonical"):
            review.validate_receipt(noncanonical, root=ROOT)

        duplicate = self.temp / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":"x","schema_version":"y"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "duplicate JSON key"):
            review.validate_receipt(duplicate, root=ROOT)

        payload["unexpected"] = True
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "extra"):
            review.validate_receipt(self._write(payload, "extra.json"), root=ROOT)

    def test_stale_input_incomplete_review_and_false_blindness_fail(self) -> None:
        payload = self._payload()
        payload["public_inputs_sha256"]["chains/sheep.json"] = "0" * 64
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "hash mismatch"):
            review.validate_receipt(self._write(payload, "stale.json"), root=ROOT)

        payload = self._payload()
        payload["completion"]["sampled_decision_records_complete"] = 211
        payload["completion"]["sampled_decision_records_incomplete"] = 1
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "all sampled decision records"):
            review.validate_receipt(self._write(payload, "partial.json"), root=ROOT)

        payload = self._payload()
        payload["outcome_blindness"]["declared"] = False
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "must be declared"):
            review.validate_receipt(self._write(payload, "unblind.json"), root=ROOT)

    def test_public_receipt_retains_reviewer_change_count_field_name(self) -> None:
        payload = self._payload()
        payload["completion"]["reviewer_no_count"] = payload["completion"].pop(
            "reviewer_change_count"
        )
        with self.assertRaisesRegex(
            review.HumanReviewReceiptError, "completion keys changed"
        ):
            review._validate_completion(payload, 212, 53)

    def test_construct_change_is_valid_audit_evidence_but_blocks_release(self) -> None:
        payload = self._payload()
        change = {
            "record_id": "CODE-sheep-010420",
            "kind": "code_decision",
            "old": {"decision": "exclude", "stage": None},
            "new": {"decision": "include", "stage": "exp_live"},
        }
        changes = [change]
        reviewed, _ = review._apply_accepted_changes(
            review.construct_projection(ROOT), changes
        )
        payload["construct"].update(
            {
                "accepted_changes": changes,
                "accepted_changes_sha256": hashlib.sha256(
                    review._canonical_json_bytes(changes)
                ).hexdigest(),
                "accepted_change_count": 1,
                "reviewed_sha256": review.construct_sha256(reviewed),
            }
        )
        payload["completion"].update(
            {
                "reviewer_change_count": 1,
                "adjudication_required_count": 1,
                "adjudication_complete_count": 1,
            }
        )
        payload["private_record"]["adjudication_sha256"] = "3" * 64
        payload["disposition"] = {
            "kind": review.CONSTRUCT_CHANGE_REQUIRED,
            "release_eligible": False,
            "registry_dependent_rerun_required": True,
            "required_new_registry_id": "oa-full-dictionary-hs92-test-r2",
            "required_new_benchmark_version": "2.2-test",
        }
        receipt = self._write(payload, "construct-change.json")
        observed = review.validate_receipt(receipt, root=ROOT)
        self.assertEqual(
            observed["disposition"]["kind"], review.CONSTRUCT_CHANGE_REQUIRED
        )
        # The public schema retains this legacy field name; it counts private
        # reviewer verdict=No rows that propose a correction.
        self.assertEqual(observed["completion"]["reviewer_change_count"], 1)
        with self.assertRaises(review.HumanReviewReleaseBlocked):
            review.validate_receipt(receipt, root=ROOT, require_release_eligible=True)

        forged = copy.deepcopy(payload)
        forged["disposition"].update(
            {
                "kind": review.NO_CONSTRUCT_CHANGE,
                "release_eligible": True,
                "registry_dependent_rerun_required": False,
                "required_new_registry_id": None,
                "required_new_benchmark_version": None,
            }
        )
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "no-change"):
            review.validate_receipt(self._write(forged, "forged.json"), root=ROOT)

    def test_accepted_code_change_outside_frozen_sample_fails(self) -> None:
        sampled = {
            row["record_id"]
            for row in json.loads(review.DEFAULT_SAMPLE.read_text(encoding="utf-8"))[
                "selected_records"
            ]
        }
        baseline = review.construct_projection(ROOT)
        source = next(
            row
            for row in baseline["decisions"]
            if row["record_id"] not in sampled and row["decision"] == "exclude"
        )
        change = {
            "record_id": source["record_id"],
            "kind": "code_decision",
            "old": {"decision": source["decision"], "stage": source["stage"]},
            "new": {"decision": "out_of_stage", "stage": None},
        }
        payload = self._payload()
        payload["construct"].update(
            {
                "accepted_changes": [change],
                "accepted_changes_sha256": hashlib.sha256(
                    review._canonical_json_bytes([change])
                ).hexdigest(),
                "accepted_change_count": 1,
            }
        )
        payload["completion"].update(
            {
                "reviewer_change_count": 1,
                "adjudication_required_count": 1,
                "adjudication_complete_count": 1,
            }
        )
        payload["private_record"]["adjudication_sha256"] = "3" * 64
        with self.assertRaisesRegex(
            review.HumanReviewReceiptError, "outside the frozen human-validation sample"
        ):
            review.validate_receipt(self._write(payload, "unsampled-change.json"), root=ROOT)

    def test_relation_drift_is_part_of_construct_even_if_rehashed_as_input(self) -> None:
        fixture = self.temp / "fixture"
        for relative in sorted(set(review.PUBLIC_INPUT_PATHS) | set(review.FREEZE_REVIEW_INPUT_PATHS)):
            source = ROOT / relative
            target = fixture / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        freeze = fixture / review.DEFAULT_FREEZE.relative_to(review.ROOT)
        freeze.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(review.DEFAULT_FREEZE, freeze)
        payload = self._payload()
        sheep_path = fixture / "chains/sheep.json"
        sheep = json.loads(sheep_path.read_text(encoding="utf-8"))
        sheep["named_sources"]["MEAT_SRC"].append("020410")
        sheep_path.write_text(json.dumps(sheep, indent=2) + "\n", encoding="utf-8")
        payload["public_inputs_sha256"] = review.current_public_input_hashes(fixture)
        receipt = self._write(payload, "relation-drift.json")
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "pre-review input drift|frozen baseline"):
            review.validate_receipt(receipt, root=fixture)

    def test_release_gate_requires_complete_protocol_bound_to_receipt(self) -> None:
        fixture = self.temp / "release-root"
        for relative in sorted(set(review.PUBLIC_INPUT_PATHS) | set(review.FREEZE_REVIEW_INPUT_PATHS)):
            source = ROOT / relative
            target = fixture / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        freeze = fixture / review.DEFAULT_FREEZE.relative_to(review.ROOT)
        freeze.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(review.DEFAULT_FREEZE, freeze)
        receipt = fixture / "chains/evidence/registry_human_review_receipt.json"
        receipt.write_bytes(review._canonical_json_bytes(self._payload()))
        protocol = fixture / "chains/evidence/registry_curation_protocol.json"
        shutil.copy2(ROOT / "chains/evidence/registry_curation_protocol.json", protocol)
        pending = json.loads(protocol.read_text(encoding="utf-8"))
        pending.pop("human_review_receipt_file")
        pending.pop("human_review_receipt_sha256")
        pending["curation"].update(
            {
                "curator_count": 1,
                "independent_second_review": False,
                "inter_annotator_agreement_available": False,
                "completion_status": "pending",
                "retained_row_level_review_record": False,
                "scope": (
                    "The 610-record frame has a frozen 212-record sample and 53 stage "
                    "definitions; no completed human validation is claimed."
                ),
            }
        )
        pending["quality_controls"]["completed_human_code_reviews"] = 0
        pending["quality_controls"]["completed_human_stage_definition_reviews"] = 0
        pending["known_limitation"] = (
            "Validation remains pending; 398 unsampled decisions are not individually "
            "human-verified."
        )
        protocol.write_text(json.dumps(pending), encoding="utf-8")
        with self.assertRaisesRegex(review.HumanReviewReleaseBlocked, "remains pending"):
            review.verify_release_gate(fixture)

        self._complete_protocol(receipt, protocol)
        observed = review.verify_release_gate(fixture)
        self.assertEqual(observed["audit_id"], self._payload()["audit_id"])

    def test_adjudication_cannot_disappear_without_construct_or_nonconstruct_resolution(self) -> None:
        payload = self._payload()
        payload["completion"].update(
            {
                "reviewer_uncertain_count": 1,
                "adjudication_required_count": 1,
                "adjudication_complete_count": 1,
            }
        )
        payload["private_record"]["adjudication_sha256"] = "4" * 64
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "partition every"):
            review.validate_receipt(self._write(payload, "dropped-adjudication.json"), root=ROOT)

    def test_receipt_cannot_self_rebaseline_with_a_replacement_freeze(self) -> None:
        fixture = self.temp / "fake-freeze-root"
        for relative in sorted(set(review.PUBLIC_INPUT_PATHS) | set(review.FREEZE_REVIEW_INPUT_PATHS)):
            source = ROOT / relative
            target = fixture / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        canonical = fixture / review.DEFAULT_FREEZE.relative_to(review.ROOT)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        forged = json.loads(review.DEFAULT_FREEZE.read_text(encoding="utf-8"))
        forged["construct"]["sha256"] = "f" * 64
        canonical.write_bytes(review._canonical_json_bytes(forged))
        payload = self._payload()
        payload["pre_review_freeze"]["sha256"] = sha256(canonical)
        receipt = self._write(payload, "self-rebaselined.json")
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "hard-pinned"):
            review.validate_receipt(receipt, root=fixture)

    def test_reviewed_private_draft_promotion_is_explicit_and_no_overwrite(self) -> None:
        fixture = self.temp / "promotion-root"
        for relative in sorted(set(review.PUBLIC_INPUT_PATHS) | set(review.FREEZE_REVIEW_INPUT_PATHS)):
            source = ROOT / relative
            target = fixture / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        freeze = fixture / review.DEFAULT_FREEZE.relative_to(review.ROOT)
        freeze.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(review.DEFAULT_FREEZE, freeze)
        private = fixture / "private/inspected"
        private.mkdir(parents=True)
        draft = private / "receipt.draft.json"
        draft.write_bytes(review._canonical_json_bytes(self._payload()))
        with self.assertRaisesRegex(review.HumanReviewReceiptError, "confirmation"):
            review.promote_reviewed_draft(draft, root=fixture, confirmation="")
        installed = review.promote_reviewed_draft(
            draft, root=fixture, confirmation=review.PROMOTION_CONFIRMATION
        )
        self.assertEqual(installed.read_bytes(), draft.read_bytes())
        with self.assertRaises(FileExistsError):
            review.promote_reviewed_draft(
                draft, root=fixture, confirmation=review.PROMOTION_CONFIRMATION
            )


if __name__ == "__main__":
    unittest.main()
