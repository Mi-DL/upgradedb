#!/usr/bin/env python3
"""Verify the hash-bound registry curation disclosure without changing the registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import build_registry_human_validation_sample as sample_plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "chains" / "evidence" / "registry_curation_protocol.json"
DEFAULT_EVIDENCE = ROOT / "chains" / "evidence" / "registry_evidence.json"
DEFAULT_RECEIPT = ROOT / "chains" / "evidence" / "registry_human_review_receipt.json"


class CurationProtocolError(ValueError):
    """The curation disclosure is stale, incomplete, or misleading."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurationProtocolError(f"cannot read valid {role} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CurationProtocolError(f"{role} must be a JSON object")
    return payload


def _exact_keys(value: dict[str, Any], expected: set[str], role: str) -> None:
    observed = set(value)
    if observed != expected:
        raise CurationProtocolError(
            f"{role} keys changed: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def validate_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL,
    evidence_path: Path = DEFAULT_EVIDENCE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    payload = _load_object(protocol_path, "curation protocol")
    evidence = _load_object(evidence_path, "registry evidence")
    schema = payload.get("schema_version")
    if schema == "upgrade-bench/registry-curation-protocol/4":
        expected_top_keys = {
            "schema_version",
            "issued_date",
            "registry_evidence_file",
            "registry_evidence_sha256",
            "curation",
            "chain_selection",
            "quality_controls",
            "revision_policy",
            "known_limitation",
        }
    else:
        raise CurationProtocolError("unexpected curation protocol schema")
    if payload.get("curation", {}).get("completion_status") == "sampled_complete":
        expected_top_keys.update({"human_review_receipt_file", "human_review_receipt_sha256"})
    _exact_keys(
        payload,
        expected_top_keys,
        "curation protocol",
    )
    if payload["registry_evidence_file"] != "chains/evidence/registry_evidence.json":
        raise CurationProtocolError("curation protocol must bind the canonical evidence path")
    if payload["registry_evidence_sha256"] != _sha256(evidence_path):
        raise CurationProtocolError("curation protocol evidence hash mismatch")

    curation = payload.get("curation")
    if not isinstance(curation, dict):
        raise CurationProtocolError("curation must be an object")
    _exact_keys(
        curation,
        {
            "curator_count",
            "independent_second_review",
            "inter_annotator_agreement_available",
            "completion_status",
            "retained_row_level_review_record",
            "scope",
            "decision_validation_sampling",
            "decision_basis",
            "workflow",
        },
        "curation",
    )
    if isinstance(curation["curator_count"], bool) or not isinstance(
        curation["curator_count"], int
    ) or curation["curator_count"] < 1:
        raise CurationProtocolError("curator_count must be a positive integer")
    if not isinstance(curation["independent_second_review"], bool):
        raise CurationProtocolError("independent_second_review must be boolean")
    if not isinstance(curation["inter_annotator_agreement_available"], bool):
        raise CurationProtocolError("inter_annotator_agreement_available must be boolean")
    if curation["inter_annotator_agreement_available"] and not curation[
        "independent_second_review"
    ]:
        raise CurationProtocolError(
            "inter-annotator agreement cannot be available without independent second review"
        )
    completion_status = curation["completion_status"]
    if completion_status == "pending":
        if curation["curator_count"] != 1:
            raise CurationProtocolError(
                "pending protocol must disclose the planned single-curator state"
            )
        if curation["independent_second_review"] is not False:
            raise CurationProtocolError(
                "pending protocol must not imply completed independent review"
            )
        if curation["inter_annotator_agreement_available"] is not False:
            raise CurationProtocolError(
                "pending protocol must not imply available inter-annotator agreement"
            )
        if curation["retained_row_level_review_record"] is not False:
            raise CurationProtocolError(
                "pending protocol must not imply a retained completed review record"
            )
    elif completion_status == "sampled_complete":
        if curation["curator_count"] != 2:
            raise CurationProtocolError(
                "sampled-complete protocol must disclose the two completed reviewers"
            )
        if curation["independent_second_review"] is not True:
            raise CurationProtocolError(
                "sampled-complete protocol must disclose independent second review"
            )
        if curation["inter_annotator_agreement_available"] is not True:
            raise CurationProtocolError(
                "sampled-complete protocol must disclose available inter-annotator agreement"
            )
        if curation["retained_row_level_review_record"] is not True:
            raise CurationProtocolError("sampled-complete protocol must retain the row-level review record")
        if payload.get("human_review_receipt_file") != (
            "chains/evidence/registry_human_review_receipt.json"
        ):
            raise CurationProtocolError("sampled-complete protocol must bind the canonical review receipt")
        review_root = evidence_path.resolve().parents[2]
        canonical_receipt = (
            review_root / "chains" / "evidence" / "registry_human_review_receipt.json"
        )
        if receipt_path.resolve(strict=False) != canonical_receipt.resolve(strict=False):
            raise CurationProtocolError(
                "sampled-complete protocol verification requires the canonical review receipt path"
            )
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise CurationProtocolError("sampled-complete protocol review receipt is missing or unsafe")
        if payload.get("human_review_receipt_sha256") != _sha256(receipt_path):
            raise CurationProtocolError("sampled-complete protocol review receipt hash mismatch")
        try:
            from registry_human_review_receipt import validate_receipt

            receipt = validate_receipt(
                receipt_path,
                root=review_root,
                require_release_eligible=True,
            )
        except (OSError, ValueError) as exc:
            raise CurationProtocolError(
                f"sampled-complete protocol review receipt is invalid: {exc}"
            ) from exc
        if curation["curator_count"] != receipt["completion"]["reviewer_count"]:
            raise CurationProtocolError("protocol curator count differs from review receipt")
    else:
        raise CurationProtocolError("completion_status must be pending or sampled_complete")

    review_root = evidence_path.resolve().parents[2]
    sample_path = review_root / "chains/evidence/registry_human_validation_sample.json"
    try:
        sample = sample_plan.load_plan(sample_path, root=review_root)
    except (OSError, ValueError) as exc:
        raise CurationProtocolError(f"cannot validate decision-validation sample: {exc}") from exc
    sampling = curation.get("decision_validation_sampling")
    if not isinstance(sampling, dict):
        raise CurationProtocolError("decision_validation_sampling must be an object")
    _exact_keys(
        sampling,
        {
            "plan_file",
            "plan_sha256",
            "plan_id",
            "sampling_frame_records",
            "sampled_decision_records",
            "unsampled_decision_records",
            "stage_definition_census_records",
            "design",
            "estimator",
            "completion_claim",
        },
        "decision_validation_sampling",
    )
    expected_sampling = {
        "plan_file": "chains/evidence/registry_human_validation_sample.json",
        "plan_sha256": _sha256(sample_path),
        "plan_id": sample["plan_id"],
        "sampling_frame_records": 610,
        "sampled_decision_records": 212,
        "unsampled_decision_records": 398,
        "stage_definition_census_records": 53,
        "design": "outcome-blind stratified probability sample without replacement with declared certainty units",
        "estimator": "design-weighted Horvitz-Thompson discrepancy estimate using stored inclusion probabilities",
        "completion_claim": "sampled human validation; not full-ledger human review",
    }
    if sampling != expected_sampling:
        raise CurationProtocolError("decision-validation sampling disclosure is stale or altered")

    scope_lower = curation["scope"].lower()
    if not all(marker in scope_lower for marker in ("610", "212", "53", "sample")):
        raise CurationProtocolError("curation scope must distinguish frame, sample, and stage census")
    limitation_lower = str(payload.get("known_limitation", "")).lower()
    if completion_status == "pending":
        if "no completed human" not in scope_lower:
            raise CurationProtocolError("pending protocol must explicitly deny completed human review")
    else:
        stale_markers = ("no completed human", "validation remains pending", "review process and reports no")
        if any(marker in scope_lower or marker in limitation_lower for marker in stale_markers):
            raise CurationProtocolError("sampled-complete protocol retains stale pending-review wording")
        if "completed" not in scope_lower:
            raise CurationProtocolError("sampled-complete protocol scope must state completed validation")
    if not isinstance(curation["decision_basis"], list) or len(curation["decision_basis"]) < 3:
        raise CurationProtocolError("decision basis is incomplete")
    if not isinstance(curation["workflow"], list) or len(curation["workflow"]) < 4:
        raise CurationProtocolError("curation workflow is incomplete")

    controls = payload.get("quality_controls")
    if not isinstance(controls, dict):
        raise CurationProtocolError("quality_controls must be an object")
    expected_counts = {
        "reviewed_codes": 610,
        "unique_reviewed_hs6": 588,
        "observable_candidate_records": 576,
        "legacy_only_records": 34,
        "included_codes": 283,
        "excluded_codes": 228,
        "out_of_stage_codes": 99,
        "historical_active_codes": 131,
        "historical_active_retained": 131,
        "historical_active_removed": 0,
        "new_active_added": 152,
        "reassigned_included_codes": 19,
    }
    evidence_summary = evidence.get("summary")
    if not isinstance(evidence_summary, dict):
        raise CurationProtocolError("registry evidence summary is missing")
    for key, expected in expected_counts.items():
        if controls.get(key) != expected or evidence_summary.get(key) != expected:
            raise CurationProtocolError(f"curation control mismatch: {key}")
    if "not a count of completed human row reviews" not in controls.get(
        "reviewed_codes_semantics", ""
    ):
        raise CurationProtocolError("legacy reviewed_codes semantics must remain explicit")
    if controls.get("planned_human_code_reviews") != 212:
        raise CurationProtocolError("planned sampled human-code review count changed")
    if controls.get("planned_human_stage_definition_reviews") != 53:
        raise CurationProtocolError("planned stage-definition census count changed")
    if controls.get("unsampled_human_code_records") != 398:
        raise CurationProtocolError("unsampled decision-record count changed")
    expected_completed_reviews = 0 if completion_status == "pending" else 212
    if controls.get("completed_human_code_reviews") != expected_completed_reviews:
        raise CurationProtocolError(
            "completed human code review count is inconsistent with curation status"
        )
    expected_completed_stages = 0 if completion_status == "pending" else 53
    if controls.get("completed_human_stage_definition_reviews") != expected_completed_stages:
        raise CurationProtocolError(
            "completed human stage-definition review count is inconsistent with curation status"
        )
    if controls.get("source_rows_automatically_regex_scanned") != 5022:
        raise CurationProtocolError("curation control mismatch: automated source scan")
    if controls.get("source_rows_manually_reviewed") != 0:
        raise CurationProtocolError("curation protocol must not claim full-table manual review")
    if controls.get("semantic_regression_gates") != 35:
        raise CurationProtocolError("curation control mismatch: semantic_regression_gates")
    if "588/588" not in controls.get("pinned_baci_dictionary_membership", ""):
        raise CurationProtocolError("pinned BACI membership disclosure changed")
    if not controls.get("lexicon_negative_control", "").startswith("PASS_TESTED_VARIANTS_ONLY"):
        raise CurationProtocolError("lexicon negative-control limitation is missing")
    if "do not establish independent semantic agreement" not in controls.get(
        "scope_of_automation", ""
    ):
        raise CurationProtocolError("automation limitation must remain explicit")

    selection = payload.get("chain_selection")
    if not isinstance(selection, dict):
        raise CurationProtocolError("chain_selection must be an object")
    if "not a random or representative sample" not in selection.get("sampling_design", ""):
        raise CurationProtocolError("chain sampling limitation must be explicit")
    if not isinstance(selection.get("criteria"), list) or len(selection["criteria"]) < 4:
        raise CurationProtocolError("chain selection criteria are incomplete")
    if "model performance" not in selection.get("excluded_selection_signal", ""):
        raise CurationProtocolError("chain selection must exclude formal model performance")
    if not payload.get("known_limitation") or not payload.get("revision_policy"):
        raise CurationProtocolError("limitation and revision policy must be explicit")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    payload = validate_protocol(args.protocol, args.evidence, receipt_path=args.receipt)
    print(
        "registry curation protocol OK "
        f"(curators={payload['curation']['curator_count']}; "
        f"status={payload['curation']['completion_status']}; "
        f"proposal_records={payload['quality_controls']['reviewed_codes']}; "
        f"completed_human_reviews={payload['quality_controls']['completed_human_code_reviews']})"
    )


if __name__ == "__main__":
    main()
