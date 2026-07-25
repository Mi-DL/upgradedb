#!/usr/bin/env python3
"""Verify the public, hash-bound sampled human-validation completion receipt.

The completed sampled row-level workbook is retained privately.  The public receipt
binds its exact bytes and a normalized semantic digest, while this verifier
rechecks every public construct input and the release disposition.  Receipt
validity and release eligibility are deliberately separate: a completed review
that accepts a construct change is valid audit evidence but cannot release the
current benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

import build_registry_human_validation_sample as sample_plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "chains" / "evidence" / "registry_human_review_receipt.json"
DEFAULT_PROTOCOL = ROOT / "chains" / "evidence" / "registry_curation_protocol.json"
DEFAULT_FREEZE = ROOT / "chains" / "evidence" / "registry_human_review_freeze.json"
DEFAULT_SAMPLE = ROOT / "chains" / "evidence" / "registry_human_validation_sample.json"

SCHEMA_VERSION = "upgrade-bench/registry-human-review-receipt/3"
FREEZE_SCHEMA = "upgrade-bench/registry-human-review-freeze/2"
CONSTRUCT_SCHEMA = "upgrade-bench/registry-construct-projection/1"
STATUS_SAMPLED_COMPLETE = "SAMPLED_COMPLETE"
STATUS_FROZEN_PRE_REVIEW = "FROZEN_PRE_REVIEW"
NO_CONSTRUCT_CHANGE = "NO_CONSTRUCT_CHANGE"
CONSTRUCT_CHANGE_REQUIRED = "CONSTRUCT_CHANGE_REQUIRED"

# Set only after the reviewed freeze artifact is generated.  This constant is
# deliberately a source-level trust anchor: a receipt cannot redefine its own
# pre-review baseline by replacing both the freeze and a self-reported hash.
EXPECTED_FREEZE_SHA256 = "0763f21dbd09fb950ef23e611a2225bf12681a1ac8648c1992f8e240f1fc3af4"

CHAIN_IDS = ("aluminium", "cocoa", "cotton", "nickel", "oilseed-soy", "sheep")
PUBLIC_INPUT_PATHS = (
    "chains/aluminium.json",
    "chains/cocoa.json",
    "chains/cotton.json",
    "chains/evidence/hs92_selected_product_codes.csv",
    "chains/evidence/registry_candidate_recall_rule.json",
    "chains/evidence/registry_evidence.json",
    "chains/evidence/registry_full_audit_ledger.csv",
    "chains/evidence/registry_full_scan_receipt.json",
    "chains/evidence/registry_human_validation_sample.json",
    "chains/nickel.json",
    "chains/oilseed-soy.json",
    "chains/sheep.json",
    "docs/REGISTRY_REVIEW_CODEBOOK.md",
    "tools/build_registry_human_validation_sample.py",
    "tools/prepare_registry_human_review_receipt.py",
    "tools/registry_human_review_receipt.py",
)
FREEZE_REVIEW_INPUT_PATHS = (
    "chains/aluminium.json",
    "chains/cocoa.json",
    "chains/cotton.json",
    "chains/evidence/hs92_selected_product_codes.csv",
    "chains/evidence/registry_candidate_recall_rule.json",
    "chains/evidence/registry_evidence.json",
    "chains/evidence/registry_full_audit_ledger.csv",
    "chains/evidence/registry_full_scan_receipt.json",
    "chains/evidence/registry_human_validation_sample.json",
    "chains/nickel.json",
    "chains/oilseed-soy.json",
    "chains/sheep.json",
    "docs/REGISTRY_REVIEW_CODEBOOK.md",
    "tools/build_registry_human_validation_sample.py",
    "tools/prepare_registry_human_review_receipt.py",
)
CONSTRUCT_FIELDS = (
    "decisions",
    "stage_definitions",
    "stages",
    "upstream",
    "upstream_map",
    "derived_from",
    "derived_from_hs",
    "produces",
    "form_of",
    "named_sources",
)
PROHIBITED_REVIEW_INPUTS = (
    "trade_values",
    "benchmark_labels",
    "cohort_impacts",
    "model_scores",
    "downstream_result_summaries",
)
OUTCOME_BLIND_CLAIM_LIMIT = (
    "Declaration and instrument-content checks do not prove the reviewer's external access history."
)
PROMOTION_CONFIRMATION = "PROMOTE-REGISTRY-HUMAN-REVIEW-RECEIPT"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{7,127}$")
_HS6_RE = re.compile(r"^[0-9]{6}$")


class HumanReviewReceiptError(ValueError):
    """The human-review receipt is missing, stale, malformed, or ineligible."""


class HumanReviewReleaseBlocked(HumanReviewReceiptError):
    """The receipt is valid audit evidence but does not authorize release."""


def _reject_constant(value: str) -> None:
    raise HumanReviewReceiptError(f"non-finite JSON constant is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HumanReviewReceiptError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, role: str, *, canonical: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HumanReviewReceiptError(f"cannot read valid {role} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HumanReviewReceiptError(f"{role} must be a JSON object")
    if canonical and raw != _canonical_json_bytes(payload):
        raise HumanReviewReceiptError(f"{role} is not canonical strict JSON")
    return payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise HumanReviewReceiptError(f"cannot hash required file {path}: {exc}") from exc
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], role: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise HumanReviewReceiptError(
            f"{role} keys changed: missing={sorted(expected_set - observed)!r}, "
            f"extra={sorted(observed - expected_set)!r}"
        )


def _require_hex(value: Any, role: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise HumanReviewReceiptError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise HumanReviewReceiptError(f"{role} must be a trimmed non-empty string")
    return value


def _require_nonnegative_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HumanReviewReceiptError(f"{role} must be a non-negative integer")
    return value


def _require_date(value: Any, role: str) -> date:
    if not isinstance(value, str):
        raise HumanReviewReceiptError(f"{role} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HumanReviewReceiptError(f"{role} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise HumanReviewReceiptError(f"{role} must be a canonical ISO date")
    return parsed


def _safe_source(root: Path, relative: str) -> Path:
    if (
        not relative
        or "\\" in relative
        or "\x00" in relative
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or PurePosixPath(relative).as_posix() != relative
    ):
        raise HumanReviewReceiptError(f"unsafe public input path: {relative!r}")
    root_resolved = root.resolve()
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise HumanReviewReceiptError(f"symbolic-link public input is forbidden: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise HumanReviewReceiptError(f"required public input is missing: {relative}") from exc
    if not resolved.is_relative_to(root_resolved) or not candidate.is_file():
        raise HumanReviewReceiptError(f"public input is not a safe regular file: {relative}")
    return candidate


def current_public_input_hashes(root: Path = ROOT) -> dict[str, str]:
    return {
        relative: _sha256_file(_safe_source(root, relative))
        for relative in PUBLIC_INPUT_PATHS
    }


def _normalized_string_list(value: Any, role: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise HumanReviewReceiptError(f"{role} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise HumanReviewReceiptError(f"{role} contains duplicates")
    return sorted(value)


def _normalized_mapping_of_lists(value: Any, role: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise HumanReviewReceiptError(f"{role} must be an object")
    result: dict[str, list[str]] = {}
    for key, items in value.items():
        if not isinstance(key, str) or not key:
            raise HumanReviewReceiptError(f"{role} has an invalid key")
        result[key] = _normalized_string_list(items, f"{role}[{key}]")
    return {key: result[key] for key in sorted(result)}


def construct_projection(root: Path = ROOT) -> dict[str, Any]:
    evidence = _load_json(
        _safe_source(root, "chains/evidence/registry_evidence.json"),
        "registry evidence",
    )
    evidence_chains = evidence.get("chains")
    if not isinstance(evidence_chains, dict) or set(evidence_chains) != set(CHAIN_IDS):
        raise HumanReviewReceiptError("registry evidence chain inventory changed")

    decisions: list[dict[str, Any]] = []
    stage_definitions: dict[str, dict[str, dict[str, str]]] = {}
    chains: dict[str, dict[str, Any]] = {}
    seen_records: set[str] = set()
    for chain_id in CHAIN_IDS:
        evidence_chain = evidence_chains[chain_id]
        if not isinstance(evidence_chain, dict):
            raise HumanReviewReceiptError(f"registry evidence chain is invalid: {chain_id}")
        raw_definitions = evidence_chain.get("stage_definitions")
        if not isinstance(raw_definitions, dict):
            raise HumanReviewReceiptError(f"stage definitions are missing: {chain_id}")
        normalized_definitions: dict[str, dict[str, str]] = {}
        for stage, definition in sorted(raw_definitions.items()):
            if not isinstance(stage, str) or not isinstance(definition, dict):
                raise HumanReviewReceiptError(f"invalid stage definition: {chain_id}/{stage}")
            _exact_keys(
                definition,
                {"canonical_definition", "specificity", "fit_rule"},
                f"stage definition {chain_id}/{stage}",
            )
            normalized_definitions[stage] = {
                key: _require_nonempty_string(
                    definition.get(key), f"stage definition {chain_id}/{stage}/{key}"
                )
                for key in ("canonical_definition", "specificity", "fit_rule")
            }
        stage_definitions[chain_id] = normalized_definitions

        raw_decisions = evidence_chain.get("decisions")
        if not isinstance(raw_decisions, list):
            raise HumanReviewReceiptError(f"decision list is missing: {chain_id}")
        for decision in raw_decisions:
            if not isinstance(decision, dict):
                raise HumanReviewReceiptError(f"decision entry is invalid: {chain_id}")
            code = str(decision.get("code", ""))
            if _HS6_RE.fullmatch(code) is None:
                raise HumanReviewReceiptError(f"decision HS6 is invalid: {chain_id}/{code}")
            record_id = f"CODE-{chain_id}-{code}"
            if record_id in seen_records:
                raise HumanReviewReceiptError(f"duplicate decision record: {record_id}")
            seen_records.add(record_id)
            verdict = decision.get("decision")
            stage = decision.get("stage")
            if verdict not in {"include", "exclude", "out_of_stage"}:
                raise HumanReviewReceiptError(f"decision value is invalid: {record_id}")
            if verdict == "include":
                if not isinstance(stage, str) or stage not in normalized_definitions:
                    raise HumanReviewReceiptError(f"included decision stage is invalid: {record_id}")
            elif stage is not None:
                raise HumanReviewReceiptError(f"non-included decision has a stage: {record_id}")
            decisions.append(
                {
                    "record_id": record_id,
                    "chain_id": chain_id,
                    "code": code,
                    "decision": verdict,
                    "stage": stage,
                }
            )

        registry = _load_json(
            _safe_source(root, f"chains/{chain_id}.json"),
            f"{chain_id} registry",
        )
        if registry.get("id") != chain_id:
            raise HumanReviewReceiptError(f"registry id mismatch: {chain_id}")
        stages = _normalized_mapping_of_lists(registry.get("stages"), f"{chain_id}.stages")
        if set(stages) != set(normalized_definitions):
            raise HumanReviewReceiptError(f"stage inventory differs between registry and evidence: {chain_id}")
        decision_stage_codes = {
            stage: sorted(
                item["code"]
                for item in decisions
                if item["chain_id"] == chain_id
                and item["decision"] == "include"
                and item["stage"] == stage
            )
            for stage in sorted(normalized_definitions)
        }
        if stages != decision_stage_codes:
            raise HumanReviewReceiptError(
                f"active stage membership differs from included evidence decisions: {chain_id}"
            )
        form_of = registry.get("form_of", [])
        if not isinstance(form_of, list):
            raise HumanReviewReceiptError(f"{chain_id}.form_of must be a list")
        normalized_form_of: list[list[str]] = []
        for index, pair in enumerate(form_of):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(item, str) or not item for item in pair)
            ):
                raise HumanReviewReceiptError(f"{chain_id}.form_of[{index}] is invalid")
            normalized_form_of.append(list(pair))
        if len({tuple(pair) for pair in normalized_form_of}) != len(normalized_form_of):
            raise HumanReviewReceiptError(f"{chain_id}.form_of contains duplicates")

        produces = registry.get("produces", {})
        if not isinstance(produces, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in produces.items()
        ):
            raise HumanReviewReceiptError(f"{chain_id}.produces must be a string mapping")
        chains[chain_id] = {
            "stages": stages,
            "upstream": _normalized_string_list(
                registry.get("upstream", []), f"{chain_id}.upstream"
            ),
            "upstream_map": _normalized_mapping_of_lists(
                registry.get("upstream_map", {}), f"{chain_id}.upstream_map"
            ),
            "derived_from": _normalized_mapping_of_lists(
                registry.get("derived_from", {}), f"{chain_id}.derived_from"
            ),
            "derived_from_hs": _normalized_mapping_of_lists(
                registry.get("derived_from_hs", {}), f"{chain_id}.derived_from_hs"
            ),
            "produces": {key: produces[key] for key in sorted(produces)},
            "form_of": sorted(normalized_form_of),
            "named_sources": _normalized_mapping_of_lists(
                registry.get("named_sources", {}), f"{chain_id}.named_sources"
            ),
        }

    decisions.sort(key=lambda item: item["record_id"])
    summary = evidence.get("summary")
    if not isinstance(summary, dict):
        raise HumanReviewReceiptError("registry evidence summary is missing")
    if len(decisions) != summary.get("decision_records"):
        raise HumanReviewReceiptError("construct decision count differs from evidence summary")
    return {
        "schema_version": CONSTRUCT_SCHEMA,
        "decisions": decisions,
        "stage_definitions": stage_definitions,
        "chains": chains,
    }


def construct_sha256(projection: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(projection))


def _find_decision(projection: dict[str, Any], record_id: str) -> dict[str, Any]:
    matches = [item for item in projection["decisions"] if item["record_id"] == record_id]
    if len(matches) != 1:
        raise HumanReviewReceiptError(f"accepted change references unknown decision: {record_id}")
    return matches[0]


def _apply_accepted_changes(
    baseline: Mapping[str, Any], changes: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(changes, list):
        raise HumanReviewReceiptError("construct accepted_changes must be a list")
    reviewed = copy.deepcopy(dict(baseline))
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(changes):
        if not isinstance(item, dict):
            raise HumanReviewReceiptError(f"accepted_changes[{index}] must be an object")
        _exact_keys(item, {"record_id", "kind", "old", "new"}, f"accepted_changes[{index}]")
        record_id = _require_nonempty_string(item.get("record_id"), "accepted change record_id")
        kind = item.get("kind")
        if kind not in {"code_decision", "stage_definition"}:
            raise HumanReviewReceiptError(f"accepted change kind is invalid: {kind!r}")
        identity = (kind, record_id)
        if identity in seen:
            raise HumanReviewReceiptError(f"duplicate accepted change: {kind}/{record_id}")
        seen.add(identity)
        old = item.get("old")
        new = item.get("new")
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise HumanReviewReceiptError(f"accepted change old/new must be objects: {record_id}")
        if kind == "code_decision":
            _exact_keys(old, {"decision", "stage"}, f"accepted change old {record_id}")
            _exact_keys(new, {"decision", "stage"}, f"accepted change new {record_id}")
            decision = _find_decision(reviewed, record_id)
            observed_old = {"decision": decision["decision"], "stage": decision["stage"]}
            if old != observed_old:
                raise HumanReviewReceiptError(f"accepted change old value mismatch: {record_id}")
            verdict = new.get("decision")
            stage = new.get("stage")
            chain_id = decision["chain_id"]
            code = decision["code"]
            stage_map = reviewed["chains"][chain_id]["stages"]
            if verdict not in {"include", "exclude", "out_of_stage"}:
                raise HumanReviewReceiptError(f"accepted change decision is invalid: {record_id}")
            if verdict == "include":
                if not isinstance(stage, str) or stage not in stage_map:
                    raise HumanReviewReceiptError(f"accepted include stage is invalid: {record_id}")
            elif stage is not None:
                raise HumanReviewReceiptError(f"accepted non-include stage must be null: {record_id}")
            if old == new:
                raise HumanReviewReceiptError(f"accepted code change does not change the construct: {record_id}")
            for codes in stage_map.values():
                if code in codes:
                    codes.remove(code)
            if verdict == "include":
                stage_map[stage].append(code)
                stage_map[stage].sort()
            decision["decision"] = verdict
            decision["stage"] = stage
        else:
            _exact_keys(
                old,
                {"canonical_definition", "specificity", "fit_rule"},
                f"accepted stage old {record_id}",
            )
            _exact_keys(
                new,
                {"canonical_definition", "specificity", "fit_rule"},
                f"accepted stage new {record_id}",
            )
            stage_matches = [
                (chain_id, record_id[len(f"STAGE-{chain_id}-") :])
                for chain_id in CHAIN_IDS
                if record_id.startswith(f"STAGE-{chain_id}-")
            ]
            if len(stage_matches) != 1 or not stage_matches[0][1]:
                raise HumanReviewReceiptError(f"stage record_id is invalid: {record_id}")
            chain_id, stage = stage_matches[0]
            definitions = reviewed["stage_definitions"].get(chain_id)
            if not isinstance(definitions, dict) or stage not in definitions:
                raise HumanReviewReceiptError(f"accepted change references unknown stage: {record_id}")
            definition = definitions[stage]
            observed_old = {
                "canonical_definition": definition["canonical_definition"],
                "specificity": definition["specificity"],
                "fit_rule": definition["fit_rule"],
            }
            if old != observed_old:
                raise HumanReviewReceiptError(f"accepted stage old value mismatch: {record_id}")
            for key in ("canonical_definition", "specificity", "fit_rule"):
                _require_nonempty_string(new.get(key), f"accepted stage {record_id}/{key}")
            if old == new:
                raise HumanReviewReceiptError(f"accepted stage change does not change the construct: {record_id}")
            definition.update(new)
        normalized.append({"record_id": record_id, "kind": kind, "old": old, "new": new})

    expected_order = sorted(normalized, key=lambda item: (item["record_id"], item["kind"]))
    if normalized != expected_order:
        raise HumanReviewReceiptError("accepted_changes must use canonical record_id/kind order")
    return reviewed, normalized


def _validate_identity(payload: Mapping[str, Any], root: Path) -> None:
    identity = payload.get("benchmark_identity")
    if not isinstance(identity, dict):
        raise HumanReviewReceiptError("benchmark_identity must be an object")
    _exact_keys(
        identity,
        {
            "benchmark_version",
            "registry_snapshot",
            "rule_id",
            "source_dictionary_member_sha256",
        },
        "benchmark_identity",
    )
    evidence = _load_json(
        _safe_source(root, "chains/evidence/registry_evidence.json"), "registry evidence"
    )
    recall = _load_json(
        _safe_source(root, "chains/evidence/registry_candidate_recall_rule.json"),
        "candidate recall rule",
    )
    scan = _load_json(
        _safe_source(root, "chains/evidence/registry_full_scan_receipt.json"),
        "full scan receipt",
    )
    scan_identity = scan.get("benchmark_identity")
    source = evidence.get("source")
    if not isinstance(scan_identity, dict) or not isinstance(source, dict):
        raise HumanReviewReceiptError("current registry identity evidence is incomplete")
    expected = {
        "benchmark_version": scan_identity.get("benchmark_version"),
        "registry_snapshot": scan_identity.get("data_snapshot"),
        "rule_id": evidence.get("rule_id"),
        "source_dictionary_member_sha256": source.get("source_metadata_member_sha256"),
    }
    if identity != expected or recall.get("rule_id") != identity.get("rule_id"):
        raise HumanReviewReceiptError("receipt benchmark identity differs from current registry evidence")
    _require_hex(identity.get("source_dictionary_member_sha256"), "source dictionary digest")


def _validate_public_inputs(payload: Mapping[str, Any], root: Path) -> None:
    observed = payload.get("public_inputs_sha256")
    if not isinstance(observed, dict):
        raise HumanReviewReceiptError("public_inputs_sha256 must be an object")
    _exact_keys(observed, PUBLIC_INPUT_PATHS, "public_inputs_sha256")
    expected = current_public_input_hashes(root)
    for relative in PUBLIC_INPUT_PATHS:
        _require_hex(observed.get(relative), f"public input digest {relative}")
        if observed[relative] != expected[relative]:
            raise HumanReviewReceiptError(f"public input hash mismatch: {relative}")


def _validate_scope(payload: Mapping[str, Any], root: Path) -> tuple[int, int, int, int, int]:
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise HumanReviewReceiptError("scope must be an object")
    _exact_keys(
        scope,
        {
            "decision_frame_records",
            "sampled_decision_records",
            "sampled_unique_hs6",
            "unique_hs6",
            "stage_definition_records",
        },
        "scope",
    )
    evidence = _load_json(
        _safe_source(root, "chains/evidence/registry_evidence.json"), "registry evidence"
    )
    summary = evidence.get("summary")
    chains = evidence.get("chains")
    if not isinstance(summary, dict) or not isinstance(chains, dict):
        raise HumanReviewReceiptError("registry evidence scope is incomplete")
    sample = sample_plan.load_plan(
        root / DEFAULT_SAMPLE.relative_to(ROOT), root=root
    )
    expected = (
        summary.get("decision_records"),
        sample["sample"]["decision_records"],
        sample["sample"]["unique_hs6"],
        summary.get("unique_reviewed_hs6"),
        sum(
            len(chain.get("stage_definitions", {}))
            for chain in chains.values()
            if isinstance(chain, dict)
        ),
    )
    observed = (
        scope.get("decision_frame_records"),
        scope.get("sampled_decision_records"),
        scope.get("sampled_unique_hs6"),
        scope.get("unique_hs6"),
        scope.get("stage_definition_records"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in observed):
        raise HumanReviewReceiptError("scope counts must be integers")
    if observed != expected:
        raise HumanReviewReceiptError(
            f"receipt scope differs from current evidence: expected={expected}, observed={observed}"
        )
    return observed


def _validate_sampling_plan_binding(payload: Mapping[str, Any], root: Path) -> dict[str, str]:
    binding = payload.get("sampling_plan")
    if not isinstance(binding, dict):
        raise HumanReviewReceiptError("sampling_plan must be an object")
    _exact_keys(
        binding,
        {"path", "sha256", "plan_id", "record_ids_sha256"},
        "sampling_plan",
    )
    expected_path = DEFAULT_SAMPLE.relative_to(ROOT).as_posix()
    if binding.get("path") != expected_path:
        raise HumanReviewReceiptError("sampling-plan path changed")
    sample_path = root / DEFAULT_SAMPLE.relative_to(ROOT)
    sample = sample_plan.load_plan(sample_path, root=root)
    expected = {
        "path": expected_path,
        "sha256": _sha256_file(sample_path),
        "plan_id": sample["plan_id"],
        "record_ids_sha256": sample["sample"]["record_ids_sha256"],
    }
    if dict(binding) != expected:
        raise HumanReviewReceiptError("sampling-plan binding differs from deterministic artifact")
    return expected


def _validate_completion(
    payload: Mapping[str, Any], sampled_decision_count: int, stage_count: int
) -> dict[str, int | bool]:
    completion = payload.get("completion")
    if not isinstance(completion, dict):
        raise HumanReviewReceiptError("completion must be an object")
    keys = {
        "sampled_decision_records_complete",
        "sampled_decision_records_not_started",
        "sampled_decision_records_incomplete",
        "stage_definitions_complete",
        "stage_definitions_not_started",
        "stage_definitions_incomplete",
        "reviewer_count",
        "row_outcome_blind_declarations_yes",
        "audit_outcome_access_declaration_present",
        "reviewer_change_count",
        "reviewer_uncertain_count",
        "adjudication_required_count",
        "adjudication_complete_count",
        "unresolved_count",
        "documentation_only_resolution_count",
    }
    _exact_keys(completion, keys, "completion")
    integer_keys = keys - {"audit_outcome_access_declaration_present"}
    normalized = {
        key: _require_nonnegative_int(completion.get(key), f"completion/{key}")
        for key in integer_keys
    }
    if completion.get("audit_outcome_access_declaration_present") is not True:
        raise HumanReviewReceiptError("audit-level outcome-access declaration is missing")
    normalized["audit_outcome_access_declaration_present"] = True
    if (
        normalized["sampled_decision_records_complete"] != sampled_decision_count
        or normalized["sampled_decision_records_not_started"] != 0
        or normalized["sampled_decision_records_incomplete"] != 0
    ):
        raise HumanReviewReceiptError("all sampled decision records must be complete")
    if (
        normalized["stage_definitions_complete"] != stage_count
        or normalized["stage_definitions_not_started"] != 0
        or normalized["stage_definitions_incomplete"] != 0
    ):
        raise HumanReviewReceiptError("all stage-definition records must be complete")
    if normalized["reviewer_count"] < 1:
        raise HumanReviewReceiptError("at least one reviewer is required")
    if normalized["row_outcome_blind_declarations_yes"] != sampled_decision_count + stage_count:
        raise HumanReviewReceiptError("every reviewed row must carry an outcome-blind declaration")
    # ``reviewer_change_count`` is retained as the public receipt field name;
    # it counts reviewer verdicts of ``No`` (that is, proposed corrections).
    required = normalized["reviewer_change_count"] + normalized["reviewer_uncertain_count"]
    if normalized["adjudication_required_count"] != required:
        raise HumanReviewReceiptError("adjudication-required count is inconsistent")
    if (
        normalized["adjudication_complete_count"] != required
        or normalized["unresolved_count"] != 0
    ):
        raise HumanReviewReceiptError("all No/Uncertain review records must be adjudicated")
    if normalized["documentation_only_resolution_count"] > required:
        raise HumanReviewReceiptError("documentation-only resolution count exceeds adjudications")
    return normalized


def _validate_outcome_blindness(payload: Mapping[str, Any]) -> None:
    outcome = payload.get("outcome_blindness")
    if not isinstance(outcome, dict):
        raise HumanReviewReceiptError("outcome_blindness must be an object")
    _exact_keys(
        outcome,
        {"declared", "instrument_forbidden_content_scan", "prohibited_inputs", "claim_limit"},
        "outcome_blindness",
    )
    if outcome.get("declared") is not True:
        raise HumanReviewReceiptError("outcome blindness must be declared")
    if outcome.get("instrument_forbidden_content_scan") != "PASS":
        raise HumanReviewReceiptError("review instrument forbidden-content scan did not pass")
    if outcome.get("prohibited_inputs") != list(PROHIBITED_REVIEW_INPUTS):
        raise HumanReviewReceiptError("outcome-blind prohibited-input inventory changed")
    if outcome.get("claim_limit") != OUTCOME_BLIND_CLAIM_LIMIT:
        raise HumanReviewReceiptError("outcome-blind claim limit changed")


def _validate_private_record(payload: Mapping[str, Any], completion: Mapping[str, Any]) -> None:
    record = payload.get("private_record")
    if not isinstance(record, dict):
        raise HumanReviewReceiptError("private_record must be an object")
    _exact_keys(
        record,
        {"workbook_sha256", "normalized_review_sha256", "adjudication_sha256"},
        "private_record",
    )
    _require_hex(record.get("workbook_sha256"), "private workbook digest")
    _require_hex(record.get("normalized_review_sha256"), "normalized review digest")
    adjudication = record.get("adjudication_sha256")
    required = int(completion["adjudication_required_count"])
    if required:
        _require_hex(adjudication, "adjudication digest")
    elif adjudication is not None:
        raise HumanReviewReceiptError("adjudication digest must be null when no adjudication is required")


def _validate_construct(
    payload: Mapping[str, Any], root: Path, completion: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> tuple[int, str, str]:
    construct = payload.get("construct")
    if not isinstance(construct, dict):
        raise HumanReviewReceiptError("construct must be an object")
    _exact_keys(
        construct,
        {
            "projection_schema",
            "fields",
            "baseline_sha256",
            "reviewed_sha256",
            "accepted_changes",
            "accepted_changes_sha256",
            "accepted_change_count",
        },
        "construct",
    )
    if construct.get("projection_schema") != CONSTRUCT_SCHEMA:
        raise HumanReviewReceiptError("construct projection schema changed")
    if construct.get("fields") != list(CONSTRUCT_FIELDS):
        raise HumanReviewReceiptError("construct field inventory changed")
    baseline = construct_projection(root)
    baseline_hash = construct_sha256(baseline)
    frozen_hash = freeze["construct"]["sha256"]
    if baseline_hash != frozen_hash:
        raise HumanReviewReceiptError(
            "current registry construct differs from the hard-pinned pre-review baseline"
        )
    if _require_hex(construct.get("baseline_sha256"), "baseline construct digest") != baseline_hash:
        raise HumanReviewReceiptError("baseline construct hash differs from current registry")
    reviewed, normalized_changes = _apply_accepted_changes(
        baseline, construct.get("accepted_changes")
    )
    if construct.get("accepted_changes") != normalized_changes:
        raise HumanReviewReceiptError("accepted changes are not canonically normalized")
    sample = sample_plan.load_plan(
        root / DEFAULT_SAMPLE.relative_to(ROOT), root=root
    )
    sampled_ids = {row["record_id"] for row in sample["selected_records"]}
    for change in normalized_changes:
        if change.get("kind") == "code_decision" and change.get("record_id") not in sampled_ids:
            raise HumanReviewReceiptError(
                "accepted code-decision change is outside the frozen human-validation sample"
            )
    change_count = _require_nonnegative_int(
        construct.get("accepted_change_count"), "accepted_change_count"
    )
    if change_count != len(normalized_changes):
        raise HumanReviewReceiptError("accepted change count is inconsistent")
    max_adjudicated = int(completion["adjudication_complete_count"])
    if change_count > max_adjudicated:
        raise HumanReviewReceiptError("accepted construct changes exceed completed adjudications")
    if (
        change_count + int(completion["documentation_only_resolution_count"])
        != int(completion["adjudication_required_count"])
    ):
        raise HumanReviewReceiptError(
            "accepted construct changes and non-construct adjudication resolutions do not "
            "partition every No/Uncertain review record"
        )
    expected_changes_hash = _sha256_bytes(_canonical_json_bytes(normalized_changes))
    if (
        _require_hex(construct.get("accepted_changes_sha256"), "accepted changes digest")
        != expected_changes_hash
    ):
        raise HumanReviewReceiptError("accepted changes digest is inconsistent")
    reviewed_hash = construct_sha256(reviewed)
    if _require_hex(construct.get("reviewed_sha256"), "reviewed construct digest") != reviewed_hash:
        raise HumanReviewReceiptError("reviewed construct digest is inconsistent")
    return change_count, baseline_hash, reviewed_hash


def validate_pre_review_freeze(
    root: Path = ROOT,
    *,
    freeze_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the canonical, source-hard-pinned pre-review baseline."""

    root = Path(root)
    canonical = root / DEFAULT_FREEZE.relative_to(ROOT)
    freeze_path = Path(freeze_path) if freeze_path is not None else canonical
    if freeze_path.resolve(strict=False) != canonical.resolve(strict=False):
        raise HumanReviewReceiptError(
            "pre-review verification requires the canonical repository freeze path"
        )
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise HumanReviewReceiptError(f"pre-review freeze is missing: {freeze_path}")
    observed_sha = _sha256_file(freeze_path)
    if _HEX64_RE.fullmatch(EXPECTED_FREEZE_SHA256) is None:
        raise HumanReviewReceiptError("pre-review freeze trust anchor is not configured")
    if observed_sha != EXPECTED_FREEZE_SHA256:
        raise HumanReviewReceiptError("pre-review freeze differs from the source-hard-pinned digest")
    payload = _load_json(freeze_path, "pre-review freeze", canonical=True)
    _exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "audit_id",
            "frozen_date",
            "benchmark_identity",
            "scope",
            "sampling_plan",
            "review_inputs_sha256",
            "instrument",
            "construct",
        },
        "pre-review freeze",
    )
    if payload.get("schema_version") != FREEZE_SCHEMA:
        raise HumanReviewReceiptError("unexpected pre-review freeze schema")
    if payload.get("status") != STATUS_FROZEN_PRE_REVIEW:
        raise HumanReviewReceiptError("pre-review freeze status is invalid")
    audit_id = payload.get("audit_id")
    if not isinstance(audit_id, str) or _AUDIT_ID_RE.fullmatch(audit_id) is None:
        raise HumanReviewReceiptError("pre-review freeze audit_id is invalid")
    _require_date(payload.get("frozen_date"), "pre-review frozen_date")
    _validate_identity(payload, root)
    _validate_scope(payload, root)
    _validate_sampling_plan_binding(payload, root)

    hashes = payload.get("review_inputs_sha256")
    if not isinstance(hashes, dict):
        raise HumanReviewReceiptError("pre-review review_inputs_sha256 must be an object")
    _exact_keys(hashes, FREEZE_REVIEW_INPUT_PATHS, "pre-review review_inputs_sha256")
    for relative in FREEZE_REVIEW_INPUT_PATHS:
        digest = _require_hex(hashes.get(relative), f"pre-review input digest {relative}")
        if digest != _sha256_file(_safe_source(root, relative)):
            raise HumanReviewReceiptError(f"pre-review input drift: {relative}")

    instrument = payload.get("instrument")
    if not isinstance(instrument, dict):
        raise HumanReviewReceiptError("pre-review instrument binding must be an object")
    _exact_keys(
        instrument,
        {
            "source_workbook_sha256",
            "immutable_semantics_sha256",
            "audit_settings_sha256",
        },
        "pre-review instrument",
    )
    for key in instrument:
        _require_hex(instrument[key], f"pre-review instrument/{key}")

    construct = payload.get("construct")
    if not isinstance(construct, dict):
        raise HumanReviewReceiptError("pre-review construct binding must be an object")
    _exact_keys(construct, {"schema_version", "fields", "sha256"}, "pre-review construct")
    if construct.get("schema_version") != CONSTRUCT_SCHEMA:
        raise HumanReviewReceiptError("pre-review construct schema changed")
    if construct.get("fields") != list(CONSTRUCT_FIELDS):
        raise HumanReviewReceiptError("pre-review construct field inventory changed")
    frozen_construct_hash = _require_hex(construct.get("sha256"), "pre-review construct digest")
    current_construct_hash = construct_sha256(construct_projection(root))
    if current_construct_hash != frozen_construct_hash:
        raise HumanReviewReceiptError(
            "current registry construct differs from the pre-review frozen baseline"
        )
    return payload


def _validate_disposition(
    payload: Mapping[str, Any], change_count: int, baseline_hash: str, reviewed_hash: str
) -> bool:
    disposition = payload.get("disposition")
    if not isinstance(disposition, dict):
        raise HumanReviewReceiptError("disposition must be an object")
    _exact_keys(
        disposition,
        {
            "kind",
            "release_eligible",
            "registry_dependent_rerun_required",
            "required_new_registry_id",
            "required_new_benchmark_version",
        },
        "disposition",
    )
    identity = payload["benchmark_identity"]
    kind = disposition.get("kind")
    if kind == NO_CONSTRUCT_CHANGE:
        if (
            change_count != 0
            or baseline_hash != reviewed_hash
            or disposition.get("release_eligible") is not True
            or disposition.get("registry_dependent_rerun_required") is not False
            or disposition.get("required_new_registry_id") is not None
            or disposition.get("required_new_benchmark_version") is not None
        ):
            raise HumanReviewReceiptError("no-change disposition invariants are inconsistent")
        return True
    if kind == CONSTRUCT_CHANGE_REQUIRED:
        new_registry = disposition.get("required_new_registry_id")
        new_benchmark = disposition.get("required_new_benchmark_version")
        if (
            change_count < 1
            or baseline_hash == reviewed_hash
            or disposition.get("release_eligible") is not False
            or disposition.get("registry_dependent_rerun_required") is not True
            or not isinstance(new_registry, str)
            or not new_registry.strip()
            or new_registry == identity["registry_snapshot"]
            or not isinstance(new_benchmark, str)
            or not new_benchmark.strip()
            or new_benchmark == identity["benchmark_version"]
        ):
            raise HumanReviewReceiptError("construct-change disposition invariants are inconsistent")
        return False
    raise HumanReviewReceiptError(f"unsupported review disposition: {kind!r}")


def validate_receipt(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    root: Path = ROOT,
    require_release_eligible: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    receipt_path = Path(receipt_path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise HumanReviewReceiptError(f"human-review receipt is missing: {receipt_path}")
    payload = _load_json(receipt_path, "human-review receipt", canonical=True)
    _exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "audit_id",
            "benchmark_identity",
            "pre_review_freeze",
            "public_inputs_sha256",
            "private_record",
            "scope",
            "sampling_plan",
            "completion",
            "outcome_blindness",
            "construct",
            "disposition",
            "review_completed_date",
            "receipt_issued_date",
        },
        "human-review receipt",
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise HumanReviewReceiptError("unexpected human-review receipt schema")
    if payload.get("status") != STATUS_SAMPLED_COMPLETE:
        raise HumanReviewReceiptError("human-review receipt status is not SAMPLED_COMPLETE")
    audit_id = payload.get("audit_id")
    if not isinstance(audit_id, str) or _AUDIT_ID_RE.fullmatch(audit_id) is None:
        raise HumanReviewReceiptError("audit_id is invalid")
    completed = _require_date(payload.get("review_completed_date"), "review_completed_date")
    issued = _require_date(payload.get("receipt_issued_date"), "receipt_issued_date")
    if issued < completed:
        raise HumanReviewReceiptError("receipt issue date predates review completion")

    freeze_binding = payload.get("pre_review_freeze")
    if not isinstance(freeze_binding, dict):
        raise HumanReviewReceiptError("pre_review_freeze must be an object")
    _exact_keys(freeze_binding, {"path", "sha256"}, "pre_review_freeze")
    expected_freeze_path = DEFAULT_FREEZE.relative_to(ROOT).as_posix()
    if freeze_binding.get("path") != expected_freeze_path:
        raise HumanReviewReceiptError("receipt pre-review freeze path changed")
    if (
        _require_hex(freeze_binding.get("sha256"), "receipt pre-review freeze digest")
        != EXPECTED_FREEZE_SHA256
    ):
        raise HumanReviewReceiptError("receipt pre-review freeze digest is not the hard-pinned digest")
    freeze = validate_pre_review_freeze(root)
    if payload.get("audit_id") != freeze.get("audit_id"):
        raise HumanReviewReceiptError("receipt audit_id differs from the pre-review freeze")

    _validate_identity(payload, root)
    if payload.get("benchmark_identity") != freeze.get("benchmark_identity"):
        raise HumanReviewReceiptError("receipt benchmark identity differs from the pre-review freeze")
    _validate_public_inputs(payload, root)
    _, sampled_decision_count, _, _, stage_count = _validate_scope(payload, root)
    if payload.get("scope") != freeze.get("scope"):
        raise HumanReviewReceiptError("receipt scope differs from the pre-review freeze")
    _validate_sampling_plan_binding(payload, root)
    if payload.get("sampling_plan") != freeze.get("sampling_plan"):
        raise HumanReviewReceiptError("receipt sampling plan differs from the pre-review freeze")
    completion = _validate_completion(payload, sampled_decision_count, stage_count)
    _validate_outcome_blindness(payload)
    _validate_private_record(payload, completion)
    change_count, baseline_hash, reviewed_hash = _validate_construct(
        payload, root, completion, freeze
    )
    release_eligible = _validate_disposition(
        payload, change_count, baseline_hash, reviewed_hash
    )
    if require_release_eligible and not release_eligible:
        raise HumanReviewReleaseBlocked(
            "completed human review accepted a construct change; a new registry/benchmark "
            "identifier and full registry-dependent rebuild are required"
        )
    return payload


def verify_release_gate(
    root: Path = ROOT,
    *,
    receipt_path: Path | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Require both a release-eligible receipt and a complete bound protocol."""

    root = Path(root)
    canonical_receipt = root / DEFAULT_RECEIPT.relative_to(ROOT)
    canonical_protocol = root / DEFAULT_PROTOCOL.relative_to(ROOT)
    receipt_path = Path(receipt_path) if receipt_path is not None else canonical_receipt
    protocol_path = Path(protocol_path) if protocol_path is not None else canonical_protocol
    if receipt_path.resolve(strict=False) != canonical_receipt.resolve(strict=False):
        raise HumanReviewReceiptError(
            "release verification requires the canonical repository review receipt path"
        )
    if protocol_path.resolve(strict=False) != canonical_protocol.resolve(strict=False):
        raise HumanReviewReceiptError(
            "release verification requires the canonical repository curation protocol path"
        )
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise HumanReviewReceiptError("canonical curation protocol is missing or unsafe")
    receipt = validate_receipt(
        receipt_path,
        root=root,
        require_release_eligible=True,
    )
    # Local import avoids making pending disclosure validation depend on a
    # completed receipt at module import time.
    from verify_registry_curation_protocol import validate_protocol

    protocol = validate_protocol(
        protocol_path,
        root / "chains/evidence/registry_evidence.json",
        receipt_path=receipt_path,
    )
    curation = protocol.get("curation", {})
    if (
        protocol.get("schema_version")
        != "upgrade-bench/registry-curation-protocol/4"
        or curation.get("completion_status") != "sampled_complete"
        or curation.get("retained_row_level_review_record") is not True
        or protocol.get("human_review_receipt_file")
        != "chains/evidence/registry_human_review_receipt.json"
        or protocol.get("human_review_receipt_sha256") != _sha256_file(receipt_path)
    ):
        raise HumanReviewReleaseBlocked(
            "curation protocol remains pending or is not bound to the completed sampled-validation receipt"
        )
    if curation.get("curator_count") != receipt["completion"]["reviewer_count"]:
        raise HumanReviewReceiptError("curation protocol reviewer count differs from receipt")
    return receipt


def promote_reviewed_draft(
    draft_path: Path,
    *,
    root: Path = ROOT,
    confirmation: str,
) -> Path:
    """Exclusively install an independently inspected private draft."""

    root = Path(root)
    if confirmation != PROMOTION_CONFIRMATION:
        raise HumanReviewReceiptError("explicit reviewed-draft promotion confirmation is missing")
    private_root = (root / "private").resolve(strict=True)
    draft_path = Path(draft_path)
    try:
        resolved = draft_path.resolve(strict=True)
    except OSError as exc:
        raise HumanReviewReceiptError("reviewed draft is missing") from exc
    if (
        not resolved.is_relative_to(private_root)
        or resolved == private_root
        or draft_path.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_nlink != 1
    ):
        raise HumanReviewReceiptError("reviewed draft must be a non-linked private regular file")
    cursor = root
    for part in resolved.relative_to(root.resolve(strict=True)).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise HumanReviewReceiptError("reviewed draft has a symbolic-link parent/member")
    payload = validate_receipt(resolved, root=root)
    raw = resolved.read_bytes()
    if raw != _canonical_json_bytes(payload):
        raise HumanReviewReceiptError("reviewed draft bytes changed after validation")
    destination = root / DEFAULT_RECEIPT.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--verify-public",
        action="store_true",
        help="validate a completed receipt, including construct-changing audit evidence",
    )
    mode.add_argument(
        "--verify-release",
        action="store_true",
        help="require a release-eligible no-construct-change receipt and complete protocol",
    )
    mode.add_argument(
        "--verify-freeze",
        action="store_true",
        help="verify the source-hard-pinned canonical pre-review freeze",
    )
    mode.add_argument(
        "--promote-reviewed-draft",
        action="store_true",
        help="exclusively install an independently inspected private draft",
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--confirm")
    args = parser.parse_args()
    try:
        if args.verify_freeze:
            payload = validate_pre_review_freeze(ROOT)
        elif args.promote_reviewed_draft:
            destination = promote_reviewed_draft(
                args.receipt,
                root=ROOT,
                confirmation=args.confirm or "",
            )
            payload = validate_receipt(destination, root=ROOT)
        elif args.verify_release:
            payload = verify_release_gate(
                ROOT,
                receipt_path=args.receipt,
                protocol_path=args.protocol,
            )
        else:
            payload = validate_receipt(args.receipt, root=ROOT)
    except (HumanReviewReceiptError, OSError) as exc:
        print(f"HUMAN REVIEW RECEIPT FAILED: {exc}", file=sys.stderr)
        return 1
    if args.verify_freeze:
        print(
            "registry human-review pre-review freeze OK "
            f"(audit_id={payload['audit_id']}; construct_sha256={payload['construct']['sha256']})"
        )
    else:
        print(
            "registry human-review receipt OK "
            f"(audit_id={payload['audit_id']}; "
            f"disposition={payload['disposition']['kind']}; "
            f"release_eligible={str(payload['disposition']['release_eligible']).lower()})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
