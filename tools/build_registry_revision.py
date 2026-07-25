#!/usr/bin/env python3
"""Materialise and verify the full-dictionary observable-attribution registry.

The adjudication universe is the union of (chain, HS6) records recalled by the
frozen lexicons and the historical 184-record proposal ledger.  The frozen
decision specification classifies every regex-recalled record; historical-only
records are retained as provenance and fail closed as exclusions.

Examples::

    python tools/build_registry_revision.py --write --output-root C:/tmp/registry-stage
    python tools/build_registry_revision.py --write
    python tools/build_registry_revision.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from build_registry_evidence import (
    INCLUDE_STAGE_META,
    PREVIOUS_STAGE,
    SOURCE_ID,
    SOURCE_METADATA_MEMBER_SHA256,
    SOURCE_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CHAIN_IDS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
RULE_PATH = ROOT / "chains" / "evidence" / "registry_candidate_recall_rule.json"
SPEC_PATH = ROOT / "chains" / "evidence" / "registry_revision_decision_spec.json"
PRE_AUDIT_PATH = (
    ROOT / "chains" / "evidence" / "registry_evidence_pre_full_dictionary.json"
)
CURRENT_EVIDENCE_PATH = ROOT / "chains" / "evidence" / "registry_evidence.json"
DEFAULT_BACI_ZIP = Path(
    os.environ.get("VCU_RAW", str(ROOT / "data" / "raw"))
) / "BACI_HS92_V202401b.zip"

EVIDENCE_REL = Path("chains/evidence/registry_evidence.json")
METADATA_REL = Path("chains/evidence/hs92_selected_product_codes.csv")
LEDGER_REL = Path("chains/evidence/registry_full_audit_ledger.csv")
RECEIPT_REL = Path("chains/evidence/registry_full_scan_receipt.json")
PRE_AUDIT_REL = Path("chains/evidence/registry_evidence_pre_full_dictionary.json")


class RegistryRevisionError(ValueError):
    """Raised when the frozen revision cannot be reproduced exactly."""


def _fail(message: str) -> None:
    raise RegistryRevisionError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read valid JSON {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"top-level JSON must be an object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _render_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _load_dictionary(archive: Path, rule: dict[str, Any]) -> tuple[bytes, list[dict[str, str]]]:
    member_name = rule["source_universe"]["member"]
    try:
        with zipfile.ZipFile(archive) as zf:
            member = zf.read(member_name)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        _fail(f"cannot read {member_name} from {archive}: {exc}")
    actual_hash = _sha256_bytes(member)
    expected_hash = rule["source_universe"]["member_sha256"]
    if actual_hash != expected_hash or actual_hash != SOURCE_METADATA_MEMBER_SHA256:
        _fail(f"BACI product dictionary hash mismatch: {actual_hash}")
    rows = list(csv.DictReader(io.StringIO(member.decode("utf-8-sig"))))
    if len(rows) != rule["source_universe"]["rows_scanned"]:
        _fail(f"source row count mismatch: {len(rows)}")
    if any(set(row) != {"code", "description"} for row in rows):
        _fail("unexpected BACI product dictionary columns")
    return member, rows


def _load_pre_audit() -> dict[str, Any]:
    path = PRE_AUDIT_PATH if PRE_AUDIT_PATH.is_file() else CURRENT_EVIDENCE_PATH
    payload = _json(path)
    summary = payload.get("summary", {})
    if summary.get("reviewed_codes") != 184 or summary.get("included_codes") != 131:
        _fail(
            "historical proposal snapshot is missing or is not the 184-record/131-active evidence"
        )
    return payload


def _historical_records(pre_audit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for chain_id in CHAIN_IDS:
        chain = pre_audit.get("chains", {}).get(chain_id, {})
        for decision in chain.get("decisions", []):
            key = (chain_id, str(decision.get("code", "")))
            if key in result:
                _fail(f"duplicate historical proposal: {key}")
            result[key] = decision
    if len(result) != 184:
        _fail(f"historical proposal record count is {len(result)}, expected 184")
    return result


def _scan_candidates(
    rows: list[dict[str, str]], rule: dict[str, Any]
) -> dict[str, set[str]]:
    numeric = [row for row in rows if re.fullmatch(r"\d{6}", row["code"])]
    if len(numeric) != rule["source_universe"]["numeric_hs6_rows"]:
        _fail(f"numeric source row count mismatch: {len(numeric)}")
    result: dict[str, set[str]] = {}
    for chain_id in CHAIN_IDS:
        lexicon = rule["recall_rule"]["lexicons"][chain_id]
        pattern = re.compile(lexicon["regex"], flags=re.IGNORECASE)
        codes = {row["code"] for row in numeric if pattern.search(row["description"])}
        if len(codes) != lexicon["candidate_records"]:
            _fail(
                f"{chain_id}: regex scan recalled {len(codes)}, expected "
                f"{lexicon['candidate_records']}"
            )
        result[chain_id] = codes
    if sum(map(len, result.values())) != rule["recall_rule"]["observable_candidate_records"]:
        _fail("observable chain-HS6 candidate total is stale")
    if len(set().union(*result.values())) != rule["recall_rule"]["observable_candidate_unique_hs6"]:
        _fail("observable unique-HS6 total is stale")
    return result


def _normalise_spec(
    spec: dict[str, Any], candidates: dict[str, set[str]]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, set[str]], dict[str, set[str]]]:
    if spec.get("rule_id") != "observable-attribution-full-dictionary-v1":
        _fail("decision specification rule_id does not match the frozen rule")
    if set(spec.get("chains", {})) != set(CHAIN_IDS):
        _fail("decision specification must contain exactly six chains")
    include_stages: dict[str, dict[str, list[str]]] = {}
    excludes: dict[str, set[str]] = {}
    out_of_stage: dict[str, set[str]] = {}
    for chain_id in CHAIN_IDS:
        item = spec["chains"][chain_id]
        stage_map = item.get("include_stages")
        if not isinstance(stage_map, dict) or set(stage_map) != set(INCLUDE_STAGE_META[chain_id]):
            _fail(f"{chain_id}: include_stages do not match the frozen stage ontology")
        clean_stages: dict[str, list[str]] = {}
        included: set[str] = set()
        for stage, values in stage_map.items():
            if not isinstance(values, list) or not values:
                _fail(f"{chain_id}/{stage}: included code list must be non-empty")
            clean = [str(code) for code in values]
            if len(clean) != len(set(clean)) or any(not re.fullmatch(r"\d{6}", x) for x in clean):
                _fail(f"{chain_id}/{stage}: invalid or duplicate included HS6")
            overlap = included & set(clean)
            if overlap:
                _fail(f"{chain_id}: included in more than one stage: {sorted(overlap)}")
            included.update(clean)
            clean_stages[stage] = clean
        ex = {str(code) for code in item.get("exclude", [])}
        oos = {str(code) for code in item.get("out_of_stage", [])}
        if included & ex or included & oos or ex & oos:
            _fail(f"{chain_id}: decision classes overlap")
        classified = included | ex | oos
        if classified != candidates[chain_id]:
            _fail(
                f"{chain_id}: observable decision coverage mismatch; "
                f"missing={sorted(candidates[chain_id]-classified)}, "
                f"extra={sorted(classified-candidates[chain_id])}"
            )
        include_stages[chain_id] = clean_stages
        excludes[chain_id] = ex
        out_of_stage[chain_id] = oos
    return include_stages, excludes, out_of_stage


def _updated_chain(chain_id: str, include_stages: dict[str, list[str]]) -> dict[str, Any]:
    chain = _json(ROOT / "chains" / f"{chain_id}.json")
    chain["stages"] = include_stages
    if chain_id == "sheep":
        derived = chain.setdefault("derived_from_hs", {})
        derived["510610"] = ["510111", "510121"]
    return chain


def _legacy_stage(decision: dict[str, Any] | None) -> str | None:
    if not decision:
        return None
    stage = decision.get("stage") if decision.get("decision") == "include" else None
    return stage or decision.get("legacy_stage")


def _decision_record(
    *,
    chain_id: str,
    code: str,
    description: str,
    decision: str,
    stage: str | None,
    source_kind: str,
    regex: str,
    historical: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy_stage = _legacy_stage(historical)
    if decision == "include":
        assert stage is not None
        meta = INCLUDE_STAGE_META[chain_id][stage]
        specificity = meta["specificity"]
        rationale = meta["fit_rationale"]
        fit_status = "supported"
        canonical = meta["canonical_definition"]
        category = "observable_attribution_and_stage_fit"
    elif decision == "out_of_stage":
        specificity = "focal-commodity-explicit"
        rationale = (
            "The official HS92 description explicitly names the focal commodity or material, "
            "but its described product form is outside every frozen stage in this chain."
        )
        fit_status = "out_of_stage"
        canonical = "No assignment: explicit focal commodity, product form outside frozen stages."
        category = "explicit_focal_product_outside_frozen_stage"
    else:
        specificity = "not-observably-attributable"
        if source_kind == "legacy_only":
            rationale = (
                "The historical proposal is retained for provenance, but the official HS92 "
                "description does not match the frozen focal lexicon. Observable attribution "
                "therefore forbids inclusion."
            )
            category = "legacy_description_lacks_focal_term"
        else:
            rationale = (
                "Although the frozen focal expression occurs in the official HS92 description, "
                "the wording is negative, generic, multi-commodity, mixed-species, or otherwise "
                "does not isolate the focal commodity or material for attribution."
            )
            category = "focal_expression_not_isolated"
        fit_status = "unsupported"
        canonical = "No active stage assignment under strict observable attribution."

    record: dict[str, Any] = {
        "chain_id": chain_id,
        "code": code,
        "decision": decision,
        "stage": stage,
        "candidate_source": source_kind,
        "matched_focal_lexicon": source_kind == "observable_regex",
        "focal_regex": regex,
        "description": description,
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "specificity": specificity,
        "rationale_category": category,
        "rationale": rationale,
        "adjudication_method": "frozen observable-attribution decision specification",
        "human_review_status": "not_performed",
        "stage_fit": {
            "status": fit_status,
            "canonical_definition": canonical,
            "evidence": description,
            "rationale": rationale,
        },
    }
    if legacy_stage:
        record["legacy_stage"] = legacy_stage
    previous = PREVIOUS_STAGE.get((chain_id, code))
    if decision == "include" and previous and previous != stage:
        record["previous_stage"] = previous
    return record


def build(archive: Path) -> dict[str, Any]:
    rule = _json(RULE_PATH)
    spec = _json(SPEC_PATH)
    pre_audit = _load_pre_audit()
    member, rows = _load_dictionary(archive, rule)
    descriptions = {
        row["code"]: row["description"].strip()
        for row in rows
        if re.fullmatch(r"\d{6}", row["code"])
    }
    candidates = _scan_candidates(rows, rule)
    include_stages, excludes, out_of_stage = _normalise_spec(spec, candidates)
    historical = _historical_records(pre_audit)

    legacy_only = {
        key for key in historical if key[1] not in candidates[key[0]]
    }
    expected_legacy = rule["recall_rule"]["legacy_carryover"]
    if len(legacy_only) != expected_legacy["additional_chain_hs6_records"]:
        _fail(f"legacy-only record count mismatch: {len(legacy_only)}")
    observable_unique = set().union(*candidates.values())
    legacy_unique = {code for _, code in legacy_only}
    if len(legacy_unique - observable_unique) != expected_legacy["additional_unique_hs6"]:
        _fail("legacy-only contribution to the union's unique-HS6 count mismatch")
    legacy_active = [
        key for key in legacy_only if historical[key].get("decision") == "include"
    ]
    if legacy_active:
        _fail(f"historical active records fail the frozen focal lexicon: {legacy_active}")

    chains_out: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    updated_chains: dict[str, dict[str, Any]] = {}
    for chain_id in CHAIN_IDS:
        updated = _updated_chain(chain_id, include_stages[chain_id])
        updated_chains[chain_id] = updated
        stage_by_code = {
            code: stage
            for stage, codes in include_stages[chain_id].items()
            for code in codes
        }
        codes = candidates[chain_id] | {code for ch, code in legacy_only if ch == chain_id}
        records: list[dict[str, Any]] = []
        regex = rule["recall_rule"]["lexicons"][chain_id]["regex"]
        for code in sorted(codes):
            if code in stage_by_code:
                decision, stage = "include", stage_by_code[code]
            elif code in out_of_stage[chain_id]:
                decision, stage = "out_of_stage", None
            else:
                decision, stage = "exclude", None
            source_kind = "observable_regex" if code in candidates[chain_id] else "legacy_only"
            record = _decision_record(
                chain_id=chain_id,
                code=code,
                description=descriptions[code],
                decision=decision,
                stage=stage,
                source_kind=source_kind,
                regex=regex,
                historical=historical.get((chain_id, code)),
            )
            records.append(record)
            ledger.append(record)

        counts = Counter(record["decision"] for record in records)
        chains_out[chain_id] = {
            "display_description": updated.get("description", ""),
            "stage_definitions": {
                stage: {
                    "canonical_definition": meta["canonical_definition"],
                    "specificity": meta["specificity"],
                    "fit_rule": meta["fit_rationale"],
                }
                for stage, meta in INCLUDE_STAGE_META[chain_id].items()
            },
            "observable_candidate_count": len(candidates[chain_id]),
            "legacy_only_count": sum(r["candidate_source"] == "legacy_only" for r in records),
            "included_count": counts["include"],
            "excluded_count": counts["exclude"],
            "out_of_stage_count": counts["out_of_stage"],
            "decisions": records,
        }

    ledger.sort(key=lambda item: (CHAIN_IDS.index(item["chain_id"]), item["code"]))
    decision_counts = Counter(record["decision"] for record in ledger)
    observable_counts = Counter(
        record["decision"]
        for record in ledger
        if record["candidate_source"] == "observable_regex"
    )
    unique_codes = {record["code"] for record in ledger}
    if len(ledger) != 610 or len(unique_codes) != 588:
        _fail(f"full ledger cardinality mismatch: records={len(ledger)}, unique={len(unique_codes)}")
    if decision_counts != Counter(include=283, exclude=228, out_of_stage=99):
        _fail(f"full ledger decision counts mismatch: {dict(decision_counts)}")
    if observable_counts != Counter(include=283, exclude=194, out_of_stage=99):
        _fail(f"observable decision counts mismatch: {dict(observable_counts)}")

    historical_active = {
        code
        for (_chain_id, code), record in historical.items()
        if record.get("decision") == "include"
    }
    revised_active = {
        record["code"] for record in ledger if record["decision"] == "include"
    }
    retained_active = historical_active & revised_active
    removed_active = historical_active - revised_active
    added_active = revised_active - historical_active
    if (len(historical_active), len(retained_active), len(removed_active), len(added_active)) != (
        131,
        131,
        0,
        152,
    ):
        _fail(
            "historical/revised active membership delta mismatch: "
            f"old={len(historical_active)}, retained={len(retained_active)}, "
            f"removed={len(removed_active)}, added={len(added_active)}"
        )

    evidence: dict[str, Any] = {
        "schema_version": "upgrade-bench/hs92-registry-evidence/3",
        "review_date": "2026-07-16",
        "rule_id": rule["rule_id"],
        "decision_policy": (
            "Strict observable attribution using only the official BACI HS92 description. "
            "Commodity-explicit blends and residual baskets remain eligible when their form is "
            "inside a frozen stage; generic, negative, mixed-species, or unresolved-material "
            "descriptions are excluded; explicit focal products outside the frozen ontology are "
            "recorded separately as out_of_stage."
        ),
        "stage_policy": (
            "Each active stage has a prospective canonical definition. Cotton apparel and "
            "homeware stages deliberately freeze the pre-audit form ontology rather than all of "
            "Chapters 61--63. No trade flow or model outcome is a recall or adjudication signal."
        ),
        "scope_limit": (
            "All 5022 source rows received the frozen automated regex scan. This is not 5022 "
            "manual reviews, and the tested lexicon negative controls support but cannot prove "
            "semantic completeness. Human review of this generated ledger was not performed."
        ),
        "source": {
            "id": SOURCE_ID,
            "publisher": "CEPII",
            "dataset": "BACI",
            "release_version": "V202401b",
            "classification": "HS 1988/1992 (HS92), six-digit subheadings",
            "metadata_member": rule["source_universe"]["member"],
            "source_metadata_member_sha256": _sha256_bytes(member),
            "description_authority": "Official HS 1988/1992 descriptions distributed in CEPII BACI V202401b",
            "citation": "Gaulier and Zignago (2010), CEPII Working Paper 2010-23",
            "license": "Etalab Open Licence 2.0, as stated by CEPII for BACI",
            "license_url": "https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf",
            "cepii_url": "https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html",
            "unsd_url": "https://unstats.un.org/unsd/classifications/econ",
            "unsd_hs1992_json_url": "https://comtradeapi.un.org/files/v1/app/reference/H0.json",
            "unsd_class_code": "H0",
            "unsd_class_name": "HS1992",
            "unsd_selected_code_membership": (
                "No new UNSD API membership claim; 588/588 codes are numeric rows in the "
                "pinned CEPII BACI HS92 dictionary"
            ),
            "selected_metadata_file": METADATA_REL.as_posix(),
            "selected_metadata_sha256": "POPULATED_AFTER_RENDER",
        },
        "summary": {
            "chain_count": len(CHAIN_IDS),
            "active_stages": sum(len(value) for value in include_stages.values()),
            "included_codes": decision_counts["include"],
            "excluded_codes": decision_counts["exclude"],
            "out_of_stage_codes": decision_counts["out_of_stage"],
            "reviewed_codes": len(ledger),
            "decision_records": len(ledger),
            "unique_reviewed_hs6": len(unique_codes),
            "observable_candidate_records": 576,
            "legacy_only_records": 34,
            "reassigned_included_codes": sum(
                "previous_stage" in record for record in ledger if record["decision"] == "include"
            ),
            "human_reviewed_records": 0,
            "historical_active_codes": len(historical_active),
            "historical_active_retained": len(retained_active),
            "historical_active_removed": len(removed_active),
            "new_active_added": len(added_active),
        },
        "chains": chains_out,
    }
    receipt = {
        "schema_version": "upgrade-bench/registry-full-scan-receipt/1",
        "status": "PASS",
        "rule_id": rule["rule_id"],
        "source_member": rule["source_universe"]["member"],
        "source_member_sha256": _sha256_bytes(member),
        "source_rows_automated_regex_scanned": len(rows),
        "source_rows_manually_reviewed": 0,
        "numeric_hs6_rows": len(descriptions),
        "observable_candidate_records_adjudicated": 576,
        "legacy_only_records_adjudicated": 34,
        "decision_records": len(ledger),
        "unique_hs6": len(unique_codes),
        "observable_decision_counts": dict(observable_counts),
        "full_ledger_decision_counts": dict(decision_counts),
        "human_review_status": "not_performed",
        "semantic_double_check": {
            "status": "PASS_AFTER_CORRECTIONS",
            "review_type": (
                "Separate partitioned AI-assisted read-only checks; not independent replication "
                "and not human review"
            ),
            "records_checked": 610,
            "corrected_records": [
                {"chain_id": "sheep", "code": "430130", "from": "out_of_stage", "to": "include"},
                {"chain_id": "cotton", "code": "580126", "from": "include", "to": "out_of_stage"},
                {"chain_id": "cotton", "code": "580310", "from": "include", "to": "out_of_stage"},
                {"chain_id": "aluminium", "code": "282690", "from": "out_of_stage", "to": "exclude"},
                {"chain_id": "aluminium", "code": "690320", "from": "exclude", "to": "out_of_stage"},
                {"chain_id": "cocoa", "code": "843820", "from": "out_of_stage", "to": "exclude"},
            ],
        },
        "lexicon_limit": (
            "Exhaustive source-table application of frozen regexes is not proof that their "
            "lexicons contain every possible synonym. See registry_lexicon_negative_control.json."
        ),
        "historical_active_records_lacking_focal_match": len(legacy_active),
        "active_membership_delta": {
            "historical": len(historical_active),
            "retained": len(retained_active),
            "removed": sorted(removed_active),
            "added_count": len(added_active),
            "revised": len(revised_active),
        },
        "replacement_cohort_required": True,
        "benchmark_identity": {
            "benchmark_version": "2.1-dev",
            "data_snapshot": "oa-full-dictionary-hs92-v202401b-20260716-r1",
        },
    }
    return {
        "evidence": evidence,
        "ledger": ledger,
        "unique_codes": sorted(unique_codes),
        "descriptions": descriptions,
        "chains": updated_chains,
        "receipt": receipt,
        "pre_audit": pre_audit,
    }


def _metadata_text(codes: list[str], descriptions: dict[str, str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["code", "description"], lineterminator="\n")
    writer.writeheader()
    for code in codes:
        writer.writerow({"code": code, "description": descriptions[code]})
    return buffer.getvalue()


def _ledger_text(records: list[dict[str, Any]]) -> str:
    fields = [
        "chain_id",
        "code",
        "decision",
        "stage",
        "candidate_source",
        "matched_focal_lexicon",
        "focal_regex",
        "description",
        "rationale_category",
        "rationale",
        "legacy_stage",
        "previous_stage",
        "adjudication_method",
        "human_review_status",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field) for field in fields})
    return buffer.getvalue()


def _outputs(built: dict[str, Any]) -> dict[Path, str]:
    metadata = _metadata_text(built["unique_codes"], built["descriptions"])
    evidence = built["evidence"]
    evidence["source"]["selected_metadata_sha256"] = _sha256_bytes(metadata.encode("utf-8"))
    outputs: dict[Path, str] = {
        METADATA_REL: metadata,
        LEDGER_REL: _ledger_text(built["ledger"]),
        EVIDENCE_REL: _render_json(evidence),
        RECEIPT_REL: _render_json(built["receipt"]),
        PRE_AUDIT_REL: _render_json(built["pre_audit"]),
    }
    for chain_id, chain in built["chains"].items():
        outputs[Path("chains") / f"{chain_id}.json"] = _render_json(chain)
    return outputs


def _write_outputs(root: Path, outputs: dict[Path, str]) -> None:
    for relative, content in outputs.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")


def _check_outputs(root: Path, outputs: dict[Path, str]) -> None:
    stale: list[str] = []
    for relative, expected in outputs.items():
        target = root / relative
        if not target.is_file() or target.read_text(encoding="utf-8") != expected:
            stale.append(relative.as_posix())
    if stale:
        _fail(f"generated registry revision outputs are missing or stale: {stale}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--baci-zip", type=Path, default=DEFAULT_BACI_ZIP)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT,
        help="write/check an isolated staging root instead of the repository",
    )
    args = parser.parse_args()
    if not args.baci_zip.is_file():
        parser.error(f"BACI archive not found: {args.baci_zip}")
    built = build(args.baci_zip)
    outputs = _outputs(built)
    output_root = args.output_root.resolve()
    if args.write:
        _write_outputs(output_root, outputs)
        print(
            f"wrote registry revision: records={len(built['ledger'])}, "
            f"unique_hs6={len(built['unique_codes'])}, root={output_root}"
        )
    else:
        _check_outputs(output_root, outputs)
        print(f"registry revision current: root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
