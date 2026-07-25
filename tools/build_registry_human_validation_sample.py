#!/usr/bin/env python3
"""Build and verify the frozen probability sample for HS92 human validation.

The 610-record rule-assigned ledger remains the sampling frame.  Human
validation covers a precommitted 212-record, outcome-blind stratified sample;
all 53 stage definitions are reviewed separately as a census.  Selection is
deterministic from a public pre-sampling commit and SHA-256 scores, so neither
Python's PRNG implementation nor repeated seed searches can change the sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "chains" / "evidence" / "registry_full_audit_ledger.csv"
OUTPUT_PATH = ROOT / "chains" / "evidence" / "registry_human_validation_sample.json"

SCHEMA_VERSION = "upgrade-bench/registry-human-validation-sample/1"
PLAN_ID = "UB-HS92-610-STRATIFIED-SAMPLE-212-20260719-R1"
STATUS = "FROZEN_BEFORE_HUMAN_VALIDATION"
FRAME_RECORDS = 610
FRAME_UNIQUE_HS6 = 588
SAMPLE_RECORDS = 212
STAGE_DEFINITION_RECORDS = 53
PRE_SAMPLING_COMMIT = "7e0c207d1d549cd6b5b06a469baaaff1d248e89f"
DOMAIN_SEPARATOR = "upgradebench/registry-human-validation/v1"
SEED_MATERIAL = f"{DOMAIN_SEPARATOR}|{PRE_SAMPLING_COMMIT}|{PLAN_ID}"
SEED_SHA256 = hashlib.sha256(SEED_MATERIAL.encode("utf-8")).hexdigest()

EXPECTED_SPLIT = {"include": 283, "exclude": 228, "out_of_stage": 99}
EXPECTED_SOURCE_SPLIT = {"observable_regex": 576, "legacy_only": 34}

BOUNDARY_IDENTITIES = {
    ("oilseed-soy", "292320"),
    ("sheep", "510910"),
    ("sheep", "510620"),
    ("sheep", "510720"),
    ("nickel", "750800"),
    ("cocoa", "180690"),
    ("nickel", "740323"),
    ("cotton", "550953"),
}

REQUIRED_LEDGER_FIELDS = {
    "chain_id",
    "code",
    "decision",
    "stage",
    "candidate_source",
    "description",
    "rationale_category",
    "rationale",
    "previous_stage",
}


class SamplePlanError(ValueError):
    """The sampling frame, deterministic allocation, or frozen artifact drifted."""


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], role: str) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise SamplePlanError(
            f"{role} keys changed: missing={sorted(wanted-observed)!r}, "
            f"extra={sorted(observed-wanted)!r}"
        )


def _record_id(row: Mapping[str, str]) -> str:
    return f"CODE-{row['chain_id']}-{row['code']}"


def _stratum_id(row: Mapping[str, str]) -> str:
    return "|".join((row["chain_id"], row["decision"], row["candidate_source"]))


def _load_frame(root: Path = ROOT) -> list[dict[str, str]]:
    path = root / LEDGER_PATH.relative_to(ROOT)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not REQUIRED_LEDGER_FIELDS.issubset(reader.fieldnames):
                raise SamplePlanError("sampling ledger schema changed")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise SamplePlanError(f"cannot read sampling ledger: {exc}") from exc
    if len(rows) != FRAME_RECORDS:
        raise SamplePlanError(f"sampling frame count changed: {len(rows)}")
    record_ids = [_record_id(row) for row in rows]
    if len(set(record_ids)) != FRAME_RECORDS:
        raise SamplePlanError("sampling frame record IDs are not unique")
    if len({row["code"] for row in rows}) != FRAME_UNIQUE_HS6:
        raise SamplePlanError("sampling frame unique-HS6 count changed")
    decision_split = {key: sum(row["decision"] == key for row in rows) for key in EXPECTED_SPLIT}
    source_split = {
        key: sum(row["candidate_source"] == key for row in rows)
        for key in EXPECTED_SOURCE_SPLIT
    }
    if decision_split != EXPECTED_SPLIT or source_split != EXPECTED_SOURCE_SPLIT:
        raise SamplePlanError("sampling frame decision/source split changed")
    return rows


def _certainty_reasons(row: Mapping[str, str], stratum_size: int) -> list[str]:
    reasons: list[str] = []
    if row["candidate_source"] == "legacy_only":
        reasons.append("legacy_only_provenance")
    if stratum_size <= 5:
        reasons.append("small_stratum_census_n_le_5")
    if row.get("previous_stage", ""):
        reasons.append("stage_reassignment")
    if (row["chain_id"], row["code"]) in BOUNDARY_IDENTITIES:
        reasons.append("published_boundary_case")
    return reasons


def _selection_score(record_id: str, stratum_id: str) -> str:
    raw = "\x00".join((DOMAIN_SEPARATOR, SEED_SHA256, stratum_id, record_id))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _allocate_random_slots(
    eligible: Mapping[str, list[dict[str, str]]], slots: int
) -> dict[str, int]:
    total = sum(len(rows) for rows in eligible.values())
    if slots < 0 or slots > total:
        raise SamplePlanError("random-slot allocation is impossible")
    allocation = {key: (slots * len(rows)) // total for key, rows in eligible.items()}
    used = sum(allocation.values())
    order = sorted(
        eligible,
        key=lambda key: (
            -((slots * len(eligible[key])) % total),
            key.encode("utf-8"),
        ),
    )
    for key in order:
        if used == slots:
            break
        if allocation[key] < len(eligible[key]):
            allocation[key] += 1
            used += 1
    if used != slots:
        raise SamplePlanError("Hamilton allocation did not exhaust the sample target")
    if any(value < 0 or value > len(eligible[key]) for key, value in allocation.items()):
        raise SamplePlanError("invalid stratum allocation")
    return allocation


def build_plan(root: Path = ROOT) -> dict[str, Any]:
    rows = _load_frame(root)
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[_stratum_id(row)].append(row)

    certainty: dict[str, list[str]] = {}
    eligible: dict[str, list[dict[str, str]]] = defaultdict(list)
    for stratum_id, members in strata.items():
        for row in members:
            reasons = _certainty_reasons(row, len(members))
            if reasons:
                certainty[_record_id(row)] = reasons
            else:
                eligible[stratum_id].append(row)

    random_target = SAMPLE_RECORDS - len(certainty)
    allocations = _allocate_random_slots(eligible, random_target)
    random_selected: dict[str, tuple[str, int]] = {}
    for stratum_id, members in eligible.items():
        ranked = sorted(
            ((_selection_score(_record_id(row), stratum_id), _record_id(row)) for row in members),
            key=lambda pair: (pair[0], pair[1].encode("utf-8")),
        )
        for rank, (score, record_id) in enumerate(ranked[: allocations[stratum_id]], start=1):
            random_selected[record_id] = (score, rank)

    selected_records: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for stratum_id in sorted(strata, key=lambda value: value.encode("utf-8")):
        members = strata[stratum_id]
        certainty_count = sum(_record_id(row) in certainty for row in members)
        random_pool = eligible.get(stratum_id, [])
        random_n = allocations.get(stratum_id, 0)
        sample_n = certainty_count + random_n
        chain_id, decision, candidate_source = stratum_id.split("|")
        stratum_rows.append(
            {
                "allocation": "certainty_plus_hamilton" if certainty_count else "hamilton",
                "candidate_source": candidate_source,
                "certainty_records": certainty_count,
                "chain_id": chain_id,
                "decision": decision,
                "frame_records": len(members),
                "random_pool_records": len(random_pool),
                "random_sample_records": random_n,
                "random_unit_design_weight": {
                    "denominator": random_n,
                    "numerator": len(random_pool),
                }
                if random_n
                else None,
                "random_unit_inclusion_probability": {
                    "denominator": len(random_pool),
                    "numerator": random_n,
                }
                if random_pool
                else None,
                "sample_records": sample_n,
                "stratum_id": stratum_id,
            }
        )
        for row in members:
            record_id = _record_id(row)
            is_certainty = record_id in certainty
            if not is_certainty and record_id not in random_selected:
                continue
            if is_certainty:
                probability = {"numerator": 1, "denominator": 1}
                weight = {"numerator": 1, "denominator": 1}
                score = None
                rank = None
            else:
                probability = {"numerator": random_n, "denominator": len(random_pool)}
                weight = {"numerator": len(random_pool), "denominator": random_n}
                score, rank = random_selected[record_id]
            selected_records.append(
                {
                    "analysis_weight": weight,
                    "candidate_source": row["candidate_source"],
                    "certainty": is_certainty,
                    "certainty_reasons": certainty.get(record_id, []),
                    "chain_id": row["chain_id"],
                    "code": row["code"],
                    "decision": row["decision"],
                    "inclusion_probability": probability,
                    "record_id": record_id,
                    "selection_rank_within_random_pool": rank,
                    "selection_score_sha256": score,
                    "stage": row["stage"] or None,
                    "stratum_id": stratum_id,
                }
            )

    selected_records.sort(key=lambda row: row["record_id"].encode("utf-8"))
    if len(selected_records) != SAMPLE_RECORDS:
        raise SamplePlanError(f"sample size changed: {len(selected_records)}")
    selected_ids = [row["record_id"] for row in selected_records]
    frame_projection = []
    for row in sorted(rows, key=lambda item: _record_id(item).encode("utf-8")):
        projection = dict(row)
        projection["record_id"] = _record_id(row)
        frame_projection.append(projection)

    plan = {
        "design": {
            "allocation": (
                "certainty units first; remaining slots allocated over noncertainty records "
                "by Hamilton largest remainder proportional to stratum random-pool size"
            ),
            "certainty_rules": [
                "candidate_source == legacy_only",
                "stratum frame size <= 5",
                "previous_stage is nonempty",
                "record is one of the eight published boundary cases",
            ],
            "confidence_reporting": (
                "Report design-weighted discrepancy estimates and intervals; simple-random-sample "
                "rule-of-three bounds are not valid for this unequal-probability design."
            ),
            "domain_separator": DOMAIN_SEPARATOR,
            "estimator": (
                "Horvitz-Thompson over the complete 610-record finite frame using exact stored "
                "inclusion probabilities; certainty units have probability one"
            ),
            "pre_sampling_commit": PRE_SAMPLING_COMMIT,
            "random_selection": (
                "within each noncertainty stratum rank SHA-256(domain_separator NUL seed_sha256 "
                "NUL stratum_id NUL record_id), then select the lowest allocated ranks"
            ),
            "sample_target_records": SAMPLE_RECORDS,
            "seed_material_sha256": SEED_SHA256,
            "stage_definition_review": "census",
            "stage_definition_records": STAGE_DEFINITION_RECORDS,
            "stratification": ["chain_id", "decision", "candidate_source"],
        },
        "frame": {
            "decision_records": FRAME_RECORDS,
            "decision_split": EXPECTED_SPLIT,
            "frame_projection_sha256": _sha256_bytes(_canonical_json_bytes(frame_projection)),
            "ledger_path": LEDGER_PATH.relative_to(ROOT).as_posix(),
            "ledger_sha256": _sha256_file(root / LEDGER_PATH.relative_to(ROOT)),
            "source_split": EXPECTED_SOURCE_SPLIT,
            "unique_hs6": FRAME_UNIQUE_HS6,
        },
        "plan_id": PLAN_ID,
        "sample": {
            "certainty_records": len(certainty),
            "decision_records": len(selected_records),
            "probability_selected_records": len(random_selected),
            "record_ids_sha256": _sha256_bytes(_canonical_json_bytes(selected_ids)),
            "unique_hs6": len({row["code"] for row in selected_records}),
        },
        "schema_version": SCHEMA_VERSION,
        "selected_records": selected_records,
        "status": STATUS,
        "strata": stratum_rows,
    }
    validate_plan(plan, root=root, compare_rebuild=False)
    return plan


def validate_plan(
    payload: Mapping[str, Any], *, root: Path = ROOT, compare_rebuild: bool = True
) -> dict[str, Any]:
    _exact_keys(
        payload,
        {"schema_version", "plan_id", "status", "frame", "design", "sample", "strata", "selected_records"},
        "sample plan",
    )
    if payload["schema_version"] != SCHEMA_VERSION or payload["plan_id"] != PLAN_ID:
        raise SamplePlanError("sample-plan identity changed")
    if payload["status"] != STATUS:
        raise SamplePlanError("sample-plan status changed")
    if not isinstance(payload["selected_records"], list) or len(payload["selected_records"]) != SAMPLE_RECORDS:
        raise SamplePlanError("sample-plan selected-record count changed")
    record_ids = [row.get("record_id") for row in payload["selected_records"] if isinstance(row, Mapping)]
    if len(record_ids) != SAMPLE_RECORDS or len(set(record_ids)) != SAMPLE_RECORDS:
        raise SamplePlanError("sample-plan record IDs are incomplete or duplicated")
    if record_ids != sorted(record_ids, key=lambda value: str(value).encode("utf-8")):
        raise SamplePlanError("sample-plan records are not in canonical record-ID order")
    if payload.get("sample", {}).get("record_ids_sha256") != _sha256_bytes(
        _canonical_json_bytes(record_ids)
    ):
        raise SamplePlanError("sample-plan record-ID digest mismatch")
    if sum(row.get("sample_records", -1) for row in payload.get("strata", [])) != SAMPLE_RECORDS:
        raise SamplePlanError("sample-plan stratum totals do not reconcile")
    if compare_rebuild:
        expected = build_plan(root)
        if _canonical_json_bytes(payload) != _canonical_json_bytes(expected):
            raise SamplePlanError("sample plan differs from deterministic rebuild")
    return dict(payload)


def load_plan(path: Path = OUTPUT_PATH, *, root: Path = ROOT) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SamplePlanError(f"cannot read valid sample plan: {exc}") from exc
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise SamplePlanError("sample plan must be canonical strict JSON")
    return validate_plan(payload, root=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = build_plan(ROOT)
    raw = _canonical_json_bytes(expected)
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
        print(
            f"wrote registry human-validation sample: {args.output} "
            f"({SAMPLE_RECORDS}/{FRAME_RECORDS}; certainty={expected['sample']['certainty_records']})"
        )
        return 0
    try:
        observed = args.output.read_bytes()
    except OSError as exc:
        raise SystemExit(f"REGISTRY HUMAN-VALIDATION SAMPLE FAILED: {exc}") from exc
    if observed != raw:
        raise SystemExit("REGISTRY HUMAN-VALIDATION SAMPLE FAILED: artifact differs from rebuild")
    print(
        "registry human-validation sample OK "
        f"({SAMPLE_RECORDS}/{FRAME_RECORDS}; certainty={expected['sample']['certainty_records']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
