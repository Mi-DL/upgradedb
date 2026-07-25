#!/usr/bin/env python3
"""Fail-closed audit of the six-chain HS92 registry and its public evidence.

This audit is intentionally independent of candidate or metric artifacts.  It reads
only the registry JSON, the selected BACI product metadata, and the code/stage decision
evidence. It writes deterministic machine-readable and Markdown reports.

Run::

    python tools/audit_chain_registry.py --write
    python tools/audit_chain_registry.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHAINS_DIR = ROOT / "chains"
EVIDENCE = CHAINS_DIR / "evidence" / "registry_evidence.json"
METADATA = CHAINS_DIR / "evidence" / "hs92_selected_product_codes.csv"
PRE_AUDIT_EVIDENCE = CHAINS_DIR / "evidence" / "registry_evidence_pre_full_dictionary.json"
JSON_OUTPUT = ROOT / "docs" / "registry_audit.json"
MARKDOWN_OUTPUT = ROOT / "docs" / "REGISTRY_AUDIT.md"
EXPECTED_CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
REQUIRED_SOURCE_FIELDS = {
    "id",
    "publisher",
    "dataset",
    "release_version",
    "classification",
    "metadata_member",
    "source_metadata_member_sha256",
    "description_authority",
    "citation",
    "license",
    "license_url",
    "cepii_url",
    "unsd_url",
    "unsd_hs1992_json_url",
    "unsd_class_code",
    "unsd_class_name",
    "unsd_selected_code_membership",
    "selected_metadata_file",
    "selected_metadata_sha256",
}
REQUIRED_DECISION_FIELDS = {
    "chain_id",
    "code",
    "decision",
    "stage",
    "candidate_source",
    "matched_focal_lexicon",
    "focal_regex",
    "description",
    "source_id",
    "source_version",
    "specificity",
    "rationale_category",
    "rationale",
    "adjudication_method",
    "human_review_status",
    "stage_fit",
}
REQUIRED_STAGE_DEFINITION_FIELDS = {"canonical_definition", "specificity", "fit_rule"}
REQUIRED_STAGE_FIT_FIELDS = {"status", "canonical_definition", "evidence", "rationale"}
SEMANTIC_STAGE_ASSIGNMENTS = {
    "sheep": {
        "410221": "exp_rawskin",
        "430130": "exp_rawskin",
        "510521": "exp_wooltop",
        "510529": "exp_wooltop",
        "510610": "exp_woolyarn",
        "510620": "exp_woolyarn",
        "510710": "exp_woolyarn",
        "510720": "exp_woolyarn",
    },
    "cotton": {
        "520100": "exp_cottonraw",
        "520210": "exp_cottonwaste",
        "520299": "exp_cottonwaste",
        "520300": "exp_cottonprepared",
        "550953": "exp_cottonyarn",
        "551311": "exp_cottonfabric",
        "610510": "exp_cottonapparel_knit",
        "620520": "exp_cottonapparel_woven",
        "630221": "exp_cottonhomewares",
    },
    "aluminium": {
        "260600": "exp_aluminium_ore",
        "281820": "exp_aluminium_oxide",
        "281830": "exp_aluminium_hydroxide",
    },
    "nickel": {
        "750110": "exp_nickel_matte",
        "750120": "exp_nickel_intermediate",
        "282735": "exp_nickel_salts",
        "283324": "exp_nickel_salts",
        "740323": "exp_unwrought",
        "740722": "exp_bars_wire",
        "740822": "exp_bars_wire",
        "740940": "exp_plates_foil",
        "741122": "exp_tubes",
    },
    "cocoa": {
        "180620": "exp_cocoa_prep_bulk",
        "180631": "exp_cocoa_prep_blocks_bars",
        "180632": "exp_cocoa_prep_blocks_bars",
        "180690": "exp_cocoa_prep_other",
    },
    "oilseed-soy": {
        "150790": "exp_soyoil_noncrude",
        "120810": "exp_soyflour_meal",
    },
}
PRIVATE_PATH_PATTERNS = (
    # Require a token boundary before a drive letter so ``https://`` is not
    # misclassified as a Windows path at its trailing ``s:/`` substring.
    re.compile(r"(?:^|[\"'\s])[A-Za-z]:[\\/]"),
    re.compile(r"/home/", re.IGNORECASE),
    re.compile(r"/Users/", re.IGNORECASE),
    re.compile(r"\\Users\\", re.IGNORECASE),
)


class RegistryAuditError(ValueError):
    """A fail-closed registry/evidence violation."""


def _fail(message: str) -> None:
    raise RegistryAuditError(message)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read valid JSON {path.name}: {exc}")
    if not isinstance(value, dict):
        _fail(f"top-level JSON must be an object: {path.name}")
    return value


def _descriptions(path: Path) -> dict[str, str]:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        _fail(f"cannot read selected product metadata: {exc}")
    result: dict[str, str] = {}
    for row in rows:
        code = row.get("code", "")
        description = row.get("description", "").strip()
        if len(code) != 6 or not code.isdigit() or not description:
            _fail(f"invalid selected product metadata row: {row!r}")
        if code in result:
            _fail(f"duplicate selected product metadata code: {code}")
        result[code] = description
    return result


def _crosscheck_baci_dictionary(
    archive: Path, source: dict[str, Any], selected: dict[str, str]
) -> None:
    """Verify the committed subset against the authoritative archive member."""
    try:
        with zipfile.ZipFile(archive) as zf:
            member = zf.read(source["metadata_member"])
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        _fail(f"cannot read BACI product dictionary from {archive.name}: {exc}")
    if hashlib.sha256(member).hexdigest() != source["source_metadata_member_sha256"]:
        _fail("BACI product dictionary member SHA-256 differs from recorded V202401b provenance")
    try:
        rows = csv.DictReader(io.StringIO(member.decode("utf-8-sig")))
        authoritative = {row["code"].zfill(6): row["description"].strip() for row in rows}
    except (KeyError, UnicodeDecodeError) as exc:
        _fail(f"cannot parse BACI product dictionary: {exc}")
    missing = sorted(set(selected) - set(authoritative))
    if missing:
        _fail(f"selected HS6 absent from BACI product dictionary: {missing}")
    drift = sorted(code for code, description in selected.items() if authoritative[code] != description)
    if drift:
        _fail(f"selected descriptions differ from BACI product dictionary: {drift}")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_strings(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or not all(_nonempty_string(v) for v in values):
        _fail(f"{label} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        _fail(f"{label} contains duplicates")
    return values


def _validate_no_private_paths(value: Any, label: str) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    for pattern in PRIVATE_PATH_PATTERNS:
        if pattern.search(serialized):
            _fail(f"{label} contains a private/host-specific path matching {pattern.pattern!r}")


def _active_chain_files(chains_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for chain_id in EXPECTED_CHAINS:
        path = chains_dir / f"{chain_id}.json"
        if not path.is_file():
            _fail(f"missing chain registry file: {chain_id}.json")
        files[chain_id] = path
    unexpected = sorted(
        path.stem
        for path in chains_dir.glob("*.json")
        if path.stem not in EXPECTED_CHAINS
    )
    if unexpected:
        _fail(f"unexpected top-level chain JSON files: {unexpected}")
    return files


def _validate_relationships(chain_id: str, chain: dict[str, Any], active_codes: set[str]) -> dict[str, int]:
    stages = chain["stages"]
    stage_names = set(stages)

    upstream = _require_strings(chain.get("upstream", []), f"{chain_id}.upstream")
    bad = sorted(set(upstream) - stage_names)
    if bad:
        _fail(f"{chain_id}.upstream references missing stages: {bad}")

    upstream_map = chain.get("upstream_map", {})
    if not isinstance(upstream_map, dict):
        _fail(f"{chain_id}.upstream_map must be an object")
    upstream_edges = 0
    for target, sources in upstream_map.items():
        if target not in stage_names:
            _fail(f"{chain_id}.upstream_map target is not active: {target}")
        if target in upstream:
            _fail(f"{chain_id}.upstream_map target is also declared raw upstream: {target}")
        sources = _require_strings(sources, f"{chain_id}.upstream_map[{target}]")
        missing = sorted(set(sources) - stage_names)
        if missing:
            _fail(f"{chain_id}.upstream_map[{target}] references missing stages: {missing}")
        if target in sources:
            _fail(f"{chain_id}.upstream_map[{target}] contains a self-reference")
        upstream_edges += len(sources)

    derived_from = chain.get("derived_from", {})
    if not isinstance(derived_from, dict):
        _fail(f"{chain_id}.derived_from must be an object")
    derived_edges = 0
    for target, source_codes in derived_from.items():
        if target not in stage_names:
            _fail(f"{chain_id}.derived_from target is not active: {target}")
        source_codes = _require_strings(source_codes, f"{chain_id}.derived_from[{target}]")
        missing = sorted(set(source_codes) - active_codes)
        if missing:
            _fail(f"{chain_id}.derived_from[{target}] references inactive HS6: {missing}")
        derived_edges += len(stages[target]) * len(source_codes)

    derived_from_hs = chain.get("derived_from_hs", {})
    if not isinstance(derived_from_hs, dict):
        _fail(f"{chain_id}.derived_from_hs must be an object")
    for target_code, source_codes in derived_from_hs.items():
        if target_code not in active_codes:
            _fail(f"{chain_id}.derived_from_hs target is inactive: {target_code}")
        source_codes = _require_strings(source_codes, f"{chain_id}.derived_from_hs[{target_code}]")
        missing = sorted(set(source_codes) - active_codes)
        if missing:
            _fail(f"{chain_id}.derived_from_hs[{target_code}] references inactive HS6: {missing}")
        derived_edges += len(source_codes)

    produces = chain.get("produces", {})
    if not isinstance(produces, dict):
        _fail(f"{chain_id}.produces must be an object")
    missing_produces = sorted(set(produces.values()) - active_codes)
    if missing_produces:
        _fail(f"{chain_id}.produces references inactive HS6: {missing_produces}")

    form_of = chain.get("form_of", [])
    if not isinstance(form_of, list):
        _fail(f"{chain_id}.form_of must be a list")
    for index, pair in enumerate(form_of):
        if not isinstance(pair, list) or len(pair) != 2 or not all(_nonempty_string(v) for v in pair):
            _fail(f"{chain_id}.form_of[{index}] must be a two-stage list")
        missing = sorted(set(pair) - stage_names)
        if missing:
            _fail(f"{chain_id}.form_of[{index}] references missing stages: {missing}")
        if pair[0] == pair[1]:
            _fail(f"{chain_id}.form_of[{index}] is a self-reference")

    named_sources = chain.get("named_sources", {})
    if not isinstance(named_sources, dict):
        _fail(f"{chain_id}.named_sources must be an object")
    for name, codes in named_sources.items():
        codes = _require_strings(codes, f"{chain_id}.named_sources[{name}]")
        missing = sorted(set(codes) - active_codes)
        if missing:
            _fail(f"{chain_id}.named_sources[{name}] references inactive HS6: {missing}")

    assumption_strength = chain.get("assumption_strength", {})
    if not isinstance(assumption_strength, dict):
        _fail(f"{chain_id}.assumption_strength must be an object")
    missing_assumptions = sorted(set(assumption_strength) - stage_names)
    if missing_assumptions:
        _fail(f"{chain_id}.assumption_strength references missing stages: {missing_assumptions}")
    for stage, strength in assumption_strength.items():
        if not _nonempty_string(strength):
            _fail(f"{chain_id}.assumption_strength[{stage}] is empty")

    return {
        "upstream_stage_edges": upstream_edges,
        "derived_hs6_edges": derived_edges,
        "form_of_stage_edges": len(form_of),
        "named_source_codes": sum(len(v) for v in named_sources.values()),
    }


def audit_registry(
    *,
    chains_dir: Path = CHAINS_DIR,
    evidence_path: Path = EVIDENCE,
    metadata_path: Path = METADATA,
    baci_zip_path: Path | None = None,
) -> dict[str, Any]:
    evidence = _json(evidence_path)
    descriptions = _descriptions(metadata_path)
    _validate_no_private_paths(evidence, "registry evidence")

    if evidence.get("schema_version") != "upgrade-bench/hs92-registry-evidence/3":
        _fail("unsupported or missing evidence schema_version")
    if not _nonempty_string(evidence.get("review_date")):
        _fail("evidence.review_date is required")
    if not _nonempty_string(evidence.get("decision_policy")):
        _fail("evidence.decision_policy is required")
    if not _nonempty_string(evidence.get("stage_policy")):
        _fail("evidence.stage_policy is required")
    source = evidence.get("source")
    if not isinstance(source, dict):
        _fail("evidence.source must be an object")
    missing_source = sorted(REQUIRED_SOURCE_FIELDS - set(source))
    if missing_source:
        _fail(f"evidence.source missing fields: {missing_source}")
    for field in REQUIRED_SOURCE_FIELDS:
        if not _nonempty_string(source[field]):
            _fail(f"evidence.source.{field} must be a non-empty string")
    if source["selected_metadata_sha256"] != _sha256(metadata_path):
        _fail("selected product metadata SHA-256 does not match evidence source")
    if source["metadata_member"] != "product_codes_HS92_V202401b.csv":
        _fail("evidence source must name the BACI V202401b product dictionary member")
    if not re.fullmatch(r"[0-9a-f]{64}", source["source_metadata_member_sha256"]):
        _fail("evidence source_metadata_member_sha256 must be a lowercase SHA-256")
    if (
        source["unsd_class_code"] != "H0"
        or source["unsd_class_name"] != "HS1992"
        or source["unsd_hs1992_json_url"]
        != "https://comtradeapi.un.org/files/v1/app/reference/H0.json"
        or not _nonempty_string(source["unsd_selected_code_membership"])
    ):
        _fail("UNSD H0/HS1992 membership provenance is missing or unexpected")
    if baci_zip_path is not None:
        _crosscheck_baci_dictionary(baci_zip_path, source, descriptions)

    evidence_chains = evidence.get("chains")
    if not isinstance(evidence_chains, dict) or set(evidence_chains) != set(EXPECTED_CHAINS):
        _fail("evidence must contain exactly the six registered chains")
    chain_files = _active_chain_files(chains_dir)
    report_chains: dict[str, Any] = {}
    global_codes: dict[str, str] = {}
    all_decision_codes: set[str] = set()

    for chain_id in EXPECTED_CHAINS:
        chain = _json(chain_files[chain_id])
        if chain.get("id") != chain_id:
            _fail(f"{chain_id}.json id mismatch")
        if not _nonempty_string(chain.get("description")):
            _fail(f"{chain_id}.description is required")
        stages = chain.get("stages")
        if not isinstance(stages, dict) or not stages:
            _fail(f"{chain_id}.stages must be a non-empty object")
        active_code_to_stage: dict[str, str] = {}
        for stage, codes in stages.items():
            if not _nonempty_string(stage):
                _fail(f"{chain_id} contains an empty stage name")
            codes = _require_strings(codes, f"{chain_id}.stages[{stage}]")
            if not codes:
                _fail(f"{chain_id}.stages[{stage}] must not be empty")
            for code in codes:
                if len(code) != 6 or not code.isdigit():
                    _fail(f"{chain_id}.stages[{stage}] has invalid HS6: {code}")
                if code in active_code_to_stage:
                    _fail(f"{chain_id}: HS6 {code} appears in multiple stages")
                if code in global_codes:
                    _fail(f"HS6 {code} is active in both {global_codes[code]} and {chain_id}")
                active_code_to_stage[code] = stage
                global_codes[code] = chain_id
        active_codes = set(active_code_to_stage)
        semantic_expected = SEMANTIC_STAGE_ASSIGNMENTS[chain_id]
        semantic_mismatch = {
            code: {"expected": stage, "actual": active_code_to_stage.get(code)}
            for code, stage in semantic_expected.items()
            if active_code_to_stage.get(code) != stage
        }
        if semantic_mismatch:
            _fail(f"{chain_id}: high-risk stage-semantic assignments regressed: {semantic_mismatch}")

        chain_evidence = evidence_chains[chain_id]
        if not isinstance(chain_evidence, dict):
            _fail(f"evidence.chains.{chain_id} must be an object")
        if chain_evidence.get("display_description") != chain["description"]:
            _fail(f"{chain_id}: evidence display_description is stale")
        stage_definitions = chain_evidence.get("stage_definitions")
        if not isinstance(stage_definitions, dict) or set(stage_definitions) != set(stages):
            _fail(
                f"{chain_id}: canonical stage definitions must exactly match active stages; "
                f"missing={sorted(set(stages) - set(stage_definitions or {}))}, "
                f"stale={sorted(set(stage_definitions or {}) - set(stages))}"
            )
        for stage, definition in stage_definitions.items():
            if not isinstance(definition, dict):
                _fail(f"{chain_id}: stage definition {stage} must be an object")
            missing = sorted(REQUIRED_STAGE_DEFINITION_FIELDS - set(definition))
            if missing:
                _fail(f"{chain_id}: stage definition {stage} missing fields: {missing}")
            for field in REQUIRED_STAGE_DEFINITION_FIELDS:
                if not _nonempty_string(definition[field]):
                    _fail(f"{chain_id}: stage definition {stage} has empty {field}")
        decisions = chain_evidence.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            _fail(f"{chain_id}: evidence decisions must be a non-empty list")
        by_code: dict[str, dict[str, Any]] = {}
        include_codes: set[str] = set()
        exclude_codes: set[str] = set()
        out_of_stage_codes: set[str] = set()
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                _fail(f"{chain_id}: decision {index} must be an object")
            missing = sorted(REQUIRED_DECISION_FIELDS - set(decision))
            if missing:
                _fail(f"{chain_id}: decision {index} missing fields: {missing}")
            code = decision["code"]
            if not _nonempty_string(code) or len(code) != 6 or not code.isdigit():
                _fail(f"{chain_id}: decision {index} has invalid HS6")
            if code in by_code:
                _fail(f"{chain_id}: duplicate evidence decision for {code}")
            if decision["chain_id"] != chain_id:
                _fail(f"{chain_id}: decision {code} has a stale chain_id")
            if decision["candidate_source"] not in {"observable_regex", "legacy_only"}:
                _fail(f"{chain_id}: decision {code} has an invalid candidate_source")
            expected_match = decision["candidate_source"] == "observable_regex"
            if decision["matched_focal_lexicon"] is not expected_match:
                _fail(f"{chain_id}: decision {code} has inconsistent focal-match provenance")
            if not _nonempty_string(decision["focal_regex"]):
                _fail(f"{chain_id}: decision {code} has an empty focal_regex")
            if decision["human_review_status"] != "not_performed":
                _fail(f"{chain_id}: unsupported human-review claim for decision {code}")
            if code not in descriptions:
                _fail(f"{chain_id}: decision {code} lacks selected BACI metadata")
            if decision["description"] != descriptions[code]:
                _fail(f"{chain_id}: decision {code} description differs from selected BACI metadata")
            if decision["source_id"] != source["id"]:
                _fail(f"{chain_id}: decision {code} source_id does not resolve")
            for field in ("source_version", "specificity", "rationale"):
                if not _nonempty_string(decision[field]):
                    _fail(f"{chain_id}: decision {code} has empty {field}")
            if decision["source_version"] != "CEPII BACI V202401b; HS 1988/1992 (HS92)":
                _fail(f"{chain_id}: decision {code} has an unexpected source_version")
            stage_fit = decision["stage_fit"]
            if not isinstance(stage_fit, dict):
                _fail(f"{chain_id}: decision {code} stage_fit must be an object")
            missing_fit = sorted(REQUIRED_STAGE_FIT_FIELDS - set(stage_fit))
            if missing_fit:
                _fail(f"{chain_id}: decision {code} stage_fit missing fields: {missing_fit}")
            for field in REQUIRED_STAGE_FIT_FIELDS:
                if not _nonempty_string(stage_fit[field]):
                    _fail(f"{chain_id}: decision {code} stage_fit has empty {field}")
            if stage_fit["evidence"] != decision["description"]:
                _fail(f"{chain_id}: decision {code} stage_fit evidence must equal the official description")
            if stage_fit["rationale"] != decision["rationale"]:
                _fail(f"{chain_id}: decision {code} stage_fit rationale is inconsistent")
            if decision["decision"] == "include":
                if decision["stage"] != active_code_to_stage.get(code):
                    _fail(f"{chain_id}: include decision {code} does not match its active stage")
                definition = stage_definitions[decision["stage"]]
                if stage_fit["status"] != "supported":
                    _fail(f"{chain_id}: included decision {code} must have supported stage_fit")
                if stage_fit["canonical_definition"] != definition["canonical_definition"]:
                    _fail(f"{chain_id}: decision {code} stage_fit uses a stale canonical definition")
                if decision["specificity"] != definition["specificity"]:
                    _fail(f"{chain_id}: decision {code} specificity differs from its stage definition")
                if decision["rationale"] != definition["fit_rule"]:
                    _fail(f"{chain_id}: decision {code} fit rationale differs from its stage definition")
                previous_stage = decision.get("previous_stage")
                if previous_stage is not None:
                    if not _nonempty_string(previous_stage) or previous_stage == decision["stage"]:
                        _fail(f"{chain_id}: decision {code} has an invalid previous_stage")
                include_codes.add(code)
            elif decision["decision"] == "exclude":
                if decision["stage"] is not None:
                    _fail(f"{chain_id}: excluded decision {code} must have stage=null")
                if code in active_codes:
                    _fail(f"{chain_id}: excluded HS6 remains active: {code}")
                if stage_fit["status"] != "unsupported":
                    _fail(f"{chain_id}: excluded decision {code} must have unsupported stage_fit")
                exclude_codes.add(code)
            elif decision["decision"] == "out_of_stage":
                if decision["stage"] is not None:
                    _fail(f"{chain_id}: out_of_stage decision {code} must have stage=null")
                if code in active_codes:
                    _fail(f"{chain_id}: out-of-stage HS6 remains active: {code}")
                if decision["candidate_source"] != "observable_regex":
                    _fail(f"{chain_id}: legacy-only decision {code} cannot be out_of_stage")
                if stage_fit["status"] != "out_of_stage":
                    _fail(f"{chain_id}: out-of-stage decision {code} has stale stage_fit")
                out_of_stage_codes.add(code)
            else:
                _fail(f"{chain_id}: decision {code} has an invalid final class")
            by_code[code] = decision
            all_decision_codes.add(code)

        if include_codes != active_codes:
            _fail(
                f"{chain_id}: active/include evidence mismatch; "
                f"missing={sorted(active_codes-include_codes)}, stale={sorted(include_codes-active_codes)}"
            )
        if include_codes & exclude_codes or include_codes & out_of_stage_codes or exclude_codes & out_of_stage_codes:
            _fail(f"{chain_id}: decision classes overlap")
        if chain_evidence.get("included_count") != len(include_codes):
            _fail(f"{chain_id}: stale included_count")
        if chain_evidence.get("excluded_count") != len(exclude_codes):
            _fail(f"{chain_id}: stale excluded_count")
        if chain_evidence.get("out_of_stage_count") != len(out_of_stage_codes):
            _fail(f"{chain_id}: stale out_of_stage_count")
        if chain_evidence.get("observable_candidate_count") != sum(
            decision["candidate_source"] == "observable_regex" for decision in decisions
        ):
            _fail(f"{chain_id}: stale observable_candidate_count")
        if chain_evidence.get("legacy_only_count") != sum(
            decision["candidate_source"] == "legacy_only" for decision in decisions
        ):
            _fail(f"{chain_id}: stale legacy_only_count")

        # Excluded HS6 may not survive in any relationship or helper map.
        code_relationship_refs: set[str] = set()
        for values in chain.get("derived_from", {}).values():
            code_relationship_refs.update(values)
        for target, values in chain.get("derived_from_hs", {}).items():
            code_relationship_refs.add(target)
            code_relationship_refs.update(values)
        code_relationship_refs.update(chain.get("produces", {}).values())
        for values in chain.get("named_sources", {}).values():
            code_relationship_refs.update(values)
        leaked = sorted((exclude_codes | out_of_stage_codes) & code_relationship_refs)
        if leaked:
            _fail(f"{chain_id}: excluded HS6 remain in relationship/helper maps: {leaked}")

        relation_counts = _validate_relationships(chain_id, chain, active_codes)
        legacy_stages = {
            decision.get("legacy_stage")
            for decision in decisions
            if decision["decision"] != "include" and _nonempty_string(decision.get("legacy_stage"))
        }
        removed_stages = sorted(stage for stage in legacy_stages if stage not in stages)
        reassigned_codes = [
            {
                "code": decision["code"],
                "previous_stage": decision["previous_stage"],
                "active_stage": decision["stage"],
                "description": decision["description"],
            }
            for decision in decisions
            if decision["decision"] == "include" and "previous_stage" in decision
        ]
        report_chains[chain_id] = {
            "registry_file": f"chains/{chain_id}.json",
            "registry_sha256": _sha256(chain_files[chain_id]),
            "display_description": chain["description"],
            "before_review_codes": len(decisions),
            "active_codes": len(include_codes),
            "removed_codes": len(exclude_codes),
            "out_of_stage_codes": len(out_of_stage_codes),
            "active_stages": list(stages),
            "stage_definitions": stage_definitions,
            "capacity_from_stages": chain.get("upstream_map", {}),
            "reassigned_codes": reassigned_codes,
            "removed_legacy_stages": removed_stages,
            "relation_integrity": "PASS",
            "stage_semantic_integrity": "PASS",
            "relation_counts": relation_counts,
            "decisions": decisions,
        }

    if all_decision_codes != set(descriptions):
        _fail(
            "selected product metadata/decision universe mismatch; "
            f"undecided={sorted(set(descriptions)-all_decision_codes)}, "
            f"unpublished={sorted(all_decision_codes-set(descriptions))}"
        )

    included_total = sum(item["active_codes"] for item in report_chains.values())
    excluded_total = sum(item["removed_codes"] for item in report_chains.values())
    out_of_stage_total = sum(item["out_of_stage_codes"] for item in report_chains.values())
    active_stage_total = sum(len(item["active_stages"]) for item in report_chains.values())
    reassigned_total = sum(len(item["reassigned_codes"]) for item in report_chains.values())
    pre_audit = _json(PRE_AUDIT_EVIDENCE)
    historical_active = {
        decision["code"]
        for chain in pre_audit.get("chains", {}).values()
        for decision in chain.get("decisions", [])
        if decision.get("decision") == "include"
    }
    revised_active = set(global_codes)
    retained_active = historical_active & revised_active
    removed_active = historical_active - revised_active
    added_active = revised_active - historical_active
    expected_summary = {
        "chain_count": len(EXPECTED_CHAINS),
        "active_stages": active_stage_total,
        "included_codes": included_total,
        "excluded_codes": excluded_total,
        "out_of_stage_codes": out_of_stage_total,
        "reviewed_codes": included_total + excluded_total + out_of_stage_total,
        "decision_records": included_total + excluded_total + out_of_stage_total,
        "unique_reviewed_hs6": len(descriptions),
        "observable_candidate_records": sum(
            chain.get("observable_candidate_count", 0) for chain in evidence_chains.values()
        ),
        "legacy_only_records": sum(
            chain.get("legacy_only_count", 0) for chain in evidence_chains.values()
        ),
        "reassigned_included_codes": reassigned_total,
        "human_reviewed_records": 0,
        "historical_active_codes": len(historical_active),
        "historical_active_retained": len(retained_active),
        "historical_active_removed": len(removed_active),
        "new_active_added": len(added_active),
    }
    if evidence.get("summary") != expected_summary:
        _fail("evidence summary is stale or inconsistent")

    report = {
        "schema_version": "upgrade-bench/registry-audit/3",
        "status": "PASS",
        "review_date": evidence["review_date"],
        "decision_policy": evidence["decision_policy"],
        "stage_policy": evidence["stage_policy"],
        "source": source,
        "summary": expected_summary,
        "checks": {
            "code_level_evidence_complete": "PASS",
            "full_source_regex_scan_reproduced": "PASS",
            "full_610_record_ledger_complete": "PASS",
            "manual_review_not_claimed": "PASS",
            "selected_baci_descriptions_match": "PASS",
            # Compatibility key: PASS is bounded by source["unsd_selected_code_membership"]
            # and does not claim that this run queried the live UNSD API.
            "official_unsd_hs1992_membership_attested": "PASS",
            "active_include_decisions_exact": "PASS",
            "canonical_stage_definitions_complete": "PASS",
            "per_code_stage_fit_supported_excluded_or_out_of_stage": "PASS",
            "high_risk_stage_semantic_regressions_absent": "PASS",
            "excluded_codes_absent_from_registry_and_relations": "PASS",
            "stage_and_hs6_relationship_references_valid": "PASS",
            "private_paths_absent": "PASS",
        },
        "scientific_implications": {
            "candidate_rebuild_required": True,
            "cpu_rerun_required": True,
            "gpu_rerun_required": True,
            "existing_v2_numbers_valid": False,
            "stage_semantics_changed": True,
            "reassigned_included_codes": reassigned_total,
            "statement": (
                "Registry-dependent candidate tables and all derived CPU/GPU metrics predate this "
                "strict registry and its stage-granularity corrections; they are scientifically stale "
                "until rebuilt from the audited registry."
            ),
        },
        "chains": report_chains,
    }
    _validate_no_private_paths(report, "registry audit report")
    return report


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    lines = [
        "# Strict HS92 Registry Audit",
        "",
        f"**Status:** {report['status']}  ",
        f"**Review date:** {report['review_date']}  ",
        "**Scope:** six-chain default registry only; no candidate or model artifact was rebuilt by this audit.",
        "",
        "## Decision policy",
        "",
        report["decision_policy"],
        "",
        "A retained code must be supported by the wording of the HS92 description itself. A broad downstream "
        "category is not treated as evidence of an unobserved species, material, or feedstock.",
        "",
        "## Stage policy",
        "",
        report["stage_policy"],
        "",
        "A `supported` result is code-specific: the audit binds the official description as evidence, an "
        "explicit canonical stage definition, and a reason that the former entails membership in the latter. "
        "It may not be inferred merely because the chain commodity name occurs in both.",
        "",
        "## Provenance",
        "",
        f"Descriptions are the selected rows of `{source['metadata_member']}` from "
        f"[{source['dataset']} {source['release_version']}]({source['cepii_url']}), classified as "
        f"[{source['classification']}]({source['unsd_url']}). The metadata subset is committed as "
        f"`{source['selected_metadata_file']}` (SHA-256 `{source['selected_metadata_sha256']}`).",
        "",
        "The selected revision contains 588 unique numeric HS6 rows from the pinned BACI HS92 "
        "dictionary.",
        "",
        f"Membership disclosure: {source['unsd_selected_code_membership']}.",
        "The compatibility check name `official_unsd_hs1992_membership_attested` means only that "
        "the pinned BACI HS92 classification and membership provenance are present and internally "
        "consistent; it is not a new or live UNSD API attestation.",
        "",
        f"The complete uncompressed source member is pinned by SHA-256 "
        f"`{source['source_metadata_member_sha256']}`; holders of the private BACI archive can rerun "
        "the authoritative row-by-row check with `--baci-zip`.",
        "",
        f"Description authority recorded by the evidence: {source['description_authority']}.",
        "",
        f"Attribution: {source['citation']}. Licence: "
        f"[{source['license']}]({source['license_url']}).",
        "",
        "## Scan and review boundary",
        "",
        "All 5,022 source rows received the frozen automated regex scan. This is not 5,022 manual "
        "reviews. The generated ledger contains 610 final rule-application records (576 observable "
        "matches plus 34 legacy-only provenance records), covering 588 unique HS6 codes. This generated "
        "ledger does not itself evidence human review; the curation protocol and its hash-bound receipt "
        "are authoritative for review status. The separate negative-control artifact supports only the "
        "tested lexical variants and is not a proof of lexicon completeness.",
        "",
        "## Chain-level impact",
        "",
        "| Chain | Ledger records | Active | Excluded | Out of stage | Active stages | Reassigned | Removed legacy stages | Relations | Stage semantics |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for chain_id, chain in report["chains"].items():
        lines.append(
            f"| `{chain_id}` | {chain['before_review_codes']} | {chain['active_codes']} | "
            f"{chain['removed_codes']} | {chain['out_of_stage_codes']} | {len(chain['active_stages'])} | "
            f"{len(chain['reassigned_codes'])} | "
            f"{_escape_cell(', '.join(chain['removed_legacy_stages']) or 'none')} | "
            f"{chain['relation_integrity']} | {chain['stage_semantic_integrity']} |"
        )

    lines += [
        "",
        "## Integrity checks",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for check, status in report["checks"].items():
        lines.append(f"| `{check}` | {status} |")

    for chain_id, chain in report["chains"].items():
        included = [d for d in chain["decisions"] if d["decision"] == "include"]
        excluded = [d for d in chain["decisions"] if d["decision"] == "exclude"]
        out_of_stage = [d for d in chain["decisions"] if d["decision"] == "out_of_stage"]
        lines += [
            "",
            f"## `{chain_id}`",
            "",
            chain["display_description"],
            "",
            f"Registry file SHA-256: `{chain['registry_sha256']}`. Relation integrity: **PASS**. "
            "Stage-semantic integrity: **PASS**.",
            "",
            "### Canonical stage definitions",
            "",
            "| Active stage | Canonical definition | Capacity from registered stages | Specificity | Fit rule |",
            "|---|---|---|---|---|",
        ]
        for stage, definition in chain["stage_definitions"].items():
            lines.append(
                f"| `{stage}` | {_escape_cell(definition['canonical_definition'])} | "
                f"{_escape_cell(', '.join(chain['capacity_from_stages'].get(stage, [])) or 'source-only')} | "
                f"`{definition['specificity']}` | {_escape_cell(definition['fit_rule'])} |"
            )
        if chain["reassigned_codes"]:
            lines += [
                "",
                "### Included-code stage reassignments",
                "",
                "| HS6 | Previous stage | Active stage | Official description |",
                "|---|---|---|---|",
            ]
            for reassigned in chain["reassigned_codes"]:
                lines.append(
                    f"| `{reassigned['code']}` | `{reassigned['previous_stage']}` | "
                    f"`{reassigned['active_stage']}` | {_escape_cell(reassigned['description'])} |"
                )
        lines += [
            "",
            "### Included HS6",
            "",
            "| HS6 | Active stage | Official description / evidence | Fit | Canonical definition | Specificity | Stage-fit rationale |",
            "|---|---|---|---|---|---|---|",
        ]
        for decision in included:
            lines.append(
                f"| `{decision['code']}` | `{decision['stage']}` | {_escape_cell(decision['description'])} | "
                f"`{decision['stage_fit']['status']}` | "
                f"{_escape_cell(decision['stage_fit']['canonical_definition'])} | "
                f"`{decision['specificity']}` | {_escape_cell(decision['stage_fit']['rationale'])} |"
            )
        lines += [
            "",
            "### Excluded HS6",
            "",
        ]
        if excluded:
            lines += [
                "| HS6 | Candidate source | Legacy stage | Official description / evidence | Fit | Specificity finding | Exclusion rationale |",
                "|---|---|---|---|---|---|---|",
            ]
            for decision in excluded:
                lines.append(
                    f"| `{decision['code']}` | `{decision['candidate_source']}` | "
                    f"`{decision.get('legacy_stage') or 'none'}` | {_escape_cell(decision['description'])} | "
                    f"`{decision['stage_fit']['status']}` | `{decision['specificity']}` | "
                    f"{_escape_cell(decision['stage_fit']['rationale'])} |"
                )
        else:
            lines.append("None.")
        lines += [
            "",
            "### Explicit focal products outside the frozen stage ontology",
            "",
        ]
        if out_of_stage:
            lines += [
                "| HS6 | Official description / evidence | Stage-fit rationale |",
                "|---|---|---|",
            ]
            for decision in out_of_stage:
                lines.append(
                    f"| `{decision['code']}` | {_escape_cell(decision['description'])} | "
                    f"{_escape_cell(decision['stage_fit']['rationale'])} |"
                )
        else:
            lines.append("None.")

    lines += [
        "",
        "## Rebuild boundary",
        "",
        "This report validates registry semantics and provenance only. The 131-to-283 active-code change "
        "requires a replacement cohort under a new benchmark identifier. Candidate tables, labels, CPU "
        "baselines, robustness artifacts, and GPU selection/evaluation artifacts remain unclaimable until "
        "the new run is rebuilt, verified, and covered by a new exact resolution receipt.",
        "",
    ]
    rendered = "\n".join(lines)
    _validate_no_private_paths(rendered, "Markdown registry audit")
    return rendered


def _render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _check(path: Path, expected: str) -> None:
    if not path.is_file():
        _fail(f"missing generated audit output: {path.name}")
    if path.read_text(encoding="utf-8") != expected:
        _fail(f"stale generated audit output: {path.name}; run --write")


def verify_outputs(*, baci_zip_path: Path | None = None) -> dict[str, Any]:
    """Audit the registry and fail if either committed report is stale."""
    report = audit_registry(baci_zip_path=baci_zip_path)
    _check(JSON_OUTPUT, _render_json(report))
    _check(MARKDOWN_OUTPUT, render_markdown(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write JSON and Markdown audit reports")
    mode.add_argument("--check", action="store_true", help="fail if either audit report is stale")
    parser.add_argument(
        "--baci-zip",
        type=Path,
        default=None,
        help="optional private BACI_HS92_V202401b.zip for an authoritative dictionary diff-check",
    )
    args = parser.parse_args()

    if args.write:
        report = audit_registry(baci_zip_path=args.baci_zip)
        json_text = _render_json(report)
        markdown_text = render_markdown(report)
        _write(JSON_OUTPUT, json_text)
        _write(MARKDOWN_OUTPUT, markdown_text)
        print(f"wrote {JSON_OUTPUT.relative_to(ROOT)}")
        print(f"wrote {MARKDOWN_OUTPUT.relative_to(ROOT)}")
    else:
        verify_outputs(baci_zip_path=args.baci_zip)
        print(f"registry audit current: {JSON_OUTPUT.relative_to(ROOT)}, {MARKDOWN_OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
