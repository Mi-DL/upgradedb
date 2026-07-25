#!/usr/bin/env python3
"""Privately verify the completed HS92 review workbook and draft its receipt.

This is the only workbook-to-receipt bridge.  It reads OOXML directly with the
standard library, treats formula caches as untrusted, reconstructs completion
from reviewer-entered cells, and compares every sampled proposal cell with the
frozen sample and current registry evidence.  It never edits the workbook and never writes the canonical
public receipt.  The only outputs are a private normalized record and a
non-canonical draft receipt in a caller-supplied empty directory.

For an outcome-blind declaration to be machine-checkable, the value cell on
the ``outcome_access_declaration`` row of Audit settings
must contain exactly OUTCOME_ACCESS_DECLARATION below.  No and Uncertain
rows additionally require a canonical adjudication JSON file; see
ADJUDICATION_SCHEMA and _validate_adjudication for its deliberately small
schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import zipfile
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET

import build_registry_human_validation_sample as sample_plan
import registry_human_review_receipt as public_receipt


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_SCHEMA = "upgrade-bench/registry-human-review-normalized-private/3"
ADJUDICATION_SCHEMA = "upgrade-bench/registry-human-review-adjudication/2"
PRIVATE_NAME = "registry_human_review_normalized.private.json"
DRAFT_NAME = "registry_human_review_receipt.draft.json"

OUTCOME_ACCESS_DECLARATION = (
    "I declare that trade values, benchmark labels, cohort impacts, model scores, "
    "and downstream result summaries were not supplied for this review."
)

FRAME_DECISION_RECORDS = sample_plan.FRAME_RECORDS
SAMPLED_DECISION_RECORDS = sample_plan.SAMPLE_RECORDS
SAMPLED_REVIEW_LAST_ROW = SAMPLED_DECISION_RECORDS + 1
STAGE_DEFINITION_RECORDS = sample_plan.STAGE_DEFINITION_RECORDS
STAGE_REVIEW_LAST_ROW = STAGE_DEFINITION_RECORDS + 1
OUTCOME_BLIND_REVIEW_ROWS = SAMPLED_DECISION_RECORDS + STAGE_DEFINITION_RECORDS

SHEET_NAMES = (
    "README",
    "Review",
    "Stage definitions",
    "Review summary",
    "Audit settings",
    "Boundary cases",
)
REVIEW_HEADERS = (
    "record_id", "chain_id", "hs6", "official_description", "candidate_source",
    "proposal_decision", "proposal_stage", "allowed_stages_for_chain",
    "canonical_stage_definition", "rationale_category", "proposal_rationale",
    "reviewer_id", "review_date", "verdict", "corrected_decision",
    "corrected_stage", "reviewer_note", "outcome_blind_declaration", "review_status",
)
STAGE_HEADERS = (
    "record_id", "chain_id", "stage", "canonical_definition", "specificity",
    "fit_rule", "active_code_count", "role", "reviewer_id", "review_date",
    "verdict", "corrected_definition", "corrected_fit_rule", "reviewer_note",
    "outcome_blind_declaration", "review_status",
)
AUDIT_HEADERS = ("field", "value", "required", "description")
BOUNDARY_HEADERS = (
    "chain_id", "hs6", "official_description", "proposal_decision",
    "proposal_stage", "rule_application",
)
AUDIT_FIELDS = (
    "audit_id", "workbook_generated_date", "registry_evidence_schema",
    "registry_evidence_sha256", "full_ledger_sha256", "review_codebook_sha256",
    "recall_rule_sha256", "source_dictionary_member_sha256",
    "selected_metadata_sha256", "human_validation_sample_sha256",
    "sample_plan_id", "decision_records", "sampled_decision_records", "unique_hs6",
    "human_reviews_completed_at_generation", "primary_reviewer", "review_start_date",
    "review_end_date", "outcome_access_declaration", "final_audit_owner", "audit_notes",
)
AUDIT_REQUIRED = {field: field != "audit_notes" for field in AUDIT_FIELDS}
AUDIT_DESCRIPTIONS = {
    "audit_id": "Unique identifier for this review instrument.",
    "workbook_generated_date": "Workbook generation date (Australia/Sydney).",
    "registry_evidence_schema": "Current evidence schema.",
    "registry_evidence_sha256": "SHA-256 of chains/evidence/registry_evidence.json.",
    "full_ledger_sha256": "SHA-256 of the 610-record CSV ledger.",
    "review_codebook_sha256": "SHA-256 of the operational review codebook.",
    "recall_rule_sha256": "SHA-256 of the frozen candidate-recall rule.",
    "source_dictionary_member_sha256": "Pinned BACI HS92 dictionary-member SHA-256.",
    "selected_metadata_sha256": "SHA-256 for the 588 unique HS6 descriptions covered by the ledger.",
    "human_validation_sample_sha256": "SHA-256 of the frozen 212-record probability-sample artifact.",
    "sample_plan_id": "Frozen probability-sampling plan identifier.",
    "decision_records": "Complete chain-HS6 machine-ledger sampling frame.",
    "sampled_decision_records": "Chain-HS6 decision records selected for human validation.",
    "unique_hs6": "Globally unique HS6 codes covered.",
    "human_reviews_completed_at_generation": "This workbook does not itself constitute completed review.",
    "primary_reviewer": "Enter reviewer identity before review starts.",
    "review_start_date": "Enter the actual start date.",
    "review_end_date": "Enter only after all retained records are complete.",
    "outcome_access_declaration": (
        "Declare that trade values, labels, cohort impacts, model scores, and downstream summaries were not supplied."
    ),
    "final_audit_owner": "Person responsible for adjudication and release gate.",
    "audit_notes": "Document exceptions or missing provenance.",
}
BOUNDARY_IDENTITIES = (
    ("oilseed-soy", "292320"), ("sheep", "510910"), ("sheep", "510620"),
    ("sheep", "510720"), ("nickel", "750800"), ("cocoa", "180690"),
    ("nickel", "740323"), ("cotton", "550953"),
)

README_CELLS = {
    "A1": "UpgradeBench HS92 registry: 212-record sampled outcome-blind human validation",
    "A3": (
        "This workbook is a review instrument, not evidence that human review has already occurred. "
        "All reviewer-entered fields are blank at generation."
    ),
    "A5": "Rule", "B5": "Operational requirement", "A6": "Sequence",
    "B6": (
        "The automated 610-record ledger and deterministic 212-record probability sample are "
        "generated and frozen first; human validation follows."
    ),
    "A7": "Evidence",
    "B7": (
        "Decide only from the pinned BACI HS92 official description and the frozen stage definition. "
        "External knowledge may clarify terminology but may not add absent attribution or product form."
    ),
    "A8": "Include",
    "B8": (
        "The official description explicitly identifies the focal commodity/material/species and the "
        "product form fits a declared stage. Explicit blends and focal residual/n.e.s. baskets remain "
        "eligible at whole-HS6 value; this does not identify within-code focal share, purity, or feedstock origin."
    ),
    "A9": "Exclude",
    "B9": (
        "Generic, negated, mixed-species, multi-commodity, or otherwise material-ambiguous descriptions "
        "that do not isolate the focal commodity/material are excluded rather than retained as a limitation."
    ),
    "A10": "Out of stage",
    "B10": "The focal commodity/material is observable, but the product form is outside every frozen stage.",
    "A11": "Outcome blindness",
    "B11": (
        "Reviewers must not receive trade values, labels, cohort impacts, model scores, metrics, or downstream summaries."
    ),
    "A12": "Change boundary",
    "B12": (
        "If every sampled decision and all stage definitions are confirmed, attach the retained review "
        "receipt without a computational rerun. "
        "Any decision or stage change creates a new registry version and requires full downstream rebuild."
    ),
    "A13": "Verdict",
    "B13": (
        "Yes = proposal correct; No = correction required (fill corrected fields + note); "
        "Uncertain = insufficient allowed evidence (fill note)."
    ),
    "A14": "Current review-design fact", "B14": "Value", "A15": "Decision-frame records",
    "B15": "610", "A16": "Sampled decision records", "B16": "212", "A17": "Full-frame split",
    "B17": "283 include / 228 exclude / 99 out of stage (chain-HS6 records)",
    "A18": "Stage definitions reviewed as census", "B18": "53",
    "A19": "Completed sampled human validations at generation", "B19": "0",
}
SUMMARY_LABELS = {
    "A1": "Sampled human-validation completion summary", "A3": "Measure", "B3": "Value",
    "A4": "Sampled decision records in workbook", "A5": "Completed sampled decision reviews",
    "A6": "Not-started decision reviews", "A7": "Incomplete decision reviews",
    "A8": "No verdicts (decision corrections)", "A9": "Uncertain decision reviews",
    "A10": "Completed stage-definition reviews", "A11": "No verdicts (stage-definition corrections)",
    "A12": "Uncertain stage-definition reviews", "A13": "Release action",
}
SUMMARY_FORMULAS = {
    "B4": f"COUNTA('Review'!A2:A{SAMPLED_REVIEW_LAST_ROW})",
    "B5": f'COUNTIF(\'Review\'!S2:S{SAMPLED_REVIEW_LAST_ROW},"Complete")',
    "B6": f'COUNTIF(\'Review\'!S2:S{SAMPLED_REVIEW_LAST_ROW},"Not started")',
    "B7": f'COUNTIF(\'Review\'!S2:S{SAMPLED_REVIEW_LAST_ROW},"Incomplete")',
    "B8": f'COUNTIF(\'Review\'!N2:N{SAMPLED_REVIEW_LAST_ROW},"No")',
    "B9": f'COUNTIF(\'Review\'!N2:N{SAMPLED_REVIEW_LAST_ROW},"Uncertain")',
    "B10": 'COUNTIF(\'Stage definitions\'!P2:P54,"Complete")',
    "B11": 'COUNTIF(\'Stage definitions\'!K2:K54,"No")',
    "B12": 'COUNTIF(\'Stage definitions\'!K2:K54,"Uncertain")',
    "B13": (
        'IF(OR(B8>0,B9>0,B11>0,B12>0),"Hold: adjudicate; any accepted change requires new registry '
        f'version and full rebuild",IF(AND(B5={SAMPLED_DECISION_RECORDS},B10={STAGE_DEFINITION_RECORDS}),'
        '"Sampled validation complete: issue retained receipt","Validation pending"))'
    ),
}

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_RE = re.compile(r"^([A-Z]{1,3})([1-9][0-9]*)$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{7,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class PrivateReviewError(ValueError):
    """The private workbook or adjudication record is not release-grade."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
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


def _safe_existing_file(path: Path, root: Path, role: str, *, private_only: bool = False) -> Path:
    root_resolved = root.resolve(strict=True)
    allowed = (root / "private").resolve(strict=True) if private_only else root_resolved
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PrivateReviewError(f"{role} is missing or unsafe") from exc
    if not resolved.is_relative_to(allowed) or (private_only and resolved == allowed):
        raise PrivateReviewError(f"{role} must be inside {'ROOT/private' if private_only else 'ROOT'}")
    cursor = root
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PrivateReviewError(f"{role} escapes the workspace") from exc
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PrivateReviewError(f"{role} has a symbolic-link parent/member")
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise PrivateReviewError(f"{role} must be a regular file")
    if resolved.stat().st_nlink != 1:
        raise PrivateReviewError(f"{role} hard links are forbidden")
    return resolved


def _safe_private_output_dir(path: Path, root: Path) -> Path:
    private_root = (root / "private").resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved == private_root or not resolved.is_relative_to(private_root):
        raise PrivateReviewError("output directory must be strictly below ROOT/private")
    cursor = root
    for part in resolved.relative_to(root.resolve(strict=True)).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PrivateReviewError("output directory has a symbolic-link parent/member")
    if not resolved.is_dir() or any(resolved.iterdir()):
        raise PrivateReviewError("private output directory must exist and be empty")
    return resolved


def _strict_text(value: str, role: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or _CONTROL_RE.search(value):
        raise PrivateReviewError(f"{role} contains invalid control characters")
    if value != value.strip():
        raise PrivateReviewError(f"{role} must not have leading or trailing whitespace")
    if not allow_empty and not value:
        raise PrivateReviewError(f"{role} is required")
    if len(value) > 10000:
        raise PrivateReviewError(f"{role} is unexpectedly long")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], role: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise PrivateReviewError(
            f"{role} keys changed: missing={sorted(expected_set-observed)!r}, extra={sorted(observed-expected_set)!r}"
        )


def _xml(raw: bytes, role: str) -> ET.Element:
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise PrivateReviewError(f"DTD/entity declarations are forbidden in {role}")
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise PrivateReviewError(f"invalid XML in {role}: {exc}") from exc


def _column_number(label: str) -> int:
    result = 0
    for char in label:
        result = result * 26 + ord(char) - 64
    return result


def _cell_coordinate(reference: str) -> tuple[int, int]:
    match = _CELL_RE.fullmatch(reference)
    if match is None:
        raise PrivateReviewError(f"invalid OOXML cell reference: {reference!r}")
    return _column_number(match.group(1)), int(match.group(2))


class Cell:
    __slots__ = ("reference", "value", "formula", "formula_attributes", "cell_type", "style")

    def __init__(
        self, reference: str, value: str, formula: str | None,
        formula_attributes: Mapping[str, str], cell_type: str | None, style: str | None,
    ) -> None:
        self.reference = reference
        self.value = value
        self.formula = formula
        self.formula_attributes = dict(formula_attributes)
        self.cell_type = cell_type
        self.style = style


class Workbook:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sheets: dict[str, dict[str, Cell]] = {}
        self.sheet_targets: dict[str, str] = {}
        self.date1904 = False
        self._load()

    def _load(self) -> None:
        if self.path.is_symlink() or not self.path.is_file() or self.path.suffix.lower() != ".xlsx":
            raise PrivateReviewError("workbook must be a non-symlink .xlsx regular file")
        mode = self.path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise PrivateReviewError("workbook is not a regular file")
        try:
            archive = zipfile.ZipFile(self.path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise PrivateReviewError(f"workbook is not a valid OOXML ZIP: {exc}") from exc
        with archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
                raise PrivateReviewError("duplicate or case-colliding OOXML package members are forbidden")
            total = 0
            for info in infos:
                pure = PurePosixPath(info.filename)
                if (
                    not info.filename or "\\" in info.filename or pure.is_absolute()
                    or ".." in pure.parts or pure.as_posix() != info.filename
                ):
                    raise PrivateReviewError(f"unsafe OOXML package member: {info.filename!r}")
                total += info.file_size
                if info.file_size > 32 * 1024 * 1024 or total > 64 * 1024 * 1024:
                    raise PrivateReviewError("OOXML package exceeds the private-review safety limit")
            required = {"[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
            if not required.issubset(names):
                raise PrivateReviewError("OOXML package is missing required workbook members")
            forbidden_parts = (
                "xl/vbaproject", "xl/externallinks/", "xl/embeddings/", "xl/activex/",
                "xl/drawings/", "xl/media/", "xl/comments", "xl/threadedcomments/",
                "customxml/",
            )
            for name in names:
                lowered = name.casefold()
                if any(lowered.startswith(prefix) for prefix in forbidden_parts):
                    raise PrivateReviewError(f"macro/external/embedded content is forbidden: {name}")
            content_types = archive.read("[Content_Types].xml").lower()
            for marker in (b"macroenabled", b"vbaproject", b"externalLink".lower(), b"oleobject", b"activex"):
                if marker.lower() in content_types:
                    raise PrivateReviewError("macro/external/embedded content type is forbidden")
            for name in names:
                if not name.endswith(".rels"):
                    continue
                rel_root = _xml(archive.read(name), name)
                for rel in rel_root.findall(f"{{{_NS_REL_PKG}}}Relationship"):
                    rel_type = rel.attrib.get("Type", "").casefold()
                    if rel.attrib.get("TargetMode", "").casefold() == "external":
                        raise PrivateReviewError(f"external OOXML relationship is forbidden: {name}")
                    if any(token in rel_type for token in ("externallink", "vbaproject", "oleobject", "activex", "hyperlink")):
                        raise PrivateReviewError(f"external/active OOXML relationship is forbidden: {name}")

            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in names:
                shared_root = _xml(archive.read("xl/sharedStrings.xml"), "shared strings")
                for item in shared_root.findall(f"{{{_NS_MAIN}}}si"):
                    shared_strings.append("".join(node.text or "" for node in item.iter(f"{{{_NS_MAIN}}}t")))

            wb_root = _xml(archive.read("xl/workbook.xml"), "workbook")
            if wb_root.find(f"{{{_NS_MAIN}}}definedNames") is not None:
                raise PrivateReviewError("workbook defined names are forbidden")
            wb_pr = wb_root.find(f"{{{_NS_MAIN}}}workbookPr")
            self.date1904 = wb_pr is not None and wb_pr.attrib.get("date1904") in {"1", "true", "True"}
            rel_root = _xml(archive.read("xl/_rels/workbook.xml.rels"), "workbook relationships")
            rels = {
                rel.attrib["Id"]: rel.attrib.get("Target", "")
                for rel in rel_root.findall(f"{{{_NS_REL_PKG}}}Relationship")
            }
            sheet_nodes = wb_root.findall(f"{{{_NS_MAIN}}}sheets/{{{_NS_MAIN}}}sheet")
            observed_names = tuple(node.attrib.get("name", "") for node in sheet_nodes)
            if observed_names != SHEET_NAMES:
                raise PrivateReviewError(f"worksheet inventory/order changed: {observed_names!r}")
            for node in sheet_nodes:
                if node.attrib.get("state", "visible") != "visible":
                    raise PrivateReviewError(f"hidden worksheet is forbidden: {node.attrib.get('name')}")
                rel_id = node.attrib.get(f"{{{_NS_REL_DOC}}}id")
                target = rels.get(rel_id or "", "")
                if target.startswith("/"):
                    target = target[1:]
                elif not target.startswith("xl/"):
                    target = "xl/" + target
                pure = PurePosixPath(target)
                if ".." in pure.parts or target not in names or not target.startswith("xl/worksheets/"):
                    raise PrivateReviewError(f"unsafe worksheet target for {node.attrib.get('name')}")
                name = node.attrib["name"]
                self.sheet_targets[name] = target
                self.sheets[name] = self._read_sheet(
                    archive, target, shared_strings, sheet_name=name
                )

            self._validate_table_parts(archive, names)

    def _read_sheet(
        self,
        archive: zipfile.ZipFile,
        target: str,
        shared_strings: list[str],
        *,
        sheet_name: str,
    ) -> dict[str, Cell]:
        root = _xml(archive.read(target), target)
        self._validate_sheet_controls(root, sheet_name)
        result: dict[str, Cell] = {}
        for row in root.findall(f"{{{_NS_MAIN}}}sheetData/{{{_NS_MAIN}}}row"):
            raw_row = row.attrib.get("r")
            if raw_row is None or not raw_row.isdigit() or int(raw_row) < 1:
                raise PrivateReviewError(f"invalid row coordinate in {target}")
            if row.attrib.get("hidden") in {"1", "true", "True"}:
                raise PrivateReviewError(f"hidden row is forbidden: {sheet_name}!{raw_row}")
            seen_columns: set[int] = set()
            for cell in row.findall(f"{{{_NS_MAIN}}}c"):
                reference = cell.attrib.get("r", "")
                column, row_number = _cell_coordinate(reference)
                if row_number != int(raw_row) or column in seen_columns or reference in result:
                    raise PrivateReviewError(f"duplicate or inconsistent cell coordinate: {target}!{reference}")
                seen_columns.add(column)
                formula_node = cell.find(f"{{{_NS_MAIN}}}f")
                formula = None if formula_node is None else (formula_node.text or "")
                formula_attributes = {} if formula_node is None else formula_node.attrib
                value_node = cell.find(f"{{{_NS_MAIN}}}v")
                value = "" if value_node is None else (value_node.text or "")
                cell_type = cell.attrib.get("t")
                if cell_type == "s":
                    try:
                        index = int(value)
                        value = shared_strings[index]
                    except (ValueError, IndexError) as exc:
                        raise PrivateReviewError(f"invalid shared-string index: {target}!{reference}") from exc
                elif cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{{{_NS_MAIN}}}t"))
                elif cell_type in {"e", "d"}:
                    raise PrivateReviewError(f"unsupported/error cell type: {target}!{reference}")
                result[reference] = Cell(
                    reference, value, formula, formula_attributes, cell_type, cell.attrib.get("s")
                )
        return result

    def _validate_sheet_controls(self, root: ET.Element, sheet_name: str) -> None:
        columns = root.find(f"{{{_NS_MAIN}}}cols")
        if columns is not None:
            for column in columns.findall(f"{{{_NS_MAIN}}}col"):
                if column.attrib.get("hidden") in {"1", "true", "True"}:
                    raise PrivateReviewError(f"hidden column is forbidden: {sheet_name}")
        expected_validations = {
            "Review": [
                (f"N2:N{SAMPLED_REVIEW_LAST_ROW}", '"Yes,No,Uncertain"'),
                (f"O2:O{SAMPLED_REVIEW_LAST_ROW}", '"include,exclude,out_of_stage"'),
                (
                    f"P2:P{SAMPLED_REVIEW_LAST_ROW}",
                    '"exp_live,exp_meat,exp_rawskin,exp_woolraw,exp_woolgrease,exp_wooltop,'
                    'exp_woolyarn,exp_cottonraw,exp_cottonwaste,exp_cottonprepared,exp_cottonyarn,'
                    'exp_cottonsewthread,exp_cottonfabric,exp_cottonknitfabric,exp_cottonapparel_woven,'
                    'exp_cottonapparel_knit,exp_cottonhomewares,exp_aluminium_ore,exp_scrap,'
                    'exp_aluminium_hydroxide,exp_aluminium_oxide,exp_corundum,exp_unwrought,'
                    'exp_semis_barrod,exp_semis_wire,exp_semis_plate,exp_semis_foil,exp_semis_tube,'
                    'exp_nickel_ore,exp_nickel_matte,exp_nickel_intermediate,exp_ferronickel,'
                    'exp_unwrought,exp_powder,exp_bars_wire,exp_plates_foil,exp_tubes,exp_other,'
                    'exp_nickel_salts,exp_cocoabean,exp_cocoawaste,exp_cocoapaste,exp_cocoabutter,'
                    'exp_cocoapowder_unsw,exp_cocoapowder_sw,exp_cocoa_prep_bulk,'
                    'exp_cocoa_prep_blocks_bars,exp_cocoa_prep_other,exp_soybean,exp_soymeal,'
                    'exp_soyoil_crude,exp_soyoil_noncrude,exp_soyflour_meal"',
                ),
                (f"R2:R{SAMPLED_REVIEW_LAST_ROW}", '"Yes,No"'),
            ],
            "Stage definitions": [
                ("K2:K54", '"Yes,No,Uncertain"'),
                ("O2:O54", '"Yes,No"'),
            ],
        }.get(sheet_name, [])
        observed_validations: list[tuple[str, str]] = []
        container = root.find(f"{{{_NS_MAIN}}}dataValidations")
        if container is not None:
            for validation in container.findall(f"{{{_NS_MAIN}}}dataValidation"):
                forbidden_text_attributes = {"promptTitle", "prompt", "errorTitle", "error"}
                if forbidden_text_attributes.intersection(validation.attrib):
                    raise PrivateReviewError(f"data-validation message content is forbidden: {sheet_name}")
                formula1 = validation.find(f"{{{_NS_MAIN}}}formula1")
                formula2 = validation.find(f"{{{_NS_MAIN}}}formula2")
                if validation.attrib.get("type") != "list" or formula1 is None or formula2 is not None:
                    raise PrivateReviewError(f"unexpected data validation: {sheet_name}")
                observed_validations.append(
                    (validation.attrib.get("sqref", ""), formula1.text or "")
                )
        if observed_validations != expected_validations:
            raise PrivateReviewError(f"data-validation inventory/content changed: {sheet_name}")

        expected_cf = []
        if sheet_name == "Review":
            expected_cf = [
                (f"S2:S{SAMPLED_REVIEW_LAST_ROW}", "containsText", "0", "1", "Complete"),
                (f"S2:S{SAMPLED_REVIEW_LAST_ROW}", "containsText", "1", "2", "Incomplete"),
                (f"S2:S{SAMPLED_REVIEW_LAST_ROW}", "containsText", "2", "3", "Not started"),
            ]
        observed_cf: list[tuple[str, str, str, str, str]] = []
        for formatting in root.findall(f"{{{_NS_MAIN}}}conditionalFormatting"):
            sqref = formatting.attrib.get("sqref", "")
            for rule in formatting.findall(f"{{{_NS_MAIN}}}cfRule"):
                if list(rule):
                    raise PrivateReviewError(f"conditional-format formulas are forbidden: {sheet_name}")
                observed_cf.append(
                    (
                        sqref,
                        rule.attrib.get("type", ""),
                        rule.attrib.get("dxfId", ""),
                        rule.attrib.get("priority", ""),
                        rule.attrib.get("text", ""),
                    )
                )
        if observed_cf != expected_cf:
            raise PrivateReviewError(f"conditional-format inventory/content changed: {sheet_name}")

    def _validate_table_parts(self, archive: zipfile.ZipFile, names: list[str]) -> None:
        expected = {
            "Review": (
                "RegistryReviewTable",
                f"A1:S{SAMPLED_REVIEW_LAST_ROW}",
                REVIEW_HEADERS,
            ),
            "Stage definitions": ("StageReviewTable", "A1:P54", STAGE_HEADERS),
            "Audit settings": (
                "AuditSettingsTable",
                f"A1:D{len(AUDIT_FIELDS) + 1}",
                AUDIT_HEADERS,
            ),
            "Boundary cases": ("BoundaryCasesTable", "A1:F9", BOUNDARY_HEADERS),
        }
        for sheet_name, (table_name, table_ref, headers) in expected.items():
            target = self.sheet_targets[sheet_name]
            rel_name = str(PurePosixPath(target).parent / "_rels" / (PurePosixPath(target).name + ".rels"))
            if rel_name not in names:
                raise PrivateReviewError(f"table relationship is missing for {sheet_name}")
            rel_root = _xml(archive.read(rel_name), rel_name)
            table_targets: list[str] = []
            for rel in rel_root.findall(f"{{{_NS_REL_PKG}}}Relationship"):
                if not rel.attrib.get("Type", "").endswith("/table"):
                    raise PrivateReviewError(f"unexpected worksheet relationship for {sheet_name}")
                raw = rel.attrib.get("Target", "")
                resolved = raw.lstrip("/") if raw.startswith("/") else str(PurePosixPath(target).parent / raw)
                table_targets.append(str(PurePosixPath(resolved)))
            if len(table_targets) != 1 or table_targets[0] not in names:
                raise PrivateReviewError(f"table target changed for {sheet_name}")
            root = _xml(archive.read(table_targets[0]), table_targets[0])
            if root.attrib.get("name") != table_name or root.attrib.get("displayName") != table_name:
                raise PrivateReviewError(f"table identity changed for {sheet_name}")
            if root.attrib.get("ref") != table_ref:
                raise PrivateReviewError(f"table range changed for {sheet_name}")
            columns = root.findall(f"{{{_NS_MAIN}}}tableColumns/{{{_NS_MAIN}}}tableColumn")
            if tuple(item.attrib.get("name") for item in columns) != headers:
                raise PrivateReviewError(f"table headers changed for {sheet_name}")


def _cell(sheet: Mapping[str, Cell], reference: str) -> Cell:
    return sheet.get(reference, Cell(reference, "", None, {}, None, None))


def _ensure_bounds(sheet: Mapping[str, Cell], role: str, max_column: int, max_row: int) -> None:
    for reference, cell in sheet.items():
        column, row = _cell_coordinate(reference)
        if (column > max_column or row > max_row) and (cell.value or cell.formula is not None):
            raise PrivateReviewError(f"unexpected extra cell in {role}: {reference}")


def _formula_template(value: str) -> str:
    return re.sub(r"(?<=[A-Z])([1-9][0-9]*)", "{row}", value)


def _expected_formula_maps() -> dict[str, dict[str, str]]:
    expected_by_sheet: dict[str, dict[str, str]] = {name: {} for name in SHEET_NAMES}
    for row in range(2, SAMPLED_REVIEW_LAST_ROW + 1):
        expected_by_sheet["Review"][f"S{row}"] = (
            f'IF(N{row}="","Not started",IF(AND(L{row}<>"",M{row}<>"",R{row}="Yes",OR('
            f'AND(N{row}="Yes",O{row}="",P{row}=""),AND(N{row}="No",O{row}<>"",Q{row}<>"",'
            f'IF(O{row}="include",AND(P{row}<>"",ISNUMBER(SEARCH(";"&P{row}&";",";"&SUBSTITUTE(H{row}," ","")&";"))),'
            f'P{row}="")),AND(N{row}="Uncertain",Q{row}<>""))),"Complete","Incomplete"))'
        )
    for row in range(2, 55):
        expected_by_sheet["Stage definitions"][f"P{row}"] = (
            f'IF(K{row}="","Not started",IF(AND(I{row}<>"",J{row}<>"",O{row}="Yes",OR('
            f'AND(K{row}="Yes",L{row}="",M{row}=""),AND(K{row}="No",L{row}<>"",M{row}<>"",'
            f'N{row}<>""),AND(K{row}="Uncertain",N{row}<>""))),"Complete","Incomplete"))'
        )
    expected_by_sheet["Review summary"] = dict(SUMMARY_FORMULAS)
    return expected_by_sheet


def _validate_formulas(workbook: Workbook) -> None:
    expected_by_sheet = _expected_formula_maps()
    for sheet_name, cells in workbook.sheets.items():
        expected = expected_by_sheet[sheet_name]
        observed_formula_refs = {reference for reference, cell in cells.items() if cell.formula is not None}
        if observed_formula_refs != set(expected):
            raise PrivateReviewError(
                f"formula inventory changed in {sheet_name}: missing={sorted(set(expected)-observed_formula_refs)!r}, "
                f"extra={sorted(observed_formula_refs-set(expected))!r}"
            )
        shared_masters: dict[str, tuple[str, str]] = {}
        for reference in observed_formula_refs:
            cell = cells[reference]
            if cell.formula_attributes.get("t") == "shared" and cell.formula:
                si = cell.formula_attributes.get("si")
                if si is None or si in shared_masters:
                    raise PrivateReviewError(f"invalid shared formula master in {sheet_name}!{reference}")
                shared_masters[si] = (reference, cell.formula)
        for reference, wanted in expected.items():
            cell = cells[reference]
            observed = cell.formula
            if observed == wanted:
                continue
            if cell.formula_attributes.get("t") == "shared" and not observed:
                si = cell.formula_attributes.get("si")
                master = shared_masters.get(si or "")
                if master and _formula_template(master[1]) == _formula_template(wanted):
                    continue
            raise PrivateReviewError(f"formula tampering detected: {sheet_name}!{reference}")


def _instrument_semantics(workbook: Workbook) -> dict[str, Any]:
    """Return immutable reviewer-visible semantics, excluding entered/cached values."""

    audit_rows: list[list[str | None]] = []
    mutable = {
        "primary_reviewer", "review_start_date", "review_end_date",
        "outcome_access_declaration", "final_audit_owner", "audit_notes",
    }
    for row_number, field in enumerate(AUDIT_FIELDS, start=2):
        row: list[str | None] = list(_value_row(workbook.sheets["Audit settings"], row_number, 4))
        if field in mutable:
            row[1] = None
        audit_rows.append(row)
    review_proposals: list[list[str]] = []
    for row in range(2, SAMPLED_REVIEW_LAST_ROW + 1):
        proposal = _value_row(workbook.sheets["Review"], row, 11)
        # Excel may serialize a six-digit HS code either as text (``010410``)
        # or as an unformatted numeric value (``10410``).  The visible code is
        # the same, so normalize it before binding immutable semantics.
        proposal[2] = proposal[2].zfill(6) if proposal[2].isdigit() else proposal[2]
        review_proposals.append(proposal)

    return {
        "schema_version": "upgrade-bench/registry-review-instrument-semantics/2",
        "sheet_inventory": list(SHEET_NAMES),
        "readme": {key: README_CELLS[key] for key in sorted(README_CELLS)},
        "review_headers": list(REVIEW_HEADERS),
        "review_proposals": review_proposals,
        "stage_headers": list(STAGE_HEADERS),
        "stage_proposals": [
            _value_row(workbook.sheets["Stage definitions"], row, 8) for row in range(2, 55)
        ],
        "audit_headers": list(AUDIT_HEADERS),
        "audit_settings": audit_rows,
        "boundary_headers": list(BOUNDARY_HEADERS),
        "boundary_cases": [
            _value_row(workbook.sheets["Boundary cases"], row, 6) for row in range(2, 10)
        ],
        "summary_labels": {key: SUMMARY_LABELS[key] for key in sorted(SUMMARY_LABELS)},
        "formulas": _expected_formula_maps(),
    }


def _audit_settings_semantics(workbook: Workbook) -> dict[str, Any]:
    values: dict[str, str | None] = {}
    mutable = {
        "primary_reviewer", "review_start_date", "review_end_date",
        "outcome_access_declaration", "final_audit_owner", "audit_notes",
    }
    for row_number, field in enumerate(AUDIT_FIELDS, start=2):
        values[field] = None if field in mutable else _cell(
            workbook.sheets["Audit settings"], f"B{row_number}"
        ).value
    return values


def _read_json(path: Path, role: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=public_receipt._object_no_duplicates,
            parse_constant=public_receipt._reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, public_receipt.HumanReviewReceiptError) as exc:
        raise PrivateReviewError(f"cannot read {role}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrivateReviewError(f"{role} must be a JSON object")
    return payload


def _load_ledger(root: Path) -> list[dict[str, str]]:
    path = root / "chains/evidence/registry_full_audit_ledger.csv"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError as exc:
        raise PrivateReviewError(f"cannot read current audit ledger: {exc}") from exc
    required = {
        "chain_id", "code", "decision", "stage", "candidate_source", "description",
        "rationale_category", "rationale",
    }
    if (
        reader.fieldnames is None
        or not required.issubset(reader.fieldnames)
        or len(rows) != FRAME_DECISION_RECORDS
    ):
        raise PrivateReviewError("current audit ledger schema/count changed")
    identities = [(row["chain_id"], row["code"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise PrivateReviewError("current audit ledger has duplicate chain/HS6 records")
    if len({row["code"] for row in rows}) != sample_plan.FRAME_UNIQUE_HS6:
        raise PrivateReviewError("current audit ledger unique-HS6 count changed")
    return rows


def _load_sample(root: Path) -> dict[str, Any]:
    path = root / sample_plan.OUTPUT_PATH.relative_to(sample_plan.ROOT)
    try:
        return sample_plan.load_plan(path, root=root)
    except (OSError, ValueError) as exc:
        raise PrivateReviewError(f"cannot validate frozen human-validation sample: {exc}") from exc


def _expected_review_rows(root: Path, ledger: list[dict[str, str]]) -> list[list[str]]:
    evidence = _read_json(root / "chains/evidence/registry_evidence.json", "registry evidence")
    chains = evidence.get("chains")
    if not isinstance(chains, dict):
        raise PrivateReviewError("registry evidence chain definitions are missing")
    sample = _load_sample(root)
    selected_ids = {row["record_id"] for row in sample["selected_records"]}
    result: list[list[str]] = []
    for row in ledger:
        chain = row["chain_id"]
        record_id = f"CODE-{chain}-{row['code']}"
        if record_id not in selected_ids:
            continue
        chain_evidence = chains.get(chain)
        if not isinstance(chain_evidence, dict) or not isinstance(chain_evidence.get("stage_definitions"), dict):
            raise PrivateReviewError(f"stage definitions missing for {chain}")
        definitions = chain_evidence["stage_definitions"]
        allowed = list(definitions)
        stage = row["stage"]
        if row["decision"] == "include" and stage in definitions:
            definition = definitions[stage]["canonical_definition"]
        elif row["decision"] == "out_of_stage":
            definition = "No assignment: explicit focal commodity, product form outside frozen stages."
        else:
            definition = "No active stage assignment under strict observable attribution."
        result.append([
            record_id, chain, row["code"], row["description"],
            row["candidate_source"], row["decision"], stage, "; ".join(allowed), definition,
            row["rationale_category"], row["rationale"],
        ])
    if len(result) != SAMPLED_DECISION_RECORDS:
        raise PrivateReviewError("frozen human-validation sample count changed")
    if {row[0] for row in result} != selected_ids:
        raise PrivateReviewError("workbook sample keys differ from frozen sample artifact")
    return result


def _expected_stage_rows(root: Path) -> list[list[str]]:
    evidence = _read_json(root / "chains/evidence/registry_evidence.json", "registry evidence")
    chains = evidence.get("chains")
    if not isinstance(chains, dict):
        raise PrivateReviewError("registry evidence chain definitions are missing")
    rows: list[list[str]] = []
    for chain in chains:
        if chain not in public_receipt.CHAIN_IDS:
            raise PrivateReviewError(f"unexpected chain in registry evidence: {chain}")
        registry = _read_json(root / f"chains/{chain}.json", f"{chain} registry")
        definitions = chains[chain].get("stage_definitions")
        stages = registry.get("stages")
        upstream = registry.get("upstream")
        if not isinstance(definitions, dict) or not isinstance(stages, dict) or not isinstance(upstream, list):
            raise PrivateReviewError(f"stage metadata missing for {chain}")
        if list(definitions) != list(stages):
            raise PrivateReviewError(f"stage order/inventory differs between evidence and registry: {chain}")
        for stage, definition in definitions.items():
            rows.append([
                f"STAGE-{chain}-{stage}", chain, stage, definition["canonical_definition"],
                definition["specificity"], definition["fit_rule"], str(len(stages[stage])),
                "upstream_source" if stage in upstream else "prediction_target",
            ])
    if len(rows) != STAGE_DEFINITION_RECORDS:
        raise PrivateReviewError("current stage-definition count changed")
    return rows


def _value_row(sheet: Mapping[str, Cell], row: int, count: int) -> list[str]:
    values: list[str] = []
    for number in range(1, count + 1):
        label = ""
        current = number
        while current:
            current, remainder = divmod(current - 1, 26)
            label = chr(65 + remainder) + label
        values.append(_cell(sheet, f"{label}{row}").value)
    return values


def _verify_static_sheets(workbook: Workbook, ledger: list[dict[str, str]]) -> None:
    _ensure_bounds(workbook.sheets["README"], "README", 6, 19)
    _ensure_bounds(workbook.sheets["Review summary"], "Review summary", 4, 13)
    _ensure_bounds(workbook.sheets["Boundary cases"], "Boundary cases", 6, 9)
    readme = workbook.sheets["README"]
    observed_nonempty = {ref: cell.value for ref, cell in readme.items() if cell.value}
    if observed_nonempty != README_CELLS:
        raise PrivateReviewError("README instrument content changed")
    summary = workbook.sheets["Review summary"]
    for reference, value in SUMMARY_LABELS.items():
        if _cell(summary, reference).value != value:
            raise PrivateReviewError(f"review-summary label changed: {reference}")
    for ref, cell in summary.items():
        if cell.value and ref not in SUMMARY_LABELS and ref not in SUMMARY_FORMULAS:
            raise PrivateReviewError(f"unexpected review-summary content: {ref}")
    boundary = workbook.sheets["Boundary cases"]
    if tuple(_value_row(boundary, 1, 6)) != BOUNDARY_HEADERS:
        raise PrivateReviewError("boundary-case headers changed")
    ledger_map = {(row["chain_id"], row["code"]): row for row in ledger}
    for row_number, identity in enumerate(BOUNDARY_IDENTITIES, start=2):
        source = ledger_map.get(identity)
        if source is None:
            raise PrivateReviewError(f"boundary case is absent from current ledger: {identity}")
        expected = [
            identity[0], identity[1], source["description"], source["decision"], source["stage"], source["rationale"],
        ]
        observed = _value_row(boundary, row_number, 6)
        if observed[1].isdigit():
            observed[1] = observed[1].zfill(6)
        if observed != expected:
            raise PrivateReviewError(f"boundary-case content changed at row {row_number}")


def _parse_iso_or_excel_date(cell: Cell, date1904: bool, role: str) -> str:
    raw = _strict_text(cell.value, role)
    try:
        parsed = date.fromisoformat(raw)
        if parsed.isoformat() == raw:
            return raw
    except ValueError:
        pass
    try:
        serial = Decimal(raw)
    except InvalidOperation as exc:
        raise PrivateReviewError(f"{role} must be an ISO date or integral Excel date") from exc
    minimum = 0 if date1904 else 1
    if serial != serial.to_integral_value() or serial < minimum:
        raise PrivateReviewError(f"{role} must be an ISO date or integral Excel date")
    integer = int(serial)
    if not date1904 and integer == 60:
        raise PrivateReviewError(f"{role} uses Excel's nonexistent 1900-02-29")
    if date1904:
        base = date(1904, 1, 1)
    else:
        base = date(1899, 12, 31) if integer < 60 else date(1899, 12, 30)
    try:
        return (base + timedelta(days=integer)).isoformat()
    except OverflowError as exc:
        raise PrivateReviewError(f"{role} is outside the supported date range") from exc


def _verify_audit_settings(
    workbook: Workbook,
    root: Path,
    ledger: list[dict[str, str]],
    freeze: Mapping[str, Any],
) -> dict[str, str]:
    sheet = workbook.sheets["Audit settings"]
    _ensure_bounds(sheet, "Audit settings", 4, len(AUDIT_FIELDS) + 1)
    if tuple(_value_row(sheet, 1, 4)) != AUDIT_HEADERS:
        raise PrivateReviewError("audit-settings headers changed")
    settings: dict[str, str] = {}
    for row_number, field in enumerate(AUDIT_FIELDS, start=2):
        values = _value_row(sheet, row_number, 4)
        if values[0] != field or values[2] != ("Yes" if AUDIT_REQUIRED[field] else "No"):
            raise PrivateReviewError(f"audit-settings row identity/required flag changed: {row_number}")
        if values[3] != AUDIT_DESCRIPTIONS[field]:
            raise PrivateReviewError(f"audit-settings description changed: {field}")
        settings[field] = values[1]
    evidence_path = root / "chains/evidence/registry_evidence.json"
    evidence = _read_json(evidence_path, "registry evidence")
    sample_path = root / sample_plan.OUTPUT_PATH.relative_to(sample_plan.ROOT)
    sample = _load_sample(root)
    expected_static = {
        "registry_evidence_schema": str(evidence.get("schema_version", "")),
        "registry_evidence_sha256": _sha256_file(evidence_path),
        "full_ledger_sha256": _sha256_file(root / "chains/evidence/registry_full_audit_ledger.csv"),
        "review_codebook_sha256": freeze["review_inputs_sha256"]["docs/REGISTRY_REVIEW_CODEBOOK.md"],
        "recall_rule_sha256": _sha256_file(root / "chains/evidence/registry_candidate_recall_rule.json"),
        "source_dictionary_member_sha256": str(evidence.get("source", {}).get("source_metadata_member_sha256", "")),
        "selected_metadata_sha256": _sha256_file(root / "chains/evidence/hs92_selected_product_codes.csv"),
        "human_validation_sample_sha256": _sha256_file(sample_path),
        "sample_plan_id": sample["plan_id"],
        "decision_records": str(len(ledger)),
        "sampled_decision_records": str(len(sample["selected_records"])),
        "unique_hs6": str(len({row["code"] for row in ledger})),
        "human_reviews_completed_at_generation": "0",
    }
    for field, wanted in expected_static.items():
        if settings[field] != wanted:
            raise PrivateReviewError(f"stale or altered audit setting: {field}")
    audit_id = _strict_text(settings["audit_id"], "audit_id")
    if _AUDIT_ID_RE.fullmatch(audit_id) is None:
        raise PrivateReviewError("audit_id is invalid")
    generated = date.fromisoformat(_strict_text(settings["workbook_generated_date"], "workbook_generated_date"))
    start_row = AUDIT_FIELDS.index("review_start_date") + 2
    end_row = AUDIT_FIELDS.index("review_end_date") + 2
    start = date.fromisoformat(
        _parse_iso_or_excel_date(_cell(sheet, f"B{start_row}"), workbook.date1904, "review_start_date")
    )
    end = date.fromisoformat(
        _parse_iso_or_excel_date(_cell(sheet, f"B{end_row}"), workbook.date1904, "review_end_date")
    )
    if not generated <= start <= end:
        raise PrivateReviewError("workbook generation/start/end dates are inconsistent")
    settings["review_start_date"] = start.isoformat()
    settings["review_end_date"] = end.isoformat()
    _strict_text(settings["primary_reviewer"], "primary_reviewer")
    _strict_text(settings["final_audit_owner"], "final_audit_owner")
    _strict_text(settings["audit_notes"], "audit_notes", allow_empty=True)
    if settings["outcome_access_declaration"] != OUTCOME_ACCESS_DECLARATION:
        raise PrivateReviewError("audit-level outcome-access declaration is missing or non-canonical")
    return settings


def build_pre_review_freeze_payload(
    workbook_path: Path,
    *,
    frozen_date: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build, but never write, the canonical pre-review freeze payload."""

    root = Path(root)
    workbook = Workbook(Path(workbook_path))
    _validate_formulas(workbook)
    ledger = _load_ledger(root)
    sample = _load_sample(root)
    _verify_static_sheets(workbook, ledger)
    decision_sheet = workbook.sheets["Review"]
    stage_sheet = workbook.sheets["Stage definitions"]
    if tuple(_value_row(decision_sheet, 1, 19)) != REVIEW_HEADERS:
        raise PrivateReviewError("review headers changed")
    if tuple(_value_row(stage_sheet, 1, 16)) != STAGE_HEADERS:
        raise PrivateReviewError("stage-definition headers changed")
    for row_number, expected in enumerate(_expected_review_rows(root, ledger), start=2):
        observed = _value_row(decision_sheet, row_number, 19)
        observed[2] = observed[2].zfill(6) if observed[2].isdigit() else observed[2]
        if observed[:11] != expected:
            raise PrivateReviewError(f"pre-review proposal mismatch: Review row {row_number}")
        if any(observed[index] for index in range(11, 18)):
            raise PrivateReviewError(f"pre-review instrument already has reviewer entries: Review row {row_number}")
    for row_number, expected in enumerate(_expected_stage_rows(root), start=2):
        observed = _value_row(stage_sheet, row_number, 16)
        if observed[:8] != expected:
            raise PrivateReviewError(f"pre-review proposal mismatch: Stage row {row_number}")
        if any(observed[index] for index in range(8, 15)):
            raise PrivateReviewError(f"pre-review instrument already has stage-review entries: row {row_number}")

    audit_sheet = workbook.sheets["Audit settings"]
    if tuple(_value_row(audit_sheet, 1, 4)) != AUDIT_HEADERS:
        raise PrivateReviewError("audit-settings headers changed")
    settings: dict[str, str] = {}
    for row_number, field in enumerate(AUDIT_FIELDS, start=2):
        values = _value_row(audit_sheet, row_number, 4)
        if (
            values[0] != field
            or values[2] != ("Yes" if AUDIT_REQUIRED[field] else "No")
            or values[3] != AUDIT_DESCRIPTIONS[field]
        ):
            raise PrivateReviewError(f"pre-review audit-settings structure changed: {field}")
        settings[field] = values[1]
    for field in (
        "primary_reviewer", "review_start_date", "review_end_date",
        "outcome_access_declaration", "final_audit_owner", "audit_notes",
    ):
        if settings[field]:
            raise PrivateReviewError(f"pre-review mutable audit setting is already populated: {field}")
    if _AUDIT_ID_RE.fullmatch(settings["audit_id"]) is None:
        raise PrivateReviewError("pre-review audit_id is invalid")
    date.fromisoformat(settings["workbook_generated_date"])
    date.fromisoformat(frozen_date)
    evidence_path = root / "chains/evidence/registry_evidence.json"
    evidence = _read_json(evidence_path, "registry evidence")
    expected_static = {
        "registry_evidence_schema": str(evidence.get("schema_version", "")),
        "registry_evidence_sha256": _sha256_file(evidence_path),
        "full_ledger_sha256": _sha256_file(root / "chains/evidence/registry_full_audit_ledger.csv"),
        "recall_rule_sha256": _sha256_file(root / "chains/evidence/registry_candidate_recall_rule.json"),
        "source_dictionary_member_sha256": str(evidence.get("source", {}).get("source_metadata_member_sha256", "")),
        "selected_metadata_sha256": _sha256_file(root / "chains/evidence/hs92_selected_product_codes.csv"),
        "human_validation_sample_sha256": _sha256_file(
            root / sample_plan.OUTPUT_PATH.relative_to(sample_plan.ROOT)
        ),
        "sample_plan_id": sample["plan_id"],
        "decision_records": str(FRAME_DECISION_RECORDS),
        "sampled_decision_records": str(SAMPLED_DECISION_RECORDS),
        "unique_hs6": str(sample_plan.FRAME_UNIQUE_HS6),
        "human_reviews_completed_at_generation": "0",
    }
    for field, wanted in expected_static.items():
        if settings[field] != wanted:
            raise PrivateReviewError(f"stale pre-review audit setting: {field}")
    current_codebook_sha = _sha256_file(root / "docs/REGISTRY_REVIEW_CODEBOOK.md")
    if settings["review_codebook_sha256"] != current_codebook_sha:
        raise PrivateReviewError("stale pre-review audit setting: review_codebook_sha256")

    review_hashes = {
        relative: _sha256_file(public_receipt._safe_source(root, relative))
        for relative in public_receipt.FREEZE_REVIEW_INPUT_PATHS
    }
    projection = public_receipt.construct_projection(root)
    scan = _read_json(root / "chains/evidence/registry_full_scan_receipt.json", "full scan receipt")
    return {
        "schema_version": public_receipt.FREEZE_SCHEMA,
        "status": public_receipt.STATUS_FROZEN_PRE_REVIEW,
        "audit_id": settings["audit_id"],
        "frozen_date": frozen_date,
        "benchmark_identity": {
            "benchmark_version": scan["benchmark_identity"]["benchmark_version"],
            "registry_snapshot": scan["benchmark_identity"]["data_snapshot"],
            "rule_id": evidence["rule_id"],
            "source_dictionary_member_sha256": evidence["source"]["source_metadata_member_sha256"],
        },
        "scope": {
            "decision_frame_records": FRAME_DECISION_RECORDS,
            "sampled_decision_records": SAMPLED_DECISION_RECORDS,
            "sampled_unique_hs6": sample["sample"]["unique_hs6"],
            "unique_hs6": sample_plan.FRAME_UNIQUE_HS6,
            "stage_definition_records": STAGE_DEFINITION_RECORDS,
        },
        "sampling_plan": {
            "path": sample_plan.OUTPUT_PATH.relative_to(sample_plan.ROOT).as_posix(),
            "sha256": _sha256_file(root / sample_plan.OUTPUT_PATH.relative_to(sample_plan.ROOT)),
            "plan_id": sample["plan_id"],
            "record_ids_sha256": sample["sample"]["record_ids_sha256"],
        },
        "review_inputs_sha256": review_hashes,
        "instrument": {
            "source_workbook_sha256": _sha256_file(Path(workbook_path)),
            "immutable_semantics_sha256": _sha256_bytes(_canonical_json_bytes(_instrument_semantics(workbook))),
            "audit_settings_sha256": _sha256_bytes(_canonical_json_bytes(_audit_settings_semantics(workbook))),
        },
        "construct": {
            "schema_version": public_receipt.CONSTRUCT_SCHEMA,
            "fields": list(public_receipt.CONSTRUCT_FIELDS),
            "sha256": public_receipt.construct_sha256(projection),
        },
    }


def _review_completion(
    workbook: Workbook, root: Path, ledger: list[dict[str, str]], settings: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[tuple[str, str], dict[str, Any]]]:
    decision_sheet = workbook.sheets["Review"]
    stage_sheet = workbook.sheets["Stage definitions"]
    _ensure_bounds(decision_sheet, "Review", 19, SAMPLED_REVIEW_LAST_ROW)
    _ensure_bounds(stage_sheet, "Stage definitions", 16, STAGE_REVIEW_LAST_ROW)
    if tuple(_value_row(decision_sheet, 1, 19)) != REVIEW_HEADERS:
        raise PrivateReviewError("review headers changed")
    if tuple(_value_row(stage_sheet, 1, 16)) != STAGE_HEADERS:
        raise PrivateReviewError("stage-definition headers changed")
    expected_decisions = _expected_review_rows(root, ledger)
    expected_stages = _expected_stage_rows(root)
    normalized_decisions: list[dict[str, Any]] = []
    normalized_stages: list[dict[str, Any]] = []
    reviewer_ids: set[str] = set()
    adjudication_required: dict[tuple[str, str], dict[str, Any]] = {}
    start = date.fromisoformat(settings["review_start_date"])
    end = date.fromisoformat(settings["review_end_date"])

    seen: set[str] = set()
    for offset, expected in enumerate(expected_decisions, start=2):
        observed = _value_row(decision_sheet, offset, 19)
        observed[2] = observed[2].zfill(6) if observed[2].isdigit() else observed[2]
        if observed[:11] != expected:
            raise PrivateReviewError(f"immutable proposal cells differ from current ledger/evidence: Review row {offset}")
        record_id = observed[0]
        if record_id in seen:
            raise PrivateReviewError(f"duplicate review record_id: {record_id}")
        seen.add(record_id)
        reviewer = _strict_text(observed[11], f"{record_id}/reviewer_id")
        review_date = date.fromisoformat(
            _parse_iso_or_excel_date(_cell(decision_sheet, f"M{offset}"), workbook.date1904, f"{record_id}/review_date")
        )
        if not start <= review_date <= end:
            raise PrivateReviewError(f"review date is outside declared audit interval: {record_id}")
        verdict = observed[13]
        corrected_decision = observed[14]
        corrected_stage = observed[15]
        note = _strict_text(observed[16], f"{record_id}/reviewer_note", allow_empty=True)
        if observed[17] != "Yes":
            raise PrivateReviewError(f"row-level outcome-blind declaration is not Yes: {record_id}")
        proposal = {"decision": observed[5], "stage": observed[6] or None}
        correction: dict[str, Any] | None = None
        if verdict == "Yes":
            if corrected_decision or corrected_stage:
                raise PrivateReviewError(f"Yes row carries correction fields: {record_id}")
        elif verdict == "No":
            if corrected_decision not in {"include", "exclude", "out_of_stage"} or not note:
                raise PrivateReviewError(f"No row lacks a valid correction/note: {record_id}")
            if corrected_decision == "include":
                allowed = observed[7].split("; ") if observed[7] else []
                if corrected_stage not in allowed:
                    raise PrivateReviewError(f"No row has an invalid corrected stage: {record_id}")
            elif corrected_stage:
                raise PrivateReviewError(f"non-include correction must not carry a stage: {record_id}")
            correction = {"decision": corrected_decision, "stage": corrected_stage or None}
            if correction == proposal:
                raise PrivateReviewError(f"No row does not change the proposal: {record_id}")
        elif verdict == "Uncertain":
            if not note or corrected_decision or corrected_stage:
                raise PrivateReviewError(f"Uncertain row is incomplete or carries correction fields: {record_id}")
        else:
            raise PrivateReviewError(f"review verdict is missing/invalid: {record_id}")
        reviewer_ids.add(reviewer)
        item = {
            "record_id": record_id, "kind": "code_decision", "chain_id": observed[1],
            "hs6": observed[2], "proposal": proposal, "reviewer_id": reviewer,
            "review_date": review_date.isoformat(), "verdict": verdict, "correction": correction,
            "reviewer_note": note, "outcome_blind_declaration": "Yes", "recomputed_status": "Complete",
        }
        normalized_decisions.append(item)
        if verdict in {"No", "Uncertain"}:
            adjudication_required[(record_id, "code_decision")] = item

    seen.clear()
    for offset, expected in enumerate(expected_stages, start=2):
        observed = _value_row(stage_sheet, offset, 16)
        if observed[:8] != expected:
            raise PrivateReviewError(
                f"immutable stage-definition cells differ from current evidence/registry: row {offset}"
            )
        record_id = observed[0]
        if record_id in seen:
            raise PrivateReviewError(f"duplicate stage record_id: {record_id}")
        seen.add(record_id)
        reviewer = _strict_text(observed[8], f"{record_id}/reviewer_id")
        review_date = date.fromisoformat(
            _parse_iso_or_excel_date(_cell(stage_sheet, f"J{offset}"), workbook.date1904, f"{record_id}/review_date")
        )
        if not start <= review_date <= end:
            raise PrivateReviewError(f"review date is outside declared audit interval: {record_id}")
        verdict = observed[10]
        corrected_definition = observed[11]
        corrected_fit_rule = observed[12]
        note = _strict_text(observed[13], f"{record_id}/reviewer_note", allow_empty=True)
        if observed[14] != "Yes":
            raise PrivateReviewError(f"row-level outcome-blind declaration is not Yes: {record_id}")
        proposal = {
            "canonical_definition": observed[3],
            "specificity": observed[4],
            "fit_rule": observed[5],
        }
        correction: dict[str, str] | None = None
        if verdict == "Yes":
            if corrected_definition or corrected_fit_rule:
                raise PrivateReviewError(f"Yes stage row carries correction fields: {record_id}")
        elif verdict == "No":
            corrected_definition = _strict_text(corrected_definition, f"{record_id}/corrected_definition")
            corrected_fit_rule = _strict_text(corrected_fit_rule, f"{record_id}/corrected_fit_rule")
            if not note:
                raise PrivateReviewError(f"No stage row lacks a reviewer note: {record_id}")
            correction = {
                "canonical_definition": corrected_definition,
                "specificity": observed[4],
                "fit_rule": corrected_fit_rule,
            }
            if correction == proposal:
                raise PrivateReviewError(f"No stage row does not change the proposal: {record_id}")
        elif verdict == "Uncertain":
            if not note or corrected_definition or corrected_fit_rule:
                raise PrivateReviewError(f"Uncertain stage row is incomplete or carries correction fields: {record_id}")
        else:
            raise PrivateReviewError(f"stage review verdict is missing/invalid: {record_id}")
        reviewer_ids.add(reviewer)
        item = {
            "record_id": record_id, "kind": "stage_definition", "chain_id": observed[1],
            "stage": observed[2], "proposal": proposal, "reviewer_id": reviewer,
            "review_date": review_date.isoformat(), "verdict": verdict, "correction": correction,
            "reviewer_note": note, "outcome_blind_declaration": "Yes", "recomputed_status": "Complete",
        }
        normalized_stages.append(item)
        if verdict in {"No", "Uncertain"}:
            adjudication_required[(record_id, "stage_definition")] = item
    if settings["primary_reviewer"] not in reviewer_ids:
        raise PrivateReviewError("primary_reviewer does not occur in the completed row set")
    return normalized_decisions, normalized_stages, reviewer_ids, adjudication_required


def _validate_adjudication(
    path: Path | None,
    required: Mapping[tuple[str, str], Mapping[str, Any]],
    settings: Mapping[str, str],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, str | None]:
    if not required:
        if path is not None:
            raise PrivateReviewError("an adjudication file is forbidden when no No/Uncertain row exists")
        return [], [], 0, None
    if path is None or path.is_symlink() or not path.is_file():
        raise PrivateReviewError("every No/Uncertain row requires an explicit canonical adjudication file")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=public_receipt._object_no_duplicates,
            parse_constant=public_receipt._reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, public_receipt.HumanReviewReceiptError) as exc:
        raise PrivateReviewError(f"invalid adjudication JSON: {exc}") from exc
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise PrivateReviewError("adjudication record must be canonical strict JSON")
    _exact_keys(payload, {"schema_version", "audit_id", "records"}, "adjudication")
    if payload["schema_version"] != ADJUDICATION_SCHEMA or payload["audit_id"] != settings["audit_id"]:
        raise PrivateReviewError("adjudication schema/audit identity mismatch")
    records = payload["records"]
    if not isinstance(records, list):
        raise PrivateReviewError("adjudication records must be a list")
    normalized: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    documentation_only = 0
    seen: set[tuple[str, str]] = set()
    start = date.fromisoformat(settings["review_start_date"])
    end = date.fromisoformat(settings["review_end_date"])
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PrivateReviewError(f"adjudication record {index} must be an object")
        _exact_keys(
            record,
            {"record_id", "kind", "reviewer_verdict", "resolution", "final", "adjudicator_id", "adjudication_date", "note"},
            f"adjudication record {index}",
        )
        record_id = _strict_text(record["record_id"], f"adjudication[{index}]/record_id")
        kind = record["kind"]
        identity = (record_id, kind)
        source = required.get(identity)
        if source is None or identity in seen:
            raise PrivateReviewError(f"extra, duplicate, or unknown adjudication: {identity}")
        seen.add(identity)
        if record["reviewer_verdict"] != source["verdict"]:
            raise PrivateReviewError(f"adjudication reviewer verdict mismatch: {record_id}")
        resolution = record["resolution"]
        if resolution not in {"retain_proposal", "accept_reviewer_change", "accept_other_change", "documentation_only"}:
            raise PrivateReviewError(f"invalid adjudication resolution: {record_id}")
        if resolution == "accept_reviewer_change" and source["verdict"] != "No":
            raise PrivateReviewError(f"cannot accept a reviewer change for an Uncertain row: {record_id}")
        adjudicator = _strict_text(record["adjudicator_id"], f"adjudication[{index}]/adjudicator_id")
        if adjudicator != settings["final_audit_owner"]:
            raise PrivateReviewError(f"adjudicator differs from final_audit_owner: {record_id}")
        adjudication_date = date.fromisoformat(_strict_text(record["adjudication_date"], "adjudication_date"))
        if not start <= adjudication_date <= end:
            raise PrivateReviewError(f"adjudication date is outside declared audit interval: {record_id}")
        note = _strict_text(record["note"], f"adjudication[{index}]/note")
        final = record["final"]
        if not isinstance(final, dict):
            raise PrivateReviewError(f"adjudication final value must be an object: {record_id}")
        if kind == "code_decision":
            _exact_keys(final, {"decision", "stage"}, f"adjudication final {record_id}")
            if final.get("decision") not in {"include", "exclude", "out_of_stage"}:
                raise PrivateReviewError(f"invalid final code decision: {record_id}")
            if final["decision"] == "include":
                chain = source["chain_id"]
                stage = final.get("stage")
                baseline = public_receipt.construct_projection(root)
                if not isinstance(stage, str) or stage not in baseline["chains"][chain]["stages"]:
                    raise PrivateReviewError(f"invalid final include stage: {record_id}")
            elif final.get("stage") is not None:
                raise PrivateReviewError(f"final non-include decision must have null stage: {record_id}")
        elif kind == "stage_definition":
            _exact_keys(
                final,
                {"canonical_definition", "specificity", "fit_rule"},
                f"adjudication final {record_id}",
            )
            _strict_text(final.get("canonical_definition"), f"adjudication final definition {record_id}")
            _strict_text(final.get("specificity"), f"adjudication final specificity {record_id}")
            _strict_text(final.get("fit_rule"), f"adjudication final fit rule {record_id}")
        else:
            raise PrivateReviewError(f"invalid adjudication kind: {kind}")
        proposal = source["proposal"]
        correction = source["correction"]
        if resolution in {"retain_proposal", "documentation_only"} and final != proposal:
            raise PrivateReviewError(f"retained/documentation-only adjudication changed the construct: {record_id}")
        if resolution == "accept_reviewer_change" and final != correction:
            raise PrivateReviewError(f"accepted reviewer correction does not match workbook: {record_id}")
        if resolution == "accept_other_change" and final == proposal:
            raise PrivateReviewError(f"accept_other_change does not change the construct: {record_id}")
        if resolution in {"retain_proposal", "documentation_only"}:
            documentation_only += 1
        if final != proposal:
            accepted.append({"record_id": record_id, "kind": kind, "old": proposal, "new": final})
        normalized.append({
            "record_id": record_id, "kind": kind, "reviewer_verdict": record["reviewer_verdict"],
            "resolution": resolution, "final": final, "adjudicator_id": adjudicator,
            "adjudication_date": adjudication_date.isoformat(), "note": note,
        })
    if seen != set(required):
        raise PrivateReviewError(f"missing adjudications: {sorted(set(required)-seen)!r}")
    wanted = sorted(normalized, key=lambda item: (item["record_id"], item["kind"]))
    if normalized != wanted:
        raise PrivateReviewError("adjudication records must use canonical record_id/kind order")
    accepted.sort(key=lambda item: (item["record_id"], item["kind"]))
    return normalized, accepted, documentation_only, _sha256_bytes(raw)


def _benchmark_identity(root: Path) -> dict[str, Any]:
    evidence = _read_json(root / "chains/evidence/registry_evidence.json", "registry evidence")
    scan = _read_json(root / "chains/evidence/registry_full_scan_receipt.json", "full scan receipt")
    return {
        "benchmark_version": scan["benchmark_identity"]["benchmark_version"],
        "registry_snapshot": scan["benchmark_identity"]["data_snapshot"],
        "rule_id": evidence["rule_id"],
        "source_dictionary_member_sha256": evidence["source"]["source_metadata_member_sha256"],
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def prepare(
    workbook_path: Path,
    output_dir: Path,
    *,
    adjudication_path: Path | None = None,
    receipt_issued_date: str,
    new_registry_id: str | None = None,
    new_benchmark_version: str | None = None,
    root: Path = ROOT,
) -> tuple[Path, Path]:
    root = Path(root)
    workbook_path = _safe_existing_file(Path(workbook_path), root, "workbook")
    output_dir = _safe_private_output_dir(Path(output_dir), root)
    if adjudication_path is not None:
        adjudication_path = _safe_existing_file(
            Path(adjudication_path), root, "adjudication", private_only=True
        )
    try:
        issued = date.fromisoformat(receipt_issued_date)
    except ValueError as exc:
        raise PrivateReviewError("receipt_issued_date must be an ISO date") from exc
    workbook = Workbook(workbook_path)
    _validate_formulas(workbook)
    ledger = _load_ledger(root)
    _verify_static_sheets(workbook, ledger)
    freeze = public_receipt.validate_pre_review_freeze(root)
    if (
        _sha256_bytes(_canonical_json_bytes(_instrument_semantics(workbook)))
        != freeze["instrument"]["immutable_semantics_sha256"]
    ):
        raise PrivateReviewError("completed workbook immutable semantics differ from the pre-review freeze")
    if (
        _sha256_bytes(_canonical_json_bytes(_audit_settings_semantics(workbook)))
        != freeze["instrument"]["audit_settings_sha256"]
    ):
        raise PrivateReviewError("completed workbook static audit settings differ from the pre-review freeze")
    settings = _verify_audit_settings(workbook, root, ledger, freeze)
    if settings["audit_id"] != freeze["audit_id"]:
        raise PrivateReviewError("completed workbook audit_id differs from the pre-review freeze")
    decisions, stages, reviewer_ids, required = _review_completion(workbook, root, ledger, settings)
    adjudications, accepted_changes, documentation_only, adjudication_sha = _validate_adjudication(
        adjudication_path, required, settings, root
    )
    completed = date.fromisoformat(settings["review_end_date"])
    if issued < completed:
        raise PrivateReviewError("receipt issue date predates completed review")
    baseline = public_receipt.construct_projection(root)
    reviewed, accepted_changes = public_receipt._apply_accepted_changes(baseline, accepted_changes)
    baseline_hash = public_receipt.construct_sha256(baseline)
    reviewed_hash = public_receipt.construct_sha256(reviewed)
    if accepted_changes:
        if not new_registry_id or not new_benchmark_version:
            raise PrivateReviewError("accepted construct changes require explicit new registry and benchmark identifiers")
        if (
            new_registry_id == _benchmark_identity(root)["registry_snapshot"]
            or new_benchmark_version == _benchmark_identity(root)["benchmark_version"]
        ):
            raise PrivateReviewError("construct-changing identifiers must differ from the current benchmark")
        disposition = {
            "kind": public_receipt.CONSTRUCT_CHANGE_REQUIRED, "release_eligible": False,
            "registry_dependent_rerun_required": True, "required_new_registry_id": new_registry_id,
            "required_new_benchmark_version": new_benchmark_version,
        }
    else:
        if new_registry_id is not None or new_benchmark_version is not None:
            raise PrivateReviewError("new identifiers are forbidden when no construct change was accepted")
        disposition = {
            "kind": public_receipt.NO_CONSTRUCT_CHANGE, "release_eligible": True,
            "registry_dependent_rerun_required": False, "required_new_registry_id": None,
            "required_new_benchmark_version": None,
        }
    workbook_sha = _sha256_file(workbook_path)
    public_hashes = public_receipt.current_public_input_hashes(root)
    normalized = {
        "schema_version": NORMALIZED_SCHEMA,
        "audit_id": settings["audit_id"],
        "workbook_sha256": workbook_sha,
        "public_inputs_sha256": public_hashes,
        "sampling_plan": freeze["sampling_plan"],
        "audit_settings": {
            key: settings[key]
            for key in (
                "workbook_generated_date", "primary_reviewer", "review_start_date", "review_end_date",
                "outcome_access_declaration", "final_audit_owner", "audit_notes",
            )
        },
        "decision_records": decisions,
        "stage_definition_records": stages,
        "adjudications": adjudications,
        "construct": {
            "baseline_sha256": baseline_hash, "reviewed_sha256": reviewed_hash,
            "accepted_changes": accepted_changes,
        },
    }
    normalized_bytes = _canonical_json_bytes(normalized)
    normalized_sha = _sha256_bytes(normalized_bytes)
    no_count = sum(item["verdict"] == "No" for item in decisions + stages)
    uncertain_count = sum(item["verdict"] == "Uncertain" for item in decisions + stages)
    receipt = {
        "schema_version": public_receipt.SCHEMA_VERSION,
        "status": public_receipt.STATUS_SAMPLED_COMPLETE,
        "audit_id": settings["audit_id"],
        "benchmark_identity": _benchmark_identity(root),
        "pre_review_freeze": {
            "path": public_receipt.DEFAULT_FREEZE.relative_to(public_receipt.ROOT).as_posix(),
            "sha256": public_receipt.EXPECTED_FREEZE_SHA256,
        },
        "public_inputs_sha256": public_hashes,
        "private_record": {
            "workbook_sha256": workbook_sha, "normalized_review_sha256": normalized_sha,
            "adjudication_sha256": adjudication_sha,
        },
        "scope": {
            "decision_frame_records": FRAME_DECISION_RECORDS,
            "sampled_decision_records": SAMPLED_DECISION_RECORDS,
            "sampled_unique_hs6": freeze["scope"]["sampled_unique_hs6"],
            "unique_hs6": sample_plan.FRAME_UNIQUE_HS6,
            "stage_definition_records": STAGE_DEFINITION_RECORDS,
        },
        "sampling_plan": freeze["sampling_plan"],
        "completion": {
            "sampled_decision_records_complete": SAMPLED_DECISION_RECORDS,
            "sampled_decision_records_not_started": 0,
            "sampled_decision_records_incomplete": 0,
            "stage_definitions_complete": STAGE_DEFINITION_RECORDS,
            "stage_definitions_not_started": 0, "stage_definitions_incomplete": 0,
            "reviewer_count": len(reviewer_ids),
            "row_outcome_blind_declarations_yes": OUTCOME_BLIND_REVIEW_ROWS,
            "audit_outcome_access_declaration_present": True,
            # Public receipt /3 keeps this historical field name; it counts No verdicts.
            "reviewer_change_count": no_count, "reviewer_uncertain_count": uncertain_count,
            "adjudication_required_count": len(required), "adjudication_complete_count": len(adjudications),
            "unresolved_count": 0, "documentation_only_resolution_count": documentation_only,
        },
        "outcome_blindness": {
            "declared": True, "instrument_forbidden_content_scan": "PASS",
            "prohibited_inputs": list(public_receipt.PROHIBITED_REVIEW_INPUTS),
            "claim_limit": public_receipt.OUTCOME_BLIND_CLAIM_LIMIT,
        },
        "construct": {
            "projection_schema": public_receipt.CONSTRUCT_SCHEMA,
            "fields": list(public_receipt.CONSTRUCT_FIELDS), "baseline_sha256": baseline_hash,
            "reviewed_sha256": reviewed_hash, "accepted_changes": accepted_changes,
            "accepted_changes_sha256": _sha256_bytes(_canonical_json_bytes(accepted_changes)),
            "accepted_change_count": len(accepted_changes),
        },
        "disposition": disposition,
        "review_completed_date": completed.isoformat(),
        "receipt_issued_date": issued.isoformat(),
    }
    receipt_bytes = _canonical_json_bytes(receipt)
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / DRAFT_NAME
        candidate.write_bytes(receipt_bytes)
        public_receipt.validate_receipt(candidate, root=root)
    private_path = output_dir / PRIVATE_NAME
    draft_path = output_dir / DRAFT_NAME
    created: list[Path] = []
    try:
        _write_exclusive(private_path, normalized_bytes)
        created.append(private_path)
        _write_exclusive(draft_path, receipt_bytes)
        created.append(draft_path)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return private_path, draft_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--receipt-issued-date", required=True)
    parser.add_argument("--new-registry-id")
    parser.add_argument("--new-benchmark-version")
    args = parser.parse_args()
    try:
        private_path, draft_path = prepare(
            args.workbook, args.output_dir, adjudication_path=args.adjudication,
            receipt_issued_date=args.receipt_issued_date,
            new_registry_id=args.new_registry_id, new_benchmark_version=args.new_benchmark_version,
        )
    except (PrivateReviewError, public_receipt.HumanReviewReceiptError, OSError, KeyError, ValueError) as exc:
        print(f"PRIVATE HUMAN REVIEW PREPARE FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "private human-review draft prepared "
        f"(private={private_path.name}; draft={draft_path.name}; canonical_receipt_written=false)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
