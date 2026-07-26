#!/usr/bin/env python3
"""Read-only final alignment gate for UpgradeBench v2 numerical claims.

This verifier never generates or rewrites an artifact.  It composes the
canonical LOCO, ULTRA, benchmark-profile, registry, paper-interface, review,
and invalidation verifiers, then checks the joins between their outputs.  The
``interface`` mode is useful while a release transaction is still being
assembled.  The ``release`` mode additionally requires the completed
outcome-blind human-review gate and a valid invalidation-resolution receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_chain_registry
import generate_v2_benchmark_profile as benchmark_profile
import generate_v2_paper_numbers as paper_numbers
import public_release_policy as public_policy
import registry_human_review_receipt
import release_manifest
import resolve_v2_invalidation
import summarize_v2_contemporary_references as contemporary_results
import summarize_v2_loco_results as loco_results
import summarize_v2_ultra_results as ultra_results


ROOT = Path(__file__).resolve().parents[1]
PAPER_JSON = Path("results_v2/paper_numbers.json")
PAPER_TEX = Path("paper/generated/v2_numbers.tex")
PROFILE_JSON = Path("results_v2/metrics/v2_benchmark_profile.json")
PROFILE_TEX = Path("paper/generated/v2_benchmark_profile.tex")
CONTEMPORARY_CONFIG = Path("configs/v2_contemporary_references.json")
CONTEMPORARY_JSON = Path("results_v2/metrics/v2_contemporary_references.json")
CONTEMPORARY_CSV = Path("results_v2/metrics/v2_contemporary_references.csv")
CONTEMPORARY_TEX = Path("paper/generated/v2_contemporary_references.tex")
REGISTRY_AUDIT = Path("docs/registry_audit.json")
REGISTRY_EVIDENCE = Path("chains/evidence/registry_evidence.json")
CURATION_PROTOCOL = Path("chains/evidence/registry_curation_protocol.json")
MANUSCRIPT_PARTS = (
    Path("paper/abstract.tex"),
    Path("paper/body.tex"),
    Path("paper/appendix.tex"),
)
PAPER_WRAPPERS = (Path("paper/main.tex"), Path("paper/main-acm.tex"))
PUBLIC_STATUS_DOCS = (
    Path("README.md"),
    Path("results_v2/README.md"),
    Path("results_v2/CLAIM_LEDGER.md"),
    Path("docs/REGISTRY_REVIEW_COMPLETION_ADDENDUM.md"),
)
PUBLIC_CHRONOLOGY_DOCS = (
    Path("README.md"),
    Path("ARTIFACT.md"),
    Path("DATA_LICENSE.md"),
    Path("BENCHMARK_V2_SPEC.md"),
    Path("results_v2/README.md"),
    Path("results_v2/CLAIM_LEDGER.md"),
    Path("docs/V2_RELEASE_WORKFLOW.md"),
    Path("docs/REGISTRY_REVIEW_COMPLETION_ADDENDUM.md"),
    Path("benchmark/upgrade-bench-v2/README.md"),
    Path("benchmark/upgrade-bench-v2/DATASHEET.md"),
    Path("paper/body.tex"),
    Path("paper/appendix.tex"),
)
OPTIONAL_PUBLIC_CHRONOLOGY_DOCS = (Path("paper/REBUTTAL_NOTES.md"),)
CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
CHAIN_MACRO = {
    "sheep": "Sheep",
    "cotton": "Cotton",
    "aluminium": "Aluminium",
    "nickel": "Nickel",
    "cocoa": "Cocoa",
    "oilseed-soy": "OilseedSoy",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MACRO_REFERENCE = re.compile(r"\\(VTwo[A-Za-z0-9]+)\b")
MACRO_DEFINITION = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\s*\{\\(VTwo[A-Za-z0-9]+)\}"
)
PROFILE_MACRO_DEFINITION = re.compile(
    r"\\newcommand\{\\(VTwo[A-Za-z0-9]+)\}\{.*\}\Z"
)
PAPER_MACRO_DEFINITION = re.compile(
    r"\\newcommand\{\\(VTwo[A-Za-z0-9]+)\}\{(.*)\}\Z"
)
STALE_RELEASE_MARKERS = (
    "working-draft state",
    "working draft withholds",
    "superseded 131-code",
    "historical layout placeholder",
    "must be regenerated",
    "still describe the superseded",
    "makes no current claim",
    "planned outcome-blind human review",
    "human review remains zero",
    "zero completed human",
    "at the present artifact state",
)
STALE_PUBLIC_STATUS_PATTERNS = (
    re.compile(r"planned outcome-blind human review", re.IGNORECASE),
    re.compile(r"human review remains (?:at )?zero", re.IGNORECASE),
    re.compile(r"zero completed (?:human reviews?|rows)", re.IGNORECASE),
    re.compile(r"no completed human review", re.IGNORECASE),
    re.compile(r"at present none exists", re.IGNORECASE),
    re.compile(r"release-pending", re.IGNORECASE),
    re.compile(r"current-registry replacement is pending", re.IGNORECASE),
    re.compile(r"regeneration is pending", re.IGNORECASE),
    re.compile(r"replacement comparison .* pending", re.IGNORECASE),
    re.compile(r"formal 283-code matched-LOCO run remains pending", re.IGNORECASE),
)
BARE_DECIMAL = re.compile(r"(?<![A-Za-z0-9])[-+]?[0-9]+\.[0-9]+(?![A-Za-z0-9])")
COMPUTE_SUBJECT = (
    r"(?:computation(?:s| branch)?|result(?:s| set| branch| artifacts?)?|"
    r"metric(?:s)?|analys(?:is|es)|evaluation(?:s)?|benchmark outputs?|"
    r"model outputs?|paper-number interface|numerical interface|"
    r"formal (?:run|evaluation)|model (?:run|evaluation)|experiment(?:s)?|"
    r"GPU run|LOCO run|ULTRA(?:-ZS)? run)"
)
COMPUTE_COMPLETION = (
    r"(?:(?:was|were|is|are|had been|have been|has been)\s+)?"
    r"(?:already\s+)?(?:completed|complete|finished|frozen|sealed|finali[sz]ed|"
    r"computed|recomputed|rebuilt|generated|verified|promoted|locked)"
)
REVIEW_TIMING = (
    r"(?:human[- ]review|manual review|registry review|construct review|review branch)\s+"
    r"(?:then\s+)?(?:began|started|commenced|followed|was initiated|was launched|"
    r"was conducted later|is underway|is in progress|remains pending|"
    r"had not (?:yet )?begun|has not (?:yet )?begun)"
)
CHRONOLOGY_LEAK_PATTERNS = (
    re.compile(
        rf"\b{COMPUTE_SUBJECT}\b.{{0,100}}?\b{COMPUTE_COMPLETION}\b"
        rf".{{0,240}}?\b{REVIEW_TIMING}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b{REVIEW_TIMING}\b.{{0,120}}?\b(?:after|following|subsequent to|"
        rf"once|only after)\b.{{0,120}}?\b{COMPUTE_SUBJECT}\b"
        rf".{{0,80}}?\b{COMPUTE_COMPLETION}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b(?:after|once|only after)\b.{{0,80}}?\b{COMPUTE_SUBJECT}\b"
        rf".{{0,80}}?\b{COMPUTE_COMPLETION}\b.{{0,120}}?\b{REVIEW_TIMING}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b(?:human[- ]review|manual review|registry review|construct review|"
        rf"review branch)\b.{{0,40}}?\b(?:did not (?:begin|start) until|"
        rf"was deferred until|was opened only after)\b.{{0,120}}?"
        rf"\b{COMPUTE_SUBJECT}\b.{{0,80}}?\b{COMPUTE_COMPLETION}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b{COMPUTE_COMPLETION}\b\s+\b{COMPUTE_SUBJECT}\b.{{0,120}}?"
        rf"\b(?:before|prior to|ahead of)\b.{{0,80}}?\b{REVIEW_TIMING}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b(?:human[- ]review|manual review|registry review|construct review|"
        rf"review branch)\b.{{0,40}}?\b(?:followed|came after)\b.{{0,80}}?"
        rf"\b{COMPUTE_COMPLETION}\b\s+\b{COMPUTE_SUBJECT}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b(?:by the time|when)\b.{{0,40}}?\b{REVIEW_TIMING}\b"
        rf".{{0,120}}?\b{COMPUTE_SUBJECT}\b.{{0,60}}?"
        rf"\b{COMPUTE_COMPLETION}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b{REVIEW_TIMING}\b.{{0,100}}?\b{COMPUTE_SUBJECT}\b"
        rf".{{0,40}}?\b(?:already|previously)\b.{{0,20}}?"
        rf"\b{COMPUTE_COMPLETION}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b{COMPUTE_SUBJECT}\b.{{0,80}}?\b{COMPUTE_COMPLETION}\b"
        rf".{{0,80}}?\b(?:only\s+)?then\s+did\s+"
        rf"(?:human[- ]review|manual review|registry review|construct review)\s+"
        rf"(?:begin|start|commence)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
REVIEW_COMPLETION_ASSERTIONS = (
    re.compile(
        r"\b(?:the|this|our|registry|outcome-blind)?\s*"
        r"(?:human[- ]review|registry review|construct review)\s+"
        r"(?:is|was|has been|had been)\s+(?:(?:now|formally)\s+)?"
        r"(?:complete|completed|finished|concluded)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(?:completed|finished)\s+(?:the\s+)?"
        r"(?:human[- ]review|registry review|construct review)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:canonical\s+)?(?:human[- ]review|registry review|construct review)"
        r"\s+receipt\s+(?:is|was|has been)\s+"
        r"(?:present|complete|completed|verified|issued|retained)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\ball\s+[0-9][0-9,{}]*(?:\s+(?:human[- ]review|registry review))?\s+"
        r"(?:rows|records|codes|decisions)\s+(?:have been|were|are)\s+"
        r"(?:human[- ]reviewed|manually reviewed|completed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:human[- ]review|registry review|construct review)\s+status\s+"
        r"(?:is|was|has been)\s+(?:complete|completed|verified)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:[0-9]+/[0-9]+|[0-9]+\s+of\s+[0-9]+)\s+"
        r"(?:human[- ]review\s+)?(?:rows|records|decisions)\s+"
        r"(?:are|were|have been)\s+complete\b",
        re.IGNORECASE,
    ),
)


class NumberAlignmentError(ValueError):
    """Raised when two governed numerical interfaces do not reconcile."""


def _reject_constant(value: str) -> None:
    raise NumberAlignmentError(f"non-finite JSON constant is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NumberAlignmentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, role: str, *, canonical: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NumberAlignmentError(f"{role} is missing or unsafe: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumberAlignmentError(f"cannot read strict JSON for {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise NumberAlignmentError(f"{role} root must be an object")
    if canonical:
        expected = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if raw != expected:
            raise NumberAlignmentError(f"{role} is not canonical sorted LF JSON")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tex_int(value: Any, role: str) -> int:
    if not isinstance(value, str) or re.fullmatch(
        r"-?(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:\{,\}[0-9]{3})+)", value
    ) is None:
        raise NumberAlignmentError(f"{role} is not a canonical TeX integer: {value!r}")
    return int(value.replace("{,}", ""))


def _exact_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NumberAlignmentError(f"{role} must be an integer")
    return value


def _verify_paper_tex_interface(
    tex_path: Path,
    numbers: Mapping[str, str],
    sources: Mapping[str, str],
) -> None:
    """Verify the TeX interface without imposing an order on macro definitions."""

    if tex_path.is_symlink() or not tex_path.is_file():
        raise NumberAlignmentError("paper-number TeX is missing or unsafe")
    try:
        content = tex_path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NumberAlignmentError("paper-number TeX is not readable UTF-8") from exc
    if not content.endswith(b"\n") or b"\r" in content:
        raise NumberAlignmentError("paper-number TeX does not use canonical LF bytes")

    expected_prefix = [
        "% AUTO-GENERATED by tools/generate_v2_paper_numbers.py; do not edit.",
        "% Source SHA-256 values:",
        *[
            f"% {digest}  {relative}"
            for relative, digest in sorted(sources.items())
        ],
    ]
    lines = text.splitlines()
    if lines[: len(expected_prefix)] != expected_prefix:
        raise NumberAlignmentError("paper-number JSON/TeX source maps differ")

    observed_numbers: dict[str, str] = {}
    for line in lines[len(expected_prefix) :]:
        match = PAPER_MACRO_DEFINITION.fullmatch(line)
        if match is None:
            raise NumberAlignmentError("paper-number TeX has a malformed macro line")
        key, rendered = match.groups()
        if key in observed_numbers:
            raise NumberAlignmentError(f"paper-number TeX repeats macro {key}")
        observed_numbers[key] = rendered

    if content != paper_numbers.render_tex(observed_numbers, sources).encode("utf-8"):
        raise NumberAlignmentError("paper-number TeX is not the generator's exact rendering")
    if observed_numbers != dict(numbers):
        raise NumberAlignmentError("paper-number JSON and TeX interfaces differ")


def _paper_interface(root: Path, *, profile: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify JSON/TeX equality and all locally available source hashes."""

    json_path = root / PAPER_JSON
    tex_path = root / PAPER_TEX
    payload = _load_json(json_path, "paper-number interface", canonical=True)
    required_fields = {
        "schema_version",
        "benchmark_version",
        "status",
        "gpu_status",
        "loco_status",
        "ultra_status",
        "gbdt_status",
        "sources",
        "numbers",
    }
    if set(payload) != required_fields:
        raise NumberAlignmentError("paper-number interface field inventory differs")
    if (
        payload.get("schema_version") != paper_numbers.PAPER_NUMBERS_SCHEMA
        or payload.get("benchmark_version") != paper_numbers.BENCHMARK_VERSION
        or payload.get("status") != "complete"
    ):
        raise NumberAlignmentError("paper-number interface identity/status is stale")
    for status in ("gpu_status", "loco_status", "ultra_status", "gbdt_status"):
        if payload.get(status) != "COMPLETE":
            raise NumberAlignmentError(f"paper-number interface {status} is incomplete")

    try:
        numbers = public_policy._validate_paper_numbers(payload.get("numbers"))
    except ValueError as exc:
        raise NumberAlignmentError(f"paper-number macro inventory is invalid: {exc}") from exc
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or set(raw_sources) != set(
        public_policy.V2_PAPER_SOURCE_PATHS
    ):
        raise NumberAlignmentError("paper-number source inventory differs from policy")
    sources: dict[str, str] = {}
    for relative, expected in raw_sources.items():
        if (
            not isinstance(relative, str)
            or public_policy.canonical_path_reason(relative) is not None
            or not isinstance(expected, str)
            or HEX64.fullmatch(expected) is None
        ):
            raise NumberAlignmentError("paper-number source map contains an unsafe entry")
        reason = public_policy.source_path_reason(relative, root, require_file=False)
        if reason is not None:
            raise NumberAlignmentError(f"paper source path is unsafe: {relative} ({reason})")
        source = root / relative
        if source.is_file():
            if _sha256(source) != expected:
                raise NumberAlignmentError(f"paper source hash is stale: {relative}")
        elif profile == "full" or not relative.startswith(
            public_policy.V2_EXTERNAL_SOURCE_PREFIXES
        ):
            raise NumberAlignmentError(f"paper source is missing: {relative}")
        sources[relative] = expected

    expected_json = paper_numbers.render_json(numbers, sources).encode("utf-8")
    if json_path.read_bytes() != expected_json:
        raise NumberAlignmentError("paper-number JSON is not the generator's exact rendering")
    _verify_paper_tex_interface(tex_path, numbers, sources)
    return payload, numbers


def _profile_paper_crosscheck(
    profile_payload: Mapping[str, Any], numbers: Mapping[str, str]
) -> None:
    if profile_payload.get("benchmark_version") != paper_numbers.BENCHMARK_VERSION:
        raise NumberAlignmentError("benchmark profile and paper benchmark versions differ")
    totals = profile_payload.get("totals")
    if not isinstance(totals, dict):
        raise NumberAlignmentError("benchmark profile totals are missing")
    joins = {
        "b1_candidate_entries": "VTwoTrackBOneCandidates",
        "b1_positive_entries": "VTwoTrackBOnePositives",
        "b2_candidate_lanes": "VTwoTrackBTwoCandidates",
        "b2_positive_lanes": "VTwoTrackBTwoPositives",
    }
    for profile_key, paper_key in joins.items():
        observed = _exact_int(totals.get(profile_key), f"profile totals.{profile_key}")
        expected = _tex_int(numbers.get(paper_key), f"paper macro {paper_key}")
        if observed != expected:
            raise NumberAlignmentError(
                f"profile/paper count mismatch: {profile_key}={observed}, "
                f"{paper_key}={expected}"
            )
    groups = _exact_int(
        totals.get("b2_positive_entry_groups"),
        "profile totals.b2_positive_entry_groups",
    )
    if groups != _tex_int(numbers.get("VTwoTrackBOnePositives"), "B1 positives"):
        raise NumberAlignmentError("B2 group count does not equal realized B1 entries")

    # These are independently generated views of the same current cohort.
    # Keep the joins explicit so a partial paper-interface refresh cannot leave
    # coverage, product-space, or reference-threshold counts on an older
    # registry while the headline/profile counts have already moved forward.
    macro_joins = (
        (
            "VTwoBOneCoverageMainCoveredRealizedEntries",
            "VTwoTrackBOnePositives",
        ),
        (
            "VTwoBOneCoverageMainEligibleMarketLateStartLanes",
            "VTwoTrackBTwoPositives",
        ),
        ("VTwoProductSpaceMainBOneCandidates", "VTwoTrackBOneCandidates"),
        ("VTwoProductSpaceMainBOnePositives", "VTwoTrackBOnePositives"),
        (
            "VTwoEligibilityThresholdHundredTrackACandidates",
            "VTwoTrackACandidates",
        ),
        (
            "VTwoEligibilityThresholdHundredTrackAPositives",
            "VTwoTrackAPositives",
        ),
        (
            "VTwoEligibilityThresholdHundredTrackBOneCandidates",
            "VTwoTrackBOneCandidates",
        ),
        (
            "VTwoEligibilityThresholdHundredTrackBOnePositives",
            "VTwoTrackBOnePositives",
        ),
        (
            "VTwoEligibilityThresholdHundredTrackBTwoCandidates",
            "VTwoTrackBTwoCandidates",
        ),
        (
            "VTwoEligibilityThresholdHundredTrackBTwoPositives",
            "VTwoTrackBTwoPositives",
        ),
    )
    for derived_key, headline_key in macro_joins:
        derived = _tex_int(numbers.get(derived_key), f"paper macro {derived_key}")
        headline = _tex_int(numbers.get(headline_key), f"paper macro {headline_key}")
        if derived != headline:
            raise NumberAlignmentError(
                f"paper cohort-count join mismatch: {derived_key}={derived}, "
                f"{headline_key}={headline}"
            )

    all_realized = _tex_int(
        numbers.get("VTwoBOneCoverageMainAllRealizedEntries"),
        "paper macro VTwoBOneCoverageMainAllRealizedEntries",
    )
    covered_realized = _tex_int(
        numbers.get("VTwoBOneCoverageMainCoveredRealizedEntries"),
        "paper macro VTwoBOneCoverageMainCoveredRealizedEntries",
    )
    all_lanes = _tex_int(
        numbers.get("VTwoBOneCoverageMainAllLateStartLanes"),
        "paper macro VTwoBOneCoverageMainAllLateStartLanes",
    )
    eligible_lanes = _tex_int(
        numbers.get("VTwoBOneCoverageMainEligibleMarketLateStartLanes"),
        "paper macro VTwoBOneCoverageMainEligibleMarketLateStartLanes",
    )
    inactive_lanes = _tex_int(
        numbers.get("VTwoBOneCoverageMainInactiveMarketLateStartLanes"),
        "paper macro VTwoBOneCoverageMainInactiveMarketLateStartLanes",
    )
    if covered_realized > all_realized:
        raise NumberAlignmentError("covered realized entries exceed all realized entries")
    if eligible_lanes + inactive_lanes != all_lanes:
        raise NumberAlignmentError("coverage lane accounting does not close")

    rows = profile_payload.get("chains")
    if not isinstance(rows, list) or len(rows) != len(CHAINS):
        raise NumberAlignmentError("benchmark profile chain inventory differs")
    by_chain = {
        row.get("chain"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("chain"), str)
    }
    if set(by_chain) != set(CHAINS):
        raise NumberAlignmentError("benchmark profile chain keys differ")
    hs6_total = 0
    for chain in CHAINS:
        graph = by_chain[chain].get("graph")
        if not isinstance(graph, dict):
            raise NumberAlignmentError(f"{chain} profile graph is missing")
        observed = _exact_int(graph.get("hs6_products"), f"{chain}.graph.hs6_products")
        macro = f"VTwoRegistry{CHAIN_MACRO[chain]}ActiveCodes"
        expected = _tex_int(numbers.get(macro), f"paper macro {macro}")
        if observed != expected:
            raise NumberAlignmentError(
                f"profile/paper HS6 mismatch for {chain}: {observed} != {expected}"
            )
        hs6_total += observed
    if hs6_total != _tex_int(
        numbers.get("VTwoRegistryIncludedCodes"), "VTwoRegistryIncludedCodes"
    ):
        raise NumberAlignmentError("per-chain profile HS6 counts do not sum to registry total")


def _observable_split(evidence: Mapping[str, Any]) -> dict[str, int]:
    counts = {"include": 0, "exclude": 0, "out_of_stage": 0}
    chains = evidence.get("chains")
    if not isinstance(chains, dict):
        raise NumberAlignmentError("registry evidence chains are missing")
    for chain in CHAINS:
        row = chains.get(chain)
        if not isinstance(row, dict) or not isinstance(row.get("decisions"), list):
            raise NumberAlignmentError(f"registry evidence decisions missing for {chain}")
        for decision in row["decisions"]:
            if not isinstance(decision, dict):
                raise NumberAlignmentError("registry evidence has a malformed decision")
            if decision.get("candidate_source") != "observable_regex":
                continue
            kind = decision.get("decision")
            if kind not in counts:
                raise NumberAlignmentError("registry evidence has an unknown decision")
            counts[kind] += 1
    return counts


def _assert_claim_values(
    text: str,
    pattern: str,
    expected: Sequence[int],
    role: str,
) -> None:
    for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
        observed = tuple(int(value.replace("{,}", "")) for value in match.groups())
        if observed != tuple(expected):
            raise NumberAlignmentError(
                f"hard-coded manuscript claim conflicts with {role}: "
                f"observed={observed}, expected={tuple(expected)}"
            )


def _lint_registry_literals(
    text: str,
    audit: Mapping[str, Any],
    evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    summary = audit.get("summary")
    quality = protocol.get("quality_controls")
    if not isinstance(summary, dict) or not isinstance(quality, dict):
        raise NumberAlignmentError("registry summary/quality controls are missing")
    observable = _observable_split(evidence)
    number = r"([0-9]+(?:\{,\}[0-9]{3})*)"

    _assert_claim_values(
        text,
        rf"current registry contains\s+{number}\s+included HS6 codes across\s+{number}",
        (summary["included_codes"], summary["active_stages"]),
        "registry included/stage counts",
    )
    _assert_claim_values(
        text,
        rf"scan yields\s+{number}.*?split into\s+{number}\s+included,\s+{number}\s+excluded,\s+and\s+{number}\s+out of stage",
        (
            summary["observable_candidate_records"],
            observable["include"],
            observable["exclude"],
            observable["out_of_stage"],
        ),
        "observable-regex split",
    )
    _assert_claim_values(
        text,
        rf"Adding\s+{number}\s+legacy-only.*?gives a\s+{number}-decision ledger covering\s+{number}\s+unique HS6 codes.*?split is\s+{number}\s+included,\s+{number}\s+excluded,\s+and\s+{number}\s+out of stage",
        (
            summary["legacy_only_records"],
            summary["decision_records"],
            summary["unique_reviewed_hs6"],
            summary["included_codes"],
            summary["excluded_codes"],
            summary["out_of_stage_codes"],
        ),
        "full registry-ledger split",
    )
    _assert_claim_values(
        text,
        rf"retains all\s+{number}\s+previously active codes and\s+adds\s+{number}",
        (summary["historical_active_retained"], summary["new_active_added"]),
        "registry revision counts",
    )
    _assert_claim_values(
        text,
        rf"all\s+{number}\s+(?:rows|product-dictionary rows|source rows)",
        (quality["source_rows_automatically_regex_scanned"],),
        "source dictionary row count",
    )

    table_rows = {}
    for match in re.finditer(
        r"(?m)^(sheep|cotton|aluminium|nickel|cocoa|oilseed-soy)\s*&\s*"
        r"([0-9]+)\s*&\s*([0-9]+)\s*&\s*([0-9]+)\s*&\s*"
        r"([0-9]+)\s*&\s*([0-9]+)\s*&\s*([0-9]+)\s*\\\\\s*$",
        text,
    ):
        table_rows[match.group(1)] = tuple(int(value) for value in match.groups()[1:])
    if table_rows:
        if set(table_rows) != set(CHAINS):
            raise NumberAlignmentError("registry construct table has a partial chain inventory")
        audit_chains = audit.get("chains")
        if not isinstance(audit_chains, dict):
            raise NumberAlignmentError("registry audit chain details are missing")
        for chain in CHAINS:
            row = audit_chains[chain]
            expected = (
                len(row["active_stages"]),
                len(row["capacity_from_stages"]),
                row["active_codes"],
                row["removed_codes"],
                row["out_of_stage_codes"],
                len(row["reassigned_codes"]),
            )
            if table_rows[chain] != expected:
                raise NumberAlignmentError(
                    f"hard-coded registry table row conflicts for {chain}: "
                    f"observed={table_rows[chain]}, expected={expected}"
                )


def _lint_manuscript(
    root: Path,
    *,
    mode: str,
    profile: str,
    paper_macros: Mapping[str, str],
    contemporary_macros: Mapping[str, str],
    profile_payload: Mapping[str, Any],
    audit: Mapping[str, Any],
    evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> int:
    if profile == "repository":
        return 0
    parts: list[str] = []
    for relative in MANUSCRIPT_PARTS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise NumberAlignmentError(f"manuscript part is missing or unsafe: {relative}")
        parts.append(path.read_text(encoding="utf-8"))
    manuscript = "\n".join(parts)
    shadowed = sorted(set(MACRO_DEFINITION.findall(manuscript)))
    if shadowed:
        raise NumberAlignmentError(
            "manuscript redefines generated macros: " + ", ".join(shadowed)
        )

    expected_profile_tex = benchmark_profile.render_tex(
        profile_payload, _sha256(root / PROFILE_JSON)
    )
    profile_macros = {
        match.group(1)
        for line in expected_profile_tex.splitlines()
        if (match := PROFILE_MACRO_DEFINITION.fullmatch(line)) is not None
    }
    known = set(paper_macros) | profile_macros | set(contemporary_macros)
    unknown = sorted(set(MACRO_REFERENCE.findall(manuscript)) - known)
    if unknown:
        raise NumberAlignmentError(
            "manuscript references undefined generated macros: " + ", ".join(unknown)
        )

    for relative in PAPER_WRAPPERS:
        wrapper = (root / relative).read_text(encoding="utf-8")
        required_inputs = (
            r"\input{generated/v2_numbers}",
            r"\input{generated/v2_benchmark_profile}",
            r"\input{generated/v2_contemporary_references}",
        )
        if any(wrapper.count(value) != 1 for value in required_inputs):
            raise NumberAlignmentError(
                f"{relative} must import each generated numerical interface exactly once"
            )

    _lint_registry_literals(manuscript, audit, evidence, protocol)
    _reject_hardcoded_result_decimals(parts[1], parts[2])
    if mode == "release":
        _reject_stale_release_language(manuscript)
    return len(parts)


def _reject_stale_release_language(manuscript: str) -> None:
    lowered = manuscript.lower()
    present = [marker for marker in STALE_RELEASE_MARKERS if marker in lowered]
    if present:
        raise NumberAlignmentError(
            "release manuscript retains stale/pending status language: "
            + ", ".join(present)
        )


def _between(text: str, start: str, end: str, role: str) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        raise NumberAlignmentError(f"cannot locate governed manuscript section: {role}")
    return text[start_index:end_index]


def _reject_hardcoded_result_decimals(body: str, appendix: str) -> None:
    """Require empirical decimal results to enter through generated macros.

    Integer budgets, years, and protocol constants remain ordinary prose.  A
    bare decimal in either governed results section is much more likely to be
    a copied point estimate or interval endpoint.  TeX layout dimensions are
    the only exception; generated macro expansions are not present in source.
    """

    governed = (
        _between(
            body,
            r"\section{Reference results on the frozen future cohort}",
            r"\section{Limitations and open questions}",
            "main results",
        ),
        _between(
            appendix,
            r"\section{Additional reference results}",
            r"\section{Reference implementation details}",
            "appendix results",
        ),
    )
    violations: list[str] = []
    for section in governed:
        for raw_line in section.splitlines():
            line = raw_line.split("%", 1)[0]
            matches = list(BARE_DECIMAL.finditer(line))
            if not matches:
                continue
            if r"\setlength" in line and all(
                line[match.end() :].lstrip().startswith("pt") for match in matches
            ):
                continue
            violations.extend(match.group(0) for match in matches)
    if violations:
        raise NumberAlignmentError(
            "results sections contain bare decimal literals instead of generated macros: "
            + ", ".join(violations[:10])
        )


def _reject_stale_public_status_docs(root: Path) -> None:
    failures: list[str] = []
    for relative in PUBLIC_STATUS_DOCS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise NumberAlignmentError(f"public status document is missing or unsafe: {relative}")
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_PUBLIC_STATUS_PATTERNS:
            if pattern.search(text):
                failures.append(f"{relative.as_posix()}:{pattern.pattern}")
        if relative == Path("results_v2/CLAIM_LEDGER.md"):
            for line_number, line in enumerate(text.splitlines(), start=1):
                if line.startswith("|") and re.search(
                    r"\b(?:pending|superseded)\b", line, flags=re.IGNORECASE
                ):
                    failures.append(f"{relative.as_posix()}:{line_number}:pending-table-row")
                if re.match(r"\s*- \[ \]", line):
                    failures.append(f"{relative.as_posix()}:{line_number}:unchecked-release-item")
    if failures:
        raise NumberAlignmentError(
            "public status documents retain pending/stale release claims: "
            + ", ".join(failures[:20])
        )


def _release_selector_includes(relative: Path, root: Path) -> bool:
    """Return whether an optional prose artifact is selected for public release."""

    name = relative.as_posix()
    selected = name in release_manifest.ROOT_RELEASE_FILES or any(
        name.startswith(prefix) for prefix in release_manifest.RELEASE_PREFIXES
    )
    return (
        selected
        and (root / relative).is_file()
        and public_policy.exclusion_reason(name, root) is None
    )


def _chronology_units(text: str) -> list[str]:
    """Return prose paragraphs and adjacent pairs, excluding fenced commands/comments."""

    without_fences = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    without_comments = "\n".join(
        line for line in without_fences.splitlines() if not line.lstrip().startswith("%")
    )
    paragraphs = [
        re.sub(r"\s+", " ", value).strip()
        for value in re.split(r"\n\s*\n", without_comments)
        if value.strip()
    ]
    units = list(paragraphs)
    units.extend(
        f"{left} {right}"
        for left, right in zip(paragraphs, paragraphs[1:])
        if len(left) + len(right) <= 2000
    )
    return units


def _conditional_review_completion(text: str, start: int) -> bool:
    prefix = text[max(0, start - 48) : start]
    return re.search(
        r"\b(?:if|when|once|after|until|unless|provided that)\s+(?:the\s+)?$",
        prefix,
        flags=re.IGNORECASE,
    ) is not None


def _negated_compute_completion(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 32) : min(len(text), end + 16)]
    completion = (
        r"(?:completed|complete|finished|frozen|sealed|finali[sz]ed|computed|"
        r"recomputed|rebuilt|generated|verified|promoted|locked)"
    )
    return (
        re.search(
            rf"\b(?:not|never)\s+(?:(?:yet|been|fully|formally)\s+){{0,3}}"
            rf"{completion}\b",
            window,
            flags=re.IGNORECASE,
        )
        is not None
        or re.search(
            rf"\bno\s+.{{0,48}}?\b{completion}\b",
            window,
            flags=re.IGNORECASE | re.DOTALL,
        )
        is not None
    )


def _reject_public_chronology_leaks(
    root: Path,
    *,
    review_receipt_verified: bool,
    profile: str = "full",
) -> None:
    """Reject public prose that exposes a compute-before-review chronology.

    Independent gate definitions and conditional rerun rules remain valid.
    An affirmative completed-review statement is allowed only after the
    canonical receipt/protocol gate has succeeded in this same verification.
    """

    if profile not in {"repository", "full"}:
        raise NumberAlignmentError(f"unknown profile: {profile}")
    paths = [
        relative
        for relative in PUBLIC_CHRONOLOGY_DOCS
        if profile == "full" or relative not in MANUSCRIPT_PARTS
    ]
    paths.extend(
        relative
        for relative in OPTIONAL_PUBLIC_CHRONOLOGY_DOCS
        if _release_selector_includes(relative, root)
    )
    chronology_failures: list[str] = []
    unproved_review_claims: list[str] = []
    for relative in paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise NumberAlignmentError(
                f"public chronology document is missing or unsafe: {relative}"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise NumberAlignmentError(
                f"cannot read public chronology document {relative}: {exc}"
            ) from exc
        for unit in _chronology_units(text):
            for pattern in CHRONOLOGY_LEAK_PATTERNS:
                match = pattern.search(unit)
                if match is not None and not _negated_compute_completion(
                    unit, match.start(), match.end()
                ):
                    chronology_failures.append(
                        f"{relative.as_posix()}:{pattern.pattern[:48]}"
                    )
                    break
            if review_receipt_verified:
                continue
            for pattern in REVIEW_COMPLETION_ASSERTIONS:
                for match in pattern.finditer(unit):
                    if not _conditional_review_completion(unit, match.start()):
                        unproved_review_claims.append(
                            f"{relative.as_posix()}:{match.group(0)}"
                        )
                        break

    if chronology_failures:
        raise NumberAlignmentError(
            "public prose leaks a computation-before-human-review chronology: "
            + ", ".join(chronology_failures[:20])
        )
    if unproved_review_claims:
        raise NumberAlignmentError(
            "public prose claims completed human review without a verified canonical receipt: "
            + ", ".join(unproved_review_claims[:20])
        )


def verify_alignment(*, mode: str = "release", profile: str = "full") -> dict[str, int]:
    if mode not in {"interface", "release"}:
        raise NumberAlignmentError(f"unknown mode: {mode}")
    if profile not in {"repository", "full"}:
        raise NumberAlignmentError(f"unknown profile: {profile}")
    root = ROOT

    loco_results.verify_outputs(
        root / "results_v2/metrics/v2_loco_transfer_summary.json",
        root / "results_v2/metrics/v2_loco_transfer_summary.csv",
    )
    if profile == "full":
        ultra_results.verify_outputs(
            root / "results_v2/metrics/v2_ultra_zero_shot_summary.json",
            root / "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
        )
    # In repository mode the paper interface below verifies the exact ULTRA
    # source hashes, while release mode additionally verifies the hash-bound
    # public resolution receipt.  Replaying ULTRA's trained-reference bridge
    # requires the external candidate inventory and is therefore a full-only
    # provenance check.
    benchmark_profile.verify_outputs(
        root / PROFILE_JSON,
        root / PROFILE_TEX,
        mode=profile,
    )
    contemporary_results.verify_outputs(
        root / CONTEMPORARY_JSON,
        root / CONTEMPORARY_CSV,
        root / CONTEMPORARY_TEX,
        config_path=root / CONTEMPORARY_CONFIG,
        profile=profile,
    )
    contemporary_macros = contemporary_results.parse_tex_macros(
        (root / CONTEMPORARY_TEX).read_bytes()
    )
    registry_report = audit_chain_registry.verify_outputs()
    audit = _load_json(root / REGISTRY_AUDIT, "registry audit")
    evidence = _load_json(root / REGISTRY_EVIDENCE, "registry evidence")
    protocol = _load_json(root / CURATION_PROTOCOL, "registry curation protocol")
    if registry_report.get("summary") != audit.get("summary"):
        raise NumberAlignmentError("registry verifier report and committed audit summary differ")

    paper_payload, macros = _paper_interface(root, profile=profile)
    profile_payload = _load_json(
        root / PROFILE_JSON, "benchmark profile", canonical=True
    )
    _profile_paper_crosscheck(profile_payload, macros)

    review_receipt_verified = False
    if mode == "release":
        receipt = registry_human_review_receipt.verify_release_gate(root)
        review_receipt_verified = isinstance(receipt, dict)
        resolve_v2_invalidation.verify_public_receipt(root, profile=profile)
        if profile == "full":
            resolve_v2_invalidation.verify_resolved(root)
            paper_numbers.verify_outputs(
                root / PAPER_TEX,
                root / PAPER_JSON,
                paths=paper_numbers.ArtifactPaths.under(root),
            )

    manuscript_parts = _lint_manuscript(
        root,
        mode=mode,
        profile=profile,
        paper_macros=macros,
        contemporary_macros=contemporary_macros,
        profile_payload=profile_payload,
        audit=audit,
        evidence=evidence,
        protocol=protocol,
    )
    if mode == "release":
        _reject_stale_public_status_docs(root)
        _reject_public_chronology_leaks(
            root,
            review_receipt_verified=review_receipt_verified,
            profile=profile,
        )
    return {
        "paper_macros": len(macros) + len(contemporary_macros),
        "paper_sources": len(paper_payload["sources"]),
        "profile_chains": len(profile_payload["chains"]),
        "manuscript_parts": manuscript_parts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("interface", "release"), default="release"
    )
    parser.add_argument(
        "--profile", choices=("repository", "full"), default="full"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_alignment(mode=args.mode, profile=args.profile)
    except (
        AssertionError,
        ImportError,
        NumberAlignmentError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"V2 NUMBER ALIGNMENT FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "V2 NUMBER ALIGNMENT PASSED "
        f"(mode={args.mode}; profile={args.profile}; "
        f"paper_macros={report['paper_macros']}; "
        f"paper_sources={report['paper_sources']}; "
        f"profile_chains={report['profile_chains']}; "
        f"manuscript_parts={report['manuscript_parts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
