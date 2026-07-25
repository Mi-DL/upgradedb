#!/usr/bin/env python3
"""Resolve the v2 registry-audit hold as one fail-closed promotion gate.

The ordinary paper-number generator intentionally refuses to write while
``INVALIDATED.json`` exists.  This tool breaks that circular dependency
without weakening the generator: it validates every replacement, renders the
current paper interface in a temporary directory, byte-verifies it, and only
then promotes the two generated files and a hash-backed ``RESOLVED`` notice.

No canonical file is written by ``--preview-dir``, ``--dry-run``,
``--verify-public-receipt``, or ``--verify-resolved``.  ``--preview-dir`` is
the review bridge for an as-yet-unfrozen current value digest: it exports the
fully verified candidate JSON/TeX bytes only below an explicit non-canonical
directory.  The public verifier never opens private raw/formal
provenance; the authoritative resolved verifier layers those private checks on
top.  The explicit ``--freshen`` mode conservatively reopens a recognized prior receipt
as the original ACTIVE hold without touching result or paper bytes; the caller
then runs ``--dry-run`` and the normal marker-last resolution transaction.
Each mutating mode requires its exact confirmation token printed by ``--help``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import generate_v2_paper_numbers as paper_numbers  # noqa: E402
import public_release_policy as public_policy  # noqa: E402
import registry_human_review_receipt as human_review  # noqa: E402


NOTICE_PATH = public_policy.PUBLIC_V2_INVALIDATION_NOTICE
PAPER_TEX_PATH = "paper/generated/v2_numbers.tex"
PAPER_JSON_PATH = "results_v2/paper_numbers.json"
GENERATED_PATHS = (PAPER_TEX_PATH, PAPER_JSON_PATH)
CONFIRMATION_TOKEN = "RESOLVE-V2-REGISTRY-AUDIT"
FRESHEN_CONFIRMATION_TOKEN = "FRESHEN-V2-RESOLUTION"

ACTIVE_SCHEMA = public_policy.V2_INVALIDATION_SCHEMA
ACTIVE_STATUS = public_policy.V2_INVALIDATION_ACTIVE_STATUS
ACTIVE_REASON = public_policy.V2_INVALIDATION_REASON
RESOLVED_STATUS = public_policy.V2_INVALIDATION_RESOLVED_STATUS
PAPER_SCHEMA = public_policy.V2_PAPER_NUMBERS_SCHEMA
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")

ACTIVE_FIELDS = public_policy.V2_INVALIDATION_ACTIVE_FIELDS
RESOLVED_FIELDS = public_policy.V2_INVALIDATION_RESOLVED_FIELDS
LEGACY_RESOLVED_FIELDS = public_policy.V2_INVALIDATION_LEGACY_RESOLVED_FIELDS
REVIEW_BINDING_FIELD = public_policy.REGISTRY_HUMAN_REVIEW_BINDING_FIELD

# Primary code surfaces whose exact bytes participated in resolution.  This is
# deliberately explicit rather than a recursive tools-directory inventory.  It
# is owned by public_release_policy so public and private gates share one list.
VERIFIER_SOURCE_PATHS = public_policy.V2_RESOLUTION_VERIFIER_SOURCE_PATHS
LEGACY_SCHEMA4_VERIFIER_SOURCE_PATHS = (
    public_policy.V2_LEGACY_SCHEMA4_VERIFIER_SOURCE_PATHS
)
LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS = (
    public_policy.V2_LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS
)
LEGACY_SCHEMA6_VERIFIER_SOURCE_PATHS = (
    public_policy.V2_LEGACY_SCHEMA6_VERIFIER_SOURCE_PATHS
)
LEGACY_SCHEMA7_VERIFIER_SOURCE_PATHS = (
    public_policy.V2_LEGACY_SCHEMA7_VERIFIER_SOURCE_PATHS
)


class ResolutionError(RuntimeError):
    """The hold cannot be resolved without weakening a required proof."""


@dataclass(frozen=True)
class ResolutionPlan:
    """Completely verified bytes and hashes awaiting the atomic promotion."""

    root: Path
    active_notice_bytes: bytes
    active_notice: Mapping[str, Any]
    generated_bytes: Mapping[str, bytes]
    canonical_before: Mapping[str, bytes | None]
    fixed_replacement_sha256: Mapping[str, str]
    replacement_sha256: Mapping[str, str]
    source_sha256: Mapping[str, str]
    verifier_sha256: Mapping[str, str]
    review_binding: Mapping[str, str]
    resolved_notice_bytes: bytes


@dataclass(frozen=True)
class PaperPreview:
    """Verified current-schema candidate bytes exported outside canonical paths."""

    root: Path
    preview_root: Path
    generated_bytes: Mapping[str, bytes]
    source_sha256: Mapping[str, str]
    number_key_count: int
    number_keys_sha256: str
    number_values_sha256: str


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _strict_json_bytes(content: bytes, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResolutionError(f"{role}: cannot read strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResolutionError(f"{role}: top-level JSON value must be an object")
    return payload


def _strict_json_file(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ResolutionError(f"{role}: cannot read {path}: {exc}") from exc
    return _strict_json_bytes(content, role), content


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResolutionError(f"cannot render deterministic strict JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _strict_canonical_json_file(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    payload, content = _strict_json_file(path, role)
    if content != _json_bytes(payload):
        raise ResolutionError(f"{role}: bytes are not canonical JSON")
    return payload, content


def _canonical_root(root: Path) -> Path:
    lexical = Path(root)
    if lexical.is_symlink():
        raise ResolutionError("repository root must not be a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ResolutionError(f"repository root cannot be resolved safely: {exc}") from exc
    if not resolved.is_dir():
        raise ResolutionError("repository root is not a directory")
    return resolved


def _safe_path(root: Path, relative: str, *, require_file: bool) -> Path:
    reason = public_policy.source_path_reason(relative, root, require_file=require_file)
    if reason is not None:
        raise ResolutionError(f"unsafe resolution path {relative!r}: {reason}")
    return root / relative


def _normalize_scope(scope: Any) -> tuple[tuple[str, ...], frozenset[str]]:
    if not isinstance(scope, list) or not scope:
        raise ResolutionError("invalidation scope must be a non-empty list")
    original: list[str] = []
    normalized: list[str] = []
    for item in scope:
        if not isinstance(item, str):
            raise ResolutionError("invalidation scope entries must be strings")
        reason = public_policy.canonical_path_reason(item)
        if reason is not None:
            raise ResolutionError(f"unsafe invalidation scope entry {item!r}: {reason}")
        canonical = item if "/" in item else f"results_v2/metrics/{item}"
        if item != canonical and canonical not in public_policy.V2_INVALIDATION_DERIVED_PATHS:
            raise ResolutionError(f"non-normative basename in invalidation scope: {item!r}")
        original.append(item)
        normalized.append(canonical)
    if len(set(normalized)) != len(normalized):
        raise ResolutionError("invalidation scope contains duplicate normalized paths")
    expected = frozenset(public_policy.V2_INVALIDATION_DERIVED_PATHS)
    observed = frozenset(normalized)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ResolutionError(
            "invalidation scope differs from the complete normative scope: "
            f"missing={missing}, extra={extra}"
        )
    return tuple(original), observed


def _require_nonempty_text(payload: Mapping[str, Any], key: str, role: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResolutionError(f"{role}: {key} must be a non-empty string")
    return value


def _validate_active_notice(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if set(payload) != set(ACTIVE_FIELDS):
        raise ResolutionError(
            "active invalidation notice has an unexpected field inventory: "
            f"expected={sorted(ACTIVE_FIELDS)}, observed={sorted(payload)}"
        )
    if payload.get("schema_version") != ACTIVE_SCHEMA:
        raise ResolutionError("active invalidation notice schema is not the frozen original")
    if payload.get("status") != ACTIVE_STATUS:
        raise ResolutionError("active invalidation notice status is not the frozen original")
    if payload.get("reason") != ACTIVE_REASON:
        raise ResolutionError("active invalidation notice reason is not the frozen original")
    for key in ("invalidated_at", "claim_policy", "resolution"):
        _require_nonempty_text(payload, key, "active invalidation notice")
    if payload.get("invalidated_at") != public_policy.V2_INVALIDATION_DATE:
        raise ResolutionError("active invalidated_at changed the frozen original date")
    original, _ = _normalize_scope(payload.get("scope"))
    return original


def _validate_digest_map(value: Any, role: str, *, expected_paths: set[str] | None = None) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ResolutionError(f"{role} must be a non-empty object")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or public_policy.canonical_path_reason(path) is not None:
            raise ResolutionError(f"{role} contains an unsafe path")
        if not isinstance(digest, str) or HEX64_RE.fullmatch(digest) is None:
            raise ResolutionError(f"{role} contains an invalid SHA-256 digest for {path!r}")
        result[path] = digest
    if expected_paths is not None and set(result) != expected_paths:
        raise ResolutionError(
            f"{role} inventory mismatch: missing={sorted(expected_paths - set(result))}, "
            f"extra={sorted(set(result) - expected_paths)}"
        )
    return result


def _validate_resolved_notice(
    payload: Mapping[str, Any],
    *,
    expected_verifier_paths: set[str] | None = None,
) -> tuple[str, ...]:
    current_inventory = set(VERIFIER_SOURCE_PATHS)
    expected_inventory = (
        current_inventory
        if expected_verifier_paths is None
        else set(expected_verifier_paths)
    )
    observed_fields = set(payload)
    allowed_fields = {frozenset(RESOLVED_FIELDS)}
    if expected_inventory != current_inventory:
        # Historical receipts are recognized only by the private freshen
        # transition.  They predate the human-review binding, while a receipt
        # freshly generated by current code may retain that binding even after
        # its verifier inventory is reduced in migration tests.
        allowed_fields.add(frozenset(LEGACY_RESOLVED_FIELDS))
    if frozenset(observed_fields) not in allowed_fields:
        raise ResolutionError(
            "resolved invalidation notice has an unexpected field inventory: "
            f"allowed={[sorted(fields) for fields in sorted(allowed_fields, key=len)]}, "
            f"observed={sorted(payload)}"
        )
    if REVIEW_BINDING_FIELD in payload:
        try:
            public_policy._validate_registry_human_review_binding(
                payload.get(REVIEW_BINDING_FIELD)
            )
        except ValueError as exc:
            raise ResolutionError(
                f"resolved notice registry human-review binding is invalid: {exc}"
            ) from exc
    elif expected_inventory == current_inventory:
        raise ResolutionError(
            "current resolved notice is missing its registry human-review binding"
        )
    if payload.get("schema_version") != ACTIVE_SCHEMA:
        raise ResolutionError("resolved notice changed the original invalidation schema")
    if payload.get("status") != RESOLVED_STATUS or payload.get("original_status") != ACTIVE_STATUS:
        raise ResolutionError("resolved notice has an invalid status transition")
    if payload.get("reason") != ACTIVE_REASON:
        raise ResolutionError("resolved notice changed the original invalidation reason")
    for key in ("invalidated_at", "claim_policy", "resolution", "resolved_at"):
        _require_nonempty_text(payload, key, "resolved invalidation notice")
    if payload.get("invalidated_at") != public_policy.V2_INVALIDATION_DATE:
        raise ResolutionError("resolved notice changed the frozen original date")
    resolved_text = str(payload.get("resolved_at"))
    try:
        resolved_at = datetime.strptime(resolved_text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ResolutionError(
            "resolved_at must be a canonical UTC timestamp at whole-second precision"
        ) from exc
    if resolved_text != resolved_at.strftime("%Y-%m-%dT%H:%M:%SZ"):
        raise ResolutionError("resolved_at is not canonical UTC")
    invalidated = datetime.strptime(
        public_policy.V2_INVALIDATION_DATE,
        "%Y-%m-%d",
    ).replace(tzinfo=timezone.utc)
    if resolved_at < invalidated:
        raise ResolutionError("resolved_at predates the original invalidation")
    original, normalized = _normalize_scope(payload.get("scope"))
    _validate_digest_map(
        payload.get("replacement_sha256"),
        "replacement_sha256",
        expected_paths=set(normalized),
    )
    gate_digest = payload.get("resolution_gate_sha256")
    if not isinstance(gate_digest, str) or HEX64_RE.fullmatch(gate_digest) is None:
        raise ResolutionError("resolution_gate_sha256 is invalid")
    _validate_digest_map(payload.get("resolution_source_sha256"), "resolution_source_sha256")
    verifiers = _validate_digest_map(
        payload.get("resolution_verifier_sha256"),
        "resolution_verifier_sha256",
        expected_paths=(
            expected_inventory
        ),
    )
    if gate_digest != verifiers["tools/resolve_v2_invalidation.py"]:
        raise ResolutionError("resolution gate hash differs from the verifier map")
    return original


def _freshened_active_from_resolved(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a recognized prior receipt and recover its frozen ACTIVE hold.

    A prior receipt may be stale precisely because generated paper bytes or
    verifier sources changed.  Refresh therefore validates its immutable
    historical fields and a closed, explicitly recognized verifier inventory,
    but does not trust its old replacement hashes.  All current artifacts and
    current verifier bytes are independently rerun before a new receipt can be
    atomically promoted.
    """

    if payload.get("status") != RESOLVED_STATUS:
        raise ResolutionError(
            "freshen requires a recognized prior RESOLVED receipt"
        )

    verifier_map = payload.get("resolution_verifier_sha256")
    observed_paths = (
        frozenset(verifier_map) if isinstance(verifier_map, dict) else frozenset()
    )
    allowed_inventories = (
        frozenset(LEGACY_SCHEMA4_VERIFIER_SOURCE_PATHS),
        frozenset(LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS),
        frozenset(LEGACY_SCHEMA6_VERIFIER_SOURCE_PATHS),
        frozenset(LEGACY_SCHEMA7_VERIFIER_SOURCE_PATHS),
        frozenset(VERIFIER_SOURCE_PATHS),
    )
    if observed_paths not in allowed_inventories:
        raise ResolutionError(
            "prior resolved notice has an unrecognized verifier inventory"
        )
    _validate_resolved_notice(
        payload,
        expected_verifier_paths=set(observed_paths),
    )
    base = {field: payload[field] for field in ACTIVE_FIELDS}
    base["status"] = ACTIVE_STATUS
    _validate_active_notice(base)
    return base


def _current_verifier_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in VERIFIER_SOURCE_PATHS:
        path = _safe_path(root, relative, require_file=True)
        result[relative] = _sha256_file(path)
    return result


def _verify_cpu_csv_bytes(root: Path) -> None:
    """Reconstruct the canonical CPU CSV byte-for-byte from its schema-2 JSON."""
    import pandas as pd
    import v2_rolling_cpu_baselines as cpu

    json_path = root / "results_v2/metrics/rolling_cpu_baselines.json"
    csv_path = root / "results_v2/metrics/rolling_cpu_baselines.csv"
    payload, _ = _strict_json_file(json_path, "rolling CPU JSON")
    # The JSON is key-sorted on disk, whereas the CSV was emitted from the
    # protocol-fixed insertion order before JSON serialization.  Restore only
    # that preregistered order before calling the generator's own row exporter.
    model_order = {
        "track_a_destination_extension": (
            "size",
            "gravity",
            "historical_logistic_size_gravity",
        ),
        "track_b1_processed_export_stage_entry": (
            "upstream_capacity",
            "historical_logistic_structural",
        ),
        "track_b2_conditional_destination_ranking": (
            "processed_importer_demand",
            "gravity",
            "historical_logistic_demand_gravity",
        ),
    }
    chains = payload.get("chains")
    if not isinstance(chains, dict) or set(chains) != set(cpu.CHAINS):
        raise ResolutionError("rolling CPU JSON does not contain the exact six chains")
    payload["chains"] = {chain: chains[chain] for chain in cpu.CHAINS}
    for chain in payload["chains"].values():
        for track_name, models in model_order.items():
            track = chain.get(track_name)
            if not isinstance(track, dict) or not isinstance(track.get("models"), dict):
                raise ResolutionError(f"rolling CPU JSON lacks {track_name} models")
            if set(track["models"]) != set(models):
                raise ResolutionError(f"rolling CPU JSON has a non-canonical {track_name} model set")
            track["models"] = {model: track["models"][model] for model in models}
    buffer = io.StringIO(newline="")
    pd.DataFrame(cpu._csv_rows(payload)).to_csv(buffer, index=False)
    expected = buffer.getvalue().encode("utf-8")
    try:
        actual = csv_path.read_bytes()
    except OSError as exc:
        raise ResolutionError(f"cannot read rolling CPU CSV: {exc}") from exc
    if actual != expected:
        raise ResolutionError("rolling CPU CSV is not byte-exact for its schema-2 JSON")


def _run_authoritative_verifiers(root: Path) -> None:
    """Run every current scientific verifier without writing any output."""
    if root != ROOT:
        raise ResolutionError(
            "authoritative verifier modules are bound to this checkout; "
            "non-default roots require injected test verifiers"
        )
    import audit_chain_registry
    import audit_v2
    import build_gpu_step3_postfreeze_attestation as gpu_postfreeze
    import summarize_v2_gpu_results as gpu_summary
    import summarize_v2_loco_results as loco_summary
    import summarize_v2_ultra_results as ultra_summary
    import v2_b1_coverage
    import v2_gbdt_baselines as gbdt_summary
    import v2_robustness
    import v2_rolling_cpu_baselines as cpu

    audit_chain_registry.verify_outputs()
    audit_v2.verify_existing_output(root / "results_v2/metrics/raw_label_audit.json")
    v2_b1_coverage._verify_report(
        root / "results_v2/metrics/b1_candidate_coverage.json",
        None,
    )
    cpu.verify_existing_output(root / "results_v2/metrics/rolling_cpu_baselines.json")
    _verify_cpu_csv_bytes(root)
    v2_robustness.verify_existing_output(
        root / "results_v2/metrics/v2_robustness.json",
        root / "results_v2/metrics/v2_robustness.csv",
    )
    verified_gpu = gpu_summary.verify_outputs(
        root / "results_v2/gpu_rolling",
        root / "results_v2/metrics/v2_gpu_rolling_summary.json",
        root / "results_v2/metrics/v2_gpu_rolling_summary.csv",
    )
    gpu_postfreeze.verify_summary_binding(
        verified_gpu,
        artifact_path=root / gpu_postfreeze.ARTIFACT_ROLE,
        root=root,
        require_full_inventory=True,
    )
    loco_summary.verify_outputs(
        root / "results_v2/metrics/v2_loco_transfer_summary.json",
        root / "results_v2/metrics/v2_loco_transfer_summary.csv",
    )
    ultra_summary.verify_outputs(
        root / "results_v2/metrics/v2_ultra_zero_shot_summary.json",
        root / "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
    )
    gbdt_summary.verify_existing_output(
        root / "results_v2/metrics/v2_gbdt_baselines.json",
        root / "results_v2/metrics/v2_gbdt_baselines.csv",
    )


def _paper_payload(numbers: Mapping[str, str], sources: Mapping[str, str]) -> dict[str, Any]:
    if paper_numbers.PAPER_NUMBERS_SCHEMA != PAPER_SCHEMA:
        raise ResolutionError(
            f"paper generator schema changed: expected {PAPER_SCHEMA!r}, "
            f"observed {paper_numbers.PAPER_NUMBERS_SCHEMA!r}"
        )
    if numbers.get("VTwoGPUStatus") != "COMPLETE":
        raise ResolutionError("formal GPU summary is not complete")
    if numbers.get("VTwoLOCOStatus") != "COMPLETE":
        raise ResolutionError("formal LOCO summary is not complete")
    if numbers.get("VTwoULTRAStatus") != "COMPLETE":
        raise ResolutionError("formal ULTRA summary is not complete")
    if numbers.get("VTwoGBDTStatus") != "COMPLETE":
        raise ResolutionError("formal GBDT summary is not complete")
    if numbers.get("VTwoProductSpaceStatus") != "COMPLETE":
        raise ResolutionError("product-space summary is not complete")
    if numbers.get("VTwoScoreRobustnessRFiveStatus") != "COMPLETE":
        raise ResolutionError("r5 score-robustness summary is not complete")
    if numbers.get("VTwoEligibilityThresholdStatus") != "COMPLETE":
        raise ResolutionError("eligibility-threshold geometry is not complete")
    return {
        "schema_version": paper_numbers.PAPER_NUMBERS_SCHEMA,
        "benchmark_version": paper_numbers.BENCHMARK_VERSION,
        "status": "complete",
        "gpu_status": "COMPLETE",
        "loco_status": "COMPLETE",
        "ultra_status": "COMPLETE",
        "gbdt_status": "COMPLETE",
        "sources": dict(sources),
        "numbers": dict(numbers),
    }


def _validate_source_hashes(root: Path, sources: Mapping[str, str]) -> dict[str, str]:
    validated = _validate_digest_map(dict(sources), "paper source SHA-256")
    for relative, expected in validated.items():
        path = _safe_path(root, relative, require_file=True)
        if _sha256_file(path) != expected:
            raise ResolutionError(f"paper source changed during collection: {relative}")
    return dict(sorted(validated.items()))


def _render_and_verify_paper(
    root: Path,
    *,
    allow_unfrozen_final_values: bool = False,
) -> tuple[dict[str, bytes], dict[str, str]]:
    paths = paper_numbers.ArtifactPaths.under(root)
    numbers, sources = paper_numbers.collect_numbers(
        paths,
        require_gpu=True,
        require_loco=True,
        require_ultra=True,
        require_gbdt=True,
    )
    validated_sources = _validate_source_hashes(root, sources)
    expected_source_paths = set(public_policy.V2_PAPER_SOURCE_PATHS)
    if set(validated_sources) != expected_source_paths:
        raise ResolutionError(
            "current paper source inventory mismatch: "
            f"missing={sorted(expected_source_paths - set(validated_sources))}, "
            f"extra={sorted(set(validated_sources) - expected_source_paths)}"
        )
    payload = _paper_payload(numbers, validated_sources)
    expected = {
        PAPER_TEX_PATH: paper_numbers.render_tex(numbers, validated_sources).encode("utf-8"),
        PAPER_JSON_PATH: _json_bytes(payload),
    }

    # Use a genuinely separate temporary directory and the generator's own
    # verifier.  The invalidation path is redirected only for this private
    # temporary verification; canonical generator writes remain blocked.
    with tempfile.TemporaryDirectory(prefix="v2-resolution-paper-") as temp_name:
        temp = Path(temp_name)
        tex_path = temp / "v2_numbers.tex"
        json_path = temp / "paper_numbers.json"
        tex_path.write_bytes(expected[PAPER_TEX_PATH])
        json_path.write_bytes(expected[PAPER_JSON_PATH])
        bypass_paths = replace(paths, invalidation=temp / "ABSENT_INVALIDATION.json")
        paper_numbers.verify_outputs(tex_path, json_path, paths=bypass_paths)
        if tex_path.read_bytes() != expected[PAPER_TEX_PATH]:
            raise ResolutionError("temporary current-schema TeX failed byte verification")
        if json_path.read_bytes() != expected[PAPER_JSON_PATH]:
            raise ResolutionError("temporary current-schema JSON failed byte verification")

    # A final independent collection closes a source-change race around the
    # temporary verifier.
    numbers_after, sources_after = paper_numbers.collect_numbers(
        paths,
        require_gpu=True,
        require_loco=True,
        require_ultra=True,
        require_gbdt=True,
    )
    validated_after = _validate_source_hashes(root, sources_after)
    if numbers_after != numbers or validated_after != validated_sources:
        raise ResolutionError("paper-number inputs changed during temporary verification")
    try:
        validated_numbers = public_policy._validate_paper_numbers(
            numbers_after,
            allow_unfrozen_inventory=allow_unfrozen_final_values,
        )
        if not (
            allow_unfrozen_final_values
            and public_policy.V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256 is None
        ):
            public_policy._validate_final_paper_number_value_digest(validated_numbers)
    except ValueError as exc:
        raise ResolutionError(f"current final paper-number contract failed: {exc}") from exc
    return expected, validated_sources


def _scope_path_checks(root: Path, *, allow_generated_missing: bool) -> None:
    expected = set(public_policy.V2_INVALIDATION_DERIVED_PATHS)
    for relative in sorted(expected):
        require = relative not in GENERATED_PATHS or not allow_generated_missing
        _safe_path(root, relative, require_file=require)
    _safe_path(root, NOTICE_PATH, require_file=True)


def _fixed_replacement_hashes(root: Path) -> dict[str, str]:
    fixed = set(public_policy.V2_INVALIDATION_DERIVED_PATHS) - set(GENERATED_PATHS)
    return {
        relative: _sha256_file(_safe_path(root, relative, require_file=True))
        for relative in sorted(fixed)
    }


def _read_optional_regular(root: Path, relative: str) -> bytes | None:
    path = _safe_path(root, relative, require_file=False)
    if not path.exists():
        return None
    if not path.is_file():
        raise ResolutionError(f"canonical destination is not a regular file: {relative}")
    return path.read_bytes()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _verify_human_review_release(root: Path) -> dict[str, Any]:
    """Require the same release-eligible review receipt used by public smoke."""

    try:
        return human_review.verify_release_gate(root)
    except (OSError, ValueError) as exc:
        raise ResolutionError(
            f"registry human-review release gate is not satisfied: {exc}"
        ) from exc


def _review_binding_from_verified_receipt(
    root: Path, receipt: Mapping[str, Any]
) -> dict[str, str]:
    """Seal the exact canonical review artifacts returned by the semantic gate."""

    audit_id = receipt.get("audit_id")
    if not isinstance(audit_id, str) or not audit_id.strip() or audit_id != audit_id.strip():
        raise ResolutionError("registry human-review receipt audit_id is invalid")
    disposition = receipt.get("disposition")
    if not isinstance(disposition, dict) or disposition.get("kind") != "NO_CONSTRUCT_CHANGE":
        raise ResolutionError(
            "registry human-review receipt is not a release-eligible no-change review"
        )
    freeze = receipt.get("pre_review_freeze")
    if not isinstance(freeze, dict) or set(freeze) != {"path", "sha256"}:
        raise ResolutionError("registry human-review receipt freeze binding is incomplete")
    if freeze.get("path") != public_policy.REGISTRY_HUMAN_REVIEW_FREEZE:
        raise ResolutionError("registry human-review receipt freeze path is not canonical")
    freeze_digest = freeze.get("sha256")
    if not isinstance(freeze_digest, str) or HEX64_RE.fullmatch(freeze_digest) is None:
        raise ResolutionError("registry human-review receipt freeze digest is invalid")

    paths = {
        "receipt": public_policy.REGISTRY_HUMAN_REVIEW_RECEIPT,
        "protocol": public_policy.REGISTRY_HUMAN_REVIEW_PROTOCOL,
        "freeze": public_policy.REGISTRY_HUMAN_REVIEW_FREEZE,
    }
    digests = {
        role: _sha256_file(_safe_path(root, relative, require_file=True))
        for role, relative in paths.items()
    }
    if digests["freeze"] != freeze_digest:
        raise ResolutionError(
            "registry human-review receipt does not bind the canonical pre-review freeze bytes"
        )
    binding = {
        "audit_id": audit_id,
        "disposition": str(disposition["kind"]),
        "receipt_path": paths["receipt"],
        "receipt_sha256": digests["receipt"],
        "protocol_path": paths["protocol"],
        "protocol_sha256": digests["protocol"],
        "freeze_path": paths["freeze"],
        "freeze_sha256": digests["freeze"],
    }
    try:
        return public_policy._validate_registry_human_review_binding(binding)
    except ValueError as exc:  # pragma: no cover - defensive parity with public verifier
        raise ResolutionError(f"registry human-review binding is invalid: {exc}") from exc


def _current_human_review_binding(root: Path) -> dict[str, str]:
    receipt = _verify_human_review_release(root)
    return _review_binding_from_verified_receipt(root, receipt)


def _require_unchanged_human_review_binding(
    root: Path, expected: Mapping[str, str]
) -> dict[str, str]:
    observed = _current_human_review_binding(root)
    if observed != dict(expected):
        raise ResolutionError(
            "registry human-review receipt, protocol, freeze, audit id, or disposition "
            "changed during resolution"
        )
    return observed


def prepare_resolution(root: Path = ROOT, *, now: datetime | None = None) -> ResolutionPlan:
    """Validate an explicit ACTIVE hold and build a plan without canonical writes."""
    root = _canonical_root(root)
    _scope_path_checks(root, allow_generated_missing=True)
    notice_path = root / NOTICE_PATH
    active, active_bytes = _strict_canonical_json_file(
        notice_path,
        "active invalidation notice",
    )
    _validate_active_notice(active)
    review_binding = _current_human_review_binding(root)

    verifier_hashes = _current_verifier_hashes(root)
    _run_authoritative_verifiers(root)
    generated, source_hashes = _render_and_verify_paper(root)
    fixed_hashes = _fixed_replacement_hashes(root)
    replacements = dict(fixed_hashes)
    replacements.update(
        {relative: _sha256_bytes(generated[relative]) for relative in GENERATED_PATHS}
    )
    expected_scope = set(public_policy.V2_INVALIDATION_DERIVED_PATHS)
    if set(replacements) != expected_scope:
        raise ResolutionError("internal error: replacement proof inventory is not exact")

    resolved_time = now if now is not None else _utc_now()
    if resolved_time.tzinfo is None:
        raise ResolutionError("resolution time must include a UTC offset")
    resolved_at = resolved_time.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    resolved: dict[str, Any] = dict(active)
    resolved.update(
        {
            "status": RESOLVED_STATUS,
            "original_status": ACTIVE_STATUS,
            "resolved_at": resolved_at,
            "replacement_sha256": dict(sorted(replacements.items())),
            "resolution_gate_sha256": verifier_hashes[
                "tools/resolve_v2_invalidation.py"
            ],
            "resolution_source_sha256": dict(sorted(source_hashes.items())),
            "resolution_verifier_sha256": dict(sorted(verifier_hashes.items())),
            REVIEW_BINDING_FIELD: dict(review_binding),
        }
    )
    _validate_resolved_notice(resolved)

    canonical_before = {
        relative: _read_optional_regular(root, relative) for relative in GENERATED_PATHS
    }
    # Preparation itself is read-only: detect concurrent proof/code edits
    # before returning even in --dry-run mode.
    if notice_path.read_bytes() != active_bytes:
        raise ResolutionError("active invalidation notice changed during preparation")
    if _fixed_replacement_hashes(root) != fixed_hashes:
        raise ResolutionError("a fixed-scope replacement changed during preparation")
    if _current_verifier_hashes(root) != verifier_hashes:
        raise ResolutionError("a resolution verifier changed during preparation")
    _validate_source_hashes(root, source_hashes)
    _require_unchanged_human_review_binding(root, review_binding)
    return ResolutionPlan(
        root=root,
        active_notice_bytes=active_bytes,
        active_notice=active,
        generated_bytes=generated,
        canonical_before=canonical_before,
        fixed_replacement_sha256=fixed_hashes,
        replacement_sha256=dict(sorted(replacements.items())),
        source_sha256=source_hashes,
        verifier_sha256=verifier_hashes,
        review_binding=review_binding,
        resolved_notice_bytes=_json_bytes(resolved),
    )


def _precommit_guard(plan: ResolutionPlan) -> None:
    root = plan.root
    _scope_path_checks(root, allow_generated_missing=True)
    if (root / NOTICE_PATH).read_bytes() != plan.active_notice_bytes:
        raise ResolutionError("invalidation notice changed before promotion")
    for relative, before in plan.canonical_before.items():
        if _read_optional_regular(root, relative) != before:
            raise ResolutionError(f"canonical destination changed before promotion: {relative}")
    if _fixed_replacement_hashes(root) != dict(plan.fixed_replacement_sha256):
        raise ResolutionError("a fixed-scope replacement changed before promotion")
    if _current_verifier_hashes(root) != dict(plan.verifier_sha256):
        raise ResolutionError("a resolution verifier changed before promotion")
    _run_authoritative_verifiers(root)
    generated, sources = _render_and_verify_paper(root)
    if generated != dict(plan.generated_bytes) or sources != dict(plan.source_sha256):
        raise ResolutionError("paper outputs or source hashes changed before promotion")
    # Keep this as the final precommit check: the semantic gate is rerun and
    # the exact receipt/protocol/freeze bytes plus public identifiers must still
    # equal the preparation snapshot.
    _require_unchanged_human_review_binding(root, plan.review_binding)


def _stage_same_directory(target: Path, content: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.resolve-",
        dir=target.parent,
    )
    path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return path


def _replace_path(source: Path, target: Path) -> None:
    """Patch point for failure-injection tests; production uses os.replace."""
    os.replace(source, target)


def _preview_targets(root: Path, preview_root: Path) -> dict[str, Path]:
    lexical = Path(preview_root)
    if lexical.exists() and (lexical.is_symlink() or not lexical.is_dir()):
        raise ResolutionError("preview destination must be a real directory")
    try:
        resolved = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ResolutionError(f"preview destination cannot be resolved safely: {exc}") from exc
    if resolved == root:
        raise ResolutionError("preview destination must not be the repository root")

    canonical = {
        (root / relative).resolve(strict=False) for relative in GENERATED_PATHS
    }
    targets = {
        relative: resolved / Path(relative) for relative in GENERATED_PATHS
    }
    for relative, target in targets.items():
        try:
            resolved_target = target.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ResolutionError(
                f"preview target cannot be resolved safely: {relative}: {exc}"
            ) from exc
        if resolved_target in canonical:
            raise ResolutionError(
                f"preview target aliases a canonical paper output: {relative}"
            )
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise ResolutionError(
                f"preview target must be absent or a regular file: {target}"
            )
    return targets


def write_paper_preview(
    root: Path = ROOT,
    *,
    preview_root: Path,
) -> PaperPreview:
    """Export reviewable current-schema bytes without relaxing canonical promotion."""

    root = _canonical_root(root)
    targets = _preview_targets(root, preview_root)
    _scope_path_checks(root, allow_generated_missing=True)
    notice_path = root / NOTICE_PATH
    active, active_bytes = _strict_canonical_json_file(
        notice_path,
        "active invalidation notice",
    )
    _validate_active_notice(active)

    verifier_hashes = _current_verifier_hashes(root)
    fixed_hashes = _fixed_replacement_hashes(root)
    _run_authoritative_verifiers(root)
    generated, sources = _render_and_verify_paper(
        root,
        allow_unfrozen_final_values=True,
    )
    payload = _strict_json_bytes(
        generated[PAPER_JSON_PATH],
        "current-schema paper preview",
    )
    try:
        numbers = public_policy._validate_paper_numbers(
            payload.get("numbers"),
            allow_unfrozen_inventory=True,
        )
    except ValueError as exc:
        raise ResolutionError(f"current-schema paper preview is invalid: {exc}") from exc
    value_digest = public_policy._paper_number_value_digest(numbers)

    # The preview is written only after the same repository/source race guards
    # used by the canonical preparation path have remained stable.
    if notice_path.read_bytes() != active_bytes:
        raise ResolutionError("active invalidation notice changed during preview")
    if _fixed_replacement_hashes(root) != fixed_hashes:
        raise ResolutionError("a fixed-scope replacement changed during preview")
    if _current_verifier_hashes(root) != verifier_hashes:
        raise ResolutionError("a resolution verifier changed during preview")
    _validate_source_hashes(root, sources)

    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    targets = _preview_targets(root, Path(preview_root))
    staged: dict[Path, Path] = {}
    try:
        for relative, target in targets.items():
            staged[target] = _stage_same_directory(target, generated[relative])
        for relative, target in targets.items():
            _replace_path(staged[target], target)
            if target.read_bytes() != generated[relative]:
                raise ResolutionError(f"preview byte verification failed: {relative}")
    finally:
        for temporary in staged.values():
            with contextlib.suppress(OSError):
                temporary.unlink()

    return PaperPreview(
        root=root,
        preview_root=Path(preview_root).resolve(strict=True),
        generated_bytes=generated,
        source_sha256=sources,
        number_key_count=len(numbers),
        number_keys_sha256=public_policy._paper_number_key_digest(numbers),
        number_values_sha256=value_digest,
    )


def _restore_snapshot(target: Path, original: bytes | None) -> None:
    if original is None:
        if target.exists() or target.is_symlink():
            target.unlink()
        return
    temporary = _stage_same_directory(target, original)
    try:
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


@contextlib.contextmanager
def _resolution_lock(root: Path) -> Iterator[None]:
    lock = root / "results_v2/metrics/.resolve_v2_invalidation.lock"
    _safe_path(
        root,
        "results_v2/metrics/.resolve_v2_invalidation.lock",
        require_file=False,
    )
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ResolutionError(f"another invalidation gate holds {lock.name}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            lock.unlink()


def freshen_resolution(
    root: Path = ROOT,
    *,
    confirmation: str,
) -> dict[str, Any]:
    """Atomically reopen a recognized prior RESOLVED receipt as ACTIVE.

    Freshening is intentionally a separate, conservative mutation.  It writes
    only the notice (the sole marker and therefore marker-last), never paper or
    result bytes.  The caller must subsequently run ``--dry-run`` and the normal
    confirmed resolution transaction to promote current outputs.
    """

    if confirmation != FRESHEN_CONFIRMATION_TOKEN:
        raise ResolutionError(
            "freshening requires --confirm " + FRESHEN_CONFIRMATION_TOKEN
        )
    root = _canonical_root(root)
    with _resolution_lock(root):
        _scope_path_checks(root, allow_generated_missing=True)
        notice_path = root / NOTICE_PATH
        prior, prior_bytes = _strict_canonical_json_file(
            notice_path,
            "prior resolved invalidation notice",
        )
        active = _freshened_active_from_resolved(prior)
        active_bytes = _json_bytes(active)
        if notice_path.read_bytes() != prior_bytes:
            raise ResolutionError("resolved notice changed before freshening")

        staged = _stage_same_directory(notice_path, active_bytes)
        attempted = False
        try:
            attempted = True
            _replace_path(staged, notice_path)
            if notice_path.read_bytes() != active_bytes:
                raise ResolutionError("freshened ACTIVE notice byte verification failed")
            observed, observed_bytes = _strict_canonical_json_file(
                notice_path,
                "freshened active invalidation notice",
            )
            _validate_active_notice(observed)
            if observed != active or observed_bytes != active_bytes:
                raise ResolutionError("freshened ACTIVE notice is not canonical")
            if public_policy.unresolved_v2_invalidation(root) is None:
                raise ResolutionError("freshened ACTIVE notice did not reopen the release hold")
        except BaseException as exc:
            rollback_error: BaseException | None = None
            if attempted:
                try:
                    _restore_snapshot(notice_path, prior_bytes)
                    if notice_path.read_bytes() != prior_bytes:
                        raise ResolutionError("prior receipt byte restoration check failed")
                except BaseException as restore_exc:  # pragma: no cover - catastrophic I/O
                    rollback_error = restore_exc
            if rollback_error is not None:
                raise ResolutionError(
                    "freshening failed and rollback could not restore the prior receipt"
                ) from rollback_error
            raise
        finally:
            with contextlib.suppress(OSError):
                staged.unlink()
    return active


def _transaction(plan: ResolutionPlan) -> None:
    root = plan.root
    destinations = [root / relative for relative in GENERATED_PATHS] + [root / NOTICE_PATH]
    snapshots: dict[Path, bytes | None] = {
        root / relative: plan.canonical_before[relative] for relative in GENERATED_PATHS
    }
    snapshots[root / NOTICE_PATH] = plan.active_notice_bytes
    contents = {
        root / PAPER_TEX_PATH: plan.generated_bytes[PAPER_TEX_PATH],
        root / PAPER_JSON_PATH: plan.generated_bytes[PAPER_JSON_PATH],
        root / NOTICE_PATH: plan.resolved_notice_bytes,
    }
    staged: dict[Path, Path] = {}
    attempted: list[Path] = []
    try:
        for target in destinations:
            staged[target] = _stage_same_directory(target, contents[target])
        for target in destinations:  # marker is intentionally last
            if target == root / NOTICE_PATH:
                # Re-run the canonical semantic review gate at the last possible
                # moment.  A receipt/protocol/freeze change after the expensive
                # precommit verification must never be sealed by the marker.
                _require_unchanged_human_review_binding(root, plan.review_binding)
            attempted.append(target)
            _replace_path(staged[target], target)
            if _sha256_file(target) != _sha256_bytes(contents[target]):
                raise ResolutionError(f"atomic replacement verification failed: {target}")
        blocker = public_policy.unresolved_v2_invalidation(root)
        if blocker is not None:
            raise ResolutionError(f"release policy rejected the resolved notice: {blocker}")
        verify_resolved(root)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target in reversed(attempted):
            try:
                _restore_snapshot(target, snapshots[target])
            except BaseException as rollback_exc:  # pragma: no cover - catastrophic I/O
                rollback_errors.append(f"{target}: {rollback_exc}")
        for target, original in snapshots.items():
            try:
                observed = target.read_bytes() if target.exists() else None
                if observed != original:
                    rollback_errors.append(f"{target}: byte restoration check failed")
            except OSError as rollback_exc:  # pragma: no cover - catastrophic I/O
                rollback_errors.append(f"{target}: {rollback_exc}")
        if rollback_errors:
            raise ResolutionError(
                "resolution failed and rollback could not restore every canonical byte: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for temporary in staged.values():
            with contextlib.suppress(OSError):
                temporary.unlink()


def resolve(root: Path = ROOT, *, confirmation: str) -> ResolutionPlan:
    """Perform the guarded promotion after an exact explicit confirmation."""
    if confirmation != CONFIRMATION_TOKEN:
        raise ResolutionError(
            "mutating resolution requires --confirm " + CONFIRMATION_TOKEN
        )
    root = _canonical_root(root)
    with _resolution_lock(root):
        plan = prepare_resolution(root)
        _precommit_guard(plan)
        _transaction(plan)
    return plan


def verify_public_receipt(
    root: Path = ROOT,
    *,
    profile: str = "full",
) -> dict[str, Any]:
    """Verify only public receipt, paper, code/config, and result bytes.

    This path never opens the private GPU run tree or raw BACI robustness
    provenance.  ``repository`` permits only absent, explicitly external
    ``data/processed_v2/`` paper sources; ``full`` requires every source.
    """

    if profile not in {"repository", "full"}:
        raise ResolutionError(f"unknown public receipt profile: {profile!r}")
    root = _canonical_root(root)
    try:
        payload = public_policy.verify_v2_resolution_receipt(
            root,
            verify_external_sources=profile == "full",
        )
    except ValueError as exc:
        raise ResolutionError(f"public resolution receipt verification failed: {exc}") from exc
    expected = payload.get(REVIEW_BINDING_FIELD)
    if not isinstance(expected, dict):  # validated above; defensive typing guard
        raise ResolutionError("public resolution receipt lacks a review binding")
    _require_unchanged_human_review_binding(root, expected)
    return payload


def verify_resolved(root: Path = ROOT) -> dict[str, Any]:
    """Private authoritative verification of a resolved notice and all proofs."""
    root = _canonical_root(root)
    verify_public_receipt(root, profile="full")
    _scope_path_checks(root, allow_generated_missing=False)
    payload, _ = _strict_canonical_json_file(
        root / NOTICE_PATH,
        "resolved invalidation notice",
    )
    _validate_resolved_notice(payload)

    replacements = _validate_digest_map(
        payload["replacement_sha256"],
        "replacement_sha256",
        expected_paths=set(public_policy.V2_INVALIDATION_DERIVED_PATHS),
    )
    for relative, expected in replacements.items():
        path = _safe_path(root, relative, require_file=True)
        if _sha256_file(path) != expected:
            raise ResolutionError(f"replacement hash mismatch: {relative}")

    verifier_hashes = _current_verifier_hashes(root)
    recorded_verifiers = _validate_digest_map(
        payload["resolution_verifier_sha256"],
        "resolution_verifier_sha256",
        expected_paths=set(VERIFIER_SOURCE_PATHS),
    )
    if recorded_verifiers != verifier_hashes:
        raise ResolutionError("resolution verifier source hashes no longer match")
    if payload["resolution_gate_sha256"] != verifier_hashes[
        "tools/resolve_v2_invalidation.py"
    ]:
        raise ResolutionError("resolution gate source hash no longer matches")

    _run_authoritative_verifiers(root)
    generated, sources = _render_and_verify_paper(root)
    recorded_sources = _validate_digest_map(
        payload["resolution_source_sha256"],
        "resolution_source_sha256",
    )
    if recorded_sources != sources:
        raise ResolutionError("resolution paper-source hashes no longer match")
    for relative, expected in generated.items():
        if (root / relative).read_bytes() != expected:
            raise ResolutionError(f"canonical paper output is not byte-exact: {relative}")

    blocker = public_policy.unresolved_v2_invalidation(root)
    if blocker is not None:
        raise ResolutionError(f"release policy still reports an unresolved hold: {blocker}")
    expected_review = payload.get(REVIEW_BINDING_FIELD)
    if not isinstance(expected_review, dict):  # validated above
        raise ResolutionError("resolved invalidation notice lacks a review binding")
    # Catch review-artifact drift that occurs while the slower authoritative
    # verifiers and paper renderer are running.
    _require_unchanged_human_review_binding(root, expected_review)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and stage temporary current-schema outputs without canonical writes",
    )
    mode.add_argument(
        "--preview-dir",
        type=Path,
        metavar="DIR",
        help=(
            "export review-only current-schema JSON/TeX below DIR without canonical writes; "
            "permits unset (not mismatched) final key/value contract constants"
        ),
    )
    mode.add_argument(
        "--verify-resolved",
        action="store_true",
        help=(
            "private authoritative verification, including private raw/formal provenance, "
            "of an already resolved marker"
        ),
    )
    mode.add_argument(
        "--verify-public-receipt",
        action="store_true",
        help="read-only public receipt verification without private raw/formal provenance",
    )
    mode.add_argument(
        "--freshen",
        action="store_true",
        help=(
            "atomically reopen a recognized prior RESOLVED receipt as the original ACTIVE hold; "
            f"requires --confirm {FRESHEN_CONFIRMATION_TOKEN}"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("repository", "full"),
        default="full",
        help="public-receipt profile; repository alone may omit external processed-v2 sources",
    )
    parser.add_argument(
        "--confirm",
        metavar="TOKEN",
        help="exact freshen or resolution token required for a mutating mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_resolved:
            payload = verify_resolved(args.root)
            print(
                "verified private authoritative v2 invalidation resolution: "
                f"{len(payload['replacement_sha256'])} exact replacement proofs"
            )
        elif args.verify_public_receipt:
            payload = verify_public_receipt(args.root, profile=args.profile)
            print(
                f"verified public v2 resolution receipt ({args.profile} profile): "
                f"{len(payload['replacement_sha256'])} exact replacement proofs"
            )
        elif args.preview_dir is not None:
            preview = write_paper_preview(
                args.root,
                preview_root=args.preview_dir,
            )
            print(
                "exported verified non-canonical current-schema preview: "
                f"{preview.preview_root}; "
                f"observed_candidate_count={preview.number_key_count}; "
                f"observed_candidate_keys_sha256={preview.number_keys_sha256}; "
                f"observed_candidate_values_sha256={preview.number_values_sha256}"
            )
        elif args.dry_run:
            plan = prepare_resolution(args.root)
            print(
                "v2 invalidation dry-run passed: "
                f"{len(plan.replacement_sha256)} replacements; no canonical files written"
            )
        elif args.freshen:
            payload = freshen_resolution(
                args.root,
                confirmation=args.confirm or "",
            )
            print(
                "v2 invalidation receipt freshened to ACTIVE hold: "
                f"{len(payload['scope'])} original scope entries; run --dry-run next"
            )
        else:
            plan = resolve(args.root, confirmation=args.confirm or "")
            print(
                "v2 invalidation resolved: "
                f"{len(plan.replacement_sha256)} replacements; release-policy hold cleared"
            )
    except (ResolutionError, ValueError, OSError) as exc:
        print(f"V2 INVALIDATION RESOLUTION REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
