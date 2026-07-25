#!/usr/bin/env python3
"""Build and verify the sanitized benchmark-scale and compute profile.

The public JSON intentionally contains only aggregate counts and hashes of
permission-gated receipts.  Private build inputs are supplied explicitly on
the command line; their paths and contents are never copied into the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
CHAIN_MACROS = {
    "sheep": "Sheep",
    "cotton": "Cotton",
    "aluminium": "Aluminium",
    "nickel": "Nickel",
    "cocoa": "Cocoa",
    "oilseed-soy": "OilseedSoy",
}
SCHEMA = "upgrade-bench-v2/benchmark-profile/2"
BENCHMARK_VERSION = "2.1-dev"
STATUS = "complete_verified_sanitized"
DEFAULT_PROFILE = ROOT / "results_v2" / "metrics" / "v2_benchmark_profile.json"
DEFAULT_TEX = ROOT / "paper" / "generated" / "v2_benchmark_profile.tex"
DEFAULT_SUMMARY = ROOT / "data" / "processed_v2" / "dataset_summary.json"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID_RE = re.compile(r"ultra-4g-zero-shot-fixed-\d{8}-r[1-9]\d*\Z")
ATTEMPT_ID_RE = re.compile(
    r"ultra-4g-zero-shot-fixed-\d{8}-r[1-9]\d*-attempt[1-9]\d*\Z"
)
UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)\Z"
)
ULTRA_PROTOCOL = "upgrade-bench-v2/ultra-4g-zero-shot/2"
ULTRA_RECEIPT_FILES = frozenset(
    {
        "extraction.json",
        "slot-set.json",
        *(f"slot-terminal-slot{index}.json" for index in range(4)),
    }
)
ULTRA_PRIVATE_FILES = frozenset(
    {
        "frozen_manifest.json",
        "SCORING_STARTED.json",
        "SCORES_COMPLETE.json",
        "LABEL_EVALUATION_STARTED.json",
        "evaluation.json",
        *(f"components/{chain}/component.json" for chain in CHAINS),
        *(f"components/{chain}/scores_A.csv" for chain in CHAINS),
        *(f"components/{chain}/scores_B.csv" for chain in CHAINS),
        "components/sheep/scores_A_repeat.csv",
        "components/sheep/scores_B_repeat.csv",
        *(f"metrics/metrics_{chain}.json" for chain in CHAINS),
    }
)
ULTRA_PRIVATE_DIRS = frozenset(
    {"components", "metrics", *(f"components/{chain}" for chain in CHAINS)}
)
ULTRA_EXTRACTION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "attempt_id",
        "trust_manifest_sha256",
        "source_root",
        "destination_root",
        "file_count",
        "directory_count",
        "inventory_sha256",
        "files",
        "manifest_sha256",
        "scoring_started_sha256",
        "score_seal_sha256",
        "evaluation_start_sha256",
        "evaluation_sha256",
        "slot_set_sha256",
        "created_at_utc",
    }
)
ULTRA_SLOT_SET_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "attempt_id",
        "slot_plan",
        "terminals",
        "manifest_sha256",
        "scoring_started_sha256",
        "trust_manifest_sha256",
        "created_at_utc",
    }
)
ULTRA_SLOT_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "slot_id",
        "host",
        "physical_device",
        "ordered_chains",
        "components",
        "manifest_sha256",
        "scoring_started_sha256",
        "trust_manifest_sha256",
        "completed_at_utc",
    }
)

# Receipt roles are fixed, while their values are generated from each formal
# run.  This avoids carrying the previous cohort's graph sizes, task counts,
# timings, or hashes into a replacement benchmark identity.
FORMAL_EVIDENCE_ROLES = frozenset(
    {
        "early_graph_freeze_sha256",
        "main_worker_claim_set_sha256",
        "gpu_inventory_sha256",
        "ultra_score_start_sha256",
        "ultra_score_seal_sha256",
        "ultra_component_set_sha256",
        "ultra_receipt_set_sha256",
    }
)


class ProfileError(ValueError):
    """Raised when a profile or one of its evidence inputs is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read valid JSON from {path}") from exc


def _require_keys(value: Mapping[str, Any], expected: Iterable[str], where: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ProfileError(
            f"{where} keys differ: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def _require_int(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileError(f"{where} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, where: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ProfileError(f"{where} must be finite and >= {minimum}")
    return result


def _require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ProfileError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _canonical_digest(items: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(sorted(items.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_set_digest(paths: Iterable[Path], base: Path) -> str:
    entries = {path.relative_to(base).as_posix(): _sha256(path) for path in sorted(paths)}
    if not entries:
        raise ProfileError(f"no evidence files found below {base}")
    return _canonical_digest(entries)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_strict_json(path: Path, where: str, *, canonical: bool = False) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    if path.is_symlink():
        raise ProfileError(f"{where} must not be a symbolic link")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProfileError(f"cannot read strict JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"{where} must be a JSON object")
    if canonical and raw != _canonical_json_bytes(value):
        raise ProfileError(f"{where} is not canonical compact JSON")
    return value


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{where} must be a non-empty string")
    return value


def _require_utc(value: Any, where: str) -> str:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise ProfileError(f"{where} must be a whole-second UTC timestamp")
    _iso_datetime(value, where)
    return value


def _require_path_suffix(value: Any, expected: str, where: str) -> str:
    path_text = _require_string(value, where).replace("\\", "/").rstrip("/")
    if path_text != expected and not path_text.endswith("/" + expected):
        raise ProfileError(f"{where} does not name {expected}")
    return path_text


def _private_inventory_digest(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(f"{file_hash}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _inventory_tree(root: Path) -> tuple[dict[str, str], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise ProfileError("ULTRA formal directory must be a non-symlink directory")
    files: dict[str, str] = {}
    directories: set[str] = set()
    try:
        for base, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            base_path = Path(base)
            if base_path.is_symlink():
                raise ProfileError("ULTRA formal tree contains a symbolic-link directory")
            for name in dirnames:
                path = base_path / name
                if path.is_symlink():
                    raise ProfileError("ULTRA formal tree contains a symbolic-link directory")
                directories.add(path.relative_to(root).as_posix())
            for name in filenames:
                path = base_path / name
                if path.is_symlink() or not path.is_file():
                    raise ProfileError("ULTRA formal tree contains a non-regular file")
                files[path.relative_to(root).as_posix()] = _sha256(path)
    except OSError as exc:
        raise ProfileError(f"cannot inventory ULTRA formal tree {root}") from exc
    return dict(sorted(files.items())), directories


def _require_receipt_directory(receipts_dir: Path) -> dict[str, Path]:
    if receipts_dir.is_symlink() or not receipts_dir.is_dir():
        raise ProfileError("ULTRA receipt directory must be a non-symlink directory")
    try:
        entries = list(receipts_dir.iterdir())
    except OSError as exc:
        raise ProfileError(f"cannot inspect ULTRA receipt directory {receipts_dir}") from exc
    observed = {path.name for path in entries}
    if observed != ULTRA_RECEIPT_FILES or len(entries) != len(ULTRA_RECEIPT_FILES):
        raise ProfileError(
            "ULTRA receipt directory must contain exactly extraction.json, slot-set.json, "
            "and four slot-terminal files"
        )
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ProfileError("ULTRA receipt directory entries must be regular non-symlink files")
    return {path.name: path for path in entries}


def _validate_ultra_component(
    component: Mapping[str, Any],
    *,
    chain: str,
    run_id: str,
    manifest_sha256: str,
    actual_files: Mapping[str, str],
) -> None:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "run_id",
        "chain",
        "created_at_utc",
        "manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "graph_sha256",
        "candidate_precommit",
        "main_target_labels_accessed",
        "main_label_derived_columns_accessed",
        "training_or_fine_tuning_performed",
        "selection_performed",
        "combined_unique_lane_rows",
        "scores",
        "repeat_scores",
        "repeatability",
        "native_backend",
        "peak_memory_allocated_bytes",
        "elapsed_seconds",
        "scoring_seconds",
        "source_sha256",
    }
    _require_keys(component, expected_keys, f"ULTRA {chain} component")
    if (
        component["schema_version"] != "upgrade-bench-v2/ultra-formal-score-component/2"
        or component["protocol"] != ULTRA_PROTOCOL
        or component["status"] != "complete_label_blind_scores"
        or component["run_id"] != run_id
        or component["chain"] != chain
        or component["manifest_sha256"] != manifest_sha256
    ):
        raise ProfileError(f"ULTRA {chain} component identity is inconsistent")
    _require_utc(component["created_at_utc"], f"ULTRA {chain} component timestamp")
    for key in ("config_sha256", "checkpoint_sha256", "graph_sha256"):
        _require_sha(component[key], f"ULTRA {chain} component {key}")
    for key in (
        "main_target_labels_accessed",
        "main_label_derived_columns_accessed",
        "training_or_fine_tuning_performed",
        "selection_performed",
    ):
        if component[key] is not False:
            raise ProfileError(f"ULTRA {chain} component violates the label-blind policy")
    _require_int(component["combined_unique_lane_rows"], f"ULTRA {chain} combined rows", 1)
    _require_int(
        component["peak_memory_allocated_bytes"], f"ULTRA {chain} peak memory", 1
    )
    _require_number(component["elapsed_seconds"], f"ULTRA {chain} elapsed seconds", 0.0)

    scores = component["scores"]
    if not isinstance(scores, Mapping) or set(scores) != {"A", "B"}:
        raise ProfileError(f"ULTRA {chain} component must bind exact A/B scores")
    for source in ("A", "B"):
        ref = scores[source]
        if not isinstance(ref, Mapping):
            raise ProfileError(f"ULTRA {chain}/{source} score reference must be an object")
        _require_keys(
            ref,
            {"path", "sha256", "rows", "identity_sha256", "score_vector_sha256", "column"},
            f"ULTRA {chain}/{source} score reference",
        )
        relative = f"components/{chain}/scores_{source}.csv"
        _require_path_suffix(ref["path"], relative, f"ULTRA {chain}/{source} score path")
        if _require_sha(ref["sha256"], f"ULTRA {chain}/{source} score hash") != actual_files[relative]:
            raise ProfileError(f"ULTRA {chain}/{source} score hash differs from the extraction")
        _require_int(ref["rows"], f"ULTRA {chain}/{source} rows", 1)
        _require_sha(ref["identity_sha256"], f"ULTRA {chain}/{source} identity hash")
        _require_sha(ref["score_vector_sha256"], f"ULTRA {chain}/{source} vector hash")
        if ref["column"] != "ultra_score":
            raise ProfileError(f"ULTRA {chain}/{source} score column is not ultra_score")

    repeat_scores = component["repeat_scores"]
    repeatability = component["repeatability"]
    if chain == "sheep":
        if not isinstance(repeat_scores, Mapping) or set(repeat_scores) != {"A", "B"}:
            raise ProfileError("ULTRA sheep component lacks exact A/B repeat scores")
        if not isinstance(repeatability, Mapping) or repeatability.get("required") is not True:
            raise ProfileError("ULTRA sheep repeatability receipt is incomplete")
        for source in ("A", "B"):
            ref = repeat_scores[source]
            if not isinstance(ref, Mapping):
                raise ProfileError(f"ULTRA sheep/{source} repeat reference must be an object")
            _require_keys(
                ref,
                {"path", "sha256", "rows", "identity_sha256", "score_vector_sha256", "column"},
                f"ULTRA sheep/{source} repeat reference",
            )
            relative = f"components/sheep/scores_{source}_repeat.csv"
            _require_path_suffix(ref["path"], relative, f"ULTRA sheep/{source} repeat path")
            if _require_sha(ref["sha256"], f"ULTRA sheep/{source} repeat hash") != actual_files[relative]:
                raise ProfileError(f"ULTRA sheep/{source} repeat hash differs from the extraction")
            _require_int(ref["rows"], f"ULTRA sheep/{source} repeat rows", 1)
            _require_sha(ref["identity_sha256"], f"ULTRA sheep/{source} repeat identity")
            _require_sha(ref["score_vector_sha256"], f"ULTRA sheep/{source} repeat vector")
            if ref["column"] != "ultra_score":
                raise ProfileError(f"ULTRA sheep/{source} repeat column is not ultra_score")
    else:
        if repeat_scores is not None or repeatability != {"required": False, "sentinel_chain": "sheep"}:
            raise ProfileError(f"ULTRA {chain} component has an unexpected repeat receipt")

    backend = component["native_backend"]
    if not isinstance(backend, Mapping):
        raise ProfileError(f"ULTRA {chain} native backend receipt is missing")
    if re.fullmatch(r"cuda:\d+", str(backend.get("device", ""))) is None:
        raise ProfileError(f"ULTRA {chain} logical CUDA device is malformed")
    _require_string(backend.get("device_name"), f"ULTRA {chain} GPU model")
    scoring_seconds = component["scoring_seconds"]
    if not isinstance(scoring_seconds, Mapping):
        raise ProfileError(f"ULTRA {chain} scoring-time receipt is missing")
    _require_keys(scoring_seconds, {"primary_run1", "repeat_run2"}, f"ULTRA {chain} scoring time")
    if _require_number(
        scoring_seconds["primary_run1"], f"ULTRA {chain} primary scoring seconds", 0.0
    ) <= 0:
        raise ProfileError(f"ULTRA {chain} primary scoring time is not positive")
    if chain == "sheep":
        if _require_number(
            scoring_seconds["repeat_run2"], "ULTRA sheep repeat scoring seconds", 0.0
        ) <= 0:
            raise ProfileError("ULTRA sheep repeat scoring time is not positive")
    elif scoring_seconds["repeat_run2"] is not None:
        raise ProfileError(f"ULTRA {chain} has an unexpected repeat scoring time")


def _validate_ultra_receipts(receipts_dir: Path, ultra_dir: Path) -> tuple[int, str]:
    """Validate private orchestration evidence and return only sanitized aggregates."""

    receipt_paths = _require_receipt_directory(receipts_dir)
    receipts = {
        name: _load_strict_json(path, f"ULTRA receipt {name}", canonical=True)
        for name, path in receipt_paths.items()
    }
    slot_set = receipts["slot-set.json"]
    extraction = receipts["extraction.json"]
    _require_keys(slot_set, ULTRA_SLOT_SET_KEYS, "ULTRA slot-set receipt")
    _require_keys(extraction, ULTRA_EXTRACTION_KEYS, "ULTRA extraction receipt")
    if (
        slot_set["schema_version"] != "upgrade-bench-v2/ultra-r4-slot-set/1"
        or slot_set["status"] != "complete"
        or extraction["schema_version"] != "upgrade-bench-v2/ultra-r4-extraction/1"
        or extraction["status"] != "complete"
    ):
        raise ProfileError("ULTRA slot-set or extraction receipt is not complete")
    run_id = _require_string(slot_set["run_id"], "ULTRA run_id")
    attempt_id = _require_string(slot_set["attempt_id"], "ULTRA attempt_id")
    if RUN_ID_RE.fullmatch(run_id) is None or ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise ProfileError("ULTRA run or attempt identity is malformed")
    if not attempt_id.startswith(run_id + "-attempt"):
        raise ProfileError("ULTRA attempt identity does not belong to its run")
    trust_sha256 = _require_sha(slot_set["trust_manifest_sha256"], "ULTRA trust manifest")
    manifest_sha256 = _require_sha(slot_set["manifest_sha256"], "ULTRA frozen manifest")
    scoring_started_sha256 = _require_sha(
        slot_set["scoring_started_sha256"], "ULTRA scoring-start marker"
    )
    _require_utc(slot_set["created_at_utc"], "ULTRA slot-set timestamp")
    if (
        extraction["run_id"] != run_id
        or extraction["attempt_id"] != attempt_id
        or extraction["trust_manifest_sha256"] != trust_sha256
    ):
        raise ProfileError("ULTRA extraction identity differs from the slot-set identity")
    _require_utc(extraction["created_at_utc"], "ULTRA extraction timestamp")
    source_root = _require_string(extraction["source_root"], "ULTRA extraction source root")
    destination_root = _require_string(
        extraction["destination_root"], "ULTRA extraction destination root"
    )
    if source_root == destination_root:
        raise ProfileError("ULTRA extraction source and destination roots must differ")

    slot_plan = slot_set["slot_plan"]
    if not isinstance(slot_plan, list) or len(slot_plan) != 4:
        raise ProfileError("ULTRA slot plan must contain exactly four slots")
    plan_by_slot: dict[str, Mapping[str, Any]] = {}
    covered_chains: list[str] = []
    physical_slots: set[tuple[str, str]] = set()
    for index, plan in enumerate(slot_plan):
        if not isinstance(plan, Mapping):
            raise ProfileError("ULTRA slot plan rows must be objects")
        _require_keys(plan, {"slot_id", "host", "device", "ordered_chains"}, "ULTRA slot plan row")
        slot_id = f"slot{index}"
        if plan["slot_id"] != slot_id:
            raise ProfileError("ULTRA slot plan must be ordered slot0 through slot3")
        host = _require_string(plan["host"], f"ULTRA {slot_id} host")
        device = _require_string(plan["device"], f"ULTRA {slot_id} physical device")
        if re.fullmatch(r"cuda:\d+", device) is None:
            raise ProfileError(f"ULTRA {slot_id} physical device is malformed")
        chains = plan["ordered_chains"]
        if (
            not isinstance(chains, list)
            or not chains
            or any(chain not in CHAINS for chain in chains)
            or len(set(chains)) != len(chains)
        ):
            raise ProfileError(f"ULTRA {slot_id} ordered chain list is invalid")
        plan_by_slot[slot_id] = plan
        covered_chains.extend(chains)
        physical_slots.add((host, device))
    if len(covered_chains) != len(CHAINS) or set(covered_chains) != set(CHAINS):
        raise ProfileError("ULTRA slot plan does not cover each benchmark chain exactly once")
    if len(physical_slots) != 4:
        raise ProfileError("ULTRA slot plan does not bind four unique physical GPUs")

    terminal_refs = slot_set["terminals"]
    if not isinstance(terminal_refs, list) or len(terminal_refs) != 4:
        raise ProfileError("ULTRA slot-set receipt must bind exactly four terminal receipts")
    component_hashes_by_chain: dict[str, str] = {}
    for index, terminal_ref in enumerate(terminal_refs):
        slot_id = f"slot{index}"
        if not isinstance(terminal_ref, Mapping):
            raise ProfileError("ULTRA terminal references must be objects")
        _require_keys(terminal_ref, {"slot_id", "sha256"}, "ULTRA terminal reference")
        if terminal_ref["slot_id"] != slot_id:
            raise ProfileError("ULTRA terminal references must be ordered slot0 through slot3")
        terminal_name = f"slot-terminal-{slot_id}.json"
        if _require_sha(terminal_ref["sha256"], f"ULTRA {slot_id} terminal hash") != _sha256(
            receipt_paths[terminal_name]
        ):
            raise ProfileError(f"ULTRA {slot_id} terminal receipt hash changed")
        terminal = receipts[terminal_name]
        _require_keys(terminal, ULTRA_SLOT_TERMINAL_KEYS, f"ULTRA {slot_id} terminal")
        plan = plan_by_slot[slot_id]
        if (
            terminal["schema_version"] != "upgrade-bench-v2/ultra-r4-slot-terminal/1"
            or terminal["status"] != "success"
            or terminal["slot_id"] != slot_id
            or terminal["host"] != plan["host"]
            or terminal["physical_device"] != plan["device"]
            or terminal["ordered_chains"] != plan["ordered_chains"]
            or terminal["manifest_sha256"] != manifest_sha256
            or terminal["scoring_started_sha256"] != scoring_started_sha256
            or terminal["trust_manifest_sha256"] != trust_sha256
        ):
            raise ProfileError(f"ULTRA {slot_id} terminal differs from its frozen slot plan")
        _require_utc(terminal["completed_at_utc"], f"ULTRA {slot_id} terminal timestamp")
        components = terminal["components"]
        if not isinstance(components, list) or len(components) != len(plan["ordered_chains"]):
            raise ProfileError(f"ULTRA {slot_id} terminal component list is incomplete")
        terminal_chains: list[str] = []
        for component_ref in components:
            if not isinstance(component_ref, Mapping):
                raise ProfileError(f"ULTRA {slot_id} component references must be objects")
            _require_keys(component_ref, {"chain", "component_sha256"}, "ULTRA component reference")
            chain = component_ref["chain"]
            terminal_chains.append(chain)
            component_hashes_by_chain[chain] = _require_sha(
                component_ref["component_sha256"], f"ULTRA {chain} terminal component hash"
            )
        if terminal_chains != plan["ordered_chains"]:
            raise ProfileError(f"ULTRA {slot_id} component order differs from its slot plan")
    if set(component_hashes_by_chain) != set(CHAINS):
        raise ProfileError("ULTRA terminal receipts do not bind all six components")

    if extraction["slot_set_sha256"] != _sha256(receipt_paths["slot-set.json"]):
        raise ProfileError("ULTRA extraction does not bind the exact slot-set receipt")
    actual_files, actual_directories = _inventory_tree(ultra_dir)
    if set(actual_files) != ULTRA_PRIVATE_FILES or actual_directories != ULTRA_PRIVATE_DIRS:
        raise ProfileError("ULTRA formal extraction is not the exact 31-file, 8-directory tree")
    files = extraction["files"]
    if not isinstance(files, Mapping) or set(files) != ULTRA_PRIVATE_FILES:
        raise ProfileError("ULTRA extraction receipt does not enumerate the exact 31 formal files")
    receipt_files = {
        relative: _require_sha(file_hash, f"ULTRA extraction file {relative}")
        for relative, file_hash in files.items()
    }
    if (
        extraction["file_count"] != 31
        or extraction["directory_count"] != 8
        or receipt_files != actual_files
        or extraction["inventory_sha256"] != _private_inventory_digest(receipt_files)
    ):
        raise ProfileError("ULTRA extraction inventory differs from its 31-file formal tree")
    _require_sha(extraction["inventory_sha256"], "ULTRA extraction inventory")
    extraction_bindings = {
        "manifest_sha256": "frozen_manifest.json",
        "scoring_started_sha256": "SCORING_STARTED.json",
        "score_seal_sha256": "SCORES_COMPLETE.json",
        "evaluation_start_sha256": "LABEL_EVALUATION_STARTED.json",
        "evaluation_sha256": "evaluation.json",
    }
    for field, relative in extraction_bindings.items():
        if extraction[field] != actual_files[relative]:
            raise ProfileError(f"ULTRA extraction {field} does not bind the formal artifact")
    if (
        extraction["manifest_sha256"] != manifest_sha256
        or extraction["scoring_started_sha256"] != scoring_started_sha256
    ):
        raise ProfileError("ULTRA extraction differs from the slot-set formal identity")

    manifest = _load_strict_json(ultra_dir / "frozen_manifest.json", "ULTRA frozen manifest")
    if (
        manifest.get("schema_version") != "upgrade-bench-v2/ultra-formal-freeze/2"
        or manifest.get("protocol") != ULTRA_PROTOCOL
        or manifest.get("status") != "frozen_before_target_scoring"
        or manifest.get("run_id") != run_id
        or manifest.get("main_target_labels_accessed") is not False
    ):
        raise ProfileError("ULTRA frozen manifest identity or label-access status is invalid")
    scoring_start = _load_strict_json(
        ultra_dir / "SCORING_STARTED.json", "ULTRA scoring-start marker", canonical=True
    )
    _require_keys(
        scoring_start,
        {"schema_version", "protocol", "run_id", "manifest_sha256", "started_at_utc", "policy"},
        "ULTRA scoring-start marker",
    )
    if (
        scoring_start["schema_version"] != "upgrade-bench-v2/ultra-formal-score-start/2"
        or scoring_start["protocol"] != ULTRA_PROTOCOL
        or scoring_start["run_id"] != run_id
        or scoring_start["manifest_sha256"] != manifest_sha256
    ):
        raise ProfileError("ULTRA scoring-start marker identity is invalid")
    _require_utc(scoring_start["started_at_utc"], "ULTRA scoring-start timestamp")

    components_by_chain: dict[str, Mapping[str, Any]] = {}
    for chain in CHAINS:
        relative = f"components/{chain}/component.json"
        if component_hashes_by_chain[chain] != actual_files[relative]:
            raise ProfileError(f"ULTRA {chain} terminal component hash differs from extraction")
        component = _load_strict_json(ultra_dir / relative, f"ULTRA {chain} component")
        _validate_ultra_component(
            component,
            chain=chain,
            run_id=run_id,
            manifest_sha256=manifest_sha256,
            actual_files=actual_files,
        )
        components_by_chain[chain] = component

    score_seal_sha256 = extraction["score_seal_sha256"]
    score_seal = _load_strict_json(
        ultra_dir / "SCORES_COMPLETE.json", "ULTRA score seal", canonical=True
    )
    _require_keys(
        score_seal,
        {
            "schema_version",
            "protocol",
            "status",
            "run_id",
            "created_at_utc",
            "manifest_sha256",
            "scoring_started_sha256",
            "component_count",
            "components",
            "native_runtime_sha256",
            "repeatability_contract",
            "sentinel_repeat_verified_before_label_unlock",
            "main_target_labels_accessed_before_seal",
            "unlock_policy",
        },
        "ULTRA score seal",
    )
    if (
        score_seal["schema_version"] != "upgrade-bench-v2/ultra-formal-score-seal/3"
        or score_seal["protocol"] != ULTRA_PROTOCOL
        or score_seal["status"] != "all_six_chains_scored_labels_unlocked"
        or score_seal["run_id"] != run_id
        or score_seal["manifest_sha256"] != manifest_sha256
        or score_seal["scoring_started_sha256"] != scoring_started_sha256
        or score_seal["component_count"] != 6
        or score_seal["sentinel_repeat_verified_before_label_unlock"] is not True
        or score_seal["main_target_labels_accessed_before_seal"] is not False
    ):
        raise ProfileError("ULTRA score seal identity or unlock ordering is invalid")
    _require_utc(score_seal["created_at_utc"], "ULTRA score-seal timestamp")
    native_runtime_sha256 = _require_sha(
        score_seal["native_runtime_sha256"], "ULTRA native runtime"
    )
    sealed_components = score_seal["components"]
    if not isinstance(sealed_components, list) or len(sealed_components) != 6:
        raise ProfileError("ULTRA score seal must contain exactly six components")
    if [row.get("chain") for row in sealed_components if isinstance(row, Mapping)] != list(CHAINS):
        raise ProfileError("ULTRA score-seal component order or coverage is invalid")
    for chain, sealed in zip(CHAINS, sealed_components):
        if not isinstance(sealed, Mapping):
            raise ProfileError("ULTRA sealed component references must be objects")
        _require_keys(
            sealed,
            {
                "chain",
                "path",
                "sha256",
                "A_score_sha256",
                "B_score_sha256",
                "A_repeat_score_sha256",
                "B_repeat_score_sha256",
                "repeatability",
                "native_runtime_sha256",
            },
            f"ULTRA {chain} sealed component",
        )
        relative = f"components/{chain}/component.json"
        _require_path_suffix(sealed["path"], relative, f"ULTRA {chain} sealed component path")
        component = components_by_chain[chain]
        if (
            sealed["sha256"] != actual_files[relative]
            or sealed["sha256"] != component_hashes_by_chain[chain]
            or sealed["A_score_sha256"] != component["scores"]["A"]["sha256"]
            or sealed["B_score_sha256"] != component["scores"]["B"]["sha256"]
            or sealed["repeatability"] != component["repeatability"]
            or sealed["native_runtime_sha256"] != native_runtime_sha256
        ):
            raise ProfileError(f"ULTRA {chain} score-seal component binding is invalid")
        if chain == "sheep":
            if (
                sealed["A_repeat_score_sha256"] != component["repeat_scores"]["A"]["sha256"]
                or sealed["B_repeat_score_sha256"] != component["repeat_scores"]["B"]["sha256"]
            ):
                raise ProfileError("ULTRA sheep repeat score-seal binding is invalid")
        elif sealed["A_repeat_score_sha256"] is not None or sealed["B_repeat_score_sha256"] is not None:
            raise ProfileError(f"ULTRA {chain} has unexpected sealed repeat scores")

    evaluation_start = _load_strict_json(
        ultra_dir / "LABEL_EVALUATION_STARTED.json", "ULTRA evaluation-start marker"
    )
    if (
        evaluation_start.get("schema_version")
        != "upgrade-bench-v2/ultra-formal-evaluation-start/3"
        or evaluation_start.get("protocol") != ULTRA_PROTOCOL
        or evaluation_start.get("run_id") != run_id
        or evaluation_start.get("score_seal_sha256") != score_seal_sha256
        or evaluation_start.get("scoring_started_sha256") != scoring_started_sha256
    ):
        raise ProfileError("ULTRA evaluation-start marker is not cross-bound to the score seal")
    evaluation = _load_strict_json(ultra_dir / "evaluation.json", "ULTRA evaluation")
    if (
        evaluation.get("schema_version") != "upgrade-bench-v2/ultra-formal-evaluation/3"
        or evaluation.get("protocol") != ULTRA_PROTOCOL
        or evaluation.get("status") != "complete"
        or evaluation.get("run_id") != run_id
        or evaluation.get("manifest_sha256") != manifest_sha256
        or evaluation.get("score_seal_sha256") != score_seal_sha256
        or evaluation.get("scoring_started_sha256") != scoring_started_sha256
    ):
        raise ProfileError("ULTRA evaluation is not cross-bound to the formal score chain")
    for chain in CHAINS:
        metric = _load_strict_json(
            ultra_dir / f"metrics/metrics_{chain}.json", f"ULTRA {chain} metrics"
        )
        if (
            metric.get("schema_version") != "upgrade-bench-v2/ultra-formal-chain-metrics/3"
            or metric.get("protocol") != ULTRA_PROTOCOL
            or metric.get("status") != "complete"
            or metric.get("run_id") != run_id
            or metric.get("chain") != chain
            or metric.get("score_seal_sha256") != score_seal_sha256
            or metric.get("scoring_started_sha256") != scoring_started_sha256
        ):
            raise ProfileError(f"ULTRA {chain} metrics are not cross-bound to the score seal")

    receipt_set_sha256 = _canonical_digest(
        {name: _sha256(path) for name, path in receipt_paths.items()}
    )
    return len(physical_slots), receipt_set_sha256


def _parse_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.lstrip().startswith("#"):
            continue
        if "=" not in raw:
            raise ProfileError(f"malformed key-value line in {path.name}")
        key, value = raw.split("=", 1)
        if key in result:
            raise ProfileError(f"duplicate key {key!r} in {path.name}")
        result[key] = value
    return result


def _iso_datetime(value: str, where: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProfileError(f"invalid timestamp at {where}") from exc
    if parsed.tzinfo is None:
        raise ProfileError(f"timestamp at {where} must include an offset")
    return parsed


def _registry_products(chain_path: Path) -> tuple[int, int, int, int]:
    payload = _load_json(chain_path)
    stages = payload.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ProfileError(f"{chain_path.name}: stages must be a non-empty object")
    hs6: set[str] = set()
    for codes in stages.values():
        if not isinstance(codes, list):
            raise ProfileError(f"{chain_path.name}: each stage must contain a code list")
        for code in codes:
            if not isinstance(code, str) or re.fullmatch(r"[0-9]{6}", code) is None:
                raise ProfileError(f"{chain_path.name}: invalid HS6 code {code!r}")
            hs6.add(code)
    hs4 = {code[:4] for code in hs6}
    hs2 = {code[:2] for code in hs6}
    return len(hs6), len(hs4), len(hs2), len(hs6 | hs4 | hs2)


def _summary_by_chain(path: Path) -> tuple[str, dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ProfileError("dataset summary benchmark_version mismatch")
    rows = payload.get("chains")
    if not isinstance(rows, list):
        raise ProfileError("dataset summary chains must be a list")
    by_chain = {row.get("chain"): row for row in rows if isinstance(row, dict)}
    if set(by_chain) != set(CHAINS) or len(rows) != len(CHAINS):
        raise ProfileError("dataset summary must contain exactly the six benchmark chains")
    return payload["benchmark_version"], by_chain


def _task_counts(row: Mapping[str, Any], chain: str) -> dict[str, int]:
    result = {
        "b1_candidate_entries": _require_int(row.get("track_b_unique_entries"), f"{chain}.B1 entries"),
        "b1_positive_entries": _require_int(row.get("track_b_positive_entries"), f"{chain}.B1 positives"),
        "b2_positive_entry_groups": _require_int(row.get("track_b_positive_entries"), f"{chain}.B2 groups"),
        "b2_candidate_lanes": _require_int(row.get("track_b2_conditional_lanes"), f"{chain}.B2 lanes"),
        "b2_positive_lanes": _require_int(row.get("track_b2_positive_lanes"), f"{chain}.B2 positives"),
    }
    if result["b1_positive_entries"] > result["b1_candidate_entries"]:
        raise ProfileError(f"{chain}: B1 positives exceed candidates")
    if result["b2_positive_entry_groups"] != result["b1_positive_entries"]:
        raise ProfileError(f"{chain}: B2 groups must equal positive B1 entries")
    if result["b2_positive_lanes"] > result["b2_candidate_lanes"]:
        raise ProfileError(f"{chain}: B2 positive lanes exceed candidates")
    return result


def _graph_counts(
    chain: str, chain_path: Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    hs6, hs4, hs2, product_nodes = _registry_products(chain_path)
    try:
        chain_receipt = freeze["chains"][chain]
        coverage = chain_receipt["cohorts"]["A"]["coverage"]
        overlap = chain_receipt["cohorts"]["A"]["early_trade_overlap"]
        graph_receipt = chain_receipt["graph"]
    except (KeyError, TypeError) as exc:
        raise ProfileError(f"formal graph receipt is incomplete for {chain}") from exc
    graph_nodes = _require_int(coverage.get("graph_entities"), f"{chain}.graph nodes", 1)
    serialized = _require_int(coverage.get("graph_forward_triples"), f"{chain}.forward triples", 1)
    relations = _require_int(coverage.get("graph_forward_relations"), f"{chain}.relations", 1)
    trade_edges = _require_int(overlap.get("early_forward_trade_triples"), f"{chain}.trade edges", 1)
    duplicate_parent_rows = hs6 - hs4
    result = {
        "nodes": graph_nodes,
        "countries": graph_nodes - product_nodes,
        "products": product_nodes,
        "hs6_products": hs6,
        "hs4_products": hs4,
        "hs2_products": hs2,
        "forward_relations": relations,
        "early_trade_edges": trade_edges,
        "serialized_forward_triples": serialized,
        "unique_forward_facts": serialized - duplicate_parent_rows,
        "duplicate_forward_rows": duplicate_parent_rows,
        "duplicate_rows_by_relation": {"hs_parent": duplicate_parent_rows},
        "graph_sha256": _require_sha(graph_receipt.get("sha256"), f"{chain}.graph sha256"),
    }
    if result["countries"] <= 0 or result["nodes"] != result["countries"] + product_nodes:
        raise ProfileError(f"{chain}: graph entity counts do not reconcile")
    if result["unique_forward_facts"] < 0:
        raise ProfileError(f"{chain}: duplicate-adjusted fact count is negative")
    _require_sha(result["graph_sha256"], f"{chain}.graph sha256")
    if graph_receipt.get("forward_triples") != serialized:
        raise ProfileError(f"{chain}: graph receipt and coverage triple counts differ")
    return result


def _collect_score_cache_keys(value: Any, target: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "score_cache_key" and isinstance(item, str):
                target.add(item)
            else:
                _collect_score_cache_keys(item, target)
    elif isinstance(value, list):
        for item in value:
            _collect_score_cache_keys(item, target)


def _build_compute(
    claims_dir: Path,
    metrics_dir: Path,
    selections_dir: Path,
    gpu_inventory_path: Path,
    ultra_dir: Path,
    ultra_receipts_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    inventory = _load_json(gpu_inventory_path)
    gpus = inventory.get("gpus")
    if (
        not isinstance(gpus, list)
        or len(gpus) != 4
        or any(not isinstance(gpu, Mapping) for gpu in gpus)
    ):
        raise ProfileError("formal GPU inventory must contain four devices")
    gpu_names = {gpu.get("name") for gpu in gpus}
    memories = {gpu.get("memory_total_mib") for gpu in gpus}
    if (
        len(gpu_names) != 1
        or not all(isinstance(name, str) and name for name in gpu_names)
        or len(memories) != 1
        or not all(isinstance(value, int) and value > 0 for value in memories)
    ):
        raise ProfileError("formal fitted-reference inventory must be homogeneous")
    gpu_model = next(iter(gpu_names))
    gpu_memory_mib = next(iter(memories))
    inventory_slots = {
        (str(gpu.get("host")), int(gpu.get("index"))) for gpu in gpus if isinstance(gpu, dict)
    }

    claim_dirs = sorted(path for path in claims_dir.iterdir() if path.is_dir())
    if len(claim_dirs) != 12:
        raise ProfileError("main fitted-reference receipt set must contain 12 chain-family jobs")
    seconds = {"kge": 0, "nbfnet": 0}
    jobs = {"kge": 0, "nbfnet": 0}
    claim_slots: set[tuple[str, int]] = set()
    claim_pairs: set[tuple[str, str]] = set()
    claim_files: list[Path] = []
    manifest_hashes: set[str] = set()
    for directory in claim_dirs:
        worker_path = directory / "worker.env"
        status_path = directory / "status"
        worker = _parse_key_values(worker_path)
        status = _parse_key_values(status_path)
        family = worker.get("family")
        chain = worker.get("chain")
        if family not in seconds or chain not in CHAINS:
            raise ProfileError(f"unexpected main worker identity in {directory.name}")
        if status.get("exit_code") != "0":
            raise ProfileError(f"main worker did not finish successfully: {directory.name}")
        started = _iso_datetime(worker.get("claimed_at", ""), f"{directory.name}.claimed_at")
        finished = _iso_datetime(status.get("finished_at", ""), f"{directory.name}.finished_at")
        elapsed = int((finished - started).total_seconds())
        if elapsed <= 0:
            raise ProfileError(f"non-positive worker duration in {directory.name}")
        seconds[family] += elapsed
        jobs[family] += 1
        pair = (chain, family)
        if pair in claim_pairs:
            raise ProfileError(f"duplicate main worker chain-family pair: {pair}")
        claim_pairs.add(pair)
        host = str(worker.get("host", "")).split(".", 1)[0]
        claim_slots.add((host, int(worker.get("physical_gpu", "-1"))))
        manifest_hashes.add(_require_sha(worker.get("manifest_sha256"), "main manifest sha256"))
        claim_files.extend((worker_path, status_path))
    if claim_pairs != {(chain, family) for chain in CHAINS for family in seconds}:
        raise ProfileError("main worker receipts do not cover every chain-family pair")
    if claim_slots != inventory_slots or len(manifest_hashes) != 1:
        raise ProfileError("main worker hardware or frozen-manifest identity is inconsistent")

    representations: dict[str, int] = {}
    selection_fits: dict[str, int] = {}
    for family in ("kge", "nbfnet"):
        count = 0
        keys: set[str] = set()
        for chain in CHAINS:
            metrics = _load_json(metrics_dir / f"metrics_{chain}_track-a_{family}.json")
            finished = metrics.get("protocol_timestamps", {}).get("representations_finished")
            if not isinstance(finished, list):
                raise ProfileError(f"missing representation receipt for {chain}/{family}")
            count += len(finished)
        for selection_path in sorted(selections_dir.glob(f"selection_*_{family}.json")):
            _collect_score_cache_keys(_load_json(selection_path), keys)
        representations[family] = count
        selection_fits[family] = len(keys)
    if jobs != {"kge": len(CHAINS), "nbfnet": len(CHAINS)}:
        raise ProfileError("main worker family counts do not cover the six chains")
    if any(value <= 0 for value in representations.values()):
        raise ProfileError("main representation receipt is empty")
    if any(value <= 0 for value in selection_fits.values()):
        raise ProfileError("historical selection receipt is empty")

    ultra_physical_gpu_count, ultra_receipt_set_sha256 = _validate_ultra_receipts(
        ultra_receipts_dir, ultra_dir
    )
    score_start_path = ultra_dir / "SCORING_STARTED.json"
    score_seal_path = ultra_dir / "SCORES_COMPLETE.json"
    score_start = _load_json(score_start_path)
    score_seal = _load_json(score_seal_path)
    start_time = _iso_datetime(score_start.get("started_at_utc", ""), "ULTRA scoring start")
    seal_time = _iso_datetime(score_seal.get("created_at_utc", ""), "ULTRA scoring seal")
    wall_seconds = int((seal_time - start_time).total_seconds())
    component_paths = sorted((ultra_dir / "components").glob("*/component.json"))
    if len(component_paths) != 6:
        raise ProfileError("ULTRA receipt set must contain six chain components")
    device_names: set[str] = set()
    kernel_seconds = 0.0
    sheep_repeat_seconds: float | None = None
    for path in component_paths:
        component = _load_json(path)
        if component.get("training_or_fine_tuning_performed") is not False:
            raise ProfileError("ULTRA benchmark training/fine-tuning flag is not false")
        device_names.add(component["native_backend"]["device_name"])
        kernel_seconds += _require_number(component["scoring_seconds"]["primary_run1"], "ULTRA kernel seconds")
        if component.get("chain") == "sheep":
            sheep_repeat_seconds = _require_number(
                component["scoring_seconds"].get("repeat_run2"), "ULTRA sheep repeat seconds"
            )
    if wall_seconds <= 0 or device_names != {gpu_model}:
        raise ProfileError("ULTRA hardware or wall-clock receipt is invalid")
    if kernel_seconds <= 0 or sheep_repeat_seconds is None or sheep_repeat_seconds <= 0:
        raise ProfileError("ULTRA scoring-time receipt is invalid")

    compute = {
        "fitted_references": {
            "hardware": {
                "gpu_model": gpu_model,
                "gpu_memory_mib": gpu_memory_mib,
                "gpu_memory_gb_reported": gpu_memory_mib // 1000,
                "physical_gpu_count": 4,
            },
            "main_refit_and_evaluation": {
                "duration_semantics": "summed one-GPU worker wall-clock",
                "pykeen_global_graph": {
                    "chain_jobs": jobs["kge"],
                    "representations": representations["kge"],
                    "worker_seconds": seconds["kge"],
                },
                "nbfnet": {
                    "chain_jobs": jobs["nbfnet"],
                    "representations": representations["nbfnet"],
                    "worker_seconds": seconds["nbfnet"],
                },
                "total_worker_seconds": sum(seconds.values()),
                "total_worker_hours": round(sum(seconds.values()) / 3600.0, 2),
            },
            "historical_selection": {
                "pykeen_cache_deduplicated_fits": selection_fits["kge"],
                "nbfnet_cache_deduplicated_fits": selection_fits["nbfnet"],
                "wall_time_retained": False,
            },
        },
        "ultra_zero_shot": {
            "gpu_model": gpu_model,
            "physical_gpu_count": ultra_physical_gpu_count,
            "benchmark_training_or_fine_tuning": False,
            "six_chain_scoring_wall_seconds": wall_seconds,
            "six_chain_scoring_wall_minutes": wall_seconds // 60,
            "primary_kernel_seconds": round(kernel_seconds, 3),
            "sheep_repeat_kernel_seconds": sheep_repeat_seconds,
            "external_pretraining_compute_known": False,
        },
    }
    evidence = {
        "main_worker_claim_set_sha256": _file_set_digest(claim_files, claims_dir),
        "gpu_inventory_sha256": _sha256(gpu_inventory_path),
        "ultra_score_start_sha256": _sha256(score_start_path),
        "ultra_score_seal_sha256": _sha256(score_seal_path),
        "ultra_component_set_sha256": _file_set_digest(component_paths, ultra_dir),
        "ultra_receipt_set_sha256": ultra_receipt_set_sha256,
    }
    return compute, evidence


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    ultra_receipts_dir = getattr(args, "ultra_receipts_dir", None)
    required = {
        "formal graph freeze": args.freeze_manifest,
        "main worker claims": args.claims_dir,
        "GPU inventory": args.gpu_inventory,
        "ULTRA formal result directory": args.ultra_dir,
        "ULTRA orchestration receipt directory": ultra_receipts_dir,
    }
    missing = [label for label, path in required.items() if path is None]
    if missing:
        raise ProfileError("--build requires: " + ", ".join(missing))
    assert ultra_receipts_dir is not None
    benchmark_version, summary = _summary_by_chain(args.dataset_summary)
    freeze = _load_json(args.freeze_manifest)
    if freeze.get("status") != "frozen_before_target_scoring":
        raise ProfileError("formal graph receipt is not a pre-scoring freeze")
    if freeze.get("main_target_labels_accessed") is not False:
        raise ProfileError("formal graph freeze reports target-label access")

    chains: list[dict[str, Any]] = []
    for chain in CHAINS:
        chains.append(
            {
                "chain": chain,
                "graph": _graph_counts(chain, ROOT / "chains" / f"{chain}.json", freeze),
                "samples": _task_counts(summary[chain], chain),
            }
        )
    compute, compute_evidence = _build_compute(
        args.claims_dir,
        args.metrics_dir,
        args.selections_dir,
        args.gpu_inventory,
        args.ultra_dir,
        ultra_receipts_dir,
    )
    totals = {
        key: sum(chain["samples"][key] for chain in chains)
        for key in (
            "b1_candidate_entries",
            "b1_positive_entries",
            "b2_positive_entry_groups",
            "b2_candidate_lanes",
            "b2_positive_lanes",
        )
    }
    totals["b2_min_entry_groups_per_chain"] = min(
        chain["samples"]["b2_positive_entry_groups"] for chain in chains
    )
    totals["b2_max_entry_groups_per_chain"] = max(
        chain["samples"]["b2_positive_entry_groups"] for chain in chains
    )
    public_sources = {
        "data/processed_v2/dataset_summary.json": _sha256(args.dataset_summary),
        "tools/generate_v2_benchmark_profile.py": _sha256(Path(__file__).resolve()),
    }
    public_sources.update(
        {
            f"chains/{chain}.json": _sha256(ROOT / "chains" / f"{chain}.json")
            for chain in CHAINS
        }
    )
    formal_evidence = {
        "early_graph_freeze_sha256": _sha256(args.freeze_manifest),
        **compute_evidence,
    }
    return {
        "schema_version": SCHEMA,
        "benchmark_version": benchmark_version,
        "status": STATUS,
        "graph_contract": {
            "fold": "main",
            "early_window": "2008-2012",
            "aggregation": "calendar_mean",
            "node_definition": "ISO3 countries plus registered HS6, HS4, and HS2 products",
            "stage_definition": "processing stages are relation labels, not nodes",
            "edge_definition": "serialized directed forward triples before model-specific inverse expansion",
            "unique_fact_definition": "exact duplicate forward triples removed",
        },
        "evaluation_contract": {
            "b1_unit": "exporter-stage entry",
            "b2_macro_unit": "realized B1 exporter-stage entry group",
            "b2_lane_rows_independent": False,
        },
        "chains": chains,
        "totals": totals,
        "compute": compute,
        "provenance": {
            "public_sources": dict(sorted(public_sources.items())),
            "formal_evidence_sha256": formal_evidence,
            "privacy_policy": "aggregate values and receipt digests only; no private paths or raw logs",
        },
    }


def _validate_compute(compute: Mapping[str, Any]) -> None:
    _require_keys(compute, {"fitted_references", "ultra_zero_shot"}, "compute")
    fitted = compute["fitted_references"]
    _require_keys(fitted, {"hardware", "main_refit_and_evaluation", "historical_selection"}, "compute.fitted")
    hardware = fitted["hardware"]
    _require_keys(hardware, {"gpu_model", "gpu_memory_mib", "gpu_memory_gb_reported", "physical_gpu_count"}, "compute.hardware")
    if not isinstance(hardware["gpu_model"], str) or not hardware["gpu_model"]:
        raise ProfileError("fitted-reference GPU model is missing")
    memory_mib = _require_int(hardware["gpu_memory_mib"], "compute.hardware memory", 1)
    if (
        hardware["gpu_memory_gb_reported"] != memory_mib // 1000
        or hardware["physical_gpu_count"] != 4
    ):
        raise ProfileError("fitted-reference hardware accounting is inconsistent")
    main = fitted["main_refit_and_evaluation"]
    _require_keys(main, {"duration_semantics", "pykeen_global_graph", "nbfnet", "total_worker_seconds", "total_worker_hours"}, "compute.main")
    if main["duration_semantics"] != "summed one-GPU worker wall-clock":
        raise ProfileError("main duration semantics mismatch")
    worker_total = 0
    for family in ("pykeen_global_graph", "nbfnet"):
        payload = main[family]
        _require_keys(payload, {"chain_jobs", "representations", "worker_seconds"}, f"compute.main.{family}")
        if _require_int(payload["chain_jobs"], f"{family}.chain_jobs") != len(CHAINS):
            raise ProfileError(f"{family} does not contain six chain jobs")
        _require_int(payload["representations"], f"{family}.representations", 1)
        worker_total += _require_int(payload["worker_seconds"], f"{family}.worker_seconds", 1)
    if (
        main["total_worker_seconds"] != worker_total
        or not math.isclose(
            _require_number(main["total_worker_hours"], "compute.main.total_worker_hours"),
            round(worker_total / 3600.0, 2),
            abs_tol=1e-12,
        )
    ):
        raise ProfileError("main total compute accounting is inconsistent")
    history = fitted["historical_selection"]
    _require_keys(
        history,
        {
            "pykeen_cache_deduplicated_fits",
            "nbfnet_cache_deduplicated_fits",
            "wall_time_retained",
        },
        "compute.history",
    )
    _require_int(
        history["pykeen_cache_deduplicated_fits"], "compute.history.pykeen fits", 1
    )
    _require_int(history["nbfnet_cache_deduplicated_fits"], "compute.history.nbfnet fits", 1)
    if history["wall_time_retained"] is not False:
        raise ProfileError("historical selection accounting is invalid")
    ultra = compute["ultra_zero_shot"]
    _require_keys(
        ultra,
        {
            "gpu_model",
            "physical_gpu_count",
            "benchmark_training_or_fine_tuning",
            "six_chain_scoring_wall_seconds",
            "six_chain_scoring_wall_minutes",
            "primary_kernel_seconds",
            "sheep_repeat_kernel_seconds",
            "external_pretraining_compute_known",
        },
        "compute.ultra",
    )
    wall_seconds = _require_int(
        ultra["six_chain_scoring_wall_seconds"], "compute.ultra.wall_seconds", 1
    )
    if (
        ultra["gpu_model"] != hardware["gpu_model"]
        or _require_int(ultra["physical_gpu_count"], "compute.ultra.gpus") != 4
        or ultra["benchmark_training_or_fine_tuning"] is not False
        or ultra["six_chain_scoring_wall_minutes"] != wall_seconds // 60
        or _require_number(ultra["primary_kernel_seconds"], "compute.ultra.kernel", 0.0) <= 0
        or _require_number(ultra["sheep_repeat_kernel_seconds"], "compute.ultra.repeat", 0.0) <= 0
        or ultra["external_pretraining_compute_known"] is not False
    ):
        raise ProfileError("ULTRA compute accounting differs from the reviewed formal receipts")


def validate_profile(profile: Mapping[str, Any], *, mode: str = "full") -> None:
    _require_keys(
        profile,
        {"schema_version", "benchmark_version", "status", "graph_contract", "evaluation_contract", "chains", "totals", "compute", "provenance"},
        "profile",
    )
    if (
        profile["schema_version"] != SCHEMA
        or profile["benchmark_version"] != BENCHMARK_VERSION
        or profile["status"] != STATUS
    ):
        raise ProfileError("profile identity/status mismatch")
    graph_contract = profile["graph_contract"]
    _require_keys(graph_contract, {"fold", "early_window", "aggregation", "node_definition", "stage_definition", "edge_definition", "unique_fact_definition"}, "graph_contract")
    if (graph_contract["fold"], graph_contract["early_window"], graph_contract["aggregation"]) != ("main", "2008-2012", "calendar_mean"):
        raise ProfileError("graph temporal contract mismatch")
    evaluation = profile["evaluation_contract"]
    _require_keys(evaluation, {"b1_unit", "b2_macro_unit", "b2_lane_rows_independent"}, "evaluation_contract")
    if evaluation["b2_lane_rows_independent"] is not False:
        raise ProfileError("B2 lane rows must not be marked independent")

    rows = profile["chains"]
    if not isinstance(rows, list) or [row.get("chain") for row in rows] != list(CHAINS):
        raise ProfileError("profile chain order or membership mismatch")
    summary_by_chain: dict[str, Any] | None = None
    if mode not in {"repository", "full"}:
        raise ProfileError("verification mode must be repository or full")
    if DEFAULT_SUMMARY.exists():
        _, summary_by_chain = _summary_by_chain(DEFAULT_SUMMARY)
    elif mode == "full":
        raise ProfileError("full verification requires data/processed_v2/dataset_summary.json")

    total_fields = (
        "b1_candidate_entries", "b1_positive_entries", "b2_positive_entry_groups",
        "b2_candidate_lanes", "b2_positive_lanes",
    )
    calculated = {key: 0 for key in total_fields}
    group_counts: list[int] = []
    for row in rows:
        _require_keys(row, {"chain", "graph", "samples"}, f"chains.{row.get('chain')}")
        chain = row["chain"]
        graph = row["graph"]
        _require_keys(
            graph,
            {"nodes", "countries", "products", "hs6_products", "hs4_products", "hs2_products", "forward_relations", "early_trade_edges", "serialized_forward_triples", "unique_forward_facts", "duplicate_forward_rows", "duplicate_rows_by_relation", "graph_sha256"},
            f"{chain}.graph",
        )
        for key in set(graph) - {"duplicate_rows_by_relation", "graph_sha256"}:
            _require_int(graph[key], f"{chain}.graph.{key}")
        _require_sha(graph["graph_sha256"], f"{chain}.graph_sha256")
        hs6, hs4, hs2, products = _registry_products(ROOT / "chains" / f"{chain}.json")
        if (
            (
                graph["hs6_products"],
                graph["hs4_products"],
                graph["hs2_products"],
                graph["products"],
            )
            != (hs6, hs4, hs2, products)
        ):
            raise ProfileError(f"{chain}: graph/profile registry reconciliation failed")
        duplicates = hs6 - hs4
        if (
            graph["countries"] <= 0
            or graph["forward_relations"] <= 0
            or graph["early_trade_edges"] <= 0
            or graph["serialized_forward_triples"] <= 0
            or graph["nodes"] != graph["countries"] + graph["products"]
        ):
            raise ProfileError(f"{chain}: nodes do not reconcile")
        if graph["serialized_forward_triples"] - graph["unique_forward_facts"] != duplicates:
            raise ProfileError(f"{chain}: duplicate triple accounting does not reconcile")
        if graph["duplicate_forward_rows"] != duplicates or graph["duplicate_rows_by_relation"] != {"hs_parent": duplicates}:
            raise ProfileError(f"{chain}: duplicate relation accounting mismatch")
        samples = row["samples"]
        _require_keys(samples, total_fields, f"{chain}.samples")
        for key in total_fields:
            _require_int(samples[key], f"{chain}.samples.{key}")
        if samples["b2_positive_entry_groups"] != samples["b1_positive_entries"]:
            raise ProfileError(f"{chain}: B2 groups must equal positive B1 entries")
        if summary_by_chain is not None and samples != _task_counts(summary_by_chain[chain], chain):
            raise ProfileError(f"{chain}: profile and dataset summary differ")
        for key in total_fields:
            calculated[key] += samples[key]
        group_counts.append(samples["b2_positive_entry_groups"])

    totals = profile["totals"]
    _require_keys(totals, set(total_fields) | {"b2_min_entry_groups_per_chain", "b2_max_entry_groups_per_chain"}, "totals")
    if {key: totals[key] for key in total_fields} != calculated:
        raise ProfileError("profile task totals do not equal the per-chain sums")
    if totals["b2_min_entry_groups_per_chain"] != min(group_counts) or totals["b2_max_entry_groups_per_chain"] != max(group_counts):
        raise ProfileError("B2 per-chain group range does not reconcile")
    _validate_compute(profile["compute"])

    provenance = profile["provenance"]
    _require_keys(provenance, {"public_sources", "formal_evidence_sha256", "privacy_policy"}, "provenance")
    expected_public_paths = {"data/processed_v2/dataset_summary.json", "tools/generate_v2_benchmark_profile.py"} | {f"chains/{chain}.json" for chain in CHAINS}
    if set(provenance["public_sources"]) != expected_public_paths:
        raise ProfileError("public source inventory mismatch")
    for path_text, digest in provenance["public_sources"].items():
        _require_sha(digest, f"public source {path_text}")
        path = ROOT / path_text
        if path.exists():
            if _sha256(path) != digest:
                raise ProfileError(f"public source hash mismatch: {path_text}")
        elif mode == "full" or path_text != "data/processed_v2/dataset_summary.json":
            raise ProfileError(f"required public source is missing: {path_text}")
    formal = provenance["formal_evidence_sha256"]
    _require_keys(formal, FORMAL_EVIDENCE_ROLES, "formal evidence")
    for role, digest in formal.items():
        _require_sha(digest, f"formal evidence {role}")
    if provenance["privacy_policy"] != "aggregate values and receipt digests only; no private paths or raw logs":
        raise ProfileError("profile privacy-policy declaration mismatch")


def _tex_integer(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def render_tex(profile: Mapping[str, Any], profile_sha256: str) -> str:
    lines = [
        "% AUTO-GENERATED by tools/generate_v2_benchmark_profile.py; do not edit.",
        f"% profile sha256: {profile_sha256}",
    ]
    graph_fields = {
        "GraphNodes": "nodes",
        "GraphCountries": "countries",
        "GraphProducts": "products",
        "GraphForwardRelations": "forward_relations",
        "GraphTradeEdges": "early_trade_edges",
        "GraphForwardTriples": "serialized_forward_triples",
        "GraphUniqueFacts": "unique_forward_facts",
    }
    sample_fields = {
        "BOneEntries": "b1_candidate_entries",
        "BOnePositives": "b1_positive_entries",
        "BTwoGroups": "b2_positive_entry_groups",
        "BTwoLanes": "b2_candidate_lanes",
        "BTwoPositives": "b2_positive_lanes",
    }
    for row in profile["chains"]:
        prefix = f"VTwoProfile{CHAIN_MACROS[row['chain']]}"
        for suffix, key in graph_fields.items():
            lines.append(f"\\newcommand{{\\{prefix}{suffix}}}{{{_tex_integer(row['graph'][key])}}}")
        for suffix, key in sample_fields.items():
            lines.append(f"\\newcommand{{\\{prefix}{suffix}}}{{{_tex_integer(row['samples'][key])}}}")
    total_macros = {
        "VTwoProfileBOneEntries": "b1_candidate_entries",
        "VTwoProfileBOnePositives": "b1_positive_entries",
        "VTwoProfileBTwoGroups": "b2_positive_entry_groups",
        "VTwoProfileBTwoLanes": "b2_candidate_lanes",
        "VTwoProfileBTwoPositives": "b2_positive_lanes",
        "VTwoProfileBTwoMinGroups": "b2_min_entry_groups_per_chain",
        "VTwoProfileBTwoMaxGroups": "b2_max_entry_groups_per_chain",
    }
    for macro, key in total_macros.items():
        lines.append(f"\\newcommand{{\\{macro}}}{{{_tex_integer(profile['totals'][key])}}}")
    fitted = profile["compute"]["fitted_references"]
    main = fitted["main_refit_and_evaluation"]
    history = fitted["historical_selection"]
    ultra = profile["compute"]["ultra_zero_shot"]
    compute_macros: list[tuple[str, str]] = [
        ("VTwoProfileGPUName", fitted["hardware"]["gpu_model"]),
        ("VTwoProfileGPUMemoryGB", str(fitted["hardware"]["gpu_memory_gb_reported"])),
        ("VTwoProfileFittedGPUCount", str(fitted["hardware"]["physical_gpu_count"])),
        ("VTwoProfileMainPyKEENRepresentations", str(main["pykeen_global_graph"]["representations"])),
        ("VTwoProfileMainNBFNetRepresentations", str(main["nbfnet"]["representations"])),
        ("VTwoProfileMainWorkerHours", f"{main['total_worker_hours']:.2f}"),
        ("VTwoProfileHistoricalPyKEENFits", str(history["pykeen_cache_deduplicated_fits"])),
        ("VTwoProfileHistoricalNBFNetFits", str(history["nbfnet_cache_deduplicated_fits"])),
        ("VTwoProfileULTRAGPUCount", str(ultra["physical_gpu_count"])),
        ("VTwoProfileULTRAWallMinutes", str(ultra["six_chain_scoring_wall_minutes"])),
    ]
    for macro, value in compute_macros:
        lines.append(f"\\newcommand{{\\{macro}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def _write_outputs(profile: Mapping[str, Any], profile_path: Path, tex_path: Path) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    profile_path.write_text(encoded, encoding="utf-8", newline="\n")
    tex_path.write_text(render_tex(profile, _sha256(profile_path)), encoding="utf-8", newline="\n")


def verify_outputs(profile_path: Path = DEFAULT_PROFILE, tex_path: Path = DEFAULT_TEX, *, mode: str = "full") -> None:
    profile = _load_json(profile_path)
    if not isinstance(profile, dict):
        raise ProfileError("benchmark profile root must be an object")
    validate_profile(profile, mode=mode)
    expected_tex = render_tex(profile, _sha256(profile_path))
    try:
        actual_tex = tex_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileError(f"cannot read generated TeX from {tex_path}") from exc
    if actual_tex != expected_tex:
        raise ProfileError("generated benchmark-profile TeX is stale")


def _path(value: str) -> Path:
    return Path(value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true", help="build JSON and TeX from formal receipts")
    action.add_argument("--verify", action="store_true", help="verify committed JSON and TeX")
    parser.add_argument("--profile", choices=("repository", "full"), default="full", help="verification profile")
    parser.add_argument("--output", type=_path, default=DEFAULT_PROFILE)
    parser.add_argument("--tex-output", type=_path, default=DEFAULT_TEX)
    parser.add_argument("--dataset-summary", type=_path, default=DEFAULT_SUMMARY)
    parser.add_argument("--freeze-manifest", type=_path)
    parser.add_argument("--claims-dir", type=_path)
    parser.add_argument("--metrics-dir", type=_path, default=ROOT / "results_v2" / "gpu_rolling" / "metrics")
    parser.add_argument("--selections-dir", type=_path, default=ROOT / "results_v2" / "gpu_rolling" / "selections")
    parser.add_argument("--gpu-inventory", type=_path)
    parser.add_argument("--ultra-dir", type=_path)
    parser.add_argument("--ultra-receipts-dir", type=_path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.build:
            profile = build_profile(args)
            validate_profile(profile, mode="full")
            _write_outputs(profile, args.output, args.tex_output)
            verify_outputs(args.output, args.tex_output, mode="full")
            print(f"wrote verified benchmark profile: {args.output}")
        else:
            verify_outputs(args.output, args.tex_output, mode=args.profile)
            print(f"verified benchmark profile ({args.profile}): {args.output}")
    except ProfileError as exc:
        print(f"benchmark profile verification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
