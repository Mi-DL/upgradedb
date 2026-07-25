"""Promote the complete formal matched-LOCO run to public JSON and CSV.

Promotion is fail closed.  A saved verification receipt is retained only as
an untrusted operational log: it is never a trust anchor.  Before any public
artifact is written, this module directly invokes the canonical formal
controller's verifier against the complete run tree, checks the current config
bytes, and rereads the 60 verified component files.  Every published aggregate
is then recomputed from 60 sanitized per-component metric records.

``--verify-output`` is intentionally public-only.  It needs neither the formal
controller nor the component tree: it validates the 60 sanitized records,
recomputes every aggregate, and byte-checks canonical JSON plus derived CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
PRIVATE_ROOT = ROOT / "results_v2" / "loco_formal"
PRIVATE_SUMMARY = PRIVATE_ROOT / "summary.json"
PRIVATE_RECEIPT = PRIVATE_ROOT / "verification_receipt.json"
PRIVATE_MANIFEST = PRIVATE_ROOT / "frozen_manifest.json"
CANONICAL_CONFIG = ROOT / "configs" / "v2_loco_formal.json"
CANONICAL_CONTROLLER = TOOLS / "v2_loco_formal.py"
DEFAULT_JSON_OUT = ROOT / "results_v2" / "metrics" / "v2_loco_transfer_summary.json"
DEFAULT_CSV_OUT = ROOT / "results_v2" / "metrics" / "v2_loco_transfer_summary.csv"

FORMAL_SUMMARY_SCHEMA = "upgrade-bench-v2/loco-formal-summary/1"
PUBLIC_SUMMARY_SCHEMA = "upgrade-bench-v2/loco-transfer-public-summary/1"
PROTOCOL = "upgrade-bench-v2/loco-tier-matched/1"
RUN_ID = "loco-tier-matched-oa-full-20260717-r2"
CLAIM_SCOPE = "matched_tier_abstracted_descriptive_transfer_diagnostic"
PAPER_SCOPE = "descriptive matched tier-abstracted in_domain-minus-loco diagnostic only"
FORMAL_STATUS = "COMPLETE_VERIFIED_DESCRIPTIVE_RESULT"

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
MODES = ("loco", "in_domain")
SEEDS = (0, 1, 2, 3, 4)
METRICS = (
    "A.lane_average_precision",
    "B1.entry_average_precision",
    "B2.conditional_recall_at_3",
)
METRIC_SOURCE_KEYS = {
    "A.lane_average_precision": ("A", "lane_average_precision"),
    "B1.entry_average_precision": ("B1", "entry_average_precision"),
    "B2.conditional_recall_at_3": ("B2", "conditional_recall_at_3"),
}
EXPECTED_COMPONENT_COUNT = len(CHAINS) * len(MODES) * len(SEEDS)

SUMMARY_ROLE = "results_v2/loco_formal/summary.json"
RECEIPT_ROLE = "results_v2/loco_formal/verification_receipt.json"
FREEZE_ROLE = "results_v2/loco_formal/frozen_manifest.json"
MARKER_ROLE = "results_v2/loco_formal/MAIN_EVALUATION_STARTED.json"

NOT_VALID_FOR = [
    "comparison with the old stage-relation NBFNet as a transfer gap",
    "mode, seed, or chain selection",
    "population inference from six fixed chains",
]
PUBLIC_LIMITATIONS = [
    "This is a descriptive matched diagnostic over six fixed chains, not population inference.",
    "The target chain's early graph is visible in both modes; this is not graph-free cold-start transfer.",
    "Training-edge volume is not equalized: each mode uses every export edge in its prescribed source set, so the gap jointly reflects source set and volume.",
    "Only in_domain minus loco under the same tier/dedup graph contract is valid.",
    "Results under a different relation-resolution graph must not be used as the matched transfer baseline.",
    "The profile, chains, modes, and seeds were fixed; no outcome-based mode, seed, or chain selection is permitted.",
]
PROMOTION_ATTESTATION = {
    "gate": "canonical_live_formal_verify_summary",
    "component_metric_source": "60_controller_verified_component_files",
    "aggregate_source": "60_sanitized_component_metric_records",
    "receipt_role": "operational_log_only_not_a_trust_anchor",
}

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PRIVATE_TEXT_PATTERNS = (
    ("Unix user-home path", re.compile(r"/(?:home|users)/[^/\s]+/", re.IGNORECASE)),
    ("Windows user-home path", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)),
    ("IHPC host alias", re.compile(r"\bmars\d+\b", re.IGNORECASE)),
    # Match the cluster's account-name family without embedding any maintainer's
    # concrete account identifier in a file that is itself publicly released.
    ("IHPC account token", re.compile(r"\bsli\d+\b", re.IGNORECASE)),
    ("private marker", re.compile(r"\bprivate\b", re.IGNORECASE)),
)


class ResultValidationError(ValueError):
    """The formal result cannot be promoted as a complete public result."""


def _fail(role: str, message: str) -> None:
    raise ResultValidationError(f"{role}: {message}")


def _strict_json_payload(raw: bytes, role: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(role, f"cannot read strict JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(role, "top-level value must be an object")
    return payload


def _strict_json_bytes(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    if path.is_symlink():
        _fail(role, "symbolic-link inputs are forbidden")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(role, f"cannot read file: {exc}")
    return _strict_json_payload(raw, role), raw


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail("source file", f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _hash(value: Any, role: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        _fail(role, "expected a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(role, "expected an object")
    if set(value) != expected:
        _fail(
            role,
            f"fields are not exact (observed={sorted(map(str, value))}, expected={sorted(expected)})",
        )
    return value


def _finite(value: Any, role: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(role, "expected a finite number")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        _fail(role, f"expected a finite number in [{lower}, {upper}]")
    return result


def _exact_int(value: Any, expected: int, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(role, f"expected integer {expected}")
    return value


def _close(actual: float, expected: float, role: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        _fail(role, f"{actual!r} != mechanically recomputed {expected!r}")


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values))


def _stat(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        _fail("aggregate", "cannot summarize an empty metric vector")
    return {"n": len(values), "mean": _mean(values), "population_std": _std(values)}


def _require_repo_role(value: Any, expected: str, role: str) -> str:
    if value != expected:
        _fail(role, f"expected canonical repository-relative role {expected!r}")
    path = PurePosixPath(expected)
    if path.is_absolute() or ".." in path.parts or "\\" in expected:
        raise AssertionError(f"invalid built-in artifact role: {expected}")
    return expected


def _privacy_audit(value: Any, role: str = "public result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"host", "hostname", "user", "username", "pid", "worker_id"}:
                _fail(role, f"operational key leaked: {key}")
            if lowered in {"path", "resolved_path", "component_inventory", "pairs"}:
                _fail(role, f"non-public field leaked: {key}")
            _privacy_audit(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _privacy_audit(child, f"{role}[{index}]")
    elif isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _fail(role, "absolute path leaked")
        for label, pattern in PRIVATE_TEXT_PATTERNS:
            if pattern.search(value):
                _fail(role, f"host/user/private token leaked: {label}")


def _expected_specs() -> list[tuple[str, str, int, str]]:
    return [
        (chain, mode, seed, f"{chain}|{mode}|seed{seed}")
        for chain in CHAINS
        for mode in MODES
        for seed in SEEDS
    ]


def _expected_inventory_ids() -> list[str]:
    return sorted(component_id for _, _, _, component_id in _expected_specs())


def _validate_inventory(value: Any) -> tuple[str, dict[str, Mapping[str, Any]]]:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPONENT_COUNT:
        _fail("formal summary.component_inventory", "expected exactly 60 component records")
    observed_ids: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    candidate_rows: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(value):
        role = f"formal summary.component_inventory[{index}]"
        row = _exact_keys(raw, {"component_id", "path", "sha256", "score_artifacts"}, role)
        component_id = row.get("component_id")
        if not isinstance(component_id, str):
            _fail(f"{role}.component_id", "expected a string")
        try:
            chain, mode, seed_label = component_id.split("|")
            seed = int(seed_label.removeprefix("seed"))
        except (ValueError, AttributeError):
            _fail(f"{role}.component_id", "invalid component identity")
        if chain not in CHAINS or mode not in MODES or seed not in SEEDS or seed_label != f"seed{seed}":
            _fail(f"{role}.component_id", "identity is outside the fixed 6x2x5 matrix")
        observed_ids.append(component_id)
        indexed[component_id] = row
        expected_path = f"results_v2/loco_formal/components/{chain}/{mode}/seed_{seed}/component.json"
        _require_repo_role(row.get("path"), expected_path, f"{role}.path")
        _hash(row.get("sha256"), f"{role}.sha256")
        artifacts = _exact_keys(row.get("score_artifacts"), {"A", "B"}, f"{role}.score_artifacts")
        for task in ("A", "B"):
            artifact = _exact_keys(
                artifacts.get(task), {"path", "sha256", "n_rows"}, f"{role}.score_artifacts.{task}"
            )
            if artifact.get("path") != f"score_{task}.csv":
                _fail(f"{role}.score_artifacts.{task}.path", "invalid score artifact basename")
            _hash(artifact.get("sha256"), f"{role}.score_artifacts.{task}.sha256")
            n_rows = artifact.get("n_rows")
            if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
                _fail(f"{role}.score_artifacts.{task}.n_rows", "expected a positive integer")
            key = (chain, task)
            if key in candidate_rows and candidate_rows[key] != n_rows:
                _fail(role, f"{task} row count changed across modes/seeds for {chain}")
            candidate_rows[key] = n_rows
    if observed_ids != _expected_inventory_ids() or len(indexed) != EXPECTED_COMPONENT_COUNT:
        _fail("formal summary.component_inventory", "component IDs/order are not the exact fixed 6x2x5 inventory")
    return _sha256_bytes(_canonical_json_bytes(value)), indexed


def _validate_metric_records(value: Any, role: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPONENT_COUNT:
        _fail(role, "expected exactly 60 component metric records")
    expected_fields = {"component_id", "chain", "mode", "seed", *METRICS}
    cleaned: list[dict[str, Any]] = []
    for index, (chain, mode, seed, component_id) in enumerate(_expected_specs()):
        row_role = f"{role}[{index}]"
        row = _exact_keys(value[index], expected_fields, row_role)
        expected_identity = {
            "component_id": component_id,
            "chain": chain,
            "mode": mode,
            "seed": seed,
        }
        for field, expected in expected_identity.items():
            if field == "seed":
                _exact_int(row.get(field), expected, f"{row_role}.{field}")
            elif row.get(field) != expected:
                _fail(f"{row_role}.{field}", f"expected {expected!r}")
        clean = dict(expected_identity)
        for metric in METRICS:
            clean[metric] = _finite(row.get(metric), f"{row_role}.{metric}", lower=0.0, upper=1.0)
        cleaned.append(clean)
    return cleaned


def _derive_metrics(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    indexed = {str(row["component_id"]): row for row in records}
    if len(indexed) != EXPECTED_COMPONENT_COUNT:
        _fail("component metric records", "component identities are not unique")
    public: dict[str, Any] = {}
    full: dict[str, Any] = {}
    for metric in METRICS:
        by_mode = {
            mode: _stat(
                [
                    float(indexed[f"{chain}|{mode}|seed{seed}"][metric])
                    for chain in CHAINS
                    for seed in SEEDS
                ]
            )
            for mode in MODES
        }
        by_chain = {
            chain: {
                mode: _stat(
                    [float(indexed[f"{chain}|{mode}|seed{seed}"][metric]) for seed in SEEDS]
                )
                for mode in MODES
            }
            for chain in CHAINS
        }
        pairs = []
        for chain in CHAINS:
            for seed in SEEDS:
                inside = float(indexed[f"{chain}|in_domain|seed{seed}"][metric])
                loco = float(indexed[f"{chain}|loco|seed{seed}"][metric])
                pairs.append({"chain": chain, "seed": seed, "value": inside - loco})
        gap_stat = _stat([float(row["value"]) for row in pairs])
        public[metric] = {
            "by_mode": by_mode,
            "by_chain": by_chain,
            "matched_gap": {"definition": "in_domain_minus_loco", **gap_stat},
        }
        full[metric] = {
            "by_mode": by_mode,
            "by_chain": by_chain,
            "matched_gap": {
                "definition": "in_domain_minus_loco",
                **gap_stat,
                "pairs": pairs,
            },
        }
    return public, full


def _validate_stat_against(
    value: Any,
    expected: Mapping[str, Any],
    role: str,
    *,
    gap: bool = False,
) -> None:
    row = _exact_keys(value, {"n", "mean", "population_std"}, role)
    _exact_int(row.get("n"), int(expected["n"]), f"{role}.n")
    mean = _finite(row.get("mean"), f"{role}.mean", lower=-1.0 if gap else 0.0, upper=1.0)
    std = _finite(row.get("population_std"), f"{role}.population_std", lower=0.0, upper=1.0)
    _close(mean, float(expected["mean"]), f"{role}.mean")
    _close(std, float(expected["population_std"]), f"{role}.population_std")


def _validate_metrics_against(
    value: Any,
    expected: Mapping[str, Any],
    role: str,
    *,
    require_pairs: bool,
) -> None:
    metrics = _exact_keys(value, set(METRICS), role)
    for metric in METRICS:
        metric_role = f"{role}.{metric}"
        row = _exact_keys(metrics[metric], {"by_mode", "by_chain", "matched_gap"}, metric_role)
        modes = _exact_keys(row.get("by_mode"), set(MODES), f"{metric_role}.by_mode")
        chains = _exact_keys(row.get("by_chain"), set(CHAINS), f"{metric_role}.by_chain")
        for mode in MODES:
            _validate_stat_against(
                modes[mode], expected[metric]["by_mode"][mode], f"{metric_role}.by_mode.{mode}"
            )
        for chain in CHAINS:
            chain_modes = _exact_keys(chains[chain], set(MODES), f"{metric_role}.by_chain.{chain}")
            for mode in MODES:
                _validate_stat_against(
                    chain_modes[mode],
                    expected[metric]["by_chain"][chain][mode],
                    f"{metric_role}.by_chain.{chain}.{mode}",
                )
        gap_fields = {"definition", "n", "mean", "population_std"}
        if require_pairs:
            gap_fields.add("pairs")
        gap = _exact_keys(row.get("matched_gap"), gap_fields, f"{metric_role}.matched_gap")
        if gap.get("definition") != "in_domain_minus_loco":
            _fail(f"{metric_role}.matched_gap.definition", "only in_domain_minus_loco is permitted")
        _validate_stat_against(
            {key: gap[key] for key in ("n", "mean", "population_std")},
            expected[metric]["matched_gap"],
            f"{metric_role}.matched_gap",
            gap=True,
        )
        if require_pairs:
            pairs = gap.get("pairs")
            expected_pairs = expected[metric]["matched_gap"]["pairs"]
            if not isinstance(pairs, list) or len(pairs) != 30:
                _fail(f"{metric_role}.matched_gap.pairs", "expected exactly 30 matched pairs")
            for index, wanted in enumerate(expected_pairs):
                pair_role = f"{metric_role}.matched_gap.pairs[{index}]"
                pair = _exact_keys(pairs[index], {"chain", "seed", "value"}, pair_role)
                if pair.get("chain") != wanted["chain"]:
                    _fail(pair_role, "chain identity/order changed")
                _exact_int(pair.get("seed"), int(wanted["seed"]), f"{pair_role}.seed")
                observed = _finite(pair.get("value"), f"{pair_role}.value", lower=-1.0, upper=1.0)
                _close(observed, float(wanted["value"]), f"{pair_role}.value")


def _validate_formal_summary(
    source: Mapping[str, Any], source_bytes: bytes
) -> dict[str, Mapping[str, Any]]:
    if source_bytes != _canonical_json_bytes(source):
        _fail("formal summary", "bytes are not the controller's canonical JSON encoding")
    fields = {
        "schema_version",
        "protocol",
        "status",
        "run_id",
        "claim_scope",
        "paper_eligible",
        "paper_eligibility_scope",
        "not_valid_for",
        "freeze_manifest",
        "freeze_manifest_file_sha256",
        "freeze_sha256",
        "config_sha256",
        "main_start_marker_sha256",
        "verified_component_count",
        "expected_component_count",
        "component_inventory",
        "component_inventory_sha256",
        "metrics",
    }
    _exact_keys(source, fields, "formal summary")
    required = {
        "schema_version": FORMAL_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": FORMAL_STATUS,
        "run_id": RUN_ID,
        "claim_scope": CLAIM_SCOPE,
        "paper_eligible": True,
        "paper_eligibility_scope": PAPER_SCOPE,
        "not_valid_for": NOT_VALID_FOR,
    }
    for field, expected in required.items():
        if source.get(field) != expected:
            _fail(f"formal summary.{field}", f"expected fixed formal value {expected!r}")
    if source.get("paper_eligible") is not True:
        _fail("formal summary.paper_eligible", "expected the JSON boolean true")
    _require_repo_role(source.get("freeze_manifest"), FREEZE_ROLE, "formal summary.freeze_manifest")
    for field in (
        "freeze_manifest_file_sha256",
        "freeze_sha256",
        "config_sha256",
        "main_start_marker_sha256",
        "component_inventory_sha256",
    ):
        _hash(source.get(field), f"formal summary.{field}")
    _exact_int(source.get("verified_component_count"), 60, "formal summary.verified_component_count")
    _exact_int(source.get("expected_component_count"), 60, "formal summary.expected_component_count")
    inventory_hash, inventory = _validate_inventory(source.get("component_inventory"))
    if inventory_hash != source["component_inventory_sha256"]:
        _fail("formal summary.component_inventory_sha256", "does not bind the exact inventory")
    return inventory


def _load_canonical_controller() -> Any:
    if not CANONICAL_CONTROLLER.is_file():
        _fail("live formal verification", "canonical controller is missing")
    spec = importlib.util.spec_from_file_location(
        "_upgrade_bench_loco_promotion_controller", CANONICAL_CONTROLLER
    )
    if spec is None or spec.loader is None:
        _fail("live formal verification", "cannot load canonical controller")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail("live formal verification", f"cannot import canonical controller: {exc}")
    origin = Path(getattr(module, "__file__", "")).resolve()
    if origin != CANONICAL_CONTROLLER.resolve():
        _fail("live formal verification", "controller module origin is not canonical")
    constants = {
        "PROTOCOL": PROTOCOL,
        "SUMMARY_SCHEMA": FORMAL_SUMMARY_SCHEMA,
        "CANONICAL_RUN_ID": RUN_ID,
        "CLAIM_SCOPE": CLAIM_SCOPE,
        "EXPECTED_COMPONENT_COUNT": EXPECTED_COMPONENT_COUNT,
    }
    for name, expected in constants.items():
        if getattr(module, name, None) != expected:
            _fail("live formal verification", f"controller {name} changed")
    return module


def _live_verify_and_load_components(
    summary_path: Path,
    source: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Run the canonical verifier and reread the exact verified component bytes."""
    summary_path = Path(summary_path).resolve()
    if summary_path != PRIVATE_SUMMARY.resolve():
        _fail("live formal verification", "promotion accepts only the canonical formal summary")
    if PRIVATE_MANIFEST.resolve() != (summary_path.parent / "frozen_manifest.json").resolve():
        raise AssertionError("canonical manifest layout changed")
    controller = _load_canonical_controller()
    try:
        summary_verification = controller.verify_summary(
            summary_path, PRIVATE_MANIFEST.resolve(), invoke_runner=True
        )
        manifest = controller.verify_freeze(PRIVATE_MANIFEST.resolve())
    except Exception as exc:
        _fail("live formal verification", f"canonical controller refused promotion: {exc}")
    if not isinstance(summary_verification, dict):
        _fail("live formal verification", "controller returned no verification object")

    config_sha256 = _sha256_file(CANONICAL_CONFIG)
    if config_sha256 != source["config_sha256"]:
        _fail("live formal verification", "current canonical config bytes do not match summary")
    if manifest.get("config_sha256") != config_sha256 or manifest.get("run_id") != RUN_ID:
        _fail("live formal verification", "verified manifest is not bound to current config/run")
    marker_path = PRIVATE_ROOT / "MAIN_EVALUATION_STARTED.json"
    sentinel_hashes = {
        CANONICAL_CONFIG: config_sha256,
        PRIVATE_MANIFEST: str(source["freeze_manifest_file_sha256"]),
        marker_path: str(source["main_start_marker_sha256"]),
        summary_path: _sha256_bytes(_canonical_json_bytes(source)),
    }
    for path, expected_hash in sentinel_hashes.items():
        if path.is_symlink() or _sha256_file(path) != expected_hash:
            _fail("live formal verification", f"verified sentinel changed: {path.name}")

    specs = {
        str(row["component_id"]): row for row in manifest.get("expected_components", [])
    }
    if set(specs) != set(_expected_inventory_ids()) or len(specs) != EXPECTED_COMPONENT_COUNT:
        _fail("live formal verification", "manifest component matrix is not exact")
    records: list[dict[str, Any]] = []
    for chain, mode, seed, component_id in _expected_specs():
        spec_row = specs[component_id]
        expected_spec = {"component_id": component_id, "chain": chain, "mode": mode, "seed": seed}
        for key, expected in expected_spec.items():
            if spec_row.get(key) != expected:
                _fail("live formal verification", f"manifest component {component_id} {key} changed")
        component_path = (PRIVATE_ROOT / str(spec_row["path"])).resolve()
        if not component_path.is_relative_to(PRIVATE_ROOT.resolve()) or component_path.is_symlink():
            _fail("live formal verification", f"unsafe component path for {component_id}")
        try:
            component_bytes = component_path.read_bytes()
        except OSError as exc:
            _fail("live formal verification", f"cannot reread {component_id}: {exc}")
        if _sha256_bytes(component_bytes) != inventory[component_id]["sha256"]:
            _fail("live formal verification", f"component changed after verification: {component_id}")
        for task in ("A", "B"):
            score_record = inventory[component_id]["score_artifacts"][task]
            score_path = (component_path.parent / str(score_record["path"])).resolve()
            if (
                not score_path.is_relative_to(component_path.parent.resolve())
                or score_path.is_symlink()
                or _sha256_file(score_path) != score_record["sha256"]
            ):
                _fail(
                    "live formal verification",
                    f"score artifact changed after verification: {component_id}|{task}",
                )
        payload = _strict_json_payload(component_bytes, f"component {component_id}")
        identity = {
            "component_id": component_id,
            "holdout_chain": chain,
            "mode": mode,
            "seed": seed,
            "run_id": RUN_ID,
        }
        for field, expected in identity.items():
            if payload.get(field) != expected:
                _fail("live formal verification", f"component {component_id} {field} changed")
        record: dict[str, Any] = {
            "component_id": component_id,
            "chain": chain,
            "mode": mode,
            "seed": seed,
        }
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            _fail("live formal verification", f"component {component_id} metrics missing")
        for label, (task, metric) in METRIC_SOURCE_KEYS.items():
            task_metrics = raw_metrics.get(task)
            if not isinstance(task_metrics, Mapping):
                _fail("live formal verification", f"component {component_id} task {task} missing")
            record[label] = _finite(
                task_metrics.get(metric),
                f"component {component_id}.{label}",
                lower=0.0,
                upper=1.0,
            )
        records.append(record)
    for path, expected_hash in sentinel_hashes.items():
        if _sha256_file(path) != expected_hash:
            _fail("live formal verification", f"verified sentinel changed during promotion: {path.name}")
    live_receipt = {
        "status": "VERIFIED_COMPLETE",
        "freeze_sha256": manifest["freeze_sha256"],
        "verified_component_count": EXPECTED_COMPONENT_COUNT,
        "summary_verification": summary_verification,
    }
    return live_receipt, records, config_sha256


def _validate_receipt_log(
    payload: Mapping[str, Any],
    raw_summary: bytes,
    source: Mapping[str, Any],
    live_receipt: Mapping[str, Any],
) -> None:
    """Check log completeness; scientific trust comes only from the live gate."""
    receipt = _exact_keys(
        payload,
        {"status", "freeze_sha256", "verified_component_count", "summary_verification"},
        "verification receipt log",
    )
    if receipt.get("status") != "VERIFIED_COMPLETE":
        _fail("verification receipt log.status", "expected VERIFIED_COMPLETE")
    _exact_int(receipt.get("verified_component_count"), 60, "verification receipt log.count")
    if _hash(receipt.get("freeze_sha256"), "verification receipt log.freeze_sha256") != source["freeze_sha256"]:
        _fail("verification receipt log.freeze_sha256", "does not match summary")
    verification = _exact_keys(
        receipt.get("summary_verification"),
        {"status", "summary", "summary_sha256", "freeze_sha256", "verified_component_count"},
        "verification receipt log.summary_verification",
    )
    if verification.get("status") != "VERIFIED":
        _fail("verification receipt log.summary_verification.status", "expected VERIFIED")
    if not isinstance(verification.get("summary"), str) or not verification.get("summary"):
        _fail("verification receipt log.summary_verification.summary", "expected a log locator string")
    _exact_int(
        verification.get("verified_component_count"),
        60,
        "verification receipt log.summary_verification.count",
    )
    if _hash(
        verification.get("freeze_sha256"),
        "verification receipt log.summary_verification.freeze_sha256",
    ) != source["freeze_sha256"]:
        _fail("verification receipt log.summary_verification.freeze_sha256", "does not match summary")
    observed_summary_hash = _hash(
        verification.get("summary_sha256"),
        "verification receipt log.summary_verification.summary_sha256",
    )
    if observed_summary_hash != _sha256_bytes(raw_summary):
        _fail("verification receipt log.summary_verification.summary_sha256", "does not match summary bytes")

    # Ignore the saved absolute locator, but require every scientific field to
    # equal the just-computed live controller result.
    live_verification = live_receipt.get("summary_verification")
    if not isinstance(live_verification, Mapping):
        _fail("live formal verification", "missing summary verification")
    compared = {
        "status": "VERIFIED_COMPLETE",
        "freeze_sha256": live_receipt.get("freeze_sha256"),
        "verified_component_count": live_receipt.get("verified_component_count"),
        "summary_status": live_verification.get("status"),
        "summary_sha256": live_verification.get("summary_sha256"),
        "summary_freeze_sha256": live_verification.get("freeze_sha256"),
        "summary_verified_component_count": live_verification.get("verified_component_count"),
    }
    observed = {
        "status": receipt.get("status"),
        "freeze_sha256": receipt.get("freeze_sha256"),
        "verified_component_count": receipt.get("verified_component_count"),
        "summary_status": verification.get("status"),
        "summary_sha256": verification.get("summary_sha256"),
        "summary_freeze_sha256": verification.get("freeze_sha256"),
        "summary_verified_component_count": verification.get("verified_component_count"),
    }
    if observed != compared:
        _fail("verification receipt log", "scientific fields differ from live controller output")


def build_summary(
    summary_path: Path = PRIVATE_SUMMARY,
    receipt_path: Path = PRIVATE_RECEIPT,
) -> dict[str, Any]:
    """Run the live gate and build a sanitized public result in memory."""
    source, source_bytes = _strict_json_bytes(summary_path, "formal summary")
    inventory = _validate_formal_summary(source, source_bytes)
    live_receipt, raw_records, config_sha256 = _live_verify_and_load_components(
        summary_path, source, inventory
    )
    if config_sha256 != source["config_sha256"]:
        _fail("live formal verification", "config byte hash mismatch")
    records = _validate_metric_records(raw_records, "verified component metric records")
    public_metrics, full_metrics = _derive_metrics(records)
    _validate_metrics_against(
        source.get("metrics"), full_metrics, "formal summary.metrics", require_pairs=True
    )

    receipt, receipt_bytes = _strict_json_bytes(receipt_path, "verification receipt log")
    _validate_receipt_log(receipt, source_bytes, source, live_receipt)

    result = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "run_id": RUN_ID,
        "claim_scope": CLAIM_SCOPE,
        "paper_eligible": True,
        "paper_eligibility_scope": PAPER_SCOPE,
        "promotion_attestation": dict(PROMOTION_ATTESTATION),
        "source_artifacts": {
            "formal_summary": {
                "artifact_role": SUMMARY_ROLE,
                "sha256": _sha256_bytes(source_bytes),
            },
            "verification_receipt": {
                "artifact_role": RECEIPT_ROLE,
                "sha256": _sha256_bytes(receipt_bytes),
            },
        },
        "provenance": {
            "freeze_manifest_artifact_role": FREEZE_ROLE,
            "freeze_manifest_file_sha256": source["freeze_manifest_file_sha256"],
            "freeze_sha256": source["freeze_sha256"],
            "config_sha256": source["config_sha256"],
            "main_start_marker_artifact_role": MARKER_ROLE,
            "main_start_marker_sha256": source["main_start_marker_sha256"],
        },
        "design": {
            "chains": list(CHAINS),
            "modes": list(MODES),
            "seeds": list(SEEDS),
            "expected_component_count": 60,
            "verified_component_count": 60,
            "comparison_definition": "in_domain_minus_loco",
        },
        "metric_records": records,
        "metrics": public_metrics,
        "limitations": list(PUBLIC_LIMITATIONS),
    }
    return validate_public_summary(result)


def validate_public_summary(value: Any) -> dict[str, Any]:
    """Validate public JSON and recompute all aggregates from metric records."""
    top = _exact_keys(
        value,
        {
            "schema_version",
            "protocol",
            "status",
            "run_id",
            "claim_scope",
            "paper_eligible",
            "paper_eligibility_scope",
            "promotion_attestation",
            "source_artifacts",
            "provenance",
            "design",
            "metric_records",
            "metrics",
            "limitations",
        },
        "public summary",
    )
    required = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "run_id": RUN_ID,
        "claim_scope": CLAIM_SCOPE,
        "paper_eligible": True,
        "paper_eligibility_scope": PAPER_SCOPE,
        "promotion_attestation": PROMOTION_ATTESTATION,
        "limitations": PUBLIC_LIMITATIONS,
    }
    for field, expected in required.items():
        if top.get(field) != expected:
            _fail(f"public summary.{field}", f"expected fixed value {expected!r}")
    if top.get("paper_eligible") is not True:
        _fail("public summary.paper_eligible", "expected the JSON boolean true")

    sources = _exact_keys(
        top.get("source_artifacts"), {"formal_summary", "verification_receipt"}, "public sources"
    )
    cleaned_sources: dict[str, Any] = {}
    for name, artifact_role in (
        ("formal_summary", SUMMARY_ROLE),
        ("verification_receipt", RECEIPT_ROLE),
    ):
        item = _exact_keys(sources[name], {"artifact_role", "sha256"}, f"public sources.{name}")
        _require_repo_role(item.get("artifact_role"), artifact_role, f"public sources.{name}.role")
        cleaned_sources[name] = {"artifact_role": artifact_role, "sha256": _hash(item.get("sha256"), f"public sources.{name}.sha256")}

    provenance = _exact_keys(
        top.get("provenance"),
        {
            "freeze_manifest_artifact_role",
            "freeze_manifest_file_sha256",
            "freeze_sha256",
            "config_sha256",
            "main_start_marker_artifact_role",
            "main_start_marker_sha256",
        },
        "public provenance",
    )
    _require_repo_role(
        provenance.get("freeze_manifest_artifact_role"), FREEZE_ROLE, "public provenance.freeze role"
    )
    _require_repo_role(
        provenance.get("main_start_marker_artifact_role"), MARKER_ROLE, "public provenance.marker role"
    )
    cleaned_provenance = {
        "freeze_manifest_artifact_role": FREEZE_ROLE,
        "freeze_manifest_file_sha256": _hash(
            provenance.get("freeze_manifest_file_sha256"), "public provenance.manifest hash"
        ),
        "freeze_sha256": _hash(provenance.get("freeze_sha256"), "public provenance.freeze hash"),
        "config_sha256": _hash(provenance.get("config_sha256"), "public provenance.config hash"),
        "main_start_marker_artifact_role": MARKER_ROLE,
        "main_start_marker_sha256": _hash(
            provenance.get("main_start_marker_sha256"), "public provenance.marker hash"
        ),
    }
    design = _exact_keys(
        top.get("design"),
        {
            "chains",
            "modes",
            "seeds",
            "expected_component_count",
            "verified_component_count",
            "comparison_definition",
        },
        "public design",
    )
    design_expected = {
        "chains": list(CHAINS),
        "modes": list(MODES),
        "seeds": list(SEEDS),
        "expected_component_count": 60,
        "verified_component_count": 60,
        "comparison_definition": "in_domain_minus_loco",
    }
    for field, expected in design_expected.items():
        if design.get(field) != expected:
            _fail(f"public design.{field}", f"expected {expected!r}")
    _exact_int(design.get("expected_component_count"), 60, "public design.expected count")
    _exact_int(design.get("verified_component_count"), 60, "public design.verified count")

    records = _validate_metric_records(top.get("metric_records"), "public metric_records")
    expected_metrics, _ = _derive_metrics(records)
    _validate_metrics_against(
        top.get("metrics"), expected_metrics, "public summary.metrics", require_pairs=False
    )
    cleaned = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "run_id": RUN_ID,
        "claim_scope": CLAIM_SCOPE,
        "paper_eligible": True,
        "paper_eligibility_scope": PAPER_SCOPE,
        "promotion_attestation": dict(PROMOTION_ATTESTATION),
        "source_artifacts": cleaned_sources,
        "provenance": cleaned_provenance,
        "design": design_expected,
        "metric_records": records,
        "metrics": expected_metrics,
        "limitations": list(PUBLIC_LIMITATIONS),
    }
    _privacy_audit(cleaned)
    return cleaned


CSV_FIELDS = (
    "scope",
    "metric",
    "chain",
    "mode",
    "definition",
    "n",
    "mean",
    "population_std",
    "source_summary_sha256",
    "verification_receipt_sha256",
    "freeze_sha256",
    "config_sha256",
    "main_start_marker_sha256",
    "run_id",
    "protocol",
)


def render_json(summary: Mapping[str, Any]) -> bytes:
    _privacy_audit(summary)
    return _canonical_json_bytes(summary, pretty=True)


def render_csv(summary: Mapping[str, Any]) -> bytes:
    _privacy_audit(summary)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    common = {
        "source_summary_sha256": summary["source_artifacts"]["formal_summary"]["sha256"],
        "verification_receipt_sha256": summary["source_artifacts"]["verification_receipt"]["sha256"],
        "freeze_sha256": summary["provenance"]["freeze_sha256"],
        "config_sha256": summary["provenance"]["config_sha256"],
        "main_start_marker_sha256": summary["provenance"]["main_start_marker_sha256"],
        "run_id": summary["run_id"],
        "protocol": summary["protocol"],
    }
    for metric in METRICS:
        values = summary["metrics"][metric]
        for mode in MODES:
            stat = values["by_mode"][mode]
            writer.writerow(
                {
                    **common,
                    "scope": "mode",
                    "metric": metric,
                    "chain": "__all__",
                    "mode": mode,
                    "definition": "",
                    "n": stat["n"],
                    "mean": format(float(stat["mean"]), ".17g"),
                    "population_std": format(float(stat["population_std"]), ".17g"),
                }
            )
        for chain in CHAINS:
            for mode in MODES:
                stat = values["by_chain"][chain][mode]
                writer.writerow(
                    {
                        **common,
                        "scope": "chain_mode",
                        "metric": metric,
                        "chain": chain,
                        "mode": mode,
                        "definition": "",
                        "n": stat["n"],
                        "mean": format(float(stat["mean"]), ".17g"),
                        "population_std": format(float(stat["population_std"]), ".17g"),
                    }
                )
        gap = values["matched_gap"]
        writer.writerow(
            {
                **common,
                "scope": "matched_gap",
                "metric": metric,
                "chain": "__all__",
                "mode": "",
                "definition": gap["definition"],
                "n": gap["n"],
                "mean": format(float(gap["mean"]), ".17g"),
                "population_std": format(float(gap["population_std"]), ".17g"),
            }
        )
    encoded = buffer.getvalue().encode("utf-8")
    rendered = encoded.decode("utf-8")
    for label, pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(rendered):
            _fail("public CSV", f"host/user/private token leaked: {label}")
    return encoded


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry on Linux; formal promotion runs on Linux."""
    if os.name == "nt":  # Windows does not expose portable directory fsync.
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_proven_crash_links(path: Path) -> None:
    """Remove only stale temp names proven to be hard links to *path*."""
    for temporary in sorted(path.parent.glob(f".{path.name}.*.tmp")):
        if (
            path.is_file()
            and not path.is_symlink()
            and temporary.is_file()
            and not temporary.is_symlink()
            and os.path.samefile(temporary, path)
        ):
            temporary.unlink()
            _fsync_directory(path.parent)
            continue
        _fail(
            "published output",
            f"unrecognized promotion temp artifact requires inspection: {temporary.name}",
        )
    if path.exists() and path.stat().st_nlink != 1:
        _fail("published output", f"{path.name} has unexpected hard links")


def _atomic_create_or_match(path: Path, content: bytes) -> None:
    """Publish exact bytes without overwriting; accept only an identical resume."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve() != path.parent:
        _fail("published output", f"{path.name} parent is not a physical directory")
    if path.is_symlink():
        _fail("published output", f"{path.name} destination is a symbolic link")
    _cleanup_proven_crash_links(path)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                _fail("published output", f"{path.name} destination is not a regular file")
            try:
                observed = path.read_bytes()
            except OSError:
                _fail("published output", f"cannot read existing {path.name}")
            if observed != content:
                _fail(
                    "published output",
                    f"refusing to overwrite non-matching existing {path.name}",
                )
        else:
            linked = True
            _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(path.parent)


def _same_physical_parent(left: Path, right: Path) -> bool:
    return Path(left).absolute().parent == Path(right).absolute().parent


def write_outputs(
    summary_path: Path,
    receipt_path: Path,
    json_out: Path,
    csv_out: Path,
) -> dict[str, Any]:
    summary = build_summary(summary_path, receipt_path)
    json_bytes = render_json(summary)
    csv_bytes = render_csv(summary)
    if not _same_physical_parent(json_out, csv_out):
        _fail("published output", "JSON and CSV must share one physical parent directory")
    _atomic_create_or_match(json_out, json_bytes)
    _atomic_create_or_match(csv_out, csv_bytes)
    verify_outputs(json_out, csv_out)
    return summary


def verify_outputs(json_out: Path, csv_out: Path) -> dict[str, Any]:
    """Public-only deterministic verifier; never reads the formal run tree."""
    json_out = Path(json_out).absolute()
    csv_out = Path(csv_out).absolute()
    if not _same_physical_parent(json_out, csv_out):
        _fail("published output", "JSON and CSV must share one physical parent directory")
    parent = json_out.parent
    if parent.is_symlink() or parent.resolve() != parent:
        _fail("published output", "public output parent is not a physical directory")
    if csv_out.is_symlink():
        _fail("published output", f"{csv_out.name} is a symbolic link")
    observed, raw = _strict_json_bytes(json_out, "published JSON")
    summary = validate_public_summary(observed)
    expected_json = render_json(summary)
    if raw != expected_json:
        _fail("published output", f"{Path(json_out).name} is stale or non-canonical")
    expected_csv = render_csv(summary)
    try:
        observed_csv = Path(csv_out).read_bytes()
    except OSError as exc:
        _fail("published output", f"cannot read {Path(csv_out).name}: {exc}")
    if observed_csv != expected_csv:
        _fail("published output", f"{Path(csv_out).name} is stale or non-deterministic")
    try:
        if json_out.read_bytes() != raw or csv_out.read_bytes() != observed_csv:
            _fail("published output", "public output changed during verification")
    except OSError as exc:
        _fail("published output", f"cannot re-read stable output bytes: {exc}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-output",
        action="store_true",
        help="public-only byte-verification of canonical JSON and derived CSV",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="run the complete live promotion gate without writing public outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_output:
            summary = verify_outputs(DEFAULT_JSON_OUT, DEFAULT_CSV_OUT)
            action = "public outputs verified"
        elif args.check_only:
            summary = build_summary(PRIVATE_SUMMARY, PRIVATE_RECEIPT)
            action = "live gate passed (no files written)"
        else:
            summary = write_outputs(
                PRIVATE_SUMMARY, PRIVATE_RECEIPT, DEFAULT_JSON_OUT, DEFAULT_CSV_OUT
            )
            action = "public outputs written"
    except ResultValidationError as exc:
        print(f"LOCO RESULT PROMOTION REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        f"LOCO {action}: {summary['design']['verified_component_count']} component records / "
        "30 matched pairs per metric; comparison=in_domain_minus_loco"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
