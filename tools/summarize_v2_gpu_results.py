"""Validate and deterministically summarize the canonical v2 GPU main run.

This is a fail-closed promotion gate, not an experiment runner.  It reads only
``results_v2/gpu_rolling`` and will not emit public artifacts until the complete
6-chain x 3-track x 2-family main evaluation is present and internally
consistent with the frozen fold2 selections.  In particular, it never chooses
a model family from main-window performance.  Promotion also requires exactly
one successful private worker claim for each of the twelve chain/family jobs,
then recomputes every main metric and deterministic cluster-bootstrap interval
from the current canonical candidates and the five frozen score columns.

Promotion additionally requires the exact eight-file private NBFNet evidence
set: four run-bound source receipts, complete four-tree and two-host runtime
comparisons, and one durable pre-main formal-gate receipt per main host.  The
public summary retains only path-free roles, hashes, equality facts, and the
explicit boundary that selection evidence is retrospective rather than a
contemporaneous freeze.

The canonical raw artifacts are private operational provenance and can contain
absolute host paths.  The generated JSON/CSV contain only repository-relative
logical roles, hashes, protocol metadata, and scientific results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from v2_gpu_protocol import (  # noqa: E402
    FAMILIES,
    PROTOCOL,
    TARGET_FOLD,
    TRACKS,
    ProtocolError,
    selection_key,
    sha256_file,
    verify_freeze_manifest,
)
import build_gpu_step3_postfreeze_attestation as postfreeze  # noqa: E402
import build_nbfnet_source_attestation as nbfnet_attestation  # noqa: E402
import v2_gpu_rolling as gpu_runner  # noqa: E402


CANONICAL_ROOT = ROOT / "results_v2" / "gpu_rolling"
CANONICAL_RUN_CONFIG = ROOT / "configs" / "v2_gpu_rolling.json"
CANONICAL_CANDIDATE_ROOT = ROOT / "data" / "processed_v2"
DEFAULT_JSON_OUT = ROOT / "results_v2" / "metrics" / "v2_gpu_rolling_summary.json"
DEFAULT_CSV_OUT = ROOT / "results_v2" / "metrics" / "v2_gpu_rolling_summary.csv"

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
SEEDS = (0, 1, 2, 3, 4)
KEYS = ("i_iso", "j_iso", "stage")
EVALUATION_SCHEMA = "upgrade-bench-v2/gpu-evaluation/1"
SUMMARY_SCHEMA = "upgrade-bench-v2/gpu-main-summary/1"
MAIN_START_SCHEMA = "upgrade-bench-v2/main-start/1"
RUN_CONFIG_SCHEMA = "upgrade-bench-v2/gpu-run-config/1"
BENCHMARK_VERSION = "2.1-dev"
FORMAL_EXECUTION_STATUS = "FORMAL_RUN_AUTHORIZED"
EXPECTED_EVALUATIONS = len(CHAINS) * len(TRACKS) * len(FAMILIES)
EXPECTED_CHAIN_FAMILY_JOBS = len(CHAINS) * len(FAMILIES)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
SAFE_WORKER_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
NBFNET_EVIDENCE_SCHEMA = "upgrade-bench-v2/nbfnet-summary-evidence/1"
NBFNET_FORMAL_GATE_SCHEMA = (
    "upgrade-bench-v2/nbfnet-formal-gate-receipt/1"
)
NBFNET_ATTESTATION_TOOL_SHA256 = (
    "88a2d0543264d9ce4b562063840eafc87fb611066eff0618c7755f04a8ca9510"
)
NBFNET_PRIVATE_DIRECTORY = "nbfnet_attestation"
NBFNET_RECEIPT_ROLES = (
    "selection-host-a",
    "selection-host-b",
    "main-host-a",
    "main-host-b",
)
NBFNET_MAIN_ROLES = ("main-host-a", "main-host-b")
NBFNET_PRIVATE_FILENAMES = {
    role: f"{role}.json" for role in NBFNET_RECEIPT_ROLES
}
NBFNET_PRIVATE_SOURCE_COMPARISON = "source-comparison.json"
NBFNET_PRIVATE_RUNTIME_COMPARISON = "runtime-comparison.json"
NBFNET_FORMAL_GATE_FILENAMES = {
    role: f"formal_gate_{role}.json" for role in NBFNET_MAIN_ROLES
}
NBFNET_PUBLIC_PROJECTION_ROLES = {
    role: f"chains/evidence/nbfnet_{role}.public.json"
    for role in NBFNET_RECEIPT_ROLES
}
NBFNET_PUBLIC_SOURCE_COMPARISON_ROLE = (
    "chains/evidence/nbfnet_source_comparison.json"
)
NBFNET_PUBLIC_RUNTIME_COMPARISON_ROLE = (
    "chains/evidence/nbfnet_runtime_comparison.json"
)
NBFNET_SELECTION_CLAIM = (
    "retrospective same-tree and non-later-mtime consistency evidence only; "
    "no selection-time prehash or contemporaneous source freeze is claimed"
)
NBFNET_MAIN_CLAIM = (
    "genuine read-only pre-main source snapshot, with executable Python bytecode "
    "outside the hashed inventory prohibited by the formal gate"
)
NBFNET_RECEIPT_SUPPORTED_CLAIM = (
    "At the recorded observation time, the tree named by NBFNET_PATH had the "
    "listed relative files and exact byte hashes and was bound to the stated "
    "frozen run and Step-3 manifest identities."
)
NBFNET_RECEIPT_UNSUPPORTED_CLAIM = (
    "Because this receipt is retrospective, it does not by itself prove that "
    "the observed tree was unchanged since selection or main execution; it is "
    "supplemental evidence, not a replacement for a contemporaneous freeze. "
    "Runtime artifacts support an execution-binary claim only when explicitly "
    "provided, and cross-host equality requires compare-runtime."
)
NBFNET_SOURCE_COMPARISON_CLAIMS = {
    "supported": (
        "The listed retrospectively observed source trees have identical complete "
        "included-file inventories and byte hashes exactly when "
        "all_source_trees_match is true."
    ),
    "not_supported": (
        "Equality between the external selection tree and a frozen main snapshot "
        "does not turn the earlier selection observation into a contemporaneous "
        "freeze; the main snapshot can be genuinely pre-main frozen separately."
    ),
}
NBFNET_RUNTIME_COMPARISON_CLAIMS = {
    "supported": (
        "The listed retrospective host receipts bind identical source/runtime "
        "bytes exactly when the two all_*_match fields are true."
    ),
    "not_supported": (
        "Cross-host byte equality does not prove that the compiled extension was "
        "loaded by every earlier worker unless contemporaneous execution evidence "
        "independently identifies that artifact."
    ),
}
PRIVATE_TEXT_PATTERNS = (
    ("Unix user-home path", re.compile(r"/(?:home|users)/[^/\s]+/", re.IGNORECASE)),
    ("Windows user-home path", re.compile(r"[A-Za-z]:\\+Users\\+", re.IGNORECASE)),
    ("institutional host alias", re.compile(r"\bmars\d+\b", re.IGNORECASE)),
)

PRIMARY_METRICS = {
    "a": "lane_average_precision",
    "b1": "entry_average_precision",
    "b2": "conditional_recall_at_3",
}
SELECTION_METRICS = {
    "a": "track_a_lane_average_precision",
    "b1": "track_b1_entry_average_precision_max_lane_score",
    "b2": "track_b2_positive_entry_macro_recall_at_3",
}
BOOTSTRAP_SPECS = {
    "a": ("exporter", "lane_average_precision"),
    "b1": ("exporter", "entry_average_precision"),
    "b2": ("exporter_stage", "positive_entry_macro_recall_at_3"),
}

COMMON_METRICS = {
    "lane_average_precision",
    "lane_roc_auc",
    "within_size_decile_auc",
}
TRACK_METRICS = {
    "a": COMMON_METRICS
    | {
        *(
            f"{kind}_at_{budget}"
            for kind in ("precision", "recall", "value_capture")
            for budget in (50, 100, 250, 500)
        ),
        *(
            f"exporter_macro_{kind}_at_{budget}"
            for kind in ("precision", "recall", "value_capture")
            for budget in (5, 10)
        ),
        "exporters",
    },
    "b1": COMMON_METRICS
    | {
        "entry_average_precision",
        "entry_roc_auc",
        *(f"entry_{kind}_at_{budget}" for kind in ("precision", "recall") for budget in (25, 50, 100, 250)),
        "entry_groups",
    },
    "b2": COMMON_METRICS
    | {
        *(f"conditional_{kind}_at_{budget}" for kind in ("recall", "value_capture") for budget in (1, 3, 5)),
        "positive_entry_groups",
    },
}
COUNT_METRICS = {"exporters", "entry_groups", "positive_entry_groups"}

KGE_MODELS = ("TransE", "RotatE", "DistMult", "ComplEx", "RGCN", "CompGCN")
FORMAL_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    model: tuple(
        {
            "embedding_dim": dim,
            "learning_rate": rate,
            "epochs": 150,
            "batch_size": 2048,
        }
        for dim in (64, 128)
        for rate in (0.005, 0.01)
    )
    for model in KGE_MODELS
}
FORMAL_GRIDS["NBFNet"] = tuple(
    {
        "layers": layers,
        "learning_rate": rate,
        "epochs": 25,
        "batch_size": 64,
        "negatives": 32,
    }
    for layers in (4, 6)
    for rate in (0.001, 0.005)
)


class ResultValidationError(ValueError):
    """The private run cannot be promoted as a complete scientific result."""


def _fail(role: str, message: str) -> None:
    raise ResultValidationError(f"{role}: {message}")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _fail(role, f"cannot read strict JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(role, "top-level value must be an object")
    return payload


def _hash(value: Any, role: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        _fail(role, "expected a lowercase SHA-256 digest")
    return value


def _finite(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(role, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(role, "non-finite numbers are not promotable")
    return result


def _finite_csv(value: Any, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        _fail(role, f"expected a numeric CSV value: {exc}")
    if not math.isfinite(result):
        _fail(role, "non-finite CSV score is not promotable")
    return result


def _probability(value: Any, role: str) -> float:
    result = _finite(value, role)
    if not 0.0 <= result <= 1.0:
        _fail(role, "expected a metric in [0, 1]")
    return result


def _integer(value: Any, role: str, *, minimum: int = 0) -> int:
    result = _finite(value, role)
    if not result.is_integer() or result < minimum:
        _fail(role, f"expected an integer >= {minimum}")
    return int(result)


def _timestamp(value: Any, role: str) -> datetime:
    if not isinstance(value, str):
        _fail(role, "expected an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        _fail(role, f"invalid ISO-8601 timestamp: {exc}")
    if parsed.tzinfo is None:
        _fail(role, "timestamp must include a UTC offset")
    return parsed


def _close(actual: Any, expected: float, role: str) -> None:
    value = _finite(actual, role)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        _fail(role, f"{value!r} != mechanically recomputed {expected!r}")


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values)


def _canonical_key(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _expected_target_role(chain: str, track: str) -> str:
    stem = "candidates" if track == "a" else "candidates_firsttime"
    return f"data/processed_v2/{stem}_{chain}.csv"


def _deployment_sources(root: Path) -> tuple[Path, Path]:
    """Return the canonical config and cohort roots for one extracted run tree."""
    root = Path(root).resolve()
    if root.name != "gpu_rolling" or root.parent.name != "results_v2":
        _fail("run namespace", "private run must be at results_v2/gpu_rolling")
    deployment_root = root.parents[1]
    return (
        deployment_root / "configs" / "v2_gpu_rolling.json",
        deployment_root / "data" / "processed_v2",
    )


def _verify_postfreeze_attestation(deployment_root: Path, run_id: str) -> dict[str, Any]:
    """Require the exact public old/new registry-equivalence receipt."""

    artifact = Path(deployment_root) / postfreeze.ARTIFACT_ROLE
    try:
        payload = postfreeze.verify_output(
            artifact,
            root=deployment_root,
            require_full_inventory=True,
        )
    except postfreeze.AttestationError as exc:
        _fail("post-freeze semantic attestation", str(exc))
    if payload.get("run_id") != run_id:
        _fail("post-freeze semantic attestation", "formal run_id differs from frozen GPU run")
    return postfreeze.summary_binding(artifact)


def _validate_metric_source(deployment_root: Path) -> str:
    """Bind recomputation to the exact runner source attested for this run.

    ``gpu_runner`` is imported from the checkout executing this promotion gate.
    A staged run may live under another root, so compare byte hashes rather than
    paths.  The post-freeze attestation independently requires the deployed
    copy to be the exact Step-3-governed source.
    """

    deployed = Path(deployment_root) / "src" / "v2_gpu_rolling.py"
    executed = Path(gpu_runner.__file__).resolve()
    if not deployed.is_file():
        _fail("metric recomputation source", f"missing deployed runner: {deployed}")
    deployed_sha256 = sha256_file(deployed)
    executed_sha256 = sha256_file(executed)
    if deployed_sha256 != executed_sha256:
        _fail(
            "metric recomputation source",
            "executed runner bytes differ from the attested deployment source",
        )
    return executed_sha256


def _validate_nbfnet_attestation_source(deployment_root: Path) -> str:
    """Require the reviewed implementation in public and private run roots."""

    deployed = Path(deployment_root) / "tools" / "build_nbfnet_source_attestation.py"
    private_run_copy = (
        Path(deployment_root)
        / "private"
        / "build_nbfnet_source_attestation_v2.py"
    )
    executed = Path(nbfnet_attestation.__file__).resolve()
    if not deployed.is_file():
        _fail("NBFNet attestation source", f"missing deployed tool: {deployed}")
    deployed_sha256 = sha256_file(deployed)
    executed_sha256 = sha256_file(executed)
    if deployed_sha256 != NBFNET_ATTESTATION_TOOL_SHA256:
        _fail("NBFNet attestation source", "deployed tool is not the reviewed byte stream")
    if executed_sha256 != deployed_sha256:
        _fail(
            "NBFNet attestation source",
            "executed tool bytes differ from the reviewed deployment source",
        )
    if (
        not private_run_copy.is_file()
        or private_run_copy.is_symlink()
        or sha256_file(private_run_copy) != deployed_sha256
    ):
        _fail(
            "NBFNet attestation source",
            "private formal-run tool copy is missing, symbolic, or byte-different",
        )
    return executed_sha256


def _validate_current_run_config(path: Path, manifest: Mapping[str, Any]) -> str:
    """Bind promotion to the current canonical authorization, not a claimed hash."""
    path = Path(path)
    if not path.is_file():
        _fail("canonical run config", f"missing {path}")
    actual_sha256 = sha256_file(path)
    expected_sha256 = _hash(
        manifest.get("run_config_sha256"), "frozen manifest run-config hash"
    )
    if actual_sha256 != expected_sha256:
        _fail(
            "canonical run config",
            f"byte hash {actual_sha256} differs from frozen {expected_sha256}",
        )
    payload = _load_json(path, "canonical run config")
    required = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": manifest.get("run_id"),
        "benchmark_version": BENCHMARK_VERSION,
        "execution_status": FORMAL_EXECUTION_STATUS,
        "formal_authorization_value": FORMAL_EXECUTION_STATUS,
        "protocol": PROTOCOL,
        "chains": list(CHAINS),
        "tracks": list(TRACKS),
        "families": list(FAMILIES),
        "expected_selection_count": EXPECTED_EVALUATIONS,
        "selection_orchestration_job_count": EXPECTED_CHAIN_FAMILY_JOBS,
        "selection_fold": "fold2",
        "target_fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            _fail("canonical run config", f"invalid or stale {field}")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        _fail("canonical run config", "selection contract is missing")
    for field, expected in {
        "split_unit": "exporter_stage",
        "split_salt": "v2-history-0",
        "selection_seed": 0,
        "evaluation_seeds": list(SEEDS),
    }.items():
        if selection.get(field) != expected:
            _fail("canonical run config", f"invalid or stale selection.{field}")
    return actual_sha256


def _candidate_snapshot(path: Path, *, fold: str, role: str) -> dict[str, Any]:
    """Recompute one canonical candidate receipt from the current CSV bytes."""
    path = Path(path)
    if not path.is_file():
        _fail(role, f"canonical candidate is missing: {path}")
    expected_metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "aggregation": "calendar_mean",
        "early_window": "1998-2002" if fold == "fold2" else "2008-2012",
        "late_window": "2008-2012" if fold == "fold2" else "2018-2022",
        "temporal_role": "history" if fold == "fold2" else "target",
    }
    required_columns = {*KEYS, "y", "size", "lateval", *expected_metadata}
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    rows = 0
    positives = 0
    identities = hashlib.sha256()
    identity_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, float | int]] = []
    previous_key: tuple[str, str, str] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if not required_columns.issubset(columns):
                _fail(role, f"candidate columns are incomplete: {sorted(required_columns - columns)}")
            for row in reader:
                key = tuple(str(row[field]) for field in KEYS)
                if any(not value for value in key):
                    _fail(role, "candidate key contains an empty value")
                if previous_key is not None and key <= previous_key:
                    _fail(role, "candidate keys are duplicated or not in canonical order")
                previous_key = key
                identities.update("\x1f".join(key).encode("utf-8"))
                identities.update(b"\n")
                for field, expected in expected_metadata.items():
                    if row.get(field) != expected:
                        _fail(role, f"candidate {field} is not {expected!r}")
                label = row.get("y")
                if label not in {"0", "1"}:
                    _fail(role, "candidate y is not complete and binary")
                size = _finite_csv(row.get("size"), f"{role}.size")
                lateval = _finite_csv(row.get("lateval"), f"{role}.lateval")
                if size < 0.0 or lateval < 0.0:
                    _fail(role, "candidate size/lateval must be non-negative")
                identity_rows.append(dict(zip(KEYS, key)))
                label_rows.append(
                    {"y": int(label), "size": size, "lateval": lateval}
                )
                rows += 1
                positives += int(label)
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail(role, f"cannot scan canonical candidate: {exc}")
    if rows < 1:
        _fail(role, "canonical candidate is empty")
    digest = sha256_file(path)
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    if after != before:
        _fail(role, "canonical candidate changed while it was being verified")
    return {
        "sha256": digest,
        "rows": rows,
        "positive_lanes": positives,
        "identity_sha256": identities.hexdigest(),
        "identities": pd.DataFrame(identity_rows, columns=list(KEYS)),
        "labels": pd.DataFrame(label_rows, columns=["y", "size", "lateval"]),
    }


def _validate_candidate_receipt(
    claimed: Mapping[str, Any], observed: Mapping[str, Any], role: str
) -> None:
    for field in ("sha256", "rows", "positive_lanes"):
        if claimed.get(field) != observed.get(field):
            _fail(role, f"current candidate {field} differs from the private artifact")


def _basename(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(role, "expected a non-empty path string")
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _read_private_kv(path: Path, *, expected_keys: set[str], role: str) -> dict[str, str]:
    """Read one worker receipt without copying any private values to output."""

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(role, f"cannot read private receipt: {exc}")
    values: dict[str, str] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line or "=" not in line:
            _fail(role, f"line {line_number} is not non-empty key=value text")
        key, value = line.split("=", 1)
        if not key or not value or key in values:
            _fail(role, f"line {line_number} has an empty/duplicate field")
        values[key] = value
    if set(values) != expected_keys:
        _fail(role, "private receipt fields differ from the worker schema")
    return values


def _validate_main_job_claims(
    claims_root: Path,
    *,
    output_root: Path,
    manifest_sha256: str,
    sync_manifest_sha256: str,
) -> None:
    """Require exactly one successful permanent claim per chain/family job."""

    claims_root = Path(claims_root)
    if not claims_root.is_dir():
        _fail("main job claims", "canonical main_job_claims directory is missing")
    expected_names = {
        f"{chain}_{family}.lock" for chain in CHAINS for family in FAMILIES
    }
    entries = list(claims_root.iterdir())
    if {entry.name for entry in entries} != expected_names or any(
        not entry.is_dir() for entry in entries
    ):
        _fail(
            "main job claims",
            "must contain exactly the 12 canonical chain-family lock directories",
        )

    env_fields = {
        "worker_id",
        "claimed_at",
        "host",
        "physical_gpu",
        "chain",
        "family",
        "manifest",
        "manifest_sha256",
        "sync_manifest",
        "sync_manifest_sha256",
    }
    status_fields = {"finished_at", "exit_code", "run_log"}
    for chain in CHAINS:
        for family in FAMILIES:
            role = f"main job claim {chain}|{family}"
            claim = claims_root / f"{chain}_{family}.lock"
            children = list(claim.iterdir())
            if {child.name for child in children} != {"pid", "worker.env", "status"} or any(
                not child.is_file() for child in children
            ):
                _fail(role, "claim must contain exactly pid, worker.env, and status")
            try:
                pid = (claim / "pid").read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                _fail(role, f"cannot read pid: {exc}")
            if not pid.isascii() or not pid.isdigit() or int(pid) < 1:
                _fail(role, "pid is not a positive decimal process id")

            env = _read_private_kv(
                claim / "worker.env", expected_keys=env_fields, role=f"{role}.worker.env"
            )
            status = _read_private_kv(
                claim / "status", expected_keys=status_fields, role=f"{role}.status"
            )
            if env["chain"] != chain or env["family"] != family:
                _fail(role, "worker.env chain/family differs from the lock identity")
            if SAFE_WORKER_ID.fullmatch(env["worker_id"]) is None:
                _fail(role, "worker_id contains unsafe characters")
            if not env["host"].strip():
                _fail(role, "host is empty")
            if not env["physical_gpu"].isascii() or not env["physical_gpu"].isdigit():
                _fail(role, "physical_gpu is not a non-negative integer")
            claimed_at = _timestamp(env["claimed_at"], f"{role}.claimed_at")
            finished_at = _timestamp(status["finished_at"], f"{role}.finished_at")
            if finished_at < claimed_at:
                _fail(role, "claim finished before it was created")
            if _basename(env["manifest"], role) != "frozen_manifest.json":
                _fail(role, "worker.env does not name the canonical frozen manifest")
            if env["manifest_sha256"] != manifest_sha256:
                _fail(role, "worker.env manifest hash differs from the canonical freeze")
            if _basename(env["sync_manifest"], role) != "STEP3_SYNC_MANIFEST.sha256":
                _fail(role, "worker.env does not name the Step-3 sync manifest")
            if env["sync_manifest_sha256"] != sync_manifest_sha256:
                _fail(role, "worker.env Step-3 digest differs from the attested snapshot")
            if status["exit_code"] != "0":
                _fail(role, "worker status is not exit_code=0")
            expected_log_name = f"{env['worker_id']}.log"
            if _basename(status["run_log"], role) != expected_log_name:
                _fail(role, "status run_log does not match worker_id")
            if not (Path(output_root) / "logs" / "main" / expected_log_name).is_file():
                _fail(role, "private main run log is missing")


def _stable_object_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_nbfnet_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = nbfnet_attestation._strict_json_file(path, role)
    except nbfnet_attestation.SourceAttestationError as exc:
        _fail(role, str(exc))
    if not isinstance(value, dict):
        _fail(role, "top-level value must be an object")
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, float) and not math.isfinite(current):
            _fail(role, "non-finite JSON numbers are forbidden")
    return value


def _expected_nbfnet_run_identity(
    *, run_id: str, manifest_sha256: str, step3_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "frozen_manifest": {
            "role": "results_v2/gpu_rolling/frozen_manifest.json",
            "sha256": manifest_sha256,
        },
        "step3_sync_manifest": {
            "role": (
                "results_v2/gpu_rolling/runs/"
                f"{run_id}/STEP3_SYNC_MANIFEST.sha256"
            ),
            "sha256": step3_manifest_sha256,
        },
    }


def _validate_nbfnet_formal_gate_receipt(
    path: Path,
    *,
    host_role: str,
    run_id: str,
    marker_started_at: datetime,
    latest_receipt_observed_at: datetime,
    manifest_sha256: str,
    step3_manifest_sha256: str,
    receipt_hashes: Mapping[str, str],
    source_comparison_sha256: str,
    runtime_comparison_sha256: str,
    source_tree_sha256: str,
    runtime_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    role = f"NBFNet formal gate {host_role}"
    payload = _strict_nbfnet_json(path, role)
    expected_top = {
        "schema_version",
        "status",
        "run_id",
        "host_role",
        "created_at_utc",
        "verified_before_main_marker",
        "attestation_tool",
        "frozen_manifest",
        "step3_sync_manifest",
        "main_marker",
        "source_snapshot",
        "private_receipts",
        "comparisons",
        "runtime_artifacts",
        "gate_result",
    }
    if set(payload) != expected_top:
        _fail(role, "top-level schema is not exact")
    for field, expected in {
        "schema_version": NBFNET_FORMAL_GATE_SCHEMA,
        "status": "PASS",
        "run_id": run_id,
        "host_role": host_role,
        "verified_before_main_marker": True,
    }.items():
        if payload.get(field) != expected:
            _fail(role, f"invalid {field}")
    created = _timestamp(payload.get("created_at_utc"), f"{role}.created_at_utc")
    if created < latest_receipt_observed_at:
        _fail(role, "formal gate predates a private receipt that it claims to bind")
    if created > marker_started_at:
        _fail(role, "formal gate receipt was created after main started")

    expected_bindings = {
        "attestation_tool": {
            "role": "private/build_nbfnet_source_attestation_v2.py",
            "sha256": NBFNET_ATTESTATION_TOOL_SHA256,
        },
        "frozen_manifest": {
            "role": "results_v2/gpu_rolling/frozen_manifest.json",
            "sha256": manifest_sha256,
        },
        "step3_sync_manifest": {
            "role": (
                "results_v2/gpu_rolling/runs/"
                f"{run_id}/STEP3_SYNC_MANIFEST.sha256"
            ),
            "sha256": step3_manifest_sha256,
        },
        "main_marker": {
            "role": "results_v2/gpu_rolling/MAIN_EVALUATION_STARTED.json",
            "absent_at_verification": True,
        },
        "source_snapshot": {
            "role": "private/nbfnet_source_frozen",
            "tree_sha256": source_tree_sha256,
            "mode_read_only": True,
            "unattested_python_bytecode_absent": True,
        },
    }
    for field, expected in expected_bindings.items():
        if payload.get(field) != expected:
            _fail(role, f"invalid or stale {field} binding")

    receipt_role_prefix = "results_v2/gpu_rolling/nbfnet_attestation"
    expected_receipts = sorted(
        (
            {
                "role": (
                    f"{receipt_role_prefix}/"
                    f"{NBFNET_PRIVATE_FILENAMES[receipt_role]}"
                ),
                "sha256": receipt_hashes[receipt_role],
            }
            for receipt_role in NBFNET_RECEIPT_ROLES
        ),
        key=lambda row: row["role"].encode("utf-8"),
    )
    if payload.get("private_receipts") != expected_receipts:
        _fail(role, "does not bind the exact four private receipts")
    comparison_role_prefix = receipt_role_prefix
    expected_comparisons = {
        "source": {
            "role": f"{comparison_role_prefix}/{NBFNET_PRIVATE_SOURCE_COMPARISON}",
            "sha256": source_comparison_sha256,
            "status": "PASS",
        },
        "runtime": {
            "role": f"{comparison_role_prefix}/{NBFNET_PRIVATE_RUNTIME_COMPARISON}",
            "sha256": runtime_comparison_sha256,
            "status": "PASS",
        },
    }
    if payload.get("comparisons") != expected_comparisons:
        _fail(role, "does not bind both exact PASS comparison artifacts")
    expected_runtime = [
        {
            "role": str(row["role"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in runtime_artifacts
    ]
    if payload.get("runtime_artifacts") != expected_runtime:
        _fail(role, "runtime artifact identity differs from the cross-host comparison")
    expected_result = {
        "source_peer_count": 3,
        "runtime_peer_count": 1,
        "all_source_trees_match": True,
        "all_runtime_artifacts_match": True,
    }
    if payload.get("gate_result") != expected_result:
        _fail(role, "formal gate result is incomplete or non-PASS")
    return {
        "host_role": host_role,
        "private_gate_receipt_sha256": sha256_file(path),
        "verified_before_main_marker": True,
        "source_snapshot_mode_read_only": True,
        "unattested_python_bytecode_absent": True,
    }


def _validate_nbfnet_private_evidence(
    *,
    deployment_root: Path,
    gpu_root: Path,
    run_id: str,
    manifest_sha256: str,
    step3_manifest_sha256: str,
    main_marker: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all private evidence, returning only a path-free public binding."""

    tool_sha256 = _validate_nbfnet_attestation_source(deployment_root)
    private_root = Path(gpu_root) / NBFNET_PRIVATE_DIRECTORY
    if not private_root.is_dir() or private_root.is_symlink():
        _fail("NBFNet evidence", "private attestation directory is missing")
    expected_private_names = {
        *NBFNET_PRIVATE_FILENAMES.values(),
        NBFNET_PRIVATE_SOURCE_COMPARISON,
        NBFNET_PRIVATE_RUNTIME_COMPARISON,
        *NBFNET_FORMAL_GATE_FILENAMES.values(),
    }
    private_entries = list(private_root.iterdir())
    if {path.name for path in private_entries} != expected_private_names or any(
        not path.is_file() or path.is_symlink() for path in private_entries
    ):
        _fail("NBFNet evidence", "private directory must contain exactly eight evidence files")

    expected_identity = _expected_nbfnet_run_identity(
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        step3_manifest_sha256=step3_manifest_sha256,
    )
    receipt_paths = {
        role: private_root / NBFNET_PRIVATE_FILENAMES[role]
        for role in NBFNET_RECEIPT_ROLES
    }
    receipts: dict[str, dict[str, Any]] = {}
    receipt_hashes: dict[str, str] = {}
    public_receipts: list[dict[str, Any]] = []
    marker_started = _timestamp(
        main_marker.get("main_started_at_utc"), "main-start marker timestamp"
    )
    receipt_observation_times: list[datetime] = []
    for host_role in NBFNET_RECEIPT_ROLES:
        path = receipt_paths[host_role]
        receipt = _strict_nbfnet_json(path, f"NBFNet private receipt {host_role}")
        try:
            nbfnet_attestation._validate_private_shape(receipt)
        except nbfnet_attestation.SourceAttestationError as exc:
            _fail(f"NBFNet private receipt {host_role}", str(exc))
        if receipt.get("run_identity") != expected_identity:
            _fail(f"NBFNet private receipt {host_role}", "formal run identity differs")
        runtime = receipt.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("host_role") != host_role:
            _fail(f"NBFNet private receipt {host_role}", "host role differs from filename")
        receipt_sha256 = sha256_file(path)
        receipt_hashes[host_role] = receipt_sha256
        receipts[host_role] = receipt

        public_role = NBFNET_PUBLIC_PROJECTION_ROLES[host_role]
        public_path = Path(deployment_root) / public_role
        try:
            public = nbfnet_attestation.verify_public_projection(public_path, path)
        except nbfnet_attestation.SourceAttestationError as exc:
            _fail(f"NBFNet public projection {host_role}", str(exc))
        _privacy_audit(public, f"NBFNet public projection {host_role}")

        timing = receipt.get("selection_timing")
        if not isinstance(timing, Mapping):
            _fail(f"NBFNet private receipt {host_role}", "selection timing is missing")
        observed = _timestamp(receipt.get("observed_at_utc"), f"{host_role}.observed_at")
        receipt_observation_times.append(observed)
        if host_role.startswith("selection-"):
            started_raw = timing.get("selection_started_at_utc")
            if started_raw is None:
                _fail(host_role, "retrospective selection evidence lacks selection start")
            started = _timestamp(started_raw, f"{host_role}.selection_started_at")
            if observed < started:
                _fail(host_role, "selection receipt is not a retrospective observation")
            if timing.get("latest_source_mtime_not_after_selection_start") is not True:
                _fail(host_role, "latest source mtime is later than selection start")
            delta = _finite(
                timing.get("seconds_from_latest_source_mtime_to_selection_start"),
                f"{host_role}.mtime delta",
            )
            if delta < 0:
                _fail(host_role, "selection mtime delta is negative")
            latest_source_mtime = _timestamp(
                receipt.get("filesystem_timestamps", {}).get(
                    "latest_included_file_mtime_utc"
                ),
                f"{host_role}.latest source mtime",
            )
            _close(
                delta,
                (started - latest_source_mtime).total_seconds(),
                f"{host_role}.mtime delta",
            )
            interpretation = str(timing.get("interpretation", "")).lower()
            if (
                "retrospective consistency fact only" not in interpretation
                or "not proof" not in interpretation
            ):
                _fail(host_role, "selection evidence overstates the retrospective mtime fact")
            if runtime.get("artifacts") != []:
                _fail(host_role, "selection receipt must not imply retrospective runtime capture")
            evidence_class = "retrospective_selection_same_tree_and_mtime"
        else:
            resolved = str(receipt.get("source", {}).get("resolved_path", "")).replace("\\", "/")
            if not resolved.endswith("/private/nbfnet_source_frozen"):
                _fail(host_role, "main receipt is not for the dedicated frozen source path")
            if observed > marker_started:
                _fail(host_role, "main frozen-source receipt was observed after main started")
            artifacts = runtime.get("artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != 1:
                _fail(host_role, "main receipt must attest exactly one runtime artifact")
            artifact = artifacts[0]
            if (
                artifact.get("role") != "rspmm-extension"
                or artifact.get("matches_expected_sha256") is not True
                or artifact.get("expected_sha256") != artifact.get("sha256")
            ):
                _fail(host_role, "main rspmm runtime identity is not externally hash-bound")
            artifact_mtime = _timestamp(
                artifact.get("mtime_utc"), f"{host_role}.rspmm mtime"
            )
            if artifact_mtime > observed or artifact_mtime > marker_started:
                _fail(host_role, "main rspmm artifact is timestamped after its pre-main receipt")
            evidence_class = "pre_main_frozen_snapshot_observation"
        public_receipts.append(
            {
                "role": host_role,
                "private_receipt_sha256": receipt_sha256,
                "public_projection_artifact_role": public_role,
                "public_projection_sha256": sha256_file(public_path),
                "evidence_class": evidence_class,
            }
        )

    source_private_path = private_root / NBFNET_PRIVATE_SOURCE_COMPARISON
    try:
        source_comparison = nbfnet_attestation.compare_source_receipts(
            [receipt_paths[role] for role in NBFNET_RECEIPT_ROLES]
        )
    except nbfnet_attestation.SourceAttestationError as exc:
        _fail("NBFNet source comparison", str(exc))
    source_bytes = nbfnet_attestation.render_json(source_comparison)
    if source_private_path.read_bytes() != source_bytes:
        _fail("NBFNet source comparison", "private comparison is stale or non-deterministic")
    source_public_path = Path(deployment_root) / NBFNET_PUBLIC_SOURCE_COMPARISON_ROLE
    if source_public_path.read_bytes() != source_bytes:
        _fail("NBFNet source comparison", "public comparison differs from private evidence")
    if (
        source_comparison.get("status") != "PASS"
        or source_comparison.get("receipt_count") != 4
        or source_comparison.get("all_source_trees_match") is not True
    ):
        _fail("NBFNet source comparison", "four complete source trees do not match")

    runtime_private_path = private_root / NBFNET_PRIVATE_RUNTIME_COMPARISON
    try:
        runtime_comparison = nbfnet_attestation.compare_runtime_receipts(
            [receipt_paths[role] for role in NBFNET_MAIN_ROLES]
        )
    except nbfnet_attestation.SourceAttestationError as exc:
        _fail("NBFNet runtime comparison", str(exc))
    runtime_bytes = nbfnet_attestation.render_json(runtime_comparison)
    if runtime_private_path.read_bytes() != runtime_bytes:
        _fail("NBFNet runtime comparison", "private comparison is stale or non-deterministic")
    runtime_public_path = Path(deployment_root) / NBFNET_PUBLIC_RUNTIME_COMPARISON_ROLE
    if runtime_public_path.read_bytes() != runtime_bytes:
        _fail("NBFNet runtime comparison", "public comparison differs from private evidence")
    if (
        runtime_comparison.get("status") != "PASS"
        or runtime_comparison.get("host_count") != 2
        or runtime_comparison.get("host_roles") != list(NBFNET_MAIN_ROLES)
        or runtime_comparison.get("all_source_trees_match") is not True
        or runtime_comparison.get("all_runtime_artifacts_match") is not True
    ):
        _fail("NBFNet runtime comparison", "two main-host runtime identities do not match")
    runtime_artifacts = runtime_comparison.get("hosts", [])[0].get("runtime_artifacts", [])
    if not isinstance(runtime_artifacts, list) or len(runtime_artifacts) != 1:
        _fail("NBFNet runtime comparison", "expected exactly one shared runtime artifact")

    tree_receipt = source_comparison["receipts"][0]
    source_tree_sha256 = str(tree_receipt["tree_sha256"])
    source_comparison_sha256 = sha256_file(source_private_path)
    runtime_comparison_sha256 = sha256_file(runtime_private_path)
    formal_gates = [
        _validate_nbfnet_formal_gate_receipt(
            private_root / NBFNET_FORMAL_GATE_FILENAMES[host_role],
            host_role=host_role,
            run_id=run_id,
            marker_started_at=marker_started,
            latest_receipt_observed_at=max(receipt_observation_times),
            manifest_sha256=manifest_sha256,
            step3_manifest_sha256=step3_manifest_sha256,
            receipt_hashes=receipt_hashes,
            source_comparison_sha256=source_comparison_sha256,
            runtime_comparison_sha256=runtime_comparison_sha256,
            source_tree_sha256=source_tree_sha256,
            runtime_artifacts=runtime_artifacts,
        )
        for host_role in NBFNET_MAIN_ROLES
    ]

    binding = {
        "schema_version": NBFNET_EVIDENCE_SCHEMA,
        "attestation_tool": {
            "artifact_role": "tools/build_nbfnet_source_attestation.py",
            "sha256": tool_sha256,
        },
        "private_receipts": public_receipts,
        "source_identity": {
            "tree_sha256": source_tree_sha256,
            "inventory_sha256": tree_receipt["inventory_sha256"],
            "file_count": tree_receipt["file_count"],
            "total_bytes": tree_receipt["total_bytes"],
        },
        "runtime_identity": {
            "host_roles": list(NBFNET_MAIN_ROLES),
            "artifacts": runtime_artifacts,
            "all_source_trees_match": True,
            "all_runtime_artifacts_match": True,
        },
        "comparisons": {
            "source": {
                "private_artifact_sha256": source_comparison_sha256,
                "public_artifact_role": NBFNET_PUBLIC_SOURCE_COMPARISON_ROLE,
                "public_artifact_sha256": sha256_file(source_public_path),
                "receipt_count": 4,
                "all_source_trees_match": True,
            },
            "runtime": {
                "private_artifact_sha256": runtime_comparison_sha256,
                "public_artifact_role": NBFNET_PUBLIC_RUNTIME_COMPARISON_ROLE,
                "public_artifact_sha256": sha256_file(runtime_public_path),
                "host_count": 2,
                "all_runtime_artifacts_match": True,
            },
        },
        "formal_main_gates": formal_gates,
        "claim_boundary": {
            "selection": NBFNET_SELECTION_CLAIM,
            "selection_evidence_is_retrospective": True,
            "selection_latest_source_mtime_not_after_start": True,
            "selection_prehash_or_contemporaneous_freeze_proved": False,
            "main": NBFNET_MAIN_CLAIM,
            "main_read_only_pre_main_snapshot_verified": True,
            "main_unattested_python_bytecode_absent": True,
        },
    }
    _privacy_audit(binding, "NBFNet public summary binding")
    verify_nbfnet_public_binding(
        {"run_id": run_id, "nbfnet_source_evidence": binding},
        root=deployment_root,
    )
    return binding


def verify_nbfnet_public_binding(
    summary: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    """Verify the path-free NBFNet evidence surface in a public checkout."""

    role = "public NBFNet evidence"
    evidence_root = Path(root) / "chains" / "evidence"
    expected_public_names = {
        Path(public_role).name
        for public_role in (
            *NBFNET_PUBLIC_PROJECTION_ROLES.values(),
            NBFNET_PUBLIC_SOURCE_COMPARISON_ROLE,
            NBFNET_PUBLIC_RUNTIME_COMPARISON_ROLE,
        )
    }
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        _fail(role, "public evidence directory is missing or is a symbolic link")
    try:
        public_entries = [
            path
            for path in evidence_root.iterdir()
            if path.name.startswith("nbfnet_")
        ]
    except OSError as exc:
        _fail(role, f"cannot enumerate the public evidence directory: {exc}")
    if {path.name for path in public_entries} != expected_public_names or any(
        not path.is_file() or path.is_symlink() for path in public_entries
    ):
        _fail(role, "chains/evidence must contain exactly six NBFNet evidence files")
    binding = summary.get("nbfnet_source_evidence")
    if not isinstance(binding, dict) or set(binding) != {
        "schema_version",
        "attestation_tool",
        "private_receipts",
        "source_identity",
        "runtime_identity",
        "comparisons",
        "formal_main_gates",
        "claim_boundary",
    }:
        _fail(role, "summary binding schema is not exact")
    if binding.get("schema_version") != NBFNET_EVIDENCE_SCHEMA:
        _fail(role, "summary binding schema version changed")
    tool = binding.get("attestation_tool")
    expected_tool = {
        "artifact_role": "tools/build_nbfnet_source_attestation.py",
        "sha256": NBFNET_ATTESTATION_TOOL_SHA256,
    }
    if tool != expected_tool:
        _fail(role, "attestation tool binding changed")
    tool_path = Path(root) / expected_tool["artifact_role"]
    if (
        not tool_path.is_file()
        or tool_path.is_symlink()
        or sha256_file(tool_path) != NBFNET_ATTESTATION_TOOL_SHA256
    ):
        _fail(role, "reviewed public attestation tool bytes are missing or stale")

    claim_boundary = binding.get("claim_boundary")
    expected_claim_boundary = {
        "selection": NBFNET_SELECTION_CLAIM,
        "selection_evidence_is_retrospective": True,
        "selection_latest_source_mtime_not_after_start": True,
        "selection_prehash_or_contemporaneous_freeze_proved": False,
        "main": NBFNET_MAIN_CLAIM,
        "main_read_only_pre_main_snapshot_verified": True,
        "main_unattested_python_bytecode_absent": True,
    }
    if claim_boundary != expected_claim_boundary:
        _fail(role, "claim boundary is missing or overstated")
    source_identity = binding.get("source_identity")
    if not isinstance(source_identity, Mapping) or set(source_identity) != {
        "tree_sha256",
        "inventory_sha256",
        "file_count",
        "total_bytes",
    }:
        _fail(role, "source identity schema is not exact")
    _hash(source_identity.get("tree_sha256"), f"{role}.tree_sha256")
    _hash(source_identity.get("inventory_sha256"), f"{role}.inventory_sha256")
    if (
        isinstance(source_identity.get("file_count"), bool)
        or not isinstance(source_identity.get("file_count"), int)
        or source_identity["file_count"] < 1
        or isinstance(source_identity.get("total_bytes"), bool)
        or not isinstance(source_identity.get("total_bytes"), int)
        or source_identity["total_bytes"] < 1
    ):
        _fail(role, "source identity counts must be positive integers")

    receipt_rows = binding.get("private_receipts")
    if not isinstance(receipt_rows, list) or [
        row.get("role") for row in receipt_rows if isinstance(row, Mapping)
    ] != list(NBFNET_RECEIPT_ROLES):
        _fail(role, "summary must bind the exact four receipt roles in canonical order")
    receipt_hashes: dict[str, str] = {}
    public_run_identity: Mapping[str, Any] | None = None
    for row in receipt_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "private_receipt_sha256",
            "public_projection_artifact_role",
            "public_projection_sha256",
            "evidence_class",
        }:
            _fail(role, "receipt binding schema is not exact")
        host_role = str(row["role"])
        expected_public_role = NBFNET_PUBLIC_PROJECTION_ROLES[host_role]
        if row.get("public_projection_artifact_role") != expected_public_role:
            _fail(role, f"public projection role changed for {host_role}")
        private_sha256 = _hash(
            row.get("private_receipt_sha256"), f"{role}.{host_role}.private"
        )
        public_sha256 = _hash(
            row.get("public_projection_sha256"), f"{role}.{host_role}.public"
        )
        receipt_hashes[host_role] = private_sha256
        public_path = Path(root) / expected_public_role
        if not public_path.is_file() or sha256_file(public_path) != public_sha256:
            _fail(role, f"public projection bytes are missing or stale for {host_role}")
        public = _strict_nbfnet_json(public_path, f"{role}.{host_role}")
        try:
            nbfnet_attestation._privacy_audit(public)
        except nbfnet_attestation.SourceAttestationError as exc:
            _fail(f"{role}.{host_role}", str(exc))
        _privacy_audit(public, f"{role}.{host_role}")
        if set(public) != {
            "schema_version",
            "attestation_type",
            "status",
            "observed_at_utc",
            "run_identity",
            "source",
            "git",
            "runtime",
            "filesystem_timestamps",
            "selection_timing",
            "claim_boundary",
            "private_evidence",
        }:
            _fail(role, f"public projection schema is not exact for {host_role}")
        if (
            public.get("schema_version") != nbfnet_attestation.PUBLIC_SCHEMA
            or public.get("attestation_type")
            != "retrospective_supplemental_nbfnet_source_tree_evidence"
            or public.get("status") != "PASS"
        ):
            _fail(role, f"invalid public projection schema/status for {host_role}")
        _timestamp(public.get("observed_at_utc"), f"{role}.{host_role}.observed_at")
        run_identity = public.get("run_identity")
        if not isinstance(run_identity, Mapping) or set(run_identity) != {
            "run_id",
            "frozen_manifest",
            "step3_sync_manifest",
        }:
            _fail(role, f"public run identity schema changed for {host_role}")
        if run_identity.get("run_id") != summary.get("run_id"):
            _fail(role, f"public projection run_id differs for {host_role}")
        if public_run_identity is None:
            public_run_identity = run_identity
        elif run_identity != public_run_identity:
            _fail(role, "public projections bind different formal runs")
        for identity_role in ("frozen_manifest", "step3_sync_manifest"):
            record = run_identity.get(identity_role)
            if not isinstance(record, Mapping) or set(record) != {"role", "sha256"}:
                _fail(role, f"public {identity_role} binding schema changed")
            _hash(record.get("sha256"), f"{role}.{identity_role}")
        expected_identity_roles = {
            "frozen_manifest": "results_v2/gpu_rolling/frozen_manifest.json",
            "step3_sync_manifest": (
                "results_v2/gpu_rolling/runs/"
                f"{summary.get('run_id')}/STEP3_SYNC_MANIFEST.sha256"
            ),
        }
        for identity_role, expected_role in expected_identity_roles.items():
            if run_identity[identity_role].get("role") != expected_role:
                _fail(role, f"public {identity_role} artifact role changed")
        if summary.get("manifest_sha256") is not None and (
            run_identity["frozen_manifest"].get("sha256")
            != summary.get("manifest_sha256")
        ):
            _fail(role, "public NBFNet evidence binds a different frozen manifest")
        postfreeze_binding = summary.get("post_freeze_semantic_attestation")
        if isinstance(postfreeze_binding, Mapping) and (
            run_identity["step3_sync_manifest"].get("sha256")
            != postfreeze_binding.get("step3_manifest_sha256")
        ):
            _fail(role, "public NBFNet evidence binds a different Step-3 manifest")
        if public.get("private_evidence") != {
            "receipt_sha256": private_sha256,
            "retained_outside_public_bundle": True,
        }:
            _fail(role, f"public projection does not bind private receipt {host_role}")
        source = public.get("source")
        if not isinstance(source, Mapping) or set(source) != {
            "selector",
            "environment_binding_verified",
            "resolved_path_redacted",
            "inventory",
        }:
            _fail(role, f"public source schema changed for {host_role}")
        if (
            source.get("selector") != "NBFNET_PATH"
            or source.get("environment_binding_verified") is not True
            or source.get("resolved_path_redacted") is not True
        ):
            _fail(role, f"source path is not redacted for {host_role}")
        inventory = source.get("inventory")
        if not isinstance(inventory, Mapping):
            _fail(role, f"source inventory is missing for {host_role}")
        try:
            nbfnet_attestation._validate_inventory(inventory)
        except nbfnet_attestation.SourceAttestationError as exc:
            _fail(f"{role}.{host_role}.inventory", str(exc))
        for field in ("tree_sha256", "inventory_sha256", "file_count", "total_bytes"):
            if inventory.get(field) != source_identity.get(field):
                _fail(role, f"source identity differs in public projection {host_role}")
        git = public.get("git")
        if not isinstance(git, Mapping) or set(git) != {
            "repository_detected",
            "head",
            "head_author_timestamp_utc",
            "head_committer_timestamp_utc",
            "source_tracked_file_count",
            "tracked_dirty",
            "dirty_tracked_paths",
            "absence_semantics",
        }:
            _fail(role, f"public Git-evidence schema changed for {host_role}")
        runtime = public.get("runtime")
        if not isinstance(runtime, Mapping) or set(runtime) != {
            "host_role",
            "observed_hostname_redacted",
            "artifacts",
            "cross_host_match_semantics",
        }:
            _fail(role, f"public runtime schema changed for {host_role}")
        if (
            runtime.get("host_role") != host_role
            or runtime.get("observed_hostname_redacted") is not True
        ):
            _fail(role, f"runtime host/path redaction changed for {host_role}")
        runtime_artifacts = runtime.get("artifacts")
        if not isinstance(runtime_artifacts, list) or any(
            not isinstance(item, Mapping)
            or set(item)
            != {
                "role",
                "size_bytes",
                "sha256",
                "expected_sha256",
                "matches_expected_sha256",
                "mtime_utc",
                "resolved_path_redacted",
            }
            or item.get("resolved_path_redacted") is not True
            for item in runtime_artifacts
        ):
            _fail(role, f"runtime path is not redacted for {host_role}")
        for item in runtime_artifacts:
            _hash(item.get("sha256"), f"{role}.{host_role}.runtime sha256")
            _integer(
                item.get("size_bytes"),
                f"{role}.{host_role}.runtime size",
                minimum=1,
            )
            _timestamp(
                item.get("mtime_utc"), f"{role}.{host_role}.runtime mtime"
            )
        timestamps = public.get("filesystem_timestamps")
        if not isinstance(timestamps, Mapping) or set(timestamps) != {
            "earliest_included_file_mtime_utc",
            "latest_included_file_mtime_utc",
            "mtime_semantics",
        }:
            _fail(role, f"filesystem timestamp schema changed for {host_role}")
        _timestamp(
            timestamps.get("earliest_included_file_mtime_utc"),
            f"{role}.{host_role}.earliest mtime",
        )
        _timestamp(
            timestamps.get("latest_included_file_mtime_utc"),
            f"{role}.{host_role}.latest mtime",
        )
        timing = public.get("selection_timing")
        if not isinstance(timing, Mapping) or set(timing) != {
            "selection_started_at_utc",
            "latest_source_mtime_not_after_selection_start",
            "seconds_from_latest_source_mtime_to_selection_start",
            "interpretation",
        }:
            _fail(role, f"selection-timing schema changed for {host_role}")
        public_claim = public.get("claim_boundary")
        if not isinstance(public_claim, Mapping) or set(public_claim) != {
            "supported",
            "not_supported",
        }:
            _fail(role, f"public claim-boundary schema changed for {host_role}")
        if public_claim != {
            "supported": NBFNET_RECEIPT_SUPPORTED_CLAIM,
            "not_supported": NBFNET_RECEIPT_UNSUPPORTED_CLAIM,
        }:
            _fail(role, f"public retrospective claim boundary changed for {host_role}")
        if host_role.startswith("selection-"):
            if row.get("evidence_class") != "retrospective_selection_same_tree_and_mtime":
                _fail(role, f"selection evidence class changed for {host_role}")
            interpretation = str(timing.get("interpretation", "")).lower()
            if (
                timing.get("latest_source_mtime_not_after_selection_start") is not True
                or "retrospective consistency fact only" not in interpretation
                or "not proof" not in interpretation
            ):
                _fail(
                    role,
                    f"selection projection loses its retrospective boundary for {host_role}",
                )
            if runtime_artifacts:
                _fail(
                    role,
                    f"selection projection has unexpected runtime artifacts for {host_role}",
                )
            _timestamp(
                timing.get("selection_started_at_utc"),
                f"{role}.{host_role}.selection start",
            )
            public_delta = _finite(
                timing.get("seconds_from_latest_source_mtime_to_selection_start"),
                f"{role}.{host_role}.mtime delta",
            )
            if public_delta < 0:
                _fail(role, f"selection mtime delta is negative for {host_role}")
            _close(
                public_delta,
                (
                    _timestamp(
                        timing.get("selection_started_at_utc"),
                        f"{role}.{host_role}.selection start",
                    )
                    - _timestamp(
                        timestamps.get("latest_included_file_mtime_utc"),
                        f"{role}.{host_role}.latest mtime",
                    )
                ).total_seconds(),
                f"{role}.{host_role}.mtime delta",
            )
        else:
            if row.get("evidence_class") != "pre_main_frozen_snapshot_observation":
                _fail(role, f"main evidence class changed for {host_role}")
            if (
                len(runtime_artifacts) != 1
                or runtime_artifacts[0].get("role") != "rspmm-extension"
                or runtime_artifacts[0].get("matches_expected_sha256") is not True
                or runtime_artifacts[0].get("expected_sha256")
                != runtime_artifacts[0].get("sha256")
            ):
                _fail(role, f"main projection lacks its expected rspmm identity for {host_role}")

    if len(set(receipt_hashes.values())) != len(NBFNET_RECEIPT_ROLES):
        _fail(role, "four receipt roles do not bind four distinct private receipts")

    comparisons = binding.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {"source", "runtime"}:
        _fail(role, "comparison binding schema is not exact")
    source_record = comparisons["source"]
    runtime_record = comparisons["runtime"]
    expected_source_fields = {
        "private_artifact_sha256",
        "public_artifact_role",
        "public_artifact_sha256",
        "receipt_count",
        "all_source_trees_match",
    }
    expected_runtime_fields = {
        "private_artifact_sha256",
        "public_artifact_role",
        "public_artifact_sha256",
        "host_count",
        "all_runtime_artifacts_match",
    }
    if (
        not isinstance(source_record, Mapping)
        or set(source_record) != expected_source_fields
    ):
        _fail(role, "source comparison binding schema is not exact")
    if (
        not isinstance(runtime_record, Mapping)
        or set(runtime_record) != expected_runtime_fields
    ):
        _fail(role, "runtime comparison binding schema is not exact")
    for item, public_role, count_field, expected_count in (
        (source_record, NBFNET_PUBLIC_SOURCE_COMPARISON_ROLE, "receipt_count", 4),
        (runtime_record, NBFNET_PUBLIC_RUNTIME_COMPARISON_ROLE, "host_count", 2),
    ):
        _hash(item.get("private_artifact_sha256"), f"{role}.private comparison")
        if (
            item.get("public_artifact_role") != public_role
            or item.get(count_field) != expected_count
        ):
            _fail(role, "comparison role/count changed")
        public_sha256 = _hash(
            item.get("public_artifact_sha256"), f"{role}.public comparison"
        )
        path = Path(root) / public_role
        if not path.is_file() or sha256_file(path) != public_sha256:
            _fail(role, f"public comparison bytes are missing or stale: {public_role}")
        if item["private_artifact_sha256"] != public_sha256:
            _fail(role, "private/public comparison byte hashes differ")

    source_public = _strict_nbfnet_json(
        Path(root) / NBFNET_PUBLIC_SOURCE_COMPARISON_ROLE, "public source comparison"
    )
    runtime_public = _strict_nbfnet_json(
        Path(root) / NBFNET_PUBLIC_RUNTIME_COMPARISON_ROLE, "public runtime comparison"
    )
    for payload, payload_role in ((source_public, "source"), (runtime_public, "runtime")):
        try:
            nbfnet_attestation._privacy_audit(payload)
        except nbfnet_attestation.SourceAttestationError as exc:
            _fail(f"public {payload_role} comparison", str(exc))
        _privacy_audit(payload, f"public {payload_role} comparison")
    if set(source_public) != {
        "schema_version",
        "status",
        "run_identity",
        "receipt_count",
        "all_source_trees_match",
        "receipts",
        "claim_boundary",
    }:
        _fail(role, "public source-comparison schema is not exact")
    if set(runtime_public) != {
        "schema_version",
        "status",
        "run_identity",
        "host_count",
        "host_roles",
        "runtime_artifact_roles",
        "all_source_trees_match",
        "all_runtime_artifacts_match",
        "hosts",
        "claim_boundary",
    }:
        _fail(role, "public runtime-comparison schema is not exact")
    for comparison, comparison_role in (
        (source_public, "source"),
        (runtime_public, "runtime"),
    ):
        boundary = comparison.get("claim_boundary")
        if not isinstance(boundary, Mapping) or set(boundary) != {
            "supported",
            "not_supported",
        }:
            _fail(role, f"public {comparison_role} claim-boundary schema changed")
    if source_public.get("claim_boundary") != NBFNET_SOURCE_COMPARISON_CLAIMS:
        _fail(role, "public source-comparison claim boundary changed")
    if runtime_public.get("claim_boundary") != NBFNET_RUNTIME_COMPARISON_CLAIMS:
        _fail(role, "public runtime-comparison claim boundary changed")
    if (
        source_public.get("run_identity") != public_run_identity
        or runtime_public.get("run_identity") != public_run_identity
    ):
        _fail(role, "public comparisons bind a different formal run")
    if (
        source_public.get("schema_version") != nbfnet_attestation.SOURCE_COMPARISON_SCHEMA
        or source_public.get("status") != "PASS"
        or source_public.get("receipt_count") != 4
        or source_public.get("all_source_trees_match") is not True
        or source_record.get("all_source_trees_match") is not True
    ):
        _fail(role, "public four-tree source comparison is not PASS")
    source_rows = source_public.get("receipts")
    expected_source_order = sorted(NBFNET_RECEIPT_ROLES, key=lambda value: value.encode("utf-8"))
    if (
        not isinstance(source_rows, list)
        or len(source_rows) != 4
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "receipt_role",
                "private_receipt_sha256",
                "tree_sha256",
                "inventory_sha256",
                "file_count",
                "total_bytes",
            }
            for row in source_rows
        )
        or [row.get("receipt_role") for row in source_rows] != expected_source_order
        or {
        str(row.get("receipt_role")): str(row.get("private_receipt_sha256"))
        for row in source_rows
        }
        != receipt_hashes
    ):
        _fail(role, "public source comparison does not bind the four private receipts")
    if any(
        row.get("tree_sha256") != source_identity["tree_sha256"]
        or row.get("inventory_sha256") != source_identity["inventory_sha256"]
        or row.get("file_count") != source_identity["file_count"]
        or row.get("total_bytes") != source_identity["total_bytes"]
        for row in source_rows
    ):
        _fail(role, "public source comparison identities differ")

    runtime_identity = binding.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or set(runtime_identity) != {
        "host_roles",
        "artifacts",
        "all_source_trees_match",
        "all_runtime_artifacts_match",
    }:
        _fail(role, "runtime identity binding schema is not exact")
    if (
        runtime_identity.get("host_roles") != list(NBFNET_MAIN_ROLES)
        or runtime_identity.get("all_source_trees_match") is not True
        or runtime_identity.get("all_runtime_artifacts_match") is not True
        or runtime_record.get("all_runtime_artifacts_match") is not True
        or runtime_public.get("schema_version")
        != nbfnet_attestation.RUNTIME_COMPARISON_SCHEMA
        or runtime_public.get("status") != "PASS"
        or runtime_public.get("host_count") != 2
        or runtime_public.get("host_roles") != list(NBFNET_MAIN_ROLES)
        or runtime_public.get("runtime_artifact_roles") != ["rspmm-extension"]
        or runtime_public.get("all_source_trees_match") is not True
        or runtime_public.get("all_runtime_artifacts_match") is not True
    ):
        _fail(role, "public two-host runtime comparison is not PASS")
    artifacts = runtime_identity.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 1
        or not isinstance(artifacts[0], Mapping)
        or set(artifacts[0]) != {"role", "sha256", "size_bytes"}
        or artifacts[0].get("role") != "rspmm-extension"
    ):
        _fail(role, "shared runtime identity is not exactly rspmm-extension")
    _hash(artifacts[0].get("sha256"), f"{role}.rspmm sha256")
    _integer(artifacts[0].get("size_bytes"), f"{role}.rspmm bytes", minimum=1)
    runtime_hosts = runtime_public.get("hosts")
    if (
        not isinstance(runtime_hosts, list)
        or len(runtime_hosts) != 2
        or any(
            not isinstance(host, Mapping)
            or set(host)
            != {
                "host_role",
                "private_receipt_sha256",
                "source_tree_sha256",
                "runtime_artifacts",
            }
            for host in runtime_hosts
        )
        or [host.get("host_role") for host in runtime_hosts] != list(NBFNET_MAIN_ROLES)
        or any(
            host.get("private_receipt_sha256") != receipt_hashes[host["host_role"]]
            or host.get("source_tree_sha256") != source_identity["tree_sha256"]
            or host.get("runtime_artifacts") != artifacts
            for host in runtime_hosts
        )
    ):
        _fail(role, "public host runtime identities differ from the summary binding")

    gates = binding.get("formal_main_gates")
    if not isinstance(gates, list) or [
        gate.get("host_role") for gate in gates if isinstance(gate, Mapping)
    ] != list(NBFNET_MAIN_ROLES):
        _fail(role, "summary does not bind exactly two main formal gates")
    for gate in gates:
        if set(gate) != {
            "host_role",
            "private_gate_receipt_sha256",
            "verified_before_main_marker",
            "source_snapshot_mode_read_only",
            "unattested_python_bytecode_absent",
        }:
            _fail(role, "formal gate public binding schema is not exact")
        _hash(gate.get("private_gate_receipt_sha256"), f"{role}.formal gate sha256")
        if any(
            gate.get(field) is not True
            for field in (
                "verified_before_main_marker",
                "source_snapshot_mode_read_only",
                "unattested_python_bytecode_absent",
            )
        ):
            _fail(role, "formal main gate public booleans are not all true")
    if len({gate["private_gate_receipt_sha256"] for gate in gates}) != 2:
        _fail(role, "two main roles do not bind two distinct formal gate receipts")
    _privacy_audit(binding, role)
    return dict(binding)


def _validate_selected_hyperparameters(family: str, selected: Mapping[str, Any], role: str) -> None:
    model = selected.get("model")
    expected_models = KGE_MODELS if family == "kge" else ("NBFNet",)
    if model not in expected_models:
        _fail(role, f"selected model {model!r} is outside the formal {family} family")
    hp = selected.get("hyperparameters")
    if not isinstance(hp, dict) or _canonical_key(hp) not in {
        _canonical_key(item) for item in FORMAL_GRIDS[str(model)]
    }:
        _fail(role, "selected hyperparameters are outside the preregistered formal grid/budget")
    _finite(selected.get("history_holdout_selection_metric_mean"), f"{role}.history metric")


def _validate_selection_for_reporting(selection: dict[str, Any], family: str, track: str, role: str) -> None:
    design = selection.get("selection_design")
    if not isinstance(design, dict):
        _fail(role, "missing selection_design")
    required_design = {
        "orchestration": "chain_multitask_shared_score_grid",
        "hp_partition": "fold2 exporter_stage dev",
        "model_partition": "fold2 exporter_stage holdout",
        "split_unit": "exporter_stage",
        "split_salt": "v2-history-0",
        "primary_metric": SELECTION_METRICS[track],
        "selection_seed": 0,
        "evaluation_seeds": list(SEEDS),
    }
    for field, expected in required_design.items():
        if design.get(field) != expected:
            _fail(role, f"selection_design.{field} is not the frozen formal value")
    if selection.get("representation_policy") != (
        "refit selected label-free model from scratch on main early graph per seed"
    ):
        _fail(role, "invalid representation policy")
    if selection.get("raw_score_policy") != "one column per seed; no cross-seed raw-score average":
        _fail(role, "invalid raw-score policy")

    selected = selection.get("selected")
    if not isinstance(selected, dict):
        _fail(role, "missing selected configuration")
    _validate_selected_hyperparameters(family, selected, f"{role}.selected")

    expected_models = KGE_MODELS if family == "kge" else ("NBFNet",)
    rows = selection.get("models")
    if not isinstance(rows, list) or [
        row.get("model") for row in rows if isinstance(row, dict)
    ] != list(expected_models):
        _fail(role, "models are not the complete preregistered family in canonical order")
    for row in rows:
        model = row["model"]
        row_role = f"{role}.models[{model}]"
        if row.get("status") != "complete":
            _fail(row_role, "formal grid is not complete")
        hp = row.get("selected_hyperparameters")
        if not isinstance(hp, dict) or _canonical_key(hp) not in {
            _canonical_key(item) for item in FORMAL_GRIDS[model]
        }:
            _fail(row_role, "per-model selected hyperparameters are outside the formal grid")
        grid = row.get("grid")
        if not isinstance(grid, list) or len(grid) != len(FORMAL_GRIDS[model]):
            _fail(row_role, "grid does not have the preregistered size")
        seen_grid = []
        for index, trial in enumerate(grid):
            if not isinstance(trial, dict) or trial.get("status") != "complete":
                _fail(row_role, f"grid trial {index} is not complete")
            trial_hp = trial.get("hyperparameters")
            if not isinstance(trial_hp, dict):
                _fail(row_role, f"grid trial {index} has no hyperparameters")
            seen_grid.append(_canonical_key(trial_hp))
            _finite(trial.get("history_dev_selection_metric"), f"{row_role}.grid[{index}]")
            _hash(trial.get("score_cache_key"), f"{row_role}.grid[{index}].score_cache_key")
            if not isinstance(trial.get("score_cache_hit"), bool):
                _fail(row_role, f"grid trial {index} has no cache attestation")
        if set(seen_grid) != {_canonical_key(item) for item in FORMAL_GRIDS[model]}:
            _fail(row_role, "grid differs from the preregistered configurations")
        per_seed = row.get("per_seed")
        if not isinstance(per_seed, list) or [
            item.get("seed") for item in per_seed if isinstance(item, dict)
        ] != list(SEEDS):
            _fail(row_role, "history holdout does not contain exactly the five frozen seeds")
        values = []
        for item in per_seed:
            if item.get("status") != "complete":
                _fail(row_role, "history holdout seed is incomplete")
            values.append(_finite(item.get("history_holdout_selection_metric"), row_role))
            _hash(item.get("score_cache_key"), f"{row_role}.per_seed.score_cache_key")
            if not isinstance(item.get("score_cache_hit"), bool):
                _fail(row_role, "history holdout seed has no cache attestation")
        _close(row.get("history_holdout_selection_metric_mean"), _mean(values), f"{row_role}.mean")
        _close(row.get("history_holdout_selection_metric_std"), _std(values), f"{row_role}.std")

    winner = max(rows, key=lambda item: item["history_holdout_selection_metric_mean"])
    if selected.get("model") != winner.get("model"):
        _fail(role, "frozen winner is not the historical-holdout winner")
    if selected.get("hyperparameters") != winner.get("selected_hyperparameters"):
        _fail(role, "frozen winner hyperparameters differ from the historical choice")
    _close(
        selected.get("history_holdout_selection_metric_mean"),
        float(winner["history_holdout_selection_metric_mean"]),
        f"{role}.selected.history metric",
    )


def _validate_main_marker(root: Path, run_id: str, manifest_sha256: str) -> dict[str, Any]:
    marker = _load_json(root / "MAIN_EVALUATION_STARTED.json", "main-start marker")
    expected = {
        "schema_version": MAIN_START_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "policy": "freeze and selections are immutable; main outputs never overwrite",
    }
    for field, value in expected.items():
        if marker.get(field) != value:
            _fail("main-start marker", f"invalid {field}")
    _timestamp(marker.get("main_started_at_utc"), "main-start marker timestamp")
    return marker


def _validate_metric_values(track: str, per_seed: Any, summary: Any, role: str) -> dict[str, Any]:
    if not isinstance(per_seed, list) or [row.get("seed") for row in per_seed if isinstance(row, dict)] != list(SEEDS):
        _fail(role, "per_seed must contain the five frozen seeds in order")
    expected_keys = TRACK_METRICS[track]
    cleaned = []
    for row in per_seed:
        if set(row) != {"seed", *expected_keys}:
            _fail(role, "per-seed metric keys do not match the evaluate-chain schema")
        clean = {"seed": int(row["seed"])}
        for metric in sorted(expected_keys):
            if metric in COUNT_METRICS:
                clean[metric] = _integer(row[metric], f"{role}.{metric}")
            else:
                clean[metric] = _probability(row[metric], f"{role}.{metric}")
        cleaned.append(clean)
    if not isinstance(summary, dict) or set(summary) != expected_keys:
        _fail(role, "summary keys do not match per-seed metrics")
    for metric in sorted(expected_keys):
        item = summary.get(metric)
        if not isinstance(item, dict) or set(item) != {"mean", "std", "n"}:
            _fail(role, f"summary.{metric} has an invalid schema")
        values = [float(row[metric]) for row in cleaned]
        if item.get("n") != len(SEEDS):
            _fail(role, f"summary.{metric}.n is not five")
        _close(item.get("mean"), _mean(values), f"{role}.summary.{metric}.mean")
        _close(item.get("std"), _std(values), f"{role}.summary.{metric}.std")
    return {"per_seed": cleaned, "summary": summary}


def _validate_bootstrap(track: str, value: Any, role: str) -> list[dict[str, Any]]:
    rng_seeds = [20260712 + seed for seed in SEEDS]
    if not isinstance(value, list) or [row.get("seed") for row in value if isinstance(row, dict)] != rng_seeds:
        _fail(role, "cluster bootstrap must contain the five frozen seeds")
    cluster_unit, metric = BOOTSTRAP_SPECS[track]
    cleaned = []
    for task_seed, row in zip(SEEDS, value):
        expected = {
            "seed": 20260712 + task_seed,
            "cluster_unit": cluster_unit,
            "metric": metric,
            "iterations": 500,
        }
        for field, wanted in expected.items():
            if row.get(field) != wanted:
                _fail(role, f"invalid bootstrap {field}")
        # The runner stores the RNG seed in the same `seed` key after merging;
        # because the cluster-bootstrap mapping is merged last.  Record both
        # identities explicitly in the sanitized output.
        lower = _probability(row.get("lower_95"), f"{role}.lower_95")
        upper = _probability(row.get("upper_95"), f"{role}.upper_95")
        if lower > upper:
            _fail(role, "bootstrap interval is reversed")
        cleaned.append(
            {
                "task_seed": task_seed,
                "bootstrap_rng_seed": int(row["seed"]),
                "cluster_unit": cluster_unit,
                "metric": metric,
                "iterations": 500,
                "lower_95": lower,
                "upper_95": upper,
            }
        )
    return cleaned


def _validate_scores(
    path: Path,
    *,
    selected_model: str,
    selection_sha256: str,
    rows: int,
    role: str,
) -> tuple[str, str, dict[int, list[float]]]:
    expected_header = [
        "i_iso",
        "j_iso",
        "stage",
        *(f"score_{selected_model}_s{seed}" for seed in SEEDS),
        "selection_sha256",
        "protocol",
    ]
    scores_by_seed: dict[int, list[float]] = {seed: [] for seed in SEEDS}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_header:
                _fail(role, "score CSV header does not match the frozen model/seeds")
            count = 0
            previous = None
            seen = set()
            identities = hashlib.sha256()
            for row in reader:
                count += 1
                key = (row["i_iso"], row["j_iso"], row["stage"])
                if not all(key) or key in seen or (previous is not None and key < previous):
                    _fail(role, "score identities are empty, duplicated, or non-canonical")
                seen.add(key)
                previous = key
                identities.update("\x1f".join(key).encode("utf-8"))
                identities.update(b"\n")
                for seed in SEEDS:
                    scores_by_seed[seed].append(
                        _finite_csv(
                            row[f"score_{selected_model}_s{seed}"],
                            f"{role}.score_s{seed}",
                        )
                    )
                if row["selection_sha256"] != selection_sha256 or row["protocol"] != PROTOCOL:
                    _fail(role, "score rows are not bound to the frozen selection/protocol")
    except OSError as exc:
        _fail(role, f"cannot read score CSV: {exc}")
    if count != rows:
        _fail(role, f"score row count {count} != target candidate rows {rows}")
    return sha256_file(path), identities.hexdigest(), scores_by_seed


def _validate_main_recomputation(
    *,
    track: str,
    candidate: Mapping[str, Any],
    scores_by_seed: Mapping[int, Sequence[float]],
    metric_values: Mapping[str, Any],
    bootstrap: Sequence[Mapping[str, Any]],
    role: str,
) -> None:
    """Recompute every main metric and interval from canonical inputs."""

    identities = candidate.get("identities")
    labels = candidate.get("labels")
    if not isinstance(identities, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        _fail(role, "canonical candidate frames are unavailable for recomputation")
    recorded_runs = metric_values.get("per_seed")
    recorded_summary = metric_values.get("summary")
    if not isinstance(recorded_runs, list) or not isinstance(recorded_summary, Mapping):
        _fail(role, "recorded metric payload is unavailable for recomputation")
    if len(bootstrap) != len(SEEDS):
        _fail(role, "recorded bootstrap payload is unavailable for recomputation")

    recomputed_runs: list[dict[str, float | int]] = []
    for index, seed in enumerate(SEEDS):
        score = scores_by_seed.get(seed)
        if score is None or len(score) != len(identities):
            _fail(role, f"seed {seed} score vector does not cover the canonical cohort")
        recomputed = gpu_runner._ranking_metrics(track, identities, labels, score)
        if set(recomputed) != TRACK_METRICS[track]:
            _fail(role, f"seed {seed} shared metric implementation returned a stale schema")
        recorded = recorded_runs[index]
        if recorded.get("seed") != seed:
            _fail(role, f"seed {seed} is not in canonical recorded order")
        for metric in sorted(TRACK_METRICS[track]):
            expected = recomputed[metric]
            if metric in COUNT_METRICS:
                if recorded.get(metric) != int(expected):
                    _fail(role, f"seed {seed} {metric} differs from mechanical recomputation")
            else:
                _close(recorded.get(metric), float(expected), f"{role}.seed{seed}.{metric}")

        bootstrap_seed = 20260712 + seed
        recomputed_bootstrap = gpu_runner._cluster_bootstrap(
            track,
            identities,
            labels,
            score,
            iters=500,
            seed=bootstrap_seed,
        )
        recorded_bootstrap = bootstrap[index]
        expected_bootstrap_fields = {
            "task_seed": seed,
            "bootstrap_rng_seed": bootstrap_seed,
            "cluster_unit": recomputed_bootstrap["cluster_unit"],
            "metric": recomputed_bootstrap["metric"],
            "iterations": recomputed_bootstrap["iterations"],
        }
        for field, expected in expected_bootstrap_fields.items():
            if recorded_bootstrap.get(field) != expected:
                _fail(role, f"seed {seed} bootstrap {field} differs from recomputation")
        _close(
            recorded_bootstrap.get("lower_95"),
            float(recomputed_bootstrap["lower_95"]),
            f"{role}.seed{seed}.bootstrap.lower_95",
        )
        _close(
            recorded_bootstrap.get("upper_95"),
            float(recomputed_bootstrap["upper_95"]),
            f"{role}.seed{seed}.bootstrap.upper_95",
        )
        recomputed_runs.append(dict(recomputed))

    recomputed_summary = gpu_runner._summarize_runs(recomputed_runs)
    if set(recomputed_summary) != TRACK_METRICS[track]:
        _fail(role, "shared summary implementation returned a stale metric schema")
    for metric in sorted(TRACK_METRICS[track]):
        expected = recomputed_summary[metric]
        recorded = recorded_summary.get(metric)
        if not isinstance(recorded, Mapping):
            _fail(role, f"summary.{metric} is unavailable for recomputation")
        if recorded.get("n") != expected["n"]:
            _fail(role, f"summary.{metric}.n differs from mechanical recomputation")
        _close(
            recorded.get("mean"),
            float(expected["mean"]),
            f"{role}.summary.{metric}.mean",
        )
        _close(
            recorded.get("std"),
            float(expected["std"]),
            f"{role}.summary.{metric}.std",
        )


def _validate_protocol_timestamps(
    value: Any,
    selections: Mapping[str, dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(role, "missing protocol_timestamps")
    if value.get("ordering_attestation") != (
        "all unique representation scores precede the target-label read"
    ):
        _fail(role, "invalid label-blind ordering attestation")
    gate = _timestamp(value.get("full_freeze_gate_verified_at_utc"), f"{role}.gate")
    label_read = _timestamp(value.get("main_target_labels_read_at_utc"), f"{role}.label_read")
    events = value.get("representations_finished")
    if not isinstance(events, list):
        _fail(role, "representations_finished must be a list")
    expected_events = []
    seen_configs = set()
    for track in TRACKS:
        selected = selections[track]["selected"]
        config_key = (selected["model"], _canonical_key(selected["hyperparameters"]))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        for seed in SEEDS:
            expected_events.append((selected["model"], selected["hyperparameters"], seed))
    if len(events) != len(expected_events):
        _fail(role, "representation event count does not match unique frozen configs x seeds")
    cleaned = []
    for index, (event, expected) in enumerate(zip(events, expected_events)):
        if not isinstance(event, dict):
            _fail(role, f"event {index} is not an object")
        model, hp, seed = expected
        if event.get("model") != model or event.get("hyperparameters") != hp or event.get("seed") != seed:
            _fail(role, f"event {index} is not the preregistered config/seed order")
        _hash(event.get("score_cache_key"), f"{role}.event[{index}].score_cache_key")
        if not isinstance(event.get("score_cache_hit"), bool):
            _fail(role, f"event {index} has no cache attestation")
        finished = _timestamp(event.get("finished_at_utc"), f"{role}.event[{index}]")
        if finished < gate or finished > label_read:
            _fail(role, "representation finish does not precede the main-label read")
        cleaned.append(
            {
                "model": model,
                "hyperparameters": hp,
                "seed": seed,
                "score_cache_key": event["score_cache_key"],
                "score_cache_hit": event["score_cache_hit"],
                "finished_at_utc": event["finished_at_utc"],
            }
        )
    return {
        "full_freeze_gate_verified_at_utc": value["full_freeze_gate_verified_at_utc"],
        "representations_finished": cleaned,
        "main_target_labels_read_at_utc": value["main_target_labels_read_at_utc"],
        "ordering_attestation": value["ordering_attestation"],
    }


def _artifact_names(root: Path, subdir: str, prefix: str, suffix: str) -> set[str]:
    directory = root / subdir
    if not directory.is_dir():
        _fail(subdir, "canonical artifact directory is missing")
    return {path.name for path in directory.glob(f"{prefix}*{suffix}") if path.is_file()}


def _privacy_audit(value: Any, role: str = "public summary") -> None:
    """Reject any accidentally copied absolute/private operational metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"host", "hostname", "user", "username", "pid", "worker_id"}:
                _fail(role, f"private operational key leaked: {key}")
            _privacy_audit(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _privacy_audit(child, f"{role}[{index}]")
    elif isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _fail(role, "absolute path leaked into public output")
        for label, pattern in PRIVATE_TEXT_PATTERNS:
            if pattern.search(value):
                _fail(role, f"private host/user token leaked: {label}")


def build_summary(
    root: Path = CANONICAL_ROOT,
    *,
    claims_root: Path | None = None,
) -> dict[str, Any]:
    """Build a sanitized post-main summary; never write on validation failure."""
    root = Path(root).resolve()
    run_config_path, candidate_root = _deployment_sources(root)
    if (root / "PILOT_INVALIDATED.json").exists():
        _fail("run namespace", "pilot is explicitly invalidated and cannot be promoted")
    manifest_path = root / "frozen_manifest.json"
    try:
        manifest, indexed = verify_freeze_manifest(manifest_path)
    except ProtocolError as exc:
        raise ResultValidationError(f"frozen manifest: {exc}") from exc
    expected_keys = {
        selection_key(chain, track, family)
        for chain in CHAINS
        for track in TRACKS
        for family in FAMILIES
    }
    if set(indexed) != expected_keys or len(indexed) != EXPECTED_EVALUATIONS:
        _fail("frozen manifest", "must contain exactly the canonical 36 selections")
    run_id = manifest["run_id"]
    if SAFE_RUN_ID.fullmatch(run_id) is None:
        _fail("frozen manifest", "run_id is not safe for public provenance")
    deployment_root = run_config_path.parents[1]
    postfreeze_binding = _verify_postfreeze_attestation(deployment_root, run_id)
    metric_source_sha256 = _validate_metric_source(deployment_root)
    run_config_sha256 = _validate_current_run_config(run_config_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    main_marker = _validate_main_marker(root, run_id, manifest_sha256)
    nbfnet_source_evidence = _validate_nbfnet_private_evidence(
        deployment_root=deployment_root,
        gpu_root=root,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        step3_manifest_sha256=postfreeze_binding["step3_manifest_sha256"],
        main_marker=main_marker,
    )
    _validate_main_job_claims(
        Path(claims_root) if claims_root is not None else root / "main_job_claims",
        output_root=root,
        manifest_sha256=manifest_sha256,
        sync_manifest_sha256=postfreeze_binding["step3_manifest_sha256"],
    )

    candidate_snapshots: dict[Path, dict[str, Any]] = {}

    def current_candidate(chain: str, track: str, fold: str) -> dict[str, Any]:
        role = _expected_target_role(chain, track)
        if fold == "fold2":
            role = role.replace(".csv", "_fold2.csv")
        path = candidate_root / Path(role).name
        if path not in candidate_snapshots:
            candidate_snapshots[path] = _candidate_snapshot(
                path, fold=fold, role=f"canonical candidate {chain}|{track}|{fold}"
            )
        return candidate_snapshots[path]

    expected_metric_names = {
        f"metrics_{chain}_track-{track}_{family}.json"
        for chain in CHAINS
        for track in TRACKS
        for family in FAMILIES
    }
    expected_score_names = {
        f"scores_{chain}_track-{track}_{family}.csv"
        for chain in CHAINS
        for track in TRACKS
        for family in FAMILIES
    }
    if _artifact_names(root, "metrics", "metrics_", ".json") != expected_metric_names:
        _fail("metrics", "must contain exactly the 36 canonical evaluation JSON files")
    if _artifact_names(root, "scores", "scores_", ".csv") != expected_score_names:
        _fail("scores", "must contain exactly the 36 canonical score CSV files")

    selections: dict[tuple[str, str, str], dict[str, Any]] = {}
    for chain in CHAINS:
        for family in FAMILIES:
            shared_contexts = set()
            history_candidates = {}
            for track in TRACKS:
                key = selection_key(chain, track, family)
                _, selection = indexed[key]
                _validate_selection_for_reporting(
                    selection, family, track, f"selection {key}"
                )
                cache = selection.get("shared_score_cache")
                shared_contexts.add(_hash(cache.get("context_sha256"), f"selection {key} cache"))
                candidate = selection.get("history_candidate")
                if not isinstance(candidate, dict):
                    _fail(f"selection {key}", "missing history candidate hash")
                _hash(candidate.get("sha256"), f"selection {key} history candidate")
                expected_history_role = _expected_target_role(chain, track).replace(
                    ".csv", "_fold2.csv"
                )
                if candidate.get("path", "").replace("\\", "/") != expected_history_role:
                    _fail(f"selection {key}", "history candidate is not the canonical fold2 input")
                history_rows = _integer(
                    candidate.get("rows"), f"selection {key} history rows", minimum=1
                )
                history_positives = _integer(
                    candidate.get("positive_lanes"), f"selection {key} history positives"
                )
                if history_positives > history_rows:
                    _fail(f"selection {key}", "history positives exceed input rows")
                _validate_candidate_receipt(
                    candidate,
                    current_candidate(chain, track, "fold2"),
                    f"selection {key} history candidate",
                )
                history_candidates[track] = candidate
                selections[(chain, track, family)] = selection
            if len(shared_contexts) != 1:
                _fail(f"selection job {chain}|{family}", "tasks do not share one score-cache context")
            if history_candidates["b1"] != history_candidates["b2"]:
                _fail(f"selection job {chain}|{family}", "B1/B2 history inputs differ")

    records = []
    target_candidate_claims: dict[tuple[str, str, str], dict[str, Any]] = {}
    score_identity_receipts: dict[tuple[str, str, str], str] = {}
    recomputation_receipts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for chain in CHAINS:
        for family in FAMILIES:
            job_selections = {
                track: selections[(chain, track, family)] for track in TRACKS
            }
            shared_timestamps = None
            target_candidates = {}
            for track in TRACKS:
                key = selection_key(chain, track, family)
                freeze_entry, selection = indexed[key]
                metric_name = f"metrics_{chain}_track-{track}_{family}.json"
                score_name = f"scores_{chain}_track-{track}_{family}.csv"
                metric_path = root / "metrics" / metric_name
                score_path = root / "scores" / score_name
                role = f"evaluation {key}"
                payload = _load_json(metric_path, role)
                required = {
                    "schema_version": EVALUATION_SCHEMA,
                    "protocol": PROTOCOL,
                    "status": "complete",
                    "chain": chain,
                    "track": track,
                    "family": family,
                    "manifest_sha256": manifest_sha256,
                    "selection_sha256": freeze_entry["sha256"],
                    "run_id": run_id,
                    "run_config_sha256": run_config_sha256,
                    "selected": selection["selected"],
                    "target_fold": TARGET_FOLD,
                    "aggregation": "calendar_mean",
                    "cohort_policy": "complete main cohort; no same-window dev/test split",
                    "orchestration": "chain_multitask_unique_config_training",
                    "seeds": list(SEEDS),
                }
                for field, expected in required.items():
                    if payload.get(field) != expected:
                        _fail(role, f"invalid or stale {field}")
                if _basename(payload.get("selection_manifest"), role) != "frozen_manifest.json":
                    _fail(role, "selection_manifest does not name the canonical freeze")
                if _basename(payload.get("run_config"), role) != "v2_gpu_rolling.json":
                    _fail(role, "run_config does not name the canonical config")
                if _basename(payload.get("score_artifact"), role) != score_name:
                    _fail(role, "score_artifact does not name the canonical score file")

                created = _timestamp(payload.get("created_at_utc"), f"{role}.created_at_utc")
                timestamps = _validate_protocol_timestamps(
                    payload.get("protocol_timestamps"), job_selections, role
                )
                label_read = _timestamp(
                    timestamps["main_target_labels_read_at_utc"], f"{role}.label_read"
                )
                if created < label_read:
                    _fail(role, "artifact was timestamped before the target-label read")
                if shared_timestamps is None:
                    shared_timestamps = timestamps
                elif timestamps != shared_timestamps:
                    _fail(f"evaluation job {chain}|{family}", "A/B1/B2 do not share one training/label-read trace")

                candidate = payload.get("target_candidate")
                if not isinstance(candidate, dict):
                    _fail(role, "missing target candidate provenance")
                expected_target_role = _expected_target_role(chain, track)
                if candidate.get("path", "").replace("\\", "/") != expected_target_role:
                    _fail(role, "target candidate is not the canonical repo-relative main input")
                target_sha = _hash(candidate.get("sha256"), f"{role}.target candidate")
                target_rows = _integer(candidate.get("rows"), f"{role}.target rows", minimum=1)
                target_positives = _integer(candidate.get("positive_lanes"), f"{role}.positives")
                if target_positives > target_rows:
                    _fail(role, "positive lanes exceed target rows")
                target_candidate_claims[(chain, track, family)] = dict(candidate)
                target_candidates[track] = dict(candidate)

                metric_values = _validate_metric_values(
                    track, payload.get("per_seed"), payload.get("summary"), role
                )
                bootstrap = _validate_bootstrap(
                    track, payload.get("cluster_bootstrap_by_seed"), role
                )
                score_sha, score_identity_sha256, scores_by_seed = _validate_scores(
                    score_path,
                    selected_model=selection["selected"]["model"],
                    selection_sha256=freeze_entry["sha256"],
                    rows=target_rows,
                    role=f"score {key}",
                )
                score_identity_receipts[(chain, track, family)] = score_identity_sha256
                recomputation_receipts[(chain, track, family)] = {
                    "scores_by_seed": scores_by_seed,
                    "metric_values": metric_values,
                    "bootstrap": bootstrap,
                }
                primary = PRIMARY_METRICS[track]
                records.append(
                    {
                        "chain": chain,
                        "track": track,
                        "family": family,
                        "primary_metric": primary,
                        "primary_values_by_seed": [
                            row[primary] for row in metric_values["per_seed"]
                        ],
                        "primary_mean": metric_values["summary"][primary]["mean"],
                        "primary_std_across_seeds": metric_values["summary"][primary]["std"],
                        "selected_model": selection["selected"]["model"],
                        "selected_hyperparameters": selection["selected"]["hyperparameters"],
                        "history_holdout_selection_metric_mean": selection["selected"][
                            "history_holdout_selection_metric_mean"
                        ],
                        "history_candidate_role": _expected_target_role(chain, track).replace(
                            ".csv", "_fold2.csv"
                        ),
                        "history_candidate_sha256": selection["history_candidate"]["sha256"],
                        "target_candidate_role": expected_target_role,
                        "target_candidate_sha256": target_sha,
                        "target_rows": target_rows,
                        "target_positive_lanes": target_positives,
                        "selection_artifact_role": (
                            "results_v2/gpu_rolling/selections/"
                            f"{freeze_entry['path'].split('/')[-1]}"
                        ),
                        "selection_sha256": freeze_entry["sha256"],
                        "metric_artifact_role": f"results_v2/gpu_rolling/metrics/{metric_name}",
                        "metric_artifact_sha256": sha256_file(metric_path),
                        "score_artifact_role": f"results_v2/gpu_rolling/scores/{score_name}",
                        "score_artifact_sha256": score_sha,
                        "metrics_by_seed": metric_values["per_seed"],
                        "metrics_summary": metric_values["summary"],
                        "cluster_bootstrap_by_seed": bootstrap,
                    }
                )
            if target_candidates["b1"] != target_candidates["b2"]:
                _fail(f"evaluation job {chain}|{family}", "B1/B2 main input provenance differs")

    # This is a post-main promotion gate.  Only after every formal
    # metric/timestamp/score artifact has passed structural checks do we parse
    # current main labels, bind score identities to the cohort, and mechanically
    # recompute every reported metric and deterministic bootstrap interval.
    if len(records) != EXPECTED_EVALUATIONS:
        _fail("evaluation inventory", "did not validate all 36 formal evaluations")
    for chain in CHAINS:
        for track in TRACKS:
            observed = current_candidate(chain, track, TARGET_FOLD)
            for family in FAMILIES:
                key = (chain, track, family)
                _validate_candidate_receipt(
                    target_candidate_claims[key],
                    observed,
                    f"evaluation {selection_key(*key)}.target candidate",
                )
                if score_identity_receipts[key] != observed["identity_sha256"]:
                    _fail(
                        f"score {selection_key(*key)}",
                        "score identities do not exactly match the current candidate cohort",
                    )
                receipt = recomputation_receipts[key]
                _validate_main_recomputation(
                    track=track,
                    candidate=observed,
                    scores_by_seed=receipt["scores_by_seed"],
                    metric_values=receipt["metric_values"],
                    bootstrap=receipt["bootstrap"],
                    role=f"evaluation {selection_key(*key)} recomputation",
                )

    macro_summary = []
    for track in TRACKS:
        for family in FAMILIES:
            rows = [
                row for row in records if row["track"] == track and row["family"] == family
            ]
            if [row["chain"] for row in rows] != list(CHAINS):
                _fail("macro summary", f"non-canonical chain order for {track}|{family}")
            chain_means = [float(row["primary_mean"]) for row in rows]
            per_seed_macro = [
                {
                    "seed": seed,
                    "mean_across_six_chains": _mean(
                        [float(row["primary_values_by_seed"][seed]) for row in rows]
                    ),
                }
                for seed in SEEDS
            ]
            macro_summary.append(
                {
                    "track": track,
                    "family": family,
                    "primary_metric": PRIMARY_METRICS[track],
                    "mean_across_six_chain_means": _mean(chain_means),
                    "std_across_six_chain_means": _std(chain_means),
                    "n_chains": len(CHAINS),
                    "per_seed_macro": per_seed_macro,
                }
            )

    result = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "run_id": run_id,
        "target_fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
        "manifest_artifact_role": "results_v2/gpu_rolling/frozen_manifest.json",
        "manifest_sha256": manifest_sha256,
        "run_config_artifact_role": "configs/v2_gpu_rolling.json",
        "run_config_sha256": run_config_sha256,
        "post_freeze_semantic_attestation": postfreeze_binding,
        "nbfnet_source_evidence": nbfnet_source_evidence,
        "metric_recomputation": {
            "source_artifact_role": "src/v2_gpu_rolling.py",
            "source_sha256": metric_source_sha256,
            "current_candidate_labels_and_values_used_post_main": True,
            "all_per_seed_metrics_and_summaries_recomputed": True,
            "all_cluster_bootstrap_intervals_recomputed": True,
        },
        "seeds": list(SEEDS),
        "complete_chain_family_jobs": EXPECTED_CHAIN_FAMILY_JOBS,
        "complete_task_evaluations": EXPECTED_EVALUATIONS,
        "reporting_policy": {
            "all_frozen_families_reported_separately": True,
            "all_six_chains_in_every_macro": True,
            "main_test_champion_selected": False,
            "macro_weighting": "unweighted arithmetic mean of six preregistered chain means",
            "raw_score_cross_seed_averaging": False,
            "bootstrap_rng_seed_mechanically_verified": True,
        },
        "records": records,
        "macro_summary": macro_summary,
    }
    _privacy_audit(result)
    return result


CSV_FIELDS = (
    "scope",
    "unit",
    "chain",
    "track",
    "family",
    "primary_metric",
    "mean",
    "std",
    "n",
    "seed_values_json",
    "selected_model",
    "selected_hyperparameters_json",
    "target_rows",
    "target_positive_lanes",
    "history_candidate_sha256",
    "target_candidate_sha256",
    "selection_sha256",
    "metric_artifact_sha256",
    "score_artifact_sha256",
    "manifest_sha256",
    "run_config_sha256",
    "post_freeze_semantic_attestation_sha256",
    "nbfnet_source_evidence_sha256",
    "metric_recomputation_source_sha256",
    "run_id",
)


def render_json(summary: Mapping[str, Any]) -> bytes:
    _privacy_audit(summary)
    return (
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def render_csv(summary: Mapping[str, Any]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    common = {
        "manifest_sha256": summary["manifest_sha256"],
        "run_config_sha256": summary["run_config_sha256"],
        "post_freeze_semantic_attestation_sha256": summary[
            "post_freeze_semantic_attestation"
        ]["sha256"],
        "nbfnet_source_evidence_sha256": _stable_object_sha256(
            summary["nbfnet_source_evidence"]
        ),
        "metric_recomputation_source_sha256": summary["metric_recomputation"][
            "source_sha256"
        ],
        "run_id": summary["run_id"],
    }
    for row in summary["records"]:
        writer.writerow(
            {
                **common,
                "scope": "chain",
                "unit": "five_seeds",
                "chain": row["chain"],
                "track": row["track"],
                "family": row["family"],
                "primary_metric": row["primary_metric"],
                "mean": format(float(row["primary_mean"]), ".17g"),
                "std": format(float(row["primary_std_across_seeds"]), ".17g"),
                "n": len(SEEDS),
                "seed_values_json": json.dumps(row["primary_values_by_seed"], separators=(",", ":")),
                "selected_model": row["selected_model"],
                "selected_hyperparameters_json": _canonical_key(row["selected_hyperparameters"]),
                "target_rows": row["target_rows"],
                "target_positive_lanes": row["target_positive_lanes"],
                "history_candidate_sha256": row["history_candidate_sha256"],
                "target_candidate_sha256": row["target_candidate_sha256"],
                "selection_sha256": row["selection_sha256"],
                "metric_artifact_sha256": row["metric_artifact_sha256"],
                "score_artifact_sha256": row["score_artifact_sha256"],
            }
        )
    for row in summary["macro_summary"]:
        writer.writerow(
            {
                **common,
                "scope": "macro",
                "unit": "six_preregistered_chains",
                "chain": "__macro__",
                "track": row["track"],
                "family": row["family"],
                "primary_metric": row["primary_metric"],
                "mean": format(float(row["mean_across_six_chain_means"]), ".17g"),
                "std": format(float(row["std_across_six_chain_means"]), ".17g"),
                "n": row["n_chains"],
                "seed_values_json": json.dumps(
                    [item["mean_across_six_chains"] for item in row["per_seed_macro"]],
                    separators=(",", ":"),
                ),
            }
        )
    encoded = buffer.getvalue().encode("utf-8")
    # CSV values are constructed from an allowlist.  Still apply a final cheap
    # privacy tripwire before promotion.
    rendered = encoded.decode("utf-8")
    for label, pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(rendered):
            _fail("public CSV", f"private token leaked: {label}")
    return encoded


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_outputs(
    root: Path,
    json_out: Path,
    csv_out: Path,
) -> dict[str, Any]:
    summary = build_summary(root)
    json_bytes = render_json(summary)
    csv_bytes = render_csv(summary)
    _atomic_write(Path(json_out), json_bytes)
    _atomic_write(Path(csv_out), csv_bytes)
    return summary


def verify_outputs(
    root: Path,
    json_out: Path,
    csv_out: Path,
) -> dict[str, Any]:
    summary = build_summary(root)
    expected = {
        Path(json_out): render_json(summary),
        Path(csv_out): render_csv(summary),
    }
    for path, wanted in expected.items():
        try:
            actual = path.read_bytes()
        except OSError as exc:
            _fail("published output", f"cannot read {path.name}: {exc}")
        if actual != wanted:
            _fail("published output", f"{path.name} is stale or non-deterministic")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-output",
        action="store_true",
        help="rebuild in memory and byte-verify the two canonical public outputs",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="validate the canonical private run without writing public outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_output:
            summary = verify_outputs(CANONICAL_ROOT, DEFAULT_JSON_OUT, DEFAULT_CSV_OUT)
            action = "verified"
        elif args.check_only:
            summary = build_summary(CANONICAL_ROOT)
            action = "validated (no files written)"
        else:
            summary = write_outputs(CANONICAL_ROOT, DEFAULT_JSON_OUT, DEFAULT_CSV_OUT)
            action = "written"
    except ResultValidationError as exc:
        print(f"GPU RESULT PROMOTION REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        f"GPU result summary {action}: "
        f"{summary['complete_chain_family_jobs']} jobs / "
        f"{summary['complete_task_evaluations']} task evaluations; "
        "main-test champion selection=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
