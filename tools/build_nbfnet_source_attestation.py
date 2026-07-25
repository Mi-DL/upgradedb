#!/usr/bin/env python3
"""Create and verify a supplemental attestation for ``NBFNET_PATH``.

The formal GPU snapshot pins UpgradeBench's own runner, but NBFNet is loaded
from an external source tree named by ``NBFNET_PATH``.  This tool records the
exact, current bytes of that tree as retrospective supplemental evidence.  It
does not claim that a post-hoc observation proves what was present at an
earlier execution time.

The private receipt retains the resolved source path.  ``project-public``
derives a path-free public receipt that keeps the complete relative-file
inventory, tree digest, run binding, Git evidence, and the SHA-256 of the
private receipt.

Exclusion policy
----------------
Only explicitly named runtime/cache/data artefacts are excluded.  Every other
regular file, regardless of suffix, is hashed.  Excluded directory basenames
are: .git, __pycache__, .pytest_cache, .mypy_cache, .ruff_cache, .tox, .venv,
venv, build, dist, data, datasets, dataset_cache, cache, checkpoints, runs,
outputs, logs, tmp, and temp.  Excluded file basenames are .DS_Store and
Thumbs.db.  Excluded suffixes are .pyc, .pyo, .log, .tmp, .swp, .pt, .pth,
.ckpt, .npy, .npz, .pkl, .pickle, and .parquet.  Symlinks that are not already
inside an excluded subtree or matched by an excluded file rule are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


PRIVATE_SCHEMA = "upgrade-bench-v2/nbfnet-source-private-attestation/1"
PUBLIC_SCHEMA = "upgrade-bench-v2/nbfnet-source-public-attestation/1"
RUNTIME_COMPARISON_SCHEMA = (
    "upgrade-bench-v2/nbfnet-cross-host-runtime-comparison/1"
)
SOURCE_COMPARISON_SCHEMA = "upgrade-bench-v2/nbfnet-source-tree-comparison/1"
SOURCE_ENVIRONMENT_VARIABLE = "NBFNET_PATH"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
SAFE_ROLE = re.compile(r"[A-Za-z0-9._/-]+\Z")
GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}\Z")

EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "build",
        "dist",
        "data",
        "datasets",
        "dataset_cache",
        "cache",
        "checkpoints",
        "runs",
        "outputs",
        "logs",
        "tmp",
        "temp",
    }
)
EXCLUDED_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db"})
EXCLUDED_FILE_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".log",
        ".tmp",
        ".swp",
        ".pt",
        ".pth",
        ".ckpt",
        ".npy",
        ".npz",
        ".pkl",
        ".pickle",
        ".parquet",
    }
)
TREE_DIGEST_ALGORITHM = (
    "sha256 over sorted UTF-8 records: "
    "relative_posix_path + NUL + decimal_size + NUL + file_sha256 + LF"
)


class SourceAttestationError(RuntimeError):
    """Raised when source evidence cannot be created or verified safely."""


@dataclass(frozen=True)
class RunIdentityInputs:
    """Externally expected identity of the frozen formal GPU run."""

    run_id: str
    frozen_manifest: Path
    frozen_manifest_sha256: str
    step3_manifest: Path
    step3_manifest_sha256: str


def _fail(message: str) -> None:
    raise SourceAttestationError(message)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceAttestationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")


def _normalise_utc(value: str, role: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise SourceAttestationError(f"{role} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        _fail(f"{role} lacks a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _strict_json_bytes(content: bytes, role: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(content.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAttestationError(f"{role} is not strict UTF-8 JSON: {exc}") from exc


def _strict_json_file(path: Path, role: str) -> Any:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise SourceAttestationError(f"cannot read {role}: {exc}") from exc
    return _strict_json_bytes(content, role)


def _portable_sort(values: Sequence[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _exclusion_policy() -> dict[str, Any]:
    return {
        "version": 1,
        "directory_basenames": _portable_sort(list(EXCLUDED_DIRECTORY_NAMES)),
        "file_basenames": _portable_sort(list(EXCLUDED_FILE_NAMES)),
        "file_suffixes_case_insensitive": _portable_sort(
            list(EXCLUDED_FILE_SUFFIXES)
        ),
        "all_other_regular_files_hashed": True,
        "included_symlink_policy": "refuse",
        "excluded_content_semantics": (
            "runtime caches, generated data/model payloads, build products, logs, and "
            "editor/OS debris are intentionally outside the source-tree trust boundary"
        ),
    }


def _excluded_file(name: str) -> bool:
    return name in EXCLUDED_FILE_NAMES or Path(name).suffix.lower() in EXCLUDED_FILE_SUFFIXES


def _validate_public_role(value: str, role: str) -> None:
    if (
        not value
        or not SAFE_ROLE.fullmatch(value)
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
    ):
        _fail(f"{role} must be a safe, non-absolute public role")


def _file_record(path: Path, relative: str) -> tuple[dict[str, Any], int]:
    try:
        before = path.stat()
    except OSError as exc:
        raise SourceAttestationError(f"cannot stat source file {relative}: {exc}") from exc
    if not path.is_file() or path.is_symlink():
        _fail(f"included source entry is not a regular non-symlink file: {relative}")
    digest = sha256_file(path)
    try:
        after = path.stat()
    except OSError as exc:
        raise SourceAttestationError(
            f"cannot restat source file {relative}: {exc}"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        _fail(f"source file changed while it was being hashed: {relative}")
    return (
        {"path": relative, "sha256": digest, "size_bytes": before.st_size},
        before.st_mtime_ns,
    )


def snapshot_tree(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash every included regular file under *source_root* deterministically."""

    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        _fail(f"NBFNET_PATH is not a directory: {source_root}")

    discovered: list[tuple[str, Path]] = []
    for directory, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        retained_directories: list[str] = []
        for name in _portable_sort(directory_names):
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            candidate = current / name
            relative = candidate.relative_to(source_root).as_posix()
            if candidate.is_symlink():
                _fail(f"included source directory is a symlink: {relative}")
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in _portable_sort(file_names):
            if _excluded_file(name):
                continue
            candidate = current / name
            relative = candidate.relative_to(source_root).as_posix()
            if (
                not relative
                or "\x00" in relative
                or "\n" in relative
                or "\r" in relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
            ):
                _fail(f"unsafe source-relative path: {relative!r}")
            discovered.append((relative, candidate))

    discovered.sort(key=lambda row: row[0].encode("utf-8"))
    files: list[dict[str, Any]] = []
    mtimes: list[int] = []
    for relative, path in discovered:
        record, mtime_ns = _file_record(path, relative)
        files.append(record)
        mtimes.append(mtime_ns)
    if not files:
        _fail("NBFNET_PATH has no files inside the declared source trust boundary")

    canonical_inventory = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    tree_hasher = hashlib.sha256()
    for row in files:
        tree_hasher.update(str(row["path"]).encode("utf-8"))
        tree_hasher.update(b"\0")
        tree_hasher.update(str(row["size_bytes"]).encode("ascii"))
        tree_hasher.update(b"\0")
        tree_hasher.update(str(row["sha256"]).encode("ascii"))
        tree_hasher.update(b"\n")

    inventory = {
        "exclusion_policy": _exclusion_policy(),
        "file_count": len(files),
        "total_bytes": sum(int(row["size_bytes"]) for row in files),
        "inventory_sha256": sha256_bytes(canonical_inventory),
        "tree_sha256": tree_hasher.hexdigest(),
        "tree_digest_algorithm": TREE_DIGEST_ALGORITHM,
        "files": files,
    }
    timestamps = {
        "earliest_included_file_mtime_utc": _utc_from_timestamp(min(mtimes) / 1e9),
        "latest_included_file_mtime_utc": _utc_from_timestamp(max(mtimes) / 1e9),
        "mtime_semantics": (
            "recorded as retrospective filesystem evidence; mtimes are not a content "
            "identity and are not used as the verification trust anchor"
        ),
    }
    return inventory, timestamps


def _resolve_source_root(source_root: Path | None) -> Path:
    raw_environment = os.environ.get(SOURCE_ENVIRONMENT_VARIABLE)
    if raw_environment is None or not raw_environment.strip():
        _fail(f"{SOURCE_ENVIRONMENT_VARIABLE} must be set for a formal source attestation")
    environment_root = Path(raw_environment).expanduser().resolve()
    selected = environment_root if source_root is None else Path(source_root).expanduser().resolve()
    if selected != environment_root:
        _fail(
            f"explicit source root differs from {SOURCE_ENVIRONMENT_VARIABLE}; "
            "refusing an unbound source attestation"
        )
    if not selected.is_dir():
        _fail(f"{SOURCE_ENVIRONMENT_VARIABLE} does not name a directory: {selected}")
    return selected


def _validate_expected_digest(value: str, role: str) -> None:
    if not HEX64.fullmatch(value):
        _fail(f"{role} must be 64 lowercase hexadecimal characters")


def attest_run_identity(inputs: RunIdentityInputs) -> dict[str, Any]:
    if not SAFE_RUN_ID.fullmatch(inputs.run_id):
        _fail("run_id contains characters outside the formal identifier alphabet")
    _validate_expected_digest(inputs.frozen_manifest_sha256, "frozen manifest SHA-256")
    _validate_expected_digest(inputs.step3_manifest_sha256, "Step-3 manifest SHA-256")

    frozen_actual = sha256_file(inputs.frozen_manifest)
    if frozen_actual != inputs.frozen_manifest_sha256:
        _fail("frozen manifest differs from the externally expected SHA-256")
    step3_actual = sha256_file(inputs.step3_manifest)
    if step3_actual != inputs.step3_manifest_sha256:
        _fail("Step-3 manifest differs from the externally expected SHA-256")

    frozen = _strict_json_file(inputs.frozen_manifest, "frozen GPU manifest")
    if not isinstance(frozen, Mapping):
        _fail("frozen GPU manifest is not a JSON object")
    if frozen.get("schema_version") != "upgrade-bench-v2/gpu-freeze/1":
        _fail("frozen GPU manifest schema is not the formal v2 freeze schema")
    if frozen.get("status") != "frozen":
        _fail("GPU manifest is not frozen")
    if frozen.get("all_selections_frozen_before_main") is not True:
        _fail("GPU manifest lacks the all-selections-frozen-before-main gate")
    if frozen.get("run_id") != inputs.run_id:
        _fail("frozen GPU manifest run_id differs from the expected run_id")

    return {
        "run_id": inputs.run_id,
        "frozen_manifest": {
            "role": "results_v2/gpu_rolling/frozen_manifest.json",
            "sha256": frozen_actual,
        },
        "step3_sync_manifest": {
            "role": (
                "results_v2/gpu_rolling/runs/"
                f"{inputs.run_id}/STEP3_SYNC_MANIFEST.sha256"
            ),
            "sha256": step3_actual,
        },
    }


def _run_git(source_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, UnicodeError) as exc:
        raise SourceAttestationError(f"cannot inspect Git provenance: {exc}") from exc


def _git_timestamp(source_root: Path, head: str, placeholder: str) -> str:
    result = _run_git(source_root, ["show", "-s", f"--format={placeholder}", head])
    if result.returncode != 0:
        _fail(f"cannot read Git commit timestamp: {result.stderr.strip()}")
    return _normalise_utc(result.stdout.strip(), "Git commit timestamp")


def _source_relative_git_paths(
    values: Sequence[str], *, repository_root: Path, source_root: Path
) -> list[str]:
    rows: list[str] = []
    for value in values:
        if not value:
            continue
        candidate = (repository_root / Path(value)).resolve()
        try:
            relative = candidate.relative_to(source_root).as_posix()
        except ValueError:
            _fail(f"Git returned a path outside NBFNET_PATH: {value!r}")
        rows.append(relative)
    return _portable_sort(rows)


def git_provenance(source_root: Path) -> dict[str, Any]:
    """Capture HEAD and tracked source-scope dirtiness, or explicit absence."""

    top = _run_git(source_root, ["rev-parse", "--show-toplevel"])
    if top.returncode != 0:
        return {
            "repository_detected": False,
            "repository_root": None,
            "head": None,
            "head_author_timestamp_utc": None,
            "head_committer_timestamp_utc": None,
            "source_tracked_file_count": 0,
            "tracked_dirty": None,
            "dirty_tracked_paths": [],
            "absence_semantics": (
                "no Git worktree was discoverable from NBFNET_PATH at observation time"
            ),
        }

    repository_root = Path(top.stdout.strip()).resolve()
    try:
        source_scope = source_root.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        _fail("Git top-level does not contain NBFNET_PATH")
    scope_argument = source_scope or "."

    head_result = _run_git(source_root, ["rev-parse", "--verify", "HEAD"])
    if head_result.returncode != 0:
        _fail("Git repository has no verifiable HEAD")
    head = head_result.stdout.strip()
    if not GIT_OBJECT_ID.fullmatch(head):
        _fail("Git HEAD is not a recognised object identifier")

    tracked = _run_git(
        source_root, ["ls-files", "-z", "--", scope_argument]
    )
    dirty = _run_git(
        source_root, ["diff", "--name-only", "-z", "HEAD", "--", scope_argument]
    )
    if tracked.returncode != 0 or dirty.returncode != 0:
        _fail("cannot enumerate tracked NBFNet source state")
    tracked_paths = _source_relative_git_paths(
        tracked.stdout.split("\0"), repository_root=repository_root, source_root=source_root
    )
    dirty_paths = _source_relative_git_paths(
        dirty.stdout.split("\0"), repository_root=repository_root, source_root=source_root
    )

    head_after = _run_git(source_root, ["rev-parse", "--verify", "HEAD"])
    if head_after.returncode != 0 or head_after.stdout.strip() != head:
        _fail("Git HEAD changed during source attestation")
    return {
        "repository_detected": True,
        "repository_root": str(repository_root),
        "head": head,
        "head_author_timestamp_utc": _git_timestamp(source_root, head, "%aI"),
        "head_committer_timestamp_utc": _git_timestamp(source_root, head, "%cI"),
        "source_tracked_file_count": len(tracked_paths),
        "tracked_dirty": bool(dirty_paths),
        "dirty_tracked_paths": dirty_paths,
        "absence_semantics": None,
    }


def _runtime_artifact_record(
    role: str, path: Path, *, expected_sha256: str | None
) -> dict[str, Any]:
    _validate_public_role(role, "runtime artifact role")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        _fail(f"runtime artifact is not a regular non-symlink file: {role}")
    try:
        before = resolved.stat()
    except OSError as exc:
        raise SourceAttestationError(f"cannot stat runtime artifact {role}: {exc}") from exc
    digest = sha256_file(resolved)
    if expected_sha256 is not None:
        _validate_expected_digest(
            expected_sha256, f"expected runtime artifact SHA-256 for {role}"
        )
        if digest != expected_sha256:
            _fail(f"runtime artifact differs from the expected SHA-256: {role}")
    try:
        after = resolved.stat()
    except OSError as exc:
        raise SourceAttestationError(
            f"cannot restat runtime artifact {role}: {exc}"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        _fail(f"runtime artifact changed while being hashed: {role}")
    return {
        "role": role,
        "resolved_path": str(resolved),
        "size_bytes": before.st_size,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "matches_expected_sha256": True if expected_sha256 is not None else None,
        "mtime_utc": _utc_from_timestamp(before.st_mtime_ns / 1e9),
    }


def capture_runtime_artifacts(
    runtime_artifacts: Sequence[tuple[str, Path]],
    *,
    host_role: str | None,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if host_role is not None:
        _validate_public_role(host_role, "host role")
    if runtime_artifacts and host_role is None:
        _fail("--host-role is required when runtime artifacts are attested")
    roles = [role for role, _ in runtime_artifacts]
    if len(roles) != len(set(roles)):
        _fail("runtime artifact roles must be unique within one host receipt")
    expected = dict(expected_sha256 or {})
    if set(expected) - set(roles):
        _fail("an expected runtime SHA-256 was supplied for an unattested role")
    records = [
        _runtime_artifact_record(role, path, expected_sha256=expected.get(role))
        for role, path in sorted(
            runtime_artifacts, key=lambda row: row[0].encode("utf-8")
        )
    ]
    return {
        "host_role": host_role,
        "observed_hostname": socket.gethostname(),
        "artifacts": records,
        "cross_host_match_semantics": (
            "one receipt records one host; compare-runtime must compare at least two "
            "host receipts before any cross-host equality claim"
        ),
    }


def _selection_timing_fact(
    timestamps: Mapping[str, Any], selection_started_at_utc: str | None
) -> dict[str, Any]:
    if selection_started_at_utc is None:
        return {
            "selection_started_at_utc": None,
            "latest_source_mtime_not_after_selection_start": None,
            "seconds_from_latest_source_mtime_to_selection_start": None,
            "interpretation": (
                "no externally recorded selection-start timestamp was supplied"
            ),
        }
    selection_text = _normalise_utc(
        selection_started_at_utc, "selection-start timestamp"
    )
    latest_text = str(timestamps["latest_included_file_mtime_utc"])
    selection = datetime.fromisoformat(selection_text)
    latest = datetime.fromisoformat(latest_text)
    delta = (selection - latest).total_seconds()
    return {
        "selection_started_at_utc": selection_text,
        "latest_source_mtime_not_after_selection_start": delta >= 0,
        "seconds_from_latest_source_mtime_to_selection_start": delta,
        "interpretation": (
            "a non-later source mtime is a retrospective consistency fact only; mtime "
            "can be preserved or altered independently of content and is not proof of a "
            "pre-selection source freeze"
        ),
    }


def build_private_receipt(
    identity_inputs: RunIdentityInputs,
    *,
    source_root: Path | None = None,
    observed_at_utc: str | None = None,
    runtime_artifacts: Sequence[tuple[str, Path]] = (),
    expected_runtime_sha256: Mapping[str, str] | None = None,
    host_role: str | None = None,
    selection_started_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a private receipt without writing it."""

    selected_root = _resolve_source_root(source_root)
    run_identity = attest_run_identity(identity_inputs)
    inventory_before, timestamps_before = snapshot_tree(selected_root)
    git = git_provenance(selected_root)
    inventory_after, timestamps_after = snapshot_tree(selected_root)
    if inventory_before != inventory_after or timestamps_before != timestamps_after:
        _fail("NBFNET_PATH changed while the source attestation was being captured")

    observed = _normalise_utc(
        observed_at_utc if observed_at_utc is not None else _utc_now(),
        "observation timestamp",
    )
    runtime = capture_runtime_artifacts(
        runtime_artifacts,
        host_role=host_role,
        expected_sha256=expected_runtime_sha256,
    )
    selection_timing = _selection_timing_fact(
        timestamps_before, selection_started_at_utc
    )
    return {
        "schema_version": PRIVATE_SCHEMA,
        "attestation_type": "retrospective_supplemental_nbfnet_source_tree_evidence",
        "status": "PASS",
        "observed_at_utc": observed,
        "run_identity": run_identity,
        "source": {
            "selector": SOURCE_ENVIRONMENT_VARIABLE,
            "environment_binding_verified": True,
            "resolved_path": str(selected_root),
            "resolved_path_sha256": sha256_bytes(str(selected_root).encode("utf-8")),
            "inventory": inventory_before,
        },
        "git": git,
        "runtime": runtime,
        "filesystem_timestamps": timestamps_before,
        "selection_timing": selection_timing,
        "claim_boundary": {
            "supported": (
                "At the recorded observation time, the tree named by NBFNET_PATH had the "
                "listed relative files and exact byte hashes and was bound to the stated "
                "frozen run and Step-3 manifest identities."
            ),
            "not_supported": (
                "Because this receipt is retrospective, it does not by itself prove that "
                "the observed tree was unchanged since selection or main execution; it is "
                "supplemental evidence, not a replacement for a contemporaneous freeze. "
                "Runtime artifacts support an execution-binary claim only when explicitly "
                "provided, and cross-host equality requires compare-runtime."
            ),
        },
    }


def render_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_private_shape(receipt: Mapping[str, Any]) -> None:
    expected = {
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
    }
    if set(receipt) != expected:
        _fail("private receipt top-level schema is not exact")
    if receipt.get("schema_version") != PRIVATE_SCHEMA or receipt.get("status") != "PASS":
        _fail("private receipt schema/status is invalid")
    _normalise_utc(str(receipt.get("observed_at_utc")), "receipt observation timestamp")
    run_identity = receipt.get("run_identity")
    if not isinstance(run_identity, Mapping) or set(run_identity) != {
        "run_id",
        "frozen_manifest",
        "step3_sync_manifest",
    }:
        _fail("private receipt run-identity schema is not exact")
    if not isinstance(run_identity.get("run_id"), str) or not SAFE_RUN_ID.fullmatch(
        str(run_identity["run_id"])
    ):
        _fail("private receipt run_id is invalid")
    for key in ("frozen_manifest", "step3_sync_manifest"):
        record = run_identity.get(key)
        if not isinstance(record, Mapping) or set(record) != {"role", "sha256"}:
            _fail(f"private receipt {key} binding schema is not exact")
        if not isinstance(record.get("role"), str):
            _fail(f"private receipt {key} lacks a role")
        _validate_public_role(str(record["role"]), f"{key} role")
        if not HEX64.fullmatch(str(record.get("sha256"))):
            _fail(f"private receipt {key} lacks a valid SHA-256")
    source = receipt.get("source")
    if not isinstance(source, Mapping):
        _fail("private receipt lacks its source record")
    if set(source) != {
        "selector",
        "environment_binding_verified",
        "resolved_path",
        "resolved_path_sha256",
        "inventory",
    }:
        _fail("private receipt source schema is not exact")
    if source.get("selector") != SOURCE_ENVIRONMENT_VARIABLE:
        _fail("private receipt source selector is not NBFNET_PATH")
    if source.get("environment_binding_verified") is not True:
        _fail("private receipt was not bound to NBFNET_PATH")
    resolved = source.get("resolved_path")
    if not isinstance(resolved, str) or not resolved:
        _fail("private receipt lacks the resolved NBFNET_PATH")
    if source.get("resolved_path_sha256") != sha256_bytes(resolved.encode("utf-8")):
        _fail("private receipt resolved-path digest is invalid")
    inventory = source.get("inventory")
    if not isinstance(inventory, Mapping):
        _fail("private receipt lacks a source inventory")
    _validate_inventory(inventory)
    if inventory.get("exclusion_policy") != _exclusion_policy():
        _fail("private receipt exclusion policy differs from the declared policy")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "host_role",
        "observed_hostname",
        "artifacts",
        "cross_host_match_semantics",
    }:
        _fail("private receipt runtime schema is not exact")
    host_role = runtime.get("host_role")
    if host_role is not None:
        if not isinstance(host_role, str):
            _fail("private receipt host role is not a string")
        _validate_public_role(host_role, "host role")
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, list):
        _fail("private receipt runtime artifacts are not a list")
    artifact_roles: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "role",
            "resolved_path",
            "size_bytes",
            "sha256",
            "expected_sha256",
            "matches_expected_sha256",
            "mtime_utc",
        }:
            _fail("private receipt runtime artifact schema is not exact")
        artifact_role = artifact.get("role")
        if not isinstance(artifact_role, str):
            _fail("private receipt runtime artifact lacks a role")
        _validate_public_role(artifact_role, "runtime artifact role")
        artifact_roles.append(artifact_role)
        if not HEX64.fullmatch(str(artifact.get("sha256"))):
            _fail("private receipt runtime artifact lacks a valid SHA-256")
        expected_digest = artifact.get("expected_sha256")
        expected_match = artifact.get("matches_expected_sha256")
        if expected_digest is None:
            if expected_match is not None:
                _fail("runtime artifact has a match flag without an expected digest")
        else:
            if not HEX64.fullmatch(str(expected_digest)) or expected_match is not True:
                _fail("runtime artifact expected-digest evidence is invalid")
            if artifact.get("sha256") != expected_digest:
                _fail("runtime artifact does not match its expected SHA-256")
        if not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] < 0:
            _fail("private receipt runtime artifact has an invalid size")
        _normalise_utc(str(artifact.get("mtime_utc")), "runtime artifact mtime")
    if artifact_roles != _portable_sort(artifact_roles) or len(artifact_roles) != len(
        set(artifact_roles)
    ):
        _fail("private receipt runtime artifact roles are not sorted and unique")
    if artifacts and host_role is None:
        _fail("private receipt runtime artifacts lack a host role")
    git = receipt.get("git")
    if not isinstance(git, Mapping) or set(git) != {
        "repository_detected",
        "repository_root",
        "head",
        "head_author_timestamp_utc",
        "head_committer_timestamp_utc",
        "source_tracked_file_count",
        "tracked_dirty",
        "dirty_tracked_paths",
        "absence_semantics",
    }:
        _fail("private receipt Git schema is not exact")
    if git.get("repository_detected") is True:
        if not GIT_OBJECT_ID.fullmatch(str(git.get("head"))):
            _fail("private receipt Git HEAD is invalid")
        _normalise_utc(
            str(git.get("head_author_timestamp_utc")), "Git author timestamp"
        )
        _normalise_utc(
            str(git.get("head_committer_timestamp_utc")), "Git committer timestamp"
        )
        if git.get("tracked_dirty") not in (True, False):
            _fail("private receipt Git dirty status is invalid")
    elif git.get("repository_detected") is not False:
        _fail("private receipt Git detection status is invalid")
    dirty_paths = git.get("dirty_tracked_paths")
    if not isinstance(dirty_paths, list) or dirty_paths != _portable_sort(dirty_paths):
        _fail("private receipt Git dirty paths are invalid")
    for dirty_path in dirty_paths:
        if (
            not isinstance(dirty_path, str)
            or PurePosixPath(dirty_path).is_absolute()
            or PureWindowsPath(dirty_path).is_absolute()
            or ".." in PurePosixPath(dirty_path).parts
        ):
            _fail("private receipt Git dirty path is unsafe")
    timestamps = receipt.get("filesystem_timestamps")
    if not isinstance(timestamps, Mapping) or set(timestamps) != {
        "earliest_included_file_mtime_utc",
        "latest_included_file_mtime_utc",
        "mtime_semantics",
    }:
        _fail("private receipt filesystem timestamp schema is not exact")
    _normalise_utc(
        str(timestamps.get("earliest_included_file_mtime_utc")),
        "earliest source mtime",
    )
    _normalise_utc(
        str(timestamps.get("latest_included_file_mtime_utc")),
        "latest source mtime",
    )
    selection = receipt.get("selection_timing")
    if not isinstance(selection, Mapping) or set(selection) != {
        "selection_started_at_utc",
        "latest_source_mtime_not_after_selection_start",
        "seconds_from_latest_source_mtime_to_selection_start",
        "interpretation",
    }:
        _fail("private receipt selection-timing schema is not exact")
    if selection.get("selection_started_at_utc") is not None:
        _normalise_utc(
            str(selection["selection_started_at_utc"]), "selection-start timestamp"
        )


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    if set(inventory) != {
        "exclusion_policy",
        "file_count",
        "total_bytes",
        "inventory_sha256",
        "tree_sha256",
        "tree_digest_algorithm",
        "files",
    }:
        _fail("source inventory schema is not exact")
    if inventory.get("tree_digest_algorithm") != TREE_DIGEST_ALGORITHM:
        _fail("source inventory tree-digest algorithm changed")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        _fail("source inventory file list is empty or invalid")
    paths: list[str] = []
    tree_hasher = hashlib.sha256()
    total_bytes = 0
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _fail("source inventory file record schema is not exact")
        path = row.get("path")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or PureWindowsPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or "\x00" in path
            or "\n" in path
            or "\r" in path
        ):
            _fail("source inventory contains an unsafe relative path")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            _fail(f"source inventory has an invalid file digest: {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _fail(f"source inventory has an invalid file size: {path}")
        paths.append(path)
        total_bytes += size
        tree_hasher.update(path.encode("utf-8"))
        tree_hasher.update(b"\0")
        tree_hasher.update(str(size).encode("ascii"))
        tree_hasher.update(b"\0")
        tree_hasher.update(digest.encode("ascii"))
        tree_hasher.update(b"\n")
    if paths != _portable_sort(paths) or len(paths) != len(set(paths)):
        _fail("source inventory paths are not sorted and unique")
    if inventory.get("file_count") != len(files):
        _fail("source inventory file count is inconsistent")
    if inventory.get("total_bytes") != total_bytes:
        _fail("source inventory total bytes are inconsistent")
    canonical_inventory = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if inventory.get("inventory_sha256") != sha256_bytes(canonical_inventory):
        _fail("source inventory aggregate digest is inconsistent")
    if inventory.get("tree_sha256") != tree_hasher.hexdigest():
        _fail("source inventory tree digest is inconsistent")


def verify_private_receipt(
    receipt_path: Path,
    identity_inputs: RunIdentityInputs,
    *,
    source_root: Path | None = None,
    runtime_artifacts: Sequence[tuple[str, Path]] | None = None,
    host_role: str | None = None,
    expected_runtime_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify receipt bytes against the current source tree and run identity."""

    receipt = _strict_json_file(receipt_path, "private NBFNet source receipt")
    if not isinstance(receipt, Mapping):
        _fail("private NBFNet source receipt is not a JSON object")
    _validate_private_shape(receipt)
    selected_root = _resolve_source_root(source_root)
    source = receipt["source"]
    if Path(str(source["resolved_path"])).resolve() != selected_root:
        _fail("private receipt names a different resolved NBFNET_PATH")
    expected_identity = attest_run_identity(identity_inputs)
    if receipt.get("run_identity") != expected_identity:
        _fail("private receipt run identity differs from the expected formal run")
    current_inventory, _ = snapshot_tree(selected_root)
    if source.get("inventory") != current_inventory:
        _fail("current NBFNET_PATH tree differs from the private receipt")
    receipt_runtime = receipt["runtime"]
    receipt_expected_runtime_digests = {
        str(row["role"]): str(row["expected_sha256"])
        for row in receipt_runtime["artifacts"]
        if row["expected_sha256"] is not None
    }
    if expected_runtime_sha256 is not None and dict(expected_runtime_sha256) != (
        receipt_expected_runtime_digests
    ):
        _fail("private receipt expected runtime digests differ from verifier inputs")
    if runtime_artifacts is None:
        runtime_inputs = [
            (str(row["role"]), Path(str(row["resolved_path"])))
            for row in receipt_runtime["artifacts"]
        ]
        expected_host_role = receipt_runtime["host_role"]
        expected_runtime_digests = receipt_expected_runtime_digests
    else:
        runtime_inputs = list(runtime_artifacts)
        expected_host_role = host_role
        expected_runtime_digests = receipt_expected_runtime_digests
    current_runtime = capture_runtime_artifacts(
        runtime_inputs,
        host_role=expected_host_role,
        expected_sha256=expected_runtime_digests,
    )
    comparable_runtime = dict(current_runtime)
    comparable_runtime["observed_hostname"] = receipt_runtime["observed_hostname"]
    if comparable_runtime != receipt_runtime:
        _fail("current runtime artifacts differ from the private receipt")
    return dict(receipt)


def _privacy_audit(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail("public receipt contains a non-string key")
            _privacy_audit(key)
            _privacy_audit(child)
    elif isinstance(value, list):
        for child in value:
            _privacy_audit(child)
    elif isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _fail("public receipt contains an absolute path")
        lowered = value.lower()
        if any(token in lowered for token in ("/home/", "/users/", "c:\\users\\")):
            _fail("public receipt contains a private home or data-root path")


def public_projection(
    private_receipt: Mapping[str, Any], *, private_receipt_sha256: str
) -> dict[str, Any]:
    """Derive the complete path-redacted public projection."""

    _validate_expected_digest(private_receipt_sha256, "private receipt SHA-256")
    _validate_private_shape(private_receipt)
    source = private_receipt["source"]
    git = private_receipt["git"]
    if not isinstance(git, Mapping):
        _fail("private receipt lacks Git evidence")
    public_git = {key: value for key, value in git.items() if key != "repository_root"}
    runtime = private_receipt["runtime"]
    public_runtime = {
        "host_role": runtime["host_role"],
        "observed_hostname_redacted": True,
        "artifacts": [
            {
                "role": row["role"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "expected_sha256": row["expected_sha256"],
                "matches_expected_sha256": row["matches_expected_sha256"],
                "mtime_utc": row["mtime_utc"],
                "resolved_path_redacted": True,
            }
            for row in runtime["artifacts"]
        ],
        "cross_host_match_semantics": runtime["cross_host_match_semantics"],
    }
    result = {
        "schema_version": PUBLIC_SCHEMA,
        "attestation_type": private_receipt["attestation_type"],
        "status": private_receipt["status"],
        "observed_at_utc": private_receipt["observed_at_utc"],
        "run_identity": private_receipt["run_identity"],
        "source": {
            "selector": source["selector"],
            "environment_binding_verified": source["environment_binding_verified"],
            "resolved_path_redacted": True,
            "inventory": source["inventory"],
        },
        "git": public_git,
        "runtime": public_runtime,
        "filesystem_timestamps": private_receipt["filesystem_timestamps"],
        "selection_timing": private_receipt["selection_timing"],
        "claim_boundary": private_receipt["claim_boundary"],
        "private_evidence": {
            "receipt_sha256": private_receipt_sha256,
            "retained_outside_public_bundle": True,
        },
    }
    _privacy_audit(result)
    return result


def project_public_receipt(private_path: Path) -> dict[str, Any]:
    try:
        private_bytes = Path(private_path).read_bytes()
    except OSError as exc:
        raise SourceAttestationError(f"cannot read private receipt: {exc}") from exc
    private = _strict_json_bytes(private_bytes, "private NBFNet source receipt")
    if not isinstance(private, Mapping):
        _fail("private NBFNet source receipt is not a JSON object")
    return public_projection(private, private_receipt_sha256=sha256_bytes(private_bytes))


def verify_public_projection(public_path: Path, private_path: Path) -> dict[str, Any]:
    actual = _strict_json_file(public_path, "public NBFNet source receipt")
    if not isinstance(actual, Mapping):
        _fail("public NBFNet source receipt is not a JSON object")
    expected = project_public_receipt(private_path)
    if actual != expected:
        _fail("public NBFNet source receipt differs from its private evidence")
    _privacy_audit(actual)
    return dict(actual)


def compare_source_receipts(private_paths: Sequence[Path]) -> dict[str, Any]:
    """Compare the complete source-tree inventories across two or more receipts."""

    if len(private_paths) < 2:
        _fail("compare-source requires at least two private source receipts")
    rows: list[dict[str, Any]] = []
    expected_run_identity: Any = None
    seen_host_roles: set[str] = set()
    for index, path in enumerate(private_paths):
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise SourceAttestationError(
                f"cannot read private source receipt: {exc}"
            ) from exc
        receipt = _strict_json_bytes(raw, "private NBFNet source receipt")
        if not isinstance(receipt, Mapping):
            _fail("private source receipt is not a JSON object")
        _validate_private_shape(receipt)
        if expected_run_identity is None:
            expected_run_identity = receipt["run_identity"]
        elif receipt["run_identity"] != expected_run_identity:
            _fail("source receipts bind different formal run identities")
        host_role = receipt["runtime"]["host_role"]
        public_role = str(host_role) if host_role is not None else f"source-receipt-{index + 1}"
        if public_role in seen_host_roles:
            _fail(f"duplicate source receipt role: {public_role}")
        seen_host_roles.add(public_role)
        inventory = receipt["source"]["inventory"]
        rows.append(
            {
                "receipt_role": public_role,
                "private_receipt_sha256": sha256_bytes(raw),
                "tree_sha256": inventory["tree_sha256"],
                "inventory_sha256": inventory["inventory_sha256"],
                "file_count": inventory["file_count"],
                "total_bytes": inventory["total_bytes"],
            }
        )
    rows.sort(key=lambda row: str(row["receipt_role"]).encode("utf-8"))
    signatures = {
        (
            row["tree_sha256"],
            row["inventory_sha256"],
            row["file_count"],
            row["total_bytes"],
        )
        for row in rows
    }
    result = {
        "schema_version": SOURCE_COMPARISON_SCHEMA,
        "status": "PASS" if len(signatures) == 1 else "MISMATCH",
        "run_identity": expected_run_identity,
        "receipt_count": len(rows),
        "all_source_trees_match": len(signatures) == 1,
        "receipts": rows,
        "claim_boundary": {
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
        },
    }
    _privacy_audit(result)
    return result


def compare_runtime_receipts(private_paths: Sequence[Path]) -> dict[str, Any]:
    """Compare source and explicitly attested runtime bytes across host receipts."""

    if len(private_paths) < 2:
        _fail("compare-runtime requires at least two private host receipts")
    rows: list[dict[str, Any]] = []
    seen_host_roles: set[str] = set()
    expected_run_identity: Any = None
    expected_artifact_roles: list[str] | None = None
    for path in private_paths:
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise SourceAttestationError(f"cannot read private host receipt: {exc}") from exc
        receipt = _strict_json_bytes(raw, "private NBFNet source receipt")
        if not isinstance(receipt, Mapping):
            _fail("private host receipt is not a JSON object")
        _validate_private_shape(receipt)
        runtime = receipt["runtime"]
        host_role = runtime["host_role"]
        if not isinstance(host_role, str) or not host_role:
            _fail("each cross-host receipt must have a non-empty host role")
        if host_role in seen_host_roles:
            _fail(f"duplicate host role in cross-host receipts: {host_role}")
        seen_host_roles.add(host_role)
        if not runtime["artifacts"]:
            _fail(f"host receipt has no explicit runtime artifacts: {host_role}")
        artifact_roles = [str(row["role"]) for row in runtime["artifacts"]]
        if expected_artifact_roles is None:
            expected_artifact_roles = artifact_roles
        elif artifact_roles != expected_artifact_roles:
            _fail("host receipts do not attest the same runtime artifact roles")
        if expected_run_identity is None:
            expected_run_identity = receipt["run_identity"]
        elif receipt["run_identity"] != expected_run_identity:
            _fail("host receipts bind different formal run identities")
        rows.append(
            {
                "host_role": host_role,
                "private_receipt_sha256": sha256_bytes(raw),
                "source_tree_sha256": receipt["source"]["inventory"]["tree_sha256"],
                "runtime_artifacts": [
                    {
                        "role": artifact["role"],
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                    }
                    for artifact in runtime["artifacts"]
                ],
            }
        )
    rows.sort(key=lambda row: str(row["host_role"]).encode("utf-8"))
    source_digests = {str(row["source_tree_sha256"]) for row in rows}
    runtime_signatures = {
        tuple(
            (artifact["role"], artifact["sha256"], artifact["size_bytes"])
            for artifact in row["runtime_artifacts"]
        )
        for row in rows
    }
    result = {
        "schema_version": RUNTIME_COMPARISON_SCHEMA,
        "status": "PASS" if len(source_digests) == 1 and len(runtime_signatures) == 1 else "MISMATCH",
        "run_identity": expected_run_identity,
        "host_count": len(rows),
        "host_roles": [row["host_role"] for row in rows],
        "runtime_artifact_roles": expected_artifact_roles,
        "all_source_trees_match": len(source_digests) == 1,
        "all_runtime_artifacts_match": len(runtime_signatures) == 1,
        "hosts": rows,
        "claim_boundary": {
            "supported": (
                "The listed retrospective host receipts bind identical source/runtime "
                "bytes exactly when the two all_*_match fields are true."
            ),
            "not_supported": (
                "Cross-host byte equality does not prove that the compiled extension was "
                "loaded by every earlier worker unless contemporaneous execution evidence "
                "independently identifies that artifact."
            ),
        },
    }
    _privacy_audit(result)
    return result


def verify_read_only_source_snapshot(source_root: Path) -> None:
    """Require no write bits on every path inside the included trust boundary."""

    source_root = Path(source_root).resolve()
    # Bytecode caches are excluded from the portable source inventory, but an
    # interpreter can still execute them.  A formal main snapshot must therefore
    # contain no executable cache outside the hashed trust boundary.  Main
    # launchers also set PYTHONDONTWRITEBYTECODE=1 so this remains true.
    bytecode_cache: list[str] = []
    for directory, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        for name in directory_names:
            if name == "__pycache__":
                bytecode_cache.append(
                    (current / name).relative_to(source_root).as_posix()
                )
        for name in file_names:
            if Path(name).suffix.lower() in {".pyc", ".pyo"}:
                bytecode_cache.append(
                    (current / name).relative_to(source_root).as_posix()
                )
    if bytecode_cache:
        preview = ", ".join(sorted(bytecode_cache)[:5])
        _fail(
            "formal main NBFNet snapshot contains executable bytecode outside "
            f"the source inventory ({len(bytecode_cache)} paths; first: {preview})"
        )

    writable: list[str] = []
    paths: list[tuple[str, Path]] = [(".", source_root)]
    for directory, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        retained: list[str] = []
        for name in _portable_sort(directory_names):
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            retained.append(name)
            candidate = current / name
            paths.append((candidate.relative_to(source_root).as_posix(), candidate))
        directory_names[:] = retained
        for name in _portable_sort(file_names):
            if _excluded_file(name):
                continue
            candidate = current / name
            paths.append((candidate.relative_to(source_root).as_posix(), candidate))
    for relative, path in paths:
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise SourceAttestationError(
                f"cannot inspect frozen source permissions for {relative}: {exc}"
            ) from exc
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable.append(relative)
    if writable:
        preview = ", ".join(writable[:5])
        _fail(
            "formal main NBFNet snapshot still has filesystem write bits "
            f"({len(writable)} paths; first: {preview})"
        )


def _require_below(path: Path, root: Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{role} must live below the isolated RUN_ROOT")
    return resolved


def formal_gate(
    *,
    receipt_path: Path,
    identity_inputs: RunIdentityInputs,
    run_root: Path,
    source_root: Path,
    runtime_artifacts: Sequence[tuple[str, Path]],
    expected_runtime_sha256: Mapping[str, str],
    host_role: str,
    source_peer_receipts: Sequence[Path],
    runtime_peer_receipts: Sequence[Path],
) -> dict[str, Any]:
    """Fail-closed pre/post-chain gate for the genuinely frozen main snapshot."""

    isolated_root = Path(run_root).expanduser().resolve()
    if not isolated_root.is_dir():
        _fail("formal RUN_ROOT is missing")
    selected_source = Path(source_root).expanduser().resolve()
    expected_source = (isolated_root / "private" / "nbfnet_source_frozen").resolve()
    if selected_source != expected_source:
        _fail("formal main NBFNET_PATH is not RUN_ROOT/private/nbfnet_source_frozen")
    _require_below(receipt_path, isolated_root, "formal source receipt")
    _require_below(identity_inputs.frozen_manifest, isolated_root, "frozen manifest")
    _require_below(identity_inputs.step3_manifest, isolated_root, "Step-3 manifest")
    for _, artifact_path in runtime_artifacts:
        resolved_artifact = _require_below(
            artifact_path, isolated_root, "formal runtime artifact"
        )
        expected_runtime_root = (isolated_root / "torch_extensions").resolve()
        try:
            resolved_artifact.relative_to(expected_runtime_root)
        except ValueError:
            _fail("formal runtime artifact is outside RUN_ROOT/torch_extensions")
    if not source_peer_receipts:
        _fail("formal gate requires at least one external/frozen source peer receipt")
    if runtime_artifacts and not runtime_peer_receipts:
        _fail("formal gate requires a peer-host runtime receipt")

    verify_read_only_source_snapshot(selected_source)
    receipt = verify_private_receipt(
        receipt_path,
        identity_inputs,
        source_root=selected_source,
        runtime_artifacts=runtime_artifacts,
        host_role=host_role,
        expected_runtime_sha256=expected_runtime_sha256,
    )
    if receipt["runtime"]["observed_hostname"] != socket.gethostname():
        _fail("formal host is not the host that created the selected private receipt")
    source_comparison = compare_source_receipts(
        [receipt_path, *source_peer_receipts]
    )
    if source_comparison["status"] != "PASS":
        _fail("external selection tree and frozen main source snapshot do not match")
    runtime_comparison: dict[str, Any] | None = None
    if runtime_artifacts:
        runtime_comparison = compare_runtime_receipts(
            [receipt_path, *runtime_peer_receipts]
        )
        if runtime_comparison["status"] != "PASS":
            _fail("formal runtime artifacts do not match across host receipts")
    return {
        "status": "PASS",
        "run_id": identity_inputs.run_id,
        "host_role": host_role,
        "source_tree_sha256": receipt["source"]["inventory"]["tree_sha256"],
        "source_peer_count": len(source_peer_receipts),
        "all_source_trees_match": True,
        "runtime_artifact_count": len(runtime_artifacts),
        "runtime_peer_count": len(runtime_peer_receipts),
        "all_runtime_artifacts_match": (
            True if runtime_comparison is not None else None
        ),
        "source_snapshot_mode_read_only": True,
    }


def atomic_create(path: Path, content: bytes) -> None:
    """Publish complete bytes atomically and refuse every existing target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SourceAttestationError(f"refusing to overwrite existing receipt: {path}") from exc
        except OSError as exc:
            raise SourceAttestationError(
                f"cannot atomically publish no-overwrite receipt {path}: {exc}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _identity_from_args(args: argparse.Namespace) -> RunIdentityInputs:
    return RunIdentityInputs(
        run_id=args.run_id,
        frozen_manifest=args.frozen_manifest,
        frozen_manifest_sha256=args.frozen_manifest_sha256,
        step3_manifest=args.step3_manifest,
        step3_manifest_sha256=args.step3_manifest_sha256,
    )


def _runtime_artifacts_from_args(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            _fail("--runtime-artifact must use ROLE=PATH syntax")
        _validate_public_role(role, "runtime artifact role")
        result.append((role, Path(raw_path)))
    return result


def _expected_runtime_sha256_from_args(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        role, separator, digest = value.partition("=")
        if not separator or not role or not digest:
            _fail("--expected-runtime-sha256 must use ROLE=SHA256 syntax")
        _validate_public_role(role, "runtime artifact role")
        _validate_expected_digest(digest, f"expected runtime SHA-256 for {role}")
        if role in result:
            _fail(f"duplicate expected runtime SHA-256 role: {role}")
        result[role] = digest
    return result


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--frozen-manifest-sha256", required=True)
    parser.add_argument("--step3-manifest", required=True, type=Path)
    parser.add_argument("--step3-manifest-sha256", required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="optional assertion; must resolve exactly to the current NBFNET_PATH",
    )
    parser.add_argument(
        "--runtime-artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "repeatable explicit runtime binary (for example "
            "rspmm-extension=/.../rspmm.so)"
        ),
    )
    parser.add_argument(
        "--host-role",
        help="non-identifying public host role; required with --runtime-artifact",
    )
    parser.add_argument(
        "--expected-runtime-sha256",
        action="append",
        default=[],
        metavar="ROLE=SHA256",
        help="repeatable externally expected digest for an attested runtime artifact",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="atomically create private evidence")
    create.add_argument("--output", required=True, type=Path)
    create.add_argument(
        "--selection-started-at-utc",
        help="externally recorded timestamp used only for an mtime consistency fact",
    )
    _add_identity_arguments(create)

    verify = subparsers.add_parser("verify", help="verify private evidence")
    verify.add_argument("--receipt", required=True, type=Path)
    _add_identity_arguments(verify)

    gate = subparsers.add_parser(
        "formal-gate",
        help="verify the read-only main snapshot, runtime, and peer receipts",
    )
    gate.add_argument("--receipt", required=True, type=Path)
    gate.add_argument("--run-root", required=True, type=Path)
    gate.add_argument(
        "--source-peer-receipt", action="append", required=True, type=Path
    )
    gate.add_argument(
        "--runtime-peer-receipt", action="append", default=[], type=Path
    )
    _add_identity_arguments(gate)

    project = subparsers.add_parser(
        "project-public", help="create a path-redacted public receipt"
    )
    project.add_argument("--private-receipt", required=True, type=Path)
    project.add_argument("--output", required=True, type=Path)

    check_public = subparsers.add_parser(
        "verify-public", help="verify public projection against private evidence"
    )
    check_public.add_argument("--private-receipt", required=True, type=Path)
    check_public.add_argument("--public-receipt", required=True, type=Path)

    compare_source = subparsers.add_parser(
        "compare-source", help="compare complete source trees across receipts"
    )
    compare_source.add_argument(
        "--private-receipt", action="append", required=True, type=Path
    )
    compare_source.add_argument("--output", required=True, type=Path)

    compare = subparsers.add_parser(
        "compare-runtime", help="compare source/runtime hashes across host receipts"
    )
    compare.add_argument(
        "--private-receipt", action="append", required=True, type=Path
    )
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            runtime_artifacts = _runtime_artifacts_from_args(args.runtime_artifact)
            expected_runtime_sha256 = _expected_runtime_sha256_from_args(
                args.expected_runtime_sha256
            )
            receipt = build_private_receipt(
                _identity_from_args(args),
                source_root=args.source_root,
                runtime_artifacts=runtime_artifacts,
                expected_runtime_sha256=expected_runtime_sha256,
                host_role=args.host_role,
                selection_started_at_utc=args.selection_started_at_utc,
            )
            atomic_create(args.output, render_json(receipt))
            action = f"created private receipt with {receipt['source']['inventory']['file_count']} files"
        elif args.command == "verify":
            runtime_artifacts = _runtime_artifacts_from_args(args.runtime_artifact)
            expected_runtime_sha256 = _expected_runtime_sha256_from_args(
                args.expected_runtime_sha256
            )
            receipt = verify_private_receipt(
                args.receipt,
                _identity_from_args(args),
                source_root=args.source_root,
                runtime_artifacts=(runtime_artifacts or None),
                host_role=args.host_role,
                expected_runtime_sha256=(expected_runtime_sha256 or None),
            )
            action = f"verified private receipt with {receipt['source']['inventory']['file_count']} files"
        elif args.command == "formal-gate":
            pending_paths = [
                args.receipt,
                *args.source_peer_receipt,
                *args.runtime_peer_receipt,
            ]
            if any(not path.is_file() for path in pending_paths):
                missing = [str(path) for path in pending_paths if not path.is_file()]
                print(
                    "NBFNET SOURCE ATTESTATION PENDING: missing receipt(s): "
                    + ", ".join(missing),
                    file=sys.stderr,
                )
                return 75
            runtime_artifacts = _runtime_artifacts_from_args(args.runtime_artifact)
            expected_runtime_sha256 = _expected_runtime_sha256_from_args(
                args.expected_runtime_sha256
            )
            if args.source_root is None:
                _fail("formal-gate requires --source-root")
            if args.host_role is None:
                _fail("formal-gate requires --host-role")
            gate_result = formal_gate(
                receipt_path=args.receipt,
                identity_inputs=_identity_from_args(args),
                run_root=args.run_root,
                source_root=args.source_root,
                runtime_artifacts=runtime_artifacts,
                expected_runtime_sha256=expected_runtime_sha256,
                host_role=args.host_role,
                source_peer_receipts=args.source_peer_receipt,
                runtime_peer_receipts=args.runtime_peer_receipt,
            )
            action = (
                "verified formal NBFNet gate: "
                f"source_match={gate_result['all_source_trees_match']} "
                f"runtime_match={gate_result['all_runtime_artifacts_match']}"
            )
        elif args.command == "project-public":
            receipt = project_public_receipt(args.private_receipt)
            atomic_create(args.output, render_json(receipt))
            action = "created path-redacted public receipt"
        elif args.command == "verify-public":
            verify_public_projection(args.public_receipt, args.private_receipt)
            action = "verified public receipt"
        elif args.command == "compare-source":
            comparison = compare_source_receipts(args.private_receipt)
            atomic_create(args.output, render_json(comparison))
            action = (
                "created source-tree comparison: "
                f"source_match={comparison['all_source_trees_match']}"
            )
        elif args.command == "compare-runtime":
            comparison = compare_runtime_receipts(args.private_receipt)
            atomic_create(args.output, render_json(comparison))
            action = (
                "created cross-host runtime comparison: "
                f"source_match={comparison['all_source_trees_match']} "
                f"runtime_match={comparison['all_runtime_artifacts_match']}"
            )
        else:  # pragma: no cover - argparse owns the command choices
            raise AssertionError(args.command)
    except SourceAttestationError as exc:
        print(f"NBFNET SOURCE ATTESTATION REFUSED: {exc}", file=sys.stderr)
        return 2
    print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
