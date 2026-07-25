"""Fail-closed replacement transaction for six derived v2 public artifacts.

The scientific generators intentionally refuse to overwrite an existing LOCO
or ULTRA JSON/CSV pair, while the benchmark-profile generator writes its JSON
and TeX sequentially.  This tool keeps those policies intact: candidate bytes
must first be generated below a dedicated staging root, are verified there,
and are then promoted under an active release hold.

The profile pair has an additional non-substitutable gate: prepare requires
the dataset summary, graph freeze, worker claims, GPU metrics/selections and
inventory, ULTRA formal tree, and ULTRA orchestration receipts; inventories
every input; independently rebuilds the profile in full mode; and
exact-compares both staged bytes.  Apply
repeats the inventories and full rebuild before and after replacement.
LOCO and ULTRA are likewise rebuilt byte-for-byte from their sealed formal
trees; their public-only verifiers are secondary checks, not trust anchors.

Ordinary filesystems cannot atomically replace six directory entries as one
operation.  Publication safety therefore comes from four jointly required
properties: the unresolved registry-audit hold remains byte-bound throughout
the transaction, snapshots and a write-ahead journal are durable before the
first replacement, every caught failure rolls back byte-for-byte, and an
interrupted ``APPLYING`` transaction must be recovered before another writer
can proceed.

This tool never generates paper numbers and never resolves the release hold.
Those actions remain owned by ``tools/resolve_v2_invalidation.py`` after all
fixed dependencies have been promoted and independently verified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_SCHEMA = "upgrade-bench-v2/canonical-replacement-transaction/1"
BASELINE_SCHEMA = "upgrade-bench-v2/canonical-replacement-baseline/1"
LOCK_SCHEMA = "upgrade-bench-v2/canonical-replacement-lock/2"

HOLD_ROLE = "results_v2/metrics/INVALIDATED.json"
REQUIRED_HOLD_STATUS = "INVALIDATED_REGISTRY_AUDIT"
LOCK_ROLE = "results_v2/.canonical_replacement_v2.lock"
# Persistent, gitignored operational evidence.  This is deliberately not the
# generic scratch ``tmp/`` tree: snapshots/journals must survive until the
# downstream release-hold resolution and an explicit archival decision.
TRANSACTION_ROOT_ROLE = "private/canonical_replacement_transactions"

ROLES = (
    "results_v2/metrics/v2_loco_transfer_summary.json",
    "results_v2/metrics/v2_loco_transfer_summary.csv",
    "results_v2/metrics/v2_ultra_zero_shot_summary.json",
    "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
    "results_v2/metrics/v2_benchmark_profile.json",
    "paper/generated/v2_benchmark_profile.tex",
)

LOCO_ROLES = ROLES[0:2]
ULTRA_ROLES = ROLES[2:4]
PROFILE_ROLES = ROLES[4:6]

# These bytes define or validate the staged public interfaces.  They are
# rebound at prepare time and checked again immediately before and after the
# canonical replacements.
BOUND_SOURCE_ROLES = (
    "tools/promote_v2_canonical_replacements.py",
    "tools/summarize_v2_loco_results.py",
    "configs/v2_loco_formal.json",
    "tools/v2_loco_formal.py",
    "tools/summarize_v2_ultra_results.py",
    "configs/v2_ultra_formal.json",
    "tools/v2_ultra_formal.py",
    "tools/generate_v2_benchmark_profile.py",
)

STATUSES = frozenset(
    {"PREPARED", "APPLYING", "PROMOTED", "ROLLED_BACK", "RECOVERY_REQUIRED"}
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TRANSACTION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
UTC_SECONDS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)

Validator = Callable[[Path, Path], None]
ReplaceFunction = Callable[[Path, Path], None]
ProfileGate = Callable[[Path, Path, Mapping[str, Path]], None]
LocoGate = Callable[[Path, Path, Path], None]
UltraGate = Callable[[Path, Path, Path], None]

CANONICAL_LOCO_FORMAL_ROLE = "results_v2/loco_formal"

PROFILE_INPUT_NAMES = (
    "dataset_summary",
    "freeze_manifest",
    "claims_dir",
    "metrics_dir",
    "selections_dir",
    "gpu_inventory",
    "ultra_dir",
    "ultra_receipts_dir",
)
PROFILE_FILE_INPUTS = frozenset(
    {"dataset_summary", "freeze_manifest", "gpu_inventory"}
)
PROFILE_DIRECTORY_INPUTS = frozenset(
    {
        "claims_dir",
        "metrics_dir",
        "selections_dir",
        "ultra_dir",
        "ultra_receipts_dir",
    }
)


class TransactionError(RuntimeError):
    """A replacement cannot proceed without weakening a required guard."""


def _fail(message: str) -> None:
    raise TransactionError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_bytes(raw: bytes, role: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"{role} is not strict UTF-8 JSON: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{role} must contain one JSON object")
    return value


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_info(content: bytes) -> dict[str, Any]:
    return {"sha256": _sha256_bytes(content), "size": len(content)}


def _require_hash(value: Any, role: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        _fail(f"{role} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{role} must be an object")
    if set(value) != expected:
        _fail(
            f"{role} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _require_info(value: Any, role: str) -> dict[str, Any]:
    row = _require_exact_keys(value, {"sha256", "size"}, role)
    digest = _require_hash(row["sha256"], f"{role}.sha256")
    size = row["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        _fail(f"{role}.size must be a nonnegative integer")
    return {"sha256": digest, "size": size}


def _canonical_root(root: Path) -> Path:
    lexical = Path(root).absolute()
    if lexical.is_symlink():
        _fail("repository root must not be a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"repository root cannot be resolved: {exc}")
    if not resolved.is_dir():
        _fail("repository root must be a directory")
    return resolved


def _role_parts(role: str) -> tuple[str, ...]:
    path = PurePosixPath(role)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in role:
        raise AssertionError(f"unsafe built-in repository role: {role!r}")
    return tuple(path.parts)


def _role_path(base: Path, role: str) -> Path:
    return Path(base).joinpath(*_role_parts(role))


def _assert_physical_directory(path: Path, role: str, *, create: bool = False) -> Path:
    lexical = Path(path).absolute()
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    if lexical.is_symlink():
        _fail(f"{role} must not be a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"{role} cannot be resolved: {exc}")
    if resolved != lexical.resolve(strict=False) or not resolved.is_dir():
        _fail(f"{role} must be one physical directory")
    return resolved


def _assert_parent_chain(base: Path, path: Path, role: str) -> None:
    base = Path(base).resolve(strict=True)
    parent = path.parent.absolute()
    try:
        relative = parent.relative_to(base)
    except ValueError:
        _fail(f"{role} escapes its allowed root")
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{role} traverses a symbolic-link directory")
        if not current.is_dir():
            _fail(f"{role} parent is not a directory")


def _read_regular(path: Path, role: str, *, base: Path | None = None) -> bytes:
    path = Path(path)
    if base is not None:
        _assert_parent_chain(base, path, role)
    if path.is_symlink() or not path.is_file():
        _fail(f"{role} must be a regular non-symbolic-link file")
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        _fail(f"cannot read {role}: {exc}")
    if before.st_nlink != 1 or after.st_nlink != 1:
        _fail(f"{role} must not have hard-link aliases")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail(f"{role} changed while it was being read")
    return content


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_replace_bytes(path: Path, content: bytes, prefix: str) -> None:
    if path.is_symlink():
        _fail(f"refusing to replace symbolic-link file {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _stage_same_directory(target: Path, content: bytes, label: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.{label}-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return temporary


def _replace_path(source: Path, target: Path) -> None:
    """Patch point for deterministic failure-injection tests."""
    os.replace(source, target)


def _inventory_digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    normalized = {
        role: {
            "sha256": _require_hash(info["sha256"], f"inventory {role}.sha256"),
            "size": info["size"],
        }
        for role, info in sorted(files.items())
    }
    return _sha256_bytes(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _bundle_infos(base: Path, *, exact_inventory: bool) -> dict[str, dict[str, Any]]:
    base = _assert_physical_directory(base, "artifact bundle")
    if exact_inventory:
        observed: set[str] = set()
        for directory, directories, filenames in os.walk(base, followlinks=False):
            directory_path = Path(directory)
            for name in list(directories):
                child = directory_path / name
                if child.is_symlink():
                    _fail("artifact bundle contains a symbolic-link directory")
            for name in filenames:
                child = directory_path / name
                if child.is_symlink() or not child.is_file():
                    _fail("artifact bundle contains a non-regular file")
                observed.add(child.relative_to(base).as_posix())
        if observed != set(ROLES):
            _fail(
                "artifact bundle inventory differs from the fixed six-role scope: "
                f"missing={sorted(set(ROLES) - observed)}, "
                f"extra={sorted(observed - set(ROLES))}"
            )
    result: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        content = _read_regular(_role_path(base, role), role, base=base)
        result[role] = _file_info(content)
    return result


def _require_repository_relative_role(value: Any, role: str) -> str:
    if not isinstance(value, str):
        _fail(f"{role} must be a repository-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
        or value.startswith("./")
    ):
        _fail(f"{role} must be a normalized repository-relative path")
    if path.as_posix() != value:
        _fail(f"{role} is not a normalized POSIX path")
    return value


def _path_below_root(root: Path, path: Path, role: str, *, directory: bool) -> tuple[Path, str]:
    root = _canonical_root(root)
    lexical = Path(path).absolute()
    if lexical.is_symlink():
        _fail(f"profile input {role} must not be a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"profile input {role} cannot be resolved: {exc}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        _fail(f"profile input {role} must be below the repository root")
    if not relative or relative == ".":
        _fail(f"profile input {role} cannot be the repository root")
    if directory:
        if not resolved.is_dir():
            _fail(f"profile input {role} must be a directory")
    elif not resolved.is_file():
        _fail(f"profile input {role} must be a regular file")
    return resolved, _require_repository_relative_role(relative, f"profile input {role}")


def _directory_inventory(path: Path, role: str) -> dict[str, dict[str, Any]]:
    root = _assert_physical_directory(path, f"profile input {role}")
    result: dict[str, dict[str, Any]] = {}
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directories):
            child = directory_path / name
            if child.is_symlink():
                _fail(f"profile input {role} contains a symbolic-link directory")
        for name in filenames:
            child = directory_path / name
            if child.is_symlink() or not child.is_file():
                _fail(f"profile input {role} contains a non-regular file")
            relative = child.relative_to(root).as_posix()
            _require_repository_relative_role(relative, f"profile input {role} member")
            result[relative] = _file_info(
                _read_regular(child, f"profile input {role}/{relative}", base=root)
            )
    if not result:
        _fail(f"profile input {role} directory is empty")
    return dict(sorted(result.items()))


def _profile_input_inventory(
    root: Path, inputs: Mapping[str, Path]
) -> tuple[dict[str, Any], dict[str, Path]]:
    if set(inputs) != set(PROFILE_INPUT_NAMES):
        _fail(
            "profile input names differ: "
            f"missing={sorted(set(PROFILE_INPUT_NAMES) - set(inputs))}, "
            f"extra={sorted(set(inputs) - set(PROFILE_INPUT_NAMES))}"
        )
    inventory: dict[str, Any] = {}
    resolved_paths: dict[str, Path] = {}
    for name in PROFILE_INPUT_NAMES:
        if name in PROFILE_FILE_INPUTS:
            resolved, repository_role = _path_below_root(
                root, Path(inputs[name]), name, directory=False
            )
            content = _read_regular(resolved, f"profile input {name}", base=root)
            inventory[name] = {
                "kind": "file",
                "role": repository_role,
                **_file_info(content),
            }
        else:
            resolved, repository_role = _path_below_root(
                root, Path(inputs[name]), name, directory=True
            )
            files = _directory_inventory(resolved, name)
            inventory[name] = {
                "kind": "directory",
                "role": repository_role,
                "file_count": len(files),
                "inventory_sha256": _inventory_digest(files),
                "files": files,
            }
        resolved_paths[name] = resolved
    return inventory, resolved_paths


def _validate_profile_input_inventory(value: Any) -> dict[str, Any]:
    rows = _require_exact_keys(value, set(PROFILE_INPUT_NAMES), "profile_inputs")
    normalized: dict[str, Any] = {}
    for name in PROFILE_INPUT_NAMES:
        row = rows[name]
        if not isinstance(row, Mapping):
            _fail(f"profile_inputs.{name} must be an object")
        kind = row.get("kind")
        if name in PROFILE_FILE_INPUTS:
            item = _require_exact_keys(
                row, {"kind", "role", "sha256", "size"}, f"profile_inputs.{name}"
            )
            if kind != "file":
                _fail(f"profile_inputs.{name}.kind must be file")
            normalized[name] = {
                "kind": "file",
                "role": _require_repository_relative_role(
                    item["role"], f"profile_inputs.{name}.role"
                ),
                **_require_info(
                    {"sha256": item["sha256"], "size": item["size"]},
                    f"profile_inputs.{name}",
                ),
            }
        elif name in PROFILE_DIRECTORY_INPUTS:
            item = _require_exact_keys(
                row,
                {
                    "kind",
                    "role",
                    "file_count",
                    "inventory_sha256",
                    "files",
                },
                f"profile_inputs.{name}",
            )
            if kind != "directory":
                _fail(f"profile_inputs.{name}.kind must be directory")
            files_raw = item["files"]
            if not isinstance(files_raw, Mapping) or not files_raw:
                _fail(f"profile_inputs.{name}.files must be a nonempty object")
            files: dict[str, dict[str, Any]] = {}
            for relative, info in files_raw.items():
                normalized_relative = _require_repository_relative_role(
                    relative, f"profile_inputs.{name}.files key"
                )
                files[normalized_relative] = _require_info(
                    info, f"profile_inputs.{name}.files.{normalized_relative}"
                )
            file_count = item["file_count"]
            if (
                isinstance(file_count, bool)
                or not isinstance(file_count, int)
                or file_count != len(files)
            ):
                _fail(f"profile_inputs.{name}.file_count mismatch")
            inventory_sha = _require_hash(
                item["inventory_sha256"], f"profile_inputs.{name}.inventory_sha256"
            )
            if inventory_sha != _inventory_digest(files):
                _fail(f"profile_inputs.{name}.inventory_sha256 mismatch")
            normalized[name] = {
                "kind": "directory",
                "role": _require_repository_relative_role(
                    item["role"], f"profile_inputs.{name}.role"
                ),
                "file_count": file_count,
                "inventory_sha256": inventory_sha,
                "files": dict(sorted(files.items())),
            }
        else:  # pragma: no cover - constants partition the exact name set
            raise AssertionError(name)
    roles = [normalized[name]["role"] for name in PROFILE_INPUT_NAMES]
    if len(set(roles)) != len(roles):
        _fail("profile inputs must have distinct repository roles")
    return normalized


def _profile_paths_from_journal(
    root: Path, journal: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        name: _role_path(root, journal["profile_inputs"][name]["role"])
        for name in PROFILE_INPUT_NAMES
    }
    observed, resolved = _profile_input_inventory(root, paths)
    if observed != dict(journal["profile_inputs"]):
        _fail("one or more formal benchmark-profile inputs changed after preparation")
    return observed, resolved


def _loco_input_inventory(root: Path, loco_dir: Path) -> tuple[dict[str, Any], Path]:
    resolved, repository_role = _path_below_root(
        root, Path(loco_dir), "loco_formal_dir", directory=True
    )
    if repository_role != CANONICAL_LOCO_FORMAL_ROLE:
        _fail(
            "LOCO formal input must be the canonical results_v2/loco_formal tree "
            "required by the live verifier"
        )
    files = _directory_inventory(resolved, "loco_formal_dir")
    return (
        {
            "kind": "directory",
            "role": repository_role,
            "file_count": len(files),
            "inventory_sha256": _inventory_digest(files),
            "files": files,
        },
        resolved,
    )


def _validate_loco_input_inventory(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        {"kind", "role", "file_count", "inventory_sha256", "files"},
        "loco_formal_input",
    )
    if row["kind"] != "directory" or row["role"] != CANONICAL_LOCO_FORMAL_ROLE:
        _fail("loco_formal_input must bind the canonical formal directory")
    files_raw = row["files"]
    if not isinstance(files_raw, Mapping) or not files_raw:
        _fail("loco_formal_input.files must be a nonempty object")
    files: dict[str, dict[str, Any]] = {}
    for relative, info in files_raw.items():
        normalized_relative = _require_repository_relative_role(
            relative, "loco_formal_input.files key"
        )
        files[normalized_relative] = _require_info(
            info, f"loco_formal_input.files.{normalized_relative}"
        )
    file_count = row["file_count"]
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(files)
    ):
        _fail("loco_formal_input.file_count mismatch")
    digest = _require_hash(
        row["inventory_sha256"], "loco_formal_input.inventory_sha256"
    )
    if digest != _inventory_digest(files):
        _fail("loco_formal_input.inventory_sha256 mismatch")
    return {
        "kind": "directory",
        "role": CANONICAL_LOCO_FORMAL_ROLE,
        "file_count": file_count,
        "inventory_sha256": digest,
        "files": dict(sorted(files.items())),
    }


def _loco_path_from_journal(
    root: Path, journal: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    path = _role_path(root, journal["loco_formal_input"]["role"])
    observed, resolved = _loco_input_inventory(root, path)
    if observed != dict(journal["loco_formal_input"]):
        _fail("LOCO formal input tree changed after preparation")
    return observed, resolved


def _load_module(path: Path, purpose: str) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        _fail(f"{purpose} is missing or not a regular file")
    name = f"_upgrade_bench_replacement_{purpose}_{_sha256_bytes(str(path).encode())[:12]}"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        _fail(f"cannot load {purpose}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    origin = Path(getattr(module, "__file__", "")).resolve()
    if origin != path.resolve():
        _fail(f"{purpose} did not load from the bound repository path")
    return module


def validate_candidate_bundle(root: Path, bundle_root: Path) -> None:
    """Run each canonical public verifier against an arbitrary six-file root."""
    root = _canonical_root(root)
    bundle_root = _assert_physical_directory(bundle_root, "candidate bundle")
    loco = _load_module(root / "tools/summarize_v2_loco_results.py", "loco")
    ultra = _load_module(root / "tools/summarize_v2_ultra_results.py", "ultra")
    profile = _load_module(
        root / "tools/generate_v2_benchmark_profile.py", "benchmark_profile"
    )
    try:
        loco.verify_outputs(
            _role_path(bundle_root, LOCO_ROLES[0]),
            _role_path(bundle_root, LOCO_ROLES[1]),
        )
        ultra.verify_outputs(
            _role_path(bundle_root, ULTRA_ROLES[0]),
            _role_path(bundle_root, ULTRA_ROLES[1]),
        )
        profile.verify_outputs(
            _role_path(bundle_root, PROFILE_ROLES[0]),
            _role_path(bundle_root, PROFILE_ROLES[1]),
            mode="repository",
        )
    except Exception as exc:
        raise TransactionError(
            f"staged public-artifact validation failed ({type(exc).__name__})"
        ) from exc


def validate_full_loco_candidate(
    root: Path, bundle_root: Path, loco_formal_dir: Path
) -> None:
    """Rebuild the LOCO public pair through its live formal gate."""
    root = _canonical_root(root)
    bundle_root = _assert_physical_directory(bundle_root, "candidate bundle")
    loco_formal_dir = _assert_physical_directory(
        loco_formal_dir, "LOCO formal input"
    )
    module = _load_module(
        root / "tools/summarize_v2_loco_results.py", "loco_full"
    )
    try:
        summary = module.build_summary(
            loco_formal_dir / "summary.json",
            loco_formal_dir / "verification_receipt.json",
        )
        json_bytes = module.render_json(summary)
        csv_bytes = module.render_csv(summary)
    except Exception as exc:
        raise TransactionError(
            f"formal LOCO public-pair rebuild failed ({type(exc).__name__})"
        ) from exc
    observed_json = _read_regular(
        _role_path(bundle_root, LOCO_ROLES[0]), "staged LOCO JSON", base=bundle_root
    )
    observed_csv = _read_regular(
        _role_path(bundle_root, LOCO_ROLES[1]), "staged LOCO CSV", base=bundle_root
    )
    if observed_json != json_bytes or observed_csv != csv_bytes:
        _fail("staged LOCO pair differs from the live formal rebuild")


def validate_full_ultra_candidate(
    root: Path, bundle_root: Path, ultra_formal_dir: Path
) -> None:
    """Rebuild the ULTRA public pair from the exact sealed formal tree."""
    root = _canonical_root(root)
    bundle_root = _assert_physical_directory(bundle_root, "candidate bundle")
    ultra_formal_dir = _assert_physical_directory(
        ultra_formal_dir, "ULTRA formal input"
    )
    module = _load_module(
        root / "tools/summarize_v2_ultra_results.py", "ultra_full"
    )
    try:
        summary = module.build_summary(ultra_formal_dir)
        json_bytes = module.render_json(summary)
        csv_bytes = module.render_csv(summary)
    except Exception as exc:
        raise TransactionError(
            f"formal ULTRA public-pair rebuild failed ({type(exc).__name__})"
        ) from exc
    observed_json = _read_regular(
        _role_path(bundle_root, ULTRA_ROLES[0]), "staged ULTRA JSON", base=bundle_root
    )
    observed_csv = _read_regular(
        _role_path(bundle_root, ULTRA_ROLES[1]), "staged ULTRA CSV", base=bundle_root
    )
    if observed_json != json_bytes or observed_csv != csv_bytes:
        _fail("staged ULTRA pair differs from the sealed formal-tree rebuild")


def validate_full_profile_candidate(
    root: Path,
    bundle_root: Path,
    profile_inputs: Mapping[str, Path],
) -> None:
    """Rebuild the profile from formal inputs and compare exact staged bytes."""
    root = _canonical_root(root)
    bundle_root = _assert_physical_directory(bundle_root, "candidate bundle")
    if set(profile_inputs) != set(PROFILE_INPUT_NAMES):
        _fail("full profile gate did not receive the exact formal-input set")
    module = _load_module(
        root / "tools/generate_v2_benchmark_profile.py", "benchmark_profile_full"
    )
    arguments = argparse.Namespace(
        dataset_summary=Path(profile_inputs["dataset_summary"]),
        freeze_manifest=Path(profile_inputs["freeze_manifest"]),
        claims_dir=Path(profile_inputs["claims_dir"]),
        metrics_dir=Path(profile_inputs["metrics_dir"]),
        selections_dir=Path(profile_inputs["selections_dir"]),
        gpu_inventory=Path(profile_inputs["gpu_inventory"]),
        ultra_dir=Path(profile_inputs["ultra_dir"]),
        ultra_receipts_dir=Path(profile_inputs["ultra_receipts_dir"]),
    )
    try:
        profile = module.build_profile(arguments)
        module.validate_profile(profile, mode="full")
        json_bytes = (
            json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        tex_bytes = module.render_tex(profile, _sha256_bytes(json_bytes)).encode("utf-8")
    except Exception as exc:
        raise TransactionError(
            f"formal benchmark-profile rebuild failed ({type(exc).__name__})"
        ) from exc
    observed_json = _read_regular(
        _role_path(bundle_root, PROFILE_ROLES[0]),
        "staged benchmark profile JSON",
        base=bundle_root,
    )
    observed_tex = _read_regular(
        _role_path(bundle_root, PROFILE_ROLES[1]),
        "staged benchmark profile TeX",
        base=bundle_root,
    )
    if observed_json != json_bytes:
        _fail("staged benchmark profile JSON differs from the formal-input rebuild")
    if observed_tex != tex_bytes:
        _fail("staged benchmark profile TeX differs from the formal-input rebuild")
    ultra_raw = _read_regular(
        _role_path(bundle_root, ULTRA_ROLES[0]),
        "staged ULTRA JSON for profile cross-binding",
        base=bundle_root,
    )
    ultra = _strict_json_bytes(ultra_raw, "staged ULTRA JSON for profile cross-binding")
    try:
        profile_seal = profile["provenance"]["formal_evidence_sha256"][
            "ultra_score_seal_sha256"
        ]
        ultra_seal = ultra["provenance"]["score_seal_sha256"]
    except (KeyError, TypeError) as exc:
        raise TransactionError(
            "profile/ULTRA score-seal cross-binding fields are missing"
        ) from exc
    _require_hash(profile_seal, "rebuilt profile ULTRA score-seal binding")
    _require_hash(ultra_seal, "staged ULTRA score-seal provenance")
    if profile_seal != ultra_seal:
        _fail("benchmark profile and staged ULTRA pair bind different score seals")


def _bound_source_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for role in BOUND_SOURCE_ROLES:
        content = _read_regular(_role_path(root, role), role, base=root)
        result[role] = _sha256_bytes(content)
    return result


def _read_hold(root: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(_role_path(root, HOLD_ROLE), HOLD_ROLE, base=root)
    payload = dict(_strict_json_bytes(raw, "active release hold"))
    status = payload.get("status")
    if status != REQUIRED_HOLD_STATUS:
        _fail(
            "canonical replacement requires the active registry-audit hold; "
            f"observed status={status!r}"
        )
    return payload, raw


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    raw = _read_regular(Path(path).absolute(), "expected-before manifest")
    payload = _strict_json_bytes(raw, "expected-before manifest")
    top = _require_exact_keys(payload, {"schema_version", "files"}, "baseline")
    if top["schema_version"] != BASELINE_SCHEMA:
        _fail("expected-before manifest schema mismatch")
    files = _require_exact_keys(top["files"], set(ROLES), "baseline.files")
    return {
        role: _require_info(files[role], f"baseline.files.{role}") for role in ROLES
    }


def _current_canonical_infos(root: Path) -> dict[str, dict[str, Any]]:
    return _bundle_infos(root, exact_inventory=False)


def _plan_material(journal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": journal["schema_version"],
        "transaction_id": journal["transaction_id"],
        "scope": journal["scope"],
        "active_hold": journal["active_hold"],
        "files": journal["files"],
        "bound_sources": journal["bound_sources"],
        "profile_inputs": journal["profile_inputs"],
        "loco_formal_input": journal["loco_formal_input"],
        "before_inventory_sha256": journal["before_inventory_sha256"],
        "candidate_inventory_sha256": journal["candidate_inventory_sha256"],
    }


def _plan_digest(journal: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            _plan_material(journal), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


JOURNAL_FIELDS = {
    "schema_version",
    "transaction_id",
    "status",
    "created_at_utc",
    "updated_at_utc",
    "committed_at_utc",
    "rolled_back_at_utc",
    "recovered_at_utc",
    "scope",
    "active_hold",
    "files",
    "bound_sources",
    "profile_inputs",
    "loco_formal_input",
    "before_inventory_sha256",
    "candidate_inventory_sha256",
    "plan_sha256",
    "applied_roles",
    "error",
}


def _validate_journal(value: Mapping[str, Any]) -> dict[str, Any]:
    top = dict(_require_exact_keys(value, JOURNAL_FIELDS, "transaction journal"))
    if top["schema_version"] != TRANSACTION_SCHEMA:
        _fail("transaction journal schema mismatch")
    transaction_id = top["transaction_id"]
    if not isinstance(transaction_id, str) or TRANSACTION_ID.fullmatch(transaction_id) is None:
        _fail("transaction journal has an invalid transaction_id")
    if top["status"] not in STATUSES:
        _fail("transaction journal has an invalid status")
    for field in (
        "created_at_utc",
        "updated_at_utc",
        "committed_at_utc",
        "rolled_back_at_utc",
        "recovered_at_utc",
    ):
        timestamp = top[field]
        if timestamp is not None and (
            not isinstance(timestamp, str) or UTC_SECONDS.fullmatch(timestamp) is None
        ):
            _fail(f"transaction journal {field} is not UTC seconds")
    if top["created_at_utc"] is None or top["updated_at_utc"] is None:
        _fail("transaction journal lacks creation/update timestamps")
    if top["scope"] != list(ROLES):
        _fail("transaction journal scope is not the fixed six-role order")
    hold = _require_exact_keys(
        top["active_hold"], {"role", "sha256", "status"}, "transaction active_hold"
    )
    if hold["role"] != HOLD_ROLE or hold["status"] != REQUIRED_HOLD_STATUS:
        _fail("transaction journal release-hold binding is invalid")
    _require_hash(hold["sha256"], "transaction active_hold.sha256")
    files = _require_exact_keys(top["files"], set(ROLES), "transaction files")
    normalized_files: dict[str, Any] = {}
    for role in ROLES:
        row = _require_exact_keys(
            files[role], {"before", "candidate", "validator"}, f"transaction files.{role}"
        )
        validator = row["validator"]
        expected_validator = (
            "loco_public_verify"
            if role in LOCO_ROLES
            else "ultra_public_verify"
            if role in ULTRA_ROLES
            else "benchmark_profile_full_rebuild_and_repository_verify"
        )
        if validator != expected_validator:
            _fail(f"transaction files.{role}.validator mismatch")
        normalized_files[role] = {
            "before": _require_info(row["before"], f"transaction files.{role}.before"),
            "candidate": _require_info(
                row["candidate"], f"transaction files.{role}.candidate"
            ),
            "validator": validator,
        }
    sources = _require_exact_keys(
        top["bound_sources"], set(BOUND_SOURCE_ROLES), "transaction bound_sources"
    )
    for role in BOUND_SOURCE_ROLES:
        _require_hash(sources[role], f"transaction bound_sources.{role}")
    top["profile_inputs"] = _validate_profile_input_inventory(top["profile_inputs"])
    top["loco_formal_input"] = _validate_loco_input_inventory(
        top["loco_formal_input"]
    )
    before_digest = _require_hash(
        top["before_inventory_sha256"], "transaction before inventory"
    )
    candidate_digest = _require_hash(
        top["candidate_inventory_sha256"], "transaction candidate inventory"
    )
    if before_digest != _inventory_digest(
        {role: normalized_files[role]["before"] for role in ROLES}
    ):
        _fail("transaction before-inventory digest mismatch")
    if candidate_digest != _inventory_digest(
        {role: normalized_files[role]["candidate"] for role in ROLES}
    ):
        _fail("transaction candidate-inventory digest mismatch")
    applied = top["applied_roles"]
    if not isinstance(applied, list) or applied != list(ROLES[: len(applied)]):
        _fail("transaction applied_roles must be one ordered scope prefix")
    if top["error"] is not None and (
        not isinstance(top["error"], str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", top["error"]) is None
    ):
        _fail("transaction error must be a safe exception-class token or null")
    if top["plan_sha256"] != _plan_digest(top):
        _fail("transaction plan digest mismatch")
    if top["status"] == "PREPARED" and applied:
        _fail("PREPARED transaction cannot contain applied roles")
    if top["status"] == "PROMOTED":
        if applied != list(ROLES) or top["committed_at_utc"] is None:
            _fail("PROMOTED transaction is incomplete")
    if top["status"] == "ROLLED_BACK" and top["rolled_back_at_utc"] is None:
        _fail("ROLLED_BACK transaction lacks a rollback timestamp")
    return top


def _transaction_parent(root: Path) -> Path:
    parent = _role_path(root, TRANSACTION_ROOT_ROLE)
    return _assert_physical_directory(parent, "transaction root", create=True)


def _transaction_directory(root: Path, transaction_id: str) -> Path:
    if TRANSACTION_ID.fullmatch(transaction_id) is None:
        _fail("transaction_id must match [a-z0-9][a-z0-9._-]{0,79}")
    return _transaction_parent(root) / transaction_id


def _assert_no_other_unfinished_transaction(root: Path, transaction_id: str) -> None:
    parent = _transaction_parent(root)
    for child in sorted(parent.iterdir(), key=lambda path: path.name):
        if child.name == transaction_id or child.name.startswith("."):
            continue
        if TRANSACTION_ID.fullmatch(child.name) is None:
            _fail(f"unrecognized entry in transaction root: {child.name}")
        if child.is_symlink() or not child.is_dir():
            _fail(f"transaction-root entry is not a physical directory: {child.name}")
        _, other = _load_journal(root, child.name)
        if other["status"] in {"APPLYING", "RECOVERY_REQUIRED"}:
            _fail(
                "another canonical replacement requires recovery before any new apply: "
                f"{child.name} ({other['status']})"
            )


def _load_journal(root: Path, transaction_id: str) -> tuple[Path, dict[str, Any]]:
    directory = _transaction_directory(root, transaction_id)
    directory = _assert_physical_directory(directory, "transaction directory")
    journal_path = directory / "transaction.json"
    raw = _read_regular(journal_path, "transaction journal", base=directory)
    payload = _validate_journal(_strict_json_bytes(raw, "transaction journal"))
    if raw != _json_bytes(payload):
        _fail("transaction journal is not canonically encoded")
    if payload["transaction_id"] != transaction_id:
        _fail("transaction directory and journal identity differ")
    return directory, payload


def _write_journal(directory: Path, journal: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _validate_journal(journal)
    path = directory / "transaction.json"
    _atomic_replace_bytes(path, _json_bytes(normalized), ".transaction.json.write-")
    observed = _read_regular(path, "transaction journal", base=directory)
    if observed != _json_bytes(normalized):
        _fail("transaction journal write verification failed")
    return normalized


def _update_journal(
    directory: Path, journal: Mapping[str, Any], **changes: Any
) -> dict[str, Any]:
    updated = dict(journal)
    updated.update(changes)
    updated["updated_at_utc"] = _utc_now()
    # Plan material never includes mutable state, so this value must remain
    # stable across PREPARED/APPLYING/PROMOTED/rollback transitions.
    updated["plan_sha256"] = _plan_digest(updated)
    return _write_journal(directory, updated)


def _assert_infos_equal(
    observed: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    role: str,
) -> None:
    if dict(observed) != dict(expected):
        _fail(f"{role} bytes differ from the transaction binding")


def _journal_infos(journal: Mapping[str, Any], which: str) -> dict[str, dict[str, Any]]:
    return {role: dict(journal["files"][role][which]) for role in ROLES}


def _validate_transaction_storage(directory: Path, journal: Mapping[str, Any]) -> None:
    before_root = directory / "before"
    candidate_root = directory / "candidate"
    before = _bundle_infos(before_root, exact_inventory=True)
    candidate = _bundle_infos(candidate_root, exact_inventory=True)
    _assert_infos_equal(before, _journal_infos(journal, "before"), "snapshot inventory")
    _assert_infos_equal(
        candidate, _journal_infos(journal, "candidate"), "candidate inventory"
    )
    _validate_stored_hold(directory, journal)


def _validate_before_storage(directory: Path, journal: Mapping[str, Any]) -> None:
    """Validate only the bytes needed for a safety-critical restoration."""
    before = _bundle_infos(directory / "before", exact_inventory=True)
    _assert_infos_equal(before, _journal_infos(journal, "before"), "snapshot inventory")
    _validate_stored_hold(directory, journal)


def _validate_stored_hold(directory: Path, journal: Mapping[str, Any]) -> None:
    raw = _read_regular(directory / "active_hold.json", "stored active release hold", base=directory)
    payload = _strict_json_bytes(raw, "stored active release hold")
    if payload.get("status") != REQUIRED_HOLD_STATUS:
        _fail("stored release hold does not record the required active status")
    if _sha256_bytes(raw) != journal["active_hold"]["sha256"]:
        _fail("stored release-hold bytes differ from the journal binding")


def _validate_live_bindings(root: Path, journal: Mapping[str, Any]) -> None:
    _, hold_raw = _read_hold(root)
    if _sha256_bytes(hold_raw) != journal["active_hold"]["sha256"]:
        _fail("active release hold changed after transaction preparation")
    if _bound_source_hashes(root) != dict(journal["bound_sources"]):
        _fail("a bound generator, controller, or config changed after preparation")


def _validate_bound_sources(root: Path, journal: Mapping[str, Any]) -> None:
    if _bound_source_hashes(root) != dict(journal["bound_sources"]):
        _fail("a bound generator, controller, or config changed after preparation")


def _validate_hold_binding(root: Path, journal: Mapping[str, Any]) -> None:
    _, hold_raw = _read_hold(root)
    if _sha256_bytes(hold_raw) != journal["active_hold"]["sha256"]:
        _fail("active release hold changed after transaction preparation")


def _load_lock(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, "replacement writer lock")
    payload = _strict_json_bytes(raw, "replacement writer lock")
    top = _require_exact_keys(
        payload,
        {"schema_version", "transaction_id", "host", "pid", "created_at_utc"},
        "lock",
    )
    if top["schema_version"] != LOCK_SCHEMA:
        _fail("replacement writer lock schema mismatch")
    transaction_id = top["transaction_id"]
    if not isinstance(transaction_id, str) or TRANSACTION_ID.fullmatch(transaction_id) is None:
        _fail("replacement writer lock transaction_id is invalid")
    host = top["host"]
    if (
        not isinstance(host, str)
        or not host
        or len(host) > 255
        or any(ord(character) < 32 for character in host)
    ):
        _fail("replacement writer lock host identity is invalid")
    pid = top["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        _fail("replacement writer lock PID is invalid")
    if not isinstance(top["created_at_utc"], str) or UTC_SECONDS.fullmatch(
        top["created_at_utc"]
    ) is None:
        _fail("replacement writer lock timestamp is invalid")
    if raw != _json_bytes(top):
        _fail("replacement writer lock is not canonical")
    return dict(top), raw


def _pid_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Windows reports an invalid-parameter OSError for a nonexistent PID.
        return False
    return True


def _host_identity() -> str:
    host = socket.gethostname().strip()
    if not host or len(host) > 255 or any(ord(character) < 32 for character in host):
        _fail("local host identity is unavailable or unsafe")
    return host


@contextmanager
def _writer_lock(root: Path, transaction_id: str, *, recovery: bool) -> Iterator[None]:
    path = _role_path(root, LOCK_ROLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_parent_chain(root, path, "replacement writer lock")
    if path.is_symlink():
        _fail("replacement writer lock path is a symbolic link")
    if path.exists():
        if not recovery:
            _fail("another canonical replacement holds the writer lock")
        prior, prior_raw = _load_lock(path)
        if prior["transaction_id"] != transaction_id:
            _fail("writer lock belongs to a different transaction")
        if prior["host"] != _host_identity():
            _fail("cross-host lock recovery is refused; recover on the lock-owning host")
        if _pid_is_alive(int(prior["pid"])):
            _fail("writer lock owner is still alive; recovery is refused")
        if path.read_bytes() != prior_raw:
            _fail("writer lock changed during recovery inspection")
        path.unlink()
        _fsync_directory(path.parent)
    payload = {
        "schema_version": LOCK_SCHEMA,
        "transaction_id": transaction_id,
        "host": _host_identity(),
        "pid": os.getpid(),
        "created_at_utc": _utc_now(),
    }
    content = _json_bytes(payload)
    try:
        _write_new_file(path, content)
    except FileExistsError as exc:
        raise TransactionError("another canonical replacement acquired the writer lock") from exc
    try:
        yield
    finally:
        try:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
                path.unlink()
                _fsync_directory(path.parent)
        except OSError:
            # Never remove a lock whose ownership cannot be proven.  A later
            # recovery performs the same identity/PID checks explicitly.
            pass


def prepare_transaction(
    root: Path,
    transaction_id: str,
    candidate_root: Path,
    expected_before_manifest: Path,
    profile_inputs: Mapping[str, Path],
    loco_formal_dir: Path,
    *,
    validator: Validator = validate_candidate_bundle,
    profile_gate: ProfileGate = validate_full_profile_candidate,
    loco_gate: LocoGate = validate_full_loco_candidate,
    ultra_gate: UltraGate = validate_full_ultra_candidate,
) -> dict[str, Any]:
    """Prepare durable candidates/snapshots without touching canonical bytes."""
    root = _canonical_root(root)
    final_directory = _transaction_directory(root, transaction_id)
    if final_directory.exists() or final_directory.is_symlink():
        _fail("transaction directory already exists")
    candidate_root = _assert_physical_directory(candidate_root, "candidate root")
    if candidate_root == root:
        _fail("candidate root must not be the canonical repository root")
    baseline = _load_baseline(expected_before_manifest)
    profile_input_inventory, resolved_profile_inputs = _profile_input_inventory(
        root, profile_inputs
    )
    loco_input_inventory, resolved_loco_dir = _loco_input_inventory(
        root, loco_formal_dir
    )
    candidate_infos = _bundle_infos(candidate_root, exact_inventory=True)
    validator(root, candidate_root)
    loco_gate(root, candidate_root, resolved_loco_dir)
    ultra_gate(root, candidate_root, resolved_profile_inputs["ultra_dir"])
    profile_gate(root, candidate_root, resolved_profile_inputs)
    current_infos = _current_canonical_infos(root)
    _assert_infos_equal(current_infos, baseline, "canonical expected-before inventory")
    _, hold_raw = _read_hold(root)
    source_hashes = _bound_source_hashes(root)

    parent = final_directory.parent
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{transaction_id}.preparing-", dir=parent)
    )
    try:
        for role in ROLES:
            before_bytes = _read_regular(_role_path(root, role), role, base=root)
            candidate_bytes = _read_regular(
                _role_path(candidate_root, role), f"candidate {role}", base=candidate_root
            )
            before_target = _role_path(temporary / "before", role)
            candidate_target = _role_path(temporary / "candidate", role)
            before_target.parent.mkdir(parents=True, exist_ok=True)
            candidate_target.parent.mkdir(parents=True, exist_ok=True)
            _write_new_file(before_target, before_bytes)
            _write_new_file(candidate_target, candidate_bytes)

        copied_before = _bundle_infos(temporary / "before", exact_inventory=True)
        copied_candidate = _bundle_infos(temporary / "candidate", exact_inventory=True)
        _assert_infos_equal(copied_before, baseline, "copied before inventory")
        _assert_infos_equal(copied_candidate, candidate_infos, "copied candidate inventory")
        _write_new_file(temporary / "active_hold.json", hold_raw)
        validator(root, temporary / "candidate")
        loco_gate(root, temporary / "candidate", resolved_loco_dir)
        ultra_gate(
            root,
            temporary / "candidate",
            resolved_profile_inputs["ultra_dir"],
        )
        profile_gate(root, temporary / "candidate", resolved_profile_inputs)
        _assert_infos_equal(
            _bundle_infos(temporary / "before", exact_inventory=True),
            copied_before,
            "copied before inventory after staged validation",
        )
        _assert_infos_equal(
            _bundle_infos(temporary / "candidate", exact_inventory=True),
            copied_candidate,
            "copied candidate inventory after staged validation",
        )

        # Race guards are repeated after all potentially expensive validation
        # and copying, immediately before the PREPARED journal is sealed.
        _assert_infos_equal(
            _current_canonical_infos(root), baseline, "canonical expected-before inventory"
        )
        _, hold_after = _read_hold(root)
        if hold_after != hold_raw:
            _fail("active release hold changed during preparation")
        if _bound_source_hashes(root) != source_hashes:
            _fail("a bound source changed during preparation")
        observed_profile_inputs, resolved_profile_inputs = _profile_input_inventory(
            root, resolved_profile_inputs
        )
        if observed_profile_inputs != profile_input_inventory:
            _fail("a formal benchmark-profile input changed during preparation")
        observed_loco_input, resolved_loco_dir = _loco_input_inventory(
            root, resolved_loco_dir
        )
        if observed_loco_input != loco_input_inventory:
            _fail("the LOCO formal input tree changed during preparation")

        files: dict[str, Any] = {}
        for role in ROLES:
            validator_name = (
                "loco_public_verify"
                if role in LOCO_ROLES
                else "ultra_public_verify"
                if role in ULTRA_ROLES
                else "benchmark_profile_full_rebuild_and_repository_verify"
            )
            files[role] = {
                "before": copied_before[role],
                "candidate": copied_candidate[role],
                "validator": validator_name,
            }
        now = _utc_now()
        journal: dict[str, Any] = {
            "schema_version": TRANSACTION_SCHEMA,
            "transaction_id": transaction_id,
            "status": "PREPARED",
            "created_at_utc": now,
            "updated_at_utc": now,
            "committed_at_utc": None,
            "rolled_back_at_utc": None,
            "recovered_at_utc": None,
            "scope": list(ROLES),
            "active_hold": {
                "role": HOLD_ROLE,
                "sha256": _sha256_bytes(hold_raw),
                "status": REQUIRED_HOLD_STATUS,
            },
            "files": files,
            "bound_sources": dict(sorted(source_hashes.items())),
            "profile_inputs": profile_input_inventory,
            "loco_formal_input": loco_input_inventory,
            "before_inventory_sha256": _inventory_digest(copied_before),
            "candidate_inventory_sha256": _inventory_digest(copied_candidate),
            "plan_sha256": "0" * 64,
            "applied_roles": [],
            "error": None,
        }
        journal["plan_sha256"] = _plan_digest(journal)
        journal = _validate_journal(journal)
        _write_new_file(temporary / "transaction.json", _json_bytes(journal))
        os.rename(temporary, final_directory)
        _fsync_directory(parent)
        observed_directory, observed = _load_journal(root, transaction_id)
        _validate_transaction_storage(observed_directory, observed)
        return observed
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _replace_one(
    root: Path,
    source_root: Path,
    role: str,
    *,
    label: str,
    replace_func: ReplaceFunction,
) -> None:
    target = _role_path(root, role)
    _assert_parent_chain(root, target, f"canonical {role}")
    if target.is_symlink() or not target.is_file():
        _fail(f"canonical {role} is not a regular file")
    content = _read_regular(_role_path(source_root, role), f"stored {role}", base=source_root)
    temporary = _stage_same_directory(target, content, label)
    try:
        replace_func(temporary, target)
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()
    if _file_info(_read_regular(target, f"canonical {role}", base=root)) != _file_info(content):
        _fail(f"canonical replacement verification failed for {role}")


def _rollback_attempted(
    root: Path,
    directory: Path,
    journal: Mapping[str, Any],
    attempted: Sequence[str],
    *,
    replace_func: ReplaceFunction,
) -> list[str]:
    errors: list[str] = []
    for role in reversed(tuple(attempted)):
        try:
            _replace_one(
                root,
                directory / "before",
                role,
                label="rollback",
                replace_func=replace_func,
            )
        except BaseException as exc:  # pragma: no cover - asserted through result state
            errors.append(f"{role}:{type(exc).__name__}")
    try:
        _assert_infos_equal(
            _current_canonical_infos(root),
            _journal_infos(journal, "before"),
            "rollback canonical inventory",
        )
    except BaseException as exc:
        errors.append(f"inventory:{type(exc).__name__}")
    return errors


def apply_transaction(
    root: Path,
    transaction_id: str,
    confirmation: str,
    *,
    validator: Validator = validate_candidate_bundle,
    profile_gate: ProfileGate = validate_full_profile_candidate,
    loco_gate: LocoGate = validate_full_loco_candidate,
    ultra_gate: UltraGate = validate_full_ultra_candidate,
    replace_func: ReplaceFunction = _replace_path,
) -> dict[str, Any]:
    """Promote a PREPARED transaction and write PROMOTED only after verification."""
    root = _canonical_root(root)
    directory, journal = _load_journal(root, transaction_id)
    if journal["status"] != "PREPARED":
        _fail(f"apply requires PREPARED status, observed {journal['status']}")
    if confirmation != journal["plan_sha256"]:
        _fail("apply confirmation must equal the prepared plan SHA-256")
    _assert_no_other_unfinished_transaction(root, transaction_id)

    with _writer_lock(root, transaction_id, recovery=False):
        _assert_no_other_unfinished_transaction(root, transaction_id)
        directory, journal = _load_journal(root, transaction_id)
        if journal["status"] != "PREPARED":
            _fail("transaction status changed before lock acquisition")
        _validate_transaction_storage(directory, journal)
        _validate_live_bindings(root, journal)
        _, profile_inputs = _profile_paths_from_journal(root, journal)
        _, loco_formal_dir = _loco_path_from_journal(root, journal)
        _assert_infos_equal(
            _current_canonical_infos(root),
            _journal_infos(journal, "before"),
            "canonical expected-before inventory",
        )
        validator(root, directory / "candidate")
        loco_gate(root, directory / "candidate", loco_formal_dir)
        ultra_gate(root, directory / "candidate", profile_inputs["ultra_dir"])
        profile_gate(root, directory / "candidate", profile_inputs)
        # Close every validation-time race before the write-ahead state moves
        # to APPLYING and before the first canonical directory entry changes.
        _validate_transaction_storage(directory, journal)
        _validate_live_bindings(root, journal)
        _profile_paths_from_journal(root, journal)
        _loco_path_from_journal(root, journal)
        _assert_infos_equal(
            _current_canonical_infos(root),
            _journal_infos(journal, "before"),
            "canonical expected-before inventory",
        )

        journal = _update_journal(
            directory,
            journal,
            status="APPLYING",
            applied_roles=[],
            error=None,
        )
        attempted: list[str] = []
        applied: list[str] = []
        try:
            for role in ROLES:
                attempted.append(role)
                _replace_one(
                    root,
                    directory / "candidate",
                    role,
                    label="promote",
                    replace_func=replace_func,
                )
                applied.append(role)
                journal = _update_journal(
                    directory, journal, applied_roles=list(applied)
                )

            _assert_infos_equal(
                _current_canonical_infos(root),
                _journal_infos(journal, "candidate"),
                "promoted canonical inventory",
            )
            validator(root, root)
            loco_gate(root, root, loco_formal_dir)
            ultra_gate(root, root, profile_inputs["ultra_dir"])
            profile_gate(root, root, profile_inputs)
            _profile_paths_from_journal(root, journal)
            _loco_path_from_journal(root, journal)
            _validate_transaction_storage(directory, journal)
            _validate_live_bindings(root, journal)
            _assert_infos_equal(
                _current_canonical_infos(root),
                _journal_infos(journal, "candidate"),
                "promoted canonical inventory after final validators",
            )
            journal = _update_journal(
                directory,
                journal,
                status="PROMOTED",
                committed_at_utc=_utc_now(),
                applied_roles=list(ROLES),
                error=None,
            )
            return journal
        except BaseException as exc:
            rollback_errors = _rollback_attempted(
                root,
                directory,
                journal,
                attempted,
                replace_func=replace_func,
            )
            status = "RECOVERY_REQUIRED" if rollback_errors else "ROLLED_BACK"
            try:
                journal = _update_journal(
                    directory,
                    journal,
                    status=status,
                    applied_roles=list(applied),
                    rolled_back_at_utc=_utc_now() if not rollback_errors else None,
                    error=type(exc).__name__,
                )
            except BaseException as journal_exc:
                raise TransactionError(
                    "replacement failed and rollback state could not be journaled"
                ) from journal_exc
            if rollback_errors:
                raise TransactionError(
                    "replacement failed and byte-exact rollback requires recovery"
                ) from exc
            raise TransactionError(
                "replacement failed; all canonical bytes were restored"
            ) from exc


def recover_transaction(
    root: Path,
    transaction_id: str,
    confirmation: str,
    *,
    replace_func: ReplaceFunction = _replace_path,
) -> dict[str, Any]:
    """Recover an interrupted writer or restore an incomplete replacement.

    PREPARED is cancelled without rewriting canonical files; APPLYING and
    RECOVERY_REQUIRED restore every before snapshot; terminal states are
    reverified and merely clear a dead same-transaction lock left in the tiny
    window after the terminal journal write.
    """
    root = _canonical_root(root)
    directory, journal = _load_journal(root, transaction_id)
    if confirmation != journal["before_inventory_sha256"]:
        _fail("recover confirmation must equal the before-inventory SHA-256")

    with _writer_lock(root, transaction_id, recovery=True):
        directory, journal = _load_journal(root, transaction_id)
        status = journal["status"]
        if status == "PROMOTED":
            _validate_transaction_storage(directory, journal)
            _assert_infos_equal(
                _current_canonical_infos(root),
                _journal_infos(journal, "candidate"),
                "promoted canonical inventory",
            )
            return journal
        if status == "ROLLED_BACK":
            _validate_before_storage(directory, journal)
            _assert_infos_equal(
                _current_canonical_infos(root),
                _journal_infos(journal, "before"),
                "rolled-back canonical inventory",
            )
            return journal
        if status == "PREPARED":
            _validate_before_storage(directory, journal)
            _validate_hold_binding(root, journal)
            _assert_infos_equal(
                _current_canonical_infos(root),
                _journal_infos(journal, "before"),
                "prepared canonical inventory",
            )
            now = _utc_now()
            return _update_journal(
                directory,
                journal,
                status="ROLLED_BACK",
                rolled_back_at_utc=now,
                recovered_at_utc=now,
                error=None,
            )
        if status not in {"APPLYING", "RECOVERY_REQUIRED"}:
            raise AssertionError(status)
        # Recovery depends only on the durable before snapshots and the active
        # publication hold.  Candidate corruption or later verifier-source
        # drift must never make byte restoration impossible.
        _validate_before_storage(directory, journal)
        _validate_hold_binding(root, journal)
        errors: list[str] = []
        for role in reversed(ROLES):
            try:
                _replace_one(
                    root,
                    directory / "before",
                    role,
                    label="recover",
                    replace_func=replace_func,
                )
            except BaseException as exc:
                errors.append(f"{role}:{type(exc).__name__}")
        try:
            _assert_infos_equal(
                _current_canonical_infos(root),
                _journal_infos(journal, "before"),
                "recovered canonical inventory",
            )
        except BaseException as exc:
            errors.append(f"inventory:{type(exc).__name__}")
        if errors:
            journal = _update_journal(
                directory,
                journal,
                status="RECOVERY_REQUIRED",
                error="RecoveryError",
            )
            raise TransactionError("recovery could not restore every canonical byte")
        now = _utc_now()
        return _update_journal(
            directory,
            journal,
            status="ROLLED_BACK",
            rolled_back_at_utc=now,
            recovered_at_utc=now,
            error=None,
        )


def verify_transaction(
    root: Path,
    transaction_id: str,
    *,
    validator: Validator = validate_candidate_bundle,
    profile_gate: ProfileGate = validate_full_profile_candidate,
    loco_gate: LocoGate = validate_full_loco_candidate,
    ultra_gate: UltraGate = validate_full_ultra_candidate,
) -> dict[str, Any]:
    """Verify durable storage plus the canonical state implied by the journal."""
    root = _canonical_root(root)
    directory, journal = _load_journal(root, transaction_id)
    _validate_transaction_storage(directory, journal)
    if journal["status"] in {"APPLYING", "RECOVERY_REQUIRED"}:
        _fail(f"transaction status {journal['status']} requires explicit recovery")
    if journal["status"] == "PREPARED":
        _validate_live_bindings(root, journal)
        _, profile_inputs = _profile_paths_from_journal(root, journal)
        _, loco_formal_dir = _loco_path_from_journal(root, journal)
        _assert_infos_equal(
            _current_canonical_infos(root),
            _journal_infos(journal, "before"),
            "prepared canonical inventory",
        )
        validator(root, directory / "candidate")
        loco_gate(root, directory / "candidate", loco_formal_dir)
        ultra_gate(root, directory / "candidate", profile_inputs["ultra_dir"])
        profile_gate(root, directory / "candidate", profile_inputs)
    elif journal["status"] == "PROMOTED":
        # The current hold may legitimately be a later RESOLVED receipt.  The
        # exact active bytes used during promotion remain sealed inside the
        # transaction and were checked immediately before PROMOTED was written.
        _validate_bound_sources(root, journal)
        _, profile_inputs = _profile_paths_from_journal(root, journal)
        _, loco_formal_dir = _loco_path_from_journal(root, journal)
        _assert_infos_equal(
            _current_canonical_infos(root),
            _journal_infos(journal, "candidate"),
            "promoted canonical inventory",
        )
        validator(root, root)
        loco_gate(root, root, loco_formal_dir)
        ultra_gate(root, root, profile_inputs["ultra_dir"])
        profile_gate(root, root, profile_inputs)
    elif journal["status"] == "ROLLED_BACK":
        _assert_infos_equal(
            _current_canonical_infos(root),
            _journal_infos(journal, "before"),
            "rolled-back canonical inventory",
        )
    else:  # pragma: no cover - _validate_journal already closes the enum
        raise AssertionError(journal["status"])
    return journal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="seal candidates and before snapshots")
    prepare.add_argument("--transaction-id", required=True)
    prepare.add_argument(
        "--candidate-root",
        type=Path,
        required=True,
        help="dedicated root containing exactly the six staged canonical roles",
    )
    prepare.add_argument(
        "--expected-before",
        type=Path,
        required=True,
        help="reviewed baseline manifest for the six currently canonical bytes",
    )
    prepare.add_argument(
        "--dataset-summary", type=Path, required=True, help="profile dataset summary"
    )
    prepare.add_argument(
        "--freeze-manifest", type=Path, required=True, help="formal early-graph freeze"
    )
    prepare.add_argument(
        "--claims-dir", type=Path, required=True, help="formal main-worker claims directory"
    )
    prepare.add_argument(
        "--metrics-dir", type=Path, required=True, help="verified GPU metrics directory"
    )
    prepare.add_argument(
        "--selections-dir",
        type=Path,
        required=True,
        help="verified GPU selections directory",
    )
    prepare.add_argument(
        "--gpu-inventory", type=Path, required=True, help="formal GPU inventory receipt"
    )
    prepare.add_argument(
        "--ultra-dir", type=Path, required=True, help="verified ULTRA formal receipt tree"
    )
    prepare.add_argument(
        "--ultra-receipts-dir",
        type=Path,
        required=True,
        help="verified ULTRA orchestration receipt tree for physical GPU provenance",
    )
    prepare.add_argument(
        "--loco-dir",
        type=Path,
        required=True,
        help="canonical verified results_v2/loco_formal tree",
    )

    apply = commands.add_parser("apply", help="promote one PREPARED transaction")
    apply.add_argument("--transaction-id", required=True)
    apply.add_argument("--confirm", required=True)

    recover = commands.add_parser("recover", help="restore all six before snapshots")
    recover.add_argument("--transaction-id", required=True)
    recover.add_argument("--confirm", required=True)

    verify = commands.add_parser("verify", help="verify journal/storage/canonical state")
    verify.add_argument("--transaction-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            journal = prepare_transaction(
                args.root,
                args.transaction_id,
                args.candidate_root,
                args.expected_before,
                {
                    "dataset_summary": args.dataset_summary,
                    "freeze_manifest": args.freeze_manifest,
                    "claims_dir": args.claims_dir,
                    "metrics_dir": args.metrics_dir,
                    "selections_dir": args.selections_dir,
                    "gpu_inventory": args.gpu_inventory,
                    "ultra_dir": args.ultra_dir,
                    "ultra_receipts_dir": args.ultra_receipts_dir,
                },
                args.loco_dir,
            )
            print(
                "PREPARED canonical replacement "
                f"{journal['transaction_id']}; confirm={journal['plan_sha256']}"
            )
        elif args.command == "apply":
            journal = apply_transaction(
                args.root, args.transaction_id, args.confirm
            )
            print(
                f"PROMOTED canonical replacement {journal['transaction_id']} "
                f"({len(ROLES)} files)"
            )
        elif args.command == "recover":
            journal = recover_transaction(
                args.root, args.transaction_id, args.confirm
            )
            print(f"ROLLED_BACK canonical replacement {journal['transaction_id']}")
        elif args.command == "verify":
            journal = verify_transaction(args.root, args.transaction_id)
            print(
                f"VERIFIED canonical replacement {journal['transaction_id']} "
                f"status={journal['status']}"
            )
        else:  # pragma: no cover - argparse closes the command enum
            raise AssertionError(args.command)
    except TransactionError as exc:
        print(f"CANONICAL REPLACEMENT REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
