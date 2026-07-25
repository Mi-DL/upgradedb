#!/usr/bin/env python3
"""Plan, build, and verify deterministic UPGRADE-BENCH data bundles.

The source checkout is the authoritative staging area.  This tool does not
upload anything and never removes source data.  It records each payload file
in ``release/DATA_ARTIFACT_INDEX.json`` and can create deterministic ZIP files
with an internal SHA-256 payload manifest plus an external checksum sidecar.

Typical release sequence::

    python tools/artifact_bundles.py write-index
    python tools/artifact_bundles.py verify-index
    python tools/artifact_bundles.py build all --output-dir dist
    python tools/artifact_bundles.py verify-archives --output-dir dist

Archive members retain repository-relative paths, so extracting a bundle at
the repository root reconstructs the expected layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence

import public_release_policy


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "release" / "DATA_ARTIFACT_INDEX.json"
SCHEMA_VERSION = 3
GITHUB_RELEASE_ASSET_LIMIT = 2 * 1024**3
# Leave room for ZIP metadata and for future small additions to a bundle.
MAX_PLANNED_UNCOMPRESSED_BYTES = 1800 * 1024**2
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
INTERNAL_MANIFEST = "UPGRADE_BENCH_PAYLOAD_MANIFEST.sha256"
INTERNAL_INDEX = "UPGRADE_BENCH_DATA_ARTIFACT_INDEX.json"
EXTERNAL_PAYLOAD_PREFIXES = ("data/processed_v2/",)
BUNDLE_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
BUNDLE_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".bak", ".tmp"}
# Compatibility aliases for callers; policy is owned by public_release_policy.
PERMISSION_GATED_PUBLIC_PATHS = public_release_policy.PERMISSION_GATED_PUBLIC_PATHS
PUBLIC_BUNDLE_OMIT_PATHS = (
    public_release_policy.PERMISSION_GATED_PUBLIC_PATHS
    | public_release_policy.INTERNAL_NESTED_MANIFESTS
)
PUBLIC_PROVENANCE_EXCLUDED_PREFIXES = public_release_policy.INTERNAL_ONLY_PREFIXES
PUBLIC_PROVENANCE_EXCLUDED_PATHS = public_release_policy.INTERNAL_ONLY_PATHS
ATOMIC_PUBLIC_RESULT_GROUPS = (
    frozenset(
        {
            "results_v2/metrics/v2_gbdt_baselines.json",
            "results_v2/metrics/v2_gbdt_baselines.csv",
        }
    ),
    frozenset(
        {
            "results_v2/metrics/v2_contemporary_references.json",
            "results_v2/metrics/v2_contemporary_references.csv",
        }
    ),
    frozenset(
        {
            "results_v2/metrics/v2_product_space_density.json",
            "results_v2/metrics/v2_product_space_density.csv",
            "results_v2/scores/v2_product_space_density_scores.csv",
        }
    ),
    frozenset(
        {
            "results_v2/metrics/v2_score_robustness_r5.json",
            "results_v2/metrics/v2_score_robustness_r5.csv",
        }
    ),
    frozenset(
        {
            "results_v2/metrics/v2_eligibility_threshold_geometry.json",
            "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
        }
    ),
)


@dataclass(frozen=True)
class BundleSpec:
    bundle_id: str
    archive: str
    description: str
    paths: tuple[str, ...]


def _rel(path: Path, root: Path = ROOT) -> str:
    return path.relative_to(root).as_posix()


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and value == path.as_posix()


def _safe_archive_name(value: object) -> bool:
    """Return whether *value* is one canonical basename-only ZIP name.

    ``PurePosixPath`` alone does not treat a backslash as a separator.  On
    Windows, however, the later ``Path(output_dir) / archive`` operation does,
    so a value such as ``..\\escape.zip`` could otherwise leave the requested
    output directory.  Reuse the repository-wide canonical-path policy before
    enforcing the bundle-specific basename-only rule.
    """
    if not isinstance(value, str) or not value.endswith(".zip"):
        return False
    if public_release_policy.canonical_path_reason(value) is not None:
        return False
    if PureWindowsPath(value).drive:
        return False
    path = PurePosixPath(value)
    return path.name == value and len(path.parts) == 1


def _contained_output_path(output_dir: Path, name: str) -> Path:
    """Resolve an output path and fail if it can escape *output_dir*.

    This is defense in depth after index validation and also rejects an
    existing output-file symlink whose target is outside the build directory.
    """
    output_root = output_dir.resolve(strict=False)
    candidate = output_dir / name
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve bundle output safely: {candidate}") from exc
    if resolved.parent != output_root:
        raise ValueError(f"bundle output resolves outside output directory: {name!r}")
    return candidate


def _is_physical_path(path: Path, *, directory: bool) -> bool:
    """Reject symlinks, junctions, and paths reached through nonphysical parents."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    lexical = Path(os.path.abspath(path))
    if os.path.normcase(str(resolved)) != os.path.normcase(str(lexical)):
        return False
    if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
        return False
    return path.is_dir() if directory else path.is_file()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _files_under(path: Path, root: Path = ROOT) -> list[str]:
    if not path.is_dir():
        return []
    selected: list[str] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if any(part in BUNDLE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if item.suffix.lower() in BUNDLE_EXCLUDED_SUFFIXES:
            continue
        selected.append(relative.as_posix())
    return sorted(selected)


def _tracked_files(prefix: str, root: Path = ROOT) -> list[str]:
    """Return tracked paths under *prefix*; use the repository manifest as fallback."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--", prefix],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        output = b""
    paths = sorted(part.decode("utf-8").replace("\\", "/") for part in output.split(b"\0") if part)
    if paths:
        return paths

    manifest = root / "RELEASE_MANIFEST.sha256"
    if manifest.is_file():
        selected: list[str] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "  " not in line:
                continue
            name = line.split("  ", 1)[1]
            if name.startswith(prefix.rstrip("/") + "/") and (root / name).is_file():
                selected.append(name)
        if selected:
            return sorted(selected)
    raise RuntimeError(f"cannot determine tracked inventory under {prefix!r}; run from a Git checkout")


def _frozen_external_paths(prefix: str, root: Path = ROOT) -> list[str]:
    """Recover an absent release-asset selector from the committed index.

    External payloads are deliberately absent from public Git.  When no bytes
    are mounted, selector comparison must still use the frozen path inventory
    rather than silently dropping the bundle.  Present bytes always win so an
    unlisted file in a partial checkout remains visible to inventory checks.
    """

    normalized = prefix.rstrip("/") + "/"
    present = _files_under(root / prefix, root)
    if present:
        return present
    index_path = root / "release" / "DATA_ARTIFACT_INDEX.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        bundles = payload["bundles"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"cannot recover absent external inventory under {prefix!r} from the frozen index"
        ) from exc
    recovered = sorted(
        {
            str(item["path"])
            for bundle in bundles
            if isinstance(bundle, dict)
            for item in bundle.get("files", [])
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and str(item["path"]).startswith(normalized)
        }
    )
    if not recovered:
        raise RuntimeError(f"frozen index has no external inventory under {prefix!r}")
    return recovered


def _v2_partition(root: Path, fold2: bool) -> list[str]:
    if public_release_policy.unresolved_v2_invalidation(root) is not None:
        # The current cohort tables are named by the active invalidation notice.
        # Keep development plans documentary-only until the corrected rebuild.
        return []
    base = root / "data" / "processed_v2"
    files = _frozen_external_paths("data/processed_v2", root)
    selected = []
    for name in files:
        stem = PurePosixPath(name).stem
        is_fold2 = stem.endswith("_fold2")
        if is_fold2 == fold2:
            selected.append(name)
    return selected


def _with_shared(paths: Iterable[str], *shared: str) -> tuple[str, ...]:
    return tuple(sorted(set(paths).union(shared)))


def _public_paths(paths: Iterable[str]) -> list[str]:
    return public_release_policy.public_paths(paths)


def _public_v2_results(root: Path) -> list[str]:
    """Select existing v2 public results by exact name; never recurse."""
    allowlist = public_release_policy.PUBLIC_V2_RESULT_ALLOWLIST
    if public_release_policy.unresolved_v2_invalidation(root) is not None:
        # Development-facing release planning may expose the invalidation
        # notice, but never the stale numeric artifacts it invalidates.
        allowlist = public_release_policy.PUBLIC_V2_INVALIDATION_HOLD_ALLOWLIST
    selected = {name for name in allowlist if (root / name).is_file()}
    for group in ATOMIC_PUBLIC_RESULT_GROUPS:
        present = selected.intersection(group)
        if present and present != group:
            missing = ", ".join(sorted(group - present))
            raise ValueError(
                "incomplete atomic public result group; missing " + missing
            )
    return sorted(selected)


def unresolved_v2_invalidation(root: Path = ROOT) -> str | None:
    """Compatibility wrapper for the shared public policy."""
    return public_release_policy.unresolved_v2_invalidation(root)


def bundle_specs(root: Path = ROOT) -> tuple[BundleSpec, ...]:
    invalidation_blocker = public_release_policy.unresolved_v2_invalidation(root)
    package_v2 = _public_paths(_files_under(root / "benchmark" / "upgrade-bench-v2", root))
    v2_main = _v2_partition(root, fold2=False)
    v2_history = _v2_partition(root, fold2=True)
    v2_results = _public_v2_results(root)
    shared_docs = (
        "DATA_LICENSE.md",
        "ARTIFACT.md",
        "docs/DATA_DISTRIBUTION.md",
        "docs/PUBLIC_RELEASE_POLICY.md",
        "docs/REGISTRY_REVIEW_CODEBOOK.md",
    )

    return (
        BundleSpec(
            "v2-standalone",
            "upgrade-bench-v2-standalone.zip",
            "Dependency-light UpgradeBench loader, evaluator, protocol examples, and package manifest.",
            _with_shared(package_v2, *shared_docs, "BENCHMARK_V2_SPEC.md"),
        ),
        BundleSpec(
            "v2-main",
            "upgrade-bench-v2-main-window.zip",
            (
                "BLOCKED by active registry invalidation; documentary payload only."
                if invalidation_blocker is not None
                else "Six-chain main-window lane, entry, destination, and summary tables."
            ),
            _with_shared(v2_main, *shared_docs, "BENCHMARK_V2_SPEC.md"),
        ),
        BundleSpec(
            "v2-history",
            "upgrade-bench-v2-historical-fold.zip",
            (
                "BLOCKED by active registry invalidation; documentary payload only."
                if invalidation_blocker is not None
                else "Six-chain historical selection-fold lane, entry, destination, and summary tables."
            ),
            _with_shared(v2_history, *shared_docs, "BENCHMARK_V2_SPEC.md"),
        ),
        BundleSpec(
            "v2-results",
            "upgrade-bench-v2-results-and-audits.zip",
            (
                "Registry invalidation notice and explanatory documents only; final freeze blocked."
                if invalidation_blocker is not None
                else "Allowlisted metrics, audits, sanitized summaries, and benchmark specification."
            ),
            tuple(
                public_release_policy.public_paths(
                    _with_shared(
                        v2_results,
                        *shared_docs,
                        "BENCHMARK_V2_SPEC.md",
                        "paper/generated/v2_numbers.tex",
                        "paper/generated/v2_benchmark_profile.tex",
                        "paper/generated/v2_contemporary_references.tex",
                    ),
                    root,
                )
            ),
        ),
    )


def _payload_digest(files: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item['sha256']}  {item['bytes']}  {item['path']}\n".encode("utf-8"))
    return digest.hexdigest()


def make_index(root: Path = ROOT) -> dict[str, object]:
    blocker = public_release_policy.unresolved_v2_invalidation(root)
    if blocker is not None:
        raise ValueError(f"refusing final public artifact index: {blocker}")
    bundles: list[dict[str, object]] = []
    seen_archives: set[str] = set()
    for spec in bundle_specs(root):
        if not _safe_archive_name(spec.archive):
            raise ValueError(f"unsafe archive name: {spec.archive!r}")
        if spec.archive in seen_archives:
            raise ValueError(f"duplicate archive name: {spec.archive}")
        seen_archives.add(spec.archive)
        files: list[dict[str, object]] = []
        for name in spec.paths:
            if not _safe_relative_path(name):
                raise ValueError(f"unsafe bundle path: {name!r}")
            reason = public_release_policy.exclusion_reason(name, root)
            if reason is not None:
                raise ValueError(f"public-policy-excluded bundle path: {name!r} ({reason})")
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(f"missing bundle payload: {name}")
            files.append({"path": name, "bytes": source.stat().st_size, "sha256": sha256_file(source)})
        total = sum(int(item["bytes"]) for item in files)
        if total > MAX_PLANNED_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"bundle {spec.bundle_id} is {total / 1024**2:.1f} MiB; "
                f"split it before the {MAX_PLANNED_UNCOMPRESSED_BYTES / 1024**2:.0f} MiB policy limit"
            )
        bundles.append(
            {
                "id": spec.bundle_id,
                "archive": spec.archive,
                "description": spec.description,
                "file_count": len(files),
                "uncompressed_bytes": total,
                "payload_sha256": _payload_digest(files),
                "files": files,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "archive_format": "zip",
        "archive_member_paths": "repository-relative",
        "github_release_asset_limit_bytes": GITHUB_RELEASE_ASSET_LIMIT,
        "planned_uncompressed_limit_bytes": MAX_PLANNED_UNCOMPRESSED_BYTES,
        "public_distribution_policy": public_release_policy.index_policy(),
        "bundles": bundles,
    }


def write_index(path: Path = INDEX_PATH, root: Path = ROOT) -> dict[str, object]:
    index = make_index(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    total_files = sum(int(bundle["file_count"]) for bundle in index["bundles"])
    total_bytes = sum(int(bundle["uncompressed_bytes"]) for bundle in index["bundles"])
    print(f"wrote {_rel(path, root)} ({len(index['bundles'])} bundles, {total_files} entries, {total_bytes / 1024**2:.1f} MiB)")
    return index


def load_index(path: Path = INDEX_PATH) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing index: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse index {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("artifact index root must be an object")
    return data


def validate_index_structure(index: dict[str, object], root: Path = ROOT) -> None:
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {index.get('schema_version')!r}")
    if index.get("hash_algorithm") != "sha256" or index.get("archive_format") != "zip":
        raise ValueError("artifact index must use SHA-256 and ZIP")
    expected_policy = public_release_policy.index_policy()
    if index.get("public_distribution_policy") != expected_policy:
        raise ValueError("artifact index public_distribution_policy mismatch")
    bundles = index.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("artifact index has no bundles")
    bundle_ids: set[str] = set()
    archives: set[str] = set()
    for bundle in bundles:
        if not isinstance(bundle, dict):
            raise ValueError("bundle entry must be an object")
        bundle_id = bundle.get("id")
        archive = bundle.get("archive")
        files = bundle.get("files")
        if not isinstance(bundle_id, str) or not bundle_id or bundle_id in bundle_ids:
            raise ValueError(f"invalid/duplicate bundle id: {bundle_id!r}")
        if (
            not isinstance(archive, str)
            or not _safe_archive_name(archive)
            or archive in archives
        ):
            raise ValueError(f"invalid/duplicate archive name: {archive!r}")
        if not isinstance(files, list) or not files:
            raise ValueError(f"bundle {bundle_id} has no files")
        bundle_ids.add(bundle_id)
        archives.add(archive)
        paths: set[str] = set()
        total = 0
        for item in files:
            if not isinstance(item, dict):
                raise ValueError(f"bundle {bundle_id} contains a non-object file entry")
            name, size, digest = item.get("path"), item.get("bytes"), item.get("sha256")
            if not isinstance(name, str) or not _safe_relative_path(name) or name in paths:
                raise ValueError(f"bundle {bundle_id} has invalid/duplicate path: {name!r}")
            reason = public_release_policy.exclusion_reason(name, root)
            if reason is not None:
                raise ValueError(
                    f"bundle {bundle_id} contains a public-policy-excluded path: "
                    f"{name!r} ({reason})"
                )
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"bundle {bundle_id} has invalid size for {name!r}")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"bundle {bundle_id} has invalid SHA-256 for {name!r}")
            paths.add(name)
            total += size
        if bundle.get("file_count") != len(files):
            raise ValueError(f"bundle {bundle_id} file_count mismatch")
        if bundle.get("uncompressed_bytes") != total:
            raise ValueError(f"bundle {bundle_id} uncompressed_bytes mismatch")
        if bundle.get("payload_sha256") != _payload_digest(files):
            raise ValueError(f"bundle {bundle_id} payload_sha256 mismatch")
        if total > int(index.get("planned_uncompressed_limit_bytes", 0)):
            raise ValueError(f"bundle {bundle_id} exceeds the declared planned size limit")


def _inventory_failures(index: dict[str, object], root: Path) -> list[str]:
    """Compare every currently present selected path even with partial external data.

    Repository-only CI may legitimately lack ``data/processed_v2``.  That does
    not permit an unrelated present file, a newly added external table, or a
    changed bundle selector to escape the frozen inventory comparison.
    """
    frozen = {str(bundle["id"]): {str(item["path"]) for item in bundle["files"]} for bundle in index["bundles"]}
    current_specs = {spec.bundle_id: set(spec.paths) for spec in bundle_specs(root)}
    failures: list[str] = []
    if set(frozen) != set(current_specs):
        missing_bundles = sorted(set(frozen) - set(current_specs))
        new_bundles = sorted(set(current_specs) - set(frozen))
        if missing_bundles:
            failures.append(f"INDEX INVENTORY missing current bundle selector(s): {', '.join(missing_bundles)}")
        if new_bundles:
            failures.append(f"INDEX INVENTORY unlisted bundle selector(s): {', '.join(new_bundles)}")
    for bundle_id in sorted(set(frozen) & set(current_specs)):
        frozen_present = {path for path in frozen[bundle_id] if (root / path).is_file()}
        current_present = {path for path in current_specs[bundle_id] if (root / path).is_file()}
        for name in sorted(current_present - frozen_present):
            failures.append(f"INDEX INVENTORY UNLISTED in {bundle_id}: {name}")
        for name in sorted(frozen_present - current_present):
            failures.append(f"INDEX INVENTORY OUT-OF-POLICY in {bundle_id}: {name}")
    return failures


def verify_index(
    path: Path = INDEX_PATH,
    root: Path = ROOT,
    *,
    allow_missing: bool = False,
    check_inventory: bool = True,
) -> bool:
    blocker = public_release_policy.unresolved_v2_invalidation(root)
    if blocker is not None:
        print(f"INDEX FAILED: refusing invalidated public inventory: {blocker}", file=sys.stderr)
        return False
    try:
        index = load_index(path)
        validate_index_structure(index, root)
    except ValueError as exc:
        print(f"INDEX FAILED: {exc}", file=sys.stderr)
        return False

    failures: list[str] = []
    missing: list[str] = []
    required_missing: list[str] = []
    for bundle in index["bundles"]:
        for item in bundle["files"]:
            name = str(item["path"])
            target = root / name
            unsafe = public_release_policy.source_path_reason(name, root)
            if unsafe is not None:
                failures.append(f"UNSAFE SOURCE: {name} ({unsafe})")
                continue
            if not target.is_file():
                if allow_missing and name.startswith(EXTERNAL_PAYLOAD_PREFIXES):
                    missing.append(name)
                else:
                    required_missing.append(name)
                continue
            if target.stat().st_size != int(item["bytes"]):
                failures.append(f"SIZE MISMATCH: {name}")
                continue
            actual = sha256_file(target)
            if actual != item["sha256"]:
                failures.append(f"HASH MISMATCH: {name}")

    failures.extend(f"MISSING: {name}" for name in required_missing)

    if check_inventory:
        try:
            failures.extend(_inventory_failures(index, root))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append(f"INDEX INVENTORY ERROR: {exc}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        print("DATA ARTIFACT INDEX FAILED", file=sys.stderr)
        return False

    file_count = sum(int(bundle["file_count"]) for bundle in index["bundles"])
    if missing:
        unique_missing = len(set(missing))
        print(
            f"verified index structure and {file_count - len(missing)} present entries; "
            f"{unique_missing} external payload paths are not in this checkout"
        )
    else:
        print(f"verified data artifact index ({len(index['bundles'])} bundles, {file_count} entries)")
    return True


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def _manifest_bytes(bundle: dict[str, object]) -> bytes:
    lines = ["# SHA-256  repository-relative path\n"]
    lines.extend(f"{item['sha256']}  {item['path']}\n" for item in bundle["files"])
    return "".join(lines).encode("utf-8")


def _build_one(
    bundle: dict[str, object],
    index_bytes: bytes,
    output_dir: Path,
    root: Path,
    *,
    force: bool,
) -> tuple[Path, str]:
    archive_name = str(bundle["archive"])
    if not _safe_archive_name(archive_name):
        raise ValueError(f"unsafe archive name: {archive_name!r}")
    archive = _contained_output_path(output_dir, archive_name)
    sidecar = _contained_output_path(output_dir, archive.name + ".sha256")
    if (archive.exists() or sidecar.exists()) and not force:
        raise FileExistsError(f"refusing to overwrite {archive}; pass --force")
    for item in bundle["files"]:
        name = str(item["path"])
        unsafe = public_release_policy.source_path_reason(name, root, require_file=True)
        if unsafe is not None:
            raise ValueError(f"unsafe bundle source: {name} ({unsafe})")
        target = root / name
        if not target.is_file() or target.stat().st_size != int(item["bytes"]) or sha256_file(target) != item["sha256"]:
            raise ValueError(f"payload does not match frozen index: {item['path']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".tmp", dir=output_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target_zip:
            target_zip.writestr(_zip_info(INTERNAL_MANIFEST), _manifest_bytes(bundle))
            target_zip.writestr(_zip_info(INTERNAL_INDEX), index_bytes)
            for item in bundle["files"]:
                name = str(item["path"])
                unsafe = public_release_policy.source_path_reason(name, root, require_file=True)
                if unsafe is not None:
                    raise ValueError(f"unsafe bundle source: {name} ({unsafe})")
                source = root / name
                with source.open("rb") as src, target_zip.open(_zip_info(str(item["path"])), "w") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        archive_digest = sha256_file(temp_path)
        if temp_path.stat().st_size >= GITHUB_RELEASE_ASSET_LIMIT:
            raise ValueError(f"archive {archive.name} exceeds GitHub's 2 GiB release-asset limit")
        os.replace(temp_path, archive)
        sidecar.write_text(f"{archive_digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    finally:
        if temp_path.exists():
            temp_path.unlink()
    print(f"built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB, sha256={archive_digest})")
    return archive, archive_digest


def build_archives(
    requested: Sequence[str],
    *,
    output_dir: Path,
    index_path: Path = INDEX_PATH,
    root: Path = ROOT,
    force: bool = False,
) -> list[tuple[Path, str]]:
    blocker = public_release_policy.unresolved_v2_invalidation(root)
    if blocker is not None:
        raise ValueError(f"refusing public bundle build: {blocker}")
    index = load_index(index_path)
    validate_index_structure(index, root)
    by_id = {str(bundle["id"]): bundle for bundle in index["bundles"]}
    wanted = list(by_id) if requested == ["all"] else list(requested)
    unknown = sorted(set(wanted) - set(by_id))
    if unknown:
        raise ValueError(f"unknown bundle id(s): {', '.join(unknown)}")
    index_bytes = index_path.read_bytes()
    built = [_build_one(by_id[bundle_id], index_bytes, output_dir, root, force=force) for bundle_id in wanted]
    checksums = _contained_output_path(output_dir, "SHA256SUMS")
    checksum_lines = []
    for bundle in index["bundles"]:
        archive = output_dir / str(bundle["archive"])
        if archive.is_file():
            checksum_lines.append(f"{sha256_file(archive)}  {archive.name}\n")
    checksums.write_text("".join(checksum_lines), encoding="utf-8", newline="\n")
    print(f"wrote {checksums} ({len(checksum_lines)} archives)")
    return built


def verify_archives(output_dir: Path, index_path: Path = INDEX_PATH) -> bool:
    """Verify the complete frozen archive set; partial build directories fail."""
    if not _is_physical_path(index_path, directory=False):
        print(
            f"ARCHIVE FAILED: index is not a physical regular file: {index_path}",
            file=sys.stderr,
        )
        return False
    try:
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ARCHIVE FAILED: cannot parse index {index_path}: {exc}", file=sys.stderr)
        return False
    if not isinstance(index, dict):
        print("ARCHIVE FAILED: artifact index root must be an object", file=sys.stderr)
        return False
    validate_index_structure(index)
    failures: list[str] = []
    bundles = list(index["bundles"])
    expected_names = {"SHA256SUMS"}
    for bundle in bundles:
        archive_name = str(bundle["archive"])
        expected_names.add(archive_name)
        expected_names.add(f"{archive_name}.sha256")

    if not _is_physical_path(output_dir, directory=True):
        failures.append(f"archive output is not a physical directory: {output_dir}")
        actual_names: set[str] = set()
    else:
        try:
            actual_names = {entry.name for entry in output_dir.iterdir()}
        except OSError as exc:
            failures.append(f"cannot inventory archive output {output_dir}: {exc}")
            actual_names = set()

    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    if missing_names:
        failures.append(f"planned archive inventory missing: {', '.join(missing_names)}")
    if unexpected_names:
        failures.append(f"unexpected archive output: {', '.join(unexpected_names)}")

    archive_digests: dict[str, str] = {}
    sidecar_snapshots: dict[str, bytes] = {}
    for bundle in bundles:
        archive_name = str(bundle["archive"])
        try:
            archive = _contained_output_path(output_dir, archive_name)
            sidecar = _contained_output_path(output_dir, archive.name + ".sha256")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if not _is_physical_path(archive, directory=False) or not _is_physical_path(
            sidecar, directory=False
        ):
            failures.append(f"archive/sidecar pair incomplete: {archive.name}")
            continue
        archive_digest = sha256_file(archive)
        archive_digests[archive.name] = archive_digest
        expected_sidecar = f"{archive_digest}  {archive.name}\n".encode("utf-8")
        try:
            sidecar_bytes = sidecar.read_bytes()
        except OSError as exc:
            failures.append(f"cannot read archive checksum for {archive.name}: {exc}")
            continue
        sidecar_snapshots[archive.name] = sidecar_bytes
        if sidecar_bytes != expected_sidecar:
            failures.append(f"archive checksum mismatch: {archive.name}")
            continue
        try:
            with zipfile.ZipFile(archive) as source_zip:
                bad = source_zip.testzip()
                if bad:
                    failures.append(f"corrupt ZIP member in {archive.name}: {bad}")
                    continue
                expected_members = {
                    INTERNAL_MANIFEST,
                    INTERNAL_INDEX,
                    *(str(item["path"]) for item in bundle["files"]),
                }
                actual_members = set(source_zip.namelist())
                if len(source_zip.namelist()) != len(actual_members) or actual_members != expected_members:
                    failures.append(f"archive member inventory mismatch: {archive.name}")
                    continue
                if source_zip.read(INTERNAL_INDEX) != index_bytes:
                    failures.append(f"embedded index mismatch: {archive.name}")
                if source_zip.read(INTERNAL_MANIFEST) != _manifest_bytes(bundle):
                    failures.append(f"embedded payload manifest mismatch: {archive.name}")
                for item in bundle["files"]:
                    member = source_zip.getinfo(str(item["path"]))
                    if member.file_size != int(item["bytes"]):
                        failures.append(f"member size mismatch in {archive.name}: {item['path']}")
                        continue
                    with source_zip.open(member) as payload:
                        if _sha256_stream(payload) != item["sha256"]:
                            failures.append(f"member hash mismatch in {archive.name}: {item['path']}")
        except (OSError, zipfile.BadZipFile):
            failures.append(f"invalid ZIP: {archive.name}")

    try:
        checksums = _contained_output_path(output_dir, "SHA256SUMS")
    except ValueError as exc:
        failures.append(str(exc))
        checksums = output_dir / "SHA256SUMS"
    checksum_bytes: bytes | None = None
    if not _is_physical_path(checksums, directory=False):
        failures.append("SHA256SUMS is missing or is not a regular physical file")
    elif len(archive_digests) == len(bundles):
        expected_checksums = "".join(
            f"{archive_digests[str(bundle['archive'])]}  {bundle['archive']}\n"
            for bundle in bundles
        ).encode("utf-8")
        try:
            checksum_bytes = checksums.read_bytes()
        except OSError as exc:
            failures.append(f"cannot read SHA256SUMS: {exc}")
        else:
            if checksum_bytes != expected_checksums:
                failures.append("SHA256SUMS inventory or digest mismatch")

    # Re-read every trust-bearing input after ZIP/member validation. This
    # catches concurrent replacement between the initial digest and the later
    # archive walk (including changes that preserve all ZIP members).
    if not _is_physical_path(index_path, directory=False):
        failures.append("artifact index changed path identity during verification")
    else:
        try:
            if index_path.read_bytes() != index_bytes:
                failures.append("artifact index changed during verification")
        except OSError as exc:
            failures.append(f"cannot re-read artifact index: {exc}")
    if _is_physical_path(output_dir, directory=True):
        try:
            final_names = {entry.name for entry in output_dir.iterdir()}
        except OSError as exc:
            failures.append(f"cannot re-inventory archive output: {exc}")
        else:
            if final_names != actual_names or final_names != expected_names:
                failures.append("archive output inventory changed during verification")
    for bundle in bundles:
        archive_name = str(bundle["archive"])
        archive = output_dir / archive_name
        sidecar = output_dir / f"{archive_name}.sha256"
        if archive_name not in archive_digests or archive_name not in sidecar_snapshots:
            continue
        if not _is_physical_path(archive, directory=False) or not _is_physical_path(
            sidecar, directory=False
        ):
            failures.append(f"archive/sidecar path identity changed: {archive_name}")
            continue
        try:
            if sha256_file(archive) != archive_digests[archive_name]:
                failures.append(f"archive changed during verification: {archive_name}")
            if sidecar.read_bytes() != sidecar_snapshots[archive_name]:
                failures.append(f"archive sidecar changed during verification: {archive_name}")
        except OSError as exc:
            failures.append(f"cannot re-verify archive stability for {archive_name}: {exc}")
    if checksum_bytes is not None:
        if not _is_physical_path(checksums, directory=False):
            failures.append("SHA256SUMS path identity changed during verification")
        else:
            try:
                if checksums.read_bytes() != checksum_bytes:
                    failures.append("SHA256SUMS changed during verification")
            except OSError as exc:
                failures.append(f"cannot re-read SHA256SUMS: {exc}")
    if failures:
        for failure in failures:
            print(f"ARCHIVE FAILED: {failure}", file=sys.stderr)
        return False
    print(f"verified exactly {len(bundles)} archive(s) in {output_dir}")
    return True


def print_plan(index: dict[str, object]) -> None:
    for bundle in index["bundles"]:
        print(
            f"{bundle['id']:<16} {int(bundle['file_count']):>4} files  "
            f"{int(bundle['uncompressed_bytes']) / 1024**2:>8.1f} MiB  {bundle['archive']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write-index", help="freeze current payload inventory and SHA-256 values")
    verify = sub.add_parser("verify-index", help="verify index structure, inventory, and payload hashes")
    verify.add_argument(
        "--allow-missing",
        action="store_true",
        help="permit externally distributed payloads to be absent (repository-only CI)",
    )
    sub.add_parser("list", help="print the frozen bundle plan")
    build = sub.add_parser("build", help="build one or more deterministic ZIP release assets")
    build.add_argument("bundles", nargs="+", help="bundle id(s), or the single value 'all'")
    build.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    build.add_argument("--force", action="store_true")
    archive_verify = sub.add_parser(
        "verify-archives",
        help="verify the complete planned ZIP/sidecar/SHA256SUMS inventory",
    )
    archive_verify.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    try:
        if args.command == "write-index":
            write_index()
            return 0
        if args.command == "verify-index":
            return 0 if verify_index(allow_missing=args.allow_missing) else 1
        if args.command == "list":
            index = load_index()
            validate_index_structure(index)
            print_plan(index)
            return 0
        if args.command == "build":
            if "all" in args.bundles and args.bundles != ["all"]:
                raise ValueError("'all' cannot be combined with individual bundle ids")
            build_archives(args.bundles, output_dir=args.output_dir, force=args.force)
            return 0
        if args.command == "verify-archives":
            return 0 if verify_archives(args.output_dir) else 1
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
