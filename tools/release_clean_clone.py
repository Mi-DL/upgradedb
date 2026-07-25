#!/usr/bin/env python3
"""Build and smoke-test a fresh-history public clone without publishing it.

The source checkout may contain private data and unsafe reachable history.  This
tool copies only files frozen in ``RELEASE_MANIFEST.sha256`` into a new tree,
creates one commit with a neutral release identity, clones that repository, and
runs the applicable public integrity/privacy/size gates in the clone. A resolved candidate also
runs release smoke; an active-hold code snapshot instead runs the registry/split boundary checks. It never
pushes, uploads, rewrites the source repository, or copies source ``.git`` data.

By default both temporary repositories are deleted after a successful or failed
check.  ``--keep-dir`` is an explicit diagnostic option and refuses an existing
path.  The ``full`` profile additionally verifies and installs all archives from
``--artifacts-dir`` before running the full-payload smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path

import artifact_bundles
import public_release_policy
import release_manifest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = "RELEASE_MANIFEST.sha256"
RELEASE_NAME = "UPGRADE-BENCH Release"
RELEASE_EMAIL = "upgrade-bench-release@example.invalid"
RELEASE_MESSAGE = "Fresh-history public release candidate"
ACTIVE_HOLD_BLOCKER = (
    "v2 result invalidation is unresolved "
    f"({public_release_policy.V2_INVALIDATION_ACTIVE_STATUS})"
)


class CleanCloneError(RuntimeError):
    """Raised when the public export or clone is not fail-closed."""


def _run(args: list[str], *, cwd: Path) -> None:
    rendered = " ".join(args)
    print(f"[clean-clone] {rendered}")
    try:
        subprocess.run(args, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise CleanCloneError(f"required executable is unavailable: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise CleanCloneError(f"command failed with exit code {exc.returncode}: {rendered}") from exc


def _sha256_raw(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_public_manifest(source: Path) -> dict[str, str]:
    manifest = source / PUBLIC_MANIFEST
    unsafe = public_release_policy.source_path_reason(
        PUBLIC_MANIFEST, source, require_file=True
    )
    if unsafe is not None:
        raise CleanCloneError(f"unsafe public manifest: {unsafe}")
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CleanCloneError(f"cannot read public manifest: {exc}") from exc
    if "# Visibility: public-repository-scope" not in text.splitlines()[:3]:
        raise CleanCloneError("root manifest is not labelled public-repository-scope")
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        try:
            digest, name = raw.split("  ", 1)
        except ValueError as exc:
            raise CleanCloneError(
                f"{PUBLIC_MANIFEST}:{line_no}: expected '<sha256>  <path>'"
            ) from exc
        if not release_manifest.HASH_RE.fullmatch(digest):
            raise CleanCloneError(f"{PUBLIC_MANIFEST}:{line_no}: invalid SHA-256")
        if public_release_policy.canonical_path_reason(name) is not None:
            raise CleanCloneError(f"{PUBLIC_MANIFEST}:{line_no}: unsafe path {name!r}")
        if name in entries:
            raise CleanCloneError(f"{PUBLIC_MANIFEST}:{line_no}: duplicate path {name!r}")
        entries[name] = digest
    if not entries:
        raise CleanCloneError("public manifest has no payload entries")
    return entries


def _tree_files(root: Path) -> set[str]:
    paths: set[str] = set()
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if ".git" in item.relative_to(root).parts:
            continue
        if item.is_symlink():
            raise CleanCloneError(f"symbolic link in clean export: {relative}")
        if item.is_file():
            paths.add(relative)
    return paths


def copy_public_tree(source: Path, destination: Path) -> dict[str, str]:
    """Copy exactly the hash-verified public manifest inventory."""
    source = source.resolve()
    if destination.exists():
        raise CleanCloneError(f"refusing existing export path: {destination}")
    entries = _read_public_manifest(source)
    destination.mkdir(parents=True)
    for name, expected in sorted(entries.items()):
        unsafe = public_release_policy.source_path_reason(name, source, require_file=True)
        if unsafe is not None:
            raise CleanCloneError(f"unsafe manifest source {name!r}: {unsafe}")
        excluded = public_release_policy.exclusion_reason(name, source)
        if excluded is not None:
            raise CleanCloneError(f"manifest selected private path {name!r}: {excluded}")
        source_file = source / name
        if release_manifest.sha256(source_file) != expected:
            raise CleanCloneError(f"source changed after manifest verification: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        if release_manifest.sha256(target) != expected:
            raise CleanCloneError(f"copied file hash mismatch: {name}")
    shutil.copyfile(source / PUBLIC_MANIFEST, destination / PUBLIC_MANIFEST)
    expected_files = set(entries) | {PUBLIC_MANIFEST}
    actual_files = _tree_files(destination)
    if actual_files != expected_files:
        raise CleanCloneError(
            "clean export inventory mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    return entries


def _install_full_payload(clone: Path, artifacts_dir: Path, python: str) -> None:
    artifacts_dir = artifacts_dir.resolve()
    if not artifacts_dir.is_dir():
        raise CleanCloneError(f"artifact directory is missing: {artifacts_dir}")
    _run(
        [
            python,
            "tools/artifact_bundles.py",
            "verify-archives",
            "--output-dir",
            str(artifacts_dir),
        ],
        cwd=clone,
    )
    index = json.loads((clone / artifact_bundles.INDEX_PATH.relative_to(ROOT)).read_text(encoding="utf-8"))
    artifact_bundles.validate_index_structure(index, clone)
    for bundle in index["bundles"]:
        archive = artifacts_dir / str(bundle["archive"])
        if not archive.is_file():
            raise CleanCloneError(f"required release archive is missing: {archive.name}")
        expected = {str(item["path"]): item for item in bundle["files"]}
        with zipfile.ZipFile(archive) as handle:
            for name, item in expected.items():
                if public_release_policy.canonical_path_reason(name) is not None:
                    raise CleanCloneError(f"unsafe archive member path: {name!r}")
                target = clone / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if (
                        not target.is_file()
                        or target.stat().st_size != int(item["bytes"])
                        or _sha256_raw(target) != item["sha256"]
                    ):
                        raise CleanCloneError(f"archive conflicts with clone file: {name}")
                    continue
                with handle.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if target.stat().st_size != int(item["bytes"]) or _sha256_raw(target) != item["sha256"]:
                    raise CleanCloneError(f"installed archive member hash mismatch: {name}")


def _assert_neutral_single_commit(clone: Path) -> None:
    count = subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"], cwd=clone, text=True
    ).strip()
    if count != "1":
        raise CleanCloneError(f"fresh-history clone has {count!r} reachable commits, expected one")
    identity = subprocess.check_output(
        ["git", "log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce"],
        cwd=clone,
    ).decode("utf-8").strip().split("\x00")
    if identity != [RELEASE_NAME, RELEASE_EMAIL, RELEASE_NAME, RELEASE_EMAIL]:
        raise CleanCloneError(f"fresh-history commit identity is not neutral: {identity!r}")
    modes = subprocess.check_output(
        ["git", "ls-files", "-s", "-z"], cwd=clone
    ).decode("utf-8", errors="strict")
    if any(record.startswith("120000 ") for record in modes.split("\x00") if record):
        raise CleanCloneError("fresh-history clone contains a tracked symbolic link")


def _audit_mode(root: Path) -> str:
    """Return the only audit mode permitted by the validated release state."""

    blocker = public_release_policy.unresolved_v2_invalidation(root)
    if blocker is None:
        return "final"
    if blocker == ACTIVE_HOLD_BLOCKER:
        return "planned"
    raise CleanCloneError(f"invalid or unsupported release state: {blocker}")


def _run_public_audit(root: Path, python: str, mode: str) -> None:
    command = [python, "tools/public_release_audit.py"]
    if mode == "planned":
        command.append("--planned-only")
    _run(command, cwd=root)


def run_clean_clone(
    *,
    source: Path = ROOT,
    profile: str = "repository",
    artifacts_dir: Path | None = None,
    keep_dir: Path | None = None,
    python: str = sys.executable,
) -> Path | None:
    source = source.resolve()
    if profile not in {"repository", "full"}:
        raise CleanCloneError(f"unsupported profile: {profile}")
    if profile == "full" and artifacts_dir is None:
        raise CleanCloneError("--profile full requires --artifacts-dir")

    audit_mode = _audit_mode(source)
    if audit_mode == "planned" and profile != "repository":
        raise CleanCloneError("active-hold snapshots support only the repository profile")

    # A validated active hold may prove a code-only public boundary, but it cannot
    # enter final audit or full-payload smoke until its receipt is resolved.
    _run([python, "tools/release_manifest.py", "--verify"], cwd=source)
    _run_public_audit(source, python, audit_mode)

    if keep_dir is not None:
        base = keep_dir.resolve()
        if base.exists():
            raise CleanCloneError(f"--keep-dir must not already exist: {base}")
        base.mkdir(parents=True)
        context = nullcontext(str(base))
    else:
        context = tempfile.TemporaryDirectory(prefix="upgrade-bench-public-preflight-")

    with context as base_name:
        base = Path(base_name)
        export = base / "export"
        clone = base / "clone"
        entries = copy_public_tree(source, export)

        _run([python, "tools/release_manifest.py", "--verify"], cwd=export)
        if _audit_mode(export) != audit_mode:
            raise CleanCloneError("export release state differs from source")
        _run_public_audit(export, python, audit_mode)
        _run(["git", "init", "--initial-branch=main"], cwd=export)
        _run(["git", "config", "user.name", RELEASE_NAME], cwd=export)
        _run(["git", "config", "user.email", RELEASE_EMAIL], cwd=export)
        _run(["git", "add", "--all"], cwd=export)
        _run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-m", RELEASE_MESSAGE],
            cwd=export,
        )
        _run(["git", "clone", "--no-local", str(export), str(clone)], cwd=base)
        _run(["git", "remote", "remove", "origin"], cwd=clone)
        _assert_neutral_single_commit(clone)

        tracked = {
            item.decode("utf-8").replace("\\", "/")
            for item in subprocess.check_output(["git", "ls-files", "-z"], cwd=clone).split(b"\x00")
            if item
        }
        expected_tracked = set(entries) | {PUBLIC_MANIFEST}
        if tracked != expected_tracked:
            raise CleanCloneError(
                "fresh clone tracked inventory mismatch: "
                f"missing={sorted(expected_tracked - tracked)}, "
                f"extra={sorted(tracked - expected_tracked)}"
            )

        if profile == "full":
            assert artifacts_dir is not None
            _install_full_payload(clone, artifacts_dir, python)

        _run([python, "tools/release_manifest.py", "--verify"], cwd=clone)
        if _audit_mode(clone) != audit_mode:
            raise CleanCloneError("clone release state differs from source")
        _run_public_audit(clone, python, audit_mode)
        _run([python, "tools/repository_size_gate.py", "--history"], cwd=clone)
        if audit_mode == "planned":
            _run([python, "tools/audit_chain_registry.py", "--check"], cwd=clone)
            _run(
                [
                    python,
                    "-m",
                    "unittest",
                    "tests.test_registry_revision",
                    "tests.test_registry_lexicon_negative_control",
                    "-v",
                ],
                cwd=clone,
            )
            _run([python, "tools/test_split.py"], cwd=clone)
        else:
            _run([python, "tools/release_smoke.py", "--profile", profile], cwd=clone)
        print(
            f"CLEAN PUBLIC CLONE PREFLIGHT PASSED ({profile}; {audit_mode} audit; "
            f"{len(expected_tracked)} tracked files; one neutral commit)"
        )
        return clone if keep_dir is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("repository", "full"), default="repository")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="directory containing all frozen ZIPs/sidecars (required for --profile full)",
    )
    parser.add_argument(
        "--keep-dir",
        type=Path,
        help="keep export/ and clone/ below this new path instead of using temporary storage",
    )
    args = parser.parse_args()
    try:
        run_clean_clone(
            profile=args.profile,
            artifacts_dir=args.artifacts_dir,
            keep_dir=args.keep_dir,
        )
    except (CleanCloneError, OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"CLEAN PUBLIC CLONE PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
