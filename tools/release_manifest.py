#!/usr/bin/env python3
"""Generate and verify deterministic SHA-256 manifests for release artifacts.

``RELEASE_MANIFEST.sha256`` is explicitly the *public* repository manifest and
uses the same policy as the public bundle selector and privacy audit. The
standalone current benchmark package has its own package-relative manifest.
``internal_release_scope`` remains available for staging-boundary tests, but
this tool never treats a private staging inventory as authorization to publish.

Usage from the repository root:

    python tools/release_manifest.py --write --scope release
    python tools/release_manifest.py --verify

The default verification scope is the public root manifest. ``--scope all``
checks both that manifest and the current standalone package manifest.

Manifest paths are repository-root-relative and use POSIX separators on every OS.
Verification does not require Git. When Git is available, strict verification also
checks that every in-scope tracked or non-ignored release file is listed exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import public_release_policy


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PACKAGE = ROOT / "benchmark" / "upgrade-bench-v2"
CURRENT_PACKAGE_MANIFEST = CURRENT_PACKAGE / "MANIFEST.sha256"
CURRENT_PACKAGE_MANIFEST_TOOL = CURRENT_PACKAGE / "generate_manifest.py"
PUBLIC_RELEASE_MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"
# Backwards-compatible name: the root manifest is public, never internal.
RELEASE_MANIFEST = PUBLIC_RELEASE_MANIFEST

ROOT_RELEASE_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "DATA_LICENSE.md",
    "ARTIFACT.md",
    "BENCHMARK_V2_SPEC.md",
    "configs/v2_gpu_rolling.json",
    "configs/v2_gbdt_baselines.json",
    "configs/v2_product_space_density.json",
    "configs/v2_score_robustness_r5.json",
    "configs/v2_eligibility_threshold_geometry.json",
    "configs/v2_contemporary_references.json",
    "configs/v2_loco_formal.json",
    "configs/v2_ultra_formal.json",
    "PROJECT_CHECKLIST.md",
    "README.md",
    "LOCAL_SETUP.md",
    "RUN_ON_IHPC.md",
    "env.sh.example",
    "run_artifact_smoke.sh",
    "run_artifact_smoke.ps1",
    "paper/generated/v2_numbers.tex",
    "paper/generated/v2_benchmark_profile.tex",
    "paper/generated/v2_contemporary_references.tex",
    "jobs/v2_gpu_select.pbs",
    "jobs/v2_gpu_evaluate.pbs",
    "jobs/v2_gpu_main_worker.sh",
    "jobs/v2_gpu_nohup_worker.sh",
}
# Development-only data and result trees are intentionally absent. The public
# repository contains one current benchmark package plus reviewed summaries.
RELEASE_PREFIXES = (
    ".github/workflows/",
    "benchmark/upgrade-bench-v2/",
    "chains/",
    "docs/",
    "release/",
    "requirements/",
    "results_v2/",
    "src/",
    "tests/",
    "tools/",
)
RELEASE_EXCLUDED_PREFIXES = public_release_policy.INTERNAL_ONLY_PREFIXES
RELEASE_EXCLUDED_FILES = frozenset(
    set(public_release_policy.INTERNAL_ONLY_PATHS)
    | set(public_release_policy.PERMISSION_GATED_PUBLIC_PATHS)
    | set(public_release_policy.INTERNAL_NESTED_MANIFESTS)
)
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".bak", ".tmp"}
BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".zip", ".pth", ".pt", ".npy", ".npz"
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def manifest_header(visibility: str) -> str:
    return (
        f"# Visibility: {visibility}\n"
        "# SHA-256 of canonical content; CRLF is normalized to LF for non-binary files.\n"
        "# Verify with: python tools/release_manifest.py --verify\n"
    )


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _git_visible_files() -> set[str] | None:
    """Return tracked + non-ignored untracked paths, or None outside a Git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return {p.decode("utf-8").replace("\\", "/") for p in out.split(b"\0") if p}


def _filesystem_files() -> set[str]:
    files: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rp = rel(path)
        pp = PurePosixPath(rp)
        if any(part in EXCLUDED_PARTS for part in pp.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.add(rp)
    return files


def visible_files() -> set[str]:
    return _git_visible_files() or _filesystem_files()


def internal_release_scope(files: set[str] | None = None) -> list[str]:
    """Return the private staging inventory before public-policy filtering."""
    files = visible_files() if files is None else files
    excluded = rel(RELEASE_MANIFEST)
    selected = []
    for p in files:
        if p == excluded:
            continue
        if p in ROOT_RELEASE_FILES or any(p.startswith(prefix) for prefix in RELEASE_PREFIXES):
            selected.append(p)
    return sorted(selected)


def public_release_scope(files: set[str] | None = None) -> list[str]:
    """Return paths permitted in the root public release manifest."""
    return [
        path
        for path in internal_release_scope(files)
        if public_release_policy.is_public_path(path, ROOT)
    ]


def release_scope(files: set[str] | None = None) -> list[str]:
    """Compatibility alias; root release scope is explicitly public."""
    return public_release_scope(files)


def sha256(path: Path) -> str:
    """Hash canonical bytes so text artifacts verify on Windows and Linux.

    Git's working-tree line endings depend on the checkout platform. Binary
    suffixes mirror ``.gitattributes``; all other release files are treated as
    text-like and CRLF pairs are normalized without changing lone CR bytes.
    """
    h = hashlib.sha256()
    binary = path.suffix.lower() in BINARY_SUFFIXES
    pending_cr = False
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            if not binary:
                if pending_cr:
                    block = b"\r" + block
                    pending_cr = False
                if block.endswith(b"\r"):
                    block = block[:-1]
                    pending_cr = True
                block = block.replace(b"\r\n", b"\n")
            h.update(block)
    if pending_cr:
        h.update(b"\r")
    return h.hexdigest()


def write_manifest(path: Path, entries: list[str], *, visibility: str = "public") -> None:
    try:
        manifest_name = rel(path)
    except ValueError as exc:
        raise ValueError(f"manifest path is outside the repository: {path}") from exc
    unsafe_manifest = public_release_policy.source_path_reason(manifest_name, ROOT)
    if unsafe_manifest is not None:
        raise ValueError(f"unsafe manifest destination {manifest_name!r}: {unsafe_manifest}")
    if len(entries) != len(set(entries)):
        raise ValueError("manifest entry inventory contains duplicate paths")
    for name in entries:
        unsafe = public_release_policy.source_path_reason(name, ROOT, require_file=True)
        if unsafe is not None:
            raise ValueError(f"unsafe manifest source {name!r}: {unsafe}")
    lines = [f"{sha256(ROOT / p)}  {p}\n" for p in entries]
    path.write_text(manifest_header(visibility) + "".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {rel(path)} ({len(entries)} files)")


def parse_manifest(path: Path) -> dict[str, str]:
    try:
        manifest_name = rel(path)
    except ValueError as exc:
        raise ValueError(f"manifest path is outside the repository: {path}") from exc
    unsafe_manifest = public_release_policy.source_path_reason(
        manifest_name, ROOT, require_file=True
    )
    if unsafe_manifest is not None:
        raise ValueError(f"unsafe manifest {manifest_name!r}: {unsafe_manifest}")
    entries: dict[str, str] = {}
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read manifest {manifest_name}: {exc}") from exc
    for line_no, raw in enumerate(raw_lines, 1):
        if not raw or raw.startswith("#"):
            continue
        try:
            digest, name = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"{rel(path)}:{line_no}: expected '<sha256>  <path>'") from exc
        pp = PurePosixPath(name)
        if not HASH_RE.fullmatch(digest):
            raise ValueError(f"{rel(path)}:{line_no}: invalid SHA-256")
        if pp.is_absolute() or ".." in pp.parts or name != pp.as_posix():
            raise ValueError(f"{rel(path)}:{line_no}: unsafe/non-canonical path {name!r}")
        if name in entries:
            raise ValueError(f"{rel(path)}:{line_no}: duplicate path {name!r}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"empty manifest: {rel(path)}")
    return entries


def verify_manifest(path: Path, expected_scope: list[str] | None = None) -> bool:
    try:
        entries = parse_manifest(path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False

    ok = True
    listed = set(entries)
    if expected_scope is not None:
        expected = set(expected_scope)
        for name in sorted(expected - listed):
            print(f"UNLISTED: {name}", file=sys.stderr)
            ok = False
        for name in sorted(listed - expected):
            print(f"OUT-OF-SCOPE: {name}", file=sys.stderr)
            ok = False

    for name, expected_digest in entries.items():
        target = ROOT / name
        unsafe = public_release_policy.source_path_reason(name, ROOT, require_file=True)
        if unsafe is not None:
            label = "MISSING" if unsafe == "selected repository file is missing" else "UNSAFE"
            print(f"{label}: {name} ({unsafe})", file=sys.stderr)
            ok = False
            continue
        actual = sha256(target)
        if actual != expected_digest:
            print(f"MISMATCH: {name}\n  expected {expected_digest}\n  actual   {actual}", file=sys.stderr)
            ok = False

    label = rel(path)
    if ok:
        print(f"verified {label} ({len(entries)} files)")
    else:
        print(f"FAILED {label}", file=sys.stderr)
    return ok


def write_all(scope: str) -> None:
    files = visible_files()
    if scope in ("package", "all"):
        subprocess.run(
            [sys.executable, str(CURRENT_PACKAGE_MANIFEST_TOOL), "--write"],
            cwd=ROOT,
            check=True,
        )
        files.add(rel(CURRENT_PACKAGE_MANIFEST))
    if scope in ("release", "all"):
        write_manifest(
            RELEASE_MANIFEST,
            public_release_scope(files),
            visibility="public-repository-scope",
        )


def verify_all(scope: str = "release", strict: bool = True) -> bool:
    files = visible_files() if strict else None
    checks = []
    if scope in ("package", "all"):
        checks.append(
            subprocess.run(
                [sys.executable, str(CURRENT_PACKAGE_MANIFEST_TOOL)],
                cwd=ROOT,
                check=False,
            ).returncode
            == 0
        )
    if scope in ("release", "all"):
        checks.append(verify_manifest(RELEASE_MANIFEST, release_scope(files) if strict else None))
    return all(checks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="regenerate the selected manifest(s)")
    action.add_argument("--verify", action="store_true", help="verify hashes (default)")
    ap.add_argument("--scope", choices=("package", "release", "all"), default="release")
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="verify listed files only; do not compare against the current release scope",
    )
    args = ap.parse_args()

    if args.write:
        write_all(args.scope)
        return 0
    return 0 if verify_all(args.scope, strict=not args.no_strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
