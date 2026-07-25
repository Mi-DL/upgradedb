#!/usr/bin/env python3
"""Enforce conservative GitHub repository size limits before a release.

The gate checks the current tracked worktree and, with ``--history``, every
reachable Git blob.  It intentionally uses limits below GitHub's hard 100 MiB
single-object limit so a release fails locally before a remote push is rejected.
External data bundles are checked separately by ``tools/artifact_bundles.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
DEFAULT_MAX_FILE_MIB = 95.0
DEFAULT_WARN_FILE_MIB = 45.0
DEFAULT_MAX_TOTAL_MIB = 350.0
DEFAULT_WARN_TOTAL_MIB = 300.0


@dataclass(frozen=True)
class SizedPath:
    path: str
    size: int
    object_id: str | None = None


def _git_output(args: list[str], *, root: Path = ROOT, input_bytes: bytes | None = None) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            input=input_bytes,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required for the repository size gate") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}") from exc


def worktree_files(root: Path = ROOT, *, include_untracked: bool = False) -> tuple[list[SizedPath], list[str]]:
    args = ["ls-files", "-z", "--cached"]
    if include_untracked:
        args.extend(["--others", "--exclude-standard"])
    names = {
        part.decode("utf-8").replace("\\", "/")
        for part in _git_output(args, root=root).split(b"\0")
        if part
    }
    files: list[SizedPath] = []
    missing: list[str] = []
    for name in sorted(names):
        target = root / name
        if not target.is_file():
            missing.append(name)
            continue
        files.append(SizedPath(name, target.stat().st_size))
    return files, missing


def reachable_blobs(root: Path = ROOT) -> list[SizedPath]:
    objects = _git_output(["rev-list", "--objects", "--all"], root=root)
    lines = _git_output(
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)"],
        root=root,
        input_bytes=objects,
    ).decode("utf-8", errors="replace")
    blobs: list[SizedPath] = []
    for line in lines.splitlines():
        fields = line.split(" ", 3)
        if len(fields) < 3 or fields[1] != "blob":
            continue
        path = fields[3] if len(fields) == 4 and fields[3] else "<path unavailable>"
        blobs.append(SizedPath(path, int(fields[2]), fields[0]))
    return blobs


def _largest(files: list[SizedPath]) -> SizedPath:
    return max(files, key=lambda item: (item.size, item.path), default=SizedPath("<none>", 0))


def _check_collection(
    label: str,
    files: list[SizedPath],
    *,
    max_file: int,
    warn_file: int,
    max_total: int,
    warn_total: int,
) -> list[str]:
    failures: list[str] = []
    total = sum(item.size for item in files)
    largest = _largest(files)
    print(
        f"{label}: {len(files):,} files/blobs, {total / MIB:.2f} MiB total; "
        f"largest {largest.size / MIB:.2f} MiB ({largest.path})"
    )
    for item in sorted((item for item in files if item.size > max_file), key=lambda item: item.size, reverse=True):
        identity = f" [{item.object_id}]" if item.object_id else ""
        failures.append(f"{label} file exceeds {max_file / MIB:.0f} MiB: {item.path}{identity} ({item.size / MIB:.2f} MiB)")
    if total > max_total:
        failures.append(f"{label} total exceeds {max_total / MIB:.0f} MiB: {total / MIB:.2f} MiB")
    if not failures:
        if largest.size > warn_file:
            print(f"WARNING: {label} largest file exceeds the {warn_file / MIB:.0f} MiB review threshold")
        if total > warn_total:
            print(f"WARNING: {label} total exceeds the {warn_total / MIB:.0f} MiB review threshold")
    return failures


def run_gate(
    *,
    root: Path = ROOT,
    include_untracked: bool = False,
    history: bool = False,
    max_file_mib: float = DEFAULT_MAX_FILE_MIB,
    warn_file_mib: float = DEFAULT_WARN_FILE_MIB,
    max_total_mib: float = DEFAULT_MAX_TOTAL_MIB,
    warn_total_mib: float = DEFAULT_WARN_TOTAL_MIB,
) -> bool:
    if not (0 < warn_file_mib <= max_file_mib and 0 < warn_total_mib <= max_total_mib):
        raise ValueError("warning limits must be positive and no greater than hard limits")
    limits = {
        "max_file": int(max_file_mib * MIB),
        "warn_file": int(warn_file_mib * MIB),
        "max_total": int(max_total_mib * MIB),
        "warn_total": int(warn_total_mib * MIB),
    }
    files, missing = worktree_files(root, include_untracked=include_untracked)
    failures = _check_collection("tracked worktree" + (" + untracked" if include_untracked else ""), files, **limits)
    if missing:
        print(f"NOTE: {len(missing)} tracked path(s) are absent from the worktree and were not sized")
    if history:
        failures.extend(_check_collection("reachable Git history", reachable_blobs(root), **limits))
    if failures:
        for failure in failures:
            print(f"SIZE GATE FAILED: {failure}", file=sys.stderr)
        return False
    print("REPOSITORY SIZE GATE PASSED")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true", help="also audit non-ignored untracked files")
    parser.add_argument("--history", action="store_true", help="audit every reachable Git blob")
    parser.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    parser.add_argument("--warn-file-mib", type=float, default=DEFAULT_WARN_FILE_MIB)
    parser.add_argument("--max-total-mib", type=float, default=DEFAULT_MAX_TOTAL_MIB)
    parser.add_argument("--warn-total-mib", type=float, default=DEFAULT_WARN_TOTAL_MIB)
    args = parser.parse_args()
    try:
        ok = run_gate(
            include_untracked=args.include_untracked,
            history=args.history,
            max_file_mib=args.max_file_mib,
            warn_file_mib=args.warn_file_mib,
            max_total_mib=args.max_total_mib,
            warn_total_mib=args.warn_total_mib,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"SIZE GATE ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
