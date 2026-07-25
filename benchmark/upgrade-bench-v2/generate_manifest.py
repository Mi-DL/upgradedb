#!/usr/bin/env python3
"""Write or verify the standalone package's deterministic SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.sha256"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "data"}
EXCLUDED_NAMES = {MANIFEST.name}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak"}


def package_files(root: Path = HERE) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in EXCLUDED_NAMES
            and path.suffix.lower() not in EXCLUDED_SUFFIXES
            and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_manifest(root: Path = HERE) -> str:
    lines = ["# SHA-256  package-relative path\n"]
    lines.extend(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in package_files(root)
    )
    return "".join(lines)


def write_manifest(path: Path = MANIFEST, root: Path = HERE) -> None:
    payload = render_manifest(root)
    path.write_text(payload, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(package_files(root))} files)")


def verify_manifest(path: Path = MANIFEST, root: Path = HERE) -> bool:
    if not path.is_file():
        print(f"MANIFEST FAILED: missing {path}", file=sys.stderr)
        return False
    expected = render_manifest(root)
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        print("MANIFEST FAILED: package inventory or SHA-256 mismatch", file=sys.stderr)
        return False
    print(f"verified {path} ({len(package_files(root))} files)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace MANIFEST.sha256 with current inventory")
    args = parser.parse_args()
    if args.write:
        write_manifest()
        return 0
    return 0 if verify_manifest() else 1


if __name__ == "__main__":
    raise SystemExit(main())
