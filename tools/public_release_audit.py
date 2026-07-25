#!/usr/bin/env python3
"""Fail if any selected public payload leaks gated data or host identity.

The audit scans both the public repository manifest selector and every planned
bundle payload.  When the frozen bundle index exists it scans that inventory as
well.  Path eligibility is imported from the same policy module used by both
selectors, so this audit is defense in depth rather than a divergent denylist.
"""

from __future__ import annotations

import argparse
import codecs
import re
import sys
from pathlib import Path

import artifact_bundles
import public_release_policy
import release_manifest


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bib", ".cfg", ".csv", ".ini", ".json", ".jsonl", ".lock", ".md",
    ".pbs", ".ps1", ".py", ".r", ".rst", ".sh", ".sha256", ".tex", ".toml",
    ".tsv", ".txt", ".yaml", ".yml", ".example",
}
TEXT_FILENAMES = {
    ".gitattributes", ".gitignore", "Dockerfile", "LICENSE", "Makefile", "MANIFEST", "VERSION",
}
SCAN_CHUNK_BYTES = 1024 * 1024
SCAN_OVERLAP_CHARS = 1024
SENSITIVE_PATTERNS = (
    (
        "absolute Unix home path",
        re.compile(r"(?:^|[=:`'\"\s])/(?:home|shared/homes)/[A-Za-z0-9._-]+(?:/|\b)", re.MULTILINE),
    ),
    (
        "absolute macOS user path",
        re.compile(r"(?:^|[=:`'\"\s])/Users/[A-Za-z0-9._-]+(?:/|\b)", re.MULTILINE),
    ),
    (
        "absolute Windows user path",
        # ``\\+`` catches both an ordinary Windows path and JSON source text
        # where every separator in a drive-letter user path is escaped in JSON.
        re.compile(
            r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+(?:[\\/]+|\b)",
            re.IGNORECASE,
        ),
    ),
    ("institutional host FQDN", re.compile(r"\bmars\d+\.ihpc\.[A-Za-z0-9.-]+", re.IGNORECASE)),
    ("bare institutional host alias", re.compile(r"\bmars\d+\b", re.IGNORECASE)),
    ("cluster account family", re.compile(r"\bsli\d+\b", re.IGNORECASE)),
)


def _sensitive_labels(text: str) -> list[str]:
    return [label for label, pattern in SENSITIVE_PATTERNS if pattern.search(text)]


def _scan_text_file(path: Path) -> tuple[list[str], str | None]:
    """Stream-scan an entire UTF-8 payload, including large CSV artifacts."""
    labels: set[str] = set()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    tail = ""
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(SCAN_CHUNK_BYTES), b""):
                text = tail + decoder.decode(block)
                labels.update(_sensitive_labels(text))
                tail = text[-SCAN_OVERLAP_CHARS:]
        final = tail + decoder.decode(b"", final=True)
        labels.update(_sensitive_labels(final))
    except UnicodeDecodeError:
        return [], "not UTF-8"
    return sorted(labels), None


def audit_selected_files(
    paths: list[str] | set[str],
    root: Path = ROOT,
    *,
    surface: str = "public release",
) -> list[str]:
    failures: list[str] = []
    for name in sorted(set(paths)):
        reason = public_release_policy.exclusion_reason(name, root)
        if reason is not None:
            failures.append(f"{surface} selected {reason}: {name}")
            continue
        path = root / name
        if not path.is_file() or (
            path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES
        ):
            continue
        labels, error = _scan_text_file(path)
        if error is not None:
            failures.append(f"declared text {surface} file is not UTF-8: {name}")
            continue
        for label in labels:
            failures.append(f"{label} in {surface} file: {name}")
    return failures


def audit_bundle_paths(paths: set[str], root: Path = ROOT) -> list[str]:
    return [
        f"public bundle selected {reason}: {name}"
        for name in sorted(paths)
        if (reason := public_release_policy.exclusion_reason(name, root)) is not None
    ]


def audit(*, root: Path = ROOT, planned_only: bool = False) -> bool:
    release_paths = set(release_manifest.public_release_scope())
    planned = {path for spec in artifact_bundles.bundle_specs(root) for path in spec.paths}
    selected = release_paths | planned
    failures = audit_bundle_paths(planned, root)
    if not planned_only:
        blocker = public_release_policy.unresolved_v2_invalidation(root)
        if blocker is not None:
            failures.append(f"final public freeze is blocked: {blocker}")
        try:
            index = artifact_bundles.load_index()
            artifact_bundles.validate_index_structure(index, root)
            frozen = {str(item["path"]) for bundle in index["bundles"] for item in bundle["files"]}
            failures.extend(audit_bundle_paths(frozen, root))
            selected.update(frozen)
        except ValueError as exc:
            failures.append(f"cannot audit frozen public index: {exc}")
    failures.extend(audit_selected_files(selected, root, surface="selected public surface"))
    if failures:
        for failure in sorted(set(failures)):
            print(f"PUBLIC RELEASE AUDIT FAILED: {failure}", file=sys.stderr)
        return False
    mode = "public repository and planned bundle contents" if planned_only else (
        "public repository, planned bundle contents, and frozen index"
    )
    print(f"PUBLIC RELEASE AUDIT PASSED ({mode})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planned-only",
        action="store_true",
        help="audit current selectors without requiring the final frozen data index",
    )
    args = parser.parse_args()
    return 0 if audit(planned_only=args.planned_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
