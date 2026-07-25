#!/usr/bin/env python3
"""Verify the minimal environment recorded for the v2 rolling CPU results.

Package and Python mismatches are errors. A platform mismatch is reported as a
warning by default because the lock is installable on supported non-Windows
platforms, but numerical output is not claimed to be byte-identical there. Pass
``--strict-platform`` when checking the original result-generation platform.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name("v2-cpu-results-lock.txt")
RESULT = ROOT / "results_v2" / "metrics" / "rolling_cpu_baselines.json"
SCRIPT = ROOT / "tools" / "v2_rolling_cpu_baselines.py"
META_PREFIX = "# v2_cpu_results_"


def parse_lock(path: Path = LOCK) -> tuple[dict[str, str], dict[str, str]]:
    """Return recorded metadata and exact package pins from the minimal lock."""
    metadata: dict[str, str] = {}
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(META_PREFIX):
            key, separator, value = line[len(META_PREFIX) :].partition(":")
            if not separator or not key.strip() or not value.strip():
                raise ValueError(f"invalid metadata line in {path}: {raw_line!r}")
            metadata[key.strip()] = value.strip()
            continue
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"lock entries must be exact pins: {raw_line!r}")
        name, version = match.groups()
        canonical = name.lower().replace("_", "-")
        if canonical in pins:
            raise ValueError(f"duplicate lock entry: {name}")
        pins[canonical] = version
    return metadata, pins


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_environment(*, strict_platform: bool = False) -> dict[str, Any]:
    metadata, pins = parse_lock()
    errors: list[str] = []
    warnings: list[str] = []

    actual_python = platform.python_version()
    expected_python = metadata.get("python")
    if actual_python != expected_python:
        errors.append(f"Python: expected {expected_python}, found {actual_python}")

    actual_implementation = platform.python_implementation()
    expected_implementation = metadata.get("implementation")
    if actual_implementation != expected_implementation:
        errors.append(
            "implementation: expected "
            f"{expected_implementation}, found {actual_implementation}"
        )

    actual_packages: dict[str, str | None] = {}
    for name, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        actual_packages[name] = actual
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual or 'not installed'}")

    actual_platform = platform.platform()
    expected_platform = metadata.get("platform")
    if actual_platform != expected_platform:
        message = f"platform: result used {expected_platform}, current host is {actual_platform}"
        (errors if strict_platform else warnings).append(message)

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    runtime = result.get("runtime", {})
    expected_runtime = {
        "python": expected_python,
        "platform": expected_platform,
        "numpy": pins.get("numpy"),
        "pandas": pins.get("pandas"),
        "scikit_learn": pins.get("scikit-learn"),
    }
    for key, expected in expected_runtime.items():
        actual = runtime.get(key)
        if actual != expected:
            errors.append(
                f"result runtime.{key}: lock records {expected!r}, artifact records {actual!r}"
            )

    recorded_script_hash = runtime.get("script_sha256")
    actual_script_hash = sha256(SCRIPT)
    if recorded_script_hash != actual_script_hash:
        errors.append(
            "rolling script hash: result records "
            f"{recorded_script_hash!r}, current source is {actual_script_hash!r}"
        )

    return {
        "ok": not errors,
        "strict_platform": strict_platform,
        "python": actual_python,
        "implementation": actual_implementation,
        "platform": actual_platform,
        "recorded_platform": expected_platform,
        "packages": actual_packages,
        "result_artifact": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-platform",
        action="store_true",
        help="treat a host-platform mismatch as an error",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)

    report = inspect_environment(strict_platform=args.strict_platform)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"v2 CPU results environment: {status}")
        print(
            f"Python {report['python']} ({report['implementation']}); "
            f"platform {report['platform']}"
        )
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
