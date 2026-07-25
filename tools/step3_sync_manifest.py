"""Generate/verify the fail-closed Step-3 GPU sync manifest mechanically."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXED = (
    "configs/v2_gpu_rolling.json",
    "chains/evidence/registry_evidence.json",
    "docs/registry_audit.json",
    "jobs/v2_gpu_main_worker.sh",
    "jobs/v2_gpu_nohup_worker.sh",
    "requirements/v2-gpu-nodeps-lock.txt",
    "results_v2/metrics/b1_candidate_coverage.json",
    "results_v2/metrics/raw_label_audit.json",
    "src/baci_filtered_cache.py",
    "src/benchmark.py",
    "src/gap_discovery.py",
    "src/split.py",
    "src/task_features.py",
    "src/temporal_backtest.py",
    "src/universe.py",
    "src/v2_gpu_protocol.py",
    "src/v2_gpu_rolling.py",
    "src/window_aggregation.py",
    "tools/v2_gpu_env_check.py",
    "tools/step3_sync_manifest.py",
)
LINE = re.compile(r"^[0-9a-f]{64}  [^\r\n]+$")


def expected_files() -> list[Path]:
    """Return the complete sync inventory without opening external payloads."""

    paths = [ROOT / name for name in FIXED]
    paths.extend(sorted((ROOT / "chains").glob("*.json")))
    # Hashing is byte-level provenance only: it never parses target labels.
    # Pin both historical and main candidate cohorts so a file cannot change
    # between selection and one-shot evaluation while still passing the worker
    # gate. The runner remains the only code that semantically opens main data.
    for chain in sorted((ROOT / "chains").glob("*.json")):
        for prefix in ("candidates", "candidates_firsttime"):
            paths.append(ROOT / "data/processed_v2" / f"{prefix}_{chain.stem}.csv")
            paths.append(
                ROOT / "data/processed_v2" / f"{prefix}_{chain.stem}_fold2.csv"
            )
    return sorted(set(path.resolve() for path in paths))


def files() -> list[Path]:
    paths = expected_files()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing sync inputs: {missing}")
    return paths


def render() -> str:
    lines = []
    for path in files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        line = f"{digest}  {relative}"
        if not LINE.fullmatch(line):
            raise SystemExit(f"invalid manifest line: {line!r}")
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="run-specific manifest path below the isolated run root",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = render()
    output = args.output.resolve()
    if args.verify:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"stale/invalid Step-3 sync manifest: {output}")
        print(f"verified {len(expected.splitlines())} hashes: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {len(expected.splitlines())} hashes: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
