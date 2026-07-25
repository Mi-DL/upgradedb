#!/usr/bin/env python3
"""Build the private, registry-bound BACI cache used for cohort regeneration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baci_filtered_cache import (  # noqa: E402
    DEFAULT_AUDIT_PATH,
    DEFAULT_CHAINS_DIR,
    DEFAULT_EVIDENCE_PATH,
    REQUIRED_YEARS,
    build_cache,
)


def main() -> int:
    default_raw = Path(
        os.environ.get("VCU_RAW", str(ROOT / "data" / "raw"))
    ) / "BACI_HS92_V202401b.zip"
    parser = argparse.ArgumentParser(
        description=(
            "Filter the 15 main/fold2 BACI years once to the exact six-chain audited "
            "HS6 union. The output is private and is accepted only below a "
            "private/tmp path. Existing output directories are never replaced."
        )
    )
    parser.add_argument(
        "--baci-zip",
        type=Path,
        default=default_raw,
        help="private BACI_HS92_V202401b.zip (default: VCU_RAW)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new private/tmp cache directory; must not already exist",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="source CSV rows read per chunk (default: 500000)",
    )
    parser.add_argument("--chains-dir", type=Path, default=DEFAULT_CHAINS_DIR)
    parser.add_argument("--registry-audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--registry-evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    args = parser.parse_args()

    manifest = build_cache(
        args.baci_zip,
        args.output,
        chains_dir=args.chains_dir,
        audit_path=args.registry_audit,
        evidence_path=args.registry_evidence,
        years=REQUIRED_YEARS,
        chunk_rows=args.chunk_rows,
    )
    print(
        "BACI private cache complete: "
        f"years={len(manifest['years'])}, "
        f"HS6={len(manifest['registry']['active_hs6_union'])}, "
        f"rows={manifest['totals']['rows']:,}, "
        f"compressed_bytes={manifest['totals']['bytes']:,}"
    )
    print(f"manifest: {Path(args.output).expanduser().resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
