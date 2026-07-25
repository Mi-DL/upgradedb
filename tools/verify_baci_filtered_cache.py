#!/usr/bin/env python3
"""Verify an existing private BACI filtered cache without reading raw BACI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baci_filtered_cache import BaciFilteredCache, REQUIRED_YEARS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = os.environ.get("VCU_BACI_CACHE")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(default) if default else None,
        help="private cache directory (or VCU_BACI_CACHE)",
    )
    parser.add_argument(
        "--years",
        default=",".join(map(str, REQUIRED_YEARS)),
        help="comma-separated years whose decompressed content must be validated",
    )
    args = parser.parse_args()
    if args.cache is None:
        parser.error("--cache (or VCU_BACI_CACHE) is required")
    try:
        requested = [int(piece) for piece in args.years.split(",") if piece.strip()]
    except ValueError as exc:
        parser.error(f"invalid --years: {exc}")
    cache = BaciFilteredCache(args.cache, requested_years=requested)
    rows = {str(year): int(len(cache.read_year(year))) for year in requested}
    source = cache.manifest["source"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema_version": cache.manifest["schema_version"],
                "manifest_years": list(cache.years),
                "validated_content_rows": rows,
                "active_hs6_count": len(cache.active_hs6_codes),
                "raw_archive_name": source["archive_name"],
                "raw_archive_sha256": source["archive_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
