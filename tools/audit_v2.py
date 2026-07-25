#!/usr/bin/env python
"""Audit v2 lane labels directly against raw BACI trade rows.

This is deliberately independent of the benchmark's aggregation implementation.
For each requested five-year window it:

1. reads the raw BACI year files;
2. sums HS6 values to ``(chain, exporter, importer, stage, year)``; and
3. sums those annual stage values and divides by exactly five calendar years.

Missing stage-years therefore contribute zero.  The audit never reads v1 candidate
tables and never writes candidate data; its only output is a JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import universe as U  # noqa: E402


RAW_DIR = Path(os.environ.get("VCU_RAW", ROOT / "data" / "raw"))
BACI_ZIP = RAW_DIR / "BACI_HS92_V202401b.zip"
PROC_V2 = ROOT / "data" / "processed_v2"
DEFAULT_OUTPUT = ROOT / "results_v2" / "metrics" / "raw_label_audit.json"

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
SNAPSHOTS = {
    "main": {
        "early": (2008, 2009, 2010, 2011, 2012),
        "late": (2018, 2019, 2020, 2021, 2022),
        "suffix": "",
        "temporal_role": "target",
    },
    "fold2": {
        "early": (1998, 1999, 2000, 2001, 2002),
        "late": (2008, 2009, 2010, 2011, 2012),
        "suffix": "_fold2",
        "temporal_role": "history",
    },
}
TRACKS = {
    "A": {
        "prefix": "candidates",
        "description": "destination extension lane",
    },
    "B": {
        "prefix": "candidates_firsttime",
        "description": "first-time processed-export candidate lane",
    },
}

KEYS = ["i_iso", "j_iso", "stage"]
RAW_KEYS = ["chain", *KEYS]
VALUE_THRESHOLD_KUSD = 100.0
WINDOW_LENGTH = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Use a portable repository-relative path where possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _flatten_choices(values: Iterable[str]) -> list[str]:
    """Accept both ``--chains sheep cotton`` and ``--chains sheep,cotton``."""
    choices: list[str] = []
    for value in values:
        choices.extend(part.strip() for part in value.split(",") if part.strip())
    return choices


def _normalise_choices(
    parser: argparse.ArgumentParser,
    values: Iterable[str],
    valid: Iterable[str],
    label: str,
    *,
    uppercase: bool = False,
) -> list[str]:
    out = _flatten_choices(values)
    if uppercase:
        out = [value.upper() for value in out]
    valid_order = list(valid)
    unknown = sorted(set(out) - set(valid_order))
    if unknown:
        parser.error(f"unknown {label}: {unknown}; choose from {valid_order}")
    # Preserve the canonical order and remove duplicates.
    selected = set(out)
    return [value for value in valid_order if value in selected]


def _candidate_path(chain: str, snapshot: str, track: str) -> Path:
    spec = SNAPSHOTS[snapshot]
    return PROC_V2 / f"{TRACKS[track]['prefix']}_{chain}{spec['suffix']}.csv"


def _window_label(years: Iterable[int]) -> str:
    years = tuple(years)
    return f"{years[0]}-{years[-1]}"


def _load_candidate_tables(
    chains: list[str], snapshots: list[str], tracks: list[str]
) -> tuple[dict[tuple[str, str, str], pd.DataFrame], pd.DataFrame]:
    tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    key_parts: list[pd.DataFrame] = []
    required = set(KEYS + ["y", "lateval", "aggregation", "early_window", "late_window"])

    for snapshot in snapshots:
        for track in tracks:
            for chain in chains:
                path = _candidate_path(chain, snapshot, track)
                if not path.is_file():
                    raise FileNotFoundError(f"missing v2 candidate table: {path}")
                header = pd.read_csv(path, nrows=0)
                missing = sorted(required - set(header.columns))
                if missing:
                    raise ValueError(f"{path} is missing required columns: {missing}")
                usecols = KEYS + ["y", "lateval", "aggregation", "early_window", "late_window"]
                cand = pd.read_csv(
                    path,
                    usecols=usecols,
                    dtype={"i_iso": str, "j_iso": str, "stage": str},
                )
                duplicate_rows = int(cand.duplicated(KEYS, keep=False).sum())
                if duplicate_rows:
                    raise ValueError(
                        f"{path} has {duplicate_rows} rows participating in duplicate lane keys"
                    )
                cand["y"] = pd.to_numeric(cand["y"], errors="raise")
                cand["lateval"] = pd.to_numeric(cand["lateval"], errors="raise")
                tables[(chain, snapshot, track)] = cand

                keys = cand[KEYS].copy()
                keys.insert(0, "chain", chain)
                key_parts.append(keys)
                print(
                    f"loaded {path.relative_to(ROOT)}: rows={len(cand):,} "
                    f"positives={int((cand.y == 1).sum()):,}"
                )

    candidate_keys = pd.concat(key_parts, ignore_index=True).drop_duplicates(RAW_KEYS)
    return tables, candidate_keys


def _chain_stage_maps(chains: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    hs_to_chain: dict[str, str] = {}
    hs_to_stage: dict[str, str] = {}
    for chain in chains:
        registry = U.get_chain(chain)
        for hs6, stage in registry.hs2stage.items():
            if hs6 in hs_to_chain:
                raise ValueError(
                    f"HS6 {hs6} occurs in both {hs_to_chain[hs6]} and {chain}; "
                    "the raw audit requires an unambiguous chain-stage mapping"
                )
            hs_to_chain[hs6] = chain
            hs_to_stage[hs6] = stage
    return hs_to_chain, hs_to_stage


def _zip_member(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        return zf.read(name)
    except KeyError as exc:
        raise FileNotFoundError(f"{name} not found in {zf.filename}") from exc


def _country_iso_map(zf: zipfile.ZipFile) -> dict[int, str]:
    countries = pd.read_csv(io.BytesIO(_zip_member(zf, "country_codes_V202401b.csv")))
    countries = countries.dropna(subset=["country_code", "country_iso3"])
    return dict(
        zip(
            pd.to_numeric(countries["country_code"], errors="raise").astype(int),
            countries["country_iso3"].astype(str),
        )
    )


def _read_stage_year(
    zf: zipfile.ZipFile,
    year: int,
    iso: dict[int, str],
    hs_to_chain: dict[str, str],
    hs_to_stage: dict[str, str],
    candidate_keys: pd.DataFrame,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read one raw year and independently sum HS6 rows to stage-year lanes."""
    member = f"BACI_HS92_Y{year}_V202401b.csv"
    hs_universe = set(hs_to_chain)
    grouped_chunks: list[pd.DataFrame] = []
    source_rows = 0
    retained_hs6_rows = 0

    try:
        stream = zf.open(member)
    except KeyError as exc:
        raise FileNotFoundError(f"{member} not found in {zf.filename}") from exc

    with stream:
        reader = pd.read_csv(
            stream,
            usecols=["i", "j", "k", "v"],
            dtype={"k": str},
            chunksize=chunk_size,
        )
        for chunk in reader:
            source_rows += len(chunk)
            chunk["k"] = chunk["k"].str.zfill(6)
            chunk = chunk[chunk["k"].isin(hs_universe)].copy()
            retained_hs6_rows += len(chunk)
            if chunk.empty:
                continue
            chunk["chain"] = chunk["k"].map(hs_to_chain)
            chunk["stage"] = chunk["k"].map(hs_to_stage)
            chunk["i_iso"] = chunk["i"].map(iso)
            chunk["j_iso"] = chunk["j"].map(iso)
            chunk["v"] = pd.to_numeric(chunk["v"], errors="raise")
            chunk = chunk.dropna(subset=RAW_KEYS + ["v"])
            # First sum within each chunk. A second group-by below combines any
            # stage-year lane split across chunk boundaries.
            grouped_chunks.append(
                chunk.groupby(RAW_KEYS, as_index=False, sort=False)["v"].sum()
            )

    if grouped_chunks:
        annual = (
            pd.concat(grouped_chunks, ignore_index=True)
            .groupby(RAW_KEYS, as_index=False, sort=False)["v"]
            .sum()
        )
        # Raw BACI is large, but the audit only needs lanes present in one of the
        # selected v2 candidate tables. This is a read-only semijoin.
        annual = annual.merge(candidate_keys, on=RAW_KEYS, how="inner", validate="one_to_one")
    else:
        annual = pd.DataFrame(columns=RAW_KEYS + ["v"])

    return annual, {
        "source_rows": int(source_rows),
        "retained_chain_hs6_rows": int(retained_hs6_rows),
        "candidate_stage_year_rows": int(len(annual)),
    }


def _independent_calendar_means(
    zf: zipfile.ZipFile,
    years: list[int],
    iso: dict[int, str],
    hs_to_chain: dict[str, str],
    hs_to_stage: dict[str, str],
    candidate_keys: pd.DataFrame,
    chunk_size: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compute stage-year sums, then divide each fixed window total by five."""
    period_for_year: dict[int, str] = {}
    for year in years:
        period_for_year[year] = next(
            _window_label(window)
            for snapshot in SNAPSHOTS.values()
            for window in (snapshot["early"], snapshot["late"])
            if year in window
        )

    parts: dict[str, list[pd.DataFrame]] = {}
    year_stats: dict[str, dict[str, int]] = {}
    for index, year in enumerate(years, start=1):
        started = time.perf_counter()
        annual, stats = _read_stage_year(
            zf,
            year,
            iso,
            hs_to_chain,
            hs_to_stage,
            candidate_keys,
            chunk_size,
        )
        parts.setdefault(period_for_year[year], []).append(annual)
        stats["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        year_stats[str(year)] = stats
        print(
            f"raw {year} [{index}/{len(years)}]: source={stats['source_rows']:,} "
            f"chain_hs6={stats['retained_chain_hs6_rows']:,} "
            f"candidate_stage_year={stats['candidate_stage_year_rows']:,} "
            f"elapsed={stats['elapsed_seconds']:.1f}s"
        )

    means: dict[str, pd.DataFrame] = {}
    for period, annual_parts in parts.items():
        window_years = [year for year in years if period_for_year[year] == period]
        if len(window_years) != WINDOW_LENGTH:
            raise AssertionError(
                f"period {period} has {len(window_years)} selected years, expected {WINDOW_LENGTH}"
            )
        if annual_parts:
            mean = (
                pd.concat(annual_parts, ignore_index=True)
                .groupby(RAW_KEYS, as_index=False, sort=False)["v"]
                .sum()
            )
            # Critical independent definition: absent annual rows are implicit
            # zeros because every window total is divided by all five years.
            mean["raw_calendar_mean_kusd"] = mean.pop("v") / float(WINDOW_LENGTH)
        else:
            mean = pd.DataFrame(columns=RAW_KEYS + ["raw_calendar_mean_kusd"])
        means[period] = mean

    return means, {"years": year_stats}


def _compact_examples(frame: pd.DataFrame, mask: np.ndarray, columns: list[str]) -> list[dict]:
    if not bool(mask.any()):
        return []
    examples = frame.loc[mask, columns].head(10).copy()
    for column in examples.select_dtypes(include=[np.number]).columns:
        examples[column] = examples[column].map(lambda value: round(float(value), 9))
    return examples.to_dict(orient="records")


def _metadata_check(cand: pd.DataFrame, snapshot: str) -> dict[str, object]:
    spec = SNAPSHOTS[snapshot]
    expected = {
        "aggregation": "calendar_mean",
        "early_window": _window_label(spec["early"]),
        "late_window": _window_label(spec["late"]),
    }
    observed = {
        column: sorted(cand[column].dropna().astype(str).unique().tolist())
        for column in expected
    }
    passed = all(observed[column] == [value] for column, value in expected.items())
    return {"pass": bool(passed), "expected": expected, "observed": observed}


def _audit_table(
    cand: pd.DataFrame,
    chain: str,
    snapshot: str,
    track: str,
    means: dict[str, pd.DataFrame],
    atol: float,
    rtol: float,
) -> dict[str, object]:
    spec = SNAPSHOTS[snapshot]
    early_period = _window_label(spec["early"])
    late_period = _window_label(spec["late"])

    def window_values(period: str, output_col: str) -> pd.DataFrame:
        frame = means[period]
        frame = frame[frame["chain"] == chain][KEYS + ["raw_calendar_mean_kusd"]].copy()
        return frame.rename(columns={"raw_calendar_mean_kusd": output_col})

    audited = cand.merge(
        window_values(early_period, "raw_early_mean_kusd"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    ).merge(
        window_values(late_period, "raw_late_mean_kusd"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    audited[["raw_early_mean_kusd", "raw_late_mean_kusd"]] = audited[
        ["raw_early_mean_kusd", "raw_late_mean_kusd"]
    ].fillna(0.0)

    stored_y_valid = audited["y"].isin([0, 1]).to_numpy()
    expected_y = (audited["raw_late_mean_kusd"].to_numpy() > VALUE_THRESHOLD_KUSD).astype(int)
    stored_y = audited["y"].to_numpy()
    label_bad = (~stored_y_valid) | (stored_y != expected_y)

    expected_lateval = np.where(
        expected_y == 1,
        audited["raw_late_mean_kusd"].to_numpy(),
        0.0,
    )
    stored_lateval = audited["lateval"].to_numpy(float)
    finite_lateval = np.isfinite(stored_lateval)
    lateval_close = np.isclose(
        stored_lateval,
        expected_lateval,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )
    lateval_bad = (~finite_lateval) | (~lateval_close)

    early_bad = audited["raw_early_mean_kusd"].to_numpy() > VALUE_THRESHOLD_KUSD + atol
    negative_rows = stored_y == 0
    negative_lateval_bad = negative_rows & (
        (~finite_lateval) | (~np.isclose(stored_lateval, 0.0, atol=atol, rtol=0.0))
    )
    below_zero_bad = finite_lateval & (stored_lateval < -atol)

    abs_error = np.abs(stored_lateval - expected_lateval)
    rel_error = abs_error / np.maximum(np.abs(expected_lateval), atol)
    metadata = _metadata_check(cand, snapshot)

    checks = {
        "metadata": metadata,
        "y_reconciliation": {
            "pass": bool(not label_bad.any()),
            "mismatches": int(label_bad.sum()),
            "agreement": float(1.0 - label_bad.mean()) if len(audited) else 1.0,
            "invalid_stored_y": int((~stored_y_valid).sum()),
            "examples": _compact_examples(
                audited.assign(expected_y=expected_y),
                label_bad,
                KEYS + ["y", "expected_y", "raw_late_mean_kusd"],
            ),
        },
        "lateval_reconciliation": {
            "pass": bool(not lateval_bad.any()),
            "mismatches": int(lateval_bad.sum()),
            "max_absolute_error_kusd": (
                float(np.nanmax(abs_error)) if len(abs_error) else 0.0
            ),
            "max_relative_error": (
                float(np.nanmax(rel_error)) if len(rel_error) else 0.0
            ),
            "atol_kusd": atol,
            "rtol": rtol,
            "examples": _compact_examples(
                audited.assign(expected_lateval_kusd=expected_lateval),
                lateval_bad,
                KEYS
                + ["y", "lateval", "expected_lateval_kusd", "raw_late_mean_kusd"],
            ),
        },
        "early_absence": {
            "pass": bool(not early_bad.any()),
            "violations_above_100_kusd": int(early_bad.sum()),
            "max_raw_early_mean_kusd": (
                float(audited["raw_early_mean_kusd"].max()) if len(audited) else 0.0
            ),
            "examples": _compact_examples(
                audited,
                early_bad,
                KEYS + ["y", "raw_early_mean_kusd"],
            ),
        },
        "negative_class_lateval_zero": {
            "pass": bool(not negative_lateval_bad.any()),
            "negative_rows": int(negative_rows.sum()),
            "violations": int(negative_lateval_bad.sum()),
            "examples": _compact_examples(
                audited,
                negative_lateval_bad,
                KEYS + ["y", "lateval", "raw_late_mean_kusd"],
            ),
        },
        "lateval_nonnegative": {
            "pass": bool(not below_zero_bad.any()),
            "violations": int(below_zero_bad.sum()),
        },
    }
    all_pass = all(check["pass"] for check in checks.values())
    path = _candidate_path(chain, snapshot, track)
    result = {
        "chain": chain,
        "snapshot": snapshot,
        "track": track,
        "track_description": TRACKS[track]["description"],
        "candidate_file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "candidate_sha256": _sha256(path),
        "rows": int(len(audited)),
        "stored_positive_rows": int((stored_y == 1).sum()),
        "raw_positive_rows": int(expected_y.sum()),
        "raw_observed_late_value_kusd": float(expected_lateval.sum()),
        "checks": checks,
        "all_pass": bool(all_pass),
    }
    print(
        f"audit {snapshot:5s} Track {track} {chain:12s}: rows={len(audited):,} "
        f"y_bad={int(label_bad.sum()):,} lateval_bad={int(lateval_bad.sum()):,} "
        f"early_bad={int(early_bad.sum()):,} neg_lateval_bad={int(negative_lateval_bad.sum()):,} "
        f"=> {'PASS' if all_pass else 'FAIL'}"
    )
    return result


def _summary(audits: list[dict[str, object]]) -> dict[str, object]:
    return {
        "audit_instances": len(audits),
        "passing_instances": sum(bool(audit["all_pass"]) for audit in audits),
        "failing_instances": sum(not bool(audit["all_pass"]) for audit in audits),
        "candidate_rows": sum(int(audit["rows"]) for audit in audits),
        "stored_positive_rows": sum(int(audit["stored_positive_rows"]) for audit in audits),
        "raw_positive_rows": sum(int(audit["raw_positive_rows"]) for audit in audits),
        "y_mismatches": sum(
            int(audit["checks"]["y_reconciliation"]["mismatches"]) for audit in audits
        ),
        "lateval_mismatches": sum(
            int(audit["checks"]["lateval_reconciliation"]["mismatches"])
            for audit in audits
        ),
        "early_absence_violations": sum(
            int(audit["checks"]["early_absence"]["violations_above_100_kusd"])
            for audit in audits
        ),
        "negative_class_lateval_violations": sum(
            int(audit["checks"]["negative_class_lateval_zero"]["violations"])
            for audit in audits
        ),
        "all_pass": all(bool(audit["all_pass"]) for audit in audits),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute v2 Track A/B lane labels from raw BACI stage-year sums over "
            "fixed five-year calendar windows."
        )
    )
    parser.add_argument(
        "--chains",
        nargs="+",
        default=list(CHAINS),
        metavar="CHAIN",
        help=f"chains to audit (space- or comma-separated; default: {','.join(CHAINS)})",
    )
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=list(SNAPSHOTS),
        metavar="SNAPSHOT",
        help="snapshots to audit: main, fold2 (default: both)",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=list(TRACKS),
        metavar="TRACK",
        help="lane tracks to audit: A, B (default: both)",
    )
    parser.add_argument(
        "--baci-zip",
        type=Path,
        default=BACI_ZIP,
        help="raw BACI HS92 zip (default: VCU_RAW/BACI_HS92_V202401b.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: results_v2/metrics/raw_label_audit.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="raw CSV rows per read chunk (default: 1000000)",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="lateval absolute tolerance in kUSD")
    parser.add_argument("--rtol", type=float, default=1e-9, help="lateval relative tolerance")
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="verify the saved full audit and current candidate hashes without reading raw BACI",
    )
    return parser


def verify_existing_output(path: Path = DEFAULT_OUTPUT) -> None:
    """Verify that a saved full audit is green and matches current candidate bytes."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "raw-label-audit/v2":
        raise ValueError(f"{path}: unexpected schema_version")
    summary = payload.get("summary", {})
    if summary.get("all_pass") is not True or summary.get("audit_instances") != 24:
        raise ValueError(f"{path}: expected a complete 24/24 passing audit")
    selection = payload.get("selection", {})
    if (selection.get("chains") != list(CHAINS)
            or selection.get("snapshots") != list(SNAPSHOTS)
            or selection.get("tracks") != list(TRACKS)):
        raise ValueError(f"{path}: audit selection is not the full release selection")

    checked = 0
    for audit in payload.get("audits", []):
        candidate = ROOT / str(audit.get("candidate_file", ""))
        expected = audit.get("candidate_sha256")
        if not candidate.is_file() or not expected:
            raise ValueError(f"{path}: missing candidate hash record")
        observed = _sha256(candidate)
        if observed != expected:
            raise ValueError(
                f"{path}: stale candidate hash for {candidate}: "
                f"expected {expected}, observed {observed}"
            )
        if audit.get("all_pass") is not True:
            raise ValueError(f"{path}: a recorded audit instance is not passing")
        checked += 1
    if checked != 24:
        raise ValueError(f"{path}: expected 24 audit records, found {checked}")
    print(f"verified complete raw-label audit and {checked} candidate hashes in {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    output = output.resolve()
    if args.verify_output:
        verify_existing_output(output)
        return 0
    chains = _normalise_choices(parser, args.chains, CHAINS, "chains")
    snapshots = _normalise_choices(parser, args.snapshots, SNAPSHOTS, "snapshots")
    tracks = _normalise_choices(parser, args.tracks, TRACKS, "tracks", uppercase=True)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")

    baci_zip = args.baci_zip.resolve()
    if not baci_zip.is_file():
        parser.error(f"BACI zip not found: {baci_zip}")

    started = time.perf_counter()
    print(
        "raw-label-audit: "
        f"chains={chains} snapshots={snapshots} tracks={tracks} "
        f"threshold={VALUE_THRESHOLD_KUSD:g} kUSD"
    )
    tables, candidate_keys = _load_candidate_tables(chains, snapshots, tracks)
    hs_to_chain, hs_to_stage = _chain_stage_maps(chains)

    requested_periods = {
        _window_label(SNAPSHOTS[snapshot][role])
        for snapshot in snapshots
        for role in ("early", "late")
    }
    years = sorted(
        {
            year
            for snapshot in snapshots
            for role in ("early", "late")
            for year in SNAPSHOTS[snapshot][role]
        }
    )
    with zipfile.ZipFile(baci_zip) as zf:
        iso = _country_iso_map(zf)
        means, raw_stats = _independent_calendar_means(
            zf,
            years,
            iso,
            hs_to_chain,
            hs_to_stage,
            candidate_keys,
            args.chunk_size,
        )
    missing_periods = sorted(requested_periods - set(means))
    if missing_periods:
        raise AssertionError(f"raw aggregation did not produce periods: {missing_periods}")

    audits = [
        _audit_table(
            tables[(chain, snapshot, track)],
            chain,
            snapshot,
            track,
            means,
            args.atol,
            args.rtol,
        )
        for snapshot in snapshots
        for track in tracks
        for chain in chains
    ]
    elapsed = time.perf_counter() - started
    summary = _summary(audits)
    report = {
        "schema_version": "raw-label-audit/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_root": "data/processed_v2",
        "raw_source": {
            "path": _display_path(baci_zip),
            "archive_name": baci_zip.name,
            "size_bytes": baci_zip.stat().st_size,
            "sha256": _sha256(baci_zip),
        },
        "selection": {
            "chains": chains,
            "snapshots": snapshots,
            "tracks": tracks,
        },
        "definition": {
            "unit": "exporter-importer-stage lane",
            "raw_value_unit": "kUSD",
            "aggregation": "sum HS6 within each stage-year, sum five annual values, divide by 5",
            "missing_stage_year_value": 0.0,
            "window_length_years": WINDOW_LENGTH,
            "positive_rule": "raw_late_calendar_mean_kusd > 100",
            "candidate_early_absence_rule": "raw_early_calendar_mean_kusd <= 100",
            "stored_lateval_rule": "raw late calendar mean for positives; zero for negatives",
        },
        "tolerances": {"lateval_atol_kusd": args.atol, "lateval_rtol": args.rtol},
        "raw_read": {
            "chunk_size": args.chunk_size,
            "years": years,
            **raw_stats,
        },
        "elapsed_seconds": round(elapsed, 3),
        "summary": summary,
        "audits": audits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"summary: {summary['passing_instances']}/{summary['audit_instances']} instances PASS, "
        f"rows={summary['candidate_rows']:,}, elapsed={elapsed:.1f}s"
    )
    print(f"saved -> {output}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
