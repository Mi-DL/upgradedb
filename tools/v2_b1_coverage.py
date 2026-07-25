#!/usr/bin/env python3
"""Measure the coverage cost of Track B1's early-demand market screen.

Track B1 deliberately enumerates destination markets that already imported a
processed stage in the early window.  That makes the released task finite and
pre-label, but it also excludes late starts into previously inactive markets.
This module computes that exclusion directly from thresholded stage-level
early/late windows.  The pure :func:`coverage_from_windows` function is kept
separate from raw-BACI I/O so the definition can be unit tested independently
and reused by the private cache/rebuild pipeline.

All input values are five-year calendar means in kUSD after application of the
benchmark's strict ``> 100`` threshold.  Consequently, the input frames should
contain only active stage lanes; they must not be candidate-table subsets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import universe as U  # noqa: E402
from baci_filtered_cache import (  # noqa: E402
    BaciFilteredCache,
    COUNTRY_CODES_MEMBER,
    registry_snapshot,
)


LANE_KEYS = ["i_iso", "j_iso", "stage"]
ENTRY_KEYS = ["i_iso", "stage"]
REQUIRED_COLUMNS = set(LANE_KEYS + ["v"])
SCHEMA_VERSION = "upgrade-bench-v2/b1-candidate-coverage/2"
DEFAULT_THRESHOLD_KUSD = 100.0
DEFAULT_CACHE = os.environ.get("VCU_BACI_CACHE")
DEFAULT_BACI_ZIP = (
    Path(os.environ.get("VCU_RAW", str(ROOT / "data" / "raw")))
    / "BACI_HS92_V202401b.zip"
)
DEFAULT_OUTPUT = ROOT / "results_v2" / "metrics" / "b1_candidate_coverage.json"
DEFAULT_CANDIDATE_ROOT = ROOT / "data" / "processed_v2"
BAD_ISO = {"ANT", "SCG", "YUG", "SUN", "CSK", "DDR"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROTOCOL_FILES = {
    "cache_reader": ROOT / "src" / "baci_filtered_cache.py",
    "candidate_generator": ROOT / "src" / "temporal_backtest.py",
    "calendar_aggregation": ROOT / "src" / "window_aggregation.py",
    "entry_view_builder": ROOT / "tools" / "build_v2_views.py",
    "registry_loader": ROOT / "src" / "universe.py",
}
SNAPSHOTS = {
    "fold2": {
        "early": [1998, 1999, 2000, 2001, 2002],
        "late": [2008, 2009, 2010, 2011, 2012],
        "suffix": "_fold2",
        "temporal_role": "history",
    },
    "main": {
        "early": [2008, 2009, 2010, 2011, 2012],
        "late": [2018, 2019, 2020, 2021, 2022],
        "suffix": "",
        "temporal_role": "target",
    },
}


def _assert_production_protocol_literals() -> None:
    """Fail closed if duplicated Track-B1 constants drift in production code."""

    source_path = PROTOCOL_FILES["candidate_generator"]
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {"_FOLDS", "THRESH", "BAD_ISO"}
    observed: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            observed[target.id] = ast.literal_eval(node.value)
    if set(observed) != wanted:
        raise ValueError("cannot recover exact B1 fold/threshold/ISO constants from production")
    if float(observed["THRESH"]) != DEFAULT_THRESHOLD_KUSD:
        raise ValueError("B1 coverage threshold differs from temporal_backtest.THRESH")
    if set(observed["BAD_ISO"]) != BAD_ISO:
        raise ValueError("B1 coverage BAD_ISO differs from temporal_backtest.BAD_ISO")
    folds = observed["_FOLDS"]
    for snapshot, spec in SNAPSHOTS.items():
        if snapshot not in folds:
            raise ValueError(f"temporal_backtest lacks the {snapshot} B1 snapshot")
        early, late, _ = folds[snapshot]
        if list(early) != spec["early"] or list(late) != spec["late"]:
            raise ValueError(f"B1 coverage years differ from temporal_backtest {snapshot}")


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _validated_window(
    frame: pd.DataFrame, label: str, threshold_kusd: float
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{label} window is missing required columns: {missing}")
    out = frame[LANE_KEYS + ["v"]].copy()
    for column in LANE_KEYS:
        if bool(out[column].isna().any()):
            raise ValueError(f"{label} window has missing {column} values")
        out[column] = out[column].astype(str)
    out["v"] = pd.to_numeric(out["v"], errors="raise")
    if bool((~np.isfinite(out["v"].to_numpy(float))).any()):
        raise ValueError(f"{label} window has non-finite values")
    if bool((out["v"] <= threshold_kusd).any()):
        raise ValueError(
            f"{label} window must contain only lanes strictly above "
            f"{threshold_kusd:g} kUSD"
        )
    if bool(out.duplicated(LANE_KEYS).any()):
        raise ValueError(f"{label} window contains duplicate stage-lane keys")
    return out.sort_values(LANE_KEYS, kind="mergesort").reset_index(drop=True)


def _stage_record_and_candidates(
    early: pd.DataFrame,
    late: pd.DataFrame,
    stage: str,
    upstream_stages: Iterable[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    early_exp = {
        name: set(early.loc[early.stage.eq(name), "i_iso"])
        for name in set(upstream_stages) | {stage}
    }
    upstream_exporters: set[str] = set()
    for upstream in upstream_stages:
        upstream_exporters.update(early_exp.get(upstream, set()))
    entrant_exporters = upstream_exporters - early_exp.get(stage, set())

    eligible_destinations = set(early.loc[early.stage.eq(stage), "j_iso"])
    released_entries = {
        (exporter, stage)
        for exporter in entrant_exporters
        if any(destination != exporter for destination in eligible_destinations)
    }

    late_values = {
        (i_iso, j_iso, lane_stage): float(value)
        for i_iso, j_iso, lane_stage, value in late[LANE_KEYS + ["v"]].itertuples(
            index=False, name=None
        )
    }
    candidate_rows: list[tuple[str, str, str, int, float]] = []
    for exporter in sorted(entrant_exporters):
        for destination in sorted(eligible_destinations):
            if destination == exporter:
                continue
            key = (exporter, destination, stage)
            value = late_values.get(key)
            candidate_rows.append(
                (
                    exporter,
                    destination,
                    stage,
                    int(value is not None),
                    float(value) if value is not None else 0.0,
                )
            )
    candidate_lanes = pd.DataFrame(
        candidate_rows, columns=LANE_KEYS + ["y", "lateval"]
    )

    realized = late.loc[
        late.stage.eq(stage)
        & late.i_iso.isin(entrant_exporters)
        & late.i_iso.ne(late.j_iso)
    ].copy()
    realized["eligible_market"] = realized.j_iso.isin(eligible_destinations)

    all_entries = set(map(tuple, realized[ENTRY_KEYS].to_numpy()))
    covered_entries = set(
        map(tuple, realized.loc[realized.eligible_market, ENTRY_KEYS].to_numpy())
    )
    inactive_entries = set(
        map(tuple, realized.loc[~realized.eligible_market, ENTRY_KEYS].to_numpy())
    )
    inactive_only_entries = inactive_entries - covered_entries
    mixed_entries = inactive_entries & covered_entries

    eligible_lanes = realized.loc[realized.eligible_market]
    inactive_lanes = realized.loc[~realized.eligible_market]
    total_value = float(realized.v.sum())
    eligible_value = float(eligible_lanes.v.sum())
    inactive_value = float(inactive_lanes.v.sum())

    record: dict[str, Any] = {
        "stage": stage,
        "upstream_stages": sorted(set(upstream_stages)),
        "n_upstream_qualified_nonincumbent_exporter_stage_pairs": len(entrant_exporters),
        "n_early_demand_destination_stage_pairs": len(eligible_destinations),
        "n_released_candidate_entries": len(released_entries),
        "n_all_realized_entries": len(all_entries),
        "n_covered_realized_entries": len(covered_entries),
        "n_inactive_only_realized_entries": len(inactive_only_entries),
        "n_mixed_realized_entries": len(mixed_entries),
        "n_all_late_start_lanes": len(realized),
        "n_eligible_market_late_start_lanes": len(eligible_lanes),
        "n_previously_inactive_market_late_start_lanes": len(inactive_lanes),
        "all_late_start_value_kusd": total_value,
        "eligible_market_late_start_value_kusd": eligible_value,
        "previously_inactive_market_late_start_value_kusd": inactive_value,
    }
    record.update(_derived_rates(record))
    return record, candidate_lanes


def _derived_rates(record: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "realized_entry_coverage": _ratio(
            record["n_covered_realized_entries"], record["n_all_realized_entries"]
        ),
        "late_start_lane_coverage": _ratio(
            record["n_eligible_market_late_start_lanes"],
            record["n_all_late_start_lanes"],
        ),
        "late_start_value_coverage": _ratio(
            record["eligible_market_late_start_value_kusd"],
            record["all_late_start_value_kusd"],
        ),
        "previously_inactive_market_lane_share": _ratio(
            record["n_previously_inactive_market_late_start_lanes"],
            record["n_all_late_start_lanes"],
        ),
        "previously_inactive_market_value_share": _ratio(
            record["previously_inactive_market_late_start_value_kusd"],
            record["all_late_start_value_kusd"],
        ),
    }


def _total_record(stages: list[dict[str, Any]]) -> dict[str, Any]:
    additive = [
        "n_upstream_qualified_nonincumbent_exporter_stage_pairs",
        "n_early_demand_destination_stage_pairs",
        "n_released_candidate_entries",
        "n_all_realized_entries",
        "n_covered_realized_entries",
        "n_inactive_only_realized_entries",
        "n_mixed_realized_entries",
        "n_all_late_start_lanes",
        "n_eligible_market_late_start_lanes",
        "n_previously_inactive_market_late_start_lanes",
        "all_late_start_value_kusd",
        "eligible_market_late_start_value_kusd",
        "previously_inactive_market_late_start_value_kusd",
    ]
    total: dict[str, Any] = {key: sum(row[key] for row in stages) for key in additive}
    total.update(_derived_rates(total))
    return total


def _coverage_and_candidate_universe(
    early: pd.DataFrame,
    late: pd.DataFrame,
    upstream_map: Mapping[str, Iterable[str]],
    *,
    threshold_kusd: float = 100.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:

    if not np.isfinite(threshold_kusd) or threshold_kusd < 0:
        raise ValueError("threshold_kusd must be finite and non-negative")
    early_checked = _validated_window(early, "early", float(threshold_kusd))
    late_checked = _validated_window(late, "late", float(threshold_kusd))
    if not upstream_map:
        raise ValueError("upstream_map must contain at least one processed target stage")

    stages: list[dict[str, Any]] = []
    candidate_parts: list[pd.DataFrame] = []
    for stage in sorted(upstream_map):
        upstream = [str(name) for name in upstream_map[stage]]
        if not upstream:
            raise ValueError(f"{stage}: upstream stage list is empty")
        record, candidates = _stage_record_and_candidates(
            early_checked, late_checked, str(stage), upstream
        )
        stages.append(record)
        candidate_parts.append(candidates)

    if candidate_parts:
        candidate_lanes = (
            pd.concat(candidate_parts, ignore_index=True)
            .sort_values(LANE_KEYS, kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        candidate_lanes = pd.DataFrame(columns=LANE_KEYS + ["y", "lateval"])
    if candidate_lanes.empty:
        candidate_entries = pd.DataFrame(
            columns=ENTRY_KEYS
            + [
                "z",
                "entry_lateval",
                "n_candidate_destinations",
                "n_materialized_destinations",
            ]
        )
    else:
        candidate_entries = (
            candidate_lanes.groupby(ENTRY_KEYS, as_index=False)
            .agg(
                z=("y", "max"),
                entry_lateval=("lateval", "sum"),
                n_candidate_destinations=("j_iso", "size"),
                n_materialized_destinations=("y", "sum"),
            )
            .sort_values(ENTRY_KEYS, kind="mergesort")
            .reset_index(drop=True)
        )

    coverage = {
        "definition": (
            "late processed-stage starts by early upstream-qualified nonincumbents; "
            "covered destinations had early processed-stage demand"
        ),
        "activity_threshold_kusd_strictly_greater_than": float(threshold_kusd),
        "stages": stages,
        "totals": _total_record(stages),
    }
    totals = coverage["totals"]
    consistency = {
        "released entries": (
            len(candidate_entries),
            totals["n_released_candidate_entries"],
        ),
        "covered entries": (
            int(candidate_entries.z.sum()),
            totals["n_covered_realized_entries"],
        ),
        "covered lanes": (
            int(candidate_lanes.y.sum()),
            totals["n_eligible_market_late_start_lanes"],
        ),
    }
    for label, (observed, expected) in consistency.items():
        if observed != expected:
            raise AssertionError(
                f"internal B1 {label} inconsistency: observed={observed}, expected={expected}"
            )
    if not np.isclose(
        float(candidate_lanes.lateval.sum()),
        float(totals["eligible_market_late_start_value_kusd"]),
        rtol=1e-12,
        atol=1e-9,
    ):
        raise AssertionError("internal B1 eligible-market value inconsistency")
    return coverage, candidate_lanes, candidate_entries


def coverage_from_windows(
    early: pd.DataFrame,
    late: pd.DataFrame,
    upstream_map: Mapping[str, Iterable[str]],
    *,
    threshold_kusd: float = 100.0,
) -> dict[str, Any]:
    """Return stage-level and overall Track-B1 coverage diagnostics.

    ``upstream_map`` is the registry contract mapping every processed target
    stage to the early stages that qualify an exporter for Track B1.  A
    realized entry is covered when at least one of its late destination lanes
    belongs to that stage's early-demand destination set.  Starts into markets
    absent from the early-demand set are measured, never silently discarded.
    """

    coverage, _, _ = _coverage_and_candidate_universe(
        early, late, upstream_map, threshold_kusd=threshold_kusd
    )
    return coverage


def candidate_universe_from_windows(
    early: pd.DataFrame,
    late: pd.DataFrame,
    upstream_map: Mapping[str, Iterable[str]],
    *,
    threshold_kusd: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the exact B1 lane universe and its ``build_v2_views`` entry view."""

    _, lanes, entries = _coverage_and_candidate_universe(
        early, late, upstream_map, threshold_kusd=threshold_kusd
    )
    return lanes, entries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_with_sha256(
    path: Path, *, dtype: Mapping[str, type]
) -> tuple[pd.DataFrame, str]:
    """Parse and hash the same immutable byte snapshot of a candidate CSV."""

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=dtype)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot parse B1 candidate input {path.name}: {exc}") from exc
    return frame, digest


def _repo_logical_path(path: Path, *, label: str, directory: bool = False) -> str:
    """Return a non-hidden repo-relative path without following an escaped symlink."""

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if directory and not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    if not directory and not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    if not relative.parts or any(part.startswith(".") for part in relative.parts):
        raise ValueError(f"{label} must be a non-hidden path below the repository root")
    return relative.as_posix()


def _path_from_report(logical: object, *, label: str, directory: bool = False) -> Path:
    """Resolve a report path while rejecting absolute paths and traversal."""

    if not isinstance(logical, str) or not logical or "\\" in logical:
        raise ValueError(f"{label} is not a canonical repository-relative path")
    pure = PurePosixPath(logical)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} or part.startswith(".") for part in pure.parts)
    ):
        raise ValueError(f"{label} is not a safe repository-relative path")
    candidate = ROOT.joinpath(*pure.parts)
    canonical = _repo_logical_path(candidate, label=label, directory=directory)
    if canonical != pure.as_posix():
        raise ValueError(f"{label} is not canonical or crosses a symlink")
    return candidate


def _country_iso_map(cache: BaciFilteredCache, baci_zip: Path) -> dict[int, str]:
    with zipfile.ZipFile(baci_zip) as source_zip:
        payload = cache.country_codes_bytes(source_zip, archive_path=baci_zip)
    countries = pd.read_csv(io.BytesIO(payload))
    countries = countries.dropna(subset=["country_code", "country_iso3"])
    return dict(
        zip(
            pd.to_numeric(countries.country_code, errors="raise").astype(int),
            countries.country_iso3.astype(str),
        )
    )


def stage_window_from_cache(
    cache: BaciFilteredCache,
    iso: Mapping[int, str],
    chain_id: str,
    years: Iterable[int],
    *,
    threshold_kusd: float = DEFAULT_THRESHOLD_KUSD,
) -> pd.DataFrame:
    """Independently aggregate a chain's complete raw cache rows to a window.

    This intentionally does not call the production window aggregation helper.
    Each stage lane is summed within each year, then across the fixed window and
    divided by every calendar year; absent years therefore contribute zero.
    """

    years = tuple(int(year) for year in years)
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be a non-empty sequence without duplicates")
    chain = U.get_chain(chain_id)
    hs6 = set(chain.all_hs)
    annual_parts: list[pd.DataFrame] = []
    for year in years:
        frame = cache.read_year(year)
        frame["k"] = frame.k.astype(str).str.zfill(6)
        frame = frame.loc[frame.k.isin(hs6), ["i", "j", "k", "v"]].copy()
        if frame.empty:
            continue
        frame["i_iso"] = frame.i.map(iso)
        frame["j_iso"] = frame.j.map(iso)
        frame["stage"] = frame.k.map(chain.hs2stage)
        frame["v"] = pd.to_numeric(frame.v, errors="raise")
        frame = frame.dropna(subset=LANE_KEYS + ["v"])
        frame = frame.loc[
            ~frame.i_iso.isin(BAD_ISO) & ~frame.j_iso.isin(BAD_ISO)
        ]
        annual_parts.append(
            frame.groupby(LANE_KEYS, as_index=False, sort=False)["v"].sum()
        )
    if not annual_parts:
        return pd.DataFrame(columns=LANE_KEYS + ["v"])
    window = (
        pd.concat(annual_parts, ignore_index=True)
        .groupby(LANE_KEYS, as_index=False, sort=False)["v"]
        .sum()
    )
    window["v"] = window.v / float(len(years))
    window = window.loc[window.v > float(threshold_kusd)]
    return window.sort_values(LANE_KEYS, kind="mergesort").reset_index(drop=True)


def _candidate_reconciliation(
    candidate_root: Path,
    chain: str,
    suffix: str,
    totals: Mapping[str, Any],
    expected_lanes: pd.DataFrame,
    expected_entries: pd.DataFrame,
    *,
    early_years: Iterable[int],
    late_years: Iterable[int],
    temporal_role: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    lane_path = candidate_root / f"candidates_firsttime_{chain}{suffix}.csv"
    entry_path = candidate_root / f"entries_firsttime_{chain}{suffix}.csv"
    for path in (lane_path, entry_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    lane_required = set(
        LANE_KEYS
        + [
            "y",
            "lateval",
            "entry_id",
            "entry_y",
            "aggregation",
            "early_window",
            "late_window",
            "temporal_role",
            "task",
            "task_unit",
        ]
    )
    entry_required = set(
        ENTRY_KEYS
        + [
            "z",
            "entry_id",
            "entry_lateval",
            "n_candidate_destinations",
            "n_materialized_destinations",
            "aggregation",
            "early_window",
            "late_window",
            "temporal_role",
            "task",
            "task_unit",
        ]
    )
    lanes, lane_sha256 = _read_csv_with_sha256(
        lane_path, dtype={"i_iso": str, "j_iso": str, "stage": str}
    )
    entries, entry_sha256 = _read_csv_with_sha256(
        entry_path, dtype={"i_iso": str, "stage": str}
    )
    for frame, path, required, keys in (
        (lanes, lane_path, lane_required, LANE_KEYS),
        (entries, entry_path, entry_required, ENTRY_KEYS),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if bool(frame[keys].isna().any().any()):
            raise ValueError(f"{path} has missing candidate identity values")
        if bool(frame.duplicated(keys).any()):
            raise ValueError(f"{path} has duplicate candidate keys")

    expected_early = f"{min(early_years)}-{max(early_years)}"
    expected_late = f"{min(late_years)}-{max(late_years)}"
    lane_constants = {
        "aggregation": "calendar_mean",
        "early_window": expected_early,
        "late_window": expected_late,
        "temporal_role": temporal_role,
        "task": "processed_export_entry_candidate_lane",
        "task_unit": "exporter_stage_destination",
    }
    entry_constants = {
        "aggregation": "calendar_mean",
        "early_window": expected_early,
        "late_window": expected_late,
        "temporal_role": temporal_role,
        "task": "processed_export_stage_entry",
        "task_unit": "exporter_stage",
    }
    for frame, path, expected_constants in (
        (lanes, lane_path, lane_constants),
        (entries, entry_path, entry_constants),
    ):
        for column, expected in expected_constants.items():
            if not frame[column].astype(str).eq(expected).all():
                raise ValueError(f"{path} has noncanonical {column}; expected {expected!r}")

    for frame, path, binary_columns in (
        (lanes, lane_path, ("y", "entry_y")),
        (entries, entry_path, ("z",)),
    ):
        for column in binary_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not frame[column].isin([0, 1]).all():
                raise ValueError(f"{path} has non-binary {column} labels")

    for frame, path, value_columns in (
        (lanes, lane_path, ("lateval",)),
        (entries, entry_path, ("entry_lateval",)),
    ):
        for column in value_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            values = frame[column].to_numpy(float)
            if bool((~np.isfinite(values)).any()) or bool((values < 0).any()):
                raise ValueError(f"{path} has invalid {column} values")
    for column in ("n_candidate_destinations", "n_materialized_destinations"):
        entries[column] = pd.to_numeric(entries[column], errors="raise")
        values = entries[column].to_numpy(float)
        if bool((~np.isfinite(values)).any()) or bool((values < 0).any()) or bool(
            (values % 1 != 0).any()
        ):
            raise ValueError(f"{entry_path} has invalid {column} counts")

    lanes = lanes.sort_values(LANE_KEYS, kind="mergesort").reset_index(drop=True)
    entries = entries.sort_values(ENTRY_KEYS, kind="mergesort").reset_index(drop=True)
    expected_lanes = expected_lanes.sort_values(
        LANE_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    expected_entries = expected_entries.sort_values(
        ENTRY_KEYS, kind="mergesort"
    ).reset_index(drop=True)

    def assert_same_keys(
        observed: pd.DataFrame,
        expected: pd.DataFrame,
        keys: list[str],
        label: str,
    ) -> None:
        observed_keys = set(map(tuple, observed[keys].to_numpy()))
        expected_keys = set(map(tuple, expected[keys].to_numpy()))
        if observed_keys != expected_keys:
            raise ValueError(
                f"{chain}{suffix}: B1 {label} identities differ from raw enumeration: "
                f"missing={len(expected_keys - observed_keys)}, "
                f"extra={len(observed_keys - expected_keys)}"
            )

    assert_same_keys(lanes, expected_lanes, LANE_KEYS, "lane")
    assert_same_keys(entries, expected_entries, ENTRY_KEYS, "entry")

    expected_entry_labels = expected_entries[ENTRY_KEYS + ["z"]]
    expected_lanes = expected_lanes.merge(
        expected_entry_labels.rename(columns={"z": "entry_y"}),
        on=ENTRY_KEYS,
        how="left",
        validate="many_to_one",
    )
    expected_lanes = expected_lanes.sort_values(
        LANE_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    for frame, expected, path, columns in (
        (lanes, expected_lanes, lane_path, ("y", "entry_y")),
        (entries, expected_entries, entry_path, ("z",)),
    ):
        for column in columns:
            if not np.array_equal(
                frame[column].to_numpy(int), expected[column].to_numpy(int)
            ):
                raise ValueError(f"{path} has labels that differ from raw enumeration")

    expected_lane_ids = (
        lanes.i_iso.astype(str) + "|" + lanes.stage.astype(str)
    )
    expected_entry_ids = (
        entries.i_iso.astype(str) + "|" + entries.stage.astype(str)
    )
    if not lanes.entry_id.astype(str).eq(expected_lane_ids).all():
        raise ValueError(f"{lane_path} has stale entry_id values")
    if not entries.entry_id.astype(str).eq(expected_entry_ids).all():
        raise ValueError(f"{entry_path} has stale entry_id values")

    numeric_pairs = (
        (lanes, expected_lanes, lane_path, "lateval", False),
        (entries, expected_entries, entry_path, "entry_lateval", False),
        (
            entries,
            expected_entries,
            entry_path,
            "n_candidate_destinations",
            True,
        ),
        (
            entries,
            expected_entries,
            entry_path,
            "n_materialized_destinations",
            True,
        ),
    )
    for observed, expected, path, column, exact in numeric_pairs:
        observed_values = observed[column].to_numpy(float)
        expected_values = expected[column].to_numpy(float)
        equal = (
            np.array_equal(observed_values, expected_values)
            if exact
            else np.allclose(
                observed_values,
                expected_values,
                rtol=1e-12,
                atol=1e-9,
                equal_nan=False,
            )
        )
        if not equal:
            raise ValueError(f"{path} has {column} values that differ from raw enumeration")

    observed_candidates = int(len(entries))
    observed_positives = int(entries.z.sum())
    expected_candidates = int(totals["n_released_candidate_entries"])
    expected_positives = int(totals["n_covered_realized_entries"])
    if observed_candidates != expected_candidates or observed_positives != expected_positives:
        raise AssertionError(f"{chain}{suffix}: exact B1 reconciliation totals are inconsistent")
    result = {
        "pass": True,
        "exact_lane_identity_label_value_reconciliation": True,
        "exact_entry_identity_label_value_reconciliation": True,
        "candidate_lanes": int(len(lanes)),
        "raw_expected_candidate_lanes": int(len(expected_lanes)),
        "positive_lanes": int(lanes.y.sum()),
        "raw_expected_positive_lanes": int(expected_lanes.y.sum()),
        "candidate_entries": observed_candidates,
        "raw_expected_candidate_entries": expected_candidates,
        "positive_entries": observed_positives,
        "raw_covered_realized_entries": expected_positives,
        "eligible_market_late_start_value_kusd": float(entries.entry_lateval.sum()),
        "raw_expected_eligible_market_late_start_value_kusd": float(
            expected_entries.entry_lateval.sum()
        ),
    }
    hashes: dict[str, str] = {}
    for path, digest in (
        (lane_path, lane_sha256),
        (entry_path, entry_sha256),
    ):
        logical = _repo_logical_path(path, label="B1 candidate input")
        hashes[logical] = digest
    return result, hashes


def _build_report(
    cache_dir: Path,
    baci_zip: Path,
    candidate_root: Path,
    *,
    threshold_kusd: float = DEFAULT_THRESHOLD_KUSD,
) -> dict[str, Any]:
    _assert_production_protocol_literals()
    if float(threshold_kusd) != DEFAULT_THRESHOLD_KUSD:
        raise ValueError("formal B1 coverage must use temporal_backtest.THRESH exactly")
    candidate_root_logical = _repo_logical_path(
        candidate_root, label="B1 candidate root", directory=True
    )
    cache = BaciFilteredCache(
        cache_dir,
        requested_years=sorted(
            {year for spec in SNAPSHOTS.values() for key in ("early", "late") for year in spec[key]}
        ),
        chains_dir=U.CHAINS_DIR,
    )
    cache_manifest_path = cache.cache_dir / "manifest.json"
    cache_manifest_sha256 = _sha256_file(cache_manifest_path)
    iso = _country_iso_map(cache, baci_zip)
    snapshots: dict[str, Any] = {}
    candidate_hashes: dict[str, str] = {}
    for snapshot, spec in SNAPSHOTS.items():
        chains: list[dict[str, Any]] = []
        for chain_id in sorted(U.CHAINS):
            if chain_id == "all":
                continue
            chain = U.get_chain(chain_id)
            early = stage_window_from_cache(
                cache, iso, chain_id, spec["early"], threshold_kusd=threshold_kusd
            )
            late = stage_window_from_cache(
                cache, iso, chain_id, spec["late"], threshold_kusd=threshold_kusd
            )
            coverage, expected_lanes, expected_entries = _coverage_and_candidate_universe(
                early, late, chain.upstream_map, threshold_kusd=threshold_kusd
            )
            reconciliation, hashes = _candidate_reconciliation(
                candidate_root,
                chain_id,
                str(spec["suffix"]),
                coverage["totals"],
                expected_lanes,
                expected_entries,
                early_years=spec["early"],
                late_years=spec["late"],
                temporal_role=str(spec["temporal_role"]),
            )
            candidate_hashes.update(hashes)
            chains.append(
                {
                    "chain": chain_id,
                    "candidate_reconciliation": reconciliation,
                    **coverage,
                }
            )
        snapshots[snapshot] = {
            "temporal_role": spec["temporal_role"],
            "early_years": spec["early"],
            "late_years": spec["late"],
            "chains": chains,
            "totals": _total_record([row["totals"] for row in chains]),
        }

    if _sha256_file(cache_manifest_path) != cache_manifest_sha256:
        raise ValueError("private BACI cache manifest changed during B1 coverage build")
    for logical, expected in candidate_hashes.items():
        candidate = _path_from_report(logical, label="candidate input")
        if _sha256_file(candidate) != expected:
            raise ValueError(
                f"candidate input changed during B1 coverage build: {logical}"
            )

    manifest = cache.manifest
    registry = manifest["registry"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "definition": {
            "activity": f"five-year calendar mean strictly greater than {threshold_kusd:g} kUSD",
            "denominator": (
                "all late processed-stage lanes and exporter-stage entries realized by "
                "early upstream-qualified nonincumbent exporters"
            ),
            "covered": (
                "late destination imported the same processed stage from any exporter "
                "in the early window"
            ),
            "excluded_market": "late destination had no early processed-stage demand",
        },
        "source": {
            "dataset": manifest["source"]["dataset"],
            "archive_name": manifest["source"]["archive_name"],
            "archive_bytes": manifest["source"]["archive_bytes"],
            "archive_sha256": manifest["source"]["archive_sha256"],
            "cache_schema": manifest["schema_version"],
            "cache_manifest_sha256": cache_manifest_sha256,
        },
        "candidate_input_root": candidate_root_logical,
        "registry": {
            "audit": registry["audit"],
            "evidence": registry["evidence"],
            "chain_count": registry["chain_count"],
            "active_hs6_count": len(registry["active_hs6_union"]),
        },
        "input_sha256": dict(sorted(candidate_hashes.items())),
        "generator": {
            "path": "tools/v2_b1_coverage.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "protocol_sha256": {
            label: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(path),
            }
            for label, path in sorted(PROTOCOL_FILES.items())
        },
        "snapshots": snapshots,
    }
    return report


def _verify_report(path: Path, baci_cache: Path | None = None) -> None:
    _assert_production_protocol_literals()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "PASS":
        raise ValueError(f"{path}: unsupported schema or non-PASS status")
    generator = payload.get("generator", {})
    if (
        not isinstance(generator, dict)
        or generator.get("path") != "tools/v2_b1_coverage.py"
        or generator.get("sha256") != _sha256_file(Path(__file__).resolve())
    ):
        raise ValueError(f"{path}: stale generator hash")

    protocol = payload.get("protocol_sha256")
    if not isinstance(protocol, dict) or set(protocol) != set(PROTOCOL_FILES):
        raise ValueError(f"{path}: incomplete protocol hashes")
    for label, current_path in PROTOCOL_FILES.items():
        expected_record = {
            "path": current_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(current_path),
        }
        if protocol.get(label) != expected_record:
            raise ValueError(f"{path}: stale {label} protocol hash")

    report_registry = payload.get("registry", {})
    if not isinstance(report_registry, dict):
        raise ValueError(f"{path}: registry provenance must be an object")
    current_registry = registry_snapshot(chains_dir=U.CHAINS_DIR)
    expected_registry = {
        "audit": current_registry["audit"],
        "evidence": current_registry["evidence"],
        "chain_count": current_registry["chain_count"],
        "active_hs6_count": len(current_registry["active_hs6_union"]),
    }
    if report_registry != expected_registry:
        raise ValueError(f"{path}: stale registry/evidence/chain snapshot")

    source = payload.get("source", {})
    if not isinstance(source, dict):
        raise ValueError(f"{path}: source provenance must be an object")
    archive_name = source.get("archive_name")
    if (
        not isinstance(archive_name, str)
        or not archive_name
        or "\\" in archive_name
        or PurePosixPath(archive_name).name != archive_name
        or not isinstance(source.get("archive_bytes"), int)
        or isinstance(source.get("archive_bytes"), bool)
        or source["archive_bytes"] <= 0
    ):
        raise ValueError(f"{path}: unsafe or malformed source archive metadata")
    for field in ("archive_sha256", "cache_manifest_sha256"):
        if not _SHA256_RE.fullmatch(str(source.get(field, ""))):
            raise ValueError(f"{path}: missing or malformed source {field}")
    if baci_cache is not None:
        validated_cache = BaciFilteredCache(
            baci_cache,
            requested_years=[],
            chains_dir=U.CHAINS_DIR,
        )
        manifest_path = validated_cache.cache_dir / "manifest.json"
        if source.get("cache_manifest_sha256") != _sha256_file(manifest_path):
            raise ValueError(f"{path}: stale private cache manifest hash")
        manifest_source = validated_cache.manifest["source"]
        for field in ("dataset", "archive_name", "archive_bytes", "archive_sha256"):
            if source.get(field) != manifest_source.get(field):
                raise ValueError(f"{path}: stale private cache source field {field}")

    expected_definition = {
        "activity": "five-year calendar mean strictly greater than 100 kUSD",
        "denominator": (
            "all late processed-stage lanes and exporter-stage entries realized by "
            "early upstream-qualified nonincumbent exporters"
        ),
        "covered": (
            "late destination imported the same processed stage from any exporter "
            "in the early window"
        ),
        "excluded_market": "late destination had no early processed-stage demand",
    }
    if payload.get("definition") != expected_definition:
        raise ValueError(f"{path}: stale B1 coverage definition")

    candidate_root_logical = payload.get("candidate_input_root")
    candidate_root = _path_from_report(
        candidate_root_logical, label="candidate_input_root", directory=True
    )
    hashes = payload.get("input_sha256")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or not all(isinstance(key, str) for key in hashes)
    ):
        raise ValueError(f"{path}: missing candidate input hashes")
    chain_ids = sorted(chain_id for chain_id in U.CHAINS if chain_id != "all")
    if len(chain_ids) != 6:
        raise ValueError(f"{path}: current registry does not contain the exact six chains")
    root_pure = PurePosixPath(str(candidate_root_logical))
    expected_logicals = {
        (root_pure / f"{stem}_{chain_id}{spec['suffix']}.csv").as_posix()
        for spec in SNAPSHOTS.values()
        for chain_id in chain_ids
        for stem in ("candidates_firsttime", "entries_firsttime")
    }
    if set(hashes) != expected_logicals:
        raise ValueError(
            f"{path}: candidate hash inventory is not exact: "
            f"missing={sorted(expected_logicals - set(hashes))}, "
            f"extra={sorted(set(hashes) - expected_logicals)}"
        )
    for logical, expected in sorted(hashes.items()):
        if not _SHA256_RE.fullmatch(str(expected)):
            raise ValueError(f"{path}: malformed candidate SHA-256 for {logical}")
        candidate = _path_from_report(logical, label="candidate input")
        if candidate.parent.resolve() != candidate_root.resolve():
            raise ValueError(f"{path}: candidate input escaped its declared root")
        if _sha256_file(candidate) != expected:
            raise ValueError(f"{path}: stale or missing candidate input {logical}")

    snapshots = payload.get("snapshots", {})
    if not isinstance(snapshots, dict) or set(snapshots) != set(SNAPSHOTS):
        raise ValueError(f"{path}: snapshot inventory is not exact")
    for snapshot, spec in SNAPSHOTS.items():
        snap = snapshots.get(snapshot, {})
        if (
            snap.get("temporal_role") != spec["temporal_role"]
            or snap.get("early_years") != spec["early"]
            or snap.get("late_years") != spec["late"]
        ):
            raise ValueError(f"{path}: stale {snapshot} temporal protocol")
        chains = snap.get("chains", [])
        if (
            not isinstance(chains, list)
            or len(chains) != 6
            or not all(isinstance(row, dict) for row in chains)
            or {row.get("chain") for row in chains} != set(chain_ids)
        ):
            raise ValueError(f"{path}: incomplete {snapshot} reconciliation")
        for row in chains:
            stages = row.get("stages")
            totals = row.get("totals")
            reconciliation = row.get("candidate_reconciliation", {})
            if (
                not isinstance(stages, list)
                or not stages
                or row.get("activity_threshold_kusd_strictly_greater_than")
                != DEFAULT_THRESHOLD_KUSD
                or totals != _total_record(stages)
                or not isinstance(reconciliation, dict)
                or reconciliation.get("pass") is not True
                or reconciliation.get(
                    "exact_lane_identity_label_value_reconciliation"
                )
                is not True
                or reconciliation.get(
                    "exact_entry_identity_label_value_reconciliation"
                )
                is not True
            ):
                raise ValueError(f"{path}: invalid exact reconciliation for {row.get('chain')}")
            for stage_record in stages:
                if not isinstance(stage_record, dict) or any(
                    stage_record.get(key) != value
                    for key, value in _derived_rates(stage_record).items()
                ):
                    raise ValueError(
                        f"{path}: stale stage-level rates for {row.get('chain')}"
                    )
            expected_values = {
                "candidate_entries": totals["n_released_candidate_entries"],
                "raw_expected_candidate_entries": totals[
                    "n_released_candidate_entries"
                ],
                "positive_entries": totals["n_covered_realized_entries"],
                "raw_covered_realized_entries": totals[
                    "n_covered_realized_entries"
                ],
                "positive_lanes": totals["n_eligible_market_late_start_lanes"],
                "raw_expected_positive_lanes": totals[
                    "n_eligible_market_late_start_lanes"
                ],
            }
            if any(reconciliation.get(key) != value for key, value in expected_values.items()):
                raise ValueError(f"{path}: stale reconciliation counts for {row.get('chain')}")
            if reconciliation.get("candidate_lanes") != reconciliation.get(
                "raw_expected_candidate_lanes"
            ):
                raise ValueError(
                    f"{path}: stale candidate-lane reconciliation for {row.get('chain')}"
                )
            for key in (
                "eligible_market_late_start_value_kusd",
                "raw_expected_eligible_market_late_start_value_kusd",
            ):
                if not np.isclose(
                    float(reconciliation.get(key, np.nan)),
                    float(totals["eligible_market_late_start_value_kusd"]),
                    rtol=1e-12,
                    atol=1e-9,
                ):
                    raise ValueError(f"{path}: stale reconciliation value for {row.get('chain')}")
        expected_snapshot_totals = _total_record([row["totals"] for row in chains])
        if snap.get("totals") != expected_snapshot_totals:
            raise ValueError(f"{path}: stale {snapshot} aggregate totals")
    print(f"verified B1 candidate-universe coverage report: {path}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baci-cache",
        type=Path,
        default=Path(DEFAULT_CACHE) if DEFAULT_CACHE else None,
        help="strict private filtered BACI cache (or VCU_BACI_CACHE)",
    )
    parser.add_argument("--baci-zip", type=Path, default=DEFAULT_BACI_ZIP)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if args.verify_output:
        _verify_report(
            output,
            args.baci_cache.resolve() if args.baci_cache is not None else None,
        )
        return 0
    if args.baci_cache is None:
        parser.error("--baci-cache (or VCU_BACI_CACHE) is required for the raw coverage build")
    report = _build_report(
        args.baci_cache.resolve(),
        args.baci_zip.resolve(),
        args.candidate_root.resolve(),
    )
    _write_json_atomic(output, report)
    print(f"wrote B1 candidate-universe coverage report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
