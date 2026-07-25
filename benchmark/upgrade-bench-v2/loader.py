#!/usr/bin/env python3
"""Dependency-light, strict loader for UPGRADE-BENCH v2 tables.

The package intentionally does not vendor the large CSV payloads.  Point
``UPGRADE_BENCH_V2_DATA`` at a directory containing the extracted v2 tables,
pass ``data_root=...``, or extract the release archives at a repository root.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
VERSION = "2.1-dev"
CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
TRACKS = ("A", "B1", "B2")
SNAPSHOTS = ("main", "fold2")

LANE_KEYS = ("i_iso", "j_iso", "stage")
ENTRY_KEYS = ("i_iso", "stage")
FORBIDDEN_SCORE_COLUMNS = frozenset(
    {"y", "z", "entry_y", "lateval", "entry_lateval", "n_materialized_destinations"}
)
BUILTIN_SCORE_COLUMNS = {
    "A": ("size", "log_exporter_capacity", "log_importer_demand", "grav", "gnn"),
    "B1": ("size", "log_upstream_capacity", "n_candidate_destinations"),
    "B2": ("size", "log_exporter_capacity", "log_importer_demand", "grav", "gnn"),
}

LANE_COLUMNS = frozenset(
    {
        "i_iso", "j_iso", "stage", "y", "size", "log_exporter_capacity",
        "log_importer_demand", "size_basis", "grav", "gnn", "lateval",
        "benchmark_version", "aggregation", "early_window", "late_window",
        "group_id", "transductive_split_unit", "transductive_split",
        "temporal_role", "task", "task_unit",
    }
)
ENTRY_COLUMNS = frozenset(
    {
        "i_iso", "stage", "z", "size", "log_upstream_capacity", "entry_lateval",
        "n_candidate_destinations", "n_materialized_destinations", "benchmark_version",
        "aggregation", "early_window", "late_window", "entry_id",
        "transductive_split_unit", "transductive_split", "temporal_role", "task",
        "task_unit",
    }
)
TRACK_B_LANE_COLUMNS = LANE_COLUMNS | {"entry_id", "entry_y"}
EXPECTED_COLUMNS = {"A": LANE_COLUMNS, "B1": ENTRY_COLUMNS, "B2": TRACK_B_LANE_COLUMNS}

SNAPSHOT_METADATA = {
    "main": {"early_window": "2008-2012", "late_window": "2018-2022", "temporal_role": "target"},
    "fold2": {"early_window": "1998-2002", "late_window": "2008-2012", "temporal_role": "history"},
}


class BenchmarkDataError(ValueError):
    """A v2 payload is missing or violates a frozen benchmark invariant."""


def normalize_track(track: str) -> str:
    value = str(track).strip().upper().replace("-", "")
    aliases = {"A": "A", "TRACKA": "A", "B1": "B1", "TRACKB1": "B1", "B2": "B2", "TRACKB2": "B2"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise BenchmarkDataError(f"track must be one of {TRACKS}; got {track!r}") from exc


def normalize_snapshot(snapshot: str) -> str:
    value = str(snapshot).strip().lower()
    aliases = {"main": "main", "target": "main", "fold2": "fold2", "history": "fold2", "historical": "fold2"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise BenchmarkDataError(f"snapshot must be one of {SNAPSHOTS}; got {snapshot!r}") from exc


def keys_for_track(track: str) -> tuple[str, ...]:
    return ENTRY_KEYS if normalize_track(track) == "B1" else LANE_KEYS


def filename_for(track: str, chain: str, snapshot: str = "main") -> str:
    track = normalize_track(track)
    snapshot = normalize_snapshot(snapshot)
    if chain not in CHAINS:
        raise BenchmarkDataError(f"chain must be one of {CHAINS}; got {chain!r}")
    stem = {
        "A": "candidates",
        "B1": "entries_firsttime",
        "B2": "destinations_given_entry",
    }[track]
    suffix = "_fold2" if snapshot == "fold2" else ""
    return f"{stem}_{chain}{suffix}.csv"


def _candidate_roots() -> list[Path]:
    repository_root = HERE.parents[1]
    cwd = Path.cwd()
    return [
        repository_root / "data" / "processed_v2",
        HERE / "data" / "processed_v2",
        HERE / "data",
        cwd / "data" / "processed_v2",
        cwd / "processed_v2",
        cwd / "data",
        cwd,
    ]


def _normalize_payload_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    nested = path / "data" / "processed_v2"
    if nested.is_dir():
        return nested
    nested = path / "processed_v2"
    if nested.is_dir():
        return nested
    return path


def resolve_data_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an extracted v2 payload directory without requiring repository code."""
    if data_root is not None:
        root = _normalize_payload_root(Path(data_root))
        if not root.is_dir():
            raise BenchmarkDataError(f"v2 data root is not a directory: {root}")
        return root

    configured = os.environ.get("UPGRADE_BENCH_V2_DATA")
    if configured:
        root = _normalize_payload_root(Path(configured))
        if not root.is_dir():
            raise BenchmarkDataError(
                f"UPGRADE_BENCH_V2_DATA does not name a directory: {root}"
            )
        return root

    for candidate in _candidate_roots():
        root = _normalize_payload_root(candidate)
        if root.is_dir() and any(root.glob("candidates_*.csv")):
            return root
    searched = ", ".join(str(path) for path in _candidate_roots())
    raise BenchmarkDataError(
        "cannot locate UPGRADE-BENCH v2 CSVs; set UPGRADE_BENCH_V2_DATA or pass "
        f"data_root. Searched: {searched}"
    )


def _fail(path: Path | str, message: str) -> None:
    raise BenchmarkDataError(f"{path}: {message}")


def _require_exact_columns(frame: pd.DataFrame, expected: frozenset[str], path: Path | str) -> None:
    actual = set(frame.columns)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        _fail(path, f"schema mismatch; missing={missing}, unexpected={unexpected}")


def _require_identity(frame: pd.DataFrame, columns: Iterable[str], path: Path | str) -> None:
    for column in columns:
        values = frame[column]
        bad = values.isna() | values.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        if bad.any():
            _fail(path, f"{column} has {int(bad.sum()):,} null/empty identities")


def _require_constant(frame: pd.DataFrame, column: str, expected: str, path: Path | str) -> None:
    values = set(frame[column].dropna().astype(str))
    if values != {expected} or frame[column].isna().any():
        _fail(path, f"{column} must be exactly {expected!r}; found {sorted(values)!r}")


def _numeric(
    frame: pd.DataFrame,
    column: str,
    path: Path | str,
    *,
    allow_na: bool = False,
    nonnegative: bool = True,
) -> np.ndarray:
    source = frame[column]
    converted = pd.to_numeric(source, errors="coerce")
    if (source.notna() & converted.isna()).any():
        _fail(path, f"{column} contains non-numeric values")
    values = converted.to_numpy(dtype=float)
    checked = values[~np.isnan(values)] if allow_na else values
    if (not allow_na and np.isnan(values).any()) or not np.isfinite(checked).all():
        _fail(path, f"{column} must be finite" + (" where present" if allow_na else ""))
    if nonnegative and (checked < 0).any():
        _fail(path, f"{column} must be non-negative")
    return values


def _binary(frame: pd.DataFrame, column: str, path: Path | str) -> np.ndarray:
    values = _numeric(frame, column, path, nonnegative=True)
    if not np.isin(values, [0.0, 1.0]).all():
        _fail(path, f"{column} must contain only 0/1 labels")
    return values.astype(np.int8)


def _official_split(chain: str, exporter: str, stage: str) -> str:
    key = f"{chain}|exporter_stage|{exporter}|{stage}"
    digest = hashlib.sha256(f"0|{key}".encode("utf-8")).digest()
    return "test" if int.from_bytes(digest[:8], "big") / 2.0**64 < 0.5 else "train"


def _validate_common(
    frame: pd.DataFrame,
    track: str,
    chain: str,
    snapshot: str,
    path: Path | str,
) -> None:
    if frame.empty:
        _fail(path, "table is empty")
    keys = keys_for_track(track)
    _require_identity(frame, keys, path)
    duplicates = int(frame.duplicated(list(keys)).sum())
    if duplicates:
        _fail(path, f"{duplicates:,} duplicate rows for key {list(keys)}")

    metadata = SNAPSHOT_METADATA[snapshot]
    for column, expected in {
        "benchmark_version": VERSION,
        "aggregation": "calendar_mean",
        "transductive_split_unit": "exporter_stage",
        **metadata,
    }.items():
        _require_constant(frame, column, expected, path)

    split = frame["transductive_split"].astype(str)
    if not split.isin(["train", "test"]).all() or frame["transductive_split"].isna().any():
        _fail(path, "transductive_split must contain only complete train/test assignments")
    expected = np.fromiter(
        (_official_split(chain, str(i), str(s)) for i, s in zip(frame["i_iso"], frame["stage"])),
        dtype="U5",
        count=len(frame),
    )
    if not np.array_equal(split.to_numpy(dtype=str), expected):
        _fail(path, "transductive_split does not match the official SHA-256 exporter-stage assignment")


def _validate_outcomes(
    frame: pd.DataFrame, label: str, value: str, path: Path | str
) -> tuple[np.ndarray, np.ndarray]:
    y = _binary(frame, label, path)
    late_value = _numeric(frame, value, path)
    if (late_value[y == 0] != 0.0).any():
        _fail(path, f"{value} must be zero when {label}=0")
    if (late_value[y == 1] <= 100.0).any():
        _fail(path, f"{value} must exceed 100 kUSD when {label}=1")
    return y, late_value


def validate_table(
    frame: pd.DataFrame,
    track: str,
    chain: str,
    snapshot: str = "main",
    *,
    source: Path | str = "<dataframe>",
) -> None:
    """Validate exact 2.1-dev schema, metadata, keys, labels, and task invariants."""
    track = normalize_track(track)
    snapshot = normalize_snapshot(snapshot)
    if chain not in CHAINS:
        raise BenchmarkDataError(f"chain must be one of {CHAINS}; got {chain!r}")
    _require_exact_columns(frame, EXPECTED_COLUMNS[track], source)
    _validate_common(frame, track, chain, snapshot, source)

    expected_id = frame["i_iso"].astype(str) + "|" + frame["stage"].astype(str)
    if track == "B1":
        if not frame["entry_id"].astype(str).equals(expected_id):
            _fail(source, "entry_id must equal '<i_iso>|<stage>'")
        _require_constant(frame, "task", "processed_export_stage_entry", source)
        _require_constant(frame, "task_unit", "exporter_stage", source)
        z, _ = _validate_outcomes(frame, "z", "entry_lateval", source)
        size = _numeric(frame, "size", source)
        capacity = _numeric(frame, "log_upstream_capacity", source)
        if not np.allclose(size, capacity, rtol=1e-9, atol=1e-8):
            _fail(source, "Track-B1 size must equal registered-upstream exporter capacity")
        candidates = _numeric(frame, "n_candidate_destinations", source)
        materialized = _numeric(frame, "n_materialized_destinations", source)
        if not np.equal(candidates, np.floor(candidates)).all() or (candidates < 1).any():
            _fail(source, "n_candidate_destinations must contain positive integers")
        if not np.equal(materialized, np.floor(materialized)).all():
            _fail(source, "n_materialized_destinations must contain non-negative integers")
        if (materialized > candidates).any() or not np.array_equal(z, (materialized > 0).astype(np.int8)):
            _fail(source, "entry label/count relationship is inconsistent")
        return

    if not frame["group_id"].astype(str).equals(expected_id):
        _fail(source, "group_id must equal '<i_iso>|<stage>'")
    _validate_outcomes(frame, "y", "lateval", source)
    size = _numeric(frame, "size", source)
    exporter = _numeric(frame, "log_exporter_capacity", source)
    importer = _numeric(frame, "log_importer_demand", source)
    if not np.allclose(size, exporter + importer, rtol=1e-9, atol=1e-8):
        _fail(source, "size must equal exporter capacity plus importer demand")
    _numeric(frame, "grav", source, allow_na=True)
    _numeric(frame, "gnn", source, allow_na=True, nonnegative=False)

    if track == "A":
        _require_constant(frame, "size_basis", "processed_exporter_plus_processed_importer", source)
        _require_constant(frame, "task", "destination_extension", source)
    else:
        _require_constant(
            frame,
            "size_basis",
            "registered_upstream_exporter_plus_processed_importer",
            source,
        )
        _require_constant(frame, "task", "conditional_destination_given_entry", source)
        if not frame["entry_id"].astype(str).equals(expected_id):
            _fail(source, "entry_id must equal '<i_iso>|<stage>'")
        entry_y = _binary(frame, "entry_y", source)
        if not (entry_y == 1).all():
            _fail(source, "Track B2 must contain only materialized entry events (entry_y=1)")
        if (frame.groupby(["i_iso", "stage"], sort=False)["y"].sum() < 1).any():
            _fail(source, "every Track-B2 entry group must contain at least one positive destination")
    _require_constant(frame, "task_unit", "exporter_stage_destination", source)


def load(
    track: str,
    chain: str,
    snapshot: str = "main",
    *,
    data_root: str | os.PathLike[str] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load one exact Track A/B1/B2 table for one chain and temporal snapshot."""
    track = normalize_track(track)
    snapshot = normalize_snapshot(snapshot)
    root = resolve_data_root(data_root)
    path = root / filename_for(track, chain, snapshot)
    if not path.is_file():
        raise BenchmarkDataError(f"missing v2 table: {path}")
    try:
        frame = pd.read_csv(
            path,
            dtype={"i_iso": str, "j_iso": str, "stage": str},
            low_memory=False,
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise BenchmarkDataError(f"cannot read {path}: {exc}") from exc
    if validate:
        validate_table(frame, track, chain, snapshot, source=path)
    frame.attrs.update(
        {
            "benchmark_track": track,
            "benchmark_chain": chain,
            "benchmark_snapshot": snapshot,
            "benchmark_source": str(path),
        }
    )
    return frame


def feature_columns(track: str) -> tuple[str, ...]:
    """Return the explicitly permitted shipped ex-ante score columns."""
    return BUILTIN_SCORE_COLUMNS[normalize_track(track)]


if __name__ == "__main__":
    root = resolve_data_root()
    print(f"UPGRADE-BENCH v2 data root: {root}")
    for snapshot in SNAPSHOTS:
        for chain in CHAINS:
            counts = [f"{track}={len(load(track, chain, snapshot, data_root=root)):,}" for track in TRACKS]
            print(f"{snapshot:5s} {chain:12s} " + "  ".join(counts))
