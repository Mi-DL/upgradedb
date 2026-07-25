#!/usr/bin/env python3
"""Read-only release validation for the UPGRADE-BENCH v2 tables.

The validator treats the lane tables as the source of truth and independently
reconstructs Track-B entry events and the conditional-destination cohort.  It
checks both the historical selection fold and the frozen main target cohort for
all six chains.  No file is written or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from split import OFFICIAL_SPLIT_UNIT, split_labels  # noqa: E402


DATA_ROOT = ROOT / "data" / "processed_v2"
CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
LANE_KEYS = ["i_iso", "j_iso", "stage"]
ENTRY_KEYS = ["i_iso", "stage"]
VERSION = "2.1-dev"

LANE_COLUMNS = {
    "i_iso",
    "j_iso",
    "stage",
    "y",
    "size",
    "log_exporter_capacity",
    "log_importer_demand",
    "size_basis",
    "grav",
    "gnn",
    "lateval",
    "benchmark_version",
    "aggregation",
    "early_window",
    "late_window",
    "group_id",
    "transductive_split_unit",
    "transductive_split",
    "temporal_role",
    "task",
    "task_unit",
}
TRACK_B_COLUMNS = LANE_COLUMNS | {"entry_id", "entry_y"}
ENTRY_COLUMNS = {
    "i_iso",
    "stage",
    "z",
    "size",
    "log_upstream_capacity",
    "entry_lateval",
    "n_candidate_destinations",
    "n_materialized_destinations",
    "benchmark_version",
    "aggregation",
    "early_window",
    "late_window",
    "entry_id",
    "transductive_split_unit",
    "transductive_split",
    "temporal_role",
    "task",
    "task_unit",
}
CONDITIONAL_COLUMNS = TRACK_B_COLUMNS


@dataclass(frozen=True)
class Snapshot:
    suffix: str
    role: str
    early_window: str
    late_window: str

    @property
    def label(self) -> str:
        return "main" if not self.suffix else self.suffix.removeprefix("_")


SNAPSHOTS = (
    Snapshot("", "target", "2008-2012", "2018-2022"),
    Snapshot("_fold2", "history", "1998-2002", "2008-2012"),
)


class V2ValidationError(AssertionError):
    """A release invariant is violated."""


def _fail(path: Path, message: str) -> None:
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = str(path)
    raise V2ValidationError(f"{label}: {message}")


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        _fail(path, "missing required v2 table")
    try:
        header = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError) as exc:
        _fail(path, f"cannot read CSV header: {exc}")
    missing = sorted(required - header)
    if missing:
        _fail(path, f"missing required columns {missing}")
    frame = pd.read_csv(
        path,
        dtype={"i_iso": str, "j_iso": str, "stage": str},
        low_memory=False,
    )
    if frame.empty:
        _fail(path, "table is empty")
    return frame


def _require_identity(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    for column in columns:
        values = frame[column]
        bad = values.isna() | values.astype(str).str.strip().isin({"", "nan", "None"})
        if bad.any():
            _fail(path, f"{column} has {int(bad.sum()):,} null/empty identities")


def _require_unique(frame: pd.DataFrame, keys: list[str], path: Path) -> None:
    duplicates = int(frame.duplicated(keys).sum())
    if duplicates:
        _fail(path, f"has {duplicates:,} duplicate rows for key {keys}")


def _require_constant(frame: pd.DataFrame, column: str, expected: str, path: Path) -> None:
    actual = set(frame[column].dropna().astype(str))
    nulls = int(frame[column].isna().sum())
    if actual != {expected} or nulls:
        _fail(path, f"{column} must be {expected!r}; found {sorted(actual)!r}, nulls={nulls}")


def _require_binary(frame: pd.DataFrame, column: str, path: Path) -> np.ndarray:
    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        values = sorted(set(frame[column].dropna().astype(str)))
        _fail(path, f"{column} must be complete binary labels; found {values[:10]}")
    return numeric.astype(np.int8)


def _require_finite_nonnegative(
    frame: pd.DataFrame,
    column: str,
    path: Path,
    *,
    allow_na: bool = False,
    nonnegative: bool = True,
) -> np.ndarray:
    source = frame[column]
    converted = pd.to_numeric(source, errors="coerce")
    parse_failures = source.notna() & converted.isna()
    if parse_failures.any():
        _fail(path, f"{column} has {int(parse_failures.sum()):,} non-numeric values")
    numeric = converted.to_numpy(float)
    if allow_na:
        checked = numeric[~np.isnan(numeric)]
    else:
        checked = numeric
    if not np.isfinite(checked).all() or (nonnegative and (checked < 0).any()):
        sign = " non-negative" if nonnegative else ""
        _fail(path, f"{column} must contain finite{sign} values"
                    + (" where present" if allow_na else ""))
    if not allow_na and len(checked) != len(numeric):
        _fail(path, f"{column} contains missing values")
    return numeric


def _require_close(
    actual: np.ndarray | pd.Series,
    expected: np.ndarray | pd.Series,
    path: Path,
    message: str,
) -> None:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=1e-9, atol=1e-8, equal_nan=True
    ):
        if left.shape == right.shape:
            delta = np.abs(left - right)
            finite = delta[np.isfinite(delta)]
            max_delta = float(finite.max()) if finite.size else float("nan")
            message = f"{message} (max absolute difference={max_delta:g})"
        _fail(path, message)


def _validate_metadata(frame: pd.DataFrame, path: Path, snapshot: Snapshot) -> None:
    _require_constant(frame, "benchmark_version", VERSION, path)
    _require_constant(frame, "aggregation", "calendar_mean", path)
    _require_constant(frame, "early_window", snapshot.early_window, path)
    _require_constant(frame, "late_window", snapshot.late_window, path)
    _require_constant(frame, "temporal_role", snapshot.role, path)
    _require_constant(frame, "transductive_split_unit", OFFICIAL_SPLIT_UNIT, path)


def _validate_split(
    frame: pd.DataFrame,
    path: Path,
    chain: str,
    *,
    group_column: str,
    importer: np.ndarray | pd.Series | None = None,
) -> None:
    expected_id = frame["i_iso"].astype(str) + "|" + frame["stage"].astype(str)
    if not frame[group_column].astype(str).equals(expected_id):
        _fail(path, f"{group_column} must equal '<i_iso>|<stage>'")

    splits = frame["transductive_split"].astype(str)
    unexpected = sorted(set(splits) - {"train", "test"})
    if unexpected or frame["transductive_split"].isna().any():
        _fail(path, f"invalid transductive_split values {unexpected}")
    leakage = frame.assign(_group=expected_id, _split=splits).groupby("_group")["_split"].nunique()
    if (leakage != 1).any():
        _fail(path, f"{int((leakage != 1).sum()):,} exporter-stage groups cross splits")

    if importer is None:
        importer = np.repeat("__ENTRY__", len(frame))
    expected_split = split_labels(
        chain,
        frame["i_iso"],
        frame["stage"],
        importer,
        unit=OFFICIAL_SPLIT_UNIT,
    )
    if not np.array_equal(splits.to_numpy(), expected_split):
        _fail(path, "transductive_split does not match the official deterministic assignment")


def _validate_outcomes(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    path: Path,
) -> np.ndarray:
    labels = _require_binary(frame, label_column, path)
    values = _require_finite_nonnegative(frame, value_column, path)
    if np.any(values[labels == 0] != 0.0):
        _fail(path, f"{value_column} must be zero whenever {label_column}=0")
    if np.any(values[labels == 1] <= 100.0):
        _fail(path, f"{value_column} must exceed 100 kUSD whenever {label_column}=1")
    return labels


def _validate_lane(
    frame: pd.DataFrame,
    path: Path,
    chain: str,
    snapshot: Snapshot,
    *,
    track: str,
    conditional: bool = False,
) -> None:
    _require_identity(frame, LANE_KEYS, path)
    _require_unique(frame, LANE_KEYS, path)
    _validate_metadata(frame, path, snapshot)
    _validate_split(frame, path, chain, group_column="group_id", importer=frame["j_iso"])
    _validate_outcomes(frame, "y", "lateval", path)

    size = _require_finite_nonnegative(frame, "size", path)
    exporter = _require_finite_nonnegative(frame, "log_exporter_capacity", path)
    importer = _require_finite_nonnegative(frame, "log_importer_demand", path)
    _require_close(size, exporter + importer, path, "size is not the sum of its components")
    _require_finite_nonnegative(frame, "grav", path, allow_na=True)
    # GNN scores are unconstrained logits; negative values are valid.
    _require_finite_nonnegative(frame, "gnn", path, allow_na=True, nonnegative=False)

    basis = {
        "A": "processed_exporter_plus_processed_importer",
        "B": "registered_upstream_exporter_plus_processed_importer",
    }[track]
    _require_constant(frame, "size_basis", basis, path)
    if track == "A":
        task = "destination_extension"
    elif conditional:
        task = "conditional_destination_given_entry"
    else:
        task = "processed_export_entry_candidate_lane"
    _require_constant(frame, "task", task, path)
    _require_constant(frame, "task_unit", "exporter_stage_destination", path)

    if track == "B":
        entry_y = _require_binary(frame, "entry_y", path)
        expected_entry = frame["i_iso"].astype(str) + "|" + frame["stage"].astype(str)
        if not frame["entry_id"].astype(str).equals(expected_entry):
            _fail(path, "entry_id must equal '<i_iso>|<stage>'")
        grouped = frame.assign(_entry_y=entry_y).groupby(ENTRY_KEYS, sort=False)
        if (grouped["log_exporter_capacity"].nunique(dropna=False) != 1).any():
            _fail(path, "log_exporter_capacity is not constant within an entry event")
        if (grouped["_entry_y"].nunique() != 1).any():
            _fail(path, "entry_y is not constant within an entry event")
        expected_y = grouped["y"].transform("max").to_numpy(dtype=np.int8)
        if not np.array_equal(entry_y, expected_y):
            _fail(path, "entry_y does not equal max(y) for its exporter-stage event")


def _validate_entries(
    lanes: pd.DataFrame,
    entries: pd.DataFrame,
    path: Path,
    chain: str,
    snapshot: Snapshot,
) -> None:
    _require_identity(entries, ENTRY_KEYS, path)
    _require_unique(entries, ENTRY_KEYS, path)
    _validate_metadata(entries, path, snapshot)
    _validate_split(entries, path, chain, group_column="entry_id")
    _require_constant(entries, "task", "processed_export_stage_entry", path)
    _require_constant(entries, "task_unit", "exporter_stage", path)
    _validate_outcomes(entries, "z", "entry_lateval", path)

    size = _require_finite_nonnegative(entries, "size", path)
    capacity = _require_finite_nonnegative(entries, "log_upstream_capacity", path)
    _require_close(size, capacity, path, "Track-B1 size must equal registered-upstream capacity")

    for column in ("n_candidate_destinations", "n_materialized_destinations"):
        values = pd.to_numeric(entries[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            _fail(path, f"{column} must contain non-negative integers")
    if (entries["n_candidate_destinations"] < 1).any():
        _fail(path, "every entry must have at least one candidate destination")
    if (entries["n_materialized_destinations"] > entries["n_candidate_destinations"]).any():
        _fail(path, "materialized destinations exceed candidate destinations")

    expected = (
        lanes.groupby(ENTRY_KEYS, as_index=False, sort=True)
        .agg(
            z=("y", "max"),
            size=("log_exporter_capacity", "first"),
            log_upstream_capacity=("log_exporter_capacity", "first"),
            entry_lateval=("lateval", "sum"),
            n_candidate_destinations=("j_iso", "size"),
            n_materialized_destinations=("y", "sum"),
            benchmark_version=("benchmark_version", "first"),
            aggregation=("aggregation", "first"),
            early_window=("early_window", "first"),
            late_window=("late_window", "first"),
            transductive_split=("transductive_split", "first"),
        )
        .sort_values(ENTRY_KEYS, kind="stable")
        .reset_index(drop=True)
    )
    expected["entry_id"] = expected["i_iso"] + "|" + expected["stage"]
    actual = entries.sort_values(ENTRY_KEYS, kind="stable").reset_index(drop=True)
    if len(actual) != len(expected):
        _fail(path, f"expected {len(expected):,} entries reconstructed from lanes; found {len(actual):,}")

    exact_columns = ENTRY_KEYS + [
        "z",
        "n_candidate_destinations",
        "n_materialized_destinations",
        "benchmark_version",
        "aggregation",
        "early_window",
        "late_window",
        "entry_id",
        "transductive_split",
    ]
    for column in exact_columns:
        if not actual[column].astype(str).equals(expected[column].astype(str)):
            _fail(path, f"entry aggregation mismatch in {column}")
    for column in ("size", "log_upstream_capacity", "entry_lateval"):
        _require_close(actual[column], expected[column], path, f"entry aggregation mismatch in {column}")


def _validate_conditional(
    lanes: pd.DataFrame,
    conditional: pd.DataFrame,
    path: Path,
    chain: str,
    snapshot: Snapshot,
) -> None:
    _validate_lane(conditional, path, chain, snapshot, track="B", conditional=True)
    if set(conditional["entry_y"].astype(int)) != {1}:
        _fail(path, "conditional cohort must contain only materialized entry events")

    expected = lanes.loc[lanes["entry_y"].astype(int) == 1].sort_values(LANE_KEYS).reset_index(drop=True)
    actual = conditional.sort_values(LANE_KEYS).reset_index(drop=True)
    if len(actual) != len(expected):
        _fail(path, f"conditional cohort expected {len(expected):,} lanes; found {len(actual):,}")
    try:
        comparable_columns = [column for column in expected.columns if column != "task"]
        pd.testing.assert_frame_equal(
            actual[comparable_columns],
            expected[comparable_columns],
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except (AssertionError, KeyError) as exc:
        detail = str(exc).splitlines()[0] if str(exc) else "row mismatch"
        _fail(path, f"is not the exact entry_y=1 subset of the Track-B lane table ({detail})")


def _summary_row(chain: str, track_a: pd.DataFrame, track_b: pd.DataFrame, entries: pd.DataFrame) -> dict:
    conditional = track_b.loc[track_b["entry_y"].astype(int) == 1]
    return {
        "chain": chain,
        "track_a_candidates": int(len(track_a)),
        "track_a_positive_lanes": int(track_a["y"].sum()),
        "track_a_base_rate": float(track_a["y"].mean()),
        "track_a_observed_late_value_kusd": float(track_a["lateval"].sum()),
        "track_b_candidate_lanes": int(len(track_b)),
        "track_b_positive_lanes": int(track_b["y"].sum()),
        "track_b_lane_base_rate": float(track_b["y"].mean()),
        "track_b_unique_entries": int(len(entries)),
        "track_b_positive_entries": int(entries["z"].sum()),
        "track_b_entry_base_rate": float(entries["z"].mean()),
        "track_b_observed_late_value_kusd": float(entries["entry_lateval"].sum()),
        "track_b2_conditional_lanes": int(len(conditional)),
        "track_b2_positive_lanes": int(conditional["y"].sum()),
        "track_b2_base_rate": float(conditional["y"].mean()),
        "track_b2_observed_late_value_kusd": float(conditional["lateval"].sum()),
    }


def _totals(rows: list[dict]) -> dict:
    result = {"chain": "TOTAL"}
    counts = [key for key in rows[0] if key != "chain" and not key.endswith("base_rate")]
    for key in counts:
        result[key] = sum(row[key] for row in rows)
    result["track_a_base_rate"] = result["track_a_positive_lanes"] / result["track_a_candidates"]
    result["track_b_lane_base_rate"] = result["track_b_positive_lanes"] / result["track_b_candidate_lanes"]
    result["track_b_entry_base_rate"] = result["track_b_positive_entries"] / result["track_b_unique_entries"]
    result["track_b2_base_rate"] = result["track_b2_positive_lanes"] / result["track_b2_conditional_lanes"]
    return result


def _compare_summary_rows(actual: list[dict], expected: list[dict], path: Path) -> None:
    by_chain = {str(row.get("chain")): row for row in actual}
    if set(by_chain) != {str(row["chain"]) for row in expected}:
        _fail(path, f"summary chains must be {[row['chain'] for row in expected]}")
    for wanted in expected:
        found = by_chain[str(wanted["chain"])]
        for key, value in wanted.items():
            if key == "chain":
                continue
            if key not in found:
                _fail(path, f"summary row {wanted['chain']} missing {key}")
            try:
                close = np.isclose(float(found[key]), float(value), rtol=1e-9, atol=1e-8)
            except (TypeError, ValueError):
                close = False
            if not close:
                _fail(path, f"summary mismatch for {wanted['chain']}.{key}: {found[key]!r} != {value!r}")


def _validate_summaries(data_root: Path, snapshot: Snapshot, rows: list[dict]) -> None:
    json_path = data_root / f"dataset_summary{snapshot.suffix}.json"
    csv_path = data_root / f"dataset_summary{snapshot.suffix}.csv"
    if not json_path.is_file():
        _fail(json_path, "missing v2 JSON summary")
    if not csv_path.is_file():
        _fail(csv_path, "missing v2 CSV summary")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(json_path, f"cannot parse summary: {exc}")

    metadata = {
        "benchmark_version": VERSION,
        "aggregation": "calendar_mean",
        "official_temporal_protocol": (
            "historical-fold selection -> frozen main target evaluation"
        ),
        "diagnostic_split_unit": OFFICIAL_SPLIT_UNIT,
        "temporal_role": snapshot.role,
        "source_suffix": snapshot.suffix,
    }
    for key, value in metadata.items():
        if payload.get(key) != value:
            _fail(json_path, f"{key} must be {value!r}; found {payload.get(key)!r}")
    if not isinstance(payload.get("chains"), list) or not isinstance(payload.get("totals"), dict):
        _fail(json_path, "summary must contain chain rows and totals")
    totals = _totals(rows)
    _compare_summary_rows(payload["chains"], rows, json_path)
    _compare_summary_rows([payload["totals"]], [totals], json_path)

    try:
        csv_rows = pd.read_csv(csv_path).to_dict(orient="records")
    except (OSError, pd.errors.ParserError) as exc:
        _fail(csv_path, f"cannot parse summary: {exc}")
    _compare_summary_rows(csv_rows, rows + [totals], csv_path)


def validate_snapshot(
    data_root: Path,
    snapshot: Snapshot,
    *,
    chains: Iterable[str] = CHAINS,
    check_summaries: bool = True,
) -> list[dict]:
    """Validate one temporal snapshot and return independently derived counts."""
    rows: list[dict] = []
    chains = tuple(chains)
    for chain in chains:
        a_path = data_root / f"candidates_{chain}{snapshot.suffix}.csv"
        b_path = data_root / f"candidates_firsttime_{chain}{snapshot.suffix}.csv"
        e_path = data_root / f"entries_firsttime_{chain}{snapshot.suffix}.csv"
        c_path = data_root / f"destinations_given_entry_{chain}{snapshot.suffix}.csv"

        track_a = _read(a_path, LANE_COLUMNS)
        _validate_lane(track_a, a_path, chain, snapshot, track="A")
        track_b = _read(b_path, TRACK_B_COLUMNS)
        _validate_lane(track_b, b_path, chain, snapshot, track="B")
        entries = _read(e_path, ENTRY_COLUMNS)
        _validate_entries(track_b, entries, e_path, chain, snapshot)
        conditional = _read(c_path, CONDITIONAL_COLUMNS)
        _validate_conditional(track_b, conditional, c_path, chain, snapshot)

        rows.append(_summary_row(chain, track_a, track_b, entries))
        print(
            f"v2 {snapshot.label:<5} {chain:<12} OK  "
            f"A={len(track_a):>7,}  B-lanes={len(track_b):>7,}  entries={len(entries):>4,}"
        )

    if check_summaries:
        _validate_summaries(data_root, snapshot, rows)
        print(f"v2 {snapshot.label:<5} summaries OK")
    return rows


def validate_release(
    data_root: Path = DATA_ROOT,
    *,
    chains: Iterable[str] = CHAINS,
    check_summaries: bool = True,
) -> dict[str, list[dict]]:
    """Validate both release snapshots without mutating the artifact."""
    data_root = Path(data_root).resolve()
    chains = tuple(chains)
    report = {
        snapshot.label: validate_snapshot(
            data_root,
            snapshot,
            chains=chains,
            check_summaries=check_summaries,
        )
        for snapshot in SNAPSHOTS
    }
    print(
        f"V2 VALIDATION PASSED: {len(SNAPSHOTS)} snapshots x "
        f"{len(chains)} chains (read-only)"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="validate tables but not dataset_summary files (mainly for unit fixtures)",
    )
    args = parser.parse_args()
    try:
        validate_release(args.data_root, check_summaries=not args.skip_summaries)
    except (V2ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"V2 VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
