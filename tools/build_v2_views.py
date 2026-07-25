#!/usr/bin/env python3
"""Materialize v2 split metadata, Track-B event views, and coverage summaries.

This tool is deterministic and reads only calendar-mean candidate tables produced by
``src/temporal_backtest.py`` under ``data/processed_v2``. It never touches frozen v1 files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from split import OFFICIAL_SPLIT_UNIT, split_labels  # noqa: E402


CHAINS = ["sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy"]
KEYS = ["i_iso", "j_iso", "stage"]
ENTRY_KEYS = ["i_iso", "stage"]
SAFE_SUFFIX = re.compile(r"^_[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validate_suffix(suffix: str) -> str:
    """Return a filename-safe optional suffix (for example, ``_fold2``)."""
    if not suffix:
        return ""
    if not SAFE_SUFFIX.fullmatch(suffix):
        raise ValueError(
            "--suffix must be empty or start with '_' and contain only letters, "
            "numbers, '_' and '-'"
        )
    return suffix


def _temporal_role(suffix: str, explicit_role: str | None = None) -> str:
    """Default unsuffixed tables to the target cohort and folds to history."""
    if explicit_role is not None:
        return explicit_role
    return "target" if not suffix else "history"


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"i_iso": str, "j_iso": str, "stage": str})
    missing = [column for column in KEYS + ["y", "lateval", "aggregation"] if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} contains duplicate candidate keys")
    modes = set(frame.aggregation.dropna().astype(str))
    if modes != {"calendar_mean"}:
        raise ValueError(f"{path} is not a canonical v2 table: aggregation={sorted(modes)}")
    # Canonicalize row order even for tables produced before deterministic set
    # iteration was enforced in temporal_backtest.py. This makes release hashes
    # independent of PYTHONHASHSEED without changing candidate identities.
    return frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _attach_transductive_metadata(
    frame: pd.DataFrame, chain: str, temporal_role: str = "target"
) -> pd.DataFrame:
    out = frame.copy()
    out["group_id"] = out.i_iso.astype(str) + "|" + out.stage.astype(str)
    out["transductive_split_unit"] = OFFICIAL_SPLIT_UNIT
    out["transductive_split"] = split_labels(
        chain,
        out.i_iso,
        out.stage,
        out.j_iso,
        unit=OFFICIAL_SPLIT_UNIT,
    )
    out["temporal_role"] = temporal_role
    leakage = out.groupby("group_id").transductive_split.nunique()
    if int((leakage > 1).sum()) != 0:
        raise AssertionError(f"{chain}: exporter-stage group crosses transductive split")
    return out


def _entry_view(
    lanes: pd.DataFrame, chain: str, temporal_role: str = "target"
) -> pd.DataFrame:
    capacity_spread = lanes.groupby(ENTRY_KEYS).log_exporter_capacity.nunique(dropna=False)
    if int((capacity_spread > 1).sum()) != 0:
        raise ValueError(f"{chain}: Track-B exporter capacity is not constant within an entry group")

    entries = (
        lanes.groupby(ENTRY_KEYS, as_index=False)
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
        )
        .sort_values(ENTRY_KEYS, kind="stable")
        .reset_index(drop=True)
    )
    entries["entry_id"] = entries.i_iso.astype(str) + "|" + entries.stage.astype(str)
    entries["transductive_split_unit"] = OFFICIAL_SPLIT_UNIT
    entries["transductive_split"] = split_labels(
        chain,
        entries.i_iso,
        entries.stage,
        np.repeat("__ENTRY__", len(entries)),
        unit=OFFICIAL_SPLIT_UNIT,
    )
    entries["temporal_role"] = temporal_role
    entries["task"] = "processed_export_stage_entry"
    entries["task_unit"] = "exporter_stage"
    return entries


def _summary_row(chain: str, track_a: pd.DataFrame, track_b: pd.DataFrame, entries: pd.DataFrame) -> dict:
    conditional = track_b.loc[track_b["entry_y"].eq(1)]
    return {
        "chain": chain,
        "track_a_candidates": int(len(track_a)),
        "track_a_positive_lanes": int(track_a.y.sum()),
        "track_a_base_rate": float(track_a.y.mean()),
        "track_a_observed_late_value_kusd": float(track_a.lateval.sum()),
        "track_b_candidate_lanes": int(len(track_b)),
        "track_b_positive_lanes": int(track_b.y.sum()),
        "track_b_lane_base_rate": float(track_b.y.mean()),
        "track_b_unique_entries": int(len(entries)),
        "track_b_positive_entries": int(entries.z.sum()),
        "track_b_entry_base_rate": float(entries.z.mean()),
        "track_b_observed_late_value_kusd": float(entries.entry_lateval.sum()),
        "track_b2_conditional_lanes": int(len(conditional)),
        "track_b2_positive_lanes": int(conditional.y.sum()),
        "track_b2_base_rate": float(conditional.y.mean()),
        "track_b2_observed_late_value_kusd": float(conditional.lateval.sum()),
    }


def build_views(
    data_root: Path,
    *,
    suffix: str = "",
    temporal_role: str | None = None,
    chains: list[str] | tuple[str, ...] = CHAINS,
) -> dict:
    """Build lane, entry, conditional-destination, and summary views.

    ``suffix`` is appended to every input and output stem. Thus ``_fold2`` reads
    ``candidates_<chain>_fold2.csv`` and writes
    ``entries_firsttime_<chain>_fold2.csv`` plus
    ``dataset_summary_fold2.{json,csv}`` without touching the main target files.
    """
    suffix = _validate_suffix(suffix)
    role = _temporal_role(suffix, temporal_role)
    if role not in {"target", "history"}:
        raise ValueError("temporal_role must be 'target' or 'history'")
    data_root = data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for chain in chains:
        path_a = data_root / f"candidates_{chain}{suffix}.csv"
        path_b = data_root / f"candidates_firsttime_{chain}{suffix}.csv"
        track_a = _attach_transductive_metadata(_read(path_a), chain, role)
        track_b = _attach_transductive_metadata(_read(path_b), chain, role)
        track_a["task"] = "destination_extension"
        track_a["task_unit"] = "exporter_stage_destination"
        # This table is the destination-level expansion used to derive B1 and
        # evaluate B2; it must not itself be mistaken for independent B1 events.
        track_b["task"] = "processed_export_entry_candidate_lane"
        track_b["task_unit"] = "exporter_stage_destination"
        entries = _entry_view(track_b, chain, role)
        entry_positive = set(entries.loc[entries.z == 1, "entry_id"])
        track_b["entry_id"] = track_b.i_iso.astype(str) + "|" + track_b.stage.astype(str)
        track_b["entry_y"] = track_b.entry_id.isin(entry_positive).astype(int)

        conditional = track_b[track_b.entry_y == 1].copy()
        conditional["task"] = "conditional_destination_given_entry"
        conditional["task_unit"] = "exporter_stage_destination"

        track_a.to_csv(path_a, index=False)
        track_b.to_csv(path_b, index=False)
        entries.to_csv(data_root / f"entries_firsttime_{chain}{suffix}.csv", index=False)
        conditional.to_csv(
            data_root / f"destinations_given_entry_{chain}{suffix}.csv", index=False
        )
        summary.append(_summary_row(chain, track_a, track_b, entries))
        print(
            f"{chain:<12} A={len(track_a):>7}/{int(track_a.y.sum()):>4}  "
            f"B lanes={len(track_b):>7}/{int(track_b.y.sum()):>3}  "
            f"B entries={len(entries):>4}/{int(entries.z.sum()):>3}"
        )

    totals = {
        key: (sum(row[key] for row in summary) if key != "chain" and not key.endswith("base_rate") else None)
        for key in summary[0]
        if key != "chain"
    }
    totals["chain"] = "TOTAL"
    totals["track_a_base_rate"] = totals["track_a_positive_lanes"] / totals["track_a_candidates"]
    totals["track_b_lane_base_rate"] = totals["track_b_positive_lanes"] / totals["track_b_candidate_lanes"]
    totals["track_b_entry_base_rate"] = totals["track_b_positive_entries"] / totals["track_b_unique_entries"]
    totals["track_b2_base_rate"] = totals["track_b2_positive_lanes"] / totals["track_b2_conditional_lanes"]
    payload = {
        "benchmark_version": "2.1-dev",
        "aggregation": "calendar_mean",
        "official_temporal_protocol": "historical-fold selection -> frozen main target evaluation",
        "diagnostic_split_unit": OFFICIAL_SPLIT_UNIT,
        "temporal_role": role,
        "source_suffix": suffix,
        "chains": summary,
        "totals": totals,
    }
    summary_json = data_root / f"dataset_summary{suffix}.json"
    summary_csv = data_root / f"dataset_summary{suffix}.csv"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary + [totals]).to_csv(summary_csv, index=False)
    print(f"saved summary -> {summary_json}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "processed_v2")
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional safe filename suffix, e.g. _fold2 (defaults to historical role).",
    )
    parser.add_argument(
        "--temporal-role",
        choices=["target", "history"],
        default=None,
        help="Override the inferred role (target without a suffix, history with one).",
    )
    args = parser.parse_args()
    build_views(args.data_root, suffix=args.suffix, temporal_role=args.temporal_role)


if __name__ == "__main__":
    main()
