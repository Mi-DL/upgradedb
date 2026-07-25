#!/usr/bin/env python3
"""Protocol-safe value, oracle-gap, and headroom diagnostics.

This is a descriptive diagnostic layer over the already frozen rolling
evaluation.  It never selects a model from main-window outcomes:

* CPU models are reconstructed with the hyperparameters recorded in the
  verified historical-fold artifact, then refit on the complete historical
  fold exactly as specified there.
* Both GPU families and every prescribed seed are reported separately from
  the frozen score CSVs.  Raw scores are never averaged and no family is
  promoted using main outcomes.
* Outcome-ranked oracles are budget-matched upper bounds, not deployable
  models.  Headroom is always oracle minus the named frozen scorer on the same
  cohort and budget.

The three task-aligned reporting points are fixed before reading target labels:
Track A global top-500 plus top-10 per exporter, Track B1 global top-50 plus
top-1 per exporter, and Track B2 top-3 within each positive entry.  Track B2 is
nested inside Track B1, so its dollars are audited but never added to B1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v2_rolling_cpu_baselines as cpu  # noqa: E402


SCHEMA_VERSION = "upgrade-bench-v2-value-diagnostics-1"
GPU_SUMMARY_SCHEMA = "upgrade-bench-v2/gpu-main-summary/1"
PROTOCOL = "strict_rolling_fold2_to_main"
CHAINS = cpu.CHAINS
GPU_FAMILIES = ("kge", "nbfnet")
GPU_SEEDS = (0, 1, 2, 3, 4)

TRACK_A_GLOBAL_K = 500
TRACK_A_EXPORTER_K = 10
TRACK_B1_GLOBAL_K = 50
TRACK_B1_EXPORTER_K = 1
TRACK_B2_ENTRY_K = 3

DEFAULT_DATA = ROOT / "data" / "processed_v2"
DEFAULT_CPU = ROOT / "results_v2" / "metrics" / "rolling_cpu_baselines.json"
DEFAULT_GPU = ROOT / "results_v2" / "metrics" / "v2_gpu_rolling_summary.json"
DEFAULT_GPU_ROOT = ROOT / "results_v2" / "gpu_rolling"
DEFAULT_JSON = ROOT / "results_v2" / "metrics" / "v2_value_diagnostics.json"
DEFAULT_CSV = ROOT / "results_v2" / "metrics" / "v2_value_diagnostics.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _stable_seed(base: int, *parts: object) -> int:
    token = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")


def _ci(values: Iterable[float]) -> list[float] | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return [float(item) for item in np.quantile(array, [0.025, 0.975])]


def _mean_std(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_seeds": int(len(records))}
    for key in keys:
        values = np.asarray([float(record[key]) for record in records], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return result


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{role}: missing columns {missing}")


def _single_value(frame: pd.DataFrame, column: str, expected: str, role: str) -> None:
    values = sorted(frame[column].dropna().astype(str).unique().tolist())
    if values != [expected]:
        raise ValueError(f"{role}: {column}={values!r}, expected {expected!r}")


def _validate_outcomes(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    keys: Sequence[str],
    role: str,
) -> None:
    _require_columns(frame, [*keys, label, value], role)
    if frame.loc[:, list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
        raise ValueError(f"{role}: null or duplicate task keys")
    y = pd.to_numeric(frame[label], errors="raise").to_numpy(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{role}: outcome is not binary")
    lateval = pd.to_numeric(frame[value], errors="raise").to_numpy(float)
    if not np.isfinite(lateval).all() or bool((lateval < 0).any()):
        raise ValueError(f"{role}: observed value is negative or non-finite")
    if bool((lateval[y == 0] != 0).any()):
        raise ValueError(f"{role}: negative outcomes carry observed late value")


@dataclass
class FormalTables:
    track_a: pd.DataFrame
    track_b_lanes: pd.DataFrame
    track_b1: pd.DataFrame
    track_b2: pd.DataFrame
    audits: dict[str, Any]


def _read_csv(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"{role}: missing {path}")
    return pd.read_csv(path, dtype={"i_iso": str, "j_iso": str, "stage": str, "entry_id": str})


def _compare_numeric(
    left: pd.Series,
    right: pd.Series,
    *,
    role: str,
    atol: float = 1e-8,
) -> None:
    if not np.allclose(
        pd.to_numeric(left, errors="raise").to_numpy(float),
        pd.to_numeric(right, errors="raise").to_numpy(float),
        rtol=0,
        atol=atol,
    ):
        raise ValueError(f"{role}: numeric columns disagree")


def _read_formal_tables(data_dir: Path, chain: str) -> FormalTables:
    paths = {
        "track_a": data_dir / f"candidates_{chain}.csv",
        "track_b_lanes": data_dir / f"candidates_firsttime_{chain}.csv",
        "track_b1": data_dir / f"entries_firsttime_{chain}.csv",
        "track_b2": data_dir / f"destinations_given_entry_{chain}.csv",
    }
    track_a, audit_a = cpu._read_candidate(data_dir, chain, track="a", historical=False)
    track_b_lanes, audit_b = cpu._read_candidate(data_dir, chain, track="b", historical=False)
    track_b1 = _read_csv(paths["track_b1"], f"{chain}/B1")
    track_b2 = _read_csv(paths["track_b2"], f"{chain}/B2")

    _validate_outcomes(
        track_a,
        label="y",
        value="lateval",
        keys=("i_iso", "j_iso", "stage"),
        role=f"{chain}/A",
    )
    _validate_outcomes(
        track_b1,
        label="z",
        value="entry_lateval",
        keys=("i_iso", "stage"),
        role=f"{chain}/B1",
    )
    _validate_outcomes(
        track_b2,
        label="y",
        value="lateval",
        keys=("i_iso", "j_iso", "stage"),
        role=f"{chain}/B2",
    )
    for frame, role, task, unit in (
        (track_b1, "B1", "processed_export_stage_entry", "exporter_stage"),
        (track_b2, "B2", "conditional_destination_given_entry", "exporter_stage_destination"),
    ):
        _require_columns(
            frame,
            ["benchmark_version", "aggregation", "early_window", "late_window", "temporal_role", "task", "task_unit"],
            f"{chain}/{role}",
        )
        _single_value(frame, "aggregation", "calendar_mean", f"{chain}/{role}")
        _single_value(frame, "early_window", "2008-2012", f"{chain}/{role}")
        _single_value(frame, "late_window", "2018-2022", f"{chain}/{role}")
        _single_value(frame, "temporal_role", "target", f"{chain}/{role}")
        _single_value(frame, "task", task, f"{chain}/{role}")
        _single_value(frame, "task_unit", unit, f"{chain}/{role}")

    derived = cpu._derive_entry_table(track_b_lanes).sort_values(
        ["i_iso", "stage"], kind="mergesort"
    ).reset_index(drop=True)
    official = track_b1.sort_values(["i_iso", "stage"], kind="mergesort").reset_index(drop=True)
    if not derived.loc[:, ["i_iso", "stage"]].equals(official.loc[:, ["i_iso", "stage"]]):
        raise ValueError(f"{chain}: formal B1 keys differ from the lane-derived entry cohort")
    if not np.array_equal(derived["z"].to_numpy(int), official["z"].to_numpy(int)):
        raise ValueError(f"{chain}: formal B1 labels differ from lane-derived labels")
    _compare_numeric(derived["entry_lateval"], official["entry_lateval"], role=f"{chain}/B1 value")
    if not np.array_equal(
        derived["n_candidate_destinations"].to_numpy(int),
        official["n_candidate_destinations"].to_numpy(int),
    ):
        raise ValueError(f"{chain}: formal B1 destination counts differ from lanes")

    positive_ids = set(official.loc[official["z"].eq(1), "entry_id"].astype(str))
    expected_b2 = track_b_lanes.loc[track_b_lanes["entry_id"].isin(positive_ids)].copy()
    expected_keys = expected_b2.loc[:, ["i_iso", "j_iso", "stage"]].sort_values(
        ["i_iso", "j_iso", "stage"], kind="mergesort"
    ).reset_index(drop=True)
    actual_keys = track_b2.loc[:, ["i_iso", "j_iso", "stage"]].sort_values(
        ["i_iso", "j_iso", "stage"], kind="mergesort"
    ).reset_index(drop=True)
    if not expected_keys.equals(actual_keys):
        raise ValueError(f"{chain}: formal B2 is not exactly the positive-B1 lane subset")

    b1_value = float(official["entry_lateval"].sum())
    b2_value = float(track_b2["lateval"].sum())
    if not math.isclose(b1_value, b2_value, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{chain}: B1/B2 nested dollar totals differ: {b1_value} vs {b2_value}")

    audits = {
        "track_a": {"path": paths["track_a"].relative_to(ROOT).as_posix(), "sha256": _sha256(paths["track_a"]), **audit_a},
        "track_b_lane_pool": {"path": paths["track_b_lanes"].relative_to(ROOT).as_posix(), "sha256": _sha256(paths["track_b_lanes"]), **audit_b},
        "track_b1": {
            "path": paths["track_b1"].relative_to(ROOT).as_posix(),
            "sha256": _sha256(paths["track_b1"]),
            "rows": int(len(official)),
            "positives": int(official["z"].sum()),
            "observed_late_value_kusd": b1_value,
        },
        "track_b2": {
            "path": paths["track_b2"].relative_to(ROOT).as_posix(),
            "sha256": _sha256(paths["track_b2"]),
            "rows": int(len(track_b2)),
            "positive_lanes": int(track_b2["y"].sum()),
            "positive_entry_groups": int(track_b2["entry_id"].nunique()),
            "observed_late_value_kusd": b2_value,
        },
    }
    return FormalTables(track_a, track_b_lanes, official, track_b2, audits)


@dataclass
class FrozenCpu:
    a_size: cpu.FrozenRawScore
    a_gravity: cpu.FrozenRawScore
    a_logistic: Any
    b1_capacity: cpu.FrozenRawScore
    b1_logistic: Any
    b2_demand: cpu.FrozenRawScore
    b2_gravity: cpu.FrozenRawScore
    b2_logistic: Any
    selections: dict[str, Any]


def _frozen_selection(chain_payload: Mapping[str, Any], track: str, model: str) -> Mapping[str, Any]:
    try:
        selection = chain_payload[track]["models"][model]["model"]["selection"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"CPU artifact lacks frozen selection {track}/{model}") from exc
    if not isinstance(selection, Mapping) or "selected_C" not in selection:
        raise ValueError(f"CPU artifact has malformed frozen selection {track}/{model}")
    return selection


def _raw_from_artifact(chain_payload: Mapping[str, Any], track: str, model: str) -> cpu.FrozenRawScore:
    record = chain_payload[track]["models"][model]["model"]
    return cpu.FrozenRawScore(
        feature=str(record["feature"]),
        fill_value=(None if record.get("historical_fill_value") is None else float(record["historical_fill_value"])),
    )


def _fit_cpu_from_artifact(
    data_dir: Path,
    chain: str,
    chain_payload: Mapping[str, Any],
    *,
    chain_seed: int,
) -> FrozenCpu:
    """Refit only; all hyperparameters come from the verified saved artifact."""
    history_a, audit_a = cpu._read_candidate(data_dir, chain, track="a", historical=True)
    history_b, audit_b = cpu._read_candidate(data_dir, chain, track="b", historical=True)
    saved_a = chain_payload["protocol_audit"]["history_track_a"]
    saved_b = chain_payload["protocol_audit"]["history_track_b"]
    if audit_a["sha256"] != saved_a["sha256"] or audit_b["sha256"] != saved_b["sha256"]:
        raise ValueError(f"{chain}: history bytes differ from the CPU rolling artifact")

    a_selection = _frozen_selection(
        chain_payload, "track_a_destination_extension", "historical_logistic_size_gravity"
    )
    if tuple(a_selection.get("feature_names", ())) != cpu.TRACK_A_FEATURES:
        raise ValueError(f"{chain}: frozen A feature schema changed")
    a_features = cpu._track_a_features(history_a)
    a_logistic = cpu._pipeline(float(a_selection["selected_C"]), chain_seed + 11)
    a_logistic.fit(a_features.loc[:, list(cpu.TRACK_A_FEATURES)], history_a["y"].to_numpy(np.int8))

    history_entries = cpu._derive_entry_table(history_b)
    b1_selection = _frozen_selection(
        chain_payload, "track_b1_processed_export_stage_entry", "historical_logistic_structural"
    )
    if tuple(b1_selection.get("feature_names", ())) != cpu.TRACK_B1_FEATURES:
        raise ValueError(f"{chain}: frozen B1 feature schema changed")
    b1_logistic = cpu._pipeline(float(b1_selection["selected_C"]), chain_seed + 21)
    b1_logistic.fit(
        history_entries.loc[:, list(cpu.TRACK_B1_FEATURES)],
        history_entries["z"].to_numpy(np.int8),
    )

    positive_ids = set(history_entries.loc[history_entries["z"].eq(1), "entry_id"].astype(str))
    history_b2 = history_b.loc[history_b["entry_id"].isin(positive_ids)].copy()
    b2_features = cpu._track_b2_features(history_b2)
    b2_selection = _frozen_selection(
        chain_payload,
        "track_b2_conditional_destination_ranking",
        "historical_logistic_demand_gravity",
    )
    if tuple(b2_selection.get("feature_names", ())) != cpu.TRACK_B2_FEATURES:
        raise ValueError(f"{chain}: frozen B2 feature schema changed")
    b2_logistic = cpu._pipeline(float(b2_selection["selected_C"]), chain_seed + 31)
    b2_logistic.fit(
        b2_features.loc[:, list(cpu.TRACK_B2_FEATURES)], history_b2["y"].to_numpy(np.int8)
    )

    return FrozenCpu(
        a_size=_raw_from_artifact(chain_payload, "track_a_destination_extension", "size"),
        a_gravity=_raw_from_artifact(chain_payload, "track_a_destination_extension", "gravity"),
        a_logistic=a_logistic,
        b1_capacity=_raw_from_artifact(
            chain_payload, "track_b1_processed_export_stage_entry", "upstream_capacity"
        ),
        b1_logistic=b1_logistic,
        b2_demand=_raw_from_artifact(
            chain_payload, "track_b2_conditional_destination_ranking", "processed_importer_demand"
        ),
        b2_gravity=_raw_from_artifact(
            chain_payload, "track_b2_conditional_destination_ranking", "gravity"
        ),
        b2_logistic=b2_logistic,
        selections={"track_a": dict(a_selection), "track_b1": dict(b1_selection), "track_b2": dict(b2_selection)},
    )


def _align_values(
    source: pd.DataFrame,
    target: pd.DataFrame,
    values: np.ndarray,
    *,
    keys: Sequence[str],
    role: str,
) -> np.ndarray:
    if len(source) != len(values):
        raise ValueError(f"{role}: source rows and score rows differ")
    left = source.loc[:, list(keys)].copy()
    left["_value"] = np.asarray(values, dtype=float)
    if left.duplicated(list(keys)).any():
        raise ValueError(f"{role}: duplicate source score keys")
    aligned = target.loc[:, list(keys)].merge(left, on=list(keys), how="left", validate="one_to_one", sort=False)
    score = aligned["_value"].to_numpy(float)
    if len(aligned) != len(target) or not np.isfinite(score).all():
        raise ValueError(f"{role}: incomplete or non-finite score alignment")
    return score


def _cpu_scores(tables: FormalTables, frozen: FrozenCpu) -> dict[str, dict[str, np.ndarray]]:
    a_features = cpu._track_a_features(tables.track_a)
    a_scores = {
        "size": frozen.a_size.predict(a_features["size"]),
        "gravity": frozen.a_gravity.predict(a_features["log_gravity"]),
        "historical_logistic_size_gravity": frozen.a_logistic.predict_proba(
            a_features.loc[:, list(cpu.TRACK_A_FEATURES)]
        )[:, 1],
    }

    derived_entries = cpu._derive_entry_table(tables.track_b_lanes)
    b1_scores_derived = {
        "upstream_capacity": frozen.b1_capacity.predict(derived_entries["log_upstream_capacity"]),
        "historical_logistic_structural": frozen.b1_logistic.predict_proba(
            derived_entries.loc[:, list(cpu.TRACK_B1_FEATURES)]
        )[:, 1],
    }
    b1_scores = {
        name: _align_values(
            derived_entries,
            tables.track_b1,
            score,
            keys=("i_iso", "stage"),
            role=f"CPU/B1/{name}",
        )
        for name, score in b1_scores_derived.items()
    }

    b2_features = cpu._track_b2_features(tables.track_b_lanes)
    b2_all = {
        "processed_importer_demand": frozen.b2_demand.predict(b2_features["log_importer_demand"]),
        "gravity": frozen.b2_gravity.predict(b2_features["log_gravity"]),
        "historical_logistic_demand_gravity": frozen.b2_logistic.predict_proba(
            b2_features.loc[:, list(cpu.TRACK_B2_FEATURES)]
        )[:, 1],
    }
    b2_scores = {
        name: _align_values(
            tables.track_b_lanes,
            tables.track_b2,
            score,
            keys=("i_iso", "j_iso", "stage"),
            role=f"CPU/B2/{name}",
        )
        for name, score in b2_all.items()
    }
    return {"a": a_scores, "b1": b1_scores, "b2": b2_scores}


def _deterministic_order(frame: pd.DataFrame, score: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("ranking score is misaligned or non-finite")
    rank = pd.DataFrame({"_score": values, "_position": np.arange(len(frame), dtype=int)})
    for key in keys:
        rank[key] = frame[key].astype(str).to_numpy()
    ordered = rank.sort_values(
        ["_score", *keys],
        ascending=[False, *([True] * len(keys))],
        kind="mergesort",
    )
    return ordered["_position"].to_numpy(int)


def _global_point(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    label: str,
    value: str,
    budget: int,
    keys: Sequence[str],
) -> dict[str, Any]:
    y = frame[label].to_numpy(int)
    lateval = frame[value].to_numpy(float)
    model_order = _deterministic_order(frame, score, keys)
    oracle_order = _deterministic_order(frame, lateval, keys)
    effective = min(int(budget), len(frame))
    model_selected = model_order[:effective]
    oracle_selected = oracle_order[:effective]
    total_value = float(lateval.sum())
    model_value = float(lateval[model_selected].sum())
    oracle_value = float(lateval[oracle_selected].sum())
    model_capture = float(model_value / total_value) if total_value else None
    oracle_capture = float(oracle_value / total_value) if total_value else None
    return {
        "requested_k": int(budget),
        "effective_k": effective,
        "total_candidates": int(len(frame)),
        "total_positive_units": int(y.sum()),
        "total_observed_late_value_kusd": total_value,
        "model_hits": int(y[model_selected].sum()),
        "oracle_hits": int(y[oracle_selected].sum()),
        "model_selected_observed_late_value_kusd": model_value,
        "oracle_selected_observed_late_value_kusd": oracle_value,
        "model_value_capture": model_capture,
        "oracle_value_capture": oracle_capture,
        "oracle_gap_value_capture": (
            float(oracle_capture - model_capture)
            if oracle_capture is not None and model_capture is not None
            else None
        ),
        "headroom_kusd": float(oracle_value - model_value),
        "fraction_of_budget_oracle_value_recovered": (
            float(model_value / oracle_value) if oracle_value > 0 else None
        ),
    }


def _group_statistics(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    group: str,
    label: str,
    value: str,
    budget: int,
    keys: Sequence[str],
) -> pd.DataFrame:
    work = frame.copy()
    work["_score"] = np.asarray(score, dtype=float)
    rows: list[dict[str, Any]] = []
    for group_id, subset in work.groupby(group, sort=True, observed=True):
        positions = subset.index.to_numpy(int)
        local = subset.reset_index(drop=True)
        model_order = _deterministic_order(local, local["_score"].to_numpy(float), keys)
        oracle_order = _deterministic_order(local, local[value].to_numpy(float), keys)
        effective = min(int(budget), len(local))
        model_selected = model_order[:effective]
        oracle_selected = oracle_order[:effective]
        y = local[label].to_numpy(int)
        lateval = local[value].to_numpy(float)
        total_value = float(lateval.sum())
        positives = int(y.sum())
        model_value = float(lateval[model_selected].sum())
        oracle_value = float(lateval[oracle_selected].sum())
        model_hits = int(y[model_selected].sum())
        oracle_hits = int(y[oracle_selected].sum())
        rows.append(
            {
                "group_id": str(group_id),
                "candidate_units": int(len(positions)),
                "positive_units": positives,
                "total_value": total_value,
                "model_selected_value": model_value,
                "oracle_selected_value": oracle_value,
                "model_value_capture": (model_value / total_value if total_value else np.nan),
                "oracle_value_capture": (oracle_value / total_value if total_value else np.nan),
                "model_recall": (model_hits / positives if positives else np.nan),
                "oracle_recall": (oracle_hits / positives if positives else np.nan),
            }
        )
    return pd.DataFrame(rows)


def _group_point(stats: pd.DataFrame, *, budget: int, unit: str) -> dict[str, Any]:
    positive = stats.loc[stats["total_value"].gt(0)].copy()
    total_value = float(positive["total_value"].sum())
    model_value = float(positive["model_selected_value"].sum())
    oracle_value = float(positive["oracle_selected_value"].sum())
    return {
        "requested_k_per_group": int(budget),
        "group_unit": unit,
        "groups": int(len(stats)),
        "groups_with_positive_value": int(len(positive)),
        "total_observed_late_value_kusd": total_value,
        "model_selected_observed_late_value_kusd": model_value,
        "oracle_selected_observed_late_value_kusd": oracle_value,
        "model_macro_value_capture": (
            float(positive["model_value_capture"].mean()) if len(positive) else None
        ),
        "oracle_macro_value_capture": (
            float(positive["oracle_value_capture"].mean()) if len(positive) else None
        ),
        "oracle_gap_macro_value_capture": (
            float((positive["oracle_value_capture"] - positive["model_value_capture"]).mean())
            if len(positive)
            else None
        ),
        "model_pooled_value_capture": (float(model_value / total_value) if total_value else None),
        "oracle_pooled_value_capture": (float(oracle_value / total_value) if total_value else None),
        "oracle_gap_pooled_value_capture": (
            float((oracle_value - model_value) / total_value) if total_value else None
        ),
        "model_macro_recall": (
            float(positive["model_recall"].mean()) if len(positive) else None
        ),
        "oracle_macro_recall": (
            float(positive["oracle_recall"].mean()) if len(positive) else None
        ),
        "headroom_kusd": float(oracle_value - model_value),
        "fraction_of_group_oracle_value_recovered": (
            float(model_value / oracle_value) if oracle_value > 0 else None
        ),
    }


def _weighted_topk_value(
    order: np.ndarray,
    row_cluster_codes: np.ndarray,
    cluster_counts: np.ndarray,
    lateval: np.ndarray,
    budget: int,
) -> float:
    multiplicity = cluster_counts[row_cluster_codes[order]].astype(np.int64, copy=False)
    cumulative = np.cumsum(multiplicity)
    before = cumulative - multiplicity
    take = np.minimum(multiplicity, np.maximum(0, int(budget) - before))
    return float(np.dot(take.astype(float), lateval[order]))


def _bootstrap_global(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    cluster: str,
    value: str,
    budget: int,
    keys: Sequence[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "paired_nonparametric_cluster_bootstrap",
        "cluster_unit": cluster,
        "draws": int(draws),
        "seed": int(seed),
    }
    if draws <= 0:
        return result
    labels, uniques = pd.factorize(frame[cluster].astype(str), sort=True)
    n_clusters = len(uniques)
    lateval = frame[value].to_numpy(float)
    totals = np.bincount(labels, weights=lateval, minlength=n_clusters)
    model_order = _deterministic_order(frame, score, keys)
    oracle_order = _deterministic_order(frame, lateval, keys)
    rng = np.random.default_rng(seed)
    model_captures: list[float] = []
    oracle_captures: list[float] = []
    gaps: list[float] = []
    headrooms: list[float] = []
    for _ in range(draws):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(sampled, minlength=n_clusters)
        total = float(np.dot(counts, totals))
        if total <= 0:
            continue
        model_value = _weighted_topk_value(model_order, labels, counts, lateval, budget)
        oracle_value = _weighted_topk_value(oracle_order, labels, counts, lateval, budget)
        model_capture = model_value / total
        oracle_capture = oracle_value / total
        model_captures.append(model_capture)
        oracle_captures.append(oracle_capture)
        gaps.append(oracle_capture - model_capture)
        headrooms.append(oracle_value - model_value)
    result.update(
        {
            "n_clusters": int(n_clusters),
            "effective_draws": int(len(gaps)),
            "model_value_capture_ci95": _ci(model_captures),
            "oracle_value_capture_ci95": _ci(oracle_captures),
            "oracle_gap_value_capture_ci95": _ci(gaps),
            "headroom_kusd_ci95": _ci(headrooms),
        }
    )
    return result


def _bootstrap_groups(
    stats: pd.DataFrame,
    *,
    unit: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "paired_nonparametric_cluster_bootstrap",
        "cluster_unit": unit,
        "draws": int(draws),
        "seed": int(seed),
        "n_clusters": int(len(stats)),
    }
    if draws <= 0:
        return result
    rng = np.random.default_rng(seed)
    macro_model: list[float] = []
    macro_oracle: list[float] = []
    macro_gap: list[float] = []
    pooled_model: list[float] = []
    pooled_oracle: list[float] = []
    pooled_gap: list[float] = []
    headrooms: list[float] = []
    for _ in range(draws):
        sampled = stats.iloc[rng.integers(0, len(stats), size=len(stats))]
        positive = sampled.loc[sampled["total_value"].gt(0)]
        if not len(positive):
            continue
        total = float(positive["total_value"].sum())
        model_value = float(positive["model_selected_value"].sum())
        oracle_value = float(positive["oracle_selected_value"].sum())
        mm = float(positive["model_value_capture"].mean())
        mo = float(positive["oracle_value_capture"].mean())
        pm = model_value / total
        po = oracle_value / total
        macro_model.append(mm)
        macro_oracle.append(mo)
        macro_gap.append(mo - mm)
        pooled_model.append(pm)
        pooled_oracle.append(po)
        pooled_gap.append(po - pm)
        headrooms.append(oracle_value - model_value)
    result.update(
        {
            "effective_draws": int(len(macro_gap)),
            "model_macro_value_capture_ci95": _ci(macro_model),
            "oracle_macro_value_capture_ci95": _ci(macro_oracle),
            "oracle_gap_macro_value_capture_ci95": _ci(macro_gap),
            "model_pooled_value_capture_ci95": _ci(pooled_model),
            "oracle_pooled_value_capture_ci95": _ci(pooled_oracle),
            "oracle_gap_pooled_value_capture_ci95": _ci(pooled_gap),
            "headroom_kusd_ci95": _ci(headrooms),
        }
    )
    return result


def _track_a_diagnostic(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    stats = _group_statistics(
        frame,
        score,
        group="i_iso",
        label="y",
        value="lateval",
        budget=TRACK_A_EXPORTER_K,
        keys=("stage", "j_iso"),
    )
    return {
        "global_budget": {
            "point": _global_point(
                frame,
                score,
                label="y",
                value="lateval",
                budget=TRACK_A_GLOBAL_K,
                keys=("i_iso", "stage", "j_iso"),
            ),
            "uncertainty": _bootstrap_global(
                frame,
                score,
                cluster="i_iso",
                value="lateval",
                budget=TRACK_A_GLOBAL_K,
                keys=("i_iso", "stage", "j_iso"),
                draws=draws,
                seed=seed,
            ),
        },
        "per_exporter_shortlist": {
            "point": _group_point(stats, budget=TRACK_A_EXPORTER_K, unit="exporter"),
            "uncertainty": _bootstrap_groups(stats, unit="exporter", draws=draws, seed=seed + 1),
        },
    }


def _track_b1_diagnostic(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    stats = _group_statistics(
        frame,
        score,
        group="i_iso",
        label="z",
        value="entry_lateval",
        budget=TRACK_B1_EXPORTER_K,
        keys=("stage",),
    )
    return {
        "global_entry_budget": {
            "point": _global_point(
                frame,
                score,
                label="z",
                value="entry_lateval",
                budget=TRACK_B1_GLOBAL_K,
                keys=("i_iso", "stage"),
            ),
            "uncertainty": _bootstrap_global(
                frame,
                score,
                cluster="i_iso",
                value="entry_lateval",
                budget=TRACK_B1_GLOBAL_K,
                keys=("i_iso", "stage"),
                draws=draws,
                seed=seed,
            ),
        },
        "per_exporter_entry_shortlist": {
            "point": _group_point(stats, budget=TRACK_B1_EXPORTER_K, unit="exporter"),
            "uncertainty": _bootstrap_groups(stats, unit="exporter", draws=draws, seed=seed + 1),
        },
    }


def _track_b2_diagnostic(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    stats = _group_statistics(
        frame,
        score,
        group="entry_id",
        label="y",
        value="lateval",
        budget=TRACK_B2_ENTRY_K,
        keys=("j_iso",),
    )
    return {
        "per_positive_entry": {
            "point": _group_point(stats, budget=TRACK_B2_ENTRY_K, unit="exporter_stage_entry"),
            "uncertainty": _bootstrap_groups(
                stats, unit="exporter_stage_entry", draws=draws, seed=seed
            ),
        }
    }


def _artifact_metric(cpu_chain: Mapping[str, Any], track: str, model: str) -> float:
    metrics = cpu_chain[track]["models"][model]["metrics"]
    if track == "track_a_destination_extension":
        return float(metrics["budgets"][f"k_{TRACK_A_GLOBAL_K}"]["value_capture"])
    if track == "track_b1_processed_export_stage_entry":
        return float(metrics["budgets"][f"k_{TRACK_B1_GLOBAL_K}"]["value_capture"])
    return float(metrics["at_k"][f"k_{TRACK_B2_ENTRY_K}"]["macro_value_capture"])


def _diagnostic_primary_value(track: str, diagnostic: Mapping[str, Any]) -> float:
    if track == "a":
        return float(diagnostic["global_budget"]["point"]["model_value_capture"])
    if track == "b1":
        return float(diagnostic["global_entry_budget"]["point"]["model_value_capture"])
    return float(diagnostic["per_positive_entry"]["point"]["model_macro_value_capture"])


def _validate_cpu_reproduction(
    cpu_chain: Mapping[str, Any],
    track: str,
    model: str,
    diagnostic: Mapping[str, Any],
) -> dict[str, float]:
    track_name = {
        "a": "track_a_destination_extension",
        "b1": "track_b1_processed_export_stage_entry",
        "b2": "track_b2_conditional_destination_ranking",
    }[track]
    saved = _artifact_metric(cpu_chain, track_name, model)
    observed = _diagnostic_primary_value(track, diagnostic)
    delta = observed - saved
    # Explicit canonical identity tie-breaking can differ from the older CPU
    # helper only for exactly tied raw scores.  The frozen supervised scores and
    # all non-tied rankings must reproduce exactly; any material difference is
    # rejected for every model.
    if abs(delta) > 1e-10:
        raise ValueError(
            f"CPU reproduction mismatch {track}/{model}: saved={saved}, observed={observed}"
        )
    return {"saved_value_capture": saved, "recomputed_value_capture": observed, "delta": delta}


def _gpu_records(summary: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    if summary.get("schema_version") != GPU_SUMMARY_SCHEMA or summary.get("protocol") != PROTOCOL:
        raise ValueError("GPU summary is stale or not strict rolling")
    records: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in summary.get("records", []):
        key = (str(record["chain"]), str(record["track"]), str(record["family"]))
        if key in records:
            raise ValueError(f"duplicate GPU summary record {key}")
        records[key] = record
    expected = {(chain, track, family) for chain in CHAINS for track in ("a", "b1", "b2") for family in GPU_FAMILIES}
    if set(records) != expected:
        raise ValueError(f"GPU summary record set differs: missing={sorted(expected - set(records))}")
    return records


def _validate_gpu_candidate_binding(
    record: Mapping[str, Any],
    tables: FormalTables,
    cpu_chain: Mapping[str, Any],
    track: str,
    *,
    role: str,
) -> None:
    """Cross-bind one GPU summary row to current target and verified history bytes."""
    target_audit = tables.audits[
        "track_a" if track == "a" else "track_b_lane_pool"
    ]
    history_audit = cpu_chain["protocol_audit"][
        "history_track_a" if track == "a" else "history_track_b"
    ]
    expected = {
        "target_candidate_role": target_audit["path"],
        "target_candidate_sha256": target_audit["sha256"],
        "target_rows": int(target_audit["rows"]),
        "target_positive_lanes": int(target_audit["positive_lanes"]),
        "history_candidate_role": history_audit["path"],
        "history_candidate_sha256": history_audit["sha256"],
    }
    for field, wanted in expected.items():
        if record.get(field) != wanted:
            raise ValueError(
                f"{role}: GPU summary {field} differs from the current/verified candidate receipt"
            )


def _load_gpu_lane_scores(
    record: Mapping[str, Any],
    target_lanes: pd.DataFrame,
    *,
    role: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    score_path = ROOT / str(record["score_artifact_role"])
    selection_path = ROOT / str(record["selection_artifact_role"])
    metric_path = ROOT / str(record["metric_artifact_role"])
    for path, expected, name in (
        (score_path, record["score_artifact_sha256"], "score"),
        (selection_path, record["selection_sha256"], "selection"),
        (metric_path, record["metric_artifact_sha256"], "metric"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{role}: stale {name} artifact {path}")
    scores = pd.read_csv(score_path, dtype={"i_iso": str, "j_iso": str, "stage": str})
    keys = ("i_iso", "j_iso", "stage")
    _require_columns(scores, [*keys, "selection_sha256", "protocol"], role)
    if scores.duplicated(list(keys)).any():
        raise ValueError(f"{role}: duplicate score keys")
    if set(scores["selection_sha256"].astype(str)) != {str(record["selection_sha256"])}:
        raise ValueError(f"{role}: score rows are not bound to the frozen selection")
    if set(scores["protocol"].astype(str)) != {PROTOCOL}:
        raise ValueError(f"{role}: score rows use the wrong protocol")
    model = str(record["selected_model"])
    expected_columns = {seed: f"score_{model}_s{seed}" for seed in GPU_SEEDS}
    _require_columns(scores, list(expected_columns.values()), role)
    aligned = target_lanes.loc[:, list(keys)].merge(
        scores.loc[:, [*keys, *expected_columns.values()]],
        on=list(keys),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(target_lanes):
        raise ValueError(f"{role}: score coverage changed row count")
    output: dict[int, np.ndarray] = {}
    for seed, column in expected_columns.items():
        values = aligned[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{role}: incomplete seed-{seed} score coverage")
        output[seed] = values
    audit = {
        "score_path": score_path.relative_to(ROOT).as_posix(),
        "score_sha256": _sha256(score_path),
        "selection_path": selection_path.relative_to(ROOT).as_posix(),
        "selection_sha256": str(record["selection_sha256"]),
        "metric_path": metric_path.relative_to(ROOT).as_posix(),
        "metric_sha256": _sha256(metric_path),
        "selected_model": model,
        "selected_hyperparameters": record["selected_hyperparameters"],
        "history_holdout_selection_metric_mean": record["history_holdout_selection_metric_mean"],
        "rows": int(len(scores)),
        "seeds": list(GPU_SEEDS),
    }
    return output, audit


def _gpu_task_scores(
    record: Mapping[str, Any],
    tables: FormalTables,
    track: str,
    *,
    role: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    lane_target = tables.track_a if track == "a" else tables.track_b_lanes
    lane_scores, audit = _load_gpu_lane_scores(record, lane_target, role=role)
    if track == "a":
        return lane_scores, audit
    if track == "b1":
        output: dict[int, np.ndarray] = {}
        for seed, score in lane_scores.items():
            lane = tables.track_b_lanes.loc[:, ["i_iso", "stage"]].copy()
            lane["_score"] = score
            entry = lane.groupby(["i_iso", "stage"], sort=True, as_index=False)["_score"].max()
            output[seed] = _align_values(
                entry,
                tables.track_b1,
                entry["_score"].to_numpy(float),
                keys=("i_iso", "stage"),
                role=f"{role}/entry",
            )
        return output, audit
    output = {
        seed: _align_values(
            tables.track_b_lanes,
            tables.track_b2,
            score,
            keys=("i_iso", "j_iso", "stage"),
            role=f"{role}/conditional",
        )
        for seed, score in lane_scores.items()
    }
    return output, audit


def _gpu_saved_value(record: Mapping[str, Any], track: str, seed: int) -> float | None:
    row = next(item for item in record["metrics_by_seed"] if int(item["seed"]) == seed)
    # The formal GPU runner did not emit B1 value-at-budget.  This tool derives
    # it deterministically from the frozen lane scores and formal entry table;
    # A and B2 do have an existing value metric that must reproduce exactly.
    key = {"a": "value_capture_at_500", "b1": None, "b2": "conditional_value_capture_at_3"}[track]
    return None if key is None else float(row[key])


def _diagnose(track: str, frame: pd.DataFrame, score: np.ndarray, *, draws: int, seed: int) -> dict[str, Any]:
    if track == "a":
        return _track_a_diagnostic(frame, score, draws=draws, seed=seed)
    if track == "b1":
        return _track_b1_diagnostic(frame, score, draws=draws, seed=seed)
    return _track_b2_diagnostic(frame, score, draws=draws, seed=seed)


def _track_frame(tables: FormalTables, track: str) -> pd.DataFrame:
    return {"a": tables.track_a, "b1": tables.track_b1, "b2": tables.track_b2}[track]


def _accounting(tables_by_chain: Mapping[str, FormalTables]) -> dict[str, Any]:
    per_chain: dict[str, Any] = {}
    totals = {"track_a": 0.0, "track_b1": 0.0, "track_b2_nested": 0.0}
    for chain in CHAINS:
        tables = tables_by_chain[chain]
        a_value = float(tables.track_a["lateval"].sum())
        b1_value = float(tables.track_b1["entry_lateval"].sum())
        b2_value = float(tables.track_b2["lateval"].sum())
        if not math.isclose(b1_value, b2_value, rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"{chain}: B1/B2 accounting is not nested")
        per_chain[chain] = {
            "track_a_observed_late_value_kusd": a_value,
            "track_b1_observed_late_value_kusd": b1_value,
            "track_b2_nested_same_dollars_kusd": b2_value,
            "unique_project_observed_late_value_kusd": a_value + b1_value,
            "forbidden_naive_a_plus_b1_plus_b2_kusd": a_value + b1_value + b2_value,
            "b2_is_nested_in_b1": True,
        }
        totals["track_a"] += a_value
        totals["track_b1"] += b1_value
        totals["track_b2_nested"] += b2_value
    if not math.isclose(totals["track_b1"], totals["track_b2_nested"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("aggregate B1/B2 nested totals differ")
    return {
        "policy": "Track B2 re-ranks the same positive-entry dollars counted once in B1; never add B1 and B2.",
        "per_chain": per_chain,
        "totals": {
            "track_a_observed_late_value_kusd": totals["track_a"],
            "track_b1_observed_late_value_kusd": totals["track_b1"],
            "track_b2_nested_same_dollars_kusd": totals["track_b2_nested"],
            "unique_project_observed_late_value_kusd": totals["track_a"] + totals["track_b1"],
            "forbidden_naive_a_plus_b1_plus_b2_kusd": sum(totals.values()),
            "b2_excluded_from_unique_sum": True,
        },
    }


def build_result(
    *,
    data_dir: Path,
    cpu_path: Path,
    gpu_path: Path,
    gpu_root: Path,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap < 0:
        raise ValueError("bootstrap draws must be non-negative")
    cpu.verify_existing_output(cpu_path)
    cpu_payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu_payload = json.loads(gpu_path.read_text(encoding="utf-8"))
    records = _gpu_records(gpu_payload)
    manifest_path = gpu_root / "frozen_manifest.json"
    if not manifest_path.is_file() or _sha256(manifest_path) != gpu_payload.get("manifest_sha256"):
        raise ValueError("GPU frozen manifest hash does not match the formal summary")

    # Freeze all CPU models from the historical artifact before opening any
    # formal main table in this script.
    base_cpu_seed = int(cpu_payload["protocol"]["random_seed"])
    frozen_cpu: dict[str, FrozenCpu] = {}
    for index, chain in enumerate(CHAINS):
        frozen_cpu[chain] = _fit_cpu_from_artifact(
            data_dir,
            chain,
            cpu_payload["chains"][chain],
            chain_seed=base_cpu_seed + index * 1000,
        )

    tables_by_chain = {chain: _read_formal_tables(data_dir, chain) for chain in CHAINS}
    chains: dict[str, Any] = {}
    score_input_audits: list[dict[str, Any]] = []
    for chain in CHAINS:
        tables = tables_by_chain[chain]
        cpu_scores = _cpu_scores(tables, frozen_cpu[chain])
        chain_result: dict[str, Any] = {
            "input_audit": tables.audits,
            "tracks": {},
        }
        for track in ("a", "b1", "b2"):
            frame = _track_frame(tables, track)
            track_result: dict[str, Any] = {
                "task": {
                    "a": "destination_extension",
                    "b1": "eligible_market_processed_export_stage_entry",
                    "b2": "conditional_destination_formation",
                }[track],
                "oracle_interpretation": "outcome-ranked, same-cohort same-budget descriptive upper bound",
                "cpu_models": {},
                "gpu_families": {},
            }
            cpu_track_name = {
                "a": "track_a_destination_extension",
                "b1": "track_b1_processed_export_stage_entry",
                "b2": "track_b2_conditional_destination_ranking",
            }[track]
            for model, score_values in cpu_scores[track].items():
                diagnostic_seed = _stable_seed(seed, chain, track, "cpu", model)
                diagnostic = _diagnose(
                    track, frame, score_values, draws=bootstrap, seed=diagnostic_seed
                )
                track_result["cpu_models"][model] = {
                    "selection_source": "verified historical CPU rolling artifact",
                    "post_hoc_main_champion_selection": False,
                    "diagnostic": diagnostic,
                    "formal_artifact_reproduction": _validate_cpu_reproduction(
                        cpu_payload["chains"][chain], track, model, diagnostic
                    ),
                    "frozen_model": cpu_payload["chains"][chain][cpu_track_name]["models"][model]["model"],
                }

            for family in GPU_FAMILIES:
                record = records[(chain, track, family)]
                _validate_gpu_candidate_binding(
                    record,
                    tables,
                    cpu_payload["chains"][chain],
                    track,
                    role=f"{chain}/{track}/{family}",
                )
                per_seed_scores, audit = _gpu_task_scores(
                    record, tables, track, role=f"{chain}/{track}/{family}"
                )
                score_input_audits.append({"chain": chain, "track": track, "family": family, **audit})
                per_seed: list[dict[str, Any]] = []
                for task_seed in GPU_SEEDS:
                    diagnostic_seed = _stable_seed(seed, chain, track, family, task_seed)
                    diagnostic = _diagnose(
                        track,
                        frame,
                        per_seed_scores[task_seed],
                        draws=bootstrap,
                        seed=diagnostic_seed,
                    )
                    observed = _diagnostic_primary_value(track, diagnostic)
                    saved = _gpu_saved_value(record, track, task_seed)
                    if saved is not None and abs(observed - saved) > 1e-10:
                        raise ValueError(
                            f"GPU reproduction mismatch {chain}/{track}/{family}/s{task_seed}: "
                            f"saved={saved}, observed={observed}"
                        )
                    per_seed.append(
                        {
                            "seed": task_seed,
                            "diagnostic": diagnostic,
                            "formal_artifact_reproduction": {
                                "status": (
                                    "reproduced_existing_formal_metric"
                                    if saved is not None
                                    else "new_deterministic_derivation_from_frozen_scores"
                                ),
                                "saved_value_capture": saved,
                                "recomputed_value_capture": observed,
                                "delta": (observed - saved if saved is not None else None),
                            },
                        }
                    )
                scalar_rows = []
                for item in per_seed:
                    diagnostic = item["diagnostic"]
                    if track == "a":
                        point = diagnostic["global_budget"]["point"]
                    elif track == "b1":
                        point = diagnostic["global_entry_budget"]["point"]
                    else:
                        point = diagnostic["per_positive_entry"]["point"]
                    scalar_rows.append(
                        {
                            "model_value_capture": _diagnostic_primary_value(track, diagnostic),
                            "oracle_value_capture": (
                                point["oracle_value_capture"]
                                if track in {"a", "b1"}
                                else point["oracle_macro_value_capture"]
                            ),
                            "oracle_gap_value_capture": (
                                point["oracle_gap_value_capture"]
                                if track in {"a", "b1"}
                                else point["oracle_gap_macro_value_capture"]
                            ),
                            "headroom_kusd": point["headroom_kusd"],
                        }
                    )
                track_result["gpu_families"][family] = {
                    "selected_model": audit["selected_model"],
                    "selected_hyperparameters": audit["selected_hyperparameters"],
                    "selection_source": "historical fold2 frozen manifest",
                    "post_hoc_main_champion_selection": False,
                    "raw_score_policy": "one diagnostic per seed; no raw-score averaging",
                    "score_audit": audit,
                    "per_seed": per_seed,
                    "summary_across_seeds": _mean_std(
                        scalar_rows,
                        (
                            "model_value_capture",
                            "oracle_value_capture",
                            "oracle_gap_value_capture",
                            "headroom_kusd",
                        ),
                    ),
                }
            chain_result["tracks"][track] = track_result
        chains[chain] = chain_result

    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": "2.1-dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "selection_window": "1998-2002 -> 2008-2012",
            "target_window": "2008-2012 -> 2018-2022",
            "target_labels_used_for_model_or_family_selection": False,
            "post_hoc_main_champion_selected": False,
            "cpu_policy": "use saved historical hyperparameters; refit preprocessing/model on full history only",
            "gpu_policy": "report both frozen families and all five seeds separately",
            "oracle_policy": "outcome-only budget-matched diagnostic; never a deployable model or selection candidate",
            "budgets": {
                "track_a_global": TRACK_A_GLOBAL_K,
                "track_a_per_exporter": TRACK_A_EXPORTER_K,
                "track_b1_global_entries": TRACK_B1_GLOBAL_K,
                "track_b1_per_exporter_entries": TRACK_B1_EXPORTER_K,
                "track_b2_per_positive_entry": TRACK_B2_ENTRY_K,
            },
            "uncertainty": {
                "method": "paired nonparametric cluster bootstrap",
                "draws": int(bootstrap),
                "base_seed": int(seed),
                "track_a": "exporter",
                "track_b1": "exporter",
                "track_b2": "exporter_stage_entry",
            },
            "dollar_accounting": "A and B1 are distinct task pools; B2 is nested in B1 and is excluded from sums.",
        },
        "inputs": {
            "cpu_rolling": {
                "path": cpu_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(cpu_path),
                "schema_version": cpu_payload["schema_version"],
            },
            "gpu_summary": {
                "path": gpu_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(gpu_path),
                "schema_version": gpu_payload["schema_version"],
            },
            "gpu_frozen_manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(manifest_path),
            },
            "gpu_score_artifacts": score_input_audits,
        },
        "accounting": _accounting(tables_by_chain),
        "chains": chains,
    }
    return _jsonable(result)


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chain in CHAINS:
        for track in ("a", "b1", "b2"):
            payload = result["chains"][chain]["tracks"][track]
            for model in sorted(payload["cpu_models"]):
                record = payload["cpu_models"][model]
                rows.append(_csv_row(chain, track, "cpu", model, None, record["diagnostic"]))
            for family in sorted(payload["gpu_families"]):
                family_record = payload["gpu_families"][family]
                for seed_record in family_record["per_seed"]:
                    rows.append(
                        _csv_row(
                            chain,
                            track,
                            family,
                            family_record["selected_model"],
                            seed_record["seed"],
                            seed_record["diagnostic"],
                        )
                    )
    return rows


def _csv_row(
    chain: str,
    track: str,
    source: str,
    model: str,
    seed: int | None,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    if track == "a":
        primary = diagnostic["global_budget"]["point"]
        group = diagnostic["per_exporter_shortlist"]["point"]
        budget = primary["requested_k"]
        model_capture = primary["model_value_capture"]
        oracle_capture = primary["oracle_value_capture"]
        gap = primary["oracle_gap_value_capture"]
    elif track == "b1":
        primary = diagnostic["global_entry_budget"]["point"]
        group = diagnostic["per_exporter_entry_shortlist"]["point"]
        budget = primary["requested_k"]
        model_capture = primary["model_value_capture"]
        oracle_capture = primary["oracle_value_capture"]
        gap = primary["oracle_gap_value_capture"]
    else:
        primary = diagnostic["per_positive_entry"]["point"]
        group = primary
        budget = primary["requested_k_per_group"]
        model_capture = primary["model_macro_value_capture"]
        oracle_capture = primary["oracle_macro_value_capture"]
        gap = primary["oracle_gap_macro_value_capture"]
    return {
        "chain": chain,
        "track": track,
        "source": source,
        "model": model,
        "seed": seed,
        "reporting_budget": budget,
        "model_value_capture": model_capture,
        "oracle_value_capture": oracle_capture,
        "oracle_gap_value_capture": gap,
        "headroom_kusd": primary["headroom_kusd"],
        "total_observed_late_value_kusd": primary["total_observed_late_value_kusd"],
        "group_model_macro_value_capture": group.get("model_macro_value_capture"),
        "group_oracle_macro_value_capture": group.get("oracle_macro_value_capture"),
        "group_oracle_gap_macro_value_capture": group.get("oracle_gap_macro_value_capture"),
        "group_model_pooled_value_capture": group.get("model_pooled_value_capture"),
        "group_oracle_pooled_value_capture": group.get("oracle_pooled_value_capture"),
    }


def render_json(result: Mapping[str, Any]) -> bytes:
    return (json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def render_csv(result: Mapping[str, Any]) -> bytes:
    rows = _csv_rows(result)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def write_outputs(result: Mapping[str, Any], json_out: Path, csv_out: Path) -> None:
    _atomic_write(json_out, render_json(result))
    _atomic_write(csv_out, render_csv(result))


def _verify_generator_hash(payload: Mapping[str, Any], path: Path) -> None:
    recorded = str(payload.get("runtime", {}).get("script_sha256", ""))
    current = _sha256(Path(__file__).resolve())
    if recorded != current:
        raise ValueError(
            f"{path}: generator hash is stale: recorded={recorded!r}, current={current!r}"
        )


def verify_existing_output(
    path: Path = DEFAULT_JSON,
    csv_path: Path | None = DEFAULT_CSV,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: stale schema")
    _verify_generator_hash(payload, path)
    protocol = payload.get("protocol", {})
    if protocol.get("target_labels_used_for_model_or_family_selection") is not False:
        raise ValueError(f"{path}: target labels influenced selection")
    if protocol.get("post_hoc_main_champion_selected") is not False:
        raise ValueError(f"{path}: post-hoc main champion was selected")
    checked = 0
    core_sources: dict[str, Path] = {}
    for role in ("cpu_rolling", "gpu_summary", "gpu_frozen_manifest"):
        record = payload.get("inputs", {}).get(role, {})
        source = ROOT / str(record.get("path", ""))
        if not source.is_file() or _sha256(source) != record.get("sha256"):
            raise ValueError(f"{path}: stale input {role}")
        core_sources[role] = source
        checked += 1
    # A matching CPU JSON is insufficient if one of the historical files it
    # binds has changed.  Re-run its own 24-input provenance verifier.
    cpu.verify_existing_output(core_sources["cpu_rolling"])
    score_audits = payload.get("inputs", {}).get("gpu_score_artifacts", [])
    if len(score_audits) != len(CHAINS) * 3 * len(GPU_FAMILIES):
        raise ValueError(f"{path}: incomplete GPU score audit inventory")
    for record in score_audits:
        for prefix in ("score", "selection", "metric"):
            source = ROOT / str(record.get(f"{prefix}_path", ""))
            expected = record.get(f"{prefix}_sha256")
            if not source.is_file() or _sha256(source) != expected:
                raise ValueError(
                    f"{path}: stale GPU {prefix} artifact for "
                    f"{record.get('chain')}/{record.get('track')}/{record.get('family')}"
                )
            checked += 1
    for chain in CHAINS:
        audits = payload.get("chains", {}).get(chain, {}).get("input_audit", {})
        for role in ("track_a", "track_b_lane_pool", "track_b1", "track_b2"):
            record = audits.get(role, {})
            source = ROOT / str(record.get("path", ""))
            if not source.is_file() or _sha256(source) != record.get("sha256"):
                raise ValueError(f"{path}: stale candidate {chain}/{role}")
            checked += 1
        chain_accounting = payload["accounting"]["per_chain"][chain]
        if chain_accounting.get("b2_is_nested_in_b1") is not True:
            raise ValueError(f"{path}: B2 is not marked nested for {chain}")
        if not math.isclose(
            float(chain_accounting["track_b1_observed_late_value_kusd"]),
            float(chain_accounting["track_b2_nested_same_dollars_kusd"]),
            rel_tol=0,
            abs_tol=1e-7,
        ):
            raise ValueError(f"{path}: B1/B2 dollars differ for {chain}")
        for track in ("a", "b1", "b2"):
            gpu = payload["chains"][chain]["tracks"][track]["gpu_families"]
            if set(gpu) != set(GPU_FAMILIES):
                raise ValueError(f"{path}: a GPU family was dropped for {chain}/{track}")
            if any(item.get("post_hoc_main_champion_selection") is not False for item in gpu.values()):
                raise ValueError(f"{path}: GPU champion selection flag failed for {chain}/{track}")
    totals = payload["accounting"]["totals"]
    expected_unique = float(totals["track_a_observed_late_value_kusd"]) + float(
        totals["track_b1_observed_late_value_kusd"]
    )
    if not math.isclose(
        expected_unique,
        float(totals["unique_project_observed_late_value_kusd"]),
        rel_tol=0,
        abs_tol=1e-6,
    ) or totals.get("b2_excluded_from_unique_sum") is not True:
        raise ValueError(f"{path}: aggregate dollar accounting double-counts B2")
    if csv_path is not None:
        csv_path = Path(csv_path)
        if not csv_path.is_file() or csv_path.read_bytes() != render_csv(payload):
            raise ValueError(f"{csv_path}: stale or inconsistent with {path.name}")
    print(f"verified {checked} source hashes and all value-accounting invariants in {path}")
    return payload


def _self_test() -> None:
    frame = pd.DataFrame(
        {
            "i_iso": ["A"] * 4 + ["B"] * 4,
            "j_iso": ["W", "X", "Y", "Z"] * 2,
            "stage": ["s"] * 8,
            "entry_id": ["A|s"] * 4 + ["B|s"] * 4,
            "y": [1, 0, 0, 0, 1, 0, 0, 0],
            "lateval": [10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        }
    )
    # Miss A's positive outside top-3 and retrieve B's positive.
    score = np.asarray([0.1, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.6])
    diagnostic = _track_b2_diagnostic(frame, score, draws=20, seed=7)
    point = diagnostic["per_positive_entry"]["point"]
    if point["model_macro_value_capture"] != 0.5 or point["oracle_macro_value_capture"] != 1.0:
        raise AssertionError(point)
    print("v2_value_diagnostics self-test: OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cpu-json", type=Path, default=DEFAULT_CPU)
    parser.add_argument("--gpu-json", type=Path, default=DEFAULT_GPU)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        if not args.verify_output:
            return 0
    if args.verify_output:
        verify_existing_output(args.json_out.resolve(), args.csv_out.resolve())
        return 0
    result = build_result(
        data_dir=args.data_dir.resolve(),
        cpu_path=args.cpu_json.resolve(),
        gpu_path=args.gpu_json.resolve(),
        gpu_root=args.gpu_root.resolve(),
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_outputs(result, args.json_out.resolve(), args.csv_out.resolve())
    print(
        f"wrote {args.json_out.resolve()} and {args.csv_out.resolve()} "
        f"({len(CHAINS)} chains, bootstrap={args.bootstrap})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
