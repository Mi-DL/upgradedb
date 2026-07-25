#!/usr/bin/env python3
"""Score-only r5 robustness for the frozen UPGRADE-BENCH graph references.

This runner never trains, calibrates, or selects a model.  It consumes the 36
formal main-score CSV files already bound by ``v2_gpu_rolling_summary.json`` and
produces four strictly descriptive diagnostics:

* paired PyKEEN-global-graph minus NBFNet cluster-bootstrap intervals, using the
  same cluster multiplicities for both families;
* B1 pooling sensitivity around the normative raw maximum, with two label-free
  ECDF-normalized alternatives reported in full rather than selected;
* the complete declared multi-cutoff budget surface from the frozen scores; and
* a secondary B1-gate -> B2-destination composition diagnostic in which every
  realized entry missed by B1 contributes zero destination recall/value.

The fixed-six-chain intervals resample clusters inside each declared chain and
never resample chains.  They are finite-benchmark descriptive intervals, not
population inference over value chains.  The verifier recomputes every result
from the governed scores and fails closed on changed inputs or inventories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v2_score_robustness_r5.json"
CONFIG_SCHEMA = "upgrade-bench-v2/score-robustness-r5-config/1"
OUTPUT_SCHEMA = "upgrade-bench-v2/score-robustness-r5/1"
KEYS = ("i_iso", "j_iso", "stage")
ENTRY_KEYS = ("i_iso", "stage")
EXPECTED_TASKS = ("a", "b1", "b2")
EXPECTED_FAMILIES = ("kge", "nbfnet")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
PAIR_DIRECTION = "kge_minus_nbfnet"
PRIMARY_METRICS = {
    "a": "lane_average_precision",
    "b1": "entry_average_precision_official_raw_max",
    "b2": "positive_entry_macro_recall_at_3",
}
CLUSTER_UNITS = {"a": "exporter", "b1": "exporter", "b2": "exporter_stage"}
CSV_FIELDS = (
    "section",
    "aggregation_scope",
    "chain",
    "task",
    "family",
    "seed",
    "method",
    "budget_scope",
    "b1_budget",
    "b2_budget",
    "requested_k",
    "metric",
    "point",
    "lower_95",
    "upper_95",
    "std",
    "n",
    "cluster_unit",
    "bootstrap_seed",
    "note",
)


class R5Error(ValueError):
    """Raised when an r5 protocol or verification invariant is violated."""


def _fail(message: str) -> None:
    raise R5Error(message)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            _fail(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise R5Error(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise R5Error(f"path escapes repository root: {path}") from exc


def _resolve_role(role: str) -> Path:
    candidate = (ROOT / role).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise R5Error(f"configured role escapes repository root: {role}") from exc
    return candidate


def _require_exact_keys(value: Mapping[str, Any], expected: Iterable[str], role: str) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        _fail(
            f"{role} keys differ; missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    _require_exact_keys(
        config,
        {
            "schema_version",
            "chains",
            "tasks",
            "families",
            "family_labels",
            "seeds",
            "formal_summary",
            "frozen_manifest",
            "frozen_run_config",
            "metric_dependency",
            "candidate_templates",
            "score_template",
            "bootstrap",
            "budgets",
            "b1_pooling_methods",
            "two_stage",
            "eligibility_threshold_geometry",
            "output_json",
            "output_csv",
        },
        "r5 config",
    )
    if config["schema_version"] != CONFIG_SCHEMA:
        _fail("unexpected r5 config schema")
    if tuple(config["tasks"]) != EXPECTED_TASKS:
        _fail(f"tasks must be exactly {EXPECTED_TASKS}")
    if tuple(config["families"]) != EXPECTED_FAMILIES:
        _fail(f"families must be exactly {EXPECTED_FAMILIES}")
    if tuple(config["seeds"]) != EXPECTED_SEEDS:
        _fail(f"seeds must be exactly {EXPECTED_SEEDS}")
    chains = tuple(config["chains"])
    if len(chains) != 6 or len(set(chains)) != 6:
        _fail("r5 requires exactly six unique declared chains")
    if config["family_labels"] != {
        "kge": "pykeen_global_graph",
        "nbfnet": "nbfnet",
    }:
        _fail("family labels or paired direction changed")

    _require_exact_keys(config["candidate_templates"], EXPECTED_TASKS, "candidate templates")
    for role in (
        config["formal_summary"],
        config["frozen_manifest"],
        config["frozen_run_config"],
        config["metric_dependency"],
        config["output_json"],
        config["output_csv"],
    ):
        if not isinstance(role, str) or not role or Path(role).is_absolute() or ".." in Path(role).parts:
            _fail(f"noncanonical configured role: {role!r}")

    _require_exact_keys(
        config["bootstrap"],
        {"iterations", "confidence_level", "seed_namespace"},
        "bootstrap",
    )
    if int(config["bootstrap"]["iterations"]) < 100:
        _fail("paired bootstrap must request at least 100 draws")
    if float(config["bootstrap"]["confidence_level"]) != 0.95:
        _fail("r5 interval confidence level is locked to 0.95")
    if not str(config["bootstrap"]["seed_namespace"]).startswith("upgrade-bench-v2"):
        _fail("bootstrap seed namespace is not benchmark-scoped")

    expected_budgets = {
        "a_global": [50, 100, 250, 500, 1000],
        "a_per_exporter": [5, 10],
        "b1_global": [25, 50, 100, 250],
        "b2_per_positive_entry": [1, 3, 5],
    }
    if config["budgets"] != expected_budgets:
        _fail("budget inventory changed from the r5 declaration")

    methods = config["b1_pooling_methods"]
    if not isinstance(methods, list) or [item.get("name") for item in methods] != [
        "official_raw_max",
        "ecdf_mean",
        "ecdf_top3_mean",
    ]:
        _fail("B1 pooling inventory or declaration order changed")
    if [item.get("kind") for item in methods] != ["raw_max", "ecdf_mean", "ecdf_topk_mean"]:
        _fail("B1 pooling kinds changed")
    if methods[0].get("normative") is not True or any(
        item.get("normative") is not False for item in methods[1:]
    ):
        _fail("exactly official_raw_max must be normative")
    if methods[2].get("top_k") != 3:
        _fail("ECDF top-k pooling is locked to three lanes")

    two_stage = config["two_stage"]
    _require_exact_keys(
        two_stage,
        {
            "enabled",
            "b1_entry_budgets",
            "b2_destination_budgets",
            "b1_pooling",
            "missed_positive_entry_contribution",
            "primary_unit",
            "description",
        },
        "two-stage declaration",
    )
    if two_stage["enabled"] is not True:
        _fail("two-stage diagnostic must be explicitly enabled")
    if two_stage["b1_entry_budgets"] != expected_budgets["b1_global"]:
        _fail("two-stage B1 budgets differ from declared B1 budgets")
    if two_stage["b2_destination_budgets"] != expected_budgets["b2_per_positive_entry"]:
        _fail("two-stage B2 budgets differ from declared B2 budgets")
    if two_stage["b1_pooling"] != "official_raw_max":
        _fail("two-stage gate must use the official frozen B1 pooling")
    if float(two_stage["missed_positive_entry_contribution"]) != 0.0:
        _fail("missed positive entries must contribute zero in the two-stage diagnostic")

    threshold = config["eligibility_threshold_geometry"]
    _require_exact_keys(threshold, {"thresholds_kusd", "status", "reason"}, "threshold geometry")
    if threshold["thresholds_kusd"] != [50.0, 100.0, 250.0]:
        _fail("threshold geometry declaration changed")
    if threshold["status"] != "not_computed_from_fixed_scores":
        _fail("r5 must not approximate alternate eligibility cohorts from fixed scores")


class ReceiptBook:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def add(self, path: Path, role: str, expected_sha256: str | None = None) -> str:
        if not path.is_file():
            _fail(f"required input is missing: {path}")
        relative = _relative(path)
        actual = sha256_file(path)
        if expected_sha256 is not None and actual != expected_sha256:
            _fail(f"input hash mismatch for {relative}: {actual} != {expected_sha256}")
        existing = self._items.get(relative)
        if existing is None:
            existing = {
                "path": relative,
                "sha256": actual,
                "size_bytes": int(path.stat().st_size),
                "roles": [],
            }
            self._items[relative] = existing
        elif existing["sha256"] != actual:
            _fail(f"input changed during r5 analysis: {relative}")
        if role not in existing["roles"]:
            existing["roles"].append(role)
            existing["roles"].sort()
        return actual

    def records(self) -> list[dict[str, Any]]:
        return [self._items[key] for key in sorted(self._items)]


def _stable_seed(namespace: str, chain: str, task: str) -> int:
    material = f"{namespace}|{chain}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def _ensure_finite(values: np.ndarray, role: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if not np.isfinite(out).all():
        _fail(f"non-finite values in {role}")
    return out


def weighted_average_precision(
    y: Sequence[int] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Exact grouped-threshold AP with optional nonnegative cluster multiplicities."""
    labels = np.asarray(y, dtype=int)
    scores = _ensure_finite(np.asarray(score, dtype=float), "average-precision scores")
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        _fail("average-precision arrays must be aligned nonempty vectors")
    if not set(np.unique(labels)).issubset({0, 1}):
        _fail("average-precision labels are not binary")
    if sample_weight is None:
        weights = np.ones(len(labels), dtype=float)
    else:
        weights = _ensure_finite(np.asarray(sample_weight, dtype=float), "sample weights")
        if weights.shape != labels.shape or (weights < 0).any():
            _fail("sample weights are misaligned or negative")
    positives = float(np.sum(weights * labels))
    negatives = float(np.sum(weights * (1 - labels)))
    if positives <= 0 or negatives <= 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_weights = weights[order]
    ordered_labels = labels[order]
    ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(order) - 1]
    cumulative_weight = np.cumsum(ordered_weights)[ends]
    cumulative_true = np.cumsum(ordered_weights * ordered_labels)[ends]
    precision = np.divide(
        cumulative_true,
        cumulative_weight,
        out=np.zeros_like(cumulative_true, dtype=float),
        where=cumulative_weight > 0,
    )
    true_increments = np.diff(np.r_[0.0, cumulative_true])
    return float(np.sum((true_increments / positives) * precision))


class WeightedAPPlan:
    """Pre-sort a fixed score vector so bootstrap draws only change row weights."""

    def __init__(self, y: Sequence[int] | np.ndarray, score: Sequence[float] | np.ndarray):
        labels = np.asarray(y, dtype=int)
        scores = _ensure_finite(np.asarray(score, dtype=float), "planned AP scores")
        if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
            _fail("planned AP arrays must be aligned nonempty vectors")
        if not set(np.unique(labels)).issubset({0, 1}):
            _fail("planned AP labels are not binary")
        self.length = len(labels)
        self.order = np.argsort(-scores, kind="mergesort")
        ordered_scores = scores[self.order]
        self.labels = labels[self.order]
        self.ends = np.r_[
            np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
            self.length - 1,
        ]

    def evaluate(self, sample_weight: Sequence[float] | np.ndarray | None = None) -> float:
        if sample_weight is None:
            weights = np.ones(self.length, dtype=float)
        else:
            raw = _ensure_finite(np.asarray(sample_weight, dtype=float), "planned AP weights")
            if raw.shape != (self.length,) or (raw < 0).any():
                _fail("planned AP weights are misaligned or negative")
            weights = raw[self.order]
        positives = float(np.sum(weights * self.labels))
        negatives = float(np.sum(weights * (1 - self.labels)))
        if positives <= 0 or negatives <= 0:
            return float("nan")
        cumulative_weight = np.cumsum(weights)[self.ends]
        cumulative_true = np.cumsum(weights * self.labels)[self.ends]
        precision = np.divide(
            cumulative_true,
            cumulative_weight,
            out=np.zeros_like(cumulative_true, dtype=float),
            where=cumulative_weight > 0,
        )
        true_increments = np.diff(np.r_[0.0, cumulative_true])
        return float(np.sum((true_increments / positives) * precision))


def _deterministic_order(frame: pd.DataFrame, score: np.ndarray, tie_columns: Sequence[str]) -> np.ndarray:
    values = _ensure_finite(np.asarray(score, dtype=float), "ranking scores")
    if len(frame) != len(values):
        _fail("ranking score and identity row counts differ")
    missing = [column for column in tie_columns if column not in frame]
    if missing:
        _fail(f"ranking frame lacks tie columns: {missing}")
    secondary = tuple(
        frame[column].astype(str).to_numpy() for column in reversed(tuple(tie_columns))
    )
    return np.lexsort(secondary + (-values,))


def midrank_ecdf(score: Sequence[float] | np.ndarray) -> np.ndarray:
    values = _ensure_finite(np.asarray(score, dtype=float), "ECDF scores")
    if not len(values):
        _fail("cannot normalize an empty score vector")
    ranks = pd.Series(values).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (ranks - 0.5) / float(len(values))


def _load_formal_summary(config: Mapping[str, Any], receipts: ReceiptBook) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    path = _resolve_role(config["formal_summary"])
    receipts.add(path, "formal_trained_reference_summary")
    summary = load_json(path)
    if summary.get("schema_version") != "upgrade-bench-v2/gpu-main-summary/1":
        _fail("formal GPU summary schema changed")
    if summary.get("status") != "complete" or summary.get("target_fold") != "main":
        _fail("formal GPU summary is not a complete main evaluation")
    if summary.get("protocol") != "strict_rolling_fold2_to_main":
        _fail("formal GPU protocol changed")
    if tuple(summary.get("seeds", [])) != EXPECTED_SEEDS:
        _fail("formal GPU seeds changed")
    if summary.get("complete_task_evaluations") != 36:
        _fail("formal GPU summary is not the complete 36-task inventory")

    manifest_path = _resolve_role(config["frozen_manifest"])
    run_config_path = _resolve_role(config["frozen_run_config"])
    if summary.get("manifest_artifact_role") != config["frozen_manifest"]:
        _fail("formal summary points to a different frozen manifest")
    if summary.get("run_config_artifact_role") != config["frozen_run_config"]:
        _fail("formal summary points to a different frozen run config")
    receipts.add(manifest_path, "frozen_gpu_manifest", summary.get("manifest_sha256"))
    receipts.add(run_config_path, "frozen_gpu_run_config", summary.get("run_config_sha256"))
    receipts.add(_resolve_role(config["metric_dependency"]), "formal_metric_semantics_dependency")

    expected = {
        (chain, task, family)
        for chain in config["chains"]
        for task in EXPECTED_TASKS
        for family in EXPECTED_FAMILIES
    }
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in summary.get("records", []):
        if not isinstance(record, dict):
            _fail("formal summary record is not an object")
        key = (record.get("chain"), record.get("track"), record.get("family"))
        if key in records:
            _fail(f"duplicate formal summary record: {key}")
        records[key] = record
    if set(records) != expected:
        _fail(
            f"formal summary inventory differs; missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )

    for key, record in records.items():
        receipts.add(
            _resolve_role(record["metric_artifact_role"]),
            f"formal_metric_artifact:{'|'.join(key)}",
            record["metric_artifact_sha256"],
        )
        receipts.add(
            _resolve_role(record["selection_artifact_role"]),
            f"frozen_selection_artifact:{'|'.join(key)}",
            record["selection_sha256"],
        )
    return summary, records


def _candidate_role(config: Mapping[str, Any], chain: str, task: str) -> str:
    return str(config["candidate_templates"][task]).format(chain=chain)


def _score_role(config: Mapping[str, Any], chain: str, task: str, family: str) -> str:
    return str(config["score_template"]).format(chain=chain, task=task, family=family)


def _load_candidate(
    role: str,
    expected_sha256: str,
    receipt_roles: Sequence[str],
    receipts: ReceiptBook,
) -> pd.DataFrame:
    path = _resolve_role(role)
    for receipt_role in receipt_roles:
        receipts.add(path, receipt_role, expected_sha256)
    metadata = pd.read_csv(
        path,
        nrows=1,
        usecols=["aggregation", "early_window", "late_window", "temporal_role"],
        dtype=str,
    )
    if len(metadata) != 1:
        _fail(f"empty candidate table: {role}")
    row = metadata.iloc[0]
    if row["aggregation"] != "calendar_mean":
        _fail(f"candidate aggregation is not calendar_mean: {role}")
    if row["early_window"] != "2008-2012" or row["late_window"] != "2018-2022":
        _fail(f"candidate windows changed: {role}")
    if row["temporal_role"] != "target":
        _fail(f"candidate is not the frozen target role: {role}")

    frame = pd.read_csv(
        path,
        usecols=[*KEYS, "y", "lateval"],
        dtype={key: str for key in KEYS},
    )
    if frame[list(KEYS)].isna().any().any() or frame.duplicated(list(KEYS)).any():
        _fail(f"candidate keys are null or duplicated: {role}")
    ordered = frame.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    if not frame.reset_index(drop=True).equals(ordered):
        _fail(f"candidate rows are not in canonical key order: {role}")
    if not set(frame["y"].dropna().unique()).issubset({0, 1}) or frame["y"].isna().any():
        _fail(f"candidate labels are not complete binary values: {role}")
    lateval = frame["lateval"].to_numpy(dtype=float)
    if not np.isfinite(lateval).all() or (lateval < 0).any():
        _fail(f"candidate late values are invalid: {role}")
    if ((frame["y"].to_numpy(dtype=int) == 0) & (lateval != 0)).any():
        _fail(f"negative candidate has nonzero late value: {role}")
    return frame


def _load_scores(
    role: str,
    record: Mapping[str, Any],
    candidate: pd.DataFrame,
    receipts: ReceiptBook,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if record.get("score_artifact_role") != role:
        _fail(f"formal score role differs for {role}")
    path = _resolve_role(role)
    receipts.add(
        path,
        f"frozen_main_score:{record['chain']}|{record['track']}|{record['family']}",
        record.get("score_artifact_sha256"),
    )
    frame = pd.read_csv(path, dtype={key: str for key in KEYS})
    expected_non_scores = {*KEYS, "selection_sha256", "protocol"}
    score_columns: dict[int, str] = {}
    model_names: set[str] = set()
    for column in frame.columns:
        match = re.fullmatch(r"score_(.+)_s([0-9]+)", column)
        if match:
            seed = int(match.group(2))
            if seed in score_columns:
                _fail(f"duplicate seed score column in {role}")
            score_columns[seed] = column
            model_names.add(match.group(1))
        elif column not in expected_non_scores:
            _fail(f"unexpected score column {column!r} in {role}")
    if set(score_columns) != set(EXPECTED_SEEDS) or len(model_names) != 1:
        _fail(f"score seed/model inventory changed in {role}")
    if set(frame.columns) != expected_non_scores | set(score_columns.values()):
        _fail(f"score CSV column inventory changed in {role}")
    if not frame.loc[:, list(KEYS)].equals(candidate.loc[:, list(KEYS)]):
        _fail(f"score identities do not exactly align with candidate rows: {role}")
    if frame["selection_sha256"].nunique(dropna=False) != 1:
        _fail(f"score CSV has multiple selection hashes: {role}")
    if frame["selection_sha256"].iloc[0] != record.get("selection_sha256"):
        _fail(f"score selection hash differs from formal summary: {role}")
    if frame["protocol"].nunique(dropna=False) != 1 or frame["protocol"].iloc[0] != "strict_rolling_fold2_to_main":
        _fail(f"score protocol differs from formal main protocol: {role}")
    model_name = next(iter(model_names))
    if model_name != record.get("selected_model"):
        _fail(f"score model {model_name!r} differs from selected model in {role}")
    scores = {
        seed: _ensure_finite(frame[column].to_numpy(dtype=float), f"{role} seed {seed}")
        for seed, column in sorted(score_columns.items())
    }
    return scores, {
        "selected_model": model_name,
        "selection_sha256": record["selection_sha256"],
        "score_sha256": record["score_artifact_sha256"],
    }


def _build_b1_entries(candidate: pd.DataFrame, lane_score: np.ndarray, method: Mapping[str, Any]) -> pd.DataFrame:
    frame = candidate.loc[:, [*KEYS, "y", "lateval"]].copy()
    frame["lane_score"] = _ensure_finite(lane_score, "B1 lane scores")
    base = (
        frame.groupby(list(ENTRY_KEYS), sort=True, as_index=False)
        .agg(y=("y", "max"), lateval=("lateval", "sum"), lane_count=("y", "size"))
        .sort_values(list(ENTRY_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    kind = method["kind"]
    if kind == "raw_max":
        pooled = (
            frame.groupby(list(ENTRY_KEYS), sort=True, as_index=False)["lane_score"]
            .max()
            .rename(columns={"lane_score": "score"})
        )
    else:
        frame["normalized_score"] = midrank_ecdf(frame["lane_score"].to_numpy(dtype=float))
        if kind == "ecdf_mean":
            pooled = (
                frame.groupby(list(ENTRY_KEYS), sort=True, as_index=False)["normalized_score"]
                .mean()
                .rename(columns={"normalized_score": "score"})
            )
        elif kind == "ecdf_topk_mean":
            top_k = int(method["top_k"])
            top = frame.sort_values(
                [*ENTRY_KEYS, "normalized_score", "j_iso"],
                ascending=[True, True, False, True],
                kind="mergesort",
            ).groupby(list(ENTRY_KEYS), sort=True, as_index=False).head(top_k)
            pooled = (
                top.groupby(list(ENTRY_KEYS), sort=True, as_index=False)["normalized_score"]
                .mean()
                .rename(columns={"normalized_score": "score"})
            )
        else:
            _fail(f"unknown B1 pooling kind: {kind}")
    entry = base.merge(pooled, on=list(ENTRY_KEYS), how="left", validate="one_to_one")
    if entry["score"].isna().any() or not np.isfinite(entry["score"].to_numpy(dtype=float)).all():
        _fail("B1 pooling produced missing or non-finite entry scores")
    return entry


def _entry_metrics(entry: pd.DataFrame, budgets: Sequence[int]) -> dict[str, float]:
    y = entry["y"].to_numpy(dtype=int)
    score = entry["score"].to_numpy(dtype=float)
    lateval = entry["lateval"].to_numpy(dtype=float)
    result: dict[str, float] = {
        "entry_average_precision": weighted_average_precision(y, score),
        "entry_groups": float(len(entry)),
        "positive_entries": float(y.sum()),
    }
    order = _deterministic_order(entry, score, ENTRY_KEYS)
    total_positive = max(float(y.sum()), 1.0)
    total_value = max(float(lateval.sum()), 1.0)
    for budget in budgets:
        chosen = order[: min(int(budget), len(order))]
        result[f"entry_precision_at_{budget}"] = float(y[chosen].mean()) if len(chosen) else 0.0
        result[f"entry_recall_at_{budget}"] = float(y[chosen].sum() / total_positive)
        result[f"entry_value_capture_at_{budget}"] = float(lateval[chosen].sum() / total_value)
        result[f"effective_k_at_{budget}"] = float(len(chosen))
    return result


def _a_budget_conditions(candidate: pd.DataFrame, score: np.ndarray, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    y = candidate["y"].to_numpy(dtype=int)
    lateval = candidate["lateval"].to_numpy(dtype=float)
    order = _deterministic_order(candidate, score, ("i_iso", "stage", "j_iso"))
    total_positive = max(float(y.sum()), 1.0)
    total_value = max(float(lateval.sum()), 1.0)
    records: list[dict[str, Any]] = []
    for budget in config["budgets"]["a_global"]:
        chosen = order[: min(int(budget), len(order))]
        records.append(
            {
                "budget_scope": "global_lanes",
                "requested_k": int(budget),
                "metrics": {
                    "precision": float(y[chosen].mean()) if len(chosen) else 0.0,
                    "recall": float(y[chosen].sum() / total_positive),
                    "observed_value_capture": float(lateval[chosen].sum() / total_value),
                    "effective_k": float(len(chosen)),
                },
            }
        )

    lane = candidate.loc[:, [*KEYS, "y", "lateval"]].copy()
    lane["score"] = score
    for budget in config["budgets"]["a_per_exporter"]:
        precisions: list[float] = []
        recalls: list[float] = []
        values: list[float] = []
        effective: list[float] = []
        for _, group in lane.groupby("i_iso", sort=True):
            top = group.sort_values(
                ["score", "stage", "j_iso"],
                ascending=[False, True, True],
                kind="mergesort",
            ).head(int(budget))
            effective.append(float(len(top)))
            precisions.append(float(top["y"].mean()) if len(top) else 0.0)
            positives = int(group["y"].sum())
            if positives:
                recalls.append(float(top["y"].sum() / positives))
            realized = float(group["lateval"].sum())
            if realized > 0:
                values.append(float(top["lateval"].sum() / realized))
        records.append(
            {
                "budget_scope": "per_exporter_lanes",
                "requested_k": int(budget),
                "metrics": {
                    "macro_precision": float(np.mean(precisions)),
                    "macro_recall": float(np.mean(recalls)),
                    "macro_observed_value_capture": float(np.mean(values)),
                    "group_count": float(len(effective)),
                    "effective_k_min": float(np.min(effective)),
                    "effective_k_mean": float(np.mean(effective)),
                    "effective_k_max": float(np.max(effective)),
                },
            }
        )
    return records


def _b1_budget_conditions(entry: pd.DataFrame, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = _entry_metrics(entry, config["budgets"]["b1_global"])
    return [
        {
            "budget_scope": "global_entries",
            "requested_k": int(budget),
            "metrics": {
                "precision": metrics[f"entry_precision_at_{budget}"],
                "recall": metrics[f"entry_recall_at_{budget}"],
                "observed_value_capture": metrics[f"entry_value_capture_at_{budget}"],
                "effective_k": metrics[f"effective_k_at_{budget}"],
            },
        }
        for budget in config["budgets"]["b1_global"]
    ]


def _positive_entry_statistics(
    candidate: pd.DataFrame,
    score: np.ndarray,
    budgets: Sequence[int],
) -> pd.DataFrame:
    lane = candidate.loc[:, [*KEYS, "y", "lateval"]].copy()
    lane["score"] = _ensure_finite(score, "B2 lane scores")
    rows: list[dict[str, Any]] = []
    for (exporter, stage), group in lane.groupby(list(ENTRY_KEYS), sort=True):
        positives = int(group["y"].sum())
        if not positives:
            continue
        total_value = float(group["lateval"].sum())
        if total_value <= 0:
            _fail(f"positive B2 entry has no observed value: {exporter}|{stage}")
        ordered = group.sort_values(
            ["score", "j_iso"], ascending=[False, True], kind="mergesort"
        )
        row: dict[str, Any] = {
            "i_iso": exporter,
            "stage": stage,
            "positive_lanes": float(positives),
            "observed_value": total_value,
        }
        for budget in budgets:
            top = ordered.head(int(budget))
            row[f"recall_at_{budget}"] = float(top["y"].sum() / positives)
            row[f"value_at_{budget}"] = float(top["lateval"].sum() / total_value)
            row[f"retrieved_positive_lanes_at_{budget}"] = float(top["y"].sum())
            row[f"retrieved_value_at_{budget}"] = float(top["lateval"].sum())
            row[f"effective_k_at_{budget}"] = float(len(top))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(list(ENTRY_KEYS), kind="mergesort").reset_index(drop=True)
    if result.duplicated(list(ENTRY_KEYS)).any():
        _fail("B2 positive entry statistics contain duplicate groups")
    return result


def _b2_budget_conditions(stats: pd.DataFrame, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for budget in config["budgets"]["b2_per_positive_entry"]:
        effective = stats[f"effective_k_at_{budget}"].to_numpy(dtype=float)
        records.append(
            {
                "budget_scope": "per_positive_entry_destinations",
                "requested_k": int(budget),
                "metrics": {
                    "macro_recall": float(stats[f"recall_at_{budget}"].mean()),
                    "macro_observed_value_capture": float(stats[f"value_at_{budget}"].mean()),
                    "positive_entry_groups": float(len(stats)),
                    "effective_k_min": float(effective.min()),
                    "effective_k_mean": float(effective.mean()),
                    "effective_k_max": float(effective.max()),
                },
            }
        )
    return records


def _close(actual: float, expected: float, role: str, tolerance: float = 2e-12) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > tolerance:
        _fail(f"formal metric mismatch for {role}: {actual} != {expected}")


def _formal_seed_record(record: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    matches = [row for row in record.get("metrics_by_seed", []) if row.get("seed") == seed]
    if len(matches) != 1:
        _fail(f"formal metric record lacks exactly one seed {seed}")
    return matches[0]


def _validate_formal_metrics(
    task: str,
    record: Mapping[str, Any],
    seed: int,
    primary: float,
    conditions: Sequence[Mapping[str, Any]],
) -> None:
    formal = _formal_seed_record(record, seed)
    if task == "a":
        _close(primary, float(formal["lane_average_precision"]), f"{record['chain']}|a|{record['family']}|s{seed}|AP")
        for condition in conditions:
            budget = condition["requested_k"]
            metrics = condition["metrics"]
            if condition["budget_scope"] == "global_lanes" and budget <= 500:
                for local, formal_name in (
                    ("precision", f"precision_at_{budget}"),
                    ("recall", f"recall_at_{budget}"),
                    ("observed_value_capture", f"value_capture_at_{budget}"),
                ):
                    _close(float(metrics[local]), float(formal[formal_name]), f"A {formal_name}")
            elif condition["budget_scope"] == "per_exporter_lanes":
                for local, formal_name in (
                    ("macro_precision", f"exporter_macro_precision_at_{budget}"),
                    ("macro_recall", f"exporter_macro_recall_at_{budget}"),
                    ("macro_observed_value_capture", f"exporter_macro_value_capture_at_{budget}"),
                ):
                    _close(float(metrics[local]), float(formal[formal_name]), f"A {formal_name}")
    elif task == "b1":
        _close(primary, float(formal["entry_average_precision"]), f"{record['chain']}|b1|{record['family']}|s{seed}|AP")
        for condition in conditions:
            budget = condition["requested_k"]
            metrics = condition["metrics"]
            _close(float(metrics["precision"]), float(formal[f"entry_precision_at_{budget}"]), f"B1 precision@{budget}")
            _close(float(metrics["recall"]), float(formal[f"entry_recall_at_{budget}"]), f"B1 recall@{budget}")
    else:
        for condition in conditions:
            budget = condition["requested_k"]
            metrics = condition["metrics"]
            _close(float(metrics["macro_recall"]), float(formal[f"conditional_recall_at_{budget}"]), f"B2 recall@{budget}")
            _close(
                float(metrics["macro_observed_value_capture"]),
                float(formal[f"conditional_value_capture_at_{budget}"]),
                f"B2 value@{budget}",
            )
        _close(primary, float(formal["conditional_recall_at_3"]), f"{record['chain']}|b2|{record['family']}|s{seed}|R@3")


def _cluster_index(values: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    labels = sorted(set(str(value) for value in values))
    mapping = {label: index for index, label in enumerate(labels)}
    return np.asarray([mapping[str(value)] for value in values], dtype=int), labels


def _interval(draws: np.ndarray, confidence: float, requested: int, role: str) -> dict[str, Any]:
    values = np.asarray(draws, dtype=float)
    finite = values[np.isfinite(values)]
    minimum = max(20, requested // 2)
    if len(finite) < minimum:
        _fail(f"too few finite paired-bootstrap draws for {role}: {len(finite)} < {minimum}")
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(finite, alpha))
    upper = float(np.quantile(finite, 1.0 - alpha))
    if lower < -1.000000000001 or upper > 1.000000000001 or lower > upper:
        _fail(f"invalid paired interval for {role}: [{lower}, {upper}]")
    return {
        "requested_draws": int(requested),
        "finite_draws": int(len(finite)),
        "lower_95": lower,
        "upper_95": upper,
    }


def _paired_chain_task(
    chain: str,
    task: str,
    candidate: pd.DataFrame,
    left_scores: Mapping[int, np.ndarray],
    right_scores: Mapping[int, np.ndarray],
    selected_models: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    iterations = int(config["bootstrap"]["iterations"])
    confidence = float(config["bootstrap"]["confidence_level"])
    rng_seed = _stable_seed(config["bootstrap"]["seed_namespace"], chain, task)
    rng = np.random.default_rng(rng_seed)
    per_seed_points: dict[int, tuple[float, float]] = {}
    per_seed_draws = {seed: np.empty(iterations, dtype=float) for seed in EXPECTED_SEEDS}

    if task == "a":
        y = candidate["y"].to_numpy(dtype=int)
        row_cluster, cluster_keys = _cluster_index(candidate["i_iso"].astype(str).tolist())
        left_plans = {seed: WeightedAPPlan(y, left_scores[seed]) for seed in EXPECTED_SEEDS}
        right_plans = {seed: WeightedAPPlan(y, right_scores[seed]) for seed in EXPECTED_SEEDS}
        for seed in EXPECTED_SEEDS:
            per_seed_points[seed] = (left_plans[seed].evaluate(), right_plans[seed].evaluate())
        for draw in range(iterations):
            sampled = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
            counts = np.bincount(sampled, minlength=len(cluster_keys)).astype(float)
            row_weights = counts[row_cluster]
            for seed in EXPECTED_SEEDS:
                per_seed_draws[seed][draw] = left_plans[seed].evaluate(
                    row_weights
                ) - right_plans[seed].evaluate(row_weights)
    elif task == "b1":
        official = config["b1_pooling_methods"][0]
        left_entry = {seed: _build_b1_entries(candidate, left_scores[seed], official) for seed in EXPECTED_SEEDS}
        right_entry = {seed: _build_b1_entries(candidate, right_scores[seed], official) for seed in EXPECTED_SEEDS}
        canonical = left_entry[0].loc[:, [*ENTRY_KEYS, "y"]]
        for seed in EXPECTED_SEEDS:
            if not left_entry[seed].loc[:, [*ENTRY_KEYS, "y"]].equals(canonical):
                _fail(f"left B1 entry universe varies by seed for {chain}")
            if not right_entry[seed].loc[:, [*ENTRY_KEYS, "y"]].equals(canonical):
                _fail(f"paired B1 entry universes differ for {chain}")
        y = canonical["y"].to_numpy(dtype=int)
        row_cluster, cluster_keys = _cluster_index(canonical["i_iso"].astype(str).tolist())
        left_plans = {
            seed: WeightedAPPlan(y, left_entry[seed]["score"].to_numpy(dtype=float))
            for seed in EXPECTED_SEEDS
        }
        right_plans = {
            seed: WeightedAPPlan(y, right_entry[seed]["score"].to_numpy(dtype=float))
            for seed in EXPECTED_SEEDS
        }
        for seed in EXPECTED_SEEDS:
            per_seed_points[seed] = (left_plans[seed].evaluate(), right_plans[seed].evaluate())
        for draw in range(iterations):
            sampled = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
            counts = np.bincount(sampled, minlength=len(cluster_keys)).astype(float)
            row_weights = counts[row_cluster]
            for seed in EXPECTED_SEEDS:
                per_seed_draws[seed][draw] = left_plans[seed].evaluate(
                    row_weights
                ) - right_plans[seed].evaluate(row_weights)
    else:
        budgets = config["budgets"]["b2_per_positive_entry"]
        left_stats = {
            seed: _positive_entry_statistics(candidate, left_scores[seed], budgets)
            for seed in EXPECTED_SEEDS
        }
        right_stats = {
            seed: _positive_entry_statistics(candidate, right_scores[seed], budgets)
            for seed in EXPECTED_SEEDS
        }
        canonical = left_stats[0].loc[:, list(ENTRY_KEYS)]
        for seed in EXPECTED_SEEDS:
            if not left_stats[seed].loc[:, list(ENTRY_KEYS)].equals(canonical):
                _fail(f"left B2 positive-entry universe varies by seed for {chain}")
            if not right_stats[seed].loc[:, list(ENTRY_KEYS)].equals(canonical):
                _fail(f"paired B2 positive-entry universes differ for {chain}")
            left = left_stats[seed]["recall_at_3"].to_numpy(dtype=float)
            right = right_stats[seed]["recall_at_3"].to_numpy(dtype=float)
            per_seed_points[seed] = (float(left.mean()), float(right.mean()))
        cluster_keys = [f"{row.i_iso}|{row.stage}" for row in canonical.itertuples(index=False)]
        for draw in range(iterations):
            sampled = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
            counts = np.bincount(sampled, minlength=len(cluster_keys)).astype(float)
            for seed in EXPECTED_SEEDS:
                left = left_stats[seed]["recall_at_3"].to_numpy(dtype=float)
                right = right_stats[seed]["recall_at_3"].to_numpy(dtype=float)
                per_seed_draws[seed][draw] = float(np.average(left - right, weights=counts))

    seed_records: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        left, right = per_seed_points[seed]
        delta = float(left - right)
        interval = _interval(per_seed_draws[seed], confidence, iterations, f"{chain}|{task}|s{seed}")
        seed_records.append(
            {
                "seed": seed,
                "left_point": left,
                "right_point": right,
                "delta": delta,
                **interval,
            }
        )
    fixed_seed_draws = np.mean(np.stack([per_seed_draws[seed] for seed in EXPECTED_SEEDS]), axis=0)
    fixed_seed_point = float(np.mean([row["delta"] for row in seed_records]))
    fixed_interval = _interval(fixed_seed_draws, confidence, iterations, f"{chain}|{task}|fixed-seed-mean")
    record = {
        "chain": chain,
        "task": task,
        "metric": PRIMARY_METRICS[task],
        "direction": PAIR_DIRECTION,
        "left_family": "kge",
        "right_family": "nbfnet",
        "left_selected_model": selected_models["kge"],
        "right_selected_model": selected_models["nbfnet"],
        "cluster_unit": CLUSTER_UNITS[task],
        "cluster_count": int(len(cluster_keys)),
        "bootstrap_rng_seed": int(rng_seed),
        "same_cluster_draws_for_both_families": True,
        "same_cluster_draws_for_all_fixed_seeds": True,
        "per_seed": seed_records,
        "fixed_five_seed_mean": {
            "seeds": list(EXPECTED_SEEDS),
            "point": fixed_seed_point,
            **fixed_interval,
        },
    }
    return record, fixed_seed_draws


def _condition_seed_summaries(records: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in key_fields)].append(record)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        rows = sorted(groups[key], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(EXPECTED_SEEDS):
            _fail(f"condition lacks the five frozen seeds: {dict(zip(key_fields, key))}")
        metric_names = set(rows[0]["metrics"])
        if any(set(row["metrics"]) != metric_names for row in rows):
            _fail("condition metric inventory varies by seed")
        summary_metrics = {}
        for metric in sorted(metric_names):
            values = np.asarray([float(row["metrics"][metric]) for row in rows], dtype=float)
            if not np.isfinite(values).all():
                _fail(f"non-finite condition metric: {metric}")
            summary_metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(len(values)),
            }
        summaries.append(
            {
                **dict(zip(key_fields, key)),
                "seeds": list(EXPECTED_SEEDS),
                "metrics": summary_metrics,
            }
        )
    return summaries


def _fixed_six_chain_summaries(
    per_chain: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    chains: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in per_chain:
        groups[tuple(record[field] for field in key_fields)].append(record)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        rows = groups[key]
        if {row["chain"] for row in rows} != set(chains) or len(rows) != len(chains):
            _fail(f"fixed-six summary lacks a complete chain inventory: {key}")
        metric_names = set(rows[0]["metrics"])
        if any(set(row["metrics"]) != metric_names for row in rows):
            _fail("fixed-six metric inventory varies by chain")
        metrics = {
            metric: float(np.mean([float(row["metrics"][metric]["mean"]) for row in rows]))
            for metric in sorted(metric_names)
        }
        output.append(
            {
                **dict(zip(key_fields, key)),
                "chains": list(chains),
                "chain_count": len(chains),
                "chain_weighting": "unweighted_mean_of_per_chain_fixed_seed_means",
                "metrics": metrics,
            }
        )
    return output


def _two_stage_condition(
    candidate: pd.DataFrame,
    b1_score: np.ndarray,
    b2_score: np.ndarray,
    b1_budget: int,
    b2_budget: int,
    official_method: Mapping[str, Any],
) -> dict[str, float]:
    entry = _build_b1_entries(candidate, b1_score, official_method)
    order = _deterministic_order(entry, entry["score"].to_numpy(dtype=float), ENTRY_KEYS)
    chosen = entry.iloc[order[: min(int(b1_budget), len(entry))]]
    chosen_keys = set(zip(chosen["i_iso"], chosen["stage"]))
    positive_entry_count = int(entry["y"].sum())
    selected_positive = int(chosen["y"].sum())

    stats = _positive_entry_statistics(candidate, b2_score, [int(b2_budget)])
    if len(stats) != positive_entry_count:
        _fail("two-stage denominator does not equal the complete positive-entry universe")
    macro_recall: list[float] = []
    macro_value: list[float] = []
    retrieved_positive_lanes = 0.0
    retrieved_value = 0.0
    for row in stats.itertuples(index=False):
        selected = (row.i_iso, row.stage) in chosen_keys
        if selected:
            macro_recall.append(float(getattr(row, f"recall_at_{b2_budget}")))
            macro_value.append(float(getattr(row, f"value_at_{b2_budget}")))
            retrieved_positive_lanes += float(
                getattr(row, f"retrieved_positive_lanes_at_{b2_budget}")
            )
            retrieved_value += float(getattr(row, f"retrieved_value_at_{b2_budget}"))
        else:
            macro_recall.append(0.0)
            macro_value.append(0.0)
    total_positive_lanes = float(stats["positive_lanes"].sum())
    total_value = float(stats["observed_value"].sum())
    effective_b1 = len(chosen)
    return {
        "positive_entry_gate_precision": float(selected_positive / max(effective_b1, 1)),
        "positive_entry_gate_recall": float(selected_positive / max(positive_entry_count, 1)),
        "e2e_macro_destination_recall": float(np.mean(macro_recall)),
        "e2e_macro_destination_observed_value_capture": float(np.mean(macro_value)),
        "e2e_micro_destination_recall": float(retrieved_positive_lanes / max(total_positive_lanes, 1.0)),
        "e2e_global_observed_value_capture": float(retrieved_value / max(total_value, 1.0)),
        "effective_b1_budget": float(effective_b1),
        "selected_positive_entries": float(selected_positive),
        "positive_entries_denominator": float(positive_entry_count),
        "positive_lanes_denominator": total_positive_lanes,
    }


def _analysis_inventory(config: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    _require_exact_keys(
        analysis,
        {
            "protocol",
            "paired_family_comparison",
            "budget_summaries",
            "b1_pooling_sensitivity",
            "two_stage_b1_b2_diagnostic",
            "eligibility_threshold_cohort_geometry",
        },
        "r5 analysis",
    )
    paired = analysis["paired_family_comparison"]
    if len(paired["per_chain"]) != 18 or len(paired["fixed_six_chain"]) != 3:
        _fail("paired comparison inventory is not 18 chain-task plus three fixed-six records")
    expected_pairs = {(chain, task) for chain in config["chains"] for task in EXPECTED_TASKS}
    actual_pairs = {(row["chain"], row["task"]) for row in paired["per_chain"]}
    if actual_pairs != expected_pairs:
        _fail("paired comparison chain-task inventory differs")
    for row in paired["per_chain"]:
        if len(row["per_seed"]) != 5 or row["direction"] != PAIR_DIRECTION:
            _fail("paired comparison seed inventory or direction differs")
        for seed_row in row["per_seed"]:
            if not -1.000000000001 <= float(seed_row["delta"]) <= 1.000000000001:
                _fail("paired point difference is outside [-1,1]")

    expected_budget = len(config["chains"]) * 2 * 5 * (7 + 4 + 3)
    if len(analysis["budget_summaries"]["per_seed"]) != expected_budget:
        _fail("budget per-seed inventory differs")
    if len(analysis["budget_summaries"]["per_chain_seed_summary"]) != len(config["chains"]) * 2 * 14:
        _fail("budget per-chain summary inventory differs")
    if len(analysis["budget_summaries"]["fixed_six_chain"]) != 2 * 14:
        _fail("budget fixed-six inventory differs")

    method_count = len(config["b1_pooling_methods"])
    if len(analysis["b1_pooling_sensitivity"]["per_seed"]) != len(config["chains"]) * 2 * 5 * method_count:
        _fail("B1 pooling per-seed inventory differs")
    if len(analysis["b1_pooling_sensitivity"]["per_chain_seed_summary"]) != len(config["chains"]) * 2 * method_count:
        _fail("B1 pooling per-chain summary inventory differs")
    if len(analysis["b1_pooling_sensitivity"]["fixed_six_chain"]) != 2 * method_count:
        _fail("B1 pooling fixed-six inventory differs")

    expected_two_stage = len(config["chains"]) * 2 * 5 * 4 * 3
    if len(analysis["two_stage_b1_b2_diagnostic"]["per_seed"]) != expected_two_stage:
        _fail("two-stage per-seed inventory differs")
    if len(analysis["two_stage_b1_b2_diagnostic"]["per_chain_seed_summary"]) != len(config["chains"]) * 2 * 4 * 3:
        _fail("two-stage per-chain summary inventory differs")
    if len(analysis["two_stage_b1_b2_diagnostic"]["fixed_six_chain"]) != 2 * 4 * 3:
        _fail("two-stage fixed-six inventory differs")
    if analysis["eligibility_threshold_cohort_geometry"]["status"] != "not_computed_from_fixed_scores":
        _fail("threshold geometry was silently approximated")


def build_analysis(config: Mapping[str, Any], receipts: ReceiptBook) -> dict[str, Any]:
    _, formal = _load_formal_summary(config, receipts)
    paired_rows: list[dict[str, Any]] = []
    paired_draws: dict[tuple[str, str], np.ndarray] = {}
    budget_rows: list[dict[str, Any]] = []
    pooling_rows: list[dict[str, Any]] = []
    two_stage_rows: list[dict[str, Any]] = []

    for chain in config["chains"]:
        candidate_cache: dict[str, pd.DataFrame] = {}
        score_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        score_meta: dict[tuple[str, str], dict[str, Any]] = {}
        for task in EXPECTED_TASKS:
            candidate_role = _candidate_role(config, chain, task)
            candidate_key = "a" if task == "a" else "b"
            expected_candidate_hashes = {
                formal[(chain, task, family)]["target_candidate_sha256"]
                for family in EXPECTED_FAMILIES
            }
            if len(expected_candidate_hashes) != 1:
                _fail(f"paired families bind different candidate hashes for {chain}|{task}")
            if candidate_key not in candidate_cache:
                candidate_cache[candidate_key] = _load_candidate(
                    candidate_role,
                    next(iter(expected_candidate_hashes)),
                    [f"target_candidate:{chain}|{task}|{family}" for family in EXPECTED_FAMILIES],
                    receipts,
                )
            else:
                actual = receipts.add(
                    _resolve_role(candidate_role),
                    f"target_candidate:{chain}|{task}",
                    next(iter(expected_candidate_hashes)),
                )
                if actual != next(iter(expected_candidate_hashes)):
                    _fail("candidate hash changed while reusing B views")
            candidate = candidate_cache[candidate_key]
            for family in EXPECTED_FAMILIES:
                record = formal[(chain, task, family)]
                if int(record["target_rows"]) != len(candidate) or int(record["target_positive_lanes"]) != int(candidate["y"].sum()):
                    _fail(f"candidate counts differ from formal summary for {chain}|{task}|{family}")
                role = _score_role(config, chain, task, family)
                score_cache[(task, family)], score_meta[(task, family)] = _load_scores(
                    role, record, candidate, receipts
                )

            selected_models = {
                family: score_meta[(task, family)]["selected_model"] for family in EXPECTED_FAMILIES
            }
            paired, draws = _paired_chain_task(
                chain,
                task,
                candidate,
                score_cache[(task, "kge")],
                score_cache[(task, "nbfnet")],
                selected_models,
                config,
            )
            paired_rows.append(paired)
            paired_draws[(chain, task)] = draws

            for family in EXPECTED_FAMILIES:
                record = formal[(chain, task, family)]
                for seed in EXPECTED_SEEDS:
                    score = score_cache[(task, family)][seed]
                    if task == "a":
                        primary = weighted_average_precision(candidate["y"].to_numpy(dtype=int), score)
                        conditions = _a_budget_conditions(candidate, score, config)
                    elif task == "b1":
                        entry = _build_b1_entries(candidate, score, config["b1_pooling_methods"][0])
                        primary = weighted_average_precision(
                            entry["y"].to_numpy(dtype=int), entry["score"].to_numpy(dtype=float)
                        )
                        conditions = _b1_budget_conditions(entry, config)
                    else:
                        stats = _positive_entry_statistics(
                            candidate, score, config["budgets"]["b2_per_positive_entry"]
                        )
                        primary = float(stats["recall_at_3"].mean())
                        conditions = _b2_budget_conditions(stats, config)
                    _validate_formal_metrics(task, record, seed, primary, conditions)
                    for condition in conditions:
                        budget_rows.append(
                            {
                                "chain": chain,
                                "task": task,
                                "family": family,
                                "seed": seed,
                                **condition,
                            }
                        )

        b_candidate = candidate_cache["b"]
        for family in EXPECTED_FAMILIES:
            for seed in EXPECTED_SEEDS:
                b1_lane_score = score_cache[("b1", family)][seed]
                for method in config["b1_pooling_methods"]:
                    entry = _build_b1_entries(b_candidate, b1_lane_score, method)
                    pooling_rows.append(
                        {
                            "chain": chain,
                            "family": family,
                            "seed": seed,
                            "method": method["name"],
                            "normative": bool(method["normative"]),
                            "metrics": _entry_metrics(entry, config["budgets"]["b1_global"]),
                        }
                    )
                for b1_budget in config["two_stage"]["b1_entry_budgets"]:
                    for b2_budget in config["two_stage"]["b2_destination_budgets"]:
                        two_stage_rows.append(
                            {
                                "chain": chain,
                                "family": family,
                                "seed": seed,
                                "b1_budget": int(b1_budget),
                                "b2_budget": int(b2_budget),
                                "metrics": _two_stage_condition(
                                    b_candidate,
                                    score_cache[("b1", family)][seed],
                                    score_cache[("b2", family)][seed],
                                    int(b1_budget),
                                    int(b2_budget),
                                    config["b1_pooling_methods"][0],
                                ),
                            }
                        )

    fixed_six_paired: list[dict[str, Any]] = []
    iterations = int(config["bootstrap"]["iterations"])
    confidence = float(config["bootstrap"]["confidence_level"])
    for task in EXPECTED_TASKS:
        task_rows = [row for row in paired_rows if row["task"] == task]
        if {row["chain"] for row in task_rows} != set(config["chains"]):
            _fail(f"fixed-six paired summary lacks chains for task {task}")
        point = float(np.mean([row["fixed_five_seed_mean"]["point"] for row in task_rows]))
        draws = np.mean(
            np.stack([paired_draws[(chain, task)] for chain in config["chains"]]), axis=0
        )
        fixed_six_paired.append(
            {
                "task": task,
                "metric": PRIMARY_METRICS[task],
                "direction": PAIR_DIRECTION,
                "point": point,
                **_interval(draws, confidence, iterations, f"fixed-six|{task}"),
                "chains": list(config["chains"]),
                "seeds": list(EXPECTED_SEEDS),
                "chain_weighting": "unweighted",
                "cluster_resampling": "stratified_within_each_fixed_chain",
                "chains_resampled": False,
                "inference_scope": (
                    "The six declared chains and five declared seeds only; this interval does not "
                    "support population inference over value chains."
                ),
            }
        )

    budget_chain = _condition_seed_summaries(
        budget_rows,
        ("chain", "task", "family", "budget_scope", "requested_k"),
    )
    budget_fixed = _fixed_six_chain_summaries(
        budget_chain,
        ("task", "family", "budget_scope", "requested_k"),
        config["chains"],
    )
    pooling_chain = _condition_seed_summaries(
        pooling_rows,
        ("chain", "family", "method", "normative"),
    )
    pooling_fixed = _fixed_six_chain_summaries(
        pooling_chain,
        ("family", "method", "normative"),
        config["chains"],
    )
    two_stage_chain = _condition_seed_summaries(
        two_stage_rows,
        ("chain", "family", "b1_budget", "b2_budget"),
    )
    two_stage_fixed = _fixed_six_chain_summaries(
        two_stage_chain,
        ("family", "b1_budget", "b2_budget"),
        config["chains"],
    )

    analysis = {
        "protocol": {
            "source": "formal frozen main score artifacts only",
            "training_performed": False,
            "fine_tuning_performed": False,
            "calibration_performed": False,
            "main_labels_used_for_method_selection": False,
            "raw_scores_averaged_across_seeds": False,
            "all_declared_sensitivity_variants_reported": True,
            "paired_direction": PAIR_DIRECTION,
            "paired_resampling": (
                "Identical cluster multiplicities are applied to both families. A and B1 use "
                "exporter clusters; B2 uses complete realized exporter-stage entries."
            ),
            "fixed_six_chain_inference_scope": (
                "Finite-benchmark descriptive intervals only. Chains are fixed and never resampled; "
                "no population inference over value chains is made."
            ),
        },
        "paired_family_comparison": {
            "per_chain": paired_rows,
            "fixed_six_chain": fixed_six_paired,
        },
        "budget_summaries": {
            "per_seed": budget_rows,
            "per_chain_seed_summary": budget_chain,
            "fixed_six_chain": budget_fixed,
            "selection_policy": "No cutoff is selected from main results; the full config-declared grid is reported.",
        },
        "b1_pooling_sensitivity": {
            "per_seed": pooling_rows,
            "per_chain_seed_summary": pooling_chain,
            "fixed_six_chain": pooling_fixed,
            "normative_method": "official_raw_max",
            "alternative_status": "descriptive sensitivity only; no alternative replaces or selects the official task score",
            "excluded_operators": {
                "raw_sum": "Translation- and group-size-sensitive across uncalibrated model scores.",
                "noisy_or": "Requires historically fitted probability calibration and independence assumptions; neither is introduced post hoc.",
            },
        },
        "two_stage_b1_b2_diagnostic": {
            "per_seed": two_stage_rows,
            "per_chain_seed_summary": two_stage_chain,
            "fixed_six_chain": two_stage_fixed,
            "status": "secondary_composition_diagnostic_not_an_official_task_redefinition",
            "denominator": (
                "All realized positive B1 exporter-stage entries. A missed entry contributes zero; "
                "conditional destination recall/value is computed only after the fixed B1 gate."
            ),
            "family_composition": (
                "Each family uses its separately frozen task-specific B1 and B2 reference scores at the same seed."
            ),
        },
        "eligibility_threshold_cohort_geometry": dict(config["eligibility_threshold_geometry"]),
    }
    _analysis_inventory(config, analysis)
    return analysis


def _csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(**values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        for key, value in values.items():
            if key not in row:
                _fail(f"unknown CSV field generated: {key}")
            row[key] = value
        rows.append(row)

    for record in analysis["paired_family_comparison"]["per_chain"]:
        for seed in record["per_seed"]:
            add(
                section="paired_family_difference",
                aggregation_scope="chain_seed",
                chain=record["chain"],
                task=record["task"],
                seed=seed["seed"],
                metric=record["metric"],
                point=seed["delta"],
                lower_95=seed["lower_95"],
                upper_95=seed["upper_95"],
                n=seed["finite_draws"],
                cluster_unit=record["cluster_unit"],
                bootstrap_seed=record["bootstrap_rng_seed"],
                note=PAIR_DIRECTION,
            )
        fixed = record["fixed_five_seed_mean"]
        add(
            section="paired_family_difference",
            aggregation_scope="chain_fixed_five_seed_mean",
            chain=record["chain"],
            task=record["task"],
            seed="mean_0_4",
            metric=record["metric"],
            point=fixed["point"],
            lower_95=fixed["lower_95"],
            upper_95=fixed["upper_95"],
            n=fixed["finite_draws"],
            cluster_unit=record["cluster_unit"],
            bootstrap_seed=record["bootstrap_rng_seed"],
            note=PAIR_DIRECTION,
        )
    for record in analysis["paired_family_comparison"]["fixed_six_chain"]:
        add(
            section="paired_family_difference",
            aggregation_scope="fixed_six_chain_fixed_five_seed_mean",
            chain="fixed_six",
            task=record["task"],
            seed="mean_0_4",
            metric=record["metric"],
            point=record["point"],
            lower_95=record["lower_95"],
            upper_95=record["upper_95"],
            n=record["finite_draws"],
            cluster_unit="stratified_within_chain",
            note=record["inference_scope"],
        )

    for section_name, payload_name, key_fields in (
        (
            "budget_summary",
            "budget_summaries",
            ("chain", "task", "family", "budget_scope", "requested_k"),
        ),
        (
            "b1_pooling_sensitivity",
            "b1_pooling_sensitivity",
            ("chain", "family", "method", "normative"),
        ),
        (
            "two_stage_b1_b2",
            "two_stage_b1_b2_diagnostic",
            ("chain", "family", "b1_budget", "b2_budget"),
        ),
    ):
        for record in analysis[payload_name]["per_chain_seed_summary"]:
            for metric, summary in record["metrics"].items():
                add(
                    section=section_name,
                    aggregation_scope="chain_fixed_five_seed_mean",
                    chain=record.get("chain", ""),
                    task=record.get("task", "b1" if section_name == "b1_pooling_sensitivity" else "b1_to_b2" if section_name == "two_stage_b1_b2" else ""),
                    family=record.get("family", ""),
                    seed="mean_0_4",
                    method=record.get("method", ""),
                    budget_scope=record.get("budget_scope", ""),
                    b1_budget=record.get("b1_budget", ""),
                    b2_budget=record.get("b2_budget", ""),
                    requested_k=record.get("requested_k", ""),
                    metric=metric,
                    point=summary["mean"],
                    std=summary["std"],
                    n=summary["n"],
                    note="descriptive fixed-seed summary",
                )
        for record in analysis[payload_name]["fixed_six_chain"]:
            for metric, point in record["metrics"].items():
                add(
                    section=section_name,
                    aggregation_scope="fixed_six_chain_fixed_five_seed_mean",
                    chain="fixed_six",
                    task=record.get("task", "b1" if section_name == "b1_pooling_sensitivity" else "b1_to_b2" if section_name == "two_stage_b1_b2" else ""),
                    family=record.get("family", ""),
                    seed="mean_0_4",
                    method=record.get("method", ""),
                    budget_scope=record.get("budget_scope", ""),
                    b1_budget=record.get("b1_budget", ""),
                    b2_budget=record.get("b2_budget", ""),
                    requested_k=record.get("requested_k", ""),
                    metric=metric,
                    point=point,
                    n=record["chain_count"],
                    note="unweighted six-chain descriptive mean; not population inference",
                )

    threshold = analysis["eligibility_threshold_cohort_geometry"]
    add(
        section="eligibility_threshold_cohort_geometry",
        aggregation_scope="not_computed",
        metric="status",
        note=f"{threshold['status']}: {threshold['reason']}",
    )
    return rows


def _csv_bytes(analysis: Mapping[str, Any]) -> tuple[bytes, int]:
    rows = _csv_rows(analysis)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8"), len(rows)


def _provenance(config_path: Path, receipts: ReceiptBook) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    return {
        "config": {
            "path": _relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "runner": {
            "path": _relative(runner),
            "sha256": sha256_file(runner),
        },
        "inputs": receipts.records(),
        "input_file_count": len(receipts.records()),
    }


def generate(config_path: Path) -> tuple[Path, Path]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)
    receipts = ReceiptBook()
    analysis = build_analysis(config, receipts)
    csv_data, csv_rows = _csv_bytes(analysis)
    csv_path = _resolve_role(config["output_csv"])
    json_path = _resolve_role(config["output_json"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_data)
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "complete_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance(config_path, receipts),
        "analysis": analysis,
        "csv_receipt": {
            "path": _relative(csv_path),
            "sha256": sha256_bytes(csv_data),
            "row_count": csv_rows,
            "columns": list(CSV_FIELDS),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    verify(config_path, json_path)
    return json_path, csv_path


def verify(config_path: Path, output_path: Path | None = None) -> None:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)
    output_path = output_path.resolve() if output_path else _resolve_role(config["output_json"])
    payload = load_json(output_path)
    _require_exact_keys(
        payload,
        {"schema_version", "status", "generated_at_utc", "provenance", "analysis", "csv_receipt"},
        "r5 output",
    )
    if payload["schema_version"] != OUTPUT_SCHEMA or payload["status"] != "complete_verified":
        _fail("r5 output schema or status changed")
    try:
        datetime.fromisoformat(str(payload["generated_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise R5Error("r5 output timestamp is invalid") from exc
    _require_exact_keys(payload["provenance"], {"config", "runner", "inputs", "input_file_count"}, "provenance")
    if payload["provenance"]["config"] != {
        "path": _relative(config_path),
        "sha256": sha256_file(config_path),
    }:
        _fail("r5 config provenance is stale")
    runner_path = Path(__file__).resolve()
    if payload["provenance"]["runner"] != {
        "path": _relative(runner_path),
        "sha256": sha256_file(runner_path),
    }:
        _fail("r5 runner provenance is stale")

    receipts = ReceiptBook()
    recomputed = build_analysis(config, receipts)
    if canonical_json(payload["analysis"]) != canonical_json(recomputed):
        _fail("r5 analysis does not exactly recompute from current governed inputs")
    expected_provenance = _provenance(config_path, receipts)
    if canonical_json(payload["provenance"]) != canonical_json(expected_provenance):
        _fail("r5 input provenance inventory or hash is stale")
    _analysis_inventory(config, payload["analysis"])

    csv_data, row_count = _csv_bytes(payload["analysis"])
    expected_receipt = {
        "path": config["output_csv"],
        "sha256": sha256_bytes(csv_data),
        "row_count": row_count,
        "columns": list(CSV_FIELDS),
    }
    if payload["csv_receipt"] != expected_receipt:
        _fail("r5 CSV receipt does not match deterministic JSON projection")
    csv_path = _resolve_role(config["output_csv"])
    if not csv_path.is_file() or csv_path.read_bytes() != csv_data:
        _fail("r5 CSV bytes do not match deterministic JSON projection")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="recompute and verify the existing JSON/CSV instead of generating them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_output:
            verify(args.config)
            print(f"VERIFIED: {_relative(_resolve_role(load_json(args.config)['output_json']))}")
        else:
            json_path, csv_path = generate(args.config)
            print(f"WROTE: {_relative(json_path)}")
            print(f"WROTE: {_relative(csv_path)}")
            print("VERIFIED: score-only r5 robustness")
        return 0
    except R5Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
