#!/usr/bin/env python3
"""Strict rolling CPU baselines for UPGRADE-BENCH v2.

Protocol enforced by this script
--------------------------------
1. Read only the historical ``*_fold2.csv`` tables.
2. Select logistic-regression regularization with grouped CV on that fold:
   exporter-grouped average precision for A/B1 and entry-grouped per-positive-
   entry macro recall@3 for B2.
3. Refit preprocessing and the classifier on the complete historical fold.
4. Only then read the main target-window tables and evaluate them in full.

The same target cohort is used for every scorer.  ``transductive_split`` is
intentionally ignored: it is not part of the rolling evaluation protocol.

Track B2 is a conditional ranking evaluation.  Scores are produced for every
main-window Track-B lane before target labels are inspected; metrics are then
computed only for exporter-stage groups that actually entered.  This oracle
conditioning defines the task and is not an end-to-end deployable gate.

Average-precision intervals resample exporters for both A and B1.  B2
recall/value intervals resample complete exporter-stage entries.  The macro
summary fixes global top-500 (A), global top-50 (B1), and per-entry top-3 (B2)
realized-value points before any main-window evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed_v2"
DEFAULT_JSON = ROOT / "results_v2" / "metrics" / "rolling_cpu_baselines.json"
DEFAULT_CSV = ROOT / "results_v2" / "metrics" / "rolling_cpu_baselines.csv"
SCHEMA_VERSION = "upgrade-bench-v2-rolling-cpu-baselines-2"

CHAINS = (
    "sheep",
    "cotton",
    "aluminium",
    "nickel",
    "cocoa",
    "oilseed-soy",
)
C_GRID = (0.01, 0.1, 1.0, 10.0)

TRACK_A_FEATURES = ("size", "log_gravity")
TRACK_B1_FEATURES = (
    "log_upstream_capacity",
    "log_candidate_destinations",
    "max_log_importer_demand",
    "mean_log_importer_demand",
    "max_log_gravity",
    "mean_log_gravity",
    "logsum_gravity",
    "gravity_coverage",
)
TRACK_B2_FEATURES = ("log_importer_demand", "log_gravity")

# These reporting points are protocol constants, not choices made after opening
# the main-window labels.  The middle registered global budget is used for A
# and B1; B2 uses the task's headline per-entry cutoff.
TRACK_A_VALUE_BUDGET = 500
TRACK_B1_VALUE_BUDGET = 50
TRACK_B2_VALUE_K = 3

TRACK_MODEL_ORDER = {
    "track_a_destination_extension": (
        "size",
        "gravity",
        "historical_logistic_size_gravity",
    ),
    "track_b1_processed_export_stage_entry": (
        "upstream_capacity",
        "historical_logistic_structural",
    ),
    "track_b2_conditional_destination_ranking": (
        "processed_importer_demand",
        "gravity",
        "historical_logistic_demand_gravity",
    ),
}


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas values into strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _entry_id(frame: pd.DataFrame) -> pd.Series:
    return frame["i_iso"].astype(str) + "|" + frame["stage"].astype(str)


def _log_gravity(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bool((numeric.dropna() < 0).any()):
        raise ValueError("gravity must be non-negative")
    return np.log1p(numeric)


def _validate_candidate_table(
    frame: pd.DataFrame,
    *,
    path: Path,
    track: str,
    expected_early: str,
    expected_late: str,
) -> pd.DataFrame:
    required = {
        "i_iso",
        "j_iso",
        "stage",
        "y",
        "size",
        "log_exporter_capacity",
        "log_importer_demand",
        "size_basis",
        "grav",
        "lateval",
        "benchmark_version",
        "aggregation",
        "early_window",
        "late_window",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    duplicate = frame.duplicated(["i_iso", "j_iso", "stage"], keep=False)
    if bool(duplicate.any()):
        example = frame.loc[duplicate, ["i_iso", "j_iso", "stage"]].head(3)
        raise ValueError(f"{path}: duplicate lane keys\n{example}")

    y = pd.to_numeric(frame["y"], errors="raise")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{path}: y is not binary")
    lateval = pd.to_numeric(frame["lateval"], errors="raise")
    if bool((lateval < 0).any()):
        raise ValueError(f"{path}: lateval must be non-negative")
    if bool(((y == 0) & (lateval != 0)).any()):
        raise ValueError(f"{path}: negative lanes must have zero lateval")
    if bool(((y == 1) & (lateval <= 0)).any()):
        raise ValueError(f"{path}: positive lanes must have positive lateval")

    singleton_expectations = {
        "aggregation": "calendar_mean",
        "early_window": expected_early,
        "late_window": expected_late,
    }
    for column, expected in singleton_expectations.items():
        values = set(frame[column].astype(str).unique())
        if values != {expected}:
            raise ValueError(f"{path}: expected {column}={expected!r}, got {sorted(values)}")

    versions = set(frame["benchmark_version"].astype(str).unique())
    if not versions or any(not version.startswith("2.") for version in versions):
        raise ValueError(f"{path}: expected a v2 benchmark_version, got {sorted(versions)}")

    expected_basis = {
        "a": "processed_exporter_plus_processed_importer",
        "b": "registered_upstream_exporter_plus_processed_importer",
    }[track]
    basis = set(frame["size_basis"].astype(str).unique())
    if basis != {expected_basis}:
        raise ValueError(f"{path}: expected size_basis={expected_basis!r}, got {sorted(basis)}")

    numeric_columns = (
        "size",
        "log_exporter_capacity",
        "log_importer_demand",
        "lateval",
    )
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {column} contains a non-finite value")

    result = frame.copy()
    result["y"] = y.astype(np.int8)
    result["lateval"] = lateval.astype(float)
    result["entry_id"] = _entry_id(result)
    # Stable input order makes all tie breaking label-independent and reproducible.
    return result.sort_values(["i_iso", "stage", "j_iso"], kind="mergesort").reset_index(drop=True)


def _read_candidate(
    data_dir: Path,
    chain: str,
    *,
    track: str,
    historical: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = "candidates" if track == "a" else "candidates_firsttime"
    suffix = "_fold2" if historical else ""
    path = data_dir / f"{prefix}_{chain}{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    expected_early, expected_late = (
        ("1998-2002", "2008-2012") if historical else ("2008-2012", "2018-2022")
    )
    frame = pd.read_csv(path)
    frame = _validate_candidate_table(
        frame,
        path=path,
        track=track,
        expected_early=expected_early,
        expected_late=expected_late,
    )
    audit = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "positive_lanes": int(frame["y"].sum()),
        "exporter_stage_groups": int(frame["entry_id"].nunique()),
        "missing_gravity": int(frame["grav"].isna().sum()),
        "early_window": expected_early,
        "late_window": expected_late,
    }
    return frame, audit


def _track_a_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "size": pd.to_numeric(frame["size"], errors="raise"),
            "log_gravity": _log_gravity(frame["grav"]),
        },
        index=frame.index,
    )


def _track_b2_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_importer_demand": pd.to_numeric(
                frame["log_importer_demand"], errors="raise"
            ),
            "log_gravity": _log_gravity(frame["grav"]),
        },
        index=frame.index,
    )


def _derive_entry_table(lanes: pd.DataFrame) -> pd.DataFrame:
    work = lanes.copy()
    work["log_gravity"] = _log_gravity(work["grav"])

    # The upstream capacity is an exporter-stage feature and must be constant
    # across candidate destinations.  Check this rather than silently averaging.
    capacity_span = work.groupby("entry_id")["log_exporter_capacity"].agg(
        lambda x: float(x.max() - x.min())
    )
    if bool((capacity_span > 1e-10).any()):
        bad = capacity_span[capacity_span > 1e-10].head(3).to_dict()
        raise ValueError(f"Track-B upstream capacity varies within an entry: {bad}")

    grouped = work.groupby(["i_iso", "stage", "entry_id"], sort=True, observed=True)
    entries = grouped.agg(
        z=("y", "max"),
        entry_lateval=("lateval", "sum"),
        log_upstream_capacity=("log_exporter_capacity", "first"),
        n_candidate_destinations=("j_iso", "size"),
        n_materialized_destinations=("y", "sum"),
        max_log_importer_demand=("log_importer_demand", "max"),
        mean_log_importer_demand=("log_importer_demand", "mean"),
        max_log_gravity=("log_gravity", "max"),
        mean_log_gravity=("log_gravity", "mean"),
        sum_gravity=("grav", "sum"),
        gravity_coverage=("grav", lambda x: float(x.notna().mean())),
    ).reset_index()
    entries["log_candidate_destinations"] = np.log1p(
        entries["n_candidate_destinations"].to_numpy(float)
    )
    entries["logsum_gravity"] = np.log1p(entries.pop("sum_gravity").to_numpy(float))
    entries["z"] = entries["z"].astype(np.int8)
    return entries


def _pipeline(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _valid_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    maximum: int = 5,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    n_groups = int(pd.Series(groups).nunique())
    for n_splits in range(min(maximum, n_groups), 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        if all(np.unique(y[train]).size == 2 and np.unique(y[valid]).size == 2
               for train, valid in splits):
            return splits, n_splits
    raise ValueError("could not construct grouped CV folds with both classes in every partition")


def _positive_entry_macro_recall_at_k(
    y: np.ndarray,
    score: np.ndarray,
    entry_groups: Sequence[str],
    tie_break_keys: Sequence[str],
    *,
    k: int = TRACK_B2_VALUE_K,
) -> float:
    """Return unweighted recall@k over positive exporter-stage entries.

    This is both the Track-B2 historical model-selection objective and its
    main headline metric.  Every supplied group must be a positive entry;
    silently dropping a zero-positive validation group would change the CV
    estimand.  Exact score ties are resolved by the canonical destination key,
    matching main evaluation.
    """
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(entry_groups, dtype=str)
    keys = np.asarray(tie_break_keys, dtype=str)
    if k <= 0:
        raise ValueError("k must be positive")
    if not (len(y) == len(score) == len(groups) == len(keys)) or not len(y):
        raise ValueError("B2 objective arrays are empty or misaligned")
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(score).all():
        raise ValueError("B2 objective requires binary labels and finite scores")

    work = pd.DataFrame(
        {"entry_group": groups, "tie_break_key": keys, "y": y, "score": score}
    )
    recalls: list[float] = []
    for entry_group, group in work.groupby("entry_group", sort=True, observed=True):
        positives = int(group["y"].sum())
        if positives <= 0:
            raise ValueError(
                f"B2 selection cohort contains zero-positive entry {entry_group!r}"
            )
        selected = group.sort_values(
            ["score", "tie_break_key"],
            ascending=[False, True],
            kind="mergesort",
        ).head(k)
        recalls.append(float(selected["y"].sum() / positives))
    return float(np.mean(recalls))


@dataclass
class FrozenClassifier:
    pipeline: Pipeline
    selection: dict[str, Any]


def _select_and_fit(
    features: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    group_unit: str,
    seed: int,
    objective: str = "average_precision",
    objective_groups: Sequence[str] | None = None,
    objective_tie_break_keys: Sequence[str] | None = None,
) -> FrozenClassifier:
    x = features.loc[:, list(feature_names)]
    if np.unique(y).size != 2:
        raise ValueError("historical training labels contain only one class")
    splits, n_splits = _valid_group_splits(y, groups, seed=seed)
    for train, valid in splits:
        overlap = set(groups[train]).intersection(groups[valid])
        if overlap:
            raise AssertionError(f"grouped CV leaked groups: {sorted(overlap)[:3]}")

    if objective == "average_precision":
        objective_name = "historical_group_cv_average_precision"
        objective_definition = "lane_or_entry_average_precision_on_each_validation_fold"
        objective_aggregation = "unweighted_mean_over_validation_folds"
    elif objective == "positive_entry_macro_recall_at_3":
        if objective_groups is None or objective_tie_break_keys is None:
            raise ValueError("B2 recall selection requires entry groups and tie-break keys")
        if len(objective_groups) != len(y) or len(objective_tie_break_keys) != len(y):
            raise ValueError("B2 objective metadata is misaligned")
        objective_name = "historical_group_cv_per_positive_entry_macro_recall_at_3"
        objective_definition = (
            "within each positive exporter-stage validation entry, rank destinations and "
            "compute recall@3; then take the unweighted mean over all out-of-fold "
            "validation entries"
        )
        objective_aggregation = (
            "fold means weighted by validation-entry count, exactly equivalent to the "
            "unweighted macro mean over all out-of-fold positive entries"
        )
    else:
        raise ValueError(f"unknown selection objective: {objective!r}")

    candidates: list[dict[str, Any]] = []
    for c_value in C_GRID:
        fold_scores: list[float] = []
        fold_objective_units: list[int] = []
        for train, valid in splits:
            model = _pipeline(c_value, seed)
            model.fit(x.iloc[train], y[train])
            score = model.predict_proba(x.iloc[valid])[:, 1]
            if objective == "average_precision":
                fold_value = float(average_precision_score(y[valid], score))
                fold_units = 1
            else:
                assert objective_groups is not None
                assert objective_tie_break_keys is not None
                fold_value = _positive_entry_macro_recall_at_k(
                    y[valid],
                    score,
                    np.asarray(objective_groups, dtype=str)[valid],
                    np.asarray(objective_tie_break_keys, dtype=str)[valid],
                    k=TRACK_B2_VALUE_K,
                )
                fold_units = int(
                    pd.Series(np.asarray(objective_groups, dtype=str)[valid]).nunique()
                )
            fold_scores.append(fold_value)
            fold_objective_units.append(fold_units)
        mean_objective = (
            float(np.average(fold_scores, weights=fold_objective_units))
            if objective == "positive_entry_macro_recall_at_3"
            else float(np.mean(fold_scores))
        )
        candidates.append(
            {
                "C": float(c_value),
                "fold_objective_values": fold_scores,
                "fold_objective_units": fold_objective_units,
                "mean_objective": mean_objective,
                "std_objective": float(np.std(fold_scores, ddof=0)),
            }
        )

    # Tie break toward the stronger regularizer (smaller C); no target-window
    # observation participates in this choice.
    selected = min(candidates, key=lambda row: (-row["mean_objective"], row["C"]))
    final_model = _pipeline(selected["C"], seed)
    final_model.fit(x, y)
    selection = {
        "feature_names": list(feature_names),
        "objective": objective_name,
        "objective_definition": objective_definition,
        "objective_aggregation": objective_aggregation,
        "objective_direction": "maximize",
        "group_unit": group_unit,
        "train_validation_group_overlap_checked": True,
        "n_splits": n_splits,
        "c_grid": list(C_GRID),
        "candidates": candidates,
        "selected_C": selected["C"],
        "selected_mean_objective": selected["mean_objective"],
        "hyperparameter_tie_break": "maximize_mean_objective_then_smaller_C",
        "ranking_tie_break": (
            "destination_iso_ascending_within_entry_for_exact_score_ties"
            if objective == "positive_entry_macro_recall_at_3"
            else "average_precision_is_score_tie_block_invariant"
        ),
        "refit_rows": int(len(y)),
        "refit_positives": int(y.sum()),
    }
    return FrozenClassifier(final_model, selection)


@dataclass
class FrozenRawScore:
    feature: str
    fill_value: float | None

    @classmethod
    def fit(cls, values: pd.Series, *, feature: str) -> "FrozenRawScore":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
        finite = numeric[np.isfinite(numeric)]
        if not len(finite):
            raise ValueError(f"historical raw feature {feature} has no finite observations")
        fill = float(np.median(finite)) if len(finite) != len(numeric) else None
        return cls(feature=feature, fill_value=fill)

    def predict(self, values: pd.Series) -> np.ndarray:
        score = pd.to_numeric(values, errors="coerce").to_numpy(float)
        if self.fill_value is not None:
            score = np.where(np.isfinite(score), score, self.fill_value)
        if not np.isfinite(score).all():
            raise ValueError(f"target raw feature {self.feature} contains non-finite values")
        return score

    def audit(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "missing_value_policy": (
                "historical_median" if self.fill_value is not None else "not_needed"
            ),
            "historical_fill_value": self.fill_value,
        }


def _weighted_average_precision(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Average precision with sample weights and exact score-tie handling."""
    order = np.argsort(-score, kind="mergesort")
    ranked_score = score[order]
    ranked_y = y[order]
    ranked_weight = weight[order]
    positive_weight = ranked_weight * ranked_y
    total_positive = float(positive_weight.sum())
    if total_positive <= 0 or float((ranked_weight * (1 - ranked_y)).sum()) <= 0:
        return float("nan")

    cumulative_positive = np.cumsum(positive_weight)
    cumulative_weight = np.cumsum(ranked_weight)
    tie_ends = np.r_[ranked_score[1:] != ranked_score[:-1], True]
    tp_at_threshold = cumulative_positive[tie_ends]
    total_at_threshold = cumulative_weight[tie_ends]
    delta_tp = np.diff(np.r_[0.0, tp_at_threshold])
    precision = np.divide(
        tp_at_threshold,
        total_at_threshold,
        out=np.zeros_like(tp_at_threshold),
        where=total_at_threshold > 0,
    )
    return float(np.sum((delta_tp / total_positive) * precision))


def _cluster_ap_ci(
    y: np.ndarray,
    score: np.ndarray,
    clusters: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> list[float] | None:
    if draws <= 0:
        return None
    codes, uniques = pd.factorize(pd.Series(clusters, dtype="string"), sort=True)
    n_clusters = len(uniques)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(sampled, minlength=n_clusters).astype(float)
        value = _weighted_average_precision(y, score, counts[codes])
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < max(20, draws // 2):
        return None
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y).size != 2:
        return None
    return float(roc_auc_score(y, score))


def _budget_metrics(
    y: np.ndarray,
    lateval: np.ndarray,
    score: np.ndarray,
    budgets: Iterable[int],
) -> dict[str, Any]:
    order = np.argsort(-score, kind="mergesort")
    positive_count = int(y.sum())
    positive_value = float(lateval.sum())
    result: dict[str, Any] = {}
    for requested in budgets:
        effective = min(int(requested), len(y))
        selected = order[:effective]
        hits = int(y[selected].sum())
        captured = float(lateval[selected].sum())
        result[f"k_{requested}"] = {
            "requested_k": int(requested),
            "effective_k": effective,
            "hits": hits,
            "precision": float(hits / effective) if effective else None,
            "recall": float(hits / positive_count) if positive_count else None,
            "observed_late_value_kusd": captured,
            "value_capture": float(captured / positive_value) if positive_value else None,
        }
    return result


def _exporter_shortlists(
    frame: pd.DataFrame,
    score: np.ndarray,
    ks: Iterable[int] = (5, 10),
) -> dict[str, Any]:
    work = frame.loc[:, ["i_iso", "y", "lateval"]].copy()
    work["score"] = score
    result: dict[str, Any] = {}
    for k in ks:
        recalls: list[float] = []
        precisions: list[float] = []
        value_shares: list[float] = []
        total_hits = 0
        total_positives = 0
        selected_value = 0.0
        total_value = 0.0
        for _, group in work.groupby("i_iso", sort=True, observed=True):
            selected = group.sort_values("score", ascending=False, kind="mergesort").head(k)
            hits = int(selected["y"].sum())
            positives = int(group["y"].sum())
            captured = float(selected["lateval"].sum())
            available = float(group["lateval"].sum())
            precisions.append(float(hits / len(selected)))
            if positives:
                recalls.append(float(hits / positives))
            if available:
                value_shares.append(float(captured / available))
            total_hits += hits
            total_positives += positives
            selected_value += captured
            total_value += available
        result[f"k_{k}_per_exporter"] = {
            "exporters": int(work["i_iso"].nunique()),
            "exporters_with_positive": len(recalls),
            "macro_precision": float(np.mean(precisions)) if precisions else None,
            "macro_recall_positive_exporters": float(np.mean(recalls)) if recalls else None,
            "micro_recall": float(total_hits / total_positives) if total_positives else None,
            "macro_value_capture_positive_exporters": (
                float(np.mean(value_shares)) if value_shares else None
            ),
            "micro_value_capture": float(selected_value / total_value) if total_value else None,
        }
    return result


def _classification_metrics(
    frame: pd.DataFrame,
    *,
    label: str,
    score: np.ndarray,
    cluster: Sequence[str],
    cluster_unit: str,
    budgets: Iterable[int],
    bootstrap_draws: int,
    seed: int,
    exporter_shortlists: bool = False,
) -> dict[str, Any]:
    y = frame[label].to_numpy(np.int8)
    lateval_column = "lateval" if label == "y" else "entry_lateval"
    lateval = frame[lateval_column].to_numpy(float)
    if len(score) != len(y) or not np.isfinite(score).all():
        raise ValueError("score vector is misaligned or non-finite")
    ap = float(average_precision_score(y, score))
    metrics: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "average_precision": ap,
        "average_precision_ci95": _cluster_ap_ci(
            y,
            score,
            cluster,
            draws=bootstrap_draws,
            seed=seed,
        ),
        "roc_auc": _safe_roc_auc(y, score),
        "total_observed_late_value_kusd": float(lateval.sum()),
        "budgets": _budget_metrics(y, lateval, score, budgets),
        "uncertainty": {
            "method": "nonparametric_cluster_bootstrap",
            "cluster_unit": cluster_unit,
            "n_clusters": int(pd.Series(cluster, dtype="string").nunique()),
            "draws": int(bootstrap_draws),
            "seed": int(seed),
            "interval": "percentile_95",
            "ap_only": True,
        },
    }
    if exporter_shortlists:
        metrics["per_exporter_shortlists"] = _exporter_shortlists(frame, score)
    return metrics


def _conditional_group_statistics(
    frame: pd.DataFrame,
    score: np.ndarray,
    ks: Sequence[int],
) -> pd.DataFrame:
    work = frame.loc[:, ["entry_id", "j_iso", "y", "lateval"]].copy()
    work["score"] = score
    rows: list[dict[str, Any]] = []
    for entry_id, group in work.groupby("entry_id", sort=True, observed=True):
        positives = int(group["y"].sum())
        total_value = float(group["lateval"].sum())
        if positives <= 0 or total_value <= 0:
            raise ValueError("Track B2 conditional cohort contains a non-entry group")
        order = group.sort_values(
            ["score", "j_iso"], ascending=[False, True], kind="mergesort"
        )
        row: dict[str, Any] = {
            "entry_id": entry_id,
            "n_candidates": int(len(group)),
            "positives": positives,
            "total_value": total_value,
        }
        for k in ks:
            selected = order.head(k)
            hits = int(selected["y"].sum())
            captured = float(selected["lateval"].sum())
            row[f"hits_{k}"] = hits
            row[f"recall_{k}"] = float(hits / positives)
            row[f"selected_value_{k}"] = captured
            row[f"value_capture_{k}"] = float(captured / total_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _conditional_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    bootstrap_draws: int,
    seed: int,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    stats = _conditional_group_statistics(frame, score, ks)
    result: dict[str, Any] = {
        "n_entry_groups": int(len(stats)),
        "n_candidate_lanes": int(len(frame)),
        "positive_lanes": int(frame["y"].sum()),
        "total_observed_late_value_kusd": float(frame["lateval"].sum()),
        "oracle_conditioning_for_evaluation": True,
        "at_k": {},
        "uncertainty": {
            "method": "nonparametric_entry_group_bootstrap",
            "cluster_unit": "exporter_stage_entry",
            "cluster_key": "entry_id",
            "n_clusters": int(len(stats)),
            "draws": int(bootstrap_draws),
            "seed": int(seed),
            "interval": "percentile_95",
        },
    }

    rng = np.random.default_rng(seed)
    bootstrap_indices = (
        rng.integers(0, len(stats), size=(bootstrap_draws, len(stats)))
        if bootstrap_draws > 0
        else None
    )
    for k in ks:
        recall = stats[f"recall_{k}"].to_numpy(float)
        value_capture = stats[f"value_capture_{k}"].to_numpy(float)
        hits = stats[f"hits_{k}"].to_numpy(float)
        positives = stats["positives"].to_numpy(float)
        selected_value = stats[f"selected_value_{k}"].to_numpy(float)
        total_value = stats["total_value"].to_numpy(float)
        metric = {
            "macro_recall": float(recall.mean()),
            "micro_recall": float(hits.sum() / positives.sum()),
            "macro_value_capture": float(value_capture.mean()),
            "micro_value_capture": float(selected_value.sum() / total_value.sum()),
            "macro_recall_ci95": None,
            "macro_value_capture_ci95": None,
        }
        if bootstrap_indices is not None:
            recall_draws = recall[bootstrap_indices].mean(axis=1)
            value_draws = value_capture[bootstrap_indices].mean(axis=1)
            metric["macro_recall_ci95"] = [
                float(x) for x in np.quantile(recall_draws, [0.025, 0.975])
            ]
            metric["macro_value_capture_ci95"] = [
                float(x) for x in np.quantile(value_draws, [0.025, 0.975])
            ]
        result["at_k"][f"k_{k}"] = metric
    return result


@dataclass
class ChainModels:
    track_a_raw_size: FrozenRawScore
    track_a_raw_gravity: FrozenRawScore
    track_a_classifier: FrozenClassifier
    track_b1_raw_capacity: FrozenRawScore
    track_b1_classifier: FrozenClassifier
    track_b2_raw_demand: FrozenRawScore
    track_b2_raw_gravity: FrozenRawScore
    track_b2_classifier: FrozenClassifier
    history_a_audit: dict[str, Any]
    history_b_audit: dict[str, Any]
    history_b1_entries: int
    history_b1_positive_entries: int
    history_b2_lanes: int
    history_b2_positive_lanes: int


def _fit_chain(data_dir: Path, chain: str, seed: int) -> ChainModels:
    # This function is the protocol boundary: it has no code path that opens a
    # main-window filename.
    history_a, audit_a = _read_candidate(
        data_dir, chain, track="a", historical=True
    )
    history_b, audit_b = _read_candidate(
        data_dir, chain, track="b", historical=True
    )

    a_features = _track_a_features(history_a)
    a_y = history_a["y"].to_numpy(np.int8)
    a_classifier = _select_and_fit(
        a_features,
        a_y,
        history_a["i_iso"].to_numpy(str),
        feature_names=TRACK_A_FEATURES,
        group_unit="exporter",
        seed=seed + 11,
        objective="average_precision",
    )
    a_size = FrozenRawScore.fit(a_features["size"], feature="size")
    a_gravity = FrozenRawScore.fit(a_features["log_gravity"], feature="log_gravity")

    history_entries = _derive_entry_table(history_b)
    b1_y = history_entries["z"].to_numpy(np.int8)
    b1_classifier = _select_and_fit(
        history_entries,
        b1_y,
        history_entries["i_iso"].to_numpy(str),
        feature_names=TRACK_B1_FEATURES,
        group_unit="exporter",
        seed=seed + 21,
        objective="average_precision",
    )
    b1_capacity = FrozenRawScore.fit(
        history_entries["log_upstream_capacity"], feature="log_upstream_capacity"
    )

    positive_entry_ids = set(
        history_entries.loc[history_entries["z"].eq(1), "entry_id"].astype(str)
    )
    history_b2 = history_b.loc[history_b["entry_id"].isin(positive_entry_ids)].copy()
    b2_features = _track_b2_features(history_b2)
    b2_y = history_b2["y"].to_numpy(np.int8)
    b2_classifier = _select_and_fit(
        b2_features,
        b2_y,
        history_b2["entry_id"].to_numpy(str),
        feature_names=TRACK_B2_FEATURES,
        group_unit="exporter_stage_entry",
        seed=seed + 31,
        objective="positive_entry_macro_recall_at_3",
        objective_groups=history_b2["entry_id"].to_numpy(str),
        objective_tie_break_keys=history_b2["j_iso"].to_numpy(str),
    )
    b2_demand = FrozenRawScore.fit(
        b2_features["log_importer_demand"], feature="log_importer_demand"
    )
    b2_gravity = FrozenRawScore.fit(
        b2_features["log_gravity"], feature="log_gravity"
    )

    return ChainModels(
        track_a_raw_size=a_size,
        track_a_raw_gravity=a_gravity,
        track_a_classifier=a_classifier,
        track_b1_raw_capacity=b1_capacity,
        track_b1_classifier=b1_classifier,
        track_b2_raw_demand=b2_demand,
        track_b2_raw_gravity=b2_gravity,
        track_b2_classifier=b2_classifier,
        history_a_audit=audit_a,
        history_b_audit=audit_b,
        history_b1_entries=int(len(history_entries)),
        history_b1_positive_entries=int(history_entries["z"].sum()),
        history_b2_lanes=int(len(history_b2)),
        history_b2_positive_lanes=int(history_b2["y"].sum()),
    )


def _evaluate_chain(
    data_dir: Path,
    chain: str,
    models: ChainModels,
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    # Main target labels are first opened here, after every preprocessing step,
    # hyperparameter and classifier for this chain has been frozen.
    target_a, target_a_audit = _read_candidate(
        data_dir, chain, track="a", historical=False
    )
    target_b, target_b_audit = _read_candidate(
        data_dir, chain, track="b", historical=False
    )

    a_features = _track_a_features(target_a)
    a_scores = {
        "size": models.track_a_raw_size.predict(a_features["size"]),
        "gravity": models.track_a_raw_gravity.predict(a_features["log_gravity"]),
        "historical_logistic_size_gravity": models.track_a_classifier.pipeline.predict_proba(
            a_features.loc[:, list(TRACK_A_FEATURES)]
        )[:, 1],
    }
    a_model_audits = {
        "size": {"kind": "fixed_raw_ranking", **models.track_a_raw_size.audit()},
        "gravity": {"kind": "fixed_raw_ranking", **models.track_a_raw_gravity.audit()},
        "historical_logistic_size_gravity": {
            "kind": "historically_selected_supervised_baseline",
            "selection": models.track_a_classifier.selection,
        },
    }
    track_a_models: dict[str, Any] = {}
    for offset, (name, score) in enumerate(a_scores.items()):
        track_a_models[name] = {
            "model": a_model_audits[name],
            "metrics": _classification_metrics(
                target_a,
                label="y",
                score=score,
                cluster=target_a["i_iso"].to_numpy(str),
                cluster_unit="exporter",
                budgets=(100, 500, 1000),
                bootstrap_draws=bootstrap_draws,
                seed=seed + 100 + offset,
                exporter_shortlists=True,
            ),
        }

    target_entries = _derive_entry_table(target_b)
    b1_scores = {
        "upstream_capacity": models.track_b1_raw_capacity.predict(
            target_entries["log_upstream_capacity"]
        ),
        "historical_logistic_structural": models.track_b1_classifier.pipeline.predict_proba(
            target_entries.loc[:, list(TRACK_B1_FEATURES)]
        )[:, 1],
    }
    b1_model_audits = {
        "upstream_capacity": {
            "kind": "fixed_raw_ranking",
            **models.track_b1_raw_capacity.audit(),
        },
        "historical_logistic_structural": {
            "kind": "historically_selected_supervised_baseline",
            "selection": models.track_b1_classifier.selection,
        },
    }
    track_b1_models: dict[str, Any] = {}
    for offset, (name, score) in enumerate(b1_scores.items()):
        track_b1_models[name] = {
            "model": b1_model_audits[name],
            "metrics": _classification_metrics(
                target_entries,
                label="z",
                score=score,
                cluster=target_entries["i_iso"].to_numpy(str),
                cluster_unit="exporter",
                budgets=(25, 50, 100),
                bootstrap_draws=bootstrap_draws,
                seed=seed + 200 + offset,
            ),
        }

    # Produce B2 scores on every Track-B lane before using target entry labels
    # to define the conditional evaluation cohort.
    b2_features_all = _track_b2_features(target_b)
    b2_scores_all = {
        "processed_importer_demand": models.track_b2_raw_demand.predict(
            b2_features_all["log_importer_demand"]
        ),
        "gravity": models.track_b2_raw_gravity.predict(b2_features_all["log_gravity"]),
        "historical_logistic_demand_gravity": (
            models.track_b2_classifier.pipeline.predict_proba(
                b2_features_all.loc[:, list(TRACK_B2_FEATURES)]
            )[:, 1]
        ),
    }
    positive_target_entry_ids = set(
        target_entries.loc[target_entries["z"].eq(1), "entry_id"].astype(str)
    )
    conditional_mask = target_b["entry_id"].isin(positive_target_entry_ids).to_numpy(bool)
    target_b2 = target_b.loc[conditional_mask].reset_index(drop=True)
    b2_model_audits = {
        "processed_importer_demand": {
            "kind": "fixed_raw_conditional_ranking",
            **models.track_b2_raw_demand.audit(),
        },
        "gravity": {
            "kind": "fixed_raw_conditional_ranking",
            **models.track_b2_raw_gravity.audit(),
        },
        "historical_logistic_demand_gravity": {
            "kind": "historically_selected_conditional_ranker",
            "selection": models.track_b2_classifier.selection,
        },
    }
    track_b2_models: dict[str, Any] = {}
    for offset, (name, score_all) in enumerate(b2_scores_all.items()):
        track_b2_models[name] = {
            "model": b2_model_audits[name],
            "metrics": _conditional_metrics(
                target_b2,
                score_all[conditional_mask],
                bootstrap_draws=bootstrap_draws,
                seed=seed + 300 + offset,
            ),
        }

    return {
        "protocol_audit": {
            "target_loaded_after_all_models_frozen": True,
            "target_labels_used_for_training_selection_imputation_or_calibration": False,
            "transductive_split_used": False,
            "history_track_a": models.history_a_audit,
            "history_track_b": models.history_b_audit,
            "target_track_a": target_a_audit,
            "target_track_b": target_b_audit,
        },
        "track_a_destination_extension": {
            "unit": "exporter_stage_destination",
            "history_rows": models.history_a_audit["rows"],
            "history_positives": models.history_a_audit["positive_lanes"],
            "target_rows": target_a_audit["rows"],
            "target_positives": target_a_audit["positive_lanes"],
            "models": track_a_models,
        },
        "track_b1_processed_export_stage_entry": {
            "unit": "exporter_stage",
            "history_rows": models.history_b1_entries,
            "history_positives": models.history_b1_positive_entries,
            "target_rows": int(len(target_entries)),
            "target_positives": int(target_entries["z"].sum()),
            "models": track_b1_models,
        },
        "track_b2_conditional_destination_ranking": {
            "unit": "destination_within_actual_exporter_stage_entry",
            "conditioning": "actual_entry_in_corresponding_late_window",
            "history_rows": models.history_b2_lanes,
            "history_positives": models.history_b2_positive_lanes,
            "target_rows": int(len(target_b2)),
            "target_positives": int(target_b2["y"].sum()),
            "target_entry_groups": int(target_b2["entry_id"].nunique()),
            "models": track_b2_models,
        },
    }


def _paired_delta_report(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, Any]:
    if tuple(left) != tuple(right):
        raise ValueError("paired model metrics use different chain ordering")
    per_chain = {name: float(left[name] - right[name]) for name in left}
    values = np.asarray(list(per_chain.values()), dtype=float)
    ties = np.isclose(values, 0.0, rtol=0, atol=1e-12)
    return {
        "orientation": "left_minus_right",
        "descriptive_mean_delta": float(values.mean()),
        "descriptive_median_delta": float(np.median(values)),
        "per_chain": per_chain,
        "sign_counts": {
            "left_better": int(np.sum((values > 0) & ~ties)),
            "ties": int(np.sum(ties)),
            "right_better": int(np.sum((values < 0) & ~ties)),
        },
        "chain_level_ci95": None,
        "inference_note": (
            "descriptive paired differences over the frozen chain registry; no chain-level "
            "confidence interval is estimated from only these chains"
        ),
    }


def _macro_summary(chains: Mapping[str, Any]) -> dict[str, Any]:
    if not chains:
        raise ValueError("cannot summarize an empty chain result")
    definitions = {
        "track_a_destination_extension": {
            "headline_metric": "average_precision",
            "headline_getter": lambda metrics: metrics["average_precision"],
            "realized_value_metric": "global_observed_late_value_capture_at_500",
            "value_getter": lambda metrics: metrics["budgets"][
                f"k_{TRACK_A_VALUE_BUDGET}"
            ]["value_capture"],
            "budget_definition": {
                "scope": "within_chain_complete_main_cohort",
                "unit": "destination_lane",
                "selection": "global_top_k_by_model_score",
                "requested_k": TRACK_A_VALUE_BUDGET,
                "effective_k": "min(requested_k, chain_target_rows)",
                "value_denominator": "all_positive_observed_late_value_in_chain",
            },
        },
        "track_b1_processed_export_stage_entry": {
            "headline_metric": "average_precision",
            "headline_getter": lambda metrics: metrics["average_precision"],
            "realized_value_metric": "global_observed_late_value_capture_at_50",
            "value_getter": lambda metrics: metrics["budgets"][
                f"k_{TRACK_B1_VALUE_BUDGET}"
            ]["value_capture"],
            "budget_definition": {
                "scope": "within_chain_complete_main_cohort",
                "unit": "exporter_stage_entry",
                "selection": "global_top_k_by_model_score",
                "requested_k": TRACK_B1_VALUE_BUDGET,
                "effective_k": "min(requested_k, chain_target_rows)",
                "value_denominator": "all_positive_observed_late_value_in_chain",
            },
        },
        "track_b2_conditional_destination_ranking": {
            "headline_metric": "per_positive_entry_macro_recall_at_3",
            "headline_getter": lambda metrics: metrics["at_k"][
                f"k_{TRACK_B2_VALUE_K}"
            ]["macro_recall"],
            "realized_value_metric": "per_positive_entry_macro_value_capture_at_3",
            "value_getter": lambda metrics: metrics["at_k"][
                f"k_{TRACK_B2_VALUE_K}"
            ]["macro_value_capture"],
            "budget_definition": {
                "scope": "within_each_actual_positive_exporter_stage_entry",
                "unit": "destination_lane",
                "selection": "top_k_by_model_score_per_entry",
                "requested_k_per_entry": TRACK_B2_VALUE_K,
                "effective_k": "min(requested_k_per_entry, entry_candidate_lanes)",
                "value_denominator": "positive_observed_late_value_within_each_entry",
                "entry_aggregation": "unweighted_mean_over_positive_entries",
            },
        },
    }
    chain_names = tuple(chains)
    summary: dict[str, Any] = {}
    for track, definition in definitions.items():
        model_names = TRACK_MODEL_ORDER[track]
        for chain_name, chain in chains.items():
            observed_models = tuple(chain[track]["models"])
            if observed_models != model_names:
                raise ValueError(
                    f"{chain_name}/{track}: expected frozen model order {model_names}, "
                    f"got {observed_models}"
                )

        headline_by_model: dict[str, dict[str, float]] = {}
        value_by_model: dict[str, dict[str, float]] = {}
        track_summary: dict[str, Any] = {
            "metric": definition["headline_metric"],
            "realized_value_metric": definition["realized_value_metric"],
            "budget_definition": definition["budget_definition"],
            "aggregation": "unweighted_mean_over_chains",
            "chain_registry": list(chain_names),
            "models": {},
        }
        for model in model_names:
            headline = {
                chain_name: float(
                    definition["headline_getter"](
                        chain[track]["models"][model]["metrics"]
                    )
                )
                for chain_name, chain in chains.items()
            }
            realized_value = {
                chain_name: float(
                    definition["value_getter"](
                        chain[track]["models"][model]["metrics"]
                    )
                )
                for chain_name, chain in chains.items()
            }
            headline_by_model[model] = headline
            value_by_model[model] = realized_value
            track_summary["models"][model] = {
                # Preserve the v1 headline fields while adding an explicit value axis.
                "macro_mean": float(np.mean(list(headline.values()))),
                "per_chain": headline,
                "realized_value": {
                    "metric": definition["realized_value_metric"],
                    "macro_mean": float(np.mean(list(realized_value.values()))),
                    "per_chain": realized_value,
                },
            }

        comparisons_out: list[dict[str, Any]] = []
        for left_model, right_model in combinations(model_names, 2):
            comparisons_out.append(
                {
                    "left_model": left_model,
                    "right_model": right_model,
                    "headline": _paired_delta_report(
                        headline_by_model[left_model], headline_by_model[right_model]
                    ),
                    "realized_value": _paired_delta_report(
                        value_by_model[left_model], value_by_model[right_model]
                    ),
                }
            )
        track_summary["pairwise_deltas"] = {
            "pair_registry": "all_unordered_pairs_from_protocol_fixed_model_order",
            "frozen_before_main": True,
            "post_hoc_champion_selection": False,
            "model_order": list(model_names),
            "comparisons": comparisons_out,
        }
        summary[track] = track_summary
    return summary


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chain_name, chain in result["chains"].items():
        for track_name in (
            "track_a_destination_extension",
            "track_b1_processed_export_stage_entry",
            "track_b2_conditional_destination_ranking",
        ):
            track = chain[track_name]
            for model_name, payload in track["models"].items():
                metrics = payload["metrics"]
                row: dict[str, Any] = {
                    "chain": chain_name,
                    "track": track_name,
                    "model": model_name,
                    "history_rows": track["history_rows"],
                    "history_positives": track["history_positives"],
                    "target_rows": track["target_rows"],
                    "target_positives": track["target_positives"],
                    "selected_C": payload["model"].get("selection", {}).get("selected_C"),
                    "selection_objective": payload["model"].get("selection", {}).get(
                        "objective"
                    ),
                }
                if track_name != "track_b2_conditional_destination_ranking":
                    reporting_k = (
                        TRACK_A_VALUE_BUDGET
                        if track_name == "track_a_destination_extension"
                        else TRACK_B1_VALUE_BUDGET
                    )
                    ci = metrics["average_precision_ci95"] or (None, None)
                    row.update(
                        {
                            "headline_metric": "average_precision",
                            "headline_value": metrics["average_precision"],
                            "base_rate": metrics["base_rate"],
                            "average_precision": metrics["average_precision"],
                            "average_precision_ci95_low": ci[0],
                            "average_precision_ci95_high": ci[1],
                            "roc_auc": metrics["roc_auc"],
                            "realized_value_metric": (
                                f"global_observed_late_value_capture_at_{reporting_k}"
                            ),
                            "reporting_budget_k": reporting_k,
                            "realized_value": metrics["budgets"][f"k_{reporting_k}"][
                                "value_capture"
                            ],
                            "budgets_json": json.dumps(metrics["budgets"], sort_keys=True),
                        }
                    )
                else:
                    row.update(
                        {
                            "headline_metric": "per_positive_entry_macro_recall_at_3",
                            "headline_value": metrics["at_k"][
                                f"k_{TRACK_B2_VALUE_K}"
                            ]["macro_recall"],
                            "realized_value_metric": (
                                "per_positive_entry_macro_value_capture_at_3"
                            ),
                            "reporting_budget_k": TRACK_B2_VALUE_K,
                            "realized_value": metrics["at_k"][
                                f"k_{TRACK_B2_VALUE_K}"
                            ]["macro_value_capture"],
                        }
                    )
                    for k in (1, 3, 5):
                        at_k = metrics["at_k"][f"k_{k}"]
                        row[f"macro_recall_at_{k}"] = at_k["macro_recall"]
                        row[f"micro_recall_at_{k}"] = at_k["micro_recall"]
                        row[f"macro_value_capture_at_{k}"] = at_k["macro_value_capture"]
                        row[f"micro_value_capture_at_{k}"] = at_k["micro_value_capture"]
                rows.append(row)
    return rows


def _self_test() -> None:
    y = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    score = np.array([0.9, 0.7, 0.7, 0.2, 0.1, 0.1], dtype=float)
    unit_weight = np.ones(len(y), dtype=float)
    observed = _weighted_average_precision(y, score, unit_weight)
    expected = float(average_precision_score(y, score))
    if not np.isclose(observed, expected, rtol=0, atol=1e-12):
        raise AssertionError((observed, expected))

    lanes = pd.DataFrame(
        {
            "i_iso": ["A", "A", "B"],
            "j_iso": ["X", "Y", "X"],
            "stage": ["s", "s", "s"],
            "y": [1, 0, 0],
            "lateval": [12.0, 0.0, 0.0],
            "log_exporter_capacity": [2.0, 2.0, 3.0],
            "log_importer_demand": [4.0, 5.0, 4.0],
            "grav": [0.2, np.nan, 0.1],
        }
    )
    lanes["entry_id"] = _entry_id(lanes)
    entries = _derive_entry_table(lanes)
    if len(entries) != 2 or int(entries["z"].sum()) != 1:
        raise AssertionError(entries)
    conditional = lanes.loc[lanes["entry_id"].eq("A|s")].reset_index(drop=True)
    metrics = _conditional_metrics(
        conditional,
        np.array([1.0, 0.0]),
        bootstrap_draws=20,
        seed=1,
    )
    if metrics["at_k"]["k_1"]["macro_recall"] != 1.0:
        raise AssertionError(metrics)
    print("v2_rolling_cpu_baselines self-test: OK")


def verify_existing_output(path: Path = DEFAULT_JSON) -> None:
    """Verify that a saved result still refers to the current input bytes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: stale schema {payload.get('schema_version')!r}; expected "
            f"{SCHEMA_VERSION!r}. The pre-audit result is invalidated and must be rebuilt "
            "only after the corrected chain registry and cohorts are frozen."
        )
    protocol = payload.get("protocol", {})
    required_false = (
        "target_labels_used_for_model_selection",
        "target_labels_used_for_imputation_scaling_or_calibration",
        "transductive_split_used",
    )
    for field in required_false:
        if protocol.get(field) is not False:
            raise ValueError(f"{path}: protocol flag {field!r} is not false")

    checked = 0
    for chain, chain_payload in payload.get("chains", {}).items():
        audit = chain_payload.get("protocol_audit", {})
        if audit.get("target_loaded_after_all_models_frozen") is not True:
            raise ValueError(f"{path}: {chain} does not record the global freeze boundary")
        for input_name in (
            "history_track_a",
            "history_track_b",
            "target_track_a",
            "target_track_b",
        ):
            record = audit.get(input_name, {})
            input_path = ROOT / str(record.get("path", ""))
            expected = record.get("sha256")
            if not input_path.is_file() or not expected:
                raise ValueError(f"{path}: missing {chain}/{input_name} input audit")
            observed = _sha256(input_path)
            if observed != expected:
                raise ValueError(
                    f"{path}: stale input hash for {chain}/{input_name}: "
                    f"expected {expected}, observed {observed}"
                )
            checked += 1
    if checked == 0:
        raise ValueError(f"{path}: no input hashes found")
    print(f"verified {checked} current input hashes in {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    chosen_chains = tuple(args.chains)
    unknown = sorted(set(chosen_chains).difference(CHAINS))
    if unknown:
        raise ValueError(f"unknown chains: {unknown}")

    # Freeze every chain before opening a single main-window file.  Besides
    # preventing data flow in code, this ordering rules out an interactive
    # analyst adapting a later chain after seeing an earlier target result.
    frozen_models: dict[str, tuple[ChainModels, int]] = {}
    for index, chain in enumerate(chosen_chains):
        chain_seed = args.seed + index * 1000
        print(f"[{chain}] fitting only on historical fold2 ...", flush=True)
        frozen_models[chain] = (_fit_chain(data_dir, chain, chain_seed), chain_seed)

    print("all requested chain models frozen; opening main target tables ...", flush=True)
    chain_results: dict[str, Any] = {}
    for chain in chosen_chains:
        models, chain_seed = frozen_models[chain]
        print(f"[{chain}] evaluating complete main target cohort ...", flush=True)
        chain_results[chain] = _evaluate_chain(
            data_dir,
            chain,
            models,
            bootstrap_draws=args.bootstrap,
            seed=chain_seed,
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": "2.1-dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "selection_window": "1998-2002 -> 2008-2012",
            "frozen_target_window": "2008-2012 -> 2018-2022",
            "selection_source": "fold2 only",
            "target_evaluation": "complete main cohort",
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_imputation_scaling_or_calibration": False,
            "transductive_split_used": False,
            "main_target_models_compared_without_post_hoc_champion_selection": True,
            "selection_objectives": {
                "track_a": "historical_exporter_group_cv_average_precision",
                "track_b1": "historical_exporter_group_cv_average_precision",
                "track_b2": (
                    "historical_exporter_stage_entry_group_cv_per_positive_entry_"
                    "macro_recall_at_3"
                ),
            },
            "main_metric_cluster_units": {
                "track_a_average_precision": "exporter",
                "track_b1_average_precision": "exporter",
                "track_b2_recall_and_value": "exporter_stage_entry",
            },
            "realized_value_reporting_points": {
                "track_a_global_top_k": TRACK_A_VALUE_BUDGET,
                "track_b1_global_top_k": TRACK_B1_VALUE_BUDGET,
                "track_b2_per_entry_top_k": TRACK_B2_VALUE_K,
            },
            "pairwise_reporting": (
                "all protocol-fixed unordered model pairs; descriptive per-chain deltas "
                "without chain-level confidence intervals"
            ),
            "frozen_pairwise_model_order": {
                track: list(models) for track, models in TRACK_MODEL_ORDER.items()
            },
            "random_seed": int(args.seed),
            "bootstrap_draws": int(args.bootstrap),
        },
        "model_families": {
            "raw_rankings": "fixed ex-ante early-window features; missing gravity uses the historical median",
            "historical_logistic": (
                "class-balanced logistic regression; Track A/B1 C selected by exporter-grouped "
                "historical-fold average precision and Track B2 C by entry-grouped per-positive-"
                "entry macro recall@3; median imputation and scaling refit on all fold2 rows"
            ),
        },
        "chains": chain_results,
    }
    result["macro_summary"] = _macro_summary(chain_results)
    return _jsonable(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chains", nargs="+", default=list(CHAINS))
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="cluster-bootstrap draws for primary metric intervals; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="verify protocol flags and current input hashes in --json-out, then exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    auxiliary_check = False
    if args.self_test:
        _self_test()
        auxiliary_check = True
    if args.verify_output:
        verify_existing_output(args.json_out.resolve())
        auxiliary_check = True
    if auxiliary_check:
        return
    result = run(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_csv_rows(result)).to_csv(args.csv_out, index=False)
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
