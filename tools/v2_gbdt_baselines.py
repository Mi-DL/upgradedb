#!/usr/bin/env python3
"""Protocol-conforming GBDT references for UpgradeBench A/B1/B2.

The frozen configuration is the only source of estimator, feature, selection,
budget, uncertainty, and read-gate choices.  The runner first selects and
refits all 18 chain--task models from the historical fold.  Only after that
global freeze succeeds can it open any main-window candidate table.

The output is deliberately independent of ``rolling_cpu_baselines``.  It
reuses that module's audited candidate readers, feature builders, group-safe
splitter, task metrics, and tie handling, but it neither reads nor rewrites the
central rolling result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in os.sys.path:
    os.sys.path.insert(0, str(TOOLS))

import v2_rolling_cpu_baselines as cpu  # noqa: E402


SCHEMA_VERSION = "upgrade-bench-v2-gbdt-baselines/1"
BENCHMARK_VERSION = "2.1-dev"
STATUS = "complete_verified"
MODEL_KEY = "historical_gbdt_same_features"
CONFIG_SCHEMA = "upgrade-bench-v2-gbdt-config/1"
CONFIG_PROTOCOL = "historical-group-select-then-frozen-main-v1"
CONFIG_STATUS = "frozen_before_first_gbdt_main_evaluation"
CONFIG_ROLE = "configs/v2_gbdt_baselines.json"
SHARED_SOURCE_ROLE = "tools/v2_rolling_cpu_baselines.py"
RUNNER_SOURCE_ROLE = "tools/v2_gbdt_baselines.py"

DEFAULT_CONFIG = ROOT / CONFIG_ROLE
DEFAULT_DATA = ROOT / "data" / "processed_v2"
DEFAULT_JSON = ROOT / "results_v2" / "metrics" / "v2_gbdt_baselines.json"
DEFAULT_CSV = ROOT / "results_v2" / "metrics" / "v2_gbdt_baselines.csv"

TRACK_ORDER = (
    "track_a_destination_extension",
    "track_b1_processed_export_stage_entry",
    "track_b2_conditional_destination_ranking",
)
TRACK_SHORT = {
    "track_a_destination_extension": "a",
    "track_b1_processed_export_stage_entry": "b1",
    "track_b2_conditional_destination_ranking": "b2",
}
FEATURES = {
    "a": tuple(cpu.TRACK_A_FEATURES),
    "b1": tuple(cpu.TRACK_B1_FEATURES),
    "b2": tuple(cpu.TRACK_B2_FEATURES),
}
OBJECTIVE_NAMES = {
    "a": "historical_exporter_group_cv_average_precision",
    "b1": "historical_exporter_group_cv_average_precision",
    "b2": "historical_positive_entry_group_cv_macro_recall_at_3",
}
GROUP_UNITS = {
    "a": "exporter",
    "b1": "exporter",
    "b2": "positive_exporter_stage_entry",
}
TASK_SEED_TAG = {"a": "track-a", "b1": "track-b1", "b2": "track-b2"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
UNIX_ABSOLUTE = re.compile(r"^/")
UNC_ABSOLUTE = re.compile(r"^\\\\")
REMOTE_ABSOLUTE = re.compile(r"^[^/\\\s:]+:[\\/]")


class GBDTProtocolError(ValueError):
    """Raised when the frozen protocol or a result artifact is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _strict_json_load(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GBDTProtocolError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GBDTProtocolError(f"non-finite JSON constant {value!r} in {path}")

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GBDTProtocolError(f"cannot read strict JSON from {path}") from exc
    if not isinstance(value, dict):
        raise GBDTProtocolError(f"{path}: JSON root must be an object")
    return value


def _exact_keys(value: object, expected: Iterable[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GBDTProtocolError(f"{where} must be an object")
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise GBDTProtocolError(
            f"{where} keys differ: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )
    return value


def _finite(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GBDTProtocolError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GBDTProtocolError(f"{where} must be finite")
    return result


def _integer(value: object, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GBDTProtocolError(f"{where} must be an integer >= {minimum}")
    return value


def _probability(value: object, where: str) -> float:
    result = _finite(value, where)
    if not 0.0 <= result <= 1.0:
        raise GBDTProtocolError(f"{where} must lie in [0,1]")
    return result


def _close(observed: object, expected: float, where: str, *, atol: float = 1e-12) -> None:
    value = _finite(observed, where)
    if not math.isclose(value, float(expected), rel_tol=1e-10, abs_tol=atol):
        raise GBDTProtocolError(f"{where}: observed {value}, expected {expected}")


def _hex(value: object, where: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise GBDTProtocolError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _iso_datetime(value: object, where: str) -> datetime:
    if not isinstance(value, str):
        raise GBDTProtocolError(f"{where} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GBDTProtocolError(f"{where} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise GBDTProtocolError(f"{where} must include a UTC offset")
    return parsed


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(base), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _assert_privacy(value: object, where: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_privacy(key, f"{where}.<key>")
            _assert_privacy(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_privacy(item, f"{where}[{index}]")
        return
    if isinstance(value, str):
        if any(
            pattern.match(value)
            for pattern in (
                WINDOWS_ABSOLUTE,
                UNIX_ABSOLUTE,
                UNC_ABSOLUTE,
                REMOTE_ABSOLUTE,
            )
        ):
            raise GBDTProtocolError(f"private/absolute text is forbidden at {where}")


def _config_reference(config_path: Path) -> dict[str, str]:
    resolved = config_path.resolve()
    if resolved != DEFAULT_CONFIG.resolve():
        raise GBDTProtocolError(
            "formal runs must use configs/v2_gbdt_baselines.json at its canonical role"
        )
    return {"path": CONFIG_ROLE, "sha256": _sha256(resolved)}


def load_frozen_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _strict_json_load(path.resolve())
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol",
            "status",
            "frozen_at_utc",
            "chains",
            "tracks",
            "estimator",
            "grid",
            "min_samples_leaf",
            "features",
            "historical_selection",
            "main_evaluation",
            "uncertainty",
            "claim_scope",
        },
        "config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol"] != CONFIG_PROTOCOL
        or config["status"] != CONFIG_STATUS
    ):
        raise GBDTProtocolError("frozen GBDT config identity/status mismatch")
    frozen_at = _iso_datetime(config["frozen_at_utc"], "config.frozen_at_utc")
    if frozen_at.utcoffset() is None or frozen_at.utcoffset().total_seconds() != 0:
        raise GBDTProtocolError("config.frozen_at_utc must be UTC")
    if config["chains"] != list(cpu.CHAINS) or config["tracks"] != ["a", "b1", "b2"]:
        raise GBDTProtocolError("config must cover the canonical six chains and A/B1/B2")

    estimator = _exact_keys(
        config["estimator"],
        {
            "class",
            "loss",
            "learning_rate",
            "l2_regularization",
            "class_weight",
            "early_stopping",
            "max_bins",
            "random_state",
        },
        "config.estimator",
    )
    if (
        estimator["class"] != "sklearn.ensemble.HistGradientBoostingClassifier"
        or estimator["loss"] != "log_loss"
        or estimator["class_weight"] != "balanced"
        or estimator["early_stopping"] is not False
        or estimator["max_bins"] != 255
    ):
        raise GBDTProtocolError("config estimator is not the reviewed leakage-safe HistGBDT")
    if _finite(estimator["learning_rate"], "config.estimator.learning_rate") <= 0:
        raise GBDTProtocolError("learning_rate must be positive")
    if _finite(estimator["l2_regularization"], "config.estimator.l2_regularization") < 0:
        raise GBDTProtocolError("l2_regularization must be non-negative")
    _integer(estimator["random_state"], "config.estimator.random_state")

    grid = config["grid"]
    if not isinstance(grid, list) or len(grid) != 4:
        raise GBDTProtocolError("config.grid must contain exactly four declared configurations")
    ids: set[str] = set()
    for index, row in enumerate(grid):
        item = _exact_keys(row, {"config_id", "max_leaf_nodes", "max_iter"}, f"config.grid[{index}]")
        config_id = item["config_id"]
        if not isinstance(config_id, str) or not config_id or config_id in ids:
            raise GBDTProtocolError("config.grid config_id values must be unique non-empty strings")
        ids.add(config_id)
        _integer(item["max_leaf_nodes"], f"config.grid[{index}].max_leaf_nodes", minimum=2)
        _integer(item["max_iter"], f"config.grid[{index}].max_iter", minimum=1)

    leaves = _exact_keys(config["min_samples_leaf"], {"a", "b1", "b2"}, "config.min_samples_leaf")
    feature_config = _exact_keys(config["features"], {"a", "b1", "b2"}, "config.features")
    for track in ("a", "b1", "b2"):
        _integer(leaves[track], f"config.min_samples_leaf.{track}", minimum=1)
        if feature_config[track] != list(FEATURES[track]):
            raise GBDTProtocolError(f"config features for {track} differ from the current logistic reference")

    selection = _exact_keys(
        config["historical_selection"],
        {"a", "b1", "b2", "maximum_group_folds", "selection_tie_break", "refit"},
        "config.historical_selection",
    )
    expected_selection = {
        "a": {"group_unit": "exporter", "objective": "grouped_average_precision"},
        "b1": {"group_unit": "exporter", "objective": "grouped_average_precision"},
        "b2": {
            "group_unit": "positive_exporter_stage_entry",
            "objective": "macro_recall_at_3",
        },
    }
    for track, expected in expected_selection.items():
        if selection[track] != expected:
            raise GBDTProtocolError(f"config historical selection differs for {track}")
    if (
        selection["maximum_group_folds"] != 5
        or selection["selection_tie_break"] != "first_config_in_declared_grid_order"
        or selection["refit"] != "complete_historical_fold"
    ):
        raise GBDTProtocolError("config historical selection policy mismatch")

    main = _exact_keys(
        config["main_evaluation"],
        {"read_gate", "target_access", "a_value_budget", "b1_value_budget", "b2_k"},
        "config.main_evaluation",
    )
    if main != {
        "read_gate": "freeze_all_18_chain_task_models_before_opening_any_main_candidate_table",
        "target_access": "one_complete_main_cohort_evaluation_per_frozen_model",
        "a_value_budget": 500,
        "b1_value_budget": 50,
        "b2_k": 3,
    }:
        raise GBDTProtocolError("config main-evaluation gate or budgets changed")

    uncertainty = _exact_keys(
        config["uncertainty"],
        {"method", "draws", "rng_seed", "interval", "cluster_unit"},
        "config.uncertainty",
    )
    if (
        uncertainty["method"] != "nonparametric_cluster_bootstrap"
        or uncertainty["draws"] != 200
        or uncertainty["interval"] != "percentile_95"
        or uncertainty["cluster_unit"]
        != {"a": "exporter", "b1": "exporter", "b2": "positive_exporter_stage_entry"}
    ):
        raise GBDTProtocolError("config uncertainty contract mismatch")
    _integer(uncertainty["rng_seed"], "config.uncertainty.rng_seed")
    if config["claim_scope"] != (
        "reviewer-motivated protocol-conforming tabular reference; not part of the original "
        "prespecified reference set"
    ):
        raise GBDTProtocolError("config claim scope changed")
    _assert_privacy(config, "config")
    return config


def _pipeline(config: Mapping[str, Any], track: str, grid_row: Mapping[str, Any]) -> Pipeline:
    estimator = config["estimator"]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    loss=str(estimator["loss"]),
                    learning_rate=float(estimator["learning_rate"]),
                    max_iter=int(grid_row["max_iter"]),
                    max_leaf_nodes=int(grid_row["max_leaf_nodes"]),
                    min_samples_leaf=int(config["min_samples_leaf"][track]),
                    l2_regularization=float(estimator["l2_regularization"]),
                    max_bins=int(estimator["max_bins"]),
                    class_weight=str(estimator["class_weight"]),
                    early_stopping=False,
                    random_state=int(estimator["random_state"]),
                ),
            ),
        ]
    )


@dataclass
class FrozenGBDT:
    pipeline: Pipeline
    selection: dict[str, Any]


def _select_and_fit(
    config: Mapping[str, Any],
    track: str,
    features: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    objective_groups: Sequence[str] | None = None,
    tie_break_keys: Sequence[str] | None = None,
) -> FrozenGBDT:
    names = FEATURES[track]
    x = features.loc[:, list(names)]
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(groups, dtype=str)
    if np.unique(y).size != 2:
        raise GBDTProtocolError(f"historical {track} labels contain only one class")
    split_seed = int(config["estimator"]["random_state"])
    splits, n_splits = cpu._valid_group_splits(
        y,
        groups,
        seed=split_seed,
        maximum=int(config["historical_selection"]["maximum_group_folds"]),
    )
    for train, valid in splits:
        if set(groups[train]).intersection(groups[valid]):
            raise GBDTProtocolError(f"historical {track} CV leaked groups")

    if track == "b2":
        if objective_groups is None or tie_break_keys is None:
            raise GBDTProtocolError("B2 selection requires entry groups and destination tie keys")
        objective_groups = np.asarray(objective_groups, dtype=str)
        tie_break_keys = np.asarray(tie_break_keys, dtype=str)
        if len(objective_groups) != len(y) or len(tie_break_keys) != len(y):
            raise GBDTProtocolError("B2 selection metadata is misaligned")

    candidates: list[dict[str, Any]] = []
    for index, grid_row in enumerate(config["grid"]):
        fold_values: list[float] = []
        fold_units: list[int] = []
        for train, valid in splits:
            model = _pipeline(config, track, grid_row)
            model.fit(x.iloc[train], y[train])
            scores = model.predict_proba(x.iloc[valid])[:, 1]
            if track in {"a", "b1"}:
                value = float(average_precision_score(y[valid], scores))
                units = 1
            else:
                assert objective_groups is not None and tie_break_keys is not None
                value = cpu._positive_entry_macro_recall_at_k(
                    y[valid],
                    scores,
                    objective_groups[valid],
                    tie_break_keys[valid],
                    k=int(config["main_evaluation"]["b2_k"]),
                )
                units = int(pd.Series(objective_groups[valid]).nunique())
            fold_values.append(value)
            fold_units.append(units)
        mean = (
            float(np.average(fold_values, weights=fold_units))
            if track == "b2"
            else float(np.mean(fold_values))
        )
        candidates.append(
            {
                "config_id": str(grid_row["config_id"]),
                "grid_index": index,
                "parameters": {
                    "max_leaf_nodes": int(grid_row["max_leaf_nodes"]),
                    "max_iter": int(grid_row["max_iter"]),
                    "min_samples_leaf": int(config["min_samples_leaf"][track]),
                },
                "fold_objective_values": fold_values,
                "fold_objective_units": fold_units,
                "mean_objective": mean,
                "std_objective": float(np.std(fold_values, ddof=0)),
            }
        )

    # Python's max returns the first item at an exact tie, preserving the
    # config's frozen declaration-order tie break.
    selected = max(candidates, key=lambda row: row["mean_objective"])
    selected_grid = config["grid"][selected["grid_index"]]
    final_model = _pipeline(config, track, selected_grid)
    final_model.fit(x, y)
    selection = {
        "feature_names": list(names),
        "group_unit": GROUP_UNITS[track],
        "objective": OBJECTIVE_NAMES[track],
        "objective_direction": "maximize",
        "objective_aggregation": (
            "validation-entry-count-weighted-fold-mean"
            if track == "b2"
            else "unweighted-mean-over-validation-folds"
        ),
        "train_validation_group_overlap_checked": True,
        "n_splits": n_splits,
        "split_seed": split_seed,
        "grid_source": CONFIG_ROLE,
        "candidates": candidates,
        "selected_config_id": selected["config_id"],
        "selected_grid_index": selected["grid_index"],
        "selected_mean_objective": selected["mean_objective"],
        "selection_tie_break": "first_config_in_declared_grid_order",
        "preprocessing": {
            "imputer": "historical_partition_median_with_missing_indicators",
            "scaler": "none_tree_model",
        },
        "early_stopping": False,
        "class_weight": "balanced",
        "refit_scope": "complete_historical_fold",
        "refit_rows": int(len(y)),
        "refit_positives": int(y.sum()),
    }
    return FrozenGBDT(final_model, selection)


@dataclass
class ChainModels:
    track_a: FrozenGBDT
    track_b1: FrozenGBDT
    track_b2: FrozenGBDT
    history_a_audit: dict[str, Any]
    history_b_audit: dict[str, Any]
    history_b1_entries: int
    history_b1_positives: int
    history_b2_lanes: int
    history_b2_positive_lanes: int
    history_b2_entry_groups: int


def _fit_chain(data_dir: Path, chain: str, config: Mapping[str, Any]) -> ChainModels:
    # This is the read boundary: there is no code path to a main filename.
    history_a, audit_a = cpu._read_candidate(data_dir, chain, track="a", historical=True)
    history_b, audit_b = cpu._read_candidate(data_dir, chain, track="b", historical=True)

    a_features = cpu._track_a_features(history_a)
    a_model = _select_and_fit(
        config,
        "a",
        a_features,
        history_a["y"].to_numpy(np.int8),
        history_a["i_iso"].to_numpy(str),
    )

    entries = cpu._derive_entry_table(history_b)
    b1_model = _select_and_fit(
        config,
        "b1",
        entries,
        entries["z"].to_numpy(np.int8),
        entries["i_iso"].to_numpy(str),
    )

    positive_ids = set(entries.loc[entries["z"].eq(1), "entry_id"].astype(str))
    history_b2 = history_b.loc[history_b["entry_id"].isin(positive_ids)].copy()
    b2_features = cpu._track_b2_features(history_b2)
    b2_model = _select_and_fit(
        config,
        "b2",
        b2_features,
        history_b2["y"].to_numpy(np.int8),
        history_b2["entry_id"].to_numpy(str),
        objective_groups=history_b2["entry_id"].to_numpy(str),
        tie_break_keys=history_b2["j_iso"].to_numpy(str),
    )
    return ChainModels(
        track_a=a_model,
        track_b1=b1_model,
        track_b2=b2_model,
        history_a_audit=audit_a,
        history_b_audit=audit_b,
        history_b1_entries=int(len(entries)),
        history_b1_positives=int(entries["z"].sum()),
        history_b2_lanes=int(len(history_b2)),
        history_b2_positive_lanes=int(history_b2["y"].sum()),
        history_b2_entry_groups=int(history_b2["entry_id"].nunique()),
    )


def _evaluate_chain(
    data_dir: Path,
    chain: str,
    models: ChainModels,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    # The caller invokes this only after all six ChainModels objects exist.
    target_a, target_a_audit = cpu._read_candidate(
        data_dir, chain, track="a", historical=False
    )
    target_b, target_b_audit = cpu._read_candidate(
        data_dir, chain, track="b", historical=False
    )
    base_bootstrap_seed = int(config["uncertainty"]["rng_seed"])
    draws = int(config["uncertainty"]["draws"])

    a_features = cpu._track_a_features(target_a)
    a_score = models.track_a.pipeline.predict_proba(
        a_features.loc[:, list(FEATURES["a"])]
    )[:, 1]
    a_metrics = cpu._classification_metrics(
        target_a,
        label="y",
        score=a_score,
        cluster=target_a["i_iso"].to_numpy(str),
        cluster_unit="exporter",
        budgets=(100, int(config["main_evaluation"]["a_value_budget"]), 1000),
        bootstrap_draws=draws,
        seed=_stable_seed(base_bootstrap_seed, chain, "a", MODEL_KEY),
        exporter_shortlists=True,
    )

    target_entries = cpu._derive_entry_table(target_b)
    b1_score = models.track_b1.pipeline.predict_proba(
        target_entries.loc[:, list(FEATURES["b1"])]
    )[:, 1]
    b1_metrics = cpu._classification_metrics(
        target_entries,
        label="z",
        score=b1_score,
        cluster=target_entries["i_iso"].to_numpy(str),
        cluster_unit="exporter",
        budgets=(25, int(config["main_evaluation"]["b1_value_budget"]), 100),
        bootstrap_draws=draws,
        seed=_stable_seed(base_bootstrap_seed, chain, "b1", MODEL_KEY),
    )

    # Score every B lane before opening the main entry condition for evaluation.
    b2_features_all = cpu._track_b2_features(target_b)
    b2_score_all = models.track_b2.pipeline.predict_proba(
        b2_features_all.loc[:, list(FEATURES["b2"])]
    )[:, 1]
    positive_target_ids = set(
        target_entries.loc[target_entries["z"].eq(1), "entry_id"].astype(str)
    )
    conditional_mask = target_b["entry_id"].isin(positive_target_ids).to_numpy(bool)
    target_b2 = target_b.loc[conditional_mask].reset_index(drop=True)
    b2_metrics = cpu._conditional_metrics(
        target_b2,
        b2_score_all[conditional_mask],
        bootstrap_draws=draws,
        seed=_stable_seed(base_bootstrap_seed, chain, "b2", MODEL_KEY),
        ks=(1, int(config["main_evaluation"]["b2_k"]), 5),
    )

    return {
        "protocol_audit": {
            "target_loaded_after_all_18_models_frozen": True,
            "target_labels_used_for_training_selection_imputation_or_calibration": False,
            "transductive_split_used": False,
            "b2_all_lanes_scored_before_target_conditioning": True,
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
            "models": {
                MODEL_KEY: {
                    "model": {
                        "kind": "historically_selected_gbdt_reference",
                        "selection": models.track_a.selection,
                    },
                    "metrics": a_metrics,
                }
            },
        },
        "track_b1_processed_export_stage_entry": {
            "unit": "exporter_stage",
            "history_rows": models.history_b1_entries,
            "history_positives": models.history_b1_positives,
            "target_rows": int(len(target_entries)),
            "target_positives": int(target_entries["z"].sum()),
            "models": {
                MODEL_KEY: {
                    "model": {
                        "kind": "historically_selected_gbdt_reference",
                        "selection": models.track_b1.selection,
                    },
                    "metrics": b1_metrics,
                }
            },
        },
        "track_b2_conditional_destination_ranking": {
            "unit": "destination_within_actual_exporter_stage_entry",
            "conditioning": "actual_entry_in_corresponding_late_window",
            "history_rows": models.history_b2_lanes,
            "history_positives": models.history_b2_positive_lanes,
            "history_entry_groups": models.history_b2_entry_groups,
            "target_rows": int(len(target_b2)),
            "target_positives": int(target_b2["y"].sum()),
            "target_entry_groups": int(target_b2["entry_id"].nunique()),
            "models": {
                MODEL_KEY: {
                    "model": {
                        "kind": "historically_selected_conditional_gbdt_ranker",
                        "selection": models.track_b2.selection,
                    },
                    "metrics": b2_metrics,
                }
            },
        },
    }


def _headline_value(track: str, payload: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, float]:
    metrics = payload["models"][MODEL_KEY]["metrics"]
    if track == "track_a_destination_extension":
        return (
            float(metrics["average_precision"]),
            float(metrics["budgets"][f"k_{config['main_evaluation']['a_value_budget']}"]["value_capture"]),
        )
    if track == "track_b1_processed_export_stage_entry":
        return (
            float(metrics["average_precision"]),
            float(metrics["budgets"][f"k_{config['main_evaluation']['b1_value_budget']}"]["value_capture"]),
        )
    at_k = metrics["at_k"][f"k_{config['main_evaluation']['b2_k']}"]
    return float(at_k["macro_recall"]), float(at_k["macro_value_capture"])


def _macro_summary(
    chains: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not chains:
        raise GBDTProtocolError("cannot summarize zero chains")
    if set(chains) != set(cpu.CHAINS):
        raise GBDTProtocolError("macro summary requires the canonical six-chain registry")
    summary: dict[str, Any] = {}
    for track in TRACK_ORDER:
        headline = {
            chain: _headline_value(track, chains[chain][track], config)[0]
            for chain in cpu.CHAINS
        }
        value = {
            chain: _headline_value(track, chains[chain][track], config)[1]
            for chain in cpu.CHAINS
        }
        metric = "average_precision" if track != TRACK_ORDER[2] else "per_positive_entry_macro_recall_at_3"
        value_metric = {
            TRACK_ORDER[0]: "global_observed_late_value_capture_at_500",
            TRACK_ORDER[1]: "global_observed_late_value_capture_at_50",
            TRACK_ORDER[2]: "per_positive_entry_macro_value_capture_at_3",
        }[track]
        summary[track] = {
            "headline_metric": metric,
            "realized_value_metric": value_metric,
            "aggregation": "unweighted_mean_over_chains",
            "chain_registry": list(cpu.CHAINS),
            "model": MODEL_KEY,
            "headline": {
                "per_chain": headline,
                "macro_mean": float(np.mean(list(headline.values()))),
                "std_across_chains": float(np.std(list(headline.values()), ddof=0)),
            },
            "realized_value": {
                "per_chain": value,
                "macro_mean": float(np.mean(list(value.values()))),
                "std_across_chains": float(np.std(list(value.values()), ddof=0)),
            },
            "chain_level_ci95": None,
            "inference_note": "descriptive over the six fixed chains; no chain-population interval",
        }
    return summary


def _candidate_inventory(chains: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roles = ("history_track_a", "history_track_b", "target_track_a", "target_track_b")
    for chain in cpu.CHAINS:
        audit = chains[chain]["protocol_audit"]
        for role in roles:
            source = audit[role]
            rows.append(
                {
                    "chain": chain,
                    "role": role,
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "rows": int(source["rows"]),
                    "positive_lanes": int(source["positive_lanes"]),
                    "early_window": source["early_window"],
                    "late_window": source["late_window"],
                }
            )
    return rows


def run(
    data_dir: Path = DEFAULT_DATA,
    config_path: Path = DEFAULT_CONFIG,
    *,
    chains: Sequence[str] = cpu.CHAINS,
    fit_chain: Callable[[Path, str, Mapping[str, Any]], ChainModels] = _fit_chain,
    evaluate_chain: Callable[[Path, str, ChainModels, Mapping[str, Any]], dict[str, Any]] = _evaluate_chain,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_frozen_config(config_path)
    requested = tuple(chains)
    if len(set(requested)) != len(requested) or any(chain not in cpu.CHAINS for chain in requested):
        raise GBDTProtocolError("requested chains contain duplicates or unknown identifiers")

    # Phase 1: no main path can be opened through fit_chain.
    frozen: dict[str, ChainModels] = {}
    for chain in requested:
        print(f"[{chain}] selecting/refitting GBDT from historical fold only ...", flush=True)
        frozen[chain] = fit_chain(data_dir, chain, config)

    # Phase 2 begins only after every requested model object exists.
    print("all historical GBDT models frozen; opening main cohorts ...", flush=True)
    results: dict[str, Any] = {}
    for chain in requested:
        print(f"[{chain}] evaluating one complete main cohort ...", flush=True)
        results[chain] = evaluate_chain(data_dir, chain, frozen[chain], config)

    config_ref = _config_reference(config_path)
    cpu_model = platform.processor().strip() or os.environ.get(
        "PROCESSOR_IDENTIFIER", "unknown-cpu"
    ).strip()
    logical_cores = os.cpu_count()
    if not cpu_model or not logical_cores:
        raise GBDTProtocolError("runtime CPU identity/logical-core count is unavailable")
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise GBDTProtocolError("runtime wall-clock duration is not positive and finite")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": config["claim_scope"],
        "config": config_ref,
        "protocol": {
            "selection_window": "1998-2002 -> 2008-2012",
            "frozen_target_window": "2008-2012 -> 2018-2022",
            "selection_source": "fold2 only",
            "read_gate": config["main_evaluation"]["read_gate"],
            "target_access": config["main_evaluation"]["target_access"],
            "target_labels_used_for_model_selection_imputation_or_calibration": False,
            "transductive_split_used": False,
            "all_models_frozen_before_any_main_read": True,
            "b2_all_lanes_scored_before_target_conditioning": True,
            "selection_objectives": OBJECTIVE_NAMES,
            "selection_group_units": GROUP_UNITS,
            "bootstrap": config["uncertainty"],
            "main_reporting": config["main_evaluation"],
        },
        "feature_registry": {track: list(FEATURES[track]) for track in ("a", "b1", "b2")},
        "inputs": {
            "candidate_files": _candidate_inventory(results) if requested == tuple(cpu.CHAINS) else [],
            "public_sources": {
                RUNNER_SOURCE_ROLE: _sha256(Path(__file__).resolve()),
                SHARED_SOURCE_ROLE: _sha256(ROOT / SHARED_SOURCE_ROLE),
                CONFIG_ROLE: config_ref["sha256"],
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "logical_cpu_cores": int(logical_cores),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "wall_elapsed_seconds": float(elapsed),
            "fit_count_upper_bound": len(requested)
            * 3
            * (len(config["grid"]) * int(config["historical_selection"]["maximum_group_folds"]) + 1),
        },
        "chains": results,
        "macro_summary": _macro_summary(results, config),
    }
    _assert_privacy(payload)
    return payload


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = load_frozen_config(ROOT / payload["config"]["path"])
    rows: list[dict[str, Any]] = []
    for chain in cpu.CHAINS:
        chain_payload = payload["chains"][chain]
        for track in TRACK_ORDER:
            task = chain_payload[track]
            model = task["models"][MODEL_KEY]
            selection = model["model"]["selection"]
            metrics = model["metrics"]
            headline, value = _headline_value(track, task, config)
            if track == TRACK_ORDER[2]:
                k_payload = metrics["at_k"][f"k_{config['main_evaluation']['b2_k']}"]
                headline_ci = k_payload["macro_recall_ci95"]
                value_ci = k_payload["macro_value_capture_ci95"]
                target_groups = task["target_entry_groups"]
            else:
                headline_ci = metrics["average_precision_ci95"]
                value_ci = None
                target_groups = ""
            rows.append(
                {
                    "chain": chain,
                    "track": TRACK_SHORT[track],
                    "model": MODEL_KEY,
                    "history_rows": task["history_rows"],
                    "history_positives": task["history_positives"],
                    "target_rows": task["target_rows"],
                    "target_positives": task["target_positives"],
                    "target_entry_groups": target_groups,
                    "selected_config_id": selection["selected_config_id"],
                    "selected_grid_index": selection["selected_grid_index"],
                    "historical_cv_objective": selection["selected_mean_objective"],
                    "n_splits": selection["n_splits"],
                    "headline_metric": payload["macro_summary"][track]["headline_metric"],
                    "headline": headline,
                    "headline_ci95_low": "" if headline_ci is None else headline_ci[0],
                    "headline_ci95_high": "" if headline_ci is None else headline_ci[1],
                    "value_metric": payload["macro_summary"][track]["realized_value_metric"],
                    "value_capture": value,
                    "value_ci95_low": "" if value_ci is None else value_ci[0],
                    "value_ci95_high": "" if value_ci is None else value_ci[1],
                }
            )
    return rows


def _csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return pd.DataFrame(_csv_rows(payload)).to_csv(index=False, lineterminator="\n").encode("utf-8")


def write_outputs(
    payload: Mapping[str, Any],
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(_strict_json_bytes(payload))
    csv_path.write_bytes(_csv_bytes(payload))


def _validate_interval(value: object, where: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise GBDTProtocolError(f"{where} must be null or a two-element interval")
    low = _probability(value[0], f"{where}[0]")
    high = _probability(value[1], f"{where}[1]")
    if low > high:
        raise GBDTProtocolError(f"{where} interval is reversed")


def _validate_selection(
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    track: str,
    history_rows: int,
    history_positives: int,
    where: str,
) -> None:
    expected_keys = {
        "feature_names",
        "group_unit",
        "objective",
        "objective_direction",
        "objective_aggregation",
        "train_validation_group_overlap_checked",
        "n_splits",
        "split_seed",
        "grid_source",
        "candidates",
        "selected_config_id",
        "selected_grid_index",
        "selected_mean_objective",
        "selection_tie_break",
        "preprocessing",
        "early_stopping",
        "class_weight",
        "refit_scope",
        "refit_rows",
        "refit_positives",
    }
    _exact_keys(selection, expected_keys, where)
    if (
        selection["feature_names"] != list(FEATURES[track])
        or selection["group_unit"] != GROUP_UNITS[track]
        or selection["objective"] != OBJECTIVE_NAMES[track]
        or selection["objective_direction"] != "maximize"
        or selection["train_validation_group_overlap_checked"] is not True
        or selection["split_seed"] != config["estimator"]["random_state"]
        or selection["grid_source"] != CONFIG_ROLE
        or selection["selection_tie_break"] != "first_config_in_declared_grid_order"
        or selection["early_stopping"] is not False
        or selection["class_weight"] != "balanced"
        or selection["refit_scope"] != "complete_historical_fold"
        or selection["preprocessing"]
        != {
            "imputer": "historical_partition_median_with_missing_indicators",
            "scaler": "none_tree_model",
        }
    ):
        raise GBDTProtocolError(f"{where}: selection contract mismatch")
    if selection["objective_aggregation"] != (
        "validation-entry-count-weighted-fold-mean"
        if track == "b2"
        else "unweighted-mean-over-validation-folds"
    ):
        raise GBDTProtocolError(f"{where}: objective aggregation mismatch")
    n_splits = _integer(selection["n_splits"], f"{where}.n_splits", minimum=2)
    if n_splits > config["historical_selection"]["maximum_group_folds"]:
        raise GBDTProtocolError(f"{where}: too many CV folds")
    if selection["refit_rows"] != history_rows or selection["refit_positives"] != history_positives:
        raise GBDTProtocolError(f"{where}: refit counts do not match the historical cohort")
    candidates = selection["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(config["grid"]):
        raise GBDTProtocolError(f"{where}: candidate grid is incomplete")
    means: list[float] = []
    for index, (candidate, grid_row) in enumerate(zip(candidates, config["grid"])):
        _exact_keys(
            candidate,
            {
                "config_id",
                "grid_index",
                "parameters",
                "fold_objective_values",
                "fold_objective_units",
                "mean_objective",
                "std_objective",
            },
            f"{where}.candidates[{index}]",
        )
        if candidate["config_id"] != grid_row["config_id"] or candidate["grid_index"] != index:
            raise GBDTProtocolError(f"{where}: grid declaration order changed")
        if candidate["parameters"] != {
            "max_leaf_nodes": grid_row["max_leaf_nodes"],
            "max_iter": grid_row["max_iter"],
            "min_samples_leaf": config["min_samples_leaf"][track],
        }:
            raise GBDTProtocolError(f"{where}: candidate parameters differ from frozen config")
        values = candidate["fold_objective_values"]
        units = candidate["fold_objective_units"]
        if not isinstance(values, list) or not isinstance(units, list) or len(values) != n_splits or len(units) != n_splits:
            raise GBDTProtocolError(f"{where}: fold trace is incomplete")
        clean_values = [_probability(value, f"{where}.fold value") for value in values]
        clean_units = [_integer(unit, f"{where}.fold units", minimum=1) for unit in units]
        expected_mean = (
            float(np.average(clean_values, weights=clean_units))
            if track == "b2"
            else float(np.mean(clean_values))
        )
        expected_std = float(np.std(clean_values, ddof=0))
        _close(candidate["mean_objective"], expected_mean, f"{where}.candidate mean")
        _close(candidate["std_objective"], expected_std, f"{where}.candidate std")
        means.append(expected_mean)
    selected_index = max(range(len(means)), key=lambda index: means[index])
    if (
        selection["selected_grid_index"] != selected_index
        or selection["selected_config_id"] != config["grid"][selected_index]["config_id"]
    ):
        raise GBDTProtocolError(f"{where}: selected config is not the historical optimum/tie winner")
    _close(selection["selected_mean_objective"], means[selected_index], f"{where}.selected mean")


def _validate_classification_metrics(
    metrics: Mapping[str, Any],
    rows: int,
    positives: int,
    where: str,
    *,
    expected_cluster: str,
    expected_draws: int,
) -> None:
    if metrics.get("n") != rows or metrics.get("positives") != positives:
        raise GBDTProtocolError(f"{where}: metric counts do not match the target cohort")
    _close(metrics.get("base_rate"), positives / rows, f"{where}.base_rate")
    _probability(metrics.get("average_precision"), f"{where}.average_precision")
    _validate_interval(metrics.get("average_precision_ci95"), f"{where}.average_precision_ci95")
    uncertainty = metrics.get("uncertainty", {})
    if (
        uncertainty.get("method") != "nonparametric_cluster_bootstrap"
        or uncertainty.get("cluster_unit") != expected_cluster
        or uncertainty.get("draws") != expected_draws
        or uncertainty.get("interval") != "percentile_95"
        or uncertainty.get("ap_only") is not True
    ):
        raise GBDTProtocolError(f"{where}: uncertainty contract mismatch")
    for key, budget in metrics.get("budgets", {}).items():
        requested = _integer(budget.get("requested_k"), f"{where}.{key}.requested_k", minimum=1)
        if key != f"k_{requested}" or budget.get("effective_k") != min(requested, rows):
            raise GBDTProtocolError(f"{where}.{key}: budget identity mismatch")
        hits = _integer(budget.get("hits"), f"{where}.{key}.hits")
        if hits > positives or hits > budget["effective_k"]:
            raise GBDTProtocolError(f"{where}.{key}: impossible hit count")
        _probability(budget.get("precision"), f"{where}.{key}.precision")
        _probability(budget.get("recall"), f"{where}.{key}.recall")
        _probability(budget.get("value_capture"), f"{where}.{key}.value_capture")


def _validate_conditional_metrics(
    metrics: Mapping[str, Any],
    rows: int,
    positives: int,
    groups: int,
    where: str,
    *,
    expected_draws: int,
) -> None:
    if (
        metrics.get("n_candidate_lanes") != rows
        or metrics.get("positive_lanes") != positives
        or metrics.get("n_entry_groups") != groups
        or metrics.get("oracle_conditioning_for_evaluation") is not True
    ):
        raise GBDTProtocolError(f"{where}: conditional metric counts/conditioning mismatch")
    uncertainty = metrics.get("uncertainty", {})
    if (
        uncertainty.get("method") != "nonparametric_entry_group_bootstrap"
        or uncertainty.get("cluster_unit") != "exporter_stage_entry"
        or uncertainty.get("draws") != expected_draws
        or uncertainty.get("interval") != "percentile_95"
    ):
        raise GBDTProtocolError(f"{where}: conditional uncertainty contract mismatch")
    for key, item in metrics.get("at_k", {}).items():
        if not key.startswith("k_"):
            raise GBDTProtocolError(f"{where}: invalid k key")
        for metric in ("macro_recall", "micro_recall", "macro_value_capture", "micro_value_capture"):
            _probability(item.get(metric), f"{where}.{key}.{metric}")
        _validate_interval(item.get("macro_recall_ci95"), f"{where}.{key}.recall_ci")
        _validate_interval(item.get("macro_value_capture_ci95"), f"{where}.{key}.value_ci")


def _verify_file_hash(path: Path, expected: str, where: str) -> None:
    if not path.is_file():
        raise GBDTProtocolError(f"{where}: required source is missing: {path}")
    if _sha256(path) != expected:
        raise GBDTProtocolError(f"{where}: source hash mismatch: {path}")


def validate_payload(
    payload: Mapping[str, Any],
    *,
    verify_sources: bool = True,
    config_path: Path = DEFAULT_CONFIG,
) -> None:
    config = load_frozen_config(config_path)
    _exact_keys(
        payload,
        {
            "schema_version",
            "benchmark_version",
            "status",
            "generated_at_utc",
            "claim_scope",
            "config",
            "protocol",
            "feature_registry",
            "inputs",
            "runtime",
            "chains",
            "macro_summary",
        },
        "result",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["benchmark_version"] != BENCHMARK_VERSION
        or payload["status"] != STATUS
        or payload["claim_scope"] != config["claim_scope"]
    ):
        raise GBDTProtocolError("result identity/status/claim scope mismatch")
    generated = _iso_datetime(payload["generated_at_utc"], "result.generated_at_utc")
    frozen = _iso_datetime(config["frozen_at_utc"], "config.frozen_at_utc")
    if generated <= frozen:
        raise GBDTProtocolError("result predates the frozen GBDT configuration")
    expected_config = _config_reference(config_path)
    if payload["config"] != expected_config:
        raise GBDTProtocolError("result is not bound to the current frozen config")

    protocol = payload["protocol"]
    expected_protocol = {
        "selection_window": "1998-2002 -> 2008-2012",
        "frozen_target_window": "2008-2012 -> 2018-2022",
        "selection_source": "fold2 only",
        "read_gate": config["main_evaluation"]["read_gate"],
        "target_access": config["main_evaluation"]["target_access"],
        "target_labels_used_for_model_selection_imputation_or_calibration": False,
        "transductive_split_used": False,
        "all_models_frozen_before_any_main_read": True,
        "b2_all_lanes_scored_before_target_conditioning": True,
        "selection_objectives": OBJECTIVE_NAMES,
        "selection_group_units": GROUP_UNITS,
        "bootstrap": config["uncertainty"],
        "main_reporting": config["main_evaluation"],
    }
    if protocol != expected_protocol:
        raise GBDTProtocolError("result protocol differs from the frozen config/read gate")
    if payload["feature_registry"] != {track: list(FEATURES[track]) for track in ("a", "b1", "b2")}:
        raise GBDTProtocolError("result feature registry changed")

    inputs = _exact_keys(payload["inputs"], {"candidate_files", "public_sources"}, "result.inputs")
    sources = _exact_keys(
        inputs["public_sources"],
        {RUNNER_SOURCE_ROLE, SHARED_SOURCE_ROLE, CONFIG_ROLE},
        "result.inputs.public_sources",
    )
    for role, digest in sources.items():
        _hex(digest, f"source hash {role}")
        if verify_sources:
            _verify_file_hash(ROOT / role, digest, f"public source {role}")

    runtime = _exact_keys(
        payload["runtime"],
        {
            "python",
            "platform",
            "cpu_model",
            "logical_cpu_cores",
            "numpy",
            "pandas",
            "scikit_learn",
            "wall_elapsed_seconds",
            "fit_count_upper_bound",
        },
        "result.runtime",
    )
    for field in ("python", "platform", "cpu_model", "numpy", "pandas", "scikit_learn"):
        if not isinstance(runtime[field], str) or not runtime[field].strip():
            raise GBDTProtocolError(f"result.runtime.{field} must be a non-empty string")
    _integer(runtime["logical_cpu_cores"], "result.runtime.logical_cpu_cores", minimum=1)
    if _finite(runtime["wall_elapsed_seconds"], "result.runtime.wall_elapsed_seconds") <= 0:
        raise GBDTProtocolError("result.runtime.wall_elapsed_seconds must be positive")
    expected_fit_bound = len(cpu.CHAINS) * 3 * (
        len(config["grid"]) * int(config["historical_selection"]["maximum_group_folds"]) + 1
    )
    if runtime["fit_count_upper_bound"] != expected_fit_bound:
        raise GBDTProtocolError("result.runtime.fit_count_upper_bound changed")

    candidates = inputs["candidate_files"]
    if not isinstance(candidates, list) or len(candidates) != 24:
        raise GBDTProtocolError("result must bind exactly 24 historical/main candidate files")
    candidate_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_roles = ("history_track_a", "history_track_b", "target_track_a", "target_track_b")
    for index, record in enumerate(candidates):
        _exact_keys(
            record,
            {"chain", "role", "path", "sha256", "rows", "positive_lanes", "early_window", "late_window"},
            f"candidate_files[{index}]",
        )
        key = (record["chain"], record["role"])
        if key in candidate_map or key[0] not in cpu.CHAINS or key[1] not in expected_roles:
            raise GBDTProtocolError(f"invalid or duplicate candidate role {key!r}")
        candidate_map[key] = record
        suffix = "_fold2" if record["role"].startswith("history_") else ""
        prefix = "candidates" if record["role"].endswith("track_a") else "candidates_firsttime"
        expected_path = f"data/processed_v2/{prefix}_{record['chain']}{suffix}.csv"
        if record["path"] != expected_path:
            raise GBDTProtocolError(f"candidate path differs from canonical role: {record['path']!r}")
        _hex(record["sha256"], f"candidate hash {key}")
        _integer(record["rows"], f"candidate rows {key}", minimum=1)
        _integer(record["positive_lanes"], f"candidate positives {key}")
        expected_windows = (
            ("1998-2002", "2008-2012")
            if record["role"].startswith("history_")
            else ("2008-2012", "2018-2022")
        )
        if (record["early_window"], record["late_window"]) != expected_windows:
            raise GBDTProtocolError(f"candidate windows changed for {key}")
        if verify_sources:
            _verify_file_hash(ROOT / expected_path, record["sha256"], f"candidate source {key}")
    if set(candidate_map) != {(chain, role) for chain in cpu.CHAINS for role in expected_roles}:
        raise GBDTProtocolError("candidate inventory does not cover all six chains and four roles")

    chains = _exact_keys(payload["chains"], cpu.CHAINS, "result.chains")
    draws = int(config["uncertainty"]["draws"])
    for chain in cpu.CHAINS:
        chain_payload = chains[chain]
        _exact_keys(chain_payload, {"protocol_audit", *TRACK_ORDER}, f"result.chains.{chain}")
        audit = chain_payload["protocol_audit"]
        _exact_keys(
            audit,
            {
                "target_loaded_after_all_18_models_frozen",
                "target_labels_used_for_training_selection_imputation_or_calibration",
                "transductive_split_used",
                "b2_all_lanes_scored_before_target_conditioning",
                *expected_roles,
            },
            f"result.chains.{chain}.protocol_audit",
        )
        if (
            audit["target_loaded_after_all_18_models_frozen"] is not True
            or audit["target_labels_used_for_training_selection_imputation_or_calibration"] is not False
            or audit["transductive_split_used"] is not False
            or audit["b2_all_lanes_scored_before_target_conditioning"] is not True
        ):
            raise GBDTProtocolError(f"{chain}: leakage/read-gate audit failed")
        for role in expected_roles:
            record = candidate_map[(chain, role)]
            source = audit[role]
            for field in ("path", "sha256", "rows", "positive_lanes", "early_window", "late_window"):
                if source.get(field) != record[field]:
                    raise GBDTProtocolError(f"{chain}/{role}: audit and source inventory differ")

        for track in TRACK_ORDER:
            short = TRACK_SHORT[track]
            task = chain_payload[track]
            models = _exact_keys(task.get("models"), {MODEL_KEY}, f"{chain}/{track}/models")
            model = models[MODEL_KEY]
            _exact_keys(model, {"model", "metrics"}, f"{chain}/{track}/{MODEL_KEY}")
            selection = model["model"].get("selection")
            if not isinstance(selection, Mapping):
                raise GBDTProtocolError(f"{chain}/{track}: missing selection record")
            history_rows = _integer(task.get("history_rows"), f"{chain}/{track}.history_rows", minimum=1)
            history_positives = _integer(task.get("history_positives"), f"{chain}/{track}.history_positives", minimum=1)
            target_rows = _integer(task.get("target_rows"), f"{chain}/{track}.target_rows", minimum=1)
            target_positives = _integer(task.get("target_positives"), f"{chain}/{track}.target_positives", minimum=1)
            _validate_selection(selection, config, short, history_rows, history_positives, f"{chain}/{track}.selection")
            if short in {"a", "b1"}:
                _validate_classification_metrics(
                    model["metrics"],
                    target_rows,
                    target_positives,
                    f"{chain}/{track}.metrics",
                    expected_cluster="exporter",
                    expected_draws=draws,
                )
            else:
                groups = _integer(task.get("target_entry_groups"), f"{chain}/{track}.target_entry_groups", minimum=1)
                _validate_conditional_metrics(
                    model["metrics"],
                    target_rows,
                    target_positives,
                    groups,
                    f"{chain}/{track}.metrics",
                    expected_draws=draws,
                )

    macro = _exact_keys(payload["macro_summary"], TRACK_ORDER, "result.macro_summary")
    expected_macro = _macro_summary(chains, config)
    for track in TRACK_ORDER:
        if macro[track].keys() != expected_macro[track].keys():
            raise GBDTProtocolError(f"macro summary keys changed for {track}")
        for scalar in ("headline_metric", "realized_value_metric", "aggregation", "chain_registry", "model", "chain_level_ci95", "inference_note"):
            if macro[track][scalar] != expected_macro[track][scalar]:
                raise GBDTProtocolError(f"macro summary field changed for {track}/{scalar}")
        for section in ("headline", "realized_value"):
            if macro[track][section]["per_chain"] != expected_macro[track][section]["per_chain"]:
                raise GBDTProtocolError(f"macro per-chain values changed for {track}/{section}")
            _close(macro[track][section]["macro_mean"], expected_macro[track][section]["macro_mean"], f"macro {track}/{section}/mean")
            _close(macro[track][section]["std_across_chains"], expected_macro[track][section]["std_across_chains"], f"macro {track}/{section}/std")
    _assert_privacy(payload)


def verify_existing_output(
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
) -> None:
    payload = _strict_json_load(json_path.resolve())
    if json_path.read_bytes() != _strict_json_bytes(payload):
        raise GBDTProtocolError(f"{json_path}: JSON bytes are not canonical")
    validate_payload(payload, verify_sources=True)
    expected_csv = _csv_bytes(payload)
    try:
        actual_csv = csv_path.read_bytes()
    except OSError as exc:
        raise GBDTProtocolError(f"cannot read GBDT CSV from {csv_path}") from exc
    if actual_csv != expected_csv:
        raise GBDTProtocolError(f"{csv_path}: CSV is stale or noncanonical")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="verify committed JSON/CSV, current config, source hashes, aggregates, and privacy",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_output:
            verify_existing_output(args.json_out.resolve(), args.csv_out.resolve())
            print(f"verified GBDT baseline artifact: {args.json_out}")
            return 0
        if args.config.resolve() != DEFAULT_CONFIG.resolve():
            raise GBDTProtocolError("formal run refuses a noncanonical config path")
        payload = run(args.data_dir.resolve(), args.config.resolve())
        validate_payload(payload, verify_sources=True, config_path=args.config.resolve())
        write_outputs(payload, args.json_out.resolve(), args.csv_out.resolve())
        verify_existing_output(args.json_out.resolve(), args.csv_out.resolve())
        print(f"wrote verified GBDT baseline artifact: {args.json_out}")
        return 0
    except GBDTProtocolError as exc:
        print(f"GBDT baseline protocol failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
