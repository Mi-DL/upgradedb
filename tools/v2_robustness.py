#!/usr/bin/env python3
"""Prespecified robustness suite for the UPGRADE-BENCH v2 CPU systems.

The suite is deliberately separate from the GPU runner.  Every supervised CPU
system is fixed by choices read from a fully verified rolling-CPU artifact,
fitted on fold2, and frozen for all six chains before a main-window table or raw
late-window label is opened.  Main labels are used only to evaluate
prespecified slices.  There is intentionally no hard-coded or fallback choice
table in this module.

The 50/100/250 kUSD threshold analyses are *outcome-only diagnostics*: they
relabel the fixed default-100-kUSD candidate cohort.  They do not rebuild early
candidate eligibility and therefore must never be described as alternate
benchmark cohorts.  Persistence is also auxiliary: annual stage totals are read
from raw BACI and the label is ``>=3 of 5 years above 100 kUSD``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TOOLS))

import universe as U  # noqa: E402
import v2_rolling_cpu_baselines as cpu  # noqa: E402


SCHEMA_VERSION = "upgrade-bench-v2/robustness/2"
DEFAULT_DATA = ROOT / "data" / "processed_v2"
DEFAULT_RAW = Path(os.environ.get("VCU_RAW", ROOT / "data" / "raw")) / "BACI_HS92_V202401b.zip"
DEFAULT_ROLLING = ROOT / "results_v2" / "metrics" / "rolling_cpu_baselines.json"
DEFAULT_JSON = ROOT / "results_v2" / "metrics" / "v2_robustness.json"
DEFAULT_CSV = ROOT / "results_v2" / "metrics" / "v2_robustness.csv"

CHAINS = tuple(cpu.CHAINS)
LATE_YEARS = (2018, 2019, 2020, 2021, 2022)
THRESHOLDS_KUSD = (50.0, 100.0, 250.0)
DEFAULT_THRESHOLD_KUSD = 100.0
PERSISTENCE_MIN_ACTIVE_YEARS = 3
HUBS = frozenset({"NLD", "BEL", "SGP", "HKG", "ARE", "PAN", "CHE"})
BAD_ISO = frozenset({"ANT", "SCG", "YUG", "SUN", "CSK", "DDR"})
BASE_SEED = 20260712
KEYS = ("i_iso", "j_iso", "stage")
RAW_KEYS = ("chain", *KEYS)
TRACKS = ("A", "B1", "B2")
ROLLING_INPUT_ROLES = (
    "history_track_a",
    "history_track_b",
    "target_track_a",
    "target_track_b",
)

A_MODELS = ("size", "gravity", "historical_logistic_size_gravity")
B1_MODELS = ("upstream_capacity", "historical_logistic_structural")
B2_MODELS = (
    "processed_importer_demand",
    "gravity",
    "historical_logistic_demand_gravity",
)

# This registry describes where and how a *historical* choice must be recorded
# in the rolling artifact.  It contains no selected values.  Selected C values
# are accepted only when they are the deterministic winners of the verified
# artifact's complete grouped-CV traces.
ROLLING_CHOICE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "A": {
        "track_key": "track_a_destination_extension",
        "model_key": "historical_logistic_size_gravity",
        "model_keys": A_MODELS,
        "feature_names": tuple(cpu.TRACK_A_FEATURES),
        "objective": "historical_group_cv_average_precision",
        "objective_definition": "lane_or_entry_average_precision_on_each_validation_fold",
        "objective_aggregation": "unweighted_mean_over_validation_folds",
        "group_unit": "exporter",
        "ranking_tie_break": "average_precision_is_score_tie_block_invariant",
    },
    "B1": {
        "track_key": "track_b1_processed_export_stage_entry",
        "model_key": "historical_logistic_structural",
        "model_keys": B1_MODELS,
        "feature_names": tuple(cpu.TRACK_B1_FEATURES),
        "objective": "historical_group_cv_average_precision",
        "objective_definition": "lane_or_entry_average_precision_on_each_validation_fold",
        "objective_aggregation": "unweighted_mean_over_validation_folds",
        "group_unit": "exporter",
        "ranking_tie_break": "average_precision_is_score_tie_block_invariant",
    },
    "B2": {
        "track_key": "track_b2_conditional_destination_ranking",
        "model_key": "historical_logistic_demand_gravity",
        "model_keys": B2_MODELS,
        "feature_names": tuple(cpu.TRACK_B2_FEATURES),
        "objective": "historical_group_cv_per_positive_entry_macro_recall_at_3",
        "objective_definition": (
            "within each positive exporter-stage validation entry, rank destinations and "
            "compute recall@3; then take the unweighted mean over all out-of-fold "
            "validation entries"
        ),
        "objective_aggregation": (
            "fold means weighted by validation-entry count, exactly equivalent to the "
            "unweighted macro mean over all out-of-fold positive entries"
        ),
        "group_unit": "exporter_stage_entry",
        "ranking_tie_break": (
            "destination_iso_ascending_within_entry_for_exact_score_ties"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _portable_command(argv: Sequence[str]) -> list[str]:
    """Record a reproducible command without leaking a host Python/user path."""
    if not argv:
        return ["python", "tools/v2_robustness.py"]
    script = Path(argv[0])
    try:
        script_text = script.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        script_text = script.name
    return ["python", script_text, *map(str, argv[1:])]


def _strict_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_strict_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return BASE_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000_000


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: boolean is not a numeric choice")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite numeric value") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite numeric value")
    return number


def _require_exact_keys(
    value: Any,
    expected: Iterable[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected an object")
    expected_set = set(expected)
    observed_set = set(map(str, value.keys()))
    if observed_set != expected_set:
        missing = sorted(expected_set.difference(observed_set))
        extra = sorted(observed_set.difference(expected_set))
        raise ValueError(f"{context}: missing={missing}, extra={extra}")
    return value


def _choice_tuple(value: Any, *, context: str) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    chains = _require_exact_keys(value, CHAINS, context=context)
    frozen: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for chain in CHAINS:
        tracks = _require_exact_keys(chains[chain], TRACKS, context=f"{context}/{chain}")
        frozen_tracks: list[tuple[str, float]] = []
        for track in TRACKS:
            c_value = _finite_float(tracks[track], context=f"{context}/{chain}/{track}")
            if c_value not in cpu.C_GRID:
                raise ValueError(
                    f"{context}/{chain}/{track}: C={c_value} is outside the historical grid"
                )
            frozen_tracks.append((track, c_value))
        frozen.append((chain, tuple(frozen_tracks)))
    return tuple(frozen)


@dataclass(frozen=True)
class RollingChoiceFreeze:
    """Immutable selection extracted from one verified rolling artifact."""

    path: Path
    artifact_sha256: str
    choices: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    verified_input_hashes: int

    def selected_c(self, chain: str, track: str) -> float:
        for chain_name, tracks in self.choices:
            if chain_name == chain:
                for track_name, value in tracks:
                    if track_name == track:
                        return value
        raise KeyError(f"unknown frozen choice {chain}/{track}")

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            chain: {track: value for track, value in tracks}
            for chain, tracks in self.choices
        }

    @property
    def choices_sha256(self) -> str:
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _validate_rolling_protocol(payload: Mapping[str, Any]) -> int:
    """Enforce the complete historical-selection and 24-input provenance gate."""
    protocol = payload.get("protocol", {})
    expected_protocol = {
        "selection_window": "1998-2002 -> 2008-2012",
        "frozen_target_window": "2008-2012 -> 2018-2022",
        "selection_source": "fold2 only",
        "target_labels_used_for_model_selection": False,
        "target_labels_used_for_imputation_scaling_or_calibration": False,
        "transductive_split_used": False,
        "main_target_models_compared_without_post_hoc_champion_selection": True,
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            raise ValueError(
                f"rolling CPU artifact has nonhistorical or unsafe protocol field "
                f"{field!r}: expected {expected!r}, observed {protocol.get(field)!r}"
            )

    expected_objectives = {
        "track_a": "historical_exporter_group_cv_average_precision",
        "track_b1": "historical_exporter_group_cv_average_precision",
        "track_b2": (
            "historical_exporter_stage_entry_group_cv_per_positive_entry_"
            "macro_recall_at_3"
        ),
    }
    if protocol.get("selection_objectives") != expected_objectives:
        raise ValueError("rolling CPU artifact has nonhistorical selection objectives")

    chains = _require_exact_keys(
        payload.get("chains", {}), CHAINS, context="rolling artifact chains"
    )
    expected_chain_keys = {
        "protocol_audit",
        *(spec["track_key"] for spec in ROLLING_CHOICE_SPECS.values()),
    }
    checked = 0
    audited_paths: set[str] = set()
    for chain in CHAINS:
        chain_payload = _require_exact_keys(
            chains[chain], expected_chain_keys, context=f"rolling artifact/{chain}"
        )
        audit = chain_payload["protocol_audit"]
        if not isinstance(audit, Mapping):
            raise ValueError(f"rolling artifact/{chain}/protocol_audit: expected an object")
        if audit.get("target_loaded_after_all_models_frozen") is not True:
            raise ValueError(f"rolling artifact/{chain}: target was not opened after global freeze")
        if (
            audit.get(
                "target_labels_used_for_training_selection_imputation_or_calibration"
            )
            is not False
        ):
            raise ValueError(f"rolling artifact/{chain}: target-label training audit is unsafe")
        if audit.get("transductive_split_used") is not False:
            raise ValueError(f"rolling artifact/{chain}: transductive audit is unsafe")
        expected_paths = {
            "history_track_a": f"data/processed_v2/candidates_{chain}_fold2.csv",
            "history_track_b": (
                f"data/processed_v2/candidates_firsttime_{chain}_fold2.csv"
            ),
            "target_track_a": f"data/processed_v2/candidates_{chain}.csv",
            "target_track_b": f"data/processed_v2/candidates_firsttime_{chain}.csv",
        }
        for role in ROLLING_INPUT_ROLES:
            record = audit.get(role)
            if not isinstance(record, Mapping):
                raise ValueError(f"rolling artifact/{chain}: missing {role} input audit")
            path = record.get("path")
            digest = record.get("sha256")
            if not isinstance(path, str) or not path:
                raise ValueError(f"rolling artifact/{chain}/{role}: missing input path")
            if path != expected_paths[role]:
                raise ValueError(
                    f"rolling artifact/{chain}/{role}: noncanonical or nonhistorical "
                    f"input path {path!r}; expected {expected_paths[role]!r}"
                )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
            ):
                raise ValueError(f"rolling artifact/{chain}/{role}: invalid input sha256")
            if path in audited_paths:
                raise ValueError(f"rolling artifact: duplicate input audit path {path}")
            audited_paths.add(path)
            checked += 1
    if checked != 24 or len(audited_paths) != 24:
        raise ValueError(f"rolling CPU artifact must attest exactly 24 inputs, found {checked}")
    return checked


def _selection_from_track(
    chain_payload: Mapping[str, Any],
    chain: str,
    track: str,
) -> float:
    spec = ROLLING_CHOICE_SPECS[track]
    track_key = str(spec["track_key"])
    model_key = str(spec["model_key"])
    track_payload = chain_payload.get(track_key)
    if not isinstance(track_payload, Mapping):
        raise ValueError(f"rolling artifact/{chain}: missing choice task {track_key}")
    models = _require_exact_keys(
        track_payload.get("models", {}),
        spec["model_keys"],
        context=f"rolling artifact/{chain}/{track_key}/models",
    )
    model_payload = models[model_key]
    if not isinstance(model_payload, Mapping):
        raise ValueError(f"rolling artifact/{chain}/{track}: missing historical model payload")
    model = model_payload.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("selection"), Mapping):
        raise ValueError(f"rolling artifact/{chain}/{track}: missing historical selection")
    selection = model["selection"]

    historical_fields = {
        "feature_names": list(spec["feature_names"]),
        "objective": spec["objective"],
        "objective_definition": spec["objective_definition"],
        "objective_aggregation": spec["objective_aggregation"],
        "group_unit": spec["group_unit"],
        "ranking_tie_break": spec["ranking_tie_break"],
        "train_validation_group_overlap_checked": True,
        "hyperparameter_tie_break": "maximize_mean_objective_then_smaller_C",
    }
    for field, expected in historical_fields.items():
        if selection.get(field) != expected:
            raise ValueError(
                f"rolling artifact/{chain}/{track}: nonhistorical selection field "
                f"{field!r}: expected {expected!r}, observed {selection.get(field)!r}"
            )
    if selection.get("c_grid") != list(cpu.C_GRID):
        raise ValueError(f"rolling artifact/{chain}/{track}: historical C grid mismatch")
    n_splits = selection.get("n_splits")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"rolling artifact/{chain}/{track}: invalid historical CV split count")

    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(cpu.C_GRID):
        raise ValueError(f"rolling artifact/{chain}/{track}: incomplete historical CV trace")
    candidate_scores: dict[float, float] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"rolling artifact/{chain}/{track}: malformed CV candidate")
        c_value = _finite_float(
            candidate.get("C"), context=f"rolling artifact/{chain}/{track}/candidate[{index}]/C"
        )
        mean = _finite_float(
            candidate.get("mean_objective"),
            context=f"rolling artifact/{chain}/{track}/candidate[{index}]/mean_objective",
        )
        fold_values = candidate.get("fold_objective_values")
        fold_units = candidate.get("fold_objective_units")
        if not isinstance(fold_values, list) or len(fold_values) != n_splits:
            raise ValueError(
                f"rolling artifact/{chain}/{track}: incomplete fold objective trace"
            )
        if not isinstance(fold_units, list) or len(fold_units) != n_splits:
            raise ValueError(f"rolling artifact/{chain}/{track}: incomplete fold unit trace")
        values = [
            _finite_float(
                value,
                context=(
                    f"rolling artifact/{chain}/{track}/candidate[{index}]/"
                    f"fold_objective_values[{fold_index}]"
                ),
            )
            for fold_index, value in enumerate(fold_values)
        ]
        units: list[int] = []
        for fold_index, value in enumerate(fold_units):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"rolling artifact/{chain}/{track}/candidate[{index}]/"
                    f"fold_objective_units[{fold_index}]: expected a positive integer"
                )
            units.append(value)
        if track in ("A", "B1") and units != [1] * n_splits:
            raise ValueError(f"rolling artifact/{chain}/{track}: AP fold units changed")
        recomputed_mean = (
            float(np.average(values, weights=units))
            if track == "B2"
            else float(np.mean(values))
        )
        if not math.isclose(mean, recomputed_mean, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"rolling artifact/{chain}/{track}: candidate mean disagrees with fold trace"
            )
        std = _finite_float(
            candidate.get("std_objective"),
            context=f"rolling artifact/{chain}/{track}/candidate[{index}]/std_objective",
        )
        if not math.isclose(
            std, float(np.std(values, ddof=0)), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                f"rolling artifact/{chain}/{track}: candidate std disagrees with fold trace"
            )
        if c_value in candidate_scores:
            raise ValueError(f"rolling artifact/{chain}/{track}: duplicate CV candidate C={c_value}")
        candidate_scores[c_value] = mean
    if set(candidate_scores) != set(map(float, cpu.C_GRID)):
        raise ValueError(f"rolling artifact/{chain}/{track}: CV trace C grid mismatch")

    expected_c = min(candidate_scores, key=lambda value: (-candidate_scores[value], value))
    selected_c = _finite_float(
        selection.get("selected_C"), context=f"rolling artifact/{chain}/{track}/selected_C"
    )
    if selected_c != expected_c:
        raise ValueError(
            f"rolling artifact/{chain}/{track}: selected_C mismatch; "
            f"deterministic historical winner={expected_c}, recorded={selected_c}"
        )
    selected_mean = _finite_float(
        selection.get("selected_mean_objective"),
        context=f"rolling artifact/{chain}/{track}/selected_mean_objective",
    )
    if not math.isclose(
        selected_mean, candidate_scores[selected_c], rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            f"rolling artifact/{chain}/{track}: selected objective does not match CV trace"
        )
    if selection.get("refit_rows") != track_payload.get("history_rows"):
        raise ValueError(f"rolling artifact/{chain}/{track}: refit/history row mismatch")
    if selection.get("refit_positives") != track_payload.get("history_positives"):
        raise ValueError(f"rolling artifact/{chain}/{track}: refit/history positive mismatch")
    return selected_c


def _freeze_choices_from_verified_rolling(rolling_path: Path) -> RollingChoiceFreeze:
    """Verify first, then parse one immutable, strictly historical choice matrix."""
    rolling_path = rolling_path.resolve()
    if not rolling_path.is_file():
        raise FileNotFoundError(rolling_path)
    before_hash = sha256_file(rolling_path)
    cpu.verify_existing_output(rolling_path)
    raw = rolling_path.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()
    if artifact_hash != before_hash:
        raise ValueError("rolling CPU artifact changed while it was being verified")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{rolling_path}: invalid rolling CPU JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{rolling_path}: rolling CPU JSON root must be an object")

    checked = _validate_rolling_protocol(payload)
    chains = payload["chains"]
    choices = {
        chain: {
            track: _selection_from_track(chains[chain], chain, track)
            for track in TRACKS
        }
        for chain in CHAINS
    }
    return RollingChoiceFreeze(
        path=rolling_path,
        artifact_sha256=artifact_hash,
        choices=_choice_tuple(choices, context="verified rolling choices"),
        verified_input_hashes=checked,
    )


@dataclass
class FrozenChain:
    models: cpu.ChainModels
    history_a_exporters: frozenset[str]
    history_a_importers: frozenset[str]
    history_b_exporters: frozenset[str]
    history_b_importers: frozenset[str]
    input_audit: dict[str, dict[str, Any]]


def _frozen_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    c_value: float,
    seed: int,
    track: str,
) -> cpu.FrozenClassifier:
    if float(c_value) not in cpu.C_GRID:
        raise ValueError(f"{track}: frozen C={c_value} is outside the historical grid")
    pipeline = cpu._pipeline(c_value, seed)
    pipeline.fit(features.loc[:, list(feature_names)], labels)
    selection = {
        "source": "verified rolling-CPU fold2 grouped-CV artifact",
        "selected_on_snapshot": "fold2",
        "selected_C": float(c_value),
        "feature_names": list(feature_names),
        "refit_rows": int(len(labels)),
        "refit_positives": int(labels.sum()),
        "main_labels_used_for_selection": False,
    }
    return cpu.FrozenClassifier(pipeline=pipeline, selection=selection)


def _fit_frozen_chain(
    data_dir: Path,
    chain: str,
    chain_seed: int,
    frozen_c: Mapping[str, float],
) -> FrozenChain:
    """Fit only verified fold2 systems; this function cannot open main files."""
    frozen_c = _require_exact_keys(
        frozen_c, TRACKS, context=f"fit-time frozen choices/{chain}"
    )
    for track in TRACKS:
        value = _finite_float(
            frozen_c[track], context=f"fit-time frozen choices/{chain}/{track}"
        )
        if value not in cpu.C_GRID:
            raise ValueError(f"{chain}/{track}: frozen C={value} is outside the historical grid")
    history_a, audit_a = cpu._read_candidate(data_dir, chain, track="a", historical=True)
    history_b, audit_b = cpu._read_candidate(data_dir, chain, track="b", historical=True)

    a_features = cpu._track_a_features(history_a)
    a_y = history_a["y"].to_numpy(np.int8)
    a_classifier = _frozen_classifier(
        a_features,
        a_y,
        cpu.TRACK_A_FEATURES,
        c_value=frozen_c["A"],
        seed=chain_seed + 11,
        track="A",
    )

    history_entries = cpu._derive_entry_table(history_b)
    b1_y = history_entries["z"].to_numpy(np.int8)
    b1_classifier = _frozen_classifier(
        history_entries,
        b1_y,
        cpu.TRACK_B1_FEATURES,
        c_value=frozen_c["B1"],
        seed=chain_seed + 21,
        track="B1",
    )

    positive_entry_ids = set(
        history_entries.loc[history_entries["z"].eq(1), "entry_id"].astype(str)
    )
    history_b2 = history_b.loc[history_b["entry_id"].isin(positive_entry_ids)].copy()
    b2_features = cpu._track_b2_features(history_b2)
    b2_y = history_b2["y"].to_numpy(np.int8)
    b2_classifier = _frozen_classifier(
        b2_features,
        b2_y,
        cpu.TRACK_B2_FEATURES,
        c_value=frozen_c["B2"],
        seed=chain_seed + 31,
        track="B2",
    )

    models = cpu.ChainModels(
        track_a_raw_size=cpu.FrozenRawScore.fit(a_features["size"], feature="size"),
        track_a_raw_gravity=cpu.FrozenRawScore.fit(
            a_features["log_gravity"], feature="log_gravity"
        ),
        track_a_classifier=a_classifier,
        track_b1_raw_capacity=cpu.FrozenRawScore.fit(
            history_entries["log_upstream_capacity"], feature="log_upstream_capacity"
        ),
        track_b1_classifier=b1_classifier,
        track_b2_raw_demand=cpu.FrozenRawScore.fit(
            b2_features["log_importer_demand"], feature="log_importer_demand"
        ),
        track_b2_raw_gravity=cpu.FrozenRawScore.fit(
            b2_features["log_gravity"], feature="log_gravity"
        ),
        track_b2_classifier=b2_classifier,
        history_a_audit=audit_a,
        history_b_audit=audit_b,
        history_b1_entries=int(len(history_entries)),
        history_b1_positive_entries=int(history_entries["z"].sum()),
        history_b2_lanes=int(len(history_b2)),
        history_b2_positive_lanes=int(history_b2["y"].sum()),
    )
    return FrozenChain(
        models=models,
        history_a_exporters=frozenset(history_a["i_iso"].astype(str)),
        history_a_importers=frozenset(history_a["j_iso"].astype(str)),
        history_b_exporters=frozenset(history_b["i_iso"].astype(str)),
        history_b_importers=frozenset(history_b["j_iso"].astype(str)),
        input_audit={"history_track_a": audit_a, "history_track_b": audit_b},
    )


def _chain_stage_maps(chains: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    hs_to_chain: dict[str, str] = {}
    hs_to_stage: dict[str, str] = {}
    for chain in chains:
        registry = U.get_chain(chain)
        for hs6, stage in registry.hs2stage.items():
            if hs6 in hs_to_chain:
                raise ValueError(f"HS6 {hs6} occurs in multiple requested chains")
            hs_to_chain[hs6] = chain
            hs_to_stage[hs6] = stage
    return hs_to_chain, hs_to_stage


def _country_iso_map(zf: zipfile.ZipFile) -> dict[int, str]:
    try:
        raw = zf.read("country_codes_V202401b.csv")
    except KeyError as exc:
        raise FileNotFoundError("country_codes_V202401b.csv missing from BACI archive") from exc
    countries = pd.read_csv(io.BytesIO(raw)).dropna(subset=["country_code", "country_iso3"])
    return dict(
        zip(
            pd.to_numeric(countries["country_code"], errors="raise").astype(int),
            countries["country_iso3"].astype(str),
        )
    )


def aggregate_candidate_late_years(
    zf: zipfile.ZipFile,
    candidate_keys: pd.DataFrame,
    *,
    iso: Mapping[int, str],
    hs_to_chain: Mapping[str, str],
    hs_to_stage: Mapping[str, str],
    years: Sequence[int] = LATE_YEARS,
    chunk_size: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Independently rebuild annual stage totals for the fixed candidate keys.

    Missing annual rows remain implicit zeros.  No production window-aggregation
    or candidate-label helper is called here.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    required = set(RAW_KEYS)
    if missing := sorted(required.difference(candidate_keys.columns)):
        raise ValueError(f"candidate_keys missing {missing}")
    keys = candidate_keys.loc[:, list(RAW_KEYS)].astype(str).drop_duplicates()
    if bool(keys.duplicated(list(RAW_KEYS)).any()):
        raise AssertionError("candidate keys were not unique after deduplication")

    hs_universe = set(hs_to_chain)
    annual_parts: list[pd.DataFrame] = []
    year_stats: dict[str, Any] = {}
    for year in years:
        member = f"BACI_HS92_Y{int(year)}_V202401b.csv"
        try:
            stream = zf.open(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{member} missing from BACI archive") from exc
        grouped_chunks: list[pd.DataFrame] = []
        source_rows = 0
        retained_rows = 0
        started = time.perf_counter()
        with stream:
            for chunk in pd.read_csv(
                stream,
                usecols=["i", "j", "k", "v"],
                dtype={"k": str},
                chunksize=chunk_size,
            ):
                source_rows += len(chunk)
                chunk["k"] = chunk["k"].str.zfill(6)
                chunk = chunk.loc[chunk["k"].isin(hs_universe)].copy()
                retained_rows += len(chunk)
                if chunk.empty:
                    continue
                chunk["chain"] = chunk["k"].map(hs_to_chain)
                chunk["stage"] = chunk["k"].map(hs_to_stage)
                chunk["i_iso"] = chunk["i"].map(iso)
                chunk["j_iso"] = chunk["j"].map(iso)
                chunk["v"] = pd.to_numeric(chunk["v"], errors="raise")
                chunk = chunk.dropna(subset=[*RAW_KEYS, "v"])
                grouped_chunks.append(
                    chunk.groupby(list(RAW_KEYS), as_index=False, sort=False)["v"].sum()
                )

        if grouped_chunks:
            annual = (
                pd.concat(grouped_chunks, ignore_index=True)
                .groupby(list(RAW_KEYS), as_index=False, sort=False)["v"]
                .sum()
                .merge(keys, on=list(RAW_KEYS), how="inner", validate="one_to_one")
            )
        else:
            annual = pd.DataFrame(columns=[*RAW_KEYS, "v"])
        annual["year"] = int(year)
        annual_parts.append(annual)
        year_stats[str(year)] = {
            "source_rows": int(source_rows),
            "retained_chain_hs6_rows": int(retained_rows),
            "candidate_stage_year_rows": int(len(annual)),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        print(
            f"raw robustness {year}: source={source_rows:,} chain_hs6={retained_rows:,} "
            f"candidate_stage_year={len(annual):,}",
            flush=True,
        )

    annual_all = pd.concat(annual_parts, ignore_index=True)
    sums = annual_all.groupby(list(RAW_KEYS), as_index=False, sort=False)["v"].sum()
    active = (
        annual_all.loc[annual_all["v"].gt(DEFAULT_THRESHOLD_KUSD)]
        .groupby(list(RAW_KEYS), as_index=False, sort=False)["year"]
        .nunique()
        .rename(columns={"year": "active_years_100kusd"})
    )
    result = keys.merge(sums, on=list(RAW_KEYS), how="left", validate="one_to_one").merge(
        active, on=list(RAW_KEYS), how="left", validate="one_to_one"
    )
    result["v"] = result["v"].fillna(0.0)
    result["active_years_100kusd"] = result["active_years_100kusd"].fillna(0).astype(np.int8)
    result["raw_late_calendar_mean_kusd"] = result.pop("v") / float(len(years))
    if bool(result["active_years_100kusd"].gt(len(years)).any()):
        raise AssertionError("active_years exceeded the fixed window length")
    return result.sort_values(list(RAW_KEYS), kind="mergesort").reset_index(drop=True), {
        "years": [int(year) for year in years],
        "window_length": int(len(years)),
        "missing_stage_year_value_kusd": 0.0,
        "annual_stage_total_rule": "sum all registered HS6 values within chain/lane/stage/year",
        "calendar_mean_rule": "sum annual stage totals / 5",
        "active_year_rule": "annual stage total > 100 kUSD",
        "year_stats": year_stats,
        "unique_candidate_keys": int(len(keys)),
        "observed_candidate_stage_year_rows": int(len(annual_all)),
    }


def labels_at_threshold(frame: pd.DataFrame, threshold_kusd: float) -> tuple[np.ndarray, np.ndarray]:
    mean = pd.to_numeric(frame["raw_late_calendar_mean_kusd"], errors="raise").to_numpy(float)
    label = mean > float(threshold_kusd)
    return label.astype(np.int8), np.where(label, mean, 0.0)


def persistence_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    active = pd.to_numeric(frame["active_years_100kusd"], errors="raise").to_numpy(int)
    mean = pd.to_numeric(frame["raw_late_calendar_mean_kusd"], errors="raise").to_numpy(float)
    label = active >= PERSISTENCE_MIN_ACTIVE_YEARS
    return label.astype(np.int8), np.where(label, mean, 0.0)


def _attach_raw_labels(frame: pd.DataFrame, chain: str, raw: pd.DataFrame) -> pd.DataFrame:
    labels = raw.loc[raw["chain"].eq(chain), [*KEYS, "raw_late_calendar_mean_kusd", "active_years_100kusd"]]
    merged = frame.merge(labels, on=list(KEYS), how="left", validate="one_to_one")
    if bool(merged["raw_late_calendar_mean_kusd"].isna().any()):
        raise ValueError(f"raw reconstruction did not cover every {chain} candidate")
    return merged


def _raw_reconciliation(frame: pd.DataFrame) -> dict[str, Any]:
    raw_y, raw_lateval = labels_at_threshold(frame, DEFAULT_THRESHOLD_KUSD)
    stored_y = frame["y"].to_numpy(np.int8)
    stored_lateval = frame["lateval"].to_numpy(float)
    mismatch = int(np.count_nonzero(raw_y != stored_y))
    error = np.abs(raw_lateval - stored_lateval)
    result = {
        "rows": int(len(frame)),
        "stored_positives": int(stored_y.sum()),
        "raw_positives": int(raw_y.sum()),
        "label_mismatches": mismatch,
        "lateval_mismatches_at_1e-6_kusd": int(np.count_nonzero(error > 1e-6)),
        "max_lateval_absolute_error_kusd": float(error.max()) if len(error) else 0.0,
        "active_years_distribution": {
            str(years): int(count)
            for years, count in frame["active_years_100kusd"].value_counts().sort_index().items()
        },
    }
    result["pass"] = bool(
        result["label_mismatches"] == 0 and result["lateval_mismatches_at_1e-6_kusd"] == 0
    )
    if not result["pass"]:
        raise ValueError(f"raw 100-kUSD labels do not reconcile with shipped labels: {result}")
    return result


def _score_main_chain(
    data_dir: Path, chain: str, frozen: FrozenChain
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_a, audit_a = cpu._read_candidate(data_dir, chain, track="a", historical=False)
    target_b, audit_b = cpu._read_candidate(data_dir, chain, track="b", historical=False)
    models = frozen.models

    a_features = cpu._track_a_features(target_a)
    target_a = target_a.copy()
    target_a["score__size"] = models.track_a_raw_size.predict(a_features["size"])
    target_a["score__gravity"] = models.track_a_raw_gravity.predict(a_features["log_gravity"])
    target_a["score__historical_logistic_size_gravity"] = (
        models.track_a_classifier.pipeline.predict_proba(
            a_features.loc[:, list(cpu.TRACK_A_FEATURES)]
        )[:, 1]
    )

    target_entries = cpu._derive_entry_table(target_b)
    target_entries["score__upstream_capacity"] = models.track_b1_raw_capacity.predict(
        target_entries["log_upstream_capacity"]
    )
    target_entries["score__historical_logistic_structural"] = (
        models.track_b1_classifier.pipeline.predict_proba(
            target_entries.loc[:, list(cpu.TRACK_B1_FEATURES)]
        )[:, 1]
    )

    b2_features = cpu._track_b2_features(target_b)
    target_b = target_b.copy()
    target_b["score__processed_importer_demand"] = models.track_b2_raw_demand.predict(
        b2_features["log_importer_demand"]
    )
    target_b["score__gravity"] = models.track_b2_raw_gravity.predict(
        b2_features["log_gravity"]
    )
    target_b["score__historical_logistic_demand_gravity"] = (
        models.track_b2_classifier.pipeline.predict_proba(
            b2_features.loc[:, list(cpu.TRACK_B2_FEATURES)]
        )[:, 1]
    )
    return target_a, target_b, target_entries, {
        "target_track_a": audit_a,
        "target_track_b": audit_b,
    }


def _classification_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    label: np.ndarray,
    lateval: np.ndarray,
    clusters: Sequence[str],
    cluster_unit: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    label = np.asarray(label, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    lateval = np.asarray(lateval, dtype=float)
    if not (len(frame) == len(label) == len(score) == len(lateval) == len(clusters)):
        raise ValueError("classification vectors are misaligned")
    if not np.isfinite(score).all() or not np.isfinite(lateval).all():
        raise ValueError("classification vectors contain non-finite values")
    positives = int(label.sum())
    classes = int(np.unique(label).size) if len(label) else 0
    result: dict[str, Any] = {
        "n": int(len(label)),
        "positives": positives,
        "base_rate": float(label.mean()) if len(label) else None,
        "average_precision": None,
        "average_precision_ci95": None,
        "roc_auc": None,
        "total_observed_late_value_kusd": float(lateval.sum()),
        "status": "complete" if classes == 2 else "single_class_or_empty",
        "uncertainty": {
            "method": "nonparametric_cluster_bootstrap",
            "cluster_unit": cluster_unit,
            "clusters": int(pd.Series(clusters, dtype="string").nunique()),
            "draws": int(bootstrap),
            "seed": int(seed),
            "interval": "percentile_95",
        },
    }
    if classes == 2:
        result["average_precision"] = float(average_precision_score(label, score))
        result["roc_auc"] = float(roc_auc_score(label, score))
        result["average_precision_ci95"] = cpu._cluster_ap_ci(
            label, score, clusters, draws=bootstrap, seed=seed
        )
    return result


def _conditional_metrics_for_slice(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    label: np.ndarray,
    lateval: np.ndarray,
    mask: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    work = frame.loc[mask, ["entry_id", "i_iso", "j_iso", "stage"]].copy()
    work["y"] = np.asarray(label, dtype=np.int8)[mask]
    work["lateval"] = np.asarray(lateval, dtype=float)[mask]
    score_slice = np.asarray(score, dtype=float)[mask]
    groups_before = int(work["entry_id"].nunique())
    positive_groups = set(
        work.groupby("entry_id", sort=False, observed=True)["y"].max().loc[lambda x: x.eq(1)].index
    )
    keep = work["entry_id"].isin(positive_groups).to_numpy(bool)
    work = work.loc[keep].reset_index(drop=True)
    score_slice = score_slice[keep]
    audit = {
        "candidate_rows_before_entry_reconditioning": int(mask.sum()),
        "entry_groups_before_entry_reconditioning": groups_before,
        "entry_groups_after_entry_reconditioning": int(len(positive_groups)),
        "dropped_zero_positive_entry_groups": int(groups_before - len(positive_groups)),
        "conditioning": "at least one positive destination under this exact slice/label definition",
    }
    if work.empty:
        return {
            **audit,
            "status": "no_positive_entry_groups",
            "n_entry_groups": 0,
            "n_candidate_lanes": 0,
            "positive_lanes": 0,
            "at_k": {},
            "uncertainty": {
                "method": "nonparametric_entry_group_bootstrap",
                "cluster_unit": "exporter_stage_entry",
                "draws": int(bootstrap),
                "seed": int(seed),
            },
        }
    metrics = cpu._conditional_metrics(
        work,
        score_slice,
        bootstrap_draws=bootstrap,
        seed=seed,
    )
    metrics.update(audit)
    metrics["status"] = "complete"
    metrics["uncertainty"]["cluster_unit"] = "exporter_stage_entry"
    return metrics


def _entry_frame_for_labels(
    b_lanes: pd.DataFrame, label: np.ndarray, lateval: np.ndarray, scored_entries: pd.DataFrame
) -> pd.DataFrame:
    work = b_lanes.copy()
    work["y"] = np.asarray(label, dtype=np.int8)
    work["lateval"] = np.asarray(lateval, dtype=float)
    entries = cpu._derive_entry_table(work)
    score_columns = [f"score__{name}" for name in B1_MODELS]
    return entries.merge(
        scored_entries.loc[:, ["entry_id", *score_columns]],
        on="entry_id",
        how="left",
        validate="one_to_one",
    )


def _evaluate_slice(
    chain: str,
    sensitivity: str,
    slice_name: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    scored_entries: pd.DataFrame,
    *,
    a_label: np.ndarray,
    a_lateval: np.ndarray,
    b_label: np.ndarray,
    b_lateval: np.ndarray,
    a_mask: np.ndarray,
    b1_mask_fn: Callable[[pd.DataFrame], np.ndarray] | None,
    b2_mask: np.ndarray,
    bootstrap: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {"tracks": {}}
    a_mask = np.asarray(a_mask, dtype=bool)
    b2_mask = np.asarray(b2_mask, dtype=bool)
    a_slice = a.loc[a_mask]
    output["tracks"]["A"] = {
        "unit": "exporter_stage_destination",
        "models": {
            model: _classification_metrics(
                a_slice,
                a.loc[a_mask, f"score__{model}"].to_numpy(float),
                label=np.asarray(a_label)[a_mask],
                lateval=np.asarray(a_lateval)[a_mask],
                clusters=a_slice["i_iso"].astype(str).to_numpy(),
                cluster_unit="exporter",
                bootstrap=bootstrap,
                seed=_stable_seed(chain, sensitivity, slice_name, "A", model),
            )
            for model in A_MODELS
        },
    }

    entries = _entry_frame_for_labels(b, b_label, b_lateval, scored_entries)
    if b1_mask_fn is not None:
        b1_mask = np.asarray(b1_mask_fn(entries), dtype=bool)
        entry_slice = entries.loc[b1_mask]
        output["tracks"]["B1"] = {
            "unit": "exporter_stage",
            "models": {
                model: _classification_metrics(
                    entry_slice,
                    entry_slice[f"score__{model}"].to_numpy(float),
                    label=entry_slice["z"].to_numpy(np.int8),
                    lateval=entry_slice["entry_lateval"].to_numpy(float),
                    clusters=entry_slice["i_iso"].astype(str).to_numpy(),
                    cluster_unit="exporter",
                    bootstrap=bootstrap,
                    seed=_stable_seed(chain, sensitivity, slice_name, "B1", model),
                )
                for model in B1_MODELS
            },
        }

    output["tracks"]["B2"] = {
        "unit": "destination_within_actual_exporter_stage_entry",
        "models": {
            model: _conditional_metrics_for_slice(
                b,
                b[f"score__{model}"].to_numpy(float),
                label=b_label,
                lateval=b_lateval,
                mask=b2_mask,
                bootstrap=bootstrap,
                seed=_stable_seed(chain, sensitivity, slice_name, "B2", model),
            )
            for model in B2_MODELS
        },
    }
    return output


def _evaluate_chain_sensitivities(
    chain: str,
    frozen: FrozenChain,
    a: pd.DataFrame,
    b: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    bootstrap: int,
) -> dict[str, Any]:
    default_a_y, default_a_value = labels_at_threshold(a, DEFAULT_THRESHOLD_KUSD)
    default_b_y, default_b_value = labels_at_threshold(b, DEFAULT_THRESHOLD_KUSD)
    all_a = np.ones(len(a), dtype=bool)
    all_b = np.ones(len(b), dtype=bool)

    result: dict[str, Any] = {
        "identity": {},
        "entity_exclusion": {},
        "threshold_outcome_only": {},
        "persistence": {},
    }

    identity_specs = {
        "exporter_seen": (
            a["i_iso"].isin(frozen.history_a_exporters).to_numpy(),
            lambda frame: frame["i_iso"].isin(frozen.history_b_exporters).to_numpy(),
            b["i_iso"].isin(frozen.history_b_exporters).to_numpy(),
        ),
        "exporter_unseen": (
            ~a["i_iso"].isin(frozen.history_a_exporters).to_numpy(),
            lambda frame: ~frame["i_iso"].isin(frozen.history_b_exporters).to_numpy(),
            ~b["i_iso"].isin(frozen.history_b_exporters).to_numpy(),
        ),
        "importer_seen": (
            a["j_iso"].isin(frozen.history_a_importers).to_numpy(),
            None,
            b["j_iso"].isin(frozen.history_b_importers).to_numpy(),
        ),
        "importer_unseen": (
            ~a["j_iso"].isin(frozen.history_a_importers).to_numpy(),
            None,
            ~b["j_iso"].isin(frozen.history_b_importers).to_numpy(),
        ),
    }
    for name, (mask_a, mask_b1, mask_b2) in identity_specs.items():
        result["identity"][name] = _evaluate_slice(
            chain,
            "identity",
            name,
            a,
            b,
            entries,
            a_label=default_a_y,
            a_lateval=default_a_value,
            b_label=default_b_y,
            b_lateval=default_b_value,
            a_mask=mask_a,
            b1_mask_fn=mask_b1,
            b2_mask=mask_b2,
            bootstrap=bootstrap,
        )

    exclusion_specs = {
        "exclude_hubs": HUBS,
        "exclude_bad_iso": BAD_ISO,
        "exclude_hubs_and_bad_iso": HUBS | BAD_ISO,
    }
    for name, excluded in exclusion_specs.items():
        mask_a = (~a["i_iso"].isin(excluded) & ~a["j_iso"].isin(excluded)).to_numpy()
        mask_b2 = (~b["i_iso"].isin(excluded) & ~b["j_iso"].isin(excluded)).to_numpy()
        result["entity_exclusion"][name] = _evaluate_slice(
            chain,
            "entity_exclusion",
            name,
            a,
            b,
            entries,
            a_label=default_a_y,
            a_lateval=default_a_value,
            b_label=default_b_y,
            b_lateval=default_b_value,
            a_mask=mask_a,
            b1_mask_fn=lambda frame, excluded=excluded: ~frame["i_iso"].isin(excluded).to_numpy(),
            b2_mask=mask_b2,
            bootstrap=bootstrap,
        )

    for threshold in THRESHOLDS_KUSD:
        a_y, a_value = labels_at_threshold(a, threshold)
        b_y, b_value = labels_at_threshold(b, threshold)
        name = f"threshold_{int(threshold)}_kusd"
        result["threshold_outcome_only"][name] = _evaluate_slice(
            chain,
            "threshold_outcome_only",
            name,
            a,
            b,
            entries,
            a_label=a_y,
            a_lateval=a_value,
            b_label=b_y,
            b_lateval=b_value,
            a_mask=all_a,
            b1_mask_fn=lambda frame: np.ones(len(frame), dtype=bool),
            b2_mask=all_b,
            bootstrap=bootstrap,
        )

    a_y, a_value = persistence_labels(a)
    b_y, b_value = persistence_labels(b)
    result["persistence"]["active_at_least_3_of_5_years_above_100_kusd"] = _evaluate_slice(
        chain,
        "persistence",
        "active_at_least_3_of_5_years_above_100_kusd",
        a,
        b,
        entries,
        a_label=a_y,
        a_lateval=a_value,
        b_label=b_y,
        b_lateval=b_value,
        a_mask=all_a,
        b1_mask_fn=lambda frame: np.ones(len(frame), dtype=bool),
        b2_mask=all_b,
        bootstrap=bootstrap,
    )
    return result


def _csv_rows(chains: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chain, chain_payload in chains.items():
        for sensitivity in ("identity", "entity_exclusion", "threshold_outcome_only", "persistence"):
            for slice_name, slice_payload in chain_payload["sensitivities"][sensitivity].items():
                for track, track_payload in slice_payload["tracks"].items():
                    for model, metrics in track_payload["models"].items():
                        if track in {"A", "B1"}:
                            ci = metrics.get("average_precision_ci95") or (None, None)
                            primary_name = "average_precision"
                            primary_value = metrics.get("average_precision")
                            n = metrics.get("n")
                            positives = metrics.get("positives")
                            groups = metrics.get("uncertainty", {}).get("clusters")
                        else:
                            at3 = metrics.get("at_k", {}).get("k_3", {})
                            ci = at3.get("macro_recall_ci95") or (None, None)
                            primary_name = "macro_recall_at_3"
                            primary_value = at3.get("macro_recall")
                            n = metrics.get("n_candidate_lanes")
                            positives = metrics.get("positive_lanes")
                            groups = metrics.get("n_entry_groups")
                        rows.append(
                            {
                                "chain": chain,
                                "sensitivity": sensitivity,
                                "slice": slice_name,
                                "track": track,
                                "model": model,
                                "status": metrics.get("status"),
                                "n": n,
                                "positives": positives,
                                "clusters_or_entry_groups": groups,
                                "primary_metric": primary_name,
                                "primary_value": primary_value,
                                "ci95_low": ci[0],
                                "ci95_high": ci[1],
                                "cluster_unit": metrics.get("uncertainty", {}).get("cluster_unit"),
                            }
                        )
    return rows


def _macro_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    frame["primary_value"] = pd.to_numeric(frame["primary_value"], errors="coerce")
    grouped = frame.groupby(
        ["sensitivity", "slice", "track", "model", "primary_metric"],
        sort=True,
        dropna=False,
    )
    result: list[dict[str, Any]] = []
    for keys, group in grouped:
        valid = group.dropna(subset=["primary_value"])
        result.append(
            {
                "sensitivity": keys[0],
                "slice": keys[1],
                "track": keys[2],
                "model": keys[3],
                "primary_metric": keys[4],
                "chains_with_defined_metric": int(len(valid)),
                "unweighted_chain_macro_mean": (
                    float(valid["primary_value"].mean()) if len(valid) else None
                ),
                "per_chain": {
                    row["chain"]: row["primary_value"]
                    for row in group[["chain", "primary_value"]].to_dict(orient="records")
                },
            }
        )
    return result


def _input_records(chains: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chain, payload in chains.items():
        for name, audit in payload["input_audit"].items():
            records.append({"chain": chain, "role": name, **audit})
    return sorted(records, key=lambda row: (row["chain"], row["role"]))


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_dir = args.data_dir.resolve()
    raw_path = args.baci_zip.resolve()
    rolling_path = args.rolling_result.resolve()
    if tuple(args.chains) != CHAINS:
        raise ValueError("formal robustness output requires all six chains in canonical order")
    if data_dir != DEFAULT_DATA.resolve():
        raise ValueError("formal robustness output requires the canonical data/processed_v2 root")

    # Phase 0: verify the rolling result and freeze all 18 historical choices
    # before parsing even a fold2 candidate table.  The rolling verifier hashes
    # candidate bytes but does not parse their features or labels.  This is the
    # only source of selected C values; there is no code-level fallback table.
    selection_freeze = _freeze_choices_from_verified_rolling(rolling_path)
    frozen_choices = selection_freeze.as_dict()
    print(
        "verified and froze 18 rolling choices before candidate-label parsing: "
        f"{selection_freeze.choices_sha256}",
        flush=True,
    )
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    # Phase 1: freeze every model and identity set using fold2 only.
    frozen: dict[str, FrozenChain] = {}
    for index, chain in enumerate(CHAINS):
        print(f"[{chain}] fitting verified CPU systems on fold2 only ...", flush=True)
        frozen[chain] = _fit_frozen_chain(
            data_dir,
            chain,
            BASE_SEED + index * 1000,
            frozen_choices[chain],
        )
    freeze_digest = selection_freeze.choices_sha256
    print(f"all fold2 CPU systems frozen: {freeze_digest}", flush=True)

    # Phase 2: open main features/labels and score every candidate once.
    scored: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    chain_shells: dict[str, Any] = {}
    candidate_key_parts: list[pd.DataFrame] = []
    for chain in CHAINS:
        print(f"[{chain}] scoring complete main cohorts ...", flush=True)
        a, b, entries, target_audit = _score_main_chain(data_dir, chain, frozen[chain])
        scored[chain] = (a, b, entries)
        keys = pd.concat([a.loc[:, list(KEYS)], b.loc[:, list(KEYS)]], ignore_index=True).drop_duplicates()
        keys.insert(0, "chain", chain)
        candidate_key_parts.append(keys)
        chain_shells[chain] = {
            "input_audit": {**frozen[chain].input_audit, **target_audit},
            "frozen_choices": {
                track: {
                    "selected_C": float(frozen_choices[chain][track]),
                    "source": "verified rolling-CPU fold2 grouped-CV artifact",
                }
                for track in TRACKS
            },
        }

    # Phase 3: independently rebuild the raw annual late labels.
    candidate_keys = pd.concat(candidate_key_parts, ignore_index=True).drop_duplicates(list(RAW_KEYS))
    hs_to_chain, hs_to_stage = _chain_stage_maps(CHAINS)
    with zipfile.ZipFile(raw_path) as zf:
        iso = _country_iso_map(zf)
        raw_labels, raw_read = aggregate_candidate_late_years(
            zf,
            candidate_keys,
            iso=iso,
            hs_to_chain=hs_to_chain,
            hs_to_stage=hs_to_stage,
            chunk_size=args.chunk_size,
        )

    # Phase 4: reconcile the default and evaluate every prespecified slice.
    for chain in CHAINS:
        a, b, entries = scored[chain]
        a = _attach_raw_labels(a, chain, raw_labels)
        b = _attach_raw_labels(b, chain, raw_labels)
        reconciliation = {
            "track_a": _raw_reconciliation(a),
            "track_b_lanes": _raw_reconciliation(b),
        }
        chain_shells[chain]["raw_reconciliation"] = reconciliation
        chain_shells[chain]["sensitivities"] = _evaluate_chain_sensitivities(
            chain,
            frozen[chain],
            a,
            b,
            entries,
            bootstrap=args.bootstrap,
        )

    csv_rows = _csv_rows(chain_shells)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": "2.1-dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prespecification": {
            "chains": list(CHAINS),
            "late_years": list(LATE_YEARS),
            "thresholds_kusd": list(THRESHOLDS_KUSD),
            "default_threshold_kusd": DEFAULT_THRESHOLD_KUSD,
            "threshold_candidate_policy": (
                "outcome-only relabeling of the fixed default-100-kUSD candidate cohort; "
                "early eligibility is not rebuilt"
            ),
            "persistence": {
                "active_year_definition": "raw annual stage total > 100 kUSD",
                "positive_rule": ">=3 active years among 2018-2022",
                "minimum_active_years": PERSISTENCE_MIN_ACTIVE_YEARS,
                "default_label_changed": False,
            },
            "identity": {
                "seen_definition": "ISO appears in the corresponding fold2 candidate table",
                "track_a_reference": "candidates_<chain>_fold2.csv",
                "track_b_reference": "candidates_firsttime_<chain>_fold2.csv",
                "exporter_disjoint_slice": "identity/exporter_unseen",
                "importer_disjoint_slice": "identity/importer_unseen (A and B2 only)",
            },
            "hubs": sorted(HUBS),
            "bad_iso": sorted(BAD_ISO),
            "entity_exclusion_rule_A_B2": "exclude rows if exporter OR importer is in the set",
            "entity_exclusion_rule_B1": "exclude entries if exporter is in the set",
            "b2_slice_conditioning": (
                "after applying the slice and auxiliary label, retain exporter-stage entries "
                "with at least one positive destination in that exact slice"
            ),
            "uncertainty": {
                "A": "exporter-cluster bootstrap",
                "B1": "exporter-cluster bootstrap",
                "B2": "exporter-stage entry bootstrap",
                "draws": int(args.bootstrap),
            },
            "frozen_C": frozen_choices,
            "freeze_digest_sha256": freeze_digest,
        },
        "protocol": {
            "selection_source": "fold2 only",
            "rolling_artifact_verified_before_any_candidate_label_parse": True,
            "all_chain_choices_frozen_before_any_candidate_label_parse": True,
            "all_chain_models_frozen_before_main_open": True,
            "main_labels_used_for_model_or_hyperparameter_selection": False,
            "main_labels_used_for_feature_fitting_imputation_or_calibration": False,
            "main_model_champion_selected": False,
            "raw_persistence_or_threshold_labels_use_production_aggregation_helper": False,
            "transductive_split_used": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "cpu_helper_sha256": sha256_file(Path(cpu.__file__).resolve()),
            "command": _portable_command(sys.argv),
            "working_directory": "repository_root",
        },
        "raw_source": {
            "path": _portable_path(raw_path),
            "archive_name": raw_path.name,
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
            "read": raw_read,
        },
        "selection_artifact": {
            "path": _portable_path(rolling_path),
            "sha256": selection_freeze.artifact_sha256,
            "choices_sha256": selection_freeze.choices_sha256,
            "frozen_choices": frozen_choices,
            "verified_input_hashes": selection_freeze.verified_input_hashes,
            "verified_before_any_candidate_label_parse": True,
            "selected_C_fields_checked_before_historical_fit": True,
        },
        "inputs": _input_records(chain_shells),
        "chains": chain_shells,
        "macro_summary": _macro_summary(csv_rows),
    }
    return _strict_jsonable(result), _strict_jsonable(csv_rows)


def _validate_prespec_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected robustness schema_version")
    prespec = payload.get("prespecification", {})
    if prespec.get("chains") != list(CHAINS):
        raise ValueError("robustness output is not the complete six-chain suite")
    if prespec.get("thresholds_kusd") != list(THRESHOLDS_KUSD):
        raise ValueError("threshold prespecification changed")
    if prespec.get("hubs") != sorted(HUBS) or prespec.get("bad_iso") != sorted(BAD_ISO):
        raise ValueError("entity-exclusion prespecification changed")
    frozen_tuple = _choice_tuple(
        prespec.get("frozen_C"), context="robustness prespecification/frozen_C"
    )
    frozen_choices = RollingChoiceFreeze(
        path=Path("."),
        artifact_sha256="",
        choices=frozen_tuple,
        verified_input_hashes=24,
    )
    if prespec.get("freeze_digest_sha256") != frozen_choices.choices_sha256:
        raise ValueError("frozen fold2 CPU choice digest changed")
    persistence = prespec.get("persistence", {})
    if persistence.get("minimum_active_years") != PERSISTENCE_MIN_ACTIVE_YEARS:
        raise ValueError("persistence prespecification changed")
    protocol = payload.get("protocol", {})
    required_true = (
        "rolling_artifact_verified_before_any_candidate_label_parse",
        "all_chain_choices_frozen_before_any_candidate_label_parse",
        "all_chain_models_frozen_before_main_open",
    )
    required_false = (
        "main_labels_used_for_model_or_hyperparameter_selection",
        "main_labels_used_for_feature_fitting_imputation_or_calibration",
        "main_model_champion_selected",
        "raw_persistence_or_threshold_labels_use_production_aggregation_helper",
        "transductive_split_used",
    )
    if any(protocol.get(field) is not True for field in required_true):
        raise ValueError("robustness freeze attestation is missing")
    if any(protocol.get(field) is not False for field in required_false):
        raise ValueError("a robustness anti-leakage attestation is not false")
    if set(payload.get("chains", {})) != set(CHAINS):
        raise ValueError("chain result set is incomplete")
    for chain, chain_payload in payload["chains"].items():
        observed_choices = chain_payload.get("frozen_choices", {})
        if set(observed_choices) != set(TRACKS):
            raise ValueError(f"{chain}: incomplete frozen choice records")
        for track in TRACKS:
            record = observed_choices[track]
            if not isinstance(record, Mapping):
                raise ValueError(f"{chain}/{track}: malformed frozen choice record")
            if record.get("selected_C") != frozen_choices.selected_c(chain, track):
                raise ValueError(f"{chain}/{track}: chain choice disagrees with frozen matrix")
            if record.get("source") != "verified rolling-CPU fold2 grouped-CV artifact":
                raise ValueError(f"{chain}/{track}: frozen choice source changed")
        reconciliation = chain_payload.get("raw_reconciliation", {})
        if reconciliation.get("track_a", {}).get("pass") is not True:
            raise ValueError(f"{chain}: Track A raw reconciliation is not passing")
        if reconciliation.get("track_b_lanes", {}).get("pass") is not True:
            raise ValueError(f"{chain}: Track B raw reconciliation is not passing")
        expected_slices = {
            "identity": {"exporter_seen", "exporter_unseen", "importer_seen", "importer_unseen"},
            "entity_exclusion": {"exclude_hubs", "exclude_bad_iso", "exclude_hubs_and_bad_iso"},
            "threshold_outcome_only": {"threshold_50_kusd", "threshold_100_kusd", "threshold_250_kusd"},
            "persistence": {"active_at_least_3_of_5_years_above_100_kusd"},
        }
        sensitivities = chain_payload.get("sensitivities", {})
        for sensitivity, names in expected_slices.items():
            if set(sensitivities.get(sensitivity, {})) != names:
                raise ValueError(f"{chain}: incomplete {sensitivity} slices")

    selection = payload.get("selection_artifact", {})
    if not isinstance(selection, Mapping):
        raise ValueError("robustness selection-artifact record is missing")
    if selection.get("verified_input_hashes") != 24:
        raise ValueError("robustness selection artifact must record 24 verified inputs")
    if selection.get("verified_before_any_candidate_label_parse") is not True:
        raise ValueError("robustness selection artifact was not verified before label parsing")
    if selection.get("selected_C_fields_checked_before_historical_fit") is not True:
        raise ValueError("robustness choices were not checked before historical fitting")
    if selection.get("frozen_choices") != frozen_choices.as_dict():
        raise ValueError("selection-artifact choices disagree with prespecification")
    if selection.get("choices_sha256") != frozen_choices.choices_sha256:
        raise ValueError("selection-artifact choice digest changed")
    artifact_hash = selection.get("sha256")
    if (
        not isinstance(artifact_hash, str)
        or len(artifact_hash) != 64
        or any(char not in "0123456789abcdef" for char in artifact_hash.lower())
    ):
        raise ValueError("selection-artifact sha256 is invalid")


def verify_existing_output(
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
) -> None:
    json_path = json_path.resolve()
    csv_path = csv_path.resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _validate_prespec_payload(payload)

    runtime = payload.get("runtime", {})
    if runtime.get("script_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError(f"{json_path}: stale generator script hash")
    if runtime.get("cpu_helper_sha256") != sha256_file(Path(cpu.__file__).resolve()):
        raise ValueError(f"{json_path}: stale CPU helper hash")

    checked = 0
    for record in payload.get("inputs", []):
        path = ROOT / str(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"{json_path}: stale or missing input {path}")
        checked += 1
    if checked != 24:
        raise ValueError(f"{json_path}: expected 24 candidate input hashes, found {checked}")

    raw = payload.get("raw_source", {})
    raw_path = Path(str(raw.get("path", "")))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    if not raw_path.is_file() or raw.get("sha256") != sha256_file(raw_path):
        raise ValueError(f"{json_path}: stale or missing raw BACI archive")

    selection = payload.get("selection_artifact", {})
    selection_path = Path(str(selection.get("path", "")))
    if not selection_path.is_absolute():
        selection_path = ROOT / selection_path
    selection_freeze = _freeze_choices_from_verified_rolling(selection_path)
    if selection.get("sha256") != selection_freeze.artifact_sha256:
        raise ValueError(f"{json_path}: stale rolling selection artifact")
    if selection.get("frozen_choices") != selection_freeze.as_dict():
        raise ValueError(f"{json_path}: frozen choices differ from rolling selection artifact")
    if selection.get("choices_sha256") != selection_freeze.choices_sha256:
        raise ValueError(f"{json_path}: rolling selection choice digest changed")
    if payload["prespecification"].get("frozen_C") != selection_freeze.as_dict():
        raise ValueError(f"{json_path}: prespecified choices differ from rolling artifact")
    rolling_raw = selection_path.read_bytes()
    if hashlib.sha256(rolling_raw).hexdigest() != selection_freeze.artifact_sha256:
        raise ValueError(f"{json_path}: rolling artifact changed after verification")
    rolling_payload = json.loads(rolling_raw.decode("utf-8"))

    outputs = payload.get("outputs", {})
    if outputs.get("csv_path") != _portable_path(csv_path):
        raise ValueError(f"{json_path}: CSV path attestation changed")
    if not csv_path.is_file() or outputs.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(f"{json_path}: stale or missing CSV output")
    rows = pd.read_csv(csv_path)
    if int(outputs.get("csv_rows", -1)) != len(rows):
        raise ValueError(f"{json_path}: CSV row count changed")

    expected_rows = pd.DataFrame(_csv_rows(payload["chains"]))
    row_key = ["chain", "sensitivity", "slice", "track", "model"]
    rows_for_check = rows.sort_values(row_key, kind="mergesort").reset_index(drop=True)
    expected_for_check = expected_rows.sort_values(row_key, kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            rows_for_check,
            expected_for_check,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{json_path}: JSON metrics and CSV rows disagree") from exc

    # The default-threshold point metrics are an exact internal control: they
    # must reproduce the already-verified complete-cohort rolling result.
    for chain in CHAINS:
        robustness_tracks = payload["chains"][chain]["sensitivities"][
            "threshold_outcome_only"
        ]["threshold_100_kusd"]["tracks"]
        reference_tracks = rolling_payload["chains"][chain]
        comparisons = (
            (
                "A",
                "track_a_destination_extension",
                "average_precision",
                A_MODELS,
            ),
            (
                "B1",
                "track_b1_processed_export_stage_entry",
                "average_precision",
                B1_MODELS,
            ),
        )
        for short_track, reference_track, metric, models in comparisons:
            for model in models:
                observed = robustness_tracks[short_track]["models"][model][metric]
                expected = reference_tracks[reference_track]["models"][model]["metrics"][metric]
                if observed != expected:
                    raise ValueError(
                        f"{json_path}: default-threshold control mismatch for "
                        f"{chain}/{short_track}/{model}"
                    )
        for model in B2_MODELS:
            observed = robustness_tracks["B2"]["models"][model]["at_k"]["k_3"][
                "macro_recall"
            ]
            expected = reference_tracks["track_b2_conditional_destination_ranking"]["models"][
                model
            ]["metrics"]["at_k"]["k_3"]["macro_recall"]
            if observed != expected:
                raise ValueError(
                    f"{json_path}: default-threshold control mismatch for {chain}/B2/{model}"
                )
    print(
        f"verified robustness suite: {len(CHAINS)} chains, {checked} candidate inputs, "
        f"{len(rows)} metric rows, raw archive and generator hashes current"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baci-zip", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--rolling-result", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chains", nargs="+", default=list(CHAINS))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--verify-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.verify_output:
        verify_existing_output(args.json_out, args.csv_out)
        return 0
    result, rows = run(args)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv_out, index=False)
    result["outputs"] = {
        "csv_path": _portable_path(args.csv_out),
        "csv_sha256": sha256_file(args.csv_out),
        "csv_rows": len(rows),
    }
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
