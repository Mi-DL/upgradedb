#!/usr/bin/env python3
"""Official, deterministic scorer for UPGRADE-BENCH v2.

Only NumPy and pandas are required. Main-snapshot external submissions need a
schema-checked self-attestation unless the caller explicitly marks the run
diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from loader import (  # noqa: E402
    FORBIDDEN_SCORE_COLUMNS,
    VERSION,
    BenchmarkDataError,
    feature_columns,
    keys_for_track,
    load,
    normalize_snapshot,
    normalize_track,
)


PROTOCOL = "historical-fold-selection-frozen-main-evaluation"
ATTESTATION_SCHEMA_VERSION = 2
ATTESTATION_TYPE = "schema_checked_self_attestation"
DEFAULT_BUDGETS = {"A": (100, 500, 1000), "B1": (25, 50, 100)}
OFFICIAL_B2_K = (1, 3, 5)
OFFICIAL_EXPORTER_K = (5, 10)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ScoreError(ValueError):
    """A score vector, score file, or scoring request is invalid."""


class ProtocolAttestationError(ValueError):
    """A required schema-checked self-attestation is absent or invalid."""


def sha256_file(path: str | Path) -> str:
    """Return the raw-byte SHA-256 of one artifact."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProtocolAttestationError(f"cannot hash required artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _finite_score(score: Iterable[float], expected_length: int) -> np.ndarray:
    try:
        values = np.asarray(list(score) if not isinstance(score, np.ndarray) else score, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ScoreError(f"scores must be numeric: {exc}") from exc
    if values.ndim != 1 or len(values) != expected_length:
        raise ScoreError(f"expected {expected_length:,} one-dimensional scores; got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ScoreError(f"scores contain {int((~np.isfinite(values)).sum()):,} NaN/Inf values")
    return values


def _tie_order(frame: pd.DataFrame, score: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    # np.lexsort uses the final key as primary.  Keys are canonical, not input-row order.
    lexical = tuple(frame[key].astype(str).to_numpy() for key in reversed(tuple(keys)))
    return np.lexsort((*lexical, -score))


def average_precision(y: Iterable[int], score: Iterable[float]) -> float | None:
    """Tie-block average precision, equivalent to threshold-based non-interpolated AP."""
    y = np.asarray(y, dtype=np.int8)
    score = _finite_score(np.asarray(score), len(y))
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="stable")
    ys, scores = y[order], score[order]
    cumulative = np.cumsum(ys)
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    group_positive = np.diff(np.r_[0, cumulative[ends]])
    precision = cumulative[ends] / (ends + 1)
    return float(np.sum(precision * group_positive) / positives)


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives, negatives = int(y.sum()), int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    statistic = ranks[y == 1].sum() - positives * (positives + 1) / 2.0
    return float(statistic / (positives * negatives))


def within_size_bin_auc(
    y: Iterable[int], score: Iterable[float], size: Iterable[float], bins: int = 10
) -> dict[str, float | int | None]:
    """Row-weighted AUC within size quantile ranges; requires >=3/3 classes per bin."""
    y = np.asarray(y, dtype=np.int8)
    score = _finite_score(np.asarray(score), len(y))
    size = _finite_score(np.asarray(size), len(y))
    if bins < 1:
        raise ScoreError("size-bin count must be positive")
    edges = np.unique(np.quantile(size, np.linspace(0.0, 1.0, bins + 1)))
    assignments = np.zeros(len(size), dtype=int) if len(edges) == 1 else np.digitize(size, edges[1:-1])
    aucs: list[float] = []
    weights: list[int] = []
    for bin_id in np.unique(assignments):
        mask = assignments == bin_id
        if int(y[mask].sum()) < 3 or int((y[mask] == 0).sum()) < 3:
            continue
        auc = _binary_auc(y[mask], score[mask])
        if auc is not None:
            aucs.append(auc)
            weights.append(int(mask.sum()))
    value = float(np.average(aucs, weights=weights)) if aucs else None
    return {
        "auc": value,
        "requested_bins": int(bins),
        "formed_bins": int(len(np.unique(assignments))),
        "valid_bins": int(len(aucs)),
        "rows_in_valid_bins": int(sum(weights)),
    }


def _budget_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    label: str,
    value: str,
    keys: Sequence[str],
    budgets: Sequence[int],
) -> dict[str, dict[str, float | int | None]]:
    y = frame[label].to_numpy(dtype=np.int8)
    late_value = frame[value].to_numpy(dtype=float)
    order = _tie_order(frame, score, keys)
    total_positive, total_value = int(y.sum()), float(late_value.sum())
    result: dict[str, dict[str, float | int | None]] = {}
    for requested in budgets:
        if int(requested) < 1:
            raise ScoreError(f"budgets must be positive integers; got {requested!r}")
        effective = min(int(requested), len(frame))
        chosen = order[:effective]
        found = int(y[chosen].sum())
        captured = float(late_value[chosen].sum())
        result[f"k_{int(requested)}"] = {
            "requested_budget": int(requested),
            "effective_budget": effective,
            "positives_found": found,
            "precision": float(found / effective) if effective else None,
            "recall": float(found / total_positive) if total_positive else None,
            "value_captured": float(captured / total_value) if total_value > 0 else None,
        }
    return result


def _exporter_shortlists(
    frame: pd.DataFrame,
    score: np.ndarray,
    ks: Sequence[int] = OFFICIAL_EXPORTER_K,
) -> dict[str, dict[str, float | int | None]]:
    """CPU-artifact-compatible Track-A shortlist metrics, grouped by exporter.

    Macro precision averages over every exporter. Macro recall and macro value
    capture average only over exporters with at least one positive/positive
    value. Micro metrics pool hits/value over all exporters. Equal-score lanes
    are ordered by ``(stage, j_iso)`` inside each exporter.
    """
    work = frame.loc[:, ["i_iso", "stage", "j_iso", "y", "lateval"]].copy()
    work["_score"] = _finite_score(score, len(work))
    if any(int(k) < 1 for k in ks):
        raise ScoreError("per-exporter cutoffs must be positive integers")
    result: dict[str, dict[str, float | int | None]] = {}
    for requested in ks:
        k = int(requested)
        recalls: list[float] = []
        precisions: list[float] = []
        value_shares: list[float] = []
        total_hits = 0
        total_positives = 0
        selected_value = 0.0
        total_value = 0.0
        exporter_count = 0
        for _, group in work.groupby("i_iso", sort=True, observed=True):
            exporter_count += 1
            order = _tie_order(
                group,
                group["_score"].to_numpy(dtype=float),
                ("stage", "j_iso"),
            )
            chosen = order[: min(k, len(group))]
            y = group["y"].to_numpy(dtype=np.int8)
            late_value = group["lateval"].to_numpy(dtype=float)
            hits = int(y[chosen].sum())
            positives = int(y.sum())
            captured = float(late_value[chosen].sum())
            available = float(late_value.sum())
            precisions.append(float(hits / len(chosen)))
            if positives:
                recalls.append(float(hits / positives))
            if available > 0:
                value_shares.append(float(captured / available))
            total_hits += hits
            total_positives += positives
            selected_value += captured
            total_value += available
        result[f"k_{k}_per_exporter"] = {
            "exporters": exporter_count,
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


def evaluate_track_a(
    frame: pd.DataFrame, score: Iterable[float], budgets: Sequence[int] = DEFAULT_BUDGETS["A"]
) -> dict[str, object]:
    values = _finite_score(score, len(frame))
    y = frame["y"].to_numpy(dtype=np.int8)
    return {
        "task": "Track A: destination extension",
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "average_precision": average_precision(y, values),
        "budgets": _budget_metrics(
            frame, values, label="y", value="lateval", keys=("i_iso", "stage", "j_iso"), budgets=budgets
        ),
        "per_exporter_shortlists": _exporter_shortlists(frame, values),
        "within_size_bin_diagnostic": within_size_bin_auc(y, values, frame["size"], bins=10),
    }


def evaluate_track_b1(
    frame: pd.DataFrame, score: Iterable[float], budgets: Sequence[int] = DEFAULT_BUDGETS["B1"]
) -> dict[str, object]:
    values = _finite_score(score, len(frame))
    z = frame["z"].to_numpy(dtype=np.int8)
    return {
        "task": "Track B1: eligible-market processed-export stage entry",
        "rows": int(len(frame)),
        "positive_entries": int(z.sum()),
        "average_precision": average_precision(z, values),
        "budgets": _budget_metrics(
            frame,
            values,
            label="z",
            value="entry_lateval",
            keys=("i_iso", "stage"),
            budgets=budgets,
        ),
    }


def evaluate_track_b2(
    frame: pd.DataFrame, score: Iterable[float], ks: Sequence[int] = OFFICIAL_B2_K
) -> dict[str, object]:
    values = _finite_score(score, len(frame))
    work = frame.copy()
    work["_score"] = values
    if any(int(k) < 1 for k in ks):
        raise ScoreError("Track-B2 cutoffs must be positive integers")
    per_k: dict[str, dict[str, float | int]] = {}
    recalls = {int(k): [] for k in ks}
    value_capture = {int(k): [] for k in ks}
    groups = list(work.groupby(["i_iso", "stage"], sort=True, observed=True))
    for _, group in groups:
        y = group["y"].to_numpy(dtype=np.int8)
        late_value = group["lateval"].to_numpy(dtype=float)
        if int(y.sum()) < 1 or float(late_value.sum()) <= 0:
            raise ScoreError("Track-B2 groups must each contain a positive destination and positive value")
        order = _tie_order(group, group["_score"].to_numpy(dtype=float), ("j_iso",))
        for cutoff in recalls:
            chosen = order[: min(cutoff, len(group))]
            recalls[cutoff].append(float(y[chosen].sum() / y.sum()))
            value_capture[cutoff].append(float(late_value[chosen].sum() / late_value.sum()))
    for cutoff in recalls:
        per_k[f"k_{cutoff}"] = {
            "k": cutoff,
            "entries": len(groups),
            "macro_recall": float(np.mean(recalls[cutoff])),
            "macro_value_captured": float(np.mean(value_capture[cutoff])),
        }
    return {
        "task": "Track B2: conditional destination formation",
        "rows": int(len(frame)),
        "entries": int(len(groups)),
        "positive_destinations": int(frame["y"].sum()),
        "at_k": per_k,
    }


def evaluate(
    track: str,
    frame: pd.DataFrame,
    score: Iterable[float],
    *,
    budgets: Sequence[int] | None = None,
    ks: Sequence[int] = OFFICIAL_B2_K,
) -> dict[str, object]:
    track = normalize_track(track)
    if track == "A":
        return evaluate_track_a(frame, score, budgets or DEFAULT_BUDGETS["A"])
    if track == "B1":
        return evaluate_track_b1(frame, score, budgets or DEFAULT_BUDGETS["B1"])
    if budgets is not None:
        raise ScoreError("--budgets is not defined for Track B2; official cutoffs are @1/3/5")
    return evaluate_track_b2(frame, score, ks)


def builtin_scores(frame: pd.DataFrame, track: str, column: str) -> np.ndarray:
    track = normalize_track(track)
    if column in FORBIDDEN_SCORE_COLUMNS:
        raise ScoreError(f"{column!r} is an outcome/leakage column and is forbidden as a score")
    allowed = feature_columns(track)
    if column not in allowed:
        raise ScoreError(f"built-in score for Track {track} must be one of {allowed}; got {column!r}")
    return _finite_score(pd.to_numeric(frame[column], errors="coerce").to_numpy(), len(frame))


def load_external_scores(
    path: str | Path,
    candidates: pd.DataFrame,
    track: str,
    *,
    score_column: str = "score",
) -> np.ndarray:
    """Load and exact-key align a submission; missing *and extra* keys are errors."""
    track = normalize_track(track)
    keys = list(keys_for_track(track))
    if score_column in FORBIDDEN_SCORE_COLUMNS:
        raise ScoreError(f"{score_column!r} is an outcome/leakage column and is forbidden as a score")
    if score_column in keys:
        raise ScoreError("score column cannot also be a candidate key")
    path = Path(path)
    try:
        scores = pd.read_csv(path, dtype={key: str for key in keys}, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise ScoreError(f"cannot read score CSV {path}: {exc}") from exc
    leaked = sorted(FORBIDDEN_SCORE_COLUMNS.intersection(scores.columns))
    if leaked:
        raise ScoreError(f"submission CSV contains forbidden outcome columns: {leaked}")
    expected_columns = set(keys) | {score_column}
    if set(scores.columns) != expected_columns:
        missing = sorted(expected_columns - set(scores.columns))
        unexpected = sorted(set(scores.columns) - expected_columns)
        raise ScoreError(f"submission schema must be exactly {keys + [score_column]}; missing={missing}, unexpected={unexpected}")
    if scores[list(keys)].isna().any().any():
        raise ScoreError("submission contains null candidate keys")
    duplicates = int(scores.duplicated(keys).sum())
    if duplicates:
        raise ScoreError(f"submission contains {duplicates:,} duplicate key rows")

    expected_index = pd.MultiIndex.from_frame(candidates[keys])
    score_index = pd.MultiIndex.from_frame(scores[keys])
    missing_keys = expected_index.difference(score_index)
    extra_keys = score_index.difference(expected_index)
    if len(missing_keys) or len(extra_keys):
        raise ScoreError(
            "submission key set must exactly equal the evaluated candidate key set; "
            f"missing={len(missing_keys):,}, extra={len(extra_keys):,}"
        )
    indexed = scores.set_index(keys)[score_column]
    try:
        aligned = indexed.reindex(expected_index)
    except ValueError as exc:
        raise ScoreError(f"cannot align submission keys: {exc}") from exc
    return _finite_score(pd.to_numeric(aligned, errors="coerce").to_numpy(), len(candidates))


def validate_attestation(
    path: str | Path,
    *,
    benchmark_data_path: str | Path | None = None,
    score_csv_path: str | Path | None = None,
    selection_config_path: str | Path | None = None,
) -> dict[str, object]:
    """Schema-check a self-attestation and, when supplied, bind exact artifacts.

    This validates document shape and raw-byte identities. It cannot independently
    verify the submitter's statements about past model-development behavior.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolAttestationError(f"cannot parse attestation {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ProtocolAttestationError("self-attestation root must be a JSON object")
    required_fields = {
        "schema_version",
        "attestation_type",
        "benchmark_version",
        "protocol",
        "submission_name",
        "run_id",
        "exact_seed_list",
        "selected_on_snapshot",
        "choices_frozen_before_main_evaluation",
        "main_labels_used_for_selection",
        "main_labels_used_for_feature_fitting",
        "main_labels_used_for_imputation_or_calibration",
        "benchmark_data_sha256",
        "score_csv_sha256",
        "selection_config_sha256",
        "attested_by",
    }
    missing = sorted(required_fields - set(document))
    unexpected = sorted(set(document) - required_fields)
    mismatches: list[str] = []
    if missing or unexpected:
        mismatches.append(f"schema fields missing={missing}, unexpected={unexpected}")
    expected = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "attestation_type": ATTESTATION_TYPE,
        "benchmark_version": VERSION,
        "protocol": PROTOCOL,
        "selected_on_snapshot": "fold2",
        "choices_frozen_before_main_evaluation": True,
        "main_labels_used_for_selection": False,
        "main_labels_used_for_feature_fitting": False,
        "main_labels_used_for_imputation_or_calibration": False,
    }
    mismatches.extend(
        f"{key}={document.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if document.get(key) != value
    )
    for key in ("submission_name", "attested_by"):
        if not isinstance(document.get(key), str) or not str(document[key]).strip():
            mismatches.append(f"{key} must be a non-empty string")
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        mismatches.append(
            "run_id must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
        )
    seeds = document.get("exact_seed_list")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        mismatches.append("exact_seed_list must be a non-empty ordered list of unique integers")
    for key in (
        "benchmark_data_sha256",
        "score_csv_sha256",
        "selection_config_sha256",
    ):
        digest = document.get(key)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            mismatches.append(f"{key} must be a lowercase 64-character SHA-256")

    paths = {
        "benchmark_data_sha256": benchmark_data_path,
        "score_csv_sha256": score_csv_path,
        "selection_config_sha256": selection_config_path,
    }
    supplied_paths = [value is not None for value in paths.values()]
    if any(supplied_paths) and not all(supplied_paths):
        mismatches.append(
            "artifact binding requires benchmark data, score CSV, and selection config paths together"
        )
    if all(supplied_paths):
        for key, artifact_path in paths.items():
            actual = sha256_file(Path(artifact_path))
            if document.get(key) != actual:
                mismatches.append(
                    f"{key} does not match raw bytes of {Path(artifact_path)}"
                )
    if mismatches:
        raise ProtocolAttestationError(
            "invalid schema-checked self-attestation: " + "; ".join(mismatches)
        )
    return document


def _protocol_record(
    *,
    snapshot: str,
    external: bool,
    attestation: str | Path | None,
    diagnostic_override: str | None,
    benchmark_data_path: str | Path | None = None,
    score_csv_path: str | Path | None = None,
    selection_config_path: str | Path | None = None,
) -> dict[str, object]:
    snapshot = normalize_snapshot(snapshot)
    if diagnostic_override is not None and not str(diagnostic_override).strip():
        raise ProtocolAttestationError("diagnostic override requires a non-empty reason")
    if attestation is not None and diagnostic_override is not None:
        raise ProtocolAttestationError("use either an attestation or a diagnostic override, not both")
    if not external:
        if attestation is not None or diagnostic_override is not None or selection_config_path is not None:
            raise ProtocolAttestationError(
                "attestation, selection config, and override apply only to external score submissions"
            )
        return {"protocol": PROTOCOL, "official": True, "status": "shipped ex-ante reference score"}
    if snapshot == "fold2":
        if selection_config_path is not None:
            raise ProtocolAttestationError(
                "--selection-config is only consumed when validating a main self-attestation"
            )
        if diagnostic_override is not None:
            return {"protocol": PROTOCOL, "official": False, "status": "diagnostic override", "reason": str(diagnostic_override)}
        if attestation is not None:
            raise ProtocolAttestationError(
                "--attestation is a main-evaluation self-attestation; do not attach it to fold2 scoring"
            )
        return {"protocol": PROTOCOL, "official": True, "status": "historical selection-fold scoring"}
    if attestation is not None:
        if selection_config_path is None:
            raise ProtocolAttestationError(
                "main self-attestation requires --selection-config so its raw-byte SHA-256 can be checked"
            )
        document = validate_attestation(
            attestation,
            benchmark_data_path=benchmark_data_path,
            score_csv_path=score_csv_path,
            selection_config_path=selection_config_path,
        )
        return {
            "protocol": PROTOCOL,
            "official": True,
            "status": "accepted schema-checked self-attestation",
            "attestation_type": ATTESTATION_TYPE,
            "verification_scope": (
                "schema and raw-byte artifact identities checked; historical protocol "
                "statements are self-reported and not independently verified"
            ),
            "submission_name": document["submission_name"],
            "run_id": document["run_id"],
            "exact_seed_list": document["exact_seed_list"],
            "attested_by": document["attested_by"],
            "benchmark_data_sha256": document["benchmark_data_sha256"],
            "score_csv_sha256": document["score_csv_sha256"],
            "selection_config_sha256": document["selection_config_sha256"],
            "attestation_sha256": sha256_file(attestation),
        }
    if diagnostic_override is not None:
        if selection_config_path is not None:
            raise ProtocolAttestationError(
                "--selection-config cannot be combined with --diagnostic-override"
            )
        return {
            "protocol": PROTOCOL,
            "official": False,
            "status": "diagnostic override; self-attestation not accepted",
            "reason": str(diagnostic_override),
        }
    raise ProtocolAttestationError(
        "main-snapshot external scores require a valid schema-checked --attestation and "
        "--selection-config; use --diagnostic-override REASON "
        "only for an explicitly non-official diagnostic"
    )


def score_request(
    *,
    track: str,
    chain: str,
    snapshot: str = "main",
    data_root: str | Path | None = None,
    column: str | None = None,
    scores_path: str | Path | None = None,
    score_column: str = "score",
    attestation: str | Path | None = None,
    selection_config_path: str | Path | None = None,
    diagnostic_override: str | None = None,
    budgets: Sequence[int] | None = None,
) -> dict[str, object]:
    track = normalize_track(track)
    snapshot = normalize_snapshot(snapshot)
    if (column is None) == (scores_path is None):
        raise ScoreError("choose exactly one of a built-in --column or external --scores")
    candidates = load(track, chain, snapshot, data_root=data_root, validate=True)
    external = scores_path is not None
    protocol = _protocol_record(
        snapshot=snapshot,
        external=external,
        attestation=attestation,
        diagnostic_override=diagnostic_override,
        benchmark_data_path=candidates.attrs.get("benchmark_source"),
        score_csv_path=scores_path,
        selection_config_path=selection_config_path,
    )
    if external:
        score = load_external_scores(scores_path, candidates, track, score_column=score_column)
        method = {"type": "external_exact_key_csv", "score_column": score_column, "path": str(Path(scores_path))}
    else:
        score = builtin_scores(candidates, track, str(column))
        method = {"type": "shipped_ex_ante_column", "column": str(column)}
    return {
        "benchmark": "UPGRADE-BENCH",
        "benchmark_version": VERSION,
        "track": track,
        "chain": chain,
        "snapshot": snapshot,
        "method": method,
        "protocol_attestation": protocol,
        "tie_break": {
            "primary": "score descending",
            "secondary": {
                "A": ["i_iso", "stage", "j_iso"],
                "B1": ["i_iso", "stage"],
                "B2": ["j_iso within each (i_iso, stage) entry"],
            }[track],
        },
        "metrics": evaluate(track, candidates, score, budgets=budgets),
    }


def _parse_positive_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique positive comma-separated integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=("A", "B1", "B2"))
    parser.add_argument("--chain", required=True)
    parser.add_argument("--snapshot", default="main", choices=("main", "fold2"))
    parser.add_argument("--data-root", type=Path)
    scores = parser.add_mutually_exclusive_group(required=True)
    scores.add_argument("--column", help="shipped ex-ante score column")
    scores.add_argument("--scores", type=Path, help="exact-key external score CSV")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--budgets", type=_parse_positive_ints, help="Track A/B1 comma-separated shortlist budgets")
    parser.add_argument(
        "--attestation",
        type=Path,
        help="schema-checked self-attestation required for official main external scoring",
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        help="frozen selection/config artifact whose SHA-256 is bound by a main self-attestation",
    )
    parser.add_argument(
        "--diagnostic-override",
        metavar="REASON",
        help="bypass main attestation and mark the result explicitly non-official",
    )
    parser.add_argument("--output", type=Path, help="write result JSON (stdout is always printed)")
    args = parser.parse_args()
    try:
        result = score_request(
            track=args.track,
            chain=args.chain,
            snapshot=args.snapshot,
            data_root=args.data_root,
            column=args.column,
            scores_path=args.scores,
            score_column=args.score_column,
            attestation=args.attestation,
            selection_config_path=args.selection_config,
            diagnostic_override=args.diagnostic_override,
            budgets=args.budgets,
        )
    except (BenchmarkDataError, ProtocolAttestationError, ScoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(payload, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
