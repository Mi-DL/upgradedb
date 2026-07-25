"""Strict rolling GPU runner for UPGRADE-BENCH v2.

This is intentionally separate from ``hp_tune.py`` (a same-window diagnostic)
and from the frozen v1 score generators.  The public formal protocol has three
process-level phases:

1. ``select-chain``: use only fold2 labels and the fold2 early graph, sharing
   each trained score grid across A, B1, and B2 for one chain/family;
2. ``freeze``: hash-lock *all* requested chain/track/family selections;
3. ``evaluate-chain``: verify the formal run configuration and full manifest,
   refit label-free representations on the main early graph, score the complete
   A/B cohort for one chain/family, and only then read main labels.

Single-task selection/evaluation commands are deliberately not exposed: their
artifacts cannot satisfy the shared-chain freeze contract.

Raw scores are retained one column per seed.  They are never averaged across
independently trained KGE/NBFNet runs because their locations/scales need not be
comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from v2_gpu_protocol import (  # noqa: E402 - deliberately lightweight
    FAMILIES,
    HISTORY_FOLD,
    PROTOCOL,
    SELECTION_SCHEMA,
    TARGET_FOLD,
    TRACKS,
    ProtocolError,
    build_freeze_manifest,
    expected_combinations,
    selection_filename,
    selection_key,
    sha256_file,
    verify_freeze_manifest,
    write_json_atomic,
)


DEFAULT_CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
DEFAULT_KGE_MODELS = ("TransE", "RotatE", "DistMult", "ComplEx", "RGCN", "CompGCN")
FORMAL_EXECUTION_STATUS = "FORMAL_RUN_AUTHORIZED"
FORMAL_BOOTSTRAP_ITERS = 500
FORMAL_BOOTSTRAP_SEED = 20260712
KEYS = ("i_iso", "j_iso", "stage")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _csv(value: str, cast=str) -> list:
    try:
        result = [cast(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("comma-separated value must not be empty")
    return result


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _resolve_run_lock(args, *, phase: str | None = None) -> tuple[str, Path, str]:
    config_path = _resolve(args.run_config)
    if not config_path.is_file():
        raise ProtocolError(f"run config is missing: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"run config is invalid JSON: {config_path}") from exc
    configured = config.get("run_id")
    if not isinstance(configured, str) or not configured.strip():
        raise ProtocolError("run config must contain a non-empty run_id")
    execution_status = config.get("execution_status")
    if execution_status != FORMAL_EXECUTION_STATUS:
        raise ProtocolError(
            "run config is not authorized for formal execution: "
            f"execution_status={execution_status!r}, expected {FORMAL_EXECUTION_STATUS!r}"
        )
    if args.run_id is not None and args.run_id != configured:
        raise ProtocolError(f"--run-id {args.run_id!r} does not match config run_id {configured!r}")
    if phase is not None:
        _validate_formal_execution_spec(args, config, phase=phase)
    return configured, config_path, sha256_file(config_path)


def _stable_frame_hash(frame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, list(KEYS)].itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_triple_hash(triples) -> str:
    digest = hashlib.sha256()
    for row in triples:
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    names = ("v2_gpu_rolling.py", "v2_gpu_protocol.py", "benchmark.py", "temporal_backtest.py")
    return {f"src/{name}": sha256_file(SRC / name) for name in names}


def _cached_score(
    *,
    scorer,
    model: str,
    hyperparameters: Mapping[str, object],
    seed: int,
    cache_root: Path,
    cache_context: Mapping[str, object],
    expected_rows: int,
):
    """Train/score once per hash-locked graph+inputs+config+seed tuple."""
    import numpy as np

    payload = {
        **dict(cache_context),
        "model": model,
        "hyperparameters": dict(hyperparameters),
        "seed": int(seed),
        "expected_rows": int(expected_rows),
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    score_path = cache_root / f"{key}.npy"
    metadata_path = cache_root / f"{key}.json"
    if score_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("key") != key or metadata.get("payload") != payload:
            raise ProtocolError(f"score-cache metadata mismatch: {metadata_path}")
        scores = np.load(score_path, allow_pickle=False)
        if scores.shape != (expected_rows,) or not np.isfinite(scores).all():
            raise ProtocolError(f"invalid cached score array: {score_path}")
        return scores.astype(np.float64, copy=False), {"key": key, "hit": True}
    scores = np.asarray(scorer(model, hyperparameters, int(seed)), dtype=np.float64)
    if scores.shape != (expected_rows,) or not np.isfinite(scores).all():
        raise ProtocolError(f"invalid fresh score array for cache key {key}")
    temporary = score_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, scores, allow_pickle=False)
    temporary.replace(score_path)
    write_json_atomic(
        metadata_path,
        {"schema_version": "upgrade-bench-v2/score-cache/1", "key": key, "payload": payload},
    )
    return scores, {"key": key, "hit": False}


def _candidate_path(candidate_root: Path, chain: str, track: str, fold: str) -> Path:
    stem = "candidates" if track == "a" else "candidates_firsttime"
    suffix = "_fold2" if fold == HISTORY_FOLD else ""
    return candidate_root / f"{stem}_{chain}{suffix}.csv"


def _expected_windows(fold: str) -> tuple[str, str]:
    if fold == HISTORY_FOLD:
        return "1998-2002", "2008-2012"
    if fold == TARGET_FOLD:
        return "2008-2012", "2018-2022"
    raise ProtocolError(f"strict rolling runner does not permit fold {fold!r}")


def _validate_candidate_metadata(path: Path, fold: str) -> None:
    import pandas as pd

    if not path.is_file():
        raise ProtocolError(f"candidate table is missing: {path}")
    header = pd.read_csv(
        path,
        nrows=1,
        usecols=["aggregation", "early_window", "late_window", "benchmark_version"],
    )
    if header.empty:
        raise ProtocolError(f"candidate table is empty: {path}")
    row = header.iloc[0]
    early, late = _expected_windows(fold)
    if str(row["aggregation"]) != "calendar_mean":
        raise ProtocolError(f"non-v2 aggregation in {path}")
    if str(row["early_window"]) != early or str(row["late_window"]) != late:
        raise ProtocolError(
            f"window mismatch in {path}: {(row['early_window'], row['late_window'])} != {(early, late)}"
        )


def _read_identities(path: Path, fold: str):
    """Read keys only; in evaluate this happens before representation fitting."""
    import pandas as pd

    _validate_candidate_metadata(path, fold)
    identities = pd.read_csv(path, usecols=list(KEYS), dtype={name: str for name in KEYS})
    identities = identities.loc[:, list(KEYS)]
    if identities.isna().any().any() or identities.duplicated().any():
        raise ProtocolError(f"candidate keys are null or duplicated: {path}")
    ordered = identities.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    if not identities.reset_index(drop=True).equals(ordered):
        raise ProtocolError(f"candidate table is not in deterministic key order: {path}")
    return identities


def _read_labels(path: Path, expected_identities):
    """Read labels/features separately and assert exact row-key alignment."""
    import pandas as pd

    cols = list(KEYS) + ["y", "size", "lateval"]
    frame = pd.read_csv(path, usecols=cols, dtype={name: str for name in KEYS})
    if not frame.loc[:, list(KEYS)].equals(expected_identities.reset_index(drop=True)):
        raise ProtocolError(f"label rows do not align with the previously scored identities: {path}")
    if not set(frame["y"].dropna().unique()).issubset({0, 1}):
        raise ProtocolError(f"non-binary target in {path}")
    return frame


def _resolve_device(requested: str, require_cuda: bool) -> str:
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ProtocolError("CUDA was requested but torch.cuda.is_available() is false")
    if require_cuda and not requested.startswith("cuda"):
        raise ProtocolError("this run requires CUDA; remove --require-cuda only for a small smoke run")
    return requested


def _safe_binary_metrics(y, score) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ProtocolError("model produced non-finite candidate scores")
    if y.sum() == 0 or (y == 0).sum() == 0:
        return {"average_precision": float("nan"), "roc_auc": float("nan")}
    return {
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def _entry_arrays(identities, y, score, lateval=None):
    import numpy as np
    import pandas as pd

    frame = identities.loc[:, ["i_iso", "stage"]].copy()
    frame["y"] = np.asarray(y, dtype=int)
    frame["score"] = np.asarray(score, dtype=float)
    if lateval is not None:
        frame["lateval"] = np.asarray(lateval, dtype=float)
    aggregations = {"y": "max", "score": "max"}
    if lateval is not None:
        aggregations["lateval"] = "sum"
    entry = frame.groupby(["i_iso", "stage"], sort=True, as_index=False).agg(aggregations)
    return entry


def _deterministic_score_order(identities, score, tie_columns) -> object:
    """Descending score order with explicit ascending identity tie-breaks."""
    import numpy as np

    values = np.asarray(score, dtype=float)
    if len(values) != len(identities):
        raise ProtocolError("score and identity row counts differ")
    if not np.isfinite(values).all():
        raise ProtocolError("model produced non-finite candidate scores")
    missing = [column for column in tie_columns if column not in identities]
    if missing:
        raise ProtocolError(f"ranking identities lack tie-break columns: {missing}")
    # np.lexsort uses its final key as primary. Therefore -score is primary,
    # followed by the requested identity columns in their declared order.
    secondary = tuple(
        identities[column].astype(str).to_numpy()
        for column in reversed(tuple(tie_columns))
    )
    return np.lexsort(secondary + (-values,))


def _selection_metric(track: str, identities, y, score, mask) -> float:
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if track == "a":
        return _safe_binary_metrics(np.asarray(y)[mask], np.asarray(score)[mask])["average_precision"]
    subset_identities = identities.loc[mask].reset_index(drop=True)
    subset_y = np.asarray(y)[mask]
    subset_score = np.asarray(score)[mask]
    if track == "b1":
        entry = _entry_arrays(subset_identities, subset_y, subset_score)
        return _safe_binary_metrics(entry["y"], entry["score"])["average_precision"]
    if track == "b2":
        return _conditional_macro(subset_identities, subset_y, subset_score, k=3)["recall"]
    raise ProtocolError(f"unknown task for selection: {track!r}")


def _selection_metric_name(track: str) -> str:
    if track == "a":
        return "track_a_lane_average_precision"
    if track == "b1":
        return "track_b1_entry_average_precision_max_lane_score"
    if track == "b2":
        return "track_b2_positive_entry_macro_recall_at_3"
    raise ProtocolError(f"unknown task for selection metric: {track!r}")


def _assert_exporter_stage_partition(identities, history_holdout) -> None:
    """Fail if any Track-A/B capability group crosses historical dev/holdout."""
    import numpy as np

    audit = identities.loc[:, ["i_iso", "stage"]].copy()
    audit["history_holdout"] = np.asarray(history_holdout, dtype=bool)
    crossing = audit.groupby(["i_iso", "stage"], sort=False)["history_holdout"].nunique()
    if int(crossing.max()) > 1:
        raise ProtocolError("historical split leaks an exporter-stage group across dev/holdout")


def _within_size_auc(y, score, size, bins=10) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    size = np.asarray(size, dtype=float)
    cuts = np.quantile(size, np.linspace(0.0, 1.0, bins + 1))
    values, weights = [], []
    for idx in range(bins):
        mask = (size >= cuts[idx]) & ((size <= cuts[idx + 1]) if idx == bins - 1 else (size < cuts[idx + 1]))
        if y[mask].sum() >= 3 and (y[mask] == 0).sum() >= 3:
            values.append(roc_auc_score(y[mask], score[mask]))
            weights.append(mask.sum())
    return float(np.average(values, weights=weights)) if values else float("nan")


def _conditional_macro(identities, y, score, lateval=None, *, k: int) -> dict[str, object]:
    """Track-B2 macro outcome over historical/target positive entry groups."""
    import numpy as np

    lane = identities.copy()
    lane["y"] = np.asarray(y, dtype=int)
    lane["score"] = np.asarray(score, dtype=float)
    if lateval is not None:
        lane["lateval"] = np.asarray(lateval, dtype=float)
    recalls, values = [], []
    group_keys = []
    for (exporter, stage), group in lane.groupby(["i_iso", "stage"], sort=True):
        positives = int(group["y"].sum())
        if positives == 0:
            continue
        top = group.sort_values(
            ["score", "j_iso"],
            ascending=[False, True],
            kind="mergesort",
        ).head(k)
        recalls.append(float(top["y"].sum() / positives))
        if lateval is not None:
            total_value = max(float(group["lateval"].sum()), 1.0)
            values.append(float(top["lateval"].sum() / total_value))
        group_keys.append(f"{exporter}|{stage}")
    return {
        "recall": float(np.mean(recalls)) if recalls else float("nan"),
        "value_capture": float(np.mean(values)) if values else float("nan"),
        "recall_by_entry": np.asarray(recalls, dtype=float),
        "value_by_entry": np.asarray(values, dtype=float),
        "group_keys": group_keys,
    }


def _ranking_metrics(track: str, identities, labels, score) -> dict[str, float]:
    import numpy as np

    y = labels["y"].to_numpy(dtype=int)
    size = labels["size"].to_numpy(dtype=float)
    lateval = labels["lateval"].to_numpy(dtype=float)
    result = {f"lane_{key}": value for key, value in _safe_binary_metrics(y, score).items()}
    result["within_size_decile_auc"] = _within_size_auc(y, score, size)
    if track == "a":
        total_pos = max(int(y.sum()), 1)
        total_value = max(float(lateval.sum()), 1.0)
        order = _deterministic_score_order(
            identities, score, ("i_iso", "stage", "j_iso")
        )
        for budget in (50, 100, 250, 500):
            chosen = order[: min(budget, len(order))]
            result[f"precision_at_{budget}"] = float(y[chosen].mean()) if len(chosen) else float("nan")
            result[f"recall_at_{budget}"] = float(y[chosen].sum() / total_pos)
            result[f"value_capture_at_{budget}"] = float(lateval[chosen].sum() / total_value)
        lane = identities.copy()
        lane["y"] = y
        lane["score"] = np.asarray(score, dtype=float)
        lane["lateval"] = lateval
        for k in (5, 10):
            precisions, recalls, values = [], [], []
            for _, group in lane.groupby("i_iso", sort=True):
                top = group.sort_values(
                    ["score", "stage", "j_iso"],
                    ascending=[False, True, True],
                    kind="mergesort",
                ).head(k)
                precisions.append(float(top["y"].mean()))
                positives = int(group["y"].sum())
                if positives:
                    recalls.append(float(top["y"].sum() / positives))
                realized = float(group["lateval"].sum())
                if realized > 0:
                    values.append(float(top["lateval"].sum() / realized))
            result[f"exporter_macro_precision_at_{k}"] = float(np.mean(precisions))
            result[f"exporter_macro_recall_at_{k}"] = float(np.mean(recalls)) if recalls else float("nan")
            result[f"exporter_macro_value_capture_at_{k}"] = float(np.mean(values)) if values else float("nan")
        result["exporters"] = int(lane["i_iso"].nunique())
        return result

    if track == "b1":
        entry = _entry_arrays(identities, y, score)
        result.update({f"entry_{key}": value for key, value in _safe_binary_metrics(entry["y"], entry["score"]).items()})
        total_positive_entries = max(int(entry["y"].sum()), 1)
        order = _deterministic_score_order(entry, entry["score"], ("i_iso", "stage"))
        for budget in (25, 50, 100, 250):
            chosen = order[: min(budget, len(order))]
            result[f"entry_precision_at_{budget}"] = (
                float(entry["y"].to_numpy()[chosen].mean()) if len(chosen) else float("nan")
            )
            result[f"entry_recall_at_{budget}"] = float(
                entry["y"].to_numpy()[chosen].sum() / total_positive_entries
            )
        result["entry_groups"] = int(len(entry))
        return result

    # Track B2: destination ranking conditional on a true entry event.
    positive_groups = None
    for k in (1, 3, 5):
        conditional = _conditional_macro(identities, y, score, lateval, k=k)
        result[f"conditional_recall_at_{k}"] = float(conditional["recall"])
        result[f"conditional_value_capture_at_{k}"] = float(conditional["value_capture"])
        positive_groups = len(conditional["group_keys"])
    result["positive_entry_groups"] = int(positive_groups or 0)
    return result


def _summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    import numpy as np

    keys = sorted(set().union(*(run.keys() for run in runs)))
    summary = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs if key in run], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "n": int(len(values)),
        }
    return summary


def _cluster_bootstrap(track: str, identities, labels, score, *, iters: int, seed: int) -> dict[str, object]:
    """Cluster bootstrap the task-primary metric at its normative unit."""
    import numpy as np

    if iters < 1:
        raise ProtocolError("bootstrap iterations must be positive")
    rng = np.random.default_rng(int(seed))
    y = labels["y"].to_numpy(dtype=int)
    score = np.asarray(score, dtype=float)
    estimates = []
    if track == "a":
        cluster_unit = "exporter"
        metric = "lane_average_precision"
        groups = [idx.to_numpy() for _, idx in identities.groupby("i_iso", sort=True).groups.items()]
        for _ in range(iters):
            sampled = rng.integers(0, len(groups), size=len(groups))
            rows = np.concatenate([groups[index] for index in sampled])
            estimates.append(_safe_binary_metrics(y[rows], score[rows])["average_precision"])
    elif track == "b1":
        cluster_unit = "exporter"
        metric = "entry_average_precision"
        entry = _entry_arrays(identities, y, score)
        ey = entry["y"].to_numpy(dtype=int)
        es = entry["score"].to_numpy(dtype=float)
        groups = [idx.to_numpy() for _, idx in entry.groupby("i_iso", sort=True).groups.items()]
        for _ in range(iters):
            sampled = rng.integers(0, len(groups), size=len(groups))
            rows = np.concatenate([groups[index] for index in sampled])
            estimates.append(_safe_binary_metrics(ey[rows], es[rows])["average_precision"])
    else:
        cluster_unit = "exporter_stage"
        metric = "positive_entry_macro_recall_at_3"
        values = _conditional_macro(identities, y, score, k=3)["recall_by_entry"]
        if not len(values):
            estimates = [float("nan")] * iters
        else:
            for _ in range(iters):
                estimates.append(float(np.mean(values[rng.integers(0, len(values), size=len(values))])))
    estimates = np.asarray(estimates, dtype=float)
    return {
        "cluster_unit": cluster_unit,
        "metric": metric,
        "iterations": int(iters),
        "seed": int(seed),
        "lower_95": float(np.nanpercentile(estimates, 2.5)),
        "upper_95": float(np.nanpercentile(estimates, 97.5)),
    }


def _prepare_runtime(chain: str, fold: str):
    """Import graph loaders only after the caller has satisfied its phase gate."""
    os.environ["VCU_FOLD"] = fold
    os.environ["VCU_WINDOW_AGG"] = "calendar_mean"
    import universe as U
    from benchmark import setup_early_graph

    U.set_active_chain(chain)
    triples, early = setup_early_graph(fold=fold, aggregation="calendar_mean")
    return U, triples, early


def _train_score_kge(triples, identities, model: str, hp: Mapping[str, object], seed: int, device: str):
    import numpy as np
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    inverse = model == "CompGCN"
    factory = TriplesFactory.from_labeled_triples(triples, create_inverse_triples=inverse)
    result = pipeline(
        training=factory,
        # PyKEEN requires a testing factory even though its built-in metrics are
        # not used here. Reusing the early graph remains label-free; the only
        # paper-facing evaluation is computed later on candidate outcomes.
        testing=factory,
        model=model,
        model_kwargs={"embedding_dim": int(hp["embedding_dim"])},
        optimizer_kwargs={"lr": float(hp["learning_rate"])},
        training_kwargs={
            "num_epochs": int(hp["epochs"]),
            "batch_size": int(hp.get("batch_size", 2048)),
            "use_tqdm": False,
        },
        random_seed=int(seed),
        device=device,
        evaluation_kwargs={"use_tqdm": False},
    )
    entity = factory.entity_to_id
    relation = factory.relation_to_id
    missing = [
        tuple(row)
        for row in identities.loc[:, list(KEYS)].to_numpy()
        if row[0] not in entity or row[1] not in entity or row[2] not in relation
    ]
    if missing:
        raise ProtocolError(f"{len(missing)} candidate triples are outside the early KGE vocabulary")
    encoded = torch.tensor(
        [[entity[i], relation[stage], entity[j]] for i, j, stage in identities.loc[:, list(KEYS)].to_numpy()],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        return result.model.score_hrt(encoded).detach().cpu().numpy().reshape(-1).astype(np.float64)


def _build_nbfnet_graph(triples, U, device: str):
    import torch
    from torch_geometric.data import Data

    all_relations = set(str(row[1]) for row in triples)
    relation_names = [name for name in U.EXPORT_RELATIONS if name in all_relations]
    relation_names += sorted(all_relations - set(relation_names))
    relations = {name: idx for idx, name in enumerate(relation_names)}
    entities = sorted(set(str(v) for row in triples for v in (row[0], row[2])))
    entity = {name: idx for idx, name in enumerate(entities)}
    heads = torch.tensor([entity[str(row[0])] for row in triples], dtype=torch.long)
    tails = torch.tensor([entity[str(row[2])] for row in triples], dtype=torch.long)
    rels = torch.tensor([relations[str(row[1])] for row in triples], dtype=torch.long)
    fwd = torch.stack([heads, tails])
    edge_index = torch.cat([fwd, torch.stack([tails, heads])], dim=1)
    edge_type = torch.cat([rels, rels + len(relations)])
    supervised = torch.tensor([str(row[1]) in set(U.EXPORT_RELATIONS) for row in triples], dtype=torch.bool)
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=len(entities),
        target_edge_index=fwd[:, supervised],
        target_edge_type=rels[supervised],
        num_relations=2 * len(relations),
    )
    for field in ("edge_index", "edge_type", "target_edge_index", "target_edge_type"):
        setattr(data, field, getattr(data, field).to(device))
    return data, entity, relations


def _train_score_nbfnet(context, identities, hp: Mapping[str, object], seed: int, device: str):
    import numpy as np
    import torch
    from gap_discovery import train_model
    from nbfnet import tasks

    data, entity, relation = context
    unknown = [
        tuple(row)
        for row in identities.loc[:, list(KEYS)].to_numpy()
        if row[0] not in entity or row[1] not in entity or row[2] not in relation
    ]
    if unknown:
        raise ProtocolError(f"{len(unknown)} candidate triples are outside the early NBFNet vocabulary")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = train_model(
        data,
        device,
        epochs=int(hp["epochs"]),
        bs=int(hp.get("batch_size", 64)),
        neg=int(hp.get("negatives", 32)),
        lr=float(hp["learning_rate"]),
        num_rel=int(data.num_relations),
        layers=int(hp["layers"]),
    )
    scores = np.empty(len(identities), dtype=np.float64)
    grouped = {}
    for idx, (i, j, stage) in enumerate(identities.loc[:, list(KEYS)].to_numpy()):
        grouped.setdefault((i, stage), []).append((idx, j))
    model.eval()
    with torch.no_grad():
        for (head, stage), rows in grouped.items():
            query = torch.tensor([[entity[head], entity[head], relation[stage]]], device=device)
            batch, _ = tasks.all_negative(data, query)
            all_scores = model(data, batch).squeeze(0).detach().cpu().numpy()
            for idx, tail in rows:
                scores[idx] = all_scores[entity[tail]]
    return scores


def _score_factory(family: str, triples, identities, U, device: str) -> Callable:
    if family == "kge":
        return lambda model, hp, seed: _train_score_kge(triples, identities, model, hp, seed, device)
    context = _build_nbfnet_graph(triples, U, device)
    return lambda model, hp, seed: _train_score_nbfnet(context, identities, hp, seed, device)


def _selection_grids(args) -> dict[str, list[dict[str, object]]]:
    if args.family == "kge":
        epochs = 150 if args.epochs is None else args.epochs
        models = _csv(args.models)
        unknown = sorted(set(models) - set(DEFAULT_KGE_MODELS))
        if unknown:
            raise ProtocolError(f"unknown KGE models: {unknown}")
        dims = _csv(args.dims, int)
        rates = _csv(args.learning_rates, float)
        return {
            model: [
                {
                    "embedding_dim": dim,
                    "learning_rate": rate,
                    "epochs": epochs,
                    "batch_size": args.kge_batch_size,
                }
                for dim in dims
                for rate in rates
            ]
            for model in models
        }
    epochs = 25 if args.epochs is None else args.epochs
    layers = _csv(args.layers, int)
    rates = _csv(args.nbfnet_learning_rates, float)
    return {
        "NBFNet": [
            {
                "layers": layer,
                "learning_rate": rate,
                "epochs": epochs,
                "batch_size": args.nbfnet_batch_size,
                "negatives": args.nbfnet_negatives,
            }
            for layer in layers
            for rate in rates
        ]
    }


def _config_value(config: Mapping[str, object], path: tuple[str, ...]) -> object:
    """Read one required formal-config field without accepting implicit defaults."""
    value: object = config
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ProtocolError(f"formal run config is missing {'.'.join(path)}")
        value = value[part]
    return value


def _same_json_value(left: object, right: object) -> bool:
    """Compare JSON values strictly enough to distinguish booleans from integers."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _require_config_value(
    config: Mapping[str, object], path: tuple[str, ...], effective: object
) -> None:
    configured = _config_value(config, path)
    if not _same_json_value(configured, effective):
        name = ".".join(path)
        raise ProtocolError(
            f"effective execution spec does not match formal config at {name}: "
            f"CLI/code={effective!r}, config={configured!r}"
        )


def _validate_formal_execution_spec(
    args, config: Mapping[str, object], *, phase: str
) -> None:
    """Fail closed when a formal CLI invocation differs from its run config.

    Only arguments that affect the requested phase/family are compared.  Thus
    irrelevant parser defaults (for example NBFNet layers during a KGE job) are
    not treated as part of the effective execution specification.
    """
    if phase not in {"select-chain", "evaluate-chain"}:
        raise ProtocolError(f"unsupported formal GPU phase: {phase!r}")

    # These values define the global freeze domain and the data protocol.  Exact
    # sequence equality is intentional because declared ordering can determine
    # deterministic tie breaking and manifest order.
    _require_config_value(config, ("protocol",), PROTOCOL)
    _require_config_value(config, ("chains",), list(DEFAULT_CHAINS))
    _require_config_value(config, ("tracks",), list(TRACKS))
    _require_config_value(config, ("families",), list(FAMILIES))
    _require_config_value(config, ("selection_fold",), HISTORY_FOLD)
    _require_config_value(config, ("target_fold",), TARGET_FOLD)
    _require_config_value(config, ("aggregation",), "calendar_mean")
    _require_config_value(
        config,
        ("expected_selection_count",),
        len(DEFAULT_CHAINS) * len(TRACKS) * len(FAMILIES),
    )
    _require_config_value(
        config,
        ("selection_orchestration_job_count",),
        len(DEFAULT_CHAINS) * len(FAMILIES),
    )

    if args.chain not in DEFAULT_CHAINS:
        raise ProtocolError(
            f"chain {args.chain!r} is outside the formal run config domain {DEFAULT_CHAINS}"
        )
    if args.family not in FAMILIES:
        raise ProtocolError(
            f"family {args.family!r} is outside the formal run config domain {FAMILIES}"
        )

    seeds = _csv(args.seeds, int)
    _require_config_value(config, ("selection", "evaluation_seeds"), seeds)
    _require_config_value(config, ("selection", "split_unit"), "exporter_stage")

    # The frozen config uses task-local metric names while selection artifacts
    # namespace the same names by task.  Derive the complete config mapping
    # from the metric function actually used by this runner, and require exact
    # key/value equality so a stale or partially changed mapping fails closed.
    metric_names: dict[str, str] = {}
    for track in TRACKS:
        namespaced = _selection_metric_name(track)
        prefix = f"track_{track}_"
        if not namespaced.startswith(prefix):
            raise ProtocolError(
                f"runner selection metric {namespaced!r} lacks required prefix {prefix!r}"
            )
        metric_names[track] = namespaced[len(prefix) :]
    _require_config_value(
        config, ("selection", "primary_metric_by_task"), metric_names
    )

    if phase == "evaluate-chain":
        if int(args.bootstrap_iters) != FORMAL_BOOTSTRAP_ITERS:
            raise ProtocolError(
                "formal evaluate-chain requires "
                f"--bootstrap-iters={FORMAL_BOOTSTRAP_ITERS}"
            )
        if int(args.bootstrap_seed) != FORMAL_BOOTSTRAP_SEED:
            raise ProtocolError(
                "formal evaluate-chain requires "
                f"--bootstrap-seed={FORMAL_BOOTSTRAP_SEED}"
            )
        return

    # Constructing the grid also applies the runner's model allow-list before
    # any candidate table or graph is opened.
    _selection_grids(args)
    _require_config_value(config, ("selection", "split_salt"), str(args.split_salt))
    _require_config_value(
        config, ("selection", "selection_seed"), int(args.selection_seed)
    )

    if args.family == "kge":
        epochs = 150 if args.epochs is None else int(args.epochs)
        _require_config_value(config, ("kge", "models"), _csv(args.models))
        _require_config_value(config, ("kge", "embedding_dims"), _csv(args.dims, int))
        _require_config_value(
            config, ("kge", "learning_rates"), _csv(args.learning_rates, float)
        )
        _require_config_value(config, ("kge", "epochs"), epochs)
        _require_config_value(
            config, ("kge", "batch_size"), int(args.kge_batch_size)
        )
        return

    epochs = 25 if args.epochs is None else int(args.epochs)
    _require_config_value(config, ("nbfnet", "model"), "NBFNet")
    _require_config_value(config, ("nbfnet", "layers"), _csv(args.layers, int))
    _require_config_value(
        config,
        ("nbfnet", "learning_rates"),
        _csv(args.nbfnet_learning_rates, float),
    )
    _require_config_value(config, ("nbfnet", "epochs"), epochs)
    _require_config_value(
        config, ("nbfnet", "batch_size"), int(args.nbfnet_batch_size)
    )
    _require_config_value(
        config, ("nbfnet", "negatives"), int(args.nbfnet_negatives)
    )


def _assert_complete_global_freeze(indexed) -> None:
    """Require all 36 chain/track/family selections before main can open."""
    required_keys = {
        selection_key(chain, track, family)
        for chain in DEFAULT_CHAINS
        for track in TRACKS
        for family in FAMILIES
    }
    if set(indexed) != required_keys:
        missing = sorted(required_keys - set(indexed))
        extra = sorted(set(indexed) - required_keys)
        raise ProtocolError(
            "main gate requires the complete 6-chain x 3-task x 2-family freeze "
            f"(missing={missing}, extra={extra})"
        )


def _select(args) -> int:
    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)
    candidate = _candidate_path(candidate_root, args.chain, args.track, HISTORY_FOLD)
    run_id, run_config, run_config_sha256 = _resolve_run_lock(args)
    grids = _selection_grids(args)
    plan = {
        "phase": "select",
        "fold": HISTORY_FOLD,
        "chain": args.chain,
        "track": args.track,
        "family": args.family,
        "candidate": str(candidate),
        "models": {key: value for key, value in grids.items()},
        "seeds": _csv(args.seeds, int),
        "main_target_labels_accessed": False,
        "run_id": run_id,
        "run_config": str(run_config),
        "run_config_sha256": run_config_sha256,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    device = _resolve_device(args.device, args.require_cuda)
    identities = _read_identities(candidate, HISTORY_FOLD)
    labels = _read_labels(candidate, identities)  # fold2 labels are permitted for selection
    y = labels["y"].to_numpy(dtype=int)

    # Explicit historical group split: inner dev chooses HP; history holdout chooses model family.
    from split import split_test_mask

    history_holdout = split_test_mask(
        args.chain,
        identities["i_iso"],
        identities["stage"],
        identities["j_iso"],
        unit="exporter_stage",
        salt=args.split_salt,
    )
    history_dev = ~history_holdout
    _assert_exporter_stage_partition(identities, history_holdout)
    U, triples, _ = _prepare_runtime(args.chain, HISTORY_FOLD)
    scorer = _score_factory(args.family, triples, identities, U, device)
    selection_seed = int(args.selection_seed)
    seeds = _csv(args.seeds, int)
    rows = []
    family_complete = True
    for model, configs in grids.items():
        best = None
        best_seed_scores = None
        trials = []
        grid_complete = True
        for hp in configs:
            print(f"[select] {args.chain} track={args.track} {model} hp={hp} seed={selection_seed}", flush=True)
            try:
                scores = scorer(model, hp, selection_seed)
                value = _selection_metric(args.track, identities, y, scores, history_dev)
            except Exception as exc:
                grid_complete = False
                trials.append({"hyperparameters": hp, "status": "failed", "error": str(exc)[:1000]})
                print(f"[select] {model} hp={hp} FAILED: {exc}", flush=True)
                continue
            trial = {"hyperparameters": hp, "status": "complete", "history_dev_selection_metric": value}
            trials.append(trial)
            if best is None or value > best[0]:
                best = (value, hp)
                best_seed_scores = scores
        if best is None:
            family_complete = False
            rows.append({"model": model, "status": "failed", "grid": trials})
            continue
        holdout_values = []
        per_seed = []
        for seed in seeds:
            try:
                scores = best_seed_scores if seed == selection_seed else scorer(model, best[1], seed)
                value = _selection_metric(args.track, identities, y, scores, history_holdout)
                holdout_values.append(value)
                per_seed.append({"seed": seed, "status": "complete", "history_holdout_selection_metric": value})
            except Exception as exc:
                family_complete = False
                per_seed.append({"seed": seed, "status": "failed", "error": str(exc)[:1000]})
                print(f"[select] {model} seed={seed} FAILED: {exc}", flush=True)
        if len(holdout_values) != len(seeds) or not grid_complete:
            family_complete = False
            rows.append(
                {
                    "model": model,
                    "status": "incomplete",
                    "selected_hyperparameters": best[1],
                    "history_dev_selection_metric": float(best[0]),
                    "per_seed": per_seed,
                    "grid": trials,
                }
            )
            continue
        import numpy as np

        rows.append(
            {
                "model": model,
                "status": "complete",
                "selected_hyperparameters": best[1],
                "history_dev_selection_metric": float(best[0]),
                "history_holdout_selection_metric_mean": float(np.mean(holdout_values)),
                "history_holdout_selection_metric_std": float(np.std(holdout_values)),
                "per_seed": per_seed,
                "grid": trials,
            }
        )
    complete_rows = [row for row in rows if row.get("status") == "complete"]
    rows.sort(key=lambda row: (row.get("status") != "complete", -row.get("history_holdout_selection_metric_mean", -1.0), row["model"]))
    if len(complete_rows) != len(grids):
        family_complete = False
    winner = max(complete_rows, key=lambda row: row["history_holdout_selection_metric_mean"]) if complete_rows else None
    payload = {
        "schema_version": SELECTION_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete" if family_complete else "incomplete",
        "created_at_utc": _utc_now(),
        "chain": args.chain,
        "track": args.track,
        "family": args.family,
        "run_id": run_id,
        "run_config": str(run_config.relative_to(ROOT) if run_config.is_relative_to(ROOT) else run_config),
        "run_config_sha256": run_config_sha256,
        "selection_fold": HISTORY_FOLD,
        "target_fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
        "main_target_labels_accessed": False,
        "selection_design": {
            "hp_partition": "fold2 exporter_stage dev",
            "model_partition": "fold2 exporter_stage holdout",
            "split_unit": "exporter_stage",
            "split_salt": str(args.split_salt),
            "primary_metric": _selection_metric_name(args.track),
            "selection_seed": selection_seed,
            "evaluation_seeds": seeds,
        },
        "history_candidate": {
            "path": str(candidate.relative_to(ROOT) if candidate.is_relative_to(ROOT) else candidate),
            "sha256": sha256_file(candidate),
            "rows": int(len(identities)),
            "positive_lanes": int(y.sum()),
        },
        "models": rows,
        "representation_policy": "refit selected label-free model from scratch on main early graph per seed",
        "raw_score_policy": "one column per seed; no cross-seed raw-score average",
    }
    if winner is not None:
        payload["selected"] = {
            "model": winner["model"],
            "hyperparameters": winner["selected_hyperparameters"],
            "history_holdout_selection_metric_mean": winner["history_holdout_selection_metric_mean"],
        }
    out = output_root / "selections" / selection_filename(args.chain, args.track, args.family)
    write_json_atomic(out, payload, overwrite=args.overwrite)
    print(f"selection written (status={payload['status']}; not yet globally frozen): {out}")
    return 0 if family_complete else 3


def _select_chain(args) -> int:
    """Shared-training fold2 selection for A, B1, and B2 on one chain/family."""
    run_id, run_config, run_config_sha256 = _resolve_run_lock(
        args, phase="select-chain"
    )

    import numpy as np
    import pandas as pd
    from split import split_test_mask

    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)
    paths = {
        "a": _candidate_path(candidate_root, args.chain, "a", HISTORY_FOLD),
        "b": _candidate_path(candidate_root, args.chain, "b1", HISTORY_FOLD),
    }
    grids = _selection_grids(args)
    if args.dry_run:
        trainings_grid = sum(len(configs) for configs in grids.values())
        max_selected = len(grids) * len(TRACKS)
        print(
            json.dumps(
                {
                    "phase": "select-chain",
                    "fold": HISTORY_FOLD,
                    "chain": args.chain,
                    "family": args.family,
                    "tasks": list(TRACKS),
                    "candidate_paths": {key: str(value) for key, value in paths.items()},
                    "grid_seed_trainings": trainings_grid,
                    "max_unique_selected_configs": max_selected,
                    "raw_scores_shared_across_tasks": True,
                    "run_id": run_id,
                    "run_config_sha256": run_config_sha256,
                    "main_target_labels_accessed": False,
                },
                indent=2,
            )
        )
        return 0
    identities = {
        "a": _read_identities(paths["a"], HISTORY_FOLD),
        "b": _read_identities(paths["b"], HISTORY_FOLD),
    }
    labels = {
        key: _read_labels(paths[key], identities[key]) for key in ("a", "b")
    }
    combined = pd.concat([identities["a"], identities["b"]], ignore_index=True)
    a_rows = len(identities["a"])
    slices = {"a": slice(0, a_rows), "b1": slice(a_rows, len(combined)), "b2": slice(a_rows, len(combined))}
    task_source = {"a": "a", "b1": "b", "b2": "b"}
    masks = {}
    for task in TRACKS:
        source = task_source[task]
        holdout = split_test_mask(
            args.chain,
            identities[source]["i_iso"],
            identities[source]["stage"],
            identities[source]["j_iso"],
            unit="exporter_stage",
            salt=args.split_salt,
        )
        _assert_exporter_stage_partition(identities[source], holdout)
        masks[task] = {"dev": ~holdout, "holdout": holdout}

    device = _resolve_device(args.device, args.require_cuda)
    U, triples, _ = _prepare_runtime(args.chain, HISTORY_FOLD)
    scorer = _score_factory(args.family, triples, combined, U, device)
    cache_context = {
        "schema_version": "upgrade-bench-v2/score-cache-context/1",
        "protocol": PROTOCOL,
        "phase": "fold2-selection",
        "fold": HISTORY_FOLD,
        "aggregation": "calendar_mean",
        "chain": args.chain,
        "family": args.family,
        "run_id": run_id,
        "run_config_sha256": run_config_sha256,
        "graph_sha256": _stable_triple_hash(triples),
        "combined_identity_sha256": _stable_frame_hash(combined),
        "candidate_files_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "code_sha256": _code_hashes(),
    }
    cache_root = output_root / "score_cache" / "fold2" / args.chain / args.family
    selection_seed = int(args.selection_seed)
    seeds = _csv(args.seeds, int)
    trials = {task: {model: [] for model in grids} for task in TRACKS}
    best = {task: {model: None for model in grids} for task in TRACKS}
    grid_complete = {model: True for model in grids}

    def task_arrays(task, combined_scores):
        source = task_source[task]
        return (
            identities[source],
            labels[source]["y"].to_numpy(dtype=int),
            combined_scores[slices[task]],
        )

    for model, configs in grids.items():
        for hp in configs:
            print(f"[select-chain] {args.chain} {args.family} {model} hp={hp} seed={selection_seed}", flush=True)
            try:
                scores, cache = _cached_score(
                    scorer=scorer,
                    model=model,
                    hyperparameters=hp,
                    seed=selection_seed,
                    cache_root=cache_root,
                    cache_context=cache_context,
                    expected_rows=len(combined),
                )
            except Exception as exc:
                grid_complete[model] = False
                for task in TRACKS:
                    trials[task][model].append(
                        {"hyperparameters": hp, "status": "failed", "error": str(exc)[:1000]}
                    )
                print(f"[select-chain] {model} hp={hp} FAILED: {exc}", flush=True)
                continue
            for task in TRACKS:
                task_ids, task_y, task_scores = task_arrays(task, scores)
                value = _selection_metric(task, task_ids, task_y, task_scores, masks[task]["dev"])
                record = {
                    "hyperparameters": hp,
                    "status": "complete",
                    "history_dev_selection_metric": value,
                    "score_cache_key": cache["key"],
                    "score_cache_hit": cache["hit"],
                }
                trials[task][model].append(record)
                current = best[task][model]
                if current is None or value > current[0]:
                    best[task][model] = (value, dict(hp))

    task_rows = {task: [] for task in TRACKS}
    for task in TRACKS:
        for model in grids:
            chosen = best[task][model]
            if chosen is None:
                task_rows[task].append(
                    {"model": model, "status": "failed", "grid": trials[task][model]}
                )
                continue
            holdout_values, per_seed = [], []
            for seed in seeds:
                try:
                    scores, cache = _cached_score(
                        scorer=scorer,
                        model=model,
                        hyperparameters=chosen[1],
                        seed=seed,
                        cache_root=cache_root,
                        cache_context=cache_context,
                        expected_rows=len(combined),
                    )
                    task_ids, task_y, task_scores = task_arrays(task, scores)
                    value = _selection_metric(task, task_ids, task_y, task_scores, masks[task]["holdout"])
                    holdout_values.append(value)
                    per_seed.append(
                        {
                            "seed": seed,
                            "status": "complete",
                            "history_holdout_selection_metric": value,
                            "score_cache_key": cache["key"],
                            "score_cache_hit": cache["hit"],
                        }
                    )
                except Exception as exc:
                    per_seed.append({"seed": seed, "status": "failed", "error": str(exc)[:1000]})
            complete = len(holdout_values) == len(seeds) and grid_complete[model]
            task_rows[task].append(
                {
                    "model": model,
                    "status": "complete" if complete else "incomplete",
                    "selected_hyperparameters": chosen[1],
                    "history_dev_selection_metric": float(chosen[0]),
                    "history_holdout_selection_metric_mean": float(np.mean(holdout_values)) if holdout_values else float("nan"),
                    "history_holdout_selection_metric_std": float(np.std(holdout_values)) if holdout_values else float("nan"),
                    "per_seed": per_seed,
                    "grid": trials[task][model],
                }
            )

    any_incomplete = False
    for task in TRACKS:
        rows = task_rows[task]
        complete_rows = [row for row in rows if row["status"] == "complete"]
        status = "complete" if len(complete_rows) == len(grids) else "incomplete"
        any_incomplete |= status != "complete"
        winner = max(complete_rows, key=lambda row: row["history_holdout_selection_metric_mean"]) if complete_rows else None
        source = task_source[task]
        payload = {
            "schema_version": SELECTION_SCHEMA,
            "protocol": PROTOCOL,
            "status": status,
            "created_at_utc": _utc_now(),
            "chain": args.chain,
            "track": task,
            "family": args.family,
            "run_id": run_id,
            "run_config": str(run_config.relative_to(ROOT) if run_config.is_relative_to(ROOT) else run_config),
            "run_config_sha256": run_config_sha256,
            "selection_fold": HISTORY_FOLD,
            "target_fold": TARGET_FOLD,
            "aggregation": "calendar_mean",
            "main_target_labels_accessed": False,
            "selection_design": {
                "orchestration": "chain_multitask_shared_score_grid",
                "hp_partition": "fold2 exporter_stage dev",
                "model_partition": "fold2 exporter_stage holdout",
                "split_unit": "exporter_stage",
                "split_salt": str(args.split_salt),
                "primary_metric": _selection_metric_name(task),
                "selection_seed": selection_seed,
                "evaluation_seeds": seeds,
            },
            "history_candidate": {
                "path": str(paths[source].relative_to(ROOT) if paths[source].is_relative_to(ROOT) else paths[source]),
                "sha256": sha256_file(paths[source]),
                "rows": int(len(identities[source])),
                "positive_lanes": int(labels[source]["y"].sum()),
            },
            "shared_score_cache": {
                "root": str(cache_root),
                "context_sha256": hashlib.sha256(
                    json.dumps(cache_context, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "combined_rows": int(len(combined)),
            },
            "models": rows,
            "representation_policy": "refit selected label-free model from scratch on main early graph per seed",
            "raw_score_policy": "one column per seed; no cross-seed raw-score average",
        }
        if winner is not None:
            payload["selected"] = {
                "model": winner["model"],
                "hyperparameters": winner["selected_hyperparameters"],
                "history_holdout_selection_metric_mean": winner["history_holdout_selection_metric_mean"],
            }
        out = output_root / "selections" / selection_filename(args.chain, task, args.family)
        write_json_atomic(out, payload, overwrite=args.overwrite)
        print(f"selection task={task} status={status} -> {out}")
    return 3 if any_incomplete else 0


def _freeze(args) -> int:
    output_root = _resolve(args.output_root)
    manifest = _resolve(args.manifest) if args.manifest else output_root / "frozen_manifest.json"
    main_marker = manifest.parent / "MAIN_EVALUATION_STARTED.json"
    if main_marker.exists():
        raise ProtocolError(f"cannot re-freeze after main evaluation has started: {main_marker}")
    combos = expected_combinations(
        _csv(args.chains), _csv(args.tracks), _csv(args.families)
    )
    payload = build_freeze_manifest(
        selection_dir=output_root / "selections",
        manifest_path=manifest,
        combinations=combos,
    )
    payload["created_at_utc"] = _utc_now()
    write_json_atomic(manifest, payload, overwrite=args.overwrite)
    verify_freeze_manifest(manifest)
    print(f"all {len(combos)} selections hash-frozen before main access: {manifest}")
    return 0


def _claim_main_start(marker: Path, *, run_id: str, manifest_sha256: str) -> dict:
    """Create one immutable main-start marker, safe under parallel evaluators."""
    payload = {
        "schema_version": "upgrade-bench-v2/main-start/1",
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "main_started_at_utc": _utc_now(),
        "policy": "freeze and selections are immutable; main outputs never overwrite",
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"main-start marker is unreadable: {marker}") from exc
        if existing.get("run_id") != run_id or existing.get("manifest_sha256") != manifest_sha256:
            raise ProtocolError("main-start marker belongs to a different run/manifest")
        return existing
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _assert_frozen_seeds(selection, requested_seeds) -> list[int]:
    frozen = [int(seed) for seed in selection["selection_design"]["evaluation_seeds"]]
    requested = [int(seed) for seed in requested_seeds]
    if requested != frozen:
        raise ProtocolError(f"evaluation seeds {requested} do not equal frozen seeds {frozen}")
    return frozen


def _evaluate(args) -> int:
    # CRITICAL GATE: this pure-Python verification happens before _prepare_runtime,
    # pandas candidate reads, or any import that can load the main target cohort.
    manifest_path = _resolve(args.manifest)
    manifest, indexed = verify_freeze_manifest(manifest_path)
    _assert_complete_global_freeze(indexed)
    run_id, run_config, run_config_sha256 = _resolve_run_lock(args)
    if run_id != manifest["run_id"] or run_config_sha256 != manifest["run_config_sha256"]:
        raise ProtocolError("requested run config does not match the frozen manifest")
    manifest_sha256 = sha256_file(manifest_path)
    gate_verified_at = _utc_now()
    key = selection_key(args.chain, args.track, args.family)
    if key not in indexed:
        raise ProtocolError(f"requested evaluation is absent from the global freeze: {key}")
    freeze_entry, selection = indexed[key]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "evaluate",
                    "gate": "verified",
                    "manifest": str(manifest_path),
                    "frozen_entries": len(manifest["entries"]),
                    "requested": key,
                    "selected": selection["selected"],
                    "target_fold": TARGET_FOLD,
                    "run_id": run_id,
                    "run_config_sha256": run_config_sha256,
                    "manifest_sha256": manifest_sha256,
                    "target_labels_will_be_read": "after all representations are fitted and candidates scored",
                },
                indent=2,
            )
        )
        return 0

    if args.overwrite:
        raise ProtocolError("main evaluation never permits --overwrite")
    seeds = _csv(args.seeds, int)
    _assert_frozen_seeds(selection, seeds)
    _claim_main_start(
        manifest_path.parent / "MAIN_EVALUATION_STARTED.json",
        run_id=run_id,
        manifest_sha256=manifest_sha256,
    )

    device = _resolve_device(args.device, args.require_cuda)
    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)
    candidate = _candidate_path(candidate_root, args.chain, args.track, TARGET_FOLD)

    identities = _read_identities(candidate, TARGET_FOLD)  # keys/metadata only; no y/lateval
    U, triples, _ = _prepare_runtime(args.chain, TARGET_FOLD)  # early graph only
    scorer = _score_factory(args.family, triples, identities, U, device)
    model = selection["selected"]["model"]
    hp = selection["selected"]["hyperparameters"]
    score_runs = []
    representation_finished = []
    for seed in seeds:
        print(f"[evaluate] {key} model={model} hp={hp} seed={seed}", flush=True)
        score_runs.append(scorer(model, hp, seed))
        representation_finished.append({"seed": seed, "finished_at_utc": _utc_now()})

    # First and only target-label read in this process.  It occurs after every
    # selected representation has been fitted and all candidate scores exist.
    labels = _read_labels(candidate, identities)
    target_labels_read_at = _utc_now()
    metrics_by_seed = []
    uncertainty_by_seed = []
    for seed, scores in zip(seeds, score_runs):
        metrics_by_seed.append({"seed": seed, **_ranking_metrics(args.track, identities, labels, scores)})
        uncertainty_by_seed.append(
            {
                "seed": seed,
                **_cluster_bootstrap(
                    args.track,
                    identities,
                    labels,
                    scores,
                    iters=args.bootstrap_iters,
                    seed=args.bootstrap_seed + seed,
                ),
            }
        )
    summary = _summarize_runs([{k: v for k, v in row.items() if k != "seed"} for row in metrics_by_seed])

    score_dir = output_root / "scores"
    metric_dir = output_root / "metrics"
    score_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"scores_{args.chain}_track-{args.track}_{args.family}.csv"
    metric_path = metric_dir / f"metrics_{args.chain}_track-{args.track}_{args.family}.json"
    if score_path.exists() or metric_path.exists():
        raise ProtocolError(f"refusing to overwrite an existing main evaluation for {key}")
    scores_out = identities.copy()
    for seed, scores in zip(seeds, score_runs):
        scores_out[f"score_{model}_s{seed}"] = scores
    scores_out["selection_sha256"] = freeze_entry["sha256"]
    scores_out["protocol"] = PROTOCOL
    scores_out.to_csv(score_path, index=False)
    payload = {
        "schema_version": "upgrade-bench-v2/gpu-evaluation/1",
        "protocol": PROTOCOL,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "chain": args.chain,
        "track": args.track,
        "family": args.family,
        "selection_manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "selection_sha256": freeze_entry["sha256"],
        "run_id": run_id,
        "run_config": str(run_config),
        "run_config_sha256": run_config_sha256,
        "selected": selection["selected"],
        "target_fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
        "cohort_policy": "complete main cohort; no same-window dev/test split",
        "representation_policy": selection["representation_policy"],
        "raw_score_policy": selection["raw_score_policy"],
        "protocol_timestamps": {
            "full_freeze_gate_verified_at_utc": gate_verified_at,
            "representations_finished": representation_finished,
            "main_target_labels_read_at_utc": target_labels_read_at,
            "ordering_attestation": "all representation finish events precede the target-label read",
        },
        "target_candidate": {
            "path": str(candidate.relative_to(ROOT) if candidate.is_relative_to(ROOT) else candidate),
            "sha256": sha256_file(candidate),
            "rows": int(len(identities)),
            "positive_lanes": int(labels["y"].sum()),
        },
        "seeds": seeds,
        "per_seed": metrics_by_seed,
        "cluster_bootstrap_by_seed": uncertainty_by_seed,
        "summary": summary,
        "score_artifact": str(score_path),
    }
    write_json_atomic(metric_path, payload, overwrite=args.overwrite)
    print(f"strict rolling scores:  {score_path}")
    print(f"strict rolling metrics: {metric_path}")
    return 0


def _evaluate_chain(args) -> int:
    """Shared-training complete-main evaluation for one chain/family."""
    run_id, run_config, run_config_sha256 = _resolve_run_lock(
        args, phase="evaluate-chain"
    )

    # Pure protocol gate first: no target loader imports above this line.
    manifest_path = _resolve(args.manifest)
    manifest, indexed = verify_freeze_manifest(manifest_path)
    _assert_complete_global_freeze(indexed)
    if run_id != manifest["run_id"] or run_config_sha256 != manifest["run_config_sha256"]:
        raise ProtocolError("requested run config does not match the frozen manifest")
    manifest_sha256 = sha256_file(manifest_path)
    gate_verified_at = _utc_now()
    selected = {}
    freeze_entries = {}
    for task in TRACKS:
        key = selection_key(args.chain, task, args.family)
        if key not in indexed:
            raise ProtocolError(f"requested evaluation is absent from the global freeze: {key}")
        freeze_entries[task], selected[task] = indexed[key]
    seeds = _csv(args.seeds, int)
    for task in TRACKS:
        _assert_frozen_seeds(selected[task], seeds)
    if args.dry_run:
        unique = {
            json.dumps(
                {
                    "model": selected[task]["selected"]["model"],
                    "hyperparameters": selected[task]["selected"]["hyperparameters"],
                },
                sort_keys=True,
            )
            for task in TRACKS
        }
        print(
            json.dumps(
                {
                    "phase": "evaluate-chain",
                    "gate": "verified",
                    "manifest_sha256": manifest_sha256,
                    "run_id": run_id,
                    "chain": args.chain,
                    "family": args.family,
                    "tasks": list(TRACKS),
                    "unique_selected_configs": len(unique),
                    "seeds": seeds,
                    "main_target_labels_accessed": False,
                },
                indent=2,
            )
        )
        return 0
    if args.overwrite:
        raise ProtocolError("main evaluation never permits --overwrite")

    import pandas as pd

    output_root = _resolve(args.output_root)
    score_dir = output_root / "scores"
    metric_dir = output_root / "metrics"
    for task in TRACKS:
        if (score_dir / f"scores_{args.chain}_track-{task}_{args.family}.csv").exists() or (
            metric_dir / f"metrics_{args.chain}_track-{task}_{args.family}.json"
        ).exists():
            raise ProtocolError(f"refusing to overwrite an existing main evaluation for task={task}")
    _claim_main_start(
        manifest_path.parent / "MAIN_EVALUATION_STARTED.json",
        run_id=run_id,
        manifest_sha256=manifest_sha256,
    )

    device = _resolve_device(args.device, args.require_cuda)
    candidate_root = _resolve(args.candidate_root)
    paths = {
        "a": _candidate_path(candidate_root, args.chain, "a", TARGET_FOLD),
        "b": _candidate_path(candidate_root, args.chain, "b1", TARGET_FOLD),
    }
    identities = {
        key: _read_identities(path, TARGET_FOLD) for key, path in paths.items()
    }
    combined = pd.concat([identities["a"], identities["b"]], ignore_index=True)
    a_rows = len(identities["a"])
    slices = {"a": slice(0, a_rows), "b1": slice(a_rows, len(combined)), "b2": slice(a_rows, len(combined))}
    task_source = {"a": "a", "b1": "b", "b2": "b"}
    U, triples, _ = _prepare_runtime(args.chain, TARGET_FOLD)
    scorer = _score_factory(args.family, triples, combined, U, device)
    cache_context = {
        "schema_version": "upgrade-bench-v2/score-cache-context/1",
        "protocol": PROTOCOL,
        "phase": "main-label-blind-scoring",
        "fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
        "chain": args.chain,
        "family": args.family,
        "run_id": run_id,
        "run_config_sha256": run_config_sha256,
        "manifest_sha256": manifest_sha256,
        "graph_sha256": _stable_triple_hash(triples),
        "combined_identity_sha256": _stable_frame_hash(combined),
        # No full candidate-file hash here: those bytes include target labels.
        "identity_sha256": {key: _stable_frame_hash(value) for key, value in identities.items()},
        "code_sha256": _code_hashes(),
    }
    cache_root = output_root / "score_cache" / "main" / args.chain / args.family
    config_by_task = {}
    for task in TRACKS:
        config_by_task[task] = (
            selected[task]["selected"]["model"],
            selected[task]["selected"]["hyperparameters"],
        )
    score_by_config_seed = {}
    representation_finished = []
    for task in TRACKS:
        model, hp = config_by_task[task]
        config_key = (model, json.dumps(hp, sort_keys=True, separators=(",", ":")))
        for seed in seeds:
            lookup = (*config_key, seed)
            if lookup in score_by_config_seed:
                continue
            print(f"[evaluate-chain] {args.chain} {args.family} model={model} hp={hp} seed={seed}", flush=True)
            scores, cache = _cached_score(
                scorer=scorer,
                model=model,
                hyperparameters=hp,
                seed=seed,
                cache_root=cache_root,
                cache_context=cache_context,
                expected_rows=len(combined),
            )
            score_by_config_seed[lookup] = scores
            representation_finished.append(
                {
                    "model": model,
                    "hyperparameters": hp,
                    "seed": seed,
                    "score_cache_key": cache["key"],
                    "score_cache_hit": cache["hit"],
                    "finished_at_utc": _utc_now(),
                }
            )

    # First target-label read, after every unique config/seed has been scored.
    labels = {key: _read_labels(paths[key], identities[key]) for key in ("a", "b")}
    target_labels_read_at = _utc_now()
    score_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    for task in TRACKS:
        source = task_source[task]
        model, hp = config_by_task[task]
        config_key = (model, json.dumps(hp, sort_keys=True, separators=(",", ":")))
        task_scores = [score_by_config_seed[(*config_key, seed)][slices[task]] for seed in seeds]
        metrics_by_seed = []
        uncertainty_by_seed = []
        for seed, scores in zip(seeds, task_scores):
            metrics_by_seed.append(
                {"seed": seed, **_ranking_metrics(task, identities[source], labels[source], scores)}
            )
            uncertainty_by_seed.append(
                {
                    "seed": seed,
                    **_cluster_bootstrap(
                        task,
                        identities[source],
                        labels[source],
                        scores,
                        iters=args.bootstrap_iters,
                        seed=args.bootstrap_seed + seed,
                    ),
                }
            )
        summary = _summarize_runs(
            [{key: value for key, value in row.items() if key != "seed"} for row in metrics_by_seed]
        )
        score_path = score_dir / f"scores_{args.chain}_track-{task}_{args.family}.csv"
        metric_path = metric_dir / f"metrics_{args.chain}_track-{task}_{args.family}.json"
        scores_out = identities[source].copy()
        for seed, scores in zip(seeds, task_scores):
            scores_out[f"score_{model}_s{seed}"] = scores
        scores_out["selection_sha256"] = freeze_entries[task]["sha256"]
        scores_out["protocol"] = PROTOCOL
        scores_out.to_csv(score_path, index=False)
        payload = {
            "schema_version": "upgrade-bench-v2/gpu-evaluation/1",
            "protocol": PROTOCOL,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "chain": args.chain,
            "track": task,
            "family": args.family,
            "selection_manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "selection_sha256": freeze_entries[task]["sha256"],
            "run_id": run_id,
            "run_config": str(run_config),
            "run_config_sha256": run_config_sha256,
            "selected": selected[task]["selected"],
            "target_fold": TARGET_FOLD,
            "aggregation": "calendar_mean",
            "cohort_policy": "complete main cohort; no same-window dev/test split",
            "orchestration": "chain_multitask_unique_config_training",
            "protocol_timestamps": {
                "full_freeze_gate_verified_at_utc": gate_verified_at,
                "representations_finished": representation_finished,
                "main_target_labels_read_at_utc": target_labels_read_at,
                "ordering_attestation": "all unique representation scores precede the target-label read",
            },
            "target_candidate": {
                "path": str(paths[source].relative_to(ROOT) if paths[source].is_relative_to(ROOT) else paths[source]),
                "sha256": sha256_file(paths[source]),
                "rows": int(len(identities[source])),
                "positive_lanes": int(labels[source]["y"].sum()),
            },
            "seeds": seeds,
            "per_seed": metrics_by_seed,
            "cluster_bootstrap_by_seed": uncertainty_by_seed,
            "summary": summary,
            "score_artifact": str(score_path),
        }
        write_json_atomic(metric_path, payload)
        print(f"strict rolling task={task}: {metric_path}")
    return 0


def _add_runtime_args(parser, *, include_track: bool = True) -> None:
    parser.add_argument("--chain", required=True, help="chain registry id")
    if include_track:
        parser.add_argument(
            "--track",
            choices=TRACKS,
            required=True,
            help="a=destination extension; b1=entry; b2=conditional destinations",
        )
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path("data/processed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("results_v2/gpu_rolling"))
    parser.add_argument("--run-config", type=Path, default=Path("configs/v2_gpu_rolling.json"))
    parser.add_argument("--run-id", default=None, help="must equal the immutable run_id in --run-config")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of silently running a heavy job on CPU")
    parser.add_argument("--dry-run", action="store_true", help="print the phase plan without opening data or training")
    parser.add_argument("--overwrite", action="store_true")


def _add_selection_grid_args(parser) -> None:
    parser.add_argument("--models", default=",".join(DEFAULT_KGE_MODELS))
    parser.add_argument("--dims", default="64,128")
    parser.add_argument("--learning-rates", default="0.005,0.01")
    parser.add_argument("--layers", default="4,6")
    parser.add_argument("--nbfnet-learning-rates", default="0.001,0.005")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="fixed training budget within a grid (default: KGE=150, NBFNet=25)",
    )
    parser.add_argument("--kge-batch-size", type=int, default=2048)
    parser.add_argument("--nbfnet-batch-size", type=int, default=64)
    parser.add_argument("--nbfnet-negatives", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split-salt", default="v2-history-0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select_chain = sub.add_parser(
        "select-chain",
        help="formal shared-training fold2 selection for A/B1/B2",
    )
    _add_runtime_args(select_chain, include_track=False)
    _add_selection_grid_args(select_chain)
    select_chain.set_defaults(func=_select_chain)

    freeze = sub.add_parser("freeze", help="hash-lock every expected fold2 selection")
    freeze.add_argument("--output-root", type=Path, default=Path("results_v2/gpu_rolling"))
    freeze.add_argument("--manifest", type=Path, default=None)
    freeze.add_argument("--chains", default=",".join(DEFAULT_CHAINS))
    freeze.add_argument("--tracks", default=",".join(TRACKS))
    freeze.add_argument("--families", default=",".join(FAMILIES))
    freeze.add_argument("--overwrite", action="store_true")
    freeze.set_defaults(func=_freeze)

    evaluate_chain = sub.add_parser(
        "evaluate-chain",
        help="formal shared-training complete-main evaluation for A/B1/B2",
    )
    _add_runtime_args(evaluate_chain, include_track=False)
    evaluate_chain.add_argument("--manifest", type=Path, required=True)
    evaluate_chain.add_argument("--seeds", default="0,1,2,3,4")
    evaluate_chain.add_argument(
        "--bootstrap-iters", type=int, default=FORMAL_BOOTSTRAP_ITERS
    )
    evaluate_chain.add_argument(
        "--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED
    )
    evaluate_chain.set_defaults(func=_evaluate_chain)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ProtocolError as exc:
        print(f"PROTOCOL ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
