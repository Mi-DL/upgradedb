"""Label-blind ULTRA feasibility smoke for the current UPGRADE-BENCH task.

This adapter is deliberately narrower than the formal rolling GPU runner.  It
scores one complete Track-A main cohort with one checkpoint fixed on the command
line.  It never reads ``y``, ``lateval``, or any other main-window outcome, and
its output is a non-claimable feasibility artifact under ``tmp/`` by default.

The implementation has two hard gates that the legacy ULTRA experiment lacked:

* candidate identities and scores must have exactly the same key set; and
* score alignment is an explicit indexed reorder, never an inner merge.

Use the formal historical-select / global-freeze / main-evaluate runner before
promoting any model comparison or paper-facing metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import types
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
ULTRA_ROOT = ROOT / "third_party" / "ULTRA"
DEFAULT_CHECKPOINT = ULTRA_ROOT / "ckpts" / "ultra_50g.pth"
DEFAULT_CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")

KEYS = ("i_iso", "j_iso", "stage")
IDENTITY_COLUMNS = KEYS + (
    "benchmark_version",
    "aggregation",
    "early_window",
    "late_window",
    "task",
    "task_unit",
)
FORBIDDEN_MAIN_COLUMNS = frozenset(
    {
        "y",
        "lateval",
        "size",
        "grav",
        "gnn",
        "log_exporter_capacity",
        "log_importer_demand",
    }
)
EXPECTED_METADATA = {
    "aggregation": "calendar_mean",
    "early_window": "2008-2012",
    "late_window": "2018-2022",
    "task": "destination_extension",
    "task_unit": "exporter_stage_destination",
}
SCHEMA_VERSION = "upgrade-bench-v2/ultra-feasibility-smoke/1"
CLAIM_STATUS = "FEASIBILITY_ONLY_NOT_A_PAPER_RESULT"


class UltraProtocolError(RuntimeError):
    """Raised when a leakage, identity, coverage, or runtime gate fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _portable_path(path: Path) -> str:
    """Avoid embedding a host home directory in a smoke manifest."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key_hash(frame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, list(KEYS)].itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_triple_hash(triples: Iterable[Sequence[object]]) -> str:
    digest = hashlib.sha256()
    for row in triples:
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_candidate_identities(path: Path):
    """Read only Track-A identity/protocol columns from a main candidate table.

    Passing an explicit ``usecols`` list is part of the leakage boundary.  Main
    labels and label-derived feature columns are not materialized in this
    process at any point.
    """
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise UltraProtocolError(f"candidate table is missing: {path}")
    requested = set(IDENTITY_COLUMNS)
    if requested & FORBIDDEN_MAIN_COLUMNS:
        raise AssertionError("identity reader requested a forbidden main column")
    try:
        frame = pd.read_csv(
            path,
            usecols=list(IDENTITY_COLUMNS),
            dtype={name: "string" for name in IDENTITY_COLUMNS},
        )
    except ValueError as exc:
        raise UltraProtocolError(f"candidate schema mismatch in {path}: {exc}") from exc
    if frame.empty:
        raise UltraProtocolError(f"candidate table is empty: {path}")
    if frame.isna().any().any():
        raise UltraProtocolError(f"candidate identity/protocol fields contain nulls: {path}")

    for name, expected in EXPECTED_METADATA.items():
        values = set(frame[name].astype(str).unique())
        if values != {expected}:
            raise UltraProtocolError(
                f"candidate {name} mismatch in {path}: {sorted(values)} != {[expected]}"
            )
    versions = set(frame["benchmark_version"].astype(str).unique())
    if len(versions) != 1 or not next(iter(versions)).startswith("2"):
        raise UltraProtocolError(f"candidate benchmark_version is not one consistent v2 value: {versions}")

    identities = frame.loc[:, list(KEYS)].astype(str)
    if identities.duplicated().any():
        raise UltraProtocolError(f"candidate keys are duplicated: {path}")
    ordered = identities.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    if not identities.reset_index(drop=True).equals(ordered):
        raise UltraProtocolError(f"candidate table is not in deterministic key order: {path}")
    metadata = {name: str(frame[name].iloc[0]) for name in IDENTITY_COLUMNS if name not in KEYS}
    return identities.reset_index(drop=True), metadata


def align_scores_exact(candidates, scored, score_column: str = "ultra_score"):
    """Return scores in candidate order, rejecting any non-bijective coverage.

    This intentionally uses exact indexed lookup rather than a relational merge.
    Missing, extra, duplicate, or non-finite scores are fatal.
    """
    import numpy as np
    import pandas as pd

    required = set(KEYS) | {score_column}
    missing_columns = required - set(scored.columns)
    if missing_columns:
        raise UltraProtocolError(f"score table is missing columns: {sorted(missing_columns)}")
    candidate_keys = candidates.loc[:, list(KEYS)].astype(str).reset_index(drop=True)
    score_frame = scored.loc[:, list(KEYS) + [score_column]].copy()
    score_frame.loc[:, list(KEYS)] = score_frame.loc[:, list(KEYS)].astype(str)
    if candidate_keys.isna().any().any() or candidate_keys.duplicated().any():
        raise UltraProtocolError("candidate keys must be non-null and unique before score alignment")
    if score_frame.loc[:, list(KEYS)].isna().any().any():
        raise UltraProtocolError("score keys contain nulls")
    if score_frame.loc[:, list(KEYS)].duplicated().any():
        raise UltraProtocolError("score keys are duplicated")

    candidate_index = pd.MultiIndex.from_frame(candidate_keys, names=list(KEYS))
    score_index = pd.MultiIndex.from_frame(score_frame.loc[:, list(KEYS)], names=list(KEYS))
    missing = candidate_index.difference(score_index)
    extra = score_index.difference(candidate_index)
    if len(missing) or len(extra):
        raise UltraProtocolError(
            f"score coverage is not exact: missing={len(missing)}, extra={len(extra)}"
        )

    indexed = score_frame.set_index(list(KEYS))[score_column]
    values = pd.to_numeric(indexed.loc[candidate_index], errors="coerce").to_numpy(dtype=float)
    if values.shape != (len(candidate_keys),) or not np.isfinite(values).all():
        raise UltraProtocolError("aligned score vector has wrong shape or non-finite values")
    result = candidate_keys.copy()
    result[score_column] = values
    if not result.loc[:, list(KEYS)].equals(candidate_keys):
        raise AssertionError("exact score alignment changed candidate order")
    return result


def load_current_early_graph(chain: str):
    """Load only the current main early graph (2008--2012, calendar mean)."""
    os.environ["VCU_FOLD"] = "main"
    os.environ["VCU_WINDOW_AGG"] = "calendar_mean"
    import universe as U
    from benchmark import setup_early_graph

    U.set_active_chain(chain)
    triples, _early = setup_early_graph(fold="main", aggregation="calendar_mean")
    return U, triples


def validate_graph_coverage(identities, triples) -> dict[str, object]:
    entities = {str(row[0]) for row in triples} | {str(row[2]) for row in triples}
    relations = {str(row[1]) for row in triples}
    heads = set(identities["i_iso"].astype(str))
    tails = set(identities["j_iso"].astype(str))
    query_relations = set(identities["stage"].astype(str))
    missing_heads = sorted(heads - entities)
    missing_tails = sorted(tails - entities)
    missing_relations = sorted(query_relations - relations)
    if missing_heads or missing_tails or missing_relations:
        raise UltraProtocolError(
            "candidate vocabulary is outside the early graph: "
            f"heads={len(missing_heads)}, tails={len(missing_tails)}, "
            f"relations={len(missing_relations)}"
        )
    return {
        "candidate_rows": int(len(identities)),
        "query_groups": int(identities.loc[:, ["i_iso", "stage"]].drop_duplicates().shape[0]),
        "graph_entities": int(len(entities)),
        "graph_forward_relations": int(len(relations)),
        "graph_forward_triples": int(len(triples)),
        "missing_heads": 0,
        "missing_tails": 0,
        "missing_relations": 0,
        "vocabulary_coverage": 1.0,
    }


def install_torch_scatter_compat() -> str:
    """Install the minimal ULTRA inference shim when torch-scatter is absent.

    ULTRA's transductive inference imports only ``scatter`` and ``scatter_add``
    along this path.  PyG's public scatter wrapper supplies the same reductions.
    The model is additionally forced onto PyG MessagePassing below, so the rspmm
    JIT extension is never imported.  This portable path is for the smoke only.
    """
    try:
        import torch_scatter  # noqa: F401

        return "native_torch_scatter"
    except ImportError:
        from torch_geometric.utils import scatter as pyg_scatter

        module = types.ModuleType("torch_scatter")

        def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
            result = pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
            if out is not None:
                out.copy_(result)
                return out
            return result

        def scatter_add(src, index, dim=-1, out=None, dim_size=None):
            return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")

        module.scatter = scatter
        module.scatter_add = scatter_add
        sys.modules["torch_scatter"] = module
        return "torch_geometric_utils_scatter_compat"


def build_ultra_graph(triples, export_relations: Sequence[str], device: str):
    import torch
    from torch_geometric.data import Data

    all_relations = {str(row[1]) for row in triples}
    relation_names = [name for name in export_relations if name in all_relations]
    relation_names += sorted(all_relations - set(relation_names))
    relations = {name: idx for idx, name in enumerate(relation_names)}
    entities = sorted({str(value) for row in triples for value in (row[0], row[2])})
    entity = {name: idx for idx, name in enumerate(entities)}

    heads = torch.tensor([entity[str(row[0])] for row in triples], dtype=torch.long)
    tails = torch.tensor([entity[str(row[2])] for row in triples], dtype=torch.long)
    rels = torch.tensor([relations[str(row[1])] for row in triples], dtype=torch.long)
    forward = torch.stack([heads, tails])
    data = Data(
        edge_index=torch.cat([forward, torch.stack([tails, heads])], dim=1),
        edge_type=torch.cat([rels, rels + len(relations)]),
        num_nodes=len(entities),
        num_relations=2 * len(relations),
    )

    # build_relation_graph allocates on the input device; CPU is the portable and
    # deterministic construction path, after which both graphs move together.
    from ultra import tasks as ultra_tasks

    ultra_tasks.build_relation_graph(data)
    data.edge_index = data.edge_index.to(device)
    data.edge_type = data.edge_type.to(device)
    data.relation_graph = data.relation_graph.to(device)
    return data, entity, relations


def load_fixed_ultra(checkpoint: Path, device: str):
    """Load exactly one predeclared checkpoint; no data-driven selection exists."""
    import torch
    from torch_geometric.nn.conv import MessagePassing
    from ultra.layers import GeneralizedRelationalConv
    from ultra.models import Ultra

    # Avoid rspmm compilation and use ULTRA's mathematically equivalent PyG
    # message()/aggregate() implementation for this portable feasibility run.
    GeneralizedRelationalConv.propagate = MessagePassing.propagate
    cfg = {
        "input_dim": 64,
        "hidden_dims": [64] * 6,
        "message_func": "distmult",
        "aggregate_func": "sum",
        "short_cut": True,
        "layer_norm": True,
    }
    model = Ultra(
        rel_model_cfg={"class": "RelNBFNet", **cfg},
        entity_model_cfg={"class": "EntityNBFNet", **cfg},
    )
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.1 compatibility
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, Mapping) or "model" not in state:
        raise UltraProtocolError(f"checkpoint does not contain a model state: {checkpoint}")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def score_complete_candidates(
    model,
    data,
    identities,
    entity: Mapping[str, int],
    relation: Mapping[str, int],
    *,
    device: str,
    batch_groups: int,
    progress: bool = True,
):
    """Score every candidate once, grouped by a shared ULTRA query."""
    import numpy as np
    import pandas as pd
    import torch

    if batch_groups < 1:
        raise UltraProtocolError("batch_groups must be positive")
    unknown = [
        tuple(row)
        for row in identities.loc[:, list(KEYS)].itertuples(index=False, name=None)
        if str(row[0]) not in entity or str(row[1]) not in entity or str(row[2]) not in relation
    ]
    if unknown:
        raise UltraProtocolError(f"{len(unknown)} candidates are outside the encoded graph vocabulary")

    groups: OrderedDict[tuple[str, str], list[tuple[int, str]]] = OrderedDict()
    for position, (head, tail, stage) in enumerate(
        identities.loc[:, list(KEYS)].itertuples(index=False, name=None)
    ):
        groups.setdefault((str(head), str(stage)), []).append((position, str(tail)))

    group_items = list(groups.items())
    scored_rows: list[tuple[str, str, str, float]] = []
    with torch.no_grad():
        for start in range(0, len(group_items), batch_groups):
            chunk = group_items[start : start + batch_groups]
            lengths = [len(rows) for _, rows in chunk]
            width = max(lengths)
            head_tensor = torch.tensor(
                [entity[head] for (head, _stage), _rows in chunk], dtype=torch.long, device=device
            )
            relation_tensor = torch.tensor(
                [relation[stage] for (_head, stage), _rows in chunk], dtype=torch.long, device=device
            )
            tail_tensor = torch.empty((len(chunk), width), dtype=torch.long, device=device)
            for row_index, ((_head, _stage), rows) in enumerate(chunk):
                encoded = torch.tensor([entity[tail] for _position, tail in rows], device=device)
                tail_tensor[row_index, : len(rows)] = encoded
                if len(rows) < width:
                    tail_tensor[row_index, len(rows) :] = encoded[0]
            batch = torch.stack(
                [
                    head_tensor[:, None].expand(-1, width),
                    tail_tensor,
                    relation_tensor[:, None].expand(-1, width),
                ],
                dim=-1,
            )
            output = model(data, batch).float().detach().cpu().numpy()
            for row_index, (((head, stage), rows), length) in enumerate(zip(chunk, lengths)):
                values = output[row_index, :length]
                for (_position, tail), value in zip(rows, values):
                    scored_rows.append((head, tail, stage, float(value)))
            if progress:
                completed = min(start + len(chunk), len(group_items))
                print(f"scored query groups {completed}/{len(group_items)}", flush=True)

    scored = pd.DataFrame(scored_rows, columns=list(KEYS) + ["ultra_score"])
    if len(scored) != len(identities) or not np.isfinite(scored["ultra_score"]).all():
        raise UltraProtocolError("ULTRA did not emit one finite score per candidate")
    return scored


def _runtime_versions() -> dict[str, object]:
    import numpy as np
    import pandas as pd
    import torch

    try:
        pyg = importlib.metadata.version("torch-geometric")
    except importlib.metadata.PackageNotFoundError:
        pyg = "unknown"
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": pyg,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    return result


def _write_json(path: Path, payload: Mapping[str, object], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise UltraProtocolError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise UltraProtocolError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _base_manifest(
    *,
    chain: str,
    candidate_path: Path,
    checkpoint: Path,
    identities,
    metadata: Mapping[str, str],
    triples,
    coverage: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_status": CLAIM_STATUS,
        "chain": chain,
        "track": "A",
        "fold": "main",
        "aggregation": "calendar_mean",
        "early_graph_window": "2008-2012",
        "target_window": "2018-2022",
        "candidate_metadata": dict(metadata),
        "candidate_source": {
            "path": _portable_path(candidate_path),
            "sha256": sha256_file(candidate_path),
            "identity_sha256": stable_key_hash(identities),
        },
        "graph_source": {
            "policy": "label_blind_current_main_early_graph_only",
            "sha256": stable_triple_hash(triples),
        },
        "checkpoint": {
            "path": _portable_path(checkpoint),
            "sha256": sha256_file(checkpoint),
            "selection_policy": "one_fixed_cli_checkpoint_no_data_driven_selection",
        },
        "main_target_labels_accessed": False,
        "main_label_derived_columns_accessed": False,
        "historical_or_main_labels_used_for_checkpoint_selection": False,
        "coverage": dict(coverage),
        "code_sha256": {
            "src/v2_ultra.py": sha256_file(Path(__file__).resolve()),
            "src/benchmark.py": sha256_file(SRC / "benchmark.py"),
            "src/temporal_backtest.py": sha256_file(SRC / "temporal_backtest.py"),
            "src/universe.py": sha256_file(SRC / "universe.py"),
            f"chains/{chain}.json": sha256_file(ROOT / "chains" / f"{chain}.json"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", choices=DEFAULT_CHAINS, default="sheep")
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=None,
        help="default: data/processed_v2/candidates_<chain>.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="one fixed ULTRA checkpoint; this adapter has no selection mode",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-groups", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate candidates, checkpoint, early graph, and 100%% vocabulary coverage without loading ULTRA",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _utc_now()
    started = time.perf_counter()
    candidate_path = _resolve(
        args.candidate_file or Path("data") / "processed_v2" / f"candidates_{args.chain}.csv"
    )
    checkpoint = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir or Path("tmp") / "v2_ultra_smoke" / args.chain)
    if not checkpoint.is_file():
        raise UltraProtocolError(f"fixed checkpoint is missing: {checkpoint}")

    identities, metadata = read_candidate_identities(candidate_path)
    U, triples = load_current_early_graph(args.chain)
    coverage = validate_graph_coverage(identities, triples)
    manifest = _base_manifest(
        chain=args.chain,
        candidate_path=candidate_path,
        checkpoint=checkpoint,
        identities=identities,
        metadata=metadata,
        triples=triples,
        coverage=coverage,
    )
    manifest["started_at_utc"] = started_at

    if args.dry_run:
        manifest.update(
            {
                "status": "dry_run_complete",
                "model_loaded": False,
                "gpu_scores_emitted": False,
                "requested_device": args.device,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "completed_at_utc": _utc_now(),
            }
        )
        out = output_dir / "dry_run_manifest.json"
        _write_json(out, manifest, args.overwrite)
        print(json.dumps({"manifest": str(out), "coverage": coverage}, indent=2))
        return 0

    import torch

    if not args.device.startswith("cuda"):
        raise UltraProtocolError("the complete smoke requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise UltraProtocolError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise UltraProtocolError(f"CUDA device does not exist: {args.device}")
    device_index = torch.cuda.current_device() if device.index is None else int(device.index)

    # Make the vendored ULTRA importable only after every label/coverage gate.
    sys.path.insert(0, str(ULTRA_ROOT))
    scatter_backend = install_torch_scatter_compat()
    # Materialize the CUDA allocator before resetting its counters.  The
    # Windows CUDA 13 build reports an invalid device when reset is the first
    # allocator operation in the process.
    torch.empty(1, device=device)
    # Use the resolved integer ordinal consistently for telemetry calls across
    # PyTorch versions.
    torch.cuda.reset_peak_memory_stats(device_index)
    model = load_fixed_ultra(checkpoint, str(device))
    data, entity, relation = build_ultra_graph(triples, U.EXPORT_RELATIONS, str(device))
    torch.cuda.synchronize(device)
    scoring_started = time.perf_counter()
    raw_scores = score_complete_candidates(
        model,
        data,
        identities,
        entity,
        relation,
        device=str(device),
        batch_groups=args.batch_groups,
    )
    torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - scoring_started
    aligned = align_scores_exact(identities, raw_scores)
    coverage.update(
        {
            "scored_rows": int(len(aligned)),
            "missing_score_keys": 0,
            "extra_score_keys": 0,
            "duplicate_score_keys": 0,
            "nonfinite_scores": 0,
            "exact_key_coverage": 1.0,
            "candidate_order_preserved": True,
        }
    )

    scores_path = output_dir / "scores.csv"
    manifest_path = output_dir / "manifest.json"
    if not args.overwrite and (scores_path.exists() or manifest_path.exists()):
        raise UltraProtocolError(f"refusing to overwrite complete smoke under {output_dir}")
    _write_csv(scores_path, aligned, args.overwrite)
    peak_bytes = int(torch.cuda.max_memory_allocated(device_index))
    manifest.update(
        {
            "status": "complete",
            "coverage": coverage,
            "model_loaded": True,
            "gpu_scores_emitted": True,
            "device": {
                "requested": args.device,
                "name": torch.cuda.get_device_name(device_index),
                "peak_memory_allocated_bytes": peak_bytes,
                "peak_memory_allocated_mib": round(peak_bytes / (1024**2), 3),
            },
            "runtime": _runtime_versions(),
            "ultra_runtime": {
                "scatter_backend": scatter_backend,
                "message_passing_backend": "torch_geometric_MessagePassing_fallback",
                "batch_groups": int(args.batch_groups),
                "scoring_seconds": round(scoring_seconds, 3),
            },
            "scores": {
                "path": _portable_path(scores_path),
                "sha256": sha256_file(scores_path),
                "column": "ultra_score",
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "completed_at_utc": _utc_now(),
        }
    )
    _write_json(manifest_path, manifest, args.overwrite)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "scores": str(scores_path),
                "coverage": coverage,
                "scoring_seconds": round(scoring_seconds, 3),
                "peak_memory_mib": round(peak_bytes / (1024**2), 3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
