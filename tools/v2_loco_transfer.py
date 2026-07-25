"""Strict, task-aligned leave-one-chain-out (LOCO) NBFNet runner.

The runner is deliberately separate from :mod:`src.loco_transfer`, whose old
full-cohort/within-decile evaluation is not comparable with the current A/B1/B2
benchmark.  Here a held-out chain's *early* graph is visible as inductive
inference context, while model fitting uses early graph edges from the other
five chains only.  Main-window outcome columns are opened only after complete
candidate scores have been materialised.

Two source-locked profiles are exposed. ``smoke-fixed-v1`` is a diagnostic
one-epoch run with a deterministic cap on supervised training edges;
``formal-fixed-v1`` has the historical NBFNet training budget and no cap.  No
command-line hyperparameter overrides are accepted, preventing target-label
tuning.  The smoke profile is never paper-eligible.

Examples (from the repository root)::

    python tools/v2_loco_transfer.py dry-run --holdout sheep
    python tools/v2_loco_transfer.py evaluate --holdout sheep \
        --profile smoke-fixed-v1 --seed 0 --device cuda

This file does not implement a long formal orchestration job.  A formal run
must use ``formal-fixed-v1`` (or a separately hash-frozen nested fold2 LOCO
selection artifact) and multiple seeds; the local command above is only a
protocol and runtime smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
NBFNET_ROOT = ROOT / "third_party" / "NBFNet-PyG"


def _set_canonical_sys_path() -> None:
    """Restore SRC > tools > vendor even after another importer mutates sys.path."""
    for path in (NBFNET_ROOT, TOOLS, SRC):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_set_canonical_sys_path()

_protocol_module = importlib.import_module("v2_gpu_protocol")
if Path(str(_protocol_module.__file__)).resolve() != (SRC / "v2_gpu_protocol.py").resolve():
    raise RuntimeError("v2_gpu_protocol was shadowed outside canonical src")
ProtocolError = _protocol_module.ProtocolError
sha256_file = _protocol_module.sha256_file
write_json_atomic = _protocol_module.write_json_atomic


CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
KEYS = ("i_iso", "j_iso", "stage")
FOLDS = ("fold2", "main")
PROTOCOL = "upgrade-bench-v2/loco-tier-matched/1"
COMPONENT_SCHEMA = "upgrade-bench-v2/loco-formal-component/1"
MODES = ("loco", "in_domain")
RAW_SOURCE_ATTESTATION = ROOT / "requirements" / "raw_source_attestation.json"

# These dictionaries are the only accepted hyperparameter sources.  In
# particular, evaluate has no --epochs/--layers/--lr knobs that could be tuned
# after inspecting the held-out main labels.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke-fixed-v1": {
        "formal_component_eligible": False,
        "purpose": "local protocol/runtime smoke only",
        "layers": 2,
        "learning_rate": 0.005,
        "epochs": 1,
        "batch_size": 64,
        "negatives": 8,
        "max_supervised_train_edges": 256,
        "query_batch_size": 8,
    },
    "formal-fixed-v1": {
        "formal_component_eligible": True,
        "purpose": "source-locked fixed configuration; no target-label selection",
        "layers": 6,
        "learning_rate": 0.005,
        "epochs": 25,
        "batch_size": 64,
        "negatives": 32,
        "max_supervised_train_edges": None,
        "query_batch_size": 8,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_module_origin(module: Any, expected_path: Path) -> Any:
    """Reject a dependency imported from outside its canonical repository file."""
    observed = getattr(module, "__file__", None)
    if not observed or Path(str(observed)).resolve() != expected_path.resolve():
        raise ProtocolError(
            f"module {getattr(module, '__name__', '<unknown>')} is shadowed: "
            f"observed={observed!r}, expected={str(expected_path.resolve())!r}"
        )
    return module


def _import_canonical_module(name: str, expected_path: Path):
    _set_canonical_sys_path()
    return _require_module_origin(importlib.import_module(name), expected_path)


def _require_resolved_module_path(name: str, expected_path: Path) -> None:
    """Check import resolution without executing a potentially heavy module."""
    spec = importlib.util.find_spec(name)
    observed = None if spec is None else spec.origin
    if not observed or Path(str(observed)).resolve() != expected_path.resolve():
        raise ProtocolError(
            f"module resolution for {name} is shadowed: observed={observed!r}, "
            f"expected={str(expected_path.resolve())!r}"
        )


def _require_external_module(name: str, *, optional: bool = False):
    _set_canonical_sys_path()
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    spec_origin = None if spec is None else spec.origin
    resolved_spec = None
    if spec_origin:
        resolved_spec = Path(str(spec_origin)).resolve()
        if not any(
            resolved_spec.is_relative_to(environment_root)
            for environment_root in {
                Path(sys.prefix).resolve(),
                Path(sys.base_prefix).resolve(),
            }
        ):
            raise ProtocolError(
                f"external module {name} resolves outside the configured Python environment: "
                f"{spec_origin}"
            )
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        if optional:
            return None
        raise
    observed = getattr(module, "__file__", None)
    if getattr(module, "__upgrade_bench_fallback__", False):
        if (
            name == "torch_scatter"
            and optional
            and isinstance(module, types.ModuleType)
            and observed is None
        ):
            return module
        raise ProtocolError(
            f"external module {name} uses an unauthorized compatibility-fallback marker"
        )
    if resolved_spec is None:
        raise ProtocolError(f"external module {name} has no importable environment spec")
    if not observed:
        raise ProtocolError(f"external module {name} has no import origin")
    resolved = Path(str(observed)).resolve()
    module_spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if (
        resolved != resolved_spec
        or not module_spec_origin
        or Path(str(module_spec_origin)).resolve() != resolved_spec
    ):
        raise ProtocolError(
            f"external module {name} import origin disagrees with its environment spec"
        )
    return module


def _validate_nbfnet_path_environment() -> None:
    """Reject an environment override that could redirect gap_discovery imports."""
    value = os.environ.get("NBFNET_PATH")
    if value is None:
        return
    if not value.strip():
        raise ProtocolError("NBFNET_PATH must be unset or name the canonical vendored NBFNet tree")
    observed = Path(value).expanduser().resolve()
    if observed != NBFNET_ROOT.resolve():
        raise ProtocolError(
            "NBFNET_PATH may not redirect formal execution: "
            f"observed={str(observed)!r}, expected={str(NBFNET_ROOT.resolve())!r}"
        )


def _validate_runtime_module_origins() -> dict[str, str]:
    _set_canonical_sys_path()
    _validate_nbfnet_path_environment()
    critical = {
        "v2_gpu_protocol": SRC / "v2_gpu_protocol.py",
        "v2_gpu_rolling": SRC / "v2_gpu_rolling.py",
        "gap_discovery": SRC / "gap_discovery.py",
        "benchmark": SRC / "benchmark.py",
        "temporal_backtest": SRC / "temporal_backtest.py",
        "universe": SRC / "universe.py",
        "baci_filtered_cache": SRC / "baci_filtered_cache.py",
        "v2_loco_formal": TOOLS / "v2_loco_formal.py",
        "nbfnet": NBFNET_ROOT / "nbfnet" / "__init__.py",
        "nbfnet.models": NBFNET_ROOT / "nbfnet" / "models.py",
        "nbfnet.layers": NBFNET_ROOT / "nbfnet" / "layers.py",
        "nbfnet.tasks": NBFNET_ROOT / "nbfnet" / "tasks.py",
    }
    origins: dict[str, str] = {}
    for name, expected in critical.items():
        _require_resolved_module_path(name, expected)
        loaded = sys.modules.get(name)
        if loaded is not None:
            _require_module_origin(loaded, expected)
        origins[name] = expected.relative_to(ROOT).as_posix()
    for name in ("torch", "numpy", "pandas", "sklearn", "torch_geometric"):
        module = _require_external_module(name)
        origins[name] = str(Path(str(module.__file__)).resolve())
    scatter = _require_external_module("torch_scatter", optional=True)
    origins["torch_scatter"] = (
        "PYG_COMPATIBILITY_FALLBACK"
        if scatter is None or getattr(scatter, "__upgrade_bench_fallback__", False)
        else str(Path(str(scatter.__file__)).resolve())
    )
    return origins


def _stable_rows_hash(rows: Iterable[Sequence[object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_path(candidate_root: Path, chain: str, track: str, fold: str) -> Path:
    if track not in {"a", "b"}:
        raise ProtocolError(f"unknown candidate track {track!r}")
    suffix = "_fold2" if fold == "fold2" else ""
    stem = "candidates" if track == "a" else "candidates_firsttime"
    return candidate_root / f"{stem}_{chain}{suffix}.csv"


def _expected_windows(fold: str) -> tuple[str, str]:
    if fold == "fold2":
        return "1998-2002", "2008-2012"
    if fold == "main":
        return "2008-2012", "2018-2022"
    raise ProtocolError(f"unsupported fold {fold!r}")


def _read_identities(path: Path, fold: str):
    """Read identity and metadata columns without opening outcome columns."""
    import pandas as pd

    early, late = _expected_windows(fold)
    usecols = list(KEYS) + ["aggregation", "early_window", "late_window"]
    frame = pd.read_csv(path, usecols=usecols, dtype={key: str for key in KEYS})
    if frame.empty:
        raise ProtocolError(f"candidate file is empty: {path}")
    metadata = frame.loc[:, ["aggregation", "early_window", "late_window"]].drop_duplicates()
    expected = pd.DataFrame(
        [{"aggregation": "calendar_mean", "early_window": early, "late_window": late}]
    )
    if not metadata.reset_index(drop=True).equals(expected):
        raise ProtocolError(f"candidate metadata/window mismatch: {path}")
    identities = frame.loc[:, list(KEYS)]
    if identities.isna().any().any() or identities.duplicated().any():
        raise ProtocolError(f"candidate keys are null or duplicated: {path}")
    ordered = identities.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    if not identities.reset_index(drop=True).equals(ordered):
        raise ProtocolError(f"candidate identities are not deterministically ordered: {path}")
    return identities.reset_index(drop=True)


def _read_labels_after_scoring(path: Path, expected_identities):
    """Open outcomes only after callers have persisted complete score files."""
    import pandas as pd

    cols = list(KEYS) + ["y", "size", "lateval"]
    frame = pd.read_csv(path, usecols=cols, dtype={key: str for key in KEYS})
    if not frame.loc[:, list(KEYS)].equals(expected_identities.reset_index(drop=True)):
        raise ProtocolError(f"outcome rows do not align with scored identities: {path}")
    if not set(frame["y"].dropna().unique()).issubset({0, 1}):
        raise ProtocolError(f"non-binary outcomes in {path}")
    return frame.loc[:, ["y", "size", "lateval"]]


def _relation_registry(U) -> tuple[dict[str, int], int]:
    registry_chains = tuple(sorted(name for name in U.CHAINS if name != "all"))
    if set(registry_chains) != set(CHAINS):
        raise ProtocolError(
            f"six-chain registry mismatch: runner={sorted(CHAINS)}, registry={list(registry_chains)}"
        )
    max_tier = max(max(U.CHAINS[chain].tiers().values()) for chain in CHAINS)
    names = [f"exp_tier{tier}" for tier in range(max_tier + 1)] + list(U.BG_RELATIONS)
    if len(names) != len(set(names)):
        raise ProtocolError("duplicate relation name in shared LOCO registry")
    return {name: index for index, name in enumerate(names)}, max_tier


def _stage_registry(U) -> dict[str, tuple[str, str, int]]:
    result: dict[str, tuple[str, str, int]] = {}
    for chain in CHAINS:
        tiers = U.CHAINS[chain].tiers()
        for stage, tier in tiers.items():
            result[f"{chain}.{stage}"] = (chain, stage, int(tier))
    return result


def _product_owners(U) -> dict[str, str]:
    owners: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}
    for chain in CHAINS:
        for product in U.CHAINS[chain].all_hs:
            if product in owners and owners[product] != chain:
                duplicates.setdefault(product, {owners[product]}).add(chain)
            owners[product] = chain
    if duplicates:
        examples = sorted((key, sorted(value)) for key, value in duplicates.items())[:5]
        raise ProtocolError(f"HS6 products are not chain-disjoint: {examples}")
    return owners


def _load_early_tables(fold: str):
    """Load the early window from the strict private cache, never the raw ZIP."""
    if fold not in FOLDS:
        raise ProtocolError(f"fold must be one of {FOLDS}")
    os.environ["VCU_FOLD"] = fold
    os.environ["VCU_WINDOW_AGG"] = "calendar_mean"
    import pandas as pd
    U = _import_canonical_module("universe", SRC / "universe.py")
    temporal = _import_canonical_module(
        "temporal_backtest", SRC / "temporal_backtest.py"
    )
    _import_canonical_module("benchmark", SRC / "benchmark.py")
    for local_name in (
        "baci_filtered_cache",
        "window_aggregation",
        "task_features",
        "split",
    ):
        _import_canonical_module(local_name, SRC / f"{local_name}.py")

    U.set_active_chain("all")
    early_years, _, _ = temporal.get_fold_spec(fold)
    cache, provenance = _validated_cache(requested_years=early_years)
    country_path = _resolve(Path(provenance["country_codes"]["path"]))
    countries = pd.read_csv(country_path)
    expected_columns = ["country_code", "country_name", "country_iso2", "country_iso3"]
    if list(countries.columns) != expected_columns:
        raise ProtocolError("country-code snapshot columns differ from the attested BACI member")
    if countries["country_code"].duplicated().any() or countries["country_iso3"].isna().any():
        raise ProtocolError("country-code snapshot contains duplicate codes or missing ISO3 values")
    iso = dict(zip(countries["country_code"].astype(int), countries["country_iso3"].astype(str)))
    early, early_hs6 = temporal.load_window(
        cache, iso, early_years, aggregation="calendar_mean"
    )
    early = early.copy()
    early_hs6 = early_hs6.copy()
    early["i_iso"] = early["i_iso"].astype(str)
    early["j_iso"] = early["j_iso"].astype(str)
    early["stage"] = early["stage"].astype(str)
    early_hs6["i_iso"] = early_hs6["i_iso"].astype(str)
    early_hs6["j_iso"] = early_hs6["j_iso"].astype(str)
    early_hs6["k"] = early_hs6["k"].astype(str).str.zfill(6)
    return U, early, early_hs6


def _stable_supervision_subset(
    export_triples: Sequence[tuple[str, str, str]], limit: int | None
) -> list[tuple[str, str, str]]:
    """Deterministically cap smoke supervision without changing graph context."""
    rows = list(export_triples)
    if limit is None or len(rows) <= int(limit):
        return rows
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256("\x1f".join(row).encode("utf-8")).hexdigest(),
            row,
        ),
    )
    return sorted(ranked[: int(limit)])


@dataclass
class GraphBundle:
    data: Any
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    provenance: dict[str, Any]
    source_namespaced_stages: tuple[str, ...] = ()
    registered_hs6: tuple[str, ...] = ()


def _build_graph(
    *,
    U,
    early,
    early_hs6,
    chains_subset: Sequence[str],
    relation_to_id: Mapping[str, int],
    device: str,
    max_supervised_edges: int | None = None,
) -> GraphBundle:
    """Build a tier-abstracted graph for exactly ``chains_subset``."""
    import torch
    from torch_geometric.data import Data

    subset = tuple(sorted(set(chains_subset)))
    unknown = sorted(set(subset) - set(CHAINS))
    if not subset or unknown:
        raise ProtocolError(f"invalid graph chain subset: subset={subset}, unknown={unknown}")

    stage_registry = _stage_registry(U)
    product_owners = _product_owners(U)
    unknown_stages = sorted(set(early["stage"]) - set(stage_registry))
    if unknown_stages:
        raise ProtocolError(f"pooled early graph has unregistered namespaced stages: {unknown_stages[:5]}")

    stage_rows = early[early["stage"].map(lambda value: stage_registry[value][0] in subset)].copy()
    stage_rows["tier_rel"] = stage_rows["stage"].map(
        lambda value: f"exp_tier{stage_registry[value][2]}"
    )
    export_triples = sorted(
        set(
            (str(row.i_iso), str(row.tier_rel), str(row.j_iso))
            for row in stage_rows.loc[:, ["i_iso", "tier_rel", "j_iso"]].itertuples(index=False)
        )
    )
    if not export_triples:
        raise ProtocolError(f"no early export triples for chains {subset}")

    products = sorted(
        product
        for chain in subset
        for product in U.CHAINS[chain].all_hs
    )
    product_set = set(products)
    parent_products = {product[:4] for product in products} | {product[:2] for product in products}
    background: set[tuple[str, str, str]] = set()

    for product in products:
        background.add((product, "hs_parent", product[:4]))
        background.add((product[:4], "hs_parent", product[:2]))
    for chain in subset:
        registry = U.CHAINS[chain]
        for source_stage, target_stage in registry.form_of:
            for source_hs in registry.stages[source_stage]:
                for target_hs in registry.stages[target_stage]:
                    background.add((source_hs, "form_of", target_hs))
        for target_stage, source_products in registry.derived_from.items():
            for target_hs in registry.stages[target_stage]:
                for source_hs in source_products:
                    if source_hs in product_set and target_hs in product_set:
                        background.add((source_hs, "derived_from", target_hs))
        for target_hs, source_products in registry.derived_from_hs.items():
            for source_hs in source_products:
                if source_hs in product_set and target_hs in product_set:
                    background.add((source_hs, "derived_from", target_hs))

    hs6_rows = early_hs6[early_hs6["k"].map(lambda value: product_owners.get(value) in subset)]
    # Match benchmark._build_labeled_triples exactly: only countries already in
    # the stage-level early graph enter supplies/demands.  HS6's lower $50
    # threshold must not silently expand the country vocabulary beyond the
    # stage graph's $100 threshold.
    countries = set(stage_rows["i_iso"].astype(str)) | set(stage_rows["j_iso"].astype(str))
    supplies = hs6_rows.loc[:, ["i_iso", "k"]].drop_duplicates()
    demands = hs6_rows.loc[:, ["j_iso", "k"]].drop_duplicates()
    supplies = supplies[supplies["i_iso"].astype(str).isin(countries)]
    demands = demands[demands["j_iso"].astype(str).isin(countries)]
    background.update((str(row.i_iso), "supplies", str(row.k)) for row in supplies.itertuples(index=False))
    background.update((str(row.j_iso), "demands", str(row.k)) for row in demands.itertuples(index=False))

    background_triples = sorted(background)
    forward_triples = sorted(set(export_triples) | set(background_triples))
    bad_relations = sorted({row[1] for row in forward_triples} - set(relation_to_id))
    if bad_relations:
        raise ProtocolError(f"graph has relations outside shared registry: {bad_relations}")

    entities = sorted({row[0] for row in forward_triples} | {row[2] for row in forward_triples})
    entity_to_id = {name: index for index, name in enumerate(entities)}
    heads = torch.tensor([entity_to_id[row[0]] for row in forward_triples], dtype=torch.long)
    tails = torch.tensor([entity_to_id[row[2]] for row in forward_triples], dtype=torch.long)
    rels = torch.tensor([relation_to_id[row[1]] for row in forward_triples], dtype=torch.long)
    forward_edge_index = torch.stack([heads, tails])
    n_forward_relations = len(relation_to_id)
    edge_index = torch.cat([forward_edge_index, torch.stack([tails, heads])], dim=1)
    edge_type = torch.cat([rels, rels + n_forward_relations])

    supervised_triples = _stable_supervision_subset(export_triples, max_supervised_edges)
    supervised_heads = torch.tensor(
        [entity_to_id[row[0]] for row in supervised_triples], dtype=torch.long
    )
    supervised_tails = torch.tensor(
        [entity_to_id[row[2]] for row in supervised_triples], dtype=torch.long
    )
    supervised_relations = torch.tensor(
        [relation_to_id[row[1]] for row in supervised_triples], dtype=torch.long
    )
    data = Data(
        edge_index=edge_index.to(device),
        edge_type=edge_type.to(device),
        num_nodes=len(entities),
        target_edge_index=torch.stack([supervised_heads, supervised_tails]).to(device),
        target_edge_type=supervised_relations.to(device),
        num_relations=2 * n_forward_relations,
    )
    provenance = {
        "chains": list(subset),
        "n_nodes": len(entities),
        "n_forward_edges": len(forward_triples),
        "n_bidirectional_edges": 2 * len(forward_triples),
        "n_export_context_edges": len(export_triples),
        "n_background_context_edges": len(background_triples),
        "n_supervised_train_edges": len(supervised_triples),
        "supervision_was_capped": len(supervised_triples) != len(export_triples),
        "forward_triples_sha256": _stable_rows_hash(forward_triples),
        "export_context_sha256": _stable_rows_hash(export_triples),
        "supervised_edges_sha256": _stable_rows_hash(supervised_triples),
        "entities_sha256": _stable_rows_hash((name,) for name in entities),
        "country_vocabulary_sha256": _stable_rows_hash((name,) for name in sorted(countries)),
        "country_vocabulary_size": len(countries),
    }
    # parent_products is computed explicitly so a missing parent construction is
    # visible in tests/audits rather than hidden in a generic entity count.
    provenance["registered_product_nodes"] = len(product_set | parent_products)
    source_stages = tuple(sorted(stage_rows["stage"].astype(str).unique()))
    registered_hs6 = tuple(products)
    provenance["source_namespaced_stages_sha256"] = _stable_rows_hash(
        (name,) for name in source_stages
    )
    provenance["registered_hs6_sha256"] = _stable_rows_hash(
        (name,) for name in registered_hs6
    )
    return GraphBundle(
        data,
        entity_to_id,
        dict(relation_to_id),
        provenance,
        source_stages,
        registered_hs6,
    )


def _candidate_union(a_identities, b_identities):
    import pandas as pd

    union = (
        pd.concat([a_identities, b_identities], ignore_index=True)
        .drop_duplicates(list(KEYS), keep="first")
        .sort_values(list(KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    return union


def _candidate_tier_relation(stage: str, chain: str, U) -> str:
    tiers = U.CHAINS[chain].tiers()
    if stage not in tiers:
        raise ProtocolError(f"candidate stage {stage!r} is not registered for {chain}")
    return f"exp_tier{int(tiers[stage])}"


def _coverage_audit(identities, *, chain: str, U, graph: GraphBundle) -> dict[str, Any]:
    unknown: list[tuple[str, str, str, str]] = []
    tier_relations: list[str] = []
    for i_iso, j_iso, stage in identities.loc[:, list(KEYS)].itertuples(index=False, name=None):
        relation = _candidate_tier_relation(str(stage), chain, U)
        tier_relations.append(relation)
        if (
            str(i_iso) not in graph.entity_to_id
            or str(j_iso) not in graph.entity_to_id
            or relation not in graph.relation_to_id
        ):
            unknown.append((str(i_iso), str(j_iso), str(stage), relation))
    if unknown:
        raise ProtocolError(
            f"candidate coverage is not 100%: missing={len(unknown)}/{len(identities)}, "
            f"examples={unknown[:5]}"
        )
    return {
        "n_candidates": int(len(identities)),
        "n_covered": int(len(identities)),
        "coverage_fraction": 1.0,
        "candidate_identity_sha256": _stable_rows_hash(
            identities.loc[:, list(KEYS)].itertuples(index=False, name=None)
        ),
        "tier_relations": sorted(set(tier_relations)),
    }


def _install_portable_scatter() -> dict[str, Any]:
    """Provide a recorded PyG fallback when torch-scatter has no local wheel."""
    try:
        import torch_scatter  # noqa: F401

        return {"backend": "torch_scatter", "propagate": "package_default"}
    except ModuleNotFoundError:
        import torch_geometric.utils

        module = types.ModuleType("torch_scatter")
        module.__upgrade_bench_fallback__ = True

        def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
            if out is not None:
                raise NotImplementedError("portable scatter fallback does not accept out=")
            return torch_geometric.utils.scatter(
                src, index, dim=dim, dim_size=dim_size, reduce=reduce
            )

        def scatter_add(src, index, dim=-1, out=None, dim_size=None):
            return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")

        module.scatter = scatter
        module.scatter_add = scatter_add
        sys.modules["torch_scatter"] = module

        # The third-party layer's fused path JIT-compiles a custom extension.
        # A portable local smoke uses PyG's ordinary message/aggregate path.
        from torch_geometric.nn.conv import MessagePassing

        layers_module = _import_canonical_module(
            "nbfnet.layers", NBFNET_ROOT / "nbfnet" / "layers.py"
        )
        GeneralizedRelationalConv = layers_module.GeneralizedRelationalConv

        GeneralizedRelationalConv.propagate = MessagePassing.propagate
        return {
            "backend": "torch_geometric.utils.scatter compatibility module",
            "propagate": "torch_geometric.nn.conv.MessagePassing.propagate",
        }


def _train_nbfnet(train_graph: GraphBundle, hp: Mapping[str, Any], *, seed: int, device: str):
    import numpy as np
    import torch

    _validate_nbfnet_path_environment()
    scatter_backend = _install_portable_scatter()
    gap_module = _import_canonical_module("gap_discovery", SRC / "gap_discovery.py")
    # gap_discovery mutates sys.path from NBFNET_PATH at import time.  Recheck
    # every NBFNet module it loads immediately, before constructing a model.
    for module_name, expected in {
        "nbfnet": NBFNET_ROOT / "nbfnet" / "__init__.py",
        "nbfnet.models": NBFNET_ROOT / "nbfnet" / "models.py",
        "nbfnet.layers": NBFNET_ROOT / "nbfnet" / "layers.py",
        "nbfnet.tasks": NBFNET_ROOT / "nbfnet" / "tasks.py",
    }.items():
        _import_canonical_module(module_name, expected)
    train_model = gap_module.train_model

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    model = train_model(
        train_graph.data,
        torch.device(device),
        epochs=int(hp["epochs"]),
        bs=int(hp["batch_size"]),
        neg=int(hp["negatives"]),
        lr=float(hp["learning_rate"]),
        num_rel=int(train_graph.data.num_relations),
        layers=int(hp["layers"]),
    )
    return model, scatter_backend


def _score_candidates(
    model,
    graph: GraphBundle,
    identities,
    *,
    chain: str,
    U,
    device: str,
    query_batch_size: int,
):
    """Score every candidate exactly once; no candidate filtering is permitted."""
    import numpy as np
    import torch
    tasks = _import_canonical_module("nbfnet.tasks", NBFNET_ROOT / "nbfnet" / "tasks.py")

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, (i_iso, j_iso, stage) in enumerate(
        identities.loc[:, list(KEYS)].itertuples(index=False, name=None)
    ):
        relation = _candidate_tier_relation(str(stage), chain, U)
        head = graph.entity_to_id[str(i_iso)]
        tail = graph.entity_to_id[str(j_iso)]
        relation_id = graph.relation_to_id[relation]
        groups.setdefault((head, relation_id), []).append((tail, index))

    ordered_groups = sorted(groups.items())
    scores = np.full(len(identities), np.nan, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ordered_groups), int(query_batch_size)):
            chunk = ordered_groups[start : start + int(query_batch_size)]
            queries = torch.tensor(
                [[head, head, relation] for (head, relation), _ in chunk],
                dtype=torch.long,
                device=device,
            )
            tail_batch, _ = tasks.all_negative(graph.data, queries)
            predictions = model(graph.data, tail_batch).detach().cpu().numpy()
            for row_index, (_, items) in enumerate(chunk):
                for tail, candidate_index in items:
                    scores[candidate_index] = float(predictions[row_index, tail])
    if not np.isfinite(scores).all():
        missing = np.flatnonzero(~np.isfinite(scores))
        raise ProtocolError(f"model failed to score {len(missing)} candidates")
    return scores, {
        "n_candidates": int(len(scores)),
        "n_finite_scores": int(np.isfinite(scores).sum()),
        "coverage_fraction": 1.0,
        "n_unique_exporter_tier_queries": len(groups),
        "query_batch_size": int(query_batch_size),
        "score_vector_sha256": hashlib.sha256(scores.astype("<f8").tobytes()).hexdigest(),
    }


def _align_union_scores(union_identities, union_scores, task_identities):
    import numpy as np
    import pandas as pd

    union_index = pd.MultiIndex.from_frame(union_identities.loc[:, list(KEYS)])
    task_index = pd.MultiIndex.from_frame(task_identities.loc[:, list(KEYS)])
    locations = union_index.get_indexer(task_index)
    if (locations < 0).any():
        raise ProtocolError("task candidate keys are missing from the scored union")
    result = np.asarray(union_scores, dtype=np.float64)[locations]
    if result.shape != (len(task_identities),) or not np.isfinite(result).all():
        raise ProtocolError("aligned task score vector is incomplete")
    return result


def _binding_value(value: str | None) -> str:
    return str(value) if value else "UNBOUND"


def _path_display(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_scores(
    path: Path,
    identities,
    scores,
    *,
    seed: int,
    profile: str,
    mode: str,
    run_id: str | None,
    config_sha256: str | None,
    freeze_sha256: str | None,
    formal_manifest_file_sha256: str | None,
    main_start_marker_file_sha256: str | None,
    main_start_marker_sha256: str | None,
) -> None:
    frame = identities.copy()
    frame[f"seed_{int(seed)}"] = scores
    frame["protocol"] = PROTOCOL
    frame["profile"] = profile
    frame["mode"] = mode
    frame["run_id"] = _binding_value(run_id)
    frame["config_sha256"] = _binding_value(config_sha256)
    frame["freeze_sha256"] = _binding_value(freeze_sha256)
    frame["formal_manifest_file_sha256"] = _binding_value(
        formal_manifest_file_sha256
    )
    frame["main_start_marker_file_sha256"] = _binding_value(
        main_start_marker_file_sha256
    )
    frame["main_start_marker_sha256"] = _binding_value(main_start_marker_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Persist enough decimal precision to reconstruct the exact float64 score
    # vector.  The component binds the in-memory vector hash before outcomes
    # are opened, so a verifier must be able to recover the same IEEE-754 bits
    # rather than a numerically close vector.
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _code_run_tag() -> str:
    """Bind default output names to the exact runner bytes."""
    return sha256_file(Path(__file__).resolve())[:12]


def _assert_outputs_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ProtocolError(f"refusing to overwrite existing LOCO artifacts: {existing}")


def _source_attestation() -> dict[str, Any]:
    """Read the source-only attestation without resolving or opening the raw ZIP."""
    if not RAW_SOURCE_ATTESTATION.is_file():
        raise ProtocolError(f"raw source attestation is missing: {RAW_SOURCE_ATTESTATION}")
    attestation = json.loads(RAW_SOURCE_ATTESTATION.read_text(encoding="utf-8"))
    if attestation.get("schema_version") != "upgrade-bench/raw-source-attestation/1":
        raise ProtocolError("unknown raw source attestation schema")
    expected = {
        "schema_version",
        "scope",
        "source",
        "country_codes",
        "cache_contract",
        "attestation",
    }
    if set(attestation) != expected:
        raise ProtocolError("raw source attestation contains fields outside the source-only schema")
    source = attestation.get("source")
    country = attestation.get("country_codes")
    cache_contract = attestation.get("cache_contract")
    if not isinstance(source, Mapping) or set(source) != {
        "archive_name", "repository_path", "size_bytes", "sha256"
    }:
        raise ProtocolError("raw source attestation source schema is not exact")
    if source.get("archive_name") != "BACI_HS92_V202401b.zip" or source.get(
        "repository_path"
    ) != "data/raw/BACI_HS92_V202401b.zip":
        raise ProtocolError("raw source attestation does not name the expected BACI release")
    expected_digest = str(source.get("sha256", ""))
    if len(expected_digest) != 64 or any(c not in "0123456789abcdef" for c in expected_digest):
        raise ProtocolError("raw source attestation lacks a valid SHA-256")
    if not isinstance(source.get("size_bytes"), int) or int(source["size_bytes"]) < 1:
        raise ProtocolError("raw source attestation lacks a valid archive size")
    if not isinstance(country, Mapping) or set(country) != {
        "member_name", "repository_path", "size_bytes", "sha256"
    }:
        raise ProtocolError("country-code attestation schema is not exact")
    if country.get("member_name") != "country_codes_V202401b.csv" or country.get(
        "repository_path"
    ) != "requirements/baci_country_codes_V202401b.csv":
        raise ProtocolError("country-code attestation does not name the repository snapshot")
    if not isinstance(cache_contract, Mapping) or set(cache_contract) != {
        "schema_version", "required_years", "source_fields"
    }:
        raise ProtocolError("cache contract attestation schema is not exact")
    return attestation


def _validated_cache(*, requested_years: Iterable[int]) -> tuple[Any, dict[str, Any]]:
    """Validate the complete cache and bind it to source-only repository metadata."""
    cache_module = _import_canonical_module(
        "baci_filtered_cache", SRC / "baci_filtered_cache.py"
    )
    BaciFilteredCache = cache_module.BaciFilteredCache
    REQUIRED_YEARS = cache_module.REQUIRED_YEARS
    SCHEMA_VERSION = cache_module.SCHEMA_VERSION

    cache_value = os.environ.get("VCU_BACI_CACHE", "").strip()
    if not cache_value:
        raise ProtocolError(
            "VCU_BACI_CACHE is required; the strict LOCO runner never opens the raw BACI ZIP"
        )
    cache = BaciFilteredCache(
        Path(cache_value), requested_years=requested_years, chains_dir=ROOT / "chains"
    )
    attestation = _source_attestation()
    manifest = cache.manifest
    source = manifest["source"]
    expected_source = attestation["source"]
    if (
        source["archive_name"] != expected_source["archive_name"]
        or source["archive_bytes"] != expected_source["size_bytes"]
        or source["archive_sha256"] != expected_source["sha256"]
    ):
        raise ProtocolError("cache source provenance differs from the source-only attestation")
    cache_contract = attestation["cache_contract"]
    if cache_contract["schema_version"] != SCHEMA_VERSION or cache_contract[
        "required_years"
    ] != list(REQUIRED_YEARS):
        raise ProtocolError("cache schema/years differ from the source-only attestation")
    if sorted(cache_contract["source_fields"]) != sorted(source):
        raise ProtocolError("cache source fields differ from the source-only attestation")

    country = attestation["country_codes"]
    country_path = _resolve(Path(str(country["repository_path"])))
    if country_path != (ROOT / "requirements" / "baci_country_codes_V202401b.csv").resolve():
        raise ProtocolError("country-code snapshot path is not canonical")
    if not country_path.is_file() or country_path.stat().st_size != country["size_bytes"]:
        raise ProtocolError("country-code snapshot is missing or has a different byte count")
    country_sha256 = sha256_file(country_path)
    cache_country = source["country_codes_member"]
    if (
        country_sha256 != country["sha256"]
        or cache_country["name"] != country["member_name"]
        or cache_country["bytes"] != country["size_bytes"]
        or cache_country["sha256"] != country_sha256
    ):
        raise ProtocolError("country-code snapshot differs from cache/source provenance")

    manifest_path = cache.cache_dir / "manifest.json"
    provenance = {
        "input_kind": "strict_private_filtered_cache",
        "cache_dir": str(cache.cache_dir),
        "cache_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "schema_version": manifest["schema_version"],
            "visibility": manifest["visibility"],
        },
        "cache_years": list(manifest["years"]),
        "cache_content_inventory_sha256": _stable_json_hash(
            {"files": manifest["files"], "totals": manifest["totals"]}
        ),
        "cache_registry_snapshot_sha256": _stable_json_hash(manifest["registry"]),
        "source": source,
        "source_attestation": {
            "path": _path_display(RAW_SOURCE_ATTESTATION),
            "sha256": sha256_file(RAW_SOURCE_ATTESTATION),
            "scope": str(attestation["scope"]),
        },
        "country_codes": {
            "path": _path_display(country_path),
            "size_bytes": country_path.stat().st_size,
            "sha256": country_sha256,
        },
        "raw_archive_opened_or_hashed": False,
        "verification": (
            "complete cache inventory/content hashes and current registry snapshot validated; "
            "archive identity inherited from cache manifest and source-only attestation"
        ),
    }
    return cache, provenance


def _raw_provenance() -> dict[str, Any]:
    """Return cache-bound provenance without opening or hashing the raw BACI ZIP."""
    _, provenance = _validated_cache(requested_years=())
    return {
        **provenance,
    }


def _code_provenance() -> dict[str, str]:
    local_paths = {
        TOOLS / "v2_loco_transfer.py",
        TOOLS / "v2_loco_formal.py",
        SRC / "gap_discovery.py",
        SRC / "benchmark.py",
        SRC / "temporal_backtest.py",
        SRC / "universe.py",
        SRC / "baci_filtered_cache.py",
        SRC / "window_aggregation.py",
        SRC / "task_features.py",
        SRC / "split.py",
        SRC / "v2_gpu_rolling.py",
        SRC / "v2_gpu_protocol.py",
    }
    source_suffixes = (".py", ".cpp", ".cu", ".h", ".cuh")
    binary_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES) + (
        ".so",
        ".pyd",
        ".dll",
        ".dylib",
        ".pyc",
    )
    vendored_paths = {
        path
        for path in NBFNET_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name.lower().endswith(source_suffixes + binary_suffixes)
    }
    paths = sorted(
        local_paths | vendored_paths,
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    if not paths or not any(path.name == "rspmm.py" for path in paths):
        raise ProtocolError("vendored NBFNet source inventory is missing rspmm.py")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ProtocolError(f"LOCO executable source files are missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in paths
    }


def _registry_provenance() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted((ROOT / "chains").glob("*.json"))
    }


def _dependency_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "ninja",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "NOT_INSTALLED"
    try:
        result["torch-scatter"] = importlib.metadata.version("torch-scatter")
    except importlib.metadata.PackageNotFoundError:
        result["torch-scatter"] = "PYG_COMPATIBILITY_FALLBACK"
    try:
        import torch

        result["torch_cuda_build"] = str(torch.version.cuda or "CPU_ONLY")
        result["torch_cxx11_abi"] = str(torch.compiled_with_cxx11_abi())
    except (ImportError, AttributeError):
        result["torch_cuda_build"] = "TORCH_NOT_IMPORTABLE"
        result["torch_cxx11_abi"] = "UNKNOWN"
    return result


def _validate_sha256(value: str | None, name: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    value = str(value or "").lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def _resolve_device(requested: str) -> str:
    torch = _require_external_module("torch")

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def _graph_contract() -> dict[str, Any]:
    return {
        "relation_resolution": (
            "chain stages mapped to shared processing tiers exp_tier0..exp_tier5 using directional "
            "derived_from, derived_from_hs, and form_of stage edges"
        ),
        "export_edge_multiplicity": "deduplicated after stage-to-tier mapping",
        "background_graph": "same registered HS hierarchy/form_of/derived_from/supplies/demands semantics",
        "country_vocabulary": "stage-level early graph countries, matching benchmark._build_labeled_triples",
        "target_graph_visibility": "target early graph visible as inference context in both modes",
        "matched_modes": {
            "loco": "parameters trained on other five chains only",
            "in_domain": "parameters trained on target chain early edges",
        },
        "permitted_transfer_drop": "loco versus in_domain under this exact tier/dedup graph contract",
        "forbidden_transfer_drop": (
            "never compare directly with the formal stage-relation graph; that mixes relation resolution, "
            "edge multiplicity, and vocabulary"
        ),
        "independent_protocol": True,
    }


def _formal_authorization(args) -> dict[str, str]:
    """Verify the canonical freeze, main marker, and global claim before training."""
    manifest_value = getattr(args, "formal_manifest", None)
    if manifest_value is None:
        raise ProtocolError("formal-fixed-v1 requires --formal-manifest")
    manifest_path = _resolve(Path(manifest_value))
    expected_manifest = (ROOT / "results_v2" / "loco_formal" / "frozen_manifest.json").resolve()
    if manifest_path != expected_manifest:
        raise ProtocolError(f"--formal-manifest must be the canonical path {expected_manifest}")
    formal = _import_canonical_module("v2_loco_formal", TOOLS / "v2_loco_formal.py")
    if manifest_path != Path(formal.CANONICAL_MANIFEST).resolve():
        raise ProtocolError("runner/formal canonical manifest paths disagree")
    manifest = formal.verify_freeze(manifest_path)
    marker = formal._verify_main_marker(manifest_path, manifest)
    exact_bindings = {
        "run_id": str(args.run_id),
        "config_sha256": str(args.config_sha256),
        "freeze_sha256": str(args.freeze_sha256),
    }
    for key, observed in exact_bindings.items():
        if manifest.get(key) != observed:
            raise ProtocolError(f"formal runner {key} differs from the canonical manifest")
    component_id = f"{args.holdout}|{args.mode}|seed{int(args.seed)}"
    expected_ids = {
        str(record.get("component_id")) for record in manifest["expected_components"]
    }
    if component_id not in expected_ids:
        raise ProtocolError(f"formal component {component_id} is outside the frozen matrix")
    marker_path = manifest_path.parent / "MAIN_EVALUATION_STARTED.json"
    return {
        "formal_manifest": _path_display(manifest_path),
        "formal_manifest_file_sha256": sha256_file(manifest_path),
        "main_start_marker": _path_display(marker_path),
        "main_start_marker_file_sha256": sha256_file(marker_path),
        "main_start_marker_sha256": str(marker["marker_sha256"]),
        "global_claim_file_sha256": str(marker["global_claim_file_sha256"]),
        "global_claim_sha256": str(marker["global_claim_sha256"]),
    }


def _revalidate_formal_authorization(args) -> None:
    if getattr(args, "profile", None) != "formal-fixed-v1":
        return
    current = _formal_authorization(args)
    if current != dict(getattr(args, "formal_authorization", {})):
        raise ProtocolError("formal freeze/main marker/global claim changed during execution")


def _validate_run_contract(args) -> None:
    if args.mode not in MODES:
        raise ProtocolError(f"mode must be one of {MODES}")
    formal = args.profile == "formal-fixed-v1"
    if formal and args.fold != "main":
        raise ProtocolError("formal-fixed-v1 components are main-fold only")
    if formal and not str(args.run_id or "").strip():
        raise ProtocolError("formal-fixed-v1 requires --run-id")
    args.config_sha256 = _validate_sha256(
        args.config_sha256, "--config-sha256", required=formal
    )
    args.freeze_sha256 = _validate_sha256(
        args.freeze_sha256, "--freeze-sha256", required=formal
    )
    if formal and args.component_output is None:
        raise ProtocolError("formal-fixed-v1 requires an explicit --component-output frozen path")
    if formal:
        output = _resolve(args.component_output)
        expected = ("components", args.holdout, args.mode, f"seed_{int(args.seed)}", "component.json")
        if tuple(output.parts[-5:]) != expected:
            raise ProtocolError(
                "formal component path must end with " + "/".join(expected)
            )
        args.formal_authorization = _formal_authorization(args)
    else:
        args.formal_authorization = {}


def _all_candidate_inputs(candidate_root: Path | str, fold: str):
    """Read only identity/window columns for the fixed six-chain A/B matrix."""
    candidate_root = _resolve(Path(candidate_root))
    paths: dict[str, dict[str, Path]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for chain in CHAINS:
        paths[chain] = {
            "A": _candidate_path(candidate_root, chain, "a", fold),
            "B": _candidate_path(candidate_root, chain, "b", fold),
        }
        identities[chain] = {
            task: _read_identities(path, fold) for task, path in paths[chain].items()
        }
    return candidate_root, paths, identities


def _frozen_input_snapshot_from_data(
    *, candidate_root: Path, fold: str, paths, identities
) -> dict[str, Any]:
    candidate_records = {
        chain: {
            task: {
                "path": _path_display(paths[chain][task]),
                "n_rows": int(len(identities[chain][task])),
                "identity_sha256": _stable_rows_hash(
                    identities[chain][task]
                    .loc[:, list(KEYS)]
                    .itertuples(index=False, name=None)
                ),
            }
            for task in ("A", "B")
        }
        for chain in CHAINS
    }
    return {
        "schema_version": "upgrade-bench-v2/loco-frozen-input-snapshot/1",
        "protocol": PROTOCOL,
        "fold": fold,
        "aggregation": "calendar_mean",
        "chains": list(CHAINS),
        "candidate_root": _path_display(candidate_root),
        "candidate_identities": candidate_records,
        "raw_baci": _raw_provenance(),
        "module_origins": _validate_runtime_module_origins(),
    }


def frozen_input_snapshot(
    candidate_root: Path | str = Path("data/processed_v2"), *, fold: str = "main"
) -> dict[str, Any]:
    """Return the outcome-blind cache/candidate snapshot used by formal freeze.

    Only candidate identity and fixed-window metadata columns are requested;
    ``y``, ``size``, and ``lateval`` are never opened.  The raw BACI archive is
    likewise never opened: :func:`_raw_provenance` validates the strict cache.
    """
    if fold not in FOLDS:
        raise ProtocolError(f"fold must be one of {FOLDS}")
    root, paths, identities = _all_candidate_inputs(candidate_root, fold)
    return _frozen_input_snapshot_from_data(
        candidate_root=root, fold=fold, paths=paths, identities=identities
    )


def _capture_input_snapshot(args):
    """Hash-lock every pre-score input without reading an outcome column."""
    candidate_root, all_paths, all_identities = _all_candidate_inputs(
        args.candidate_root, args.fold
    )
    frozen_inputs = _frozen_input_snapshot_from_data(
        candidate_root=candidate_root,
        fold=args.fold,
        paths=all_paths,
        identities=all_identities,
    )
    paths = all_paths[args.holdout]
    identities = all_identities[args.holdout]
    snapshot = {
        "runner_contract": {
            "protocol": PROTOCOL,
            "fold": args.fold,
            "holdout_chain": args.holdout,
            "mode": args.mode,
            "profile_name": args.profile,
            "seed": int(args.seed),
            "run_id": _binding_value(args.run_id),
            "config_sha256": _binding_value(args.config_sha256),
            "freeze_sha256": _binding_value(args.freeze_sha256),
            "candidate_root": _path_display(candidate_root),
        },
        "profile_sha256": _stable_json_hash(PROFILES[args.profile]),
        "graph_contract_sha256": _stable_json_hash(_graph_contract()),
        "frozen_input_snapshot": frozen_inputs,
        "frozen_input_snapshot_sha256": _stable_json_hash(frozen_inputs),
        "candidate_identities": frozen_inputs["candidate_identities"][args.holdout],
        "raw_baci": frozen_inputs["raw_baci"],
        "module_origins": frozen_inputs["module_origins"],
        "code_sha256": _code_provenance(),
        "chain_registry_sha256": _registry_provenance(),
        "dependency_versions": _dependency_versions(),
    }
    return snapshot, candidate_root, paths, identities


def _finalize_input_snapshot(args, start_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    _revalidate_formal_authorization(args)
    end_snapshot, _, _, _ = _capture_input_snapshot(args)
    start_sha = _stable_json_hash(start_snapshot)
    end_sha = _stable_json_hash(end_snapshot)
    if start_sha != end_sha or dict(start_snapshot) != end_snapshot:
        raise ProtocolError(
            "runner/dependencies/registry/candidate identities/raw source changed during execution: "
            f"start={start_sha}, end={end_sha}"
        )
    return {
        "start": dict(start_snapshot),
        "start_sha256": start_sha,
        "end": end_snapshot,
        "end_sha256": end_sha,
        "unchanged": True,
    }


def _base_run_payload(*, command: str, args, device: str, problem) -> dict[str, Any]:
    profile = dict(PROFILES[args.profile])
    snapshot = problem["input_snapshot_start"]
    authorization = dict(getattr(args, "formal_authorization", {}))
    return {
        "schema_version": COMPONENT_SCHEMA,
        "protocol": PROTOCOL,
        "created_at_utc": _utc_now(),
        "command": command,
        "fold": args.fold,
        "aggregation": "calendar_mean",
        "mode": args.mode,
        "run_id": _binding_value(args.run_id),
        "config_sha256": _binding_value(args.config_sha256),
        "freeze_sha256": _binding_value(args.freeze_sha256),
        "formal_manifest": authorization.get("formal_manifest", "UNBOUND"),
        "formal_manifest_file_sha256": authorization.get(
            "formal_manifest_file_sha256", "UNBOUND"
        ),
        "main_start_marker": authorization.get("main_start_marker", "UNBOUND"),
        "main_start_marker_file_sha256": authorization.get(
            "main_start_marker_file_sha256", "UNBOUND"
        ),
        "main_start_marker_sha256": authorization.get(
            "main_start_marker_sha256", "UNBOUND"
        ),
        "global_claim_file_sha256": authorization.get(
            "global_claim_file_sha256", "UNBOUND"
        ),
        "global_claim_sha256": authorization.get("global_claim_sha256", "UNBOUND"),
        "component_id": f"{args.holdout}|{args.mode}|seed{int(args.seed)}",
        "chain": args.holdout,
        "holdout_chain": args.holdout,
        "train_chains": list(problem["train_chains"]),
        "all_six_chains": list(CHAINS),
        "seed": int(args.seed),
        "device": device,
        "profile_name": args.profile,
        "profile": profile,
        "profile_sha256": snapshot["profile_sha256"],
        "main_outcomes_used_for_training_or_selection": False,
        "selection_policy": {
            "kind": "source_locked_fixed_profile",
            "command_line_hyperparameter_overrides_permitted": False,
            "formal_freeze_binding_required": args.profile == "formal-fixed-v1",
        },
        "inductive_boundary": {
            "target_early_graph_visible_for_message_passing": True,
            "target_chain_early_edges_used_for_parameter_training": args.mode == "in_domain",
            "target_main_outcomes_opened_only_after_complete_scores_are_persisted": command == "evaluate",
            "not_graph_free_cold_start": True,
        },
        "tasks": {
            "A": "candidate lane ranking",
            "B1": "entry label and score are max over candidate lanes",
            "B2": "within-positive-entry destination ranking; macro recall@3",
        },
        "tie_breaking": {
            "A": "descending score, then i_iso/stage/j_iso ascending",
            "B1": "descending max-lane score, then i_iso/stage ascending",
            "B2": "within entry descending score, then j_iso ascending",
        },
        "graph_contract": _graph_contract(),
        "graph_contract_sha256": snapshot["graph_contract_sha256"],
        "frozen_input_snapshot_sha256": snapshot["frozen_input_snapshot_sha256"],
        "shared_relation_registry": dict(problem["relation_to_id"]),
        "max_processing_tier": int(problem["max_tier"]),
        "shared_relation_registry_sha256": _stable_json_hash(problem["relation_to_id"]),
        "raw_baci": snapshot["raw_baci"],
        "module_origins": snapshot["module_origins"],
        "chain_registry_sha256": snapshot["chain_registry_sha256"],
        "code_sha256": snapshot["code_sha256"],
        "dependency_versions": snapshot["dependency_versions"],
    }


def _prepare_problem(args, *, device: str):
    _validate_run_contract(args)
    start_snapshot, candidate_root, paths, identities = _capture_input_snapshot(args)
    U, early, early_hs6 = _load_early_tables(args.fold)
    relation_to_id, max_tier = _relation_registry(U)
    union = _candidate_union(identities["A"], identities["B"])
    profile = PROFILES[args.profile]
    train_chains = (
        sorted(set(CHAINS) - {args.holdout}) if args.mode == "loco" else [args.holdout]
    )
    train_graph = _build_graph(
        U=U,
        early=early,
        early_hs6=early_hs6,
        chains_subset=train_chains,
        relation_to_id=relation_to_id,
        device=device,
        max_supervised_edges=profile["max_supervised_train_edges"],
    )
    inference_graph = _build_graph(
        U=U,
        early=early,
        early_hs6=early_hs6,
        chains_subset=[args.holdout],
        relation_to_id=relation_to_id,
        device=device,
        max_supervised_edges=None,
    )
    target_prefix = f"{args.holdout}."
    target_hs6 = set(U.CHAINS[args.holdout].all_hs)
    train_target_stages = sorted(
        stage for stage in train_graph.source_namespaced_stages if stage.startswith(target_prefix)
    )
    train_target_hs6 = sorted(target_hs6 & set(train_graph.registered_hs6))
    if args.mode == "loco" and (train_target_stages or train_target_hs6):
        raise ProtocolError("LOCO train graph contains held-out stage or HS6 registry content")
    if args.mode == "in_domain" and (
        not train_target_stages or set(train_graph.registered_hs6) != target_hs6
    ):
        raise ProtocolError("in_domain train graph is not exactly the target chain registry")
    coverage = {
        "A": _coverage_audit(identities["A"], chain=args.holdout, U=U, graph=inference_graph),
        "B": _coverage_audit(identities["B"], chain=args.holdout, U=U, graph=inference_graph),
        "union": _coverage_audit(union, chain=args.holdout, U=U, graph=inference_graph),
    }
    return {
        "U": U,
        "candidate_root": candidate_root,
        "relation_to_id": relation_to_id,
        "max_tier": max_tier,
        "paths": paths,
        "identities": identities,
        "union": union,
        "train_chains": train_chains,
        "train_graph": train_graph,
        "inference_graph": inference_graph,
        "coverage": coverage,
        "input_snapshot_start": start_snapshot,
        "exclusion_audit": {
            "mode": args.mode,
            "target_chain": args.holdout,
            "target_chain_absent_from_train_registry": args.mode == "loco",
            "target_namespaced_stages_in_train": train_target_stages,
            "target_hs6_in_train": train_target_hs6,
        },
    }


def run_dry(args) -> Path:
    device = _resolve_device(args.device)
    problem = _prepare_problem(args, device=device)
    immutability = _finalize_input_snapshot(args, problem["input_snapshot_start"])
    payload = _base_run_payload(command="dry-run", args=args, device=device, problem=problem)
    payload.update(
        {
            "status": "DRY_RUN_COMPLETE_LABELS_NOT_OPENED",
            "formal_component_eligible": False,
            "paper_eligible": False,
            "candidate_inputs": {
                task: {
                    "path": _path_display(path),
                    "identity_sha256": problem["coverage"][task]["candidate_identity_sha256"],
                    "full_file_sha256": "not computed before scoring because file also contains outcomes",
                }
                for task, path in problem["paths"].items()
            },
            "coverage": problem["coverage"],
            "train_graph": problem["train_graph"].provenance,
            "target_inference_graph": problem["inference_graph"].provenance,
            "target_exclusion_audit": problem["exclusion_audit"],
            "immutability_snapshot": immutability,
            "outcomes_read": False,
        }
    )
    output = _resolve(args.output) if args.output else (
        _resolve(args.output_root)
        / f"dry_run_{args.fold}_{args.holdout}_{args.mode}_{args.profile}_{_code_run_tag()}.json"
    )
    write_json_atomic(output, payload)
    print(f"dry-run complete; 100% A/B candidate coverage; outcomes not opened -> {output}")
    return output


def run_evaluate(args) -> Path:
    torch = _require_external_module("torch")

    rolling = _import_canonical_module("v2_gpu_rolling", SRC / "v2_gpu_rolling.py")
    _ranking_metrics = rolling._ranking_metrics

    output_root = _resolve(args.output_root)
    if args.component_output is not None:
        metrics_output = _resolve(args.component_output)
        score_paths = {
            "A": metrics_output.parent / "score_A.csv",
            "B": metrics_output.parent / "score_B.csv",
        }
    else:
        run_tag = (
            f"{args.fold}_{args.holdout}_{args.mode}_{args.profile}_seed{args.seed}_{_code_run_tag()}"
        )
        score_paths = {
            task: output_root / "scores" / f"{run_tag}_{task.lower()}.csv"
            for task in ("A", "B")
        }
        metrics_output = output_root / "metrics" / f"{run_tag}.json"
    _assert_outputs_absent([*score_paths.values(), metrics_output])

    device = _resolve_device(args.device)
    problem = _prepare_problem(args, device=device)
    profile = PROFILES[args.profile]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model, scatter_backend = _train_nbfnet(
        problem["train_graph"], profile, seed=args.seed, device=device
    )
    union_scores, scoring_audit = _score_candidates(
        model,
        problem["inference_graph"],
        problem["union"],
        chain=args.holdout,
        U=problem["U"],
        device=device,
        query_batch_size=int(profile["query_batch_size"]),
    )
    task_scores = {
        task: _align_union_scores(problem["union"], union_scores, identities)
        for task, identities in problem["identities"].items()
    }

    # This is the protocol gate: every identity and finite score is persisted
    # before any y/size/lateval column is opened below.
    _revalidate_formal_authorization(args)
    for task in ("A", "B"):
        authorization = dict(getattr(args, "formal_authorization", {}))
        _write_scores(
            score_paths[task],
            problem["identities"][task],
            task_scores[task],
            seed=args.seed,
            profile=args.profile,
            mode=args.mode,
            run_id=args.run_id,
            config_sha256=args.config_sha256,
            freeze_sha256=args.freeze_sha256,
            formal_manifest_file_sha256=authorization.get(
                "formal_manifest_file_sha256"
            ),
            main_start_marker_file_sha256=authorization.get(
                "main_start_marker_file_sha256"
            ),
            main_start_marker_sha256=authorization.get("main_start_marker_sha256"),
        )
    score_hashes_before_labels = {task: sha256_file(path) for task, path in score_paths.items()}
    candidate_hashes_before_labels = {
        task: sha256_file(path) for task, path in problem["paths"].items()
    }

    labels = {
        task: _read_labels_after_scoring(problem["paths"][task], problem["identities"][task])
        for task in ("A", "B")
    }
    metrics = {
        "A": _ranking_metrics("a", problem["identities"]["A"], labels["A"], task_scores["A"]),
        "B1": _ranking_metrics("b1", problem["identities"]["B"], labels["B"], task_scores["B"]),
        "B2": _ranking_metrics("b2", problem["identities"]["B"], labels["B"], task_scores["B"]),
    }
    candidate_hashes_end = {task: sha256_file(path) for task, path in problem["paths"].items()}
    if candidate_hashes_before_labels != candidate_hashes_end:
        raise ProtocolError("candidate file changed while outcomes were being evaluated")
    score_hashes_end = {task: sha256_file(path) for task, path in score_paths.items()}
    if score_hashes_before_labels != score_hashes_end:
        raise ProtocolError("score artifact changed after pre-outcome persistence")
    immutability = _finalize_input_snapshot(args, problem["input_snapshot_start"])
    payload = _base_run_payload(command="evaluate", args=args, device=device, problem=problem)
    formal_component = args.profile == "formal-fixed-v1"
    payload.update(
        {
            "status": (
                "FIXED_PROFILE_COMPONENT_COMPLETE" if formal_component else "SMOKE_COMPLETE"
            ),
            # A single chain/seed worker can never, by itself, support a paper
            # claim.  Formal eligibility requires an external six-chain,
            # multi-seed freeze/aggregation manifest.
            "paper_eligible": False,
            "formal_component_eligible": formal_component,
            "paper_eligibility_blocker": (
                "diagnostic smoke profile"
                if not formal_component
                else "requires complete six-chain x two-mode x multi-seed frozen manifest"
            ),
            "component_output": (
                "component.json" if formal_component else _path_display(metrics_output)
            ),
            "candidate_inputs": {
                task: {
                    "path": _path_display(path),
                    "identity_sha256": problem["coverage"][task]["candidate_identity_sha256"],
                    "full_file_sha256_computed_after_scoring": candidate_hashes_end[task],
                }
                for task, path in problem["paths"].items()
            },
            "coverage": problem["coverage"],
            "train_graph": problem["train_graph"].provenance,
            "target_inference_graph": problem["inference_graph"].provenance,
            "target_exclusion_audit": problem["exclusion_audit"],
            "scatter_runtime": scatter_backend,
            "scoring": scoring_audit,
            "score_artifacts": {
                task: {
                    "path": (
                        path.name if args.component_output is not None else _path_display(path)
                    ),
                    "sha256": score_hashes_before_labels[task],
                    "n_rows": int(len(problem["identities"][task])),
                }
                for task, path in score_paths.items()
            },
            "outcomes_read_after_complete_score_persistence": True,
            "immutability_snapshot": immutability,
            "metrics": metrics,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
            ),
        }
    )
    output = metrics_output
    write_json_atomic(output, payload)
    print(f"evaluation complete; A/B coverage=100%; A/B1/B2 metrics -> {output}")
    return output


def _artifact_path(value: str, component_path: Path) -> Path:
    """Resolve repository-relative inputs such as canonical candidate tables."""
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    repo_path = (ROOT / path).resolve()
    if repo_path.exists():
        return repo_path
    return (component_path.parent / path).resolve()


def _component_relative_artifact_path(value: str, component_path: Path) -> Path:
    """Resolve a formal component artifact only inside its component directory."""
    path = Path(str(value))
    if path.is_absolute() or not str(value).strip():
        raise ProtocolError("formal score artifact path must be non-empty and relative")
    component_dir = component_path.parent.resolve()
    resolved = (component_dir / path).resolve()
    if not resolved.is_relative_to(component_dir):
        raise ProtocolError("formal score artifact path escapes its component directory")
    return resolved


def _assert_nested_equal(expected: Any, actual: Any, label: str) -> None:
    import math

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise ProtocolError(
                f"{label} keys differ: expected={sorted(expected)}, actual={sorted(actual)}"
            )
        for key in expected:
            _assert_nested_equal(expected[key], actual[key], f"{label}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ProtocolError(f"{label} list lengths differ")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_nested_equal(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        left, right = float(expected), float(actual)
        if math.isnan(left) and math.isnan(right):
            return
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
            raise ProtocolError(f"{label} differs: expected={left}, actual={right}")
        return
    if expected != actual:
        raise ProtocolError(f"{label} differs: expected={expected!r}, actual={actual!r}")


def _assert_no_forbidden_selection_keys(payload: object, path: str = "component") -> None:
    forbidden = {"champion", "winner", "best", "selected_mode"}
    if isinstance(payload, Mapping):
        hits = forbidden & set(payload)
        if hits:
            raise ProtocolError(f"{path} contains forbidden post-hoc selection keys: {sorted(hits)}")
        for key, value in payload.items():
            _assert_no_forbidden_selection_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_no_forbidden_selection_keys(value, f"{path}[{index}]")


def _payload_binding(value: object) -> str | None:
    text = str(value)
    return None if text == "UNBOUND" else text


def verify_component(
    path: Path | str,
    *,
    require_formal: bool = False,
    expected_run_id: str | None = None,
    expected_config_sha256: str | None = None,
    expected_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail-closed verification of a single A/B1/B2 component artifact."""
    np = _require_external_module("numpy")
    pd = _require_external_module("pandas")

    rolling = _import_canonical_module("v2_gpu_rolling", SRC / "v2_gpu_rolling.py")
    _ranking_metrics = rolling._ranking_metrics

    component_path = _resolve(Path(path))
    if not component_path.is_file():
        raise ProtocolError(f"component is missing: {component_path}")
    try:
        payload = json.loads(component_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"component is invalid JSON: {component_path}") from exc
    if payload.get("schema_version") != COMPONENT_SCHEMA:
        raise ProtocolError("component schema mismatch")
    if payload.get("protocol") != PROTOCOL:
        raise ProtocolError("component protocol mismatch")
    if payload.get("command") != "evaluate":
        raise ProtocolError("verify accepts evaluate components only")
    if payload.get("mode") not in MODES or payload.get("holdout_chain") not in CHAINS:
        raise ProtocolError("component mode/chain is invalid")
    if payload.get("chain") != payload.get("holdout_chain"):
        raise ProtocolError("component chain aliases disagree")
    expected_component_id = (
        f"{payload.get('holdout_chain')}|{payload.get('mode')}|seed{payload.get('seed')}"
    )
    if payload.get("component_id") != expected_component_id:
        raise ProtocolError("component identity does not match chain/mode/seed")
    if payload.get("aggregation") != "calendar_mean":
        raise ProtocolError("component aggregation is not the fixed calendar mean")
    if payload.get("main_outcomes_used_for_training_or_selection") is not False:
        raise ProtocolError("component does not assert outcome-blind training/selection")
    if payload.get("paper_eligible") is not False:
        raise ProtocolError("a component must never be paper-eligible by itself")
    if payload.get("graph_contract") != _graph_contract():
        raise ProtocolError("component tier-matched graph contract changed")
    if payload.get("graph_contract_sha256") != _stable_json_hash(_graph_contract()):
        raise ProtocolError("component graph contract hash mismatch")
    _assert_no_forbidden_selection_keys(payload)

    profile_name = str(payload.get("profile_name"))
    if profile_name not in PROFILES or payload.get("profile") != PROFILES[profile_name]:
        raise ProtocolError("component profile is unknown or modified")
    if payload.get("profile_sha256") != _stable_json_hash(PROFILES[profile_name]):
        raise ProtocolError("component profile hash mismatch")
    formal_component = profile_name == "formal-fixed-v1"
    expected_status = (
        "FIXED_PROFILE_COMPONENT_COMPLETE" if formal_component else "SMOKE_COMPLETE"
    )
    if payload.get("status") != expected_status:
        raise ProtocolError("component completion status does not match the locked profile")
    if bool(payload.get("formal_component_eligible")) != formal_component:
        raise ProtocolError("formal component eligibility does not match the locked profile")
    if formal_component and payload.get("component_output") != "component.json":
        raise ProtocolError("formal component_output must be the canonical basename component.json")
    if require_formal and not formal_component:
        raise ProtocolError("formal verification rejects smoke components")

    bindings = {
        "run_id": expected_run_id,
        "config_sha256": expected_config_sha256,
        "freeze_sha256": expected_freeze_sha256,
    }
    for key, expected in bindings.items():
        if expected is not None and payload.get(key) != expected:
            raise ProtocolError(f"component {key} does not match the formal freeze")
    if require_formal and any(payload.get(key) in (None, "UNBOUND", "") for key in bindings):
        raise ProtocolError("formal component lacks run/config/freeze binding")

    candidate_inputs = payload.get("candidate_inputs", {})
    if set(candidate_inputs) != {"A", "B"}:
        raise ProtocolError("component candidate_inputs must contain exactly A and B")
    candidate_paths = {
        task: _artifact_path(candidate_inputs[task]["path"], component_path)
        for task in ("A", "B")
    }
    if candidate_paths["A"].parent != candidate_paths["B"].parent:
        raise ProtocolError("A/B candidate files do not share one candidate root")
    args = argparse.Namespace(
        command="evaluate",
        holdout=str(payload["holdout_chain"]),
        fold=str(payload["fold"]),
        mode=str(payload["mode"]),
        profile=profile_name,
        seed=int(payload["seed"]),
        device="cpu",
        candidate_root=candidate_paths["A"].parent,
        output_root=component_path.parent,
        output=None,
        component_output=component_path,
        run_id=_payload_binding(payload.get("run_id")),
        config_sha256=_payload_binding(payload.get("config_sha256")),
        freeze_sha256=_payload_binding(payload.get("freeze_sha256")),
        formal_manifest=_payload_binding(payload.get("formal_manifest")),
    )
    problem = _prepare_problem(args, device="cpu")
    current_snapshot = problem["input_snapshot_start"]
    recorded_immutability = payload.get("immutability_snapshot", {})
    if recorded_immutability.get("unchanged") is not True:
        raise ProtocolError("component lacks a successful start/end immutability check")
    _assert_nested_equal(recorded_immutability.get("start"), current_snapshot, "startup_snapshot")
    _assert_nested_equal(recorded_immutability.get("end"), current_snapshot, "ending_snapshot")
    snapshot_sha = _stable_json_hash(current_snapshot)
    if recorded_immutability.get("start_sha256") != snapshot_sha or recorded_immutability.get(
        "end_sha256"
    ) != snapshot_sha:
        raise ProtocolError("component snapshot hash mismatch")

    if payload.get("code_sha256") != current_snapshot["code_sha256"]:
        raise ProtocolError("component code hashes differ from current files")
    if payload.get("frozen_input_snapshot_sha256") != current_snapshot[
        "frozen_input_snapshot_sha256"
    ]:
        raise ProtocolError("component frozen input snapshot differs from current cache/candidates")
    if payload.get("chain_registry_sha256") != current_snapshot["chain_registry_sha256"]:
        raise ProtocolError("component registry hashes differ from current files")
    if payload.get("raw_baci") != current_snapshot["raw_baci"]:
        raise ProtocolError("component BACI provenance differs from the current strict cache")
    if payload.get("dependency_versions") != current_snapshot["dependency_versions"]:
        raise ProtocolError("component dependency versions differ from current runtime")
    if payload.get("module_origins") != current_snapshot["module_origins"]:
        raise ProtocolError("component canonical module origins differ from current runtime")
    if formal_component:
        for key, expected in args.formal_authorization.items():
            if payload.get(key) != expected:
                raise ProtocolError(f"component {key} differs from current main authorization")
    if payload.get("train_graph") != problem["train_graph"].provenance:
        raise ProtocolError("component training graph hash/provenance mismatch")
    if payload.get("target_inference_graph") != problem["inference_graph"].provenance:
        raise ProtocolError("component inference graph hash/provenance mismatch")
    if payload.get("target_exclusion_audit") != problem["exclusion_audit"]:
        raise ProtocolError("component target-chain exclusion audit mismatch")
    if payload.get("coverage") != problem["coverage"]:
        raise ProtocolError("component candidate coverage audit mismatch")

    for task in ("A", "B"):
        frozen_record = current_snapshot["frozen_input_snapshot"]["candidate_identities"][
            args.holdout
        ][task]
        frozen_candidate_path = _resolve(Path(frozen_record["path"]))
        if candidate_paths[task] != frozen_candidate_path:
            raise ProtocolError(f"{task} candidate path differs from the frozen input snapshot")
        expected_identity = problem["coverage"][task]["candidate_identity_sha256"]
        if (
            candidate_inputs[task].get("identity_sha256") != expected_identity
            or expected_identity != frozen_record["identity_sha256"]
        ):
            raise ProtocolError(f"{task} candidate identity hash mismatch")
        if candidate_inputs[task].get("full_file_sha256_computed_after_scoring") != sha256_file(
            candidate_paths[task]
        ):
            raise ProtocolError(f"{task} candidate full-file hash mismatch")

    score_artifacts = payload.get("score_artifacts", {})
    if set(score_artifacts) != {"A", "B"}:
        raise ProtocolError("component score_artifacts must contain exactly A and B")
    task_scores: dict[str, np.ndarray] = {}
    verifier_authorization = dict(getattr(args, "formal_authorization", {}))
    expected_binding_columns = {
        "protocol": PROTOCOL,
        "profile": profile_name,
        "mode": args.mode,
        "run_id": _binding_value(args.run_id),
        "config_sha256": _binding_value(args.config_sha256),
        "freeze_sha256": _binding_value(args.freeze_sha256),
        "formal_manifest_file_sha256": verifier_authorization.get(
            "formal_manifest_file_sha256", "UNBOUND"
        ),
        "main_start_marker_file_sha256": verifier_authorization.get(
            "main_start_marker_file_sha256", "UNBOUND"
        ),
        "main_start_marker_sha256": verifier_authorization.get(
            "main_start_marker_sha256", "UNBOUND"
        ),
    }
    for task in ("A", "B"):
        artifact = score_artifacts[task]
        if formal_component and Path(str(artifact.get("path", ""))).parts != (
            f"score_{task}.csv",
        ):
            raise ProtocolError(
                f"formal {task} score artifact path must be exactly score_{task}.csv"
            )
        score_path = (
            _component_relative_artifact_path(artifact["path"], component_path)
            if formal_component
            else _artifact_path(artifact["path"], component_path)
        )
        if not score_path.is_file() or sha256_file(score_path) != artifact.get("sha256"):
            raise ProtocolError(f"{task} score artifact hash mismatch")
        # Pandas' default C-parser float conversion is not guaranteed to
        # round-trip decimal text to the original float64 bits.  A one-ULP
        # change would make the pre-outcome score-vector hash unverifiable.
        score_column = f"seed_{args.seed}"
        frame = pd.read_csv(
            score_path,
            # Force the score column to remain floating point even when an
            # entire artifact happens to contain integer-looking decimals
            # such as ``-0``.  Otherwise dtype inference can erase signed-zero
            # bits before the bound vector hash is reconstructed.
            dtype={**{key: str for key in KEYS}, score_column: np.float64},
            float_precision="round_trip",
        )
        if int(artifact.get("n_rows", -1)) != len(frame) or len(frame) != len(
            problem["identities"][task]
        ):
            raise ProtocolError(f"{task} score artifact row count mismatch")
        if not frame.loc[:, list(KEYS)].equals(problem["identities"][task]):
            raise ProtocolError(f"{task} score keys do not match candidate identities")
        required_columns = set(KEYS) | {score_column} | set(expected_binding_columns)
        if not required_columns.issubset(frame.columns):
            raise ProtocolError(f"{task} score artifact lacks required provenance columns")
        for column, expected in expected_binding_columns.items():
            if set(frame[column].astype(str).unique()) != {expected}:
                raise ProtocolError(f"{task} score {column} binding mismatch")
        values = frame[score_column].to_numpy(dtype=np.float64)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ProtocolError(f"{task} score vector is incomplete or non-finite")
        task_scores[task] = values

    union_index = pd.MultiIndex.from_frame(problem["union"].loc[:, list(KEYS)])
    union_scores = np.full(len(problem["union"]), np.nan, dtype=np.float64)
    for task in ("A", "B"):
        task_index = pd.MultiIndex.from_frame(problem["identities"][task].loc[:, list(KEYS)])
        locations = union_index.get_indexer(task_index)
        existing = np.isfinite(union_scores[locations])
        if existing.any() and not np.array_equal(
            union_scores[locations][existing], task_scores[task][existing]
        ):
            raise ProtocolError("overlapping A/B candidates have inconsistent scores")
        union_scores[locations] = task_scores[task]
    if not np.isfinite(union_scores).all():
        raise ProtocolError("score artifacts do not cover the full A/B candidate union")
    score_vector_sha = hashlib.sha256(union_scores.astype("<f8").tobytes()).hexdigest()
    scoring = payload.get("scoring", {})
    if (
        scoring.get("score_vector_sha256") != score_vector_sha
        or int(scoring.get("n_candidates", -1)) != len(union_scores)
        or int(scoring.get("n_finite_scores", -1)) != len(union_scores)
        or float(scoring.get("coverage_fraction", -1)) != 1.0
    ):
        raise ProtocolError("component union scoring audit mismatch")

    labels = {
        task: _read_labels_after_scoring(candidate_paths[task], problem["identities"][task])
        for task in ("A", "B")
    }
    recomputed_metrics = {
        "A": _ranking_metrics("a", problem["identities"]["A"], labels["A"], task_scores["A"]),
        "B1": _ranking_metrics("b1", problem["identities"]["B"], labels["B"], task_scores["B"]),
        "B2": _ranking_metrics("b2", problem["identities"]["B"], labels["B"], task_scores["B"]),
    }
    _assert_nested_equal(payload.get("metrics"), recomputed_metrics, "metrics")
    end_check = _finalize_input_snapshot(args, problem["input_snapshot_start"])
    if end_check.get("unchanged") is not True:
        raise ProtocolError("verifier inputs changed during verification")

    return {
        "schema_version": "upgrade-bench-v2/loco-component-verification/1",
        "status": "VERIFIED",
        "component": _path_display(component_path),
        "component_sha256": sha256_file(component_path),
        "formal": formal_component,
        "mode": args.mode,
        "chain": args.holdout,
        "seed": args.seed,
        "run_id": payload.get("run_id"),
        "config_sha256": payload.get("config_sha256"),
        "freeze_sha256": payload.get("freeze_sha256"),
        "paper_eligible": False,
    }


def run_verify(args) -> dict[str, Any]:
    report = verify_component(
        args.component,
        require_formal=args.require_formal,
        expected_run_id=args.run_id,
        expected_config_sha256=args.config_sha256,
        expected_freeze_sha256=args.freeze_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_component_inputs(command_parser) -> None:
        command_parser.add_argument("--holdout", required=True, choices=CHAINS)
        command_parser.add_argument("--fold", default="main", choices=FOLDS)
        command_parser.add_argument("--mode", default="loco", choices=MODES)
        command_parser.add_argument(
            "--profile", default="smoke-fixed-v1", choices=tuple(PROFILES)
        )
        command_parser.add_argument("--seed", type=int, default=0)
        command_parser.add_argument("--device", default="auto")
        command_parser.add_argument(
            "--candidate-root", type=Path, default=Path("data/processed_v2")
        )
        command_parser.add_argument(
            "--output-root", type=Path, default=Path("results_v2/loco_smoke")
        )
        command_parser.add_argument("--run-id", default=None)
        command_parser.add_argument("--config-sha256", default=None)
        command_parser.add_argument("--freeze-sha256", default=None)
        command_parser.add_argument(
            "--formal-manifest",
            type=Path,
            default=None,
            help="canonical frozen manifest; mandatory for formal-fixed-v1 evaluate",
        )

    dry = commands.add_parser("dry-run", help="build/validate graphs and identities without outcomes")
    add_component_inputs(dry)
    dry.add_argument("--output", type=Path, default=None, help="dry-run JSON override")
    dry.set_defaults(component_output=None)

    evaluate = commands.add_parser("evaluate", help="train, persist full scores, then evaluate")
    add_component_inputs(evaluate)
    evaluate.add_argument(
        "--component-output",
        type=Path,
        default=None,
        help="exact component JSON path; required for formal-fixed-v1",
    )
    evaluate.set_defaults(output=None)

    verify = commands.add_parser("verify", help="fail-closed revalidation of one component")
    verify.add_argument("--component", type=Path, required=True)
    verify.add_argument("--require-formal", action="store_true")
    verify.add_argument("--run-id", default=None)
    verify.add_argument("--config-sha256", default=None)
    verify.add_argument("--freeze-sha256", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "dry-run":
        run_dry(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_verify(args)


if __name__ == "__main__":
    main()
