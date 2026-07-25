"""
Multi-chain value-chain universe (HS92) — the chain REGISTRY the KG and the upgrade
back-test are built from.

A *chain* is a value chain: upstream RAW stages an exporter already does, and downstream
PROCESSED stages it can upgrade INTO, linked by `derived_from` and directional `form_of`
(structural value-chain assumptions, not species/material-tagged trade facts). Each chain is one JSON file in
`<repo>/chains/*.json` (override dir via $VCU_CHAINS_DIR); add a chain = add a file.

The ACTIVE chain (env $VCU_CHAIN, default "sheep") is exposed through module-level globals
(STAGES, EXPORT_RELATIONS, DERIVED_FROM, DERIVED_FROM_HS, HS2STAGE, ALL_HS, stage_of, plus the
sheep named-source aliases MEAT_SRC/WOOL_SRC/...) so existing single-chain consumers keep
working unchanged; call set_active_chain("cocoa") to switch. The default registry follows the
strict, code-level HS92 evidence in ``chains/evidence/registry_evidence.json`` and is checked by
``tools/audit_chain_registry.py``. Historical registry-dependent artifacts are not compatible
with a registry change and must be rebuilt before scientific use.

New chain-driven accessors (use these instead of hard-coded sheep literals when generalizing
consumers): UPSTREAM (downstream stage -> required upstream stages), UPGRADE_STAGES, PRODUCES
(FAO item code -> HS6), FORM_OF (list of (from_stage, to_stage) pairs).
"""
import os
import json
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAINS_DIR = Path(os.environ.get("VCU_CHAINS_DIR", str(ROOT.parent / "chains")))

# background (non-export) relations — chain-agnostic, fixed order (REL indexing depends on it)
BG_RELATIONS = ["produces", "form_of", "derived_from", "hs_parent", "supplies", "demands"]


class Chain:
    """One value chain loaded from a registry JSON file."""

    def __init__(self, d):
        self.id = d["id"]
        self.description = d.get("description", "")
        self.stages = {k: list(v) for k, v in d["stages"].items()}
        self.upstream = list(d.get("upstream", []))                       # raw stages (not targets)
        self.upstream_map = {k: list(v) for k, v in d.get("upstream_map", {}).items()}
        self.derived_from = {k: list(v) for k, v in d.get("derived_from", {}).items()}
        self.derived_from_hs = {k: list(v) for k, v in d.get("derived_from_hs", {}).items()}
        self.produces = {int(k): v for k, v in d.get("produces", {}).items()}   # FAO item -> HS6
        self.form_of = [tuple(p) for p in d.get("form_of", [])]                  # (from_stage, to_stage)
        self.named_sources = {k: list(v) for k, v in d.get("named_sources", {}).items()}
        # derived
        self.export_relations = list(self.stages.keys())
        self.hs2stage = {hs: st for st, codes in self.stages.items() for hs in codes}
        self.all_hs = sorted(self.hs2stage)
        self.downstream = [s for s in self.stages if s not in self.upstream]

    def stage_of(self, hs6):
        return self.hs2stage.get(hs6)

    def tiers(self):
        """Stage -> processing tier (depth in the registry's directional stage DAG).

        ``derived_from``, ``derived_from_hs``, and ``form_of`` all encode a
        processing direction and therefore contribute direct upstream-stage
        edges. Raw/source stages are tier 0; every other stage is one plus the
        maximum tier of its direct sources. These tiers are the basis for the
        chain-agnostic ``exp_tier{k}`` relations used by LOCO transfer.
        """
        direct_sets = {
            st: set(self.hs2stage.get(h) for h in srcs if self.hs2stage.get(h))
            for st, srcs in self.derived_from.items()
        }
        # Per-HS relations carry stage-granularity distinctions (for example,
        # carded-wool yarn from raw wool versus combed-wool yarn from wool tops).
        # They must contribute to the stage DAG as well as to graph edges.
        for target_hs, srcs in self.derived_from_hs.items():
            target_stage = self.hs2stage.get(target_hs)
            if target_stage is not None:
                direct_sets.setdefault(target_stage, set()).update(
                    self.hs2stage[h] for h in srcs if h in self.hs2stage
                )
        # form_of is directional (source form -> processed form). It remains a
        # background graph relation, but its stage direction must also inform
        # LOCO's processing-tier abstraction.
        for source_stage, target_stage in self.form_of:
            if source_stage in self.stages and target_stage in self.stages:
                direct_sets.setdefault(target_stage, set()).add(source_stage)
        direct = {stage: sorted(sources) for stage, sources in direct_sets.items()}

        # Reject a malformed registry with a deterministic, actionable error
        # instead of recursing indefinitely while computing tiers. Validate the
        # complete stage graph even when a cycle happens to touch a declared raw
        # stage, whose tier is otherwise pinned to zero below.
        settled = set()
        active = []

        def validate_acyclic(stage):
            if stage in settled:
                return
            if stage in active:
                start = active.index(stage)
                cycle = active[start:] + [stage]
                raise ValueError(
                    f"{self.id}: cycle in processing-tier DAG: {' -> '.join(cycle)}"
                )
            active.append(stage)
            for source in direct.get(stage, ()):
                validate_acyclic(source)
            active.pop()
            settled.add(stage)

        for stage in self.stages:
            validate_acyclic(stage)

        cache = {}

        def t(st):
            if st in cache:
                return cache[st]
            cache[st] = 0 if (st in self.upstream or not direct.get(st)) else 1 + max(t(u) for u in direct[st])
            return cache[st]

        return {st: t(st) for st in self.stages}

    def __repr__(self):
        return f"Chain({self.id!r}, {len(self.stages)} stages, {len(self.all_hs)} HS6)"


def _load_chains():
    reg = {}
    for fp in sorted(glob.glob(str(CHAINS_DIR / "*.json"))):
        with open(fp) as f:
            d = json.load(f)
        reg[d["id"]] = Chain(d)
    if not reg:
        raise RuntimeError(f"no chain JSON files found in {CHAINS_DIR}")
    return reg


def _merge_chains(reg):
    """Synthetic POOLED chain 'all': union of every registry chain into one graph. Stage names are
    namespaced by chain (`<chain>.<stage>`) so collisions (e.g. aluminium vs nickel exp_unwrought)
    are avoided; HS6 codes are already disjoint across chains. Enables `--chain all` (one pooled
    graph) everywhere the active-chain globals are used. NOTE: relations stay chain-specific (no
    abstraction) -> cross-chain transfer rides only on shared country nodes + the HS hierarchy."""
    d = {"id": "all", "description": "pooled union of all registry chains (stage names namespaced by chain)",
         "stages": {}, "upstream": [], "upstream_map": {}, "derived_from": {}, "derived_from_hs": {},
         "produces": {}, "form_of": [], "named_sources": {}}
    for cid, c in reg.items():
        p = lambda s: f"{cid}.{s}"
        for st, codes in c.stages.items():
            d["stages"][p(st)] = list(codes)
        d["upstream"] += [p(s) for s in c.upstream]
        for st, ups in c.upstream_map.items():
            d["upstream_map"][p(st)] = [p(u) for u in ups]
        for st, srcs in c.derived_from.items():
            d["derived_from"][p(st)] = list(srcs)            # values are HS6 (disjoint) -> no rename
        d["derived_from_hs"].update({k: list(v) for k, v in c.derived_from_hs.items()})
        for k, v in c.produces.items():
            d["produces"][str(k)] = v
        d["form_of"] += [[p(a), p(b)] for a, b in c.form_of]
        for k, v in c.named_sources.items():
            d["named_sources"].setdefault(k, []).extend(v)
    return Chain(d)


CHAINS = _load_chains()
if "all" not in CHAINS:
    CHAINS["all"] = _merge_chains(CHAINS)


def get_chain(cid):
    return CHAINS[cid]


def set_active_chain(cid):
    """Bind the module-level globals to chain `cid` (the active single-chain view)."""
    global ACTIVE_CHAIN, STAGES, EXPORT_RELATIONS, DERIVED_FROM, DERIVED_FROM_HS, HS2STAGE, ALL_HS
    global UPSTREAM, UPGRADE_STAGES, PRODUCES, FORM_OF
    global MEAT_SRC, WOOL_SRC, SKIN_SRC, WOOLTOP_SRC, YARN_SRC, FABRIC_SRC
    c = CHAINS[cid]
    ACTIVE_CHAIN = cid
    STAGES = c.stages
    EXPORT_RELATIONS = c.export_relations
    DERIVED_FROM = c.derived_from
    DERIVED_FROM_HS = c.derived_from_hs
    HS2STAGE = c.hs2stage
    ALL_HS = c.all_hs
    UPSTREAM = c.upstream_map
    UPGRADE_STAGES = list(c.upstream_map)
    PRODUCES = c.produces
    FORM_OF = c.form_of
    ns = c.named_sources
    MEAT_SRC = ns.get("MEAT_SRC", [])
    WOOL_SRC = ns.get("WOOL_SRC", [])
    SKIN_SRC = ns.get("SKIN_SRC", [])
    WOOLTOP_SRC = ns.get("WOOLTOP_SRC", [])
    YARN_SRC = ns.get("YARN_SRC", [])
    FABRIC_SRC = ns.get("FABRIC_SRC", [])


def stage_of(hs6: str):
    """Stage of an HS6 code in the ACTIVE chain (None if not in this chain)."""
    return HS2STAGE.get(hs6)


# bind globals to the active chain at import (default: sheep => byte-identical legacy behaviour)
set_active_chain(os.environ.get("VCU_CHAIN", "sheep"))
