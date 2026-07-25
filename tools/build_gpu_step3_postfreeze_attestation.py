#!/usr/bin/env python3
"""Build and verify the GPU Step-3 post-freeze semantic attestation.

The formal GPU run was frozen against a 50-file Step-3 snapshot.  Later
registry wording clarifications changed four JSON artifacts at byte level: the
sheep display description; canonical-definition/rationale text in the registry
evidence and its generated audit view; and the B1 coverage receipt that carries
their hashes.  A separate LOCO-only correction made ``Chain.tiers()`` consume
directional ``form_of`` edges and reject cycles.  Prospective runner hardening
then bound formal CLI values to the immutable run config and removed two
single-task commands whose artifacts could not satisfy the shared-chain freeze
contract.  This verifier reverse-patches only those exact JSON leaves and
source snippets, then proves that all six machine-facing registry projections,
the run configuration, all 24 candidate files, and every coverage quantity
remained unchanged.  It does not claim that the frozen run used the later
runner gates.

The public artifact contains only repository-relative paths, hashes, and the
allowlisted semantic text.  It never records a host, account, or absolute path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "upgrade-bench-v2/gpu-step3-post-freeze-semantic-attestation/1"
RUN_ID = "v2.1-gpu-rolling-oa-full-20260716-r1"
ARTIFACT_ROLE = "chains/evidence/gpu_step3_postfreeze_semantic_attestation.json"
DEFAULT_OUTPUT = ROOT / ARTIFACT_ROLE
STEP3_MANIFEST_ROLE = (
    "results_v2/gpu_rolling/runs/v2.1-gpu-rolling-oa-full-20260716-r1/"
    "STEP3_SYNC_MANIFEST.sha256"
)
STEP3_MANIFEST_SHA256 = (
    "55bda1a886304ae06795f2254339ee868e8f5105b95590f2390d2786b6e5a348"
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class AttestationError(RuntimeError):
    """Raised when the frozen/current equivalence proof does not close."""


# Exact bytes of the externally recorded, immutable Step-3 hash list.  Keeping
# the list in executable source lets a public checkout reconstruct and hash the
# private run-specific manifest without publishing any private run product.
FROZEN_STEP3_MANIFEST = """4208a43ec676d5a0a6b209d95d1f171c273c8c6138817dc1189dc88f8d4affb4  chains/aluminium.json
38b38f994e2a202227e9c945ce5180ada7233b4ed3161788e03eb3f1902d64b4  chains/cocoa.json
e2f10638f2f6c99ebf5f8d117938fcd0de36d8eb89bbc2d9648da7024fd8c907  chains/cotton.json
1e10e0f15ef06fb404ad86d29ccef659e50512951fee27f2c5773eb62731ea12  chains/evidence/registry_evidence.json
ea45d8304e726595157b9f1d388bd5dabf6b8887949333ba8c3c428a4f1a3123  chains/nickel.json
4d3d9d930852f644e82917131d6377ffdf7a3467968c578e00b878a01710b752  chains/oilseed-soy.json
d06a7c5d2ef15eafca6d3403b72ceaf1e7d16d3f08b3670a2524dd9f9e71dcd4  chains/sheep.json
d7b7a94d1daa490a7d8127c4f1662a8e33a1f76501b54563532508ce501b7bd4  configs/v2_gpu_rolling.json
9c1f76f8e19f792f6bcc6d3b7f25a184236cd025aac238cfbe2f7fd17d0972df  data/processed_v2/candidates_aluminium.csv
6f8eb34a81e8f509a1126cc60c8a18b077b6f3920a32f383e591513cd2f9f649  data/processed_v2/candidates_aluminium_fold2.csv
bc888a0e93a5e6720361d2250960c198e4ccccbe302a7f3409df708304dec2ad  data/processed_v2/candidates_cocoa.csv
3242df52635d1857d3cdabd0284ea16c75bcef21f88823752e35704e7319c834  data/processed_v2/candidates_cocoa_fold2.csv
4e7b9724d34ab830a06cc356feaecb24db8f5fa9081401fe46c1cfbfe9a22373  data/processed_v2/candidates_cotton.csv
3334b9bd03803cae1b05c37f74380e97be46309a899a65a8ad9ce20f2afbbabe  data/processed_v2/candidates_cotton_fold2.csv
3022bd78b4f689c4235632cb3974fb22ca1021a48df94f8ecb00c433d2b4d5ed  data/processed_v2/candidates_firsttime_aluminium.csv
475b561b7c0eada0d4c60b38a60f1a4566979350b100c88035a3b55f591b5422  data/processed_v2/candidates_firsttime_aluminium_fold2.csv
d876f6646165dc4895f7cf27f370426c608e80b53c7b6ac46427e0969a1073e5  data/processed_v2/candidates_firsttime_cocoa.csv
1afd4735b8e880b80d8f732cd3a3963cb776e6022f6cb60ebe558a1267fe9e3d  data/processed_v2/candidates_firsttime_cocoa_fold2.csv
09cf7fb3d24e0b8a5241ed92d19c0531f3d1560d39e72ee64100d4a4fe70fa9d  data/processed_v2/candidates_firsttime_cotton.csv
3396df9922af8447cacdcd3994e5053fdee0fe2ee870c4ab8aba532ad5767a71  data/processed_v2/candidates_firsttime_cotton_fold2.csv
77b0e4a0aeb8e7839e6aee399c32137ac8d4f568f1afc27532efab8f9737e5a2  data/processed_v2/candidates_firsttime_nickel.csv
182d002e10bfd4bb20d8ff0f552aed2eef65163f399540b2e0b2b3b3ad53a1ec  data/processed_v2/candidates_firsttime_nickel_fold2.csv
acdccd4555c30b334ac731344ed7138995015b6cebc0f144776000eaf28a1dd7  data/processed_v2/candidates_firsttime_oilseed-soy.csv
c4d387c765d37cdf989719d99460a2433eac18f3b297c34bcc03344d5c770908  data/processed_v2/candidates_firsttime_oilseed-soy_fold2.csv
766ee136a8529a39b9302ee0fec75f5839884b920c5ae926c8b2134077a31bca  data/processed_v2/candidates_firsttime_sheep.csv
a8c29cad8557a74e796201edcc6c2c24490fe07a0fd79dd6572474e1dd386ffe  data/processed_v2/candidates_firsttime_sheep_fold2.csv
0ebad7822622a459de3367181b93c9c65ba6daa87e51743b107134e5003c516f  data/processed_v2/candidates_nickel.csv
16cf742cfd226819684cc2b9895a70ac8c449a1ff55ea27904fe0a534425d62f  data/processed_v2/candidates_nickel_fold2.csv
98ee13bb3538e1ef97e6d42df7b11297c0cf7a6fd52acc32e3a8f48b9e24d944  data/processed_v2/candidates_oilseed-soy.csv
6870445537ca404c4500e01efe5a3a68fcaeeb93821cee2058d741d51b81743a  data/processed_v2/candidates_oilseed-soy_fold2.csv
84a1b20792abdea98fc601d01c080d4e0f44743b97fabf5307b86d4182bf6674  data/processed_v2/candidates_sheep.csv
b10bba638f60fbac6f2b35992a13122affbb50fe785060ba1f20be96d00e540c  data/processed_v2/candidates_sheep_fold2.csv
46d694cf6a979b64cbc92a2e9b6f0b1e8af934d9696a155dbfc3f6bc70fdf5e5  docs/registry_audit.json
3c055602fdb282b7527748a90feabe32aeb6994deeb7a9fa8f2e5b9a6db1deff  jobs/v2_gpu_main_worker.sh
6437f6537b3a5621403f630d6e92e8eee25afc9d766771a2925dcc0976b384aa  jobs/v2_gpu_nohup_worker.sh
ffb54ee6c3aabc81d9d3b18ae8b93db6b7cb7d716b5bafdf4522021391cc52ea  requirements/v2-gpu-nodeps-lock.txt
b48d157d59043ef71f56f9cf79204123b2faadb15076d0aa6493e15204921d8a  results_v2/metrics/b1_candidate_coverage.json
9bb93976254589eeeda46633e599114316676bba3d830c3e6bd1ab619f705854  results_v2/metrics/raw_label_audit.json
d015035fae6375d5e322688ec861eedc7dbe46c8e765be7b58187d07752c9d97  src/baci_filtered_cache.py
effe7b8951b76c3835b77fbc981195e810cc9aa512c2512c1fb829e5cedc202f  src/benchmark.py
59bb18cb4072f3ed6851d8a8a7fc98af1fb6039cba4273e1fb0feea4e0f70f25  src/gap_discovery.py
c131f5f0b15760bb2382f1425649bd29f2a13fcd171e0e821579b1b2fd176341  src/split.py
9a6f100f14101d36681dc960b5cddbc13b2c892a7b48c6ea01f9a515669856a8  src/task_features.py
0164ba98f755c3e056a59011fa5cc5c0d253fd4a289a5b0cb5c0fbe1a492ab45  src/temporal_backtest.py
c8a1e41431cc94ff42bf5a41f227864a37eebe43a17e109aea8e7de52f3bd374  src/universe.py
2c594a4cd14c3851b01bc931e8f776d1adadc4c77b17b581eab6aabac72ca818  src/v2_gpu_protocol.py
c821c4027b199c2a115ba6abe9dfd2361bdd70a61cf812e75d014c7e786b6645  src/v2_gpu_rolling.py
3050f7770e9a7695ad22c74bf092f26edc22f6163ae44861e4e9e87cc9a39f15  src/window_aggregation.py
11edd63530ba5279656d368769983894be77a0f44b4728c8f2b6ee43b938e31a  tools/step3_sync_manifest.py
971bacfe1b95887740e1fc5d53d2a077f33a82b05fcb8164e62e2efd5e63b93a  tools/v2_gpu_env_check.py
"""

OLD_DESCRIPTION = (
    "Strict sheep-meat and wool chain: live sheep -> sheep meat; raw wool -> "
    "wool grease, wool tops/combed wool, and wool yarn. Raw sheep/lamb skins "
    "are retained as a source stage. Mixed-species and mixed-material HS92 "
    "codes are excluded."
)
NEW_DESCRIPTION = (
    "Strict sheep-meat and wool chain: live sheep -> sheep meat; raw wool -> "
    "wool grease, wool tops/combed wool, and wool yarn. Raw sheep/lamb skins "
    "are retained as a source stage. Mixed-species and unresolved-material "
    "HS92 codes are excluded; commodity-explicit blends remain eligible under "
    "the published HS6-basket rule."
)

FIT_SUFFIX = (
    ", which is inside the canonical stage definition. No unobserved product form, "
    "feedstock, species, or end use is inferred from the commodity name alone."
)


def _stage_text_change(
    frozen_definition: str,
    current_definition: str,
    frozen_observed_scope: str,
    current_observed_scope: str,
) -> dict[str, str]:
    return {
        "frozen_definition": frozen_definition,
        "current_definition": current_definition,
        "frozen_fit_rule": (
            f"The official HS92 description identifies {frozen_observed_scope}{FIT_SUFFIX}"
        ),
        "current_fit_rule": (
            f"The official HS92 description identifies {current_observed_scope}{FIT_SUFFIX}"
        ),
    }


# These are wording corrections only.  The dynamic leaf allowlist below applies
# them to the generated stage-definition record and to every included decision
# already assigned to that stage.  Decision/status/stage/code fields are never
# allowlisted.
SEMANTIC_STAGE_TEXT_CHANGES: dict[tuple[str, str], dict[str, str]] = {
    ("sheep", "exp_woolgrease"): _stage_text_change(
        "Wool grease other than crude and fatty substances derived from wool grease, "
        "including lanolin.",
        "Non-crude wool grease and fatty substances derived from wool grease, including "
        "lanolin. This form-limited stage deliberately excludes crude wool grease, which "
        "is recorded out of stage.",
        "wool grease or fatty substances derived from it",
        "non-crude wool grease or fatty substances derived from it",
    ),
    ("sheep", "exp_wooltop"): _stage_text_change(
        "Wool tops and other combed wool, in fragments or otherwise; this stage excludes yarn.",
        "Wool tops and other combed wool, in fragments or otherwise; this form-limited "
        "stage excludes carded-wool fibre and yarn.",
        "wool tops or other combed wool, not yarn",
        "wool tops or other combed wool, not yarn",
    ),
    ("cotton", "exp_cottonyarn"): _stage_text_change(
        "Cotton yarn other than sewing thread.",
        "Non-sewing yarn explicitly containing or made of cotton, including "
        "cross-material blends.",
        "cotton yarn other than sewing thread",
        "non-sewing yarn explicitly containing cotton, including cross-material blends",
    ),
    ("cotton", "exp_cottonapparel_woven"): _stage_text_change(
        "Selected cotton apparel that is not knitted or crocheted (the registry's "
        "non-knit/woven apparel umbrella).",
        "Selected non-knit cotton apparel: men's or boys' shirts; men's or boys' "
        "trousers, bib and brace overalls, breeches, and shorts; and the corresponding "
        "women's or girls' lower-body garments. This form-limited stage does not represent "
        "all Chapter 62 cotton apparel.",
        "cotton apparel explicitly described as not knitted or crocheted",
        "one of the declared non-knit cotton-apparel forms",
    ),
    ("cotton", "exp_cottonapparel_knit"): _stage_text_change(
        "Selected cotton apparel that is knitted or crocheted.",
        "Selected knitted or crocheted cotton apparel: men's or boys' shirts; T-shirts, "
        "singlets, and other vests; men's or boys' and women's or girls' trousers and "
        "related lower-body garments; and jerseys, pullovers, cardigans, waistcoats, and "
        "similar articles. This form-limited stage does not represent all Chapter 61 "
        "cotton apparel.",
        "cotton apparel explicitly described as knitted or crocheted",
        "one of the declared knitted or crocheted cotton-apparel forms",
    ),
    ("cotton", "exp_cottonhomewares"): _stage_text_change(
        "Selected cotton bed, kitchen, and toilet linens used as the registry's "
        "home-textile umbrella.",
        "Cotton bed linen and cotton kitchen or toilet linen. This form-limited stage "
        "does not represent all cotton household articles.",
        "bed, kitchen, or toilet linen explicitly of cotton",
        "bed, kitchen, or toilet linen explicitly of cotton",
    ),
    ("nickel", "exp_unwrought"): _stage_text_change(
        "Unwrought nickel, alloyed or not.",
        "Nickel-explicit unwrought metal and alloy baskets: unwrought nickel and "
        "explicitly named copper-nickel or copper-nickel-zinc alloys.",
        "unwrought nickel",
        "unwrought nickel or a nickel-explicit copper-nickel alloy basket",
    ),
    ("nickel", "exp_bars_wire"): _stage_text_change(
        "Nickel bars, rods, profiles, and wire.",
        "Nickel-explicit bars, rods, profiles, and wire: nickel products and explicitly "
        "named copper-nickel or copper-nickel-zinc alloy baskets in those forms.",
        "nickel bars, rods, profiles, or wire",
        "nickel-explicit bars, rods, profiles, or wire, including copper-nickel alloy baskets",
    ),
    ("nickel", "exp_plates_foil"): _stage_text_change(
        "Nickel plates, sheets, strip, and foil.",
        "Nickel-explicit plates, sheets, strip, and foil: nickel products and explicitly "
        "named copper-nickel or copper-nickel-zinc alloy baskets in those forms.",
        "nickel plates, sheets, strip, or foil",
        "nickel-explicit plates, sheets, strip, or foil, including copper-nickel alloy baskets",
    ),
    ("nickel", "exp_tubes"): _stage_text_change(
        "Nickel tubes, pipes, and tube/pipe fittings.",
        "Nickel-explicit tubes, pipes, and fittings: nickel products and explicitly named "
        "copper-nickel or copper-nickel-zinc alloy baskets in those forms.",
        "nickel tubes, pipes, or fittings",
        "nickel-explicit tubes, pipes, or fittings, including copper-nickel alloy baskets",
    ),
}

FROZEN_UNIVERSE_OVERVIEW = (
    "A *chain* is a value chain: upstream RAW stages an exporter already does, and downstream\n"
    "PROCESSED stages it can upgrade INTO, linked by `derived_from` (a structural value-chain\n"
    "assumption, not a species/material-tagged trade fact). Each chain is one JSON file in\n"
)
CURRENT_UNIVERSE_OVERVIEW = (
    "A *chain* is a value chain: upstream RAW stages an exporter already does, and downstream\n"
    "PROCESSED stages it can upgrade INTO, linked by `derived_from` and directional `form_of`\n"
    "(structural value-chain assumptions, not species/material-tagged trade facts). Each chain is one JSON file in\n"
)
FROZEN_TIERS_DOCSTRING = (
    "        \"\"\"stage -> processing TIER (depth from raw in the derived_from DAG). Raw stages = 0;\n"
    "        tier(s) = 1 + max(tier of its direct upstream stages). Basis for relation abstraction\n"
    "        (the chain-agnostic exp_tier{k} relations used by the LOCO transfer task).\"\"\"\n"
)
CURRENT_TIERS_DOCSTRING = (
    "        \"\"\"Stage -> processing tier (depth in the registry's directional stage DAG).\n"
    "\n"
    "        ``derived_from``, ``derived_from_hs``, and ``form_of`` all encode a\n"
    "        processing direction and therefore contribute direct upstream-stage\n"
    "        edges. Raw/source stages are tier 0; every other stage is one plus the\n"
    "        maximum tier of its direct sources. These tiers are the basis for the\n"
    "        chain-agnostic ``exp_tier{k}`` relations used by LOCO transfer.\n"
    "        \"\"\"\n"
)
FROZEN_TIERS_FORM_OF_AND_CYCLE_BLOCK = (
    "        direct = {stage: sorted(sources) for stage, sources in direct_sets.items()}\n"
    "        cache = {}\n"
)
CURRENT_TIERS_FORM_OF_AND_CYCLE_BLOCK = (
    "        # form_of is directional (source form -> processed form). It remains a\n"
    "        # background graph relation, but its stage direction must also inform\n"
    "        # LOCO's processing-tier abstraction.\n"
    "        for source_stage, target_stage in self.form_of:\n"
    "            if source_stage in self.stages and target_stage in self.stages:\n"
    "                direct_sets.setdefault(target_stage, set()).add(source_stage)\n"
    "        direct = {stage: sorted(sources) for stage, sources in direct_sets.items()}\n"
    "\n"
    "        # Reject a malformed registry with a deterministic, actionable error\n"
    "        # instead of recursing indefinitely while computing tiers. Validate the\n"
    "        # complete stage graph even when a cycle happens to touch a declared raw\n"
    "        # stage, whose tier is otherwise pinned to zero below.\n"
    "        settled = set()\n"
    "        active = []\n"
    "\n"
    "        def validate_acyclic(stage):\n"
    "            if stage in settled:\n"
    "                return\n"
    "            if stage in active:\n"
    "                start = active.index(stage)\n"
    "                cycle = active[start:] + [stage]\n"
    "                raise ValueError(\n"
    "                    f\"{self.id}: cycle in processing-tier DAG: {' -> '.join(cycle)}\"\n"
    "                )\n"
    "            active.append(stage)\n"
    "            for source in direct.get(stage, ()):\n"
    "                validate_acyclic(source)\n"
    "            active.pop()\n"
    "            settled.add(stage)\n"
    "\n"
    "        for stage in self.stages:\n"
    "            validate_acyclic(stage)\n"
    "\n"
    "        cache = {}\n"
)

UNIVERSE_TEXT_CHANGES = (
    {
        "id": "module_overview_declares_directional_form_of",
        "frozen": FROZEN_UNIVERSE_OVERVIEW,
        "current": CURRENT_UNIVERSE_OVERVIEW,
    },
    {
        "id": "tiers_contract_declares_full_directional_stage_dag",
        "frozen": FROZEN_TIERS_DOCSTRING,
        "current": CURRENT_TIERS_DOCSTRING,
    },
    {
        "id": "tiers_adds_form_of_edges_and_deterministic_cycle_rejection",
        "frozen": FROZEN_TIERS_FORM_OF_AND_CYCLE_BLOCK,
        "current": CURRENT_TIERS_FORM_OF_AND_CYCLE_BLOCK,
    },
)

FROZEN_RUNNER_OVERVIEW = '''and from the frozen v1 score generators.  The protocol has three process-level
phases:

1. ``select``: use only fold2 candidate labels and the fold2 early graph;
2. ``freeze``: hash-lock *all* requested chain/track/family selections;
3. ``evaluate``: verify the full manifest before importing a main-data loader,
   refit a label-free representation on the main early graph, score the complete
   cohort, and only then read main labels for one-shot evaluation.
'''
CURRENT_RUNNER_OVERVIEW = '''and from the frozen v1 score generators.  The public formal protocol has three
process-level phases:

1. ``select-chain``: use only fold2 labels and the fold2 early graph, sharing
   each trained score grid across A, B1, and B2 for one chain/family;
2. ``freeze``: hash-lock *all* requested chain/track/family selections;
3. ``evaluate-chain``: verify the formal run configuration and full manifest,
   refit label-free representations on the main early graph, score the complete
   A/B cohort for one chain/family, and only then read main labels.

Single-task selection/evaluation commands are deliberately not exposed: their
artifacts cannot satisfy the shared-chain freeze contract.
'''

FROZEN_RUNNER_RUN_LOCK = '''def _resolve_run_lock(args) -> tuple[str, Path, str]:
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
    return configured, config_path, sha256_file(config_path)
'''
CURRENT_RUNNER_RUN_LOCK = '''def _resolve_run_lock(args, *, phase: str | None = None) -> tuple[str, Path, str]:
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
'''

FROZEN_RUNNER_FORMAL_CONSTANTS = '''DEFAULT_KGE_MODELS = ("TransE", "RotatE", "DistMult", "ComplEx", "RGCN", "CompGCN")
FORMAL_EXECUTION_STATUS = "FORMAL_RUN_AUTHORIZED"
KEYS = ("i_iso", "j_iso", "stage")
'''
CURRENT_RUNNER_FORMAL_CONSTANTS = '''DEFAULT_KGE_MODELS = ("TransE", "RotatE", "DistMult", "ComplEx", "RGCN", "CompGCN")
FORMAL_EXECUTION_STATUS = "FORMAL_RUN_AUTHORIZED"
FORMAL_BOOTSTRAP_ITERS = 500
FORMAL_BOOTSTRAP_SEED = 20260712
KEYS = ("i_iso", "j_iso", "stage")
'''

FROZEN_RUNNER_FORMAL_VALIDATORS = '''def _assert_complete_global_freeze(indexed) -> None:
'''
CURRENT_RUNNER_FORMAL_VALIDATORS = '''def _config_value(config: Mapping[str, object], path: tuple[str, ...]) -> object:
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
'''

FROZEN_RUNNER_SELECT_CHAIN_GATE = '''def _select_chain(args) -> int:
    """Shared-training fold2 selection for A, B1, and B2 on one chain/family."""
    import numpy as np
    import pandas as pd
    from split import split_test_mask

    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)
    run_id, run_config, run_config_sha256 = _resolve_run_lock(args)
'''
CURRENT_RUNNER_SELECT_CHAIN_GATE = '''def _select_chain(args) -> int:
    """Shared-training fold2 selection for A, B1, and B2 on one chain/family."""
    run_id, run_config, run_config_sha256 = _resolve_run_lock(
        args, phase="select-chain"
    )

    import numpy as np
    import pandas as pd
    from split import split_test_mask

    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)
'''

FROZEN_RUNNER_EVALUATE_CHAIN_GATE = '''def _evaluate_chain(args) -> int:
    """Shared-training complete-main evaluation for one chain/family."""
    import pandas as pd

    # Pure protocol gate first: no target loader imports above this line.
    manifest_path = _resolve(args.manifest)
    manifest, indexed = verify_freeze_manifest(manifest_path)
    _assert_complete_global_freeze(indexed)
    run_id, run_config, run_config_sha256 = _resolve_run_lock(args)
'''
CURRENT_RUNNER_EVALUATE_CHAIN_GATE = '''def _evaluate_chain(args) -> int:
    """Shared-training complete-main evaluation for one chain/family."""
    run_id, run_config, run_config_sha256 = _resolve_run_lock(
        args, phase="evaluate-chain"
    )

    # Pure protocol gate first: no target loader imports above this line.
    manifest_path = _resolve(args.manifest)
    manifest, indexed = verify_freeze_manifest(manifest_path)
    _assert_complete_global_freeze(indexed)
'''

FROZEN_RUNNER_LATE_TARGET_IMPORT = '''    if args.overwrite:
        raise ProtocolError("main evaluation never permits --overwrite")

    output_root = _resolve(args.output_root)
'''
CURRENT_RUNNER_LATE_TARGET_IMPORT = '''    if args.overwrite:
        raise ProtocolError("main evaluation never permits --overwrite")

    import pandas as pd

    output_root = _resolve(args.output_root)
'''

FROZEN_RUNNER_SELECT_PARSER = '''    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="select class/HP using fold2 only")
    _add_runtime_args(select)
    _add_selection_grid_args(select)
    select.set_defaults(func=_select)

    select_chain = sub.add_parser(
        "select-chain",
        help="shared-training fold2 selection producing independent A/B1/B2 artifacts",
'''
CURRENT_RUNNER_SELECT_PARSER = '''    sub = parser.add_subparsers(dest="command", required=True)

    select_chain = sub.add_parser(
        "select-chain",
        help="formal shared-training fold2 selection for A/B1/B2",
'''

FROZEN_RUNNER_EVALUATE_PARSER = '''    freeze.set_defaults(func=_freeze)

    evaluate = sub.add_parser("evaluate", help="one-shot complete-main evaluation after freeze gate")
    _add_runtime_args(evaluate)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--seeds", default="0,1,2,3,4")
    evaluate.add_argument("--bootstrap-iters", type=int, default=500)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20260712)
    evaluate.set_defaults(func=_evaluate)

    evaluate_chain = sub.add_parser(
        "evaluate-chain",
        help="shared-training complete-main evaluation for independent A/B1/B2 selections",
'''
CURRENT_RUNNER_EVALUATE_PARSER = '''    freeze.set_defaults(func=_freeze)

    evaluate_chain = sub.add_parser(
        "evaluate-chain",
        help="formal shared-training complete-main evaluation for A/B1/B2",
'''

FROZEN_RUNNER_BOOTSTRAP_PARSER_DEFAULTS = '''    evaluate_chain.add_argument("--manifest", type=Path, required=True)
    evaluate_chain.add_argument("--seeds", default="0,1,2,3,4")
    evaluate_chain.add_argument("--bootstrap-iters", type=int, default=500)
    evaluate_chain.add_argument("--bootstrap-seed", type=int, default=20260712)
'''
CURRENT_RUNNER_BOOTSTRAP_PARSER_DEFAULTS = '''    evaluate_chain.add_argument("--manifest", type=Path, required=True)
    evaluate_chain.add_argument("--seeds", default="0,1,2,3,4")
    evaluate_chain.add_argument(
        "--bootstrap-iters", type=int, default=FORMAL_BOOTSTRAP_ITERS
    )
    evaluate_chain.add_argument(
        "--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED
    )
'''

RUNNER_TEXT_CHANGES = (
    {"id": "formal_workflow_contract_only_exposes_shared_chain_commands", "frozen": FROZEN_RUNNER_OVERVIEW, "current": CURRENT_RUNNER_OVERVIEW},
    {"id": "run_lock_invokes_phase_specific_execution_spec_validation", "frozen": FROZEN_RUNNER_RUN_LOCK, "current": CURRENT_RUNNER_RUN_LOCK},
    {"id": "formal_evaluation_bootstrap_constants_are_named_and_fixed", "frozen": FROZEN_RUNNER_FORMAL_CONSTANTS, "current": CURRENT_RUNNER_FORMAL_CONSTANTS},
    {"id": "formal_execution_spec_compares_cli_and_code_to_config", "frozen": FROZEN_RUNNER_FORMAL_VALIDATORS, "current": CURRENT_RUNNER_FORMAL_VALIDATORS},
    {"id": "selection_validates_formal_spec_before_data_imports", "frozen": FROZEN_RUNNER_SELECT_CHAIN_GATE, "current": CURRENT_RUNNER_SELECT_CHAIN_GATE},
    {"id": "evaluation_validates_formal_spec_before_manifest_and_data", "frozen": FROZEN_RUNNER_EVALUATE_CHAIN_GATE, "current": CURRENT_RUNNER_EVALUATE_CHAIN_GATE},
    {"id": "evaluation_defers_target_data_import_until_after_gates", "frozen": FROZEN_RUNNER_LATE_TARGET_IMPORT, "current": CURRENT_RUNNER_LATE_TARGET_IMPORT},
    {"id": "parser_removes_nonfreezable_single_task_select", "frozen": FROZEN_RUNNER_SELECT_PARSER, "current": CURRENT_RUNNER_SELECT_PARSER},
    {"id": "parser_removes_nonfreezable_single_task_evaluate", "frozen": FROZEN_RUNNER_EVALUATE_PARSER, "current": CURRENT_RUNNER_EVALUATE_PARSER},
    {"id": "formal_evaluation_parser_uses_fixed_bootstrap_constants", "frozen": FROZEN_RUNNER_BOOTSTRAP_PARSER_DEFAULTS, "current": CURRENT_RUNNER_BOOTSTRAP_PARSER_DEFAULTS},
)

ALLOWED_CHANGES: dict[str, dict[str, Any]] = {
    "chains/sheep.json": {
        "current_sha256": "2b8be327d51a1b621c2136f4dae74bba0669935ec90933d14f2ad55f1739af4c",
        "current_newline": "lf",
        "json_changes": (
            {"pointer": "/description", "frozen": OLD_DESCRIPTION, "current": NEW_DESCRIPTION},
        ),
    },
    "chains/evidence/registry_evidence.json": {
        "current_sha256": "3187a5b6adb06e08d458538336cb88588137aaa7540df5e88022051534a40f51",
        "current_newline": "lf",
        "include_semantic_stage_text_changes": True,
        "json_changes": (
            {
                "pointer": "/chains/sheep/display_description",
                "frozen": OLD_DESCRIPTION,
                "current": NEW_DESCRIPTION,
            },
        ),
    },
    "docs/registry_audit.json": {
        "current_sha256": "0ae65bcc6870cbf1bc77f03821fbe42bd3eaebf466bed496eac015c5e9511c61",
        "current_newline": "lf",
        "include_semantic_stage_text_changes": True,
        "json_changes": (
            {
                "pointer": "/chains/sheep/registry_sha256",
                "frozen": "d06a7c5d2ef15eafca6d3403b72ceaf1e7d16d3f08b3670a2524dd9f9e71dcd4",
                "current": "2b8be327d51a1b621c2136f4dae74bba0669935ec90933d14f2ad55f1739af4c",
            },
            {
                "pointer": "/chains/sheep/display_description",
                "frozen": OLD_DESCRIPTION,
                "current": NEW_DESCRIPTION,
            },
        ),
    },
    "results_v2/metrics/b1_candidate_coverage.json": {
        "current_sha256": "260e0c4e64d9640810b7b0cb45cf8a3b38cd91e914647e28fe7bafddc8e19733",
        "current_newline": "crlf",
        "json_changes": (
            {
                "pointer": "/generated_at",
                "frozen": "2026-07-16T11:46:48+00:00",
                "current": "2026-07-17T08:25:18+00:00",
            },
            {
                "pointer": "/source/cache_manifest_sha256",
                "frozen": "4c0ff190adb422b4253f3acd50fb7bbd4ac0faff6683910b1dd4c5803bacdb8d",
                "current": "f0cf1c224c4a2c239288d93eb6e884ad2cfed2bd6404f5932e3824536b3718ce",
            },
            {
                "pointer": "/registry/audit/sha256",
                "frozen": "46d694cf6a979b64cbc92a2e9b6f0b1e8af934d9696a155dbfc3f6bc70fdf5e5",
                "current": "0ae65bcc6870cbf1bc77f03821fbe42bd3eaebf466bed496eac015c5e9511c61",
            },
            {
                "pointer": "/registry/evidence/sha256",
                "frozen": "1e10e0f15ef06fb404ad86d29ccef659e50512951fee27f2c5773eb62731ea12",
                "current": "3187a5b6adb06e08d458538336cb88588137aaa7540df5e88022051534a40f51",
            },
            {
                "pointer": "/protocol_sha256/registry_loader/sha256",
                "frozen": "c8a1e41431cc94ff42bf5a41f227864a37eebe43a17e109aea8e7de52f3bd374",
                "current": "6b60cd436150e8a99cfd68b3e99044141a32bd50ee92838bbd11955ac7108fc1",
            },
        ),
    },
    "src/universe.py": {
        "kind": "text",
        "current_sha256": "6b60cd436150e8a99cfd68b3e99044141a32bd50ee92838bbd11955ac7108fc1",
        "current_newline": "lf",
        "text_changes": UNIVERSE_TEXT_CHANGES,
    },
    "src/v2_gpu_rolling.py": {
        "kind": "text",
        "current_sha256": "8508a05935ea1253275f3226e30a30af9fc613b4b1c4ffcd2c6f8bcd0d4a050d",
        "current_newline": "lf",
        "text_changes": RUNNER_TEXT_CHANGES,
    },
}

CHAIN_MACHINE_FIELDS = (
    "id",
    "stages",
    "upstream",
    "upstream_map",
    "derived_from",
    "derived_from_hs",
    "produces",
    "form_of",
    "named_sources",
    "assumption_strength",
)
CHAIN_IDS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")


def _fail(message: str) -> None:
    raise AttestationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError as exc:
        raise AttestationError(f"cannot hash {path}: {exc}") from exc


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            _fail(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"JSON root is not an object: {path}")
    return value


def _stable_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _pretty_bytes(value: Any, newline: str = "lf") -> bytes:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if newline == "crlf":
        text = text.replace("\n", "\r\n")
    elif newline != "lf":
        _fail(f"unknown newline convention: {newline}")
    return text.encode("utf-8")


def _parse_manifest() -> dict[str, str]:
    raw = FROZEN_STEP3_MANIFEST.encode("utf-8")
    if sha256_bytes(raw) != STEP3_MANIFEST_SHA256:
        _fail("embedded Step-3 manifest does not match the externally recorded SHA-256")
    rows: dict[str, str] = {}
    previous = ""
    for line in FROZEN_STEP3_MANIFEST.splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  [^\r\n]+", line):
            _fail(f"malformed embedded Step-3 line: {line!r}")
        digest, relative = line.split("  ", 1)
        if relative in rows or relative <= previous:
            _fail("embedded Step-3 inventory is duplicated or not in canonical order")
        if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
            _fail("embedded Step-3 inventory contains an absolute path")
        rows[relative] = digest
        previous = relative
    if len(rows) != 50:
        _fail(f"embedded Step-3 inventory has {len(rows)} entries, expected 50")
    return rows


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        _fail(f"invalid JSON pointer: {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _pointer_child(cursor: Any, part: str, pointer: str) -> Any:
    if isinstance(cursor, Mapping):
        if part not in cursor:
            _fail(f"missing JSON pointer: {pointer}")
        return cursor[part]
    if isinstance(cursor, list):
        if not re.fullmatch(r"0|[1-9][0-9]*", part):
            _fail(f"invalid list index in JSON pointer: {pointer}")
        index = int(part)
        if index >= len(cursor):
            _fail(f"missing JSON pointer: {pointer}")
        return cursor[index]
    _fail(f"missing JSON pointer: {pointer}")


def _pointer_get(value: Any, pointer: str) -> Any:
    cursor: Any = value
    for part in _pointer_parts(pointer):
        cursor = _pointer_child(cursor, part, pointer)
    return cursor


def _pointer_set(value: dict[str, Any], pointer: str, replacement: Any) -> None:
    parts = _pointer_parts(pointer)
    cursor: Any = value
    for part in parts[:-1]:
        cursor = _pointer_child(cursor, part, pointer)
    final = parts[-1]
    if isinstance(cursor, dict) and final in cursor:
        cursor[final] = replacement
        return
    if isinstance(cursor, list) and re.fullmatch(r"0|[1-9][0-9]*", final):
        index = int(final)
        if index < len(cursor):
            cursor[index] = replacement
            return
    _fail(f"missing JSON pointer: {pointer}")


def _pointer_delete(value: dict[str, Any], pointer: str) -> None:
    parts = _pointer_parts(pointer)
    cursor: Any = value
    for part in parts[:-1]:
        cursor = _pointer_child(cursor, part, pointer)
    final = parts[-1]
    if isinstance(cursor, dict) and final in cursor:
        del cursor[final]
        return
    if isinstance(cursor, list) and re.fullmatch(r"0|[1-9][0-9]*", final):
        index = int(final)
        if index < len(cursor):
            del cursor[index]
            return
    _fail(f"missing JSON pointer: {pointer}")


def _semantic_stage_json_changes(current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact generated text leaves changed by the clarification."""

    chains = current.get("chains")
    if not isinstance(chains, Mapping):
        _fail("registry evidence/audit lacks a chains object")
    changes: list[dict[str, Any]] = []
    for (chain_id, stage), text in sorted(SEMANTIC_STAGE_TEXT_CHANGES.items()):
        chain = chains.get(chain_id)
        if not isinstance(chain, Mapping):
            _fail(f"missing semantic-text chain {chain_id}")
        definitions = chain.get("stage_definitions")
        decisions = chain.get("decisions")
        if not isinstance(definitions, Mapping) or not isinstance(decisions, list):
            _fail(f"incomplete semantic-text records for {chain_id}")
        definition = definitions.get(stage)
        if not isinstance(definition, Mapping):
            _fail(f"missing semantic-text stage {chain_id}/{stage}")

        stage_fields = (
            ("canonical_definition", "frozen_definition", "current_definition"),
            ("fit_rule", "frozen_fit_rule", "current_fit_rule"),
        )
        for field, frozen_key, current_key in stage_fields:
            frozen_value = text[frozen_key]
            current_value = text[current_key]
            if definition.get(field) != current_value:
                _fail(f"unexpected current semantic text for {chain_id}/{stage}/{field}")
            if frozen_value != current_value:
                changes.append(
                    {
                        "pointer": f"/chains/{chain_id}/stage_definitions/{stage}/{field}",
                        "frozen": frozen_value,
                        "current": current_value,
                    }
                )

        included_indices: list[int] = []
        for index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping):
                _fail(f"invalid decision record for {chain_id}")
            if decision.get("decision") == "include" and decision.get("stage") == stage:
                included_indices.append(index)
        if not included_indices:
            _fail(f"semantic-text stage has no included decisions: {chain_id}/{stage}")

        for index in included_indices:
            decision = decisions[index]
            stage_fit = decision.get("stage_fit")
            if not isinstance(stage_fit, Mapping):
                _fail(f"included semantic-text decision lacks stage_fit: {chain_id}/{index}")
            expected_current = {
                "rationale": text["current_fit_rule"],
                "stage_fit/canonical_definition": text["current_definition"],
                "stage_fit/rationale": text["current_fit_rule"],
            }
            expected_frozen = {
                "rationale": text["frozen_fit_rule"],
                "stage_fit/canonical_definition": text["frozen_definition"],
                "stage_fit/rationale": text["frozen_fit_rule"],
            }
            for suffix in expected_current:
                parts = suffix.split("/")
                observed: Any = decision
                for part in parts:
                    if not isinstance(observed, Mapping) or part not in observed:
                        _fail(f"missing semantic-text decision field: {chain_id}/{index}/{suffix}")
                    observed = observed[part]
                if observed != expected_current[suffix]:
                    _fail(f"unexpected semantic-text decision value: {chain_id}/{index}/{suffix}")
                if expected_frozen[suffix] != expected_current[suffix]:
                    changes.append(
                        {
                            "pointer": f"/chains/{chain_id}/decisions/{index}/{suffix}",
                            "frozen": expected_frozen[suffix],
                            "current": expected_current[suffix],
                        }
                    )
    return changes


def _json_changes(relative: str, current: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = ALLOWED_CHANGES[relative]
    changes = list(spec["json_changes"])
    if spec.get("include_semantic_stage_text_changes") is True:
        changes.extend(_semantic_stage_json_changes(current))
    pointers = [str(change["pointer"]) for change in changes]
    if len(pointers) != len(set(pointers)):
        _fail(f"duplicate allowed JSON pointer for {relative}")
    return sorted(changes, key=lambda item: str(item["pointer"]))


def _changed_json_proof(root: Path, relative: str, frozen_sha256: str) -> dict[str, Any]:
    spec = ALLOWED_CHANGES[relative]
    path = root / relative
    current = _load_json(path)
    current_bytes = path.read_bytes()
    if sha256_bytes(current_bytes) != spec["current_sha256"]:
        _fail(f"current byte hash changed for {relative}")
    if current_bytes != _pretty_bytes(current, spec["current_newline"]):
        _fail(f"current JSON serialization is not the frozen convention for {relative}")

    frozen = copy.deepcopy(current)
    allowed_pointers: list[str] = []
    for change in _json_changes(relative, current):
        pointer = change["pointer"]
        if _pointer_get(current, pointer) != change["current"]:
            _fail(f"unexpected current value at {relative}{pointer}")
        _pointer_set(frozen, pointer, change["frozen"])
        allowed_pointers.append(pointer)
    if sha256_bytes(_pretty_bytes(frozen, "lf")) != frozen_sha256:
        _fail(f"allowed reverse patch does not reconstruct frozen bytes for {relative}")

    current_projection = copy.deepcopy(current)
    frozen_projection = copy.deepcopy(frozen)
    for pointer in allowed_pointers:
        _pointer_delete(current_projection, pointer)
        _pointer_delete(frozen_projection, pointer)
    current_projection_sha = sha256_bytes(_stable_bytes(current_projection))
    frozen_projection_sha = sha256_bytes(_stable_bytes(frozen_projection))
    if current_projection_sha != frozen_projection_sha:
        _fail(f"non-allowlisted JSON content changed for {relative}")

    return {
        "path": relative,
        "frozen_sha256": frozen_sha256,
        "current_sha256": spec["current_sha256"],
        "frozen_serialization": "UTF-8 LF, indent=2, trailing newline",
        "current_serialization": (
            "UTF-8 CRLF, indent=2, trailing newline"
            if spec["current_newline"] == "crlf"
            else "UTF-8 LF, indent=2, trailing newline"
        ),
        "allowed_json_pointers": allowed_pointers,
        "allowlisted_fields_removed_projection_sha256": current_projection_sha,
        "reverse_patch_reconstructs_frozen_bytes": True,
        "non_allowlisted_json_content_unchanged": True,
    }


def _changed_text_proof(root: Path, relative: str, frozen_sha256: str) -> dict[str, Any]:
    """Prove an exact, finite source-text diff by reverse patch and projection."""

    spec = ALLOWED_CHANGES[relative]
    path = root / relative
    current_bytes = path.read_bytes()
    if sha256_bytes(current_bytes) != spec["current_sha256"]:
        _fail(f"current byte hash changed for {relative}")
    try:
        current = current_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise AttestationError(f"cannot decode UTF-8 source {relative}: {exc}") from exc
    if spec["current_newline"] != "lf" or "\r" in current or not current.endswith("\n"):
        _fail(f"current source newline convention changed for {relative}")

    changes = list(spec.get("text_changes", ()))
    if not changes:
        _fail(f"source allowlist is empty for {relative}")
    identifiers = [change.get("id") for change in changes]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        _fail(f"source allowlist has an invalid identifier for {relative}")
    if len(identifiers) != len(set(identifiers)):
        _fail(f"source allowlist has duplicate identifiers for {relative}")

    frozen = current
    proof_rows: list[dict[str, Any]] = []
    for change in changes:
        identifier = str(change["id"])
        frozen_snippet = change.get("frozen")
        current_snippet = change.get("current")
        if (
            not isinstance(frozen_snippet, str)
            or not isinstance(current_snippet, str)
            or not frozen_snippet
            or not current_snippet
            or frozen_snippet == current_snippet
        ):
            _fail(f"invalid exact source snippets for {relative}:{identifier}")
        if frozen.count(current_snippet) != 1:
            _fail(f"current source snippet is not unique for {relative}:{identifier}")
        frozen = frozen.replace(current_snippet, frozen_snippet, 1)
        proof_rows.append(
            {
                "id": identifier,
                "frozen_snippet_sha256": sha256_bytes(frozen_snippet.encode("utf-8")),
                "current_snippet_sha256": sha256_bytes(current_snippet.encode("utf-8")),
                "frozen_line_count": frozen_snippet.count("\n"),
                "current_line_count": current_snippet.count("\n"),
            }
        )
    if sha256_bytes(frozen.encode("utf-8")) != frozen_sha256:
        _fail(f"allowed reverse patch does not reconstruct frozen bytes for {relative}")

    current_projection = current
    frozen_projection = frozen
    for change in changes:
        identifier = str(change["id"])
        marker = f"__ALLOWLISTED_TEXT_CHANGE_{identifier}__"
        if marker in current_projection or marker in frozen_projection:
            _fail(f"source projection marker collision for {relative}:{identifier}")
        if current_projection.count(change["current"]) != 1:
            _fail(f"current source projection is ambiguous for {relative}:{identifier}")
        if frozen_projection.count(change["frozen"]) != 1:
            _fail(f"frozen source projection is ambiguous for {relative}:{identifier}")
        current_projection = current_projection.replace(change["current"], marker, 1)
        frozen_projection = frozen_projection.replace(change["frozen"], marker, 1)
    current_projection_sha = sha256_bytes(current_projection.encode("utf-8"))
    frozen_projection_sha = sha256_bytes(frozen_projection.encode("utf-8"))
    if current_projection_sha != frozen_projection_sha:
        _fail(f"non-allowlisted source content changed for {relative}")

    return {
        "path": relative,
        "frozen_sha256": frozen_sha256,
        "current_sha256": spec["current_sha256"],
        "frozen_serialization": "UTF-8 LF, trailing newline",
        "current_serialization": "UTF-8 LF, trailing newline",
        "allowed_text_changes": proof_rows,
        "allowlisted_text_replaced_projection_sha256": current_projection_sha,
        "reverse_patch_reconstructs_frozen_bytes": True,
        "non_allowlisted_source_content_unchanged": True,
    }


def _attestation_json_pointers(value: Mapping[str, Any]) -> set[str]:
    proofs = value.get("changed_json_proofs", [])
    if not isinstance(proofs, list):
        return set()
    return {
        pointer
        for proof in proofs
        if isinstance(proof, Mapping)
        for pointer in proof.get("allowed_json_pointers", [])
        if isinstance(pointer, str)
    }


def _privacy_audit(value: Any, *, json_pointers: set[str] | None = None) -> None:
    json_pointers = set() if json_pointers is None else json_pointers
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail("public attestation contains a non-string key")
            _privacy_audit(child, json_pointers=json_pointers)
    elif isinstance(value, list):
        for child in value:
            _privacy_audit(child, json_pointers=json_pointers)
    elif isinstance(value, str):
        if value not in json_pointers and (
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
        ):
            _fail("public attestation contains an absolute path")
        lowered = value.lower()
        private_tokens = (
            "mars" + "10",
            "mars" + "29",
            "sli" + "6@",
            "/ho" + "me/",
            "c:\\" + "users\\",
        )
        if any(token in lowered for token in private_tokens):
            _fail("public attestation contains a private host, account, or home path")


def _machine_semantic_projection(root: Path) -> dict[str, Any]:
    """Prove all six active membership/stage/relation projections are unchanged."""

    machine_by_chain: dict[str, dict[str, Any]] = {}
    chain_assignments: list[tuple[str, str, str]] = []
    chain_rows: list[dict[str, Any]] = []
    for chain_id in CHAIN_IDS:
        relative = f"chains/{chain_id}.json"
        chain = _load_json(root / relative)
        required_fields = set(CHAIN_MACHINE_FIELDS) - {"assumption_strength"}
        if not required_fields.issubset(chain) or not set(chain).issubset(
            {*CHAIN_MACHINE_FIELDS, "description"}
        ):
            _fail(f"{chain_id} registry has an unexpected top-level field")
        if chain_id == "sheep" and chain.get("description") != NEW_DESCRIPTION:
            _fail("sheep registry display clarification is not the expected current text")
        machine = {field: chain[field] for field in CHAIN_MACHINE_FIELDS if field in chain}
        machine_by_chain[chain_id] = machine
        stages = chain.get("stages")
        if not isinstance(stages, Mapping):
            _fail(f"{chain_id} registry lacks stages")
        assignments: list[tuple[str, str, str]] = []
        for stage, codes in stages.items():
            if not isinstance(stage, str) or not isinstance(codes, list):
                _fail(f"{chain_id} registry has an invalid stage mapping")
            for code in codes:
                if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
                    _fail(f"{chain_id}/{stage} has a noncanonical HS6 code")
                assignments.append((chain_id, stage, code))
        if len(assignments) != len(set(assignments)):
            _fail(f"{chain_id} registry has duplicate membership/stage assignments")
        chain_assignments.extend(assignments)
        digest = sha256_bytes(_stable_bytes(machine))
        chain_rows.append(
            {
                "chain": chain_id,
                "path": relative,
                "active_assignment_count": len(assignments),
                "frozen_sha256": digest,
                "current_sha256": digest,
                "membership_stage_relation_unchanged": True,
            }
        )

    evidence = _load_json(root / "chains/evidence/registry_evidence.json")
    evidence_chains = evidence.get("chains")
    if not isinstance(evidence_chains, Mapping) or set(evidence_chains) != set(CHAIN_IDS):
        _fail("registry evidence does not contain the exact six chains")
    evidence_assignments: list[tuple[str, str, str]] = []
    for chain_id in CHAIN_IDS:
        decisions = evidence_chains[chain_id].get("decisions")
        if not isinstance(decisions, list):
            _fail(f"registry evidence lacks decisions for {chain_id}")
        for decision in decisions:
            if not isinstance(decision, Mapping):
                _fail(f"registry evidence has an invalid decision for {chain_id}")
            if decision.get("decision") == "include":
                stage = decision.get("stage")
                code = decision.get("code")
                if not isinstance(stage, str) or not isinstance(code, str):
                    _fail(f"registry evidence has an invalid active assignment for {chain_id}")
                evidence_assignments.append((chain_id, stage, code))

    chain_assignments.sort()
    evidence_assignments.sort()
    if evidence_assignments != chain_assignments:
        _fail("registry evidence active membership/stage mapping differs from chain files")
    membership_sha = sha256_bytes(_stable_bytes(chain_assignments))
    machine_sha = sha256_bytes(_stable_bytes(machine_by_chain))
    return {
        "path": "chains/*.json",
        "chain_count": len(CHAIN_IDS),
        "included_fields": list(CHAIN_MACHINE_FIELDS),
        "excluded_display_only_field": "description",
        "chains": chain_rows,
        "active_assignment_count": len(chain_assignments),
        "evidence_active_membership_stage_sha256": membership_sha,
        "chain_active_membership_stage_sha256": membership_sha,
        "evidence_active_membership_stage_matches_chain_files": True,
        "frozen_sha256": machine_sha,
        "current_sha256": machine_sha,
        "membership_stage_relation_unchanged": True,
    }


def build_attestation(root: Path = ROOT, *, require_full_inventory: bool = True) -> dict[str, Any]:
    """Rebuild the attestation from the current canonical tree.

    ``require_full_inventory=False`` is the public-clone mode.  It reconstructs
    the fixed 50-entry receipt and verifies all six public changed artifacts,
    while retaining the private full-inventory verification claim generated by
    the canonical production mode.  The GPU promotion and resolution gates
    always use the default full mode.
    """

    root = Path(root).resolve()
    frozen = _parse_manifest()
    allowed = set(ALLOWED_CHANGES)
    if not allowed.issubset(frozen):
        _fail("allowed change inventory is not contained in Step-3")

    inventory: list[dict[str, Any]] = []
    observed_changed: set[str] = set()
    for relative, frozen_sha in frozen.items():
        path = root / relative
        if relative in allowed:
            current_sha = sha256_file(path)
        elif require_full_inventory:
            current_sha = sha256_file(path)
        else:
            current_sha = frozen_sha
        status = "unchanged" if current_sha == frozen_sha else "allowed_post_freeze_change"
        if status != "unchanged":
            observed_changed.add(relative)
        inventory.append(
            {
                "path": relative,
                "frozen_sha256": frozen_sha,
                "current_sha256": current_sha,
                "status": status,
            }
        )
    if observed_changed != allowed:
        _fail(
            "current Step-3 inventory differs outside the exact six-file allowlist: "
            f"{sorted(observed_changed ^ allowed)}"
        )

    json_proofs = [
        _changed_json_proof(root, relative, frozen[relative])
        for relative in sorted(allowed)
        if ALLOWED_CHANGES[relative].get("kind", "json") == "json"
    ]
    source_proofs = [
        _changed_text_proof(root, relative, frozen[relative])
        for relative in sorted(allowed)
        if ALLOWED_CHANGES[relative].get("kind", "json") == "text"
    ]
    candidate_rows = [
        row for row in inventory if row["path"].startswith("data/processed_v2/candidates")
    ]
    if len(candidate_rows) != 24 or any(row["status"] != "unchanged" for row in candidate_rows):
        _fail("the 24 fold2/main candidate bytes are not exactly Step-3 identical")
    config_row = next(
        (row for row in inventory if row["path"] == "configs/v2_gpu_rolling.json"), None
    )
    if config_row is None or config_row["status"] != "unchanged":
        _fail("the formal GPU run config is not exactly Step-3 identical")

    machine_projection = _machine_semantic_projection(root)

    coverage = _load_json(root / "results_v2/metrics/b1_candidate_coverage.json")
    coverage_definition_sha = sha256_bytes(_stable_bytes(coverage.get("definition")))
    coverage_snapshots_sha = sha256_bytes(_stable_bytes(coverage.get("snapshots")))
    candidate_map = {row["path"]: row["current_sha256"] for row in candidate_rows}

    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "run_id": RUN_ID,
        "scope": (
            "post-freeze registry wording, derived provenance, and exact LOCO tier-source "
            "correction, plus prospective formal-run CLI/config hardening; no GPU result "
            "recomputation"
        ),
        "generator": {
            "path": "tools/build_gpu_step3_postfreeze_attestation.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "verification_contract": {
            "canonical_generation_requires_all_50_current_files": True,
            "gpu_summary_and_resolution_require_full_inventory_verification": True,
            "public_clone_mode_reconstructs_the_fixed_receipt_without_external_candidates": True,
        },
        "frozen_step3_manifest": {
            "artifact_role": STEP3_MANIFEST_ROLE,
            "sha256": STEP3_MANIFEST_SHA256,
            "entry_count": len(frozen),
            "inventory_reconstructed_from_public_hash_receipt": True,
        },
        "inventory": inventory,
        "comparison": {
            "entry_count": len(inventory),
            "exact_byte_match_count": len(inventory) - len(allowed),
            "allowed_changed_file_count": len(allowed),
            "allowed_changed_files": sorted(allowed),
            "no_other_step3_file_changed": True,
        },
        "changed_json_proofs": json_proofs,
        "changed_source_proofs": source_proofs,
        "machine_semantic_projection": machine_projection,
        "run_config": {
            "path": config_row["path"],
            "frozen_sha256": config_row["frozen_sha256"],
            "current_sha256": config_row["current_sha256"],
            "exact_bytes_unchanged": True,
        },
        "candidate_inputs": {
            "count": len(candidate_rows),
            "sha256_by_path": candidate_map,
            "all_exact_bytes_unchanged": True,
        },
        "coverage_invariants": {
            "path": "results_v2/metrics/b1_candidate_coverage.json",
            "definition_sha256": coverage_definition_sha,
            "snapshots_and_all_quantities_sha256": coverage_snapshots_sha,
            "definition_unchanged": True,
            "snapshots_and_all_quantities_unchanged": True,
            "changed_fields_are_only_timestamp_cache_registry_and_registry_loader_provenance": True,
        },
        "claim_boundary": {
            "supported": (
                "The post-freeze registry clarifications changed four JSON artifacts only at "
                "display wording, canonical-definition/rationale text, or derived-provenance "
                "fields. The fifth changed file is exactly reverse-patched to the three declared "
                "LOCO Chain.tiers documentation/form_of/cycle-rejection snippets. The sixth is "
                "exactly reverse-patched to ten prospective fail-closed runner changes that bind "
                "the shared-chain CLI to the formal config, including its complete task-metric "
                "mapping and fixed evaluation-bootstrap settings, and remove non-freezable "
                "single-task commands. The formal config, candidate cohort, training and metric implementation "
                "outside those exact gates, all six registry membership/stage/relation projections, "
                "all 24 candidate bytes, and all B1 coverage quantities are unchanged."
            ),
            "not_supported": (
                "This attestation does not alter the immutable Step-3 snapshot, selections, "
                "model outputs, metrics, or the formal GPU run, and it does not claim that the "
                "frozen run was executed under the later CLI/config gates or command-surface "
                "restriction. It also does not claim that the current LOCO processing-tier "
                "abstraction is byte- or behavior-identical to the frozen implementation."
            ),
        },
    }
    _privacy_audit(result, json_pointers=_attestation_json_pointers(result))
    return result


def render(attestation: Mapping[str, Any]) -> bytes:
    _privacy_audit(
        attestation,
        json_pointers=_attestation_json_pointers(attestation),
    )
    return (
        json.dumps(attestation, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def verify_output(
    path: Path = DEFAULT_OUTPUT,
    *,
    root: Path = ROOT,
    require_full_inventory: bool = True,
) -> dict[str, Any]:
    expected = build_attestation(root, require_full_inventory=require_full_inventory)
    try:
        actual = Path(path).read_bytes()
    except OSError as exc:
        raise AttestationError(f"cannot read attestation artifact {path}: {exc}") from exc
    if actual != render(expected):
        _fail("post-freeze semantic attestation is stale or non-deterministic")
    return expected


def verify_summary_binding(
    summary: Mapping[str, Any],
    *,
    artifact_path: Path = DEFAULT_OUTPUT,
    root: Path = ROOT,
    require_full_inventory: bool = False,
) -> dict[str, Any]:
    attestation = verify_output(
        artifact_path, root=root, require_full_inventory=require_full_inventory
    )
    binding = summary.get("post_freeze_semantic_attestation")
    if not isinstance(binding, Mapping):
        _fail("GPU summary lacks the post-freeze semantic attestation binding")
    expected = {
        "artifact_role": ARTIFACT_ROLE,
        "sha256": sha256_file(artifact_path),
        "schema_version": SCHEMA,
        "step3_manifest_sha256": STEP3_MANIFEST_SHA256,
        "allowed_changed_file_count": len(ALLOWED_CHANGES),
        "machine_semantics_and_candidate_bytes_unchanged": True,
    }
    if dict(binding) != expected:
        _fail("GPU summary post-freeze semantic attestation binding is stale or incomplete")
    if summary.get("run_id") != RUN_ID or attestation.get("run_id") != RUN_ID:
        _fail("GPU summary and attestation run identifiers disagree")
    return attestation


def summary_binding(artifact_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    return {
        "artifact_role": ARTIFACT_ROLE,
        "sha256": sha256_file(artifact_path),
        "schema_version": SCHEMA,
        "step3_manifest_sha256": STEP3_MANIFEST_SHA256,
        "allowed_changed_file_count": len(ALLOWED_CHANGES),
        "machine_semantics_and_candidate_bytes_unchanged": True,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--public-check",
        action="store_true",
        help="verify the published receipt/current public JSONs without external candidate payloads",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check or args.public_check:
            result = verify_output(
                args.output,
                root=ROOT,
                require_full_inventory=not args.public_check,
            )
            action = "verified"
        else:
            result = build_attestation(ROOT, require_full_inventory=True)
            _atomic_write(args.output, render(result))
            action = "wrote"
    except AttestationError as exc:
        print(f"GPU POST-FREEZE ATTESTATION REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        f"{action} {result['comparison']['entry_count']} Step-3 entries; "
        f"{result['comparison']['allowed_changed_file_count']} allowlisted artifact changes; "
        "24 candidate files unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
