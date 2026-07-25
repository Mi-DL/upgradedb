"""Formal fixed-checkpoint ULTRA zero-shot protocol for UpgradeBench.

The controller has four irreversible phases:

1. ``freeze`` binds the exact ``ultra_4g`` checkpoint, raw BACI archive,
   source tree, disclosed pretraining provenance, six early graphs, full A/B
   candidate-file byte hashes, and their identity-only cohorts.
2. ``score-chain`` scores the complete A and B lane cohorts for one chain with
   the official native ULTRA backend.  Sheep is scored twice in the same
   process/device as a predeclared repeatability sentinel.  Neither run parses
   target outcomes.
3. ``seal-scores`` verifies and hash-seals all six chain components.
4. ``evaluate`` is the only command that reads target outcomes, and it refuses
   to do so until the complete six-chain score seal verifies.

A is evaluated directly.  B1 takes the maximum lane score and label within an
exporter-stage entry.  B2 reuses the same B lane scores and ranks destinations
only inside realized positive entries.  There is no selection, fine-tuning,
checkpoint override, or compatibility-shim path in this formal controller.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
ULTRA_ROOT = ROOT / "third_party" / "ULTRA"
sys.path.insert(0, str(SRC))

CONFIG_SCHEMA = "upgrade-bench-v2/ultra-formal-config/2"
FREEZE_SCHEMA = "upgrade-bench-v2/ultra-formal-freeze/2"
COMPONENT_SCHEMA = "upgrade-bench-v2/ultra-formal-score-component/2"
SCORE_START_SCHEMA = "upgrade-bench-v2/ultra-formal-score-start/2"
SCORE_SEAL_SCHEMA = "upgrade-bench-v2/ultra-formal-score-seal/3"
EVALUATION_START_SCHEMA = "upgrade-bench-v2/ultra-formal-evaluation-start/3"
METRIC_SCHEMA = "upgrade-bench-v2/ultra-formal-chain-metrics/3"
EVALUATION_SCHEMA = "upgrade-bench-v2/ultra-formal-evaluation/3"
PROTOCOL = "upgrade-bench-v2/ultra-4g-zero-shot/2"
AUTHORIZED_STATUS = "FORMAL_ZERO_SHOT_RUN_AUTHORIZED"
CANONICAL_RUN_ID = "ultra-4g-zero-shot-fixed-20260717-r4"
CANONICAL_CONFIG = ROOT / "configs" / "v2_ultra_formal.json"
CANONICAL_CHECKPOINT_NAME = "ultra_4g"
CANONICAL_CHECKPOINT_PATH = "third_party/ULTRA/ckpts/ultra_4g.pth"
CANONICAL_CHECKPOINT_SHA256 = (
    "48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0"
)
CANONICAL_CHECKPOINT_BYTES = 2127350
CANONICAL_VENDORED_MODULE_PATHS = {
    "ultra": "ultra/__init__.py",
    "ultra.layers": "ultra/layers.py",
    "ultra.models": "ultra/models.py",
    "ultra.tasks": "ultra/tasks.py",
    "ultra.rspmm.rspmm": "ultra/rspmm/rspmm.py",
}
SCORE_UNLOCK_POLICY = "evaluation may read labels only after this exact seal verifies"
SCORING_START_POLICY = (
    "freeze is immutable; target labels remain locked until the six-chain score seal"
)
ULTRA_INFERENCE_SOURCE_LF_SHA256 = {
    "ultra/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ultra/base_nbfnet.py": "a327f9912308be70629c10af1307b3e7c20adbc353b65c9ab641fa6dbfa23fc4",
    "ultra/datasets.py": "a7d5584a2b1825651f9b311e96f0c6ef6dde049754dfcc7cd2988c57017e2322",
    "ultra/datasets_query.py": "c82a04f477724cf3b6d69ef9e2bcdc2c03b7da0b856c47c915d8b3608168a510",
    "ultra/layers.py": "ae455da48d74f18a5cd6c4f5190669c7a688602ff134f8c0bdbb7579af13b5f6",
    "ultra/models.py": "65b0015bf31b036ca7c5e9a2e0aa0c1f5a95f4c844d7219d73c0d8b8af3aa7bd",
    "ultra/query_utils.py": "9b41775a5b54a0e40b9bd3acc24e0f20e54ecf0150f30beae382aa95c36191a0",
    "ultra/rspmm/__init__.py": "bfc17ff9a568eaa11f62ae31d4bf6d1ecda7f119ed9b7300316a85f05f420498",
    "ultra/rspmm/rspmm.py": "c9c3e8a78c6b69ed62fc142839239a2d50bce7b795dd78a5149d412059e51167",
    "ultra/rspmm/source/operator.cuh": "5a4255123a2c9fd62f18e8663eb531bdada06cd98b11e553a52cb7d052e3f118",
    "ultra/rspmm/source/rspmm.cpp": "e944e01f619804b667dfa5e9c2dff013469ae75f818ebfd20a1c68039bb4ccb9",
    "ultra/rspmm/source/rspmm.cu": "ba5ca65bfe4969cc945ea50e5eb533d04ad9ac1585d6d667fea9875db670c224",
    "ultra/rspmm/source/rspmm.h": "1fa726c7db45655d09f2bd2e67907a7bb298d847533fb93afb92e809beb08a92",
    "ultra/rspmm/source/util.cuh": "eb6f48c7f40955a608d461c9bfb2cc9362a830cae5162fc1c360235eed963a98",
    "ultra/tasks.py": "d3697fd038c9f9fc5072444416a870ba6eb3717e65766eae39dbc0773525a07c",
    "ultra/ultraquery.py": "0cb132099b208ab8961a479395c15ab6d2b1cf3847c1b3c9e12ccc3fdec41c8a",
    "ultra/util.py": "d10cd4ff0e06e25e31222f876393bd85fb8fbce3af727708432908a39a150822",
    "ultra/variadic.py": "ab4152e3f32d931b78e28a675fa4fde3a62a68074765a4fa78e17147cfe37f6d",
}
PROVENANCE = {
    "upstream_repository": "https://github.com/DeepGraphLearning/ULTRA",
    "upstream_commit": "427966ad8ed60420eef034063d44f3153addff90",
    "checkpoint_release_commit": "68433c19a465735ae59f7f947e6dd062bab7b445",
    "checkpoint_git_blob_sha1": "b10ca4f3a26c5084a6ec72e77dfc756b04acef4c",
    "checkpoint_training_config_commit": "68433c19a465735ae59f7f947e6dd062bab7b445",
    "checkpoint_training_config_git_blob": "073ce8a72c20474006a0adc861b1f80bda42f7c7",
    "vendored_source_root": "third_party/ULTRA",
    "selection_basis": (
        "provenance completeness only; no current-protocol ULTRA outcome was computed or used "
        "for checkpoint selection"
    ),
    "source_tree_receipt": {
        "status": "PASS",
        "comparison": (
            "the listed formal inference source files match upstream commit "
            "427966ad8ed60420eef034063d44f3153addff90 after LF normalization"
        ),
        "normalization": "listed formal inference files only; __pycache__ excluded",
        "normalized_lf_sha256": ULTRA_INFERENCE_SOURCE_LF_SHA256,
    },
}
LICENSE = {
    "spdx": "MIT",
    "path": "third_party/ULTRA/LICENSE",
    "sha256": "91776f870b5502e85931e2856e2367423da25d6883269f2f3c655cb7fc450a0e",
}
TRAINING_DISCLOSURE = {
    "pretraining_graphs": ["FB15k237", "WN18RR", "CoDExMedium", "NELL995"],
    "training_steps": 400000,
    "checkpoint_training_seed_disclosed": False,
    "reference_cli_default_seed": 1024,
    "reference_cli_default_seed_scope": (
        "upstream CLI default only; not evidence of the checkpoint training seed"
    ),
    "readme_path": "third_party/ULTRA/README.md",
    "readme_sha256": "6ff54381b4da55c25d84553db66b185c6bc2bd3b0736ebe0d834a22540042029",
    "vendored_reference_config_path": "third_party/ULTRA/config/transductive/pretrain_4g.yaml",
    "vendored_reference_config_sha256": "23257430dd594a3a4535eb135c866c4d56ad217c394fce2dee27db8a8d005fe3",
    "vendored_reference_config_status": (
        "semantic reference only; local class/path renames mean its bytes are not asserted "
        "identical to the checkpoint training config"
    ),
}
OVERLAP_POLICY = {
    "benchmark_sources": ["CEPII BACI", "CEPII Gravity"],
    "disclosed_pretraining_list_contains_benchmark_sources": False,
    "dataset_level_assessment": (
        "the disclosed four-graph pretraining mixture does not list BACI or CEPII Gravity"
    ),
    "scope_limit": (
        "dataset provenance excludes direct listed-dataset overlap, not every possible shared public fact"
    ),
    "ultra_50g_formal_use": (
        "forbidden because its 50-graph training list is not publicly disclosed"
    ),
}
RAW_SOURCE_POLICY = {
    "attestation_path": "requirements/raw_source_attestation.json",
    "attestation_sha256": "99a241cb9aff4ef9d0ab702e288b0341c43207bb5462dbb891fd9769f29f8471",
    "archive_name": "BACI_HS92_V202401b.zip",
    "archive_bytes": 2450783074,
    "archive_sha256": "1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a",
    "runtime_path_resolution": "temporal_backtest.BACI_ZIP; host path is not serialized",
    "gravity_is_formal_runtime_dependency": False,
}
DATA_PRECOMMIT_POLICY = {
    "full_candidate_csv_bytes_hashed_before_scoring": True,
    "byte_hashing_is_semantic_target_access": False,
    "target_columns_parsed_before_global_score_seal": False,
    "candidate_bytes_reverified_at_score_seal_and_evaluation": True,
}

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
KEYS = ("i_iso", "j_iso", "stage")
IDENTITY_COLUMNS = KEYS + (
    "benchmark_version",
    "aggregation",
    "early_window",
    "late_window",
    "task",
    "task_unit",
)
FORBIDDEN_TARGET_COLUMNS = frozenset(
    {
        "y",
        "entry_y",
        "lateval",
        "size",
        "grav",
        "gnn",
        "log_exporter_capacity",
        "log_importer_demand",
    }
)
SOURCE_SPECS: dict[str, dict[str, str]] = {
    "A": {
        "filename": "candidates_{chain}.csv",
        "task": "destination_extension",
        "task_unit": "exporter_stage_destination",
    },
    "B": {
        "filename": "candidates_firsttime_{chain}.csv",
        "task": "processed_export_entry_candidate_lane",
        "task_unit": "exporter_stage_destination",
    },
}

MODEL_POLICY = {
    "mode": "external_pretrained_zero_shot",
    "checkpoint_count": 1,
    "checkpoint_selection": "fixed_before_target_scoring",
    "historical_or_target_label_selection": False,
    "fine_tuning": False,
    "training": False,
    "checkpoint_search": False,
    "seed_or_hyperparameter_search": False,
}
TASK_CONTRACT = {
    "score_sources": ["A_destination_extension_lanes", "B_entry_candidate_lanes"],
    "A": "direct lane scores; headline lane average precision",
    "B1": (
        "exporter-stage label and score are max over B candidate lanes, observed value is summed; "
        "headlines are entry average precision and value capture@50"
    ),
    "B2": (
        "same B lane scores, ranked only within realized positive exporter-stage entries; "
        "headline macro recall@3"
    ),
    "shared_B_scores": True,
}
PHASE_ORDER = [
    (
        "freeze config, checkpoint, raw BACI, source, early graphs, full candidate-file byte "
        "hashes, and A/B identity cohorts without semantically parsing outcomes"
    ),
    (
        "score complete A and B lane cohorts for each of all six chains, including the "
        "predeclared same-process sheep repeat"
    ),
    "verify and hash-seal all six score components",
    "only after the global score seal, read target labels once and evaluate A, B1, and B2",
]
RUNTIME_CONTRACT = {
    "device": "cuda_required",
    "backend": "official_ULTRA_native_rspmm_and_native_torch_scatter",
    "torch_scatter_compat_shim": False,
    "message_passing_fallback": False,
    "batch_query_groups": 8,
}
REPEATABILITY_CONTRACT = {
    "sentinel_chain": "sheep",
    "runs": 2,
    "scope": "complete de-duplicated combined A+B lane union",
    "same_process": True,
    "same_device": True,
    "same_model_instance": True,
    "inference_seed": 1024,
    "inference_seed_scope": (
        "inference control only; not the undisclosed checkpoint training seed"
    ),
    "numeric_policy": {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "global_deterministic_algorithms": False,
    },
    "primary_run": "run1 fixed before scoring; repeat cannot select or replace it",
    "score_acceptance_rtol": 1e-6,
    "score_acceptance_atol": 1e-7,
    "metric_absolute_delta_max": 1e-10,
    "metric_gate": [
        "A.lane_average_precision",
        "A.value_capture_at_500",
        "B1.entry_average_precision",
        "B1.entry_value_capture_at_50",
        "B2.conditional_recall_at_3",
        "B2.conditional_value_capture_at_3",
    ],
    "persist_repeat_A_B_scores": True,
    "model_or_result_selection": False,
}
REPEAT_METRIC_PATHS = {
    "A.lane_average_precision": ("A", "lane_average_precision"),
    "A.value_capture_at_500": ("A", "value_capture_at_500"),
    "B1.entry_average_precision": ("B1", "entry_average_precision"),
    "B1.entry_value_capture_at_50": ("B1", "entry_value_capture_at_50"),
    "B2.conditional_recall_at_3": ("B2", "conditional_recall_at_3"),
    "B2.conditional_value_capture_at_3": ("B2", "conditional_value_capture_at_3"),
}
REPORTING_CONTRACT = {
    "model_label": "external pretrained zero-shot",
    "trained_reference_artifact": "results_v2/metrics/v2_gpu_rolling_summary.json",
    "trained_reference_artifact_sha256": (
        "5978ca62462f68ffc93054fd1c448ee768646ec82a8b61b5d843d56559193acd"
    ),
    "trained_reference_families": ["kge", "nbfnet"],
    "headline_metric_by_task": {
        "A": "lane_average_precision",
        "B1": "entry_average_precision",
        "B2": "conditional_recall_at_3",
    },
    "value_metric_by_task": {
        "A": "value_capture_at_500",
        "B1": "entry_value_capture_at_50",
        "B2": "conditional_value_capture_at_3",
    },
    "report_all_chain_task_headlines": 18,
    "aggregate": "unweighted mean of six unrounded chain values for each task",
    "chain_comparison": (
        "higher/equal/lower counts versus each trained family using unrounded values and exact equality"
    ),
    "partial_success_mean": "forbidden",
    "abstract_mention_rule": (
        "task mean strictly outside the closed interval of kge and nbfnet task means, and ULTRA "
        "is on that same side of both references in at least five of six chains"
    ),
    "abstract_if_no_task_passes": "omit ULTRA from abstract",
    "forbidden_claims": [
        "fair-compute comparison",
        "champion claim",
        "statistical significance claim",
    ],
}


class ProtocolError(RuntimeError):
    """A formal leakage, source, completion, or native-backend gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def stable_object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=8)
def _stable_file_sha256_cached(
    resolved_path: str, size_bytes: int, mtime_ns: int
) -> str:
    path = Path(resolved_path)
    before = path.stat()
    if before.st_size != size_bytes or before.st_mtime_ns != mtime_ns:
        raise ProtocolError(f"file changed before hashing: {path}")
    digest = sha256_file(path)
    after = path.stat()
    if after.st_size != size_bytes or after.st_mtime_ns != mtime_ns:
        raise ProtocolError(f"file changed while hashing: {path}")
    return digest


def stable_file_sha256(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _stable_file_sha256_cached(str(path), stat.st_size, stat.st_mtime_ns)


def normalized_lf_sha256(path: Path) -> str:
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Compute Git's content-addressed blob id without trusting a local clone."""
    path = Path(path)
    digest = hashlib.sha1()  # noqa: S324 - required to verify a declared Git object id
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
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


def stable_triple_hash(triples) -> str:
    digest = hashlib.sha256()
    for row in triples:
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _portable(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read {role} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{role} must be a JSON object: {path}")
    return value


def _strict_canonical_json(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    """Read one immutable formal marker without JSON or byte-level ambiguity."""

    path = Path(path)
    if path.is_symlink():
        raise ProtocolError(f"{role} must not be a symbolic link: {path}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"cannot read strict {role} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{role} must be a JSON object: {path}")
    if raw != canonical_json_bytes(value):
        raise ProtocolError(f"{role} bytes are not canonical compact JSON: {path}")
    return value, raw


def verify_scoring_started(
    marker_path: Path, manifest_path: Path, config: Mapping[str, object]
) -> tuple[dict[str, Any], str]:
    """Verify the coordinator-owned scoring marker and hash its actual bytes."""

    marker, raw = _strict_canonical_json(marker_path, "formal scoring-start marker")
    expected_fields = {
        "schema_version",
        "protocol",
        "run_id",
        "manifest_sha256",
        "started_at_utc",
        "policy",
    }
    if set(marker) != expected_fields:
        raise ProtocolError(
            "formal scoring-start marker fields are not exact: "
            f"observed={sorted(marker)}, expected={sorted(expected_fields)}"
        )
    expected = {
        "schema_version": SCORE_START_SCHEMA,
        "protocol": PROTOCOL,
        "run_id": config.get("run_id"),
        "manifest_sha256": sha256_file(manifest_path),
        "policy": SCORING_START_POLICY,
    }
    for field, wanted in expected.items():
        _assert_exact(marker.get(field), wanted, f"scoring-start marker {field}")
    started_at = marker.get("started_at_utc")
    if not isinstance(started_at, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)", started_at
    ):
        raise ProtocolError("scoring-start marker timestamp is invalid")
    return marker, hashlib.sha256(raw).hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError(f"artifact already exists and is immutable: {path}") from exc


def _write_csv_exclusive(path: Path, frame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(
            path,
            index=False,
            mode="x",
            lineterminator="\n",
            float_format="%.17g",
        )
    except FileExistsError as exc:
        raise ProtocolError(f"artifact already exists and is immutable: {path}") from exc


def _assert_exact(value: object, expected: object, role: str) -> None:
    if value != expected:
        raise ProtocolError(f"{role} mismatch: observed={value!r}, expected={expected!r}")


def validate_config_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol",
        "run_id",
        "execution_status",
        "chains",
        "fold",
        "aggregation",
        "candidate_root",
        "checkpoint",
        "provenance",
        "license",
        "training_disclosure",
        "overlap_policy",
        "raw_source_policy",
        "data_precommit_policy",
        "model_policy",
        "task_contract",
        "phase_order",
        "runtime_contract",
        "repeatability_contract",
        "reporting_contract",
        "output_root",
    }
    if set(payload) != required:
        raise ProtocolError(
            "config fields are not exact: "
            f"missing={sorted(required - set(payload))}, extra={sorted(set(payload) - required)}"
        )
    _assert_exact(payload.get("schema_version"), CONFIG_SCHEMA, "config schema")
    _assert_exact(payload.get("protocol"), PROTOCOL, "config protocol")
    _assert_exact(payload.get("run_id"), CANONICAL_RUN_ID, "config run_id")
    _assert_exact(payload.get("execution_status"), AUTHORIZED_STATUS, "execution status")
    _assert_exact(payload.get("chains"), list(CHAINS), "six-chain order")
    _assert_exact(payload.get("fold"), "main", "fold")
    _assert_exact(payload.get("aggregation"), "calendar_mean", "aggregation")
    _assert_exact(payload.get("candidate_root"), "data/processed_v2", "candidate root")
    _assert_exact(
        payload.get("checkpoint"),
        {
            "name": CANONICAL_CHECKPOINT_NAME,
            "path": CANONICAL_CHECKPOINT_PATH,
            "sha256": CANONICAL_CHECKPOINT_SHA256,
            "bytes": CANONICAL_CHECKPOINT_BYTES,
        },
        "fixed checkpoint",
    )
    _assert_exact(payload.get("provenance"), PROVENANCE, "upstream provenance")
    _assert_exact(payload.get("license"), LICENSE, "license provenance")
    _assert_exact(payload.get("training_disclosure"), TRAINING_DISCLOSURE, "training disclosure")
    _assert_exact(payload.get("overlap_policy"), OVERLAP_POLICY, "overlap policy")
    _assert_exact(payload.get("raw_source_policy"), RAW_SOURCE_POLICY, "raw-source policy")
    _assert_exact(
        payload.get("data_precommit_policy"), DATA_PRECOMMIT_POLICY, "data-precommit policy"
    )
    _assert_exact(payload.get("model_policy"), MODEL_POLICY, "model policy")
    _assert_exact(payload.get("task_contract"), TASK_CONTRACT, "task contract")
    _assert_exact(payload.get("phase_order"), PHASE_ORDER, "phase order")
    _assert_exact(payload.get("runtime_contract"), RUNTIME_CONTRACT, "runtime contract")
    _assert_exact(
        payload.get("repeatability_contract"),
        REPEATABILITY_CONTRACT,
        "repeatability contract",
    )
    _assert_exact(payload.get("reporting_contract"), REPORTING_CONTRACT, "reporting contract")
    reference = _resolve(REPORTING_CONTRACT["trained_reference_artifact"])
    if not reference.is_file():
        raise ProtocolError(f"trained-reference artifact is missing: {reference}")
    _assert_exact(
        sha256_file(reference),
        REPORTING_CONTRACT["trained_reference_artifact_sha256"],
        "trained-reference artifact hash",
    )
    output_root = payload.get("output_root")
    _assert_exact(output_root, "results_v2/ultra_formal", "canonical output_root")
    resolved = _resolve(output_root)
    if resolved == ROOT or resolved == Path(resolved.anchor):
        raise ProtocolError("output_root is unsafe")
    return dict(payload)


def load_and_validate_config(path: Path = CANONICAL_CONFIG) -> dict[str, Any]:
    return validate_config_payload(_load_json(Path(path), "formal ULTRA config"))


def _validate_provenance_files() -> None:
    evidence = {
        LICENSE["path"]: LICENSE["sha256"],
        TRAINING_DISCLOSURE["readme_path"]: TRAINING_DISCLOSURE["readme_sha256"],
        TRAINING_DISCLOSURE["vendored_reference_config_path"]: TRAINING_DISCLOSURE[
            "vendored_reference_config_sha256"
        ],
    }
    for logical, expected in evidence.items():
        path = _resolve(logical)
        if not path.is_file():
            raise ProtocolError(f"ULTRA provenance evidence is missing: {path}")
        _assert_exact(sha256_file(path), expected, f"ULTRA provenance evidence {logical}")
    for logical, expected in ULTRA_INFERENCE_SOURCE_LF_SHA256.items():
        path = ULTRA_ROOT / logical
        if not path.is_file():
            raise ProtocolError(f"normalized ULTRA source receipt file is missing: {path}")
        _assert_exact(
            normalized_lf_sha256(path),
            expected,
            f"normalized upstream ULTRA source {logical}",
        )


def _resolved_raw_baci_path() -> Path:
    from temporal_backtest import BACI_ZIP

    return Path(BACI_ZIP).resolve()


def verify_raw_source() -> dict[str, object]:
    """Verify raw BACI bytes without opening any target candidate columns."""
    attestation_path = _resolve(RAW_SOURCE_POLICY["attestation_path"])
    if not attestation_path.is_file():
        raise ProtocolError(f"raw-source attestation is missing: {attestation_path}")
    _assert_exact(
        sha256_file(attestation_path),
        RAW_SOURCE_POLICY["attestation_sha256"],
        "raw-source attestation sha256",
    )
    attestation = _load_json(attestation_path, "raw-source attestation")
    source = attestation.get("source")
    if not isinstance(source, Mapping):
        raise ProtocolError("raw-source attestation lacks source record")
    _assert_exact(source.get("archive_name"), RAW_SOURCE_POLICY["archive_name"], "BACI name")
    _assert_exact(source.get("size_bytes"), RAW_SOURCE_POLICY["archive_bytes"], "BACI bytes")
    _assert_exact(source.get("sha256"), RAW_SOURCE_POLICY["archive_sha256"], "BACI attested hash")
    archive = _resolved_raw_baci_path()
    if not archive.is_file():
        raise ProtocolError(f"temporal_backtest.BACI_ZIP is missing: {archive}")
    _assert_exact(archive.name, RAW_SOURCE_POLICY["archive_name"], "runtime BACI name")
    _assert_exact(archive.stat().st_size, RAW_SOURCE_POLICY["archive_bytes"], "runtime BACI bytes")
    _assert_exact(
        stable_file_sha256(archive), RAW_SOURCE_POLICY["archive_sha256"], "runtime BACI hash"
    )
    return {
        "archive_name": RAW_SOURCE_POLICY["archive_name"],
        "archive_bytes": RAW_SOURCE_POLICY["archive_bytes"],
        "archive_sha256": RAW_SOURCE_POLICY["archive_sha256"],
        "attestation_path": RAW_SOURCE_POLICY["attestation_path"],
        "attestation_sha256": RAW_SOURCE_POLICY["attestation_sha256"],
        "runtime_path_resolution": RAW_SOURCE_POLICY["runtime_path_resolution"],
        "host_path_serialized": False,
        "byte_hashing_is_semantic_target_access": False,
        "gravity_opened_by_formal_graph_builder": False,
    }


def _candidate_path(config: Mapping[str, object], chain: str, source: str) -> Path:
    if chain not in CHAINS or source not in SOURCE_SPECS:
        raise ProtocolError(f"unknown candidate cohort: chain={chain!r}, source={source!r}")
    root = _resolve(str(config["candidate_root"]))
    return root / SOURCE_SPECS[source]["filename"].format(chain=chain)


def read_candidate_identities(path: Path, source: str):
    """Read identity/protocol columns only; target columns never enter memory."""
    import pandas as pd

    if source not in SOURCE_SPECS:
        raise ProtocolError(f"unknown identity source {source!r}")
    if set(IDENTITY_COLUMNS) & FORBIDDEN_TARGET_COLUMNS:
        raise AssertionError("formal identity reader requests a forbidden target column")
    try:
        frame = pd.read_csv(
            path,
            usecols=list(IDENTITY_COLUMNS),
            dtype={column: "string" for column in IDENTITY_COLUMNS},
        )
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read identity-only candidate cohort {path}: {exc}") from exc
    if frame.empty or frame.isna().any().any():
        raise ProtocolError(f"identity-only candidate cohort is empty or contains nulls: {path}")
    expected = SOURCE_SPECS[source]
    metadata_expectations = {
        "aggregation": "calendar_mean",
        "early_window": "2008-2012",
        "late_window": "2018-2022",
        "task": expected["task"],
        "task_unit": expected["task_unit"],
    }
    for field, wanted in metadata_expectations.items():
        values = set(frame[field].astype(str).unique())
        if values != {wanted}:
            raise ProtocolError(
                f"{source} candidate {field} mismatch in {path}: {sorted(values)} != {[wanted]}"
            )
    versions = set(frame["benchmark_version"].astype(str).unique())
    if len(versions) != 1 or not next(iter(versions)).startswith("2"):
        raise ProtocolError(f"candidate benchmark_version is not one v2 value: {versions}")
    identities = frame.loc[:, list(KEYS)].astype(str).reset_index(drop=True)
    if identities.duplicated().any():
        raise ProtocolError(f"candidate identities are duplicated: {path}")
    ordered = identities.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    if not identities.equals(ordered):
        raise ProtocolError(f"candidate identities are not in deterministic key order: {path}")
    metadata = {
        field: str(frame[field].iloc[0])
        for field in IDENTITY_COLUMNS
        if field not in KEYS
    }
    return identities, metadata


def _load_early_graph(chain: str):
    from v2_ultra import load_current_early_graph

    return load_current_early_graph(chain)


def _validate_graph_coverage(identities, triples) -> dict[str, object]:
    from v2_ultra import validate_graph_coverage

    return validate_graph_coverage(identities, triples)


def validate_zero_candidate_early_trade_overlap(
    identities, triples, export_relations: Sequence[str]
) -> dict[str, int]:
    export_set = set(map(str, export_relations))
    early_trade = {
        (str(row[0]), str(row[2]), str(row[1]))
        for row in triples
        if str(row[1]) in export_set
    }
    candidate = set(
        identities.loc[:, list(KEYS)].itertuples(index=False, name=None)
    )
    overlap = candidate & early_trade
    if overlap:
        raise ProtocolError(
            f"candidate cohort overlaps {len(overlap)} observed early forward trade triples"
        )
    return {
        "candidate_rows": int(len(identities)),
        "early_forward_trade_triples": int(len(early_trade)),
        "overlap_rows": 0,
    }


def _source_hashes() -> dict[str, str]:
    local = {
        Path(__file__).resolve(),
        SRC / "v2_ultra.py",
        SRC / "v2_gpu_rolling.py",
        SRC / "v2_gpu_protocol.py",
        SRC / "benchmark.py",
        SRC / "temporal_backtest.py",
        SRC / "universe.py",
        SRC / "baci_filtered_cache.py",
        SRC / "window_aggregation.py",
        SRC / "task_features.py",
        SRC / "split.py",
        ROOT / "requirements" / "raw_source_attestation.json",
    }
    local.update(ROOT / "chains" / f"{chain}.json" for chain in CHAINS)
    suffixes = {".py", ".cpp", ".cu", ".h", ".cuh"}
    ultra = {
        path
        for path in (ULTRA_ROOT / "ultra").rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and "__pycache__" not in path.parts
    }
    paths = sorted(local | ultra, key=_portable)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ProtocolError(f"formal source files are missing: {missing}")
    return {_portable(path): sha256_file(path) for path in paths}


def _cohort_record(
    path: Path, identities, metadata: Mapping[str, str], coverage, early_trade_overlap
) -> dict[str, object]:
    return {
        "path": _portable(path),
        "full_file_bytes": int(path.stat().st_size),
        "full_file_sha256": stable_file_sha256(path),
        "full_file_hash_semantics": (
            "opaque byte precommit before scoring; hashing does not parse or inspect target columns"
        ),
        "rows": int(len(identities)),
        "identity_sha256": stable_key_hash(identities),
        "metadata": dict(metadata),
        "coverage": dict(coverage),
        "early_trade_overlap": dict(early_trade_overlap),
        "target_columns_semantically_accessed": False,
    }


def build_freeze_payload(config_path: Path = CANONICAL_CONFIG) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_and_validate_config(config_path)
    _validate_provenance_files()
    raw_source = verify_raw_source()
    checkpoint = _resolve(str(config["checkpoint"]["path"]))
    if not checkpoint.is_file():
        raise ProtocolError(f"fixed ULTRA checkpoint is missing: {checkpoint}")
    observed_checkpoint = sha256_file(checkpoint)
    _assert_exact(
        observed_checkpoint,
        str(config["checkpoint"]["sha256"]),
        "fixed checkpoint sha256",
    )
    _assert_exact(checkpoint.stat().st_size, CANONICAL_CHECKPOINT_BYTES, "fixed checkpoint bytes")
    _assert_exact(
        git_blob_sha1(checkpoint),
        PROVENANCE["checkpoint_git_blob_sha1"],
        "fixed checkpoint Git blob",
    )
    chains: dict[str, object] = {}
    for chain in CHAINS:
        identities = {}
        cohorts = {}
        for source in SOURCE_SPECS:
            path = _candidate_path(config, chain, source)
            frame, metadata = read_candidate_identities(path, source)
            identities[source] = frame
            cohorts[source] = (path, metadata)
        U, triples = _load_early_graph(chain)
        chain_cohorts = {}
        for source in SOURCE_SPECS:
            path, metadata = cohorts[source]
            coverage = _validate_graph_coverage(identities[source], triples)
            overlap = validate_zero_candidate_early_trade_overlap(
                identities[source], triples, U.EXPORT_RELATIONS
            )
            chain_cohorts[source] = _cohort_record(
                path, identities[source], metadata, coverage, overlap
            )
        a_keys = set(identities["A"].itertuples(index=False, name=None))
        b_keys = set(identities["B"].itertuples(index=False, name=None))
        cross_overlap = len(a_keys & b_keys)
        if cross_overlap:
            raise ProtocolError(f"A/B candidate identities overlap for {chain}: {cross_overlap}")
        chains[chain] = {
            "graph": {
                "policy": "main early graph only; 2008-2012 calendar mean",
                "sha256": stable_triple_hash(triples),
                "forward_triples": int(len(triples)),
            },
            "cohorts": chain_cohorts,
            "cohort_accounting": {
                "A_rows": int(len(identities["A"])),
                "B_rows": int(len(identities["B"])),
                "A_B_identity_overlap_rows": 0,
                "combined_unique_rows": int(len(identities["A"]) + len(identities["B"])),
            },
            "component_path": f"components/{chain}/component.json",
        }
    return {
        "schema_version": FREEZE_SCHEMA,
        "protocol": PROTOCOL,
        "status": "frozen_before_target_scoring",
        "created_at_utc": _utc_now(),
        "run_id": config["run_id"],
        "config": {"path": _portable(config_path), "sha256": sha256_file(config_path)},
        "checkpoint": {
            "name": CANONICAL_CHECKPOINT_NAME,
            "path": _portable(checkpoint),
            "sha256": observed_checkpoint,
            "bytes": checkpoint.stat().st_size,
        },
        "provenance": PROVENANCE,
        "license": LICENSE,
        "training_disclosure": TRAINING_DISCLOSURE,
        "overlap_policy": OVERLAP_POLICY,
        "raw_source": raw_source,
        "data_precommit_policy": DATA_PRECOMMIT_POLICY,
        "model_policy": MODEL_POLICY,
        "task_contract": TASK_CONTRACT,
        "runtime_contract": RUNTIME_CONTRACT,
        "repeatability_contract": REPEATABILITY_CONTRACT,
        "reporting_contract": REPORTING_CONTRACT,
        "main_target_labels_accessed": False,
        "source_sha256": _source_hashes(),
        "chains": chains,
    }


def _manifest_path(config: Mapping[str, object]) -> Path:
    return _resolve(str(config["output_root"])) / "frozen_manifest.json"


def require_exact_chain_mapping(chains: object) -> None:
    """Accept deterministic JSON key sorting while requiring exactly six chains."""
    if not isinstance(chains, Mapping) or set(chains) != set(CHAINS):
        raise ProtocolError("freeze is not exactly the canonical six-chain matrix")


def verify_freeze_manifest(
    path: Path, config_path: Path = CANONICAL_CONFIG
) -> dict[str, Any]:
    path = Path(path).resolve()
    config_path = Path(config_path).resolve()
    config = load_and_validate_config(config_path)
    _validate_provenance_files()
    raw_source = verify_raw_source()
    manifest = _load_json(path, "formal ULTRA freeze manifest")
    _assert_exact(manifest.get("schema_version"), FREEZE_SCHEMA, "freeze schema")
    _assert_exact(manifest.get("protocol"), PROTOCOL, "freeze protocol")
    _assert_exact(manifest.get("status"), "frozen_before_target_scoring", "freeze status")
    _assert_exact(manifest.get("run_id"), config["run_id"], "freeze run_id")
    _assert_exact(manifest.get("main_target_labels_accessed"), False, "freeze label access")
    config_ref = manifest.get("config")
    if not isinstance(config_ref, Mapping):
        raise ProtocolError("freeze lacks config binding")
    _assert_exact(config_ref.get("sha256"), sha256_file(config_path), "config sha256")
    checkpoint = _resolve(str(config["checkpoint"]["path"]))
    checkpoint_ref = manifest.get("checkpoint")
    if not isinstance(checkpoint_ref, Mapping):
        raise ProtocolError("freeze lacks checkpoint binding")
    _assert_exact(checkpoint_ref.get("sha256"), sha256_file(checkpoint), "checkpoint sha256")
    _assert_exact(checkpoint_ref.get("sha256"), CANONICAL_CHECKPOINT_SHA256, "checkpoint lock")
    _assert_exact(checkpoint_ref.get("bytes"), CANONICAL_CHECKPOINT_BYTES, "checkpoint bytes")
    _assert_exact(
        git_blob_sha1(checkpoint),
        PROVENANCE["checkpoint_git_blob_sha1"],
        "checkpoint Git blob",
    )
    _assert_exact(manifest.get("provenance"), PROVENANCE, "frozen provenance")
    _assert_exact(manifest.get("license"), LICENSE, "frozen license")
    _assert_exact(
        manifest.get("training_disclosure"), TRAINING_DISCLOSURE, "frozen training disclosure"
    )
    _assert_exact(manifest.get("overlap_policy"), OVERLAP_POLICY, "frozen overlap policy")
    _assert_exact(manifest.get("raw_source"), raw_source, "frozen raw BACI source")
    _assert_exact(
        manifest.get("data_precommit_policy"), DATA_PRECOMMIT_POLICY, "frozen data precommit"
    )
    _assert_exact(manifest.get("model_policy"), MODEL_POLICY, "frozen model policy")
    _assert_exact(manifest.get("task_contract"), TASK_CONTRACT, "frozen task contract")
    _assert_exact(manifest.get("runtime_contract"), RUNTIME_CONTRACT, "runtime contract")
    _assert_exact(
        manifest.get("repeatability_contract"),
        REPEATABILITY_CONTRACT,
        "repeatability contract",
    )
    _assert_exact(manifest.get("reporting_contract"), REPORTING_CONTRACT, "reporting contract")
    _assert_exact(manifest.get("source_sha256"), _source_hashes(), "source hashes")
    chains = manifest.get("chains")
    require_exact_chain_mapping(chains)
    for chain in CHAINS:
        record = chains.get(chain)
        if not isinstance(record, Mapping):
            raise ProtocolError(f"freeze chain record is invalid: {chain}")
        _assert_exact(
            record.get("component_path"),
            f"components/{chain}/component.json",
            f"{chain} component path",
        )
        cohorts = record.get("cohorts")
        if not isinstance(cohorts, Mapping) or set(cohorts) != set(SOURCE_SPECS):
            raise ProtocolError(f"freeze lacks exact A/B cohorts for {chain}")
        for source in SOURCE_SPECS:
            cohort = cohorts[source]
            if not isinstance(cohort, Mapping) or int(cohort.get("rows", 0)) < 1:
                raise ProtocolError(f"freeze cohort is invalid for {chain}/{source}")
            digest = cohort.get("identity_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ProtocolError(f"freeze identity hash is invalid for {chain}/{source}")
            full_digest = cohort.get("full_file_sha256")
            if not isinstance(full_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", full_digest):
                raise ProtocolError(f"freeze full-file hash is invalid for {chain}/{source}")
            if int(cohort.get("full_file_bytes", 0)) < 1:
                raise ProtocolError(f"freeze full-file size is invalid for {chain}/{source}")
            _assert_exact(
                cohort.get("target_columns_semantically_accessed"),
                False,
                f"{chain}/{source} frozen target access",
            )
            overlap = cohort.get("early_trade_overlap")
            if not isinstance(overlap, Mapping) or overlap.get("overlap_rows") != 0:
                raise ProtocolError(f"freeze does not prove zero early-trade overlap: {chain}/{source}")
        expected_accounting = {
            "A_rows": int(cohorts["A"]["rows"]),
            "B_rows": int(cohorts["B"]["rows"]),
            "A_B_identity_overlap_rows": 0,
            "combined_unique_rows": int(cohorts["A"]["rows"]) + int(cohorts["B"]["rows"]),
        }
        _assert_exact(record.get("cohort_accounting"), expected_accounting, f"{chain} accounting")
    return manifest


def verify_chain_inputs(
    manifest: Mapping[str, object], config: Mapping[str, object], chain: str
):
    frozen = manifest["chains"][chain]
    identities = {}
    for source in SOURCE_SPECS:
        path = _candidate_path(config, chain, source)
        record = frozen["cohorts"][source]
        _assert_exact(path.stat().st_size, record["full_file_bytes"], f"{chain}/{source} full bytes")
        _assert_exact(
            stable_file_sha256(path),
            record["full_file_sha256"],
            f"{chain}/{source} full-file precommit",
        )
        frame, metadata = read_candidate_identities(path, source)
        identities[source] = frame
        _assert_exact(stable_key_hash(frame), record["identity_sha256"], f"{chain}/{source} identity hash")
        _assert_exact(int(len(frame)), int(record["rows"]), f"{chain}/{source} rows")
        _assert_exact(metadata, record["metadata"], f"{chain}/{source} metadata")
    U, triples = _load_early_graph(chain)
    cross_overlap = len(
        set(identities["A"].itertuples(index=False, name=None))
        & set(identities["B"].itertuples(index=False, name=None))
    )
    _assert_exact(cross_overlap, 0, f"{chain} A/B identity overlap")
    _assert_exact(stable_triple_hash(triples), frozen["graph"]["sha256"], f"{chain} graph hash")
    for source in SOURCE_SPECS:
        coverage = _validate_graph_coverage(identities[source], triples)
        _assert_exact(coverage, frozen["cohorts"][source]["coverage"], f"{chain}/{source} coverage")
        overlap = validate_zero_candidate_early_trade_overlap(
            identities[source], triples, U.EXPORT_RELATIONS
        )
        _assert_exact(
            overlap,
            frozen["cohorts"][source]["early_trade_overlap"],
            f"{chain}/{source} early-trade overlap",
        )
    return U, triples, identities


def _claim_marker(path: Path, payload: Mapping[str, object], identity_fields: Sequence[str]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(payload))
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _load_json(path, "formal phase marker")
        for field in identity_fields:
            _assert_exact(existing.get(field), payload.get(field), f"existing marker {field}")
        return existing
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return dict(payload)


def _require_native_torch_scatter() -> dict[str, object]:
    """Reject the in-process smoke shim and require an installed package."""
    if "torch_scatter" in sys.modules:
        existing = sys.modules["torch_scatter"]
        if getattr(existing, "__spec__", None) is None or not getattr(existing, "__file__", None):
            raise ProtocolError("formal ULTRA forbids the synthetic torch_scatter compatibility shim")
    try:
        module = importlib.import_module("torch_scatter")
    except ImportError as exc:
        raise ProtocolError("formal ULTRA requires the native torch-scatter package") from exc
    origin = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    scatter = getattr(module, "scatter", None)
    if not origin or spec is None or scatter is None:
        raise ProtocolError("torch_scatter is not a native installed package")
    if getattr(scatter, "__module__", "") == "v2_ultra":
        raise ProtocolError("formal ULTRA forbids v2_ultra's torch_scatter compatibility shim")
    try:
        version = importlib.metadata.version("torch-scatter")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProtocolError("native torch-scatter distribution metadata is missing") from exc
    origin_path = Path(origin).resolve()
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    native_extensions = sorted(
        path
        for path in origin_path.parent.iterdir()
        if path.is_file() and path.name.endswith(extension_suffixes)
    )
    if not native_extensions:
        raise ProtocolError("torch-scatter package has no native extension binaries")
    return {
        "package": "torch-scatter",
        "version": version,
        "origin_name": origin_path.name,
        "origin_sha256": sha256_file(origin_path),
        "native_extension_sha256": {
            path.name: sha256_file(path) for path in native_extensions
        },
    }


def _vendored_ultra_module_receipt(module_name: str) -> dict[str, str]:
    if module_name not in CANONICAL_VENDORED_MODULE_PATHS:
        raise ProtocolError(f"unrecognized vendored ULTRA module: {module_name}")
    module = importlib.import_module(module_name)
    origin = getattr(module, "__file__", None)
    if not origin:
        raise ProtocolError(f"vendored ULTRA module lacks file origin: {module_name}")
    path = Path(origin).resolve()
    try:
        relative = path.relative_to(ULTRA_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ProtocolError(
            f"ULTRA module resolved outside the vendored source root: {module_name} -> {path}"
        ) from exc
    _assert_exact(
        relative,
        CANONICAL_VENDORED_MODULE_PATHS[module_name],
        f"canonical vendored module path for {module_name}",
    )
    return {
        "module": module_name,
        "vendored_relative_path": relative,
        "sha256": sha256_file(path),
    }


def require_native_backend(device: str) -> tuple[object, dict[str, object]]:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch

    if not device.startswith("cuda"):
        raise ProtocolError("formal ULTRA scoring requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise ProtocolError("formal ULTRA scoring requires torch.cuda.is_available()")
    requested = torch.device(device)
    ordinal = torch.cuda.current_device() if requested.index is None else int(requested.index)
    if ordinal >= torch.cuda.device_count():
        raise ProtocolError(f"CUDA device does not exist: {device}")
    scatter = _require_native_torch_scatter()
    sys.path[:] = [entry for entry in sys.path if Path(str(entry or os.curdir)).resolve() != ULTRA_ROOT.resolve()]
    sys.path.insert(0, str(ULTRA_ROOT))
    vendored_modules = {
        name: _vendored_ultra_module_receipt(name)
        for name in ("ultra", "ultra.layers", "ultra.models", "ultra.tasks")
    }
    from ultra.layers import GeneralizedRelationalConv

    propagate = GeneralizedRelationalConv.propagate
    if getattr(propagate, "__module__", None) != "ultra.layers" or not getattr(
        propagate, "__qualname__", ""
    ).startswith("GeneralizedRelationalConv."):
        raise ProtocolError("formal ULTRA forbids the MessagePassing fallback monkeypatch")
    # Importing this module builds/loads ULTRA's official rspmm extension.  A
    # pure Python or compatibility replacement is not accepted.
    rspmm_module = importlib.import_module("ultra.rspmm.rspmm")
    vendored_modules["ultra.rspmm.rspmm"] = _vendored_ultra_module_receipt(
        "ultra.rspmm.rspmm"
    )
    extension = getattr(rspmm_module, "rspmm", None)
    extension_origin = getattr(extension, "__file__", None)
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if not extension_origin or not str(extension_origin).endswith(extension_suffixes):
        raise ProtocolError("formal ULTRA requires the compiled native rspmm extension")
    capability = torch.cuda.get_device_capability(ordinal)
    driver_version = None
    driver_getter = getattr(torch.cuda, "driver_version", None)
    if callable(driver_getter):
        try:
            driver_version = str(driver_getter())
        except Exception:
            driver_version = None
    return torch.device(device), {
        "backend": RUNTIME_CONTRACT["backend"],
        "torch_scatter": scatter,
        "vendored_ultra_modules": vendored_modules,
        "rspmm_extension_name": Path(extension_origin).name,
        "rspmm_extension_sha256": sha256_file(Path(extension_origin)),
        "message_passing_fallback": False,
        "compatibility_shim": False,
        "device": device,
        "device_name": torch.cuda.get_device_name(ordinal),
        "device_capability": [int(capability[0]), int(capability[1])],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvidia_driver_via_torch": driver_version,
        "torch_geometric": importlib.metadata.version("torch-geometric"),
    }


def _load_native_ultra(checkpoint: Path, device: str):
    """Load the fixed architecture without changing ULTRA's propagate method."""
    import torch
    from ultra.models import Ultra

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
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, Mapping) or "model" not in state:
        raise ProtocolError(f"checkpoint does not contain a model state: {checkpoint}")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def _score_paths(output_root: Path, chain: str) -> dict[str, Path]:
    component_root = output_root / "components" / chain
    return {
        "A": component_root / "scores_A.csv",
        "B": component_root / "scores_B.csv",
        "A_repeat": component_root / "scores_A_repeat.csv",
        "B_repeat": component_root / "scores_B_repeat.csv",
        "component": component_root / "component.json",
    }


def project_union_scores(identities, union_scores):
    """Select one frozen cohort from a scored A/B union without accepting gaps.

    ``align_scores_exact`` deliberately rejects extras.  The formal scorer first
    scores the de-duplicated A/B union once, so this projection performs an
    indexed subset and then invokes the strict bijective alignment gate.
    """
    import pandas as pd
    from v2_ultra import align_scores_exact

    target = identities.loc[:, list(KEYS)].astype(str).reset_index(drop=True)
    scored = union_scores.loc[:, list(KEYS) + ["ultra_score"]].copy()
    scored.loc[:, list(KEYS)] = scored.loc[:, list(KEYS)].astype(str)
    if scored.loc[:, list(KEYS)].duplicated().any():
        raise ProtocolError("scored A/B union contains duplicate identities")
    target_index = pd.MultiIndex.from_frame(target, names=list(KEYS))
    indexed = scored.set_index(list(KEYS))
    missing = target_index.difference(indexed.index)
    if len(missing):
        raise ProtocolError(f"scored A/B union is missing {len(missing)} target identities")
    projected = indexed.loc[target_index].reset_index()
    return align_scores_exact(target, projected)


def stable_score_vector_hash(scored) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(stable_key_hash(scored).encode("ascii"))
    values = np.asarray(scored["ultra_score"], dtype="<f8")
    if not np.isfinite(values).all():
        raise ProtocolError("cannot hash non-finite ULTRA scores")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def score_repeatability(primary, repeat) -> dict[str, object]:
    import numpy as np

    if not primary.loc[:, list(KEYS)].equals(repeat.loc[:, list(KEYS)]):
        raise ProtocolError("repeat score identities differ from primary score identities")
    first = primary["ultra_score"].to_numpy(dtype=np.float64)
    second = repeat["ultra_score"].to_numpy(dtype=np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ProtocolError("repeat score arrays are invalid")
    delta = np.abs(first - second)
    exact_hash_primary = stable_score_vector_hash(primary)
    exact_hash_repeat = stable_score_vector_hash(repeat)
    rtol = float(REPEATABILITY_CONTRACT["score_acceptance_rtol"])
    atol = float(REPEATABILITY_CONTRACT["score_acceptance_atol"])
    numeric_allclose = bool(np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=False))
    return {
        "primary_score_vector_sha256": exact_hash_primary,
        "repeat_score_vector_sha256": exact_hash_repeat,
        "exact_hash_equality": exact_hash_primary == exact_hash_repeat,
        "max_absolute_score_delta": float(delta.max()) if len(delta) else 0.0,
        "mean_absolute_score_delta": float(delta.mean()) if len(delta) else 0.0,
        "numeric_acceptance_rtol": rtol,
        "numeric_acceptance_atol": atol,
        "numeric_allclose": numeric_allclose,
    }


def _score_chain(args) -> int:
    import random
    import numpy as np
    import pandas as pd
    import torch
    from v2_ultra import align_scores_exact, build_ultra_graph, score_complete_candidates

    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path)
    manifest_path = _manifest_path(config)
    manifest = verify_freeze_manifest(manifest_path, config_path)
    manifest_sha256 = sha256_file(manifest_path)
    output_root = _resolve(str(config["output_root"]))
    paths = _score_paths(output_root, args.chain)
    if any(path.exists() for path in paths.values()):
        raise ProtocolError(f"refusing to overwrite an existing score component: {args.chain}")
    verify_scoring_started(output_root / "SCORING_STARTED.json", manifest_path, config)
    U, triples, identities = verify_chain_inputs(manifest, config, args.chain)
    device, backend = require_native_backend(args.device)
    inference_seed = int(REPEATABILITY_CONTRACT["inference_seed"])
    random.seed(inference_seed)
    np.random.seed(inference_seed)
    torch.manual_seed(inference_seed)
    torch.cuda.manual_seed_all(inference_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    backend["inference_seed"] = inference_seed
    backend["inference_seed_scope"] = REPEATABILITY_CONTRACT["inference_seed_scope"]
    backend["numeric_policy"] = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "global_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    _assert_exact(
        backend["numeric_policy"],
        REPEATABILITY_CONTRACT["numeric_policy"],
        "frozen inference numeric policy",
    )
    checkpoint = _resolve(str(config["checkpoint"]["path"]))
    started = time.perf_counter()
    torch.empty(1, device=device)
    ordinal = torch.cuda.current_device() if device.index is None else int(device.index)
    torch.cuda.reset_peak_memory_stats(ordinal)
    model = _load_native_ultra(checkpoint, str(device))
    data, entity, relation = build_ultra_graph(triples, U.EXPORT_RELATIONS, str(device))
    combined = (
        pd.concat([identities["A"], identities["B"]], ignore_index=True)
        .drop_duplicates(list(KEYS))
        .sort_values(list(KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    random.seed(inference_seed)
    np.random.seed(inference_seed)
    torch.manual_seed(inference_seed)
    torch.cuda.manual_seed_all(inference_seed)
    primary_started = time.perf_counter()
    raw = score_complete_candidates(
        model,
        data,
        combined,
        entity,
        relation,
        device=str(device),
        batch_groups=int(RUNTIME_CONTRACT["batch_query_groups"]),
    )
    torch.cuda.synchronize(device)
    primary_seconds = time.perf_counter() - primary_started
    aligned_union = align_scores_exact(combined, raw)
    score_outputs = {
        source: project_union_scores(identities[source], aligned_union)
        for source in SOURCE_SPECS
    }
    repeat_outputs = None
    repeat_diagnostic = {
        "required": False,
        "sentinel_chain": REPEATABILITY_CONTRACT["sentinel_chain"],
    }
    repeat_seconds = None
    if args.chain == REPEATABILITY_CONTRACT["sentinel_chain"]:
        random.seed(inference_seed)
        np.random.seed(inference_seed)
        torch.manual_seed(inference_seed)
        torch.cuda.manual_seed_all(inference_seed)
        repeat_started = time.perf_counter()
        raw_repeat = score_complete_candidates(
            model,
            data,
            combined,
            entity,
            relation,
            device=str(device),
            batch_groups=int(RUNTIME_CONTRACT["batch_query_groups"]),
        )
        torch.cuda.synchronize(device)
        repeat_seconds = time.perf_counter() - repeat_started
        aligned_union_repeat = align_scores_exact(combined, raw_repeat)
        repeat_diagnostic = {
            "required": True,
            "contract": REPEATABILITY_CONTRACT,
            **score_repeatability(aligned_union, aligned_union_repeat),
        }
        if not repeat_diagnostic["numeric_allclose"]:
            raise ProtocolError(
                "same-process sheep repeat scores exceed the frozen numeric acceptance gate"
            )
        repeat_outputs = {
            source: project_union_scores(identities[source], aligned_union_repeat)
            for source in SOURCE_SPECS
        }
    for source in SOURCE_SPECS:
        _write_csv_exclusive(paths[source], score_outputs[source])
        roundtrip = _read_score_file(paths[source], identities[source])
        _assert_exact(
            stable_score_vector_hash(roundtrip),
            stable_score_vector_hash(score_outputs[source]),
            f"{args.chain}/{source} deterministic CSV round trip",
        )
    if repeat_outputs is not None:
        for source in SOURCE_SPECS:
            repeat_path = paths[f"{source}_repeat"]
            _write_csv_exclusive(repeat_path, repeat_outputs[source])
            roundtrip = _read_score_file(repeat_path, identities[source])
            _assert_exact(
                stable_score_vector_hash(roundtrip),
                stable_score_vector_hash(repeat_outputs[source]),
                f"{args.chain}/{source} repeat deterministic CSV round trip",
            )
    score_refs = {
        source: {
            "path": _portable(paths[source]),
            "sha256": sha256_file(paths[source]),
            "rows": int(len(score_outputs[source])),
            "identity_sha256": stable_key_hash(score_outputs[source]),
            "score_vector_sha256": stable_score_vector_hash(score_outputs[source]),
            "column": "ultra_score",
        }
        for source in SOURCE_SPECS
    }
    repeat_score_refs = None
    if repeat_outputs is not None:
        repeat_score_refs = {
            source: {
                "path": _portable(paths[f"{source}_repeat"]),
                "sha256": sha256_file(paths[f"{source}_repeat"]),
                "rows": int(len(repeat_outputs[source])),
                "identity_sha256": stable_key_hash(repeat_outputs[source]),
                "score_vector_sha256": stable_score_vector_hash(repeat_outputs[source]),
                "column": "ultra_score",
            }
            for source in SOURCE_SPECS
        }
        repeat_diagnostic["exact_file_hash_equality_by_source"] = {
            source: score_refs[source]["sha256"] == repeat_score_refs[source]["sha256"]
            for source in SOURCE_SPECS
        }
    component = {
        "schema_version": COMPONENT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete_label_blind_scores",
        "run_id": config["run_id"],
        "chain": args.chain,
        "created_at_utc": _utc_now(),
        "manifest_sha256": manifest_sha256,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "graph_sha256": stable_triple_hash(triples),
        "candidate_precommit": {
            source: {
                "full_file_bytes": manifest["chains"][args.chain]["cohorts"][source][
                    "full_file_bytes"
                ],
                "full_file_sha256": manifest["chains"][args.chain]["cohorts"][source][
                    "full_file_sha256"
                ],
                "byte_hashing_is_semantic_target_access": False,
            }
            for source in SOURCE_SPECS
        },
        "main_target_labels_accessed": False,
        "main_label_derived_columns_accessed": False,
        "training_or_fine_tuning_performed": False,
        "selection_performed": False,
        "combined_unique_lane_rows": int(len(combined)),
        "scores": score_refs,
        "repeat_scores": repeat_score_refs,
        "repeatability": repeat_diagnostic,
        "native_backend": backend,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(ordinal)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scoring_seconds": {
            "primary_run1": round(primary_seconds, 3),
            "repeat_run2": None if repeat_seconds is None else round(repeat_seconds, 3),
        },
        "source_sha256": manifest["source_sha256"],
    }
    _write_json_exclusive(paths["component"], component)
    print(json.dumps({"component": str(paths["component"]), "scores": score_refs}, indent=2))
    return 0


def _read_score_file(path: Path, expected_identities):
    import pandas as pd
    from v2_ultra import align_scores_exact

    try:
        scored = pd.read_csv(
            path,
            usecols=list(KEYS) + ["ultra_score"],
            dtype={key: str for key in KEYS},
            float_precision="round_trip",
        )
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read formal ULTRA score file {path}: {exc}") from exc
    return align_scores_exact(expected_identities, scored)


def verify_component(
    manifest_path: Path, config_path: Path, chain: str
) -> dict[str, Any]:
    import pandas as pd

    config = load_and_validate_config(config_path)
    manifest = verify_freeze_manifest(manifest_path, config_path)
    manifest_sha256 = sha256_file(manifest_path)
    output_root = _resolve(str(config["output_root"]))
    paths = _score_paths(output_root, chain)
    component = _load_json(paths["component"], f"{chain} ULTRA score component")
    _assert_exact(component.get("schema_version"), COMPONENT_SCHEMA, f"{chain} component schema")
    _assert_exact(component.get("protocol"), PROTOCOL, f"{chain} component protocol")
    _assert_exact(component.get("status"), "complete_label_blind_scores", f"{chain} component status")
    _assert_exact(component.get("chain"), chain, f"{chain} component identity")
    _assert_exact(component.get("run_id"), config["run_id"], f"{chain} run_id")
    _assert_exact(component.get("manifest_sha256"), manifest_sha256, f"{chain} manifest hash")
    _assert_exact(component.get("config_sha256"), sha256_file(config_path), f"{chain} config hash")
    _assert_exact(component.get("checkpoint_sha256"), CANONICAL_CHECKPOINT_SHA256, f"{chain} checkpoint")
    _assert_exact(
        component.get("graph_sha256"),
        manifest["chains"][chain]["graph"]["sha256"],
        f"{chain} graph hash",
    )
    _assert_exact(component.get("source_sha256"), manifest["source_sha256"], f"{chain} sources")
    for field in (
        "main_target_labels_accessed",
        "main_label_derived_columns_accessed",
        "training_or_fine_tuning_performed",
        "selection_performed",
    ):
        _assert_exact(component.get(field), False, f"{chain} {field}")
    backend = component.get("native_backend")
    if not isinstance(backend, Mapping):
        raise ProtocolError(f"{chain} component lacks native backend attestation")
    _assert_exact(backend.get("backend"), RUNTIME_CONTRACT["backend"], f"{chain} native backend")
    _assert_exact(backend.get("message_passing_fallback"), False, f"{chain} fallback")
    _assert_exact(backend.get("compatibility_shim"), False, f"{chain} shim")
    _assert_exact(
        backend.get("inference_seed"), REPEATABILITY_CONTRACT["inference_seed"], f"{chain} seed"
    )
    _assert_exact(
        backend.get("inference_seed_scope"),
        REPEATABILITY_CONTRACT["inference_seed_scope"],
        f"{chain} seed scope",
    )
    _assert_exact(
        backend.get("numeric_policy"),
        REPEATABILITY_CONTRACT["numeric_policy"],
        f"{chain} numeric policy",
    )
    module_receipts = backend.get("vendored_ultra_modules")
    expected_module_names = set(CANONICAL_VENDORED_MODULE_PATHS)
    if not isinstance(module_receipts, Mapping) or set(module_receipts) != expected_module_names:
        raise ProtocolError(f"{chain} component lacks exact vendored ULTRA module receipts")
    for name, receipt in module_receipts.items():
        if not isinstance(receipt, Mapping) or receipt.get("module") != name:
            raise ProtocolError(f"{chain} invalid vendored module receipt: {name}")
        relative = str(receipt.get("vendored_relative_path", ""))
        _assert_exact(
            relative,
            CANONICAL_VENDORED_MODULE_PATHS[name],
            f"{chain} canonical vendored module path for {name}",
        )
        path = ULTRA_ROOT / relative
        if not path.is_file() or receipt.get("sha256") != sha256_file(path):
            raise ProtocolError(f"{chain} vendored module receipt changed: {name}")
    scatter = backend.get("torch_scatter")
    if not isinstance(scatter, Mapping) or not re.fullmatch(
        r"[0-9a-f]{64}", str(scatter.get("origin_sha256", ""))
    ):
        raise ProtocolError(f"{chain} component lacks a hash-bound native torch-scatter runtime")
    scatter_extensions = scatter.get("native_extension_sha256")
    if not isinstance(scatter_extensions, Mapping) or not scatter_extensions or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in scatter_extensions.values()
    ):
        raise ProtocolError(f"{chain} component lacks hash-bound torch-scatter binaries")
    if not re.fullmatch(r"[0-9a-f]{64}", str(backend.get("rspmm_extension_sha256", ""))):
        raise ProtocolError(f"{chain} component lacks a hash-bound native rspmm runtime")
    identities = {}
    scored_by_source = {}
    for source in SOURCE_SPECS:
        candidate = _candidate_path(config, chain, source)
        frozen = manifest["chains"][chain]["cohorts"][source]
        _assert_exact(candidate.stat().st_size, frozen["full_file_bytes"], f"{chain}/{source} bytes")
        _assert_exact(
            stable_file_sha256(candidate),
            frozen["full_file_sha256"],
            f"{chain}/{source} precommitted candidate hash",
        )
        _assert_exact(
            component.get("candidate_precommit", {}).get(source),
            {
                "full_file_bytes": frozen["full_file_bytes"],
                "full_file_sha256": frozen["full_file_sha256"],
                "byte_hashing_is_semantic_target_access": False,
            },
            f"{chain}/{source} component candidate precommit",
        )
        identities[source], _metadata = read_candidate_identities(candidate, source)
        _assert_exact(stable_key_hash(identities[source]), frozen["identity_sha256"], f"{chain}/{source} identity")
        score_ref = component.get("scores", {}).get(source)
        if not isinstance(score_ref, Mapping):
            raise ProtocolError(f"{chain} component lacks {source} scores")
        _assert_exact(score_ref.get("path"), _portable(paths[source]), f"{chain}/{source} score path")
        _assert_exact(score_ref.get("sha256"), sha256_file(paths[source]), f"{chain}/{source} score hash")
        scored = _read_score_file(paths[source], identities[source])
        scored_by_source[source] = scored
        _assert_exact(score_ref.get("rows"), int(len(scored)), f"{chain}/{source} score-ref rows")
        _assert_exact(
            score_ref.get("identity_sha256"),
            stable_key_hash(scored),
            f"{chain}/{source} score-ref identity",
        )
        _assert_exact(score_ref.get("column"), "ultra_score", f"{chain}/{source} score column")
        _assert_exact(
            score_ref.get("score_vector_sha256"),
            stable_score_vector_hash(scored),
            f"{chain}/{source} score vector",
        )
        _assert_exact(int(len(scored)), int(frozen["rows"]), f"{chain}/{source} score rows")
        _assert_exact(stable_key_hash(scored), frozen["identity_sha256"], f"{chain}/{source} score identities")
    combined = (
        pd.concat([identities["A"], identities["B"]], ignore_index=True)
        .drop_duplicates(list(KEYS))
        .sort_values(list(KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    _assert_exact(
        component.get("combined_unique_lane_rows"), int(len(combined)), f"{chain} combined rows"
    )
    _assert_exact(
        int(len(combined)),
        manifest["chains"][chain]["cohort_accounting"]["combined_unique_rows"],
        f"{chain} frozen combined rows",
    )
    repeatability = component.get("repeatability")
    if chain == REPEATABILITY_CONTRACT["sentinel_chain"]:
        if not isinstance(repeatability, Mapping) or repeatability.get("required") is not True:
            raise ProtocolError("sheep component lacks the required repeatability gate")
        repeat_by_source = {}
        repeat_refs = component.get("repeat_scores")
        if not isinstance(repeat_refs, Mapping) or set(repeat_refs) != set(SOURCE_SPECS):
            raise ProtocolError("sheep component lacks exact repeat A/B score references")
        for source in SOURCE_SPECS:
            repeat_path = paths[f"{source}_repeat"]
            ref = repeat_refs[source]
            repeated = _read_score_file(repeat_path, identities[source])
            repeat_by_source[source] = repeated
            _assert_exact(ref.get("path"), _portable(repeat_path), f"sheep/{source} repeat path")
            _assert_exact(ref.get("sha256"), sha256_file(repeat_path), f"sheep/{source} repeat hash")
            _assert_exact(ref.get("rows"), int(len(repeated)), f"sheep/{source} repeat rows")
            _assert_exact(ref.get("identity_sha256"), stable_key_hash(repeated), f"sheep/{source} repeat identity")
            _assert_exact(ref.get("column"), "ultra_score", f"sheep/{source} repeat column")
            _assert_exact(
                ref.get("score_vector_sha256"),
                stable_score_vector_hash(repeated),
                f"sheep/{source} repeat vector",
            )
        primary_union = (
            pd.concat([scored_by_source["A"], scored_by_source["B"]], ignore_index=True)
            .drop_duplicates(list(KEYS))
            .sort_values(list(KEYS), kind="mergesort")
            .reset_index(drop=True)
        )
        repeat_union = (
            pd.concat([repeat_by_source["A"], repeat_by_source["B"]], ignore_index=True)
            .drop_duplicates(list(KEYS))
            .sort_values(list(KEYS), kind="mergesort")
            .reset_index(drop=True)
        )
        observed_repeat = score_repeatability(primary_union, repeat_union)
        for field, value in observed_repeat.items():
            _assert_exact(repeatability.get(field), value, f"sheep repeatability {field}")
        _assert_exact(repeatability.get("contract"), REPEATABILITY_CONTRACT, "sheep repeat contract")
        expected_file_equality = {
            source: component["scores"][source]["sha256"]
            == component["repeat_scores"][source]["sha256"]
            for source in SOURCE_SPECS
        }
        _assert_exact(
            repeatability.get("exact_file_hash_equality_by_source"),
            expected_file_equality,
            "sheep repeat file equality",
        )
        _assert_exact(observed_repeat["numeric_allclose"], True, "sheep repeat score gate")
    else:
        _assert_exact(component.get("repeat_scores"), None, f"{chain} unexpected repeat scores")
        _assert_exact(
            repeatability,
            {"required": False, "sentinel_chain": REPEATABILITY_CONTRACT["sentinel_chain"]},
            f"{chain} repeat policy",
        )
    return component


def require_complete_component_set(chains: Sequence[str]) -> None:
    observed = list(chains)
    if observed != list(CHAINS):
        missing = sorted(set(CHAINS) - set(observed))
        extra = sorted(set(observed) - set(CHAINS))
        raise ProtocolError(
            f"target labels remain locked until exact six-chain score completion: "
            f"missing={missing}, extra={extra}, order={observed}"
        )


def _seal_scores(args) -> int:
    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path)
    manifest_path = _manifest_path(config)
    verify_freeze_manifest(manifest_path, config_path)
    manifest_sha256 = sha256_file(manifest_path)
    output_root = _resolve(str(config["output_root"]))
    entries = []
    runtime_hashes = set()
    for chain in CHAINS:
        component = verify_component(manifest_path, config_path, chain)
        component_path = _score_paths(output_root, chain)["component"]
        runtime = dict(component["native_backend"])
        # Device ordinal/name and timestamps may differ across workers; the
        # actual software/backend identity must be identical across all chains.
        runtime.pop("device", None)
        runtime.pop("device_name", None)
        runtime_sha256 = stable_object_sha256(runtime)
        runtime_hashes.add(runtime_sha256)
        entries.append(
            {
                "chain": chain,
                "path": _portable(component_path),
                "sha256": sha256_file(component_path),
                "A_score_sha256": component["scores"]["A"]["sha256"],
                "B_score_sha256": component["scores"]["B"]["sha256"],
                "A_repeat_score_sha256": (
                    component["repeat_scores"]["A"]["sha256"]
                    if component["repeat_scores"] is not None
                    else None
                ),
                "B_repeat_score_sha256": (
                    component["repeat_scores"]["B"]["sha256"]
                    if component["repeat_scores"] is not None
                    else None
                ),
                "repeatability": component["repeatability"],
                "native_runtime_sha256": runtime_sha256,
            }
        )
    require_complete_component_set([entry["chain"] for entry in entries])
    if len(runtime_hashes) != 1:
        raise ProtocolError(
            "six-chain formal components used different native software/backend runtimes"
        )
    _scoring_started, scoring_started_sha256 = verify_scoring_started(
        output_root / "SCORING_STARTED.json", manifest_path, config
    )
    seal = {
        "schema_version": SCORE_SEAL_SCHEMA,
        "protocol": PROTOCOL,
        "status": "all_six_chains_scored_labels_unlocked",
        "run_id": config["run_id"],
        "created_at_utc": _utc_now(),
        "manifest_sha256": manifest_sha256,
        "scoring_started_sha256": scoring_started_sha256,
        "component_count": len(entries),
        "components": entries,
        "native_runtime_sha256": next(iter(runtime_hashes)),
        "repeatability_contract": REPEATABILITY_CONTRACT,
        "sentinel_repeat_verified_before_label_unlock": True,
        "main_target_labels_accessed_before_seal": False,
        "unlock_policy": SCORE_UNLOCK_POLICY,
    }
    seal_path = output_root / "SCORES_COMPLETE.json"
    _write_json_exclusive(seal_path, seal)
    print(f"six-chain ULTRA score seal complete; target-label evaluation unlocked: {seal_path}")
    return 0


def verify_score_seal(
    seal_path: Path, manifest_path: Path, config_path: Path
) -> dict[str, Any]:
    config = load_and_validate_config(config_path)
    verify_freeze_manifest(manifest_path, config_path)
    output_root = _resolve(str(config["output_root"]))
    _scoring_started, scoring_started_sha256 = verify_scoring_started(
        output_root / "SCORING_STARTED.json", manifest_path, config
    )
    seal = _load_json(seal_path, "formal ULTRA score seal")
    _assert_exact(seal.get("schema_version"), SCORE_SEAL_SCHEMA, "score seal schema")
    _assert_exact(seal.get("protocol"), PROTOCOL, "score seal protocol")
    _assert_exact(seal.get("status"), "all_six_chains_scored_labels_unlocked", "score seal status")
    _assert_exact(seal.get("run_id"), config["run_id"], "score seal run_id")
    _assert_exact(seal.get("manifest_sha256"), sha256_file(manifest_path), "score seal manifest hash")
    _assert_exact(
        seal.get("scoring_started_sha256"),
        scoring_started_sha256,
        "score seal scoring-start hash",
    )
    _assert_exact(seal.get("main_target_labels_accessed_before_seal"), False, "pre-seal labels")
    _assert_exact(
        seal.get("repeatability_contract"), REPEATABILITY_CONTRACT, "sealed repeatability contract"
    )
    _assert_exact(
        seal.get("sentinel_repeat_verified_before_label_unlock"),
        True,
        "sentinel repeat gate",
    )
    _assert_exact(seal.get("component_count"), len(CHAINS), "score seal component count")
    _assert_exact(seal.get("unlock_policy"), SCORE_UNLOCK_POLICY, "score seal unlock policy")
    entries = seal.get("components")
    if (
        not isinstance(entries, list)
        or len(entries) != len(CHAINS)
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        raise ProtocolError("score seal must contain exactly six mapping components")
    require_complete_component_set([entry.get("chain") for entry in entries])
    for entry in entries:
        chain = str(entry["chain"])
        component_path = _score_paths(output_root, chain)["component"]
        _assert_exact(entry.get("path"), _portable(component_path), f"{chain} sealed component path")
        _assert_exact(entry.get("sha256"), sha256_file(component_path), f"{chain} sealed component hash")
        component = verify_component(manifest_path, config_path, chain)
        _assert_exact(entry.get("A_score_sha256"), component["scores"]["A"]["sha256"], f"{chain} A seal")
        _assert_exact(entry.get("B_score_sha256"), component["scores"]["B"]["sha256"], f"{chain} B seal")
        _assert_exact(
            entry.get("A_repeat_score_sha256"),
            None
            if component["repeat_scores"] is None
            else component["repeat_scores"]["A"]["sha256"],
            f"{chain} A repeat seal",
        )
        _assert_exact(
            entry.get("B_repeat_score_sha256"),
            None
            if component["repeat_scores"] is None
            else component["repeat_scores"]["B"]["sha256"],
            f"{chain} B repeat seal",
        )
        _assert_exact(
            entry.get("repeatability"), component["repeatability"], f"{chain} repeatability seal"
        )
        runtime = dict(component["native_backend"])
        runtime.pop("device", None)
        runtime.pop("device_name", None)
        runtime_sha256 = stable_object_sha256(runtime)
        _assert_exact(
            entry.get("native_runtime_sha256"), runtime_sha256, f"{chain} native runtime seal"
        )
        _assert_exact(
            seal.get("native_runtime_sha256"), runtime_sha256, f"{chain} shared native runtime"
        )
    return seal


def _read_target_labels(path: Path, expected_identities):
    """The only target-outcome reader in this controller; call after score seal."""
    import numpy as np
    import pandas as pd

    columns = list(KEYS) + ["y", "size", "lateval"]
    try:
        frame = pd.read_csv(path, usecols=columns, dtype={key: str for key in KEYS})
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read target outcomes from {path}: {exc}") from exc
    if not frame.loc[:, list(KEYS)].equals(expected_identities.reset_index(drop=True)):
        raise ProtocolError(f"target rows do not align with frozen identities: {path}")
    if frame[["y", "size", "lateval"]].isna().any().any():
        raise ProtocolError(f"target y/size/lateval contains nulls: {path}")
    if set(frame["y"].unique()) - {0, 1}:
        raise ProtocolError(f"target outcomes are not complete binary values: {path}")
    for column in ("size", "lateval"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ProtocolError(f"target {column} must be finite and nonnegative: {path}")
        frame[column] = values
    frame["y"] = frame["y"].astype(int)
    return frame


def b1_entry_value_metrics(b_identities, b_labels, b_score) -> dict[str, float]:
    """Fixed-budget B1 observed-value capture under canonical entry ranking."""
    from v2_gpu_rolling import _deterministic_score_order, _entry_arrays

    entry = _entry_arrays(
        b_identities,
        b_labels["y"].to_numpy(dtype=int),
        b_score,
        b_labels["lateval"].to_numpy(dtype=float),
    )
    order = _deterministic_score_order(entry, entry["score"], ("i_iso", "stage"))
    values = entry["lateval"].to_numpy(dtype=float)
    total_value = max(float(values.sum()), 1.0)
    result: dict[str, float] = {
        "entry_observed_value_kusd": float(values.sum()),
    }
    for budget in (25, 50, 100, 250):
        chosen = order[: min(budget, len(order))]
        result[f"entry_value_capture_at_{budget}"] = float(values[chosen].sum() / total_value)
    return result


def derive_task_metrics(a_identities, a_labels, a_score, b_identities, b_labels, b_score):
    """Reuse the canonical rolling task definitions on the fixed ULTRA scores."""
    from v2_gpu_rolling import _ranking_metrics

    metrics = {
        "A": _ranking_metrics("a", a_identities, a_labels, a_score),
        "B1": _ranking_metrics("b1", b_identities, b_labels, b_score),
        "B2": _ranking_metrics("b2", b_identities, b_labels, b_score),
    }
    metrics["B1"].update(b1_entry_value_metrics(b_identities, b_labels, b_score))
    return metrics


def repeat_metric_gate(primary: Mapping[str, object], repeat: Mapping[str, object]) -> dict[str, object]:
    import math

    threshold = float(REPEATABILITY_CONTRACT["metric_absolute_delta_max"])
    rows = {}
    passed = True
    for name, (task, metric) in REPEAT_METRIC_PATHS.items():
        first = float(primary[task][metric])
        second = float(repeat[task][metric])
        if not math.isfinite(first) or not math.isfinite(second):
            raise ProtocolError(f"repeatability metric is non-finite: {name}")
        delta = abs(first - second)
        metric_pass = delta <= threshold
        passed &= metric_pass
        rows[name] = {
            "primary_run1": first,
            "repeat_run2": second,
            "absolute_delta": delta,
            "passed": metric_pass,
        }
    return {
        "metric_absolute_delta_max": threshold,
        "metrics": rows,
        "all_metrics_pass": bool(passed),
        "primary_run_policy": REPEATABILITY_CONTRACT["primary_run"],
    }


def _trained_reference_values() -> dict[str, dict[str, dict[str, float]]]:
    import math

    path = _resolve(REPORTING_CONTRACT["trained_reference_artifact"])
    _assert_exact(
        sha256_file(path),
        REPORTING_CONTRACT["trained_reference_artifact_sha256"],
        "trained-reference summary hash",
    )
    payload = _load_json(path, "trained-reference summary")
    _assert_exact(
        payload.get("schema_version"),
        "upgrade-bench-v2/gpu-main-summary/1",
        "trained-reference schema",
    )
    _assert_exact(payload.get("status"), "complete", "trained-reference status")
    _assert_exact(payload.get("target_fold"), "main", "trained-reference target fold")
    _assert_exact(payload.get("aggregation"), "calendar_mean", "trained-reference aggregation")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise ProtocolError("trained-reference summary must contain exactly 36 records")
    track_map = {"a": "A", "b1": "B1", "b2": "B2"}
    values = {
        family: {task: {} for task in ("A", "B1", "B2")}
        for family in REPORTING_CONTRACT["trained_reference_families"]
    }
    for row in records:
        if not isinstance(row, Mapping):
            raise ProtocolError("trained-reference record is not an object")
        family = str(row.get("family"))
        track = track_map.get(str(row.get("track")))
        chain = str(row.get("chain"))
        if family not in values or track is None or chain not in CHAINS:
            raise ProtocolError("trained-reference record is outside the frozen matrix")
        value = float(row.get("primary_mean"))
        if not math.isfinite(value) or chain in values[family][track]:
            raise ProtocolError("trained-reference value is non-finite or duplicated")
        values[family][track][chain] = value
    for family in values:
        for task in values[family]:
            if list(values[family][task]) != list(CHAINS):
                raise ProtocolError(f"trained-reference matrix is incomplete: {family}/{task}")
    return values


def build_reporting_summary(chain_metrics: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    import math

    if list(chain_metrics) != list(CHAINS):
        raise ProtocolError("reporting requires all six chains in canonical order")
    references = _trained_reference_values()
    headlines = []
    task_means = {}
    value_means = {}
    comparisons = {}
    abstract_rule = {}
    for task in ("A", "B1", "B2"):
        headline_key = REPORTING_CONTRACT["headline_metric_by_task"][task]
        value_key = REPORTING_CONTRACT["value_metric_by_task"][task]
        ultra_by_chain = {}
        value_by_chain = {}
        for chain in CHAINS:
            headline = float(chain_metrics[chain][task][headline_key])
            value = float(chain_metrics[chain][task][value_key])
            if not math.isfinite(headline) or not math.isfinite(value):
                raise ProtocolError(f"ULTRA reporting metric is non-finite: {chain}/{task}")
            ultra_by_chain[chain] = headline
            value_by_chain[chain] = value
            headlines.append(
                {
                    "chain": chain,
                    "task": task,
                    "metric": headline_key,
                    "value": headline,
                }
            )
        task_means[task] = math.fsum(ultra_by_chain.values()) / len(CHAINS)
        value_means[task] = math.fsum(value_by_chain.values()) / len(CHAINS)
        comparisons[task] = {}
        reference_means = {}
        for family in REPORTING_CONTRACT["trained_reference_families"]:
            ref = references[family][task]
            reference_means[family] = math.fsum(ref.values()) / len(CHAINS)
            counts = {"higher": 0, "equal": 0, "lower": 0}
            per_chain = {}
            for chain in CHAINS:
                if ultra_by_chain[chain] > ref[chain]:
                    relation = "higher"
                elif ultra_by_chain[chain] < ref[chain]:
                    relation = "lower"
                else:
                    relation = "equal"
                counts[relation] += 1
                per_chain[chain] = {
                    "ultra": ultra_by_chain[chain],
                    "reference": ref[chain],
                    "relation": relation,
                }
            comparisons[task][family] = {
                "counts": counts,
                "per_chain_unrounded": per_chain,
                "reference_unweighted_six_chain_mean": reference_means[family],
            }
        lower_mean = min(reference_means.values())
        upper_mean = max(reference_means.values())
        if task_means[task] > upper_mean:
            side = "higher"
            same_side_count = sum(
                ultra_by_chain[chain] > max(references[family][task][chain] for family in references)
                for chain in CHAINS
            )
        elif task_means[task] < lower_mean:
            side = "lower"
            same_side_count = sum(
                ultra_by_chain[chain] < min(references[family][task][chain] for family in references)
                for chain in CHAINS
            )
        else:
            side = "inside_reference_mean_interval"
            same_side_count = 0
        abstract_rule[task] = {
            "ultra_mean": task_means[task],
            "reference_mean_interval_closed": [lower_mean, upper_mean],
            "side": side,
            "same_side_of_both_chain_count": int(same_side_count),
            "eligible_for_abstract_mention": (
                side in {"higher", "lower"} and same_side_count >= 5
            ),
        }
    if len(headlines) != REPORTING_CONTRACT["report_all_chain_task_headlines"]:
        raise ProtocolError("partial-success reporting is forbidden")
    return {
        "model_label": REPORTING_CONTRACT["model_label"],
        "all_18_chain_task_headlines": headlines,
        "unweighted_six_chain_headline_means": task_means,
        "unweighted_six_chain_value_means": value_means,
        "trained_reference_comparisons": comparisons,
        "abstract_mention_rule": abstract_rule,
        "abstract_should_mention_ultra": any(
            row["eligible_for_abstract_mention"] for row in abstract_rule.values()
        ),
        "forbidden_claims": REPORTING_CONTRACT["forbidden_claims"],
    }


def _evaluate(args) -> int:
    # CRITICAL GATE: opaque candidate bytes were precommitted earlier, but no
    # target-column parse or target metric computation occurs before the global
    # six-chain score seal verifies.
    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path)
    manifest_path = _manifest_path(config)
    output_root = _resolve(str(config["output_root"]))
    seal_path = output_root / "SCORES_COMPLETE.json"
    seal = verify_score_seal(seal_path, manifest_path, config_path)
    manifest = verify_freeze_manifest(manifest_path, config_path)
    seal_sha256 = sha256_file(seal_path)
    _scoring_started, scoring_started_sha256 = verify_scoring_started(
        output_root / "SCORING_STARTED.json", manifest_path, config
    )
    _assert_exact(
        seal.get("scoring_started_sha256"),
        scoring_started_sha256,
        "pre-evaluation scoring-start hash",
    )
    _claim_marker(
        output_root / "LABEL_EVALUATION_STARTED.json",
        {
            "schema_version": EVALUATION_START_SCHEMA,
            "protocol": PROTOCOL,
            "run_id": config["run_id"],
            "score_seal_sha256": seal_sha256,
            "scoring_started_sha256": scoring_started_sha256,
            "started_at_utc": _utc_now(),
            "ordering_attestation": "all six A/B score components completed before first target-label read",
        },
        (
            "schema_version",
            "protocol",
            "run_id",
            "score_seal_sha256",
            "scoring_started_sha256",
        ),
    )
    final_path = output_root / "evaluation.json"
    metric_paths = {chain: output_root / "metrics" / f"metrics_{chain}.json" for chain in CHAINS}
    if final_path.exists() or any(path.exists() for path in metric_paths.values()):
        raise ProtocolError("formal ULTRA target evaluation is immutable and already exists")

    # Load all identities, scores, and labels only after the global gate above.
    loaded = {}
    first_label_read_at = None
    for chain in CHAINS:
        identities = {}
        labels = {}
        scores = {}
        repeat_scores = None
        paths = _score_paths(output_root, chain)
        for source in SOURCE_SPECS:
            candidate = _candidate_path(config, chain, source)
            cohort = manifest["chains"][chain]["cohorts"][source]
            _assert_exact(
                stable_file_sha256(candidate),
                cohort["full_file_sha256"],
                f"{chain}/{source} evaluation candidate precommit",
            )
            identities[source], _metadata = read_candidate_identities(candidate, source)
            scores[source] = _read_score_file(paths[source], identities[source])
            if first_label_read_at is None:
                first_label_read_at = _utc_now()
            labels[source] = _read_target_labels(candidate, identities[source])
        if chain == REPEATABILITY_CONTRACT["sentinel_chain"]:
            repeat_scores = {
                source: _read_score_file(paths[f"{source}_repeat"], identities[source])
                for source in SOURCE_SPECS
            }
        loaded[chain] = (identities, labels, scores, repeat_scores)

    metric_refs = []
    chain_metrics = {}
    for chain in CHAINS:
        identities, labels, scores, repeat_scores = loaded[chain]
        metrics = derive_task_metrics(
            identities["A"],
            labels["A"],
            scores["A"]["ultra_score"].to_numpy(dtype=float),
            identities["B"],
            labels["B"],
            scores["B"]["ultra_score"].to_numpy(dtype=float),
        )
        chain_metrics[chain] = metrics
        repeated_metrics = None
        metric_repeatability = None
        if repeat_scores is not None:
            repeated_metrics = derive_task_metrics(
                identities["A"],
                labels["A"],
                repeat_scores["A"]["ultra_score"].to_numpy(dtype=float),
                identities["B"],
                labels["B"],
                repeat_scores["B"]["ultra_score"].to_numpy(dtype=float),
            )
            metric_repeatability = repeat_metric_gate(metrics, repeated_metrics)
            if not metric_repeatability["all_metrics_pass"]:
                raise ProtocolError(
                    "sheep repeat headline/value metrics exceed the frozen 1e-10 gate"
                )
        payload = {
            "schema_version": METRIC_SCHEMA,
            "protocol": PROTOCOL,
            "status": "complete",
            "run_id": config["run_id"],
            "chain": chain,
            "created_at_utc": _utc_now(),
            "model": "ULTRA ultra_4g fixed zero-shot",
            "checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256,
            "score_seal_sha256": seal_sha256,
            "scoring_started_sha256": scoring_started_sha256,
            "task_contract": TASK_CONTRACT,
            "target_sources": {
                source: {
                    "path": _portable(_candidate_path(config, chain, source)),
                    "sha256": stable_file_sha256(_candidate_path(config, chain, source)),
                    "precommitted_sha256": manifest["chains"][chain]["cohorts"][source][
                        "full_file_sha256"
                    ],
                    "bytes": _candidate_path(config, chain, source).stat().st_size,
                    "precommitted_bytes": manifest["chains"][chain]["cohorts"][source][
                        "full_file_bytes"
                    ],
                    "rows": int(len(labels[source])),
                    "positive_lanes": int(labels[source]["y"].sum()),
                    "score_sha256": seal["components"][CHAINS.index(chain)][f"{source}_score_sha256"],
                    "repeat_score_sha256": seal["components"][CHAINS.index(chain)][
                        f"{source}_repeat_score_sha256"
                    ],
                }
                for source in SOURCE_SPECS
            },
            "metrics": metrics,
            "repeat_metrics": repeated_metrics,
            "repeatability_metric_gate": metric_repeatability,
        }
        _write_json_exclusive(metric_paths[chain], payload)
        metric_refs.append(
            {
                "chain": chain,
                "path": _portable(metric_paths[chain]),
                "sha256": sha256_file(metric_paths[chain]),
            }
        )
    reporting_summary = build_reporting_summary(chain_metrics)
    evaluation = {
        "schema_version": EVALUATION_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "run_id": config["run_id"],
        "created_at_utc": _utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "score_seal_sha256": seal_sha256,
        "scoring_started_sha256": scoring_started_sha256,
        "checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256,
        "provenance": PROVENANCE,
        "training_disclosure": TRAINING_DISCLOSURE,
        "overlap_policy": OVERLAP_POLICY,
        "model_policy": MODEL_POLICY,
        "task_contract": TASK_CONTRACT,
        "reporting_contract": REPORTING_CONTRACT,
        "reporting_summary": reporting_summary,
        "ordering": {
            "score_component_count_before_label_read": len(seal["components"]),
            "all_six_score_components_verified_before_label_read": True,
            "first_target_label_read_at_utc": first_label_read_at,
        },
        "metric_artifacts": metric_refs,
    }
    _write_json_exclusive(final_path, evaluation)
    print(f"formal six-chain ULTRA evaluation complete: {final_path}")
    return 0


def _freeze(args) -> int:
    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path)
    output_root = _resolve(str(config["output_root"]))
    for later in (
        output_root / "SCORING_STARTED.json",
        output_root / "SCORES_COMPLETE.json",
        output_root / "LABEL_EVALUATION_STARTED.json",
        output_root / "evaluation.json",
    ):
        if later.exists():
            raise ProtocolError(f"cannot freeze after a later phase has started: {later}")
    manifest_path = _manifest_path(config)
    payload = build_freeze_payload(config_path)
    _write_json_exclusive(manifest_path, payload)
    verify_freeze_manifest(manifest_path, config_path)
    print(f"formal ULTRA protocol frozen before target scoring: {manifest_path}")
    return 0


def _assert_nested_close(observed: object, expected: object, role: str) -> None:
    import math

    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ProtocolError(f"{role} object keys differ")
        for key in expected:
            _assert_nested_close(observed[key], expected[key], f"{role}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ProtocolError(f"{role} list shape differs")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _assert_nested_close(left, right, f"{role}[{index}]")
        return
    if isinstance(expected, float):
        try:
            value = float(observed)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{role} is not numeric") from exc
        if math.isnan(expected) and math.isnan(value):
            return
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ProtocolError(f"{role} numeric mismatch: {value!r} != {expected!r}")
        return
    _assert_exact(observed, expected, role)


def verify_evaluation_artifacts(
    evaluation_path: Path, manifest_path: Path, seal_path: Path, config_path: Path
) -> dict[str, Any]:
    config = load_and_validate_config(config_path)
    manifest = verify_freeze_manifest(manifest_path, config_path)
    seal = verify_score_seal(seal_path, manifest_path, config_path)
    seal_sha256 = sha256_file(seal_path)
    output_root = _resolve(str(config["output_root"]))
    _scoring_started, scoring_started_sha256 = verify_scoring_started(
        output_root / "SCORING_STARTED.json", manifest_path, config
    )
    _assert_exact(
        seal.get("scoring_started_sha256"),
        scoring_started_sha256,
        "verified seal scoring-start hash",
    )
    evaluation = _load_json(evaluation_path, "formal ULTRA evaluation")
    _assert_exact(evaluation.get("schema_version"), EVALUATION_SCHEMA, "evaluation schema")
    _assert_exact(evaluation.get("protocol"), PROTOCOL, "evaluation protocol")
    _assert_exact(evaluation.get("status"), "complete", "evaluation status")
    _assert_exact(evaluation.get("run_id"), config["run_id"], "evaluation run_id")
    _assert_exact(evaluation.get("manifest_sha256"), sha256_file(manifest_path), "evaluation manifest")
    _assert_exact(evaluation.get("score_seal_sha256"), seal_sha256, "evaluation score seal")
    _assert_exact(
        evaluation.get("scoring_started_sha256"),
        scoring_started_sha256,
        "evaluation scoring-start hash",
    )
    _assert_exact(evaluation.get("checkpoint_sha256"), CANONICAL_CHECKPOINT_SHA256, "evaluation checkpoint")
    _assert_exact(evaluation.get("provenance"), PROVENANCE, "evaluation provenance")
    _assert_exact(
        evaluation.get("training_disclosure"), TRAINING_DISCLOSURE, "evaluation training disclosure"
    )
    _assert_exact(evaluation.get("overlap_policy"), OVERLAP_POLICY, "evaluation overlap policy")
    _assert_exact(evaluation.get("model_policy"), MODEL_POLICY, "evaluation model policy")
    _assert_exact(evaluation.get("task_contract"), TASK_CONTRACT, "evaluation task contract")
    _assert_exact(evaluation.get("reporting_contract"), REPORTING_CONTRACT, "evaluation reporting contract")
    refs = evaluation.get("metric_artifacts")
    if (
        not isinstance(refs, list)
        or len(refs) != len(CHAINS)
        or any(not isinstance(ref, Mapping) for ref in refs)
    ):
        raise ProtocolError("evaluation metric_artifacts must be exactly six objects")
    require_complete_component_set([ref.get("chain") for ref in refs])
    marker = _load_json(
        output_root / "LABEL_EVALUATION_STARTED.json", "formal label-evaluation marker"
    )
    _assert_exact(marker.get("schema_version"), EVALUATION_START_SCHEMA, "evaluation marker schema")
    _assert_exact(marker.get("protocol"), PROTOCOL, "evaluation marker protocol")
    _assert_exact(marker.get("run_id"), config["run_id"], "evaluation marker run_id")
    _assert_exact(marker.get("score_seal_sha256"), seal_sha256, "evaluation marker seal")
    _assert_exact(
        marker.get("scoring_started_sha256"),
        scoring_started_sha256,
        "evaluation marker scoring-start hash",
    )
    _assert_exact(
        marker.get("ordering_attestation"),
        "all six A/B score components completed before first target-label read",
        "evaluation marker ordering",
    )
    marker_started = marker.get("started_at_utc")
    if not isinstance(marker_started, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)", marker_started
    ):
        raise ProtocolError("evaluation marker timestamp is invalid")
    ordering = evaluation.get("ordering")
    if not isinstance(ordering, Mapping):
        raise ProtocolError("evaluation lacks ordering receipt")
    _assert_exact(
        ordering.get("score_component_count_before_label_read"), len(CHAINS), "ordering component count"
    )
    _assert_exact(
        ordering.get("all_six_score_components_verified_before_label_read"),
        True,
        "ordering score gate",
    )
    first_read = ordering.get("first_target_label_read_at_utc")
    if not isinstance(first_read, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)", first_read
    ):
        raise ProtocolError("evaluation first-label timestamp is invalid")
    if marker_started > first_read:
        raise ProtocolError("evaluation marker timestamp follows the first target-label read")
    chain_metrics = {}
    for chain, ref in zip(CHAINS, refs):
        metric_path = output_root / "metrics" / f"metrics_{chain}.json"
        _assert_exact(ref.get("chain"), chain, f"{chain} metric ref chain")
        _assert_exact(ref.get("path"), _portable(metric_path), f"{chain} canonical metric path")
        _assert_exact(ref.get("sha256"), sha256_file(metric_path), f"{chain} metric artifact hash")
        artifact = _load_json(metric_path, f"{chain} metrics")
        _assert_exact(artifact.get("schema_version"), METRIC_SCHEMA, f"{chain} metric schema")
        _assert_exact(artifact.get("protocol"), PROTOCOL, f"{chain} metric protocol")
        _assert_exact(artifact.get("status"), "complete", f"{chain} metric status")
        _assert_exact(artifact.get("run_id"), config["run_id"], f"{chain} metric run_id")
        _assert_exact(artifact.get("chain"), chain, f"{chain} metric identity")
        _assert_exact(artifact.get("score_seal_sha256"), seal_sha256, f"{chain} metric seal")
        _assert_exact(
            artifact.get("scoring_started_sha256"),
            scoring_started_sha256,
            f"{chain} metric scoring-start hash",
        )
        _assert_exact(artifact.get("checkpoint_sha256"), CANONICAL_CHECKPOINT_SHA256, f"{chain} metric checkpoint")
        _assert_exact(artifact.get("task_contract"), TASK_CONTRACT, f"{chain} metric task contract")
        identities, labels, scores, repeats = {}, {}, {}, None
        paths = _score_paths(output_root, chain)
        target_sources = artifact.get("target_sources")
        if not isinstance(target_sources, Mapping) or set(target_sources) != set(SOURCE_SPECS):
            raise ProtocolError(f"{chain} metric target sources are not exact A/B")
        for source in SOURCE_SPECS:
            candidate = _candidate_path(config, chain, source)
            frozen = manifest["chains"][chain]["cohorts"][source]
            _assert_exact(stable_file_sha256(candidate), frozen["full_file_sha256"], f"{chain}/{source} verify target")
            identities[source], _metadata = read_candidate_identities(candidate, source)
            labels[source] = _read_target_labels(candidate, identities[source])
            scores[source] = _read_score_file(paths[source], identities[source])
            source_ref = target_sources[source]
            expected_source_ref = {
                "path": _portable(candidate),
                "sha256": frozen["full_file_sha256"],
                "precommitted_sha256": frozen["full_file_sha256"],
                "bytes": frozen["full_file_bytes"],
                "precommitted_bytes": frozen["full_file_bytes"],
                "rows": int(len(labels[source])),
                "positive_lanes": int(labels[source]["y"].sum()),
                "score_sha256": seal["components"][CHAINS.index(chain)][f"{source}_score_sha256"],
                "repeat_score_sha256": seal["components"][CHAINS.index(chain)][f"{source}_repeat_score_sha256"],
            }
            _assert_exact(source_ref, expected_source_ref, f"{chain}/{source} target source receipt")
        recomputed = derive_task_metrics(
            identities["A"],
            labels["A"],
            scores["A"]["ultra_score"].to_numpy(dtype=float),
            identities["B"],
            labels["B"],
            scores["B"]["ultra_score"].to_numpy(dtype=float),
        )
        _assert_nested_close(artifact.get("metrics"), recomputed, f"{chain} recomputed metrics")
        chain_metrics[chain] = recomputed
        if chain == REPEATABILITY_CONTRACT["sentinel_chain"]:
            repeats = {
                source: _read_score_file(paths[f"{source}_repeat"], identities[source])
                for source in SOURCE_SPECS
            }
            repeat_metrics = derive_task_metrics(
                identities["A"],
                labels["A"],
                repeats["A"]["ultra_score"].to_numpy(dtype=float),
                identities["B"],
                labels["B"],
                repeats["B"]["ultra_score"].to_numpy(dtype=float),
            )
            gate = repeat_metric_gate(recomputed, repeat_metrics)
            _assert_exact(gate["all_metrics_pass"], True, "recomputed sheep repeat metric gate")
            _assert_nested_close(artifact.get("repeat_metrics"), repeat_metrics, "sheep repeat metrics")
            _assert_nested_close(
                artifact.get("repeatability_metric_gate"), gate, "sheep repeat metric gate"
            )
        else:
            _assert_exact(artifact.get("repeat_metrics"), None, f"{chain} repeat metrics")
            _assert_exact(
                artifact.get("repeatability_metric_gate"), None, f"{chain} repeat metric gate"
            )
    reporting = build_reporting_summary(chain_metrics)
    _assert_nested_close(
        evaluation.get("reporting_summary"), reporting, "recomputed reporting summary"
    )
    return evaluation


def _verify(args) -> int:
    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path)
    manifest_path = _manifest_path(config)
    verify_freeze_manifest(manifest_path, config_path)
    if args.level in {"scores", "evaluation"}:
        output_root = _resolve(str(config["output_root"]))
        verify_score_seal(output_root / "SCORES_COMPLETE.json", manifest_path, config_path)
    if args.level == "evaluation":
        output_root = _resolve(str(config["output_root"]))
        verify_evaluation_artifacts(
            output_root / "evaluation.json",
            manifest_path,
            output_root / "SCORES_COMPLETE.json",
            config_path,
        )
    print(f"formal ULTRA verification PASS: level={args.level}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="freeze six-chain label-blind inputs and fixed checkpoint")
    freeze.set_defaults(func=_freeze)
    score = sub.add_parser("score-chain", help="emit complete A/B lane scores for one chain")
    score.add_argument("--chain", choices=CHAINS, required=True)
    score.add_argument("--device", default="cuda:0")
    score.set_defaults(func=_score_chain)
    seal = sub.add_parser("seal-scores", help="verify all six components before labels unlock")
    seal.set_defaults(func=_seal_scores)
    evaluate = sub.add_parser("evaluate", help="evaluate A/B1/B2 after the score seal")
    evaluate.set_defaults(func=_evaluate)
    verify = sub.add_parser("verify", help="verify frozen, score-sealed, or evaluated artifacts")
    verify.add_argument("--level", choices=("freeze", "scores", "evaluation"), default="evaluation")
    verify.set_defaults(func=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
