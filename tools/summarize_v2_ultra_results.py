"""Promote the formal ULTRA zero-shot run to sanitized public JSON and CSV.

The default command validates the complete extracted formal result, including
its hash chain and the exact sheep repeat, before publishing an allowlisted
summary.  Raw score rows, machine/user locators, and target-table hashes are
never copied into the public artifacts.

``--verify-output`` is deliberately independent of the extracted run.  It
recomputes every public aggregate and trained-reference comparison from the
public metric records, then checks canonical JSON and derived CSV bytes using
only the public outputs, the frozen config, the trained reference summary, and
the current promotion/formal-controller tool hashes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import build_gpu_step3_postfreeze_attestation as postfreeze


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PRIVATE_ROOT = (
    ROOT / "tmp" / "ultra_formal_r4_extracted" / "results_v2" / "ultra_formal"
)
CANONICAL_CONFIG = ROOT / "configs" / "v2_ultra_formal.json"
CANONICAL_FORMAL_CONTROLLER = ROOT / "tools" / "v2_ultra_formal.py"
CANONICAL_TRAINED_SUMMARY = ROOT / "results_v2" / "metrics" / "v2_gpu_rolling_summary.json"
DEFAULT_JSON_OUT = ROOT / "results_v2" / "metrics" / "v2_ultra_zero_shot_summary.json"
DEFAULT_CSV_OUT = ROOT / "results_v2" / "metrics" / "v2_ultra_zero_shot_summary.csv"

PUBLIC_SUMMARY_SCHEMA = "upgrade-bench-v2/ultra-zero-shot-public-summary/1"
PROTOCOL = "upgrade-bench-v2/ultra-4g-zero-shot/2"
RUN_ID = "ultra-4g-zero-shot-fixed-20260717-r4"
STATUS = "complete_verified_sanitized"
SCORING_START_SCHEMA = "upgrade-bench-v2/ultra-formal-score-start/2"
SCORE_SEAL_SCHEMA = "upgrade-bench-v2/ultra-formal-score-seal/3"
EVALUATION_START_SCHEMA = "upgrade-bench-v2/ultra-formal-evaluation-start/3"
METRIC_SCHEMA = "upgrade-bench-v2/ultra-formal-chain-metrics/3"
EVALUATION_SCHEMA = "upgrade-bench-v2/ultra-formal-evaluation/3"
FROZEN_TRAINED_RUNNER_SHA256 = (
    "c821c4027b199c2a115ba6abe9dfd2361bdd70a61cf812e75d014c7e786b6645"
)
CURRENT_TRAINED_RUNNER_SHA256 = (
    "8508a05935ea1253275f3226e30a30af9fc613b4b1c4ffcd2c6f8bcd0d4a050d"
)
FROZEN_POSTFREEZE_ATTESTATION_SHA256 = (
    "962c61173513b0ec29d868725ab2e24a93149819d03c1b0263cadcfbc8cc341b"
)
FROZEN_POSTFREEZE_ALLOWED_CHANGED_FILE_COUNT = 5
SCORING_START_POLICY = (
    "freeze is immutable; target labels remain locked until the six-chain score seal"
)

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
TASKS = ("A", "B1", "B2")
FAMILIES = ("kge", "nbfnet")
HEADLINE_METRICS = {
    "A": "lane_average_precision",
    "B1": "entry_average_precision",
    "B2": "conditional_recall_at_3",
}
VALUE_METRICS = {
    "A": "value_capture_at_500",
    "B1": "entry_value_capture_at_50",
    "B2": "conditional_value_capture_at_3",
}
REPEAT_METRICS = (
    "A.lane_average_precision",
    "A.value_capture_at_500",
    "B1.entry_average_precision",
    "B1.entry_value_capture_at_50",
    "B2.conditional_recall_at_3",
    "B2.conditional_value_capture_at_3",
)

MODEL = {
    "label": "external pretrained zero-shot",
    "checkpoint_name": "ultra_4g",
    "checkpoint_count": 1,
    "checkpoint_fixed_before_target_scoring": True,
    "checkpoint_training_seed_disclosed": False,
    "training_performed": False,
    "fine_tuning_performed": False,
    "result_selection_performed": False,
}
DESIGN = {
    "chains": list(CHAINS),
    "tasks": list(TASKS),
    "checkpoint_count": 1,
    "expected_chain_task_records": 18,
    "target_early_graph_used": True,
    "target_labels_used_for_training_or_selection": False,
    "shared_B_scores_for_B1_and_B2": True,
    "aggregate": "unweighted mean of six unrounded chain values for each task",
}
CLAIM_BOUNDARIES = {
    "result_type": "descriptive external-pretrained zero-shot reference",
    "valid_uses": [
        "six-chain task-specific benchmark comparison",
        "reproducibility check of the fixed public checkpoint reference",
    ],
    "not_valid_for": [
        "fair-compute comparison",
        "champion claim",
        "statistical significance claim",
        "causal industrial-upgrading interpretation",
        "population inference beyond the six fixed chains",
    ],
    "trained_reference_comparison_is_descriptive": True,
    "causal_interpretation": False,
    "checkpoint_or_result_selected_on_target_outcomes": False,
    "disclosed_pretraining_list_contains_benchmark_sources": False,
    "pretraining_graphs": ["FB15k237", "WN18RR", "CoDExMedium", "NELL995"],
    "checkpoint_training_steps": 400000,
    "inference_seed": 1024,
    "inference_seed_is_not_checkpoint_training_seed_evidence": True,
}

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)\Z")
PRIVATE_TEXT_PATTERNS = (
    ("Unix user-home locator", re.compile(r"/(?:home|users)/[^/\s]+/", re.IGNORECASE)),
    ("Windows user-home locator", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)),
    ("IHPC host alias", re.compile(r"\bmars\d+\b", re.IGNORECASE)),
    ("IHPC account token", re.compile(r"\bsli\d+\b", re.IGNORECASE)),
)
FORBIDDEN_PUBLIC_KEYS = {
    "host",
    "hostname",
    "user",
    "username",
    "pid",
    "worker_id",
    "resolved_path",
    "score_vector_sha256",
    "score_artifact_sha256",
    "score_file_sha256",
    "raw_score",
}


class ResultValidationError(ValueError):
    """The formal result or public promotion failed a fail-closed check."""


def _fail(role: str, message: str) -> None:
    raise ResultValidationError(f"{role}: {message}")


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _gpu_summary_json_bytes(value: Any) -> bytes:
    """Render the canonical trained-reference summary serialization."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail("source artifact", f"cannot hash {Path(path).name}: {exc}")
    return digest.hexdigest()


def _strict_json_payload(raw: bytes, role: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(role, f"cannot read strict JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(role, "top-level value must be an object")
    return payload


def _strict_json_file(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    if path.is_symlink():
        _fail(role, "symbolic-link inputs are forbidden")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(role, f"cannot read {path.name}: {exc}")
    return _strict_json_payload(raw, role), raw


def _formal_json(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = _strict_json_file(path, role)
    if raw != _canonical_json_bytes(payload):
        _fail(role, "formal bytes are not canonical compact JSON")
    return payload, raw


def _hash(value: Any, role: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        _fail(role, "expected a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(role, "expected an object")
    if set(value) != expected:
        _fail(
            role,
            f"fields are not exact (observed={sorted(map(str, value))}, expected={sorted(expected)})",
        )
    return value


def _validate_scoring_started(
    private_root: Path, manifest_sha256: str
) -> tuple[dict[str, Any], str]:
    marker, raw = _formal_json(
        Path(private_root) / "SCORING_STARTED.json", "scoring-start marker"
    )
    fields = {
        "schema_version",
        "protocol",
        "run_id",
        "manifest_sha256",
        "started_at_utc",
        "policy",
    }
    marker = dict(_exact_keys(marker, fields, "scoring-start marker"))
    required = {
        "schema_version": SCORING_START_SCHEMA,
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "policy": SCORING_START_POLICY,
    }
    for field, expected in required.items():
        if marker.get(field) != expected:
            _fail("scoring-start marker", f"invalid {field}")
    if (
        not isinstance(marker.get("started_at_utc"), str)
        or ISO_UTC.fullmatch(marker["started_at_utc"]) is None
    ):
        _fail("scoring-start marker", "timestamp is invalid")
    return marker, _sha256_bytes(raw)


def _finite(value: Any, role: str, *, lower: float = 0.0, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(role, "expected a finite number")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        _fail(role, f"expected a finite number in [{lower}, {upper}]")
    return result


def _integer(value: Any, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(role, f"expected an integer >= {minimum}")
    return value


def _close(observed: Any, expected: float, role: str) -> float:
    value = _finite(observed, role, lower=-1.0, upper=1.0)
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-15):
        _fail(role, f"{value!r} != mechanically recomputed {expected!r}")
    return value


def _nested_close(observed: Any, expected: Any, role: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            _fail(role, "object fields differ")
        for key in expected:
            _nested_close(observed[key], expected[key], f"{role}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            _fail(role, "list shape differs")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _nested_close(left, right, f"{role}[{index}]")
        return
    if isinstance(expected, float):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            _fail(role, "expected numeric value")
        value = float(observed)
        if not math.isfinite(value) or not math.isclose(
            value, expected, rel_tol=0.0, abs_tol=1e-15
        ):
            _fail(role, f"numeric mismatch: {value!r} != {expected!r}")
        return
    if observed != expected:
        _fail(role, f"{observed!r} != {expected!r}")


def _mean(values: Sequence[float]) -> float:
    if len(values) != len(CHAINS):
        _fail("six-chain mean", "requires exactly six values")
    return math.fsum(values) / len(values)


def _privacy_audit(value: Any, role: str = "public summary") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PUBLIC_KEYS or "path" in lowered or "candidate" in lowered:
                _fail(role, f"non-public field leaked: {key}")
            _privacy_audit(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _privacy_audit(child, f"{role}[{index}]")
    elif isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _fail(role, "absolute locator leaked")
        for label, pattern in PRIVATE_TEXT_PATTERNS:
            if pattern.search(value):
                _fail(role, f"operational locator leaked: {label}")


def _verify_current_trained_reference_bridge(
    trained: Mapping[str, Any],
    trained_raw: bytes,
    observed_sha256: str,
    frozen_sha256: str,
    *,
    require_full_inventory: bool = True,
) -> None:
    """Prove that the current GPU summary is the frozen reporting input.

    The formal ULTRA config froze the trained-reference summary before the GPU
    runner received prospective fail-closed CLI/config gates.  Regenerating the
    GPU summary after that hardening changed exactly three provenance leaves.
    This bridge accepts the regenerated summary only after the current
    post-freeze attestation validates it, then reverse-projects those three
    leaves and requires the exact frozen artifact hash.  Metrics, selections,
    model values, and every other summary field therefore remain hash-bound.
    """

    if observed_sha256 == frozen_sha256:
        _fail("trained reference bridge", "bridge is unnecessary for frozen bytes")
    if _sha256_bytes(trained_raw) != observed_sha256:
        _fail("trained reference bridge", "observed hash is not bound to supplied bytes")
    try:
        canonical_current = _gpu_summary_json_bytes(trained)
    except (TypeError, ValueError) as exc:
        _fail("trained reference bridge", f"cannot render canonical current summary: {exc}")
    if trained_raw != canonical_current:
        _fail("trained reference bridge", "current GPU summary bytes are not canonical")

    recomputation = trained.get("metric_recomputation")
    if not isinstance(recomputation, Mapping):
        _fail("trained reference bridge", "metric_recomputation is missing")
    if recomputation.get("source_sha256") != CURRENT_TRAINED_RUNNER_SHA256:
        _fail("trained reference bridge", "current runner hash is not the allowlisted value")

    try:
        postfreeze.verify_summary_binding(
            trained,
            artifact_path=ROOT / postfreeze.ARTIFACT_ROLE,
            root=ROOT,
            require_full_inventory=require_full_inventory,
        )
    except postfreeze.AttestationError as exc:
        _fail("trained reference bridge", f"current post-freeze attestation failed: {exc}")

    projected = copy.deepcopy(dict(trained))
    projected_recomputation = projected.get("metric_recomputation")
    projected_binding = projected.get("post_freeze_semantic_attestation")
    if not isinstance(projected_recomputation, dict) or not isinstance(projected_binding, dict):
        _fail("trained reference bridge", "reverse-projection fields are missing")
    projected_recomputation["source_sha256"] = FROZEN_TRAINED_RUNNER_SHA256
    projected_binding["sha256"] = FROZEN_POSTFREEZE_ATTESTATION_SHA256
    projected_binding["allowed_changed_file_count"] = (
        FROZEN_POSTFREEZE_ALLOWED_CHANGED_FILE_COUNT
    )
    try:
        projected_sha256 = _sha256_bytes(_gpu_summary_json_bytes(projected))
    except (TypeError, ValueError) as exc:
        _fail("trained reference bridge", f"cannot render reverse projection: {exc}")
    if projected_sha256 != frozen_sha256:
        _fail(
            "trained reference bridge",
            "reverse projection does not reconstruct the frozen trained-reference bytes",
        )


def _load_config_and_references() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    config, config_raw = _strict_json_file(CANONICAL_CONFIG, "formal config")
    expected = {
        "schema_version": "upgrade-bench-v2/ultra-formal-config/2",
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "chains": list(CHAINS),
        "fold": "main",
        "aggregation": "calendar_mean",
    }
    for field, wanted in expected.items():
        if config.get(field) != wanted:
            _fail("formal config", f"{field} is not the fixed r4 value")
    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("name") != "ultra_4g":
        _fail("formal config", "fixed ultra_4g checkpoint is absent")
    _hash(checkpoint.get("sha256"), "formal config checkpoint")
    if config.get("model_policy", {}).get("checkpoint_count") != 1:
        _fail("formal config", "checkpoint_count is not one")
    if config.get("model_policy", {}).get("historical_or_target_label_selection") is not False:
        _fail("formal config", "target-label selection is not disabled")
    reporting = config.get("reporting_contract")
    if not isinstance(reporting, Mapping):
        _fail("formal config", "reporting contract is missing")
    if reporting.get("headline_metric_by_task") != HEADLINE_METRICS:
        _fail("formal config", "headline metric map changed")
    if reporting.get("value_metric_by_task") != VALUE_METRICS:
        _fail("formal config", "value metric map changed")
    trained_hash = _hash(
        reporting.get("trained_reference_artifact_sha256"), "formal config trained summary"
    )
    trained, trained_raw = _strict_json_file(
        CANONICAL_TRAINED_SUMMARY, "trained reference summary"
    )
    observed_trained_hash = _sha256_bytes(trained_raw)
    if observed_trained_hash != trained_hash:
        _verify_current_trained_reference_bridge(
            trained,
            trained_raw,
            observed_trained_hash,
            trained_hash,
        )
    # Public ULTRA provenance continues to identify the artifact frozen into
    # the formal reporting contract.  The current bytes are an equivalent
    # recomputation source admitted only by the proof above; they must not be
    # mislabeled as the artifact the immutable ULTRA run consumed.
    return config, _sha256_bytes(config_raw), trained, trained_hash


def _trained_reference_values(trained: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    records = trained.get("records")
    if not isinstance(records, list):
        _fail("trained reference summary", "records are missing")
    track_map = {"a": "A", "b1": "B1", "b2": "B2"}
    values = {family: {task: {} for task in TASKS} for family in FAMILIES}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            _fail(f"trained reference record {index}", "expected an object")
        family = row.get("family")
        task = track_map.get(str(row.get("track")))
        chain = row.get("chain")
        if family not in FAMILIES or task not in TASKS or chain not in CHAINS:
            continue
        if row.get("primary_metric") != HEADLINE_METRICS[task]:
            _fail(f"trained reference {family}/{task}/{chain}", "primary metric changed")
        if chain in values[family][task]:
            _fail(f"trained reference {family}/{task}/{chain}", "duplicate value")
        values[family][task][chain] = _finite(
            row.get("primary_mean"), f"trained reference {family}/{task}/{chain}"
        )
    for family in FAMILIES:
        for task in TASKS:
            if list(values[family][task]) != list(CHAINS):
                _fail(f"trained reference {family}/{task}", "six-chain matrix is incomplete")
    return values


def _validate_metric_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 18:
        _fail("metric_records", "expected exactly 18 chain-task rows")
    fields = {
        "chain",
        "task",
        "headline_metric",
        "headline_value",
        "value_metric",
        "value_value",
    }
    cleaned = []
    index = 0
    for task in TASKS:
        for chain in CHAINS:
            role = f"metric_records[{index}]"
            row = _exact_keys(value[index], fields, role)
            if row.get("chain") != chain or row.get("task") != task:
                _fail(role, "chain/task identity or canonical order changed")
            if row.get("headline_metric") != HEADLINE_METRICS[task]:
                _fail(role, "headline metric changed")
            if row.get("value_metric") != VALUE_METRICS[task]:
                _fail(role, "value metric changed")
            cleaned.append(
                {
                    "chain": chain,
                    "task": task,
                    "headline_metric": HEADLINE_METRICS[task],
                    "headline_value": _finite(row.get("headline_value"), f"{role}.headline"),
                    "value_metric": VALUE_METRICS[task],
                    "value_value": _finite(row.get("value_value"), f"{role}.value"),
                }
            )
            index += 1
    return cleaned


def _derive_reporting(
    records: Sequence[Mapping[str, Any]], references: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    indexed = {(row["task"], row["chain"]): row for row in records}
    task_summaries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    abstract_tasks: dict[str, Any] = {}
    formal_comparisons: dict[str, Any] = {}
    for task in TASKS:
        headline = [float(indexed[(task, chain)]["headline_value"]) for chain in CHAINS]
        values = [float(indexed[(task, chain)]["value_value"]) for chain in CHAINS]
        ultra_mean = _mean(headline)
        task_summaries[task] = {
            "headline_metric": HEADLINE_METRICS[task],
            "value_metric": VALUE_METRICS[task],
            "unweighted_six_chain_headline_mean": ultra_mean,
            "unweighted_six_chain_value_mean": _mean(values),
        }
        comparisons[task] = {}
        formal_comparisons[task] = {}
        reference_means: dict[str, float] = {}
        for family in FAMILIES:
            ref = references[family][task]
            reference_mean = _mean([float(ref[chain]) for chain in CHAINS])
            reference_means[family] = reference_mean
            counts = {"higher": 0, "equal": 0, "lower": 0}
            per_chain = {}
            for chain in CHAINS:
                ultra = float(indexed[(task, chain)]["headline_value"])
                reference = float(ref[chain])
                relation = "higher" if ultra > reference else "lower" if ultra < reference else "equal"
                counts[relation] += 1
                per_chain[chain] = {
                    "ultra": ultra,
                    "reference": reference,
                    "relation": relation,
                }
            comparisons[task][family] = {
                "counts": counts,
                "reference_unweighted_six_chain_mean": reference_mean,
            }
            formal_comparisons[task][family] = {
                "counts": counts,
                "per_chain_unrounded": per_chain,
                "reference_unweighted_six_chain_mean": reference_mean,
            }
        lower_mean = min(reference_means.values())
        upper_mean = max(reference_means.values())
        if ultra_mean > upper_mean:
            side = "higher"
            same_side = sum(
                float(indexed[(task, chain)]["headline_value"])
                > max(float(references[family][task][chain]) for family in FAMILIES)
                for chain in CHAINS
            )
        elif ultra_mean < lower_mean:
            side = "lower"
            same_side = sum(
                float(indexed[(task, chain)]["headline_value"])
                < min(float(references[family][task][chain]) for family in FAMILIES)
                for chain in CHAINS
            )
        else:
            side = "inside_reference_mean_interval"
            same_side = 0
        abstract_tasks[task] = {
            "eligible_for_abstract_mention": bool(
                side in {"higher", "lower"} and same_side >= 5
            ),
            "same_side_of_both_chain_count": int(same_side),
            "side": side,
            "ultra_mean": ultra_mean,
            "reference_mean_interval_closed": [lower_mean, upper_mean],
        }
    abstract_rule = {
        "tasks": abstract_tasks,
        "abstract_should_mention_ultra": any(
            row["eligible_for_abstract_mention"] for row in abstract_tasks.values()
        ),
    }
    formal_reporting = {
        "abstract_mention_rule": abstract_tasks,
        "abstract_should_mention_ultra": abstract_rule["abstract_should_mention_ultra"],
        "all_18_chain_task_headlines": [
            {
                "chain": row["chain"],
                "metric": row["headline_metric"],
                "task": row["task"],
                "value": row["headline_value"],
            }
            for row in records
        ],
        "forbidden_claims": [
            "fair-compute comparison",
            "champion claim",
            "statistical significance claim",
        ],
        "model_label": "external pretrained zero-shot",
        "trained_reference_comparisons": formal_comparisons,
        "unweighted_six_chain_headline_means": {
            task: task_summaries[task]["unweighted_six_chain_headline_mean"] for task in TASKS
        },
        "unweighted_six_chain_value_means": {
            task: task_summaries[task]["unweighted_six_chain_value_mean"] for task in TASKS
        },
    }
    return task_summaries, comparisons, abstract_rule, formal_reporting


def _expected_sheep_repeat(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "sentinel_chain",
        "runs",
        "same_process",
        "same_device",
        "same_model_instance",
        "primary_run_fixed",
        "score_gate_pass",
        "metric_gate_pass",
        "A_score_file_bytes_equal",
        "B_score_file_bytes_equal",
        "score_vector_hash_equal",
        "max_absolute_score_delta",
        "mean_absolute_score_delta",
        "metric_absolute_delta_max",
        "metric_absolute_deltas",
    }
    row = _exact_keys(value, fields, "sheep_exact_repeat")
    expected_fixed = {
        "sentinel_chain": "sheep",
        "runs": 2,
        "same_process": True,
        "same_device": True,
        "same_model_instance": True,
        "primary_run_fixed": True,
        "score_gate_pass": True,
        "metric_gate_pass": True,
        "A_score_file_bytes_equal": True,
        "B_score_file_bytes_equal": True,
        "score_vector_hash_equal": True,
    }
    for field, expected in expected_fixed.items():
        if row.get(field) != expected:
            _fail(f"sheep_exact_repeat.{field}", f"expected {expected!r}")
    for field in ("max_absolute_score_delta", "mean_absolute_score_delta"):
        if _finite(row.get(field), f"sheep_exact_repeat.{field}") != 0.0:
            _fail(f"sheep_exact_repeat.{field}", "exact repeat requires zero")
    threshold = _finite(row.get("metric_absolute_delta_max"), "sheep repeat threshold")
    if threshold != 1e-10:
        _fail("sheep_exact_repeat.metric_absolute_delta_max", "frozen threshold changed")
    deltas = _exact_keys(
        row.get("metric_absolute_deltas"), set(REPEAT_METRICS), "sheep repeat metric deltas"
    )
    cleaned_deltas = {}
    for metric in REPEAT_METRICS:
        delta = _finite(deltas.get(metric), f"sheep repeat {metric}")
        if delta != 0.0:
            _fail(f"sheep repeat {metric}", "exact repeat requires zero")
        cleaned_deltas[metric] = 0.0
    return {**expected_fixed, "max_absolute_score_delta": 0.0, "mean_absolute_score_delta": 0.0, "metric_absolute_delta_max": 1e-10, "metric_absolute_deltas": cleaned_deltas}


def _validate_component_and_scores(
    private_root: Path,
    chain: str,
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    component_path = private_root / "components" / chain / "component.json"
    component, component_raw = _formal_json(component_path, f"{chain} component")
    if _sha256_bytes(component_raw) != entry.get("sha256"):
        _fail(f"{chain} component", "seal component hash mismatch")
    required = {
        "schema_version": "upgrade-bench-v2/ultra-formal-score-component/2",
        "protocol": PROTOCOL,
        "status": "complete_label_blind_scores",
        "run_id": RUN_ID,
        "chain": chain,
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "main_target_labels_accessed": False,
        "main_label_derived_columns_accessed": False,
        "training_or_fine_tuning_performed": False,
        "selection_performed": False,
    }
    for field, expected in required.items():
        if component.get(field) != expected:
            _fail(f"{chain} component", f"invalid {field}")
    if component.get("source_sha256") != manifest.get("source_sha256"):
        _fail(f"{chain} component", "source hash receipt changed")
    backend = component.get("native_backend")
    if not isinstance(backend, Mapping):
        _fail(f"{chain} component", "native backend receipt is missing")
    if backend.get("backend") != "official_ULTRA_native_rspmm_and_native_torch_scatter":
        _fail(f"{chain} component", "native backend changed")
    if backend.get("compatibility_shim") is not False or backend.get("message_passing_fallback") is not False:
        _fail(f"{chain} component", "fallback/shim is not forbidden")
    runtime = dict(backend)
    runtime.pop("device", None)
    runtime.pop("device_name", None)
    runtime_sha256 = _sha256_bytes(_canonical_json_bytes(runtime))
    if entry.get("native_runtime_sha256") != runtime_sha256:
        _fail(f"{chain} component", "native runtime receipt mismatch")
    scores = component.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != {"A", "B"}:
        _fail(f"{chain} component", "primary score receipts are incomplete")
    for source in ("A", "B"):
        score_path = private_root / "components" / chain / f"scores_{source}.csv"
        observed = _sha256_file(score_path)
        if scores[source].get("sha256") != observed or entry.get(f"{source}_score_sha256") != observed:
            _fail(f"{chain}/{source}", "score receipt mismatch")
    repeat_public = None
    if chain == "sheep":
        repeats = component.get("repeat_scores")
        repeatability = component.get("repeatability")
        if not isinstance(repeats, Mapping) or not isinstance(repeatability, Mapping):
            _fail("sheep repeat", "repeat receipts are missing")
        exact_files = {}
        for source in ("A", "B"):
            primary = private_root / "components" / chain / f"scores_{source}.csv"
            repeat = private_root / "components" / chain / f"scores_{source}_repeat.csv"
            primary_raw = primary.read_bytes()
            repeat_raw = repeat.read_bytes()
            repeat_hash = _sha256_bytes(repeat_raw)
            if repeats[source].get("sha256") != repeat_hash or entry.get(f"{source}_repeat_score_sha256") != repeat_hash:
                _fail(f"sheep/{source} repeat", "repeat score receipt mismatch")
            exact_files[source] = primary_raw == repeat_raw
        contract = repeatability.get("contract")
        if not isinstance(contract, Mapping):
            _fail("sheep repeat", "contract is missing")
        fixed_contract = {
            "runs": 2,
            "same_process": True,
            "same_device": True,
            "same_model_instance": True,
            "primary_run": "run1 fixed before scoring; repeat cannot select or replace it",
        }
        for field, expected in fixed_contract.items():
            if contract.get(field) != expected:
                _fail("sheep repeat", f"contract {field} changed")
        exact_vector = repeatability.get("primary_score_vector_sha256") == repeatability.get(
            "repeat_score_vector_sha256"
        )
        if (
            not all(exact_files.values())
            or repeatability.get("exact_hash_equality") is not True
            or not exact_vector
            or repeatability.get("numeric_allclose") is not True
            or float(repeatability.get("max_absolute_score_delta", -1.0)) != 0.0
            or float(repeatability.get("mean_absolute_score_delta", -1.0)) != 0.0
        ):
            _fail("sheep repeat", "score repeat is not exact")
        repeat_public = {
            "sentinel_chain": "sheep",
            "runs": 2,
            "same_process": True,
            "same_device": True,
            "same_model_instance": True,
            "primary_run_fixed": True,
            "score_gate_pass": True,
            "metric_gate_pass": True,
            "A_score_file_bytes_equal": True,
            "B_score_file_bytes_equal": True,
            "score_vector_hash_equal": True,
            "max_absolute_score_delta": 0.0,
            "mean_absolute_score_delta": 0.0,
            "metric_absolute_delta_max": 1e-10,
            "metric_absolute_deltas": {metric: 0.0 for metric in REPEAT_METRICS},
        }
    elif component.get("repeat_scores") is not None:
        _fail(f"{chain} component", "unexpected repeat scores")
    return {"runtime_sha256": runtime_sha256}, repeat_public


def build_summary(private_root: Path = CANONICAL_PRIVATE_ROOT) -> dict[str, Any]:
    """Validate the complete extracted formal tree and build an allowlisted summary."""
    private_root = Path(private_root).resolve()
    config, config_sha256, trained, trained_sha256 = _load_config_and_references()
    references = _trained_reference_values(trained)

    manifest, manifest_raw = _formal_json(private_root / "frozen_manifest.json", "freeze manifest")
    manifest_sha256 = _sha256_bytes(manifest_raw)
    manifest_required = {
        "schema_version": "upgrade-bench-v2/ultra-formal-freeze/2",
        "protocol": PROTOCOL,
        "status": "frozen_before_target_scoring",
        "run_id": RUN_ID,
        "main_target_labels_accessed": False,
    }
    for field, expected in manifest_required.items():
        if manifest.get(field) != expected:
            _fail("freeze manifest", f"invalid {field}")
    if manifest.get("config", {}).get("sha256") != config_sha256:
        _fail("freeze manifest", "config hash mismatch")
    formal_controller_sha256 = _sha256_file(CANONICAL_FORMAL_CONTROLLER)
    if manifest.get("source_sha256", {}).get("tools/v2_ultra_formal.py") != formal_controller_sha256:
        _fail("freeze manifest", "formal controller hash mismatch")
    if manifest.get("reporting_contract") != config.get("reporting_contract"):
        _fail("freeze manifest", "reporting contract changed")
    if manifest.get("model_policy") != config.get("model_policy"):
        _fail("freeze manifest", "model policy changed")
    if list(manifest.get("chains", {})) != sorted(CHAINS):
        _fail("freeze manifest", "chain map is not the exact sorted JSON matrix")
    for chain in CHAINS:
        accounting = manifest["chains"][chain].get("cohort_accounting")
        if not isinstance(accounting, Mapping) or accounting.get("A_B_identity_overlap_rows") != 0:
            _fail(f"freeze manifest {chain}", "A/B identity overlap is not zero")
        for source in ("A", "B"):
            if manifest["chains"][chain]["cohorts"][source]["early_trade_overlap"].get("overlap_rows") != 0:
                _fail(f"freeze manifest {chain}/{source}", "early-trade overlap is not zero")

    _scoring_started, scoring_started_sha256 = _validate_scoring_started(
        private_root, manifest_sha256
    )

    seal, seal_raw = _formal_json(private_root / "SCORES_COMPLETE.json", "score seal")
    seal_sha256 = _sha256_bytes(seal_raw)
    seal_required = {
        "schema_version": SCORE_SEAL_SCHEMA,
        "protocol": PROTOCOL,
        "status": "all_six_chains_scored_labels_unlocked",
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "scoring_started_sha256": scoring_started_sha256,
        "component_count": 6,
        "sentinel_repeat_verified_before_label_unlock": True,
        "main_target_labels_accessed_before_seal": False,
    }
    for field, expected in seal_required.items():
        if seal.get(field) != expected:
            _fail("score seal", f"invalid {field}")
    native_runtime_sha256 = _hash(seal.get("native_runtime_sha256"), "score seal native runtime")
    components = seal.get("components")
    if not isinstance(components, list) or [row.get("chain") for row in components] != list(CHAINS):
        _fail("score seal", "component matrix/order is not exact")
    sheep_repeat = None
    for chain, entry in zip(CHAINS, components):
        if not isinstance(entry, Mapping):
            _fail("score seal", "component receipt is not an object")
        receipt, repeat = _validate_component_and_scores(
            private_root, chain, entry, manifest, manifest_sha256, config_sha256
        )
        if receipt["runtime_sha256"] != native_runtime_sha256:
            _fail(f"{chain} component", "six-chain native runtime differs")
        if repeat is not None:
            sheep_repeat = repeat
    if sheep_repeat is None:
        _fail("sheep repeat", "exact repeat receipt is absent")

    marker, marker_raw = _formal_json(
        private_root / "LABEL_EVALUATION_STARTED.json", "evaluation-start marker"
    )
    marker_sha256 = _sha256_bytes(marker_raw)
    marker_required = {
        "schema_version": EVALUATION_START_SCHEMA,
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "score_seal_sha256": seal_sha256,
        "scoring_started_sha256": scoring_started_sha256,
        "ordering_attestation": "all six A/B score components completed before first target-label read",
    }
    for field, expected in marker_required.items():
        if marker.get(field) != expected:
            _fail("evaluation-start marker", f"invalid {field}")
    if not isinstance(marker.get("started_at_utc"), str) or ISO_UTC.fullmatch(marker["started_at_utc"]) is None:
        _fail("evaluation-start marker", "timestamp is invalid")

    evaluation, evaluation_raw = _formal_json(private_root / "evaluation.json", "evaluation")
    evaluation_sha256 = _sha256_bytes(evaluation_raw)
    checkpoint_sha256 = _hash(config["checkpoint"]["sha256"], "checkpoint")
    evaluation_required = {
        "schema_version": EVALUATION_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "score_seal_sha256": seal_sha256,
        "scoring_started_sha256": scoring_started_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_policy": config["model_policy"],
        "task_contract": config["task_contract"],
        "reporting_contract": config["reporting_contract"],
    }
    for field, expected in evaluation_required.items():
        if evaluation.get(field) != expected:
            _fail("evaluation", f"invalid {field}")
    ordering = evaluation.get("ordering")
    if not isinstance(ordering, Mapping) or ordering.get("score_component_count_before_label_read") != 6 or ordering.get("all_six_score_components_verified_before_label_read") is not True:
        _fail("evaluation", "pre-label ordering receipt is invalid")
    if marker["started_at_utc"] > str(ordering.get("first_target_label_read_at_utc", "")):
        _fail("evaluation", "label read precedes its start marker")

    refs = evaluation.get("metric_artifacts")
    if not isinstance(refs, list) or [row.get("chain") for row in refs] != list(CHAINS):
        _fail("evaluation", "metric artifact matrix/order is not exact")
    metric_hashes: dict[str, str] = {}
    metrics_by_chain: dict[str, Mapping[str, Any]] = {}
    target_hashes: set[str] = set()
    for chain, ref in zip(CHAINS, refs):
        metric, metric_raw = _formal_json(
            private_root / "metrics" / f"metrics_{chain}.json", f"{chain} metrics"
        )
        metric_hash = _sha256_bytes(metric_raw)
        if ref.get("sha256") != metric_hash:
            _fail(f"{chain} metrics", "evaluation hash receipt mismatch")
        metric_hashes[chain] = metric_hash
        required = {
            "schema_version": METRIC_SCHEMA,
            "protocol": PROTOCOL,
            "status": "complete",
            "run_id": RUN_ID,
            "chain": chain,
            "checkpoint_sha256": checkpoint_sha256,
            "score_seal_sha256": seal_sha256,
            "scoring_started_sha256": scoring_started_sha256,
            "task_contract": config["task_contract"],
        }
        for field, expected in required.items():
            if metric.get(field) != expected:
                _fail(f"{chain} metrics", f"invalid {field}")
        targets = metric.get("target_sources")
        if not isinstance(targets, Mapping) or set(targets) != {"A", "B"}:
            _fail(f"{chain} metrics", "target receipts are incomplete")
        seal_entry = components[CHAINS.index(chain)]
        for source in ("A", "B"):
            for field in ("sha256", "precommitted_sha256"):
                target_hashes.add(_hash(targets[source].get(field), f"{chain}/{source} {field}"))
            if targets[source].get("score_sha256") != seal_entry.get(f"{source}_score_sha256"):
                _fail(f"{chain}/{source}", "metric-to-seal score receipt mismatch")
        task_metrics = metric.get("metrics")
        if not isinstance(task_metrics, Mapping) or set(task_metrics) != set(TASKS):
            _fail(f"{chain} metrics", "A/B1/B2 results are incomplete")
        metrics_by_chain[chain] = task_metrics
        if chain == "sheep":
            repeat_metrics = metric.get("repeat_metrics")
            gate = metric.get("repeatability_metric_gate")
            if repeat_metrics != task_metrics or not isinstance(gate, Mapping) or gate.get("all_metrics_pass") is not True:
                _fail("sheep metric repeat", "repeat metrics are not exact")
            if gate.get("metric_absolute_delta_max") != 1e-10:
                _fail("sheep metric repeat", "metric threshold changed")
            gate_rows = gate.get("metrics")
            if not isinstance(gate_rows, Mapping) or set(gate_rows) != set(REPEAT_METRICS):
                _fail("sheep metric repeat", "gate matrix is incomplete")
            for name in REPEAT_METRICS:
                row = gate_rows[name]
                if row.get("passed") is not True or float(row.get("absolute_delta", -1.0)) != 0.0 or row.get("primary_run1") != row.get("repeat_run2"):
                    _fail(f"sheep metric repeat {name}", "metric is not exactly repeated")
        elif metric.get("repeat_metrics") is not None or metric.get("repeatability_metric_gate") is not None:
            _fail(f"{chain} metrics", "unexpected repeat result")

    records = []
    for task in TASKS:
        for chain in CHAINS:
            values = metrics_by_chain[chain][task]
            records.append(
                {
                    "chain": chain,
                    "task": task,
                    "headline_metric": HEADLINE_METRICS[task],
                    "headline_value": _finite(
                        values.get(HEADLINE_METRICS[task]), f"{chain}/{task} headline"
                    ),
                    "value_metric": VALUE_METRICS[task],
                    "value_value": _finite(
                        values.get(VALUE_METRICS[task]), f"{chain}/{task} value"
                    ),
                }
            )
    task_summaries, comparisons, abstract_rule, formal_reporting = _derive_reporting(
        records, references
    )
    _nested_close(evaluation.get("reporting_summary"), formal_reporting, "formal reporting summary")

    generator_sha256 = _sha256_file(Path(__file__).resolve())
    provenance_receipts = {
        "config_sha256": config_sha256,
        "formal_controller_sha256": formal_controller_sha256,
        "generator_tool_sha256": generator_sha256,
        "trained_reference_summary_sha256": trained_sha256,
        "frozen_manifest_sha256": manifest_sha256,
        "scoring_started_sha256": scoring_started_sha256,
        "score_seal_sha256": seal_sha256,
        "evaluation_start_marker_sha256": marker_sha256,
        "evaluation_sha256": evaluation_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "native_runtime_sha256": native_runtime_sha256,
        "chain_metric_artifact_sha256": metric_hashes,
    }
    provenance = {
        **provenance_receipts,
        "formal_receipt_set_sha256": _sha256_bytes(_canonical_json_bytes(provenance_receipts)),
    }
    result = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": STATUS,
        "run_id": RUN_ID,
        "model": dict(MODEL),
        "provenance": provenance,
        "design": dict(DESIGN),
        "metric_records": records,
        "task_summaries": task_summaries,
        "reference_comparisons": comparisons,
        "abstract_rule": abstract_rule,
        "sheep_exact_repeat": sheep_repeat,
        "claim_boundaries": dict(CLAIM_BOUNDARIES),
    }
    sanitized = validate_public_summary(result)
    rendered = render_json(sanitized).decode("utf-8")
    leaked = sorted(digest for digest in target_hashes if digest in rendered)
    if leaked:
        _fail("public summary", "a target-table hash leaked")
    return sanitized


def validate_public_summary(value: Any) -> dict[str, Any]:
    """Validate and mechanically reconstruct a public ULTRA summary."""
    top_fields = {
        "schema_version",
        "protocol",
        "status",
        "run_id",
        "model",
        "provenance",
        "design",
        "metric_records",
        "task_summaries",
        "reference_comparisons",
        "abstract_rule",
        "sheep_exact_repeat",
        "claim_boundaries",
    }
    top = _exact_keys(value, top_fields, "public summary")
    fixed = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": STATUS,
        "run_id": RUN_ID,
        "model": MODEL,
        "design": DESIGN,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    for field, expected in fixed.items():
        if top.get(field) != expected:
            _fail(f"public summary.{field}", "fixed public contract changed")

    config, config_sha256, trained, trained_sha256 = _load_config_and_references()
    formal_controller_sha256 = _sha256_file(CANONICAL_FORMAL_CONTROLLER)
    generator_sha256 = _sha256_file(Path(__file__).resolve())
    provenance_fields = {
        "config_sha256",
        "formal_controller_sha256",
        "generator_tool_sha256",
        "trained_reference_summary_sha256",
        "frozen_manifest_sha256",
        "scoring_started_sha256",
        "score_seal_sha256",
        "evaluation_start_marker_sha256",
        "evaluation_sha256",
        "checkpoint_sha256",
        "native_runtime_sha256",
        "chain_metric_artifact_sha256",
        "formal_receipt_set_sha256",
    }
    provenance = _exact_keys(top.get("provenance"), provenance_fields, "public provenance")
    live = {
        "config_sha256": config_sha256,
        "formal_controller_sha256": formal_controller_sha256,
        "generator_tool_sha256": generator_sha256,
        "trained_reference_summary_sha256": trained_sha256,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
    }
    for field, expected in live.items():
        if provenance.get(field) != expected:
            _fail(f"public provenance.{field}", "current public dependency hash differs")
    for field in (
        "frozen_manifest_sha256",
        "scoring_started_sha256",
        "score_seal_sha256",
        "evaluation_start_marker_sha256",
        "evaluation_sha256",
        "native_runtime_sha256",
    ):
        _hash(provenance.get(field), f"public provenance.{field}")
    metric_hashes = _exact_keys(
        provenance.get("chain_metric_artifact_sha256"),
        set(CHAINS),
        "public provenance chain metrics",
    )
    cleaned_metric_hashes = {
        chain: _hash(metric_hashes.get(chain), f"public provenance metric {chain}")
        for chain in CHAINS
    }
    receipts = {
        "config_sha256": config_sha256,
        "formal_controller_sha256": formal_controller_sha256,
        "generator_tool_sha256": generator_sha256,
        "trained_reference_summary_sha256": trained_sha256,
        "frozen_manifest_sha256": provenance["frozen_manifest_sha256"],
        "scoring_started_sha256": provenance["scoring_started_sha256"],
        "score_seal_sha256": provenance["score_seal_sha256"],
        "evaluation_start_marker_sha256": provenance["evaluation_start_marker_sha256"],
        "evaluation_sha256": provenance["evaluation_sha256"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "native_runtime_sha256": provenance["native_runtime_sha256"],
        "chain_metric_artifact_sha256": cleaned_metric_hashes,
    }
    expected_receipt_hash = _sha256_bytes(_canonical_json_bytes(receipts))
    if provenance.get("formal_receipt_set_sha256") != expected_receipt_hash:
        _fail("public provenance.formal_receipt_set_sha256", "receipt set is not bound")

    records = _validate_metric_records(top.get("metric_records"))
    references = _trained_reference_values(trained)
    task_summaries, comparisons, abstract_rule, _formal = _derive_reporting(records, references)
    _nested_close(top.get("task_summaries"), task_summaries, "task_summaries")
    _nested_close(top.get("reference_comparisons"), comparisons, "reference_comparisons")
    _nested_close(top.get("abstract_rule"), abstract_rule, "abstract_rule")
    sheep_repeat = _expected_sheep_repeat(top.get("sheep_exact_repeat"))

    cleaned = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "status": STATUS,
        "run_id": RUN_ID,
        "model": dict(MODEL),
        "provenance": {**receipts, "formal_receipt_set_sha256": expected_receipt_hash},
        "design": dict(DESIGN),
        "metric_records": records,
        "task_summaries": task_summaries,
        "reference_comparisons": comparisons,
        "abstract_rule": abstract_rule,
        "sheep_exact_repeat": sheep_repeat,
        "claim_boundaries": dict(CLAIM_BOUNDARIES),
    }
    _privacy_audit(cleaned)
    return cleaned


CSV_FIELDS = (
    "row_type",
    "chain",
    "task",
    "reference_family",
    "headline_metric",
    "headline_value",
    "value_metric",
    "value_value",
    "higher",
    "equal",
    "lower",
    "reference_unweighted_six_chain_mean",
    "abstract_side",
    "same_side_of_both_chain_count",
    "eligible_for_abstract_mention",
    "config_sha256",
    "formal_controller_sha256",
    "generator_tool_sha256",
    "trained_reference_summary_sha256",
    "formal_receipt_set_sha256",
    "run_id",
    "protocol",
)


def render_json(summary: Mapping[str, Any]) -> bytes:
    _privacy_audit(summary)
    return _canonical_json_bytes(summary, pretty=True)


def render_csv(summary: Mapping[str, Any]) -> bytes:
    _privacy_audit(summary)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    provenance = summary["provenance"]
    common = {
        "config_sha256": provenance["config_sha256"],
        "formal_controller_sha256": provenance["formal_controller_sha256"],
        "generator_tool_sha256": provenance["generator_tool_sha256"],
        "trained_reference_summary_sha256": provenance["trained_reference_summary_sha256"],
        "formal_receipt_set_sha256": provenance["formal_receipt_set_sha256"],
        "run_id": summary["run_id"],
        "protocol": summary["protocol"],
    }
    for row in summary["metric_records"]:
        writer.writerow(
            {
                **common,
                "row_type": "chain_task",
                "chain": row["chain"],
                "task": row["task"],
                "headline_metric": row["headline_metric"],
                "headline_value": format(float(row["headline_value"]), ".17g"),
                "value_metric": row["value_metric"],
                "value_value": format(float(row["value_value"]), ".17g"),
            }
        )
    for task in TASKS:
        row = summary["task_summaries"][task]
        writer.writerow(
            {
                **common,
                "row_type": "task_summary",
                "chain": "__six_chain_mean__",
                "task": task,
                "headline_metric": row["headline_metric"],
                "headline_value": format(
                    float(row["unweighted_six_chain_headline_mean"]), ".17g"
                ),
                "value_metric": row["value_metric"],
                "value_value": format(
                    float(row["unweighted_six_chain_value_mean"]), ".17g"
                ),
            }
        )
        for family in FAMILIES:
            comparison = summary["reference_comparisons"][task][family]
            writer.writerow(
                {
                    **common,
                    "row_type": "reference_comparison",
                    "chain": "__all__",
                    "task": task,
                    "reference_family": family,
                    "headline_metric": row["headline_metric"],
                    "higher": comparison["counts"]["higher"],
                    "equal": comparison["counts"]["equal"],
                    "lower": comparison["counts"]["lower"],
                    "reference_unweighted_six_chain_mean": format(
                        float(comparison["reference_unweighted_six_chain_mean"]), ".17g"
                    ),
                }
            )
        abstract = summary["abstract_rule"]["tasks"][task]
        writer.writerow(
            {
                **common,
                "row_type": "abstract_rule",
                "chain": "__all__",
                "task": task,
                "headline_metric": row["headline_metric"],
                "headline_value": format(float(abstract["ultra_mean"]), ".17g"),
                "abstract_side": abstract["side"],
                "same_side_of_both_chain_count": abstract["same_side_of_both_chain_count"],
                "eligible_for_abstract_mention": str(
                    bool(abstract["eligible_for_abstract_mention"])
                ).lower(),
            }
        )
    encoded = buffer.getvalue().encode("utf-8")
    rendered = encoded.decode("utf-8")
    for label, pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(rendered):
            _fail("public CSV", f"operational locator leaked: {label}")
    return encoded


def _same_parent(left: Path, right: Path) -> bool:
    return Path(left).absolute().parent == Path(right).absolute().parent


def _atomic_create_or_match(path: Path, content: bytes) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        _fail("published output", f"{path.name} is a symbolic link")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            _fail("published output", f"refusing to overwrite non-matching {path.name}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                _fail("published output", f"concurrent non-matching {path.name}")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_outputs(
    private_root: Path = CANONICAL_PRIVATE_ROOT,
    json_out: Path = DEFAULT_JSON_OUT,
    csv_out: Path = DEFAULT_CSV_OUT,
) -> dict[str, Any]:
    summary = build_summary(private_root)
    if not _same_parent(json_out, csv_out):
        _fail("published output", "JSON and CSV must share one directory")
    _atomic_create_or_match(json_out, render_json(summary))
    _atomic_create_or_match(csv_out, render_csv(summary))
    return verify_outputs(json_out, csv_out)


def verify_outputs(
    json_out: Path = DEFAULT_JSON_OUT,
    csv_out: Path = DEFAULT_CSV_OUT,
) -> dict[str, Any]:
    """Verify public bytes without reading the extracted formal result tree."""
    if not _same_parent(json_out, csv_out):
        _fail("published output", "JSON and CSV must share one directory")
    observed, raw = _strict_json_file(json_out, "published JSON")
    summary = validate_public_summary(observed)
    if raw != render_json(summary):
        _fail("published output", f"{Path(json_out).name} is non-canonical or stale")
    try:
        csv_raw = Path(csv_out).read_bytes()
    except OSError as exc:
        _fail("published output", f"cannot read {Path(csv_out).name}: {exc}")
    if csv_raw != render_csv(summary):
        _fail("published output", f"{Path(csv_out).name} is stale or non-deterministic")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-output",
        action="store_true",
        help="verify public JSON/CSV using only public dependencies and tool hashes",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="validate the extracted formal result without writing public outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_output:
            summary = verify_outputs()
            action = "public outputs verified"
        elif args.check_only:
            summary = build_summary()
            action = "formal extraction validated (no files written)"
        else:
            summary = write_outputs()
            action = "public outputs written"
    except ResultValidationError as exc:
        print(f"ULTRA RESULT PROMOTION REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        f"ULTRA zero-shot summary {action}: "
        f"{len(summary['metric_records'])} chain-task rows; "
        f"abstract_should_mention_ultra={summary['abstract_rule']['abstract_should_mention_ultra']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
