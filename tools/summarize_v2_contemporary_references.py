"""Promote and verify the five fixed-configuration contemporary references.

The public profile is self-contained.  It verifies a sanitized JSON/CSV/TeX
interface containing 90 method-chain-task five-seed summaries and 15 six-chain
summaries.  The full profile additionally rehashes the private source artifacts,
re-runs the fixed collector and summarizer, exactly rebuilds the final reporting
receipt, and proves that the public matrices are exact projections of that sealed
five-seed result.

Raw scores, per-seed artifact hashes, machine locators, and operational timestamps
are intentionally omitted from the public interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    ROOT / "tmp" / "contemporary_refs_multiseed" / "20260722-r1"
)
DEFAULT_CONFIG = ROOT / "configs" / "v2_contemporary_references.json"
DEFAULT_JSON_OUT = ROOT / "results_v2" / "metrics" / "v2_contemporary_references.json"
DEFAULT_CSV_OUT = ROOT / "results_v2" / "metrics" / "v2_contemporary_references.csv"
DEFAULT_TEX_OUT = ROOT / "paper" / "generated" / "v2_contemporary_references.tex"

CONFIG_SCHEMA = "upgrade-bench-v2/contemporary-reference-public-config/2"
PUBLIC_SCHEMA = "upgrade-bench-v2/contemporary-reference-public-summary/2"
PUBLIC_STATUS = "complete_verified_sanitized_fixed_configuration_five_seed"
RUN_ID = "contemporary-multiseed-20260722-r1"

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
TASKS = ("A", "B1", "B2")
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("motif", "flock", "tabm", "xrfm", "tabiclv2")
GRAPH_METHODS = ("motif", "flock")
TABULAR_METHODS = ("tabm", "xrfm", "tabiclv2")
EXPECTED_CONFIGURATION_IDS = {
    "motif": "MOTIF_inference_transductive",
    "flock": "zeroshot_transductive_n16_ensemble16",
    "tabm": "tabm_k8_d64_b2_e32",
    "xrfm": "xrfm_gpu_leaf60000_l2_bw5_reg1e-3_i1",
    "tabiclv2": "tabiclv2_e2_amp0",
}
METRIC_CONTRACT = {
    "A": {"headline_metric": "average_precision", "value_metric": "value_capture_at_500"},
    "B1": {"headline_metric": "average_precision", "value_metric": "value_capture_at_50"},
    "B2": {"headline_metric": "macro_recall_at_3", "value_metric": "macro_value_capture_at_3"},
}
AGGREGATION_CONTRACT = {
    "score_averaging_before_metric_evaluation": False,
    "seed_first": "compute each task metric independently for every method-chain-seed bundle",
    "per_chain": "arithmetic mean and population standard deviation across seeds 0-4",
    "six_chain": (
        "for each seed take the unweighted mean of six chain metrics, then report "
        "the arithmetic mean and population standard deviation across the five seed macros"
    ),
    "partial_success_means": "forbidden",
    "population_standard_deviation_ddof": 0,
}

METHOD_TEX = {
    "motif": "MOTIF",
    "flock": "Flock",
    "tabm": "TabM",
    "xrfm": "XRFM",
    "tabiclv2": "TabICLvTwo",
}
TASK_TEX = {"A": "A", "B1": "BOne", "B2": "BTwo"}

SOURCE_PATHS = {
    "multiseed_config": Path("MULTISEED_CONFIG.json"),
    "static_task_manifest": Path("STATIC_TASK_MANIFEST.json"),
    "summarizer": Path("summarize_multiseed.py"),
    "global_controller": Path("global/global_controller.py"),
    "collector": Path("reporting/collect_task_metrics.py"),
    "final_reporting_receipt_tool": Path("reporting/seal_final_reporting_receipt.py"),
    "collection_receipt": Path("reporting/collected_20260723_r1/collection_receipt.json"),
    "collected_task_metrics_jsonl": Path(
        "reporting/collected_20260723_r1/collected_task_metrics.jsonl"
    ),
    "multiseed_summary": Path("reporting/summary_20260723_r1/multiseed_summary.json"),
    "multiseed_task_rows_csv": Path(
        "reporting/summary_20260723_r1/multiseed_task_rows.csv"
    ),
    "multiseed_aggregates_csv": Path(
        "reporting/summary_20260723_r1/multiseed_aggregates.csv"
    ),
    "final_reporting_receipt": Path(
        "reporting/final_receipt_20260723_r1/FINAL_REPORTING_RECEIPT.json"
    ),
    "graph_evaluation": Path(
        "reporting/source_artifacts_20260723_r1/graph/evaluation.json"
    ),
    "graph_score_seal": Path(
        "reporting/source_artifacts_20260723_r1/graph/SCORES_COMPLETE.json"
    ),
    "tabular_evaluation": Path(
        "reporting/source_artifacts_20260723_r1/tabular/evaluation.json"
    ),
    "tabular_evaluation_by_seed_csv": Path(
        "reporting/source_artifacts_20260723_r1/tabular/evaluation_by_seed.csv"
    ),
    "tabular_per_chain_summary_csv": Path(
        "reporting/source_artifacts_20260723_r1/tabular/per_chain_summary.csv"
    ),
    "tabular_evaluation_receipt": Path(
        "reporting/source_artifacts_20260723_r1/tabular/receipt.json"
    ),
    "tabular_score_seal": Path(
        "reporting/source_artifacts_20260723_r1/tabular/score_seal.json"
    ),
    "global_score_seal": Path(
        "reporting/source_artifacts_20260723_r1/gate/seal_host/GLOBAL_SCORE_SEAL.json"
    ),
    "global_seal_contract": Path(
        "reporting/source_artifacts_20260723_r1/gate/seal_host/GLOBAL_SEAL_CONTRACT.json"
    ),
    "main_outcome_access_marker": Path(
        "reporting/source_artifacts_20260723_r1/gate/seal_host/MAIN_OUTCOME_ACCESS_STARTED.json"
    ),
    "gate_transfer_manifest": Path(
        "reporting/source_artifacts_20260723_r1/gate/materialization_host/GATE_TRANSFER_MANIFEST.json"
    ),
    "gate_materialization_receipt": Path(
        "reporting/source_artifacts_20260723_r1/gate/materialization_host/GLOBAL_GATE_MATERIALIZATION_R1.json"
    ),
}

EXPECTED_EVIDENCE_DIGESTS = {
    "final_reporting_receipt": "e9f6b09897974c12b08baac00cf07ad59ae74756e824b5480514ca4ebb0c4e04",
    "multiseed_summary": "a59a79abc12271868e8b0653aa0fd0c1ff8fbfaa393f083a3504bf1f9352d56d",
    "collection_receipt": "bd3538fab9bb9be2260e2319ca69d7852f6df04da1e04cac8715fff5629f24e9",
    "collected_task_metrics_jsonl": "579efad28a899a838ae7a65a497c44571ab1f3eb4429c6f4b96747134410f272",
    "multiseed_task_rows_csv": "16a23b51c161e643f5b1687a3156ccca017f8dce4e3a79cbd58ff9f596368d89",
    "multiseed_aggregates_csv": "cd065fdab5c36ee24247fc26719accb87a0e193a073f87e290003da56412f207",
    "graph_evaluation": "05bb9ea28b10c2ba26e7c798525ea3d4af08adc0db3ce44b4be37bc320dcd089",
    "graph_score_seal": "61b78b1ad6a0dc5891f9dfbf4357124e196fe3c98e98572f870191fc79810d85",
    "tabular_evaluation": "e9c6ca40ce3006e7dddac89101c54dbb089fca46cc82380d5388b80ea3c79419",
    "tabular_evaluation_by_seed_csv": "abcc64066114564c1c45f20904a668154b3830244caccbfbfe94579a2fbaa03e",
    "tabular_per_chain_summary_csv": "ad215fe161d211bfcda29f007c07ffe5d264597394c0f40531653b70af792eba",
    "tabular_evaluation_receipt": "9d5571def7b8b46a810990a6cc2af6dd38f404789a57c9023daa9550946f0859",
    "tabular_score_seal": "efac05bcdd8122a49d3cd4fa756132b0632ba3af17419d85c30a910353e4c88f",
    "global_score_seal": "a0f56753c8b75ee525d781b62bebba454004de535114268b0d3fb10770f7d608",
    "global_seal_contract": "e29b59b02f421af41684efecb3f217354c7960d31eab212e5f69bc84aecf75ef",
    "main_outcome_access_marker": "5a6e11e03f6fcb2f0764e45408a070d64dd84c897a2cd3b2faace6dfc4fb06b0",
    "gate_transfer_manifest": "cb7afcb9c7744c81cae38b0a4657dd96722940ef911879d059d5bf158cefe702",
    "gate_materialization_receipt": "7b674b0c31ab64de45a88f39052488cbd602ff8055ec14630362e6ec22028928",
    "multiseed_config": "a5456b16cb09799d3c5faa1742745e67d2f16d7005fd8124066986871e125ef2",
    "static_task_manifest": "6c853e0b7262cc9e6cfb9416a005eb797f25532030508b1ffc5390e0a35c80d6",
    "collector": "8585499de2b2b48b7b6486638e45887978440ca83b52a08f604f8c8180f5b7a7",
    "summarizer": "8ba2b7ecb4c3806d2b83a1ff6abb0203f71d01dd8ed5a129a9e2523b63cc21fd",
    "global_controller": "e32b2c22e28d74c871bd7ab48410b547404eecfdbf0038e83234eebe73500ba1",
    "final_reporting_receipt_tool": "bf0f7fd416a8dc58ef52775a9a1909507fca269d27e15acb2a589401bac6ccff",
}
EXPECTED_RECORD_MATRIX_SHA256 = "eab0b15bd1b2ece8347ce578d02ceae78a5eb424f4459a35c365254d3d0af541"
EXPECTED_AGGREGATE_MATRIX_SHA256 = "192b272953fdd7b0f9f7b030ba370382edaa2627e7e867945bd01f183fc38af0"

HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
TEX_MACRO_RE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}$")
PRIVATE_PATTERNS = (
    re.compile(r"/(?:home|users|scratch)/[^/\s]+/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE),
    re.compile(r"\\\\[^\\\s]+[\\/]", re.IGNORECASE),
    re.compile(r"\bfile://", re.IGNORECASE),
    re.compile(r"\bmars\d+\b", re.IGNORECASE),
    re.compile(r"\bsli\d+\b", re.IGNORECASE),
    re.compile(r"\bihpc\.uts\.edu\.au\b", re.IGNORECASE),
)
FORBIDDEN_PUBLIC_KEYS = {
    "host", "hostname", "user", "username", "pid", "root", "source_root",
    "repository_root", "candidate_root", "output_root", "run_root",
}


class ContemporaryReferenceError(ValueError):
    """A fail-closed promotion or verification check failed."""


def _fail(role: str, message: str) -> None:
    raise ContemporaryReferenceError(f"{role}: {message}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        _fail("JSON rendering", str(exc))
    return (rendered + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail("file hash", f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _strict_json_bytes(raw: bytes, role: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(role, f"invalid strict JSON: {exc}")
    if not isinstance(value, dict):
        _fail(role, "top-level value must be an object")
    return value


def _read_regular(path: Path, role: str) -> bytes:
    path = Path(path)
    if path.is_symlink():
        _fail(role, "symbolic-link input is forbidden")
    if not path.is_file():
        _fail(role, f"regular file is missing: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        _fail(role, f"cannot read {path}: {exc}")


def _strict_json_file(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, role)
    return _strict_json_bytes(raw, role), raw


def _hex64(value: Any, role: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        _fail(role, "expected a lowercase SHA-256 digest")
    return value


def _finite_unit(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(role, "expected a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        _fail(role, "expected a finite number in [0, 1]")
    return number


def _same_float(observed: Any, expected: float, role: str) -> None:
    value = _finite_unit(observed, role)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15):
        _fail(role, f"{value!r} != recomputed {expected!r}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        _fail("aggregation", "cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _population_sd(values: Sequence[float]) -> float:
    center = _mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / len(values))


def _exact_sequence(value: Any, expected: Sequence[Any], role: str) -> None:
    if not isinstance(value, list) or value != list(expected):
        _fail(role, f"expected {list(expected)!r}")


def _privacy_audit(value: Any, role: str = "public artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PUBLIC_KEYS or lowered.endswith("_path"):
                _fail(role, f"forbidden public field {key!r}")
            _privacy_audit(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _privacy_audit(child, f"{role}[{index}]")
    elif isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _fail(role, "absolute path leaked")
        if any(pattern.search(value) for pattern in PRIVATE_PATTERNS):
            _fail(role, "private machine or account locator leaked")


def _privacy_audit_bytes(raw: bytes, role: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(role, f"not UTF-8: {exc}")
    if any(pattern.search(text) for pattern in PRIVATE_PATTERNS):
        _fail(role, "private machine or account locator leaked")


def _load_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], bytes]:
    config, raw = _strict_json_file(path, "public fixed configuration")
    required_top = {
        "schema_version", "run_id", "status", "benchmark_version", "benchmark",
        "claim_scope", "metric_contract", "feature_and_scoring_contract", "methods",
        "formal_evidence",
    }
    if set(config) != required_top:
        _fail("public fixed configuration", "top-level fields changed")
    if config.get("schema_version") != CONFIG_SCHEMA:
        _fail("public fixed configuration", "schema version changed")
    if config.get("run_id") != RUN_ID:
        _fail("public fixed configuration", "run_id changed")
    if config.get("status") != "fixed_configuration_five_seed_references":
        _fail("public fixed configuration", "status changed")
    if config.get("benchmark_version") != "2.1-dev":
        _fail("public fixed configuration", "benchmark version changed")

    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        _fail("public fixed configuration", "benchmark is missing")
    _exact_sequence(benchmark.get("chains"), CHAINS, "configured chains")
    _exact_sequence(benchmark.get("tasks"), TASKS, "configured tasks")
    _exact_sequence(benchmark.get("seeds"), SEEDS, "configured seeds")
    expected_counts = {
        "expected_method_chain_seed_bundles": 150,
        "expected_seed_task_metric_records": 450,
        "expected_method_chain_task_records": 90,
        "expected_six_chain_aggregates": 15,
    }
    for field, expected in expected_counts.items():
        if benchmark.get(field) != expected:
            _fail("public fixed configuration", f"{field} differs")
    if benchmark.get("aggregation_contract") != AGGREGATION_CONTRACT:
        _fail("public fixed configuration", "aggregation contract changed")
    if config.get("metric_contract") != METRIC_CONTRACT:
        _fail("public fixed configuration", "metric contract changed")

    claim = config.get("claim_scope")
    if not isinstance(claim, Mapping):
        _fail("public fixed configuration", "claim scope is missing")
    if claim.get("classification") != "contemporary fixed-configuration references":
        _fail("public fixed configuration", "claim classification changed")
    for field in (
        "main_window_target_outcomes_used_for_training_or_configuration_selection",
        "historical_outcome_metrics_used_for_configuration_selection",
        "training_uncertainty_comparable_across_methods",
    ):
        if claim.get(field) is not False:
            _fail("public fixed configuration", f"{field} must be false")
    if claim.get("historical_labels_used_for_model_fitting") is not True:
        _fail("public fixed configuration", "historical-label scope changed")

    methods = config.get("methods")
    if not isinstance(methods, list) or len(methods) != len(METHODS):
        _fail("public fixed configuration", "requires exactly five methods")
    if [row.get("method") for row in methods if isinstance(row, Mapping)] != list(METHODS):
        _fail("public fixed configuration", "method order or membership changed")
    for row in methods:
        if not isinstance(row, Mapping):
            _fail("public fixed configuration", "method metadata must be objects")
        method = str(row["method"])
        family = "graph" if method in GRAPH_METHODS else "tabular"
        if row.get("family") != family:
            _fail("public fixed configuration", f"invalid family for {method}")
        if row.get("configuration_id") != EXPECTED_CONFIGURATION_IDS[method]:
            _fail("public fixed configuration", f"invalid configuration id for {method}")
        _exact_sequence(row.get("seeds"), SEEDS, f"{method} seeds")
        commit = row.get("upstream_commit")
        if method in GRAPH_METHODS:
            if not isinstance(commit, str) or HEX40_RE.fullmatch(commit) is None:
                _fail("public fixed configuration", f"invalid upstream commit for {method}")
        elif commit is not None:
            _fail("public fixed configuration", f"unattested commit for {method}")
        checkpoint = row.get("checkpoint_sha256")
        if checkpoint is not None:
            _hex64(checkpoint, f"{method} checkpoint")

    evidence = config.get("formal_evidence")
    if not isinstance(evidence, Mapping):
        _fail("public fixed configuration", "formal evidence is missing")
    expected_evidence_keys = {
        "final_reporting_receipt", "multiseed_summary", "collection_receipt",
        "collected_task_metrics_jsonl", "summary_outputs", "source_artifacts",
        "implementation_sha256", "multiseed_config_sha256", "static_task_manifest_sha256",
        "sanitized_record_matrix_sha256", "sanitized_aggregate_matrix_sha256",
    }
    if set(evidence) != expected_evidence_keys:
        _fail("public fixed configuration", "formal evidence fields changed")
    for role, schema, status in (
        (
            "final_reporting_receipt",
            "upgrade-bench-v2/contemporary-multiseed-final-reporting-receipt/1",
            "all_reporting_outputs_exactly_recomputed_and_provenance_bound",
        ),
        (
            "multiseed_summary",
            "upgrade-bench-v2/contemporary-multiseed-summary/1",
            "complete_all_150_bundles_all_450_task_rows",
        ),
        (
            "collection_receipt",
            "upgrade-bench-v2/contemporary-multiseed-collection-receipt/1",
            "complete_verified_180_graph_plus_270_tabular_rows",
        ),
    ):
        item = evidence.get(role)
        if not isinstance(item, Mapping) or item.get("schema_version") != schema:
            _fail("public fixed configuration", f"{role} schema changed")
        if item.get("status") != status:
            _fail("public fixed configuration", f"{role} status changed")
        if item.get("sha256") != EXPECTED_EVIDENCE_DIGESTS[role]:
            _fail("public fixed configuration", f"{role} digest changed")
    task_jsonl = evidence.get("collected_task_metrics_jsonl")
    if task_jsonl != {
        "row_count": 450,
        "sha256": EXPECTED_EVIDENCE_DIGESTS["collected_task_metrics_jsonl"],
    }:
        _fail("public fixed configuration", "collected task metric evidence changed")
    if evidence.get("summary_outputs") != {
        "multiseed_task_rows_csv_sha256": EXPECTED_EVIDENCE_DIGESTS["multiseed_task_rows_csv"],
        "multiseed_aggregates_csv_sha256": EXPECTED_EVIDENCE_DIGESTS["multiseed_aggregates_csv"],
    }:
        _fail("public fixed configuration", "summary output evidence changed")
    expected_sources = {
        f"{role}_sha256": EXPECTED_EVIDENCE_DIGESTS[role]
        for role in (
            "graph_evaluation", "graph_score_seal", "tabular_evaluation",
            "tabular_evaluation_by_seed_csv", "tabular_per_chain_summary_csv",
            "tabular_evaluation_receipt", "tabular_score_seal", "global_score_seal",
            "global_seal_contract", "main_outcome_access_marker", "gate_transfer_manifest",
            "gate_materialization_receipt",
        )
    }
    if evidence.get("source_artifacts") != expected_sources:
        _fail("public fixed configuration", "source artifact evidence changed")
    expected_implementations = {
        role: EXPECTED_EVIDENCE_DIGESTS[role]
        for role in ("collector", "summarizer", "global_controller", "final_reporting_receipt_tool")
    }
    if evidence.get("implementation_sha256") != expected_implementations:
        _fail("public fixed configuration", "implementation evidence changed")
    if evidence.get("multiseed_config_sha256") != EXPECTED_EVIDENCE_DIGESTS["multiseed_config"]:
        _fail("public fixed configuration", "multiseed config digest changed")
    if evidence.get("static_task_manifest_sha256") != EXPECTED_EVIDENCE_DIGESTS["static_task_manifest"]:
        _fail("public fixed configuration", "static manifest digest changed")
    if evidence.get("sanitized_record_matrix_sha256") != EXPECTED_RECORD_MATRIX_SHA256:
        _fail("public fixed configuration", "record matrix digest changed")
    if evidence.get("sanitized_aggregate_matrix_sha256") != EXPECTED_AGGREGATE_MATRIX_SHA256:
        _fail("public fixed configuration", "aggregate matrix digest changed")
    legacy = json.dumps(config, sort_keys=True).lower()
    if any(token in legacy for token in ("post-freeze", "post_freeze", "exploratory", "single-point")):
        _fail("public fixed configuration", "legacy exploratory or single-point wording remains")
    _privacy_audit(config, "public fixed configuration")
    return config, raw


def _source_path(source_root: Path, role: str) -> Path:
    raw_root = Path(source_root)
    if raw_root.is_symlink():
        _fail(f"formal {role}", "source root must not be a symbolic link")
    root = raw_root.resolve()
    relative = SOURCE_PATHS[role]
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"formal {role}", "symbolic-link input is forbidden")
    try:
        candidate = (root / relative).resolve(strict=True)
    except OSError as exc:
        _fail(f"formal {role}", f"cannot resolve input: {exc}")
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail(f"formal {role}", "input escapes the fixed source root")
    if not candidate.is_file():
        _fail(f"formal {role}", "input must be a regular non-symlink file")
    return candidate


def _load_fixed_module(path: Path, role: str, expected_sha256: str) -> ModuleType:
    raw = _read_regular(path, role)
    if _sha256_bytes(raw) != expected_sha256:
        _fail(role, "source digest differs from the fixed evidence")
    name = f"_upgrade_bench_public_{role.replace(' ', '_')}_{expected_sha256[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(role, "cannot construct import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(role, f"cannot import fixed implementation: {exc}")
    return module


def _validate_fixed_source_hashes(source_root: Path) -> None:
    roles = (
        "multiseed_config", "static_task_manifest", "summarizer", "global_controller",
        "collector", "final_reporting_receipt_tool", "collection_receipt",
        "collected_task_metrics_jsonl", "multiseed_summary", "multiseed_task_rows_csv",
        "multiseed_aggregates_csv", "final_reporting_receipt", "graph_evaluation",
        "graph_score_seal", "tabular_evaluation", "tabular_evaluation_by_seed_csv",
        "tabular_per_chain_summary_csv", "tabular_evaluation_receipt", "tabular_score_seal",
        "global_score_seal", "global_seal_contract", "main_outcome_access_marker",
        "gate_transfer_manifest", "gate_materialization_receipt",
    )
    for role in roles:
        observed = _sha256_file(_source_path(source_root, role))
        if observed != EXPECTED_EVIDENCE_DIGESTS[role]:
            _fail(f"formal {role}", "bytes differ from the fixed evidence")


def _recompute_collection(source_root: Path) -> None:
    collector = _load_fixed_module(
        _source_path(source_root, "collector"),
        "formal collector",
        EXPECTED_EVIDENCE_DIGESTS["collector"],
    )
    if not all(hasattr(collector, name) for name in ("build_payloads", "verify_outputs")):
        _fail("formal collector", "required recomputation interface is missing")
    args = SimpleNamespace(
        graph_evaluation=_source_path(source_root, "graph_evaluation"),
        graph_seal=_source_path(source_root, "graph_score_seal"),
        tabular_evaluation=_source_path(source_root, "tabular_evaluation"),
        tabular_by_seed=_source_path(source_root, "tabular_evaluation_by_seed_csv"),
        tabular_per_chain=_source_path(source_root, "tabular_per_chain_summary_csv"),
        tabular_receipt=_source_path(source_root, "tabular_evaluation_receipt"),
        tabular_seal=_source_path(source_root, "tabular_score_seal"),
        output_dir=_source_path(source_root, "collection_receipt").parent,
    )
    try:
        payloads = collector.build_payloads(args)
        collector.verify_outputs(payloads, args.output_dir)
    except Exception as exc:
        _fail("formal collector recomputation", str(exc))


def _recompute_final_receipt(source_root: Path, config: Mapping[str, Any]) -> None:
    tool = _load_fixed_module(
        _source_path(source_root, "final_reporting_receipt_tool"),
        "formal final receipt tool",
        EXPECTED_EVIDENCE_DIGESTS["final_reporting_receipt_tool"],
    )
    if not all(hasattr(tool, name) for name in ("build_receipt", "_canonical_receipt")):
        _fail("formal final receipt tool", "required recomputation interface is missing")
    args = SimpleNamespace(
        collection_receipt=_source_path(source_root, "collection_receipt"),
        collection_jsonl=_source_path(source_root, "collected_task_metrics_jsonl"),
        summary_json=_source_path(source_root, "multiseed_summary"),
        summary_task_csv=_source_path(source_root, "multiseed_task_rows_csv"),
        summary_aggregate_csv=_source_path(source_root, "multiseed_aggregates_csv"),
        global_score_seal=_source_path(source_root, "global_score_seal"),
        main_outcome_marker=_source_path(source_root, "main_outcome_access_marker"),
        global_contract=_source_path(source_root, "global_seal_contract"),
        gate_transfer_manifest=_source_path(source_root, "gate_transfer_manifest"),
        gate_materialization_receipt=_source_path(source_root, "gate_materialization_receipt"),
        config=_source_path(source_root, "multiseed_config"),
        manifest=_source_path(source_root, "static_task_manifest"),
        collector=_source_path(source_root, "collector"),
        summarizer=_source_path(source_root, "summarizer"),
        global_controller=_source_path(source_root, "global_controller"),
        output=_source_path(source_root, "final_reporting_receipt"),
    )
    try:
        expected = tool.build_receipt(args)
        expected_raw = tool._canonical_receipt(expected)
    except Exception as exc:
        _fail("formal final receipt recomputation", str(exc))
    observed, observed_raw = _strict_json_file(args.output, "formal final reporting receipt")
    if observed_raw != expected_raw or observed != expected:
        _fail("formal final reporting receipt", "differs from exact recomputation")
    evidence = config["formal_evidence"]
    if _sha256_bytes(observed_raw) != evidence["final_reporting_receipt"]["sha256"]:
        _fail("formal final reporting receipt", "digest differs from public evidence")
    if expected.get("run_id") != RUN_ID or expected.get("coverage") != {
        "methods": list(METHODS),
        "chains": list(CHAINS),
        "seeds": list(SEEDS),
        "tasks": list(TASKS),
        "method_chain_seed_bundles": 150,
        "task_metric_rows": 450,
        "per_chain_task_summaries": 90,
        "six_chain_task_summaries": 15,
    }:
        _fail("formal final reporting receipt", "coverage differs")


def _validate_source_summary(
    source_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    summary, raw = _strict_json_file(
        _source_path(source_root, "multiseed_summary"), "formal multiseed summary"
    )
    if _sha256_bytes(raw) != config["formal_evidence"]["multiseed_summary"]["sha256"]:
        _fail("formal multiseed summary", "digest differs from public evidence")
    expected_top = {
        "aggregation_contract", "bundle_count", "chains", "config_sha256", "methods",
        "per_chain_summaries", "per_chain_summary_count", "run_id", "schema_version",
        "seeds", "six_chain_summaries", "six_chain_summary_count",
        "static_task_manifest_sha256", "status", "task_row_count", "task_rows", "tasks",
    }
    if set(summary) != expected_top:
        _fail("formal multiseed summary", "top-level fields changed")
    expected_scalars = {
        "schema_version": "upgrade-bench-v2/contemporary-multiseed-summary/1",
        "status": "complete_all_150_bundles_all_450_task_rows",
        "run_id": RUN_ID,
        "bundle_count": 150,
        "task_row_count": 450,
        "per_chain_summary_count": 90,
        "six_chain_summary_count": 15,
        "config_sha256": EXPECTED_EVIDENCE_DIGESTS["multiseed_config"],
        "static_task_manifest_sha256": EXPECTED_EVIDENCE_DIGESTS["static_task_manifest"],
        "aggregation_contract": AGGREGATION_CONTRACT,
    }
    for field, expected in expected_scalars.items():
        if summary.get(field) != expected:
            _fail("formal multiseed summary", f"{field} differs")
    _exact_sequence(summary.get("chains"), CHAINS, "formal summary chains")
    _exact_sequence(summary.get("tasks"), TASKS, "formal summary tasks")
    _exact_sequence(summary.get("seeds"), SEEDS, "formal summary seeds")
    source_methods = summary.get("methods")
    public_methods = {row["method"]: row for row in config["methods"]}
    expected_methods = [
        {
            "method": method,
            "display_name": public_methods[method]["display_name"],
            "family": public_methods[method]["family"],
            "configuration_id": public_methods[method]["configuration_id"],
        }
        for method in METHODS
    ]
    if source_methods != expected_methods:
        _fail("formal multiseed summary", "method metadata differs")
    if not isinstance(summary.get("task_rows"), list) or len(summary["task_rows"]) != 450:
        _fail("formal multiseed summary", "requires exactly 450 seed-task rows")
    return summary


def _validate_formal_envelope(
    source_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_fixed_source_hashes(source_root)
    _recompute_collection(source_root)
    _recompute_final_receipt(source_root, config)
    return _validate_source_summary(source_root, config)


def _source_per_chain_map(summary: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows = summary.get("per_chain_summaries")
    if not isinstance(rows, list) or len(rows) != 90:
        _fail("formal per-chain summaries", "requires exactly 90 rows")
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    expected_fields = {
        "method", "display_name", "family", "chain", "task", "seed_count",
        "headline_metric", "headline_mean", "headline_population_sd", "value_metric",
        "value_capture_mean", "value_capture_population_sd", "per_seed",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            _fail("formal per-chain summaries", f"row {index} fields differ")
        key = (str(row.get("method")), str(row.get("chain")), str(row.get("task")))
        if key in result or key[0] not in METHODS or key[1] not in CHAINS or key[2] not in TASKS:
            _fail("formal per-chain summaries", f"invalid or duplicate key {key!r}")
        per_seed = row.get("per_seed")
        if not isinstance(per_seed, list) or len(per_seed) != 5:
            _fail("formal per-chain summaries", f"{key!r} lacks five seed rows")
        if any(not isinstance(item, Mapping) or set(item) != {"seed", "headline_value", "value_capture"} for item in per_seed):
            _fail("formal per-chain summaries", f"{key!r} seed-row fields differ")
        _exact_sequence([item["seed"] for item in per_seed], SEEDS, f"{key!r} seed coverage")
        headline_values = [_finite_unit(item["headline_value"], f"{key!r} seed headline") for item in per_seed]
        value_values = [_finite_unit(item["value_capture"], f"{key!r} seed value") for item in per_seed]
        _same_float(row.get("headline_mean"), _mean(headline_values), f"{key!r} headline mean")
        _same_float(row.get("headline_population_sd"), _population_sd(headline_values), f"{key!r} headline SD")
        _same_float(row.get("value_capture_mean"), _mean(value_values), f"{key!r} value mean")
        _same_float(row.get("value_capture_population_sd"), _population_sd(value_values), f"{key!r} value SD")
        if row.get("seed_count") != 5:
            _fail("formal per-chain summaries", f"{key!r} seed count differs")
        result[key] = row
    return result


def _extract_records(
    summary: Mapping[str, Any], method_metadata: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    source = _source_per_chain_map(summary)
    records: list[dict[str, Any]] = []
    for method in METHODS:
        for chain in CHAINS:
            for task in TASKS:
                row = source[(method, chain, task)]
                metadata = method_metadata[method]
                if row.get("display_name") != metadata["display_name"] or row.get("family") != metadata["family"]:
                    _fail("formal per-chain summaries", f"metadata differs for {method}/{chain}/{task}")
                if row.get("headline_metric") != METRIC_CONTRACT[task]["headline_metric"] or row.get("value_metric") != METRIC_CONTRACT[task]["value_metric"]:
                    _fail("formal per-chain summaries", f"metric contract differs for {method}/{chain}/{task}")
                records.append(
                    {
                        "method": method,
                        "display_name": metadata["display_name"],
                        "family": metadata["family"],
                        "chain": chain,
                        "task": task,
                        "seed_count": 5,
                        "headline_metric": row["headline_metric"],
                        "headline_mean": _finite_unit(row["headline_mean"], "headline mean"),
                        "headline_population_sd": _finite_unit(row["headline_population_sd"], "headline SD"),
                        "value_metric": row["value_metric"],
                        "value_capture_mean": _finite_unit(row["value_capture_mean"], "value mean"),
                        "value_capture_population_sd": _finite_unit(row["value_capture_population_sd"], "value SD"),
                    }
                )
    return records


def _source_aggregate_map(summary: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = summary.get("six_chain_summaries")
    if not isinstance(rows, list) or len(rows) != 15:
        _fail("formal six-chain summaries", "requires exactly 15 rows")
    expected_fields = {
        "method", "display_name", "family", "task", "chain_count", "seed_count",
        "headline_metric", "headline_unweighted_six_chain_mean",
        "headline_population_sd_across_seed_macros", "value_metric",
        "value_capture_unweighted_six_chain_mean",
        "value_capture_population_sd_across_seed_macros", "per_seed_six_chain_means",
    }
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            _fail("formal six-chain summaries", f"row {index} fields differ")
        key = (str(row.get("method")), str(row.get("task")))
        if key in result or key[0] not in METHODS or key[1] not in TASKS:
            _fail("formal six-chain summaries", f"invalid or duplicate key {key!r}")
        per_seed = row.get("per_seed_six_chain_means")
        expected_seed_fields = {
            "seed", "headline_unweighted_six_chain_mean",
            "value_capture_unweighted_six_chain_mean",
        }
        if not isinstance(per_seed, list) or len(per_seed) != 5 or any(
            not isinstance(item, Mapping) or set(item) != expected_seed_fields for item in per_seed
        ):
            _fail("formal six-chain summaries", f"{key!r} seed rows differ")
        _exact_sequence([item["seed"] for item in per_seed], SEEDS, f"{key!r} seed coverage")
        headlines = [_finite_unit(item["headline_unweighted_six_chain_mean"], f"{key!r} seed headline") for item in per_seed]
        values = [_finite_unit(item["value_capture_unweighted_six_chain_mean"], f"{key!r} seed value") for item in per_seed]
        _same_float(row.get("headline_unweighted_six_chain_mean"), _mean(headlines), f"{key!r} headline mean")
        _same_float(row.get("headline_population_sd_across_seed_macros"), _population_sd(headlines), f"{key!r} headline SD")
        _same_float(row.get("value_capture_unweighted_six_chain_mean"), _mean(values), f"{key!r} value mean")
        _same_float(row.get("value_capture_population_sd_across_seed_macros"), _population_sd(values), f"{key!r} value SD")
        if row.get("chain_count") != 6 or row.get("seed_count") != 5:
            _fail("formal six-chain summaries", f"{key!r} counts differ")
        result[key] = row
    return result


def _extract_aggregates(
    summary: Mapping[str, Any], method_metadata: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    source = _source_aggregate_map(summary)
    aggregates: list[dict[str, Any]] = []
    for method in METHODS:
        for task in TASKS:
            row = source[(method, task)]
            metadata = method_metadata[method]
            if row.get("display_name") != metadata["display_name"] or row.get("family") != metadata["family"]:
                _fail("formal six-chain summaries", f"metadata differs for {method}/{task}")
            if row.get("headline_metric") != METRIC_CONTRACT[task]["headline_metric"] or row.get("value_metric") != METRIC_CONTRACT[task]["value_metric"]:
                _fail("formal six-chain summaries", f"metric contract differs for {method}/{task}")
            aggregates.append(
                {
                    "method": method,
                    "display_name": metadata["display_name"],
                    "family": metadata["family"],
                    "task": task,
                    "chain_count": 6,
                    "seed_count": 5,
                    "headline_metric": row["headline_metric"],
                    "headline_unweighted_six_chain_mean": _finite_unit(row["headline_unweighted_six_chain_mean"], "aggregate headline mean"),
                    "headline_population_sd_across_seed_macros": _finite_unit(row["headline_population_sd_across_seed_macros"], "aggregate headline SD"),
                    "value_metric": row["value_metric"],
                    "value_capture_unweighted_six_chain_mean": _finite_unit(row["value_capture_unweighted_six_chain_mean"], "aggregate value mean"),
                    "value_capture_population_sd_across_seed_macros": _finite_unit(row["value_capture_population_sd_across_seed_macros"], "aggregate value SD"),
                }
            )
    return aggregates


def _build_summary(
    config: Mapping[str, Any], config_sha256: str, tool_sha256: str,
    records: list[dict[str, Any]], aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "run_id": config["run_id"],
        "benchmark_version": config["benchmark_version"],
        "config_sha256": config_sha256,
        "promotion_tool_sha256": tool_sha256,
        "claim_scope": config["claim_scope"],
        "benchmark": config["benchmark"],
        "metric_contract": config["metric_contract"],
        "feature_and_scoring_contract": config["feature_and_scoring_contract"],
        "methods": config["methods"],
        "formal_evidence": config["formal_evidence"],
        "record_count": len(records),
        "records": records,
        "aggregate_count": len(aggregates),
        "aggregates": aggregates,
    }
    _privacy_audit(summary)
    return summary


def _csv_float(value: Any) -> str:
    return repr(_finite_unit(value, "CSV numeric value"))


def _render_csv(summary: Mapping[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = [
        "method", "display_name", "family", "chain", "task", "seed_count",
        "headline_metric", "headline_mean", "headline_population_sd", "value_metric",
        "value_capture_mean", "value_capture_population_sd",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in summary["records"]:
        rendered = dict(row)
        for field in (
            "headline_mean", "headline_population_sd", "value_capture_mean",
            "value_capture_population_sd",
        ):
            rendered[field] = _csv_float(row[field])
        writer.writerow(rendered)
    raw = output.getvalue().encode("utf-8")
    _privacy_audit_bytes(raw, "public CSV")
    return raw


def _render_tex(summary: Mapping[str, Any]) -> bytes:
    lines = [
        "% Generated by tools/summarize_v2_contemporary_references.py; do not edit.",
        f"% config_sha256 {summary['config_sha256']}",
        f"% promotion_tool_sha256 {summary['promotion_tool_sha256']}",
        f"\\newcommand{{\\VTwoContemporaryMethodCount}}{{{len(METHODS)}}}",
        f"\\newcommand{{\\VTwoContemporarySeedCount}}{{{len(SEEDS)}}}",
        f"\\newcommand{{\\VTwoContemporaryRecordCount}}{{{summary['record_count']}}}",
        f"\\newcommand{{\\VTwoContemporaryAggregateCount}}{{{summary['aggregate_count']}}}",
        "\\newcommand{\\VTwoContemporaryStatus}{fixed-configuration five-seed}",
    ]
    aggregate_map = {(row["method"], row["task"]): row for row in summary["aggregates"]}
    for method in METHODS:
        for task in TASKS:
            row = aggregate_map[(method, task)]
            prefix = f"VTwoContemporary{METHOD_TEX[method]}Track{TASK_TEX[task]}"
            lines.extend(
                [
                    "\\newcommand{\\%sHeadline}{%.4f}" % (prefix, float(row["headline_unweighted_six_chain_mean"])),
                    "\\newcommand{\\%sHeadlineSD}{%.4f}" % (prefix, float(row["headline_population_sd_across_seed_macros"])),
                    "\\newcommand{\\%sValue}{%.4f}" % (prefix, float(row["value_capture_unweighted_six_chain_mean"])),
                    "\\newcommand{\\%sValueSD}{%.4f}" % (prefix, float(row["value_capture_population_sd_across_seed_macros"])),
                ]
            )
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    _privacy_audit_bytes(raw, "public TeX")
    return raw


def parse_tex_macros(raw: bytes | str) -> dict[str, str]:
    """Parse the simple generated ``\\newcommand`` definitions."""

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            _fail("TeX macros", f"not UTF-8: {exc}")
    elif isinstance(raw, str):
        text = raw
    else:
        _fail("TeX macros", "expected bytes or text")
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("\\newcommand"):
            continue
        match = TEX_MACRO_RE.fullmatch(line)
        if match is None:
            _fail("TeX macros", f"invalid generated macro on line {line_number}")
        name, value = match.groups()
        if name in result:
            _fail("TeX macros", f"duplicate macro {name}")
        result[name] = value
    return result


def _validate_public_summary(
    summary: Mapping[str, Any], config: Mapping[str, Any], config_raw: bytes
) -> None:
    expected_top = {
        "schema_version", "status", "run_id", "benchmark_version", "config_sha256",
        "promotion_tool_sha256", "claim_scope", "benchmark", "metric_contract",
        "feature_and_scoring_contract", "methods", "formal_evidence", "record_count",
        "records", "aggregate_count", "aggregates",
    }
    if set(summary) != expected_top:
        _fail("public summary", "top-level fields changed")
    expected_scalars = {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "run_id": RUN_ID,
        "benchmark_version": config["benchmark_version"],
        "config_sha256": _sha256_bytes(config_raw),
        "promotion_tool_sha256": _sha256_file(Path(__file__)),
        "claim_scope": config["claim_scope"],
        "benchmark": config["benchmark"],
        "metric_contract": config["metric_contract"],
        "feature_and_scoring_contract": config["feature_and_scoring_contract"],
        "methods": config["methods"],
        "formal_evidence": config["formal_evidence"],
        "record_count": 90,
        "aggregate_count": 15,
    }
    for field, expected in expected_scalars.items():
        if summary.get(field) != expected:
            _fail("public summary", f"{field} differs from the fixed contract")

    records = summary.get("records")
    record_fields = {
        "method", "display_name", "family", "chain", "task", "seed_count",
        "headline_metric", "headline_mean", "headline_population_sd", "value_metric",
        "value_capture_mean", "value_capture_population_sd",
    }
    if not isinstance(records, list) or len(records) != 90:
        _fail("public summary", "requires exactly 90 records")
    method_metadata = {row["method"]: row for row in config["methods"]}
    expected_record_keys = [
        (method, chain, task) for method in METHODS for chain in CHAINS for task in TASKS
    ]
    observed_record_keys: list[tuple[str, str, str]] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != record_fields:
            _fail("public summary", f"record {index} fields differ")
        key = (str(row.get("method")), str(row.get("chain")), str(row.get("task")))
        observed_record_keys.append(key)
        if key[0] not in METHODS or key[1] not in CHAINS or key[2] not in TASKS:
            _fail("public summary", f"invalid record key {key!r}")
        metadata = method_metadata[key[0]]
        if row.get("display_name") != metadata["display_name"] or row.get("family") != metadata["family"]:
            _fail("public summary", f"metadata differs for {key!r}")
        if row.get("seed_count") != 5:
            _fail("public summary", f"seed count differs for {key!r}")
        if row.get("headline_metric") != METRIC_CONTRACT[key[2]]["headline_metric"] or row.get("value_metric") != METRIC_CONTRACT[key[2]]["value_metric"]:
            _fail("public summary", f"metric contract differs for {key!r}")
        for field in (
            "headline_mean", "headline_population_sd", "value_capture_mean",
            "value_capture_population_sd",
        ):
            _finite_unit(row.get(field), f"record {key!r} {field}")
    if observed_record_keys != expected_record_keys:
        _fail("public summary", "record order or 5 x 6 x 3 coverage differs")
    if _sha256_bytes(_canonical_json_bytes(records)) != EXPECTED_RECORD_MATRIX_SHA256:
        _fail("public summary", "record matrix differs from the fixed public digest")

    aggregates = summary.get("aggregates")
    aggregate_fields = {
        "method", "display_name", "family", "task", "chain_count", "seed_count",
        "headline_metric", "headline_unweighted_six_chain_mean",
        "headline_population_sd_across_seed_macros", "value_metric",
        "value_capture_unweighted_six_chain_mean",
        "value_capture_population_sd_across_seed_macros",
    }
    if not isinstance(aggregates, list) or len(aggregates) != 15:
        _fail("public summary", "requires exactly 15 aggregates")
    expected_aggregate_keys = [(method, task) for method in METHODS for task in TASKS]
    observed_aggregate_keys: list[tuple[str, str]] = []
    for index, row in enumerate(aggregates):
        if not isinstance(row, Mapping) or set(row) != aggregate_fields:
            _fail("public summary", f"aggregate {index} fields differ")
        key = (str(row.get("method")), str(row.get("task")))
        observed_aggregate_keys.append(key)
        if key[0] not in METHODS or key[1] not in TASKS:
            _fail("public summary", f"invalid aggregate key {key!r}")
        metadata = method_metadata[key[0]]
        if row.get("display_name") != metadata["display_name"] or row.get("family") != metadata["family"]:
            _fail("public summary", f"aggregate metadata differs for {key!r}")
        if row.get("chain_count") != 6 or row.get("seed_count") != 5:
            _fail("public summary", f"aggregate counts differ for {key!r}")
        if row.get("headline_metric") != METRIC_CONTRACT[key[1]]["headline_metric"] or row.get("value_metric") != METRIC_CONTRACT[key[1]]["value_metric"]:
            _fail("public summary", f"aggregate metric contract differs for {key!r}")
        for field in (
            "headline_unweighted_six_chain_mean",
            "headline_population_sd_across_seed_macros",
            "value_capture_unweighted_six_chain_mean",
            "value_capture_population_sd_across_seed_macros",
        ):
            _finite_unit(row.get(field), f"aggregate {key!r} {field}")
        selected = [record for record in records if record["method"] == key[0] and record["task"] == key[1]]
        _same_float(
            row["headline_unweighted_six_chain_mean"],
            _mean([float(record["headline_mean"]) for record in selected]),
            f"aggregate {key!r} headline mean",
        )
        _same_float(
            row["value_capture_unweighted_six_chain_mean"],
            _mean([float(record["value_capture_mean"]) for record in selected]),
            f"aggregate {key!r} value mean",
        )
    if observed_aggregate_keys != expected_aggregate_keys:
        _fail("public summary", "aggregate order or 5 x 3 coverage differs")
    if _sha256_bytes(_canonical_json_bytes(aggregates)) != EXPECTED_AGGREGATE_MATRIX_SHA256:
        _fail("public summary", "aggregate matrix differs from the fixed public digest")
    _privacy_audit(summary)


def _build_from_formal(
    source_root: Path, config: Mapping[str, Any], config_raw: bytes
) -> dict[str, Any]:
    source_summary = _validate_formal_envelope(source_root, config)
    metadata = {row["method"]: row for row in config["methods"]}
    records = _extract_records(source_summary, metadata)
    aggregates = _extract_aggregates(source_summary, metadata)
    summary = _build_summary(
        config, _sha256_bytes(config_raw), _sha256_file(Path(__file__)), records, aggregates
    )
    _validate_public_summary(summary, config, config_raw)
    return summary


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        _fail("output write", f"cannot write {path}: {exc}")


def promote(
    *, source_root: Path = DEFAULT_SOURCE_ROOT, config_path: Path = DEFAULT_CONFIG,
    json_path: Path = DEFAULT_JSON_OUT, csv_path: Path = DEFAULT_CSV_OUT,
    tex_path: Path = DEFAULT_TEX_OUT,
) -> dict[str, Any]:
    """Recompute private provenance and publish the sanitized triplet."""

    config, config_raw = _load_config(config_path)
    summary = _build_from_formal(Path(source_root), config, config_raw)
    json_raw = _canonical_json_bytes(summary)
    csv_raw = _render_csv(summary)
    tex_raw = _render_tex(summary)
    _privacy_audit_bytes(json_raw, "public JSON")
    _atomic_write(json_path, json_raw)
    _atomic_write(csv_path, csv_raw)
    _atomic_write(tex_path, tex_raw)
    return verify_outputs(
        json_path, csv_path, tex_path, config_path=config_path, profile="repository"
    )


def verify_outputs(
    json_path: Path = DEFAULT_JSON_OUT, csv_path: Path = DEFAULT_CSV_OUT,
    tex_path: Path = DEFAULT_TEX_OUT, *, config_path: Path = DEFAULT_CONFIG,
    profile: str = "repository", source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    """Verify the public triplet; ``full`` also exactly replays private provenance."""

    if profile not in {"repository", "full"}:
        _fail("verification profile", "expected 'repository' or 'full'")
    config, config_raw = _load_config(config_path)
    summary, json_raw = _strict_json_file(json_path, "public JSON")
    _validate_public_summary(summary, config, config_raw)
    if json_raw != _canonical_json_bytes(summary):
        _fail("public JSON", "bytes are not canonical pretty JSON")
    _privacy_audit_bytes(json_raw, "public JSON")
    csv_raw = _read_regular(csv_path, "public CSV")
    tex_raw = _read_regular(tex_path, "public TeX")
    if csv_raw != _render_csv(summary):
        _fail("public CSV", "bytes do not exactly match the JSON records")
    if tex_raw != _render_tex(summary):
        _fail("public TeX", "bytes do not exactly match the JSON aggregates")
    macros = parse_tex_macros(tex_raw)
    expected_macro_count = 5 + len(METHODS) * len(TASKS) * 4
    if len(macros) != expected_macro_count:
        _fail("public TeX", f"expected {expected_macro_count} macros")
    if profile == "full":
        reproduced = _build_from_formal(Path(source_root), config, config_raw)
        if _canonical_json_bytes(reproduced) != json_raw:
            _fail("full verification", "formal promotion differs from public JSON")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--promote", action="store_true", help="promote formal evidence")
    action.add_argument("--verify", action="store_true", help="verify existing outputs")
    parser.add_argument(
        "--profile", choices=("repository", "full"), default="repository",
        help="full additionally replays the fixed private provenance chain",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.verify:
            summary = verify_outputs(
                args.json_out, args.csv_out, args.tex_out, config_path=args.config,
                profile=args.profile, source_root=args.source_root,
            )
            verb = "verified"
        else:
            summary = promote(
                source_root=args.source_root, config_path=args.config,
                json_path=args.json_out, csv_path=args.csv_out, tex_path=args.tex_out,
            )
            verb = "promoted and verified"
    except ContemporaryReferenceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"PASS: {verb} {summary['record_count']} chain means and "
        f"{summary['aggregate_count']} six-chain aggregates over five seeds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
