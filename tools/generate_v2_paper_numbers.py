#!/usr/bin/env python3
"""Generate the paper's numerical interface from claimable v2 artifacts.

This module is intentionally stricter than a report formatter.  It validates
the registry audit/evidence pair, candidate and protocol hashes, the raw-label
audit, rolling CPU schema 2, robustness schema 2, B1 candidate coverage, the
formal GPU summary, verified value/headroom diagnostics, the public-only
verified matched-LOCO summary, and the public-only verified external-pretrained
ULTRA summary, plus the frozen-config GBDT JSON/CSV pair, before it will write
or verify canonical paper outputs.  Pure collection may expose explicit
``PENDING`` GPU/LOCO/ULTRA/GBDT status for review tooling; canonical writes
never accept a pending state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import build_gpu_step3_postfreeze_attestation as gpu_postfreeze


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactPaths:
    """All machine-readable inputs used by the paper-number gate."""

    root: Path
    summary: Path
    rolling: Path
    raw_audit: Path
    robustness: Path
    registry_audit: Path
    registry_evidence: Path
    b1_coverage: Path
    gpu_summary: Path
    gpu_postfreeze_attestation: Path
    value_diagnostics: Path
    value_diagnostics_csv: Path
    value_diagnostics_generator: Path
    loco_summary: Path
    loco_summary_csv: Path
    loco_summary_generator: Path
    loco_config: Path
    ultra_summary: Path
    ultra_summary_csv: Path
    ultra_summary_generator: Path
    ultra_config: Path
    ultra_formal_controller: Path
    gbdt_summary: Path
    gbdt_summary_csv: Path
    gbdt_summary_generator: Path
    gbdt_config: Path
    product_space_summary: Path
    product_space_summary_csv: Path
    product_space_scores: Path
    product_space_generator: Path
    product_space_config: Path
    score_robustness_r5: Path
    score_robustness_r5_csv: Path
    score_robustness_r5_generator: Path
    score_robustness_r5_config: Path
    eligibility_threshold_geometry: Path
    eligibility_threshold_geometry_csv: Path
    eligibility_threshold_geometry_generator: Path
    eligibility_threshold_geometry_config: Path
    invalidation: Path

    @classmethod
    def under(cls, root: Path) -> "ArtifactPaths":
        root = Path(root).resolve()
        return cls(
            root=root,
            summary=root / "data" / "processed_v2" / "dataset_summary.json",
            rolling=root / "results_v2" / "metrics" / "rolling_cpu_baselines.json",
            raw_audit=root / "results_v2" / "metrics" / "raw_label_audit.json",
            robustness=root / "results_v2" / "metrics" / "v2_robustness.json",
            registry_audit=root / "docs" / "registry_audit.json",
            registry_evidence=root / "chains" / "evidence" / "registry_evidence.json",
            b1_coverage=root / "results_v2" / "metrics" / "b1_candidate_coverage.json",
            gpu_summary=root / "results_v2" / "metrics" / "v2_gpu_rolling_summary.json",
            gpu_postfreeze_attestation=root
            / "chains"
            / "evidence"
            / "gpu_step3_postfreeze_semantic_attestation.json",
            value_diagnostics=root
            / "results_v2"
            / "metrics"
            / "v2_value_diagnostics.json",
            value_diagnostics_csv=root
            / "results_v2"
            / "metrics"
            / "v2_value_diagnostics.csv",
            value_diagnostics_generator=root / "tools" / "v2_value_diagnostics.py",
            loco_summary=root
            / "results_v2"
            / "metrics"
            / "v2_loco_transfer_summary.json",
            loco_summary_csv=root
            / "results_v2"
            / "metrics"
            / "v2_loco_transfer_summary.csv",
            loco_summary_generator=root / "tools" / "summarize_v2_loco_results.py",
            loco_config=root / "configs" / "v2_loco_formal.json",
            ultra_summary=root
            / "results_v2"
            / "metrics"
            / "v2_ultra_zero_shot_summary.json",
            ultra_summary_csv=root
            / "results_v2"
            / "metrics"
            / "v2_ultra_zero_shot_summary.csv",
            ultra_summary_generator=root / "tools" / "summarize_v2_ultra_results.py",
            ultra_config=root / "configs" / "v2_ultra_formal.json",
            ultra_formal_controller=root / "tools" / "v2_ultra_formal.py",
            gbdt_summary=root
            / "results_v2"
            / "metrics"
            / "v2_gbdt_baselines.json",
            gbdt_summary_csv=root
            / "results_v2"
            / "metrics"
            / "v2_gbdt_baselines.csv",
            gbdt_summary_generator=root / "tools" / "v2_gbdt_baselines.py",
            gbdt_config=root / "configs" / "v2_gbdt_baselines.json",
            product_space_summary=root
            / "results_v2"
            / "metrics"
            / "v2_product_space_density.json",
            product_space_summary_csv=root
            / "results_v2"
            / "metrics"
            / "v2_product_space_density.csv",
            product_space_scores=root
            / "results_v2"
            / "scores"
            / "v2_product_space_density_scores.csv",
            product_space_generator=root / "tools" / "v2_product_space_density.py",
            product_space_config=root / "configs" / "v2_product_space_density.json",
            score_robustness_r5=root
            / "results_v2"
            / "metrics"
            / "v2_score_robustness_r5.json",
            score_robustness_r5_csv=root
            / "results_v2"
            / "metrics"
            / "v2_score_robustness_r5.csv",
            score_robustness_r5_generator=root / "tools" / "v2_score_robustness_r5.py",
            score_robustness_r5_config=root / "configs" / "v2_score_robustness_r5.json",
            eligibility_threshold_geometry=root
            / "results_v2"
            / "metrics"
            / "v2_eligibility_threshold_geometry.json",
            eligibility_threshold_geometry_csv=root
            / "results_v2"
            / "metrics"
            / "v2_eligibility_threshold_geometry.csv",
            eligibility_threshold_geometry_generator=root
            / "tools"
            / "v2_eligibility_threshold_geometry.py",
            eligibility_threshold_geometry_config=root
            / "configs"
            / "v2_eligibility_threshold_geometry.json",
            invalidation=root / "results_v2" / "metrics" / "INVALIDATED.json",
        )


DEFAULT_PATHS = ArtifactPaths.under(ROOT)
SUMMARY = DEFAULT_PATHS.summary
ROLLING = DEFAULT_PATHS.rolling
RAW_AUDIT = DEFAULT_PATHS.raw_audit
ROBUSTNESS = DEFAULT_PATHS.robustness
REGISTRY_AUDIT = DEFAULT_PATHS.registry_audit
REGISTRY_EVIDENCE = DEFAULT_PATHS.registry_evidence
B1_COVERAGE = DEFAULT_PATHS.b1_coverage
GPU_SUMMARY = DEFAULT_PATHS.gpu_summary
VALUE_DIAGNOSTICS = DEFAULT_PATHS.value_diagnostics
VALUE_DIAGNOSTICS_CSV = DEFAULT_PATHS.value_diagnostics_csv
VALUE_DIAGNOSTICS_GENERATOR = DEFAULT_PATHS.value_diagnostics_generator
LOCO_SUMMARY = DEFAULT_PATHS.loco_summary
LOCO_SUMMARY_CSV = DEFAULT_PATHS.loco_summary_csv
LOCO_SUMMARY_GENERATOR = DEFAULT_PATHS.loco_summary_generator
LOCO_CONFIG = DEFAULT_PATHS.loco_config
ULTRA_SUMMARY = DEFAULT_PATHS.ultra_summary
ULTRA_SUMMARY_CSV = DEFAULT_PATHS.ultra_summary_csv
ULTRA_SUMMARY_GENERATOR = DEFAULT_PATHS.ultra_summary_generator
ULTRA_CONFIG = DEFAULT_PATHS.ultra_config
ULTRA_FORMAL_CONTROLLER = DEFAULT_PATHS.ultra_formal_controller
GBDT_SUMMARY = DEFAULT_PATHS.gbdt_summary
GBDT_SUMMARY_CSV = DEFAULT_PATHS.gbdt_summary_csv
GBDT_SUMMARY_GENERATOR = DEFAULT_PATHS.gbdt_summary_generator
GBDT_CONFIG = DEFAULT_PATHS.gbdt_config
PRODUCT_SPACE_SUMMARY = DEFAULT_PATHS.product_space_summary
PRODUCT_SPACE_SUMMARY_CSV = DEFAULT_PATHS.product_space_summary_csv
PRODUCT_SPACE_SCORES = DEFAULT_PATHS.product_space_scores
PRODUCT_SPACE_GENERATOR = DEFAULT_PATHS.product_space_generator
PRODUCT_SPACE_CONFIG = DEFAULT_PATHS.product_space_config
SCORE_ROBUSTNESS_R5 = DEFAULT_PATHS.score_robustness_r5
SCORE_ROBUSTNESS_R5_CSV = DEFAULT_PATHS.score_robustness_r5_csv
SCORE_ROBUSTNESS_R5_GENERATOR = DEFAULT_PATHS.score_robustness_r5_generator
SCORE_ROBUSTNESS_R5_CONFIG = DEFAULT_PATHS.score_robustness_r5_config
ELIGIBILITY_THRESHOLD_GEOMETRY = DEFAULT_PATHS.eligibility_threshold_geometry
ELIGIBILITY_THRESHOLD_GEOMETRY_CSV = DEFAULT_PATHS.eligibility_threshold_geometry_csv
ELIGIBILITY_THRESHOLD_GEOMETRY_GENERATOR = (
    DEFAULT_PATHS.eligibility_threshold_geometry_generator
)
ELIGIBILITY_THRESHOLD_GEOMETRY_CONFIG = (
    DEFAULT_PATHS.eligibility_threshold_geometry_config
)
INVALIDATION = DEFAULT_PATHS.invalidation
DEFAULT_TEX = ROOT / "paper" / "generated" / "v2_numbers.tex"
DEFAULT_JSON = ROOT / "results_v2" / "paper_numbers.json"

PAPER_NUMBERS_SCHEMA = "upgrade-bench-v2-paper-numbers-8"
BENCHMARK_VERSION = "2.1-dev"
ROLLING_SCHEMA = "upgrade-bench-v2-rolling-cpu-baselines-2"
RAW_AUDIT_SCHEMA = "raw-label-audit/v2"
ROBUSTNESS_SCHEMA = "upgrade-bench-v2/robustness/2"
REGISTRY_AUDIT_SCHEMA = "upgrade-bench/registry-audit/3"
REGISTRY_EVIDENCE_SCHEMA = "upgrade-bench/hs92-registry-evidence/3"
B1_COVERAGE_SCHEMA = "upgrade-bench-v2/b1-candidate-coverage/2"
GPU_SUMMARY_SCHEMA = "upgrade-bench-v2/gpu-main-summary/1"
VALUE_DIAGNOSTICS_SCHEMA = "upgrade-bench-v2-value-diagnostics-1"
LOCO_PUBLIC_SUMMARY_SCHEMA = "upgrade-bench-v2/loco-transfer-public-summary/1"
LOCO_PROTOCOL = "upgrade-bench-v2/loco-tier-matched/1"
LOCO_RUN_ID = "loco-tier-matched-oa-full-20260717-r2"
LOCO_CLAIM_SCOPE = "matched_tier_abstracted_descriptive_transfer_diagnostic"
LOCO_PAPER_SCOPE = (
    "descriptive matched tier-abstracted in_domain-minus-loco diagnostic only"
)
ULTRA_PUBLIC_SUMMARY_SCHEMA = "upgrade-bench-v2/ultra-zero-shot-public-summary/1"
ULTRA_PROTOCOL = "upgrade-bench-v2/ultra-4g-zero-shot/2"
ULTRA_RUN_ID = "ultra-4g-zero-shot-fixed-20260717-r4"
ULTRA_PUBLIC_STATUS = "complete_verified_sanitized"
GBDT_PUBLIC_SUMMARY_SCHEMA = "upgrade-bench-v2-gbdt-baselines/1"
GBDT_PUBLIC_STATUS = "complete_verified"
GBDT_MODEL_KEY = "historical_gbdt_same_features"
GBDT_CONFIG_SCHEMA = "upgrade-bench-v2-gbdt-config/1"
GBDT_CONFIG_STATUS = "frozen_before_first_gbdt_main_evaluation"
GBDT_CONFIG_PROTOCOL = "historical-group-select-then-frozen-main-v1"
GBDT_SHARED_SOURCE_ROLE = "tools/v2_rolling_cpu_baselines.py"
GBDT_CPU_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,+()/\-]{0,127}\Z")
PRODUCT_SPACE_SCHEMA = "upgrade-bench-v2-product-space-density/1"
PRODUCT_SPACE_CONFIG_SCHEMA = "upgrade-bench-v2-product-space-density-config/1"
PRODUCT_SPACE_CONFIG_PROTOCOL = (
    "early-full-economy-scorer-freeze-then-b1-evaluation-v1"
)
PRODUCT_SPACE_CONFIG_STATUS = "frozen_before_first_product_space_main_evaluation"
PRODUCT_SPACE_STATUS = "complete_verified"
PRODUCT_SPACE_CLAIM_SCOPE = (
    "reviewer-motivated post-hoc B1-only descriptive domain reference; "
    "not part of the original prespecified reference set"
)
SCORE_ROBUSTNESS_R5_SCHEMA = "upgrade-bench-v2/score-robustness-r5/1"
SCORE_ROBUSTNESS_R5_CONFIG_SCHEMA = (
    "upgrade-bench-v2/score-robustness-r5-config/1"
)
SCORE_ROBUSTNESS_R5_STATUS = "complete_verified"
ELIGIBILITY_THRESHOLD_GEOMETRY_SCHEMA = (
    "upgrade-bench-v2-eligibility-threshold-geometry/1"
)
ELIGIBILITY_THRESHOLD_GEOMETRY_CONFIG_SCHEMA = (
    "upgrade-bench-v2-eligibility-threshold-geometry-config/1"
)
ELIGIBILITY_THRESHOLD_GEOMETRY_STATUS = "complete_verified"
ELIGIBILITY_THRESHOLD_GEOMETRY_SCOPE = (
    "cohort geometry only; no model scoring, rerun, or performance claim"
)
GPU_PROTOCOL = "strict_rolling_fold2_to_main"

REGISTRY_CHECKS = (
    "code_level_evidence_complete",
    "full_source_regex_scan_reproduced",
    "full_610_record_ledger_complete",
    "manual_review_not_claimed",
    "selected_baci_descriptions_match",
    "official_unsd_hs1992_membership_attested",
    "active_include_decisions_exact",
    "canonical_stage_definitions_complete",
    "per_code_stage_fit_supported_excluded_or_out_of_stage",
    "high_risk_stage_semantic_regressions_absent",
    "excluded_codes_absent_from_registry_and_relations",
    "stage_and_hs6_relationship_references_valid",
    "private_paths_absent",
)

CHAINS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
LOCO_MODES = ("loco", "in_domain")
LOCO_SEEDS = (0, 1, 2, 3, 4)
LOCO_METRIC_SPECS = (
    ("A.lane_average_precision", "TrackA"),
    ("B1.entry_average_precision", "TrackBOne"),
    ("B2.conditional_recall_at_3", "TrackBTwo"),
)
ULTRA_TASK_SPECS = {
    "A": {
        "tex": "TrackA",
        "headline_metric": "lane_average_precision",
        "headline_tex": "LaneAP",
        "value_metric": "value_capture_at_500",
        "value_tex": "ValueCaptureFiveHundred",
    },
    "B1": {
        "tex": "TrackBOne",
        "headline_metric": "entry_average_precision",
        "headline_tex": "EntryAP",
        "value_metric": "entry_value_capture_at_50",
        "value_tex": "EntryValueCaptureFifty",
    },
    "B2": {
        "tex": "TrackBTwo",
        "headline_metric": "conditional_recall_at_3",
        "headline_tex": "ConditionalRecallThree",
        "value_metric": "conditional_value_capture_at_3",
        "value_tex": "ConditionalValueCaptureThree",
    },
}
GBDT_TRACK_SPECS = {
    "track_a_destination_extension": {
        "tex": "TrackA",
        "headline_metric": "average_precision",
        "headline_tex": "AP",
        "value_metric": "global_observed_late_value_capture_at_500",
        "value_tex": "ValueCaptureFiveHundred",
    },
    "track_b1_processed_export_stage_entry": {
        "tex": "TrackBOne",
        "headline_metric": "average_precision",
        "headline_tex": "AP",
        "value_metric": "global_observed_late_value_capture_at_50",
        "value_tex": "ValueCaptureFifty",
    },
    "track_b2_conditional_destination_ranking": {
        "tex": "TrackBTwo",
        "headline_metric": "per_positive_entry_macro_recall_at_3",
        "headline_tex": "RecallThree",
        "value_metric": "per_positive_entry_macro_value_capture_at_3",
        "value_tex": "ValueCaptureThree",
    },
}
CHAIN_TEX = {
    "sheep": "Sheep",
    "cotton": "Cotton",
    "aluminium": "Aluminium",
    "nickel": "Nickel",
    "cocoa": "Cocoa",
    "oilseed-soy": "OilseedSoy",
}
HEX = frozenset("0123456789abcdef")

TRACK_SPECS: dict[str, dict[str, Any]] = {
    "track_a_destination_extension": {
        "tex": "TrackA",
        "metric": "average_precision",
        "value_metric": "global_observed_late_value_capture_at_500",
        "models": (
            ("size", "Size", "AP", "ValueCaptureFiveHundred"),
            ("gravity", "Gravity", "AP", "ValueCaptureFiveHundred"),
            (
                "historical_logistic_size_gravity",
                "Logistic",
                "AP",
                "ValueCaptureFiveHundred",
            ),
        ),
        "budget": {
            "scope": "within_chain_complete_main_cohort",
            "unit": "destination_lane",
            "selection": "global_top_k_by_model_score",
            "requested_k": 500,
            "effective_k": "min(requested_k, chain_target_rows)",
            "value_denominator": "all_positive_observed_late_value_in_chain",
        },
    },
    "track_b1_processed_export_stage_entry": {
        "tex": "TrackBOne",
        "metric": "average_precision",
        "value_metric": "global_observed_late_value_capture_at_50",
        "models": (
            ("upstream_capacity", "Capacity", "AP", "ValueCaptureFifty"),
            (
                "historical_logistic_structural",
                "Logistic",
                "AP",
                "ValueCaptureFifty",
            ),
        ),
        "budget": {
            "scope": "within_chain_complete_main_cohort",
            "unit": "exporter_stage_entry",
            "selection": "global_top_k_by_model_score",
            "requested_k": 50,
            "effective_k": "min(requested_k, chain_target_rows)",
            "value_denominator": "all_positive_observed_late_value_in_chain",
        },
    },
    "track_b2_conditional_destination_ranking": {
        "tex": "TrackBTwo",
        "metric": "per_positive_entry_macro_recall_at_3",
        "value_metric": "per_positive_entry_macro_value_capture_at_3",
        "models": (
            (
                "processed_importer_demand",
                "Demand",
                "RecallThree",
                "ValueCaptureThree",
            ),
            ("gravity", "Gravity", "RecallThree", "ValueCaptureThree"),
            (
                "historical_logistic_demand_gravity",
                "Logistic",
                "RecallThree",
                "ValueCaptureThree",
            ),
        ),
        "budget": {
            "scope": "within_each_actual_positive_exporter_stage_entry",
            "unit": "destination_lane",
            "selection": "top_k_by_model_score_per_entry",
            "requested_k_per_entry": 3,
            "effective_k": "min(requested_k_per_entry, entry_candidate_lanes)",
            "value_denominator": "positive_observed_late_value_within_each_entry",
            "entry_aggregation": "unweighted_mean_over_positive_entries",
        },
    },
}

VALUE_TRACK_SPECS: dict[str, dict[str, Any]] = {
    "a": {
        "tex": "TrackA",
        "suffix": "ValueCaptureFiveHundred",
        "diagnostic": "global_budget",
        "model_field": "model_value_capture",
        "oracle_field": "oracle_value_capture",
        "gap_field": "oracle_gap_value_capture",
        "pooled_oracle_field": "oracle_value_capture",
        "task": "destination_extension",
        "budget_field": "track_a_global",
        "budget": 500,
        "accounting_field": "track_a_observed_late_value_kusd",
        "cpu_models": (
            "size",
            "gravity",
            "historical_logistic_size_gravity",
        ),
    },
    "b1": {
        "tex": "TrackBOne",
        "suffix": "ValueCaptureFifty",
        "diagnostic": "global_entry_budget",
        "model_field": "model_value_capture",
        "oracle_field": "oracle_value_capture",
        "gap_field": "oracle_gap_value_capture",
        "pooled_oracle_field": "oracle_value_capture",
        "task": "eligible_market_processed_export_stage_entry",
        "budget_field": "track_b1_global_entries",
        "budget": 50,
        "accounting_field": "track_b1_observed_late_value_kusd",
        "cpu_models": (
            "upstream_capacity",
            "historical_logistic_structural",
        ),
    },
    "b2": {
        "tex": "TrackBTwo",
        "suffix": "MacroValueCaptureThree",
        "diagnostic": "per_positive_entry",
        "model_field": "model_macro_value_capture",
        "oracle_field": "oracle_macro_value_capture",
        "gap_field": "oracle_gap_macro_value_capture",
        "pooled_oracle_field": "oracle_pooled_value_capture",
        "task": "conditional_destination_formation",
        "budget_field": "track_b2_per_positive_entry",
        "budget": 3,
        "accounting_field": "track_b2_nested_same_dollars_kusd",
        "cpu_models": (
            "processed_importer_demand",
            "gravity",
            "historical_logistic_demand_gravity",
        ),
    },
}

VALUE_FAMILY_TEX = {"kge": "KGE", "nbfnet": "NBFNet"}
VALUE_SEEDS = (0, 1, 2, 3, 4)


class PaperNumberValidationError(ValueError):
    """A source artifact is stale, incomplete, or scientifically unclaimable."""


def _fail(role: str, message: str) -> None:
    raise PaperNumberValidationError(f"{role}: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _fail(str(path), f"cannot read strict JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(str(path), "top-level JSON value must be an object")
    return payload


def _hex64(value: Any, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - HEX:
        _fail(role, "expected a lowercase SHA-256 digest")
    return value


def _finite(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(role, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(role, "expected a finite number")
    return result


def _integer(value: Any, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(role, f"expected an integer >= {minimum}")
    return value


def _close(observed: Any, expected: Any, role: str, *, atol: float = 1e-12) -> None:
    observed_value = _finite(observed, role)
    expected_value = _finite(expected, role)
    if not math.isclose(observed_value, expected_value, rel_tol=1e-12, abs_tol=atol):
        _fail(role, f"expected {expected_value!r}, found {observed_value!r}")


def _exact_keys(value: Any, expected: Sequence[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(role, "expected an object")
    if set(value) != set(expected):
        _fail(role, f"expected keys {sorted(expected)!r}, found {sorted(value)!r}")
    return value


def _repo_file(paths: ArtifactPaths, relative: Any, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(role, "missing repository-relative path")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        _fail(role, f"unsafe/non-portable repository path {relative!r}")
    resolved = (paths.root / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(paths.root)
    except ValueError:
        _fail(role, f"path escapes repository root: {relative!r}")
    if not resolved.is_file():
        _fail(role, f"missing source file {relative!r}")
    return resolved


def _verify_hash(
    paths: ArtifactPaths,
    relative: Any,
    expected: Any,
    role: str,
    cache: dict[Path, str],
) -> str:
    expected_digest = _hex64(expected, role)
    source = _repo_file(paths, relative, role)
    observed = cache.setdefault(source, _sha256(source))
    if observed != expected_digest:
        _fail(role, f"stale hash for {relative}: expected {expected_digest}, observed {observed}")
    return expected_digest


def _assert_claimable_sources(paths: ArtifactPaths = DEFAULT_PATHS) -> None:
    """Reject an active/broken hold; accept only a fully proved resolution.

    The resolution gate keeps the public notice as provenance, so marker
    existence alone is no longer sufficient after that gate succeeds.  The
    shared release policy independently checks the exact normative scope and
    every replacement byte.  Import locally to keep the paper collector free
    of a module-level release-policy dependency.
    """
    if not paths.invalidation.exists():
        return
    import public_release_policy

    canonical_notice = paths.root / public_release_policy.PUBLIC_V2_INVALIDATION_NOTICE
    if Path(paths.invalidation).absolute() != canonical_notice.absolute():
        raise RuntimeError(
            "refusing canonical paper numbers: invalidation marker is not the "
            "canonical repository notice"
        )
    unsafe = public_release_policy.source_path_reason(
        public_release_policy.PUBLIC_V2_INVALIDATION_NOTICE,
        paths.root,
        require_file=True,
    )
    if unsafe is not None:
        raise RuntimeError(
            "refusing canonical paper numbers: invalidation marker is unsafe "
            f"({unsafe})"
        )
    notice = _load(paths.invalidation)
    status = notice.get("status", "UNKNOWN")
    if status not in {"RESOLVED", "SUPERSEDED"}:
        raise RuntimeError(
            "refusing canonical paper numbers while the invalidation marker exists "
            f"({status}); resolve it only through the formal replacement-artifact gate"
        )
    blocker = public_release_policy.unresolved_v2_invalidation(paths.root)
    if blocker is not None:
        raise RuntimeError(
            "refusing canonical paper numbers: resolved invalidation proof is invalid "
            f"({blocker})"
        )


def _commas(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _decimal(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def _billions(kusd: float, digits: int = 2) -> str:
    return f"{float(kusd) / 1_000_000.0:.{digits}f}"


def _validate_registry(
    paths: ArtifactPaths,
    audit: Mapping[str, Any],
    evidence: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    # The audit owns the exact set of code--stage mappings whose semantics are
    # regression-gated.  Import the registry contract rather than duplicating
    # its current count as a paper literal.
    from audit_chain_registry import SEMANTIC_STAGE_ASSIGNMENTS

    role = "registry audit/evidence"
    if audit.get("schema_version") != REGISTRY_AUDIT_SCHEMA or audit.get("status") != "PASS":
        _fail(role, "registry audit is not schema-3 PASS")
    if evidence.get("schema_version") != REGISTRY_EVIDENCE_SCHEMA:
        _fail(role, "registry evidence is not schema 3")
    if audit.get("review_date") != evidence.get("review_date"):
        _fail(role, "review dates disagree")
    for field in ("decision_policy", "stage_policy", "source", "summary"):
        if audit.get(field) != evidence.get(field):
            _fail(role, f"audit/evidence {field} payloads disagree")

    summary = _exact_keys(
        audit.get("summary"),
        (
            "chain_count",
            "active_stages",
            "included_codes",
            "excluded_codes",
            "out_of_stage_codes",
            "reviewed_codes",
            "decision_records",
            "unique_reviewed_hs6",
            "observable_candidate_records",
            "legacy_only_records",
            "reassigned_included_codes",
            "human_reviewed_records",
            "historical_active_codes",
            "historical_active_retained",
            "historical_active_removed",
            "new_active_added",
        ),
        f"{role}/summary",
    )
    if summary["chain_count"] != len(CHAINS):
        _fail(role, "registry does not contain the six frozen chains")
    included = _integer(summary["included_codes"], f"{role}/included", minimum=1)
    excluded = _integer(summary["excluded_codes"], f"{role}/excluded")
    out_of_stage = _integer(summary["out_of_stage_codes"], f"{role}/out of stage")
    reviewed = _integer(summary["reviewed_codes"], f"{role}/reviewed", minimum=1)
    decision_records = _integer(
        summary["decision_records"], f"{role}/decision records", minimum=1
    )
    unique_reviewed = _integer(
        summary["unique_reviewed_hs6"], f"{role}/unique reviewed HS6", minimum=1
    )
    observable_records = _integer(
        summary["observable_candidate_records"],
        f"{role}/observable candidate records",
        minimum=1,
    )
    legacy_records = _integer(
        summary["legacy_only_records"], f"{role}/legacy-only records"
    )
    human_reviewed = _integer(
        summary["human_reviewed_records"], f"{role}/human-reviewed records"
    )
    historical_active = _integer(
        summary["historical_active_codes"], f"{role}/historical active"
    )
    historical_retained = _integer(
        summary["historical_active_retained"], f"{role}/historical retained"
    )
    historical_removed = _integer(
        summary["historical_active_removed"], f"{role}/historical removed"
    )
    new_active = _integer(summary["new_active_added"], f"{role}/new active")
    if included + excluded + out_of_stage != reviewed or reviewed != decision_records:
        _fail(
            role,
            "include + exclude + out_of_stage counts do not equal decision records",
        )
    if observable_records + legacy_records != decision_records:
        _fail(role, "observable + legacy-only records do not equal decision records")
    if unique_reviewed > decision_records or human_reviewed > decision_records:
        _fail(role, "unique or human-reviewed counts exceed decision records")
    if historical_retained + historical_removed != historical_active:
        _fail(role, "historical active retention counts do not reconcile")
    if historical_retained + new_active != included:
        _fail(role, "retained historical + new active counts do not equal includes")

    source = audit.get("source", {})
    if source.get("release_version") != "V202401b" or source.get("unsd_class_code") != "H0":
        _fail(role, "unexpected BACI/HS92 registry source")
    selected_metadata = source.get("selected_metadata_file")
    if selected_metadata != "chains/evidence/hs92_selected_product_codes.csv":
        _fail(role, "registry audit does not name the canonical selected-code metadata")
    _verify_hash(
        paths,
        selected_metadata,
        source.get("selected_metadata_sha256"),
        f"{role}/selected metadata",
        hashes,
    )
    _hex64(source.get("source_metadata_member_sha256"), f"{role}/BACI metadata member")
    selected_metadata_path = _repo_file(paths, selected_metadata, f"{role}/selected metadata")
    selected_descriptions: dict[str, str] = {}
    try:
        with selected_metadata_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != ["code", "description"]:
                _fail(role, "selected-code metadata has a non-canonical header")
            for row in reader:
                code = row.get("code")
                description = row.get("description")
                if (
                    not isinstance(code, str)
                    or len(code) != 6
                    or not code.isdigit()
                    or not isinstance(description, str)
                    or not description
                ):
                    _fail(role, "selected-code metadata contains a malformed row")
                if code in selected_descriptions:
                    _fail(role, f"selected-code metadata duplicates HS6 {code}")
                selected_descriptions[code] = description
    except (OSError, csv.Error) as exc:
        _fail(role, f"cannot read selected-code metadata: {exc}")
    if not selected_descriptions:
        _fail(role, "selected-code metadata is empty")

    checks = _exact_keys(audit.get("checks"), REGISTRY_CHECKS, f"{role}/checks")
    if any(value != "PASS" for value in checks.values()):
        _fail(role, "one or more registry purity checks are not PASS")

    audit_chains = _exact_keys(audit.get("chains"), CHAINS, f"{role}/audit chains")
    evidence_chains = _exact_keys(evidence.get("chains"), CHAINS, f"{role}/evidence chains")
    if set(SEMANTIC_STAGE_ASSIGNMENTS) != set(CHAINS):
        _fail(role, "semantic-stage gate registry does not cover the six frozen chains")
    semantic_gate_total = sum(
        len(assignments) for assignments in SEMANTIC_STAGE_ASSIGNMENTS.values()
    )

    active_total = removed_total = out_of_stage_total = 0
    reassigned_total = stage_total = target_total = 0
    observable_total = legacy_total = human_reviewed_total = 0
    reviewed_decision_keys: set[tuple[str, str]] = set()
    reviewed_unique_codes: set[str] = set()
    per_chain: dict[str, dict[str, int]] = {}
    for chain in CHAINS:
        a = audit_chains[chain]
        e = evidence_chains[chain]
        if not isinstance(a, Mapping) or not isinstance(e, Mapping):
            _fail(role, f"malformed chain payload for {chain}")
        expected_registry_file = f"chains/{chain}.json"
        if a.get("registry_file") != expected_registry_file:
            _fail(role, f"{chain} audit does not name the canonical registry file")
        _verify_hash(
            paths,
            expected_registry_file,
            a.get("registry_sha256"),
            f"{role}/{chain}/registry",
            hashes,
        )
        registry_payload = _load(
            _repo_file(paths, expected_registry_file, f"{role}/{chain}/registry")
        )
        if registry_payload.get("id") != chain:
            _fail(role, f"{chain} hashed registry has the wrong chain identifier")
        if a.get("display_description") != registry_payload.get("description"):
            _fail(role, f"{chain} description disagrees with the hashed registry")
        if a.get("relation_integrity") != "PASS" or a.get("stage_semantic_integrity") != "PASS":
            _fail(role, f"{chain} relation/stage semantic integrity is not PASS")
        if a.get("stage_definitions") != e.get("stage_definitions"):
            _fail(role, f"{chain} stage definitions disagree between audit and evidence")
        if a.get("decisions") != e.get("decisions"):
            _fail(role, f"{chain} code decisions disagree between audit and evidence")
        registry_stages = registry_payload.get("stages")
        if not isinstance(registry_stages, Mapping) or not registry_stages:
            _fail(role, f"{chain} hashed registry stages are malformed")
        active_code_stage: dict[str, str] = {}
        for stage, codes in registry_stages.items():
            if not isinstance(stage, str) or not stage or not isinstance(codes, list) or not codes:
                _fail(role, f"{chain} hashed registry stage {stage!r} is malformed")
            for code in codes:
                if not isinstance(code, str) or len(code) != 6 or not code.isdigit():
                    _fail(role, f"{chain} hashed registry contains malformed HS6 code {code!r}")
                if code in active_code_stage:
                    _fail(role, f"{chain} hashed registry assigns HS6 {code} more than once")
                active_code_stage[code] = stage

        stage_definitions = a.get("stage_definitions")
        if not isinstance(stage_definitions, Mapping):
            _fail(role, f"{chain} canonical stage definitions are malformed")
        active_stages = a.get("active_stages")
        if (
            not isinstance(active_stages, list)
            or len(active_stages) != len(set(active_stages))
            or set(active_stages) != set(stage_definitions)
        ):
            _fail(role, f"{chain} active-stage registry is incomplete")
        if set(active_stages) != set(registry_stages):
            _fail(role, f"{chain} active stages disagree with the hashed registry")
        targets = a.get("capacity_from_stages")
        if not isinstance(targets, Mapping) or not targets:
            _fail(role, f"{chain} target-stage registry is malformed")
        if targets != registry_payload.get("upstream_map"):
            _fail(role, f"{chain} target stages disagree with the hashed registry")
        if not set(targets).issubset(active_stages):
            _fail(role, f"{chain} target-stage registry contains an inactive stage")
        decisions = a.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            _fail(role, f"{chain} code-decision registry is malformed")
        decision_by_code: dict[str, Mapping[str, Any]] = {}
        included_decisions: dict[str, Mapping[str, Any]] = {}
        excluded_decisions: dict[str, Mapping[str, Any]] = {}
        out_of_stage_decisions: dict[str, Mapping[str, Any]] = {}
        recomputed_reassigned: list[dict[str, Any]] = []
        for row in decisions:
            if not isinstance(row, Mapping):
                _fail(role, f"{chain} contains a non-object code decision")
            code = row.get("code")
            if not isinstance(code, str) or len(code) != 6 or not code.isdigit():
                _fail(role, f"{chain} decision contains malformed HS6 code {code!r}")
            if code in decision_by_code:
                _fail(role, f"{chain} contains duplicate decision for HS6 {code}")
            if code not in selected_descriptions:
                _fail(role, f"{chain} decision for HS6 {code} is outside selected metadata")
            if row.get("description") != selected_descriptions[code]:
                _fail(role, f"{chain} decision description drifted for HS6 {code}")
            decision_by_code[code] = row
            reviewed_decision_keys.add((chain, code))
            reviewed_unique_codes.add(code)
            source_kind = row.get("candidate_source")
            if source_kind == "observable_regex":
                observable_total += 1
            elif source_kind == "legacy_only":
                legacy_total += 1
            else:
                _fail(role, f"{chain} decision for HS6 {code} has invalid candidate source")
            if row.get("human_review_status") != "not_performed":
                human_reviewed_total += 1
            decision = row.get("decision")
            stage_fit = row.get("stage_fit")
            if not isinstance(stage_fit, Mapping):
                _fail(role, f"{chain} decision for HS6 {code} has malformed stage_fit")
            if decision == "include":
                included_decisions[code] = row
                expected_stage = active_code_stage.get(code)
                if expected_stage is None or row.get("stage") != expected_stage:
                    _fail(
                        role,
                        f"{chain} include decision for HS6 {code} disagrees with the hashed registry",
                    )
                if stage_fit.get("status") != "supported":
                    _fail(role, f"{chain} included HS6 {code} is not supported")
                if "previous_stage" in row:
                    previous_stage = row.get("previous_stage")
                    if (
                        not isinstance(previous_stage, str)
                        or not previous_stage
                        or previous_stage == expected_stage
                    ):
                        _fail(role, f"{chain} reassignment for HS6 {code} is malformed")
                    recomputed_reassigned.append(
                        {
                            "code": code,
                            "previous_stage": previous_stage,
                            "active_stage": expected_stage,
                            "description": row.get("description"),
                        }
                    )
            elif decision == "exclude":
                excluded_decisions[code] = row
                if code in active_code_stage or row.get("stage") is not None:
                    _fail(role, f"{chain} excluded HS6 {code} remains active")
                if stage_fit.get("status") != "unsupported":
                    _fail(role, f"{chain} excluded HS6 {code} is not unsupported")
            elif decision == "out_of_stage":
                out_of_stage_decisions[code] = row
                if code in active_code_stage or row.get("stage") is not None:
                    _fail(role, f"{chain} out-of-stage HS6 {code} remains active")
                if stage_fit.get("status") != "out_of_stage":
                    _fail(role, f"{chain} out-of-stage HS6 {code} has the wrong fit status")
            else:
                _fail(
                    role,
                    f"{chain} decision for HS6 {code} is not include, exclude, or out_of_stage",
                )

        if set(included_decisions) != set(active_code_stage):
            _fail(
                role,
                f"{chain} include decisions do not exactly cover the hashed registry codes",
            )
        active = len(active_code_stage)
        removed = len(excluded_decisions)
        outside = len(out_of_stage_decisions)
        before = len(decision_by_code)
        reassigned = len(recomputed_reassigned)
        if _integer(a.get("active_codes"), f"{role}/{chain}/active") != active:
            _fail(role, f"{chain} self-reported active-code count disagrees with the hashed registry")
        if _integer(a.get("removed_codes"), f"{role}/{chain}/removed") != removed:
            _fail(role, f"{chain} self-reported excluded-code count disagrees with decisions")
        if _integer(a.get("out_of_stage_codes"), f"{role}/{chain}/out of stage") != outside:
            _fail(role, f"{chain} self-reported out-of-stage count disagrees with decisions")
        if _integer(a.get("before_review_codes"), f"{role}/{chain}/before") != before:
            _fail(role, f"{chain} self-reported reviewed-code count disagrees with decisions")
        if (
            e.get("included_count") != active
            or e.get("excluded_count") != removed
            or e.get("out_of_stage_count") != outside
        ):
            _fail(role, f"{chain} evidence decision counts do not reconcile")
        chain_observable = sum(
            row.get("candidate_source") == "observable_regex" for row in decisions
        )
        chain_legacy = sum(row.get("candidate_source") == "legacy_only" for row in decisions)
        if (
            e.get("observable_candidate_count") != chain_observable
            or e.get("legacy_only_count") != chain_legacy
        ):
            _fail(role, f"{chain} candidate-source counts do not reconcile")
        if a.get("reassigned_codes") != recomputed_reassigned:
            _fail(role, f"{chain} reassigned-code record disagrees with decisions")

        for code, expected_stage in SEMANTIC_STAGE_ASSIGNMENTS[chain].items():
            decision = decision_by_code.get(code)
            stage_fit = decision.get("stage_fit") if isinstance(decision, Mapping) else None
            if (
                not isinstance(decision, Mapping)
                or decision.get("stage") != expected_stage
                or not isinstance(stage_fit, Mapping)
                or stage_fit.get("status") != "supported"
            ):
                _fail(
                    role,
                    f"{chain} semantic-stage gate regressed for {code}",
                )
        active_total += active
        removed_total += removed
        out_of_stage_total += outside
        reassigned_total += reassigned
        stage_total += len(registry_stages)
        target_total += len(targets)
        per_chain[chain] = {
            "active": active,
            "removed": removed,
            "out_of_stage": outside,
            "reviewed": before,
            "active_stages": len(registry_stages),
            "target_stages": len(targets),
            "reassigned": reassigned,
        }

    if reviewed_unique_codes != set(selected_descriptions):
        missing = sorted(set(selected_descriptions) - reviewed_unique_codes)
        extra = sorted(reviewed_unique_codes - set(selected_descriptions))
        _fail(
            role,
            "the union of decision HS6 codes does not exactly cover selected metadata "
            f"(missing={missing!r}, extra={extra!r})",
        )

    recomputed_summary = {
        "chain_count": len(CHAINS),
        "active_stages": stage_total,
        "included_codes": active_total,
        "excluded_codes": removed_total,
        "out_of_stage_codes": out_of_stage_total,
        "reviewed_codes": len(reviewed_decision_keys),
        "decision_records": len(reviewed_decision_keys),
        "unique_reviewed_hs6": len(reviewed_unique_codes),
        "observable_candidate_records": observable_total,
        "legacy_only_records": legacy_total,
        "reassigned_included_codes": reassigned_total,
        "human_reviewed_records": human_reviewed_total,
        "historical_active_codes": historical_active,
        "historical_active_retained": historical_retained,
        "historical_active_removed": historical_removed,
        "new_active_added": new_active,
    }
    if dict(summary) != recomputed_summary:
        _fail(
            role,
            f"self-reported summary {dict(summary)!r} disagrees with recomputed registry {recomputed_summary!r}",
        )
    if (included, excluded, out_of_stage, reviewed) != (
        recomputed_summary["included_codes"],
        recomputed_summary["excluded_codes"],
        recomputed_summary["out_of_stage_codes"],
        recomputed_summary["reviewed_codes"],
    ):
        _fail(role, "validated summary counts changed during registry recomputation")
    return {
        "summary": recomputed_summary,
        "purity_checks_passed": sum(value == "PASS" for value in checks.values()),
        "purity_checks_total": len(REGISTRY_CHECKS),
        "per_chain": per_chain,
        "target_stages": target_total,
        "semantic_stage_gates": semantic_gate_total,
    }


def _validate_summary(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    role = "dataset summary"
    expected_header = {
        "benchmark_version": BENCHMARK_VERSION,
        "aggregation": "calendar_mean",
        "official_temporal_protocol": "historical-fold selection -> frozen main target evaluation",
        "diagnostic_split_unit": "exporter_stage",
        "temporal_role": "target",
        "source_suffix": "",
    }
    for field, expected in expected_header.items():
        if summary.get(field) != expected:
            _fail(role, f"unexpected {field}: {summary.get(field)!r}")
    rows = summary.get("chains")
    if not isinstance(rows, list) or [row.get("chain") for row in rows] != list(CHAINS):
        _fail(role, "chain rows are not in the frozen six-chain order")
    totals = summary.get("totals")
    if not isinstance(totals, Mapping) or totals.get("chain") != "TOTAL":
        _fail(role, "missing TOTAL row")
    by_chain = {row["chain"]: row for row in rows}
    additive = (
        "track_a_candidates",
        "track_a_positive_lanes",
        "track_a_observed_late_value_kusd",
        "track_b_candidate_lanes",
        "track_b_positive_lanes",
        "track_b_unique_entries",
        "track_b_positive_entries",
        "track_b_observed_late_value_kusd",
        "track_b2_conditional_lanes",
        "track_b2_positive_lanes",
        "track_b2_observed_late_value_kusd",
    )
    for field in additive:
        _close(totals.get(field), sum(_finite(row.get(field), f"{role}/{field}") for row in rows), f"{role}/totals/{field}", atol=1e-6)
    rate_specs = (
        ("track_a_base_rate", "track_a_positive_lanes", "track_a_candidates"),
        ("track_b_lane_base_rate", "track_b_positive_lanes", "track_b_candidate_lanes"),
        ("track_b_entry_base_rate", "track_b_positive_entries", "track_b_unique_entries"),
        ("track_b2_base_rate", "track_b2_positive_lanes", "track_b2_conditional_lanes"),
    )
    for row in [*rows, totals]:
        for rate, positives, denominator in rate_specs:
            _close(row.get(rate), row[positives] / row[denominator], f"{role}/{row['chain']}/{rate}")
    return by_chain


def _candidate_hash_map(
    paths: ArtifactPaths,
    rolling: Mapping[str, Any],
    summary_by_chain: Mapping[str, Mapping[str, Any]],
    hashes: dict[Path, str],
) -> dict[str, str]:
    role = "rolling CPU"
    if rolling.get("schema_version") != ROLLING_SCHEMA or rolling.get("benchmark_version") != BENCHMARK_VERSION:
        _fail(role, "expected current rolling CPU schema 2 / benchmark 2.1-dev")
    protocol = rolling.get("protocol", {})
    expected_protocol = {
        "bootstrap_draws": 200,
        "selection_window": "1998-2002 -> 2008-2012",
        "frozen_target_window": "2008-2012 -> 2018-2022",
        "selection_source": "fold2 only",
        "target_evaluation": "complete main cohort",
        "target_labels_used_for_model_selection": False,
        "target_labels_used_for_imputation_scaling_or_calibration": False,
        "transductive_split_used": False,
        "main_target_models_compared_without_post_hoc_champion_selection": True,
        "pairwise_reporting": "all protocol-fixed unordered model pairs; descriptive per-chain deltas without chain-level confidence intervals",
        "random_seed": 20260712,
        "selection_objectives": {
            "track_a": "historical_exporter_group_cv_average_precision",
            "track_b1": "historical_exporter_group_cv_average_precision",
            "track_b2": "historical_exporter_stage_entry_group_cv_per_positive_entry_macro_recall_at_3",
        },
        "main_metric_cluster_units": {
            "track_a_average_precision": "exporter",
            "track_b1_average_precision": "exporter",
            "track_b2_recall_and_value": "exporter_stage_entry",
        },
        "realized_value_reporting_points": {
            "track_a_global_top_k": 500,
            "track_b1_global_top_k": 50,
            "track_b2_per_entry_top_k": 3,
        },
        "frozen_pairwise_model_order": {
            key: [model[0] for model in spec["models"]] for key, spec in TRACK_SPECS.items()
        },
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            _fail(role, f"protocol field {field!r} is stale or missing")

    chains = _exact_keys(rolling.get("chains"), CHAINS, f"{role}/chains")
    candidate_hashes: dict[str, str] = {}
    input_roles = ("history_track_a", "history_track_b", "target_track_a", "target_track_b")
    for chain in CHAINS:
        audit = chains[chain].get("protocol_audit", {})
        if audit.get("target_loaded_after_all_models_frozen") is not True:
            _fail(role, f"{chain} does not attest the global model-freeze boundary")
        if audit.get("target_labels_used_for_training_selection_imputation_or_calibration") is not False:
            _fail(role, f"{chain} target-label leakage flag is not false")
        if audit.get("transductive_split_used") is not False:
            _fail(role, f"{chain} transductive flag is not false")
        for input_role in input_roles:
            record = audit.get(input_role, {})
            digest = _verify_hash(
                paths,
                record.get("path"),
                record.get("sha256"),
                f"{role}/{chain}/{input_role}",
                hashes,
            )
            candidate_hashes[record["path"]] = digest
        target_a = audit["target_track_a"]
        target_b = audit["target_track_b"]
        row = summary_by_chain[chain]
        if target_a.get("rows") != row["track_a_candidates"] or target_a.get("positive_lanes") != row["track_a_positive_lanes"]:
            _fail(role, f"{chain} Track A target counts disagree with dataset summary")
        if target_b.get("rows") != row["track_b_candidate_lanes"] or target_b.get("positive_lanes") != row["track_b_positive_lanes"]:
            _fail(role, f"{chain} Track B target counts disagree with dataset summary")
    if len(candidate_hashes) != 24:
        _fail(role, f"expected 24 unique candidate hashes, found {len(candidate_hashes)}")
    return candidate_hashes


def _validate_pair_report(
    report: Mapping[str, Any],
    left: Mapping[str, float],
    right: Mapping[str, float],
    role: str,
) -> None:
    if report.get("orientation") != "left_minus_right" or report.get("chain_level_ci95") is not None:
        _fail(role, "paired-delta orientation/inference contract changed")
    per_chain = _exact_keys(report.get("per_chain"), CHAINS, f"{role}/per_chain")
    expected = {chain: left[chain] - right[chain] for chain in CHAINS}
    for chain in CHAINS:
        _close(per_chain[chain], expected[chain], f"{role}/{chain}")
    values = [expected[chain] for chain in CHAINS]
    _close(report.get("descriptive_mean_delta"), statistics.fmean(values), f"{role}/mean")
    _close(report.get("descriptive_median_delta"), statistics.median(values), f"{role}/median")
    signs = report.get("sign_counts", {})
    expected_signs = {
        "left_better": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "right_better": sum(value < -1e-12 for value in values),
    }
    if signs != expected_signs:
        _fail(role, "paired-delta sign counts do not reconcile")


def _validate_cpu_macros(rolling: Mapping[str, Any]) -> None:
    role = "rolling CPU macro summary"
    macros = _exact_keys(rolling.get("macro_summary"), tuple(TRACK_SPECS), role)
    for track, spec in TRACK_SPECS.items():
        payload = macros[track]
        if payload.get("metric") != spec["metric"] or payload.get("realized_value_metric") != spec["value_metric"]:
            _fail(role, f"{track} headline/value metric changed")
        if payload.get("aggregation") != "unweighted_mean_over_chains" or payload.get("chain_registry") != list(CHAINS):
            _fail(role, f"{track} chain-macro contract changed")
        if payload.get("budget_definition") != spec["budget"]:
            _fail(role, f"{track} reporting-budget contract changed")
        model_order = [item[0] for item in spec["models"]]
        models = _exact_keys(payload.get("models"), model_order, f"{role}/{track}/models")
        headline: dict[str, dict[str, float]] = {}
        value: dict[str, dict[str, float]] = {}
        for model in model_order:
            model_payload = models[model]
            per_chain = _exact_keys(model_payload.get("per_chain"), CHAINS, f"{role}/{track}/{model}/per_chain")
            headline[model] = {chain: _finite(per_chain[chain], f"{role}/{track}/{model}/{chain}") for chain in CHAINS}
            _close(model_payload.get("macro_mean"), statistics.fmean(headline[model].values()), f"{role}/{track}/{model}/macro")
            rv = model_payload.get("realized_value", {})
            if rv.get("metric") != spec["value_metric"]:
                _fail(role, f"{track}/{model} realized-value metric changed")
            rv_chain = _exact_keys(rv.get("per_chain"), CHAINS, f"{role}/{track}/{model}/value/per_chain")
            value[model] = {chain: _finite(rv_chain[chain], f"{role}/{track}/{model}/value/{chain}") for chain in CHAINS}
            _close(rv.get("macro_mean"), statistics.fmean(value[model].values()), f"{role}/{track}/{model}/value/macro")

        pairs = payload.get("pairwise_deltas", {})
        if pairs.get("model_order") != model_order or pairs.get("frozen_before_main") is not True:
            _fail(role, f"{track} pair registry was not frozen before main")
        if pairs.get("post_hoc_champion_selection") is not False or pairs.get("pair_registry") != "all_unordered_pairs_from_protocol_fixed_model_order":
            _fail(role, f"{track} pairwise reporting contract changed")
        expected_pairs = list(itertools.combinations(model_order, 2))
        comparisons = pairs.get("comparisons")
        if not isinstance(comparisons, list) or [(row.get("left_model"), row.get("right_model")) for row in comparisons] != expected_pairs:
            _fail(role, f"{track} does not report every protocol-fixed unordered pair")
        for row, (left, right) in zip(comparisons, expected_pairs):
            _validate_pair_report(row.get("headline", {}), headline[left], headline[right], f"{role}/{track}/{left}-{right}/headline")
            _validate_pair_report(row.get("realized_value", {}), value[left], value[right], f"{role}/{track}/{left}-{right}/value")


def _validate_raw_audit(
    paths: ArtifactPaths,
    audit: Mapping[str, Any],
    candidate_hashes: Mapping[str, str],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    role = "raw-label audit"
    if audit.get("schema_version") != RAW_AUDIT_SCHEMA:
        _fail(role, "expected raw-label-audit/v2")
    if audit.get("candidate_root") != "data/processed_v2":
        _fail(role, "candidate root is not portable/canonical")
    if audit.get("selection") != {"chains": list(CHAINS), "snapshots": ["main", "fold2"], "tracks": ["A", "B"]}:
        _fail(role, "audit selection is not the complete 6x2x2 release audit")
    definition = audit.get("definition", {})
    expected_definition = {
        "unit": "exporter-importer-stage lane",
        "raw_value_unit": "kUSD",
        "aggregation": "sum HS6 within each stage-year, sum five annual values, divide by 5",
        "missing_stage_year_value": 0.0,
        "window_length_years": 5,
        "positive_rule": "raw_late_calendar_mean_kusd > 100",
        "candidate_early_absence_rule": "raw_early_calendar_mean_kusd <= 100",
        "stored_lateval_rule": "raw late calendar mean for positives; zero for negatives",
    }
    if definition != expected_definition:
        _fail(role, "label/value definition changed")
    summary = audit.get("summary", {})
    if summary.get("all_pass") is not True or summary.get("audit_instances") != 24 or summary.get("passing_instances") != 24 or summary.get("failing_instances") != 0:
        _fail(role, "complete 24-instance audit is not green")
    for field in ("y_mismatches", "lateval_mismatches", "early_absence_violations", "negative_class_lateval_violations"):
        if summary.get(field) != 0:
            _fail(role, f"nonzero audit failure count: {field}")
    raw = audit.get("raw_source", {})
    if raw.get("archive_name") != "BACI_HS92_V202401b.zip" or raw.get("path") != raw.get("archive_name"):
        _fail(role, "raw source is not the portable BACI V202401b archive role")
    _hex64(raw.get("sha256"), f"{role}/raw source")
    _integer(raw.get("size_bytes"), f"{role}/raw size", minimum=1)

    rows = audit.get("audits")
    if not isinstance(rows, list) or len(rows) != 24:
        _fail(role, "expected 24 detailed audit rows")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("chain"), row.get("snapshot"), row.get("track"))
        if key in seen or key[0] not in CHAINS or key[1] not in ("main", "fold2") or key[2] not in ("A", "B"):
            _fail(role, f"duplicate/unknown audit key {key!r}")
        seen.add(key)
        candidate = row.get("candidate_file")
        digest = _verify_hash(paths, candidate, row.get("candidate_sha256"), f"{role}/{key}", hashes)
        if candidate_hashes.get(candidate) != digest:
            _fail(role, f"{key} candidate hash disagrees with rolling CPU provenance")
        checks = row.get("checks")
        if row.get("all_pass") is not True or not isinstance(checks, Mapping) or any(not isinstance(check, Mapping) or check.get("pass") is not True for check in checks.values()):
            _fail(role, f"{key} detailed audit is not all-pass")
        if row.get("stored_positive_rows") != row.get("raw_positive_rows"):
            _fail(role, f"{key} positive counts do not reconcile")
    return {"raw_sha256": raw["sha256"], "raw_size": raw["size_bytes"], "rows": rows}


def _validate_robustness(
    paths: ArtifactPaths,
    robustness: Mapping[str, Any],
    rolling: Mapping[str, Any],
    candidate_hashes: Mapping[str, str],
    raw_info: Mapping[str, Any],
    hashes: dict[Path, str],
) -> None:
    role = "robustness"
    if robustness.get("schema_version") != ROBUSTNESS_SCHEMA or robustness.get("benchmark_version") != BENCHMARK_VERSION:
        _fail(role, "expected current robustness schema 2 / benchmark 2.1-dev")
    protocol = robustness.get("protocol", {})
    expected_flags = {
        "all_chain_choices_frozen_before_any_candidate_label_parse": True,
        "all_chain_models_frozen_before_main_open": True,
        "main_labels_used_for_feature_fitting_imputation_or_calibration": False,
        "main_labels_used_for_model_or_hyperparameter_selection": False,
        "main_model_champion_selected": False,
        "raw_persistence_or_threshold_labels_use_production_aggregation_helper": False,
        "rolling_artifact_verified_before_any_candidate_label_parse": True,
        "selection_source": "fold2 only",
        "transductive_split_used": False,
    }
    if protocol != expected_flags:
        _fail(role, "frozen rolling/anti-leakage protocol flags changed")
    prespec = robustness.get("prespecification", {})
    if prespec.get("chains") != list(CHAINS) or prespec.get("thresholds_kusd") != [50.0, 100.0, 250.0]:
        _fail(role, "chain/threshold prespecification changed")
    if prespec.get("late_years") != [2018, 2019, 2020, 2021, 2022]:
        _fail(role, "late-year prespecification changed")
    if prespec.get("threshold_candidate_policy") != "outcome-only relabeling of the fixed default-100-kUSD candidate cohort; early eligibility is not rebuilt":
        _fail(role, "threshold sensitivity is not fixed-cohort outcome-only relabeling")
    persistence = prespec.get("persistence", {})
    if persistence != {
        "active_year_definition": "raw annual stage total > 100 kUSD",
        "default_label_changed": False,
        "minimum_active_years": 3,
        "positive_rule": ">=3 active years among 2018-2022",
    }:
        _fail(role, "persistence prespecification changed")
    if prespec.get("uncertainty") != {
        "A": "exporter-cluster bootstrap",
        "B1": "exporter-cluster bootstrap",
        "B2": "exporter-stage entry bootstrap",
        "draws": 200,
    }:
        _fail(role, "bootstrap cluster-unit contract changed")

    selection = robustness.get("selection_artifact", {})
    if selection.get("path") != "results_v2/metrics/rolling_cpu_baselines.json":
        _fail(role, "selection artifact role changed")
    rolling_digest = _sha256(paths.rolling)
    if selection.get("sha256") != rolling_digest:
        _fail(role, "selection artifact hash does not match rolling CPU bytes")
    if selection.get("verified_input_hashes") != 24 or selection.get("verified_before_any_candidate_label_parse") is not True or selection.get("selected_C_fields_checked_before_historical_fit") is not True:
        _fail(role, "selection artifact freeze verification is incomplete")
    choices = selection.get("frozen_choices")
    if choices != prespec.get("frozen_C"):
        _fail(role, "frozen choices disagree with prespecification")
    canonical = json.dumps(choices, sort_keys=True, separators=(",", ":")).encode("utf-8")
    choices_digest = hashlib.sha256(canonical).hexdigest()
    if selection.get("choices_sha256") != choices_digest or prespec.get("freeze_digest_sha256") != choices_digest:
        _fail(role, "frozen-choice digest does not reconcile")

    inputs = robustness.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 24:
        _fail(role, "expected 24 robustness input records")
    seen: set[str] = set()
    for item in inputs:
        path = item.get("path")
        if path in seen:
            _fail(role, f"duplicate robustness input {path!r}")
        seen.add(path)
        digest = _verify_hash(paths, path, item.get("sha256"), f"{role}/input/{path}", hashes)
        if candidate_hashes.get(path) != digest:
            _fail(role, f"input hash disagrees with rolling CPU for {path}")
    if set(candidate_hashes) != seen:
        _fail(role, "robustness inputs do not exactly cover the rolling 24-file cohort")
    raw = robustness.get("raw_source", {})
    if raw.get("sha256") != raw_info["raw_sha256"] or raw.get("size_bytes") != raw_info["raw_size"]:
        _fail(role, "raw BACI provenance disagrees with raw-label audit")
    output = robustness.get("outputs", {})
    _verify_hash(paths, output.get("csv_path"), output.get("csv_sha256"), f"{role}/CSV", hashes)
    if output.get("csv_rows") != 504:
        _fail(role, "robustness CSV does not record the complete 504-row suite")
    _validate_robustness_structure(robustness)


def _robustness_macro_row(
    robustness: Mapping[str, Any],
    *,
    sensitivity: str,
    slice_name: str,
    track: str,
    model: str,
    metric: str,
    expected_defined: int,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in robustness.get("macro_summary", [])
        if row.get("sensitivity") == sensitivity
        and row.get("slice") == slice_name
        and row.get("track") == track
        and row.get("model") == model
    ]
    role = f"robustness macro/{sensitivity}/{slice_name}/{track}/{model}"
    if len(matches) != 1:
        _fail(role, f"expected one row, found {len(matches)}")
    row = matches[0]
    if row.get("primary_metric") != metric or row.get("chains_with_defined_metric") != expected_defined:
        _fail(role, "metric/defined-chain count changed")
    per_chain = _exact_keys(row.get("per_chain"), CHAINS, f"{role}/per_chain")
    values = [_finite(value, f"{role}/{chain}") for chain, value in per_chain.items() if value is not None]
    if len(values) != expected_defined:
        _fail(role, "defined per-chain values do not match the declared count")
    _close(row.get("unweighted_chain_macro_mean"), statistics.fmean(values), f"{role}/macro")
    return row


def _validate_robustness_structure(robustness: Mapping[str, Any]) -> None:
    role = "robustness result structure"
    chains = _exact_keys(robustness.get("chains"), CHAINS, f"{role}/chains")
    expected_slices = {
        "identity": {"exporter_seen", "exporter_unseen", "importer_seen", "importer_unseen"},
        "entity_exclusion": {"exclude_hubs", "exclude_bad_iso", "exclude_hubs_and_bad_iso"},
        "threshold_outcome_only": {"threshold_50_kusd", "threshold_100_kusd", "threshold_250_kusd"},
        "persistence": {"active_at_least_3_of_5_years_above_100_kusd"},
    }
    for chain in CHAINS:
        payload = chains[chain]
        if payload.get("raw_reconciliation", {}).get("track_a", {}).get("pass") is not True or payload.get("raw_reconciliation", {}).get("track_b_lanes", {}).get("pass") is not True:
            _fail(role, f"{chain} raw reconciliation is not green")
        sensitivities = payload.get("sensitivities", {})
        if set(sensitivities) != set(expected_slices):
            _fail(role, f"{chain} sensitivity registry is incomplete")
        for sensitivity, slices in expected_slices.items():
            if set(sensitivities[sensitivity]) != slices:
                _fail(role, f"{chain}/{sensitivity} slice registry is incomplete")


def _validate_b1_coverage(
    paths: ArtifactPaths,
    coverage: Mapping[str, Any],
    registry_info: Mapping[str, Any],
    raw_info: Mapping[str, Any],
    candidate_hashes: Mapping[str, str],
    summary_by_chain: Mapping[str, Mapping[str, Any]],
    hashes: dict[Path, str],
) -> None:
    role = "B1 candidate coverage"
    if coverage.get("schema_version") != B1_COVERAGE_SCHEMA or coverage.get("status") != "PASS":
        _fail(role, "expected schema-2 PASS coverage artifact")
    expected_definition = {
        "activity": "five-year calendar mean strictly greater than 100 kUSD",
        "denominator": "all late processed-stage lanes and exporter-stage entries realized by early upstream-qualified nonincumbent exporters",
        "covered": "late destination imported the same processed stage from any exporter in the early window",
        "excluded_market": "late destination had no early processed-stage demand",
    }
    if coverage.get("definition") != expected_definition:
        _fail(role, "candidate-universe coverage definition changed")
    source = coverage.get("source", {})
    if source.get("dataset") != "CEPII BACI HS92 V202401b" or source.get("archive_sha256") != raw_info["raw_sha256"] or source.get("archive_bytes") != raw_info["raw_size"]:
        _fail(role, "raw BACI provenance disagrees with raw-label audit")
    if source.get("cache_schema") != "upgrade-bench/private-baci-filtered-cache/1":
        _fail(role, "unexpected private filtered-cache schema")
    _hex64(source.get("cache_manifest_sha256"), f"{role}/cache manifest")

    registry = coverage.get("registry", {})
    for field, path, schema in (
        ("audit", paths.registry_audit, REGISTRY_AUDIT_SCHEMA),
        ("evidence", paths.registry_evidence, REGISTRY_EVIDENCE_SCHEMA),
    ):
        record = registry.get(field, {})
        expected_path = path.relative_to(paths.root).as_posix()
        if record.get("path") != expected_path or record.get("schema_version") != schema:
            _fail(role, f"registry {field} role/schema changed")
        _verify_hash(paths, expected_path, record.get("sha256"), f"{role}/registry/{field}", hashes)
    reg_summary = registry_info["summary"]
    if registry.get("chain_count") != len(CHAINS) or registry.get("active_hs6_count") != reg_summary["included_codes"]:
        _fail(role, "registry counts disagree with audited registry")

    for field in ("generator",):
        record = coverage.get(field, {})
        _verify_hash(paths, record.get("path"), record.get("sha256"), f"{role}/{field}", hashes)
    protocol_hashes = coverage.get("protocol_sha256")
    expected_protocol_roles = {
        "cache_reader",
        "calendar_aggregation",
        "candidate_generator",
        "entry_view_builder",
        "registry_loader",
    }
    if not isinstance(protocol_hashes, Mapping) or set(protocol_hashes) != expected_protocol_roles:
        _fail(role, "protocol-source hash registry is incomplete")
    for field, record in protocol_hashes.items():
        _verify_hash(paths, record.get("path"), record.get("sha256"), f"{role}/protocol/{field}", hashes)

    inputs = coverage.get("input_sha256")
    if not isinstance(inputs, Mapping) or len(inputs) != 24:
        _fail(role, "expected 24 candidate/entry input hashes")
    for relative, digest in inputs.items():
        observed = _verify_hash(paths, relative, digest, f"{role}/input/{relative}", hashes)
        if "/candidates_firsttime_" in f"/{relative}" and candidate_hashes.get(relative) != observed:
            _fail(role, f"candidate input disagrees with rolling/raw audit: {relative}")

    snapshots = _exact_keys(coverage.get("snapshots"), ("fold2", "main"), f"{role}/snapshots")
    expected_windows = {
        "fold2": ("history", [1998, 1999, 2000, 2001, 2002], [2008, 2009, 2010, 2011, 2012]),
        "main": ("target", [2008, 2009, 2010, 2011, 2012], [2018, 2019, 2020, 2021, 2022]),
    }
    for snapshot, (temporal_role, early, late) in expected_windows.items():
        payload = snapshots[snapshot]
        if payload.get("temporal_role") != temporal_role or payload.get("early_years") != early or payload.get("late_years") != late:
            _fail(role, f"{snapshot} temporal role/window changed")
        rows = payload.get("chains")
        if not isinstance(rows, list) or [row.get("chain") for row in rows] != sorted(CHAINS):
            _fail(role, f"{snapshot} chain coverage rows are incomplete/out of order")
        totals = payload.get("totals", {})
        additive = (
            "n_upstream_qualified_nonincumbent_exporter_stage_pairs",
            "n_early_demand_destination_stage_pairs",
            "n_released_candidate_entries",
            "n_all_realized_entries",
            "n_covered_realized_entries",
            "n_inactive_only_realized_entries",
            "n_mixed_realized_entries",
            "n_all_late_start_lanes",
            "n_eligible_market_late_start_lanes",
            "n_previously_inactive_market_late_start_lanes",
            "all_late_start_value_kusd",
            "eligible_market_late_start_value_kusd",
            "previously_inactive_market_late_start_value_kusd",
        )
        for field in additive:
            _close(totals.get(field), sum(_finite(row.get("totals", {}).get(field), f"{role}/{snapshot}/{field}") for row in rows), f"{role}/{snapshot}/totals/{field}", atol=1e-6)
        denominator_specs = (
            ("realized_entry_coverage", "n_covered_realized_entries", "n_all_realized_entries"),
            ("late_start_lane_coverage", "n_eligible_market_late_start_lanes", "n_all_late_start_lanes"),
            ("late_start_value_coverage", "eligible_market_late_start_value_kusd", "all_late_start_value_kusd"),
            ("previously_inactive_market_lane_share", "n_previously_inactive_market_late_start_lanes", "n_all_late_start_lanes"),
            ("previously_inactive_market_value_share", "previously_inactive_market_late_start_value_kusd", "all_late_start_value_kusd"),
        )
        for row in [*rows, {"chain": "TOTAL", "totals": totals}]:
            values = row["totals"]
            if row.get("candidate_reconciliation", {}).get("pass") is False:
                _fail(role, f"{snapshot}/{row['chain']} candidate reconciliation failed")
            for metric, numerator, denominator in denominator_specs:
                expected = values[numerator] / values[denominator] if values[denominator] else 0.0
                _close(values.get(metric), expected, f"{role}/{snapshot}/{row['chain']}/{metric}")
        if snapshot == "main":
            for row in rows:
                chain = row["chain"]
                reconciliation = row.get("candidate_reconciliation", {})
                summary = summary_by_chain[chain]
                if reconciliation.get("candidate_lanes") != summary["track_b_candidate_lanes"] or reconciliation.get("positive_lanes") != summary["track_b_positive_lanes"] or reconciliation.get("candidate_entries") != summary["track_b_unique_entries"] or reconciliation.get("positive_entries") != summary["track_b_positive_entries"]:
                    _fail(role, f"main/{chain} candidate counts disagree with dataset summary")
                _close(reconciliation.get("eligible_market_late_start_value_kusd"), summary["track_b_observed_late_value_kusd"], f"{role}/main/{chain}/covered value", atol=1e-6)


def _validate_gpu_postfreeze_binding(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> None:
    """Cross-bind the public GPU summary to the current registry equivalence proof."""

    role = "GPU post-freeze semantic attestation"
    try:
        gpu_postfreeze.verify_summary_binding(
            payload,
            artifact_path=paths.gpu_postfreeze_attestation,
            root=paths.root,
            require_full_inventory=False,
        )
    except gpu_postfreeze.AttestationError as exc:
        _fail(role, str(exc))
    observed = _sha256(paths.gpu_postfreeze_attestation)
    previous = hashes.setdefault(paths.gpu_postfreeze_attestation, observed)
    if previous != observed:
        _fail(role, "attestation changed after public binding verification")


def _validate_gpu_summary(
    payload: Mapping[str, Any],
    candidate_hashes: Mapping[str, str] | None = None,
) -> None:
    role = "formal GPU summary"
    if payload.get("schema_version") != GPU_SUMMARY_SCHEMA or payload.get("status") != "complete":
        _fail(role, "expected complete gpu-main-summary/1")
    if payload.get("protocol") != GPU_PROTOCOL or payload.get("target_fold") != "main" or payload.get("aggregation") != "calendar_mean":
        _fail(role, "strict rolling fold2-to-main protocol changed")
    if payload.get("seeds") != [0, 1, 2, 3, 4] or payload.get("complete_chain_family_jobs") != 12 or payload.get("complete_task_evaluations") != 36:
        _fail(role, "formal job/task/seed completeness gate failed")
    for field in ("manifest_sha256", "run_config_sha256"):
        _hex64(payload.get(field), f"{role}/{field}")
    if payload.get("manifest_artifact_role") != "results_v2/gpu_rolling/frozen_manifest.json":
        _fail(role, "freeze-manifest artifact role changed")
    if payload.get("run_config_artifact_role") != "configs/v2_gpu_rolling.json":
        _fail(role, "run-config artifact role changed")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        _fail(role, "missing formal run identifier")
    reporting = payload.get("reporting_policy", {})
    expected_reporting = {
        "all_frozen_families_reported_separately": True,
        "all_six_chains_in_every_macro": True,
        "main_test_champion_selected": False,
        "macro_weighting": "unweighted arithmetic mean of six preregistered chain means",
        "raw_score_cross_seed_averaging": False,
        "bootstrap_rng_seed_mechanically_verified": True,
    }
    if reporting != expected_reporting:
        _fail(role, "GPU reporting/no-champion policy changed")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 36:
        _fail(role, "expected 36 chain-track-family records")
    record_map: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (row.get("chain"), row.get("track"), row.get("family"))
        if key in record_map or key[0] not in CHAINS or key[1] not in ("a", "b1", "b2") or key[2] not in ("kge", "nbfnet"):
                _fail(role, f"duplicate/unknown record key {key!r}")
        if row.get("primary_metric") != {"a": "lane_average_precision", "b1": "entry_average_precision", "b2": "conditional_recall_at_3"}[key[1]]:
            _fail(role, f"wrong primary metric for {key!r}")
        values = row.get("primary_values_by_seed")
        if not isinstance(values, list) or len(values) != 5:
            _fail(role, f"{key!r} does not contain five seed values")
        finite_values = [_finite(value, f"{role}/{key}/seed") for value in values]
        _close(row.get("primary_mean"), statistics.fmean(finite_values), f"{role}/{key}/mean")
        _close(
            row.get("primary_std_across_seeds"),
            statistics.pstdev(finite_values),
            f"{role}/{key}/std",
        )
        for field in (
            "history_candidate_sha256",
            "target_candidate_sha256",
            "selection_sha256",
            "metric_artifact_sha256",
            "score_artifact_sha256",
        ):
            _hex64(row.get(field), f"{role}/{key}/{field}")
        source_stem = "candidates" if key[1] == "a" else "candidates_firsttime"
        expected_history = f"data/processed_v2/{source_stem}_{key[0]}_fold2.csv"
        expected_target = f"data/processed_v2/{source_stem}_{key[0]}.csv"
        if row.get("history_candidate_role") != expected_history or row.get("target_candidate_role") != expected_target:
            _fail(role, f"candidate artifact roles changed for {key!r}")
        if candidate_hashes is not None and (
            row.get("history_candidate_sha256") != candidate_hashes.get(expected_history)
            or row.get("target_candidate_sha256") != candidate_hashes.get(expected_target)
        ):
            _fail(role, f"candidate hashes disagree with rolling CPU for {key!r}")
        record_map[key] = row
    expected_keys = set(itertools.product(CHAINS, ("a", "b1", "b2"), ("kge", "nbfnet")))
    if set(record_map) != expected_keys:
        _fail(role, "records do not cover the exact 6x3x2 grid")

    macros = payload.get("macro_summary")
    if not isinstance(macros, list) or len(macros) != 6:
        _fail(role, "expected six track-family macro rows")
    seen: set[tuple[str, str]] = set()
    for row in macros:
        key = (row.get("track"), row.get("family"))
        if key in seen or key[0] not in ("a", "b1", "b2") or key[1] not in ("kge", "nbfnet"):
            _fail(role, f"duplicate/unknown macro key {key!r}")
        seen.add(key)
        chain_means = [record_map[(chain, *key)]["primary_mean"] for chain in CHAINS]
        if row.get("n_chains") != 6 or row.get("primary_metric") != record_map[(CHAINS[0], *key)]["primary_metric"]:
            _fail(role, f"{key!r} macro metadata changed")
        _close(row.get("mean_across_six_chain_means"), statistics.fmean(chain_means), f"{role}/{key}/macro mean")
        _close(
            row.get("std_across_six_chain_means"),
            statistics.pstdev(chain_means),
            f"{role}/{key}/macro std",
        )
        seed_rows = row.get("per_seed_macro")
        if not isinstance(seed_rows, list) or [item.get("seed") for item in seed_rows] != [0, 1, 2, 3, 4]:
            _fail(role, f"{key!r} per-seed macro registry changed")
        for seed, seed_row in enumerate(seed_rows):
            expected = statistics.fmean(record_map[(chain, *key)]["primary_values_by_seed"][seed] for chain in CHAINS)
            _close(seed_row.get("mean_across_six_chains"), expected, f"{role}/{key}/seed/{seed}")


def _verify_value_diagnostics_first(paths: ArtifactPaths) -> Mapping[str, Any]:
    """Run the diagnostics artifact's own provenance/accounting verifier first."""
    import v2_value_diagnostics

    try:
        payload = v2_value_diagnostics.verify_existing_output(
            paths.value_diagnostics,
            paths.value_diagnostics_csv,
        )
    except Exception as exc:
        _fail("value diagnostics verifier", str(exc))
    if not isinstance(payload, Mapping):
        _fail("value diagnostics verifier", "verifier did not return an object")
    return payload


def _safe_relative_reference(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(role, "missing repository-relative source path")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or "\\" in value:
        _fail(role, f"unsafe/non-portable repository path {value!r}")
    return value


def _validate_value_diagnostics(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    summary_by_chain: Mapping[str, Mapping[str, Any]],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Validate and reduce the neutral value/headroom interface.

    The returned interface deliberately contains both frozen GPU families and
    an outcome-ranked descriptive oracle.  It contains no family comparison,
    winner, or target-period champion selection.
    """
    role = "value diagnostics"
    _exact_keys(
        payload,
        (
            "accounting",
            "benchmark_version",
            "chains",
            "generated_at_utc",
            "inputs",
            "protocol",
            "runtime",
            "schema_version",
        ),
        role,
    )
    if (
        payload.get("schema_version") != VALUE_DIAGNOSTICS_SCHEMA
        or payload.get("benchmark_version") != BENCHMARK_VERSION
    ):
        _fail(role, "expected current value-diagnostics schema 1 / benchmark 2.1-dev")
    if not isinstance(payload.get("generated_at_utc"), str) or not payload["generated_at_utc"]:
        _fail(role, "missing generation timestamp")

    runtime = _exact_keys(
        payload.get("runtime"),
        ("numpy", "pandas", "python", "script_sha256"),
        f"{role}/runtime",
    )
    expected_generator = "tools/v2_value_diagnostics.py"
    generator_path = _repo_file(paths, expected_generator, f"{role}/generator")
    if generator_path != paths.value_diagnostics_generator.resolve():
        _fail(role, "configured diagnostics generator is not the canonical repository file")
    _verify_hash(
        paths,
        expected_generator,
        runtime.get("script_sha256"),
        f"{role}/generator",
        hashes,
    )

    protocol = _exact_keys(
        payload.get("protocol"),
        (
            "budgets",
            "cpu_policy",
            "dollar_accounting",
            "gpu_policy",
            "oracle_policy",
            "post_hoc_main_champion_selected",
            "selection_window",
            "target_labels_used_for_model_or_family_selection",
            "target_window",
            "uncertainty",
        ),
        f"{role}/protocol",
    )
    if (
        protocol.get("target_labels_used_for_model_or_family_selection") is not False
        or protocol.get("post_hoc_main_champion_selected") is not False
    ):
        _fail(role, "target-label/no-main-champion policy failed")
    if protocol.get("gpu_policy") != "report both frozen families and all five seeds separately":
        _fail(role, "both-family/five-seed reporting policy changed")
    if protocol.get("oracle_policy") != (
        "outcome-only budget-matched diagnostic; never a deployable model or selection candidate"
    ):
        _fail(role, "oracle is not restricted to a descriptive upper-bound diagnostic")
    if protocol.get("dollar_accounting") != (
        "A and B1 are distinct task pools; B2 is nested in B1 and is excluded from sums."
    ):
        _fail(role, "B1/B2 non-additivity policy changed")
    budgets = _exact_keys(
        protocol.get("budgets"),
        (
            "track_a_global",
            "track_a_per_exporter",
            "track_b1_global_entries",
            "track_b1_per_exporter_entries",
            "track_b2_per_positive_entry",
        ),
        f"{role}/budgets",
    )
    for track, spec in VALUE_TRACK_SPECS.items():
        if budgets.get(spec["budget_field"]) != spec["budget"]:
            _fail(role, f"{track} paper diagnostic budget changed")

    inputs = _exact_keys(
        payload.get("inputs"),
        ("cpu_rolling", "gpu_frozen_manifest", "gpu_score_artifacts", "gpu_summary"),
        f"{role}/inputs",
    )
    core_specs = {
        "cpu_rolling": (paths.rolling, ROLLING_SCHEMA),
        "gpu_summary": (paths.gpu_summary, GPU_SUMMARY_SCHEMA),
    }
    for name, (source, schema) in core_specs.items():
        record = _exact_keys(
            inputs.get(name),
            ("path", "schema_version", "sha256"),
            f"{role}/inputs/{name}",
        )
        expected_relative = source.relative_to(paths.root).as_posix()
        if record.get("path") != expected_relative or record.get("schema_version") != schema:
            _fail(role, f"{name} does not bind the canonical schema/path")
        _verify_hash(
            paths,
            expected_relative,
            record.get("sha256"),
            f"{role}/inputs/{name}",
            hashes,
        )
    manifest = _exact_keys(
        inputs.get("gpu_frozen_manifest"),
        ("path", "sha256"),
        f"{role}/inputs/gpu_frozen_manifest",
    )
    if manifest.get("path") != "results_v2/gpu_rolling/frozen_manifest.json":
        _fail(role, "GPU frozen-manifest path changed")
    _verify_hash(
        paths,
        manifest.get("path"),
        manifest.get("sha256"),
        f"{role}/inputs/gpu_frozen_manifest",
        hashes,
    )

    score_rows = inputs.get("gpu_score_artifacts")
    if not isinstance(score_rows, list) or len(score_rows) != 36:
        _fail(role, "expected the complete 6x3x2 GPU score provenance inventory")
    score_inventory: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    score_keys = (
        "chain",
        "family",
        "history_holdout_selection_metric_mean",
        "metric_path",
        "metric_sha256",
        "rows",
        "score_path",
        "score_sha256",
        "seeds",
        "selected_hyperparameters",
        "selected_model",
        "selection_path",
        "selection_sha256",
        "track",
    )
    for row in score_rows:
        row = _exact_keys(row, score_keys, f"{role}/GPU score inventory row")
        key = (row.get("chain"), row.get("track"), row.get("family"))
        if (
            key in score_inventory
            or key[0] not in CHAINS
            or key[1] not in VALUE_TRACK_SPECS
            or key[2] not in VALUE_FAMILY_TEX
        ):
            _fail(role, f"duplicate/unknown GPU score inventory key {key!r}")
        if row.get("seeds") != list(VALUE_SEEDS) or _integer(
            row.get("rows"), f"{role}/{key}/rows", minimum=1
        ) < 1:
            _fail(role, f"incomplete GPU score inventory metadata for {key!r}")
        if not isinstance(row.get("selected_model"), str) or not row["selected_model"]:
            _fail(role, f"missing historically selected model for {key!r}")
        if not isinstance(row.get("selected_hyperparameters"), Mapping):
            _fail(role, f"missing historical hyperparameters for {key!r}")
        _finite(row.get("history_holdout_selection_metric_mean"), f"{role}/{key}/history metric")
        for prefix in ("score", "selection", "metric"):
            relative = _safe_relative_reference(
                row.get(f"{prefix}_path"), f"{role}/{key}/{prefix}"
            )
            _verify_hash(
                paths,
                relative,
                row.get(f"{prefix}_sha256"),
                f"{role}/{key}/{prefix}",
                hashes,
            )
        score_inventory[key] = row
    if set(score_inventory) != set(
        itertools.product(CHAINS, VALUE_TRACK_SPECS, VALUE_FAMILY_TEX)
    ):
        _fail(role, "GPU score provenance does not cover the exact 6x3x2 grid")

    chains = _exact_keys(payload.get("chains"), CHAINS, f"{role}/chains")
    candidate_roles = {
        "track_a": "data/processed_v2/candidates_{chain}.csv",
        "track_b_lane_pool": "data/processed_v2/candidates_firsttime_{chain}.csv",
        "track_b1": "data/processed_v2/entries_firsttime_{chain}.csv",
        "track_b2": "data/processed_v2/destinations_given_entry_{chain}.csv",
    }
    chain_family_means = {
        track: {family: [] for family in VALUE_FAMILY_TEX}
        for track in VALUE_TRACK_SPECS
    }
    chain_oracles = {track: [] for track in VALUE_TRACK_SPECS}
    pooled_oracle_selected = {track: 0.0 for track in VALUE_TRACK_SPECS}
    pooled_oracle_total = {track: 0.0 for track in VALUE_TRACK_SPECS}
    observed_by_chain: dict[str, dict[str, float]] = {}

    for chain in CHAINS:
        chain_payload = _exact_keys(
            chains[chain], ("input_audit", "tracks"), f"{role}/{chain}"
        )
        input_audit = _exact_keys(
            chain_payload.get("input_audit"),
            candidate_roles,
            f"{role}/{chain}/input audit",
        )
        for audit_role, template in candidate_roles.items():
            record = input_audit[audit_role]
            if not isinstance(record, Mapping):
                _fail(role, f"malformed candidate audit for {chain}/{audit_role}")
            expected_path = template.format(chain=chain)
            if record.get("path") != expected_path:
                _fail(role, f"candidate path changed for {chain}/{audit_role}")
            _verify_hash(
                paths,
                expected_path,
                record.get("sha256"),
                f"{role}/{chain}/{audit_role}",
                hashes,
            )

        tracks = _exact_keys(
            chain_payload.get("tracks"),
            VALUE_TRACK_SPECS,
            f"{role}/{chain}/tracks",
        )
        observed_by_chain[chain] = {}
        for track, spec in VALUE_TRACK_SPECS.items():
            track_payload = _exact_keys(
                tracks[track],
                ("cpu_models", "gpu_families", "oracle_interpretation", "task"),
                f"{role}/{chain}/{track}",
            )
            if track_payload.get("task") != spec["task"] or track_payload.get(
                "oracle_interpretation"
            ) != "outcome-ranked, same-cohort same-budget descriptive upper bound":
                _fail(role, f"task/oracle interpretation changed for {chain}/{track}")
            cpu_models = _exact_keys(
                track_payload.get("cpu_models"),
                spec["cpu_models"],
                f"{role}/{chain}/{track}/CPU models",
            )
            if any(
                not isinstance(item, Mapping)
                or item.get("post_hoc_main_champion_selection") is not False
                for item in cpu_models.values()
            ):
                _fail(role, f"CPU no-main-champion flag failed for {chain}/{track}")
            families = _exact_keys(
                track_payload.get("gpu_families"),
                VALUE_FAMILY_TEX,
                f"{role}/{chain}/{track}/GPU families",
            )
            reference_oracle: tuple[float, float, float] | None = None
            for family in VALUE_FAMILY_TEX:
                family_payload = _exact_keys(
                    families[family],
                    (
                        "per_seed",
                        "post_hoc_main_champion_selection",
                        "raw_score_policy",
                        "score_audit",
                        "selected_hyperparameters",
                        "selected_model",
                        "selection_source",
                        "summary_across_seeds",
                    ),
                    f"{role}/{chain}/{track}/{family}",
                )
                if (
                    family_payload.get("post_hoc_main_champion_selection") is not False
                    or family_payload.get("raw_score_policy")
                    != "one diagnostic per seed; no raw-score averaging"
                    or family_payload.get("selection_source")
                    != "historical fold2 frozen manifest"
                ):
                    _fail(role, f"GPU frozen/no-champion policy failed for {chain}/{track}/{family}")
                inventory_audit = {
                    key: value
                    for key, value in score_inventory[(chain, track, family)].items()
                    if key not in {"chain", "track", "family"}
                }
                if family_payload.get("score_audit") != inventory_audit:
                    _fail(role, f"GPU score audit is not source-complete for {chain}/{track}/{family}")

                per_seed = family_payload.get("per_seed")
                if (
                    not isinstance(per_seed, list)
                    or [item.get("seed") for item in per_seed] != list(VALUE_SEEDS)
                ):
                    _fail(role, f"expected five ordered seeds for {chain}/{track}/{family}")
                scalar_rows: list[dict[str, float]] = []
                family_reference: tuple[float, float, float] | None = None
                for seed_row in per_seed:
                    seed = seed_row["seed"]
                    diagnostic = seed_row.get("diagnostic")
                    if not isinstance(diagnostic, Mapping):
                        _fail(role, f"missing diagnostic for {chain}/{track}/{family}/s{seed}")
                    report = diagnostic.get(spec["diagnostic"])
                    point = report.get("point") if isinstance(report, Mapping) else None
                    if not isinstance(point, Mapping):
                        _fail(role, f"missing point estimate for {chain}/{track}/{family}/s{seed}")
                    model = _finite(point.get(spec["model_field"]), f"{role}/{chain}/{track}/{family}/s{seed}/model")
                    oracle = _finite(point.get(spec["oracle_field"]), f"{role}/{chain}/{track}/{family}/s{seed}/oracle")
                    gap = _finite(point.get(spec["gap_field"]), f"{role}/{chain}/{track}/{family}/s{seed}/gap")
                    headroom = _finite(point.get("headroom_kusd"), f"{role}/{chain}/{track}/{family}/s{seed}/headroom")
                    oracle_selected = _finite(point.get("oracle_selected_observed_late_value_kusd"), f"{role}/{chain}/{track}/{family}/s{seed}/oracle selected")
                    model_selected = _finite(point.get("model_selected_observed_late_value_kusd"), f"{role}/{chain}/{track}/{family}/s{seed}/model selected")
                    total = _finite(point.get("total_observed_late_value_kusd"), f"{role}/{chain}/{track}/{family}/s{seed}/total")
                    pooled_oracle = _finite(point.get(spec["pooled_oracle_field"]), f"{role}/{chain}/{track}/{family}/s{seed}/pooled oracle")
                    if not all(0.0 <= value <= 1.0 + 1e-12 for value in (model, oracle, gap, pooled_oracle)):
                        _fail(role, f"capture/gap outside [0,1] for {chain}/{track}/{family}/s{seed}")
                    if total <= 0.0 or min(oracle_selected, model_selected, headroom) < -1e-7:
                        _fail(role, f"invalid observed-value accounting for {chain}/{track}/{family}/s{seed}")
                    _close(gap, oracle - model, f"{role}/{chain}/{track}/{family}/s{seed}/oracle gap")
                    _close(headroom, oracle_selected - model_selected, f"{role}/{chain}/{track}/{family}/s{seed}/headroom", atol=1e-6)
                    _close(pooled_oracle, oracle_selected / total, f"{role}/{chain}/{track}/{family}/s{seed}/pooled oracle", atol=1e-12)
                    current_reference = (oracle, oracle_selected, total)
                    if family_reference is None:
                        family_reference = current_reference
                    else:
                        for index, label in enumerate(("oracle", "oracle selected", "total")):
                            _close(current_reference[index], family_reference[index], f"{role}/{chain}/{track}/{family}/{label}", atol=1e-7)
                    scalar_rows.append(
                        {
                            "model_value_capture": model,
                            "oracle_value_capture": oracle,
                            "oracle_gap_value_capture": gap,
                            "headroom_kusd": headroom,
                        }
                    )

                summary = _exact_keys(
                    family_payload.get("summary_across_seeds"),
                    (
                        "headroom_kusd",
                        "model_value_capture",
                        "n_seeds",
                        "oracle_gap_value_capture",
                        "oracle_value_capture",
                    ),
                    f"{role}/{chain}/{track}/{family}/summary",
                )
                if summary.get("n_seeds") != len(VALUE_SEEDS):
                    _fail(role, f"wrong seed count for {chain}/{track}/{family}")
                for field in (
                    "model_value_capture",
                    "oracle_value_capture",
                    "oracle_gap_value_capture",
                    "headroom_kusd",
                ):
                    item = _exact_keys(
                        summary.get(field),
                        ("mean", "std"),
                        f"{role}/{chain}/{track}/{family}/summary/{field}",
                    )
                    values = [row[field] for row in scalar_rows]
                    _close(item.get("mean"), statistics.fmean(values), f"{role}/{chain}/{track}/{family}/{field}/mean", atol=1e-10)
                    _close(item.get("std"), statistics.pstdev(values), f"{role}/{chain}/{track}/{family}/{field}/std", atol=1e-10)
                chain_family_means[track][family].append(
                    _finite(summary["model_value_capture"]["mean"], f"{role}/{chain}/{track}/{family}/chain mean")
                )
                if family_reference is None:
                    _fail(role, f"missing oracle reference for {chain}/{track}/{family}")
                if reference_oracle is None:
                    reference_oracle = family_reference
                else:
                    for index, label in enumerate(("oracle", "oracle selected", "total")):
                        _close(family_reference[index], reference_oracle[index], f"{role}/{chain}/{track}/cross-family {label}", atol=1e-7)

            if reference_oracle is None:
                _fail(role, f"missing oracle for {chain}/{track}")
            oracle, oracle_selected, total = reference_oracle
            chain_oracles[track].append(oracle)
            pooled_oracle_selected[track] += oracle_selected
            pooled_oracle_total[track] += total
            observed_by_chain[chain][track] = total

    accounting = _exact_keys(
        payload.get("accounting"),
        ("per_chain", "policy", "totals"),
        f"{role}/accounting",
    )
    if accounting.get("policy") != (
        "Track B2 re-ranks the same positive-entry dollars counted once in B1; never add B1 and B2."
    ):
        _fail(role, "accounting policy no longer forbids B1+B2")
    per_chain_accounting = _exact_keys(
        accounting.get("per_chain"), CHAINS, f"{role}/accounting/per chain"
    )
    accounting_fields = (
        "b2_is_nested_in_b1",
        "forbidden_naive_a_plus_b1_plus_b2_kusd",
        "track_a_observed_late_value_kusd",
        "track_b1_observed_late_value_kusd",
        "track_b2_nested_same_dollars_kusd",
        "unique_project_observed_late_value_kusd",
    )
    summed = {"a": 0.0, "b1": 0.0, "b2": 0.0}
    for chain in CHAINS:
        row = _exact_keys(
            per_chain_accounting[chain], accounting_fields, f"{role}/accounting/{chain}"
        )
        if row.get("b2_is_nested_in_b1") is not True:
            _fail(role, f"B2 is not marked nested for {chain}")
        a_value = _finite(row.get("track_a_observed_late_value_kusd"), f"{role}/{chain}/A value")
        b1_value = _finite(row.get("track_b1_observed_late_value_kusd"), f"{role}/{chain}/B1 value")
        b2_value = _finite(row.get("track_b2_nested_same_dollars_kusd"), f"{role}/{chain}/B2 value")
        _close(b1_value, b2_value, f"{role}/{chain}/B1-B2 nested dollars", atol=1e-7)
        _close(row.get("unique_project_observed_late_value_kusd"), a_value + b1_value, f"{role}/{chain}/unique dollars", atol=1e-6)
        _close(row.get("forbidden_naive_a_plus_b1_plus_b2_kusd"), a_value + b1_value + b2_value, f"{role}/{chain}/naive audit", atol=1e-6)
        _close(a_value, observed_by_chain[chain]["a"], f"{role}/{chain}/A diagnostic accounting", atol=1e-6)
        _close(b1_value, observed_by_chain[chain]["b1"], f"{role}/{chain}/B1 diagnostic accounting", atol=1e-6)
        _close(b2_value, observed_by_chain[chain]["b2"], f"{role}/{chain}/B2 diagnostic accounting", atol=1e-6)
        _close(a_value, summary_by_chain[chain]["track_a_observed_late_value_kusd"], f"{role}/{chain}/A dataset summary", atol=1e-6)
        _close(b1_value, summary_by_chain[chain]["track_b_observed_late_value_kusd"], f"{role}/{chain}/B1 dataset summary", atol=1e-6)
        _close(b2_value, summary_by_chain[chain]["track_b2_observed_late_value_kusd"], f"{role}/{chain}/B2 dataset summary", atol=1e-6)
        summed["a"] += a_value
        summed["b1"] += b1_value
        summed["b2"] += b2_value

    totals = _exact_keys(
        accounting.get("totals"),
        (
            "b2_excluded_from_unique_sum",
            "forbidden_naive_a_plus_b1_plus_b2_kusd",
            "track_a_observed_late_value_kusd",
            "track_b1_observed_late_value_kusd",
            "track_b2_nested_same_dollars_kusd",
            "unique_project_observed_late_value_kusd",
        ),
        f"{role}/accounting/totals",
    )
    if totals.get("b2_excluded_from_unique_sum") is not True:
        _fail(role, "B2 is not excluded from the unique-dollar sum")
    _close(totals.get("track_a_observed_late_value_kusd"), summed["a"], f"{role}/total A", atol=1e-6)
    _close(totals.get("track_b1_observed_late_value_kusd"), summed["b1"], f"{role}/total B1", atol=1e-6)
    _close(totals.get("track_b2_nested_same_dollars_kusd"), summed["b2"], f"{role}/total B2", atol=1e-6)
    _close(summed["b1"], summed["b2"], f"{role}/aggregate B1-B2 nested dollars", atol=1e-6)
    _close(totals.get("unique_project_observed_late_value_kusd"), summed["a"] + summed["b1"], f"{role}/total unique dollars", atol=1e-6)
    _close(totals.get("forbidden_naive_a_plus_b1_plus_b2_kusd"), sum(summed.values()), f"{role}/total naive audit", atol=1e-6)

    return {
        "macro": {
            track: {
                **{
                    family: statistics.fmean(chain_family_means[track][family])
                    for family in VALUE_FAMILY_TEX
                },
                "oracle": statistics.fmean(chain_oracles[track]),
            }
            for track in VALUE_TRACK_SPECS
        },
        "pooled_oracle": {
            track: pooled_oracle_selected[track] / pooled_oracle_total[track]
            for track in VALUE_TRACK_SPECS
        },
        "accounting": {key: totals[key] for key in totals},
        "target_labels_used_for_selection": False,
        "post_hoc_main_champion_selected": False,
    }


def _verify_loco_summary_first(
    paths: ArtifactPaths,
    hashes: dict[Path, str],
) -> Mapping[str, Any]:
    """Run the public-only LOCO verifier from the exact bound source file.

    This intentionally calls only ``verify_outputs``.  It neither imports the
    private formal controller nor dereferences the private formal summary,
    receipt, manifest, component tree, or score artifacts named as provenance
    inside the sanitized public JSON.
    """

    role = "public LOCO summary verifier"
    expected_generator = _repo_file(
        paths,
        "tools/summarize_v2_loco_results.py",
        f"{role}/generator",
    )
    configured_generator = paths.loco_summary_generator.resolve()
    if configured_generator != expected_generator:
        _fail(role, "configured verifier is not the canonical repository file")

    raw_inputs = (
        paths.loco_summary_generator,
        paths.loco_summary,
        paths.loco_summary_csv,
    )
    for source in raw_inputs:
        if source.is_symlink() or not source.is_file():
            _fail(role, f"missing, non-regular, or symbolic-link input: {source.name}")
    inputs = tuple(source.resolve() for source in raw_inputs)
    before = {source: _sha256(source) for source in inputs}

    spec = importlib.util.spec_from_file_location(
        f"_upgrade_bench_public_loco_{before[configured_generator][:16]}",
        configured_generator,
    )
    if spec is None or spec.loader is None:
        _fail(role, "cannot load the canonical public verifier")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(role, f"cannot import the canonical public verifier: {exc}")
    origin = Path(getattr(module, "__file__", "")).resolve()
    if origin != configured_generator:
        _fail(role, "loaded verifier origin is not canonical")
    verify = getattr(module, "verify_outputs", None)
    if not callable(verify):
        _fail(role, "canonical public verifier has no verify_outputs function")
    try:
        payload = verify(paths.loco_summary, paths.loco_summary_csv)
    except Exception as exc:
        _fail(role, str(exc))
    if not isinstance(payload, Mapping):
        _fail(role, "verifier did not return a validated object")

    after = {source: _sha256(source) for source in inputs}
    if after != before:
        _fail(role, "JSON, CSV, or verifier source changed during verification")
    hashes.update(before)
    return payload


def _loco_stat(
    value: Any,
    role: str,
    *,
    expected_n: int,
    mean_lower: float,
) -> dict[str, float | int]:
    row = _exact_keys(value, ("n", "mean", "population_std"), role)
    n = _integer(row.get("n"), f"{role}/n")
    if n != expected_n:
        _fail(role, f"expected n={expected_n}, found {n}")
    mean = _finite(row.get("mean"), f"{role}/mean")
    population_std = _finite(
        row.get("population_std"), f"{role}/population std"
    )
    if not mean_lower <= mean <= 1.0:
        _fail(role, f"mean is outside [{mean_lower}, 1]")
    if not 0.0 <= population_std <= 1.0:
        _fail(role, "population std is outside [0, 1]")
    return {"n": n, "mean": mean, "population_std": population_std}


def _validate_loco_summary(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Reduce the verified public LOCO artifact to the paper-only interface."""

    role = "public LOCO summary"
    required = {
        "schema_version": LOCO_PUBLIC_SUMMARY_SCHEMA,
        "protocol": LOCO_PROTOCOL,
        "status": "complete",
        "run_id": LOCO_RUN_ID,
        "claim_scope": LOCO_CLAIM_SCOPE,
        "paper_eligible": True,
        "paper_eligibility_scope": LOCO_PAPER_SCOPE,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            _fail(f"{role}/{field}", f"expected {expected!r}")
    if payload.get("paper_eligible") is not True:
        _fail(role, "paper_eligible must be the JSON boolean true")

    design = _exact_keys(
        payload.get("design"),
        (
            "chains",
            "modes",
            "seeds",
            "expected_component_count",
            "verified_component_count",
            "comparison_definition",
        ),
        f"{role}/design",
    )
    expected_design = {
        "chains": list(CHAINS),
        "modes": list(LOCO_MODES),
        "seeds": list(LOCO_SEEDS),
        "expected_component_count": 60,
        "verified_component_count": 60,
        "comparison_definition": "in_domain_minus_loco",
    }
    for field, expected in expected_design.items():
        if design.get(field) != expected:
            _fail(f"{role}/design/{field}", f"expected {expected!r}")

    canonical_config = _repo_file(
        paths,
        "configs/v2_loco_formal.json",
        f"{role}/config",
    )
    if canonical_config != paths.loco_config.resolve() or paths.loco_config.is_symlink():
        _fail(role, "configured LOCO config is not the canonical regular repository file")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail(f"{role}/provenance", "expected an object")
    _verify_hash(
        paths,
        "configs/v2_loco_formal.json",
        provenance.get("config_sha256"),
        f"{role}/config",
        hashes,
    )

    metrics = _exact_keys(
        payload.get("metrics"),
        tuple(metric for metric, _ in LOCO_METRIC_SPECS),
        f"{role}/metrics",
    )
    reduced_metrics: dict[str, Any] = {}
    for metric, _ in LOCO_METRIC_SPECS:
        metric_role = f"{role}/metrics/{metric}"
        row = _exact_keys(
            metrics.get(metric),
            ("by_mode", "by_chain", "matched_gap"),
            metric_role,
        )
        by_mode = _exact_keys(
            row.get("by_mode"), LOCO_MODES, f"{metric_role}/by mode"
        )
        mode_stats = {
            mode: _loco_stat(
                by_mode.get(mode),
                f"{metric_role}/{mode}",
                expected_n=30,
                mean_lower=0.0,
            )
            for mode in LOCO_MODES
        }
        gap = _exact_keys(
            row.get("matched_gap"),
            ("definition", "n", "mean", "population_std"),
            f"{metric_role}/matched gap",
        )
        if gap.get("definition") != "in_domain_minus_loco":
            _fail(metric_role, "only the paired in_domain_minus_loco gap is permitted")
        gap_stat = _loco_stat(
            {key: gap[key] for key in ("n", "mean", "population_std")},
            f"{metric_role}/matched gap",
            expected_n=30,
            mean_lower=-1.0,
        )
        reduced_metrics[metric] = {
            "by_mode": mode_stats,
            "matched_gap": gap_stat,
        }

    return {
        "chain_count": len(CHAINS),
        "seed_count": len(LOCO_SEEDS),
        "verified_component_count": 60,
        "matched_pair_count_per_metric": 30,
        "metrics": reduced_metrics,
    }


def _verify_ultra_summary_first(
    paths: ArtifactPaths,
    hashes: dict[Path, str],
) -> Mapping[str, Any]:
    """Run the exact public-only ULTRA verifier without opening private scores."""

    role = "public ULTRA summary verifier"
    expected_generator = _repo_file(
        paths,
        "tools/summarize_v2_ultra_results.py",
        f"{role}/generator",
    )
    configured_generator = paths.ultra_summary_generator.resolve()
    if configured_generator != expected_generator:
        _fail(role, "configured verifier is not the canonical repository file")

    raw_inputs = (
        paths.ultra_summary_generator,
        paths.ultra_summary,
        paths.ultra_summary_csv,
        paths.ultra_config,
        paths.ultra_formal_controller,
        paths.gpu_summary,
    )
    for source in raw_inputs:
        if source.is_symlink() or not source.is_file():
            _fail(role, f"missing, non-regular, or symbolic-link input: {source.name}")
    inputs = tuple(source.resolve() for source in raw_inputs)
    before = {source: _sha256(source) for source in inputs}

    spec = importlib.util.spec_from_file_location(
        f"_upgrade_bench_public_ultra_{before[configured_generator][:16]}",
        configured_generator,
    )
    if spec is None or spec.loader is None:
        _fail(role, "cannot load the canonical public verifier")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(role, f"cannot import the canonical public verifier: {exc}")
    if Path(getattr(module, "__file__", "")).resolve() != configured_generator:
        _fail(role, "loaded verifier origin is not canonical")
    verify = getattr(module, "verify_outputs", None)
    if not callable(verify):
        _fail(role, "canonical public verifier has no verify_outputs function")
    try:
        payload = verify(paths.ultra_summary, paths.ultra_summary_csv)
    except Exception as exc:
        _fail(role, str(exc))
    if not isinstance(payload, Mapping):
        _fail(role, "verifier did not return a validated object")

    after = {source: _sha256(source) for source in inputs}
    if after != before:
        _fail(role, "public inputs or verifier source changed during verification")
    hashes.update(before)
    return payload


def _validate_ultra_summary(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Reduce the verified external-pretrained reference to paper-safe values."""

    role = "public ULTRA summary"
    for field, expected in (
        ("schema_version", ULTRA_PUBLIC_SUMMARY_SCHEMA),
        ("protocol", ULTRA_PROTOCOL),
        ("status", ULTRA_PUBLIC_STATUS),
        ("run_id", ULTRA_RUN_ID),
    ):
        if payload.get(field) != expected:
            _fail(f"{role}/{field}", f"expected {expected!r}")

    # The canonical public verifier checks provenance hashes in full.  The
    # paper gate separately binds the exact current source bytes so the public
    # summary cannot be paired with a different controller or configuration.
    for relative, path in (
        ("tools/summarize_v2_ultra_results.py", paths.ultra_summary_generator),
        ("configs/v2_ultra_formal.json", paths.ultra_config),
        ("tools/v2_ultra_formal.py", paths.ultra_formal_controller),
        ("results_v2/metrics/v2_gpu_rolling_summary.json", paths.gpu_summary),
    ):
        canonical = _repo_file(paths, relative, f"{role}/{relative}")
        if path.is_symlink() or path.resolve() != canonical:
            _fail(role, f"configured source is not canonical: {relative}")
        hashes[path.resolve()] = _sha256(path)

    model = payload.get("model")
    design = payload.get("design")
    if not isinstance(model, Mapping) or not isinstance(design, Mapping):
        _fail(role, "model and design must be objects")
    if design.get("chains") != list(CHAINS):
        _fail(role, "design chains are not the canonical six-chain order")
    if design.get("tasks") != list(ULTRA_TASK_SPECS):
        _fail(role, "design tasks are not A/B1/B2 in canonical order")
    checkpoint_count = _integer(
        design.get("checkpoint_count"), f"{role}/design/checkpoint count"
    )
    if checkpoint_count != 1:
        _fail(role, "exactly one fixed checkpoint is required")
    target_graph = design.get("target_early_graph_used")
    target_labels = design.get("target_labels_used_for_training_or_selection")
    seed_disclosed = model.get("checkpoint_training_seed_disclosed")
    if target_graph is not True or target_labels is not False or seed_disclosed is not False:
        _fail(
            role,
            "zero-shot target-graph, no-target-label, and undisclosed-training-seed boundaries changed",
        )

    records = payload.get("metric_records")
    if not isinstance(records, list) or len(records) != len(CHAINS) * len(ULTRA_TASK_SPECS):
        _fail(role, "metric_records must contain all 18 chain-task records")
    expected_pairs = [
        (task, chain) for task in ULTRA_TASK_SPECS for chain in CHAINS
    ]
    normalized_records: list[dict[str, Any]] = []
    for index, (task, chain) in enumerate(expected_pairs):
        row = _exact_keys(
            records[index],
            (
                "chain",
                "task",
                "headline_metric",
                "headline_value",
                "value_metric",
                "value_value",
            ),
            f"{role}/metric_records/{index}",
        )
        spec = ULTRA_TASK_SPECS[task]
        expected = {
            "chain": chain,
            "task": task,
            "headline_metric": spec["headline_metric"],
            "value_metric": spec["value_metric"],
        }
        for field, expected_value in expected.items():
            if row.get(field) != expected_value:
                _fail(
                    f"{role}/metric_records/{index}/{field}",
                    f"expected {expected_value!r}",
                )
        headline = _finite(row.get("headline_value"), f"{role}/{task}/{chain}/headline")
        value = _finite(row.get("value_value"), f"{role}/{task}/{chain}/value")
        if not 0.0 <= headline <= 1.0 or not 0.0 <= value <= 1.0:
            _fail(role, "ULTRA retrieval and value metrics must lie in [0, 1]")
        normalized_records.append({**expected, "headline_value": headline, "value_value": value})

    task_summaries = _exact_keys(
        payload.get("task_summaries"),
        tuple(ULTRA_TASK_SPECS),
        f"{role}/task_summaries",
    )
    normalized_summaries: dict[str, dict[str, Any]] = {}
    for task, spec in ULTRA_TASK_SPECS.items():
        summary = _exact_keys(
            task_summaries[task],
            (
                "headline_metric",
                "value_metric",
                "unweighted_six_chain_headline_mean",
                "unweighted_six_chain_value_mean",
            ),
            f"{role}/task_summaries/{task}",
        )
        if summary.get("headline_metric") != spec["headline_metric"] or summary.get(
            "value_metric"
        ) != spec["value_metric"]:
            _fail(role, f"task summary metric names changed for {task}")
        task_rows = [row for row in normalized_records if row["task"] == task]
        headline_mean = statistics.fmean(row["headline_value"] for row in task_rows)
        value_mean = statistics.fmean(row["value_value"] for row in task_rows)
        _close(
            summary.get("unweighted_six_chain_headline_mean"),
            headline_mean,
            f"{role}/{task}/headline mean",
            atol=1e-15,
        )
        _close(
            summary.get("unweighted_six_chain_value_mean"),
            value_mean,
            f"{role}/{task}/value mean",
            atol=1e-15,
        )
        normalized_summaries[task] = {
            "unweighted_six_chain_headline_mean": headline_mean,
            "unweighted_six_chain_value_mean": value_mean,
        }

    comparisons = _exact_keys(
        payload.get("reference_comparisons"),
        tuple(ULTRA_TASK_SPECS),
        f"{role}/reference_comparisons",
    )
    normalized_comparisons: dict[str, Any] = {}
    for task in ULTRA_TASK_SPECS:
        families = _exact_keys(
            comparisons[task], ("kge", "nbfnet"), f"{role}/comparisons/{task}"
        )
        normalized_comparisons[task] = {}
        for family in ("kge", "nbfnet"):
            family_row = _exact_keys(
                families[family],
                ("counts", "reference_unweighted_six_chain_mean"),
                f"{role}/comparisons/{task}/{family}",
            )
            counts_row = _exact_keys(
                family_row["counts"],
                ("higher", "equal", "lower"),
                f"{role}/comparisons/{task}/{family}/counts",
            )
            counts = {
                key: _integer(value, f"{role}/{task}/{family}/{key}")
                for key, value in counts_row.items()
            }
            if any(value < 0 for value in counts.values()) or sum(counts.values()) != len(CHAINS):
                _fail(role, f"comparison counts do not sum to six for {task}/{family}")
            reference_mean = _finite(
                family_row["reference_unweighted_six_chain_mean"],
                f"{role}/{task}/{family}/reference mean",
            )
            if not 0.0 <= reference_mean <= 1.0:
                _fail(role, "trained-reference mean is outside [0, 1]")
            normalized_comparisons[task][family] = {
                "counts": counts,
                "reference_unweighted_six_chain_mean": reference_mean,
            }

    abstract = _exact_keys(
        payload.get("abstract_rule"),
        ("tasks", "abstract_should_mention_ultra"),
        f"{role}/abstract_rule",
    )
    if not isinstance(abstract["abstract_should_mention_ultra"], bool):
        _fail(role, "abstract_should_mention_ultra must be boolean")
    abstract_tasks = _exact_keys(
        abstract["tasks"], tuple(ULTRA_TASK_SPECS), f"{role}/abstract_rule/tasks"
    )
    normalized_abstract_tasks: dict[str, Any] = {}
    any_eligible = False
    for task in ULTRA_TASK_SPECS:
        rule = abstract_tasks[task]
        if not isinstance(rule, Mapping):
            _fail(role, f"abstract rule for {task} must be an object")
        eligible = rule.get("eligible_for_abstract_mention")
        same_side = _integer(
            rule.get("same_side_of_both_chain_count"),
            f"{role}/abstract_rule/{task}/same-side count",
        )
        if not isinstance(eligible, bool) or not 0 <= same_side <= len(CHAINS):
            _fail(role, f"abstract rule fields are invalid for {task}")
        if eligible and same_side < 5:
            _fail(role, f"abstract eligibility lacks five-chain support for {task}")
        any_eligible = any_eligible or eligible
        normalized_abstract_tasks[task] = {
            "eligible_for_abstract_mention": eligible,
            "same_side_of_both_chain_count": same_side,
        }
    if abstract["abstract_should_mention_ultra"] is not any_eligible:
        _fail(role, "overall abstract gate is not the OR of the task gates")

    repeat = payload.get("sheep_exact_repeat")
    if not isinstance(repeat, Mapping):
        _fail(role, "sheep_exact_repeat must be an object")
    score_gate = repeat.get("score_gate_pass")
    metric_gate = repeat.get("metric_gate_pass")
    if score_gate is not True or metric_gate is not True:
        _fail(role, "the prespecified sheep repeatability gates did not both pass")

    return {
        "chain_count": len(CHAINS),
        "task_count": len(ULTRA_TASK_SPECS),
        "headline_record_count": len(normalized_records),
        "checkpoint_count": checkpoint_count,
        "target_early_graph_used": target_graph,
        "target_labels_used_for_training_or_selection": target_labels,
        "checkpoint_training_seed_disclosed": seed_disclosed,
        "repeatability_score_gate_pass": score_gate,
        "repeatability_metric_gate_pass": metric_gate,
        "metric_records": normalized_records,
        "task_summaries": normalized_summaries,
        "reference_comparisons": normalized_comparisons,
        "abstract_rule": {
            "tasks": normalized_abstract_tasks,
            "abstract_should_mention_ultra": abstract[
                "abstract_should_mention_ultra"
            ],
        },
    }


def _verify_gbdt_summary_first(
    paths: ArtifactPaths,
    hashes: dict[Path, str],
) -> Mapping[str, Any]:
    """Run the exact GBDT JSON/CSV verifier before reading paper values."""

    role = "public GBDT summary verifier"
    expected_generator = _repo_file(
        paths,
        "tools/v2_gbdt_baselines.py",
        f"{role}/generator",
    )
    configured_generator = paths.gbdt_summary_generator.resolve()
    if configured_generator != expected_generator:
        _fail(role, "configured verifier is not the canonical repository file")

    shared_source = _repo_file(paths, GBDT_SHARED_SOURCE_ROLE, f"{role}/shared source")
    raw_inputs = (
        paths.gbdt_summary_generator,
        paths.gbdt_summary,
        paths.gbdt_summary_csv,
        paths.gbdt_config,
        shared_source,
    )
    for source in raw_inputs:
        if source.is_symlink() or not source.is_file():
            _fail(role, f"missing, non-regular, or symbolic-link input: {source.name}")
    inputs = tuple(source.resolve() for source in raw_inputs)
    before = {source: _sha256(source) for source in inputs}

    module_name = f"_upgrade_bench_public_gbdt_{before[configured_generator][:16]}"
    spec = importlib.util.spec_from_file_location(module_name, configured_generator)
    if spec is None or spec.loader is None:
        _fail(role, "cannot load the canonical public verifier")
    module = importlib.util.module_from_spec(spec)
    prior_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(role, f"cannot import the canonical public verifier: {exc}")
    finally:
        if prior_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior_module
    if Path(getattr(module, "__file__", "")).resolve() != configured_generator:
        _fail(role, "loaded verifier origin is not canonical")
    verify = getattr(module, "verify_existing_output", None)
    if not callable(verify):
        _fail(role, "canonical public verifier has no verify_existing_output function")
    try:
        verify(paths.gbdt_summary, paths.gbdt_summary_csv)
    except Exception as exc:
        _fail(role, str(exc))

    payload = _load(paths.gbdt_summary)
    after = {source: _sha256(source) for source in inputs}
    if after != before:
        _fail(role, "JSON, CSV, config, or verifier source changed during verification")
    hashes.update(before)
    return payload


def _gbdt_interval(value: Any, role: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        _fail(role, "expected a two-element 95% interval")
    low = _finite(value[0], f"{role}/lower")
    high = _finite(value[1], f"{role}/upper")
    if not 0.0 <= low <= high <= 1.0:
        _fail(role, "95% interval must be ordered within [0, 1]")
    return low, high


def _validate_gbdt_summary(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Reduce the fully verified GBDT artifact to the schema-7 paper interface."""

    role = "public GBDT summary"
    if payload.get("schema_version") != GBDT_PUBLIC_SUMMARY_SCHEMA:
        _fail(f"{role}/schema_version", f"expected {GBDT_PUBLIC_SUMMARY_SCHEMA!r}")
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        _fail(f"{role}/benchmark_version", f"expected {BENCHMARK_VERSION!r}")
    if payload.get("status") != GBDT_PUBLIC_STATUS:
        _fail(f"{role}/status", f"expected {GBDT_PUBLIC_STATUS!r}")

    config = _load(paths.gbdt_config)
    for field, expected in (
        ("schema_version", GBDT_CONFIG_SCHEMA),
        ("protocol", GBDT_CONFIG_PROTOCOL),
        ("status", GBDT_CONFIG_STATUS),
    ):
        if config.get(field) != expected:
            _fail(f"{role}/config/{field}", f"expected {expected!r}")
    config_record = _exact_keys(payload.get("config"), ("path", "sha256"), f"{role}/config")
    if config_record.get("path") != "configs/v2_gbdt_baselines.json":
        _fail(role, "result config path is not the frozen canonical role")

    inputs = _exact_keys(
        payload.get("inputs"),
        ("candidate_files", "public_sources"),
        f"{role}/inputs",
    )
    public_sources = _exact_keys(
        inputs.get("public_sources"),
        (
            "tools/v2_gbdt_baselines.py",
            GBDT_SHARED_SOURCE_ROLE,
            "configs/v2_gbdt_baselines.json",
        ),
        f"{role}/inputs/public_sources",
    )
    source_paths = {
        "tools/v2_gbdt_baselines.py": paths.gbdt_summary_generator,
        GBDT_SHARED_SOURCE_ROLE: paths.root / GBDT_SHARED_SOURCE_ROLE,
        "configs/v2_gbdt_baselines.json": paths.gbdt_config,
    }
    for relative, path in source_paths.items():
        canonical = _repo_file(paths, relative, f"{role}/{relative}")
        if path.is_symlink() or path.resolve() != canonical:
            _fail(role, f"configured source is not canonical: {relative}")
        _verify_hash(
            paths,
            relative,
            public_sources.get(relative),
            f"{role}/{relative}",
            hashes,
        )
    if config_record.get("sha256") != public_sources.get(
        "configs/v2_gbdt_baselines.json"
    ):
        _fail(role, "config record and public-source hash disagree")
    candidate_files = inputs.get("candidate_files")
    if not isinstance(candidate_files, list) or len(candidate_files) != 24:
        _fail(role, "verified candidate inventory must contain 24 files")

    runtime = _exact_keys(
        payload.get("runtime"),
        (
            "python",
            "platform",
            "cpu_model",
            "logical_cpu_cores",
            "numpy",
            "pandas",
            "scikit_learn",
            "wall_elapsed_seconds",
            "fit_count_upper_bound",
        ),
        f"{role}/runtime",
    )
    cpu_model = runtime.get("cpu_model")
    if not isinstance(cpu_model, str) or GBDT_CPU_MODEL_RE.fullmatch(cpu_model) is None:
        _fail(role, "runtime CPU model is empty or not TeX-safe portable ASCII")
    logical_cpu_cores = _integer(
        runtime.get("logical_cpu_cores"),
        f"{role}/runtime/logical_cpu_cores",
        minimum=1,
    )
    fit_count_upper_bound = _integer(
        runtime.get("fit_count_upper_bound"),
        f"{role}/runtime/fit_count_upper_bound",
        minimum=1,
    )
    wall_elapsed_seconds = _finite(
        runtime.get("wall_elapsed_seconds"),
        f"{role}/runtime/wall_elapsed_seconds",
    )
    if wall_elapsed_seconds <= 0.0:
        _fail(role, "runtime wall_elapsed_seconds must be positive")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("bootstrap"), Mapping):
        _fail(role, "protocol bootstrap record is missing")
    bootstrap_draws = _integer(
        protocol["bootstrap"].get("draws"),
        f"{role}/protocol/bootstrap/draws",
        minimum=1,
    )
    if bootstrap_draws != config.get("uncertainty", {}).get("draws"):
        _fail(role, "result bootstrap draws disagree with the frozen config")

    chains = _exact_keys(payload.get("chains"), CHAINS, f"{role}/chains")
    macro_summary = _exact_keys(
        payload.get("macro_summary"),
        tuple(GBDT_TRACK_SPECS),
        f"{role}/macro_summary",
    )
    records: list[dict[str, Any]] = []
    for chain in CHAINS:
        chain_payload = chains[chain]
        if not isinstance(chain_payload, Mapping):
            _fail(f"{role}/chains/{chain}", "expected an object")
        for track, spec in GBDT_TRACK_SPECS.items():
            task = chain_payload.get(track)
            if not isinstance(task, Mapping):
                _fail(f"{role}/{chain}/{track}", "missing task object")
            models = _exact_keys(
                task.get("models"),
                (GBDT_MODEL_KEY,),
                f"{role}/{chain}/{track}/models",
            )
            model = models[GBDT_MODEL_KEY]
            if not isinstance(model, Mapping) or not isinstance(model.get("metrics"), Mapping):
                _fail(f"{role}/{chain}/{track}", "missing verified model metrics")
            metrics = model["metrics"]
            if track == "track_b2_conditional_destination_ranking":
                at_k = metrics.get("at_k")
                if not isinstance(at_k, Mapping) or not isinstance(at_k.get("k_3"), Mapping):
                    _fail(f"{role}/{chain}/{track}", "missing conditional k=3 metrics")
                headline = _finite(
                    at_k["k_3"].get("macro_recall"),
                    f"{role}/{chain}/{track}/headline",
                )
                lower, upper = _gbdt_interval(
                    at_k["k_3"].get("macro_recall_ci95"),
                    f"{role}/{chain}/{track}/headline_ci95",
                )
            else:
                headline = _finite(
                    metrics.get("average_precision"),
                    f"{role}/{chain}/{track}/headline",
                )
                lower, upper = _gbdt_interval(
                    metrics.get("average_precision_ci95"),
                    f"{role}/{chain}/{track}/headline_ci95",
                )
            if not 0.0 <= headline <= 1.0:
                _fail(f"{role}/{chain}/{track}/headline", "metric must lie in [0, 1]")
            records.append(
                {
                    "chain": chain,
                    "track": track,
                    "headline": headline,
                    "headline_ci95_low": lower,
                    "headline_ci95_high": upper,
                }
            )

    task_summaries: dict[str, dict[str, float]] = {}
    for track, spec in GBDT_TRACK_SPECS.items():
        summary = macro_summary[track]
        if not isinstance(summary, Mapping):
            _fail(f"{role}/macro_summary/{track}", "expected an object")
        if (
            summary.get("headline_metric") != spec["headline_metric"]
            or summary.get("realized_value_metric") != spec["value_metric"]
            or summary.get("aggregation") != "unweighted_mean_over_chains"
            or summary.get("chain_registry") != list(CHAINS)
            or summary.get("model") != GBDT_MODEL_KEY
        ):
            _fail(role, f"macro summary identity changed for {track}")
        headline_section = _exact_keys(
            summary.get("headline"),
            ("per_chain", "macro_mean", "std_across_chains"),
            f"{role}/macro_summary/{track}/headline",
        )
        value_section = _exact_keys(
            summary.get("realized_value"),
            ("per_chain", "macro_mean", "std_across_chains"),
            f"{role}/macro_summary/{track}/realized_value",
        )
        per_chain = _exact_keys(
            headline_section.get("per_chain"),
            CHAINS,
            f"{role}/macro_summary/{track}/headline/per_chain",
        )
        track_records = {row["chain"]: row for row in records if row["track"] == track}
        for chain in CHAINS:
            _close(
                per_chain.get(chain),
                track_records[chain]["headline"],
                f"{role}/macro_summary/{track}/headline/{chain}",
            )
        expected_headline_mean = statistics.fmean(
            track_records[chain]["headline"] for chain in CHAINS
        )
        _close(
            headline_section.get("macro_mean"),
            expected_headline_mean,
            f"{role}/macro_summary/{track}/headline/macro_mean",
        )
        value_mean = _finite(
            value_section.get("macro_mean"),
            f"{role}/macro_summary/{track}/realized_value/macro_mean",
        )
        if not 0.0 <= value_mean <= 1.0:
            _fail(role, f"realized-value mean is outside [0, 1] for {track}")
        task_summaries[track] = {
            "headline_mean": expected_headline_mean,
            "value_mean": value_mean,
        }

    return {
        "chain_count": len(CHAINS),
        "task_count": len(GBDT_TRACK_SPECS),
        "headline_record_count": len(records),
        "cpu_model": cpu_model,
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "logical_cpu_cores": logical_cpu_cores,
        "fit_count_upper_bound": fit_count_upper_bound,
        "bootstrap_draws": bootstrap_draws,
        "headline_records": records,
        "task_summaries": task_summaries,
    }


def _load_bound_module(
    paths: ArtifactPaths,
    *,
    relative: str,
    configured: Path,
    role: str,
) -> tuple[Any, Path]:
    """Load an exact repository-bound verifier without trusting import search."""

    expected = _repo_file(paths, relative, f"{role}/runner")
    configured = configured.resolve()
    if configured != expected or configured.is_symlink():
        _fail(role, "configured verifier is not the canonical regular repository file")
    digest = _sha256(configured)
    module_name = f"_upgrade_bench_{configured.stem}_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, configured)
    if spec is None or spec.loader is None:
        _fail(role, "cannot load the canonical verifier")
    module = importlib.util.module_from_spec(spec)
    prior_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(role, f"cannot import the canonical verifier: {exc}")
    finally:
        if prior_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior_module
    if Path(getattr(module, "__file__", "")).resolve() != configured:
        _fail(role, "loaded verifier origin is not canonical")
    return module, configured


def _verify_product_space_first(
    paths: ArtifactPaths,
    hashes: dict[Path, str],
) -> Mapping[str, Any]:
    """Run the product-space artifact's public keyed-score verifier first."""

    role = "product-space summary verifier"
    raw_inputs = (
        paths.product_space_summary,
        paths.product_space_summary_csv,
        paths.product_space_scores,
        paths.product_space_generator,
        paths.product_space_config,
    )
    for source in raw_inputs:
        if source.is_symlink() or not source.is_file():
            _fail(role, f"missing, non-regular, or symbolic-link input: {source.name}")
    inputs = tuple(source.resolve() for source in raw_inputs)
    before = {source: _sha256(source) for source in inputs}
    module, _ = _load_bound_module(
        paths,
        relative="tools/v2_product_space_density.py",
        configured=paths.product_space_generator,
        role=role,
    )
    verify = getattr(module, "verify_existing_output", None)
    if not callable(verify):
        _fail(role, "canonical verifier has no verify_existing_output function")
    try:
        verify(
            paths.product_space_summary,
            paths.product_space_summary_csv,
            paths.product_space_scores,
            verify_raw_archive=False,
        )
    except Exception as exc:
        _fail(role, str(exc))
    payload = _load(paths.product_space_summary)
    after = {source: _sha256(source) for source in inputs}
    if after != before:
        _fail(role, "JSON, CSV, keyed scores, config, or verifier changed during verification")
    hashes.update(before)
    return payload


def _validate_product_space_summary(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Reduce the verified product-space result to a paper-only interface."""

    role = "product-space summary"
    _exact_keys(
        payload,
        (
            "benchmark_version",
            "claim_scope",
            "cohorts",
            "config",
            "generated_at_utc",
            "inputs",
            "limitations",
            "macro_summary",
            "protocol",
            "runtime",
            "schema_version",
            "score_artifact",
            "scorers",
            "status",
        ),
        role,
    )
    for field, expected in (
        ("schema_version", PRODUCT_SPACE_SCHEMA),
        ("benchmark_version", BENCHMARK_VERSION),
        ("status", PRODUCT_SPACE_STATUS),
        ("claim_scope", PRODUCT_SPACE_CLAIM_SCOPE),
    ):
        if payload.get(field) != expected:
            _fail(f"{role}/{field}", f"expected {expected!r}")

    config = _load(paths.product_space_config)
    for field, expected in (
        ("schema_version", PRODUCT_SPACE_CONFIG_SCHEMA),
        ("protocol", PRODUCT_SPACE_CONFIG_PROTOCOL),
        ("status", PRODUCT_SPACE_CONFIG_STATUS),
    ):
        if config.get(field) != expected:
            _fail(f"{role}/config/{field}", f"expected {expected!r}")
    config_record = _exact_keys(
        payload.get("config"), ("path", "sha256"), f"{role}/config receipt"
    )
    if config_record.get("path") != "configs/v2_product_space_density.json":
        _fail(role, "result config path is not the canonical frozen role")
    _verify_hash(
        paths,
        config_record["path"],
        config_record.get("sha256"),
        f"{role}/config receipt",
        hashes,
    )

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        _fail(f"{role}/protocol", "expected an object")
    required_protocol = {
        "task": "B1 processed exporter-stage entry only",
        "selection_mode": "none_single_predeclared_formula",
        "historical_labels_used_for_selection": False,
        "main_labels_used_for_selection_or_calibration": False,
        "all_scorers_and_registry_mappings_frozen_before_any_outcome_read": True,
        "target_self_relation": "exclude q=p from numerator and denominator",
        "full_product_universe": (
            "complete HS92 product dictionary, not the six-chain registry union"
        ),
    }
    for field, expected in required_protocol.items():
        if protocol.get(field) != expected:
            _fail(f"{role}/protocol/{field}", f"expected {expected!r}")
    uncertainty = protocol.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        _fail(f"{role}/protocol/uncertainty", "expected an object")
    if (
        uncertainty.get("method") != "nonparametric_cluster_bootstrap"
        or uncertainty.get("cluster_unit") != "exporter"
        or uncertainty.get("interval") != "percentile_95"
        or uncertainty.get("ap_only") is not True
    ):
        _fail(role, "product-space uncertainty protocol changed")
    bootstrap_draws = _integer(
        uncertainty.get("draws"), f"{role}/protocol/uncertainty/draws", minimum=1
    )

    score = _exact_keys(
        payload.get("score_artifact"),
        (
            "canonical_order",
            "columns",
            "config_sha256",
            "freeze_sha256",
            "path",
            "purpose",
            "rows",
            "sha256",
        ),
        f"{role}/score artifact",
    )
    if score.get("path") != "results_v2/scores/v2_product_space_density_scores.csv":
        _fail(role, "keyed-score path is not canonical")
    if score.get("config_sha256") != config_record.get("sha256"):
        _fail(role, "keyed-score and result config hashes disagree")
    if score.get("freeze_sha256") != protocol.get("freeze_sha256"):
        _fail(role, "keyed-score and protocol freeze hashes disagree")
    _verify_hash(
        paths,
        score["path"],
        score.get("sha256"),
        f"{role}/keyed scores",
        hashes,
    )
    score_rows = _integer(score.get("rows"), f"{role}/score rows", minimum=1)

    scorers = _exact_keys(payload.get("scorers"), ("historical", "main"), f"{role}/scorers")
    matrix_by_cohort: dict[str, Mapping[str, Any]] = {}
    for cohort in ("historical", "main"):
        scorer = scorers[cohort]
        if not isinstance(scorer, Mapping) or not isinstance(
            scorer.get("matrix_audit"), Mapping
        ):
            _fail(f"{role}/scorers/{cohort}", "missing matrix audit")
        matrix = scorer["matrix_audit"]
        matrix_by_cohort[cohort] = matrix
        for field in ("countries", "products", "target_products"):
            _integer(matrix.get(field), f"{role}/{cohort}/{field}", minimum=1)
        if matrix.get("countries") != matrix.get("country_iso3_identities"):
            _fail(role, f"country identity coverage changed for {cohort}")
        if matrix.get("formula") != "prospective_hidalgo_density_target_diagonal_excluded":
            _fail(role, f"product-space formula changed for {cohort}")
        if matrix.get("target_diagonal_nonzero_before_exclusion") != matrix.get(
            "target_products"
        ):
            _fail(role, f"target diagonal audit is incomplete for {cohort}")
        _close(
            matrix.get("target_diagonal_max_after_exclusion"),
            0.0,
            f"{role}/{cohort}/target diagonal after exclusion",
        )
    for field in (
        "countries",
        "products",
        "target_products",
        "country_vocabulary_sha256",
        "product_vocabulary_sha256",
        "target_product_vocabulary_sha256",
    ):
        if matrix_by_cohort["historical"].get(field) != matrix_by_cohort["main"].get(
            field
        ):
            _fail(role, f"historical/main product universe differs for {field}")

    cohorts = _exact_keys(payload.get("cohorts"), ("historical", "main"), f"{role}/cohorts")
    macros = _exact_keys(
        payload.get("macro_summary"), ("historical", "main"), f"{role}/macro summary"
    )
    reduced_cohorts: dict[str, Any] = {}
    total_score_rows = 0
    for cohort in ("historical", "main"):
        chain_payloads = _exact_keys(cohorts[cohort], CHAINS, f"{role}/{cohort}/chains")
        per_chain: dict[str, Any] = {}
        rows_total = 0
        positives_total = 0
        membership_candidates = 0
        material_chains = 0
        for chain in CHAINS:
            item = chain_payloads[chain]
            if not isinstance(item, Mapping):
                _fail(f"{role}/{cohort}/{chain}", "expected an object")
            inputs = item.get("input")
            metrics = item.get("metrics")
            audit = item.get("score_audit")
            if not all(isinstance(value, Mapping) for value in (inputs, metrics, audit)):
                _fail(f"{role}/{cohort}/{chain}", "input, metrics, and score audit are required")
            n = _integer(inputs.get("rows"), f"{role}/{cohort}/{chain}/rows", minimum=1)
            positives = _integer(
                inputs.get("positives"), f"{role}/{cohort}/{chain}/positives", minimum=1
            )
            if metrics.get("n") != n or metrics.get("positives") != positives:
                _fail(role, f"metric denominators disagree for {cohort}/{chain}")
            if audit.get("candidate_rows") != n:
                _fail(role, f"score-audit row count disagrees for {cohort}/{chain}")
            ap = _finite(metrics.get("average_precision"), f"{role}/{cohort}/{chain}/AP")
            ci = metrics.get("average_precision_ci95")
            if not isinstance(ci, list) or len(ci) != 2:
                _fail(f"{role}/{cohort}/{chain}/AP CI", "expected a two-element interval")
            ci_low = _finite(ci[0], f"{role}/{cohort}/{chain}/AP CI lower")
            ci_high = _finite(ci[1], f"{role}/{cohort}/{chain}/AP CI upper")
            if not 0.0 <= ci_low <= ap <= ci_high <= 1.0:
                _fail(role, f"AP or AP interval is invalid for {cohort}/{chain}")
            budgets = _exact_keys(metrics.get("budgets"), ("k_50",), f"{role}/{cohort}/{chain}/budgets")
            budget = budgets["k_50"]
            if not isinstance(budget, Mapping) or budget.get("requested_k") != 50:
                _fail(role, f"B1 value budget changed for {cohort}/{chain}")
            value = _finite(
                budget.get("value_capture"), f"{role}/{cohort}/{chain}/value@50"
            )
            if not 0.0 <= ap <= 1.0 or not 0.0 <= value <= 1.0:
                _fail(role, f"metric outside [0,1] for {cohort}/{chain}")
            covered = _integer(
                audit.get("exporter_dictionary_covered_rows"),
                f"{role}/{cohort}/{chain}/dictionary-covered rows",
            )
            _close(
                audit.get("exporter_dictionary_coverage"),
                1.0,
                f"{role}/{cohort}/{chain}/dictionary coverage",
            )
            if covered != n:
                _fail(role, f"dictionary coverage row count changed for {cohort}/{chain}")
            with_membership = _integer(
                audit.get("candidates_with_any_target_hs6_rca_membership"),
                f"{role}/{cohort}/{chain}/membership candidates",
            )
            material = audit.get("self_diagonal_exclusion_material")
            if not isinstance(material, bool) or material != (with_membership > 0):
                _fail(role, f"diagonal materiality audit disagrees for {cohort}/{chain}")
            rows_total += n
            positives_total += positives
            membership_candidates += with_membership
            material_chains += int(material)
            per_chain[chain] = {
                "ap": ap,
                "ap_ci_low": ci_low,
                "ap_ci_high": ci_high,
                "value": value,
            }
        total_score_rows += rows_total

        macro = macros[cohort]
        if not isinstance(macro, Mapping):
            _fail(f"{role}/macro/{cohort}", "expected an object")
        if (
            macro.get("aggregation") != "unweighted_mean_over_six_fixed_chains"
            or macro.get("chain_registry") != list(CHAINS)
            or macro.get("headline_metric") != "average_precision"
            or macro.get("value_metric")
            != "global_observed_late_value_capture_at_50"
            or macro.get("chain_level_ci95") is not None
        ):
            _fail(role, f"macro identity changed for {cohort}")
        headline = macro.get("headline")
        realized = macro.get("realized_value")
        if not isinstance(headline, Mapping) or not isinstance(realized, Mapping):
            _fail(role, f"macro metrics missing for {cohort}")
        headline_per_chain = _exact_keys(
            headline.get("per_chain"), CHAINS, f"{role}/macro/{cohort}/headline"
        )
        value_per_chain = _exact_keys(
            realized.get("per_chain"), CHAINS, f"{role}/macro/{cohort}/value"
        )
        for chain in CHAINS:
            _close(
                headline_per_chain[chain],
                per_chain[chain]["ap"],
                f"{role}/macro/{cohort}/{chain}/AP",
            )
            _close(
                value_per_chain[chain],
                per_chain[chain]["value"],
                f"{role}/macro/{cohort}/{chain}/value",
            )
        ap_mean = statistics.fmean(per_chain[chain]["ap"] for chain in CHAINS)
        value_mean = statistics.fmean(per_chain[chain]["value"] for chain in CHAINS)
        _close(headline.get("macro_mean"), ap_mean, f"{role}/macro/{cohort}/AP mean")
        _close(realized.get("macro_mean"), value_mean, f"{role}/macro/{cohort}/value mean")
        reduced_cohorts[cohort] = {
            "ap_mean": ap_mean,
            "value_mean": value_mean,
            "per_chain": per_chain,
            "rows": rows_total,
            "positives": positives_total,
            "membership_candidates": membership_candidates,
            "material_chains": material_chains,
        }
    if total_score_rows != score_rows:
        _fail(role, "keyed-score row count does not equal both cohort inventories")

    main_matrix = matrix_by_cohort["main"]
    return {
        "bootstrap_draws": bootstrap_draws,
        "score_rows": score_rows,
        "countries": int(main_matrix["countries"]),
        "products": int(main_matrix["products"]),
        "target_products": int(main_matrix["target_products"]),
        "target_diagonal_nonzero_before_exclusion": int(
            main_matrix["target_diagonal_nonzero_before_exclusion"]
        ),
        "target_diagonal_max_after_exclusion": float(
            main_matrix["target_diagonal_max_after_exclusion"]
        ),
        "cohorts": reduced_cohorts,
    }


def _verify_score_robustness_r5_first(
    paths: ArtifactPaths,
    hashes: dict[Path, str],
) -> Mapping[str, Any]:
    """Recompute the complete r5 score-only analysis before using its values."""

    role = "score robustness r5 verifier"
    raw_inputs = (
        paths.score_robustness_r5,
        paths.score_robustness_r5_csv,
        paths.score_robustness_r5_generator,
        paths.score_robustness_r5_config,
    )
    for source in raw_inputs:
        if source.is_symlink() or not source.is_file():
            _fail(role, f"missing, non-regular, or symbolic-link input: {source.name}")
    inputs = tuple(source.resolve() for source in raw_inputs)
    before = {source: _sha256(source) for source in inputs}
    config = _load(paths.score_robustness_r5_config)
    if config.get("output_json") != "results_v2/metrics/v2_score_robustness_r5.json":
        _fail(role, "frozen config output_json is not canonical")
    if config.get("output_csv") != "results_v2/metrics/v2_score_robustness_r5.csv":
        _fail(role, "frozen config output_csv is not canonical")
    module, _ = _load_bound_module(
        paths,
        relative="tools/v2_score_robustness_r5.py",
        configured=paths.score_robustness_r5_generator,
        role=role,
    )
    verify = getattr(module, "verify", None)
    if not callable(verify):
        _fail(role, "canonical verifier has no verify function")
    try:
        verify(paths.score_robustness_r5_config, paths.score_robustness_r5)
    except Exception as exc:
        _fail(role, str(exc))
    payload = _load(paths.score_robustness_r5)
    after = {source: _sha256(source) for source in inputs}
    if after != before:
        _fail(role, "JSON, CSV, config, or verifier changed during verification")
    hashes.update(before)
    return payload


def _validate_score_robustness_r5(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Reduce the fully recomputed r5 diagnostics to stable paper macros."""

    role = "score robustness r5"
    _exact_keys(
        payload,
        ("analysis", "csv_receipt", "generated_at_utc", "provenance", "schema_version", "status"),
        role,
    )
    if payload.get("schema_version") != SCORE_ROBUSTNESS_R5_SCHEMA:
        _fail(f"{role}/schema_version", f"expected {SCORE_ROBUSTNESS_R5_SCHEMA!r}")
    if payload.get("status") != SCORE_ROBUSTNESS_R5_STATUS:
        _fail(f"{role}/status", f"expected {SCORE_ROBUSTNESS_R5_STATUS!r}")

    config = _load(paths.score_robustness_r5_config)
    if config.get("schema_version") != SCORE_ROBUSTNESS_R5_CONFIG_SCHEMA:
        _fail(role, "frozen r5 config schema changed")
    provenance = _exact_keys(
        payload.get("provenance"),
        ("config", "input_file_count", "inputs", "runner"),
        f"{role}/provenance",
    )
    for key, expected_path, source in (
        ("config", "configs/v2_score_robustness_r5.json", paths.score_robustness_r5_config),
        ("runner", "tools/v2_score_robustness_r5.py", paths.score_robustness_r5_generator),
    ):
        receipt = _exact_keys(
            provenance.get(key), ("path", "sha256"), f"{role}/provenance/{key}"
        )
        if receipt.get("path") != expected_path:
            _fail(role, f"{key} provenance path is not canonical")
        _verify_hash(paths, expected_path, receipt.get("sha256"), f"{role}/{key}", hashes)
        if source.resolve() != (paths.root / expected_path).resolve():
            _fail(role, f"configured {key} is not canonical")
    governed_inputs = provenance.get("inputs")
    if not isinstance(governed_inputs, list):
        _fail(f"{role}/provenance/inputs", "expected a list")
    input_file_count = _integer(
        provenance.get("input_file_count"), f"{role}/input file count", minimum=1
    )
    if input_file_count != len(governed_inputs):
        _fail(role, "governed input count disagrees with receipt inventory")

    csv_receipt = _exact_keys(
        payload.get("csv_receipt"), ("columns", "path", "row_count", "sha256"), f"{role}/CSV receipt"
    )
    if csv_receipt.get("path") != "results_v2/metrics/v2_score_robustness_r5.csv":
        _fail(role, "r5 CSV path is not canonical")
    _verify_hash(
        paths,
        csv_receipt["path"],
        csv_receipt.get("sha256"),
        f"{role}/CSV",
        hashes,
    )
    csv_rows = _integer(csv_receipt.get("row_count"), f"{role}/CSV rows", minimum=1)

    analysis = _exact_keys(
        payload.get("analysis"),
        (
            "b1_pooling_sensitivity",
            "budget_summaries",
            "eligibility_threshold_cohort_geometry",
            "paired_family_comparison",
            "protocol",
            "two_stage_b1_b2_diagnostic",
        ),
        f"{role}/analysis",
    )
    protocol = analysis.get("protocol")
    if not isinstance(protocol, Mapping):
        _fail(f"{role}/protocol", "expected an object")
    expected_protocol = {
        "all_declared_sensitivity_variants_reported": True,
        "calibration_performed": False,
        "fine_tuning_performed": False,
        "main_labels_used_for_method_selection": False,
        "paired_direction": "kge_minus_nbfnet",
        "raw_scores_averaged_across_seeds": False,
        "source": "formal frozen main score artifacts only",
        "training_performed": False,
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            _fail(f"{role}/protocol/{field}", f"expected {expected!r}")

    paired_section = analysis["paired_family_comparison"]
    if not isinstance(paired_section, Mapping):
        _fail(f"{role}/paired", "expected an object")
    fixed_rows = paired_section.get("fixed_six_chain")
    per_chain_rows = paired_section.get("per_chain")
    if not isinstance(fixed_rows, list) or len(fixed_rows) != 3:
        _fail(role, "paired fixed-six inventory must contain exactly three tasks")
    if not isinstance(per_chain_rows, list) or len(per_chain_rows) != 18:
        _fail(role, "paired per-chain inventory must contain 18 records")
    expected_metrics = {
        "a": "lane_average_precision",
        "b1": "entry_average_precision_official_raw_max",
        "b2": "positive_entry_macro_recall_at_3",
    }
    paired: dict[str, dict[str, float]] = {}
    for row in fixed_rows:
        if not isinstance(row, Mapping):
            _fail(role, "paired fixed-six row is not an object")
        task = row.get("task")
        if task not in expected_metrics or task in paired:
            _fail(role, f"unexpected or duplicate paired task {task!r}")
        if (
            row.get("metric") != expected_metrics[task]
            or row.get("direction") != "kge_minus_nbfnet"
            or row.get("chains") != list(CHAINS)
            or row.get("seeds") != [0, 1, 2, 3, 4]
            or row.get("chains_resampled") is not False
            or row.get("chain_weighting") != "unweighted"
            or row.get("cluster_resampling") != "stratified_within_each_fixed_chain"
        ):
            _fail(role, f"paired protocol identity changed for {task}")
        draws = _integer(row.get("requested_draws"), f"{role}/paired/{task}/draws", minimum=1)
        finite_draws = _integer(
            row.get("finite_draws"), f"{role}/paired/{task}/finite draws", minimum=1
        )
        if draws != finite_draws:
            _fail(role, f"non-finite paired bootstrap draw for {task}")
        point = _finite(row.get("point"), f"{role}/paired/{task}/point")
        low = _finite(row.get("lower_95"), f"{role}/paired/{task}/lower")
        high = _finite(row.get("upper_95"), f"{role}/paired/{task}/upper")
        if not -1.0 <= low <= point <= high <= 1.0:
            _fail(role, f"paired interval is invalid for {task}")
        chain_points = [
            candidate["fixed_five_seed_mean"]["point"]
            for candidate in per_chain_rows
            if isinstance(candidate, Mapping) and candidate.get("task") == task
        ]
        if len(chain_points) != len(CHAINS):
            _fail(role, f"paired per-chain task inventory changed for {task}")
        _close(point, statistics.fmean(chain_points), f"{role}/paired/{task}/fixed-six point")
        paired[task] = {"point": point, "low": low, "high": high, "draws": draws}
    if set(paired) != set(expected_metrics):
        _fail(role, "paired task inventory is incomplete")

    pooling_section = analysis["b1_pooling_sensitivity"]
    if not isinstance(pooling_section, Mapping):
        _fail(f"{role}/pooling", "expected an object")
    pooling_rows = pooling_section.get("fixed_six_chain")
    expected_pooling = {
        (family, method)
        for family in ("kge", "nbfnet")
        for method in ("official_raw_max", "ecdf_mean", "ecdf_top3_mean")
    }
    pooling: dict[tuple[str, str], dict[str, float]] = {}
    if not isinstance(pooling_rows, list) or len(pooling_rows) != len(expected_pooling):
        _fail(role, "B1 pooling fixed-six inventory changed")
    for row in pooling_rows:
        if not isinstance(row, Mapping):
            _fail(role, "pooling row is not an object")
        key = (row.get("family"), row.get("method"))
        if key not in expected_pooling or key in pooling:
            _fail(role, f"unexpected or duplicate pooling row {key!r}")
        if (
            row.get("chains") != list(CHAINS)
            or row.get("chain_count") != len(CHAINS)
            or row.get("normative") is not (key[1] == "official_raw_max")
        ):
            _fail(role, f"pooling aggregation identity changed for {key!r}")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            _fail(role, f"pooling metrics missing for {key!r}")
        ap = _finite(metrics.get("entry_average_precision"), f"{role}/pooling/{key}/AP")
        value = _finite(
            metrics.get("entry_value_capture_at_50"),
            f"{role}/pooling/{key}/value@50",
        )
        if not 0.0 <= ap <= 1.0 or not 0.0 <= value <= 1.0:
            _fail(role, f"pooling AP/value is outside [0,1] for {key!r}")
        pooling[key] = {"ap": ap, "value": value}

    budget_section = analysis["budget_summaries"]
    if not isinstance(budget_section, Mapping):
        _fail(f"{role}/budgets", "expected an object")
    budget_rows = budget_section.get("fixed_six_chain")
    expected_budget_keys = {
        *( ("a", family, "global_lanes", k) for family in ("kge", "nbfnet") for k in (50, 100, 250, 500, 1000) ),
        *( ("a", family, "per_exporter_lanes", k) for family in ("kge", "nbfnet") for k in (5, 10) ),
        *( ("b1", family, "global_entries", k) for family in ("kge", "nbfnet") for k in (25, 50, 100, 250) ),
        *( ("b2", family, "per_positive_entry_destinations", k) for family in ("kge", "nbfnet") for k in (1, 3, 5) ),
    }
    budget_map: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    if not isinstance(budget_rows, list) or len(budget_rows) != len(expected_budget_keys):
        _fail(role, "budget fixed-six inventory changed")
    for row in budget_rows:
        if not isinstance(row, Mapping):
            _fail(role, "budget row is not an object")
        key = (
            row.get("task"),
            row.get("family"),
            row.get("budget_scope"),
            row.get("requested_k"),
        )
        if key not in expected_budget_keys or key in budget_map:
            _fail(role, f"unexpected or duplicate budget row {key!r}")
        if row.get("chains") != list(CHAINS) or row.get("chain_count") != len(CHAINS):
            _fail(role, f"budget aggregation identity changed for {key!r}")
        if not isinstance(row.get("metrics"), Mapping):
            _fail(role, f"budget metrics missing for {key!r}")
        budget_map[key] = row["metrics"]

    headline_keys = {
        "a": ("global_lanes", 500, "recall", "observed_value_capture"),
        "b1": ("global_entries", 50, "recall", "observed_value_capture"),
        "b2": (
            "per_positive_entry_destinations",
            3,
            "macro_recall",
            "macro_observed_value_capture",
        ),
    }
    budgets: dict[str, dict[str, dict[str, float]]] = {}
    for task, (scope, k, recall_field, value_field) in headline_keys.items():
        budgets[task] = {}
        for family in ("kge", "nbfnet"):
            metrics = budget_map[(task, family, scope, k)]
            recall = _finite(metrics.get(recall_field), f"{role}/budget/{task}/{family}/recall")
            value = _finite(metrics.get(value_field), f"{role}/budget/{task}/{family}/value")
            if not 0.0 <= recall <= 1.0 or not 0.0 <= value <= 1.0:
                _fail(role, f"headline budget metric outside [0,1] for {task}/{family}")
            budgets[task][family] = {"recall": recall, "value": value, "k": k}

    two_stage_section = analysis["two_stage_b1_b2_diagnostic"]
    if not isinstance(two_stage_section, Mapping):
        _fail(f"{role}/two stage", "expected an object")
    two_stage_rows = two_stage_section.get("fixed_six_chain")
    expected_e2e_keys = {
        (family, b1_k, b2_k)
        for family in ("kge", "nbfnet")
        for b1_k in (25, 50, 100, 250)
        for b2_k in (1, 3, 5)
    }
    e2e_map: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    if not isinstance(two_stage_rows, list) or len(two_stage_rows) != len(expected_e2e_keys):
        _fail(role, "two-stage fixed-six inventory changed")
    for row in two_stage_rows:
        if not isinstance(row, Mapping):
            _fail(role, "two-stage row is not an object")
        key = (row.get("family"), row.get("b1_budget"), row.get("b2_budget"))
        if key not in expected_e2e_keys or key in e2e_map:
            _fail(role, f"unexpected or duplicate two-stage row {key!r}")
        if row.get("chains") != list(CHAINS) or row.get("chain_count") != len(CHAINS):
            _fail(role, f"two-stage aggregation identity changed for {key!r}")
        if not isinstance(row.get("metrics"), Mapping):
            _fail(role, f"two-stage metrics missing for {key!r}")
        e2e_map[key] = row["metrics"]
    e2e: dict[str, dict[str, float]] = {}
    for family in ("kge", "nbfnet"):
        metrics = e2e_map[(family, 50, 3)]
        selected = {
            "gate_recall": _finite(
                metrics.get("positive_entry_gate_recall"), f"{role}/e2e/{family}/gate recall"
            ),
            "destination_recall": _finite(
                metrics.get("e2e_macro_destination_recall"),
                f"{role}/e2e/{family}/destination recall",
            ),
            "value_capture": _finite(
                metrics.get("e2e_global_observed_value_capture"),
                f"{role}/e2e/{family}/global value",
            ),
            "macro_destination_value_capture": _finite(
                metrics.get("e2e_macro_destination_observed_value_capture"),
                f"{role}/e2e/{family}/macro destination value",
            ),
            "micro_destination_recall": _finite(
                metrics.get("e2e_micro_destination_recall"),
                f"{role}/e2e/{family}/micro destination recall",
            ),
        }
        if any(not 0.0 <= value <= 1.0 for value in selected.values()):
            _fail(role, f"two-stage metric outside [0,1] for {family}")
        e2e[family] = selected

    geometry = analysis["eligibility_threshold_cohort_geometry"]
    if not isinstance(geometry, Mapping) or (
        geometry.get("status") != "not_computed_from_fixed_scores"
        or geometry.get("thresholds_kusd") != [50.0, 100.0, 250.0]
    ):
        _fail(role, "eligibility-threshold geometry status changed")
    return {
        "input_file_count": input_file_count,
        "csv_rows": csv_rows,
        "paired": paired,
        "pooling": pooling,
        "budgets": budgets,
        "e2e": e2e,
    }


def _verify_eligibility_threshold_geometry_first(
    paths: ArtifactPaths,
    hashes: dict[Path, str],
) -> Mapping[str, Any]:
    """Run the exact threshold-geometry JSON/CSV verifier before reduction."""

    role = "eligibility-threshold geometry verifier"
    raw_inputs = (
        paths.eligibility_threshold_geometry,
        paths.eligibility_threshold_geometry_csv,
        paths.eligibility_threshold_geometry_generator,
        paths.eligibility_threshold_geometry_config,
    )
    for source in raw_inputs:
        if source.is_symlink() or not source.is_file():
            _fail(role, f"missing, non-regular, or symbolic-link input: {source.name}")
    inputs = tuple(source.resolve() for source in raw_inputs)
    before = {source: _sha256(source) for source in inputs}
    module, _ = _load_bound_module(
        paths,
        relative="tools/v2_eligibility_threshold_geometry.py",
        configured=paths.eligibility_threshold_geometry_generator,
        role=role,
    )
    verify = getattr(module, "verify_output", None)
    if not callable(verify):
        _fail(role, "canonical verifier has no verify_output function")
    try:
        verify(
            paths.eligibility_threshold_geometry,
            paths.eligibility_threshold_geometry_csv,
        )
    except Exception as exc:
        _fail(role, str(exc))
    payload = _load(paths.eligibility_threshold_geometry)
    after = {source: _sha256(source) for source in inputs}
    if after != before:
        _fail(role, "JSON, CSV, config, or verifier changed during verification")
    hashes.update(before)
    return payload


def _validate_eligibility_threshold_geometry(
    paths: ArtifactPaths,
    payload: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    """Validate exact cohort geometry and expose aggregate retention numbers."""

    role = "eligibility-threshold geometry"
    _exact_keys(
        payload,
        (
            "canonical_100kusd_gate",
            "config",
            "generated_at_utc",
            "inputs",
            "protocol",
            "runtime",
            "schema_version",
            "scope",
            "status",
            "thresholds",
        ),
        role,
    )
    for field, expected in (
        ("schema_version", ELIGIBILITY_THRESHOLD_GEOMETRY_SCHEMA),
        ("status", ELIGIBILITY_THRESHOLD_GEOMETRY_STATUS),
        ("scope", ELIGIBILITY_THRESHOLD_GEOMETRY_SCOPE),
    ):
        if payload.get(field) != expected:
            _fail(f"{role}/{field}", f"expected {expected!r}")

    config = _load(paths.eligibility_threshold_geometry_config)
    if (
        config.get("schema_version") != ELIGIBILITY_THRESHOLD_GEOMETRY_CONFIG_SCHEMA
        or config.get("status") != "frozen_before_threshold_geometry_rebuild"
        or config.get("thresholds_kusd") != [50, 100, 250]
        or config.get("reference_threshold_kusd") != 100.0
        or config.get("chains") != list(CHAINS)
        or config.get("scope") != ELIGIBILITY_THRESHOLD_GEOMETRY_SCOPE
    ):
        _fail(role, "frozen threshold-geometry config identity changed")
    _exact_keys(config.get("tasks"), ("a", "b1", "b2"), f"{role}/config/tasks")
    config_receipt = _exact_keys(
        payload.get("config"), ("path", "sha256"), f"{role}/config receipt"
    )
    if config_receipt.get("path") != "configs/v2_eligibility_threshold_geometry.json":
        _fail(role, "threshold-geometry config path is not canonical")
    _verify_hash(
        paths,
        config_receipt["path"],
        config_receipt.get("sha256"),
        f"{role}/config",
        hashes,
    )
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("aggregation")
        != "stage_by_year_sum_then_mean_over_all_five_calendar_years"
        or protocol.get("comparison") != "strictly_greater_than_threshold"
        or protocol.get("early_window") != "2008-2012"
        or protocol.get("late_window") != "2018-2022"
        or protocol.get("thresholds_kusd") != [50, 100, 250]
        or protocol.get("reference_threshold_kusd") != 100
        or protocol.get("model_scores_or_performance_computed") is not False
    ):
        _fail(role, "threshold-geometry protocol identity changed")

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        _fail(f"{role}/inputs", "expected an object")
    source_hashes = _exact_keys(
        inputs.get("source_hashes"),
        (
            "configs/v2_eligibility_threshold_geometry.json",
            "tools/v2_eligibility_threshold_geometry.py",
            "tools/v2_product_space_density.py",
        ),
        f"{role}/source hashes",
    )
    for relative, digest in source_hashes.items():
        _verify_hash(paths, relative, digest, f"{role}/source/{relative}", hashes)
    if source_hashes["configs/v2_eligibility_threshold_geometry.json"] != config_receipt.get(
        "sha256"
    ):
        _fail(role, "config and source-hash receipts disagree")

    gate = _exact_keys(
        payload.get("canonical_100kusd_gate"),
        ("chains", "requirement", "status"),
        f"{role}/100k gate",
    )
    if gate.get("status") != "PASS" or gate.get("requirement") != config.get(
        "canonical_gate"
    ):
        _fail(role, "canonical 100-kUSD gate did not pass exactly")
    gate_chains = _exact_keys(gate.get("chains"), CHAINS, f"{role}/100k gate/chains")
    for chain in CHAINS:
        gate_tasks = _exact_keys(
            gate_chains[chain], ("a", "b1", "b2", "b_lanes"), f"{role}/100k gate/{chain}"
        )
        for task, record in gate_tasks.items():
            if not isinstance(record, Mapping) or (
                record.get("candidate_keys_exact") is not True
                or record.get("positive_keys_exact") is not True
            ):
                _fail(role, f"100-kUSD exact-key gate failed for {chain}/{task}")

    threshold_payloads = _exact_keys(
        payload.get("thresholds"), ("50", "100", "250"), f"{role}/thresholds"
    )
    task_ids = ("a", "b1", "b2")
    reduced: dict[int, dict[str, dict[str, float | int]]] = {}
    reference_by_chain: dict[tuple[str, str], tuple[int, int]] = {}
    for chain in CHAINS:
        hundred_chains = threshold_payloads["100"].get("chains")
        if not isinstance(hundred_chains, Mapping) or chain not in hundred_chains:
            _fail(role, f"100-kUSD chain inventory missing {chain}")
        for task in task_ids:
            record = hundred_chains[chain].get(task)
            if not isinstance(record, Mapping):
                _fail(role, f"100-kUSD record missing {chain}/{task}")
            reference_by_chain[(chain, task)] = (
                _integer(record.get("candidate_count"), f"{role}/100/{chain}/{task}/candidates"),
                _integer(record.get("positive_count"), f"{role}/100/{chain}/{task}/positives"),
            )
            gate_record = gate_chains[chain][task]
            if (
                gate_record.get("candidate_count") != reference_by_chain[(chain, task)][0]
                or gate_record.get("positive_count") != reference_by_chain[(chain, task)][1]
            ):
                _fail(role, f"100-kUSD geometry disagrees with exact gate for {chain}/{task}")

    overlap_keys = (
        "intersection",
        "jaccard",
        "left_only",
        "reference_only",
        "retention_vs_reference",
        "union",
    )
    record_keys = (
        "base_rate",
        "candidate_count",
        "candidate_key_sha256",
        "candidate_overlap_vs_100",
        "positive_count",
        "positive_key_sha256",
        "positive_overlap_vs_100",
    )
    summary_keys = (
        "candidate_count",
        "candidate_jaccard_vs_100",
        "candidate_retention_vs_100",
        "positive_count",
        "positive_jaccard_vs_100",
        "positive_retention_vs_100",
    )
    for threshold in (50, 100, 250):
        threshold_record = _exact_keys(
            threshold_payloads[str(threshold)], ("chains", "summary"), f"{role}/{threshold}"
        )
        chains = _exact_keys(
            threshold_record.get("chains"), CHAINS, f"{role}/{threshold}/chains"
        )
        summaries = _exact_keys(
            threshold_record.get("summary"), task_ids, f"{role}/{threshold}/summary"
        )
        reduced[threshold] = {}
        for task in task_ids:
            candidate_count = 0
            positive_count = 0
            candidate_intersection = 0
            positive_intersection = 0
            candidate_union = 0
            positive_union = 0
            reference_candidates = 0
            reference_positives = 0
            for chain in CHAINS:
                tasks = _exact_keys(
                    chains[chain], task_ids, f"{role}/{threshold}/{chain}/tasks"
                )
                record = _exact_keys(
                    tasks[task], record_keys, f"{role}/{threshold}/{chain}/{task}"
                )
                candidates = _integer(
                    record.get("candidate_count"),
                    f"{role}/{threshold}/{chain}/{task}/candidates",
                )
                positives = _integer(
                    record.get("positive_count"),
                    f"{role}/{threshold}/{chain}/{task}/positives",
                )
                if positives > candidates:
                    _fail(role, f"positives exceed candidates for {threshold}/{chain}/{task}")
                expected_base = positives / candidates if candidates else 0.0
                _close(
                    record.get("base_rate"),
                    expected_base,
                    f"{role}/{threshold}/{chain}/{task}/base rate",
                )
                reference_candidate, reference_positive = reference_by_chain[(chain, task)]
                overlap_values: dict[str, Mapping[str, Any]] = {}
                for kind, current, reference in (
                    ("candidate", candidates, reference_candidate),
                    ("positive", positives, reference_positive),
                ):
                    overlap = _exact_keys(
                        record.get(f"{kind}_overlap_vs_100"),
                        overlap_keys,
                        f"{role}/{threshold}/{chain}/{task}/{kind} overlap",
                    )
                    intersection = _integer(overlap.get("intersection"), f"{role}/intersection")
                    left_only = _integer(overlap.get("left_only"), f"{role}/left only")
                    reference_only = _integer(
                        overlap.get("reference_only"), f"{role}/reference only"
                    )
                    union = _integer(overlap.get("union"), f"{role}/union")
                    if (
                        intersection + left_only != current
                        or intersection + reference_only != reference
                        or union != intersection + left_only + reference_only
                    ):
                        _fail(role, f"overlap accounting failed for {threshold}/{chain}/{task}/{kind}")
                    _close(
                        overlap.get("retention_vs_reference"),
                        intersection / reference if reference else 1.0,
                        f"{role}/{threshold}/{chain}/{task}/{kind} retention",
                    )
                    _close(
                        overlap.get("jaccard"),
                        intersection / union if union else 1.0,
                        f"{role}/{threshold}/{chain}/{task}/{kind} jaccard",
                    )
                    overlap_values[kind] = overlap
                candidate_count += candidates
                positive_count += positives
                candidate_intersection += int(overlap_values["candidate"]["intersection"])
                positive_intersection += int(overlap_values["positive"]["intersection"])
                candidate_union += int(overlap_values["candidate"]["union"])
                positive_union += int(overlap_values["positive"]["union"])
                reference_candidates += reference_candidate
                reference_positives += reference_positive
            summary = _exact_keys(
                summaries[task], summary_keys, f"{role}/{threshold}/summary/{task}"
            )
            candidate_retention = (
                candidate_intersection / reference_candidates if reference_candidates else 1.0
            )
            positive_retention = (
                positive_intersection / reference_positives if reference_positives else 1.0
            )
            for field, expected in (
                ("candidate_count", candidate_count),
                ("positive_count", positive_count),
                ("candidate_retention_vs_100", candidate_retention),
                ("positive_retention_vs_100", positive_retention),
                (
                    "candidate_jaccard_vs_100",
                    candidate_intersection / candidate_union if candidate_union else 1.0,
                ),
                (
                    "positive_jaccard_vs_100",
                    positive_intersection / positive_union if positive_union else 1.0,
                ),
            ):
                _close(summary.get(field), expected, f"{role}/{threshold}/{task}/{field}")
            reduced[threshold][task] = {
                "candidates": candidate_count,
                "positives": positive_count,
                "candidate_retention": candidate_retention,
                "positive_retention": positive_retention,
            }
    return {"exact_100_gate_pass": True, "thresholds": reduced}


def _robustness_macro_mean(
    robustness: Mapping[str, Any],
    *, sensitivity: str,
    slice_name: str,
    track: str,
    model: str,
    primary_metric: str,
    expected_chains: int,
) -> float:
    row = _robustness_macro_row(
        robustness,
        sensitivity=sensitivity,
        slice_name=slice_name,
        track=track,
        model=model,
        metric=primary_metric,
        expected_defined=expected_chains,
    )
    return float(row["unweighted_chain_macro_mean"])


def _add_cpu_numbers(numbers: dict[str, str], rolling: Mapping[str, Any]) -> None:
    macros = rolling["macro_summary"]
    for track, spec in TRACK_SPECS.items():
        track_payload = macros[track]
        model_tex = {model: tag for model, tag, _, _ in spec["models"]}
        metric_tex = {model: metric for model, _, metric, _ in spec["models"]}
        value_tex = {model: value for model, _, _, value in spec["models"]}
        for model, tag, metric_name, value_name in spec["models"]:
            payload = track_payload["models"][model]
            numbers[f"VTwo{spec['tex']}{tag}{metric_name}"] = _decimal(payload["macro_mean"])
            numbers[f"VTwo{spec['tex']}{tag}{value_name}"] = _decimal(payload["realized_value"]["macro_mean"])
            for chain in CHAINS:
                chain_tag = CHAIN_TEX[chain]
                numbers[f"VTwo{chain_tag}{spec['tex']}{tag}{metric_name}"] = _decimal(payload["per_chain"][chain])
                numbers[f"VTwo{chain_tag}{spec['tex']}{tag}{value_name}"] = _decimal(payload["realized_value"]["per_chain"][chain])
        for comparison in track_payload["pairwise_deltas"]["comparisons"]:
            left = comparison["left_model"]
            right = comparison["right_model"]
            pair = f"{model_tex[left]}Minus{model_tex[right]}"
            headline = comparison["headline"]
            value = comparison["realized_value"]
            numbers[f"VTwo{spec['tex']}{pair}{metric_tex[left]}DeltaMean"] = _decimal(headline["descriptive_mean_delta"])
            numbers[f"VTwo{spec['tex']}{pair}{metric_tex[left]}DeltaMedian"] = _decimal(headline["descriptive_median_delta"])
            numbers[f"VTwo{spec['tex']}{pair}{value_tex[left]}DeltaMean"] = _decimal(value["descriptive_mean_delta"])
            numbers[f"VTwo{spec['tex']}{pair}{value_tex[left]}DeltaMedian"] = _decimal(value["descriptive_median_delta"])
            for sign, suffix in (("left_better", "LeftBetter"), ("ties", "Ties"), ("right_better", "RightBetter")):
                numbers[f"VTwo{spec['tex']}{pair}{metric_tex[left]}{suffix}Chains"] = _commas(headline["sign_counts"][sign])
                numbers[f"VTwo{spec['tex']}{pair}{value_tex[left]}{suffix}Chains"] = _commas(value["sign_counts"][sign])
            for chain in CHAINS:
                chain_tag = CHAIN_TEX[chain]
                numbers[f"VTwo{chain_tag}{spec['tex']}{pair}{metric_tex[left]}Delta"] = _decimal(headline["per_chain"][chain])
                numbers[f"VTwo{chain_tag}{spec['tex']}{pair}{value_tex[left]}Delta"] = _decimal(value["per_chain"][chain])


def _add_registry_numbers(numbers: dict[str, str], registry: Mapping[str, Any]) -> None:
    summary = registry["summary"]
    checks_passed = registry["purity_checks_passed"]
    checks_total = registry["purity_checks_total"]
    numbers.update(
        {
            "VTwoRegistryChainCount": _commas(summary["chain_count"]),
            "VTwoRegistryActiveStages": _commas(summary["active_stages"]),
            "VTwoRegistryTargetStages": _commas(registry["target_stages"]),
            "VTwoRegistrySemanticStageGates": _commas(
                registry["semantic_stage_gates"]
            ),
            "VTwoRegistryIncludedCodes": _commas(summary["included_codes"]),
            "VTwoRegistryExcludedCodes": _commas(summary["excluded_codes"]),
            "VTwoRegistryReviewedCodes": _commas(summary["reviewed_codes"]),
            "VTwoRegistryReassignedCodes": _commas(summary["reassigned_included_codes"]),
            "VTwoRegistryReviewedCodeCoverage": _percent(1.0),
            "VTwoRegistryRetainedCodeShare": _percent(summary["included_codes"] / summary["reviewed_codes"]),
            "VTwoRegistryExcludedCodeShare": _percent(summary["excluded_codes"] / summary["reviewed_codes"]),
            "VTwoRegistryPurityChecksPassed": _commas(checks_passed),
            "VTwoRegistryPurityChecksTotal": _commas(checks_total),
            "VTwoRegistryPurityCheckPassRate": _percent(checks_passed / checks_total),
        }
    )
    for chain, payload in registry["per_chain"].items():
        tag = CHAIN_TEX[chain]
        numbers[f"VTwoRegistry{tag}ActiveCodes"] = _commas(payload["active"])
        numbers[f"VTwoRegistry{tag}ExcludedCodes"] = _commas(payload["removed"])
        numbers[f"VTwoRegistry{tag}ActiveStages"] = _commas(
            payload["active_stages"]
        )
        numbers[f"VTwoRegistry{tag}TargetStages"] = _commas(
            payload["target_stages"]
        )
        numbers[f"VTwoRegistry{tag}ReassignedCodes"] = _commas(
            payload["reassigned"]
        )
        numbers[f"VTwoRegistry{tag}RetainedCodeShare"] = _percent(payload["active"] / payload["reviewed"])


def _add_coverage_numbers(numbers: dict[str, str], coverage: Mapping[str, Any]) -> None:
    field_names = {
        "n_all_realized_entries": ("AllRealizedEntries", _commas),
        "n_covered_realized_entries": ("CoveredRealizedEntries", _commas),
        "n_all_late_start_lanes": ("AllLateStartLanes", _commas),
        "n_eligible_market_late_start_lanes": ("EligibleMarketLateStartLanes", _commas),
        "n_previously_inactive_market_late_start_lanes": ("InactiveMarketLateStartLanes", _commas),
        "realized_entry_coverage": ("RealizedEntryCoverage", _percent),
        "late_start_lane_coverage": ("LateStartLaneCoverage", _percent),
        "late_start_value_coverage": ("LateStartValueCoverage", _percent),
        "previously_inactive_market_lane_share": ("InactiveMarketLaneShare", _percent),
        "previously_inactive_market_value_share": ("InactiveMarketValueShare", _percent),
    }
    for snapshot, snapshot_tag in (("fold2", "FoldTwo"), ("main", "Main")):
        payload = coverage["snapshots"][snapshot]
        for field, (suffix, formatter) in field_names.items():
            numbers[f"VTwoBOneCoverage{snapshot_tag}{suffix}"] = formatter(payload["totals"][field])
        for row in payload["chains"]:
            chain_tag = CHAIN_TEX[row["chain"]]
            for field in (
                "realized_entry_coverage",
                "late_start_lane_coverage",
                "late_start_value_coverage",
                "previously_inactive_market_lane_share",
                "previously_inactive_market_value_share",
            ):
                suffix = field_names[field][0]
                numbers[f"VTwoBOneCoverage{snapshot_tag}{chain_tag}{suffix}"] = _percent(row["totals"][field])


def _add_robustness_numbers(numbers: dict[str, str], robustness: Mapping[str, Any]) -> None:
    def ap(sensitivity: str, slice_name: str, track: str) -> float:
        model = {"A": "historical_logistic_size_gravity", "B1": "historical_logistic_structural"}[track]
        return _robustness_macro_mean(
            robustness,
            sensitivity=sensitivity,
            slice_name=slice_name,
            track=track,
            model=model,
            primary_metric="average_precision",
            expected_chains=6,
        )

    prespec = robustness["prespecification"]
    thresholds = prespec["thresholds_kusd"]
    numbers.update(
        {
            "VTwoRobustChainCount": _commas(len(CHAINS)),
            "VTwoRobustBootstrapDraws": _commas(prespec["uncertainty"]["draws"]),
            "VTwoRobustThresholdFiftyKUSD": _commas(thresholds[0]),
            "VTwoRobustThresholdHundredKUSD": _commas(thresholds[1]),
            "VTwoRobustThresholdTwoFiftyKUSD": _commas(thresholds[2]),
            "VTwoRobustPersistenceMinimumYears": _commas(prespec["persistence"]["minimum_active_years"]),
            "VTwoRobustPersistenceWindowYears": _commas(len(prespec["late_years"])),
            "VTwoRobustAExporterSeenAP": _decimal(ap("identity", "exporter_seen", "A")),
            "VTwoRobustAExporterUnseenAP": _decimal(ap("identity", "exporter_unseen", "A")),
            "VTwoRobustBOneExporterSeenAP": _decimal(ap("identity", "exporter_seen", "B1")),
            "VTwoRobustBOneExporterUnseenAP": _decimal(ap("identity", "exporter_unseen", "B1")),
            "VTwoRobustAHubExcludedAP": _decimal(ap("entity_exclusion", "exclude_hubs", "A")),
            "VTwoRobustBOneHubExcludedAP": _decimal(ap("entity_exclusion", "exclude_hubs", "B1")),
        }
    )
    threshold_names = (("50", "Fifty"), ("100", "Hundred"), ("250", "TwoFifty"))
    for value, tag in threshold_names:
        numbers[f"VTwoRobustAThreshold{tag}AP"] = _decimal(ap("threshold_outcome_only", f"threshold_{value}_kusd", "A"))
        numbers[f"VTwoRobustBOneThreshold{tag}AP"] = _decimal(ap("threshold_outcome_only", f"threshold_{value}_kusd", "B1"))
    persistence_slice = "active_at_least_3_of_5_years_above_100_kusd"
    numbers["VTwoRobustAPersistenceThreeOfFiveAP"] = _decimal(ap("persistence", persistence_slice, "A"))
    numbers["VTwoRobustBOnePersistenceThreeOfFiveAP"] = _decimal(ap("persistence", persistence_slice, "B1"))

    b2_model = "historical_logistic_demand_gravity"
    b2 = _robustness_macro_row(
        robustness,
        sensitivity="identity",
        slice_name="importer_unseen",
        track="B2",
        model=b2_model,
        metric="macro_recall_at_3",
        expected_defined=5,
    )
    group_totals = {
        "before": 0,
        "dropped": 0,
        "after": 0,
        "candidate_lanes": 0,
        "positive_lanes": 0,
    }
    defined_chains = 0
    for chain in CHAINS:
        models = robustness["chains"][chain]["sensitivities"]["identity"]["importer_unseen"]["tracks"]["B2"]["models"]
        reference: tuple[Any, ...] | None = None
        for model in ("processed_importer_demand", "gravity", "historical_logistic_demand_gravity"):
            payload = models[model]
            observed = (
                payload.get("status"),
                payload.get("entry_groups_before_entry_reconditioning"),
                payload.get("dropped_zero_positive_entry_groups"),
                payload.get("entry_groups_after_entry_reconditioning"),
                payload.get("n_entry_groups"),
                payload.get("n_candidate_lanes"),
                payload.get("positive_lanes"),
            )
            if reference is None:
                reference = observed
            elif observed != reference:
                _fail("B2 importer-unseen groups", f"{chain} model payloads disagree on conditioning cohort")
        assert reference is not None
        status, before, dropped, after, n_groups, candidate_lanes, positive_lanes = reference
        before = _integer(before, f"B2 importer-unseen/{chain}/before")
        dropped = _integer(dropped, f"B2 importer-unseen/{chain}/dropped")
        after = _integer(after, f"B2 importer-unseen/{chain}/after")
        if before - dropped != after or after != n_groups:
            _fail("B2 importer-unseen groups", f"{chain} before/dropped/after groups do not reconcile")
        if (status == "complete") != (after > 0) or status not in ("complete", "no_positive_entry_groups"):
            _fail("B2 importer-unseen groups", f"{chain} status does not match positive-entry groups")
        if status == "complete":
            defined_chains += 1
            observed_metric = models[b2_model]["at_k"]["k_3"]["macro_recall"]
            _close(observed_metric, b2["per_chain"][chain], f"B2 importer-unseen/{chain}/macro recall")
        elif b2["per_chain"][chain] is not None:
            _fail("B2 importer-unseen groups", f"{chain} undefined cohort has a macro value")
        group_totals["before"] += before
        group_totals["dropped"] += dropped
        group_totals["after"] += after
        group_totals["candidate_lanes"] += _integer(candidate_lanes, f"B2 importer-unseen/{chain}/lanes")
        group_totals["positive_lanes"] += _integer(positive_lanes, f"B2 importer-unseen/{chain}/positives")
    if defined_chains != 5:
        _fail("B2 importer-unseen groups", "expected five chains with conditioned positive-entry groups")
    numbers.update(
        {
            "VTwoRobustBTwoImporterUnseenRecallThree": _decimal(b2["unweighted_chain_macro_mean"]),
            "VTwoRobustBTwoImporterUnseenChains": _commas(defined_chains),
            "VTwoRobustBTwoImporterUnseenPositiveEntryGroups": _commas(group_totals["after"]),
            # Backward-compatible alias, now explicitly computed after exact-slice reconditioning.
            "VTwoRobustBTwoImporterUnseenEntryGroups": _commas(group_totals["after"]),
            "VTwoRobustBTwoImporterUnseenEntryGroupsBeforeConditioning": _commas(group_totals["before"]),
            "VTwoRobustBTwoImporterUnseenDroppedZeroPositiveEntryGroups": _commas(group_totals["dropped"]),
            "VTwoRobustBTwoImporterUnseenCandidateLanes": _commas(group_totals["candidate_lanes"]),
            "VTwoRobustBTwoImporterUnseenPositiveLanes": _commas(group_totals["positive_lanes"]),
        }
    )


def _add_gpu_numbers(
    numbers: dict[str, str],
    gpu: Mapping[str, Any] | None,
    summary_by_chain: Mapping[str, Mapping[str, Any]],
) -> None:
    if gpu is None:
        numbers["VTwoGPUStatus"] = "PENDING"
        return
    numbers.update(
        {
            "VTwoGPUStatus": "COMPLETE",
            "VTwoGPUSeedCount": _commas(len(gpu["seeds"])),
            "VTwoGPUCompleteChainFamilyJobs": _commas(gpu["complete_chain_family_jobs"]),
            "VTwoGPUCompleteTaskEvaluations": _commas(gpu["complete_task_evaluations"]),
        }
    )
    track_tags = {"a": "TrackA", "b1": "TrackBOne", "b2": "TrackBTwo"}
    family_tags = {"kge": "KGE", "nbfnet": "NBFNet"}
    for row in gpu["macro_summary"]:
        prefix = f"VTwoGPU{track_tags[row['track']]}{family_tags[row['family']]}"
        numbers[f"{prefix}Mean"] = _decimal(row["mean_across_six_chain_means"])
        numbers[f"{prefix}StdAcrossChains"] = _decimal(row["std_across_six_chain_means"])
    for row in gpu["records"]:
        prefix = f"VTwo{CHAIN_TEX[row['chain']]}GPU{track_tags[row['track']]}{family_tags[row['family']]}"
        numbers[f"{prefix}Mean"] = _decimal(row["primary_mean"])
        numbers[f"{prefix}StdAcrossSeeds"] = _decimal(
            row["primary_std_across_seeds"]
        )

    record_map = {
        (row["chain"], row["track"], row["family"]): row["primary_mean"]
        for row in gpu["records"]
    }
    weight_fields = {
        "a": "track_a_candidates",
        "b1": "track_b_unique_entries",
        "b2": "track_b_positive_entries",
    }
    decision_track_tags = {"a": "A", "b1": "BOne", "b2": "BTwo"}
    for track, weight_field in weight_fields.items():
        weights = {
            chain: _integer(
                summary_by_chain[chain][weight_field],
                f"decision-weighted GPU/{track}/{chain}/{weight_field}",
                minimum=1,
            )
            for chain in CHAINS
        }
        denominator = sum(weights.values())
        for family, family_tag in family_tags.items():
            weighted_mean = sum(
                _finite(
                    record_map[(chain, track, family)],
                    f"decision-weighted GPU/{track}/{family}/{chain}",
                )
                * weights[chain]
                for chain in CHAINS
            ) / denominator
            numbers[
                f"VTwoDecisionWeighted{decision_track_tags[track]}{family_tag}"
            ] = _decimal(weighted_mean)


def _add_value_diagnostic_numbers(
    numbers: dict[str, str], value: Mapping[str, Any]
) -> None:
    """Expose parallel family/oracle diagnostics without selecting a winner."""
    for track, spec in VALUE_TRACK_SPECS.items():
        prefix = f"VTwoValueDiagnostic{spec['tex']}"
        suffix = spec["suffix"]
        for family, family_tex in VALUE_FAMILY_TEX.items():
            numbers[f"{prefix}{family_tex}SixChainMean{suffix}"] = _decimal(
                value["macro"][track][family]
            )
        numbers[f"{prefix}OracleSixChainMean{suffix}"] = _decimal(
            value["macro"][track]["oracle"]
        )
        pooled_suffix = (
            "PooledValueCaptureThree" if track == "b2" else f"Pooled{suffix}"
        )
        numbers[f"{prefix}Oracle{pooled_suffix}"] = _decimal(
            value["pooled_oracle"][track]
        )

    accounting = value["accounting"]
    numbers.update(
        {
            "VTwoValueDiagnosticUniqueObservedValueB": _billions(
                accounting["unique_project_observed_late_value_kusd"], 4
            ),
            "VTwoValueDiagnosticBTwoNestedObservedValueB": _billions(
                accounting["track_b2_nested_same_dollars_kusd"], 4
            ),
            "VTwoValueDiagnosticBOneBTwoObservedValueDifferenceKUSD": _decimal(
                accounting["track_b1_observed_late_value_kusd"]
                - accounting["track_b2_nested_same_dollars_kusd"]
            ),
            "VTwoValueDiagnosticBTwoExcludedFromUniqueSum": (
                "TRUE" if accounting["b2_excluded_from_unique_sum"] else "FALSE"
            ),
            "VTwoValueDiagnosticTargetLabelsUsedForSelection": (
                "TRUE" if value["target_labels_used_for_selection"] else "FALSE"
            ),
            "VTwoValueDiagnosticPostHocMainChampionSelected": (
                "TRUE" if value["post_hoc_main_champion_selected"] else "FALSE"
            ),
        }
    )


def _add_loco_numbers(
    numbers: dict[str, str],
    loco: Mapping[str, Any] | None,
) -> None:
    """Expose fixed descriptive LOCO means and population standard deviations.

    ``PopulationStd`` is deliberately explicit in every macro name.  It is the
    descriptive population standard deviation over the fixed 30 chain/seed
    components or matched pairs; this interface exposes no CI, standard error,
    significance test, or population-inference quantity.
    """

    if loco is None:
        numbers["VTwoLOCOStatus"] = "PENDING"
        return
    numbers.update(
        {
            "VTwoLOCOStatus": "COMPLETE",
            "VTwoLOCOChainCount": _commas(loco["chain_count"]),
            "VTwoLOCOSeedCount": _commas(loco["seed_count"]),
            "VTwoLOCOVerifiedComponentCount": _commas(
                loco["verified_component_count"]
            ),
            "VTwoLOCOMatchedPairCountPerMetric": _commas(
                loco["matched_pair_count_per_metric"]
            ),
        }
    )
    for metric, track_tex in LOCO_METRIC_SPECS:
        prefix = f"VTwoLOCO{track_tex}"
        values = loco["metrics"][metric]
        for mode, mode_tex in (("in_domain", "InDomain"), ("loco", "LOCO")):
            stat = values["by_mode"][mode]
            numbers[f"{prefix}{mode_tex}Mean"] = _decimal(stat["mean"])
            numbers[f"{prefix}{mode_tex}PopulationStd"] = _decimal(
                stat["population_std"]
            )
        gap = values["matched_gap"]
        numbers[f"{prefix}MatchedGapMean"] = _decimal(gap["mean"])
        numbers[f"{prefix}MatchedGapPopulationStd"] = _decimal(
            gap["population_std"]
        )


def _add_ultra_numbers(
    numbers: dict[str, str],
    ultra: Mapping[str, Any] | None,
) -> None:
    """Expose the external-pretrained zero-shot reference without seed claims."""

    if ultra is None:
        numbers["VTwoULTRAStatus"] = "PENDING"
        return

    def truth(value: Any) -> str:
        return "TRUE" if value is True else "FALSE"

    numbers.update(
        {
            "VTwoULTRAStatus": "COMPLETE",
            "VTwoULTRAChainCount": _commas(ultra["chain_count"]),
            "VTwoULTRATaskCount": _commas(ultra["task_count"]),
            "VTwoULTRAHeadlineRecordCount": _commas(
                ultra["headline_record_count"]
            ),
            "VTwoULTRACheckpointCount": _commas(ultra["checkpoint_count"]),
            "VTwoULTRATargetLabelsUsedForTrainingSelection": truth(
                ultra["target_labels_used_for_training_or_selection"]
            ),
            "VTwoULTRATargetEarlyGraphUsed": truth(
                ultra["target_early_graph_used"]
            ),
            "VTwoULTRACheckpointTrainingSeedDisclosed": truth(
                ultra["checkpoint_training_seed_disclosed"]
            ),
            "VTwoULTRARepeatabilityScoreGatePass": truth(
                ultra["repeatability_score_gate_pass"]
            ),
            "VTwoULTRARepeatabilityMetricGatePass": truth(
                ultra["repeatability_metric_gate_pass"]
            ),
            "VTwoULTRAAbstractMentionEligible": truth(
                ultra["abstract_rule"]["abstract_should_mention_ultra"]
            ),
        }
    )

    for row in ultra["metric_records"]:
        spec = ULTRA_TASK_SPECS[row["task"]]
        prefix = f"VTwo{CHAIN_TEX[row['chain']]}ULTRA{spec['tex']}"
        numbers[f"{prefix}{spec['headline_tex']}"] = _decimal(
            row["headline_value"]
        )

    for task, spec in ULTRA_TASK_SPECS.items():
        summary = ultra["task_summaries"][task]
        prefix = f"VTwoULTRA{spec['tex']}"
        numbers[f"{prefix}{spec['headline_tex']}SixChainMean"] = _decimal(
            summary["unweighted_six_chain_headline_mean"]
        )
        numbers[f"{prefix}{spec['value_tex']}SixChainMean"] = _decimal(
            summary["unweighted_six_chain_value_mean"]
        )

        for family, family_tex in (("kge", "KGE"), ("nbfnet", "NBFNet")):
            counts = ultra["reference_comparisons"][task][family]["counts"]
            for relation, relation_tex in (
                ("higher", "Higher"),
                ("equal", "Equal"),
                ("lower", "Lower"),
            ):
                numbers[
                    f"VTwoULTRA{spec['tex']}Vs{family_tex}{relation_tex}Chains"
                ] = _commas(counts[relation])

        rule = ultra["abstract_rule"]["tasks"][task]
        numbers[f"VTwoULTRA{spec['tex']}AbstractSameSideBothChains"] = _commas(
            rule["same_side_of_both_chain_count"]
        )
        numbers[f"VTwoULTRA{spec['tex']}AbstractEligible"] = truth(
            rule["eligible_for_abstract_mention"]
        )


def _add_gbdt_numbers(
    numbers: dict[str, str],
    gbdt: Mapping[str, Any] | None,
) -> None:
    if gbdt is None:
        numbers["VTwoGBDTStatus"] = "PENDING"
        return

    numbers.update(
        {
            "VTwoGBDTStatus": "COMPLETE",
            "VTwoGBDTChainCount": _commas(gbdt["chain_count"]),
            "VTwoGBDTTaskCount": _commas(gbdt["task_count"]),
            "VTwoGBDTHeadlineRecordCount": _commas(gbdt["headline_record_count"]),
            "VTwoGBDTCPUModel": str(gbdt["cpu_model"]),
            "VTwoGBDTWallSeconds": _decimal(gbdt["wall_elapsed_seconds"]),
            "VTwoGBDTLogicalCores": _commas(gbdt["logical_cpu_cores"]),
            "VTwoGBDTFitCountUpperBound": _commas(gbdt["fit_count_upper_bound"]),
            "VTwoGBDTBootstrapDraws": _commas(gbdt["bootstrap_draws"]),
        }
    )
    for row in gbdt["headline_records"]:
        spec = GBDT_TRACK_SPECS[row["track"]]
        prefix = f"VTwo{CHAIN_TEX[row['chain']]}GBDT{spec['tex']}{spec['headline_tex']}"
        numbers[prefix] = _decimal(row["headline"])
        numbers[f"{prefix}CILower"] = _decimal(row["headline_ci95_low"])
        numbers[f"{prefix}CIUpper"] = _decimal(row["headline_ci95_high"])

    for track, spec in GBDT_TRACK_SPECS.items():
        summary = gbdt["task_summaries"][track]
        prefix = f"VTwoGBDT{spec['tex']}"
        numbers[f"{prefix}{spec['headline_tex']}SixChainMean"] = _decimal(
            summary["headline_mean"]
        )
        numbers[f"{prefix}{spec['value_tex']}SixChainMean"] = _decimal(
            summary["value_mean"]
        )


def _add_product_space_numbers(
    numbers: dict[str, str],
    product_space: Mapping[str, Any],
) -> None:
    numbers.update(
        {
            "VTwoProductSpaceStatus": "COMPLETE",
            "VTwoProductSpaceBootstrapDraws": _commas(
                product_space["bootstrap_draws"]
            ),
            "VTwoProductSpaceScoreRows": _commas(product_space["score_rows"]),
            "VTwoProductSpaceCountryUniverse": _commas(
                product_space["countries"]
            ),
            "VTwoProductSpaceProductUniverse": _commas(
                product_space["products"]
            ),
            "VTwoProductSpaceTargetProductUniverse": _commas(
                product_space["target_products"]
            ),
            "VTwoProductSpaceTargetDiagonalNonzeroBeforeExclusion": _commas(
                product_space["target_diagonal_nonzero_before_exclusion"]
            ),
            "VTwoProductSpaceTargetDiagonalMaxAfterExclusion": _decimal(
                product_space["target_diagonal_max_after_exclusion"]
            ),
        }
    )
    for cohort, cohort_tag in (("main", "Main"), ("historical", "Historical")):
        section = product_space["cohorts"][cohort]
        numbers[f"VTwoProductSpace{cohort_tag}BOneAPSixChainMean"] = _decimal(
            section["ap_mean"]
        )
        numbers[
            f"VTwoProductSpace{cohort_tag}BOneValueCaptureFiftySixChainMean"
        ] = _decimal(section["value_mean"])
        numbers[f"VTwoProductSpace{cohort_tag}BOneCandidates"] = _commas(
            section["rows"]
        )
        numbers[f"VTwoProductSpace{cohort_tag}BOnePositives"] = _commas(
            section["positives"]
        )
        numbers[
            f"VTwoProductSpace{cohort_tag}CandidatesWithTargetMembership"
        ] = _commas(section["membership_candidates"])
        numbers[f"VTwoProductSpace{cohort_tag}DiagonalMaterialChains"] = _commas(
            section["material_chains"]
        )
        for chain in CHAINS:
            chain_tag = CHAIN_TEX[chain]
            row = section["per_chain"][chain]
            prefix = f"VTwo{chain_tag}ProductSpace{cohort_tag}BOne"
            numbers[f"{prefix}AP"] = _decimal(row["ap"])
            numbers[f"{prefix}ValueCaptureFifty"] = _decimal(row["value"])
            if cohort == "main":
                numbers[f"{prefix}APCILower"] = _decimal(row["ap_ci_low"])
                numbers[f"{prefix}APCIUpper"] = _decimal(row["ap_ci_high"])

    main = product_space["cohorts"]["main"]
    numbers["VTwoProductSpaceBOneAPSixChainMean"] = _decimal(main["ap_mean"])
    numbers["VTwoProductSpaceBOneValueCaptureFiftySixChainMean"] = _decimal(
        main["value_mean"]
    )
    for chain in CHAINS:
        chain_tag = CHAIN_TEX[chain]
        row = main["per_chain"][chain]
        numbers[f"VTwo{chain_tag}ProductSpaceBOneAP"] = _decimal(row["ap"])
        numbers[f"VTwo{chain_tag}ProductSpaceBOneValueCaptureFifty"] = _decimal(
            row["value"]
        )


def _add_score_robustness_r5_numbers(
    numbers: dict[str, str],
    robustness: Mapping[str, Any],
) -> None:
    numbers.update(
        {
            "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
            "VTwoScoreRobustnessRFiveInputFileCount": _commas(
                robustness["input_file_count"]
            ),
            "VTwoScoreRobustnessRFiveCSVRows": _commas(robustness["csv_rows"]),
        }
    )
    task_tags = {"a": "TrackA", "b1": "TrackBOne", "b2": "TrackBTwo"}
    for task, task_tag in task_tags.items():
        row = robustness["paired"][task]
        prefix = f"VTwoPaired{task_tag}"
        numbers[f"{prefix}Delta"] = _decimal(row["point"])
        numbers[f"{prefix}CILower"] = _decimal(row["low"])
        numbers[f"{prefix}CIUpper"] = _decimal(row["high"])
        numbers[f"{prefix}BootstrapDraws"] = _commas(row["draws"])

    family_tags = {"kge": "KGE", "nbfnet": "NBFNet"}
    method_tags = {
        "official_raw_max": "RawMax",
        "ecdf_mean": "ECDFMean",
        "ecdf_top3_mean": "ECDFTopThree",
    }
    for (family, method), row in robustness["pooling"].items():
        prefix = f"VTwoPooling{family_tags[family]}{method_tags[method]}BOne"
        numbers[f"{prefix}AP"] = _decimal(row["ap"])
        numbers[f"{prefix}ValueCaptureFifty"] = _decimal(row["value"])

    for task, task_tag in task_tags.items():
        for family, family_tag in family_tags.items():
            row = robustness["budgets"][task][family]
            prefix = f"VTwoBudget{task_tag}{family_tag}"
            numbers[f"{prefix}RecallHeadline"] = _decimal(row["recall"])
            numbers[f"{prefix}ValueHeadline"] = _decimal(row["value"])
            numbers[f"{prefix}K"] = _commas(row["k"])

    for family, family_tag in family_tags.items():
        row = robustness["e2e"][family]
        prefix = f"VTwoEndToEnd{family_tag}"
        numbers[f"{prefix}GateRecall"] = _decimal(row["gate_recall"])
        numbers[f"{prefix}DestinationRecall"] = _decimal(
            row["destination_recall"]
        )
        numbers[f"{prefix}ValueCapture"] = _decimal(row["value_capture"])
        numbers[f"{prefix}MacroDestinationValueCapture"] = _decimal(
            row["macro_destination_value_capture"]
        )
        numbers[f"{prefix}MicroDestinationRecall"] = _decimal(
            row["micro_destination_recall"]
        )


def _add_eligibility_threshold_geometry_numbers(
    numbers: dict[str, str],
    geometry: Mapping[str, Any],
) -> None:
    numbers["VTwoEligibilityThresholdStatus"] = "COMPLETE"
    numbers["VTwoEligibilityThresholdExactHundredGatePass"] = (
        "TRUE" if geometry["exact_100_gate_pass"] else "FALSE"
    )
    threshold_tags = {50: "Fifty", 100: "Hundred", 250: "TwoFifty"}
    task_tags = {"a": "TrackA", "b1": "TrackBOne", "b2": "TrackBTwo"}
    for threshold, threshold_tag in threshold_tags.items():
        for task, task_tag in task_tags.items():
            row = geometry["thresholds"][threshold][task]
            prefix = f"VTwoEligibilityThreshold{threshold_tag}{task_tag}"
            numbers[f"{prefix}Candidates"] = _commas(row["candidates"])
            numbers[f"{prefix}Positives"] = _commas(row["positives"])
            numbers[f"{prefix}CandidateRetention"] = _decimal(
                row["candidate_retention"]
            )
            numbers[f"{prefix}PositiveRetention"] = _decimal(
                row["positive_retention"]
            )


def collect_numbers(
    paths: ArtifactPaths = DEFAULT_PATHS,
    *,
    require_gpu: bool = False,
    require_loco: bool = False,
    require_ultra: bool = False,
    require_gbdt: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate all available sources and return flat TeX macros plus hashes.

    With a false ``require_*`` flag, a completely absent corresponding result
    is represented only by a ``PENDING`` status.  A partial LOCO, ULTRA, or
    GBDT JSON/CSV pair is always invalid.  Canonical writes and verifies require
    formal GPU, LOCO, ULTRA, and GBDT inputs and therefore cannot promote
    pending or placeholder data.  Product-space, score-robustness r5, and
    eligibility-threshold geometry are mandatory schema-8 inputs.
    """
    hashes: dict[Path, str] = {}
    verified_value = _verify_value_diagnostics_first(paths)
    value_diagnostics = _load(paths.value_diagnostics)
    if value_diagnostics != verified_value:
        _fail(
            "value diagnostics",
            "strict JSON reload disagrees with the artifact returned by its verifier",
        )
    if require_gpu and not paths.gpu_summary.is_file():
        raise RuntimeError(
            "refusing canonical paper numbers: formal GPU summary is PENDING "
            f"({paths.gpu_summary.relative_to(paths.root).as_posix()} is missing)"
        )

    loco_presence = (paths.loco_summary.is_file(), paths.loco_summary_csv.is_file())
    if any(loco_presence) and not all(loco_presence):
        _fail(
            "public LOCO summary",
            "JSON and CSV must either both exist or both be absent",
        )
    if require_loco and not all(loco_presence):
        raise RuntimeError(
            "refusing canonical paper numbers: formal LOCO summary is PENDING "
            f"({paths.loco_summary.relative_to(paths.root).as_posix()} and its CSV are required)"
        )
    loco_info: dict[str, Any] | None = None
    if all(loco_presence):
        verified_loco = _verify_loco_summary_first(paths, hashes)
        loco_info = _validate_loco_summary(paths, verified_loco, hashes)

    ultra_presence = (
        paths.ultra_summary.is_file(),
        paths.ultra_summary_csv.is_file(),
    )
    if any(ultra_presence) and not all(ultra_presence):
        _fail(
            "public ULTRA summary",
            "JSON and CSV must either both exist or both be absent",
        )
    if require_ultra and not all(ultra_presence):
        raise RuntimeError(
            "refusing canonical paper numbers: formal ULTRA summary is PENDING "
            f"({paths.ultra_summary.relative_to(paths.root).as_posix()} and its CSV are required)"
        )
    ultra_info: dict[str, Any] | None = None
    if all(ultra_presence):
        verified_ultra = _verify_ultra_summary_first(paths, hashes)
        ultra_info = _validate_ultra_summary(paths, verified_ultra, hashes)

    gbdt_presence = (
        paths.gbdt_summary.is_file(),
        paths.gbdt_summary_csv.is_file(),
    )
    if any(gbdt_presence) and not all(gbdt_presence):
        _fail(
            "public GBDT summary",
            "JSON and CSV must either both exist or both be absent",
        )
    if require_gbdt and not all(gbdt_presence):
        raise RuntimeError(
            "refusing canonical paper numbers: formal GBDT summary is PENDING "
            f"({paths.gbdt_summary.relative_to(paths.root).as_posix()} and its CSV are required)"
        )
    gbdt_info: dict[str, Any] | None = None
    if all(gbdt_presence):
        verified_gbdt = _verify_gbdt_summary_first(paths, hashes)
        gbdt_info = _validate_gbdt_summary(paths, verified_gbdt, hashes)

    verified_product_space = _verify_product_space_first(paths, hashes)
    product_space_info = _validate_product_space_summary(
        paths, verified_product_space, hashes
    )
    verified_score_robustness_r5 = _verify_score_robustness_r5_first(paths, hashes)
    score_robustness_r5_info = _validate_score_robustness_r5(
        paths, verified_score_robustness_r5, hashes
    )
    verified_threshold_geometry = _verify_eligibility_threshold_geometry_first(
        paths, hashes
    )
    threshold_geometry_info = _validate_eligibility_threshold_geometry(
        paths, verified_threshold_geometry, hashes
    )

    summary = _load(paths.summary)
    rolling = _load(paths.rolling)
    raw_audit = _load(paths.raw_audit)
    robustness = _load(paths.robustness)
    registry_audit = _load(paths.registry_audit)
    registry_evidence = _load(paths.registry_evidence)
    coverage = _load(paths.b1_coverage)

    summary_by_chain = _validate_summary(summary)
    registry_info = _validate_registry(paths, registry_audit, registry_evidence, hashes)
    candidate_hashes = _candidate_hash_map(paths, rolling, summary_by_chain, hashes)
    _validate_cpu_macros(rolling)
    raw_info = _validate_raw_audit(paths, raw_audit, candidate_hashes, hashes)
    _validate_robustness(paths, robustness, rolling, candidate_hashes, raw_info, hashes)
    _validate_b1_coverage(paths, coverage, registry_info, raw_info, candidate_hashes, summary_by_chain, hashes)

    gpu: dict[str, Any] | None = None
    if paths.gpu_summary.is_file():
        gpu = _load(paths.gpu_summary)
        _validate_gpu_postfreeze_binding(paths, gpu, hashes)
        _validate_gpu_summary(gpu, candidate_hashes)
    elif require_gpu:
        raise RuntimeError(
            "refusing canonical paper numbers: formal GPU summary is PENDING "
            f"({paths.gpu_summary.relative_to(paths.root).as_posix()} is missing)"
        )

    value_info = _validate_value_diagnostics(
        paths,
        value_diagnostics,
        summary_by_chain,
        hashes,
    )

    total = summary["totals"]
    numbers: dict[str, str] = {
        "VTwoTrackACandidates": _commas(total["track_a_candidates"]),
        "VTwoTrackAPositives": _commas(total["track_a_positive_lanes"]),
        "VTwoTrackABaseRate": _percent(total["track_a_base_rate"]),
        "VTwoTrackAObservedValueB": _billions(total["track_a_observed_late_value_kusd"]),
        "VTwoTrackBOneCandidates": _commas(total["track_b_unique_entries"]),
        "VTwoTrackBOnePositives": _commas(total["track_b_positive_entries"]),
        "VTwoTrackBOneBaseRate": _percent(total["track_b_entry_base_rate"]),
        "VTwoTrackBOneObservedValueB": _billions(total["track_b_observed_late_value_kusd"]),
        "VTwoTrackBTwoCandidates": _commas(total["track_b2_conditional_lanes"]),
        "VTwoTrackBTwoPositives": _commas(total["track_b2_positive_lanes"]),
        "VTwoTrackBTwoBaseRate": _percent(total["track_b2_base_rate"]),
        "VTwoTrackBTwoObservedValueB": _billions(total["track_b2_observed_late_value_kusd"]),
    }
    _add_cpu_numbers(numbers, rolling)
    _add_registry_numbers(numbers, registry_info)
    _add_coverage_numbers(numbers, coverage)
    _add_robustness_numbers(numbers, robustness)
    _add_gpu_numbers(numbers, gpu, summary_by_chain)
    _add_value_diagnostic_numbers(numbers, value_info)
    _add_loco_numbers(numbers, loco_info)
    _add_ultra_numbers(numbers, ultra_info)
    _add_gbdt_numbers(numbers, gbdt_info)
    _add_product_space_numbers(numbers, product_space_info)
    _add_score_robustness_r5_numbers(numbers, score_robustness_r5_info)
    _add_eligibility_threshold_geometry_numbers(numbers, threshold_geometry_info)

    source_paths = (
        paths.summary,
        paths.rolling,
        paths.raw_audit,
        paths.robustness,
        paths.registry_audit,
        paths.registry_evidence,
        paths.b1_coverage,
        paths.value_diagnostics,
        paths.value_diagnostics_csv,
        paths.value_diagnostics_generator,
        paths.product_space_summary,
        paths.product_space_summary_csv,
        paths.product_space_scores,
        paths.product_space_generator,
        paths.product_space_config,
        paths.score_robustness_r5,
        paths.score_robustness_r5_csv,
        paths.score_robustness_r5_generator,
        paths.score_robustness_r5_config,
        paths.eligibility_threshold_geometry,
        paths.eligibility_threshold_geometry_csv,
        paths.eligibility_threshold_geometry_generator,
        paths.eligibility_threshold_geometry_config,
    )
    if gpu is not None:
        source_paths += (paths.gpu_summary,)
    if loco_info is not None:
        source_paths += (
            paths.loco_summary,
            paths.loco_summary_csv,
            paths.loco_summary_generator,
            paths.loco_config,
        )
    if ultra_info is not None:
        source_paths += (
            paths.ultra_summary,
            paths.ultra_summary_csv,
            paths.ultra_summary_generator,
            paths.ultra_config,
            paths.ultra_formal_controller,
        )
    if gbdt_info is not None:
        source_paths += (
            paths.gbdt_summary,
            paths.gbdt_summary_csv,
            paths.gbdt_summary_generator,
            paths.gbdt_config,
        )
    sources: dict[str, str] = {}
    for path in source_paths:
        observed = _sha256(path)
        previously_verified = hashes.setdefault(path, observed)
        if previously_verified != observed:
            _fail(
                "paper source inventory",
                f"source changed after verification: {path.relative_to(paths.root).as_posix()}",
            )
        sources[path.relative_to(paths.root).as_posix()] = observed
    return numbers, sources


def render_tex(numbers: Mapping[str, str], sources: Mapping[str, str]) -> str:
    lines = [
        "% AUTO-GENERATED by tools/generate_v2_paper_numbers.py; do not edit.",
        "% Source SHA-256 values:",
    ]
    lines.extend(f"% {digest}  {path}" for path, digest in sorted(sources.items()))
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in numbers.items())
    return "\n".join(lines) + "\n"


def render_json(numbers: Mapping[str, str], sources: Mapping[str, str]) -> str:
    """Render the complete current paper interface canonically."""

    if numbers.get("VTwoGPUStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with GPU status pending")
    if numbers.get("VTwoLOCOStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with LOCO status pending")
    if numbers.get("VTwoULTRAStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with ULTRA status pending")
    if numbers.get("VTwoGBDTStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with GBDT status pending")
    if numbers.get("VTwoProductSpaceStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with product-space status pending")
    if numbers.get("VTwoScoreRobustnessRFiveStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with r5 robustness status pending")
    if numbers.get("VTwoEligibilityThresholdStatus") != "COMPLETE":
        raise ValueError("cannot render canonical paper JSON with threshold-geometry status pending")
    payload = {
        "schema_version": PAPER_NUMBERS_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "gpu_status": "COMPLETE",
        "loco_status": "COMPLETE",
        "ultra_status": "COMPLETE",
        "gbdt_status": "COMPLETE",
        "sources": dict(sources),
        "numbers": dict(numbers),
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def write_outputs(
    tex_path: Path,
    json_path: Path,
    *,
    paths: ArtifactPaths = DEFAULT_PATHS,
) -> None:
    _assert_claimable_sources(paths)
    numbers, sources = collect_numbers(
        paths,
        require_gpu=True,
        require_loco=True,
        require_ultra=True,
        require_gbdt=True,
    )
    tex_path = Path(tex_path).resolve()
    json_path = Path(json_path).resolve()
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tex = render_tex(numbers, sources)
    tex_path.write_text(tex, encoding="utf-8", newline="\n")
    json_path.write_text(
        render_json(numbers, sources),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {_display(tex_path, paths.root)}")
    print(f"wrote {_display(json_path, paths.root)}")


def verify_outputs(
    tex_path: Path,
    json_path: Path,
    *,
    paths: ArtifactPaths = DEFAULT_PATHS,
) -> None:
    _assert_claimable_sources(paths)
    numbers, sources = collect_numbers(
        paths,
        require_gpu=True,
        require_loco=True,
        require_ultra=True,
        require_gbdt=True,
    )
    tex_path = Path(tex_path).resolve()
    json_path = Path(json_path).resolve()
    if tex_path.read_text(encoding="utf-8") != render_tex(numbers, sources):
        raise ValueError(f"stale generated TeX: {tex_path}")
    if json_path.read_bytes() != render_json(numbers, sources).encode("utf-8"):
        raise ValueError(f"stale or non-canonical generated JSON: {json_path}")
    print(f"verified {len(numbers)} paper numbers and {len(sources)} source hashes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_outputs(args.tex_out, args.json_out)
    else:
        write_outputs(args.tex_out, args.json_out)


if __name__ == "__main__":
    main()
