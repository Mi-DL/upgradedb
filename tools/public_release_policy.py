#!/usr/bin/env python3
"""Single source of truth for the UPGRADE-BENCH public release boundary.

Maintainer staging may contain permission-gated data and machine/run-specific
provenance. Public selectors are allowlist-first for reviewed results and deny
known internal surfaces everywhere else. Bundle planning,
the public repository manifest, and the privacy audit all import this module so
that a path cannot be public under one tool and internal under another.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping


PERMISSION_GATED_PUBLIC_PATHS = frozenset()

REGISTRY_HUMAN_REVIEW_RECEIPT = (
    "chains/evidence/registry_human_review_receipt.json"
)
REGISTRY_HUMAN_REVIEW_PROTOCOL = (
    "chains/evidence/registry_curation_protocol.json"
)
REGISTRY_HUMAN_REVIEW_FREEZE = (
    "chains/evidence/registry_human_review_freeze.json"
)
REGISTRY_HUMAN_REVIEW_RECEIPT_SCHEMA = (
    "upgrade-bench/registry-human-review-receipt/3"
)
REGISTRY_HUMAN_REVIEW_BINDING_FIELD = "registry_human_review"
REGISTRY_HUMAN_REVIEW_BINDING_FIELDS = frozenset(
    {
        "audit_id",
        "disposition",
        "receipt_path",
        "receipt_sha256",
        "protocol_path",
        "protocol_sha256",
        "freeze_path",
        "freeze_sha256",
    }
)

# Public ZIPs use generated payload manifests.  No staging-only nested
# manifest is permitted to cross the public boundary.
INTERNAL_NESTED_MANIFESTS = frozenset()

# The v2 result bundle is deliberately exact-name allowlisted.  New files under
# results_v2 are private by default.  Only reviewed summaries emitted by their
# fail-closed generators may cross this boundary; raw run artifacts never
# become public merely because they appear in the staging tree.
PUBLIC_V2_RESULT_ALLOWLIST = frozenset(
    {
        "results_v2/CLAIM_LEDGER.md",
        "results_v2/README.md",
        "results_v2/paper_numbers.json",
        "results_v2/metrics/raw_label_audit.json",
        "results_v2/metrics/b1_candidate_coverage.json",
        "results_v2/metrics/INVALIDATED.json",
        "results_v2/metrics/rolling_cpu_baselines.csv",
        "results_v2/metrics/rolling_cpu_baselines.json",
        "results_v2/metrics/v2_gpu_rolling_summary.csv",
        "results_v2/metrics/v2_gpu_rolling_summary.json",
        "results_v2/metrics/v2_gbdt_baselines.csv",
        "results_v2/metrics/v2_gbdt_baselines.json",
        "results_v2/metrics/v2_loco_transfer_summary.csv",
        "results_v2/metrics/v2_loco_transfer_summary.json",
        "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
        "results_v2/metrics/v2_ultra_zero_shot_summary.json",
        "results_v2/metrics/v2_benchmark_profile.json",
        "results_v2/metrics/v2_contemporary_references.csv",
        "results_v2/metrics/v2_contemporary_references.json",
        "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
        "results_v2/metrics/v2_eligibility_threshold_geometry.json",
        "results_v2/metrics/v2_product_space_density.csv",
        "results_v2/metrics/v2_product_space_density.json",
        "results_v2/metrics/v2_robustness.csv",
        "results_v2/metrics/v2_robustness.json",
        "results_v2/metrics/v2_score_robustness_r5.csv",
        "results_v2/metrics/v2_score_robustness_r5.json",
        "results_v2/metrics/v2_value_diagnostics.csv",
        "results_v2/metrics/v2_value_diagnostics.json",
        "results_v2/scores/v2_product_space_density_scores.csv",
    }
)

# This is the sole public exception to the general score-artifact exclusion.
# It contains deterministic B1 candidate identities, the label-free
# product-space density, and already-released outcomes so that consumers can
# recompute the public metric pair without the raw BACI archive.  Formal graph
# scores and every other score path remain private by default.
PUBLIC_V2_DERIVED_SCORE_ALLOWLIST = frozenset(
    {"results_v2/scores/v2_product_space_density_scores.csv"}
)

PUBLIC_V2_INVALIDATION_NOTICE = "results_v2/metrics/INVALIDATED.json"

# The invalidation receipt is a public, machine-verifiable release contract.
# Keep its current verifier inventory here, next to the public path policy, so
# the resolver and public consumer checks cannot silently drift apart.
V2_INVALIDATION_SCHEMA = "upgrade-bench-v2/result-invalidation/1"
V2_INVALIDATION_ACTIVE_STATUS = "INVALIDATED_REGISTRY_AUDIT"
V2_INVALIDATION_RESOLVED_STATUS = "RESOLVED"
V2_INVALIDATION_DATE = "2026-07-12"
V2_INVALIDATION_REASON = (
    "Non-chain-specific HS92 codes in multiple registries change early "
    "eligibility, labels, features, and graphs."
)
V2_INVALIDATION_ACTIVE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "invalidated_at",
        "scope",
        "reason",
        "claim_policy",
        "resolution",
    }
)
V2_INVALIDATION_LEGACY_RESOLVED_FIELDS = frozenset(
    V2_INVALIDATION_ACTIVE_FIELDS
    | {
        "original_status",
        "resolved_at",
        "replacement_sha256",
        "resolution_gate_sha256",
        "resolution_source_sha256",
        "resolution_verifier_sha256",
    }
)
V2_INVALIDATION_RESOLVED_FIELDS = frozenset(
    V2_INVALIDATION_LEGACY_RESOLVED_FIELDS
    | {REGISTRY_HUMAN_REVIEW_BINDING_FIELD}
)
# Frozen verifier inventory used by the completed schema-5 receipt.  It is
# recognized only by the private ``freshen`` transition and is not sufficient
# for current public receipt verification.
V2_LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS = (
    "tools/resolve_v2_invalidation.py",
    "tools/public_release_policy.py",
    "tools/generate_v2_paper_numbers.py",
    "tools/audit_chain_registry.py",
    "tools/audit_v2.py",
    "tools/v2_b1_coverage.py",
    "tools/v2_rolling_cpu_baselines.py",
    "tools/v2_robustness.py",
    "tools/summarize_v2_gpu_results.py",
    "tools/summarize_v2_loco_results.py",
    "src/v2_gpu_protocol.py",
)
# Frozen verifier inventory used by the completed schema-6 receipt.  Private
# formal score trees and checkpoint bytes never enter this public inventory.
V2_LEGACY_SCHEMA6_VERIFIER_SOURCE_PATHS = (
    V2_LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS
    + ("tools/summarize_v2_ultra_results.py",)
)
# Frozen verifier inventory used by the completed schema-7 receipt.
V2_LEGACY_SCHEMA7_VERIFIER_SOURCE_PATHS = (
    V2_LEGACY_SCHEMA6_VERIFIER_SOURCE_PATHS
    + ("tools/v2_gbdt_baselines.py",)
)
# The active schema-8 receipt additionally binds the three r5
# generators/verifiers and the registry human-review release gate. Their
# result/config bytes are direct paper-number sources where applicable; listing
# every gate here makes resolution fail closed if a verifier is changed
# independently of a result refresh.
V2_RESOLUTION_VERIFIER_SOURCE_PATHS = (
    V2_LEGACY_SCHEMA7_VERIFIER_SOURCE_PATHS
    + (
        "tools/build_gpu_step3_postfreeze_attestation.py",
        "tools/build_nbfnet_source_attestation.py",
        "tools/v2_product_space_density.py",
        "tools/v2_score_robustness_r5.py",
        "tools/v2_eligibility_threshold_geometry.py",
        "tools/registry_human_review_receipt.py",
        "tools/verify_registry_curation_protocol.py",
    )
)
# Recognized only by the private freshen transition.  It is never sufficient
# for current public receipt verification.
V2_LEGACY_SCHEMA4_VERIFIER_SOURCE_PATHS = tuple(
    path
    for path in V2_LEGACY_SCHEMA5_VERIFIER_SOURCE_PATHS
    if path != "tools/summarize_v2_loco_results.py"
)
V2_PAPER_NUMBERS_PATH = "results_v2/paper_numbers.json"
V2_PAPER_TEX_PATH = "paper/generated/v2_numbers.tex"
# The committed interface is the reviewed schema-8 snapshot. Historical
# schema-5/6/7 identities remain frozen below for migration and regression
# tests.
V2_PAPER_CURRENT_NUMBERS_SCHEMA = "upgrade-bench-v2-paper-numbers-8"
V2_PAPER_CURRENT_NUMBER_KEY_COUNT = 857
V2_PAPER_CURRENT_NUMBER_KEYS_SHA256 = (
    "bcd64e94804bea64fb66d5e73e8d463597d68a8d88a46b7d7230c42a2dfd4dda"
)
V2_PAPER_CURRENT_NUMBER_VALUES_SHA256 = (
    "152048039fd1482e069139f113745c2987796fd15b74fa4822e65f4dd357ef04"
)

V2_PAPER_FINAL_NUMBERS_SCHEMA = "upgrade-bench-v2-paper-numbers-8"
V2_PAPER_NUMBERS_SCHEMA = V2_PAPER_FINAL_NUMBERS_SCHEMA
V2_PAPER_NUMBERS_BENCHMARK_VERSION = "2.1-dev"
V2_PAPER_NUMBERS_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_version",
        "status",
        "gpu_status",
        "loco_status",
        "ultra_status",
        "gbdt_status",
        "sources",
        "numbers",
    }
)
V2_PAPER_SCHEMA5_SOURCE_PATHS = frozenset(
    {
        "data/processed_v2/dataset_summary.json",
        "results_v2/metrics/rolling_cpu_baselines.json",
        "results_v2/metrics/raw_label_audit.json",
        "results_v2/metrics/v2_robustness.json",
        "docs/registry_audit.json",
        "chains/evidence/registry_evidence.json",
        "results_v2/metrics/b1_candidate_coverage.json",
        "results_v2/metrics/v2_gpu_rolling_summary.json",
        "results_v2/metrics/v2_value_diagnostics.json",
        "results_v2/metrics/v2_value_diagnostics.csv",
        "tools/v2_value_diagnostics.py",
        "results_v2/metrics/v2_loco_transfer_summary.json",
        "results_v2/metrics/v2_loco_transfer_summary.csv",
        "tools/summarize_v2_loco_results.py",
        "configs/v2_loco_formal.json",
    }
)
# Historical schema 6 added exactly the public ULTRA JSON/CSV pair and the three
# public code/config inputs that define and verify it.  The checkpoint and raw
# formal run tree are intentionally absent.
V2_PAPER_SCHEMA6_SOURCE_PATHS = frozenset(
    set(V2_PAPER_SCHEMA5_SOURCE_PATHS)
    | {
        "results_v2/metrics/v2_ultra_zero_shot_summary.json",
        "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
        "tools/summarize_v2_ultra_results.py",
        "configs/v2_ultra_formal.json",
        "tools/v2_ultra_formal.py",
    }
)
# Active schema 7 adds the exact GBDT JSON/CSV pair plus the public runner and
# frozen configuration that define and verify the result.  Candidate-table and
# shared-runner hashes remain transitively checked by the GBDT verifier.
V2_PAPER_SCHEMA7_SOURCE_PATHS = frozenset(
    set(V2_PAPER_SCHEMA6_SOURCE_PATHS)
    | {
        "results_v2/metrics/v2_gbdt_baselines.json",
        "results_v2/metrics/v2_gbdt_baselines.csv",
        "tools/v2_gbdt_baselines.py",
        "configs/v2_gbdt_baselines.json",
    }
)
# Active schema 8 adds the product-space B1 reference, paired score robustness,
# and exact eligibility-threshold geometry.  The product keyed-score CSV is a
# deliberate public recomputation surface; no formal graph score file enters
# this source map.
V2_PAPER_SCHEMA8_SOURCE_PATHS = frozenset(
    set(V2_PAPER_SCHEMA7_SOURCE_PATHS)
    | {
        "results_v2/metrics/v2_product_space_density.json",
        "results_v2/metrics/v2_product_space_density.csv",
        "results_v2/scores/v2_product_space_density_scores.csv",
        "tools/v2_product_space_density.py",
        "configs/v2_product_space_density.json",
        "results_v2/metrics/v2_score_robustness_r5.json",
        "results_v2/metrics/v2_score_robustness_r5.csv",
        "tools/v2_score_robustness_r5.py",
        "configs/v2_score_robustness_r5.json",
        "results_v2/metrics/v2_eligibility_threshold_geometry.json",
        "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
        "tools/v2_eligibility_threshold_geometry.py",
        "configs/v2_eligibility_threshold_geometry.json",
    }
)
V2_PAPER_SOURCE_PATHS = V2_PAPER_SCHEMA8_SOURCE_PATHS

# Frozen historical schema-5 contract.  These constants remain for migration
# tests and must not be reused as the schema-6 contract.
V2_PAPER_SCHEMA5_FINAL_NUMBER_KEY_COUNT = 566
V2_PAPER_SCHEMA5_FINAL_NUMBER_KEYS_SHA256 = (
    "d0aa87e1a1e5cbb6b0ed80c4831c8203f1f737eb2766a3a08e3b49da598340cb"
)
# Canonical digest of the reviewed complete 566-entry schema-5 number map.
# This in-repository digest detects accidental or partial drift; it is not an
# external signature against simultaneous verifier-code changes.
V2_PAPER_SCHEMA5_FINAL_NUMBER_VALUES_SHA256: str | None = (
    "c9b2e0fabc071458bd03b0977e261e9d3c088ce9f615a69b9d3927c9c5b7a9e8"
)
# Frozen historical schema-6 contract reviewed with the complete ULTRA-backed
# macro inventory.  These values must not be redefined for schema 7.
V2_PAPER_SCHEMA6_FINAL_NUMBER_KEY_COUNT: int | None = 625
V2_PAPER_SCHEMA6_FINAL_NUMBER_KEYS_SHA256: str | None = (
    "b672a934bceb01e8cedaa0cc81ad54037b1bdfb8f44d79967ac0ac26577fcd6b"
)
V2_PAPER_SCHEMA6_FINAL_NUMBER_VALUES_SHA256: str | None = (
    "6c833f296b101049a9a1ea28247fbae77b9fa6ee2e32f7427ed7be2ccf12fe9b"
)
# Frozen schema-7 contract reviewed against the complete real GBDT-backed
# 694-value interface.  These values must never be derived from a synthetic fixture.
V2_PAPER_SCHEMA7_FINAL_NUMBER_KEY_COUNT: int | None = 694
V2_PAPER_SCHEMA7_FINAL_NUMBER_KEYS_SHA256: str | None = (
    "fdbd7453ac32c2719d62f6de3594b52810dd7b54b49ad3dd1c8aee3555a3d84c"
)
V2_PAPER_SCHEMA7_FINAL_NUMBER_VALUES_SHA256: str | None = (
    "d618c970e9caa547563879cbec64fc9ee259f50a36931e8d3d741941692aab43"
)
# Frozen schema-8 contract reviewed against the complete r5-backed interface.
V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT: int | None = 857
V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256: str | None = (
    "bcd64e94804bea64fb66d5e73e8d463597d68a8d88a46b7d7230c42a2dfd4dda"
)
V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256: str | None = (
    "152048039fd1482e069139f113745c2987796fd15b74fa4822e65f4dd357ef04"
)
# These inputs are distributed as indexed release assets rather than Git
# objects.  Repository-profile verification never opens them, even when they
# are mounted; full-profile verification requires and hashes them.  No other
# missing source is tolerated.
V2_EXTERNAL_SOURCE_PREFIXES = ("data/processed_v2/",)
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_SECONDS_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
_PAPER_NUMBER_KEY_RE = re.compile(r"VTwo[A-Za-z0-9]+\Z")
_PAPER_NUMBER_VALUE_RE = re.compile(
    r"-?(?:[0-9]+|[0-9]{1,3}(?:\{,\}[0-9]{3})+)(?:\.[0-9]+)?(?:\\%)?\Z"
)
_PAPER_SAFE_CPU_MODEL_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9 .,+()/\-]{0,127}\Z"
)
_TEX_SOURCE_RE = re.compile(r"% ([0-9a-f]{64})  ([^\s]+)\Z")
_TEX_MACRO_RE = re.compile(
    r"\\newcommand\{\\([A-Za-z][A-Za-z0-9]*)\}\{(.+)\}\Z"
)
PUBLIC_V2_INVALIDATION_HOLD_ALLOWLIST = frozenset(
    {
        "results_v2/CLAIM_LEDGER.md",
        "results_v2/README.md",
        PUBLIC_V2_INVALIDATION_NOTICE,
    }
)
# These summaries were introduced after the 2026-07-12 registry invalidation
# was resolved.  They have their own fail-closed generation/verification gates
# and must not retroactively change the exact historical resolution receipt.
POST_RESOLUTION_V2_RESULT_PATHS = frozenset(
    {
        "results_v2/metrics/v2_loco_transfer_summary.csv",
        "results_v2/metrics/v2_loco_transfer_summary.json",
        "results_v2/metrics/v2_value_diagnostics.csv",
        "results_v2/metrics/v2_value_diagnostics.json",
        "results_v2/metrics/v2_ultra_zero_shot_summary.csv",
        "results_v2/metrics/v2_ultra_zero_shot_summary.json",
        "results_v2/metrics/v2_gbdt_baselines.csv",
        "results_v2/metrics/v2_gbdt_baselines.json",
        "results_v2/metrics/v2_benchmark_profile.json",
        "results_v2/metrics/v2_contemporary_references.csv",
        "results_v2/metrics/v2_contemporary_references.json",
        "results_v2/metrics/v2_eligibility_threshold_geometry.csv",
        "results_v2/metrics/v2_eligibility_threshold_geometry.json",
        "results_v2/metrics/v2_product_space_density.csv",
        "results_v2/metrics/v2_product_space_density.json",
        "results_v2/metrics/v2_score_robustness_r5.csv",
        "results_v2/metrics/v2_score_robustness_r5.json",
        "results_v2/scores/v2_product_space_density_scores.csv",
    }
)
V2_INVALIDATION_DERIVED_PATHS = frozenset(
    (
        PUBLIC_V2_RESULT_ALLOWLIST
        - PUBLIC_V2_INVALIDATION_HOLD_ALLOWLIST
        - POST_RESOLUTION_V2_RESULT_PATHS
    )
    | {"paper/generated/v2_numbers.tex"}
)
# Any future unresolved invalidation suppresses every numeric v2 result,
# including summaries added after the historical receipt above.
V2_INVALIDATION_HOLD_PATHS = frozenset(
    (PUBLIC_V2_RESULT_ALLOWLIST - PUBLIC_V2_INVALIDATION_HOLD_ALLOWLIST)
    | {
        "paper/generated/v2_numbers.tex",
        "paper/generated/v2_benchmark_profile.tex",
        "paper/generated/v2_contemporary_references.tex",
    }
)
V2_INVALIDATION_NOTICE_GOVERNED_PATHS = frozenset(
    set(V2_INVALIDATION_HOLD_PATHS)
    | {
        "release/DATA_ARTIFACT_INDEX.json",
        "RELEASE_MANIFEST.sha256",
    }
)

PUBLIC_PORTABLE_JOBS = frozenset(
    {
        "jobs/v2_gpu_evaluate.pbs",
        "jobs/v2_gpu_main_worker.sh",
        "jobs/v2_gpu_nohup_worker.sh",
        "jobs/v2_gpu_select.pbs",
    }
)

# Public Git carries the current benchmark/release dependency closure, not every
# exploratory module retained in maintainer staging. Exact path lists
# keep ordinary ``git push`` planning and clean-export selection from silently
# reintroducing superseded analyses through broad directory recursion.
PUBLIC_CURRENT_SOURCE_ALLOWLIST = frozenset(
    {
        "src/baci_filtered_cache.py",
        "src/benchmark.py",
        "src/gap_discovery.py",
        "src/split.py",
        "src/task_features.py",
        "src/temporal_backtest.py",
        "src/universe.py",
        "src/v2_gpu_protocol.py",
        "src/v2_gpu_rolling.py",
        "src/v2_ultra.py",
        "src/window_aggregation.py",
    }
)
PUBLIC_CURRENT_TOOL_ALLOWLIST = frozenset(
    {
        "tools/artifact_bundles.py",
        "tools/audit_chain_registry.py",
        "tools/audit_v2.py",
        "tools/build_baci_filtered_cache.py",
        "tools/build_gpu_step3_postfreeze_attestation.py",
        "tools/build_nbfnet_source_attestation.py",
        "tools/build_paper_review.py",
        "tools/build_registry_evidence.py",
        "tools/build_registry_human_validation_sample.py",
        "tools/build_registry_lexicon_negative_control.py",
        "tools/build_registry_revision.py",
        "tools/build_v2_views.py",
        "tools/generate_v2_benchmark_profile.py",
        "tools/generate_v2_paper_numbers.py",
        "tools/public_release_audit.py",
        "tools/public_release_policy.py",
        "tools/prepare_registry_human_review_receipt.py",
        "tools/promote_v2_canonical_replacements.py",
        "tools/release_clean_clone.py",
        "tools/release_manifest.py",
        "tools/release_smoke.py",
        "tools/registry_human_review_receipt.py",
        "tools/repository_size_gate.py",
        "tools/resolve_v2_invalidation.py",
        "tools/step3_sync_manifest.py",
        "tools/summarize_v2_gpu_results.py",
        "tools/summarize_v2_loco_results.py",
        "tools/summarize_v2_ultra_results.py",
        "tools/summarize_v2_contemporary_references.py",
        "tools/test_split.py",
        "tools/v2_b1_coverage.py",
        "tools/v2_gpu_env_check.py",
        "tools/v2_gbdt_baselines.py",
        "tools/v2_loco_transfer.py",
        "tools/v2_ultra_formal.py",
        "tools/v2_robustness.py",
        "tools/v2_rolling_cpu_baselines.py",
        "tools/v2_value_diagnostics.py",
        "tools/validate_v2.py",
        "tools/verify_baci_filtered_cache.py",
        "tools/verify_registry_curation_protocol.py",
        "tools/verify_v2_number_alignment.py",
        "tools/v2_eligibility_threshold_geometry.py",
        "tools/v2_product_space_density.py",
        "tools/v2_score_robustness_r5.py",
    }
)
PUBLIC_CURRENT_TEST_ALLOWLIST = frozenset(
    {
        "tests/test_baci_filtered_cache.py",
        "tests/test_build_v2_views.py",
        "tests/test_build_paper_review.py",
        "tests/test_chain_registry_audit.py",
        "tests/test_generate_v2_paper_numbers.py",
        "tests/test_gpu_step3_postfreeze_attestation.py",
        "tests/test_nbfnet_source_attestation.py",
        "tests/test_prepare_registry_human_review_receipt.py",
        "tests/test_promote_v2_canonical_replacements.py",
        "tests/test_release_distribution.py",
        "tests/test_resolve_v2_invalidation.py",
        "tests/test_step3_sync_manifest.py",
        "tests/test_summarize_v2_gpu_results.py",
        "tests/test_summarize_v2_ultra_results.py",
        "tests/test_summarize_v2_contemporary_references.py",
        "tests/test_task_features.py",
        "tests/test_upgrade_bench_v2_package.py",
        "tests/test_v2_b1_coverage.py",
        "tests/test_v2_benchmark_profile.py",
        "tests/test_v2_cpu_results_env.py",
        "tests/test_v2_gpu_main_worker.py",
        "tests/test_v2_gpu_metrics.py",
        "tests/test_v2_gpu_protocol.py",
        "tests/test_v2_gbdt_baselines.py",
        "tests/test_v2_loco_transfer.py",
        "tests/test_v2_ultra_formal.py",
        "tests/test_v2_ultra_protocol.py",
        "tests/test_v2_robustness.py",
        "tests/test_v2_rolling_cpu_baselines.py",
        "tests/test_v2_eligibility_threshold_geometry.py",
        "tests/test_v2_product_space_density.py",
        "tests/test_v2_score_robustness_r5.py",
        "tests/test_v2_value_diagnostics.py",
        "tests/test_validate_v2.py",
        "tests/test_window_aggregation.py",
        "tests/test_registry_curation_protocol.py",
        "tests/test_registry_human_review_receipt.py",
        "tests/test_registry_human_validation_sample.py",
        "tests/test_registry_lexicon_negative_control.py",
        "tests/test_registry_revision.py",
        "tests/test_verify_v2_number_alignment.py",
    }
)
PUBLIC_EXACT_PREFIX_ALLOWLISTS = {
    "src/": PUBLIC_CURRENT_SOURCE_ALLOWLIST,
    "tools/": PUBLIC_CURRENT_TOOL_ALLOWLIST,
    "tests/": PUBLIC_CURRENT_TEST_ALLOWLIST,
}

INTERNAL_ONLY_PREFIXES = (
    ".private/",
    "private/",
    # Conversation/export artifacts can contain retained row-level review
    # workbooks and must never become public merely because they were tracked.
    "output/",
    "outputs/",
    # Retained in the private workspace for historical reproduction only. The
    # current public project has one benchmark definition and does not ship the
    # noncurrent package or its incompatible findings in Git or data bundles.
    "benchmark/upgrade-bench-v1/",
    "data/processed/",
    "results/",
    "results/logs/",
    "results_v2/gpu_rolling/",
    "results_v2/gpu_smoke/",
    "results_v2/loco_formal/",
    "results_v2/ultra_formal/",
    "third_party/ULTRA/",
)

# Raw inputs and raw-derived caches are accepted by the cohort code only below
# an explicit private/scratch surface.  Release policy is intentionally a little
# broader than ``.gitignore``: an accidentally tracked ``raw``/``cache`` tree is
# still private even if it appears below an otherwise public prefix.
PRIVATE_CACHE_PATH_COMPONENTS = frozenset(
    {"private", ".private", "tmp", "temp", "raw", "cache", ".cache", "caches"}
)

# These are run products, not portable scientific summaries.  Public evidence
# is restricted to exact allowlisted summary files; score caches, selection
# artifacts, checkpoints, claims, and logs stay internal no matter where they
# are accidentally copied in the staging tree.
INTERNAL_ARTIFACT_PATH_COMPONENTS = frozenset(
    {
        "logs",
        "claims",
        "job_claims",
        "gpu_rolling",
        "gpu-smoke",
        "gpu_smoke",
        "scores",
        "score_artifacts",
        "score-cache",
        "score_cache",
        "selection-artifacts",
        "selection_artifacts",
        "checkpoints",
    }
)
INTERNAL_ARTIFACT_SUFFIXES = frozenset({".log", ".ckpt", ".pt", ".pth"})
INTERNAL_ARTIFACT_FILENAME_MARKERS = (
    "gpu-score",
    "gpu_score",
    "score-cache",
    "score_cache",
    "selection-artifact",
    "selection_artifact",
)
RAW_BACI_DATA_PREFIXES = ("baci_hs92_", "baci-filtered-cache", "baci_filtered_cache")
RAW_DATA_SUFFIXES = frozenset({".csv", ".gz", ".parquet", ".feather", ".zip"})

INTERNAL_ONLY_PATHS = frozenset(
    {
        # Internal progress/review state is not part of a scientific artifact.
        "PROJECT_CHECKLIST.md",
        # Historical/raw provenance containing local paths or host identity.
        "results/README.md",
        "results/metrics/taxonomy_coder2_raw.json",
        # Host-specific operations, inventories, and local environment notes.
        "RUN_ON_IHPC.md",
        "configs/v2_gpu_hosts.json",
        "requirements/v2-gpu-local-plan.md",
        # The ULTRA feasibility smoke depends on a private vendored checkout,
        # checkpoint, adapter, and protocol test.  Publishing its runbook alone
        # would create a dangling, non-paper-facing public entry point.
        "requirements/ultra-smoke.md",
        # The formal LOCO freeze controller and its adversarial fixture encode
        # the private cluster's canonical claim-root path.  The scientific
        # runner/config remain publishable; these two operational files stay
        # in the private evidence snapshot so host/account identity cannot
        # leak through the repository or a release bundle.
        "tools/v2_loco_formal.py",
        "tests/test_v2_loco_formal.py",
        # Adversarial privacy fixtures intentionally contain synthetic private
        # host/account paths and therefore remain outside the public tree.
        "tests/test_summarize_v2_loco_results.py",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key {key!r}")
        payload[key] = value
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot render canonical strict JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _strict_canonical_json_file(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{role} cannot be read: {exc}") from exc
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{role} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} root is not an object")
    if content != _canonical_json_bytes(payload):
        raise ValueError(f"{role} bytes are not canonical JSON")
    return payload, content


def _validate_digest_map(
    value: object,
    role: str,
    *,
    expected_paths: set[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{role} must be a non-empty object")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        if not isinstance(relative, str) or canonical_path_reason(relative) is not None:
            raise ValueError(f"{role} contains an unsafe path")
        if not isinstance(digest, str) or _HEX64_RE.fullmatch(digest) is None:
            raise ValueError(f"{role} contains an invalid SHA-256 for {relative!r}")
        result[relative] = digest
    if expected_paths is not None and set(result) != expected_paths:
        raise ValueError(
            f"{role} inventory mismatch: missing={sorted(expected_paths - set(result))}, "
            f"extra={sorted(set(result) - expected_paths)}"
        )
    return result


def _validate_registry_human_review_binding(value: object) -> dict[str, str]:
    """Validate the exact public review artifacts sealed by resolution.

    This deliberately checks bytes and public identifiers only.  The semantic
    no-change review gate is rerun by ``resolve_v2_invalidation`` and release
    smoke; keeping the byte binding here also makes ordinary public receipt and
    invalidation-hold checks fail if any review artifact later drifts.
    """

    if not isinstance(value, dict) or set(value) != set(
        REGISTRY_HUMAN_REVIEW_BINDING_FIELDS
    ):
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            "registry human-review binding field inventory mismatch: "
            f"missing={sorted(REGISTRY_HUMAN_REVIEW_BINDING_FIELDS - observed)}, "
            f"extra={sorted(observed - REGISTRY_HUMAN_REVIEW_BINDING_FIELDS)}"
        )
    expected_paths = {
        "receipt_path": REGISTRY_HUMAN_REVIEW_RECEIPT,
        "protocol_path": REGISTRY_HUMAN_REVIEW_PROTOCOL,
        "freeze_path": REGISTRY_HUMAN_REVIEW_FREEZE,
    }
    result: dict[str, str] = {}
    for key, expected in expected_paths.items():
        if value.get(key) != expected:
            raise ValueError(
                f"registry human-review binding must use canonical {key}: {expected}"
            )
        result[key] = expected
        digest_key = key.replace("_path", "_sha256")
        digest = value.get(digest_key)
        if not isinstance(digest, str) or _HEX64_RE.fullmatch(digest) is None:
            raise ValueError(
                f"registry human-review binding has invalid {digest_key}"
            )
        result[digest_key] = digest
    audit_id = value.get("audit_id")
    if not isinstance(audit_id, str) or not audit_id.strip() or audit_id != audit_id.strip():
        raise ValueError("registry human-review binding audit_id is invalid")
    disposition = value.get("disposition")
    if disposition != "NO_CONSTRUCT_CHANGE":
        raise ValueError(
            "registry human-review binding is not release-eligible no-change evidence"
        )
    result["audit_id"] = audit_id
    result["disposition"] = disposition
    return result


def _normalized_invalidation_scope(scope: object) -> frozenset[str]:
    if not isinstance(scope, list) or not scope:
        raise ValueError("resolved receipt scope must be a non-empty list")
    normalized: list[str] = []
    for item in scope:
        if not isinstance(item, str) or canonical_path_reason(item) is not None:
            raise ValueError("resolved receipt scope contains an unsafe entry")
        canonical = item if "/" in item else f"results_v2/metrics/{item}"
        if item != canonical and canonical not in V2_INVALIDATION_DERIVED_PATHS:
            raise ValueError(f"resolved receipt scope contains a non-normative basename: {item!r}")
        normalized.append(canonical)
    if len(set(normalized)) != len(normalized):
        raise ValueError("resolved receipt scope contains duplicate normalized paths")
    observed = frozenset(normalized)
    expected = frozenset(V2_INVALIDATION_DERIVED_PATHS)
    if observed != expected:
        raise ValueError(
            "resolved receipt scope differs from the fail-closed public result hold: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return observed


def _validate_resolution_timestamps(payload: Mapping[str, Any]) -> None:
    if payload.get("invalidated_at") != V2_INVALIDATION_DATE:
        raise ValueError("resolved receipt changed the original invalidation date")
    resolved_at = payload.get("resolved_at")
    if not isinstance(resolved_at, str) or _UTC_SECONDS_RE.fullmatch(resolved_at) is None:
        raise ValueError("resolved_at must be a canonical UTC timestamp at whole-second precision")
    try:
        parsed = datetime.strptime(resolved_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("resolved_at is not a valid UTC timestamp") from exc
    if parsed.date() < date.fromisoformat(V2_INVALIDATION_DATE):
        raise ValueError("resolved_at predates the original invalidation")


def _paper_number_key_digest(keys: Iterable[str]) -> str:
    content = ("\n".join(sorted(keys)) + "\n").encode("ascii")
    return hashlib.sha256(content).hexdigest()


def _paper_number_value_digest(numbers: Mapping[str, str]) -> str:
    """Hash the exact canonical key/value map, independent of JSON/TeX order."""

    return hashlib.sha256(_canonical_json_bytes(numbers)).hexdigest()


def _validate_paper_numbers(
    value: object,
    *,
    allow_unfrozen_inventory: bool = False,
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("paper-number interface numbers must be a non-empty object")
    numbers: dict[str, str] = {}
    for key, rendered in value.items():
        if not isinstance(key, str) or _PAPER_NUMBER_KEY_RE.fullmatch(key) is None:
            raise ValueError("paper-number interface contains an invalid macro name")
        if not isinstance(rendered, str) or not rendered:
            raise ValueError(f"paper-number interface contains an empty/non-string value: {key}")
        cpu_model = key == "VTwoGBDTCPUModel" and (
            _PAPER_SAFE_CPU_MODEL_RE.fullmatch(rendered) is not None
        )
        if not cpu_model and rendered not in {"COMPLETE", "TRUE", "FALSE"} and (
            _PAPER_NUMBER_VALUE_RE.fullmatch(rendered) is None
        ):
            raise ValueError(f"paper-number interface contains a malformed value: {key}")
        numbers[key] = rendered
    observed_digest = _paper_number_key_digest(numbers)
    expected_count = V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT
    expected_digest = V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256
    if expected_count is None or expected_digest is None:
        if allow_unfrozen_inventory and expected_count is None and expected_digest is None:
            return numbers
        raise ValueError(
            "current schema-8 final paper-number macro inventory is not frozen; "
            "review the complete key set, set the schema-8 count/digest constants, "
            "then rerun the resolution transaction; "
            f"observed_candidate_count={len(numbers)}, "
            f"observed_candidate_sha256={observed_digest}"
        )
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("current paper-number key-count constant is invalid")
    if not isinstance(expected_digest, str) or _HEX64_RE.fullmatch(expected_digest) is None:
        raise ValueError("current paper-number key-digest constant is invalid")
    if (
        len(numbers) != expected_count
        or observed_digest != expected_digest
    ):
        raise ValueError(
            "paper-number macro inventory mismatch: "
            f"expected_count={expected_count}, "
            f"observed_count={len(numbers)}, observed_sha256={observed_digest}"
        )
    return numbers


def _validate_final_paper_number_value_digest(numbers: Mapping[str, str]) -> None:
    expected = V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256
    observed = _paper_number_value_digest(numbers)
    if expected is None:
        raise ValueError(
            "current schema-8 final paper-number value digest is not frozen; review the "
            "complete values, set V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256, "
            "then rerun the dry-run/resolve transaction; "
            f"observed_candidate_sha256={observed}"
        )
    if not isinstance(expected, str) or _HEX64_RE.fullmatch(expected) is None:
        raise ValueError("current paper-number value-digest constant is invalid")
    if observed != expected:
        raise ValueError(
            "paper-number frozen value digest mismatch: "
            f"expected={expected}, observed={observed}"
        )


def _verify_paper_tex_interface(
    root: Path,
    sources: Mapping[str, str],
    numbers: Mapping[str, str],
) -> None:
    unsafe = source_path_reason(V2_PAPER_TEX_PATH, root, require_file=True)
    if unsafe is not None:
        raise ValueError(
            "paper-number TeX interface path is unsafe or missing: "
            f"{V2_PAPER_TEX_PATH} ({unsafe})"
        )
    try:
        content = (root / V2_PAPER_TEX_PATH).read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"paper-number TeX interface cannot be read as UTF-8: {exc}") from exc
    if not content.endswith(b"\n") or b"\r" in content:
        raise ValueError("paper-number TeX interface does not use canonical LF bytes")
    lines = text.splitlines()
    if lines[:2] != [
        "% AUTO-GENERATED by tools/generate_v2_paper_numbers.py; do not edit.",
        "% Source SHA-256 values:",
    ]:
        raise ValueError("paper-number TeX interface has an invalid canonical header")

    observed_sources: dict[str, str] = {}
    observed_source_order: list[str] = []
    index = 2
    while index < len(lines) and lines[index].startswith("% "):
        match = _TEX_SOURCE_RE.fullmatch(lines[index])
        if match is None:
            raise ValueError("paper-number TeX interface has a malformed source proof")
        digest, relative = match.groups()
        if canonical_path_reason(relative) is not None or relative in observed_sources:
            raise ValueError("paper-number TeX interface has an unsafe/duplicate source proof")
        observed_sources[relative] = digest
        observed_source_order.append(relative)
        index += 1
    if observed_sources != dict(sources) or observed_source_order != sorted(sources):
        raise ValueError("paper-number JSON/TeX source maps differ")

    observed_numbers: dict[str, str] = {}
    for line in lines[index:]:
        match = _TEX_MACRO_RE.fullmatch(line)
        if match is None:
            raise ValueError("paper-number TeX interface has a malformed macro line")
        key, rendered = match.groups()
        if key in observed_numbers:
            raise ValueError(f"paper-number TeX interface repeats macro {key}")
        observed_numbers[key] = rendered
    if observed_numbers != dict(numbers):
        raise ValueError("paper-number JSON/TeX macro maps differ")


def _is_external_v2_source(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in V2_EXTERNAL_SOURCE_PREFIXES)


def _verify_digest_bound_files(
    root: Path,
    digests: Mapping[str, str],
    role: str,
    *,
    verify_external_sources: bool,
) -> None:
    for relative, expected in digests.items():
        # Repository policy may be evaluated once per selected path.  External
        # processed tables are covered by their frozen map but are opened only
        # by the explicit full-profile verifier, avoiding repeated hashing of
        # the large release-asset payload during ordinary path selection.
        if not verify_external_sources and _is_external_v2_source(relative):
            continue
        unsafe = source_path_reason(relative, root, require_file=False)
        if unsafe is not None:
            raise ValueError(f"{role} path is unsafe: {relative} ({unsafe})")
        path = root / relative
        if not path.exists():
            raise ValueError(f"{role} file is missing: {relative}")
        if not path.is_file():
            raise ValueError(f"{role} path is not a regular file: {relative}")
        if _sha256_file(path) != expected:
            raise ValueError(f"{role} hash mismatch: {relative}")


def verify_v2_resolution_receipt(
    root: Path,
    *,
    verify_external_sources: bool = True,
) -> dict[str, Any]:
    """Verify the public receipt without private raw data or formal run trees.

    Repository-only callers may set ``verify_external_sources=False``.  The
    external map remains structurally/hash bound in the paper interface, but
    ``data/processed_v2/`` release-asset bytes are neither required nor opened.
    Every repository code/config/result/source byte is always verified.
    """

    root = Path(root)
    unsafe = source_path_reason(PUBLIC_V2_INVALIDATION_NOTICE, root, require_file=True)
    if unsafe is not None:
        raise ValueError(
            "v2 invalidation receipt path is unsafe or missing: "
            f"{PUBLIC_V2_INVALIDATION_NOTICE} ({unsafe})"
        )
    notice = root / PUBLIC_V2_INVALIDATION_NOTICE
    payload, _ = _strict_canonical_json_file(notice, "v2 invalidation receipt")
    if set(payload) != set(V2_INVALIDATION_RESOLVED_FIELDS):
        raise ValueError(
            "resolved receipt field inventory mismatch: "
            f"expected={sorted(V2_INVALIDATION_RESOLVED_FIELDS)}, "
            f"observed={sorted(payload)}"
        )
    if payload.get("schema_version") != V2_INVALIDATION_SCHEMA:
        raise ValueError("resolved receipt schema is not the frozen invalidation schema")
    if (
        payload.get("status") != V2_INVALIDATION_RESOLVED_STATUS
        or payload.get("original_status") != V2_INVALIDATION_ACTIVE_STATUS
    ):
        raise ValueError("resolved receipt has an invalid status transition")
    if payload.get("reason") != V2_INVALIDATION_REASON:
        raise ValueError("resolved receipt changed the original invalidation reason")
    for key in ("claim_policy", "resolution"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"resolved receipt {key} must be a non-empty string")
    _validate_resolution_timestamps(payload)
    review_binding = _validate_registry_human_review_binding(
        payload.get(REGISTRY_HUMAN_REVIEW_BINDING_FIELD)
    )

    normalized_scope = _normalized_invalidation_scope(payload.get("scope"))
    replacements = _validate_digest_map(
        payload.get("replacement_sha256"),
        "replacement_sha256",
        expected_paths=set(normalized_scope),
    )
    verifiers = _validate_digest_map(
        payload.get("resolution_verifier_sha256"),
        "resolution_verifier_sha256",
        expected_paths=set(V2_RESOLUTION_VERIFIER_SOURCE_PATHS),
    )
    sources = _validate_digest_map(
        payload.get("resolution_source_sha256"),
        "resolution_source_sha256",
        expected_paths=set(V2_PAPER_SOURCE_PATHS),
    )
    gate_digest = payload.get("resolution_gate_sha256")
    gate_path = "tools/resolve_v2_invalidation.py"
    if not isinstance(gate_digest, str) or _HEX64_RE.fullmatch(gate_digest) is None:
        raise ValueError("resolution_gate_sha256 is invalid")
    if gate_digest != verifiers[gate_path]:
        raise ValueError("resolution gate hash differs from the exact verifier map")

    unsafe = source_path_reason(V2_PAPER_NUMBERS_PATH, root, require_file=True)
    if unsafe is not None:
        raise ValueError(
            "current paper-number interface path is unsafe or missing: "
            f"{V2_PAPER_NUMBERS_PATH} ({unsafe})"
        )
    paper, _ = _strict_canonical_json_file(
        root / V2_PAPER_NUMBERS_PATH,
        "current paper-number interface",
    )
    if set(paper) != set(V2_PAPER_NUMBERS_FIELDS):
        raise ValueError(
            "paper-number field inventory mismatch: "
            f"expected={sorted(V2_PAPER_NUMBERS_FIELDS)}, observed={sorted(paper)}"
        )
    if (
        paper.get("schema_version") != V2_PAPER_NUMBERS_SCHEMA
        or paper.get("benchmark_version") != V2_PAPER_NUMBERS_BENCHMARK_VERSION
        or paper.get("status") != "complete"
        or paper.get("gpu_status") != "COMPLETE"
        or paper.get("loco_status") != "COMPLETE"
        or paper.get("ultra_status") != "COMPLETE"
        or paper.get("gbdt_status") != "COMPLETE"
    ):
        raise ValueError("paper-number interface is not complete canonical schema 8")
    numbers = _validate_paper_numbers(paper.get("numbers"))
    if (
        numbers.get("VTwoGPUStatus") != "COMPLETE"
        or numbers.get("VTwoLOCOStatus") != "COMPLETE"
        or numbers.get("VTwoULTRAStatus") != "COMPLETE"
        or numbers.get("VTwoGBDTStatus") != "COMPLETE"
        or numbers.get("VTwoProductSpaceStatus") != "COMPLETE"
        or numbers.get("VTwoScoreRobustnessRFiveStatus") != "COMPLETE"
        or numbers.get("VTwoEligibilityThresholdStatus") != "COMPLETE"
    ):
        raise ValueError("paper-number interface has invalid or incomplete number fields")
    paper_sources = _validate_digest_map(
        paper.get("sources"),
        "paper-number sources",
        expected_paths=set(V2_PAPER_SOURCE_PATHS),
    )
    if sources != paper_sources:
        raise ValueError(
            "resolution_source_sha256 differs from the current paper-number source map"
        )
    _verify_paper_tex_interface(root, paper_sources, numbers)
    # Cross-format equality is checked first so a forged synchronized interface
    # cannot hide a JSON/TeX disagreement behind the frozen-map comparison.
    _validate_final_paper_number_value_digest(numbers)

    _verify_digest_bound_files(
        root,
        replacements,
        "resolved replacement",
        verify_external_sources=True,
    )
    _verify_digest_bound_files(
        root,
        verifiers,
        "resolution verifier",
        verify_external_sources=True,
    )
    _verify_digest_bound_files(
        root,
        sources,
        "paper-number source",
        verify_external_sources=verify_external_sources,
    )
    _verify_digest_bound_files(
        root,
        {
            review_binding["receipt_path"]: review_binding["receipt_sha256"],
            review_binding["protocol_path"]: review_binding["protocol_sha256"],
            review_binding["freeze_path"]: review_binding["freeze_sha256"],
        },
        "registry human-review binding",
        verify_external_sources=True,
    )
    return payload


def canonical_path_reason(path: object) -> str | None:
    """Return a reason when a repository-relative path is not canonical/safe."""
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        return "unsafe/non-canonical repository path"
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or PureWindowsPath(path).is_absolute()
        or ".." in pure.parts
        or path != pure.as_posix()
    ):
        return "unsafe/non-canonical repository path"
    return None


def source_path_reason(
    path: object,
    root: Path,
    *,
    require_file: bool = False,
) -> str | None:
    """Reject paths that escape *root* or traverse any symbolic link.

    Public manifests and bundles copy bytes, rather than preserving links.  A
    link could therefore smuggle host-private bytes into an apparently safe
    repository-relative member.  Reject links even when they resolve back
    inside the repository so the frozen identity is unambiguous.
    """
    reason = canonical_path_reason(path)
    if reason is not None:
        return reason
    assert isinstance(path, str)  # narrowed by ``canonical_path_reason``
    pure = PurePosixPath(path)
    lexical = root
    try:
        for part in pure.parts:
            lexical = lexical / part
            if lexical.is_symlink():
                return "symbolic-link path is not publishable"
        resolved_root = root.resolve()
        resolved = lexical.resolve(strict=False)
        if not resolved.is_relative_to(resolved_root):
            return "repository path resolves outside the release root"
        if lexical.exists() and not lexical.is_file():
            return "selected repository path is not a regular file"
        if require_file and not lexical.is_file():
            return "selected repository file is missing"
    except (OSError, RuntimeError):
        return "repository path cannot be resolved safely"
    return None


def unresolved_v2_invalidation(root: Path) -> str | None:
    """Return a repository-profile blocker while a v2 hold is unresolved.

    The resolved path performs the repository public receipt check.  Externally
    distributed ``data/processed_v2/`` source bytes are not opened or required;
    callers claiming a full payload must call ``verify_v2_resolution_receipt``
    with its default ``verify_external_sources=True`` as an additional gate.
    """
    path = root / PUBLIC_V2_INVALIDATION_NOTICE
    if not path.is_file():
        governed_files = V2_INVALIDATION_NOTICE_GOVERNED_PATHS
        if any((root / name).is_file() for name in governed_files):
            return "v2 invalidation notice is missing while governed release artifacts exist"
        processed = root / "data" / "processed_v2"
        try:
            if processed.is_dir() and any(item.is_file() for item in processed.rglob("*")):
                return "v2 invalidation notice is missing while governed data artifacts exist"
        except OSError:
            return "v2 invalidation notice is missing and the governed data surface is unreadable"
        return None
    try:
        candidate, _ = _strict_canonical_json_file(path, "v2 invalidation receipt")
        if candidate.get("status") == V2_INVALIDATION_ACTIVE_STATUS:
            if set(candidate) != set(V2_INVALIDATION_ACTIVE_FIELDS):
                raise ValueError("active invalidation field inventory mismatch")
            if (
                candidate.get("schema_version") != V2_INVALIDATION_SCHEMA
                or candidate.get("invalidated_at") != V2_INVALIDATION_DATE
                or candidate.get("reason") != V2_INVALIDATION_REASON
            ):
                raise ValueError("active invalidation changed its frozen origin")
            for key in ("claim_policy", "resolution"):
                value = candidate.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"active invalidation {key} must be non-empty")
            _normalized_invalidation_scope(candidate.get("scope"))
            return f"v2 result invalidation is unresolved ({V2_INVALIDATION_ACTIVE_STATUS})"
        verify_v2_resolution_receipt(root, verify_external_sources=False)
    except (OSError, ValueError) as exc:
        return f"v2 invalidation receipt is unresolved or invalid: {exc}"
    return None


def exclusion_reason(path: str, root: Path | None = None) -> str | None:
    """Return why *path* is outside the public surface, or ``None``."""
    unsafe = canonical_path_reason(path)
    if unsafe is not None:
        return unsafe
    if root is not None:
        unsafe = source_path_reason(path, root)
        if unsafe is not None:
            return unsafe
    pure = PurePosixPath(path)
    parts_casefold = tuple(part.casefold() for part in pure.parts)
    path_casefold = path.casefold()
    basename_casefold = pure.name.casefold()
    if any(
        part in PRIVATE_CACHE_PATH_COMPONENTS
        for part in parts_casefold
    ):
        return "private/raw/cache/tmp input or scratch path"
    if (
        any(part in INTERNAL_ARTIFACT_PATH_COMPONENTS for part in parts_casefold)
        and path not in PUBLIC_V2_DERIVED_SCORE_ALLOWLIST
    ):
        return "raw log/GPU run artifact path"
    if pure.suffix.casefold() in INTERNAL_ARTIFACT_SUFFIXES:
        return "raw log/GPU checkpoint artifact"
    if any(marker in basename_casefold for marker in INTERNAL_ARTIFACT_FILENAME_MARKERS):
        return "raw GPU score/selection artifact"
    if (
        pure.suffix.casefold() in RAW_DATA_SUFFIXES
        and basename_casefold.startswith(RAW_BACI_DATA_PREFIXES)
    ):
        return "raw BACI/cache payload"
    if path_casefold in {name.casefold() for name in PERMISSION_GATED_PUBLIC_PATHS}:
        return "permission-gated institutional extract"
    if path_casefold in {name.casefold() for name in INTERNAL_NESTED_MANIFESTS}:
        return "internal nested manifest covers permission-gated payloads"
    if path_casefold in {name.casefold() for name in INTERNAL_ONLY_PATHS}:
        return "internal-only staging/provenance path"
    for prefix in INTERNAL_ONLY_PREFIXES:
        if path_casefold.startswith(prefix.casefold()):
            return f"internal-only prefix {prefix}"
    for prefix, allowlist in PUBLIC_EXACT_PREFIX_ALLOWLISTS.items():
        if path_casefold.startswith(prefix.casefold()) and path not in allowlist:
            return f"not on the current public {prefix.rstrip('/')} allowlist"
    if path_casefold.startswith("results_v2/") and path not in PUBLIC_V2_RESULT_ALLOWLIST:
        return "v2 result is not on the explicit public allowlist"
    if (
        root is not None
        and path == "release/DATA_ARTIFACT_INDEX.json"
        and unresolved_v2_invalidation(root) is not None
    ):
        return "final public artifact index is suppressed by an unresolved invalidation hold"
    if (
        root is not None
        and path in V2_INVALIDATION_HOLD_PATHS
        and unresolved_v2_invalidation(root) is not None
    ):
        return "v2 numeric result is suppressed by an unresolved invalidation hold"
    if path_casefold.startswith("jobs/") and path not in PUBLIC_PORTABLE_JOBS:
        return "non-portable/internal scheduler path"
    return None


def is_public_path(path: str, root: Path | None = None) -> bool:
    return exclusion_reason(path, root) is None


def public_paths(paths: Iterable[str], root: Path | None = None) -> list[str]:
    return sorted({path for path in paths if is_public_path(path, root)})


def index_policy() -> dict[str, object]:
    """Machine-readable policy embedded in the public artifact index."""
    return {
        "visibility": "public",
        "selection_model": "exact-current-code-and-result-allowlists-plus-shared-exclusions",
        "permission_gated_paths_excluded": sorted(PERMISSION_GATED_PUBLIC_PATHS),
        "internal_nested_manifests_excluded": sorted(INTERNAL_NESTED_MANIFESTS),
        "internal_prefixes_excluded": list(INTERNAL_ONLY_PREFIXES),
        "private_cache_path_components_excluded": sorted(PRIVATE_CACHE_PATH_COMPONENTS),
        "internal_artifact_path_components_excluded": sorted(INTERNAL_ARTIFACT_PATH_COMPONENTS),
        "internal_artifact_suffixes_excluded": sorted(INTERNAL_ARTIFACT_SUFFIXES),
        "raw_baci_data_prefixes_excluded": list(RAW_BACI_DATA_PREFIXES),
        "symbolic_links_allowed": False,
        "repository_relative_paths_required": True,
        "internal_paths_excluded": sorted(INTERNAL_ONLY_PATHS),
        "public_portable_jobs": sorted(PUBLIC_PORTABLE_JOBS),
        "public_current_source_allowlist": sorted(PUBLIC_CURRENT_SOURCE_ALLOWLIST),
        "public_current_tool_allowlist": sorted(PUBLIC_CURRENT_TOOL_ALLOWLIST),
        "public_current_test_allowlist": sorted(PUBLIC_CURRENT_TEST_ALLOWLIST),
        "public_v2_result_allowlist": sorted(PUBLIC_V2_RESULT_ALLOWLIST),
        "public_v2_derived_score_allowlist": sorted(
            PUBLIC_V2_DERIVED_SCORE_ALLOWLIST
        ),
        "post_resolution_v2_result_paths": sorted(POST_RESOLUTION_V2_RESULT_PATHS),
        "v2_invalidation_derived_paths": sorted(V2_INVALIDATION_DERIVED_PATHS),
        "v2_invalidation_hold_paths": sorted(V2_INVALIDATION_HOLD_PATHS),
        "v2_invalidation_notice_governed_paths": sorted(
            V2_INVALIDATION_NOTICE_GOVERNED_PATHS
        ),
        "v2_resolution_verifier_source_paths": list(
            V2_RESOLUTION_VERIFIER_SOURCE_PATHS
        ),
        "v2_paper_legacy_schema5_source_paths": sorted(V2_PAPER_SCHEMA5_SOURCE_PATHS),
        "v2_paper_legacy_schema6_source_paths": sorted(V2_PAPER_SCHEMA6_SOURCE_PATHS),
        "v2_paper_legacy_schema7_source_paths": sorted(V2_PAPER_SCHEMA7_SOURCE_PATHS),
        "v2_paper_current_source_paths": sorted(V2_PAPER_SOURCE_PATHS),
        "v2_paper_current_numbers_schema": V2_PAPER_CURRENT_NUMBERS_SCHEMA,
        "v2_paper_current_number_key_count": V2_PAPER_CURRENT_NUMBER_KEY_COUNT,
        "v2_paper_current_number_keys_sha256": V2_PAPER_CURRENT_NUMBER_KEYS_SHA256,
        "v2_paper_current_number_values_sha256": V2_PAPER_CURRENT_NUMBER_VALUES_SHA256,
        "v2_paper_final_numbers_schema": V2_PAPER_FINAL_NUMBERS_SCHEMA,
        "v2_paper_schema6_final_number_key_count": V2_PAPER_SCHEMA6_FINAL_NUMBER_KEY_COUNT,
        "v2_paper_schema6_final_number_keys_sha256": V2_PAPER_SCHEMA6_FINAL_NUMBER_KEYS_SHA256,
        "v2_paper_schema6_final_number_values_sha256": (
            V2_PAPER_SCHEMA6_FINAL_NUMBER_VALUES_SHA256
        ),
        "v2_paper_schema7_final_number_key_count": V2_PAPER_SCHEMA7_FINAL_NUMBER_KEY_COUNT,
        "v2_paper_schema7_final_number_keys_sha256": V2_PAPER_SCHEMA7_FINAL_NUMBER_KEYS_SHA256,
        "v2_paper_schema7_final_number_values_sha256": (
            V2_PAPER_SCHEMA7_FINAL_NUMBER_VALUES_SHA256
        ),
        "v2_paper_schema8_final_number_key_count": V2_PAPER_SCHEMA8_FINAL_NUMBER_KEY_COUNT,
        "v2_paper_schema8_final_number_keys_sha256": V2_PAPER_SCHEMA8_FINAL_NUMBER_KEYS_SHA256,
        "v2_paper_schema8_final_number_values_sha256": (
            V2_PAPER_SCHEMA8_FINAL_NUMBER_VALUES_SHA256
        ),
        "v2_public_receipt_profiles": {
            "repository": "verify-public-git-bytes-without-opening-external-processed-v2-sources",
            "full": "verify-all-public-git-and-external-source-bytes",
        },
        "registry_human_review_receipt_path": REGISTRY_HUMAN_REVIEW_RECEIPT,
        "registry_human_review_protocol_path": REGISTRY_HUMAN_REVIEW_PROTOCOL,
        "registry_human_review_freeze_path": REGISTRY_HUMAN_REVIEW_FREEZE,
        "registry_human_review_receipt_schema": REGISTRY_HUMAN_REVIEW_RECEIPT_SCHEMA,
        "registry_human_review_resolution_binding_field": (
            REGISTRY_HUMAN_REVIEW_BINDING_FIELD
        ),
        "registry_human_review_required_for_release": True,
        "unresolved_v2_invalidation_blocks_final_index": True,
    }
