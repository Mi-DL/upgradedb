# UPGRADE-BENCH claim-to-artifact ledger

Last updated: 2026-07-23 (Australia/Sydney).

This ledger governs current UpgradeBench claims only. Reader-visible claims must be supported by the
named authoritative artifact and retain every scope qualifier below. The presence of code,
configuration, or a planned path does not make a numerical claim release-eligible.

Status vocabulary is literal:

- **ready** — may be cited with the listed scope and qualifier;
- **ready only as descriptive diagnostic** — may be cited only with the adjacent limitation;
- **smoke only; not a result** — engineering feasibility evidence only; and
- **not claimable** — no current design or authorized artifact supports the claim.

Release readiness is the conjunction of independent construct-review and computation branches; a
row's readiness label describes its evidence, not the order in which work occurred. The current
observable-attribution registry contains 283 included HS6 codes. The canonical resolution receipt
co-binds the complete current-registry result set and the hash-bound, outcome-blind sampled-validation
evidence. Two reviewers completed the frozen stratified sample of 212 of 610 machine decisions, and
all 53 stage definitions were accepted; 398 decisions remain outside individual human validation.
Original verdicts are retained privately and final adjudication accepted no construct change, so the
bound computations remain applicable. Any future accepted construct change requires a new
registry/benchmark version and complete registry-dependent rebuild before resealing.

## Claim matrix

| claim / paper object | status | authoritative artifact | required verifier / qualifier |
|---|---|---|---|
| Main and historical cohort counts, positives, base rates, and observed late value across all rebuilt tables/views | **ready** | `data/processed_v2/dataset_summary.json`; `data/processed_v2/dataset_summary_fold2.json` | Checked by `python tools/validate_v2.py` and co-bound by the canonical resolution receipt |
| Calendar-mean aggregation and task-specific units for Tracks A, B1, and B2 | **ready** | `BENCHMARK_V2_SPEC.md`; replacement table schema | Registry-independent normative definition; unit/split tests plus `python tools/validate_v2.py` |
| Strict-registry construction: frozen regex scan of 5,022 source rows; 576 observable plus 34 legacy-only records; 610 chain--HS6 decisions over 588 unique HS6 codes; 283 included, 228 excluded decision records, 99 out of stage, 53 active stages, and 19 retained-code stage reassignments | **ready** | `chains/evidence/registry_evidence.json`; `docs/registry_audit.json` | The 228 excluded count is at chain--HS6 decision-record level; 207 globally unique HS6 codes have at least one excluded decision. This is a generated construct claim, not a review-status claim. Finite negative controls do not prove lexicon completeness; `python tools/audit_chain_registry.py --check` |
| Frozen sampled human-validation design: 212 of 610 decision records, with all 53 stage definitions designated for census validation | **ready** | `chains/evidence/registry_human_validation_sample.json`; `chains/evidence/registry_curation_protocol.json` | No fixed fraction, including 25%, is an academic sufficiency criterion. The 398 other decisions are unsampled, and full-frame discrepancy estimates require the stored design weights rather than a simple-random-sample bound. |
| Completed sampled human-validation result | **ready** | `chains/evidence/registry_human_review_receipt.json`; `chains/evidence/registry_curation_protocol.json` | Two-reviewer validation covers 212 sampled decisions and the 53-definition census, not the 398 unsampled decisions; final adjudication accepted no construct change. |
| Raw label, late-value, early-absence, and negative-late-value reconciliation | **ready** | `results_v2/metrics/raw_label_audit.json` | `python tools/audit_v2.py --verify-output`; exact replacement proof is bound by the resolution receipt |
| B1 eligible-market boundary and candidate coverage | **ready** | `results_v2/metrics/b1_candidate_coverage.json` | `python tools/v2_b1_coverage.py --verify-output`; retain eligible-market scope |
| Group-safe same-window diagnostic assignment rule | **ready** | `BENCHMARK_V2_SPEC.md`; regenerated `group_id`, `transductive_split_unit`, and `transductive_split` fields | Call assignments transductive diagnostics, not independent forecasts |
| Track A Forward CPU ranking, budget, shortlist, value-capture, and uncertainty metrics | **ready** | `results_v2/metrics/rolling_cpu_baselines.json` | Locked historical selection and frozen main evaluation |
| Track B1 eligible-market entry ranking and value metrics | **ready** | same rolling CPU JSON | Retain eligible-market scope |
| Track B2 conditional destination recall and observed-late-value capture | **ready** | same rolling CPU JSON | B2 is conditional and nested within B1 |
| Protocol-conforming nonlinear GBDT reference across six chains and A/B1/B2 | **ready** | `results_v2/metrics/v2_gbdt_baselines.json`; `.csv` | `python tools/v2_gbdt_baselines.py --verify-output`; descriptive reviewer-motivated reference, not a target-selected champion |
| Full-HS92 product-space density reference for B1 | **ready only as descriptive diagnostic** | `results_v2/metrics/v2_product_space_density.json`; `.csv`; keyed-score CSV | `python tools/v2_product_space_density.py --verify-output`; B1-only association reference |
| Global-graph-pool and path-based NBFNet within-chain Forward references | **ready** | `results_v2/metrics/v2_gpu_rolling_summary.json`; `.csv` | `python tools/summarize_v2_gpu_results.py --verify-output`; private score trees are not public artifacts |
| Paired graph-score uncertainty, pooling/budget sensitivity, and B1-to-B2 cascade | **ready only as descriptive diagnostic** | `results_v2/metrics/v2_score_robustness_r5.json`; `.csv` | `python tools/v2_score_robustness_r5.py --verify-output`; fixed-chain descriptive uncertainty |
| Exact 50/100/250-kUSD eligibility-cohort geometry | **ready only as descriptive diagnostic** | `results_v2/metrics/v2_eligibility_threshold_geometry.json`; `.csv` | Alternative arms describe rebuilt cohort geometry, not rescored performance |
| Per-chain early-graph scale, B1/B2 effective sample units, and aggregate compute accounting | **ready** | `results_v2/metrics/v2_benchmark_profile.json`; `paper/generated/v2_benchmark_profile.tex` | `python tools/generate_v2_benchmark_profile.py --verify`; derived from current formal receipts |
| Fixed external-pretrained ULTRA-ZS reference across six chains and A/B1/B2 | **ready only as descriptive diagnostic** | `results_v2/metrics/v2_ultra_zero_shot_summary.json`; `.csv` | `python tools/summarize_v2_ultra_results.py --verify-output`; fixed official checkpoint, unmatched compute/data, no UpgradeBench-label adaptation |
| Persistence, identity, hub, reporting-entity, and outcome-threshold robustness slices | **ready only as descriptive diagnostic** | `results_v2/metrics/v2_robustness.json` | `python tools/v2_robustness.py --verify-output`; the raw archive remains private |
| Importer-unseen conditional-B2 slice | **ready only as descriptive diagnostic** | identity slice in `results_v2/metrics/v2_robustness.json` | Retain finite-cohort and sample-size limitations |
| Unique observed-value accounting | **ready** | `results_v2/metrics/v2_value_diagnostics.json`; `.csv` | `python tools/v2_value_diagnostics.py --verify-output`; B2 remains nested, not additive |
| Same-budget outcome-ranked oracle and model-to-oracle gap | **ready only as descriptive diagnostic** | same value-diagnostic pair | Outcome-ranked oracles are non-deployable descriptive upper bounds |
| Tier-abstracted matched NBFNet parameter-transport gap (`in_domain - loco`) | **ready only as descriptive diagnostic** | `results_v2/metrics/v2_loco_transfer_summary.json`; `.csv` | `python tools/summarize_v2_loco_results.py --verify-output`; source set and training volume are not isolated |

## Value-diagnostic guardrail

The value oracle ranks candidates with their realized outcomes and evaluates at the same reporting
budget as each model. It answers a descriptive accounting question: how much realized value lies
inside the candidate set at that budget under outcome ranking. It does not measure deployable
performance, ex-ante predictability, causal uplift, productive capacity, welfare, or a practically
attainable ceiling. Every paper sentence, table label, and caption containing an oracle value or gap
must carry this interpretation.

## Current-registry release obligations

The items below define a dependency set, not a progress log or execution chronology. Machine
receipts, rather than checkboxes in this document, establish whether each obligation is satisfied.

- Freeze and audit the current strict registry, including the complete decision ledger.
- Rebuild and verify both temporal cohorts and all CPU-side audit/reference artifacts.
- Complete, safely promote, and formally summarize the GPU Forward and matched-LOCO results.
- Regenerate and verify GPU-score robustness, value diagnostics, GBDT, product-space, CPU
  robustness, and threshold-geometry artifacts.
- Regenerate and verify the ULTRA-ZS comparison and benchmark-profile artifacts.
- Complete the outcome-blind 212-record sampled registry validation and 53-definition census, then
  retain the hash-bound receipt. Do not describe the 398 unsampled records as individually reviewed.
  If any code
  decision, stage membership or assignment, stage definition, or semantic `upstream`,
  `upstream_map`, `derived_from`, `derived_from_hs`, `produces`, `form_of`, or
  `named_sources` changes, assign a new registry/benchmark version and rebuild every
  registry-dependent result. If no such change is accepted, retain the co-bound receipt without a
  computational rerun.
- Regenerate and verify the paper-number interface, governed-source inventory, TeX wrappers, and
  PDFs from only current-registry artifacts.
- Rebuild, smoke-test, reseal, and publish the release package only after one resolution receipt
  co-binds every applicable obligation above.

Paper numbers and public release remain fail-closed unless the corresponding machine receipts prove
the complete dependency set. When JSON and convenience CSV differ, the machine-readable verified
JSON is authoritative for its bound release. Only successful current public verifiers and an updated
ledger entry can establish a claimable status.
