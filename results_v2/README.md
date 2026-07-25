# UPGRADE-BENCH results and claim artifacts

This directory is the machine-readable result surface for the current UpgradeBench benchmark. Its
name is retained for path compatibility; the paper treats UpgradeBench as one project rather than a
comparison between repository generations.

> **Resolved scientific contract.** The strict registry contains 283 included HS6 codes. Release eligibility is
> the conjunction of two independent branches:
> complete current-registry computations and a hash-bound sampled human-validation receipt. Their
> presentation here does not assert an execution order. The canonical review accepted no construct
> change, and `metrics/INVALIDATED.json` now records the verified `RESOLVED` transition that co-binds
> both branches and the current manuscript interface. A future accepted construct change starts a
> new registry/benchmark version and the full registry-dependent rebuild. For the current public
> reviewer release candidate, packaging, remote CI, and anonymous asset-download checksum checks
> have passed. Final archival publication, DOI, and manuscript/venue metadata remain separate
> administrative steps.

A file is not a paper claim merely because it exists. Only artifacts named by `CLAIM_LEDGER.md`,
covered by the appropriate verifier, and included in the final paper-number interface are claimable.
Private provenance and exploratory research notes may remain in maintainer staging without entering
the public result set.

## Rolling evaluation protocol

`tools/v2_rolling_cpu_baselines.py` enforces the temporal boundary:

1. It opens only historical inputs for `1998–2002 -> 2008–2012` during model selection.
2. Per chain and task, it selects logistic-regression regularization using group-safe historical CV.
   Track A and B1 use exporter groups; Track B2 keeps complete exporter-stage entries together and
   uses its per-positive-entry ranking objective.
3. It refits imputation, scaling, and the selected classifier on the historical cohort.
4. Only after those choices are frozen does it open and evaluate the complete
   `2008–2012 -> 2018–2022` target cohort. Same-window diagnostic split labels are ignored.

No target-window distribution statistic is used for preprocessing, fitting, model selection,
calibration, or imputation.

Track B1 predicts one `(exporter, processed stage)` event within destinations that already had early
processed-stage demand. It is not an exhaustive label for entry into previously inactive markets.
Track B2 scores destination lanes and evaluates ranking/value metrics only within B1 groups that
actually enter. B2 is therefore positive-entry-conditioned, not an end-to-end gate. Its value is
nested within B1 event value and must not be added as an independent value pool.

All result files inherit the registry's observable-attribution boundary. Unresolved focal
attribution is excluded; commodity-explicit blend and residual codes retain whole published HS6
basket value without identifying focal-material share, purity, physical composition, or dominant
feedstock. Results therefore concern commodity-explicit HS6 baskets and analytical families rather
than pure-material trade.

## Reference baselines and budgets

- **Track A:** exporter/importer size, gravity, and a historically selected size+gravity logistic
  model; global budgets `k={100,500,1000}` plus exporter shortlists at `k={5,10}`.
- **Track B1:** registered-upstream capacity and historically selected structural features; entry
  budgets `k={25,50,100}`.
- **Track B2:** importer demand, gravity, and a historically selected demand+gravity ranker;
  within-entry budgets `k={1,3,5}`.

The summary points used for cross-chain descriptive comparisons are global top-500 for A, global
top-50 for B1, and per-entry top-3 for B2. Cross-chain model deltas and population standard
deviations are descriptive; they are not confidence intervals over a sampled population of chains.

The reviewer-motivated GBDT reference uses the same task features as the logistic systems, with a
four-point `HistGradientBoostingClassifier` grid fixed in
`../configs/v2_gbdt_baselines.json`. A/B1 select by exporter-group historical AP; B2 selects by
positive-entry-group macro recall@3. Every one of the 18 chain--task models is selected and refit
before any main table is opened, and every B2 lane is scored before realized-entry conditioning.
Main uncertainty uses 200 exporter-cluster draws for A/B1 and entry-group draws for B2. The
replacement files have been rebuilt for the 283-code cohorts, pass their verifier, and are bound by
the canonical resolution receipt. This additional tabular reference was not part of the original prespecified reference set
and is not a target-selected champion.

Three post-protocol coverage artifacts answer reviewer-facing robustness questions without changing
the benchmark definition. `metrics/v2_product_space_density.{json,csv}` plus the keyed score CSV
provide a full-HS92, early-window B1 product-space reference; its public verifier recomputes main and
historical AP/value metrics without raw BACI. `metrics/v2_score_robustness_r5.{json,csv}` recomputes
same-candidate paired graph-score intervals, pooling and budget sensitivity, and the fixed B1-to-B2
cascade from frozen scores. `metrics/v2_eligibility_threshold_geometry.{json,csv}` rebuilds exact
50/100/250-kUSD candidate and positive sets; the 100-kUSD arm must reproduce every canonical key and
label. The alternative-threshold arms are cohort geometry, not rescored model performance.

## Benchmark scale and compute profile

`metrics/v2_benchmark_profile.json` and `../paper/generated/v2_benchmark_profile.tex` are the
verified current JSON/TeX pair. They are checked against repository sources and the explicitly
supplied private evidence, remain separate from the paper-number map, and expose only aggregates and
evidence digests, never private paths or contents.

## Value diagnostics

`metrics/v2_value_diagnostics.{json,csv}` reports observed late-value capture at the fixed budgets
above and binds every number to the verified benchmark/result inputs. The target-outcome-ranked
same-budget comparator is deliberately labeled as an oracle diagnostic. It is not deployable, does
not estimate an intervention effect, and is not a realizable policy ceiling. The pooled value
descriptions and the conditional B2 panel must be interpreted under the task populations documented
in the JSON; B2 is nested and non-additive.

## Graph and transfer evidence

`metrics/v2_gpu_rolling_summary.{json,csv}` is the existing sanitized graph rolling summary. It
reports the prespecified task/chain/model evaluations separately and binds the frozen selections,
metrics, scores, seeds, and source identities without publishing the raw run tree. No target-test
champion is selected.

The external-pretrained ULTRA-ZS reference is a separate evidence regime. One fixed `ultra_4g`
checkpoint is used without UpgradeBench-label training, checkpoint search, fine-tuning, calibration,
or score transformation, while each target chain's early graph is available at inference. Thus it is
zero-shot with respect to benchmark-label adaptation, not graph-free cold start. The current
`metrics/v2_ultra_zero_shot_summary.{json,csv}` pair is verified and receipt-bound. External pretraining data and
compute remain unmatched, and one checkpoint cannot support a training-seed interval or champion,
significance, population-generalization, or causal claim.

Matched LOCO remains a tier-abstracted NBFNet parameter-transport diagnostic over six held-out
chains, two modes, and five seeds. Both modes use the held-out chain's early graph at inference, and
training-edge volume is not equalized, so any paired difference jointly reflects source set and
volume rather than isolating domain mismatch. The current
`metrics/v2_loco_transfer_summary.{json,csv}` pair is verified and receipt-bound under this
descriptive limitation.

## Authorized result surface

The paths below describe the governed current-registry result surface bound by the resolution
receipt.

- `metrics/rolling_cpu_baselines.{json,csv}` — source-bound rolling CPU results, historical CV
  traces, selected regularization, fixed budgets, target metrics, and protocol flags.
- `metrics/raw_label_audit.json` — independent raw-BACI aggregation check for all 24
  chain/snapshot/track instances and 862,435 lane rows.
- `metrics/b1_candidate_coverage.json` — exact reconciliation of B1 candidate lanes, entry views,
  early-demand screening, and observed-value coverage.
- `metrics/v2_robustness.{json,csv}` — annual persistence, threshold, and importer-unseen
  diagnostics bound to the private raw archive. The importer-unseen B2 slice is small-sample and not
  a headline generalization.
- `metrics/v2_gpu_rolling_summary.{json,csv}` — sanitized verified graph rolling result.
- `metrics/v2_gbdt_baselines.{json,csv}` — formal reviewer-motivated GBDT results with historical
  grid-selection traces, all-models-before-main audit flags, clustered uncertainty, runtime, and
  exact config/runner/candidate-source hashes.
- `metrics/v2_product_space_density.{json,csv}` and
  `scores/v2_product_space_density_scores.csv` — replacement B1 product-space result and keyed
  score surface with raw-provenance hashes.
- `metrics/v2_score_robustness_r5.{json,csv}` — paired graph-score intervals, pooling/budget
  sensitivity, and fixed two-stage cascade diagnostics.
- `metrics/v2_eligibility_threshold_geometry.{json,csv}` — exact score-free cohort geometry for
  the 50/100/250-kUSD eligibility cutoffs.
- `metrics/v2_value_diagnostics.{json,csv}` — fixed-budget value diagnostics with explicit oracle
  and nesting semantics.
- `metrics/v2_loco_transfer_summary.{json,csv}` — sanitized current-registry matched-LOCO summary.
- `metrics/v2_ultra_zero_shot_summary.{json,csv}` — sanitized current-registry ULTRA-ZS summary.
- `metrics/v2_benchmark_profile.json` and `../paper/generated/v2_benchmark_profile.tex` — current
  graph-scale, effective-sample, and compute profile.
- `metrics/v2_contemporary_references.{json,csv}` and
  `../paper/generated/v2_contemporary_references.tex` — independently generated post-freeze
  contemporary-reference aggregates and their manuscript macro interface.
- `paper_numbers.json` and `../paper/generated/v2_numbers.tex` — receipt-bound current manuscript
  interface.
- `metrics/INVALIDATED.json` — fail-closed replacement/invalidation receipt. It must bind the exact
  final scientific artifacts before publication; canonical strict JSON, the exact field/scope/map
  inventories, current public verifier bytes, and the replacement paper source interface are enforced.
  Manual edits are not a supported resolution path.

## Verification

From the repository root, the currently available scientific artifacts can be checked with:

```powershell
& '.\.venv\Scripts\python.exe' tools\audit_chain_registry.py --check
& '.\.venv\Scripts\python.exe' tools\validate_v2.py
& '.\.venv\Scripts\python.exe' tools\audit_v2.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_b1_coverage.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_rolling_cpu_baselines.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_gbdt_baselines.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_product_space_density.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_score_robustness_r5.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_eligibility_threshold_geometry.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_robustness.py --verify-output
& '.\.venv\Scripts\python.exe' tools\summarize_v2_gpu_results.py --verify-output
& '.\.venv\Scripts\python.exe' tools\v2_value_diagnostics.py --verify-output
& '.\.venv\Scripts\python.exe' tools\summarize_v2_contemporary_references.py --verify --profile repository
```

The robustness command requires the authorized private raw archive. The LOCO, ULTRA, benchmark
profile, paper-number, and resolution verifiers additionally check the sanitized result pairs,
paper source map, JSON/TeX agreement, canonical values, and governed public bytes.
`metrics/INVALIDATED.json` is the authoritative receipt and currently records `RESOLVED`.

The in-repository digest is a deterministic drift/freeze contract, not an external signature. It
does not claim protection against a maintainer who changes both verifier code and its constants;
that stronger threat model requires an independently protected tag, cryptographic signature, or
DOI-backed archival record and digest.

The writers reject non-finite JSON output, duplicate keys, inconsistent labels/values, unexpected
calendar windows, source-hash drift, and task-specific size-basis violations.
