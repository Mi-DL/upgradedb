# UPGRADE-BENCH benchmark specification

Status: **normative benchmark specification**. Current paper claims and benchmark releases must
follow this document and the claim-to-artifact ledger.

Release contract: the strict observable-attribution registry contains 283 included HS6 codes.
Public release co-seals two independent branches: the complete
current-registry computation/result evidence and a canonical hash-bound outcome-blind sampled human-validation
receipt. Their presentation does not assert an execution order. A receipt that accepts no construct
change leaves the bound computations applicable; an accepted construct change requires a new
registry/benchmark version and full registry-dependent rebuild before resealing. The current public
reviewer release candidate includes the receipt-bound sanitized result interfaces, packaging
metadata, and bundle index. Its remote CI and anonymous asset-download checksum checks have passed.
The fixed review instrument contains no trade values, labels, cohort impacts, model scores, or
downstream result summaries. Final archival publication, DOI, and manuscript/venue metadata remain
separate administrative steps. This contract does not alter the normative definitions below.

## 1. Window aggregation

For a window containing the calendar years `Y`, first aggregate all registered HS6 products to a
stage total for each `(exporter, importer, stage, year)`:

```text
stage_value[i,j,s,y] = sum(value[i,j,k,y] for k mapped to stage s)
```

A missing BACI row is a zero flow for benchmark aggregation. The window value is the arithmetic
mean over **all calendar years in the window**, including zero-flow years:

```text
window_mean[i,j,s] = sum(stage_value[i,j,s,y] for y in Y) / len(Y)
```

The order is therefore **stage-by-year sum, then calendar-year mean**. It is not the mean over years
in which an HS6 flow happens to be present.

- An early lane is established iff `early_mean > 100` (BACI values are USD thousands).
- A late lane materializes iff `late_mean > 100`.
- The 100 kUSD cutoff is a **fixed nominal threshold**. It is not deflated or rescaled for changes
  in the price level or world trade between the historical and main folds, so it must not be
  interpreted as constant real economic significance. Threshold and time-period sensitivity must
  be reported as diagnostics; changing the default requires a new benchmark version and rebuild.
- Exact threshold-cohort geometry is released separately for 50, 100, and 250 kUSD. Each arm
  rebuilds early eligibility, incumbent status, importer demand, late labels, B1 groups, and B2
  conditioning from annual raw inputs under the same calendar-complete aggregation. The 100-kUSD
  arm must exactly reproduce every released A/B1/B2 key and label before either alternative is
  accepted. This artifact reports counts and key overlap only; it neither reuses fixed-cohort model
  scores nor changes the 100-kUSD default.
- `lateval` is `late_mean` for a materialized lane and zero otherwise.
- A persistence-sensitivity release may add `active_years`, defined as the number of calendar years
  whose stage total exceeds the threshold. This derived field must be rebuilt from annual inputs and
  validated before any persistence result is reported; it is not present in the current `2.1-dev`
  lane tables and does not change the default label.
- A strict persistence label (for example, active in at least 3 of 5 years) is an auxiliary slice,
  never described as the shipped/default label.

The same aggregation must be used for candidate construction, labels, market-size features, graph
facts, audits, and all temporal folds.

## 2. Tasks and units of observation

### Track A: destination extension

Unit: `(exporter i, processed stage s, importer j)`.

The exporter has both a flow in at least one registered upstream stage for `s` and an established
processed-stage export in the early window. Registered upstream stages are the raw or intermediate
stages explicitly listed in the chain's `upstream_map[s]`; they must not be described as raw inputs
when the registry permits an intermediate stage. The candidate lane `(i,s,j)` is not established
early, and importer `j` has early demand for stage `s`. The target is late-window lane
materialization.

This track measures **new destination formation by an incumbent processed-stage exporter**. It is
not first-time industrial upgrading and must not be described as such.

### Track B1: eligible-market processed-export stage entry

Unit: `(exporter i, processed stage s)`.

The exporter has a flow in at least one registered upstream stage for `s` but no established export
of processed stage `s` in the early window. An eligible destination `j` must already import stage
`s` from at least one country in the early window and must differ from exporter `i`. The target
`z[i,s]` is one iff at least one eligible `(i,s,j)` lane materializes late.

Each eligible-market entry event appears exactly once. Official splits and metrics operate on this
unit. Track B1 does **not** enumerate exports to destinations with no early processed-stage demand
and is not an exhaustive indicator of any first processed-stage export.

### Track B2: conditional destination formation

Unit: `(exporter i, processed stage s, importer j)`, evaluated only for positive Track-B1
eligible-market entries and only over the same early-demand destination set.

This is a conditional ranking task: given that `(i,s)` enters, rank the destinations it reaches. It
must be reported separately from entry prediction; multiple positive destinations are outcomes of
one entry event, not independent entry events.

Both Track B tasks are **processed-export emergence proxies**. Domestic processing capability is
not identified without external production/capacity validation.

### Registry construct contract

Every chain registry must attach an authoritative HS92 description, source/version, inclusion
rationale, and specificity status to each product code. The official description is the sole
adjudicative evidence; external knowledge may clarify terminology but may not supply an absent focal
commodity, material, species, product form, or end use. A code that does not identify the named
commodity or material may not be silently used for a commodity-specific claim. Commodity-explicit
blends and residual `n.e.s.` categories may be retained, whereas generic, multi-commodity, or
material-ambiguous categories that do not isolate the focal commodity or material are excluded.
Inclusion of a blend or residual category supports inference only at the published HS6-basket level;
it does not identify the basket's internal composition. Sensitivity analysis is not a substitute for
code-level specificity. Modeled
`derived_from`/`upstream_map` relations are eligibility assumptions, not measured input-output
coefficients. Candidate-universe completeness is claimable only relative to the
frozen focal lexicons, the audited registry, and the task's early-demand
destination screen. Passing finite lexical negative controls supports the
tested variants but does not prove lexicon completeness.

Every active stage must also have a canonical definition as a registry-defined analytical product
family. Such a family need not correspond one-to-one to a unique physical operation or every HS
product-form distinction. Each included code needs a code-level
`stage_fit` with status `supported`, the official HS92 description as evidence, and a rationale
showing that the description falls inside the stage definition. The presence of the chain commodity
name alone cannot imply a processing form, feedstock, species, or end use. Closely related HS6 forms
may be pooled when the analytical family is the intended benchmark unit; distinct forms must be
split when pooling would change that unit or make the entry/incumbency condition misleading. Entry
and incumbency are then measured at the declared family level and may mask first entry into one
constituent form. The operational boundary cases and minimal review fields are fixed in
the immutable pre-review `docs/REGISTRY_REVIEW_CODEBOOK.md`. Post-review completion status,
agreement statistics, and the unsampled-record boundary are reported in
`docs/REGISTRY_REVIEW_COMPLETION_ADDENDUM.md`.

The cotton apparel and homeware stages preserve the pre-audit, precision-oriented form ontology
around representative basic forms: five knitted and three non-knitted apparel HS6s, plus bed and
kitchen/toilet linen, all with explicit cotton attribution. The subset was not selected using trade
values or model outcomes and is not an exhaustive claim over Chapters 61 to 63.

The canonical code-level decisions are `chains/evidence/registry_evidence.json`; the generated
machine-readable and human-readable integrity reports are `docs/registry_audit.json` and
`docs/REGISTRY_AUDIT.md`. Any accepted change to a code decision, stage membership or assignment,
stage definition, or semantic `upstream`, `upstream_map`, `derived_from`, `derived_from_hs`,
`produces`, `form_of`, or `named_sources` field creates a new registry version and requires a full
rebuild of registry-dependent cohorts, results, summaries, paper interfaces, and release seals. A
sampled-validation receipt that accepts none of those changes adds only that co-bound receipt and refreshed
documentation/manifests; it does not require a computational rerun. The same no-rerun rule applies
to a documentation-only interpretation that leaves all of those construct fields unchanged.

In the hash-bound evidence/audit schema, `reviewed_codes=610` and the Markdown
`PASS` status are legacy compatibility labels for 610 chain--HS6 decision
records and internal source/mapping checks. The ledger covers 588 unique HS6
codes. These labels do not mean that 610 human row reviews have been completed.
The legacy machine field `review_date` is the generated-ledger audit date, not
a human-review completion date; likewise, `excluded_codes=228` counts
chain--HS6 exclusion decisions, corresponding to 207 globally unique excluded
HS6 codes.

Human validation is not a census of this machine ledger. The frozen design
selects 212 of the 610 decision records (34.8%) using certainty units plus a
deterministic, hash-ranked probability sample stratified by chain, decision,
and candidate source. All 53 stage definitions form a separate census. The
remaining 398 decision records are not individually human-verified. Because
inclusion probabilities differ, any full-frame discrepancy estimate must be
design-weighted; a simple-random-sample error bound does not apply. The 212-row
target is a design choice, not an academic threshold such as 25%.

The separate `chains/evidence/registry_curation_protocol.json` binds the curation process to the
evidence hash. LLM assistance was used to develop supporting code, the frozen lexicons, initial
decision proposals, and partitioned rechecks of the generated ledger. Those rechecks corrected six
boundary records but were neither human review nor independent replication. Release-valid
human-validation status comes only from the canonical hash-bound receipt and its retained sampled
row-level records. Two reviewers completed the sampled validation and final adjudication, and the
canonical receipt records no accepted construct change. This remains sampled validation rather than
a census of the other 398 decision records or an independent replication. Automated source,
relation, and semantic-regression checks detect drift; they do
not convert a curatorial judgment into independently replicated ground truth. The same sidecar
records the purposive chain criteria and confirms that formal main-cohort model performance was not
a chain inclusion signal.

The current documented strict-registry snapshot contains six chains, 53 active
stages, and 283 included HS6 codes. Frozen regexes were applied automatically
to all 5,022 source rows, producing 576 observable chain--HS6 records: 283
included, 194 excluded, and 99 out of stage. Adding 34 legacy-only provenance
records gives 610 chain--HS6 decision records (588 unique HS6 codes): 283 included, 228
excluded, and 99 out of stage. The 228 excluded figure counts chain--HS6 decision records; after
deduplication across chains, 207 globally unique HS6 codes have at least one excluded decision. The
audit records 19 retained-code stage
reassignments. These are generated construct-audit counts, not completed human
reviews and not by themselves evidence of release resolution; completed-review and resolution
status come from the canonical sampled-validation and `RESOLVED` receipts.

## 3. Ex-ante size baselines

- Track A lane size: processed-stage exporter capacity plus processed-stage importer demand.
- Track B1 entry size: registered-upstream exporter capacity, summed over the raw and/or
  intermediate stages declared in `upstream_map[s]`. It must not reuse processed-stage exporter
  volume, which is identically zero by construction.
- Track B2 lane size: registered-upstream exporter capacity plus processed-stage importer demand.

All components are computed from the early calendar window using the aggregation in Section 1.

The canonical product-space density is an additional **B1-only** domain reference. For each early
window, revealed-comparative-advantage membership and symmetric product proximity are constructed
over the complete HS92 product dictionary, not only the six-chain registry union. A candidate
exporter--stage score is the mean density of the stage's registered target products, excluding each
target product's self-relation from both numerator and denominator. The formula, full-economy early
scorers, and all registry stage mappings must be frozen before either historical or main B1 outcome
tables are opened. Labels are not used for selection or calibration. This association score is not
a causal estimate of productive capability and cannot rank A or B2 destinations.

## 4. Evaluation protocol

The primary protocol is forward and label-frozen, with versioned rolling updates supported when a
later source vintage supplies a new complete target window:

1. Use the historical fold `1998-2002 -> 2008-2012` to select model classes, hyperparameters, and
   supervised combiner weights.
2. Freeze those choices.
3. Refit label-free graph representations on the main early graph `2008-2012` where required.
4. Evaluate once on the complete main target cohort `2018-2022` without using its labels for model
   selection, feature fitting, imputation, or calibration.

An external-pretrained zero-shot reference may instead fix a public-provenance checkpoint before
target scoring, provided that no UpgradeBench label is used for checkpoint choice, training,
fine-tuning, calibration, or score transformation. Reusing the main early graph as inference context
is allowed but must be disclosed: this is no benchmark-label adaptation, not graph-free cold start.
Candidate-only score generation must finish and seal every prescribed chain before target-label
columns are parsed for evaluation. Complete candidate files may previously undergo deterministic,
outcome-blind byte-level hashing for provenance; this does not decode or use the labels. For the
formal ULTRA-ZS reference, the gate requires 6/6 sealed chain score components,
18/18 chain--task evaluation records, and the prespecified sheep same-process repeat across both A+B
score files and all six derived metrics.

External checkpoint results must be reported separately from benchmark-trained multi-seed families.
Unless training data and compute are matched, they are descriptive references rather than fair-compute
rankings. One fixed checkpoint receives a point estimate, not a fabricated training-seed interval;
chain-direction summaries do not establish a champion, statistical significance, population
generalization, or causality.

An additional tabular GBDT reference may use the same early-window task features as the logistic
references. Its estimator grid, minimum leaf sizes, historical objectives, group units, budgets,
bootstrap, and random seed must be frozen in a public configuration before main evaluation. A/B1
select by exporter-group historical AP; B2 selects by positive-entry-group macro recall@3. Every
chain--task model must be selected and refit before any main candidate file is opened. B2 must score
all main lanes before the realized-entry condition is applied for metrics. This reference is
reported as reviewer-motivated coverage outside the original prespecified reference set; it cannot
be used for post-hoc target-window champion selection.

The product-space B1 reference follows a stricter no-selection read gate: both temporal windows'
full-economy early scorers and all six target-stage mappings are built and frozen before any B1
outcome table is opened. Its public keyed scores may contain only released candidate identities,
the frozen label-free density, released outcomes needed for metric recomputation, and hash bindings;
the raw BACI archive is not a public payload.

Same-window candidate splits are diagnostic only and must be named `transductive`. When such a
diagnostic split is used, the default grouping key is `(exporter, stage)` so one capability/entry
group cannot cross development and test. Additional exporter-disjoint and importer-disjoint slices
are required.

Official split assignments are materialized in release tables or a keyed split manifest. No script
may silently construct an alternative `RandomState`, MD5, or row-order split.

When target labels are distributed publicly, a submission's self-attestation and hashes prove
schema/provenance consistency but cannot independently prove that the submitter did not inspect
test labels. Blind leaderboard claims therefore require a separately held-out or hosted evaluation;
local public-label evaluation is a reproducibility benchmark, not a leakage certificate.

## 5. Metrics and uncertainty

- Track A: average precision (AP), recall/precision at budget, value-captured at budget, and per-exporter shortlist
  metrics are primary; within-size-bin AUC is a structural diagnostic.
- Track B1: average precision (AP), recall/precision at an exporter-stage budget, and observed-late-value capture at
  that budget are primary.
- Track B2: per-entry recall at `k` and value-captured at `k` are primary.
- Use `realized late value`; causal-sounding terms such as `realizable value` are prohibited.
- Confidence intervals resample exporters for Track A/B1 and exporter-stage entry groups for Track
  B2, not individual dyadic rows.
- Model-family selection must use historical/development data; no downstream metric may be reported
  after selecting the model on the same test labels.
- Paired graph-family intervals use identical candidate rows, fixed seeds, and cluster
  multiplicities for both families. Chains remain the six declared benchmark chains and are never
  resampled; an aggregate interval therefore describes this finite benchmark only and is not
  population inference over value chains.
- Alternative B1 score pooling, budget sweeps, and two-stage B1-to-B2 diagnostics are sensitivity
  analyses. They must report every declared arm, cannot replace the official raw-max B1 score, and
  cannot be used for target-window method selection. A missed B1 entry contributes zero to the
  two-stage destination denominator.

## 6. Artifact identity and provenance

- Version-tagged paths, schema identifiers, and the `2.1-dev` string are artifact/API compatibility
  surfaces; they do not define separate scientific stories.
- Candidate tables remain in a dedicated output root until all schema, count, audit, and metric
  checks pass.
- Every released table includes `benchmark_version`, `aggregation`, `window`, and an explicit split
  assignment or keyed split-manifest reference.
- A formal external-pretrained result is public only through a sanitized JSON/CSV pair whose
  public-only verifier checks record completeness, aggregates, comparison qualifiers, source hashes,
  and repeatability gates without opening private scores, checkpoints, or raw provenance.
- The formal GBDT result is public only as `results_v2/metrics/v2_gbdt_baselines.{json,csv}` with
  `configs/v2_gbdt_baselines.json` and its verifier. The JSON must bind the frozen grid, all
  historical selection traces, main read-gate flags, 200-draw cluster uncertainty, runtime, public
  source hashes, and all 24 candidate files; the CSV must be deterministically regenerated from it.
- The product-space result is atomic across
  `results_v2/metrics/v2_product_space_density.{json,csv}` and
  `results_v2/scores/v2_product_space_density_scores.csv`. Its public verifier must recompute the
  metrics and uncertainty from the keyed scores without opening raw BACI; full maintainer
  verification additionally rebuilds RCA membership, proximity, and scores from the attested raw
  archive.
- `results_v2/metrics/v2_score_robustness_r5.{json,csv}` contains paired fixed-benchmark
  intervals, B1 pooling sensitivity, budget sweeps, and the two-stage diagnostic, but no formal
  model scores. Exact recomputation requires the private frozen graph score tree; those scores,
  selection artifacts, logs, and machine state remain outside the public release.
- `results_v2/metrics/v2_eligibility_threshold_geometry.{json,csv}` is a cohort-geometry artifact,
  not a model-result artifact. The JSON must bind the 50/100/250-kUSD protocol, raw-source receipt,
  registry and processed-table hashes, and the exact canonical-100-kUSD reconstruction gate; the
  CSV is a deterministic projection.
- `results_v2/metrics/v2_benchmark_profile.json` and
  `paper/generated/v2_benchmark_profile.tex` are a separate, post-resolution aggregate interface,
  not additional members of the governed paper-number result map. The current profile and
  paper-number interfaces are receipt-bound and verified independently. Profile artifacts may expose graph-scale,
  effective-sample, and compute aggregates plus public-source and private-evidence hashes, but no
  private claims, inventory, receipts, paths, logs, or raw/formal contents. Public repository
  verification is `python tools/generate_v2_benchmark_profile.py --verify --profile repository`;
  rebuilding and maintainer-only full provenance verification require explicitly supplied private
  evidence, which must not be copied into either public artifact.
- Paper figures and headline numbers may be generated only from artifacts authorized by the current
  claim-to-artifact ledger.
