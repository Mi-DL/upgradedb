# Datasheet: UPGRADE-BENCH (`2.1-dev` artifact compatibility version)

## Purpose and status

UPGRADE-BENCH studies whether graph and ex-ante trade-structure signals help
rank future processed-export emergence and destination formation. It defines
calendar-complete aggregation, three task-specific prediction units, ex-ante
size features, and historical selection followed by frozen main evaluation.
The `2.1-dev` string and version-tagged paths are artifact/API compatibility
identifiers, not separate scientific stories.

This standalone package supplies validation and evaluation code. The large CSV
tables are separate immutable artifacts, so installing the code does not imply
acceptance of or access to third-party source-data terms.

The current registry is defined by the strict audit and evidence artifacts. The
frozen chain lexicons were applied automatically to all 5,022 rows of the
pinned BACI HS92 product dictionary, yielding 576 observable chain--HS6
records. Adding 34 legacy-only provenance records produces a 610-record audit
ledger covering 588 unique HS6 codes. Among the observable records, 283 are
included, 194 excluded, and 99 out of stage; across the full ledger the counts
are 283 included, 228 excluded chain--HS6 decision records, and 99 out of stage.
After deduplication across chains, 207 globally unique HS6 codes have at least one excluded decision.
The 283 included codes map
to 53 active stages, with 19 retained-code stage reassignments. This was a
5,022-row automated regex scan followed by generated rule application to the
610 ledger records; it does not constitute 5,022 or 610 human reviews. The separate
negative-control artifact tests declared synonym and spelling variants but
does not prove lexicon completeness. The frozen outcome-blind human-validation
design selects 212 decision records (34.8% of the 610-record frame), combining
declared certainty units with deterministic hash-ranked probability selection
within chain-by-decision-by-source strata. All 53 stage definitions are a
separate census; the remaining 398 decisions are not individually
human-verified. Because selection probabilities differ, full-frame discrepancy
estimates require the stored design weights rather than simple-random-sample
bounds. No fixed fraction, including 25%, is an academic sufficiency criterion;
the 212-row target is design-specific. Two reviewers completed the sampled validation, all 53 stage
definitions were accepted, and final adjudication accepted no construct change. Registry decisions use the official BACI HS92 description as
the sole adjudicative evidence. Commodity-explicit blends and residual or `n.e.s.` baskets may be
included at whole-HS6-basket value; generic, negated, mixed-species, or unresolved-material
descriptions are excluded, and an explicitly focal product outside the frozen ontology is recorded
as out of stage. External knowledge may clarify terminology but cannot supply absent attribution or
stage fit. The cotton apparel and homeware stages preserve the pre-audit, precision-oriented form
ontology around representative basic forms: five knitted and three non-knitted apparel HS6s, plus
bed and kitchen/toilet linen, all with explicit cotton attribution. The subset was not selected
using trade values or model outcomes and is not an exhaustive claim over Chapters 61 to 63.

Release eligibility co-seals two independent branches: the complete 283-code computation/result
evidence and a canonical hash-bound outcome-blind sampled human-validation receipt. Their presentation does not assert
an execution order. A no-construct-change receipt leaves the bound computations applicable; an
accepted construct change requires a new registry/benchmark version and full registry-dependent
rebuild before resealing. The current sampled-validation and resolution receipts record no accepted
construct change and bind the governed result interface. The fixed review instrument excludes trade values, labels, cohort impacts,
model scores, and downstream result summaries. Publication, DOI, authorship, venue/cycle metadata,
and other administrative metadata are separate release gates.

## Instances

The six registered chains are sheep, cotton, aluminium, nickel, cocoa, and
oilseed/soy. Each chain has a historical selection snapshot and a main target
snapshot:

| Snapshot | Early inputs | Late outcomes | Role |
|---|---|---|---|
| `fold2` | 1998–2002 | 2008–2012 | historical model/hyperparameter selection |
| `main` | 2008–2012 | 2018–2022 | frozen target evaluation |

Replacement 283-code main-snapshot CPU cohort totals across the six chains are:

| Task | Units | Positives | Observed late value |
|---|---:|---:|---:|
| Track A lanes | 317,624 | 12,273 | 16,862,139.3308 kUSD |
| Track B1 eligible-market entries | 1,518 | 270 | 1,623,471.7642 kUSD |
| Track B2 conditional eligible-market lanes | 33,433 | 556 | 1,623,471.7642 kUSD |

Replacement 283-code historical-snapshot CPU cohort totals are:

| Task | Units | Positives | Observed late value |
|---|---:|---:|---:|
| Track A lanes | 238,176 | 15,627 | 19,842,889.5220 kUSD |
| Track B1 eligible-market entries | 1,392 | 298 | 2,009,698.8436 kUSD |
| Track B2 conditional eligible-market lanes | 33,722 | 639 | 2,009,698.8436 kUSD |

These replacement counts and values come from `dataset_summary.json` and
`dataset_summary_fold2.json` in the processed-v2 payload. The repository
validator has checked 48 rebuilt tables/views: 24 base lane tables (A and pre-view B for six
chains and two snapshots) and 24 derived B1/B2 views. The 36 A/B1/B2 views are
the evaluation objects exposed by the standalone loader. Their local verification does not by
itself unblock result claims or public release.

## Unit and label definitions

**Track A** has one row per `(exporter, processed stage, importer)`. The exporter
has flow in at least one registered upstream stage and already has established
processed-stage exports in the early window; the lane to the candidate importer
is new. `y=1` when that late lane exceeds the release threshold. It is destination
extension by an incumbent processed-stage exporter.

**Track B1** has one row per unique `(exporter, processed stage)`. The exporter
has a registered upstream-stage flow but no established processed-stage export
in the early window. Eligible destinations already import the processed stage
in the early window. `z=1` when at least one such destination lane materializes
late; this is not an exhaustive indicator of exports to previously inactive
markets.

**Track B2** contains the candidate destination lanes only for B1 groups with
`z=1`. Metrics rank destinations separately inside each entry group. Repeating
one positive entry across candidate destinations does not create multiple entry
events.

These are eligible-market processed-export emergence proxies. They do not
establish domestic factory capacity, domestic value added, welfare effects, or
a causal effect of any policy.

“Registered upstream” means the raw and/or intermediate predecessor stages
explicitly listed in the chain registry's `upstream_map` for the target stage.
It does not necessarily mean a raw commodity.

## Eligible-market coverage boundary

The B1 early-demand screen is pre-label and makes destination enumeration
finite, but it is not a global universe of late starts. A separate audit
reconstructed all late processed-stage starts by early upstream-qualified
nonincumbent exporters from raw BACI, then reconciled released lane and entry
identities, labels, and values exactly.

| Snapshot | All realized entries | Covered entries | Inactive-only entries | All late-start lanes | Eligible lanes | Previously inactive lanes | All late-start value | Eligible value | Value coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fold2` | 316 | 298 | 18 | 724 | 639 | 85 | 2,061,337.1208 kUSD | 2,009,698.8436 kUSD | 97.49% |
| `main` | 280 | 270 | 10 | 578 | 556 | 22 | 1,628,610.3160 kUSD | 1,623,471.7642 kUSD | 99.68% |

Entry coverage is 94.30% in `fold2` and 96.43% in `main`; lane coverage is
88.26% and 96.19%, respectively. A covered entry can also contain starts into
previously inactive markets, so the entry, lane, and value denominators answer
different questions. B1/B2 results apply only to the released eligible-market
cohort and must not be described as exhaustive first-export prediction.

## Aggregation and threshold

Registered HS6 values are first summed to `(exporter, importer, stage, year)`.
The stage total is then averaged over every calendar year in the five-year
window; missing BACI rows contribute zero. A lane is established/materialized
when its window mean exceeds 100 kUSD. `lateval` is the materialized late mean
and is zero for negative rows.

The cutoff is fixed in nominal USD; it is not deflated or normalized by world
trade scale. It therefore does not have constant real economic meaning across
the historical and main periods. The legacy robustness artifact's alternate
outcome thresholds relabel its fixed candidate cohort and are not full
threshold-specific rebuilds. The separate eligibility-geometry artifact does
rebuild early eligibility and late labels from annual raw inputs at 50, 100,
and 250 kUSD; its 100-kUSD arm exactly matches every released A/B1/B2 key and
label. It reports cohort counts and key overlap only, not model performance or
constant-dollar sensitivity, and does not change the default threshold.

The sequence “stage-by-year sum, then full-calendar mean” is normative. An
active-year persistence field is not present in `2.1-dev` and must not be
inferred from these window tables.

## Files and schemas

For every chain, the main files are:

- `candidates_<chain>.csv` — Track A;
- `candidates_firsttime_<chain>.csv` — pre-view Track-B lane construction and
  raw-label-audit input;
- `entries_firsttime_<chain>.csv` — Track B1; and
- `destinations_given_entry_<chain>.csv` — Track B2.

Insert `_fold2` before `.csv` for the historical snapshot. The scorer requires
the exact development schema. Identity and outcome fields include:

| Track | Exact key | Label | Realized late value |
|---|---|---|---|
| A | `i_iso,j_iso,stage` | `y` | `lateval` |
| B1 | `i_iso,stage` | `z` | `entry_lateval` |
| B2 | `i_iso,j_iso,stage` | `y` (`entry_y` is always 1) | `lateval` |

Every table also records `benchmark_version`, `aggregation`, early/late windows,
temporal role, task name/unit, and the exporter-stage-grouped diagnostic split.
Track-A `size` is processed exporter capacity plus processed importer demand;
Track-B1 `size` is registered-upstream exporter capacity; Track-B2 `size` is
registered-upstream exporter capacity plus processed importer demand.

`grav` and `gnn` are optional ex-ante score fields and may be missing in a
development payload. Missing scores are never silently imputed by the official
evaluator. `lateval`, `entry_lateval`, labels, and materialized-destination
counts are outcomes and must never enter a model score.

## Recommended protocol

1. Use `fold2` labels for model-family, hyperparameter, and supervised combiner
   selection.
2. Freeze all choices.
3. Refit only label-free representations on the main early graph if needed.
4. Score the entire released, track-specific main candidate cohort without
   using main labels for selection, fitting, imputation, or calibration. B1/B2
   remain limited to the eligible-market cohort defined above.
5. Attach a schema-checked self-attestation to every external main run, binding
   the exact benchmark CSV, score CSV, frozen selection config, run ID, and seed
   list.

A fixed external-pretrained checkpoint may be reported as a separate zero-shot reference when it is
chosen before target scoring and receives no UpgradeBench-label training, checkpoint search,
fine-tuning, calibration, or score transformation. If it uses the target early graph, report that
explicitly: this is no benchmark-label adaptation rather than graph-free cold start. Candidate-only
scores must be sealed before labels are opened. The formal ULTRA-ZS gate requires 6/6 sealed chain
components, 18/18 chain--task records, and the sheep same-process repeat over both A+B score files and
all six derived metrics.

The self-attestation is not an independent audit of historical behavior. The
evaluator checks its exact schema and recomputes the three raw-byte SHA-256
values; statements about label use and freezing remain submitter declarations.
An invalid/mismatched attestation is rejected. An explicit diagnostic override
is the only bypass and always yields `official: false`.

Because the reproducibility package distributes main labels, this mechanism
cannot support an independently blind public leaderboard. Enforced blind
comparison requires a hosted evaluator or a separately held-out future cohort.

Same-window `train/test` assignments are label-independent SHA-256 hashes over
`(chain, exporter, stage)` and keep every exporter-stage group intact. They are
diagnostics only and should be named transductive. The standalone scorer does
not silently subset the official temporal cohort by this field.

## Evaluation

Track A reports average precision, shortlist precision/recall/realized-value
capture, within-size-bin AUC as a diagnostic, and per-exporter shortlists at 5
and 10. For exporter shortlists, macro precision includes every exporter;
macro recall and value capture include only exporters with a positive outcome,
while micro metrics pool all hits/value. With no positive exporter, recall/value
is undefined (`null`). Track B1 reports entry AP and entry shortlist metrics.
Track B2 reports per-entry macro recall and value capture at 1/3/5. Equal-score
shortlist ties use canonical identity keys, never CSV row order.

No uncertainty interval should resample individual lanes as if independent.
Repository experiments resample exporters for Track A/B1 and exporter-stage
entry groups for Track B2. The small
standalone scorer intentionally reports point estimates only.

The protocol definitions in this section remain normative. Numerical model and diagnostic claims
must come from the receipt-bound result artifacts rather than being inferred from method
descriptions or artifact paths alone.

Within the strict registry, commodity-explicit blends and residual categories may be retained at
whole-HS6-basket value, while generic or material-ambiguous categories are excluded;
`docs/REGISTRY_REVIEW_CODEBOOK.md` fixes this boundary.

The external-pretrained ULTRA-ZS reference uses the same task cohorts, tie rules, and budgets with one
fixed checkpoint. External pretraining data and compute are unmatched and the training seed is
undisclosed; it cannot establish a fair-compute winner, champion, statistical significance,
population generalization, or causality.

The GBDT reference uses the same early-window task features as the logistic references. Its fixed
four-point grid is selected only on grouped historical folds, all 18 chain--task models are selected
and refit before any main table is opened, and every B2 lane is scored before realized-entry
conditioning. It is a reviewer-motivated tabular reference, not a target-selected champion.

The product-space-density reference is B1-only. It constructs early-window RCA membership over the
complete HS92 dictionary, computes symmetric product proximity, excludes self-relations, and averages
density over registered products in the candidate stage. It is an association reference, not a
causal capability estimate and not an A/B2 destination ranker.

Paired graph robustness uses identical candidates, fixed seeds, and cluster draws for PyKEEN and
NBFNet. Chains are fixed rather than sampled, so intervals describe this benchmark rather than a
population of value chains. The linked B1-to-B2 cascade remains diagnostic and does not replace either
official task.

The 50/100/250-kUSD geometry audit rebuilds candidate and positive sets at each threshold. It reports
cohort geometry only; no model is rescored and outcome-only relabeling is not treated as a rebuilt
threshold cohort.

## Provenance, licensing, and privacy

The benchmark is derived from country/product-level trade and gravity inputs;
the authoritative release ledger records raw-source versions, hashes, and
redistribution decisions. Consult `DATA_LICENSE.md` and the documentation packed
with each data artifact before redistribution. The MIT license in this directory
covers the evaluator code and documentation, not third-party databases.

The benchmark uses CEPII BACI HS92 `V202401b`. The raw archive recorded by
the audits has SHA-256
`1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a`;
the registry decisions cite the official HS92 product descriptions distributed
in that release. Raw archives and the private filtered cache are not public
artifact contents.

LLM assistance was used to develop supporting code, the frozen lexicons, initial decision proposals,
and partitioned rechecks of the generated ledger. Those rechecks were neither human review nor
independent replication. All 5,022 source descriptions received the
automated regex scan; the resulting 576 observable records and 34 legacy-only
provenance records form the complete 610-record ledger (588 unique HS6 codes).
Release-valid sampled human-validation status comes only from the canonical hash-bound receipt and
its retained 212 sampled row-level records and 53 stage-definition records. The 398 unsampled
decisions are not individually human-verified. Lexicon completeness is supported only for the variants exercised by the
negative-control artifact and is not proved. The hash-bound
`chains/evidence/registry_curation_protocol.json` records that limitation, the purposive six-chain
inclusion criteria, the decision workflow, and the versioned correction policy. Semantic regression
tests protect the recorded mapping from drift; they are not evidence of independent semantic
agreement.

Any change accepted through sampled human validation or by maintainers to a code decision, stage membership or
assignment, stage definition, or semantic `upstream`, `upstream_map`, `derived_from`,
`derived_from_hs`, `produces`, `form_of`, or `named_sources` field creates a new registry/benchmark
version and requires a full rebuild of registry-dependent cohorts, results, summaries, paper
interfaces, and release seals. If the validation accepts no such change, its co-bound receipt and
refreshed documentation/manifests are added without a computational rerun.

The released units are country/stage/destination aggregates and are not intended
to contain personal information. Country codes include historical reporting
entities; consumers should not reinterpret them as present-day political or
legal assertions.

## Known limitations and appropriate use

- Labels reflect recorded international trade above a threshold and inherit
  reporting, mirror-flow, re-export, classification, and aggregation noise.
- Six registry-defined chains are not a representative census of industrial
  activity.
- The code-to-stage registry was developed with LLM assistance. Its 610 rule-application records are
  source-backed and fully enumerated; release-valid review evidence comes only from the canonical
  sampled-validation receipt. The completed two-reviewer validation covers 212 decisions and all 53
  stage definitions, leaving 398 decision records without individual human validation. The final
  adjudication accepted no construct change. That receipt does not by itself establish independent
  replication, and the finite lexical
  negative controls do not prove recall of every possible description wording.
- Included blends and residual HS6 baskets are valued in full; their internal
  focal-material shares are not observed. Analytical stage pooling can mask
  first entry into one constituent HS6 form.
- Threshold changes alter both eligibility and outcomes, and the released geometry audit does not
  establish that model ordering is invariant across rebuilt cohorts.
- Product-space density is an association-based B1 reference and may be unstable for small trade
  baskets; it does not identify productive capability or destination choice.
- Candidate-universe completeness is only relative to the audited registry and
  early-demand importer screen. Track B1 is not an exhaustive indicator of any
  first processed-stage export.
- The schema-checked main-run attestation binds exact bytes and rejects malformed
  declarations, but it is submitter self-attestation, not an independent audit
  of label use or a blind-leaderboard mechanism.
- The external-pretrained ULTRA-ZS reference uses each target early graph, comes from one checkpoint,
  and is not resource-matched to benchmark-trained references. It is neither graph-free cold start
  nor a compute-efficiency or significance comparison.
- Export emergence is not proof of domestic production or transformation.
- Rankings must not be used as automated eligibility, sanctions, credit, or
  investment decisions. Domain review and current external evidence are needed.
- `2.1-dev` is not the final signed release; author names, archival DOI, and
  final artifact version must be resolved before publication.

Report results by chain and task, preserve the snapshot/protocol distinction,
state the benchmark version, and disclose all diagnostics or overrides.
