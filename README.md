# UPGRADE-BENCH

UPGRADE-BENCH is a value-aware, temporally forward benchmark for studying when new
processed-export relationships emerge across six global value chains. It provides three linked
prediction tasks, leakage-audited temporal evaluation, reference CPU and graph baselines,
observed-value diagnostics, and hash-bound artifact verification. The contribution is the benchmark
and its audited evaluation contract, not a proposed model.

> **Verified development snapshot.** This repository state is not yet the final published bundle. The
> observable-attribution revision defines a registry with 283 included
> HS6 codes. A public release must co-seal two independent branches: the complete current-registry
> computation/result branch and a hash-bound sampled human-validation receipt. Listing these gates
> does not assert an execution order. The canonical receipt records no accepted construct change,
> and `results_v2/metrics/INVALIDATED.json` now carries the verified `RESOLVED` transition that
> co-binds both branches and the current paper-number interface. A future accepted construct change
> requires a new registry/benchmark version and a full registry-dependent rebuild.
> Publication, remote CI, authorship/order, venue/cycle metadata, hosting, DOI, and final release
> metadata are separate release obligations.

The machine-readable claim boundary is documented in
[`results_v2/CLAIM_LEDGER.md`](results_v2/CLAIM_LEDGER.md). Some machine-interface paths retain
versioned identifiers for provenance, but the scientific artifact is one UpgradeBench project.

## Prediction tasks

Every task constructs candidates and features from an early trade window and evaluates outcomes in
a later, non-overlapping window.

- **Track A — destination extension.** Unit: `(exporter, processed stage, importer)`. Rank new
  destinations for an exporter already active in the processed stage.
- **Track B1 — eligible-market processed-stage entry.** Unit: `(exporter, processed stage)`. Predict
  whether an upstream-qualified exporter with no early processed-stage export reaches at least one
  destination that already imported the processed stage.
- **Track B2 — destination ranking conditional on entry.** Unit:
  `(exporter, processed stage, importer)`. Among B1 groups that enter in the late window, rank the
  destination lanes they reach.

Track B is an export-based emergence proxy. It does not identify domestic production capacity or an
intervention effect. B2 is positive-entry-conditioned rather than an end-to-end entry system.

## Evaluation contract

For each lane and stage, registered HS6 products are summed within each calendar year and averaged
over all five years in the window, including zero-flow years:

```text
stage_value[i,j,s,y] = sum_k value[i,j,k,y]
window_mean[i,j,s]   = sum_y stage_value[i,j,s,y] / number_of_calendar_years
```

The 100 kUSD threshold is fixed in nominal USD. Inflation sensitivity is therefore a limitation,
not something hidden by the aggregation rule.

Model classes, hyperparameters, imputers, and supervised weights are selected on the historical
`1998–2002 -> 2008–2012` snapshot, frozen, and then evaluated once on the complete
`2008–2012 -> 2018–2022` target cohort. Same-window dev/test splits are diagnostic only and keep
complete exporter or exporter-stage groups together. The normative task, split, aggregation, and
metric definitions are in [BENCHMARK_V2_SPEC.md](BENCHMARK_V2_SPEC.md).

## Evidence and result boundaries

The current registry comes from an automated frozen-regex scan of all 5,022
BACI HS92 product-description rows. It produced 576 observable chain--HS6
records; 34 legacy-only provenance records bring the ledger to 610 records over
588 unique HS6 codes. Observable decisions are 283 included, 194 excluded, and
99 out of stage; full-ledger decisions are 283 included, 228 excluded, and 99
out of stage. The outcome-blind human-validation design freezes a stratified
sample of 212 decision records (34.8% of the 610-record machine-assigned frame)
and a census of all 53 stage definitions. The other 398 decision records are
not claimed as individually human-verified. Two reviewers completed the sampled
validation, all 53 definitions were accepted, and final adjudication accepted no
construct change. Release-valid status is established by the canonical hash-bound
sampled-validation and resolution receipts. The
finite lexical negative controls test declared variants but do not prove
lexicon completeness.

Unresolved focal attribution is excluded. Commodity-explicit blend and residual codes are retained
as whole published HS6 baskets, but their inclusion does not identify focal-material share, purity,
physical composition, or dominant feedstock; claims concern commodity-explicit HS6 baskets and
registry-defined analytical families.

The replacement CPU, graph, robustness, raw-label, coverage, and value-diagnostic work is
source-hash bound, checked against the 283-code cohorts, and co-bound by the canonical resolution
receipt. The value diagnostic adds
fixed-budget observed-value capture and same-budget descriptive comparisons. Its
target-outcome-ranked comparator is an oracle diagnostic: it is not deployable, causal, or a
realizable policy ceiling. Track B2 value is nested within Track B1 entry value and must not be added
to Track A/B1 totals.

The current `results_v2/metrics/v2_benchmark_profile.json` and
`paper/generated/v2_benchmark_profile.tex` are intentionally absent from this development snapshot.
No graph scale, sample-unit, or compute claim should be inferred from their absence. The current
profile may be promoted only after the
active hold is resolved and its repository and private-provenance checks pass.

The canonical product-space density remains a B1-only, reviewer-motivated domain reference. For
each early window it constructs country--product RCA membership over the complete HS92 dictionary,
computes symmetric product proximity, excludes each target product's self-relation, and averages
density over the registered products in the candidate stage. The replacement artifact and exact
keyed-score CSV have been rebuilt for the 283-code cohorts and pass their local verifier. Likewise,
the replacement paired graph-score artifact uses identical candidate sets, fixed seeds, and cluster
multiplicities for PyKEEN and NBFNet and passes its local verifier. Their numerical results are
deliberately not promoted here until the complete replacement paper interface and release receipt
exist.

A separate raw-derived cohort-geometry audit rebuilds eligibility and labels at 50, 100, and
250 kUSD. Its 100-kUSD reconstruction exactly matches every canonical A/B1/B2 key and label. The
alternative thresholds change both candidate and positive sets substantially, especially for B2;
the artifact intentionally reports cohort counts and overlap only, not reused model scores or a
performance sensitivity. The 100-kUSD benchmark definition remains unchanged.

The GBDT reference uses `HistGradientBoostingClassifier` with the same task-aligned early features
as the logistic references. Its replacement 18-model artifact was selected and refit under the
frozen historical protocol, rebuilt for the 283-code cohorts, and passes its local verifier. It
remains a reviewer-motivated reference outside the original prespecified set, not a post-hoc model
champion, and its values await promotion through the replacement paper interface.

Matched LOCO and ULTRA-ZS remain release gates. The LOCO design remains a tier-abstracted
NBFNet parameter-transport diagnostic over six held-out chains, two modes, and five seeds; ULTRA-ZS
remains a fixed external-pretrained, no-benchmark-label-adaptation reference with unmatched external
training data and compute. No current LOCO or ULTRA aggregate is promoted in this development
snapshot. Only receipt-bound current-registry summaries may support a result, direction, or model
comparison in this README or the paper.

## Repository map

- [chains/](chains/) — six-chain registry, stage definitions, HS6 membership, and upstream links.
- [chains/evidence/registry_evidence.json](chains/evidence/registry_evidence.json) — evidence for
  include/exclude decisions and canonical stage definitions.
- [chains/evidence/registry_curation_protocol.json](chains/evidence/registry_curation_protocol.json)
  — hash-bound LLM-assistance disclosure, outcome-blind sampled human-validation contract,
  chain-selection criteria, controls, and revision policy.
- [chains/evidence/registry_human_validation_sample.json](chains/evidence/registry_human_validation_sample.json)
  — frozen 212-record stratified sample, exact inclusion probabilities, and stage-definition census.
- [docs/REGISTRY_REVIEW_CODEBOOK.md](docs/REGISTRY_REVIEW_CODEBOOK.md) — operational inclusion,
  analytical-family, and sampled-validation rules for the 610-record machine ledger
  (588 unique HS6 codes).
- [docs/V2_RELEASE_WORKFLOW.md](docs/V2_RELEASE_WORKFLOW.md) — fail-closed dependency graph for
  formal result replacement, outcome-blind review branching, paper-interface review, and release
  sealing.
- [docs/REGISTRY_AUDIT.md](docs/REGISTRY_AUDIT.md) — hash-bound internal source/mapping report and
  rebuild boundary; its legacy `PASS`/`reviewed_codes` labels do not claim completed human review.
- [chains/evidence/registry_lexicon_negative_control.json](chains/evidence/registry_lexicon_negative_control.json)
  — empirical checks of declared lexical variants; passing them does not prove lexicon completeness.
- [tools/build_registry_revision.py](tools/build_registry_revision.py) and
  [tools/build_registry_lexicon_negative_control.py](tools/build_registry_lexicon_negative_control.py)
  — current registry and lexical-negative-control generators; byte-for-byte regeneration requires
  the pinned external BACI archive.
- [chains/evidence/registry_evidence_pre_full_dictionary.json](chains/evidence/registry_evidence_pre_full_dictionary.json)
  — proposal provenance needed to recover the 34 legacy-only ledger records; it is not an active
  registry or result artifact.
- [src/window_aggregation.py](src/window_aggregation.py) — canonical calendar-window aggregation.
- [src/temporal_backtest.py](src/temporal_backtest.py) — candidate and feature construction.
- [src/split.py](src/split.py) — stable group-safe diagnostic splits.
- [src/task_features.py](src/task_features.py) — task-specific early-window features.
- [tools/build_v2_views.py](tools/build_v2_views.py) — materializes B1/B2 views and metadata.
- [configs/v2_loco_formal.json](configs/v2_loco_formal.json) — source-locked, host-neutral formal
  tier-abstracted matched-LOCO design; operational claims/logs remain private.
- [configs/v2_ultra_formal.json](configs/v2_ultra_formal.json) — frozen external-pretrained
  zero-shot protocol; checkpoint, scores, receipts, and formal run state remain private.
- [configs/v2_gbdt_baselines.json](configs/v2_gbdt_baselines.json) — frozen reviewer-motivated
  GBDT grid, task features, historical selection objectives, main read gate, budgets, and bootstrap.
- [configs/v2_product_space_density.json](configs/v2_product_space_density.json) — frozen B1
  product-space formula, source universe, read gate, ranking, and uncertainty contract.
- [configs/v2_score_robustness_r5.json](configs/v2_score_robustness_r5.json) — fixed-seed paired
  graph comparison, pooling alternatives, budgets, and two-stage diagnostic contract.
- [configs/v2_eligibility_threshold_geometry.json](configs/v2_eligibility_threshold_geometry.json)
  — exact 50/100/250-kUSD cohort-rebuild and canonical-100-kUSD gate contract.
- `data/processed_v2/` — mount point for immutable main/history payloads intended for separate
  release assets, not ordinary Git objects.
- [results_v2/](results_v2/) — receipt-bound claim ledger and authorized sanitized summaries.
- [ARTIFACT.md](ARTIFACT.md) — verification and release-freeze procedure.

## Maintainer-only: edit and build a review PDF

This workflow is for the complete maintainer working tree used to edit the manuscript. A public
clean export must include the selected manuscript dependencies needed to compile the paper or keep
the review builder outside the public release manifest.

Edit the manuscript sources directly; do not edit `paper/generated/*.tex`:

- author, email, affiliation, title, and venue metadata: `paper/main-acm.tex`;
- abstract: `paper/abstract.tex`;
- main text: `paper/body.tex`;
- acknowledgments: `paper/acknowledgments.tex`;
- appendix: `paper/appendix.tex`;
- references: `paper/refs.bib`;
- figures: `paper/figures/`.

Create or refresh the sealed review-number snapshot when a release audit must bind the paper to the
current data, metrics, paper-number sources, and verification code. While the invalidation hold is
active, the snapshot comes from the verified non-canonical preview; after resolution, it copies the
receipt-verified canonical JSON/TeX bytes:

```powershell
& .\.venv\Scripts\python.exe tools\build_paper_review.py --refresh-numbers
```

The full refresh can take several minutes. Normal wording, author, bibliography, and layout changes
use the fast path:

```powershell
& .\.venv\Scripts\python.exe tools\build_paper_review.py
```

The command validates the sealed snapshot's receipt, bytes, JSON/TeX projection, number digests,
the benchmark profile, and the independent contemporary-reference JSON/CSV/TeX interface; builds
in a fresh directory below `tmp/pdfs/`; and atomically writes
`output/pdf/upgrade-bench-kdd27-review.pdf`. The ignored snapshot is stored below
`output/paper-review-cache/`. If its governed binding is older than the current repository state, the
normal build emits a warning and continues with the intact sealed snapshot. Use
`--require-current-numbers` to turn that warning into a release-audit failure, or
`--refresh-numbers` to renew the binding. A missing, malformed, or modified snapshot still fails. The
builder never rewrites the canonical paper-number JSON/TeX files.

## Reproduce and verify

Create the locked CPU-results environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/v2-cpu-results-lock.txt
python requirements/verify_v2_cpu_results_env.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The lock records the source
platform and package versions; it does not promise byte-identical floating-point output across
platforms or BLAS/OpenMP implementations. See [LOCAL_SETUP.md](LOCAL_SETUP.md).

That lock is only the environment record for the replacement forward-CPU artifact; it intentionally
does not install the graph dependencies imported by the complete repository test collection. Use a
separate test environment for the full checks below:

```bash
python -m venv .venv-tests
. .venv-tests/bin/activate
python -m pip install -r benchmark/upgrade-bench-v2/requirements.txt \
  scikit-learn==1.9.0 pycountry==26.2.16
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric==2.7.0
```

On Windows PowerShell, activate that environment with `.venv-tests\Scripts\Activate.ps1`.

The repository is a verified development snapshot with a resolved scientific claim interface.
Release-smoke, clean-clone, remote-CI, packaging, and publication gates remain separate. After
installing the main and historical bundles as documented in
[docs/DATA_DISTRIBUTION.md](docs/DATA_DISTRIBUTION.md), run the currently applicable scientific
checks:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tools/build_registry_evidence.py --check
python tools/audit_chain_registry.py --check
python tools/verify_registry_curation_protocol.py
python tools/validate_v2.py
python tools/audit_v2.py --verify-output
python tools/v2_b1_coverage.py --verify-output
python tools/v2_rolling_cpu_baselines.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/v2_product_space_density.py --verify-output
python tools/v2_eligibility_threshold_geometry.py --verify-output
python tools/summarize_v2_gpu_results.py --verify-output
python tools/v2_value_diagnostics.py --verify-output
```

The product-space command recomputes all metrics from its keyed scores without opening the
raw BACI archive. Add `--verify-raw` only in a private provenance checkout to rebuild the early RCA
memberships, proximities, and candidate scores from the attested raw archive. The threshold-geometry
verification checks the replacement pair, registries, and canonical processed tables; generating it
again requires the raw archive.

The paired graph robustness verifier recomputes from the private formal score tree and is therefore
a maintainer-provenance check, not a code-only public-clone command:

```bash
python tools/v2_score_robustness_r5.py --verify-output
```

The eventual public release will carry only its score-free JSON/CSV analysis pair, public config and runner;
formal graph scores, selection artifacts, logs, and machine state remain excluded.

The robustness verifier also binds the private raw BACI archive and is intended for a private
provenance checkout:

```bash
python tools/v2_robustness.py --verify-output
```

The LOCO, ULTRA, benchmark-profile, paper-number, and public-receipt outputs are admitted by the
canonical resolution receipt. Re-run their read-only verifiers after any governed artifact change;
see [ARTIFACT.md](ARTIFACT.md) and [docs/DATA_DISTRIBUTION.md](docs/DATA_DISTRIBUTION.md).

## Git repository versus data assets

The public Git repository is intended to contain current code, documentation, tests,
portable job templates, manifests, the source-locked formal configurations, and only those
lightweight sanitized summaries authorized by the replacement receipt.
Maintainer-only packages and exploratory output trees are explicitly outside the public selection.
The processed main/history tables under `data/processed_v2/` are immutable external
payloads covered by `release/DATA_ARTIFACT_INDEX.json`; they belong in versioned GitHub Release
assets and, at final publication, a DOI-bearing archival mirror.

Raw third-party archives, filtered BACI caches, formal LOCO/ULTRA score trees, logs, checkpoints, host
inventory, and permission-gated institutional extracts are never public payloads. A maintainer
staging checkout may contain private files or reachable private history and must not be pushed as
the public repository. The supported preflight creates a fresh one-commit tree from the exact
public manifest inventory:

```bash
python tools/release_clean_clone.py --profile repository
python tools/release_clean_clone.py --profile full --artifacts-dir dist/final-release
```

These commands verify a candidate export; they do not upload it. Source versions, attribution, and
redistribution constraints are recorded in [DATA_LICENSE.md](DATA_LICENSE.md),
[requirements/DATA.md](requirements/DATA.md), and
[docs/PUBLIC_RELEASE_POLICY.md](docs/PUBLIC_RELEASE_POLICY.md).
