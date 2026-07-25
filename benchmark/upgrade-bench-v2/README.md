# UPGRADE-BENCH standalone evaluator

This is the dependency-light loader and official scorer for UPGRADE-BENCH. The
`2.1-dev` string and version-tagged directory/API names are retained as artifact
compatibility identifiers; they do not define a separate scientific story. The
large candidate tables are deliberately not tracked in Git and are distributed
as separate GitHub prerelease assets, so this directory contains no duplicate
data payload.

> **Public reviewer-release status (2026-07-26).** The current registry contains 283 included HS6 codes.
> Replacement cohorts, formal references, the benchmark profile, the paper-number interface, and
> the sampled-validation receipt have been verified. The canonical
> `results_v2/metrics/INVALIDATED.json` receipt records the `RESOLVED` transition. Repository
> packaging, release-asset checksum verification, anonymous download checks, and remote CI have
> been completed for the public GitHub prerelease. This distribution is a reviewer release
> candidate, not the final archival release; its archival DOI and final release version remain
> pending.

The three evaluation objects are deliberately separate:

- **Track A — destination extension:** rank new importer lanes for an exporter
  that has a registered upstream-stage flow and already exports the processed
  stage.
- **Track B1 — eligible-market processed-export stage entry:** rank one unique
  `(exporter, processed stage)` event per row; a positive requires a late lane to
  a destination with early processed-stage demand.
- **Track B2 — destination formation conditional on entry:** for each Track-B1
  event that materializes, rank its candidate destinations.

Tracks B1/B2 measure processed-export emergence, not domestic processing
capacity. Track A is destination extension, not first-time upgrading.
Here, a **registered upstream stage** is a raw or intermediate stage explicitly
listed in the chain registry's `upstream_map` for the target processed stage. It
must not be shortened to “raw input” when the registered predecessor is
intermediate.

## Install

Python 3.10 or newer is recommended. Only NumPy and pandas are required:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`.

## Put the data where the loader can find it

For the public reviewer release candidate, download and extract both UPGRADE-BENCH data assets
from the GitHub prerelease when running the full forward protocol:

- `upgrade-bench-v2-historical-fold.zip` for `fold2`; and
- `upgrade-bench-v2-main-window.zip` for `main`.

Archive members retain repository-relative paths. Any of these layouts works:

```text
<root>/data/processed_v2/candidates_sheep.csv
<root>/processed_v2/candidates_sheep.csv
<root>/candidates_sheep.csv
```

Then either pass `--data-root <root>` or set:

```bash
export UPGRADE_BENCH_V2_DATA=/path/to/root-or-processed_v2
```

The loader also discovers `data/processed_v2` relative to a repository checkout
or the current directory. It supports all six chain IDs: `sheep`, `cotton`,
`aluminium`, `nickel`, `cocoa`, and `oilseed-soy`.

Running `python loader.py` loads and validates every Track A/B1/B2 table in both
snapshots. Loading is intentionally strict: the exact schema, metadata, unique
keys, label/value consistency, size construction, task unit, and committed
SHA-256 exporter-stage diagnostic split must all agree with `2.1-dev`.

The data payload is defined by the strict documented registry. Frozen regexes
were applied automatically to all 5,022 BACI HS92 source rows, producing 576
observable chain--HS6 records. With 34 legacy-only provenance records, the
complete ledger has 610 records covering 588 unique HS6 codes. Observable
decisions are 283 included, 194 excluded, and 99 out of stage; full-ledger
decisions are 283 included, 228 excluded, and 99 out of stage. The included
codes map to 53 active stages across the six chains, and the audit records 19
retained-code stage reassignments. This is automated full-table regex
application and generated rule application; it does not constitute human
review. The finite lexical negative controls test declared variants but do not prove
lexicon completeness. Every active stage has a canonical definition and every
included code has an official-description `stage_fit` decision in
`chains/evidence/registry_evidence.json`. The registry and supporting code were
developed with LLM assistance; the curation sidecar and canonical hash-bound
receipt separately establish release-valid sampled human-validation status. The
frozen design covers 212 of the 610 machine decisions and all 53 stage
definitions; the other 398 decisions are not individually human-verified. The
sampled validation was completed by two reviewers, and final adjudication accepted no construct
change. This enumerated
construct is not a claim to cover all products or industries.

Unresolved focal attribution is excluded. Commodity-explicit blend and residual codes are retained
as whole published HS6 baskets, but the benchmark cannot recover their focal-material share, purity,
physical composition, or dominant feedstock. Results therefore concern commodity-explicit HS6
baskets and registry-defined analytical families rather than pure-material products.

Repository validation covers 48 objects: Track-A and pre-view Track-B lane
tables plus derived B1 and B2 views for six chains and two snapshots. The
standalone loader exposes the 36 official evaluation views (A/B1/B2); the
pre-view Track-B lane tables remain construction/audit inputs.

## Score built-in ex-ante columns

`size` is complete in every table and is the safest smoke-test baseline:

```bash
python eval.py --track A  --chain sheep --snapshot fold2 --column size
python eval.py --track B1 --chain sheep --snapshot main  --column size
python eval.py --track B2 --chain sheep --snapshot main  --column size
```

Permitted built-in columns are printed by `loader.feature_columns(track)`:

| Track | Ex-ante columns accepted as scores |
|---|---|
| A | `size`, `log_exporter_capacity`, `log_importer_demand`, `grav`, `gnn` |
| B1 | `size`, `log_upstream_capacity`, `n_candidate_destinations` |
| B2 | `size`, `log_exporter_capacity`, `log_importer_demand`, `grav`, `gnn` |

The scorer rejects any selected column containing NaN/Inf. In `2.1-dev`, some
optional `grav`/`gnn` values are intentionally unavailable; that is not silently
imputed by the evaluator.

Outcome fields are never valid scores. In particular, `y`, `z`, `entry_y`,
`lateval`, `entry_lateval`, and `n_materialized_destinations` are blocked.

## Score an external submission

The CSV schema is exact: no index or outcome columns, no duplicate keys, and no
missing or extra candidates.

```text
Track A/B2: i_iso,j_iso,stage,score
Track B1:   i_iso,stage,score
```

First select the model and every supervised/hyperparameter choice on `fold2`:

```bash
python eval.py --track A --chain sheep --snapshot fold2 \
  --scores scores/sheep_a_fold2.csv
```

Freeze those choices before the main evaluation and retain the exact selection
configuration as a file. `selection_config.example.json` is a minimal suggested
shape, but the evaluator hashes raw bytes and does not require a particular
config format. Copy and truthfully complete `protocol_attestation.example.json`,
then run exactly once on the entire installed main candidate cohort. For B1/B2,
that cohort is explicitly limited to eligible markets:

```bash
python eval.py --track A --chain sheep --snapshot main \
  --scores scores/sheep_a_main.csv \
  --attestation protocol_attestation.json \
  --selection-config frozen/sheep_a_selection.json \
  --output results/sheep_a_main.json
```

The scorer will not label a main external result official without a valid
self-attestation stating that selection used `fold2`, choices were frozen, and
main labels were not used for selection, feature fitting, imputation, or
calibration. Attestation schema version 2 also requires a stable `run_id`, a non-empty ordered list
of exact integer seeds, and raw-byte SHA-256 values for the evaluated benchmark
CSV, submitted score CSV, and frozen selection config. The evaluator recomputes
all three hashes and rejects a mismatch.

The example contains placeholder zero hashes. Compute the three actual values
from this directory before editing the attestation:

```bash
python -c "from eval import sha256_file; import sys; [print(sha256_file(p), p) for p in sys.argv[1:]]" \
  /path/to/candidates_sheep.csv scores/sheep_a_main.csv frozen/sheep_a_selection.json
```

This is a **schema-checked self-attestation**, not an independent leakage audit.
It checks the document shape and binds exact artifact bytes; it cannot verify
the truth of statements about how a model was developed. A valid document makes
the result protocol-eligible on the submitter's attestation; it does not
independently establish protocol compliance.

The package distributes main labels for reproducibility, so a valid
self-attestation does not create a blind leaderboard. Independently enforced
comparisons require a hosted evaluator or separately held-out future cohort.

Invalid, incomplete, stale-hash, or wrong-artifact attestations are rejected and
produce no official result. They are never silently downgraded. To proceed with
exploratory work, omit the invalid attestation and use the explicit diagnostic
override instead:

```bash
python eval.py --track A --chain sheep --snapshot main \
  --scores exploratory.csv \
  --diagnostic-override "same-window ablation; not an official result"
```

Override output is marked `official: false`; it must not be reported as a main
benchmark result.

## Metrics and deterministic ranking

- Track A: average precision; precision, recall, and realized-value capture at
  absolute shortlist budgets; plus row-weighted AUC within size-quantile bins as
  a structural diagnostic. It also reports per-exporter shortlists at 5 and 10
  lanes, matching the CPU artifact: macro precision averages over all exporters;
  macro recall and macro value capture average only over exporters with at least
  one positive; micro recall/value pool their denominators across exporters. If
  there is no positive exporter, recall/value fields are `null` rather than zero.
- Track B1: eligible-market entry-level average precision; precision, recall,
  and entry-value capture at exporter-stage budgets.
- Track B2: per-entry macro recall and realized-value capture at `1`, `3`, and
  `5` destinations.

Average precision treats equal scores as a threshold block. Budget cutoffs are
deterministic: score descending, then lexical `(i_iso, stage, j_iso)` for Track A,
`(i_iso, stage)` for Track B1, and `j_iso` within each entry for Track B2. Input
row order never decides a tie. Track-A exporter shortlists use `(stage, j_iso)`
inside each exporter after score descending.

The official temporal result uses the entire benchmark-defined, track-specific cohort.
The candidate-table `train/test` field is a same-window, exporter-stage-grouped
**transductive diagnostic**, not a replacement for historical selection
followed by frozen main evaluation.

For B1, the cohort contains every benchmark-defined **eligible-market** `(exporter,
processed stage)` candidate, not every possible future export market. The
early-demand screen excludes destinations with no early demand for that
processed stage. The replacement raw coverage audit found that the main screen
covers 270 of 280 realized entries, 556 of 578 late-start lanes, and 99.68% of
their observed late value. Results must retain this boundary in their task name
and interpretation.

The benchmark design includes one fixed external-pretrained ULTRA-ZS reference scored under these
same task cohorts, tie rules, and budgets. The checkpoint receives no UpgradeBench-label training,
selection, fine-tuning, calibration, or score transformation, but it uses each target early graph.
The current `../../results_v2/metrics/v2_ultra_zero_shot_summary.{json,csv}` pair is verified and
bound by the canonical resolution receipt.
External compute/data are unmatched and there is one checkpoint, so any eventual result is a
descriptive no-label-adaptation reference, not graph-free cold start, a fair-compute ranking,
champion, significance test, population estimate, or causal result.

The repository also reports a formal reviewer-motivated GBDT reference using the same early-window
features as the logistic systems. Its four-point grid is selected on grouped historical folds, all
18 chain--task models are refit before any main table is opened, and B2 scores every lane before
realized-entry conditioning. The replacement pair is
`../../results_v2/metrics/v2_gbdt_baselines.{json,csv}` and verifies from the repository root with
`python tools/v2_gbdt_baselines.py --verify-output`. The replacement artifact is verified
against the 283-code cohorts and bound to the current paper interface and resolution receipt. This
reference was not in the original prespecified set and is not a target-selected
champion.

## Verify this evaluator

From this directory:

```bash
python generate_manifest.py
python -m py_compile loader.py eval.py generate_manifest.py
```

From the repository root:

```bash
python -m unittest tests.test_upgrade_bench_v2_package -v
```

`MANIFEST.sha256` covers the standalone code and documentation but deliberately
excludes any locally extracted `data/` directory. Data archives have independent
payload manifests and checksums.

See `DATASHEET.md` for composition and limitations, and the repository-level
`BENCHMARK_V2_SPEC.md` for the normative benchmark construction protocol.

The `2.1-dev` string is the package's artifact/API compatibility version. Release eligibility
co-seals the complete 283-code computation/result branch and the canonical outcome-blind
sampled human-validation receipt as independent gates; their presentation does not assert an execution order.
A no-construct-change receipt leaves the bound computations applicable, while an accepted construct
change requires a new registry/benchmark version and full registry-dependent rebuild. Current
claim-bearing outputs enter the package only after those gates pass. This package is currently
distributed as a public reviewer release candidate through a GitHub prerelease, for which
repository packaging, release-asset verification, public download checks, and remote CI have
passed. It is not the final archival release. The archival DOI and final artifact version remain
pending, and the retained `2.1-dev` identifier must be interpreted only as the package's
artifact/API compatibility version.
