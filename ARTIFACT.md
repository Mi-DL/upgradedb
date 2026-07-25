# UPGRADE-BENCH artifact verification

This is the release-facing verification entry point. It verifies derived benchmark artifacts; it
does not rebuild the benchmark from restricted-size raw archives and does not require a GPU.

> **Development state.** This checkout is an active-hold, code-only snapshot for the current
> 283-code registry. No claim-bearing LOCO or ULTRA summary, benchmark profile, paper-number map,
> six-role replacement receipt, final bundle index, or processed-data payload is promoted here.
> Invalidation resolution and the outcome-blind sampled human-validation receipt remain separate pending gates and are
> joined for release without asserting an execution order. Current scientific components may be
> verified in a maintainer/full-payload checkout, but no isolated component makes the paper or release
> interface complete. Publication, authorship, venue/cycle metadata, hosting, DOI, and other
> administrative metadata remain separate gates.

## Minimal setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r benchmark/upgrade-bench-v2/requirements.txt scikit-learn
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Verify current scientific components

The commands below define the component-verification contract. Claim-bearing outputs are
intentionally absent from this development snapshot; run these commands only in a maintainer or
full-payload checkout in which the governed inputs have been installed. From that repository root:

```bash
python tools/verify_registry_curation_protocol.py
python tools/validate_v2.py
python tools/v2_rolling_cpu_baselines.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/v2_product_space_density.py --verify-output
python tools/v2_eligibility_threshold_geometry.py --verify-output
python tools/audit_v2.py --verify-output
python tools/v2_b1_coverage.py --verify-output
python tools/summarize_v2_gpu_results.py --verify-output
python tools/v2_value_diagnostics.py --verify-output
```

These checks bind the strict registry and both temporal snapshots, candidate keys, labels and value
support, forward CPU choices, the independent raw-label and B1 coverage audits, the existing graph
forward summary, and fixed-budget value diagnostics to their recorded inputs.

After the current LOCO and ULTRA replacements have been verified and promoted in the six-role
transaction, verify the regenerated sanitized benchmark profile and its TeX companion with:

```bash
python tools/generate_v2_benchmark_profile.py --verify --profile repository
```

The profile pair is absent from this active-hold checkout, so this command is expected to fail until
the governed current pair is installed. That pair reports only aggregate graph scale, B1/B2
effective sample units, compute, and
evidence hashes. It is independent of the paper-number interface and contains no private claims or
host inventory. Rebuilding it and performing the maintainer-only full-provenance check require
explicitly supplied private formal evidence; those inputs are not copied into either public output.

The GBDT command verifies `results_v2/metrics/v2_gbdt_baselines.{json,csv}` against
`configs/v2_gbdt_baselines.json`, the runner and shared feature/metric code, and all 24 historical
and main candidate files. Four tree/iteration configurations are compared only with grouped
historical objectives; all 18 selected models are refit before any main table is opened, and B2
scores all lanes before evaluation conditioning. The artifact records 200-draw exporter/entry
cluster intervals. It is an additional reviewer-motivated tabular reference rather than an original
prespecified baseline or target-selected champion. Numerical values must come from a receipt-bound
current interface and are not reproduced in this development snapshot.

## Product-space, paired-score, and threshold-geometry evidence

The product-space release is one atomic three-file object:

- `results_v2/metrics/v2_product_space_density.json`;
- `results_v2/metrics/v2_product_space_density.csv`; and
- `results_v2/scores/v2_product_space_density_scores.csv`.

The keyed-score CSV is an explicit exception to the general rule that score trees stay private. It
contains only released B1 identities/outcomes, the label-free density, and hash bindings needed to
recompute the public metrics. `python tools/v2_product_space_density.py --verify-output` checks all
keys, bindings, metrics, 200-draw exporter intervals, and deterministic JSON/CSV bytes without
opening raw BACI. In a private provenance checkout, add `--verify-raw` to hash the attested archive
and rebuild full-economy RCA membership, product proximity, and every keyed score. Any promoted
values are descriptive over the six fixed chains and do not define a causal capability measure;
current numerical values are not reproduced in this development snapshot.

`results_v2/metrics/v2_score_robustness_r5.{json,csv}` contains no formal model scores. It reports
same-candidate, same-seed, paired PyKEEN-minus-NBFNet cluster intervals; all declared B1 pooling and
budget arms; and the linked two-stage B1-to-B2 diagnostic. Its full verifier deliberately requires
the private frozen graph score tree:

```bash
python tools/v2_score_robustness_r5.py --verify-output
```

The public pair and frozen config/runner may be inspected and receipt-hashed without publishing
that tree. The aggregate intervals hold the six chains fixed and therefore support finite-benchmark
description only, not population inference over value chains.

`results_v2/metrics/v2_eligibility_threshold_geometry.{json,csv}` is a score-free cohort audit.
The public/full-payload verifier checks canonical bytes, registry and processed-table hashes, and
the exact 100-kUSD reconstruction gate without opening raw BACI. Regenerating the 50/100/250-kUSD
cohorts requires the private attested archive. The artifact reports candidate/positive counts and
key overlap only; it does not reuse fixed-cohort scores or claim performance robustness.

The robustness verifier additionally binds the private raw BACI archive used by the annual
persistence and threshold audits. Run it only in a private provenance checkout where that archive is
available:

```bash
python tools/v2_robustness.py --verify-output
```

The raw archive and filtered cache must not be copied into a public bundle merely to make this
additional check available.

## LOCO and ULTRA-ZS release gates

Matched LOCO is a tier-abstracted NBFNet parameter-transport diagnostic. Chain-specific export stages
are mapped to shared processing tiers and export edges are deduplicated after that mapping.
In-domain parameters are trained on the target chain and leave-one-chain-out parameters on the other
chains; both modes use the held-out chain's early graph at inference. Training-edge volume is not
equalized, so `in_domain - loco` jointly reflects source set and volume. It is descriptive
parameter-transport evidence, not graph-free cold start, an isolated domain effect, population
inference, or a causal estimate.

The formal LOCO controller, component scores, worker receipts, logs, machine paths, and private
formal tree are not public payloads. Public consumers verify only a receipt-authorized sanitized
JSON/CSV pair with:

```bash
python tools/summarize_v2_loco_results.py --verify-output
```

ULTRA-ZS fixes one external checkpoint before target scoring and permits no UpgradeBench-label
training, checkpoint search, fine-tuning, calibration, or score transformation. Each target chain's
early graph remains inference context, so this is a no-label-adaptation reference rather than
graph-free cold start. External data and compute are unmatched and a single checkpoint has no
training-seed interval; it supports no fair-compute, champion, significance, population, or causal
claim.

The checkpoint, scores, seals, receipts, vendored source, logs, and host state remain private. Public
consumers verify only a receipt-authorized sanitized JSON/CSV pair with:

```bash
python tools/summarize_v2_ultra_results.py --verify-output
```

Neither current pair is promoted in this active-hold snapshot. No partial formal output,
intermediate aggregate, or private verification receipt is a paper number.

## Verify a frozen public candidate

The following is the target gate sequence for a complete frozen public candidate. In the current
active-hold state, the sequence is expected to stop at an absent current interface or unresolved
review/receipt gate; that refusal is not a reason to weaken or bypass a check.

```bash
python tools/release_manifest.py --verify --scope all
python tools/artifact_bundles.py verify-index
python tools/verify_registry_curation_protocol.py
python tools/validate_v2.py
python tools/v2_rolling_cpu_baselines.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/v2_product_space_density.py --verify-output
python tools/v2_eligibility_threshold_geometry.py --verify-output
python tools/audit_v2.py --verify-output
python tools/v2_b1_coverage.py --verify-output
python tools/summarize_v2_gpu_results.py --verify-output
python tools/v2_value_diagnostics.py --verify-output
python tools/summarize_v2_loco_results.py --verify-output
python tools/summarize_v2_ultra_results.py --verify-output
python tools/generate_v2_benchmark_profile.py --verify --profile full
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile full
python tools/generate_v2_paper_numbers.py --verify
python tools/release_smoke.py --profile full
```

Convenience wrappers invoke the artifact smoke workflow from any working directory:

```bash
bash run_artifact_smoke.sh
```

```powershell
.\run_artifact_smoke.ps1
```

Once the current six-role replacement transaction and sampled human-validation receipt exist, the full profile checks,
among other invariants:

- standalone-package and repository manifests plus the external bundle index;
- the public/private boundary, including permission-gated extracts and host-specific run state;
- the hash-bound LLM-assistance disclosure, outcome-blind sampled human-validation contract and
  release-eligible receipt, including the 212-of-610 decision sample, all 53 stage definitions,
  exact design weights, and explicit 398-record unsampled boundary; chain-selection criteria,
  operational review codebook, automated controls, and correction policy, without treating semantic
  regression tests or a single curator as independent agreement;
- both temporal snapshots across all six chains and every released A/B lane and entry view;
- calendar aggregation metadata, key uniqueness, labels, values, group splits, and event identities;
- forward CPU protocol flags and all recorded candidate-table hashes;
- the formal GBDT pair, frozen grid/config, 18 historical selections/refits, all-models-before-main
  gate, 200-draw cluster intervals, runtime metadata, and all 24 candidate hashes;
- the atomic product-space JSON/CSV/keyed-score triple, no-label read gate, metric recomputation,
  200-draw exporter intervals, and all candidate/config/registry hashes;
- the score-free paired graph robustness pair and its fixed-chain inference boundary, pooling,
  budget, and two-stage inventories, while leaving the private formal score tree outside the public
  payload;
- the exact 50/100/250-kUSD cohort geometry pair and canonical 100-kUSD key/label gate, without
  treating it as a model-performance result;
- the independent raw-label, B1 coverage, and robustness artifacts;
- the existing sanitized graph forward summary without publishing its selections or score tree;
- fixed-budget value diagnostics and their non-deployable oracle labeling;
- the current sanitized tier-abstracted matched-LOCO pair, without private formal provenance;
- the current sanitized external-pretrained ULTRA-ZS pair, including the 6/6 score-seal, 18/18 record, and
  sheep-repeat gates, without its checkpoint, scores, or private formal provenance;
- the separate sanitized benchmark-profile JSON/TeX pair, including aggregate graph scale,
  effective sample units, compute accounting, and private-evidence receipt hashes;
- the strict public resolution receipt, exact verifier/replacement maps, paper-number JSON/TeX
  source hashes, governed source inventory, cross-format value agreement, and exact canonical
  key/value-map digest;
- the standalone loader/evaluator API and a finite scorer result; and
- imports, privacy patterns, path safety, and payload-manifest integrity.

It never writes data, scores, splits, or result files.

Repository-resident hashes are deterministic drift/freeze checks, not external
signatures: simultaneous edits to verifier code, constants, and receipts require
an independently protected release tag, cryptographic signature, or DOI-backed
archival record and digest to detect.

A complete release-candidate code-only clone, before external processed-data bundles are installed,
uses the repository profile below. The present fail-closed checkout is expected to refuse this
sequence until the current replacement interfaces and sampled human-validation receipt exist.

```bash
python tools/artifact_bundles.py verify-index --allow-missing
python tools/generate_v2_benchmark_profile.py --verify --profile repository
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile repository
python tools/release_smoke.py --profile repository
```

That profile validates the repository surface, bundle catalog, current package, loader, and
evaluator. It verifies every receipt-bound repository code/config/result/source byte but does not
open or require externally distributed `data/processed_v2/` source bytes. It deliberately does not
claim that absent data payloads passed full verification. The sanitized public LOCO and ULTRA pairs
are recomputed by their public verifiers in both profiles. Repository smoke also checks the GBDT
canonical JSON/CSV, schema, aggregates, and privacy; its 24 candidate-file hashes are deferred to
the full profile because those tables are external release assets.

`--verify-public-receipt` is the consumer-facing check and never opens private raw BACI provenance
or the private formal GPU/LOCO/ULTRA run trees. In a private provenance checkout, maintainers may
additionally run `python tools/resolve_v2_invalidation.py --verify-resolved`; that authoritative
check first runs the full public receipt verifier and then reruns the private scientific verifiers.

## Release scopes

`RELEASE_MANIFEST.sha256` covers the intended public Git repository: code, documentation, tests,
portable jobs, environment declarations, bundle catalog, lightweight sanitized result summaries,
paper-facing generated macros, and the current benchmark package. It excludes raw third-party
archives, permission-gated institutional extracts, private formal score/run trees, logs, caches,
checkpoints, host operations, and internal progress files.

The only public score-path exception is the keyed product-space B1 CSV described above. Its schema
and contents are verifier-constrained to public candidate identities, a label-free derived score,
released outcomes, and hash bindings; no graph score tensor or formal run artifact is covered by
that exception.

Large processed tables under `data/processed_v2/` are covered byte-for-byte by
`release/DATA_ARTIFACT_INDEX.json` and deterministic bundle manifests. They are versioned GitHub
Release assets rather than ordinary Git objects. The exact split is documented in
`docs/DATA_DISTRIBUTION.md`; the selector and privacy audit share
`tools/public_release_policy.py`.

## Freeze or update a release

Do not regenerate hashes merely because an intermediate packaging check is red. Complete and verify
the current six-role replacement transaction, sampled human-validation receipt, invalidation resolution, manifests,
bundles, and clean-clone profiles before treating the candidate as release-ready.
For any subsequent claim-bearing change, complete the marker-last release transaction below:

```bash
# 1. Promote/verify sanitized formal summaries and the formal GBDT reference.
python tools/summarize_v2_loco_results.py
python tools/summarize_v2_loco_results.py --verify-output
python tools/summarize_v2_ultra_results.py --check-only
python tools/summarize_v2_ultra_results.py
python tools/summarize_v2_ultra_results.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/v2_product_space_density.py --verify-output
python tools/v2_eligibility_threshold_geometry.py --verify-output
# Maintainer-provenance checkout only: requires the frozen private graph score tree.
python tools/v2_score_robustness_r5.py --verify-output

# 2. Generate fresh review copies below tmp;
#    this never writes either canonical paper-number path and prints the exact
#    observed_candidate_sha256.
python tools/resolve_v2_invalidation.py \
  --freshen --confirm FRESHEN-V2-RESOLUTION
python tools/resolve_v2_invalidation.py --preview-dir tmp/schema8-review
# Review tmp/schema8-review/results_v2/paper_numbers.json and the matching TeX
# interface, set the reported digest in tools/public_release_policy.py, and
# review that code change. The strict dry-run remains blocked until it is set.
python tools/resolve_v2_invalidation.py --dry-run
python tools/resolve_v2_invalidation.py \
  --confirm RESOLVE-V2-REGISTRY-AUDIT
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile full
# Private staging provenance only; not expected to work in a public clone.
python tools/resolve_v2_invalidation.py --verify-resolved
python tools/generate_v2_paper_numbers.py --verify

# 3. Freeze and verify the standalone-package manifest first.
python tools/release_manifest.py --write --scope package
python tools/release_manifest.py --verify --scope package

# 4. Freeze the external data index after the package manifest is current.
python tools/artifact_bundles.py write-index
python tools/artifact_bundles.py verify-index

# 5. Freeze the public repository scope, which includes the frozen data index.
python tools/release_manifest.py --write --scope release
python tools/release_manifest.py --verify --scope all

# 6. Build and verify deterministic external assets in a new output directory.
python tools/artifact_bundles.py build all --output-dir dist/final-release
python tools/artifact_bundles.py verify-archives --output-dir dist/final-release

# 7. Exercise repository-only and full-payload fresh-history exports.
python tools/release_clean_clone.py --profile repository
python tools/release_clean_clone.py --profile full --artifacts-dir dist/final-release
```

Never push unrestricted maintainer staging history. The clean-clone tool copies only the public
manifest inventory into a neutral, one-commit history and runs the privacy, history-size, manifest,
and smoke gates. It does not upload or alter the source checkout. Current manifests, bundle indexes,
clean-clone checks, and remote CI must pass under the active release contract before publication.

Final author names/order, target cycle, repository owner/name, artifact host, archival DOI, raw-data
access dates, and release version remain unresolved metadata. Artifact tooling must not infer them.
