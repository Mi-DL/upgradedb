# Public release boundary

A maintainer source checkout may contain private provenance and exploratory
material outside the publishable tree. A release manifest proves integrity of
the selected tree; it is not, by itself, permission to redistribute every file
that may exist in maintainer staging. `RELEASE_MANIFEST.sha256` is specifically
the **public repository scope**. The broader staging inventory is available to
boundary tests as `internal_release_scope()` but is never written as the root
release manifest. The default public data artifacts are the bundles declared in
`release/DATA_ARTIFACT_INDEX.json`.

The public Git repository is limited to current code, documentation, tests,
portable job templates, manifests, exact scientific configurations, and
lightweight sanitized result summaries. Its manifest excludes noncurrent
packages and exploratory output trees. Large current main/history tables
under `data/processed_v2/` are immutable, indexed payloads for versioned GitHub
Release assets and an eventual DOI-bearing archival mirror. Raw inputs, caches,
formal run trees, and gated extracts belong to neither public surface.

No release tool uploads, deletes, untracks, rewrites history, or migrates data to
Git LFS.

## Maintainer staging boundary

Permission-gated inputs and exploratory material are outside the current
benchmark. Entire private prefixes are excluded from public Git and every public
bundle; no current benchmark contract depends on those files. Free research
access does not establish a blanket redistribution right. `DATA_LICENSE.md`
records the public source terms and attribution requirements. Every current
public ZIP receives a freshly generated payload manifest and external checksum.

The supported publication path is a clean, one-commit export containing exactly
the public manifest inventory. A direct push from an unrestricted maintainer
staging checkout is not a supported public-release path because it may carry
files or reachable history outside that inventory. `tools/release_clean_clone.py`
builds and tests the candidate public tree without pushing or rewriting its
source; selecting the final public owner/name and publishing the verified tree
remain explicit maintainer actions.

## Scheduler scripts

Site-specific scheduler and host-launch files can contain queue names,
home-directory paths, environment locations, and user identifiers. Maintainers
may retain them as private provenance, but they are outside the public repository
scope. The public scope includes only the portable, parameterized launchers and
direct-host workers:

- `jobs/v2_gpu_select.pbs`
- `jobs/v2_gpu_evaluate.pbs`
- `jobs/v2_gpu_nohup_worker.sh`
- `jobs/v2_gpu_main_worker.sh`

The workers, dependency overlay lock, environment checker, sync-manifest helper,
and host-neutral operations runbook are public reproducibility code. They use
parameters/placeholders rather than institutional identities. The concrete host
inventory and local GPU plan remain private staging files. Public launchers
require the caller to provide the run root/interpreter and use environment
variables for raw data and external NBFNet paths.

Private LOCO claims/receipts, host launchers, and adversarial fixtures remain private. Host-neutral
formal controllers, exact scientific configurations, and public summary verifiers are released only
where selected by the shared policy. The LOCO and ULTRA designs are published by exact name as
`configs/v2_loco_formal.json` and `configs/v2_ultra_formal.json`; the policy does not open the
surrounding `configs/` directory recursively. The formal tabular reference is likewise published
only through `configs/v2_gbdt_baselines.json`. ULTRA checkpoints, vendored upstream code, JIT caches,
scores, receipts, logs, and formal run state remain private.

## Public provenance

Raw console logs and host-run directories can contain usernames, hostnames,
absolute paths, PIDs, storage layout, and unrelated process inventories. The
following are retained internally but excluded from public release scope and
public bundles by the shared policy in `tools/public_release_policy.py`:

- `results/logs/`
- all of `results_v2/gpu_rolling/` (selections, freeze records, score/cache
  material, run registries, locks, logs, status, and invalidated pilots)
- all of `results_v2/gpu_smoke/`
- all of `results_v2/loco_formal/` (component scores, manifests, claims,
  receipts, status, and formal run provenance)
- all of `results_v2/ultra_formal/` (scores, seals, components, receipts, status,
  repeat checks, and formal run provenance)
- internal progress checklists and host-specific operations/configuration files

`v2-results` is not a recursive directory bundle. It uses an exact-name
allowlist for the CPU audit/baseline/robustness outputs, paper-number interface,
claim documentation, and reviewed sanitized outputs. The allowlist includes:

- `results_v2/metrics/v2_gpu_rolling_summary.json`
- `results_v2/metrics/v2_gpu_rolling_summary.csv`
- `results_v2/metrics/v2_value_diagnostics.json`
- `results_v2/metrics/v2_value_diagnostics.csv`
- `results_v2/metrics/v2_loco_transfer_summary.json`
- `results_v2/metrics/v2_loco_transfer_summary.csv`
- `results_v2/metrics/v2_ultra_zero_shot_summary.json`
- `results_v2/metrics/v2_ultra_zero_shot_summary.csv`
- `results_v2/metrics/v2_gbdt_baselines.json`
- `results_v2/metrics/v2_gbdt_baselines.csv`
- `results_v2/metrics/v2_product_space_density.json`
- `results_v2/metrics/v2_product_space_density.csv`
- `results_v2/scores/v2_product_space_density_scores.csv`
- `results_v2/metrics/v2_score_robustness_r5.json`
- `results_v2/metrics/v2_score_robustness_r5.csv`
- `results_v2/metrics/v2_eligibility_threshold_geometry.json`
- `results_v2/metrics/v2_eligibility_threshold_geometry.csv`
- `results_v2/metrics/v2_benchmark_profile.json`
- `results_v2/metrics/v2_contemporary_references.json`
- `results_v2/metrics/v2_contemporary_references.csv`

`paper/generated/v2_benchmark_profile.tex` is the generated manuscript companion to the last JSON.
Together they are a sanitized, post-resolution aggregate profile of per-chain graph scale, B1/B2
effective sample units, and compute. They are independent of the paper-number interface and do not
extend its governed source map. The profile contains only
aggregate values, public-source hashes, and private-evidence receipt hashes; private claim files,
host inventory, raw receipts, logs, paths, and formal run contents remain outside the public scope.
Public clones verify the pair with:

```bash
python tools/generate_v2_benchmark_profile.py --verify --profile repository
```

A replacement build and maintainer-only full provenance pass require the private evidence inputs to
be supplied explicitly. Those inputs are read only to validate aggregates and hashes and are never
copied into the JSON or TeX output.

The contemporary-reference JSON/CSV pair is atomic, post-resolution, and independent of the
schema-8 paper-number receipt. Its deterministic TeX companion is a third manuscript interface;
all three numeric files remain governed by any active invalidation hold. The public verifier binds
the pair and TeX macros to `configs/v2_contemporary_references.json` and the current summarizer,
without publishing formal score trees, host paths, or run receipts.

The value diagnostics label target-outcome-ranked comparisons as descriptive
oracles, not deployable or causal results. The public LOCO role contains only sanitized,
tier-abstracted, export-edge-deduplicated NBFNet aggregates. Only `in_domain - loco` under
that exact graph contract is a valid matched comparison; stage-relation graph
results are not a cross-subtraction baseline. Its public verifier recomputes the
paired summaries without access to private provenance. No current pair is present during the hold.

The ULTRA-ZS pair is the only public surface for the formal external-pretrained reference. Its
public-only verifier must enforce the six-of-six chain score seal, all 18 chain--task records, the
sheep score/metric repeat gates, six-chain aggregates, comparison directions, and exact public
source/config/controller hashes. It must not open or reveal checkpoint bytes, score files, the raw
BACI archive, private receipts, host paths, or `results_v2/ultra_formal/`. ULTRA-ZS uses each target
early graph but no UpgradeBench label for training, checkpoint selection, fine-tuning, calibration,
or score transformation. The sanitized result is descriptive: external pretraining resources are
unmatched, one checkpoint has no training-seed interval, and no fair-compute, champion,
statistical-significance, population, graph-free-cold-start, or causal claim is permitted.

The GBDT JSON/CSV pair is an atomic public result: neither file may be bundled alone. Its public
configuration fixes the four estimator choices, task features, minimum leaf sizes, historical
group/objective rules, all-18-models-before-main read gate, budgets, 200-draw cluster bootstrap, and
seed. The verifier checks canonical bytes, historical selection traces, target counts and metrics,
aggregates, runtime, privacy, the config/runner/shared-code hashes, and all 24 historical/main
candidate hashes. Repository smoke verifies its canonical JSON/CSV, schema, aggregates, and privacy
without opening absent release-asset tables; full smoke additionally re-hashes all candidate files.
The artifact is explicitly reviewer-motivated and outside the original prespecified reference set.

The product-space result is atomic across its JSON, CSV, and keyed-score CSV. Its public verifier
recomputes the released B1 metrics from the keyed scores without opening raw BACI; the private full
verifier additionally rebuilds early RCA, proximity, and density. The paired-score JSON/CSV is
score-derived from the already frozen graph surface and supports finite-benchmark uncertainty,
pooling/budget, and cascade diagnostics only. The eligibility-geometry JSON/CSV is score-free and
must exactly reproduce the released 100-kUSD cohort before reporting the 50/250-kUSD alternatives.
None of these artifacts authorizes a population, significance, compute-efficiency, causal, or
inflation-adjustment claim.

Any new file under `results_v2/` is private until it is deliberately reviewed
and added to the shared allowlist. Raw GPU files can therefore never become
public merely by being copied into the staging checkout.

`results_v2/metrics/INVALIDATED.json` is a fail-closed release receipt. Whenever
any claim-bearing source or generated artifact changes, the final bundle remains
blocked until the receipt is regenerated with an exact SHA-256 proof inventory.
The current development snapshot has verified current-registry LOCO, ULTRA, and GBDT result pairs,
a receipt-bound paper-number interface, and a public `RESOLVED` transition. The sampled human-review
receipt is a sampled-validation receipt: it covers a frozen stratified sample of
212 of 610 machine decisions plus a census of all 53 stage definitions, not the
398 unsampled decisions. Two reviewers completed the sampled validation and final adjudication
accepted no construct change. The receipt and invalidation resolution are co-bound without implying
an execution order. Current clean-clone and remote-CI verification, publication, DOI, authorship, venue/cycle metadata, and other administrative metadata
remain pending.

A public receipt is accepted only as canonical strict JSON with the exact frozen
field inventory, `RESOLVED` status transition, original status/reason/date,
whole-second UTC resolution timestamp, normalized scope, replacement map,
resolution-gate hash, current verifier map, and paper-source map. `SUPERSEDED`,
minimal markers, duplicate keys, non-finite values, extra fields, and noncanonical
bytes are rejected. The receipt's source map must equal the source map inside the complete schema-8
`results_v2/paper_numbers.json`; GPU, LOCO, ULTRA, and GBDT statuses and their paper macros must be
`COMPLETE`. The final number count, governed-source inventory, and key/value digests must come from
the fully generated, reviewed current interface rather than a guess.
JSON and TeX source/value maps must agree exactly
before their canonical key/value map is frozen. Every replacement, verifier, and
repository-resident source is hash checked.

These repository-resident hashes detect accidental drift and incomplete freeze
transactions. They are not an external signature and do not claim resistance to
a maintainer changing verifier code and constants together. An independently
protected release tag, cryptographic signature, or DOI-backed archival record
and digest is required for that threat model.

For claim-bearing changes, the only supported transition is the gate below. `--freshen` atomically
reopens the old `RESOLVED` receipt as an active
hold; the following dry run is read-only, and the resolution command rechecks
every authoritative verifier, replacement byte, source hash, and release-policy
hold. Both mutations require literal confirmation tokens. Run them only after
the complete sanitized LOCO and ULTRA JSON/CSV pairs and the formal GBDT pair pass their verifiers.
The post-freshen `--preview-dir` mode runs the scientific and schema checks but
permits an unset (never a mismatched) final value digest. It exports exact
non-canonical JSON/TeX review copies and reports `observed_candidate_sha256`.
Review the complete active schema map and generated key count, set its exact count and digests in the
policy, review the code change, and then run the strict dry run before resolution.

```bash
python tools/summarize_v2_loco_results.py --verify-output
python tools/summarize_v2_ultra_results.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/resolve_v2_invalidation.py \
  --freshen --confirm FRESHEN-V2-RESOLUTION
python tools/resolve_v2_invalidation.py --preview-dir tmp/schema8-review
# Review tmp/schema8-review/results_v2/paper_numbers.json and the matching TeX.
python tools/resolve_v2_invalidation.py --dry-run
python tools/resolve_v2_invalidation.py \
  --confirm RESOLVE-V2-REGISTRY-AUDIT
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile full
# Private provenance checkout only:
python tools/resolve_v2_invalidation.py --verify-resolved
```

The gate renders the current paper JSON/TeX schema in a temporary directory
before a marker-last transaction. Validation or replacement failure restores
the prior canonical paper bytes and fail-closed notice; manually deleting or
editing the notice is not a supported resolution path.

`--verify-public-receipt` is the public, read-only verifier. It never opens the
private raw BACI archive or private GPU/LOCO/ULTRA formal run trees. Its `repository`
profile verifies the complete receipt structure and every bound public-Git
code/config/result/source byte, while treating `data/processed_v2/` source bytes
as external release assets and not opening them. Repository smoke separately verifies the GBDT
canonical pair and internal aggregates without those tables. The `full` profile requires and hashes
every external source, including the 24 candidate files bound by GBDT. `--verify-resolved` is a maintainer-only
private authoritative check: it first runs the full public verifier, then reruns
the raw/formal scientific provenance verifiers available only in the private
staging checkout.

Public result artifacts should retain scientific provenance—software versions,
accelerator model/class, seeds, protocol phase, source/input hashes, metrics,
and upstream dependency commits—while replacing:

- usernames with `<user>`;
- hostnames with an environment class such as `external-gpu-host`;
- absolute workspace/raw/cache paths with repository-relative roles;
- PIDs and unrelated process/storage inventory with counts or omission.

The release audit scans the contents of both the public repository scope and all
planned/frozen bundle payloads. It rejects Unix user-home paths, ordinary or
JSON-escaped Windows user paths, institutional host FQDNs and bare host aliases,
and known account identifiers. The selector, index validator, manifest, and
audit share one policy module. A formal GPU/LOCO/ULTRA summary must be sanitized and
frozen only after its complete run passes canonical verification; incomplete
raw run records are not silently promoted to release evidence.

`.gitignore` guards raw GPU run state, `results_v2/ultra_formal/`, host inventory, LaTeX intermediates,
and workstation output. It deliberately does not ignore `data/processed_v2`:
those tables remain intended public release-asset payloads even though they are
not ordinary Git objects. NumPy `.npy`/`.npz` files are always hashed as binary;
text line-ending normalization never applies to them.

Private BACI filtered caches are a separate raw-derived input surface. Their
manifest and annual `i,j,k,year,v` files are never public artifacts. `.gitignore`
blocks conventional cache names and private directories, while the shared
policy independently rejects path components named `private`, `.private`,
`tmp`, `temp`, `raw`, `cache`, `.cache`, or `caches`, raw BACI archive/table
names, logs, GPU score/selection caches, and checkpoints. It also rejects
non-canonical, absolute, parent-traversing, or symbolic-link sources. Thus an
accidentally unignored or tracked raw-derived file cannot enter a manifest or
ZIP by being moved under an otherwise public prefix.

## Fresh-history clone gate

After final manifests and archives are frozen, run:

```bash
python tools/release_clean_clone.py --profile repository
python tools/release_clean_clone.py --profile full --artifacts-dir dist/final-release
```

The tool verifies the maintainer source tree, copies exactly the public root
manifest inventory, creates one commit using a neutral `example.invalid`
identity, clones it without source Git history, removes the local origin, and
runs the audit mode permitted by the validated release state. An active hold permits only the
repository profile and runs planned-boundary, registry, split, manifest, privacy, and history-size
checks; a resolved candidate runs final audit and release smoke. Full mode first verifies and
installs every frozen archive. Temporary trees are removed by
default. This is a preflight, not authorization to publish: final authors/order,
venue/cycle metadata, repository owner/name, artifact hosting, archival DOI, and release timing
remain maintainer decisions.

Current repository/full smoke, clean-clone verification, and remote CI remain release gates.

## CI boundary

Repository CI detects the active hold. During the hold it runs the code-only boundary, manifest,
registry, package, split, and size/history checks without claiming absent data or results. After
resolution it additionally runs the full unit suite, repository smoke, data-index checks, the v2
validator, formal result verifiers, and the generated paper-number interface. Excluded scheduler,
vendored-model, checkpoint, controller, and private formal-tree tests are not part of public CI.
The final result profile also
requires the sanitized value, LOCO, ULTRA, and GBDT result pairs. Manual dispatch can require
the full payload by setting the boolean `require_full_payload` input. This input is a fail-closed
assertion, not a payload acquisition mechanism: the workflow does not download or install external
payloads. If the input is true while the invalidation hold is active, or if
`data/processed_v2/dataset_summary.json` is absent, the job fails instead of falling back to the
repository profile. When the payload marker is present, the existing full validator and full-profile
checks remain responsible for rejecting an incomplete or inconsistent payload. Local Windows
evidence does not substitute for the pending remote-CI run.

LaTeX compilation is not a mandatory GitHub-hosted CI step because the minimal
runner intentionally does not install a TeX distribution. Paper-number drift is
still checked in CI; the release workstation must compile and visually inspect
the PDFs as a separate checklist item.

Exploratory records remain private and are not selected into the current claim
or release surfaces.
