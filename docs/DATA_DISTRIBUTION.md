
# Data distribution and release runbook

This runbook separates the Git repository from the large, immutable benchmark
payloads. It is intentionally executable but does not authorize or perform an
upload, Git LFS migration, history rewrite, or deletion.

## Decision

Use a normal GitHub repository for current code, documentation, tests,
portable scientific configuration, lightweight sanitized results, and
`release/DATA_ARTIFACT_INDEX.json`. Publish the large processed main/history
archives as versioned GitHub Release assets and mirror the final release in a
DOI-bearing research archive such as Zenodo when the final archival release is
prepared. The current public reviewer release candidate uses a versioned GitHub
prerelease; its DOI-bearing archival mirror is not yet complete.

This is preferable to adding the processed benchmark CSVs to ordinary Git history:

- GitHub warns for regular Git objects above 50 MiB and blocks objects above
  100 MiB. GitHub recommends keeping repositories small, ideally below 1 GiB.
  See [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- A GitHub Release may contain up to 1,000 assets; each asset must be below
  2 GiB, with no stated total-size or bandwidth cap. See
  [About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
- Git LFS remains a fallback, but it changes the contributor/fork workflow and
  has plan-dependent storage and per-file limits. The current immutable
  benchmark snapshots fit the release-asset model better. See
  [About Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

A maintainer staging checkout may retain exploratory outputs, but neither the
fresh-history public Git manifest nor the public bundle catalog selects them.
The current main/history payload under
`data/processed_v2/` is not deleted or altered by this runbook. Version-like
path and bundle identifiers are stable implementation names, not separate
scientific stories.

## Versioned bundle layout

`python tools/artifact_bundles.py list` prints the counts and sizes for the current indexed bundle
set. After claim-bearing changes, regenerate the index before treating those values or archive
hashes as final. The deterministic layout contains:

| Bundle ID | Intended contents |
|---|---|
| `v2-standalone` | dependency-light UpgradeBench loader, evaluator, protocol examples, and nested manifest |
| `v2-main` | six-chain main-window lane, entry, destination, and summary tables |
| `v2-history` | six-chain historical selection-fold equivalents |
| `v2-results` | exact-name allowlisted metrics/audits and sanitized public summaries; never raw GPU trees |

Every bundle carries `DATA_LICENSE.md`, `ARTIFACT.md`, this runbook, the immutable pre-review
registry codebook, and its post-review completion addendum;
the internal `PROJECT_CHECKLIST.md` is deliberately excluded. V2 bundles also carry
`BENCHMARK_V2_SPEC.md`. The v2-results bundle carries the paper-facing generated
macro file as well. ZIP members use repository-relative paths, so the archives
can be extracted at the repository root. Each ZIP contains:

- `UPGRADE_BENCH_PAYLOAD_MANIFEST.sha256`, covering every payload member; and
- `UPGRADE_BENCH_DATA_ARTIFACT_INDEX.json`, a copy of the complete frozen index.

The build also emits `<archive>.zip.sha256` and an aggregate `SHA256SUMS` file.
ZIP timestamps, order, permissions, compression level, and member paths are
fixed, so identical inputs produce byte-identical archives.

Default public bundles exclude every permission-gated input and maintainer-only
exploratory artifact listed by `docs/PUBLIC_RELEASE_POLICY.md`. The release
tools do not delete local staging files.

The v2-results selector is an explicit allowlist. `results_v2/gpu_smoke/` and
all raw `results_v2/gpu_rolling/` selections, caches, scores, logs, run
registries, status, and pilot records are internal. All of
`results_v2/loco_formal/` and `results_v2/ultra_formal/` are likewise private formal provenance.
The current reviewed public results are limited to exact-name sanitized pairs, including:

- `results_v2/metrics/v2_gpu_rolling_summary.{json,csv}`;
- `results_v2/metrics/v2_value_diagnostics.{json,csv}`;
- `results_v2/metrics/v2_loco_transfer_summary.{json,csv}`; and
- `results_v2/metrics/v2_ultra_zero_shot_summary.{json,csv}`;
- `results_v2/metrics/v2_gbdt_baselines.{json,csv}`;
- `results_v2/metrics/v2_benchmark_profile.json`; and
- `paper/generated/v2_benchmark_profile.tex`.

The benchmark-profile JSON/TeX pair is a sanitized, post-resolution aggregate surface for
per-chain early-graph scale, B1/B2 effective sample units, and compute accounting. It is independent
of the paper-number interface and does not extend its governed source map. It contains aggregate
values, public-source hashes, and private-evidence receipt hashes only, not private claims,
inventories, receipts, paths, logs, or formal run contents. A code-only public clone can verify it with:

```bash
python tools/generate_v2_benchmark_profile.py --verify --profile repository
```

Rebuilding the pair and performing the maintainer-only full provenance pass require explicitly
supplied private evidence. Those inputs remain private and are represented publicly only by their
reviewed SHA-256 identities.

The current tier-abstracted matched NBFNet LOCO role is a sanitized aggregate pair. Both modes
use the target chain's early graph at inference, training-edge volumes are not matched, and only
`in_domain - loco` under the fixed tier/dedup contract is a valid matched comparison. It is
descriptive parameter-transport evidence, not graph-free cold start, an isolated domain effect,
population inference, or causal evidence.

The current external-pretrained ULTRA-ZS role is likewise a sanitized aggregate pair. ULTRA-ZS uses
each target early graph, so it is not graph-free cold start. External pretraining resources and
compute are unmatched and one checkpoint has no training-seed interval; the result supports no
fair-compute ranking, champion, significance, population, or causal claim. Checkpoints, scores,
seals, receipts, logs, raw BACI provenance, vendored source, and host state remain private.

Both sanitized pairs are present in the governed result surface and are bound by the canonical
resolution receipt; private checkpoints, scores, and run trees remain excluded.

The authorized formal GBDT result is an atomic JSON/CSV pair governed by
`configs/v2_gbdt_baselines.json`. The public configuration fixes the four estimator choices,
task-aligned features, grouped historical objectives, all-18-models-before-main read gate, budgets,
200-draw cluster bootstrap, and seed. Its verifier checks historical traces, main metrics,
aggregates, runtime, canonical bytes, privacy, public-source hashes, and all 24 candidate files. In
repository-only mode, release smoke verifies the canonical pair and its internal contracts without
pretending absent external candidate tables were hashed; full mode checks every candidate source.

`results_v2/metrics/INVALIDATED.json` is fail-closed. Any change to a
claim-bearing source or generated artifact blocks final bundle freezing until an
exact replacement SHA-256 inventory is written through the supported resolution
transaction. If a future active hold is opened, no current-registry result pair, paper-number
interface, replacement receipt, bundle index, or clean-clone record is promoted through that hold. The
outcome-blind sampled human-validation receipt and invalidation resolution remain required gates without implying an
execution order. The current receipt is `RESOLVED`; the fresh-history repository, deterministic
bundles, remote CI, GitHub prerelease, and anonymous asset verification are complete for the public
reviewer release candidate. DOI registration and the final archival mirror remain pending.

## Repository gates

The repository policy is enforced by:

```bash
python tools/repository_size_gate.py --history
```

The defaults are deliberately below GitHub's hard limits:

- 95 MiB maximum for any tracked worktree file or reachable Git blob;
- 350 MiB maximum total for the tracked worktree and for reachable unique blobs;
- review warnings at 45 MiB per file and 300 MiB total.

The aggregate limit prevents many individually legal v2 CSVs from silently
turning the code repository into a data archive. To audit all local non-ignored
files without treating them as the intended Git payload, use an explicit larger
diagnostic ceiling:

```bash
python tools/repository_size_gate.py --include-untracked \
  --max-total-mib 900 --warn-total-mib 500
```

`.github/workflows/release-artifact.yml` detects the active hold. While it is active, CI verifies
the code-only public boundary, manifests, registry audits, registry generator tests, package tests,
and split invariants without pretending external data or current result pairs are present. After
resolution, the repository and full-payload branches additionally verify the data index, complete
unit suite, result interfaces, standalone loader, and artifact smoke. Private scheduler, vendored
model source, checkpoints, formal controllers, and private run trees are never probed by the public
workflow. TeX compilation remains release-workstation visual QA because the minimal runner does not
install a TeX distribution.

## Freeze and pre-upload procedure

Run these commands from a staging checkout that contains all frozen benchmark
payloads. Do not change a payload after the index is written. These gates were completed for the
current reviewer release candidate and must be repeated after any governed change.

```bash
# 0. Promote/verify sanitized formal summaries and GBDT, then verify the schema-8 public
#    scientific interface. Private formal provenance is never bundled.
python tools/summarize_v2_loco_results.py --verify-output
python tools/summarize_v2_ultra_results.py --check-only
python tools/summarize_v2_ultra_results.py
python tools/summarize_v2_ultra_results.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile full
# Private provenance checkout only:
python tools/resolve_v2_invalidation.py --verify-resolved
python tools/generate_v2_paper_numbers.py --verify
python tools/generate_v2_benchmark_profile.py --verify --profile full

# Confirm the planned public repository and bundle inventories before any write.
python tools/public_release_audit.py --planned-only

# 1. Freeze and verify the standalone-package manifest first.
python tools/release_manifest.py --write --scope package
python tools/release_manifest.py --verify --scope package

# 2. Freeze the external bundle inventory and raw-byte hashes after the
#    package manifest is current.
python tools/artifact_bundles.py write-index
python tools/artifact_bundles.py verify-index

# 3. Freeze the PUBLIC repository scope, including the data index, then verify
#    both repository and package manifests.
python tools/release_manifest.py --write --scope release
python tools/release_manifest.py --verify --scope all

# 4. Run repository and full-payload gates.
python tools/repository_size_gate.py --history
python tools/public_release_audit.py
python tools/release_smoke.py --profile repository
python tools/release_smoke.py --profile full

# 5. Build deterministic assets in a new output directory and verify the exact inventory.
python tools/artifact_bundles.py build all --output-dir <new-empty-release-dir>
python tools/artifact_bundles.py verify-archives --output-dir <new-empty-release-dir>

# 6. Exercise a one-commit public clone, first code-only and then with assets.
python tools/release_clean_clone.py --profile repository
python tools/release_clean_clone.py --profile full --artifacts-dir <new-empty-release-dir>
```

`dist/` is ignored by Git. The final output directory must be new and empty;
`verify-archives` rejects a missing planned file, any extra file, checksum drift,
and changes observed during verification. Do not use `--force` for a final
freeze. The build refuses any archive at or above 2 GiB and never contacts a host.

Before upload, compare two independent builds where practical. At minimum,
retain `release/DATA_ARTIFACT_INDEX.json`, `<new-empty-release-dir>/SHA256SUMS`, every archive
sidecar, the release tag/commit, and the full-smoke log.

These commands do not authorize a direct push from an unrestricted maintainer
staging checkout, which may contain files or reachable history outside the
public inventory. The clean-clone tool proves that a manifest-only,
fresh-history export can pass, but it does not push it. The current repository and GitHub
prerelease were published separately after this gate. Final archival metadata, DOI registration,
and archival-mirror timing remain explicit maintainer decisions.

## Upload procedure (manual, not run by the tooling)

For the current reviewer release candidate, the versioned GitHub prerelease and all indexed assets
were published, anonymously redownloaded, and checksum-verified. The reusable procedure below still
applies to a final archival release; the DOI-bearing mirror has not yet been completed.

1. Create a signed or protected release tag from the commit that contains the
   matching artifact index and repository manifest.
2. Create a draft GitHub Release.
3. Upload the ZIP files, individual `.sha256` sidecars, `SHA256SUMS`, and
   `release/DATA_ARTIFACT_INDEX.json`.
4. Download the draft assets into a fresh clone, extract them at the repository
   root, and run the full verification commands below.
5. Only then publish the GitHub Release and deposit the same byte-identical
   assets in the archival host. Record the DOI and immutable URLs in release
   notes; do not edit a frozen asset in place.

An optional command-line upload, after explicit authorization and review, would
look like this; it is documentation, not an action performed by this project:

```bash
gh release create <tag> --draft --verify-tag \
  <new-empty-release-dir>/*.zip <new-empty-release-dir>/*.zip.sha256 \
  <new-empty-release-dir>/SHA256SUMS \
  release/DATA_ARTIFACT_INDEX.json
```

## Consumer verification

Download the required bundles and their sidecars. First verify each archive
against its sidecar or `SHA256SUMS`, then extract at the repository root. Finally
run:

```bash
python tools/artifact_bundles.py verify-index
python tools/release_manifest.py --verify --scope all
python tools/public_release_audit.py
python tools/generate_v2_benchmark_profile.py --verify --profile full
python tools/v2_gbdt_baselines.py --verify-output
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile full
python tools/release_smoke.py --profile full
```

The full profile requires every public/external input and verifies schemas, keys,
frozen candidate hashes, rolling CPU and raw-label provenance, the exact
receipt-bound value-diagnostic and schema-8 interfaces, the formal GBDT reference, the sanitized
tier-abstracted matched-LOCO summary, sanitized external-pretrained ULTRA-ZS summary, and evaluator
behavior. It intentionally does not require or claim to reverify private GPU score/selection artifacts,
formal run trees, checkpoints, or raw BACI provenance. Full mode does require and hash every
external `data/processed_v2/` source named by the schema-8 paper interface, including all 24 GBDT
candidate files. The public receipt enforces the frozen governed number inventory, exact JSON/TeX
source/value-map agreement, and reviewed canonical key/value-map digest. This repository
digest is a drift/freeze contract, not an external signature against simultaneous
verifier-code changes. That threat model requires an independently protected
release tag, cryptographic signature, or DOI-backed archival record and digest. A
repository-only clone may instead run:

```bash
python tools/artifact_bundles.py verify-index --allow-missing
python tools/generate_v2_benchmark_profile.py --verify --profile repository
python tools/resolve_v2_invalidation.py --verify-public-receipt --profile repository
python tools/release_smoke.py --profile repository
```

Repository mode still verifies the exact receipt/status/scope/maps and every
bound public-Git code, configuration, metric, and source byte. It does not open
or require external `data/processed_v2/` release-asset bytes. Maintainers with
the private provenance checkout may additionally run `--verify-resolved`, which
layers the private raw/formal scientific verifiers on top of the full public
receipt check; public consumers should not need that private command.

## Release checklist — public reviewer candidate status

- [x] Software and project-authored data licenses are separated.
- [x] Third-party source and redistribution conditions are recorded.
- [x] Deterministic bundle planning, manifests, checksum sidecars, size gates, and repository/full
  smoke profiles are implemented.
- [x] Public selection uses exact allowlists and excludes raw data, private score/run trees,
  checkpoints, host state, permission-gated inputs, and maintainer-only controllers.
- [x] The CI branch validates the selected code and artifact boundary.
- [x] Resolved the current-registry hold with a release-eligible sampled human-validation
  receipt (212 of 610 decision records plus all 53 stage definitions) and a public resolution
  receipt. The 398 unsampled decision records are not claimed as individually human-verified.
- [x] Generated and verified the current result and paper-number interfaces under the resolved
  release contract.
- [x] Regenerated and verified the reviewer-snapshot manifests, deterministic archives, and clean
  clones.
- [x] Ran full-payload and cross-platform release QA, including remote CI.
- [x] Published a versioned GitHub prerelease and anonymously redownloaded and verified its indexed
  assets.
- [ ] Confirm final archival metadata and mirror the byte-identical final assets to a DOI-bearing
  archival host.

The remaining unchecked item is the final archival mirror and DOI registration. Manuscript
authorship and venue submission metadata remain outside this artifact-distribution checklist.

Maintainer-only exploratory outputs remain outside both current public surfaces
even when retained in staging for provenance.
