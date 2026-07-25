# Running UPGRADE-BENCH locally

> **Public reviewer release candidate.** This checkout contains the current registry, evaluator
> code, tests, configurations, release policy, audit evidence, authorized sanitized result pairs,
> benchmark profile, paper-number map, sampled human-validation receipt, `RESOLVED` receipt, and
> data-artifact index. The external `data/processed_v2/` tables are distributed as versioned GitHub
> Release assets rather than Git objects. Repository-only checks validate the checked-in surface;
> checks that traverse the processed tables require the mounted release bundles.

The repository is path-agnostic. Scripts resolve the repository from their own
location and accept machine-specific paths through `VCU_RAW`, `NBFNET_PATH`,
`PYTHON`, and `VCU_CHAINS_DIR`.

## What needs raw data or a GPU?

| Operation | Raw archives | GPU |
|---|---:|---:|
| Validate downloaded release tables and CPU results | no | no |
| Rerun the rolling CPU baselines from mounted `data/processed_v2/` assets | no | no |
| Rebuild v2 candidates and independently audit labels | BACI and Gravity | no |
| Rerun KGE/path-GNN baselines | BACI and Gravity | yes |
| Verify the formal GBDT JSON/CSV pair | no | no |
| Verify the sanitized ULTRA-ZS public pair | no | no |
| Rerun the formal ULTRA-ZS scoring/evaluation | BACI plus frozen checkpoint/source | yes |

The core candidate tables and rolling CPU baselines require BACI and Gravity;
permission-gated institutional inputs are not part of the public benchmark.

## Reproduce the frozen v2 CPU result environment

The frozen current-registry rolling CPU artifact was generated with CPython 3.12.13 on
`Windows-11-10.0.26200-SP0`. Its exact minimal Python package versions are in
`requirements/v2-cpu-results-lock.txt`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/v2-cpu-results-lock.txt
python requirements/verify_v2_cpu_results_env.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The verifier checks the Python implementation/version, every minimal package
pin, the runtime metadata inside
`results_v2/metrics/rolling_cpu_baselines.json`, and the recorded rolling-script
hash. That result file is included in the current public repository. Use `--strict-platform` to
require the original Windows platform.

Exact pins do **not** imply byte-identical results across operating systems,
CPU instruction sets, BLAS/OpenMP implementations, or wheel builds. A different
platform is reported explicitly; it is a warning unless `--strict-platform` is
used. `requirements/requirements.txt` and `requirements/requirements-lock.txt`
describe the broader analysis/KGE stack and are not the authoritative lock for
the frozen rolling CPU artifact.

The CPU-results lock is not a complete repository-test environment. In a
separate environment, install the dependencies imported by the public test
collection:

```bash
python -m venv .venv-tests
. .venv-tests/bin/activate
python -m pip install -r benchmark/upgrade-bench-v2/requirements.txt \
  scikit-learn==1.9.0 pycountry==26.2.16
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric==2.7.0
```

On Windows PowerShell, activate with `.venv-tests\Scripts\Activate.ps1`.

## Verify the repository-only release candidate

```bash
python -m unittest \
  tests.test_release_distribution.PublicReleaseAuditTests \
  tests.test_release_distribution.RepositorySizeGateTests \
  tests.test_upgrade_bench_v2_package \
  tests.test_chain_registry_audit \
  tests.test_registry_curation_protocol \
  tests.test_registry_revision \
  tests.test_registry_lexicon_negative_control -v
python tools/public_release_audit.py --planned-only
python tools/release_manifest.py --verify --scope all
python tools/audit_chain_registry.py --check
python tools/test_split.py
```

The raw-dependent registry tests skip only their source-rebuild cases when the pinned BACI archive
is not mounted; committed ledger, rule, and claim-boundary checks still run. To reproduce the
registry evidence byte-for-byte, mount `BACI_HS92_V202401b.zip` under `VCU_RAW` and run:

```bash
python tools/build_registry_evidence.py --check
```

## Verify the installed full payload

After the governed main/history release assets and authorized result pairs are installed, run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tools/validate_v2.py
python tools/v2_rolling_cpu_baselines.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/summarize_v2_loco_results.py --verify-output
python tools/summarize_v2_ultra_results.py --verify-output
```

The last two commands verify only authorized sanitized aggregate pairs; they do not open private
score trees, checkpoints, raw BACI archives, or host provenance.

After the main and historical release bundles are extracted at the repository root, the canonical
v2 inputs are mounted under `data/processed_v2/`; authorized lightweight CPU metrics and audit
artifacts are then available under `results_v2/`. The rolling result is produced by selecting
on the historical `fold2` window, freezing all choices, and then evaluating the
complete main cohort.

The `train/test` column in shipped tables is only a same-window transductive
diagnostic. It hashes `(exporter, stage)` as one group, so all destinations for
the same capability/entry event remain together. It is not the official
historical-selection-to-main evaluation split.

## Raw-data rebuild

Copy the source archives into a private directory and point `VCU_RAW` to it.
See `requirements/DATA.md` for filenames, versions, and licenses. The raw
archives are never written into release bundles.

```bash
export VCU_RAW=/private/path/to/raw
python src/temporal_backtest.py --chain cocoa --upgrade --enum-only \
  --aggregation calendar_mean --output-dir data/processed_v2
VCU_FOLD=fold2 python src/temporal_backtest.py --chain cocoa --upgrade --enum-only \
  --aggregation calendar_mean --output-dir data/processed_v2
```

Repeat with `--first-time` to construct Track B candidates. After all chains and
both windows are rebuilt, materialize B1/B2 views and run the independent raw
audit:

```bash
python tools/build_v2_views.py
python tools/build_v2_views.py --suffix _fold2
python tools/audit_v2.py
```

## GPU environments

GPU execution is isolated from the CPU/release environment. Follow the v2 job
configuration and the host-specific overlay/lock files in `requirements/`.
NBFNet additionally requires its external checkout and CUDA extension setup.

For ULTRA, follow the frozen formal protocol in `requirements/ultra-formal.md`. It requires the
frozen `ultra_4g`
checkpoint, native `torch_scatter` and `rspmm`, six candidate-only score components sealed before
label access, then 18 chain--task evaluations and the sheep same-process repeat. `ULTRA-ZS` uses the
target early graph and means no benchmark-label adaptation; it is not graph-free cold start. Keep
`results_v2/ultra_formal/`, checkpoints, scores, receipts, JIT caches, and host logs private.

Validate the exact torch/PyG/CUDA stack, compiler, external source revision, and
accelerator access on the execution host before a graph rerun. Keep that GPU
environment separate from the locked CPU-results environment above.

## Public-release preflight

During development, run the non-mutating selector/privacy check:

```bash
python tools/public_release_audit.py --planned-only
```

The planned-only audit and repository-profile clean clone prove that the checked-in surface obeys
the public selection, privacy, manifest, and size boundaries. They do not validate external
processed-data bytes; run the full-payload profile with the mounted release bundles for that scope.
The current reviewer release candidate includes the resolved review/result receipts, frozen data
index and assets, and has passed remote CI and anonymous asset-download checksum checks. Final
archival publication and its DOI remain separate. Test a fresh-history clone instead of pushing any
maintainer staging history:

```bash
python tools/release_clean_clone.py --profile repository
python tools/release_clean_clone.py --profile full --artifacts-dir dist/final-release
```

The tool uses temporary directories by default and never uploads. Use
`--keep-dir <new-path>` only for an inspectable local export; the path must not
already exist. Raw BACI archives, filtered caches, logs, GPU scores, selection
artifacts, checkpoints, and host-specific run state are forbidden from both the
public manifest and bundle index.

## Maintainer workspace boundary

Exploratory outputs may remain in a maintainer workspace for provenance, but the
public manifest and bundle index exclude them.
