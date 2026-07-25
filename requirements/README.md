# Reproduction environments and source data

This directory separates the frozen environment for the external current-registry CPU artifact
from broader analysis and GPU environments.

## Files

| File | Purpose |
|---|---|
| `v2-cpu-results-lock.txt` | Exact minimal pins recorded for the frozen rolling CPU artifact |
| `verify_v2_cpu_results_env.py` | Checks Python/packages, result runtime metadata, platform, and source hash |
| `requirements.txt` | Curated broader analysis/KGE dependencies; not the CPU-results lock |
| `requirements-lock.txt` | Full archived GPU-capable analysis snapshot; platform-specific and not a cross-platform byte-reproducibility promise |
| `environment.yml` | Conda alternative for the broader analysis/KGE stack |
| `v2-gpu-nodeps-lock.txt` | Isolated no-dependency overlay pins for the cluster GPU environment |
| `v2-gpu-overlay.txt` | Minimal PyKEEN overlay for an already validated CUDA base stack |
| `nbfnet-setup.md` | External NBFNet checkout, commit, patches, and CUDA setup |
| `ultra-formal.md` | Frozen external-pretrained ULTRA-ZS protocol, private phase order, and public sanitizer |
| `DATA.md` | Raw archive versions, provenance, and redistribution boundaries |

## Frozen rolling CPU result environment

The result-generation environment was:

- CPython 3.12.13;
- Windows `Windows-11-10.0.26200-SP0`;
- NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.0;
- scikit-learn 1.9.0, joblib 1.5.3, and threadpoolctl 3.6.0.

Create a clean environment and verify it:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/v2-cpu-results-lock.txt
python requirements/verify_v2_cpu_results_env.py
```

Use `--strict-platform` when validating the original result host. Exact package
pins are necessary environment evidence, but they do not guarantee
byte-identical numerical output across operating systems, CPU/BLAS/OpenMP
implementations, or different wheel builds.

This lock verifies the rolling-CPU artifact after its governed result file is installed. The
code-only active-hold snapshot does not contain that result. The lock intentionally omits torch and
PyG, which are imported by part of the repository test
collection. Use a separate test environment for complete repository checks:

```bash
python -m venv .venv-tests
. .venv-tests/bin/activate
python -m pip install -r benchmark/upgrade-bench-v2/requirements.txt \
  scikit-learn==1.9.0 pycountry==26.2.16
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric==2.7.0
python -m unittest discover -s tests -p "test_*.py" -v
```

## Reproduce v2 artifacts

The canonical v2 tables will be distributed as release assets and appear under
`data/processed_v2/` after extraction. Authorized lightweight rolling/audit outputs appear under
`results_v2/` after the active hold is resolved:

```bash
python tools/validate_v2.py
python tools/v2_rolling_cpu_baselines.py --verify-output
python tools/v2_gbdt_baselines.py --verify-output
python tools/audit_v2.py --verify-output
```

The formal GBDT artifact is a reviewer-motivated CPU reference using
`HistGradientBoostingClassifier` and the same early-window task features as the logistic references.
Its four-point grid is selected only on grouped historical folds, all 18 models are frozen before
main access, and B2 scores all lanes before conditioning. It does not require a GPU.

To rebuild candidates from private raw archives, set `VCU_RAW` and run both the
main and historical selection windows:

```bash
python src/temporal_backtest.py --chain cocoa --upgrade --enum-only \
  --aggregation calendar_mean --output-dir data/processed_v2
VCU_FOLD=fold2 python src/temporal_backtest.py --chain cocoa --upgrade --enum-only \
  --aggregation calendar_mean --output-dir data/processed_v2
```

Repeat for all six chain IDs and with `--first-time` for Track B, then run
`tools/build_v2_views.py` for the main snapshot and with `--suffix _fold2` for
the selection snapshot.

The official protocol selects on `1998-2002 -> 2008-2012`, freezes every model
and preprocessing choice, and evaluates once on the complete
`2008-2012 -> 2018-2022` cohort. Any shipped same-window `train/test` field is a
transductive diagnostic grouped by `(exporter, stage)`; it is not a lane-level
official split and does not replace rolling evaluation.

## Broader analysis and GPU environments

Use `requirements.txt` (or `environment.yml`) for tools that additionally need
PyTorch, PyG, PyKEEN, plotting, or network clients. Install a PyTorch build that
matches the host before the remaining dependencies. This broad stack is kept
separate from the authoritative CPU-results lock.

NBFNet and strict GPU reruns use an independently validated CUDA environment and
the overlays documented here. Never infer GPU readiness from an environment
name: verify the exact torch/PyG/CUDA stack, compiler, accelerator access, and
external source revision on the execution host before running a graph result.

The formal ULTRA-ZS run is a separate environment contract. It fixes one `ultra_4g` checkpoint,
requires CUDA plus native `torch_scatter` and `rspmm`, and rejects compatibility shims and the
feasibility-smoke fallback. It uses each chain's early graph but no UpgradeBench label for checkpoint
choice, training, fine-tuning, calibration, or score transformation. Its formal gate requires a
complete sealed chain-score inventory before label access, all chain--task evaluation records, and
the prescribed same-process repeat. External pretraining resources are not matched to the trained graph families,
and the single checkpoint has no training-seed interval; this is a descriptive zero-shot reference,
not graph-free cold start, a fair-compute champion, or a significance claim.

After private evaluation verification, the public result pair is produced and checked as documented
in `ultra-formal.md`:

```bash
python tools/summarize_v2_ultra_results.py --check-only
python tools/summarize_v2_ultra_results.py
python tools/summarize_v2_ultra_results.py --verify-output
```

Only the last command is needed by public consumers. It does not open the checkpoint, score files,
raw BACI archive, or `results_v2/ultra_formal/`. The current aggregate pair is intentionally absent
from this active-hold snapshot and becomes public only through the governed promotion and receipt
sequence.

## Maintainer workspace boundary

Maintainer-only exploratory outputs are not selected into the public manifest
or bundle catalog and are not part of this environment contract.
