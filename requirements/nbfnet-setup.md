# NBFNet-PyG setup (external, not vendored)

The path-GNN baseline (NBFNet) is **not** in this repo. It is the upstream KiddoZhu/NBFNet-PyG.
Point the env var `NBFNET_PATH` at your checkout (set it in `env.sh`); the `src/` scripts and `run_*.sh`
read it and add it to `sys.path`, defaulting to `<repo>/third_party/NBFNet-PyG` when unset.

## Pinned source
- **Repo:** https://github.com/KiddoZhu/NBFNet-PyG.git
- **Commit used:** `0c2d262`
- **Where to put it:** clone to `<repo>/third_party/NBFNet-PyG` (the default `NBFNET_PATH`), or
  anywhere and `export NBFNET_PATH=/your/path` in `env.sh`.

```bash
git clone https://github.com/KiddoZhu/NBFNet-PyG.git third_party/NBFNet-PyG
cd third_party/NBFNet-PyG && git checkout 0c2d262
```

## Why a SEPARATE environment
NBFNet ships a hand-written CUDA kernel (`rspmm`) that is JIT-compiled at import and was written for
an older PyG. It runs in its own conda env (`nbfnet`), distinct from the broader analysis env
(`requirements.txt`) and the minimal rolling-CPU environment
(`v2-cpu-results-lock.txt`). PyKEEN and NBFNet GPU reruns must use the strict v2
historical-selection/freeze/main protocol; neither environment is needed for the
committed rolling CPU references.

For the formal v2.1 rerun, the external selection tree is retained as
retrospective supplemental evidence and must not be described as having been
hash-frozen before selection. Main evaluation instead uses
`RUN_ROOT/private/nbfnet_source_frozen`: a complete matching source snapshot
with no `__pycache__`, `.pyc`, or `.pyo` files and no filesystem write bits.
Both execution hosts bind that snapshot and the run-local JIT `rspmm.so` through
the private formal-gate receipts before main starts and re-verify them around
each assigned chain.

## Required source patches (apply to the checkout)
Written for old PyG; needs 2 edits to run on PyG 2.7 / torch 2.6:

1. **`nbfnet/layers.py` — MessagePassing private-API renames (4 spots).** PyG 2.4+ renamed the dunder
   methods to single-underscore and renamed the inspector call:
   | old | new |
   |---|---|
   | `self.__check_input__` | `self._check_input` |
   | `self.__collect__` | `self._collect` |
   | `self.__fused_user_args__` | `self._fused_user_args` |
   | `self.inspector.distribute(...)` | `self.inspector.collect_param_data(...)` |
2. **`nbfnet/rspmm/source/rspmm.h` — header move.** torch 2.4+ moved
   `ATen/SparseTensorUtils.h` → `ATen/native/SparseTensorUtils.h` (namespace `at::sparse` unchanged).

## Modules / env to load before running NBFNet
```bash
module load cuda/12.6 gcc/gcc-12          # nvcc for the rspmm JIT; GCC>=9 for torch 2.6 (cuda/12.4 + gcc-11 also work)
conda activate nbfnet
export TORCH_CUDA_ARCH_LIST="8.9"          # L4 = sm_89; avoids compiling all arches
```
- `cuda/12.6` (or 12.4): the torch wheel only ships the CUDA *runtime*; the rspmm JIT needs `nvcc`.
- `gcc/gcc-11`: torch 2.6 needs GCC ≥ 9 (system default on this RHEL 8.5 box is GCC 8.5).
- After editing any `rspmm` `.cpp/.h/.cu`, clear the JIT cache or the stale build is reused:
  `rm -rf ~/.cache/torch_extensions/*/rspmm/`

## If NBFNet is unavailable
The data validators, standalone scorer, and rolling CPU references run without
it. A missing NBFNet environment does **not** authorize substituting an
unverified precomputed score column: graph-model claims require a source-locked
run selected on the historical fold and evaluated after the global freeze gate.
