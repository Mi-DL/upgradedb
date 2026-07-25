# Portable v2 GPU selection and main-evaluation operations

This runbook is host-neutral. It applies to an isolated run root on one or more
GPU nodes that share the same filesystem. Site-specific host inventories,
accounts, queues, and absolute storage paths must remain outside the public
repository.

The formal main phase is a one-shot operation. It is authorized only after all
twelve historical-fold chain/family selections finish, the exact 36 task
selections are frozen, and the selection-time source/input manifest digest is
independently recorded.

## Required variables

Set these variables for the local site; the values below are placeholders:

```bash
export RUN_ROOT=/path/to/isolated-run-root
export BASE_PYTHON=/path/to/audited-cuda-python
export VCU_RAW=/path/to/read-only-raw-data
export NBFNET_PATH=/path/to/NBFNet-PyG
export SYNC_MANIFEST="$RUN_ROOT/results_v2/gpu_rolling/runs/<run-id>/STEP3_SYNC_MANIFEST.sha256"
```

`RUN_ROOT` must contain the versioned project snapshot, corrected
`data/processed_v2` tables, and a clean `results_v2/gpu_rolling` namespace.
Raw data and external NBFNet source may live elsewhere and are supplied through
environment variables. Do not commit a host inventory or the values above.

## Pin the selection-time snapshot

Before formal selection, generate the run-specific manifest explicitly:

```bash
cd "$RUN_ROOT"
"$BASE_PYTHON" tools/step3_sync_manifest.py --output "$SYNC_MANIFEST"
sha256sum "$SYNC_MANIFEST"
```

Record that manifest digest outside the run namespace. Do not regenerate the
manifest after selection starts. A later mismatch requires adjudication, never
an in-place update.

The manifest pins all 24 corrected A/B candidate CSV byte streams (main and
historical), the six registries, registry evidence/audit, raw-label audit, B1
coverage audit, run config, runner, cache-reader dependency, workers, and
environment gate. Computing SHA-256 is a label-blind provenance operation: the
manifest generator never parses a main table, and semantic main label access
remains exclusively behind the immutable main-start gate.

The immutable run config must have `execution_status` set exactly to
`FORMAL_RUN_AUTHORIZED` before this manifest is generated. The runner rejects
every other value. For a future cohort rebuild, keep its run-specific config
unauthorized until the corrected data/audit gate passes; the reviewed authorized
value is then part of the pinned run-config hash for both selection and main
evaluation.

Use `jobs/v2_gpu_nohup_worker.sh` for direct-host smoke/selection or the portable
PBS selection launcher where a scheduler is available. Each chain/family job
scores all three tasks from one shared representation grid. Across the formal
selection phase, assign each of these twelve combinations exactly once:

```text
sheep/{kge,nbfnet}          cotton/{kge,nbfnet}
aluminium/{kge,nbfnet}      nickel/{kge,nbfnet}
cocoa/{kge,nbfnet}          oilseed-soy/{kge,nbfnet}
```

Multiple nodes may participate only when they see the same `RUN_ROOT`; otherwise
their locks, freeze state, and outputs are not globally coordinated.

## Global freeze gate

After all twelve selection jobs succeed, verify that
`MAIN_EVALUATION_STARTED.json` does not exist, then create the one global freeze
without `--overwrite`:

```bash
cd "$RUN_ROOT"
"$BASE_PYTHON" src/v2_gpu_rolling.py freeze \
  --output-root results_v2/gpu_rolling \
  --chains sheep,cotton,aluminium,nickel,cocoa,oilseed-soy \
  --tracks a,b1,b2 --families kge,nbfnet
```

The command must report exactly 36 complete selections. `evaluate-chain
--dry-run` independently re-hashes all 36 files, rejects missing or extra
entries, checks the frozen run-config hash and evaluation seeds, and neither
opens a main candidate table nor creates the main marker.

## One main chain/family per GPU

Retrieve the selection-time manifest digest from its independent record and set
it without recomputing it from the mutable run directory:

```bash
export RECORDED_SYNC_SHA256=REPLACE_WITH_RECORDED_64_HEX_DIGEST

nohup env \
  RUN_ROOT="$RUN_ROOT" \
  BASE_PYTHON="$BASE_PYTHON" \
  VCU_RAW="$VCU_RAW" \
  NBFNET_PATH="$NBFNET_PATH" \
  MAIN_EVALUATION_CONFIRM=FROZEN_36_HASH_VERIFIED \
  SYNC_MANIFEST="$SYNC_MANIFEST" \
  SYNC_MANIFEST_SHA256="$RECORDED_SYNC_SHA256" \
  bash jobs/v2_gpu_main_worker.sh kge 0 sheep \
  > results_v2/gpu_rolling/logs/main/worker-sheep-kge-gpu0.log 2>&1 &
```

Repeat only for chain/family combinations assigned to that worker. CUDA and compiler module
names can be overridden with `CUDA_MODULE` and `GCC_MODULE`; sites without an
environment-modules setup use their already audited base environment.

**Serialize the first main start.** Initialize the global marker through one
reviewed launcher before scaling to the remaining workers. Start one light
sentinel job, then wait until the marker is complete and bound to the canonical
freeze:

```bash
"$BASE_PYTHON" -c 'import hashlib,json,os,pathlib; r=pathlib.Path(os.environ["RUN_ROOT"]); f=r/"results_v2/gpu_rolling/frozen_manifest.json"; m=json.loads((f.parent/"MAIN_EVALUATION_STARTED.json").read_text()); assert m["schema_version"]=="upgrade-bench-v2/main-start/1"; assert m["manifest_sha256"]==hashlib.sha256(f.read_bytes()).hexdigest()'
```

Only after that command succeeds may the remaining GPUs be started, one command
at a time. They verify the already complete marker against the same run and
manifest before opening any main candidate table. This launch barrier is part
of operations provenance; it does not create or edit the marker itself.

## Worker gates

The main worker fails closed unless all of the following hold:

1. the run config carries the exact formal-execution authorization state;
2. the explicit main-evaluation confirmation token is present;
3. the externally pinned sync-manifest digest matches and every selection-time
   source/config/fold2-input hash still verifies;
4. the audited CUDA/PyKEEN environment passes;
5. `evaluate-chain --dry-run` verifies the exact 36-entry freeze, every
   selection hash, the run-config hash, and frozen seeds;
6. the requested chain/family matches its exclusive assignment authorization;
7. the real runner repeats all gates and atomically creates or verifies the
   immutable global main-start marker before opening a main table.

No formal command uses `--overwrite`. Each assignment has one formal execution,
and the run directory remains immutable for verification.

Raw selections, caches, scores, assignment receipts, logs, environment reports, and run status
under `results_v2/gpu_rolling/` remain private. After a complete valid run,
`tools/summarize_v2_gpu_results.py` is the only promotion path to the allowlisted
public summary JSON/CSV. Monitor the private run read-only with accelerator
status, log tails, and worker status files, and never rerun `freeze` after the
global main marker exists.
