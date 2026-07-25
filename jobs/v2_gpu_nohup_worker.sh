#!/bin/bash
# Direct-host worker for GPU machines that do not expose a batch scheduler.
# Usage (inside an isolated run root):
#   RUN_ROOT=$PWD BASE_PYTHON=/path/to/cuda/python bash jobs/v2_gpu_nohup_worker.sh kge 0 smoke
#   RUN_ROOT=$PWD BASE_PYTHON=/path/to/cuda/python bash jobs/v2_gpu_nohup_worker.sh nbfnet 1 select
set -euo pipefail

FAMILY="${1:?family must be kge or nbfnet}"
PHYSICAL_GPU="${2:?physical GPU index is required}"
MODE="${3:-select}"
if [[ "$FAMILY" != "kge" && "$FAMILY" != "nbfnet" ]]; then
  echo "unknown family: $FAMILY" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "select" ]]; then
  echo "mode must be smoke or select" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the isolated project snapshot}"
BASE_PYTHON="${BASE_PYTHON:?set BASE_PYTHON to the existing CUDA Python}"
if [[ "$MODE" == "select" && -f "$RUN_ROOT/results_v2/gpu_rolling/PILOT_INVALIDATED.json" ]]; then
  echo "refusing formal selection in an invalidated run namespace; create a new RUN_ROOT" >&2
  exit 5
fi
VCU_RAW="${VCU_RAW:-$RUN_ROOT/data/raw}"
NBFNET_PATH="${NBFNET_PATH:-$RUN_ROOT/third_party/NBFNet-PyG}"
OVERLAY="$RUN_ROOT/python_packages_nodeps"
LOG_ROOT="$RUN_ROOT/results_v2/gpu_rolling/logs"
PID_ROOT="$RUN_ROOT/results_v2/gpu_rolling/pids"
MANIFEST_ROOT="$RUN_ROOT/manifest"
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$MANIFEST_ROOT" "$RUN_ROOT/cache" "$RUN_ROOT/tmp" "$RUN_ROOT/torch_extensions"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export VCU_RAW NBFNET_PATH
export PYTHONPATH="$OVERLAY:$NBFNET_PATH:${PYTHONPATH:-}"
export PATH="$(dirname "$BASE_PYTHON"):$PATH"
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export TMPDIR="$RUN_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/torch_extensions"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [[ "$FAMILY" == "nbfnet" && -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module load "${CUDA_MODULE:-cuda/12.6}" "${GCC_MODULE:-gcc/gcc-12}"
fi

cd "$RUN_ROOT"
HOST_SHORT="$(hostname -s)"
WORKER_ID="${WORKER_ID:-${MODE}_${FAMILY}_${HOST_SHORT}_gpu${PHYSICAL_GPU}}"
ENV_REPORT="$MANIFEST_ROOT/env_${WORKER_ID}.json"
"$BASE_PYTHON" tools/v2_gpu_env_check.py \
  --expected-torch 2.6.0+cu126 --expected-pykeen 1.11.1 --require-cuda \
  --forbid-prefix "$OVERLAY" > "$ENV_REPORT"
CHAINS=(sheep cotton aluminium nickel cocoa oilseed-soy)
if [[ -n "${CHAIN_LIST:-}" ]]; then
  IFS=',' read -r -a CHAINS <<< "$CHAIN_LIST"
fi
if [[ "$MODE" == "smoke" ]]; then
  CHAINS=("${SMOKE_CHAIN:-sheep}")
fi

LOCK_DIR="$PID_ROOT/${WORKER_ID}.lock"
if ! mkdir "$LOCK_DIR"; then
  echo "worker lock already exists: $LOCK_DIR" >&2
  exit 4
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"
printf '%s\n' "started_at=$(date -Is)" "host=$HOST_SHORT" "physical_gpu=$PHYSICAL_GPU" \
  "family=$FAMILY" "mode=$MODE" "chains=${CHAINS[*]}" > "$LOCK_DIR/worker.env"
trap 'rc=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date -Is)" "$rc" > "$LOCK_DIR/status"' EXIT

REGISTRY="$MANIFEST_ROOT/${FAMILY}_${MODE}_${HOST_SHORT}_gpu${PHYSICAL_GPU}_jobs.tsv"
if [[ ! -f "$REGISTRY" ]]; then
  printf 'started_at\tfinished_at\tstatus\tfamily\tphysical_gpu\tchain\ttasks\tlog\tcommand\n' > "$REGISTRY"
fi

for CHAIN in "${CHAINS[@]}"; do
  STARTED="$(date -Is)"
    LOG="$LOG_ROOT/${WORKER_ID}_${CHAIN}_tasks-A-B1-B2.log"
  OUTPUT_ROOT="results_v2/gpu_rolling"
  EXTRA=()
  if [[ "$MODE" == "smoke" ]]; then
    OUTPUT_ROOT="results_v2/gpu_smoke/$WORKER_ID"
    if [[ "$FAMILY" == "kge" ]]; then
      # All six models must pass compatibility; the runner records individual
      # failures and emits unfreezable incomplete artifacts instead of
      # aborting at the first failure.
      EXTRA=(--dims 8 --learning-rates 0.01 --epochs 1 --seeds 0 --overwrite)
    else
      EXTRA=(--layers 2 --nbfnet-learning-rates 0.005 --epochs 1 --seeds 0 --overwrite)
    fi
  fi
  CMD=("$BASE_PYTHON" src/v2_gpu_rolling.py select-chain
    --chain "$CHAIN" --family "$FAMILY"
    --candidate-root data/processed_v2 --output-root "$OUTPUT_ROOT"
    --require-cuda "${EXTRA[@]}")
  INPUT_A="data/processed_v2/candidates_${CHAIN}_fold2.csv"
  INPUT_B="data/processed_v2/candidates_firsttime_${CHAIN}_fold2.csv"
  {
    echo "started_at=$STARTED"
    echo "host=$(hostname) family=$FAMILY physical_gpu=$PHYSICAL_GPU visible_gpu=$CUDA_VISIBLE_DEVICES chain=$CHAIN tasks=a,b1,b2 mode=$MODE"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    echo "hashes:"
    sha256sum src/v2_gpu_rolling.py src/v2_gpu_protocol.py src/benchmark.py src/temporal_backtest.py configs/v2_gpu_rolling.json "$INPUT_A" "$INPUT_B"
    printf 'command='; printf '%q ' "${CMD[@]}"; echo
  } > "$LOG"
  set +e
  "${CMD[@]}" >> "$LOG" 2>&1
  RC=$?
  set -e
  FINISHED="$(date -Is)"
  STATUS="passed"
  if [[ $RC -ne 0 ]]; then STATUS="failed:$RC"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' \
    "$STARTED" "$FINISHED" "$STATUS" "$FAMILY" "$PHYSICAL_GPU" "$CHAIN" "a,b1,b2" "$LOG" >> "$REGISTRY"
  printf '%q ' "${CMD[@]}" >> "$REGISTRY"
  printf '\n' >> "$REGISTRY"
  if [[ $RC -ne 0 ]]; then
    echo "job failed; see $LOG" >&2
    exit "$RC"
  fi
done
