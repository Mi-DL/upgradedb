#!/bin/bash
# Fail-closed direct-host launcher for one frozen v2 main chain/family job.
#
# This script is intentionally separate from v2_gpu_nohup_worker.sh: selection
# and main evaluation have different authority and failure semantics.  It never
# creates MAIN_EVALUATION_STARTED.json itself.  The non-dry-run
# ``evaluate-chain`` process must claim that immutable marker before it can open
# a main candidate table.
#
# Usage (normally under nohup on any shared-filesystem GPU node):
#   MAIN_EVALUATION_CONFIRM=FROZEN_36_HASH_VERIFIED \
#   RUN_ROOT=/path/to/isolated-run-root \
#   BASE_PYTHON=/path/to/audited-cuda-python \
#   SYNC_MANIFEST=/path/to/isolated-run-root/results_v2/gpu_rolling/runs/<run-id>/STEP3_SYNC_MANIFEST.sha256 \
#   SYNC_MANIFEST_SHA256=<sha256-of-that-manifest> \
#   bash jobs/v2_gpu_main_worker.sh kge 0 sheep
set -euo pipefail

FAMILY="${1:?family must be kge or nbfnet}"
PHYSICAL_GPU="${2:?physical GPU index is required}"
CHAIN="${3:?chain is required}"

case "$FAMILY" in
  kge|nbfnet) ;;
  *) echo "family must be kge or nbfnet: $FAMILY" >&2; exit 2 ;;
esac
case "$CHAIN" in
  sheep|cotton|aluminium|nickel|cocoa|oilseed-soy) ;;
  *) echo "unknown/non-canonical chain: $CHAIN" >&2; exit 2 ;;
esac
if [[ ! "$PHYSICAL_GPU" =~ ^[0-9]+$ ]]; then
  echo "physical GPU must be a non-negative integer: $PHYSICAL_GPU" >&2
  exit 2
fi
if [[ "${MAIN_EVALUATION_CONFIRM:-}" != "FROZEN_36_HASH_VERIFIED" ]]; then
  echo "refusing main evaluation without MAIN_EVALUATION_CONFIRM=FROZEN_36_HASH_VERIFIED" >&2
  exit 3
fi

RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the isolated shared project snapshot}"
BASE_PYTHON="${BASE_PYTHON:?set BASE_PYTHON to the audited CUDA Python}"
SYNC_MANIFEST="${SYNC_MANIFEST:?set SYNC_MANIFEST to the selection-time Step-3 sync manifest}"
SYNC_MANIFEST_SHA256="${SYNC_MANIFEST_SHA256:?set SYNC_MANIFEST_SHA256 to its externally recorded SHA-256}"
if [[ ! "$SYNC_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "SYNC_MANIFEST_SHA256 must be 64 lowercase hexadecimal characters" >&2
  exit 3
fi
if [[ ! -d "$RUN_ROOT" || ! -x "$BASE_PYTHON" || ! -f "$SYNC_MANIFEST" ]]; then
  echo "RUN_ROOT, BASE_PYTHON, or SYNC_MANIFEST is missing/unusable" >&2
  exit 3
fi

RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
SYNC_MANIFEST="$(readlink -f "$SYNC_MANIFEST")"
case "$SYNC_MANIFEST" in
  "$RUN_ROOT"/*) ;;
  *) echo "SYNC_MANIFEST must live below RUN_ROOT" >&2; exit 3 ;;
esac

OUTPUT_ROOT="$RUN_ROOT/results_v2/gpu_rolling"
MANIFEST="$OUTPUT_ROOT/frozen_manifest.json"
RUN_CONFIG="$RUN_ROOT/configs/v2_gpu_rolling.json"
CANDIDATE_ROOT="$RUN_ROOT/data/processed_v2"
OVERLAY="$RUN_ROOT/python_packages_nodeps"
VCU_RAW="${VCU_RAW:-$RUN_ROOT/data/raw}"
NBFNET_PATH="${NBFNET_PATH:-$RUN_ROOT/third_party/NBFNet-PyG}"
SEEDS="${SEEDS:-0,1,2,3,4}"
if [[ -f "$OUTPUT_ROOT/PILOT_INVALIDATED.json" ]]; then
  echo "refusing main evaluation in an invalidated run namespace" >&2
  exit 3
fi
if [[ ! -f "$MANIFEST" || ! -f "$RUN_CONFIG" ]]; then
  echo "canonical frozen manifest or run config is missing" >&2
  exit 3
fi
if [[ ! -f "$CANDIDATE_ROOT/candidates_${CHAIN}.csv" || \
      ! -f "$CANDIDATE_ROOT/candidates_firsttime_${CHAIN}.csv" ]]; then
  echo "canonical main candidate inputs are missing for chain=$CHAIN" >&2
  exit 3
fi
if [[ ! -d "$VCU_RAW" ]]; then
  echo "read-only raw BACI root is missing: $VCU_RAW" >&2
  exit 3
fi
if [[ "$FAMILY" == "nbfnet" && ! -d "$NBFNET_PATH" ]]; then
  echo "NBFNet source tree is missing: $NBFNET_PATH" >&2
  exit 3
fi

LOG_ROOT="$OUTPUT_ROOT/logs/main"
CLAIM_ROOT="$OUTPUT_ROOT/main_job_claims"
PROVENANCE_ROOT="$RUN_ROOT/manifest"
mkdir -p "$LOG_ROOT" "$CLAIM_ROOT" "$PROVENANCE_ROOT" \
  "$RUN_ROOT/cache" "$RUN_ROOT/tmp" "$RUN_ROOT/torch_extensions"

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
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORKER_ID="${WORKER_ID:-main_${CHAIN}_${FAMILY}_${HOST_SHORT}_gpu${PHYSICAL_GPU}_${STAMP}_$$}"
if [[ ! "$WORKER_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "WORKER_ID contains unsafe characters: $WORKER_ID" >&2
  exit 3
fi
ENV_REPORT="$PROVENANCE_ROOT/env_${WORKER_ID}.json"
PREFLIGHT_LOG="$LOG_ROOT/${WORKER_ID}_preflight.log"
RUN_LOG="$LOG_ROOT/${WORKER_ID}.log"

# Pin the exact selection-time snapshot before trusting its hash list.  The
# verifier covers runner/protocol/config/graph-building code, chain registry,
# and all fold2 candidate inputs used to create the 36 selections.
ACTUAL_SYNC_SHA256="$(sha256sum "$SYNC_MANIFEST" | awk '{print $1}')"
if [[ "$ACTUAL_SYNC_SHA256" != "$SYNC_MANIFEST_SHA256" ]]; then
  echo "selection-time sync-manifest digest mismatch" >&2
  exit 3
fi
"$BASE_PYTHON" tools/step3_sync_manifest.py --verify --output "$SYNC_MANIFEST"

"$BASE_PYTHON" tools/v2_gpu_env_check.py \
  --expected-torch 2.6.0+cu126 --expected-pykeen 1.11.1 --require-cuda \
  --forbid-prefix "$OVERLAY" > "$ENV_REPORT"
nvidia-smi -i "$PHYSICAL_GPU" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$LOG_ROOT/${WORKER_ID}_gpu.txt"

CMD=("$BASE_PYTHON" src/v2_gpu_rolling.py evaluate-chain
  --chain "$CHAIN" --family "$FAMILY"
  --candidate-root "$CANDIDATE_ROOT" --output-root "$OUTPUT_ROOT"
  --run-config "$RUN_CONFIG" --manifest "$MANIFEST"
  --seeds "$SEEDS" --require-cuda)

# Gate 1 is read-only: evaluate-chain --dry-run re-hashes every selection,
# requires exactly 6 x 3 x 2 = 36 entries, checks config/seeds, and neither
# opens main candidates nor creates the main-start marker.
{
  echo "preflight_started_at=$(date -Is)"
  echo "worker_id=$WORKER_ID host=$(hostname) physical_gpu=$PHYSICAL_GPU chain=$CHAIN family=$FAMILY"
  echo "sync_manifest=$SYNC_MANIFEST sync_manifest_sha256=$SYNC_MANIFEST_SHA256"
  sha256sum "$MANIFEST" "$RUN_CONFIG" jobs/v2_gpu_main_worker.sh
  printf 'dry_run_command='; printf '%q ' "${CMD[@]}"; echo ' --dry-run'
  "${CMD[@]}" --dry-run
  echo "preflight_finished_at=$(date -Is)"
} > "$PREFLIGHT_LOG" 2>&1

# A shared-filesystem mkdir is the per-chain/family single-writer claim.  It is
# deliberately permanent on both success and failure; retries require explicit
# human adjudication and can never silently overwrite a partial formal run.
CLAIM_DIR="$CLAIM_ROOT/${CHAIN}_${FAMILY}.lock"
if ! mkdir "$CLAIM_DIR"; then
  echo "main chain/family is already claimed: $CLAIM_DIR" >&2
  exit 4
fi
printf '%s\n' "$$" > "$CLAIM_DIR/pid"
printf '%s\n' \
  "worker_id=$WORKER_ID" "claimed_at=$(date -Is)" "host=$(hostname)" \
  "physical_gpu=$PHYSICAL_GPU" "chain=$CHAIN" "family=$FAMILY" \
  "manifest=$MANIFEST" "manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')" \
  "sync_manifest=$SYNC_MANIFEST" "sync_manifest_sha256=$SYNC_MANIFEST_SHA256" \
  > "$CLAIM_DIR/worker.env"
finish_claim() {
  rc=$?
  temporary="$CLAIM_DIR/status.tmp.$$"
  printf 'finished_at=%s\nexit_code=%s\nrun_log=%s\n' "$(date -Is)" "$rc" "$RUN_LOG" > "$temporary"
  mv "$temporary" "$CLAIM_DIR/status"
}
trap finish_claim EXIT

# Gate 2 is inside the actual runner and repeats all freeze/hash/config checks.
# Only that process can atomically claim MAIN_EVALUATION_STARTED.json; there is
# no worker-side marker shortcut.
{
  echo "started_at=$(date -Is)"
  printf 'command='; printf '%q ' "${CMD[@]}"; echo
  "${CMD[@]}"
  echo "finished_at=$(date -Is)"
} > "$RUN_LOG" 2>&1
