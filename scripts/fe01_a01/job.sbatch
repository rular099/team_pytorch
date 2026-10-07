#!/usr/bin/env bash
# Resources are supplied explicitly by submit.sh, no nested sbatch.
set -euo pipefail
stage=${1:?environment|audit|diagnostics|pilot|train|eval|pack}
source "${A01_CODE_ROOT:?}/scripts/fe01_a01/env.sh"
index=${SLURM_ARRAY_TASK_ID:-0}
args=(scripts/fe01_a01/run.py "$stage" --index "$index" --output "$A01_OUTPUT_ROOT" --device cuda --world "$((A01_TRAIN_NODES*A01_DEVICES_PER_NODE))")
case "$stage" in
 environment) args+=(--output "$A01_OUTPUT_ROOT/environment");;
 audit|diagnostics|pilot|eval|pack) ;;
 train)
   export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   export MASTER_PORT=$((A01_MASTER_PORT+index))
   export A01_WORLD_SIZE=$((SLURM_JOB_NUM_NODES*A01_DEVICES_PER_NODE))
   exec srun --kill-on-bad-exit=1 --ntasks="$A01_WORLD_SIZE" --ntasks-per-node="$A01_DEVICES_PER_NODE" \
     bash -c 'export RANK="${SLURM_PROCID:?}" WORLD_SIZE="${A01_WORLD_SIZE:?}" LOCAL_RANK="${SLURM_LOCALID:-0}"; exec "$@"' \
     a01-rank "$A01_PYTHON" "${args[@]}"
   ;;
 *) echo 'Unsupported stage' >&2;exit 2;;
esac
exec "$A01_PYTHON" "${args[@]}"
