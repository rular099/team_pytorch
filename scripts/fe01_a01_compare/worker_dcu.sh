#!/usr/bin/env bash
set -Eeuo pipefail
session=${1:?};stage=${2:?};export ON_EVAL_ENV_FILE=${3:?}
release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$release_root/scripts/fe01_a01_compare/compute_env.sh"
cd "$release_root"
"$ON_EVAL_PYTHON" -B -c 'import torch; print("ON_V4_DEVICE",torch.__version__,torch.version.hip,torch.cuda.is_available(),flush=True); assert torch.cuda.is_available(), "DCU unavailable inside allocated srun step"'
exec "$ON_EVAL_PYTHON" -B -m fe01_a01_compare.cli worker --session "$session" --stage "$stage" --index "${SLURM_ARRAY_TASK_ID:?}" --device cuda
