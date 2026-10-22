#!/usr/bin/env bash
# Submitted explicitly by submit_stage.sh. No child job submission.
set -Eeuo pipefail
session=${1:?};stage=${2:?};export ON_EVAL_ENV_FILE=${3:?}
source "$ON_EVAL_ENV_FILE"
release_root=${ON_EVAL_RELEASE_ROOT:?}
source "$release_root/scripts/fe01_a01_compare/compute_env.sh"
cd "$release_root"
exec "$ON_EVAL_PYTHON" -B -m fe01_a01_compare.cli worker --session "$session" --stage "$stage" --index "${SLURM_ARRAY_TASK_ID:?}" --device cpu
