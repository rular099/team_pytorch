#!/usr/bin/env bash
# DCU initialization/check must occur inside the actual srun step.
set -Eeuo pipefail
session=${1:?};stage=${2:?};export ON_EVAL_ENV_FILE=${3:?}
source "$ON_EVAL_ENV_FILE"
release_root=${ON_EVAL_RELEASE_ROOT:?}
exec srun --nodes=1 --ntasks=1 --gres="${ON_EVAL_GRES:?}" --kill-on-bad-exit=1 \
  bash "$release_root/scripts/fe01_a01_compare/worker_dcu.sh" "$session" "$stage" "$ON_EVAL_ENV_FILE"
