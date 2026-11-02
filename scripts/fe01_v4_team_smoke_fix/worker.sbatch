#!/usr/bin/env bash
# Explicit TEAM task 3 repair; GPU checks occur inside the real srun step.
set -Eeuo pipefail
session=${1:?};export ON_EVAL_ENV_FILE=${2:?};fix_root=${3:?}
source "$ON_EVAL_ENV_FILE"
exec srun --nodes=1 --ntasks=1 --gres="${ON_EVAL_GRES:?}" --kill-on-bad-exit=1 \
  bash "$fix_root/scripts/fe01_v4_team_smoke_fix/worker.sh" "$session" "$ON_EVAL_ENV_FILE" "$fix_root"
