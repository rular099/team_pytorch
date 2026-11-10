#!/usr/bin/env bash
set -Eeuo pipefail
export FE01_V4_ENV_FILE=${1:?}
export FE01_V4_RUN_ID=${2:?}
v4s_root=${3:?}
shift 3
exec srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
  bash "$v4s_root/scripts/fe01_v4_compute_submit/control_step.sh" \
  "$FE01_V4_ENV_FILE" "$FE01_V4_RUN_ID" "$v4s_root" "$@"
