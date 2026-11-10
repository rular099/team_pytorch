#!/usr/bin/env bash
set -Eeuo pipefail
v4s_manifest=${1:?}; v4s_pin=${2:?}; export FE01_V4_ENV_FILE=${3:?}
v4s_root=${4:?}; v4s_device=${5:?}
# Private environment and device initialization belong to the allocated step.
v4s_args=(--nodes=1 --ntasks=1 --kill-on-bad-exit=1)
if [[ "$v4s_device" == cuda ]]; then
  v4s_args+=("--gres=${ON_EVAL_GRES:?}")
fi
exec srun "${v4s_args[@]}" bash "$v4s_root/scripts/fe01_v4_compute_submit/pool_step.sh" \
  "$v4s_manifest" "$v4s_pin" "$FE01_V4_ENV_FILE" "$v4s_root"
