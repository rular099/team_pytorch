#!/usr/bin/env bash
# COMPUTE NODE: give each distributed rank its own import/config caches.
set -euo pipefail
export RANK="${SLURM_PROCID:?}" WORLD_SIZE="${A01_WORLD_SIZE:?}" LOCAL_RANK="${SLURM_LOCALID:-0}"
cache_root="${A01_OUTPUT_ROOT:?}/.cache/${SLURM_JOB_ID:?}/rank${RANK}"
export MPLCONFIGDIR="$cache_root/matplotlib"
export XDG_CACHE_HOME="$cache_root/xdg"
export SEISBENCH_CACHE_ROOT="$cache_root/seisbench"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$SEISBENCH_CACHE_ROOT"
exec "$@"
