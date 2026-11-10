#!/usr/bin/env bash
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Allocated compute node required}"
: "${SLURM_STEP_ID:?Allocated srun step required}"
v4s_manifest=${1:?}; v4s_pin=${2:?}
export FE01_V4_ENV_FILE=${3:?}; export ON_EVAL_ENV_FILE=$FE01_V4_ENV_FILE
v4s_root=${4:?}
set +u
source "$FE01_V4_ENV_FILE"
source "${ON_EVAL_RELEASE_ROOT:?}/scripts/fe01_a01_compare/compute_env.sh"
set -u
cd "$ON_EVAL_RELEASE_ROOT"
exec "$ON_EVAL_PYTHON" -B "$v4s_root/scripts/fe01_v4_compute_submit/runtime.py" --manifest "$v4s_manifest" --pin "$v4s_pin"
