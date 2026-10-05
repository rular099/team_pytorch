#!/bin/bash
# Source only inside an allocation. Do not run model forwards on the login node.
set -eo pipefail
: "${FE01_REVIEW_ENV:?Set FE01_REVIEW_ENV to the absolute private cluster env file}"
source "$FE01_REVIEW_ENV"
if ! type module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
fi
if [ "${FE01_MODULE_LOADS:-none}" != none ]; then
  read -ra fe01_review_modules <<< "$FE01_MODULE_LOADS"
  module load "${fe01_review_modules[@]}"
fi
if [ -n "${FE01_CONDA_SH:-}" ]; then
  source "$FE01_CONDA_SH"
elif type conda >/dev/null 2>&1; then
  fe01_review_conda_base="$(conda info --base)"
  source "$fe01_review_conda_base/etc/profile.d/conda.sh"
else
  echo 'Conda unavailable; fill FE01_CONDA_SH in the private env file' >&2; exit 2
fi
conda activate "${FE01_CONDA_ENV:-zb}"
set -u
export FE01_REVIEW_PYTHON="${CONDA_PREFIX:?}/bin/python"
export FE01_REVIEW_RUN_ROOT="${FE01_REVIEW_OUTPUT_ROOT:?}/${REVIEW_RUN_ID:?}"
export PYTHONPATH="${FE01_CODE_ROOT:?}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$FE01_REVIEW_RUN_ROOT/cache/matplotlib/${SLURM_JOB_ID:?}"
export XDG_CACHE_HOME="$FE01_REVIEW_RUN_ROOT/cache/xdg/$SLURM_JOB_ID"
export SEISBENCH_CACHE_ROOT="$FE01_REVIEW_RUN_ROOT/cache/seisbench/$SLURM_JOB_ID"
export TMPDIR="$FE01_REVIEW_RUN_ROOT/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$SEISBENCH_CACHE_ROOT" "$TMPDIR"
cd "$FE01_CODE_ROOT"
