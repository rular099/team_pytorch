#!/usr/bin/env bash
# COMPUTE ONLY. This file is sourced inside the allocated srun step for DCU.
set +u
: "${SLURM_JOB_ID:?Allocated compute node required}"
set -eo pipefail
source "${ON_EVAL_ENV_FILE:?}"
release_root=${ON_EVAL_RELEASE_ROOT:?}
if [[ -n "${ON_EVAL_ENV_HOOK:-}" ]]; then
  # Optional user-supplied known-working hook; no arbitrary environment replacement.
  source "$ON_EVAL_ENV_HOOK"
else
  type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
  # Remove only a conflicting ROCm module before the exact working A01 module list.
  read -r -a modules <<< "${A01_MODULES:?Reuse the already working A01 DTK/MPI list}"
  for requested in "${modules[@]}"; do
    if [[ "$requested" == compiler/rocm/* ]]; then
      IFS=: read -r -a loaded <<< "${LOADEDMODULES:-}"
      for entry in "${loaded[@]}"; do
        if [[ "$entry" == compiler/rocm/* && "$entry" != "$requested" ]]; then module unload "$entry"; fi
      done
    fi
    if [[ ":${LOADEDMODULES:-}:" != *":$requested:"* ]]; then module load "$requested"; fi
    [[ ":${LOADEDMODULES:-}:" == *":$requested:"* ]] || { echo "Module load failed: $requested" >&2;return 2; }
  done
  conda_base=${ON_EVAL_CONDA_BASE:-$(conda info --base)}
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "${ON_EVAL_CONDA_ENV:-zb}"
fi
set -u
export ON_EVAL_PYTHON="${CONDA_PREFIX:?}/bin/python"
export PYTHONPATH="$release_root:${ON_EVAL_INHERITED_CODE_ROOT:?}/vendor${PYTHONPATH:+:$PYTHONPATH}"
export DTBENCH_ROOT="$ON_EVAL_INHERITED_CODE_ROOT/vendor"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cache_root="${ON_EVAL_OUTPUT_ROOT:?}/.cache/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID:-0}"
export MPLCONFIGDIR="$cache_root/matplotlib" XDG_CACHE_HOME="$cache_root/xdg" SEISBENCH_CACHE_ROOT="$cache_root/seisbench"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$SEISBENCH_CACHE_ROOT"
