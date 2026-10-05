#!/bin/bash
# Source only inside an allocation. Do not run model forwards on the login node.
set -eo pipefail
: "${FE01_REVIEW_ENV:?Set FE01_REVIEW_ENV to the absolute private cluster env file}"
source "$FE01_REVIEW_ENV"
if ! type module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
fi
if [ "${FE01_MODULE_LOADS:-none}" != none ]; then
  if ! type module >/dev/null 2>&1; then
    echo 'FE01_ENV_ERROR: module command unavailable' >&2; exit 2
  fi
  # Slurm --export=ALL inherits the login shell's ROCm/module environment.
  # Rebuild it before loading the version linked by the zb PyTorch build.
  if ! module purge; then
    echo 'FE01_ENV_ERROR: module purge failed' >&2; exit 2
  fi
  read -ra fe01_review_modules <<< "$FE01_MODULE_LOADS"
  for fe01_review_module in "${fe01_review_modules[@]}"; do
    if ! module load "$fe01_review_module"; then
      echo "FE01_ENV_ERROR: module load failed: $fe01_review_module" >&2; exit 2
    fi
    # Some Tcl module wrappers print ERROR but return zero. Check the actual
    # environment instead of treating that return code as proof of success.
    case ":${LOADEDMODULES:-}:" in
      *":$fe01_review_module:"*) ;;
      *) echo "FE01_ENV_ERROR: requested module is not loaded: $fe01_review_module; loaded=${LOADEDMODULES:-none}" >&2; exit 2 ;;
    esac
  done
  echo "FE01_MODULES_READY: ${LOADEDMODULES:-none}"
fi
if [ -n "${FE01_CONDA_SH:-}" ]; then
  source "$FE01_CONDA_SH"
elif type conda >/dev/null 2>&1; then
  fe01_review_conda_base="$(conda info --base)"
  source "$fe01_review_conda_base/etc/profile.d/conda.sh"
else
  echo 'Conda unavailable; fill FE01_CONDA_SH in the private env file' >&2; exit 2
fi
if ! conda activate "${FE01_CONDA_ENV:-zb}"; then
  echo 'FE01_ENV_ERROR: conda activation failed' >&2; exit 2
fi
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
# CPU checkpoint/metadata jobs also import this ROCm-linked PyTorch build.
# Check its shared libraries before creating any evaluation-stage outputs.
if ! "$FE01_REVIEW_PYTHON" -c 'import json,torch; print("FE01_RUNTIME_READY: " + json.dumps(dict(torch=torch.__version__,rocm=torch.version.hip,accelerator_available=torch.cuda.is_available())))'; then
  echo 'FE01_ENV_ERROR: PyTorch import failed after module/conda setup; evaluation has not started' >&2
  exit 2
fi
