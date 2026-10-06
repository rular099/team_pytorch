#!/bin/bash
#SBATCH --job-name=fe01_envdiag_20261006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=fe01_envdiag_%j.out
#SBATCH --error=fe01_envdiag_%j.err
# Standalone diagnostic, outside the pinned evaluation source directories.
# No identity, HDF5, checkpoint, training or model-forward work is executed.
set -eo pipefail
echo 'FE01_DIAG_VERSION=20261006-1'
echo "FE01_DIAG_JOB=${SLURM_JOB_ID:-unset} HOST=${HOSTNAME:-unknown} SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "FE01_DIAG_BASH_ENV=${BASH_ENV:-unset}"
fe01_diag_root="${FE01_CODE_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the FE01 project root}}"
fe01_diag_env="${FE01_REVIEW_ENV:-$fe01_diag_root/scripts/fe01_review/cluster_zb.env}"
echo "FE01_DIAG_ENV_FILE=$fe01_diag_env"
if [ ! -f "$fe01_diag_env" ]; then
  echo 'FE01_DIAG_ERROR: private environment file missing' >&2; exit 2
fi
source "$fe01_diag_env"
fe01_diag_root="${FE01_CODE_ROOT:?}"
echo "FE01_DIAG_CODE_ROOT=$fe01_diag_root"
echo 'FE01_DIAG_EXPECTED_SOURCE=4c9c1757f726e571022a398cb5a217ca6c325cbc'
fe01_diag_mismatch=0
fe01_diag_check_file() {
  local fe01_diag_file="$fe01_diag_root/$1" fe01_diag_expected="$2" fe01_diag_actual=missing
  if [ -f "$fe01_diag_file" ]; then
    fe01_diag_actual="$(sha256sum "$fe01_diag_file")"
    fe01_diag_actual="${fe01_diag_actual%% *}"
  fi
  echo "FE01_DIAG_FILE=$fe01_diag_file"
  echo "FE01_DIAG_EXPECTED_SHA=$fe01_diag_expected ACTUAL_SHA=$fe01_diag_actual"
  if [ "$fe01_diag_actual" = "$fe01_diag_expected" ]; then
    echo 'FE01_DIAG_FILE_MATCH=YES'
  else
    echo 'FE01_DIAG_FILE_MATCH=NO'
    fe01_diag_mismatch=1
  fi
}
fe01_diag_check_file scripts/fe01_review/env.sh 889baabc4dc8836312ea4f011ed83ccc0ad6a9d543b5e2ea8c5eb9c5445ff7f0
fe01_diag_check_file scripts/fe01_review/identity_job.sbatch ec0d4a48130749516255f75111bb57512e0a2f154dc94b3ae7ff23c09860ce2c

if ! type module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if ! type module >/dev/null 2>&1; then
  echo 'FE01_DIAG_ERROR: module command unavailable' >&2; exit 2
fi
echo "FE01_DIAG_MODULES_INHERITED=${LOADEDMODULES:-none}"
if ! module purge; then
  echo 'FE01_DIAG_ERROR: module purge failed' >&2; exit 2
fi
echo "FE01_DIAG_MODULES_AFTER_PURGE=${LOADEDMODULES:-none}"
# Record and explicitly unload any ROCm family member retained after purge.
IFS=: read -ra fe01_diag_inherited_modules <<< "${LOADEDMODULES:-}"
for fe01_diag_module in "${fe01_diag_inherited_modules[@]}"; do
  case "$fe01_diag_module" in
    compiler/rocm/*)
      echo "FE01_DIAG_EXPLICIT_UNLOAD=$fe01_diag_module"
      if ! module unload "$fe01_diag_module"; then
        echo "FE01_DIAG_ERROR: cannot unload $fe01_diag_module" >&2; exit 2
      fi
      case ":${LOADEDMODULES:-}:" in
        *":$fe01_diag_module:"*) echo "FE01_DIAG_ERROR: $fe01_diag_module remains loaded" >&2; exit 2 ;;
      esac
      ;;
  esac
done
read -ra fe01_diag_modules <<< "${FE01_MODULE_LOADS:-compiler/rocm/dtk-23.04 apps/miniconda/3}"
for fe01_diag_module in "${fe01_diag_modules[@]}"; do
  if ! module load "$fe01_diag_module"; then
    echo "FE01_DIAG_ERROR: module load failed: $fe01_diag_module" >&2; exit 2
  fi
  case ":${LOADEDMODULES:-}:" in
    *":$fe01_diag_module:"*) ;;
    *) echo "FE01_DIAG_ERROR: requested module absent: $fe01_diag_module; loaded=${LOADEDMODULES:-none}" >&2; exit 2 ;;
  esac
done
echo "FE01_DIAG_MODULES_READY=${LOADEDMODULES:-none}"
if [ -n "${FE01_CONDA_SH:-}" ]; then
  source "$FE01_CONDA_SH"
elif type conda >/dev/null 2>&1; then
  fe01_diag_conda_base="$(conda info --base)"
  source "$fe01_diag_conda_base/etc/profile.d/conda.sh"
else
  echo 'FE01_DIAG_ERROR: conda unavailable after module loading' >&2; exit 2
fi
if ! conda activate "${FE01_CONDA_ENV:-zb}"; then
  echo 'FE01_DIAG_ERROR: conda activation failed' >&2; exit 2
fi
echo "FE01_DIAG_PYTHON=${CONDA_PREFIX:?}/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! "$CONDA_PREFIX/bin/python" -c 'import sys,json,torch; print("FE01_DIAG_TORCH=" + json.dumps(dict(python=sys.executable,torch_file=torch.__file__,torch_version=torch.__version__,rocm=torch.version.hip,accelerator_available=torch.cuda.is_available())))'; then
  echo 'FE01_DIAG_ERROR: torch import failed after module/conda initialization' >&2; exit 2
fi
echo 'FE01_DIAG_TORCH_IMPORT=PASS'
if [ "$fe01_diag_mismatch" = 1 ]; then
  echo 'FE01_DIAG_RESULT=SCRIPT_VERSION_MISMATCH: runtime probe passed; installed scripts differ from the envfix package'
  exit 3
fi
echo 'FE01_DIAG_RESULT=PASS: installed envfix scripts match and torch imports; no evaluation was run'
