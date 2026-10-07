#!/usr/bin/env bash
# Source inside an allocated COMPUTE NODE. Purge conflicts; keep known DTK+MPI.
set +u
set -eo pipefail
: "${SLURM_JOB_ID:?Use the explicit submit.sh entry on the login node}"
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${A01_ENV_FILE:-$code/a01.private.env}"
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
read -r -a modules <<< "$A01_MODULES"
for entry in "${modules[@]}"; do
  module load "$entry"
  [[ ":${LOADEDMODULES:-}:" == *":$entry:"* ]] || { echo "Module load failed despite exit status: $entry" >&2; exit 2; }
done
conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
set +u
conda activate "$A01_CONDA_ENV"
set -u
export A01_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONPATH="$A01_CODE_ROOT/vendor:$A01_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$A01_OUTPUT_ROOT/.cache/$SLURM_JOB_ID/matplotlib"
export XDG_CACHE_HOME="$A01_OUTPUT_ROOT/.cache/$SLURM_JOB_ID/xdg"
export SEISBENCH_CACHE_ROOT="$A01_OUTPUT_ROOT/.cache/$SLURM_JOB_ID/seisbench"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$SEISBENCH_CACHE_ROOT"
cd "$A01_CODE_ROOT"
"$A01_PYTHON" -c 'import torch; print("A01_ENV_READY", torch.__version__, torch.version.hip, torch.cuda.is_available(), flush=True); assert torch.cuda.is_available(), "DCU unavailable in allocation"'
