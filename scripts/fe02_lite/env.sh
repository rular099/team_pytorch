#!/usr/bin/env bash
set -eo pipefail
set +u
# Preserve the FE02 fix for legacy Conda/module hooks under inherited nounset.
PS1="${PS1-}"
export CONDA_CHANGEPS1=false
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if [[ "${FE02_LITE_MODULE_LOADS:-none}" != none ]]; then
    type module >/dev/null 2>&1 || { echo 'module unavailable' >&2; exit 2; }
    module purge
    read -ra lite_modules <<< "$FE02_LITE_MODULE_LOADS"
    for lite_module in "${lite_modules[@]}"; do module load "$lite_module"; done
fi
if [[ -n "${FE02_LITE_CONDA_SH:-}" ]]; then source "$FE02_LITE_CONDA_SH";
elif type conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh";
else echo 'Set FE02_LITE_CONDA_SH to the existing working conda.sh' >&2; exit 2; fi
conda activate "${FE02_LITE_CONDA_ENV:-zb}"
set -u
export FE02_LITE_PYTHON="${CONDA_PREFIX:?}/bin/python"
export PYTHONPATH="$FE02_LITE_CODE_ROOT/vendor:$FE02_LITE_CODE_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$FE02_LITE_OUTPUT_ROOT/cache/matplotlib/${SLURM_JOB_ID:?}"
export XDG_CACHE_HOME="$FE02_LITE_OUTPUT_ROOT/cache/xdg/$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$FE02_LITE_CODE_ROOT"
"$FE02_LITE_PYTHON" scripts/fe02/verify_source.py --root "$FE02_LITE_CODE_ROOT"
"$FE02_LITE_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Working compute device required"; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name())'
