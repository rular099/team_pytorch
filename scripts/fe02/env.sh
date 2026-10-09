#!/usr/bin/env bash
# Only called inside an allocation. Module/conda changes never touch login shell.
set -eo pipefail
# This file is sourced by job.sh, which already enabled nounset. Merely omitting
# -u above does not disable the inherited option. Legacy module/Conda hooks read
# optional shell variables, so suspend nounset only during bootstrap.
set +u
PS1="${PS1-}"
export CONDA_CHANGEPS1=false
: "${SLURM_JOB_ID:?Submit manually with sbatch}" "${FE02_ENV_FILE:?}"
source "$FE02_ENV_FILE"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if [[ "${FE02_MODULE_LOADS:-none}" != none ]]; then
    type module >/dev/null 2>&1 || { echo 'module unavailable' >&2; exit 2; }
    module purge
    read -ra fe02_modules <<< "$FE02_MODULE_LOADS"
    for fe02_module in "${fe02_modules[@]}"; do
        module load "$fe02_module"
        case ":${LOADEDMODULES:-}:" in
            *":$fe02_module:"*) ;;
            *) echo "Requested module not actually loaded: $fe02_module" >&2; exit 2;;
        esac
    done
fi
if [[ -n "${FE02_CONDA_SH:-}" ]]; then source "$FE02_CONDA_SH";
elif type conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh";
else echo 'Set FE02_CONDA_SH to the working conda.sh' >&2; exit 2; fi
conda activate "${FE02_CONDA_ENV:-zb}"
set -u
export FE02_PYTHON="${CONDA_PREFIX:?}/bin/python"
export PYTHONPATH="$FE02_CODE_ROOT/vendor:$FE02_CODE_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$FE02_OUTPUT_ROOT/cache/matplotlib/$SLURM_JOB_ID"
export XDG_CACHE_HOME="$FE02_OUTPUT_ROOT/cache/xdg/$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$FE02_CODE_ROOT"
"$FE02_PYTHON" scripts/fe02/verify_source.py --root "$FE02_CODE_ROOT"
"$FE02_PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.version.hip, "FE02 requires working ROCm/DCU torch"; print("FE02 runtime:", torch.__version__, torch.version.hip, torch.cuda.get_device_name())'
