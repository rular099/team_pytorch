#!/usr/bin/env bash
set -euo pipefail
: "${FE01_ENV_FILE:?Set FE01_ENV_FILE to the private cluster environment file}"
source "$FE01_ENV_FILE"
for key in FE01_CODE_ROOT FE01_OUTPUT_ROOT DATA_ROOT FE01_SPLIT_MANIFEST FE01_WEIGHTS_ROOT FE01_PYTHON FE01_MODULE_LOADS; do
    value=${!key:-}
    if [[ -z "$value" || "$value" == *'__FILL__'* ]]; then
        printf 'Missing cluster field: %s\n' "$key" >&2
        exit 2
    fi
done
[[ "${RESET_WEIGHT_PATH:-0}" == 0 && "${AUTO_SBATCH:-0}" == 0 ]] || { echo 'Deletion/automatic submission is forbidden' >&2; exit 2; }
[[ "$FE01_CODE_ROOT" = /* && "$FE01_OUTPUT_ROOT" = /* ]] || { echo 'Code/output roots must be absolute' >&2; exit 2; }
cd "$FE01_CODE_ROOT"
export MPLCONFIGDIR="$FE01_OUTPUT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$FE01_OUTPUT_ROOT/.cache/xdg"
export SEISBENCH_CACHE_ROOT="$FE01_OUTPUT_ROOT/.cache/seisbench"
export OMP_NUM_THREADS="${FE01_CPUS_PER_TASK:-1}"
if [[ "${FE01_MODULE_LOADS}" != none ]]; then
    type module >/dev/null 2>&1 || { echo 'Cluster module command unavailable' >&2; exit 2; }
    if [[ -n "${FE01_MODULE_UNLOAD:-}" ]]; then module unload "$FE01_MODULE_UNLOAD"; fi
    read -r -a fe01_modules <<< "$FE01_MODULE_LOADS"
    module load "${fe01_modules[@]}"
fi
[[ -x "$FE01_PYTHON" ]] || { echo 'Dedicated Python missing' >&2; exit 2; }
