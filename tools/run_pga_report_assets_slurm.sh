#!/usr/bin/env bash

# Slurm launcher for generating academic-report figures/tables.
# Usage:
#   bash tools/run_pga_report_assets_slurm.sh <results_root> [output_dir]
#   AUTO_SBATCH=0 bash tools/run_pga_report_assets_slurm.sh <results_root> [output_dir]
#
# Key env vars:
#   WORKDIR             team_pytorch repo root on the cluster
#   JOB_NAME            Slurm job name
#   SLURM_PARTITION     Slurm partition
#   SLURM_CPUS_PER_TASK CPUs for plotting/eval
#   SLURM_TIME          Wallclock limit
#   SLURM_LOG_DIR       Slurm log directory
#   CONDA_ENV           Conda env to activate
#   MODULE_UNLOAD       Optional module to unload
#   MODULE_LOADS        Space-separated modules to load
#   REPORT_RESULT_PATTERN  Glob pattern under results_root
#   REPORT_PRIMARY_MODEL   Model id or result-dir substring for detailed plots
#   REPORT_SELECTED_MODELS Comma-separated model ids for station-count plot
#   REPORT_EXTRA_ARGS      Extra args passed to generate_pga_report_assets.py

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_WORKDIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKDIR=${WORKDIR:-$DEFAULT_WORKDIR}

RESULTS_ROOT=${1:-"$WORKDIR/chaosuan_res"}
OUTPUT_DIR=${2:-"$WORKDIR/reports/pga_academic_report_assets"}

JOB_NAME=${JOB_NAME:-pga-report-assets}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-02:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}

if [[ -z "${SLURM_JOB_ID:-}" && "${AUTO_SBATCH:-1}" != "0" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    exec sbatch \
        --job-name="$JOB_NAME" \
        --partition="$SLURM_PARTITION" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --time="$SLURM_TIME" \
        --chdir="$WORKDIR" \
        --output="$SLURM_LOG_DIR/%x-%j.out" \
        --error="$SLURM_LOG_DIR/%x-%j.err" \
        --export=ALL,WORKDIR="$WORKDIR" \
        "$0" "$RESULTS_ROOT" "$OUTPUT_DIR"
fi

cd "$WORKDIR"

restore_nounset=0
if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
fi
if [[ -f /etc/profile ]]; then
    # shellcheck disable=SC1091
    source /etc/profile
fi
if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if [[ "$restore_nounset" -eq 1 ]]; then
    set -u
fi

if command -v module >/dev/null 2>&1 || declare -F module >/dev/null 2>&1; then
    if [[ -n "${MODULE_UNLOAD:-}" ]]; then
        module unload "$MODULE_UNLOAD" || true
    fi
    for mod in $MODULE_LOADS; do
        module load "$mod"
    done
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    restore_nounset=0
    if [[ $- == *u* ]]; then
        restore_nounset=1
        set +u
    fi
    export PS1=${PS1:-}
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        source activate "$CONDA_ENV"
    fi
    if [[ "$restore_nounset" -eq 1 ]]; then
        set -u
    fi
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}

python tools/generate_pga_report_assets.py \
    --results-root "$RESULTS_ROOT" \
    --pattern "${REPORT_RESULT_PATTERN:-weights_japan*_pga15*}" \
    --output-dir "$OUTPUT_DIR" \
    --primary "${REPORT_PRIMARY_MODEL:-}" \
    --selected-models "${REPORT_SELECTED_MODELS:-}" \
    ${REPORT_EXTRA_ARGS:-}

