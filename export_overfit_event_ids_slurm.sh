#!/usr/bin/env bash

# Slurm/HPC launcher for export_overfit_event_ids.py.
# Usage:
#   bash export_overfit_event_ids_slurm.sh [config.json] [output.txt]
#   AUTO_SBATCH=0 bash export_overfit_event_ids_slurm.sh [config.json] [output.txt]

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:-pga_configs/transformer_japan_overfit_pga15_stage2_512_b0_baseline_noamp_chaosuan.json}
OUTPUT_INPUT=${2:-}

JOB_NAME=${JOB_NAME:-team-export-events}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_NODES=${SLURM_NODES:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}
SLURM_TIME=${SLURM_TIME:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}

resolve_path() {
    local p=$1
    local base=${2:-$PWD}
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *) printf '%s\n' "$base/$p" ;;
    esac
}

CONFIG=$(resolve_path "$CONFIG_INPUT" "$SUBMIT_DIR")
if [[ -n "$OUTPUT_INPUT" ]]; then
    OUTPUT=$(resolve_path "$OUTPUT_INPUT" "$SUBMIT_DIR")
else
    OUTPUT=""
fi

if [[ ! -f "$WORKDIR/export_overfit_event_ids.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/export_overfit_event_ids.py" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && "${AUTO_SBATCH:-1}" != "0" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[INFO] submitting event-id export to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION nodes=$SLURM_NODES"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "CONFIG_INPUT=$CONFIG"
        "OUTPUT_INPUT=$OUTPUT"
        "SLURM_LOG_DIR=$SLURM_LOG_DIR"
    )
    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$SLURM_PARTITION"
        --nodes="$SLURM_NODES"
        --ntasks-per-node=1
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
    )
    if [[ -n "$SLURM_TIME" ]]; then
        SBATCH_CMD+=(--time="$SLURM_TIME")
    fi
    SBATCH_CMD+=(
        --chdir="$WORKDIR"
        --output="$SLURM_LOG_DIR/%x-%j.out"
        --error="$SLURM_LOG_DIR/%x-%j.err"
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")"
        "$0" "$CONFIG" "$OUTPUT"
    )
    exec "${SBATCH_CMD[@]}"
fi

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] config: $CONFIG_INPUT"
echo "[INFO] output: ${OUTPUT_INPUT:-<from config>}"

export COLORTERM=${COLORTERM:-truecolor}

restore_nounset=0
if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
fi
if [[ -f /etc/profile ]]; then
    source /etc/profile
fi
if [[ -f /etc/profile.d/modules.sh ]]; then
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
else
    echo "[WARN] module command is unavailable; skipping module load." >&2
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

CMD=(python export_overfit_event_ids.py --config "$CONFIG")
if [[ -n "$OUTPUT_INPUT" ]]; then
    CMD+=(--output "$OUTPUT_INPUT")
fi

echo "[INFO] python: $(command -v python || echo '<missing>')"
echo "[INFO] command: ${CMD[*]}"
"${CMD[@]}"

