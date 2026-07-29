#!/usr/bin/env bash

# Slurm launcher for tools/check_dpk_prior_cache_coverage.py.
#
# Usage:
#   bash tools/check_dpk_prior_cache_coverage_slurm.sh <config.json> [split] [cache.h5|-] [extra check args...]
#
# Examples:
#   bash tools/check_dpk_prior_cache_coverage_slurm.sh pga_configs/rt40.json train
#   bash tools/check_dpk_prior_cache_coverage_slurm.sh pga_configs/rt40.json dev -
#   AUTO_SBATCH=0 bash tools/check_dpk_prior_cache_coverage_slurm.sh pga_configs/rt40.json train
#
# Common env overrides:
#   WORKDIR=/abs/path/to/team_pytorch
#   SLURM_PARTITION=diting
#   SLURM_CPUS_PER_TASK=8
#   SLURM_TIME=02:00:00
#   SLURM_GPUS=0              # set to 1 only if the partition requires a DCU/GPU allocation
#   MAX_SAMPLES=100           # optional quick smoke check
#   STOP_AFTER_MISSES=1       # optional fail-fast debugging
#   AUTO_SBATCH=0             # run inside an existing Slurm allocation

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash tools/check_dpk_prior_cache_coverage_slurm.sh <config.json> [split] [cache.h5|-] [extra args...]}
shift

SPLIT=${1:-${SPLIT:-train}}
if [[ $# -gt 0 ]]; then
    shift
fi

CACHE_INPUT=${1:-${CACHE:-auto}}
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-dpk-cover}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-02:00:00}
SLURM_GPUS=${SLURM_GPUS:-0}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
COVERAGE_OUTPUT_DIR=${COVERAGE_OUTPUT_DIR:-"$WORKDIR/logs/dpk_prior_cache_coverage"}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
SOURCE=${SOURCE:-dpk_finetuned}
MODE=${MODE:-event}
MAX_MISSES=${MAX_MISSES:-20}
MAX_SAMPLES=${MAX_SAMPLES:-}
STOP_AFTER_MISSES=${STOP_AFTER_MISSES:-0}

resolve_path() {
    local p=$1
    local base=${2:-$PWD}
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *) printf '%s\n' "$base/$p" ;;
    esac
}

SCRIPT_PATH=$(resolve_path "$0" "$SUBMIT_DIR")
CONFIG=$(resolve_path "$CONFIG_INPUT" "$SUBMIT_DIR")
if [[ "$CACHE_INPUT" == "auto" || "$CACHE_INPUT" == "-" || -z "$CACHE_INPUT" ]]; then
    CACHE=""
else
    CACHE=$(resolve_path "$CACHE_INPUT" "$SUBMIT_DIR")
fi

if [[ ! -f "$WORKDIR/tools/check_dpk_prior_cache_coverage.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/tools/check_dpk_prior_cache_coverage.py" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi
if [[ -n "$CACHE" && ! -f "$CACHE" ]]; then
    echo "Cache file not found: $CACHE" >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && "${AUTO_SBATCH:-1}" != "0" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[INFO] submitting DPK prior cache coverage check to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION split=$SPLIT gpus=$SLURM_GPUS"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "SLURM_LOG_DIR=$SLURM_LOG_DIR"
        "COVERAGE_OUTPUT_DIR=$COVERAGE_OUTPUT_DIR"
        "CONDA_ENV=$CONDA_ENV"
        "MODULE_UNLOAD=$MODULE_UNLOAD"
        "MODULE_LOADS=$MODULE_LOADS"
        "SOURCE=$SOURCE"
        "MODE=$MODE"
        "MAX_MISSES=$MAX_MISSES"
        "MAX_SAMPLES=$MAX_SAMPLES"
        "STOP_AFTER_MISSES=$STOP_AFTER_MISSES"
        "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
    )
    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$SLURM_PARTITION"
        --nodes=1
        --ntasks-per-node=1
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
        --chdir="$WORKDIR"
        --output="$SLURM_LOG_DIR/%x-%j.out"
        --error="$SLURM_LOG_DIR/%x-%j.err"
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")"
    )
    if [[ -n "$SLURM_TIME" ]]; then
        SBATCH_CMD+=(--time="$SLURM_TIME")
    fi
    if [[ "$SLURM_GPUS" != "0" ]]; then
        SBATCH_CMD+=(--gres="${SLURM_GRES_RESOURCE}:${SLURM_GPUS}")
    fi
    SBATCH_CMD+=("$SCRIPT_PATH" "$CONFIG" "$SPLIT" "${CACHE:-auto}")
    if ((EXTRA_ARG_COUNT > 0)); then
        SBATCH_CMD+=("${EXTRA_ARGS[@]}")
    fi
    exec "${SBATCH_CMD[@]}"
fi

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] config: $CONFIG"
echo "[INFO] split: $SPLIT"
echo "[INFO] cache: ${CACHE:-<infer from config>}"

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
        # shellcheck disable=SC1090
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        source activate "$CONDA_ENV"
    fi
    if [[ "$restore_nounset" -eq 1 ]]; then
        set -u
    fi
fi

mkdir -p "$COVERAGE_OUTPUT_DIR"
CONFIG_STEM=$(basename "$CONFIG")
CONFIG_STEM=${CONFIG_STEM%.json}
OUTPUT_TXT=${OUTPUT_TXT:-"$COVERAGE_OUTPUT_DIR/${CONFIG_STEM}_${SPLIT}_${SLURM_JOB_ID:-local}.txt"}

CHECK_ARGS=(
    --config "$CONFIG"
    --split "$SPLIT"
    --source "$SOURCE"
    --mode "$MODE"
    --max_misses "$MAX_MISSES"
    --stop_after_misses "$STOP_AFTER_MISSES"
)
if [[ -n "$CACHE" ]]; then
    CHECK_ARGS+=(--cache "$CACHE")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
    CHECK_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if ((EXTRA_ARG_COUNT > 0)); then
    CHECK_ARGS+=("${EXTRA_ARGS[@]}")
fi

CMD=(python tools/check_dpk_prior_cache_coverage.py "${CHECK_ARGS[@]}")
echo "[INFO] python: $(command -v python || echo '<missing>')"
echo "[INFO] output: $OUTPUT_TXT"
echo "[INFO] command: ${CMD[*]}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "${CMD[@]}" 2>&1 | tee "$OUTPUT_TXT"
else
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_TXT"
fi
