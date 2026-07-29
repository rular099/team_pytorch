#!/usr/bin/env bash

# Single-node Slurm launcher for oracle-nonzero masked DiTing feature similarity.
#
# Usage:
#   bash tools/diagnose_nonzero_masked_diting_similarity_slurm.sh <config.json> [output_dir] [extra args...]
#
# Examples:
#   bash tools/diagnose_nonzero_masked_diting_similarity_slurm.sh \
#     pga_configs/transformer_japan_overfit_pga15_stage2_512_rt31_oracle_nonzero_station_pool_chaosuan.json
#
#   SPLIT=train MAX_SAMPLES=-1 STATION_BATCH_SIZE=1 \
#     bash tools/diagnose_nonzero_masked_diting_similarity_slurm.sh pga_configs/xxx.json
#
#   # Exact realtime snapshot: waveform cut at first P pick + 3 seconds.
#   SPLIT=val ELAPSED_TIMES=3 MAX_SAMPLES=-1 \
#     bash tools/diagnose_nonzero_masked_diting_similarity_slurm.sh pga_configs/xxx.json
#
# Common env overrides:
#   WORKDIR=/abs/path/to/team_pytorch
#   SPLIT=train|val|both
#   MAX_SAMPLES=-1                                # default: full selected split
#   MAX_MATCHING_SAMPLES=-1
#   ELAPSED_TIMES=3                                # comma list, e.g. 1,3,5,10
#   ELAPSED_TOLERANCE=1e-4
#   ELAPSED_BIN_EDGES=0,1,3,5,10,20,40,90          # default: infer from config
#   STATION_BATCH_SIZE=1
#   DITING_CONFIG=/abs/path/to/diting_1200m_backbone_attnpool.yml
#   DITING_PRETRAINED=/abs/path/to/MAE/mp_rank_00_model_states.pt
#   CHECKPOINT=/abs/path/to/full_model_last.pth   # optional; normally not needed
#   AUTO_SBATCH=0                                 # run inside current allocation

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash tools/diagnose_nonzero_masked_diting_similarity_slurm.sh <config.json> [output_dir] [extra args...]}
shift

CONFIG_BASENAME=$(basename "$CONFIG_INPUT")
CONFIG_STEM=${CONFIG_BASENAME%.json}
SPLIT=${SPLIT:-train}
OUTPUT_INPUT=${1:-"$WORKDIR/logs/nonzero_masked_diting_similarity/${CONFIG_STEM}_${SPLIT}"}
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-nz-diting}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GPUS=${SLURM_GPUS:-1}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-04:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}

DITING_CONFIG=${DITING_CONFIG:-$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}

MAX_SAMPLES=${MAX_SAMPLES:--1}
MAX_MATCHING_SAMPLES=${MAX_MATCHING_SAMPLES:--1}
START_INDEX=${START_INDEX:-0}
STRIDE=${STRIDE:-1}
STATION_BATCH_SIZE=${STATION_BATCH_SIZE:-1}
MIN_STATIONS=${MIN_STATIONS:-2}
MIN_PAIR_TOKENS=${MIN_PAIR_TOKENS:-1}
NONZERO_EPS=${NONZERO_EPS:-1e-8}
ELAPSED_TIMES=${ELAPSED_TIMES:-}
ELAPSED_TOLERANCE=${ELAPSED_TOLERANCE:-1e-4}
ELAPSED_BIN_EDGES=${ELAPSED_BIN_EDGES:-}
LOG_EVERY=${LOG_EVERY:-100}
INPUT_STATION_SELECTION=${INPUT_STATION_SELECTION:-config}
DEVICE=${DEVICE:-cuda:0}

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
OUTPUT_DIR=$(resolve_path "$OUTPUT_INPUT" "$SUBMIT_DIR")
if [[ -n "${CHECKPOINT:-}" ]]; then
    CHECKPOINT=$(resolve_path "$CHECKPOINT" "$SUBMIT_DIR")
fi

if [[ ! -f "$WORKDIR/tools/diagnose_nonzero_masked_diting_similarity.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/tools/diagnose_nonzero_masked_diting_similarity.py" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$DITING_CONFIG" ]]; then
    echo "DITING config file not found: $DITING_CONFIG" >&2
    exit 1
fi
if [[ -n "${CHECKPOINT:-}" && ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint file not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && "${AUTO_SBATCH:-1}" != "0" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[INFO] submitting nonzero-masked DiTing similarity diagnostic to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION split=$SPLIT gpus=$SLURM_GPUS max_samples=$MAX_SAMPLES"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "SPLIT=$SPLIT"
        "DITING_CONFIG=$DITING_CONFIG"
        "DITING_PRETRAINED=$DITING_PRETRAINED"
        "CONDA_ENV=$CONDA_ENV"
        "MODULE_UNLOAD=$MODULE_UNLOAD"
        "MODULE_LOADS=$MODULE_LOADS"
        "MAX_SAMPLES=$MAX_SAMPLES"
        "MAX_MATCHING_SAMPLES=$MAX_MATCHING_SAMPLES"
        "START_INDEX=$START_INDEX"
        "STRIDE=$STRIDE"
        "STATION_BATCH_SIZE=$STATION_BATCH_SIZE"
        "MIN_STATIONS=$MIN_STATIONS"
        "MIN_PAIR_TOKENS=$MIN_PAIR_TOKENS"
        "NONZERO_EPS=$NONZERO_EPS"
        "ELAPSED_TIMES=$ELAPSED_TIMES"
        "ELAPSED_TOLERANCE=$ELAPSED_TOLERANCE"
        "ELAPSED_BIN_EDGES=$ELAPSED_BIN_EDGES"
        "LOG_EVERY=$LOG_EVERY"
        "INPUT_STATION_SELECTION=$INPUT_STATION_SELECTION"
        "DEVICE=$DEVICE"
    )
    if [[ -n "${CHECKPOINT:-}" ]]; then
        EXPORT_VARS+=("CHECKPOINT=$CHECKPOINT")
    fi
    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$SLURM_PARTITION"
        --nodes=1
        --ntasks-per-node=1
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
        --gres="${SLURM_GRES_RESOURCE}:${SLURM_GPUS}"
        --time="$SLURM_TIME"
        --chdir="$WORKDIR"
        --output="$SLURM_LOG_DIR/%x-%j.out"
        --error="$SLURM_LOG_DIR/%x-%j.err"
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")"
        "$SCRIPT_PATH"
        "$CONFIG"
        "$OUTPUT_DIR"
    )
    if ((EXTRA_ARG_COUNT > 0)); then
        SBATCH_CMD+=("${EXTRA_ARGS[@]}")
    fi
    exec "${SBATCH_CMD[@]}"
fi

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] config: $CONFIG"
echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] split: $SPLIT"
echo "[INFO] elapsed_times: ${ELAPSED_TIMES:-<all>}"
echo "[INFO] elapsed_bin_edges: ${ELAPSED_BIN_EDGES:-<infer from config>}"
echo "[INFO] diting_config: $DITING_CONFIG"
echo "[INFO] diting_pretrained: $DITING_PRETRAINED"
echo "[INFO] checkpoint: ${CHECKPOINT:-<none>}"

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

mkdir -p "$OUTPUT_DIR"
OUTPUT_TXT=${OUTPUT_TXT:-"$OUTPUT_DIR/run_${SLURM_JOB_ID:-local}.txt"}

ARGS=(
    --config "$CONFIG"
    --diting_config "$DITING_CONFIG"
    --diting_pretrained "$DITING_PRETRAINED"
    --output_dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --device "$DEVICE"
    --max_samples "$MAX_SAMPLES"
    --max_matching_samples "$MAX_MATCHING_SAMPLES"
    --start_index "$START_INDEX"
    --stride "$STRIDE"
    --station_batch_size "$STATION_BATCH_SIZE"
    --min_stations "$MIN_STATIONS"
    --min_pair_tokens "$MIN_PAIR_TOKENS"
    --nonzero_eps "$NONZERO_EPS"
    --input_station_selection "$INPUT_STATION_SELECTION"
    --log_every "$LOG_EVERY"
)
if [[ -n "${ELAPSED_TIMES:-}" ]]; then
    ARGS+=(--elapsed_times "$ELAPSED_TIMES" --elapsed_tolerance "$ELAPSED_TOLERANCE")
fi
if [[ -n "${ELAPSED_BIN_EDGES:-}" ]]; then
    ARGS+=(--elapsed_bin_edges "$ELAPSED_BIN_EDGES")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi
if ((EXTRA_ARG_COUNT > 0)); then
    ARGS+=("${EXTRA_ARGS[@]}")
fi

CMD=(python tools/diagnose_nonzero_masked_diting_similarity.py "${ARGS[@]}")
echo "[INFO] python: $(command -v python || echo '<missing>')"
echo "[INFO] output txt: $OUTPUT_TXT"
echo "[INFO] command: ${CMD[*]}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "${CMD[@]}" 2>&1 | tee "$OUTPUT_TXT"
else
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_TXT"
fi

echo "[INFO] diagnostic finished: $OUTPUT_DIR"
