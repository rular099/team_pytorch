#!/usr/bin/env bash

# Eval-only station-pool phase-window diagnostics for KNET rt46 and rt48.
#
# The two Slurm array tasks run in parallel.  Each task evaluates both train and
# validation using full_model_last.pth and writes per-station, per-event, and
# post-P-duration-binned CSV files.
#
# Submit:
#   bash tools/run_rt46_rt48_station_pool_phase_attention_slurm.sh
#
# Dry run:
#   DRY_RUN=1 bash tools/run_rt46_rt48_station_pool_phase_attention_slurm.sh
#
# Run sequentially inside an existing allocation:
#   AUTO_SBATCH=0 bash tools/run_rt46_rt48_station_pool_phase_attention_slurm.sh
#
# Useful overrides:
#   CHECKPOINT_TAG=last|best
#   SPLIT=both|train|val
#   MAX_SAMPLES=-1
#   POST_P_BIN_EDGES=0,1,3,5,10,20,40,90
#   POOL_NAMES=base_x,x
#   SLURM_TIME=1-00:00:00
#   ALLOW_OVERWRITE=1

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKDIR/logs/station_pool_phase_attention_rt46_rt48"}

JOB_NAME=${JOB_NAME:-team-phase-attn}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_GPUS=${SLURM_GPUS:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-1-00:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}

DITING_CONFIG=${DITING_CONFIG:-"$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml"}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}

CHECKPOINT_TAG=${CHECKPOINT_TAG:-last}
SPLIT=${SPLIT:-both}
MAX_SAMPLES=${MAX_SAMPLES:--1}
START_INDEX=${START_INDEX:-0}
STRIDE=${STRIDE:-1}
# Pass comma-containing values as script arguments when submitting through
# sbatch.  Slurm uses commas as separators in --export and would otherwise
# split these values into invalid environment entries.
POOL_NAMES=${1:-${POOL_NAMES:-base_x,x}}
POST_P_BIN_EDGES=${2:-${POST_P_BIN_EDGES:-0,1,3,5,10,20,40,90}}
INPUT_STATION_SELECTION=${INPUT_STATION_SELECTION:-config}
LOG_EVERY=${LOG_EVERY:-50}
DEVICE=${DEVICE:-cuda:0}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
AUTO_SBATCH=${AUTO_SBATCH:-1}
DRY_RUN=${DRY_RUN:-0}

resolve_path() {
    local path_value=$1
    local base=${2:-$PWD}
    case "$path_value" in
        /*) printf '%s\n' "$path_value" ;;
        *) printf '%s\n' "$base/$path_value" ;;
    esac
}

SCRIPT_PATH=$(resolve_path "$0" "$SUBMIT_DIR")

if [[ ! -f "$WORKDIR/tools/diagnose_station_pool_phase_attention.py" ]]; then
    echo "Diagnostic script not found under WORKDIR: $WORKDIR" >&2
    exit 1
fi
if [[ ! -f "$DITING_CONFIG" ]]; then
    echo "DiTing config not found: $DITING_CONFIG" >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$AUTO_SBATCH" == "1" ]]; then
        mkdir -p "$SLURM_LOG_DIR"
        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --partition="$SLURM_PARTITION"
            --nodes=1
            --ntasks-per-node=1
            --cpus-per-task="$SLURM_CPUS_PER_TASK"
            --gres="${SLURM_GRES_RESOURCE}:${SLURM_GPUS}"
            --time="$SLURM_TIME"
            --array=0-1
            --chdir="$WORKDIR"
            --output="$SLURM_LOG_DIR/%x-%A_%a.out"
            --error="$SLURM_LOG_DIR/%x-%A_%a.err"
            --export="ALL,WORKDIR=$WORKDIR,CONFIG_DIR=$CONFIG_DIR,OUTPUT_ROOT=$OUTPUT_ROOT,DITING_CONFIG=$DITING_CONFIG,DITING_PRETRAINED=$DITING_PRETRAINED,CHECKPOINT_TAG=$CHECKPOINT_TAG,SPLIT=$SPLIT,MAX_SAMPLES=$MAX_SAMPLES,START_INDEX=$START_INDEX,STRIDE=$STRIDE,INPUT_STATION_SELECTION=$INPUT_STATION_SELECTION,LOG_EVERY=$LOG_EVERY,DEVICE=$DEVICE,ALLOW_OVERWRITE=$ALLOW_OVERWRITE,AUTO_SBATCH=0"
            "$SCRIPT_PATH"
            "$POOL_NAMES"
            "$POST_P_BIN_EDGES"
        )
        echo "[INFO] submitting rt46/rt48 phase-attention array"
        echo "[INFO] command: ${SBATCH_CMD[*]}"
        if [[ "$DRY_RUN" == "1" ]]; then
            exit 0
        fi
        "${SBATCH_CMD[@]}"
        echo "[INFO] after both tasks finish, run:"
        echo "python tools/compare_station_pool_phase_attention.py \\"
        echo "  --rt46_dir '$OUTPUT_ROOT/rt46_${CHECKPOINT_TAG}' \\"
        echo "  --rt48_dir '$OUTPUT_ROOT/rt48_${CHECKPOINT_TAG}' \\"
        echo "  --output_dir '$OUTPUT_ROOT/paired_rt48_minus_rt46_${CHECKPOINT_TAG}'"
        exit 0
    fi

    echo "[INFO] AUTO_SBATCH=0: running rt46 and rt48 sequentially"
    for task_id in 0 1; do
        SLURM_ARRAY_TASK_ID=$task_id AUTO_SBATCH=0 \
            bash "$SCRIPT_PATH" "$POOL_NAMES" "$POST_P_BIN_EDGES"
    done
    python "$WORKDIR/tools/compare_station_pool_phase_attention.py" \
        --rt46_dir "$OUTPUT_ROOT/rt46_${CHECKPOINT_TAG}" \
        --rt48_dir "$OUTPUT_ROOT/rt48_${CHECKPOINT_TAG}" \
        --output_dir "$OUTPUT_ROOT/paired_rt48_minus_rt46_${CHECKPOINT_TAG}"
    exit 0
fi

case "$SLURM_ARRAY_TASK_ID" in
    0)
        RT_ID=46
        CONFIG_NAME=transformer_japan_overfit_pga15_stage2_512_rt46_knet_cached_dpk_event_temporal_residual_scale4_chaosuan.json
        ;;
    1)
        RT_ID=48
        CONFIG_NAME=transformer_japan_overfit_pga15_stage2_512_rt48_knet_cached_dpk_event_station_pool_temporal_residual_scale4_chaosuan.json
        ;;
    *)
        echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID; expected 0 or 1." >&2
        exit 1
        ;;
esac

CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

WEIGHT_PATH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$CONFIG_PATH")
if [[ -z "$WEIGHT_PATH" || "$WEIGHT_PATH" == "/" || "$WEIGHT_PATH" == "." || "$WEIGHT_PATH" == ".." ]]; then
    echo "Unsafe weight_path in $CONFIG_PATH: '$WEIGHT_PATH'" >&2
    exit 1
fi
case "$WEIGHT_PATH" in
    /*) WEIGHT_DIR="$WEIGHT_PATH" ;;
    *) WEIGHT_DIR="$WORKDIR/${WEIGHT_PATH#./}" ;;
esac

CHECKPOINT="$WEIGHT_DIR/full_model_${CHECKPOINT_TAG}.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ -f "$WEIGHT_DIR/config.json" ]]; then
    EVAL_CONFIG="$WEIGHT_DIR/config.json"
else
    EVAL_CONFIG="$CONFIG_PATH"
fi

OUTPUT_DIR="$OUTPUT_ROOT/rt${RT_ID}_${CHECKPOINT_TAG}"
SUMMARY_JSON="$OUTPUT_DIR/summary.json"
if [[ -s "$SUMMARY_JSON" && "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "[INFO] existing completed output; skipping rt${RT_ID}: $SUMMARY_JSON"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_TXT="$OUTPUT_DIR/run_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}.txt"

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
    for module_name in $MODULE_LOADS; do
        module load "$module_name"
    done
else
    echo "[WARN] module command unavailable; skipping module load." >&2
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

cd "$WORKDIR"

CMD=(
    python tools/diagnose_station_pool_phase_attention.py
    --config "$EVAL_CONFIG"
    --checkpoint "$CHECKPOINT"
    --model_name "rt${RT_ID}_knet"
    --diting_config "$DITING_CONFIG"
    --diting_pretrained "$DITING_PRETRAINED"
    --output_dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --pool_names "$POOL_NAMES"
    --post_p_bin_edges "$POST_P_BIN_EDGES"
    --input_station_selection "$INPUT_STATION_SELECTION"
    --device "$DEVICE"
    --max_samples "$MAX_SAMPLES"
    --start_index "$START_INDEX"
    --stride "$STRIDE"
    --log_every "$LOG_EVERY"
)

echo "[INFO] rt=$RT_ID"
echo "[INFO] config=$EVAL_CONFIG"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] split=$SPLIT"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] command: ${CMD[*]}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        "${CMD[@]}" 2>&1 | tee "$OUTPUT_TXT"
else
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_TXT"
fi

echo "[INFO] rt${RT_ID} phase-attention eval finished: $OUTPUT_DIR"
