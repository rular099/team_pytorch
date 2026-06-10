#!/usr/bin/env bash

# Single-node Slurm launcher for tools/precompute_dpk_priors.py.
#
# Usage:
#   bash tools/precompute_dpk_priors_slurm.sh <config.json> [output_dir] [extra precompute args...]
#
# Common env overrides:
#   SPLIT=train|dev|test|all
#   STATION_BATCH_SIZE=4
#   DITING_CONFIG=/abs/path/to/diting_1200m_backbone_attnpool.yml
#   DITING_PRETRAINED=/abs/path/to/MAE/mp_rank_00_model_states.pt
#   DPK_CHECKPOINT=/abs/path/to/model-4-latest.pth
#   AUTO_SBATCH=0       # run directly in current allocation/session

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash tools/precompute_dpk_priors_slurm.sh <config.json> [output_dir] [extra args...]}
shift

SPLIT=${SPLIT:-train}
CONFIG_BASENAME=$(basename "$CONFIG_INPUT")
CONFIG_STEM=${CONFIG_BASENAME%.json}
OUTPUT_INPUT=${1:-"$WORKDIR/dpk_prior_cache/${CONFIG_STEM}_${SPLIT}"}
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-dpk-prior}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}

DITING_CONFIG=${DITING_CONFIG:-$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}
DPK_CHECKPOINT=${DPK_CHECKPOINT:-/public/home/test_bigmodel/seismogram/mx/ckpt/1200m_dpk/mae_init/720w_sft/model-4-latest.pth}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
STATION_BATCH_SIZE=${STATION_BATCH_SIZE:-4}
TOKEN_FLOOR=${TOKEN_FLOOR:-0.0001}
DPK_WEIGHT_TEMPERATURE=${DPK_WEIGHT_TEMPERATURE:-1.0}
DPK_WEIGHT_RESAMPLE=${DPK_WEIGHT_RESAMPLE:-max}
SAVE_DTYPE=${SAVE_DTYPE:-float16}
HDF5_NAME=${HDF5_NAME:-dpk_priors.h5}
HDF5_PRIOR_DTYPE=${HDF5_PRIOR_DTYPE:-float32}
HDF5_COMPRESSION=${HDF5_COMPRESSION:-none}
HDF5_GZIP_LEVEL=${HDF5_GZIP_LEVEL:-4}
HDF5_CHUNK_ROWS=${HDF5_CHUNK_ROWS:-1024}

resolve_path() {
    local p=$1
    local base=${2:-$PWD}
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *) printf '%s\n' "$base/$p" ;;
    esac
}

CONFIG=$(resolve_path "$CONFIG_INPUT" "$SUBMIT_DIR")
OUTPUT_DIR=$(resolve_path "$OUTPUT_INPUT" "$SUBMIT_DIR")

if [[ -z "${SLURM_JOB_ID:-}" && "${AUTO_SBATCH:-1}" != "0" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[INFO] submitting DPK prior precompute to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION nodes=1 cards=$SLURM_GPUS_PER_NODE split=$SPLIT"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "SPLIT=$SPLIT"
        "DITING_CONFIG=$DITING_CONFIG"
        "DITING_PRETRAINED=$DITING_PRETRAINED"
        "DPK_CHECKPOINT=$DPK_CHECKPOINT"
        "STATION_BATCH_SIZE=$STATION_BATCH_SIZE"
        "TOKEN_FLOOR=$TOKEN_FLOOR"
        "DPK_WEIGHT_TEMPERATURE=$DPK_WEIGHT_TEMPERATURE"
        "DPK_WEIGHT_RESAMPLE=$DPK_WEIGHT_RESAMPLE"
        "SAVE_DTYPE=$SAVE_DTYPE"
        "HDF5_NAME=$HDF5_NAME"
        "HDF5_PRIOR_DTYPE=$HDF5_PRIOR_DTYPE"
        "HDF5_COMPRESSION=$HDF5_COMPRESSION"
        "HDF5_GZIP_LEVEL=$HDF5_GZIP_LEVEL"
        "HDF5_CHUNK_ROWS=$HDF5_CHUNK_ROWS"
    )
    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$SLURM_PARTITION"
        --nodes=1
        --ntasks-per-node="$SLURM_GPUS_PER_NODE"
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
        --gres="${SLURM_GRES_RESOURCE}:${SLURM_GPUS_PER_NODE}"
        --chdir="$WORKDIR"
        --output="$SLURM_LOG_DIR/%x-%j.out"
        --error="$SLURM_LOG_DIR/%x-%j.err"
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")"
    )
    if [[ -n "$SLURM_TIME" ]]; then
        SBATCH_CMD+=(--time="$SLURM_TIME")
    fi
    SBATCH_CMD+=("$0" "$CONFIG" "$OUTPUT_DIR")
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
echo "[INFO] diting_config: $DITING_CONFIG"
echo "[INFO] diting_pretrained: $DITING_PRETRAINED"
echo "[INFO] dpk_checkpoint: $DPK_CHECKPOINT"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$DITING_CONFIG" ]]; then
    echo "DITING config file not found: $DITING_CONFIG" >&2
    exit 1
fi

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
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        source activate "$CONDA_ENV"
    fi
    if [[ "$restore_nounset" -eq 1 ]]; then
        set -u
    fi
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
    MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n "${SLURM_JOBID:-$$}" | tail -c 4))}
else
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    MASTER_PORT=${MASTER_PORT:-29500}
fi
export MASTER_ADDR MASTER_PORT

mkdir -p "$OUTPUT_DIR"

PRECOMPUTE_ARGS=(
    --config "$CONFIG"
    --diting_config "$DITING_CONFIG"
    --diting_pretrained "$DITING_PRETRAINED"
    --dpk_checkpoint "$DPK_CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --station_batch_size "$STATION_BATCH_SIZE"
    --token_floor "$TOKEN_FLOOR"
    --dpk_weight_temperature "$DPK_WEIGHT_TEMPERATURE"
    --dpk_weight_resample "$DPK_WEIGHT_RESAMPLE"
    --save_dtype "$SAVE_DTYPE"
    --hdf5_name "$HDF5_NAME"
    --hdf5_prior_dtype "$HDF5_PRIOR_DTYPE"
    --hdf5_compression "$HDF5_COMPRESSION"
    --hdf5_gzip_level "$HDF5_GZIP_LEVEL"
    --hdf5_chunk_rows "$HDF5_CHUNK_ROWS"
)
if ((EXTRA_ARG_COUNT > 0)); then
    PRECOMPUTE_ARGS+=("${EXTRA_ARGS[@]}")
fi
PRECOMPUTE_CMD=(python tools/precompute_dpk_priors.py "${PRECOMPUTE_ARGS[@]}")

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export DIRECT_WORLD_SIZE="$SLURM_GPUS_PER_NODE"
    echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT world_size=$DIRECT_WORLD_SIZE"
    echo "[INFO] launching: srun --ntasks=$DIRECT_WORLD_SIZE --ntasks-per-node=$SLURM_GPUS_PER_NODE ${PRECOMPUTE_CMD[*]}"
    srun --ntasks="$DIRECT_WORLD_SIZE" --ntasks-per-node="$SLURM_GPUS_PER_NODE" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        bash -c 'export RANK="${SLURM_PROCID:?}"; export WORLD_SIZE="${DIRECT_WORLD_SIZE:?}"; export LOCAL_RANK="${SLURM_LOCALID:-0}"; echo "[INFO] rank=${RANK}/${WORLD_SIZE} local_rank=${LOCAL_RANK} host=$(hostname)"; exec "$@"' \
        bash "${PRECOMPUTE_CMD[@]}"
else
    if command -v torchrun >/dev/null 2>&1; then
        echo "[INFO] launching local torchrun with nproc_per_node=$SLURM_GPUS_PER_NODE"
        torchrun --standalone --nproc_per_node="$SLURM_GPUS_PER_NODE" \
            tools/precompute_dpk_priors.py "${PRECOMPUTE_ARGS[@]}"
    else
        echo "[WARN] torchrun unavailable; running single-process precompute." >&2
        "${PRECOMPUTE_CMD[@]}"
    fi
fi

echo "[INFO] DPK prior precompute finished: $OUTPUT_DIR"
