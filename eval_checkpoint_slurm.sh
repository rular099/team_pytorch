#!/usr/bin/env bash

# Slurm/HPC launcher for eval_checkpoint.py only.
# Usage:
#   bash eval_checkpoint_slurm.sh <config.json> [eval_checkpoint.py extra args...]
#   AUTO_SBATCH=0 bash eval_checkpoint_slurm.sh <config.json> [extra args...]
#
# Key env vars:
#   DITING_CONFIG      YAML config for the DiTing frontend
#   DITING_PRETRAINED  Optional pretrained checkpoint path used by the YAML / override
#   JOB_NAME           Slurm job name
#   SLURM_LOG_DIR      Directory for sbatch stdout/stderr
#   WORKDIR            Repo root to cd into before launching eval
#   SLURM_PARTITION    Partition name
#   SLURM_GPUS_PER_NODE GPUs per node to request; eval uses one process
#   SLURM_CPUS_PER_TASK CPUs per task
#   SLURM_TIME         Wallclock limit, e.g. 04:00:00
#   CONDA_ENV          Conda env name to activate after module loading
#   MODULE_UNLOAD      Optional module to unload
#   MODULE_LOADS       Space-separated modules to load
#   EVAL_CHECKPOINT    Optional single checkpoint path; when unset, evaluates full_model_last.pth and full_model_best.pth if present
#   EVAL_CONFIG        Optional config used for eval; defaults to weight_path/config.json when present, then input config
#   EVAL_SINGLE_STATION_CHECKPOINT Optional single-station checkpoint path; defaults to best/last under weight_path
#   EVAL_DEVICE        Optional eval device, e.g. cuda:0
#   EVAL_OUTPUT_TXT    Optional eval stdout/stderr path; for multi-checkpoint eval, the checkpoint tag is appended
#   EVAL_OUTPUT_NPZ    Optional eval npz path; for multi-checkpoint eval, the checkpoint tag is appended
#   EVAL_NUM_SHARDS    Number of independent eval shards; >1 submits a Slurm array job
#   EVAL_SHARD_ID      Explicit shard id for direct runs; normally taken from SLURM_ARRAY_TASK_ID
#   EVAL_INPUT_STATION_SELECTION Optional eval input-station strategy: config, random, p_pick, epidist
#   SLURM_ARRAY_PARALLELISM Optional max concurrent array tasks, e.g. 4
#   SKIP_SINGLE_STATION Set to 1 to disable single-station checkpoint eval

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
REPO_ROOT=${REPO_ROOT:-"$SUBMIT_DIR"}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash eval_checkpoint_slurm.sh <config.json> [eval_checkpoint.py extra args...]}
shift
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-eval}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-04:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
DITING_CONFIG=${DITING_CONFIG:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
SKIP_SINGLE_STATION=${SKIP_SINGLE_STATION:-0}
EVAL_NUM_SHARDS=${EVAL_NUM_SHARDS:-1}

resolve_path() {
    local p=$1
    local base=${2:-$PWD}
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *) printf '%s\n' "$base/$p" ;;
    esac
}

CONFIG=$(resolve_path "$CONFIG_INPUT" "$PWD")
if [[ -n "${DITING_PRETRAINED:-}" ]]; then
    DITING_PRETRAINED=$(resolve_path "$DITING_PRETRAINED" "$PWD")
fi

if [[ ! -f "$WORKDIR/eval_checkpoint.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/eval_checkpoint.py" >&2
    echo "Set WORKDIR=/abs/path/to/team_pytorch when submitting." >&2
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

if grep -q '\${DITING_PRETRAINED}' "$DITING_CONFIG"; then
    if [[ -z "${DITING_PRETRAINED:-}" ]]; then
        echo "DITING_CONFIG expects \${DITING_PRETRAINED}, but DITING_PRETRAINED is unset." >&2
        exit 1
    fi
fi

if [[ -z "${SLURM_JOB_ID:-}" && "${AUTO_SBATCH:-1}" != "0" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[INFO] submitting eval to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION gpus_per_node=$SLURM_GPUS_PER_NODE shards=$EVAL_NUM_SHARDS"
    SLURM_STDOUT="$SLURM_LOG_DIR/%x-%j.out"
    SLURM_STDERR="$SLURM_LOG_DIR/%x-%j.err"
    ARRAY_SPEC=()
    if (( EVAL_NUM_SHARDS > 1 )); then
        if [[ -n "${SLURM_ARRAY_PARALLELISM:-}" ]]; then
            ARRAY_SPEC=(--array="0-$((EVAL_NUM_SHARDS - 1))%${SLURM_ARRAY_PARALLELISM}")
        else
            ARRAY_SPEC=(--array="0-$((EVAL_NUM_SHARDS - 1))")
        fi
        SLURM_STDOUT="$SLURM_LOG_DIR/%x-%A_%a.out"
        SLURM_STDERR="$SLURM_LOG_DIR/%x-%A_%a.err"
    fi
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "REPO_ROOT=$REPO_ROOT"
        "SLURM_LOG_DIR=$SLURM_LOG_DIR"
        "DITING_CONFIG=$DITING_CONFIG"
        "CONFIG_INPUT=$CONFIG"
        "EVAL_NUM_SHARDS=$EVAL_NUM_SHARDS"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        EXPORT_VARS+=("DITING_PRETRAINED=$DITING_PRETRAINED")
    fi
    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$SLURM_PARTITION"
        --nodes=1
        --ntasks-per-node=1
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
        --gres="dcu:${SLURM_GPUS_PER_NODE}"
        --time="$SLURM_TIME"
        --chdir="$WORKDIR"
        --output="$SLURM_STDOUT"
        --error="$SLURM_STDERR"
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")"
        "${ARRAY_SPEC[@]}"
        "$0"
        "$CONFIG"
    )
    if ((EXTRA_ARG_COUNT > 0)); then
        SBATCH_CMD+=("${EXTRA_ARGS[@]}")
    fi
    exec "${SBATCH_CMD[@]}"
fi

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] input config: $CONFIG"
echo "[INFO] diting_config: $DITING_CONFIG"
echo "[INFO] diting_pretrained: ${DITING_PRETRAINED:-<unset>}"
EVAL_SHARD_ID=${EVAL_SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}
if (( EVAL_NUM_SHARDS < 1 )); then
    echo "EVAL_NUM_SHARDS must be >= 1, got $EVAL_NUM_SHARDS" >&2
    exit 1
fi
if (( EVAL_SHARD_ID < 0 || EVAL_SHARD_ID >= EVAL_NUM_SHARDS )); then
    echo "EVAL_SHARD_ID must be in [0, $EVAL_NUM_SHARDS), got $EVAL_SHARD_ID" >&2
    exit 1
fi
echo "[INFO] eval shard: $EVAL_SHARD_ID/$EVAL_NUM_SHARDS"
if ((EXTRA_ARG_COUNT > 0)); then
    echo "[INFO] extra args: ${EXTRA_ARGS[*]}"
else
    echo "[INFO] extra args: <none>"
fi

export COLORTERM=${COLORTERM:-truecolor}

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

WEIGHT_PATH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$CONFIG")
if [[ -z "$WEIGHT_PATH" || "$WEIGHT_PATH" == "/" || "$WEIGHT_PATH" == "." || "$WEIGHT_PATH" == ".." ]]; then
    echo "Unsafe weight_path in config: '$WEIGHT_PATH'" >&2
    exit 1
fi
case "$WEIGHT_PATH" in
    /*) WEIGHT_DIR="$WEIGHT_PATH" ;;
    *) WEIGHT_DIR="$WORKDIR/$WEIGHT_PATH" ;;
esac

if [[ -z "${EVAL_CONFIG:-}" && -f "$WEIGHT_DIR/config.json" ]]; then
    EVAL_CONFIG="$WEIGHT_DIR/config.json"
else
    EVAL_CONFIG=${EVAL_CONFIG:-"$CONFIG"}
fi
EVAL_CONFIG=$(resolve_path "$EVAL_CONFIG" "$PWD")
if [[ ! -f "$EVAL_CONFIG" ]]; then
    echo "Eval config file not found: $EVAL_CONFIG" >&2
    exit 1
fi

EVAL_CHECKPOINTS=()
EVAL_TAGS=()
EXPLICIT_EVAL_CHECKPOINT=0
if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
    EXPLICIT_EVAL_CHECKPOINT=1
    if [[ ! -f "$EVAL_CHECKPOINT" ]]; then
        echo "Eval checkpoint not found: $EVAL_CHECKPOINT" >&2
        exit 1
    fi
    EVAL_CHECKPOINTS+=("$EVAL_CHECKPOINT")
    EVAL_TAGS+=("$(basename "$EVAL_CHECKPOINT" .pth | sed 's/^full_model_//')")
else
    if [[ -f "$WEIGHT_DIR/full_model_last.pth" ]]; then
        EVAL_CHECKPOINTS+=("$WEIGHT_DIR/full_model_last.pth")
        EVAL_TAGS+=("last")
    fi
    if [[ -f "$WEIGHT_DIR/full_model_best.pth" ]]; then
        EVAL_CHECKPOINTS+=("$WEIGHT_DIR/full_model_best.pth")
        EVAL_TAGS+=("best")
    fi
    if (( ${#EVAL_CHECKPOINTS[@]} == 0 )); then
        LATEST_CHECKPOINT=$(python -c 'import glob, os, re, sys
weight_dir = sys.argv[1]
paths = glob.glob(os.path.join(weight_dir, "full_model_*.pth"))
def epoch(path):
    m = re.search(r"full_model_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1
print(max(paths, key=epoch) if paths else "")' "$WEIGHT_DIR")
        if [[ -n "$LATEST_CHECKPOINT" && -f "$LATEST_CHECKPOINT" ]]; then
            EVAL_CHECKPOINTS+=("$LATEST_CHECKPOINT")
            EVAL_TAGS+=("$(basename "$LATEST_CHECKPOINT" .pth | sed 's/^full_model_//')")
        fi
    fi
fi
if (( ${#EVAL_CHECKPOINTS[@]} == 0 )); then
    echo "Eval checkpoint not found under $WEIGHT_DIR" >&2
    exit 1
fi

if [[ "$SKIP_SINGLE_STATION" != "1" ]]; then
    if [[ -n "${EVAL_SINGLE_STATION_CHECKPOINT:-}" ]]; then
        if [[ ! -f "$EVAL_SINGLE_STATION_CHECKPOINT" ]]; then
            echo "Single-station eval checkpoint not found: $EVAL_SINGLE_STATION_CHECKPOINT" >&2
            exit 1
        fi
    elif [[ -f "$WEIGHT_DIR/single_station_best.pth" ]]; then
        EVAL_SINGLE_STATION_CHECKPOINT="$WEIGHT_DIR/single_station_best.pth"
    elif [[ -f "$WEIGHT_DIR/single_station_last.pth" ]]; then
        EVAL_SINGLE_STATION_CHECKPOINT="$WEIGHT_DIR/single_station_last.pth"
    elif [[ -f "$WEIGHT_DIR/single_station_final.pth" ]]; then
        EVAL_SINGLE_STATION_CHECKPOINT="$WEIGHT_DIR/single_station_final.pth"
    fi
fi

tagged_output_path() {
    local path=$1
    local tag=$2
    local dir base stem ext
    dir=$(dirname "$path")
    base=$(basename "$path")
    case "$base" in
        *.*)
            stem=${base%.*}
            ext=.${base##*.}
            ;;
        *)
            stem=$base
            ext=
            ;;
    esac
    printf '%s/%s_%s%s\n' "$dir" "$stem" "$tag" "$ext"
}

echo "[INFO] python: $(command -v python || echo '<missing>')"
echo "[INFO] eval config: $EVAL_CONFIG"
echo "[INFO] eval single-station checkpoint: ${EVAL_SINGLE_STATION_CHECKPOINT:-<disabled>}"
echo "[INFO] eval checkpoints: ${EVAL_CHECKPOINTS[*]}"
echo "[INFO] eval input-station selection: ${EVAL_INPUT_STATION_SELECTION:-config}"
SHARD_TAG=
if (( EVAL_NUM_SHARDS > 1 )); then
    SHARD_TAG=$(printf 'shard%03dof%03d' "$EVAL_SHARD_ID" "$EVAL_NUM_SHARDS")
fi

for idx in "${!EVAL_CHECKPOINTS[@]}"; do
    checkpoint=${EVAL_CHECKPOINTS[$idx]}
    tag=${EVAL_TAGS[$idx]}
    if (( EXPLICIT_EVAL_CHECKPOINT == 1 )); then
        output_npz=${EVAL_OUTPUT_NPZ:-"$WEIGHT_DIR/eval_results.npz"}
        output_txt=${EVAL_OUTPUT_TXT:-"$WEIGHT_DIR/eval_results.txt"}
    else
        output_npz=${EVAL_OUTPUT_NPZ:-"$WEIGHT_DIR/eval_results.npz"}
        output_txt=${EVAL_OUTPUT_TXT:-"$WEIGHT_DIR/eval_results.txt"}
        output_npz=$(tagged_output_path "$output_npz" "$tag")
        output_txt=$(tagged_output_path "$output_txt" "$tag")
    fi
    if [[ -n "$SHARD_TAG" ]]; then
        output_npz=$(tagged_output_path "$output_npz" "$SHARD_TAG")
        output_txt=$(tagged_output_path "$output_txt" "$SHARD_TAG")
    fi
    mkdir -p "$(dirname "$output_txt")" "$(dirname "$output_npz")"

    EVAL_CMD=(
        python eval_checkpoint.py
        --config "$EVAL_CONFIG"
        --diting_config "$DITING_CONFIG"
        --checkpoint "$checkpoint"
        --output "$output_npz"
        --num_shards "$EVAL_NUM_SHARDS"
        --shard_id "$EVAL_SHARD_ID"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        EVAL_CMD+=(--diting_pretrained "$DITING_PRETRAINED")
    fi
    if [[ -n "${EVAL_SINGLE_STATION_CHECKPOINT:-}" ]]; then
        EVAL_CMD+=(--single_station_checkpoint "$EVAL_SINGLE_STATION_CHECKPOINT")
    fi
    if [[ "$SKIP_SINGLE_STATION" == "1" ]]; then
        EVAL_CMD+=(--skip_single_station)
    fi
    if [[ -n "${EVAL_DEVICE:-}" ]]; then
        EVAL_CMD+=(--device "$EVAL_DEVICE")
    fi
    if [[ -n "${EVAL_INPUT_STATION_SELECTION:-}" ]]; then
        EVAL_CMD+=(--input_station_selection "$EVAL_INPUT_STATION_SELECTION")
    fi
    if ((EXTRA_ARG_COUNT > 0)); then
        EVAL_CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "[INFO] running eval ($tag): ${EVAL_CMD[*]}"
    echo "[INFO] eval full checkpoint ($tag): $checkpoint"
    echo "[INFO] eval txt ($tag): $output_txt"
    echo "[INFO] eval npz ($tag): $output_npz"
    "${EVAL_CMD[@]}" >"$output_txt" 2>&1
    echo "[INFO] eval finished ($tag); results written to $output_txt"
done
