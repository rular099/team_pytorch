#!/usr/bin/env bash

# Slurm/HPC launcher for eval_attention.py.
# Usage:
#   bash eval_attention_slurm.sh <config.json> [eval_attention.py extra args...]
#   AUTO_SBATCH=0 bash eval_attention_slurm.sh <config.json> [extra args...]
#
# Key env vars:
#   DITING_CONFIG      YAML config for the DiTing frontend
#   DITING_PRETRAINED  Optional pretrained checkpoint path used by the YAML / override
#   JOB_NAME           Slurm job name
#   SLURM_LOG_DIR      Directory for sbatch stdout/stderr
#   WORKDIR            Repo root to cd into before launching eval
#   SLURM_PARTITION    Partition name
#   SLURM_GPUS_PER_NODE GPUs per node to request; attention eval uses one process
#   SLURM_CPUS_PER_TASK CPUs per task
#   SLURM_TIME         Wallclock limit, e.g. 01:00:00
#   CONDA_ENV          Conda env name to activate after module loading
#   MODULE_UNLOAD      Optional module to unload
#   MODULE_LOADS       Space-separated modules to load
#   EVAL_CHECKPOINT    Optional single checkpoint path; when unset, evaluates full_model_last.pth and full_model_best.pth if present
#   EVAL_CONFIG        Optional config used for eval; defaults to weight_path/config.json when present, then input config
#   ATTENTION_OUTPUT_NPZ Optional output npz path; for multi-checkpoint eval, the checkpoint tag is appended
#   ATTENTION_OUTPUT_TXT Optional stdout/stderr path; for multi-checkpoint eval, the checkpoint tag is appended
#   ATTENTION_SPLITS   Comma-separated splits, e.g. val or train,val
#   ATTENTION_EVENT_INDICES Optional event indices, e.g. 'train:33,21;val:87,63'
#   ATTENTION_MAX_EVENTS Number of auto-selected events per split
#   ATTENTION_PROBE_EVENTS Auto-selection probe prefix size; 0 means full split
#   ATTENTION_STATION_COUNTS Optional station-count sweep, e.g. 3,5,8,12,16,25
#   ATTENTION_TOPK     Number of top-attended station slots to save per target
#   ATTENTION_SEED     Seed for station count sweep target selection consistency
#   EVAL_INPUT_STATION_SELECTION Input-station strategy: config, random, p_pick, epidist

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
REPO_ROOT=${REPO_ROOT:-"$SUBMIT_DIR"}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash eval_attention_slurm.sh <config.json> [eval_attention.py extra args...]}
shift
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-attn-eval}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-01:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
DITING_CONFIG=${DITING_CONFIG:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
ATTENTION_SPLITS=${ATTENTION_SPLITS:-val}
ATTENTION_MAX_EVENTS=${ATTENTION_MAX_EVENTS:-3}
ATTENTION_PROBE_EVENTS=${ATTENTION_PROBE_EVENTS:-0}
ATTENTION_STATION_COUNTS=${ATTENTION_STATION_COUNTS:-}
ATTENTION_TOPK=${ATTENTION_TOPK:-5}
ATTENTION_SEED=${ATTENTION_SEED:-1234}
EVAL_INPUT_STATION_SELECTION=${EVAL_INPUT_STATION_SELECTION:-epidist}

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

if [[ ! -f "$WORKDIR/eval_attention.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/eval_attention.py" >&2
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
    echo "[INFO] submitting attention eval to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION gpus_per_node=$SLURM_GPUS_PER_NODE"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "REPO_ROOT=$REPO_ROOT"
        "SLURM_LOG_DIR=$SLURM_LOG_DIR"
        "DITING_CONFIG=$DITING_CONFIG"
        "CONFIG_INPUT=$CONFIG"
        "ATTENTION_SPLITS=$ATTENTION_SPLITS"
        "ATTENTION_MAX_EVENTS=$ATTENTION_MAX_EVENTS"
        "ATTENTION_PROBE_EVENTS=$ATTENTION_PROBE_EVENTS"
        "ATTENTION_STATION_COUNTS=$ATTENTION_STATION_COUNTS"
        "ATTENTION_TOPK=$ATTENTION_TOPK"
        "ATTENTION_SEED=$ATTENTION_SEED"
        "EVAL_INPUT_STATION_SELECTION=$EVAL_INPUT_STATION_SELECTION"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        EXPORT_VARS+=("DITING_PRETRAINED=$DITING_PRETRAINED")
    fi
    if [[ -n "${ATTENTION_EVENT_INDICES:-}" ]]; then
        EXPORT_VARS+=("ATTENTION_EVENT_INDICES=$ATTENTION_EVENT_INDICES")
    fi
    if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
        EXPORT_VARS+=("EVAL_CHECKPOINT=$EVAL_CHECKPOINT")
    fi
    if [[ -n "${EVAL_CONFIG:-}" ]]; then
        EXPORT_VARS+=("EVAL_CONFIG=$EVAL_CONFIG")
    fi
    if [[ -n "${ATTENTION_OUTPUT_NPZ:-}" ]]; then
        EXPORT_VARS+=("ATTENTION_OUTPUT_NPZ=$ATTENTION_OUTPUT_NPZ")
    fi
    if [[ -n "${ATTENTION_OUTPUT_TXT:-}" ]]; then
        EXPORT_VARS+=("ATTENTION_OUTPUT_TXT=$ATTENTION_OUTPUT_TXT")
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
        --output="$SLURM_LOG_DIR/%x-%j.out"
        --error="$SLURM_LOG_DIR/%x-%j.err"
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")"
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
echo "[INFO] attention splits: $ATTENTION_SPLITS"
echo "[INFO] attention event indices: ${ATTENTION_EVENT_INDICES:-<auto>}"
echo "[INFO] attention station counts: ${ATTENTION_STATION_COUNTS:-<config>}"
echo "[INFO] input-station selection: $EVAL_INPUT_STATION_SELECTION"
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
echo "[INFO] eval checkpoints: ${EVAL_CHECKPOINTS[*]}"

for idx in "${!EVAL_CHECKPOINTS[@]}"; do
    checkpoint=${EVAL_CHECKPOINTS[$idx]}
    tag=${EVAL_TAGS[$idx]}
    if (( EXPLICIT_EVAL_CHECKPOINT == 1 )); then
        output_npz=${ATTENTION_OUTPUT_NPZ:-"$WEIGHT_DIR/eval_attention.npz"}
        output_txt=${ATTENTION_OUTPUT_TXT:-"$WEIGHT_DIR/eval_attention.txt"}
    else
        output_npz=${ATTENTION_OUTPUT_NPZ:-"$WEIGHT_DIR/eval_attention.npz"}
        output_txt=${ATTENTION_OUTPUT_TXT:-"$WEIGHT_DIR/eval_attention.txt"}
        output_npz=$(tagged_output_path "$output_npz" "$tag")
        output_txt=$(tagged_output_path "$output_txt" "$tag")
    fi
    mkdir -p "$(dirname "$output_txt")" "$(dirname "$output_npz")"

    ATTENTION_CMD=(
        python eval_attention.py
        --config "$EVAL_CONFIG"
        --diting_config "$DITING_CONFIG"
        --checkpoint "$checkpoint"
        --output "$output_npz"
        --splits "$ATTENTION_SPLITS"
        --max_events "$ATTENTION_MAX_EVENTS"
        --probe_events "$ATTENTION_PROBE_EVENTS"
        --topk "$ATTENTION_TOPK"
        --seed "$ATTENTION_SEED"
        --input_station_selection "$EVAL_INPUT_STATION_SELECTION"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        ATTENTION_CMD+=(--diting_pretrained "$DITING_PRETRAINED")
    fi
    if [[ -n "${ATTENTION_EVENT_INDICES:-}" ]]; then
        ATTENTION_CMD+=(--event_indices "$ATTENTION_EVENT_INDICES")
    fi
    if [[ -n "${ATTENTION_STATION_COUNTS:-}" ]]; then
        ATTENTION_CMD+=(--station_counts "$ATTENTION_STATION_COUNTS")
    fi
    if [[ -n "${EVAL_DEVICE:-}" ]]; then
        ATTENTION_CMD+=(--device "$EVAL_DEVICE")
    fi
    if ((EXTRA_ARG_COUNT > 0)); then
        ATTENTION_CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "[INFO] running attention eval ($tag): ${ATTENTION_CMD[*]}"
    echo "[INFO] eval full checkpoint ($tag): $checkpoint"
    echo "[INFO] attention txt ($tag): $output_txt"
    echo "[INFO] attention npz ($tag): $output_npz"
    "${ATTENTION_CMD[@]}" >"$output_txt" 2>&1
    echo "[INFO] attention eval finished ($tag); results written to $output_txt"
done
