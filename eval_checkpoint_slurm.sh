#!/usr/bin/env bash

# Standalone Slurm/HPC launcher for team_pytorch/eval_checkpoint.py.
# It mirrors the RUN_EVAL block in train_light_slurm.sh:
#   - default config copy: logs/<weight_path>/config.json
#   - default outputs: logs/<weight_path>/eval_results_last.txt/.npz
#                      logs/<weight_path>/eval_results_best.txt/.npz
#     plus one adjacent *.metrics.json file containing formal PGA metrics
#   - no sharding; one complete eval output per checkpoint
#
# Usage:
#   bash eval_checkpoint_slurm.sh <config.json> [eval_checkpoint.py extra args...]
#   AUTO_SBATCH=0 bash eval_checkpoint_slurm.sh <config.json> [extra args...]
#
# Key env vars:
#   WORKDIR            Repo root to cd into before launching eval
#   DITING_CONFIG      YAML config for the DiTing frontend
#   DITING_PRETRAINED  Optional pretrained checkpoint path used by YAML / override
#   JOB_NAME           Slurm job name
#   SLURM_LOG_DIR      Directory for sbatch stdout/stderr
#   SLURM_PARTITION    Partition name
#   SLURM_GPUS         GPUs to request; eval uses one process
#   SLURM_CPUS_PER_TASK CPUs per task
#   SLURM_TIME         Wallclock limit; defaults to 12:00:00
#   RUN_LOG_DIR        Optional run log dir; defaults to logs/<weight_path>
#   EVAL_CHECKPOINT    Optional checkpoint path; when unset, evaluates last and best
#   EVAL_SINGLE_STATION_CHECKPOINT Optional single-station checkpoint path
#   EVAL_DEVICE        Optional eval device, e.g. cuda:0
#   EVAL_OUTPUT_TXT    Optional stdout/stderr path; only valid with EVAL_CHECKPOINT
#   EVAL_OUTPUT_NPZ    Optional npz path; only valid with EVAL_CHECKPOINT
#   EVAL_OUTPUT_SUFFIX Optional suffix for default eval_results_<label> outputs
#   CONDA_ENV          Conda env name to activate after module loading
#   MODULE_UNLOAD      Optional module to unload
#   MODULE_LOADS       Space-separated modules to load

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash eval_checkpoint_slurm.sh <config.json> [eval_checkpoint.py extra args...]}
shift
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-eval-light}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GPUS=${SLURM_GPUS:-1}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-12:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
DITING_CONFIG=${DITING_CONFIG:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}
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

SCRIPT_PATH=$(resolve_path "$0" "$SUBMIT_DIR")
CONFIG=$(resolve_path "$CONFIG_INPUT" "$PWD")
if [[ -n "${DITING_PRETRAINED:-}" ]]; then
    DITING_PRETRAINED=$(resolve_path "$DITING_PRETRAINED" "$PWD")
fi
if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
    EVAL_CHECKPOINT=$(resolve_path "$EVAL_CHECKPOINT" "$PWD")
fi
if [[ -n "${EVAL_SINGLE_STATION_CHECKPOINT:-}" ]]; then
    EVAL_SINGLE_STATION_CHECKPOINT=$(resolve_path "$EVAL_SINGLE_STATION_CHECKPOINT" "$PWD")
fi

if [[ ! -f "$WORKDIR/eval_checkpoint.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/eval_checkpoint.py" >&2
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
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION gpus=$SLURM_GPUS"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "SLURM_LOG_DIR=$SLURM_LOG_DIR"
        "DITING_CONFIG=$DITING_CONFIG"
        "DITING_PRETRAINED=$DITING_PRETRAINED"
        "CONDA_ENV=$CONDA_ENV"
        "MODULE_UNLOAD=$MODULE_UNLOAD"
        "MODULE_LOADS=$MODULE_LOADS"
    )
    if [[ -n "${RUN_LOG_DIR:-}" ]]; then
        EXPORT_VARS+=("RUN_LOG_DIR=$RUN_LOG_DIR")
    fi
    if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
        EXPORT_VARS+=("EVAL_CHECKPOINT=$EVAL_CHECKPOINT")
    fi
    if [[ -n "${EVAL_SINGLE_STATION_CHECKPOINT:-}" ]]; then
        EXPORT_VARS+=("EVAL_SINGLE_STATION_CHECKPOINT=$EVAL_SINGLE_STATION_CHECKPOINT")
    fi
    if [[ -n "${EVAL_DEVICE:-}" ]]; then
        EXPORT_VARS+=("EVAL_DEVICE=$EVAL_DEVICE")
    fi
    if [[ -n "${EVAL_OUTPUT_TXT:-}" ]]; then
        EXPORT_VARS+=("EVAL_OUTPUT_TXT=$EVAL_OUTPUT_TXT")
    fi
    if [[ -n "${EVAL_OUTPUT_NPZ:-}" ]]; then
        EXPORT_VARS+=("EVAL_OUTPUT_NPZ=$EVAL_OUTPUT_NPZ")
    fi
    if [[ -n "${EVAL_OUTPUT_SUFFIX:-}" ]]; then
        EXPORT_VARS+=("EVAL_OUTPUT_SUFFIX=$EVAL_OUTPUT_SUFFIX")
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
    )
    if ((EXTRA_ARG_COUNT > 0)); then
        SBATCH_CMD+=("${EXTRA_ARGS[@]}")
    fi
    exec "${SBATCH_CMD[@]}"
fi

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] config: $CONFIG"
echo "[INFO] diting_config: $DITING_CONFIG"
echo "[INFO] diting_pretrained: ${DITING_PRETRAINED:-<unset>}"
if ((EXTRA_ARG_COUNT > 0)); then
    echo "[INFO] extra eval args: ${EXTRA_ARGS[*]}"
else
    echo "[INFO] extra eval args: <none>"
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

WEIGHT_PATH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$CONFIG")
if [[ -z "$WEIGHT_PATH" || "$WEIGHT_PATH" == "/" || "$WEIGHT_PATH" == "." || "$WEIGHT_PATH" == ".." ]]; then
    echo "Unsafe weight_path in config: '$WEIGHT_PATH'" >&2
    exit 1
fi
case "$WEIGHT_PATH" in
    /*) WEIGHT_DIR="$WEIGHT_PATH" ;;
    *) WEIGHT_DIR="$WORKDIR/$WEIGHT_PATH" ;;
esac
WEIGHT_LOG_NAME=${WEIGHT_PATH#./}
case "$WEIGHT_LOG_NAME" in
    /*) WEIGHT_LOG_NAME=$(basename "$WEIGHT_LOG_NAME") ;;
esac
RUN_LOG_DIR=${RUN_LOG_DIR:-"$WORKDIR/logs/$WEIGHT_LOG_NAME"}

mkdir -p "$RUN_LOG_DIR"
RUN_CONFIG="$CONFIG"
if [[ -f "$WEIGHT_DIR/config.json" ]]; then
    RUN_CONFIG="$WEIGHT_DIR/config.json"
fi
cp "$RUN_CONFIG" "$RUN_LOG_DIR/config.json"
echo "[INFO] eval config copied to: $RUN_LOG_DIR/config.json"

SINGLE_STATION_ENABLED=$(python -c 'import json, sys
cfg = json.load(open(sys.argv[1]))
print("1" if cfg["training_params"].get("single_station_pretrain", {}).get("enabled", False) else "0")' "$CONFIG")
if [[ -z "${EVAL_SINGLE_STATION_CHECKPOINT:-}" && "$SINGLE_STATION_ENABLED" == "1" ]]; then
    if [[ -f "$WEIGHT_DIR/single_station_best.pth" ]]; then
        EVAL_SINGLE_STATION_CHECKPOINT="$WEIGHT_DIR/single_station_best.pth"
    elif [[ -f "$WEIGHT_DIR/single_station_last.pth" ]]; then
        EVAL_SINGLE_STATION_CHECKPOINT="$WEIGHT_DIR/single_station_last.pth"
    elif [[ -f "$WEIGHT_DIR/single_station_final.pth" ]]; then
        EVAL_SINGLE_STATION_CHECKPOINT="$WEIGHT_DIR/single_station_final.pth"
    else
        echo "Single-station eval checkpoint not found under $WEIGHT_DIR" >&2
        exit 1
    fi
fi

EVAL_CONFIG="$CONFIG"
if [[ -f "$WEIGHT_DIR/config.json" ]]; then
    EVAL_CONFIG="$WEIGHT_DIR/config.json"
fi

EVAL_CHECKPOINT_PATHS=()
EVAL_CHECKPOINT_LABELS=()
if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$EVAL_CHECKPOINT" ]]; then
        echo "Eval checkpoint not found: $EVAL_CHECKPOINT" >&2
        exit 1
    fi
    EVAL_CHECKPOINT_PATHS+=("$EVAL_CHECKPOINT")
    EVAL_CHECKPOINT_LABELS+=("custom")
else
    for spec in "last:full_model_last.pth" "best:full_model_best.pth"; do
        label=${spec%%:*}
        filename=${spec#*:}
        path="$WEIGHT_DIR/$filename"
        if [[ -f "$path" ]]; then
            EVAL_CHECKPOINT_PATHS+=("$path")
            EVAL_CHECKPOINT_LABELS+=("$label")
        else
            echo "[WARN] eval checkpoint missing, skipping: $path" >&2
        fi
    done
fi
if (( ${#EVAL_CHECKPOINT_PATHS[@]} == 0 )); then
    echo "No eval checkpoints found under $WEIGHT_DIR" >&2
    exit 1
fi
if (( ${#EVAL_CHECKPOINT_PATHS[@]} > 1 )) && { [[ -n "${EVAL_OUTPUT_TXT:-}" ]] || [[ -n "${EVAL_OUTPUT_NPZ:-}" ]]; }; then
    echo "EVAL_OUTPUT_TXT/EVAL_OUTPUT_NPZ can only be used when EVAL_CHECKPOINT selects one checkpoint." >&2
    exit 1
fi
if (( ${#EVAL_CHECKPOINT_PATHS[@]} > 1 )); then
    for arg in "${EXTRA_ARGS[@]}"; do
        case "$arg" in
            --metrics_output|--metrics_output=*)
                echo "--metrics_output can only be used when EVAL_CHECKPOINT selects one checkpoint." >&2
                exit 1
                ;;
        esac
    done
fi

for idx in "${!EVAL_CHECKPOINT_PATHS[@]}"; do
    EVAL_CHECKPOINT_PATH=${EVAL_CHECKPOINT_PATHS[$idx]}
    EVAL_LABEL=${EVAL_CHECKPOINT_LABELS[$idx]}
    EVAL_SUFFIX=${EVAL_OUTPUT_SUFFIX:-}
    if [[ "$EVAL_LABEL" == "custom" ]]; then
        EVAL_OUTPUT_NPZ_PATH=${EVAL_OUTPUT_NPZ:-"$RUN_LOG_DIR/eval_results_custom.npz"}
        EVAL_OUTPUT_TXT_PATH=${EVAL_OUTPUT_TXT:-"$RUN_LOG_DIR/eval_results_custom.txt"}
    else
        EVAL_OUTPUT_NPZ_PATH="$RUN_LOG_DIR/eval_results_${EVAL_LABEL}${EVAL_SUFFIX}.npz"
        EVAL_OUTPUT_TXT_PATH="$RUN_LOG_DIR/eval_results_${EVAL_LABEL}${EVAL_SUFFIX}.txt"
    fi
    EVAL_METRICS_JSON_PATH="${EVAL_OUTPUT_NPZ_PATH%.npz}.metrics.json"
    mkdir -p "$(dirname "$EVAL_OUTPUT_TXT_PATH")" "$(dirname "$EVAL_OUTPUT_NPZ_PATH")"

    EVAL_CMD=(
        python eval_checkpoint.py
        --config "$EVAL_CONFIG"
        --diting_config "$DITING_CONFIG"
        --checkpoint "$EVAL_CHECKPOINT_PATH"
        --output "$EVAL_OUTPUT_NPZ_PATH"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        EVAL_CMD+=(--diting_pretrained "$DITING_PRETRAINED")
    fi
    if [[ -n "${EVAL_SINGLE_STATION_CHECKPOINT:-}" ]]; then
        EVAL_CMD+=(--single_station_checkpoint "$EVAL_SINGLE_STATION_CHECKPOINT")
    fi
    if [[ -n "${EVAL_DEVICE:-}" ]]; then
        EVAL_CMD+=(--device "$EVAL_DEVICE")
    fi
    for ((i = 0; i < EXTRA_ARG_COUNT; i++)); do
        case "${EXTRA_ARGS[$i]}" in
            --overfit_n)
                if ((i + 1 < EXTRA_ARG_COUNT)); then
                    EVAL_CMD+=(--overfit_n "${EXTRA_ARGS[$((i + 1))]}")
                    i=$((i + 1))
                fi
                ;;
            --overfit_n=*)
                EVAL_CMD+=("${EXTRA_ARGS[$i]}")
                ;;
            --splits|--metrics_output)
                if ((i + 1 >= EXTRA_ARG_COUNT)); then
                    echo "Missing value for ${EXTRA_ARGS[$i]}" >&2
                    exit 2
                fi
                EVAL_CMD+=("${EXTRA_ARGS[$i]}" "${EXTRA_ARGS[$((i + 1))]}")
                if [[ "${EXTRA_ARGS[$i]}" == "--metrics_output" ]]; then
                    EVAL_METRICS_JSON_PATH=${EXTRA_ARGS[$((i + 1))]}
                fi
                i=$((i + 1))
                ;;
            --splits=*)
                EVAL_CMD+=("${EXTRA_ARGS[$i]}")
                ;;
            --metrics_output=*)
                EVAL_CMD+=("${EXTRA_ARGS[$i]}")
                EVAL_METRICS_JSON_PATH=${EXTRA_ARGS[$i]#--metrics_output=}
                ;;
            --input_station_selection|--case_station_sweep|--case_station_counts|--case_splits|--case_event_indices|--case_max_events|--case_seed|--skip_single_station|--skip_diagnostics|--waveform_station_permutation|--waveform_station_permutation_seed|--no_permute_cached_token_weights)
                EVAL_CMD+=("${EXTRA_ARGS[$i]}")
                if [[ "${EXTRA_ARGS[$i]}" != "--case_station_sweep" && "${EXTRA_ARGS[$i]}" != "--skip_single_station" && "${EXTRA_ARGS[$i]}" != "--skip_diagnostics" && "${EXTRA_ARGS[$i]}" != "--no_permute_cached_token_weights" ]] && ((i + 1 < EXTRA_ARG_COUNT)); then
                    EVAL_CMD+=("${EXTRA_ARGS[$((i + 1))]}")
                    i=$((i + 1))
                fi
                ;;
            --input_station_selection=*|--case_station_counts=*|--case_splits=*|--case_event_indices=*|--case_max_events=*|--case_seed=*|--waveform_station_permutation=*|--waveform_station_permutation_seed=*)
                EVAL_CMD+=("${EXTRA_ARGS[$i]}")
                ;;
            --num_shards|--num_shards=*|--shard_id|--shard_id=*)
                echo "Refusing shard arg in standalone eval: ${EXTRA_ARGS[$i]}" >&2
                echo "This script intentionally writes one complete eval output per checkpoint." >&2
                exit 1
                ;;
        esac
    done

    echo "[INFO] running eval ($EVAL_LABEL): ${EVAL_CMD[*]}"
    echo "[INFO] eval config copied to: $RUN_LOG_DIR/config.json"
    echo "[INFO] eval full checkpoint: $EVAL_CHECKPOINT_PATH"
    echo "[INFO] eval single-station checkpoint: ${EVAL_SINGLE_STATION_CHECKPOINT:-<disabled>}"
    echo "[INFO] eval txt: $EVAL_OUTPUT_TXT_PATH"
    echo "[INFO] eval npz: $EVAL_OUTPUT_NPZ_PATH"
    echo "[INFO] eval metrics: $EVAL_METRICS_JSON_PATH"
    "${EVAL_CMD[@]}" >"$EVAL_OUTPUT_TXT_PATH" 2>&1
    if [[ ! -s "$EVAL_OUTPUT_NPZ_PATH" ]]; then
        echo "Eval finished without a non-empty NPZ: $EVAL_OUTPUT_NPZ_PATH" >&2
        exit 1
    fi
    if [[ ! -s "$EVAL_METRICS_JSON_PATH" ]]; then
        echo "Eval finished without a non-empty formal metrics JSON: $EVAL_METRICS_JSON_PATH" >&2
        exit 1
    fi
    echo "[INFO] eval finished ($EVAL_LABEL); results written to $EVAL_OUTPUT_TXT_PATH"
done
