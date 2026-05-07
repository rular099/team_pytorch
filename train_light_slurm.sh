#!/usr/bin/env bash

# Slurm/HPC launcher for team_pytorch/train_light.py.
# Usage:
#   bash train_light_slurm.sh <config.json> [train_light.py extra args...]
#   AUTO_SBATCH=0 bash train_light_slurm.sh <config.json> [extra args...]
#
# Key env vars:
#   DITING_CONFIG      YAML config for the DiTing frontend
#   DITING_PRETRAINED  Optional pretrained checkpoint path used by the YAML / override
#   JOB_NAME           Slurm job name
#   SLURM_LOG_DIR      Directory for sbatch stdout/stderr
#   WORKDIR            Repo root to cd into before launching training
#   SLURM_PARTITION    Partition name
#   SLURM_NODES        Number of nodes to request when auto-submitting
#   SLURM_GPUS_PER_NODE GPUs per node to request and pass to torchrun
#   SLURM_CPUS_PER_TASK CPUs per task
#   SLURM_TIME         Optional wallclock limit, e.g. 24:00:00; unset by default
#   CONDA_ENV          Conda env name to activate after module loading
#   MODULE_UNLOAD      Optional module to unload
#   MODULE_LOADS       Space-separated modules to load
#   RESET_WEIGHT_PATH  Delete training_params.weight_path before training when set to 1
#   RUN_EVAL           Run eval_checkpoint.py after successful training when set to 1
#   EVAL_CHECKPOINT    Optional checkpoint path; defaults to full_model_last.pth, then best, then latest epoch checkpoint
#   EVAL_SINGLE_STATION_CHECKPOINT Optional single-station checkpoint path; defaults to best/last under weight_path
#   EVAL_DEVICE        Optional eval device, e.g. cuda:0
#   EVAL_OUTPUT_TXT    Optional eval stdout/stderr path; defaults to weight_path/eval_results.txt
#   EVAL_OUTPUT_NPZ    Optional eval npz path; defaults to weight_path/eval_results.npz

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
REPO_ROOT=${REPO_ROOT:-"$SUBMIT_DIR"}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}

CONFIG_INPUT=${1:?Usage: bash train_light_slurm.sh <config.json> [train_light.py extra args...]}
shift
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT=$#

JOB_NAME=${JOB_NAME:-team-train-light}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_NODES=${SLURM_NODES:-4}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
DITING_CONFIG=${DITING_CONFIG:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/diting/config/diting_1200m_backbone_attnpool.yml}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
RESET_WEIGHT_PATH=${RESET_WEIGHT_PATH:-1}
RUN_EVAL=${RUN_EVAL:-1}

resolve_path() {
    local p=$1
    local base=${2:-$PWD}
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *) printf '%s\n' "$base/$p" ;;
    esac
}

CONFIG=$(resolve_path "$CONFIG_INPUT" "$PWD")
#DITING_CONFIG=$(resolve_path "$DITING_CONFIG" "$WORKDIR")
DITING_PRETRAINED=/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt
if [[ -n "${DITING_PRETRAINED:-}" ]]; then
    DITING_PRETRAINED=$(resolve_path "$DITING_PRETRAINED" "$PWD")
fi

if [[ ! -f "$WORKDIR/train_light.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
    echo "Expected file not found: $WORKDIR/train_light.py" >&2
    echo "Set WORKDIR=/abs/path/to/team_pytorch when submitting." >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$DITING_CONFIG" ]]; then
    echo "DITING config file not found: $DITING_CONFIG" >&2
    echo "Set DITING_CONFIG to an absolute path or a path relative to WORKDIR=$WORKDIR." >&2
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
    echo "[INFO] submitting to Slurm"
    echo "[INFO] job_name=$JOB_NAME partition=$SLURM_PARTITION nodes=$SLURM_NODES gpus_per_node=$SLURM_GPUS_PER_NODE"
    EXPORT_VARS=(
        "WORKDIR=$WORKDIR"
        "REPO_ROOT=$REPO_ROOT"
        "SLURM_LOG_DIR=$SLURM_LOG_DIR"
        "DITING_CONFIG=$DITING_CONFIG"
        "CONFIG_INPUT=$CONFIG"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        EXPORT_VARS+=("DITING_PRETRAINED=$DITING_PRETRAINED")
    fi
    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME" \
        --partition="$SLURM_PARTITION" \
        --nodes="$SLURM_NODES" \
        --ntasks-per-node=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gres="dcu:${SLURM_GPUS_PER_NODE}" \
    )
    if [[ -n "$SLURM_TIME" ]]; then
        SBATCH_CMD+=(--time="$SLURM_TIME")
    fi
    SBATCH_CMD+=(
        --chdir="$WORKDIR" \
        --output="$SLURM_LOG_DIR/%x-%j.out" \
        --error="$SLURM_LOG_DIR/%x-%j.err" \
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")" \
        "$0" "$CONFIG"
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
    # Many HPC sites define the module function in /etc/profile for batch shells.
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

if [[ "$RESET_WEIGHT_PATH" == "1" ]]; then
    WEIGHT_PATH_TO_RESET=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$CONFIG")
    if [[ -z "$WEIGHT_PATH_TO_RESET" || "$WEIGHT_PATH_TO_RESET" == "/" || "$WEIGHT_PATH_TO_RESET" == "." || "$WEIGHT_PATH_TO_RESET" == ".." ]]; then
        echo "Refusing to reset unsafe weight_path: '$WEIGHT_PATH_TO_RESET'" >&2
        exit 1
    fi
    case "$WEIGHT_PATH_TO_RESET" in
        /*) WEIGHT_DIR_TO_RESET="$WEIGHT_PATH_TO_RESET" ;;
        *) WEIGHT_DIR_TO_RESET="$WORKDIR/$WEIGHT_PATH_TO_RESET" ;;
    esac
    echo "[INFO] RESET_WEIGHT_PATH=1; removing weight dir: $WEIGHT_DIR_TO_RESET"
    rm -rf -- "$WEIGHT_DIR_TO_RESET"
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

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_BIN=(torchrun)
elif command -v python >/dev/null 2>&1; then
    TORCHRUN_BIN=(python -m torch.distributed.run)
else
    echo "Neither torchrun nor python is available after environment setup." >&2
    exit 1
fi

echo "[INFO] python: $(command -v python || echo '<missing>')"
echo "[INFO] torchrun: $(command -v torchrun || echo '<python -m torch.distributed.run>')"

mkdir -p "$SLURM_LOG_DIR"

TORCHRUN_NNODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
    MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n "${SLURM_JOBID:-$$}" | tail -c 4))}
else
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    MASTER_PORT=${MASTER_PORT:-29500}
fi

TRAIN_CMD=(
    "${TORCHRUN_BIN[@]}"
    --nnodes="$TORCHRUN_NNODES"
    --nproc_per_node="$SLURM_GPUS_PER_NODE"
    --rdzv_backend=c10d
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv_id="${SLURM_JOBID:-local}"
    --node_rank="${SLURM_NODEID:-0}"
    train_light.py
    --config "$CONFIG"
    --diting_config "$DITING_CONFIG"
)

if [[ -n "${DITING_PRETRAINED:-}" ]]; then
    TRAIN_CMD+=(--diting_pretrained "$DITING_PRETRAINED")
fi
if ((EXTRA_ARG_COUNT > 0)); then
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] launching: ${TRAIN_CMD[*]}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        "${TRAIN_CMD[@]}"
else
    # Direct fallback for local smoke tests.
    DIRECT_CMD=(
        "${TORCHRUN_BIN[@]}"
        --standalone
        --nproc_per_node="$SLURM_GPUS_PER_NODE"
        train_light.py
        --config "$CONFIG"
        --diting_config "$DITING_CONFIG"
    )
    if [[ -n "${DITING_PRETRAINED:-}" ]]; then
        DIRECT_CMD+=(--diting_pretrained "$DITING_PRETRAINED")
    fi
    if ((EXTRA_ARG_COUNT > 0)); then
        DIRECT_CMD+=("${EXTRA_ARGS[@]}")
    fi
    "${DIRECT_CMD[@]}"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
    EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-$(python -c 'import glob, os, re, sys
weight_dir = sys.argv[1]
for name in ("full_model_last.pth", "full_model_best.pth"):
    path = os.path.join(weight_dir, name)
    if os.path.exists(path):
        print(path)
        sys.exit(0)
paths = glob.glob(os.path.join(weight_dir, "full_model_*.pth"))
def epoch(path):
    m = re.search(r"full_model_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1
print(max(paths, key=epoch) if paths else "")' "$WEIGHT_DIR")}
    if [[ -z "$EVAL_CHECKPOINT" || ! -f "$EVAL_CHECKPOINT" ]]; then
        echo "Eval checkpoint not found under $WEIGHT_DIR" >&2
        exit 1
    fi
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

    EVAL_OUTPUT_NPZ=${EVAL_OUTPUT_NPZ:-"$WEIGHT_DIR/eval_results.npz"}
    EVAL_OUTPUT_TXT=${EVAL_OUTPUT_TXT:-"$WEIGHT_DIR/eval_results.txt"}
    EVAL_CONFIG="$CONFIG"
    if [[ -f "$WEIGHT_DIR/config.json" ]]; then
        EVAL_CONFIG="$WEIGHT_DIR/config.json"
    fi
    mkdir -p "$(dirname "$EVAL_OUTPUT_TXT")" "$(dirname "$EVAL_OUTPUT_NPZ")"

    EVAL_CMD=(
        python eval_checkpoint.py
        --config "$EVAL_CONFIG"
        --diting_config "$DITING_CONFIG"
        --checkpoint "$EVAL_CHECKPOINT"
        --output "$EVAL_OUTPUT_NPZ"
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
        esac
    done

    echo "[INFO] running eval: ${EVAL_CMD[*]}"
    echo "[INFO] eval full checkpoint: $EVAL_CHECKPOINT"
    echo "[INFO] eval single-station checkpoint: ${EVAL_SINGLE_STATION_CHECKPOINT:-<disabled>}"
    echo "[INFO] eval txt: $EVAL_OUTPUT_TXT"
    echo "[INFO] eval npz: $EVAL_OUTPUT_NPZ"
    "${EVAL_CMD[@]}" >"$EVAL_OUTPUT_TXT" 2>&1
    echo "[INFO] eval finished; results written to $EVAL_OUTPUT_TXT"
fi
