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
#   SLURM_TIME         Wallclock limit, e.g. 24:00:00
#   CONDA_ENV          Conda env name to activate after module loading
#   MODULE_UNLOAD      Optional module to unload
#   MODULE_LOADS       Space-separated modules to load

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
REPO_ROOT=${REPO_ROOT:-"$SUBMIT_DIR"}
WORKDIR=${WORKDIR:-"$REPO_ROOT"}

CONFIG_INPUT=${1:?Usage: bash train_light_slurm.sh <config.json> [train_light.py extra args...]}
shift
EXTRA_ARGS=("$@")

JOB_NAME=${JOB_NAME:-team-train-light}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_NODES=${SLURM_NODES:-1}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-24:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$WORKDIR/logs/slurm"}
DITING_CONFIG=${DITING_CONFIG:-./diting/config/diting_1200m_backbone_attnpool.yml}
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

CONFIG=$(resolve_path "$CONFIG_INPUT" "$PWD")
DITING_CONFIG=$(resolve_path "$DITING_CONFIG" "$WORKDIR")
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
    exec sbatch \
        --job-name="$JOB_NAME" \
        --partition="$SLURM_PARTITION" \
        --nodes="$SLURM_NODES" \
        --ntasks-per-node=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gres="dcu:${SLURM_GPUS_PER_NODE}" \
        --time="$SLURM_TIME" \
        --chdir="$WORKDIR" \
        --output="$SLURM_LOG_DIR/%x-%j.out" \
        --error="$SLURM_LOG_DIR/%x-%j.err" \
        --export="$(IFS=,; echo "ALL,${EXPORT_VARS[*]}")" \
        "$0" "$CONFIG" "${EXTRA_ARGS[@]}"
fi

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] config: $CONFIG"
echo "[INFO] diting_config: $DITING_CONFIG"
echo "[INFO] diting_pretrained: ${DITING_PRETRAINED:-<unset>}"
echo "[INFO] extra args: ${EXTRA_ARGS[*]:-<none>}"

if [[ -f /etc/profile ]]; then
    # Many HPC sites define the module function in /etc/profile for batch shells.
    # shellcheck disable=SC1091
    source /etc/profile
fi
if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
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
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        source activate "$CONDA_ENV"
    fi
fi

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
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

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
    DIRECT_CMD+=("${EXTRA_ARGS[@]}")
    "${DIRECT_CMD[@]}"
fi
