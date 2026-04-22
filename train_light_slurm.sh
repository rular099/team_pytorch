#!/usr/bin/env bash

# Submit train_light.py to Slurm, following the few-shot pattern:
#   wrapper script -> sbatch script -> srun python ...
#
# Usage:
#   bash train_light_slurm.sh <config.json> [train_light.py extra args...]

set -euo pipefail

CONFIG_INPUT=${1:?Usage: bash train_light_slurm.sh <config.json> [extra args...]}
shift
EXTRA_ARGS=("$@")

DEFAULT_WORKDIR=/public/home/zhangbei/work_dir/zhangbei/teamexp_claude/team_pytorch
DEFAULT_DITING_CONFIG=$DEFAULT_WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml
DEFAULT_DITING_PRETRAINED=/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt

WORKDIR=${WORKDIR:-$DEFAULT_WORKDIR}
WORKDIR=$(cd "$WORKDIR" && pwd)
CONFIG=$(cd "$(dirname "$CONFIG_INPUT")" && pwd)/$(basename "$CONFIG_INPUT")
DITING_CONFIG_INPUT=${DITING_CONFIG:-$DEFAULT_DITING_CONFIG}
DITING_CONFIG=$(cd "$(dirname "$DITING_CONFIG_INPUT")" && pwd)/$(basename "$DITING_CONFIG_INPUT")
SBATCH_SCRIPT=${SBATCH_SCRIPT:-$WORKDIR/train_light_job.sbatch}

JOB_NAME=${JOB_NAME:-team-train-light}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_NODES=${SLURM_NODES:-1}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-24:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$WORKDIR/logs/slurm}

if [[ ! -f "$WORKDIR/train_light.py" ]]; then
    echo "WORKDIR does not look like team_pytorch repo root: $WORKDIR" >&2
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

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "SBATCH script not found: $SBATCH_SCRIPT" >&2
    exit 1
fi

mkdir -p "$SLURM_LOG_DIR"

DITING_PRETRAINED_ARG=${DITING_PRETRAINED:-$DEFAULT_DITING_PRETRAINED}

echo "[INFO] submitting to Slurm"
echo "[INFO] workdir=$WORKDIR"
echo "[INFO] config=$CONFIG"
echo "[INFO] diting_config=$DITING_CONFIG"
echo "[INFO] diting_pretrained=$DITING_PRETRAINED_ARG"
echo "[INFO] extra args: ${EXTRA_ARGS[*]:-<none>}"

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
    "$SBATCH_SCRIPT" \
    "$WORKDIR" \
    "$CONFIG" \
    "$DITING_CONFIG" \
    "$DITING_PRETRAINED_ARG" \
    "$SLURM_GPUS_PER_NODE" \
    "${EXTRA_ARGS[@]}"
