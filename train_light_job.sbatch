#!/bin/bash
#SBATCH -J team-train-light
#SBATCH -p diting
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:4

set -e

module unload compiler/rocm/2.9
module load compiler/rocm/dtk-23.04
module load apps/miniconda/3
source activate lsm_env

WORKDIR=$1
CONFIG=$2
DITING_CONFIG=$3
DITING_PRETRAINED=$4
NPROC_PER_NODE=$5
shift 5
EXTRA_ARGS=("$@")

DITINGBENCH_ROOT=${DITINGBENCH_ROOT:-$(cd "$WORKDIR/.." && pwd)/ditingbench}
if [[ ! -d "$DITINGBENCH_ROOT/dtbench" ]]; then
    echo "DITINGBENCH_ROOT is invalid: $DITINGBENCH_ROOT" >&2
    echo "Expected package dir not found: $DITINGBENCH_ROOT/dtbench" >&2
    echo "Place ditingbench next to team_pytorch, or set DITINGBENCH_ROOT explicitly." >&2
    exit 1
fi
export PYTHONPATH="$DITINGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export MASTER_PORT=$(expr 10000 + $(echo -n "${SLURM_JOBID}" | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
TORCHRUN_RDZV_MODE=${TORCHRUN_RDZV_MODE:-static}

cd "$WORKDIR"
echo "[INFO] cd to: $(pwd)"
echo "[INFO] python: $(command -v python)"
echo "[INFO] DITINGBENCH_ROOT=$DITINGBENCH_ROOT"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

case "$TORCHRUN_RDZV_MODE" in
    static)
        TRAIN_CMD=(
            python -m torch.distributed.run
            --nnodes="${SLURM_JOB_NUM_NODES:-1}"
            --nproc_per_node="$NPROC_PER_NODE"
            --node_rank="${SLURM_NODEID:-0}"
            --master_addr="$MASTER_ADDR"
            --master_port="$MASTER_PORT"
            train_light.py
            --config "$CONFIG"
            --diting_config "$DITING_CONFIG"
        )
        ;;
    c10d)
        TRAIN_CMD=(
            python -m torch.distributed.run
            --nnodes="${SLURM_JOB_NUM_NODES:-1}"
            --nproc_per_node="$NPROC_PER_NODE"
            --rdzv_backend=c10d
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
            --rdzv_id="${SLURM_JOBID}"
            --node_rank="${SLURM_NODEID:-0}"
            train_light.py
            --config "$CONFIG"
            --diting_config "$DITING_CONFIG"
        )
        ;;
    *)
        echo "Unsupported TORCHRUN_RDZV_MODE=$TORCHRUN_RDZV_MODE; expected static or c10d" >&2
        exit 1
        ;;
esac

if [[ "$DITING_PRETRAINED" != "__EMPTY__" ]]; then
    TRAIN_CMD+=(--diting_pretrained "$DITING_PRETRAINED")
fi
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

echo "[INFO] launching: ${TRAIN_CMD[*]}"
echo "[INFO] rdzv_mode=$TORCHRUN_RDZV_MODE"

srun "${TRAIN_CMD[@]}"
