#!/bin/bash
# Usage:
#   bash train_light.sh <config>                          # full training, 4 GPUs
#   bash train_light.sh <config> --overfit_n 16           # overfit test, 1 GPU
#   bash train_light.sh <config> --overfit_n 16 --test_run
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  (default: auto based on overfit mode)
#   NPROC_PER_NODE        (default: 1 if overfit, 4 otherwise)
#   DITING_CONFIG         (default: ./diting/config/conf_reg.yml)

set -e

CONFIG=${1:?Usage: bash train_light.sh <config> [extra args...]}
shift

DITING_CONFIG=${DITING_CONFIG:-./diting/config/conf_reg.yml}

# Extract weight_path from config
WEIGHT_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training_params']['weight_path'])")

# Detect overfit mode from args
OVERFIT=0
for arg in "$@"; do
    if [ "$arg" = "--overfit_n" ]; then
        OVERFIT=1
    fi
done

if [ "$OVERFIT" = "1" ]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:-1}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    NPROC_PER_NODE=${NPROC_PER_NODE:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi

echo "=== Train Config ==="
echo "  Config:    $CONFIG"
echo "  Weights:   $WEIGHT_PATH"
echo "  Diting:    $DITING_CONFIG"
echo "  GPUs:      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, nproc=$NPROC_PER_NODE"
echo "  Extra args: $@"
echo ""

# Clean previous run artifacts
rm -rf "$WEIGHT_PATH" "runs/$WEIGHT_PATH" train_ev.csv test_ev.csv event_metadata.csv
echo "Cleaned previous results."

# Run training
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    train_light.py --config "$CONFIG" --diting_config "$DITING_CONFIG" "$@"
