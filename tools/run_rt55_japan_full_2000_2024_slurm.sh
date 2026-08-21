#!/usr/bin/env bash

# Submit the rt55 KNET-only 2000-2024 run. Re-running in full mode resumes
# strictly from full_model_last.pth when it exists.
#
# Full run:
#   bash tools/run_rt55_japan_full_2000_2024_slurm.sh
#
# One-epoch, limited-data smoke run:
#   RUN_MODE=smoke bash tools/run_rt55_japan_full_2000_2024_slurm.sh
#
# Continue the same full run to a larger total epoch target:
#   EPOCHS_FULL_MODEL=20 bash tools/run_rt55_japan_full_2000_2024_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG=${CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$WORKDIR/train_light_slurm.sh}
JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/origin_corrected_diting_vel_acc_vs30}
RUN_MODE=${RUN_MODE:-full}
AUTO_RESUME_FULL_MODEL=${AUTO_RESUME_FULL_MODEL:-1}
REQUIRE_RESUME=${REQUIRE_RESUME:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_TRANSFER_CHECK=${SKIP_TRANSFER_CHECK:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}

case "$RUN_MODE" in
    full)
        JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
        EPOCHS_FULL_MODEL=${EPOCHS_FULL_MODEL:-12}
        JOB_NAME=${JOB_NAME:-team-full-rt55}
        SLURM_NODES=${SLURM_NODES:-4}
        SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
        SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
        SLURM_TIME=${SLURM_TIME:-3-00:00:00}
        test_args=()
        ;;
    smoke)
        JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-weights_japan_full_2000_2024_rt55_smoke_seed42}
        EPOCHS_FULL_MODEL=${EPOCHS_FULL_MODEL:-1}
        JOB_NAME=${JOB_NAME:-team-full-rt55-smoke}
        SLURM_NODES=${SLURM_NODES:-1}
        SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
        SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
        SLURM_TIME=${SLURM_TIME:-04:00:00}
        test_args=(--test_run)
        ;;
    *)
        echo "RUN_MODE must be full or smoke, got: $RUN_MODE" >&2
        exit 2
        ;;
esac

if [[ ! "$EPOCHS_FULL_MODEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPOCHS_FULL_MODEL must be a positive integer, got: $EPOCHS_FULL_MODEL" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Full-data config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Training launcher not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

transfer_path=$(python -c '
import json, os, sys
value = json.load(open(sys.argv[1]))["training_params"].get("transfer_model_path", "")
print(os.path.expanduser(os.path.expandvars(value)))
' "$CONFIG")
if [[ -n "$transfer_path" && "$SKIP_TRANSFER_CHECK" != "1" ]]; then
    case "$transfer_path" in
        /*) transfer_file="$transfer_path" ;;
        *) transfer_file="$WORKDIR/${transfer_path#./}" ;;
    esac
    if [[ ! -s "$transfer_file" ]]; then
        echo "Missing or empty transfer checkpoint: $transfer_file" >&2
        exit 1
    fi
fi
if [[ "${RESET_WEIGHT_PATH:-0}" == "1" ]]; then
    echo "RESET_WEIGHT_PATH=1 is forbidden by the full-data launcher." >&2
    exit 1
fi

missing=0
for year in $(seq 2000 2024); do
    shard="$JAPAN_FULL_DATA_ROOT/$year/japan_${year}.hdf5"
    if [[ ! -s "$shard" ]]; then
        echo "Missing or empty annual shard: $shard" >&2
        missing=1
    fi
done
if [[ "$missing" == "1" ]]; then
    echo "Run tools/validate_japan_full_training_data.py after finishing the upload." >&2
    exit 1
fi

case "$JAPAN_FULL_WEIGHT_PATH" in
    /*) weight_dir="$JAPAN_FULL_WEIGHT_PATH" ;;
    *) weight_dir="$WORKDIR/${JAPAN_FULL_WEIGHT_PATH#./}" ;;
esac

resume_args=()
if [[ "$AUTO_RESUME_FULL_MODEL" == "1" || "$REQUIRE_RESUME" == "1" ]]; then
    if [[ -s "$weight_dir/full_model_last.pth" ]]; then
        resume_args=(--resume_full_model last)
    elif [[ "$REQUIRE_RESUME" == "1" ]]; then
        echo "Required continuation checkpoint is missing: $weight_dir/full_model_last.pth" >&2
        exit 1
    elif [[ -d "$weight_dir" ]]; then
        entry_count=$(find "$weight_dir" -maxdepth 1 -mindepth 1 -printf x | wc -c)
        if ((entry_count > 1)) || { ((entry_count == 1)) && [[ ! -f "$weight_dir/config.json" ]]; }; then
            echo "Weight directory is non-empty but has no full_model_last.pth: $weight_dir" >&2
            exit 1
        fi
    fi
fi

export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH

if [[ "$ALLOW_ACTIVE_JOB" != "1" ]] && command -v squeue >/dev/null 2>&1; then
    active_job=$(squeue --noheader --user "$(id -un)" --name "$JOB_NAME" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    if [[ -n "$active_job" ]]; then
        echo "A same-name Slurm job is already active: $active_job" >&2
        echo "Set ALLOW_ACTIVE_JOB=1 only if a duplicate run is intentional." >&2
        exit 1
    fi
fi

echo "[INFO] mode=$RUN_MODE config=$CONFIG"
echo "[INFO] data_root=$JAPAN_FULL_DATA_ROOT"
echo "[INFO] weight_dir=$weight_dir"
echo "[INFO] total_epoch_target=$EPOCHS_FULL_MODEL"
echo "[INFO] resources=${SLURM_NODES} nodes x ${SLURM_GPUS_PER_NODE} DCUs, walltime=$SLURM_TIME"
if ((${#resume_args[@]})); then
    echo "[INFO] continuation: ${resume_args[*]}"
else
    echo "[INFO] continuation: new run"
fi

# Some older Bash versions treat an initialized-but-empty array expansion as
# an unbound variable under `set -u`.  Build one guaranteed-nonempty argument
# vector and append optional flags only when present.
train_args=("$CONFIG" --epochs_full_model "$EPOCHS_FULL_MODEL")
if ((${#test_args[@]})); then
    train_args+=("${test_args[@]}")
fi
if ((${#resume_args[@]})); then
    train_args+=("${resume_args[@]}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[OK] dry-run validation passed; no job submitted."
    exit 0
fi

WORKDIR="$WORKDIR" \
JOB_NAME="$JOB_NAME" \
SLURM_NODES="$SLURM_NODES" \
SLURM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE" \
SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
SLURM_TIME="$SLURM_TIME" \
RUN_EVAL=0 \
RESET_WEIGHT_PATH=0 \
AUTO_SBATCH=1 \
bash "$TRAIN_SCRIPT" "${train_args[@]}"
