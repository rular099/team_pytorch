#!/usr/bin/env bash

# Submit the rt56 random-geometry experiments from the pinned rt55 epoch-32
# checkpoint.  ACTION=all submits two independent jobs:
#   1. ep32 zero-shot validation with a causal random input mask;
#   2. six-epoch mixed-random fine-tuning (50% rt55, 50% random geometry).
#
# Dry run:
#   DRY_RUN=1 ACTION=all bash tools/run_rt56_random_geometry_slurm.sh
#
# Submit both jobs:
#   CONFIRM_RT56=1 ACTION=all bash tools/run_rt56_random_geometry_slurm.sh
#
# Submit only one branch:
#   CONFIRM_RT56=1 ACTION=zero_shot bash tools/run_rt56_random_geometry_slurm.sh
#   CONFIRM_RT56=1 ACTION=finetune bash tools/run_rt56_random_geometry_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG=${CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$WORKDIR/train_light_slurm.sh}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}
JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
RT55_WEIGHT_NAME=${RT55_WEIGHT_NAME:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT55_WEIGHT_DIR=${RT55_WEIGHT_DIR:-$WORKDIR/$RT55_WEIGHT_NAME}
RT55_EP32_CHECKPOINT=${RT55_EP32_CHECKPOINT:-$RT55_WEIGHT_DIR/full_model_best_ep32.pth}
# The parent rt55 config references this placeholder.  rt56 overrides its
# weight_path after inheritance, but exporting it also keeps older config
# loaders (which expanded parents before merging) operational on the cluster.
JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-$RT55_WEIGHT_NAME}
RT56_WEIGHT_NAME=${RT56_WEIGHT_NAME:-weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT56_WEIGHT_PATH=${RT56_WEIGHT_PATH:-$RT56_WEIGHT_NAME}
ACTION=${ACTION:-all}
EPOCHS_FULL_MODEL=${EPOCHS_FULL_MODEL:-6}
CONFIRM_RT56=${CONFIRM_RT56:-0}
DRY_RUN=${DRY_RUN:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}

ZERO_SHOT_JOB_NAME=${ZERO_SHOT_JOB_NAME:-team-rt56-zero-shot}
FINETUNE_JOB_NAME=${FINETUNE_JOB_NAME:-team-rt56-finetune}
ZERO_SHOT_TIME=${ZERO_SHOT_TIME:-2-00:00:00}
FINETUNE_TIME=${FINETUNE_TIME:-3-00:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
FINETUNE_NODES=${FINETUNE_NODES:-4}
FINETUNE_GPUS_PER_NODE=${FINETUNE_GPUS_PER_NODE:-4}

case "$ACTION" in
    zero_shot|finetune|all) ;;
    *)
        echo "ACTION must be zero_shot, finetune, or all; got: $ACTION" >&2
        exit 2
        ;;
esac
if [[ ! "$EPOCHS_FULL_MODEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPOCHS_FULL_MODEL must be a positive integer; got: $EPOCHS_FULL_MODEL" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "1" && "$CONFIRM_RT56" != "1" ]]; then
    echo "rt56 submission requires CONFIRM_RT56=1." >&2
    exit 2
fi

require_file() {
    local path=$1
    local label=$2
    if [[ -s "$path" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN WARN] $label is not visible: $path" >&2
        return 0
    fi
    echo "$label is missing or empty: $path" >&2
    exit 1
}

require_file "$CONFIG" "rt56 config"
require_file "$TRAIN_SCRIPT" "training launcher"
require_file "$EVAL_SCRIPT" "evaluation launcher"
require_file "$RT55_EP32_CHECKPOINT" "pinned rt55 epoch-32 checkpoint"

missing_shard=0
for year in $(seq 2000 2024); do
    shard="$JAPAN_FULL_DATA_ROOT/$year/japan_${year}.hdf5"
    if [[ ! -s "$shard" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[DRY-RUN WARN] annual shard is not visible: $shard" >&2
        else
            echo "Missing or empty annual shard: $shard" >&2
            missing_shard=1
        fi
    fi
done
if [[ "$missing_shard" == "1" ]]; then
    exit 1
fi

case "$RT56_WEIGHT_PATH" in
    /*) RT56_WEIGHT_DIR=$RT56_WEIGHT_PATH ;;
    *) RT56_WEIGHT_DIR="$WORKDIR/${RT56_WEIGHT_PATH#./}" ;;
esac
if [[ -z "$RT56_WEIGHT_DIR" || "$RT56_WEIGHT_DIR" == "/" || "$RT56_WEIGHT_DIR" == "$WORKDIR" ]]; then
    echo "Unsafe rt56 weight directory: $RT56_WEIGHT_DIR" >&2
    exit 1
fi

ZERO_SHOT_LOG_DIR=${ZERO_SHOT_LOG_DIR:-$WORKDIR/logs/$RT56_WEIGHT_NAME/zero_shot_ep32}
ZERO_SHOT_STEM="$ZERO_SHOT_LOG_DIR/eval_validation_ep32_zero_shot_random_mask"
if [[ "$ACTION" != "finetune" && "$ALLOW_EXISTING_OUTPUT" != "1" ]] && {
    [[ -s "$ZERO_SHOT_STEM.txt" ]] ||
    [[ -s "$ZERO_SHOT_STEM.npz" ]] ||
    [[ -s "$ZERO_SHOT_STEM.metrics.json" ]]
}; then
    echo "Zero-shot output already exists; refusing to overwrite: $ZERO_SHOT_STEM" >&2
    exit 1
fi

if [[ "$ACTION" == "finetune" || "$ACTION" == "all" ]]; then
    if [[ -d "$RT56_WEIGHT_DIR" ]]; then
        entry_count=$(find "$RT56_WEIGHT_DIR" -maxdepth 1 -mindepth 1 -printf x | wc -c)
        if ((entry_count > 0)); then
            echo "rt56 fine-tune weight directory must be new and empty: $RT56_WEIGHT_DIR" >&2
            exit 1
        fi
    fi
fi

check_active_job() {
    local job_name=$1
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]] || ! command -v squeue >/dev/null 2>&1; then
        return 0
    fi
    local active_job
    active_job=$(squeue --noheader --user "$(id -un)" --name "$job_name" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    if [[ -n "$active_job" ]]; then
        echo "A same-name Slurm job is already active: $job_name $active_job" >&2
        exit 1
    fi
}

if [[ "$ACTION" == "zero_shot" || "$ACTION" == "all" ]]; then
    check_active_job "$ZERO_SHOT_JOB_NAME"
fi
if [[ "$ACTION" == "finetune" || "$ACTION" == "all" ]]; then
    check_active_job "$FINETUNE_JOB_NAME"
fi

export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT55_EP32_CHECKPOINT RT56_WEIGHT_PATH

echo "[INFO] action=$ACTION"
echo "[INFO] source_checkpoint=$RT55_EP32_CHECKPOINT"
echo "[INFO] inherited_rt55_weight_path=$JAPAN_FULL_WEIGHT_PATH"
echo "[INFO] rt56_weight_dir=$RT56_WEIGHT_DIR"
echo "[INFO] zero_shot_output=$ZERO_SHOT_STEM"
echo "[INFO] fine_tune_epochs=$EPOCHS_FULL_MODEL"
echo "[INFO] protocol=50% rt55 + 50% causal random geometry for train; 100% causal random geometry for val"

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$ACTION" == "zero_shot" || "$ACTION" == "all" ]]; then
        printf '[DRY-RUN] WORKDIR=%q RUN_LOG_DIR=%q JOB_NAME=%q SLURM_TIME=%q EVAL_CHECKPOINT=%q EVAL_OUTPUT_TXT=%q EVAL_OUTPUT_NPZ=%q AUTO_SBATCH=1 bash %q %q --splits val --skip_single_station --skip_diagnostics\n' \
            "$WORKDIR" "$ZERO_SHOT_LOG_DIR" "$ZERO_SHOT_JOB_NAME" "$ZERO_SHOT_TIME" \
            "$RT55_EP32_CHECKPOINT" "$ZERO_SHOT_STEM.txt" "$ZERO_SHOT_STEM.npz" \
            "$EVAL_SCRIPT" "$CONFIG"
    fi
    if [[ "$ACTION" == "finetune" || "$ACTION" == "all" ]]; then
        printf '[DRY-RUN] WORKDIR=%q JOB_NAME=%q SLURM_NODES=%q SLURM_GPUS_PER_NODE=%q SLURM_CPUS_PER_TASK=%q SLURM_TIME=%q RUN_EVAL=0 RESET_WEIGHT_PATH=0 AUTO_SBATCH=1 bash %q %q --epochs_full_model %q\n' \
            "$WORKDIR" "$FINETUNE_JOB_NAME" "$FINETUNE_NODES" "$FINETUNE_GPUS_PER_NODE" \
            "$SLURM_CPUS_PER_TASK" "$FINETUNE_TIME" "$TRAIN_SCRIPT" "$CONFIG" "$EPOCHS_FULL_MODEL"
    fi
    echo "[OK] rt56 dry-run validation passed; no job submitted."
    exit 0
fi

if [[ "$ACTION" == "zero_shot" || "$ACTION" == "all" ]]; then
    WORKDIR="$WORKDIR" \
    RUN_LOG_DIR="$ZERO_SHOT_LOG_DIR" \
    JOB_NAME="$ZERO_SHOT_JOB_NAME" \
    SLURM_TIME="$ZERO_SHOT_TIME" \
    SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
    EVAL_CHECKPOINT="$RT55_EP32_CHECKPOINT" \
    EVAL_OUTPUT_TXT="$ZERO_SHOT_STEM.txt" \
    EVAL_OUTPUT_NPZ="$ZERO_SHOT_STEM.npz" \
    AUTO_SBATCH=1 \
    bash "$EVAL_SCRIPT" "$CONFIG" --splits val --skip_single_station --skip_diagnostics
fi

if [[ "$ACTION" == "finetune" || "$ACTION" == "all" ]]; then
    WORKDIR="$WORKDIR" \
    JOB_NAME="$FINETUNE_JOB_NAME" \
    SLURM_NODES="$FINETUNE_NODES" \
    SLURM_GPUS_PER_NODE="$FINETUNE_GPUS_PER_NODE" \
    SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
    SLURM_TIME="$FINETUNE_TIME" \
    RUN_EVAL=0 \
    RESET_WEIGHT_PATH=0 \
    AUTO_SBATCH=1 \
    bash "$TRAIN_SCRIPT" "$CONFIG" --epochs_full_model "$EPOCHS_FULL_MODEL"
fi

echo "[INFO] requested rt56 jobs submitted."
