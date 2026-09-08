#!/usr/bin/env bash

# RT57 GTNP v2: train a station-distinctive temporal residual head from a
# verified RT56 checkpoint, then evaluate the fixed epoch-6 checkpoint on
# deterministic random validation, normal validation, and the required
# waveform/station permutation control. No test split is touched.
#
# Dry run:
#   DRY_RUN=1 ACTION=all bash tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh
#
# Submit training plus dependent validation jobs:
#   CONFIRM_RT57=1 ACTION=all bash tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh
#
# Evaluate an already completed epoch-6 checkpoint:
#   CONFIRM_RT57=1 ACTION=eval bash tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG=${CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json}
NORMAL_CONFIG=${NORMAL_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_normal_validation_chaosuan.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$WORKDIR/train_light_slurm.sh}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}
JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT55_EP32_CHECKPOINT=${RT55_EP32_CHECKPOINT:-$WORKDIR/$JAPAN_FULL_WEIGHT_PATH/full_model_best_ep32.pth}
RT56_WEIGHT_PATH=${RT56_WEIGHT_PATH:-weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT56_BASE_CHECKPOINT=${RT56_BASE_CHECKPOINT:-$WORKDIR/$RT56_WEIGHT_PATH/full_model_best.pth}
RT56_BASE_CHECKPOINT_SHA256=${RT56_BASE_CHECKPOINT_SHA256:-}
RT57_WEIGHT_PATH=${RT57_WEIGHT_PATH:-weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42}
ACTION=${ACTION:-all}
EPOCHS_FULL_MODEL=${EPOCHS_FULL_MODEL:-6}
CONFIRM_RT57=${CONFIRM_RT57:-0}
DRY_RUN=${DRY_RUN:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}

TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-team-rt57-gtnp-v2-train}
RANDOM_JOB_NAME=${RANDOM_JOB_NAME:-team-rt57-gtnp-v2-random-val}
NORMAL_JOB_NAME=${NORMAL_JOB_NAME:-team-rt57-gtnp-v2-normal-val}
PERMUTATION_JOB_NAME=${PERMUTATION_JOB_NAME:-team-rt57-gtnp-v2-waveperm-val}
TRAIN_TIME=${TRAIN_TIME:-3-00:00:00}
EVAL_TIME=${EVAL_TIME:-1-00:00:00}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
TRAIN_NODES=${TRAIN_NODES:-4}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
EVAL_GPUS=${EVAL_GPUS:-1}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}

case "$ACTION" in
    train|eval|all) ;;
    *)
        echo "ACTION must be train, eval, or all; got: $ACTION" >&2
        exit 2
        ;;
esac
if [[ "$EPOCHS_FULL_MODEL" != "6" ]]; then
    echo "RT57 protocol requires exactly six epochs; got: $EPOCHS_FULL_MODEL" >&2
    exit 2
fi
if [[ -n "$RT56_BASE_CHECKPOINT_SHA256" && ! "$RT56_BASE_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "RT56_BASE_CHECKPOINT_SHA256 must be a full 64-character SHA-256; got: $RT56_BASE_CHECKPOINT_SHA256" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "1" && "$CONFIRM_RT57" != "1" ]]; then
    echo "RT57 submission requires CONFIRM_RT57=1." >&2
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

require_file "$CONFIG" "RT57 config"
require_file "$NORMAL_CONFIG" "RT57 normal-validation config"
require_file "$TRAIN_SCRIPT" "training launcher"
require_file "$EVAL_SCRIPT" "evaluation launcher"
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    require_file "$RT56_BASE_CHECKPOINT" "selected RT56 base checkpoint"
    if [[ "$DRY_RUN" != "1" ]]; then
        actual_rt56_sha256=$(sha256sum "$RT56_BASE_CHECKPOINT" | awk '{print $1}')
        if [[ -n "$RT56_BASE_CHECKPOINT_SHA256" && "$actual_rt56_sha256" != "$RT56_BASE_CHECKPOINT_SHA256" ]]; then
            echo "RT56 checkpoint SHA-256 mismatch: expected $RT56_BASE_CHECKPOINT_SHA256, got $actual_rt56_sha256" >&2
            exit 1
        fi
    else
        actual_rt56_sha256='<computed at submission>'
    fi
else
    actual_rt56_sha256='<not required for ACTION=eval>'
fi

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

case "$RT57_WEIGHT_PATH" in
    /*) RT57_WEIGHT_DIR=$RT57_WEIGHT_PATH ;;
    *) RT57_WEIGHT_DIR="$WORKDIR/${RT57_WEIGHT_PATH#./}" ;;
esac
if [[ -z "$RT57_WEIGHT_DIR" || "$RT57_WEIGHT_DIR" == "/" || "$RT57_WEIGHT_DIR" == "$WORKDIR" ]]; then
    echo "Unsafe RT57 weight directory: $RT57_WEIGHT_DIR" >&2
    exit 1
fi
RT57_EPOCH6_CHECKPOINT="$RT57_WEIGHT_DIR/full_model_last.pth"
RT57_WEIGHT_LOG_NAME=${RT57_WEIGHT_PATH#./}
case "$RT57_WEIGHT_LOG_NAME" in
    /*) RT57_WEIGHT_LOG_NAME=$(basename "$RT57_WEIGHT_LOG_NAME") ;;
esac

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    if [[ -d "$RT57_WEIGHT_DIR" ]]; then
        entry_count=$(find "$RT57_WEIGHT_DIR" -maxdepth 1 -mindepth 1 -printf x | wc -c)
        if ((entry_count > 0)); then
            echo "RT57 training weight directory must be new and empty: $RT57_WEIGHT_DIR" >&2
            exit 1
        fi
    fi
else
    require_file "$RT57_EPOCH6_CHECKPOINT" "RT57 epoch-6 checkpoint"
fi

RANDOM_LOG_DIR=${RANDOM_LOG_DIR:-$WORKDIR/logs/$RT57_WEIGHT_LOG_NAME/epoch6_random_validation}
NORMAL_LOG_DIR=${NORMAL_LOG_DIR:-$WORKDIR/logs/$RT57_WEIGHT_LOG_NAME/epoch6_normal_validation}
PERMUTATION_LOG_DIR=${PERMUTATION_LOG_DIR:-$WORKDIR/logs/$RT57_WEIGHT_LOG_NAME/epoch6_random_waveform_permutation}
RANDOM_STEM="$RANDOM_LOG_DIR/eval_validation_epoch6_random"
NORMAL_STEM="$NORMAL_LOG_DIR/eval_validation_epoch6_normal"
PERMUTATION_STEM="$PERMUTATION_LOG_DIR/eval_validation_epoch6_random_waveform_roll"

guard_output() {
    local stem=$1
    if [[ "$ALLOW_EXISTING_OUTPUT" == "1" ]]; then
        return 0
    fi
    if [[ -s "$stem.txt" || -s "$stem.npz" || -s "$stem.metrics.json" ]]; then
        echo "Evaluation output already exists; refusing to overwrite: $stem" >&2
        exit 1
    fi
}
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    guard_output "$RANDOM_STEM"
    guard_output "$NORMAL_STEM"
    guard_output "$PERMUTATION_STEM"
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
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    check_active_job "$TRAIN_JOB_NAME"
fi
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    check_active_job "$RANDOM_JOB_NAME"
    check_active_job "$NORMAL_JOB_NAME"
    check_active_job "$PERMUTATION_JOB_NAME"
fi

export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT55_EP32_CHECKPOINT
export RT56_WEIGHT_PATH RT56_BASE_CHECKPOINT RT57_WEIGHT_PATH

echo "[INFO] action=$ACTION"
echo "[INFO] source_rt56_checkpoint=$RT56_BASE_CHECKPOINT"
echo "[INFO] source_rt56_sha256=$actual_rt56_sha256"
echo "[INFO] rt57_epoch6_checkpoint=$RT57_EPOCH6_CHECKPOINT"
echo "[INFO] protocol=train 75% random geometry + 25% normal replay; validate random + normal only"
echo "[INFO] control=random validation with waveform station roll while coordinates stay fixed"
echo "[INFO] split policy=validation only; test is forbidden"

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
        printf '[DRY-RUN] WORKDIR=%q JOB_NAME=%q SLURM_PARTITION=%q SLURM_NODES=%q SLURM_GPUS_PER_NODE=%q SLURM_CPUS_PER_TASK=%q SLURM_TIME=%q RUN_EVAL=0 RESET_WEIGHT_PATH=0 AUTO_SBATCH=1 bash %q %q --epochs_full_model 6\n' \
            "$WORKDIR" "$TRAIN_JOB_NAME" "$SLURM_PARTITION" "$TRAIN_NODES" \
            "$TRAIN_GPUS_PER_NODE" "$SLURM_CPUS_PER_TASK" "$TRAIN_TIME" \
            "$TRAIN_SCRIPT" "$CONFIG"
    fi
    if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
        dependency_text='<afterok:TRAIN_JOB_ID for ACTION=all>'
        if [[ "$ACTION" == "eval" ]]; then
            dependency_text='<none>'
        fi
        printf '[DRY-RUN] three epoch-6 validation jobs: dependency=%s checkpoint=%q random_config=%q normal_config=%q outputs=(%q,%q,%q)\n' \
            "$dependency_text" "$RT57_EPOCH6_CHECKPOINT" "$CONFIG" "$NORMAL_CONFIG" \
            "$RANDOM_STEM.npz" "$NORMAL_STEM.npz" "$PERMUTATION_STEM.npz"
    fi
    echo "[OK] RT57 dry-run validation passed; no job submitted."
    exit 0
fi

train_job_id=''
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    train_submit_output=$( \
        WORKDIR="$WORKDIR" \
        JOB_NAME="$TRAIN_JOB_NAME" \
        SLURM_PARTITION="$SLURM_PARTITION" \
        SLURM_NODES="$TRAIN_NODES" \
        SLURM_GPUS_PER_NODE="$TRAIN_GPUS_PER_NODE" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        SLURM_TIME="$TRAIN_TIME" \
        RUN_EVAL=0 \
        RESET_WEIGHT_PATH=0 \
        AUTO_SBATCH=1 \
        bash "$TRAIN_SCRIPT" "$CONFIG" --epochs_full_model 6
    )
    printf '%s\n' "$train_submit_output"
    train_job_id=$(printf '%s\n' "$train_submit_output" | awk '/Submitted batch job/ {job=$NF} END {print job}')
    if [[ ! "$train_job_id" =~ ^[0-9]+$ ]]; then
        echo "Could not parse RT57 training job ID; dependent eval jobs were not submitted." >&2
        exit 1
    fi
    echo "[OK] RT57 training submitted as job $train_job_id"
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    dependency=''
    if [[ -n "$train_job_id" ]]; then
        dependency="afterok:$train_job_id"
    fi

    submit_eval() {
        local job_name=$1
        local config_path=$2
        local log_dir=$3
        local stem=$4
        local prefer_requested=$5
        shift 5
        local output
        output=$( \
            WORKDIR="$WORKDIR" \
            JOB_NAME="$job_name" \
            SLURM_PARTITION="$SLURM_PARTITION" \
            SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
            SLURM_TIME="$EVAL_TIME" \
            SLURM_GPUS="$EVAL_GPUS" \
            SLURM_GRES_RESOURCE="$SLURM_GRES_RESOURCE" \
            SLURM_DEPENDENCY="$dependency" \
            EXPECTED_CHECKPOINT_EPOCH=6 \
            EVAL_PREFER_REQUESTED_CONFIG="$prefer_requested" \
            RUN_LOG_DIR="$log_dir" \
            EVAL_CHECKPOINT="$RT57_EPOCH6_CHECKPOINT" \
            EVAL_OUTPUT_TXT="$stem.txt" \
            EVAL_OUTPUT_NPZ="$stem.npz" \
            AUTO_SBATCH=1 \
            bash "$EVAL_SCRIPT" "$config_path" \
                --splits val --skip_single_station --skip_diagnostics "$@"
        )
        printf '%s\n' "$output"
    }

    submit_eval "$RANDOM_JOB_NAME" "$CONFIG" "$RANDOM_LOG_DIR" "$RANDOM_STEM" 1
    submit_eval "$NORMAL_JOB_NAME" "$NORMAL_CONFIG" "$NORMAL_LOG_DIR" "$NORMAL_STEM" 1
    submit_eval "$PERMUTATION_JOB_NAME" "$CONFIG" "$PERMUTATION_LOG_DIR" "$PERMUTATION_STEM" 1 \
        --waveform_station_permutation roll
fi

echo "[OK] requested RT57 jobs submitted."
