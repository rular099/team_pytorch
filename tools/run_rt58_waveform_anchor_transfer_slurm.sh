#!/usr/bin/env bash

# RT58 Waveform-Anchor Transfer Field: one fixed eight-epoch training run from
# the verified RT57 epoch-6 full_model_last.pth, followed by validation only.
# DRY_RUN=1 is the safe default and submits no Slurm jobs.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics}
LEGACY_WORKDIR=${LEGACY_WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG=${CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json}
NORMAL_CONFIG=${NORMAL_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_normal_validation_chaosuan.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$WORKDIR/train_light_slurm.sh}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}

JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT55_EP32_CHECKPOINT=${RT55_EP32_CHECKPOINT:-$LEGACY_WORKDIR/$JAPAN_FULL_WEIGHT_PATH/full_model_best_ep32.pth}
RT56_WEIGHT_PATH=${RT56_WEIGHT_PATH:-weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT56_BASE_CHECKPOINT=${RT56_BASE_CHECKPOINT:-$LEGACY_WORKDIR/$RT56_WEIGHT_PATH/full_model_best.pth}
RT57_WEIGHT_PATH=${RT57_WEIGHT_PATH:-weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42}
RT57_BASE_CHECKPOINT=${RT57_BASE_CHECKPOINT:-$LEGACY_WORKDIR/$RT57_WEIGHT_PATH/full_model_last.pth}
RT57_BASE_CHECKPOINT_SHA256=${RT57_BASE_CHECKPOINT_SHA256:-}
RT58_WEIGHT_PATH=${RT58_WEIGHT_PATH:-$WORKDIR/weights_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42}

ACTION=${ACTION:-all}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_RT58=${CONFIRM_RT58:-0}
RUN_WAVEFORM_ROLL=${RUN_WAVEFORM_ROLL:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-git}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-}
EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256:-}
EPOCHS_FULL_MODEL=${EPOCHS_FULL_MODEL:-8}

TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-team-rt58-watf-train}
RANDOM_JOB_NAME=${RANDOM_JOB_NAME:-team-rt58-watf-random-val}
NORMAL_JOB_NAME=${NORMAL_JOB_NAME:-team-rt58-watf-normal-val}
ROLL_JOB_NAME=${ROLL_JOB_NAME:-team-rt58-watf-waveform-roll-val}
TRAIN_TIME=${TRAIN_TIME:-23:50:00}
EVAL_TIME=${EVAL_TIME:-23:50:00}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
TRAIN_NODES=${TRAIN_NODES:-4}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
EVAL_GPUS=${EVAL_GPUS:-1}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}

case "$ACTION" in
    train|eval|all) ;;
    *) echo "ACTION must be train, eval, or all; got: $ACTION" >&2; exit 2 ;;
esac
if [[ "$EPOCHS_FULL_MODEL" != "8" ]]; then
    echo "RT58 protocol requires exactly eight new epochs; got: $EPOCHS_FULL_MODEL" >&2
    exit 2
fi
if [[ "$RUN_WAVEFORM_ROLL" != "0" && "$RUN_WAVEFORM_ROLL" != "1" ]]; then
    echo "RUN_WAVEFORM_ROLL must be 0 or 1; got: $RUN_WAVEFORM_ROLL" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "1" && "$CONFIRM_RT58" != "1" ]]; then
    echo "RT58 submission requires CONFIRM_RT58=1 (or keep DRY_RUN=1)." >&2
    exit 2
fi

require_file() {
    local path=$1
    local label=$2
    if [[ -s "$path" ]]; then return 0; fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN WARN] $label is not visible: $path" >&2
        return 0
    fi
    echo "$label is missing or empty: $path" >&2
    exit 1
}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

SOURCE_MANIFEST_FILES=(
    gemini_models.py
    train_light.py
    eval_checkpoint.py
    train_light_slurm.sh
    eval_checkpoint_slurm.sh
    pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_normal_validation_chaosuan.json
    tools/run_rt58_waveform_anchor_transfer_slurm.sh
)

source_manifest_sha256() {
    local rel digest
    for rel in "${SOURCE_MANIFEST_FILES[@]}"; do
        if [[ ! -f "$WORKDIR/$rel" ]]; then
            echo "missing:$rel"
            continue
        fi
        digest=$(file_sha256 "$WORKDIR/$rel")
        echo "$rel:$digest"
    done | sha256sum | awk '{print $1}'
}

verify_source_identity() {
    case "$SOURCE_IDENTITY_MODE" in
        git)
            if ! git -C "$WORKDIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
                echo "SOURCE_IDENTITY_MODE=git requires a Git worktree at $WORKDIR." >&2
                echo "For a manually uploaded folder, use SOURCE_IDENTITY_MODE=uploaded_sha256." >&2
                exit 2
            fi
            local actual_commit dirty
            actual_commit=$(git -C "$WORKDIR" rev-parse HEAD)
            dirty=$(git -C "$WORKDIR" status --porcelain)
            if [[ -n "$dirty" ]]; then
                echo "RT58 requires a clean Git worktree; found:" >&2
                printf '%s\n' "$dirty" >&2
                exit 1
            fi
            if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_GIT_COMMIT" ]]; then
                EXPECTED_GIT_COMMIT=$actual_commit
            fi
            if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
                echo "EXPECTED_GIT_COMMIT must be the reviewed full 40-character commit." >&2
                exit 2
            fi
            if [[ "$actual_commit" != "$EXPECTED_GIT_COMMIT" ]]; then
                echo "Git commit mismatch: expected $EXPECTED_GIT_COMMIT, got $actual_commit" >&2
                exit 1
            fi
            echo "[OK] clean Git source verified: $actual_commit"
            ;;
        uploaded_sha256)
            local actual_manifest
            actual_manifest=$(source_manifest_sha256)
            if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_SOURCE_MANIFEST_SHA256" ]]; then
                EXPECTED_SOURCE_MANIFEST_SHA256=$actual_manifest
            fi
            if [[ ! "$EXPECTED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
                echo "uploaded_sha256 mode requires EXPECTED_SOURCE_MANIFEST_SHA256." >&2
                exit 2
            fi
            if [[ "$actual_manifest" != "$EXPECTED_SOURCE_MANIFEST_SHA256" ]]; then
                echo "Uploaded source manifest mismatch: expected $EXPECTED_SOURCE_MANIFEST_SHA256, got $actual_manifest" >&2
                exit 1
            fi
            echo "[OK] uploaded source manifest verified: $actual_manifest"
            ;;
        *)
            echo "SOURCE_IDENTITY_MODE must be git or uploaded_sha256; got: $SOURCE_IDENTITY_MODE" >&2
            exit 2
            ;;
    esac
}

for required_spec in \
    "$CONFIG:RT58 config" \
    "$NORMAL_CONFIG:RT58 normal-validation config" \
    "$TRAIN_SCRIPT:training launcher" \
    "$EVAL_SCRIPT:evaluation launcher"; do
    require_file "${required_spec%%:*}" "${required_spec#*:}"
done
verify_source_identity

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    require_file "$RT57_BASE_CHECKPOINT" "RT57 epoch-6 base checkpoint"
    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ -s "$RT57_BASE_CHECKPOINT" ]]; then
            actual_rt57_sha256=$(file_sha256 "$RT57_BASE_CHECKPOINT")
        else
            actual_rt57_sha256='<computed before real submission>'
        fi
    else
        if [[ ! "$RT57_BASE_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
            echo "Real training requires RT57_BASE_CHECKPOINT_SHA256." >&2
            exit 2
        fi
        actual_rt57_sha256=$(file_sha256 "$RT57_BASE_CHECKPOINT")
        if [[ "$actual_rt57_sha256" != "$RT57_BASE_CHECKPOINT_SHA256" ]]; then
            echo "RT57 checkpoint SHA-256 mismatch: expected $RT57_BASE_CHECKPOINT_SHA256, got $actual_rt57_sha256" >&2
            exit 1
        fi
    fi
else
    actual_rt57_sha256='<not required for ACTION=eval>'
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
if [[ "$missing_shard" == "1" ]]; then exit 1; fi

case "$RT58_WEIGHT_PATH" in
    /*) RT58_WEIGHT_DIR=$RT58_WEIGHT_PATH ;;
    *) RT58_WEIGHT_DIR=$WORKDIR/${RT58_WEIGHT_PATH#./} ;;
esac
if [[ -z "$RT58_WEIGHT_DIR" || "$RT58_WEIGHT_DIR" == "/" || "$RT58_WEIGHT_DIR" == "$WORKDIR" ]]; then
    echo "Unsafe RT58 weight directory: $RT58_WEIGHT_DIR" >&2
    exit 1
fi
RT58_EPOCH8_CHECKPOINT="$RT58_WEIGHT_DIR/full_model_last.pth"
RT58_LOG_NAME=$(basename "$RT58_WEIGHT_DIR")

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    if [[ -d "$RT58_WEIGHT_DIR" ]]; then
        entry_count=$(find "$RT58_WEIGHT_DIR" -maxdepth 1 -mindepth 1 -printf x | wc -c)
        if ((entry_count > 0)); then
            echo "RT58 training weight directory must be new and empty: $RT58_WEIGHT_DIR" >&2
            exit 1
        fi
    fi
else
    require_file "$RT58_EPOCH8_CHECKPOINT" "RT58 epoch-8 full_model_last.pth"
fi

RANDOM_LOG_DIR=${RANDOM_LOG_DIR:-$WORKDIR/logs/$RT58_LOG_NAME/epoch8_random_validation}
NORMAL_LOG_DIR=${NORMAL_LOG_DIR:-$WORKDIR/logs/$RT58_LOG_NAME/epoch8_normal_validation}
ROLL_LOG_DIR=${ROLL_LOG_DIR:-$WORKDIR/logs/$RT58_LOG_NAME/epoch8_random_waveform_roll}
RANDOM_STEM=$RANDOM_LOG_DIR/eval_validation_epoch8_random
NORMAL_STEM=$NORMAL_LOG_DIR/eval_validation_epoch8_normal
ROLL_STEM=$ROLL_LOG_DIR/eval_validation_epoch8_random_waveform_roll

guard_output() {
    local stem=$1
    if [[ "$ALLOW_EXISTING_OUTPUT" == "1" ]]; then return 0; fi
    if [[ -s "$stem.txt" || -s "$stem.npz" || -s "$stem.metrics.json" ]]; then
        echo "Evaluation output already exists; refusing to overwrite: $stem" >&2
        exit 1
    fi
}
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    guard_output "$RANDOM_STEM"
    guard_output "$NORMAL_STEM"
    if [[ "$RUN_WAVEFORM_ROLL" == "1" ]]; then guard_output "$ROLL_STEM"; fi
fi

check_active_job() {
    local job_name=$1
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]] || ! command -v squeue >/dev/null 2>&1; then return 0; fi
    local active
    active=$(squeue --noheader --user "$(id -un)" --name "$job_name" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    if [[ -n "$active" ]]; then
        echo "A same-name Slurm job is already active: $job_name $active" >&2
        exit 1
    fi
}
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then check_active_job "$TRAIN_JOB_NAME"; fi
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    check_active_job "$RANDOM_JOB_NAME"
    check_active_job "$NORMAL_JOB_NAME"
    if [[ "$RUN_WAVEFORM_ROLL" == "1" ]]; then check_active_job "$ROLL_JOB_NAME"; fi
fi

export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT55_EP32_CHECKPOINT
export RT56_WEIGHT_PATH RT56_BASE_CHECKPOINT RT57_WEIGHT_PATH RT57_BASE_CHECKPOINT RT58_WEIGHT_PATH

echo "[INFO] action=$ACTION dry_run=$DRY_RUN source_mode=$SOURCE_IDENTITY_MODE"
echo "[INFO] source_rt57_checkpoint=$RT57_BASE_CHECKPOINT"
echo "[INFO] source_rt57_sha256=$actual_rt57_sha256 (epoch metadata checked on compute node)"
echo "[INFO] official_rt58_checkpoint=$RT58_EPOCH8_CHECKPOINT"
echo "[INFO] protocol=one seed-42 run, exactly 8 epochs; fixed random + normal validation"
echo "[INFO] split policy=validation only; test is forbidden"

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
        printf '[DRY-RUN] one training job: WORKDIR=%q nodes=%q dcu_per_node=%q time=%q checkpoint=%q expected_epoch=6 bash %q %q --epochs_full_model 8\n' \
            "$WORKDIR" "$TRAIN_NODES" "$TRAIN_GPUS_PER_NODE" "$TRAIN_TIME" \
            "$RT57_BASE_CHECKPOINT" "$TRAIN_SCRIPT" "$CONFIG"
    fi
    if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
        printf '[DRY-RUN] two primary validation jobs: checkpoint=%q expected_epoch=8 random=%q normal=%q\n' \
            "$RT58_EPOCH8_CHECKPOINT" "$RANDOM_STEM.npz" "$NORMAL_STEM.npz"
        if [[ "$RUN_WAVEFORM_ROLL" == "1" ]]; then
            printf '[DRY-RUN] optional waveform-roll validation after both primary jobs: %q\n' "$ROLL_STEM.npz"
        fi
    fi
    echo "[OK] RT58 dry-run validation passed; no job submitted."
    exit 0
fi

train_job_id=''
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    train_submit=$( \
        WORKDIR="$WORKDIR" \
        JOB_NAME="$TRAIN_JOB_NAME" \
        SLURM_PARTITION="$SLURM_PARTITION" \
        SLURM_NODES="$TRAIN_NODES" \
        SLURM_GPUS_PER_NODE="$TRAIN_GPUS_PER_NODE" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        SLURM_TIME="$TRAIN_TIME" \
        EXPECTED_LOAD_CHECKPOINT="$RT57_BASE_CHECKPOINT" \
        EXPECTED_LOAD_CHECKPOINT_EPOCH=6 \
        RUN_EVAL=0 RESET_WEIGHT_PATH=0 AUTO_SBATCH=1 \
        bash "$TRAIN_SCRIPT" "$CONFIG" --epochs_full_model 8
    )
    printf '%s\n' "$train_submit"
    train_job_id=$(printf '%s\n' "$train_submit" | awk '/Submitted batch job/ {job=$NF} END {print job}')
    if [[ ! "$train_job_id" =~ ^[0-9]+$ ]]; then
        echo "Could not parse RT58 training job ID; validation jobs were not submitted." >&2
        exit 1
    fi
fi

submit_eval() {
    local job_name=$1 config_path=$2 log_dir=$3 stem=$4 dependency=$5
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
        EXPECTED_CHECKPOINT_EPOCH=8 \
        EVAL_PREFER_REQUESTED_CONFIG=1 \
        RUN_LOG_DIR="$log_dir" \
        EVAL_CHECKPOINT="$RT58_EPOCH8_CHECKPOINT" \
        EVAL_OUTPUT_TXT="$stem.txt" \
        EVAL_OUTPUT_NPZ="$stem.npz" \
        AUTO_SBATCH=1 \
        bash "$EVAL_SCRIPT" "$config_path" \
            --splits val --skip_single_station --skip_diagnostics "$@"
    )
    printf '%s\n' "$output" >&2
    printf '%s\n' "$output" | awk '/Submitted batch job/ {job=$NF} END {print job}'
}

if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    primary_dependency=''
    if [[ -n "$train_job_id" ]]; then primary_dependency="afterok:$train_job_id"; fi
    random_job_id=$(submit_eval "$RANDOM_JOB_NAME" "$CONFIG" "$RANDOM_LOG_DIR" "$RANDOM_STEM" "$primary_dependency")
    normal_job_id=$(submit_eval "$NORMAL_JOB_NAME" "$NORMAL_CONFIG" "$NORMAL_LOG_DIR" "$NORMAL_STEM" "$primary_dependency")
    if [[ ! "$random_job_id" =~ ^[0-9]+$ || ! "$normal_job_id" =~ ^[0-9]+$ ]]; then
        echo "Could not parse primary RT58 validation job IDs." >&2
        exit 1
    fi
    if [[ "$RUN_WAVEFORM_ROLL" == "1" ]]; then
        roll_dependency="afterok:$random_job_id:$normal_job_id"
        roll_job_id=$(submit_eval "$ROLL_JOB_NAME" "$CONFIG" "$ROLL_LOG_DIR" "$ROLL_STEM" "$roll_dependency" \
            --waveform_station_permutation roll)
        if [[ ! "$roll_job_id" =~ ^[0-9]+$ ]]; then
            echo "Could not parse RT58 waveform-roll job ID." >&2
            exit 1
        fi
    fi
fi

echo "[OK] requested RT58 jobs submitted."
