#!/usr/bin/env bash

# RT59-v3: one fixed eight-epoch train from RT58 epoch 8, followed by the
# paired random-geometry and normal-geometry validation jobs.  The random job
# exports the fixed-context u/d roll control in its ordinary forward pass.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics}
CONFIG=${CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json}
NORMAL_CONFIG=${NORMAL_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_normal_validation_chaosuan.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$WORKDIR/train_light_slurm.sh}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}

JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT55_EP32_CHECKPOINT=${RT55_EP32_CHECKPOINT:-$WORKDIR/$JAPAN_FULL_WEIGHT_PATH/full_model_best_ep32.pth}
RT56_WEIGHT_PATH=${RT56_WEIGHT_PATH:-weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT56_BASE_CHECKPOINT=${RT56_BASE_CHECKPOINT:-$WORKDIR/$RT56_WEIGHT_PATH/full_model_best.pth}
RT57_WEIGHT_PATH=${RT57_WEIGHT_PATH:-weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42}
RT57_BASE_CHECKPOINT=${RT57_BASE_CHECKPOINT:-$WORKDIR/$RT57_WEIGHT_PATH/full_model_last.pth}
RT58_WEIGHT_PATH=${RT58_WEIGHT_PATH:-weights_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42}
case "$RT58_WEIGHT_PATH" in
    /*) RT58_WEIGHT_DIR=$RT58_WEIGHT_PATH ;;
    *) RT58_WEIGHT_DIR=$WORKDIR/${RT58_WEIGHT_PATH#./} ;;
esac
RT58_BASE_CHECKPOINT=${RT58_BASE_CHECKPOINT:-$RT58_WEIGHT_DIR/full_model_last.pth}
RT58_BASE_CHECKPOINT_SHA256=${RT58_BASE_CHECKPOINT_SHA256:-}
RT59_WEIGHT_PATH=${RT59_WEIGHT_PATH:-$WORKDIR/weights_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42}

ACTION=${ACTION:-all}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_RT59=${CONFIRM_RT59:-0}
SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-git}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-}
EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256:-}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}

TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-team-rt59-v3-train}
RANDOM_JOB_NAME=${RANDOM_JOB_NAME:-team-rt59-v3-random-val}
NORMAL_JOB_NAME=${NORMAL_JOB_NAME:-team-rt59-v3-normal-val}
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
if [[ "$DRY_RUN" != "1" && "$CONFIRM_RT59" != "1" ]]; then
    echo "RT59 submission requires CONFIRM_RT59=1 (or keep DRY_RUN=1)." >&2
    exit 2
fi

require_file() {
    local path=$1 label=$2
    if [[ -s "$path" ]]; then return 0; fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN WARN] $label is not visible: $path" >&2
        return 0
    fi
    echo "$label is missing or empty: $path" >&2
    exit 1
}

file_sha256() { sha256sum "$1" | awk '{print $1}'; }

SOURCE_MANIFEST_FILES=(
    gemini_models.py
    gemini_util_light.py
    loader_light.py
    train_light.py
    eval_checkpoint.py
    train_light_slurm.sh
    eval_checkpoint_slurm.sh
    tools/rt59_dual_objective.py
    tools/run_rt59_dual_objective_transport_v3_slurm.sh
    tools/analyze_rt59_dual_objective_npz.py
    pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_normal_validation_chaosuan.json
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
                echo "Git mode requires a worktree; uploaded folders use SOURCE_IDENTITY_MODE=uploaded_sha256." >&2
                exit 2
            fi
            local actual_commit dirty
            actual_commit=$(git -C "$WORKDIR" rev-parse HEAD)
            dirty=$(git -C "$WORKDIR" status --porcelain)
            if [[ -n "$dirty" ]]; then
                echo "RT59 requires a clean Git worktree:" >&2
                printf '%s\n' "$dirty" >&2
                exit 1
            fi
            if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_GIT_COMMIT" ]]; then
                EXPECTED_GIT_COMMIT=$actual_commit
            fi
            if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
                echo "EXPECTED_GIT_COMMIT must be a full reviewed 40-character commit." >&2
                exit 2
            fi
            [[ "$actual_commit" == "$EXPECTED_GIT_COMMIT" ]] || {
                echo "Git commit mismatch: expected $EXPECTED_GIT_COMMIT, got $actual_commit" >&2
                exit 1
            }
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
            [[ "$actual_manifest" == "$EXPECTED_SOURCE_MANIFEST_SHA256" ]] || {
                echo "Uploaded source manifest mismatch: expected $EXPECTED_SOURCE_MANIFEST_SHA256, got $actual_manifest" >&2
                exit 1
            }
            echo "[OK] uploaded source manifest verified: $actual_manifest"
            ;;
        *) echo "SOURCE_IDENTITY_MODE must be git or uploaded_sha256." >&2; exit 2 ;;
    esac
}

for required in \
    "$CONFIG:RT59 config" \
    "$NORMAL_CONFIG:RT59 normal config" \
    "$TRAIN_SCRIPT:training launcher" \
    "$EVAL_SCRIPT:evaluation launcher"; do
    require_file "${required%%:*}" "${required#*:}"
done
verify_source_identity

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    require_file "$RT58_BASE_CHECKPOINT" "RT58 epoch-8 checkpoint"
    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ -s "$RT58_BASE_CHECKPOINT" ]]; then
            actual_rt58_sha256=$(file_sha256 "$RT58_BASE_CHECKPOINT")
        else
            actual_rt58_sha256='<computed before real submission>'
        fi
    else
        if [[ ! "$RT58_BASE_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
            echo "Real training requires RT58_BASE_CHECKPOINT_SHA256." >&2
            exit 2
        fi
        actual_rt58_sha256=$(file_sha256 "$RT58_BASE_CHECKPOINT")
        [[ "$actual_rt58_sha256" == "$RT58_BASE_CHECKPOINT_SHA256" ]] || {
            echo "RT58 checkpoint SHA-256 mismatch: expected $RT58_BASE_CHECKPOINT_SHA256, got $actual_rt58_sha256" >&2
            exit 1
        }
    fi
else
    actual_rt58_sha256='<not required for ACTION=eval>'
fi

missing_shard=0
for year in $(seq 2000 2024); do
    shard=$JAPAN_FULL_DATA_ROOT/$year/japan_${year}.hdf5
    if [[ ! -s "$shard" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[DRY-RUN WARN] annual shard is not visible: $shard" >&2
        else
            echo "Missing annual shard: $shard" >&2
            missing_shard=1
        fi
    fi
done
[[ "$missing_shard" == "0" ]] || exit 1

case "$RT59_WEIGHT_PATH" in
    /*) RT59_WEIGHT_DIR=$RT59_WEIGHT_PATH ;;
    *) RT59_WEIGHT_DIR=$WORKDIR/${RT59_WEIGHT_PATH#./} ;;
esac
if [[ -z "$RT59_WEIGHT_DIR" || "$RT59_WEIGHT_DIR" == "/" || "$RT59_WEIGHT_DIR" == "$WORKDIR" ]]; then
    echo "Unsafe RT59 weight directory: $RT59_WEIGHT_DIR" >&2
    exit 1
fi
RT59_EPOCH8_CHECKPOINT=$RT59_WEIGHT_DIR/full_model_last.pth
RT59_LOG_NAME=$(basename "$RT59_WEIGHT_DIR")
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    if [[ -d "$RT59_WEIGHT_DIR" ]]; then
        entry_count=$(find "$RT59_WEIGHT_DIR" -maxdepth 1 -mindepth 1 -printf x | wc -c)
        [[ "$entry_count" == "0" ]] || {
            echo "RT59 training directory must be new and empty: $RT59_WEIGHT_DIR" >&2
            exit 1
        }
    fi
else
    require_file "$RT59_EPOCH8_CHECKPOINT" "RT59 epoch-8 checkpoint"
fi

RANDOM_LOG_DIR=${RANDOM_LOG_DIR:-$WORKDIR/logs/$RT59_LOG_NAME/epoch8_random_validation}
NORMAL_LOG_DIR=${NORMAL_LOG_DIR:-$WORKDIR/logs/$RT59_LOG_NAME/epoch8_normal_validation}
RANDOM_STEM=$RANDOM_LOG_DIR/eval_validation_epoch8_random
NORMAL_STEM=$NORMAL_LOG_DIR/eval_validation_epoch8_normal
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]] && [[ "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    for stem in "$RANDOM_STEM" "$NORMAL_STEM"; do
        if [[ -s "$stem.npz" || -s "$stem.txt" || -s "$stem.metrics.json" ]]; then
            echo "Evaluation output exists; refusing overwrite: $stem" >&2
            exit 1
        fi
    done
fi

check_active_job() {
    local job_name=$1
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]] || ! command -v squeue >/dev/null 2>&1; then return; fi
    local active
    active=$(squeue --noheader --user "$(id -un)" --name "$job_name" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    [[ -z "$active" ]] || { echo "Same-name job active: $job_name $active" >&2; exit 1; }
}
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then check_active_job "$TRAIN_JOB_NAME"; fi
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    check_active_job "$RANDOM_JOB_NAME"
    check_active_job "$NORMAL_JOB_NAME"
fi

export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT55_EP32_CHECKPOINT
export RT56_WEIGHT_PATH RT56_BASE_CHECKPOINT RT57_WEIGHT_PATH RT57_BASE_CHECKPOINT
export RT58_WEIGHT_PATH RT58_BASE_CHECKPOINT RT59_WEIGHT_PATH

echo "[INFO] action=$ACTION dry_run=$DRY_RUN source_mode=$SOURCE_IDENTITY_MODE"
echo "[INFO] RT58 base=$RT58_BASE_CHECKPOINT sha256=$actual_rt58_sha256"
echo "[INFO] RT59 checkpoint=$RT59_EPOCH8_CHECKPOINT"
echo "[INFO] protocol=seed42, exactly 8 new epochs, train -> random/normal val; no test"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] no Slurm jobs submitted"
    echo "[DRY-RUN] train nodes=$TRAIN_NODES dcu_per_node=$TRAIN_GPUS_PER_NODE time=$TRAIN_TIME"
    echo "[DRY-RUN] random NPZ=$RANDOM_STEM.npz (includes fixed-context feature roll)"
    echo "[DRY-RUN] normal NPZ=$NORMAL_STEM.npz"
    exit 0
fi

train_job_id=''
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    submit=$(WORKDIR="$WORKDIR" JOB_NAME="$TRAIN_JOB_NAME" \
        SLURM_PARTITION="$SLURM_PARTITION" SLURM_NODES="$TRAIN_NODES" \
        SLURM_GPUS_PER_NODE="$TRAIN_GPUS_PER_NODE" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" SLURM_TIME="$TRAIN_TIME" \
        EXPECTED_LOAD_CHECKPOINT="$RT58_BASE_CHECKPOINT" \
        EXPECTED_LOAD_CHECKPOINT_EPOCH=8 RUN_EVAL=0 RESET_WEIGHT_PATH=0 \
        AUTO_SBATCH=1 bash "$TRAIN_SCRIPT" "$CONFIG" --epochs_full_model 8)
    printf '%s\n' "$submit"
    train_job_id=$(printf '%s\n' "$submit" | awk '/Submitted batch job/ {job=$NF} END {print job}')
    [[ "$train_job_id" =~ ^[0-9]+$ ]] || {
        echo "Could not parse RT59 training job ID; validation was not submitted." >&2
        exit 1
    }
fi

submit_eval() {
    local name=$1 config=$2 log_dir=$3 stem=$4 dependency=$5 output
    output=$(WORKDIR="$WORKDIR" JOB_NAME="$name" \
        SLURM_PARTITION="$SLURM_PARTITION" SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        SLURM_TIME="$EVAL_TIME" SLURM_GPUS="$EVAL_GPUS" \
        SLURM_GRES_RESOURCE="$SLURM_GRES_RESOURCE" SLURM_DEPENDENCY="$dependency" \
        EXPECTED_CHECKPOINT_EPOCH=8 EVAL_PREFER_REQUESTED_CONFIG=1 \
        RUN_LOG_DIR="$log_dir" EVAL_CHECKPOINT="$RT59_EPOCH8_CHECKPOINT" \
        EVAL_OUTPUT_TXT="$stem.txt" EVAL_OUTPUT_NPZ="$stem.npz" AUTO_SBATCH=1 \
        bash "$EVAL_SCRIPT" "$config" --splits val --skip_single_station --skip_diagnostics)
    printf '%s\n' "$output" >&2
    printf '%s\n' "$output" | awk '/Submitted batch job/ {job=$NF} END {print job}'
}

if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    dependency=''
    [[ -z "$train_job_id" ]] || dependency=afterok:$train_job_id
    random_job_id=$(submit_eval "$RANDOM_JOB_NAME" "$CONFIG" "$RANDOM_LOG_DIR" "$RANDOM_STEM" "$dependency")
    normal_job_id=$(submit_eval "$NORMAL_JOB_NAME" "$NORMAL_CONFIG" "$NORMAL_LOG_DIR" "$NORMAL_STEM" "$dependency")
    [[ "$random_job_id" =~ ^[0-9]+$ && "$normal_job_id" =~ ^[0-9]+$ ]] || {
        echo "Could not parse RT59 validation job IDs." >&2
        exit 1
    }
    echo "[OK] validation jobs: random=$random_job_id normal=$normal_job_id"
fi
echo "[OK] requested RT59-v3 jobs submitted."
