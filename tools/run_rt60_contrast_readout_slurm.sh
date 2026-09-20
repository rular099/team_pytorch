#!/usr/bin/env bash

# RT60: one fixed eight-epoch readout-only train from the exact RT59 epoch-8
# checkpoint, then paired random/normal validation.  No test, roll, smoke or sweep.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics}
CONFIG=${CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_chaosuan.json}
NORMAL_CONFIG=${NORMAL_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_normal_validation_chaosuan.json}
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
RT58_BASE_CHECKPOINT=${RT58_BASE_CHECKPOINT:-$WORKDIR/$RT58_WEIGHT_PATH/full_model_last.pth}
RT59_PARENT_CHECKPOINT=${RT59_PARENT_CHECKPOINT:-}
RT59_PARENT_CHECKPOINT_SHA256=${RT59_PARENT_CHECKPOINT_SHA256:-}
RT60_WEIGHT_PATH=${RT60_WEIGHT_PATH:-$WORKDIR/weights_japan_full_2000_2024_rt60_contrast_readout_seed42}

ACTION=${ACTION:-all}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_RT60=${CONFIRM_RT60:-0}
SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-git}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-}
EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256:-}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}

TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-team-rt60-readout-train}
RANDOM_JOB_NAME=${RANDOM_JOB_NAME:-team-rt60-random-val}
NORMAL_JOB_NAME=${NORMAL_JOB_NAME:-team-rt60-normal-val}
TRAIN_TIME=${TRAIN_TIME:-23:50:00}
EVAL_TIME=${EVAL_TIME:-23:50:00}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
TRAIN_NODES=${TRAIN_NODES:-4}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
EVAL_GPUS=${EVAL_GPUS:-1}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}

case "$ACTION" in train|eval|all) ;; *) echo "ACTION must be train, eval, or all; got: $ACTION" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" != "1" && "$CONFIRM_RT60" != "1" ]]; then
    echo "RT60 submission requires CONFIRM_RT60=1 (or keep DRY_RUN=1)." >&2
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

# Complete transitive runtime/config manifest, including every inherited config.
SOURCE_MANIFEST_FILES=(
    gemini_models.py gemini_util_light.py loader_light.py train_light.py eval_checkpoint.py
    train_light_slurm.sh eval_checkpoint_slurm.sh
    tools/rt59_dual_objective.py tools/rt60_contrast_objective.py
    tools/analyze_rt59_dual_objective_npz.py tools/run_rt60_contrast_readout_slurm.sh
    pga_configs/transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_normal_validation_chaosuan.json
)
source_manifest_sha256() {
    local rel digest
    for rel in "${SOURCE_MANIFEST_FILES[@]}"; do
        [[ -f "$WORKDIR/$rel" ]] || { echo "missing:$rel"; continue; }
        digest=$(file_sha256 "$WORKDIR/$rel")
        echo "$rel:$digest"
    done | sha256sum | awk '{print $1}'
}

actual_manifest=$(source_manifest_sha256)
case "$SOURCE_IDENTITY_MODE" in
    git)
        git -C "$WORKDIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
            echo "Git mode requires a worktree; uploaded folders use SOURCE_IDENTITY_MODE=uploaded_sha256." >&2; exit 2;
        }
        actual_commit=$(git -C "$WORKDIR" rev-parse HEAD)
        dirty=$(git -C "$WORKDIR" status --porcelain)
        [[ -z "$dirty" ]] || { echo "RT60 requires a clean Git worktree:" >&2; printf '%s\n' "$dirty" >&2; exit 1; }
        if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_GIT_COMMIT" ]]; then EXPECTED_GIT_COMMIT=$actual_commit; fi
        [[ "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "EXPECTED_GIT_COMMIT must be a full reviewed commit." >&2; exit 2; }
        [[ "$actual_commit" == "$EXPECTED_GIT_COMMIT" ]] || { echo "Git commit mismatch: expected $EXPECTED_GIT_COMMIT, got $actual_commit" >&2; exit 1; }
        echo "[OK] clean Git source verified: $actual_commit"
        ;;
    uploaded_sha256)
        if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_SOURCE_MANIFEST_SHA256" ]]; then EXPECTED_SOURCE_MANIFEST_SHA256=$actual_manifest; fi
        [[ "$EXPECTED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "uploaded_sha256 mode requires EXPECTED_SOURCE_MANIFEST_SHA256." >&2; exit 2; }
        [[ "$actual_manifest" == "$EXPECTED_SOURCE_MANIFEST_SHA256" ]] || { echo "Uploaded source manifest mismatch: expected $EXPECTED_SOURCE_MANIFEST_SHA256, got $actual_manifest" >&2; exit 1; }
        echo "[OK] uploaded source manifest verified: $actual_manifest"
        ;;
    *) echo "SOURCE_IDENTITY_MODE must be git or uploaded_sha256." >&2; exit 2 ;;
esac
RT60_SOURCE_MANIFEST_SHA256=$actual_manifest

for required in "$CONFIG:RT60 config" "$NORMAL_CONFIG:RT60 normal config" "$TRAIN_SCRIPT:training launcher" "$EVAL_SCRIPT:evaluation launcher"; do
    require_file "${required%%:*}" "${required#*:}"
done

if [[ -z "$RT59_PARENT_CHECKPOINT" ]]; then
    echo "RT59_PARENT_CHECKPOINT must name the real RT59 epoch-8 full_model_last.pth; it is never guessed." >&2
    exit 2
fi
require_file "$RT59_PARENT_CHECKPOINT" "RT59 epoch-8 parent checkpoint"
if [[ -s "$RT59_PARENT_CHECKPOINT" ]]; then
    actual_parent_sha=$(file_sha256 "$RT59_PARENT_CHECKPOINT")
else
    actual_parent_sha='<computed before real submission>'
fi
if [[ "$DRY_RUN" == "1" && -z "$RT59_PARENT_CHECKPOINT_SHA256" && "$actual_parent_sha" != '<computed before real submission>' ]]; then
    RT59_PARENT_CHECKPOINT_SHA256=$actual_parent_sha
fi
if [[ "$DRY_RUN" != "1" ]]; then
    [[ "$RT59_PARENT_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "Real RT60 work requires RT59_PARENT_CHECKPOINT_SHA256." >&2; exit 2; }
    [[ "$actual_parent_sha" == "$RT59_PARENT_CHECKPOINT_SHA256" ]] || { echo "RT59 parent SHA-256 mismatch: expected $RT59_PARENT_CHECKPOINT_SHA256, got $actual_parent_sha" >&2; exit 1; }
fi

parent_config=$(dirname "$RT59_PARENT_CHECKPOINT")/config.json
require_file "$parent_config" "resolved RT59 parent config"
if [[ -s "$parent_config" ]]; then
    python - "$parent_config" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if int(c.get('training_params', {}).get('epochs_full_model', -1)) != 8:
    raise SystemExit('RT59 parent resolved config does not declare epochs_full_model=8')
if not c.get('model_params', {}).get('use_pga_anchor_residual_transport', False):
    raise SystemExit('RT59 parent resolved config lacks the RT59 transport head')
if 'RT59-v3' not in str(c.get('experiment_note', '')):
    raise SystemExit('RT59 parent task identity is not RT59-v3')
print('[OK] RT59 parent resolved config task/epoch contract verified')
PY
fi

missing_shard=0
for year in $(seq 2000 2024); do
    shard=$JAPAN_FULL_DATA_ROOT/$year/japan_${year}.hdf5
    if [[ ! -s "$shard" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then echo "[DRY-RUN WARN] annual shard is not visible: $shard" >&2; else echo "Missing annual shard: $shard" >&2; missing_shard=1; fi
    fi
done
[[ "$missing_shard" == "0" ]] || exit 1

case "$RT60_WEIGHT_PATH" in /*) RT60_WEIGHT_DIR=$RT60_WEIGHT_PATH ;; *) RT60_WEIGHT_DIR=$WORKDIR/${RT60_WEIGHT_PATH#./} ;; esac
if [[ -z "$RT60_WEIGHT_DIR" || "$RT60_WEIGHT_DIR" == "/" || "$RT60_WEIGHT_DIR" == "$WORKDIR" ]]; then echo "Unsafe RT60 weight directory: $RT60_WEIGHT_DIR" >&2; exit 1; fi
RT60_EPOCH8_CHECKPOINT=$RT60_WEIGHT_DIR/full_model_last.pth
RT60_LOG_NAME=$(basename "$RT60_WEIGHT_DIR")
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    if [[ -d "$RT60_WEIGHT_DIR" ]]; then
        entry_count=$(find "$RT60_WEIGHT_DIR" -maxdepth 1 -mindepth 1 -printf x | wc -c)
        [[ "$entry_count" == "0" ]] || { echo "RT60 training directory must be new and empty: $RT60_WEIGHT_DIR" >&2; exit 1; }
    fi
else
    require_file "$RT60_EPOCH8_CHECKPOINT" "RT60 epoch-8 checkpoint"
fi

RANDOM_LOG_DIR=${RANDOM_LOG_DIR:-$WORKDIR/logs/$RT60_LOG_NAME/epoch8_random_validation}
NORMAL_LOG_DIR=${NORMAL_LOG_DIR:-$WORKDIR/logs/$RT60_LOG_NAME/epoch8_normal_validation}
RANDOM_STEM=$RANDOM_LOG_DIR/eval_validation_epoch8_random
NORMAL_STEM=$NORMAL_LOG_DIR/eval_validation_epoch8_normal
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]] && [[ "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    for stem in "$RANDOM_STEM" "$NORMAL_STEM"; do
        [[ ! -s "$stem.npz" && ! -s "$stem.txt" && ! -s "$stem.metrics.json" ]] || { echo "Evaluation output exists; refusing overwrite: $stem" >&2; exit 1; }
    done
fi

check_active_job() {
    local job_name=$1 active
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]] || ! command -v squeue >/dev/null 2>&1; then return; fi
    active=$(squeue --noheader --user "$(id -un)" --name "$job_name" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    [[ -z "$active" ]] || { echo "Same-name job active: $job_name $active" >&2; exit 1; }
}
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then check_active_job "$TRAIN_JOB_NAME"; fi
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then check_active_job "$RANDOM_JOB_NAME"; check_active_job "$NORMAL_JOB_NAME"; fi

export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT55_EP32_CHECKPOINT
export RT56_WEIGHT_PATH RT56_BASE_CHECKPOINT RT57_WEIGHT_PATH RT57_BASE_CHECKPOINT
export RT58_WEIGHT_PATH RT58_BASE_CHECKPOINT RT59_PARENT_CHECKPOINT
export RT59_PARENT_CHECKPOINT_SHA256 RT60_WEIGHT_PATH RT60_SOURCE_MANIFEST_SHA256

echo "[INFO] action=$ACTION dry_run=$DRY_RUN source_mode=$SOURCE_IDENTITY_MODE"
echo "[INFO] source_manifest_sha256=$RT60_SOURCE_MANIFEST_SHA256"
echo "[INFO] RT59 parent=$RT59_PARENT_CHECKPOINT sha256=$actual_parent_sha"
echo "[INFO] RT60 checkpoint=$RT60_EPOCH8_CHECKPOINT"
echo "[INFO] protocol=seed42, 8 new epochs, 6 tensors/67077 scalars, train -> random/normal val; no test/smoke/sweep"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] no Slurm jobs submitted"
    echo "[DRY-RUN] train nodes=$TRAIN_NODES dcu_per_node=$TRAIN_GPUS_PER_NODE time=$TRAIN_TIME"
    echo "[DRY-RUN] random NPZ=$RANDOM_STEM.npz"
    echo "[DRY-RUN] normal NPZ=$NORMAL_STEM.npz"
    exit 0
fi

train_job_id=''
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
    submit=$(WORKDIR="$WORKDIR" JOB_NAME="$TRAIN_JOB_NAME" SLURM_PARTITION="$SLURM_PARTITION" \
        SLURM_NODES="$TRAIN_NODES" SLURM_GPUS_PER_NODE="$TRAIN_GPUS_PER_NODE" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" SLURM_TIME="$TRAIN_TIME" \
        EXPECTED_LOAD_CHECKPOINT="$RT59_PARENT_CHECKPOINT" EXPECTED_LOAD_CHECKPOINT_EPOCH=8 \
        RUN_EVAL=0 RESET_WEIGHT_PATH=0 AUTO_SBATCH=1 \
        bash "$TRAIN_SCRIPT" "$CONFIG" --epochs_full_model 8)
    printf '%s\n' "$submit"
    train_job_id=$(printf '%s\n' "$submit" | awk '/Submitted batch job/ {job=$NF} END {print job}')
    [[ "$train_job_id" =~ ^[0-9]+$ ]] || { echo "Could not parse RT60 training job ID; validation was not submitted." >&2; exit 1; }
fi

submit_eval() {
    local name=$1 config=$2 log_dir=$3 stem=$4 dependency=$5 output
    output=$(WORKDIR="$WORKDIR" JOB_NAME="$name" SLURM_PARTITION="$SLURM_PARTITION" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" SLURM_TIME="$EVAL_TIME" \
        SLURM_GPUS="$EVAL_GPUS" SLURM_GRES_RESOURCE="$SLURM_GRES_RESOURCE" \
        SLURM_DEPENDENCY="$dependency" EXPECTED_CHECKPOINT_EPOCH=8 \
        EVAL_PREFER_REQUESTED_CONFIG=1 RUN_LOG_DIR="$log_dir" \
        EVAL_CHECKPOINT="$RT60_EPOCH8_CHECKPOINT" EVAL_OUTPUT_TXT="$stem.txt" \
        EVAL_OUTPUT_NPZ="$stem.npz" AUTO_SBATCH=1 \
        bash "$EVAL_SCRIPT" "$config" --splits val --skip_single_station --skip_diagnostics)
    printf '%s\n' "$output" >&2
    printf '%s\n' "$output" | awk '/Submitted batch job/ {job=$NF} END {print job}'
}
if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    dependency=''; [[ -z "$train_job_id" ]] || dependency=afterok:$train_job_id
    random_job_id=$(submit_eval "$RANDOM_JOB_NAME" "$CONFIG" "$RANDOM_LOG_DIR" "$RANDOM_STEM" "$dependency")
    normal_job_id=$(submit_eval "$NORMAL_JOB_NAME" "$NORMAL_CONFIG" "$NORMAL_LOG_DIR" "$NORMAL_STEM" "$dependency")
    [[ "$random_job_id" =~ ^[0-9]+$ && "$normal_job_id" =~ ^[0-9]+$ ]] || { echo "Could not parse RT60 validation job IDs." >&2; exit 1; }
    echo "[OK] validation jobs: random=$random_job_id normal=$normal_job_id"
fi
echo "[OK] requested RT60 jobs submitted."
