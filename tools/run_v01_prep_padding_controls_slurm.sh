#!/usr/bin/env bash

# V01 controlled velocity/P-prefix experiment.  The user submits this script;
# it never evaluates test data and defaults to a no-submit dry run.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics}
ACC_DATA_ROOT=${ACC_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
VELOCITY_DATA_ROOT=${VELOCITY_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/hinet_data}
FROZEN_SPLIT_MANIFEST=${FROZEN_SPLIT_MANIFEST:-$WORKDIR/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/split_events.csv}
RT55_EP32_CHECKPOINT=${RT55_EP32_CHECKPOINT:-$WORKDIR/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/full_model_best_ep32.pth}
V01_RUN_ROOT=${V01_RUN_ROOT:-$WORKDIR/v01_velocity_prep_padding_seed42}
V01_CACHE_ROOT=${V01_CACHE_ROOT:-$V01_RUN_ROOT/derived_cache}
V01_VFULL_WEIGHT_PATH=${V01_VFULL_WEIGHT_PATH:-$V01_RUN_ROOT/weights_vfull}
V01_VMISSING_WEIGHT_PATH=${V01_VMISSING_WEIGHT_PATH:-$V01_RUN_ROOT/weights_vmissing}
V01_APAIR_WEIGHT_PATH=${V01_APAIR_WEIGHT_PATH:-$V01_RUN_ROOT/weights_apair}
V01_EVAL_ROOT=${V01_EVAL_ROOT:-$V01_RUN_ROOT/eval}
V01_REPORT_ROOT=${V01_REPORT_ROOT:-$V01_RUN_ROOT/report}

TRAIN_SCRIPT=${TRAIN_SCRIPT:-$WORKDIR/train_light_slurm.sh}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}
PREFLIGHT_TOOL=${PREFLIGHT_TOOL:-$WORKDIR/tools/build_v01_paired_manifest.py}
ANALYZE_TOOL=${ANALYZE_TOOL:-$WORKDIR/tools/analyze_v01_padding_controls.py}
VFULL_CONFIG=${VFULL_CONFIG:-$WORKDIR/pga_configs/v01_velocity_full.json}
VMISSING_CONFIG=${VMISSING_CONFIG:-$WORKDIR/pga_configs/v01_velocity_missing.json}
APAIR_CONFIG=${APAIR_CONFIG:-$WORKDIR/pga_configs/v01_acc_pair.json}
NORMAL_CONFIG=${NORMAL_CONFIG:-$WORKDIR/pga_configs/v01_validation_normal.json}
RANDOM_CONFIG=${RANDOM_CONFIG:-$WORKDIR/pga_configs/v01_validation_random.json}

ACTION=${ACTION:-all}
ARMS=${ARMS:-vfull,vmissing,apair}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_V01=${CONFIRM_V01:-0}
ALLOW_PARTIAL_SNAPSHOT=${ALLOW_PARTIAL_SNAPSHOT:-1}
SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-git}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-}
EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256:-}
RESUME_V01=${RESUME_V01:-0}
ALLOW_EXISTING_EVAL=${ALLOW_EXISTING_EVAL:-0}

SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-102400M}
PREFLIGHT_CPUS=${PREFLIGHT_CPUS:-16}
PREFLIGHT_MEM=${PREFLIGHT_MEM:-192000M}
PREFLIGHT_TIME=${PREFLIGHT_TIME:-23:50:00}
TRAIN_NODES=${TRAIN_NODES:-4}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
TRAIN_TIME=${TRAIN_TIME:-23:50:00}
EVAL_GPUS=${EVAL_GPUS:-1}
EVAL_TIME=${EVAL_TIME:-12:00:00}
ANALYZE_TIME=${ANALYZE_TIME:-02:00:00}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
DITING_CONFIG=${DITING_CONFIG:-$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}

case "$ACTION" in preflight|train|eval|analyze|all) ;; *) echo "ACTION must be preflight, train, eval, analyze, or all; got $ACTION" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" != "1" && "$CONFIRM_V01" != "1" ]]; then
    echo "Formal V01 submission requires CONFIRM_V01=1 DRY_RUN=0." >&2
    exit 2
fi

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
declare -A ARM_ENABLED=([vfull]=0 [vmissing]=0 [apair]=0)
for arm in "${ARM_LIST[@]}"; do
    arm=${arm//[[:space:]]/}
    [[ -v "ARM_ENABLED[$arm]" ]] || { echo "Unknown arm: $arm" >&2; exit 2; }
    ARM_ENABLED[$arm]=1
done

require_file() {
    local path=$1 label=$2
    if [[ -s "$path" ]]; then return; fi
    if [[ "$DRY_RUN" == "1" ]]; then echo "[DRY-RUN WARN] $label not visible: $path" >&2; return; fi
    echo "$label missing or empty: $path" >&2; exit 1
}
file_sha256() { sha256sum "$1" | awk '{print $1}'; }

SOURCE_MANIFEST_FILES=(
    gemini_models.py gemini_util_light.py loader_light.py train_light.py eval_checkpoint.py
    train_light_slurm.sh eval_checkpoint_slurm.sh tools/hinet_raw_archive.py
    tools/velocity_waveform_backend.py tools/prep_padding_protocol.py
    tools/build_v01_paired_manifest.py tools/analyze_v01_padding_controls.py
    tools/run_v01_prep_padding_controls_slurm.sh
    pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
    pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
    pga_configs/v01_common_rt55_model.json pga_configs/v01_velocity_full.json
    pga_configs/v01_velocity_missing.json pga_configs/v01_acc_pair.json
    pga_configs/v01_validation_normal.json pga_configs/v01_validation_random.json
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
        if git -C "$WORKDIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            actual_commit=$(git -C "$WORKDIR" rev-parse HEAD)
            dirty=$(git -C "$WORKDIR" status --porcelain)
            if [[ -n "$dirty" && "$DRY_RUN" != "1" ]]; then
                echo "V01 production submission requires a clean worktree:" >&2; printf '%s\n' "$dirty" >&2; exit 1
            fi
            if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_GIT_COMMIT" ]]; then EXPECTED_GIT_COMMIT=$actual_commit; fi
            [[ "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "EXPECTED_GIT_COMMIT must be a full commit." >&2; exit 2; }
            [[ "$actual_commit" == "$EXPECTED_GIT_COMMIT" ]] || { echo "Git commit mismatch." >&2; exit 1; }
        else
            echo "Uploaded folders must set SOURCE_IDENTITY_MODE=uploaded_sha256." >&2; exit 2
        fi
        ;;
    uploaded_sha256)
        if [[ "$DRY_RUN" == "1" && -z "$EXPECTED_SOURCE_MANIFEST_SHA256" ]]; then EXPECTED_SOURCE_MANIFEST_SHA256=$actual_manifest; fi
        [[ "$EXPECTED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "uploaded_sha256 mode requires EXPECTED_SOURCE_MANIFEST_SHA256." >&2; exit 2; }
        [[ "$actual_manifest" == "$EXPECTED_SOURCE_MANIFEST_SHA256" ]] || { echo "Uploaded source manifest mismatch." >&2; exit 1; }
        ;;
    *) echo "SOURCE_IDENTITY_MODE must be git or uploaded_sha256." >&2; exit 2 ;;
esac

for required in \
    "$TRAIN_SCRIPT:training launcher" "$EVAL_SCRIPT:evaluation launcher" \
    "$PREFLIGHT_TOOL:preflight tool" "$ANALYZE_TOOL:analysis tool" \
    "$VFULL_CONFIG:vfull config" "$VMISSING_CONFIG:vmissing config" \
    "$APAIR_CONFIG:apair config" "$NORMAL_CONFIG:normal config" \
    "$RANDOM_CONFIG:random config" "$FROZEN_SPLIT_MANIFEST:frozen split" \
    "$RT55_EP32_CHECKPOINT:RT55 epoch-32 checkpoint"; do
    require_file "${required%%:*}" "${required#*:}"
done

if [[ "$DRY_RUN" != "1" ]]; then
    [[ -d "$ACC_DATA_ROOT" ]] || { echo "ACC_DATA_ROOT not found: $ACC_DATA_ROOT" >&2; exit 1; }
    [[ -d "$VELOCITY_DATA_ROOT" ]] || { echo "VELOCITY_DATA_ROOT not found: $VELOCITY_DATA_ROOT" >&2; exit 1; }
fi

for arm in vfull vmissing apair; do
    [[ "${ARM_ENABLED[$arm]}" == 1 ]] || continue
    case "$arm" in
        vfull) weight=$V01_VFULL_WEIGHT_PATH ;;
        vmissing) weight=$V01_VMISSING_WEIGHT_PATH ;;
        apair) weight=$V01_APAIR_WEIGHT_PATH ;;
    esac
    if [[ "$ACTION" =~ ^(train|all)$ && -d "$weight" ]] && find "$weight" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        if [[ "$RESUME_V01" != "1" ]]; then
            echo "Refusing to overwrite non-empty arm output: $weight (set RESUME_V01=1 for an explicit resume)" >&2; exit 1
        fi
        require_file "$weight/full_model_last.pth" "$arm resume checkpoint"
        require_file "$weight/config.json" "$arm resolved resume config"
    fi
done

COMMON_EXPORT="WORKDIR=$WORKDIR,ACC_DATA_ROOT=$ACC_DATA_ROOT,VELOCITY_DATA_ROOT=$VELOCITY_DATA_ROOT,FROZEN_SPLIT_MANIFEST=$FROZEN_SPLIT_MANIFEST,RT55_EP32_CHECKPOINT=$RT55_EP32_CHECKPOINT,V01_RUN_ROOT=$V01_RUN_ROOT,V01_CACHE_ROOT=$V01_CACHE_ROOT,V01_VFULL_WEIGHT_PATH=$V01_VFULL_WEIGHT_PATH,V01_VMISSING_WEIGHT_PATH=$V01_VMISSING_WEIGHT_PATH,V01_APAIR_WEIGHT_PATH=$V01_APAIR_WEIGHT_PATH,CONDA_ENV=$CONDA_ENV,MODULE_UNLOAD=$MODULE_UNLOAD,MODULE_LOADS=$MODULE_LOADS,DITING_CONFIG=$DITING_CONFIG,DITING_PRETRAINED=$DITING_PRETRAINED"

SBATCH_BASE=(sbatch --parsable --partition="$SLURM_PARTITION" --chdir="$WORKDIR")
if [[ -n "$SLURM_ACCOUNT" ]]; then SBATCH_BASE+=(--account="$SLURM_ACCOUNT"); fi
mkdir_command="mkdir -p '$V01_RUN_ROOT/logs' '$V01_EVAL_ROOT' '$V01_REPORT_ROOT'"
if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$V01_RUN_ROOT/logs" "$V01_EVAL_ROOT" "$V01_REPORT_ROOT"
fi

submit_or_print() {
    local dependency=$1; shift
    local -a command=("${SBATCH_BASE[@]}")
    if [[ -n "$dependency" ]]; then command+=(--dependency="afterok:$dependency"); fi
    command+=("$@")
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[DRY-RUN] ' >&2
        printf '%q ' "${command[@]}" >&2
        printf '\n' >&2
        echo "DRYJOB$RANDOM"
    else
        "${command[@]}"
    fi
}

preflight_job=""
if [[ "$ACTION" == preflight || "$ACTION" == all ]]; then
    partial_arg=""; [[ "$ALLOW_PARTIAL_SNAPSHOT" == 1 ]] && partial_arg="--allow-partial-snapshot"
    preflight_command="$mkdir_command && python '$PREFLIGHT_TOOL' --acc-root '$ACC_DATA_ROOT' --velocity-root '$VELOCITY_DATA_ROOT' --split-manifest '$FROZEN_SPLIT_MANIFEST' --output-root '$V01_CACHE_ROOT' $partial_arg"
    preflight_job=$(submit_or_print "" \
        --job-name=v01-preflight --nodes=1 --ntasks=1 --cpus-per-task="$PREFLIGHT_CPUS" \
        --mem="$PREFLIGHT_MEM" --time="$PREFLIGHT_TIME" \
        --output="$V01_RUN_ROOT/logs/%x-%j.out" --error="$V01_RUN_ROOT/logs/%x-%j.err" \
        --export="ALL,$COMMON_EXPORT" --wrap="$preflight_command" | tail -n 1)
    echo "[V01] preflight_job=$preflight_job"
fi

declare -A TRAIN_JOB
if [[ "$ACTION" == train || "$ACTION" == all ]]; then
    for arm in vfull vmissing apair; do
        [[ "${ARM_ENABLED[$arm]}" == 1 ]] || continue
        case "$arm" in
            vfull) config=$VFULL_CONFIG ;;
            vmissing) config=$VMISSING_CONFIG ;;
            apair) config=$APAIR_CONFIG ;;
        esac
        resume_arg=""; [[ "$RESUME_V01" == 1 ]] && resume_arg="--resume_full_model last"
        train_command="$mkdir_command && AUTO_SBATCH=0 RUN_EVAL=0 RESET_WEIGHT_PATH=0 bash '$TRAIN_SCRIPT' '$config' --epochs_full_model 8 $resume_arg"
        TRAIN_JOB[$arm]=$(submit_or_print "$preflight_job" \
            --job-name="v01-$arm-train" --nodes="$TRAIN_NODES" \
            --ntasks-per-node="$TRAIN_GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            --gres="$SLURM_GRES_RESOURCE:$TRAIN_GPUS_PER_NODE" --mem="$SLURM_MEM" --time="$TRAIN_TIME" \
            --output="$V01_RUN_ROOT/logs/%x-%j.out" --error="$V01_RUN_ROOT/logs/%x-%j.err" \
            --export="ALL,$COMMON_EXPORT,SLURM_GPUS_PER_NODE=$TRAIN_GPUS_PER_NODE" \
            --wrap="$train_command" | tail -n 1)
        echo "[V01] ${arm}_train_job=${TRAIN_JOB[$arm]}"
    done
fi

declare -a EVAL_JOBS=()
if [[ "$ACTION" == eval || "$ACTION" == all ]]; then
    for checkpoint_arm in vfull vmissing apair; do
        [[ "${ARM_ENABLED[$checkpoint_arm]}" == 1 ]] || continue
        case "$checkpoint_arm" in
            vfull) checkpoint=$V01_VFULL_WEIGHT_PATH/full_model_last.pth ;;
            vmissing) checkpoint=$V01_VMISSING_WEIGHT_PATH/full_model_last.pth ;;
            apair) checkpoint=$V01_APAIR_WEIGHT_PATH/full_model_last.pth ;;
        esac
        views=("$checkpoint_arm")
        if [[ "$checkpoint_arm" == vfull || "$checkpoint_arm" == vmissing ]]; then views=(vfull vmissing); fi
        for view in "${views[@]}"; do
            case "$view" in
                vfull) arm_config=$VFULL_CONFIG ;;
                vmissing) arm_config=$VMISSING_CONFIG ;;
                apair) arm_config=$APAIR_CONFIG ;;
            esac
            for protocol in normal random; do
                [[ "$protocol" == normal ]] && eval_config=$NORMAL_CONFIG || eval_config=$RANDOM_CONFIG
                stem="$V01_EVAL_ROOT/${checkpoint_arm}__${view}__${protocol}"
                if [[ "$ALLOW_EXISTING_EVAL" != "1" && ( -e "$stem.npz" || -e "$stem.txt" || -e "$stem.metrics.json" ) ]]; then
                    echo "Refusing to overwrite existing evaluation output: $stem" >&2; exit 1
                fi
                dependency=${TRAIN_JOB[$checkpoint_arm]:-}
                eval_export="$COMMON_EXPORT,V01_ARM_CONFIG=$arm_config,EVAL_CHECKPOINT=$checkpoint,EVAL_OUTPUT_TXT=$stem.txt,EVAL_OUTPUT_NPZ=$stem.npz,EXPECTED_CHECKPOINT_EPOCH=8,EVAL_PREFER_REQUESTED_CONFIG=1"
                eval_command="$mkdir_command && AUTO_SBATCH=0 bash '$EVAL_SCRIPT' '$eval_config' --splits val --skip_single_station --skip_diagnostics"
                job=$(submit_or_print "$dependency" \
                    --job-name="v01-${checkpoint_arm}-${view}-${protocol}" --nodes=1 --ntasks=1 \
                    --cpus-per-task="$SLURM_CPUS_PER_TASK" --gres="$SLURM_GRES_RESOURCE:$EVAL_GPUS" \
                    --mem="$SLURM_MEM" --time="$EVAL_TIME" --output="$V01_RUN_ROOT/logs/%x-%j.out" \
                    --error="$V01_RUN_ROOT/logs/%x-%j.err" --export="ALL,$eval_export" \
                    --wrap="$eval_command" | tail -n 1)
                EVAL_JOBS+=("$job")
                echo "[V01] eval_job=$job checkpoint=$checkpoint_arm view=$view protocol=$protocol"
            done
        done
    done
fi

if [[ "$ACTION" == analyze || "$ACTION" == all ]]; then
    dependency=""
    if ((${#EVAL_JOBS[@]})); then dependency=$(IFS=:; echo "${EVAL_JOBS[*]}"); fi
    analyze_command="$mkdir_command && python '$ANALYZE_TOOL' --eval-dir '$V01_EVAL_ROOT' --output-dir '$V01_REPORT_ROOT' --bootstrap-draws 5000 --bootstrap-seed 20260915"
    analyze_job=$(submit_or_print "$dependency" \
        --job-name=v01-analyze --nodes=1 --ntasks=1 --cpus-per-task=4 --time="$ANALYZE_TIME" \
        --mem="$SLURM_MEM" \
        --output="$V01_RUN_ROOT/logs/%x-%j.out" --error="$V01_RUN_ROOT/logs/%x-%j.err" \
        --export="ALL,$COMMON_EXPORT" --wrap="$analyze_command" | tail -n 1)
    echo "[V01] analyze_job=$analyze_job"
fi

echo "[OK] V01 action=$ACTION arms=$ARMS dry_run=$DRY_RUN source_manifest_sha256=$actual_manifest"
