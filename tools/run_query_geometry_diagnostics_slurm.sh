#!/usr/bin/env bash

# Validation-only Slurm launcher for RT55/RT56 query-geometry diagnostics.
#
# Actions:
#   rt55_normal : RT55 checkpoint + pinned normal validation
#   rt55_random : RT55 checkpoint + pinned fixed-random validation (zero-shot)
#   rt56_random : RT56 checkpoint + pinned fixed-random validation
#   rt56_normal : RT56 checkpoint + pinned normal validation (retention)
#
# Dry run (default; submits nothing):
#   DRY_RUN=1 ACTION=all bash tools/run_query_geometry_diagnostics_slurm.sh
#
# Runtime/correctness smoke test (not evidence for model conclusions):
#   SMOKE=1 ACTION=rt55_normal DRY_RUN=0 CONFIRM_QUERY_DIAGNOSTICS=1 \
#     EXPECTED_GIT_COMMIT=<reviewed-full-commit> \
#     bash tools/run_query_geometry_diagnostics_slurm.sh
#
# SMOKE=1 defaults MAX_EVENTS to 8 and requires one action at a time. After the
# reviewed commit is synchronized to the cluster, measure smoke throughput and
# memory before any full run. MAX_EVENTS=0 evaluates all validation events.

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
LAUNCHER_REPO_ROOT=$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
DIAGNOSTIC_SCRIPT=${DIAGNOSTIC_SCRIPT:-$WORKDIR/tools/diagnose_query_geometry_sensitivity.py}

JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
RT55_WEIGHT_NAME=${RT55_WEIGHT_NAME:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT55_WEIGHT_DIR=${RT55_WEIGHT_DIR:-$WORKDIR/$RT55_WEIGHT_NAME}
RT55_CHECKPOINT=${RT55_CHECKPOINT:-${RT55_EP32_CHECKPOINT:-$RT55_WEIGHT_DIR/full_model_best_ep32.pth}}
JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-$RT55_WEIGHT_NAME}
RT56_WEIGHT_NAME=${RT56_WEIGHT_NAME:-weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT56_WEIGHT_PATH=${RT56_WEIGHT_PATH:-$RT56_WEIGHT_NAME}
case "$RT56_WEIGHT_PATH" in
    /*) RT56_WEIGHT_DIR=$RT56_WEIGHT_PATH ;;
    *) RT56_WEIGHT_DIR=$WORKDIR/${RT56_WEIGHT_PATH#./} ;;
esac
RT56_CHECKPOINT=${RT56_CHECKPOINT:-${RT56_EP6_CHECKPOINT:-$RT56_WEIGHT_DIR/full_model_best.pth}}

CONFIG_SOURCE_MODE=${CONFIG_SOURCE_MODE:-resolved}
case "$CONFIG_SOURCE_MODE" in
    resolved)
        RT55_CONFIG=${RT55_CONFIG:-$RT55_WEIGHT_DIR/config.json}
        RT56_CONFIG=${RT56_CONFIG:-$RT56_WEIGHT_DIR/config.json}
        ;;
    source)
        RT55_CONFIG=${RT55_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json}
        RT56_CONFIG=${RT56_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json}
        ;;
    *)
        echo "CONFIG_SOURCE_MODE must be resolved or source; got: $CONFIG_SOURCE_MODE" >&2
        exit 2
        ;;
esac

ACTION=${ACTION:-all}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_QUERY_DIAGNOSTICS=${CONFIRM_QUERY_DIAGNOSTICS:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_GIT_COMMIT_MISMATCH=${ALLOW_GIT_COMMIT_MISMATCH:-0}
ALLOW_UNSAFE_ENCODER_SOURCE_MISMATCH=${ALLOW_UNSAFE_ENCODER_SOURCE_MISMATCH:-0}
SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-git}
EXPECTED_DIAGNOSTIC_SHA256=${EXPECTED_DIAGNOSTIC_SHA256:-}
EXPECTED_LAUNCHER_SHA256=${EXPECTED_LAUNCHER_SHA256:-}
SMOKE=${SMOKE:-0}
SEED=${SEED:-42}
if [[ -z "${MAX_EVENTS+x}" ]]; then
    if [[ "$SMOKE" == "1" ]]; then
        MAX_EVENTS=8
    else
        MAX_EVENTS=0
    fi
fi
STATION_COUNTS=${STATION_COUNTS:-1,3,5,8,12,16}
RADIAL_SCALES=${RADIAL_SCALES:-0,0.5,1,1.5}
PAIR_SAMPLE_LIMIT=${PAIR_SAMPLE_LIMIT:-4096}
EQUIVARIANCE_TOLERANCE=${EQUIVARIANCE_TOLERANCE:-1e-5}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-0}
ENCODER_SHA256=${ENCODER_SHA256:-0}
RT55_EXPECTED_EPOCH=${RT55_EXPECTED_EPOCH:-32}
RT56_EXPECTED_EPOCH=${RT56_EXPECTED_EPOCH:-6}

if [[ -z "${OUT+x}" ]]; then
    if [[ "$SMOKE" == "1" ]]; then
        OUT=$WORKDIR/logs/query_geometry_diagnostics_smoke
    else
        OUT=$WORKDIR/logs/query_geometry_diagnostics_20260902
    fi
fi

SLURM_PARTITION=${SLURM_PARTITION:-diting}
QUERYDIAG_GRES_RESOURCE=${QUERYDIAG_GRES_RESOURCE:-dcu}
QUERYDIAG_GRES_COUNT=${QUERYDIAG_GRES_COUNT:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_TIME=${SLURM_TIME:-23:50:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$OUT/slurm}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
DITING_CONFIG=${DITING_CONFIG:-$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}

case "$ACTION" in
    all|rt55_normal|rt55_random|rt56_random|rt56_normal) ;;
    *)
        echo "ACTION must be all, rt55_normal, rt55_random, rt56_random, or rt56_normal; got: $ACTION" >&2
        exit 2
        ;;
esac
for integer_spec in \
    "MAX_EVENTS:$MAX_EVENTS" \
    "RT55_EXPECTED_EPOCH:$RT55_EXPECTED_EPOCH" \
    "RT56_EXPECTED_EPOCH:$RT56_EXPECTED_EPOCH"; do
    integer_name=${integer_spec%%:*}
    integer_value=${integer_spec#*:}
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "$integer_name must be a non-negative integer; got: $integer_value" >&2
        exit 2
    fi
done
if [[ "$SMOKE" == "1" && "$ACTION" == "all" ]]; then
    echo "SMOKE=1 requires one ACTION at a time; start with ACTION=rt55_normal." >&2
    exit 2
fi
if [[ "$DRY_RUN" != "1" && "$CONFIRM_QUERY_DIAGNOSTICS" != "1" ]]; then
    echo "Submission requires CONFIRM_QUERY_DIAGNOSTICS=1 (or use DRY_RUN=1)." >&2
    exit 2
fi

resolve_path() {
    local path=$1
    local base=${2:-$PWD}
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s\n' "$base/$path" ;;
    esac
}

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

file_sha256() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        printf '%s\n' unavailable
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf '%s\n' sha256sum-unavailable
    fi
}

optional_file_sha256() {
    local enabled=$1
    local path=$2
    if [[ "$enabled" == "1" ]]; then
        file_sha256 "$path"
    else
        printf '%s\n' not-requested
    fi
}

repository_commit() {
    local repository=$1
    git -C "$repository" rev-parse HEAD 2>/dev/null || true
}

SCRIPT_PATH=$(resolve_path "$0" "$SUBMIT_DIR")

verify_uploaded_sources() {
    local diagnostic_sha launcher_sha
    if [[ ! "$EXPECTED_DIAGNOSTIC_SHA256" =~ ^[0-9a-fA-F]{64}$ || \
          ! "$EXPECTED_LAUNCHER_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
        echo "uploaded_sha256 mode requires full EXPECTED_DIAGNOSTIC_SHA256 and EXPECTED_LAUNCHER_SHA256 values." >&2
        exit 2
    fi
    require_file "$DIAGNOSTIC_SCRIPT" "Diagnostic tool"
    require_file "$SCRIPT_PATH" "Diagnostic launcher"
    diagnostic_sha=$(file_sha256 "$DIAGNOSTIC_SCRIPT")
    launcher_sha=$(file_sha256 "$SCRIPT_PATH")
    if [[ "$diagnostic_sha" != "$EXPECTED_DIAGNOSTIC_SHA256" || \
          "$launcher_sha" != "$EXPECTED_LAUNCHER_SHA256" ]]; then
        echo "Uploaded source SHA-256 mismatch: diagnostic=$diagnostic_sha launcher=$launcher_sha" >&2
        exit 1
    fi
    echo "[INFO] uploaded diagnostic and launcher SHA-256 identities matched."
}

case "$SOURCE_IDENTITY_MODE" in
    git)
        SUBMISSION_GIT_COMMIT=$(repository_commit "$WORKDIR")
        if [[ -z "$SUBMISSION_GIT_COMMIT" ]]; then
            SUBMISSION_GIT_COMMIT=$(repository_commit "$LAUNCHER_REPO_ROOT")
        fi
        EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-$SUBMISSION_GIT_COMMIT}
        if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
            echo "EXPECTED_GIT_COMMIT must be a full 40-character commit in git mode; got: $EXPECTED_GIT_COMMIT" >&2
            exit 2
        fi
        ;;
    uploaded_sha256)
        SUBMISSION_GIT_COMMIT=$(repository_commit "$WORKDIR")
        SUBMISSION_GIT_COMMIT=${SUBMISSION_GIT_COMMIT:-unavailable}
        EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-unavailable}
        verify_uploaded_sources
        ;;
    *)
        echo "SOURCE_IDENTITY_MODE must be git or uploaded_sha256; got: $SOURCE_IDENTITY_MODE" >&2
        exit 2
        ;;
esac

action_spec() {
    local diagnostic_action=$1
    case "$diagnostic_action" in
        rt55_normal)
            SPEC_CONFIG=$RT55_CONFIG
            SPEC_CHECKPOINT=$RT55_CHECKPOINT
            SPEC_EXPECTED_EPOCH=$RT55_EXPECTED_EPOCH
            SPEC_PROTOCOL=normal
            SPEC_OUTPUT=$OUT/rt55_normal_querydiag
            SPEC_JOB_NAME=team-rt55-normal-qdiag
            ;;
        rt55_random)
            SPEC_CONFIG=$RT56_CONFIG
            SPEC_CHECKPOINT=$RT55_CHECKPOINT
            SPEC_EXPECTED_EPOCH=$RT55_EXPECTED_EPOCH
            SPEC_PROTOCOL=random
            SPEC_OUTPUT=$OUT/rt55_random_querydiag
            SPEC_JOB_NAME=team-rt55-random-qdiag
            ;;
        rt56_random)
            SPEC_CONFIG=$RT56_CONFIG
            SPEC_CHECKPOINT=$RT56_CHECKPOINT
            SPEC_EXPECTED_EPOCH=$RT56_EXPECTED_EPOCH
            SPEC_PROTOCOL=random
            SPEC_OUTPUT=$OUT/rt56_random_querydiag
            SPEC_JOB_NAME=team-rt56-random-qdiag
            ;;
        rt56_normal)
            SPEC_CONFIG=$RT55_CONFIG
            SPEC_CHECKPOINT=$RT56_CHECKPOINT
            SPEC_EXPECTED_EPOCH=$RT56_EXPECTED_EPOCH
            SPEC_PROTOCOL=normal
            SPEC_OUTPUT=$OUT/rt56_normal_querydiag
            SPEC_JOB_NAME=team-rt56-normal-qdiag
            ;;
        *)
            echo "Unknown diagnostic action: $diagnostic_action" >&2
            exit 2
            ;;
    esac
}

output_state() {
    local prefix=$1
    local present=0
    local path
    for path in \
        "$prefix.summary.json" \
        "$prefix.samples.npz" \
        "$prefix.resolved_config.json" \
        "$prefix.complete.json"; do
        if [[ -e "$path" ]]; then
            present=$((present + 1))
        fi
    done
    if ((present == 0)); then
        printf '%s\n' absent
    elif ((present == 4)); then
        printf '%s\n' complete
    else
        printf 'partial-%s-of-4\n' "$present"
    fi
}

check_active_job() {
    local job_name=$1
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]]; then
        echo "[UNSAFE WARN] ALLOW_ACTIVE_JOB=1; same-name job guard bypassed for $job_name." >&2
        return 0
    fi
    if ! command -v squeue >/dev/null 2>&1; then
        return 0
    fi
    local active_job
    active_job=$(squeue --noheader --user "$(id -un)" --name "$job_name" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    if [[ -n "$active_job" ]]; then
        echo "A same-name Slurm job is already active: $job_name $active_job" >&2
        exit 1
    fi
}

print_action_identity() {
    local diagnostic_action=$1
    echo "[INFO] action=$diagnostic_action split=val protocol=$SPEC_PROTOCOL"
    echo "[INFO] source_identity_mode=$SOURCE_IDENTITY_MODE"
    echo "[INFO] expected_git_commit=$EXPECTED_GIT_COMMIT"
    echo "[INFO] submission_git_commit=$SUBMISSION_GIT_COMMIT"
    echo "[INFO] diagnostic_sha256=$(file_sha256 "$DIAGNOSTIC_SCRIPT")"
    echo "[INFO] expected_diagnostic_sha256=${EXPECTED_DIAGNOSTIC_SHA256:-not-used}"
    echo "[INFO] launcher_sha256=$(file_sha256 "$SCRIPT_PATH")"
    echo "[INFO] expected_launcher_sha256=${EXPECTED_LAUNCHER_SHA256:-not-used}"
    echo "[INFO] config_source_mode=$CONFIG_SOURCE_MODE"
    echo "[INFO] config=$SPEC_CONFIG"
    echo "[INFO] config_sha256=$(file_sha256 "$SPEC_CONFIG")"
    echo "[INFO] checkpoint=$SPEC_CHECKPOINT"
    echo "[INFO] checkpoint_expected_epoch=$SPEC_EXPECTED_EPOCH"
    echo "[INFO] checkpoint_sha256=$(optional_file_sha256 "$CHECKPOINT_SHA256" "$SPEC_CHECKPOINT")"
    echo "[INFO] diting_config=$DITING_CONFIG"
    echo "[INFO] diting_config_sha256=$(file_sha256 "$DITING_CONFIG")"
    echo "[INFO] diting_pretrained=$DITING_PRETRAINED"
    echo "[INFO] diting_pretrained_sha256=$(optional_file_sha256 "$ENCODER_SHA256" "$DITING_PRETRAINED")"
    echo "[INFO] output_prefix=$SPEC_OUTPUT"
    echo "[INFO] max_events=$MAX_EVENTS station_counts=$STATION_COUNTS radial_scales=$RADIAL_SCALES"
    echo "[INFO] allow_unsafe_encoder_source_mismatch=$ALLOW_UNSAFE_ENCODER_SOURCE_MISMATCH"
    if [[ "$SMOKE" == "1" ]]; then
        echo "[WARN] prefix-event smoke output is for runtime/correctness only, not model conclusions."
    fi
}

if [[ "$ACTION" == "all" ]]; then
    ACTION_LIST=(rt55_normal rt55_random rt56_random rt56_normal)
else
    ACTION_LIST=("$ACTION")
fi

require_file "$DIAGNOSTIC_SCRIPT" "Diagnostic tool"
require_file "$RT55_CONFIG" "RT55 config"
require_file "$RT56_CONFIG" "RT56 config"
require_file "$DITING_CONFIG" "DiTing config"
require_file "$DITING_PRETRAINED" "DiTing pretrained checkpoint"

export WORKDIR DIAGNOSTIC_SCRIPT RT55_CONFIG RT56_CONFIG CONFIG_SOURCE_MODE
export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT56_WEIGHT_PATH
export RT55_CHECKPOINT RT56_CHECKPOINT RT55_EXPECTED_EPOCH RT56_EXPECTED_EPOCH OUT
export DRY_RUN CONFIRM_QUERY_DIAGNOSTICS ALLOW_ACTIVE_JOB ALLOW_GIT_COMMIT_MISMATCH
export ALLOW_UNSAFE_ENCODER_SOURCE_MISMATCH
export SOURCE_IDENTITY_MODE EXPECTED_DIAGNOSTIC_SHA256 EXPECTED_LAUNCHER_SHA256
export EXPECTED_GIT_COMMIT SUBMISSION_GIT_COMMIT SMOKE
export SEED MAX_EVENTS STATION_COUNTS RADIAL_SCALES PAIR_SAMPLE_LIMIT
export EQUIVARIANCE_TOLERANCE CHECKPOINT_SHA256 ENCODER_SHA256 ALLOW_EXISTING_OUTPUT
export SLURM_PARTITION QUERYDIAG_GRES_RESOURCE QUERYDIAG_GRES_COUNT
export SLURM_CPUS_PER_TASK
export SLURM_TIME SLURM_LOG_DIR CONDA_ENV MODULE_UNLOAD MODULE_LOADS
export DITING_CONFIG DITING_PRETRAINED

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submitted=0
    skipped=0
    for diagnostic_action in "${ACTION_LIST[@]}"; do
        action_spec "$diagnostic_action"
        require_file "$SPEC_CONFIG" "$diagnostic_action config"
        require_file "$SPEC_CHECKPOINT" "$diagnostic_action checkpoint"
        state=$(output_state "$SPEC_OUTPUT")
        if [[ "$state" != "absent" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
            echo "[INFO] existing $state output; skip $diagnostic_action: $SPEC_OUTPUT"
            skipped=$((skipped + 1))
            continue
        fi
        check_active_job "$SPEC_JOB_NAME"
        print_action_identity "$diagnostic_action"
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '[DRY-RUN] sbatch --job-name=%q --partition=%q --nodes=1 --ntasks-per-node=1 --cpus-per-task=%q --gres=%q --time=%q --chdir=%q --output=%q --error=%q --export=ALL %q %q\n' \
                "$SPEC_JOB_NAME" "$SLURM_PARTITION" "$SLURM_CPUS_PER_TASK" \
                "$QUERYDIAG_GRES_RESOURCE:$QUERYDIAG_GRES_COUNT" "$SLURM_TIME" "$WORKDIR" \
                "$SLURM_LOG_DIR/%x-%j.out" "$SLURM_LOG_DIR/%x-%j.err" \
                "$SCRIPT_PATH" "$diagnostic_action"
        else
            mkdir -p "$SLURM_LOG_DIR" "$(dirname -- "$SPEC_OUTPUT")"
            sbatch \
                --job-name="$SPEC_JOB_NAME" \
                --partition="$SLURM_PARTITION" \
                --nodes=1 \
                --ntasks-per-node=1 \
                --cpus-per-task="$SLURM_CPUS_PER_TASK" \
                --gres="$QUERYDIAG_GRES_RESOURCE:$QUERYDIAG_GRES_COUNT" \
                --time="$SLURM_TIME" \
                --chdir="$WORKDIR" \
                --output="$SLURM_LOG_DIR/%x-%j.out" \
                --error="$SLURM_LOG_DIR/%x-%j.err" \
                --export=ALL \
                "$SCRIPT_PATH" "$diagnostic_action"
        fi
        submitted=$((submitted + 1))
    done
    echo "[INFO] query-diagnostic submission complete: requested=$submitted skipped=$skipped dry_run=$DRY_RUN"
    exit 0
fi

if (($# != 1)); then
    echo "Slurm worker requires exactly one diagnostic action argument." >&2
    exit 2
fi
DIAGNOSTIC_ACTION=$1
action_spec "$DIAGNOSTIC_ACTION"
require_file "$SPEC_CONFIG" "$DIAGNOSTIC_ACTION config"
require_file "$SPEC_CHECKPOINT" "$DIAGNOSTIC_ACTION checkpoint"

cd "$WORKDIR"
ACTUAL_GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || true)
if [[ "$SOURCE_IDENTITY_MODE" == "git" ]]; then
    if [[ "$ACTUAL_GIT_COMMIT" != "$EXPECTED_GIT_COMMIT" ]]; then
        if [[ "$ALLOW_GIT_COMMIT_MISMATCH" != "1" ]]; then
            echo "Worker Git commit mismatch: expected=$EXPECTED_GIT_COMMIT actual=$ACTUAL_GIT_COMMIT" >&2
            exit 1
        fi
        echo "[UNSAFE WARN] ALLOW_GIT_COMMIT_MISMATCH=1 expected=$EXPECTED_GIT_COMMIT actual=$ACTUAL_GIT_COMMIT" >&2
    fi
else
    verify_uploaded_sources
fi

state=$(output_state "$SPEC_OUTPUT")
if [[ "$state" != "absent" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    echo "Output set is $state; refusing worker overwrite: $SPEC_OUTPUT" >&2
    exit 1
fi
OUTPUT_LOCK=$SPEC_OUTPUT.lock
mkdir -p "$(dirname -- "$SPEC_OUTPUT")"
if ! mkdir "$OUTPUT_LOCK" 2>/dev/null; then
    echo "Another worker already owns output lock: $OUTPUT_LOCK" >&2
    exit 1
fi
cleanup_output_lock() {
    rmdir "$OUTPUT_LOCK" 2>/dev/null || true
}
trap cleanup_output_lock EXIT INT TERM

echo "[INFO] repository=$(pwd)"
echo "[INFO] branch=$(git branch --show-current 2>/dev/null || true)"
echo "[INFO] actual_git_commit=$ACTUAL_GIT_COMMIT"
print_action_identity "$DIAGNOSTIC_ACTION"

restore_nounset=0
if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
fi
if [[ -f /etc/profile ]]; then
    # shellcheck disable=SC1091
    source /etc/profile
fi
if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if [[ "$restore_nounset" -eq 1 ]]; then
    set -u
fi

if command -v module >/dev/null 2>&1 || declare -F module >/dev/null 2>&1; then
    if [[ -n "$MODULE_UNLOAD" ]]; then
        module unload "$MODULE_UNLOAD" || true
    fi
    for module_name in $MODULE_LOADS; do
        module load "$module_name"
    done
else
    echo "[WARN] module command is unavailable; skipping module load." >&2
fi

if [[ -n "$CONDA_ENV" ]]; then
    restore_nounset=0
    if [[ $- == *u* ]]; then
        restore_nounset=1
        set +u
    fi
    export PS1=${PS1:-}
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1090
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        source activate "$CONDA_ENV"
    fi
    if [[ "$restore_nounset" -eq 1 ]]; then
        set -u
    fi
fi

mkdir -p "$(dirname -- "$SPEC_OUTPUT")"
DIAGNOSTIC_ARGS=(
    --config "$SPEC_CONFIG"
    --config-source-mode "$CONFIG_SOURCE_MODE"
    --deployment-source-mode "$SOURCE_IDENTITY_MODE"
    --checkpoint "$SPEC_CHECKPOINT"
    --expected-checkpoint-epoch "$SPEC_EXPECTED_EPOCH"
    --protocol "$SPEC_PROTOCOL"
    --split val
    --output-prefix "$SPEC_OUTPUT"
    --device cuda:0
    --seed "$SEED"
    --max-events "$MAX_EVENTS"
    --station-counts "$STATION_COUNTS"
    --radial-scales "$RADIAL_SCALES"
    --pair-sample-limit "$PAIR_SAMPLE_LIMIT"
    --equivariance-tolerance "$EQUIVARIANCE_TOLERANCE"
    --diting-config "$DITING_CONFIG"
    --diting-pretrained "$DITING_PRETRAINED"
)
if [[ "$CHECKPOINT_SHA256" == "1" ]]; then
    DIAGNOSTIC_ARGS+=(--checkpoint-sha256)
fi
if [[ "$ENCODER_SHA256" == "1" ]]; then
    DIAGNOSTIC_ARGS+=(--encoder-sha256)
fi
if [[ "$ALLOW_UNSAFE_ENCODER_SOURCE_MISMATCH" == "1" ]]; then
    DIAGNOSTIC_ARGS+=(--allow-unsafe-encoder-source-mismatch)
fi
if [[ "$ALLOW_EXISTING_OUTPUT" == "1" ]]; then
    DIAGNOSTIC_ARGS+=(--force)
fi

# SLURM_GPUS is an srun input option equivalent to --gpus.  Never pass a
# request-side counter under that reserved name: this cluster allocates the
# non-GPU-named GRES dcu, and mixing --gpus=1 with dcu:1 makes step creation
# fail with "Invalid generic resource (gres) specification".
env -u SLURM_GPUS srun --ntasks=1 \
    python "$DIAGNOSTIC_SCRIPT" "${DIAGNOSTIC_ARGS[@]}"

for output_path in \
    "$SPEC_OUTPUT.summary.json" \
    "$SPEC_OUTPUT.samples.npz" \
    "$SPEC_OUTPUT.resolved_config.json" \
    "$SPEC_OUTPUT.complete.json"; do
    if [[ ! -s "$output_path" ]]; then
        echo "Expected diagnostic output is missing or empty: $output_path" >&2
        exit 1
    fi
done
echo "[OK] query-geometry diagnostic complete: $SPEC_OUTPUT"
