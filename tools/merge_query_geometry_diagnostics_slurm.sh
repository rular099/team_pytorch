#!/usr/bin/env bash

# Submit one short compute-node job to verify and merge a completed diagnostic
# shard array. The login node does not need torch or numpy.
#
# Example after all eight rt55_normal shards complete:
#   ACTION=rt55_normal NUM_EVENT_SHARDS=8 DRY_RUN=0 \
#     CONFIRM_QUERYDIAG_MERGE=1 SOURCE_IDENTITY_MODE=uploaded_sha256 \
#     EXPECTED_DIAGNOSTIC_SHA256=<sha256> EXPECTED_MERGE_SHA256=<sha256> \
#     EXPECTED_MERGE_LAUNCHER_SHA256=<sha256> \
#     bash tools/merge_query_geometry_diagnostics_slurm.sh

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
DIAGNOSTIC_SCRIPT=${DIAGNOSTIC_SCRIPT:-$WORKDIR/tools/diagnose_query_geometry_sensitivity.py}
MERGE_SCRIPT=${MERGE_SCRIPT:-$WORKDIR/tools/merge_query_geometry_diagnostics_shards.py}
ACTION=${ACTION:-rt55_normal}
NUM_EVENT_SHARDS=${NUM_EVENT_SHARDS:-8}
OUT=${OUT:-$WORKDIR/logs/query_geometry_diagnostics_20260902}
INPUT_PREFIX_BASE=${INPUT_PREFIX_BASE:-$OUT/${ACTION}_querydiag}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-$OUT/${ACTION}_querydiag_merged}

DRY_RUN=${DRY_RUN:-1}
CONFIRM_QUERYDIAG_MERGE=${CONFIRM_QUERYDIAG_MERGE:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-git}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-}
EXPECTED_DIAGNOSTIC_SHA256=${EXPECTED_DIAGNOSTIC_SHA256:-}
EXPECTED_MERGE_SHA256=${EXPECTED_MERGE_SHA256:-}
EXPECTED_MERGE_LAUNCHER_SHA256=${EXPECTED_MERGE_LAUNCHER_SHA256:-}
if [[ "$SOURCE_IDENTITY_MODE" == "git" && -z "$EXPECTED_GIT_COMMIT" ]]; then
    EXPECTED_GIT_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || true)
fi

SLURM_PARTITION=${SLURM_PARTITION:-diting}
QUERYDIAG_GRES_RESOURCE=${QUERYDIAG_GRES_RESOURCE:-dcu}
QUERYDIAG_GRES_COUNT=${QUERYDIAG_GRES_COUNT:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}
SLURM_TIME=${SLURM_TIME:-01:00:00}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$OUT/slurm}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
JOB_NAME=${JOB_NAME:-team-${ACTION//_/-}-qmerge}

case "$ACTION" in
    rt55_normal|rt55_random|rt56_random|rt56_normal) ;;
    *)
        echo "Unsupported ACTION: $ACTION" >&2
        exit 2
        ;;
esac
if [[ ! "$NUM_EVENT_SHARDS" =~ ^[0-9]+$ ]] || ((NUM_EVENT_SHARDS < 2)); then
    echo "NUM_EVENT_SHARDS must be an integer of at least 2; got: $NUM_EVENT_SHARDS" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "1" && "$CONFIRM_QUERYDIAG_MERGE" != "1" ]]; then
    echo "Submission requires CONFIRM_QUERYDIAG_MERGE=1 (or use DRY_RUN=1)." >&2
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
    sha256sum "$1" | awk '{print $1}'
}

verify_sources() {
    require_file "$DIAGNOSTIC_SCRIPT" "Diagnostic tool"
    require_file "$MERGE_SCRIPT" "Shard merge tool"
    local diagnostic_sha merge_sha merge_launcher_sha
    diagnostic_sha=$(file_sha256 "$DIAGNOSTIC_SCRIPT")
    merge_sha=$(file_sha256 "$MERGE_SCRIPT")
    merge_launcher_sha=$(file_sha256 "$SCRIPT_PATH")
    case "$SOURCE_IDENTITY_MODE" in
        git)
            local actual_commit
            actual_commit=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || true)
            if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
                echo "Git mode requires a full EXPECTED_GIT_COMMIT." >&2
                exit 2
            fi
            if [[ "$actual_commit" != "$EXPECTED_GIT_COMMIT" ]]; then
                echo "Merge source Git commit mismatch: expected=$EXPECTED_GIT_COMMIT actual=$actual_commit" >&2
                exit 1
            fi
            ;;
        uploaded_sha256)
            if [[ ! "$EXPECTED_DIAGNOSTIC_SHA256" =~ ^[0-9a-fA-F]{64}$ || \
                  ! "$EXPECTED_MERGE_SHA256" =~ ^[0-9a-fA-F]{64}$ || \
                  ! "$EXPECTED_MERGE_LAUNCHER_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
                echo "uploaded_sha256 mode requires full diagnostic, merge, and merge-launcher SHA-256 values." >&2
                exit 2
            fi
            if [[ "$diagnostic_sha" != "$EXPECTED_DIAGNOSTIC_SHA256" || \
                  "$merge_sha" != "$EXPECTED_MERGE_SHA256" || \
                  "$merge_launcher_sha" != "$EXPECTED_MERGE_LAUNCHER_SHA256" ]]; then
                echo "Uploaded merge source SHA-256 mismatch." >&2
                exit 1
            fi
            ;;
        *)
            echo "SOURCE_IDENTITY_MODE must be git or uploaded_sha256." >&2
            exit 2
            ;;
    esac
    echo "[INFO] diagnostic_sha256=$diagnostic_sha"
    echo "[INFO] merge_sha256=$merge_sha"
    echo "[INFO] merge_launcher_sha256=$merge_launcher_sha"
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
        [[ ! -e "$path" ]] || present=$((present + 1))
    done
    if ((present == 0)); then
        printf '%s\n' absent
    elif ((present == 4)); then
        printf '%s\n' complete
    else
        printf 'partial-%s-of-4\n' "$present"
    fi
}

require_all_shards() {
    local shard_id prefix suffix
    for ((shard_id = 0; shard_id < NUM_EVENT_SHARDS; shard_id++)); do
        suffix=$(printf '.shard-%05d-of-%05d' "$shard_id" "$NUM_EVENT_SHARDS")
        prefix=$INPUT_PREFIX_BASE$suffix
        require_file "$prefix.complete.json" "Shard $shard_id completion manifest"
        require_file "$prefix.summary.json" "Shard $shard_id summary"
        require_file "$prefix.samples.npz" "Shard $shard_id samples"
        require_file "$prefix.resolved_config.json" "Shard $shard_id resolved config"
    done
}

check_active_job() {
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]] || ! command -v squeue >/dev/null 2>&1; then
        return 0
    fi
    local active_job
    active_job=$(squeue --noheader --user "$(id -un)" --name "$JOB_NAME" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    if [[ -n "$active_job" ]]; then
        echo "A same-name merge job is already active: $JOB_NAME $active_job" >&2
        exit 1
    fi
}

SCRIPT_PATH=$(resolve_path "$0" "$SUBMIT_DIR")
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    verify_sources
    require_all_shards
    state=$(output_state "$OUTPUT_PREFIX")
    if [[ "$state" != "absent" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
        echo "Merge output is $state; refusing overwrite: $OUTPUT_PREFIX" >&2
        exit 1
    fi
    check_active_job
    export WORKDIR DIAGNOSTIC_SCRIPT MERGE_SCRIPT ACTION NUM_EVENT_SHARDS OUT
    export INPUT_PREFIX_BASE OUTPUT_PREFIX SOURCE_IDENTITY_MODE EXPECTED_GIT_COMMIT
    export EXPECTED_DIAGNOSTIC_SHA256 EXPECTED_MERGE_SHA256
    export EXPECTED_MERGE_LAUNCHER_SHA256 ALLOW_EXISTING_OUTPUT
    export CONDA_ENV MODULE_UNLOAD MODULE_LOADS
    echo "[INFO] action=$ACTION shards=$NUM_EVENT_SHARDS"
    echo "[INFO] input_prefix_base=$INPUT_PREFIX_BASE"
    echo "[INFO] output_prefix=$OUTPUT_PREFIX"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[DRY-RUN] sbatch --job-name=%q --partition=%q --nodes=1 --ntasks=1 --cpus-per-task=%q --gres=%q --time=%q --chdir=%q --output=%q --error=%q --export=ALL %q\n' \
            "$JOB_NAME" "$SLURM_PARTITION" "$SLURM_CPUS_PER_TASK" \
            "$QUERYDIAG_GRES_RESOURCE:$QUERYDIAG_GRES_COUNT" "$SLURM_TIME" \
            "$WORKDIR" "$SLURM_LOG_DIR/%x-%j.out" "$SLURM_LOG_DIR/%x-%j.err" \
            "$SCRIPT_PATH"
        exit 0
    fi
    mkdir -p "$SLURM_LOG_DIR" "$(dirname -- "$OUTPUT_PREFIX")"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$SLURM_PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gres="$QUERYDIAG_GRES_RESOURCE:$QUERYDIAG_GRES_COUNT" \
        --time="$SLURM_TIME" \
        --chdir="$WORKDIR" \
        --output="$SLURM_LOG_DIR/%x-%j.out" \
        --error="$SLURM_LOG_DIR/%x-%j.err" \
        --export=ALL \
        "$SCRIPT_PATH"
    exit 0
fi

verify_sources
require_all_shards
state=$(output_state "$OUTPUT_PREFIX")
if [[ "$state" != "absent" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    echo "Merge output is $state; refusing overwrite: $OUTPUT_PREFIX" >&2
    exit 1
fi
OUTPUT_LOCK=$OUTPUT_PREFIX.lock
mkdir -p "$(dirname -- "$OUTPUT_PREFIX")"
if ! mkdir "$OUTPUT_LOCK" 2>/dev/null; then
    echo "Another worker already owns merge output lock: $OUTPUT_LOCK" >&2
    exit 1
fi
trap 'rmdir "$OUTPUT_LOCK" 2>/dev/null || true' EXIT INT TERM

restore_nounset=0
if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
fi
[[ ! -f /etc/profile ]] || source /etc/profile
[[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
if [[ "$restore_nounset" -eq 1 ]]; then
    set -u
fi
if command -v module >/dev/null 2>&1 || declare -F module >/dev/null 2>&1; then
    [[ -z "$MODULE_UNLOAD" ]] || module unload "$MODULE_UNLOAD" || true
    for module_name in $MODULE_LOADS; do
        module load "$module_name"
    done
fi
if [[ -n "$CONDA_ENV" ]]; then
    restore_nounset=0
    if [[ $- == *u* ]]; then
        restore_nounset=1
        set +u
    fi
    export PS1=${PS1:-}
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        source activate "$CONDA_ENV"
    fi
    [[ "$restore_nounset" -ne 1 ]] || set -u
fi

MERGE_ARGS=(
    --input-prefix-base "$INPUT_PREFIX_BASE"
    --num-shards "$NUM_EVENT_SHARDS"
    --output-prefix "$OUTPUT_PREFIX"
)
[[ "$ALLOW_EXISTING_OUTPUT" != "1" ]] || MERGE_ARGS+=(--force)
env -u SLURM_GPUS srun --ntasks=1 python "$MERGE_SCRIPT" "${MERGE_ARGS[@]}"
[[ -s "$OUTPUT_PREFIX.complete.json" ]] || {
    echo "Merged completion manifest is missing: $OUTPUT_PREFIX.complete.json" >&2
    exit 1
}
echo "[OK] query-geometry shards merged: $OUTPUT_PREFIX"
