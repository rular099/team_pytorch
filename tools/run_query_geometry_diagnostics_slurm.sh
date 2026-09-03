#!/usr/bin/env bash

# Validation-only Slurm launcher for query-geometry diagnostics.
#
# The four actions keep model/checkpoint and validation protocol explicit:
#   rt55_normal : RT55 ep32 + RT55 normal validation config
#   rt55_random : RT55 ep32 + RT56 fixed-random validation config (zero-shot)
#   rt56_random : RT56 ep6  + RT56 fixed-random validation config
#   rt56_normal : RT56 ep6  + RT55 normal validation config (retention)
#
# This launcher never selects or evaluates the held-out test split.  It defaults
# to a dry run and refuses existing outputs.  Examples from the repository root:
#
#   ACTION=all bash tools/run_query_geometry_diagnostics_slurm.sh
#   DRY_RUN=0 CONFIRM_QUERY_DIAGNOSTICS=1 ACTION=all \
#     bash tools/run_query_geometry_diagnostics_slurm.sh
#
# For a short cluster smoke test before the full run:
#
#   MAX_EVENTS=8 OUT=/new/output/path ACTION=rt55_normal \
#     DRY_RUN=0 CONFIRM_QUERY_DIAGNOSTICS=1 \
#     bash tools/run_query_geometry_diagnostics_slurm.sh
#
# MAX_EVENTS=0 evaluates all validation events.  Query interventions require
# roughly five forwards per realtime sample with the default radial scales, so
# choose SLURM_TIME/MAX_EVENTS according to the cluster's effective QoS limit.

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
DIAGNOSTIC_SCRIPT=${DIAGNOSTIC_SCRIPT:-$WORKDIR/tools/diagnose_query_geometry_sensitivity.py}
RT55_CONFIG=${RT55_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json}
RT56_CONFIG=${RT56_CONFIG:-$WORKDIR/pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json}

JAPAN_FULL_DATA_ROOT=${JAPAN_FULL_DATA_ROOT:-/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30}
RT55_WEIGHT_NAME=${RT55_WEIGHT_NAME:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT55_WEIGHT_DIR=${RT55_WEIGHT_DIR:-$WORKDIR/$RT55_WEIGHT_NAME}
RT55_EP32_CHECKPOINT=${RT55_EP32_CHECKPOINT:-$RT55_WEIGHT_DIR/full_model_best_ep32.pth}
JAPAN_FULL_WEIGHT_PATH=${JAPAN_FULL_WEIGHT_PATH:-$RT55_WEIGHT_NAME}
RT56_WEIGHT_NAME=${RT56_WEIGHT_NAME:-weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT56_WEIGHT_PATH=${RT56_WEIGHT_PATH:-$RT56_WEIGHT_NAME}
case "$RT56_WEIGHT_PATH" in
    /*) RT56_WEIGHT_DIR=$RT56_WEIGHT_PATH ;;
    *) RT56_WEIGHT_DIR=$WORKDIR/${RT56_WEIGHT_PATH#./} ;;
esac
RT56_EP6_CHECKPOINT=${RT56_EP6_CHECKPOINT:-$RT56_WEIGHT_DIR/full_model_best.pth}

OUT=${OUT:-$WORKDIR/logs/query_geometry_diagnostics_20260902}
ACTION=${ACTION:-all}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_QUERY_DIAGNOSTICS=${CONFIRM_QUERY_DIAGNOSTICS:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
SEED=${SEED:-42}
MAX_EVENTS=${MAX_EVENTS:-0}
STATION_COUNTS=${STATION_COUNTS:-1,3,5,8,12,16}
RADIAL_SCALES=${RADIAL_SCALES:-0,0.5,1,1.5}
PAIR_SAMPLE_LIMIT=${PAIR_SAMPLE_LIMIT:-4096}
EQUIVARIANCE_TOLERANCE=${EQUIVARIANCE_TOLERANCE:-1e-5}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-0}

SLURM_PARTITION=${SLURM_PARTITION:-diting}
SLURM_GRES_RESOURCE=${SLURM_GRES_RESOURCE:-dcu}
SLURM_GPUS=${SLURM_GPUS:-1}
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
if [[ ! "$MAX_EVENTS" =~ ^[0-9]+$ ]]; then
    echo "MAX_EVENTS must be zero or a positive integer; got: $MAX_EVENTS" >&2
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

action_spec() {
    local diagnostic_action=$1
    case "$diagnostic_action" in
        rt55_normal)
            SPEC_CONFIG=$RT55_CONFIG
            SPEC_CHECKPOINT=$RT55_EP32_CHECKPOINT
            SPEC_PROTOCOL=normal
            SPEC_OUTPUT=$OUT/rt55_ep32_normal_querydiag
            SPEC_JOB_NAME=team-rt55-normal-qdiag
            ;;
        rt55_random)
            SPEC_CONFIG=$RT56_CONFIG
            SPEC_CHECKPOINT=$RT55_EP32_CHECKPOINT
            SPEC_PROTOCOL=random
            SPEC_OUTPUT=$OUT/rt55_ep32_random_querydiag
            SPEC_JOB_NAME=team-rt55-random-qdiag
            ;;
        rt56_random)
            SPEC_CONFIG=$RT56_CONFIG
            SPEC_CHECKPOINT=$RT56_EP6_CHECKPOINT
            SPEC_PROTOCOL=random
            SPEC_OUTPUT=$OUT/rt56_ep6_random_querydiag
            SPEC_JOB_NAME=team-rt56-random-qdiag
            ;;
        rt56_normal)
            SPEC_CONFIG=$RT55_CONFIG
            SPEC_CHECKPOINT=$RT56_EP6_CHECKPOINT
            SPEC_PROTOCOL=normal
            SPEC_OUTPUT=$OUT/rt56_ep6_normal_querydiag
            SPEC_JOB_NAME=team-rt56-normal-qdiag
            ;;
        *)
            echo "Unknown diagnostic action: $diagnostic_action" >&2
            exit 2
            ;;
    esac
}

output_exists() {
    local prefix=$1
    [[ -e "$prefix.summary.json" || -e "$prefix.samples.npz" || -e "$prefix.resolved_config.json" ]]
}

SCRIPT_PATH=$(resolve_path "$0" "$SUBMIT_DIR")
ACTION_LIST=()
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

export WORKDIR DIAGNOSTIC_SCRIPT RT55_CONFIG RT56_CONFIG
export JAPAN_FULL_DATA_ROOT JAPAN_FULL_WEIGHT_PATH RT56_WEIGHT_PATH
export RT55_EP32_CHECKPOINT RT56_EP6_CHECKPOINT OUT
export DRY_RUN CONFIRM_QUERY_DIAGNOSTICS
export SEED MAX_EVENTS STATION_COUNTS RADIAL_SCALES PAIR_SAMPLE_LIMIT
export EQUIVARIANCE_TOLERANCE CHECKPOINT_SHA256 ALLOW_EXISTING_OUTPUT
export SLURM_PARTITION SLURM_GRES_RESOURCE SLURM_GPUS SLURM_CPUS_PER_TASK
export SLURM_TIME SLURM_LOG_DIR CONDA_ENV MODULE_UNLOAD MODULE_LOADS
export DITING_CONFIG DITING_PRETRAINED

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submitted=0
    skipped=0
    for diagnostic_action in "${ACTION_LIST[@]}"; do
        action_spec "$diagnostic_action"
        require_file "$SPEC_CONFIG" "$diagnostic_action config"
        require_file "$SPEC_CHECKPOINT" "$diagnostic_action checkpoint"
        if output_exists "$SPEC_OUTPUT" && [[ "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
            echo "[INFO] existing output; skip $diagnostic_action: $SPEC_OUTPUT"
            skipped=$((skipped + 1))
            continue
        fi
        echo "[INFO] action=$diagnostic_action protocol=$SPEC_PROTOCOL"
        echo "[INFO] config=$SPEC_CONFIG"
        echo "[INFO] checkpoint=$SPEC_CHECKPOINT"
        echo "[INFO] output_prefix=$SPEC_OUTPUT"
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '[DRY-RUN] sbatch --job-name=%q --partition=%q --nodes=1 --ntasks-per-node=1 --cpus-per-task=%q --gres=%q --time=%q --chdir=%q --output=%q --error=%q --export=ALL %q %q\n' \
                "$SPEC_JOB_NAME" "$SLURM_PARTITION" "$SLURM_CPUS_PER_TASK" \
                "$SLURM_GRES_RESOURCE:$SLURM_GPUS" "$SLURM_TIME" "$WORKDIR" \
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
                --gres="$SLURM_GRES_RESOURCE:$SLURM_GPUS" \
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
if output_exists "$SPEC_OUTPUT" && [[ "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    echo "Output already exists; refusing worker overwrite: $SPEC_OUTPUT" >&2
    exit 1
fi

cd "$WORKDIR"
echo "[INFO] repository=$(pwd)"
echo "[INFO] branch=$(git branch --show-current 2>/dev/null || true)"
echo "[INFO] commit=$(git rev-parse HEAD 2>/dev/null || true)"
echo "[INFO] action=$DIAGNOSTIC_ACTION split=val protocol=$SPEC_PROTOCOL"
echo "[INFO] config=$SPEC_CONFIG"
echo "[INFO] checkpoint=$SPEC_CHECKPOINT"
echo "[INFO] output_prefix=$SPEC_OUTPUT"
echo "[INFO] max_events=$MAX_EVENTS station_counts=$STATION_COUNTS radial_scales=$RADIAL_SCALES"

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
    --checkpoint "$SPEC_CHECKPOINT"
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
if [[ "$ALLOW_EXISTING_OUTPUT" == "1" ]]; then
    DIAGNOSTIC_ARGS+=(--force)
fi

srun --ntasks=1 python "$DIAGNOSTIC_SCRIPT" "${DIAGNOSTIC_ARGS[@]}"

for output_path in \
    "$SPEC_OUTPUT.summary.json" \
    "$SPEC_OUTPUT.samples.npz" \
    "$SPEC_OUTPUT.resolved_config.json"; do
    if [[ ! -s "$output_path" ]]; then
        echo "Expected diagnostic output is missing or empty: $output_path" >&2
        exit 1
    fi
done
echo "[OK] query-geometry diagnostic complete: $SPEC_OUTPUT"
