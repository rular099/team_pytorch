#!/usr/bin/env bash

# Submit the two rt55 best-checkpoint validation jobs required for the
# waveform-use control:
#   1. normal station-waveform pairing
#   2. waveform-station-roll pairing (metadata/targets remain fixed)
#
# Both jobs evaluate validation only with the fixed realtime protocol from the
# resolved run config and write TXT, NPZ, and formal metrics JSON outputs.
#
# Usage:
#   bash tools/eval_rt55_validation_normal_roll_slurm.sh
#   DRY_RUN=1 bash tools/eval_rt55_validation_normal_roll_slurm.sh
#   ACTION=normal bash tools/eval_rt55_validation_normal_roll_slurm.sh
#   ACTION=roll bash tools/eval_rt55_validation_normal_roll_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
WEIGHT_NAME=${WEIGHT_NAME:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
WEIGHT_DIR=${WEIGHT_DIR:-$WORKDIR/$WEIGHT_NAME}
CONFIG=${CONFIG:-$WEIGHT_DIR/config.json}
CHECKPOINT=${CHECKPOINT:-$WEIGHT_DIR/full_model_best.pth}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}
RUN_LOG_DIR=${RUN_LOG_DIR:-$WORKDIR/logs/$WEIGHT_NAME}
ACTION=${ACTION:-all}
PERMUTATION_SEED=${PERMUTATION_SEED:-12345}
SLURM_TIME=${SLURM_TIME:-16:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

submitted_count=0
skipped_count=0

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

output_exists() {
    local txt_path=$1
    local npz_path=$2
    local metrics_path=$3
    if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
        return 1
    fi
    if [[ -s "$txt_path" || -s "$npz_path" || -s "$metrics_path" ]]; then
        echo "[INFO] existing validation output; skip: $txt_path"
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    return 1
}

submit_eval() {
    local mode=$1
    local output_stem
    local output_txt
    local output_npz
    local output_metrics
    local job_name
    local eval_args

    if [[ "$mode" == "normal" ]]; then
        output_stem="$RUN_LOG_DIR/eval_validation_best_normal"
        job_name="team-rt55-val-normal"
        eval_args=(--splits val --skip_single_station --skip_diagnostics)
    else
        output_stem="$RUN_LOG_DIR/eval_validation_best_waveform_station_roll"
        job_name="team-rt55-val-roll"
        eval_args=(
            --splits val
            --skip_single_station
            --skip_diagnostics
            --waveform_station_permutation roll
            --waveform_station_permutation_seed "$PERMUTATION_SEED"
        )
    fi
    output_txt="${output_stem}.txt"
    output_npz="${output_stem}.npz"
    output_metrics="${output_stem}.metrics.json"

    if output_exists "$output_txt" "$output_npz" "$output_metrics"; then
        return 0
    fi

    echo "[INFO] rt55 validation $mode"
    echo "[INFO] checkpoint: $CHECKPOINT"
    echo "[INFO] output stem: $output_stem"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[DRY-RUN] WORKDIR=%q JOB_NAME=%q EVAL_CHECKPOINT=%q bash %q %q' \
            "$WORKDIR" "$job_name" "$CHECKPOINT" "$EVAL_SCRIPT" "$CONFIG"
        printf ' %q' "${eval_args[@]}"
        printf '\n'
        submitted_count=$((submitted_count + 1))
        return 0
    fi

    WORKDIR="$WORKDIR" \
    RUN_LOG_DIR="$RUN_LOG_DIR" \
    JOB_NAME="$job_name" \
    SLURM_TIME="$SLURM_TIME" \
    SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
    EVAL_CHECKPOINT="$CHECKPOINT" \
    EVAL_OUTPUT_TXT="$output_txt" \
    EVAL_OUTPUT_NPZ="$output_npz" \
    AUTO_SBATCH=1 \
    bash "$EVAL_SCRIPT" "$CONFIG" "${eval_args[@]}"
    submitted_count=$((submitted_count + 1))
}

case "$ACTION" in
    all|normal|roll) ;;
    *)
        echo "ACTION must be all, normal, or roll; got: $ACTION" >&2
        exit 2
        ;;
esac

require_file "$EVAL_SCRIPT" "Eval launcher"
require_file "$CONFIG" "Resolved rt55 run config"
require_file "$CHECKPOINT" "rt55 best checkpoint"

if [[ "$ACTION" == "all" || "$ACTION" == "normal" ]]; then
    submit_eval normal
fi
if [[ "$ACTION" == "all" || "$ACTION" == "roll" ]]; then
    submit_eval roll
fi

echo "[INFO] rt55 validation submission complete: submitted=$submitted_count skipped=$skipped_count dry_run=$DRY_RUN"
