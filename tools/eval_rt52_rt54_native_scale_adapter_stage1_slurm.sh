#!/usr/bin/env bash

# Eval-only launcher for the stage-1 adapter matrix.
# Each eval_checkpoint.py invocation evaluates both train and validation.
#
# Defaults submit normal and waveform-station roll eval for rt52-rt54 using
# full_model_last.pth. Existing non-empty outputs are skipped.
#
# Usage:
#   bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#
# Overrides:
#   ACTION=normal bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   ACTION=mismatch RT_LIST="53 54" bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   ALLOW_OVERWRITE=1 bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   DRY_RUN=1 bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"$WORKDIR/eval_checkpoint_slurm.sh"}
RT_LIST=${RT_LIST:-"52 53 54"}
ACTION=${ACTION:-all}
PERMUTATION=${PERMUTATION:-roll}
PERMUTATION_SEED=${PERMUTATION_SEED:-12345}
SLURM_TIME=${SLURM_TIME:-12:00:00}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
SKIP_MISSING_CHECKPOINT=${SKIP_MISSING_CHECKPOINT:-0}
DRY_RUN=${DRY_RUN:-0}

submitted_count=0
skipped_count=0

config_for_rt() {
    case "$1" in
        46) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt46_knet_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        52) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        53) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt53_knet_nlta_s_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        54) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt54_knet_nlta_m_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        *)
            echo "Unsupported rt id: $1" >&2
            return 1
            ;;
    esac
}

resolve_run_paths() {
    local rt=$1
    local weight_path
    local log_name

    RESOLVED_CONFIG="$CONFIG_DIR/$(config_for_rt "$rt")"
    if [[ ! -f "$RESOLVED_CONFIG" ]]; then
        echo "Config not found for rt${rt}: $RESOLVED_CONFIG" >&2
        return 1
    fi
    weight_path=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$RESOLVED_CONFIG")
    if [[ -z "$weight_path" || "$weight_path" == "/" || "$weight_path" == "." || "$weight_path" == ".." ]]; then
        echo "Unsafe weight_path for rt${rt}: '$weight_path'" >&2
        return 1
    fi
    case "$weight_path" in
        /*)
            RESOLVED_WEIGHT_DIR="$weight_path"
            log_name=$(basename "$weight_path")
            ;;
        *)
            RESOLVED_WEIGHT_DIR="$WORKDIR/${weight_path#./}"
            log_name=${weight_path#./}
            ;;
    esac
    RESOLVED_CHECKPOINT="$RESOLVED_WEIGHT_DIR/full_model_last.pth"
    RESOLVED_LOG_DIR="$WORKDIR/logs/$log_name"
}

checkpoint_available() {
    local rt=$1
    if [[ -f "$RESOLVED_CHECKPOINT" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN WARN] checkpoint is not visible: $RESOLVED_CHECKPOINT" >&2
        return 0
    fi
    echo "Missing full_model_last.pth for rt${rt}: $RESOLVED_CHECKPOINT" >&2
    if [[ "$SKIP_MISSING_CHECKPOINT" == "1" ]]; then
        skipped_count=$((skipped_count + 1))
        return 1
    fi
    exit 1
}

output_exists() {
    local txt_path=$1
    local npz_path=$2
    if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
        return 1
    fi
    if [[ -s "$txt_path" || -s "$npz_path" ]]; then
        echo "[INFO] existing output; skip: $txt_path"
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    return 1
}

submit_eval() {
    local rt=$1
    local mode=$2
    local suffix
    local output_txt
    local output_npz
    local job_name

    resolve_run_paths "$rt"
    checkpoint_available "$rt" || return 0
    if [[ "$mode" == "normal" ]]; then
        suffix=""
        job_name="team-nlta-s1-normal-rt${rt}"
    else
        suffix="_waveform_station_${PERMUTATION}"
        job_name="team-nlta-s1-mismatch-rt${rt}"
    fi
    output_txt="$RESOLVED_LOG_DIR/eval_results_last${suffix}.txt"
    output_npz="$RESOLVED_LOG_DIR/eval_results_last${suffix}.npz"
    if output_exists "$output_txt" "$output_npz"; then
        return 0
    fi

    echo "[INFO] submit rt${rt} ${mode}: $RESOLVED_CHECKPOINT"
    echo "[INFO] output: $output_txt"
    if [[ "$DRY_RUN" == "1" ]]; then
        submitted_count=$((submitted_count + 1))
        return 0
    fi
    if [[ "$mode" == "mismatch" ]]; then
        WORKDIR="$WORKDIR" \
        JOB_NAME="$job_name" \
        SLURM_TIME="$SLURM_TIME" \
        EVAL_CHECKPOINT="$RESOLVED_CHECKPOINT" \
        EVAL_OUTPUT_TXT="$output_txt" \
        EVAL_OUTPUT_NPZ="$output_npz" \
        bash "$EVAL_SCRIPT" "$RESOLVED_CONFIG" \
            --skip_single_station \
            --waveform_station_permutation "$PERMUTATION" \
            --waveform_station_permutation_seed "$PERMUTATION_SEED"
    else
        WORKDIR="$WORKDIR" \
        JOB_NAME="$job_name" \
        SLURM_TIME="$SLURM_TIME" \
        EVAL_CHECKPOINT="$RESOLVED_CHECKPOINT" \
        EVAL_OUTPUT_TXT="$output_txt" \
        EVAL_OUTPUT_NPZ="$output_npz" \
        bash "$EVAL_SCRIPT" "$RESOLVED_CONFIG" \
            --skip_single_station
    fi
    submitted_count=$((submitted_count + 1))
}

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "Eval launcher not found: $EVAL_SCRIPT" >&2
    exit 1
fi
case "$ACTION" in
    all|normal|mismatch) ;;
    *)
        echo "Unsupported ACTION='$ACTION'; expected all, normal, or mismatch." >&2
        exit 1
        ;;
esac

for rt in $RT_LIST; do
    if [[ "$ACTION" == "all" || "$ACTION" == "normal" ]]; then
        submit_eval "$rt" normal
    fi
    if [[ "$ACTION" == "all" || "$ACTION" == "mismatch" ]]; then
        submit_eval "$rt" mismatch
    fi
done

echo "[INFO] stage-1 eval submission complete: submitted=${submitted_count}, skipped=${skipped_count}, dry_run=${DRY_RUN}"
