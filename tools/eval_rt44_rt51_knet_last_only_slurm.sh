#!/usr/bin/env bash

# Submit eval-only jobs for the completed KNET-only rt44-rt51 experiments.
#
# This launcher never enters train_light.py and evaluates only
# full_model_last.pth. By default it:
#   1. fills missing normal evals for rt45, rt48, and rt49;
#   2. runs waveform-station roll mismatch evals for rt44-rt51.
#
# Usage on the cluster:
#   bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#
# Useful overrides:
#   ACTION=normal bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#   ACTION=mismatch bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#   NORMAL_RT_LIST="45 48 49" bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#   MISMATCH_RT_LIST="46 51" bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#   SLURM_TIME=1-00:00:00 bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#   DRY_RUN=1 bash tools/eval_rt44_rt51_knet_last_only_slurm.sh
#
# Existing non-empty result files are skipped. Set ALLOW_OVERWRITE=1 only
# after confirming that replacing those eval outputs is intentional.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"$WORKDIR/eval_checkpoint_slurm.sh"}

ACTION=${ACTION:-all}
NORMAL_RT_LIST=${NORMAL_RT_LIST:-"45 48 49"}
MISMATCH_RT_LIST=${MISMATCH_RT_LIST:-"44 45 46 47 48 49 50 51"}
PERMUTATION=${PERMUTATION:-roll}
PERMUTATION_SEED=${PERMUTATION_SEED:-12345}
SLURM_TIME=${SLURM_TIME:-12:00:00}
NORMAL_JOB_NAME_PREFIX=${NORMAL_JOB_NAME_PREFIX:-team-eval-knet-normal}
MISMATCH_JOB_NAME_PREFIX=${MISMATCH_JOB_NAME_PREFIX:-team-eval-knet-mismatch}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
SKIP_MISSING_CHECKPOINT=${SKIP_MISSING_CHECKPOINT:-0}
DRY_RUN=${DRY_RUN:-0}

submitted_count=0
skipped_count=0

config_for_rt() {
    case "$1" in
        44) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt44_knet_cached_dpk_event_temporal_residual_scale0_chaosuan.json" ;;
        45) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt45_knet_cached_dpk_event_temporal_residual_scale2_chaosuan.json" ;;
        46) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt46_knet_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        47) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt47_knet_cached_dpk_event_temporal_residual_scale8_chaosuan.json" ;;
        48) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt48_knet_cached_dpk_event_station_pool_temporal_residual_scale4_chaosuan.json" ;;
        49) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt49_knet_cached_dpk_event_layerwise_temporal_readout_scale4_chaosuan.json" ;;
        50) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt50_knet_cached_dpk_event_temporal_residual_independent_scale4_chaosuan.json" ;;
        51) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt51_knet_cached_dpk_event_temporal_residual_stationroll_control_scale4_chaosuan.json" ;;
        *)
            echo "Unsupported rt id: $1" >&2
            return 1
            ;;
    esac
}

resolve_run_paths() {
    local rt=$1
    local cfg_name
    local cfg_path
    local weight_path
    local weight_log_name

    cfg_name=$(config_for_rt "$rt")
    cfg_path="$CONFIG_DIR/$cfg_name"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found for rt${rt}: $cfg_path" >&2
        return 1
    fi

    weight_path=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$cfg_path")
    if [[ -z "$weight_path" || "$weight_path" == "/" || "$weight_path" == "." || "$weight_path" == ".." ]]; then
        echo "Unsafe weight_path for rt${rt}: '$weight_path'" >&2
        return 1
    fi

    case "$weight_path" in
        /*)
            RESOLVED_WEIGHT_DIR="$weight_path"
            weight_log_name=$(basename "$weight_path")
            ;;
        *)
            RESOLVED_WEIGHT_DIR="$WORKDIR/${weight_path#./}"
            weight_log_name=${weight_path#./}
            ;;
    esac

    RESOLVED_CONFIG="$cfg_path"
    RESOLVED_CHECKPOINT="$RESOLVED_WEIGHT_DIR/full_model_last.pth"
    RESOLVED_LOG_DIR="$WORKDIR/logs/$weight_log_name"
}

check_checkpoint() {
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
        echo "[WARN] skipping rt${rt}" >&2
        skipped_count=$((skipped_count + 1))
        return 1
    fi
    exit 1
}

should_skip_output() {
    local txt_path=$1
    local npz_path=$2
    if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
        return 1
    fi
    if [[ -s "$txt_path" || -s "$npz_path" ]]; then
        echo "[INFO] existing output detected; skipping:"
        echo "       $txt_path"
        echo "       $npz_path"
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    return 1
}

submit_normal_eval() {
    local rt=$1
    local output_txt
    local output_npz

    resolve_run_paths "$rt"
    check_checkpoint "$rt" || return 0

    output_txt="$RESOLVED_LOG_DIR/eval_results_last.txt"
    output_npz="$RESOLVED_LOG_DIR/eval_results_last.npz"
    if should_skip_output "$output_txt" "$output_npz"; then
        return 0
    fi

    echo "[INFO] submitting rt${rt} KNET normal eval, last checkpoint only"
    echo "       checkpoint: $RESOLVED_CHECKPOINT"
    echo "       output:     $output_txt"
    if [[ "$DRY_RUN" == "1" ]]; then
        submitted_count=$((submitted_count + 1))
        return 0
    fi

    WORKDIR="$WORKDIR" \
    JOB_NAME="${NORMAL_JOB_NAME_PREFIX}-rt${rt}" \
    SLURM_TIME="$SLURM_TIME" \
    EVAL_CHECKPOINT="$RESOLVED_CHECKPOINT" \
    EVAL_OUTPUT_TXT="$output_txt" \
    EVAL_OUTPUT_NPZ="$output_npz" \
    bash "$EVAL_SCRIPT" "$RESOLVED_CONFIG" \
        --skip_single_station
    submitted_count=$((submitted_count + 1))
}

submit_mismatch_eval() {
    local rt=$1
    local output_suffix
    local output_txt
    local output_npz

    resolve_run_paths "$rt"
    check_checkpoint "$rt" || return 0

    output_suffix="_waveform_station_${PERMUTATION}"
    output_txt="$RESOLVED_LOG_DIR/eval_results_last${output_suffix}.txt"
    output_npz="$RESOLVED_LOG_DIR/eval_results_last${output_suffix}.npz"
    if should_skip_output "$output_txt" "$output_npz"; then
        return 0
    fi

    echo "[INFO] submitting rt${rt} KNET waveform-station ${PERMUTATION} eval, last checkpoint only"
    echo "       checkpoint: $RESOLVED_CHECKPOINT"
    echo "       output:     $output_txt"
    if [[ "$DRY_RUN" == "1" ]]; then
        submitted_count=$((submitted_count + 1))
        return 0
    fi

    WORKDIR="$WORKDIR" \
    JOB_NAME="${MISMATCH_JOB_NAME_PREFIX}-rt${rt}" \
    SLURM_TIME="$SLURM_TIME" \
    EVAL_CHECKPOINT="$RESOLVED_CHECKPOINT" \
    EVAL_OUTPUT_TXT="$output_txt" \
    EVAL_OUTPUT_NPZ="$output_npz" \
    bash "$EVAL_SCRIPT" "$RESOLVED_CONFIG" \
        --waveform_station_permutation "$PERMUTATION" \
        --waveform_station_permutation_seed "$PERMUTATION_SEED" \
        --skip_single_station
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

if [[ "$ACTION" == "all" || "$ACTION" == "normal" ]]; then
    for rt in $NORMAL_RT_LIST; do
        submit_normal_eval "$rt"
    done
fi

if [[ "$ACTION" == "all" || "$ACTION" == "mismatch" ]]; then
    for rt in $MISMATCH_RT_LIST; do
        submit_mismatch_eval "$rt"
    done
fi

echo "[INFO] eval-only submission complete: submitted=${submitted_count}, skipped=${skipped_count}, dry_run=${DRY_RUN}"
