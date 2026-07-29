#!/usr/bin/env bash

# Submit rt44-rt51 KNET-only prior-filtered token experiments.
#
# Run tools/precompute_rt44_rt51_knet_dpk_priors_slurm.sh first and verify cache
# coverage before submitting training. This launcher deliberately does not set
# RESET_WEIGHT_PATH=1.
#
# Usage on the cluster:
#   bash tools/run_rt44_rt51_knet_slurm.sh
#
# Useful overrides:
#   RT_LIST="46 50 51" bash tools/run_rt44_rt51_knet_slurm.sh
#   RUN_EVAL=0 SLURM_TIME=3-00:00:00 bash tools/run_rt44_rt51_knet_slurm.sh
#   AUTO_RESUME_FULL_MODEL=0 bash tools/run_rt44_rt51_knet_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$WORKDIR/train_light_slurm.sh"}
RT_LIST=${RT_LIST:-"44 45 46 47 48 49 50 51"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-rt-knet}
AUTO_RESUME_FULL_MODEL=${AUTO_RESUME_FULL_MODEL:-1}

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

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Train launcher not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

weight_dir_for_config() {
    local cfg_path=$1
    local weight_path
    weight_path=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["training_params"]["weight_path"])' "$cfg_path")
    if [[ -z "$weight_path" || "$weight_path" == "/" || "$weight_path" == "." || "$weight_path" == ".." ]]; then
        echo "Unsafe weight_path in config: '$weight_path'" >&2
        return 1
    fi
    case "$weight_path" in
        /*) printf '%s\n' "$weight_path" ;;
        *) printf '%s\n' "$WORKDIR/$weight_path" ;;
    esac
}

resume_args_for_weight_dir() {
    local weight_dir=$1
    if [[ "${RESET_WEIGHT_PATH:-0}" == "1" || "$AUTO_RESUME_FULL_MODEL" != "1" ]]; then
        return 0
    fi
    if [[ -f "$weight_dir/full_model_last.pth" ]]; then
        printf '%s\n' "--resume_full_model last"
        return 0
    fi
    if [[ -f "$weight_dir/full_model_best.pth" ]]; then
        printf '%s\n' "--resume_full_model best"
        return 0
    fi
    if [[ -f "$weight_dir/full_model_init.pth" ]]; then
        printf '%s\n' "--resume_full_model init"
        return 0
    fi
    if [[ -d "$weight_dir" ]]; then
        shopt -s nullglob
        local entries=("$weight_dir"/*)
        shopt -u nullglob
        if (( ${#entries[@]} == 0 )); then
            return 0
        fi
        if (( ${#entries[@]} == 1 )) && [[ "$(basename "${entries[0]}")" == "config.json" ]]; then
            return 0
        fi
        echo "Weight dir is non-empty but no full_model checkpoint is available: $weight_dir" >&2
        echo "Existing entries:" >&2
        find "$weight_dir" -maxdepth 1 -mindepth 1 -printf '  %f\n' >&2
        echo "Move this directory aside or intentionally rerun with RESET_WEIGHT_PATH=1 after confirming no useful checkpoint exists." >&2
        return 1
    fi
}

for rt in $RT_LIST; do
    cfg_name=$(config_for_rt "$rt")
    cfg_path="$CONFIG_DIR/$cfg_name"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found: $cfg_path" >&2
        exit 1
    fi
    weight_dir=$(weight_dir_for_config "$cfg_path")
    train_extra_args=()
    resume_arg=$(resume_args_for_weight_dir "$weight_dir")
    if [[ -n "$resume_arg" ]]; then
        read -r -a train_extra_args <<< "$resume_arg"
        echo "[INFO] existing checkpoint detected for rt${rt}; using: ${train_extra_args[*]}"
    fi
    echo "[INFO] submitting rt${rt} KNET-only: $cfg_path"
    if (( ${#train_extra_args[@]} > 0 )); then
        WORKDIR="$WORKDIR" \
        JOB_NAME="${JOB_NAME_PREFIX}${rt}" \
        RESET_WEIGHT_PATH="${RESET_WEIGHT_PATH:-0}" \
        bash "$TRAIN_SCRIPT" "$cfg_path" "${train_extra_args[@]}"
    else
        WORKDIR="$WORKDIR" \
        JOB_NAME="${JOB_NAME_PREFIX}${rt}" \
        RESET_WEIGHT_PATH="${RESET_WEIGHT_PATH:-0}" \
        bash "$TRAIN_SCRIPT" "$cfg_path"
    fi
done
