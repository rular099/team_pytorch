#!/usr/bin/env bash

# Resume rt44-rt51 KNET-only training after a Slurm timeout.
#
# This launcher only resumes from existing full-model checkpoints. It never
# deletes weight directories and never silently starts a fresh run.
#
# Usage on the cluster:
#   bash tools/resume_rt44_rt51_knet_slurm.sh
#
# Useful overrides:
#   RT_LIST="51" bash tools/resume_rt44_rt51_knet_slurm.sh
#   SLURM_TIME=3-00:00:00 bash tools/resume_rt44_rt51_knet_slurm.sh
#   RUN_EVAL=1 bash tools/resume_rt44_rt51_knet_slurm.sh
#   SKIP_MISSING_CHECKPOINT=1 bash tools/resume_rt44_rt51_knet_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$WORKDIR/train_light_slurm.sh"}
RT_LIST=${RT_LIST:-"44 45 46 47 48 49 50 51"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-rt-knet-resume}
RUN_EVAL=${RUN_EVAL:-0}
SLURM_TIME=${SLURM_TIME:-3-00:00:00}
SKIP_MISSING_CHECKPOINT=${SKIP_MISSING_CHECKPOINT:-0}

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

resume_spec_for_weight_dir() {
    local weight_dir=$1
    if [[ -f "$weight_dir/full_model_last.pth" ]]; then
        printf '%s\n' last
        return 0
    fi
    if [[ -f "$weight_dir/full_model_best.pth" ]]; then
        printf '%s\n' best
        return 0
    fi
    if [[ -f "$weight_dir/full_model_init.pth" ]]; then
        printf '%s\n' init
        return 0
    fi
    return 1
}

if [[ "${RESET_WEIGHT_PATH:-0}" == "1" ]]; then
    echo "Refusing RESET_WEIGHT_PATH=1 in resume launcher." >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Train launcher not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

for rt in $RT_LIST; do
    cfg_name=$(config_for_rt "$rt")
    cfg_path="$CONFIG_DIR/$cfg_name"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found: $cfg_path" >&2
        exit 1
    fi

    weight_dir=$(weight_dir_for_config "$cfg_path")
    if ! resume_spec=$(resume_spec_for_weight_dir "$weight_dir"); then
        echo "No full-model checkpoint found for rt${rt}: $weight_dir" >&2
        if [[ "$SKIP_MISSING_CHECKPOINT" == "1" ]]; then
            echo "[WARN] skipping rt${rt}; set SKIP_MISSING_CHECKPOINT=0 to fail instead." >&2
            continue
        fi
        exit 1
    fi

    echo "[INFO] resuming rt${rt} KNET-only from ${resume_spec}: $cfg_path"
    WORKDIR="$WORKDIR" \
    JOB_NAME="${JOB_NAME_PREFIX}${rt}" \
    RUN_EVAL="$RUN_EVAL" \
    SLURM_TIME="$SLURM_TIME" \
    RESET_WEIGHT_PATH=0 \
    bash "$TRAIN_SCRIPT" "$cfg_path" --resume_full_model "$resume_spec"
done
