#!/usr/bin/env bash

# Submit rt44-rt51 prior-filtered token experiments.
#
# These experiments use the fixed-time cached DPK event priors generated from
# rt34_fixedtime. The launcher deliberately does not set RESET_WEIGHT_PATH=1.
#
# Usage on the cluster:
#   bash tools/run_rt44_rt51_slurm.sh
#
# Useful overrides:
#   RT_LIST="46 50 51" bash tools/run_rt44_rt51_slurm.sh
#   RUN_EVAL=0 SLURM_TIME=3-00:00:00 bash tools/run_rt44_rt51_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$WORKDIR/train_light_slurm.sh"}
RT_LIST=${RT_LIST:-"44 45 46 47 48 49 50 51"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-rt}

config_for_rt() {
    case "$1" in
        44) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt44_cached_dpk_event_temporal_residual_scale0_chaosuan.json" ;;
        45) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt45_cached_dpk_event_temporal_residual_scale2_chaosuan.json" ;;
        46) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt46_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        47) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt47_cached_dpk_event_temporal_residual_scale8_chaosuan.json" ;;
        48) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt48_cached_dpk_event_station_pool_temporal_residual_scale4_chaosuan.json" ;;
        49) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt49_cached_dpk_event_layerwise_temporal_readout_scale4_chaosuan.json" ;;
        50) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt50_cached_dpk_event_temporal_residual_independent_scale4_chaosuan.json" ;;
        51) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt51_cached_dpk_event_temporal_residual_stationroll_control_scale4_chaosuan.json" ;;
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

for rt in $RT_LIST; do
    cfg_name=$(config_for_rt "$rt")
    cfg_path="$CONFIG_DIR/$cfg_name"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found: $cfg_path" >&2
        exit 1
    fi
    echo "[INFO] submitting rt${rt}: $cfg_path"
    WORKDIR="$WORKDIR" \
    JOB_NAME="${JOB_NAME_PREFIX}${rt}" \
    RESET_WEIGHT_PATH="${RESET_WEIGHT_PATH:-0}" \
    bash "$TRAIN_SCRIPT" "$cfg_path"
done
