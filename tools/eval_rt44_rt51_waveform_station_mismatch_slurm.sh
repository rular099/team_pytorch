#!/usr/bin/env bash

# Submit waveform-station mismatch evaluation for rt44-rt51.
# Outputs keep the same layout as eval_checkpoint_slurm.sh:
#   logs/<weight_path>/eval_results_last_waveform_station_<mode>.txt/.npz
#   logs/<weight_path>/eval_results_best_waveform_station_<mode>.txt/.npz

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"$WORKDIR/eval_checkpoint_slurm.sh"}
RT_LIST=${RT_LIST:-"44 45 46 47 48 49 50 51"}
PERMUTATION=${PERMUTATION:-roll}
PERMUTATION_SEED=${PERMUTATION_SEED:-12345}
EVAL_OUTPUT_SUFFIX=${EVAL_OUTPUT_SUFFIX:-"_waveform_station_${PERMUTATION}"}
SLURM_TIME=${SLURM_TIME:-12:00:00}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-eval-mismatch}

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

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "Eval launcher not found: $EVAL_SCRIPT" >&2
    exit 1
fi

for rt in $RT_LIST; do
    cfg_name=$(config_for_rt "$rt")
    cfg_path="$CONFIG_DIR/$cfg_name"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found: $cfg_path" >&2
        exit 1
    fi
    echo "[INFO] submitting rt${rt} waveform-station mismatch eval"
    WORKDIR="$WORKDIR" \
    JOB_NAME="${JOB_NAME_PREFIX}-rt${rt}" \
    SLURM_TIME="$SLURM_TIME" \
    EVAL_OUTPUT_SUFFIX="$EVAL_OUTPUT_SUFFIX" \
    bash "$EVAL_SCRIPT" "$cfg_path" \
        --waveform_station_permutation "$PERMUTATION" \
        --waveform_station_permutation_seed "$PERMUTATION_SEED" \
        --skip_single_station
done
