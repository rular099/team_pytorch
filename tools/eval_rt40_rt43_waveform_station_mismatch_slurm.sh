#!/usr/bin/env bash

# Submit waveform-station mismatch evaluation for rt40-rt43.
# Each job uses eval_checkpoint_slurm.sh, so outputs keep the same layout as
# train_light_slurm.sh RUN_EVAL:
#   logs/<weight_path>/eval_results_last_waveform_station_<mode>.txt/.npz
#   logs/<weight_path>/eval_results_best_waveform_station_<mode>.txt/.npz
#
# Usage on the cluster:
#   bash tools/eval_rt40_rt43_waveform_station_mismatch_slurm.sh
#
# Useful overrides:
#   RT_LIST="40 41" PERMUTATION=random PERMUTATION_SEED=2026 bash tools/eval_rt40_rt43_waveform_station_mismatch_slurm.sh
#   AUTO_SBATCH=0 bash tools/eval_rt40_rt43_waveform_station_mismatch_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"$WORKDIR/eval_checkpoint_slurm.sh"}
RT_LIST=${RT_LIST:-"40 41 42 43"}
PERMUTATION=${PERMUTATION:-roll}
PERMUTATION_SEED=${PERMUTATION_SEED:-12345}
EVAL_OUTPUT_SUFFIX=${EVAL_OUTPUT_SUFFIX:-"_waveform_station_${PERMUTATION}"}
SLURM_TIME=${SLURM_TIME:-12:00:00}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-eval-mismatch}

config_for_rt() {
    case "$1" in
        40) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt40_dpk_event_station_pool_scale0_chaosuan.json" ;;
        41) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt41_dpk_event_station_pool_scale2_chaosuan.json" ;;
        42) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt42_dpk_event_station_pool_scale4_chaosuan.json" ;;
        43) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt43_dpk_event_station_pool_scale8_chaosuan.json" ;;
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
    echo "[INFO] config: $cfg_path"
    echo "[INFO] permutation=$PERMUTATION seed=$PERMUTATION_SEED suffix=$EVAL_OUTPUT_SUFFIX"
    WORKDIR="$WORKDIR" \
    JOB_NAME="${JOB_NAME_PREFIX}-rt${rt}" \
    SLURM_TIME="$SLURM_TIME" \
    EVAL_OUTPUT_SUFFIX="$EVAL_OUTPUT_SUFFIX" \
    bash "$EVAL_SCRIPT" "$cfg_path" \
        --waveform_station_permutation "$PERMUTATION" \
        --waveform_station_permutation_seed "$PERMUTATION_SEED" \
        --skip_single_station
done
