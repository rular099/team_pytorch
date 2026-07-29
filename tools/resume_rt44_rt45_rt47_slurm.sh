#!/usr/bin/env bash

# Resume timed-out rt44/rt45/rt47 runs from full_model_last.pth.
#
# Usage on the cluster:
#   bash tools/resume_rt44_rt45_rt47_slurm.sh
#
# Useful overrides:
#   RT_LIST="44 47" bash tools/resume_rt44_rt45_rt47_slurm.sh
#   SLURM_TIME=3-00:00:00 bash tools/resume_rt44_rt45_rt47_slurm.sh
#   RUN_EVAL=0 bash tools/resume_rt44_rt45_rt47_slurm.sh
#
# This script never resets or deletes weight directories.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$WORKDIR/train_light_slurm.sh"}
RT_LIST=${RT_LIST:-"44 45 47"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-resume-rt}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-last}

config_for_rt() {
    case "$1" in
        44) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt44_cached_dpk_event_temporal_residual_scale0_chaosuan.json" ;;
        45) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt45_cached_dpk_event_temporal_residual_scale2_chaosuan.json" ;;
        47) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt47_cached_dpk_event_temporal_residual_scale8_chaosuan.json" ;;
        *)
            echo "Unsupported rt id for this resume script: $1" >&2
            return 1
            ;;
    esac
}

weight_path_for_rt() {
    case "$1" in
        44) printf '%s\n' "weights_japan_overfit_pga15_stage2_512_rt44_cached_dpk_event_temporal_residual_scale0" ;;
        45) printf '%s\n' "weights_japan_overfit_pga15_stage2_512_rt45_cached_dpk_event_temporal_residual_scale2" ;;
        47) printf '%s\n' "weights_japan_overfit_pga15_stage2_512_rt47_cached_dpk_event_temporal_residual_scale8" ;;
        *)
            echo "Unsupported rt id for this resume script: $1" >&2
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
    weight_path=$(weight_path_for_rt "$rt")
    weight_dir="$WORKDIR/$weight_path"
    last_ckpt="$weight_dir/full_model_last.pth"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found: $cfg_path" >&2
        exit 1
    fi
    if [[ "$RESUME_CHECKPOINT" == "last" && ! -f "$last_ckpt" ]]; then
        echo "Resume checkpoint not found for rt${rt}: $last_ckpt" >&2
        echo "Refusing to submit a job that would fail or restart from scratch." >&2
        exit 1
    fi
    echo "[INFO] submitting rt${rt} resume from --resume_full_model ${RESUME_CHECKPOINT}"
    echo "[INFO] config: $cfg_path"
    echo "[INFO] checkpoint: $last_ckpt"
    WORKDIR="$WORKDIR" \
    JOB_NAME="${JOB_NAME_PREFIX}${rt}" \
    RESET_WEIGHT_PATH=0 \
    bash "$TRAIN_SCRIPT" "$cfg_path" --resume_full_model "$RESUME_CHECKPOINT"
done
