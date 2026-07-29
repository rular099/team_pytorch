#!/usr/bin/env bash

# Submit stage-1 padding-mask / native-scale adapter experiments.
#
# E0 is the existing KNET-only rt46 run and is intentionally not resubmitted.
# This script submits:
#   rt52: legacy adapter + explicit padding mask
#   rt53: NLTA-S + explicit padding mask
#   rt54: NLTA-M + explicit padding mask
#
# Usage:
#   bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#
# Overrides:
#   RT_LIST="52 53" bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   AUTO_RESUME_FULL_MODEL=0 bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   DRY_RUN=1 bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$WORKDIR/train_light_slurm.sh"}
RT_LIST=${RT_LIST:-"52 53 54"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-nlta-s1-rt}
AUTO_RESUME_FULL_MODEL=${AUTO_RESUME_FULL_MODEL:-1}
RUN_EVAL=${RUN_EVAL:-0}
DRY_RUN=${DRY_RUN:-0}

config_for_rt() {
    case "$1" in
        52) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        53) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt53_knet_nlta_s_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
        54) printf '%s\n' "transformer_japan_overfit_pga15_stage2_512_rt54_knet_nlta_m_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json" ;;
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
        *) printf '%s\n' "$WORKDIR/${weight_path#./}" ;;
    esac
}

validate_config() {
    local rt=$1
    local cfg_path=$2
    python -c '
import json
import sys

rt = int(sys.argv[1])
cfg = json.load(open(sys.argv[2]))
mp = cfg["model_params"]
tp = cfg["training_params"]
gp = tp["generator_params"][0]
expected = {
    52: ("legacy", None, None),
    53: ("nlta", 256, 96),
    54: ("nlta", 384, 128),
}
adapter, x_width, side_width = expected[rt]
assert mp.get("diting_station_adapter", "legacy") == adapter
assert gp.get("emit_waveform_padding_mask") is True
assert tp.get("station_filter") == "knet"
assert mp.get("station_token_weight_mode") == "none"
assert mp.get("temporal_token_weight_mode") == "cached_dpk_event"
if adapter == "nlta":
    assert mp.get("diting_station_metadata_mode", "none") == "none"
    assert mp["diting_nlta_x_channels"] == x_width
    assert mp["diting_nlta_side_channels"] == side_width
print("[OK] rt{}: adapter={}, mask=true, weight_path={}".format(
    rt, adapter, tp["weight_path"]
))
' "$rt" "$cfg_path"
}

resume_arg_for_weight_dir() {
    local weight_dir=$1
    if [[ "${RESET_WEIGHT_PATH:-0}" == "1" ]]; then
        echo "RESET_WEIGHT_PATH=1 is not allowed by this stage-1 launcher." >&2
        return 1
    fi
    if [[ "$AUTO_RESUME_FULL_MODEL" != "1" ]]; then
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
        echo "Weight directory is non-empty but has no resumable full-model checkpoint: $weight_dir" >&2
        find "$weight_dir" -maxdepth 1 -mindepth 1 -printf '  %f\n' >&2
        return 1
    fi
}

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Train launcher not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

for rt in $RT_LIST; do
    cfg_path="$CONFIG_DIR/$(config_for_rt "$rt")"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found: $cfg_path" >&2
        exit 1
    fi
    validate_config "$rt" "$cfg_path"
    weight_dir=$(weight_dir_for_config "$cfg_path")
    resume_arg=$(resume_arg_for_weight_dir "$weight_dir")
    resume_args=()
    if [[ -n "$resume_arg" ]]; then
        read -r -a resume_args <<< "$resume_arg"
    fi
    echo "[INFO] rt${rt} config: $cfg_path"
    echo "[INFO] rt${rt} weights: $weight_dir"
    if [[ -n "$resume_arg" ]]; then
        echo "[INFO] rt${rt} resume: $resume_arg"
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi
    WORKDIR="$WORKDIR" \
    JOB_NAME="${JOB_NAME_PREFIX}${rt}" \
    RUN_EVAL="$RUN_EVAL" \
    RESET_WEIGHT_PATH=0 \
    bash "$TRAIN_SCRIPT" "$cfg_path" "${resume_args[@]}"
done

echo "[INFO] stage-1 training submission complete; E0 remains existing rt46."
