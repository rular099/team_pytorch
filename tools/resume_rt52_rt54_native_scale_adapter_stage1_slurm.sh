#!/usr/bin/env bash

# Continue timed-out rt52-rt54 stage-1 runs from full_model_last.pth.
#
# The checkpoint stores the model, optimizer, LR scheduler, and the number of
# completed epochs. Work from an interrupted partial epoch is intentionally
# discarded; training restarts at the next epoch after the saved checkpoint.
#
# Usage:
#   bash tools/resume_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#
# Common overrides:
#   RT_LIST="52" bash tools/resume_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   SLURM_TIME=72:00:00 bash tools/resume_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#   DRY_RUN=1 bash tools/resume_rt52_rt54_native_scale_adapter_stage1_slurm.sh
#
# Safety controls:
#   SKIP_MISSING_CHECKPOINT=1  Skip runs without full_model_last.pth.
#   ALLOW_ACTIVE_JOB=1        Submit even if a same-name Slurm job is active.

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$WORKDIR/train_light_slurm.sh"}
RT_LIST=${RT_LIST:-"52 53 54"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-nlta-s1-rt}
SLURM_TIME=${SLURM_TIME:-48:00:00}
RUN_EVAL=${RUN_EVAL:-0}
SKIP_MISSING_CHECKPOINT=${SKIP_MISSING_CHECKPOINT:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
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

checkpoint_for_rt() {
    local rt=$1
    local cfg_path
    local weight_dir
    cfg_path="$CONFIG_DIR/$(config_for_rt "$rt")"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Config not found for rt${rt}: $cfg_path" >&2
        return 1
    fi
    weight_dir=$(weight_dir_for_config "$cfg_path")
    printf '%s\n' "$weight_dir/full_model_last.pth"
}

active_job_for_name() {
    local job_name=$1
    local owner
    if ! command -v squeue >/dev/null 2>&1; then
        return 1
    fi
    owner=$(id -un)
    squeue --noheader --user "$owner" --name "$job_name" --format='%A %T' 2>/dev/null |
        awk 'NF {print; found=1} END {exit found ? 0 : 1}'
}

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Train launcher not found: $TRAIN_SCRIPT" >&2
    exit 1
fi
if [[ "${RESET_WEIGHT_PATH:-0}" == "1" ]]; then
    echo "RESET_WEIGHT_PATH=1 is forbidden for continuation jobs." >&2
    exit 1
fi

# First pass: fail before submitting anything if a required checkpoint is absent.
for rt in $RT_LIST; do
    checkpoint=$(checkpoint_for_rt "$rt")
    if [[ ! -s "$checkpoint" ]]; then
        if [[ "$SKIP_MISSING_CHECKPOINT" == "1" ]]; then
            echo "[WARN] rt${rt}: missing checkpoint; it will be skipped: $checkpoint" >&2
            continue
        fi
        echo "Missing or empty continuation checkpoint for rt${rt}: $checkpoint" >&2
        echo "Use RT_LIST to select only runs with full_model_last.pth." >&2
        exit 1
    fi
    echo "[OK] rt${rt}: continuation checkpoint: $checkpoint"
done

submitted_count=0
skipped_count=0
for rt in $RT_LIST; do
    cfg_path="$CONFIG_DIR/$(config_for_rt "$rt")"
    checkpoint=$(checkpoint_for_rt "$rt")
    if [[ ! -s "$checkpoint" ]]; then
        skipped_count=$((skipped_count + 1))
        continue
    fi

    job_name="${JOB_NAME_PREFIX}${rt}"
    active_job=""
    if active_job=$(active_job_for_name "$job_name"); then
        if [[ "$ALLOW_ACTIVE_JOB" != "1" ]]; then
            echo "[WARN] rt${rt}: active job already exists; skipping: $active_job" >&2
            skipped_count=$((skipped_count + 1))
            continue
        fi
        echo "[WARN] rt${rt}: submitting despite active job: $active_job" >&2
    fi

    echo "[INFO] rt${rt}: resume from full_model_last.pth"
    echo "[INFO] rt${rt}: walltime=$SLURM_TIME, config=$cfg_path"
    if [[ "$DRY_RUN" == "1" ]]; then
        submitted_count=$((submitted_count + 1))
        continue
    fi

    WORKDIR="$WORKDIR" \
    JOB_NAME="$job_name" \
    SLURM_TIME="$SLURM_TIME" \
    RUN_EVAL="$RUN_EVAL" \
    RESET_WEIGHT_PATH=0 \
    AUTO_SBATCH=1 \
    bash "$TRAIN_SCRIPT" "$cfg_path" \
        --resume_full_model last
    submitted_count=$((submitted_count + 1))
done

echo "[INFO] continuation submission complete: submitted=${submitted_count}, skipped=${skipped_count}, dry_run=${DRY_RUN}"
