#!/usr/bin/env bash

# Precompute the shared KNET-only fixed-time DPK prior cache for rt44-rt51.
#
# The cache is generated from the rt44 KNET-only config because rt44-rt51 share
# the same KNET-only data split, realtime fixed-time sampling, and DPK prior
# source. The resulting train/dev HDF5 files are referenced by all rt44-rt51
# KNET-only configs.
#
# Usage on the cluster:
#   bash tools/precompute_rt44_rt51_knet_dpk_priors_slurm.sh
#
# Useful overrides:
#   SPLITS="train dev" SLURM_TIME=12:00:00 bash tools/precompute_rt44_rt51_knet_dpk_priors_slurm.sh
#   STATION_BATCH_SIZE=8 bash tools/precompute_rt44_rt51_knet_dpk_priors_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
CONFIG_DIR=${CONFIG_DIR:-"$WORKDIR/pga_configs"}
PRECOMPUTE_SCRIPT=${PRECOMPUTE_SCRIPT:-"$WORKDIR/tools/precompute_dpk_priors_slurm.sh"}
SPLITS=${SPLITS:-"train dev"}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-team-dpk-knet}

CACHE_CONFIG_NAME=${CACHE_CONFIG_NAME:-transformer_japan_overfit_pga15_stage2_512_rt44_knet_cached_dpk_event_temporal_residual_scale0_chaosuan.json}
CACHE_STEM=${CACHE_CONFIG_NAME%.json}
CACHE_CONFIG="$CONFIG_DIR/$CACHE_CONFIG_NAME"

if [[ ! -f "$CACHE_CONFIG" ]]; then
    echo "KNET cache config not found: $CACHE_CONFIG" >&2
    exit 1
fi
if [[ ! -f "$PRECOMPUTE_SCRIPT" ]]; then
    echo "Precompute launcher not found: $PRECOMPUTE_SCRIPT" >&2
    exit 1
fi

for split in $SPLITS; do
    case "$split" in
        train|dev|test) ;;
        *)
            echo "Unsupported split: $split" >&2
            exit 1
            ;;
    esac
    out_dir="$WORKDIR/dpk_prior_cache/${CACHE_STEM}_${split}"
    echo "[INFO] submitting KNET DPK prior precompute split=$split output=$out_dir"
    WORKDIR="$WORKDIR" \
    SPLIT="$split" \
    JOB_NAME="${JOB_NAME_PREFIX}-${split}" \
    bash "$PRECOMPUTE_SCRIPT" "$CACHE_CONFIG" "$out_dir"
done
