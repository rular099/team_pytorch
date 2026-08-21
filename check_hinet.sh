#!/usr/bin/env bash
set -euo pipefail

YEAR=${YEAR:-2024}
HDF5_ROOT=${HDF5_ROOT:-'/run/media/zhangb/My Passport/knet_converted/origin_corrected_diting_vel_acc_vs30'}
DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-'/run/media/zhangb/My Passport/hinet_data'}
OUTPUT_DIR=${OUTPUT_DIR:-"$DOWNLOAD_ROOT/waveform_qc/$YEAR"}
MAX_PLOTS=${MAX_PLOTS:-10}

python tools/plot_hinet_accel_velocity_qc.py \
  --hdf5 "$HDF5_ROOT/$YEAR/japan_${YEAR}.hdf5" \
  --download-root "$DOWNLOAD_ROOT" \
  --show-candidate-picks \
  --max-plots "$MAX_PLOTS" \
  --output-dir "$OUTPUT_DIR"
