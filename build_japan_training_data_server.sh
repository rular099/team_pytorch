#!/usr/bin/env bash
set -euo pipefail

YEAR=${YEAR:-2024}
WAVEFORM_ROOT=${WAVEFORM_ROOT:-/public/home/zhangbei/work_dir/zhangbei/japan_knet}
CONVERTED_ROOT=${CONVERTED_ROOT:-/public/home/zhangbei/work_dir/zhangbei/japan_knet_converted}
OUTPUT_VARIANT=${OUTPUT_VARIANT:-legacy_diting_vel}
OUTPUT_DIR=${OUTPUT_DIR:-$CONVERTED_ROOT/$OUTPUT_VARIANT/$YEAR}
VS30_CSV=${VS30_CSV:-resources/vs30/japan_vs30_jshis_station_cache.csv}
VS30_MAX_DISTANCE_KM=${VS30_MAX_DISTANCE_KM:-1.0}

python ./build_japan_training_data.py \
--waveform_root "$WAVEFORM_ROOT" \
--output_dir "$OUTPUT_DIR" \
--year "$YEAR" \
--pick_mode travel_time \
--travel_time_model jma2001a \
--jma_search_margin_seconds 10 \
--run_diting \
--diting_p_th 0.1 \
--diting_s_th 0.1 \
--diting_d_th 0.1 \
--ditingbench_root /public/home/zhangbei/work_dir/zhangbei/xiaozhuowei/ditingbench \
--diting_weights /opt/zb/ckpt/1200m/dpk/720w_sft/model-4-latest.pth \
--diting_device cuda:0 \
--final_pick diting_vel \
--n_diagnostic_plots 48 \
--jma_search_margin_seconds 10.0 \
--jma_search_margin_per_km 0.03 \
--jma_search_margin_max_seconds 60.0 \
--fallback_search_half_window_seconds 60.0 \
--vs30_csv "$VS30_CSV" \
--vs30_max_distance_km "$VS30_MAX_DISTANCE_KM" \
--require_vs30 \
--overwrite
