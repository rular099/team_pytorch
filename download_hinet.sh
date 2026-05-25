#!/bin/bash
python tools/download_hinet_velocity.py \
  --hdf5 '/run/media/zhangb/My Passport/knet_converted/2024/japan_2024.hdf5' \
  --origin-corrections jma_origin_corrections/japan_2024_origin_corrections.csv \
  --mode smoketest \
  --num-events 5 \
  --match-distance-km 0.5 \
  --overwrite-matches \
  --output-root '/run/media/zhangb/My Passport/hinet_data/2024'
