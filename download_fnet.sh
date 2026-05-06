#!/bin/bash
python tools/download_fnet_event_windows.py \
--strong-root /run/media/zhangb/aa0013a6-c6ff-4112-9526-410918058645/Japandata/waveformsnew \
--output-root '/run/media/zhangb/My Passport/Japandata/fnet' \
--years 2024 \
--limit-events 2 \
--channels '*' \
--network BO \
--routing-client eida-routing \
--probe-inventory \
--dry-run
