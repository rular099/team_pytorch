#!/bin/bash

#  支持 vertical、ns、ew 和 norm 四种分量。每张图包含：
#
#  - K-NET/KiK-net 原始加速度；
#  - Hi-net 速度计下载波形；
#  - 相对理论 P 到时和绝对 JST 双行刻度；
#  - 发震时刻、理论 P 时刻和最终 pick 偏差；
#  - K-NET—Hi-net 台站间距；
#  - 震中距、震级、深度和传感器类型；
#  - 采样间隔、缺口数、有限值比例；
#  - 自动 PASS/WARN/FAIL 状态。
#
#  同时生成 qc_summary.csv，检查理论 P 是否位于两套记录内、时间是否单调、窗口覆盖、NaN/Inf、数据缺口及台站匹配距离。
#

python tools/plot_hinet_accel_velocity_qc.py \
   --max-plots 20 \
   --component vertical \
   --show-candidate-picks \
   --save-pdf \
   --output-dir '/run/media/zhangb/My Passport/hinet_data/2024_2/waveform_qc'

##  检查全部下载记录：

#python tools/plot_hinet_accel_velocity_qc.py \
#  --max-plots 0 \
#  --component vertical \
#  --save-pdf \
#  --output-dir '/run/media/zhangb/My Passport/hinet_data/2024/waveform_qc'
## 也可筛选：

#python tools/plot_hinet_accel_velocity_qc.py \
#  --event-id 20240101185300 \
#  --station ISKH01 \
#  --component vertical \
#  --save-pdf \
#  --output-dir '/run/media/zhangb/My Passport/hinet_data/2024/waveform_qc'
