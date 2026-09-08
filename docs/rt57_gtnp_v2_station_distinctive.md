# RT57 GTNP v2 station-distinctive residual

RT57 从一个明确选择的 RT56 checkpoint 做 weight-only 初始化，冻结完整 RT56，首轮只训练
`pga_temporal_residual_head.*`。旧 RT55/RT56 配置和关闭新开关时的推理路径保持不变。

## 实现范围

- `StationDistinctiveTokenAdapter` 读取最终 DiTing pre-pooling temporal tokens、显式 temporal
  validity mask、原始 masked waveform 的 11 维 log-amplitude 特征，以及 2 维有效时长特征。
- adapter 使用 dilation 1/4 的两层 masked depthwise-separable temporal block，并拼接 masked
  mean、masked std 和 four-query attention pooling，得到 256 维 absolute station state `u`。
- event-common state 是有效输入台站 `u` 的 masked mean；默认用 stop-gradient center 得到
  station residual `d`。单输入台站时 `d` 为零，但 pair message 同时使用 `u` 和 `d`。
- query/station pair 同时使用 frozen RT56 query embedding、event embedding、`u`、`d` 和
  absolute/relative geometry。learned pair scorer 汇聚 target-conditioned temporal value 与 gated
  pair value。
- 最终 residual scalar layer 精确零初始化，只平移全部 PGA MDN component means；mixture logits
  和 sigma 不变。
- 辅助目标包括 predictive mean、temporal residual、within-event query PGA difference，以及
  multi-station local PGA residual / single-station absolute local PGA。没有 raw-cosine penalty。

训练日志会记录旧 pooled station cosine、新 `u`/`d` cosine、`d/u` norm ratio、有效秩、局部
PGA 辅助误差、pair/temporal value norm 及其比值，以及 residual delta 的均值、标准差和
P95-P05 range。正式评估 NPZ 额外导出局部 PGA 两个 head 的预测和 input-station PGA targets。

## 超算启动

先把本分支代码完整上传到超算 repo，再选择 RT56 checkpoint。默认选择 RT56
`full_model_best.pth`；如实际选定的是其他文件，必须显式覆盖路径。可选 SHA-256 参数会在提交前
做强校验；未提供时脚本仍会计算并打印实际 SHA-256，保留在提交日志中。

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team

export RT56_BASE_CHECKPOINT="$PWD/weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42/full_model_best.pth"
# 可选但建议：export RT56_BASE_CHECKPOINT_SHA256='<64-character sha256>'

DRY_RUN=1 ACTION=all bash tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh
CONFIRM_RT57=1 ACTION=all bash tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh
```

`ACTION=all` 提交一个固定六 epoch 的 4 节点训练任务，以及三个 `afterok` validation 任务：

1. deterministic causal-random validation；
2. RT55/RT56 normal validation；
3. causal-random validation 的 waveform-only station-roll control。

三个评估任务都读取固定的 `full_model_last.pth`，并在运行前检查 checkpoint metadata 中
`epoch == 6`。任何任务都不会使用 held-out test。

如果训练已经完成，只提交评估：

```bash
CONFIRM_RT57=1 ACTION=eval bash tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh
```

预期输出位于：

```text
logs/weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42/
  epoch6_random_validation/
  epoch6_normal_validation/
  epoch6_random_waveform_permutation/
```

random validation 与 RT56 对齐后，可用现有离线工具做 paired 分析：

```bash
python tools/analyze_random_geometry_full_npz.py \
  --baseline-npz '<rt56-random-validation.npz>' \
  --candidate-npz 'logs/weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42/epoch6_random_validation/eval_validation_epoch6_random.npz' \
  --baseline-metrics '<rt56-random-validation.metrics.json>' \
  --candidate-metrics 'logs/weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42/epoch6_random_validation/eval_validation_epoch6_random.metrics.json' \
  --baseline-name rt56_ep6 \
  --candidate-name rt57_ep6 \
  --output-prefix 'logs/weights_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42/rt56_vs_rt57_random' \
  --trusted-pickle-input
```

是否进入下一阶段必须同时看 random non-input、单输入台站空间差分、normal retention、概率校准
和 waveform-only permutation 对新 residual contribution 的破坏程度。若空间指标改善但 residual
delta 对 waveform permutation 几乎不敏感，应标记为 geometry-prior-dominated，不能视为已经解决
station waveform collapse。
