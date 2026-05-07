# 当前研究现状总结

更新日期：2026-05-07

本文档总结当前 PGA 估计项目的研究目标、数据与模型设置、已完成实验、主要结果、报告材料状态和后续建议。本文面向后续继续实验和撰写学术报告使用，不展开早期结构坍塌排查细节。

## 1. 研究目标

本项目目标是利用有限数量输入台站的三分量强震波形和台站位置，预测一组目标台站的 log PGA。目标台站可以不是输入台站，因此任务本质是多台站观测条件下的 site-specific PGA 空间估计。

当前研究定位：

- 基于 TEAM 的可变台站集合建模思路。
- 使用 DiTing backbone 提取单台波形表征。
- 不显式使用绝对振幅信息，重点考察波形形状表征、台站集合建模和几何信息是否足以支持 PGA 强度估计。
- 使用 target-wise cross-attention readout：每个目标台站 query 独立 cross-attend 输入台站 tokens。

当前报告题目采用：

```text
基于多台站波形表征的目标台站 PGA 估计
```

## 2. 数据与评估口径

数据来源为 NIED K-NET 和 KiK-net 强震台网。

- K-NET：地表强震动台网，主要记录三分量地表强震加速度。
- KiK-net：地表 + 井下强震动台网，包含 surface/borehole 成对记录。
- 本项目使用其中的波形、台站位置、事件信息、P pick 和 PGA 标签。

当前统计：

| Split | Events | Station records | Median M | Max M | Median epicentral distance | Max PGA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 699 | 30,506 | 3.7 | 7.6 | 110.0 km | 28.28 m/s2 |
| Validation | 101 | 8,544 | 4.0 | 6.0 | 108.7 km | 5.72 m/s2 |
| Test | 201 | 13,052 | 4.0 | 7.1 | 103.7 km | 5.74 m/s2 |

台网统计：

| Network | Unique stations | Station records | Sensor records | Sensor classes |
| --- | ---: | ---: | ---: | --- |
| K-NET | 960 | 21,323 | 960 | single_surface |
| KiK-net | 529 | 30,779 | 1,057 | borehole, surface |

评估口径：

- 主指标在 log PGA 空间计算。
- 主 eval 使用 `input_station_selection=epidist`，即按震中距最近的台站选择输入台站。
- full model 同时评估 `full_model_best.pth` 和 `full_model_last.pth`。
- single-station 结果作为波形表征能力参考，不和 multi-station 做严格同样本排名。

## 3. 当前模型方案

当前模型链路：

```text
input waveforms
  -> frozen DiTing encoder
  -> trainable station adapter
  -> station tokens + coordinate encoding
  -> TEAM-style Transformer
  -> target-wise PGA cross-attention
  -> point log PGA prediction
```

关键设计：

- `pga_readout_mode=target_cross_attention`
- PGA target query 包含目标位置编码和可学习 query token，不再只是位置编码。
- station valid mask 只屏蔽无效台站。
- PGA target query cross-attend station tokens；PGA target 不作为普通 token 参与 station self-attention。
- 当前报告 PPT 中已加入 attention 可视化，用于展示某个目标台站 query 对输入台站的注意力分布。

当前全数据综合配置对应：

```text
pga_configs/transformer_japan_full_pga15_b1_b3_b5_b7_noamp_lr_chaosuan.json
```

组合因素：

- b1：single-station pretrain 中提高 PGA 权重。
- b3：PGA target normalization。
- b5：Huber loss delta=3。
- b7：relative geometry 相关配置。
- 不使用显式振幅信息。
- 不使用 AdamW。
- LR scheduler 使用较温和的衰减策略，并设置 `min_lr=1e-5`。

## 4. 已完成实验与结论

### 4.1 Stage1 消融实验

Stage1 使用 `overfit_n=128`，主要目的是筛选影响 PGA 性能的因素。已完成 b0-b7 八组实验。

核心结论：

- b1 有稳定正贡献：提高 single-station pretrain 中 PGA 权重后，train 和 validation 都有改善。
- b3 对训练拟合有帮助，但 validation 收益不稳定。
- b5 有正信号，Huber delta=3 比当前 baseline 更适合作为组合因素。
- b7 在 validation 上效果最好，说明相对几何或相关空间信息对泛化有帮助。
- b2 只训 PGA 和 b4 MSE 不适合作为当前主线。
- station feature 仍有趋同倾向，但 cosine similarity 不能单独解释最终 PGA 性能。

### 4.2 全数据综合配置结果

最新全数据综合配置结果位于：

```text
chaosuan_res/weights_japan_full_pga15_b1_b3_b5_b7_noamp_lr
```

这里的数值由 8 个 eval shard 的 `.npz` 合并后重新统计得到。

| Checkpoint | Split | N targets | MAE | RMSE | Corr | R2 | Slope | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | train | 8,252 | 0.2396 | 0.3274 | 0.8585 | 0.7359 | 0.7079 | 0.0009 |
| best | val | 1,406 | 0.3151 | 0.4115 | 0.6824 | 0.4592 | 0.5186 | -0.0121 |
| last | train | 8,252 | 0.2403 | 0.3289 | 0.8580 | 0.7334 | 0.7811 | 0.0034 |
| last | val | 1,406 | 0.3214 | 0.4194 | 0.6849 | 0.4384 | 0.5866 | -0.0203 |

解读：

- `best` 在 validation MAE/RMSE/R2 上略优于 `last`。
- `last` 的 slope 更高，说明动态范围稍好，但误差略大。
- validation slope 仍明显小于 1，强弱 PGA 动态范围压缩仍是主要问题。
- validation bias 接近 0，整体均值偏差不大；问题主要是 slope 和强 PGA 分桶偏差。

single-station 参考结果：

| Split | N samples | MAE | RMSE | Corr | R2 | Slope | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5,675 | 0.3268 | 0.4256 | 0.7599 | 0.5767 | 0.5979 | -0.0035 |
| val | 915 | 0.3520 | 0.4579 | 0.5896 | 0.2936 | 0.4447 | -0.0893 |

解读：

- 在全数据综合配置下，multi-station full model 明显优于 single-station PGA 参考模型。
- single-station validation 明显弱于 train，说明单台波形表征仍有泛化压力。
- 这也说明多台站空间信息对目标台站 PGA 估计是有实际贡献的。

### 4.3 报告图件使用的参考结果

当前学术报告 PPT 的多数图件来自：

```text
reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets
```

该结果的合并统计为：

| Checkpoint | Split | N targets | MAE | RMSE | Corr | R2 | Slope | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | train | 8,233 | 0.2058 | 0.2836 | 0.8932 | 0.7977 | 0.7997 | 0.0067 |
| best | val | 1,406 | 0.2936 | 0.3855 | 0.7354 | 0.5275 | 0.6233 | -0.0162 |
| last | train | 8,233 | 0.2066 | 0.2837 | 0.8938 | 0.7976 | 0.8054 | 0.0228 |
| last | val | 1,406 | 0.2950 | 0.3860 | 0.7353 | 0.5263 | 0.6285 | 0.0020 |

这组结果目前更适合做报告示例图件，因为对应的 `eval_results_best.npz` 已包含：

- train/validation 预测散点图。
- 按输入台站数分桶。
- 按震中距分桶。
- 按 true PGA 强度分桶。
- case study station-count sweep。
- attention 可视化。

但需要注意：报告图件和最新全数据综合配置不是同一个结果目录。正式定稿时建议统一使用最终选定模型重新生成所有图件。

## 5. 主要科学观察

### 5.1 输入台站数量有效

在报告图件使用的结果中，validation 按输入台站数分桶显示：

| Input station bucket | Events | Targets | MAE | RMSE | Corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 21 | 271 | 0.3382 | 0.4340 | 0.6012 |
| 4-5 | 12 | 158 | 0.3130 | 0.3884 | 0.7166 |
| 6-10 | 25 | 340 | 0.3001 | 0.3868 | 0.7494 |
| 11-15 | 13 | 188 | 0.2779 | 0.3719 | 0.7959 |
| 16+ | 29 | 435 | 0.2698 | 0.3628 | 0.7607 |

趋势清楚：输入台站数越多，MAE 大体下降，相关性提高。这支持多台站 Transformer 的必要性。

### 5.2 震中距影响非单调

validation 按目标台站震中距分桶：

| Distance bin | Targets | Mean dist. | MAE | RMSE | Bias | Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-50 km | 415 | 27.3 km | 0.3258 | 0.4269 | -0.1082 | 0.6538 |
| 50-100 km | 377 | 74.5 km | 0.2929 | 0.3714 | -0.0028 | 0.7042 |
| 100-200 km | 475 | 139.9 km | 0.2670 | 0.3503 | 0.0578 | 0.7359 |
| 200-400 km | 113 | 242.6 km | 0.3067 | 0.4028 | 0.1420 | 0.6700 |
| 400-800 km | 13 | 539.6 km | 0.4633 | 0.6111 | 0.3432 | 0.4894 |

误差不是随震中距单调变化。近场可能受强 PGA、复杂传播和 P pick 对齐影响；远场样本数少，且模型有系统高估趋势。

### 5.3 强 PGA 动态范围仍被压缩

validation 按 true PGA 强度分桶：

| True log PGA bin | N | Label mean | MAE | Bias | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| [-2.86, -2.09] | 279 | -2.451 | 0.356 | 0.318 | 0.463 |
| [-2.09, -1.69] | 278 | -1.853 | 0.254 | 0.119 | 0.311 |
| [-1.69, -1.45] | 279 | -1.574 | 0.223 | 0.010 | 0.284 |
| [-1.45, -1.16] | 278 | -1.315 | 0.268 | -0.106 | 0.348 |
| [-1.16, 0.18] | 279 | -0.852 | 0.382 | -0.334 | 0.487 |

这说明模型对弱 PGA 倾向高估，对强 PGA 倾向低估，典型表现为动态范围压缩。后续优化不能只看 MAE，还要持续监控 slope 和强 PGA bin 的 Bias。

### 5.4 Attention 可视化已可展示不同输入台站数

最新 `eval_attention_best.npz` 已包含 validation split 的输入台站数 sweep：

```text
requested_station_count = 3, 5, 8, 12, 16, 25
```

报告中当前选取 3、8、12、25 个输入台站的 attention 示例。可视化方式：

- 红圈：当前正在预测 PGA 的目标台站。
- 三角形：输入台站。
- 三角形大小和颜色：该目标台站 query 对输入台站的 attention weight。
- 红线：attention 最高的若干输入台站。

该图适合说明模型在预测某个目标台站 PGA 时，是否更依赖近场台站、事件附近台站或局部空间邻近台站。

## 6. 当前报告材料状态

主要 PPT 文件：

```text
reports/pga_academic_report_draft_current.pptx
```

主要图件目录：

```text
reports/pga_academic_report_assets_current
```

已完成并反复修正的图件：

- 训练数据分布图。
- K-NET/KiK-net 台站分布图，使用 Cartopy/Natural Earth 缓存底图。
- train/validation 主指标图。
- single-station vs multi-station 对比图。
- predicted vs true PGA。
- residual vs true PGA。
- true PGA strength bins。
- station-count buckets。
- epicentral-distance buckets。
- case study station-count sweep。
- case study target-wise PGA + spatial residual map。
- attention target map。
- P pick 诊断图。

重要约束：

- 所有地图图件现在应强制使用 Cartopy 缓存的 Natural Earth 底图；不要再退回简化日本轮廓。
- PPT 文件本身不应提交到 git。
- 图件可以作为本地报告资产使用，是否提交需单独决定。

## 7. 当前问题与局限

1. 强 PGA 动态范围仍不足。
   - slope 仍显著小于 1。
   - 强 PGA bin 负 Bias 明显。
   - 需要继续探索 loss reweighting、强 PGA oversampling、bin-aware loss 或 calibration。

2. P pick 数据质量是限制因素。
   - 当前 P pick 主要由自动算法产生，未逐条人工精修。
   - 低信噪比、复杂震相、截断记录会影响窗口对齐。
   - P pick 偏差会传递到单台波形表征和多台站时序一致性。

3. P pick 前冗余不足。
   - K-NET/KiK-net 记录中 P pick 前可用冗余较短。
   - 当前不足部分使用 0 padding。
   - 真实在线应用中对应位置应是背景噪声，后续需要缩小训练-应用差异。

4. 文献数值对比仍不严格。
   - 当前只做 TEAM/QuakeFormer 方法定位。
   - 没有在同一数据集、同一划分、同一指标下复现 TEAM/QuakeFormer，因此不能做严格数值排名。

5. case study 数据源需要统一。
   - 已发现普通 `val_*` 和 `case_sweep_val_*` 在个别事件上可能存在元数据错位。
   - 对报告中的 case study，建议统一从 `case_sweep_val_*` 取同一事件和同一输入台站数记录。

## 8. 建议下一步

### 8.1 结果统一

正式报告前，建议选定一个最终模型目录，重新生成全部图件：

```text
eval_results_best.npz
eval_results_last.npz
eval_attention_best.npz
```

并确保：

- 主结果、case study、attention 图来自同一模型。
- station-count sweep 和 target-wise case study 使用同一事件源。
- 所有地图使用 Cartopy 缓存底图。

### 8.2 模型改进

优先方向：

- 针对强 PGA 动态范围做 loss 或采样加权。
- 对 PGA 分桶监控 slope、Bias、MAE，而不是只看整体 MAE。
- 继续比较 `best` 和 `last`：`best` 误差低，`last` slope 高，可能需要单独选择汇报指标最均衡的 checkpoint。
- 对 target normalization、Huber delta 和 relative geometry 做更严格的全数据对照。

### 8.3 数据改进

优先方向：

- 建立更可靠的 P pick 审核或筛选机制。
- 增加真实背景噪声 padding，而不是 0 padding。
- 检查 `eval_results.npz` 中普通 split 数组和 case sweep 数组的事件元数据一致性。
- 对 K-NET 与 KiK-net 分开统计性能，检查地表/井下记录对模型误差的影响。

### 8.4 报告完善

当前学术报告可以保留的核心叙事：

- PGA 是快速地震动评估的重要指标。
- TEAM/QuakeFormer 证明多台站 Transformer 适合可变台站集合和目标位置预测。
- 本项目展示了不显式使用绝对振幅信息时，波形形状 + 多台站空间信息仍可支持 PGA 强度估计。
- 输入台站数增加明显降低误差。
- 强 PGA 动态范围仍是核心挑战。
- 数据质量，尤其 P pick 和 pre-pick padding，是后续落地的重要限制。

