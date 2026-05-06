# 学术报告规划：基于 TEAM 改进的多台站 PGA 估计

更新日期：2026-05-05

本文档给出面向学术汇报的报告结构、建议图表、文献背景和结果展示方式。报告不展示早期结构坍塌和 stage1 消融细节；这些内容只作为内部研发过程，不进入主线叙事。

## 1. 报告定位

建议题目：

```text
基于 DiTing 波形表征与 TEAM Transformer 的多台站 PGA 估计
```

英文题目可选：

```text
Multi-station PGA Estimation with a DiTing-enhanced TEAM Transformer
```

核心叙事：

1. PGA 是震后快速响应、烈度图和工程地震动评价中的关键强地面运动指标。
2. TEAM 证明了 Transformer 可以处理任意数量、任意位置的台站波形并预测目标位置 PGA。
3. QuakeFormer 进一步强调了多台站观测、遮蔽建模和绝对/相对空间关系在地震动预测中的价值。
4. 本项目在 TEAM 框架基础上引入 DiTing 波形表征、target-wise PGA cross-attention 和相对几何 bias，面向日本强震数据进行 PGA 估计。
5. 全数据实验展示最终模型的精度、空间泛化、强弱 PGA 表现和输入台站数量敏感性。

## 2. 推荐报告结构

### Slide 1: 标题页

- 标题、作者、单位、日期。
- 背景图可以用日本台站/事件空间分布。

### Slide 2: 研究问题

问题定义：

```text
给定若干输入台站的三分量波形与台站坐标，预测一组目标台站的 PGA。
```

强调：

- 目标是 site-specific ground motion prediction。
- 输入台站数量和空间分布每个事件不同。
- 目标台站可以不是输入台站。

建议图件：

- `dataset_overview` 或事件/台站地图。
- 如果暂时没有地图，先用任务示意图代替。

### Slide 3: PGA 背景

讲清楚 PGA 定义：

```text
PGA = max_t |a(t)|
```

如果标签为水平分量最大值，应写：

```text
PGA_H = max(PGA_EW, PGA_NS)
```

如果标签来自数据文件已有 `pga` 字段，应说明：

- 单位
- 是否取 log
- 是否使用水平分量或三分量
- 是否标准化

建议要点：

- PGA 是某一地点的峰值地面加速度，不是地震整体震级。
- PGA 在 ShakeMap、烈度估计、快速灾情判断和工程地震动中广泛使用。
- PGA 受震级、震源距离、传播路径、场地条件和局部放大影响，空间变化复杂。

参考文献：

- Wald et al. (1999), TriNet ShakeMaps.
- USGS ShakeMap manual/background.
- Boore & Atkinson (2008), GMPE.
- Boore et al. (2014) 或 Campbell & Bozorgnia (2014), NGA-West2.

### Slide 4: 相关工作

建议做一页文献定位表，而不是长篇介绍。

| 方法 | 核心任务 | 空间建模 | 和本项目关系 |
| --- | --- | --- | --- |
| TEAM | 实时 PGA 分布/预警阈值 | 任意台站和目标位置 Transformer | 本项目基础框架 |
| QuakeFormer | 预测、补间、早期预警统一建模 | masked Transformer，绝对+相对空间嵌入 | 支持相对几何和部分观测建模的重要参考 |
| 本项目 | 不显式使用振幅的目标台站 PGA 估计 | DiTing + TEAM transformer + target cross-attention + relative geometry | 面向日本数据的 PGA 估计实现 |

报告中可说：

> 本项目不是从零提出新的 EEW 框架，而是在 TEAM 多台站 PGA 预测框架上，替换和增强波形表征与目标台站 readout，并吸收 QuakeFormer 对空间依赖建模的启发。

建议图件：

- `literature_model_comparison.png`
- `literature_metric_comparison.png`，如果已经有可比数值。

注意：TEAM/QuakeFormer 原文任务、数据划分和标签定义可能不同。只有在同一数据集和同一评价口径下复现后，才适合做严格数值比较；否则主报告中做方法定位对比，数值表标注为“reported/reproduced under comparable setting”。

### Slide 5: 方法总览

建议画模型结构图：

```text
Input station waveforms
  -> DiTing encoder
  -> station adapter
  -> station tokens
  -> TEAM-style Transformer
  -> PGA target query
  -> cross-attention over station tokens
  -> log PGA prediction
```

需要解释的三个设计：

1. DiTing backbone：利用预训练波形表征。
2. Target-wise cross-attention：每个目标台站独立查询输入台站信息。
3. Relative geometry bias：显式编码目标台站和输入台站的相对位置关系。

建议图件：

- 模型结构图，可以基于现有 `docs/figures/` 图件修改。

### Slide 6: 数据集与实验设置

展示：

- 数据来源：日本强震数据。
- 训练/验证/测试划分。
- 事件数量、台站数量、目标 PGA 数量。
- 输入波形窗口。
- 输入台站随机数量设置。
- PGA 标签变换方式。

建议表格：

| Split | Events | Input station samples | PGA targets | Magnitude range | log PGA range |
| --- | ---: | ---: | ---: | ---: | ---: |

如果 eval 文件没有这些统计，需要从数据构建脚本或 HDF5 元数据另行统计。

### Slide 7: 训练策略

简洁说明：

- single-station pretrain：`mag/epidist/pga` 多任务，PGA 权重较高。
- full model：PGA-only objective。
- Huber loss。
- PGA target normalization。
- validation-best checkpoint。
- LR scheduler：`patience=12, factor=0.5, min_lr=1e-5`。

不要展示调参过程，只讲最终训练方案。

### Slide 8: 主结果表

展示全数据结果。建议至少包含：

| Model | Split | MAE | RMSE | Corr | R2 | Slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TEAM baseline/reproduced | val/test | | | | | |
| QuakeFormer/reproduced | val/test | | | | | |
| Ours | val/test | | | | | |

如果 TEAM/QuakeFormer 没有同口径复现：

- 表中只放本项目 baseline 和 final model。
- 另用“方法定位表”比较 TEAM/QuakeFormer。
- 明确说明不同论文结果不可直接数值比较。

建议图件：

- `main_pga_metrics.png`
- `validation_metric_bars.png`

### Slide 9: Predicted vs True PGA

核心图。

展示：

- x: true log PGA
- y: predicted log PGA
- `y=x`
- fitted line
- MAE/R2/corr/slope 标注

建议图件：

- `scatter_<primary>_val.png`
- 如空间允许，并排放 train 和 val/test。

这页用于证明模型不是只学均值，并展示动态范围。

### Slide 10: Residual Analysis

展示 residual vs true PGA：

```text
residual = pred - label
```

目标：

- 看强 PGA 是否系统性低估。
- 看低 PGA 是否高估。
- 判断动态范围压缩程度。

建议图件：

- `residual_vs_true_<primary>_val.png`

### Slide 11: 按 PGA 强度分桶

按 true PGA 分位数分桶，例如 5 桶：

| True PGA bin | N | MAE | Bias | RMSE |
| --- | ---: | ---: | ---: | ---: |

建议图件：

- `pga_strength_bins_<primary>_val.png`

这页直接回答：

> 模型对强/弱 PGA 是否同样有效？

### Slide 12: 按输入台站数量分桶

展示不同输入台站数量下的 MAE/RMSE/corr：

```text
1
2-3
4-5
6-10
11-15
16+
```

建议图件：

- `val_mae_by_station_count.png`

这页说明多台站观测对模型性能的影响。

### Slide 13: 按震中距分桶

展示不同目标台站震中距下的 MAE/RMSE/bias：

```text
0-50
50-100
100-200
200-400
400-800
800+
```

建议图件：

- `val_mae_by_epicentral_distance.png`
- `epicentral_distance_buckets.csv/md`

这页说明模型在近场和远场目标台站上的误差差异。该图需要 `eval_results.npz` 中包含：

- `<split>_pga_target_abs`
- `<split>_loc_label_abs`
- `<split>_pga_label`
- `<split>_pga_mu_best`
- `<split>_pga_target_valid`

当前新版 `eval_checkpoint.py` 已保存这些字段；旧版 npz 没有坐标时会自动跳过震中距图。

### Slide 14: 空间案例分析

建议选 2-3 个事件：

1. 预测效果好的事件。
2. 强 PGA 被低估的事件。
3. 输入台站较少但预测可接受的事件。

每个 case-study 图包含：

- 目标台站 true/pred PGA 对比。
- residual 诊断；如果有坐标，则横轴为震中距。
- 空间 residual map：震源位置、输入台站、目标台站 residual。
- 标题中标注输入台站数和台站数分桶。

建议图件：

- `case_study_*.png`

如果当前 `eval_results.npz` 没有保存目标台站坐标，脚本会退化为 target-index 级别 case 图；若需要地图和震中距 case 分析，需要用新版 `eval_checkpoint.py` 重新推理并保存：

- input station coordinates
- target station coordinates
- event location
- label PGA
- predicted PGA

### Slide 15: 和 TEAM / QuakeFormer 的关系与比较

建议分两部分：

1. 方法层面：
   - TEAM：任意台站输入、目标 PGA 预测，是本项目基础。
   - QuakeFormer：masked Transformer、统一地震动预测任务、绝对/相对空间嵌入，对本项目 relative geometry 设计有启发。
   - 本项目：DiTing waveform representation + TEAM target readout + relative geometry，专注日本 PGA。

2. 数值层面：
   - 如果有同数据集复现结果，放主表。
   - 如果只有论文报告结果，不要直接和本项目数值混排；可以放“reported on different datasets/tasks”说明。

### Slide 16: 结论与展望

结论建议三点：

1. 在 TEAM 框架基础上引入 DiTing 表征和 target cross-attention，完成了多台站 PGA 估计模型。
2. 全数据实验显示模型在 MAE/RMSE/R2/corr 上优于基线，并可在不同输入台站数量下工作。
3. 强 PGA 动态范围仍是后续重点，需要进一步研究强震样本重加权、calibration loss 或不确定性建模。

## 3. 图表生成代码

主脚本：

```bash
python tools/generate_pga_report_assets.py \
  --results-root chaosuan_res \
  --pattern 'weights_japan*_pga15*' \
  --output-dir reports/pga_academic_report_assets \
  --primary weights_japan_full_pga15_b1_b3_b5_b7_noamp_lr
```

单机运行：

```bash
bash tools/run_pga_report_assets_local.sh chaosuan_res reports/pga_academic_report_assets
```

Slurm 运行：

```bash
bash tools/run_pga_report_assets_slurm.sh chaosuan_res reports/pga_academic_report_assets
```

如果需要选择主模型：

```bash
REPORT_PRIMARY_MODEL=weights_japan_full_pga15_b1_b3_b5_b7_noamp_lr \
bash tools/run_pga_report_assets_slurm.sh chaosuan_res reports/pga_academic_report_assets
```

如果需要把 TEAM/QuakeFormer 复现指标加入数值对比：

1. 先运行脚本生成模板：

```bash
bash tools/run_pga_report_assets_local.sh chaosuan_res reports/pga_academic_report_assets
```

2. 编辑：

```text
reports/pga_academic_report_assets/literature_metric_template.csv
```

3. 再运行：

```bash
python tools/generate_pga_report_assets.py \
  --results-root chaosuan_res \
  --output-dir reports/pga_academic_report_assets \
  --literature-metrics-csv reports/pga_academic_report_assets/literature_metric_template.csv
```

## 4. 代码会生成的主要产物

表格：

- `main_pga_metrics.csv/md/png`
- `station_count_buckets.csv/md`
- `single_station_metrics.csv/md`
- `literature_model_comparison.csv/md/png`
- `literature_metric_comparison.csv/md/png`
- `references.csv/md`

图件：

- `validation_metric_bars.png`
- `train_val_metric_bars.png`
- `val_mae_by_station_count.png`
- `val_mae_by_epicentral_distance.png`
- `scatter_*_val.png`
- `residual_vs_true_*_val.png`
- `pga_strength_bins_*_val.png`
- `case_study_*.png`
- `loss_curve_*.png`
- `literature_model_comparison.png`

所有图片默认使用较大的字号和较高 dpi，适合投影汇报。

## 5. 推荐引用

TEAM:

> Muenchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021). The transformer earthquake alerting model: a new versatile approach to earthquake early warning. Geophysical Journal International, 225(1), 646-656. https://doi.org/10.1093/gji/ggaa609

TEAM software:

> Muenchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021). TEAM - The Transformer Earthquake Alerting Model. GFZ Data Services. https://doi.org/10.5880/GFZ.2.4.2021.003

QuakeFormer:

> Feng, Y., Zhu, W., & Lu, X. (2024). QuakeFormer: A Uniform Approach to Earthquake Ground Motion Prediction Using Masked Transformers. arXiv:2412.00815.

ShakeMap / PGA application:

> Wald, D. J., Quitoriano, V., Heaton, T. H., Kanamori, H., Scrivner, C. W., & Worden, C. B. (1999). TriNet ShakeMaps: Rapid generation of peak ground motion and intensity maps for earthquakes in Southern California. Earthquake Spectra, 15(3), 537-555. https://doi.org/10.1193/1.1586057

GMPE:

> Boore, D. M., & Atkinson, G. M. (2008). Ground-motion prediction equations for the average horizontal component of PGA, PGV, and 5%-damped PSA. Earthquake Spectra, 24(1), 99-138. https://doi.org/10.1193/1.2830434

NGA-West2:

> Boore, D. M., Stewart, J. P., Seyhan, E., & Atkinson, G. M. (2014). NGA-West2 equations for predicting PGA, PGV, and 5% damped PSA for shallow crustal earthquakes. Earthquake Spectra, 30(3), 1057-1085. https://doi.org/10.1193/070113EQS184M

Deep learning PGA:

> Liu, Y., Zhao, Q., & Wang, Y. (2024). Peak ground acceleration prediction for on-site earthquake early warning with deep learning. Scientific Reports, 14, 5485. https://doi.org/10.1038/s41598-024-56004-6
