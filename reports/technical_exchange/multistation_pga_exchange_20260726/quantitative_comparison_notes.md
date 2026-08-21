# 定量对比口径与出处

本文件用于后续制作 PPT 图表和页脚注释，避免把不同任务、数据集和对数单位的数字直接当成同一排行榜。

## 1. TEAM（arXiv:2009.06316v4）

来源：`2009.06316v4.pdf`。

### 日本测试集阈值预警

来源：Table S1（PDF 第 28 页）。论文表中写作 AUC；结合 Figure 4 的 precision–recall 曲线，在报告中写作 PR-AUC。

| PGA 阈值 | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| 1%g | 0.70 | 0.77 | 0.73 | 0.82 |
| 2%g | 0.69 | 0.69 | 0.69 | 0.76 |
| 5%g | 0.59 | 0.67 | 0.63 | 0.68 |
| 10%g | 0.50 | 0.60 | 0.54 | 0.56 |
| 20%g | 0.33 | 0.48 | 0.39 | 0.35 |

### 相对预警时间

来源：Table S3（PDF 第 28 页）。正值表示表中第二种方法的平均预警时间更长。

| 对比 | 1%g | 2%g | 5%g | 10%g | 20%g |
| --- | ---: | ---: | ---: | ---: | ---: |
| TEAM 相对 EPS | +0.39 s | +0.43 s | +0.70 s | +0.31 s | +0.61 s |
| TEAM 相对 PLUM-like | +8.98 s | +8.24 s | +6.35 s | +5.01 s | +0.55 s |

补充事实：论文报告一次时间步处理 25 个输入台站、246 个目标位置，在未优化工作站上耗时约 0.15 s。

## 2. QuakeFormer（arXiv:2412.00815v1）

来源：`2412.00815v1.pdf`。

### Forecasting / interpolation

来源：Figure 3；R² 为读图近似值。

| 条件 | Seen station R² | Unseen station R² |
| --- | ---: | ---: |
| QuakeFormer, station mask ratio = 1 | ≈0.917 | ≈0.837 |
| QuakeFormer, station mask ratio ≈ 0.1 | ≈0.937 | ≈0.858 |
| ASK14 | 0.71 | 0.67 |

残差标准差来自 Figure 3 和 Table 3，均为自然对数单位：

| 模型 | Seen station σ | Unseen station σ |
| --- | ---: | ---: |
| QuakeFormer | 0.57 | 0.75 |
| ASK14 | 1.09 | 1.07 |

### EEW 随时间变化

来源：Figure 7；约 15 s 处读图近似值。

| 模型 | Seen station R² | Unseen station R² |
| --- | ---: | ---: |
| Uniform pretrain + EEW finetune，使用波形 | ≈0.94 | ≈0.87 |
| Single EEW model，使用波形 | ≈0.925 | ≈0.84 |
| Uniform pretrain + EEW finetune，屏蔽波形 | ≈0.86 | ≈0.80 |

## 3. 本项目

### 历史 full-data 结果

训练事件数 699，validation 事件数 101。

| Split | MAE | RMSE | Corr | R² | slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.2234 | 0.3073 | 0.8750 | 0.7633 | 0.7634 |
| Validation | 0.2966 | 0.3875 | 0.7278 | 0.5221 | 0.5931 |

模型原始误差在 log10 PGA 空间。与 QuakeFormer 比较残差标准差时乘以 `ln(10)`：

- Train：0.30581 log10 → 0.70415 natural log。
- Validation：0.38746 log10 → 0.89215 natural log。

### 512-event overfit 实验

使用各 run 的 last checkpoint。rt47 缺少完整 normal-eval NPZ，不进入完整表格。

| Run | Train MAE | Train R² | Train slope | Val MAE | Val R² | Val slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rt44 | 0.0491 | 0.9573 | 0.9540 | 0.3080 | 0.4787 | 0.5382 |
| rt45 | 0.0605 | 0.9520 | 0.9800 | 0.3054 | 0.4873 | 0.5905 |
| rt46 | 0.0498 | 0.9587 | 0.9845 | 0.3034 | 0.4997 | 0.5913 |
| rt48 | 0.0491 | 0.9551 | 0.9716 | 0.2968 | 0.5122 | 0.5820 |
| rt49 | 0.0699 | 0.9565 | 1.0228 | 0.3018 | 0.5023 | 0.6205 |
| rt50 | 0.0758 | 0.9375 | 0.9571 | 0.3060 | 0.4863 | 0.5969 |
| rt51 | 0.0610 | 0.9504 | 0.9262 | 0.3025 | 0.5049 | 0.5446 |

rt48 残差标准差：

- Train：0.11595 log10 → 0.26697 natural log。
- Validation：0.38984 log10 → 0.89763 natural log。

### TEAM 风格阈值指标的探索性换算

使用 rt48、固定 20 s 输入窗口，以点预测作为排序分数。F1 的 score cutoff 在 train 和 validation 各自独立优化；PR-AUC 为阈值无关排序指标。该过程不等同于 TEAM 的概率预警随时间评估，train 行只用于表征拟合容量。

| PGA 阈值 | TEAM test F1 / PR-AUC | 本项目 train F1 / PR-AUC | Train 正例数 | 本项目 validation F1 / PR-AUC | Validation 正例数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1%g | 0.730 / 0.820 | 0.965 / 0.995 | 543 | 0.620 / 0.642 | 125 |
| 2%g | 0.690 / 0.760 | 0.952 / 0.985 | 196 | 0.545 / 0.536 | 55 |
| 5%g | 0.630 / 0.680 | 0.932 / 0.985 | 49 | 0.438 / 0.372 | 13 |

TEAM 论文只公开日本 test 阈值结果，没有对应 train 行。本项目 validation 的 10%g 只有 3 个正例，不报告；20%g 无正例。

### KNET-only rt44–rt51 完整 last eval

来源：`chaosuan_res/logs.zip` 中各 run 的 normal 与 waveform–station roll eval。以下均为 KNET-only 口径：

| Run | Train MAE | Train R² | Train slope | Val MAE | Val R² | Val slope | Strong bias | Roll ΔMAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rt44 | 0.0540 | 0.9512 | 0.9896 | 0.2976 | 0.4552 | 0.5617 | -0.2812 | +0.0120 |
| rt45 | 0.0994 | 0.9142 | 0.9541 | 0.3092 | 0.4264 | 0.5554 | -0.1967 | +0.0110 |
| rt46 | 0.0733 | 0.9319 | 0.9515 | 0.2971 | 0.4579 | 0.5471 | -0.2401 | +0.0101 |
| rt47 | 0.0490 | 0.9535 | 0.9691 | 0.2998 | 0.4511 | 0.5487 | -0.2899 | +0.0108 |
| rt48 | 0.0498 | 0.9549 | 0.9839 | 0.2877 | 0.4817 | 0.5723 | -0.2670 | +0.0159 |
| rt49 | 0.0552 | 0.9588 | 0.9336 | 0.2881 | 0.4873 | 0.5362 | -0.2825 | +0.0095 |
| rt50 | 0.0618 | 0.9424 | 0.9081 | 0.2991 | 0.4502 | 0.5208 | -0.2768 | +0.0080 |
| rt51 | 0.0742 | 0.9324 | 0.8809 | 0.2968 | 0.4540 | 0.5192 | -0.3072 | +0.0150 |
| Mean | 0.0646 | 0.9424 | 0.9464 | 0.2969 | 0.4580 | 0.5452 | -0.2677 | +0.0115 |

平均 validation−train MAE gap 为 0.2323，单 run 约为 0.210–0.251。rt48 的 validation MAE 最低，rt49 的 validation R² 最高；两者应视为并列候选。

必须区分两种比较：

- 非配对的 7-run 均值为 mixed-source 0.3034 → KNET-only 0.2965，表面变化 -0.0069，6/7 run 改善；但 validation target 数分别为 4512 与 4082。
- 固定相同 event、time 和 target coordinate 后有 2716 个共同目标，KNET−mixed 的平均 MAE 差为 +0.0001，event-bootstrap 95% CI 为 [-0.0124, +0.0124]。因此没有配对证据说明 KNET-only 训练本身改善 held-out 泛化。

station-roll 结果：

- mixed-source 7-run 平均 train / validation ΔMAE 为 +0.1114 / +0.0122。
- KNET-only 8-run 平均 train / validation ΔMAE 为 +0.0988 / +0.0115；8/8 validation MAE 变差，7/8 event-bootstrap 区间排除 0。
- KNET-only validation R² 平均下降 0.0433，strong-PGA bias 从 -0.2677 加深为 -0.3049。
- rt51 station-roll training control 的 validation ΔMAE 为 +0.0150、ΔR² 为 -0.0573，没有改善鲁棒性。

校准结果：

- 平均 1σ / 2σ 覆盖率为 13.8% / 27.0%，预测方差明显过度自信。
- 平均 Brier 为 0.2265，高于正例率常数基线 0.2112；只有 rt49 的 0.2049 略优于基线。

## 5. Full-data 预期的表达边界

- 观测到的 capacity：512-event train MAE 0.049–0.076、R² 0.938–0.959。
- 观测到的系统锚点：历史 full-data validation MAE 0.2966、R² 0.5221。
- PPT 可使用 MAE 0.28–0.30、R² 0.52–0.60 作为实验规划区间，但必须标注“预期区间，非观测结果”。
- 最低验收标准是同时超过历史 full-data 的 MAE、R² 和 slope，并缩小 train–validation gap；不能把 512-event train 结果直接外推成 full-data validation 结果。

## 6. 跨论文对比必须保留的注释

- TEAM：概率输出、阈值预警、随时间评估，日本/意大利数据。
- QuakeFormer：California 2000–2024，约 20 万事件、400 万条记录，含 seen/unseen station 划分。
- 本项目：日本 K-NET/KiK-net，点预测 log PGA；当前探索性阈值表只取固定 20 s 快照。
- 因此数字只用于量级、研究差距和共性瓶颈定位，不构成严格 benchmark 排名。
