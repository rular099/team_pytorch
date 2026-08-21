# 单页替换稿：文献与本项目的 3 s 数值对比

## Slide 1：文献与本项目的数值对比——本项目统一为 3 s

- 左栏保留 TEAM 日本 test 的阈值结果，并将本项目统一替换为 `rt48@3 s` 的 train / validation。
- TEAM 是从首个 P 波后 0.5 s 开始、每 0.1 s 更新、最长约 25 s 的概率预警评估；本项目是首个有效 P 波后固定 3 s 的点预测快照。
- 右栏保留 QuakeFormer Figure 3 的 forecasting 结果；该任务只使用事件参数和场地信息，不使用实时波形，因此没有“多少秒”的定义。
- 右栏本项目仅列 `rt48@3 s` 的 train / validation，并明确 train 只表征拟合容量。
- 版式角色：左右双栏定量对比；沿用原第 25 页的低饱和科研答辩风格。
- Required images：
  - 原第 25 页；严格编辑目标；保持标题层级、左右双栏、表格、口径提示框、配色和留白，仅替换本页指定文字与数字。

    ![Original comparison slide](../multistation_pga_exchange_20260726/origin_image/slide_25.png)

### 左栏精确数据

| PGA 阈值 | TEAM test F1 / PR-AUC | 本项目 rt48@3 s train F1 / PR-AUC | 本项目 rt48@3 s validation F1 / PR-AUC |
| --- | ---: | ---: | ---: |
| 1%g | 0.730 / 0.820 | 0.942 / 0.975 | 0.573 / 0.560 |
| 2%g | 0.690 / 0.760 | 0.920 / 0.965 | 0.545 / 0.548 |
| 5%g | 0.630 / 0.680 | 0.945 / 0.957 | 0.488 / 0.507 |

正例数：train 为 644 / 237 / 46，validation 为 113 / 56 / 19。

### 右栏精确数据

| 条件 / 口径 | R² | σ_ln |
| --- | ---: | ---: |
| QuakeFormer forecasting，seen held-out stations | ≈0.917 | 0.570 |
| QuakeFormer forecasting，unseen held-out stations | ≈0.837 | 0.750 |
| 本项目 rt48@3 s validation | 0.490 | 0.927 |
| 本项目 rt48@3 s train（容量参照） | 0.955 | 0.265 |

QuakeFormer R² 为 Figure 3 读图近似；其 forecasting 结果无实时波形、无固定秒数。σ 均为自然对数单位。
