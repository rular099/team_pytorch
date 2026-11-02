# 来源、口径与定量结论登记

日期：2026-09-30。仓库：`rular099/team_pytorch`。分支：`exp/v01-velocity-prep-padding-control`。分析基点：`8f54477ed11bf2623f40121648f7fcc756ada1a2`。本轮未提交，正式文件哈希在 `native_deck_manifest.json` / `native_qa.json`。

## 本项目当前正式结果

权威目录：`reports/rt61_wave_geometry_validation_20260928/`。

- `summary.json`：split / checkpoint / training contract / source identity / fixed fields / bootstrap。
- `group_metrics.csv`：target-weighted MAE / RMSE / bias / R² / slope / NLL / Brier / coverage；PGA 坐标为 log10(m/s²)。
- `paired_ci.csv`：同一 event-ID 聚类，5,000 draws，seed 20260915；不是多训练种子的 CI。
- `field_metrics.csv`、`mechanism_gates.csv`、`legacy_gates.csv`、`useful_joint_progress_gates.csv`、`decision.json`：field 等权、n≥5；预注册验收，不由本报告重新选择门槛。
- `fixed_examples.csv` / `fixed_examples.png`：典型空间场案例的完整原始证据入口。主报告用汇总空间指标展示，不将挑选案例当总体证据。

RT61 身份：epoch 8、seed42，父模型 RT59 epoch8；rank16 adapter + remote readout，共 9 tensors / 99,621 scalars，其余冻结。Japan 2000–2024 全部 25 年训练 shards；训练/验证/测试分别为 9,627 / 1,383 / 2,769 events。新结果只使用固定 validation。

正式 NPZ SHA256：

```text
random: ace4310564a4dfba2af061ebee8134846e1aeb5fc7b10a0d6a5a363f1fbc0f8b
normal: e2fcba599345d18a23064f8ceaf39197f19052fdfb768320db99fd5f38e34599
```

point estimate 是 log10 PGA 坐标上的 MDN 混合均值。四组全部点与正式 CSV 逐项检查；源图及绘图参数记录在 `assets/tables/rt61_sample_figure_provenance.json`。Random / normal all / normal input / normal non-input 的有效目标数为 75,654 / 89,770 / 63,651 / 26,119。event ID 去重与有效 realtime row 单独计数，不把多个时刻台站记录当独立事件。

RT57 γ=1.66、RT59 reference、RT61 candidate 都取自 RT61 同次正式导出。相对 RT57，RT61 random MAE 下降 3.36%、normal non-input 下降 5.29%；相对 RT59 仅约 0.21%、0.03%。Normal non-input 的增量 MAE CI 包含零，RMSE略升；RT61 的四个 useful-progress gates 均未通过，故保留 RT59。

## 原报告与旧散点

原报告：`../team_pytorch/reports/technical_exchange/multistation_pga_exchange_20260726/multistation_pga_exchange_20260726.pptx`；追溯 Slide 11 的生成素材与引用。

旧散点原始 NPZ：`../team_pytorch/reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_best.npz`。

旧原图：`../team_pytorch/reports/pga_academic_report_assets_current/scatter_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets_best_val.png`；指标核对表：同目录 `main_pga_metrics.csv`。

核验为 101 validation rows、1,393 有效目标：MAE 0.2965706066、RMSE 0.3874600210、R² 0.5220562611、slope 0.5930814850。七月训练规模为 699 events，与目前协议和样本量不同；只作为历史背景对照，不报告跨协议的“严格提升百分比”。旧 checkpoint 的具体 epoch 本次未独立确认，引用身份为原 `best` 导出而不是猜测 epoch。

Slide 12 新旧轴域统一为 [-3.0, 1.1]，x/y 等比例，全部有效点保留。旧点 α=0.30、新点 α=0.08，在图下注明；点云深浅不能用来比较数量。其余当前散点 α=0.08、统一 [-2.1, 1.1] 轴域，配 1:1 参考线、OLS 和密度等高线。

## RT55 独立 test 历史锚点

`../team_pytorch/reports/rt55_rt56_current_results/rt55/tables/rt55_aggregate_metrics.csv` 及关联 metadata。身份核验为 RT55 epoch20，不按目录名推断成 epoch32。

normal all：170,767 targets / 2,769 events，MAE 0.1151、RMSE 0.1777、R² 0.7578；normal non-input：48,632 targets / 2,547 events，MAE 0.1947、RMSE 0.2527。该结果不是 RT61 test，也不用于本报告的候选选择或阈值调优。

## 公开文献固定版本

1. Münchmeyer, Bindi, Leser & Tilmann，TEAM：[arXiv:2009.06316v4 PDF](https://arxiv.org/pdf/2009.06316v4)。Supplement Table S1（PDF 第 28 页）日本 test：1%g F1 0.73 / PR-AUC 0.82；2%g 0.69 / 0.76；5%g 0.63 / 0.68。TEAM 为完整实时预警评估，概率决策阈值逐个 PGA 阈值优化；不等于本项目固定时刻快照。
2. Feng, Zhu & Lu，QuakeFormer：[arXiv:2412.00815v1 HTML](https://arxiv.org/html/2412.00815v1)。Fig.3 forecasting seen / unseen station R² 约 0.917 / 0.837（原图近似读数，非精确表格值）；Table3 的 ln 残差 σ 为 0.57 / 0.75。该对照来自 California catalog-parameter forecasting，不是相同实时波形任务；unseen station 是其 station-heldout 定义，不可对应本项目 non-input。
3. [NIED K-NET / KiK-net 数据说明](https://www.kyoshin.bosai.go.jp/en//about_pubdata/)，[官方数据记录](https://nied-repo.bosai.go.jp/records/6020)，DOI: 10.17598/NIED.0004。本项目这轮训练为过滤后的 K-NET-only split，不能泛称已经训练全部 K-NET/KiK-net 原始记录。

文献核验使用上述原文，不依赖非官方博客或二手性能榜。GMPE / ASK14 在方法定位页只作概念描述，不引用未经核验的独立性能数值。文献框架页采用原生概念摘要，不将其冒充论文原图。

## 本报告新增的派生指标

`assets/tables/rt61_threshold_metrics.csv`：1/5/20 s × 1/2/5%g，g=9.80665 m/s²，使用反归一化 MDN 的混合超阈值概率；固定 α=0.5，无 cutoff 优化。F1 和 tie-aware 梯形 PR-AUC 单列，AP 另存而不替代 PR-AUC；不计算预警提前量。20s random 有 9,603 targets / 1,310 events。用于描述性文献定位，不用于选模型。

`assets/tables/rt61_literature_residual_metrics.csv`：先移除残差 bias 的平方，再将 log10 残差标准差换为 ln 标准差。Random 为 0.7137、normal non-input 为 0.5652；不是将 RMSE 直接改名成 σ。地区、幅值定义和真实样本方差仍不相同，换单位不消除协议差异。

`rt61_scatter_strata.csv` / `rt61_error_strata.csv`：按实际 station_valid_count、截止时刻、目录震级、真实 PGA 和目标触发状态分层。M≥7 只有 6 个事件，不将 624 target-time records 当独立样本。不同 K / 时刻的目标集合不同，不能从描述性趋势推断单一因素的因果效应。

## 实施与未完成工作

模型架构来源：`docs/rt57_gtnp_v2_station_distinctive.md`、`docs/rt59_dual_objective_transport_v3.md`、`docs/rt61_wave_geometry_residual.md`；当前后续状态：`SESSION_SUMMARY.md` 和 `docs/v01_velocity_prep_padding.md`。V01 暂无已确认成功的 HPC 结果，不宣称性能改善。

本轮只读既有模型结果、生成报告，不运行新训练、不使用新 held-out test、不修改 RT55 代码、不提交或推送。
