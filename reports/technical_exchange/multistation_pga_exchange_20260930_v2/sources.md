# v2 来源与比较口径

2026-09-30；`rular099/team_pytorch` / `exp/v01-velocity-prep-padding-control`；代码基点 `8f54477ed11bf2623f40121648f7fcc756ada1a2`。旧 v1 的 [源登记](../multistation_pga_exchange_20260930/sources.md) 对保留的 RT61 图、formal gates 与数据身份继续适用；本文件覆盖已替换的历史主线和文献阈值规则。

## 七月锚点

原报告 `../team_pytorch/reports/technical_exchange/multistation_pga_exchange_20260726/`：PPTX、`quantitative_comparison_notes.md`、讲稿与第 11/25/26/27 页素材逐项核对。

| 材料 | 原始来源（仓库根目录相对路径） | 用途 / 样本 |
| --- | --- | --- |
| full-data 旧 Slide11 | `reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_best.npz` | 101 validation rows / 1,393 targets；回归散点与 n≥5 field 比值重算 |
| rt48 KNET 旧 Slide26/27 | `../chaosuan_res/weights_japan_overfit_pga15_stage2_512_rt48_knet_cached_dpk_event_station_pool_temporal_residual_scale4/eval_results_last.npz` | 364 validation rows / 4,082 targets；回归、经验 sigma coverage、Brier |
| rt48 旧 Slide25 | `../chaosuan_res/weights_japan_overfit_pga15_stage2_512_rt48_cached_dpk_event_station_pool_temporal_residual_scale4/eval_results_last.npz` | 混合网络诊断；20s 643 targets，1/2/5%g 正例 125/55/13；不与 KNET 导出混淆 |

旧 full-data checkpoint epoch 本次未独立确认，只引用原 `best` 导出身份。旧导出没有完整事件 ID，不从 realtime rows 推测新式唯一事件数。上次 Slide27 的 13.8% / 27.0% 是八次实验均值；本次实际 rt48 的 1σ/2σ coverage 为 11.3% / 21.4%，不用均值冒充单模型。

## 本次 normal 代表与最新研究结果

RT55 ep20：`../chaosuan_res/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/` 的 `eval_validation_best_normal.npz`、同名 `.metrics.json` 与 `config.json`。先核验 epoch=20、split=val、waveform permutation=none；all targets=89,770，non-input=26,119。all MAE/RMSE/R²/slope/Brier 与 formal metrics 逐项核对，不相信文件名的 best。

RT55 ep20 与 ep32 validation/test 的既有汇总：`../team_pytorch/reports/rt55_rt56_current_results/rt55/tables/rt55_aggregate_metrics.csv`。ep20 是所核查 formal normal 候选中最低 MAE 代表，并非每一项都优于 ep32。既有 test 不用于该展示选择或 cutoff 优化。

RT61 epoch8 seed42：`reports/rt61_wave_geometry_validation_20260928/` formal tables 与 `summary.json`。原始 NPZ 来自用户上传结果 zip，工作读取副本在 `/tmp/rt61_ppt_source/`；normal/random 的 SHA256 必须与 formal summary 一致。反归一化用 **training resolved config**，而非尚未展开继承的 validation config。normal all/input/non-input 与 random 不混合。

六份 NPZ 及 RT55 metadata 的 SHA256 记录于 `assets/tables/revision_summary.json`。新增图与 native slide 映射/图像哈希记录于 `native_deck_manifest.json`；最终 PPT/PDF 哈希在 `native_qa.json`。

## 派生图和指标规则

- `meeting_comparison_metrics.csv`：有效 target 加权；坐标 log10(m/s²)，residual std 转 ln 时去掉 bias 平方。不是把 RMSE 改名为 sigma。
- `meeting_scatter_comparison`：旧 full-data → RT55 ep20 normal all/non-input → RT61 random。统一 [−3.0,1.1]，等比例；全部有效点；旧 alpha=0.30、新 alpha=0.08。密度深浅不是样本数量排名。
- `normal_geometry_strata.csv`：RT61 同 checkpoint，K 来自实际 valid station 计数；按 K=1/2–5/≥6 与 1–5/10–20/40–90s；图 all-target、右侧 remote，源表完整保留两种总体。
- `meeting_field_metrics.csv`：同一导出 row 的 n≥5 有效目标，P95−P05 比值 field 等权。分母零跳过，不加 epsilon；范围比不证明空间排序正确，也不是地图面积。
- `meeting_threshold_comparison.csv`：20s，g=9.80665m/s²。旧规则为 point-score F1*（validation 内最优 cutoff）和 tie-aware 梯形 PR-AUC；另存 AP，绝不把 AP 标成 PR-AUC。F1* 是 retrospective oracle，不是部署或独立 test。MDN 固定 probability cutoff=0.5 也单列保留。
- 旧 rt48@20s validation 的三组 F1* / PR-AUC 重算结果逐项断言等于原 PPT 四舍五入数字：0.620/0.642、0.545/0.536、0.438/0.372。
- `meeting_calibration_progress`：旧 rt48 KNET 与本次 RT55 normal / RT61 normal / RT61 random。Brier 使用既定 logPGA≥−1.2；coverage 使用 mean±sigma/2sigma，不是 MDN 精确分位数区间。

## 文献原文

[TEAM v4 Table S1](https://arxiv.org/pdf/2009.06316v4)：日本 test 的 1/2/5%g F1 / PR-AUC 为 0.73/0.82、0.69/0.76、0.63/0.68，完整 EEW 时间过程并优化 cutoff。本项目仅 20s 快照，不能按表格直接排名。

[QuakeFormer v1](https://arxiv.org/html/2412.00815v1)：Fig.3 forecasting seen/unseen station 的 R² 原图近似 0.917/0.837，Table3 的 ln residual std=0.57/0.75。California catalog forecasting、station-heldout 与我们的 normal/random/non-input 定义不同，数值近似标识保留。

旧报告的高分包括训练结果：1/2/5%g F1=0.965/0.952/0.932，PR-AUC=0.995/0.985/0.985。不可拿它替代其 validation。没有足够证据把全部文献差距归因于某一个结构或数据因素；第 20 页解释的是已核验的展示口径差异。

历史变化非配对，不重算 paired CI、不伪造纯架构提升；保留的 RT61 vs RT59 gates/CI 只用于内部模型决策。没有新训练、模型推理或 held-out test 运行。
