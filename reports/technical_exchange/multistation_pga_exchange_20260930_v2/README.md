# 多台站 PGA 技术报告：交流间对照修订版

日期：2026-09-30；22 页主报告 + 6 页附录。封面仅标题与日期；文字、表格与架构为可编辑原生 PPT 对象，科研图由核验数据绘制。

## 交付

- [修订版 PPTX](multistation_pga_exchange_20260930_v2.pptx)
- [固定版式 PDF](multistation_pga_exchange_20260930_v2.pdf)
- [第 6–8 页详细技术说明及指标答疑](TECHNICAL_DETAILS.md)
- [逐页讲稿](speech.md)、[来源与口径](sources.md)、[质量检查](QA.md)
- [工程交接](CODEX_RESULT_20260930_REPORT_REVISION.md)

采用 figure-designer 的证据与质量检查原则：统一新旧散点轴域、等比例坐标、透明度和来源标注；明确不同目标总体和评分口径，不将跨协议变化画成配对的架构收益。原 v1 报告未覆盖。

## 对七条意见的落实

| 用户意见 | 修订后的页码 / 材料 |
| --- | --- |
| 对比上次交流，不以 RT57 为起点 | 第 2、10–12 页从七月 full-data 和 rt48 KNET 结果起步；旧图原始 NPZ 重绘 |
| 第 14 页是否 random，补 normal | 第 14 页明确 random；新增第 15 页 normal，同一 RT61 ep8、seed42；均按实际 K 与 cutoff 分层 |
| 解释空间范围比；展示交流间变化 | 第 17 页改为旧 full-data / 本次 normal / 新增 random；解释 P95−P05 比值、直观例子与局限 |
| 概率页也比较两次交流 | 第 18 页用原报告 rt48 实际 validation 对照 RT55 normal，并保留最新 normal / random |
| 文献差距为何看起来变大 | 第 19 页补 normal，用旧点评分规则重算；第 20 页解释 train/val、normal/random、point/probability cutoff 的差异 |
| normal 展示过弱 | 第 2、3、10–13、15、17–20 页均展示 normal；all 与 non-input 分开 |
| 解释 RT57–61、架构、双路细节 | 第 6 页中文化概览；独立 TECHNICAL_DETAILS.md 完整展开，不增加主报告技术密度 |

## 当前代表结果与边界

七月 full-data validation MAE 0.2966 / R² 0.5221；七月最佳 KNET rt48 diagnostic MAE 0.2877，也保留而不是只挑较差基线。

本次 normal 的最低 MAE 代表为 RT55 ep20 validation：all MAE 0.1162 / R² 0.7524，non-input MAE 0.1949。ep32 的 R² 略高而 MAE 稍高，不把两个 checkpoint 的最优单项拼接。最新 RT61 normal all MAE 0.1250；新增 random MAE 0.2373。

本次新比较均为 **validation**。25 年全部训练 shards 的 split 为 9,627 / 1,383 / 2,769 train/val/test events，仍为过滤后的 K-NET-only 数据而非全部原始网络记录。旧/新事件、输入与目标协议不同，历史变化是描述性结果，不是单模块的严格因果增益。

文献页复现旧规则的 F1* 使用 validation 内最优 point-score cutoff，仅用于回顾；不是部署阈值或独立 test。固定 MDN 概率 cutoff=0.5 的结果仍保留于第 20、25 页和源表。TEAM 完整 EEW 与 QuakeFormer catalog forecasting 均不是同协议复现，不宣称直接胜出。

## 本地复现

从仓库根目录运行。当前 Python 为 `python`；需要 numpy/scipy/matplotlib/Pillow/python-pptx、LibreOffice、Poppler 与 Noto Sans CJK SC。字体未嵌入 PPT，跨机器展示建议备 PDF。

v2 重用 v1 的原生组件和已核验 RT61 科研素材，必须保留相邻 v1 目录。下列原始 NPZ 是本机用户提供的可信导出；SHA256 与 metadata 见源登记。临时 RT61 目录可从用户原始 zip 恢复。

```bash
REPORT_DIR=reports/technical_exchange/multistation_pga_exchange_20260930_v2
REPORT_PYTHON=python
RT55_RESULTS=../chaosuan_res/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42
JULY_KNET_RESULTS=../chaosuan_res/weights_japan_overfit_pga15_stage2_512_rt48_knet_cached_dpk_event_station_pool_temporal_residual_scale4
JULY_THRESHOLD_RESULTS=../chaosuan_res/weights_japan_overfit_pga15_stage2_512_rt48_cached_dpk_event_station_pool_temporal_residual_scale4

"$REPORT_PYTHON" "$REPORT_DIR/prepare_revision_assets.py" \
  --random-npz /tmp/rt61_ppt_source/eval_validation_epoch8_random.npz \
  --normal-npz /tmp/rt61_ppt_source/eval_validation_epoch8_normal.npz \
  --normal-config reports/rt61_wave_geometry_validation_20260928/training_config.sanitized.json \
  --rt55-npz "$RT55_RESULTS/eval_validation_best_normal.npz" \
  --rt55-config "$RT55_RESULTS/config.json" \
  --rt55-metrics "$RT55_RESULTS/eval_validation_best_normal.metrics.json" \
  --old-full-npz reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_best.npz \
  --old-knet-npz "$JULY_KNET_RESULTS/eval_results_last.npz" \
  --old-threshold-npz "$JULY_THRESHOLD_RESULTS/eval_results_last.npz"
"$REPORT_PYTHON" "$REPORT_DIR/test_revision_assets.py"
"$REPORT_PYTHON" "$REPORT_DIR/build_revision_deck.py"
libreoffice -env:UserInstallation=file:///tmp/rt61_native_ppt_lo_20260930_v2 \
  --headless --convert-to pdf --outdir "$REPORT_DIR" \
  "$REPORT_DIR/multistation_pga_exchange_20260930_v2.pptx"
"$REPORT_PYTHON" "$REPORT_DIR/qa_revision.py" --render
```

人工查看全部页面后才运行 `qa_revision.py --visual-review-complete`。`assets/tables/` 保存新增比较、分层、空间与阈值源表；`assets/figures/` 保存 PNG/PDF/SVG。PPT 内 8 张科研图等比例嵌入，无生成模型重绘、无丢点抽样。

仅新增本报告目录；未修改训练、推理、RT55 加载、采样器或超算脚本。未启动新超算任务，未提交或推送。
