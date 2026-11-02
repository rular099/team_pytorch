# 多台站 PGA 技术交流报告｜2026-09-30

已完成原生 PPT 排版：20 页主报告 + 6 页附录，16:9，内部技术交流。沿用 2026-07-26 旧报告的白灰底、蓝/青/珊瑚强调色和技术卡片风格；封面只有标题和日期。

## 交付文件

- [可编辑 PPTX](multistation_pga_exchange_20260930.pptx)：标题、正文、架构、卡片、表格均为原生对象，每页有中文讲稿备注。
- [PDF 预览](multistation_pga_exchange_20260930.pdf)：26 页固定版式，建议跨机器演示时备用。
- [逐页讲稿](speech.md)：与 PPT 备注一致。
- [来源与比较口径](sources.md)、[质量核验](QA.md)、[工程交接](CODEX_RESULT_20260930_NATIVE_TECHNICAL_REPORT.md)。

科研图不是可编辑的 PPT 图表，而是核验数据生成的原始 PNG 等比例嵌入；另提供 `assets/figures/` 下的 PDF/SVG 和 `assets/tables/` 下的源指标。7 张科研图未经过生成模型改绘。

## 本次新增的旧图对照

Slide 12 从旧报告 Slide 11 引用的原始 `eval_results_best.npz` 重绘：101 validation events、1,393 有效目标，MAE 0.2966、RMSE 0.3875、R² 0.5221、slope 0.5931。与 RT61 normal all、normal non-input 统一轴域、等比例显示；所有有效点保留。

这不是同数据、同目标、同截止时刻的配对试验。历史页只表达规模与结果变化，不把跨协议差异写成架构贡献百分比。严格提升另用 RT61 同次导出的 RT57 γ=1.66 / RT59 reference / RT61 candidate 配对比较。Slide 13–15 展示总体、台站数、截止时刻、真实 PGA、震级、目标触发状态的分层结果。

## 结果边界

- RT61：固定 epoch 8、seed 42，Japan 2000–2024 全部 25 年训练 shards；新结果仍为 validation，不是 held-out test。
- Random MAE 0.2373 / RMSE 0.3101 / R² 0.3994；normal all MAE 0.1250；normal non-input MAE 0.1992。
- RT61 的空间场与实用收益门槛未全通过，RT59 仍为 development parent。不能称 RT61 全面最优。
- 文献比较分别对照 TEAM 的阈值指标、QuakeFormer 的 R² / ln 残差标准差，同时明确实时快照、完整 EEW、catalog forecasting 和 seen/unseen station 的区别。
- V01 只报告准备状态，未虚构超算执行成功或性能结果。

## 本地复现

以下从本仓库根目录运行。依赖：Python、numpy、scipy、matplotlib、Pillow、python-pptx；PDF 导出使用 LibreOffice，预览使用 Poppler。当前已验证 Python 位于 `/home/zhangb/software/miniconda3/envs/general2/bin/python`。构图脚本使用当前机器的 Noto Sans CJK SC 字体文件；跨机器构建需先核对字体路径。PPT 未嵌入字体，其他机器若无该字体可能发生换行变化，PDF 不受此影响。

使用任务专属变量填写可信 NPZ 的实际位置：

```bash
REPORT_DIR=reports/technical_exchange/multistation_pga_exchange_20260930
RT61_RANDOM_NPZ=/tmp/rt61_ppt_source/eval_validation_epoch8_random.npz
RT61_NORMAL_NPZ=/tmp/rt61_ppt_source/eval_validation_epoch8_normal.npz
JULY_SCATTER_NPZ=../team_pytorch/reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_best.npz
REPORT_PYTHON=/home/zhangb/software/miniconda3/envs/general2/bin/python

"$REPORT_PYTHON" "$REPORT_DIR/prepare_sample_scatter.py" \
  --random-npz "$RT61_RANDOM_NPZ" --normal-npz "$RT61_NORMAL_NPZ"
"$REPORT_PYTHON" "$REPORT_DIR/prepare_report_assets.py" \
  --random-npz "$RT61_RANDOM_NPZ" --normal-npz "$RT61_NORMAL_NPZ" \
  --old-npz "$JULY_SCATTER_NPZ"
"$REPORT_PYTHON" "$REPORT_DIR/test_report_assets.py"
"$REPORT_PYTHON" "$REPORT_DIR/build_native_deck.py"
libreoffice -env:UserInstallation=file:///tmp/rt61_native_ppt_lo_20260930 \
  --headless --convert-to pdf --outdir "$REPORT_DIR" \
  "$REPORT_DIR/multistation_pga_exchange_20260930.pptx"
"$REPORT_PYTHON" "$REPORT_DIR/qa_native_deck.py" --render
```

原始 RT61 NPZ 来自用户提供的 `chaosuan_res/weights_japan_full_2000_2024_rt61_wave_geometry_residual_seed42.zip`；临时解压目录不是长期数据源。脚本检查 NPZ 的完整 SHA256、MDN 均值和正式指标；`allow_pickle=True` 仅用于这些已核验的可信导出，不用于未知文件。

预览在 `preview/`；人工审阅后才执行 `qa_native_deck.py --visual-review-complete` 记录通过，不能把纯结构检查当成版式审阅。

## 变更范围

仓库 `rular099/team_pytorch`；分支 `exp/v01-velocity-prep-padding-control`；分析基点 `8f54477ed11bf2623f40121648f7fcc756ada1a2`。本轮只新增本报告目录，没有修改训练、推理、采样器、超算启动器或 RT55 加载行为；没有提交、推送或启动新超算任务。历史图片样张保留在 `drafts/`、`origin_image/`，不属于正式图表证据。
