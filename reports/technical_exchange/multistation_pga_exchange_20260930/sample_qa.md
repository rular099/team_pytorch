# Slide 13 样张质量审查

日期：2026-09-30。

**历史记录说明**：下文记述原生排版获批前的图片样张审查。用户已于 2026-09-30 同意改用原生 PPT；26 页正式 PPTX/PDF 已完成。生成式样张均未使用，最新正式核验见 `QA.md`、`native_qa.json`。下文源图哈希对应当时字号版本；正式交付源图哈希以 `assets/tables/rt61_sample_figure_provenance.json` 为准。

## 结论

`origin_image/slide_13.png` 是第三版**布局样张**，不是已批准的正式结果图。它展示四联散点图 + 三组结论卡的版式。文字和关键指标可用于审阅，但图片后端没有精确保留科学图的几何比例及原始像素内容，因此不能将该图片视为可复现的科学图导出。

原始科学图 `assets/figures/rt61_scatter_four_populations.png`、对应 PDF/SVG、指标表和 provenance 已完成正式结果交叉检查，可直接用于技术报告。建议改为原生 PPT 排版：图作为原始 PNG 等比例嵌入，标题、结论卡和其他正文采用可编辑对象。旧版风格、26 页页纲及封面约定不变。

## 原始科学证据

- 使用 RT61 epoch 8、seed 42 的 Japan 2000–2024 固定 validation 导出；不是 held-out test。
- 四组分别为 random、normal all、normal input、normal non-input。
- 原始图绘制所有有效目标点，无抽样，透明度 0.08；点数包含同一事件不同实时截止时刻的目标记录，不是独立事件数。
- 预测为 log10 PGA 坐标上的 MDN 混合均值，单位基准 m/s²。
- 两维轴域均为 [-2.1, 1.1]，源图 `aspect='equal'`；全部有效点均在轴域内。
- 点数、MAE、R²、斜率与正式 `reports/rt61_wave_geometry_validation_20260928/group_metrics.csv` 对应行一致，程序逐项检查容差为 2e-9。
- 完整来源、源 NPZ 的 SHA256、汇总表 SHA256、精确指标和渲染参数：`assets/tables/rt61_sample_figure_provenance.json`。
- 原始图 SHA256：`547ec3711109bbdd08a50e32e81d539271f7bd24e379d3acd115b0bfc6f60510`。

## 样张生成记录

选定后端：内置 `image_gen.imagegen`；本次没有切换后端。所有引用本地图片均先通过 `view_image` 检查。

1. 首版：`prompts/sample_slide_13.md`，输入原始四联图与旧版 Slide 11 风格参考。结果留存在 `drafts/slide_13_wide_axes.png`。QA：横向拉伸坐标，拒绝用于正式结果。
2. 修订：`prompts/sample_slide_13_repair.md`，输入首版与原始四联图。结果留存在 `drafts/slide_13_nonuniform_units.png`。QA：仍未等比例保留图；出现 y 轴单位 m/s³ 等非源图文字，拒绝用于正式结果。
3. 加强严格资产约束：`prompts/sample_slide_13_strict_asset.md`，重新以原始四联图为第一输入，旧版 Slide 11 为风格参考。结果选作 `origin_image/slide_13.png`。QA：版式改善、可见关键指标与正式四舍五入值一致；图的比例及内容仍并非对原始源图的无损等比嵌入。**仅为布局样张，不批准为正式科学图。**

第三版生成原文件：`/home/zhangb/.codex/generated_images/01a02dd8-fdf8-7962-941c-30f830375be2/exec-87c40b58-4874-4fa5-b38a-8a324929de73.png`；原文件保留。

第三版布局样张 SHA256：`e198f0a361e71835263c29bae45dc581bed92d9b4f53a1ff8621dea9efb79fbb`。与原始科学图区分记录，不代表源图的科学保真检查通过。

## 工作流限制与待决定事项

`codex-ppt` 的硬约束要求最终页面由选定图片后端生成，明确将本地排版和手动叠加列为不可使用的回退方案。因此本次不擅自将原图覆盖到生成页面，也不擅自切换制作流程。

待用户确认：是否保留已确认风格和页纲，改用原生可编辑 PPT 排版并直接嵌入原始科学图。若同意，本页重新作为原生样张生成，科学图不再通过图片模型转绘。

整套 26 页及 PPTX 尚未生成；未创建最终 `deck_spec.json`、批量页面任务或演讲备注；没有提交、推送或更改训练/推理代码。
