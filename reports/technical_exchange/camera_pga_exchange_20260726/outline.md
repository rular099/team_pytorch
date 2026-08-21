# 摄像头 PGA 估计技术交流：最新进展与阶段结论

状态：大纲待确认
计划日期：2026-07-26
建议页数：14 页
定位：在 2026-06-29 交流稿基础上压缩背景，增加 DPK/temporal residual、实时窗口、station-roll 反事实诊断和 KNET-only 对照。

## Slide 1：标题页

- 标题：基于视频摄像头与多台站波形的 PGA 估计
- 副标题：模型进展、反事实诊断与下一步验证
- 日期：2026-07-26
- 版式角色：封面
- 视觉建议：日本强震台站空间分布与波形纹理作为低对比背景。

## Slide 2：应用闭环与本次更新

- 摄像头视频先转换为相对振动曲线。
- 多台站模型为摄像头位置生成历史事件参考 PGA。
- 参考 PGA 与视频振动强度共同拟合摄像头标定系数。
- 本次更新重点不是重复应用背景，而是回答模型是否真正利用台站波形以及当前性能边界。
- 版式角色：工作流 + 本次汇报问题。

## Slide 3：任务定义与模型链路

- 输入：若干台站的三分量波形、台站位置和有效 mask。
- 查询：任意目标位置，包括没有强震仪的摄像头位置。
- 输出：目标位置的 log PGA / PGA。
- 模型链路：DiTing encoder → station adapter → TEAM-style Transformer → target cross-attention → PGA。
- 版式角色：任务示意 + 模型架构。
- Required images：
  - 多台站输入输出示意；严格输入资产；保留输入台站、目标位置和输出关系。

    ![Multi-station task](../assets/task_setup_multistation.png)

  - 模型结构图；严格输入资产；保留模块名称与连接关系。

    ![Architecture](../assets/architecture_implementation_detail.png)

## Slide 4：数据、评估口径与实验问题

- 数据来自日本 K-NET / KiK-net 强震记录；当前可核验结果为 mixed-source 设置。
- 指标在 log PGA 空间计算：MAE、RMSE、Corr、R²、slope 和 bias。
- 采用 train-first 诊断：先判断模型能否拟合，再判断这种能力能否泛化。
- 最新实验问题：DPK 先验和 temporal residual 能否保留台站波形差异并改善目标 PGA。
- 版式角色：实验设置。
- Required images：
  - 事件与台站空间分布；严格输入资产；保留地图内容与图例。

    ![Event station map](../../pga_academic_report_assets_current/data_event_station_map.png)

## Slide 5：rt44–rt51 实验设计

- rt44–rt47：DPK event temporal residual，token scale 为 0 / 2 / 4 / 8。
- rt48：station-pool + temporal residual。
- rt49：layerwise temporal readout。
- rt50：independent residual / uniform weighting 对照。
- rt51：station-roll control，检验 matched station-token 路径。
- 关键诊断同时看 normal eval 和 waveform-station roll mismatch eval。
- 版式角色：实验矩阵。

## Slide 6：最新主结果——总体精度

- 在本地完整 last-checkpoint 结果中，rt48 的 validation MAE 最低：0.2968，Corr 为 0.7241。
- rt46、rt49、rt51 的 validation MAE 分别为 0.3034、0.3018、0.3025，差异不大。
- DPK temporal scale 从 0 到 4 有小幅收益，但没有证据支持“scale 越大越好”。
- station-pool + temporal residual 是当前最有竞争力的组合；layerwise 与 independent 分支没有形成稳定优势。
- rt47 normal eval 文件残缺，主表不把它作为可比较结果。
- 版式角色：数据证据。
- Required images：
  - 新生成的 rt44–rt51 normal-eval 指标图；严格数据资产；数值来自 `chaosuan_res/*rt44*` 至 `*rt51*` 的 `eval_results_last.npz`。

## Slide 7：实时性——输入波形看多久

- 1 s 时 validation MAE 约为 0.38–0.40。
- 5 s 后多数模型降至约 0.28–0.30。
- 40 s 附近通常达到最好或接近最好表现；rt46 在 40 s 为 0.2586，rt48 为 0.2589。
- 90 s 并未稳定继续改善，说明更长输入不必然带来更好估计。
- 实际应用需要在预警时效与 PGA 精度之间选取工作点。
- 版式角色：时间序列 / 折线证据。
- Required images：
  - 新生成的 rt44–rt51 validation MAE–elapsed-time 折线图；严格数据资产。

## Slide 8：多台站信息是否有帮助

- 单台站输入下 validation MAE 约为 0.45–0.57。
- 6–10 台站后明显下降；11–15 台站区间多数模型达到最低误差。
- 16+ 台站不再单调改善，可能同时受到事件难度、空间分布和输入选择影响。
- 结论是“多台站有价值”，但不是简单的台站数越多越好。
- 版式角色：分桶结果。
- Required images：
  - 新生成的输入台站数量分桶图；严格数据资产。

## Slide 9：核心反事实——打乱波形与台站匹配

- 操作：事件内滚动打乱 waveform–station 对应关系，保持其余评估流程不变。
- train MAE 增量为 +0.0678 至 +0.1397，说明过拟合后模型确实使用了 matched waveform。
- validation MAE 仅增加 +0.0103 至 +0.0151，退化显著小于 train。
- rt48 的 train / validation 增量分别为 +0.1397 / +0.0151，是最清楚的代表案例。
- 版式角色：normal vs mismatch 对比。
- Required images：
  - 新生成的 normal 与 station-roll MAE 增量图；严格数据资产。

## Slide 10：诊断结论——瓶颈在泛化，不在可拟合性

- 当前模型有能力利用匹配的台站波形，因为 train mismatch 会明显恶化。
- 但这种 station-specific 依赖没有充分迁移到 validation。
- 这与 station-level 表征趋向 event-common、波形特征 collapse 的观察一致。
- DPK prior、temporal residual 和结构变体改善了总体误差，但尚未根治 station-use 泛化弱。
- 版式角色：结论推理链。
- 视觉建议：训练与验证两条路径的因果对照图。

## Slide 11：KNET-only 控制实验

- 目的：去除 K-NET / KiK-net、地表 / 井下传感器混合带来的域差异。
- 对照：同样的 rt44–rt51 结构、normal eval 和 station-roll mismatch eval。
- 核心判断：KNET-only 是否扩大 validation mismatch delta，并降低 feature collapse。
- 当前本地工作区没有找到 `_knet` 结果目录，不能用 mixed-source 数值替代。
- 版式角色：受控实验对比。
- Required images：
  - KNET-only rt44–rt51 normal/mismatch 结果；严格输入资产；待同步 `eval_results_last.npz` 后生成。

## Slide 12：当前误差结构与应用边界

- 弱 PGA 存在正偏、强 PGA 仍系统性低估，预测动态范围仍被压缩。
- 早期 1–3 s、未触发目标和长 lead-time 目标更困难。
- 摄像头标定不能只用全局平均 MAE，需要按强弱 PGA、时间窗口和目标类型分层验证。
- 在形成稳定的 KNET-only / station-use 结论前，模型适合作为参考 PGA 生成器，不应被表述为已完成的部署方案。
- 版式角色：风险与边界。

## Slide 13：对摄像头标定流程的直接含义

- 摄像头参考 PGA 应附带模型版本、输入时长和可用台站数量。
- 标定系数建议按事件留出验证，避免同一批事件同时拟合与评估。
- 强 PGA 事件需要单独检查，避免系统性低估传递到 `k_camera`。
- 传统方法、深度模型和视频估计必须统一事件、目标位置、单位与时间窗。
- 版式角色：方法落地。

## Slide 14：阶段结论与下一步

- 当前最佳 mixed-source 结果为 rt48：validation MAE 0.2968、Corr 0.7241。
- 多台站和更长波形窗口有效，但收益存在饱和。
- 最重要的新证据：模型在 train 能使用 matched waveform，在 validation 使用明显不足。
- 下一步优先完成 KNET-only normal/mismatch 对照和 feature-similarity 诊断。
- 随后再决定是否开展 KiK-net surface / borehole 分域实验，并推进摄像头标定闭环。
- 版式角色：总结与行动项。

## 需要用户确认的内容

1. 是否沿用“摄像头 PGA 标定”作为主叙事，并把模型诊断作为本次更新重点。
2. 是否接受 14 页结构。
3. KNET-only 结果是否已经同步到其他目录；若有，请提供目录或把结果放入当前工作区。
4. 标题页作者、单位是否需要补充。
