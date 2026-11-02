# 技术交流讲稿

## Slide 1: 多台站波形驱动的任意目标位置 PGA 估计

今天汇报随机观测几何下 PGA 预测的最新进展。重点看点误差、未观测目标、空间场和概率校准，以及这些结果是否足以支持下一轮模型决策。

---

注意点：
内部技术交流；主报告 20 页，附录 6 页。

来源：报告日期 2026-09-30；源代码分析基点 8f54477。

## Slide 2: 结论先行——点预测改善，空间场目标尚未完成

先给结论。RT61 的随机几何点误差略有改善，相对较早的 RT57 参考已有稳定收益；但相对 RT59 的增量很小，不能只凭最小 MAE 决定更换主线。空间差分仍未达到预注册门槛，所以当前保留 RT59。下面将把四类目标分开，避免把输入站的高精度当作远程推断能力。

---

注意点：
全页为 validation，单 seed 42。RT57 为同次导出的 γ=1.66 frozen historical reference，不是原始未缩放 RT57。

来源：group_metrics.csv；decision.json；useful_joint_progress_gates.csv。

## Slide 3: 任务升级——随机观测台站，预测随机目标位置

这里的随机几何不是把波形乱配到另一台站，而是在因果可用输入中随机选择台站。波形和真实台站坐标始终正确对应。Normal 允许目标与输入同址，random 正式评估排除输入站，因此二者的总体 MAE不能直接互相排名。远程位置预测是更关键的能力。

---

注意点：
不要将 non-input 自动称为 unseen station：当前没有独立的 station-heldout split。

来源：gemini_util_light.py；RT56/RT59/RT61 validation config；正式 observable route 导出。

## Slide 4: 相关工作——与 TEAM 最接近，与 QuakeFormer 任务可相互启发

TEAM 是最直接的任务参照；QuakeFormer 的价值是统一遮蔽任务和显式空间依赖。我们当前仍在冻结波形骨干的约束下尝试小规模残差机制。因此后面既报告预测误差，也报告阈值预警、空间差分和未观测站点指标，不能仅用一个 R² 概括系统。

---

注意点：
原文核验：TEAM supplement Table S1；QuakeFormer Fig. 3 / Table 3。图示为原生概念摘要，不冒充论文原图。

来源：https://arxiv.org/pdf/2009.06316v4；https://arxiv.org/html/2412.00815v1；ASK14 由 QuakeFormer 原文引用。

## Slide 5: 训练使用 25 年全量 shards；当前新结果仍全部是 validation

训练不是 512-event 子集。RT61 的 resolved contract 已确认 2000到2024全部25个年度 shards。但新的正式预测只来自固定验证集。Random 的1310个事件指有有效目标的事件，不应误写为重新分出的更小验证集。目标数是事件、实时截止时刻和目标站记录之和，不能当独立样本数。

---

注意点：
全量指本项目过滤后的训练划分，不是所有原始日本记录无过滤；test 绝不参与本轮选择。

来源：summary.json training_config_contract / counts；PROJECT_CONTEXT split snapshot。

## Slide 6: 技术演进——从容量诊断到点预测与空间场的联合验收

七月主要验证模型能否拟合，九月则转向任意观测几何和未观测目标。RT59 保留为主线，不是因为所有指标达标，而是它在既定边界下提供最有价值的点预测改进。RT60、RT61 是有明确机制问题的小实验，未晋级不代表毫无价值，也不应被表述为已经完成目标。

---

注意点：
本页为机制路线，不用不同轮次的非配对数值画连续增长曲线。

来源：RT57–RT61 CODEX_RESULT；各轮 RESULT_REVIEW；SESSION_SUMMARY。

## Slide 7: 当前系统架构——波形站点状态、目标查询与残差双路由

上排给出共享系统，DiTing 的时序 token、幅度与有效时长共同形成站点表示，而不是只看一个高度相似的池化向量。目标查询结合台站与事件记忆产生基准 MDN。RT59 增加同址和远程双分支；RT61只在远程 pair representation 上检验低秩波形—几何交互。对RT61而言，上游都被冻结。

---

注意点：
冻结状态针对 RT61 本轮，不等于所有历史训练阶段都冻结 TEAM。absolute u 与 residual d 同时使用，单站 d=0 不代表没有波形。

来源：docs/rt57_gtnp_v2_station_distinctive.md；docs/rt59_dual_objective_transport_v3.md；docs/rt61_wave_geometry_residual.md。

## Slide 8: RT59 双路由——RT61 只检验远程波形 × 几何增量

Local 的唯一匹配完全由 forward 可见坐标和mask决定，重复坐标的歧义情况落到 transport。RT61 的输出投影零初始化，teacher始终使用原RT59表示，因此可以精确比较同一次forward的候选和参考。重要的是，只训练新adapter和remote readout，不能把它的结果归因于重新训练全模型。

---

注意点：
共9个tensor / 99,621 scalars；forward 不读标签。normal input变化上限检查单独核验。

来源：RT61 trainable_manifest.json；reference_invariance.json；模型 forward。

## Slide 9: 训练目标——点误差、空间反差、概率与保留能力联合验收

分组加权避免数量很多的normal input目标压过random和remote目标。验收要求是合取关系，不能把18/25当作总体72分就自动晋级。RT61有13个机制门槛通过，但normal non-input RMSE仍轻微退化；实用进步还要求单站pairwise改善至少2%。

---

注意点：
ASCII: L = grouped_point + 0.05 * parent_regret + 0.40 * field_contrast.
Raw LaTeX: \mathcal L=\mathcal L_{point}+0.05\mathcal L_{regret}+0.40\mathcal L_{contrast}.

来源：RT59 / RT61 design docs；mechanism_gates.csv；legacy_gates.csv。

## Slide 10: 最新总体指标——random 小幅最优，normal 保留但尚未晋级

表格突出的是最新候选行，不是对所有指标都标成最佳。Random MAE和RMSE都有小幅收益；normal all差异很小，normal non-input RMSE由0.25565变成0.25572，略有退化。这样的变化和置信区间共同说明，当前不能宣布一个全面优于父模型的新方案。

---

注意点：
所有目标数固定；表中RT57使用γ=1.66 historical export。

来源：group_metrics.csv；decision.json；legacy / mechanism / useful_joint_progress_gates.csv。

## Slide 11: 严格同口径提升——RT57 γ=1.66 → RT59 → RT61

同口径指同一组事件、时间和目标的paired comparison。相对RT57的random MAE降低3.36%，normal non-input降低5.29%；相对RT59则只降低约0.21%和0.03%。CI按事件聚类重采样，避免把同一事件的多个时刻与台站当成独立样本。

---

注意点：
曲线纵轴为局部范围，标注精确数值；没有把跨模型bootstrap当作multi-seed不确定性。5000 draws, seed 20260915。

来源：group_metrics.csv；paired_ci.csv；paired_mae_progress.png。

## Slide 12: 旧版散点图对照——点误差更低，但远程目标仍有压缩

用户提到的旧散点确实存在。这里从原始NPZ重绘，确认1393个有效目标和原报告指标完全一致。当前normal all明显更接近1:1线，但它包含大量输入台站目标；只看normal non-input则仍有明显压缩，R²也低于旧版。数据和目标协议不同，不能把这页的差异直接解释为模型架构带来的提升。

---

注意点：
旧图best checkpoint epoch未在本次独立核验；只使用旧评估导出身份。旧版原图保持在历史目录，本页不是从生成式幻灯片抠图。

来源：old_vs_rt61_metrics.csv；report_asset_provenance.json；July Slide 11 prompt / main_pga_metrics.csv；旧eval_results_best.npz。

## Slide 13: 真实 PGA 与预测 PGA——输入站更准，远程目标仍有压缩

先看normal input，它最接近对角线。再看normal non-input和random，拟合线更平，强震端容易被拉回事件平均水平。这里的点估计是在log10坐标上的混合均值，不是取最强分量，也不是在线性PGA空间求期望。原始图直接嵌入，无裁切、无拉伸，所有有效点均参与统计。

---

注意点：
原始PNG 3072×2520，PPT中只等比缩放。点数为target-time记录，不是独立事件。

来源：rt61_sample_panel_metrics.csv；rt61_sample_figure_provenance.json；正式random / normal epoch8 NPZ。

## Slide 14: 散点分层——多站改善明显，时间收益不是单调关系

上排按真实有效K分组，单站的整体斜率只有0.1886，说明它最容易接近常数输出。下排按时间分组，early的MAE反而更低，但后期R²提高，这不矛盾，因为不同组的真实PGA分布和目标集合变化。Random每个时刻重新选择输入，不能把这页当作严格同一场景时间扫参。

---

注意点：
每组全量有效点，相同轴域和透明度；early=1/3/5，mid=10/20，late=40/90秒。

来源：rt61_scatter_strata.csv；正式 NPZ station_valid_count / requested_elapsed_time。

## Slide 15: 误差分层——少站、高 PGA 与高震级仍是主要难点

只看总体平均会掩盖长尾。随机几何中高PGA分桶的MAE约0.551，明显高于总体0.237。M≥7只有6个事件，同一事件多时刻多台站形成624个记录，不能据此给出很窄的泛化保证。台站数趋势也受到事件难度和几何覆盖影响，不能简单认定越多台站必然越好。

---

注意点：
分层为random正式目标；不把按幅值分桶的偏差解释成单独的训练因果机制。

来源：rt61_error_strata.csv；正式 strata_metrics.csv。

## Slide 16: 空间场指标——小幅恢复反差，仍未达到预注册目标

这里在每个事件—时间场内比较不同query，所以能够回答系统是否真的生成空间差异。RT61相对RT59恢复了一些范围，但相对RT57历史参考仍更压缩。单站pairwise MAE的改善虽然CI排除0，效应只有0.385%，未达到预设2%，不宜把统计显著写成实用突破。

---

注意点：
空间范围采用P95−P05，不是max−min；field等权，至少5个有效目标。完整固定案例可在原始fixed_examples.png审阅。

来源：summary.json random_fields；field_metrics.csv；paired_ci.csv；fixed_examples.csv。

## Slide 17: 概率输出与校准——概率分数改善，coverage 保真仍不够

NLL和Brier相对RT57都改善，说明均值修正有概率收益。但这不是联合优化不确定性的结果，RT59到RT61基本不改变logits和sigma。coverage变化超过保真门槛说明输出分布的行为发生变化，不等于它一定朝坏方向变化。混合分布可能非高斯，所以图中的高斯覆盖率只是参考。

---

注意点：
Brier使用正式评估既定阈值，与后续1/2/5%g探索表分开。pp是百分点。

来源：group_metrics.csv；legacy_gates.csv；reference_invariance.json。

## Slide 18: 文献指标对照——用于定位差距，不构成直接排名

TEAM表给出了相同PGA阈值的文献数字，RT61则使用MDN超阈值概率和固定0.5概率cutoff。我们没有在验证结果上优化F1，因此不与TEAM的F1最优cutoff作胜负结论。QuakeFormer一侧是catalog forecasting而不是相同的实时波形任务；残差std已换为ln单位，但地区、PGA定义和样本方差仍不同。RT55 ep20 test单列作历史锚点。

---

注意点：
1%g / 2%g / 5%g用g=9.80665m/s²。σ为去bias后的残差std，不是RMSE。TEAM PR-AUC为表中AUC结合PR曲线的解释。

来源：https://arxiv.org/pdf/2009.06316v4 (Table S1)；https://arxiv.org/html/2412.00815v1 (Fig.3, Table3)；rt61_threshold_metrics.csv；rt61_literature_residual_metrics.csv；RT55 aggregate CSV。

## Slide 19: 当前问题——空间压缩、单站、长尾与独立泛化证据

问题需要分事实和假设。我们已经能看到空间场压缩和长尾困难，但不能仅凭这些现象断言DiTing骨干一定不适合PGA。冻结骨干、补零、幅度域和仪器响应都有可能影响结果，需要受控实验隔离。开发验证集已被反复使用，最终机制通过后仍需要独立泛化证据。

---

注意点：
不将任意non-input位置称为真正训练未见station；不给出尚未运行的RT61 test数字。

来源：formal metrics / gates / decision；SESSION_SUMMARY。

## Slide 20: 阶段结论——保留 RT59，下一步先完成 V01 受控数据实验

总结不是继续重复当前设置。RT61已经回答了低秩波形—几何增量的可用性，但效应不足以换主线。V01将速度输入和P前缺失拆成受控问题，保持事件、坐标和时刻一致。现阶段报告只说明准备状态，不能声称V01已经带来收益。新的候选机制通过后，再安排held-out test而不是用test继续开发。

---

注意点：
V01从预注册RT55 ep32起点，不是从RT61开始。速度灵敏度换算不等于完整去响应。

来源：decision.json；SESSION_SUMMARY；docs/v01_velocity_prep_padding.md。

## Slide 21: 附录｜RT55–RT61 的实验身份与模型决策

这页用于复核实验身份。RT57原始epoch6结果和后续gamma=1.66冻结参考不是同一个数值口径，因此都注明。RT60和RT61同从RT59固定parent出发，并非RT61继承RT60。RT55 test目录名曾误导checkpoint身份，本报告只认metadata核验的epoch20。

---

注意点：
目录名不是checkpoint身份；不同轮次不以小数尾差作为新发现。

来源：RT55/56 PROJECT_CONTEXT；RT57–61 CODEX_RESULT；RT61 source identity。

## Slide 22: 附录｜RT61 完整指标、formal counts 与概率输出

完整表用于回答口径问题。点误差是target-weighted，空间指标则另用field等权。预测sigma是混合分布的二阶矩标准差。coverage使用均值正负sigma覆盖，不是按分位数计算的精确interval。有效行数按至少一个有效目标统计；唯一事件数按导出event ID去重，不能从总行数除以七反推。

---

注意点：
全部metric来自正式CSV；events/有效rows来自正式NPZ的valid target mask。

来源：group_metrics.csv；rt61_sample_figure_provenance.json。

## Slide 23: 附录｜TEAM 风格阈值指标——固定 cutoff 与时间快照

阈值先从%g转换到m/s²再取log10，随后对每个MDN分量求tail probability并按mixture weight求和。固定0.5判定计算F1；PR曲线对所有唯一概率分数分组，避免ties影响。图中normal non-input高阈值正例很少。源CSV还提供其他时刻用于透明审阅，没有用这些指标选择模型。

---

注意点：
Terminal: p_exceed = sum_m weight[m] * P_m(PGA >= threshold).
Raw LaTeX: p=\sum_m w_m\Pr_m(Y\geq y_{th}).

来源：rt61_threshold_metrics.csv；report_asset_provenance.json；TEAM Table S1。

## Slide 24: 附录｜RT61 参数边界——99,621 scalars，其余冻结

参数量准确来自运行时manifest，而不是手算层尺寸。额外小delta checkpoint仍依赖精确RT59 parent和共享状态hash，不能单独加载成完整模型。本报告制作没有改变任何训练或推理入口，因此不会影响现有超算训练和RT55兼容路径。

---

注意点：
不宣称本报告额外重跑了全部模型回归；沿用已有兼容性验证记录。

来源：trainable_manifest.json；reference_invariance.json；RT61 design doc；本报告文件变更清单。

## Slide 25: 附录｜统计、门槛与复现边界

门槛不是由本次画图重新制定，而来自预注册设计。特别是2%单站pairwise门槛不同于legacy绝对0.33门槛，两者在文件中都有独立记录。本报告不新增置信区间、不改变原有决策，只用源CSV和bootstrap结果解释边界。

---

注意点：
CI来自既有正式报告；没有新增bootstrap选择或test阈值优化。

来源：paired_ci.csv；mechanism_gates.csv；legacy_gates.csv；useful_joint_progress_gates.csv。

## Slide 26: 附录｜参考文献与结果来源

交付目录包含科学图的PNG、PDF和SVG，以及派生指标、原始来源hash、构建脚本和逐页讲稿。PPT的文字、卡片、表格和架构都是原生对象；科学图作为原图等比例嵌入。旧图和新图的所有关键数字均可回到原NPZ或正式CSV核验。

---

注意点：
本报告未提交新超算任务、未打开heldout test、未推送代码。公开文献引用固定v4/v1以避免版本漂移。

来源：report_asset_provenance.json；native_deck_manifest.json；sources.md；speech.md。
