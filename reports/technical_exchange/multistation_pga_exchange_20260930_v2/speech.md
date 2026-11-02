# 修订版逐页讲稿

## Slide 1: 多台站波形驱动的任意目标位置 PGA 估计

本次汇报以七月交流为起点，先展示常规 normal geometry 的系统进展，再介绍新增 random 任务、空间场瓶颈和下一步数据机制实验。研究过程与实现细节在附录及独立技术说明中。

注意点：
内部技术交流；主报告 22 页，附录 6 页。

来源：报告日期 2026-09-30；源代码分析基点 8f54477。

## Slide 2: 从上次交流到本次——normal 精度大幅改善，新增 random 挑战

先回答上次到这次是否进步。最强normal代表的整体MAE已降到0.1162，独立test另有证据。Random是新增的更难任务，不应把它较低的R²误写成常规场景退步。旧KNET rt48的0.2877也保留在后表，不回避上次最佳诊断候选。

注意点：
RT55 ep20为本次已完成normal validation中最低MAE的代表，ep32的R²略高；不拼接不同模型最优单项。

来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 3: 任务升级——随机观测台站，预测随机目标位置

这里的随机几何不是把波形乱配到另一台站，而是在因果可用输入中随机选择台站。波形和真实台站坐标始终正确对应。Normal 允许目标与输入同址，random 正式评估排除输入站，因此二者的总体 MAE不能直接互相排名。远程位置预测是更关键的能力。

注意点：
不要将 non-input 自动称为 unseen station：当前没有独立的 station-heldout split。

来源：gemini_util_light.py；RT56/RT59/RT61 validation config；正式 observable route 导出。

## Slide 4: 相关工作——与 TEAM 最接近，与 QuakeFormer 任务可相互启发

TEAM 是最直接的任务参照；QuakeFormer 的价值是统一遮蔽任务和显式空间依赖。我们当前仍在冻结波形骨干的约束下尝试小规模残差机制。因此后面既报告预测误差，也报告阈值预警、空间差分和未观测站点指标，不能仅用一个 R² 概括系统。

注意点：
原文核验：TEAM supplement Table S1；QuakeFormer Fig. 3 / Table 3。图示为原生概念摘要，不冒充论文原图。

来源：https://arxiv.org/pdf/2009.06316v4；https://arxiv.org/html/2412.00815v1；ASK14 由 QuakeFormer 原文引用。

## Slide 5: 训练使用 25 年全量 shards；当前新结果仍全部是 validation

训练不是 512-event 子集。RT61 的 resolved contract 已确认 2000到2024全部25个年度 shards。但新的正式预测只来自固定验证集。Random 的1310个事件指有有效目标的事件，不应误写为重新分出的更小验证集。目标数是事件、实时截止时刻和目标站记录之和，不能当独立样本数。

注意点：
全量指本项目过滤后的训练划分，不是所有原始日本记录无过滤；test 绝不参与本轮选择。

来源：summary.json training_config_contract / counts；PROJECT_CONTEXT split snapshot。

## Slide 6: 技术演进——每轮解决什么问题，而不只是实验名字

这页将名字翻译成研究问题。RT57解决站点表示相似，RT58尝试波形锚点迁移，RT59把同址与远程分开，RT60只更新最后读出，RT61再加入低秩条件交互。具体数学、输入、冻结边界和失败原因放在独立答疑文档。

注意点：
RT60不是RT61的parent；不宣称五次续训都带来全面提升。

来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 7: 当前系统架构——波形站点状态、目标查询与残差双路由

上排给出共享系统，DiTing 的时序 token、幅度与有效时长共同形成站点表示，而不是只看一个高度相似的池化向量。目标查询结合台站与事件记忆产生基准 MDN。RT59 增加同址和远程双分支；RT61只在远程 pair representation 上检验低秩波形—几何交互。对RT61而言，上游都被冻结。

注意点：
冻结状态针对 RT61 本轮，不等于所有历史训练阶段都冻结 TEAM。absolute u 与 residual d 同时使用，单站 d=0 不代表没有波形。

来源：docs/rt57_gtnp_v2_station_distinctive.md；docs/rt59_dual_objective_transport_v3.md；docs/rt61_wave_geometry_residual.md。

## Slide 8: RT59 双路由——RT61 只检验远程波形 × 几何增量

Local 的唯一匹配完全由 forward 可见坐标和mask决定，重复坐标的歧义情况落到 transport。RT61 的输出投影零初始化，teacher始终使用原RT59表示，因此可以精确比较同一次forward的候选和参考。重要的是，只训练新adapter和remote readout，不能把它的结果归因于重新训练全模型。

注意点：
共9个tensor / 99,621 scalars；forward 不读标签。normal input变化上限检查单独核验。

来源：RT61 trainable_manifest.json；reference_invariance.json；模型 forward。

## Slide 9: 训练目标——点误差、空间反差、概率与保留能力联合验收

分组加权避免数量很多的normal input目标压过random和remote目标。验收要求是合取关系，不能把18/25当作总体72分就自动晋级。RT61有13个机制门槛通过，但normal non-input RMSE仍轻微退化；实用进步还要求单站pairwise改善至少2%。

注意点：
ASCII: L = grouped_point + 0.05 * parent_regret + 0.40 * field_contrast.
Raw LaTeX: \mathcal L=\mathcal L_{point}+0.05\mathcal L_{regret}+0.40\mathcal L_{contrast}.

来源：RT59 / RT61 design docs；mechanism_gates.csv；legacy_gates.csv。

## Slide 10: 总体指标——上次基线、本次 normal 与新增 random 同页呈现

本页主线改为交流间比较。RT55保留的normal能力是重要应用成果，RT61展示最新研究能力。旧数据集和现在不同，random也不是旧normal的直接续跑，因此不在表格里做单一胜负排行。

注意点：
normal ep32不是ep20；表中全部为validation，独立test另行标注。

来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 11: 交流间的主要增量——更大规模、常规场景精度与任务范围

柱图从上次交流起点出发，包含旧full-data和旧最佳KNET诊断。现在normal的误差低很多；random任务更难，因此其R²低于旧锚点并不否定normal进步。历史差值是描述性变化，不是严格配对的因果收益。

注意点：


来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 12: 旧版散点对照——normal 更贴近真实值，random 仍有明显压缩

这页直接满足旧图与新图对照。Normal代表来自RT55 ep20，random来自最新RT61。Normal all明显更贴近1:1，远程目标与random仍压缩；旧图与当前不同样本，点云密度不直接比较数量。

注意点：


来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 13: 真实 PGA 与预测 PGA——输入站更准，远程目标仍有压缩

先看normal input，它最接近对角线。再看normal non-input和random，拟合线更平，强震端容易被拉回事件平均水平。这里的点估计是在log10坐标上的混合均值，不是取最强分量，也不是在线性PGA空间求期望。原始图直接嵌入，无裁切、无拉伸，所有有效点均参与统计。

注意点：
原始PNG 3072×2520，PPT中只等比缩放。点数为target-time记录，不是独立事件。

来源：rt61_sample_panel_metrics.csv；rt61_sample_figure_provenance.json；正式random / normal epoch8 NPZ。

## Slide 14: Random geometry 分层——台站数与截止时刻（RT61）

明确纠正原第14页标识：全部是random geometry结果。输入台站从因果可用站中随机选择，目标排除输入站。这页与下一页使用同一个RT61 checkpoint，但目标和台站集合不同。

注意点：


来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 15: Normal geometry 分层——多站与后期预测显著改善（RT61）

这里新增normal分层。随时间增加，all-target MAE从early约0.189到late约0.055，明显优于random趋势；但remote从0.201到0.187，改善小很多。必须解释目标组成变化，不能把后期高R²推广到远程任意位置。

注意点：
early=1/3/5、mid=10/20、late=40/90秒；实际K由station_valid_count核验。

来源：normal_geometry_strata.csv；TECHNICAL_DETAILS.md。

## Slide 16: Random 误差分层——少站、高 PGA 与高震级仍是主要难点

只看总体平均会掩盖长尾。随机几何中高PGA分桶的MAE约0.551，明显高于总体0.237。M≥7只有6个事件，同一事件多时刻多台站形成624个记录，不能据此给出很窄的泛化保证。台站数趋势也受到事件难度和几何覆盖影响，不能简单认定越多台站必然越好。

注意点：
分层为random正式目标；不把按幅值分桶的偏差解释成单独的训练因果机制。

来源：rt61_error_strata.csv；正式 strata_metrics.csv。

## Slide 17: 空间强弱反差——normal 恢复较好，random 仍明显压缩

空间范围比是场内PGA反差而非地图面积。使用旧full-data原NPZ重算，与本次normal和random用同一公式比较。Normal总体约0.83，random约0.49，说明最强点预测并未解决随机场动态范围。范围比1也不保证空间形状正确。

注意点：
图中normal代表为RT55 ep20；完整源表同时给RT61 normal all 0.8278、non-input 0.6800，以及单站结果。

来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 18: 概率输出的交流间进展——明显缓解上次的过度自信

这页不再从RT57开始。旧PPT的概率问题是KNET八轮模型过度自信，选其rt48实际导出对照，本次normal代表与最新normal/random都展示。Brier统一既定−1.2logPGA阈值。覆盖率改善很明显，但非高斯MDN不宜仅凭两条高斯参考线宣布完全校准。

注意点：
旧rt48为4,082 targets，当前normal为89,770，random为75,654，不能把coverage变化解释成配对百分点因果提升。

来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 19: 文献与两次交流对照——补齐 normal，并复现旧评分规则

文献页补齐normal代表和remote。旧三行点评分F1和PR-AUC已经从原NPZ精确重算，现在按相同point score、相同20秒和最优validation cutoff规则显示。F1星号只是探索性上限，不能视为部署性能。右表补旧full-data和最强normal，而不再只放random。

注意点：
TEAM论文表为概率预警完整时间；QF为catalog forecasting且station-heldout，R²原图近似；current non-input不等于unseen station。

来源：https://arxiv.org/pdf/2009.06316v4；https://arxiv.org/html/2412.00815v1；meeting_threshold_comparison.csv。

## Slide 20: 为什么上一版文献表显得变差——三种口径混在了视觉主线里

用户记忆有其来源：旧图中训练高分很突出，旧F1阈值在validation内优化。上一版还把更难random放在正中央，弱化normal。因此不能把视觉上差很多解释成系统全面退步。现在既呈现旧规则对照，也保留固定MDN阈值。

注意点：
没有用新阈值选择checkpoint或打开test；数据任务差异只作边界说明，不当作已经证明的全部原因。

来源：assets/tables/revision_summary.json；TECHNICAL_DETAILS.md

## Slide 21: 当前问题——空间压缩、单站、长尾与独立泛化证据

问题需要分事实和假设。我们已经能看到空间场压缩和长尾困难，但不能仅凭这些现象断言DiTing骨干一定不适合PGA。冻结骨干、补零、幅度域和仪器响应都有可能影响结果，需要受控实验隔离。开发验证集已被反复使用，最终机制通过后仍需要独立泛化证据。

注意点：
不将任意non-input位置称为真正训练未见station；不给出尚未运行的RT61 test数字。

来源：formal metrics / gates / decision；SESSION_SUMMARY。

## Slide 22: 阶段结论——保留 RT59，下一步先完成 V01 受控数据实验

总结不是继续重复当前设置。RT61已经回答了低秩波形—几何增量的可用性，但效应不足以换主线。V01将速度输入和P前缺失拆成受控问题，保持事件、坐标和时刻一致。现阶段报告只说明准备状态，不能声称V01已经带来收益。新的候选机制通过后，再安排held-out test而不是用test继续开发。

注意点：
V01从预注册RT55 ep32起点，不是从RT61开始。速度灵敏度换算不等于完整去响应。

来源：decision.json；SESSION_SUMMARY；docs/v01_velocity_prep_padding.md。

## Slide 23: 附录｜RT55–RT61 的实验身份与模型决策

这页用于复核实验身份。RT57原始epoch6结果和后续gamma=1.66冻结参考不是同一个数值口径，因此都注明。RT60和RT61同从RT59固定parent出发，并非RT61继承RT60。RT55 test目录名曾误导checkpoint身份，本报告只认metadata核验的epoch20。

注意点：
目录名不是checkpoint身份；不同轮次不以小数尾差作为新发现。

来源：RT55/56 PROJECT_CONTEXT；RT57–61 CODEX_RESULT；RT61 source identity。

## Slide 24: 附录｜RT61 完整指标、formal counts 与概率输出

完整表用于回答口径问题。点误差是target-weighted，空间指标则另用field等权。预测sigma是混合分布的二阶矩标准差。coverage使用均值正负sigma覆盖，不是按分位数计算的精确interval。有效行数按至少一个有效目标统计；唯一事件数按导出event ID去重，不能从总行数除以七反推。

注意点：
全部metric来自正式CSV；events/有效rows来自正式NPZ的valid target mask。

来源：group_metrics.csv；rt61_sample_figure_provenance.json。

## Slide 25: 附录｜TEAM 风格阈值指标——固定 cutoff 与时间快照

阈值先从%g转换到m/s²再取log10，随后对每个MDN分量求tail probability并按mixture weight求和。固定0.5判定计算F1；PR曲线对所有唯一概率分数分组，避免ties影响。图中normal non-input高阈值正例很少。源CSV还提供其他时刻用于透明审阅，没有用这些指标选择模型。

注意点：
Terminal: p_exceed = sum_m weight[m] * P_m(PGA >= threshold).
Raw LaTeX: p=\sum_m w_m\Pr_m(Y\geq y_{th}).

来源：rt61_threshold_metrics.csv；report_asset_provenance.json；TEAM Table S1。

## Slide 26: 附录｜RT61 参数边界——99,621 scalars，其余冻结

参数量准确来自运行时manifest，而不是手算层尺寸。额外小delta checkpoint仍依赖精确RT59 parent和共享状态hash，不能单独加载成完整模型。本报告制作没有改变任何训练或推理入口，因此不会影响现有超算训练和RT55兼容路径。

注意点：
不宣称本报告额外重跑了全部模型回归；沿用已有兼容性验证记录。

来源：trainable_manifest.json；reference_invariance.json；RT61 design doc；本报告文件变更清单。

## Slide 27: 附录｜统计、门槛与复现边界

门槛不是由本次画图重新制定，而来自预注册设计。特别是2%单站pairwise门槛不同于legacy绝对0.33门槛，两者在文件中都有独立记录。本报告不新增置信区间、不改变原有决策，只用源CSV和bootstrap结果解释边界。

注意点：
CI来自既有正式报告；没有新增bootstrap选择或test阈值优化。

来源：paired_ci.csv；mechanism_gates.csv；legacy_gates.csv；useful_joint_progress_gates.csv。

## Slide 28: 附录｜参考文献与结果来源

交付目录包含科学图的PNG、PDF和SVG，以及派生指标、原始来源hash、构建脚本和逐页讲稿。PPT的文字、卡片、表格和架构都是原生对象；科学图作为原图等比例嵌入。旧图和新图的所有关键数字均可回到原NPZ或正式CSV核验。

注意点：
本报告未提交新超算任务、未打开heldout test、未推送代码。公开文献引用固定v4/v1以避免版本漂移。

来源：report_asset_provenance.json；native_deck_manifest.json；sources.md；speech.md。
