# 补充材料

## S1 数据支持与实际实现

所有证据属于 validation，种子 42；selected 检查点轮次为 DiTing12、PhaseNet12、EQT10、TEAM11。原始数据引用 NIED K-NET, KiK-net 数据 DOI 10.17598/NIED.0004，本研究筛选 K-NET。训练/验证审计后人口为 9084/1310 事件，训练标签标准化仅使用 146299 个允许训练标签。事件划分与数据分片来源保存在 evidence/data_audit_excerpt.json 和 input_manifest.json；原始私有路径不写入论文源文件。

原生窗分别为 DiTing/TEAM 10000、PhaseNet3001、EQT6000 样本。以五秒前缀加当前样本计，理论可容纳的首 P 后最大时间分别为 94.99、25.00 和 54.99 秒。这是窗能力，不能解释为已测量的长时性能。共同旧协议只评价 1/3/5/10/20 秒；训练时间采样按各自能力条件化，因此训练时间暴露不完全相同。T1 保留样本长度、训练预算与参数来源。

DiTing完整模型的十二轮预算逐行见 [S1_DiTing_training_budget.csv](assets/source_data/S1_DiTing_training_budget.csv)，仅提取epoch、updates、批量和实际/计划样本数；没有引入其他实验条件。原表固定提交与字节摘要保存在input manifest。

有效样本先掩码去均值，再以三分量联合峰值归一化。十个物理尺度量和一个时长量形成十一维分支，并与波形、绝对坐标融合。冻结骨干、适配器、台站 memory、四层 target cross-attention、三分量 MDN 构成 PGA 实际前向链。历史协议中的通用 Transformer 描述不能证明当前配置执行了台站自注意力。逐函数来源、分支与检查点配置见 [executed_model_path.md](executed_model_path.md)。

为明确概率量的定义，先给出终端可读形式。这里所有 μ、σ、y 都在 log10(m/s²) 坐标，π 非负且和为一；预测点值是该坐标的分布均值，不是在线性 PGA 坐标求均值后取对数。

```text
            3
p(y|x,q) =  Σ  πⱼ N(y ; μⱼ, σⱼ²)
           j=1

          3                         3
E[y] =    Σ πⱼ μⱼ       Var[y] =    Σ πⱼ(σⱼ² + μⱼ²) − E[y]²
         j=1                       j=1
```

原始 LaTeX 如下。

```latex
p(y\mid x,q)=\sum_{j=1}^{3}\pi_j\mathcal{N}(y;\mu_j,\sigma_j^2),
\quad \mathbb{E}[y]=\sum_{j=1}^{3}\pi_j\mu_j,
\quad \operatorname{Var}(y)=\sum_{j=1}^{3}\pi_j(\sigma_j^2+\mu_j^2)-\mathbb{E}[y]^2.
```

代码读取归档的 pga 标签；对与该标签一致的瞬时幅度序列作越阈时刻判断。本文不擅自把 PGA 定义扩写成未经本轮上游审计确认的分量组合方式。上游完整因果性、DiTing 精确语料重合和许可见 GAPS。

## S2 全部分层与空间支持

完整分层效果、支持和自助法支持分别为 [S2_E05.csv](assets/source_data/S2_E05.csv)、[S2_E06.csv](assets/source_data/S2_E06.csv)、[S2_E07.csv](assets/source_data/S2_E07.csv)，包含 selected 与 epoch12。震级、最终 log-PGA、震中距、深度、最近输入距离、输入数、缺失比例、SNR 代理、起始时间、观测时长、输入跨度、凸包状态及方位缺口均保留原有层和 Unknown。每行保留事件数、请求数、有支持单元、缺失单元、有效重复数与描述性标记，不能把重复请求当作独立地震。

原有交互分析见 [S2_interactions.csv](assets/source_data/S2_interactions.csv) 与 [S2_interaction_contrasts.csv](assets/source_data/S2_interaction_contrasts.csv)，支持见 [S2_interaction_support.csv](assets/source_data/S2_interaction_support.csv)。这些是原先执行的探索性对照，未在本轮新增阈值、交互或显著性筛选。输入数量变化时同一事件可出现在多个层，分层事件数不应相加。

空间完整效果见 [S2_spatial_full.csv](assets/source_data/S2_spatial_full.csv)。level/shape/field 为 dex²，range/pair 为 dex，各层只按相应合格场计算。K1 的空间层可能只有少数有支持单元且有效重复数偏低，必须与空单元比例同时解释。正文不使用其点估计作为主要论据。全目标对 PhaseNet 的不利 shape 和 pair 结果、未触发非输入层及强 PGA 层均保留，不因叙事方向删除。

## S3 全部概率指标

全部已保存绝对指标和配对效果见 [S3_E03.csv](assets/source_data/S3_E03.csv) 与 [S3_E04.csv](assets/source_data/S3_E04.csv)，包含主端点、首秒、早期以及不同目标角色和检查点政策。NLL、CRPS 和 Brier 不共用单位；coverage 是比例，width/sigma 是 dex。pooled 的 R²、bias、预测对真值斜率与其相反方向斜率分别保留，不得互换。

既有附加 68% 区间宽度端点另见 [S3_width68.csv](assets/source_data/S3_width68.csv) 和 [S3_width68_effects.csv](assets/source_data/S3_width68_effects.csv)。它们是已有输出，不能说成本轮新发明或独立验证的端点。区间窄和覆盖不足可能同时出现，必须联同 NLL、CRPS、Brier、bias 和斜率解释。

新固定查询回放的 PIT 与可靠性分箱状态为 **BLOCKED**。本地授权结果目录搜索记录见 evidence/local_export_search.json：未找到覆盖该固定查询人口的逐目标完整 MDN 导出。已有六事件试验或旧五时点结果均不能替代。恢复此诊断需要原始事件/目标身份、每时点混合权重/均值/尺度、支持掩码、最终标签和时钟元数据；本轮没有提交导出任务。

## S4 检查点与事件稳健性

selected 与固定 epoch12 的全部指标/配对表仍在 S3 中，逐事件剔除、事件胜率和贡献集中度见 [S4_E09.csv](assets/source_data/S4_E09.csv)。它们用于固定训练结果的敏感性检查，不是新随机种子。DiTing 对 PhaseNet 主端点逐事件剔除范围为 −0.003139 至 −0.002470 dex，事件胜率为 0.548092；人群为 1310 个验证事件的动态非输入十单元。

另列历史参照实验 [S4_run_summary.csv](assets/source_data/S4_run_summary.csv) 及 [S4_audit_summary.csv](assets/source_data/S4_audit_summary.csv)，来源为论文基点所保存的 reports/fe01_hpc_results_20261004。历史 TEAM/PhaseNet/EQT 各有种子 42/43/44，其选轮、请求数、训练更新数和来源身份均随表保留。这一历史表没有 DiTing 重复训练，不能把九组参照记录合并成当前四模型比较的重复数，也不能据此计算 DiTing 跨种子置信区间。

## S5 场质量与全部回放条件

所有模型、normal/random、三个固定 panel、预算 0.15/0.20/0.25 dex、station/field gate、已有时钟和 5/10/20 秒报告点见 [S5_E10.csv](assets/source_data/S5_E10.csv)。station gate 要求 MAE≤b、|bias|≤b/2，且至少 90% 目标绝对误差≤2b；field gate 再要求去均值 RMSE≤b、|预测 range−真值 range|≤b。缺失或重复目标、无支持值和不适格事件分别保留，不能仅从已成功案例计算比例。

预算 0.20 的原有配对确认比例和双方确认条件时间差见 [S5_E11.csv](assets/source_data/S5_E11.csv)，全部原 HPC 预算配对结果见 [S5_original_paired.csv](assets/source_data/S5_original_paired.csv)。random 只有理想时钟，任何计算或渲染组合均不得补写。数值与渲染的交付时刻以首 P 为零点；回顾性三帧起点不是第三帧确认时刻。

首次通过后的变化和确认后已知不通过计数见 [S5_E13.csv](assets/source_data/S5_E13.csv)。不同模型的条件确认集合不同，退化比例只作描述。期限前确认数与 unknown 产生的 lower/upper 是信息界，原有 bootstrap CI 单列，二者不能拼成一条区间。

## S6 失败与误报全网格

渲染场景的全部两 PGA 阈值、五告警概率及四提前量见 [S6_E17.csv](assets/source_data/S6_E17.csv)；同时保留事件等权和 affected 目标合并的失败比例、完整负例的 FPR、不完整负例、已知误报、precision 与 20 秒 Brier。原有理想/数值/渲染事件配对差值界见 [S6_E19.csv](assets/source_data/S6_E19.csv)，已重算渲染实例见 [S6_E18.csv](assets/source_data/S6_E18.csv)。未知不能当成零失败，不完整负例不能默认为完整阴性。

失败下界分子为已知失败，上界再加入未知，两者分母均为 affected 目标；timely+failures+unknown 必须等于 affected。配对下界为 DiTing 下界减对照上界，上界为 DiTing 上界减对照下界，使用事件等权；与合并请求后再作差不同。tau=0 时仍要求提前量严格为正。本文不进行阈值调优、不提供连续地区的盲区面积，也不把目录事件误报率换算为每天次数。

## S7 原始预选案例

[全部十二例索引](assets/cases/CASE_INDEX.md) 保留每例 5 秒和 20 秒原始关键帧，共 24 张，复制前后字节 SHA256 一致；[case manifest](assets/cases/manifest.json) 给出结果提交与路径。每个案例目录的 frame_sources.json 保留数据、事件、几何和原图来源。它们由既有元数据与哈希预选，未按本轮误差挑选，没有遗漏不利案例。

案例图保留原图例、同帧比较、未支持网格及回顾性真值标记。其原始像素不被伪造放大为新的高分辨率观测，连续网格没有独立真值。完整案例浏览页见 preview_cases.html。正文 7 组新绘图提供可编辑 SVG 和 PDF；这些历史案例为原始 PNG，性质与新绘矢量图不同。

## S8 可复现性与检查

输入 manifest 固定到各文件的完整提交 SHA、Git blob 和实际字节 SHA256，图表 manifest 记录过滤、人群、单位、权重、不确定性、脚本摘要、source_data 与产物摘要。scripts/audit_inputs.py 审计输入包、完整模型设置和只读范围；scripts/build_assets.py 在单线程 CPU 上生成数值素材与图表；scripts/compose.py 生成表格及预览；tests 仅核验来源与人口边界，不加载科学模型。

执行记录及数值核对见 qa 目录。脚本重新聚合保存的非输入事件单元，复算四模型主端点 MAE/RMSE，检查 TTUM 资格守恒、失败结局守恒和负例误报率，未重新推理或自助抽样。原始 216 个 TTUM 区间的历史复现属于输入包上一次审阅记录，本轮不冒称再次重算它们。

科学工作树、配置和全局会话文件均不修改，论文单独分支和基点写入 HANDOFF。精确未决项见 [GAPS.md](GAPS.md)。本轮模型运行、GPU、训练、超算和测试集访问均未执行；局部 CPU 数字核对不能代替超算实验成功证据。
