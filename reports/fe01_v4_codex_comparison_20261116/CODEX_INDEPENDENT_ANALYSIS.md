# Codex 独立分析：FE01 ON V4（2026-11-15，供独立分析后对照）

本报告在取得ChatGPT结论之前写成，不放进中立GitHub结果目录或ChatGPT ZIP。
仓库rular099/team_pytorch；整理基线99d9b1c4bb5c47fc26125df7eea2648917e90462；
本地分支review/fe01-v4-results-20261115，同步exp/fe01-on-warning-map-evaluation。
数据为runs/v4pack/fe01_on_eval_v4_review.tar.gz，SHA256
7a5c19ad82d45c8c3a40a00a00b29856f4fd8720634fdbdf371e350f4bc32add。
以下路径均相对reports/fe01_v4_hpc_results_20261115。

我的总体判断：这批结果支持DiTing在旧协议的非输入平均误差上有优势，尤其早期与多输入条件；
它没有证明DiTing普遍更早交付可用地图或拥有更小预警失败区域。新固定未输入panel的理想质量收益、
实测计算代价、渲染工作流代价必须分开。PhaseNet在这批运行中是重要的准确率—计算代价参照。
这不是超算执行失败的结论；完整必需scope已完成，关键支持记录和区间在本地复现通过。

**先确认证据。** 输入SHA、gzip和1016条manifest全部通过；12案例453文件与原索引字节相符；49个V4主程序源文件匹配pin。
本地复核864条TTUM支持记录、216个已保存区间、6个旧主MAE区间、104800个FIFO帧、8组latency汇总通过。
四项latency及41项render在超算快照中完整通过；pack自身RUNNING是包内的过程快照，之后日志记录exit0。
完整权重、逐台站新预测和全部79009个原产物实体未下载；review JSON有脱敏，不假装本地重新验证了原私有seal的所有字节。
来源：verification.json、local_numerical_checks.json、metadata/run_identity.json。

**旧预测主端点：优势存在，幅度与人口有关。** 这里是legacy_snapshot_v1、validation/seed42、ON，
normal/random × 1/3/5/10/20秒十cell等权；DiTing12/PhaseNet12/EQT10/TEAM11固定selected。
每模型1310事件、139440非输入请求，MAE坐标为log10(m/s²)。DiTing为0.222859，PhaseNet0.225697，
EQT0.236407，TEAM0.245159。DiTing减对照的MAE及固定模型事件cluster 95%区间为：

| 对照 | 差值dex | 95%区间 |
| --- | --- | --- |
| PhaseNet | −0.002838 | [−0.005120, −0.000580] |
| EQT | −0.013548 | [−0.016358, −0.010949] |
| TEAM | −0.022299 | [−0.026050, −0.018831] |

区间2000次、seed20261007，不含训练seed和选轮不确定性，探索性且未校正多重比较。
固定epoch12保留相同方向，但不是独立训练重复。全目标194265请求时DiTing−PhaseNet为+0.000842
[−0.001105,+0.002788]，不能说全目标也有可靠优势。
同一legacy非输入人口的第一秒为1310事件/38795请求，差−0.009990 [−0.015231,−0.004792]；
K1为660事件/28535请求，差+0.002285 [−0.008184,+0.023314]，1999个有效重采样；
Kmulti为1219事件/110905请求，差−0.003491 [−0.006011,−0.000854]，2000次有效。
因此我更相信条件性、早期/多输入的收益，而不支持“所有输入规模均改善”。未触发非输入主端点差+0.001629，区间跨零。
来源：derived/legacy_primary_pairwise_recheck.csv；source_tables/analyze/0/legacy_with_physical_covariates/paired_effects.csv，
按checkpoint_policy=selected、metric=mae及上述target_group/endpoint定位。

**概率预测不能只看点误差。** 同一旧非输入主人口，DiTing的equal-cell RMSE0.296500、NLL0.115770、
Brier0.169824、CRPS0.159220，与较小MAE方向一致；但68%覆盖0.614525、95%覆盖0.914884仍低于名义值。
PhaseNet的95%覆盖0.913038，EQT为0.922287；EQT覆盖较高并不自动抵消其较大的MAE/NLL等。
DiTing equal-cell sigma0.248804，pooled R2约0.432956、prediction-on-truth slope约0.480508。
后两项是pooled权重，不能与equal-cell指标混作同一统计量。平均bias近零也不能证明强/弱震条件无偏。
这些记录提示压缩和欠覆盖仍需关注；没有支持“表征/空间退化已经解决”的机制结论。
来源：同目录cross_encoder_metrics.csv，selected/noninput/primary_10cell；Brier阈值−1.2 log10(m/s²)。

**新固定未输入TTUM：收益会随交付时钟改变。** 这是新fixed_query_replay_v4，同一1310事件、1–20秒逐帧、
固定panel，selected epoch不变。以下统一取预定预算0.20 dex、field gate、首P后20秒；不是调优得到的预算。
normal fixed_never_input只有207个至少5目标的适格事件，另外1103个不适格，不能向全1310事件外推。

| 时钟 | DiTing确认/207 | PhaseNet确认/207 | 确认比例差D−P | 配对95%区间 |
| --- | --- | --- | --- | --- |
| 理想cutoff | 88 | 73 | +7.246百分点 | [+1.449,+13.527]百分点 |
| 实测数值FIFO | 49 | 69 | −9.662百分点 | [−15.942,−2.899]百分点 |
| 共享渲染FIFO场景 | 15 | 16 | −0.483百分点 | [−4.831,+3.382]百分点 |

两个模型Unknown分别为6/5，未作为成功；区间对全部207适格事件抽样，不只抽成功事件。
理想/数值两个既有区间已复现，渲染区间为本轮从完整事件记录新增。数值场景双方可确定的59个事件中，
DiTing确认交付平均晚10.757秒，区间[7.594,13.839]秒；这是条件时间差，不能说全部207事件平均都晚这么多。
random fixed_never_input适格539事件时，理想确认为114/539对106/539，差+1.484百分点
[−1.299,+4.267]百分点；没有random实测整图时钟，不能补出random部署优势。
normal全目标1057个适格事件在同一field/0.20/20秒条件下，理想达标DiTing676、PhaseNet804，
数值为455对783，渲染为78对125。all与never-input的排序和资格人口均不同。
来源：derived/ttum_support_recomputed.csv、ttum_paired_b020_recomputed.csv，gate=field_pass；
必须保留all/never-input两套结果，也不能把旧139440非输入请求叫作这里的固定未输入人口。

**三帧确认是一个时点，不保证后续一直稳定。** 上述normal never-input、field/0.20下，
DiTing88个确认事件中34个后来又出现已知不通过帧；PhaseNet为73个中的47个。
这是last_nonpassing_frame晚于stable_confirmation的描述性计数，不改写原预定TTUM端点，也不是工程稳态认证。
它说明“曾有三帧连续通过”与“持续稳定”要在论文表述中分开。
来源：derived/post_first_pass_degradation.csv，scene=ideal_cutoff_replay；定义见method_sources/map_readiness.py。

**计算与绘图成本。** 四模型同步warm benchmark使用共同6个元数据预选事件、normal/random×1/10/20秒36条件，
每条件2 warmup、3 measured，每scope108记录；K为1–25、station Q为3–80、有效grid Q为1–281。
whole-map P50/P95秒为DiTing0.585/4.501、PhaseNet0.235/1.278、EQT0.292/1.676、TEAM0.271/1.496。
这些是完整数值帧耗时而非query分摊；不是108个独立事件的泛化证据。记录同为Device 66a1，不证明运行资源完全匹配。
DiTing cold_load101.279秒与其他模型22–44秒分开记录，包含认证/加载；不把它当每帧延迟或包含全部进程启动。

完整normal场景为1310事件×4模型×20帧，104800帧；共享四模型CPU比较图render+publish P50约3.233秒/帧，
为每模型场景统一计入。第20帧最终交付时钟的事件P50为DiTing95.746秒、PhaseNet71.691秒，均相对首P。
这是包含排队的20秒数据帧交付，不是TTUM，也不是特制单模型线上渲染的耗时。
我据此认为当前比较图工作流对实时展示有明显阻塞，但不据此宣布某架构在任何硬件/并行方式下不可实时。
来源：derived/latency_recomputed.csv、queue_quantiles.csv、metadata场景合同与method_sources/delivery_analysis.py。

**shadow没有给出统一的DiTing优势。** 以合同内的一个例子说明：normal、rendered场景、PGA阈值−1.2、
p_alarm=0.5、tau=3秒；完整阈值网格保留，不用此例选择阈值。
never-input有312个非空panel事件，214个有受影响目标，共2336目标；这与TTUM的207事件分母不同。
DiTing目标加权失败界为[30.69%,32.79%]，PhaseNet[30.31%,32.45%]，识别界重叠。
事件等权差的识别下/上界点估计为−0.055174/+0.042994；各自95%区间为
[−0.097193,−0.015182]及[+0.002843,+0.082499]，不能把下界为负单独当作DiTing更小shadow的证明。
完整负例2507目标上的误报为DiTing1210/2507、PhaseNet1225/2507；这是目录事件内空间误报，不能换算每日误报数。

全目标则有1193个受影响事件、11111目标，DiTing目标加权失败界[53.51%,54.25%]，
PhaseNet[51.49%,52.21%]。对应事件等权差的上下界均为正且各自区间不跨零；在此设置下支持DiTing失败比例更高。
完整负例11152个上的误报为4891/11152对4781/11152。必须同时看误报代价和完整网格，不推广为所有阈值下的结论。
来源：derived/shadow_rendered_summary.csv、shadow_rendered_example_pairwise.csv；ideal/numeric界见原shadow_paired_summary.csv。
unknown/不完整观测、事件等权与目标加权均分别保留。面积一直NA，没有证据估计连续地面预警盲区面积。

**分层和图的限制。** 物理SNR代理的Unknown有50774/139440条legacy非输入请求，涉及1035个事件；
低SNR [0,10)仅24事件/518请求。因而不能把有限分层效果当成已识别的预训练机制。
已查看一个元数据预选案例的20秒图，标题K=Q=5，全部查询为输入台站；该例不能单独证明未输入位置泛化。
灰色缺高程网格、站点支撑边界和无连续真值都保持显式。12例按元数据/hash选，不按误差选。
来源：legacy目录strata_support.csv；cases/32f8668a90b45f9ada23/keyframe_20.png及对应frame_sources.json。

**下一步只作为提议。** 第一优先是从既有保存预测完成所需固定query点精度/概率校准的全人口导出；
当前包不足以复算新逐台站MDN，不应把旧主MAE填进新回放协议。
如继续研究时效，先固定同一硬件、同一K/Q和cutoff、同一冷/热状态，用相同已保存数值对照数值交付与简化/解耦渲染，
将模型计算代价和比较图代价隔离；仍不重选checkpoint或放宽门。
机制归因需要另行预注册的小型对照，而不是根据本批单seed混合训练历史宣布预训练encoder优越。
这些均未在本轮执行；没有申请追加训练、修改阈值或访问test。

对照时请特别检查：不同人口被叫作同一个noninput、条件时间差被当全部事件差、Unknown上下界被当95%CI、
三帧确认被当永久稳定、共享绘图场景被当部署时延、目录误报被当每日误报、以及单seed效果被当普遍机制。
若ChatGPT在严格相同定义和数据上得出不同数值，应先按源CSV复算，再修改解释；不以观点投票决定结果。
