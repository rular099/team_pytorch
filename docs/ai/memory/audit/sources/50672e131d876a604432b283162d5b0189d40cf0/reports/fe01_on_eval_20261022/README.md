# FE01 ON 综合评价 V4：实际CPU证据与手动超算交付

仓库 `rular099/team_pytorch`，新分支 `exp/fe01-on-warning-map-evaluation`，基点
`29ffb20ba0effc0c345a0329ba3f8c658dc8cef0`。实施/验证日期2026-10-22；任务ID沿用20261021。

实际CPU分析、源码、运行手册与小型合成动态图均已实现。**新正式超算任务NOT_SUBMITTED，生产模型smoke/逐秒/grid/时延NOT_RUN。**
查看[超算运行手册](../../docs/ai/FE01_ON_EVAL_RUNBOOK.md)、[科学协议](../../docs/ai/FE01_ON_EVAL_PROTOCOL.md)、
[产物字典](../../docs/ai/FE01_ON_EVAL_ARTIFACTS.md)及[ChatGPT复核任务](CHATGPT_REVIEW_PROMPT.md)。

## 得到支持的差异

真实来源为此前已核验的 `outputs/review.tar.gz`，SHA
`7a6ecf2260f1fd8d7f1d188c48d6b0f19bbdc433bb861879fb51ac549f209986`。
本地只读已解出的四ON selected/epoch12预测，没有读取生产权重、Japan波形或test。
每份旧导出1310事件/194265请求；noninput139440。ID/标签/输入/整数时钟严格配对，C解析器round_trip读取。

下表为selected、normal/random×1/3/5/10/20十cell等权noninput MAE，单位dex=log10(m/s²)。
差值DiTing减对照；2000次dataset/event簇配对、seed20261007，固定模型事件区间。

| 对照 | DiTing−对照 | 95%事件区间 |
|---|---:|---:|
| PhaseNet ep12 | −0.002838 | [−0.005120, −0.000580] |
| EQT ep10 | −0.013548 | [−0.016358, −0.010949] |
| TEAM ep11 | −0.022299 | [−0.026050, −0.018831] |

DiTing固定ep12。第一秒两cell相对PhaseNet为−0.009990 [−0.015231,−0.004792]，1310事件/38795请求；
早期1/3/5秒六cell为−0.003528 [−0.006291,−0.000718]，1310事件/98889请求。
来源：[paired_effects.csv](existing/paired_effects.csv)。epoch12检查单列，方向保持；不是新的训练重复。

相对PhaseNet，noninput逐事件可用cell平均loss的胜率54.81%，事件macro平均差−0.001303；
它们与原目标/cell主差−0.002838是不同权重。LOO主差范围[−0.003139,−0.002470]，
绝对贡献最大的前10事件合计11.03%。来源：[event_robustness.csv](existing/event_robustness.csv)、[event_effects.csv.gz](existing/event_effects.csv.gz)。

这些结果支持本批固定方案的点误差差异，不能消除单seed、validation选轮、原生窗口/训练时间曝光/预训练等混杂。
主要图：[配对效应](figures/real/paired_primary_early.png)、[逐cell有符号贡献](figures/real/first_second_signed_contributions.png)。

## 不占优或不确定的条件

相对PhaseNet的selected结果：

- all-target主MAE差+0.000842，CI[−0.001105,+0.002788]，1310事件/194265请求；点排序与noninput相反。
- K1_noninput差+0.002285，CI[−0.008184,+0.023314]，660事件/28535请求；不是所有K条件都占优。
- untriggered_noninput差+0.001629，CI[−0.004597,+0.008876]，1292事件/95590请求。
- 两geometry等权5秒差+0.002332，CI[−0.001194,+0.005746]；1秒之外多时刻区间跨零。

完整不利/有利及Unknown分层都在[全部分层效应](existing/stratified_effects.csv)和[支持量](existing/strata_support.csv)，不筛选显著组。
PhaseNet对照第一秒两个cell占总主差约70.41%的**有符号代数贡献**；不是MAE下降70%，不是因果归因。
K1稀疏重采样有效数/空cell比例见[bootstrap支持](existing/bootstrap_support.csv)，不得只读CI是否过零。

noninput场指标以>=5目标的field等权，944事件/6789fields；等距离指标928事件/6644fields。
相对PhaseNet，shape MSE差+0.000161 [−0.000552,+0.000895]；field MSE差−0.001075 [−0.002477,+0.000352]；
等距离误差差+0.000171 [−0.002174,+0.002674]，均不支持普遍空间结构优势。
range误差差−0.012130 [−0.016226,−0.007759]。来源：[空间配对表](existing/spatial_paired_effects.csv)。
场level/shape单位dex²，range/等距离误差为dex；不能与目标等权MAE混排。

## 概率、分层和交互的范围

[全指标表](existing/cross_encoder_metrics.csv)同时保留cell等权/pooled、MAE/RMSE/bias、NLL/CRPS/Brier、覆盖率、sigma、P90/P95、R²/slope。
selected noninput四组95%覆盖率约91.17%–92.23%，均低于名义值；DiTing更低的proper scores不等于各条件完全校准。
68%真实混合分位宽度及配对区间在[补表](additional_endpoints/width68_metrics.csv)、[区间](additional_endpoints/width68_paired_effects.csv)。
没有调sigma，NLL按原raw-logit精确评分保存值，Brier沿用−1.2阈值。

四类交互既有完整条件cell表，也有[共同支持域标准化的difference-of-differences](interaction_contrasts/interaction_contrasts.csv)。
固定参考水平和权重见[交互合同](interaction_contrasts/contract.json)，先冻结支持再看效应。
SNR、真实mask缺测、最差站P后时长、UTC/origin缺本地原始数据，明确Unknown/不可识别；需要HPC CPU sidecar。
不能凭这些表声称低SNR优势、实际触发提前、站点未见泛化或连续运行假警报率。

## 新回放/地图/TTUM/shadow的当前状态

新请求是fixed_query_replay_v4，固定真实query即使变成input也保留；与legacy_snapshot_v1分人口/hash。
实现四ON逐秒1–20、normal/random station回放、全量normal网格计划、random独立入口、origin及40/90独立stage。
网格第三坐标由生产代码追踪为km高程，只在台站凸包内显式估计，缺失点NA；没有连续网格真值。

TTUM完整b=.15/.20/.25研究预算、固定all/never-input面板，首次/稳定起点/确认分离；未达标/缺帧/删失保留。
shadow完整2个PGA阈值×5个概率阈值×4个lead门，伴随误报，未知给信息界；面积没有独立支撑则NA。
两者共用FIFO交付时钟，计算超过1s保留积压，通信UNKNOWN。工程标准未提供，不选择有利阈值命名工程合格。

实际可浏览的[20帧动态图/动画/索引](synthetic/map_index.html)与[shadow台站分类](synthetic/synthetic_event/shadow_station_map.html)
**均为SYNTHETIC、不是Japan新前向成绩**。全部raw合成数值、输出seal与源码身份一起保留，证明端到端实际执行。
不能从已有MAE优势推出shadow更小、TTUM更早，这些须待用户手动完成正式补算。

## 执行来源、失败记录与复算

完整真实CPU重跑为real_comparison_03。前两次退出137（SIGKILL），没有冒充成功；
改为读取必要列/共享重复字符串/流式写event与field后，相同人口/分桶/2000draws完整通过。
记录见[actual_runs.json](verification/actual_runs.json)。大逐field表本地只读保留，路径/大小/SHA见[索引](verification/large_local_outputs.json)。

`existing/source_identity.json`保留CPU开跑时的独立实现树指纹。随后只继续完善运营/回放/交付代码；
[executed_cpu_sources](executed_cpu_sources/)的五个实际统计reader文件与最终实现相同，逐一核SHA。
`existing/SOURCE_SEAL.json`验证原完整CPU目录；Git轻量副本省略大field文件，以本报告manifest为准，不能当原目录完整副本。
原115中50个本地源文件、46个A01核心不变；65个dtbench依赖本地缺失，运行时仍需HPC原vendor及完整门。
真实设备加载/完整MDN等价/未来与缓存的模型级门尚未运行，不把CPU窗口/合成测试当它们的PASS。

可在有numpy/pandas/scipy/matplotlib/Pillow的CPU环境，使用**新目录**复算；不要覆盖原产物：

```bash
python -m fe01_a01_compare.cli compare-existing-local --source /path/to/extracted_review --output /new/main
python -m fe01_a01_compare.cli interaction-existing-local --source /path/to/extracted_review --output /new/interactions
python -m fe01_a01_compare.cli synthetic-e2e --output /new/synthetic
```

精确部署release/result commit及SHA在最终交付消息和源码包的SOURCE_RELEASE.json。
HPC提交命令全部使用[运行手册](../../docs/ai/FE01_ON_EVAL_RUNBOOK.md)的脚本；本轮没有自动sbatch/srun、重训、OFF扩展或test。
