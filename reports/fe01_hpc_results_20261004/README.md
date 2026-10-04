# FE01 超算结果分析包（2026-10-04）

本包整理用户下载到 `chaosuan_res/fe01_runs_20261002` 的真实Japan超算输出。
TEAM scratch、PhaseNet frozen、EQT frozen各3个seed（42/43/44），共9组，
每组曲线记录完整12轮。**本包的模型成绩全部属于 validation；没有最终 test 成绩。**

仓库：`rular099/team_pytorch`；分支：`exp/fe01-feature-extractor-site-effects`。
训练 runtime 记录的源码commit与此次整理基点均为
`73ae9dec50713805c878b5e4efeb8264c01d0c76`。
runtime 的 `git_dirty` 为null，不能宣称超算工作目录已由Git证明干净。
115个受审计源码/依赖文件的本地内容指纹与九组超算lock中的code SHA一致：
`3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`。
原始HDF5与预训练权重未在本轮重新载入，数据SHA等身份来自已保存的audit。

## 1. 当前验证结果

主指标：`log10(m/s²)` 坐标下的非输入目标MAE。
在normal/random两个几何协议与T=1、3、5、10、20秒组成的10个cell内，
先分别对目标求MAE，再对10个cell等权平均。它与把所有目标混在一起的pooled MAE不同。
每个run按既定规则取12轮曲线中该指标最小的一轮，同分取更早epoch。
九组对应预测文件已独立复算，与训练曲线相符（绝对容差1e-12）。
本包未取得 `best.pth`，因此“选中epoch”指曲线/预测文件选定值，尚未核对实际checkpoint内部epoch及SHA。

下表所有概率指标也先按相同10个cell等权平均，再跨3个seed平均。
MAE标准差为3个seed的样本标准差（ddof=1），不是置信区间。
NLL/CRPS来自保存的完整Gaussian-mixture评分，Brier阈值为log10 PGA=-1.2；
95%区间为混合分布的2.5%至97.5%分位数，68%区间为15.8655%至84.1345%分位数。
MAE、CRPS、区间宽度的坐标均为log10(m/s²)，Brier与覆盖率无量纲；NLL基于该坐标的密度。

| 模型 | seed 42/43/44 选中 epoch | MAE均值 ± 样本标准差 | NLL均值 | CRPS均值 | Brier均值 | 95%覆盖率 | 68%覆盖率 | 95%区间宽度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PhaseNet frozen | 12,10,12 | 0.225818 ± 0.000125 | 0.144996 | 0.161981 | 0.174182 | 91.29% | 60.69% | 0.974604 |
| EQT frozen | 10,10,10 | 0.235296 ± 0.001310 | 0.160533 | 0.167480 | 0.179936 | 92.35% | 62.63% | 1.046037 |
| TEAM scratch | 11,11,11 | 0.241817 ± 0.003011 | 0.201636 | 0.172631 | 0.185608 | 91.34% | 60.77% | 1.044227 |

已核验事实：PhaseNet在这批选中epoch的共同时间验证中，主MAE、NLL、CRPS与Brier均值最低；
EQT的95%和68%覆盖率更接近名义覆盖率。三组都存在欠覆盖。
PhaseNet区间更窄，不能仅凭点预测精度推断概率校准更好。
这些事实支持后续评估的优先顺序；它们还不能替代独立测试验证。

| 模型 | seed | 选中epoch | 主MAE | 最后epoch主MAE | Slurm job ID |
| --- | --- | --- | --- | --- | --- |
| EQT frozen | 42 | 10 | 0.236407270 | 0.236871195 | 29181344 |
| EQT frozen | 43 | 10 | 0.235630379 | 0.236561404 | 29229582 |
| EQT frozen | 44 | 10 | 0.233851242 | 0.234305494 | 29219507 |
| PhaseNet frozen | 42 | 12 | 0.225697105 | 0.225697105 | 29181346 |
| PhaseNet frozen | 43 | 10 | 0.225808450 | 0.225943037 | 29219510 |
| PhaseNet frozen | 44 | 12 | 0.225947500 | 0.225947500 | 29219511 |
| TEAM scratch | 42 | 11 | 0.245158731 | 0.245331143 | 29181345 |
| TEAM scratch | 43 | 11 | 0.240979612 | 0.241024712 | 29219508 |
| TEAM scratch | 44 | 11 | 0.239313415 | 0.239420952 | 29219509 |

来源：[run_summary.csv](run_summary.csv)、[seed_summary.csv](seed_summary.csv)，
选中CSV路径及原文件SHA在run_summary中；全部轮次见[training_curves.csv](training_curves.csv)。
`train_loss_rank0`只是rank0损失，不是跨rank平均训练损失。

## 2. 比较人口与审计

九组使用相同9084个train事件、1310个validation事件的审计后cohort；
训练目标标准化只由train的146299个允许KNET标签拟合，mean=-1.1224620633068207、std=0.431328951490345。
cohort排除规则为没有KNET支持或1秒时没有因果输入/查询，保存audit计数为616个排除事件；
不能把历史RT55原始validation事件数直接当作本轮1310事件的等价比较。

每组选中预测文件有194265个目标记录，全部supported且主要评分有限：

| 人口 | 每run目标记录数 |
| --- | --- |
| 全目标 | 194265 |
| observed_input | 54825 |
| 非输入目标（主比较人口） | 139440 |
| triggered_noninput | 43850 |
| untriggered_noninput | 95590 |

非输入目标由最后两类组成。全目标、非输入目标和各角色指标分别导出，避免输入站点遮蔽未输入目标误差。
以上计数包括多个时间/协议的同一事件、站点，不能当作独立样本数。
不同cell的可查询目标数会变化；详见逐cell表中的events/targets/decision_rows。

九组exact population指纹一致：
`e96284780cb5f72b952c1697d4d264d307e7139a153201fa222566015aec7f69`。
额外核对了标签、input_ids、站点角色、输入数量与因果决策/历史边界，亦完全一致。
公共下游结构指纹在九组一致；公共初始参数指纹在同seed的三模型之间一致。
保存的因果扰动、query独立性、sampler-only审计均通过。
这些审计是已保存超算证据，本轮没有重新执行模型forward；每run另抽32条导出的MDN参数复算评分，均相符。

来源：[audit_summary.csv](audit_summary.csv)、[shared_cohorts.csv](shared_cohorts.csv)、
[evidence_checks.json](evidence_checks.json)、[metadata/](metadata/)、[audited_source_files.json](audited_source_files.json)。
metadata为可读派生快照：绝对根路径改为环境变量占位符，完整cohort改为共享表引用；
它们不能冒充原文件字节或直接用作运行配置。原文件字节SHA见source_inventory。
capability_manifest中只在对应模型audit记录中确认该模型实际forward；其余模型的声明不代表同时验证。

## 3. 逐时间、几何协议的非输入目标MAE

下表取每个run各自选中epoch，再对3个seed均值汇总，均为validation、log10(m/s²)。
每个cell人口跨模型/seed一致；全部计数与角色细分在链接CSV中。

| 几何协议 | T/秒 | TEAM | PhaseNet | EQT |
| --- | --- | --- | --- | --- |
| normal | 1 | 0.266580 | 0.258600 | 0.261668 |
| normal | 3 | 0.224658 | 0.206446 | 0.216985 |
| normal | 5 | 0.218111 | 0.194294 | 0.210500 |
| normal | 10 | 0.210551 | 0.188632 | 0.206251 |
| normal | 20 | 0.201459 | 0.185852 | 0.200662 |
| random | 1 | 0.276421 | 0.269080 | 0.271299 |
| random | 3 | 0.252377 | 0.234476 | 0.241623 |
| random | 5 | 0.251529 | 0.232137 | 0.242906 |
| random | 10 | 0.261191 | 0.244373 | 0.251721 |
| random | 20 | 0.255294 | 0.244286 | 0.249348 |

来源：[validation_by_time_geometry_role.csv](validation_by_time_geometry_role.csv)。
该表同时含all、noninput、observed_input、triggered_noninput、untriggered_noninput，
以及RMSE、bias、R²、双向slope、NLL、CRPS、Brier、sigma和覆盖率等。
某角色在某cell没有目标时不生成该角色行；不能按缺行补零或把角色行与noninput合计再次相加。

## 4. 同seed的事件配对差异

仅使用共同非输入目标；精确匹配dataset/event/time/station/protocol键并核对标签。
每seed配对1310个事件、139440个目标记录。
调用现有 `fe01.metrics.paired_bootstrap`，以event_id为簇，所有时间、协议和站点随事件一起抽样，
5000次，bootstrap RNG seed=20261001。主差值仍在每次抽样内按10个cell等权计算。
差值为右模型减左模型，负值表示右模型MAE更低。

| 左模型 → 右模型 | seed | 右减左主MAE | 事件bootstrap 95%区间 |
| --- | --- | --- | --- |
| TEAM scratch → PhaseNet frozen | 42 | -0.019462 | [-0.022903, -0.016251] |
| TEAM scratch → PhaseNet frozen | 43 | -0.015171 | [-0.018423, -0.012141] |
| TEAM scratch → PhaseNet frozen | 44 | -0.013366 | [-0.016074, -0.010699] |
| TEAM scratch → EQT frozen | 42 | -0.008751 | [-0.011717, -0.005808] |
| TEAM scratch → EQT frozen | 43 | -0.005349 | [-0.008139, -0.002589] |
| TEAM scratch → EQT frozen | 44 | -0.005462 | [-0.007681, -0.003277] |
| EQT frozen → PhaseNet frozen | 42 | -0.010710 | [-0.012578, -0.008854] |
| EQT frozen → PhaseNet frozen | 43 | -0.009822 | [-0.011837, -0.007898] |
| EQT frozen → PhaseNet frozen | 44 | -0.007904 | [-0.010094, -0.005890] |

九个区间均为负，表明这批固定选中模型的验证差异在事件抽样下方向稳定。
这些区间**没有纳入checkpoint选择不确定性**，也不是跨seed总体置信区间；
不把验证集选择后比较写成独立test显著性结论，不把三个seed当作3930个独立事件。
来源：[validation_paired_event_bootstrap.csv](validation_paired_event_bootstrap.csv)。
包含逐事件、逐cell的误差/概率分数加总的
[validation_noninput_event_cells.csv](validation_noninput_event_cells.csv) 可用于复核cell平均和差值。
该表没有逐站点预测，也没有逐目标MDN，因此不能据它重新计算所有站点/分布分析。

## 5. 训练预算、实际时间采样和成本

九组均记录12轮，每轮212个optimizer updates，总2544 updates；global batch128、microbatch8。
每轮计划27252条（9084事件×3 draws），实际27136条，每轮按等预算丢弃116条。
每run收到192份journal（12轮×16 ranks），325632条；九组共2930688条。
全部可用日志为train/supported，保存的排他cutoff与历史/已接收样本边界检查通过。

没有checkpoint可读取 `committed_journals`，因此下表严格标作**可用journal的曝光统计**。
每epoch/rank只有一份日志，且逐轮数量与曲线一致，尚不能宣称已由checkpoint确认提交归属。
区间左闭右开，90秒并入最后一段；表中为3个seed各run比例的均值。

| 模型 | 1–3s | 3–5s | 5–10s | 10–20s | 20–40s | 40–90s | 实际范围 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EQT frozen | 21.54% | 21.46% | 21.50% | 16.07% | 16.21% | 3.21% | 1–54.99s |
| PhaseNet frozen | 25.45% | 25.35% | 25.37% | 19.05% | 4.78% | 0.00% | 1–25s |
| TEAM scratch | 20.02% | 19.95% | 20.04% | 14.95% | 14.99% | 10.04% | 1–90s |

能力条件化会改变早期区间采样概率。PhaseNet与EQT不应被说成实际接受了TEAM的完整1–90秒训练分布。
共同时间验证可以比较当前协议结果，但不能单独识别“只换encoder”或“预训练本身”的因果贡献。
40秒、90秒或随机验证时刻的实际预测结果未在本批提供；audit的支持计划CSV和随机时刻manifest不能代替成绩。

| 模型 | 12轮计时合计均值/小时 | 跨seed样本标准差/小时 | 训练参数 | encoder参数 |
| --- | --- | --- | --- | --- |
| EQT frozen | 10.699 | 0.034 | 42438214 | 376935 |
| PhaseNet frozen | 5.895 | 0.120 | 42550214 | 268443 |
| TEAM scratch | 6.122 | 0.130 | 48307702 | 5886488 |

成本计时来自rank0各epoch的elapsed_seconds，包含训练、等待、validation及CSV写出，
不包含排队、完整启动/审计/加载与后续checkpoint写出，不能写成纯训练速度或完整作业墙钟时间。
16个rank由日志核实；4节点×4卡部署来自此前提交方案，本批未提供正式sacct/Slurm分配证据。
不同job在不同节点/时间运行，成本仅作当前配置的描述性记录。
来源：[journal_summary.json](journal_summary.json)、[training_time_exposure.csv](training_time_exposure.csv)、
[training_curves.csv](training_curves.csv)。
三个pilot各1轮、3 updates、48 samples，见[pilot_summary.csv](pilot_summary.csv)，不混入正式榜单。

## 6. RT55/RT61 与未提供证据

用户选择快速推进并复用现有RT55/RT61，因此本批没有重新训练DiTing公共下游组。
已有RT55完整系统可作为后续独立参考，不要求因这份分析包重新训练它；
需要用明确的原checkpoint及匹配配置，在共同事件、时间、目标协议上复评，才能填入可比列。
本批没有RT55同协议复评，历史标量成绩不混入这份排行榜。
RT61是含额外模块的完整系统参考，不能替代严格同公共下游的DiTing frozen extractor对照。
原V2四个主组各3 seed的完整控制实验尚未由当前九组覆盖。

| 证据 | 本批状态 |
| --- | --- |
| 9组正式训练曲线和每轮validation预测 | 已提供；12轮记录完整 |
| 独立test / 最终评估provenance | NOT_PROVIDED |
| best/last/init checkpoint及SHA/内部epoch | NOT_PROVIDED |
| RT55/RT61在共同协议上的复评 | NOT_PROVIDED |
| 40/90秒、非标准随机时刻最终评价 | NOT_PROVIDED；只有计划manifest/支持表 |
| 场地残差、spatial holdout、格点、真实回放与延迟评价 | NOT_PROVIDED |
| 正式训练的Slurm完成/退出码记录 | NOT_PROVIDED；仅旧pilot audit的4个日志文件 |
| 原始offline滤波/重采样的在线因果性 | 未认证 |

本轮只整理已有文件，没有提交超算任务、重训、开启test或改变任何模型/训练配置。

## 7. 文件与复现

原始2201个文件、14193566080 bytes（约14.19GB，十进制）保留在本地原目录。
Git同步的是分析表、可读证据快照、复现脚本和来源索引；大预测CSV、journal、HDF5、权重和旧Slurm日志不放进Git。
[source_inventory.csv](source_inventory.csv) 给出相对原目录的路径、字节数与SHA256：
2199个非cache文件已读字节计算SHA，2个派生cache只记录体积并明确标为未hash。
选中预测文件的SHA也单列于run_summary；metadata输入身份与结果报告身份分开。

在**本地**FE01仓库根目录，用具备pandas/numpy/scipy以及匹配dtbench源码的环境执行：

```bash
.venv-fe01/bin/python scripts/fe01/summarize_hpc_results.py \
  --source ../chaosuan_res/fe01_runs_20261002 \
  --output reports/fe01_hpc_results_reproduced \
  --journal-workers 3 --bootstrap-draws 5000
```

这是读取已有CSV/JSON/JSONL的本地整理命令，不需要GPU、不读取原始HDF5、不提交sbatch。
输出目录须不存在，原始结果只读。该命令重建数值汇总/证据文件；当前中文README、
ChatGPT提示和交接记录是人工整理文档，不由脚本生成。
首次处理会读取约14GB并作事件bootstrap；运行环境版本记录于evidence_checks。
[SOURCE_IDENTITY.json](SOURCE_IDENTITY.json) 固定源码/reader/原始上传的身份，
[artifact_manifest.sha256](artifact_manifest.sha256) 校验本分析包文件（不含清单自身）。

## 8. 交给ChatGPT

先提供本README和[CHATGPT_ANALYSIS_PROMPT.md](CHATGPT_ANALYSIS_PROMPT.md)，
按需要加载小型CSV和metadata；不要要求一次逐行阅读所有日志或22MB事件表。
若连接器看不到当前实验分支，应使用这次同步完成后给出的**精确commit链接**。
请它先回报实际读取的repo/branch/commit及split，再分析模型精度、校准、时间/几何差异和下一步评价优先级。
当前结果不支持“RT55已被超过”“场地效应已改善”或“最终test结论”。
