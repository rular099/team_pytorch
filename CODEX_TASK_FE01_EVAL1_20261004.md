# FE01-EVAL1：复用现有权重的同人口复评与空间诊断

日期：2026-10-04。作者角色：ChatGPT 独立研究审阅与方案设计。
本文件替代上一轮 `FE01_REVIEW_HANDOFF_b2756825_20261004.md` 的“下一步工作”部分；不更改已经保存的训练协议、checkpoint 选择或历史结果。

## 0. 本轮已经完成的研究决策与分工

**唯一交付目标**：不重训、不开 test，通过现有预测的再分析、旧权重身份核验及必要的新增 validation 前向，回答“PhaseNet/EQT/TEAM 与现有 DiTing 完整系统相比如何；优势是否覆盖极早期、非输入/未触发位置和事件内空间差异”。这不是新的 encoder 训练实验。

ChatGPT 已完成：
- 核对远端分支 HEAD 与上一轮审阅相同；重读上一轮任务书及本次相关实际源码。
- 用已读取的 run_summary 数值复算选中 epoch 与固定 epoch12 的家族平均 MAE，排序不变。
- 确定保留 PhaseNet 为主要候选、EQT 为极早期概率对照、TEAM 为从头训练参考，但本轮不据此改训练。
- 固定下列矩阵、主/次端点、因果与兼容边界、分析方法和停止条件。
- 明确 `fe01.engine.code_identity()` 的路径范围：本轮新增工具必须位于范围外，不得修改旧 lock 绕过校验。
- 对“同参考残差误差恒等于原误差”、事件场 MSE 分解、等 cell 与 pooled 加权的差别、平均 seed loss 与 ensemble loss 的差别完成独立合成数值检查。附带脚本/JSON 是数学与已报告表值复算，不是生产模型测试。

Codex 负责：按本任务实施独立只读工具与测试，利用本地已存在的结果做实际再分析，准备用户手动运行的 checkpoint 核验与超算评价脚本，保存可独立复核的证据，提交/推送本任务文件。不要把研究判断、指标定义或计算范围重新留给用户决定。

用户负责：提供本地无法解析的集群账户/路径信息，并手动提交需要设备或超算数据的作业。Codex 不远程执行，不自动 sbatch/srun，不自动追加计算。

## 1. 精确身份与入口

```text
repository: rular099/team_pytorch
branch: exp/fe01-feature-extractor-site-effects
base_commit: b2756825190f93034ece436fab778e8a69adb41c
results_commit: b5b2421f2667461e8ae785bcccc01e00411d905d
recorded_training_source_commit: 73ae9dec50713805c878b5e4efeb8264c01d0c76
saved_audited_runtime_code_sha256: 3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea
```

在既有 FE01 独立 worktree 工作，不切换、不清理主线或 V01 目录。先报告真实 HEAD、branch、status。HEAD 与 base 不同先核对 diff，不强制 reset/rebase，不覆盖本地未提交改动。

依次阅读 AGENTS.md、docs/ai/PROJECT_CONTEXT.md、docs/ai/README.md、SESSION_SUMMARY.md；随后阅读：
- reports/fe01_hpc_results_20261004/README.md、CHATGPT_ANALYSIS_PROMPT.md、evidence_checks.json、run_summary.csv、seed_summary.csv、audit_summary.csv、training_time_exposure.csv、training_curves.csv、validation_by_time_geometry_role.csv、validation_paired_event_bootstrap.csv、source_inventory.csv、metadata/。
- docs/ai/FE01_PROTOCOL.md、FE01_REPO_AUDIT.md、FE01_HPC_RUNBOOK.md、FE01_ARTIFACT_SCHEMA.md、CODEX_RESULT_20261004_FE01_HPC_RESULTS.md。
- fe01/data.py、windows.py、sampling.py、extractors.py、model.py、engine.py、metrics.py、analysis.py、spatial.py、replay.py。
- scripts/fe01/run.py、analyze.py、summarize_hpc_results.py；RT55/RT61 的实际模型加载与输出反标准化路径及匹配测试。

**不能直接把 Git 内 metadata/*.json 当成原始运行配置**：它们是改写路径后的可读快照。先只读定位原始上传目录、source_inventory 列出的预测/配置/lock/manifest 和本机已有权重。路径不能从模型记忆猜测。没有生产文件时实现 fail-fast 检查与缺项清单，继续完成不依赖缺项的部分。

## 2. 冻结的模型与计算矩阵

### 2.1 已训练 FE01 九组：不重新选 epoch

| model family | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| phasenet_pretrained_frozen | 12 | 10 | 12 |
| eqt_pretrained_frozen | 10 | 10 | 10 |
| team_original_scratch | 11 | 11 | 11 |

这里的 epoch 当前由曲线/CSV 确认，必须再与实际 checkpoint 校验。`last` 为 epoch12 只用于身份/提交日志及已有最终轮数的敏感性核对；不作为本轮新选择。

### 2.2 复用的两个完整系统参考

- RT55：固定历史预选 epoch32，以实际 checkpoint 内部元数据、原始 config、state 和 SHA 为准。禁止拿名字含 ep32 的 epoch20 权重替代。
- RT61：固定该实验最终 epoch8；核验其父模型/必要组件和实际 config。标记 `enhanced_system_reference`，不伪装成同公共下游的 DiTing 特征组。
- 两者均标记 `legacy_reused_system`；不增加 RT59 等默认矩阵，不新训 DiTing，不把同一个旧模型复制成三份“seed 对照”。缺哪一个就将该参考标为 BLOCKED，其他有效工作照常完成。

### 2.3 避免不必要重算

A. 已有九组选中 epoch 的 fixed-time CSV：核验原字节 SHA、身份和小样本前向一致性后直接复用。**不默认重新跑九组完整 1/3/5/10/20 秒**。

B. 新增 GPU 任务：
1. 权重与既有 CSV 的小样本前向核验；
2. RT55/RT61 的完整共同人口复评；
3. 九组及可用参考在同一个已有随机时刻 manifest 上的 validation；
4. 原生能力范围内缺失的 40/90 秒：TEAM 各 seed 与 RT55/RT61 可评 40/90；EQT 各 seed 只评40；PhaseNet两项 N/A。以实际能力核验为准；短记录不授权截短科学历史；不自动改滚动模式。

C. 图与回放另作独立可手动提交的任务：只使用 seed42 的 PhaseNet/EQT/TEAM，加可用 RT55/RT61，共至多5个 checkpoint，3个固定验证事件。共同范围先做1–20秒、1秒步进。长时段区域动画、全国高分辨率网格均不默认运行。

D. 数据侧只读任务：一个共同 train-only 参考拟合；以九组原始 CSV 做分层/残差/固定目标面板分析。能本地完成就本地执行；缺 HDF5 时生成独立 CPU 超算脚本，不为此启动训练。

## 3. 评价合同：不得在实现中改变

### 3.1 主人口与主端点

- split=validation；共同 cohort 1,310 事件。原训练 cohort 9,084 事件，不能与历史 RT55 的全部 cohort 混称。
- 时间参考为 first_p_pick，T={1,3,5,10,20}s；geometry={normal,random}。
- 原九组每run：全目标194,265条；noninput139,440条；observed_input54,825条；triggered_noninput43,850条；untriggered_noninput95,590条。
- 原目标人口 SHA：`e96284780cb5f72b952c1697d4d264d307e7139a153201fa222566015aec7f69`。
- 原标签/输入/决策 SHA：`f204deca06725af5513895416d5b0691c75c12221127854c32d1d497855155bb`。
- 数量是多时刻/协议下的记录数，不是独立观测数量。新长时刻/随机时刻/诊断子群的数量必须实际导出，不能照抄以上数目。

主 MAE：每个时间×geometry cell 内先对 noninput 目标平均绝对误差，再十个cell等权。点预测为混合分布在 log10(m/s²) 坐标的均值。

```text
Primary MAE = (1/10) × Σ_cell [(1/N_cell) × Σ_target |prediction - truth|]
```
原始 LaTeX：
```latex
\mathrm{MAE}_{\mathrm{primary}}=\frac{1}{10}\sum_{c=1}^{10}\frac{1}{N_c}\sum_{i\in c}|\hat y_i-y_i|.
```

完整请求身份至少包含 dataset_id、event_id、绝对/样本决策截止、history起止、输入station IDs与顺序、输入真实坐标/物理支持、query IDs/坐标、target role、单位。请求由锁定 manifest 决定，不允许各模型再次自由运行 sampler 后声称已配对。不同原生长度造成的左 padding 形状差异不应被误判为物理历史不同。

保持历史主表不变。缺预测、数值失败、cohort或输入/目标不一致必须输出差异并阻止正式配对主表；不得静默取交集改善分数。交集结果只能作为另名探索性附表，列排除分母与原因。

### 3.2 次端点与统计

- 单独列1秒、1/3/5秒早期三点等权，以及normal/random；早期表不是替换既定主端点。
- all、noninput、triggered_noninput、untriggered_noninput、实际input_count=1必须单列。
- MAE、RMSE、R²、预测对真值的 slope、bias、NLL、CRPS、Brier、predictive sigma、混合分位数95%/68.27% coverage和width；Brier阈值固定−1.2。
- 分层沿用已有震级、PGA、距离、深度桶；修正/明确桶端点标签，不能例如右闭区间标为严格小于。padding 分开报告原生长度左padding比例与共同物理历史的storage缺失比例，避免把10000/3001的固有padding差异解释为数据质量差异。
- 分位数coverage与mean±sigma coverage分列；NLL密度坐标必须完成每个模型自己的反标准化。不能用FE01 normalization解码RT55。
- 总RMSE由总平方误差再开方；mean(cell RMSE)如展示须同名明确，不能冒充pooled RMSE。
- 在锁定checkpoint下做5,000次事件簇配对bootstrap，RNG seed=20261001；整事件所有时刻/站点/协议/重复draw共同移动。用(dataset_id,event_id)键或证明event_id全局唯一。
- FE01三seed分别报分数与事件CI；家族均值/样本标准差分列。旧RT55或RT61只有一份参考：逐个FE01 seed与同一参考配对，不能称为“同seed DiTing”。可对三个已固定模型的**平均逐目标损失**做事件bootstrap，标明只对固定三模型条件化，不包含训练随机性或选epoch不确定性。
- 禁止先平均三seed预测再称为三seed平均指标；那是新ensemble，本轮不做。
- 已知validation参与选epoch；新随机时刻是同一事件上的时间泛化检查，不是独立test。不得重新选checkpoint、改阈值、temperature或sigma来让校准变好。

### 3.3 随机时刻 manifest

优先用现有 seed42 对应 audit 产出的随机时刻manifest作为全模型共同请求，核验原字节SHA和cohort，并报告与其他manifest是否相同。它应先于本轮新增预测锁定，不重新按模型抽样。当前生成代码为每事件3个[1,20]秒非固定时刻；实际行数、离散样本与重复时刻以文件核验为准。

若同事件存在重复T：保留原始manifest，不重抽；物理快照只计算一次，以draw_id映射回原采样计划，报告unique snapshot数和draw数。随机时间次指标先事件内按原draw权重平均、再事件等权、geometry等权；不得把重复draw当作独立事件。将长域随机评价留给后续，不擅自把现有[1,20]manifest叫做全1–90验证。

## 4. 工程实现关键约束：旧 runtime 冻结，新评估独立

### 4.1 路径与源码身份

`fe01/engine.py::code_identity()` 包含 `fe01/*.py`、若干legacy文件、`tools/**/*.py`、DiTing YAML及dtbench源码。本轮直接修改 fe01/analysis.py 或新增 fe01/*.py 都会改变训练code hash；现有 `export_evaluation()` 会拒绝。

**因此默认不修改 fe01/、tools/、gemini_models.py、gemini_util_light.py、train_light.py、diting/config/、dtbench或旧配置/lock。**

新增建议位置：
```text
fe01_review/                      # 独立包，不在旧 code_identity 的 glob 中
  __init__.py
  checkpoint_inventory.py
  requests.py
  runners.py
  diagnostics.py
  provenance.py
scripts/fe01_review/
  run.py
  print_submit_commands.sh
  identity_job.sbatch
  evaluate_job.sbatch
  reference_job.sbatch
  replay_job.sbatch
  pack_results.py
configs/fe01_review/
  eval1_contract.json
  checkpoint_manifest.example.json
  cluster.env.example
  jobs_manifest.example.json
  source_pin.json
  # 冻结的请求/案例manifest在真实文件核验后写出，不能生成虚构内容。
tests/test_fe01_review_*.py
docs/ai/FE01_EVAL1_PROTOCOL.md
docs/ai/FE01_EVAL1_HPC_RUNBOOK.md
docs/ai/CODEX_RESULT_20261004_FE01_EVAL1_PREP.md
reports/fe01_eval1_local_<unique-id>/
```

命名可根据现有文件小幅调整，但新增位置和边界不能退回旧 runtime。新runner分开记录 training_source_sha、training_code_sha、evaluation_source_sha、evaluation_module_sha、parent protocol.lock SHA。运行前仍严格核验原115文件与原lock一致，不删除检查、不重签旧lock、不将其改成宽松警告。

同一文件路径的诊断实现修复放在新包内并明确新分析版本；不声称历史已发布报告字节发生改变。没有真正错误时复用原函数，不重复拷贝整套模型。

### 4.2 旧权重适配：两个验证层次不能混淆

**L0：原模型数值等价检查。** 加载RT55/RT61原类、原必要模块、原config和state，strict keys/shapes。对同一个已构造输入tensor，直接旧模型与新包装器输出应一致；核验query单个/批次/顺序，证明包装器没有改权重或读出。L0不是历史整条非因果预处理复现。

**C1：共同严格prefix回放。** 原始波形先在锁定绝对T切断，再做前缀内统计/原生窗口打包。旧模型保持原本所需的通道、归一化方式、尺度特征定义、坐标变换和反标准化；不得把 `FE01FullModel.shape_and_scale` 直接替换进旧模型。必要的loader级去均值只能用前缀有效点，并在bridge specification写清楚。

C1结果命名 `rt55_strict_prefix_replay` / `rt61_strict_prefix_replay`。它们是旧权重在新因果输入协议下的复评，不是历史RT55整条pipeline的原样成绩。FE01本就与旧系统的前处理/训练方式不同，主表必须标注“系统级比较”，不推导纯特征贡献。

把原始request、物理波形/有效区间指纹、模型输入指纹、preprocessor ID都导出。不同前处理允许不同输入tensor指纹，但不得包含不同的未来信息。禁止把真实Magnitude/hypocenter或查询站最终PGA/完整波形传入神经网络的预测路径；它们只能作为标签/事后诊断参考。

若某旧模块必须依赖无法因果提供的输入或checkpoint缺组件，隔离并报BLOCKED该reference；不编造默认值，不静默关闭模块，不增加重训。历史标量始终不填入C1表。

### 4.3 权重身份与小样本复算

inventory逐个只读载入，优先CPU且一次一个大文件。输出：文件SHA/bytes、内部epoch/updates/best_epoch、config和state结构指纹、必要parent/encoder身份、训练source、选中CSV SHA、committed_journals归属。init缺失不自动阻塞已选checkpoint复评，但要标出初始化身份未复验。last缺失只阻塞last/journal完整认证，不伪装已完成；selected checkpoint缺失阻塞该run新增前向。

小样本核验清单：依据已有request元数据，用固定哈希排序，从 normal/random × 1/20秒的4层各取8个decision，共32个；如未覆盖实际单输入和多输入，追加不超过8个元数据选定decision。所有模型使用同一请求集合，非支持/空群报告缺项，不按预测误差挑样本。核对每个decision全部query的物理单位输出/MDN；查询chunk15与单点的一致性另取固定小子集。

新FP32同设备检查使用既有query audit的 atol=rtol=1e-5 为预先默认，分别报告logits或weights、mu、sigma和log-PGA预测的误差；仅在有可复现不同硬件数值证据时另报诊断，禁止为通过而自动放宽阈值。输入/ID/label/clock/mask应精确一致，不适用浮点宽容地改人口。小样本失败不得自动启动全量重算或继续标PASS。

## 5. 固定目标与输入面板：区分信息增益和人口迁移

先只读现有CSV完成：按每个geometry构造跨1/3/5/10/20秒均有预测的event-station交集，保留每时刻真实target role，报告全集与交集的事件/目标损失。另列全过程都noninput的固定面板；这种面板是事后诊断，不冒充在线选点规则。不能因为站后来成为input就继续把该行称为noninput。

从已有CSV能得到的是“固定目标、自然输入变化”的轨迹，不能伪造固定输入反事实。

固定输入反事实只在上述3个案例做新增前向：按1秒的可用性固定S0，后续T只延长S0的可见前缀，不新增输入站；query固定为共同站集且排除S0。与自然新增输入的同坐标输出并列，另列两条路径均noninput的共同子集。原始raw坐标不变，模型自身相对坐标中心若随自然输入变化必须记录，不能悄悄冻结改变模型。

两条路径都每T重新编码当前prefix；同一T内才可跨query缓存。不能把全事件features截token当作因果features，也不能用旧T的缓存冒充新T。

可证伪解释：若固定目标/输入后时间反弹消失，说明原反弹可由人口/几何变化解释；若仍显著恶化，再定位表示/幅值/截断/读出。无论哪种结果，本轮都不自动改模型。

## 6. 场地相关空间诊断：本轮必须可审阅，但不冒充真正空间留出

### 6.1 不允许的“证明”

```text
(prediction − reference) − (truth − reference) = prediction − truth
```
原始 LaTeX：
```latex
(\hat y-b)-(y-b)=\hat y-y.
```
所以单独将PGA误差改名“残差误差”没有新增场地证据。非同心圆外观也不构成地质成因证明。

### 6.2 共用一个train-only参照

沿用 `spatial.reference_design/fit_reference` 的既定设计：intercept、M−4、(M−4)^2、log10(max(R,1km))、R/100、depth/100；R为该代码定义的hypocentral近似；ridge penalty=.1，site shrinkage=10，迭代20次。只用允许train event-station最终PGA各一次，不重复时间，不读validation/test拟合，不因当前结果调超参数。

输出 no-site 参考与train-site参考两种。固定源train cohort/标签SHA、参数、样本量；旧模型也用同一参考进行本轮诊断，不能各自拟合一个有利参照。真实震源M/位置用于参照时标“事后oracle诊断”，不得列作实时信息相同的竞争模型。查询站train-site效果不得输入神经网络。

### 6.3 必须报告的三个互补层次

1. **每事件的整体偏移与空间形状分开。** 在同一event/T/geometry/目标群体内计算：
```text
error_i = prediction_i − truth_i
field MSE = mean(error)^2 + mean((error_i − mean(error))^2)
```
原始 LaTeX：
```latex
\frac{1}{n}\sum_i e_i^2=\bar e^2+\frac{1}{n}\sum_i(e_i-\bar e)^2.
```
同时报centered MAE/RMSE、所有站对的PGA差值MAE、P95−P05 range误差/ratio；每field至少5目标，真值range<.05 dex时ratio为undefined，不补零。偏移与形状使用相同目标集合和权重；未触发子群应在自身群体重新中心化。补均值与shape项是否共同改善的表，不仅总MAE。

2. **近等距离站对。** 沿用现有5km epicentral距离差门限，报告站对数量、差值误差/符号准确率，按event-field等权；对truth差值接近0的符号指标单列可判定分母。不能把大量共享台站的站对当独立样本。与no-site及train-site参考同时比较，不能只看模型之间谁画得更不规则。

3. **跨事件台站残差。** 各time/geometry分别比较event-centered observed/predicted residual在台站上的跨事件平均；至少10个唯一validation事件的站点纳入主恢复统计，其余标sparse并保留数量。报告站点等权恢复MAE、相关性、slope、符号一致、误差CI，按事件簇重算站点统计而不是把同一台站重复时刻作为独立事件。方位/强度稳定性表保留time/geometry维度，不能无意加权为记录次数。相关性不能替代幅值恢复。

对“已见台站新事件”“模型训练中未出现的台站”“仅缺input波形”的身份作明确区分。本轮没有训练期空间隔离，不能把评价隐藏站点称为spatial holdout；原有spatial训练配置不提交。若site/field证据不足，输出INCOMPLETE而不是用总MAE判PASS。

## 7. 图、回放和报告

选择3个案例：以共同cohort元数据中的震级排序，在约25%、50%、90%分位附近按固定规则选取满足至少5个可查询站的不同事件，平局按dataset/event字典序；先锁案例manifest，不按模型误差挑选。额外最差案例只允许从锁定规则的误差排序生成并标“事后失败展示”，不能替代这3例。

每例共同1–20秒逐秒prefix回放；固定seed42三模型+两个参考。不训练、不切滚动。使用相同输入规则、query、地图范围与网格；10km间距、观测站外扩50km用于工作性区域展示，稀疏覆盖/域外标明。网格不等于实际可辨空间分辨率。无DEM时网格高程0仅标作统一演示假设，地面真实性定量评分限真实站点；不暗中插值观测PGA构造“网格真值”。

需要图：每time/geometry的truth-prediction密度/对角线；PIT/reliability与coverage-width；输入站数、未触发与时间分层；field level/shape误差；台站残差恢复；案例真实站点散点、各模型网格预测、误差、相对no-site参照与不确定性。同指标固定色标，图例解释输入/查询站、单位、时间参考、稀疏遮罩和数量。

每一帧预测同一个事件最终PGA，不改为当前已发生峰值。场地图先以log10 PGA mixture mean为标准；线性PGA视图需要明确是10**E[log PGA]、median还是E[PGA]，禁止混称。

计时记录：冷启动单列；稳态整帧包括数据读取/截断/预处理、encoder、全部query、分布评分、I/O各项及端到端；render单列；GPU同步和实际设备/网格点数。报告p50/p95/max、超过1秒帧比例；慢于更新周期时保留原始耗时，不靠丢帧后的视频FPS宣称实时。真实trigger/通信延迟未测时称理想触发因果回放，离线过滤上游因果性仍为uncertified。

新手可读中文报告必须说明：目标、模型与训练差异、数据可见性、协议、人口数量、方法原理、结果、反例、缺项和结论界限。每张关键图说明看什么、数值意味着什么、不能说明什么；图源CSV/JSON随图保存。

## 8. 超算脚本与证据包

新CLI限 inventory/plan/verify/export-existing/evaluate/reference/analyze/replay/pack，不提供train/resume/test动作。只读原结果，输出新 `${FE01_REVIEW_OUTPUT_ROOT}/${REVIEW_RUN_ID}`，目录已存在非空即拒绝。自动下载关闭。

提交打印器默认 DRY_RUN=1，仅生成命令、job manifest及日志目录；假的 sbatch/srun marker测试必须证明准备阶段不调用scheduler。用户复制打印的命令才提交；脚本不能在失败后自动重提交。不要硬编码私有路径/account/partition。沿用实际ROCm/DCU环境，不替换torch或强装CUDA。

至少提供以下分阶段手册：
1. CPU inventory与sacct状态导出，缺项不假冒COMPLETED；
2. 原CSV与新runner小样本一致性/因果/query测试；
3. RT55/RT61共同fixed人口复评；
4. 所有已认证模型的共同随机时刻及能力内长时刻补评；
5. train-only reference、已有CSV本地分析；
6. 代表性3例回放；
7. manifest校验、轻量打包与结果回传。

jobs_manifest给每项model/run/checkpoint/epochs、split、request-manifest SHA、任务范围、dependency、是否需要设备、预期固定人口或待统计分母，不给无测量依据的运行时估计。最少任务的组合可以合并同模型固定/随机/长时刻，但依赖identity门槛必须明确；不重复CPU可复用的统计，不全量重读14GB做每张图。

证据至少包括：
- checkpoint_inventory、request/clock/population audit、原source/新analysis source pins、原CSV SHA及reused-result manifest；
- 小样本MDN/单位/clock复现、未来样本扰动、query与cache检查报告；
- 每run/time/geometry/role metrics与paired CI、实际counts、随机draw映射、失败/缺失行；
- field误差分解、等距离站对、台站残差恢复与reference身份；
- fixed-target自然输入与固定输入反事实、role迁移、案例与网格/coverage manifest；
- 图、逐图caption、图源、replay latency；
- summary.json、gates.csv（完整性与协议门，不把所有模型都要求性能胜出）、provenance.json、artifact_manifest.sha256、中文README。

Git保留轻量表、图和逐事件/逐field加总，原始大checkpoint/波形/全量CSV/动画留用户结果包。不能仅上传summary缺可复核分母；不能把动画存储路径当作已生成动画的证明。

## 9. 验收和停止条件

工程验收：
- 原runtime code SHA保持不变，所有新增推理/分析的身份另记；RT55默认代码/config/state schema不变。
- 已选checkpoint身份、源CSV与请求人口匹配；全部未知字段显式unknown。
- 未来样本NaN/大脉冲修改，T以前预处理/特征/预测不变；明确这不认证上游offline滤波。
- raw/normalization/反标准化与legacy bridge L0/C1区分，metadata labels不进forward。
- query重排/分块/追加无关query、相同T缓存与不同T拒绝缓存均通过；模型全eval/frozen不改变buffer。
- 主端点、残差恒等式、MSE分解、empty/NaN/zero-variance、seed均值与ensemble区别、bootstrap簇/分母的合成单测；新随机manifest重复draw语义可复现。
- `python -m compileall`新模块，`bash -n`新shell/sbatch，聚焦pytest及相关旧回归；实际命令/退出码/skip理由归档。
- 用fake scheduler证明无自动提交。`git diff --check`和仓库私有路径泄露检查通过。

科学输出只作判断输入：保留原9组主结果；同群体完整系统比较、1秒概率反例、level vs shape、site重复残差至少分别给出结果/缺项。没有显著提升不是工程FAIL；它是有效负结果。需要决定是否扩展训练/真正spatial holdout时，由ChatGPT审阅本轮结果与用户共同决定，不在当前脚本自动做。

停止点：本地缺设备/权重/数据时结束于代码测试、实际可完成分析和可提交手册，列明HPC PENDING；不得声称HPC已执行。身份不一致则隔离对应模型，不能重新选epoch/重训掩盖。test在本轮硬关闭。

## 10. 完整交接格式

[AI-HANDOFF]
task_id: 20261004-fe01-eval1-existing-model-comparison
repo: rular099/team_pytorch
branch: exp/fe01-feature-extractor-site-effects
base_commit: b2756825190f93034ece436fab778e8a69adb41c
goal: 复用九组选中权重及RT55 ep32/RT61 ep8，在不训练和不开test下完成同人口系统比较与早期/空间诊断的工程准备、可本地执行分析和手动超算脚本。
verified_facts:
  - 远端HEAD及上述源码/报告已核验；原九组全部是validation，选epoch来自曲线/CSV，尚需checkpoint内部身份。
  - code_identity含fe01/*.py和tools/**/*.py，新增工具应移到fe01_review与scripts/fe01_review，不放宽旧校验。
  - compare_runs硬编码diting_pretrained_frozen配对，不能改旧模型family假装同特征对照；需要独立system comparison。
  - 原spatial函数具备train-only reference、field/site指标，可在新分析包复用并强化群体/时间/seed分层。
  - 原始逐目标CSV/HDF5/checkpoint不在ChatGPT运行环境；全量bootstrap和生产前向本轮未执行。
inferences:
  - 主要信息缺口是DiTing系统同人口证据及空间shape/重复台站残差，而不是更多训练。
  - 同基线残差MAE不增加场地信息；population迁移可混杂时间曲线，应独立诊断。
files_to_inspect:
  - 本文第1节列出的入口、源代码、测试、原始结果与config/lock/provenance。
constraints:
  - 原RT55/RT61和FE01 runtime默认行为/旧lock/旧结果不变。
  - 不train/resume/test/调参/校准/自动sbatch/srun/下载/破坏性Git。
  - 只工作于独立FE01 worktree，新的分析和输出有独立identity。
proposed_change:
  - 按第4节新增fe01_review、scripts/fe01_review、configs/fe01_review和聚焦tests/docs；本文件不是已实施修改。
  - 只读旧预测先复用，新增前向仅补identity/reference/random/long-time/3-case replay缺项。
acceptance_checks:
  - 完整执行第9节；真实不可执行项写NOT_RUN及原因，不用合成结果冒充Japan。
hpc_followup:
  - 用户手动按第8节分阶段提交；Codex本轮只准备/打印，不执行远端。
risks:
  - validation选择偏差、原生能力条件化曝光、预训练/归一化/adapter混杂、旧权重strict-prefix分布变化。
  - 稀疏未触发/台站群体、没有真正spatial训练隔离、upstream offline causality与通信延迟未认证。
open_questions:
  - 无需用户重新决定本轮科学范围；账户/资源/路径只在现有文件不能解析时补填。
  - 新DiTing受控训练、参数改进、校准或真正spatial holdout均不属于本轮，后续另行决定。
[/AI-HANDOFF]

Codex 完成后返回：
```text
[CODEX-RESULT]
task_id: 20261004-fe01-eval1-existing-model-comparison
base_commit: b2756825190f93034ece436fab778e8a69adb41c
result_commit: <真实完整SHA；未commit则none>
branch: exp/fe01-feature-extractor-site-effects
changed_files:
  - <实际文件和作用>
verification:
  - <实际命令、PASS/FAIL/NOT_RUN及原因>
compatibility:
  - <原runtime指纹、RT55默认state/输出回归、新分析identity>
local_results:
  - <真正完成的9组再分析、缺项、产物路径与hash>
checkpoint_status:
  - <11个候选逐项身份状态；不可见不写通过>
hpc_status:
  - NOT_SUBMITTED_THIS_TASK
manual_next_commands:
  - <用户应先执行的准确命令；新目录、输入和输出明确>
remaining_risks:
  - <仍未核验事项和blockers>
review_request:
  - 请ChatGPT按result_commit核查legacy桥接、原源码锁保持、统计端点/人口与手动提交边界。
[/CODEX-RESULT]
```
