# DiTing ON 综合评价 V4：横向比较、分层、逐事件 shadow、动态 PGA 热力图与可用成图时间

**任务日期：2026-10-21。任务类型：独立评价工具实现、已有结果再分析、超算补算交付；不训练。**

本文件完整合并并替代三个实施入口：

- `DITING_ON_COMPARISON_HANDOFF_20261021.md`（V1 横向配对）；
- `DITING_ON_STRATIFIED_COMPARISON_HANDOFF_V2_20261021.md`（V2 分层与及时性）；
- `DITING_ON_SHADOW_ADDENDUM_V3_20261021.md`（V3 有效预警盲区）；
- 新增用户要求：每次地震随时间更新的最终 PGA 预测热力图/等值线、台站验证、达到误差要求的最早及稳定成图时间。

**V4 的执行范围变更：** 用户已要求将必要补算准备为超算任务。因此，新时刻/固定查询面板/网格前向和延迟实测不再仅列为“未授权的后续设想”，而是本轮必须实现、测试并交付手动提交脚本的工作。Codex 不自动提交或运行正式计算；用户在超算上手动执行。旧版“仅 CPU、禁止实现新前向”的范围限制被本文件取代，旧版“不重训、不打开 test、不改变旧模型”的限制继续有效。

> 本轮目的不是预设 DiTing 必须胜出，而是检验：在共同信息、目标、规则和计算条件下，它是否更早给出准确、可靠且可行动的最终 PGA 空间预测。所有不利和不确定结果保留。

---

## 1. 身份、分工与研究边界

### 1.1 证据基点

```text
repository: rular099/team_pytorch
source_branch: exp/fe01-a01-absolute-amplitude-ablation
base_commit: 29ffb20ba0effc0c345a0329ba3f8c658dc8cef0
source_results: reports/fe01_a01_hpc_results_20261021/
task_id: 20261021-fe01-on-integrated-warning-map-v4
```

上述是 ChatGPT 本轮实际读取的远端身份，不代表 Codex 本地 HEAD 或超算已部署内容。开始前只读记录 HEAD、worktree、status、相关 staged/unstaged diff；有变化先核对相关差异。不得 reset、clean、强制 checkout、自动 stash 或覆盖未提交工作。

若 V1/V2/V3 已在独立评价目录开工，在同一独立实现上增量扩展，不删掉重做、不重复造两套统计。若尚未实施，建议从上述精确 base 建新分支 `exp/fe01-on-warning-map-evaluation`，独立 worktree `../team_pytorch_fe01_on_eval`。不得切换或覆盖 FE01-A01、FE01-EVAL1、FE02、V01 正在使用的工作目录。

### 1.2 分工

- **ChatGPT：** 本文件已确定研究问题、对照、阈值纪律、人口/时钟/统计定义、模块边界和验收条件。后续基于精确 result commit 审阅代码和证据。
- **Codex：** 读真实代码与可见资料、实现工具、执行本地测试和可执行的真实只读分析、打包独立 release、交付 Bash/Slurm 入口和运行手册。不能只返回另一份规划。
- **用户：** 运行手动超算提交命令。大 checkpoint、HDF5 和完整预测通常在超算，不要求用户搬回本地或重复上传。

### 1.3 固定科学范围

- 四种模型：DiTing、PhaseNet、EQT、TEAM；全部 **ON / physical_scale_keep_duration**，validation、seed42。
- 固定 selected：DiTing ep12、PhaseNet ep12、EQT ep10、TEAM ep11。已有固定 epoch12 导出作稳健性检查。
- selected/epoch12 不是独立训练重复。密集回放与网格默认只跑 selected，不为 epoch12 复制整套昂贵回放。
- 不重选 checkpoint、不调整 sigma、不改网络结构、encoder、adapter、损失或训练 sampler，不新增 ON/OFF/multi-seed/空间留出训练。
- 不打开 held-out test，不把 validation 改名 test。旧 RT55/RT61 完整系统不加入本次排名，FE02 也不混入。
- ON/OFF 仅保留已经完成的辅助消融，不新增 OFF 前向或训练。
- 新端点属已有结果之后提出的探索性评价；“运行前冻结合同”不等于原始研究已预注册。

### 1.4 先读资料

依次读 `AGENTS.md`、`docs/ai/PROJECT_CONTEXT.md`、`docs/ai/README.md`、`SESSION_SUMMARY.md`，再读本文件及 source_results 中 `CHATGPT_ANALYSIS_PROMPT.md`、`README.md`、`DATA_DICTIONARY.md`、`evidence_checks.json`、`export_condition_checks.json`、各表及 `method_sources/`、`source_metadata/`。

然后追踪：dataset/sampler → cutoff/storage mask → 输入/查询台站坐标 → shape_and_scale/encoder/adapter → PGA readout → 损失定义 → 精确 MDN 导出 → 统计/回放。

直接相关文件包括：

```text
fe01/data.py, windows.py, model.py, extractors.py, engine.py, metrics.py, spatial.py
fe01_a01/model.py, identity.py, provenance.py, evaluation.py, equivalence.py
fe01_review/requests.py, runners.py, replay.py, diagnostics.py
scripts/fe01_a01_ops/ 中已有只读扫描、修复、status 和封装工具
相关 configs、tests 及 docs/ai/FE01_A01_* 运行手册
```

特别读取旧 scratch encoder 检查、float CSV round-trip、RNG resume 与多来源 seal 的历史修复。不要把当时故障重新引入本任务。

---

## 2. 本轮已核验的事实与不可越过的推断

1. 源结果每份完整导出为 1,310 事件、194,265 请求；noninput 139,440，observed_input 54,825，triggered_noninput 43,850，untriggered_noninput 95,590。它们是旧快照协议的核对数，不是新固定全查询回放的必然计数。
2. 原主指标：首 P 后 1/3/5/10/20 秒 × normal/random 十 cell 等权 noninput MAE；cell 内目标请求等权，单位 log10(m/s²)。
3. 目前四组是同种公共下游的方案比较；跨前端仍有原生窗口、训练时间曝光、预训练和 adapter 混杂，不是预训练架构纯因果实验。
4. 当前模型输出 **最终 PGA 概率分布**，不输出首次越阈时间。每帧预测要与同一台站最终真实 PGA 比较；累计至当前峰值可另作描述，不能替换标签。
5. 当前 `fe01_review/replay.py` 已有有限案例、网格/站点预测、同截止缓存和分项计时，但其旧实现只画 1/5/20 秒、没有动画，且记录 `grid_elevation=0m demonstration only`。不能原样包装成新的正式动态空间验证。
6. 旧 `forward_seconds` 是 query 分摊时间；不能直接乘除后当整区域在线时延。
7. 源头离线处理仍为 `UPSTREAM_CAUSALITY_UNKNOWN`，触发依据目录拾取的回放是理想触发回放，未认证实际通信和连续事件检测。
8. 真实连续地面没有全覆盖观测真值。网格预测不等于网格有真值；主要准确度在真实台站评分。
9. “第一秒贡献总体优势约70.4%”是旧表值差的代数分解，不是 MAE 降低70.4%，也不是因果归因；本轮重新计算并保留正负 cell。

不能预设“较低 MAE → 更小 shadow”或“空间范围更大 → 地图更准确”；这些由新评价直接检验。

---

## 3. 实现隔离与身份管理

新增代码优先放在上一版指定的独立目录，已有实现则复用：

```text
fe01_a01_compare/
  inventory.py, contracts.py, sources.py
  statistics.py, stratification.py, covariates.py
  quality.py, clocks.py, crossings.py
  replay.py, query_grid.py, latency.py
  shadow.py, map_readiness.py, plotting.py
  packaging.py, cli.py
scripts/fe01_a01_compare/
configs/fe01_a01_compare/
tests/test_fe01_on_*.py
docs/ai/FE01_ON_EVAL_*.md
```

不要改旧115文件、`fe01_a01/`、`fe01_review/`、`scripts/fe01_a01/`、`configs/fe01_a01/`、旧 lock/config/checkpoint/seal。尤其 `fe01_review/*.py` 属于 A01 内容指纹；旧 A01 的 tests/docs 也被清单覆盖。新测试/文档使用 `FE01_ON_EVAL` 而不是触发旧清单的 `FE01_A01` 名称。

可导入已认证实现，或新增明确的 wrapper/类/依赖注入；不以全局 monkeypatch、删除检查、重签旧 lock 绕过原身份。确需修正评价适配时只影响新路径，保存差异、数值回归和新的 execution SHA。

四种 ON checkpoint 通过原 config、内部 epoch、权重 SHA、状态结构/内容 SHA、encoder SHA、归一化和原审计定位，不能仅看路径名。TEAM encoder 是训练过的，不要求它仍等于初始化 SHA；冻结 encoder 仍须做恰当的一致性检查。

旧 source seal 验原未脱敏字节，新归档 manifest 验导出的脱敏字节；两者不可混算。输入只读，输出必须与旧训练/评价目录分离，使用新 run ID 和不可覆盖的 stage/task attempt。

---

## 4. 交付路线：三类计算共享同一个事实库

### A. 已有预测 CPU 分析（先交付）

不加载模型，复用已封存的四组 ON selected/epoch12 预测与逐事件表，完成原端点、跨模型配对、现有字段分层、事件稳健性、已有稀疏时刻曲线。现有资料不足不妨碍交付可算部分。

### B. 波形/目录只读 CPU 补充

从超算原数据补协变量、SNR、逐站有效长度、UTC/origin 时钟、真实阈值首次越过时间、静态网格元数据。按事件读一次并去重缓存，不为每模型/每query重复读波形。不调用模型。

### C. 已冻结 checkpoint 的 GPU/DCU 补前向

构造新且共同的固定目标回放，补首 P 后逐秒站点预测、网格场、同设备时延。先 smoke 与小规模 pilot，再由用户手动提交全量 stage。**不重新训练**。

最终所有统计（分层、shadow、地图准确度和 TTUM）尽量复用同一 station replay store；画图只读 numeric store，不重新跑网络。

---

## 5. 共用请求与结果 schema

### 5.1 三种请求协议单独命名

1. `legacy_snapshot_v1`：旧的正常/随机查询人口，保持原十 cell 和完整 hash，不改成全查询人口。
2. `fixed_query_replay_v4`：每事件固定真实站点集合，首 P 后1–20秒，normal/random 分别使用共同输入计划。所有目标始终输出，不因其成为输入站而删除。
3. `grid_replay_v4`：同事件、同截止、同输入计划、同固定网格，网格没有真实 PGA 标签。

新请求清单的不同人口不是原协议损坏；必须新命名、新hash，禁止伪装成原194,265请求。

### 5.2 共享 sidecar

至少分为 `event_covariates`、`input_station_covariates`、`request_covariates`、`crossing_truth`、`grid_manifest`。所有模型共享同一份标签/协变量，不按模型注意力、预测或不同原生填充率各自分桶。

键包括 dataset_id、event_id、协议ID、geometry、整数 cutoff、station_id 或 grid_point_id。ID按字符串读取，CSV float 使用 round_trip；不要依靠文件行序。记录原请求时刻和量化后的有效截止，不能将浮点时钟舍入差默认为同一输入。

来源字段至少包含：数据/模型/代码/合同/hash、checkpoint epoch/policy/seed、时间参考、输入 IDs及顺序、mask身份、可见历史、单位、scope、available_at_decision、retrospective_only、quality/status/reason。

### 5.3 模型接口与未来隔离

目录震源/震级、完整目标波形、最终PGA、S到时和越阈时间只用于评分或图注，不能成为 forward 输入、归一化或输入站选择条件。合法目标人口由已有数据合同定义，不新增按标签强弱筛选输入。

原PGA标签分量定义、station第三坐标含义/单位、地理坐标原点变换必须追踪生产链；不从字段名猜测。新路径与已认证前向在相同输入上数值一致。

---

## 6. 四模型横向统计（完整保留V1）

### 6.1 对照和端点

差值统一 **DiTing减对照**，分别对 PhaseNet、EQT、TEAM。MAE等误差/概率评分负差为更优；覆盖率/宽度不机械按方向判优。

原十 cell 主端点不变；新增第一秒两个 cell、早期1/3/5秒六 cell 为重点探索性副端点。selected 与 epoch12 单列；all、noninput、triggered/untriggered、K1/Kmulti 都报告，不以 all 代替主排名。

指标：MAE、先平均 cell MSE 后开方的 RMSE、精确 MDN NLL/CRPS/既有阈值Brier、bias、预测对真值 slope、R²、P90/P95绝对误差、sigma、68%/95%分位覆盖和宽度。不可平均cell R²或P95后冒充pooled端点；低方差时R²/slope为NA并报告原因。

NLL从raw logits稳定计算；均值是log10 PGA坐标中的混合均值。若图改用m/s²，`10**E[log10 PGA]`不是线性混合均值，须准确命名。概率分位数按混合分布，不用mean±2sigma替代。

### 6.2 区间、稳健性与支持

- 原主统计采用2000次、RNG20261007事件簇bootstrap；同一事件所有站、时刻、geometry及四模型一起抽。
- 每次重采样重新计算RMSE再做模型差，不能给MSE差开方。
- 逐事件loss差、胜率、中位数、leave-one-event-out和事件贡献集中度另报；事件等权不替换target/cell等权。
- 主人口不匹配须fail并导出差异，不能静默取交集。计划允许的新子群可做显式共同支持子集，但披露排除人口。
- 每桶给events、stations、decisions、requests、fields、支持/缺失cell、有效重复数、空cell比例、缺元数据比例。
- <20独立事件的副分层只作描述，不加胜出/显著标签；这不是20以上必然有效的保证。
- 大量空cell剔除的区间标条件化/不稳定；不补零、不增加draws掩盖稀疏性。
- 多端点均属探索性、默认未校正；不得只展示显著或有利分组。训练seed和选轮不确定性不由事件bootstrap覆盖。

空间场继续使用>=5目标的event/time/geometry field，level/shape/range与等距离站对原定义保持不变，明确field等权和实际分母。

---

## 7. 分层比较与质量补充（完整保留V2）

### 7.1 基本分桶

| 因素 | 首轮统一边界/定义 |
|---|---|
| 目录震级 | M≤4、4<M≤5、5<M≤6、M>6；先核查震级标度，不默认Mw |
| 查询站真实最终PGA y | y≤−2、−2<y≤−1.2、−1.2<y≤0、y>0；y=log10(m/s²) |
| 查询站震中距km | [0,25]、(25,50]、(50,100]、(100,200]、>200 |
| 震源深度km | [0,10]、(10,30]、(30,70]、(70,300]、>300 |
| 实际有效输入数K | 1、2、3–4、5–8、9–16、17–25 |
| 最近输入站距离km | [0,5]、(5,20]、(20,50]、(50,100]、>100 |
| 共同物理历史缺测率 | 0、(0,.25]、(.25,.5]、>.5、Unknown |
| 输入SNR代理量dB | <0、[0,10)、[10,20)、≥20、Unknown |
| 发震后数据截止时间s | [0,5)、[5,10)、[10,20)、[20,40)、≥40、Unknown/异常 |

边界单位、闭开侧写入合同；不按结果优化。源标签分层是离线诊断，不代表模型当时知道震级/真实PGA/震中。高PGA桶的coverage不能机械要求每桶95%，正式可靠性分析另按可见输入变量分组。

增加输入凸包内外/退化、输入分布跨度或方位缺口；K<3、共线或重合标degenerate。面积/距离用明确投影/测地方法，不把经纬度直接当km。

增加有效P后观测秒数的最小值/中位数、已有>=1秒P后观测的输入数。S到时有可信来源才使用；未知为NA，不用估计走时冒充实测。站点训练曝光、区域/VS30只在已有可信元数据可连接时补，不自动新增外部数据库。

### 7.2 只做预定的四类交互

时间×K；时间×SNR；真实PGA×最近输入站距离；震级×震中距。先只看支持量，冻结可报告cell，再看模型效应。若宣称低SNR优势更强，比较效应差并对共同时间/K支持域标准化；不能用“一个显著另一个不显著”代替交互检验。

### 7.3 SNR计算合同

主名 `causal_first_second_snr_proxy_db`。每个selected input i：噪声窗[P_i−5s,P_i−1s)，信号观测窗[P_i,min(P_i+1s,T)]，仅有效三分量采样。噪声窗覆盖>=80%且有效>=2s；信号窗覆盖>=80%且有效>=0.5s。去噪声均值，再比较观测窗与噪声窗的三分量均方能量。

```text
SNR_proxy_i(T) = 10 log10(E_observed_i(T) / E_noise_i)
```

```latex
\mathrm{SNR}^{proxy}_i(T)=10\log_{10}\frac{E^{observed}_i(T)}{E^{noise}_i}.
```

分子是含噪声的观测量，不声称已获得纯净信号能量。P前补零不充当噪声；缺窗、零分母、时钟问题、可核验clipping等明确Unknown。极小正数不能偷偷制造超高SNR。

decision主SNR取可计算输入站中位数，同时保存下四分位、有效站数/K。可算站比例<80%主分层为Unknown。所有模型使用相同物理质量sidecar，不用各encoder不同归一化或不同native长度分母定义。

真实缺测与原生容量左填充分开；合法零不是缺测。已导出valid_seconds平均值不能恢复最差站或每分量质量，所需细节补读mask。

---

## 8. 回放计算计划：默认逐秒公共域，固定查询站点

### 8.1 主计算范围

- cohort：锁定的1,310个validation事件；每事件保留原资料可核验的真实目标站全集，所有模型一致。
- times：首P后1,2,…,20秒，正常/随机geometry均实现；模型使用原native_prefix历史，不改rolling。
- selected四模型：最多 **209,600个模型×事件×时刻×geometry帧**，这是任务计数上界，不是独立样本量，实际支持/无输入/失败分列。
- 数值station replay覆盖全部合格事件；每事件地图也提供全量任务，不只交几个好看的示例。
- 数值grid replay和动画默认先跑normal的全事件计划；random提供单独、同样完整的提交stage，主结论不得把缺少random网格说成已完成两协议地图。station统计两协议都保留。
- 未支持的事件保留占位记录和原因，不静默消失。原cohort本就筛去了一些事件，不能将其称为全部日本地震的端到端检测评测。

### 8.2 Pilot与全量

先按元数据/hash选择6个试运行事件（事件不足则实际数量），覆盖K1/多台、记录长度和查询数；不能按预测误差筛选。Pilot只验证正确性、吞吐/内存/磁盘和出图，不用于调科学阈值。

pilot通过后用户分别手动提交全量station/grid阶段。prepare必须打印真实任务数、模型加载次数、Q/grid点数分布和预期文件量；pilot可产生资源测量，但不得在超预算时悄悄删事件、稀疏时刻或降低科学范围。需要用户调资源时只影响该阶段，不阻塞CPU统计。

模型每worker加载一次，依事件分块处理；不得每帧重载DiTing。无训练DDP要求；默认一任务一DCU，实际内存以pilot核实，不能强行继承16卡训练配置。

### 8.3 输入规则与缓存

- normal按同一因果到时/有效数据/容量规则增加站点；random固定每事件站点优先级及抽取K规则，禁止每帧独立换站。若与旧snapshot随机实现不同，单列replay协议，不伪装旧分数。
- 既有1/3/5/10/20点且输入一致时，对重叠站点检查复现；输入不一致必须报告manifest差异，不能跳过一致性门后混用。
- 同一cutoff的station/grid/query chunk可以复用经验证的station编码；跨时间、输入ID、mask、坐标、模式、checkpoint绝不复用旧特征。
- query顺序、batch大小、追加grid queries不应改变原station预测；必须有回归测试。
- raw数据缓存允许减少I/O，但每次归一化、幅值、encoder只能访问当前切片；未来NaN/Inf/脉冲与合法采样正控制完整测试。
- grid路径不得调用“仅能查询event station ID”的接口偷换成最近台站；用经核验的任意query坐标接口，不改模型参数。

### 8.4 固定评价面板

- `fixed_all_targets`：系统成图/主shadow面板，全程同一批真实站点，即使后来成为input也不删除。
- `fixed_never_input_targets`：在共同1–20s回放范围内从未被选为input的固定目标，仅用于空间推断副分析；这是回顾性评分面板，不参与在线输入选择。报告排除后的事件/站点数。
- `current_noninput_targets`：当前角色的描述性曲线，可变人口，不能拿来定义主TTUM。
- 高PGA子面板固定于真实最终标签，只作离线风险分析；样本不足NA。

### 8.5 发震时钟和长时间扩展

第一版必须把所有已有/逐秒预测转换到核验后的origin时间轴；这是重新标注同一绝对cutoff，不是改变history起点。

另交 `origin_replay` 显式stage，可按配置[5,10,15,20,30]秒构造严格origin截止：先转成相对原first-P reference的elapsed，再调用原prefix。不得简单换loader reference造成pre-P历史改变。elapsed<1s、超native能力、无输入分别标out_of_domain/unsupported/no_input。

40/90秒只作显式extension stage。禁止把短窗模型切成rolling补表；原生能力N/A不等于晚期不能用滚动模型。全模型主要排名仍限共同公共域。

---

## 9. 统一网格与动态热力图

### 9.1 每事件固定空间合同

区域由静态站点坐标/事前规则确定，四模型/全时刻相同，不按哪个模型的预测最强来改变范围。默认用事件候选真实台站范围加20km边界作离线展示域；标注这是回顾性选图域，不是已验证的在线震源定位范围。

用局部米制投影生成10km网格，默认max_points=4096。超过上限时，在看预测前按统一整数倍确定性增大间距直到达限，并保存真实网格；不能仅为DiTing降分辨率。所有图显示实际间距，不能将网格间距称为预测分辨率。

网格第三坐标必须与训练台站metadata的含义/单位一致。优先只读已有可信DEM/静态台站元数据；缺失则：

1. 输出真实站点预测图、台站支持的预测插值展示与TTUM，不阻塞主站点评分；
2. 允许明确标 `metadata_estimated` 的站点高程插值，只在可支持区域，报告方法；
3. 固定0高程只能单列 `demonstration_only`，不能悄悄进入正式网格比较/面积结论。

没有本地海岸线/底图时使用无底图的真实坐标图，不自动联网下载或使用假地形。

### 9.2 每帧交付内容

- 主图：模型对同一事件**最终log10 PGA**的网格预测，固定colorbar，可带固定级别等值线；不画成当前累计峰值。
- 同坐标叠加观测站真实最终PGA色点，输入站单独符号/描边；真实值图层醒目标 `retrospective truth — not available online`。
- 辅图：站点 `prediction−truth` 误差散点、混合分布95%区间宽度或超阈概率图。
- 多模型对照：相同event/T/view/colormap/levels，不以自动缩放增强某模型视觉。
- 标注first-P时间、origin时间（可得时）、数据截止、计算完成/交付场景、实际K/Q、checkpoint、单位和元数据/观测支撑状态。
- 震中符号仅为回顾性图注，不能表示模型已知源位置。
- 台站误差使用对其真实坐标的直接模型前向；网格插值到站点的误差可另报以检查渲染近似，不能与直接query分数混合。
- `DiTing−PhaseNet`预测差图不是准确度差图；只有站点上有真值时才画绝对误差差。
- 不把台站误差连续插值结果命名为实测整区域误差图。

预测色标默认[-4,1]dex，误差色标[-1,1]dex，区间宽度[0,3]dex；溢出明确标，原数值不裁。等值线默认[-2,-1.2,-0.5,0]dex，所有模型共用。可配置，但运行前冻结，不依预测随帧漂移。

空间显示支撑约定（例如距输入站100km内）只作为图层，不作为“预测正确”的证明，更不得据此删掉远端误差。未支撑区灰网纹，raw grid预测可保留但明确性质。

### 9.3 输出规模与案例选择

所有合格事件保存逐秒数值和动态展示产物；normal全量为默认成图交付，random另stage全量可提交。先交每事件低分辨率预览/HTML索引及GIF或MP4，高清PNG只需1/3/5/10/20及TTUM前后帧。全帧PNG可配置。

安装可用时MP4；无ffmpeg则GIF/HTML帧浏览，不能因此没有动态产物。不能只交生成视频的代码而没有在可执行环境跑通的合成样例。

主报告挑选12个事件作详细解释，按震级/距离/K/资料质量及固定hash事先选，不能只挑DiTing赢的事件。另可按预定规则列出改善最大/退化最大的病例，必须对称展示并标为结果后案例。

网格高PGA面积为模型预测量；与独立插值参考的IoU只作辅助，不可当真实连续区域得分。网络预测等值线不能反向定义自身评分真值。

---

## 10. TTUM：台站支撑的“达到可接受误差的最早成图时间”

### 10.1 名称和可解释性

TTUM在本项目中是新定义的**离线评价量**，不是既有工程标准，也不是模型上线时可根据未知最终真值自动判断的状态。

更准确名称：`station_supported_time_to_usable_map`。它说明在这批固定观测站点上何时满足误差要求，不认证未观测连续地面都准确。

只把各时刻预测与同一真实最终PGA比较。模型每帧PGA变小/变大都允许，不加单调性约束或平滑来优化TTUM。

### 10.2 误差标准：不追着结果定阈值

业务“合格标准”尚未给定，`engineering_acceptance=null`。不能从validation效果选出DiTing最有利的阈值再称工程标准。

为保证本轮可以运行，提供**明确标注为研究敏感性**的三个完整预算 `b ∈ {0.15,0.20,0.25} dex`，全部输出，不挑最佳b作唯一结论。任一预算b的固定站点面板至少N=5：

```text
station_pass(b,t) = [MAE(t) <= b]
                    AND [abs(bias(t)) <= b/2]
                    AND [fraction(abs(error(t)) <= 2b) >= 0.90]
```

```latex
G_b(t)=\mathbf1\{\mathrm{MAE}\le b\}\mathbf1\{|\mathrm{bias}|\le b/2\}
        \mathbf1\{N^{-1}\sum_s\mathbf1(|e_s|\le2b)\ge0.90\}.
```

`field_pass`更严格：在station_pass基础上增加 centered_RMSE≤b 和 `abs(prediction_range−truth_range)≤b`，range是field内P95−P05。极低真值range时不报range ratio，不使用“range至少恢复一定比例”强迫平坦真值制造变化。N<5、缺真值、输出缺失不得判pass。

高PGA面板另外完整报告误差和风险指标，有足够站点时可计算同样TTUM，不强迫每个事件满足高PGA目标最少数量。概率校准不要求每个小事件95%覆盖恰等95%；逐事件宽度/coverage仅描述，正式校准在足够人口聚合检查。

### 10.3 首次、稳定起点、稳定确认分开

在固定逐秒时刻表上：

```text
T_first = 第一帧满足G_b的时间
T_stable_onset = 首次连续3帧均满足G_b的起始帧时间（回顾性）
T_stable_confirm = 上述3帧中的最后一帧时间
```

```latex
T_{first}=\min\{t_k:G_b(t_k)=1\},\quad
T_{onset}=\min\{t_k: G_b(t_k)=G_b(t_{k+1})=G_b(t_{k+2})=1\},\quad
T_{confirm}=t_{k+2}.
```

“连续3帧”指新合同的1s连续更新，缺一帧打断，不能把旧1/3/5s三点当3秒稳定。不把稳定起点回填成当时已确认；真实最终标签本来也是离线信息。

另报首次过线后的退化次数、最后不达标帧、到5/10/20秒已达到稳定确认的事件比例。后续又退化可在表中揭示，连续3帧不等于此后永久稳定。

### 10.4 未达标、删失和比较

- 全域有完整输出但到20s仍未达标：`not_reached_within_20s`，不是删除；在20s终点CDF分母仍保留。
- 帧缺失、原生不支持、时钟/真值问题：Unknown/unsupported单列；不能当好结果或按某个任意大秒数填值。
- 19/20s开始满足但没有后续足够帧：stable确认未知/右删失，不能认定稳定。
- 第一帧就达标：`first_observed_at_1s`，不能断言1秒之前未达标；物理连续首次达标时间左删失。
- 未定义阈值时输出完整误差—时间曲线和预算敏感性，不伪造工程达标时间。
- 每事件所有模型同一评价面板、预算、更新表和观测终点。目标随角色迁移而减少的current_noninput不能作为主TTUM。
- 以全部合格事件为分母报告达到比例/上下界；中位数若不到一半事件达标则写“未在观察窗口内达到”，不能只在成功者中求中位数。
- 配对“早了几秒”仅对双方均可确定的子集给条件分布，并同时报告双方达标/仅一方达标/均未达标/未知计数；不用只保留共同成功者形成虚假普遍优势。
- bootstrap按事件；逐事件TTUM与原target/cell端点是不同权重。

### 10.5 观测时间与交付时间

同时保存相对first-P、相对origin、数值地图就绪和可视化交付场景下的TTUM。缺真实通信时延时不能叫部署实测TTUM。

`numeric_map_ready`包括需要的预处理、编码、全部站点/网格query和概率计算；`rendered_map_ready`再加实际渲染/发布。离线存档与统计打包开销另列，不混成用户服务时延。站点-only前向快不代表同事件网格图也同样快。

---

## 11. 逐事件 shadow / 阈值条件有效预警盲区（完整保留V3）

### 11.1 参数与默认研究网格

业务阈值未定，不自动宣布某个p*为运行策略。为完成方法验证，完整输出固定研究网格：

```text
y_threshold ∈ {-1.2, 0.0}，PGA单位m/s²，A=10**y_threshold
p_alarm ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
minimum_lead_seconds ∈ {0, 3, 5, 10}
```

这些是探索性敏感性，不是工程阈值，不在validation中挑最佳p*或单独给各模型调阈值。已有Brier主阈值−1.2保持不变。业务参数将来明确时作为新合同版本重算，不能覆盖原研究表。

### 11.2 定义与分类

```text
H_e(A) = 真实最终PGA >= A的目标站集合
p_m(s,t) = 完整MDN给出的超阈概率
t_alert = 满足固定报警规则的警报最早送达时间
L = 真实首次越阈时间 - t_alert
及时 = (L>0) 且 (L>=tau)
```

```latex
H_e(A)=\{s:Y_{e,s}\ge A\},\quad L_{m,e,s}=t^{cross}_{e,s}(A)-t^{alert}_{m,e,s},
\qquad timely=\mathbf1\{L>0\}\mathbf1\{L\ge\tau\}.
```

默认报警为首次p>=p*、发出后锁存；撤销/多次确认只能单独策略。若要求连续若干帧，以确认帧发报，不倒填起点。p必须从完整MDN计算，不能均值越阈冒充。

分类：timely、positive_but_short、late_or_zero、not_alerted_by_deadline、unknown_prediction_horizon、unknown_crossing、unknown_clock、missing_export、unsupported_history等。tau=0且L=0仍是迟报；L=tau>0算满足。

H为空的事件shadow比例NA，不记0当成功。真实日志确认no_input/系统失败且确实没有服务，与文件丢失导致未知分开。共同输出窗口开始前，只有明确声明“系统在此前不提供服务”的回放策略才能作系统未及时服务判断；不能推断物理上永远不可能预警。

目标deadline若晚于完整预测覆盖终点，无警报结局为未知，不将20–30s未观察的未来算漏报。概率不必单调，旧稀疏帧不插值。

### 11.3 真实首次越阈时间

先核验PGA标签生产的分量/滤波/单位定义，用目标完整记录复算最终峰值与标签的一致性。首次越阈不是最大峰值时刻，也不能直接用S到时代替。

记录左端已超过阈值、越阈前有缺测、clipping、波形不足或标签不一致，首次越阈可能删失或未知。可给严格界，但不得伪造精确时刻。目标完整波形只由评分模块读取，不传入模型流程。

### 11.4 站点级与面积分开

每事件在同一H上输出及时、迟报、截止未报、未知数量和lead分布。N_H中N_bad确定失败、N_unknown未知时：

```text
失败比例下界 = N_bad/N_H
失败比例上界 = (N_bad+N_unknown)/N_H
```

```latex
B_{lo}=N_{bad}/N_H,\qquad B_{hi}=(N_{bad}+N_{unknown})/N_H.
```

这是信息不完整的界，不是95%统计CI。已知子集比例必须附未知比例，不因缺输出获益。

每次地震的真实坐标分类图是必交。连续平方公里面积只有独立共同空间支撑时另算：等积投影、裁剪且限距的台站代表区域、共同陆域/研究域/权重，标 `station_support_weighted_area_estimate`。没有支撑则NA，不能用模型自己的高PGA预测区域定义它自己的真值分母，也不能用失败台站凸包冒充真实shadow。

可选等效圆半径r=sqrt(area/pi)仅作面积摘要，不代表圆形真实盲区或界外安全保证。

### 11.5 必须并列误报代价

同一A、p*、tau下并列shadow、及时检出、假阳性率、警报精确率和概率评分；始终报警基线应显示极小shadow但高误报，作为测试控制。

假阳性率分母是真实未超阈且有完整观察的站点；警报精确率分母是报警站点，两者不能混用。当前地震目录仅支持这些事件内空间假警报，不支持每日连续运行假警报率。

完整输出shadow–false-alarm曲线，不按每模型寻优；不同阈值下的结果不可混入一个最佳排名。

---

## 12. 时延：公平硬件、共同调度、不可忽略积压

新交付 `latency` stage，真实DCU上按相同precision、device、K、query/grid数、batch/chunk、warmup和缓存原则测量；checkpoint不同原生encoder长度允许作为方案成本。保存cold load与warm inference，GPU计时正确同步，报告P50/P95及每帧原始项。

至少分开：I/O、预处理、encoder、全部query/readout、MDN概率、数值结果就绪、绘图、发布、离线归档。网格端到端与station-only分别测，不能用均摊每query时间代替整图。

报告两种范围：

1. `ideal_cutoff_replay`：数据截止即可用、外加延迟为0，仅隔离预测性能。
2. `measured_compute_replay`：实测算法时间+共同明确的队列政策，通信未知，不伪装部署实测。需要网络/触发延迟时另给事前锁定scenario，记录假设，避免重复计算重叠延迟。

默认每事件顺序FIFO模拟：本帧开始=max(该帧数据可得时刻,上一帧完成时刻)，完成=开始+实际服务时长；保存排队量。若处理>1s，不允许把每帧都标为T+单帧时长而忽略积压。latest-only跳帧是单独共同策略，不能只给较慢模型换策略。

按真实/模拟送达顺序计算报警，不按cutoff顺序假设都准时；过时地图是否发布由统一合同决定。独立数组批处理的吞吐不是单事件在线延迟，离线并行不能用于偷换实时能力。

不将模型预测误差已达标与用户已经收到达标地图混为同一个TTUM。shadow和TTUM都共享同一交付时钟模块。

---

## 13. 必须交付的 Bash / Slurm 入口

### 13.1 文件要求

至少交付并实际做shell语法和fake-scheduler测试：

```text
scripts/fe01_a01_compare/prepare.sh
scripts/fe01_a01_compare/submit_stage.sh
scripts/fe01_a01_compare/status.sh
scripts/fe01_a01_compare/resume_failed.sh
scripts/fe01_a01_compare/build_bundle.sh
scripts/fe01_a01_compare/jobs/cpu_stage.sbatch
scripts/fe01_a01_compare/jobs/inference_array.sbatch
scripts/fe01_a01_compare/jobs/render_array.sbatch
configs/fe01_a01_compare/cluster.env.example
```

文件可合理拆分，但要保持下面的命令合同有对应可执行入口，不只在文档写伪命令。

### 13.2 Stage及依赖

| Stage | 资源 | 输入/目的 | 必须前置 |
|---|---|---|---|
| inventory | CPU计算节点 | 定位/核验旧导出、ckpt内部身份、数据/源码/合同 | prepare |
| compare_existing | CPU | 复算已有ON横向和现成字段分层 | 旧导出来源通过；无需等待所有ckpt可用 |
| covariates | CPU数组 | SNR、完整性、时钟、静态几何 | 数据/请求身份 |
| crossing_truth | CPU数组 | 峰值标签和首次越阈、删失状态 | 数据/分量合同 |
| model_smoke | 每任务1DCU | 四ON加载、旧点数值等价、query/cache/未来检查 | checkpoint/source/原依赖通过 |
| replay_pilot | DCU数组 | 6事件固定面板逐秒与小网格验证 | model_smoke、请求/grid合同 |
| replay_stations | DCU数组 | 所有事件逐秒固定真实站点 | pilot通过 |
| replay_grid | DCU数组 | 所有事件固定网格；normal默认，random显式参数 | pilot和grid元数据通过 |
| latency | DCU + CPU | 公共同硬件的station/grid/渲染分项与队列记录 | pilot通过 |
| origin_replay | DCU数组，可选 | 严格origin时刻，保持原历史起点 | clock audit和smoke |
| long_extension | DCU数组，可选 | 已支持模型40/90s | capacity audit和smoke |
| analyze | CPU | 分层、概率、shadow、TTUM、配对汇总 | 对应数据阶段完整，未知范围明确 |
| render | CPU数组 | 所有事件动态地图、等值线、图注和索引 | 数值grid/station产物seal |
| pack | CPU | 轻量证据和完整产物索引 | 选择的必需scope已完成或明确blocked |

默认用户流程含：inventory、compare_existing、covariates、crossing_truth、model_smoke、replay_pilot、replay_stations、replay_grid(normal)、latency、analyze、render、pack。origin_replay、long_extension及random网格全量单独命令，不能自动连跑。

### 13.3 调用形式（Codex必须实现并给出填好可解析环境后的准确命令）

```bash
# 登录节点：只读解析/准备，不加载大模型，不提交
bash scripts/fe01_a01_compare/prepare.sh \
  --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID

# 以下每条只提交指定stage，由用户逐条手动调用
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage inventory
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage compare_existing
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage covariates
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage crossing_truth
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage model_smoke
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage replay_pilot
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage replay_stations
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage replay_grid --geometry normal
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage latency
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage analyze
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage render --geometry normal
bash scripts/fe01_a01_compare/submit_stage.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID --stage pack

# 只读状态，不自动补交
bash scripts/fe01_a01_compare/status.sh --env /path/to/on_eval.private.env --run-id YOUR_NEW_RUN_ID
```

这些是本任务要求的**接口合同**，不是声称上述脚本已存在。最终运行手册须替换为实施后实际可运行命令，说明每一步在哪台机器执行、输入输出、前置状态、资源变量和结果回传位置。

### 13.4 环境、重试和安全

- 用户ckpt已在超算，发现程序只在明确源根/原plan中有界扫描，不递归整个home或按mtime/最低loss猜权重；多个合法候选显式歧义。
- private env只保存本地/超算，提供example；继承已工作的ROCm/DTK/MPI/conda初始化，不擅自重装、module purge后丢依赖或要求登录节点检测GPU。
- prepare/status/import/测试不调用sbatch/srun；用户显式submit才执行。Codex本轮不调用submit。
- 每阶段记录task manifest、源码/环境/合同/ckpt/request SHA、Slurm JobID及真实状态；submitted不等于completed。兼容旧sacct，不能依赖不支持的--array。
- 核验完整seal和输出才标COMPLETE；队列消失不能认定成功。
- 数组任务粒度按模型×事件块×geometry，模型一次载入；设置并发上限防HDF5共享I/O洪峰。
- 失败保留attempt，成功原子发布。显式resume_failed仅补未成功任务，前置hash不变才可复用；不覆盖成功目录、不重复提交运行中任务。
- 允许stage间显式afterok依赖，但不自动推进未知后续阶段。pack可收集失败证据，但总报告不得标全通过。
- 数据/权重只读挂载或路径引用；不要为了新评价复制巨量权重或写进Git。
- 正式计算用全新release和run root，不能覆盖正在跑的A01/EVAL1/FE02源码。

---

## 14. 输出清单与报告形式

至少保留以下机器可读产物，CSV可压缩、Parquet/HDF5需与现有环境兼容，不要求临时安装新依赖：

```text
analysis_contract.json / source_identity.json / source_inventory.csv
checkpoint_inventory.json / request_manifest.* / grid_manifest.*
metadata_coverage.csv / strata_support.csv / clock_audit.csv
request_covariates.* / input_station_covariates.* / snr_quality_audit.csv
threshold_crossing_audit.csv / crossing_truth.*
cross_encoder_metrics.csv / paired_effects.csv / bootstrap_support.csv
interaction_effects.csv / event_robustness.csv / first_second_contributions.csv
station_replay.* / grid_replay.* / replay_support.csv
latency_frames.csv / queue_replay.csv / timing_contract.json
shadow_station_outcomes.* / shadow_event_summary.csv / shadow_pairwise_events.csv
shadow_support_audit.csv / shadow_area_support.csv
map_frame_metrics.csv / map_readiness_contract.json
map_readiness_by_frame.csv / event_time_to_usable_map.csv
event_time_to_stable_usable_map.csv / ttum_pairwise_events.csv / ttum_support.csv
figures/ / maps/event_id/... / animations/ / map_index.html
jobs_manifest.json / status_summary.csv / verification/ / limitations.md
```

站点导出保留raw MDN logits/means/sigmas及精确评分；grid按事件×模型×geometry连续存储，避免每站/每query小文件。保存所有时间帧数值，不只PNG。计算缓存按hash可追溯，原文件不改。

报告正文以“得到支持的优势 / 不占优或不确定条件 / 稀疏与因果/时延限制 / 最小后续证据”组织。主要图：

1. 全范围/第一秒/早期的四ON配对效应；
2. 全预定分层与交互支持图；
3. 四模型逐事件最终PGA动态图、真值站点/误差/区间宽度，统一色标等值线；
4. TTUM CDF/到5/10/20s已稳定可用比例，包含未达标/未知；
5. shadow与误报曲线及真实台站分类地图；
6. 逐事件配对效果、成功/失败/未知四象限和资源代价。

轻量review包包含所有汇总、逐事件/支持统计、合同/身份、测试日志、代表图/动画、完整图源索引和精确执行来源；大grid/全视频可另作full_visuals包留超算。不可只上传README和漂亮图，也不要求全部大预测进入Git。源seal与脱敏包manifest分开验证。

---

## 15. 必须实际执行的测试与验收

### 15.1 身份与兼容

- 旧115及A01认证核心、旧tests/docs清单不变；RT55默认加载/推理聚焦回归不改变。
- wrong epoch、wrong config、wrong encoder、wrong source、wrong request、缺预训练身份均拒绝；TEAM训练encoder与初态区分。
- ON新wrapper对原相同输入输出/完整MDN等价；原selected重叠点复现。
- CSV默认浮点解析漂移测试；round_trip保持人口，不放宽标签容差掩盖错误。

### 15.2 统计/质量

- 主MAE权重、RMSE开方位置、差值方向、同步事件重采样、R²低方差、分位数非可加等测试。
- 空cell/Unknown不补零，统计非独立性和稀疏支持导出。
- SNR的P前补零、合法零、零噪声、信号不足、当前以后脉冲、不同native长度同物理分桶。
- 震中距与最近input距离不能混列；km/degree、JST/UTC和第三坐标单位审计。

### 15.3 模型/回放

- future NaN/Inf/脉冲、最后合法采样正控制、label-only改变；输入统计/特征/输出因果性边界。
- cutoff整数端点；任意query、query顺序/分块/grid追加不改变station输出。
- 同截止缓存一致；跨截止/站点/mask/ckpt复用被拒绝。
- normal/random输入计划共同且可复现；无input、不支持history、真实失败不能静默跳过。

### 15.4 shadow

- L=0、L=tau、正但不足、迟报、截止未报、H为空；deadline超出预测终点为未知。
- 左删失/缺测首次越阈、峰值定义不一致、时间交付乱序、概率非单调。
- 缺文件不等于未报警；目标变input仍在固定面板；恒报警控制暴露误报。
- 平方面积共用空间权重；无真值区不画实测shadow；完整目标波形绝不入模。

### 15.5 TTUM/地图

- 首次通过后失败、连续三帧、缺一帧重置、尾部未确认、全程不达标、首帧达标、无高PGA目标。
- 只有一个query不能认证空间图；N<5场指标NA；固定目标角色变迁不减少主分母。
- 平坦真值的range处理；常数预测即使MAE过线也不自动获得field通过。
- 稳定onset和confirmation不同，回顾性真值使用标识清楚；不能只在成功事件上统计全部事件TTUM。
- same grid/levels/colors、等值线真实坐标、单位、第三坐标、缺测和图层标签；核对输出视频时间戳及图源。
- grid插值与direct站点query评分分列，不能用插值真值给连续域打真值分。

### 15.6 运行与资源

- 所有新增Python编译/import、聚焦pytest；所有bash/sbatch `bash -n`。
- fake scheduler确认prepare/status/test不提交；重复submit/running/success/failed/idempotent resume行为。
- CPU路径不需导入大模型；GPU检查在分配的step中；1DCU smoke在可见设备执行否则NOT_RUN。
- 固定合成小事件端到端生成数值、shadow、TTUM和动画，合成证据醒目标注；不能替代Japan实测。
- 真实产物不可见或缺设备时继续完成可执行模块和脚本，状态NOT_RUN/BLOCKED，不伪造PASS或任务完成。

---

## 16. 完整 [AI-HANDOFF]

[AI-HANDOFF]
task_id: 20261021-fe01-on-integrated-warning-map-v4
repo: rular099/team_pytorch
branch: exp/fe01-on-warning-map-evaluation（若已在独立V1–V3工作区实施则增量沿用并报告）
base_commit: 29ffb20ba0effc0c345a0329ba3f8c658dc8cef0
goal:
  交付四ON模型的可复算横向/分层评价、逐事件shadow、随时间更新PGA图与TTUM，
  包括必要超算补前向和完整分阶段Bash/Slurm脚本；不训练、不访问test。
verified_facts:
  - 源报告全为validation/seed42，原主人口1310事件/194265请求，noninput139440。
  - 四ON selected为DiTing12/PhaseNet12/EQT10/TEAM11，checkpoint身份需在超算重核。
  - 旧导出时刻1/3/5/10/20秒；random query角色迁移，不能直接充当固定全查询回放。
  - 原模型输出最终PGA的MDN，未提供未来首次越阈时间。
  - 已有有限replay包含同截止cache与grid，但只画1/5/20、无动画、高程0为演示。
  - A01源码指纹包含fe01_review等，不能直接修改旧模块后沿用旧认证。
inferences:
  - 极早期指标优势可能改善shadow/成图时间，但须直接评价误报、空间准确度与交付延迟。
  - 分层能界定优势条件，不能自动排除预训练/原生窗口/采样等混杂。
files_to_inspect:
  - 本文件第1.4节列明的入口、源码、结果、配置、launcher、tests、provenance。
constraints:
  - 新行为独立路径；原RT55配置/checkpoint加载/推理和原115/A01认证内容不变。
  - 不重训、不扩大OFF、不重新选轮/调参/校准、不访问test、不自动sbatch/srun。
  - V4允许实现并交付必要新前向，实际提交由用户手动执行。
  - 不覆盖旧实验，不泄露私有路径/密钥，不改旧lock/seal。
proposed_change:
  - fe01_a01_compare/：sources/contracts、统计分层、协变量、时钟/越阈、回放/grid、时延、shadow、TTUM、绘图和封装。
  - scripts/fe01_a01_compare/：prepare、submit_stage、status、resume、独立CPU/DCU/render sbatch与release包。
  - configs/fe01_a01_compare/：固定对照、请求/阈值/地图/TTUM合同和private env示例。
  - tests/test_fe01_on_*与docs/ai/FE01_ON_EVAL_*：聚焦回归和完整手动运行手册。
acceptance_checks:
  - 本文件第15节全部适用检查；不可执行项保留实际状态。
  - 不只交设计：本地可执行代码/合成端到端/可见真实CPU分析确实执行。
  - 主端点复现、固定人群/时钟、完整状态/失败分母、真实图源、未知空间支撑不冒充真值。
  - TTUM首达/稳定起点/确认分离，shadow伴随误报，研究阈值不冒充工程标准。
hpc_followup:
  - 用户按第13节手动提交CPU补数据、checkpoint核验、DCU smoke/pilot、全量站点/网格、latency、统计/render/pack。
  - 提供actual可运行命令与阶段依赖，四ON权重在超算原处只读复用，不要求搬回本地。
risks:
  - 单seed/validation选轮、次要多重探索、少量事件与空cell条件化。
  - 上游因果性、实际触发/通信未知、模型慢导致积压、站点/网格真值不等价。
  - 第三坐标或真实首次越阈定义缺失时部分产品只能带边界或NA，不可伪造。
open_questions:
  - 工程正式误差/报警标准仍未指定；本轮完整研究敏感性可运行，不阻塞实现。
  - 账户/partition/ckpt根等优先只读解析已有私有配置；真正缺项写env待填，不重复询问已确定原则。
  - 用户在全量GPU提交前确认pilot资源量，属于手动执行决定，不要求Codex停止实现。
[/AI-HANDOFF]

## 17. Codex 最终返回

按仓库完整 `[CODEX-RESULT]` 返回：task_id、base_commit、精确result_commit、实际branch、changed_files、真实测试/真实只读分析、compatibility、hpc_status、remaining_risks和review_request。

另附：源与目标身份映射、各stage提交命令、预计任务计数/实测pilot资源、数据缺项、阈值与元数据合同、源码包+SHA、可供ChatGPT审阅的轻量证据。未由用户提交的正式GPU/CPU超算计算写NOT_SUBMITTED，不能用本地合成smoke冒充。

仅提交本任务文件；实际推送并确认后才写已同步。缺生产ckpt不应阻止本地工具、统计和脚本实施。不要只返回新的规划或要求用户把超算checkpoint传回电脑。

---

## 18. 本轮材料来源与状态

本文件合并的V1/V2/V3用户已提供Markdown均已读取；合并输入SHA见包内 `HANDOFF_SOURCES.json`。新增实现计划基于本轮实际读取：上述commit的仓库入口、A01 provenance、FE01 windows、旧replay和多根超算eval手册，以及本对话已核验数据字典/评估链路。

本文件与配套contract/prompt是**实施规范**，不是仓库代码变更、模型补算、已运行的提交脚本或新的科研结果。新阈值网格是建议的研究敏感性合同，必须在执行新比较前记录并冻结，不能宣称此前已经预注册。
