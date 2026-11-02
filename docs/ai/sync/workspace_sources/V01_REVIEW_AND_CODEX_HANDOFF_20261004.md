# V01 首轮结果独立审阅与最小闭环任务书

日期：2026-10-04。任务状态：审阅与设计；未修改仓库、推送代码或提交超算作业。

## 1. 版本与结论

- Repository：`rular099/team_pytorch`
- Branch：`exp/v01-velocity-prep-padding-control`
- 本次实际读取的 HEAD / evidence commit：`74c55aa5442b4200961c88ceee3d11af5033275d`
- 结果整理的父提交：`8829029f7a1a3ef49cedeccef9385298c683ac01`
- 原 V01 实现基线：`9c95dbfaf92f36b2673d816026cd3f25b91eec66`
- 审阅入口：`docs/ai/CHATGPT_REVIEW_REQUEST_20261003_V01.md`

**判断：工程上部分完成，科学上得到有价值但范围有限的反预期结果。不能宣布 V01 已完成，不能认定速度优于加速度，也不能认定历史补零是／不是主要瓶颈。保留全部现有 epoch-8 权重，不重新三臂训练、不重建整套 cache、不再次执行旧 recovery/all，不启动 RT62 或架构修改。**

下一步只有一个主任务：V01 validation closure。查明并修复 normal query-mask 问题、明确 A-pair 空样本与分母、补齐必要既有 provenance，再用冻结权重补缺失评估。

## 2. 本次证据与可见性

已读 AGENTS、PROJECT_CONTEXT、协作 README、SESSION_SUMMARY、原 V01 prompt/规范、本轮 request/result/review、training/checkpoint/NPZ/dose audits、配对 CI、指标分层，以及 generator、缓存构建、evaluator、launcher、结果分析器和针对性测试。

独立执行的计算：
1. 根据已提交精确汇总值复算 MSE = bias² + centered error variance，以及阈值分层 MAE 的加权分解。
2. 用与仓库相同的 NumPy `asarray` + `&=` 逻辑，复现 normal caller mask 被修改、random caller 临时 mask 被修改而正式 target mask 不受影响的差异。

**没有真实 checkpoint/HDF5/NPZ；没有复跑正式 inference、训练或 5000-draw bootstrap。合成别名检查不是对真实 idx=7 的端到端复现。** 同目录 `independent_checks.py`、`independent_checks.json` 保留可执行计算。

## 3. 当前结果的正确解释

固定 seed42、最终 epoch8，random validation，7 个时刻 1/3/5/10/20/40/90 s；四个速度格共 1194 events、8358 event/time rows、84259 target observations，全部为 sampler non-input。早期 1/3/5 s 合并为 36111 observations、1194 events。PGA 坐标 log10(m/s²)，点估计为该坐标内 MDN mixture mean。

命名：第一字母为训练臂，第二字母为推理视图，F=完整前缀，M=人工缺失。

| 指标 | FF | FM | MF | MM |
|---|---:|---:|---:|---:|
| MAE | 0.244516 | 0.244091 | 0.236893 | 0.236895 |
| RMSE | 0.310286 | 0.309952 | 0.303735 | 0.303782 |
| NLL | 0.171931 | 0.170561 | 0.142036 | 0.141841 |
| Brier，阈值 -1.2 | 0.202340 | 0.201869 | 0.194258 | 0.194188 |

MM−FF MAE=-0.007620，配对事件聚类 95% CI [-0.008285,-0.006954]；早期=-0.008716，CI [-0.009655,-0.007787]。但固定完整视图换模型 MF−FF=-0.007622，而固定缺失训练模型换视图 MM−MF=+0.000001888，CI [-0.000187,+0.000199]。故大差异与训练臂相联系，不是删除前缀的直接推理收益。训练臂的因果归因还受单 seed、未回传训练计划／运行身份等限制；不能宣称已证明正则化机制。

### 3.1 总体 MSE 改善主要是偏差减少，不是误差云收紧

终端公式：

```text
e_i = prediction_i - truth_i
bias = mean(e)
MSE = mean(e²) = bias² + mean((e - bias)²)
```

原始 LaTeX：

```latex
e_i=\hat y_i-y_i,\qquad b=\frac1N\sum_i e_i,\qquad
\operatorname{MSE}=b^2+\frac1N\sum_i(e_i-b)^2.
```

由 `metrics_by_time_and_group.csv` 的精确值计算：

| 量 | FF | MM |
|---|---:|---:|
| MSE，dex² | 0.0962773935 | 0.0922836408 |
| bias²，dex² | 0.0083839818 | 0.0043255825 |
| 去均值误差方差，dex² | 0.0878934117 | 0.0879580583 |
| 去均值误差标准差，dex | 0.2964682306 | 0.2965772383 |

MSE 下降 0.0039937528，bias² 下降 0.0040583993，后者略大于前者。这个恒等式是样本上的描述性分解，不是训练机制因果解释，也不能用来擅自给 validation 预测减去 bias。

### 3.2 较大 PGA 目标并没有同步改善

同一 random/epoch8/all7 cohort：
- y >= -1.2：35612 observations、1051 contributing events；MAE 0.219369→0.228564（+4.19%）；RMSE 0.296178→0.307913；bias -0.117053→-0.142999；NLL 0.130237→0.171732；Brier 0.130815→0.149857。
- y < -1.2：48647 observations、1158 contributing events；MAE 0.262924→0.242994（-7.58%）。

这是按目标 PGA 而不是震级的分组，不称作大震／高烈度事件。当前分层没有专门 paired CI；不能把点估计方向称作已确证的总体退化。下一轮只从既有 NPZ 补充该层 CI，不调阈值或选模型。

### 3.3 实际处理不是“完全没有 P 前记录”

38302 次选中 source observations 中，21380 次发生删除；平均可见 P 前支持 9.091→6.089 s，删除中位数 0.970 s，P90 8.700 s。M 视图只有15次 source observation 的 P 前有效时长为零。当前结果不能外推到全面零前缀、P 初至缺失、P 后数据丢失或其它人工剂量。

### 3.4 空间指标要统一定义，不能跨版本误比

本次 `tools/summarize_v01_results.py::field_metrics` 用的是 **max−min / max−min（np.ptp）的 ratio，再对 fields 取中位数**，不是 RT60/61 的 P95−P05 平均比值。现有值要保留原名字并标注定义。

实际单输入、每场至少5个query：1514 fields、781 events；FF/MM ratio median 0.006364/0.006350，pairwise-delta MAE 0.354286/0.354288。绝对单站点误差降低不等于空间形状改善。此处只是复用 RT55 架构的数据实验，不在下一轮引入空间架构修复。

## 4. normal 失败：优先验证的具体代码机制

### 4.1 已核验代码事实

`gemini_util_light.py::_select_realtime_cutout`：

```python
valid = np.asarray(station_valid_full[0], dtype=bool)
valid &= self._realtime_pick_valid(picks)
```

输入本身是 bool ndarray 时，`np.asarray` 通常共享底层存储；`&=` 原地更新 caller 的 mask。

- 纯 normal config 将 `causal_random_input_mask.enabled` 设为 false，没有前置 cutout。后面的调用直接传正式 `station_valid_full`。
- random 配置，以及 mixed 训练的 enabled=true 路径，在更早阶段传临时 `full_coord_valid` 计算 cutout；无论这次随机 mask 是否最终 applied，只要 enabled 就经过这个预先计算。
- 后续 PGA 候选是 `finite(pga) & station_valid_full`。一个 query 的 PGA/坐标合法，并不要求其 P pick 落在 source waveform window 内；query pick 可以为负、晚于窗口或无法用于时钟。
- `_get_one` 2191 行对空候选报 `Found event without PGA idx=7`。缓存 builder 独立存 source 与 query，source PGA=NaN，query 保留实测 PGA 及变换后的 P sample。

### 4.2 独立合成复现

用1个source（pick1000、PGA=NaN）和2个query（pick -50及10100，有限PGA），10000点窗口：

```text
原正式 validity       [True, True, True]
normal旧时钟调用后    [True, False, False] -> query候选0
random临时时钟路径    正式mask未改         -> query候选2
对局部valid先copy      正式mask未改         -> query候选2
截止时刻保持相同
```

这是确定的代码副作用和可重现实例。**实际失败 event 是否恰好满足该条件、上传源码是否与Git相同，仍需真实cache行和runtime身份确认。** 不应直接宣布 idx=7 的唯一根因已完全认证。

### 4.3 最小修复方向

只在新的显式 V01 validation flag 下，对计算时钟的 mask 作独立副本，分离：
- source/clock eligibility；
- query-label/coordinate eligibility；
- input station validity；
- target triggered/untriggered 的评估分类。

不应让查询点是否可预测取决于其自身波形是否存在、是否已经触发或其pick是否在输入窗内。不应把未知pick强行改成0或触发状态。

RT55/旧路径默认保持原状；本次不顺带全局修复历史数值。若真实根因不是别名，则以真实 trace 为准，但也要保留该别名反例及无副作用单元测试。

## 5. A-pair 重复和分母

A-pair目前8358 rows、8313唯一event/time、1193events、84329target observations。重复是45条额外导出行，**不等于已证实45个独立坏事件**。`__getitem__` 捕获 `_EmptySample` 后取下一index，会产生静默替代，这与现象一致但仍需逐条映射。

要求建立不可变的 requested sample ledger，每个请求恰有一个 outcome：
- predicted；
- no_available_source（无可观测输入，明确弃权）；
- invalid_label_or_metadata（数据错误／不合格）；
- implementation_error（不能当正常跳过）。

同时保留 requested event/time/plan ID、actual event/time、source/query IDs、绝对截止、row selector、reason。评估不得用邻近事件替代当前请求；不得为填满计数复制输出。

默认优先离线恢复旧AA：只有在重复条目的全部预测/标签/输入身份/截止均一致，且确证它们是同一实际请求重复时，才保留一份并发布去重日志与缺失清单。否则不做静默去重。真正无可用输入的请求保留在coverage分母，不强行喂全零假台站。

A/V比较同时给出：各自可用群体、共同可评估event/query/absolute-cutoff交集、共同remote分组、无预测率和丢失事件/目标数。共同交集指标只能描述“共同可评估条件下”的比较，不能掩盖A系统弃权或冒充相同传感器/相同几何的模态因果实验。

## 6. 缺口优先级与停止条件

### P0：从现有运行目录回传，不重新preflight/解码全量数据

- preflight_summary.json、protocol_lock.json、cohort计数及排除清单。
- 原split manifest的可审阅train/dev子集及完整原文件hash；与cache各事件split一致性、train/dev互斥。
- 三臂最终checkpoint hash/epoch/loss/update；parent文件/encoder文件与加载tensor身份；trainable名单、world size、batch/accumulation、实际训练采样计划或其证据。
- 当时记录的uploaded-source manifest/config identity；当前重新计算hash不能假装证明历史运行源码。
- derived cache的schema/单位/response attrs、source/query传感器映射及源文件hash。可通过小JSON/CSV/sidecar补全，无需raw weights/NPZ入Git。
- 失败index映射与最小trace、A-pair重复/遗漏清单。

已有材料不存在时写 NOT_AVAILABLE，不因缺一个旧日志就自动重训；但相应科学结论降级为条件性描述。

### P1：纯离线补统计

- 保留原表，新增 corrected/closure version，不覆盖旧报告。
- 分层 paired CI：阈值上下、untriggered、单输入、early135/all7的MAE/RMSE/NLL/Brier；缺样本的组写N/A。
- target-micro与event-macro，5000event-cluster draws，seed20260915；不改阈值、不做选择。
- FF/MM MSE/bias²/centered-error分解；做CI需从原NPZ按event重采样，不能从总体summary伪造。
- 原np.ptp median继续报告，另增明确命名的P95−P05指标仅作口径桥接；不把它替换为事后gate。
- 固定axes的散点密度、分层残差图和图源表，解释正负bias；不得用validation bias校正预测。

### P2：缺失评估，默认5个normal、0个新训练

优先诊断修复后，用同一既有epoch8 checkpoint补FF/FM/MF/MM/AA五个normal。新目录、显式协议版本。

四个速度random原则上复用；用针对性软件fixture/记录签名证明本次补丁未影响其采样/输入/标签路径。需要传感器ID时优先从旧NPZ original indices与cache构建sidecar，不重跑四次encoder。

AA random只在旧结果不能可靠恢复、存在被错误丢掉但可恢复的请求、或必须重新导出请求身份时补跑1次。真实无信号无需重跑“强行凑齐”。若修复影响random数值，不混用新旧：报告影响范围，由用户决定相应受影响格的额外重放。

若新五格仍错误、split错位、缓存/单位身份冲突或训练实际协议严重偏离：不自动扩大到新训练，先返回证据和最小后续决策。

## 7. 仍然保留的科学边界

- mask正确并不证明编码器不受缺失影响；当前干预是“删除真实前缀+合法mask”的联合变化。
- 当前代码仍包含全记录中心化与`:cutout+1`去均值路径，不能仅凭新metadata-support字段宣称严格在线因果性已认证。此轮不悄悄修改这些数值再混用原结果；继承问题单列在limitations，必要时用小例子记录，不恢复整套RT62工作。
- 原始post-P波形等同性应该在单位转换后、干预前/后、归一化前核验；不同支持下归一化后post-P数值不同不自动意味着样本被错误修改。
- 可以接受不支持原先假设的结果。不得为了获得“补零有害”而加大剂量、换阈值、换checkpoint或挑seed。
- 如果补齐normal后仍是训练臂总体偏差改善而非完整前缀收益，则本轮可按限定条件的负机制结果收尾；不能靠不断扩实验逃避结论。

## 8. 可直接交给 Codex 的完整交接

```text
[AI-HANDOFF]
task_id: 20261004-v01-validation-closure
repo: rular099/team_pytorch
branch: exp/v01-velocity-prep-padding-control
base_commit: 74c55aa5442b4200961c88ceee3d11af5033275d

goal:
  保留现有三臂epoch8权重，最小化修复并闭合V01评估协议：
  normal query-mask根因、A-pair空样本/分母、必需provenance和5个normal结果。

verified_facts:
  - evidence同base；三臂checkpoint审计为epoch8、optimizer steps1496。
  - 保存191张量init相同；704个encoder张量未包含在权重包。
  - 四格速度random：1194events/8358rows/84259target observations。
  - 五个normal同报Found event without PGA idx=7。
  - AA random有45条额外重复，不能直接做A/V公平比较。
  - _select_realtime_cutout的np.asarray(bool)加&=有caller mask别名副作用。
  - normal直接传正式station_valid_full；random/mixed-enabled走临时mask前置时钟。
  - MM总体改善伴随bias下降；阈值以上目标的MAE/NLL/Brier点估计变差。

inferences:
  - 别名造成窗外query失效是normal失败的可执行候选根因，
    但实际idx7仍需cache/runtime追踪。
  - AA重复与_EmptySample替代一致，不能凭45推断45个坏事件。
  - 当前证据不支持“P前缺失明显损伤”主假设，不证明普遍无害或正则化机制。

files_to_inspect:
  - AGENTS.md、docs/ai/PROJECT_CONTEXT.md、docs/ai/README.md、SESSION_SUMMARY.md
  - docs/ai/CHATGPT_REVIEW_REQUEST_20261003_V01.md
  - docs/ai/CODEX_RESULT_20261003_V01_PARTIAL_VALIDATION.md
  - docs/ai/V01_prompt.md和原V01完整规范
  - reports/v01_velocity_padding_validation_20261003/*及source_evidence
  - gemini_util_light.py::_select_realtime_cutout/_get_one/__getitem__/JointGenerator
  - tools/build_v01_paired_manifest.py
  - eval_checkpoint.py::build_datasets/run_inference
  - tests/test_v01_padding_controls.py、tests/test_v01_recovery_launchers.py
  - tools/summarize_v01_results.py和现有Slurm入口

constraints:
  - 核验HEAD差异和用户未跟踪文件，保留全部旧结果及权重。
  - 不改gemini_models.py/DiTing/loss/标签/模型结构，不改变训练权重。
  - RT55-RT61旧默认路径不变；新行为仅显式V01 validation opt-in。
  - 不再次all/recover/preflight，不重建cache、不重训、不加epoch/seed/sweep。
  - 不恢复RT62，不访问held-out test波形或metrics。
  - 不把ValueError改成无条件skip，不用后续样本代替当前请求。
  - 不为使指标变好选择缺失剂量、阈值、checkpoint或bias correction。
  - 不新增硬编码私有路径；新产物由环境变量/CLI指定。

proposed_change:
  1. 新增tools/audit_v01_validation_contract.py（或同等职责最小工具）：
     读取现有cache和回传文件，不重做materialization。
     建立request ledger；追踪idx7到shard/event/row_selector。
     记录query标签/坐标/P sample、source_role和clock前后mask。
     构建轻量sensor-id/provenance sidecar，回传实际train/dev计数。
  2. gemini_util_light.py：
     先用实际生产类复现mask别名；若确认，给V01 opt-in路径copy局部时钟mask，
     分离query有效性与clock/input eligibility，不改变截止定义。
     旧路径保留；新评估禁止_EmptySample邻样本替代。
     只有明确的data/availability outcome可记录无预测，真正实现错误仍fail。
  3. eval_checkpoint.py：
     最小接入显式requested/actual身份、skip ledger、真实sensorIDs和绝对cutoff。
     需要时采用明确evaluation iterator，不能让JointGenerator丢失request身份。
     每个请求恰有一个outcome；模型仅处理合法有输入样本。
  4. pga_configs新增V01 closure eval配置或等价override：
     不覆盖旧V01配置。只开启目标mask隔离/显式空样本合同。
  5. tools/summarize_v01_results.py或新增closure分析器：
     保留旧报告；新增完整/共同群体/弃权率、分层CI、误差分解、图源。
     pairing必须一对一；覆盖率和unmatched显式统计，不用inner join静默丢目标。
     标清np.ptp中位数与P95−P05口径，不改旧gate。
  6. 新增tests/test_v01_validation_closure.py：
     真实PreloadedEventGenerator的clock不修改输入mask；
     negative/0/late/unknown query picks仍可有合法PGA；query-only不能入输入；
     source正常但全部query窗外仍可预测；normal/mixed/random各路径覆盖；
     _EmptySample不替代，request ledger计数可对账；legacy行为回归。
  7. 新增tools/complete_v01_validation_slurm.sh：
     evaluation-only，默认5normal，条件性1AA random；无training/preflight分支。
     DRY_RUN默认1、显式CONFIRM、唯一目录、依赖ID和源码manifest记录。
     登录节点只需轻量依赖，epoch/hash核验在适当环境执行。

acceptance_checks:
  - 用真实idx7及生产generator定位，提交修复前失败/后成功的trace。
  - 合成反例仅证明机制；不能替代实际行验证。
  - normal clock不改变query-validity；合法query不由自身波形可见性决定。
  - 旧RT55/RT56 config/state_dict/inference路径与mixed训练路径兼容。
  - 既有random未受本次patch影响的证据；若受影响不得混报。
  - 三臂epoch8、step1496及hash匹配；补全缺失encoder/parent/split身份或标未知。
  - 从现有文件回传train/dev计数、protocol lock、source identities；
    不把历史manifest或当前hash冒称实际旧运行证据。
  - 每个request有且仅有一个预测/弃权/错误结果，45重复逐条映射。
  - 仅在逐字段一致性证实后去重旧AA；保留原文件和缺失分母。
  - 分别报告五格normal真实counts，不强制套历史89770/26119。
  - 共同query/cutoff A/V比较明确是不同输入系统的桥接，另报coverage/remote。
  - above-threshold、untriggered、single-station及early/all7点/概率分层CI；
    5000 event-cluster draws，seed20260915，不据此调实验。
  - shell bash-n、Python compile和针对性/既有回归tests真实执行。

hpc_followup:
  - 不自动提交，交付用户可执行脚本及完整dry-run/正式命令。
  - 默认复用：weights_vfull_retry1、weights_vmissing、weights_apair的epoch8 last。
  - 三个last文件SHA依次为：
    09c65503ff6e7d848899081110142487c0622f1cc5fd143df3579d55e38f4763
    9173d8d60d657b6b6eebb6a1dedef6c4af139319328c787c233187b2acde3b8a
    a2837d575e4a81e6386e2cb004669a3b0052e9ab093aa3893fcda7b76c74cb32
  - 默认补FF/FM/MF/MM/AA normal五格；旧四格速度random直接复用。
  - AA random先离线修复；不能可靠恢复才增加一格eval，写出理由。
  - 不覆盖eval_retry1，不再次调用整体recovery/all。
  - GPU资源沿现有单次eval；不为本任务申请训练DDP资源。

risks:
  - normal修复可能改变旧误过滤query群体，必须新协议版本和分母说明。
  - 实际训练cohort/encoder运行身份缺失，same-init证明目前仅限保存张量。
  - 相同seed和step并不证明完全相同实际训练样本，尤其含空样本替代。
  - 共同可评估交集有选择偏差，必须伴随弃权率与缺失分层。
  - 当前去均值实时依赖疑问不在本轮偷偷改动，不能宣称完整在线认证。

open_questions:
  - none for bounded implementation。
  - 只有发现必须新增训练或改变数值协议才报告新决策，不自行扩大范围。
[/AI-HANDOFF]

完成后按docs/ai/README.md返回[CODEX-RESULT]：精确result commit、文件清单、
真实测试结果、legacy兼容性、HPC状态、还缺什么，以及最小后续审阅入口。
没有执行的步骤标NOT RUN；不要把代码已实现与结果已完成混写。
```

## 9. 预期收尾条件

工程闭环：5个normal取得合法结果或明确可审计的无输入结果；不再静默替代；A-pair恢复或缺失分母明确；核心数据与运行身份可追踪；原随机四格在限定协议下可重算。

科学闭环：报告是否支持“当前剂量的P前缺失存在显著/实用代价”，并明确normal、阈值以上目标、单站空间性能的反例。不需要所有指标改善才算实验完成；不能得到所期待正结果也可以形成可信的限定性负结论。

若缺失代价仍很小，停止继续通过加大剂量/扫参寻找想要结论。后续模型精度研究另立任务，不能借本轮normal故障继续叠加模型变化。
