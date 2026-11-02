# FE01-A01：查询几何诊断与绝对振幅消融

## 固定身份与工作边界

- Repository：`rular099/team_pytorch`。
- 本次实际读取的分支：`exp/fe01-feature-extractor-site-effects`。
- Base commit：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`。
- 建议新工作分支：`exp/fe01-a01-absolute-amplitude-ablation`，使用独立 worktree；不在当前 FE01-EVAL1 工作目录实施新训练改动。
- 用户当前要求：解释单台空间输出近常数、多台可出现空间动态范围的原因；核对坐标与振幅输入；设计各提取器去除绝对振幅的对照；确认振幅只来自决策时刻可得数据。
- 本任务不等于放弃 FE01-EVAL1：TEAM loader gate 修复和已有系统同人口复评继续走原工作流，避免与另一个 Codex 会话同时修改文件。
- 本任务生成工具、测试、配置和手动提交脚本，不自动训练、sbatch/srun、下载、调参或开启 test。

## 一、已核验事实：不要从文件名或旧注释推断活动路径

1. 当前 FE01 resolved config：`use_coords_abs=true`、`use_coords_rel=false`、`coord_fusion_mode=add`、`station_context_mode=off`、`pga_readout_mode=target_cross_attention`、`pga_readout_layers=4`、`readout_first_residual=true`、`pga_use_event_context=false`、`use_rope=false`、`use_vs30=false`、`use_amplitude_info=true`、`disable_waveform_scale=false`。
2. `gemini_models.PositionEmbedding` 对三维地理坐标使用固定多频率 sin/cos 编码，不是台站序号位置编码。`_station_coord_features` / `_pga_coord_embedding` 的绝对坐标分支直接返回绝对编码；该分支不使用 `coords_abs_weight=0.1` 的相对/绝对混合权重。
3. `FullModel.forward`：前端 station embedding 经 LayerNorm，加振幅投影，再加输入站坐标编码；查询向量为查询站坐标编码加 learned query token。
4. `_station_memory_for_readout` 在 `station_context_mode=off` 时直接返回 station features；当前 PGA 路径不调用台站 self-attention Transformer。事件定位/震级辅助头并不以其预测坐标/震级直接反馈 PGA。
5. `CrossAttentionReadout` 仅使用有效 station tokens 作为 keys/values；一个有效 key 时，每头注意力权重为 1，纯 attention 输出与 query 无关。实际配置保留 query residual，后续也有 residual/FFN，因此不能断言整个模型数学上被迫恒定。该类旧 docstring 关于“不保留 query residual”不适用于当前配置。
6. `fe01/model.py::shape_and_scale` 使用 received/storage mask，先求可见样本统计。前端使用每台三分量共同峰值归一化；旁路为 3 个 log10 std、3 个 log10 RMS、3 个 log10 peak、1 个 log10 station peak、1 个 log1p 有效秒数，共 11 维。
7. FE01 数据由 `read_event` 先读到 exclusive cutoff，`build_window` 再根据公共 prefix 与原生容量构造右对齐窗口；整个事件最终 PGA 作为标签读取，不属于振幅旁路输入。
8. 已存在 3 个家族×3 seed 的 amplitude-on FE01 训练；当前没有新训同公共下游的 DiTing 对照。RT55/RT61 是已有完整系统，不得当成这种 DiTing 训练对照。
9. 源头离线滤波/重采样的在线因果性未认证；对已处理 HDF5 的 prefix 测试不能证明原始采集到 HDF5 的整个链路无未来依赖。

事实入口：
- `reports/fe01_hpc_results_20261004/metadata/formal__phasenet_pretrained_frozen__seed42.json`，并核验 TEAM/EQT 的对应 resolved config。
- `gemini_models.py`：PositionEmbedding、CrossAttentionReadout、CrossAttentionRefinementBlock、FullModel.forward、_station_memory_for_readout、_station_coord_features、_pga_coord_embedding。
- `fe01/model.py`、`fe01/data.py`、`fe01/windows.py`、`fe01/extractors.py`、`fe01/engine.py`、`fe01/metrics.py`。
- `fe01_review/runners.py`、`reports/fe01_eval1_hpc_gates_20261006/`。

## 二、本轮科学判断与禁止混淆

- “输入/查询台站位置已知”与“震源位置可从波形推断”是不同问题。空间动态图可以来自 query 对带坐标/振幅的站点特征加权，不要求先完成震源定位。
- 单 key 的注意力选择通道退化是可推导事实；query 残差弱、幅值分支占优、地理编码尺度不适配、损失/样本覆盖或信息不足是仍需验证的假设。
- 多台输出范围更大不等于范围正确；需联合真值范围、shape error、MAE/RMSE、NLL/coverage。
- 只增加 key/query 打分的位置编码、RoPE、相对距离 bias 或头数，不能单独消除一个有效 key 的 softmax 退化。
- 归一化输入去除了每台总体物理倍数，不等于删除波形包络、分量比、SNR 或频谱中的统计线索。
- 推理时把幅值旁路关掉是已有模型的依赖诊断；不能代替从训练起去幅值的能力对照。
- 不能把“旧 DiTing 震级任务曾不显式输入幅值”直接推广成“PGA 不需要物理幅值”。该旧任务的实际 config/checkpoint/消融未在本轮重新核验。

## 三、阶段 A：不训练的几何与因果诊断

### A1. 查询坐标的信息流

从当前固定 validation 请求中，依据可见 input_count、目标数和截止时间预选诊断样本，不按误差选择。首轮每个 input_count 分层最多 32 个 decision，尽可能覆盖共同 1/3/5/10/20 秒；缺项记录实际 N，不补造。正式指标继续使用全 validation，探针只用于机制检查。

- 至少覆盖真实 K=1 与 K>=2；K 是有效 key 数，不是 25 个 padding slots。
- 固定 event/T/波形/幅值/masks，只让实际查询坐标变化。导出坐标编码、attention 纯输出、query residual 后输出、每层读出、最终 MDN（weights/mu/sigma）的跨 query 变化。
- hook 仅在新诊断 wrapper 生效；不改原 state_dict 或 buffer。梯度探针可在 eval 模式开放输入梯度，但不执行 backward 更新/optimizer。
- 量化 `d output / d query_coord`，并用以公里为尺度的有限差分复核，防止“度制输入梯度大小”被误读。共同输入站坐标、waveform/scale branch 也做分路径敏感性。
- 输出实际 checkpoint 中的 first-residual/FFN/attention/scale gates、各分支特征范数、query residual 到 MDN 的梯度。初始化值不等于训练后值。
- 检查经纬度偏移和第三坐标单位；没有真实 HDF5/provenance 不宣称米/公里正确。
- 诊断坐标置换须仅作机制干预，不作为真实地理预测分数。对 station permutation 使用波形/坐标/mask 一起置换的等价控制。

可证伪预期：K=1 纯 attention 对 query 的梯度为零，但 residual 路径不必为零；若 PE 差异正常而 readout/MDN 的差异快速消失，支持下游抑制假设；若 PE 本身恒定或量纲错误，优先排查坐标路径。单 key 相同 values 的复制仅作合成模块测试，不能虚构重复台站进入真实评分。

### A2. 实际 K 的公平控制

需要评估 K 的因果差别时，固定同一 event/T，从可用台站按预先稳定的优先级构造嵌套 K=1/3/5/8 子集；查询排除最大输入集合，保证同一对照的 target population 不变。按每组共同可用人口报告 events/targets，不把不同 K 的原始子群直接视作配对。原生长度/截止保持不变。不要为填满 8 站把事件时刻往后挪。

### A3. 幅值可得性与未来污染

- 正常路径读取并断言最后允许索引 n_T，所有统计与 encoder 只依赖 n<=n_T，exclusive cutoff=n_T+1。
- 从原始记录的内存副本改变 n_T+1 及其后波形为大脉冲/NaN/Inf，保持 labels、coords、picks、allowed masks、metadata、请求和随机种子固定。
- 逐项比较 selected IDs、received mask、前缀均值/方差/归一化 peak、10个幅值统计、有效秒数、encoder、MDN、概率输出。
- 包含触发边界、原生容量边界、尾部缺测、内部合法零、全空 invalid station，不能仅测一个 1 秒样本。
- 作为正控制，仅增大最后一个合法样本，允许统计/输出发生变化；不要因果 mask 把合法末点也删除。
- 不使用整条事件的峰值归一化，不先编码完整事件再截 token，不让跨 T cache 混用。
- 记录 upstream 原始滤波/重采样流程的可见性。有原始波形才做“先改未来再预处理”的端到端检验；否则保留 UNKNOWN，不扩大工作去转换全数据。

## 四、阶段 B：各提取器“无绝对振幅”训练对照

### B1. 唯一主要变量

新增显式模式（名称可调整，语义必须一致）：

```
absolute_amplitude_mode: physical_scale_keep_duration | no_absolute_scale_keep_duration
```

- ON：完全保留原 FE01 的 11 维旁路。
- OFF：前10维固定为0，最后的 log1p(valid_seconds) 保留，11维 shape、projection、gate 与公共下游结构不变。
- 不直接用 `use_amplitude_info=false` 作为主要 OFF，因为这会删除整个旁路，连有效时长一起去掉。
- 前端收到的归一化波形在 ON/OFF 中必须完全相同；不改变 joint/per-component normalization、不改变频段、时长、采样或损失。
- 保留每台三分量共同归一化，以免额外删除分量比例/极化信息。称为“无显式绝对物理振幅”，不称为“完全没有幅值形状线索”。
- 原普通/随机 geometry、target sampling、cutoff、seed、辅助震级/位置 loss 均不变；真实最终 PGA 仍只作监督，不改变为当前峰值。
- FE01 Base 中关闭其他 raw-energy/DPK/temporal/anchor 路径；若诊断 RT61，则另行追踪这些旁路，不能认为一个布尔开关遮断所有幅值来源。

### B2. 尺度不变性验收

固定 metadata、mask、station selection 与时钟，对每个输入站分别乘正数 c_s（每站三分量共用同一个倍数）。例如 c_s 在 0.1/1/10 中选择，避免数值 floor 饱和；另测整事件统一倍数。

- normalized waveform 应在数值容差内相同。
- OFF 模型的所有 delivered features 和最终 MDN 应在容差内不变。
- ON 的10维 log10幅值统计应加 log10(c_s)，duration不变。
- ON 最终预测是否敏感是模型依赖诊断，不能要求任意未训练随机模型必须变化。
- 不把输入站任意增益干预当成物理地震事件，再强行要求未观测 query PGA 标签同步变化。尺度干预用于旁路验证，不用于真实评价评分。

### B3. 实验矩阵与预算

支持 TEAM original、PhaseNet、EQT、DiTing 四种 frontend。PhaseNet/EQT/DiTing 冻结原预训练 encoder，训练原 adapter 与公共下游；TEAM 使用原 scratch 策略。

首轮只准备 seed42 的配对版本：
- TEAM/PhaseNet/EQT：各新增1个 OFF 训练；ON 尽量复用原 FE01 seed42，但必须验证新的 ON 路径、共同初始化、数据/采样/loss/训练预算与旧版一致。
- DiTing：当前没有 FE01 公共下游的匹配 ON，严格幅值消融须新建 ON 与 OFF 一对；不拿 RT55/RT61 作这对的 ON。
- 在上述复用条件通过时，首轮新增训练数为5，而非8；如果条件不通过，报告原因，不能悄悄重跑更多任务。
- 若用户暂不投入 DiTing 配对训练，仍准备其配置但标记 NOT_RUN，先执行另外三种，不填伪造 DiTing 列。
- 不默认扩展至seed43/44；是否扩展由首轮证据与预算决定。

每家族 ON/OFF 从相同公共初始权重、相同原 encoder、相同初始 adapter 起点训练，不从 ON 的最终 checkpoint 微调成 OFF。读取 init.pth 有助于精确重放；缺失时重建并验证初始内容 SHA，不猜测。

预算沿用 FE01：12 epochs、每事件3 draws、global batch128；当前 cohort9084事件时每epoch212 updates，总2544 updates。各家族原生时间支持范围可不同，但同一家族 ON/OFF 的每个 event/epoch/draw 计划必须相同。不要为了本次测试改成新的时间分布。唯一允许的实验分歧是 amplitude policy。

checkpoint selection 沿用原 validation 十cell noninput MAE、同分较早epoch；另报固定最后epoch12，以区分选epoch影响。禁止 test 选择模型/阈值/超参。

### B4. 评价协议

- 主人口：共同 validation 1310 events；正常情况下每run194265 all-target记录、139440 noninput记录。必须重新核验key/label/request指纹，不机械填数。
- 主指标：1/3/5/10/20秒 × normal/random 十cell等权 noninput MAE，log10(m/s²)。
- 联合报告 all/noninput/untriggered、actual K=1 与多台、MAE/RMSE/R²/bias/slope、精确 MDN NLL/CRPS/Brier、sigma、分位数coverage与width。
- 分开报告 level MSE、shape MSE、P95−P05 range recovery、近等距离站对误差；不因动态范围变大就判定改善。
- 事件簇bootstrap保留所有站/时刻，seed变异另列；本轮单seed不是多seed最终结论。
- 新增 ON/OFF 的局部幅值与空间效果对照，不把某模型在更短时间域上的平均值与另一模型完整90秒域平均值直接排名。
- ON/OFF trainable参数结构相同，但 OFF 前10个投影输入列无数据梯度是预期，不伪称有效容量完全相同。

解释规则：
- OFF若更差，不否定其波形形状特征有效；可能说明物理绝对尺度不可缺少。
- OFF若level更差而shape不变/更好，应分别陈述，不能用单一主MAE掩盖机制。
- ON若在幅值only控制之上有增益才直接支持额外形状贡献；本轮不默认增加新的amplitude-only训练。
- 某提取器OFF优于ON只证明当前协议下的优化/泛化现象，不代表物理振幅没有信息。

## 五、兼容性、交付和手动超算

- 不改既有115文件训练快照，不覆盖旧run、lock、checkpoint和当前EVAL1审计证据。
- 新代码优先放新包 `fe01_a01/`、`scripts/fe01_a01/`、`configs/fe01_a01/`、`tests/test_fe01_a01_*.py`。允许复用原函数，但不得用全局monkeypatch让正在执行的EVAL1模型发生变化。
- 如新训练不得不改训练入口，放独立新入口，生成自己的源码及依赖指纹；保留原训练语义，新增小型训练循环等价测试，不能重签旧lock绕过检查。
- 输出分为 diagnostics、amplitude_on、amplitude_off；推理关闭旁路结果标 `inference_intervention`，独立训练OFF标 `trained_ablation`。
- 超算提交脚本必须由用户手动调用；准备脚本只打印命令。保持已有可工作DCU/ROCm环境，不触发旧MPI恢复或重装torch。
- 交付可逐stage执行的环境检查、权重/数据身份检查、diagnostic、train、eval与轻量pack脚本；无自动test入口，无破坏性覆盖。
- 新source commit变化时重新核验对应诊断gate；旧PASS只证明旧source。不能让新训练任务误读EVAL1旧封存SHA。
- 必须记录硬件/软件版本、实际events/targets、checkpoint内部epoch/SHA、resolved config、encoder与公共初始化SHA、mask/cutoff/request/采样计划指纹、训练曲线、梯度/gate诊断和逐目标MDN。

## 六、AI-HANDOFF

```text
[AI-HANDOFF]
task_id: 20261007-fe01-a01-geometry-absolute-amplitude
repo: rular099/team_pytorch
branch: exp/fe01-a01-absolute-amplitude-ablation (proposed new worktree branch)
base_commit: ba4fa7740d9fde5fde87a7b0ae0397037209baf4
goal: 追踪查询坐标/幅值信息流，验证当前前缀幅值因果性，并交付四种frontend的绝对幅值ON/OFF受控训练方案与手动超算脚本。
verified_facts:
  - 当前FE01为地理sin/cos坐标加法、station_context off、query cross-attention且保留query residual。
  - 单有效key的attention权重恒为1，但不能据此断言整个带残差模型必须恒定。
  - 现有旁路11维中10维是绝对幅值，1维是有效时长，use_amplitude_info当前为true。
  - FE01正常路径先截前缀再求幅值；上游offline处理未认证。
inferences:
  - 近常数可能涉及query残差利用不足，而非输入站位置缺失；checkpoint级敏感性待测。
  - 多台可通过带坐标的特征加权恢复空间变化，不证明中间先定位震源。
files_to_inspect:
  - 本文第一节列出的活动config、核心model/data/window/readout/metric及结果provenance。
constraints:
  - 保留原RT55/RT61与FE01-EVAL1的模型/配置/旧lock/工作目录。
  - 新任务独立worktree、显式amplitude policy，不自动训练/提交/下载/test。
  - 若HEAD不同先核验diff，不机械套用base。
proposed_change:
  - 新增fe01_a01诊断wrapper、绝对幅值保留/屏蔽模式、训练/评价入口和聚焦测试。
  - 新增四frontend ON/OFF配置；优先复用已验证旧ON，DiTing需匹配的新ON/OFF。
acceptance_checks:
  - 活动分支追踪、K=1 attention/残差分离、actual gates/gradients、公里尺度有限差分。
  - 无绝对幅值的逐站共同增益不变性；10维屏蔽、时长保留。
  - 从内存原始记录开始的cutoff后一采样/未来脉冲/NaN因果测试。
  - ON路径与原FE01输出/共同初始化/训练语义一致；旧RT55默认行为回归。
  - 旧115文件SHA不变，新source/lock另记；准备脚本fake-scheduler无提交。
hpc_followup:
  - 输出用户可手动执行的stage命令、资源和产物清单；首轮新训练最多5组seed42，取决于旧ON复用验证。
  - EVAL1已计划的正式复评继续独立推进，不被本支线阻塞。
risks:
  - 推理去幅值的分布外效应；幅值旁路与duration混淆；eps/clamp破坏低能量尺度不变性。
  - 台站/时间与预训练/adapter混杂，单seed不构成最终结论。
  - RT61附加尺度/anchor路径未遮断时不能称无幅值；历史震级任务结论不可直接迁移PGA。
open_questions:
  - 只有无法从本地已有资料解析的账户/路径/权重可用性需用户补填；不重复询问已明确的split/窗口/手动提交限制。
[/AI-HANDOFF]
```

Codex 返回完整 `[CODEX-RESULT]`，包含base/result commit、实际改动与测试、原115文件回归、每个探针实际人口、实测gate/sensitivity结果、训练矩阵中实际可复用ON与未运行项、用户手动命令。不得只返回另一份规划，不得把合成测试当作生产科学结论。

## 附：本轮ChatGPT实际完成的验证

`mechanism_checks.py` / `.json` 是独立CPU示例，未导入完整仓库或checkpoint。注意力检查使用PyTorch MultiheadAttention，幅值纯函数转录自上述commit的shape_and_scale。实际结果：单key纯attention跨query差异及query梯度为0；加入query残差后输出可变化；归一化对逐站正增益在float64误差约2.22e-16内不变；幅值10维按log10增益平移、duration不变；固定有效mask后的未来NaN/大脉冲不改变统计。不是训练后模型利用率证据，也不是原始观测到HDF5的端到端因果证明。
