# V01：速度波形训练与 P 前缺失/补零对照

设计日期：2026-09-29。状态：ChatGPT 的审阅与实施任务书；尚未修改仓库、建立分支、执行训练或提交 Slurm 作业。

## 0. 版本、目标与本轮边界

- repo：`rular099/team_pytorch`
- 实际核验分支：`rt61-wave-geometry-residual-conditioning`
- base commit：`9c95dbfaf92f36b2673d816026cd3f25b91eec66`
- 建议实施分支：`exp/v01-velocity-prep-padding-control`
- task_id：`20260929-v01-velocity-prep-padding-control`
- 暂缓 RT62 的独立因果桥接和后续模型优化，不使用 RT62 编号。
- 研究目标：用速度计的完整 P 前记录构造受控实验，检验 P 前缺失及其零填充表示是否损害 PGA 预测，同时评估速度输入的可用性。
- 输出目标仍为查询加速度台站的 PGA 概率分布，坐标仍为 `log10(m/s²)`，不是 PGV，也不是速度记录微分得到的新标签。
- 本轮只修改数据接入、配对/干预、配置、入口透传、评估导出、测试和启动器。禁止改 `gemini_models.py`、DiTing 模型源码、模型张量形状、attention/readout/MDN/模型内部幅值计算。

重要：保持模型计算一致，不等于保留错误的数据单位、伪造台站坐标，或把未观测区间当作有效样本。必要的新数据处理必须显式命名，并在全部新实验臂一致执行。

## 1. 已核验事实与尚未知内容

### 已核验

1. `AGENTS.md`、`docs/ai/PROJECT_CONTEXT.md`、`docs/ai/README.md` 与 `SESSION_SUMMARY.md` 均已读取。旧 RT55/RT56 配置和 checkpoint 兼容性不可破坏。
2. `SESSION_SUMMARY.md` 记录 97.37% KNET 记录有前置 storage padding；这是历史数据快照，不等于当前模型窗口中有同等比例的 P 前缺失。
3. RT55 配置开启 `emit_waveform_padding_mask=true`，`noise_seconds=5`，`integrate=false`，实时验证时刻为 1/3/5/10/20/40/90 秒。RT56 在相同模型上进行 50% normal-style / 50% causal-random 的训练采样。
4. `gemini_models.py::FullModel.forward` 使用显式 waveform mask；`_normalize` 在有 mask 时排除无效样本估计均值和标准差。
5. `_encode_station_waveform` 调用 `self.waveform_model[0](waveform)` 时未传时间 mask，后续 adapter 才接收 mask。不能据此断言整个 encoder 已隔离零填充影响；也不能仅凭这一点量化实际损失。
6. 当前 `_get_one` 直接读取 `data/<event>/waveforms/coords/p_picks/pga` 一类训练 HDF5 字段；Hi-net 原始归档不是这种 schema。
7. `docs/hinet_velocity_download.md` 和 `tools/hinet_raw_archive.py` 提供年度 CNT/CH 字节归档、原始计数解码、绝对时间戳、通道响应字段及进程独立 reader。
8. 2026-08-11 的旧审计报告称：当时速度数据为未做响应校正的 raw counts；按 0.5 km 水平距离匹配；有早年缺失及部分三分量不全。这些计数不能直接视为当前数据事实。
9. Hi-net 与 KiK-net 井下仪器存在共址关系，而当前 RT55 主线是 KNET-only。水平坐标相近不能代替传感器身份、深度或地表/井下的一致性。
10. 旧 `loader_light.TrainDevTestSplitter` 依赖原事件列表及顺序。过滤为速度交集后重新调用默认 splitter 会有改变 split 的风险。

### 必须由 Codex 从真实文件核验

- 用户所指的新速度数据是否就是上述 Hi-net 年度归档，或已有解码/校正后的另一个版本。
- 实际 schema、单位、分量、时区、采样率、响应校正状态和来源 hash。
- KNET 对应输入对的数量；KIK surface / borehole 对的数量；真正可用于本轮的 train/validation events、source stations、query targets 和 fields。
- 旧 padding 有多少真正进入当前模型窗口，是否还截掉 P 初至或 P 后样本。
- 真实 RT55 epoch-32 parent 权重、encoder 来源与 resolved config 的身份。

不要要求用户重新手工描述能从本地文件读取的信息；先查工作区和配置。缺失数据时完成不依赖数据的实现，并明确标注 NOT RUN，不虚构最新覆盖率。

## 2. 科学判断与可检验假设

H1：P 前缺失导致输入支持不足/边界和特征变化，对 PGA 预测造成可测损害。
H2：主要问题是无效零值被误标为有效，而不是零作为占位符本身。
H3：速度组改善主要来自信号物理量、仪器频响、井下噪声/场地条件、样本群体或域适配，不是 P 前缺失。
H4：原始档案的前置 padding 多，但在实际模型裁窗中影响有限，或主要误差来自其它环节。

本轮不预设哪项成立。必须区分：
- 数值填充值的影响；
- 缺失真实前缀及合法 mask 的联合影响；
- 未观测到 P 初至/P 后信号的影响（本轮主干预不删除 P 后信号）；
- 速度与加速度输入系统差异。

不能从“V_full 优于历史加速度结果”直接得出“补零是主因”。不能把一次 seed 的区间当作跨训练随机性的保证。

## 3. 固定模型与训练设计

### 3.1 为什么不直接把速度塞入 RT61 readout-only 微调

本轮是输入域变化，不是最终读出的小修。冻结一个只适应加速度的 station adapter/TEAM，只训练 RT61 末端旁路，会把“速度输入是否有用”与“旧加速度表示不可适配”混在一起。

本轮采用明确已有的加速度模型配置：**RT55 模型结构 + RT56 mixed-geometry 采样**。RT56 没有改变 RT55 模型计算结构。本轮是独立的数据机制实验，不是把 RT59 development parent 替换掉，更不声称复刻 RT61 的训练目标。

### 3.2 所有正式新训练臂统一

- `model_params` 完整继承并冻结为解析后的 RT55 配置；保存 canonical JSON/hash，不只比几个显式字段。
- `gemini_models.py` 与实际 DiTing 源码/权重 hash 保持相同。
- 使用同一个已核验 RT55 epoch-32 checkpoint 做 weight-only 初始化；核验真实 epoch/loss/SHA，不凭文件名。
- 新 optimizer、scheduler、epoch、best-loss 状态；各臂起点 tensor-mapping hash 完全一致。
- 冻结原 DiTing encoder；按 RT55/RT56 规则训练 station adapter、TEAM 与原任务 heads；保存相同 trainable parameter 白名单。不得启用 RT59/60/61 的 readout-only freeze。
- 同一套原 PGA/震级/位置任务、损失公式及权重；不引入 contrast、distillation、新局部标签或新辅助头。
- 同一 PGA normalization；warm start 时沿用 parent 的 train-derived mean/std，禁止速度单位影响 PGA 标签去归一化。
- seed=42；8 个新 epoch；固定最终 epoch 8 为正式比较点；第 0 步用于记录域偏移和固定权重敏感性。
- LR/adapter LR/TEAM LR 均为 `1e-4`，8 轮保持固定；其余 Adam 参数等使用共同解析配置。可用现有 scheduler 设置保证预算内不降 LR，必须测试，不允许各臂基于 validation 走不同学习率。
- full matched train cohort，不仅 smoke 子集；每个 epoch 事件/时刻/输入抽样计划一致。
- 训练 50% normal-style + 50% causal random；采样数量和比例沿 RT56。验证为两套固定计划。
- 同一 batch size、world size、梯度累计、训练步数、混合精度和计算预算。DDP 补齐重复项要记录、指标汇总要去重。
- 禁止根据 validation 选 epoch、挑 seed、改掩码强度或自动续训。

这是统一加速度预训练起点的有限预算迁移实验，能严格比较 V_full/V_missing，但不能证明两种物理量各自训练充分后的最优性能。若速度曲线仍明显未收敛，结果应标注预算限制，而不是宣称速度无效。

## 4. 实验臂与比较矩阵

### 正式训练，默认准备三个臂

| arm | 输入 | 作用 |
|---|---|---|
| A_pair | 速度组对应 source pairs 的实测加速度，保留其真实缺失状态 | 同事件/同查询的加速度桥接控制 |
| V_full | 同批真实速度计记录，保留模型窗口内全部可用 P 前样本 | 速度完整前缀模型 |
| V_missing | 与 V_full 完全相同的记录，但人为删除一段 P 前样本并填零、mask=False | 核心缺失对照 |

A_pair 输入传感器不一定是旧 RT55 的 KNET 输入。优先报告 KNET-compatible source subset；其它合法匹配可以作为明确标识的 KiK-net/Hi-net bridge cohort，但不能冒充 KNET-only 结果。查询目标仍保持原 KNET PGA 定义。不得靠扩大匹配距离或伪造 source coords 来凑数量。

同时准备 A_hist：已有 RT55 epoch-32 权重在共同 validation query 集上的重新评估；这是零新增训练的历史参考，输入站/预处理不同处全部标注。已有 RT59/RT61 的总体数字可放背景表，但不得拿它们代替 A_pair。

如果可用 source pairs 太少，不能悄悄把不可比组混合。仍交付 V_full/V_missing 的两臂训练能力以及真实样本数，并把 A_pair 标为不可用/有限；此时只能归因于速度数据内的缺失处理，不能完成跨仪器归因。

### 核心 2×2 交叉评估（只多推理，不多训练）

| checkpoint | eval V_full | eval V_missing |
|---|---|---|
| 训练 V_full 得到的 M_full | R_FF | R_FM |
| 训练 V_missing 得到的 M_missing | R_MF | R_MM |

- R_FM−R_FF：同一完整训练模型面对人为缺失的直接敏感性。
- R_MM−R_FF：各自在对应支持条件训练后的性能差，能区分“只是分布不匹配”与仍存在的缺失代价。
- R_MF−R_MM：缺失训练模型收到真实完整前缀后的表现；也可能有域偏移，不能预设必须改善。
- 每格同时跑 normal-style/random，严格相同 query/decision plan。

### 低成本诊断，不新增正式训练

1. **masked filler invariant**：在最终模型入口，将相同 mask=False 的实验区间用不同有限占位数填充；mask 不变。原模型应在入网 mask 后给出一致输出；报告误差容限和实际 max delta。该检查只证明填充值被覆盖，不证明删除前缀没有影响。
2. **zero-as-valid diagnostic**：只将人工删除区间的模型输入 mask 临时设为 True，零值不变；真实缺口仍不可改成有效。用于测量 mask 误标效应，不是可部署候选、不参与 checkpoint 选择。
3. 固定嵌套剂量推理：P 前保留 0/1/3/5 秒及 full；只做描述，不据 validation 选择生产保留时长。仍保留全部 P 后允许样本。若某记录可见 P 前不足该剂量，报告实际有效干预量。
4. 可在确有完整 P 前上下文的真实加速度 train/validation 子集上做同样固定权重删除对照；该子集选择偏差需报告，不自动增加训练。

本轮不默认启动“速度微分后加速度”的额外训练。只有同源 V 对照显示实质影响、且确需把结论迁移回加速度域时，另行设计同记录微分的完整/缺失对照；微分不能恢复原本缺失信息，也会改变频带与噪声。

## 5. 数据契约：独立 source 与 query，不借用坐标伪造同站

建议新建 `tools/velocity_waveform_backend.py`、`tools/prep_padding_protocol.py`，或等价独立模块。不要复制完整模型/训练循环。

canonical event sample 至少包含：

```text
schema_version, event_id, split, origin_time_utc, source_file_hashes
sources:
  source_sensor_id, source_site_id, network, sensor_class, component_order
  actual latitude/longitude + elevation convention, sensor_depth provenance
  raw_counts_or_waveforms, original_time_axis, sampling_rate
  units, calibration_status, response metadata/hash
  channel_sample_valid (S,C,T), station availability
  source P pick time + source/method
  paired_acc_sensor_id, match distance, matching tier
queries:
  target_sensor_id, target_site_id, actual target coordinates
  original acceleration pga, target_valid, pga definition/source hash
  target P pick and lead-time provenance
plans:
  event_id, protocol, epoch or fixed-eval index
  decision timestamp/current sample, physical crop origin
  selected source IDs, selected query IDs, random seed/plan hash
intervention:
  template_id, donor split, retained_preP_seconds, intervention mask/hash
```

- 输入坐标是实际速度传感器坐标；目标坐标是实际 PGA 观测仪器坐标。不得写成相同坐标以触发 local 路由。
- 以传感器 ID 为实际 input membership，另导出 `paired_site`。井下速度输入并不等于地表加速度目标已经被直接观测。
- 所有臂都报告 all targets；主要跨域可比群体用预注册的 common-remote targets，排除任一 source arm 的实际输入目标及配对站址目标。距离匹配阈值不等同于 input 身份。
- V_full/V_missing 的物理输入身份完全相同，二者的 all/non-input/untriggered 比较是真配对。
- 某速度协议没有实际 input target 时，input 计数为 0、指标为 N/A，禁止造出历史约 70% input 群体。
- 同一 Hi-net sensor 对到多个加速度行时必须去重；尤其不能把同一个速度波形复制成 surface/borehole 两个输入站。保留完整映射，多对一不变成多台站观测。
- 坐标数值单位/高程符号转换按已核验旧接口执行；新增深度/网络 provenance 不可偷偷作为额外网络特征。
- 已有 forward 能接受独立 input coords 与 query coords；要修复的是数据接口，不是模型计算。

## 6. 缺失模板和严格配对

### 6.1 先测真正入模的缺失

分别统计原 storage padding、模型窗内缺失、P 前有效秒数、P 初至是否覆盖、P 后已观测秒数。必须从 storage support 和时间戳计算，不能以 `abs(waveform)>eps` 判定每个时间点是否真实存在。

旧 `_crop_aligned_event_window` 在长记录上随机选一个合法 pick 作为裁窗锚点，且受完整记录长度影响；同 seed 不保证两种文件长度下物理裁窗一致。新增计划必须锁定绝对时间和参考，不能让速度记录变长顺便改变 t=1 秒的含义。

保持原模型输入长度与采样率（从 resolved config 验证，预期 10000 点/100 Hz），不把全部 P 前 120 秒塞入模型，也不扩大计算图。结构性 station padding、未来时间 padding 仍存在，V_full 不是“张量中完全没有零”。

### 6.2 主干预

- 模板库只从原加速度 **train** 的 KNET storage metadata 构建，不使用 validation/test 误差选择模板。
- 模板是相对该站 P 到时的真实前置支持形状/保留时长，不含波形幅值和 PGA。
- 对同站真实对应模板可另导出诊断；正式主模式使用固定的 train-KNET 模板分布，避免 Hi-net 主要对应 KiK-net 时把缺失分布悄悄换成 KiK-net。
- 每个 `(seed, event_id, unique source_sensor_id)` 固定一个模板，跨 epoch 和多个实时刻使用同一物理支持；不得每个时刻独立重采样导致“消失样本重新出现”。
- V_missing 只在 `time < source_P_time` 删除模板指定的样本，填零且 model sample mask=False。
- 不删除 P 初至或 P 后样本；P 本身定义为保留。若原加速度在 P 后才开始记录，这一额外信息缺失单列，不能混入主干预。
- 在数据依赖的中心化/归一化之前施加干预；不能先用已删除前缀估计噪声、均值、滤波状态或增益，再只把输出抹零。
- 同记录在逐点单位换算后的 P 后原始数值必须完全相同。后续合法归一化因删除前缀而变化属于研究机制，应记录而非强行抵消。
- 当采用有记忆的响应校正/滤波时，要单独界定状态是否可用；不能让缺失组暗中继承完整组的滤波状态。

### 6.3 配对锁定

- event split 先从原 frozen manifest 读取，再与实际速度/加速度覆盖求交；不能在过滤后的列表上重新划分。
- test ID 仅用于拒绝混入，不加载 test 波形、标签来调试或拟合缺失模板。
- 每个训练 step 的事件、时刻、query、逻辑 source slots、随机计划相同。V_full/V_missing 选中的物理 source IDs 必须完全一致。
- 源站是否因果可用，主实验用共同元数据和未被干预的合法观测部分决定，不能因删除 P 前噪声就重新选站或替换事件。
- 不允许旧 `__getitem__` 静默跳到下一事件破坏配对。跳过/无输入必须在计划里显式记录，并对所有需配对视图一致处理。
- P picks、t0、time bins 和 target label 固定。重新从速度全记录挑更准的 P 波不得与 padding 干预一起混入主对照。
- 新选择器以 manifest IDs 而非 worker/global RNG 偶然顺序为准，DDP/worker count 改变不应改变已冻结 eval plan。

## 7. 速度单位、响应和时间处理

### 默认最小速度视图

若真实来源是年度 CNT raw counts：复用 reader 和其 `counts_per_physical_unit`，逐通道做已验证的灵敏度/增益换算，保存 `waveform_quantity=velocity_sensor_output`、`units=m/s`、`response_correction=sensitivity_only`。

必须明确：这属于统一单位的速度计输出，仪器频率响应形状仍保留，不是已经完整反褶积的宽频真实地动速度。它能用于 V_full/V_missing 的受控实验，因为仪器与标定完全相同；A_pair/V_full 比较仍包括频响差异。

- 若用户新数据已有可靠全响应校正，不重复校正，核验并继承其单位/处理 provenance。
- SAC 经 win2sac_32 转换通常已去灵敏度并乘 1e9，不能再当 counts 除一次增益；必须处理 nm/s→m/s，保存转换链。
- 无合法响应/增益/单位的通道不可默默当 m/s。训练前 fail closed，并给出通道计数。
- 主实验不新增全事件零相位滤波、全记录反褶积或微分。若必须重采样或去响应，只从允许的前缀及明确合法的历史状态计算，记录延迟/边界；不可先处理未来尾段再截断。
- 不更改 PGA 标签。不对原加速度设置 `integrate=true` 并把它当作新独立速度观测；积分不会生成已缺失的真实 P 前记录。
- preflight 必须记录 SI 单位下输入 peak/std/RMS 的分位数、std 接近或低于模型 `1e-8` 下限的比例，以及波形幅值特征下限命中率。不能因速度单位数值较小导致模型被下限压扁而误判为前缀效应；也不能根据 validation 调倍率或擅改模型 epsilon。需要额外输入单位倍率时列为新的数据协议决定，不混入当前正式比较。
- 三分量顺序跟旧模型一致，保存通道映射；U/Z、N/1、E/2 等别名不能掩盖未知真实方位或井下旋转。
- 不能把 CH 站点海拔直接当作井下传感器高程；可见深度信息不足时记录缺失和现有接口采用的约定，不臆造值。
- JST/UTC 明确转换；同一 event_id 不代表已对齐的零时刻；以绝对时间、origin correction 和采样端点核验。
- 饱和、坏道和部分分量缺失需要 flags。主质量规则训练前冻结；按全记录/PGA 极值挑除 validation 难例不允许。后验 QC 可以描述但不能悄悄改变正式主群体。

## 8. 新入口的因果性与 RT62 关系

不开展 RT62 独立全链路重放，不修改旧行为。但不能在新实验里复制已知的数据错误：

- 新 data backend 在明确决策时刻后，使用 exclusive cutout=`current_sample+1`。
- 不使用 `:cutout+1` 去均值；不先全记录去均值再做可用池 `has_signal`。
- 若有效速度比 `1e-8` 小，不能简单套用加速度数值阈值导致换模态就失去台站；新配对计划以元数据支持和明确观测规则为主。不同物理单位的 sentinel 策略必须显式统一为新的数据契约，不改模型内部归一化 epsilon。
- 全部新臂共用上述 new-backend contract；历史旧结果标为 legacy，不将其与新臂差异直接归因于 padding。
- 原始加速度 HDF5 是否已经包含离线全记录处理，需要 provenance 标注；仅通过新 loader 的前缀测试不能认证上游所有处理均为在线因果。
- 真实长记录的额外过去信息不得被不对称用于噪声标准化或隐藏滤波 warm-up。归一化观测域/支持必须记录。

## 9. HPC 数据读取与资源设计

- 不在训练的每个 station/每个实时刻反复解码整事件 CNT。按事件、进程有界 LRU 复用解码结果。
- 每个 DataLoader worker 懒加载独立只读 reader，禁止父进程打开 HDF5 后跨 fork 复用。
- 使用 `max_open_archives`、cache event/byte 限额、线程上限；记录 cache hit、I/O wait、batch latency、峰值 CPU/GPU 内存。
- 优先使用原 archive，不另存永久三套波形。V_full/V_missing 是同一源的视图，不复制波形大文件。
- 节点本地 `$SLURM_TMPDIR` 可选临时缓存，内容由 source hash、单位处理版本和采样率定址；不要把不可复现的缓存当源数据。
- `.partial.h5` 只能在固定快照/只读且写入停止时读取已提交、经 hash 验证的记录。禁止边下载边训练可变数据。
- snapshot 来源、可用事件和 CH/CNT hash 固定；无网计算节点不得运行下载、安装或凭据请求。

## 10. 必须保存的证据

### HPC：完整产物

1. 源归档/训练 HDF5、split、source-pair mapping、PGA target manifest、模板库、固定采样计划、配置/代码/encoder/parent 的 hashes。
2. 所有训练 checkpoint 与恢复所需状态；实际 epoch/loss/训练步数、trainable list/hash、LR 曲线、梯度和数据吞吐。
3. 每个 arm、checkpoint、eval view、protocol 的逐目标 NPZ/Parquet：
   event ID、query sensor/site ID、源站 ID 列表、decision timestamp、原始/裁窗 P time、mask/plan hash、单位/处理版本、真实 PGA、raw-log-coordinate MDN logits/mu/sigma、点均值、predictive sigma、NLL、threshold probability、input/paired-site/untriggered flags。
4. 每源站在模型窗内的真实支持与人工干预 mask、preP/postP 有效秒数、padding 类型比例、P 是否保留、所用模板 ID；被排除/跳过原因。
5. 原始样本→校准→裁窗→干预→中心化/归一化→模型输入的有限固定案例，含数值校验。限制原始 Hi-net 波形的复制/公开，遵从其使用条件。
6. stage0/epoch8 四格 cross-eval、A_pair、A_hist 的输出和全部失败结果。不能只保留效果好的视图。

### GitHub：轻量审阅包

- `RESULT_REVIEW.md`, `summary.json`, `protocol_lock.json`, `effective_configs/`
- `cohort_counts.csv`, `source_pair_summary.csv`, `exclusions.csv`
- `padding_exposure.csv`, `intervention_checks.json`, `mask_invariance_checks.json`
- `group_metrics.csv`, `strata_metrics.csv`, `cross_eval_metrics.csv`, `paired_ci.csv`
- `field_metrics.csv`, `field_error_decomposition.csv`
- `training_summary.csv`, `parameter_identity.json`, `checkpoint_manifest.json`
- 原/新预处理差异说明、源文件 hash、完整复现命令、HPC job ID/状态、未执行事项。
- 图：一致坐标/分箱的真值—预测密度图，normal/random 分开且含 remote/untriggered；误差对 P 前有效秒数、缺失比例、postP 秒数；早期时刻误差；四格交叉矩阵；coverage/校准；固定抽样的空间场成功和失败案例。
- 图必须附所用 counts、protocol、checkpoint、单位和绘图源数值；raw Hi-net 波形不默认公开到 GitHub。

## 11. 指标、主终点和结论规则

- PGA 固定 `log10(m/s²)`；point estimate 为该坐标的 predictive mixture mean，不与线性 PGA 的期望混用。
- 主要群体：common-remote；同时报告 all、actual-input、paired-site、triggered-noninput、untriggered。
- 主终点：固定 epoch8，分别对 normal-style/random 的 1/3/5 秒 common-remote targets 汇总 MAE；这两个协议是预先指定的共同主要比较，不能只挑成功的一项宣称全面改善。
- 若某群体没有足够数据，先报告 N/A 和原因；不能事后换群体。
- 核心比较为 R_MM−R_FF；同 checkpoint 的 R_FM−R_FF 作为缺失敏感性证据。报告差值、相对效应和事件聚类 95% CI。
- 描述性实用门槛：相对 MAE 改善至少 2%，且配对 CI 支持改善；2% 是预注册研究阈值，不保证能达到。只有小效应/CI 跨零时写“未检测到足够大的效果”，不能写“证明无影响”。
- 总体还必须报告 MAE/RMSE/R²、bias、prediction-on-truth slope 与 truth-on-prediction calibration slope、P90/P95 abs error、within0.1/0.2 dex。
- NLL、Brier（原阈值 -1.2）、predictive sigma、旧 mean±sigma coverage，同时给精确 mixture-quantile coverage/PIT 以免把 MDN 当单高斯。
- 空间指标：field mean error、centered error、P95−P05 range error、pairwise delta MAE，n>=5；实际单站单列。
- 每项 metrics 和每个分桶给 contributing events、decision rows、targets；单站是 actual source sensor 数，不是 requested count。
- bootstrap 单位 event ID，同一事件不同台站/时刻一起重采样；5000 draws、seed20260915。主估计 target-weighted，另给 event-macro 作为稳健性描述。
- 同一事件所有 eval views/arms 共享重采样索引；不可按 target 独立 bootstrap。
- 跨 A/V source coords 或输入身份不同，配对只在“同事件-同决策-同查询标签”的层面成立，不能声称输入也完全相同。
- 给出匹配交集与更大可用群体的区分，不能比较不同 counts 的总 MAE 作为纯干预。

结论分类：
A. V_full 优于 V_missing，缺失训练后仍有代价：支持 P 前缺失及现有处理链的损害，限当前数据/模型/预算。
B. 完整训练模型受缺失伤害，但 M_missing 在缺失视图追回：更像分布不匹配/适应问题，不证明不可恢复信息损失。
C. V_full 和 V_missing 接近，但都优于 A_pair：速度/仪器/数据域因素更值得考虑，不能归因于补零。
D. masked filler 不变、zero-as-valid 退化：数值占位符本身被覆盖，误标有效支持会造成影响。
E. 早期效果明显、晚期消失：支持早期上下文机制；不能推广为所有时刻的大幅收益。
F. V_full 更差、差异很小或 CI 太宽：如实报告噪声稀释、域适配预算、样本量限制等备选解释，不据此调 test。

## 12. 验收测试

必须先本地 synthetic，再小量真实读取 preflight；通过后交付 full-cohort HPC 运行脚本，不无限追加 smoke。

- 精确保留 legacy data backend、RT55/RT56 config hashes、旧 state_dict names/shapes 和 checkpoint 推理。
- A_pair/V_full/V_missing model_params、初始权重、训练白名单完全一致；`gemini_models.py` 无 diff。
- CNT 解码和 CH gain 转换对已知合成/官方转换结果校验，SAC 的 nm/s 与 raw counts 不混淆。
- 三分量缺失、异采样率、重复/非单调时间、时区、跨分钟拼接、相同/冲突重叠段、饱和 flag 处理有测试。
- event split 在过滤前后不变；test 数据不进入任何模板/梯度/选择。
- 相同时间计划与 query label hashes；V_full/V_missing 原始 P 后值相同、只删除 P 前、干预 mask 是原有效 mask 子集。
- 当前物理截止、crop anchor、source IDs、target IDs、source coordinates 在两个 V views 完全一致；未来修改不改变当前输入。
- 被删除前缀改成任意数值后，V_missing 的允许输入不变，防隐藏归一化/滤波状态泄漏。
- 改变未来后缀、未来 NaN/Inf，不影响当前可用波形、mask、source/query plan。不能仅检测输出 shape。
- masked filler 不变性在生产 FullModel 路径验证；zero-as-valid 不得修改真实缺口掩码或原始文件。
- 多 worker/fork、CPU小样本、单GPU一个优化步、可用时多rank DDP、断点恢复与固定计划一致性。
- formal eval merge 去重、event-cluster CI、MDN 去归一化、source-vs-query membership 与 counters 有单元测试。
- `bash -n`、`py_compile`、既有 `tests/test_hinet_raw_archive.py` 与相关 causal/config/checkpoint 回归真实执行；无环境则标 NOT RUN。

## 13. 文件级实施建议

新增（名称可作小幅调整，最终 prompt/文档和命令必须与实际文件一致）：

```text
tools/velocity_waveform_backend.py           # 归档/HDF5探测、counts/units、worker/cache
tools/prep_padding_protocol.py              # source/query契约、固定计划、支持干预
tools/build_v01_paired_manifest.py          # metadata审计、split继承、模板/配对/统计
tools/analyze_v01_padding_controls.py       # 四格评估、配对统计与图
tools/run_v01_prep_padding_controls_slurm.sh # preflight→三臂训练→验证→分析
pga_configs/v01_common_rt55_model.json
pga_configs/v01_acc_pair.json
pga_configs/v01_velocity_full.json
pga_configs/v01_velocity_missing.json
pga_configs/v01_validation_normal.json
pga_configs/v01_validation_random.json
tests/test_v01_velocity_backend.py
tests/test_v01_padding_controls.py
docs/v01_velocity_prep_padding.md
docs/ai/CODEX_RESULT_<date>_V01_VELOCITY_PADDING.md
```

必要小改：
- `loader_light.py`：显式 split manifest 支持/metadata routing，旧默认不变。
- `gemini_util_light.py`：独立新 backend factory 或子类共享通用变换；不把整份 `_get_one` 分叉成不可维护副本。
- `train_light.py`、`eval_checkpoint.py`：新 backend 的必要透传、provenance、固定评估计划；继续复用旧模型构建和损失/分布计算。
- `tests/test_causal_random_geometry.py` 等：新接口和旧默认回归。

禁止修改：旧 RT55–RT61 配置、旧结果目录、`gemini_models.py` 与 DiTing 计算代码。若确有无法保持 forward 不变的阻塞，明确报告并停在该修改边界，不用模型改动偷换任务。

## 14. 超算脚本交付契约

必须交付真实脚本，不只写 sbatch 使用建议。复用现有 `tools/run_rt56_random_geometry_slurm.sh`、`train_light_slurm.sh`、`eval_checkpoint_slurm.sh` 的合法入口，不能假定 GPU/DCU 环境与本地一样。

脚本要求：
- `set -euo pipefail`，默认 `DRY_RUN=1`，正式提交需 `CONFIRM_V01=1 DRY_RUN=0`。
- ACTION 支持 `preflight|train|eval|analyze|all`；ARMS 支持 `vfull,vmissing,apair`，默认准备三臂。
- `all` 用可核验 Slurm `afterok` 依赖串联 preflight、训练、验证、分析；一个前置失败不允许后续继续运行。
- 两个 V 模型的 cross-eval 必须固定到正确 epoch8 checkpoint，不能使用模糊 best/last 文件名代替元数据检查。
- 训练输出目录全新；已存在非空目录拒绝覆盖。resume 只能显式开启，核验 arm/protocol/parent/manifest/data hashes 并恢复原随机计划。
- 不删除或 RESET 旧实验；不允许脚本默认 RUN_EVAL 触发 test；正式评估只传 `--splits val`。
- CPU preflight 不占整组GPU；训练资源、partition、account、节点/GPU/DCU数、batch size、cpus、内存、time limit 由环境变量传入，继承现有启动器的真实变量名。
- 支持只生成 manifest/preflight 供核验；报告实际样本数、吞吐和总训练更新数。
- 记录 sbatch 命令、job IDs、source commit、dirty status（生产默认拒绝dirty）、manifest hashes、环境版本、stdout/stderr路径。
- 大型数据和 checkpoint 缺失时 DRY_RUN 可以列出缺项；不得写“真实训练测试通过”。

最终给用户一个可复制的 env 示例（所有路径为占位符）：

```bash
export WORKDIR=/path/to/pinned/team_pytorch
export ACC_DATA_ROOT=/path/to/acceleration_hdf5
export VELOCITY_DATA_ROOT=/path/to/velocity_archive_or_verified_dataset
export FROZEN_SPLIT_MANIFEST=/path/to/original_split_manifest
export RT55_EP32_CHECKPOINT=/path/to/verified_epoch32.pth
export V01_RUN_ROOT=/path/to/new/v01_run
# encoder/dtbench/Slurm资源变量使用实际启动器验证过的名称

DRY_RUN=1 ACTION=all ARMS=vfull,vmissing,apair \
  bash tools/run_v01_prep_padding_controls_slurm.sh

CONFIRM_V01=1 DRY_RUN=0 ACTION=all ARMS=vfull,vmissing,apair \
  bash tools/run_v01_prep_padding_controls_slurm.sh
```

上述文件名/命令是交付契约，不表示脚本已在仓库存在。Codex 结果里必须附实际脚本内容、实际可用命令、资源设置与所有环境变量说明，由用户在超算提交，不替用户提前提交训练。

## 15. 可直接执行的 AI-HANDOFF

[AI-HANDOFF]
task_id: 20260929-v01-velocity-prep-padding-control
repo: rular099/team_pytorch
branch: exp/v01-velocity-prep-padding-control
base_commit: 9c95dbfaf92f36b2673d816026cd3f25b91eec66
goal: 实现模型计算不变的速度输入训练及完整/人工P前缺失配对对照，交付HPC脚本与足以独立归因的证据。
verified_facts:
  - 当前RT55有显式storage mask与masked normalization，但encoder调用未接收时间mask。
  - 当前训练HDF5与Hi-net年度CNT/CH字节归档schema不同，已有worker-safe reader可复用。
  - Hi-net旧审计为raw counts且0.5km水平近邻匹配，不保证与KNET同仪器/同深度。
  - RT56保持RT55模型结构，用50% normal/50% causal-random采样。
inferences:
  - P前缺失可能造成部分性能损失；尚不能判断为主要瓶颈。
  - 单独速度对加速度的胜负不能归因为补零。
files_to_inspect:
  - AGENTS.md、docs/ai/PROJECT_CONTEXT.md、docs/ai/README.md、SESSION_SUMMARY.md
  - docs/hinet_velocity_download.md、reports/hinet_dataset_audit_20260811/*
  - loader_light.py、gemini_util_light.py、gemini_models.py、train_light.py、eval_checkpoint.py
  - tools/hinet_raw_archive.py、RT55/RT56配置和Slurm脚本、对应测试
constraints:
  - 暂缓独立RT62任务，不改gemini_models.py/DiTing计算/旧配置/checkpoint行为。
  - 固定PGA标签、原event split、物理截止和采样计划；不读取test做实验。
  - 速度source与加速度query身份/坐标分离，不伪造同站或复制波形增加台站数。
  - 本任务书第3至14节为实验、单位、统计、测试与HPC交付规范。
proposed_change:
  - 独立速度backend、source/query及padding干预协议、metadata manifest工具。
  - 原训练/评估入口只作必要透传；损失和模型forward保持不变。
  - A_pair/V_full/V_missing同初始RT55ep32、同RT55结构和RT56采样、同8轮预算。
  - 固定四格cross-eval、mask诊断、分层CI与全部provenance导出。
  - 新分支隔离；高复用不等于raw数据schema可直接替换data_path。
acceptance_checks:
  - 真实schema/units/three-component/timeaxis核验、原split继承和train模板隔离。
  - 模型源码、model_params、初始tensor hash、trainable列表与损失一致。
  - 两个V视图raw postP值一致、仅P前干预、同source/query/decision plans。
  - future/deleted-prefix变形不变性、masked filler、worker/DDP/resume/旧RT55回归。
  - 三臂及四格逐目标导出、实际counts、event-cluster5000次CI。
  - 脚本bash-n和DRY_RUN通过；未实际执行项目如实标NOT RUN。
hpc_followup:
  - 交付真实run_v01_prep_padding_controls_slurm.sh及环境变量说明。
  - 用户在超算执行；不自行提交，不自动扫参/增epoch/开test。
  - full matched train；验证使用固定epoch8和val两协议；资源与输出不覆盖。
risks:
  - speed和acc仪器/场地/频响与输入域变化；A_pair可能不是KNET-only输入。
  - 不完整预P以外的P/postP缺失，本轮未全部解释。
  - 同加速度预训练起点的短预算迁移，不证明各模态全量训练的上限。
  - 一seed与反复开发validation，只支持本协议下结论。
open_questions:
  - 实际速度格式/覆盖/父权重由本地preflight解析；非必须先向用户追问。
  - 若必须改变模型计算、PGA定义或查询网络才可运行，报告阻塞，禁止静默改动。
[/AI-HANDOFF]

返回 `[CODEX-RESULT]`：base/result commit、branch、changed_files、真实测试命令与PASS/FAIL/NOT RUN、RT55兼容性、HPC状态、数据/单位/统计限制，以及用户可直接执行的真实Slurm脚本和提交命令。代码提交推送后供ChatGPT按精确commit复核。

## 16. 审阅依据

以下均已通过GitHub读取于上述base commit；引文描述依据原文件而非旧聊天的行号：
- `gemini_models.py` 约5080–5140：masked normalization；约5570–5650：encoder与adapter调用；约5860–5970：输入mask/归一化/幅值支路。
- `gemini_util_light.py` 约370–435：crop/支持mask/中心化；约1580–1750：HDF5读取与causal随机预选择；约2080–2335：target选择、validity和有效时长。
- `loader_light.py` 开头：按原事件数组划分；`tools/hinet_raw_archive.py` 约650–930：archive reader及worker-safe dataset。
- `pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json`
- `pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json`
- `tools/run_rt56_random_geometry_slurm.sh`；`eval_checkpoint.py` 开头MDN单位转换。
- `tests/test_hinet_raw_archive.py`；`docs/hinet_velocity_download.md`
- `reports/hinet_dataset_audit_20260811/acceleration_velocity_dataset_audit.md`（历史审计，不代表当前完成量）。

外部原始文档（核验于2026-09-29，供Codex核查仪器/单位）：
```text
NIED Hi-net站点与仪器说明：https://www.hinet.bosai.go.jp/summary/
NIED K-NET/KiK-net说明：https://www.kyoshin.bosai.go.jp/en/about_kyoshin/
HinetPy响应/灵敏度：https://seisman.github.io/HinetPy/appendix/response.html
HinetPy SAC单位说明：https://seisman.github.io/HinetPy/tutorial/conversion.html
ObsPy bandpass：https://docs.obspy.org/packages/autogen/obspy.signal.filter.bandpass.html
ObsPy remove_response：https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html
```
