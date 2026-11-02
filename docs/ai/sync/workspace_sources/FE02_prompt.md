
[AI-HANDOFF]

task_id: 20261008-fe02-event-fusion-readout-v1

repo: rular099/team_pytorch

source_branch: exp/fe01-feature-extractor-site-effects

base_commit: ba4fa7740d9fde5fde87a7b0ae0397037209baf4

target_branch: exp/fe02-event-fusion-readout

goal: 固定同一个 DiTing 预训练 encoder，比较事件表示的不同融合方式，判断能否改善单台 PGA 的空间变化和预测精度，同时改善或保持 normal/random、多台和概率校准表现。

## 1. 本轮任务与工作边界

请完成代码实现、配置生成、本地验证、超算运行脚本、证据封装、提交与推送，不能只返回新的计划。

本轮只使用 DiTing encoder。FE02 的训练、评价和架构比较不包含其他特征提取器，也不开展不同 DiTing 版本、随机 encoder 或 encoder 解冻对照。

用户明确要求：

- 让模型通过波形、输入台站坐标、查询坐标和可学习网络学习空间关系。
- 不向模型加入人工距离衰减、方位修正、GMPE、Vs30 修正或其他手工几何修正。
- 比较后置事件融合、仅事件表示解码、事件 token 加入 attention memory 等设想。
- 同时关注单台、多台，以及 normal/random 的预测精度，不能只展示动态范围变大。

分工：

- Codex：实现、测试、运行接口、实验执行支持、证据封装和推送。
- ChatGPT：后续独立审阅、科学解释和最终架构判断。
- 正式超算任务沿用现有项目的用户手动提交流程。

本轮完成本地可执行工作并提供完整手动命令。命令打印器不能自行调用 sbatch/srun，不自动提交 test，不在登录节点运行正式训练或大规模推理。

如果本地缺少生产数据、DiTing 权重或设备，明确标记依赖这些资源的检查为 NOT_RUN/BLOCKED，继续完成其他实现与验证。不得用合成数据、mock encoder 或本地 dry-run 冒充真实 DiTing/HPC 证据。

## 2. 开始前核验与独立工作目录

按顺序阅读：

1. AGENTS.md
2. docs/ai/PROJECT_CONTEXT.md
3. docs/ai/README.md
4. SESSION_SUMMARY.md
5. docs/ai/FE01_PROTOCOL.md
6. docs/ai/FE01_HPC_RUNBOOK.md
7. docs/ai/FE01_ARTIFACT_SCHEMA.md

检查直接相关代码：

- gemini_models.py
- fe01/config.py
- fe01/model.py
- fe01/extractors.py
- fe01/data.py
- fe01/windows.py
- fe01/engine.py
- fe01/analysis.py
- diting/config/diting_1200m_backbone_attnpool.yml
- configs/fe01/formal/formal__diting_pretrained_frozen__seed42.json
- scripts/fe01/prepare_runs.py
- scripts/fe01/run.py
- scripts/fe01/collect_runs.py
- scripts/fe01/select_run.py
- scripts/fe01/print_submit_commands.sh
- scripts/fe01/job.sh
- scripts/fe01/register_diting.py
- 相关测试和现有结果 provenance

开始时记录：

- 仓库、实际分支、完整 SHA。
- 当前工作目录。
- tracked/staged/untracked 状态。
- 本任务与既有 FE01、V01、RT55 工作目录的关系。

从给定 base commit 创建独立 worktree/checkout，使用目标分支。不要覆盖原工作目录、旧配置、旧 checkpoint、旧 lock 或已有结果。

若源分支已前进，记录差异，不自动吸收无关变更。若本地已开始同一 FE02 任务，先核对身份与已有改动，再继续；不能 reset 未知工作。

建议新增：

```text
fe02/
configs/fe02/
scripts/fe02/
docs/ai/FE02_*
reports/fe02_local_20261008/
fe02_cluster.env.example
```

原 FE01 是固定下游的前端比较协议。本轮属于独立架构实验，不得塞回旧 FE01 的运行矩阵。

## 3. 已核验事实与待检验假设

### verified_facts

在给定 base 上：

- 实际配置使用 event_cross_attention 和 target_cross_attention。
- pga_use_event_context=false。
- station_context_mode=off。
- PGA readout 为 4 层，event readout 为 1 层。
- readout_first_residual=true。
- 当前 output_mlp_dims=[64]。
- MLP 默认只给非最后层添加激活，所以当前主 PGA MLP 是一个 Linear。
- 全局 output_mlp_dims 同时影响 magnitude 和 PGA，不能直接修改它来实现仅 PGA 的非线性对照。
- query_no_transformer 配合 pga_use_event_context=true，可以实现“查询位置＋事件表示”读出。
- 现有 event 注入发生在 PGA readout 之后。
- 本任务的 event-memory 模式需要新增实现。
- 现有训练采样已包含单台请求，不需要为某个候选单独改变采样分布。
- DiTing 入口已支持严格加载 encoder、冻结 encoder、训练现有 attention-pool adapter。

以实际代码、resolved config 和 checkpoint provenance 为准；旧文档中“两层多台 Transformer”的概述不能用来误开启当前关闭的 station context。

### inferences

- 单台时加入一个可区分的 event token，可以解除 attention 只有一个有效 key 时的权重退化。
- event token 仍来自相同观测，不增加独立台站信息。
- 新的空间变化可能来自非线性解码能力，也可能来自事件摘要内容，需要通过对照区分。
- event-only 可能足够，也可能丢失查询相关的局部信息。
- attention 热图、非同心地图或更大的动态范围，都不能单独证明学到了物理传播或地质场地效应。

## 4. 唯一前端：DiTing

所有组统一使用：

```text
model_family: diting_pretrained_frozen
encoder: DiTing MAE 1200M
frontend: backbone_attn_pool
native_n_samples: 10000
sampling_rate: 100 Hz
component_order: NEZ
pre-P history: 5 s
encoder: frozen，保持 eval
现有 attention-pool adapter: trainable
下游: 按各架构定义训练
```

### 4.1 使用实际 DiTing 路径

优先复用：

- fe01/model.py 的 diting_pretrained_frozen 分支。
- fe01/extractors.py 的 StationFrontend。
- diting/config/diting_1200m_backbone_attnpool.yml。

不能只修改 model_family 字符串。

必须审计：

- 实际 encoder 类名。
- 实际 adapter 类名。
- DiTing YAML 解析后的有效参数。
- 原生输入长度、分量顺序、输入单位。
- 原生 feature shape。
- 下游 1000 维接口。
- encoder/adapter 的训练边界。

注意：当前外部 station_waveform_model 注入路径的实际前端由 DiTing 构造过程决定，不能仅依据继承配置中的残留 diting_frontend 字符串判断实际使用的模型。

### 4.2 权重与冻结

使用用户已有的真实 MAE 1200M checkpoint，登记来源、字节数和 SHA。

必须核验有效 encoder 的完整 keys/shapes，并严格加载。

权重注册成功只代表文件身份已登记，不代表加载、前向或冻结审计通过。

要求：

- encoder 参数不进入优化器。
- parent.train() 后 encoder 仍保持 eval。
- encoder 参数及运行统计不随训练变化。
- adapter 和允许训练的下游模块正常得到梯度。
- 各架构使用同一个 encoder checkpoint SHA。
- 仅加载预训练 encoder，不加载旧训练完成的 adapter、PGA、event、location 或 magnitude head。

预训练语料和本研究数据的重叠情况如无法核验，记录 unknown。冻结 encoder 不等于排除了预训练数据重叠。

## 5. 架构矩阵

使用真实 model_family=diting_pretrained_frozen，另设 variant_id。不要把不同架构伪装成不同前端。

| variant_id | PGA 主路径 | 后置 event 相加 | event memory | 仅 PGA MLP |
|---|---|---|---|---|
| R0 / legacy_station_linear | target_cross_attention | false | false | 原 [64] |
| A / station_nonlinear | target_cross_attention | false | false | [64,64] |
| B / event_post_add | target_cross_attention | true | false | [64,64] |
| C / event_only | query_no_transformer | true | false | [64,64] |
| M / event_memory | target_cross_attention | false | true | [64,64] |

### 5.1 所有组共同保持

- no_event_token=false。
- event_readout_mode=event_cross_attention。
- station_context_mode=off。
- readout_first_residual=true。
- 原输入与查询坐标 embedding。
- 原绝对坐标模式。
- 原 embedding 维度、attention head 数、readout 层数及残差/FFN 设置。
- 原 PGA MDN 分量数与尺度约束。
- 原 magnitude/location 辅助头结构。
- magnitude/location/PGA 损失权重 0.02/0.02/1。
- 原 distribution-mean Huber 辅助项，权重 0.1。
- 原因果振幅路径。

R0、A 虽然不将 event 输入 PGA，仍正常计算并监督同一个 event 分支。

### 5.2 各组回答的问题

- R0→A：仅增加末端非线性，能否改善单台输出？
- A→B：事件摘要能否补充原台站读出？
- B/M→C：仅使用事件摘要与查询位置是否足够？
- A/B→M：事件在 attention 内参与读取是否更有效？

C 绕过 PGA cross-attention，活跃参数和计算深度与其他组不同。报告这种差异，不将它称为严格等容量消融。

### 5.3 本轮不默认扩展

不默认新增：

- 不同 encoder。
- encoder 解冻。
- 不同辅助损失权重。
- 不同 gate 初值搜索。
- 不同 PGA MLP 宽度搜索。
- station/event joint self-attention。
- 多个 event tokens。
- 人工几何修正。

## 6. 最小模型实现

### 6.1 仅 PGA 的解码头覆盖

在 build_transformer_model 参数列表末端、**kwargs 前新增：

```python
pga_output_mlp_dims=None
```

默认 None 时完整沿用旧 output_mlp_dims。

显式指定时仅影响：

- 主 mlp_pga。
- 主 PGA PointOutput/MixtureOutput 的输入维度。

A/B/C/M 使用 [64,64]，当前激活规则下应实际生成：

```python
Linear(1000, 64)
ReLU()
Linear(64, 64)
```

再连接原 PGA MDN。

不要修改全局 output_mlp_dims，不改变 magnitude/location heads，不顺带改未启用的 temporal/delta heads。

对非空、正整数维度列表进行验证，并测试实际模块中确有 ReLU。

### 6.2 配置解析

在 FE02 层使用明确的融合枚举：

```text
none
post_add
event_only
memory
```

展开为上表中的底层配置，并检查冲突。

在 legacy factory / FullModel 增加默认关闭的：

```python
pga_event_memory=False
```

现有 post-add、event-only 尽量复用。

当前 factory 对未知 kwargs 可能只打印 warning。FE02 必须通过严格参数验证，防止拼错的新字段被静默忽略。

故意拼错参数时应 fail-fast。审计实际分支、MLP、mapper、gate 与 resolved config 一致。
