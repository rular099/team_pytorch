# FE02：事件融合与 PGA 读出（2026-10-08）

任务 `20261008-fe02-event-fusion-readout-v1`；仓库 `rular099/team_pytorch`；
基点 `ba4fa7740d9fde5fde87a7b0ae0397037209baf4`；
分支 `exp/fe02-event-fusion-readout`；独立目录 `team_pytorch_fe02`。
这是独立架构实验，不加入 FE01 前端比较矩阵，不改原 FE01/V01/RT55 目录。
用户指令原文存于 [FE02_TASK_PROMPT.md](FE02_TASK_PROMPT.md)。

## 五组固定架构

| variant_id | 名称 | PGA 路径 | event 融合 | 主 PGA MLP |
| --- | --- | --- | --- | --- |
| R0 | legacy_station_linear | target cross-attention | 无 | Linear(1000,64) |
| A | station_nonlinear | target cross-attention | 无 | Linear(1000,64), ReLU, Linear(64,64) |
| B | event_post_add | target cross-attention | 读出后：原 event mapper + 原可学习 gate | 同 A |
| C | event_only | query_no_transformer | 查询 embedding + 同 B 的 event mapper/gate | 同 A |
| M | event_memory | target cross-attention | 在每层 attention 的 key/value 中追加一个事件 token | 同 A |

R0/A 仍计算同一 event 分支并监督震级、位置。所有组 event_cross_attention、
station_context_mode=off；PGA readout 4层、event readout 1层，1000维、10 attention heads，
first residual、既有 FFN/residual gate 设置保持不变。不要按旧文档“多台两层
Transformer”误启用 station context：该 Transformer 注册在模型中，当前路径并不调用。

M 先用原 event cross-attention 从有效观测台站生成事件摘要，经过新增
Linear(1000,1000) mapper 和可学习 type embedding，再追加到台站 memory 末尾。
台站 mask 保持原值，末尾 event mask 为 True。单台有两个有效 key（台站和摘要），
但没有增加独立观测信息。事件 token 没有虚构的台站位置；不加入距离 bias、RoPE 或
几何修正。M 无额外 gate，不进行 gate 初值搜索。
B/C 沿用原 post-add gate 初值0：初始没有 event→PGA 增量，gate 可学习；这不是
强制永远关闭。C 绕过4层 PGA cross-attention，注册参数与活跃计算容量不同，
不得称为严格等容量消融。

## 唯一真实前端和初始化

使用 FE01 的真实 `diting_pretrained_frozen` 构建分支和 `StationFrontend`，
DiTing YAML 为 `diting/config/diting_1200m_backbone_attnpool.yml`：MAE1200M、
width1792/depth24/patch50、10000样本、100Hz、NEZ、5秒 pre-P。
输入原始物理单位 m/s²，FE01 有效样本去均值/三分量共同 peak 归一化；
原因果物理振幅路径保持11维，不加入 GMPE/Vs30/手工距离衰减或方位修正。
继承 model_params 中残留的 `diting_frontend=vit_adapter` 不是实际前端依据；
有效 YAML、实际 encoder/adapter 类、原生 feature shape 和1000维接口由真实 audit 记录。

只严格加载 encoder 完整 keys/shapes，不加载旧 adapter/PGA/辅助头/optimizer。
encoder 不入 optimizer，parent.train 后仍 eval；audit 以一次真实 train-event 更新核验
encoder 参数和 buffers 不变、adapter/下游梯度，随后恢复全部参数与 RNG。
audit 更新不属于正式训练样本预算。B/C 初始 gate 为0时 mapper 梯度可以为零，
同时报告 gradient present 与 nonzero，不将两者混为一谈。
同 seed 的未改动下游 tensor 从同 RNG 初始化的 R0 工厂重建并复制，避免
新增 PGA 层消耗 RNG 导致 event/其他公共模块初始值漂移；adapter 初始化也成对。
这只复制新随机初始化，不引入已训练权重。encoder checkpoint SHA 必须五组一致。
注册文件 SHA 仅证明身份；没有设备 strict load/forward 的结果就保持 NOT_RUN。
预训练语料与本研究数据重叠为 unknown。

## 训练、选择和评价

复用 FE01 V2 原生因果 prefix、冻结 RT55 event split、train-only normalization、
50% normal/50% causal random、每事件每 epoch 三次 early-weighted 随机时刻、
原目标角色分配。单台请求原本已存在，五组不改变采样分布。
正式各组12 epochs、有效batch128、microbatch8、Adam lr0.001、固定更新 cosine，
不 early stop，损失 mag/loc/PGA=0.02/0.02/1，distribution-mean Huber 权重0.1。
正式配置没有事件数量/更新上限；实际 cohort 由同一支持规则和 audit 锁定，
不得把元数据源总数当作有效训练事件数。

按 validation 的 1/3/5/10/20秒、normal/random 十个 cell 等权 noninput MAE 选 best，
并列取早期；40/90秒是独立长时刻诊断，不进入主选择。默认只 seed42 的5组；
seed43/44 配置已准备，但用户未显式提交不启动。可选 pilot 与正式输出/预算隔离，
不是必跑的另一轮 smoke。正式 audit 本身包含前向、梯度和查询独立性门。

评价同时导出 observed input / triggered noninput / untriggered、normal/random，
单台和多台的 MAE/RMSE/R²、bias、双向 slope、NLL、CRPS、Brier、精确 MDN
68/95%区间覆盖、mean±sigma、宽度/PIT。空间指标同时给 centered MAE、
pairwise delta MAE、P95−P05 range ratio，至少5个非输入目标、真实范围<0.05时
不算 range ratio。范围变大不单独构成成功。

配对统计严格比较每条 query 的数据集/事件/时刻/台站/角色/输入 IDs/真值及
同一 encoder/split/population/source；预定 R0→A、A→B、B→C、M→C、A→M、B→M、
R0→M，5000次事件簇 bootstrap，seed20261008。单台/多台分层为探索性，
仅存在的 cell 必须报告数量，不能冒充完整十 cell 主指标。不给所有架构改
`model_family`；逐目标 `variant_id` 和 `event_fusion` 单独保留。

## 兼容性与解释边界

工厂新增 `pga_output_mlp_dims=None`（仅主 PGA MLP/MDN/point 输入维度）与
`pga_event_memory=False`（默认无新 tensor/RNG消耗）；不改全局 output_mlp_dims
或未启用 temporal/delta heads。FE01 engine 新增显式 experiment hooks，省略时
走原构建/审计/训练路径。旧实验输出的 source lock 不被改写或强行迁移。
旧 RT55 原配置、checkpoint loading 与推理必须保留。

采样/实际数据路径沿用 FE01，不读 test 波形或标签、不做 test 选择；本轮
FE02 CLI 不提供 test 参数。catalog picks 是回顾性触发代理、上游 offline
filter/resample 未认证在线因果。attention 图、非同心图或动态范围改善
都不能单独证明物理传播/地质场地机制。单 seed 不能证明稳健优胜。

完整操作见 [FE02_HPC_RUNBOOK.md](FE02_HPC_RUNBOOK.md)。
