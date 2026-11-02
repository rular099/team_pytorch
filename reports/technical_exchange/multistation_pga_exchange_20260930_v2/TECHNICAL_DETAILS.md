# 第 6–8 页技术细节与报告口径说明

这是内部交流的答疑文档，不要求在主 PPT 中逐段讲完。对应修订版第 6 页（实验路线）、第 7 页（架构）、第 8 页（local/transport），同时回答空间指标与文献比较的问题。

仓库：`rular099/team_pytorch`；分支：`exp/v01-velocity-prep-padding-control`；代码基点：`8f54477ed11bf2623f40121648f7fcc756ada1a2`。事实优先依据实现、正式导出和结果表；设计文档中“尚未运行”的旧状态由 2026-09-28 正式结果覆盖。

## 1. 这次交流究竟比上次进步在哪里

上次报告有两条基线，不能只取其中较差的一条：

- 第 11 页：699 train events / 101 validation events 的 full-data 系统锚点，validation MAE 0.2966、RMSE 0.3875、R² 0.5221、slope 0.5931。
- 第 26 页：512-event KNET controlled 实验，rt48 validation MAE 0.2877（该组最低），rt49 的 R² 略高。它们不是全量正式系统成绩，也不能拿 train R²≈0.95 代表跨事件能力。

本次 normal 的低 MAE 代表是 RT55 ep20 **validation**：MAE 0.1162、RMSE 0.1812、R² 0.7524；同一 checkpoint 的 non-input MAE 0.1949、R² 0.4174。RT55 ep32 的 R² 略高（0.7541），但 MAE 稍高（0.1166），所以没有一个 checkpoint 在所有指标都最优。主报告固定用 ep20 展示低 MAE 的完整预测，附表保留 ep32，避免把不同 checkpoint 的最优单项拼成不存在的“超级模型”。

最新 RT61 normal MAE 0.1250、non-input 0.1992；新增 random 的 MAE 0.2373、R² 0.3994。Random 是更难的新任务，不是上次旧任务的同口径回归。

观察层面，旧 full-data MAE 0.2966 到本次 normal 0.1162，差约 −0.1804；R² 由 0.5221 到 0.7524，增加约 0.2303。但数据规模、网络筛选、事件集合、输入/目标协议也变了，这不是配对的纯架构贡献。另一方面，normal non-input 的 R² 仍偏低，不能用 all-target 的高分证明已经解决远程空间预测。

本轮选“normal 代表”仅用于展示既有 validation 结果，未读新 test、未重训、未改变任何部署 checkpoint。RT55 ep20 既有 test 只单列作独立锚点。

## 2. 第 6 页：RT57–RT61 不是五次简单续训

| 实验 | 中文含义 | 试图解决的问题 | 真正训练的部分 | 为什么不能视作已解决 |
| --- | --- | --- | --- | --- |
| RT57 station-distinctive | 保留台站之间波形差异的表示 | 同事件不同台站的池化向量太相似，目标查询难以区分 | 新时序/台站 residual head；RT56 主体冻结 | 单站时空间斜率仍低；表示有差异不等于正确空间输出 |
| RT58 waveform-anchor transfer field | 用输入站的波形锚点向目标位置迁移 | 缺少与真实输入波形相连的局部强度依据 | anchor、集合编码、pair transfer、输出 gate；RT57 主体冻结 | 新 anchor correction 未改善随机目标误差，单站范围下降 |
| RT59 dual-objective residual transport | 同址局部修正与远程残差迁移分开 | random 适配不能破坏 normal，且绝对强度与空间差异需要分工 | local 与 transport 分支；旧分支冻结保留 | 点误差明显降低，但空间场动态范围仍不足 |
| RT60 contrast readout | 只改最后读出层，直接约束场内强弱反差 | 既有 pair 表示是否已有可恢复的空间信息 | transport 最后 MLP：6 tensors / 67,077 scalars | 空间收益很小，normal remote 退化；不晋级 |
| RT61 wave-geometry residual | 给波形状态与目标几何加低秩乘性交互 | 如果最后 readout 不够，是否需更显式的条件交互 | rank16 adapter + 同一 readout：9 tensors / 99,621 scalars | random 点误差略降，但实用门槛未过；不晋级 |

初始化链为 RT56 → RT57 → RT58 → RT59；RT60、RT61 都直接从 RT59 epoch8 出发，**不是 RT60 → RT61**。这几轮的新分支最后都只平移 MDN component means，不是每轮都重新训练 DiTing/TEAM 或不确定性分布。

### 2.1 Station-distinctive：保留什么差异

DiTing 对每个台站输出一串 pre-pooling temporal tokens。若先压成一个通用事件向量，同一事件各站可能方向很相似，PGA 所需的幅度、频率和持续时长差异容易被弱化。RT57 不强迫向量彼此正交，而是使用更直接的输入信息：

1. 显式 token validity mask，阻止 padding token 被当作有效波形。
2. dilation 为 1、4 的两层 masked depthwise-separable 时序块。
3. masked mean、masked std、4 个 learned attention queries 的汇聚。
4. 11 维 log-amplitude：三分量各自 std/RMS/peak（9维），全台站 RMS/peak（2维）。实现基于有效掩码处理后的波形，不能称为未来完整 PGA 真值。
5. 2 维时长特征：有效样本比例、归一化的 log(1+有效样本数)。注意包含 crop/storage 缺口的影响，不只是标称“20秒”。

得到 256 维台站绝对状态 u，并减去有效台站的事件共同中心，得到差异状态 d。

终端式：

```text
                 Σ_i valid[i] · u[i]
event_u = ---------------------------------
                 Σ_i valid[i]

d[i] = u[i] − stop_gradient(event_u)
```

原始 LaTeX：

```latex
\bar u=\frac{\sum_i v_i u_i}{\sum_i v_i},\qquad
d_i=u_i-\operatorname{stopgrad}(\bar u).
```

u 表示“这个台站自身的波形状态”，d 表示“相对同一事件其他输入站的偏离”。单站时 d=0，但系统同时使用 u，所以不是没有波形信息。stop-gradient 只控制反向梯度，并不意味着运行时拿到了真实震源或目录标签。

### 2.2 Anchor transfer：锚点不是直接拿真实 PGA 当输入

RT58 从 u、d 和事件 embedding 预测每个输入站的绝对强度 anchor，再结合 query 与 station 的几何关系产生 transfer。锚点标签只出现在训练损失中，推理时用网络预测的 anchor，不能把完整真实输入站 PGA 喂进 forward。

多个输入站的 candidate 按 learned score 加权，形成目标位置的候选场值；它与 frozen RT57 γ=1.66 的差，经 gate 后修正 MDN 均值。距离门控在同址为零，强调“离开锚点后如何变化”。整个路径不读取真实震中、真实 source distance 或目标 PGA 标签。

γ=1.66 是 frozen RT57 temporal residual 的固定缩放，不是另一场独立训练，也不是 σ 的缩放。本报告主线不再把这个晚于七月的参考当作“上次交流”。

### 2.3 Dual-objective / transport：为什么要两路

normal 中同址输入目标有可直接观测的波形；remote 目标则只有其他站波形和目标位置。同一映射同时兼顾两类目标容易失衡。RT59 将 local 与 transport 分离，并把损失按 random non-input / normal input / normal other 分组，权重 0.50 / 0.25 / 0.25。

Local 预测输入站相对 frozen base 的局部修正；transport 组合 anchor residual、站点强度 level 和 query-conditioned relative residual。既有 RT58 模块仍注册在模型中以保留 checkpoint 兼容，但 RT59 启用时不再应用旧 RT58 field delta。

### 2.4 Contrast readout：把“事件整体偏移”与“场内形状”分开

RT60 固定上游表示，只更新 transport 最后 readout。误差 e 是预测减真实；contrast 对同一事件/时刻的多个 remote query，惩罚相对差异误差。若所有位置同时偏移同一个常数，contrast 不变，因此它必须与 point loss 一起用。

终端式：

```text
                       Σ_j (e[j] − mean(e))²
field_contrast = -------------------------------- , n ≥ 2
                                  n − 1
```

原始 LaTeX：

```latex
S_{field}=\frac{1}{n-1}\sum_j(e_j-\bar e)^2,\quad n\ge2.
```

这与所有 query pairs 的 0.5×差分平方均值等价，实施用 O(Q) 的式子而非显式 O(Q²)。Observed input 不混入 remote pairs；single/multi 和 normal/random 有固定 field bucket 权重。

### 2.5 Wave × geometry：为什么是乘法

RT61 让同一个波形状态在不同目标几何条件下产生不同修正，而不是只拼接一个目标向量。波形状态与 10 维几何特征分别投影到 rank16，经逐元素乘积再映射回 pair hidden state。

终端式：

```text
a[i]    = tanh(Ww · LN(z[i]))
b[q,i]  = tanh(Wg · geometry[q,i])
h1[q,i] = h0[q,i] + Wo · (a[i] ⊙ b[q,i])
```

原始 LaTeX：

```latex
a_i=\tanh(W_w\operatorname{LN}(z_i)),\quad
b_{qi}=\tanh(W_g\phi_{qi}),\quad
h^1_{qi}=h^0_{qi}+W_o(a_i\odot b_{qi}).
```

Wo 零初始化，所以刚加入 adapter 时输出等于 RT59。Frozen teacher 和台站加权 score 一直看 h0，student readout 看 h1，避免对照随着新模块漂移。Adapter-off 用“已训练 RT61 readout + 关闭 adapter”，并不等于独立训练 RT60 或原 RT59。

## 3. 第 7 页：从输入到 PGA 输出的完整路径

```text
截至 cutoff 的波形 + storage support mask + 真实 station 坐标
  → 冻结 DiTing encoder：单站波形/时序 tokens
  → station adapter / TEAM-style 台站集合上下文
  → target query embedding 与目标位置查询
  → base MDN：各 mixture 的权重、均值、sigma
  → RT57 时序站点状态；RT58 anchor；RT59 local/transport
  → [RT61 可选] remote wave×geometry 增量
  → 同幅平移 MDN means → 概率分布 / log10 坐标上的点均值
```

- station adapter 是连接 DiTing 与 TEAM 表示空间的适配层，不等于 RT61 的新低秩 adapter。
- TEAM-style 是多台站集合建模；target query 表示要预测的具体位置。事件 embedding 是网络表示，不是把真实震源参数暗中输入。
- query 与 station 的几何向量由目标坐标、台站坐标、相对坐标和其范数组合成 10 维。范数在当前模型坐标系，不应直接写为“真实千米距离”。
- 单输入 key 的 attention 权重必然为 1：仅依赖 attention 权重改变不一定能产生目标差异，还需 query-conditioned value 或显式 pair 映射。RT57/59/61 正是在 value/残差侧引入波形和几何。
- mask 必须同时控制波形支持、有效台站、时序 tokens 和 query。台站缺失、尚未触发、存储 padding 不是同一种“零波形”。

预测点是 log10(m/s²) 空间上的混合均值，不是线性 PGA 空间的算术期望。MDN 的 component sigma 与整体 mixture predictive sigma 也不同，后者包含分量间均值差异。

## 4. 第 8 页：local / transport 的严格路由

唯一同址匹配由 valid station/query 坐标决定，容差为当前坐标分量 atol=1e-6、rtol=0：

- 恰好匹配一个有效输入站 → local。
- 不匹配，或重复坐标产生歧义 → transport。
- invalid query、没有有效 station → 修正为零。

forward 不读 PGA 标签、导出的 target type、normal/random 标志。这些字段可用于离线分组与训练损失，不能用来“告诉模型答案”。Non-input 表示当前时刻没作为输入，不保证该 station 从未出现在训练中。

为获得输入坐标的 base mean/sigma，RT59 复用同一次 cached station/event state，再解码一次小型 query batch；不重跑 DiTing。Anchor residual 是预测 anchor 减 base input mean，而不是减真实标签。Transport 给每个输入站生成 target-conditioned candidate，再按站点权重汇总。

RT61 的 normal input 增量恒为零，因为只更新 remote 分支。Local 保留不等于所有 normal non-input 都完全不变，后者仍走 transport。

所有分量统一平移 Δ 后，mixture weights、component sigma、整体 mixture variance 保持不变；对真实标签的覆盖率仍会变化，因为分布中心移动。这样解释了“点误差改善”“coverage 变化”与“没有重新学习 uncertainty”可以同时成立。

## 5. 空间范围比：同一场景里台站间的强弱反差恢复了多少

Field 指同一事件、同一 cutoff、同一输入台站集合下，至少 5 个有效目标位置的预测场。先分别计算真实和预测 log10 PGA 的 P95−P05，再取比值；各 field 等权平均。

终端式：

```text
                 P95(predicted logPGA) − P05(predicted logPGA)
range_ratio = ------------------------------------------------
                    P95(true logPGA) − P05(true logPGA)
```

原始 LaTeX：

```latex
r_{field}=\frac{Q_{0.95}(\hat y)-Q_{0.05}(\hat y)}{Q_{0.95}(y)-Q_{0.05}(y)}.
```

- 接近 0：各目标预测几乎一样，空间强弱被压平。
- 小于 1：预测反差不足。
- 接近 1：恢复了类似的整体反差；**不保证哪个台站强、哪个台站弱的顺序正确**。
- 大于 1：反差可能放大，不一定更好。

例：真实 log PGA 的 P95−P05 为 1.0 dex，预测为 0.4 dex，范围比就是 0.4。真实对应约 10 倍幅值范围，预测约 2.51 倍，因此不能说“恢复了 40% 的线性 PGA 幅度”。分母为零的 field 不参与 ratio 平均，不加任意 epsilon。

它不是地图覆盖面积、震中距范围或全数据集散点图斜率。范围比也不敏感于所有目标一起偏移的常数，需要与 point MAE 和 pairwise difference 一起看。旧/新数据的 field 范围分布不同，因此重算同一公式能提供历史背景，但不能给出配对空间提升 CI。

## 6. 为什么此前看起来比文献差很多

核对原报告第 25 页后，旧 rt48@20s validation 的 F1 / PR-AUC 为：1%g 0.620 / 0.642，2%g 0.545 / 0.536，5%g 0.438 / 0.372。Train 为 0.965 / 0.995、0.952 / 0.985、0.932 / 0.985；高分是训练容量，不是跨事件成绩。

上一版新报告有三处展示差异：

1. 左表几乎只放更难的 random，没放 normal；normal all 又含大量已观测输入目标，需与 non-input 一起展示。
2. 旧 F1 根据 point score 在各 validation split 内优化 cutoff；新 F1 用 MDN 概率的固定 0.5 cutoff。两者既换了评分又换了决策规则，不是同口径退步。
3. 右表拿 catalog-parameter forecasting 的 QuakeFormer 数字对照 realtime arbitrary geometry，任务和地区不一致；R² 还依赖真实样本方差，不能只从跨数据 R² 大小推断谁更好。

修订同时提供“复现旧规则的点预测 F1* / PR-AUC”和“固定 MDN 概率 cutoff 的部署型描述”。F1* 是 retrospective validation-oracle 上限，不能当已经锁定阈值的独立 test 或实际预警成绩；没有用该 cutoff 改模型或重跑 held-out test。仍不能对 TEAM 宣称直接胜出，因为 TEAM 是完整时间演化预警评价，这里只有 20 秒快照且没有提前量。

公开原文：[TEAM v4 Table S1](https://arxiv.org/pdf/2009.06316v4)、[QuakeFormer v1 Fig.3 / Table3 / EEW 定义](https://arxiv.org/html/2412.00815v1)。旧点阈值三行已从原 rt48 NPZ 重算并与旧 PPT 逐项核对。

## 7. 源码定位与未解决问题

实现入口（报告制作不改这些文件）：

- `gemini_models.py:1053`：StationDistinctiveTokenAdapter；`:3144`：11维幅度；`:5521`：2维时长。
- `gemini_models.py:961`：几何特征；`:1474`：PGAAnchorTransferHead。
- `gemini_models.py:1702`：RT61WaveGeometryAdapter；`:1739`：RT59ResidualTransportHead；`:2062`：严格路由。
- `tools/analyze_rt59_dual_objective_npz.py:172`：P95−P05；`:178`：空间指标。
- `docs/rt57_gtnp_v2_station_distinctive.md`、`docs/rt58_waveform_anchor_transfer.md`、`docs/rt59_dual_objective_transport_v3.md`、`docs/rt60_contrast_readout.md`、`docs/rt61_wave_geometry_residual.md`：各轮固定合同。
- `assets/tables/revision_summary.json`：本次所有历史/当前来源 SHA256 与派生指标；`meeting_threshold_comparison.csv`：同评分规则对照。

现在可以确认 normal 的系统级点误差较旧报告低很多、概率覆盖率明显改善，random 从原 zero-shot 也有进展。仍不能确认任意未见台站外推已解决；单站空间场、强 PGA、训练种子稳健性和 RT57–61 独立 test 都存在缺口。V01 是受控输入数据机制实验，尚无确认成功的性能结果，不属于本次模型性能进展。
