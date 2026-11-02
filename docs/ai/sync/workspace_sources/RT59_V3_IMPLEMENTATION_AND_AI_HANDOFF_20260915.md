# RT59-v3：局地校正与空间传递分路、random/normal 联合优化

日期：2026-09-15。性质：实施设计、可执行参考函数及 Codex 交接，不是已完成的仓库改动或性能结果。

本文件**完整替代**此前 RT59 v1/v2 执行指示。v2 主要提高了 normal 验收要求，却保留了不充分对应目标的训练机制；v3 实质修改路由、损失归约和 normal 保护方式。只有一个联合方案、一个 seed、一轮固定训练，不把各修改的独立贡献说成已经识别。

## 1. 本轮实际核验

```text
repository:       rular099/team_pytorch
branch:           rt58-waveform-anchor-transfer-field
remote HEAD:      40830acd475bd40347fef03427b52c4a04432dc3
evidence parent:  37d97ad9ad67dfec11f188e2a350a8ab8b8b2056
```

本轮重新读取了分支状态以及以下代码区间；此前对同一 commit 的协作说明、RT58 模块、配置、测试和报告阅读仍有效。

| 源码/证据 | 位置 | 已核验的事实 |
|---|---|---|
| `gemini_util_light.py` | 1690–1920、1920–2095 | 先确定 realtime cutoff，再选择/约束输入；右对齐波形并掩掉未来/未触发台站；PGA 标签与输入 mask 分开 |
| `gemini_util_light.py` | 2400–2610 | `causal_random_mask_applied`、input PGA、target type/indices 等是训练/评估 metadata |
| `gemini_util_light.py` | 2580–2640 | 当前位置变换语义；配置 `scale_metadata=false`，不能任意把 model-coordinate 距离当公里 |
| `train_light.py` | `train_model`，约 2309–2490 | 原 PGA NLL/mean loss 与 auxiliary losses 汇总；没有本方案的协议分组归约 |
| `train_light.py` | 1550–1750 | candidate loss 的 pair mask；normal distillation 对齐 base mean；query-difference loss |
| `gemini_models.py` | `PGAAnchorTransferHead`，1469–1660 | signed tanh final gate、station aggregation 和 `gate*(field-base)` |
| `gemini_models.py` | 5062–5145 | mixture mean 与全部 component means 的同量平移 helper |
| `gemini_models.py` | 5310–5830 | station/event/query readout、RT57 temporal residual、RT58 mean shift 集成点 |
| `eval_checkpoint.py` | 310–550 | 固定 split 构造、validation overrides 与 deterministic sampling |
| RT58 validation README | 约 275–330 | normal all/input/non-input 计数与误差；原 go/no |
| RT55/RT57/RT58 configs | 已读取 | normalization、freeze、训练协议与配置继承 |

这些引用全部锁定上述 HEAD。GitHub 轻量证据没有 checkpoint 本体、原始 NPZ 或训练曲线；不声称本轮重新做过真实数据复算。

### 1.1 normal 的具体问题

同协议 frozen RT57 gamma=1.66 与 RT58 epoch8，normal validation：1,383 events，9,681 realtime rows，89,770 targets；其中 input 63,651、non-input 26,119。PGA 坐标为 `log10(m/s²)`。

| 目标 | base MAE | RT58 MAE |
|---|---:|---:|
| all | 0.131016 | 0.131610 |
| input | 0.098452 | 0.099104 |
| non-input | 0.210371 | 0.210826 |

input 占 validation targets 约 70.91%。**这不是 train 中 input 目标占比**，训练有自己的采样配比，必须记录实际训练计数。现有 RT58 的绝对 anchor 不能凭另一个总体 MAE 就宣布比这些 input base 更精确，因此本方案不直接用 anchor 替换 base。

### 1.2 三个需要修复的训练机制

1. 有本站波形的目标与远处 query 经同一个 transfer/fusion 头处理；normal 输入站可能受不准确空间传递牵连。
2. 80/20 样本比例和全体有效 targets 的 pooled loss 不等于对两种协议、两类 normal 目标有明确的优化预算。
3. 无条件 distillation 惩罚离开旧预测，包括旧预测错误而 student 修正正确的情况。

以上第1项是有证据支持的设计动机，不是已独立证实的唯一根因；第2、3项的损失语义可以直接由代码确定。

## 2. 唯一方案及不变项

方案：**冻结 RT57 gamma=1.66 和 RT58 anchor；非输入站沿用 base-relative residual transport；对有同址有效本站波形的 query，新增独立 local residual head；用显式分组真值损失联合训练。**

保持：同一 checkpoint；seed42；8 个新 epoch；原 80% random /20% normal 采样；原 split、cutoff、台站数集合、query exclusion；MDN logits 与 sigmas；不解冻 DiTing/旧 adapter/TEAM。改变的是新 correction 参数化、优化目标与固定新 head 学习率日程，不是增加多轮候选搜索。

`random` 指 causal random-geometry 选择协议，不是只打乱相同输入集合的数组顺序。联合重排波形/坐标/mask 后结果必须不变。

## 3. Forward：两条参数隔离的路径

### 3.1 冻结量与单位

```text
b_j, s_j = frozen RT57 gamma=1.66 在 public query j 的 mixture mean / predictive sigma
b_i, s_i = SAME frozen base 在 selected input i 坐标的 mean / predictive sigma
a_i      = frozen RT58 AnchorHead([u_i,d_i,z_e])
e_i      = a_i - b_i
```

```latex
b_j=\mathbb{E}_{57,\gamma=1.66}[Y_j\mid X_{\le t}],\quad
b_i=\mathbb{E}_{57,\gamma=1.66}[Y_i\mid X_{\le t}],\quad
a_i=A_{58}(u_i,d_i,z_e),\quad e_i=a_i-b_i.
```

模型内部使用 train-normalized PGA；`PGA_mean=-1.1228067351176172`、`PGA_std=0.4312468402225416`。绝对量导出乘 std 加 mean，delta/transfer 只乘 std。不得把某个 MDN component mean 当 predictive mean。

### 3.2 观测路由不是协议路由

根据当前 forward 可见的 station/query coordinates 与有效波形 station mask，生成同址匹配。每一坐标分量 `atol=1e-6`、`rtol=0`，使用当前 model-coordinate units，仅用于身份匹配；不使用公里近邻半径。

- 唯一有效匹配：走 local。
- 无匹配：走 transport。
- 多个同址候选：不任取第一个；fallback transport，导出 ambiguous 标记与计数。
- 无有效输入：correction=0；无效 query：correction=0。

**禁止把 `causal_random_mask_applied`、input PGA、target_type、query 的未来触发状态传入 forward 路由。** protocol bit 只用于训练分组与日志。保持原正式评估用 target type/indices 的分类不变，另报 observable route 与正式 input 分类的对照；不能悄悄把 ambiguous input 改名成真实 non-input，也不能排除这些 targets。

### 3.3 local 路径：直接修正已观测站，而不是把它当远程 query

```text
h_i   = GELU(Linear(LayerNorm([u_i,d_i,z_e]), 128))
l_i   = Linear(GELU(Linear([h_i, a_i-b_i, b_i, log(s_i)], 128)), 1)
m_j   = b_j + l_i       (query j 唯一匹配 selected input i)
```

```latex
h_i=\mathrm{GELU}(W_h\mathrm{LN}[u_i,d_i,z_e]+c_h),\quad
\ell_i=H_{local}([h_i,a_i-b_i,b_i,\log s_i]),\quad m_j=b_j+\ell_i.
```

这里 `s_i` 是冻结 MDN 的 predictive sigma，不是输入标签统计。log 前仅做数值保护。所有旧特征 detach；local 参数与 transport 参数完全分离。标量 amplitude/base 特征在高维 LayerNorm 之后拼入，以免在高维输入拼接中弱化其直接通路。最后一层全零初始化；无额外乘法 gate，无 ReLU 限制 delta 符号。

该支路由 normal 中唯一同址 query 的真实 PGA 监督，**不拿 frozen anchor 当真值**，也不把 local 输出回灌 transport。初始输出严格等于 base；单独改变 local 参数不改变 non-input 输出，这是结构性质，不是经验期待。

### 3.4 transport 路径：保留 v2 的相对残差定义

```text
k_i    = tanh(LevelMLP([u_i,d_i,z_e]))
h_ji   = PairEncoder([u_i,d_i,z_e,z_new,q_emb_j,g_ji])
r_ji   = tanh(||q_j-x_i||/0.05) * ResidualMLP([h_ji,e_i,b_j-b_i])
w_ji   = masked_softmax(ScoreHead(h_ji), over valid selected inputs)
c_ji   = b_j + k_i*e_i + r_ji
D_j    = sum_i w_ji*(k_i*e_i+r_ji)
m_j    = b_j+D_j           (没有唯一同址观测的 query)
```

```latex
k_i=\tanh K([u_i,d_i,z_e]),\quad
r_{ji}=\tanh(\|q_j-x_i\|/0.05)R([h_{ji},e_i,b_j-b_i]),\quad
D_j=\sum_iw_{ji}(k_ie_i+r_{ji}),\quad m_j=b_j+D_j.
```

`k_i` 不依赖 query。空间项不再乘旧 RT58 final gate；没有凸融合假设。单站 non-input query 差分满足：

```text
m_j-m_k = (b_j-b_k) + (r_ji-r_ki)
```

```latex
m_j-m_k=(b_j-b_k)+(r_{ji}-r_{ki}).
```

模型参数：station_dim256、event/query_dim1000、hidden256、set_layers2、heads4、沿用10维 geometry。LevelMLP 与 ResidualMLP 末层全零；set_input/set_blocks/pair_encoder/score_head 从 RT58 同名/同形模块显式复制。旧 transfer/output_gate 不复制到新输出层。distance_scale0.05 是沿用当前归零因子，不是重新解释过的物理距离模型。

### 3.5 两条支路组合与 MDN

```text
m_j = b_j + I_j*l_match(j) + (1-I_j)*D_j
mu'_jk = mu_base_jk + (m_j-b_j)
alpha'_jk = alpha_base_jk
sigma'_jk = sigma_base_jk
```

```latex
m_j=b_j+I_j\ell_{match(j)}+(1-I_j)D_j,\quad
\mu'_{jk}=\mu^{base}_{jk}+m_j-b_j,\quad
\alpha'_{jk}=\alpha^{base}_{jk},\ \sigma'_{jk}=\sigma^{base}_{jk}.
```

`I_j` 是可见同址观测标志，不是 normal/random 标志。局地/空间路径之间不共享任何可训练参数。为维持优化隔离，梯度裁剪也按 local/transport 分别进行，不能让 local 的大梯度通过全局 clip norm 缩小整个 transport 更新。

固定 logits/sigmas 意味着 mixture predictive variance 不变；不能靠缩小 sigma 把“点云收拢”伪装成预测改善。本轮保留这一控制变量，概率校准失败如实记录。

## 4. b_i 的具体工程实现

此项不能让 Codex 用每个输入站单独重跑 FullModel 来敷衍。

新增仅供 RT59 使用的 `FrozenPGAContext` / query 解码接口，至少包含 station memory、station feature embedding、temporal tokens/masks、u/d、event embedding、coords center、可选 VS30/config 所需状态。

1. 原一次 waveform/station 编码产生 context；冻结量用 `torch.no_grad()`，旧模块持续 eval。
2. 公共 query 按原 RT57 路径解码，得到 base MDN、query readout embedding、temporal base/delta 等。
3. context 下把 selected input coordinates 作为**一个 query batch**解码，得到 b_i/s_i。只重复轻量 query readout，不重复 DiTing/station/event 编码。
4. query decoder 返回显式结构；不得依赖另一次调用覆盖后的 `_last_*`。抽取共享纯 helper 时，旧 flag=false 的调用顺序、输出和参数键必须保持原行为。
5. public Q 维与内部 S 维严格隔离：内部查询不进入 labels、target_valid、evaluation target count、正式 alignment keys，也不使 `self.n_pga_targets` 被临时改写。
6. 验证同址 public b_j 与内部 b_i 一致；冻结 base 的公共预测不能因多请求了内部 queries 改变；query 分批/排列不改变结果。
7. 实现只需要支持本次已解析 RT58 配置路径。遇到未支持 readout mode 显式 fail，不静默换 head；不能臆测配置继承。不得用 local_absolute_pred/anchor_pred 冒充 b_i。
8. 若复用旧模块产生诊断副作用，必须在 helper 内收集返回，并在完成后恢复公共 query 的诊断；测试覆盖所有实际被 eval export 消费的字段。不要笼统清空全部 `_last_*`。

冻结特征后仍需交给新头计算参数梯度，使用 no_grad 或 detach，不把整个 forward 包进 no_grad；也不在可能需要保存冻结输入张量以求参数梯度时任意使用 inference_mode。

## 5. Loss：真实分组、明确梯度方向

### 5.1 三个训练分组

```text
R  = causal_random_mask_applied 的有效 non-input queries
NI = normal 中具有唯一有效同址观测的 queries
NO = normal 中其余有效 queries（ambiguous 单列对照，不改正式 input 分类）
```

每组独立归一化，再加权：

```text
L_main = 0.50 * mean_R(phi)
       + 0.25 * mean_NI(phi)
       + 0.25 * mean_NO(phi)
```

```latex
L_{main}=0.50\overline{\phi}_R+0.25\overline{\phi}_{NI}+0.25\overline{\phi}_{NO}.
```

保留80/20采样不是“继续原 pooled loss”。这里明确给 normal 合计0.50的目标系数；它**不等于**实际梯度范数恰好占50%，也不增加 normal 样本数量。NI/NO 切分不会被 normal validation 的71%/29%比例自动支配。组内一条样本有几个 targets，仍按既定 pooled-target 定义计数。

### 5.2 点预测与概率损失

对 normalized mean error `e=m-y`：

```text
phi = NLL(final MDN, y)
    + 0.10 * SmoothL1_beta=1(e)
    + 0.10 * e^2
```

```latex
\phi_j=-\log p_{final}(y_j)+0.10\,S_1(m_j-y_j)+0.10(m_j-y_j)^2.
```

明确新增 MSE 项，使较大均值误差有随误差增大的梯度，而不是只提出 RMSE 改善门槛。保留 SmoothL1 和 NLL，不强制 slope=1，也不把 loss 改成唯 MSE。权重是本方案的预先指定工程起点，不是假称已经验证的最优值。

### 5.3 normal 保护：不惩罚真正改对的预测

禁用旧 `normal_replay_distillation_loss`。改为：

```text
G_j = max(0, SmoothL1_beta=1(m_j-y_j)
             - stopgrad(SmoothL1_beta=1(b_j-y_j)))
L_guard = 0.05 * [0.25*mean_NI(G) + 0.25*mean_NO(G)]
```

```latex
G_j=[S_1(m_j-y_j)-\operatorname{sg}S_1(b_j-y_j)]_+,\quad
L_{guard}=0.05(0.25\overline G_{NI}+0.25\overline G_{NO}).
```

这是训练标签参与的损失，不是推理时用真值挑选 base/student。student 比 base 更接近真值时 G=0，不阻止改善；变差时才加惩罚。它是训练正则，不是验证集永不退化的数学保证，也不是 teacher 已提供额外信息的证据。

### 5.4 transport 辅助项与梯度隔离

仅在 R/NO 的 query 上计算以下辅助项；**唯一同址 query 必须从所有 transport auxiliary 中排除**。

- relative transfer：weight0.20，SmoothL1 beta=0.20/PGA_std；
  target=`(y_query_raw-y_input_raw)/PGA_std - (b_j-b_i)`，input labels 只在此处使用。
- candidate：weight0.05，beta=0.20/PGA_std；c_ji 对 normalized query truth。
- final within-event query difference：weight0.40，beta=0.15/PGA_std，最多105个现有合法pairs；只用本组非唯一同址 queries，**不建立 NI–NO 跨支路 pair**。

完整目标：

```text
L = 0.50*[mean_R(phi)  +0.20*T_R  +0.05*C_R  +0.40*D_R]
  + 0.25*[mean_NO(phi) +0.20*T_NO +0.05*C_NO +0.40*D_NO +0.05*mean_NO(G)]
  + 0.25*[mean_NI(phi) +0.05*mean_NI(G)]
```

```latex
L=.50(\bar\phi_R+.20T_R+.05C_R+.40D_R)
 +.25(\bar\phi_{NO}+.20T_{NO}+.05C_{NO}+.40D_{NO}+.05\bar G_{NO})
 +.25(\bar\phi_{NI}+.05\bar G_{NI}).
```

T/C：先对每个 query 的合法 input station pairs 求均值，再对组内有合法 pairs 的 queries 求均值，避免多输入站自然得到更多 pair 权重；分母不是全部targets数。D：先对每个合法 realtime field 的pairs求均值，再对本组有效 fields求均值。所有分母独立记录，不能用一个计数复用三种 reduction。auxiliary 空集返回图连接的0。

禁用旧 anchor_absolute、temporal_residual、station_local、原 pooled distribution_mean、旧 within-event/anchor pair/distillation aggregator；RT59 新 loss 是替代其 PGA 部分，而不是再叠加一次。

### 5.5 DDP 必须实现全局分母

每个 rank 上不能分别 `.mean()` 然后让 DDP 平均，尤其每卡 batch8、normal 稀少时。

```text
L_rank,g = W * weight_g * local_sum_g / global_count_g
```

```latex
L_{r,g}=W\lambda_g\frac{\sum_{j\in g,r}\ell_j}{\sum_r n_{g,r}}.
```

W 是标准 DDP 梯度平均的 world size。仅 counts 与日志用 detach/all_reduce；本地 numerator 保留 autograd。所有rank按同一顺序参与collectives，不能仅有该组样本的rank进入。全局空组为0、不向其他组重新分配权重。本地空组保留零梯度图。当前方案不加梯度累积/自定义通信hook；如果实际环境已有，必须重新证明归约等价，不能照搬W因子。

NaN mask 必须在 loss 算术前处理；不能计算 NaN loss 后乘0。本地参考代码已覆盖不均匀两分片的梯度代数等价；**未运行真实多进程/多机 DDP**，仓库集成仍须局部验证。

## 6. 优化器、预算、冻结与兼容

- 全部旧参数与buffers冻结并eval，包括RT58 anchor；两个新支路仅在train时开启train。
- 新模块 prefix：`pga_anchor_residual_transport_head.`，内部 `local.*` 与 `transport.*`，无共享可训练层。
- load RT58 epoch8 weight-only：strict load仅允许新prefix缺失。默认关闭新flag时新模块不注册、不新增键。
- warm copy RT58 的 set/pair/score 仅在fresh initialization做；恢复RT59 checkpoint时绝不重新复制。
- 新 `freeze_mode=dual_anchor_residual_only`。
- optimizer算法、betas、eps、weight_decay取RT58实际resolved设置并记录；只有新prefix参与。两个参数组初始lr都5e-4。
- 不继承 random-val-only ReduceLROnPlateau 的决策；使用预先固定 epoch 日程：epochs1–4 lr5e-4，5–6 lr2.5e-4，7–8 lr1.25e-4。按已完成epoch恢复，checkpoint保存next-step日程状态。
- 梯度各支路独立clip norm=1.0；记录clip前后norm。不增加每步多次autograd.grad诊断或PCGrad。
- seed42，固定8新epoch；无epoch selection、无多seed/sweep、无自动续训。
- 每epoch记录R/NI/NO target与event/realtime计数、loss各项、branch梯度norm、lr、已有random validation scalar。normal正式validation只在epoch8一次，不新增每epoch全normal评估。
- 原80/20sampler与数据不改，因此解冻/协议切换不是本轮混杂变量；但新路由/损失/归约/日程是联合方案，不能独立给它们分配性能贡献。

## 7. 集成执行顺序

这三步是一个实现与一个训练任务，不是三轮实验。

1. 在新config/flag下落地冻结context解码、local/transport模块、observable route；通过zero-init/public-base一致性与旧配置回归。
2. 接入分组loss替代分支、全局分母、独立clip/fixed schedule、exports。确认normal labels改变loss和新头梯度但不改变forward；确认局地loss不会触及transport参数。
3. 提交代码并返回准确result commit、source manifest与launcher命令；经过既定代码审阅/用户执行授权后，只提交固定epoch8 train -> random/normal validation两作业。不要先提交smoke或旧任务重跑。

## 8. 验证、控制与判定

### 8.1 必须留在同一次新评估中的控制

在新epoch8 random和normal forward中保存frozen gamma=1.66 base与final，使比较严格配对；`delta=0` baseline不需另跑旧模型评估。normal附带local/transport的route与delta分解。R/NO只受transport影响、NI只受local影响；这一输出分解是结构审计，不是多个训练模型消融。

保留上一交接规定的一次random head-boundary feature-pairing control：同次缓存、固定correct base/event/query/geometry/masks，只roll有效selected u/d并重算anchor和新head，单站恒等。它只属于新头配对敏感性，不称为旧full-system waveform-roll，也不再追加额外roll job。

### 8.2 人群、指标与计数

random：预期9,681 realtime rows，75,654 non-input targets，1,310有有效targets events；空间>=5targets fields5,762；actual-one-station且>=5targets fields1,494。

normal：预期9,681 rows，1,383events，89,770targets；input63,651、non-input26,119。每个elapsed/actual-station-count/triggered/untriggered子组及空间有效统计分别导出真实event/target/field计数，缺数据不得硬填预期数。

主指标：MAE/RMSE/bias/R2/slope与intercept；NLL、Brier阈值-1.2log10(m/s²)、predictive sigma和coverage；误差P90/P95，|error|<=0.1/0.2dex比例。点估计仍为log-coordinate mixture mean，不能用physical mean或best component混淆。

真值–预测图必须相同目标、坐标界限和密度色阶；分别展示normal all/input/non-input与random，不删离群点。slope按prediction对truth的OLS方向，理想线y=x；range放大不是精度证据。误差尾部与时间分桶由同一次NPZ计算，不申请新diagnostics作业。

paired event_id bootstrap5000，seed20260915；按同一event保留重复时刻。区间不覆盖训练seed或validation反复选方案的不确定性。

### 8.3 go/no：保留v2双提升，并完整冻结判据

所有比较对同次same-protocol frozen gamma=1.66 base，candidate-minus-base为差值方向。以下全部满足才是本轮 GO_FOR_DEVELOPMENT；不是多数票。

原random门槛不放宽：
1. random MAE<=0.242；2. slope>=0.45；3. R2>=0.39；
4. >=5target mean per-field range ratio>=0.55；
5. actual-one-station同口径ratio>=0.25；
6. 同单站pairwise-delta MAE<=0.330；7. requested1s MAE<=0.253。

概率护栏沿用v2前瞻定义，不追溯用于RT58补PASS：random NLL<=0.22，random paired ΔBrier95%CI上界<=0，random coverage1/2相对base各绝对变化<=0.01。0.01只是工程retention预算；mixture mean±sigma不是一般MDN的精确nominal区间。normal NLL与Brier点估计各不高于同次base，coverage1/2变化各<=0.01；全部报告实际数值与CI。

normal必须真实改善：normal all-target和真实non-input两组的MAE与RMSE四项，paired Δ各95%CI上界<0；真实input MAE/RMSE点估计不得高于base。旧normal上限0.136/0.218仅保留为历史红线，不代替改善要求。

另要求random ΔMAE95%CI上界<0，以及single-station range absolute error点估计不高于base。normal all/non-input的|bias|与|slope-1|不应恶化；将其作为明确报告的联合诊断，不得以拉斜率代替MAE/RMSE。P90/P95/固定误差带是描述性辅助，不事后挑一项替代主判据。

原RT58历史“3PASS/6FAIL/1未确立”不重写；本v3实验定义必须在新训练前提交。未满足双提升只报告部分进展，不声称用户目标已完成。

## 9. 已执行的本地工作与边界

本次在独立sandbox生成：
- `rt59_dual_reference.py`：同址路由、独立local head、MDN mean shift、per-target truth loss/guard、DDP全局计数归约参考。
- `test_rt59_dual_reference.py`：16项单元测试。
- `rt59_dual_reference_verification.json` 与原始测试输出。

环境：Python3.13.5，PyTorch2.10.0+cpu；执行16 tests，0 failures，0 errors。

验证包括zero-init、旧输入detach、local/remote路由梯度隔离、联合station排列不变、路由的query分批不变、ambiguous fallback、invalid NaN mask、MDN variance不变、normal真值改善不受guard惩罚、组内重复不会改变组权重、缺失组不重分权重、两不均匀分片模拟DDP梯度等价、单站level差分消除。

**未验证**：实际RT57 cached decoder、真实MDN/encoder checkpoint加载、仓库DDP集成、GPU/Slurm、PGA性能。参考文件不是完整模型patch，不能直接替换 `gemini_models.py`；Codex应据此集成并加上仓库特有回归。

## 10. 风险和可证伪结论

- 冻结表示可能缺少继续改善所需信息；local学不到收益时应承认，不扩大解冻范围。
- random与normal non-input仍共享transport，分组优化不保证两个域梯度永不冲突。
- unique-location路由会区分“有本站观测”和“只查询空间位置”；不保证在匹配边界处形成严格连续物理场。重复坐标必须计数，不能靠label判断分支。
- fixed sigma可以限制概率校准；不得为点图改善牺牲NLL/Brier后称整体成功。
- fixed8epoch预算可能不足，但不能事后续训到过门槛。
- normal改进只相对current RT57 development base；不能自动宣称超过RT55 epoch32。
- 参考测试保证特定数值和梯度性质，不保证validation收益；真实性能必须由这一个新训练结果裁决。

## 11. 完整 AI-HANDOFF

```text
[AI-HANDOFF]
task_id: 20260915-rt59-dual-objective-transport-v3
supersedes: 20260915-rt59-anchor-residual-transport (v1/v2)
repo: rular099/team_pytorch
branch: rt58-waveform-anchor-transfer-field
base_commit: 40830acd475bd40347fef03427b52c4a04432dc3
evidence_commit: 37d97ad9ad67dfec11f188e2a350a8ab8b8b2056
status: PROPOSED_REPOSITORY_IMPLEMENTATION_WITH_TESTED_STANDALONE_HELPERS

goal:
  实施本文件第2–8节唯一联合方案：独立local residual用于同址观测，
  base-relative transport用于其他query；分组真值损失、MSE与truth-regret guard。
  同一固定checkpoint同时改善random和normal精度，禁止仅提升normal验收门槛。

verified_facts:
  - 本轮remote HEAD仍为base_commit，父提交为evidence_commit。
  - 本文件第1节列出已读源码区间与证据。
  - normal validation all/input/non-input为89770/63651/26119；
    RT58相对gamma166 base三组MAE均略退化。
  - 当前旧loss是pooled aggregation，normal distillation向旧mean靠拢。
  - 独立sandbox的16项新参考单元测试通过，非仓库/HPC/PGA验证。

inferences:
  - normal输入站需要独立局地修正通路，不能默认RT58 anchor优于其base。
  - 显式protocol/group归约与不惩罚改善的guard比单纯增加门槛更对应双提升。
  - 仍需实际validation确认，不能保证所有分组都会改善。

files_to_inspect:
  - AGENTS.md；docs/ai/PROJECT_CONTEXT.md；docs/ai/README.md；SESSION_SUMMARY.md
  - RT58 result handoff与reports/rt58_waveform_anchor_transfer_validation_20260915/
  - gemini_util_light.py；gemini_models.py；train_light.py；eval_checkpoint.py
  - RT55/RT56/RT57/RT58及其normal configs、实际resolved RT58 config
  - tests/test_rt58_waveform_anchor_transfer.py、相关RT57/mask/formal-eval tests
  - tools/run_rt58_waveform_anchor_transfer_slurm.sh
  - 随附rt59_dual_reference.py及test_rt59_dual_reference.py

constraints:
  - 首先核对worktree与HEAD。HEAD漂移先审diff，不reset、不覆盖Codex已有工作。
  - 若v1/v2已经开训，保留其定义/结果，不把旧run改名v3、不自动kill/restart；
    说明状态，v3作为不同实验定义等待实际执行授权。
  - 保留RT55/RT56/RT57/旧RT58的加载推理、config字节、默认行为和关闭开关schema。
  - 不改变data/split/sampler/cutoff/gamma；不解冻旧模型；不使用held-out test。
  - 不重复smoke、query diagnostics、旧formal validation或test；不加sweep/seeds。
  - labels/protocol bit不进forward；禁止oracle gating。
  - 本文第2–8节是本task唯一完整技术规范，不再并行遵循冲突的v1/v2方案。

proposed_change:
  - gemini_models.py：
      新use_pga_anchor_residual_transport flag默认false；
      新pga_anchor_residual_transport_head.local/transport模块；
      FrozenPGAContext与pure frozen query decoder；
      按本文件3.2节观测同址路由；只平移MDN means。
      老use_pga_anchor_transfer仍可构造供checkpoint加载；新flag时旧delta不应用。
  - train_light.py：
      freeze_mode=dual_anchor_residual_only；warm-copy仅fresh load执行；
      新rt59_grouped_objective替代旧PGA loss block，训练和验证传参明确；
      全局DDP分母；独立branch clipping；fixed8epoch lr schedule；
      记录分组counts/loss/branch grad/lr与provenance。
  - 新tools/rt59_dual_objective.py（可用轻量共享模块）：
      同址match、分组归约与per-element losses；
      基于参考函数集成，不复制整个FullModel或引入额外第三方依赖。
  - eval_checkpoint.py：
      新conditional val_rt59_* exports：base MDN/mean、input base mean/sigma、
      frozen anchor、local delta、level、relative transfer、weights、
      public applied delta、observed/ambiguous route和fixed-context rolled delta；
      原public alignment与target数量不变；真实target_type保持原语义。
  - 新configs：
      pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json
      及对应normal_validation companion。
      继承RT58但显式关闭全部被新objective替代的旧loss aggregator，防重复计费。
      runtime train80%random/20%normal；objective weights .50/.25/.25。
      具体所有loss数值、beta、reduction按第5节；lr/epoch按第6节。
  - 新tests/test_rt59_dual_objective_transport.py：
      集成参考16tests并增加真实tiny FullModel context/decode/checkpoint/export/DDP测试。
  - 新tools/run_rt59_dual_objective_transport_v3_slurm.sh：
      默认DRY_RUN；单train -> random/normal；feature-roll同random job。
  - 新tools/analyze_rt59_dual_objective_npz.py与docs/rt59_dual_objective_transport_v3.md。
      输出轻量README/summary/gates/group_metrics/CI与源码身份，原结果不改数字。

acceptance_checks:
  - local/transport无共享trainable参数；freeze旧params与buffers，持续eval。
  - zero-init等于gamma166 base；mean-only shift保留weights/sigmas/predictive variance。
  - 同址unique路由无label/protocol依赖；duplicates fallback且计数，invalid/empty安全。
  - local-only主/guard loss对transport梯度0，remote-only含aux对local梯度0；
    独立gradient clipping不能有跨branch统一缩放。
  - public与内部input query基线一致；internalquery不污染公共Q轴、diagnostics与exports；
    encoder只执行原每个station那次编码，不按internalquery重复。
  - 联合station permutation、publicquery顺序/分批不变；同坐标relative transfer=0。
  - 缺组/不均匀rank/DDP counts与梯度等价；valid NaN失败，invalid NaN在算术前屏蔽。
  - truth改善时regret=0、变差时>0；loss权重仅乘一次、beta dex/model不混。
  - inputlabels只在loss；改标签不改变forward；target metadata分类不改变推理路由。
  - strict旧checkpoint加载、新prefix whitelist、fresh/resume初始化、fixed lr恢复正确。
  - py_compile；新launcher bash -n；git diff --check；新unit tests与相关局部旧回归。
  - 不声称独立参考tests代替仓库集成或真实checkpoint回归。

hpc_followup:
  - 构造并记录RT58_BASE_CHECKPOINT与其SHA；compute node校验源checkpoint epoch8。
    外部冻结DiTing encoder/checkpoint_format/source hash按现有机制校验。
  - 路径全部通过WORKDIR/JAPAN_FULL_DATA_ROOT/RT58_BASE_CHECKPOINT/
    RT58_BASE_CHECKPOINT_SHA256/RT59_WEIGHT_PATH等变量；不复制旧私有路径默认值。
  - source manifest含全部被执行源码/共享helpers/继承configs/launchers。
    支持clean exact Git SHA或uploaded_sha256，不以文件名确认身份。
  - DRY_RUN=1默认，CONFIRM_RT59=1才可实提交；ACTION=train|eval|all，无test action。
    固定seed42/8新epoch；输入与输出都为epoch8时以task_id/run_id/source hashes区分。
  - 训练新空目录；full_model_last.pth即本轮epoch8；不挑best、不追加轮次。
  - 一个train，两个afterok validation；new random内缓存feature控制，无额外旧roll。
  - 新epoch8 paired baseline/final指标、样本数、组内分布、统一点图与bootstrap按第8节。
  - 缺文件报告dependency-blocked，不换checkpoint、不重训anchor、不造结果。
  - 没有实际执行授权则只完成代码、unit tests、commit与DRY_RUN交接。

risks:
  - 第10节全部风险：表示不足、remote双协议冲突、fixedsigma、querydecoder副作用、
    同址歧义、8epoch预算与development适应偏差。
  - 这是联合方案，不能从一次新训练拆解每个改动的独立贡献。

open_questions:
  - none requiring another scientific choice；实际路径/版本/hash由已有环境解析。
    实际训练提交仍服从项目既定代码审阅与用户授权流程。

required_return:
  - 按docs/ai/README.md返回CODEX-RESULT：task/base/result commit/branch、
    changed_files、verification(pass/fail/not run+原因)、compatibility、
    hpc_status(job IDs/真实状态)、remaining_risks、review_request。
  - 实施、单元测试、提交、超算执行、PGA结果分别报告。
[/AI-HANDOFF]
```

## 12. 来源索引

仓库来源均为 `rular099/team_pytorch@40830acd475bd40347fef03427b52c4a04432dc3`，精确路径/区间见第1节。RT58数值来源为 `reports/rt58_waveform_anchor_transfer_validation_20260915/README.md`、`summary.json` 和 `gates.csv`；它们是已提交报告，不等同于本轮独立复算NPZ。

PyTorch 官方文档（核对于2026-09-15）：`DistributedDataParallel` 关于梯度平均；`SmoothL1Loss` 关于beta与尾部线性语义；`Autograd mechanics` 关于freeze、eval、no_grad、inference_mode区分。本方案不要求升级HPC到网页最新版本；按集群现有版本记录和测试。随附参考实测版本为2.10.0+cpu。
