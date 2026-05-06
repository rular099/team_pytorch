# PGA 模型路线设计建议

更新日期：2026-04-30

本文档总结当前 `team_pytorch` 项目在 PGA 预测模型设计上的路线判断，供后续开发者参考。背景信息见 `jieshao.txt` 和 `current_project_status.md`。

## 1. 当前问题判断

当前项目的主线是保留 TEAM 的多台站建模框架，将原始 waveform model 替换为 DiTing 预训练 encoder，并增加 DiTing-to-TEAM station adapter。

已有实验表明：

- Single-station model 是有效的，单台波形表征对 `pga`、`mag`、`epidist` 都有可学习信号。
- Full model 在最简单的 single input、same-station PGA 任务上输出塌成近似常数。
- 不使用显式振幅信息时，single-station 结果仍然可用，说明问题不太像是“波形里没有 PGA 信息”。

因此，当前更应怀疑的是 full model 中的 PGA 信息通路，尤其是：

- PGA target coordinate token 是否有效 attend 到 station token。
- Transformer / mask / padding 逻辑是否导致 target token 收不到 station 信息。
- PGA head 是否主要学到 bias，而没有使用 station embedding。
- DiTing station embedding 接入 TEAM transformer 后是否被归一化、坐标融合或初始化破坏。

不要因为 full model 坍塌就直接否定 DiTing backbone。现有证据更支持“DiTing 表征有效，但 TEAM 风格 PGA query readout 不稳定”。

## 2. 三类参考路线

### 2.1 当前 DiTing + TEAM 路线

当前 full model 的核心链路是：

```text
waveform
  -> DiTing encoder
  -> station adapter
  -> LayerNorm
  -> optional amplitude/scale injection
  -> coordinate embedding/fusion
  -> optional event token
  -> TEAM transformer
  -> mag / loc / pga heads
```

优点：

- DiTing 已经过多种下游任务验证，是强 waveform feature extractor。
- 可以保留 TEAM 对多台站、mag、loc、PGA 联合建模的整体框架。
- 当前代码中已经有 single-station pretrain、full model train、诊断 readout mode 等工程基础。

缺点：

- DiTing 输出分布、adapter、LayerNorm、坐标 embedding、event token、PGA query token、mask、loss/head 任一环节出问题，PGA 都可能塌成常数。
- 当前默认的 `query_transformer` readout 对训练稳定性要求较高。
- PGA target token 的信息路径较长：target token 需要通过 full self-attention 间接获取 station 信息。

### 2.2 GTEAM 路线

GTEAM 的主链路和 TEAM 很接近：

```text
waveform
  -> CNN waveform embedding
  -> station coordinate embedding
  -> event token + PGA target coordinate token
  -> Transformer encoder
  -> mag / loc / pga MLP heads
```

优点：

- 结构比当前 DiTing 接入方案简单。
- 和 TEAM 思路一致，target coordinate token + Transformer readout 是可行路线。
- 训练数据来自 DiTing 2.0 中国台网，任务形式和当前项目有参考价值。

缺点：

- waveform encoder 是 CNN/MLP，表达力通常弱于已有 DiTing backbone。
- 代码和文档中虽然提到 GNN，但核心 `WaveformFullmodel` 更接近 Transformer token 模型，并没有显式 station-station graph message passing。
- 不能直接解决当前项目里“强 station embedding 接入 full readout 后坍塌”的问题。

结论：GTEAM 可以作为 TEAM/PGA query 实现参考，但不建议直接替换当前方案。

### 2.3 GRAPES 路线

GRAPES 是 graph model 范式：

```text
raw station waveform
  -> per-station preprocess / encoder
  -> station graph construction
  -> graph convolution / message passing
  -> per-station PGA prediction
```

它的关键特点是显式使用台站间空间关系：

- 站点是图节点。
- 边由 kNN、距离阈值、小世界长边等规则构造。
- 边权可包含距离或经验衰减关系。
- 通过 graph convolution 在 station nodes 间传播信息。

优点：

- 对 PGA 空间场预测有很强的 inductive bias。
- station-station 空间传播路径明确，适合地震动场插值和传播。
- 对“输入若干台站，预测台站集合上的 PGA”很自然。

缺点：

- 原生目标更偏向站点 PGA，不是任意 target coordinate query。
- mag/loc 不是 GRAPES 的核心输出。
- 如果完全改成 GRAPES，会放弃当前已经验证有效的 DiTing single-station pipeline 和 TEAM 多任务框架。

结论：不建议短期推倒重来改成纯 GRAPES，但强烈建议借鉴它的空间归纳偏置。

## 3. 推荐路线

推荐的主路线是：

```text
DiTing station encoder
  -> station embeddings
  -> optional station-station transformer / graph-aware mixing
  -> target-to-station cross attention PGA head
  -> PGA prediction
```

也就是保留 DiTing + TEAM 的主框架，但将 PGA readout 从 full self-attention 中拆出来，改成更明确的 target-aware readout。

### 3.1 Target-to-station cross attention 是什么

它不等于把模型整体改成 graph model。它仍然可以是 Transformer/attention 架构的一部分。

核心形式：

```text
query = target coordinate embedding
key   = station embeddings
value = station embeddings
attention bias = f(target-station distance, azimuth, elevation, valid mask)
output = target-specific PGA embedding
```

再接：

```text
target-specific PGA embedding -> PGA MLP/head
```

与当前 `query_transformer` 的区别：

- 当前方式：把 station tokens 和 target tokens 拼成一个长序列，依赖 full self-attention 自动学会 target 从 station 取信息。
- 建议方式：让每个 target 显式地 attend 到所有有效 station，PGA 信息路径更短、更可控。

### 3.2 为什么预计效果更好

从效果和风险综合判断，优先级如下：

1. `DiTing + target-to-station cross attention`
   - 最适合当前目标：给定若干输入台站，预测任意位置 PGA。
   - 充分利用已验证有效的 DiTing station embedding。
   - 信息通路短，训练比 full self-attention query readout 更稳。

2. `DiTing + graph/cross-attention hybrid`
   - 长期上限可能最高。
   - 可以先做 station-station 信息传播，再做 target-to-station readout。
   - 工程和调参复杂度更高，适合在 cross attention 跑通后推进。

3. 原始 TEAM/GTEAM full self-attention query readout
   - 理论可行，但当前实验已在最简单 PGA 任务上坍塌。
   - 不建议继续只在这条链路上反复调参。

4. 纯 GRAPES graph model
   - 对站点 PGA 场预测有优势。
   - 但与当前任意 target PGA、mag、loc 联合任务不完全匹配。
   - 不建议短期作为主路线。

## 4. 建议实验顺序

### 4.1 先完成现有诊断实验

继续使用已有的三种 PGA readout mode：

- `direct_station`
- `query_no_transformer`
- `query_transformer`

判读方式：

- 如果 `direct_station` 都学不会，优先查 label、loss、PGA target valid、head 初始化、adapter 权重加载。
- 如果 `direct_station` 能学会，但 `query_transformer` 仍坍塌，问题集中在 transformer/query/mask/attention 链路。
- `query_no_transformer` 只用 target coordinate embedding，不应期待它在 same-station PGA 上表现很好，它主要用于排查 head 和 target embedding 是否异常。

### 4.2 新增 `target_cross_attention` readout

建议在当前 `pga_readout_mode` 中新增第四种模式：

```json
"pga_readout_mode": "target_cross_attention"
```

最小实现版本：

```text
station_feature_emb: (B, S, D)
pga_emb:             (B, T, D)
station_valid:       (B, S)
pga_target_valid:    (B, T)

pga_readout_emb = CrossAttention(
    query=pga_emb,
    key=station_feature_emb,
    value=station_feature_emb,
    key_padding_mask=station_valid
)

PGA = mlp_pga(output_model_pga(pga_readout_emb))
```

第一版先不要引入复杂图结构，只做 masked cross attention，目标是验证：

- target 是否能稳定从 station embedding 中取信息。
- same-station PGA 是否不再塌成常数。
- multi-target / holdout PGA 是否优于 `query_transformer`。

### 4.3 再加入空间先验

如果 `target_cross_attention` 跑通，再逐步加入：

- target-station 距离 bias。
- 方位角、相对高程、相对坐标 MLP bias。
- KNN mask，只让 target attend 最近的若干输入台站。
- 简单 GMPE/attenuation prior，例如把距离衰减作为 attention bias 或 residual feature。
- travel-time 或 P/S 到时相关特征。

注意：这些增强应逐项 ablation，不要一次性全部加入。

### 4.4 最后考虑 graph-aware hybrid

如果 cross attention 已经稳定，再考虑引入 GRAPES 风格的 station-station 空间传播：

```text
station embeddings
  -> station-station graph/attention mixing
  -> target-to-station cross attention
  -> PGA head
```

可选实现：

- 基于距离 bias 的 station self-attention。
- KNN graph message passing。
- Edge-conditioned graph convolution。

这一步属于长期增强，不应阻塞当前问题诊断。

## 5. 工程实现注意点

实现 `target_cross_attention` 时建议注意：

- 明确 mask 语义：`True` 到底表示 valid 还是 masked，必须和当前 transformer 代码保持一致。
- 输出诊断量：记录 target attention 到 station 的质量，例如 entropy、top-k station distance、valid station attention mass。
- 先用 point output 跑通，再考虑 MDN output。
- same-station PGA 任务是必要 sanity check：输入台站和 target 是同一站时，模型至少应能接近 single-station head 的表现。
- 不要过早解冻 DiTing encoder。先确认 adapter/readout 链路可学，再决定是否 fine-tune backbone。
- 保留 mag/loc 的 event token 路线，不必强行让 PGA 和 mag/loc 共用完全相同的读出。

## 6. 总结结论

当前最合理的方向不是替换 DiTing，也不是立即改成纯 Graph model，而是：

```text
保留 DiTing station encoder，
保留 TEAM 多任务框架，
将 PGA readout 改成 target-to-station cross attention，
随后逐步加入 GRAPES 风格的空间先验。
```

这条路线同时利用了：

- DiTing 的强单台 waveform 表征。
- Transformer/cross-attention 对任意 target query 的灵活性。
- GRAPES 对 PGA 空间场预测的物理/几何归纳偏置。

从当前证据看，它比继续死磕原始 `query_transformer` 更稳，也比直接重写成 graph model 风险更低。
