# RT58 Waveform-Anchor Transfer Field validation evidence

更新日期：2026-09-15（Asia/Shanghai）

本目录给 ChatGPT Project 提供 RT58 epoch 8 的轻量、可追溯结果材料。原始
object-array NPZ 共约 502 MB，保留在用户提供的 `../chaosuan_res/`，不提交 Git。
本文只讨论固定 validation，不使用 held-out test，也不从多个 epoch 中重新选择
checkpoint。

## 1. 结论先行

RT58 **没有通过预先声明的整体 go 条件**：10 个数值条件中 4 个通过、6 个失败。

- 相对 RT57 `gamma=1`，RT58 random MAE、RMSE、R2、NLL、场内排序和动态范围均有
  小幅改善。
- 但 RT58 的实验定义同时把冻结 RT57 residual scale 从 1 改成了 1.66。拆除这个
  已知收益后，相对同一 RT58 NPZ 内可精确重建的冻结 `gamma=1.66` base，新
  anchor-transfer correction 使 random MAE **变差 0.000458 dex**，event-cluster
  95% CI `[+0.000053, +0.000860]`。
- 新 correction 使 1 s MAE 改善 `0.002200 dex`，也使场内 Pearson 和
  pairwise-delta MAE略有改善；但 mean per-field range ratio 从 `0.510231` 降至
  `0.492254`，actual-one-station range ratio 从 `0.207285` 降至 `0.183340`。
- anchor 本身很准：43,118 个有效输入台站上 MAE `0.116664 dex`、相关系数
  `0.896430`。然而 ungated anchor field 的 query MAE 为 `0.283641 dex`，明显差于
  冻结 base 的 `0.245578 dex`；模型把 correction 压到绝对值中位数
  `0.009653 dex`。这支持预设 no-go 分支：当前瓶颈是 transfer/fusion（单台站时
  已不存在 station aggregation 歧义），不应先解冻 DiTing。

因此，RT58 checkpoint 有研究价值，但不应作为 RT57 `gamma=1.66` 的替代模型锁定。

## 2. 仓库、模型和协议

- repository：`rular099/team_pytorch`
- branch：`rt58-waveform-anchor-transfer-field`
- RT58 implementation commit：`7ff060806609b9c3d0425b939e7c75dd447ba20c`
- 本次分析开始时 branch HEAD：`0e8537513e623432339d84ae2525b588340b18c9`
- 超算上传源码 manifest（提交前 launcher 已报告验证通过）：
  `5618bc48b783545464f7308a3a8227bc2e22c418f0f72ebfd5ae55bca765bfe0`
- checkpoint：`full_model_last.pth`
- checkpoint metadata：`epoch=8`、`loss=1.3893789052963257`、
  `checkpoint_format=non_encoder_v1`
- seed：42；固定 8 个新 epoch；不做 epoch selection
- split：validation only
- PGA 坐标：`log10(m/s^2)`
- point estimate：MDN predictive mixture mean
- random protocol：9,681 realtime samples、1,310 events、75,654 non-input
  targets；deterministic random mask applied fraction 1.0
- normal protocol：9,681 realtime samples、1,383 events、89,770 targets
- uncertainty：5,000 次 percentile bootstrap，按 `event_id` 聚类；所有同一事件的
  重复实时时刻一起重采样

RT57 与 RT58 random NPZ 的事件顺序、target labels/validity/indices/types、query 与
station coordinates、station masks、requested elapsed time 以及 requested/selected
station counts 均精确相同。NPZ 重算的 RT57/RT58 point metrics 与各自 formal JSON 在
`rtol=atol=1e-12` 下匹配。

checkpoint 本体和训练标量日志不在用户上传的结果包中，因此本地不能重新核验
checkpoint SHA-256，也不能复原 8 个 epoch 的 loss curve。`loss=1.3893789` 包含新的
auxiliary objectives，不能与 RT57 checkpoint loss 横向当成同一指标。

## 3. 输入产物身份

| Artifact | SHA-256 |
|---|---|
| resolved training `config.json` | `0397478013f51aa342f23add32834f07a8a1181c201749d7e7f24a7b7f0b4b38` |
| normal formal metrics JSON | `b3fd5f41d125afe56fff5c9d8accdf959db9d92b4bf2bfb53f8dc97a026a3953` |
| normal NPZ | `4f3fc619cbce66e2f7a27edb321b02d8f4987655b40337e2c77a66d6959f34f3` |
| random formal metrics JSON | `c86be000e9d304c53da610612e6686e0d42cbee5d9ac1a63261740590ac8c1d3` |
| random NPZ | `51eb48dcce3fa32d3e4a567769b567f4bb3c1158cdb429336d50490c6e61995a` |
| random waveform-roll formal metrics JSON | `27be956dbef293f73bd309bf473e6711cbd46d40393d67a0ebb48e8bccd6dd5e` |
| random waveform-roll NPZ | `89a3ed238253b4f58ad64f9bb988355be9eff470c210adc387817112f11afd79` |
| 用户上传的完整 zip | `8516eee1f79697f9cfb0834dc718a911240df33fd9c8db31091a4f3b0bbcbd3f` |

## 4. Formal random validation

所有误差单位均为 `log10(m/s^2)`。

| Metric | RT57 gamma=1 | RT57 frozen gamma=1.66 | RT58 final |
|---|---:|---:|---:|
| MAE | 0.247857 | 0.245578 | 0.246036 |
| RMSE | 0.320040 | 0.318243 | 0.317724 |
| bias | 0.036834 | 0.034466 | 0.040806 |
| correlation | 0.607537 | 0.613545 | 0.616998 |
| R2 | 0.360340 | 0.367504 | 0.369566 |
| slope | 0.379428 | 0.400316 | 0.397270 |
| NLL | 0.220260 | 0.217530 | 0.214663 |
| Brier at -1.2 | 0.192239 | 0.189875 | 0.190384 |
| predictive sigma mean | 0.303748 | 0.303748 | 0.303748 |
| coverage 1 sigma | 0.673937 | 0.681762 | 0.678338 |
| coverage 2 sigma | 0.947604 | 0.948198 | 0.948925 |

`gamma=1.66` base 由 RT58 NPZ 的 `val_pga_temporal_final` 直接取得；RT58 final 满足：

```text
RT58_final = RT57_base_gamma1.66 + anchor_applied_delta
```

对应原始 LaTeX：

```latex
\mathrm{RT58}_{\mathrm{final}}
=\mathrm{RT57}_{\gamma=1.66}+\Delta_{\mathrm{anchor}}.
```

在全部 75,654 个有效 targets 上，等式的最大绝对数值误差为
`3.50e-7 dex`。

### 4.1 RT57 gamma=1 到 RT58 final 的严格配对差异

| Population | Targets | RT57 MAE | RT58 MAE | Delta | 95% event-cluster CI |
|---|---:|---:|---:|---:|---:|
| all/non-input | 75,654 | 0.247857 | 0.246036 | -0.001821 | [-0.002438, -0.001205] |
| triggered non-input | 33,601 | 0.232664 | 0.233438 | +0.000774 | [+0.000305, +0.001246] |
| untriggered | 42,053 | 0.259997 | 0.256102 | -0.003895 | [-0.004904, -0.002953] |

相对 `gamma=1` 的总体收益只有 `0.74%`，且 triggered targets 明确变差。

### 4.2 新 anchor correction 相对 gamma=1.66 base 的严格配对差异

| Population | Base MAE | RT58 MAE | Delta | 95% event-cluster CI |
|---|---:|---:|---:|---:|
| all/non-input | 0.245578 | 0.246036 | +0.000458 | [+0.000053, +0.000860] |
| triggered non-input | 0.230893 | 0.233438 | +0.002545 | [+0.002056, +0.003039] |
| untriggered | 0.257311 | 0.256102 | -0.001209 | [-0.001770, -0.000651] |

这表明最终 overall gain 来自 `gamma=1.66`，而不是来自 anchor head。anchor head 在
untriggered target 上有收益，但不足以抵消 triggered target 的退化。

### 4.3 Requested elapsed time

| Time | gamma=1.66 base MAE | RT58 MAE | Delta |
|---:|---:|---:|---:|
| 1 s | 0.255091 | 0.252891 | -0.002200 |
| 3 s | 0.231696 | 0.231861 | +0.000165 |
| 5 s | 0.231414 | 0.232098 | +0.000684 |
| 10 s | 0.248160 | 0.249136 | +0.000976 |
| 20 s | 0.255142 | 0.256400 | +0.001258 |
| 40 s | 0.246079 | 0.248176 | +0.002097 |
| 90 s | 0.247530 | 0.249651 | +0.002121 |

RT58 只在 1 s 明确改善；在 5、10、20、40、90 s 变差。1 s 的预设阈值以
`0.000109 dex` 的极小余量通过，不应描述为稳健越过阈值。

## 5. Spatial-field result

以下为至少 5 个有效 query targets 的 5,762 个 realtime fields，逐 field 等权平均。

| Metric | RT57 gamma=1 | gamma=1.66 base | RT58 final |
|---|---:|---:|---:|
| Pearson | 0.494516 | 0.500284 | 0.502256 |
| pairwise-delta MAE | 0.301876 | 0.298782 | 0.298348 |
| true P95-P05 range | 0.831276 | 0.831276 | 0.831276 |
| predicted P95-P05 range | 0.360140 | 0.398650 | 0.383494 |
| mean per-field range ratio | 0.464444 | 0.510231 | 0.492254 |
| range absolute error | 0.478105 | 0.442355 | 0.455552 |

相对 `gamma=1.66` base：

- Pearson `+0.001971`，95% CI `[+0.000525, +0.003285]`；
- pairwise-delta MAE `-0.000434`，95% CI `[-0.000754, -0.000117]`；
- range absolute error `+0.013197`，95% CI `[+0.012101, +0.014321]`，明确变差；
- mean per-field range ratio 下降 `0.017977`。

因此 RT58 略微改善空间排序和差分误差，但进一步压缩空间幅度。这些方向不能合并成
“空间建模整体改善”。

### 5.1 Actual selected-station-count = 1

至少 5 个有效 targets 的 actual-one-station fields 共 1,494 个。

| Metric | gamma=1.66 base | RT58 final | Delta |
|---|---:|---:|---:|
| Pearson | 0.322403 | 0.323969 | +0.001566 |
| pairwise-delta MAE | 0.336126 | 0.336713 | +0.000587 |
| predicted P95-P05 range | 0.161006 | 0.141997 | -0.019009 |
| mean per-field range ratio | 0.207285 | 0.183340 | -0.023945 |

单输入时不存在多 station weight aggregation，仍出现 range 收缩和 pairwise-delta
退化。因此至少一部分问题位于 station-to-query transfer 或 final fusion，而不只是
多台站聚合。

## 6. Anchor-transfer head 内部证据

### 6.1 Input-station anchor

| Population | Targets | MAE | RMSE | bias | correlation |
|---|---:|---:|---:|---:|---:|
| all valid selected inputs | 43,118 | 0.116664 | 0.168643 | +0.000499 | 0.896430 |
| actual one station | 2,690 | 0.146232 | 0.212001 | -0.026875 | 0.839154 |
| actual multi-station | 40,428 | 0.114697 | 0.165355 | +0.002321 | 0.900460 |

这证明 `AnchorHead([u_i,d_i,z_e])` 能从 causal waveform representation 推断最终
input-station PGA；它不是“anchor 自身学不会”的 no-go 分支。

waveform roll 后，多台站 anchor MAE 从 `0.114697` 增至 `0.293783`，说明 anchor
确实依赖正确配对的台站波形。单台站 roll 是恒等置换，因此数值不变。

### 6.2 Transfer、aggregation 和 final gate

| Diagnostic | Value |
|---|---:|
| all valid station-query candidate pairs | 365,691 |
| pair-candidate MAE | 0.255543 |
| pair-candidate RMSE | 0.328573 |
| ungated anchor-field MAE | 0.283641 |
| ungated anchor-field slope | 0.358581 |
| mean absolute anchor-field/base gap | 0.140976 |
| mean absolute applied delta | 0.016915 |
| median absolute applied delta | 0.009653 |
| inferred absolute gate mean | 0.105573 |
| inferred absolute gate median | 0.099190 |
| station-weight effective-count mean | 2.050500 |
| station-weight max-weight mean | 0.779561 |

导出的 gate 诊断由以下可观测量反推：

```text
                    anchor_applied_delta
inferred_gate = --------------------------------
                 anchor_field_mean - base_mean
```

只统计分母绝对值大于 `1e-5 dex` 的 75,650 个 targets。

对应原始 LaTeX：

```latex
\widehat g
=\frac{\Delta_{\mathrm{anchor}}}
{m_{\mathrm{anchor}}-m_{\mathrm{base}}}.
```

gate 的反推最大绝对值为 `0.450687`，station weights 的和与 1 的最大误差为
`1.55e-7`。从结果看，模型合理地识别到 ungated field 比 base 差并强烈抑制它；但这也
导致新 head 对最终输出的贡献很小。

## 7. Waveform-roll control

Random correct-pairing 与 waveform-only station roll 严格对齐：

| Metric | Correct | Rolled | Rolled - correct |
|---|---:|---:|---:|
| MAE | 0.246036 | 0.260295 | +0.014259 |
| RMSE | 0.317724 | 0.333396 | +0.015672 |
| R2 | 0.369566 | 0.305838 | -0.063727 |
| slope | 0.397270 | 0.372025 | -0.025245 |
| NLL | 0.214663 | 0.286743 | +0.072080 |
| Brier | 0.190384 | 0.203375 | +0.012991 |
| spatial Pearson | 0.502256 | 0.444912 | -0.057343 |
| spatial pairwise-delta MAE | 0.298348 | 0.313766 | +0.015418 |

总体 MAE penalty 的 event-cluster 95% CI 为 `[+0.012975, +0.015518]`。这证明完整
RT58 系统仍依赖正确 waveform/station pairing。

但 full roll 同时扰动冻结 RT57 base。固定 correct base、只替换 rolled anchor delta：

```text
counterfactual = correct_RT57_base_gamma1.66 + rolled_anchor_delta
```

对应原始 LaTeX：

```latex
\mathrm{counterfactual}
=\mathrm{base}_{\mathrm{correct},\gamma=1.66}
+\Delta_{\mathrm{anchor,rolled}}.
```

- MAE 仅增加 `0.000204 dex`，95% CI `[+0.000049, +0.000363]`；
- spatial Pearson 下降 `0.001719`，95% CI `[-0.002307, -0.001168]`；
- spatial pairwise-delta MAE 增加 `0.000572`，95% CI
  `[+0.000426, +0.000725]`。

内部 anchor 在 roll 后变化很大，而最终 anchor delta 的 mean absolute change 只有
`0.007514 dex`。因此 full-system roll penalty 主要仍来自冻结 RT57 路径；新 head
使用 station-specific waveform 的最终强度很弱。

## 8. Normal-validation retention

| Population / metric | gamma=1.66 base | RT58 final |
|---|---:|---:|
| all targets (89,770), MAE | 0.131016 | 0.131610 |
| all targets, RMSE | 0.195544 | 0.195316 |
| all targets, R2 | 0.711582 | 0.712255 |
| all targets, NLL | not recomputed here | -0.822084 |
| all targets, Brier | not recomputed here | 0.101325 |
| input targets (63,651), MAE | 0.098452 | 0.099104 |
| non-input targets (26,119), MAE | 0.210371 | 0.210826 |
| triggered non-input (4,003), MAE | 0.186254 | 0.188465 |
| untriggered (22,116), MAE | 0.214737 | 0.214873 |

相对 `gamma=1.66` base，normal all-target 和 non-input MAE 分别轻微变差
`0.000594` 和 `0.000455 dex`，但都远低于预设 retention 上限。相对原 RT57
`gamma=1`，RT58 normal all-target MAE 从 `0.131302` 变为 `0.131610`，non-input
MAE 从 `0.213383` 改善到 `0.210826`。

## 9. 预声明 go/no 审计

| # | Criterion | RT58 | Result |
|---:|---|---:|---|
| 1 | random MAE <= 0.242 | 0.246036 | FAIL |
| 2 | random slope >= 0.45 | 0.397270 | FAIL |
| 3 | random R2 >= 0.39 | 0.369566 | FAIL |
| 4 | >=5-target mean range ratio >= 0.55 | 0.492254 | FAIL |
| 5 | actual-one-station mean range ratio >= 0.25 | 0.183340 | FAIL |
| 6 | actual-one-station pairwise-delta MAE <= 0.330 | 0.336713 | FAIL |
| 7 | requested-1-s MAE <= 0.253 | 0.252891 | PASS, margin 0.000109 |
| 8 | random NLL <= 0.22 and calibration not materially degraded | 0.214663 | PROVISIONAL PASS |
| 9 | normal all-target MAE <= 0.136 | 0.131610 | PASS |
| 10 | normal non-input MAE <= 0.218 | 0.210826 | PASS |

第 8 条没有预先量化“materially degraded”：NLL 改善 `0.002867`，Brier 变差
`0.000509`，1-sigma coverage 下降 `0.003423`，2-sigma coverage 提高
`0.000727`。本报告暂记 provisional pass，请 ChatGPT 审查这一判断。即使把第 8 条
记为 pass，整体仍是 4/10，通过数不足。

## 10. 可支持与不可支持的解释

### 已验证事实

- RT58 epoch 8 的三项 validation 均完成，三者 checkpoint metadata 一致。
- anchor regression 准确并依赖正确波形配对。
- 新 correction 对 1 s、untriggered targets 和场内排序有小幅收益。
- 新 correction 相对 `gamma=1.66` base 使总体 MAE、triggered MAE、空间 range、
  one-station range 与 one-station pairwise error 变差。
- final correction 被 gate 显著抑制；固定 base 的 rolled-anchor penalty 很小。

### 合理推断

- RT58 证明“直接 waveform anchor”是可学习的，但当前 transfer field 不能把准确的
  input-site anchor 可靠转化成 query-site absolute PGA。
- 单输入台站结果排除了“只有 multi-station aggregation 才是瓶颈”的解释；
  transfer/fusion 至少也是瓶颈。
- 在重新设计 transfer/fusion 前，没有结果依据优先解冻 DiTing。

### 尚不支持

- 不能声称 RT58 优于预先指定的 `gamma=1.66` development baseline。
- 不能把 full waveform-roll penalty 归因于新 anchor head。
- 单 seed validation 不能支持 test/generalization 或统计稳健性声明。
- 没有 checkpoint 文件/训练标量，不能审计 8 epoch 内部轨迹或 checkpoint SHA。

## 11. 给 ChatGPT Project 的审阅请求

请读取本报告、同目录 `summary.json` 与 `gates.csv`，并按以下顺序回复：

1. `审查结论`：核对 gamma=1 与 gamma=1.66 贡献拆分、metric direction、聚类
   bootstrap、one-station 解释和 fixed-base roll 归因。
2. `RT58 go/no`：严格应用上述 10 条预声明条件；明确第 8 条是否可算 pass。
3. `根因判断`：anchor、station-to-query transfer、aggregation、gate/fusion 四者中，
   哪个是现有证据支持的首要瓶颈；不要把 full roll gap 误归因于新 head。
4. `现在保留什么`：决定 RT57 `gamma=1.66` 是否继续作为 development baseline，
   RT58 checkpoint 是保留为诊断性结果还是继续微调。
5. `下一步唯一高价值工作`：最多提出一个新实验，优先快速推进；不要要求重复 smoke、
   query-geometry diagnostics、formal test 或可直接从现有 NPZ 回答的检查。
6. `给 Codex 的下一轮 AI-HANDOFF`：若需改代码，给出完整、可执行 handoff，包含精确
   architecture/config/loss/trainability/compatibility/tests/Slurm/validation-only go-no
   要求。RT55/RT56/RT57 原加载与推理必须保持兼容。

## 12. 核心重现命令

RT57 `gamma=1` 到 RT58 的严格配对统计由已有工具生成：

```bash
python tools/analyze_random_geometry_full_npz.py \
  --baseline-npz '<RT57 random NPZ>' \
  --candidate-npz '<RT58 random NPZ>' \
  --baseline-metrics '<RT57 random metrics JSON>' \
  --candidate-metrics '<RT58 random metrics JSON>' \
  --baseline-name 'RT57 gamma=1 epoch6' \
  --candidate-name 'RT58 gamma=1.66 epoch8' \
  --output-prefix '<output prefix>' \
  --bootstrap-replicates 5000 \
  --seed 20260915 \
  --trusted-pickle-input
```

注意：该工具现有 Markdown decision prose 有已知硬编码方向词错误；本报告只使用其
JSON/CSV 数值和按 metric direction 判定的表格，不提交自动生成的错误 prose。
`gamma=1.66` base 直接读取 RT58 NPZ 的 `val_pga_temporal_final`；anchor/head
diagnostics 分别读取 `val_pga_anchor_*` exports。原始 NPZ 是用户提供的可信超算产物，
因为含 object arrays，只有显式 `--trusted-pickle-input` 才加载。
