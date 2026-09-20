# RT59-v3 epoch-8 validation 统计口径更正

更新日期：2026-09-20（Asia/Shanghai）

本目录是
[`../rt59_dual_objective_transport_v3_validation_20260920/`](../rt59_dual_objective_transport_v3_validation_20260920/)
的追加勘误，不覆盖原始归档。它只使用已有的两份 RT59 epoch-8 formal validation
NPZ 和配套 metrics JSON，未重新 forward、未接触 held-out test，也未改变任何预测值。

## 1. 更正后的结论

严格按预声明规则，结论仍为 **NO_GO**：25 个 required gate 中 18 个通过、7 个失败；
另有 4 个 diagnostic gate，4 个均通过，但它们不参与 conjunctive GO 判定。旧报告把
29 项全部称为“必需门槛”是错误的。

更正不否定 RT59 已获得的 pointwise/probabilistic 收益，也不把历史失败追溯改为通过：

- random MAE 为 `0.237828`，相对同次 frozen base 改善 `-0.007749 dex`，95% CI
  `[-0.008785, -0.006701]`；NLL 与 Brier 同向改善。
- normal all/input/non-input MAE 和 RMSE 全部改善；normal non-input MAE 从
  `0.210371` 降至 `0.199301`。
- requested 1 s random MAE 为 `0.243391`，通过 `<=0.253` 门槛。
- 仍失败的 7 项是 random slope、两项 P95−P05 range ratio、单站 pairwise-delta
  MAE、random/normal 1-sigma coverage retention，以及单站 P95−P05 range absolute
  error 相对 base 的变化。

因此，RT59 的正确结论是：点预测和概率分数得到一致改善，但最终空间场仍被压缩，
不能宣布完整空间场目标达成。这也是 RT60 只再拟合最终 spatial readout 的直接依据；
已知 legacy gate 失败不构成阻止该单一机制实验的身份或 schema 冲突。

## 2. 两项统计口径修正

### 2.1 Field range

正式 range 指标恢复为每个 field 内、至少 5 个有效 target 的 NumPy 默认线性百分位
`P95 - P05`，然后对 field 等权平均。旧分析器使用的 peak-to-peak (`max - min`)
结果仍以显式 `ptp_*_diagnostic` 键保留，不能替代正式 gate。

| Population / metric | frozen base | RT59 final | final - base |
|---|---:|---:|---:|
| random all, P95−P05 range ratio | 0.510231 | 0.475336 | -0.034895 |
| random all, P95−P05 range abs. error | 0.442355 | 0.465251 | +0.022896 |
| actual-one-station, P95−P05 range ratio | 0.207285 | 0.178619 | -0.028666 |
| actual-one-station, P95−P05 range abs. error | 0.658562 | 0.680775 | +0.022213 |
| actual-one-station pairwise-delta MAE | 0.336126 | 0.336202 | +0.000076 |

共有 5,762 个满足 target 数条件的 random fields，其中 1,494 个属于 actual-one-station。
正式 range-ratio gate 分别要求 `>=0.55` 和 `>=0.25`，因此二者仍失败；单站 range
absolute error 的变化要求 `<=0`，也仍失败。

### 2.2 Gate inventory

`normal_all` 和 `normal_noninput` 的 absolute bias change 与 absolute slope-error change
共 4 项按原协议属于诊断保护项，不是 required gate。它们本次全部方向良好，但不能
用于抵消任一 required gate 的失败。完整逐项表见 [`gates.csv`](gates.csv)。

## 3. Fixed-context rolled control

已有 NPZ 同时保存 correct-route 与固定其余上下文、仅滚动有效 selected station
feature pairing 的输出，因此无需新增 forward。下表为 `rolled - correct`：

| Protocol / population | MAE change | RMSE change |
|---|---:|---:|
| random all | +0.001063 | +0.000497 |
| random multi-station | +0.001442 | +0.000715 |
| random single-station | 0 | 0 |
| normal all | +0.001572 | +0.001678 |
| normal multi-station | +0.001640 | +0.001822 |
| normal single-station | 0 | 0 |

多站错配会降低精度，单站滚动是恒等置换。这说明 RT59 确实使用了正确的 station-feature
pairing；它是机制控制，不是空间精度已经改善的证据。

## 4. 数据、协议与身份

- 两份 formal export 均为 `val`，事件 ID 实际覆盖 Japan 2000–2024；旧报告中的
  “Japan 2018 fixed validation”表述已更正。
- random：1,310 events、9,681 realtime rows、75,654 targets。
- normal：1,383 events、9,681 realtime rows、89,770 targets，其中 formal input
  63,651、non-input 26,119。
- formal input mask 与 observable route 逐元素一致；ambiguous route target 为 0。
- PGA 坐标为 `log10(m/s^2)`；point estimate 为 predictive mixture mean。
- uncertainty 为 event-ID clustered paired bootstrap，5,000 draws，seed `20260915`。
- RT59 implementation commit：
  `54fd63d623ac5837e16a5d175d34041b7488661e`。
- RT59 counter fix commit：
  `7e00824b3aab7a15286dfd9bf9a264b03080c1cd`。
- submitted source manifest SHA-256：
  `3e1164bc4fb0441fb33e5d8cd03aa1708416708d930b01e71c4a82fd3779715e`。

输入摘要：

| Artifact | SHA-256 |
|---|---|
| random NPZ | `4c035ef8a34b69be44dffa1f05486cd96afdebccef7fdcbccff8cbe030b2da5f` |
| normal NPZ | `3553cc017798c59de891702906bfb0d3f14560432809a8942a76b2cb50fe559a` |
| random formal metrics JSON | `0a27ee92eb69462d305d867170044def5e100f3c6bd016d8311acfca91ea16ba` |
| normal formal metrics JSON | `dc24e5eeb87a72a0f2b5dcd7ed366e8ea4839667b4323d181612ec642f194046` |
| resolved training config | `693f8338843cef803e594b1956a48e7301ce604d69998d5a3665a143a1743e94` |

`summary.json` 同时记录事件 ID digest、逐年事件计数、配置原始/脱敏 digest 和实现
provenance。若 event ID、单位、mask/count、base alignment 或 checkpoint identity 不一致，
新版分析器会 fail closed，而不是生成可判定的结果。

## 5. 归档文件

- [`summary.json`](summary.json)：完整机器可读结果与 provenance。
- [`gates.csv`](gates.csv)：25 required + 4 diagnostic 的逐项判定。
- [`field_metrics.csv`](field_metrics.csv)：逐 field 轻量统计，可独立复核 range 与
  pairwise 指标。
- [`group_metrics.csv`](group_metrics.csv)、[`paired_ci.csv`](paired_ci.csv)、
  [`strata_counts.csv`](strata_counts.csv)：总体、CI 和分层统计。
- `*_config.sanitized.json`：脱敏后的 resolved/input configs。
- [`truth_prediction_density.png`](truth_prediction_density.png)：truth/prediction 密度图。
- [`artifact_manifest.sha256`](artifact_manifest.sha256)：本目录归档摘要。
