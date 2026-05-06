# Japan P Pick Strategy

本文档总结 `build_japan_training_data.py` / `japan_dataset_builder.py` 当前的日本数据 P pick 生成策略。

## 默认配置

默认训练标签写入 `p_picks`，当前默认参数为：

```bash
--pick_mode travel_time
--travel_time_model jma2001a
--final_pick stalta
--target_sampling_rate 100
--jma_search_margin_seconds 10
--jma_search_margin_per_km 0.03
--jma_search_margin_max_seconds 60
--fallback_search_half_window_seconds 60
--stalta_sta_seconds 0.2
--stalta_lta_seconds 1.0
--stalta_threshold_ratio 2.5
--stalta_feature vertical
--stalta_highpass_hz 0.5
```

`--run_diting` 默认不开。只有开启后才会额外生成 DiTing acceleration / velocity picks。

## 原始 trigger

原始数据中的 `Record Time` 被转换为台站记录中的 trigger，保存为：

- `p_pick_trigger_aligned`
- `p_pick_observed_aligned`

这个 trigger 只作为参考和诊断字段，不作为默认最终 P pick。已有样本显示 trigger 可能在有效波形内，但明显不是真实 P 到时。

## 理论 P 到时

默认使用 `JMA2001A` 走时表计算理论 P 到时：

- 正常输出为 `p_pick_predicted_aligned`
- 如果 JMA 表不可用或网格越界，则 fallback 到 `ak135`
- 实际使用模型记录在 `p_pick_travel_time_model_used`
- JMA 越界标记为 `p_pick_jma_grid_clipped`
- ak135 fallback 标记为 `p_pick_ak135_fallback`

可选模式：

- `--travel_time_model jma2001a`
- `--travel_time_model ak135`
- `--travel_time_model constant`

`constant` 模式使用常数 P 波速度和速度上下限生成搜索窗。

## 理论搜索窗

JMA/ak135 模式下，理论搜索窗不是固定 `travel_pred +/- 10s`，而是随震中距放宽：

```text
margin = min(
    jma_search_margin_max_seconds,
    jma_search_margin_seconds + epicentral_distance_km * jma_search_margin_per_km
)

raw_search = [travel_pred - margin, travel_pred + margin]
```

默认值下：

```text
margin = min(60, 10 + epicentral_distance_km * 0.03)
```

相关输出字段：

- `p_pick_search_raw_left_aligned`
- `p_pick_search_raw_right_aligned`
- `p_pick_search_margin_seconds`
- `p_pick_travel_time_fast_aligned`
- `p_pick_travel_time_slow_aligned`

实际搜索窗还会裁剪到可用数据范围：

```text
record_start <= pick <= min(valid_end, pga_time - min_margin_seconds)
```

其中 `min_margin_seconds` 当前内部默认是 `0.5s`。这可以避免 P pick 落到无效 padding 区、有效记录末尾之后，或 PGA 之后。

如果理论搜索窗和可用数据范围相交：

```text
p_pick_search_source = travel_time_window
```

如果理论搜索窗完全不相交，则先把理论点 clip 成 `travel_coarse`：

```text
travel_coarse = clip(travel_pred, record_start, min(valid_end, pga_time - 0.5s))
```

再围绕它生成 fallback 搜索窗：

```text
fallback_search = travel_coarse +/- fallback_search_half_window_seconds
```

再裁剪到可用数据范围，标记为：

```text
p_pick_search_source = clipped_travel_time_fallback
```

默认 fallback 半窗为 `60s`。

## Travel Coarse

`p_pick_repaired_aligned` 在图例中显示为 `travel_coarse`。它不是原始理论到时，而是理论到时经过数据边界约束后的粗 pick：

```text
p_pick_repaired_aligned = clip(
    p_pick_predicted_aligned,
    record_start,
    min(valid_end, pga_time - 0.5s)
)
```

因此 `travel_pred` 和 `travel_coarse` 可能不一致。不一致通常说明理论 P 到时落在有效记录外、PGA 后，或过于接近 PGA。

## STA/LTA Refinement

默认最终 pick 是 STA/LTA refined pick：

```bash
--final_pick stalta
```

STA/LTA 在实际搜索窗内运行：

- 默认特征：`vertical`，即 UD 分量绝对值
- 可选：`norm`
- 默认 STA 窗：`0.2s`
- 默认 LTA 窗：`1.0s`
- 默认阈值：`2.5`
- 默认高通：`0.5 Hz`

流程：

1. 取台站有效记录段。
2. 对有效记录段做高通，减少低频大振幅对 STA/LTA 的干扰。
3. 按 `vertical` 或 `norm` 生成 characteristic function。
4. 计算 STA/LTA ratio。
5. 在搜索窗内寻找第一个超过阈值的位置。
6. 如果没有超过阈值，则取搜索窗内 ratio 最大值。

输出字段：

- `stalta_refined_pick_aligned`
- `stalta_method`
- `stalta_ratio_peak`
- `stalta_ratio_at_pick`
- `stalta_search_left_aligned`
- `stalta_search_right_aligned`
- `stalta_highpass_hz`
- `stalta_boundary_mode`
- `stalta_boundary_warmup_search`

### 数据开头的 Boundary Pick

当搜索窗贴着有效记录开头时，P 到时可能已经在数据开始前或非常靠近开头。此时没有足够的 P 前 LTA 历史。

默认允许 boundary pick：

```bash
# 默认开启
--no_stalta_boundary_pick  # 可关闭
```

开启时，如果搜索窗从 LTA warm-up 区开始，代码不再强制把前 `lta_seconds` 的 ratio 清零。若最终 pick 落在初始 LTA warm-up 区，则：

```text
stalta_method = stalta_boundary_threshold
```

或：

```text
stalta_method = stalta_boundary_argmax
```

这表示 pick 是“最早可观测起跳附近”的估计；如果真实 P 到时早于记录开始，代码无法恢复缺失的起跳前波形。

## DiTing Picks

加 `--run_diting` 后会额外生成两组 DiTing pick：

- `p_pick_diting_acc_aligned`
- `p_pick_diting_vel_aligned`

同时保存概率/分数：

- `diting_acc_score`
- `diting_vel_score`
- `diting_acc_probability`
- `diting_vel_probability`

以及错误状态：

- `diting_acc_error`
- `diting_vel_error`

默认 DiTing 阈值：

```bash
--diting_p_th 0.1
--diting_s_th 0.1
--diting_d_th 0.3
```

其中 `diting_d_th` 对应 detection threshold。

### DiTing 输入窗口

DiTing 输入采样率为 `100 Hz`，模型窗口长度为 `100s`。

当前策略是：

1. 先取整条有效台站记录。
2. 对整条有效记录做 detrend 和减均值。
3. acceleration 模式：直接在整条有效记录预处理后截取 search window。
4. velocity 模式：
   - 对整条有效记录做高通，默认 `--velocity_highpass_hz 0.05`
   - 对整条有效记录积分成速度
   - 再 detrend 和减均值
   - 然后截取 search window
5. 将 search window 放到 100s DiTing 输入窗口尾部，前面补 0。

如果 DiTing 返回的 pick 落在 search window 外，记录错误：

```text
p_pick_outside_search_window
```

如果没有 P pick，记录：

```text
no_p_pick
```

## Final Pick 选择

最终训练标签统一写入：

```text
p_picks = p_pick_refined_aligned
```

由 `--final_pick` 决定来源：

| `--final_pick` | 最终 pick 来源 | 缺失时 fallback |
| --- | --- | --- |
| `travel_time` | `p_pick_repaired_aligned` | 无 |
| `stalta` | `stalta_refined_pick_aligned` | 无 |
| `diting_acc` | `p_pick_diting_acc_aligned` | `stalta_refined_pick_aligned` |
| `diting_vel` | `p_pick_diting_vel_aligned` | `stalta_refined_pick_aligned` |

注意：为了 HDF5 中保存整数数组，`p_pick_diting_acc_aligned` / `p_pick_diting_vel_aligned` 的缺失值写入时会填成 `p_pick_repaired_aligned`。判断 DiTing 是否真的成功，应查看：

- `diting_acc_error`
- `diting_vel_error`
- `diting_acc_probability`
- `diting_vel_probability`

## Diagnostics

如果设置 diagnostics 输出，会生成：

- `station_pick_differences.csv`
- `pick_difference_summary.csv`
- `pick_search_range_summary.csv`
- `pick_difference_summary_table.png`
- 各 pick 相对 travel coarse 的差异直方图
- 随机抽样波形人工核查图

QC 波形图当前显示：

- 上图：UD 加速度，单位 `m/s^2`
- 下图：UD 速度，单位 `m/s`

图中会标出：

- trigger
- travel_pred
- travel_coarse
- pick_search
- stalta
- diting_acc / diting_vel 及概率
- final
- PGA

标题包含：

- event id
- wave index
- station
- magnitude
- epicentral distance
- DiTing probability
- search source

## Small Sample Test

小样本测试可用：

```bash
python build_japan_training_data.py \
  --waveform_root /path/to/waveformsnew \
  --output_dir /tmp/japan_test \
  --year 2024 \
  --num_events 1 \
  --overwrite
```

`--num_events` 是 `--limit_events` 的别名，当前含义是“每个 year 处理前 N 个 event tar”，不是最终写出的 station 行数。

