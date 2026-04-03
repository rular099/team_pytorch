# TODO

## Dynamic waveform windowing

Current: HDF5 stores fixed-length waveforms (10000 samples), cutout zeros out samples after a time point.
70% of diting input is zeros, which may hurt feature extraction.

Proposed: HDF5 stores full-length waveforms (variable, can be >> 10000 samples).
Generator dynamically extracts a 10000-sample window at training time:
  1. Randomly sample a cutout time (simulating "current moment" in EEW)
  2. Extract [cutout - 10000, cutout] as the input window
  3. All stations share the same cutout time (absolute time alignment)
  4. Stations whose P-pick > cutout get zeroed (no signal yet)

Benefits:
  - diting sees dense signal instead of mostly zeros
  - More diverse training samples (different windows from same event)
  - Better matches real-time inference (sliding window over incoming data)

Files to modify:
  - `convert_japan_data.py`: store full-length aligned waveforms (remove time_after cap)
  - `gemini_util_light.py`: `PreloadedEventGenerator` cutout logic, extract window instead of zeroing
  - Config: remove `noise_seconds`, use window-based cutout parameters instead

## Diting-TEAM 接入方式改进

Current: diting 整体作为 waveform_model，经过 EncoderFeatures (emg任务头 + GAP) 输出
(B, 256) 全局向量，再通过单个 nn.Linear(256, 500) 映射到 TEAM embedding 空间。
冻结时只有 dt2team 可训练。

问题:
  - GAP 丢失时间维度信息，不利于 station-level 的 PGA 预测
  - emg 任务头为震级回归优化，不一定保留 PGA 所需的台站区分信息
  - 单线性层表达能力不足

### 方案 A: 绕过 EncoderFeatures，直接使用 encoder 输出

取 ViTAdapter 输出的 patch 级特征 x (B, 200, 512)，用新的 adapter 替代
EncoderFeatures + dt2team。保留时间维度信息。

实现要点:
  - 修改 waveform_model 的构建，只保留 ViTAdapter (encoder)
  - 新增 adapter: 例如 1D conv + pooling + MLP，将 (B, 200, 512) → (B, 500)
  - 需要处理 ViTAdapter 返回 list 的问题（取最后一个元素 x）

### 方案 B: 添加 adapter 结构

保留 EncoderFeatures 但不使用 emg 任务头，添加新的 adapter 模块。
可参考 diting 中 InteractionBlock 的设计，用非线性变换替代纯线性层。

实现要点:
  - 在 EncoderFeatures 后添加 MLP adapter (如 256 → 512 → 500, with ReLU)
  - 或在 FPN 特征上添加新的 task-specific 分支
  - 可选择性解冻 adapter 相关层

### 方案 C2: 重新初始化 FPN + bottleneck

如果只重新初始化 projlast_emg 效果不好，进一步将 FPN 横向连接和 bottleneck 也重新初始化。
FPN 虽然是通用多尺度融合结构，但预训练权重仍可能带有震级任务的偏置。

实现要点:
  - EncoderFeatures 继承自 UPerHead，FPN 相关层包括:
    - `lateral_convs`: 横向 1x1 Conv1d，将 encoder 各尺度投影到 out_channels
    - `up_convs`: ConvTranspose1d，自顶向下上采样
    - `fpn_bottleneck`: Conv1d，将拼接后的多尺度特征降维
    - `psp_bottleneck`: PSP 模块的 bottleneck
  - 用 kaiming_normal_ 重新初始化上述层的 weight，zeros_ 初始化 bias
  - 在 train_light.py 冻结 encoder 之后、创建 optimizer 之前执行

### 方案 C: 解冻 EncoderFeatures 层

保持 backbone (ViTAdapter) 冻结，解冻 EncoderFeatures 让其为 TEAM 任务适配。
同时将 dt2team 从线性层改为小型 MLP。

实现要点:
  - 修改 train_light.py 冻结逻辑，只冻结 waveform_model[0] (encoder)
  - 解冻 waveform_model[1] (EncoderFeatures) 和 dt2team
  - dt2team 改为 MLP (256 → 512 → 500, with ReLU)
  - 注意学习率：adapter/head 可用较大 lr，encoder 冻结或极小 lr
