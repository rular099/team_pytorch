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

## PGA target sampling strategy

Current:
  - `pga_targets` usually come from station coordinates that are already included in the input station set
  - this is good for "read out PGA more accurately when local waveform is available"
  - but it is weaker for true spatial generalization to unseen coordinates

Desired:
  - support both:
    1. better prediction at locations with waveform observations
    2. generalization to arbitrary spatial coordinates

Proposed:
  - use mixed PGA target sampling during training
  - sample part of `pga_targets` from observed/input stations
  - sample part of `pga_targets` from inactive or held-out stations (`pga_from_inactive=True` style)
  - optionally add non-station query coordinates if labels can be constructed, to better approximate continuous-space interpolation

Benefits:
  - preserves strong supervision for station locations with local waveform evidence
  - improves held-out station prediction
  - narrows the gap between current station-to-station supervision and true arbitrary-coordinate inference

Files to modify:
  - `gemini_util_light.py`: `PreloadedEventGenerator` target sampling logic
  - Config: expose ratios / modes for observed vs inactive vs arbitrary-coordinate PGA targets

## Two-stage training to reduce amplitude shortcut domination

Hypothesis:
  - PGA is strongly correlated with amplitude scale.
  - The current end-to-end objective lets the back-end exploit this shortcut too early.
  - `station_adapter` then receives weak credit and collapses.

Goal:
  - first train `station_adapter` to produce usable station-level waveform embeddings
  - then fine-tune the full event-level model

Suggested staged training:

### Stage 1: station adapter warm-up

Trainable:
  - `waveform_model[1]` (`station_adapter`)

Frozen:
  - `waveform_model[0]` (`ViTAdapter`)
  - TEAM transformer
  - `mlp_pga`
  - `output_model_pga`
  - other event-level heads

Auxiliary supervision:
  - add a small station-level PGA head on top of station embeddings
  - predict per-station scalar PGA with masked L1 or MSE
  - only use valid stations

Suggested defaults:
  - `phase1_epochs = 20~50`
  - auxiliary loss weight `= 1.0`
  - `lr = 3e-4`

### Stage 2: joint fine-tuning

Trainable:
  - `station_adapter`
  - TEAM transformer
  - `mlp_pga`
  - `output_model_pga`

Loss:
  - normal event-level PGA loss
  - optionally keep station-level auxiliary loss with small weight

Suggested defaults:
  - auxiliary loss weight `= 0.1`
  - `lr = 1e-4 ~ 3e-4`

Why this is useful:
  - directly tests whether current failure is mainly a credit-assignment problem
  - if stage 1 still collapses, adapter structure is likely the main bottleneck
  - if stage 1 works but stage 2 re-collapses, the back-end shortcut is likely the bottleneck

## QuakeFormer-informed follow-up improvements

Context:
  - QuakeFormer shows that absolute location information, relative spatial dependency, VS30, and seen/unseen station evaluation are tightly coupled.
  - Our current 512-event ablations already cover amplitude on/off, absolute vs relative coordinates, rel+abs fusion weights, and add vs concat coordinate fusion.
  - The next improvements should focus on proving whether gains come from transferable spatial/site information or from memorizing seen station/location priors.

### P0: add station / spatial holdout evaluation

Goal:
  - evaluate future-event performance separately on stations seen during training and stations held out spatially.
  - make absolute-coordinate, VS30, and site-effect conclusions interpretable.

Suggested protocol:
  - split stations into train/val/test station sets by coordinate, not only by station code.
  - train only with train-station records for the event subset.
  - evaluate on:
    - future events + seen stations
    - future events + unseen/held-out stations
  - keep the current event-wise split metrics for comparability.

Metrics to report:
  - MAE, RMSE, Corr, R2, Slope, Bias
  - strong-PGA bin bias
  - near-source residuals
  - seen-station vs unseen-station residual gap

Files likely to modify:
  - `train_light.py`: split construction / metadata filtering for station holdout
  - `eval_checkpoint.py`: separate metric reporting for seen vs unseen stations
  - optional analysis script for post-hoc station-holdout summaries

### P0: residual decomposition for event and site effects

Goal:
  - diagnose whether each model reduces repeatable source/site errors or only improves aggregate MAE.
  - determine whether absolute coordinates are learning useful site effects or overfitting station identity.

Suggested analysis:
  - total residual: `ln_pga_pred - ln_pga_true`
  - event residual: mean residual per event
  - station/site residual: mean within-event residual per station
  - compare station residual distributions for seen vs held-out stations
  - correlate station residual with VS30, elevation, station location, and distance bins when available

Use for:
  - b0-b17 comparison
  - later VS30 ablation
  - deciding whether absolute coordinate resolution should be reduced

### P1: VS30 feature ablation

Goal:
  - test whether explicit VS30 improves site generalization beyond absolute coordinates.
  - answer whether absolute position alone learns richer or poorer site information than VS30.

Minimum experiment matrix:
  - absolute coordinates only
  - relative coordinates only
  - absolute coordinates + VS30
  - relative coordinates + VS30

Important requirement:
  - VS30 must be available for both input stations and target PGA query locations.
  - Station/spatial holdout evaluation is required; event split alone is not enough to interpret VS30 gains.

Files likely to modify:
  - data conversion / metadata loading path: add VS30 to station metadata
  - `gemini_util_light.py`: include VS30 in generated station and PGA-target features
  - `gemini_models.py`: add a site-attribute embedding branch or extend target query embedding
  - configs: expose `use_vs30` and missing-value handling

### P1: absolute location embedding resolution ablation

Goal:
  - test whether high-frequency absolute coordinate embeddings help site-specific prediction or cause spatial overfitting.

Motivation:
  - QuakeFormer reports that high-resolution absolute positional encoding improves seen-station performance but can hurt unseen-station generalization and create spatial artifacts.
  - Our current minimum wavelength is fine-grained, e.g. `0.01` degrees in the PGA configs.

Suggested configs:
  - current wavelength setting
  - coarser minimum wavelength, e.g. `0.03`
  - coarser minimum wavelength, e.g. `0.05`
  - optionally `0.1` if spatial holdout shows clear overfitting

Interpretation:
  - if seen-station improves but unseen-station degrades, absolute PE is likely memorizing local/station priors.
  - if coarser PE preserves unseen-station performance with small seen-station loss, prefer coarser PE for regional generalization.

### P1: relative geometry inside TEAM self-attention

Goal:
  - move relative spatial dependency from only token embeddings / target readout toward the station self-attention mechanism.

Current status:
  - we support relative coordinate embeddings.
  - b7 adds distance bias in PGA target cross-attention.
  - TEAM station self-attention does not yet have RoPE or pairwise distance bias.

Candidate approaches:
  - add pairwise distance / relative-coordinate bias to TEAM self-attention scores.
  - add RoPE-style relative geographic encoding to station self-attention.
  - compare against b7 target-readout-only distance bias.

Expected benefit:
  - better propagation/path modeling among input stations.
  - less reliance on absolute location memorization.

Files likely to modify:
  - `gemini_models.py`: `MultiHeadSelfAttention` / Transformer block inputs and attention score bias
  - `train_light.py`: pass station coordinates or relative geometry into transformer blocks if needed
  - configs: expose `team_relative_attention` / distance-bias parameters

### P2: TEAM-level masked station pretraining

Goal:
  - pretrain the TEAM transformer on a task closer to final PGA prediction than current single-station pretrain.

Suggested task:
  - given a subset of stations with waveform / coordinates / optional PGA observations, predict masked target-station PGA.
  - vary station mask ratio to cover forecasting-like and interpolation-like settings.
  - compute loss only on masked targets.

Why:
  - QuakeFormer benefits from unified masked forecasting/interpolation training.
  - Our current single-station pretrain teaches station embeddings, but does not directly teach multi-station spatial propagation.

Risks:
  - requires careful data-flow changes.
  - may introduce leakage if target PGA observations are not masked consistently.
  - should be attempted after b0-b17 and station-holdout evaluation clarify the current bottleneck.

Files likely to modify:
  - `gemini_util_light.py`: masked station / masked PGA-target sampling
  - `train_light.py`: new pretrain phase before full-model training
  - `gemini_models.py`: optional intensity/PGA observation token embedding if observed PGA is used as input

### P2: probabilistic PGA output and calibration

Goal:
  - output both PGA prediction and uncertainty for confidence-aware forecasting.

Candidate approach:
  - keep `target_cross_attention` readout.
  - replace point-only output head with Gaussian or Student-t head: `mu`, `log_sigma`, optionally `nu`.
  - train with NLL and report deterministic metrics using `mu`.

Additional metrics:
  - NLL
  - coverage at nominal intervals
  - calibration curves
  - CRPS if implemented
  - uncertainty vs residual correlation

Important:
  - current `target_cross_attention` path is implemented for point output only.
  - this should be treated as a separate modeling/metric change, not mixed into the lightweight coordinate ablation.
