# Stage2 512-Event b19-b24 Experiment Plan

Date: 2026-05-13

This document describes the next 512-event PGA ablation batch after b18. It is intended as a handoff note for whoever analyzes the new results after the supercomputer jobs finish.

## Motivation

The previous stage2 results showed:

- b3, `pga_norm_noamp`, is still the best global run.
- b18, which combined PGA normalization, amplitude, absolute-coordinate concat, and PGA-heavy single-station pretraining, did not beat b3.
- Coordinate `concat` is no longer part of this follow-up. This batch fixes coordinate encoding to absolute coordinates with additive fusion.
- Loss comparisons must be redone under PGA target normalization because the Huber delta now operates in normalized PGA units.

The purpose of b19-b24 is to ask clean one-factor questions around the b3 anchor.

## Fixed Anchor

All b19-b24 configs are derived from:

```text
pga_configs/transformer_japan_overfit_pga15_stage2_512_b3_pga_norm_noamp_chaosuan.json
```

The following settings must remain fixed:

```text
overfit_n = 512
deterministic_sampling = true
overfit_event_ids_path = pga_configs/stage2_512_event_ids.txt
pga_target_normalization.enabled = true
pga_target_normalization.mean = auto
pga_target_normalization.std = auto
use_coords_abs = true
use_coords_rel = false
use_coords_rel_abs_fusion = false
coord_fusion_mode = add
output_distribution = point
pga_readout_mode = target_cross_attention
event_readout_mode = event_cross_attention
```

Do not compare these runs to concat or relative-coordinate experiments as if they were part of the same factor sweep. This batch intentionally removes coordinate-fusion structure as a variable.

## Experiment Matrix

| Exp | Config | Question | Changes from b3 |
|---|---|---|---|
| b19 | `transformer_japan_overfit_pga15_stage2_512_b19_pga_norm_amp_abs_add_chaosuan.json` | Does explicit amplitude still help after PGA normalization? | `use_amplitude_info=true` |
| b20 | `transformer_japan_overfit_pga15_stage2_512_b20_pga_norm_pga08_noamp_chaosuan.json` | Does PGA-heavy single-station pretraining still help after PGA normalization? | single-station weights `{mag:0.1, epidist:0.1, pga:0.8}` |
| b21 | `transformer_japan_overfit_pga15_stage2_512_b21_pga_norm_amp_pga08_chaosuan.json` | Do amplitude and PGA-heavy pretraining combine constructively without concat? | b19 + b20 changes |
| b22 | `transformer_japan_overfit_pga15_stage2_512_b22_pga_norm_mse_noamp_chaosuan.json` | What does MSE do in normalized PGA space? | `full_model_loss=mse` |
| b23 | `transformer_japan_overfit_pga15_stage2_512_b23_pga_norm_huber05_noamp_chaosuan.json` | Does a more robust normalized Huber loss improve average or strong-PGA behavior? | `full_model_loss=huber`, `full_model_huber_delta=0.5` |
| b24 | `transformer_japan_overfit_pga15_stage2_512_b24_pga_norm_huber2_noamp_chaosuan.json` | Does a wider normalized Huber loss move toward MSE-like behavior? | `full_model_loss=huber`, `full_model_huber_delta=2.0` |

There is intentionally no Huber3 run in this batch. In normalized PGA space, Huber3 is expected to be close to MSE unless residuals exceed roughly 3 normalized units.

## How To Run

Before launching training, verify the shared event list exists:

```bash
wc -l pga_configs/stage2_512_event_ids.txt
head pga_configs/stage2_512_event_ids.txt
```

The line count should be 512.

Suggested launch commands:

```bash
bash train_light_slurm.sh pga_configs/transformer_japan_overfit_pga15_stage2_512_b19_pga_norm_amp_abs_add_chaosuan.json
bash train_light_slurm.sh pga_configs/transformer_japan_overfit_pga15_stage2_512_b20_pga_norm_pga08_noamp_chaosuan.json
bash train_light_slurm.sh pga_configs/transformer_japan_overfit_pga15_stage2_512_b21_pga_norm_amp_pga08_chaosuan.json
bash train_light_slurm.sh pga_configs/transformer_japan_overfit_pga15_stage2_512_b22_pga_norm_mse_noamp_chaosuan.json
bash train_light_slurm.sh pga_configs/transformer_japan_overfit_pga15_stage2_512_b23_pga_norm_huber05_noamp_chaosuan.json
bash train_light_slurm.sh pga_configs/transformer_japan_overfit_pga15_stage2_512_b24_pga_norm_huber2_noamp_chaosuan.json
```

For manual evaluation, use the trained directory config, not the original config:

```text
<weight_path>/config.json
```

This is required because PGA normalization writes the actual mean and std into the trained config.

## Analysis Protocol

For each experiment, collect both:

```text
eval_results_best.txt / eval_results_best.npz
eval_results_last.txt / eval_results_last.npz
```

Report at least:

```text
Val MAE
Val RMSE
Val Corr
Val R2
Val slope
Val bias = mean(pred - label)
Train MAE
Train slope
```

Also report strong-PGA slices:

```text
Top 20% validation PGA labels: MAE, bias, slope
Validation labels >= -1: MAE, bias
Validation labels >= 0: MAE, bias, if enough samples exist
```

The `label >= 0` slice may be too small to interpret. In the current copied results there were only 2 validation targets in that bin.

## Comparisons To Make

Use b3 as the primary anchor:

```text
b3_pga_norm_noamp
```

Primary comparisons:

- b19 vs b3: isolated amplitude effect under PGA normalization.
- b20 vs b3: isolated PGA-heavy pretrain effect under PGA normalization.
- b21 vs b19 and b20: whether amplitude and PGA-heavy pretraining combine constructively.
- b22 vs b3: normalized MSE versus normalized Huber1.
- b23 vs b3: normalized Huber0.5 versus normalized Huber1.
- b24 vs b22 and b3: whether Huber2 behaves closer to MSE or to Huber1.

Secondary comparisons:

- b21 vs b18: effect of removing concat from the previous combined run.
- b22/b24 vs b5: only as a qualitative dynamic-range reference. b5 did not use PGA normalization, so it is not a clean loss-only comparison.
- b19 vs b8: only as a qualitative amplitude reference. b8 did not use PGA normalization.

## Decision Rules

Do not pick the next mainline by validation MAE alone.

A run is a strong candidate only if it improves or matches b3 on most of:

```text
Val MAE
Val RMSE
Val Corr
Val R2
Val slope
Top20% PGA MAE
Top20% PGA bias
```

Interpretation guide:

- If b19 beats b3, amplitude is useful even after target normalization.
- If b20 beats b3, keep PGA-heavy single-station pretraining as a mainline option.
- If b21 beats b19 and b20, amplitude and PGA-heavy pretraining are complementary.
- If b22 improves strong-PGA bias but hurts MAE, MSE may be useful only with explicit strong-PGA objectives or calibration.
- If b23 improves MAE but worsens slope, Huber0.5 is too robust and increases dynamic-range compression.
- If b24 is close to b22, Huber2 is effectively MSE-like in normalized PGA space.

## Expected Failure Modes

- A run can have lower MAE but worse slope. That likely means stronger dynamic-range compression.
- A run can have low global bias but still underpredict strong PGA. Always check top20% bias.
- Best checkpoint and last checkpoint can disagree. If best is much better than last, report that the configuration may be unstable or overfits late.
- b3/b19-b24 use `pga_target_normalization.mean=auto` in the source config. Manual eval with the original source config can be wrong; use the saved trained `config.json`.

## Current Best Reference Before b19-b24

Current best global reference:

```text
b3_pga_norm_noamp
Val MAE  = 0.3374
Val RMSE = 0.4455
Val Corr = 0.6516
Val R2   = 0.3979
Slope    = 0.5198
Bias     = 0.0419
```

Strong-PGA reference:

```text
b5_huber3_noamp
Top20% MAE  = 0.3845
Top20% Bias = -0.3207
```

b5 is not the global best, but it is the current strong-PGA reference.
