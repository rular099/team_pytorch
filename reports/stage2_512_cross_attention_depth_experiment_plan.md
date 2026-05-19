# Stage2 512 Cross-Attention Depth and Collapse Diagnostics

Date: 2026-05-18

This note records the b30-b35 cross-attention depth result and defines b36-b47
to diagnose why naive 3-4 layer PGA target cross-attention collapses on the
train set.

## Implementation

`CrossAttentionReadout` now supports multiple layers while keeping the old
one-layer behavior as the default:

```json
"pga_readout_layers": 3,
"event_readout_layers": 1,
"readout_ffn_hidden_dim": 1000
```

For `readout_layers=1`, the first layer keeps the original parameter names and
forward behavior so old configs and checkpoints remain compatible. Extra layers
are appended only when explicitly requested.

Each additional layer performs:

```text
PGA query -> cross-attention over station_feature_emb -> residual/norm -> FFN -> residual/norm
```

For b30-b35, the station features remain fixed memory. The b41/b47 diagnostics
below explicitly turn on `station_context_mode=transformer_pre_readout` to test
station-station interaction without RoPE.

The follow-up diagnostics add backward-compatible optional switches:

```json
"readout_first_residual": false,
"readout_residual_gates": false,
"readout_residual_gate_init": 0.0,
"readout_ffn_gate_init": 0.0,
"readout_inject_base_query": false,
"readout_query_injection_gate_init": 1.0,
"readout_use_ffn": true,
"station_context_mode": "off"
```

Old configs do not set these fields and keep the previous code path.

## b30-b35 Matrix

Use the same 512-event split, deterministic sampling, PGA target normalization,
and target-cross-attention readout as b19/b29.

| Exp | Anchor | PGA cross-attn layers | Event cross-attn layers | Purpose |
|---|---|---:|---:|---|
| b19 | clean baseline | 1 | 1 | current clean global anchor |
| b30 | b19 | 2 | 1 | shallow query refinement |
| b31 | b19 | 3 | 1 | medium query refinement |
| b32 | b19 | 4 | 1 | deeper query refinement |
| b29 | strong-PGA baseline | 1 | 1 | current user-approved strong-PGA anchor |
| b33 | b29 | 2 | 1 | strong-PGA shallow refinement |
| b34 | b29 | 3 | 1 | strong-PGA medium refinement |
| b35 | b29 | 4 | 1 | strong-PGA deeper refinement |

## Configs

```text
pga_configs/transformer_japan_overfit_pga15_stage2_512_b30_pga_norm_amp_xattn2_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b31_pga_norm_amp_xattn3_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b32_pga_norm_amp_xattn4_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b33_pga_norm_amp_pga08_strongw2_xattn2_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b34_pga_norm_amp_pga08_strongw2_xattn3_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b35_pga_norm_amp_pga08_strongw2_xattn4_chaosuan.json
```

## b30-b35 Result

| Exp | Anchor | Layers | Ckpt | Train MAE | Val MAE | Slope | Bias | Top20 MAE | Top20 Bias |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| b19 ref | clean | 1 | last | 0.0760 | 0.3265 | 0.5587 | 0.0115 | 0.4501 | -0.3861 |
| b30 | b19 | 2 | best | 0.1086 | 0.3229 | 0.5685 | 0.0219 | 0.4240 | -0.3440 |
| b31 | b19 | 3 | best | 0.2684 | 0.3524 | 0.3868 | -- | 0.5559 | -0.5427 |
| b32 | b19 | 4 | best | 0.4388 | 0.4462 | 0.0745 | -- | 0.6883 | -0.6883 |
| b29 ref | strong | 1 | last | 0.2260 | 0.4252 | 0.5383 | 0.3007 | 0.2756 | -0.1265 |
| b33 | b29 | 2 | last | 0.0916 | 0.3359 | 0.5563 | 0.0334 | 0.4216 | -0.3610 |
| b34 | b29 | 3 | best | 0.4464 | 0.4246 | 0.1954 | -- | 0.5475 | -0.5475 |
| b35 | b29 | 4 | last | 0.4566 | 0.4577 | 0.1094 | -- | 0.5125 | -0.5121 |

Interpretation:

- b30/b33 show that two readout layers are trainable; b30 slightly improves the
  clean anchor's Val MAE and strong-PGA bias.
- b31/b32/b34/b35 are not merely overfit. They lose train-set fit, compress
  prediction dynamic range, and show very low slope.
- The likely failure modes are architectural or optimization-related: the first
  readout layer discards the original query residual, extra layers are not
  identity-initialized, target query/position information may decay across
  layers, FFN/post-norm may compress the query, and repeated attention over
  fixed station memory may oversmooth.

## b36-b47 Diagnostic Matrix

All b36-b47 keep `pga_readout_layers=4` and `event_readout_layers=1`.

| Exp | Anchor | Main factor | Key config |
|---|---|---|---|
| b36 | b32 clean xattn4 | gate / identity init | `readout_residual_gates=true`, gate init 0 |
| b37 | b32 | first-layer query residual | b36 + `readout_first_residual=true` |
| b38 | b32 | per-layer query/position injection | b36 + `readout_inject_base_query=true` |
| b39 | b32 | extra FFN influence | b36 + `readout_use_ffn=false` |
| b40 | b32 | readout geometry bias | b36 + `pga_distance_bias=true` |
| b41 | b32 | station-station interaction | b36 + `station_context_mode=transformer_pre_readout` |
| b42 | b35 strong xattn4 | gate / identity init | same as b36 on strong branch |
| b43 | b35 | first-layer query residual | same as b37 on strong branch |
| b44 | b35 | per-layer query/position injection | same as b38 on strong branch |
| b45 | b35 | extra FFN influence | same as b39 on strong branch |
| b46 | b35 | readout geometry bias | same as b40 on strong branch |
| b47 | b35 | station-station interaction | same as b41 on strong branch |

## b36-b47 Configs

```text
pga_configs/transformer_japan_overfit_pga15_stage2_512_b36_pga_norm_amp_xattn4_gate0_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b37_pga_norm_amp_xattn4_gate0_firstres_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b38_pga_norm_amp_xattn4_gate0_injectq_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b39_pga_norm_amp_xattn4_gate0_noffn_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b40_pga_norm_amp_xattn4_gate0_distbias_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b41_pga_norm_amp_xattn4_gate0_stationctx_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b42_pga_norm_amp_pga08_strongw2_xattn4_gate0_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b43_pga_norm_amp_pga08_strongw2_xattn4_gate0_firstres_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b44_pga_norm_amp_pga08_strongw2_xattn4_gate0_injectq_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b45_pga_norm_amp_pga08_strongw2_xattn4_gate0_noffn_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b46_pga_norm_amp_pga08_strongw2_xattn4_gate0_distbias_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_b47_pga_norm_amp_pga08_strongw2_xattn4_gate0_stationctx_chaosuan.json
```

## b36-b47 Result

Results below use the lower-Val-MAE checkpoint for each experiment. This is an
overfit/capacity experiment, so the Train MAE and slope are as important as Val
MAE.

| Exp | Main factor | Ckpt | Train MAE | Val MAE | Val slope | Val bias | Top20 MAE | Top20 bias |
|---|---|---|---:|---:|---:|---:|---:|---:|
| b32 ref | clean xattn4 collapse | best | 0.4388 | 0.4462 | 0.0745 | 0.0157 | 0.6883 | -0.6883 |
| b36 | gate0 | best | 0.1018 | 0.3163 | 0.5204 | 0.0123 | 0.4481 | -0.4053 |
| b37 | gate0 + first residual | last | **0.0589** | **0.2964** | 0.5251 | -0.0036 | 0.4642 | -0.4261 |
| b38 | gate0 + inject query | last | 0.0784 | 0.3194 | 0.5149 | 0.0239 | 0.4595 | -0.4111 |
| b39 | gate0 + no FFN | last | 0.0941 | 0.3204 | 0.5814 | -0.0462 | 0.4814 | -0.4222 |
| b40 | gate0 + distance bias | best | 0.1585 | 0.3247 | 0.4997 | -0.1032 | 0.5534 | -0.5378 |
| b41 | gate0 + station context | last | 0.4620 | 0.4540 | 0.0000 | 0.0659 | 0.7092 | -0.7092 |
| b35 ref | strong xattn4 collapse | last | 0.4566 | 0.4577 | 0.1094 | 0.1554 | 0.5125 | -0.5121 |
| b42 | strong + gate0 | last | 0.1205 | 0.3429 | 0.5513 | -0.0517 | 0.5032 | -0.4641 |
| b43 | strong + gate0 + first residual | last | **0.0600** | **0.2966** | 0.5548 | 0.0421 | 0.3928 | -0.3458 |
| b44 | strong + gate0 + inject query | best | 0.0971 | 0.3329 | 0.5350 | 0.0649 | 0.4104 | -0.3470 |
| b45 | strong + gate0 + no FFN | best | 0.0953 | 0.3332 | 0.5662 | 0.0592 | 0.4025 | -0.3288 |
| b46 | strong + gate0 + distance bias | last | 0.0929 | 0.3290 | 0.5389 | 0.0451 | 0.4248 | -0.3683 |
| b47 | strong + gate0 + station context | best | 0.4621 | 0.4470 | 0.0244 | 0.0686 | 0.6837 | -0.6837 |

Interpretation:

- `readout_residual_gates=true` with zero-initialized gates fixes the train-set
  collapse. b32 to b36 improves Train MAE from 0.4388 to 0.1018; b35 to b42
  improves Train MAE from 0.4566 to 0.1205.
- The first-layer query residual is the strongest single improvement. b37 and
  b43 both reach about 0.296 Val MAE and about 0.06 Train MAE.
- Per-layer base-query injection is not the main missing factor. It helps keep
  the model trainable but does not beat first residual.
- The extra FFN is not the collapse source. Removing it does not beat b37/b43.
- Readout distance bias does not help this batch and hurts clean strong-PGA
  bias in b40.
- Current `station_context_mode=transformer_pre_readout` is not usable for
  downstream RoPE yet. b41/b47 collapse to nearly constant predictions. The
  station-context path needs its own gated/identity bypass before RoPE tests.

Current recommendation:

- Use b43 as the new balanced structural anchor: it keeps strong-PGA weighting,
  fixes multi-layer train fit, and has the best global Val MAE in this batch.
- Keep b29-last as the strong-PGA calibration reference because its top20 MAE
  and bias remain more aggressive (`0.2756 / -0.1265`) despite poor global Val
  MAE.
- Do not run RoPE on the current station-context path before adding a gated or
  zero-init station-context bypass.

## Metrics

Report both train and validation:

- global MAE, RMSE, Corr, R2, slope, bias;
- top20 strong-PGA MAE and bias;
- `label >= -1` MAE and bias;
- train MAE and strong-PGA train metrics, because this is still an overfit/capacity experiment;
- attention diagnostics if enabled in a later rerun.

## Decision Rules

- b36/b42 confirmed that non-identity extra layers were the main collapse
  trigger.
- b37/b43 confirmed that first-layer query residual is the most useful fix.
- b41/b47 showed that station context cannot be used as the RoPE control until
  that path has its own stable residual/gate design.
