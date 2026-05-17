# Stage2 512 Cross-Attention Depth Experiment Plan

Date: 2026-05-17

This plan tests whether the one-layer PGA target cross-attention readout is a
capacity bottleneck. It is separate from the later station self-attention/RoPE
work: these experiments repeatedly update the PGA query by attending to fixed
station features, but they do not add station-station self-attention.

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

The station features remain fixed memory in this experiment. A separate
`station_context_mode=transformer_pre_readout` experiment is still required to
test station-station self-attention and RoPE.

## Experiment Matrix

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

## Metrics

Report both train and validation:

- global MAE, RMSE, Corr, R2, slope, bias;
- top20 strong-PGA MAE and bias;
- `label >= -1` MAE and bias;
- train MAE and strong-PGA train metrics, because this is still an overfit/capacity experiment;
- attention diagnostics if enabled in a later rerun.

## Decision Rules

- If b30-b32 improve train fit without helping strong-PGA validation, query
  depth is not the limiting factor for strong-PGA behavior.
- If b33-b35 improve b29 strong-PGA bins while reducing global overprediction,
  keep the best layer count as the strong-PGA anchor for the next position/site
  experiment.
- If deeper layers only worsen weak-PGA overprediction, return to b29-last and
  move on to VS30/station-context/RoPE.
