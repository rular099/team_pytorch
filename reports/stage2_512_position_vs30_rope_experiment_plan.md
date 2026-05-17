# Stage2 512 Position-Information Experiment Plan: VS30 + RoPE

Date: 2026-05-17

This plan follows the b25-b29 strong-PGA follow-up. The user-selected strong-PGA
candidate is b29-last, so validation MAE is not the main selection criterion for
that line. The next question is whether location/site information improves the
mechanism or only memorizes seen station/location priors.

## Current Anchors

- Clean mechanism anchor: b19 `pga_norm_amp_abs_add`.
- Lowest global-Val-MAE reference: b21 `pga_norm_amp_pga08`.
- Strong-PGA anchor: b29-last `pga_norm_amp_pga08_strongw2`.
- Diagnostic references: b5, b27, b28.

## Key Code Caveat

b19/b29 use:

```json
"pga_readout_mode": "target_cross_attention",
"event_readout_mode": "event_cross_attention"
```

In this path, PGA readout cross-attends directly to `station_feature_emb`. The
TEAM self-attention transformer is not run before PGA prediction. Therefore,
adding RoPE only inside `MultiHeadSelfAttention` would not affect b19/b29 PGA
outputs unless the station tokens are explicitly contextualized before readout.

The RoPE experiment must therefore include a station-context control:

```text
station_context_mode = off/current
station_context_mode = transformer_pre_readout, use_team_rope = false
station_context_mode = transformer_pre_readout, use_team_rope = true
```

Do not compare current b19 directly against `transformer_pre_readout + RoPE` and
interpret the whole delta as RoPE.

## Implementation Design

### VS30

VS30 is not currently in the training graph. Local audit of
`team_pytorch/japan_overfit.hdf5` shows event datasets only contain:

```text
coords, p_picks, pga, waveforms
```

The b19/b29 configs point to the supercomputer file
`/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/japan_2024.hdf5`,
which is not available locally. Before implementation, audit that file for VS30
or station identifiers that can be joined to an external VS30 table.

Required VS30 path:

1. Data layer:
   - support per-event `vs30` aligned with `coords`, or external station table
     keyed by station code;
   - if using an external table, HDF5 must expose stable station identifiers for
     both input stations and target PGA stations;
   - return zero-imputed VS30 plus a validity mask when `use_vs30=false` or when
     old configs do not request VS30.
2. Generator:
   - extend inputs after current `[waveforms, metadata, station_valid,
     pga_targets, pga_target_valid]` only when the model declares site features,
     or add a backward-compatible optional dictionary path;
   - produce `station_site_features` and `pga_target_site_features`;
   - normalize as `log(vs30)` using train-set mean/std and keep a validity bit.
3. Model:
   - config default `use_vs30=false`;
   - add `station_site_proj([log_vs30_norm, valid]) -> emb_dim` with a small
     learnable gate;
   - add `target_site_proj([log_vs30_norm, valid]) -> emb_dim` to PGA query
     embeddings with a separate gate;
   - record diagnostics for VS30 valid ratio and gate values.

Fail closed: if `use_vs30=true` and target-station VS30 cannot be mapped, do not
silently run a station-only VS30 ablation.

### RoPE

RoPE should be added to TEAM station self-attention, not to the target
cross-attention readout in the first pass.

Recommended config:

```json
"station_context_mode": "off",
"mad_params": {
  "use_team_rope": false,
  "rope_coord_mode": "relative_xy_km",
  "rope_coord_scale": 100.0
}
```

Implementation shape:

1. Add `station_context_mode` to `FullModel`.
2. When `station_context_mode="transformer_pre_readout"`, run the station
   transformer on station tokens before event/PGA cross-attention.
3. Pass station coordinates into `Transformer.forward(...)`,
   `TransformerBlock.forward(...)`, and `MultiHeadSelfAttention.forward(...)`.
4. Apply 2D continuous RoPE to Q/K before attention scores, using projected
   relative x/y station coordinates. Use station-center-relative coordinates so
   absolute location is still controlled by the existing coordinate embedding.
5. Keep RoPE disabled by default so old configs are unchanged.

## Event-Split Smoke Matrix

These runs check that the new paths train and that the factor directions are not
obviously broken. They are not sufficient for claiming site generalization.

| Exp | Anchor | Station context | Coord mode | VS30 | RoPE | Purpose |
|---|---|---|---|---|---|---|
| pos-a | b19 | off/current | abs add | off | off | reproduce clean anchor |
| pos-b | b19 | off/current | abs add | on | off | VS30 effect without station context |
| pos-c | b19 | transformer_pre_readout | abs add | off | off | station-context control |
| pos-d | b19 | transformer_pre_readout | abs add | off | on | RoPE isolated against pos-c |
| pos-e | b19 | transformer_pre_readout | abs add | on | off | VS30 under station context |
| pos-f | b19 | transformer_pre_readout | abs add | on | on | VS30 + RoPE interaction |
| pos-g | b29 | off/current | abs add | off | off | reproduce strong-PGA anchor |
| pos-h | b29 | transformer_pre_readout | abs add | off | off | strong-PGA station-context control |
| pos-i | b29 | transformer_pre_readout | abs add | on | on | strong-PGA VS30 + RoPE |
| pos-j | b19 | transformer_pre_readout | relative/geometry-reduced | off | on | geometry reduced RoPE check |
| pos-k | b19 | transformer_pre_readout | relative/geometry-reduced | on | on | VS30 under geometry reduction |

## Station/Spatial Holdout Matrix

After event-split smoke passes, repeat the informative subset under a station or
spatial holdout protocol:

| Exp | Anchor | Station context | VS30 | RoPE |
|---|---|---|---|---|
| hold-a | b19 | off/current | off | off |
| hold-b | b19 | off/current | on | off |
| hold-c | b19 | transformer_pre_readout | off | off |
| hold-d | b19 | transformer_pre_readout | off | on |
| hold-e | b19 | transformer_pre_readout | on | on |
| hold-f | b29 | off/current | off | off |
| hold-g | b29 | transformer_pre_readout | off | off |
| hold-h | b29 | transformer_pre_readout | on | on |

Report seen-station and held-out-station metrics separately.

## Metrics

Always report train and validation:

- MAE, RMSE, Corr, R2, slope, bias.
- Top20 strong-PGA MAE and bias.
- `label >= -1` MAE and bias.
- Seen vs held-out station metrics.
- Residual by target distance to nearest input station.
- Residual by VS30 bin if VS30 exists.
- Residual by station/spatial holdout group.

## Decision Rules

- VS30 is useful only if it helps held-out station/spatial performance, or
  reduces site-correlated residuals without simply increasing train memorization.
- RoPE is useful only if `transformer_pre_readout + RoPE` beats the
  `transformer_pre_readout` control on relevant train-fit and holdout metrics.
- For b29-line runs, prioritize strong-PGA fit and bias. For b19-line runs,
  prioritize mechanism clarity and balanced global metrics.
