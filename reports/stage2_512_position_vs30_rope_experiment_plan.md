# Stage2 512 Position-Information Experiment Plan: VS30 + RoPE

Date: 2026-05-23

This plan follows the b43/b43_clean stage2 anchor. The next question is whether
location/site information improves the mechanism or only memorizes seen
station/location priors.

## Current Anchors

- Legacy clean mechanism anchor: b19 `pga_norm_amp_abs_add`.
- Legacy strong-PGA reference: b29-last `pga_norm_amp_pga08_strongw2`.
- Current structure anchor: b43 `pga_norm_amp_pga08_strongw2_xattn4_gate0_firstres`.
- Clean-data balanced anchor: b43_clean best in
  `weights_japan_overfit_pga15_stage2_512_b43_pga_norm_amp_pga08_strongw2_xattn4_gate0_firstres_new`.

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

The RoPE experiment must therefore include a station-context control. The raw
`transformer_pre_readout` path collapsed in b41/b47, so the default control for
new experiments should be the gated identity-initialized path:

```text
station_context_mode = off/current
station_context_mode = gated_transformer_pre_readout, use_team_rope = false
station_context_mode = gated_transformer_pre_readout, use_team_rope = true
```

Do not compare current b43 directly against `gated_transformer_pre_readout +
RoPE` and interpret the whole delta as RoPE.

## Implementation Status (2026-05-23)

VS30 source and data:

- Source: J-SHIS Mesh Search API, field `AVS` (30 m average S-wave velocity).
- Downloader: `tools/download_japan_vs30.py`.
- Downloaded station table:
  `resources/vs30/japan_vs30_jshis_station_cache.csv`.
- Coverage: 1489 unique station-coordinate rows, 1483 valid VS30 values.
- Missing rows: 6 KNG20x island/near-offshore stations returned J-SHIS 404 at
  1 km radius and are represented by `vs30_valid=0`.

HDF5 construction:

- `build_japan_training_data.py` accepts `--vs30_csv`,
  `--vs30_max_distance_km`, and `--require_vs30`.
- `japan_dataset_builder.py` writes station-aligned `vs30`, `vs30_valid`,
  `vs30_query_distance_km`, `vs30_source`, `vs30_mesh_code`, and
  `vs30_match_method` datasets when `--vs30_csv` is provided.
- Default behavior is non-destructive: missing VS30 stays masked. Use
  `--require_vs30` only for a strict ablation that drops stations without VS30.

Training/model:

- Generator switch: `model_params.use_vs30=true` makes
  `PreloadedEventGenerator` append station and target VS30 tensors after
  `pga_target_valid`; old configs keep the previous input layout.
- Model switch: `use_vs30=false` by default. When enabled, station and target
  VS30 use `[log(vs30 / 760), valid]` projections with separate learnable gates.
- RoPE switch: `use_team_rope=false` by default; top-level model params are
  forwarded into TEAM self-attention. The first pass uses
  `rope_coord_mode="relative_xy_km"` and `rope_coord_scale=100.0`.
- Station context now supports `gated_transformer_pre_readout` with
  `station_context_gate_init=0.0`; this is the intended RoPE control path.

## Implementation Design

### VS30

VS30 path:

1. Data layer:
   - support per-event `vs30` aligned with `coords`;
   - join the external station table by `station_code`, with coordinate fallback
     only when the code is missing;
   - return zero-imputed VS30 plus a validity mask when old HDF5 files lack
     VS30.
2. Generator:
   - extend inputs after `[waveforms, metadata, station_valid, pga_targets,
     pga_target_valid]` only when `use_vs30=true`;
   - produce station and target VS30 tensors plus validity masks.
3. Model:
   - config default `use_vs30=false`;
   - add `station_site_proj([log(vs30 / 760), valid]) -> emb_dim` with a
     learnable gate initialized by `vs30_init_gate`;
   - add `target_site_proj([log(vs30 / 760), valid]) -> emb_dim` to PGA query
     embeddings with a separate gate;
   - record diagnostics for VS30 valid ratio and gate values.

Fail closed in analysis: if target VS30 valid ratio is low, do not interpret the
run as a clean target-site VS30 ablation.

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

1. Use `station_context_mode="gated_transformer_pre_readout"` for the main
   control. Keep raw `transformer_pre_readout` only for backward-compatible
   debugging.
2. In gated mode, run the station transformer on station tokens and combine it
   as `station_feature + gate * (transformer_context - station_feature)`.
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
| pos-a | b43_clean | off/current | abs add | off | off | reproduce clean-data anchor |
| pos-b | b43_clean | off/current | abs add | on | off | VS30 effect without station context |
| pos-c | b43_clean | gated_transformer_pre_readout | abs add | off | off | station-context control |
| pos-d | b43_clean | gated_transformer_pre_readout | abs add | off | on | RoPE isolated against pos-c |
| pos-e | b43_clean | gated_transformer_pre_readout | abs add | on | off | VS30 under station context |
| pos-f | b43_clean | gated_transformer_pre_readout | abs add | on | on | VS30 + RoPE interaction |
| pos-g | b43 | off/current | abs add | off | off | old-data anchor bridge |
| pos-h | b43 | gated_transformer_pre_readout | abs add | off | off | old-data station-context control |
| pos-i | b43 | gated_transformer_pre_readout | abs add | on | on | old-data VS30 + RoPE bridge |
| pos-j | b43_clean | gated_transformer_pre_readout | relative/geometry-reduced | off | on | geometry-reduced RoPE check |
| pos-k | b43_clean | gated_transformer_pre_readout | relative/geometry-reduced | on | on | VS30 under geometry reduction |

## Station/Spatial Holdout Matrix

After event-split smoke passes, repeat the informative subset under a station or
spatial holdout protocol:

| Exp | Anchor | Station context | VS30 | RoPE |
|---|---|---|---|---|
| hold-a | b19 | off/current | off | off |
| hold-b | b19 | off/current | on | off |
| hold-c | b43_clean | gated_transformer_pre_readout | off | off |
| hold-d | b43_clean | gated_transformer_pre_readout | off | on |
| hold-e | b43_clean | gated_transformer_pre_readout | on | on |
| hold-f | b43 | off/current | off | off |
| hold-g | b43 | gated_transformer_pre_readout | off | off |
| hold-h | b43 | gated_transformer_pre_readout | on | on |

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
- RoPE is useful only if `gated_transformer_pre_readout + RoPE` beats the
  `gated_transformer_pre_readout` control on relevant train-fit and holdout
  metrics.
- For b29-line runs, prioritize strong-PGA fit and bias. For b19-line runs,
  prioritize mechanism clarity and balanced global metrics.
