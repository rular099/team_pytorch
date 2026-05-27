# Stage2 512 Layerwise Station-Target Architecture Plan

Date: 2026-05-26
Updated: 2026-05-27

This report records the new branch architecture after the `pos-r1` to `pos-r8`
results showed that replacing or heavily transforming the DiTing station memory
hurts train fit. The goal is to test a stronger station-target co-evolution
model while keeping the old b43/pos-r anchor exactly reproducible.

## Code Provenance

Historical configs and results up to the completed `pos-a` to `pos-f` and
`pos-r1` to `pos-r8` experiments should be reproduced from this code commit:

```text
f86f4a58b14780b66a91507bb1c3e94973140c45
Update strict VS30 Japan rebuild diagnostics
```

That commit is on branch `zhangb/diting-backbone-attnpool-team`. The new
architecture work is on branch:

```text
zhangb/layerwise-station-temporal
```

The new branch intentionally does not preserve every old configuration behavior
as a design constraint. For old-result reproduction, use the commit above.

## Architecture

![Layerwise station-target architecture](assets/stage2_512_layerwise_station_temporal_architecture.svg)

The new path is `station_context_mode=layerwise_station_target`.

Main sequence:

```text
T1 = Cross_1(T0, S0)
S1 = S0 + g_s1 * DeltaStation_1(S0)

T2 = Cross_2(T1, S1)
S2 = S1 + g_s2 * DeltaStation_2(S1)

T3 = Cross_3(T2, S2)
S3 = S2 + g_s3 * DeltaStation_3(S2)

T4 = Cross_4(T3, S3)
```

Important details:

- The first target cross-attention layer reads raw DiTing station memory `S0`.
  This keeps the strongest existing path reachable.
- Station updates are pre-norm residual deltas:
  `S = S + gate_attn * SelfAttn(LN(S))`, then
  `S = S + gate_ffn * FFN(LN(S))`.
- There is one station-delta gate family, not separate `gate` and `gamma`
  scalars.
- Targets never attend other targets.
- `use_rope=true` is still a single switch and applies to both station and
  target coordinate systems.
- The optional temporal branch performs target-conditioned pooling over DiTing
  time tokens using target query, station memory, event latent, station/target
  geometry, and a learned time basis.

## b53-b60 Result Summary

The completed `b53` to `b60` results changed the interpretation of the first
layerwise plan:

- `b53` is the new-branch fixed-`S0` anchor and is effectively the b43-style
  fixed-station-memory path under the new code.
- `b54` adds low-weight event mag/loc auxiliary losses while keeping the PGA
  forward path the same as `b53`; this remains the best anchor for the next
  temporal-residual tests.
- `b55` adds layerwise PGA residual refinement but does not add new station
  information. The residual heads barely open, so it does not improve over
  `b54`.
- `b56` transforms `S0` through station-delta blocks. The results show that
  this currently disrupts the useful DiTing station memory rather than exposing
  better features.
- `b57` combines station delta and layerwise PGA residual refinement and is
  worse than the anchor. It should not be used as the base for further
  temporal/RoPE/VS30 tests.
- `b58` to `b60` did not produce valid full-model results. They should be
  treated as memory/implementation failures of the old full temporal path, not
  as evidence against temporal information.

The key design correction is to keep the b54 PGA path intact and add a separate
raw-token residual branch, instead of placing temporal pooling on top of the
already-failing `b57` path.

## Temporal Residual Branch

The first `b58` to `b60` configs attempted target-conditioned temporal pooling
directly on DiTing time tokens with shape:

```text
X: B x S x L x C, C = 1792
```

That path hit HIP OOM inside `TargetConditionedTemporalPool` while stacking and
normalizing the full token tensor. Even the later `b58_c256` style is not the
right scientific comparison, because it is still built on the poor `b57`
station-delta/layerwise-PGA base.

The new `b62` and `b63` configs instead use this structure:

```text
base path:       F -> station adapter -> S0 -> target cross-attention -> pga_base
residual path:   F -> Linear(1792,256) -> target-conditioned temporal pool -> delta_pga
final output:    pga_base + delta_pga
```

![Temporal residual branch](assets/stage2_512_temporal_residual_architecture.svg)

Important implementation details:

- The temporal channel projection is applied per station before stacking across
  stations, so the raw `B x S x L x 1792` activation is not materialized.
- There is no learnable scalar gate on `delta_pga`. The branch is initialized as
  identity by zero-initializing the final `delta_pga` projection, which keeps
  gradients usable.
- `b62` trains only on the final PGA output.
- `b63` uses the same forward graph as `b62`, but adds an auxiliary residual
  loss: `loss(delta_pga, y - stopgrad(pga_base))`.

## Layerwise PGA Refinement

When `pga_layerwise_refinement=true`, each target readout layer contributes a
cumulative PGA estimate:

```text
pga1 = Head_1(T1)
pga2 = pga1 + gate_2 * DeltaHead_2(T2)
pga3 = pga2 + gate_3 * DeltaHead_3(T3)
pga4 = pga3 + gate_4 * DeltaHead_4(T4)
```

The final output is `pga4`. Configs with layerwise refinement also enable an
auxiliary loss on the intermediate cumulative estimates with weights
`[0.1, 0.1, 0.2]`. This tests whether later layers can act as stable residual
corrections rather than forcing all useful information into `S0`.

## Event Auxiliary Head

The model already produces event magnitude and location heads when event tokens
are enabled. The new configs use them as low-weight auxiliary supervision:

```json
"res_comps": ["mag", "loc", "pga"],
"res_weight": [0.02, 0.02, 1.0]
```

The event representation is inferred from station evidence. Source coordinates
are not fed as teacher-forced inputs to the PGA readout.

## Generated Experiment Matrix

The configs are JSON files and can be launched with the unchanged single-job
interface:

```bash
bash train_light_slurm.sh pga_configs/<config>.json
```

| Exp | Purpose | Key switches | Config |
|---|---|---|---|
| b53 | New-branch anchor; fixed `S0`, PGA-only loss | `station_context_mode=off`, no new losses | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b53_branch_anchor_pga_only_chaosuan.json` |
| b54 | Test event mag/loc auxiliary loss alone | b53 + `res_comps=[mag,loc,pga]` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b54_event_aux_chaosuan.json` |
| b55 | Test layerwise PGA residual refinement on fixed `S0` | b54 + `pga_layerwise_refinement=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b55_event_aux_layerpga_chaosuan.json` |
| b56 | Test small-gated station delta path | b54 + `station_context_mode=layerwise_station_target` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b56_event_aux_stationdelta_chaosuan.json` |
| b57 | Test station deltas plus layerwise PGA refinement | b56 + `pga_layerwise_refinement=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b57_event_aux_stationdelta_layerpga_chaosuan.json` |
| b58 | Test target-conditioned temporal pooling | b57 + `use_target_temporal_pooling=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b58_event_aux_stationdelta_layerpga_temporal_chaosuan.json` |
| b59 | Test synchronized station/target RoPE on the full core path | b58 + `use_rope=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b59_event_aux_stationdelta_layerpga_temporal_rope_chaosuan.json` |
| b60 | Test VS30 after the full core path is enabled | b59 + `use_vs30=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b60_event_aux_stationdelta_layerpga_temporal_rope_vs30_chaosuan.json` |
| b58_c256 | Superseded memory-fix attempt on the old b57 base | b58 + `temporal_token_dim=256` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b58_c256_event_aux_stationdelta_layerpga_temporal_chaosuan.json` |
| b59_c256 | Superseded RoPE follow-up on the old b57 base | b59 + `temporal_token_dim=256` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b59_c256_event_aux_stationdelta_layerpga_temporal_rope_chaosuan.json` |
| b60_c256 | Superseded VS30 follow-up on the old b57 base | b60 + `temporal_token_dim=256` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b60_c256_event_aux_stationdelta_layerpga_temporal_rope_vs30_chaosuan.json` |
| b62 | Test raw-`F` temporal residual beyond the b54 fixed-`S0` path | b54 + `use_pga_temporal_residual=true`, no residual aux loss | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b62_event_aux_temporal_residual_chaosuan.json` |
| b63 | Test whether explicit residual supervision opens the `delta_pga` branch | b62 + `pga_temporal_residual_loss.enabled=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b63_event_aux_temporal_residual_aux_chaosuan.json` |

## How To Interpret Results

Use adjacent comparisons, not just absolute validation MAE:

- b53 vs pos-r1: checks that the new branch has not changed the anchor path.
- b54 vs b53: event auxiliary supervision effect.
- b55 vs b54: whether layerwise PGA residual refinement helps without station
  memory updates.
- b56 vs b54: station-delta effect without layerwise PGA residuals.
- b57 vs b56 and b57 vs b55: whether station deltas and PGA deltas cooperate.
- b58 to b60: record them as old temporal-token memory/implementation failures
  on a poor b57 base.
- b58_c256 to b60_c256: superseded; do not prioritize unless a historical
  comparison is explicitly needed.
- b62 vs b54: whether raw DiTing time tokens `F` contain useful PGA residual
  information beyond the adapter-compressed `S0`.
- b63 vs b62: whether explicit residual supervision helps the zero-init
  `delta_pga` branch become active.

Always report train MAE, validation MAE, slope, prediction standard deviation,
strong-PGA bias for `label >= -1.0`, weak-bin bias, and gate diagnostics:
station delta gates, PGA delta gates, temporal residual `delta_abs_mean`,
temporal entropy, VS30 gates, and readout gates.
