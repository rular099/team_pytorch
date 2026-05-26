# Stage2 512 Layerwise Station-Target Architecture Plan

Date: 2026-05-26

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

```yaml
res_comps: [mag, loc, pga]
res_weight: [0.02, 0.02, 1.0]
```

The event representation is inferred from station evidence. Source coordinates
are not fed as teacher-forced inputs to the PGA readout.

## Generated Experiment Matrix

The configs are YAML files and can be launched with the unchanged single-job
interface:

```bash
bash train_light_slurm.sh pga_configs/<config>.yml
```

| Exp | Purpose | Key switches | Config |
|---|---|---|---|
| b53 | New-branch anchor; fixed `S0`, PGA-only loss | `station_context_mode=off`, no new losses | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b53_branch_anchor_pga_only_chaosuan.yml` |
| b54 | Test event mag/loc auxiliary loss alone | b53 + `res_comps=[mag,loc,pga]` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b54_event_aux_chaosuan.yml` |
| b55 | Test layerwise PGA residual refinement on fixed `S0` | b54 + `pga_layerwise_refinement=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b55_event_aux_layerpga_chaosuan.yml` |
| b56 | Test small-gated station delta path | b54 + `station_context_mode=layerwise_station_target` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b56_event_aux_stationdelta_chaosuan.yml` |
| b57 | Test station deltas plus layerwise PGA refinement | b56 + `pga_layerwise_refinement=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b57_event_aux_stationdelta_layerpga_chaosuan.yml` |
| b58 | Test target-conditioned temporal pooling | b57 + `use_target_temporal_pooling=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b58_event_aux_stationdelta_layerpga_temporal_chaosuan.yml` |
| b59 | Test synchronized station/target RoPE on the full core path | b58 + `use_rope=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b59_event_aux_stationdelta_layerpga_temporal_rope_chaosuan.yml` |
| b60 | Test VS30 after the full core path is enabled | b59 + `use_vs30=true` | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b60_event_aux_stationdelta_layerpga_temporal_rope_vs30_chaosuan.yml` |

## How To Interpret Results

Use adjacent comparisons, not just absolute validation MAE:

- b53 vs pos-r1: checks that the new branch has not changed the anchor path.
- b54 vs b53: event auxiliary supervision effect.
- b55 vs b54: whether layerwise PGA residual refinement helps without station
  memory updates.
- b56 vs b54: station-delta effect without layerwise PGA residuals.
- b57 vs b56 and b57 vs b55: whether station deltas and PGA deltas cooperate.
- b58 vs b57: target-conditioned temporal pooling effect.
- b59 vs b58: synchronized RoPE effect.
- b60 vs b59: VS30 effect; interpret only if VS30 gates open and train fit is
  not damaged.

Always report train MAE, validation MAE, slope, prediction standard deviation,
strong-PGA bias for `label >= -1.0`, weak-bin bias, and gate diagnostics:
station delta gates, PGA delta gates, temporal pooling gates and entropy, VS30
gates, and readout gates.
