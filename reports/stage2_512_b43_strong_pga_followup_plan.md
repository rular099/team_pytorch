# Stage2 512 b43 Strong-PGA Follow-up Plan

Date: 2026-05-19

This batch uses b43 as the structural anchor:

```text
b43 = pga_norm_amp_pga08_strongw2_xattn4_gate0_firstres
```

b43 fixes the multi-layer readout train-fit collapse and has strong global
validation behavior, but it is still less aggressive than b29-last on high-PGA
targets. The purpose of b48-b52 is to test whether b43 can recover part of the
b29-last strong-PGA correction without losing the b43 train fit and global
calibration.

## Baselines

| Exp | Ckpt | Train MAE | Val MAE | Val slope | Top20 MAE | Top20 bias |
|---|---|---:|---:|---:|---:|---:|
| b29 strong reference | last | 0.2260 | 0.4252 | 0.5383 | 0.2756 | -0.1265 |
| b43 balanced anchor | last | 0.0600 | 0.2966 | 0.5548 | 0.3928 | -0.3458 |

## Configs

All configs keep the b43 model structure:

```json
"pga_readout_layers": 4,
"event_readout_layers": 1,
"readout_residual_gates": true,
"readout_residual_gate_init": 0.0,
"readout_ffn_gate_init": 0.0,
"readout_first_residual": true
```

| Exp | Config | Strong-PGA change |
|---|---|---|
| b48 | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b48_pga_norm_amp_pga08_strongw4_xattn4_gate0_firstres_chaosuan.json` | threshold `-1.2`, strong weight `4.0` |
| b49 | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b49_pga_norm_amp_pga08_strongw6_xattn4_gate0_firstres_chaosuan.json` | threshold `-1.2`, strong weight `6.0` |
| b50 | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b50_pga_norm_amp_pga08_strongw4_thm10_xattn4_gate0_firstres_chaosuan.json` | threshold `-1.0`, strong weight `4.0` |
| b51 | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b51_pga_norm_amp_pga08_labelstrat50_strongw2_xattn4_gate0_firstres_chaosuan.json` | train-only label-stratified targets, strong fraction `0.5`, threshold `-1.2`, strong weight `2.0` |
| b52 | `pga_configs/transformer_japan_overfit_pga15_stage2_512_b52_pga_norm_amp_pga08_labelstrat50_strongw4_xattn4_gate0_firstres_chaosuan.json` | train-only label-stratified targets, strong fraction `0.5`, threshold `-1.2`, strong weight `4.0` |

## Decision Criteria

Primary checks:

- Train MAE should stay near b43 and preferably below `0.10`.
- Val slope should stay near `0.55`.
- Top20 bias should move from b43's `-0.3458` toward b29-last's `-0.1265`.
- Weak-PGA overprediction should not become as severe as b29-last.

Interpretation:

- If b48/b49 improve strong bins but global bias becomes too positive, use the
  lower weight.
- If b50 helps more than b48, the true strong-PGA threshold is likely closer to
  `label >= -1.0` than `label >= -1.2`.
- If b51/b52 beat b48/b49 on strong bins without weak-PGA overprediction,
  target sampling is more effective than loss weighting alone.

## Related Diagnostics

The P3 residual analysis is in:

```text
reports/stage2_512_p3_residual_diagnostics.md
```

It shows b43 greatly improves global/event/site residuals over b29-last, while
remaining too conservative on the highest-PGA bins.
