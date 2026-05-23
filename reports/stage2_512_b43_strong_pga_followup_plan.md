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

## Results and Interpretation

Updated: 2026-05-23

The b48-b52 results must be interpreted by their original purpose, not by
validation MAE alone. They were designed to test whether the b43 structure can
recover part of the b29-last strong-PGA correction while preserving the b43
train fit and global calibration.

There are two result groups:

- `*_old` directories use the same dataset as previous experiments and are the
  fair comparison set against b29 and b43.
- directories without `_old` use the cleaned training data. The cleaned data
  changes the effective problem: valid validation targets drop from 757 to 656,
  the valid training targets drop from 4648 to 4138, and the PGA label
  distribution shifts stronger. A b43 clean-data anchor is now available in
  `weights_japan_overfit_pga15_stage2_512_b43_pga_norm_amp_pga08_strongw2_xattn4_gate0_firstres_new`.

Metric note: the historical reports use a specific top20 strong-PGA definition.
For this review, label-bin metrics such as `label >= -1.0` and weak-bin bias
are the safer comparable diagnostics because they match the earlier residual
diagnostic bins for b29 and b43.

### Old-data Results

Old-data anchors:

| Exp | Role | Ckpt | Train MAE | Val MAE | Val slope | `label >= -1.0` MAE / bias | Weak-bin bias |
|---|---|---|---:|---:|---:|---:|---:|
| b43 | balanced anchor | last | 0.0600 | 0.2966 | 0.5548 | 0.4607 / -0.4388 | +0.1755 |
| b29 | strong-PGA reference | last | 0.2260 | 0.4252 | 0.5383 | 0.2832 / -0.2019 | +0.4366 |

b43 keeps excellent train fit and global validation behavior but remains too
conservative on strong PGA. b29 is much more aggressive in strong-PGA bins, but
its positive global and weak-bin bias show that it partly fixes strong PGA by
raising weak PGA too much.

| Exp | Ckpt | Train MAE | Val MAE | Val slope | `label >= -1.0` MAE / bias | Weak-bin bias | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| b48_old | best | 0.1901 | 0.3826 | 0.5587 | 0.3896 / -0.3561 | +0.2240 | Strong-PGA improves over b43, but train fit and global MAE degrade too much. |
| b49_old | best | 0.2892 | 0.4267 | 0.4739 | 0.3231 / -0.2754 | +0.3911 | Strong-PGA moves toward b29, but weak-PGA overprediction and global degradation become b29-like. |
| b50_old | best | 0.1322 | 0.3496 | 0.5377 | 0.4031 / -0.3611 | +0.2390 | Threshold `-1.0` is more balanced than pure higher weighting, but still misses the train-fit target. |
| b50_old | last | 0.2389 | 0.4199 | 0.5197 | 0.3006 / -0.2225 | +0.4335 | Strong-PGA recovery is large, but late training creates b29-like weak-PGA overprediction. |
| b51_old | best | 0.1989 | 0.3395 | 0.5847 | 0.3392 / -0.2864 | +0.2273 | Label-stratified sampling is more promising than pure weight increase, but train fit is still far from b43. |
| b51_old | last | 0.2048 | 0.3643 | 0.6627 | 0.2970 / -0.1815 | +0.3007 | Strong-PGA gets closer to b29, but global and weak-bin bias worsen. |
| b52_old | best | 0.2630 | 0.3825 | 0.5860 | 0.3532 / -0.3029 | +0.1798 | Heavier label-stratified setting is not better than b51. |
| b52_old | last | 0.4285 | 0.5213 | 0.5075 | 0.2449 / -0.0555 | +0.5831 | Reject: strong-PGA looks good only because the whole calibration is badly shifted upward. |

Old-data conclusion: none of b48-b52_old should replace b43. The experiments
do show that strong-PGA underprediction can be moved, but all successful
strong-PGA movements pay too much in train fit, global validation behavior, or
weak-PGA overprediction. b50 and b51 are the useful mechanism signals:
`label >= -1.0` appears to be a better strong-PGA threshold than `-1.2`, and
moderate target sampling is less damaging than simply increasing strong loss
weight.

### Clean-data Results

The cleaned data changes the dataset composition, so the clean runs answer a
different question from the `_old` runs. The new b43 clean-data result provides
the fair clean anchor that was missing in the first review.

Caveat: the b43 clean result directory is named with `_new`, but its
`config.json` still has `training_params.weight_path` set to the old b43
directory name. The eval logs therefore show checkpoint paths without `_new`.
The training curves and effective target counts differ from old b43, so the
result is useful as a clean-data b43 evaluation, but future configs should use a
distinct `weight_path` to avoid checkpoint/eval ambiguity.

| Exp | Ckpt | Train MAE | Val MAE | Val slope | `label >= -1.0` MAE / bias | Weak-bin bias | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| b43_clean | best | 0.0790 | 0.3260 | 0.5968 | 0.4409 / -0.3215 | +0.1993 | Clean-data balanced anchor. Best global MAE, good train fit, and stable slope. |
| b43_clean | last | 0.0525 | 0.3281 | 0.5669 | 0.4227 / -0.3042 | +0.2631 | Stronger train fit and slightly better strong-PGA bias, but more weak-bin overprediction. |
| b48_clean | best | 0.2106 | 0.3610 | 0.4975 | 0.3479 / -0.2573 | +0.3766 | Strong-PGA improves, but weak-PGA bias is high. |
| b49_clean | best | 0.1827 | 0.3673 | 0.4711 | 0.3826 / -0.3152 | +0.3640 | Higher weight does not give a cleaner tradeoff. |
| b50_clean | best | 0.1200 | 0.3399 | 0.5821 | 0.3724 / -0.2837 | +0.2694 | Best strong-PGA calibration candidate among clean variants, but no longer the global anchor. |
| b50_clean | last | 0.1064 | 0.3391 | 0.5484 | 0.3859 / -0.3116 | +0.2963 | Keeps useful train fit and strong-PGA movement, with more weak-bin bias than b43_clean. |
| b51_clean | best | 0.2095 | 0.3357 | 0.5233 | 0.4677 / -0.4180 | +0.2014 | Not a strong-PGA fix and not better than b43_clean as a balanced model. |
| b52_clean | best | 0.2233 | 0.3763 | 0.5223 | 0.3502 / -0.2552 | +0.3831 | Strong-PGA moves, but train/global and weak-bin tradeoffs are worse than b50. |

Clean-data conclusion: b43_clean best is the clean-data balanced anchor. It has
the best global MAE, much better train fit than b48-b52_clean, and the strongest
validation slope in the clean group. b50_clean remains the primary
strong-PGA-calibration candidate because it reduces `label >= -1.0` MAE/bias
relative to b43_clean, but it pays with worse global MAE and more weak-bin
overprediction. b51_clean should no longer be treated as the clean global
candidate because b43_clean is better balanced and fits train much better.

### Updated Recommendation

1. Keep b43 as the current old-data balanced structural anchor.
2. Keep b29-last only as a strong-PGA calibration reference, not as a global
   model candidate.
3. Do not adopt b48-b52_old as a replacement for b43.
4. Use b43_clean best as the cleaned-data balanced anchor.
5. Fix future clean-run configs so `weight_path` is distinct from old b43,
   avoiding checkpoint and eval-path ambiguity.
6. For the next strong-PGA calibration sweep, start from the b50 idea:
   threshold `label >= -1.0`, but reduce the effective strong pressure
   compared with b50/b52, for example strong weight `2.0` or `3.0`, or a mild
   label-stratified variant. Every future report should include train MAE,
   validation slope, `label >= -1.0` bias, and weak-bin bias.
