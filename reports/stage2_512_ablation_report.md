# Stage2 512-Event PGA Ablation Report

Date: 2026-05-12

This report summarizes the 512-event stage2 PGA ablation results for experiments b0-b17.
The goal is to identify which tested factors are useful, what the best observed run is,
and what combined configuration should be tested next.

## Scope And Caveats

- Metrics are copied from the text eval logs under `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/`.
- b0-b7 currently have one local eval file each: `eval_results.txt`, loaded from `full_model_last.pth`.
- b8-b17 have both `eval_results_best.txt` and `eval_results_last.txt`; the main table reports the checkpoint with lower validation MAE for each experiment.
- This means b0-b7 and b8-b17 are not perfectly matched by checkpoint selection policy. The source file column records the exact file used for every row.
- All configs point to the shared event list path `pga_configs/stage2_512_event_ids.txt`. The local result bundle inspected here did not include `overfit_event_ids.txt` or `stage2_512_event_ids.txt`; reviewers should verify the original supercomputer workspace has the 512-line shared event list.
- The recommended "best combination" below is an evidence-based next config, not a directly trained b0-b17 run. The direct best observed experiment is reported separately.

## Overall Result

The best observed single run is **b3_pga_norm_noamp**:

- Validation MAE: **0.3374**
- Validation RMSE: **0.4455**
- Validation Corr: **0.6516**
- Validation R2: **0.3979**
- Validation slope: **0.5198**
- Source: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b3_pga_norm_noamp/eval_results.txt`

Among the new b8-b17 feature/fusion experiments, the best observed run is **b8_abs_add_amp**:

- Validation MAE: **0.3522**
- Validation RMSE: **0.4586**
- Validation Corr: **0.6148**
- Validation R2: **0.3619**
- Validation slope: **0.4548**
- Source: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_best.txt`

## Recommended Combination To Test Next

If constructing a new combined experiment, the most defensible candidate is:

```text
PGA target normalization enabled
+ absolute coordinate input
+ coord_fusion_mode = concat
+ use_amplitude_info = true
+ target_cross_attention PGA readout
+ event_cross_attention event readout
+ shared deterministic 512-event split
```

Rationale:

- PGA target normalization is the strongest tested single factor: b3 is best overall.
- Explicit amplitude information improves the baseline substantially: b8 beats b0 by 0.0416 MAE.
- Absolute-coordinate concat improves over absolute-coordinate add: b15 beats b0 by 0.0317 MAE.
- Pure relative coordinates and relative-coordinate concat are weak; they should not be part of the first combined config.
- Weak rel+abs fusion weights are not useful. If rel+abs is kept, only `coords_abs_weight=1.0` looks competitive, but absolute-only concat is simpler and cleaner.

This combined config has not yet been directly tested. It should be treated as a next experiment, not as a proven additive gain.

## Factor Conclusions

### Target Normalization

Best setting: **enable PGA target normalization**.

b3 is the best overall result and also has one of the better slopes. It improves b0 from 0.3938 MAE to 0.3374 MAE and improves R2 from 0.2184 to 0.3979.

### Single-Station Pretrain Weighting

Best tested setting: **PGA-heavy multitask pretrain**, b1.

b1 improves b0 clearly: 0.3531 MAE vs 0.3938. PGA-only pretrain b2 also helps but is weaker than b1. Keeping mag/epidist auxiliary tasks with higher PGA weight appears better than dropping them completely.

### Full-Model Loss

Best MAE/RMSE setting among tested losses: **MSE**, b4.

b4 reaches 0.3477 MAE and 0.3944 R2, second only to b3 by MAE. However its slope is 0.4025, worse than b3, b5, and b0, so it likely worsens dynamic-range compression. Huber delta 3.0, b5, gives a better slope of 0.5403 but worse MAE.

### Amplitude Information

Best setting: **use amplitude information**.

b8 improves the baseline substantially:

- b0: MAE 0.3938, RMSE 0.5076, R2 0.2184
- b8 best: MAE 0.3522, RMSE 0.4586, R2 0.3619

The slope drops from 0.5269 to 0.4548, so amplitude helps average error but does not solve strong-PGA compression by itself.

### Coordinate Type

Best tested simple coordinate type: **absolute coordinates**.

Pure relative coordinates, b9, are not better than b0 by a meaningful margin and have much weaker corr/R2/slope:

- b0 abs add: MAE 0.3938, Corr 0.6007, R2 0.2184, slope 0.5269
- b9 rel add: MAE 0.3917, Corr 0.5107, R2 0.2482, slope 0.3179

The MAE tie hides that relative coordinates compress the prediction range much more.

### Relative + Absolute Fusion Weight

Best tested rel+abs setting: **add fusion with `coords_abs_weight=1.0`**, b14.

The sweep is not monotonic. Weak absolute weights do not help:

- w=0.01, b10: MAE 0.3918
- w=0.03, b11: MAE 0.4045
- w=0.10, b12: MAE 0.4097
- w=0.30, b13: MAE 0.3925
- w=1.00, b14: MAE 0.3653

Interpretation: in the current formulation `rel_emb + w * abs_emb`, the model only benefits when the absolute component has comparable scale to the relative component.

### Add vs Concat

Best tested setting: **absolute-coordinate concat**, b15.

For absolute coordinates, concat improves the baseline:

- b0 abs add: MAE 0.3938
- b15 abs concat: MAE 0.3621

For relative coordinates, concat is harmful:

- b9 rel add: MAE 0.3917
- b16 rel concat: MAE 0.4211

For rel+abs at w=0.1, concat is much worse:

- b12 rel+abs add w=0.1: MAE 0.4097
- b17 rel+abs concat w=0.1: MAE 0.4536

Concat should only be carried forward for absolute coordinates unless a new design changes how relative coordinates are represented.

### Distance Bias / Relative Geometry

b7 improves b0 by MAE and R2, but is not among the strongest stage2 results:

- b7: MAE 0.3693, R2 0.3084
- b0: MAE 0.3938, R2 0.2184

It remains a useful secondary factor, but should not take priority over target normalization, amplitude, or absolute-coordinate concat.

### Station Embedding Decorrelation

b6 is slightly better than b0 by MAE but not competitive with b1/b3/b4/b8/b15:

- b6: MAE 0.3855
- b0: MAE 0.3938

It is not a priority for the next combined experiment.

## Main Metrics Table

| Exp | Tested factor | Reported ckpt | Val MAE | RMSE | Corr | R2 | Slope | Bias | Train MAE | Source eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b0 | baseline: abs coord add, no amplitude, Huber1, multitask single-station pretrain | full_model_last.pth@150 | 0.3938 | 0.5076 | 0.6007 | 0.2184 | 0.5269 | -0.9356 | 0.2316 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b0_baseline_noamp/eval_results.txt` |
| b1 | single-station pretrain PGA-heavy weights 0.8/0.1/0.1 | full_model_last.pth@150 | 0.3531 | 0.4538 | 0.6627 | 0.3751 | 0.4882 | -0.9913 | 0.1933 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b1_single_pga08_noamp/eval_results.txt` |
| b2 | single-station pretrain PGA-only | full_model_last.pth@150 | 0.3784 | 0.4856 | 0.6147 | 0.2846 | 0.5039 | -0.9561 | 0.1909 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b2_single_pga_only_noamp/eval_results.txt` |
| b3 | PGA target normalization enabled | full_model_last.pth@150 | 0.3374 | 0.4455 | 0.6516 | 0.3979 | 0.5198 | -0.7577 | 0.0866 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b3_pga_norm_noamp/eval_results.txt` |
| b4 | full-model MSE loss | full_model_last.pth@150 | 0.3477 | 0.4468 | 0.6338 | 0.3944 | 0.4025 | -1.0441 | 0.2007 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b4_mse_noamp/eval_results.txt` |
| b5 | full-model Huber delta 3.0 | full_model_last.pth@150 | 0.3606 | 0.4725 | 0.6330 | 0.3228 | 0.5403 | -0.6673 | 0.1428 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b5_huber3_noamp/eval_results.txt` |
| b6 | station embedding decorrelation weight 1e-4 | full_model_last.pth@150 | 0.3855 | 0.4952 | 0.6224 | 0.2559 | 0.4630 | -1.0904 | 0.2678 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b6_station_decor_1em4_noamp/eval_results.txt` |
| b7 | PGA target cross-attention with distance bias / relative geometry | full_model_last.pth@150 | 0.3693 | 0.4774 | 0.6248 | 0.3084 | 0.4564 | -1.0579 | 0.2413 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b7_relative_geometry_noamp/eval_results.txt` |
| b8 | absolute coords add + amplitude scale path enabled | full_model_best.pth@106 | 0.3522 | 0.4586 | 0.6148 | 0.3619 | 0.4548 | -0.9198 | 0.2178 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_best.txt` |
| b9 | relative coords add, no amplitude | full_model_best.pth@123 | 0.3917 | 0.4978 | 0.5107 | 0.2482 | 0.3179 | -1.1311 | 0.2717 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b9_rel_add_noamp/eval_results_best.txt` |
| b10 | relative+absolute coords add, abs weight 0.01 | full_model_best.pth@128 | 0.3918 | 0.5004 | 0.5149 | 0.2403 | 0.3446 | -1.1093 | 0.2378 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b10_rel_abs_w001_add_noamp/eval_results_best.txt` |
| b11 | relative+absolute coords add, abs weight 0.03 | full_model_best.pth@115 | 0.4045 | 0.5095 | 0.4889 | 0.2124 | 0.3049 | -1.2102 | 0.2765 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b11_rel_abs_w003_add_noamp/eval_results_best.txt` |
| b12 | relative+absolute coords add, abs weight 0.10 | full_model_best.pth@106 | 0.4097 | 0.5154 | 0.4725 | 0.1942 | 0.2996 | -1.1979 | 0.2733 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b12_rel_abs_w010_add_noamp/eval_results_best.txt` |
| b13 | relative+absolute coords add, abs weight 0.30 | full_model_best.pth@143 | 0.3925 | 0.4918 | 0.5290 | 0.2663 | 0.3401 | -1.1127 | 0.2169 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b13_rel_abs_w030_add_noamp/eval_results_best.txt` |
| b14 | relative+absolute coords add, abs weight 1.00 | full_model_last.pth@150 | 0.3653 | 0.4635 | 0.6043 | 0.3482 | 0.3988 | -0.9334 | 0.1081 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b14_rel_abs_w100_add_noamp/eval_results_last.txt` |
| b15 | absolute coords concat, no amplitude | full_model_best.pth@132 | 0.3621 | 0.4792 | 0.5912 | 0.3033 | 0.4702 | -0.9208 | 0.2769 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b15_abs_concat_noamp/eval_results_best.txt` |
| b16 | relative coords concat, no amplitude | full_model_best.pth@133 | 0.4211 | 0.5207 | 0.4270 | 0.1775 | 0.1986 | -1.3683 | 0.3725 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b16_rel_concat_noamp/eval_results_best.txt` |
| b17 | relative+absolute coords concat, abs weight 0.10 | full_model_best.pth@67 | 0.4536 | 0.5540 | 0.3475 | 0.0690 | 0.1566 | -1.5212 | 0.4353 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b17_rel_abs_w010_concat_noamp/eval_results_best.txt` |

## Ranking By Validation MAE

1. b3 `b3_pga_norm_noamp`: MAE 0.3374, RMSE 0.4455, Corr 0.6516, R2 0.3979, slope 0.5198
2. b4 `b4_mse_noamp`: MAE 0.3477, RMSE 0.4468, Corr 0.6338, R2 0.3944, slope 0.4025
3. b8 `b8_abs_add_amp`: MAE 0.3522, RMSE 0.4586, Corr 0.6148, R2 0.3619, slope 0.4548
4. b1 `b1_single_pga08_noamp`: MAE 0.3531, RMSE 0.4538, Corr 0.6627, R2 0.3751, slope 0.4882
5. b5 `b5_huber3_noamp`: MAE 0.3606, RMSE 0.4725, Corr 0.6330, R2 0.3228, slope 0.5403
6. b15 `b15_abs_concat_noamp`: MAE 0.3621, RMSE 0.4792, Corr 0.5912, R2 0.3033, slope 0.4702
7. b14 `b14_rel_abs_w100_add_noamp`: MAE 0.3653, RMSE 0.4635, Corr 0.6043, R2 0.3482, slope 0.3988
8. b7 `b7_relative_geometry_noamp`: MAE 0.3693, RMSE 0.4774, Corr 0.6248, R2 0.3084, slope 0.4564
9. b2 `b2_single_pga_only_noamp`: MAE 0.3784, RMSE 0.4856, Corr 0.6147, R2 0.2846, slope 0.5039
10. b6 `b6_station_decor_1em4_noamp`: MAE 0.3855, RMSE 0.4952, Corr 0.6224, R2 0.2559, slope 0.4630
11. b9 `b9_rel_add_noamp`: MAE 0.3917, RMSE 0.4978, Corr 0.5107, R2 0.2482, slope 0.3179
12. b10 `b10_rel_abs_w001_add_noamp`: MAE 0.3918, RMSE 0.5004, Corr 0.5149, R2 0.2403, slope 0.3446
13. b13 `b13_rel_abs_w030_add_noamp`: MAE 0.3925, RMSE 0.4918, Corr 0.5290, R2 0.2663, slope 0.3401
14. b0 `b0_baseline_noamp`: MAE 0.3938, RMSE 0.5076, Corr 0.6007, R2 0.2184, slope 0.5269
15. b11 `b11_rel_abs_w003_add_noamp`: MAE 0.4045, RMSE 0.5095, Corr 0.4889, R2 0.2124, slope 0.3049
16. b12 `b12_rel_abs_w010_add_noamp`: MAE 0.4097, RMSE 0.5154, Corr 0.4725, R2 0.1942, slope 0.2996
17. b16 `b16_rel_concat_noamp`: MAE 0.4211, RMSE 0.5207, Corr 0.4270, R2 0.1775, slope 0.1986
18. b17 `b17_rel_abs_w010_concat_noamp`: MAE 0.4536, RMSE 0.5540, Corr 0.3475, R2 0.0690, slope 0.1566

## Data Sources For Review

Primary metrics:

- b0-b7: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/eval_results.txt`
- b8-b17 best checkpoint eval: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/eval_results_best.txt`
- b8-b17 last checkpoint eval: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/eval_results_last.txt`

Config sources:

- `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/config.json`

Training diagnostics available in each result directory:

- `train_epoch_loss.csv`
- `val_epoch_loss.csv`
- `train_loss.csv`
- `diag_*.csv`
- `data_*.csv`
- `manifest.json`

Shared split source referenced by configs:

- `pga_configs/stage2_512_event_ids.txt`

Important audit note: the inspected local result copy does not contain this event-id file, so reviewers should verify it in the original training workspace.

## Appendix: b8-b17 Best vs Last

| Exp | Ckpt | Epoch | Val MAE | RMSE | Corr | R2 | Slope | Train MAE | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| b8 | best | 106 | 0.3522 | 0.4586 | 0.6148 | 0.3619 | 0.4548 | 0.2178 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_best.txt` |
| b8 | last | 150 | 0.3715 | 0.4805 | 0.6373 | 0.2995 | 0.4814 | 0.2560 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_last.txt` |
| b9 | best | 123 | 0.3917 | 0.4978 | 0.5107 | 0.2482 | 0.3179 | 0.2717 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b9_rel_add_noamp/eval_results_best.txt` |
| b9 | last | 150 | 0.4040 | 0.5158 | 0.5048 | 0.1929 | 0.3752 | 0.2036 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b9_rel_add_noamp/eval_results_last.txt` |
| b10 | best | 128 | 0.3918 | 0.5004 | 0.5149 | 0.2403 | 0.3446 | 0.2378 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b10_rel_abs_w001_add_noamp/eval_results_best.txt` |
| b10 | last | 150 | 0.4338 | 0.5529 | 0.5158 | 0.0727 | 0.3940 | 0.2012 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b10_rel_abs_w001_add_noamp/eval_results_last.txt` |
| b11 | best | 115 | 0.4045 | 0.5095 | 0.4889 | 0.2124 | 0.3049 | 0.2765 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b11_rel_abs_w003_add_noamp/eval_results_best.txt` |
| b11 | last | 150 | 0.4131 | 0.5338 | 0.5052 | 0.1356 | 0.3606 | 0.1845 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b11_rel_abs_w003_add_noamp/eval_results_last.txt` |
| b12 | best | 106 | 0.4097 | 0.5154 | 0.4725 | 0.1942 | 0.2996 | 0.2733 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b12_rel_abs_w010_add_noamp/eval_results_best.txt` |
| b12 | last | 150 | 0.4568 | 0.5743 | 0.5062 | -0.0007 | 0.4021 | 0.2251 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b12_rel_abs_w010_add_noamp/eval_results_last.txt` |
| b13 | best | 143 | 0.3925 | 0.4918 | 0.5290 | 0.2663 | 0.3401 | 0.2169 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b13_rel_abs_w030_add_noamp/eval_results_best.txt` |
| b13 | last | 150 | 0.4318 | 0.5366 | 0.5253 | 0.1265 | 0.3311 | 0.3146 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b13_rel_abs_w030_add_noamp/eval_results_last.txt` |
| b14 | best | 103 | 0.3703 | 0.4684 | 0.6059 | 0.3343 | 0.3911 | 0.1861 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b14_rel_abs_w100_add_noamp/eval_results_best.txt` |
| b14 | last | 150 | 0.3653 | 0.4635 | 0.6043 | 0.3482 | 0.3988 | 0.1081 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b14_rel_abs_w100_add_noamp/eval_results_last.txt` |
| b15 | best | 132 | 0.3621 | 0.4792 | 0.5912 | 0.3033 | 0.4702 | 0.2769 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b15_abs_concat_noamp/eval_results_best.txt` |
| b15 | last | 150 | 0.3643 | 0.4795 | 0.5818 | 0.3026 | 0.4451 | 0.2538 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b15_abs_concat_noamp/eval_results_last.txt` |
| b16 | best | 133 | 0.4211 | 0.5207 | 0.4270 | 0.1775 | 0.1986 | 0.3725 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b16_rel_concat_noamp/eval_results_best.txt` |
| b16 | last | 150 | 0.4311 | 0.5307 | 0.4133 | 0.1455 | 0.2358 | 0.3536 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b16_rel_concat_noamp/eval_results_last.txt` |
| b17 | best | 67 | 0.4536 | 0.5540 | 0.3475 | 0.0690 | 0.1566 | 0.4353 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b17_rel_abs_w010_concat_noamp/eval_results_best.txt` |
| b17 | last | 150 | 0.4578 | 0.5717 | 0.3842 | 0.0084 | 0.2720 | 0.3128 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b17_rel_abs_w010_concat_noamp/eval_results_last.txt` |

