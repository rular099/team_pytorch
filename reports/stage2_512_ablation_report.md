# Stage2 512-Event PGA Ablation Report

Date: 2026-05-17

This report summarizes the 512-event stage2 PGA ablation results for experiments b0-b29.
The goal is to identify which tested factors are useful, how the b19-b24 focused follow-up
changes the earlier b0-b18 interpretation, what b25-b29 show about strong-PGA fitting, and
what should be tested next.

## Scope And Caveats

- Metrics were recomputed from `eval_results*.npz` under `../chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/`.
- `Bias` means `mean(pred - label)` on valid PGA targets. This differs from earlier drafts where the values looked like the linear-fit intercept.
- Because this is a 512-event overfit/small-sample experiment, train-set metrics are part of the diagnosis, not just auxiliary logging. A useful run should first demonstrate that it can fit the selected train events, then be compared on validation behavior.
- b0-b7 currently have one local eval file each: `eval_results.npz` / `eval_results.txt`, loaded from `full_model_last.pth`.
- b8-b29 have both `eval_results_best.*` and `eval_results_last.*`; the main table reports b0-b24 by lower validation MAE, while b25-b29 are reported separately because their purpose is strong-PGA fitting rather than global validation MAE.
- This means b0-b7 and b8-b29 are not perfectly matched by checkpoint selection policy. The source file column records the exact file used for every row.
- All configs point to the shared event list path `pga_configs/stage2_512_event_ids.txt`. The local result bundle inspected here did not include `overfit_event_ids.txt` or `stage2_512_event_ids.txt`; reviewers should verify the original supercomputer workspace has the 512-line shared event list.
- Strong-PGA conclusions are noisier than global metrics. In the validation set there are 757 valid PGA targets, but only 88 targets with `label >= -1` and only 2 targets with `label >= 0`.

## Overall Result

The lowest validation MAE is now **b21_pga_norm_amp_pga08**:

- Validation MAE: **0.3251**
- Validation RMSE: **0.4327**
- Validation Corr: **0.6714**
- Validation R2: **0.4321**
- Validation slope: **0.5415**
- Validation bias: **-0.0127**
- Source: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b21_pga_norm_amp_pga08/eval_results_last.txt`

The cleanest new single-factor anchor is **b19_pga_norm_amp_abs_add**:

- Validation MAE: **0.3265**
- Validation RMSE: **0.4306**
- Validation Corr: **0.6777**
- Validation R2: **0.4374**
- Validation slope: **0.5587**
- Validation bias: **0.0115**
- Source: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b19_pga_norm_amp_abs_add/eval_results_last.txt`

b19 is `b3 + amplitude` with the same absolute-coordinate add fusion. It improves b3 by MAE
(`0.3374 -> 0.3265`) and also improves RMSE, Corr, R2, and slope. b21 has slightly lower
MAE, but it mixes amplitude with PGA-heavy single-station pretraining and has worse top-20%
strong-PGA bias than b19.

b5 remains the best strong-PGA reference by top-20% MAE and bias, even though it is not the
best global model.

For the strong-PGA follow-up, the user-selected model is **b29_pga_norm_amp_pga08_strongw2**
at the last checkpoint, despite worse global validation MAE:

- Train MAE: **0.2260**
- Validation MAE: **0.4252**
- Top-20% strong-PGA MAE: **0.2756**
- Top-20% strong-PGA bias: **-0.1265**
- `label >= -1` MAE: **0.2832**
- `label >= -1` bias: **-0.2019**

This should be interpreted as a deliberately strong-PGA-oriented checkpoint, not as the best
global validation model.

On train-set fit capacity, the leading candidates also look credible:

- b19: train MAE **0.0760**, train R2 **0.9531**, train slope **0.9558**
- b21: train MAE **0.0830**, train R2 **0.9472**, train slope **0.9398**
- b24: train MAE **0.0804**, train R2 **0.9517**, train slope **0.9627**
- b3: train MAE **0.0866**, train R2 **0.9453**, train slope **0.9180**

This matters because several weaker coordinate/fusion variants are not merely worse on validation;
they also fit the train subset poorly. For example, b16 has train MAE 0.3725 and b17 has train
MAE 0.4353, so their failure is already visible as poor overfit capacity.

## Overfit Fit-Capacity Check

This table ranks the best train-fit runs using the same checkpoint reported in the main validation
table. `Gap` is `Val MAE - Train MAE`; it is not a generalization estimate for deployment, but it
helps separate "can fit the selected events" from "also keeps validation error low".

| Rank | Exp | Reported ckpt | Train MAE | Train RMSE | Train R2 | Train Slope | Val MAE | Gap |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | b19 | last@150 | 0.0760 | 0.1289 | 0.9531 | 0.9558 | 0.3265 | 0.2505 |
| 2 | b24 | last@150 | 0.0804 | 0.1309 | 0.9517 | 0.9627 | 0.3328 | 0.2524 |
| 3 | b21 | last@150 | 0.0830 | 0.1367 | 0.9472 | 0.9398 | 0.3251 | 0.2421 |
| 4 | b3 | eval@150 | 0.0866 | 0.1394 | 0.9453 | 0.9180 | 0.3374 | 0.2507 |
| 5 | b20 | best@131 | 0.0925 | 0.1500 | 0.9367 | 0.9088 | 0.3407 | 0.2482 |
| 6 | b22 | best@132 | 0.1058 | 0.1644 | 0.9239 | 0.8666 | 0.3320 | 0.2262 |
| 7 | b14 | last@150 | 0.1081 | 0.1727 | 0.9161 | 0.8880 | 0.3653 | 0.2572 |
| 8 | b18 | last@150 | 0.1282 | 0.1996 | 0.8874 | 0.8123 | 0.3542 | 0.2260 |
| 9 | b23 | best@110 | 0.1358 | 0.2148 | 0.8702 | 0.8309 | 0.3498 | 0.2139 |
| 10 | b5 | eval@150 | 0.1428 | 0.2093 | 0.8767 | 0.9055 | 0.3606 | 0.2177 |

Train-fit interpretation:

- b19 is strongest under the combined criterion: best train MAE among reported checkpoints, best RMSE/Corr/R2/slope among b19-b24 on validation, and a clean single-factor delta over b3.
- b21 remains the lowest validation-MAE run and also fits train well, but it is a mixed-factor setting and has worse strong-PGA bias than b19.
- b24 is important as a fit-capacity reference: it fits train nearly as well as b19 and has the best train slope among the reported checkpoints, but it does not beat b19/b21 on validation MAE.
- b16/b17 should not be treated as "validation-only" failures. Their train MAE is very high, which suggests the relative-concat representation is limiting fitting/optimization before generalization is even considered.

## b19-b24 Interpretation

b19-b24 were designed as focused deltas around b3:

```text
b19 = b3 + amplitude, absolute coords add
b20 = b3 + PGA-heavy single-station pretrain
b21 = b3 + amplitude + PGA-heavy single-station pretrain
b22 = b3 + normalized MSE
b23 = b3 + Huber delta 0.5
b24 = b3 + Huber delta 2.0
```

Observed behavior:

- b19 and b21 both beat b3 by validation MAE; b19 has the best RMSE, Corr, R2, and slope among b19-b24.
- b19, b21, and b24 all demonstrate strong train-set fit, so their validation differences are not caused by obvious underfitting on the selected train events.
- b20 does not beat b3, so PGA-heavy pretrain alone is not enough under PGA target normalization.
- b22 is competitive by MAE/RMSE/R2, but its slope is lower than b19/b21/b24.
- b23 is not competitive; Huber delta 0.5 is too conservative for this setting.
- b24 is competitive and has good top-20% slope, but it does not beat b19/b21 globally.
- The earlier b18 bundle remains a useful warning: adding amplitude, concat fusion, and PGA-heavy pretrain together did not produce an additive gain.
- Strong-PGA underprediction remains unresolved across b19-b24.

## Recommended Next Tests

Use **b19_pga_norm_amp_abs_add** as the clean main anchor if interpretability and single-factor
progress matter. Keep **b21_pga_norm_amp_pga08** as the lowest-MAE candidate, but treat it as a
mixed-factor model.

Recommended next steps:

1. Score each overfit run on both train-fit capacity and validation behavior; do not rank by validation MAE alone.
2. Treat b29-last as the current strong-PGA candidate because the current priority is strong-PGA fitting rather than validation MAE.
3. Stop the loss/sampling sweep for now; b25-b29 show that strong-PGA bias can be moved, and b29-last is satisfactory for the current purpose.
4. Run the next experiment block on position/site information: VS30 plus TEAM station self-attention RoPE.
5. Use b19 as the clean mechanism anchor and b29 as the strong-PGA anchor when testing VS30/RoPE.
6. Add a station-context control for RoPE, because b19/b29 currently use cross-attention readouts that bypass TEAM self-attention in the PGA path.
7. Run station/spatial holdout before interpreting absolute-coordinate gains as transferable site/path information.

## Factor Conclusions

### Target Normalization

Best setting: **enable PGA target normalization**.

b3 remains the key enabling change. It improves b0 from 0.3938 MAE to 0.3374 MAE and improves R2 from 0.2184 to 0.3979. The new best runs, b19 and b21, both build on b3, so the conclusion is stronger than before: PGA target normalization should stay in the mainline.

### Single-Station Pretrain Weighting

Best tested standalone setting without normalization: **PGA-heavy multitask pretrain**, b1.

b1 improves b0 clearly: 0.3531 MAE vs 0.3938. PGA-only pretrain b2 also helps but is weaker than b1. Keeping mag/epidist auxiliary tasks with higher PGA weight appears better than dropping them completely.

Under PGA target normalization, the result is weaker. b20, which tests PGA-heavy pretrain alone on top of b3, reaches 0.3407 MAE and does not beat b3. b21 combines PGA-heavy pretrain with amplitude and reaches the lowest MAE, but b19 shows that most of that gain can be obtained cleanly from amplitude alone.

### Full-Model Loss

Best global normalized loss follow-up: **MSE**, b22, but not by enough to replace b19/b21.

b22 reaches 0.3320 MAE, 0.4317 RMSE, and 0.4345 R2, which is competitive with b19/b21. Its slope is only 0.5008, lower than b19, b21, and b24. Huber delta 0.5, b23, is not competitive. Huber delta 2.0, b24, is competitive and has good top-20% slope, but it does not beat b19/b21 globally. Huber delta 3.0, b5, still gives the best strong-PGA top-20% MAE and bias, but worse global MAE.

### Amplitude Information

Best setting: **use amplitude information with PGA target normalization**, b19.

b8 improves the baseline substantially:

- b0: MAE 0.3938, RMSE 0.5076, R2 0.2184
- b8 best: MAE 0.3522, RMSE 0.4586, R2 0.3619

The b8 best checkpoint also has near-zero global bias, but its slope is only 0.4548. b19 shows that amplitude remains useful after target normalization: b3 improves from 0.3374 MAE to 0.3265 MAE, with better RMSE, Corr, R2, and slope. Amplitude helps average error, but it still does not solve strong-PGA compression by itself.

### Coordinate Type

Best tested simple coordinate type: **absolute coordinates**.

Pure relative coordinates, b9, are not better than b0 in a useful way and have much weaker corr/R2/slope:

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

Best tested concat setting: **absolute-coordinate concat**, b15.

For absolute coordinates, concat improves the baseline:

- b0 abs add: MAE 0.3938
- b15 abs concat: MAE 0.3621

For relative coordinates, concat is harmful:

- b9 rel add: MAE 0.3917
- b16 rel concat: MAE 0.4211

For rel+abs at w=0.1, concat is much worse:

- b12 rel+abs add w=0.1: MAE 0.4097
- b17 rel+abs concat w=0.1: MAE 0.4536

Concat should only be carried forward for absolute coordinates unless a new design changes how relative coordinates are represented. Even for absolute coordinates, b18 shows that concat is not the preferred follow-up once PGA target normalization and amplitude are active: b18 is worse than the simpler b19 absolute-add model.

### Distance Bias / Relative Geometry

b7 improves b0 by MAE and R2, but is not among the strongest stage2 results:

- b7: MAE 0.3693, R2 0.3084
- b0: MAE 0.3938, R2 0.2184

It remains a useful secondary factor, but should not take priority over target normalization, amplitude, or the b19/b21 line.

### Station Embedding Decorrelation

b6 is slightly better than b0 by MAE but not competitive with b1/b3/b4/b8/b15/b18/b19/b21:

- b6: MAE 0.3855
- b0: MAE 0.3938

It is not a priority for the next experiment batch.

## Strong-PGA Check

This table uses the top 20% validation PGA labels. Positive bias means overprediction; negative bias means underprediction.

| Exp | Ckpt | Top20 MAE | Top20 Bias | Top20 Slope |
|---|---|---:|---:|---:|
| b0 | eval | 0.5826 | -0.5661 | 0.3832 |
| b1 | eval | 0.5713 | -0.5597 | 0.3907 |
| b3 | eval | 0.4541 | -0.3987 | 0.3847 |
| b4 | eval | 0.5537 | -0.5420 | 0.3991 |
| b5 | eval | 0.3845 | -0.3207 | 0.4267 |
| b8 | best | 0.4890 | -0.4685 | 0.3879 |
| b14 | last | 0.4430 | -0.4205 | 0.1516 |
| b15 | best | 0.5258 | -0.4934 | 0.3633 |
| b18 | last | 0.4435 | -0.4114 | 0.5098 |
| b19 | last | 0.4501 | -0.3861 | 0.3881 |
| b20 | best | 0.4910 | -0.4476 | 0.3618 |
| b21 | last | 0.4773 | -0.4318 | 0.4023 |
| b22 | best | 0.4876 | -0.4551 | 0.4171 |
| b23 | best | 0.4942 | -0.4510 | 0.4612 |
| b24 | last | 0.4550 | -0.4065 | 0.4751 |

Strong-PGA takeaway:

- b5 is the best strong-PGA run by top20 MAE and bias.
- b19 is the best b19-b24 run by top20 MAE and bias, but it only slightly improves b3 on this slice.
- b21 is the best global-MAE model, but its top20 MAE and bias are worse than b19.
- Strong-PGA underprediction remains substantial. b5 should be treated as the strong-PGA reference, not as the global best model.

## b25-b29 Strong-PGA Follow-Up

b25-b29 were designed to test whether explicit strong-PGA target sampling or loss weighting can
move the high-label bins. These runs should not be ranked by global validation MAE alone.

| Exp | Ckpt | Train MAE | Val MAE | Slope | Bias | Top20 MAE | Top20 Bias | `label>=-1` MAE | `label>=-1` Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b25 strongw2 | last | 0.1126 | 0.3570 | 0.5528 | 0.0016 | 0.4443 | -0.4099 | 0.5079 | -0.4938 |
| b26 strongw4 | last | 0.1630 | 0.3867 | 0.5181 | 0.1136 | 0.3927 | -0.3108 | 0.4212 | -0.3783 |
| b27 labelstrat | best | 0.2146 | 0.3475 | 0.5456 | 0.0509 | 0.3699 | -0.3309 | 0.4211 | -0.3979 |
| b28 labelstrat+strongw2 | best | 0.2780 | 0.3835 | 0.5109 | 0.0741 | 0.3435 | -0.3010 | 0.3795 | -0.3574 |
| b29 b21+strongw2 | best | 0.1084 | 0.3467 | 0.5345 | 0.0638 | 0.4004 | -0.3512 | 0.4354 | -0.4139 |
| b29 b21+strongw2 | last | 0.2260 | 0.4252 | 0.5383 | 0.3007 | 0.2756 | -0.1265 | 0.2832 | -0.2019 |

Interpretation:

- b29-last is the current user-approved strong-PGA candidate. It gives the best top20 MAE and the least negative strong-PGA bias among b25-b29, at the cost of worse global validation MAE and positive global bias.
- b28/b29 show that the strong-PGA underprediction can be moved by sampling/weighting. The side effect is overprediction in weaker PGA ranges, so these are not clean global models.
- The next priority is no longer another strong-PGA loss sweep. The useful next question is whether location/site information, especially VS30 and TEAM station self-attention RoPE, improves the mechanism rather than simply memorizing seen station locations.

## Main Metrics Table

| Exp | Tested factor | Reported ckpt | Val MAE | RMSE | Corr | R2 | Slope | Bias | Train MAE | Source eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b0 | baseline: abs coord add, no amplitude, Huber1, multitask single-station pretrain | full_model_last.pth@150 | 0.3938 | 0.5076 | 0.6007 | 0.2184 | 0.5269 | -0.1477 | 0.2316 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b0_baseline_noamp/eval_results.txt` |
| b1 | single-station pretrain PGA-heavy weights 0.8/0.1/0.1 | full_model_last.pth@150 | 0.3531 | 0.4538 | 0.6627 | 0.3751 | 0.4882 | -0.1390 | 0.1933 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b1_single_pga08_noamp/eval_results.txt` |
| b2 | single-station pretrain PGA-only | full_model_last.pth@150 | 0.3784 | 0.4856 | 0.6147 | 0.2846 | 0.5039 | -0.1299 | 0.1909 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b2_single_pga_only_noamp/eval_results.txt` |
| b3 | PGA target normalization enabled | full_model_last.pth@150 | 0.3374 | 0.4455 | 0.6516 | 0.3979 | 0.5198 | 0.0419 | 0.0866 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b3_pga_norm_noamp/eval_results.txt` |
| b4 | full-model MSE loss | full_model_last.pth@150 | 0.3477 | 0.4468 | 0.6338 | 0.3944 | 0.4025 | -0.0491 | 0.2007 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b4_mse_noamp/eval_results.txt` |
| b5 | full-model Huber delta 3.0 | full_model_last.pth@150 | 0.3606 | 0.4725 | 0.6330 | 0.3228 | 0.5403 | 0.0984 | 0.1428 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b5_huber3_noamp/eval_results.txt` |
| b6 | station embedding decorrelation weight 1e-4 | full_model_last.pth@150 | 0.3855 | 0.4952 | 0.6224 | 0.2559 | 0.4630 | -0.1962 | 0.2678 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b6_station_decor_1em4_noamp/eval_results.txt` |
| b7 | PGA target cross-attention with distance bias / relative geometry | full_model_last.pth@150 | 0.3693 | 0.4774 | 0.6248 | 0.3084 | 0.4564 | -0.1527 | 0.2413 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b7_relative_geometry_noamp/eval_results.txt` |
| b8 | absolute coords add + amplitude scale path enabled | full_model_best.pth@106 | 0.3522 | 0.4586 | 0.6148 | 0.3619 | 0.4548 | -0.0119 | 0.2178 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_best.txt` |
| b9 | relative coords add, no amplitude | full_model_best.pth@123 | 0.3917 | 0.4978 | 0.5107 | 0.2482 | 0.3179 | 0.0047 | 0.2717 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b9_rel_add_noamp/eval_results_best.txt` |
| b10 | relative+absolute coords add, abs weight 0.01 | full_model_best.pth@128 | 0.3918 | 0.5004 | 0.5149 | 0.2403 | 0.3446 | -0.0177 | 0.2378 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b10_rel_abs_w001_add_noamp/eval_results_best.txt` |
| b11 | relative+absolute coords add, abs weight 0.03 | full_model_best.pth@115 | 0.4045 | 0.5095 | 0.4889 | 0.2124 | 0.3049 | -0.0527 | 0.2765 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b11_rel_abs_w003_add_noamp/eval_results_best.txt` |
| b12 | relative+absolute coords add, abs weight 0.10 | full_model_best.pth@106 | 0.4097 | 0.5154 | 0.4725 | 0.1942 | 0.2996 | -0.0315 | 0.2733 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b12_rel_abs_w010_add_noamp/eval_results_best.txt` |
| b13 | relative+absolute coords add, abs weight 0.30 | full_model_best.pth@143 | 0.3925 | 0.4918 | 0.5290 | 0.2663 | 0.3401 | -0.0137 | 0.2169 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b13_rel_abs_w030_add_noamp/eval_results_best.txt` |
| b14 | relative+absolute coords add, abs weight 1.00 | full_model_last.pth@150 | 0.3653 | 0.4635 | 0.6043 | 0.3482 | 0.3988 | 0.0678 | 0.1081 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b14_rel_abs_w100_add_noamp/eval_results_last.txt` |
| b15 | absolute coords concat, no amplitude | full_model_best.pth@132 | 0.3621 | 0.4792 | 0.5912 | 0.3033 | 0.4702 | -0.0385 | 0.2769 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b15_abs_concat_noamp/eval_results_best.txt` |
| b16 | relative coords concat, no amplitude | full_model_best.pth@133 | 0.4211 | 0.5207 | 0.4270 | 0.1775 | 0.1986 | -0.0337 | 0.3725 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b16_rel_concat_noamp/eval_results_best.txt` |
| b17 | relative+absolute coords concat, abs weight 0.10 | full_model_best.pth@67 | 0.4536 | 0.5540 | 0.3475 | 0.0690 | 0.1566 | -0.1166 | 0.4353 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b17_rel_abs_w010_concat_noamp/eval_results_best.txt` |
| b18 | PGA norm + amplitude + absolute coords concat + single-station PGA weight 0.8 | full_model_last.pth@150 | 0.3542 | 0.4791 | 0.6087 | 0.3038 | 0.5268 | -0.0158 | 0.1282 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b18_pga_norm_amp_abs_concat_pga08/eval_results_last.txt` |
| b19 | b3 + amplitude, absolute coords add | full_model_last.pth@150 | 0.3265 | 0.4306 | 0.6777 | 0.4374 | 0.5587 | 0.0115 | 0.0760 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b19_pga_norm_amp_abs_add/eval_results_last.txt` |
| b20 | b3 + single-station PGA-heavy weights 0.8/0.1/0.1 | full_model_best.pth@131 | 0.3407 | 0.4439 | 0.6475 | 0.4022 | 0.5026 | -0.0123 | 0.0925 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b20_pga_norm_pga08_noamp/eval_results_best.txt` |
| b21 | b3 + amplitude + single-station PGA-heavy weights 0.8/0.1/0.1 | full_model_last.pth@150 | 0.3251 | 0.4327 | 0.6714 | 0.4321 | 0.5415 | -0.0127 | 0.0830 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b21_pga_norm_amp_pga08/eval_results_last.txt` |
| b22 | b3 + full-model MSE loss | full_model_best.pth@132 | 0.3320 | 0.4317 | 0.6650 | 0.4345 | 0.5008 | -0.0030 | 0.1058 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b22_pga_norm_mse_noamp/eval_results_best.txt` |
| b23 | b3 + Huber delta 0.5 | full_model_best.pth@110 | 0.3498 | 0.4571 | 0.6344 | 0.3660 | 0.5226 | -0.0130 | 0.1358 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b23_pga_norm_huber05_noamp/eval_results_best.txt` |
| b24 | b3 + Huber delta 2.0 | full_model_last.pth@150 | 0.3328 | 0.4390 | 0.6643 | 0.4152 | 0.5450 | 0.0230 | 0.0804 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b24_pga_norm_huber2_noamp/eval_results_last.txt` |

## Ranking By Validation MAE

1. b21 `b21_pga_norm_amp_pga08`: MAE 0.3251, RMSE 0.4327, Corr 0.6714, R2 0.4321, slope 0.5415, bias -0.0127
2. b19 `b19_pga_norm_amp_abs_add`: MAE 0.3265, RMSE 0.4306, Corr 0.6777, R2 0.4374, slope 0.5587, bias 0.0115
3. b22 `b22_pga_norm_mse_noamp`: MAE 0.3320, RMSE 0.4317, Corr 0.6650, R2 0.4345, slope 0.5008, bias -0.0030
4. b24 `b24_pga_norm_huber2_noamp`: MAE 0.3328, RMSE 0.4390, Corr 0.6643, R2 0.4152, slope 0.5450, bias 0.0230
5. b3 `b3_pga_norm_noamp`: MAE 0.3374, RMSE 0.4455, Corr 0.6516, R2 0.3979, slope 0.5198, bias 0.0419
6. b20 `b20_pga_norm_pga08_noamp`: MAE 0.3407, RMSE 0.4439, Corr 0.6475, R2 0.4022, slope 0.5026, bias -0.0123
7. b4 `b4_mse_noamp`: MAE 0.3477, RMSE 0.4468, Corr 0.6338, R2 0.3944, slope 0.4025, bias -0.0491
8. b23 `b23_pga_norm_huber05_noamp`: MAE 0.3498, RMSE 0.4571, Corr 0.6344, R2 0.3660, slope 0.5226, bias -0.0130
9. b8 `b8_abs_add_amp`: MAE 0.3522, RMSE 0.4586, Corr 0.6148, R2 0.3619, slope 0.4548, bias -0.0119
10. b1 `b1_single_pga08_noamp`: MAE 0.3531, RMSE 0.4538, Corr 0.6627, R2 0.3751, slope 0.4882, bias -0.1390
11. b18 `b18_pga_norm_amp_abs_concat_pga08`: MAE 0.3542, RMSE 0.4791, Corr 0.6087, R2 0.3038, slope 0.5268, bias -0.0158
12. b5 `b5_huber3_noamp`: MAE 0.3606, RMSE 0.4725, Corr 0.6330, R2 0.3228, slope 0.5403, bias 0.0984
13. b15 `b15_abs_concat_noamp`: MAE 0.3621, RMSE 0.4792, Corr 0.5912, R2 0.3033, slope 0.4702, bias -0.0385
14. b14 `b14_rel_abs_w100_add_noamp`: MAE 0.3653, RMSE 0.4635, Corr 0.6043, R2 0.3482, slope 0.3988, bias 0.0678
15. b7 `b7_relative_geometry_noamp`: MAE 0.3693, RMSE 0.4774, Corr 0.6248, R2 0.3084, slope 0.4564, bias -0.1527
16. b2 `b2_single_pga_only_noamp`: MAE 0.3784, RMSE 0.4856, Corr 0.6147, R2 0.2846, slope 0.5039, bias -0.1299
17. b6 `b6_station_decor_1em4_noamp`: MAE 0.3855, RMSE 0.4952, Corr 0.6224, R2 0.2559, slope 0.4630, bias -0.1962
18. b9 `b9_rel_add_noamp`: MAE 0.3917, RMSE 0.4978, Corr 0.5107, R2 0.2482, slope 0.3179, bias 0.0047
19. b10 `b10_rel_abs_w001_add_noamp`: MAE 0.3918, RMSE 0.5004, Corr 0.5149, R2 0.2403, slope 0.3446, bias -0.0177
20. b13 `b13_rel_abs_w030_add_noamp`: MAE 0.3925, RMSE 0.4918, Corr 0.5290, R2 0.2663, slope 0.3401, bias -0.0137
21. b0 `b0_baseline_noamp`: MAE 0.3938, RMSE 0.5076, Corr 0.6007, R2 0.2184, slope 0.5269, bias -0.1477
22. b11 `b11_rel_abs_w003_add_noamp`: MAE 0.4045, RMSE 0.5095, Corr 0.4889, R2 0.2124, slope 0.3049, bias -0.0527
23. b12 `b12_rel_abs_w010_add_noamp`: MAE 0.4097, RMSE 0.5154, Corr 0.4725, R2 0.1942, slope 0.2996, bias -0.0315
24. b16 `b16_rel_concat_noamp`: MAE 0.4211, RMSE 0.5207, Corr 0.4270, R2 0.1775, slope 0.1986, bias -0.0337
25. b17 `b17_rel_abs_w010_concat_noamp`: MAE 0.4536, RMSE 0.5540, Corr 0.3475, R2 0.0690, slope 0.1566, bias -0.1166

## Data Sources For Review

Primary metrics:

- b0-b7: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/eval_results.txt`
- b8-b29 best checkpoint eval: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/eval_results_best.txt`
- b8-b29 last checkpoint eval: `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b*/eval_results_last.txt`

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

## Appendix: b8-b24 Global-Metric Best vs Last

b25-b29 best/last strong-PGA rows are summarized in the dedicated strong-PGA follow-up section above.

| Exp | Ckpt | Epoch | Val MAE | RMSE | Corr | R2 | Slope | Bias | Train MAE | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b8 | best | 106 | 0.3522 | 0.4586 | 0.6148 | 0.3619 | 0.4548 | -0.0119 | 0.2178 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_best.txt` |
| b8 | last | 150 | 0.3715 | 0.4805 | 0.6373 | 0.2995 | 0.4814 | -0.1748 | 0.2560 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b8_abs_add_amp/eval_results_last.txt` |
| b9 | best | 123 | 0.3917 | 0.4978 | 0.5107 | 0.2482 | 0.3179 | 0.0047 | 0.2717 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b9_rel_add_noamp/eval_results_best.txt` |
| b9 | last | 150 | 0.4040 | 0.5158 | 0.5048 | 0.1929 | 0.3752 | 0.0404 | 0.2036 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b9_rel_add_noamp/eval_results_last.txt` |
| b10 | best | 128 | 0.3918 | 0.5004 | 0.5149 | 0.2403 | 0.3446 | -0.0177 | 0.2378 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b10_rel_abs_w001_add_noamp/eval_results_best.txt` |
| b10 | last | 150 | 0.4338 | 0.5529 | 0.5158 | 0.0727 | 0.3940 | 0.2085 | 0.2012 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b10_rel_abs_w001_add_noamp/eval_results_last.txt` |
| b11 | best | 115 | 0.4045 | 0.5095 | 0.4889 | 0.2124 | 0.3049 | -0.0527 | 0.2765 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b11_rel_abs_w003_add_noamp/eval_results_best.txt` |
| b11 | last | 150 | 0.4131 | 0.5338 | 0.5052 | 0.1356 | 0.3606 | 0.1584 | 0.1845 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b11_rel_abs_w003_add_noamp/eval_results_last.txt` |
| b12 | best | 106 | 0.4097 | 0.5154 | 0.4725 | 0.1942 | 0.2996 | -0.0315 | 0.2733 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b12_rel_abs_w010_add_noamp/eval_results_best.txt` |
| b12 | last | 150 | 0.4568 | 0.5743 | 0.5062 | -0.0007 | 0.4021 | 0.2393 | 0.2251 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b12_rel_abs_w010_add_noamp/eval_results_last.txt` |
| b13 | best | 143 | 0.3925 | 0.4918 | 0.5290 | 0.2663 | 0.3401 | -0.0137 | 0.2169 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b13_rel_abs_w030_add_noamp/eval_results_best.txt` |
| b13 | last | 150 | 0.4318 | 0.5366 | 0.5253 | 0.1265 | 0.3311 | -0.2136 | 0.3146 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b13_rel_abs_w030_add_noamp/eval_results_last.txt` |
| b14 | best | 103 | 0.3703 | 0.4684 | 0.6059 | 0.3343 | 0.3911 | -0.1015 | 0.1861 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b14_rel_abs_w100_add_noamp/eval_results_best.txt` |
| b14 | last | 150 | 0.3653 | 0.4635 | 0.6043 | 0.3482 | 0.3988 | 0.0678 | 0.1081 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b14_rel_abs_w100_add_noamp/eval_results_last.txt` |
| b15 | best | 132 | 0.3621 | 0.4792 | 0.5912 | 0.3033 | 0.4702 | -0.0385 | 0.2769 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b15_abs_concat_noamp/eval_results_best.txt` |
| b15 | last | 150 | 0.3643 | 0.4795 | 0.5818 | 0.3026 | 0.4451 | -0.0285 | 0.2538 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b15_abs_concat_noamp/eval_results_last.txt` |
| b16 | best | 133 | 0.4211 | 0.5207 | 0.4270 | 0.1775 | 0.1986 | -0.0337 | 0.3725 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b16_rel_concat_noamp/eval_results_best.txt` |
| b16 | last | 150 | 0.4311 | 0.5307 | 0.4133 | 0.1455 | 0.2358 | -0.0131 | 0.3536 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b16_rel_concat_noamp/eval_results_last.txt` |
| b17 | best | 67 | 0.4536 | 0.5540 | 0.3475 | 0.0690 | 0.1566 | -0.1166 | 0.4353 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b17_rel_abs_w010_concat_noamp/eval_results_best.txt` |
| b17 | last | 150 | 0.4578 | 0.5717 | 0.3842 | 0.0084 | 0.2720 | 0.1065 | 0.3128 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b17_rel_abs_w010_concat_noamp/eval_results_last.txt` |
| b18 | best | 96 | 0.3614 | 0.4926 | 0.5989 | 0.2638 | 0.5424 | 0.0154 | 0.1564 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b18_pga_norm_amp_abs_concat_pga08/eval_results_best.txt` |
| b18 | last | 150 | 0.3542 | 0.4791 | 0.6087 | 0.3038 | 0.5268 | -0.0158 | 0.1282 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b18_pga_norm_amp_abs_concat_pga08/eval_results_last.txt` |
| b19 | best | 120 | 0.3272 | 0.4313 | 0.6709 | 0.4356 | 0.5185 | -0.0366 | 0.1081 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b19_pga_norm_amp_abs_add/eval_results_best.txt` |
| b19 | last | 150 | 0.3265 | 0.4306 | 0.6777 | 0.4374 | 0.5587 | 0.0115 | 0.0760 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b19_pga_norm_amp_abs_add/eval_results_last.txt` |
| b20 | best | 131 | 0.3407 | 0.4439 | 0.6475 | 0.4022 | 0.5026 | -0.0123 | 0.0925 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b20_pga_norm_pga08_noamp/eval_results_best.txt` |
| b20 | last | 150 | 0.3443 | 0.4461 | 0.6472 | 0.3964 | 0.5133 | 0.0198 | 0.0810 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b20_pga_norm_pga08_noamp/eval_results_last.txt` |
| b21 | best | 118 | 0.3322 | 0.4426 | 0.6495 | 0.4058 | 0.5038 | -0.0067 | 0.1100 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b21_pga_norm_amp_pga08/eval_results_best.txt` |
| b21 | last | 150 | 0.3251 | 0.4327 | 0.6714 | 0.4321 | 0.5415 | -0.0127 | 0.0830 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b21_pga_norm_amp_pga08/eval_results_last.txt` |
| b22 | best | 132 | 0.3320 | 0.4317 | 0.6650 | 0.4345 | 0.5008 | -0.0030 | 0.1058 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b22_pga_norm_mse_noamp/eval_results_best.txt` |
| b22 | last | 150 | 0.3395 | 0.4434 | 0.6764 | 0.4034 | 0.6128 | -0.0207 | 0.1042 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b22_pga_norm_mse_noamp/eval_results_last.txt` |
| b23 | best | 110 | 0.3498 | 0.4571 | 0.6344 | 0.3660 | 0.5226 | -0.0130 | 0.1358 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b23_pga_norm_huber05_noamp/eval_results_best.txt` |
| b23 | last | 150 | 0.3545 | 0.4638 | 0.6604 | 0.3473 | 0.5875 | -0.1094 | 0.1720 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b23_pga_norm_huber05_noamp/eval_results_last.txt` |
| b24 | best | 118 | 0.3391 | 0.4485 | 0.6533 | 0.3898 | 0.5426 | -0.0433 | 0.1062 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b24_pga_norm_huber2_noamp/eval_results_best.txt` |
| b24 | last | 150 | 0.3328 | 0.4390 | 0.6643 | 0.4152 | 0.5450 | 0.0230 | 0.0804 | `chaosuan_res/weights_japan_overfit_pga15_stage2_512_b24_pga_norm_huber2_noamp/eval_results_last.txt` |
