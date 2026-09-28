# CODEX-RESULT: RT61 wave-geometry residual conditioning validation

请 ChatGPT Project 审阅 RT61 固定 epoch-8 validation 结果，并给出下一步工作建议。
正式分析使用 random/normal validation、同一 forward 导出的 RT59 reference 与
adapter-off counterfactual；没有打开 held-out test，也没有选择中间 epoch。

```text
[CODEX-RESULT]
task_id: 20260925-rt61-wave-geometry-residual-conditioning
branch: rt61-wave-geometry-residual-conditioning
base_commit: 00d624d2fb01c6dd98cac150043dd15b7d8366ff
implementation_commit: dfacd49803cc120ad224845d5a675a055d7e0f4f
analysis_commit: 7603b9e1881d846734ffc640a9843dc6d0321fad
result_commit: e12a94043544e3d6a7e78c77a5f5b53cd0e3d662
submitted_source_manifest_sha256: cba3c2bdcded908ef42280efe97f78d3525c1c50b4b03206a322f504187012f7
source_archives:
  zip:
    sha256: e6d27bb11a05a2f634004f3d2068503086b5210a6fdc7777b30ef710ca5686d1
    size_bytes: 87176439
  tar_gz:
    sha256: 3c90d8addd339dafd2f87f5c0e7d359a07f0d3bff75a50ef126efd8d732e05ab
    size_bytes: 58470
protocol:
  split: validation only; held-out test unopened
  training_data: inherited Japan 2000-2024 train split, all 25 annual shards
  seed: 42
  training: exactly 8 new epochs; fixed final epoch 8
  bootstrap: 5000 paired event-id cluster draws, seed 20260915
  random: 9681 rows, 1310 events, 75654 targets
  normal: 9681 rows, 1383 events, 89770 targets
  normal_input_targets: 63651
  normal_noninput_targets: 26119
identity:
  rt59_parent_checkpoint_sha256: 5dbc15c6af5b8d341dfd4a377a68217b1e1aa3589550690997c02e81332aa1cd
  immutable_readout_sha256: db11b3c6152ea8ea76d78441769da04e6f03bb313050d70c64e4d12e1259f021
  shared_state_fingerprint: 9af5406450d7810dc2e29a6febfbaf13ac9072c6f25cfd6ea5eb73b8ad4e864b
  frozen_shared_state_verified_unchanged: true
  trainable: 9 tensors / 99621 scalars
decision:
  rt61_mechanism_pass: false (13/14)
  useful_joint_progress: false (0/4)
  legacy_full_go: false (18/25 required)
  recommendation: retain_rt59_reference
key_results_vs_same_forward_rt59_reference:
  random:
    mae: 0.2378282721 -> 0.2373237250; paired delta -0.0005045472
    rmse: 0.3112498024 -> 0.3101126477; paired delta -0.0011371548
    nll_delta: -0.0028499807
    brier_delta: -0.0005571331
  actual_one_station_spatial:
    pairwise_delta_mae: 0.3362015968 -> 0.3349077853
    relative_improvement: 0.384832 percent; preregistered requirement 2 percent
    paired_delta: -0.0012938115; 95 percent CI [-0.0021494314, -0.0007795663]
    range_abs_error: 0.6807751238 -> 0.6712517516
    range_abs_error_paired_delta: -0.0095233722; CI upper -0.0080236786
    range_ratio: 0.1786186181 -> 0.1905135095; legacy threshold 0.25
  normal_noninput:
    mae: 0.1993005403 -> 0.1992424090
    mae_paired_delta: -0.0000581313; 95 percent CI upper 0.0003898144
    rmse: 0.2556522440 -> 0.2557236810
    rmse_paired_delta: +0.0000714370; 95 percent CI upper 0.0007800131
    nll_delta: -0.0002903736
    brier_delta: -0.0003135477
  normal_input_increment_max_abs: 0.0
mechanism_failure:
  - normal_noninput_rmse_delta was positive (+0.0000714370); all other 13 mechanism gates passed.
useful_joint_progress_failures:
  - mechanism pass was false.
  - actual-one-station pairwise-delta improvement was 0.385 percent, below 2 percent.
  - normal non-input MAE paired-CI upper bound was positive.
  - normal non-input RMSE paired-CI upper bound was positive.
adapter_off_signal:
  - Applying the trained readout to h0 without the adapter gives normal non-input
    MAE 0.1985618641 and RMSE 0.2551717999, both better than full RT61.
  - Full RT61 gives MAE 0.1992424090 and RMSE 0.2557236810.
  - This is a same-checkpoint counterfactual, not an independently trained ablation,
    but it suggests the adapter offsets the readout-only normal benefit.
legacy_required_failures:
  - random slope
  - random all-field range ratio
  - random actual-one-station range ratio
  - random actual-one-station pairwise-delta MAE threshold
  - random coverage-1 absolute change
  - actual-one-station range absolute error versus historical RT57
  - normal coverage-1 absolute change
training:
  final_epoch: 8
  final_validation_loss: 0.6470828652
  minimum_validation_loss_diagnostic_epoch: 8
  readout_and_adapter_gradients_nonzero: true
  schedule_verified: true
verification:
  expected counts and config contract: PASS
  candidate/reference reconstruction and MDN identity checks: PASS
  normal observed-input invariance: PASS, max absolute increment 0
  field MSE decomposition max residual: 4.03e-16
  artifact manifest verification: PASS
  report path-leak scan: PASS
  scientific_validity: true
missing_artifact:
  - The supplied zip/tar.gz contain validation NPZ, resolved configs and training scalar logs,
    but no .pth checkpoint and no full_model_last.rt61_delta.pth.
  - Therefore the report verifies the checkpoint identity embedded in metrics but cannot archive
    the required lightweight reconstructable delta package in this commit.
review_request:
  - Audit the 13/14, 0/4 and 18/25 gate decisions and event-cluster CIs.
  - Assess whether the adapter-off counterfactual supports abandoning this adapter formulation,
    or motivates a single more targeted experiment with stronger normal protection.
  - Recommend the next scientific step without reopening held-out test, repeating old smoke tests,
    sweeping ranks/weights/seeds, or selecting an intermediate epoch.
  - State whether the missing epoch-8 trainable-delta should be fetched before closing RT61.
[/CODEX-RESULT]
```

## 审阅入口

1. `reports/rt61_wave_geometry_validation_20260928/RESULT_REVIEW.md`
2. `reports/rt61_wave_geometry_validation_20260928/summary.json`
3. `reports/rt61_wave_geometry_validation_20260928/mechanism_gates.csv`
4. `reports/rt61_wave_geometry_validation_20260928/useful_joint_progress_gates.csv`
5. `reports/rt61_wave_geometry_validation_20260928/legacy_gates.csv`
6. `reports/rt61_wave_geometry_validation_20260928/paired_ci.csv`
7. `reports/rt61_wave_geometry_validation_20260928/field_error_decomposition.csv`
8. `reports/rt61_wave_geometry_validation_20260928/reference_invariance.json`
9. `reports/rt61_wave_geometry_validation_20260928/training_summary.csv`
10. `reports/rt61_wave_geometry_validation_20260928/fixed_examples.png`
11. `reports/rt61_wave_geometry_validation_20260928/truth_prediction_density.png`
12. `reports/rt61_wave_geometry_validation_20260928/artifact_manifest.sha256`
