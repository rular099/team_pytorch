# CODEX-RESULT: RT59-v3 epoch-8 validation analysis

请 ChatGPT Project 在 GitHub 上审阅 evidence commit
`b709825b78e7579dbf05c8d4eacda5b29b9e58ae`。本轮严格结论为
`NO_GO`（29 个 conjunctive gates 中 22 PASS / 7 FAIL），但同一 checkpoint 已同时显著
改善 random 与 normal pointwise error。请区分这一实质进展与尚未解决的 spatial-range
问题，并给出下一轮唯一、可执行、快速推进的工作建议。

```text
[CODEX-RESULT]
task_id: 20260920-rt59-dual-objective-transport-v3-validation
base_commit: 7adb7b41a3f14222d39d733f68f1d3522ee5e88d
result_commit: b709825b78e7579dbf05c8d4eacda5b29b9e58ae
branch: rt59-dual-objective-transport-v3
submitted_source_manifest_sha256: 3e1164bc4fb0441fb33e5d8cd03aa1708416708d930b01e71c4a82fd3779715e
changed_files:
  - reports/rt59_dual_objective_transport_v3_validation_20260920/RESULT_REVIEW.md
    - protocol/provenance, training summary, random/normal paired results,
      spatial-field audit, forward controls, predeclared go/no and review request
  - reports/rt59_dual_objective_transport_v3_validation_20260920/summary.json
    - complete machine-readable metrics, strata, bootstrap intervals and gates
  - reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv
    - exact 29-gate conjunctive decision table
  - reports/rt59_dual_objective_transport_v3_validation_20260920/training_summary.csv
    - all eight epochs of loss, branch gradient norms and group target counts
  - reports/rt59_dual_objective_transport_v3_validation_20260920/forward_control.json
    - observable route, mean-shift identity and fixed-context feature-roll evidence
  - paired_ci.csv, group_metrics.csv, strata_counts.csv,
    truth_prediction_density.png, artifact_manifest.sha256, analyzer README
verification:
  - user ZIP SHA-256: 006ca5c7c153243b07f7419f550a89a54e7475498674d8b1fc152d41770064ca
  - user training-log tar.gz SHA-256: 9dbef551ebc37dded392523f139741866c3ace1df71ef55d762c298af58a323b
  - epoch-8 checkpoint metadata agrees across random/normal: epoch 8,
    loss 0.7177332043647766, checkpoint_format non_encoder_v1
  - random counts: 9681 rows, 1310 events, 75654 targets
  - normal counts: 9681 rows, 1383 events, 89770 targets;
    63651 formal input and 26119 formal non-input targets
  - normal observable route: 63651 unique observed, 0 ambiguous; exact formal-input count match
  - recomputed point metrics agree with formal metrics JSON
  - exported frozen base agrees with base MDN mixture mean (analyzer hard check)
  - final = frozen base + RT59 applied delta: max error random 3.17e-7 dex,
    normal 3.43e-7 dex
  - fixed-context selected-u/d roll changes every multi-station target correction;
    actual-one-station roll maximum change is exactly zero
  - paired event_id percentile bootstrap: 5000 draws, seed 20260915
  - all required quantities present; gates parse as 29 rows, 22 pass, 7 fail
  - summary.json and forward_control.json parse: pass
  - git diff --check: pass
compatibility:
  - result-only change; no model, loader, config, launcher or RT55--RT59 behavior changed
hpc_status:
  - completed by user: full 2000--2024 (25 resolved yearly shards), seed42,
    fixed eight-epoch RT59-v3 train, epoch-8 random validation and normal validation
  - held-out test not used
headline_results:
  - random MAE 0.245578 -> 0.237828; paired delta -0.007749,
    95% CI [-0.008785, -0.006701]
  - random R2 0.367504 -> 0.394995; NLL and Brier both improve
  - requested-1-s MAE 0.255091 -> 0.243391; this gate PASSES
  - normal all/input/non-input MAE and RMSE all improve;
    non-input MAE 0.210371 -> 0.199301
  - seven failures: random slope, all-field range ratio, one-station range ratio,
    one-station pairwise-delta MAE, random coverage-1 retention,
    one-station range absolute error delta, normal coverage-1 retention
remaining_risks:
  - result archives exclude full_model_last.pth; checkpoint-body SHA-256 cannot be recomputed locally
  - single seed and validation only; no held-out-test or generalization claim
  - one joint intervention cannot identify independent contributions of routing, grouped loss,
    MSE, regret guard and learning-rate schedule
review_request:
  - Audit base/final semantics, formal input versus observable route, bootstrap direction,
    spatial-field definitions and all 29 predeclared gates; confirm NO_GO 22/29.
  - Do not misclassify requested 1 s as a failure, and do not turn the two coverage-1
    retention failures into a false claim that NLL/Brier degraded.
  - Decide whether RT59 epoch8 should be retained as the current strongest pointwise
    development checkpoint without declaring spatial-field success.
  - Identify one evidence-supported primary bottleneck and propose at most one next
    high-value experiment aimed directly at spatial range / one-station field difference.
  - Do not request repeated smoke, old query diagnostics, formal test, sweeps, or checks
    already answerable from the submitted NPZ/report.
  - If code work is recommended, provide a complete AI-HANDOFF preserving RT55--RT59
    checkpoint loading and inference compatibility.
[/CODEX-RESULT]
```

## 审阅入口

按顺序读取：

1. `reports/rt59_dual_objective_transport_v3_validation_20260920/RESULT_REVIEW.md`
2. `reports/rt59_dual_objective_transport_v3_validation_20260920/README.md`
3. `reports/rt59_dual_objective_transport_v3_validation_20260920/summary.json`
4. `reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv`
5. `reports/rt59_dual_objective_transport_v3_validation_20260920/training_summary.csv`
6. `reports/rt59_dual_objective_transport_v3_validation_20260920/forward_control.json`
7. `docs/rt59_dual_objective_transport_v3.md`
8. `docs/ai/CODEX_RESULT_20260915_RT59_DUAL_OBJECTIVE_TRANSPORT_V3.md`

最重要的审查点是：RT59 已经显著改善 random/normal 的总体和分组点误差，并且概率
NLL/Brier 同向改善；然而 all-field range ratio 从 `0.504499` 降至 `0.467950`，
actual-one-station range ratio 从 `0.203564` 降至 `0.174563`，单站 range absolute
error 增加 `0.027007 dex`。下一步需要直接处理 transport 的空间对比，而不是重复证明
head 会使用正确配对的 waveform/station feature。
