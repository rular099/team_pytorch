# CODEX-RESULT: RT61 wave-geometry residual conditioning

请 ChatGPT Project 审阅分支 `rt61-wave-geometry-residual-conditioning`。本轮只实现任务书规定的
单一 RT61 实验、回归测试、正式结果分析器和默认 dry-run 的超算 launcher；没有提交超算作业，
没有读取 held-out test，也没有生成正式训练结果。

```text
[CODEX-RESULT]
task_id: 20260925-rt61-wave-geometry-residual-conditioning
base_commit: 00d624d2fb01c6dd98cac150043dd15b7d8366ff
implementation_commit: dfacd49803cc120ad224845d5a675a055d7e0f4f
result_commit: dfacd49803cc120ad224845d5a675a055d7e0f4f
branch: rt61-wave-geometry-residual-conditioning
uploaded_source_manifest_sha256: cba3c2bdcded908ef42280efe97f78d3525c1c50b4b03206a322f504187012f7
parent_checkpoint:
  task: RT59-v3 epoch 8
  sha256: 5dbc15c6af5b8d341dfd4a377a68217b1e1aa3589550690997c02e81332aa1cd
  immutable_readout_sha256: db11b3c6152ea8ea76d78441769da04e6f03bb313050d70c64e4d12e1259f021
implementation:
  - Added a default-off rank-16 waveform/event by query-geometry residual adapter.
  - The adapter output projection is zero initialized, so the initial student equals RT59.
  - The immutable reference and station weighting consume h0; only the student readout consumes h1.
  - Exactly the existing six residual-readout tensors plus three adapter tensors are trainable.
  - Production dimensions give 9 tensors and 99,621 trainable scalars.
  - Sampling, splits, cutoff, normalization, MDN shape, RT60 loss weights and observed-input
    routing are unchanged.
  - Checkpoints contain immutable-reference identity, exact trainable manifest and frozen-shared
    fingerprint verification; every checkpoint also emits a lightweight RT61 trainable delta.
  - Evaluation exports candidate, immutable reference and trained-readout/adapter-off counterfactual
    from the same forward pass.
  - The analyzer reports the original 14 mechanism gates, 25 required legacy gates, four
    diagnostics, useful_joint_progress, field error decomposition, event-cluster CIs, strata,
    training curves and fixed-rule common-coordinate success/failure maps.
protocol:
  - seed42; all existing 2000-2024 training shards; exactly eight new epochs.
  - fixed learning-rate schedule: 1e-4 x 4, 5e-5 x 2, 2.5e-5 x 2; gradient clip 1.
  - fixed final epoch 8 only; random and normal validation only.
  - no test, smoke, roll, sweep, extra seed or automatic continuation.
verification:
  - python -m unittest discover -s tests -p 'test_*.py' -v: PASS, 108 tests.
  - RT61 focused tests: PASS, 6 tests.
  - RT57-RT60 focused regression suites: PASS, 38 tests.
  - Initial numerical identity, reference-output isolation, trainable whitelist, two-step gradient
    reachability, station/query permutation, query chunk/repeat/append, K=1, invalid-query/station
    NaN masks, checkpoint/resume and delta reconstruction are covered.
  - Python compilation, launcher bash syntax and git diff whitespace checks: PASS.
  - uploaded_sha256 ACTION=all DRY_RUN=1: PASS; no Slurm job submitted.
hpc_status:
  - NOT_SUBMITTED.
  - The real RT59 checkpoint and DCU/PyTorch-1.13 distributed run remain HPC-only checks.
  - Upload this branch snapshot, verify the printed source manifest, then launch exactly one
    train -> random/normal afterok validation graph with tools/run_rt61_wave_geometry_slurm.sh.
compatibility:
  - The RT61 model switch defaults false and adds no parameter or buffer to legacy models.
  - RT60 and RT61 switches are mutually exclusive; RT55-RT60 state loading and inference stay on
    their original routes.
  - Existing RT60 objective behavior is unchanged unless its new optional reference attribute is
    explicitly supplied by RT61.
remaining_risks:
  - One seed and repeatedly used development validation are not independent generalization evidence.
  - The adapter may still be unable to overcome the limited information in early single-station input.
  - Normal protection is a soft objective term; useful_joint_progress therefore requires both normal
    non-input MAE and RMSE paired-CI upper bounds below zero.
review_request:
  - Audit h0/h1 separation, immutable reference output, station-weight isolation and observed route.
  - Audit exact trainability, zero-init identity, global objective reductions, schedule, checkpoint
    metadata and lightweight-delta reconstruction.
  - Audit analyzer gate denominators, event-cluster bootstrap, field decomposition, fixed example
    selection and source/checkpoint/config fail-closed checks.
  - If implementation review passes, recommend only the single documented formal run; do not request
    repeated smoke tests, held-out test evaluation, parameter sweeps or extra epochs.
[/CODEX-RESULT]
```

## 审阅入口

1. `docs/rt61_wave_geometry_residual.md`
2. `tools/rt61_wave_geometry.py`
3. `gemini_models.py` 中 RT61 adapter、h0/h1 与 reference 路径
4. `train_light.py`、`eval_checkpoint.py`
5. 两个 RT61 配置与 `tools/run_rt61_wave_geometry_slurm.sh`
6. `tools/analyze_rt61_wave_geometry_npz.py`
7. `tests/test_rt61_wave_geometry.py`
8. `docs/ai/RT60_REVIEW_RT61_CODEX_PROMPT_20260925.md`
