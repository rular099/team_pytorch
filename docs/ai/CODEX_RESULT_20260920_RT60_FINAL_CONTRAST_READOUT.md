# CODEX-RESULT: RT60 final-contrast readout

请 ChatGPT Project 审阅分支 `rt60-final-contrast-readout`。RT59 原结果已按正式
P95−P05 field-range 与 25 required / 4 diagnostic 的口径更正；RT60 单一 readout-only
实验已实现、测试并给出默认 dry-run 的超算 launcher，但尚未提交任何超算作业。

```text
[CODEX-RESULT]
task_id: 20260920-rt60-final-contrast-readout
base_commit: d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd
implementation_commit: 033f01a481fdb7499aff836abe6aaae35b72e5ca
result_commit: b5cfadd9d621744815ba45461727a9a8d6b9b6bb
branch: rt60-final-contrast-readout
uploaded_source_manifest_sha256: 405d533f7613adb4a6e52ac23096ae41a1c50d173417265b500c943bb7a27a38
changed_files:
  - tools/analyze_rt59_dual_objective_npz.py
    - Restore per-field NumPy-default P95-P05 range, >=5 valid-target eligibility
      and equal field weighting; retain PTP only under diagnostic keys.
    - Separate 25 required gates from four diagnostic gates; fail closed on event,
      unit, mask/count, route/formal-target, base and checkpoint identity conflicts.
    - Add correct-vs-fixed-context-roll metrics, full field CSV, sanitized configs,
      event manifests and corrected implementation/source provenance.
  - reports/rt59_v3_review_correction_20260920/
    - Archive corrected machine-readable RT59 evidence without rerunning forward.
  - reports/rt59_dual_objective_transport_v3_validation_20260920/{README.md,RESULT_REVIEW.md}
    - Preserve the old report for audit while linking prominently to the correction.
  - gemini_models.py
    - Add default-off RT60 same-forward immutable RT59 teacher readout and student
      increment; observed inputs retain zero increment; legacy state dict unchanged.
  - tools/rt60_contrast_objective.py
    - Add exact readout/reference hashing, parent non-readout fingerprints, grouped
      point/regret objective and O(Q) final-field contrast with global DDP denominators.
  - train_light.py
    - Add readout-only freeze/optimizer/clip/schedule, exact six-tensor/67077-scalar
      assertions, reference checkpoint/resume metadata, and epoch loss/count/grad logs.
  - eval_checkpoint.py
    - Conditionally restore/export RT60 reference MDN/mean and student increment.
  - pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_chaosuan.json
  - pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_normal_validation_chaosuan.json
    - Fixed seed42, eight-new-epoch protocol; old RT59 auxiliaries disabled.
  - tools/run_rt60_contrast_readout_slurm.sh
    - Default-dry-run source/parent/data verification and one train -> two afterok val jobs.
  - tests/test_rt60_contrast_readout.py
    - Algebra, masks/NaN, DDP shards, teacher/student, compatibility, mutation,
      checkpoint/resume, export and complete tiny train/validation coverage.
  - docs/rt60_contrast_readout.md
    - Protocol, compatibility contract, objective and exact manual-HPC commands.
verification:
  - Phase-A inputs: existing formal RT59 epoch-8 random/normal val NPZ and metrics only;
    no new forward and no held-out test.
  - corrected result: NO_GO; 18/25 required pass, 7 fail; 4/4 diagnostics pass.
  - canonical random P95-P05 range ratio: base 0.510231 -> final 0.475336.
  - canonical actual-one-station range ratio: base 0.207285 -> final 0.178619.
  - actual-one-station range-absolute-error delta: +0.022213.
  - old point/probability metrics match the archived report at tolerance 1e-12.
  - formal input versus observable route: 63651/63651 elementwise; ambiguous targets 0.
  - corrected report uses 5000 event-ID bootstrap draws, seed 20260915;
    artifact manifest, path-redaction and clean-worktree provenance checks pass.
  - python -m unittest tests.test_rt60_contrast_readout
    tests.test_rt59_dual_objective_transport tests.test_rt58_waveform_anchor_transfer
    tests.test_scheduler_checkpoint_resume: PASS, 31 tests.
  - python -m unittest tests.test_causal_random_geometry
    tests.test_rt57_gtnp_v2_station_distinctive tests.test_eval_checkpoint_formal
    tests.test_random_geometry_full_npz_analysis: PASS, 26 tests.
  - python -m py_compile on changed Python modules: PASS.
  - bash -n on RT60/train/eval launchers: PASS.
  - git diff --check: PASS.
  - uploaded_sha256 ACTION=all DRY_RUN=1: PASS; no job submitted.
compatibility:
  - RT60 model flag defaults false and adds no registered teacher module/buffer, so old
    RT55-RT59 parameter keys/shapes and inference route remain unchanged.
  - Legacy RT55/RT56/RT57/RT58/RT59 focused regression suites pass.
  - Existing RT59 base-output semantics remain RT57 gamma=1.66; RT60 reference uses new,
    conditional export keys and cannot silently replace the historical base.
hpc_status:
  - NOT_SUBMITTED. User must upload this branch snapshot and run the documented dry-run,
    then one confirmed ACTION=all launch using the exact RT59 retry1 epoch-8 checkpoint.
  - Required job graph: one 4-node/16-DCU eight-epoch train, followed by random and
    normal epoch-8 validation through afterok; no test/smoke/roll/sweep jobs.
remaining_risks:
  - Only the final readout is trainable; frozen features may be insufficient to recover
    spatial contrast.
  - Squared field contrast can chase noise, so mechanism gates jointly protect point
    error, observed inputs and probability scores.
  - One seed and repeatedly used development validation do not establish generalization.
  - PyTorch-1.13/DCU multi-node execution and the real parent checkpoint are HPC-only and
    remain unverified until the formal run.
review_request:
  - First audit the Phase-A P95-P05 implementation, 25/4 gate split, identity checks,
    rolled-control interpretation and unchanged legacy point/probability metrics.
  - Audit RT60 exact trainability, immutable teacher across resume, same-forward reuse,
    observed-input invariance, loss signs/weights and DDP target-versus-field reductions.
  - Audit the launcher parent checkpoint/task/epoch/source checks and RT55-RT59
    default-off compatibility.
  - If implementation review passes, recommend only the documented single formal run;
    do not request repeated smoke, old diagnostics, test evaluation, sweeps or extra epochs.
  - When RT60 results arrive, judge rt60_mechanism_pass against RT59 separately from
    legacy_full_go; mechanism pass must not be described as test/generalization success.
[/CODEX-RESULT]
```

## 审阅入口

按顺序读取：

1. `reports/rt59_v3_review_correction_20260920/RESULT_REVIEW.md`
2. `reports/rt59_v3_review_correction_20260920/summary.json`
3. `reports/rt59_v3_review_correction_20260920/gates.csv`
4. `docs/rt60_contrast_readout.md`
5. `tools/rt60_contrast_objective.py`
6. `gemini_models.py`、`train_light.py`、`eval_checkpoint.py`
7. 两个 RT60 configs 与 `tools/run_rt60_contrast_readout_slurm.sh`
8. `tests/test_rt60_contrast_readout.py`
