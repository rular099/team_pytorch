# CODEX-RESULT: RT60 final-contrast readout formal validation

请 ChatGPT Project 审阅分支 `rt60-final-contrast-readout` 的结果证据提交
`81be754bee32e41a28c79dc5611ff26f55efebd4`。本轮只分析用户提供的 RT60 epoch-8
random/normal formal validation 和训练日志，没有运行 held-out test，也没有提交新超算任务。

```text
[CODEX-RESULT]
task_id: 20260920-rt60-final-contrast-readout
base_commit: 080a53a541bd026bd054d8d8f603bb05d912def8
result_commit: 81be754bee32e41a28c79dc5611ff26f55efebd4
branch: rt60-final-contrast-readout
changed_files:
  - tools/analyze_rt60_contrast_readout_npz.py
    - Add fail-closed three-model RT57-historical / RT59-reference / RT60-candidate
      identity, MDN de-normalization, point/probability/field metrics, exact formal
      counts, two-layer gate decisions, 5000-draw event-cluster bootstrap, training
      log/config/archive provenance, sanitized configs and compact review exports.
  - reports/rt60_contrast_readout_validation_20260923/*
    - Archive the full machine-readable result, 5762 eligible random-field records,
      training/strata/group/CI/gate tables, density figure, human review and SHA manifest.
  - docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md
    - Provide this ChatGPT review handoff.
verification:
  - Source ZIP SHA-256: 26c75ca9dc730c20919855d9bff334495efe51b0acf9d474774d47662ca8c84a.
  - Source TAR.GZ SHA-256: 948b5f875166c5575b7107bd231ea7b61c1c1fda09547169eec4da1fbaba87ee.
  - Archive path traversal/absolute-path audit: PASS; neither archive contains a .pth checkpoint.
  - Formal identity: val only, epoch 8, task/parent/source hashes, normalization and
    random/normal checkpoint metadata all agree: PASS.
  - Counts: random 9681 rows / 1310 events / 75654 targets; normal 9681 rows /
    1383 events / 89770 targets (63651 input, 26119 non-input): PASS.
  - Candidate = RT59 reference + RT60 increment within 3.58e-7; input increment exactly
    zero; logits unchanged; component sigma change below 4.0e-8: PASS.
  - python tools/analyze_rt60_contrast_readout_npz.py on both formal NPZs:
    PASS; rt60_mechanism_pass=false (8/14), legacy_full_go=false (18/25).
  - event-cluster bootstrap: 5000 draws, seed 20260915; expected field counts
    5762 all / 1494 actual-one-station: PASS.
  - artifact_manifest.sha256: all 14 listed report artifacts PASS.
  - python -m unittest tests.test_rt60_contrast_readout: PASS, 7 tests.
  - python -m py_compile tools/analyze_rt60_contrast_readout_npz.py: PASS.
  - git diff --check: PASS.
compatibility:
  - This result turn changes no model, training, evaluation-forward, config or launcher
    behavior. RT55-RT60 checkpoint loading/inference paths are untouched.
  - The analyzer preserves val_rt59_base_* as historical RT57 gamma=1.66 and uses the
    separate val_rt60_reference_* export for the same-forward RT59 comparison.
hpc_status:
  - COMPLETED from user-provided archives. No new HPC job was submitted by Codex.
  - Training used all 25 Japan 2000-2024 annual training shards with the inherited
    event split; formal evaluation used the fixed validation split only. No test/smoke/
    roll/sweep/extra epoch was run.
results:
  - Intended spatial mechanism moved in the correct direction: actual-one-station
    pairwise-delta MAE 0.336202 -> 0.335647, paired 95% CI for delta
    [-0.001122, -0.000235]; one-station P95-P05 range absolute error
    0.680775 -> 0.676849, CI [-0.005019, -0.003049].
  - Random MAE was essentially unchanged (0.237828 -> 0.237824); random RMSE/NLL/Brier
    improved slightly.
  - Normal non-input MAE worsened 0.199301 -> 0.199880 and RMSE 0.255652 -> 0.256526;
    both paired CIs are strictly positive. Normal all/non-input NLL and Brier also worsen.
  - Normal input predictions are targetwise unchanged, so normal degradation is remote-only.
  - The conjunctive RT60 mechanism decision therefore fails. Legacy also remains NO-GO:
    18/25 required gates pass; slope, field contrast, pairwise threshold and two coverage
    retention conditions remain unresolved.
  - Pre-registered action: retain RT59 epoch-8 as the development parent, archive RT60
    as a partial-mechanism/negative result, and do not auto-extend training.
remaining_risks:
  - This is repeatedly used development validation, one seed, not held-out test evidence.
  - Spatial improvements are statistically clear but very small; practical significance
    and transfer to new events are unestablished.
  - The supplied archives omit the RT60 checkpoint, so the report certifies exports and
    checkpoint metadata but cannot independently reload the trained weights.
review_request:
  - Read RESULT_REVIEW.md and summary.json first, then audit the three-model identity,
    raw-coordinate MDN conversion, P95-P05 field definition, equal-field aggregation,
    actual-one-station mask and event-cluster CI implementation.
  - Confirm the 14 mechanism gates and 25 required + 4 diagnostic legacy gates against
    the pre-registered RT60 prompt; do not replace RT59 reference with historical RT57.
  - Assess the interpretation: final-readout-only fitting weakly recovers spatial contrast
    but is insufficient to preserve normal remote performance.
  - Recommend one next, minimal, falsifiable research step. Do not request repeated old
    smoke/diagnostics, automatic extra epochs, a parameter/seed sweep or held-out test.
[/CODEX-RESULT]
```

## 审阅入口

1. `reports/rt60_contrast_readout_validation_20260923/RESULT_REVIEW.md`
2. `reports/rt60_contrast_readout_validation_20260923/summary.json`
3. `reports/rt60_contrast_readout_validation_20260923/mechanism_gates.csv`
4. `reports/rt60_contrast_readout_validation_20260923/legacy_gates.csv`
5. `reports/rt60_contrast_readout_validation_20260923/paired_ci.csv`
6. `reports/rt60_contrast_readout_validation_20260923/field_metrics.csv`
7. `reports/rt60_contrast_readout_validation_20260923/training_summary.csv`
8. `tools/analyze_rt60_contrast_readout_npz.py`
9. `../RT59_REVIEW_RT60_CODEX_PROMPT_20260920.md` 第 4.4 节
