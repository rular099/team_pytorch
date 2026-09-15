# CODEX-RESULT: RT58 WATF epoch-8 validation analysis

请 ChatGPT Project 在 GitHub 上审阅 evidence commit
`37d97ad9ad67dfec11f188e2a350a8ab8b8b2056`，并给出下一轮唯一、可执行且快速推进的
工作建议。

```text
[CODEX-RESULT]
task_id: 20260915-rt58-waveform-anchor-transfer-validation
base_commit: 0e8537513e623432339d84ae2525b588340b18c9
result_commit: 37d97ad9ad67dfec11f188e2a350a8ab8b8b2056
branch: rt58-waveform-anchor-transfer-field
changed_files:
  - reports/rt58_waveform_anchor_transfer_validation_20260915/README.md
    - full protocol, paired analysis, anchor diagnostics, fixed-base roll attribution,
      predeclared go/no audit and review questions
  - reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json
    - compact machine-readable metrics and artifact SHA-256 identities
  - reports/rt58_waveform_anchor_transfer_validation_20260915/gates.csv
    - exact ten-criterion go/no table
verification:
  - RT57/RT58 random NPZ alignment: pass for event/target/mask/geometry/time/station-count arrays
  - recomputed point metrics versus formal JSON: pass at rtol=atol=1e-12
  - event-cluster percentile bootstrap: 5000 replicates, seed 20260915
  - RT58 final reconstruction from gamma=1.66 base plus anchor delta: max error 3.50e-7 dex
  - summary.json parse: pass
  - gates.csv parse and row count: pass, 10 rows
  - git diff --check: pass
compatibility:
  - result-only change; no model, loader, config, launcher or RT55/RT56/RT57 behavior changed
hpc_status:
  - completed by user: fixed RT58 epoch-8 random validation, normal validation and random waveform roll
  - held-out test not used
remaining_risks:
  - uploaded result bundle excludes checkpoint body and training scalar logs; checkpoint SHA and epoch trajectory cannot be re-audited locally
  - single seed and validation only
  - criterion 8 uses the unquantified phrase "calibration not materially degraded" and is marked provisional pass
review_request:
  - Audit gamma=1 versus gamma=1.66 contribution separation, metric directions,
    event-cluster bootstrap, one-station interpretation and fixed-base roll attribution.
  - Apply the original ten go/no criteria without changing thresholds after seeing results.
  - Select at most one next high-value experiment; do not request repeated smoke,
    query-geometry diagnostics, formal test or checks already answerable from these NPZs.
  - If code work is recommended, provide a complete AI-HANDOFF preserving RT55/RT56/RT57 loading and inference.
[/CODEX-RESULT]
```

## 审阅入口

按顺序读取：

1. `reports/rt58_waveform_anchor_transfer_validation_20260915/README.md`
2. `reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json`
3. `reports/rt58_waveform_anchor_transfer_validation_20260915/gates.csv`
4. `docs/rt58_waveform_anchor_transfer.md`
5. `docs/ai/CODEX_RESULT_20260912_RT58_WAVEFORM_ANCHOR_TRANSFER.md`

最重要的审查点是：RT58 相对 RT57 `gamma=1` 的小幅总体改善，是否主要只是已知的
`gamma=1.66` scale 收益。现有严格配对结果显示，新 anchor correction 相对相同
`gamma=1.66` base 的 random MAE 变差 `+0.000458 dex`，而准确 input anchor 没有
转化为更好的空间幅度；actual-one-station mean range ratio 从 `0.207285` 降至
`0.183340`。请据此判断下一步应如何重新设计 transfer/fusion，而不是默认继续训练
当前 RT58 或先解冻 DiTing。
