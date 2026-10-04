# CODEX-RESULT：FE01 九组超算结果整理与ChatGPT交接

用户提供 `chaosuan_res/fe01_runs_20261002`，要求整理并同步远端供ChatGPT分析。
结果包见 [README](../../reports/fe01_hpc_results_20261004/README.md)，
可直接复制的分析要求见 [CHATGPT_ANALYSIS_PROMPT](../../reports/fe01_hpc_results_20261004/CHATGPT_ANALYSIS_PROMPT.md)。

```text
[CODEX-RESULT]
task_id: 20261004-fe01-hpc-result-handoff
repo: rular099/team_pytorch
branch: exp/fe01-feature-extractor-site-effects
base_commit: 73ae9dec50713805c878b5e4efeb8264c01d0c76
result_commit: b5b2421f2667461e8ae785bcccc01e00411d905d
handoff_commit: the Git commit containing this record; verified full remote HEAD is given to the user after push
changed_files:
  - reports/fe01_hpc_results_20261004/: 28 files; Chinese report, ChatGPT prompt, run/seed/cell/event summaries, paired bootstrap, audit/runtime/config snapshots, inventory, provenance and verification
  - scripts/fe01/summarize_hpc_results.py: reproducible local reader of saved CSV/JSON/JSONL, without model forward/HDF5 access/scheduler submission
  - docs/ai/CODEX_RESULT_20261004_FE01_HPC_RESULTS.md: this review handoff
  - docs/ai/README.md: link to current FE01 result evidence
  - SESSION_SUMMARY.md: dated FE01 progress entry
verification:
  - local reader on provided upload: PASS; 9 formal runs, each 12 epochs, plus 3 separate pilots
  - all nine selected validation scores recomputed: PASS; tolerance 1e-12 vs curves
  - saved source config / protocol lock / effective resolved config: PASS
  - same audited cohort, exact query population, labels, inputs and causal decision boundaries: PASS across all 9 runs
  - common downstream structure: PASS across all runs; common initial values: PASS within each seed
  - exact mixture scoring from 32 MDN rows per run: PASS
  - 192 journals/run, 325632 rows/run, epoch/rank grid and curve sample counts: PASS
  - saved journal causal sample boundaries: PASS; checkpoint committed_journals membership NOT_VERIFIED
  - 5000 event-cluster bootstrap draws for 9 same-seed comparisons: PASS; no empty-cell replicates
  - exported 99567 event-cell rows independently reconstruct all 9 primary metrics: PASS
  - source inventory: PASS; 2201 files / 14193566080 bytes; SHA256 read for 2199 noncache files; 2 derived cache files explicitly unhashed
  - 115 source/dependency-file identity: PASS; local source and archive manifest agree with saved HPC code hash 3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea
  - reader syntax/import and generated JSON parse: PASS
  - git diff --cached --check: PASS for result commit
  - artifact_manifest.sha256: emitted for all analysis-bundle files except the manifest itself
compatibility:
  - RT55 model/inference/loading/training/config code was not edited; all source-scope hashes remain identical to HPC audit
  - model regression suite NOT_RUN: result-reader/report changes only; no model behavior changed
hpc_status:
  - user-provided runtime job IDs: EQT 29181344/29229582/29219507; PhaseNet 29181346/29219510/29219511; TEAM 29181345/29219508/29219509, each listed in seed42/43/44 order
  - curves and saved epoch metadata record all 12 epochs and 2544 updates for each formal run
  - scheduler COMPLETED / exit code NOT_PROVIDED; no formal Slurm or sacct completion evidence in upload
  - NOT_SUBMITTED_THIS_TASK; no sbatch/srun/train/test/remote calculation invoked
remaining_risks:
  - validation is used for checkpoint selection and current reporting; no independent test result
  - checkpoint binaries absent: best/last identity, epoch/SHA and committed journal membership cannot be checked
  - training runtime git_dirty is null; clean HPC Git state is not certified
  - native-window/model-conditioned training times differ; shared-time validation does not isolate pretraining or encoder-only causal effect
  - RT55/RT61 comparable reevaluation not supplied; existing RT55 is reused without retraining, and historical scalars are not inserted into the FE01 table
  - DiTing fresh-common-downstream seed group absent in current quick-reuse plan; original four-family controlled matrix remains incomplete
  - random validation times, 40/90s final metrics, site/spatial/replay/test evidence NOT_PROVIDED
  - offline filtering/resampling online causality remains uncertified
review_request:
  - review result_commit and reports/fe01_hpc_results_20261004/README.md first
  - assess accuracy jointly with coverage, interval width, NLL/CRPS/Brier, seeds and event-paired uncertainty
  - prioritize next evaluation work while reusing existing RT55/RT61; do not default to retraining or test-driven tuning
[/CODEX-RESULT]
```

## 已核验的当前数值

validation共同1/3/5/10/20秒、normal/random、noninput等10个cell权重的MAE，
坐标 `log10(m/s²)`；每run使用预定规则从12轮曲线选出epoch：

| 模型 | seed42/43/44 epoch | 3 seed平均MAE | 样本标准差 | 95%覆盖率均值 |
| --- | --- | --- | --- | --- |
| PhaseNet frozen | 12 / 10 / 12 | 0.225818 | 0.000125 | 91.29% |
| EQT frozen | 10 / 10 / 10 | 0.235296 | 0.001310 | 92.35% |
| TEAM scratch | 11 / 11 / 11 | 0.241817 | 0.003011 | 91.34% |

每run比较1310个validation事件、139440个非输入目标记录；全目标194265条。
固定选中模型的同seed事件bootstrap三种配对共9个主差值区间均低于零，
支持当前验证排序方向的稳定性。区间未包括checkpoint选择不确定性，
不能当作独立test或跨seed总体显著性证据。

PhaseNet点/整体概率评分较好，EQT覆盖率更接近名义值，三组均欠覆盖；
尚不支持“已超过RT55”“场地效应已改善”或“最终test胜出”的结论。

## 原始产物与本次同步范围

原始约14.19GB保持在用户上传的本地目录；Git结果包约23.86MB，
包括22.50MB逐事件cell加总表及小型表/JSON/说明。它不含逐目标原始预测、
大journal、HDF5、权重或Slurm原文。
本批旧Slurm仅有两个pilot-audit作业的4个日志，不能用于证明九个正式训练的scheduler完成状态。
Git推送后的完整HEAD通过远端branch SHA复核；本文件固定引用数值结果commit，
避免用“最新版”作为结果身份。
