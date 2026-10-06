# CODEX-RESULT：V01 validation closure 结果整理与远端审阅

```text
[CODEX-RESULT]
task_id: 20261006-v01-validation-closure-results
base_commit: 7d87c4007b78405bb901c67ff0b78dc5d0e85a8e
result_commit: 本文所在提交；精确40位SHA在最终回复和git log提供
branch: exp/v01-velocity-prep-padding-control
changed_files:
  - reports/v01_validation_closure_20261006/: 总结、复算脚本、验证JSON、轻量CSV/PNG/身份与来源证据
  - docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md: 独立审阅与单一最小下一步决策请求
  - docs/ai/README.md / PROJECT_CONTEXT.md / SESSION_SUMMARY.md / docs/v01_validation_closure.md: 当前入口与已完成状态
verification:
  - all10 NPZ and identity ledgers hashes: PASS; original files untouched
  - returned runtime180 source files / six prepared configs / frozen split hash: PASS
  - train/dev disjoint and every validation request in frozen dev: PASS
  - six new ledgers complete, no duplicates/substitution/implementation error: PASS
  - all250 metric strata, 4750 numeric fields recomputed from NPZ: PASS
  - primary FF/MM normal+random, all7+early135, five strata, 400 CI rows recomputed: PASS
  - event-cluster 5000 draws, seed20260915, actual errors, no bias correction
  - all14 one-to-one physical sensor/exact UTC outer joins and labels: PASS
  - density and stratified residual bin arrays independently recomputed: PASS; PNG inspected
  - original evidence copies hash-identical; raw NPZ/weights/waves/full ledgers excluded from Git
  - authored-file diff whitespace: PASS; original submitted_jobs.tsv retains one trailing tab for empty audit dependency
  - full unittest discover: PASS,159 tests,0 skipped
  - runtime/model/train/eval/config/Slurm code unchanged from base
compatibility:
  - result/documentation-only; no RT55 defaults, parameter shapes, load/forward or checkpoint changed
  - existing RT55–RT61 regression suite PASS
hpc_status:
  - user completed audit29318960, six eval29318961-29318966, analysis29318967 at application level
  - sacct NOT_AVAILABLE; scheduler COMPLETED/exit code/resource use not asserted
  - no new HPC task submitted by Codex; old epoch8 weights and successful random results reused
remaining_risks:
  - historical training runtime/encoder tensor identity and actual sample trajectory remain unverified
  - one seed/eight epochs/development validation; no held-out test claim
  - A/V common-available selection bias, distinct instruments/depth/sites/frequency response
  - paired-sensor remote is only proxy; true physical-site remote not certified
  - full-record centering/cutout+1 inherited; raw post-P/strict-online causality not fully certified
review_request:
  - decide whether the V01 controlled experiment can close as a limited negative mechanism result
  - weigh high-PGA degradation against aggregate bias improvement and near-zero M-model view effect
  - do not silently expand to retraining/preflight/matrices; specify only indispensable minimal follow-up
[/CODEX-RESULT]
```

## 一句话总结

工程闭环完成；normal和random都显示M训练臂总体更好，但高PGA目标更差、单输入空间场仍
压缩，A/V共同群体不能明确判MM/AA胜负。这不是速度性能上限、test或padding因果机制证明。

详细指标、样本量、CI、口径与证据定位：
`reports/v01_validation_closure_20261006/RESULT_REVIEW.md`。
ChatGPT 阅读入口：`docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md`。

## 关键数值（validation / epoch8 / all7 / log10(m/s²)）

| geometry | FF MAE | MM MAE | MM−FF 95% CI | 高PGA MM−FF MAE |
| --- | ---: | ---: | --- | ---: |
| normal | 0.215428 | 0.207152 | [-0.009000,-0.007552] | +0.005165 |
| random | 0.244516 | 0.236895 | [-0.008285,-0.006954] | +0.009195 |

两个场景各84,259targets/1,194events，非输入query-only；高PGA固定阈值≥-1.2，
normal35,917/random35,612targets，各1,051贡献events，高PGA MAE/NLL/Brier CI同向退化。
AA正常弃权率normal0.4068%/random0.5384%，缺失请求入分母，不再邻居替代。
MM−AA common-available MAE CI均跨零，不宣称等效或任何模态普遍优胜。

## 本轮产物边界

Git轻量包约15MiB，最大文件约5.8MB（主要FF/MM event sufficient stats），没有100MB级文件。
原118MB full sufficient table、6个新NPZ、完整requests/identity计划、旧NPZ、checkpoint/HDF5
均保留用户本地/超算，不入Git。没有修改用户未跟踪的技术交流目录或 `tmp.tar.gz`。
完整source copy清单为 `artifact_copy_manifest.json`，当前源码hash由回传manifest固定。
共82份原始小产物按字节复制；`submitted_jobs.tsv` 首行的空 dependency 列带末尾tab，
为保持原始hash保留，不为消除lint提示而改写证据。自编代码/文档无whitespace错误。

工程后续无需用户再次提交计算；若方便，可补回既有job IDs的sacct文本用于资源/状态归档，
不影响当前指标审阅，也不为此重跑任何任务。
