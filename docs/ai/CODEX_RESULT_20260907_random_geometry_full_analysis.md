# CODEX-RESULT：RT55/RT56 random-geometry 全量离线配对分析

请 ChatGPT Project 读取本文件，并在 GitHub 上独立审查 `result_commit`。本轮没有再提交超算任务；分析完全复用了已经完成的两个正式 validation NPZ。

```text
[CODEX-RESULT]
task_id: 20260907-random-geometry-full-offline-analysis
base_commit: 597893e40e4df14c478b7812af302b3232167815
result_commit: ef98ffa342f52d01f749809a39665ea6c1ee3127
branch: query-geometry-diagnostics
changed_files:
  - tools/analyze_random_geometry_full_npz.py: 新增严格对齐检查、正式点指标反校验、目标类型/台站数/截止时刻分层、逐样本空间场比较和 event_id cluster bootstrap；对象数组仅在显式 --trusted-pickle-input 后加载。
  - tests/test_random_geometry_full_npz_analysis.py: 覆盖二维目标事件聚合、配对点误差 bootstrap 和空间改善统计。
  - reports/rt55_rt56_random_geometry_full_20260907.summary.json: 全量机器可读结果、输入 SHA256 和分析协议。
  - reports/rt55_rt56_random_geometry_full_20260907.paired.csv: 目标类型及台站数/截止时刻分层的紧凑配对表。
  - reports/rt55_rt56_random_geometry_full_20260907.md: 面向决策的结果摘要和空间场表。
verification:
  - python -m unittest tests.test_random_geometry_full_npz_analysis tests.test_query_geometry_diagnostics tests.test_eval_checkpoint_formal -v: PASS, 42 tests
  - git diff --check: PASS
  - 正式离线命令（2,000 次 event_id cluster bootstrap）: PASS
  - 两个 NPZ 的 12 个样本身份/几何/标签数组逐项完全相等: PASS
  - 从 NPZ 重算的 targets/MAE/RMSE/bias/correlation/R2/slope/intercept/sigma/coverage 与两个 formal metrics JSON 在 rtol=atol=1e-12 内一致: PASS
compatibility:
  - 本轮没有修改 train_light.py、eval_checkpoint.py、模型结构、配置、checkpoint 加载或推理路径；RT55 生产加载和推理行为不变。
  - 既有 query-geometry 与 formal-evaluation 回归测试全部通过。
hpc_status:
  - not submitted；复用 chaosuan_res 中已完成的 RT55 zero-shot random-mask 与 RT56 finetuned random-geometry NPZ。
remaining_risks:
  - 这是 validation 配对分析，不是独立 test 结论；RT56 best checkpoint 的选择可能带来 validation selection optimism。
  - 点指标按 target 加权，空间指标按 realtime sample 等权，两类量回答的问题不同。
  - bootstrap 以 event_id 为 cluster，把同一事件的所有截止时刻和重复 occurrence 一起重采样；这是保守相关性处理，但不能替代独立 test。
  - 输入 NPZ 为历史 object-array 格式，必须信任来源；工具对此要求显式确认。
  - formal NLL 数值来自随 NPZ 一起生成的 formal metrics JSON；本工具反校验的是可由已加载均值、方差和标签直接复现的点指标与 coverage。
review_request:
  - 审查 ef98ffa342f52d01f749809a39665ea6c1ee3127 的对齐、对象数组转换、target/sample 权重、event_id cluster bootstrap、空间指标方向与置信区间实现是否正确。
  - 检查以下结论是否被数据充分支持，并指出任何过度解释：RT56 应保留；总体与空间排序改善；1 秒窗口退化；空间 P95-P05 动态范围仍受压缩。
  - 结合当前已有的 RT55 normal、RT55 zero-shot random、RT56 random 和 RT56 normal 结果，给出按优先级排序的下一步工作建议。
  - 下一步建议必须分成“现在必须做”“下一项高价值工作”“以后可选”，以快速推进为原则；不要建议重复现有 smoke 或重复能够从现有 NPZ 离线回答的诊断。
  - 如果确实需要新的超算实验，请最多推荐一个最小实验，并给出它要区分的假设、所需 split、主要指标和明确 go/no-go 标准。
[/CODEX-RESULT]
```

## 已核验的核心数值

- 样本完全配对：9,681 个 realtime samples、1,310 个唯一 event IDs、75,654 个有效目标。
- 总体 MAE：RT55 `0.2831`，RT56 `0.2572`；差值（RT56 − RT55）`-0.0259`，event-cluster 95% CI `[-0.0288, -0.0230]`。
- 总体 RMSE：RT55 `0.3820`，RT56 `0.3313`；差值 `-0.0507`，95% CI `[-0.0549, -0.0465]`。
- untriggered 目标的 MAE 改善 `-0.0327`，大于 triggered non-input 的 `-0.0173`；两者 CI 都不跨 0。
- 六个 requested input-station-count 分层的 MAE 都改善，cluster CI 均不跨 0。
- 截止时刻为 1 s 时 MAE 从 `0.2556` 变为 `0.2639`，差值 `+0.0082`，95% CI `[+0.0040, +0.0125]`；3 s 仅小幅改善，5–90 s 改善逐渐明显。
- 至少五个有效目标的空间场中，mean Pearson 从 `0.3600` 提升到 `0.4148`，差值 `+0.0548`，95% CI `[+0.0473, +0.0627]`；event-centered 和 pairwise-delta 误差也改善。
- 同一空间场集合的 mean P95-P05 range absolute error 从 `0.5082` 变为 `0.5205`，差值 `+0.0123`，95% CI `[+0.0095, +0.0151]`。候选模型的平均预测范围 `0.3169`，真值为 `0.8313`，说明空间振幅压缩仍然明显，且平均意义下略有加重。
- 正式 uncertainty 指标同时改善：coverage 1σ 从 `0.4423` 到 `0.6549`，coverage 2σ 从 `0.6918` 到 `0.9411`；predictive sigma mean 从 `0.1897` 增至 `0.3037`。

## 请 ChatGPT 输出

请按以下结构回应：

1. `审查结论`：通过 / 有条件通过 / 不通过，并列出代码或统计问题。
2. `结果解释`：区分已证实事实、合理推断和当前不能声称的结论。
3. `下一步优先级`：现在必须做、下一项高价值工作、以后可选。
4. `最小新实验（仅在必要时）`：一个实验，不要扩展成新的诊断矩阵。
5. `给 Codex 的下一轮 AI-HANDOFF`：若需要改代码，提供可直接执行的完整交接块；若不需要，明确写 `proposed_change: none`。
