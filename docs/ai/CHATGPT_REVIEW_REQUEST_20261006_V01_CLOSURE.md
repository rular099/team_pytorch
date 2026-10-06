# 请 ChatGPT 审阅：V01 validation closure 真实结果

日期：2026-10-06。Repository：`rular099/team_pytorch`。
Branch：`exp/v01-velocity-prep-padding-control`。
本轮分析 base：`7d87c4007b78405bb901c67ff0b78dc5d0e85a8e`。
结果提交是包含本文的提交，精确40位SHA由用户转交；请先确认实际读到的commit，不默认读主分支。

## 本轮请求

**请独立审阅已经回传的完整验证闭环，然后给出一个明确的下一步决策。**
本轮不要求立即写代码或启动实验。不要把原失败状态当成当前状态；不要重复三臂训练、
preflight、四格速度 random、旧 recovery/all 或 query-geometry smoke。
若现有证据足以将 V01 按限定条件的负机制结果收尾，请明确说可以收尾。
若必须补充，请指出究竟是哪一个结论受哪一个缺口阻挡、现有证据为何不足、
最小行动与停止条件；不要默认再增加大矩阵、seed/剂量/epoch扫描或held-out test。

## 建议阅读顺序

1. `AGENTS.md`、`docs/ai/PROJECT_CONTEXT.md` 前部当前入口、`docs/ai/README.md`。
2. `docs/ai/CODEX_RESULT_20261006_V01_CLOSURE_RESULTS.md`。
3. `reports/v01_validation_closure_20261006/RESULT_REVIEW.md`（本轮详细结论、分母、口径）。
4. 同目录 `verification.json`、`headline_metrics.csv`、`paired_effects.csv`、
   `primary_ff_mm_recomputed_ci.csv`、`primary_ff_mm_event_sufficient_statistics.csv`。
5. 同目录 `evidence/report/metrics_by_stratum.csv`、
   `paired_event_cluster_ci_and_mse_decomposition.csv`、`coverage_and_denominators.json`、
   `one_to_one_outer_match_audit.json`、`acc_velocity_common_population_metrics.csv`、
   两份 `field_*ratio*.csv`、两张 PNG 与 bin counts。
6. 同目录 `evidence/audit/`：三份真实idx7 trace、gate、checkpoint_before、
   random signatures、AA重复映射、真实loader/cache counts和source provenance。
7. 同目录 `evidence/source_evidence/derived_cache/`、三臂训练config、
   `frozen_train_dev_membership.csv`、`no_prediction_requests.json`、
   `artifact_copy_manifest.json`（精确原文件hash，复制内容未改）。
8. 背景 `V01_REVIEW_AND_CODEX_HANDOFF_20261004.md`、原 V01 规范与10-03/10-04报告。
   历史 AA 重复导出的指标不能替代本轮 strict AA 结果。

不需要 Git 里的raw NPZ/完整ledger/大权重：这些仍在用户本地和超算。
本轮直接读全部10份NPZ/身份记录，核验4,750个metric值，重算normal/random主要FF/MM
all7/early135五分层的400行5000-draw CI，14组physical-ID/UTC outer join全部对账。
具体实现 `reports/v01_validation_closure_20261006/prepare_review.py` 可复算；没有新模型推理。

## 已核验事实

- 真实idx7事件 `20041029141300`：合法query `NIG019` 的P sample=-109，旧clock别名
  确实擦除query mask并报原错误；修复后query恢复、source可用、事件/绝对cutoff不变。
- 三臂评估前后last SHA、epoch8、optimizer step1496不变；init SHA相同。
  runtime180文件、6份新config、原split与protocol-lock哈希均通过本地核验。
- 五格normal和一格AA random完整；四格速度random没有重跑encoder，真实sensor/UTC/label
  sidecar和有界signature通过。复用范围不是完整full-model随机结果重放。
- 各速度格1,194events/8,358requests/84,259targets；均是V01 query-only、非输入类型，
  **不是历史RT55含输入目标的normal-all**。
- AA normal 8,324预测/34弃权，random 8,313预测/45弃权，均1,193可预测events。
  新AA event/time唯一，旧45个重复替代请求不再污染指标。
- 最终物化/loader候选train7,967/dev1,194events，source120,101/query142,597，21年；
  不是初始metadata 8,439/1,225，也不是完整日本速度归档或实际逐批曝光轨迹。
- sacct尚未回传；有job IDs29318960–29318967与完整应用产物，不能捏造调度器状态/资源数据。

## 请重点审查的科学判断

1. **核心假设是否应接受限定负结果？** MM−FF的all7 MAE差值normal=-0.008275、
   random=-0.007620，CI都不跨零；固定视图换训练臂几乎解释全部幅度。
   固定M模型切view近零，固定F模型有微小micro改善但macro CI跨零。
   是否支持“当前干预与预算下未观察到保留更多前缀的总体优势”，而不是“padding无害”
   或“已证明删除有益/正则化”？
2. **阈值以上的退化应如何进入主结论？** 目标PGA≥-1.2时，MM−FF MAE normal
   +0.005165、random+0.009195；NLL/Brier与早期1/3/5s也变差，主要CI均同向。
   不用总体收益掩盖这个trade-off，不按validation调阈值或挑checkpoint。
3. **bias/variance解释是否恰当？** normal/random MSE减少主要由bias²降低承担，
   centered-variance差值CI均跨零；高PGA负bias更深。没有应用bias correction。
   不把这个描述性恒等式上升为训练机制因果解释。
4. **A/V比较是否仍应保持条件性？** common available normal83,995targets、random83,162；
   MM−AA MAE CI分别[-0.001159,+0.001832]、[-0.000174,+0.002464]。
   AA对FF更好但对MM不能明确判胜，也不证明等效。覆盖率、unmatched和不同仪器/深度/
   频响/几何要一起解释；common-remote仅paired-sensor exclusion proxy。
5. **空间场是否仍是独立限制？** 单输入>=5queries，FF/MM max-min ratio median约0.006，
   pairwise-delta MAE约0.35，normal/random都未实质改善。
   P95−P05 ratio mean另存，不能混口径或把V01数据机制实验变成新架构任务。
6. **哪些剩余缺口是必须补、哪些只需limitations？** 历史training-source/encoder tensor
   identity、完整实际采样计划、raw post-P等同性、严格在线centering/cutout+1仍有边界；
   请勿把当前hash当历史证据，也勿因为旧记录缺失就自动要求重训。

## 期望输出

- 分别判定工程闭环、科学可报告范围、尚不能支持的因果/泛化结论。
- 给出可以直接用于内部报告的3–5条结论，保留normal应用场景与高PGA trade-off。
- 明确本轮能否收尾；如需补充，只列优先级最高且有验收/停止条件的最小下一步。
- 若需要Codex实施，另给带精确base_commit的 `AI-HANDOFF`；此次审阅本身不假装已改代码。

不自动启动新超算任务。下一步的学术判断由用户决定。
