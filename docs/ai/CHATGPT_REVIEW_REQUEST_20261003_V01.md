# 给 ChatGPT 的 V01 结果审阅请求

请通过 GitHub 阅读 `rular099/team_pytorch` 的
`exp/v01-velocity-prep-padding-control` 分支，先回报实际读取的完整 commit SHA。
请读取本请求所在结果提交，不要默认读主分支或旧 PROJECT_CONTEXT 的活跃分支。

阅读顺序：

1. `docs/ai/CODEX_RESULT_20261003_V01_PARTIAL_VALIDATION.md`。
2. `reports/v01_velocity_padding_validation_20261003/RESULT_REVIEW.md`。
3. 同目录 `evaluation_status.json`、`training_contract.json`、`checkpoint_audit.json`、
   `paired_event_cluster_ci.csv`、`npz_contract_audit.json`、`intervention_support_audit.json`，
   必要时查看 `source_evidence` 和分层指标表。
4. 解释机制前读 `docs/ai/V01_prompt.md` 与完整 V01 实施规范；诊断正常几何失败再看
   `gemini_util_light.py`、`eval_checkpoint.py`，不要仅依据 report 标题猜成功状态。

请独立分析以下问题，明确分开事实、推断和建议：

- 三臂同一起点、8 epoch、相同 optimizer step 的 random 结果，能支持多强的
  “P 前缺失是否有显著代价”结论？MM−FF 的改善主要来自训练臂而非 inference 视图，
  是否需要修正原研究假设？为什么不能直接归因于补零或速度本身？
- 结合 NLL/Brier/bias/coverage 和单输入约 0.0063 的空间 range ratio，评价本轮
  机制证据，而不是只按总体 MAE 排名。不要把 V01 与 RT59/RT61 不同 cohort 当公平横比。
- 五个 normal 在 `Found event without PGA idx=7` 处失败，最小可验证的数据路径
  根因检查是什么？缺什么材料？应否保留所有现有权重，仅修 opt-in V01 数据生成后
  补 normal，不重训或重建整套 cache？不要仅将 ValueError 换成跳过来掩盖 cohort 漂移。
- A-pair 出现 45 条额外 event/time duplicate，与 `_EmptySample` 后续替代机制一致但
  尚未逐条证实。怎样明确处理空样本/分母，才能得到公平且可追溯的 A/V 比较？
- preflight/cohort/split/source identity 和真实 sensor IDs 等缺口中，哪些是做下一步
  判断必须的，哪些可以暂缓？优先低成本回传已有文本，而不是重复运行实验。

请给一个有优先级的下一步决定。若需要 Codex 改代码或用户跑超算，输出明确
`[AI-HANDOFF]`，注明目标分支和本次精确 base commit、最小文件级改动、RT55 兼容路径、
验收条件、复用哪些现有 checkpoint/结果、是否只补某几个 validation。

用户希望快速推进。没有充分理由不新增 smoke、重复训练、下载/物化、扫参、多 seed、
架构升级或 RT62；不访问 held-out test，不自动提交超算任务。结果报告目前是 partial
validation，不要声称 V01 全流程完成；只有连接器确实读到 commit 才声称已读远端材料。
