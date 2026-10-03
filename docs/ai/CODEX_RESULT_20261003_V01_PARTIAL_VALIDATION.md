# CODEX-RESULT：V01 首轮结果整理与远端审阅交付

日期：2026-10-03。原始材料：用户提供的 `chaosuan_res/vel`；结果入口：
`reports/v01_velocity_padding_validation_20261003/RESULT_REVIEW.md`。

## 已核实事实

1. 三臂 init/last/best 共 9 个真实 checkpoint：init epoch 0，last/best epoch 8；
   三臂 init 保存张量和文件 hash 相同；最后 optimizer step 均 1496。冻结 encoder 不在
   non-encoder checkpoint 中，未收到实际 encoder / RT55 parent / split 文件。
2. 四个速度 random 导出：1194 事件、84259 目标观测；各时刻固定 12037 个目标。
   query 坐标、标签、cutoff、类型对齐，全部 one-to-one。
3. A-pair random：1193 事件、84329 目标观测，8358 行中仅 8313 唯一 event/time，
   有 45 条额外重复，故不作跨臂 paired CI。
4. 五个 normal 日志全部在 generator 中失败：`Found event without PGA idx=7`；
   没有 normal NPZ / metrics。并非只是用户遗漏了文件。
5. MM vs FF random 合并 MAE 0.236895 vs 0.244516；early135 paired delta
   -0.008716，95% event-cluster CI [-0.009655, -0.007787]。
6. 同一模型视图变化影响小：FM−FF early135 -0.001075；MM−MF early135
   -0.000332，CI 跨零。不能把两个训练臂的差异解释成纯 inference 缺失效应。
7. 单输入的场景内空间 range ratio 中位数约 0.0063；空间塌缩未解决。
8. 三臂配置指向 2004–2024 的 21 个 shard；继承 full_data_manifest 是旧数据描述，
   不能证明实际 train cohort 或全量 Hi-net 训练。标签仍是 KNET query PGA。

## 本轮改动

- 独立归档此前未提交的 recovery 修复：
  `8829029f7a1a3ef49cedeccef9385298c683ac01`，不把它当成本轮新设计。
- 新增只读离线整理工具 `tools/summarize_v01_results.py` 和 5 项统计/配对测试。
- 新增约 3 MB 轻量结果包：指标分层、25 个 paired comparisons/time-window CI、
  每事件 bootstrap sufficient statistics、模型/配置/文件 hash、normal traceback 摘录。
- 更新当前交接和 ChatGPT 阅读入口；不上传 raw checkpoint / NPZ / 波形 / 完整 Slurm log。
- 本轮没有修 normal generator，也没有改变训练/评估协议；没有代用户提交超算作业。

## 验证与兼容性

```bash
python -m unittest discover -s tests
# 135 tests PASS，包括 RT55 兼容性和 V01 recovery/统计测试

bash -n tools/run_v01_prep_padding_controls_slurm.sh \
  tools/recover_v01_prep_padding_controls_slurm.sh \
  train_light_slurm.sh eval_checkpoint_slurm.sh
# PASS

python tools/summarize_v01_results.py \
  --input-root ../chaosuan_res/vel \
  --output-dir reports/v01_velocity_padding_validation_20261003 --audit-checkpoints
# PASS；5 NPZ 复算主要指标匹配 metrics.json；9 checkpoint 元数据/哈希读验
```

不修改 RT55 的模型计算、原始配置、标签、loss 或默认 inference；
`gemini_models.py` / `gemini_util_light.py` / `train_light.py` / `eval_checkpoint.py`
相对本轮开始 HEAD 均未变。具体 hash 和缺失项见报告 `analysis_provenance.json`。
本地测试不等于超算运行或完整数据契约证明。

## 请求 ChatGPT 审阅

见 `CHATGPT_REVIEW_REQUEST_20261003_V01.md`。优先独立核验已有 random 机制结果，
并给出最小下一步：是否只需补 normal；A-pair 重复/空样本如何处理；是否还有任何必须
补的证据。不要用 validation 选 epoch/扫参，不发散为新架构或一串重复 smoke。

```text
[CODEX-RESULT]
task_id: 20261003-v01-partial-validation-review
base_commit: 8f54477ed11bf2623f40121648f7fcc756ada1a2
result_commit: 本文件所在结果提交，完整 SHA 由最终用户回复和 git log 提供
branch: exp/v01-velocity-prep-padding-control
changed_files:
  - recovery runtime/helper/tests/docs: 归档前轮未提交修复，独立 commit 8829029...
  - tools/summarize_v01_results.py and tests/test_v01_result_summary.py
  - reports/v01_velocity_padding_validation_20261003/*: 轻量实证材料
  - docs/ai and SESSION_SUMMARY.md: 更新当前状态和审阅入口
verification:
  - unittest: 135 PASS; shell syntax PASS; real artifact recomputation PASS
compatibility:
  - RT55 regression suite PASS; model/primary runtime/config untouched this turn
hpc_status:
  - user-provided epoch-8 checkpoints verified; 5 random exported, 5 normal failed
  - no new HPC job submitted this turn; sacct not supplied
remaining_risks:
  - normal absent; A-pair duplicates; missing cohort/source/sensor/raw-wave provenance
  - single seed; validation only; non-encoder checkpoint does not verify frozen encoder body
review_request:
  - verify evidence and interpretation; propose only the smallest required next step
[/CODEX-RESULT]
```
