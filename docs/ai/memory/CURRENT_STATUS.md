# CURRENT_STATUS

<!-- ai-memory:begin -->
更新时间：2026-11-04T07:09:47+00:00

生成所依据的协调分支 commit：`5bcf4f558f9571a2ad44ff2639f62bb64e228c2b`

协调分支：`chore/persistent-ai-handoff`

此 SHA 是生成输入，不是整个科研项目的单一 HEAD；各线见下表。

| track_id | branch | commit_sha | last_verified_at | current_state |
|---|---|---|---|---|
| RT55_RT56 | `query-geometry-diagnostics` | `e74f977271b0bcde42c7ac7a18cb77ac60107e34` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| RT57 | `rt57-geo-temporal-neural-process` | `143b63af5d0a477ac388d310dc470eb7349b5c90` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| RT58 | `rt58-waveform-anchor-transfer-field` | `40830acd475bd40347fef03427b52c4a04432dc3` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| RT59 | `rt59-dual-objective-transport-v3` | `d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| RT60 | `rt60-final-contrast-readout` | `00d624d2fb01c6dd98cac150043dd15b7d8366ff` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| RT61 | `rt61-wave-geometry-residual-conditioning` | `9c95dbfaf92f36b2673d816026cd3f25b91eec66` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| FE01 | `exp/fe01-feature-extractor-site-effects` | `577a0cec43643597a61209209a770aeecf5fefa0` | 2026-11-04T07:09:47+00:00 | `in_progress` |
| FE01_A01 | `exp/fe01-a01-absolute-amplitude-ablation` | `29ffb20ba0effc0c345a0329ba3f8c658dc8cef0` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| FE01_ON | `exp/fe01-on-warning-map-evaluation` | `50672e131d876a604432b283162d5b0189d40cf0` | 2026-11-01T07:24:10+00:00 | `in_progress` |
| FE02 | `exp/fe02-event-fusion-readout` | `618661f99afb67f5c23c9f1a3fd713a0a156e7b5` | 2026-11-04T07:09:47+00:00 | `in_progress` |
| V01 | `exp/v01-velocity-prep-padding-control` | `d8f2436273fbe990e77f7151d86a6b629e99c5c1` | 2026-11-01T07:24:10+00:00 | `verified_complete` |
| LEGACY | `zhangb/layerwise-station-temporal` | `8d0131db09f9d89e215b27dda77399b481b3a92a` | 2026-11-01T07:24:10+00:00 | `unknown` |

### RT55_RT56

完成范围为已归档基线/随机几何诊断，不代表整体科研目标完成。

**verified_results**

- 9月快照核验的是RT55 epoch20 test；含ep32_retry的目录不能证明epoch32 test。epoch32 validation roll证明系统使用匹配波形，不能证明adapter不collapse。 来源：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)
- RT56 epoch6 random非输入validation改善，同时normal非输入validation退化；不能把validation改名test。 来源：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)；[docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md](https://github.com/rular099/team_pytorch/blob/e74f977271b0bcde42c7ac7a18cb77ac60107e34/docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md)

**pending_tasks**

- 历史指标用原provenance核对；新实验优先进入自己的实验线，不续写RT55兼容基线。

**evidence_paths**

- [docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)
- [docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md](https://github.com/rular099/team_pytorch/blob/e74f977271b0bcde42c7ac7a18cb77ac60107e34/docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md)

同步：retained_snapshot；来源 `docs/ai/memory/observations/RT55_RT56.json`。

### RT57

固定六轮validation报告已归档。

**verified_results**

- random/normal配对与指标复算在报告中通过；gamma=1.66当时为离线分析选择。 来源：[docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/143b63af5d0a477ac388d310dc470eb7349b5c90/docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md)

**pending_tasks**

- 需追溯时保留gamma与后续RT58 frozen-base的区别；不自动重训。

**evidence_paths**

- [docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/143b63af5d0a477ac388d310dc470eb7349b5c90/docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md)

同步：retained_snapshot；来源 `docs/ai/memory/observations/RT57.json`。

### RT58

固定epoch8验证与十项门检查已归档。

**verified_results**

- 报告给出anchor correction相对相同gamma基线random MAE +0.000458 dex；不能把gamma收益算作新anchor收益。 来源：[docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md)；[reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json)

**pending_tasks**

- 保留负结果，追溯RT59重新设计；不因总指标好看而改对照。

**evidence_paths**

- [docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md)
- [reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json)
- [pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json)

同步：retained_snapshot；来源 `docs/ai/memory/observations/RT58.json`。

### RT59

epoch8报告及29项门已归档；假设未全面通过。

**verified_results**

- 报告门表22 PASS/7 FAIL；normal与random点误差及NLL/Brier改善不能掩盖空间范围/校准保持门失败。 来源：[docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md)；[reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv)

**pending_tasks**

- 后续比较保留同一forward的RT59 reference，避免混用历史RT57。

**evidence_paths**

- [docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md)
- [reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv)
- [pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json)

同步：retained_snapshot；来源 `docs/ai/memory/observations/RT59.json`。

### RT60

epoch8验证与mechanism/legacy门已归档。

**verified_results**

- mechanism 8/14、legacy 18/25；按预注册行动保留RT59，并归档RT60部分机制/负结果。 来源：[docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md)；[reports/rt60_contrast_readout_validation_20260923/summary.json](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/reports/rt60_contrast_readout_validation_20260923/summary.json)

**pending_tasks**

- 不自动延长训练；与RT61结论分开保存。

**evidence_paths**

- [docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md)
- [reports/rt60_contrast_readout_validation_20260923/summary.json](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/reports/rt60_contrast_readout_validation_20260923/summary.json)

同步：retained_snapshot；来源 `docs/ai/memory/observations/RT60.json`。

### RT61

epoch8验证报告已完成；轻量delta权重归档缺口仍在。

**verified_results**

- mechanism 13/14、useful 0/4、legacy 18/25；adapter-off是同checkpoint反事实，不能当独立训练消融。 来源：[docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md)；[reports/rt61_wave_geometry_validation_20260928/summary.json](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/reports/rt61_wave_geometry_validation_20260928/summary.json)

**pending_tasks**

- 如需可重建归档，核对epoch8 trainable-delta；不为此重训或打开test。

**evidence_paths**

- [docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md)
- [reports/rt61_wave_geometry_validation_20260928/summary.json](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/reports/rt61_wave_geometry_validation_20260928/summary.json)
- [pga_configs/transformer_japan_full_2000_2024_rt61_wave_geometry_residual_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/pga_configs/transformer_japan_full_2000_2024_rt61_wave_geometry_residual_seed42_chaosuan.json)

同步：retained_snapshot；来源 `docs/ai/memory/observations/RT61.json`。

### FE01

九组训练结果已归档；EVAL1身份/前向门部分完成，完整补评未全部取得。

**verified_results**

- 原九组各12轮的报告与EVAL1 summary均已归档；本轮只核验保存文件，未重算大型原始产物。 来源：[reports/fe01_hpc_results_20261004/README.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_hpc_results_20261004/README.md)；[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)
- EVAL1 identity 11组通过、forward 8 PASS/3 BLOCKED；缺预训练清单SHA的TEAM scratch在前向前被阻挡，不证明权重损坏。 来源：[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)

**pending_tasks**

- 按FE01_A01与FE01_ON各自新来源审阅后续，不能把旧EVAL1阻挡当新线仍未完成。
- 站点泛化、场地效应、回放和held-out test要逐项查实际证据。

**evidence_paths**

- [reports/fe01_hpc_results_20261004/README.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_hpc_results_20261004/README.md)
- [reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)
- [docs/ai/FE01_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/docs/ai/FE01_PROTOCOL.md)
- [configs/fe01/formal/formal__phasenet_pretrained_frozen__seed42.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/configs/fe01/formal/formal__phasenet_pretrained_frozen__seed42.json)
- [docs/ai/README.md](https://github.com/rular099/team_pytorch/blob/577a0cec43643597a61209209a770aeecf5fefa0/docs/ai/README.md)
- [docs/ai/START_HERE.md](https://github.com/rular099/team_pytorch/blob/5bcf4f558f9571a2ad44ff2639f62bb64e228c2b/docs/ai/START_HERE.md)

同步：audited_from_pinned_evidence_not_deployed；来源 `exp/fe01-feature-extractor-site-effects`。

### FE01_A01

完整validation封包已发布；独立研究解释待审阅。

**verified_results**

- 已发布中立报告记录16份导出：各1310events/194265请求，noninput139440；selected和epoch12不是独立重复。 来源：[reports/fe01_a01_hpc_results_20261021/README.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/README.md)
- 报告四家族OFF−ON主MAE差均为正；需同时保留概率/空间分层与K1 field仅1253/2000有效draw的限制。 来源：[reports/fe01_a01_hpc_results_20261021/README.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/README.md)；[reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md)

**pending_tasks**

- 按中立CHATGPT_ANALYSIS_PROMPT独立审阅，尊重原任务中Codex判断另行封存的边界。
- 不要按旧10月15日状态重交评价或pack。

**evidence_paths**

- [reports/fe01_a01_hpc_results_20261021/README.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/README.md)
- [reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md)

同步：retained_snapshot；来源 `docs/ai/memory/observations/FE01_A01.json`。

### FE01_ON

真实CPU旧导出再分析已完成；生产inventory恢复/模型回放尚无完成回执。

**verified_results**

- 真实CPU比较与合成动态图均存在，但合成图醒目标SYNTHETIC，不能当Japan新前向结果。 来源：[reports/fe01_on_eval_20261022/README.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/reports/fe01_on_eval_20261022/README.md)
- 旧inventory报Pretrained manifest SHA differs；修复保留认证manifest原路径与SHA，新run只手动恢复inventory，生产模型级验证尚未运行。 来源：[docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md)；[docs/ai/FE01_ON_EVAL_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_PROTOCOL.md)

**pending_tasks**

- 核对新inventory run的seal与真实回执；沿用新run/source，不自动重提CPU或启动后续阶段。
- 只有取得真实模型级门后才能解释TTUM/shadow、生产网格与实测时延。

**evidence_paths**

- [reports/fe01_on_eval_20261022/README.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/reports/fe01_on_eval_20261022/README.md)
- [docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md)
- [docs/ai/FE01_ON_EVAL_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_PROTOCOL.md)

同步：retained_snapshot；来源 `docs/ai/memory/observations/FE01_ON.json`。

### FE02

五组seed42的12轮训练和固定validation证据已归档并复算；独立科研审阅和来源缺口仍待完成。

**verified_results**

- 已归档并复算五组seed42各12轮；共同validation十cell等权noninput域为normal/random×1/3/5/10/20秒，每组1310事件/139440非输入目标，PGA坐标log10(m/s²)；所选R0/A/B/C/M epoch为10/11/10/12/10。 来源：[reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)；[reports/fe02_hpc_results_20261104/data/headline_primary.csv](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/headline_primary.csv)；[reports/fe02_hpc_results_20261104/data/verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/verification.json)
- 轻量事件表独立重建五组主分数及63行5000次event-cluster bootstrap，seed20261008；本次为已保存复算记录的接手核验，不再次运行模型或原始数据计算。 来源：[reports/fe02_hpc_results_20261104/data/compact_verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/compact_verification.json)；[reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)
- 160份恢复回执覆盖10作业×16rank，shim SHA一致；最后R0/A/B/C launcher SHA不同于仓库入口，原入口字节未取得。 来源：[reports/fe02_hpc_results_20261104/data/verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/verification.json)；[reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)

**pending_tasks**

- 按已发布ChatGPT任务独立审阅点/概率/空间/几何/输入数分层及CI，比较Codex结论。
- 补齐后四组launcher来源、checkpoint字节或可核验元数据、committed journals及sacct；缺失保持UNKNOWN。
- 是否追加multi-seed、独立评价或新架构由用户决定；不自动训练、续跑、提交作业或动用held-out test。

**evidence_paths**

- [docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md)
- [docs/ai/FE02_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/FE02_PROTOCOL.md)
- [scripts/fe02/job.sh](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/scripts/fe02/job.sh)
- [scripts/fe02/env.sh](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/scripts/fe02/env.sh)
- [reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)
- [reports/fe02_hpc_results_20261104/data/verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/verification.json)
- [reports/fe02_hpc_results_20261104/data/compact_verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/compact_verification.json)
- [reports/fe02_hpc_results_20261104/data/headline_primary.csv](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/headline_primary.csv)
- [reports/fe02_hpc_results_20261104/CHATGPT_ANALYSIS_PROMPT.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/CHATGPT_ANALYSIS_PROMPT.md)
- [docs/ai/START_HERE.md](https://github.com/rular099/team_pytorch/blob/5bcf4f558f9571a2ad44ff2639f62bb64e228c2b/docs/ai/START_HERE.md)

同步：audited_from_pinned_evidence_not_deployed；来源 `exp/fe02-event-fusion-readout`。

### V01

validation工程闭环与证据包完成；限定科学负机制结论待独立收尾审阅。

**verified_results**

- 报告核验10格导出、4750个metric值和400行CI；fixed epoch8/单seed/非输入query-only，不能与RT55 normal-all混排。 来源：[reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)；[reports/v01_validation_closure_20261006/verification.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/verification.json)
- MM总体MAE更低但高PGA退化；A/V common-available MM−AA区间跨零，不证明等效或模态普遍胜出。 来源：[reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)；[reports/v01_validation_closure_20261006/headline_metrics.csv](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/headline_metrics.csv)

**pending_tasks**

- 按原独立审阅请求判定能否限定收尾；不默认重训、扫参数或打开test。

**evidence_paths**

- [reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)
- [reports/v01_validation_closure_20261006/verification.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/verification.json)
- [reports/v01_validation_closure_20261006/headline_metrics.csv](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/headline_metrics.csv)
- [reports/v01_validation_closure_20261006/evidence/configs/vfull__vfull__normal.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/evidence/configs/vfull__vfull__normal.json)
- [docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md)

同步：retained_snapshot；来源 `docs/ai/memory/observations/V01.json`。

### LEGACY

历史导航恢复，不断言早期b系列及所有未展开分支的最终状态。

**verified_results**

- Git保留早期stage2实验计划和历史branch；远端24个branch完整清单见audit/remote_refs.json。 来源：[reports/stage2_512_layerwise_station_temporal_plan.md](https://github.com/rular099/team_pytorch/blob/8d0131db09f9d89e215b27dda77399b481b3a92a/reports/stage2_512_layerwise_station_temporal_plan.md)

**pending_tasks**

- 需要某项早期决策时从固定报告与配置补录；缺聊天依据标needs_history_review。
- graph分支62c821042baeb435a9345af4e323314d14784c8c仅确认存在，未恢复完整结果。

**evidence_paths**

- [reports/stage2_512_layerwise_station_temporal_plan.md](https://github.com/rular099/team_pytorch/blob/8d0131db09f9d89e215b27dda77399b481b3a92a/reports/stage2_512_layerwise_station_temporal_plan.md)

同步：retained_snapshot；来源 `docs/ai/memory/observations/LEGACY.json`。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
