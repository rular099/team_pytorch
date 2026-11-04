# FE02 DECISIONS

<!-- ai-memory:begin -->
### D-FE02-001 · 2026-10-08

- 实验线：FE02
- 背景：事件摘要以什么路径进入PGA？
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：固定R0/A/B/C/M五组；C非严格等容量，M只增加key不增加独立观测信息。
- 适用范围：FE02矩阵
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/FE02_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/FE02_PROTOCOL.md)

### D-FE02-002 · 2026-10-14

- 实验线：FE02
- 背景：直接改受锁engine会破坏已有resume身份。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：通过opt-in独立hash-pinned兼容层保持RNG字节/顺序，并保留原lock与last-only恢复。
- 适用范围：已知9f70371来源任务
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md)

### D-FE02-HANDOFF-20261104 · 2026-11-04

- 实验线：FE02
- 背景：新会话需要可追溯接手，且不能把待办当执行授权或覆盖另一条实验线。
- 讨论选项：扩展已有record与生成视图，复用现有检查器；没有另建并行科研事实源。
- 决定：本线独占状态目录；接手默认只读；结果commit、未完成和恢复步骤明确记录；HPC/协议变更依当前用户授权。
- 适用范围：FE02接手与收尾
- 后续推翻：未见记录
- 历史依据：source_recorded
- 来源：[docs/ai/START_HERE.md](https://github.com/rular099/team_pytorch/blob/5bcf4f558f9571a2ad44ff2639f62bb64e228c2b/docs/ai/START_HERE.md)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
