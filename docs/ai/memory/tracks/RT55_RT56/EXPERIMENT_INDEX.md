# RT55_RT56 EXPERIMENT_INDEX

<!-- ai-memory:begin -->
### RT55

- 科学问题：匹配波形是否影响最终系统？
- branch：`query-geometry-diagnostics`
- commit：`e74f977271b0bcde42c7ac7a18cb77ac60107e34`
- 状态：`verified_complete`
- 数据/协议：固定split；epoch20 test与epoch32 validation分别报告，MDN mixture mean。
- 结果性质：系统依赖波形证据；空间外推能力未认证。
- 审阅：needs_independent_review
- 关系：RT56–RT61兼容基线
- 原始入口：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)

### RT56

- 科学问题：随机非输入查询训练是否改善空间任务？
- branch：`query-geometry-diagnostics`
- commit：`e74f977271b0bcde42c7ac7a18cb77ac60107e34`
- 状态：`verified_complete`
- 数据/协议：epoch6 random/normal validation，严格逐目标配对。
- 结果性质：随机任务收益与normal遗忘同时存在。
- 审阅：needs_independent_review
- 关系：RT55迁移
- 原始入口：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)；[docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md](https://github.com/rular099/team_pytorch/blob/e74f977271b0bcde42c7ac7a18cb77ac60107e34/docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
