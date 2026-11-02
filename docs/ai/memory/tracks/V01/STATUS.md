# V01 STATUS

<!-- ai-memory:begin -->
track_id: `V01`

branch: `exp/v01-velocity-prep-padding-control`

commit_sha: `d8f2436273fbe990e77f7151d86a6b629e99c5c1`

last_verified_at: `2026-11-02T08:55:52+00:00`

current_state: `verified_complete`

原证据日期：2026-10-06；本次只核验可见来源，不代表重新跑过实验。

完成范围：validation工程闭环与证据包完成；限定科学负机制结论待独立收尾审阅。

速度/P前缺失控制闭环已完成；MM总体优于FF但高PGA更差，不能据此证明padding无害或速度上限。

### verified_results

- 报告核验10格导出、4750个metric值和400行CI；fixed epoch8/单seed/非输入query-only，不能与RT55 normal-all混排。 来源：[reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)；[reports/v01_validation_closure_20261006/verification.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/verification.json)
- MM总体MAE更低但高PGA退化；A/V common-available MM−AA区间跨零，不证明等效或模态普遍胜出。 来源：[reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)；[reports/v01_validation_closure_20261006/headline_metrics.csv](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/headline_metrics.csv)
- 2026-11-02补存9月30日技术交流v1/v2的讲稿、复现脚本与轻量表；这是历史材料归档，不替代10月6日V01 closure当前结论。 来源：[docs/ai/SYNC_SCOPE_20261102.md](https://github.com/rular099/team_pytorch/blob/52ea935249e490033f49209e106eecdedaa9275a/docs/ai/SYNC_SCOPE_20261102.md)

### pending_tasks

- 按原独立审阅请求判定能否限定收尾；不默认重训、扫参数或打开test。

### limitations

- 仅开发validation/单seed；不等于held-out test、跨seed显著性或未见台站泛化。
- 历史训练runtime/encoder身份与完整采样轨迹仍未认证。
- 上游全记录centering/cutout+1的严格在线因果性未认证；无sacct不宣称调度COMPLETED。

### evidence_paths

- [reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)
- [reports/v01_validation_closure_20261006/verification.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/verification.json)
- [reports/v01_validation_closure_20261006/headline_metrics.csv](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/headline_metrics.csv)
- [reports/v01_validation_closure_20261006/evidence/configs/vfull__vfull__normal.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/evidence/configs/vfull__vfull__normal.json)
- [docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md)
- [docs/ai/SYNC_SCOPE_20261102.md](https://github.com/rular099/team_pytorch/blob/52ea935249e490033f49209e106eecdedaa9275a/docs/ai/SYNC_SCOPE_20261102.md)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
