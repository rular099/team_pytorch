# 历史聊天补录模板

未读取全部旧聊天。只补录Git未保存的讨论，不能把模型回忆当用户决定。
先填本模板并提交审阅草稿，确认来源后再加入对应线record；协调者汇总全局。

```text
history_import_id: HIST-<track>-<日期>-<序号>
track_id: <FE01/FE02/...>
source_chat_date: <含时区；未知则unknown>
source_chat_title_or_locator: <可定位标题/用户提供导出位置；没有则unknown>
source_excerpt: <必要的短摘录；不要复制整段私密聊天>
speaker: <用户/ChatGPT/Codex；未知则unknown>
decision_id: <已有ID或新ID>
problem: <当时问题>
options_actually_discussed: <原文支持的选项；没有记载就写未提供>
decision: <决定、建议、待讨论三者区分>
reason_in_source: <没有原始理由就写needs_history_review>
scope: <实验线、数据/协议版本、是否限本轮>
git_evidence: <完整SHA:path；缺失则unknown>
conflicting_sources: <多个来源及差异>
supersedes: <ID或none；不能删除原条目>
review_status: needs_history_review
reviewer_and_date: <待审阅>
```

原Git报告与用户补录冲突时保留双来源与时间范围；实际新授权按当前用户指令执行，
历史事实仍需核验。只有取得对应缺失决策的可靠来源后才建议归档那部分聊天；不要删除聊天或Git历史。
