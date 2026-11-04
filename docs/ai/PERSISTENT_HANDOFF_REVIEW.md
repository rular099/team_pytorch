# 请审阅：统一持久 AI 接手入口

仓库 `rular099/team_pytorch`，独立分支 `chore/persistent-ai-handoff`。
base `826acaadc89979d6ad324d014123def8353a429c`。
请以用户交付的完整 result commit 固定本轮diff，先报告实际读取的branch和SHA。

先读 [START_HERE](START_HERE.md)、[审计与四情形验收](handoff/AUDIT.md)、
[结构化验收](handoff/acceptance.json) 和 [部署与恢复](HANDOFF_WORKFLOW.md)。
再检查 FE01/FE02 的 record、STATUS、NEXT_TASK、DECISIONS，以及只读helper、
原memory工具增量和tests/test_ai_handoff.py；勿重新解释旧报告中的历史进度为当前状态。

请独立核验：

1. 是否沿用已有record单一事实源，而非复制出另一份会漂移的进度。
2. AGENTS能否路由正确实验线；歧义、错分支、未部署、未提交视图和陈旧状态是否明确报告。
3. 是否始终把NEXT_TASK当提案；代码实施与科研审阅角色、HPC/协议授权是否清楚。
4. 两条线并发更新是否只写各自track，coordinator是否无法从科研分支运行。
5. commit/path/字段/生成视图/历史ID检查和staged-only是否保留用户已有工作。
6. FE02新完整结果是否正确覆盖旧部分进度，FE01原线与A01/ON是否保持独立。
7. 是否明确区分四种读取模拟PASS与真实分支/ChatGPT账户部署；源码fingerprint风险是否说明。

科研源码、配置、launchers和历史结果不在本轮变更范围。若建议改机制，请输出AI-HANDOFF，
不要自动合并至两条科研线，也不要开始训练或提交作业。
