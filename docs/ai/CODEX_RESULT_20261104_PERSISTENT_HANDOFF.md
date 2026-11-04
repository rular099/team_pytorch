# 统一持久 AI 接手入口交付

入口：[START_HERE](START_HERE.md)；[独立审阅任务](PERSISTENT_HANDOFF_REVIEW.md)；
[审计与四情形验收](handoff/AUDIT.md)、[结构化检查记录](handoff/acceptance.json)。

[CODEX-RESULT]
task_id: 20261104-persistent-ai-handoff
repo: rular099/team_pytorch
base_commit: 826acaadc89979d6ad324d014123def8353a429c
implementation_commit: 5bcf4f558f9571a2ad44ff2639f62bb64e228c2b
result_commit: 本状态发布提交，完整SHA由最终会话回执提供
branch: chore/persistent-ai-handoff
changed_files:
  - AGENTS/START_HERE/协作入口: 自动路由、只读接手、权限和一次性部署说明
  - FE01/FE02 record及五种视图: 复用历史、补齐NEXT_TASK、结果commit/证据/未完成/恢复步骤
  - 既有memory维护工具增量和ai_handoff: 分支/path/SHA/字段/只读检查、协调者门和staged-only范围
  - tests/test_ai_handoff.py及docs/ai/handoff: 四种恢复、并发、错误路径与原工作区保护验收
verification:
  - 25项标准库/临时Git仓库测试PASS；四种接手和并行隔离均PASS
  - Python编译/diff检查/入口链接检查PASS；memory check 0 errors，3个历史来源提示保留
compatibility:
  - 原模型/配置/launchers/reports与实验结果未改；原FE01/FE02及旧协调worktree保持
  - 维护部署不迁移旧HPC source；tools加入可能影响未来源码身份，详见START_HERE
hpc_status:
  - 本轮无训练/推理/提交/续跑；四情形为程序化读取模拟，不是四个独立模型会话
remaining_risks:
  - 原科研分支尚需一次性审阅引入；ChatGPT Project设置和仓库授权需用户保存一次
  - 状态需每轮维护，协调汇总需显式更新来源；检查器不能代替OS ACL或独立科研审阅
review_request:
  - 按精确最终commit审阅路径/角色/身份与并行隔离；不自动合并或启动NEXT_TASK
memory_updates:
  - 两线last_session记录本轮实现commit、维护决策和恢复步骤；其他10线owned records未改
[/CODEX-RESULT]
