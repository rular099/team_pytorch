# 2026-11-02 轻量终端同步快照

来源分支 `zhangb/native-scale-adapter-scaling`，代码基线 `8bde65cb64fd860adf552dfe56505b4b31a78b13`；同步分支 `sync/rt55-rt56`。
本轮同步 88 份原有文件，6909470 字节；新增的同步说明/清单另计。没有训练、评价前向或超算提交。
科研模型、实验配置、旧报告和source lock均未改。历史报告中的日期、旧待办与旧NOT_SUBMITTED保留历史语境。

## 从哪里恢复

在新机器先读取GitHub的 `chore/workspace-sync` 分支 `docs/ai/sync/TERMINAL_HANDOFF.md`。
该入口列出本线精确commit、其他并行工作区、USB文件清单及恢复顺序。
`SYNC_INVENTORY_20261102.json`区分原文件SHA与公开副本SHA；文档/结果中的私有路径被变量化，数值未改写。
旧产物中的SHA/bytes清单是原封包来源记录，不是本轻量子集完整性证明。本次只核验传输字节，不重新认证历史科学结果。

## Git与本地产物的边界

原始NPZ/HDF5/权重、逐目标大CSV、PPT/PDF、重复压缩包与渲染草稿留在原机器；见总入口下每工作区local-artifacts清单。
RT55/56报告的PNG预览随Git同步；演示文稿目录只同步代码、讲稿、技术说明和轻量源表，PPT/渲染图须由USB恢复。
原README中的PDF/PPT等链接因此指向本地产物；未上传的文件不等于不存在，缺失也不证明任务未完成。
清单不是备份；实际USB文件尚未核验，不能据此删除、clean或reset原目录。

旧外部review包若有MANIFEST，只证明其原包；代码按该包README中的原commit追溯，重复代码和归档未再次上传。
所有当前科学状态以总入口及各线带来源的记忆为准，不能把2026年9月的计划当成今天仍未执行。

公开副本规范了文本尾部空白；报告脚本仅将字体路径改为NATIVE_REPORT_FONT/NATIVE_REPORT_FONT_BOLD及当前用户目录，科学计算未改。
