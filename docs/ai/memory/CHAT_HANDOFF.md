# CHAT_HANDOFF

<!-- ai-memory:begin -->
科研目标：利用截至当前时刻可用的多台站三分量波形及输入/查询台站位置，预测查询台站最终 PGA 的概率分布；同时关注非输入目标、空间动态范围、校准与实时可用性。报告坐标 log10(m/s²)。保留 RT55 checkpoint 加载与推理兼容，不用 held-out test 调参。

当前任务为跨终端轻量同步。独立sync分支已保存原未提交报告与任务依据；恢复工作区和U盘校验请先读[终端恢复入口](../sync/TERMINAL_HANDOFF.md)。本轮没有新增科研训练/评价，原六个工作区保留，外接chaosuan_res尚不可读取，不能将清单当作已验证备份。

全局由协调者维护，各科研实例只写自己的 tracks/<ID>/。公共改动经审阅后再引入工作分支，本轮没有开始新训练或提交超算。

状态核验时间：2026-11-02T08:55:53+00:00。精确 branch/commit 见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。

- **RT55_RT56** (`verified_complete`)：RT55确实使用匹配波形；RT56随机几何适配有收益也有normal遗忘，任意查询空间辨识仍未解决。 9月快照核验的是RT55 epoch20 test；含ep32_retry的目录不能证明epoch32 test。epoch32 validation roll证明系统使用匹配波形，不能证明adapter不collapse。 [本线来源](tracks/RT55_RT56/STATUS.md)。
- **RT57** (`verified_complete`)：GTNP v2已评估；残差缩放gamma=1.66来自同一validation选择，不能当独立test证据。 random/normal配对与指标复算在报告中通过；gamma=1.66当时为离线分析选择。 [本线来源](tracks/RT57/STATUS.md)。
- **RT58** (`verified_complete`)：WATF相对相同gamma=1.66基线的random MAE变差，准确input anchor未转成空间幅度收益。 报告给出anchor correction相对相同gamma基线random MAE +0.000458 dex；不能把gamma收益算作新anchor收益。 [本线来源](tracks/RT58/STATUS.md)。
- **RT59** (`verified_complete`)：RT59同时改善random/normal点误差，但严格门为NO_GO 22/29，仍作为后续开发reference。 报告门表22 PASS/7 FAIL；normal与random点误差及NLL/Brier改善不能掩盖空间范围/校准保持门失败。 [本线来源](tracks/RT59/STATUS.md)。
- **RT60** (`verified_complete`)：final-contrast readout空间改善很小且normal非输入误差变差；保留RT59 parent。 mechanism 8/14、legacy 18/25；按预注册行动保留RT59，并归档RT60部分机制/负结果。 [本线来源](tracks/RT60/STATUS.md)。
- **RT61** (`verified_complete`)：wave-geometry conditioning机制门13/14、联合有用进展0/4；未达到预定成功条件，保留RT59。 mechanism 13/14、useful 0/4、legacy 18/25；adapter-off是同checkpoint反事实，不能当独立训练消融。 [本线来源](tracks/RT61/STATUS.md)。
- **FE01** (`in_progress`)：三家族×三seed原比较已完成；EVAL1八组forward PASS、TEAM三组被评价清单条件阻挡，后续A01/ON另线推进。 原九组各12轮的报告与EVAL1 summary均已归档；本轮只核验保存文件，未重算大型原始产物。 [本线来源](tracks/FE01/STATUS.md)。
- **FE01_A01** (`verified_complete`)：10月21日五项seal和16份validation预测已归档；已取代10月15日等待pack的旧进度。 已发布中立报告记录16份导出：各1310events/194265请求，noninput139440；selected和epoch12不是独立重复。 [本线来源](tracks/FE01_A01/STATUS.md)。
- **FE01_ON** (`in_progress`)：四ON旧导出CPU比较已发布；10月25日修复inventory清单路径，真实修复inventory与新回放结果待核验。 真实CPU比较与合成动态图均存在，但合成图醒目标SYNTHETIC，不能当Japan新前向结果。 [本线来源](tracks/FE01_ON/STATUS.md)。
- **FE02** (`blocked`)：事件融合R0/A/B/C/M独立于FE01；M组5/12轮后超时，CPU RNG续跑兼容层已交付，真实恢复待证。 M/seed42 audit及5轮记录存在；续跑在RNG恢复时报错，没有进入新一轮，不能写成12轮完成。 [本线来源](tracks/FE02/STATUS.md)。
- **V01** (`verified_complete`)：速度/P前缺失控制闭环已完成；MM总体优于FF但高PGA更差，不能据此证明padding无害或速度上限。 报告核验10格导出、4750个metric值和400行CI；fixed epoch8/单seed/非输入query-only，不能与RT55 normal-all混排。 [本线来源](tracks/V01/STATUS.md)。
- **LEGACY** (`unknown`)：早期512-event容量/空间消融保留索引；graph与其余历史分支只确认存在，最终执行进度unknown。 Git保留早期stage2实验计划和历史branch；远端24个branch完整清单见audit/remote_refs.json。 [本线来源](tracks/LEGACY/STATUS.md)。

尚未解决：任意非输入查询的空间辨识、单输入空间场压缩、normal/random收益保持与概率校准。单seed和反复使用validation限制结论；原始权重/数据/调度退出码缺失必须分别保留，不能用缺同步推断未完成。

优先顺序：先审阅本记忆体系；FE01 ON核对inventory修复的实际回执；FE02核对RNG兼容receipt和后续epoch；FE01-A01读取10月21日完整结果，V01按限定科学结论独立审阅收尾。以上不自动触发计算。

追溯方法：用 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) 找实验ID，用 [DECISION_LOG.md](DECISION_LOG.md) 找决策ID，再打开条目中的固定commit链接。[PROJECT_MEMORY.md](PROJECT_MEMORY.md) 保存长期规则；每线 STATUS/DECISIONS/EXPERIMENT_INDEX/CHAT_HANDOFF 提供详细入口。Git之外的历史聊天未全部读取，缺原讨论的原因标 needs_history_review；使用 [HISTORY_IMPORT_TEMPLATE.md](HISTORY_IMPORT_TEMPLATE.md) 补录。旧报告保留其日期语境，不能当当前全局状态。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
