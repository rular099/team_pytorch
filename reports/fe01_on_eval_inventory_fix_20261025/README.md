# V4 inventory manifest 路径失败与恢复交付

任务 `20261025-fe01-on-inventory-manifest-fix`；基点
`a5c677c06ae67ed5c47b0cab0905908e1275cf81`，分支
`exp/fe01-on-warning-map-evaluation`，工作区 `team_pytorch_fe01_on_eval`。
这是既有 V4 的运营修复，不改变模型、checkpoint、人口、采样、指标或阈值。

用户提供的 Job31089276_0 日志显示 inventory 在
`sources.inspect → check_family` 抛出 `Pretrained manifest SHA differs`。
原实现将 DiTing 的独立 manifest 改到通用清单，两个清单具有不同身份。
`timm` FutureWarning 和可选 apex/xformers 提示不是该 traceback 的退出原因。
原日志没有实际运行时 manifest 字节，不能声称权重损坏或给出未取得的 actual SHA。

修复保留配置中可访问的原路径；原文件缺失时可复用既有 A01_DITING_MANIFEST 或按原 basename 搬目录。
原清单与 asset 的 SHA 校验、checkpoint 完整字节、epoch/config/state/encoder 检查全部保留。
新增路径/expected/actual SHA 错误信息和逐模型进度，下一份日志可直接定位文件。
旧源码、锁、seal 和在跑作业不改。

使用[修复运行手册](../../docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md)中的
`recover_inventory.sh`，在新 run `on_v4_20261025_inventory_fix` 只提交修复后的 inventory。
三个独立 CPU 阶段中已封存项目会保留原 seal/执行来源后复用；未封存项目不拷半成品，等待旧作业结束后再次恢复。
移植后的新 seal 认证本次传输，收据认证原执行，不能把旧结果说成新代码重新计算。

实际本地验证：**69 tests PASS**；编译、所有模块 Torch-free import、Bash 语法 PASS。
包括不同家族、独立 A01 原路径、错误 SHA/缺文件负向门，原配置不变，CPU 源码/人口/字节保护，
损坏拒绝、缓存原 seal、真实 prepare/恢复编排的 fake sbatch、在跑去重与迟完成结果补入。
原 a5c677c release 与修复源码之间的 CPU 兼容门已独立实际执行 PASS。

证据：

- [本地完整验证日志](local_verification.log)
- [真实原 release 与修复执行树兼容证明](source_compatibility.json)
- [输入日志 SHA、既有锁身份、验证范围与 HPC 状态](verification.json)

本地没有生产 checkpoint/Japan HDF5/DCU，真实修复 inventory、设备前向和后续阶段 **NOT_RUN**。
修复重提 **NOT_SUBMITTED**；其他旧作业状态没有下载证据，脚本在超算以实际完整 seal 判断，未推测完成。
原 CPU 分析报告仍以其原执行来源解读，此次未追加任何 Japan 模型成绩。
