# TEAM model_smoke 32163754_3：证据与受控重试交付

任务 `20261102-fe01-v4-team-smoke-repeatability`，基点
`50672e131d876a604432b283162d5b0189d40cf0`；分支 `exp/fe01-on-warning-map-evaluation`，
工作区 `team_pytorch_fe01_on_eval`。实际修复commit见交付消息/小包marker，正式修复任务 **NOT_SUBMITTED**。

## 已取得的事实

用户给出的err/out记录Job32163754_3在分配DCU上运行，TEAM selected ep11身份检查通过。
失败位于原 `verify_smoke` 的 `Future changed station encoder features`，绝对阈值1e-5。
依照未修改的原控制流，该次比较前的实际forward tensor、归一化/幅值统计hash和特征长度门已通过。
这不是又一次checkpoint清单身份失败。日志没有实际feature差值、事件、时刻或future变体，无法定量归因。

## 当前推断和处理

同一输入的重复计算可作为空白对照，帮助区分设备/状态重复性问题与future扰动检查失败。
原V4未设置现有A01 `paired_execution` 的审计控制，也没有保存这些对照；
这使backend数值重复性成为待验证解释，**没有证明它就是本次根因，也没有证明未来信息泄漏**。

新增小型外部运营入口只重提TEAM task3；沿用原V4 release/run/env及所有已有结果。
原runner的bytecode、模型/选择/预处理/预测和全部原smoke门保持原样，
仅在私有namespace副本中绑定加对照/诊断的callback，原模块全局绑定不变。
audit局部复用A01 paired_execution，原绝对阈值仍1e-5，失败仍停止并保留attempt。
运营源码commit/SHA和backend设置独立封入结果；原status和下游父seal认证完整证据。
具体命令与产物见[操作说明](../../docs/ai/FE01_V4_TEAM_SMOKE_FIX.md)。

## 实际本地验证

17项聚焦测试通过，覆盖真实原runner的bytecode/绑定保持、严格绝对阈值、NaN/Inf/缺特征负向门、
blank重复失败不误报future因果、原future失败继续传播、成功/失败backend恢复、只sbatch 3号、
去重/UNKNOWN拒绝、运营源码篡改拒绝及原status所需seal字段兼容。
backend与模型特征使用模拟/合成对象，scheduler使用fake脚本，**不是生产DCU/model结果**。
编译/Torch-free import/Bash语法与原冻结核心文件身份另行核对。

[验证日志](local_verification.log)、[输入SHA与源码/设备证据范围](verification.json)。
本地没有生产checkpoint/Japan HDF5/DCU，受控真实模型检查 **NOT_RUN**。
未重训、未改变checkpoint/阈值/人口、未打开test、未修改旧seal或原V4 execution tree。
