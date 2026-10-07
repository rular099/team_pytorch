# A01 audit等价门控修复：本地验证

修复基点：`7d91791c2b9d2a86148147e132dbf2e81a83f52e`。
分支：`exp/fe01-a01-absolute-amplitude-ablation`，独立A01工作区。
来源：用户提供的超算traceback显示3项audit在旧ON等价性断言失败，没有具体数值报告。
用户已提交的这批audit不是本地结果；未收到Job ID、完整日志或另外两项的门控文件。

已复现：旧实现对不同随机状态下的非零dropout训练前向作完全相等比较，会失败。
生产配置dropout为零，因此该复现验证代码缺陷，不足以确定生产失败原因。
原断言同时要求GPU更新后的全部权重SHA一致；位级差异的实际影响待新版DCU报告确认。

修复重放同一随机状态，并保留严格init/SHA/名称/shape/dtype/整数状态。
FP32 eval/train前向、loss、梯度和更新后全状态采用固定数值容差；保存全部分项结果，
更新SHA照常记录，失败时指出具体检查和报告路径。仅audit局部改变后端设置，退出恢复。
原训练循环、optimizer、数据、采样、11维ON/OFF策略及原FE01/RT55/RT61源码未改变。

本地实际执行：61项聚焦与原加载/窗口/采样/指标回归全部通过，见 [pytest输出](pytest.txt)。
Python compile/import、Python3.9语法、shell bash-n、原115文件SHA通过，见 [核验](verification.json)。
三个原工作区HEAD/branch/status/staged/unstaged指纹未变，见 [工作区核验](original_workspace_verification.json)。

新增测试实际证据：

- [重放dropout的正例](paired_dropout_PASS.json)：eval/train前向、loss、梯度和更新通过。
- [反向梯度的负例](reversed_gradient_FAIL.json)：初始状态、前向、loss一致，梯度/更新被拒绝。
- 另测试末位浮点差异允许，NaN、缺参数、shape/dtype改变、超容差差异拒绝；
  正常及异常退出均恢复CPU随机状态与原后端设置。

本地CPU合成验证不能代替实际DCU或已训练生产checkpoint核验。
新版超算environment/audit及正式训练/评价状态：**NOT_SUBMITTED**。
旧版用户audit已有3项失败；另外两项是否通过须查看真实门控文件。

上传与重提步骤见 [修复说明](../../docs/ai/FE01_A01_AUDIT_FIX.md)。
新源码SHA使旧门控失效，新批次须先手动environment、通过后audit 0-4。
旧失败、成功目录均保留，不拷旧lock冒充新来源通过，也不自动开启训练。
