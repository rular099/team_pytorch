# FE01-A01 audit等价门控修复与重提

基于 `7d91791c2b9d2a86148147e132dbf2e81a83f52e`，同一 A01 独立分支。
用户已提交5项audit，其中3项在旧ON复用的 `reuse_equivalence` 中失败。
这条路径只用于TEAM/PhaseNet/EQT；另外两项的完整门控证据尚未下载核验。
timm的FutureWarning不是本次异常原因，无需重装环境。

## 已核验的问题与修复

旧代码把 eval前向、loss完全相等和更新后全部权重SHA相等合并成一个笼统断言，
没有保存哪项不一致。它还没有在两次训练前向前恢复同一随机状态。
小型非零dropout用例本地可复现旧断言失败；修复后通过。
当前生产配置dropout为零，所以随机状态缺陷不足以证明本次生产失败的具体来源。
GPU归约/卷积反向的末位浮点差异是合理推断，仍须用新版实际报告核验。

修复只改变A01 audit比较器，不改变训练、模型、数据、采样或loss定义：

- 原init、checkpoint/config/encoder/旧lock以及冻结115文件SHA仍严格核验。
- 新旧ON的eval/train前向重放相同CPU/DCU随机状态。
- audit范围内设置deterministic cuDNN，关闭benchmark与TF32；退出时恢复设置和随机状态。
- FP32前向、loss、梯度、更新后全状态逐元素检查atol=1e-6、rtol=1e-5。
  名称、shape、dtype、梯度有无、整数状态及optimizer/scheduler设置保持严格一致。
- 更新后的两个SHA继续记录，同时明确区别位级相等与数值等价。
- 在拒绝复用之前保存 `ON_equivalence.json`；训练后ON前向另存独立报告。
  不通过放大阈值、删旧SHA、改lock或改随机种子来绕过实际不等价。

新负例把A01梯度反向，虽然初始权重、前向和loss一致，梯度和更新门控仍失败；
NaN、缺参数、shape/dtype漂移和超容差差异也会拒绝。
本地没有DCU或生产checkpoint，实际超算等价性仍为待核验。

## 文件放哪里，接下来提交什么

把以下三个新文件放在超算原 `team_pytorch_fe01` 目录同一个位置：

1. `fe01_a01_audit_fix_source.tar.gz`
2. `fe01_a01_audit_fix_source.tar.gz.sha256`
3. `deploy_a01_audit_fix.sh`

在**登录节点**执行 `bash deploy_a01_audit_fix.sh`。
它校验源码包并创建新release，复制旧release的私有环境设置，保留module、conda、
partition、资源、原数据与权重路径，只改新代码/输出/批次以及A01专用DiTing manifest位置。
旧release、失败目录、日志和已产生的门控保持原样；它不会提交作业或导入torch。
如果目标release已经存在则停止；不要删除旧输出以强行重试。

进入脚本打印的新目录，先提交一个environment作业：

```bash
unset A01_ENV_FILE
bash scripts/fe01_a01/submit.sh environment
```

**该作业通过后**，在同一新目录提交：

```bash
bash scripts/fe01_a01/submit.sh audit 0-4
```

新源码SHA使旧门控失效，所以全部5项都要重跑audit，不复制旧的两个成功lock。
每次显式提交仅一个stage；没有自动开启diagnostics/pilot/train/eval。
全部audit通过后继续原runbook的diagnostics阶段。

## 新版若仍失败

日志现在会指出具体检查和详细JSON位置。在新输出根：

```text
audits/a01__team_original_scratch__off__seed42/ON_equivalence.json
audits/a01__phasenet_pretrained_frozen__off__seed42/ON_equivalence.json
audits/a01__eqt_pretrained_frozen__off__seed42/ON_equivalence.json
```

重点查看 `failed_checks`、`checks.*.failures`、`largest_differences` 和 `execution`。
例如仅更新后SHA不同、数值比较通过会正常放行；梯度/参数超容差仍停止。
若在更早的文件身份或配置检查失败，相应JSON可能尚未生成，错误会明确指出前置条件。
下载对应JSON及Slurm out/err即可判断后续修复，不能把本地CPU通过称为DCU已通过。
