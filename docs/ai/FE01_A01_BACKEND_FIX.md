# FE01-A01：本轮失败原因与下一步提交

本次依据用户提供的 `fe01_a01_audit_reports_20261007T153854_116671.tar.gz`，
SHA256 为 `5ec24de8ba5245ca87189f01c5ef30241ddd734983db81c58c5417ea989bf124`。
失败源码为 `753702a7b4f727d31c5ebb14f54f7f3f0e528e55`。
详细证据与限制见 [真实日志汇总](../../reports/fe01_a01_backend_fix_20261007/README.md)。

## 1. 实际发生了什么

| 索引 | 模型 | 实际 audit | 后续日志 |
|---|---|---|---|
| 0 | TEAM OFF | 更新后11个元素超出旧容差，最大2.17e-6 | 后续缺 audit/diagnostics 门控退出 |
| 1 | PhaseNet OFF | AUDIT_PASS | diagnostics 到78/78；pilot无异常记录；train导入SeisBench时缓存JSON错误 |
| 2 | EQT OFF | 更新后5个元素超出旧容差，最大1.91e-6 | 后续缺 audit/diagnostics 门控退出 |
| 3 | DiTing ON | AUDIT_PASS | 前缀与统计差为0，encoder重复前向差3.81e-6触发旧绝对阈值 |
| 4 | DiTing OFF | AUDIT_PASS | 前缀与统计差为0，encoder重复前向差1.91e-6触发旧绝对阈值 |

这份包未包含 PhaseNet 的完整 diagnostics/pilot JSON，不能只凭进度行认证完整门控。
全部 train 日志均有异常退出，未提供完成的 A01 正式训练证据；旧 FE01 ON 成绩继续保留。
没有 sacct 记录，因此不推断调度器最终状态。

## 2. 新版修改与验证

- ON 复用用真实初始化、实际小批输入，在**计算节点单线程 CPU** 精确比较全部前向、
  loss、梯度和 Adam 更新；更新后的状态 SHA 必须一致。DCU 另查初始化和冻结训练后
  ON 前向。没有放宽旧 GPU 参数更新阈值；正式训练的模型、loss、优化器、DDP、预算不变。
- 未来扰动必须保持完整六个模型输入与前缀统计精确相同。仅下游重复前向使用既有
  相对/绝对容差，并重放随机状态。输入变化仍立即失败，失败 JSON 会保留。
- 每个训练 rank 使用独立缓存，修复日志中 SeisBench 首次导入的 `config.json` 竞争。
- 提交前检查前置门控，未通过时直接在登录节点停止，不申请训练节点。

本地70项回归通过，包括真实离线 PhaseNet/EQT encoder、缩小的合成下游模型、
单个权重末位错误必须失败、真实输入泄漏必须失败、16 rank缓存隔离与假调度器零误提交。
本地没有生产初始化 checkpoint、Japan HDF5 或 DCU，**新版生产门控尚未执行**。
CPU 精确更新控制证明的是实现等价；并不证明 DCU Adam 更新逐位可重现。

## 3. 本地：只上传这三个新文件

从 `team_pytorch_fe01_a01/artifacts/fe01_a01/backend_fix/` 上传：

1. `fe01_a01_backend_fix_source.tar.gz`
2. `fe01_a01_backend_fix_source.tar.gz.sha256`
3. `deploy_a01_backend_fix.sh`

三者放入超算的**原 FE01 代码目录**，与已有 `.a01_releases/` 同级。
该实际路径已经填写在私有部署脚本里。无需再次上传 HDF5、大权重或旧结果。
新脚本读取 `.a01_releases/753702a7b4f7/a01.private.env`，保留已工作的
zb、module、partition、资源和数据路径，只建立新 release、新 batch、新输出根。
不覆盖旧发布版，也不复用或重签旧门控。

## 4. 超算登录节点：部署和第一项作业

在上传的文件夹执行：

```bash
bash deploy_a01_backend_fix.sh
```

这一步只部署、校验和检查文件，不提交作业。然后进入脚本打印的**新 release 目录**，执行：

```bash
unset A01_ENV_FILE
bash scripts/fe01_a01/submit.sh environment
```

`submit.sh` 是可直接执行的提交脚本，内部调用 sbatch；计算通过 `job.sbatch` 在分配的节点进行。
不要在登录节点直接运行 `run.py` 或训练 Python。

**environment 成功后**，再单独执行：

```bash
bash scripts/fe01_a01/submit.sh audit 0-4
```

这里重新检查五项，因为源码 SHA 已改变，旧版三个 AUDIT_PASS 不能代替新版门控。
下一行只在五项 audit 均成功后执行：

```bash
bash scripts/fe01_a01/submit.sh diagnostics 0-4
```

## 5. 后续也一次提交一个阶段

| 必须先完成 | 登录节点提交命令 | 资源 |
|---|---|---|
| 对应五项 diagnostics PASS | `bash scripts/fe01_a01/submit.sh pilot 0-4` | 每项1节点1卡 |
| 对应五项 pilot PASS | `bash scripts/fe01_a01/submit.sh train 0-4` | 每项4节点16卡，最多3项并行 |
| 对应训练完整12轮 | `bash scripts/fe01_a01/submit.sh eval 0-3` | 每项1节点1卡 |
| 索引3的ON评价与索引4训练完成 | `bash scripts/fe01_a01/submit.sh eval 4` | 1节点1卡 |
| 需要汇总当前证据 | `bash scripts/fe01_a01/submit.sh pack` | 1节点1卡 |

索引固定为0 TEAM OFF、1 PhaseNet OFF、2 EQT OFF、3 DiTing ON、4 DiTing OFF。
若只想先推进已成功家族，可用单索引，如 `train 1`；其对应门控都必须通过。
`0-2` 是三个旧 FE01 ON 可复用家族的 OFF，不会重训它们的 ON。
DiTing 3/4 必须匹配初始化/采样，RT55/RT61仍不作为它们的 ON 对照。
提交器只检查、不自动补交缺失阶段，也不自动等待或串联训练。

查看新结果根目录：

```bash
source a01.private.env
printf '%s\n' "$A01_OUTPUT_ROOT"
```

Slurm日志在该目录的 `slurm/`；audit门控在 `audits/<run_id>/protocol.lock.json`。
出现 `A01_PREFLIGHT_BLOCKED` 时，查看错误指向的前置文件；该次没有提交任何作业。
源码变更或非空失败目录的重试使用新 batch，保留旧目录。不要删除 lock、强改状态或改 SHA。
全量评价继续锁定 validation；源头预处理仍为 `UPSTREAM_CAUSALITY_UNKNOWN`。
