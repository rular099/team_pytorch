# FE01-A01 实际失败报告与本地修复核验

仓库 `rular099/team_pytorch`，分支 `exp/fe01-a01-absolute-amplitude-ablation`。
本轮修改前 HEAD 与失败超算源码均为 `753702a7b4f727d31c5ebb14f54f7f3f0e528e55`。
研究任务基线仍为 `ba4fa7740d9fde5fde87a7b0ae0397037209baf4`。

## 已核验的实际证据

输入为用户下载的 `fe01_a01_audit_reports_20261007T153854_116671.tar.gz` 和 `.sha256`。
实算 SHA 与声明一致：`5ec24de8ba5245ca87189f01c5ef30241ddd734983db81c58c5417ea989bf124`。
包内60个普通文件，成员 SHA 在 [archive_manifest.json](archive_manifest.json)。
原文件只读；公开副本替换私有路径，cohort数组压缩为数量与内容SHA，并标DISPLAY_ONLY。
这些展示副本不能执行，也不能用来重新签发任何门控；原文件SHA仍由archive manifest记录。
超算源码身份为 `ef82d183e8163a040571636e10cc46cca3b4c74cccf4f9807d310b801c31edae`，
冻结原115文件为 `3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`。

| 阶段 | 用户实际提交的 Job | 可访问证据 |
|---|---:|---|
| environment | 29397740 | PASS，torch1.13.1、ROCm5.4、Python3.9、DCU可用 |
| audit | 29397744 | 索引1/3/4有AUDIT_PASS；0/2仅updated_state失败 |
| diagnostics | 29397945 | 1日志到78/78且无异常；3/4前缀检验失败；0/2缺audit门控 |
| pilot | 29398636 | 1无异常记录；0/2/3/4缺diagnostics.json |
| train | 29399144 | 1在SeisBench导入阶段出现JSONDecodeError；其余缺audit/pilot门控 |

没有 sacct 记录；diagnostics/pilot JSON 未包含在这个收集包中。
因此 PhaseNet 相应阶段只能标 `LOG_COMPLETED/NO_JSON` 与 `NO_EXCEPTION/NO_JSON`，
不能据此认证完整 PASS；正式 train 无可确认成功完成的 epoch。
没有本批评价指标，也不能把导入失败当作 OFF 的性能证据。

### TEAM/EQT：已核验的更新差异

三组均严格加载相同实际 epoch0 状态，前向与 loss 都通过。固定小批更新控制如下；
它使用合成标签，不是 validation 性能分数。

| 模型 | eval前向最大绝对差 | 梯度最大绝对差 | 更新后最大绝对差 | 超出既有容差的元素 |
|---|---:|---:|---:|---:|
| TEAM | 4.6566e-9 | 2.3842e-7 | 2.1723e-6 | 11，分布在3个权重张量 |
| PhaseNet | 5.9605e-8 | 1.1921e-7 | 4.0233e-7 | 0 |
| EQT | 5.9605e-8 | 1.1921e-7 | 1.9073e-6 | 5，全部在attention in_proj_weight |

原比较为 atol1e-6、rtol1e-5逐元素条件。完整失败参数、元素计数与相对容差比例见
[hpc_ON_summary.json](hpc_ON_summary.json)；原结构报告的脱敏副本在 `hpc_redacted/audits/`。
**推断**：这种差异与后端 FP32 归约差经 Adam 更新放大相容，但包中没有逐元素梯度
数组或同模型重复更新对照，尚不足以直接证明全部差异都是舍入。

新版不提高参数更新容差。用同一份真实 epoch0 状态与实际输入，在计算节点的单线程
CPU 严格核对 eval/train前向、loss、梯度和完整更新后状态SHA；任何一项不精确都失败。
DCU 的初始化与冻结 trained ON 前向另设数值门控，容差保持1e-6/1e-5。
这种验收认证结构实现和 DCU 前向兼容，不认证 DCU 上两次 Adam 更新逐位相同。
正式训练继续使用原 DCU/DDP、Adam、clip、cosine、loss 和12轮/2544更新。

### DiTing：旧前缀检验混入重复前向差异

索引3/4旧检验记录的 received/station mask、mean、variance、normalized peak、完整
normalized tensor、raw amplitude、duration与delivered amplitude最大差均为0。
encoder/adapter分别差3.8147e-6与1.9073e-6，MDN分别差5.9605e-8与1.8626e-9，
概率差为0；见 [hpc_future_errors.json](hpc_future_errors.json)。
旧判定将任何输出绝对差超过1e-6都称为泄漏，未分开检验入模信息与后端重现误差。

新版要求完整六个模型输入和全部前缀统计精确相同，选站也相同；重复前向重放RNG与
审计backend控制。encoder/MDN/概率用既有1e-6/1e-5容差。输入发生变化、非有限输出
或输出超差都失败并先保存JSON。旧日志没有完整原始输入数组，不能倒推新版一定PASS。
HDF之前的离线滤波/重采样仍未认证：`UPSTREAM_CAUSALITY_UNKNOWN`。

### PhaseNet：实际训练导入竞争

`train_29399144_1.err` 在多个rank导入SeisBench的 `__init__.py`、读取缓存
`config.json` 时出现 `JSONDecodeError: Expecting value: line 1 column 1`。
旧启动器让16个rank共享首次创建的缓存目录；该首次创建流程不是原子写入。
新版rank启动器在Python导入前设置 `.cache/<job_id>/rank<rank>/` 的独立缓存。
不修改权重路径或安装包，不改变训练样本、参数或loss。
其它任务失败原因在 [hpc_error_summary.json](hpc_error_summary.json)，无须重复提交
已缺前置门控的多节点训练。新版登录节点标准库预检会在 sbatch 前停止这类提交。

## 本地验证：与真实超算结果分开

[pytest_output.txt](pytest_output.txt)：70 passed，1个timm弃用警告，21.87秒。
执行的是全部A01聚焦测试，以及review runner、统计、RT61几何、原生窗口和时间采样回归。

`local_synthetic/` 是实际本地测试输出：TEAM使用scratch前端，PhaseNet/EQT使用已存在的
真实离线注册encoder；下游宽度20、合成HDF与标签。不能把它们当作生产Japan结果。

- 三个家族 CPU 对照的前向、loss、梯度最大差都为0，更新后的完整状态SHA相同。
- 故意把一个更新权重移动一个浮点末位，CPU精确门控失败，即使数值容差门控通过。
- 故意向完整模型输入注入变化，前缀门控失败，并保留失败JSON。
- 独立控制允许相对幅值较小的encoder末位差；输入和统计仍必须精确相同。
- 16个本地子进程并发写/读各自rank缓存配置，路径互异；这不是实际DCU/DDP训练。
- 假sbatch测试：缺失、过期和DiTing不匹配门控都不提交；已配套单项仅提交指定阶段。
- Python3.9语法、编译/import、bash语法、115文件来源与原工作区对比另见
  `compatibility.json` 与随交付包给出的验证结果。

## 仍待用户手动超算核验

新 release 的 environment、五项 audit/diagnostics/pilot、正式训练、validation评价均
`NOT_SUBMITTED`。生产CPU精确更新、DCU前向、真实DiTing重复前向和DDP缓存修复仍需执行。
没有自动sbatch/srun、正式训练、完整评价、test访问、权重下载或环境重装。

上传与命令见 [新部署说明](../../docs/ai/FE01_A01_BACKEND_FIX.md)。
先部署并提交 environment，成功后再 audit；不要一次复制全部阶段命令。
源码SHA变化要求在新输出根重新认证，旧锁/失败目录都保持原样。

ChatGPT审阅应重点检查：CPU精确更新控制是否覆盖真实初始化与完整状态；独立DCU前向
门控是否仍严格；未来扰动是否保持完整输入相同；rank缓存是否在Python首次导入前
设置；预检是否在sbatch之前；训练模型/loss/预算与原115文件是否保持一致。
不能将本地合成PASS写成新生产audit通过，也不能把单seed或未知上游因果性写成最终结论。
