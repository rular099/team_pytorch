# FE01 V4 TEAM model_smoke 单项受控重试（2026-11-02）

**2026-11-08 后续真实运行：** Job33156666_3的日志已出现`TEAM_MODEL_SMOKE_PASS_WITH_OPERATIONAL_REVISION`。
TEAM ep11/seed42在两个validation smoke事件、normal/random及1/3/5/10/20秒共20个请求上，
严格deterministic algorithms启用、warn_only关闭；同输入feature及完整MDN检查全部通过。
日志另有48条future NaN/Inf/pulse feature检查，全部通过；feature最大差7.62939453125e-6，仍低于原1e-5门。
这些是小样本模型门结果，不是全validation性能或逐bit一致性证明。

PASS打印在attempt封存/原子发布之前，目前仅取得out/err，尚未独立核验`stages/model_smoke/3/SEAL.json`。
下一步用原50672e1的status确认四组model_smoke均为`COMPLETE_SEAL_VERIFIED`，再手动提交`replay_pilot`。
继续使用既有c66db619e99a运营包、原主程序/run/env；本次仅记录结果，不重新封包或重提TEAM。
详见[本批逐请求日志核验](../../reports/fe01_v4_team_smoke_repeatability_20261108/job_33156666_log_review.json)。

**2026-11-08 最新设备结果与修复：** Job33116200_3 已实际运行6e2ddb5小包，DCU可见、TEAM selected ep11身份通过，
在第一个normal/1秒请求的同输入重复检查失败，encoder feature最大差为1.52587890625e-5，原门为1e-5。
该请求来自validation，事件japan_2006.hdf5/20060913043100；日志只覆盖这一失败请求，不能当成完整split成绩。
future NaN/Inf/pulse门尚未运行，不能将这次失败称作未来信息泄漏。

原A01 paired_execution没有设置全局deterministic algorithms；PyTorch 1.13.1的
[ROCm BLAS源码](https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/cuda/CublasHandlePool.cpp#L55)
按此全局标志决定是否允许rocBLAS atomics，设置cudnn.deterministic并不设置该标志。
新小包仅在TEAM smoke audit内加`torch.use_deterministic_algorithms(True, warn_only=False)`，
成功/异常均恢复原enabled/warn_only；原卷积设置和RNG控制保留，生产推理/latency不改。
同时记录六项backend设置并在stdout打印feature与完整MDN字段的最大差值。
这是对可确认的审计控制缺口的修复，实际设备根因及新包能否通过仍待DCU复跑；不放宽绝对阈值、不增加relative tolerance或CPU通行路径。
出现无确定性实现的算子时继续失败，保留原异常和诊断，不转成warning或跳过原门。

当前请使用本轮交付的新commit小包；6e2ddb5及更早包保留为历史来源，不再作为本次重试入口。
仅重提TEAM task3，原50672e1主程序、run/env及其他成功项保持。新包重提本轮尚未提交。
设备日志SHA、事实/推断、32项专项测试和验收边界见
[本批验证记录](../../reports/fe01_v4_team_smoke_repeatability_20261108/verification.json)。

**2026-11-05 提交入口更新：** 原fe358c3小包在用户无参数调用时出现 `args[@]: unbound variable`，
失败发生在Python/sbatch之前。新版移除参数数组，直接用 `"$@"` 转交原参数，并给私有变量加前缀。
支持不带参数、显式run ID、`--env PATH`及`--env=PATH`，保留带空格的路径；缺少env值会明确报错。
请使用最新小包的相同提交入口；不要手改旧包，以免源码SHA不匹配。原run/env及TEAM检查保持不变。

用户提供 `model_smoke_32163754_3.err/.out`。原 V4 release 为
`50672e131d876a604432b283162d5b0189d40cf0`，运行目录为
`outputs/fe01_on_eval_v4/on_v4_20261025_inventory_fix`。

## 已确认事实与尚未确认的原因

作业在分配的 srun step 内报告 DCU 可见，TEAM selected epoch11 的 `INVENTORY_PASS` 已输出。
失败为 `Future changed station encoder features`。按原检查顺序，在该次特征比较之前，
实际前向输入的完整 tensor hash、归一化与幅值统计 hash、特征列表长度的检查均已通过。
此前的 query 顺序、分块和 cache/grid-append 检查也已通过该次调用。

原门对特征使用绝对阈值1e-5，但没有同输入空白重复对照，也没有在 smoke 内设置已有 A01 的
`paired_execution` 控制；失败时没有保存实际差值、具体事件/时刻和 future 变体。
因此**不能仅凭这份日志断定未来数据泄漏或断定设备数值波动**。
本次做受控重试与诊断，实际原因和成功状态仍须超算结果证实。

## 重试做什么

小包位于 `scripts/fe01_v4_team_smoke_fix/`，独立于原 V4 execution tree。
旧 V4 源码、plan/env、checkpoint、旧 attempt/seal 与其他三个模型任务均保持原状。

- 只允许原50672e1源码/执行SHA和 `model_smoke` 的 TEAM task3；核 inventory/covariates 完整前置门。
- 使用相同原 runner 的**同一 bytecode**，在私有 globals 副本中仅绑定 audit callback。
  不改原模块、类或全局绑定；原模型加载、事件选择、预处理、回放与完整原 smoke 检查照常执行。
- audit 内复用原 `fe01_a01.equivalence.paired_execution`：确定性卷积、关闭 benchmark/TF32、保存和恢复 RNG。
  设置只用于 smoke 检查，并在成功或失败时恢复；生产推理与 latency stage 的默认值不变。
- 先对同一事件/请求做两次无 future 修改的前向，比较完整MDN和encoder特征。
  同输入仍超原阈值则报 `SAME_INPUT_REPEATABILITY_FAILED`，不误标为未来扰动影响。
- 原 future NaN/Inf/pulse、query permutation/chunk、cache、label、最后合法采样和 legacy overlap 门均保留。
  **原绝对阈值1e-5不放宽，不增加相对容差通行，不以CPU结果替代DCU检查。**
- 增补逐事件/时刻的 blank、baseline、三个 future 特征差值/shape/dtype/hash/幅值与恢复状态。
  原检查仍失败时继续失败，不写成功 seal。

原执行SHA继续认证未修改的V4代码；新增运营源码SHA/commit另写入 `operational_execution.json` 和 seal。
旧原生status完整校验这些新增文件，后续父seal SHA也包含本次运营证据；不把原执行SHA说成完整新运行身份。

## 在超算执行

复制本轮小包 `.tar.gz` 与同名 `.sha256` 到项目根目录，按交付消息中的实际包名校验、解压。
不需要再次传132MiB完整V4源码包。以交付消息中的 `FIX` 指向小包解压目录，执行：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
conda activate zb
bash "$FIX/scripts/fe01_v4_team_smoke_fix/submit_team_smoke.sh"
```

这条入口显式 `sbatch --array=3%1`，资源沿用原私有环境。DCU检查在真正分配的srun step内。
只提交这一项，不自动启动其他阶段；成功和在跑项不重复提交，scheduler UNKNOWN时拒绝重提。

继续使用原运行的状态脚本：

```bash
bash on_eval_v4_50672e131d87/scripts/fe01_a01_compare/status.sh \
  --env "$PWD/on_eval_v4_inventory_fix.private.env" \
  --run-id on_v4_20261025_inventory_fix
```

新 stdout 会打印准确 job ID 与日志文件。位置仍为：

```text
outputs/fe01_on_eval_v4/on_v4_20261025_inventory_fix/slurm/model_smoke_新JOBID_3.err
outputs/fe01_on_eval_v4/on_v4_20261025_inventory_fix/slurm/model_smoke_新JOBID_3.out
```

失败尝试保存在同运行下 `attempts/model_smoke/3_随机后缀/`，stdout打印该准确路径。
其中 `smoke_diagnostics/请求HASH/` 包含：

```text
00_scope.json / 01_backend.json / 02_same_input_repeat.json
03_baseline.json / 04_future_NaN.json / 05_future_Inf.json / 06_future_pulse.json
99_pass.json 或 99_failure.json
```

若仍失败，请下载新err/out及该attempt内小型诊断JSON。已成功结果仍原子发布至 `stages/model_smoke/3/`。
四个model_smoke均完整通过后，再用原 `submit_stage.sh --stage replay_pilot`，保持原run/env。

## 本地验证边界

测试使用合成特征/MDN、原A01控制器的模拟backend和fake scheduler；
覆盖原runner bytecode/全局绑定不变、严格阈值、NaN/Inf/缺特征拒绝、blank失败不进入future门、
原future失败继续失败、成功/失败恢复backend、只提交3、去重/UNKNOWN拒绝、运营源码防篡改及seal兼容。
这些证明入口与诊断逻辑，不证明Japan模型或DCU数值已经通过。
本地缺生产checkpoint/Japan波形/DCU，修复重提 **NOT_SUBMITTED**。
2026-11-05 新增5项真实shell入口测试，专项测试合计22项通过；解释器用记录参数的替身，未提交真实作业。
