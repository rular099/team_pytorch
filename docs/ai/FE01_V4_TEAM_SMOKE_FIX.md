# FE01 V4 TEAM model_smoke 单项受控重试（2026-11-02）

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
