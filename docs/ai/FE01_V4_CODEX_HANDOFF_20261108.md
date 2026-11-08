# FE01 ON V4：跨服务器 Codex 接手文档（2026-11-08）

**同日接手后的状态更新：** 已取得Job33116200_3失败日志并新增严格确定性审计控制，详见
[当前TEAM重试说明](FE01_V4_TEAM_SMOKE_FIX.md)。下文是本次更新前的历史交接快照；
其中6e2ddb5包现已被本轮新交付包替代，新包设备结果仍待用户手动提交后核验。

**当前交接点：TEAM model_smoke 的单项受控重试包已交付，最新包修复了无参数提交错误；尚未收到该新版作业结果。先检查当前超算状态，再决定补交或继续 pilot。**

本文整理截至本对话可见证据，不代表已经远程查询了超算。仓库里的早期 `NOT_SUBMITTED` 是当时交付状态，不能据此否认用户后来已经提交的作业；同样不能把本地测试通过当作超算通过。

## 1. 从哪个仓库、分支和版本接手

| 项目 | 准确身份 |
| --- | --- |
| 仓库 | `rular099/team_pytorch` |
| 本方活跃分支 | `exp/fe01-on-warning-map-evaluation` |
| 本文整理前本地/远端核对一致的实施 HEAD | `6e2ddb5dc0a52ff663099d0f0410a637261406bb` |
| 本机工作区 | `/home/zhangb/work/people/zhangbei/team_claude/team_pytorch_fe01_on_eval` |
| 本机整理前工作区状态 | clean |
| V4 原任务 ID | `20261021-fe01-on-integrated-warning-map-v4` |
| 本次文档交接 ID | `20261108-fe01-v4-cross-server-codex-handoff` |

本次交接提交只更新文档。本文所在 commit 是交接版本，`6e2ddb5...` 是交接依据的实施版本；不要为了让文档 HEAD 与实施版本相同而回退代码。
另一台服务器的绝对路径、Python环境和下载文件可能不同，本机路径不能当成对方已有资源。
若另一台已经有本文未覆盖的新commit或超算结果，先比较并保留它们，更新交接事实，不机械回退到本快照。
本文覆盖本方持续跟进的FE01/A01/V4；RT55/RT56背景见项目上下文。FE02、V01和其他geometry工作区的最新进度本轮未复核，
不能把本文作为它们已完成或需要重做的依据，也不要把这些独立工作区的文件混进本任务提交。

另一台 Codex 先报告自己的 `git status --short`、branch、HEAD，并核对是否包含本分支。
已有改动不 reset/clean/stash，不强推覆盖其他 Codex 工作。可从现有仓库创建独立工作区读取远端：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git fetch origin exp/fe01-on-warning-map-evaluation
git worktree add -b handoff/fe01-on-20261108 ../team_pytorch_fe01_on_handoff FETCH_HEAD
```

最后一条仅在该分支/目录尚不存在时执行；已存在则检查并使用原工作区。修改前确认其他 Codex 是否也在改同一文件，提交前再次核对远端，使用正常合并流程。

阅读顺序：`AGENTS.md` → [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) → [README.md](README.md) → 根目录 [SESSION_SUMMARY.md](../../SESSION_SUMMARY.md) → 本文及V4合同。
`PROJECT_CONTEXT.md` 是2026-09-02的RT55/RT56背景快照；FE01当前阶段由本交接、实际产物及后续更新说明，不使用旧快照里的活跃分支覆盖本分支。

## 2. 当前任务与固定边界

用户要完成 FE01 **ON 综合评价 V4**，延续已有训练/评价，不是重新训练实验。
权威任务书与机器合同已入Git：

- [V4完整任务书](../../configs/fe01_a01_compare/DITING_ON_INTEGRATED_EVALUATION_HANDOFF_V4_20261021.md)
- [V4研究合同](../../configs/fe01_a01_compare/DITING_ON_V4_RESEARCH_CONTRACT.json)
- [协议](FE01_ON_EVAL_PROTOCOL.md)、[产物说明](FE01_ON_EVAL_ARTIFACTS.md)

固定 validation、seed42、全部 amplitude ON；selected 为 DiTing ep12、PhaseNet ep12、EQT ep10、TEAM ep11。
已保存的 epoch12 只作独立稳健性比较。不能重训/续训、新增OFF或多seed、重选checkpoint、调sigma、打开held-out test。
用户明确 RT55/RT61 已有训练结果，不要重新训练它们。

保留 RT55 默认加载/推理，旧115认证文件、`fe01_a01/`、`fe01_review/`、旧config/lock/seal不改。
所有新增执行身份单独记录；TEAM是可训练scratch encoder，不能重新要求训练后encoder等于初态。
CSV使用C解析器 `float_precision='round_trip'`，ID保留字符串，不通过放宽容差或取交集绕过人口门。

正式超算任务由用户手动提交；Codex可实现、测试、封包和给实际Bash/sbatch入口，不能自动执行真实sbatch/srun。
用户希望快速推进并保留成功项；请给可运行的提交脚本及准确命令，不把命令藏在TXT里，也不让用户手选数组下标。

## 3. 版本关系：最容易接错的地方

| commit | 内容与当前用途 |
| --- | --- |
| `29ffb20ba0effc0c345a0329ba3f8c658dc8cef0` | A01完整旧结果整理后的V4基点，来源分支 `exp/fe01-a01-absolute-amplitude-ablation` |
| `a5c677c06ae67ed5c47b0cab0905908e1275cf81` | V4首次实现/CPU分析/脚本；旧run `on_v4_20261022` 的来源，inventory曾失败 |
| `50672e131d876a604432b283162d5b0189d40cf0` | 修复manifest路径并迁移已完成CPU结果；**当前超算冻结主程序版本** |
| `fe358c38f061a06d1b1e66c57574a9f44b3c0fc7` | TEAM受控单项smoke小包初版；其无参数提交入口有问题，已被下一行替代 |
| `6e2ddb5dc0a52ff663099d0f0410a637261406bb` | **当前有效的TEAM单项小包**：修复空参数/私有env同名变量覆盖，22项专项测试通过 |

**Git最新实施版本6e2ddb5、超算主程序50672e1、TEAM运营小包6e2ddb5是不同角色。**
小包加载现有50672e1主程序，沿用它的plan/env；新增运营SHA/commit另外写入结果和seal。
不能把新Git HEAD全量打包覆盖 `on_eval_v4_50672e131d87`，也不能手改marker使新代码冒充旧版本。
即使代码文件相同，整个 `source_identity` 还包括result_commit，换主程序commit可能触发旧plan身份拒绝。

当前冻结主程序 execution SHA：

```text
b9dd0368adc7041bb0c1e36a6899d49014ac9545ae7f44850274099bab6dc15c
```

22项测试验证了新提交入口；不是22次真实DCU/model测试。最新原V4源码与96份本地冻结核心文件校验均保持一致。
另外65份dtbench依赖需计算节点原vendor与完整运行时门校验，不能称为本地全部115文件已验证。

## 4. 已经实际完成的工作

1. A01用户下载的旧 `outputs/review.tar.gz` 已完整整理、校验并同步轻量事实资料；入口是
   [A01中立报告](../../reports/fe01_a01_hpc_results_20261021/README.md)与
   [独立分析提示词](../../reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md)。
2. `fe01_a01_compare/` 已实现四ON旧预测比较、事件簇配对区间、固定分层/四交互、真实混合68%宽度；
   本地真实保存预测的CPU分析实际跑完，未读生产权重/波形/test。
3. 已实现因果SNR/mask/UTC/origin和首次真实越阈提取、固定query逐秒station/grid、TTUM/shadow、
   同cutoff缓存、分项时延/FIFO、全量动态图/HTML、封包、状态与失败恢复；生产实测完成范围见下一节，不能以“实现了”代替“跑完了”。
4. 小型合成数据实际跑通20帧动态图/动画/HTML与TTUM/shadow，明确标SYNTHETIC，不能作为Japan成绩。

真实CPU结果入口：[V4报告](../../reports/fe01_on_eval_20261022/README.md)、
[ChatGPT复核提示词](../../reports/fe01_on_eval_20261022/CHATGPT_REVIEW_PROMPT.md)。
四模型各1310个validation事件、194265条旧snapshot请求，noninput139440条。
selected、normal/random及1/3/5/10/20十cell等权noninput MAE，单位dex=`log10(m/s²)`，
2000次事件簇配对重采样、seed20261007，DiTing减对照：

| 对照 | MAE差 | 95%固定模型事件区间 |
| --- | ---: | --- |
| PhaseNet ep12 | -0.002838 | [-0.005120, -0.000580] |
| EQT ep10 | -0.013548 | [-0.016358, -0.010949] |
| TEAM ep11 | -0.022299 | [-0.026050, -0.018831] |

原表见 [paired_effects.csv](../../reports/fe01_on_eval_20261022/existing/paired_effects.csv)。
all-target、K1、untriggered及部分空间field/shape指标并不支持普遍优势；第一秒贡献约70.41%只是有符号代数贡献。
这些是单seed、validation选轮、固定模型的旧snapshot结果，不是训练seed区间，也不能推出新TTUM更早或shadow更小。
新fixed-query人口与旧194265请求分开认证，完整20秒逐帧输出与旧五时刻不能混名。

## 5. 截至交接的超算证据

| 阶段/事项 | 本方真正拿到的证据 | 接手时如何处理 |
| --- | --- | --- |
| 旧inventory Job31089276_0 | 实际err/out：`Pretrained manifest SHA differs` | 已由50672e1修复；不从旧run重新开始 |
| covariates | 用户明确回报 `{"COMPLETE_SEAL_VERIFIED":41}` | 保留，当前状态仍以完整seal检查为准 |
| inventory的修复运行 | 用户曾回报RUNNING；之后Job32163754_3进入smoke，按原代码其inventory/covariates前置门已通过 | 不把较早RUNNING当现在状态；先status |
| TEAM model_smoke Job32163754_3 | 实际err/out：DCU可见、TEAM ep11 `INVENTORY_PASS`，随后future encoder feature门失败 | 当前设备重试的主要待验证项 |
| 其他3个model_smoke | 没有下载到逐项成功证据 | 不推断已完成，查看status |
| 旧小包无参数提交 | 用户回报 `args[@]: unbound variable`，发生在调用Python/sbatch前 | 已用6e2ddb5修复，不重复推荐旧fe358c3入口 |
| 最新6e2ddb5小包重提 | 已交付、封包解包/无参数入口本地验证通过；**尚未收到用户新的job号/日志/成功反馈** | 先status，未成功再用当前小包补3号 |
| compare_existing/crossing_truth正式阶段 | 当前完整状态未提供 | 本地CPU旧预测分析已完成不等于HPC这些阶段已完成 |
| pilot、全量replay/grid、latency、analyze/render/pack | 本方未提交，未收到对应完成证据 | 当前状态待确认；按依赖推进，不宣称已完成或断言用户从未提交 |

原始日志在本机workspace外层的 `outputs/fe01_v4/inventory_31089276_0.err/.out` 与
`model_smoke_32163754_3.err/.out`，这些原件不在Git；输入SHA及摘要已在对应修复报告中。
不能假设另一台服务器也有这些本地文件。

## 6. 当前有效的超算位置和先做的只读检查

```text
项目根目录：/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
conda：zb
主程序：项目根目录/on_eval_v4_50672e131d87
当前私有环境：项目根目录/on_eval_v4_inventory_fix.private.env
当前run：outputs/fe01_on_eval_v4/on_v4_20261025_inventory_fix
旧run：outputs/fe01_on_eval_v4/on_v4_20261022（保留原状）
A01来源session：outputs/a01_eval_20261014T102355__51fzm5p
当前TEAM小包目录：项目根目录/v4_team_smoke_fix_6e2ddb5dc0a5
```

用户在超算登录节点首先执行（Codex不要自行提交作业）：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
conda activate zb
S="$PWD/on_eval_v4_50672e131d87/scripts/fe01_a01_compare"
ONENV="$PWD/on_eval_v4_inventory_fix.private.env"
ONRUN=on_v4_20261025_inventory_fix
bash "$S/status.sh" --env "$ONENV" --run-id "$ONRUN"
```

status/prepare不加载模型也不提交。只有 `COMPLETE_SEAL_VERIFIED` 算完成；作业号、目录非空、队列消失、UNKNOWN都不是成功。
若3号已成功，保留并继续pilot；若在跑/排队，等待；若FAILED等明确失败且需要重试，使用下一节。
UNKNOWN先查原job/err/out，INVALID_SEAL先核文件/来源，不删除seal、不重签绕过、不贸然重提。

## 7. 当前TEAM单项重提命令

当前有效的包：

```text
fe01_v4_team_smoke_fix_6e2ddb5dc0a5.tar.gz
fe01_v4_team_smoke_fix_6e2ddb5dc0a5.tar.gz.sha256
首次交付包大小：20155 bytes
首次交付包SHA256：ced69bf8b5f58e5ac0df13fd658738b7ad83eecca7f00788465f11a2e80642e6
```

包尚未解压时，在项目根目录校验并解压：

```bash
sha256sum -c fe01_v4_team_smoke_fix_6e2ddb5dc0a5.tar.gz.sha256
tar --keep-old-files --no-same-owner -xzf fe01_v4_team_smoke_fix_6e2ddb5dc0a5.tar.gz
```

若该目录已经存在且通过其marker校验，直接使用，勿重复解压覆盖。用户手动提交：

```bash
bash v4_team_smoke_fix_6e2ddb5dc0a5/scripts/fe01_v4_team_smoke_fix/submit_team_smoke.sh
```

脚本明确只 `sbatch --array=3%1`，保留其他任务；成功/在跑会跳过，UNKNOWN拒绝重复提交。
其内部实际srun step使用原已工作的DTK/MPI/conda配置，不重装Torch/ROCm，不在登录节点要求DCU可见。

该版本修复 `args[@]` 空参数/环境覆盖问题：不使用通用args数组，直接转交 `"$@"`，支持无参数、显式run ID、
`--env PATH`、`--env=PATH`和空格路径。不要手改旧包文件，否则运营源码hash会不符。

## 8. TEAM受控重试的科学边界及失败取证

Job32163754_3失败前，实际forward输入和归一化统计hash门已通过；旧日志没有实际feature差值。
不能武断宣布“未来信息泄漏已证实”或“只是DCU浮点误差”。
运营入口保持原runner的同一bytecode，在私有namespace副本中仅绑定audit callback，不修改原模块全局/类。
audit局部复用既有 `fe01_a01.equivalence.paired_execution`，保存并恢复backend/RNG设置。
先做完全同输入的两次前向，对encoder及完整MDN仍使用原绝对1e-5阈值；
全部原future NaN/Inf/pulse、query顺序/分块、cache、label、合法最后采样和legacy overlap门保留。
不增加relative tolerance通行，不跳过失败门，不以CPU模型结果代替DCU门，不把audit设置当生产latency默认值。

原V4 source SHA认证原代码，新增运行身份写入 `operational_execution.json` 与 `SEAL.json`；
下游父seal SHA包含这些证据。完整实现/测试范围见
[TEAM重试说明](FE01_V4_TEAM_SMOKE_FIX.md)、[22项入口修复报告](../../reports/fe01_v4_team_smoke_shell_fix_20261105/README.md)。

若仍失败，下载新job的 `.err/.out`，以及其stdout打印的准确attempt目录中的：

```text
FAILURE.json
ATTEMPT.json
operational_execution.json
runtime_environment.json
smoke_diagnostics/请求HASH/00_scope.json
smoke_diagnostics/请求HASH/01_backend.json
smoke_diagnostics/请求HASH/02_same_input_repeat.json
smoke_diagnostics/请求HASH/03_baseline.json
smoke_diagnostics/请求HASH/04_future_NaN.json（执行到时才存在）
smoke_diagnostics/请求HASH/05_future_Inf.json
smoke_diagnostics/请求HASH/06_future_pulse.json
smoke_diagnostics/请求HASH/99_failure.json 或 99_pass.json
```

日志仍在 `当前run/slurm/model_smoke_新JOBID_3.err/.out`；失败attempt在 `当前run/attempts/model_smoke/3_随机后缀/`。
成功结果才原子发布至 `当前run/stages/model_smoke/3/`。
先读blank repeat：若它已经失败，不能把随后尚未执行的future控制当成失败根因。
若blank通过而future失败，比较对应特征/输入/normalization/backend证据，不改阈值掩盖问题。

## 9. smoke全过后怎么继续

以下命令均由用户在超算登录节点逐步手动执行，沿用第6节的 `S/ONENV/ONRUN`。
跨shell需重新设置变量。每个代码块的前置条件满足后才执行，不能整页一次粘贴连跑。

四个model_smoke全部完整通过后：

```bash
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage replay_pilot
```

四个pilot通过后，审阅 `stages/replay_pilot/*/worker.json` 的实际耗时/显存/文件大小/支持状态，再提交：

```bash
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage replay_stations
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage replay_grid --geometry normal
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage latency
```

默认块32事件，station两种geometry共328项，normal grid164项；每项加载对应模型一次，单节点1DCU。
默认并发8，可在submit前设 `ON_EVAL_MAX_CONCURRENT=16`；它是**每个数组的并发任务数**，多个数组并发上限会相加，
不是每任务16节点。用户曾表示约16节点可用，仍须按实际配额及pilot资源记录设置；prepare后不改冻结block划分。

station、normal grid、crossing_truth及covariates全部完整通过：

```bash
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage analyze
```

analyze通过且normal grid完整后：

```bash
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage render --geometry normal
```

compare_existing、analyze、render_normal、latency全部完整通过后：

```bash
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage pack
```

pack成功后下载 `当前run/stages/pack/0/fe01_on_eval_v4_review.tar.gz` 与同名 `.sha256`。
全量图库入口 `stages/pack/0/full_visuals_index.html`，大图源/数值/视频按索引留超算；Git只保存轻量资料。
随机geometry网格、origin固定时刻、40/90秒是显式可选stage，不自动追加，不混入主排名。
其他阶段明确失败使用原 `resume_failed.sh --stage 对应阶段`，它跳过成功/在跑项；TEAM当前诊断重提优先使用专用小包。

完整初始流程见[总运行手册](FE01_ON_EVAL_RUNBOOK.md)，但其早期示例run是 `on_v4_20261022`，
**接手当前进度必须保持本交接的50672e1程序及修复run/env**，不要机械复制回旧run或重新prepare。

## 10. 另一台服务器只有Git，没有本机artifacts怎么办

源码、合同、报告和测试都已在本分支；生产权重/HDF5、私有env、原始大CSV和release二进制不在Git。
不要要求用户把超算checkpoint搬到开发服务器。私有env由超算既有配置继承，不能使用报告脱敏的 `PRIVATE_PATH` 当实际路径。

当前小包可从其**准确实施commit**重建，不必从头补V4：

```bash
git worktree add --detach ../fe01_team_smoke_release_6e2ddb5 6e2ddb5dc0a52ff663099d0f0410a637261406bb
cd ../fe01_team_smoke_release_6e2ddb5
python3 -B scripts/fe01_v4_team_smoke_fix/build_bundle.py \
  --output artifacts/fe01_on_eval/team_smoke_release_rebuilt
```

目录已存在时不要重复创建；build要求工作区干净、输出目录全新。这一步仅stdlib/git，不读模型、数据或提交作业。
生成tar、同名sha和SOURCE_RELEASE.json，top directory仍是 `v4_team_smoke_fix_6e2ddb5dc0a5`。
**gzip时间等会使重新封包字节SHA与首次交付不同：使用本次生成的配套sha，不能套用上面的首次包SHA。**
同一代码commit/运营源码hash与同一个archive字节hash是不同身份；保留本次封包信息，不伪造旧摘要。
超算已有正确6e2ddb5目录时无需重传/覆盖。

测试入口：

```bash
python3 -m pytest -q tests/test_fe01_v4_team_smoke_fix.py
bash -n scripts/fe01_v4_team_smoke_fix/submit_team_smoke.sh
```

该专项需pytest/numpy/pandas/scipy及原代码可导入，不需Torch/DCU/生产checkpoint；fake scheduler/解释器不会提交真实作业。
完整旧V4 CPU验证入口为 `scripts/fe01_a01_compare/verify_local.sh`，按实际Python依赖运行。
本机曾使用的Python/pytest缓存路径不是对方必备路径；不要为测试主动重装超算已工作的zb/ROCm环境。

## 11. 后续修改与验收

继续排错时优先保留冻结主程序，用带独立运营身份的新修复入口；源码/科学输入确需改变时制定新run与来源迁移，
不能覆盖旧程序、plan、锁、成功seal或隐藏执行差异。
目前 `scripts/fe01_v4_team_smoke_fix/` 位于原V4 execution scanner之外是有意的运营隔离；独立marker/运营文件hash必须继续校验。
同目录直接编辑会使已封包marker失效，需新提交、新包及准确交付命令。

提交前只暂存本任务文件，核对远端是否已有其他Codex进度，正常推送后用远端完整SHA复核。
新本地验证和真实HPC结论分别报告，结束时按[协作约定](README.md)给完整CODEX-RESULT。

接手后的第一份反馈应包括：实际branch/HEAD/dirty状态、是否包含6e2ddb5修复、最新超算status/TEAM重试job或尚缺什么证据、
下一个明确动作。尚无新job结果时，不从头重训、重跑已完成CPU阶段或宣称TEAM门已经通过。

## 12. 可直接转交给另一台Codex的提示词

```text
请接手 rular099/team_pytorch 的 FE01 ON V4，读取分支
exp/fe01-on-warning-map-evaluation 中的
docs/ai/FE01_V4_CODEX_HANDOFF_20261108.md，并按 AGENTS.md 阅读相关上下文。
实施基线为6e2ddb5dc0a52ff663099d0f0410a637261406bb，交接文档提交在其后。
先报告你实际读取的branch/HEAD/status，保留你已有的更晚进展。
当前任务是跟进TEAM model_smoke单项受控重试及其后的pilot/全量评价。
主程序仍为超算上的50672e1，当前有效运营小包为6e2ddb5，先核对最新status/日志。
尚无新版TEAM重试成功证据，不要假定它已通过，也不要从头重训或重跑成功项。
正式sbatch/srun由用户手动执行；需要算力时给出准确Bash提交脚本和命令。
```
