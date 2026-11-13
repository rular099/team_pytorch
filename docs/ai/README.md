# ChatGPT Project 与 Codex 协作约定

**最新渲染日志证据（2026-11-13）：** control33793740确认提交render_normal pool33794066，四worker覆盖原41项；
全部逻辑任务各一次exit0，四个最终POOL_SUMMARY齐全。0号stderr前两行为Slurm switch/generic插件版本不兼容报错，
其后该slot的11项仍全部exit0；其余stderr为route及pandas FutureWarning/DtypeWarning，不能把插件报错统称为警告。
见[渲染日志复核](../../reports/fe01_v4_render_logs_20261113/render_33794066_log_review.json)。尚未下载视觉payload/seal，
下一步沿用1cebcdc包手动`submit.sh --stage pack`；计算节点先验证compare_existing/analyze/render_normal/latency全部原seal，
通过才调度单项CPU pack。latency仍无新完成证据，门不通过时传control日志，不绕过门或盲目重提。
pack成功后下载原run的`stages/pack/0/fe01_on_eval_v4_review.tar.gz`及其`.sha256`；全量图源继续保留超算。

**最新分析日志证据（2026-11-12）：** `analyze_pool_33717765_0.out/.err`记录唯一analyze任务exit0及最终POOL_SUMMARY，
两套旧预测policy（selected/epoch12）的三组比较/分层完成标记齐全；stderr仅route及两条pandas FutureWarning。
详见[分析日志复核](../../reports/fe01_v4_analyze_logs_20261112/analyze_33717765_log_review.json)。
未收到分析payload/seal，尚不能报告新性能数值；epoch12日志属于旧预测再分析，不改变新回放selected模型。
下一步沿用1cebcdc包手动`submit.sh --stage render_normal`，计算节点先验证原完整前置seal；
normal渲染仍为原41逻辑块、1310锁定事件，默认最多4个CPU worker，不申请DCU。latency状态仍待pack前核验。

**最新回放日志证据（2026-11-11）：** 用户提供的`runs/slurm0001.zip`覆盖台站pool33474027全部328项、
normal-grid pool33474094全部164项；每项恰有一次exit0，八个worker均打印最终汇总路径。
stderr仅Slurm route、timm/pandas FutureWarning及其源码续行。latency33441392只有启动/身份日志，完成状态未知。
详见[逐任务复核](../../reports/fe01_v4_full_replay_logs_20261111/verification.json)。当前证据为日志执行成功，
尚未独立取得全量payload/seal或准确率指标。下一步沿用原1cebcdc包手动`submit.sh --stage analyze`，
由计算节点先验证原全部前置seal；latency是pack前置，不阻塞analyze。不重跑回放或重新封包。

**最新调度边界（2026-11-10）：** 用户要求登录节点只提交任务，所有Python/散列/seal与状态计算均在计算节点。
当前使用[计算节点调度入口](FE01_ON_EVAL_RUNBOOK.md)，旧submit/status的登录节点Python用法已被替代。
原主程序50672e1、TEAM smoke运营包c66db61及成功结果继续保留；latency Job33441392已提交，本次默认不重提。

更新日期：2026-11-08（基础协作约定始于2026-09-02）

**最新跨服务器接手入口（2026-11-08）：[FE01 ON V4 Codex交接](FE01_V4_CODEX_HANDOFF_20261108.md)。**
当前运营实施版本c66db619e99a；超算主程序仍50672e1。Job33156666_3已提供TEAM模型门PASS日志，
尚待原status完整核验四组smoke的seal；通过后手动进入pilot，保留已有成功项。

**最新实物核验（2026-11-09）：** 已下载并核验四smoke/四pilot共8份seal、333个封存文件，
pilot通过。当前进入手动全量station/normal-grid/latency阶段，见[复核证据](../../reports/fe01_v4_pilot_results_20261109/verification.json)。
继续既有50672e1主程序、c66db61运营包及原run，不因本次报告提交更新执行身份。

**同日后续更新：** 已收到Job33116200_3日志，TEAM同输入重复前向失败，future门尚未执行。
当前修复及后续Job33156666_3的PASS证据见[TEAM smoke操作说明](FE01_V4_TEAM_SMOKE_FIX.md)；
6e2ddb5保留为历史来源，运营包仍用c66db619e99a，无需为结果文档更新重新封包。

## 1. 协作边界

这个流程把 GitHub 作为共享的、可追溯的事实来源，但不假设两个 AI 之间存在自动通信。

- 用户：决定研究目标、实验优先级和是否执行有成本或有风险的操作。
- ChatGPT Project：通过 GitHub 插件/连接器阅读指定分支，负责独立分析、方案审阅、文献化解释和提交前复核。
- Codex：在本地工作区检查真实文件，实施修改，运行可执行的验证，提交并推送 Git 变更。
- 超算：保存大数据、权重、训练日志和正式评估产物；GitHub 通常只保存代码、配置、启动器、测试和轻量文档。

GitHub 连接器是否能写 issue、PR 或代码取决于实际安装项和权限。除非工具明确返回成功，ChatGPT 不应声称已经修改或推送仓库。最稳妥的默认方式是：ChatGPT 生成交接块，用户转交给 Codex；Codex 推送后，ChatGPT 按精确 commit/PR 复核。

## 2. 每轮工作的固定流程

1. ChatGPT 确认 `rular099/team_pytorch`、目标分支和当前 commit。
2. ChatGPT 阅读 `AGENTS.md`、`docs/ai/PROJECT_CONTEXT.md` 和与任务直接相关的代码、配置、测试及 provenance。
3. ChatGPT 区分事实、推断和建议，并输出 `AI-HANDOFF`。
4. 用户把交接块发送给 Codex；若交接会改变实验定义，用户先作决定。
5. Codex 在同一 base commit 上实施；若 HEAD 已变化，先报告差异，不机械套用旧方案。
6. Codex 运行本地验证、提交、推送，并输出 `CODEX-RESULT`。
7. ChatGPT 读取新 commit 或 PR diff，检查实现是否满足验收条件和 RT55 兼容性。

一个对话只处理一个明确产出。架构讨论、实现、超算运行和结果解释可以分别开聊，但都使用同一项目文档和精确 commit。

## 3. 事实来源优先级

发生冲突时按以下顺序处理，并显式报告冲突：

1. 当前分支上的代码、resolved config、测试和 checkpoint/result provenance；
2. `docs/ai/PROJECT_CONTEXT.md` 的带日期快照；
3. `SESSION_SUMMARY.md` 和专题文档；
4. 聊天中的用户补充；
5. 模型记忆或未核验推断。

文件名和输出目录名不是 checkpoint 身份证据。应检查 checkpoint 内 epoch/loss、评估 split、resolved config 和 metrics/NPZ 的来源。

## 4. ChatGPT 交给 Codex 的格式

```text
[AI-HANDOFF]
task_id: YYYYMMDD-short-name
repo: rular099/team_pytorch
branch: <target branch>
base_commit: <full SHA>
goal: <一个可验收的目标>
verified_facts:
  - <事实 + 文件/行号或结果来源>
inferences:
  - <明确标为推断>
files_to_inspect:
  - <path>
constraints:
  - preserve RT55 loading and inference compatibility
proposed_change:
  - <文件级修改；若只是诊断则写 none>
acceptance_checks:
  - <测试、指标或行为>
hpc_followup:
  - <需要超算执行的命令/产物；没有则写 none>
risks:
  - <兼容性、泄漏、算力、统计风险>
open_questions:
  - <必须由用户决定的问题；没有则写 none>
[/AI-HANDOFF]
```

如果只是原因分析，`proposed_change` 必须写 `none`，不要把建议伪装成已实现代码。

## 5. Codex 返回的格式

```text
[CODEX-RESULT]
task_id: <same task_id>
base_commit: <SHA before work>
result_commit: <full SHA, or none>
branch: <branch>
changed_files:
  - <path + change summary>
verification:
  - <command>: <pass/fail/not run + reason>
compatibility:
  - <RT55 regression evidence>
hpc_status:
  - <not submitted/submitted/job id/completed>
remaining_risks:
  - <known limitation>
review_request:
  - <what ChatGPT should review at result_commit>
[/CODEX-RESULT]
```

## 6. 避免冲突

- 不让 ChatGPT 和 Codex 同时改同一文件；先冻结交接所依据的 base commit。
- ChatGPT 提出的 patch 必须视为建议，Codex 应结合当前工作树重新核验。
- Codex 只暂存本任务文件，不顺带提交用户已有的未跟踪结果或脚本。
- 重要实验变更使用新 config、新输出目录和新实验标识，不回写 RT55 原始配置或结果。
- 推送后用 commit SHA 复核，不用“最新版”“刚才那版”等相对描述。

## 7. GitHub Project 初次连接检查

- 授权仓库：`rular099/team_pytorch`。
- 活跃开发分支：以用户指定为准；本快照使用 `zhangb/native-scale-adapter-scaling`，不要默认读取 `team_collab_baseline`。
- 在第一个 Project 对话里要求 ChatGPT 回报它实际读到的 branch、commit，以及 `AGENTS.md` 首个标题。
- 若连接器只能看到默认分支，应先通过正常 PR/合并流程把这些协作文档带到默认分支，不能假装已经读取当前实验分支。

## 8. FE01 V2 独立工程交付（2026-10-01）

任务 `20261001-fe01-native-window-random-time-v2` 在独立分支
`exp/fe01-feature-extractor-site-effects`，基于
`9c95dbfaf92f36b2673d816026cd3f25b91eec66`。V2原生窗口/随机时刻方案替代V1；
未回写RT55或混入V01。入口：[协议](FE01_PROTOCOL.md)、[代码审计](FE01_REPO_AUDIT.md)、
[运行手册](FE01_HPC_RUNBOOK.md)、[验证](FE01_VERIFICATION.md)、[证据schema](FE01_ARTIFACT_SCHEMA.md)。
本轮停止于手动超算交接，NOT_SUBMITTED，87项本地聚焦测试通过；真实DiTing/生产数据/
设备资源/正式评价仍待手动audit。不要把reports/fe01_local_20261001的合成证据作为Japan结果。

## 9. FE01 超算结果交接（2026-10-04）

用户下载的 `chaosuan_res/fe01_runs_20261002` 已整理为
[真实超算结果报告](../../reports/fe01_hpc_results_20261004/README.md) 与
[ChatGPT分析要求](../../reports/fe01_hpc_results_20261004/CHATGPT_ANALYSIS_PROMPT.md)。
具体身份、检查和限制见 [CODEX-RESULT](CODEX_RESULT_20261004_FE01_HPC_RESULTS.md)。
结果commit `b5b2421f2667461e8ae785bcccc01e00411d905d`，仍在
`exp/fe01-feature-extractor-site-effects`；训练记录source为
`73ae9dec50713805c878b5e4efeb8264c01d0c76`。

TEAM/PhaseNet/EQT各3 seed的曲线均有12轮，当前成绩为预定选epoch后的validation。
RT55复用原训练权重，尚未取得共同协议复评；本轮没有新超算提交或test评价。
第8节是2026-10-01实现交付快照，不能用其NOT_SUBMITTED描述当前已提供的训练记录。

## 10. FE01-EVAL1 实际超算核验交接（2026-10-06）

用户下载的 `fe01_eval1_mpi_29351034` 已整理为
[身份/前向/参照报告](../../reports/fe01_eval1_hpc_gates_20261006/README.md) 与
[ChatGPT分析任务](../../reports/fe01_eval1_hpc_gates_20261006/CHATGPT_ANALYSIS_PROMPT.md)。
整理base `221f8b24483148d630e0eb85a179d763f8c580ac`；超算评价source为
`4c9c1757f726e571022a398cb5a217ca6c325cbc`，仍在FE01独立分支。
11组checkpoint身份通过；PhaseNet/EQT各3seed及RT55 ep32/RT61 ep8，共8组小样本
forward PASS。TEAM三组scratch因评价入口无条件比较预训练SHA与原lock中的null而BLOCKED；
本轮定位原因，没有修改runner或旧lock。共同train-only参照已生成，全部915个validation
台站均在九组FE01训练曝光表中见过，不能宣称未见台站泛化。

本批identity/verify/reference实际Job为29351785/29354874/29354884；未提供sacct，
后续正式评价/回放未提供结果。14份原阶段seal的30个文件和84个tar成员一致性校验通过。
第9节及前轮CSV报告为历史训练/性能证据，不应用其“待核验”概括本批已经提供的门。

## 11. FE01-A01 最新恢复入口（2026-10-11）

A01在独立分支 `exp/fe01-a01-absolute-amplitude-ablation`。
数组30265280两项DiTing均通过16卡通信，ON恢复epoch5时RNG设备类型错误，
OFF20分钟启动超时且卡点未知。当前使用
[新恢复包说明](FE01_A01_RESUME_FIX.md)和
[本批证据报告](../../reports/fe01_a01_continue_20261011/README.md)。
运营兼容循环独立记录SHA，不修改旧认证源码/门控，不把其核心SHA当成完整执行身份。
本地验证不等于真实续训成功；本轮未提交真实超算作业，也未进行生产评价或test访问。

## 12. A01 分批结果评价与打包（2026-10-14）

用户报告train全部完成，最终checkpoint身份仍待计算节点实际检查。
使用[跨批次eval/pack入口](FE01_A01_MULTIROOT_EVAL_PACK.md)，只读原训练结果，
将成功评价写入新session；无需合并权重或改旧配置/lock。
手动分阶段提交0–3评价、4评价、CPU打包，不自动串联；包只含通过seal的成功结果。
实现与边界见[交付报告](../../reports/fe01_a01_eval_pack_20261014/README.md)。

## 13. TEAM eval单项修复（2026-10-14）

用户订正后只有0失败、1/2/3仍运行。已修正scratch TEAM被误当作冻结encoder的运营检查。
当前[独立0号重提/兼容pack入口](FE01_A01_TEAM_EVAL_FIX.md)保持原评价session和在跑工具不变，
新0记录真实修复执行SHA；原4照旧，最终用修复入口pack验证混合来源。
不重新训练、不重新prepare、不重新签名旧门控；真实重评/pack待用户执行。

## 14. A01 当前完成情况和 pack 时机（2026-10-15）

新增[只读五项评价状态入口](FE01_A01_EVAL_STATUS.md)，逐项验证完整 seal 及文件 SHA256，
显示最近重提、排队/运行/失败/未知，并在 3 成功后提示单独提交 4、五项齐全后提示修复入口 pack。
用户报告修复入口执行成功，不等于本地已取得五项完成证据；检查脚本不提交任何作业。

## 15. A01 DiTing OFF 配对读取修复（2026-10-20）

用户提供完整状态输出：0–3 通过，4 在读取已封存 ON CSV 的人口门处失败。
当前使用[单项 4 修复入口及兼容 pack](FE01_A01_DITING_EVAL_FIX.md)：round_trip 读取后
仍执行原严格生产人口门，新增独立 [4] revision，不改旧 seal/核心，不重跑前四项。
同包 status 去掉旧集群不支持的 sacct --array。最终 pack 必须使用本次新入口。
真实 CSV 尚未本地取得，精度复现为合成证据；若实际身份仍不符会停止并写小型诊断 JSON。

## 16. A01 实际结果已整理供独立分析（2026-10-21）

用户下载的 `outputs/review.tar.gz` 已完整核验，五项评价完成证据齐全。
当前入口为[中立事实报告](../../reports/fe01_a01_hpc_results_20261021/README.md)及
[ChatGPT独立分析任务](../../reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md)。
16份导出、960个score cell和8个noninput主区间均从保存数据复核通过；全部为validation、seed42。
报告显式区分selected/epoch12、目标/field权重及诊断checkpoint状态和稀疏bootstrap支持。
本地Codex独立分析另外封存，未混入中立报告或上传ZIP，以便用户收到ChatGPT结论后对照。
没有加载模型/波形、追加训练/eval或访问test。

## 17. FE01 ON V4 独立评价交付（2026-10-22）

新分支`exp/fe01-on-warning-map-evaluation`、独立工作区`team_pytorch_fe01_on_eval`，基点29ffb20。
四ON selected固定DiTing12/PhaseNet12/EQT10/TEAM11，不训练、不访问test。
入口：[完整运行手册](FE01_ON_EVAL_RUNBOOK.md)、[科学协议](FE01_ON_EVAL_PROTOCOL.md)、
[实际验证](FE01_ON_EVAL_VERIFICATION.md)、[真实CPU结果与复核任务](../../reports/fe01_on_eval_20261022/README.md)。
已实际完成既有预测跨模型/固定分层/四交互/68%宽度分析和合成动态地图/TTUM/shadow端到端，55项CPU聚焦测试通过。
原A01和其他实验工作区不变。新的checkpoint/DCU/波形补算需用户用独立release脚本手动提交，正式HPC为NOT_SUBMITTED；
本地合成证据不代表Japan新逐秒/网格/警报成绩。不要把新固定query人口当旧194265请求。

## 18. V4 inventory 清单路径修复（2026-10-25）

实际 Job31089276_0 在校验预训练 manifest 时失败；通用 relocation 错把 DiTing 的独立清单改成
PhaseNet/EQT 的通用 manifest。修复保留可访问的原路径、原 SHA 与 asset 门，旧核心不变。
使用[新 release 恢复脚本](FE01_ON_EVAL_INVENTORY_FIX.md)只提交新 inventory，已封存 CPU 项附原始身份复用，
旧 run 和在跑作业不动。修复重提尚未由本地自动提交，真实通过状态仍待用户运行。

## 19. V4 TEAM model_smoke 单项受控重试（2026-11-02）

实际 Job32163754_3 DCU可见、TEAM ep11身份通过，失败在future encoder feature绝对一致性门；
此前实际输入/归一化hash门通过。旧日志缺blank重复与实际差值，原因暂不能定为泄漏或设备数值波动。
使用[独立TEAM重试小包](FE01_V4_TEAM_SMOKE_FIX.md)，沿用原50672e1 release和原run，仅sbatch 3号。
原runner/gates不改，新增audit采用既有A01 paired_execution并补同输入重复对照和逐future诊断；原阈值不放宽。
新增运营身份随原stage seal独立保存，其他模型/成功结果保留。实际受控DCU重试尚未提交。

2026-11-05 更新：提交入口无参数调用出现 `args[@]: unbound variable`，新版改为原生`"$@"`且使用私有变量前缀，
补上实际shell无参数/显式参数/带空格env/同名变量覆盖/缺值测试，22项专项通过。
仍沿用原run，仅重提TEAM；新的实际设备结果待用户提交，见同一[操作说明](FE01_V4_TEAM_SMOKE_FIX.md)。

## 20. 跨服务器Codex交接（2026-11-08）

用户要求另一台服务器的Codex接着当前进度推进。完整[接手文档](FE01_V4_CODEX_HANDOFF_20261108.md)
冻结本对话可见状态、五个关键版本、原run/env、当前小包/摘要、下一步及后续阶段命令、可重建资源和未取得的证据。
最新有效TEAM小包实施commit为6e2ddb5；没有收到其新job结果，不能把已交付当成超算通过。
本次仅同步文档，不修改执行源码或运行身份，不提交/取消任何超算任务。

## 21. TEAM同输入重复失败与严格确定性审计（2026-11-08）

Job33116200_3实际通过DCU/TEAM ep11身份门，在第一个validation请求的blank repeat检查失败：
feature最大绝对差1.52587890625e-5，阈值1e-5；尚无完整MDN差值或该attempt诊断JSON。
现有卷积确定性控制没有设置PyTorch全局deterministic algorithms，1.13.1源码显示ROCm BLAS原子模式由后者控制。
新运营小包仅在smoke audit启用严格全局模式，恢复旧enabled/warn_only，保留原runner字节码/门/阈值，打印backend与MDN差值。
32项本地专项通过，真实设备新包尚未提交；不以控制缺口推断已确定实际算子根因。
操作入口和证据见同一[TEAM重试说明](FE01_V4_TEAM_SMOKE_FIX.md)。

## 22. TEAM Job33156666_3真实模型门日志通过（2026-11-08）

用户提供out/err，TEAM ep11/seed42的两个validation smoke事件×两geometry×五时刻共20个请求重复检查全部通过；
48条打印的future特征检查也全部通过，feature最大差7.62939453125e-6，原门1e-5不变。
stdout最终出现模型门PASS；该标记先于封存，故仍需原status确认四组smoke均`COMPLETE_SEAL_VERIFIED`后再手动pilot。
逐请求/行号/日志SHA及证据边界见[日志核验JSON](../../reports/fe01_v4_team_smoke_repeatability_20261108/job_33156666_log_review.json)。
本轮只更新证据与文档；继续原50672e1主程序、c66db61运营包和修复run，不重训/重提或重新prepare。

## 23. 四组smoke与pilot实物核验完成（2026-11-09）

用户下载fe01_v4_pilot_review_20261109T113132.tar.gz，SHA40764a1cb0cdb34e2d5e080c621da78a7d223b9c5a6512288f976e1b62307395。
8份smoke/pilot seal及333个封存文件通过；pilot对四份smoke父seal SHA逐一匹配，TEAM运营源码确为c66db61，20次审计设置恢复通过。
每模型为validation/seed42的共同6个元数据预选事件、normal/random、逐秒1–20、240帧，
6960条台站请求（4882非输入、2078输入）及76840条网格请求；网格24040支持、52800缺高程支撑留NA。
模型epoch固定12/12/10/11，完整MDN、概率、输出人口和物理输入配对核验通过；不以pilot比较准确率或调参。

worker总耗时分钟DiTing8.55/PhaseNet3.26/EQT3.94/TEAM3.32，PyTorch峰值allocated GiB分别4.93/.17/.18/.21；
这不是正式重复warm latency或总设备/主机内存。原32事件分块、per-worker资源和既有run保持。
下一步手动提交replay_stations、normal replay_grid、latency；可分别cap6/6/4，三数组合计最多16个worker。
CPU inventory/covariates原文件未随包，现有status报告完整通过，记录的父SHA彼此一致；不宣称本地重验未提供原件。
当前包的status中全量阶段尚未提交，本轮没有sbatch/srun/模型重执行；后续实时状态仍由原status核验。
证据与重现脚本见reports/fe01_v4_pilot_results_20261109/，无需再次传递或重建HPC程序包。

## 24. 登录节点边界与小worker pool调度（2026-11-10）

用户回报replay_stations 328项和normal grid164项sbatch失败，原入口未显示捕获的stderr；
latency四项已提交Job33441392。不能仅凭非零返回断定MaxArraySize或账户限额。
新增独立compute-submit小包：登录节点仅shell+一次sbatch，CPU计算节点核验原plan/前置seal/源码并做映射调度。
每阶段默认4个Slurm slot顺序执行原逻辑任务，每个原32事件块使用新Python进程和同一原worker bytecode。
输出/逻辑编号/科学协议不变，新映射和运营身份入seal；不改原SLURM_ARRAY_TASK_ID或旧锁/成功项。
全成功/在跑跳过，UNKNOWN拒绝，原始sbatch stdout/stderr落盘；状态查询也作为CPU作业运行。
当前入口/失败取证及局部并发定义见[运行手册顶部](FE01_ON_EVAL_RUNBOOK.md)。本轮不提交真实作业。

## 25. 计算节点控制作业已实际提交两个pool（2026-11-11）

用户提供runs/control_33473739.out/.err，stdout明确SUBMITTED_POOL：
replay_stations job33474027 workers4 logical_tasks328，replay_grid_normal job33474094 workers4 logical_tasks164。
源码打印该行在真实sbatch返回job ID且作业回执落盘之后；本批stderr无traceback/提交拒绝。
当前不再把新入口统称NOT_SUBMITTED，但不能将两个pool已提交称为回放完成；后续看pool日志/计算节点status与完整seal。
latency33441392没有新状态，不重提。代码/模型/配置不改，仅保存日志SHA、行号和交接事实。
