# V01 validation closure：仅补验证，不重训

任务 `20261004-v01-validation-closure`，基线
`74c55aa5442b4200961c88ceee3d11af5033275d`，分支
`exp/v01-velocity-prep-padding-control`。完整任务书见
`docs/ai/V01_prompt_2.md` 和 `docs/ai/V01_REVIEW_AND_CODEX_HANDOFF_20261004.md`。

## 本轮范围与状态

2026-10-05：用户已提交首轮 closure，audit 在读取标量 `v01_reference_p_pick` 时失败。
两处新审计读取已修复，生产格式的标量 fixture 和全套 159 项本地回归通过。
修复版超算未提交；真实 derived cache 的 idx7 trace 与新指标仍待生成。
本地 fixture 不能替代真实失败行。详见 `docs/ai/CODEX_RESULT_20261005_V01_SCALAR_FIX.md`。
不重训、不 preflight、不旧 all/recover、不 RT62、不访问 held-out test。

旧 `eval_retry1`、原报告、模型、训练代码、loss、PGA 定义、三臂 checkpoint 都保留。
新增开关只有 `training_params.v01_validation_closure=true`，限定 realtime val；
RT55/RT56 默认保留旧 clock alias、空样本替代和原推理数值。
新协议为 `v01-validation-closure-v1`；normal 会恢复过去被 clock mask 错删的合法
query，因此不能与旧错误协议混称同一个分母。

AA 离线逐字段审计：15 组重复 event/time，共 45 条额外行，15 组预测均不完全一致。
原 NPZ SHA256 为
`c8d5b53d7b98e332c349f57dc5738df390e204d393ab52d78fcf4d7b8429ee7d`。
不能安全去重，未修改原文件。证据在
`reports/v01_validation_closure_20261004/aa_offline_duplicate_audit.json`。
因此本次建议显式增加 **一格 AA random**，不是重跑四格速度 random。

## 上传和启动

使用交付的完整源码包解压到现有 `_vel` 项目目录，或同步精确 result commit 的源码。
不要使用旧 recovery/all 脚本。无需登录节点 torch，无需 `.git`、GitHub 连接或
`EXPECTED_GIT_COMMIT`；提交器会校验 `tools/v01_closure_source_manifest.json`。
不要把旧权重或 derived cache 移到新代码包里；沿用旧 resolved config 的绝对路径。

在超算登录节点：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics_vel
export WORKDIR="$PWD"
export V01_RUN_ROOT="$WORKDIR/v01_velocity_prep_padding_seed42"
# 标量读取修复后的重试：保留 validation_closure_v1 的原日志/来源快照
export V01_CLOSURE_ROOT="$V01_RUN_ROOT/validation_closure_v1_retry1"
export DITING_CONFIG="$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml"
export DITING_PRETRAINED="/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt"

# 预览：5 normal + 已核实必要的 1 AA random + 审计/分析；不提交，不建目录
INCLUDE_AA_RANDOM=1 bash tools/complete_v01_validation_slurm.sh

# 正式提交：必须显式确认；登录节点运行 bash，不要对提交器再 sbatch
DRY_RUN=0 CONFIRM_V01_CLOSURE=1 INCLUDE_AA_RANDOM=1 \
  bash tools/complete_v01_validation_slurm.sh
```

默认不加 `INCLUDE_AA_RANDOM=1` 时，仅五个 normal；分析会明确 AA random 未闭环，
并且不拿旧 AA 重复导出做 A/V 配对。当前非一致重复已核实，所以建议使用上面的六格命令。
没有任何训练分支。新增路径以 `$WORKDIR`、`$V01_RUN_ROOT` 推导，不写入新的私有路径。

默认新目录为 `$V01_RUN_ROOT/validation_closure_v1`，存在即拒绝再次提交/覆盖。
若确有失败后重提需求，先检查已提交 ID 和产物，再显式设置新的
`V01_CLOSURE_ROOT`，不要直接重复调用、删除目录或重训。源码提交后不得在运行途中更改。

本次读取错误发生在 CPU audit，依赖其成功的评估不应启动；实际队列仍须核查。
上传补丁前，用原 `validation_closure_v1/submitted_jobs.tsv` 中的明确 job ID 检查
`squeue` / `sacct`，仅取消其中仍待运行、依赖已失败的本轮 closure 作业。
不要批量取消其他实验，也不要继续旧 audit job：它的源码哈希快照与补丁不同。
修复后的完整 runtime manifest SHA256 为
`88d4570c2f87c3f0daf00eece6e87dc0a733c78a4d86963645152d1538d9fc86`。
新目录由 prepare 写入新的 snapshot；保留旧 snapshot 供追溯。

资源：评估单节点、单 DCU、8 CPU、23:50:00；审计/分析 CPU-only、04:00:00。
默认不传 `--mem`，避免之前不可满足的固定内存申请；需要时使用站点允许的 `SLURM_MEM`。
可设置 `SLURM_PARTITION`、`SLURM_ACCOUNT`、`SLURM_CPUS_PER_TASK`、`EVAL_TIME`、`AUDIT_TIME`。
没有嵌套 srun gres，不申请 4 节点 DDP。仍需站点允许 CPU-only 作业进入所选 partition。

## 门禁与作业依赖

审计 -> afterok 评估 -> afterok 分析。完整 ID/依赖保存在 `submitted_jobs.tsv`。

审计从现有 cache 追踪第一个 shard 的实际 base event idx7，保存物理 row_selector、
source/query sensor IDs、原 PGA/坐标/P samples，以及 clock 前后 mask。要求：
旧路径实际出现 `Found event without PGA`；新版 query mask 不变；相同绝对截止；
真实输入/查询可生成。若恰好是无可用 source 的 AA 请求，须在真实生产 sampler 已恢复合法
query 后明确记录 no_available_source（不伪造预测、不换事件），也可通过这一根因门禁。
任一根因假设不成立就失败，后续评估不得运行。

审计还做：

- 固定 last SHA256、epoch8、optimizer step1496，以及三个相同 init 文件 SHA；
- 只复制既有 protocol lock/preflight/cohort/split/config/source manifest；缺项标 NOT_AVAILABLE，不重建；
- cache train/dev 计数与实际 loader 筛选后 eligible event/metadata row 计数，二者分开；
- 原 encoder 路径复用、当前 encoder/parent/split/YAML hash；缺少历史 hash 时不冒称旧运行证据；
- 四格速度 random 全量请求/输出身份、PGA 标签、绝对 cutoff 对账；
- 有界真实 cache 输入/label/旧 info SHA：第一 shard idx7 全部七个时间 + 每 shard 首请求，
  开关前后逐字节相同；配合生产机制回归，**不是完整 encoder 数值重跑**；
- AA 全部 45 个替代请求的实际事件/时间映射及原请求 strict availability，保留不可恢复证据。

如真实 trace 或旧 random 签名不匹配，脚本失败并停止，不自动扩大运行。
检查 `logs/v01-closure-audit-*.out/.err` 和 `audit/*.idx7_trace.json`；给我这些文件即可定位。
失败依赖作业可能停在 DependencyNeverSatisfied；检查 `submitted_jobs.tsv` 后按 ID 取消，
不要为解决依赖失败启动旧 all/recover。

## 新评估输出

`eval/<cell>.npz` / `.metrics.json` / `.txt`，以及
`<cell>.npz.val.requests.jsonl` 和 `.summary.json`。
每个请求记录 requested/actual event、时间、真实 source/query ID、UTC cutoff、原 row 与结果：
predicted、no_available_source、invalid_label_or_metadata、implementation_error。
实现错误记录后失败，不把 ValueError 一概 skip；失败 summary 明确 pending 请求未执行。
成功时 requested = 四类 outcome 数量之和；不得用后续 event/time 替代当前请求。

每个评估开始/结束核验原 last 文件 SHA；CPU 前后再次核验三臂 epoch/step/hash。
评估不写 checkpoint，不选 best/新 epoch。

## 新报告与科学边界

分析器写 `report/`，保留旧结果报告：

- all7、early135、1/3/5s 的 all / 阈值上 / 阈值下 / untriggered / single-source；
- MAE、RMSE、NLL、Brier 的 target-micro 和 event-macro 配对 event-cluster CI，
  5000 次、seed20260915；macro RMSE 定义为各 event RMSE 的均值；
- 物理 query ID + event + requested time + exact absolute UTC cutoff 一对一 outer join；
  匹配量、左右 unmatched CSV 和分层 eligibility 数量均保留；
- FF/MM 原误差的 MSE、bias²、centered variance 及同一 event bootstrap 的 CI；不做 bias correction；
- 历史 `np.ptp ratio median` 与新增 `P95−P05 ratio mean` 分文件，不替换旧指标；
- A/V 各自 available/full request coverage、缺失事件/候选 query 机会、common-available 交集；
- 已选 source 的 paired acceleration sensor ID 排除后的 common-remote **sensor proxy**。
  Cache 中 sensor pairing 不足以认证“不同真实物理台址”；真实 site-remote 标 NOT_AVAILABLE，
  不把坐标近邻或数组 slot 冒充真实站点身份。

无预测目标机会分母是 sampling 前有效 query；预测 targets 是选中 query，二者不可相除冒称
sampling 后目标弃权率。各 arm 的 full prediction rate 以 frozen event/time request 数量为分母。
共同可预测交集存在选择偏差，须和完整请求覆盖率一起看。
A/V 涉及不同仪器、深度、场地、频响和输入域，不单独证明 padding 因果效应。

阈值固定为 -1.2 log10(m/s²)，指 PGA 目标点，不指大震震级。单 seed、8 epoch、开发 validation、
原 full-record centering 与 cutout+1 的在线因果性风险仍保留，本轮不偷偷改协议。

图表按 figure-designer 的 experimental-results 规则：2×2 FF/MM normal/random 密度图，
共同 [-3,1] 坐标与 Viridis 对数计数；分层残差图使用 [-2,2]，颜色+实/虚线双编码；
9pt、SVG+PNG、单位明确、无装饰、阈值上/下并列。`density_bin_counts.csv`、
`residual_bin_counts.csv`、`figure_count_audit.json` 给出全部计数及视野外数量。
当前只能通过合成 fixture 核验绘图/计数接口；真实图须等超算结果，不声称已经完成视觉审阅。

## 回传结果

作业完成后，在运行根目录打包新目录（不打包权重、raw archive 或重复 loader CSV）：

```bash
cd "$V01_RUN_ROOT"
tar --exclude='validation_closure_v1_retry1/metadata_cache' \
  -czf v01_validation_closure_v1_retry1_results.tar.gz validation_closure_v1_retry1
sacct -j <submitted_jobs.tsv中的逗号分隔ID> \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P
```

把结果包与 sacct 文本下载到 `chaosuan_res/vel/`。不要把真实波形/大权重推 Git；
下一轮再整理真实闭环结果给 ChatGPT。本轮不把“代码完成”写成“超算结果完成”。
