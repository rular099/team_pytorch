# FE01-A01 本地实施报告（2026-10-07）

本轮交付独立代码、8份 ON/OFF 配置、5项新训练矩阵、逐阶段超算提交脚本、
本地诊断与轻量证据。正式超算任务状态全部为 **NOT_SUBMITTED**。
真实已训练 checkpoint 的机制定位与 ON 复用最终验收为 **NOT_RUN，待超算执行**。

任务ID：`20261007-fe01-a01-geometry-absolute-amplitude`。
基线：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`。
分支：`exp/fe01-a01-absolute-amplitude-ablation`。
结果 commit 由交付的源码包 `FE01_A01_SOURCE_IDENTITY.json` 和最终 CODEX-RESULT 标明。
新增源码与115文件依赖的逐文件 SHA 在 `verification/source_identity.json`。
本地执行时该清单记录未提交 worktree 状态；发布包另记录真正的结果 commit。

## 1. 已核验事实与实施内容

- 原公共下游使用地理 sin/cos 编码、absolute coordinates、station context off、
  target cross-attention 和 first query residual。它不是台站序号编码。
- 单有效 key 的 pure attention 与 query 无关，但 first residual 仍携带 query。
  first residual gate 若存在，乘在 attention 分支上；未设置该参数时系数为1。
- 幅值旁路是11维。OFF仅屏蔽前10维 std/RMS/peak 的绝对尺度，保留末维时长；
  参数结构、投影、gate、归一化形状和分量比例不变。
- `A01FullModel` 经显式 factory 注入；原115文件和 EVAL1 runners 未修改。
- 独立训练循环保留原 Adam、loss 分母、梯度累积、clip、DDP、drop-last、cosine、
  十cell noninput MAE 与 tie-earlier 选择规则；从认证 epoch0 初始化。
- TEAM-null/pretrained-SHA、错 epoch/encoder/config/training source 等正负测试已执行。
- 原三个工作区的 HEAD、branch、status、staged/unstaged diff SHA均与接手前一致。
  原训练115文件 SHA保持
  `3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`。

## 2. 本地机制诊断结果：合成例子，不是生产结论

合成波形使用固定随机种子；下游维度20、4层 readout，未进行正式训练。
TEAM使用 scratch frontend；PhaseNet/EQT使用本地已注册真实 STEAD v2 encoder，
仍然不能将其称为已训练 FE01 生产模型。

| 合成 K=1 / ON | pure attention跨query最大差异 | residual后跨query RMS | 最终均值跨query范围 |
|---|---:|---:|---:|
| TEAM | 0 | 0.44309 | 0.00053632 |
| PhaseNet | 0 | 0.39744 | 0.00044477 |
| EQT | 0 | 0.37271 | 0.00043142 |

位置编码确实区分这些查询；单 key pure attention 恒定；residual恢复查询差异；
最终输出头仍可传递很小的均值与分布差异。这里验证了追踪和可微路径，
没有定位真实训练后模型的唯一衰减机制，也不能把不同表示量纲的 RMS直接相除作因果解释。

查询梯度使用 eval + enable_grad，不走 detached cache，无 optimizer 更新。
0.01km有限差分与自动微分的均值最大绝对差约为 TEAM 9.4e-6、PhaseNet 1.5e-5、EQT 1.2e-5；
另外保存0.1/1km结果、weights/mu/sigma导数及每次实际 gate/分支范数。
纬经度转换使用实际绝对纬度；高程单位未认证，不报告高程公里导数。
同步 station 置换通过等价检查；坐标单独置换仅标为机制干预。

图、CSV、完整 trace 数组及图注分别位于：

- [TEAM 查询流](synthetic/team_original_scratch/figures/query_flow.png)
- [TEAM gate与分支范数](synthetic/team_original_scratch/figures/gates.png)
- [TEAM 尺度与未来扰动](synthetic/team_original_scratch/figures/interventions.png)
- `synthetic/<family>/query_stages.csv`、`query_sensitivity.csv`、`gates.csv`、`branch_norms.csv`
- `synthetic/<family>/traces/*/query_trace_arrays.npz`、`query_gates_branches.json`
- `synthetic/<family>/figures/CAPTIONS.md`（中英文证据边界）

另两家族都有同样的三张图与源数据。所有图明确标注 SYNTHETIC。

## 3. 幅值可得性与尺度检查

三个可用家族的合成触发边界、原生容量边界、尾部缺测、内部合法零、invalid station，
都完成截止后 pulse/NaN/Inf 检查。选站、mask、统计、encoder/adapter、MDN与概率不变；
未来扰动的归一化波形、encoder/adapter与MDN最大误差均为0。
最后合法样本正控制能改变相关统计；未要求任意 checkpoint 的最终预测必变。
固定请求时改变最终 PGA 标签值，不改变 forward；目标只作 loss/评分监督。

正常数值区间的事件共同增益和逐站0.1/1/10增益检查通过：
归一化波形最大差约1.2e-7；OFF frontend/adapter最大差小于7e-7；
OFF MDN最大差小于1e-9；ON前10维 log10平移误差小于2.4e-7；时长不变。
极低能量进入 eps/clamp 区时另列 `EPS_CLAMP_LIMITATION`，不宣称精确尺度不变。

已完成三家族 OFF 单 microbatch 更新 smoke：从 factory epoch0 起步，loss有限、
参数确实更新，预训练 encoder保持冻结/eval。详见 `off_smoke/OFF_single_update_smoke.json`。
这不是 global batch128/world16 的正式训练，也不是消融效果证据。

上游原始滤波/重采样链路及原始波形不可见，状态保持 **UPSTREAM_CAUSALITY_UNKNOWN**。
HDF后前缀检查不能证明 HDF生成前不存在未来依赖。

## 4. 真实人口与探针：只做 metadata，不做性能推断

从原 TEAM seed42 epoch11 CSV核对：1310 events、194265 all rows、139440 noninput rows。
population SHA为 `e96284780cb5f72b952c1697d4d264d307e7139a153201fa222566015aec7f69`；
label/input/clock SHA为 `f204deca06725af5513895416d5b0691c75c12221127854c32d1d497855155bb`。
源CSV SHA为 `d58a7b410e8d8df5bf46ba6e7022d07740b57bb9d932140c4754d2d57149c8bc`。

按时间、geometry、实际K1/Kmulti、Q1–4/Q≥5各层最多32，选择1128 decisions、12076 query requests。
缺项和实际数量见 `real_metadata_probe_strata.csv`；选择不使用误差或标签值。
`real_metadata_probe_requests.csv.gz` 保留物理站点/输入/窗口/截止身份，真实前向仍未执行。

旧 fixed_requests 再序列化副本的标签显示浮点会变，严格指纹检查已拒绝该副本。
A01用原冻结 CSV字节，核对全部请求键、HDF float32物理标签和整数时钟后，
才统一标签的十进制显示身份。缺项输出差异并失败，不取交集或重签旧 lock。

采样预览见 `sampling_preview/` 与可执行 `sampling_preview_source.py`：
每家族真实 train cohort 前32 events、12 epochs、每事件3 draws，仅 metadata时间计划，
不是全训练 journal或真实选站证据。四家族各自100000 draws 的采样器审计通过。
程序按 PyTorch sampler/loader 核验完整9084-event、world16预算为每epoch212 updates、
总2544 updates，消费27136、舍弃116样本。完整计划由超算 audit生成。

## 5. 首轮矩阵与旧 ON 复用状态

| 索引 | 新训练 | 对照 | 本轮状态 |
|---|---|---|---|
| 0 | TEAM seed42 OFF | 原 ON epoch11；另报告epoch12 | CONDITIONAL_PENDING_HPC |
| 1 | PhaseNet seed42 OFF | 原 ON epoch12 | CONDITIONAL_PENDING_HPC |
| 2 | EQT seed42 OFF | 原 ON epoch10；另报告epoch12 | CONDITIONAL_PENDING_HPC |
| 3 | DiTing seed42 ON | 新匹配公共下游与adapter初始状态 | NOT_SUBMITTED |
| 4 | DiTing seed42 OFF | 索引3 | NOT_SUBMITTED |

原 init/best/last 的冻结 inventory SHA已加入配置；现有元数据支持准备复用。
本地小型 ON 前向/初始状态/单步更新等价通过，但本地没有生产 checkpoint bytes，
不能因此宣称三个生产旧 ON 的 A01复用已经最终 PASS。
超算 audit核对实际权重、全部配置/数据/预算/world16与真实 trained ON前向。
失败时阻止对应家族，不静默增加训练数量。
DiTing预训练 encoder 的真实路径/SHA来自旧 EVAL1 inventory，仅进入私有部署文件。
原共享权重清单保持不动，新注册记录只写 A01专用 manifest。
DiTing完整权重/前向本地为 NOT_RUN，不用随机 encoder替代。

## 6. 评价、测试与下一步

新评价锁定validation共同十cell，输出all/noninput/triggered/untriggered/K1/Kmulti。
精确MDN NLL从原 logits稳定计算，避免 float32 softmax下溢；有专门负例测试。
同时输出MAE/RMSE/R²/bias/slope、CRPS/Brier/sigma、coverage/width、
level/shape MSE、P95-P05范围、近等距离站对误差和事件簇bootstrap。
稀疏子群明确缺cell、事件/target/field数、有效重复数；单seed只作首轮证据。
旧 ON保持冻结epoch，不因OFF结果重新选择；新旧都另报epoch12。

实际执行55项聚焦/原RT55-RT61加载桥接与指标回归、Python compile/import、
全部A01 shell bash-n、fake scheduler（prepare零调用；显式submit只一阶段）、合成诊断与OFF更新。
实际输出在 `verification/`；有既有 timm deprecation warning，无测试失败。
既有依赖提示 apex/xformers可选，未为此安装或改变环境。

操作步骤见 [逐阶段超算说明](../../docs/ai/FE01_A01_HPC_RUNBOOK.md)。
上传源码包+SHA+已填写路径的部署脚本；登录节点部署并 prepare，然后手动提交 environment、
audit、diagnostics、pilot、train、eval、pack。每次仅一阶段，源码改变须重新门控。
本轮没有 sbatch/srun真实调用，没有正式训练、完整生产评价或 held-out test访问。

仍待验证：真实已训练 gate/分支尺度与空间常数现象的关系、完整下游前缀因果性实测、
DiTing匹配初始化/原生前向、真实 ON/OFF消融收益以及跨seed稳定性。
请优先审阅诊断归因边界、旧ON条件复用、DDP loss分母与严格人口身份。
