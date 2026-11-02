# FE01 V2：原生窗口能力、随机因果时刻训练与场地相关空间预测评价

日期：2026-10-01（根据用户本轮两项提议修订）
性质：可直接交给本地 Codex 的实施任务书。训练与正式评估由用户手动提交到超算。
仓库：`rular099/team_pytorch`
已读取的代码基点：`rt61-wave-geometry-residual-conditioning` / `9c95dbfaf92f36b2673d816026cd3f25b91eec66`
建议支线：`exp/fe01-feature-extractor-site-effects`
建议独立工作目录：原仓库的同级目录 `team_pytorch_fe01`。

> 本文是一份实施设计，不是已执行实验的结果。不得将计划、静态代码发现、合成数据测试或本地 dry-run 写成真实数据训练成功。文中路径与函数名若属于“建议新增”，由 Codex 在该独立分支创建；已有接口必须以实际源码为准，不能臆造参数后声称可运行。


## V2 修订优先级与结论边界

**本文件整体替代 V1 任务书用于后续实施；V1 文件保留归档，不覆盖已有代码或实验。**

1. 主任务由“强制所有特征网络使用完全相同历史长度”调整为“相同真实决策截止时刻、相同数据来源/下游结构、各自已核验的原生窗口能力”。不为凑齐 90 秒结果而强行改 PhaseNet/EQT 输入长度、压缩时间轴或拼接多窗口。超出某个已声明能力域的项目记录 N/A，不算作错误，不补零分，也不填入总排名。
2. 区分三个量：绝对决策时刻 T、相对统一事件参考的经过时间 tau、实际喂给模型的历史区间/有效秒数。30 秒输入窗口不等于只能在首 P 后 30 秒内运行。主表按用户要求比较能够保留所要求累积历史的模型；原生滚动短窗是单独的运行语义，不混在累积历史结果中。
3. 训练使用样本时钟分辨率下的随机经过时间，范围最小 1 秒、最大 90 秒，并侧重早期；验证保留固定秒数。现有 RT55 已有 random_bins + 区间内均匀采样，故本轮是增加新分布/能力域/严格因果边界，不是宣称首次从离散训练改成随机训练。
4. 已核验当前模型前向会调用 waveform encoder，RT55 配置关闭 DPK prior cache。历史上固定时点是否最初为缓存而设，现有证据不足确认。不得把 DPK prior cache、同一 T 的查询上下文复用和跨 T 的预计算特征混为一谈。
5. 增加逐秒回放和端到端未来扰动检查；训练和正式评价仍全部由用户手动提交到超算。原 RT55 代码默认、checkpoint 和推理行为不变，FE01 的因果预处理修正只走新显式配置路径。

主结论准确命名为“原生窗口下的可部署特征方案比较”，其中窗口容量、表示、预训练和训练时间覆盖都可能影响性能；不能仅凭该实验唯一归因为 encoder 架构差异。相同历史长度的严格归因实验不再是默认交付门槛，必要时另设小型诊断，不自动增加全量训练。

## 0. 分工与任务边界

ChatGPT 负责本任务的科学问题、控制变量、评价设计、结果独立审阅和后续方向判断。Codex 负责代码追踪、局部实现、测试、配置、离线权重包、超算脚本、评价程序与证据封装。用户负责确认超算账户/数据/权重路径，在超算手动提交训练及正式评价，并回传运行产物。

本轮 Codex 必须完成“可以交给用户提交”的工程产物，不得代用户提交 `sbatch`、`srun`、远程训练、正式 test 评估或耗时的数据全量转换。可以执行小规模本地 CPU/已有设备 smoke、合成数据测试、脚本语法检查和无副作用 dry-run。

目标：在同一数据来源、同一绝对决策截止、各自已声明的历史能力域及相同特征之后的下游结构下，对比 DiTing、原始 TEAM 特征网络、PhaseNet、EQTransformer。评价既包括 PGA 点预测与概率质量，也包括未观测目标位置的空间差异恢复，以及跨事件稳定的场地相关残差。不能以“地图不圆”“散点看着好”代替定量证据。

## 1. 必须先做：隔离与来源核验

### 1.1 不碰当前工作区

先只读记录：`pwd`、`git status --short --branch`、`git rev-parse HEAD`、`git worktree list --porcelain`、已有 FE01 相关本地/远端分支。记录原工作区 tracked diff 与未跟踪文件清单；检查并发 Codex 会话可能使用的目录。

禁止在原工作区执行 checkout/switch、reset、clean、stash、删除文件、自动整理未跟踪文件、合并支线或覆盖输出。不要将用户尚未提交的改动自动带入实验。

若未有合适的独立工作区，在确认基点 commit 本地存在后创建：

```bash
# 下面的路径须替换为审计得到的真实路径；不要照抄占位符。
git -C /absolute/path/team_pytorch worktree add \
  -b exp/fe01-feature-extractor-site-effects \
  /absolute/path/team_pytorch_fe01 \
  9c95dbfaf92f36b2673d816026cd3f25b91eec66
```

若 commit 尚不存在，可先 fetch 对应远端分支，再核验 SHA；不能默认使用变化中的 HEAD。若同名分支/目录已存在，检查后安全复用，或新建带日期后缀的名称；禁止 `-B`、`--force` 和删目录重来。创建后仅在 FE01 工作目录写入。最后复核原工作区 HEAD、tracked diff、未跟踪清单均未因本任务改变。

worktree 只隔离工作文件，不隔离环境和输出路径。必须另设 FE01 环境、输出、缓存和日志根目录；原数据与预训练权重只读共享。禁止原地升级主线环境。不要直接把 linked worktree 的 `.git` 文件复制到超算；使用该支线的独立 clone 或固定 commit 的源代码归档，并记录 SHA/源码包校验和。

### 1.2 必读与核验材料

先读 `AGENTS.md`、`docs/ai/PROJECT_CONTEXT.md`、`docs/ai/README.md`、`SESSION_SUMMARY.md`，再读：

- `gemini_models.py`、`gemini_util_light.py`、`train_light.py`、`eval_checkpoint.py`；
- `diting/config/diting_1200m_backbone_attnpool.yml`，以及实际调用的 `dtbench` backbone/建模代码；
- `pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json` 及其完整继承链；
- RT56 的 causal random geometry sampler/config/tests；
- `docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md`，用于辨认现有主线和参考系统，而非使用其结果调本实验；
- `train_light_slurm.sh`、`eval_checkpoint_slurm.sh` 及现有 Slurm/DDP 启动实现。

注意入口文档存在历史快照：实际分支、resolved config、split manifest、checkpoint provenance 优先于文档标题和旧聊天记忆。已查看的 RT61 分支与速度支线 `exp/v01-velocity-prep-padding-control` 分开；本任务不得混入速度转换、另一套数据或补零修正。

生成 `docs/ai/FE01_REPO_AUDIT.md`，准确列出代码基点、执行路径、真实输入长度/采样率/分量顺序/物理单位、storage-valid mask、时间参考、特征边界、幅值分支、下游结构、冻结规则、数据划分和启动器危险默认项。路径不在本机可见时明确标为“待超算审计”，不能填造结果。

## 2. 实验设计：先固定问题，再写模型替换

### 2.1 统一下游的具体选择

本轮 FE01 的主对照统一使用仓库中已有的 **RT55 基础下游结构**：既有位置编码、多台站 Transformer、查询处理、PGA MDN，以及既有震级/位置辅助头（若基础配置启用）。这是支线的统一科学底座，不是把主线回滚到 RT55，也不是声称 RT55 优于 RT59。

替换边界定在“单台站波形表示进入既有位置融合/多台站上下文之前”。先准确定位实际函数和张量，再在协议中写死。特征提取端包括必要的时序聚合和维度适配；边界之后，四组的模块拓扑、宽度、位置编码、查询机制、输出参数化及 loss 完全相同。不得因为某个 encoder 不好接入，偷偷给它更深的下游网络。

主对照不新增 VS30、地质特征或站点 ID embedding。先审计现有基础配置是否实际使用这些字段；数据文件中有该字段不等于网络用了它。已有静态协变量必须四组保持一致，并在低信息控制中同等提供；若控制组不止坐标，应准确命名为 static_only，而不是 coords_only。任意格点接口不得依赖未定义的新站点 ID。

RT57–RT61 增强/残差/teacher 路径在主对照中统一关闭，尤其不得继承“绝大多数模块冻结、只训练少量 RT61 参数”的优化器规则。RT59/RT61 当前完整系统只列为单独的应用参考，不与新训四组混成严格配对的特征排名。后续是否将优胜 extractor 接入 RT59，由对照结果决定，不在本轮顺便修改。

四组的共同下游都重新训练；同一 seed 从逐参数相同的公共初始 state_dict 出发。不要让 DiTing 继承已训练的 TEAM/PGA head，而其他组从随机下游开始。特别检查 RT55 继承链里的 `transfer_model_path`、resume、teacher、optimizer/scheduler 状态，本实验主组不得继承这些已训练下游权重。

如果只有统一底座表现弱，不能直接断言全部特征无效；需区分下游空间表达能力的瓶颈。这一点在最终报告列为结论边界。

### 2.2 主组、控制组、结论范围

主组：

| run family | 特征初始化 | 特征训练策略 | 含义 |
|---|---|---|---|
| diting_pretrained_frozen | 当前实际 DiTing checkpoint | encoder 冻结，聚合/适配与公共下游训练 | 当前预训练表示的迁移价值 |
| phasenet_pretrained_frozen | 经核验的 SeisBench 权重 | 同上 | PhaseNet 选定中间表示的迁移价值 |
| eqt_pretrained_frozen | 经核验的 SeisBench 权重 | 同上 | EQTransformer 选定中间表示的迁移价值 |
| team_original_scratch | 原 TEAM CNN/MLP 的有效实现 | 从头训练特征端和同一公共下游 | 不借助外部预训练的实际基线 |

没有可验证、与本任务无标签泄漏的 TEAM 预训练权重时，**不得冻结一个随机 TEAM CNN 当作主要 TEAM 基线**。没有匹配权重就如实使用 scratch。若找到了合格权重，可额外加 TEAM frozen，但不能覆盖 scratch 的身份。

这些主组回答“现成特征方案在相同下游和各自原生窗口能力下哪种更有效”，不能单独证明“架构本身谁更优”：预训练数据、目标、参数量和可训练参数不同。要回答纯架构因果问题，需要另做一致预训练或全部 scratch 的实验；本轮不默认展开昂贵的大矩阵。

另准备两个低信息控制配置：

- `amplitude_only`：保留同样可见波形的幅值/有效时长统计与坐标，学习型波形表示替换为常量；单独训练同一下游。检查提升是否超过“看到已经到达的振幅”。
- `coords_only`：学习型波形表示和全部波形派生统计均替换为常量，仅保留共同的坐标与协议允许的可用性信息；单独训练，不把测试时置零的 OOD 扰动当作独立基线。

另提供 `diting_random_frozen` 配置用于同结构预训练控制，首轮不默认占用完整训练预算。随机冻结只是一种阴性对照，不是合理的最佳 scratch 基线。低学习率末层解冻、PhaseNet/EQT 层位敏感性、统一 scratch 比较均属于后续实验，不得看 test 后临时加入。

### 2.3 运行规模与阶段

先做接口/合成数据/少量本地可用样本检查。随后交付但不提交四个主组 seed=42 的 pilot 配置。pilot 只用于验证收敛、资源、数据与权重加载，不产生正式优胜结论。

正式主对照预生成 seeds=[42,43,44]，共 12 个主组训练配置；控制组也有对应独立配置，但默认不自动串行全部运行。预算、batch、最大 optimizer updates、共同 scheduler、验证频率、停止规则和 checkpoint 选择规则在正式运行前写入不可变协议。允许按显存用不同 microbatch，但通过梯度累积保持有效 batch、每事件决策样本数及 optimizer updates 可比；不同能力域的时间分布不宣称完全一致，必须报告实际时间暴露直方图。另报告 walltime/吞吐/显存。

采用现有可复用的 50% normal + 50% causal random geometry 混合训练协议，使四组都有未观测查询任务的训练覆盖；不改 loss 定义和权重。正式验证同时保留 normal 与固定 random geometry。若现有 loader 的确切语义不同，先修正 FE01 协议描述并做样本审计，不能仅改一个名字当作实现。

checkpoint 选择预注册为：在四个主组共同支持、事先锁定的固定 validation 时点与配对目标上，对 normal-noninput 和 random-noninput 分别计算各时点等权 MAE，再将两种协议等权平均；最小者入选，并列取较早 epoch。共同时间集合由窗口/参考/前置历史的样本索引审计确定，不能看结果后删时点。较长历史结果单列诊断，不用各模型不同的支持时间集合分别平均后排总名次。不要让 easy input targets 的数量支配选择。NLL/校准是独立报告项；不得看 test 再换选择标准。若需要调参，只在 validation 上使用各模型相同数量的预先定义试验，不以相同学习率替代“充分训练”的判断。

## 3. 特征接口与关键实现细节

### 3.1 不要认错 DiTing

本次读取到的配置是 `diting_1200m_backbone_attnpool.yml`：MAE backbone、`backbone_attn_pool`、`in_samples=10000`、`patch_size=50`、`model_depth=24`，宽度从 hps 解析。它不是凭名称猜测的 DiTingMotion 三分类网络，也不能用 SeisBench 里名称含 diting 的 PhaseNet 权重替代。[R4]

实际 checkpoint 与配置是否一致，由 Codex 审计验证。记录来源、精确 hash、encoder 层/模块路径、raw token 与最终单台站特征形状。复用当前实际特征路线；不喂分类概率当通用特征，不无依据新增 STFT/额外谱分支。若审计发现某一路包含额外输入信息，则各组统一提供或将其单列为不同实验条件。

### 3.2 每种特征端的要求

DiTing：保留实际 MAE 特征和现有时间聚合/幅值信息；冻结的范围必须覆盖参数、BN running state（如存在）与 dropout 状态。父模型 `.train()` 后再次检查 encoder 仍处于预期 eval 状态。

TEAM：定位并测试原 TEAM CNN/MLP 的实际可用实现；参考作者原始实现核对卷积轴、pool、flatten、归一化与幅值拼接。已查看的 `NormalizedScaleEmbedding` 有先归一化再算尺度、tuple-dim `torch.max`、字符串 activation 被调用和固定 flatten/MLP 构造等静态疑点，不能直接把它当作已通过测试的原始基线。[R3] 在 FE01 隔离模块修复/移植，保留原意并说明任何适配偏离；不要默默改主线旧类。窗口变化时显式计算 flatten 尺寸，不能沿用未经验证的硬编码值。

PhaseNet：主特征层选 encoder bottleneck，即上采样 decoder 之前的中间表示，不使用 P/S/noise 概率。通过锁定版本的源码实现 extractor 或稳定的显式接口；若用 hook，必须有调用次数、层路径、shape 和数值一致性检查，不可依赖“倒数第几个 module”的脆弱序号。[W2]

EQTransformer：主特征层选检测/拾取头之前的共享干路输出。当前官方源码对应 encoder → res_cnn_stack → bi_lstm_stack → transformer_d0 → transformer_d，不只取最后的 detection/P/S 三条概率。精确路径以锁定版本为准。[W3]

新增统一接口，至少输出 `station_features`、有效性 mask 和调试 metadata。维度适配仅采用预先指定的浅层方案，不做针对某模型的额外深网络。原 TEAM 的 CNN/flatten/MLP 与其他 backbone 的时序聚合可以归属特征端，但必须公开此差别；不得把这种比较误称为仅改变一个卷积/Transformer 层的纯架构消融。对时间序列特征采用同规则的 valid-mask-aware 聚合，必要维度投影后严格接回公共边界。记录原生维度、适配维度、聚合方式和各部分参数量。

原生 DiTing 路径与 FE01 包装路径若为等价重构，必须通过同权重数值回归；若为了统一输入/接口而有实际变化，另保留原生参考，并明确包装本身的影响。

### 3.3 同一截止时刻，保留各自原生窗口能力

延续当前加速度数据、物理单位、100 Hz 数据基准与相同事件划分，不做 acc→vel、不切 V01 数据。实际模型采样率、分量顺序和输入长度以锁定 checkpoint/版本为准。

建立 `capability_manifest.json`，逐模型记录：原生输入样本数、采样率、样本端点约定、预训练长度、选定特征层、经过验证的输入长度、是否只测原生长度、历史不足时的 padding 方式、共同前置噪声秒数及所支持的累积历史范围。默认长度不是某个网络家族的理论上限。SeisBench 源码中 PhaseNet 默认 3001 样本/100 Hz，并提供独立 VariableLengthPhaseNet；本任务不自动替换为后者，也不把其存在当作当前权重已验证长窗口有效。[W2,W3]

**统一时钟与样本索引**：先在原始事件时间轴上确定 T，再为所有输入站应用同一绝对截止；不能每站分别用其自身 P 拾取加 tau，导致实际上看到不同绝对时刻。输入站仅由统一的截止/到达/触发规则决定，不由某模型的未来分类输出决定。记录最后一个允许观测的样本索引 `current_sample` 以及排他的 `cutout_exclusive=current_sample+1`。新严格因果模式用整数时钟或不向未来取整的方法，记录请求 T 与实际 T；不静默把尚未到达的采样取进来。

正式主协议名建议为 `native_prefix_v2`：共同要求保留从统一参考前 h 秒至 T 的累积历史，h 延用审计后的基础值（当前配置为 5 秒）。模型保持其原生 tensor 长度；所需真实时间样本不足时在左侧 padding，最新允许观测的采样对齐右端。只要所要求的累积历史可以完整容纳，即参与该时刻的比较；容纳不了就 N/A，绝不偷偷丢掉早期历史却仍标为完整累积窗口。storage 缺测不缩短协议要求的历史长度，否则缺测多的样本会被错误视为满足长窗口能力。

容量判断必须按 `n_required=current_sample-history_start_sample+1 <= native_n_samples` 做，不能把“约30秒”直接写成 `tau<=30`。例如保留首 P 前 5 秒时，约 30 秒输入容量对应的首 P 后累积时间上限约 25 秒；6000 样本的 60 秒表述还涉及 inclusive/exclusive 边界，精确值按索引审计。与这些边界不符的 30/60/90 秒条目都不能硬填结果。后续正式表可按 tau 展示，同时显式列历史秒数与支持状态。

另实现 `native_rolling_v2` **可选且独立命名**的回放模式：每次取终止于 T 的最近原生窗口，不修改特征网络。即使首 P 后已到 90 秒，30 秒窗口仍可处理此刻之前最近约 30 秒的数据；但这不是使用累计 90 秒历史。该模式不得默认混入主表，也不能在仅按 native_prefix 训练的 checkpoint 上未经说明就当作 in-distribution 正式性能。配置必须记录是否接受过同一种滚动窗口训练；未接受时只能标为迁移/OOD 诊断。首轮默认主训练和正式评分只用 native_prefix_v2，滚动训练作为显式后续配置，不自动扩预算。

不强制所有网络共同裁短，不改宽度/感受野，不拉伸时间轴，不为短窗模型强行做多窗聚合。若未来另做等历史长度诊断，应作为独立协议，不用它抹掉较长历史能力的应用价值。

正好为 0 的有效观测不等于缺测。分开记录 storage padding、历史不足填充、future 未到达区与缺失分量 mask；上述无效区不进入幅值/均值/std/聚合。输入支持不足、无输入站、无目标、模型故障均使用不同 reason code。共同支持域里模型数值失败是 failure，不得改标“不支持”后静默排除。

**新增因果预处理审计**：当前源码 `_select_realtime_cutout` 将 cutout 定义为排他端点，而 `_get_one` 的 adjust_mean 分支使用 `:cutout+1`。静态上有将下一采样纳入均值的风险；另有截断前全记录去均值步骤。先做完整 loader→特征→预测的合成/实际可用小样本反例测试，再在所有 FE01 组共用的显式严格因果路径中修正。不能直接更改 RT55 legacy 默认。只改均值切片还不够，站点可用性、归一化、滤波、重采样和幅值支路也必须接受同一边界测试。当前 ChatGPT 只做了切片逻辑的独立合成示例，不等于真实仓库集成测试通过，更不代表已量化旧结果影响。

### 3.4 幅值与预处理

从该模型声明的实际可见历史区间内、截至 T 的真实有效且尚未归一化波形计算公共 amplitude side-channel，沿用现有可解释尺度统计/接口；共同下游边界不变，必要公共尺度拼接归入特征接口。绝不能先除以 max 再计算原始 max；不能用完整事件峰值归一化截断波形。

四组使用相同定义和单位的物理振幅侧通道；支持的历史区间不同导致统计不同是显式协议差别。在共同 prefix 历史上统计应一致。滚动短窗不得从窗外历史计算累计峰值后偷偷补给模型，除非另建并命名共同历史记忆实验。PhaseNet/EQT 的原生归一化不能不加审计地由 `annotate/classify` 黑盒完成：这些接口的窗口组织、预处理、输出都是为拾取任务设计的。[W2,W3] 主实验明确一种共同规范，例如在有效可见区间去均值、以三分量共同尺度归一化并额外保留原始尺度，不让 padding 改变归一化结果。这个选择与 checkpoint 原生训练规范的差异写入 manifest；必要时预注册一个 native-normalization 敏感性实验，而不是因错误预处理给预训练模型下结论。

测试 raw waveform 乘 10 时，归一化形状特征符合预期、log10 幅值侧通道增加 1；不要求未训练整网预测也必然严格增加 1。确认分量顺序/正负极性、m/s² 与 g、log10 与自然对数不会混用。

### 3.5 SeisBench 权重与离线运行

在独立本地环境使用官方 `list_pretrained()` 检查可用权重，优先为 PhaseNet/EQT 选择同来源数据集且无已知 Japan 验证/测试重叠的权重；不能假定某个名字一定存在。记录版本、来源训练数据与许可、模型参数、采样率、输入分量顺序、规范化、文件 sha256 与下载日期。

保存精确版本，不在正式配置里使用漂移的 `latest`。核查官方对旧权重规范化/采样率的警告。若预训练语料或时段重叠未知，列为限制，不能写“无数据泄漏”。当前 DiTing 权重也做同样来源审计。

提供本地下载/导出工具和 manifest；在超算从校验过的本地包加载，缺失文件立即失败，不尝试联网。不得在运行过程中偷偷下载或切换其他 checkpoint；不得自动接受 backbone 缺失/形状不匹配的大量 key。PhaseNet/EQT/TEAM 运行不得要求 DiTing 权重或加载 1200M 无关模型。

## 4. 数据、随机时间与评价人口

### 4.1 不变的数据约束

沿用 Japan 2000–2024 年度 HDF5、K-NET-only 与既有 event-disjoint manifest。文档数量为 train 9627、validation 1383、test 2769 个事件，实施时仍须重新审计实际过滤、数据可见性和样本数，不是留出年份测试。[R1,R2]

不得重新随机划分、混入 KiK-net、速度数据或未声明记录。归一化/参考模型/站点统计仅在对应 train 拟合；不改变最终事件 PGA 标签定义。保持 all、input、triggered-noninput、untriggered-noninput 分组与单位 log10(m/s²)。开发不读取 test 标签调参；test 仍需用户手动显式启动并记录整个项目的既往暴露情况。

### 4.2 已核验的随机时间现状

当前 RT55 配置明确 `train_time_mode=random_bins`、`train_times=null`、七个 train_time_bins、每事件每 epoch 3 次抽样及 `bin_sampling=without_replacement`。`on_epoch_end` 抽 bin，`_select_realtime_cutout` 再用区间内均匀随机数得到时间，然后换算到样本索引。固定 1/3/5/10/20/40/90 秒是 validation 设置，不是该训练路径只能看七个时刻。[R2,R7]

当前 `_encode_station_waveform` 会在 forward 中调用 `waveform_model[0](waveform)`；RT55 关闭 DPK prior cache，所以不需要先生成这七个时刻的 DiTing 波形特征。[R2,R8] 不能把“冻结参数”理解为“免做编码前向”。实际超算任务是否另有 override，必须查看其 resolved config；本次结论限定于核验的代码与配置，不代替未见作业日志。

### 4.3 FE01 V2 时间采样：1–90 秒，早期优先

在新配置显式增加概率参数并实现验证；沿用旧随机时间实现的可复用部分，缺省配置必须保持旧 RT55 行为。建议参数语义（不是声称仓库已存在这些字段）：

```yaml
fe01_protocol_version: 2
window_protocol: native_prefix_v2
realtime_training:
  enabled: true
  mode: train
  reference: first_p_pick
  train_time_mode: random_bins
  train_times: null
  train_time_bins: [[1, 3], [3, 5], [5, 10], [10, 20], [20, 40], [40, 90]]
  train_time_bin_probabilities: [0.20, 0.20, 0.20, 0.15, 0.15, 0.10]
  bins_per_event_per_epoch: 3
  bin_sampling: with_replacement
  val_times: [1, 3, 5, 10, 20, 40, 90]
```

概率表是本轮事前建议，不是调优结果：期望 60% 决策样本落在 1–10 秒，75% 在 1–20 秒，同时保留晚期覆盖。不再包含 0–1 秒。每事件每 epoch 仍抽 3 个时刻；按一次抽样所需概率独立选 bin，再在其中随机取时间。采用 with_replacement，避免把加权不放回的 bin 选择概率误称为最后样本时间占比。允许抽到同 bin 或同一采样时刻，不通过未声明去重改变分布；需导出重复率。

实际采样在采样率允许的时间格点上进行，无需假装无限精度连续时间。明确相邻 bin 边界归属、全局 1 秒下限与 90 秒上限，以及 sample-grid 取整规则。先生成统一时间流，再选输入/查询，所有模型共享事件参考和时钟定义。

对 native_prefix 不支持整段 1–90 秒的模型，将同一个基础分布条件化到其能力域，并归一化；若能力上限穿过某个 bin，应按该 bin 原有概率密度和截去长度正确裁剪概率质量，不能缩短 bin 却保留整个原概率。条件化发生在训练抽样阶段，不是读到非法样本后静默截断为最大长度。记录每组有效上限、保留概率质量及最终分布；每组仍保持同样每事件抽样数/有效 batch/update 预算。不同训练时间支持并不逐样本完全相同，报告中明确这是原生能力比较的混杂因素，不声称纯表示因果归因。

随机源与网络初始化、dropout、DataLoader worker/DDP rank 分离；用稳定 hash 的 dataset_id/event_id/epoch/draw/stream_name 等控制。同样的 run seed 和协议必须可复现；resume 后下一 epoch 和采样序列要延续而非重复。禁止用 Python 进程随机 hash，也不能以各模型参数初始化消耗 RNG 的次数决定数据抽样。

先对采样器生成至少 100000 个不涉及真实标签的抽样检查：分布、边界、能力域条件化、不同 epoch 变化、resume、重复率和各个 bin 的期望/实测计数。真实训练另外保存实际被训练/跳过的 event/time/target 计数，不能用理论概率冒充实际数据分布。时间偏置通过采样实现，首轮不再叠加同方向 loss 时间权重，以免重复加权。

### 4.4 固定验证、共同排名与能力表

固定验证继续提供首 P 后 1/3/5/10/20/40/90 秒；在执行前用 capability_manifest 标定每个模型×时点支持与否。主排名仅基于全部候选都支持的固定时间与一一对应的 event/target ID；每个时点保留实际 events、decision rows、input/target counts。共同支持集合确定后锁定，不由表现决定。

长历史表只列具备能力的模型，缺项标明 `unsupported_history`。不支持不等于预测失败。较晚时点的更好分数不能拿来与另一个模型早期分数排名；也不得各组分别对自己全部支持时间平均后作总榜。输出共同域分数和每模型完整能力域分数，两种表含义分开。

对固定 evaluation event/time，统一 input eligibility、normal/random seed、query universe 与目标角色；窗口适配后因缺测不足产生的差异须显式导出。若共同域仍有无法配对样本，给出交集统计以及各组独立覆盖/失败统计，不能仅删掉困难样本。

另提供固定 seed 的非标准时刻 validation manifest（例如每个 validation event 3 个时间格点随机时刻，排除七个固定值），用于检查时间插值/泛化，不用于扩大 checkpoint 搜索。该 manifest 在正式运行前锁定；能力域筛选规则与主表一致。固定时点与随机时点分数分开，不合并当作独立事件样本。

### 4.5 时间参考与回放边界

主协议保持首 P 参考以便项目对照；另提供 origin_time+T 回放，用于用户提出的发震后区域图。二者分开命名、分开作图。首个 P pick 是历史目录到时还是当时已发出的在线 trigger，必须标明；没有真实在线检测器/网络到达日志时，最多称为理想触发/零通信延迟的因果波形回放，不能称为上线系统实测。

随机经过时间 tau 不等同于额外随机切掉过去窗口。native_prefix 下只随机改变截止时间，历史起点由协议固定；native_rolling 下由 T 与原生窗长确定起点。首轮不再独立随机另一个历史长度 L，避免新增变量。

### 4.6 实时回放实现与验收

新增独立回放入口，默认在预选 validation 案例上从首 P 后 1 秒逐秒推进到 90 秒；不插值七个预测点制造“实时”动画。每帧都从此刻实际允许的原始数据重新构建输入，输出同一事件最终 PGA 的条件分布。默认 native_prefix，短窗超过支持域即如实结束/标 unsupported，其他模型继续；可选 native_rolling 单独输出并标训练是否匹配。

每个 case 的 query 集合固定，输入集合随真实可用性更新。normal 模式按统一到达/触发/容量规则纳入台站；random 几何回放可固定一次站点优先级与数量规则，随可用集合逐渐纳入。禁止每帧独立重抽一套随机站点而把采样跳变归于模型预测不稳定。target 类型可随时间改变，评分必须保留当帧实际角色。

同一 T 可复用 encoder/context 供许多 query chunks；跨 T 不得直接复用上一帧完整 token 或把完整事件特征裁短，除非证明与新的因果全量前向数值等价。双向/全局上下文 encoder 允许在已经到达的窗口内计算，但这不意味着它有精确的流式增量状态。首轮优先验证每帧全窗口重算的正确性，不为了速度改网络结构。

测试每帧输出与独立执行同一截止的离线因果 forward 一致。未来扰动从原始记录及 metadata 边界开始，而不是从已做过全记录归一化的 tensor 开始。保留真实零值、record-start、掉包、触发恰在截止点/下一点、首 P 参考变化、tau 小数秒和窗口上限的边界案例。

导出每帧 requested/actual decision time、latest received sample time、history start/end、valid/pre-P/post-P seconds、输入 IDs、每个 query 的分布与误差；统计整事件编码/下游/全网格耗时、P50/P95、吞吐、显存、首次加载时间。结果实际可用时刻要加数据等待与计算延迟，不能把标为 T 的离线预测默认为 T 时刻已经发出。若一次整区域推理超过 1 秒，就不宣称支持 1 Hz 整区域实时更新；硬件不可见时只交测量工具。

每帧不强制 PGA 均值单调增加，也不强制概率 sigma 逐帧缩小；最终目标未变，但新增信息可能修正判断。时间跳变、coverage 与误差随时间的变化是诊断，不直接新增平滑/单调性 loss。

## 5. 常规指标与统计

PGA 真值为现有规范的事件最终峰值，不是截断到 T 的局部峰值；核验分量合成方式、标签截止区间与 `log10(m/s²)` 定义。报告 MAE、RMSE、R²、bias、回归 slope、相关性与 truth-prediction 密度图，同时提供线性物理单位的辅助解释。定义在 log 坐标/线性坐标的 mean、median、mixture expectation，不能混淆。

概率评价使用完整 MDN：NLL、CRPS（可复现实现）、既定 PGA 阈值 Brier、可靠性曲线、预测区间 coverage 与宽度。区间按 mixture CDF 求分位数，不把混合各分量 σ 的平均当作整体置信区间。报告无效预测数，禁止筛除后只展示漂亮指标。

所有关键指标按时间、输入数、震级、距离、PGA 大小、padding、目标类型分桶并列样本数。只在样本量支持时解释极端事件/高 PGA；不能把平均优势推广到罕见强震。

按 event ID 做配对 cluster bootstrap，建议 5000 draws，保留同一事件下所有时间/台站相关性；不要把几十万 target 行当作独立样本。每个 seed 单独报结果，另报跨 seed 一致性；不把 event bootstrap 当成已经覆盖优化随机性。检验优胜时报告效果量和 CI，预先指定主对比（DiTing 对三组、以及 amplitude-only），其余分桶标为探索性并注意多重比较。

资源报告按 native_prefix/native_rolling、历史长度和共同截止分开，含全部/可训练/encoder/adapter/公共下游参数、权重大小、显存、吞吐、冷启动和固定 n_input/T 下端到端延迟；同一硬件实测，不能用参数量臆测推理速度。

## 6. 场地相关学习：定量证据链

### 6.1 两种能力不能混写

A. 已见台站的新事件：是否恢复跨事件可重复的放大/衰减差异。它可能包含位置记忆，但仍是有用能力。

B. 从未训练的台站/区域：是否在没有该目标点波形和标签参与训练时泛化。只有 A 通过不能声称 B 已通过。

先实现 A 的完整评价；另提供 B 的独立空间留出训练/评价配置。B 不能只对原来见过这些台站标签的 checkpoint 加一个 test mask：留出台站要从该实验所有训练事件的输入和目标两种角色同时移除，训练统计也必须重算。测试目标与当次输入台站排除重合。验证与测试空间块分开，留出规则仅用站点位置/覆盖等非标签信息预注册，并说明区域 buffer 与邻近站处理。事件也必须与训练互斥。

为控制资源，先交付 B 的所有四组配置但默认不提交；可先做一 seed 的空间 pilot，正式三 seed 至少比较 DiTing 与 validation 预选的最强竞争者。没有在 B 上运行的模型不得出现在“空间泛化排名”的结论中。

### 6.2 参考衰减与可重复台站项

只在训练数据拟合固定参考关系，例如：

```text
y_es = f(M_e, R_es, depth_e, 已允许的震源项) + event_effect_e + site_effect_s + residual_es
```

先建立无 site term 的距离/震源参考，再建立含训练站点随机效应的参考。距离类型、震级定义和拟合方法写清楚；若调用已有适用日本的 GMPE，必须核验单位、适用范围和所需输入，不能随手选择一个海外 GMPE 当“真实衰减”。诊断中使用目录最终 M/震源位置时，标为 retrospective/oracle reference，不得把它当作与波形模型相同实时信息预算的竞争模型。

对 held-out 事件计算观测与模型的参考残差：

```text
r_obs(e,s)  = log10(PGA_obs(e,s))  - f_train(M_e,R_es,depth_e)
r_pred(e,s) = log10(PGA_pred(e,s)) - f_train(M_e,R_es,depth_e)
```

进一步在同一 event/time 的同一目标支持集上分别去事件均值，比较事件内部空间差异，避免一个事件整体偏高/低被称为场地效应。跨 held-out 事件聚合台站残差，报告每站事件数、分布、相关性、MAE、符号/排序恢复率与置信区间；建议正式台站统计至少 10 个独立事件，并另外报告稀疏站点而非静默删去。稳定性检查应考虑震中方向、距离与强弱地震分桶；场地响应可能随强度变化。

测试标签推导的去均值/站点残差仅用于评分，不得回填给网络或用于更新参考模型。若用 shrinkage/mixed-effects，其参数在 training 上确定，不能从 test 拟合 site correction 再声称模型自主学会了它。

这些仍是“场地相关或台站特异的可重复残差”，不是地质因果分解的唯一识别：震源辐射、传播路径、方位、仪器和场地信息可能混合。[W5,W6] VS30/地质类别可作为独立外部验证和分桶，不能只给某一模型额外输入；与 VS30 简单相关不能代替残差恢复检验。

### 6.3 局部对比和空间形状

实现同一事件内近似等震源距离的站对比较。站对规则/距离容差在训练/validation 上固定，对所有模型一致；比较观测与预测的 logPGA 差值、符号、误差与排序。这有助于避免仅靠径向衰减获得高分，但单个站对仍可能含路径/方位差异，不作唯一场地归因。

复用或提供 event-level pairwise-delta MAE、空间 range/std 恢复比和去均值场误差；同时看绝对误差。预测范围扩大并不自动正确，不能通过加噪声或过度空间起伏“改善”range ratio。真值空间差异过小时对比值指标做事前定义的稳定处理，并列被排除的数量。

辅助机制检查允许做 waveform-station roll、坐标/波形置换、频带或幅值扰动，但这些是可能 OOD 的敏感性诊断；必须同物理规则、同随机种子实施，不能把置乱后变差直接等同于真实的场地泛化。

## 7. PGA 区域地图与动画

先做数据支持充分的地震影响区；日本全国图作为带覆盖标识的展示，而不是默认所有格点都可信。目标格点仅提供协议允许的查询坐标/静态公共变量，不能伪造目标波形。禁止为格点分配最近台站 ID 来冒充未观测位置预测，禁止把台站预测插值图称为逐格点模型推理。

若当前基础下游的多查询交互会使预测依赖查询集合，必须检测并明确：同一目标单点/不同 chunk/不同顺序推理结果一致到容差，才能将分块结果视为同一个连续场。否则在不偷改公共下游结构的前提下，对每个 query 独立计算或使用固定且公开的查询集合，并标明限制。

图上保留当前可用输入台站、未输入观测台站、震源位置（回放可用）、时间定义、有效数据覆盖和不支持区域。经纬度转米制距离/规则空间网格时使用一致地理算法；不能把一度经纬当作等距。网格间距是显示分辨率，不是经验证的场地分辨率。

每个预选事件/时间至少输出：输入与观测站点；真实 PGA 的站点散点；模型逐格点预测及站点预测；固定衰减参考；相对参考的预测残差；站点观测残差与预测误差；预测分位数/不确定性及覆盖 mask。不同模型同一时间使用同一色标、范围、网格与站点集。不得绘制未经观测的“连续真实 PGA”并据此评分。

案例选择规则在看模型结果前固定：从相应 split 依据震级、地理覆盖、有效站点数、事件类型等选 3–5 个有代表性的事件，保存 selection manifest。不得挑选最漂亮或 DiTing 恰好最好的个例。输出多时刻静态图和第 4.6 节逐帧真实推理结果驱动的动画；不支持的模型/时点清楚留空，不能复制最后一帧伪装更新。动态图旁标明“各帧都预测同一个最终 PGA 目标，只是输入截止时间在变”。

没有连续地面真值时，定量分数只在真正 held-out 的观测点计算。禁止用同一批站点标签插值生成伪真值后评分。MDN 不确定性是模型条件分布，不等于完整域外认知不确定性；域外/稀疏区单独遮罩，不凭平滑漂亮的图证明可信。

## 8. 必须新增/更新的测试

测试必须覆盖：配置继承解析和 opt-in registry；四 encoder 的输入/输出 shape 与分量顺序；TEAM 数值/轴/尺度逻辑；官方 native 长度中间特征一致性；能力域与 N/A 判定；排他 cutoff 与不向未来取整；storage/future/missing masks；真零与缺测；all-empty 输入；未来扰动不变；幅值×10；frozen 参数与 BN 状态不变；trainable 参数进入优化器且收到梯度；不同 run 的公共初始 state 相同；checkpoint 重载数值一致；无缺失权重 silent fallback；非 DiTing 不加载 DiTing；随机时间概率/条件化/epoch/DDP/resume；评价样本配对；逐秒回放与独立同截止前向一致；query chunk/order 不变性；MDN 单位/分位数/阈值换算；事件/台站泄漏检查；旧 RT55 路径回归。

做一个极小合成可学任务：固定输入波形的可见振幅/形状对应连续标签，验证 TEAM scratch 与公共下游能下降；对于大 DiTing，本地资源不足时分清 meta-shape test、mock test 与真实权重 forward，不能用 mock 通过代替真实预训练加载成功。真实权重的大模型 smoke 留给用户在超算 audit 作业手动执行。

每项测试记录命令、退出码、样本类型、资源和结果。执行 `bash -n`、必要的 Python compile/import、聚焦 pytest；未执行项明确列出理由与超算复现命令。

## 9. 超算提交脚本：全部由用户手动运行

### 9.1 新建独立脚本与参数

建议产物（可按仓库规范小幅调整，但必须给准确最终路径）：

```text
configs/fe01/                         # 协议/模型/seed/空间划分配置
fe01_cluster.env.example             # 环境、数据、权重和调度参数
scripts/fe01/download_weights.py     # 本地下载并生成离线清单
scripts/fe01/prepare_runs.py          # 解析协议生成 runs.tsv，默认不提交
scripts/fe01/audit_job.sbatch
scripts/fe01/train_array.sbatch
scripts/fe01/eval_job.sbatch
scripts/fe01/site_map_job.sbatch
scripts/fe01/replay_job.sbatch       # 用户手动执行的逐秒回放
scripts/fe01/print_submit_commands.sh
scripts/fe01/pack_source.sh
scripts/fe01/pack_results.py
docs/ai/FE01_HPC_RUNBOOK.md
```

已有 launcher 使用 Slurm、ROCm/DTK 模块和特定分布式方式，并含主线 WORKDIR 与 DiTing 配置/权重默认；新脚本必须审计后适配，不能替换为假定 CUDA/NVIDIA 的万能模板，也不能原封不动继承会强制查 DiTing 权重的路径。[R5] 保留用户现有可用的 DCU/DDP 启动经验，不在这一支线顺便重构分布式训练。

ENV 至少包括 FE01_CODE_ROOT、FE01_OUTPUT_ROOT、DATA_ROOT、各模型本地 checkpoint、独立环境、module loads、partition/account/qos、nodes、devices_per_node、cpus、memory、walltime。未知账户和资源标为待用户填写并 fail-fast，不编造。本地依赖约束须兼容超算既有 PyTorch/ROCm，不能 `pip install -U` 覆盖已工作的框架。

所有作业必须默认不覆盖：`RESET_WEIGHT_PATH=0`，无自动删除 checkpoint，无 AUTO_SBATCH 自动重提交流程；由用户显式执行 `sbatch`。输出路径解析成绝对路径并验证落在 FE01_OUTPUT_ROOT 下；目录已存在且非显式 resume 时拒绝启动。resume 校验 run identity、数据/权重/协议 hash 和 checkpoint 状态，不能把上一个实验的 optimizer 当作新训练。

作业启动时写出 host/job/device、Git SHA/源码包 hash、Python/PyTorch/SeisBench/ROCm 版本、resolved config、checkpoint SHA、样本配对/划分 manifest 与实际命令。训练保存 best/last、optimizer/scheduler/RNG 与足够的轻量诊断，合理的断点保存；遇到异常让 Slurm 失败，不在异常后仍打印成功。

### 9.2 手动提交与 dry-run

`prepare_runs.py` 和 `print_submit_commands.sh` 只创建配置/目录并打印命令，不调用 sbatch/srun，不发起训练。提前建立 Slurm log 目录：不能指望作业正文 mkdir 来修复调度器打开 stdout 时不存在的目录。不要在 `#SBATCH` 中使用不会展开的普通 shell 变量；动态参数通过用户的 sbatch 命令传入或事先渲染经过检查的脚本。

Runbook 给出真实可运行的逐步命令：本地下载/校验权重 → 打包专用支线源码 → 用户同步到超算独立目录 → 填写 env → CPU/设备 audit → 单 run pilot → 正式 array → validation 汇总 → 经协议确认后显式 test → 空间留出实验 → 地图/逐秒回放/结果封装。说明如何手动查询日志、失败重试与严格续训。列清 0-based array 索引到 model/seed/protocol 的映射，禁止 off-by-one 和不同数组任务写同一目录。

测试默认不打开：evaluation 入口默认 validation，test 需要用户手动给出显式开关和冻结协议标识。地图 split 与 metrics split 必须明确；不要提前为“展示”查看 test 选模型。

## 10. 证据封装与交接

在专用分支提供研究协议、实现说明、完整配置、测试、脚本、manifest schema 和分析程序。不要把私人绝对路径/凭据、原始波形或巨型 checkpoint 直接推入 Git。正式运行后应同步供 ChatGPT 独立审计的轻量证据，而不只是摘要和图片。

至少生成：

```text
reports/fe01_<runset>/
  README.md
  protocol.lock.json
  resolved_configs/
  runs_manifest.csv
  data_split_audit.json
  pretrained_manifest.json
  model_interface_audit.json
  capability_manifest.json
  model_time_support.csv
  sampler_distribution_audit.json
  actual_training_time_histograms.csv
  causal_boundary_audit.json
  random_validation_times_manifest.csv
  replay_latency.csv
  replay_metrics_by_time.csv
  downstream_initial_state_fingerprints.json
  test_exposure_ledger.json
  training_curves.csv
  metrics_by_time_target.csv
  paired_event_bootstrap.csv
  efficiency.csv
  site_residuals.csv
  station_counts.csv
  spatial_holdout_manifest.csv
  case_selection_manifest.csv
  figure_data/
  figures/
  artifact_manifest.sha256
```

另导出逐 event/time/query 预测文件（可在外部存放压缩 NPZ/Parquet，不必硬塞 Git），至少包含 event_id、station_id/坐标、时间参考、requested/actual decision time、排他 cutoff、历史起止/长度、window protocol、supported/status/reason、split、seed、输入目标角色、输入站 ID 列表或其可还原 manifest、真值、点预测、完整 MDN 参数、valid mask、padding/有效秒数、单位、checkpoint/配置 hash。大型结果不能进 Git 时，给用户明确的打包路径和可上传审阅包；保留可复核的事件级指标、固定样例与图源数据在仓库。

场地分析与地图须有原始站点数据、逐格点模型输出、coverage mask 和作图脚本，使 ChatGPT 可以独立检查，不依赖 Codex 的文字结论。

最终 Codex 交接采用仓库 `docs/ai/README.md` 的既有格式，额外写明：

1. 实际 worktree/branch/base/implementation commit；原工作区保护核验结果；
2. 对照边界和公共结构指纹；与原 TEAM/原生 DiTing 的偏离；
3. 已真实执行的测试及未执行的训练/设备测试；
4. 尚缺数据/权重/账户字段及对应的 fail-fast 检查；
5. 用户在超算应手动执行的准确命令、pilot/正式训练规模、各组时间支持域及逐秒回放命令；
6. 将来必须回传哪些文件，哪些只在超算留档；
7. 不得写“已达到前沿”“已学到真实场地效应”“全日本预测可信”，除非有对应证据；当前只交付代码与可运行方案。

完成本轮后停止于等待用户手动训练的工程交接点，不自动追加结构搜索、速度数据实验、更多特征模型或论文写作。

## 11. 本任务依据与事实边界

以下仓库事实由 ChatGPT 于 2026-10-01 通过 GitHub 读取；本地/超算实时状态仍须 Codex/用户核验。条目不是对代码运行正确性的背书。

- [R1] `docs/ai/PROJECT_CONTEXT.md`：RT55/RT56 基础架构、Japan 2000–2024 K-NET-only split、log10(m/s²)、all 与 non-input 区分。
- [R2] `pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json`：继承配置、first_p_pick 时间表、storage-valid mask、KNET 数据与 transfer_model_path。
- [R3] `gemini_models.py`：`NormalizedScaleEmbedding` 的静态疑点，以及 RT61/RT59 的附加残差路径。
- [R4] `diting/config/diting_1200m_backbone_attnpool.yml`：真实使用的 MAE/1200M/backbone_attn_pool 配置与 10000 样本输入。
- [R5] `train_light_slurm.sh`：Slurm/ROCm 环境、固定主线默认路径、DiTing 依赖与执行行为。
- [R7] `gemini_util_light.py`：`_normalize_realtime_training`、`_select_realtime_cutout`、`on_epoch_end`，区间随机时间实现；`_get_one` 截断/去均值/右对齐及不支持旧 sliding_window 的限制。
- [R8] `gemini_models.py`：`FullModel._encode_station_waveform`，前向内调用 encoder 与 DPK prior cache 的区别。
- [R6] `docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md`：RT61 验证交接报告；其建议保留 RT59 reference，并非所有当前文档里的旧 baseline 都代表最新完成状态。

以上均固定于 commit `9c95dbfaf92f36b2673d816026cd3f25b91eec66`。本任务未对任何主组实际训练，也未确认超算上的权重文件存在。

外部一手参考（执行时锁定库版本并以该版本源码为准）：

```text
[W1] Git worktree 官方文档
https://git-scm.com/docs/git-worktree/2.48.0

[W2] SeisBench PhaseNet 官方实现：中间层、输入长度、归一化及历史权重警告
https://seisbench.readthedocs.io/en/stable/_modules/seisbench/models/phasenet.html

[W3] SeisBench EQTransformer 官方实现：共享干路、窗口 assert 与 encoder padding
https://seisbench.readthedocs.io/en/stable/_modules/seisbench/models/eqtransformer.html

[W4] SeisBench 预训练权重、作者原始 TEAM
https://seisbench.readthedocs.io/en/stable/pages/models/pretrained_models.html
https://github.com/yetinam/TEAM

[W5] Stewart, Afshari & Goulet (2017), Non-Ergodic Site Response in Seismic Hazard Analysis
DOI: 10.1193/081716EQS135M
https://journals.sagepub.com/doi/abs/10.1193/081716eqs135m

[W6] USGS: Local site effects and dynamic soil behavior (2001)
https://www.usgs.gov/publications/local-site-effects-and-dynamic-soil-behavior
```


## 12. 完整 AI-HANDOFF（V2）

```text
[AI-HANDOFF]
task_id: 20261001-fe01-native-window-random-time-v2
repo: rular099/team_pytorch
branch: exp/fe01-feature-extractor-site-effects
base_commit: 9c95dbfaf92f36b2673d816026cd3f25b91eec66
goal: 在独立 worktree 交付四特征网络原生窗口对照、1–90 秒早期加权随机因果训练、固定时点评价及逐秒 PGA 回放的可手动提交版本。
verified_facts:
  - 读取的 RT61 远端 HEAD 仍为上述完整 SHA；本次未读取到 FE01 result commit，不假设用户本地尚无修改。
  - RT55 config 已是 random_bins、train_times=null、每事件每 epoch 3 次；七个固定秒数用于 validation。
  - gemini_util_light.py 的 on_epoch_end 抽 bin；_select_realtime_cutout 在 bin 内随机采样并生成当前样本与排他 cutoff。
  - gemini_models.py::_encode_station_waveform 在 forward 内调用 waveform_model[0](waveform)；RT55 config 关闭 DPK prior cache。
  - 旧 realtime generator 对 sliding_window=True 有显式不支持条件，不能仅打开此旧开关声称已实现 native rolling。
  - adjust_mean 分支的 :cutout+1 与排他 cutoff 定义存在一采样前视风险，尚未执行真实仓库端到端反例。
inferences:
  - 不必为了固定缓存把训练限制到七个时刻；已有机制可以扩展到新的时间密度。
  - 不同原生历史能力/训练支持的比较是部署方案比较，不是纯 encoder 架构归因。
  - 任意时间能做离线推理不等于具备达标的实时吞吐；必须实测含所有 query 的整帧延迟。
files_to_inspect:
  - AGENTS.md; docs/ai/PROJECT_CONTEXT.md; docs/ai/README.md; SESSION_SUMMARY.md
  - gemini_util_light.py; gemini_models.py; train_light.py; eval_checkpoint.py
  - RT55/RT56 配置的完整继承链与真实 resolved config
  - 锁定 SeisBench 版本的 PhaseNet/EQTransformer 源码、权重 metadata
  - tests/test_causal_random_geometry.py 与现有 sampler/mask/regression tests
  - 原 V1 已实施的文件、diff、FE01 当前 HEAD 和运行状态（如有）
constraints:
  - 保持 RT55 原配置、checkpoint keys/shapes、默认 sampler/预处理和推理行为；严格因果修正在新 FE01 opt-in 路径。
  - 不切换/覆盖原工作区，不 reset/clean/stash，不混入 V01，不覆盖已有输出。
  - 原生窗口，不强改网络长度/时间尺度/多窗聚合；长历史 N/A 与数值 failure 分开。
  - 公共下游结构、初始化规则、标签、loss、split 不变；差异只在已声明前端/时间协议。
  - 训练、设备审计、正式验证/test/回放均由用户手动提交；本地仅允许有限 smoke 与测试。
proposed_change:
  - 新增/更新 configs/fe01/*：协议版本、窗口能力、加权随机时间、共同验证支持集合、严格因果预处理开关。
  - 新增独立 capability/window builder：同一绝对 T 构建各站历史和 masks，native_prefix 主协议、native_rolling 可选独立协议。
  - 最小 opt-in 扩展 gemini_util_light.py 或 FE01 专用 sampler：区间概率、能力域条件化、稳定 RNG、请求/实际时刻及边界修正。
  - 修改 FE01 特征 wrapper：保留各自 native shape，mask-aware 聚合与幅值统计只能使用声明区间。
  - 新增回放工具：逐秒构建输入、同 T query context 复用、独立前向一致性与整区域延迟导出；不改公共下游架构。
  - 更新评价/场地/地图：共同支持排名、长窗能力表、N/A reason、实际 events/targets、随机验证时刻和回放图源。
  - 新增 tests/test_fe01_time_sampling.py、test_fe01_native_windows.py、test_fe01_causal_replay.py（路径按仓库规范），更新 HPC runbook 和结果 manifest。
acceptance_checks:
  - 原工作区前后状态不被本任务改写；若 FE01 HEAD != base，先审阅已有 diff 再增量实施，不重置。
  - 100000 次 sampler-only 抽样审计通过；记录计数/期望/容差，不把模拟当真实训练。
  - 同一绝对 T、不越未来样本、容量 boundary、h+tau、mask、幅值、reference、缺测、全空、不同 sample rate 的测试。
  - 从原始记录扰动截止后第一个点及所有后续点，严格因果 FE01 输入、特征和预测保持不变；记录 legacy 差异但不更改 legacy 默认。
  - 固定与随机 validation 的共同支持配对、N/A/failure 区分、所有分桶 sample counts；不读 test 调实验。
  - 逐秒回放等于各截止独立因果前向；不同帧不偷偷复用全事件特征；同 T query chunk/order 回归。
  - 四模型公共下游指纹/冻结/梯度/checkpoint 重载及旧 RT55 回归；bash -n、compile/import、聚焦测试。
hpc_followup:
  - Codex 交付 audit、pilot、formal train array、validation、site map、replay、pack-results 脚本及准确手动命令；不调用 sbatch/srun。
  - 用户先执行真实权重/数据因果 audit；若不通过不得开启正式训练。
  - 先四组 seed42 pilot，再按锁定协议手动提交正式 seeds42/43/44；不自动增加滚动训练或等窗训练。
  - 保存完整 checkpoint/config/provenance、实际时间直方图、逐目标分布与 site/replay/latency 图源。
risks:
  - 原生窗口与训练域不同形成合理但必须披露的混杂；共同时间排名不能冒充纯架构结论。
  - catalog P 到时不等于实际 trigger availability；离线预处理或重采样也可能含未来信息。
  - strict cutoff 修正使 FE01 与历史 RT55 旧指标不完全可比，必须重训/同协议参考，不能回写历史结果。
  - window capacity 与 pre-P 长度可能使 PhaseNet 在首 P 后约25秒而非30秒达到累计域边界。
  - 全日本 query 计算、早期缺测和单输入空间辨识可能限制实时应用，不能由动画外观证明。
open_questions:
  - 用户填写超算 account/partition/环境/数据和权重路径；缺项交付 env 模板与 fail-fast，不编造。
  - 当前默认保留基础 pre-P 5秒、native_prefix 主协议和逐秒回放；不为扩大短窗支持域自动删除 pre-P。
  - native_rolling 的匹配训练与完全相同历史归因实验都不在首轮默认成本内，仅保留显式配置/入口。
[/AI-HANDOFF]
```

### Codex 返回要求

按 `docs/ai/README.md` 返回 `[CODEX-RESULT]`，准确填写本轮 task_id、实际 base/result commit、changed_files、verification、compatibility、hpc_status、remaining_risks、review_request。`hpc_status` 必须写实际状态；本轮未提交则写 `NOT_SUBMITTED`。对于已在 V1 基础上写过代码的目录，说明保留/撤销了哪些方案决策，以及为何不需要覆盖主线。
