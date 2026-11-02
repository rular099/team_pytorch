请实施 FE01-A01：查询几何诊断与绝对振幅消融。

先阅读随附的：
FE01_GEOMETRY_AMPLITUDE_HANDOFF_20261007.md

以下指令与该任务书共同构成本轮实施合同。不要只返回另一份规划；
请完成代码、测试、本地可执行的诊断，以及用户手动执行的超算脚本。
正式训练和大规模评价由用户在超算上手动提交，本轮不得自动执行。

[AI-HANDOFF]

task_id: 20261007-fe01-a01-geometry-absolute-amplitude
repo: rular099/team_pytorch
source_branch: exp/fe01-feature-extractor-site-effects
base_commit: ba4fa7740d9fde5fde87a7b0ae0397037209baf4
target_branch: exp/fe01-a01-absolute-amplitude-ablation
target_worktree: 新建独立 worktree，建议兄弟目录 team_pytorch_fe01_a01

goal:
  完成三个可验收目标：
  A. 定位单输入站下查询坐标的信息在哪一段被削弱，
     区分位置编码、注意力选择、query residual、后续读出与输出头。
  B. 核验所有显式振幅信息只依赖当前决策时刻可得的有效波形，
     并清楚区分下游前缀因果性与源头离线预处理的因果性。
  C. 在不改变特征提取之后的模型结构、采样和损失的前提下，
     准备 TEAM、PhaseNet、EQT、DiTing 的绝对振幅 ON/OFF 对照，
     交付训练、评价与证据导出的手动超算脚本。

division_of_work:
  - ChatGPT 已确定研究问题、对照定义、评价方法和验收边界。
  - Codex 负责核验本地事实、实施、测试、小规模可执行诊断、
    提交代码和整理证据。
  - 用户负责超算资源确认及正式作业提交。
  - 生产数据、checkpoint 或设备不可见时，标明 NOT_RUN/BLOCKED，
    继续完成其余可执行部分，不虚构结果。

verified_facts:
  以下来自 base commit，实施前须结合实际 resolved config 复核：
  - FE01 使用地理坐标 sin/cos 编码，不是台站序号编码。
  - 当前 use_coords_abs=true、station_context_mode=off、
    pga_readout_mode=target_cross_attention、readout_first_residual=true。
  - 只有一个有效 key 时，纯 attention 聚合不随 query 改变；
    但 query residual 仍在，整个模型不是数学上必然恒定。
  - 当前 use_amplitude_info=true。
  - shape_and_scale 旁路共11维：
    3维log10 std、3维log10 RMS、3维log10 peak、
    1维log10总体peak、1维log1p有效秒数。
  - FE01 正常路径先截取可见前缀，再计算归一化和幅值统计。
  - 上游离线滤波/重采样的在线因果性尚未认证。
  - 已有 TEAM/PhaseNet/EQT 的 FE01 amplitude-ON 训练。
  - 目前没有匹配的、新训同公共下游 DiTing ON/OFF 对照；
    RT55/RT61不能直接代替这对实验的ON组。

inferences:
  - 单台空间近常数可能涉及 query residual 利用不足、
    分支尺度、读出优化或信息不足，不能预先归罪于位置编码。
  - 多台输出空间变化可以来自对带坐标台站特征的不同加权，
    不证明模型一定先推断出了震源位置。
  - 去掉显式绝对振幅后，模型仍可使用归一化波形形状；
    不能把它称作“完全没有幅值形状信息”。

files_to_inspect:
  按顺序阅读：
  1. AGENTS.md
  2. docs/ai/PROJECT_CONTEXT.md
  3. docs/ai/README.md
  4. SESSION_SUMMARY.md
  5. 随附任务书
  6. 本任务直接相关文件：
     - fe01/model.py
     - fe01/data.py
     - fe01/windows.py
     - fe01/extractors.py
     - fe01/config.py
     - fe01/engine.py
     - fe01/metrics.py
     - gemini_models.py
     - fe01_review/runners.py
     - fe01_review/checkpoint_inventory.py
     - reports/fe01_hpc_results_20261004/metadata/
     - reports/fe01_eval1_hpc_gates_20261006/
     - 相关原训练配置、launcher、tests、checkpoint与provenance

  gemini_models.py重点追踪：
     PositionEmbedding
     CrossAttentionReadout
     CrossAttentionRefinementBlock
     FullModel.forward
     _station_coord_features
     _pga_coord_embedding
     _station_memory_for_readout
     振幅投影、门控、PGA读出及输出头

constraints:
  1. 工作区与协作
     - 先记录原工作区HEAD、status、staged/unstaged diff。
     - 新建独立worktree，不在FE01-EVAL1或V01工作目录切换分支。
     - 同名目录/分支已存在时先检查，不覆盖、不强制重建。
     - HEAD与base不同先审查差异，不机械套用，不reset/clean/stash。
     - 不与FE01-EVAL1的Codex会话同时修改同一文件。
     - 本支线不承担FE01-EVAL1的TEAM gate修复和正式复评。

  2. 兼容性
     - 不修改原115文件训练快照，不修改已有config/lock/checkpoint。
     - 不覆盖任何旧训练、失败或评价目录。
     - 不删除SHA校验，不重签旧lock绕过检查。
     - 不修改原RT55/RT61默认加载和推理行为。
     - 不在本轮更换位置编码、增加几何解码器、改变读出结构。
       本轮对几何先诊断，振幅消融只改变输入信息策略。

  3. 执行边界
     - 不自动sbatch/srun，不启动正式训练或完整生产评价。
     - 允许本地聚焦测试、合成单步smoke和小规模诊断。
     - 不访问held-out test，不用test选择任何参数或实验定义。
     - 不自动下载权重，不重装torch或更换已工作的ROCm/DTK环境。
     - 不重跑旧MPI恢复脚本。
     - 私有路径和账户从已有本地资料/环境变量解析，不写入Git。

proposed_change:

  一、独立代码与来源管理

  新代码优先放：
     fe01_a01/
     scripts/fe01_a01/
     configs/fe01_a01/
     tests/test_fe01_a01_*.py
     docs/ai/FE01_A01_*.md

  使用独立入口和显式注入/继承实现。
  不通过全局monkeypatch改变原训练模块行为。
  原训练源码身份和A01新增源码身份分别记录。

  新loader必须按实际模型家族验证预训练依赖：
     - TEAM scratch不要求不存在的预训练清单SHA。
     - pretrained模型必须有合法非空SHA并严格匹配。
     - 不写成“SHA为空一律跳过”。
     - 未知家族或身份矛盾必须失败。
  不为修复上述条件而改动FE01-EVAL1正在使用的runners.py。

  二、阶段A：不训练的查询几何诊断

  A1. 固定探针
     - 从锁定validation请求中，按input_count、时间和目标数选样本；
       不按模型误差选择。
     - 覆盖实际K=1与多台，K是有效key数，不是padding后的槽数。
     - 每个预定分层首轮最多32个decision，缺项报告实际数量。
     - 保留event/T/输入站/查询站/原生窗口的身份清单。

  A2. 逐层追踪
     固定波形、幅值、输入坐标、mask与时间，只改变查询坐标，
     导出：
       - 输入地理坐标与位置编码；
       - query进入attention之前的表示；
       - 纯attention输出及权重；
       - query residual合并后的输出；
       - 各层readout表示；
       - 最终MDN weights、mu、sigma与预测均值。

     计算跨query变化、各分支范数、实际训练后的gate、
     查询坐标到最终输出的敏感性，以及公里尺度有限差分。

     梯度诊断必须经过真正可微路径：
       - model保持eval、参数不更新；
       - 必要输入requires_grad=true；
       - 不被现有runner的no_grad、detach或缓存切断。
     同时用有限差分复核，不能只相信一次自动微分结果。

     读取实际checkpoint gate，不能用初始化配置冒充训练后值。
     区分均值近常数与整个预测分布近常数。

  A3. 控制实验
     - station顺序等价控制：波形/坐标/mask同步置换。
     - 坐标单独置换只作为机制干预，不当作真实预测评分。
     - 对同一event/T，从可用台站构造嵌套K=1/3/5/8集合。
     - query排除最大输入集合，保证K对照使用相同目标人口。
     - 台站不足时报告不支持，不把T向后移动来凑台站。
     - 不制造重复台站加入真实评分。

     需要回答：
       编码本身是否区分查询？
       这种差异在哪一层明显减小？
       单key退化之外，query residual是否仍影响最终PGA？
       振幅分支是否与几何敏感性减弱同时出现？

     相关现象只能提供诊断证据，不能直接写成唯一因果机制。

  三、阶段A：振幅可得性与泄漏检查

  A4. 追踪完整链路
     dataset/sampler
       → 绝对截止时刻
       → 原始波形读取与mask
       → 去均值、归一化、幅值统计
       → encoder和adapter
       → 幅值旁路
       → readout与MDN
       → loss/evaluation export

     全事件最终PGA只允许作监督或评分标签，
     不得用于前缀归一化、输入幅值统计或forward条件。

  A5. 未来扰动
     固定标签、坐标、picks、mask规则、请求和随机种子，
     将截止后一采样及后续波形改为大脉冲、NaN、Inf。
     逐项检查：
       selected IDs、received mask、去均值统计、
       归一化peak、前10维幅值、有效秒数、
       encoder特征、MDN和概率输出不变。

     覆盖触发边界、原生容量边界、尾部缺测、内部合法零、
     invalid station，不只检查单个1秒样本。

     正控制：修改最后一个合法样本，确认相关输入统计能发生变化；
     不要求任意checkpoint的最终预测都必须变化。

     有原始预处理代码和波形时，再检查：
       “先修改未来原始波形 → 再滤波/重采样 → 再截前缀”。
     否则明确标记UPSTREAM_CAUSALITY_UNKNOWN。
     不用HDF5之后的测试证明HDF5生成之前无未来依赖，
     也不因此自动开展全数据转换。

  四、阶段B：实现唯一的绝对振幅ON/OFF变量

  新增显式策略：
     physical_scale_keep_duration
     no_absolute_scale_keep_duration

  ON：
     完全保留原11维旁路。

  OFF：
     前10维绝对幅值统计固定为0；
     第11维log1p有效秒数保留；
     保持11维形状、projection、gate及下游参数结构不变。

  不用use_amplitude_info=false代替主要OFF：
     该方式会把有效时长一并去掉，造成第二个变量。

  ON/OFF必须使用相同归一化波形：
     不改变joint peak归一化、分量比例、频带、
     padding、时间采样、台站选择、目标选择、损失和辅助任务。

  检查不存在绕过屏蔽的raw-energy、DPK、temporal或anchor路径。
  RT61若作为额外诊断对象，要单独列其幅值依赖；
  未追踪全部旁路前，不能称其为“无幅值RT61”。

  A01模式必须记录到resolved config和checkpoint；
  原状态字典的参数命名/形状不得因ON/OFF发生漂移。

  五、尺度不变性测试

  固定metadata、mask、station selection、T，对每台站的三个分量
  共同乘正数c_s；测试整事件共同增益及逐站不同增益。

  例如使用0.1、1、10，正常信号避免进入eps/clamp饱和区；
  低能量极端情况另测并报告。

  预期：
     - normalized waveform在数值容差内不变；
     - OFF交付给下游的特征及最终MDN不变；
     - ON前10维log10幅值按log10(c_s)平移；
     - duration不变。

  不要求随机模型的最终ON输出必然敏感。
  不把人工逐站增益当成物理一致的新地震来评分。

  明确区分：
     inference_intervention：
       对原ON checkpoint在推理时屏蔽幅值，只诊断模型依赖。
     trained_ablation：
       从训练开始就屏蔽幅值，评价无绝对尺度的可学习能力。

  六、首轮训练矩阵：只准备seed42，不自动运行

  家族              ON                            OFF
  TEAM              尽量复用已有FE01 ON           新训1组
  PhaseNet          尽量复用已有FE01 ON           新训1组
  EQT               尽量复用已有FE01 ON           新训1组
  DiTing            新训匹配的FE01 ON             新训1组

  合法复用旧ON时，新训练共5组。
  不默认扩展seed43/44，不默认加amplitude-only或架构搜索。
  复用失败则报告原因与替代方案，不悄悄增加训练数量。

  旧ON复用必须核验：
     - 新ON前向与旧FE01等价；
     - encoder、adapter与公共初始权重一致；
     - 原train cohort、目标标准化、采样计划与预算一致；
     - optimizer、loss分母、scheduler、drop-last与DDP语义一致。
  仅相同global batch，不足以证明训练可比。

  优先读取已保存init.pth恢复初始模型状态。
  缺失时只能重建并核验指纹，不凭seed相同声称等价。

  OFF不能从ON最终checkpoint微调。
  DiTing使用核验后的预训练encoder，ON/OFF的公共下游和adapter
  从同一初始状态开始；不拿RT55/RT61作为DiTing ON组。

  保留：
     - PhaseNet/EQT/DiTing encoder冻结且保持eval；
     - TEAM原scratch训练策略；
     - 每家族自己的原生窗口能力；
     - 同一家族ON/OFF完全相同的event/epoch/draw计划；
     - 12 epochs、每事件3 draws、global batch128；
     - cohort为9084事件时，每epoch212 updates，总2544 updates。
  实际计数必须由程序核验，不能硬填。

  选checkpoint仍用原validation共同十cell noninput MAE，
  同分选较早epoch；另报告统一epoch12。
  不重新选择旧ON的epoch来迎合新OFF结果。

  七、评价与解释

  主评价：
     validation；
     首P后1/3/5/10/20秒 × normal/random；
     每cell目标内平均，再十cell等权；
     PGA单位log10(m/s²)。

  原正常人口参考：
     1310 events；
     194265 all-target records/run；
     139440 noninput records/run。
  这些数只作核对目标，必须验证完整key/label/request指纹。
  不一致时输出差异，不能静默取交集后仍称原协议比较。

  联合报告：
     all / noninput / triggered_noninput / untriggered；
     实际K=1与多台；
     MAE、RMSE、R²、bias、slope；
     精确MDN NLL、CRPS、Brier、predictive sigma、
     分位数coverage和interval width；
     level MSE、shape MSE、P95-P05范围恢复、
     近等距离站对差值误差。

  不能因为动态范围变大就判定改善。
  不能把不同模型不相同的长时间支持域平均后直接排名。
  bootstrap以事件为簇，报告实际事件/目标/field数量、
  空cell比例和有效重复数；稀疏子群只作相应强度的解释。

  特别区分：
     OFF是否只是整体level变差？
     空间shape是否同时变差、保持或改善？
     单台与多台、早期与后期是否表现不同？
  不把单seed结论写成最终跨seed结论。

acceptance_checks:
  - 新worktree与原工作区隔离，原115文件指纹不变。
  - A01新增源码及依赖有独立manifest和SHA。
  - 原RT55加载/推理聚焦回归不被改变。
  - ON与原FE01的前向、初始状态和小型训练更新等价。
  - ON/OFF维度和参数结构一致，OFF只去10维绝对幅值。
  - 有效时长、mask、归一化波形在两组一致。
  - 逐站增益不变性与未来扰动测试通过。
  - 查询编码、纯attention、residual、最终MDN分阶段诊断。
  - 专门测试新loader：
    TEAM null预训练SHA按合同通过；
    pretrained缺SHA/错误SHA失败；
    错epoch、错encoder、错config、错训练源码失败。
  - 实际执行Python编译/import、聚焦pytest、bash -n。
  - fake scheduler测试确认准备流程不调用sbatch/srun。
  - 缺生产资源的检查写NOT_RUN，不使用合成结果代替。

hpc_followup:
  交付用户可手动执行的stage：
     environment
     identity/audit
     diagnostics
     pilot
     train
     eval
     pack

  准备脚本只检查与打印命令。
  用户显式调用的提交入口，每次只能提交指定stage，
  不自动串联正式训练、重提交或开启test。

  保留现有可工作的DCU/ROCm环境。
  每个stage明确：
     在本地、登录节点还是计算节点执行；
     输入路径、资源参数、前置条件、输出路径；
     失败如何定位、如何使用新run ID恢复。

  大权重前向不在登录节点执行。
  大数据和原权重只读复用，不要求重复打包上传。
  提供独立A01源码包、SHA和最小部署说明。

  FE01-EVAL1继续原流程，不被A01训练等待或代码冲突阻塞。

risks:
  - 推理时去幅值会造成分布变化，不能代替训练消融。
  - 删除有效时长会引入额外混杂。
  - eps/clamp、隐含能量旁路可能破坏尺度不变性。
  - 单key退化不等于整个带残差网络必然常数。
  - probe敏感性不是全validation泛化结论。
  - 原生窗口、预训练与adapter差异仍影响跨家族归因。
  - 上游offline处理未核验时不能宣称完整在线因果性。
  - 单seed对照只是首轮证据。

open_questions:
  - 不重复询问已经确定的split、窗口、幅值定义和手动提交原则。
  - 路径、账户、资源和权重优先从已有本地资料解析。
  - 确实不可解析时列为待用户填写项，继续完成不受影响部分。
  - DiTing权重不可见时准备配置和检查器并标NOT_RUN，
    不用随机encoder替代，不阻塞其他三种模型。

[/AI-HANDOFF]

最终交付要求：

1. 实际代码、配置、测试和手动超算脚本，不只交付设计文档。
2. 一份中文报告，分别列明：
   已核验事实、机制诊断结果、仍待检验假设和限制。
3. 核心诊断图附图源数据与图注：
   query差异沿网络的变化、分支范数/gate、尺度/未来扰动误差。
   真实checkpoint结果和合成示例必须分开标注。
4. 实验矩阵说明：
   哪些ON能够复用、哪些证据支持复用、哪些任务仍未执行。
5. 轻量审阅包至少包括：
   resolved config、数据/权重/源码/初始状态SHA、
   探针和采样manifest、实际测试输出、gate/sensitivity表、
   因果与尺度测试结果、评价schema和用户手动命令。
6. 仅提交本任务文件；推送后核对远端并返回精确result commit。
   推送失败如实说明，不声称已经同步。

请按仓库约定返回完整[CODEX-RESULT]：
   task_id
   base_commit
   result_commit
   branch
   changed_files
   verification
   compatibility
   hpc_status
   remaining_risks
   review_request

其中hpc_status必须反映实际状态。
本輪未由用户手动提交的正式训练与评价，应明确写NOT_SUBMITTED。
