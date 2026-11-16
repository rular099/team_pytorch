# FE01 论文素材与完整首稿任务书

任务ID：`20261116-fe01-paper-materials-and-first-draft-v2`。

请依据本包的大纲和素材规格，在独立论文工作区完成可复现图表、中文论文完整首稿和审阅包。**不是继续返回一份计划。** 本任务不要求重训、新模型推理或新的超算任务；现有数据缺项时完成能够完成的正文及素材，单列缺口。

## v2 范围修订

用户已明确要求简化题目和章节标题，并不在本篇讲ON/OFF消融。本节优先于v1。

题目固定为《基于 DiTing 的早期 PGA 概率预测》。正文使用“1 引言、2 数据与方法、3 结果、4 讨论、5 结论”，小节按新版大纲执行。题目、节标题与图表短标题不使用冒号或长副标题；代码和文献原题名不受该排版限制。

删除幅值消融图、对应正文/摘要/结论、补充材料和幅值机制命题，不是仅挪到附录。真实模型的物理幅值和有效时长信息仍必须如实说明，不改变权重、输入或预测。A01已训练完整模型的来源如需追溯仍可只读查证；不能因取消消融叙事而改写训练历史。

若v1首稿或论文工作分支已存在，先保存状态和已有修改，安全复用或新建修订分支，不删除、重置或覆盖用户未提交文件。报告实际选择的路径和分支；不要求为命名一致强行重建。

正文7组图、3表；原F5取消，原F6/F7/F8变为新F5/F6/F7。C06退出本篇，其他claim_id保持稳定。素材、正文图号、索引、图注和构建脚本一并修订，不遗留旧引用。

## 一、身份、权限和保护范围

Repository：`rular099/team_pytorch`。
建议基点：`560a29d8c82ba634549ed886840b2a2a539e0114`。
来源分支：`exp/fe01-on-warning-map-evaluation`。
建议新分支：`paper/fe01-diting-pga-manuscript-v2`。
建议独立worktree：`team_pytorch_fe01_paper`（名称只是建议，先核对已有目录；已有论文工作区不强制重建）。

本任务允许：只读核验Git/本地已授权文件；在独立论文目录新增文稿、CPU结果读取/整理/绘图脚本、测试、构建说明和输出；联网核验论文/数据引用。不得把本任务解释为生产实验授权。

禁止：sbatch/srun/ssh远程执行、GPU模型加载/重推理、新训练/续跑、held-out test访问、改变checkpoint/阈值/入组规则/指标/采样、校准sigma、丢弃失败事件、重签旧lock、修改旧seal、覆盖报告、切换/清理其他worktree、向科学分支强推或合并、修改FE02。

先保存原工作区HEAD、status、tracked diff、未跟踪清单。检查分支/目录重名，不能使用reset/clean/stash/force。若HEAD已变化，先审查diff；结果来源仍固定本任务列出的commit，不自动改成最新表。

可在论文工作分支作可追溯本地提交。发布到远端只沿用当前用户已有的明确发布授权；没有授权则交付本地commit和审阅ZIP，不擅自写GitHub。无论是否发布，都输出真实base/result完整SHA，不能说“已同步”而没有成功回执。

### 恢复顺序

1. 读根AGENTS以及活动分支的`docs/ai/START_HERE.md`。本次读取560a29d8时该文件不存在；不得声称已部署。
2. 回退只读`chore/persistent-ai-handoff`的固定commit `7445a22e6c9fef3e623e73b21f03bbeb5bf58ec0`下START_HERE和FE01的STATUS/NEXT_TASK/DECISIONS，记录其为较旧主线快照。
3. 当前论文主体属于FE01_ON；A01仅按需核对本文完整模型的训练与checkpoint来源，不作为本篇消融证据。11-15/11-16报告覆盖旧ON待办；不要用旧FE01/EVAL1的TEAM阻挡解释新线。FE02完全排除。
4. 本任务只写论文工作记录；不要为“接手”自动更新全局memory或把论文写作进展改写成科研实验完成。确有本线状态部署时，遵循本线所有权、单独状态提交；未部署则保存论文自己的HANDOFF，不移植全套记忆工具。

## 二、输入文件与事实优先级

本包：`01_PAPER_OUTLINE_CN.md`、`03_ASSET_SPEC.json`、`04_CLAIM_REGISTER.json`、`05_EVIDENCE_INDEX.json`、`06_REFERENCE_SEEDS.json`、`07_REVIEW_CHECKLIST.md`。

先读取V4固定结果目录的README、DATA_DICTIONARY、verification、local_numerical_checks、SOURCE_MANIFEST、DELIVERY_MANIFEST及method_sources；再读取V4源表；仅在完整模型身份需要时查A01相应清单，不导出其OFF对照结果。先核验实际文件而非抄AI摘要。

顺序：真实结果表及身份 > 已pin实际源码/配置 > 科学合同 > 中立报告 > AI解释。不得用论文、AI报告或手工表格替代不存在的实验数据。Codex封存与ChatGPT独立分析可作审阅背景，但不是第二份实验。

将每一实际读取文件写入`evidence/input_manifest.json`：repo、full_commit、path、git_blob_sha（可用时）、实际sha256、bytes、读取方式、是否原字节/脱敏副本、可用范围。旧私有seal与脱敏JSON不应强行用同一字节SHA比较。

不能只读README就宣称复算过所有指标；不能因为新包有79009条索引就宣称本地核验79009个原实体。

## 三、建议输出布局（新建，实际采用后记录）

```text
papers/fe01_diting_pga_v2/
  README.md
  manuscript_zh.md
  manuscript.tex                     # 可编辑排版源；环境支持时
  supplement_zh.md
  captions_zh.md
  references.bib
  references_audit.json
  GAPS.md
  CLAIM_EVIDENCE.json
  HANDOFF_FOR_CHATGPT.md
  evidence/
    input_manifest.json
    scope_audit.json
    executed_model_path.md
    method_definitions.json
    statistics_registry.json
  assets/
    panels/                          # 独立panel SVG/PDF/PNG
    figures/                         # 正文7组图和补充图
    tables/                          # 表格MD/CSV/LaTeX
    source_data/                     # 每panel实际作图数据
    FIGURE_MANIFEST.json
    FIGURE_CAPTIONS.json
  scripts/
    audit_inputs.py
    build_source_data.py
    make_figures.py
    make_tables.py
    validate_manuscript.py
    build_all.py
  tests/
    test_source_integrity.py
    test_estimands.py
    test_figure_data.py
  qa/
    numerical_checks.json
    claim_checks.json
    rendering_checks.md
    build.log
    environment.json
  build/
    manuscript_preview.html
    manuscript.pdf                   # 实际编译成功时
    figure_contact_sheet.png
    paper_review_bundle.zip
    DELIVERY_MANIFEST.json
```

这些都是建议新增文件，不是声称仓库已经有这些脚本。大逐目标表、波形、权重不进入Git；源表可只保留小型派生绘图表及完整原路径/hash。PDF/PNG/ZIP按项目体积策略处理并保留实际交付位置，不能用未上传链接伪装远端可读。

## 四、分阶段执行与验收门

### P0 身份/输入审计

- 确认精确ref、模型内部epoch身份、split=validation、seed42、ON。
- 分开V4结果与必要的历史完整模型来源，分开代码commit、结果commit、写作commit；ON标签只在内部身份核验中保留，不生成ON/OFF比较。
- 核验实际可读的输入与发布manifest；归档不存在则记录缺失，不绕过SHA。
- 输出每张素材的状态：READY_FROM_TABLE / REQUIRES_EXISTING_LOCAL_EXPORT / BLOCKED_MISSING_SOURCE / NOT_APPLICABLE。项目最终PDF未编译用NOT_BUILT，不写PASS。
- 完整新回放逐目标MDN和frame metrics没有进入中立轻量包。先找已经授权且真实可读的本地原产物；仍缺就写GAPS。禁止从旧snapshot伪造逐秒曲线或从覆盖率指标反推PIT图。

### P1 原始数据到作图表

- 所有已报告指标以原表为准，按明确key筛选，拒绝重复或缺行；禁止在绘图代码手写图上数值。
- 主分析固定selected，epoch12只作单列敏感性；不得将两者视为独立训练样本。
- 只允许既有结果的读取、精确重新聚合、百分比/单位转换、已定义端点的复算和版式展示。本轮不新增优化、阈值搜索、模型混合或“挑最优”的分析。
- 新计算的任何CI必须是既有协议端点的明确复算，源数据足够且复现原方法；记录source、seed、draws、结果差异。缺少足够数据时保留原配对CI，不从模型各自CI拼配对CI，不合成绝对CI。
- 不开展新的消融或家族×幅值交互。持久达标端点、异步队列性能验证仅可作为未来工作，不在本轮执行。既有degradation汇总可直接展示，不改原TTUM成功定义。
- 逐panel保存真正使用的CSV和metadata，列出包含的keys、单位、过滤前后人数与reason。不是只记录一条目录名。

### P2 模型路径和数据示意图

- 以实际配置和源码追踪为依据，输出`executed_model_path.md`，列函数/文件/行号、参数、启用状态。
- 输入波形、存储mask、当前截止mask、归一化、10维幅值+时长、station与query坐标、adapter/readout/MDN路径可单独标注。
- 未启用模块不得画实线执行路径；辅助震级/位置头不反馈PGA时不画反馈。
- frozen encoder明确与trainable adapter/downstream区分；TEAM前端scratch另列。
- 图为“机制示意”而非实验曲线：可用规则生成的波形示意，但必须显著标SCHEMATIC且不赋予真实事件身份；不生成虚构attention图、feature embedding或地质背景。
- 科学图标/箭头使用简单可编辑矢量；无需生成式插画，不引入视觉上像证据的假热图。

### P3 可复现科学图表

- 按ASSET_SPEC完成7组正文图、3表及必要补充。先独立panel再合版，保存可编辑SVG和矢量PDF，另给PNG预览；栅格类不少于300dpi。地图原图不得靠放大伪造细节。
- 四模型在全文使用相同视觉编码，使用可辨识颜色并辅以点型/线型；黑白打印可区分。字体小于约8pt、轴标重叠、裁切图例必须修正；缩至论文栏宽检查可读性。
- 不使用雷达图、双轴拼异单位、3D柱状图、截轴夸大、只画显著子组。效应图有零线、方向说明；概率覆盖图有名义线，宽度与覆盖并列。
- p50/p95画成分布摘要，不画成均值CI；Unknown识别界与bootstrap CI用不同形状/图例；不可由百分位线暗示95%统计置信。
- 每张图注必须含：协议、split、seed、epoch/policy、目标人口、事件数/请求或field数、权重、单位、CI方法、稀疏支持、scene。必要细节可用统一脚注加panel差异，不能完全省略。
- legacy五时刻只显示五个观测点；连线仅为视觉引导、不平滑成每秒推理。TTUM若用完整事件记录可画经验确认比例阶梯，不从5/10/20点平滑伪造全时钟。
- 所有模型和反例保留；正文主讲noninput时，all-target对照和K1/untriggered不利结果必须在正文文字及定位明确的补图出现。
- 图5理想与数值/渲染紧邻，图6不得把共享四模型绘图叫单模型部署；图7不得把台站失败比例叫连续面积。

### P4 完成首稿

- 按新版五节大纲写成完整段落，不只列标题/要点；所有论文标题不使用冒号补充说明。中文学术表达，术语首次定义。摘要约400–600字；讨论不重复全部结果表。
- 模型名：DiTing方案或基于DiTing的特征方案。PhaseNet/EQT为特定权重的中间表示迁移，不代表原拾取系统在其本职任务被击败；TEAM为当前协议下的scratch实现，不等于重现原论文全部配置。
- 方法必须注明共同下游不消除前端适配、原生窗口、预训练、能力条件化时间采样混杂。
- 摘要和结论中的数字由生成的表/metrics文件注入或核对，不手工复制后漂移。
- 主文每项定量结论给内部claim_id并映射源表；审阅版可用HTML注释或独立claim索引，清洁版不强行展示内部Git细节。
- 所有数字与结论限制一并写。主MAE为非输入，不因全目标结果不好才临时换主端点；但首秒和分层的重点安排是写作组织，不冒称整个故事事前预注册。
- NLL不做不恰当百分比宣传；log-MAE下降不称线性PGA下降；第一秒cutoff不称1秒送达；normal never-input 207事件不称全1310泛化。
- 本篇正文和补充材料均不写ON/OFF消融、幅值贡献或家族幅值互补结论。保留实际物理幅值/时长输入的正常方法说明，不改模型，不删除历史产物。
- 未得数据的章节写已知范围、缺口和影响；保留可审阅首稿，不用占位假数/随机图补齐。图号缺失标GAP，不能在正文声称见不存在的图。

### P5 文献、引用与公开性

- 原论文、模型实现/权重metadata、数据源各自引用。`06_REFERENCE_SEEDS.json`是检索起点，需实际阅读全文相关段落和核对BibTeX字段。
- DiTing精确预训练checkpoint对应论文、语料、许可、输入处理若未核实，保留待补项；不把DiTingMotion、同名PhaseNet权重或其他DiTing模型论文当作本模型来源。
- PhaseNet/EQT原论文的训练数据不能替代本项目实际SeisBench STEAD v2权重来源。引用架构原文和实际权重来源两层信息。
- 查找近期同任务研究以判断相关工作与新颖性，但不能用不同数据/标签/时间/人口的论文指标直接排排行榜。不写“首次”“SOTA”除非有充分可追溯支持。
- 正文、参考文献、BibTeX双向检查；所有DOI/作者/年份实际核验。不能凭LLM生成不存在的文献。不能因网页迁移的online date把原发表年改错。
- NIED数据DOI与本研究K-NET-only使用范围分开；公开许可按实际条件核查，不上传原波形或权重。

### P6 数值与版式QA

必须执行并记录实际命令、返回码和stdout摘要：

1. 输入sha256一致；构建后原源文件再次散列相同，证明未回写。
2. 主端点4模型/selected/noninput/primary_10cell行唯一；1310 events、139440 requests、10 cells一致。
3. 主MAE、首秒MAE、主要CI、TTUM计数能由源表定位；四舍五入容差与输出位数匹配。锚点不是允许硬编码绘图的替代数据。
4. V4配对差固定DiTing−comparator；确认没有导出OFF条件、幅值消融图或原C06命题。完整模型真实输入与来源仍记录。
5. RMSE、field权重、event权重、pooled权重不混用；PIT等无源数据则BLOCKED而非PASS。
6. TTUM total=eligible+ineligible；normal never-input=207，random=539，all=1057；Unknown与未达标/超deadline区别。
7. Shadow及时+已知失败+Unknown=affected；没有affected不是0失败；false-positive分母为完整负例。
8. 表中p50/p95不是CI；shared-render numeric_ready不是独立numeric场景。
9. 图片/表/标题/正文/摘要数值一致，无缺图链接、裁切、乱码、默认轴歧义。实际打开所有正文图和渲染预览逐页检查；缺渲染能力记NOT_RENDERED，不伪称视觉QA通过。
10. 同一输入重复构建时source_data和数值一致。压缩包/字体等非科学字节若有时间戳差异，解释而非据此误报科研数值变化。
11. 原科研worktree的HEAD/status/tracked diff与前快照一致；没有触发模型加载、HPC命令或test。
12. 题目及节/小节/图表短标题不含冒号；正文、补充与摘要没有ON/OFF对照或原C06结论；7图3表编号及引用完整，物理幅值输入未被错误删去。

## 五、论文所需数据的分级处理

A类：已有小CSV/JSON——必须完成源表驱动的素材和正文。
B类：Git中的完整事件压缩CSV——可做不改变端点的确定性呈现/既有统计复核，记录读取成功与覆盖。
C类：本地已存在且身份可核验的逐台站MDN/图源——只读补充；是否存在不能推测。
D类：不存在或不可读——记录缺口、影响的结论、后续所需最小产物；不自动连超算提取，不生成替代证据。

完成A/B类的完整初稿，不因C/D缺口只返回计划。

## 六、交付与后续审阅接口

最终至少交付：完整中文正文、补充材料、图注、可审计参考文献、7组正文图/3表的实际完成状态、每图源数据与完整manifest、脚本/测试/环境、数值QA、版式QA、GAPS、可阅读预览和审阅ZIP。

PDF只在成功编译且逐页查看后标为BUILT/RENDERED；无LaTeX环境则先提供HTML预览和可编辑正文，不为本任务升级科学模型环境。工作草稿里的作者/资金待补项与科学数值缺失必须区别。

`HANDOFF_FOR_CHATGPT.md`必须包含：base/result_commit；实际源数据refs；各图表文件/claim_id；已知限制与疑问；新派生计算清单；未完成素材及原因；建议ChatGPT首先审阅的三处科学风险；原状态保护结果。

输出以下字段，不只说“已完成”：

```text
[CODEX-RESULT]
task_id: 20261116-fe01-paper-materials-and-first-draft-v2
repo: rular099/team_pytorch
branch: <实际独立论文分支>
base_commit: <完整SHA>
result_commit: <完整SHA或none并说明>
input_result_commits: <V4固定结果；仅有必要身份追溯时另列历史完整模型来源SHA>
changed_files: <仅本任务文件>
manuscript: <真实路径、版本、字数、完整/缺项>
figures_tables: <逐项READY/BLOCKED/NOT_RENDERED>
verification: <实际命令、返回码、数值和视觉QA范围>
compatibility: <原模型/lock/seal/科学worktree未改的实测记录>
new_analyses: <既有端点复算/展示列表，不冒充新训练>
hpc_status: NOT_SUBMITTED
held_out_test: NOT_ACCESSED
publication_status: <本地提交/经授权push及实际回执>
remaining_risks: <缺证据与统计/物理/出版风险>
review_request: HANDOFF_FOR_CHATGPT.md + manuscript + assets + source_data + QA
[/CODEX-RESULT]
```

ChatGPT下一阶段的工作是审核证据—图表—正文一致性、收紧归因与统计措辞、调整论证和润色，不替Codex补造未执行产物。
