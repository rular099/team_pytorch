# ChatGPT 论文审阅交接

本轮已实际完成中文连续首稿及素材，不是待实施计划。请先阅读论文PDF或preview.html，再用claim_map、claim_trace与source_data逐条核对，不用默认分支或旧论文版本代替。

## 版本与提交

仓库 `rular099/team_pytorch`，独立论文分支 `paper/fe01-diting-pga-manuscript-v2`。
论文基点 `560a29d8c82ba634549ed886840b2a2a539e0114`。
完整正文、图表、来源和QA材料提交 `7dddbee6a0c8b546af08857429a5f05ba2a65334`。
固定中立结果 `b366434d2cceb55d188576ce1084f571e09da097`。
执行源 `50672e131d876a604432b283162d5b0189d40cf0`。
DiTing完整模型身份只读追溯 `29ffb20ba0effc0c345a0329ba3f8c658dc8cef0`，历史路径核验 `577a0cec43643597a61209209a770aeecf5fefa0`。

本文件及material_commit_receipt在材料提交后的交接提交中加入。2026-11-16用户明确要求推送后，已将完整交付提交 `1903bdf34d227034a181f19936ca3428204f362e` 推送至GitHub并核对远端SHA。GitHub连接器已确认本论文分支存在，可直接读取。发布核验见 [publication_record.json](publication_record.json)；审阅ZIP根目录REVIEW_IDENTITY.json记录的是原交付快照，当前分支的后续更新为发布文档。

## 实际交付

- 正文 `manuscript.md`，题目《基于 DiTing 的早期 PGA 概率预测》，五节连续中文，正文9090汉字，摘要455汉字，均按verification.json口径；整合稿 `build/paper_full.md`。
- `paper_review.pdf` 27页，`supplement_review.pdf` 4页，均由本地Firefox打印HTML实际生成；`paper.tex`可编辑但未编译。HTML主稿、补充和案例浏览页均提供。
- F1、F2、F3、F4、F5、F6、F7均 **READY**；37个独立主面板，各组SVG/PDF/300dpi PNG；原始24张案例PNG另列，不冒充矢量数据。
- T1、T2、T3均 **READY**，CSV与可读Markdown；T2按人口、端点和评分类型分块，仍为一组主表。
- S1–S8均有实际材料。S3中的保存评分/覆盖/宽度已提供，新固定query的PIT和可靠性分箱 **BLOCKED**；其余缺项见GAPS。
- `assets/source_data`、输入/图表manifest、11个允许结论的精确源行、实际模型路径、参考文献BibTeX及核验、脚本与测试、QA均提供。

## 已验证事实

四模型为validation、seed42；selected轮次依次DiTing12、PhaseNet12、EQT10、TEAM11。主非输入端点1310事件/139440请求、十单元等权；全部目标194265请求，二者不混用。PGA误差使用log10(m/s²)。新never-input人口不等于旧动态非输入，normal TTUM适格207/1310，而shadow有目标312、其中214有affected；这些资格不同。

实际PGA路径包含物理幅值/有效时长、归一化、绝对坐标与四层target cross-attention，台站上下文自注意力和PGA事件上下文没有启用；辅助震级/位置不输入PGA。模型参数与训练预算来自归档审计，未在本轮加载权重重新计数。旧通用协议不替代实际配置分支。

平均精度与分布评分提升同时伴随全目标/部分子组的不利结果、欠覆盖和交付成本。理想确认优势不能改称交付优势，三帧首次确认后仍可能再次不通过。未知识别上下界、事件bootstrap区间、warm分位数和条件双方确认时间差均分开说明。

## 实际检查

10项专项测试通过；15项数值检查通过；196个固定Git来源重新散列一致；完整CPU素材重建前后70张已有数据CSV逐字节一致，新增训练预算表另按原行校验。完整命令流水线与后续检查记录见qa。没有重新bootstrap或声称重算全部历史区间。

已查看7组图、27页主稿、4页补充、24个原关键帧的总览，并放大图2/4/7所在PDF页。7个标题和图注均同页，公式源码保持同页。独立面板格式与摘要自动验证，没有宣称逐个打开所有SVG/PDF。最后git diff --check通过；生成稿尾部空行已修复。源码修改仅论文目录。

四个科学工作树的HEAD/分支/tracked与staged diff及既有未跟踪路径保持。RT55生产代码、配置、lock/seal和全局会话未修改；没有运行RT55模型回归。制稿阶段没有新训练、推理、GPU、超算、SSH或held-out test访问。随后授权的发布通过GitHub SSH 443端口传输已有Git对象。

## 推断与待决问题

DiTing在极早期有限观测中可能受益于波形表征，但当前是完整方案比较，不能拆解为预训练或某一模块的因果效应。单种子、验证选轮、不同原生窗/训练暴露与探索性多重比较限制外推。

DiTing MAE1200M精确论文、语料重合和许可尚未证实；不能拿同名数据集或其他模型文献代替。新固定查询MDN诊断缺源，不使用六事件试验或旧五时点填补。两篇近期文献的主要来源索引可核验，直接打开403/CDN受阻的状态如实保留。作者、单位、资助和利益冲突需作者确认。上游在线因果性、通信时延、严格配对硬件和连续网格真值尚无验证。

## 审阅请求

请依照随包07_REVIEW_CHECKLIST逐项审稿，先给总体评价，再给必须修改、建议修改和语言润色。每项定位section/paragraph/figure/panel/claim_id与固定证据，说明是否改变科学含义。重点核对目标人口、权重、欠覆盖、场质量与交付时钟、失败/FPR/Unknown，以及对PhaseNet的条件增益是否被过度归因。请返回修订正文、图注建议、作者查询单与修改记录。任何新增训练、阈值选择或模型导出建议只列建议，不自动执行。
