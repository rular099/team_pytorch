# 基于 DiTing 的早期 PGA 概率预测

中文连续首稿 v2，主文五节，7组正文图、3组正文表，补充材料S1–S8。所有新图由固定结果表驱动；没有训练、模型推理、GPU、超算或测试集访问。

阅读入口为 [整合预览](preview.html)、[论文PDF](paper_review.pdf)、[正文源稿](manuscript.md)、[补充材料](supplement.md) 与 [交接说明](HANDOFF_FOR_CHATGPT.md)。PDF由本地Firefox打印HTML生成，不是LaTeX编译产物；paper.tex是另行提供的可编辑源，未声称编译通过。

## 证据版本

仓库 rular099/team_pytorch，论文分支 paper/fe01-diting-pga-manuscript-v2，论文基点 560a29d8c82ba634549ed886840b2a2a539e0114。固定结果 b366434d2cceb55d188576ce1084f571e09da097；主执行代码 50672e131d876a604432b283162d5b0189d40cf0；DiTing完整模型身份只读追溯 29ffb20ba0effc0c345a0329ba3f8c658dc8cef0。历史路径说明577a0cec43643597a61209209a770aeecf5fefa0只用于实际输入路径核验。论文实际材料与交接提交见交接文档和审阅包REVIEW_IDENTITY.json，不用默认分支替代。

## 目录

- manuscript.md为正文；build/paper_full.md为插入图表与参考文献的整合稿。
- assets/figures提供F1–F7各SVG/PDF/300dpi PNG；assets/panels提供37个独立面板及RMSE伴随图；assets/tables保存T1–T3及T2可读子表。
- assets/source_data保存每面板实际数据和完整补充表；assets/asset_manifest.json记录过滤、分母、权重、单位、源文件/脚本/产物摘要。
- assets/cases保存12个原始预选案例的5秒和20秒关键帧，字节保持不变，不伪称矢量观测。
- evidence/input_manifest.json记录所有实际读取的固定Git对象；claim_trace.json将11个允许结论绑定至实际源行；executed_model_path.md给出实现分支核验。
- qa保存本轮实际检查、测试日志、输入复核、PDF分页与图表视觉检查；references_audit.json区分主要来源核验与直接打开受阻；GAPS.md保留未决项。

## CPU复现

在包含上述Git提交对象的仓库中，使用Python、numpy、pandas、matplotlib、Pillow和mistune。精确已用版本见evidence/environment.json；不要升级科学模型环境来制稿。Noto CJK字体路径按机器设置；当前脚本使用系统字体。所有制图和再聚合是CPU素材工作，不加载科学模型。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -B papers/fe01_diting_pga_v2/scripts/reproduce.py
```

audit_inputs默认重读固定Git证据与已封存的本地搜索记录。若需要重新枚举授权的本地下载目录，显式设置FE01_ARTIFACT_ROOT；不设置任何超算地址，不自动SSH、拉数据或提交任务。

本机实际执行使用已有general Python环境，未安装或升级包。复现需已有输入提交；source_data可在没有原始数据/权重时直接审阅，但完整重绘要能读取固定Git对象。

## 浏览器PDF

浏览器打印是可选独立步骤，不是复现脚本隐式启动的外部服务。准备一个临时Firefox profile，启动headless Firefox并仅监听本地9237端口，再执行scripts/render_preview.py。脚本只访问本地HTML，写入paper_review.pdf、supplement_review.pdf和截图。浏览器和本地socket权限需按执行环境处理；没有授权时保留HTML，不绕过限制。

## 范围与未决项

完整模型保留物理幅值与有效时长。本文只讨论已有模型组合比较与交付边界，不据名称推断DiTing语料，也不把验证选轮、单种子bootstrap或示意网格视为独立测试。精确预训练论文/语料/许可、新固定查询MDN诊断、作者署名和资助等仍待补齐。对照与所有不利结果、Unknown、资格不足和未执行组合均保留。

原科学代码、配置、lock/seal和其他工作树不改；最终交付为独立论文分支的本地提交与ZIP，远端发布未执行。
