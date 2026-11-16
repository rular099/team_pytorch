# 本轮实际检查

状态为可供科学审阅的首稿，不代表期刊终校或外部同行审稿。制备日期2026-11-16。

## 来源与数值

输入包10项声明摘要逐项核对，中立结果156个交付成员按固定Git对象重新读取并核对字节SHA。后续配置、训练预算、模型与空间方法、历史背景读取合计196个固定Git来源文件，最终重新散列全部一致；本地指令文件另列。来源见evidence/input_manifest.json与input_hashes_after.json。

audit_inputs.py、build_assets.py、claim_trace.py、finalize_manifest.py、compose.py和verify.py组成的CPU复现流水线实际返回0，命令/起止记录见reproduction_commands.json与对应日志。既有70张数值CSV在完整重建前后逐字节相同，见reproduction_data_check.json。其后补充了DiTing完整模型十二轮预算来源和空间定义核读，并在文稿/排版修改后重新运行audit、claim_trace、compose、verify与浏览器打印，均返回0；这些补充不重新制图或改变指标。

numerical_checks.json包含15项通过的检查，覆盖四模型旧十单元MAE/RMSE复算、固定目标资格、失败结局、完整负例FPR和全normal帧数等。tests/test_paper.py实际10项通过，包括模型轮次与激活路径、不同人口不可替换、未知界限守恒、不利结果保留、11个结论映射、蓝图舍入锚点、主文/摘要字数与范围、面板源和300dpi、脚本编译及禁止科学模型导入。最终verify.py返回0，完整日志为focused_tests.log。没有重新计算2000次bootstrap、没有执行新模型或读取测试集。

## 图表与文稿

正文9090个汉字，摘要455个汉字；口径见verification.json，正文排除摘要、参考文献、图表与补充材料。F1–F7的37个独立主面板及7个组装图均实际生成SVG/PDF/300dpi PNG。T1–T3实际生成CSV和Markdown，T2按人口/端点及点误差/概率/校准分块以便阅读。主图不是从蓝图锚点画出的示意性能；图1的合成波形明确标SCHEMATIC。

原始12例各5/20秒关键帧共24张，全部复制自固定结果，字节一致。案例没有按误差筛选，保留无网格支持的原图。其PNG与正文新矢量图分别说明。

本地Firefox160.0实际生成paper_review.pdf和supplement_review.pdf，分别27页、4页。未安装TeX引擎，因此paper.tex未编译，不能称为LaTeX编译成功。浏览器操作只访问本地HTML；受沙箱限制的浏览器启动和本地套接字连接在获准后执行。初始无profile/沙箱崩溃、首次分页跨图与字形问题均已修正，没有把失败当成功。初版失败分页缩略图只作历史记录，位于history/initial_pagination_failed.png。

实际查看全部7组图、27页正文与4页补充缩略图，并单独放大正文图2/4/7所在页；24原案例也逐张纳入总览。正文7组图的标题和图注在同一页，详见pdf_structure_check.json。独立面板的SVG/PDF没有逐个单独打开；其绘制函数与组装图相同，格式/文件摘要另行自动核对。最终视觉记录为visual_review.json，未冒称期刊排版完成。

## 范围与兼容性

生产代码、配置、lock/seal与全局会话文件零修改，未访问其他研究方向结果。四个受保护科学工作树的HEAD、分支、tracked/staged diff、简要状态与既有未跟踪路径保持，见worktree_protection_after.json；不声称启动时对所有未跟踪内容做过完整字节散列。RT55源码未改，未额外运行RT55权重回归。

没有SSH、Slurm、训练、推理、GPU或held-out test行为。未推送论文分支。DiTing预训练精确来源/许可和新回放概率诊断仍在GAPS中，模型性能解释保留单种子、验证选轮、欠覆盖、稀疏支持与交付工作流限制。
