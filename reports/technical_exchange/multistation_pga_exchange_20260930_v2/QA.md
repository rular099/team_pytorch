# 修订报告质量记录

日期：2026-09-30。科学证据与布局使用 figure-designer 的分组、对照与质检原则；实际交付保持用户确认的原生 PPT 排版。

## 自动检查

- PPTX：28 页，22 页主报告 + 6 页附录；PDF 同为 28 页。
- 封面恰好两个对象：标题、日期。
- 每页有讲稿备注；285 个原生非空文字对象、11 个原生表格。
- 8 张科研图与 PNG 源文件逐字节 SHA256 一致，零裁切、等比例嵌入；无生成模型改绘。
- ZIP 包完整、slide/notes 数量匹配；全部形状位于画布范围内。
- RT55 ep20 身份按 checkpoint metadata、split、permutation 核验，all formal 指标逐项一致；RT61 NPZ SHA256 与 formal summary 一致。
- 新增指标测试 5 项通过：最优 F1 cutoff、同分排序不获益、无正例处理、范围比的平移不变及非充分性、真实跨度零跳过。保留原科学指标测试 4 项也全部通过。
- 四个新增 Python 文件编译检查通过；`git diff --check` 通过；对新增文本额外检查尾部空白，无命中。

最终文件 hash 与自动检查明细保存在 `native_qa.json`。`visual_review` 只在最终 PDF 页面人工审阅完成后标为 PASS。

## 人工审阅

检查全部 28 页 contact sheets，并查看 normal 分层、历史散点、总体指标、空间反差、校准与文献口径页的 1600px 单页。确认无明显重叠、正文截断、科研图拉伸或数据标签错误；中文与单位可读。

第 14/15 页分别为 random/normal 分层；新增 normal 使用同一 RT61 checkpoint，右侧 remote 与图中 all-target 不混淆。历史空间和 coverage 来源均改为七月实际导出，不再拿 RT57 作“上次交流”。文献 F1* 星号和 validation 内优化的限制显式显示。

## 保留的风险和非结论

- 旧/新不是配对相同事件和目标协议；历史差值仅描述进展。
- normal all 包含大量输入站，不能把其高分当任意远程位置能力。
- validation 反复用于开发；F1* 是探索性 oracle，非独立 test 或锁定的部署阈值。
- TEAM 完整 EEW 与 QuakeFormer catalog forecasting 不同任务；不直接排名。
- 字体没有嵌入 PPT；无 Noto Sans CJK SC 的电脑可能换行变化，建议备 PDF。
- 仅报告修订，没有重新运行训练、推理或 RT55 回归；工程兼容性依据“没有修改运行代码”，不是宣称重跑所有兼容测试。
- 原 v1 PPT/PDF hash 保持 `a1b7a43f13b05f136d3c3c7df9ab30ff859722979b97988c8c5d569c0a959f80` / `385f5cc0e4d38f48d3a60dc0c5a5707a7cf44c5e29fa2e9e4d37f8ffae0c3d2a`。
