# CODEX-RESULT：交流间对照与 normal geometry 补强

```text
[CODEX-RESULT]
task_id: 20260930-technical-report-seven-point-revision
repo: rular099/team_pytorch
base_commit: 8f54477ed11bf2623f40121648f7fcc756ada1a2
result_commit: none (本轮未请求提交或推送)
branch: exp/v01-velocity-prep-padding-control
changed_files:
  - reports/technical_exchange/multistation_pga_exchange_20260930_v2/: 新增独立修订目录，保留 v1
  - multistation_pga_exchange_20260930_v2.pptx / .pdf: 28 页，22 主报告 + 6 附录
  - prepare_revision_assets.py / assets/: 七月 full-data、rt48 与本次 normal/random 原始结果重算
  - build_revision_deck.py / speech.md: 原生可编辑排版、讲稿、每页备注
  - TECHNICAL_DETAILS.md: RT57–61、架构、local/transport、空间比值和文献评分答疑
  - test_revision_assets.py / qa_revision.py / native_qa.json: 数值与文件/版式质量核验
  - README.md / sources.md / QA.md: 七条意见落实、复现与限制
verification:
  - prepare_revision_assets.py: PASS；旧 rt48 F1/PR-AUC 与七月 PPT 三行四舍五入数字一致
  - RT55 ep20 metadata/split/permutation + formal metrics: PASS
  - RT61 normal/random NPZ SHA256 + MDN 反归一化: PASS
  - test_revision_assets.py: PASS (5 tests)
  - 原 test_report_assets.py: PASS (4 tests)
  - py_compile 四个新增 Python 文件: PASS
  - LibreOffice PDF + qa_revision.py: PASS (28 slides/pages, native notes, image hash/aspect/crop/bounds)
  - 最终逐页人工版式核验: 见 native_qa.json 的 visual_review
  - git diff --check + 新增文本尾部空白检查: PASS
  - 原 v1 PPT/PDF SHA256: unchanged
compatibility:
  - 仅新增报告目录；未改模型、训练、推理、采样器、配置或超算启动器
  - 原 RT55 加载/推理代码完全未触碰；本轮未另跑模型回归
hpc_status:
  - not submitted；本轮无需用户运行超算任务
remaining_risks:
  - 旧/新事件与协议不同，跨交流变化非配对架构因果收益
  - normal all 包含输入目标；non-input 和 random 单列
  - 20s point F1* 在 validation 内优化，仅复现七月探索规则，不能作部署或独立 test
  - 公开文献并非同协议复现；PPT 跨机器字体可能替换，备 PDF
review_request:
  - 审查主报告是否以七月交流为真正历史起点，而非 RT57
  - 审查 normal/all/non-input、validation/test、point-score/probability-cutoff 的口径边界
  - 审查 TECHNICAL_DETAILS.md 对 RT57–61 和第 7–8 页的解释是否忠于源码
[/CODEX-RESULT]
```

主要事实：normal 低 MAE 代表为 RT55 ep20 validation（all 0.1162，non-input 0.1949），最新 RT61 normal all 0.1250、random 0.2373。七月 full-data 0.2966 与最佳 KNET rt48 diagnostic 0.2877 同时保留。ep32 的 R² 略高，不拼接 checkpoint。

需要 ChatGPT 独立审查时，交付本目录、指定上述代码基点及本地未提交状态；本文件不声称材料已经同步到 GitHub。
