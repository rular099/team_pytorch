# FE01 V4：Codex封存分析公开与ChatGPT对照（2026-11-16）

按用户2026-11-16的明确要求，本目录公开此前封存的Codex分析，供ChatGPT读取并对照。
仓库`rular099/team_pytorch`，分支`exp/fe01-on-warning-map-evaluation`；本次基线
`b366434d2cceb55d188576ce1084f571e09da097`。请以用户给出的本次结果commit为公开版本身份。

- [Codex独立分析正文](CODEX_INDEPENDENT_ANALYSIS.md)：2026-11-15写成，按原字节公开。
- [Codex逐项结论表](CODEX_COMPARISON_ROWS.csv)：与正文同日封存，提供10个对照主题的判断、来源与限制。
- [公开及SHA核验记录](DISCLOSURE.json)：两份文件的SHA/字节数均与先前承诺一致。
- [原始SHA承诺](../fe01_v4_hpc_results_20261115/CODEX_ANALYSIS_COMMITMENT.json)：已在基线commit中发布。

正文SHA256为`ad1a68d2526e0926e6662fa4d9b4065af8a1e7ad63d5e1cddcac5f0c05d46abb`；
逐项表SHA256为`66cf80a1775ecb6b287561ad15d107313337ffbedd581e9425a05cf61c405852`。
两份文件未因本次公开修改。正文中的“此前封存”和“中立包不含判断”描述其形成和存放背景；
本次通过单独比较目录公开，原中立数据目录、manifest和ZIP保持原样。

供ChatGPT执行的对照步骤：

1. 先说明已经完成的独立分析所依据的repo/commit或上传数据。如果尚未独立定稿，可先按
   [原分析任务](../fe01_v4_hpc_results_20261115/CHATGPT_ANALYSIS_PROMPT.md)阅读中立数据；一旦读过本目录的判断，应如实说明。
2. 读取正文和逐项表，并核对其SHA与原承诺一致。
3. 按[对照模板](../fe01_v4_hpc_results_20261115/COMPARISON_TEMPLATE.md)逐项列出两方结论、原始数值、
   一致或分歧、需要纠正的定义/计算及共同可支持的结论。正文内所有相对源表路径均相对原中立报告目录。
4. 分歧优先检查人口、权重、时钟、Unknown界限和条件样本；以[中立结果数据](../fe01_v4_hpc_results_20261115/README.md)
   和已pin方法为依据，不把两方意见数量当证据。若无法读取压缩事件表，说明限制并使用此前中立ZIP。

当前尚未收到ChatGPT实际结论，未填造比较结果。本次仅公开原有分析，不重新计算指标、不修改模型/配置，不提交超算作业。
