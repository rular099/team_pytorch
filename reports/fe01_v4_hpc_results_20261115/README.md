# FE01 ON V4 超算结果：独立分析数据包（2026-11-15）

本目录保存可核验的数值、来源、方法和预定案例；Codex 的解释另行封存。请先读
[ChatGPT分析任务](CHATGPT_ANALYSIS_PROMPT.md)，独立形成结论后，再由用户提供 Codex 分析进行对照。

仓库 `rular099/team_pytorch`，同步分支 `exp/fe01-on-warning-map-evaluation`；本轮整理基线
`99d9b1c4bb5c47fc26125df7eea2648917e90462`。本地工作分支为 `review/fe01-v4-results-20261115`。
请以用户给出的本次 result commit 读取数据，不用默认分支。超算冻结主程序仍为
`50672e131d876a604432b283162d5b0189d40cf0`；调度包仍为 `1cebcdc61285fb99767509cb2e513bee07aefb05`。
本轮没有重训、重推理、修改运行配置、访问 held-out test 或提交超算任务。

推荐阅读顺序：

1. [核验结果](verification.json)、[数值阅读表](KEY_RESULTS.md)、[数据字典](DATA_DICTIONARY.md)。
2. `source_tables/analyze/0/legacy_with_physical_covariates/` 中的原始主指标、配对区间、分层和支持表；
   [主MAE复核](derived/legacy_primary_pairwise_recheck.csv)。
3. [TTUM完整支持复核](derived/ttum_support_recomputed.csv)、[预算0.20的配对区间复核](derived/ttum_paired_b020_recomputed.csv)。
4. [时延复核](derived/latency_recomputed.csv)、[FIFO时钟分布](derived/queue_quantiles.csv)、
   [渲染shadow汇总](derived/shadow_rendered_summary.csv)，并同时读原始阈值网格表和场景定义。
5. [静态统计图](figures/FIGURE_SOURCES.json)、[预定案例索引](CASE_INDEX.md)、`method_sources/` 中与HPC pin一致的方法源码。
6. 需要复算时使用 `derived/*.csv.gz` 中的完整事件记录；每个压缩CSV独立可读。

全部结果为 **validation、seed42、ON**。新逐秒回放仅使用固定 selected：DiTing12、PhaseNet12、EQT10、TEAM11。
PGA 坐标是 `log10(m/s^2)`。旧预测再分析同时含 selected 和固定 epoch12；两者不是两次独立训练重复。
DiTing 本批为 FE01-A01 ON epoch12，不应替换成历史 RT55 epoch32 的其他协议结果。

本目录有三种不同的评价人口/时钟：

- 旧 `legacy_snapshot_v1`：normal/random × 1/3/5/10/20秒的10 cell；主非输入人口为1310事件、139440请求，
  全目标为194265请求。这里的 noninput 是逐请求角色；不是新回放的 fixed_never_input。
- 新 `fixed_query_replay_v4`：同一1310事件、每秒1–20秒、固定query panel；TTUM要求至少5个固定目标。
  normal/random 的 fixed_never_input 适格事件数分别为207/539；all-target均为1057。资格之外的事件单列保留。
- normal 完整网格的计算时钟与共享四模型比较图的渲染FIFO场景；通信/真实触发时延未知。
  random没有完成网格/渲染场景，不填造其时钟。

输入为 `runs/v4pack/fe01_on_eval_v4_review.tar.gz`，107210231 bytes，SHA256：
`7a5c19ad82d45c8c3a40a00a00b29856f4fd8720634fdbdf371e350f4bc32add`。
同名SHA文件、gzip、1016条review manifest均通过；12案例的453个文件与原图库索引SHA一致；49个V4主程序文件与HPC pin一致。
原冻结核心记录96文件已检、65个dtbench依赖本地未检；本轮不将这些外部依赖的元数据记录当成本地字节复验。
完整索引共79009条原产物；全量回放/图源和权重仍在超算，本地没有重新验证所有79009个实体文件。

`status_summary`在pack过程中生成，pack自身显示RUNNING；之后提供的stdout记录pack task0退出码0和最终汇总。
四项latency、analyze、41项normal渲染在该超算快照里均为 `COMPLETE_SEAL_VERIFIED`；因此此前latency未知的交接状态已更新。
原plan/contract中保留的实施模板状态不代表这次实际作业未执行。原私有JSON在review封装时有路径脱敏，
其字节不一定等于原seal；以review manifest验证收到的副本，完整原SHA索引用于保留原件身份。

本地复核结果见 [local_numerical_checks.json](local_numerical_checks.json)：864条TTUM支持记录、216个已保存TTUM区间、
6个旧预测主MAE区间、104800个FIFO帧、8组latency汇总均复核通过。预算0.20的渲染配对区间是本轮从事件记录新算的，
与HPC已有区间分别标明。全部区间仍为固定模型的事件cluster bootstrap，2000次、seed20261007，探索性且未校正多重比较。
历史ON事件表经本次八份prediction SHA与先前A01 manifest逐一对上；不包含OFF结果。

没有收到全量新回放的逐台站预测/MDN或完整frame metrics，故本目录不声称重新计算了全量固定query的MAE、NLL或校准。
网格无共同独立连续地面真值，面积均NA；selected案例有缺高程/支撑的灰色区域。空间图不能替代未输入目标或面积评分。

复现命令（在本地分析机执行；不是超算登录节点命令）：

```bash
python reports/fe01_v4_hpc_results_20261115/review_bundle.py --archive /path/to/fe01_on_eval_v4_review.tar.gz --output /new/review-root
python reports/fe01_v4_hpc_results_20261115/build_review.py --raw /new/review-root/raw --output /new/neutral
python reports/fe01_v4_hpc_results_20261115/make_figures.py --review /new/neutral
```

所有输出目录须为新路径。环境版本和脚本SHA见 `metadata/review_software.json`、`figures/FIGURE_SOURCES.json`。
数据的原成员SHA、派生表输入和行数见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，最终交付文件SHA见
`DELIVERY_MANIFEST.json`。完整107MB原包不入Git；另提供只含本目录的ChatGPT ZIP。
Codex封存承诺见 `CODEX_ANALYSIS_COMMITMENT.json`；独立分析正文及其局部派生判断不在Git或中立ZIP中。
