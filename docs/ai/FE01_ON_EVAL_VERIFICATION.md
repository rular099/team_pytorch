# FE01 ON V4 实际验证记录

实施基点29ffb20ba0effc0c345a0329ba3f8c658dc8cef0；独立分支exp/fe01-on-warning-map-evaluation。
实际日期2026-10-22。本轮未提交任何正式超算作业、未加载生产checkpoint、未读取Japan波形、未访问held-out test。

## 本地实际运行

命令：

```bash
ON_EVAL_TEST_PYTHON=/home/zhangb/software/miniconda/envs/ai/bin/python \
ON_EVAL_TEST_DEPS="$PWD/artifacts/fe01_on_eval/test_deps" \
bash scripts/fe01_a01_compare/verify_local.sh
```

Python编译通过；全部新模块import通过且sys.modules不含torch；全部12个Bash/sbatch语法通过；**55项pytest通过**。
pytest下载在本任务隔离artifacts目录，没有更改conda/Torch/ROCm。部署源码不携带这份测试依赖，也不要求在超算重装软件。
日志见[local_verification.log](../../reports/fe01_on_eval_20261022/verification/local_verification.log)。

| 验收范围 | 实际证据/状态 |
|---|---|
| 十cell权重、RMSE开方、差值方向、配对人口/标签拒绝、低方差、稀疏支持 | CPU pytest PASS；实际旧预测统计完整运行 |
| CSV默认解析漂移、round_trip精确往返 | CPU pytest PASS；真实导出仍通过原严格人口门 |
| SNR缺噪声窗/补零/零能量/信号不足/未来污染、合法零 | CPU pytest PASS；Japan真实SNR待CPU sidecar |
| 真实首次越阈而非峰值时刻、缺测/左删失/标签不一致 | CPU pytest PASS；真实Japan crossing_truth NOT_RUN |
| JST→UTC、轴不一致、FIFO积压、投影km/第三坐标、网格cap与退化 | CPU pytest PASS；Japan绝对时钟/网格高程支撑待核 |
| TTUM首帧/退化/连续三帧/缺帧重置/尾部删失/N<5/平坦range/固定角色迁移 | CPU pytest PASS；Japan TTUM NOT_RUN |
| shadow lead=0、=tau、正但不足、超观察窗口、缺文件、no_input、空H、非单调、送达乱序、恒报警误报控制 | CPU pytest PASS；Japan shadow NOT_RUN |
| 四格交互共同域、联合事件抽样、缺SNR不可识别 | CPU pytest PASS；真实已有字段交互已运行 |
| prepare/status不提交、running/success不重提、FAILED显式resume、UNKNOWN拒绝、原子attempt/seal、旧sacct格式 | fake scheduler/文件系统 pytest PASS；未调用真实sbatch/srun |
| HDF只读/共享数据散列缓存 | 小型合成HDF pytest PASS；4并发请求只散列1次 |
| wrong epoch/config/source/缺预训练身份/错误family/seed/mode、TEAM合法null | Torch-free metadata负例 PASS |
| 实际完整tensor/encoder拒绝、四ON旧点完整MDN等价、query/cache/未来在encoder与输出上的门 | 已实现计算节点model_smoke；**NOT_RUN：无本地生产权重/DCU** |
| RT55默认加载/推理动态回归 | **NOT_RUN：本地Torch/生产依赖缺失**；原入口/配置/115与A01核心未改，不能把不改源码宣称为真实设备回归 |

完整mask/window CPU future NaN/Inf/脉冲与最后合法采样正控制通过；不冒充完整模型前向控制已通过。
HPC smoke会检查实际输入/归一化统计/encoder features/完整MDN，并保存原snapshot相同输入重叠复现或明确输入差异。

## 实际Japan旧结果CPU分析

`compare-existing-local`完整real_comparison_03 PASS，1310事件，各导出194265请求，所有ON/validation/seed42门和原byte/source/revision门通过。
1188条主要配对effect、288条单因子effect、180条空间effect，2000draws不缩减。
四交互和68%区间宽度补表也各自实际运行并seal。结果在[实际报告](../../reports/fe01_on_eval_20261022/README.md)。

前两次CPU进程退出137（SIGKILL）已保留；第三次改必要列/字符串共享/流式导出后通过，没有缩小人口或重抽有利分组。
内核OOM事件未另行采集，不将退出137报告为已认证的具体内核故障。HPC资源仍须pilot实测。
五个实际统计reader源码的SHA与最终实现一致，快照与开跑树身份保留；后来完善的未执行运营代码不能当作HPC执行证据。

## 合成端到端与图源

实际执行：保存480条station预测、5040条grid预测 → round_trip重读 → 研究全阈值shadow/误报 → TTUM全预算/固定面板 →20帧动态图、MP4、HTML和真实坐标分类交互页。
四个“模型条件”是固定合成数值生成器，不是真实模型。所有图和页面标SYNTHETIC。
本地ffmpeg可用，实际MP4生成成功；代码有GIF及HTML回退。
[浏览入口](../../reports/fe01_on_eval_20261022/synthetic/map_index.html)；[源身份核对](../../reports/fe01_on_eval_20261022/verification/synthetic_reader_identity.json)。

## 隔离与剩余门

原五个工作区HEAD/status/tracked diff/staged与接手快照一致，见[保护检查](../../reports/fe01_on_eval_20261022/verification/protected_worktrees.json)。
原115中50本地文件及46个A01核心文件逐SHA通过；65个dtbench源码本地缺失，完整运行门要求超算原vendor。
没有修改旧fe01/、fe01_a01/、fe01_review/、scripts/fe01_a01/、configs/fe01_a01/、旧tests/docs清单或RT55默认入口。

所有正式HPC stage均NOT_SUBMITTED。需用户依次手动提交CPU补数据、DCU smoke/pilot、全量station/normal-grid、latency、analyze/render/pack。
通信、连续触发与上游离线处理因果性未知；未观测网格无真值，面积NA；研究预算不是工程标准。
