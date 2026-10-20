# DiTing OFF 配对失败修复与旧 Slurm 兼容（2026-10-20）

基线 `236f72797ef4ccdec584cfbba952987e64a26437`，分支
`exp/fe01-a01-absolute-amplitude-ablation`。

用户提供的真实 Job30516145_4 日志在 strict_pair 人口门报错，设备 ready；
用户粘贴的完整状态检查显示 0–3 通过、4 缺失、pack 未开始。该状态未由本地完整结果复核。
traceback 同行有左右两侧检查，结合 OFF export 已通过同一门且身份列未修改，数据流指向 ON CSV 重读。
本地可用 pandas2.2.3 合成复现 50000 个标签二次默认读的 483 个浮点变化；round_trip 无变化。
这不是新 Japan 指标，也不是实际 ON CSV 根因已完全认证。

新增 serialized_on helper 在每轮 OFF 导出前按 seal SHA 验证 ON CSV，使用 round_trip 精确读取，
继续执行原 production 人口/标签/输入/时钟 SHA、数量、角色和物理规则；若失败留小型 JSON。
不替换标签、不取交集、不扩容差。selected/epoch12 检查成为新 4 seal 必需证据。
新增 [4] revision 保留原核心和 plan/tools/env/seal，兼容真实历史 TEAM [0] revision。
新入口默认只提交 4，status 只读，pack 手动提交 CPU 作业并核验三种执行来源。
旧 TEAM pack 不接受新 4，所以 pack 必须用本次新入口；未改签旧 seal。
status 去掉集群不支持的 sacct --array，不用父作业完成代替子项状态。

17 项本地检查通过，包含实际 pandas 往返、真实差异拒绝、提前停止推理、两轮流程和原子 seal、
只提交 4、旧来源兼容/篡改拒绝、无 Torch CPU pack、完整性判据和旧 sacct 查询。
推理/统计采用明确替身，未把合成结果当实际任务完成；旧 Torch 解释器已失效，当前环境无 Torch，
未重复跑旧模型套件。认证核心内容 SHA 不变，四个其他工作树快照保持一致。
原始日志保留在用户 outputs，不提交 raw logs；字节 SHA 和验证细节见 verification.json。

操作见 [超算提交说明](../../docs/ai/FE01_A01_DITING_EVAL_FIX.md)。
本轮未自动提交真实 sbatch/srun、训练/eval/pack，未访问 held-out test。
