# A01 分批结果的 eval / pack 交付

仓库rular099/team_pytorch，分支exp/fe01-a01-absolute-amplitude-ablation，
基线`df371f39d2a69df3c0e32cfbe5c935c8a5678ffd`。
用户2026-10-14报告train全部完成、结果位于不同目录。本轮没有新的完整超算训练产物，
所以不把此说明提升为本地已认证的checkpoint完成事实。

此前retry/continue源码显示，operations按作业分目录，而权重仍写原A01_OUTPUT_ROOT/run_id。
用户也可能新建过batch；新入口同时支持同根、多根训练，prepare会打印实际选中的五个目录。

新增[提交脚本](../../scripts/fe01_a01_ops/eval_pack.sh)与
[运营runner](../../scripts/fe01_a01_ops/eval_pack.py)，部署在原认证源码之外。
[中文说明](../../docs/ai/FE01_A01_MULTIROOT_EVAL_PACK.md)区分本地上传、登录节点准备/提交、
计算节点完整身份检查和导出、CPU打包。

已实施：

- 登录节点只用stdlib和原配置/preflight工具，扫描12轮/2544updates、原代码/配置/门控、
  best/last、原audit对应关系；有重复完整候选就停止，失败/未完成候选不混入。
- 从冻结模板恢复记录中的路径变量，完整source config SHA必须等于旧audit。
  保留每个batch的原config/output_root和原lock，不修改路径后重签，不搬动权重。
- 另建统一评价session，冻结run清单、原输入stat/SHA及运营代码SHA。
  计算节点严格加载best/last内部身份、world16、完整更新预算、初始化和encoder元数据；
  最终仍由原export/a01_checkpoint严格加载模型状态、数据声明和原验证人口。
- selected沿用已注册训练期selection，另报固定epoch12；前三家族复评冻结旧ON，
  DiTing OFF与本次匹配ON比较。原十个validation cell、精确logits概率指标、事件bootstrap不变。
- 用户分别提交eval_first(0–3)、eval_diting_off(4)、pack；没有自动sbatch/srun或队列链。
  GPU初始化在actual srun中；跨节点使用session共享脚本而非Slurm本地spool路径。
- eval失败只留下attempt；两份导出及所需比较全部成功后原子发布完整结果和SHA seal。
  重提只补未完成项，pack要求五项和比较全部完整、seal通过，不覆盖旧包。
- CPU包收集多源轻量证据和独立训练兼容身份，排除权重/波形/失败attempt/私有env。
  source seal与脱敏后archive seal分别说明，避免误用SHA。

[verification.json](verification.json)记录本地测试、上传包校验、核心指纹和原工作树保护。
72项本地测试通过，1项实际CUDA/ROCm恢复测试因本地无设备跳过。
跨批次/原配置路由、checkpoint身份拒绝、重复候选、配对初始化、失败attempt保留、
手动阶段顺序、跳过完成项、缺评价阻止pack、包内排除大权重、CPU无需torch、
真实step环境入口和shared spool路径均有本地测试。

**真实HPC prepare/完整eval/pack：NOT_RUN；新作业：NOT_SUBMITTED；held-out test：NOT_ACCESSED。**
