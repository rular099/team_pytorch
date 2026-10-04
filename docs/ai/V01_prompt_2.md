请执行 V01 validation closure，先阅读附件：
V01_REVIEW_AND_CODEX_HANDOFF_20261004.md

本轮不重训、不重新preflight、不执行旧all/recover、不恢复RT62。

[AI-HANDOFF]
task_id: 20261004-v01-validation-closure
repo: rular099/team_pytorch
branch: exp/v01-velocity-prep-padding-control
base_commit: 74c55aa5442b4200961c88ceee3d11af5033275d

goal:
  保留已有三臂epoch8权重，最小化修复normal评估，
  明确A-pair重复/空样本分母，补齐必要身份并完成原定验证。

verified_facts:
  - 四格速度random为1194events、8358rows、84259目标观测。
  - 三臂保存张量起点相同，epoch8，optimizer steps1496。
  - 五个normal均在Found event without PGA idx=7失败。
  - AA random存在45条额外重复，尚不能公平横比。
  - _select_realtime_cutout中np.asarray(bool)加&=会修改caller mask。
  - normal直接传正式station_valid_full；
    random/mixed-enabled路径提前使用临时mask。
  - MM总体改善伴随平均偏差减小；阈值以上目标点/概率指标变差。

inferences:
  - 别名副作用是normal错误的具体候选根因，
    但实际idx7必须用真实cache及运行源码确认。
  - AA重复与_EmptySample替代一致，尚未逐条认证。
  - 当前结果不支持本次剂量的缺失造成明显损害，
    也不证明删除前缀普遍有益或已证实正则化机制。

files_to_inspect:
  - 项目四个入口文档、原V01规范和本次review request/result
  - reports/v01_velocity_padding_validation_20261003/*
  - gemini_util_light.py：
    _select_realtime_cutout、_get_one、__getitem__、JointGenerator
  - eval_checkpoint.py：build_datasets、run_inference
  - tools/build_v01_paired_manifest.py
  - tools/summarize_v01_results.py
  - 现有V01 tests及Slurm入口

constraints:
  - HEAD变化先报告diff，保留用户工作区和全部旧产物。
  - 不改模型/DiTing/loss/PGA定义，不更新任何训练权重。
  - 新行为仅显式V01 validation opt-in，RT55旧默认不变。
  - 不把ValueError改成无条件skip，不用后续事件替代当前请求。
  - 不访问held-out test，不挑epoch、阈值、剂量或seed。
  - 不新增硬编码私有路径，不覆盖旧eval_retry1。

proposed_change:
  - 先建立最小真实trace：
    idx7 -> shard/event/row_selector；
    保存source/query身份、PGA/坐标/P sample和clock前后mask。
  - 若确认别名根因，V01新模式只对局部clock mask作copy，
    分离query有效性和input/clock eligibility，截止定义不变。
  - 新增显式request ledger和空样本处理：
    每个请求恰有一次预测/弃权/错误，禁止邻样本替代。
  - 从现有cache/NPZ生成sensor-ID及provenance sidecar；
    先回传既有preflight/cohort/split/source材料，不重建数据。
  - AA优先离线恢复，只有重复逐字段一致才去重；
    始终保留原文件、缺失请求和无预测率分母。
  - 新报告补阈值上下、untriggered、单站和early/all7配对CI，
    以及MSE/bias²/去均值误差分解。
  - 标清当前空间ratio是np.ptp中位数；
    新增P95-P05口径时另列，不替换历史指标。
  - 新增生产generator回归tests与evaluation-only补跑脚本。

acceptance_checks:
  - 真实失败行修复前/后trace；合成反例不能代替真实行验证。
  - clock不修改query mask；窗外/未知query pick不自动取消合法PGA。
  - query-only行绝不进入有效输入。
  - legacy与mixed训练路径兼容；旧random未受补丁影响有证据。
  - 原三臂epoch8/step1496及权重hash不变。
  - 实际train/dev计数、split/encoder/source身份补全或明确标未知。
  - request数量 = predicted + 各类无预测/错误数量，无静默替代。
  - 所有配对一对一，unmatched显式统计，不靠inner join丢目标。
  - 五格normal据实报告样本数，不套历史RT55计数。
  - CI用5000次event-cluster，seed20260915；
    新分层CI只解释结果，不用于调实验。
  - bash -n、py_compile、针对性及旧兼容tests真实执行。

hpc_followup:
  - 交付tools/complete_v01_validation_slurm.sh或同职责脚本。
  - 默认DRY_RUN=1，正式提交需要显式确认；由用户提交。
  - 复用weights_vfull_retry1、weights_vmissing、weights_apair。
  - 默认只补FF/FM/MF/MM/AA五个normal，无训练分支。
  - AA random不能可靠离线恢复时才增加一格，并写明理由。
  - 四格速度random原则上复用；若发现数值受影响先报告，
    不混合新旧协议，也不自动扩大运行。
  - 新输出目录、源码manifest和job依赖留档；
    不再次调用整体recovery/all。

risks:
  - 修复可能恢复旧代码误删的query，必须给新协议版本和分母。
  - 同seed/step不能认证实际训练样本完全相同。
  - 共同可评估A/V交集存在选择偏差，需同步报告弃权率。
  - 继承的实时去均值问题单列，不在本轮偷偷改变数值协议。

open_questions:
  - none for bounded implementation。
  - 只有发现必须新增训练或改变数值协议时，
    返回证据和待决策事项，不自行扩大范围。
[/AI-HANDOFF]

最终返回[CODEX-RESULT]，给出精确result commit、
实际测试结果、RT55兼容性、补跑脚本完整内容及可复制命令。
明确区分代码完成、本地测试、作业提交和结果完成。
