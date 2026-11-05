# TEAM smoke 提交入口参数兼容修复

任务 `20261105-fe01-v4-team-smoke-shell-args`，基点
`fe358c38f061a06d1b1e66c57574a9f44b3c0fc7`，分支 `exp/fe01-on-warning-map-evaluation`。
用户报告 `args[@]: unbound variable`，未提供该次shell版本和完整行号。
原提交入口在严格nounset模式下对空参数使用通用args数组，并在source私有环境后再次展开该同名数组。
参数处理具有空数组兼容和环境变量覆盖风险，错误发生在调用Python/sbatch之前。

修复移除数组，直接转交 `"$@"`，提交脚本私有变量使用统一前缀；支持无参数、显式run ID、
空格路径、两种env参数形式，缺少env值时给出明确错误。
原受控TEAM检查、阈值、运营身份记录、原运行目录/环境/成功结果均保持。
只需复制新小包和SHA到超算项目根目录，按交付消息校验解压，运行同名 `submit_team_smoke.sh`。
详细步骤见[运行说明](../../docs/ai/FE01_V4_TEAM_SMOKE_FIX.md)。

22项专项测试通过，包括5项新增的实际shell入口测试；使用记录argv的解释器替身，未触及scheduler/模型。
测试故意在私有env设置同名args/tool_dir/env_file，验证新入口不会丢失或污染参数。
已有17项运营检查同时通过，编译、import和shell语法另行验证。
原超算Bash版本未取得，没有把本机GNU Bash5的测试声称为该版本实测。

证据：[完整本地验证](local_verification.log)、[范围与身份记录](verification.json)。
正式修复任务 **NOT_SUBMITTED**；本地无生产checkpoint/Japan HDF5/DCU，实际TEAM模型结果仍待用户运行。
