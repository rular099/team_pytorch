# A01 eval 索引0 TEAM身份检查修复（2026-10-14）

2026-10-15：不确定哪些已完成时，先使用[只读五项评价检查](FE01_A01_EVAL_STATUS.md)。
它会验证新旧来源的完整结果，显示是否该提交 4，以及何时可以 pack。

用户订正后的实际情况：eval_first中1、2、3正常运行，只有0退出；
错误位于checkpoint_metadata的初始化/encoder组合检查。以这条订正为准。
基线`e1ad3bf547392738ee5e120b7d4d33f8e42d7d30`，分支
`exp/fe01-a01-absolute-amplitude-ablation`。

TEAM scratch的StationFrontend没有冻结encoder，训练后encoder SHA正常会变化。
原A01 checkpoint loader已区分scratch与frozen，但新增运营检查错误地对全部家族要求
encoder SHA等于audit初态。这是运营入口缺陷，不能据此要求TEAM重训。
组合报错不提供两项SHA的实际值；公共初始化记录仍需独立验证，不跳过该检查。

## 本地上传

上传 `artifacts/fe01_a01/team_eval_fix_20261014/fe01_a01_team_eval_fix_20261014.zip`
到原FE01根目录，与旧 `a01_eval_pack_20261014/` 同级。解压后是新的
`a01_team_eval_fix_20261014/`。旧session的plan.json、tools、runtime.private.env、
已经运行的1/2/3及训练权重均保持原样。
包内README按本次用户给出的session写好完整路径；私有路径没有写入新增源码。

## 登录节点只重提0

在原FE01根目录解压新ZIP，令A01_EVAL_SESSION指向**已经创建的原评价session**，
然后执行：

```bash
unzip -n fe01_a01_team_eval_fix_20261014.zip
bash a01_team_eval_fix_20261014/submit_team_eval_fix.sh
```

新脚本优先读取A01_EVAL_SESSION；未设置时读取相邻旧上传目录的eval_session.path。
默认仅提交索引0的sbatch任务，1节点1DCU，不重新提交1/2/3。
不得重新prepare或把修正版覆盖到原session/tools；原冻结脚本仍供正在运行的任务使用。
提交日志在新上传目录，计算日志在原session/slurm/eval_team_<作业号>_0.out/err。

修复仍保留原config/协议/源码/world16/epoch/更新数/公共初始化检查。
TEAM从model_state_dict提取encoder真实张量，计算SHA并匹配checkpoint记录的训练后SHA；
PhaseNet/EQT/DiTing继续要求冻结encoder SHA等于audit初态，再由原loader核验实际模型。
原selected和epoch12评价、冻结旧TEAM ON和配对比较均不改变。

## 后续4与pack

索引3成功完成后，DiTing OFF继续使用原入口：

```bash
bash a01_eval_pack_20261014/eval_pack.sh eval_diting_off
```

五项成功后，**pack改用新入口**：

```bash
bash a01_team_eval_fix_20261014/submit_team_eval_fix.sh pack
```

这是单独的CPU sbatch任务，不会自动串联评价或打包。新入口识别0的新执行身份及
1/2/3/4的原执行身份，全部seal/比较完整才打包。旧pack会因0执行SHA不同而拒绝，
因此不再用旧入口pack，也不改签0的seal让它伪装成旧源码执行。

仍从**原评价session**下载review.tar.gz和review.tar.gz.sha256。
包内保留原评价plan、各索引执行SHA、修复revision和pack实际执行身份。

## 运行中的session如何保持兼容

新代码只在session/operations下登记独立只读revision，绑定原plan SHA、原认证核心SHA、
原两份冻结工具SHA和三份修复工具SHA。允许修正版评价的索引只有0；pack另行允许。
新0结果引用真实的新执行SHA和revision，不篡改原plan或其他结果seal。
原1/2/3完成时继续写自己的原执行SHA，原4也继续使用原工具；新版pack按每项来源核对。
revision被篡改或用于别的索引会停止。所有失败日志/attempt保留。

本地验证包括真实小型TEAM前向/Adam更新、旧检查复现、新检查通过、原loader严格载入、
错误权重/初始化拒绝、三类冻结encoder拒绝错误SHA、只提交0、旧工具/plan保持字节不变、
新旧seal混合CPU打包。当前84项通过、1项实际设备测试跳过。
真实超算0号重评及后续pack仍待用户运行，本轮未提交实际作业。
