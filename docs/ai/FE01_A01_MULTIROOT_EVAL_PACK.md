# A01 分批训练后的 eval / pack（2026-10-14）

2026-10-20 当前进度：用户检查 0–3 完整通过，只需
[DiTing OFF 单项修复重提及新版 pack](FE01_A01_DITING_EVAL_FIX.md)。
保留本页原 session，使用该新入口处理 4 和最终 pack；下面是初版历史操作。

**本页初版的TEAM身份检查有误，已有session中只修复0的步骤见
[TEAM eval修复](FE01_A01_TEAM_EVAL_FIX.md)。保留运行中的1/2/3，最终pack使用修复入口。**

不需要合并或搬动best.pth/last.pth。新入口把五组完整训练的位置记录在清单中，
每组继续使用自己的原config和audit，评价结果写入一个新的统一目录。
旧retry/continue本身只分开保存operations日志，权重仍在原A01_OUTPUT_ROOT/run_id；
若你另建过batch，下面的扫描也支持权重分散在多个batch。

本轮基线 `df371f39d2a69df3c0e32cfbe5c935c8a5678ffd`，分支
`exp/fe01-a01-absolute-amplitude-ablation`。本地没有你的完整训练权重；
“五组已经完成”来自你的说明，脚本会在超算再次检查实际文件。

## 本地：上传一个包

将 `artifacts/fe01_a01/eval_pack_20261014/fe01_a01_eval_pack_20261014.zip`
上传到超算的FE01根目录：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
```

只上传这个新包。不要覆盖旧release，不用重新上传数据、模型权重或audit。

## 超算登录节点：准备并提交第一批eval

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
unzip -n fe01_a01_eval_pack_20261014.zip
export A01_ENV_FILE="$PWD/.a01_releases/3ecf060a3334/a01.private.env"
bash a01_eval_pack_20261014/eval_pack.sh prepare
bash a01_eval_pack_20261014/eval_pack.sh eval_first
```

prepare只做文件与元数据检查，列出下面五组的真实权重目录，并打印
`EVALUATION_OUTPUT=<新的统一输出目录>`；不加载大权重，不导入torch，不提交作业。
eval_first才调用sbatch，提交索引0、1、2、3；每项1节点1DCU，最多4项同时运行。
环境仍来自现有private env，使用原zb/ROCm。GPU检查在真正的srun step内执行。

| 索引 | 任务 | eval内容 |
|---|---|---|
| 0 | TEAM OFF | 本组selected/epoch12、冻结旧TEAM ON和配对比较 |
| 1 | PhaseNet OFF | 本组selected/epoch12、冻结旧PhaseNet ON和配对比较 |
| 2 | EQT OFF | 本组selected/epoch12、冻结旧EQT ON和配对比较 |
| 3 | DiTing ON | 本组selected/epoch12，为下一步OFF比较准备参照 |
| 4 | DiTing OFF | 本组selected/epoch12，与本次索引3结果严格配对 |

## 第一批eval完成后：提交DiTing OFF

仍在同一个FE01根目录执行：

```bash
bash a01_eval_pack_20261014/eval_pack.sh eval_diting_off
```

这里只提交索引4，1节点1DCU。索引3两份完整评价未通过seal时会停止，不会提交。
每次eval同时保留原selected checkpoint与固定epoch12结果，不重新挑选模型。

## DiTing OFF也完成后：提交pack

```bash
bash a01_eval_pack_20261014/eval_pack.sh pack
```

pack也是sbatch任务，使用CPU，不申请DCU；五组完整eval及所需ON/OFF比较未齐时会停止。
它从各训练根读取轻量证据，合并本次成功eval，生成：

```text
<EVALUATION_OUTPUT>/review.tar.gz
<EVALUATION_OUTPUT>/review.tar.gz.sha256
```

下载这两个文件即可交回分析。pack结束时打印 `A01_REVIEW_PACKAGE=` 的完整绝对路径。
包包含原config/audit/采样/诊断/pilot、训练曲线、checkpoint身份与SHA、成功评价/比较，
以及可找到的独立训练兼容入口身份记录；不包含权重、波形、私有env、失败评价attempt、
逐层数组或完整sample journal。训练期12份validation导出不重复装入包。
源eval seal核对原文件，包内A01_REVIEW_MANIFEST核对脱敏后的实际归档字节，两种SHA作用不同。

## 目录查找、失败与重提

prepare默认只扫描当前输出根、FE01根下的 `outputs/*` 和
`.a01_releases/*/outputs/*`，不会递归读取HDF5或加载checkpoint。
按12轮/2544updates、原源码/config/audit、diagnostics/pilot、best/last是否齐全筛选，
不按目录时间或文件新旧选择。同一任务有两份完整候选时停止，不猜哪份最好。
拒绝原因保存在上传工具目录的 `eval_candidates.json` 和 `eval_pack_prepare_*.log`。
若停止，请下载这两个小文件交回定位，不要先复制权重或删除失败目录。

如果结果确实放在上述范围之外，可在prepare前设置
`A01_EVAL_SEARCH_ROOTS` 为额外batch输出根，多个路径用冒号分隔。
多个完整候选可用 `A01_EVAL_RUN_MAP` 指向JSON对象，key是字符串索引，value是该组
包含best/last/resolved_config的run目录；只需写要消歧的项，仍会严格验证全部原身份。
重构原配置只恢复冻结模板中的路径变量，并要求完整配置SHA等于原audit；
不会改写旧配置、旧lock、旧checkpoint或重新签名。

所有本次计算日志在 `EVALUATION_OUTPUT/slurm/`；上传目录另保存每次提交的log。
每个eval先写独立 `attempts/`。两份导出和所需比较全部成功后，才原子发布到
`results/indexN/` 并生成EVAL_COMPLETE及文件SHA。失败attempt原样保留。
再次调用eval_first只提交0–3中尚未成功的项；eval_diting_off同理；不重评已成功项。
pack只打包成功发布并通过seal的结果，不把失败残留当作完成。

默认eval申请24小时、主机内存64G；可在提交前设置 `A01_EVAL_TIME`、
`A01_SINGLE_MEM`。pack默认2小时/8G，可用 `A01_PACK_TIME`、`A01_PACK_MEM`。
资源变量从提交环境继承；若同名变量也写在private env中，则以private env为准。
每次只提交指定阶段，不自动串联、续队列、重新训练或开启test。

原115文件与A01认证核心SHA保持不变，新入口独立冻结并记录SHA。
它调用原export、export_old_ON、compare和review_package，保留原validation十cell、
精确MDN概率计算和事件bootstrap定义。真实完整评价及超算扫描均为本轮NOT_RUN；
本地假调度器/合成身份测试不能代替超算结果。
