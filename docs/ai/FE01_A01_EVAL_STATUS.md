# A01 五项评价状态检查与 pack 时机（2026-10-15）

2026-10-20 更新：用户确认 0–3 完整通过，4 配对失败。当前使用
[DiTing OFF 单项修复包](FE01_A01_DITING_EVAL_FIX.md)内的 status/重提/pack 入口；
它兼容不支持 sacct --array 的集群，并识别 4 的新增执行来源。
下文旧包及旧 pack 命令是 2026-10-15 的历史交付，不用于本次修复后的 4。

适用分支 `exp/fe01-a01-absolute-amplitude-ablation`；实现基线
`f29ecdcb4dcc4bdce9eaf560bcbc9453c2b4506e`。
用户报告 TEAM 修复入口执行成功；本地尚未取得本批实际评价输出，不能据此判定五项均完成。

## 五个编号分别是什么

| 编号 | 新训练模型 | 评价内容 | 提交入口 |
| --- | --- | --- | --- |
| 0 | TEAM OFF | selected / epoch12，复用旧 ON 并做配对比较 | 修复入口默认模式，只重提这一项 |
| 1 | PhaseNet OFF | selected / epoch12，复用旧 ON 并做配对比较 | 原 eval_first 已提交 |
| 2 | EQT OFF | selected / epoch12，复用旧 ON 并做配对比较 | 原 eval_first 已提交 |
| 3 | matched DiTing ON | selected / epoch12，供 4 使用 | 原 eval_first 已提交 |
| 4 | matched DiTing OFF | selected / epoch12，与 3 配对比较 | 3 完整通过后单独提交 eval_diting_off |

以上“已提交”来自此前用户提供的情况，不代表已完成。修复入口默认仅提交 0，
不会替用户提交 4 或 pack。`SUBMITTED` 只说明提交入口已返回作业号。

## 一次性上传小型只读检查包

将本地 `artifacts/fe01_a01/eval_status_20261015/fe01_a01_eval_status_20261015.zip`
上传到超算 **FE01 根目录**，与已有两个上传目录同级。
在该目录的登录节点执行，具体 session 路径使用旧评价输出目录：

```bash
unzip -n fe01_a01_eval_status_20261015.zip
export A01_EVAL_SESSION="$PWD/outputs/你的原评价session目录名"
bash a01_eval_status_20261015/check_eval_status.sh
```

之后只需再次执行最后一行检查。也支持读取相邻旧上传目录的 `eval_session.path`。
本包只包含状态脚本、字节相同的只读结果校验 helper、README 和 SHA256SUMS；
不修改 session/plan、原 tools、revision、private env、结果或训练权重，也不会登记新 revision。
不需要额外 sbatch 或 DCU 分配：该检查只读 JSON、计算文件 SHA256、查询调度状态，
不导入 torch 或载入 checkpoint，Python 使用 `-B` 禁止写 bytecode。

## 如何看检查输出

每行显示编号、模型、结果状态、最近一次提交的作业号及 Slurm 状态。
完成标准复用修复版 pack 的 `result_gate`：两套 epoch 所需结果、对照与比较全部存在，
seal 为 PASS，validation/plan/核心源码/逐项执行来源一致，所有声明文件 SHA256 一致。
兼容 0 的真实修复来源及其余四项旧来源；不改变旧 seal。
只有 selected 输出、失败 attempt、Slurm 显示 COMPLETED 或 stderr 为空都不足以通过。
完整 SHA256 检查可能耗时几分钟，每校验一项会立即打印一行。

按 `jobs.tsv` 追加顺序选最近一次提交，因此旧 0 失败不会覆盖新 0 重提。
分别用 [squeue](https://slurm.schedmd.com/squeue.html) 展开查询活跃数组项，
用 [sacct](https://slurm.schedmd.com/sacct.html) 展开查询已结束数组项，队列结果优先。
查询缺失、权限不足或超时标成未知，不把队列消失当作完成。
有成功 seal 的项目以完整校验为准，不受旧失败作业状态影响；失败或未知项打印对应 err 路径。

## 下一步与 pack

- `3 完成，4 未见提交记录`：检查会显示以下命令。可以与其他仍在跑的评价并行；本次只提交 4。

  ```bash
  bash a01_eval_pack_20261014/eval_pack.sh eval_diting_off
  ```

- `PACK_READY=YES` 且 pack 尚未提交：说明五项全部校验通过，可以手动提交 CPU pack。

  ```bash
  bash a01_team_eval_fix_20261014/submit_team_eval_fix.sh pack
  ```

- pack 已排队或运行：等待并重新检查，不重复提交。pack 失败或文件不完整：先检查原 session/slurm 中 pack 日志。
- `PACK_STATUS=COMPLETE`：归档及 SHA256 文件匹配。检查同时显示原 session 下两个下载文件路径。

pack 必须用 TEAM 修复入口，保留 0 的真实新执行身份；旧 pack 无法接受这一项新来源。
状态检查不会自动串联或重提任何作业，也不代表新的真实超算结果已经取得。
