# A01 DiTing OFF 配对读取修复（2026-10-20）

2026-10-21：用户已完成4号和pack，实际结果见
[中立结果报告](../../reports/fe01_a01_hpc_results_20261021/README.md)。
五项seal及16份预测均已核验；实际两份ON读取检查各记录174个默认解析truth变化，round_trip通过。
本页后续“尚未取得/待执行”描述为2026-10-20历史交付状态，不应据此要求重新提交。

分支 `exp/fe01-a01-absolute-amplitude-ablation`，基线
`236f72797ef4ccdec584cfbba952987e64a26437`。

## 本批事实与未确认项

用户提供的状态检查显示 0 TEAM OFF、1 PhaseNet OFF、2 EQT OFF、3 DiTing ON
均完整通过，4 DiTing OFF 未完成，pack 尚未开始。这里的完成来自用户在超算运行完整
result_gate 的输出，本地并未取得四组完整预测文件，不能据此生成新指标。

Job 30516145_4 的 err 显示：设备环境 ready，进入配对 compare 后，在 strict_pair 的
人口门抛出 `Locked population/labels/input/clock mismatch`。
该行同时调用左右两侧检查，日志没有直接标明失败侧；但 index4 的 left 来自 index3
已封存的预测 CSV，right 在 OFF export 中已通过同一人口门且身份列未再改变，数据流指向 ON 重读侧。
实际失败的 selected/epoch12 具体是哪轮，现有短日志未标明。
这些日志没有 Torch/ROCm 初始化失败或资源不足证据。

本地 pandas 2.2.3 可复现序列化风险：固定 RNG42 产生的 50000 个合成 float32 标签，
按原流程首次默认读取后再 to_csv，第二次默认读取有 483 个 float64 值变化；
round_trip 读取有 0 个变化。这是合成复现，尚不能证明超算真实 ON CSV 的每项差异。
[pandas 2.2 文档](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.read_csv.html)
规定了 C 引擎的 round_trip 转换选项。

另一个独立问题已由实际 stderr 确认：集群 sacct 不支持 `--array`，造成任务状态未知；
不影响 0–3 已完成文件的 SHA 验证。新版查询去掉该选项，仍按实际数组 task ID 匹配；
不会用父作业状态替代子项状态，也不把查询失败当完成。

## 一次上传与三个操作

本地上传包：
`artifacts/fe01_a01/diting_eval_fix_20261020/fe01_a01_diting_eval_fix_20261020.zip`。
上传到超算原 FE01 根目录，与旧两个上传目录同级，解压为
`a01_diting_eval_fix_20261020/`。设置 A01_EVAL_SESSION 为原评价 session。
包内 README 已给出用户本次 session 的完整路径。

在超算登录节点，手动提交**只重跑 4** 的单项 sbatch（1 节点 1 DCU）：

```bash
unzip -n fe01_a01_diting_eval_fix_20261020.zip
bash a01_diting_eval_fix_20261020/submit_diting_eval_fix.sh
```

提交成功返回作业号；不代表计算已完成。之后在登录节点只读检查：

```bash
bash a01_diting_eval_fix_20261020/submit_diting_eval_fix.sh status
```

五项完整通过、显示 PACK_READY=YES 且 pack 尚未提交时，手动提交 CPU sbatch：

```bash
bash a01_diting_eval_fix_20261020/submit_diting_eval_fix.sh pack
```

**此后 pack 使用本次新入口。** 旧 TEAM 修复入口内冻结的校验器不认识 4 的新来源，
不能再用它打包；不应改写 4 的执行 SHA 来冒充旧运行。
pack 完成后，从原 session 下载 review.tar.gz 和 review.tar.gz.sha256。
检查会显示完整下载路径。三个操作互相独立，不自动串联。

## 修复边界及失败诊断

不重新训练、不重跑 0–3、不重新 prepare，不覆盖原 plan/tools/env/seal 或失败 attempt。
认证核心、旧 115 文件、模型与选 epoch 规则保持不变。
在原 session/operations 新增内容寻址的 diting_eval_fix revision，允许评价编号只有 [4]。
新 seal 记录四份真实运营源码 SHA；验证器继续接受旧 TEAM [0] revision 和原 1–3 来源。
包中 eval_pack.sh 保留原 sbatch/srun 资源及实际 step 内初始化设备的流程。

每轮 OFF 导出前，先验证对应 ON CSV 的字节 SHA，再以 round_trip 读取，立即调用原
production=True 人口门。仍要求原人口/标签/输入/时钟 SHA、194265 行、1310 事件及
原 target-role 数量和物理时钟规则；不改标签、不放宽数值容差、不取交集、不跳过失败。
严格通过后才继续该轮 OFF 推理和原 compare。检查报告属于新 4 seal 的必需文件：
`paired_input_checks/selected.json`、`paired_input_checks/epoch12.json`。

若仍失败，进程会打印 `A01_SERIALIZED_ON_CHECK=FAIL details=...`，将预期和实测
字节 SHA、人口/标签时钟 SHA 或具体规则错误写入该 JSON，并保留失败 attempt。
此时不应 pack，也不反复原样重提；下载该小 JSON 和同作业 out/err 继续核查实际差异。
日志仍在原 session/slurm/eval_diting_off_<job>_4.out/err。

本地 17 项检查通过，使用真实 pandas 解析器、合成锁定表及假调度器；模型推理/统计导出测试明确使用替身。
旧 Python 环境路径失效，当前环境无 Torch，未重复运行依赖 Torch 的旧模型测试；
未更改模型/训练代码，实际权重载入与完整生产评价仍由超算原 loader 执行。
本轮未自动提交真实 sbatch/srun、训练、eval、pack 或访问 held-out test。
