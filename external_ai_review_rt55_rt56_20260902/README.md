# External AI review bundle: RT55 / RT56

生成日期：2026-09-02（Asia/Shanghai）

主入口是 `REQUEST_TO_EXTERNAL_AI_CN.md`。该文档只描述任务、代码事实、实验协议、当前结果和待分析问题，不包含预设原因或候选解决方案。

## 目录

```text
REQUEST_TO_EXTERNAL_AI_CN.md       外部审查请求与问题描述
MANIFEST_SHA256.txt                除清单自身外，包内全部文件的 SHA-256
code/
  team_pytorch/                    当前训练、模型、采样、评估、启动与配置代码
  ditingbench/                     gemini_models.py 实际依赖的 DiTing frontend 代码
results/
  run_snapshots/                   实际运行时合并配置
  rt55/
    formal_metrics/                正式 metrics JSON
    tables/                        聚合、分桶、flat audit 和案例数据
    figures/                       关键 PNG
    training_diagnostics/          精简训练与 readout diagnostics CSV
  rt56/                            同上
  shared/                          checkpoint provenance 与指标复算审计
```

## 代码版本

- team_pytorch Git commit: `2acbb75cf1266ab32d5f851913a34e7ac2c2ccbf`
- commit time: `2026-08-25 08:44:06 +0800`
- commit subject: `Fix rt56 inherited environment expansion`
- 报告生成脚本和结果目录在打包时属于工作区未跟踪产物；其内容也由 `MANIFEST_SHA256.txt` 固定。

## 未包含内容

- 模型 checkpoint；
- 日本年度 HDF5；
- 原始逐目标 NPZ；
- 约 20 MB/张的 scatter source CSV；
- Slurm 运行环境本身和预训练权重；
- 缓存、`__pycache__`、日志压缩包和无关实验配置。

因此该包适合静态代码审查和结果诊断，但不能在没有外部数据、checkpoint 和 DiTing 预训练权重的情况下完整复现训练或验证。

## 依赖提示

Python 依赖列表见：

- `code/team_pytorch/requirements.txt`
- `code/ditingbench/requirements.txt`

实际训练运行在 Slurm/DCU 环境中，启动脚本中的绝对路径是原运行环境记录，不保证在其他机器可直接使用。
