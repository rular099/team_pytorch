# 当前工作区同步范围（2026-11-02）

当前机器已完成迁移和worktree整理。本次直接在既有分支提交，不创建分支或worktree。

## 提交Git

- 已有代码、任务书和交接记录。
- 报告正文、讲稿、复现脚本、聚合/分层CSV、metrics/provenance JSON。
- 报告正文引用的必要PNG预览；本轮单文件不超过1MiB。
- 只提交原文件，不改变模型、实验配置、科学计算逻辑或已有产物内容。

## 本地保留／用U盘拷贝

- 权重、checkpoint、HDF5、NPZ和大体积逐目标CSV。
- PPT/PDF/SVG成品、渲染预览/草稿、重复压缩包和外部审查包中的重复代码/数据。
- 原始超算产物和日志；已有轻量报告足够时不重复上传。
- 缓存、虚拟环境、bytecode和私有配置不纳入Git。

本地产物仅添加忽略规则，未删除；忽略规则不代表U盘已经完成备份。

## 本轮提交

| 既有分支 | 提交 |
|---|---|
| `zhangb/native-scale-adapter-scaling` | `579e1d920ea3f1e224f747e36d7332cb53926ce1` |
| `exp/fe01-feature-extractor-site-effects` | `577a0cec43643597a61209209a770aeecf5fefa0` |
| `exp/v01-velocity-prep-padding-control` | `555ed0d20e049c0a88f45e7c36888dd64fdfcc95` |

FE02、A01等没有本轮未提交变更的工作区保持原状；不把较旧的本地HEAD推回覆盖较新的远端。
