# RT55 / RT56 项目上下文快照

更新日期：2026-09-02（Asia/Shanghai）

这是给 GitHub 连接器和新 AI 对话使用的紧凑入口。它不替代原始 metrics、NPZ、checkpoint 或 resolved config。定量分析前仍须核验具体 provenance。

## 1. 仓库和研究目标

- GitHub：`rular099/team_pytorch`
- 当前活跃分支快照：`zhangb/native-scale-adapter-scaling`
- 本文档编写前实现 HEAD：`2acbb75cf1266ab32d5f851913a34e7ac2c2ccbf`
- 任务：根据截至某个实时时刻可用的多台站三分量波形和输入/查询台站位置，预测查询台站的 PGA 概率分布。
- PGA 报告坐标：`log10(m/s^2)`。
- 当前主模型：冻结 DiTing waveform encoder，可训练 station adapter，TEAM 风格多台站 Transformer，面向目标台站的 PGA readout，同时保留震级/位置辅助任务。

核心代码入口：

| 路径 | 作用 |
| --- | --- |
| `train_light.py` | 配置加载、训练、微调和 checkpoint 初始化/恢复。 |
| `gemini_util_light.py` | 数据生成、实时截断、输入/目标选择和 random geometry masking。 |
| `gemini_models.py` | DiTing adapter、TEAM 上下文和概率 PGA readout。 |
| `eval_checkpoint.py` | split 评估、扰动对照、概率指标和 NPZ/JSON 导出。 |
| `pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json` | RT55 配置。 |
| `pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json` | RT56 配置，继承 RT55。 |
| `tools/eval_rt55_validation_normal_roll_slurm.sh` | RT55 normal/roll validation。 |
| `tools/eval_rt55_test_formal_slurm.sh` | RT55 正式 held-out test。 |
| `tools/run_rt56_random_geometry_slurm.sh` | RT56 zero-shot 与 mixed-random 微调提交入口。 |
| `tests/test_causal_random_geometry.py` | random geometry、target exclusion、环境展开等回归测试。 |

## 2. 数据划分

Japan 2000--2024 K-NET-only，按每个年度 shard 内事件互斥划分，不是 held-out-year 外推：

| split | events | station rows |
| --- | ---: | ---: |
| train | 9,627 | 146,842 |
| validation/dev | 1,383 | 22,336 |
| held-out test | 2,769 | 42,368 |

三个 split 都覆盖 2000--2024。训练/验证/测试的震级中位数约为 3.9/4.0/3.9；M>=6 事件分别为 167/24/42，解释高震级分桶时必须报告样本量。

## 3. RT55 定义和当前证据

RT55 使用 earliest/causal 风格的原任务输入与目标协议、legacy station adapter、显式 storage-valid waveform mask，关闭 DPK cache/token weighting/temporal residual。训练到 checkpoint epoch 34，validation objective 最优点为 epoch 32。RT55 的既有 config、checkpoint 加载和推理是后续开发必须保留的兼容基线。

已核验的主要结果：

| checkpoint / split / group | targets | MAE | RMSE | R2 | slope | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep20 held-out test, all | 170,767 | 0.1151 | 0.1777 | 0.7578 | 0.7783 | -0.8720 | 0.0909 |
| ep20 held-out test, non-input | 48,632 | 0.1947 | 0.2527 | 0.4029 | 0.5181 | -0.0073 | 0.1588 |
| ep32 validation, all | 89,770 | 0.1166 | 0.1806 | 0.7541 | 0.7720 | -0.8867 | 0.0934 |
| ep32 validation, non-input | 26,119 | 0.1986 | 0.2581 | 0.4117 | 0.5188 | 0.0023 | 0.1643 |
| ep32 validation waveform roll, all | 89,770 | 0.2490 | 0.3243 | 0.2069 | 0.4881 | 1.9937 | 0.2443 |

注意：当前可核验的 held-out test checkpoint 是 epoch 20。其输出目录名曾包含 `ep32_retry`，但 checkpoint provenance 显示 epoch 20；不能根据目录名把它称为 epoch 32 test。当前没有已核验的 epoch 32 held-out test。

重要现象：

- all-target MAE 被较容易的输入台站目标显著拉低，应用判断应同时查看 non-input 和 untriggered 目标。
- waveform roll 明显破坏结果，证明系统确实使用正确配对的波形信息；这不证明它已学会可靠的任意空间外推。
- 至少 5 个有效目标时，ep20 test 有 519/13,467（3.85%）个 realtime sample 的预测空间范围不超过 0.03 dex；其中满足条件的 518 个单输入台站 sample 全部近常数。MAE>=0.5 的严重 flat-diverse case 为 15 个，占合格 sample 的约 0.11%。

## 4. RT56 定义和当前证据

RT56 从固定 RT55 epoch 32 checkpoint 做 weight-only 初始化，重新开始 optimizer、scheduler、epoch 和 best-loss 状态；模型参数结构不变，DiTing encoder 继续冻结。训练 batch 为 50% RT55 原协议 + 50% causal random geometry；random 分支从截至当前时刻已触发且波形有效的台站中随机选择输入，并保证 PGA query target 排除输入台站。validation 使用固定 seed 的 100% random geometry，zero-shot 和微调结果按目标一一配对。

| model / validation protocol / group | targets | MAE | RMSE | R2 | slope | NLL | Brier | 1-sigma / 2-sigma coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RT55 ep32 zero-shot random, non-input | 75,654 | 0.2831 | 0.3820 | 0.0885 | 0.2959 | 1.4739 | 0.2498 | 0.442 / 0.692 |
| RT56 ep6 fine-tuned random, non-input | 75,654 | 0.2572 | 0.3313 | 0.3144 | 0.3478 | 0.2553 | 0.1999 | 0.655 / 0.941 |
| RT56 ep6 normal, all | 89,770 | 0.1338 | 0.2000 | 0.6984 | 0.7317 | -0.8163 | 0.1028 | 0.755 / 0.973 |
| RT56 ep6 normal, non-input | 26,119 | 0.2220 | 0.2838 | 0.2886 | 0.5138 | 0.0468 | 0.1776 | 0.672 / 0.960 |

严格配对的 random validation 中，MAE 相对 zero-shot 下降约 9.1%，RMSE 下降约 13.3%，R2 和概率校准明显改善。代价是 normal validation 遗忘：all-target MAE 相对 RT55 ep32 从 0.1166 升至 0.1338，non-input MAE 从 0.1986 升至 0.2220。

坐标排除审计在 1e-6 度容差下发现 75,654 个 random targets 中 0 个与有效选中输入台站重合；最小测地距离约 0.535 km。当前没有 RT56 held-out test，只有 seed 42。

## 5. 当前待回答的问题

以下是问题陈述，不代表已经选定解决方案：

1. 为什么带输入/查询位置编码的当前 readout 在单输入台站时仍退化为近常数空间输出？位置、波形和多台站上下文分别在何处被削弱或绕过？
2. 如何提高 arbitrary non-input query 的空间辨识和动态范围，同时保留概率校准？
3. 如何减少 RT56 random-geometry 适配带来的 RT55 normal-task 遗忘？
4. 需要哪些受控实验才能区分 sampler、mask、坐标编码、single-key cross-attention、event representation、loss 和数据覆盖的影响？
5. 如何在不查看 held-out test 的前提下定义 checkpoint 选择与 Pareto 标准，并通过 multi-seed 和受控 station-count sweep 验证？

## 6. 产物可见性

详细 RT55/RT56 报告、图、CSV、NPZ 和 checkpoint provenance 当前主要位于本地工作树、`../chaosuan_res/` 和超算运行目录，可能不会被 GitHub 连接器读取。本快照引用的详细本地报告目录为 `reports/rt55_rt56_current_results/`；在本次协作文档提交前它仍是未跟踪产物。

如果分析需要分桶明细、个例图、逐目标预测、attention 或 checkpoint tensor，应要求用户上传对应文件或让 Codex/超算导出；不能用本文摘要补造缺失数据。
