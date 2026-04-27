# TEAM PyTorch DiTing 项目说明

## 项目简介

本项目是在 TEAM (Transformer Earthquake Alerting Model) 思路基础上扩展的 PyTorch 实验代码，重点用于结合 DiTing backbone 的多台站地震预警和 PGA 预测实验。

代码支持从地震波形、台站元数据和 PGA 目标中训练并评估两类模型：

- single-station model：单台站预训练模型，用于学习单台站波形特征；
- full model：多台站 TEAM-style Transformer 模型，用于联合预测震级、震源位置和 PGA。

当前主要任务包括：

- 震级预测；
- 震源位置预测；
- 多目标台站 PGA 预测；
- single-station 预训练；
- full model 多卡训练；
- 训练后自动评估 full model 和 single-station model。

## 代码结构

核心文件和目录如下：

```text
train_light.py              PyTorch 主训练入口
train_light_slurm.sh        Slurm 训练和训练后评估脚本
eval_checkpoint.py          checkpoint 评估脚本
gemini_models.py            TEAM/DiTing 相关模型结构
gemini_util_light.py        数据生成、预处理和 batch 组织
loader_light.py             训练数据加载
pga_configs/                PGA 相关训练配置
diting/                     DiTing backbone 相关代码和配置
requirements.txt            Python 依赖列表
runs/                       TensorBoard 输出目录
logs/                       训练标量和诊断日志
<weight_path>/              checkpoint、config 备份和评估结果
```

## 环境依赖

依赖文件来自 `ditingbench/requirements.txt`，已经复制到当前项目：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- PyTorch
- xFormers
- NumPy / SciPy / pandas / h5py
- Matplotlib
- ObsPy
- SeisBench
- PyYAML
- tqdm
- DiTing 相关依赖

实际 GPU/DCU 环境中的 PyTorch、xFormers、ROCm 或 CUDA 版本需要与运行机器匹配。

## 配置文件

训练由 JSON 配置文件控制，示例配置位于：

```text
pga_configs/
magloc_configs/
```

当前 PyTorch 训练主要使用 `pga_configs/` 下的配置。

常用字段：

```json
{
  "model_params": {},
  "training_params": {}
}
```

`model_params` 控制模型结构，例如：

- `max_stations`
- `n_pga_targets`
- `use_coords_abs`
- `use_coords_rel`
- `pga_mixture`
- `magnitude_mixture`
- `location_mixture`

`training_params` 控制训练和数据，例如：

- `data_path`：训练数据路径；
- `weight_path`：checkpoint 和评估结果输出目录；
- `epochs_full_model`：full model 训练轮数；
- `single_station_pretrain`：single-station 预训练配置；
- `res_comps`：参与训练的输出分量，例如 `["mag", "loc", "pga"]`；
- `res_weight`：各输出分量 loss 权重；
- `full_model_loss`：full model loss 类型；
- `generator_params`：数据生成和采样参数。

`weight_path` 必须为空，除非启动脚本在训练前自动清理它。

## 训练方式

### 直接运行 Python

基础命令：

```bash
python train_light.py \
  --config pga_configs/your_config.json \
  --diting_config diting/config/your_diting_config.yml \
  --diting_pretrained /path/to/diting_checkpoint.pt
```

常用额外参数：

```bash
--test_run
--overfit_n 16
--no_multiprocessing
--skip_single_station_pretrain
--single_station_only
```

### 使用 Slurm 脚本

推荐在集群上使用：

```bash
bash train_light_slurm.sh <config.json> [train_light.py extra args...]
```

脚本负责：

- 提交 Slurm job；
- 初始化运行环境；
- 启动 `torchrun` 分布式训练；
- 可选清理旧的 `weight_path`；
- 训练成功后自动运行评估。

常用环境变量：

| 变量 | 说明 |
| --- | --- |
| `WORKDIR` | 仓库路径。 |
| `DITING_CONFIG` | DiTing YAML 配置。 |
| `DITING_PRETRAINED` | DiTing 预训练 checkpoint。 |
| `SLURM_PARTITION` | Slurm 分区。 |
| `SLURM_NODES` | 节点数。 |
| `SLURM_GPUS_PER_NODE` | 每节点 GPU/DCU 数，也是每节点训练进程数。 |
| `SLURM_CPUS_PER_TASK` | 每个 Slurm task 的 CPU 数。 |
| `RESET_WEIGHT_PATH` | 设为 `1` 时训练前删除 `weight_path`；设为 `0` 时保留。 |
| `RUN_EVAL` | 设为 `1` 时训练后自动评估；设为 `0` 时跳过。 |
| `EVAL_CHECKPOINT` | 手动指定 full model checkpoint。 |
| `EVAL_SINGLE_STATION_CHECKPOINT` | 手动指定 single-station checkpoint。 |
| `EVAL_OUTPUT_TXT` | 手动指定评估文本输出。 |
| `EVAL_OUTPUT_NPZ` | 手动指定评估数组输出。 |
| `EVAL_DEVICE` | 手动指定评估设备，例如 `cuda:0`。 |

示例：

```bash
RESET_WEIGHT_PATH=1 RUN_EVAL=1 bash train_light_slurm.sh pga_configs/your_config.json
```

如果训练命令中传入 `--overfit_n`，Slurm 脚本会把该参数继续传给 eval，保证训练和评估使用一致的数据切分。

## 训练流程

完整训练流程如下：

```text
读取 JSON config
加载数据
构造数据生成器
可选 single-station pretrain
构造 full model
加载 DiTing backbone / pretrained checkpoint
启动 DDP 多卡训练
保存 full_model_*.pth checkpoint
训练成功后自动评估
```

full model checkpoint 会按 epoch 保存为：

```text
<weight_path>/full_model_<epoch>.pth
```

single-station checkpoint 通常保存为：

```text
<weight_path>/single_station_best.pth
<weight_path>/single_station_final.pth
```

## 训练后评估

训练完成后，`train_light_slurm.sh` 默认会自动调用：

```bash
python eval_checkpoint.py
```

评估包括两部分：

- full model：默认选择 `<weight_path>/full_model_*.pth` 中 epoch 最大的 checkpoint；
- single-station model：优先使用 `<weight_path>/single_station_best.pth`，没有则使用 `<weight_path>/single_station_final.pth`。

如果配置启用了 `single_station_pretrain`，但找不到 single-station checkpoint，脚本会直接报错，避免静默跳过 single-station 评估。

默认评估输出：

```text
<weight_path>/eval_results.txt
<weight_path>/eval_results.npz
```

其中：

- `eval_results.txt` 保存完整评估日志和指标；
- `eval_results.npz` 保存 `eval_checkpoint.py` 输出的原始数组。

也可以手动运行 eval：

```bash
python eval_checkpoint.py \
  --config pga_configs/your_config.json \
  --diting_config diting/config/your_diting_config.yml \
  --diting_pretrained /path/to/diting_checkpoint.pt \
  --checkpoint <weight_path>/full_model_291.pth \
  --single_station_checkpoint <weight_path>/single_station_best.pth \
  --output <weight_path>/eval_results.npz
```

如果需要保存文本日志：

```bash
python eval_checkpoint.py ... > <weight_path>/eval_results.txt 2>&1
```

## 输出文件

一次完整训练通常会产生：

```text
<weight_path>/config.json
<weight_path>/single_station_best.pth
<weight_path>/single_station_final.pth
<weight_path>/full_model_init.pth
<weight_path>/full_model_<epoch>.pth
<weight_path>/eval_results.txt
<weight_path>/eval_results.npz
logs/<weight_path>/*.csv
runs/<weight_path>/
```

说明：

- `config.json` 是训练时配置备份；
- `single_station_*.pth` 是 single-station 模型 checkpoint；
- `full_model_*.pth` 是 full model checkpoint；
- `eval_results.txt` 是人可读评估结果；
- `eval_results.npz` 是可进一步分析的评估数组；
- `logs/` 保存 CSV 标量和诊断日志；
- `runs/` 保存 TensorBoard 日志。

## 开发说明

本仓库不是原始 TEAM 代码的简单复现，而是面向当前 PyTorch + DiTing backbone 实验流程的工作代码。

原始 TEAM 的模型思想、论文和 citation 仍然有参考价值，但本项目的实际训练入口、配置方式、评估方式和输出结构以本文档为准。
