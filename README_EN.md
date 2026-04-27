# TEAM PyTorch DiTing Project

## Overview

This repository is a PyTorch experiment codebase built around the TEAM (Transformer Earthquake Alerting Model) idea and extended with a DiTing-backed waveform encoder.

It trains and evaluates two model stages from seismic waveforms, station metadata, and PGA targets:

- single-station model: a single-station pretraining model for waveform feature learning;
- full model: a multi-station TEAM-style Transformer for magnitude, location, and PGA prediction.

The current workflow focuses on:

- magnitude prediction;
- event location prediction;
- multi-target station PGA prediction;
- single-station pretraining;
- distributed full-model training;
- automatic post-training evaluation of both the full model and the single-station model.

## Repository Layout

Key files and directories:

```text
train_light.py              Main PyTorch training entry point
train_light_slurm.sh        Slurm training and post-training evaluation launcher
eval_checkpoint.py          Checkpoint evaluation script
gemini_models.py            TEAM/DiTing-related model definitions
gemini_util_light.py        Data generation, preprocessing, and batch assembly
loader_light.py             Training data loading
pga_configs/                PGA training configs
diting/                     DiTing backbone code and configs
requirements.txt            Python dependency list
runs/                       TensorBoard output
logs/                       Scalar and diagnostic training logs
<weight_path>/              Checkpoints, config backup, and evaluation outputs
```

## Dependencies

The dependency list is copied from `ditingbench/requirements.txt` into this project:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- xFormers
- NumPy / SciPy / pandas / h5py
- Matplotlib
- ObsPy
- SeisBench
- PyYAML
- tqdm
- DiTing-related dependencies

The actual PyTorch, xFormers, ROCm, or CUDA versions must match the target machine.

## Configuration

Training is controlled by JSON config files. Example configs are under:

```text
pga_configs/
magloc_configs/
```

The current PyTorch workflow mostly uses configs under `pga_configs/`.

Typical top-level structure:

```json
{
  "model_params": {},
  "training_params": {}
}
```

`model_params` controls the model architecture, for example:

- `max_stations`
- `n_pga_targets`
- `use_coords_abs`
- `use_coords_rel`
- `pga_mixture`
- `magnitude_mixture`
- `location_mixture`

`training_params` controls data and optimization, for example:

- `data_path`: training data path;
- `weight_path`: output directory for checkpoints and evaluation results;
- `epochs_full_model`: number of full-model training epochs;
- `single_station_pretrain`: single-station pretraining config;
- `res_comps`: enabled output components, such as `["mag", "loc", "pga"]`;
- `res_weight`: loss weights for output components;
- `full_model_loss`: full-model loss type;
- `generator_params`: data generation and sampling parameters.

`weight_path` must be empty unless the launcher removes it before training.

## Training

### Direct Python Run

Basic command:

```bash
python train_light.py \
  --config pga_configs/your_config.json \
  --diting_config diting/config/your_diting_config.yml \
  --diting_pretrained /path/to/diting_checkpoint.pt
```

Common extra flags:

```bash
--test_run
--overfit_n 16
--no_multiprocessing
--skip_single_station_pretrain
--single_station_only
```

### Slurm Launcher

On a cluster, use:

```bash
bash train_light_slurm.sh <config.json> [train_light.py extra args...]
```

The launcher handles:

- Slurm job submission;
- runtime environment initialization;
- distributed training with `torchrun`;
- optional cleanup of the old `weight_path`;
- automatic evaluation after successful training.

Useful environment variables:

| Variable | Description |
| --- | --- |
| `WORKDIR` | Repository path. |
| `DITING_CONFIG` | DiTing YAML config. |
| `DITING_PRETRAINED` | DiTing pretrained checkpoint. |
| `SLURM_PARTITION` | Slurm partition. |
| `SLURM_NODES` | Number of nodes. |
| `SLURM_GPUS_PER_NODE` | GPUs/DCUs per node and training processes per node. |
| `SLURM_CPUS_PER_TASK` | CPUs per Slurm task. |
| `RESET_WEIGHT_PATH` | Set to `1` to remove `weight_path` before training; set to `0` to keep it. |
| `RUN_EVAL` | Set to `1` to run evaluation after training; set to `0` to skip it. |
| `EVAL_CHECKPOINT` | Manually choose the full-model checkpoint. |
| `EVAL_SINGLE_STATION_CHECKPOINT` | Manually choose the single-station checkpoint. |
| `EVAL_OUTPUT_TXT` | Manually choose the evaluation text output path. |
| `EVAL_OUTPUT_NPZ` | Manually choose the evaluation NPZ output path. |
| `EVAL_DEVICE` | Manually choose the evaluation device, for example `cuda:0`. |

Example:

```bash
RESET_WEIGHT_PATH=1 RUN_EVAL=1 bash train_light_slurm.sh pga_configs/your_config.json
```

If `--overfit_n` is passed to training, the launcher forwards it to evaluation so both stages use the same overfit split.

## Training Flow

The full workflow is:

```text
Read JSON config
Load data
Build data generators
Optionally run single-station pretraining
Build the full model
Load DiTing backbone / pretrained checkpoint
Run DDP multi-device training
Save full_model_*.pth checkpoints
Run automatic evaluation after successful training
```

Full-model checkpoints are saved as:

```text
<weight_path>/full_model_<epoch>.pth
```

Single-station checkpoints are usually saved as:

```text
<weight_path>/single_station_best.pth
<weight_path>/single_station_final.pth
```

## Evaluation

After training, `train_light_slurm.sh` runs:

```bash
python eval_checkpoint.py
```

Evaluation covers:

- full model: latest `full_model_*.pth` under `<weight_path>`;
- single-station model: `single_station_best.pth` first, then `single_station_final.pth`.

If `single_station_pretrain` is enabled in the config but no single-station checkpoint is found, the launcher fails instead of silently skipping single-station evaluation.

Default evaluation outputs:

```text
<weight_path>/eval_results.txt
<weight_path>/eval_results.npz
```

Where:

- `eval_results.txt` contains the full human-readable evaluation log and metrics;
- `eval_results.npz` contains raw arrays saved by `eval_checkpoint.py`.

Manual evaluation command:

```bash
python eval_checkpoint.py \
  --config pga_configs/your_config.json \
  --diting_config diting/config/your_diting_config.yml \
  --diting_pretrained /path/to/diting_checkpoint.pt \
  --checkpoint <weight_path>/full_model_291.pth \
  --single_station_checkpoint <weight_path>/single_station_best.pth \
  --output <weight_path>/eval_results.npz
```

To save the text log:

```bash
python eval_checkpoint.py ... > <weight_path>/eval_results.txt 2>&1
```

## Outputs

A complete training run typically produces:

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

Notes:

- `config.json` is the backed-up training config;
- `single_station_*.pth` are single-station checkpoints;
- `full_model_*.pth` are full-model checkpoints;
- `eval_results.txt` is the human-readable evaluation output;
- `eval_results.npz` stores arrays for further analysis;
- `logs/` stores CSV scalar and diagnostic logs;
- `runs/` stores TensorBoard logs.

## Development Notes

This repository is not a plain reproduction of the upstream TEAM codebase. It is the working codebase for the current PyTorch + DiTing-backbone experiment workflow.

The original TEAM publications and model ideas remain relevant, but the actual training entry points, configs, evaluation flow, and outputs should be understood from this documentation.
