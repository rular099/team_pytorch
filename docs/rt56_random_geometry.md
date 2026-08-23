# rt56 causal random-geometry experiment

## Purpose

rt56 tests whether the rt55 epoch-32 model can predict PGA at arbitrary
non-input stations when the available waveform stations do not follow the
usual earliest-P-arrival ordering.  It is a robustness experiment, not a
replacement for the frozen rt55 formal-test protocol.

The model architecture is unchanged.  An rt55 checkpoint can still be loaded
by the original rt55 config and evaluation scripts.

## Sampling protocol

At every realtime cutoff, the generator first determines all stations that:

1. have finite coordinates;
2. have a valid P pick at or before the current cutoff; and
3. contain nonzero waveform data before that cutoff.

The random-geometry branch draws a maximum input count from `1,3,5,8,12,16`,
samples up to that many stations from the complete causal set before the
model's 25-station truncation, and zeros every unselected waveform and
sample-valid mask.  It reserves at least one finite-PGA station outside the
input set; when every eligible station has already triggered, the realized
input count can therefore be one lower than the requested count.

Training uses a fixed mixture:

- 50% original rt55 input and target sampling;
- 50% causal random inputs, with PGA targets restricted to non-input stations.

Validation always uses causal random inputs and non-input PGA targets.  The
same deterministic sampling seed is used for epoch-32 zero-shot validation and
fine-tuned-model validation, enabling paired comparison.

## Fine-tuning semantics

The new config uses `training_params.load_model_path` to initialize model
weights from the pinned epoch-32 checkpoint.  It does not resume the rt55
optimizer, scheduler, epoch number, or best-loss state.  The DiTing encoder
remains frozen, the station adapter is not reinitialized, and rt55's older
transfer checkpoint is explicitly disabled.

Defaults:

- learning rate: `1e-4` for adapter and TEAM groups;
- epoch target: 6 new fine-tuning epochs;
- validation-monitored ReduceLROnPlateau;
- new independent weight directory.

## Submission

From the supercomputer repository root:

```bash
DRY_RUN=1 ACTION=all bash tools/run_rt56_random_geometry_slurm.sh

CONFIRM_RT56=1 ACTION=all bash tools/run_rt56_random_geometry_slurm.sh
```

The first command validates paths, annual shards, output guards, active-job
guards, and prints both submissions without calling `sbatch`.  The second
submits zero-shot validation and mixed-random fine-tuning as independent jobs.

Submit only one branch with `ACTION=zero_shot` or `ACTION=finetune`.

Important environment overrides:

```bash
RT55_EP32_CHECKPOINT=/absolute/path/full_model_best_ep32.pth
RT56_WEIGHT_PATH=weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42
EPOCHS_FULL_MODEL=6
```

## Output locations

Epoch-32 zero-shot validation:

```text
logs/weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42/
  zero_shot_ep32/eval_validation_ep32_zero_shot_random_mask.txt
  zero_shot_ep32/eval_validation_ep32_zero_shot_random_mask.npz
  zero_shot_ep32/eval_validation_ep32_zero_shot_random_mask.metrics.json
```

Fine-tuned checkpoints:

```text
weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42/
  config.json
  full_model_init.pth
  full_model_best.pth
  full_model_last.pth
```

`full_model_init.pth` is the weight-only epoch-32 initialization saved in the
new run directory with fine-tuning epoch zero; it does not overwrite or rename
the source rt55 checkpoint.

## Evaluation constraints

- Select the fine-tuned checkpoint using validation only.
- Do not modify the running rt55 epoch-20/epoch-32 formal-test jobs.
- Do not use rt55 test results to change rt56 masking, target ratios, learning
  rate, epoch target, or checkpoint selection.
- Evaluate retention later by loading the selected rt56 checkpoint with the
  original resolved rt55 config.
- Treat epoch 20 as a sensitivity reference, not an alternative selected model.
