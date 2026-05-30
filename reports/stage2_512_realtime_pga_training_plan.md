# Stage2 512 Realtime PGA Training Plan

Date: 2026-05-29

This branch records the realtime PGA task redesign. The model architecture is
kept close to the current best baseline first; the first change is the training
and evaluation sample definition.

## Branch And Code Provenance

Realtime training work is isolated on:

```text
zhangb/realtime-pga-training
```

Probabilistic realtime-output experiments are isolated on:

```text
zhangb/realtime-probabilistic-output
```

The branch was created from the completed temporal-residual diagnostic branch:

```text
branch: zhangb/layerwise-station-temporal
commit: 8d0131db09f9d89e215b27dda77399b481b3a92a
title: Add frozen-base temporal residual diagnostics
```

Historical configs have the following code provenance:

| Experiment family | Configs | Code commit | Branch | Notes |
|---|---|---|---|---|
| Position / VS30 / RoPE baseline | `pos-a` to `pos-f`, `pos-r1` to `pos-r8` | `f86f4a58b14780b66a91507bb1c3e94973140c45` | `zhangb/diting-backbone-attnpool-team` | Last compatible commit before the layerwise station-target branch. |
| Layerwise station / temporal residual diagnostics | `b53` to `b71` | `8d0131db09f9d89e215b27dda77399b481b3a92a` | `zhangb/layerwise-station-temporal` | Contains the b54 baseline, b62-b67 residual diagnostics, and b68-b71 frozen-base checks. |

Realtime configs generated after this point should be reproduced from the exact
commit listed in the registry below.

## Reproduction Registry

Every runnable realtime config must be registered here when it is created.

| Config | Git commit | Branch | Purpose | Notes |
|---|---|---|---|---|
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt1_b54_realtime_bins3_chaosuan.json` | `5609fa245cf6a36809b18454b8ba2f8c3dc299c9` | `zhangb/realtime-pga-training` | First realtime PGA training config | b54 architecture, full-model warm start from `weights_japan_overfit_pga15_stage2_512_b54_event_aux/full_model_best.pth`; train samples 3 random bins per event per epoch; validation sweeps fixed realtime points. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt2_b54_realtime_gaussian_nll_chaosuan.json` | `9cb57b61d489d44888932a0691bd209589d5f68a` | `zhangb/realtime-probabilistic-output` | Single-Gaussian probabilistic baseline | Same realtime sampling and b54 warm start as rt1; mag/loc/PGA are Gaussian heads optimized by weighted NLL. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt3_b54_realtime_gaussian_nll_meanaux02_chaosuan.json` | `9cb57b61d489d44888932a0691bd209589d5f68a` | `zhangb/realtime-probabilistic-output` | Gaussian NLL plus mean regularization | rt2 plus `distribution_mean_loss` with 0.2-weight Huber loss on predictive means. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt4_b54_realtime_pga_mdn3_magloc_gaussian_nll_chaosuan.json` | `9cb57b61d489d44888932a0691bd209589d5f68a` | `zhangb/realtime-probabilistic-output` | PGA-only mixture test | mag/loc use single-Gaussian heads; PGA uses a 3-component MDN. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt5_b54_realtime_all_mdn3_nll_chaosuan.json` | `9cb57b61d489d44888932a0691bd209589d5f68a` | `zhangb/realtime-probabilistic-output` | Full mixture test | mag/loc/PGA all use 3-component MDN heads. |

## Task Definition

The realtime task simulates the information available after an earthquake as
time advances.

For each event and current time `t`:

```text
t = current_time - first_valid_input_station_p_pick
```

The model receives only information available by `current_time`:

- stations with `p_pick <= current_time` can be valid inputs;
- each input station waveform is cut at `current_time` and right-aligned in the
  fixed input window;
- stations whose P wave has not arrived are not valid input stations;
- target stations are still sampled from both input and non-input stations.

The target remains the final station PGA. This means the model learns two
realistic behaviors at the same time:

- update final PGA estimates for already triggered input stations;
- predict final PGA for other stations, including stations that have not
  triggered yet.

## Time Sampling

Training uses random time sampling inside bins instead of fixed discrete times.
The planned bins are:

```text
[0, 1], [1, 3], [3, 5], [5, 10], [10, 20], [20, 40], [40, 90] seconds
```

To avoid spreading 150 epochs too thinly across seven bins, each event should
sample multiple bins per epoch:

```text
bins_per_event_per_epoch = 3
bin_sampling = without_replacement
```

This gives about `150 * 3 / 7 = 64` effective epochs per bin while preserving
random within-bin timing.

Validation uses fixed time points for reproducible time-evolution curves:

```text
1, 3, 5, 10, 20, 40, 90 seconds
```

## Target Sampling

Targets should be a controlled mixture rather than excluding input stations.
The initial target categories are:

- input stations: already triggered and used as model input;
- triggered non-input stations: already triggered but not selected as input;
- untriggered stations: P wave has not arrived at `current_time`.

The first planned ratio is:

```text
input_ratio = 0.3
triggered_noninput_ratio = 0.2
untriggered_ratio = 0.5
```

If one category has too few candidates for an event/time, the sampler should
fill remaining target slots from the other available categories.

## Implementation Notes

The current baseline generator is not sufficient for this task:

- overfit mode currently forces a fixed late cutout at `cutout_end`, which
  removes the realtime time-sweep behavior;
- random input station counts are not equivalent to physical time progression;
- `disable_station_foreshadowing` is currently not an active control path;
- validation needs deterministic expansion over fixed time points.

The first implementation should therefore add explicit realtime sampling to the
data generator, then add configs using the unchanged launch command:

```bash
bash train_light_slurm.sh pga_configs/<config>.json
```

Implemented behavior in `rt1`:

- `PreloadedEventGenerator` expands training samples to
  `bins_per_event_per_epoch=3` realtime samples per event per epoch;
- validation samples are expanded across fixed `1/3/5/10/20/40/90s` current
  times;
- waveform cutout is defined relative to the first valid P pick of the event;
- `trigger_based` is forced on for realtime samples so stations whose P wave
  has not arrived cannot contribute waveform features;
- input station count is determined by current time instead of
  `random_input_station_count`;
- PGA targets are sampled from a controlled input / triggered non-input /
  untriggered mixture, initially `0.3 / 0.2 / 0.5`;
- `p_pick_info` now carries realtime diagnostics:
  `realtime_elapsed_time`, `realtime_time_bin`, `realtime_current_sample`,
  `realtime_target_type`, `realtime_target_lead_time`, and
  `pga_target_indices`;
- `eval_checkpoint.py` saves those fields and prints realtime PGA breakdowns
  by current time, target type, target lead time, input station count, and the
  strong-PGA threshold.

## Required Metrics

Realtime evaluation must report PGA metrics by:

- current time: `1/3/5/10/20/40/90s`;
- target type: input, triggered non-input, untriggered;
- target lead time: `target_p_pick - current_time`;
- input station count;
- strong-PGA bins.

The aggregate validation MAE is secondary. The important question is how the
error and strong-PGA bias evolve as more stations and longer waveforms become
available.

## rt1 Result Summary

Completed result directory:

```text
weights_japan_overfit_pga15_stage2_512_rt1_b54_realtime_bins3
```

Main observations:

- The realtime task definition is working: validation PGA error generally
  improves as current time advances. With the last checkpoint, MAE moves from
  `0.3897` at `t=1s` to `0.2197` at `t=90s`.
- Aggregate validation favors the last checkpoint slightly over the best-loss
  checkpoint: `MAE 0.2694` vs `0.2744`.
- Strong-PGA behavior tells a different story. The last checkpoint has stronger
  high-PGA underprediction: strong-bin `MAE 0.3348`, bias `-0.2666`, compared
  with best checkpoint strong-bin `MAE 0.2872`, bias `-0.2140`.
- Target type remains important: last-checkpoint MAE is `0.2314` for input
  targets, `0.2918` for triggered non-input targets, and `0.3558` for
  untriggered targets.
- Lead time is the hardest operational axis: last-checkpoint MAE is `0.2414`
  for already-arrived targets, `0.3202` for `0-5s` lead, `0.4261` for `5-20s`
  lead, and `0.7454` for `20s+` lead, though the last bucket has only 29
  targets.

Conclusion: rt1 is a usable realtime baseline, but checkpoint/model selection
must include strong-PGA, untriggered-target, and positive-lead metrics. Overall
MAE alone can prefer a checkpoint that is worse for warning-relevant strong
motions.

## Probabilistic Output Experiments

Motivation: realtime PGA prediction should eventually produce warning
probabilities such as `P(PGA >= threshold)`, not only a point estimate.
Therefore rt2-rt5 replace deterministic heads with Gaussian or mixture-density
heads and optimize negative log likelihood.

Implementation rules:

- `output_distribution="gaussian"` uses one Gaussian component per output.
- `output_distribution="mdn"` uses the configured mixture counts.
- PGA NLL is computed in the same normalized space as point-regression training;
  evaluation converts both predictive mean and sigma back to raw PGA units.
- For Gaussian/MDN evaluation, the default point estimate is the predictive
  mixture mean, not the argmax component mean.
- `eval_checkpoint.py` reports predictive sigma coverage and, when a
  `pga_loss_weighting.threshold` is present, Brier-style diagnostics for
  `P(PGA >= threshold)`.
- Transfer from the deterministic b54 checkpoint intentionally skips
  shape-mismatched output-head tensors while reusing compatible backbone,
  adapter, readout, and MLP weights.

Planned interpretation:

- rt2 checks whether pure single-Gaussian NLL is stable and calibrated.
- rt3 checks whether a small Huber loss on the predictive mean prevents NLL from
  improving calibration while degrading MAE or strong-PGA bias.
- rt4 checks whether PGA needs multimodality while mag/loc can remain Gaussian.
- rt5 checks whether making all heads mixtures improves NLL/calibration enough
  to justify the extra flexibility.
