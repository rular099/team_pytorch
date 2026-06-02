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
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt6_b54_realtime_pga_mdn3_meanaux005_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Mean-aux sweep low weight | rt4-style PGA MDN, mag/loc single-Gaussian, `distribution_mean_loss.weight=0.05`. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt7_b54_realtime_pga_mdn3_meanaux010_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Mean-aux anchor | rt4-style PGA MDN, mag/loc single-Gaussian, `distribution_mean_loss.weight=0.10`. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt8_b54_realtime_pga_mdn3_meanaux020_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Mean-aux high weight | rt4-style PGA MDN, mag/loc single-Gaussian, `distribution_mean_loss.weight=0.20`. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt9_b54_realtime_pga_mdn3_mag_mdn3_loc_gaussian_meanaux010_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Magnitude MDN test | PGA and mag use 3-component MDN; loc stays single-Gaussian; mean aux weight 0.10. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt10_b54_realtime_pga_mdn3_mag_gaussian_loc_mdn3_meanaux010_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Location MDN test | PGA and loc use 3-component MDN; mag stays single-Gaussian; mean aux weight 0.10. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt11_b54_realtime_all_mdn3_meanaux010_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Full MDN plus mean aux | mag/loc/PGA all use 3-component MDN; mean aux weight 0.10. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt12_b54_realtime_pga_mdn3_meanaux010_vs30_siteaffine_abs_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | VS30 site-affine under absolute coords | rt7 plus target VS30 output modulation; uses the VS30 HDF5 path. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt13_b54_realtime_pga_mdn3_meanaux010_relcoords_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Relative-coordinate control | rt7 with relative-only coordinate embedding and VS30 off. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt14_b54_realtime_pga_mdn3_meanaux010_vs30_siteaffine_relcoords_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Relative coords plus VS30 | Relative-only coordinate embedding plus target VS30 output modulation. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt15_b54_realtime_pga_mdn3_meanaux010_vs30_siteaffine_relabs001_chaosuan.json` | `43eaa1058f532fb2742b1b6b826da8cc54dc9356` | `zhangb/realtime-probabilistic-output` | Weak absolute coordinate hybrid plus VS30 | Relative/absolute fusion with `coords_abs_weight=0.01` plus target VS30 output modulation. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt16_rt10anchor_ditingmae_no_full_transfer_chaosuan.json` | `pending` | `zhangb/realtime-probabilistic-output` | No full-model transfer | rt10 heads/loss/absolute coordinates, but no b54 transfer; frozen DiTing MAE encoder and train TEAM-side adapter/readouts from scratch. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt17_rt10anchor_transfer_except_station_adapter_chaosuan.json` | `pending` | `zhangb/realtime-probabilistic-output` | Transfer except station adapter | rt10 heads/loss/absolute coordinates, transfer b54 except `waveform_model.1.*`, and reinitialize the station adapter. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt18_rt10anchor_adapter_poolquery_coords_chaosuan.json` | `pending` | `zhangb/realtime-probabilistic-output` | Metadata-aware adapter pooling | rt17 plus station coordinate embedding injected into adapter pooling queries before station embedding pooling. |
| `pga_configs/transformer_japan_overfit_pga15_stage2_512_rt19_rt10anchor_adapter_featurefilm_coords_chaosuan.json` | `pending` | `zhangb/realtime-probabilistic-output` | Metadata-aware adapter FiLM | rt17 plus station coordinate embedding used as FiLM modulation on DiTing encoder features before station embedding pooling. |

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

For probabilistic and multi-head experiments, evaluation must also report
train and validation metrics for magnitude and location, not only PGA:

- magnitude: MAE, RMSE, bias, correlation, R2, slope/intercept, and predictive
  sigma coverage when the head is Gaussian/MDN;
- location: per-dimension MAE/RMSE/bias/correlation/R2/slope plus vector-error
  norm, in both the training target coordinate view and the absolute-coordinate
  view when available;
- PGA: the realtime breakdown above plus aggregate calibration and strong-bin
  probability diagnostics.

The aggregate validation PGA MAE is secondary. For this overfit phase, model
selection must consider train fit, validation behavior, strong-PGA bias,
untriggered-target behavior, and whether mag/loc degrade when their heads are
changed.

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

Completed rt2-rt5 overfit interpretation:

- rt4 had the best last-checkpoint train PGA fit among rt2-rt5, while rt2 had
  the best aggregate validation PGA MAE. Because this is an overfit-capacity
  phase, rt4 is the better structural anchor than rt2.
- rt3's mean auxiliary loss improved strong-PGA train bias but hurt validation
  MAE at weight 0.20, so the next sweep should test smaller mean-aux weights.
- rt5 did not clearly justify making all heads MDN at once. Mag and loc MDN
  should be tested separately and evaluated directly on mag/loc metrics.

## rt6-rt15 Combination Search

The next round is intentionally not a full factorial matrix. It uses the rt4
PGA-MDN anchor, then tests only factors that can plausibly improve the overfit
capacity or generalization story.

| Config | mean aux | mag head | loc head | coords | VS30 path | Purpose |
|---|---:|---|---|---|---|---|
| rt6 | 0.05 | Gaussian | Gaussian | absolute | off | Low mean-aux weight. |
| rt7 | 0.10 | Gaussian | Gaussian | absolute | off | Main mean-aux anchor. |
| rt8 | 0.20 | Gaussian | Gaussian | absolute | off | High mean-aux stress test. |
| rt9 | 0.10 | MDN3 | Gaussian | absolute | off | Is mag multimodality useful by itself? |
| rt10 | 0.10 | Gaussian | MDN3 | absolute | off | Is loc multimodality useful by itself? |
| rt11 | 0.10 | MDN3 | MDN3 | absolute | off | Combined mag/loc MDN after single-factor checks. |
| rt12 | 0.10 | Gaussian | Gaussian | absolute | site-affine | Tests whether the new VS30 structure helps even with absolute coordinates. |
| rt13 | 0.10 | Gaussian | Gaussian | relative only | off | Coordinate-generalization control without VS30. |
| rt14 | 0.10 | Gaussian | Gaussian | relative only | site-affine | Tests the intended transferable setting: relative coords plus VS30. |
| rt15 | 0.10 | Gaussian | Gaussian | relative + 0.01 absolute | site-affine | Hybrid setting that keeps weak regional/path cues while forcing VS30 usage. |

The VS30 configs use the same mirrored HDF5 path as the earlier VS30
position experiments:

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/japan_2024.hdf5
```

The initial rt12/rt14/rt15 configs pointed to the strict rebuild path
`/public/home/zhangbei/work_dir/zhangbei/japan_knet_converted/origin_corrected_diting_vel_acc_vs30/2024/japan_2024.hdf5`;
that path was not present on the cluster run and caused `FileNotFoundError`
before training started. The configs were corrected after that failed attempt.

### VS30 Site-Affine Head

The previous VS30 design injected station/target VS30 as additive embeddings
behind scalar gates initialized at zero. Those gates stayed near zero in the
position experiments, so the branch did not prove useful.

New configs set `vs30_injection_mode="pga_site_affine"`. This mode leaves the
station/query token path unchanged and applies target-site modulation directly
to the PGA output:

```text
mu_final = (1 + 0.1 * tanh(scale_raw)) * mu_base + bias
```

The affine parameters are predicted from `[pga_readout_embedding,
log(vs30/760), valid, base_mean]`. The final affine layer is zero-initialized,
so the starting model is exactly the base model, but VS30 receives a direct
gradient from the PGA loss. This can represent additive residuals, multiplicative
site amplification, or a learned combination of both.

For MDN PGA heads, the same affine transform is applied to every component mean.
Sigma modulation is implemented behind `vs30_site_affine_sigma_shift` but kept
off in rt12/rt14/rt15 to isolate mean effects first.

### How To Read rt6-rt15

This round is a decision tree, not a leaderboard. The purpose is to decide
which factors deserve to be combined in the next compact round.

First, choose the mean-auxiliary weight from rt6/rt7/rt8:

- compare train and validation PGA mean metrics, strong-PGA bias, untriggered
  target metrics, and probability calibration;
- prefer the smallest mean-aux weight that improves train strong-PGA bias or
  mean prediction without increasing validation error too much;
- if rt8 repeats rt3's pattern of better strong-bin train bias but worse
  validation, keep rt7 or rt6 as the combination anchor.

Second, evaluate mag/loc MDN from rt9/rt10/rt11:

- rt9 is accepted only if magnitude metrics improve directly versus rt7
  without causing a meaningful PGA regression;
- rt10 is accepted only if location metrics improve directly versus rt7,
  especially absolute-coordinate vector error, without causing a meaningful PGA
  regression;
- rt11 is useful only if both single-factor tests help or if the combined model
  shows a clear interaction that is visible in mag/loc metrics, not just PGA;
- if MDN improves only sigma/NLL-style calibration while worsening predictive
  mean MAE/bias, do not adopt it for the current overfit-capacity objective.

Third, interpret VS30 and coordinate experiments:

- rt12 versus rt7 tests the new VS30 site-affine architecture under absolute
  coordinates. A small effect is expected because absolute coordinates can
  already encode site and path information.
- rt13 versus rt7 measures the cost of removing absolute coordinates. If rt13
  collapses on train fit, relative-only geometry is not enough for the current
  overfit target.
- rt14 versus rt13 tests the intended transferable setting: relative coordinates
  plus VS30. This is the key comparison for whether VS30 adds recoverable site
  information when absolute coordinates are removed.
- rt15 versus rt14 tests whether a weak absolute-coordinate channel
  (`coords_abs_weight=0.01`) recovers regional/path information without letting
  absolute coordinates dominate the VS30 branch.

For VS30 configs, also inspect training diagnostics:

- `data/vs30_target_valid_ratio` and `data/vs30_station_valid_ratio` must be
  high enough that a negative result is meaningful;
- `vs30_site_scale_delta_mean`, `vs30_site_scale_delta_std`, and
  `vs30_site_bias_abs_mean` indicate whether the site-affine branch is being
  used;
- if rt14/rt15 do not improve over their non-VS30 coordinate controls and the
  site-affine diagnostics stay near zero, stop VS30 experiments for this Japan
  overfit round.

### rt16-rt19 Station-Adapter Transfer and Metadata-Injection Ablation

The single-station pretrain concern is different from the old temporal-token
memory experiments. The question is whether a station adapter trained through
single-station/event-level targets compresses every station toward the same
event representation and therefore limits multi-station PGA use.

Use rt10 as the anchor here only in the sense of heads, loss, realtime sampling,
and absolute-coordinate setup. Do not interpret `rt10anchor` as loading an rt10
checkpoint. The direct comparison is:

| Config | What Changes From rt10 | What It Tests |
|---|---|---|
| rt16 | Removes `transfer_model_path`; TEAM-side adapter/readouts start from scratch. | Whether DiTing MAE features are sufficient without inheriting b54 single-station/full-model training. |
| rt17 | Keeps b54 transfer but excludes `waveform_model.1.*` and reinitializes the adapter. | Whether the station adapter specifically is the limiting inherited component, while preserving other b54 initialization. |
| rt18 | Same as rt17, but station coordinate embedding biases the adapter attention-pooling queries. | Whether location should guide how DiTing tokens are pooled into station embeddings. |
| rt19 | Same as rt17, but station coordinate embedding applies FiLM modulation to DiTing encoder features before pooling. | Whether location should modulate the encoder feature channels before the adapter pools them. |

These configs should have much lower memory risk than the old b58-b60
temporal-token path because they keep the same pooled station-embedding path as
rt10. If either improves train PGA fit or station-count/target-type breakdowns,
then revisit a compressed temporal-token branch later; do not jump directly to
the full raw-token scheme.

How to analyze rt16-rt19:

1. First compare rt16 and rt17 against rt10. If both are clearly worse on train
   PGA fit, validation PGA MAE, and location metrics, then the old b54 transfer
   is probably still useful and the single-station compression concern is not
   the dominant bottleneck.
2. Compare rt17 against rt10 specifically to isolate the station adapter. rt17
   keeps the non-adapter b54 initialization but removes `waveform_model.1.*`.
   If rt17 improves train fit, station-count breakdowns, or input/triggered
   target PGA without hurting validation, the inherited adapter is suspect.
3. Compare rt18 and rt19 against rt17, not directly against rt10. This isolates
   the value of early coordinate injection after the inherited adapter has
   already been removed.
4. For rt18/rt19, require evidence on both PGA and event heads. PGA conclusions
   need train/val PGA MAE, strong-PGA bias, untriggered MAE, Brier, and
   current-time breakdowns. Location conclusions need loc vector error and loc
   NLL/MDN behavior. A lower PGA MAE with much worse loc is not a clean win.
5. Inspect station diagnostics when available: `station_adapter_raw_norm`,
   `wave_emb_norm`, `station_emb_norm`, and `station_emb_cosine_mean`. A useful
   adapter change should not simply collapse station embeddings to a narrower
   representation.

Follow-up design after rt16-rt19:

- If rt17 beats rt10, keep transfer-except-adapter as the default warm start for
  future rt configs.
- If rt18 beats rt17, prefer metadata-aware pooling query as the low-risk early
  coordinate-injection design.
- If rt19 beats rt17 and rt18, prefer feature-FiLM, but rerun once with a lower
  `diting_station_metadata_scale` such as `0.03` to confirm the gain is not from
  an overly strong coordinate shortcut.
- If rt18/rt19 both lose to rt17, keep position injection after the station
  adapter and do not add more adapter metadata variants in this round.
- Only if rt17/rt18/rt19 show a clear benefit should the next round revisit
  compressed temporal-token branches. Keep that branch memory-bounded by
  projecting tokens early and avoiding target-by-time expansion.

### Second-Round Branching Rules

After rt6-rt15 complete, run at most four to six follow-up configs. Do not make
a full matrix.

Use these rules:

- Mean aux: take the best of rt6/rt7/rt8 as the default weight. If none improves
  rt4 on train fit and strong-PGA bias, disable mean aux in the second round.
- Mag head: include mag MDN only if rt9 improves magnitude metrics versus rt7
  and does not hurt PGA. Otherwise keep magnitude single-Gaussian.
- Loc head: include loc MDN only if rt10 improves location metrics versus rt7
  and does not hurt PGA. Otherwise keep location single-Gaussian.
- Coord/VS30: for best Japan-only overfit performance, absolute coordinates may
  remain the best choice. For transfer-oriented performance, prefer rt14 or rt15
  only if they recover a substantial fraction of rt7/rt12 performance while
  improving the relative-coordinate control.
- VS30 sigma shift: test `vs30_site_affine_sigma_shift=true` only if rt12,
  rt14, or rt15 shows a mean-prediction benefit and the remaining weakness is
  calibration/coverage.

Recommended second-round templates:

| Template | Purpose |
|---|---|
| `best_abs` | Best mean-aux + accepted mag/loc heads + absolute coordinates + VS30 off. This is the Japan-overfit anchor. |
| `best_abs_vs30` | Same heads/mean-aux as `best_abs`, but absolute coordinates plus VS30 site-affine, only if rt12 helps. |
| `best_rel_vs30` | Same heads/mean-aux, relative coordinates plus VS30 site-affine, only if rt14 beats rt13. |
| `best_relabs001_vs30` | Same heads/mean-aux, relative + weak absolute coordinates plus VS30, only if rt15 beats rt14 or is close to absolute-coordinate performance. |
| `best_vs30_sigma` | Add `vs30_site_affine_sigma_shift=true`, only after a VS30 mean benefit is established. |
| `best_no_meanaux` | Optional sanity check if the selected MDN/VS30 combination already fixes mean bias and mean aux becomes unnecessary. |

The next handoff should summarize each completed config with both train and
validation rows. A config is not considered better unless the conclusion is
consistent across the relevant task head: PGA conclusions require PGA metrics,
mag-head conclusions require magnitude metrics, and loc-head conclusions require
location metrics.
