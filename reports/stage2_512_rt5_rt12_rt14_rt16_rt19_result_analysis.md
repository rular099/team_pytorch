# Stage2 512 rt5/rt12/rt14/rt16-rt19 Result Analysis

Source: `../chaosuan_res` completed runs. The primary comparison below uses the
last checkpoint, because this series is an overfit/stress test and the `best`
checkpoints are not consistently best for PGA point MAE. For example, rt10 has
`val_pga_mae=0.2667` at last but `0.3009` at best.

rt16 was evaluated as 8 shards. The rt16 metrics below are recomputed by
merging `eval_results_last_shard000of008.npz` through
`eval_results_last_shard007of008.npz`, so they are directly comparable with the
single-file eval results from the other runs.

## Last-Checkpoint Metrics

| run | purpose | train PGA MAE | val PGA MAE | val untriggered MAE | val strong MAE | val strong bias | val Brier | val mag MAE | val loc vector |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rt5 | all MDN3, no mean aux | 0.1512 | 0.2851 | 0.3769 | 0.2856 | -0.2152 | 0.1529 | 0.3655 | 15.2842 |
| rt7 | PGA MDN3 + mean aux 0.10 anchor | 0.1250 | 0.2703 | 0.3547 | 0.2962 | -0.2270 | 0.1451 | 0.3878 | 15.5780 |
| rt10 | rt7 + loc MDN3 | 0.1296 | 0.2667 | 0.3495 | 0.3052 | -0.2400 | 0.1431 | 0.3596 | 15.2190 |
| rt12 | rt7 + VS30 site-affine, absolute coords | 0.1567 | 0.2660 | 0.3160 | 0.3758 | -0.3470 | 0.1334 | 0.3685 | 15.5104 |
| rt13 | rt7, relative coords only | 0.1942 | 0.2891 | 0.4383 | 0.3407 | -0.2908 | 0.1304 | 0.3493 | 19.1884 |
| rt14 | rt13 + VS30 site-affine | 0.1681 | 0.2782 | 0.4325 | 0.3095 | -0.2378 | 0.1268 | 0.3599 | 19.3815 |
| rt16 | rt10 anchor, no full-model transfer | 0.1427 | 0.2979 | 0.3229 | 0.3651 | -0.2792 | 0.1663 | 0.3711 | 14.0274 |
| rt17 | rt10 anchor, transfer except station adapter | 0.1481 | 0.2802 | 0.3722 | 0.2825 | -0.1822 | 0.1531 | 0.3666 | 14.6636 |
| rt18 | rt17 + adapter pool-query coords | 0.1464 | 0.2750 | 0.3373 | 0.3592 | -0.3071 | 0.1374 | 0.4000 | 16.1056 |
| rt19 | rt17 + adapter feature-FiLM coords | 0.1412 | 0.2799 | 0.3785 | 0.3038 | -0.1876 | 0.1565 | 0.3519 | 15.8956 |

## Conclusions

1. rt5 does not justify all-MDN heads.
   Compared with rt10, rt5 is worse on train PGA (+0.0216), val PGA (+0.0184),
   untriggered PGA (+0.0274), Brier (+0.0098), mag (+0.0059), and loc vector
   (+0.0652). It only improves strong-PGA MAE and bias slightly. Keep the prior
   decision: loc MDN is useful; mag/all-MDN is not the default.

2. VS30 under absolute coordinates is useful but not a clean win.
   rt12 versus rt7 isolates the VS30 site-affine branch. It improves val PGA
   (0.2660 vs 0.2703), untriggered PGA (0.3160 vs 0.3547), Brier
   (0.1334 vs 0.1451), mag (0.3685 vs 0.3878), and loc vector
   (15.5104 vs 15.5780). However, train PGA is much worse (0.1567 vs 0.1250)
   and strong-PGA bias becomes substantially more negative (-0.3470 vs -0.2270).
   Against rt10, rt12 is only a tiny val-PGA improvement (0.2660 vs 0.2667) and
   has worse train fit, strong-PGA behavior, mag, and loc. Use rt12 only if the
   priority is untriggered PGA/calibration, not as the balanced default.

3. Relative coordinates plus VS30 recover part of the PGA loss but not enough.
   rt14 improves over rt13 on train PGA (0.1681 vs 0.1942), val PGA
   (0.2782 vs 0.2891), strong-PGA MAE/bias, and Brier. The improvement is real,
   and VS30 diagnostics are active, but untriggered PGA barely improves and loc
   vector remains poor (19.3815 vs 15.2-ish for absolute-coordinate anchors).
   For Japan-only overfit, absolute coordinates remain better. rt14 is only a
   transfer-oriented branch.

4. Skipping full-model transfer does not beat rt10, but it changes the failure
   mode. rt16 has the best val loc vector in this group (14.0274), and it also
   improves untriggered PGA versus rt10 (0.3229 vs 0.3495). However, it is much
   worse on balanced PGA: val PGA 0.2979 versus rt10 0.2667, Brier 0.1663
   versus 0.1431, strong-PGA MAE 0.3651 versus 0.3052, and strong bias -0.2792
   versus -0.2400. The best checkpoint is even worse (`val_pga_mae=0.3434`),
   so this is not a checkpoint-selection artifact.

5. Reinitializing or metadata-conditioning the station adapter also does not
   beat rt10. rt17 improves loc vector and strong-PGA bias versus rt10, but it
   worsens train PGA, val PGA, untriggered PGA, Brier, and mag. rt18 improves
   rt17 on val PGA, untriggered PGA, and Brier, but at the cost of much worse
   strong-PGA bias, mag, and loc. rt19 improves train PGA and mag versus rt17,
   but does not improve the balanced PGA/loc objective. The rt18/rt19 early
   coordinate-injection ideas are therefore not accepted yet.

6. The station-feature collapse concern is supported by the eval diagnostics,
   and rt16-rt19 do not fix it enough to help PGA. The motivating hypothesis for
   rt16-rt19 was that supervised single-station mag/loc training may make
   different station features collapse toward the same event-level
   representation. rt16 is the only run that removes the full supervised
   transfer and its raw-feature cosine drops from rt10's 0.9973 to 0.9967, but
   the features are still nearly collinear. rt17/rt18/rt19 are higher
   (0.9982/0.9981/0.9982). The lower training CSV values
   (`diag_station_emb_cosine_mean`, about 0.73) are later transformer-input
   embeddings after coordinate/scale/metadata fusion, not the same
   representation as the eval diagnostic. They should not be used alone to
   claim that raw station feature collapse has been relieved.

7. Low-station-count evidence is mixed. rt16 improves the `n=2-3` val PGA bucket
   slightly versus rt10 (0.3705 vs 0.3830), but it is worse for `n=1`, `n=4-5`,
   `n=6-10`, `n=11-15`, and `n=16+`. Therefore rt16 does not yet demonstrate
   better multi-station information use; it shows a narrow low-count gain plus
   broad PGA degradation.

## Diagnostics

VS30 branches are not dead. Last-10 mean diagnostics:

| run | target VS30 valid ratio | station VS30 valid ratio | site scale delta mean | site scale delta std | site bias abs mean |
|---|---:|---:|---:|---:|---:|
| rt12 | 0.7448 | 0.4523 | -0.0081 | 0.0134 | 0.0480 |
| rt14 | 0.7448 | 0.4523 | -0.0086 | 0.0106 | 0.1021 |

rt14 uses a larger site bias than rt12, consistent with relative coordinates
forcing VS30 to carry more site information. The effect is still insufficient
to replace absolute coordinates.

Station-feature similarity diagnostics use two different measurement points:

| run | transfer setup | eval raw-feature cosine mean | train CSV transformer-input cosine, last-10 mean | train epoch loss, last-10 mean | val epoch loss, last-10 mean |
|---|---|---:|---:|---:|---:|
| rt10 | original b54 transfer | 0.9973 | 0.7354 | 0.0998 | 2.3251 |
| rt16 | no full-model transfer | 0.9967 | 0.7280 | 0.2610 | 2.3790 |
| rt17 | transfer except station adapter | 0.9982 | 0.7323 | 0.3252 | 2.0242 |
| rt18 | rt17 + pool-query coords | 0.9981 | 0.7329 | 0.2801 | 2.1781 |
| rt19 | rt17 + feature-FiLM coords | 0.9982 | 0.7302 | 0.2426 | 2.1761 |

The eval raw-feature cosine is the more direct evidence for the user's concern:
different station features for the same event are nearly collinear before the
multi-station transformer. rt16 slightly lowers this cosine, but the reduction
is tiny and does not translate into better PGA. The CSV cosine is still useful,
but it measures a post-fusion embedding that already includes coordinate/scale
effects. A better probe would compute per-event station-embedding variance and
train lightweight probes for station coordinates, distance/azimuth, VS30, and
PGA residuals from frozen station embeddings.

Val PGA MAE by input-station count:

| run | n=1 | n=2-3 | n=4-5 | n=6-10 | n=11-15 | n=16+ |
|---|---:|---:|---:|---:|---:|---:|
| rt10 | 0.3466 | 0.3830 | 0.2606 | 0.2789 | 0.2258 | 0.2560 |
| rt16 | 0.3938 | 0.3705 | 0.3190 | 0.3073 | 0.2398 | 0.2942 |
| rt17 | 0.3840 | 0.4210 | 0.2670 | 0.2990 | 0.2460 | 0.2630 |
| rt18 | 0.3450 | 0.3990 | 0.2560 | 0.2900 | 0.2340 | 0.2640 |
| rt19 | 0.3820 | 0.4320 | 0.2610 | 0.2930 | 0.2440 | 0.2640 |

rt16 is only better than rt10 in the `n=2-3` bucket. That is an interesting
low-count signal, but it is not enough to claim better multi-station use because
all other station-count buckets are worse.

## Recommended Next Step

The current balanced Japan-overfit anchor remains rt10: PGA MDN3, mag Gaussian,
loc MDN3, mean-aux 0.10, absolute coordinates, VS30 off.

If the next objective prioritizes untriggered targets or probability
calibration, run one focused follow-up that combines rt10 with the rt12 VS30
site-affine branch. Do not launch a broad VS30 matrix yet, because rt12's gain
is narrow and its strong-PGA degradation is large.

Do not continue the rt16-style "no full-model transfer" branch as the default.
It is useful evidence that removing supervised transfer can improve loc and
slightly reduce raw-feature cosine, but it hurts the core PGA objective. The
next representation-focused step should keep rt10 as the anchor and explicitly
preserve station/path/site residual information.

The follow-up configs rt20-rt25 implement that plan:

| run | intervention | intended question |
|---|---|---|
| rt20 | centered decorrelation on raw station adapter embeddings | Can a small anti-collapse regularizer reduce raw feature collinearity without hurting PGA? |
| rt21 | event-centered station residual branch added before coordinate fusion | Does an explicit station-specific path improve PGA readout? |
| rt22 | station-local PGA residual auxiliary loss | Do direct station-level gradients reduce collapse even if the main token path is unchanged? |
| rt23 | residual branch + station-local PGA aux | Main candidate for weak station-specific-gradient failure. |
| rt24 | DiTing adapter pool queries increased from 4 to 8 | Is collapse partly an adapter pooling bottleneck? |
| rt25 | poolq8 + residual branch + station-local PGA aux | Highest-capacity candidate; compare mainly against rt23. |

When analyzing these runs, keep the rt10 anchor row visible and report both the
new collapse diagnostics and ordinary task metrics. The key new diagnostics are
`raw_station_emb_cosine_mean`, `wave_station_emb_cosine_mean`,
`station_residual_emb_cosine_mean`, and `station_residual_norm_ratio`. A run is
not accepted merely because cosine decreases; it must also preserve or improve
train PGA fit, validation PGA MAE, strong-PGA bias/Brier, mag, and loc.

Operational note: the original rt16 failure log was consistent with a
distributed launcher teardown problem, not a model/batch failure:

- the job ran from 2026-06-02 15:55 to 2026-06-03 08:39;
- most ranks printed `NCCL watchdog thread terminated normally`;
- ranks 1-15 terminated around 08:33:55, while rank 0 terminated around
  08:39:01, roughly 306 seconds later;
- the visible failure is `Error waiting on exit barrier ... RuntimeError:
  Socket Timeout`, followed by a segmentation fault in
  `torch.distributed.elastic.rendezvous`;
- because `train_light_slurm.sh` uses `set -e`, the failed torchrun return
  prevents the post-training eval block from running.

This likely happened because non-zero ranks exited while rank 0 was still doing
post-training sanity/export work, exceeding torch elastic's 300-second exit
barrier. The code has been updated to run a final distributed barrier after
rank 0 cleanup so all workers exit together. The Slurm launcher has also been
updated to default to `DISTRIBUTED_LAUNCHER=slurm_direct`, which starts one
Python process per GPU through Slurm and sets `RANK/WORLD_SIZE/LOCAL_RANK`
directly. This avoids torch elastic rendezvous during startup. If torchrun must
be used for a diagnostic run, set `DISTRIBUTED_LAUNCHER=torchrun`; that path now
uses `SLURM_PROCID` for `--node_rank` instead of relying on `SLURM_NODEID`.
