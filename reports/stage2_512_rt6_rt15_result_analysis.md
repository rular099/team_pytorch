# Stage2 512 rt6-rt15 Partial Result Analysis

Date: 2026-06-01

This note records the first rt6-rt15 result pass from
`/home/zhangb/work/people/zhangbei/team_claude/chaosuan_res`.

## Run Status

Completed:

- rt6, rt7, rt8: mean-auxiliary weight sweep.
- rt9, rt10, rt11: mag/loc MDN variants.
- rt13: relative-coordinate control.

Failed before training:

- rt12, rt14, rt15: all VS30 site-affine configs.
- Root cause: `FileNotFoundError` opening
  `/public/home/zhangbei/work_dir/zhangbei/japan_knet_converted/origin_corrected_diting_vel_acc_vs30/2024/japan_2024.hdf5`.
- The xFormers, OMP, NCCL, and Slurm route messages in the logs are warnings or
  downstream distributed-launch noise. The first actionable error is the missing
  HDF5 file.
- The VS30 config `data_path` values have been corrected to the mirrored path
  used by previous VS30 runs:
  `/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/japan_2024.hdf5`.

Second failure after the data-path fix:

- One of rt12/rt14/rt15 started running, while two jobs failed at the first
  backward pass.
- Root cause: the original VS30 site-affine implementation modified
  `output_pga[..., 1]` and optionally `output_pga[..., 2]` in-place after
  cloning. For MDN PGA this slice has shape `[batch, target, component]`, which
  matches the error tensor `[32, 15, 3]`.
- The fix is to reconstruct the output without in-place slice assignment:
  `torch.stack([alpha_logits, component_mu_final, component_sigma_final],
  dim=-1)`.
- Re-run the failed VS30 jobs after the non-inplace site-affine fix. If the
  still-running job was launched before this fix, treat it cautiously; if it
  finishes, keep the result, but if it fails with the same autograd message it
  should be restarted from the fixed commit.

## Main Last-Checkpoint Metrics

Because this is still an overfit-capacity phase, the last checkpoint is more
informative than the best validation-loss checkpoint. The best-loss checkpoints
usually underfit train PGA.

| Config | mean aux | mag mix | loc mix | coords | train PGA MAE | train PGA R2 | train strong bias | val PGA MAE | val strong bias | val untriggered MAE | val Brier | val mag MAE | val loc vector | val depth MAE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rt4 | 0 | 1 | 1 | abs | 0.1230 | 0.8524 | -0.1320 | 0.2699 | -0.2599 | 0.3443 | 0.1397 | 0.3891 | 16.3489 | 16.2764 |
| rt5 | 0 | 3 | 3 | abs | 0.1331 | 0.8424 | -0.0890 | 0.2730 | -0.2279 | 0.3565 | 0.1346 | 0.3641 | 16.3335 | 16.2602 |
| rt6 | 0.05 | 1 | 1 | abs | 0.1325 | 0.8418 | -0.1094 | 0.2666 | -0.2617 | 0.3561 | 0.1453 | 0.3455 | 16.9431 | 16.8747 |
| rt7 | 0.10 | 1 | 1 | abs | 0.1250 | 0.8464 | -0.0791 | 0.2703 | -0.2270 | 0.3547 | 0.1451 | 0.3878 | 15.5780 | 15.5059 |
| rt8 | 0.20 | 1 | 1 | abs | 0.1343 | 0.8367 | -0.1079 | 0.2724 | -0.2537 | 0.3552 | 0.1457 | 0.3530 | 15.7678 | 15.6955 |
| rt9 | 0.10 | 3 | 1 | abs | 0.1374 | 0.8213 | -0.0057 | 0.2738 | -0.2026 | 0.3757 | 0.1624 | 0.3798 | 15.8793 | 15.8135 |
| rt10 | 0.10 | 1 | 3 | abs | 0.1296 | 0.8481 | -0.0946 | 0.2667 | -0.2400 | 0.3495 | 0.1431 | 0.3596 | 15.2190 | 15.1361 |
| rt11 | 0.10 | 3 | 3 | abs | 0.1274 | 0.8439 | -0.0546 | 0.2740 | -0.2070 | 0.3669 | 0.1505 | 0.3660 | 15.1399 | 15.0379 |
| rt13 | 0.10 | 1 | 1 | rel | 0.1942 | 0.7227 | -0.2142 | 0.2891 | -0.2908 | 0.4383 | 0.1304 | 0.3493 | 19.1884 | 19.1512 |

## Interpretation

Mean auxiliary loss:

- rt6 gives the best validation PGA MAE among completed configs (`0.2666`) and
  the best validation magnitude MAE (`0.3455`), but its train PGA fit is weaker
  than rt4/rt7.
- rt7 is the best mean-aux setting for train PGA fit and strong-PGA train bias
  among the Gaussian mag/loc variants.
- rt8 does not improve the balance. A 0.20 mean-aux weight again looks too high
  for this branch.
- Practical choice: keep `0.05` if validation PGA/mag is prioritized; keep
  `0.10` if the overfit-capacity and strong-PGA train-bias objective is
  prioritized.

Mag MDN:

- rt9 improves train magnitude MAE (`0.1879`) and nearly removes train strong-PGA
  bias (`-0.0057`), and it also has the least negative validation strong-PGA
  bias among completed runs (`-0.2026`).
- The cost is clear: validation PGA MAE worsens to `0.2738`, untriggered-target
  MAE worsens to `0.3757`, and Brier worsens to `0.1624`.
- Do not adopt mag MDN as the default unless strong-PGA bias becomes the primary
  objective.

Loc MDN:

- rt10 is the strongest completed absolute-coordinate candidate. It matches
  rt6 on validation PGA MAE (`0.2667`), improves validation untriggered MAE
  (`0.3495`), and improves validation location vector error (`15.2190`).
- It does not beat rt7/rt11 on strong-PGA bias, but it is the best overall
  balance among completed rt6-rt11 runs.
- rt11 improves validation location slightly further (`15.1399`) and improves
  strong-PGA bias, but hurts validation PGA MAE (`0.2740`) and untriggered MAE
  (`0.3669`).
- Practical choice: include loc MDN in the next absolute-coordinate anchor;
  do not include mag MDN unless the next objective explicitly weights strong
  PGA bias above aggregate/untriggered PGA error.

Coordinate control:

- rt13 confirms that relative-only coordinates are not enough by themselves:
  train PGA MAE degrades from rt7's `0.1250` to `0.1942`, validation PGA MAE
  degrades to `0.2891`, and validation location vector error degrades to
  `19.1884`.
- This does not invalidate VS30. It means rt14/rt15 are necessary tests:
  VS30 has to recover information lost when absolute coordinates are removed.

## Recommended Next Actions

1. Rerun rt12, rt14, and rt15 after the corrected `data_path` change.
2. Treat rt10 as the current best completed absolute-coordinate anchor.
3. Keep rt6 as a simpler fallback if loc MDN is considered too costly or if
   later VS30 results interact badly with loc MDN.
4. Do not run a large second-round matrix yet. Wait for rt12/rt14/rt15.
5. If VS30 finishes successfully:
   - compare rt12 against rt7 and rt10 to decide whether VS30 adds value under
     absolute coordinates;
   - compare rt14 against rt13 to decide whether relative coordinates plus VS30
     recover transferable site information;
   - compare rt15 against rt14 to decide whether a weak absolute-coordinate
     channel is useful.
6. Run rt16-rt19 as a separate station-adapter transfer/metadata-injection
   ablation if resources are available. These configs keep the rt10
   heads/loss/absolute-coordinate setup but remove or isolate the b54
   station-adapter transfer; rt18/rt19 additionally inject station coordinate
   embeddings inside the adapter before station embedding pooling.

## How To Read rt16-rt19 When Results Arrive

- rt10 remains the external anchor: same heads, mean-aux weight, realtime
  sampling, and absolute coordinates.
- rt16 tests the strongest version of "skip inherited single-station/full-model
  station representation": no b54 transfer at all.
- rt17 is the cleaner adapter-only test: transfer b54 except
  `waveform_model.1.*`, then reinitialize the station adapter.
- rt18 and rt19 should be compared primarily against rt17. rt18 injects station
  coordinate embedding into adapter pooling queries; rt19 uses the same
  coordinate embedding as feature-level FiLM before pooling.
- Judge the result with both train and validation metrics. A useful change
  should improve or preserve validation PGA MAE while also helping train PGA
  fit, strong-PGA bias, current-time breakdowns, and target-type breakdowns.
  Because rt10 uses loc MDN, also record magnitude MAE, loc vector error, and
  loc NLL/MDN metrics; do not record only PGA.
- If rt17 beats rt10, use transfer-except-adapter as the future warm-start
  default. If rt18 or rt19 beats rt17, keep the better early coordinate-injection
  adapter. If both lose to rt17, keep coordinate injection after the station
  adapter and stop adding adapter metadata variants in this round.

Tentative second-round candidates after VS30 reruns:

- `best_abs`: rt10-style loc MDN, mag Gaussian, PGA MDN3, mean aux 0.10,
  absolute coordinates, VS30 off.
- `best_abs_vs30`: same as `best_abs` plus VS30 site-affine, only if rt12 helps.
- `best_rel_vs30` or `best_relabs001_vs30`: only if rt14/rt15 beat rt13 by a
  large margin.
- `strong_bias_variant`: mag+loc MDN only if the next objective prioritizes
  strong-PGA bias over aggregate and untriggered PGA MAE.
