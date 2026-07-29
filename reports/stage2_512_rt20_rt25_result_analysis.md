# rt20-rt25 Station-Feature Collapse Follow-Up Analysis

Date: 2026-06-06

## Status

rt20, rt21, and rt24 have complete `best` and `last` eval outputs. rt22, rt23,
and rt25 trained to epoch 150, but their eval logs stopped during amplitude
sensitivity diagnostics:

```text
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

This is an eval script issue, not a training failure. The station-local PGA aux
tensor is appended to the model forward output for DDP, while
`eval_checkpoint.py` still used `outputs[-1]` as the PGA head in
`diagnose_amplitude_sensitivity`. The script has been fixed to index PGA via
`raw_model.output_layout.index('pga')`. Re-run eval for rt22, rt23, and rt25
before making final conclusions about the aux branch.

## Completed Eval Metrics

Use rt10 `last` as the main anchor because it is the accepted balanced
Japan-overfit baseline.

| run | ckpt | train PGA | val PGA | val untriggered | val strong | strong bias | Brier | val mag | val loc vec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rt10 | last | 0.1296 | 0.2667 | 0.3495 | 0.3052 | -0.2400 | 0.1431 | 0.3596 | 15.2190 |
| rt20 | last | 0.1365 | 0.2802 | 0.3727 | 0.2775 | -0.1780 | 0.1616 | 0.3954 | 15.5504 |
| rt21 | last | 0.1440 | 0.2858 | 0.3815 | 0.2671 | -0.1556 | 0.1537 | 0.4446 | 14.7790 |
| rt24 | last | 0.1307 | 0.2763 | 0.3598 | 0.2737 | -0.1843 | 0.1616 | 0.3456 | 15.8463 |
| rt20 | best | 0.1838 | 0.2763 | 0.3386 | 0.3379 | -0.3049 | 0.1250 | 0.3921 | 16.2625 |
| rt21 | best | 0.2233 | 0.3043 | 0.4028 | 0.2875 | -0.1675 | 0.1554 | 0.4347 | 16.0492 |
| rt24 | best | 0.2020 | 0.2846 | 0.3522 | 0.3205 | -0.2844 | 0.1278 | 0.3768 | 16.8376 |

Station-count validation PGA MAE:

| run | ckpt | n=1 | n=2-3 | n=4-5 | n=6-10 | n=11-15 | n=16+ |
|---|---|---:|---:|---:|---:|---:|---:|
| rt10 | last | 0.3466 | 0.3830 | 0.2606 | 0.2789 | 0.2258 | 0.2560 |
| rt20 | last | 0.3544 | 0.3999 | 0.3040 | 0.3102 | 0.2482 | 0.2594 |
| rt21 | last | 0.3714 | 0.4233 | 0.2934 | 0.3033 | 0.2727 | 0.2632 |
| rt24 | last | 0.3426 | 0.3939 | 0.2704 | 0.2867 | 0.2491 | 0.2633 |

## Training-Curve And Diagnostic Summary

| run | train last10 | val min | val min epoch | val last10 | raw cos last10 | station cos last10 | residual ratio | residual gate | local aux loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rt10 | 0.0998 | 1.2680 | 10 | 2.3251 | n/a | 0.7354 | n/a | n/a | n/a |
| rt20 | 0.1935 | 1.2783 | 32 | 2.2937 | 0.9939 | 0.7358 | 0.0391 | n/a | n/a |
| rt21 | 0.2302 | 1.2816 | 23 | 2.1218 | 0.9950 | 0.7215 | 0.0359 | 0.0518 | n/a |
| rt22 | 0.0970 | 1.2402 | 33 | 2.2278 | 0.9940 | 0.7369 | 0.0390 | n/a | 0.0030 |
| rt23 | 0.2013 | 1.2728 | 34 | 2.2666 | 0.9951 | 0.7194 | 0.0356 | 0.0538 | 0.0031 |
| rt24 | 0.1568 | 1.2391 | 21 | 2.4778 | 0.9943 | 0.7355 | 0.0385 | n/a | n/a |
| rt25 | 0.2347 | 1.2502 | 35 | 2.1099 | 0.9950 | 0.7197 | 0.0356 | 0.0514 | 0.0031 |

The local aux branch is learning a non-trivial station residual predictor:
`diag_station_local_pga_aux_std` is about 0.66-0.69 near the end, and the
weighted aux loss drops to roughly 0.0026-0.0031. However, the full eval metrics
for rt22/rt23/rt25 are missing until eval is rerun.

## Interpretation

1. None of rt20, rt21, or rt24 replaces rt10 as the balanced anchor. All three
   are worse on overall val PGA. rt20 and rt24 last checkpoints are close but
   still behind rt10: 0.2802 and 0.2763 versus 0.2667. rt21 is further behind
   at 0.2858.

2. The strong-PGA behavior improves in the last checkpoints, but the gain is
   too narrow. rt20/rt21/rt24 last reduce strong MAE from rt10's 0.3052 to
   0.2775/0.2671/0.2737 and reduce the negative strong bias. But they worsen
   untriggered targets and overall PGA. This means the new branches are shifting
   the bias tradeoff, not giving a clean improvement.

3. rt20 best is the only completed run with a clear calibration/untriggered
   signal: Brier improves from rt10-last 0.1431 to 0.1250 and untriggered MAE
   improves from 0.3495 to 0.3386. But it pays for that with worse strong PGA
   (0.3379 and bias -0.3049) and worse loc. It is not acceptable as the default.

4. The explicit residual branch is being suppressed. The residual gate starts
   at 0.1 and ends around 0.05 in rt21/rt23/rt25. The residual norm ratio is
   only about 0.036, so the model uses the residual path weakly. This supports
   the suspicion that a simple additive residual branch is not strong enough to
   change the PGA readout behavior.

5. Poolq8 does not fix collapse. rt24 last has nearly the same train PGA as rt10
   and slightly better mag, but worse val PGA, untriggered targets, Brier, and
   loc. It should not be kept unless rt25 later proves that poolq8 helps when
   combined with the local aux branch.

6. Raw station-feature collapse remains. Eval raw-feature cosine for completed
   runs stays near rt10: rt10 0.9973, rt20 0.9973, rt21 0.9978, rt24 0.9974.
   The training CSV raw-cosine diagnostics are lower numerically, but the
   pattern is still not a convincing anti-collapse gain. Event-centering gives
   negative residual cosine by construction; that does not mean the original
   station embeddings are less collapsed.

## Recommended Next Action

1. Re-run eval for rt22, rt23, and rt25 after pulling the fixed
   `eval_checkpoint.py`. These are the only runs that directly test the
   station-local PGA auxiliary signal, and their training curves are not enough
   to decide.

2. If only one aux run can be evaluated first, prioritize rt22. It has the best
   train last10 loss among rt20-rt25 and a lower val minimum than rt10, while
   avoiding the residual-add branch that the model appears to suppress.

3. If rt22/rt23/rt25 still fail to beat rt10 on balanced PGA, stop this family
   as the default path. The next diagnostic should first determine whether
   pre-adapter DiTing tokens `F` contain usable station-specific PGA residual
   information, rather than adding another weak gated vector into the shared
   station token.

4. That frozen-base realtime diagnostic is encoded as rt26-rt29 in
   `reports/stage2_512_layerwise_station_temporal_plan.md`. Analyze those runs
   around `pga_temporal_base`, `pga_temporal_final`, `|delta|`,
   `corr(delta, label-base)`, and the station-roll/zero-token controls. Ordinary
   final PGA MAE alone is not enough evidence that raw station waveform features
   are being used.
