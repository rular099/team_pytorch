# RT55/RT56 random-geometry full validation: offline paired analysis

- Baseline: `RT55 ep32 zero-shot random-mask`
- Candidate: `RT56 mixed-random finetuned best`
- Exactly aligned samples: 9,681
- Unique event IDs: 1,310
- Valid targets: 75,654
- Coordinate: `log10(m/s^2)`
- Uncertainty: 95% percentile bootstrap CI, clustered by `event_id`.

## Decision summary

- The candidate improves target-weighted MAE by 0.0259 (9.1% relative); candidate-minus-baseline delta -0.0259, 95% CI [-0.0288, -0.0230].
- The gain is present for every requested station-count stratum, and it is larger for untriggered than triggered non-input targets.
- The 1 s window regresses: MAE 0.2556 to 0.2639; delta 0.0082.
- For fields with at least five targets, mean Pearson improves by 0.0548, but mean P95-P05 range error worsens by 0.0123. The model still compresses spatial amplitude.
- Therefore the finetuning is useful and should be retained, but the earliest window and spatial dynamic range remain explicit follow-up targets.

## Formal full-validation metrics

| Metric | Baseline | Candidate |
|---|---:|---:|
| mae | 0.2831 | 0.2572 |
| rmse | 0.3820 | 0.3313 |
| bias | -0.0913 | 0.0404 |
| correlation | 0.4405 | 0.5710 |
| r2 | 0.0885 | 0.3144 |
| nll | 1.4739 | 0.2553 |
| predictive_sigma_mean | 0.1897 | 0.3037 |
| coverage_1sigma | 0.4423 | 0.6549 |
| coverage_2sigma | 0.6918 | 0.9411 |

## Paired point-error comparison

Negative MAE/RMSE deltas favor the candidate.

| Target population | Targets | Baseline MAE | Candidate MAE | MAE delta | 95% CI | Improved targets |
|---|---:|---:|---:|---:|---:|---:|
| all | 75,654 | 0.2831 | 0.2572 | -0.0259 | [-0.0288, -0.0230] | 53.2% |
| non_input | 75,654 | 0.2831 | 0.2572 | -0.0259 | [-0.0288, -0.0229] | 53.2% |
| triggered_noninput | 33,601 | 0.2590 | 0.2417 | -0.0173 | [-0.0209, -0.0136] | 53.0% |
| untriggered | 42,053 | 0.3024 | 0.2697 | -0.0327 | [-0.0370, -0.0286] | 53.3% |
| input | 0 | NA | NA | NA | NA | NA |

## Point error by requested input-station count

| Requested stations | Samples | Targets | Baseline MAE | Candidate MAE | MAE delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,540 | 15,157 | 0.3234 | 0.3027 | -0.0206 | [-0.0266, -0.0144] |
| 3 | 1,590 | 14,427 | 0.2826 | 0.2598 | -0.0229 | [-0.0289, -0.0169] |
| 5 | 1,590 | 12,694 | 0.2782 | 0.2457 | -0.0325 | [-0.0379, -0.0270] |
| 8 | 1,646 | 12,072 | 0.2700 | 0.2405 | -0.0295 | [-0.0348, -0.0241] |
| 12 | 1,668 | 10,856 | 0.2676 | 0.2455 | -0.0220 | [-0.0287, -0.0149] |
| 16 | 1,647 | 10,448 | 0.2627 | 0.2332 | -0.0295 | [-0.0352, -0.0238] |

## Point error by requested elapsed time

| Elapsed time (s) | Samples | Targets | Baseline MAE | Candidate MAE | MAE delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 1,821 | 16,348 | 0.2556 | 0.2639 | 0.0082 | [0.0040, 0.0125] |
| 3.0 | 1,310 | 11,052 | 0.2446 | 0.2411 | -0.0035 | [-0.0072, -0.0001] |
| 5.0 | 1,310 | 10,071 | 0.2541 | 0.2412 | -0.0130 | [-0.0177, -0.0085] |
| 10.0 | 1,310 | 9,491 | 0.2960 | 0.2579 | -0.0381 | [-0.0448, -0.0316] |
| 20.0 | 1,310 | 9,603 | 0.3216 | 0.2686 | -0.0529 | [-0.0604, -0.0454] |
| 40.0 | 1,310 | 9,647 | 0.3187 | 0.2641 | -0.0546 | [-0.0620, -0.0474] |
| 90.0 | 1,310 | 9,442 | 0.3181 | 0.2624 | -0.0558 | [-0.0636, -0.0487] |

## Paired spatial-field comparison (at least 5 valid targets)

The delta is candidate minus baseline. Pearson/Spearman are better when positive; error metrics are better when negative.

| Population | Metric | Paired samples | Baseline | Candidate | Delta | 95% CI | Candidate better |
|---|---|---:|---:|---:|---:|---:|---:|
| all | pearson | 5,762 | 0.3600 | 0.4148 | 0.0548 | [0.0473, 0.0627] | 61.2% |
| all | spearman | 5,762 | 0.3432 | 0.3936 | 0.0503 | [0.0425, 0.0582] | 57.1% |
| all | event_centered_mae | 5,762 | 0.2210 | 0.2119 | -0.0091 | [-0.0100, -0.0083] | 62.2% |
| all | event_centered_rmse | 5,762 | 0.2710 | 0.2593 | -0.0116 | [-0.0126, -0.0107] | 63.8% |
| all | pairwise_delta_mae | 5,762 | 0.3270 | 0.3134 | -0.0136 | [-0.0147, -0.0124] | 63.0% |
| all | pairwise_delta_rmse | 5,762 | 0.4028 | 0.3855 | -0.0173 | [-0.0188, -0.0159] | 63.8% |
| all | range_abs_error | 5,762 | 0.5082 | 0.5205 | 0.0123 | [0.0095, 0.0151] | 55.0% |
| non_input | pearson | 5,762 | 0.3600 | 0.4148 | 0.0548 | [0.0472, 0.0632] | 61.2% |
| non_input | spearman | 5,762 | 0.3432 | 0.3936 | 0.0503 | [0.0423, 0.0581] | 57.1% |
| non_input | event_centered_mae | 5,762 | 0.2210 | 0.2119 | -0.0091 | [-0.0099, -0.0083] | 62.2% |
| non_input | event_centered_rmse | 5,762 | 0.2710 | 0.2593 | -0.0116 | [-0.0126, -0.0106] | 63.8% |
| non_input | pairwise_delta_mae | 5,762 | 0.3270 | 0.3134 | -0.0136 | [-0.0148, -0.0124] | 63.0% |
| non_input | pairwise_delta_rmse | 5,762 | 0.4028 | 0.3855 | -0.0173 | [-0.0188, -0.0159] | 63.8% |
| non_input | range_abs_error | 5,762 | 0.5082 | 0.5205 | 0.0123 | [0.0093, 0.0151] | 55.0% |
| triggered_noninput | pearson | 2,530 | 0.3543 | 0.4148 | 0.0605 | [0.0494, 0.0722] | 60.8% |
| triggered_noninput | spearman | 2,530 | 0.3304 | 0.3877 | 0.0574 | [0.0458, 0.0694] | 55.3% |
| triggered_noninput | event_centered_mae | 2,530 | 0.2034 | 0.1923 | -0.0110 | [-0.0123, -0.0097] | 64.2% |
| triggered_noninput | event_centered_rmse | 2,530 | 0.2480 | 0.2337 | -0.0143 | [-0.0158, -0.0128] | 65.2% |
| triggered_noninput | pairwise_delta_mae | 2,530 | 0.3047 | 0.2876 | -0.0171 | [-0.0190, -0.0153] | 64.9% |
| triggered_noninput | pairwise_delta_rmse | 2,530 | 0.3718 | 0.3504 | -0.0214 | [-0.0237, -0.0193] | 65.2% |
| triggered_noninput | range_abs_error | 2,530 | 0.4071 | 0.4330 | 0.0259 | [0.0214, 0.0302] | 51.7% |
| untriggered | pearson | 3,516 | 0.3534 | 0.3888 | 0.0355 | [0.0262, 0.0456] | 58.8% |
| untriggered | spearman | 3,516 | 0.3367 | 0.3687 | 0.0321 | [0.0224, 0.0423] | 53.0% |
| untriggered | event_centered_mae | 3,516 | 0.2135 | 0.2077 | -0.0059 | [-0.0069, -0.0048] | 58.6% |
| untriggered | event_centered_rmse | 3,516 | 0.2611 | 0.2536 | -0.0075 | [-0.0087, -0.0064] | 59.8% |
| untriggered | pairwise_delta_mae | 3,516 | 0.3190 | 0.3101 | -0.0089 | [-0.0103, -0.0074] | 58.7% |
| untriggered | pairwise_delta_rmse | 3,516 | 0.3915 | 0.3802 | -0.0113 | [-0.0131, -0.0095] | 59.8% |
| untriggered | range_abs_error | 3,516 | 0.5025 | 0.5035 | 0.0010 | [-0.0022, 0.0041] | 56.9% |
| input | pearson | 0 | NA | NA | NA | NA | NA |
| input | spearman | 0 | NA | NA | NA | NA | NA |
| input | event_centered_mae | 0 | NA | NA | NA | NA | NA |
| input | event_centered_rmse | 0 | NA | NA | NA | NA | NA |
| input | pairwise_delta_mae | 0 | NA | NA | NA | NA | NA |
| input | pairwise_delta_rmse | 0 | NA | NA | NA | NA | NA |
| input | range_abs_error | 0 | NA | NA | NA | NA | NA |

## Interpretation guardrails

- This is paired validation analysis, not an independent test-set claim.
- Event-cluster bootstrap keeps all time windows/repeated occurrences of one event in the same resampling unit.
- Target-weighted point metrics and sample-weighted spatial metrics answer different questions and must not be mixed.
- Full numerical details, including selected-station-count strata and the >=2-target spatial view, are in the companion JSON/CSV outputs.
