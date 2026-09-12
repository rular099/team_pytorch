# CODEX-RESULT: RT57 GTNP v2 full validation review request

请 ChatGPT Project 在 GitHub 上独立审查本报告、RT57 实现以及统计解释，并给出唯一、快速推进的下一步建议。
不要建议重复已经完成的 smoke、query-geometry diagnostics 或能够直接从现有 NPZ 回答的检查。

```text
[CODEX-RESULT]
task_id: 20260912-rt57-gtnp-v2-full-validation-analysis
branch: rt57-geo-temporal-neural-process
implementation_commit: 49c6d30335c5e6df7cfbde1ef03caf784ae3aef6
experiment: one seed, fixed six epochs, validation only
checkpoint_epoch: 6
checkpoint_reported_loss: 1.2587252855300903
hpc_status: completed
test_split_used: false
raw_npz_pushed: false (trusted local HPC artifacts, approximately 190 MB total)
[/CODEX-RESULT]
```

## 1. Protocol and alignment audit

The RT57 random validation, RT56 random baseline and waveform-roll control contain exactly aligned:

- 9,681 realtime samples;
- 1,310 unique event IDs;
- 75,654 valid PGA targets;
- 15 target slots per sample;
- only triggered-noninput or untriggered targets under random geometry;
- identical event IDs/order, target labels/validity/indices/types, query coordinates,
  input-station coordinates/validity, requested elapsed times and random selected/requested
  station counts between RT56 and RT57.

All point metrics recomputed from the NPZ match the formal metrics JSON at
`rtol=atol=1e-12`. Paired uncertainty below uses 5,000 percentile bootstrap replicates,
clustered by `event_id`, seed `20260912`.

The normal RT56 and RT57 archives also have identical event/sample order, labels,
target masks/indices, station geometry/validity and elapsed times: 9,681 samples,
1,383 events and 89,770 valid targets.

## 2. Input artifact identities

```text
RT56 random NPZ
  5ca1868b988c0281f840700f49d3978c2f0bafb2eddf77b897d1132483d972d7
RT56 random metrics JSON
  4aa2085c8d42b240ab0f39fd7dd1dca7130d628d57ec085d4760e88d75944d59
RT56 normal NPZ
  56cd7896e98d0be1fdc23e6c5e0c85cb826f9372573a587a0cc3fab55c3d1273
RT56 normal metrics JSON
  5ac3572281d491113a9cdff13552128f629b567e56790a550641e21e12cc5173

RT57 random NPZ
  c12483b6a7f7a0287dfa5d5e7080b08f2bc0c84e06a6e56607d49d06741d146d
RT57 random metrics JSON
  c7be6f5d21c386cd409bcc0f700025260b50897464bd14e91427d5426e250a19
RT57 normal NPZ
  b2ac2da46a60ab774881fb24fe29332e9b95fd2ae08f5f15df77cd7d72a160d0
RT57 normal metrics JSON
  f32d8fc971e5798da18f1322a2e24a36845b6c0d826c1cbf718e6e2e80bc23ca
RT57 waveform-roll NPZ
  5302cd36d710cc8fcd690e24eed1e4ca0ede3457d2ea6e367b06e6751a36c73f
RT57 waveform-roll metrics JSON
  274d5ce84d21c151b503f84849d7342fccb4364a049e0489289c66279552c982
```

## 3. Formal random-validation result

All PGA values and errors are in `log10(m/s^2)`.

| Metric | RT56 | RT57 | RT57 - RT56 |
|---|---:|---:|---:|
| MAE | 0.257229 | 0.247857 | -0.009371 |
| RMSE | 0.331337 | 0.320040 | -0.011297 |
| bias | 0.040420 | 0.036834 | -0.003587 |
| correlation | 0.570998 | 0.607537 | +0.036539 |
| R2 | 0.314386 | 0.360340 | +0.045954 |
| slope | 0.347779 | 0.379428 | +0.031649 |
| NLL | 0.255337 | 0.220260 | -0.035077 |
| Brier at -1.2 | 0.199924 | 0.192239 | -0.007685 |
| predictive sigma mean | 0.303748 | 0.303748 | unchanged by design |
| coverage 1 sigma | 0.654876 | 0.673937 | +0.019060 |
| coverage 2 sigma | 0.941127 | 0.947604 | +0.006477 |

Target-weighted MAE delta is `-0.009371`, with event-cluster 95% CI
`[-0.010480, -0.008218]`; 57.4% of targets and 64.9% of realtime samples improve.

### Target and requested-station strata

| Population | Targets | RT56 MAE | RT57 MAE | Delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| triggered non-input | 33,601 | 0.241671 | 0.232664 | -0.009007 | [-0.009964, -0.008040] |
| untriggered | 42,053 | 0.269659 | 0.259997 | -0.009662 | [-0.011300, -0.007902] |

| Requested inputs | Targets | RT56 MAE | RT57 MAE | Delta | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 15,157 | 0.302749 | 0.291627 | -0.011122 | [-0.013408, -0.008527] |
| 3 | 14,427 | 0.259765 | 0.250223 | -0.009542 | [-0.011588, -0.007672] |
| 5 | 12,694 | 0.245663 | 0.237013 | -0.008650 | [-0.010054, -0.007292] |
| 8 | 12,072 | 0.240514 | 0.230976 | -0.009538 | [-0.011469, -0.007830] |
| 12 | 10,856 | 0.245546 | 0.238226 | -0.007320 | [-0.010680, -0.003143] |
| 16 | 10,448 | 0.233192 | 0.223780 | -0.009411 | [-0.011537, -0.007519] |

Every requested-station stratum improves with a CI excluding zero.

### Requested elapsed time

| Time | RT56 MAE | RT57 MAE | Delta | 95% CI |
|---:|---:|---:|---:|---:|
| 1 s | 0.263879 | 0.257191 | -0.006689 | [-0.009469, -0.003423] |
| 3 s | 0.241143 | 0.233508 | -0.007635 | [-0.009047, -0.006305] |
| 5 s | 0.241161 | 0.233507 | -0.007654 | [-0.009086, -0.006254] |
| 10 s | 0.257884 | 0.249340 | -0.008544 | [-0.010489, -0.006696] |
| 20 s | 0.268645 | 0.257815 | -0.010830 | [-0.012811, -0.008976] |
| 40 s | 0.264111 | 0.249620 | -0.014491 | [-0.016864, -0.012393] |
| 90 s | 0.262377 | 0.250380 | -0.011997 | [-0.013828, -0.010239] |

RT57 improves every elapsed-time stratum relative to RT56. At 1 s it remains
approximately 0.0016 worse than the historical RT55 zero-shot value of 0.2556,
so it nearly, but not fully, recovers that older early-window result.

## 4. Spatial-field result

For 5,762 realtime fields with at least five valid targets, metrics are averaged
equally across fields.

| Metric | RT56 | RT57 | Delta | 95% CI |
|---|---:|---:|---:|---:|
| Pearson | 0.414838 | 0.494516 | +0.079678 | [+0.0697, +0.0899] |
| Spearman | 0.3936 | 0.4736 | +0.0800 | [+0.0697, +0.0907] |
| event-centered MAE | 0.2119 | 0.2040 | -0.0079 | [-0.0087, -0.0071] |
| pairwise-delta MAE | 0.313409 | 0.301876 | -0.011533 | [-0.0127, -0.0103] |
| true P95-P05 range | 0.831276 | 0.831276 | 0 |
| predicted P95-P05 range | 0.316946 | 0.360140 | +0.043194 | not bootstrapped here |
| mean per-field range ratio | 0.413017 | 0.464444 | +0.051426 | not bootstrapped here |
| range absolute error | 0.520483 | 0.478105 | -0.042378 | [-0.0453, -0.0395] |

RT57 therefore improves within-event ordering, pairwise differences and spatial
dynamic range. It still compresses amplitude substantially: the ratio of mean
predicted range to mean true range is only `0.360140 / 0.831276 = 0.4332`.

Important tool issue: the current `tools/analyze_random_geometry_full_npz.py`
markdown decision prose hardcodes the words `regresses` and `worsens` for the
1 s and range-error rows. Those words happened to match the earlier RT55-to-RT56
case but are wrong here because both RT57 deltas are negative. The numerical JSON,
CSV, tables and recomputation are correct; do not use the hardcoded prose as evidence.

## 5. Actual one-input-station spatial result

For actual selected-station-count equal to one and at least five valid targets
(1,494 fields, 767 event clusters):

| Metric | RT56 | RT57 | Delta | 95% event-cluster CI |
|---|---:|---:|---:|---:|
| predicted P95-P05 range | 0.003797 | 0.097347 | +0.093551 | [+0.087600, +0.099988] |
| mean per-field range ratio | 0.005101 | 0.125352 | +0.120251 | [+0.111914, +0.130028] |
| range absolute error | 0.815030 | 0.721542 | -0.093488 | [-0.099862, -0.087472] |
| pairwise-delta MAE | 0.354079 | 0.340582 | -0.013497 | [-0.015396, -0.011704] |
| Pearson | 0.136114 | 0.323983 | +0.187869 | [+0.155442, +0.220127] |

The ratio of mean predicted range to mean true range changes from `0.00464` to
`0.11889`, a material relative improvement, but the remaining absolute compression
is still severe.

## 6. Waveform-station permutation control

The full RT57 model with correct station/waveform association versus waveform-only
station roll gives:

| Metric | Correct | Rolled | Rolled - correct |
|---|---:|---:|---:|
| MAE | 0.247857 | 0.261727 | +0.013870 |
| RMSE | 0.320040 | 0.335321 | +0.015281 |
| correlation | 0.607537 | 0.560923 | -0.046614 |
| R2 | 0.360340 | 0.297797 | -0.062543 |
| slope | 0.379428 | 0.352088 | -0.027339 |
| NLL | 0.220260 | 0.289800 | +0.069540 |
| Brier | 0.192239 | 0.205214 | +0.012975 |
| spatial Pearson (at least 5 targets) | 0.494516 | 0.434991 | -0.059526 |
| spatial pairwise-delta MAE | 0.301876 | 0.317461 | +0.015585 |

Full-model MAE degradation is `+0.013870`, event-cluster 95% CI
`[+0.0126, +0.0152]`. Requested station count one is unchanged, as expected for
a one-element permutation. Counts 3, 5, 8, 12 and 16 degrade by 0.0157, 0.0199,
0.0190, 0.0158 and 0.0162 respectively, each with CI excluding zero.

This full-model control also perturbs the frozen RT56 base, so it cannot by itself
attribute degradation to the new residual branch. The NPZ exports allow a fixed-base
counterfactual:

```text
correct_final      = correct_RT56_base + correct_RT57_delta
counterfactual     = correct_RT56_base + rolled_RT57_delta
```

```latex
\mathrm{correct\_final}=\mathrm{base}_{\mathrm{correct}}+\Delta_{\mathrm{correct}},
\qquad
\mathrm{counterfactual}=\mathrm{base}_{\mathrm{correct}}+\Delta_{\mathrm{rolled}}.
```

Under this fixed base, replacing only the delta increases MAE from 0.247857 to
0.249504: `+0.001647`, 95% CI `[+0.001301, +0.001991]`. For actual multi-station
samples the penalty is `+0.002234`, CI `[+0.001759, +0.002714]`. The correct and
rolled delta differ by mean absolute 0.02248 on multi-station targets, with
correlation 0.7871.

The task-relevant local auxiliary gives stronger direct evidence that `u/d` carries
station-specific waveform information:

| Multi-station local residual | Correct | Rolled |
|---|---:|---:|
| targets | 40,428 | 40,428 |
| MAE | 0.094864 | 0.280022 |
| correlation | 0.867133 | -0.043037 |
| zero-prediction baseline MAE | 0.208759 | 0.208759 |

For one-station absolute local PGA, correct MAE is 0.205468 and correlation is
0.793878; the roll is identity and therefore unchanged.

Interpretation proposed for review: the new representation demonstrably retains
station-specific waveform information and the final residual is measurably sensitive
to the waveform/station association, so the gain is not purely geometry-prior-driven.
However, most aggregate residual benefit survives the fixed-base rolled-delta
counterfactual; geometry/event-common information still appears dominant in the final
PGA correction. This is partial rather than complete resolution of station collapse.

## 7. Normal-validation retention

| Metric | RT56 | RT57 | Delta |
|---|---:|---:|---:|
| all-target MAE | 0.133781 | 0.131302 | -0.002479 |
| RMSE | 0.199958 | 0.195701 | -0.004257 |
| R2 | 0.698414 | 0.711119 | +0.012705 |
| NLL | -0.816265 | -0.829016 | -0.012751 |
| Brier | 0.102763 | 0.101152 | -0.001611 |

All-target paired MAE delta is `-0.002479`, 95% CI
`[-0.002925, -0.002053]`.

| Normal population | Targets | RT56 MAE | RT57 MAE | Delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| input | 63,651 | 0.097597 | 0.097620 | +0.000023 | [-0.000232, +0.000272] |
| all non-input | 26,119 | 0.221959 | 0.213383 | -0.008576 | [-0.009721, -0.007488] |
| triggered non-input | 4,003 | 0.208384 | 0.190548 | -0.017836 | [-0.020706, -0.015073] |
| untriggered | 22,116 | 0.224416 | 0.217517 | -0.006900 | [-0.008037, -0.005825] |

Normal input performance is statistically retained and normal non-input performance
improves.

## 8. Predeclared go/no audit for the current gamma=1 model

| Gate | Result |
|---|---|
| random non-input MAE no worse than RT56 | PASS |
| random slope at least 0.40 | FAIL: 0.379428 |
| actual one-station spatial range ratio materially improves | PASS |
| one-station pairwise-delta MAE does not worsen | PASS |
| waveform-only permutation measurably degrades RT57 | PASS, with attribution nuance above |
| normal retention | PASS |

Strict result: five of six gates pass. The current `gamma=1` checkpoint/config should
be retained as a successful model result, but it does not satisfy the original
all-gates go criterion because slope is 0.02057 below threshold.

## 9. Existing-NPZ residual-scale diagnostic

No retraining is required to test whether the remaining slope miss is simple residual
under-scaling. The diagnostic prediction is first written in terminal-friendly form:

```text
PGA_final = PGA_RT56_base + gamma * delta_RT57
```

Raw LaTeX:

```latex
\mathrm{PGA}_{\mathrm{final}}
=\mathrm{PGA}_{\mathrm{RT56\ base}}+\gamma\,\Delta_{\mathrm{RT57}}.
```

A validation-only grid from gamma 0 to 3 in increments of 0.01 found:

- minimum random MAE at gamma 1.77;
- minimum random RMSE / maximum R2 at gamma 1.59;
- minimum random NLL at gamma 1.44;
- the first grid point with slope at least 0.40 is gamma 1.66.

Gamma 1.66 is reported as a simple gate-oriented compromise, not a formal result and
not an independent test claim:

| Metric | RT57 gamma=1 | Post-hoc gamma=1.66 |
|---|---:|---:|
| random MAE | 0.247857 | 0.245578 |
| random RMSE | 0.320040 | 0.318243 |
| random R2 | 0.360340 | 0.367504 |
| random slope | 0.379428 | 0.400316 |
| random NLL | 0.220260 | 0.217530 |
| random Brier | 0.192239 | 0.189875 |
| random 1-sigma coverage | 0.673937 | 0.681762 |
| mean spatial range ratio (at least 5 targets) | 0.464444 | 0.510231 |
| spatial pairwise-delta MAE | 0.301876 | 0.298782 |
| actual-one-station range ratio | 0.125352 | 0.207285 |
| actual-one-station pairwise-delta MAE | 0.340582 | 0.336126 |
| normal MAE | 0.131302 | 0.131012 |
| normal NLL | -0.829016 | -0.822428 |
| normal Brier | 0.101152 | 0.100961 |

At gamma 1.66, normal NLL becomes worse than gamma 1 but remains better than RT56
`-0.816265`; normal MAE and Brier remain slightly better than gamma 1. The scalar was
selected on the same validation set, has not been implemented as a config switch and
must not be presented as test performance.

## 10. Requested ChatGPT review

Please respond with exactly these sections:

1. `审查结论`
   - Pass / conditional pass / fail.
   - Audit the metric directions, event-cluster bootstrap interpretation, one-station
     spatial conclusion and fixed-base waveform-roll attribution.

2. `可以声称什么`
   - Separate verified facts, reasonable inferences and claims that remain unsupported.
   - In particular decide whether it is justified to say station-feature collapse is
     partially solved, and whether the final residual remains geometry-dominated.

3. `当前 RT57 的 go/no`
   - Apply the predeclared six gates rather than silently changing them after seeing data.

4. `gamma=1.66 是否可接受`
   - Decide whether a default-off, RT55/RT56-compatible inference residual-scale switch
     selected on validation is scientifically acceptable.
   - Decide whether the existing offline evidence is sufficient or whether one formal
     validation rerun is required before locking it.

5. `下一步优先级`
   - `现在必须做`
   - `下一项高价值工作`
   - `以后可选`
   - Optimize for fast progress. Do not request another diagnostic matrix or repeated
     smoke tests.

6. `最小新实验（最多一个）`
   - If any HPC work is necessary, recommend at most one experiment, state the hypothesis,
     split, primary metrics and explicit go/no rule. Prefer no-retraining reuse of the
     completed epoch-6 checkpoint if statistically defensible.

7. `给 Codex 的下一轮 AI-HANDOFF`
   - If code changes are recommended, provide a complete executable handoff block with
     exact compatibility, config, test and launcher requirements.
   - RT55 loading/inference and all disabled-switch outputs must remain unchanged.
   - If no code change is needed, state `proposed_change: none` explicitly.

## 11. Reproduction command for the core RT56-to-RT57 pairing

```bash
python tools/analyze_random_geometry_full_npz.py \
  --baseline-npz '<RT56 random NPZ>' \
  --candidate-npz '<RT57 random NPZ>' \
  --baseline-metrics '<RT56 random metrics JSON>' \
  --candidate-metrics '<RT57 random metrics JSON>' \
  --baseline-name RT56 \
  --candidate-name RT57 \
  --output-prefix '<output prefix>' \
  --bootstrap-replicates 5000 \
  --seed 20260912 \
  --trusted-pickle-input
```

The raw object-array NPZ files are trusted artifacts supplied from the user's HPC run.
No held-out test data were inspected in this analysis.
