# RT59-v3 paired validation analysis

> **Archived result with a known metric-definition error.** The formal P95-P05
> correction and 25-required/4-diagnostic gate inventory are in
> [`../rt59_v3_review_correction_20260920/`](../rt59_v3_review_correction_20260920/).

- Decision: **NO_GO**
- Random targets: 75,654
- Normal targets: 89,770 (input 63,651, non-input 26,119)
- Bootstrap: event_id paired, 5000 draws, seed 20260915

The decision is conjunctive. `INCOMPLETE_EVIDENCE` means at least one required quantity was absent; it must not be treated as a pass.

| Gate | Value | Rule | Pass |
|---|---:|---:|:---:|
| random_mae | 0.23782827292020176 | <= 0.242 | True |
| random_slope | 0.40245343115082327 | >= 0.45 | False |
| random_r2 | 0.39499520226406926 | >= 0.39 | True |
| random_range_ratio_ge5 | 0.4679495191499757 | >= 0.55 | False |
| random_one_station_range_ratio_ge5 | 0.1745629992826593 | >= 0.25 | False |
| random_one_station_pairwise_delta_mae | 0.336201596663309 | <= 0.33 | False |
| random_requested1_mae | 0.24339119931581682 | <= 0.253 | True |
| random_nll | 0.19901795601251107 | <= 0.22 | True |
| random_delta_mae_ci_upper | -0.006700927930248778 | < 0.0 | True |
| random_delta_brier_ci_upper | -0.004929818689071523 | <= 0.0 | True |
| random_coverage1_absolute_change | 0.020712718428635646 | <= 0.01 | False |
| random_coverage2_absolute_change | 0.00155973246622787 | <= 0.01 | True |
| normal_all_delta_mae_ci_upper | -0.005522173295425348 | < 0.0 | True |
| normal_all_delta_rmse_ci_upper | -0.008499416753756376 | < 0.0 | True |
| normal_noninput_delta_mae_ci_upper | -0.009911931927171479 | < 0.0 | True |
| normal_noninput_delta_rmse_ci_upper | -0.012490213955583172 | < 0.0 | True |
| normal_input_mae_delta | -0.0039322662074091075 | <= 0.0 | True |
| normal_input_rmse_delta | -0.006774727740309711 | <= 0.0 | True |
| normal_all_mae_historical_redline | 0.12500635570447707 | <= 0.136 | True |
| normal_noninput_mae_historical_redline | 0.19930054193565463 | <= 0.218 | True |
| normal_nll_delta | -0.02591293810548101 | <= 0.0 | True |
| normal_brier_delta | -0.005552949034443033 | <= 0.0 | True |
| one_station_range_abs_error_delta | 0.027007405856846134 | <= 0.0 | False |
| normal_coverage1_absolute_change | 0.0148601982845048 | <= 0.01 | False |
| normal_coverage2_absolute_change | 0.003408711150718502 | <= 0.01 | True |
| normal_all_absolute_bias_change | -0.012094614830316673 | <= 0.0 | True |
| normal_all_absolute_slope_error_change | -0.0065948597850304935 | <= 0.0 | True |
| normal_noninput_absolute_bias_change | -0.02098947094120189 | <= 0.0 | True |
| normal_noninput_absolute_slope_error_change | -0.009469914642014299 | <= 0.0 | True |
