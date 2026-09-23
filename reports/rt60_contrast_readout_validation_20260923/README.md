# RT60 final-contrast readout validation

- `rt60_mechanism_pass`: **False** (8/14 gates pass)
- `legacy_full_go`: **False** (18/25 required gates pass)
- Recommendation: retain the RT59 same-forward reference as the development parent.
- Evidence: fixed epoch-8 Japan 2000-2024 validation only; no held-out test.
- Bootstrap: 5000 paired event-cluster draws, seed 20260915.

## RT60 mechanism gates

| Gate | Value | Rule | Pass |
|---|---:|---:|:---:|
| one_station_pairwise_delta_mae_ci_upper | -0.00023474567205445588 | < 0.0 | True |
| one_station_range_abs_error_ci_upper | -0.003048898272655661 | < 0.0 | True |
| random_all_range_abs_error_delta | -0.005765102265537783 | <= 0.0 | True |
| random_mae_delta | -4.041065500171026e-06 | <= 0.0 | True |
| random_rmse_delta | -0.0004965401257044921 | <= 0.0 | True |
| normal_noninput_mae_delta | 0.0005798566878364719 | <= 0.0 | False |
| normal_noninput_rmse_delta | 0.0008742499220860123 | <= 0.0 | False |
| normal_input_max_abs_change | 0.0 | <= 1e-07 | True |
| random_nll_delta | -0.0016678075107380008 | <= 0.0 | True |
| random_brier_delta | -0.00015150251641452006 | <= 0.0 | True |
| normal_all_nll_delta | 0.0003728666675086867 | <= 0.0 | False |
| normal_all_brier_delta | 5.099755635147585e-05 | <= 0.0 | False |
| normal_noninput_nll_delta | 0.0012814635832951102 | <= 0.0 | False |
| normal_noninput_brier_delta | 0.0001752546671272559 | <= 0.0 | False |

## Legacy gates

| Gate | Value | Rule | Pass | Required |
|---|---:|---:|:---:|:---:|
| random_mae | 0.23782423209972556 | <= 0.242 | True | True |
| random_slope | 0.4096128308729092 | >= 0.45 | False | True |
| random_r2 | 0.3969240014144638 | >= 0.39 | True | True |
| random_range_ratio_ge5 | 0.48266912799369277 | >= 0.55 | False | True |
| random_one_station_range_ratio_ge5 | 0.18334815063458884 | >= 0.25 | False | True |
| random_one_station_pairwise_delta_mae | 0.3356466441933151 | <= 0.33 | False | True |
| random_requested1_mae | 0.24329246709114052 | <= 0.253 | True | True |
| random_nll | 0.1973501568132868 | <= 0.22 | True | True |
| random_delta_mae_ci_upper | -0.0066737806586033045 | < 0.0 | True | True |
| random_delta_brier_ci_upper | -0.005074590771883902 | <= 0.0 | True | True |
| random_coverage1_absolute_change | 0.020580537711158642 | <= 0.01 | False | True |
| random_coverage2_absolute_change | 0.0020355830491448623 | <= 0.01 | True | True |
| normal_all_delta_mae_ci_upper | -0.005348807374408626 | < 0.0 | True | True |
| normal_all_delta_rmse_ci_upper | -0.00818147424113941 | < 0.0 | True | True |
| normal_noninput_delta_mae_ci_upper | -0.009256278960139407 | < 0.0 | True | True |
| normal_noninput_delta_rmse_ci_upper | -0.011517431657485548 | < 0.0 | True | True |
| normal_input_mae_delta | -0.003932266040724747 | <= 0.0 | True | True |
| normal_input_rmse_delta | -0.006774727599448166 | <= 0.0 | True | True |
| normal_all_mae_historical_redline | 0.1251750671088236 | <= 0.136 | True | True |
| normal_noninput_mae_historical_redline | 0.19988039835877427 | <= 0.218 | True | True |
| normal_nll_delta | -0.025540104016029352 | <= 0.0 | True | True |
| normal_brier_delta | -0.005501961443897266 | <= 0.0 | True | True |
| one_station_range_abs_error_delta | 0.018287061344348352 | <= 0.0 | False | True |
| normal_coverage1_absolute_change | 0.0146931046006461 | <= 0.01 | False | True |
| normal_coverage2_absolute_change | 0.0034978277821098303 | <= 0.01 | True | True |
| normal_all_absolute_bias_change | -0.011921796795917379 | <= 0.0 | True | False |
| normal_all_absolute_slope_error_change | -0.007493515796393235 | <= 0.0 | True | False |
| normal_noninput_absolute_bias_change | -0.020395501902103513 | <= 0.0 | True | False |
| normal_noninput_absolute_slope_error_change | -0.013438439894340903 | <= 0.0 | True | False |
