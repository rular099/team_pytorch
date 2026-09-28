# RT61 wave-geometry residual conditioning validation

- `rt61_mechanism_pass`: **False** (13/14 gates pass)
- `useful_joint_progress`: **False**
- `legacy_full_go`: **False** (18/25 required gates pass)
- Recommendation: retain the RT59 same-forward reference as the development parent.
- Evidence: fixed epoch-8 Japan 2000-2024 validation only; no held-out test.
- Bootstrap: 5000 paired event-cluster draws, seed 20260915.

## RT61 mechanism gates

| Gate | Value | Rule | Pass |
|---|---:|---:|:---:|
| one_station_pairwise_delta_mae_ci_upper | -0.0007795662799108564 | < 0.0 | True |
| one_station_range_abs_error_ci_upper | -0.008023678607812073 | < 0.0 | True |
| random_all_range_abs_error_delta | -0.008220439248565496 | <= 0.0 | True |
| random_mae_delta | -0.0005045471802861412 | <= 0.0 | True |
| random_rmse_delta | -0.0011371547579903107 | <= 0.0 | True |
| normal_noninput_mae_delta | -5.813134901419548e-05 | <= 0.0 | True |
| normal_noninput_rmse_delta | 7.143701299616723e-05 | <= 0.0 | False |
| normal_input_max_abs_change | 0.0 | <= 1e-07 | True |
| random_nll_delta | -0.002849980742910707 | <= 0.0 | True |
| random_brier_delta | -0.0005571331004194657 | <= 0.0 | True |
| normal_all_nll_delta | -8.446827047081662e-05 | <= 0.0 | True |
| normal_all_brier_delta | -9.122179860833468e-05 | <= 0.0 | True |
| normal_noninput_nll_delta | -0.00029037357839306263 | <= 0.0 | True |
| normal_noninput_brier_delta | -0.00031354770212596583 | <= 0.0 | True |

## Useful joint progress gates

| Gate | Value | Rule | Pass |
|---|---:|---:|:---:|
| mechanism_pass | False | is True | False |
| one_station_pairwise_delta_mae_2pct_improvement | 0.33490778532430027 | <= 0.3294775649089464 | False |
| normal_noninput_mae_paired_ci_upper | 0.0003898143544613707 | < 0.0 | False |
| normal_noninput_rmse_paired_ci_upper | 0.000780013100324618 | < 0.0 | False |

## Legacy gates

| Gate | Value | Rule | Pass | Required |
|---|---:|---:|:---:|:---:|
| random_mae | 0.23732372496781431 | <= 0.242 | True | True |
| random_slope | 0.4113167520291103 | >= 0.45 | False | True |
| random_r2 | 0.39940791079965343 | >= 0.39 | True | True |
| random_range_ratio_ge5 | 0.48551286704129676 | >= 0.55 | False | True |
| random_one_station_range_ratio_ge5 | 0.19051350946423887 | >= 0.25 | False | True |
| random_one_station_pairwise_delta_mae | 0.33490778532430027 | <= 0.33 | False | True |
| random_requested1_mae | 0.24287509827966805 | <= 0.253 | True | True |
| random_nll | 0.1961679809523508 | <= 0.22 | True | True |
| random_delta_mae_ci_upper | -0.007131945283892239 | < 0.0 | True | True |
| random_delta_brier_ci_upper | -0.00548104924337238 | <= 0.0 | True | True |
| random_coverage1_absolute_change | 0.021400058159515623 | <= 0.01 | False | True |
| random_coverage2_absolute_change | 0.002062019192640263 | <= 0.01 | True | True |
| normal_all_delta_mae_ci_upper | -0.0055315108594729216 | < 0.0 | True | True |
| normal_all_delta_rmse_ci_upper | -0.00845881903598759 | < 0.0 | True | True |
| normal_noninput_delta_mae_ci_upper | -0.009870761085698508 | < 0.0 | True | True |
| normal_noninput_delta_rmse_ci_upper | -0.012276275865978327 | < 0.0 | True | True |
| normal_input_mae_delta | -0.003932266087546196 | <= 0.0 | True | True |
| normal_input_rmse_delta | -0.00677472735173415 | <= 0.0 | True | True |
| normal_all_mae_historical_redline | 0.12498944104184562 | <= 0.136 | True | True |
| normal_noninput_mae_historical_redline | 0.19924240896410886 | <= 0.218 | True | True |
| normal_nll_delta | -0.025997437853906735 | <= 0.0 | True | True |
| normal_brier_delta | -0.005644180719431943 | <= 0.0 | True | True |
| one_station_range_abs_error_delta | 0.01268971894918835 | <= 0.0 | False | True |
| normal_coverage1_absolute_change | 0.015094129441907134 | <= 0.01 | False | True |
| normal_coverage2_absolute_change | 0.0035312465188815922 | <= 0.01 | True | True |
| normal_all_absolute_bias_change | -0.01229680305243308 | <= 0.0 | True | False |
| normal_all_absolute_slope_error_change | -0.007729898161616511 | <= 0.0 | True | False |
| normal_noninput_absolute_bias_change | -0.021684384374762028 | <= 0.0 | True | False |
| normal_noninput_absolute_slope_error_change | -0.013627609206463664 | <= 0.0 | True | False |
