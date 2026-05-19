# Stage2 512 P3 Residual Diagnostics

Date: 2026-05-19

This report uses current local `chaosuan_res` eval artifacts. It compares b29-last, the user-approved strong-PGA reference, against b43-last, the new balanced structural anchor.

## Data Availability

- Available now: eval npz predictions/labels, target coordinates, event indices, station counts, train logs, and diag csv files.
- Not available in local `chaosuan_res`: `.pth/.pt` checkpoints, so learned gate parameter values cannot be read from current local files.
- Code has been updated after this analysis so future runs log readout gate values
  into `diag_*.csv`, e.g. `diag_pga_cross_extra_layer1_attn_gate.csv` and
  `diag_pga_cross_extra_attn_gate_mean.csv`.
- Site residual uses rounded target coordinates as a station/site proxy because target station ids are not stored in the eval npz.

## Model Summary

| Model | MAE | Bias | Slope | Pred std | Event residual MAE | Site-proxy residual MAE | Residual-label corr | Residual-distance corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b29_last | 0.4252 | 0.3007 | 0.5383 | 0.4732 | 0.3384 | 0.3761 | -0.5947 | 0.3088 |
| b43_last | 0.2966 | 0.0421 | 0.5548 | 0.4351 | 0.1300 | 0.2390 | -0.6531 | 0.3082 |

Interpretation: b43 greatly improves global calibration and event/site residual magnitudes relative to b29-last, but still has negative residual-label correlation, meaning high-PGA targets remain underpredicted.

## Strong/Weak Label Bins

| Model | Label bin | N | Label mean | Pred mean | MAE | Bias |
|---|---|---:|---:|---:|---:|---:|
| b29_last | [-1.4,-1.2) | 101 | -1.3006 | -1.0495 | 0.3303 | 0.2510 |
| b29_last | [-1.2,-1.0) | 60 | -1.1166 | -1.1459 | 0.2679 | -0.0293 |
| b29_last | [-1.0,-0.6) | 64 | -0.8466 | -0.9625 | 0.2207 | -0.1160 |
| b29_last | >=-0.6 | 24 | -0.3887 | -0.8199 | 0.4497 | -0.4311 |
| b43_last | [-1.4,-1.2) | 101 | -1.3006 | -1.3519 | 0.2192 | -0.0513 |
| b43_last | [-1.2,-1.0) | 60 | -1.1166 | -1.3413 | 0.2952 | -0.2247 |
| b43_last | [-1.0,-0.6) | 64 | -0.8466 | -1.1767 | 0.3603 | -0.3302 |
| b43_last | >=-0.6 | 24 | -0.3887 | -1.1173 | 0.7286 | -0.7286 |

Interpretation: b29-last is much more aggressive on high labels, while b43 trades strong-PGA recall for much better global behavior. b48-b52 should test whether b43 can recover part of b29-last strong-PGA correction without losing train fit.

## Input Count and Distance Effects

Detailed bins are in `stage2_512_p3_context_bins.csv`. Use them to check whether residuals concentrate at low station count or larger target-event distance. Distances are computed from the first two local coordinate dimensions, so treat them as relative geometry bins rather than exact kilometers.

## Station Context Collapse

| Model | Series | First | Last | Mean last10 | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| b41_clean_stationctx | diag_pga_mu_best_valid_std | 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0023 |
| b41_clean_stationctx | diag_pga_point_mu_std | 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0023 |
| b41_clean_stationctx | diag_station_context_delta_norm | 26.6058 | 34.3768 | 34.3882 | 26.6058 | 39.2226 |
| b41_clean_stationctx | diag_station_context_cosine_mean | 0.7801 | 1.0000 | 1.0000 | 0.7801 | 1.0000 |
| b47_strong_stationctx | diag_pga_mu_best_valid_std | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.1233 |
| b47_strong_stationctx | diag_pga_point_mu_std | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.1247 |
| b47_strong_stationctx | diag_station_context_delta_norm | 26.4274 | 36.0040 | 35.9609 | 26.4274 | 41.1113 |
| b47_strong_stationctx | diag_station_context_cosine_mean | 0.7832 | 1.0000 | 1.0000 | 0.7832 | 1.0000 |
| b43_stable_no_stationctx | diag_pga_mu_best_valid_std | 0.0025 | 0.5699 | 0.5767 | 0.0025 | 0.6737 |
| b43_stable_no_stationctx | diag_pga_point_mu_std | 0.0026 | 0.6013 | 0.6314 | 0.0026 | 0.7745 |

Interpretation: b41/b47 collapse is visible directly in PGA prediction std approaching zero. The station-context tensors remain nonzero, but the readout/head maps them to nearly constant PGA. This supports adding an explicit gated/identity station-context bypass before using this path as a RoPE carrier.

## Output Files

- `stage2_512_p3_model_summary.csv`
- `stage2_512_p3_label_bins.csv`
- `stage2_512_p3_prediction_bins.csv`
- `stage2_512_p3_context_bins.csv`
- `stage2_512_p3_station_context_diag.csv`
