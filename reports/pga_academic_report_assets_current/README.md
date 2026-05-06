# PGA Report Assets

Generated assets directory: `reports/pga_academic_report_assets_current`

## Inputs

- `reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_best.txt`; npz: `reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_best.npz`
- `reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_last.txt`; npz: `reports/pga_report_inputs/weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets/eval_results_last.npz`

## Primary Model

`weights_japan_overfit_pga15_cross_attention_overfitall_fixed_inputs_random_targets` checkpoint `best`

## Main Figures

- `main_pga_metrics.png`: table of train/validation PGA metrics.
- `validation_metric_bars.png`: validation MAE/R2/slope comparison when multiple models are provided.
- `validation_metric_summary.png`: compact validation table when only one model is provided.
- `train_val_metric_bars.png`: train vs validation metrics.
- `val_mae_by_station_count.png`: validation MAE by input station count.
- `val_mae_by_epicentral_distance.png`: validation MAE by target epicentral distance when coordinate arrays are available.
- `scatter_*_val.png`: predicted vs true PGA for the primary model.
- `residual_vs_true_*_val.png`: residual diagnostics for the primary model.
- `pga_strength_bins_*_val.png`: performance by true PGA strength.
- `case_study_*.png`: event-level target, distance, and spatial residual diagnostics when coordinate arrays are available.
- `case_station_sweep_*.png`: event-level error change across requested input-station counts when eval npz contains `case_sweep_*` arrays.
- `loss_curve_*.png`: training and validation loss curve for the primary model.
- `literature_model_comparison.png`: positioning relative to TEAM and QuakeFormer.
