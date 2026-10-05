All physical PGA axes use log10(m/s²); errors/widths in dex, MSE in dex².

eqt_pretrained_frozen_normal_density_pit.png: validation fixed selected checkpoint seed42, noninput, normal, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.

eqt_pretrained_frozen_random_density_pit.png: validation fixed selected checkpoint seed42, noninput, random, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.

eqt_pretrained_frozen_calibration_counts.png: seed42; reliability pools raw bin counts over geometries, while coverage-width and counts retain geometry. Calibration is unmodified. Sources: reliability_bins.csv.gz, metrics_by_time_geometry_role.csv. Mean ± sigma coverage is a separate metric, not this quantile coverage.

phasenet_pretrained_frozen_normal_density_pit.png: validation fixed selected checkpoint seed42, noninput, normal, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.

phasenet_pretrained_frozen_random_density_pit.png: validation fixed selected checkpoint seed42, noninput, random, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.

phasenet_pretrained_frozen_calibration_counts.png: seed42; reliability pools raw bin counts over geometries, while coverage-width and counts retain geometry. Calibration is unmodified. Sources: reliability_bins.csv.gz, metrics_by_time_geometry_role.csv. Mean ± sigma coverage is a separate metric, not this quantile coverage.

team_original_scratch_normal_density_pit.png: validation fixed selected checkpoint seed42, noninput, normal, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.

team_original_scratch_random_density_pit.png: validation fixed selected checkpoint seed42, noninput, random, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.

team_original_scratch_calibration_counts.png: seed42; reliability pools raw bin counts over geometries, while coverage-width and counts retain geometry. Calibration is unmodified. Sources: reliability_bins.csv.gz, metrics_by_time_geometry_role.csv. Mean ± sigma coverage is a separate metric, not this quantile coverage.

level_shape_equal_distance.png: each eligible event field has >=5 targets; cells and fixed seeds displayed with equal weight. MSE components use the same field and satisfy level + shape = field MSE. Near-distance pairs use <=5 km epicentral radius difference; errors averaged within field before cells. Sources: field_summary.csv and event_fields_primary.csv.gz (noninput/untriggered subset; complete other-role fields in full evidence). These are spatial diagnostics; station repeatability requires the missing train-only reference.

input_and_untriggered_counts.png: deduplicated event/time/geometry decisions on left, repeated target rows on right. Original common population equal across nine systems. Sources: input_decision_counts.csv, metrics_by_time_geometry_role.csv. Neither denominator is a count of independent station samples.
