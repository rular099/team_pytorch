# V01 validation closure

Validation only; fixed epoch8/step1496 weights. No training, preflight, or held-out test.

Read metrics_by_stratum.csv with coverage_and_denominators.json and the outer-match audit. A/V common-available comparisons are conditional and carry selection bias; common remote is a paired-sensor exclusion proxy, not certified physical-site separation.

MSE = bias squared + centered error variance is recomputed from saved predictions with event-cluster CIs; no bias correction is applied. Above-threshold means target PGA >= -1.2 log10(m/s^2), not event magnitude.

Historical np.ptp ratio median and new P95-P05 ratio mean are different quantities in separate files. Density/residual SVGs use fixed axes and audited bin counts; out-of-view points are reported.

AA random closure: complete.
