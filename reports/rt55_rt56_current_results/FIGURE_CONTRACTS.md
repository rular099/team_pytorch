# RT55/RT56 comprehensive-results figure contracts

Backend: Python (NumPy, Pandas, SciPy, Matplotlib and Seaborn only).

Export contract: editable SVG and PDF are primary; 300 dpi PNG is the preview.
Every quantitative figure has an adjacent CSV source-data table. Text is kept
editable in SVG/PDF, panels use a colour-blind-safe palette plus line/marker
encoding, and all uncertainty intervals are clustered at event level.

## RT55 standalone report

Core conclusion: RT55 generalizes consistently under its original normal
protocol, but its headline score is dominated by easy input-station targets;
non-input, untriggered and long-lead targets reveal the operational difficulty.

Figure archetype: quantitative grid with spatial case-study plates.

Evidence hierarchy:

1. Hero evidence: epoch-20 validation/test agreement and epoch-32 validation,
   reported for all targets and non-input targets separately.
2. Validation evidence: predicted-versus-true density, residual structure,
   magnitude/station/time/lead/distance strata and probabilistic calibration.
3. Control: epoch-32 waveform-station roll evaluation.
4. Representative evidence: deterministically selected typical, strong-motion,
   non-input-success and failure cases. Selection rules and all candidates are
   exported; examples are not manually cherry-picked.

Reviewer risks:

- The only complete held-out test checkpoint is epoch 20, despite its source
  directory name containing `ep32_retry`; it must never be labelled epoch 32.
- The split is event-disjoint within each year, not held-out-year.
- Only seed 42 is available; event bootstrap does not represent seed variance.
- Target-level observations from the same event/time are correlated, so both
  target-weighted and event-balanced summaries are required.

## RT56 adaptation report

Core conclusion: mixed-random fine-tuning makes the model exploit additional
causal stations and repairs random-geometry uncertainty calibration, but the
gain is uneven across lead time/PGA strength and causes measurable normal-task
forgetting.

Figure archetype: paired quantitative grid with zero-shot/fine-tuned spatial
case-study plates.

Evidence hierarchy:

1. Hero evidence: strictly paired RT55 epoch-32 zero-shot random geometry versus
   RT56 epoch-6 fine-tuned random geometry on identical samples and targets.
2. Validation evidence: requested/available/selected geometry, station-count,
   time, target-type, lead-time, magnitude and strong/weak-PGA strata.
3. Control: RT55 epoch-32 normal versus RT56 epoch-6 normal, with non-input-only
   results elevated to a primary analysis.
4. Representative evidence: deterministic largest/typical improvement,
   degradation, strong-PGA recovery, weak-PGA overprediction, unknown-lead gain
   and positive-lead degradation cases.

Reviewer risks:

- RT55 normal and RT56 random totals are not task-equivalent. Cross-protocol
  comparisons are contextual references only; zero-shot/fine-tuned random is
  the strict adaptation comparison.
- Random geometry has no held-out test yet and is reported as validation only.
- The index arrays used by input and target exports do not share a reliable
  namespace. Exclusion is audited geometrically: no valid random target may
  share the coordinates of a valid selected input station.
- Unknown target lead time is common and must be reported separately.

## Shared statistics

- PGA coordinate: log10(m/s^2).
- Strong-motion threshold: -1.2 log10(m/s^2).
- Point metrics: MAE, RMSE, R2, correlation, slope, intercept and bias.
- Distribution metrics: NLL, predictive sigma, 1-sigma/2-sigma coverage,
  standardized residuals and empirical coverage curves.
- Warning metrics: Brier score/skill, ROC-AUC, PR-AUC and calibration error.
- Error tails: median, P90 and P95 absolute error; fractions above 0.3/0.5 dex.
- Magnitude bins: [3.0,3.5), [3.5,4.0), [4.0,4.5), [4.5,5.0),
  [5.0,6.0), and M>=6.0, with event and target counts shown.
- Confidence intervals: deterministic event-cluster bootstrap, seed 42, with
  the number of bootstrap replicates recorded in provenance.
