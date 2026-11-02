# Figure contract

- Core conclusion: Existing validation evidence is limited to 23 catalog-M=3.0
  events; RT59 PGA errors are larger than the all-magnitude baseline, and the
  sparse non-factorial station strata do not support a stable time/station gain.
- Figure archetype: quantitative grid.
- Target/output: compact technical report; editable SVG plus PDF and PNG preview.
- Backend: Python only (matplotlib).
- Final size: 7.2 × 6.2 inches (approximately 183 × 157 mm).
- Panel map:
  - a: joint time/station MAE.
  - b: joint time/station RMSE.
  - c: MAE trajectory and event-cluster 95% interval; low-support cells shown
    as hollow unconnected markers.
  - d: unique events and targets supporting every joint cell.
- Evidence hierarchy:
  - Hero evidence: MAE/RMSE joint grids and support grid.
  - Validation evidence: event-cluster intervals for adequately supported
    one-station cells.
  - Robustness warning: hatching/dagger for cells with fewer than 10 events.
- Statistics: target-level MAE/RMSE; 5,000 event-ID cluster bootstrap draws;
  one training seed.
- Source data: `joint_strata.csv`, `time_strata.csv`, and
  `station_strata.csv`.
- Reviewer risks: only 23 events, no M<3.0 events, almost no multi-station
  coverage, reused development validation, and no independent seed variance.
