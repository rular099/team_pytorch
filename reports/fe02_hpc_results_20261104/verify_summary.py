#!/usr/bin/env python
"""Reproduce headline and all 63 bootstrap intervals from compact event tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / 'data'


def main():
    cells = pd.read_csv(ROOT / 'primary_event_cells_noninput.csv', dtype={'event_id': str})
    headline = pd.read_csv(ROOT / 'headline_primary.csv')
    paired = pd.read_csv(ROOT / 'paired_event_bootstrap.csv')
    assert cells.groupby('event_id').dataset_id.nunique().max() == 1
    for row in headline.itertuples():
        part = cells.loc[cells.variant_id.eq(row.variant_id)]
        aggregated = part.groupby(['geometry_protocol', 'elapsed_time'])[['sum_abs_error', 'targets']].sum()
        value = np.mean(aggregated.sum_abs_error / aggregated.targets)
        np.testing.assert_allclose(value, row.common_equal_time_protocol_noninput_mae, atol=1e-11, rtol=0)
        assert part.targets.sum() == row.primary_targets
    max_error = 0.
    for row in paired.itertuples():
        left = cells.loc[cells.variant_id.eq(row.baseline)]
        right = cells.loc[cells.variant_id.eq(row.competitor)]
        if row.input_count_stratum != 'all_counts':
            left = left.loc[left.input_count_stratum.eq(row.input_count_stratum)]
            right = right.loc[right.input_count_stratum.eq(row.input_count_stratum)]
        if row.geometry_protocol != 'equal_normal_random':
            left = left.loc[left.geometry_protocol.eq(row.geometry_protocol)]
            right = right.loc[right.geometry_protocol.eq(row.geometry_protocol)]
        keys = ['dataset_id', 'event_id', 'geometry_protocol', 'elapsed_time', 'input_count_stratum']
        joined = left.merge(right, on=keys, validate='one_to_one', suffixes=('_left', '_right'))
        assert len(joined) == len(left) == len(right)
        assert (joined.targets_left == joined.targets_right).all()
        joined['delta_sum'] = joined.sum_abs_error_right - joined.sum_abs_error_left
        sums = joined.pivot(index='event_id', columns=['geometry_protocol', 'elapsed_time'], values='delta_sum').fillna(0)
        counts = joined.pivot(index='event_id', columns=['geometry_protocol', 'elapsed_time'], values='targets_left').fillna(0)
        assert len(sums) == row.events and counts.to_numpy().sum() == row.targets
        a, n = sums.to_numpy(), counts.to_numpy()
        expected = np.mean(a.sum(0) / n.sum(0))
        assert abs(expected - row.equal_time_protocol_delta_mae) < 1e-11
        rng = np.random.default_rng(20261008)
        values, pooled, empty = [], [], 0
        for _ in range(5000):
            ids = rng.integers(0, len(a), len(a))
            numerator, denominator = a[ids].sum(0), n[ids].sum(0)
            pooled.append(numerator.sum()/denominator.sum())
            if (denominator > 0).all():
                values.append(np.mean(numerator/denominator))
            else:
                empty += 1
        actual = np.quantile(values, [.025, .975])
        reference = np.array([row.equal_time_protocol_ci_low, row.equal_time_protocol_ci_high])
        max_error = max(max_error, float(np.abs(actual-reference).max()))
        np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=0)
        np.testing.assert_allclose(np.quantile(pooled, [.025, .975]), [row.ci_low, row.ci_high], atol=1e-11, rtol=0)
        assert empty == row.empty_cell_bootstrap_replicates
    result = dict(status='PASS', headline_rows=len(headline), bootstrap_rows=len(paired),
                  draws_per_row=5000, seed=20261008, max_ci_abs_difference=max_error,
                  scope='reconstructed from portable event/cell error sums; raw predictions not required')
    (ROOT / 'compact_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
