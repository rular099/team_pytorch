#!/usr/bin/env python
"""Create standalone SVGs from the portable FE02 tables; no raw data needed."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fe02-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['svg.hashsalt'] = 'fe02-seed42-validation-20261104'

ROOT = Path(__file__).resolve().parent
COLORS = dict(R0='#626b74', A='#b99038', B='#246ca7', C='#9c60a4', M='#ca5347')


def main():
    data, figures = ROOT / 'data', ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    curve = pd.read_csv(data / 'training_curves.csv')
    metrics = pd.read_csv(data / 'metrics_time_geometry_count_target.csv')
    endpoint = pd.read_csv(data / 'endpoints.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    for variant, color in COLORS.items():
        part = curve.loc[curve.variant_id.eq(variant)]
        axes[0, 0].plot(part.epoch, part.validation_selection, label=variant, color=color)
        for ax, protocol in [(axes[0, 1], 'normal'), (axes[1, 0], 'random')]:
            part = metrics.loc[metrics.variant_id.eq(variant) & metrics.target_group.eq('noninput') &
                               metrics.input_count_stratum.eq('all_counts') & metrics.geometry_protocol.eq(protocol)]
            ax.plot(part.elapsed_time, part.mae, marker='o', ms=3, color=color, label=variant)
        part = endpoint.loc[endpoint.variant_id.eq(variant) & endpoint.domain.eq('primary_1_3_5_10_20') &
                            endpoint.target_group.eq('noninput') & endpoint.geometry_protocol.eq('equal_normal_random')]
        ys = [part.loc[part.input_count_stratum.eq(k), 'equal_cell_mae'].iloc[0] for k in ['all_counts', 'single', 'multi']]
        axes[1, 1].plot([0, 1, 2], ys, marker='o', color=color, label=variant)
    axes[0, 0].set(title='Training selection (validation, 10 equal cells)', xlabel='Epoch', ylabel='Noninput MAE (dex)')
    axes[0, 0].legend(ncol=5, frameon=False, fontsize=8)
    for ax, title in [(axes[0, 1], 'Normal geometry'), (axes[1, 0], 'Random geometry')]:
        ax.set(title=title, xlabel='Elapsed seconds (log scale)', ylabel='Noninput MAE (dex)', xscale='log')
        ax.set_xticks([1, 3, 5, 10, 20, 40, 90], labels=['1', '3', '5', '10', '20', '40*', '90*'])
    axes[1, 1].set(title='Common 1/3/5/10/20s, equal-cell endpoint', ylabel='Noninput MAE (dex)')
    axes[1, 1].set_xticks([0, 1, 2], labels=['All input counts', 'Single (exploratory)', 'Multi (exploratory)'])
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.suptitle('FE02 seed42 | selected validation checkpoints | log10(m/s²)\n*40/90s diagnostic; single normal late cells have only 6–8 events', fontsize=12)
    fig.savefig(figures / 'validation_accuracy.svg', metadata={'Date': None})
    plt.close(fig)

    fields = pd.read_csv(data / 'spatial_summary_by_cell.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    primary = endpoint.loc[endpoint.domain.eq('primary_1_3_5_10_20') & endpoint.target_group.eq('noninput') &
                           endpoint.input_count_stratum.eq('all_counts') & endpoint.geometry_protocol.eq('equal_normal_random')]
    for x, variant in enumerate(COLORS):
        row = primary.loc[primary.variant_id.eq(variant)].iloc[0]
        axes[0].scatter(row.equal_cell_mae, row.equal_cell_covered95, s=55, c=COLORS[variant])
        axes[0].annotate(variant, (row.equal_cell_mae, row.equal_cell_covered95), xytext=(4, 3), textcoords='offset points')
        axes[1].bar(x, row.equal_cell_nll, color=COLORS[variant])
        part = fields.loc[fields.variant_id.eq(variant) & fields.input_count_stratum.eq('single') & fields.elapsed_time.isin([1,3,5,10,20])]
        # Field macro averages, then existing cells equally weighted. Denominators
        # are in spatial_summary_by_cell.csv; this is exploratory, not primary.
        axes[2].bar(x, part.pairwise_delta_mae_mean.mean(), color=COLORS[variant])
    axes[0].axhline(.95, ls='--', lw=1, color='gray')
    axes[0].set(xlabel='Noninput equal-cell MAE (dex)', ylabel='Exact MDN 95% coverage', title='Point accuracy and calibration')
    axes[1].set(title='Noninput equal-cell NLL', ylabel='NLL (log-density coordinate)')
    axes[2].set(title='Single-input spatial error', ylabel='Pairwise-delta MAE (dex)')
    for ax in axes[1:]:
        ax.set_xticks(range(5), labels=list(COLORS))
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('FE02 seed42 validation | spatial fields require ≥5 noninput targets', fontsize=12)
    fig.savefig(figures / 'calibration_and_spatial.svg', metadata={'Date': None})
    plt.close(fig)
    for path in figures.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


if __name__ == '__main__':
    main()
