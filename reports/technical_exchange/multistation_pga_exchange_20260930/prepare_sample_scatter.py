#!/usr/bin/env python3
"""Reproduce the RT61 sample evidence figure from trusted formal exports.

This script produces verified chart assets only. The user approved native PPT
composition on 2026-09-30; these assets are embedded unchanged and proportionally.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.analyze_rt59_dual_objective_npz import (
    _point_metrics,
    _stack,
    _target_matrix,
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_export(path, expected_sha256):
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f'NPZ identity mismatch: {path}; actual={actual}')
    # These user-supplied archives are the formal exports already authenticated
    # in the RT61 report. Object arrays require allow_pickle for this format.
    with np.load(path, allow_pickle=True) as archive:
        truth = _target_matrix(_stack(archive, 'val_pga_label'))
        prediction = _target_matrix(_stack(archive, 'val_pga_mu_best'))
        valid = _stack(archive, 'val_pga_target_valid').astype(bool)
        valid = valid.reshape(truth.shape)
        target_type = _stack(archive, 'val_realtime_target_type').reshape(truth.shape)
        observed = _stack(archive, 'val_rt59_route_observed').astype(bool)
        observed = observed.reshape(truth.shape)
        event_ids = _stack(archive, 'val_event_id').reshape(truth.shape[0], -1)[:, 0]
    if not np.array_equal(observed, target_type == 0):
        raise ValueError('Observable route disagrees with formal input membership.')
    if not np.isfinite(truth[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise ValueError('Non-finite truth/prediction in valid target slots.')
    return {
        'truth': truth, 'prediction': prediction, 'valid': valid,
        'input': target_type == 0, 'event_ids': event_ids.astype(str),
        'sha256': actual,
    }


def contour_thresholds(histogram, fractions=(0.95, 0.80, 0.50)):
    ordered = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    return np.unique([
        ordered[min(np.searchsorted(cumulative, fraction), len(ordered) - 1)]
        for fraction in fractions
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-npz', type=Path, required=True)
    parser.add_argument('--normal-npz', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    deck = args.output_dir.resolve()
    figures = deck / 'assets' / 'figures'
    tables = deck / 'assets' / 'tables'
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(deck / '.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    from scipy.ndimage import gaussian_filter

    font_paths = [
        Path(os.environ.get('NATIVE_REPORT_FONT', str(Path.home() / '.fonts/noto/NotoSansCJKsc-Regular.otf'))),
        Path(os.environ.get('NATIVE_REPORT_FONT_BOLD', str(Path.home() / '.fonts/noto/NotoSansCJKsc-Bold.otf'))),
    ]
    if not font_paths[0].exists():
        raise FileNotFoundError('CJK font unavailable; supply a CJK font before plotting.')
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        'font.family': 'Noto Sans CJK SC',
        'axes.unicode_minus': False,
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

    report = ROOT / 'reports' / 'rt61_wave_geometry_validation_20260928'
    summary_path = report / 'summary.json'
    summary = json.loads(summary_path.read_text())
    inputs = summary['sources']
    random = load_export(args.random_npz, inputs['random_npz']['sha256'])
    normal = load_export(args.normal_npz, inputs['normal_npz']['sha256'])
    for protocol in ('random', 'normal'):
        metadata = inputs[f'{protocol}_metrics']
        if metadata['checkpoint']['epoch'] != 8 or not all(metadata['checks'].values()):
            raise ValueError(f'{protocol} formal provenance checks are incomplete.')
    rows = list(csv.DictReader((report / 'group_metrics.csv').open()))
    authoritative = {
        row['population']: row for row in rows if row['model'] == 'candidate_rt61'
    }
    populations = [
        ('random', 'Random geometry · 非输入目标', random, random['valid'], '#547D9D'),
        ('normal_all', 'Normal · 全部目标', normal, normal['valid'], '#547D9D'),
        ('normal_input', 'Normal · 输入台站目标', normal,
         normal['valid'] & normal['input'], '#4B9991'),
        ('normal_noninput', 'Normal · 非输入目标', normal,
         normal['valid'] & ~normal['input'], '#547D9D'),
    ]
    metrics_rows = []
    for name, _, data, mask, _ in populations:
        values = _point_metrics(data['truth'], data['prediction'], mask)
        expected = authoritative[name]
        for metric in ('targets', 'mae', 'rmse', 'r2', 'slope', 'bias', 'intercept'):
            if not np.isclose(values[metric], float(expected[metric]), atol=2e-9, rtol=1e-9):
                raise ValueError(f'{name} metric mismatch: {metric}')
        values.update({
            'population': name,
            'events': len(np.unique(data['event_ids'][mask.any(axis=1)])),
            'realtime_rows': int(mask.any(axis=1).sum()),
        })
        metrics_rows.append(values)

    all_values = np.concatenate([
        data[key][mask]
        for _, _, data, mask, _ in populations
        for key in ('truth', 'prediction')
    ])
    # Include all valid predictions and all truths. No extreme point is removed.
    axis_min = np.floor((all_values.min() - 0.07) * 10) / 10
    axis_max = np.ceil((all_values.max() + 0.07) * 10) / 10
    extent = (float(axis_min), float(axis_max))
    ticks = np.arange(np.ceil(axis_min * 2) / 2, axis_max + 0.01, 0.5)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.5))
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.083, top=0.943,
                        wspace=0.19, hspace=0.30)
    rng = np.random.default_rng(20260930)
    plotted_ranges = {}
    for ax, panel, metrics in zip(axes.flat, populations, metrics_rows):
        name, title, data, mask, color = panel
        truth = data['truth'][mask]
        prediction = data['prediction'][mask]
        order = rng.permutation(truth.size)
        # The same alpha, marker size and all-point policy apply to every panel.
        ax.scatter(truth[order], prediction[order], s=4.0, color=color,
                   alpha=0.08, linewidths=0, rasterized=True, zorder=2)
        hist, x_edges, y_edges = np.histogram2d(
            truth, prediction, bins=105, range=(extent, extent)
        )
        smooth = gaussian_filter(hist, sigma=1.3)
        centers_x = (x_edges[:-1] + x_edges[1:]) / 2
        centers_y = (y_edges[:-1] + y_edges[1:]) / 2
        ax.contour(centers_x, centers_y, smooth.T,
                   levels=contour_thresholds(smooth), colors=[color],
                   linewidths=0.85, alpha=0.70, zorder=3)
        ax.plot(extent, extent, color='#535E69', linestyle=(0, (5, 4)),
                linewidth=1.2, zorder=4)
        ax.plot(extent, metrics['slope'] * np.asarray(extent) + metrics['intercept'],
                color='#D66F63', linewidth=1.3, alpha=0.90, zorder=4)
        ax.set(xlim=extent, ylim=extent, aspect='equal',
               xticks=ticks, yticks=ticks,
               xlabel='真实 PGA  [log10(m/s²)]',
               ylabel='预测 PGA  [log10(m/s²)]')
        ax.set_title(title, loc='left', color='#303C49', pad=11, fontweight='bold')
        ax.text(0.04, 0.965,
                f"n = {metrics['targets']:,}\n"
                f"MAE = {metrics['mae']:.4f}    R² = {metrics['r2']:.4f}\n"
                f"slope = {metrics['slope']:.4f}",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=14.4, color='#35414E', linespacing=1.4,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.90, 'pad': 5},
                zorder=5)
        ax.grid(color='#E7EDF1', linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, color='#7C8D99')
        for spine in ax.spines.values():
            spine.set_color('#A8B7C3')
            spine.set_linewidth(0.8)
        plotted_ranges[name] = {
            'truth_min': float(truth.min()), 'truth_max': float(truth.max()),
            'prediction_min': float(prediction.min()),
            'prediction_max': float(prediction.max()),
            'points_outside_axes': 0,
        }
    image_path = figures / 'rt61_scatter_four_populations.png'
    fig.savefig(image_path, dpi=240)
    fig.savefig(figures / 'rt61_scatter_four_populations.pdf', dpi=240)
    fig.savefig(figures / 'rt61_scatter_four_populations.svg', dpi=240)
    plt.close(fig)
    metrics_path = tables / 'rt61_sample_panel_metrics.csv'
    with metrics_path.open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    provenance = {
        'repository': 'rular099/team_pytorch',
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'analysis_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'split': 'validation', 'epoch': 8, 'training_seed': 42,
        'coordinate': 'log10(m/s^2)',
        'point_estimate': 'predictive mixture mean in log10 PGA coordinate',
        'aggregation': 'target weighted; all valid target slots across realtime rows',
        'source_npz_sha256': {'random': random['sha256'], 'normal': normal['sha256']},
        'formal_summary_sha256': sha256(summary_path),
        'formal_group_metrics_sha256': sha256(report / 'group_metrics.csv'),
        'plot_script_sha256': sha256(__file__),
        'rendering': {
            'all_valid_points': True, 'subsampling': False, 'point_alpha': 0.08,
            'point_size_pt2': 4.0, 'draw_order_seed': 20260930,
            'axis_limits_both_dimensions': extent, 'equal_aspect': True,
            'density': '105x105 histogram, Gaussian sigma=1.3 bins',
            'contour_enclosed_probability': [0.50, 0.80, 0.95],
            'fit_line': 'OLS prediction versus truth; all valid targets',
        },
        'panels': metrics_rows, 'ranges': plotted_ranges,
        'figure_png_sha256': sha256(image_path),
    }
    (tables / 'rt61_sample_figure_provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n'
    )
    print(json.dumps({'figure': str(image_path), 'panels': metrics_rows,
                      'axis_limits': extent, 'checks': 'all formal metrics match'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
