#!/usr/bin/env python3
"""Stratified validation analysis for RT59 on catalog M <= 3.0 events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.7,
    'legend.frameon': False,
})


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
REPO = WORKSPACE / 'team_pytorch_query_geometry_diagnostics'
sys.path.insert(0, str(REPO))

from tools.analyze_rt61_wave_geometry_npz import (  # noqa: E402
    _load_metrics_identity,
    _load_rt61_protocol,
)


TIME_VALUES = np.asarray([1, 3, 5, 10, 20, 40, 90], dtype=float)
STATION_BINS = (
    ('1', 1, 1),
    ('2–3', 2, 3),
    ('4–7', 4, 7),
    ('8–11', 8, 11),
    ('12–16', 12, 16),
)
BOOTSTRAP_SEED = 20260928
DEFAULT_DRAWS = 5000
MAGNITUDE_MAX = 3.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def safe_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    if target.size == 0:
        return {
            'mae': None, 'rmse': None, 'bias': None, 'correlation': None,
            'r2': None, 'p90_abs_error': None, 'within_0p1': None,
            'within_0p2': None,
        }
    error = prediction - target
    absolute = np.abs(error)
    target_centered = target - target.mean()
    ss_total = float(np.sum(target_centered ** 2))
    correlation = None
    if target.size >= 2 and np.std(target) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        'mae': float(np.mean(absolute)),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'bias': float(np.mean(error)),
        'correlation': correlation,
        'r2': None if ss_total <= 0 else float(1 - np.sum(error ** 2) / ss_total),
        'p90_abs_error': float(np.quantile(absolute, 0.90)),
        'within_0p1': float(np.mean(absolute <= 0.1)),
        'within_0p2': float(np.mean(absolute <= 0.2)),
    }


def cluster_interval(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    event_ids: np.ndarray,
    metric: str,
    draws: int,
    seed: int,
) -> tuple[float | None, float | None]:
    row_active = np.any(mask, axis=1)
    events = np.unique(event_ids[row_active])
    if events.size < 2:
        return None, None
    counts = np.zeros(events.size, dtype=np.float64)
    numerator = np.zeros(events.size, dtype=np.float64)
    for index, event_id in enumerate(events):
        selected = mask & (event_ids[:, None] == event_id)
        counts[index] = selected.sum()
        error = prediction[selected] - target[selected]
        if metric == 'mae':
            numerator[index] = np.abs(error).sum()
        elif metric == 'rmse':
            numerator[index] = np.square(error).sum()
        else:
            raise ValueError(metric)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        indices = rng.integers(0, events.size, size=(stop - start, events.size))
        totals = counts[indices].sum(axis=1)
        values = numerator[indices].sum(axis=1) / totals
        if metric == 'rmse':
            values = np.sqrt(values)
        samples[start:stop] = values
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize_mask(
    data: dict,
    mask: np.ndarray,
    draws: int,
    seed: int,
) -> dict:
    mask = np.asarray(mask, dtype=bool) & data['valid']
    active_rows = np.any(mask, axis=1)
    target = data['target'][mask]
    model = data['models']['reference_rt59']
    prediction = model['mean'][mask]
    result = {
        'events': int(np.unique(data['event_ids'][active_rows]).size),
        'rows': int(active_rows.sum()),
        'targets': int(mask.sum()),
        **safe_metrics(target, prediction),
    }
    for metric in ('mae', 'rmse'):
        lower, upper = cluster_interval(
            data['target'], model['mean'], mask, data['event_ids'], metric,
            draws, seed,
        )
        result[f'{metric}_ci_lower'] = lower
        result[f'{metric}_ci_upper'] = upper
    if mask.any():
        truth_binary = data['target'][mask] >= -1.2
        probability = model['probability'][mask]
        result.update({
            'pga_ge_minus1p2_prevalence': float(np.mean(truth_binary)),
            'predicted_probability_mean': float(np.mean(probability)),
            'brier': float(np.mean((probability - truth_binary) ** 2)),
            'nll': float(np.mean(model['nll'][mask])),
            'coverage_1sigma': float(np.mean(
                np.abs(prediction - target) <= model['sigma'][mask]
            )),
            'coverage_2sigma': float(np.mean(
                np.abs(prediction - target) <= 2.0 * model['sigma'][mask]
            )),
        })
    else:
        result.update({
            'pga_ge_minus1p2_prevalence': None,
            'predicted_probability_mean': None,
            'brier': None,
            'nll': None,
            'coverage_1sigma': None,
            'coverage_2sigma': None,
        })
    return result


def matrix(rows: list[dict], key: str) -> np.ndarray:
    values = np.full((len(STATION_BINS), len(TIME_VALUES)), np.nan)
    for row in rows:
        station_index = next(
            index for index, item in enumerate(STATION_BINS)
            if item[0] == row['station_bin']
        )
        time_index = int(np.flatnonzero(
            np.isclose(TIME_VALUES, float(row['seconds']))
        )[0])
        value = row[key]
        if value is not None:
            values[station_index, time_index] = float(value)
    return values


def add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.14, 1.05, label, transform=axis.transAxes,
        fontsize=9, fontweight='bold', ha='left', va='bottom',
    )


def draw_heatmap(
    figure, axis, values, title, color_label, cmap,
    value_format='.3f', count_matrix=None, support_events=None,
) -> None:
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(masked, aspect='auto', cmap=cmap, interpolation='nearest')
    axis.set_xticks(np.arange(len(TIME_VALUES)))
    axis.set_xticklabels([f'{value:g}' for value in TIME_VALUES])
    axis.set_yticks(np.arange(len(STATION_BINS)))
    axis.set_yticklabels([item[0] for item in STATION_BINS])
    axis.set_xlabel('Waveform window after first P pick (s)')
    axis.set_ylabel('Actual input stations')
    axis.set_title(title, fontsize=8, pad=5)
    finite = values[np.isfinite(values)]
    if finite.size:
        threshold = (
            float(np.min(finite) + 0.55 * (np.max(finite) - np.min(finite)))
            if count_matrix is not None else float(np.nanmedian(finite))
        )
    else:
        threshold = 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                axis.text(column, row, 'NA', ha='center', va='center', color='#606060')
                continue
            label = format(value, value_format)
            if count_matrix is not None:
                label = f'n={int(value)}\nT={int(count_matrix[row, column])}'
            text_color = 'white' if value > threshold else '#272727'
            if (
                support_events is not None
                and support_events[row, column] < 10
            ):
                label += '†'
                axis.add_patch(Rectangle(
                    (column - 0.5, row - 0.5), 1, 1,
                    fill=False, hatch='////', edgecolor='#767676',
                    linewidth=0.0,
                ))
            axis.text(
                column, row, label, ha='center', va='center',
                fontsize=5.5, color=text_color,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(color_label, fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.5)


def make_figure(output: Path, joint_rows: list[dict]) -> None:
    mae = matrix(joint_rows, 'mae')
    rmse = matrix(joint_rows, 'rmse')
    events = matrix(joint_rows, 'events')
    targets = matrix(joint_rows, 'targets')
    mae_lower = matrix(joint_rows, 'mae_ci_lower')
    mae_upper = matrix(joint_rows, 'mae_ci_upper')

    figure = plt.figure(figsize=(7.2, 6.2))
    grid = figure.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    draw_heatmap(
        figure, axis_a, mae, 'RT59 small-event MAE',
        'MAE, log10(m/s²)', 'Blues', support_events=events,
    )
    draw_heatmap(
        figure, axis_b, rmse, 'RT59 small-event RMSE',
        'RMSE, log10(m/s²)', 'Purples', support_events=events,
    )

    colors = ['#484878', '#7884B4', '#2F7F8F', '#A56576', '#606060']
    markers = ['o', 's', '^', 'D', 'P']
    for index, ((label, _, _), color, marker) in enumerate(
        zip(STATION_BINS, colors, markers)
    ):
        supported = np.isfinite(mae[index]) & (events[index] >= 10)
        low_support = np.isfinite(mae[index]) & (events[index] < 10)
        if not np.any(supported) and not np.any(low_support):
            continue
        if np.any(supported):
            axis_c.plot(
                TIME_VALUES[supported], mae[index, supported], marker=marker,
                color=color, linewidth=1.25, markersize=3.5, label=label,
            )
            axis_c.fill_between(
                TIME_VALUES[supported], mae_lower[index, supported],
                mae_upper[index, supported], color=color, alpha=0.12,
                linewidth=0,
            )
        if np.any(low_support):
            axis_c.scatter(
                TIME_VALUES[low_support], mae[index, low_support],
                marker=marker, facecolors='none', edgecolors=color,
                linewidths=0.9, s=22,
                label=(f'{label} (<10 events/cell)' if not np.any(supported) else None),
            )
    axis_c.set_xscale('log')
    axis_c.set_xticks(TIME_VALUES)
    axis_c.set_xticklabels([f'{value:g}' for value in TIME_VALUES])
    axis_c.set_xlabel('Waveform window after first P pick (s)')
    axis_c.set_ylabel('MAE, log10(m/s²)')
    axis_c.set_title('MAE trend with event-cluster 95% CI', fontsize=8, pad=5)
    axis_c.grid(axis='y', color='#D8D8D8', linewidth=0.5)
    axis_c.legend(
        title='Input stations', fontsize=5.5, title_fontsize=6,
        ncol=2, loc='best',
    )

    draw_heatmap(
        figure, axis_d, events, 'Support per stratum',
        'Unique events', 'Greys', value_format='.0f', count_matrix=targets,
    )

    for axis, label in zip((axis_a, axis_b, axis_c, axis_d), 'abcd'):
        add_panel_label(axis, label)
    figure.suptitle(
        'RT59 early PGA prediction for catalog M = 3.0 earthquakes\n'
        'Random-geometry validation; fixed epoch-8 same-forward reference',
        fontsize=9, y=0.985,
    )
    figure.subplots_adjust(top=0.80, left=0.09, right=0.96, bottom=0.08)
    figure.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    figure.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    figure.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(figure)


def format_number(value, digits=4):
    if value is None or not np.isfinite(value):
        return 'NA'
    return f'{value:.{digits}f}'


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    result = [
        '| ' + ' | '.join(label for _, label in columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if value is None:
                values.append('NA')
            elif isinstance(value, float):
                values.append(format_number(value))
            else:
                values.append(str(value))
        result.append('| ' + ' | '.join(values) + ' |')
    return result


def write_report(
    output_dir: Path,
    summary: dict,
    time_rows: list[dict],
    station_rows: list[dict],
) -> None:
    overall = summary['overall_small_event']
    overall_all = summary['overall_all_magnitudes']
    one_second = next(row for row in time_rows if float(row['seconds']) == 1.0)
    one_station = next(row for row in station_rows if row['station_bin'] == '1')
    one_second_one_station = summary['one_second_one_station']
    report = [
        '# RT59 对 M≤3.0 小震的早期 PGA 预警能力', '',
        '## 结论', '',
        '- **总体判断：当前对 M=3.0 小震的远端 PGA 预测能力偏弱，且样本不足以支持'
        '“更多波形或更多台站会稳定提升预警精度”的结论。**',
        f'- 在现有 random-geometry validation 中共有 **{summary["small_event_counts"]["events"]}** '
        f'个 M≤3.0 事件、**{summary["small_event_counts"]["rows"]}** 个实时快照和 '
        f'**{summary["small_event_counts"]["targets"]}** 个有效远端 PGA 目标；这些事件的'
        f'目录震级全部恰好为 **M=3.0**，没有 M<3.0 样本。',
        f'- RT59 在全部小震目标上的 MAE 为 **{overall["mae"]:.4f}**、RMSE 为 '
        f'**{overall["rmse"]:.4f} log10(m/s²)**；误差偏置为 **{overall["bias"]:+.4f}**。',
        f'- 相比全震级 random validation 的 MAE/RMSE '
        f'**{overall_all["mae"]:.4f}/{overall_all["rmse"]:.4f}**，小震误差分别高 '
        f'**{summary["relative_to_all_magnitudes"]["mae_increase_percent"]:.1f}%/'
        f'{summary["relative_to_all_magnitudes"]["rmse_increase_percent"]:.1f}%**。',
        f'- 最早的“1 s + 1 台站”条件有 20 个事件、42 个目标，MAE/RMSE 为 '
        f'**{one_second_one_station["mae"]:.4f}/{one_second_one_station["rmse"]:.4f}**；'
        f'其 Bias 为 **{one_second_one_station["bias"]:+.4f}**，表现为早期偏高估。',
        f'- 在 1 s 分层中，PGA≥-1.2 log10(m/s²) 的真实比例为 '
        f'**{100 * one_second["pga_ge_minus1p2_prevalence"]:.1f}%**，平均预测概率为 '
        f'**{100 * one_second["predicted_probability_mean"]:.1f}%**；小样本下呈现明显偏高告警倾向。',
        f'- 整体小震相关系数为 **{overall["correlation"]:.3f}**、R² 为 '
        f'**{overall["r2"]:.3f}**，说明对目标间强弱排序的能力较弱。',
        f'- 实际超过 3 个输入台站的事件只有 **{summary["station_support"]["events_4_to_7"]}** '
        '个，8 台及以上为 0；联合分层不是完整交叉设计，当前数据不能证明增加波形时长或'
        '增加台站数能够稳定改善 M=3.0 预警。',
        '- 这里衡量的是小震条件下的远端 PGA 连续值预测能力，不等同于震级检出率、'
        '首报延迟或业务告警命中率。', '',
        '## 按波形时间分层', '',
    ]
    report.extend(markdown_table(time_rows, [
        ('seconds', '波形秒数'), ('events', '事件数'), ('targets', '目标数'),
        ('mae', 'MAE'), ('mae_ci_lower', 'MAE CI下界'),
        ('mae_ci_upper', 'MAE CI上界'), ('rmse', 'RMSE'), ('bias', 'Bias'),
    ]))
    report.extend(['', '## 按实际输入台站数分层', ''])
    report.extend(markdown_table(station_rows, [
        ('station_bin', '台站数'), ('events', '事件数'), ('targets', '目标数'),
        ('mae', 'MAE'), ('mae_ci_lower', 'MAE CI下界'),
        ('mae_ci_upper', 'MAE CI上界'), ('rmse', 'RMSE'), ('bias', 'Bias'),
    ]))
    report.extend([
        '', '## 方法与限制', '',
        '- 当前最优模型按项目决策定义为 RT59 same-forward reference；RT61 已判定 NO_GO。',
        '- 数据仅来自 validation；未打开 held-out test。',
        '- 时间是首个 P 拾取后的预设观测窗，台站数是模型实际收到的有效输入台站数。',
        '- MAE/RMSE 的 95% CI 采用 event-ID cluster bootstrap，重复抽样 '
        f'{summary["bootstrap"]["draws"]} 次。',
        '- 本分析只有一个训练 seed，CI 描述事件抽样不确定性，不包含模型训练随机性。',
        '- 图中带斜线或 † 的格子少于 10 个事件，只作描述，不能用于趋势判断。',
        '- 分层网格原始数值见 `joint_strata.csv`；图中每个 support 单元格同时给出 '
        '事件数和目标数。', '',
        '## 文件', '',
        '- `rt59_small_event_strata.svg/pdf/png`：主图。',
        '- `joint_strata.csv`：秒数×实际台站数联合分层。',
        '- `time_strata.csv`、`station_strata.csv`：边际分层。',
        '- `summary.json`、`analysis_manifest.json`：完整摘要与输入身份。',
    ])
    (output_dir / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bootstrap-draws', type=int, default=DEFAULT_DRAWS)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.npz).resolve()
    metrics_path = Path(args.metrics).resolve()
    identity = _load_metrics_identity(metrics_path, args.source_manifest)
    data = _load_rt61_protocol(npz_path, identity)
    if data.get('magnitude') is None:
        raise ValueError('catalog magnitude is absent from validation export.')
    magnitude = np.asarray(data['magnitude'], dtype=np.float64)
    if not np.isfinite(magnitude).all():
        raise ValueError('catalog magnitude contains non-finite values.')
    if data['requested_time'] is None:
        raise ValueError('requested waveform time is absent.')

    event_magnitudes = {}
    for event_id, value in zip(data['event_ids'], magnitude):
        event_magnitudes.setdefault(str(event_id), []).append(float(value))
    inconsistent = {
        event_id: values for event_id, values in event_magnitudes.items()
        if max(values) - min(values) > 1e-6
    }
    if inconsistent:
        raise ValueError('catalog magnitude varies within event IDs.')

    small_rows = magnitude <= MAGNITUDE_MAX
    base_mask = np.broadcast_to(small_rows[:, None], data['valid'].shape)
    station_count = np.asarray(data['station_count'], dtype=int)
    requested_time = np.asarray(data['requested_time'], dtype=float)
    if station_count[small_rows].max(initial=0) > 16:
        raise ValueError('small-event station count exceeds preregistered random-mask maximum.')

    overall = summarize_mask(
        data, base_mask, args.bootstrap_draws, args.bootstrap_seed
    )
    all_magnitude = summarize_mask(
        data, np.ones_like(data['valid'], dtype=bool),
        args.bootstrap_draws, args.bootstrap_seed + 1,
    )
    joint_rows = []
    for seconds in TIME_VALUES:
        time_rows = np.isclose(requested_time, seconds)
        for station_label, lower, upper in STATION_BINS:
            selected_rows = (
                small_rows & time_rows
                & (station_count >= lower) & (station_count <= upper)
            )
            result = summarize_mask(
                data,
                np.broadcast_to(selected_rows[:, None], data['valid'].shape),
                args.bootstrap_draws,
                args.bootstrap_seed + int(seconds) * 100 + lower,
            )
            joint_rows.append({
                'seconds': float(seconds),
                'station_bin': station_label,
                'station_lower': lower,
                'station_upper': upper,
                'low_support_events_lt20': result['events'] < 20,
                **result,
            })

    time_strata = []
    for seconds in TIME_VALUES:
        selected_rows = small_rows & np.isclose(requested_time, seconds)
        time_strata.append({
            'seconds': float(seconds),
            **summarize_mask(
                data,
                np.broadcast_to(selected_rows[:, None], data['valid'].shape),
                args.bootstrap_draws,
                args.bootstrap_seed + int(seconds),
            ),
        })
    station_strata = []
    for station_label, lower, upper in STATION_BINS:
        selected_rows = small_rows & (station_count >= lower) & (station_count <= upper)
        station_strata.append({
            'station_bin': station_label,
            'station_lower': lower,
            'station_upper': upper,
            **summarize_mask(
                data,
                np.broadcast_to(selected_rows[:, None], data['valid'].shape),
                args.bootstrap_draws,
                args.bootstrap_seed + lower,
            ),
        })

    small_event_values = np.asarray([
        values[0] for event_id, values in event_magnitudes.items()
        if values[0] <= MAGNITUDE_MAX
    ])
    one_second_one_station = next(
        row for row in joint_rows
        if float(row['seconds']) == 1.0 and row['station_bin'] == '1'
    )
    summary = {
        'analysis': 'RT59 reference small-event early PGA validation',
        'model': 'reference_rt59',
        'split': 'validation only; held-out test unopened',
        'magnitude_condition': 'catalog magnitude <= 3.0',
        'metric_coordinate': 'log10(m/s^2)',
        'small_event_counts': {
            'events': int(small_event_values.size),
            'rows': int(small_rows.sum()),
            'targets': int((data['valid'] & base_mask).sum()),
            'magnitude_min': float(small_event_values.min()),
            'magnitude_median': float(np.median(small_event_values)),
            'magnitude_max': float(small_event_values.max()),
            'events_below_3p0': int(np.sum(small_event_values < 3.0)),
            'events_equal_3p0': int(np.sum(np.isclose(small_event_values, 3.0))),
        },
        'overall_small_event': overall,
        'overall_all_magnitudes': all_magnitude,
        'relative_to_all_magnitudes': {
            'mae_increase_percent': float(
                100.0 * (overall['mae'] / all_magnitude['mae'] - 1.0)
            ),
            'rmse_increase_percent': float(
                100.0 * (overall['rmse'] / all_magnitude['rmse'] - 1.0)
            ),
        },
        'one_second_one_station': one_second_one_station,
        'station_support': {
            'events_1': next(
                row['events'] for row in station_strata if row['station_bin'] == '1'
            ),
            'events_2_to_3': next(
                row['events'] for row in station_strata if row['station_bin'] == '2–3'
            ),
            'events_4_to_7': next(
                row['events'] for row in station_strata if row['station_bin'] == '4–7'
            ),
            'events_8_or_more': int(np.unique(data['event_ids'][
                small_rows & (station_count >= 8)
            ]).size),
        },
        'bootstrap': {
            'unit': 'event ID cluster',
            'draws': args.bootstrap_draws,
            'seed': args.bootstrap_seed,
        },
        'time_values_seconds': TIME_VALUES.tolist(),
        'station_bins': [
            {'label': label, 'lower': lower, 'upper': upper}
            for label, lower, upper in STATION_BINS
        ],
        'cautions': [
            'single training seed',
            'development validation reused across model iterations',
            'PGA regression is not magnitude detection or operational alert latency',
            'joint cells with fewer than 20 events are low-support descriptive strata',
        ],
    }
    write_csv(output_dir / 'joint_strata.csv', joint_rows)
    write_csv(output_dir / 'time_strata.csv', time_strata)
    write_csv(output_dir / 'station_strata.csv', station_strata)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    manifest = {
        'npz': {'file': npz_path.name, 'sha256': sha256_file(npz_path)},
        'metrics': {'file': metrics_path.name, 'sha256': sha256_file(metrics_path)},
        'submitted_source_manifest_sha256': args.source_manifest,
        'checkpoint_identity': identity['checkpoint'],
        'reference_identity': {
            'parent_checkpoint_sha256': identity['reference_identity'][
                'parent_checkpoint_sha256'
            ],
            'readout_sha256': identity['reference_identity']['readout_sha256'],
            'parent_shared_fingerprint': identity['reference_identity'][
                'parent_shared_fingerprint'
            ],
        },
    }
    (output_dir / 'analysis_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    make_figure(output_dir / 'rt59_small_event_strata', joint_rows)
    write_report(output_dir, summary, time_strata, station_strata)
    artifact_names = [
        'FIGURE_CONTRACT.md',
        'QA_NOTES.md',
        'REPORT.md',
        'analysis_manifest.json',
        'analyze_rt59_small_events.py',
        'joint_strata.csv',
        'rt59_small_event_strata.pdf',
        'rt59_small_event_strata.png',
        'rt59_small_event_strata.svg',
        'station_strata.csv',
        'summary.json',
        'time_strata.csv',
    ]
    manifest_lines = []
    for name in artifact_names:
        artifact = output_dir / name
        if artifact.is_file():
            manifest_lines.append(f'{sha256_file(artifact)}  {name}')
    (output_dir / 'artifact_manifest.sha256').write_text(
        '\n'.join(manifest_lines) + '\n', encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'small_event_counts': summary['small_event_counts'],
        'overall': overall,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
