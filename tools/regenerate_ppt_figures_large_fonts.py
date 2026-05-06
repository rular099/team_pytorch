from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path('/tmp/japan_build_test3')
OUT_DIR = Path(__file__).resolve().parents[1] / 'ppt_figures' / 'data_overview_2024_overfit_ppt'
OUT_DIR.mkdir(parents=True, exist_ok=True)

station_cache_path = BASE_DIR / 'station_cache_d9a7d4d6.csv'
split_events_path = BASE_DIR / 'split_events.csv'
split_stations_path = BASE_DIR / 'split_stations.csv'

plt.rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 220,
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 12.5,
    'ytick.labelsize': 12.5,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

station_cache = pd.read_csv(station_cache_path)
split_events = pd.read_csv(split_events_path)
split_stations = pd.read_csv(split_stations_path)

for df in [station_cache, split_events, split_stations]:
    if 'EVENT' in df.columns:
        df['EVENT'] = df['EVENT'].astype(str)

full_events = station_cache.drop_duplicates('EVENT').copy()
overfit_events = split_events[split_events['is_overfit_selected']].copy()
overfit_stations = split_stations[split_stations['is_overfit_selected']].copy()

split_events = overfit_events
split_stations = overfit_stations

train_events = split_events[split_events['split'] == 'train'].copy()
dev_events = split_events[split_events['split'] == 'dev'].copy()
test_events = split_events[split_events['split'] == 'test'].copy()


def style_axis(ax, xlabel=None, ylabel=None, title=None, rotate_x=0):
    if title:
        ax.set_title(title, pad=10, weight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(axis='x', rotation=rotate_x)


# 1) full_dataset_scale
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)

axes[0].bar(['Events', 'Station rows'], [len(full_events), len(station_cache)], color=['#005f73', '#0a9396'])
style_axis(axes[0], ylabel='Count', title='Full Available Data Volume')

network_counts = station_cache['source_network'].value_counts().sort_index()
axes[1].bar(network_counts.index, network_counts.values, color=['#94d2bd', '#ee9b00'])
style_axis(axes[1], xlabel='Source network', ylabel='Count', title='Network Composition')

sensor_counts = station_cache['sensor_class'].value_counts()
axes[2].bar(sensor_counts.index, sensor_counts.values, color=['#ca6702', '#bb3e03', '#ae2012'])
style_axis(axes[2], xlabel='Sensor class', ylabel='Count', title='Sensor Composition', rotate_x=18)

fig.savefig(OUT_DIR / 'full_dataset_scale_ppt.png', bbox_inches='tight')
plt.close(fig)

# 2) magnitude_coverage
bins = np.arange(
    math.floor(full_events['Magnitude'].min() * 2) / 2,
    math.ceil(full_events['Magnitude'].max() * 2) / 2 + 0.5,
    0.5,
)

fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)

axes[0].hist(full_events['Magnitude'], bins=bins, alpha=0.75, color='#0a9396', label='Full available')
axes[0].hist(overfit_events['Magnitude'], bins=bins, alpha=0.75, color='#ee9b00', label='Overfit selected')
style_axis(axes[0], xlabel='Magnitude', ylabel='Number of events', title='Magnitude Coverage: Full vs Overfit Selected')
axes[0].legend(frameon=False)

for split_name, df, color in [('train', train_events, '#005f73'), ('dev', dev_events, '#94d2bd'), ('test', test_events, '#bb3e03')]:
    axes[1].hist(df['Magnitude'], bins=bins, alpha=0.65, label=split_name, color=color)
style_axis(axes[1], xlabel='Magnitude', ylabel='Number of events', title='Magnitude Coverage by Split')
axes[1].legend(frameon=False)

fig.savefig(OUT_DIR / 'magnitude_coverage_ppt.png', bbox_inches='tight')
plt.close(fig)

# 3) geographic_coverage
fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)

sc = axes[0].scatter(
    full_events['Longitude'], full_events['Latitude'],
    c=full_events['Magnitude'], s=24, cmap='viridis', alpha=0.82,
    edgecolors='none'
)
style_axis(axes[0], xlabel='Longitude', ylabel='Latitude', title='Full Available Events')
cbar = fig.colorbar(sc, ax=axes[0], shrink=0.95)
cbar.set_label('Magnitude', fontsize=14)
cbar.ax.tick_params(labelsize=11.5)

axes[1].scatter(full_events['Longitude'], full_events['Latitude'], s=12, color='lightgray', alpha=0.35, label='Full available')
axes[1].scatter(overfit_events['Longitude'], overfit_events['Latitude'], s=34, color='#ee9b00', alpha=0.9, label='Overfit selected')
style_axis(axes[1], xlabel='Longitude', ylabel='Latitude', title='Overfit Selection Coverage')
axes[1].legend(frameon=False)

fig.savefig(OUT_DIR / 'geographic_coverage_ppt.png', bbox_inches='tight')
plt.close(fig)

# 4) split_and_overfit_distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)

split_counts = split_events['split'].value_counts().reindex(['train', 'dev', 'test']).fillna(0)
axes[0].bar(split_counts.index, split_counts.values, color=['#005f73', '#94d2bd', '#bb3e03'])
style_axis(axes[0], xlabel='Split', ylabel='Number of events', title='Event Counts By Split')

overfit_counts = split_events.groupby('split')['is_overfit_selected'].sum().reindex(['train', 'dev', 'test']).fillna(0)
axes[1].bar(overfit_counts.index, overfit_counts.values, color=['#ee9b00', '#ee9b00', '#ee9b00'])
style_axis(axes[1], xlabel='Split', ylabel='Number of events', title='Overfit Selected Events By Split')

fig.savefig(OUT_DIR / 'split_and_overfit_distribution_ppt.png', bbox_inches='tight')
plt.close(fig)

print('Generated:')
for p in sorted(OUT_DIR.iterdir()):
    print('-', p)
