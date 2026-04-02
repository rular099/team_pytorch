"""
Convert Japan KiK-net data (CSV + flat HDF5) to the TEAM training format.

Time-alignment strategy (matching japan.py / Italy data format):
  - All station waveforms within an event are aligned to a common absolute time axis
  - The reference point is the earliest P-pick (trigger) across all stations
  - Output window: time_before seconds before earliest P-pick + time_after seconds after
  - P-picks are stored as sample indices relative to the aligned waveform start
  - This enables the cutout mechanism to simulate real-time EEW data truncation

KiK-net data format:
  - Each station triggers independently; trigger point is at 15s into the recording
  - recordtime = end of recording; rec_start = recordtime - durationtimes
  - The trigger time (≈ P-pick) in absolute time = rec_start + 15s

Usage:
    python convert_japan_data.py --year 2020 --japan_dir /path/to/Japandata --output japan_overfit.hdf5 \\
        --max_events 16 --min_stations 5 --sensor surface
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


KIKNET_PRETRIGGER = 15  # seconds of pre-trigger buffer in KiK-net recordings


def load_japan_csv(csv_path):
    """Load and parse Japan KiK-net CSV metadata."""
    df = pd.read_csv(csv_path)
    df['event_id'] = (df['origintime'].astype(str) + '_' +
                      df['lat'].astype(str) + '_' +
                      df['long'].astype(str))
    return df


def group_events(df, sensor='surface'):
    """Group CSV rows by event, filtering by sensor type."""
    if sensor == 'surface':
        df = df[df['dir'] == 2].copy()
    elif sensor == 'borehole':
        df = df[df['dir'] == 1].copy()

    events = {}
    for event_id, group in df.groupby('event_id'):
        events[event_id] = group
    return events


def parse_absolute_time(time_str):
    """Parse KiK-net time string to Unix timestamp (seconds)."""
    return datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S').timestamp()


def align_event_waveforms(waveforms_raw, trigger_times_abs, sampling_rate,
                          time_before, time_after):
    """Align station waveforms to a common time axis.

    Args:
        waveforms_raw: list of (3, n_samples) arrays, one per station
        trigger_times_abs: array of absolute trigger times (seconds) per station
        sampling_rate: Hz
        time_before: seconds before earliest trigger to include
        time_after: seconds after earliest trigger to include

    Returns:
        aligned: (n_stations, total_samples, 3) array
        p_picks: (n_stations,) sample indices of P-picks in aligned frame
    """
    n_stations = len(waveforms_raw)
    total_samples = int((time_before + time_after) * sampling_rate)
    pretrigger_samples = int(KIKNET_PRETRIGGER * sampling_rate)

    # Earliest trigger is the reference (sample = time_before * sampling_rate)
    earliest_trigger = np.min(trigger_times_abs)
    ref_sample = int(time_before * sampling_rate)  # where earliest P-pick lands

    aligned = np.zeros((n_stations, total_samples, 3), dtype=np.float64)
    p_picks = np.zeros(n_stations, dtype=np.int64)

    for i in range(n_stations):
        wf = waveforms_raw[i]  # (3, n_samples_i)
        n_samp = wf.shape[1]

        # This station's trigger offset from earliest trigger (in samples)
        dt = trigger_times_abs[i] - earliest_trigger
        trigger_offset = int(round(dt * sampling_rate))

        # In aligned frame, this station's trigger is at:
        trigger_in_aligned = ref_sample + trigger_offset
        p_picks[i] = trigger_in_aligned

        # In the raw waveform, trigger is at sample pretrigger_samples (=1500)
        # So raw sample j corresponds to aligned sample:
        #   aligned_idx = trigger_in_aligned + (j - pretrigger_samples)
        # => j = aligned_idx - trigger_in_aligned + pretrigger_samples

        # Copy range: aligned [0, total_samples) ← raw [src_start, src_end)
        src_start = pretrigger_samples - trigger_in_aligned
        src_end = src_start + total_samples

        # Clip to valid raw range
        al_start = max(0, -src_start)
        al_end = total_samples - max(0, src_end - n_samp)
        raw_start = max(0, src_start)
        raw_end = min(n_samp, src_end)

        if raw_start < raw_end and al_start < al_end:
            length = min(al_end - al_start, raw_end - raw_start)
            aligned[i, al_start:al_start + length, :] = wf[:, raw_start:raw_start + length].T

    return aligned, p_picks


def convert_japan_to_team(japan_dir, year, output_path, max_events=None,
                          min_stations=1, sensor='surface',
                          time_before=5, time_after=25):
    """Convert Japan KiK-net data to TEAM HDF5 format with proper time alignment.

    Args:
        japan_dir: Path to Japandata directory
        year: Year to convert
        output_path: Output HDF5 file path
        max_events: Limit number of events (None = all)
        min_stations: Minimum stations per event
        sensor: 'surface', 'borehole', or 'both'
        time_before: seconds before earliest P-pick to include
        time_after: seconds after earliest P-pick to include
    """
    japan_dir = Path(japan_dir)
    csv_path = japan_dir / f'{year}.csv'
    h5_path = japan_dir / f'{year}.h5'
    sampling_rate = 100

    print(f'Loading CSV: {csv_path}')
    df = load_japan_csv(csv_path)

    print(f'Grouping events (sensor={sensor})...')
    events = group_events(df, sensor=sensor)

    events = {k: v for k, v in events.items() if len(v) >= min_stations}
    print(f'Events with >= {min_stations} stations: {len(events)}')

    if max_events:
        event_mags = {k: v['mag'].iloc[0] for k, v in events.items()}
        sorted_events = sorted(event_mags, key=event_mags.get, reverse=True)
        events = {k: events[k] for k in sorted_events[:max_events]}
        print(f'Selected {len(events)} events (by magnitude)')

    print(f'Loading waveforms from: {h5_path}')
    japan_h5 = h5py.File(h5_path, 'r')

    event_rows = []
    total_samples = int((time_before + time_after) * sampling_rate)

    with h5py.File(output_path, 'w') as out_h5:
        meta_grp = out_h5.create_group('metadata')
        data_grp = out_h5.create_group('data')

        meta_grp.create_dataset('sampling_rate', data=sampling_rate)
        meta_grp.create_dataset('time_before', data=time_before)
        meta_grp.create_dataset('time_after', data=time_after)

        event_count = 0
        for event_id, group in events.items():
            row0 = group.iloc[0]
            event_name = f'event_{event_count:05d}'

            waveforms_raw = []
            trigger_times_abs = []
            coords_list = []
            pga_list = []

            for _, station_row in group.iterrows():
                key = station_row['key']
                if key not in japan_h5:
                    continue

                wf = japan_h5[key][()]  # (3, n_samples)
                if wf.shape[0] != 3:
                    continue

                # Compute absolute trigger time:
                # rec_end = recordtime, rec_start = rec_end - durationtimes
                # trigger = rec_start + KIKNET_PRETRIGGER
                rec_end = parse_absolute_time(station_row['recordtime'])
                rec_start = rec_end - station_row['durationtimes']
                trigger_abs = rec_start + KIKNET_PRETRIGGER

                waveforms_raw.append(wf)
                trigger_times_abs.append(trigger_abs)
                coords_list.append([
                    station_row['stationlat'],
                    station_row['stationlong'],
                    station_row['stationheight(m)'] / 1000.0
                ])
                pga_list.append(station_row['PGA'])

            if len(waveforms_raw) < min_stations:
                continue

            trigger_times_abs = np.array(trigger_times_abs)

            # Align all station waveforms to common time axis
            aligned, p_picks = align_event_waveforms(
                waveforms_raw, trigger_times_abs, sampling_rate,
                time_before, time_after)

            coords = np.array(coords_list, dtype=np.float64)
            pga = np.array(pga_list, dtype=np.float64)
            pga = np.where(np.isfinite(pga), pga, 0.0)

            # Store
            evt_grp = data_grp.create_group(event_name)
            evt_grp.create_dataset('waveforms', data=aligned)
            evt_grp.create_dataset('coords', data=coords)
            evt_grp.create_dataset('pga', data=np.log10(pga + 1e-10))
            evt_grp.create_dataset('p_picks', data=p_picks)

            event_rows.append({
                'EVENT': event_name,
                'Latitude': row0['lat'],
                'Longitude': row0['long'],
                'DEPTH': row0['depth'],
                'Magnitude': row0['mag'],
            })
            event_count += 1

        # Store event metadata
        event_df = pd.DataFrame(event_rows)
        em_grp = meta_grp.create_group('event_metadata')
        for col in event_df.columns:
            vals = event_df[col].values
            if vals.dtype.kind in ('U', 'O'):
                em_grp.create_dataset(col, data=vals.astype('S'))
            else:
                em_grp.create_dataset(col, data=vals)

        print(f'\nConversion complete: {output_path}')
        print(f'  Events: {event_count}')
        print(f'  Waveform length: {total_samples} samples ({time_before}+{time_after}s @ {sampling_rate}Hz)')
        print(f'  Sampling rate: {sampling_rate} Hz')
        print(f'  Magnitude range: {event_df["Magnitude"].min():.1f} - {event_df["Magnitude"].max():.1f}')
        print(f'  Lat range: {event_df["Latitude"].min():.2f} - {event_df["Latitude"].max():.2f}')
        print(f'  Lon range: {event_df["Longitude"].min():.2f} - {event_df["Longitude"].max():.2f}')

    japan_h5.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Japan KiK-net data to TEAM format')
    parser.add_argument('--japan_dir', required=True, help='Path to Japandata directory')
    parser.add_argument('--year', type=int, required=True, help='Year to convert')
    parser.add_argument('--output', required=True, help='Output HDF5 path')
    parser.add_argument('--max_events', type=int, default=None, help='Max events to include')
    parser.add_argument('--min_stations', type=int, default=3, help='Min stations per event')
    parser.add_argument('--sensor', choices=['surface', 'borehole', 'both'], default='surface')
    parser.add_argument('--time_before', type=int, default=5, help='Seconds before earliest P-pick')
    parser.add_argument('--time_after', type=int, default=25, help='Seconds after earliest P-pick')
    args = parser.parse_args()

    convert_japan_to_team(
        japan_dir=args.japan_dir,
        year=args.year,
        output_path=args.output,
        max_events=args.max_events,
        min_stations=args.min_stations,
        sensor=args.sensor,
        time_before=args.time_before,
        time_after=args.time_after,
    )
