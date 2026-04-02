"""
Convert Japan KiK-net data (CSV + flat HDF5) to the TEAM training format.

Usage:
    # Convert full year
    python convert_japan_data.py --year 2020 --japan_dir /path/to/Japandata --output japan_2020.hdf5

    # Convert small subset for overfit testing
    python convert_japan_data.py --year 2020 --japan_dir /path/to/Japandata --output japan_overfit.hdf5 \
        --max_events 16 --min_stations 3 --sensor surface
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def load_japan_csv(csv_path):
    """Load and parse Japan KiK-net CSV metadata."""
    df = pd.read_csv(csv_path)
    # Create a unique event identifier from origin time + location
    df['event_id'] = (df['origintime'].astype(str) + '_' +
                      df['lat'].astype(str) + '_' +
                      df['long'].astype(str))
    return df


def group_events(df, sensor='surface'):
    """Group CSV rows by event, filtering by sensor type.

    Args:
        df: Japan CSV DataFrame
        sensor: 'surface' (dir=2), 'borehole' (dir=1), or 'both'
    """
    if sensor == 'surface':
        df = df[df['dir'] == 2].copy()
    elif sensor == 'borehole':
        df = df[df['dir'] == 1].copy()
    # else 'both' — keep all

    events = {}
    for event_id, group in df.groupby('event_id'):
        events[event_id] = group
    return events


def estimate_p_pick(waveform, sampling_rate=100, sta_len=0.5, lta_len=5.0, threshold=3.0):
    """Simple STA/LTA P-pick estimation on the vertical component.

    Returns sample index of estimated P arrival, or 0 if not found.
    """
    # Use vertical component (index 2, typically UD)
    trace = waveform[2, :]  # shape (n_samples,)
    trace = np.abs(trace)

    nsta = int(sta_len * sampling_rate)
    nlta = int(lta_len * sampling_rate)

    if len(trace) < nlta + nsta:
        return 0

    # Compute STA/LTA ratio
    sta = np.zeros(len(trace))
    lta = np.zeros(len(trace))
    for i in range(nlta, len(trace)):
        sta[i] = np.mean(trace[i - nsta:i])
        lta[i] = np.mean(trace[i - nlta:i])

    lta[lta == 0] = 1e-10
    ratio = sta / lta

    # Find first crossing of threshold
    picks = np.where(ratio[nlta:] > threshold)[0]
    if len(picks) > 0:
        return picks[0] + nlta
    return 0


def convert_japan_to_team(japan_dir, year, output_path, max_events=None,
                          min_stations=1, sensor='surface', estimate_picks=True):
    """Convert Japan KiK-net data to TEAM HDF5 format.

    Args:
        japan_dir: Path to Japandata directory
        year: Year to convert (e.g., 2020)
        output_path: Output HDF5 file path
        max_events: Limit number of events (None = all)
        min_stations: Minimum stations per event to include
        sensor: 'surface', 'borehole', or 'both'
        estimate_picks: Whether to estimate P-picks via STA/LTA
    """
    japan_dir = Path(japan_dir)
    csv_path = japan_dir / f'{year}.csv'
    h5_path = japan_dir / f'{year}.h5'

    print(f'Loading CSV: {csv_path}')
    df = load_japan_csv(csv_path)

    print(f'Grouping events (sensor={sensor})...')
    events = group_events(df, sensor=sensor)

    # Filter by min_stations
    events = {k: v for k, v in events.items() if len(v) >= min_stations}
    print(f'Events with >= {min_stations} stations: {len(events)}')

    if max_events:
        # Sort by magnitude (descending) to get more interesting events first
        event_mags = {k: v['mag'].iloc[0] for k, v in events.items()}
        sorted_events = sorted(event_mags, key=event_mags.get, reverse=True)
        selected = sorted_events[:max_events]
        events = {k: events[k] for k in selected}
        print(f'Selected {len(events)} events (by magnitude)')

    print(f'Loading waveforms from: {h5_path}')
    japan_h5 = h5py.File(h5_path, 'r')

    # Build event metadata and data
    event_rows = []
    sampling_rate = 100  # KiK-net standard

    with h5py.File(output_path, 'w') as out_h5:
        # Create groups
        meta_grp = out_h5.create_group('metadata')
        data_grp = out_h5.create_group('data')

        meta_grp.create_dataset('sampling_rate', data=sampling_rate)

        event_count = 0
        for event_id, group in events.items():
            row0 = group.iloc[0]
            event_name = f'event_{event_count:05d}'

            # Collect station waveforms
            waveforms_list = []
            coords_list = []
            pga_list = []
            p_picks_list = []
            valid_stations = 0

            for _, station_row in group.iterrows():
                key = station_row['key']
                if key not in japan_h5:
                    continue

                wf = japan_h5[key][()]  # shape (3, n_samples)
                if wf.shape[0] != 3:
                    continue

                waveforms_list.append(wf)  # (3, n_samples)
                coords_list.append([
                    station_row['stationlat'],
                    station_row['stationlong'],
                    station_row['stationheight(m)'] / 1000.0  # m → km
                ])
                pga_list.append(station_row['PGA'])

                if estimate_picks:
                    p_pick = estimate_p_pick(wf, sampling_rate)
                else:
                    p_pick = 0
                p_picks_list.append(p_pick)
                valid_stations += 1

            if valid_stations < min_stations:
                continue

            # Find common length (pad shorter waveforms)
            lengths = [wf.shape[1] for wf in waveforms_list]
            max_len = max(lengths)

            # Stack into (n_stations, n_samples, 3) — transpose from (3, n_samples)
            waveforms = np.zeros((valid_stations, max_len, 3), dtype=np.float64)
            for i, wf in enumerate(waveforms_list):
                n_samples = wf.shape[1]
                waveforms[i, :n_samples, :] = wf.T  # (3, n_samples) → (n_samples, 3)

            coords = np.array(coords_list, dtype=np.float64)
            pga = np.array(pga_list, dtype=np.float64)
            p_picks = np.array(p_picks_list, dtype=np.int64)

            # Replace invalid PGA
            pga = np.where(np.isfinite(pga), pga, 0.0)

            # Store in HDF5
            evt_grp = data_grp.create_group(event_name)
            evt_grp.create_dataset('waveforms', data=waveforms)
            evt_grp.create_dataset('coords', data=coords)
            evt_grp.create_dataset('pga', data=np.log10(pga + 1e-10))  # log10(PGA) as in TEAM
            evt_grp.create_dataset('p_picks', data=p_picks)

            # Event metadata row
            event_rows.append({
                'EVENT': event_name,
                'Latitude': row0['lat'],
                'Longitude': row0['long'],
                'DEPTH': row0['depth'],
                'Magnitude': row0['mag'],
            })
            event_count += 1

        # Store event metadata as individual h5py datasets (avoids pytables dependency)
        event_df = pd.DataFrame(event_rows)
        em_grp = meta_grp.create_group('event_metadata')
        for col in event_df.columns:
            vals = event_df[col].values
            if vals.dtype.kind in ('U', 'O'):  # string columns
                em_grp.create_dataset(col, data=vals.astype('S'))
            else:
                em_grp.create_dataset(col, data=vals)

        print(f'\nConversion complete: {output_path}')
        print(f'  Events: {event_count}')
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
    parser.add_argument('--sensor', choices=['surface', 'borehole', 'both'], default='surface',
                        help='Sensor type to use')
    parser.add_argument('--no_picks', action='store_true', help='Skip P-pick estimation')
    args = parser.parse_args()

    convert_japan_to_team(
        japan_dir=args.japan_dir,
        year=args.year,
        output_path=args.output,
        max_events=args.max_events,
        min_stations=args.min_stations,
        sensor=args.sensor,
        estimate_picks=not args.no_picks,
    )
