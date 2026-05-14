# Hi-net Velocity Download Workflow

This document describes the first-pass Hi-net velocity waveform downloader added
for matching Hi-net stations to existing K-NET/KiK-net acceleration events.

## Scope

The downloader in `tools/download_hinet_velocity.py` supports:

- full HDF5 event download;
- random smoke-test download;
- event-id file selection, including `pga_configs/stage2_512_event_ids.txt`;
- K-NET/KiK-net to Hi-net station matching by horizontal distance;
- JMA2001A theoretical P arrivals with ak135 fallback;
- event-level raw Hi-net WIN32 download;
- raw-count MiniSEED station cuts when a local WIN32 reader can map channels;
- channel table / response metadata caching for later physical-unit conversion.

The first version does not support selecting events from
`/run/media/zhangb/aa0013a6-c6ff-4112-9526-410918058645/Japandata/waveformsnew/`.

## Dependencies

The script uses existing project dependencies plus:

- `HinetPy` for authenticated Hi-net access;
- `obspy` for ak135 travel times and best-effort WIN32 to MiniSEED conversion.

Install dependencies in the active environment before real downloads:

```bash
pip install -r requirements.txt
```

Hi-net credentials are read from environment variables:

```bash
export HINET_USER='your_hinet_user'
export HINET_PASSWORD='your_hinet_password'
```

Do not put credentials into config files or commit them.

## Output Layout

With `--output-root hinet_velocity_downloads`, the script writes:

```text
hinet_velocity_downloads/
  inventory/
    hinet_stations.csv
  matches/
    hinet_kiknet_station_matches.csv
  raw/
    <event_id>/
      *.cnt
      *.ch
  responses/
    <event_id>/
      *.ch
      *.channels.csv
      SAC_PZs_*              # if HinetPy response extraction succeeds
  mseed/
    <event_id>/
      <knet_station>__<hinet_station>.mseed
  manifests/
    download_manifest.csv
    summary.csv
    events/
      <event_id>/
        arrivals.csv
        download_manifest.csv
```

Raw Hi-net `.cnt` files are kept as the authoritative waveform archive. MiniSEED
files are raw-count cuts around the theoretical P arrival and are written only
when conversion succeeds. The manifest records conversion failures without
discarding raw downloads.

## Station Matching

The script reads K-NET/KiK-net station metadata from
`metadata/station_metadata` when present. If that table is absent, it falls back
to per-event datasets only when `station_codes` and `coords` exist.

Hi-net station metadata is cached in `inventory/hinet_stations.csv`. Matching is
done by haversine distance between K-NET/KiK-net and Hi-net station coordinates.
The default threshold is `1.0 km`.

The match table includes:

- K-NET/KiK-net station code and coordinates;
- nearest Hi-net station and coordinates;
- nearest and second-nearest distances;
- `accepted`;
- `ambiguous_within_2x`.

Inspect this file before large downloads:

```bash
column -s, -t hinet_velocity_downloads/matches/hinet_kiknet_station_matches.csv | less -S
```

If the distance distribution looks too loose, rerun with a stricter threshold:

```bash
python tools/download_hinet_velocity.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --mode smoketest \
  --num-events 3 \
  --match-distance-km 0.5 \
  --overwrite-matches \
  --output-root hinet_velocity_smoke
```

## Travel Times and Windows

For every matched event-station pair:

1. predict P travel time using JMA2001A;
2. fall back to ak135 if the JMA table is unavailable or the requested point is
   outside the JMA grid;
3. define the station cut as `P - 120 s` to `P + 120 s`;
4. download one event-level raw Hi-net window from the earliest cut start to the
   latest cut end among matched stations.

The manifest records:

- `travel_time_model`: `jma2001a` or `ak135`;
- `travel_time_status`: `ok` or `fallback`;
- `p_seconds_after_origin`;
- `ppick_time_jst`;
- `cut_start_jst`;
- `cut_end_jst`.

## Commands

### Smoke Test

Run a small random event sample first:

```bash
python tools/download_hinet_velocity.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --mode smoketest \
  --num-events 3 \
  --seed 42 \
  --output-root hinet_velocity_smoke
```

Use `--dry-run` to plan windows and matching without downloading waveforms. If
the Hi-net inventory cache does not exist yet, real network access is still
needed once to create it:

```bash
python tools/download_hinet_velocity.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --mode smoketest \
  --num-events 3 \
  --dry-run \
  --output-root hinet_velocity_smoke
```

### Stage2 512 Events

```bash
python tools/download_hinet_velocity.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --event-ids pga_configs/stage2_512_event_ids.txt \
  --output-root hinet_velocity_stage2_512
```

### Full HDF5

```bash
python tools/download_hinet_velocity.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --mode all \
  --output-root hinet_velocity_all
```

## Useful Options

- `--match-distance-km 1.0`: station matching threshold.
- `--pre-seconds 120 --post-seconds 120`: station cut window around predicted P.
- `--overwrite-inventory`: refresh cached Hi-net station list.
- `--overwrite-matches`: rebuild station matching table.
- `--overwrite-raw`: redownload raw event windows.
- `--no-write-mseed`: keep raw WIN32 and metadata only.
- `--pad-mseed`: zero-pad MiniSEED cuts when raw coverage is short. This is off
  by default because padded zeros can be confused with real raw counts.
- `--sleep-seconds N`: sleep between events to reduce server load.

## Notes for Developers

- Raw `.cnt` and `.ch` files are the primary preservation format. Do not delete
  them after MiniSEED conversion.
- HinetPy's SAC conversion tools may remove sensitivity and output physical
  velocity units. This workflow avoids using SAC conversion as the waveform
  source because the requested data product is raw counts.
- The response cache stores the Hi-net channel table and attempts to write SAC
  pole-zero files via HinetPy. If response extraction fails, the channel table is
  still saved and the manifest records the failure.
- The MiniSEED path depends on mapping WIN32 channel ids back to Hi-net station
  components from the channel table. If this mapping fails in a new HinetPy
  version, inspect `responses/<event_id>/*.channels.csv` and adjust
  `read_channel_table()` or `COMPONENT_MAP` in the script.
- Existing dirty files in the training/report workflow are unrelated to this
  downloader. Keep future downloader changes scoped to `tools/` and `docs/`
  unless the HDF5 schema itself changes.
