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
      *.cnt                  # merged event window if catwin32 is available
      *.ch
      segments/
        *.cnt                # one-minute Hi-net files kept if merge fails
        *.euc.ch
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

Raw Hi-net `.cnt` files are kept as the authoritative waveform archive. If
HinetPy downloads the one-minute files but local `catwin32` merging fails, the
manifest records `raw_status=downloaded_unmerged` and keeps the segment files
under `raw/<event_id>/segments/`. MiniSEED station cuts are written from the
merged file when available, or directly from the raw segments with a pure Python
WIN32 parser when `catwin32` is unavailable.

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

Some K-NET/KiK-net event headers only preserve origin time to the minute. Before
large Hi-net downloads, fetch JMA daily hypocenters and create a second-level
origin correction table:

```bash
python tools/fetch_jma_hypocenters.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --output-csv jma_origin_corrections/japan_2024_origin_corrections.csv \
  --catalog-csv jma_origin_corrections/jma_2024_daily_catalog.csv \
  --summary-json jma_origin_corrections/japan_2024_origin_corrections_summary.json \
  --cache-dir jma_origin_corrections/cache
```

Then pass the accepted corrections to the Hi-net downloader:

```bash
python tools/download_hinet_velocity.py \
  --hdf5 /path/to/japan_2024.hdf5 \
  --origin-corrections jma_origin_corrections/japan_2024_origin_corrections.csv \
  --mode smoketest \
  --num-events 3 \
  --output-root hinet_velocity_smoke
```

The downloader filters to `accepted==1` by default. Use
`--use-unaccepted-origin-corrections` only for manually reviewed ambiguous
matches.

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

### QC Plot: Acceleration vs Hi-net Velocity

After a smoke test, draw matched K-NET/KiK-net acceleration and Hi-net velocity
for manual timing checks:

```bash
python tools/plot_hinet_accel_velocity_qc.py \
  --event-id 20240101185300 \
  --station ISKH01 \
  --output-dir hinet_velocity_qc
```

The script defaults to the `--hdf5` and `--output-root` values recorded in
`download_hinet.sh` and shows theoretical P +/- 50 s. The default plot keeps the
waveforms readable: theoretical P is the only vertical pick line; PGA is marked
with a small triangle on the acceleration panel time axis; trigger and final
training picks are marked with small triangles on the velocity panel time axis.
`travel_pred` is not drawn separately because it is the training-side
theoretical P estimate, and `travel_coarse` is not drawn because it is the
clipped coarse pick derived from that estimate.

`qc_summary.csv` still records every available pick offset from the theoretical
P time. For detailed debugging, add `--show-candidate-picks` to mark STALTA and
DiTing candidate picks on the velocity-panel time axis, and
`--show-search-windows` to shade the travel-time/STALTA search windows. The
script first uses MiniSEED if present, then falls back to raw Hi-net `*.cnt`
plus `*.ch` files when MiniSEED was not produced.

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
