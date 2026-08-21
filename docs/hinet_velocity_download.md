# Hi-net annual raw-count archive

`tools/download_hinet_velocity.py` downloads Hi-net WIN32 raw counts for the
events and matched stations in the converted Japan HDF5 datasets. The default
storage mode keeps exactly one permanent waveform representation: the original
CNT bytes inside one HDF5 archive per year.

## Full 2000–2024 download

Set Hi-net credentials in the environment and run the launcher:

```bash
export HINET_USER='your_hinet_user'
export HINET_PASSWORD='your_hinet_password'
bash download_hinet.sh
```

The launcher processes years from 2024 down to 2000. Events inside each year
are also processed newest first. Its defaults are:

```text
HDF5_ROOT=/run/media/zhangb/My Passport/knet_converted/origin_corrected_diting_vel_acc_vs30
OUTPUT_ROOT=/run/media/zhangb/My Passport/hinet_data
START_YEAR=2000
END_YEAR=2024
MATCH_DISTANCE_KM=0.5
PRE_SECONDS=120
POST_SECONDS=120
MAX_YEAR_ATTEMPTS=3
HINET_TIMEOUT_SECONDS=300
HINET_RETRIES=3
STATION_BATCH_SIZE=40
HINET_DOWNLOAD_THREADS=1
MINUTE_FALLBACK=1
FALLBACK_SPAN_MINUTES=1
SUBREQUEST_SLEEP_SECONDS=0
```

Any value can be overridden without editing the script. For example:

```bash
START_YEAR=2023 END_YEAR=2024 MAX_YEAR_ATTEMPTS=5 bash download_hinet.sh
```

The origin-corrected source HDF5 files already store the corrected
`Origin_Time(JST)` and correction provenance. Do not pass the old annual
`--origin-corrections` CSV when using this HDF5 root.

## Permanent layout

```text
hinet_data/
  archive/
    hinet_raw_2000.h5
    ...
    hinet_raw_2024.h5
  catalog/
    hinet_stations.csv
    hinet_kiknet_station_matches_YYYY.csv
    hinet_events_YYYY.csv
    hinet_attempts_YYYY.csv
    hinet_archive_YYYY.json
  logs/
    download_hinet_YYYY.log
  .staging/
    YYYY/raw/<event_id>/        # temporary; removed after verified commit
```

An in-progress archive is named `hinet_raw_YYYY.partial.h5`. After every event,
the downloader:

1. downloads native CNT segments and their `.ch` table to `.staging`;
2. appends the exact bytes to the annual archive;
3. stores byte offsets, lengths, original names and SHA256 hashes;
4. reads the bytes back from HDF5 and verifies the hashes;
5. removes that event's staging directory only after verification.

The committed event row is the transaction marker. On restart, an uncommitted
append tail is truncated and already committed events are skipped. Once every
source event has a terminal committed record, the partial archive is marked
complete and atomically renamed to `hinet_raw_YYYY.h5`.

Resume also checks an archive identity containing the source HDF5 stat, station
match CSV hash, travel-time table hash, event selection and requested window.
If these inputs change, the downloader refuses to mix incompatible events in
the same annual archive and requires a new archive path.

Events without an accepted station match are committed with
`raw_status=no_matched_stations`; download failures are not committed and are
retried on the next attempt or launcher rerun.

## Robust retries for incomplete years

Each event is downloaded with the following all-or-nothing procedure:

1. unique Hi-net stations are split into batches (40 by default), which keeps
   large-earthquake ZIP responses smaller and reduces the risk of the old
   60-second HinetPy transport timeout;
2. the complete event window is requested once for each station batch;
3. if that request fails, the same batch is retried as consecutive one-minute
   requests;
4. every returned CNT set is checked for the full requested time coverage;
   every requested station must have vertical, north and east channel-table
   entries, and those exact channels must be present throughout the CNT window;
5. the event is committable only after every station batch and every time slice
   passes validation.

When an event needs multiple provider requests, unique `.ch` payloads are
concatenated byte-for-byte (without line rewriting) into the event's archived
channel table. Repeated identical minute tables are deduplicated by SHA-256.

The downloader overrides HinetPy 0.12's private ZIP transport so that the
configured timeout is also used by its internally created download clients.
HTTP, timeout and invalid/empty ZIP errors are retained in the annual attempt
journal instead of being collapsed to `NoneType object is not iterable`.

Transport settings are deliberately excluded from `archive_identity`: changing
the timeout, batch size or fallback strategy does not change the requested
scientific dataset, so existing `.partial.h5` archives remain resumable. The
strategy actually used for each newly committed event is stored in its manifest
as `raw_download_strategy` and `raw_batch_count`.

To retry the eight post-2004 years that currently each lack one technical
event, while reusing every committed event:

```bash
for year in 2005 2007 2012 2013 2016 2018 2022 2024; do
  START_YEAR="$year" END_YEAR="$year" \
  HINET_TIMEOUT_SECONDS=600 STATION_BATCH_SIZE=40 \
  MAX_YEAR_ATTEMPTS=1 bash download_hinet.sh
done
```

For provider throttling, set `SUBREQUEST_SLEEP_SECONDS=1`. To diagnose whether
the minute fallback is responsible for a result, disable it explicitly with
`MINUTE_FALLBACK=0`; this is not recommended for recovery runs.

## Archive schema

```text
/raw/cnt_bytes                   concatenated native CNT uint8 bytes
/raw/channel_table_bytes         concatenated original .ch bytes
/raw/manifest_bytes              per-event station manifest CSV bytes
/index/segments                  CNT offset/length/name/hash rows
/index/events                    committed event rows and manifest offsets
/index/attempts                  download attempt and failure journal
/metadata/provenance_json        command and preprocessing provenance
```

CNT and `.ch` payloads are deduplicated by SHA256. HDF5 compression is not
applied to CNT because WIN32 counts are already compact; HDF5 Fletcher32 checks
are enabled in addition to the per-payload SHA256 hashes.

The archive does **not** persist MiniSEED, SAC, PZ files, NPY/NPZ waveform
arrays, or decoded waveform shards. Original `.ch` bytes are sufficient to
regenerate response information later.

## Direct archive access

`tools/hinet_raw_archive.py` provides process-local readers and a collection
index across multiple annual files:

```python
from pathlib import Path
from tools.hinet_raw_archive import AnnualHinetArchiveReader

archive = Path('/path/to/hinet_data/archive/hinet_raw_2024.h5')
with AnnualHinetArchiveReader(archive) as reader:
    event_ids = reader.event_ids()
    manifest = reader.manifest(event_ids[0])
    channel_name, channel_table = reader.channel_table_item(event_ids[0])
    series = reader.read_series(event_ids[0], {'4a93', '4a94', '4a95'})
```

`series[channel_id]` is a pair of absolute timestamps and decoded `int32`
counts. Window selection, padding masks, resampling and normalization remain
runtime operations and do not create a second permanent waveform copy.

In a PyTorch DataLoader, do not open an h5py handle in the parent process and
pass it to forked workers. Each persistent worker should lazily open its own
read-only `AnnualHinetArchiveReader` and cache only a small number of decoded
events in memory. Temporary decoded caches may be placed in node-local
`$SLURM_TMPDIR` and discarded after the job.

For a frozen event-level split, the module also provides a map-style dataset:

```python
from pathlib import Path
from tools.hinet_raw_archive import HinetArchiveEventDataset

archives = sorted(Path('/path/to/hinet_data/archive').glob('hinet_raw_*.h5'))
train = HinetArchiveEventDataset.from_event_id_file(
    archives,
    Path('splits/v1/train_events.txt'),
    components=('U', 'N', 'E'),
    max_open_archives=4,
)
sample = train[0]
# sample: event metadata, station manifest, parsed channel table and transient
# channel_id -> (absolute timestamps, int32 counts) series
```

The existing waveform QC tool also detects the annual archive directly:

```bash
python tools/plot_hinet_accel_velocity_qc.py \
  --hdf5 '/path/to/2024/japan_2024.hdf5' \
  --download-root '/path/to/hinet_data' \
  --event-id 20240101193800
```

## One-year command

The launcher invokes this equivalent command for each year:

```bash
python -u tools/download_hinet_velocity.py \
  --hdf5 '/path/to/2024/japan_2024.hdf5' \
  --year 2024 \
  --mode all \
  --storage-mode annual-hdf5 \
  --archive-path '/path/to/hinet_data/archive/hinet_raw_2024.h5' \
  --output-root '/path/to/hinet_data' \
  --match-distance-km 0.5 \
  --pre-seconds 120 \
  --post-seconds 120 \
  --hinet-timeout-seconds 300 \
  --hinet-retries 3 \
  --station-batch-size 40 \
  --minute-fallback \
  --fallback-span-minutes 1 \
  --no-write-mseed \
  --response-mode none
```

The legacy file-tree output remains available with `--storage-mode files`.
MiniSEED and SAC PZ products are opt-in there via `--write-mseed` and
`--response-mode pz`; they are intentionally forbidden in annual archive mode.
