# V01 velocity / P-prefix padding control

## Scope

V01 keeps `gemini_models.py`, the DiTing computation, RT55 tensor shapes, PGA
definition and losses unchanged.  It changes only the data contract:

- Hi-net CNT counts are divided once by each archived CH
  `counts_per_physical_unit`; the resulting quantity is velocity sensor output
  in m/s with `response_correction=sensitivity_only`.
- Physical velocity sources and KNET PGA queries keep separate sensor IDs and
  coordinates.  Query-only rows cannot enter a waveform input slot.
- `V_full` and `V_missing` share one derived velocity cache.  `V_missing`
  deletes only samples strictly before P, writes zero, and sets the storage
  mask false before centering/normalization.  P and post-P support are inherited
  unchanged.
- Train-only KNET storage support supplies deterministic retained-prefix
  templates.  The RT55 split is read before the velocity intersection.  Test
  waveforms are not materialized or evaluated.
- `A_pair` is a paired acceleration bridge.  The current audit shows it is
  dominated by KiK/Hi-net pairs and therefore is not an RT55 KNET-only result.

The preflight creates two reproducible caches under `V01_CACHE_ROOT`: one
velocity cache and one paired-acceleration cache.  It does not write a third
`V_missing` waveform copy.

## Locally verified data snapshot (2026-09-29)

The external disk is mounted read-only:

```text
/dev/sdd1 -> /run/media/zhangb/My Passport (fuseblk, ro)
```

No Hi-net downloader is running.  Archive/catalog timestamps stopped on
2026-08-11.  The immutable snapshot contains 12 completed years and 13 partial
years:

```text
complete: 2006 2008 2009 2010 2011 2014 2015 2017 2019 2020 2021 2023
partial:  2000 2001 2002 2003 2004 2005 2007 2012 2013 2016 2018 2022 2024
```

The historical download counters remain 12,708 committed events: 12,392 with
waveform data and 316 without a station match, from 14,153 requested events.
There are 1,445 unarchived events.  The archive is 26 GiB, the catalog is 14
MiB, and the acceleration HDF5 root is 60 GiB.  Years 2000--2003 have no usable
waveform events; V01 configs therefore cover 2004--2024.

The metadata-only V01 audit against the real RT55 frozen split found:

| item | count |
|---|---:|
| train events | 8,439 |
| validation events | 1,225 |
| source-event sensor rows | 129,636 |
| KNET query targets | 153,462 |
| KNET-compatible source rows | 2,094 |
| KiK/Hi-net bridge source rows | 127,542 |
| excluded frozen test events (not opened as waveforms) | 2,483 |
| train-only prefix templates | 146,842 |

The retained-pre-P template median is 7.21 s; 3.21%, 16.10% and 32.78% retain
less than 1 s, 3 s and 5 s respectively.  Therefore the artificial intervention
has zero dose for part of the 5 s pre-P model window by design; this is reported
rather than silently strengthening the missingness distribution.

These are metadata/preflight counts, not completed training counts.  Full
materialization can reduce them if channel-level validation rejects a source.
The HPC `preflight_summary.json` is authoritative for the actual run.

## Data to upload to the supercomputer

Required new velocity data (do not upload `.lock` files):

```text
local source:
  /run/media/zhangb/My Passport/hinet_data/archive/hinet_raw_2004*.h5
  ...
  /run/media/zhangb/My Passport/hinet_data/archive/hinet_raw_2024*.h5

recommended HPC destination:
  /public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data/archive/
```

Upload the matching catalog for provenance as well, although the builder reads
the event manifest and CH table embedded in each archive:

```text
local:
  /run/media/zhangb/My Passport/hinet_data/catalog/
HPC:
  /public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data/catalog/
```

The 2000--2003 partial archives may be omitted for V01.  The following data are
already referenced by the launcher and only need uploading if they are absent
on the cluster:

```text
acceleration HDF5 local:
  /run/media/zhangb/My Passport/knet_converted/origin_corrected_diting_vel_acc_vs30/
default HPC:
  /public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30/

frozen split local:
  ../chaosuan_res/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/split_events.csv
default HPC:
  /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/split_events.csv

RT55 epoch-32 parent default HPC:
  /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/full_model_best_ep32.pth
```

Upload the complete repository/branch too; do not upload only the new tools,
because the source-identity manifest covers inherited configs and runtime code.

## Submission

First set cluster paths.  For an uploaded folder without `.git`, run one dry
run to obtain the printed source manifest hash, then repeat with that hash:

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics_vel

export WORKDIR=$PWD
export ACC_DATA_ROOT=/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30
export VELOCITY_DATA_ROOT=/public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data
export RT55_RUN_ROOT=/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42
export FROZEN_SPLIT_MANIFEST=$RT55_RUN_ROOT/split_events.csv
export RT55_EP32_CHECKPOINT=$RT55_RUN_ROOT/full_model_best_ep32.pth
export V01_RUN_ROOT=$WORKDIR/v01_velocity_prep_padding_seed42
export SOURCE_IDENTITY_MODE=uploaded_sha256

DRY_RUN=1 ACTION=all ARMS=vfull,vmissing,apair \
  bash tools/run_v01_prep_padding_controls_slurm.sh

export EXPECTED_SOURCE_MANIFEST_SHA256=<hash printed by dry-run>
CONFIRM_V01=1 DRY_RUN=0 ACTION=all ARMS=vfull,vmissing,apair \
  bash tools/run_v01_prep_padding_controls_slurm.sh
```

For a Git clone, use `SOURCE_IDENTITY_MODE=git` and set the reviewed full
`EXPECTED_GIT_COMMIT`.  The velocity-only core run is:

```bash
CONFIRM_V01=1 DRY_RUN=0 ACTION=all ARMS=vfull,vmissing \
  bash tools/run_v01_prep_padding_controls_slurm.sh
```

Individual stages use `ACTION=preflight|train|eval|analyze`.  A stopped training
job can be continued explicitly without deleting the output:

```bash
CONFIRM_V01=1 DRY_RUN=0 ACTION=train ARMS=vfull RESUME_V01=1 \
  bash tools/run_v01_prep_padding_controls_slurm.sh
```

Default resources are configurable through `SLURM_PARTITION`,
`SLURM_GRES_RESOURCE` (default `dcu`), `SLURM_ACCOUNT`, `PREFLIGHT_CPUS`,
`PREFLIGHT_MEM`, `PREFLIGHT_TIME`, `TRAIN_NODES`, `TRAIN_GPUS_PER_NODE`,
`TRAIN_TIME`, `EVAL_GPUS`, `EVAL_TIME`, `SLURM_CPUS_PER_TASK`, `SLURM_MEM`,
`CONDA_ENV`, `MODULE_UNLOAD`, `MODULE_LOADS`, `DITING_CONFIG`, and
`DITING_PRETRAINED`.  Defaults stay below the site's observed one-day effective
limit (`23:50:00`).  The CPU preflight defaults to 8 CPUs and 102400M because
the `diting` partition rejected the earlier 192000M request as unsatisfiable.

## Results to return

Return the complete `V01_RUN_ROOT` directory or at least:

- `derived_cache/protocol_lock.json`, `preflight_summary.json`,
  `cohort_counts_by_year.csv`, and `cohort_audit.csv`;
- each arm's resolved `config.json`, `split_events.csv`, `split_stations.csv`,
  `full_model_init.pth`, `full_model_last.pth`, training logs and scalar data;
- all files under `eval/` (`*.npz`, `*.metrics.json`, `*.txt`);
- all files under `report/`;
- `logs/` and `sacct -j <jobids> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P`.

Do not return raw Hi-net waveforms to GitHub.  The lightweight review package
should contain hashes, aggregate tables and approved fixed examples only.
