# CODEX-RESULT: V01 velocity / P-prefix padding control

## Identity

- task: `20260929-v01-velocity-prep-padding-control`
- branch: `exp/v01-velocity-prep-padding-control`
- base commit: `9c95dbfaf92f36b2673d816026cd3f25b91eec66`
- implementation commit: `6a341fb936035d4654db4ddbba974a3cdd471352`
- `gemini_models.py` SHA256: `cd8481286436342d09781888967dc757f4bde383f4d344033d866c6b06b7be84`
- canonical RT55/V01 `model_params` SHA256: `d0ee04ee99872c2eeed644c1ce05ec822b311eb610b58c732465d024bcc1ef6b`

`gemini_models.py`, the DiTing implementation, and all RT55--RT61 configs were
left unchanged.  The exact final branch head (including this result document)
is reported in the user-facing handoff and can be obtained with
`git rev-parse HEAD`.

## Delivered

- A Hi-net annual-archive backend that decodes an event once, converts archived
  CNT counts to m/s using the per-channel sensitivity, preserves E/N/U order,
  and uses process-local bounded reader handles.
- A derived-cache builder that inherits the frozen RT55 split before filtering,
  materializes train/dev only, separates physical velocity source identities
  and coordinates from KNET PGA queries, builds the paired acceleration bridge,
  and records source hashes and unit provenance.
- Deterministic train-only P-prefix templates and a runtime `V_missing` view.
  Only valid samples strictly before P are removed; P and post-P support are
  inherited.  Removed samples are zero and mask false before centering.
- `A_pair`, `V_full`, and `V_missing` configs using identical RT55 model params,
  RT56 mixed geometry, seed 42, eight epochs and fixed learning rates of 1e-4.
- A val-only normal/random 2x2 cross-evaluation schedule for the two velocity
  checkpoints, paired event-cluster bootstrap aggregation, and an A-pair val
  evaluation path.
- A dry-by-default Slurm launcher with `preflight|train|eval|analyze|all`,
  `afterok` dependencies, non-overwrite guards, explicit resume, Git/uploaded
  source identity modes, and resource/path overrides.
- Legacy defaults remain off: the new split, metadata-support and intervention
  paths activate only through V01 config fields.

The canonical full launcher is
`tools/run_v01_prep_padding_controls_slurm.sh`; path/resource documentation and
copy-paste commands are in `docs/v01_velocity_prep_padding.md`.

## Real local data audit

On 2026-09-29 the external data volume was mounted read-only:

```text
/dev/sdd1 -> /run/media/zhangb/My Passport (fuseblk, ro)
```

No real downloader process was running.  The most recent archive/catalog
updates were 2026-08-11.  The snapshot has 12 complete years
(`2006,2008,2009,2010,2011,2014,2015,2017,2019,2020,2021,2023`) and 13 partial
years (`2000,2001,2002,2003,2004,2005,2007,2012,2013,2016,2018,2022,2024`).
Historical counters are 14,153 requested, 12,708 committed, 12,392 with
waveforms, 316 without matching stations and 1,445 unarchived.  Archive,
catalog and acceleration roots occupy about 26 GiB, 14 MiB and 60 GiB.

The real metadata-only frozen-split audit found 8,439 train events, 1,225 dev
events, 129,636 source-event sensor rows, 153,462 KNET query targets, 2,094
KNET-compatible source rows, 127,542 KiK/Hi-net bridge rows, and 2,483 excluded
test events.  The train-only template library has 146,842 rows.  These are
preflight counts, not completed training counts; channel validation during full
materialization may reduce them.

## Data transfer contract

Upload the annual velocity archive files for 2004--2024 from:

```text
/run/media/zhangb/My Passport/hinet_data/archive/hinet_raw_2004*.h5
...
/run/media/zhangb/My Passport/hinet_data/archive/hinet_raw_2024*.h5
```

to:

```text
/public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data/archive/
```

Do not upload `.lock` files.  The recommended provenance catalog transfer is:

```text
/run/media/zhangb/My Passport/hinet_data/catalog/
  -> /public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data/catalog/
```

The 2000--2003 velocity archives are not referenced by V01.  Existing
acceleration data, frozen split, and parent checkpoint are expected at the
paths documented in `docs/v01_velocity_prep_padding.md`; upload them only if
they are absent on the supercomputer.

## Verification actually run

PASS:

```text
python -m py_compile tools/prep_padding_protocol.py tools/velocity_waveform_backend.py tools/build_v01_paired_manifest.py tools/analyze_v01_padding_controls.py gemini_util_light.py loader_light.py train_light.py eval_checkpoint.py
bash -n tools/run_v01_prep_padding_controls_slurm.sh
python -m unittest tests.test_v01_velocity_backend tests.test_v01_padding_controls
  -> Ran 10 tests, OK
python -m unittest discover -s tests -p 'test_*.py'
  -> Ran 118 tests, OK
git diff --check
DRY_RUN=1 ACTION=all ARMS=vfull,vmissing,apair bash tools/run_v01_prep_padding_controls_slurm.sh
  -> preflight + 3 train + 10 val-eval + analyze dependency graph, exit 0
python tools/build_v01_paired_manifest.py ... --audit-only
  -> real frozen-split metadata audit completed
one-event 2024 materialization plus derived-cache generator read
  -> completed locally
```

After updating the defaults to the user's actual supercomputer upload paths,
the dry-run source manifest SHA256 is
`7cfc0a7049b16b9270298a69020847393e2ac423884fe87fbad3a0f3777838b9`.
Uploaded-folder users must still verify the value printed by the copy on the
cluster and pin that printed value for formal submission.

NOT RUN locally:

- full 2004--2024 cache materialization;
- a real RT55 epoch-32 checkpoint load/inference (the checkpoint is absent in
  this local result tree);
- single-DCU optimizer step, multi-rank DCU/DDP, resume on the cluster;
- the three full-cohort eight-epoch trainings and formal validation analysis;
- production FullModel masked-filler/zero-as-valid and fixed 0/1/3/5/full dose
  diagnostic jobs;
- A-historical stage-0 reevaluation and the complete probability/spatial plot
  suite requested by the experiment specification.

The last three items remain explicit review/follow-up gaps; the current
analyzer provides point-error summaries and paired event-cluster MAE intervals,
not the complete final scientific report.  They must not be described as
completed evidence.

## HPC status and next action

No Slurm job was submitted by Codex.  The user should first upload the pinned
branch/folder and data, run the launcher's dry run, then submit either all three
arms or the two velocity core arms using the exact commands in
`docs/v01_velocity_prep_padding.md`.  Return `V01_RUN_ROOT`, the job IDs and
`sacct` table for analysis.  Test data are not scheduled or read.
