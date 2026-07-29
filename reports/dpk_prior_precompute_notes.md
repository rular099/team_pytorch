# DPK Prior Precompute Notes

## Purpose

This utility precomputes DPK-derived temporal priors for the station adapter.
It is meant to test whether a full DPK fine-tuned encoder provides sharper or
meaningfully different eventness priors than the current runtime path used by
rt34/rt35/rt39.

Two prior sources are computed for the same waveform windows:

- `mae_head`: current MAE/pretrained DiTing encoder plus the DPK task head.
- `dpk_finetuned`: DPK fine-tuned DiTing encoder plus the same DPK task head.

For each source, the script writes both:

- `event`: DPK `det`/eventness probability only.
- `all`: max over `det`, `ppk`, and `spk`.

Each signal is resampled to the station-adapter token lengths for `f2`, `f3`,
`f4`, and `x`, then normalized exactly like the online soft-bias path:

```text
prior = clamp(signal, floor)
prior = prior / mean(prior)
```

The mean normalization makes the prior scale stable. In the attention soft-bias
path, only relative variation across tokens matters.

## Running

Default single-node 4-card run:

```bash
bash tools/precompute_dpk_priors_slurm.sh \
  pga_configs/transformer_japan_overfit_pga15_stage2_512_rt34_fixedtime_dpk_event_station_pool_chaosuan.json
```

Useful overrides:

```bash
SPLIT=dev STATION_BATCH_SIZE=2 \
bash tools/precompute_dpk_priors_slurm.sh <config.json> <output_dir>
```

The DPK checkpoint default is:

```text
/public/home/test_bigmodel/seismogram/mx/ckpt/1200m_dpk/mae_init/720w_sft/model-4-latest.pth
```

## Outputs

The output directory contains:

- `dpk_priors.h5`: merged training cache. This is the default file to read
  during cached-prior training.
- `dpk_priors_rank*.npz`: per-rank intermediate shards plus row-aligned
  metadata. These are mainly for debugging and recovery.
- `station_prior_stats_rank*.csv`: per-station effective token counts and
  `mae_head` vs `dpk_finetuned` difference metrics.
- `cache_index.csv` and `cache_index.npz`: global lookup tables mapping each
  cached station record to `(hdf5_path, h5_row)` and, for debugging, also to
  `(shard_path, row_in_shard)`.
- `summary.json`: global mean effective counts and source comparison metrics.
- `model_compare.json`: exact encoder weight comparison and runtime policy.
- `manifest.json`: paths to all shards and diagnostics.

If `model_compare.json` reports `all_equal=true`, the script records
`runtime_policy=shared_current_encoder` and only keeps one encoder in GPU
memory. If `all_equal=false`, it records `runtime_policy=separate_dpk_encoder`
and computes priors with two separate encoder forward passes.

The HDF5 cache is row-aligned: row `i` of any dataset under `/priors`,
`/meta`, `/index`, and `/stats` refers to the same station record. The most
useful lookup keys are:

- `event_station_time_key = split|dataset_id|event_id|realtime_current_sample|original_station_index`
- `sample_station_key = split|dataset_id|sample_index|station_slot`

For realtime training, `event_station_time_key` is the training-time lookup key.
It identifies the actual event cutout and original station, so it is robust to
DataLoader ordering and avoids false hits when `sample_index` changes across
runs. `sample_station_key` is kept as a debugging key and fallback for
non-realtime datasets only.

Recommended training-time read pattern:

```python
import h5py

h5 = h5py.File("dpk_priors.h5", "r")
key_to_row = {
    key.decode() if isinstance(key, bytes) else key: int(row)
    for key, row in zip(h5["index/event_station_time_key"][:], h5["index/h5_row"][:])
}
row = key_to_row[
    f"{split}|{dataset_id}|{event_id}|{realtime_current_sample}|{original_station_index}"
]
prior = h5["priors/dpk_finetuned/event/x"][row]
```

The merged HDF5 defaults to `float32` prior arrays and no compression. This is
larger than compressed NPZ but more stable across environments and faster for
random row reads in training.

## What To Check

The first diagnostics to inspect are:

- `*_event_x_effective` and `*_event_f2_effective`: lower values mean a sharper
  eventness prior; values close to the full token count mean the prior is almost
  uniform.
- `compare_event_x_corr` and `compare_event_f2_corr`: high correlation means the
  two prior sources behave similarly after token resampling.
- `compare_event_x_l1` and `compare_event_f2_l1`: small values mean the two
  normalized priors differ little in magnitude.

DPK fine-tuned priors are expected to be the preferred source when they are
clearly sharper or less uniform than `mae_head`. The training path now supports
using `dpk_priors.h5` directly as a station-token prior cache.

Current follow-up:

- rt40 to rt43 are a `token_weight_scale` sweep on cached
  `dpk_finetuned/event` station-pooling priors: scale 0, 2, 4, and 8.
- This sweep tests whether the prior bias is too weak relative to the learned
  attention score. The configs now consume `dpk_priors.h5` through
  `training_params.dpk_prior_cache`.
- Keep `dpk_weight_temperature=1.0` in this sweep. Sweeping temperature and
  scale together is redundant because both mainly change the log-prior bias
  strength.
- Cached-prior training passes row-aligned station token weights as an extra
  tensor in `inputs`. The lookup key is
  `split|dataset_id|event_id|realtime_current_sample|original_station_index`.
  The rt40-rt43 configs use fixed train realtime times
  `[1, 3, 5, 10, 20, 40, 90]` and point to a fixed-time rt34 cache anchor. This
  makes cache lookup an exact match instead of trying to cover the continuous
  random-cut space. In fixed-time mode the training sample key is stable across
  epochs as `event + fixed_time`; do not include epoch-dependent randomness in
  station selection unless the cache is regenerated to cover those variants.
- Precompute both train and dev split caches. The scale-sweep configs use
  `missing_policy=error` so that missing train/dev rows fail fast instead of
  silently turning into no-prior samples.
- Do not reuse the earlier random-cut rt34 cache for rt40-rt43. Its
  `(event, current_sample, station)` coverage is not aligned with fixed-time
  training and will miss rows.
- 2026-06-16 fix: `_crop_aligned_event_window()` previously chose the crop
  anchor with global `np.random.choice`, so `deterministic_sampling=true` did
  not actually make realtime samples deterministic. This made precomputed cache
  rows and later coverage/training rows disagree even when config and explicit
  cache paths were correct. Regenerate fixed-time train/dev caches after this
  fix; old fixed-time caches generated before the fix should be discarded.

Cache commands:

```bash
SPLIT=train bash tools/precompute_dpk_priors_slurm.sh \
  pga_configs/transformer_japan_overfit_pga15_stage2_512_rt34_fixedtime_dpk_event_station_pool_chaosuan.json

SPLIT=dev bash tools/precompute_dpk_priors_slurm.sh \
  pga_configs/transformer_japan_overfit_pga15_stage2_512_rt34_fixedtime_dpk_event_station_pool_chaosuan.json
```

Before launching rt40-rt43, run an exact coverage check against the generated
HDF5 files. Do not run this on the login node; submit it through Slurm:

```bash
bash tools/check_dpk_prior_cache_coverage_slurm.sh \
  pga_configs/transformer_japan_overfit_pga15_stage2_512_rt40_dpk_event_station_pool_scale0_chaosuan.json train

bash tools/check_dpk_prior_cache_coverage_slurm.sh \
  pga_configs/transformer_japan_overfit_pga15_stage2_512_rt40_dpk_event_station_pool_scale0_chaosuan.json dev
```

Coverage must be `1.00000000`, and
`station_retention_after_cache_filter` should be close to `1.00000000`. If
coverage is lower, regenerate the cache before training. A miss such as
`train|0|20240113155600|1160|0` means the train cache does not contain that
event/current-sample/original-station row; it is not fixed by rerunning rt40
unless the cache itself is regenerated with the same config and event-id file.

The wrapper writes detailed output to
`logs/dpk_prior_cache_coverage/<config>_<split>_<jobid>.txt`. It infers the
cache path from `training_params.dpk_prior_cache.paths[split]`; pass an explicit
third argument only when checking a cache that is not referenced by the config.
