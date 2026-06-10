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
  pga_configs/transformer_japan_overfit_pga15_stage2_512_rt34_dpk_event_station_pool_chaosuan.json
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

- `sample_station_key = split|dataset_id|sample_index|station_slot`
- `event_station_time_key = split|dataset_id|event_id|realtime_current_sample|original_station_index`

For deterministic generators, `sample_station_key` is the fastest training-time
lookup. `event_station_time_key` is more explicit and should be used when
checking cache consistency across runs.

Recommended training-time read pattern:

```python
import h5py

h5 = h5py.File("dpk_priors.h5", "r")
key_to_row = {
    key.decode() if isinstance(key, bytes) else key: int(row)
    for key, row in zip(h5["index/sample_station_key"][:], h5["index/h5_row"][:])
}
row = key_to_row[f"{split}|{dataset_id}|{sample_index}|{station_slot}"]
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

If DPK fine-tuned priors are clearly sharper or less uniform than `mae_head`,
the next step is to wire `dpk_priors.h5` as a training-time cache and run a
cached-prior version of rt34/rt35.
