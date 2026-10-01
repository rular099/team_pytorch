# FE01 repository audit — 2026-10-01

Repository `rular099/team_pytorch`; implementation worktree `team_pytorch_fe01`,
branch `exp/fe01-feature-extractor-site-effects`, clean base
`9c95dbfaf92f36b2673d816026cd3f25b91eec66`. Source implementation SHA is reported
in the final CODEX-RESULT and source bundle SOURCE_IDENTITY.json.

## Workspace protection

Original `team_pytorch_query_geometry_diagnostics` was on
`exp/v01-velocity-prep-padding-control`, HEAD
`8f54477ed11bf2623f40121648f7fcc756ada1a2`, with uncommitted V01 recovery changes
and untracked archive/presentation artifacts. Original `team_pytorch` was on
`zhangb/native-scale-adapter-scaling`, HEAD
`8bde65cb64fd860adf552dfe56505b4b31a78b13`. Both heads, short status, staged and
unstaged tracked diffs were snapshotted read-only before creating this worktree.
No checkout/stash/reset/clean/merge was performed there. Final exact snapshot
comparison is in FE01_VERIFICATION.md. V01 code/data/outputs are excluded.

## Code trace and verified facts

- RT55 config fully resolves through its existing inheritance chain. The width is
  **1000**, max input stations 25, targets 15; it uses target/event cross-attention,
  PGA MDN and magnitude/location heads. use_vs30 and all later residual/teacher,
  DPK and temporal enhancement paths are off. Static data availability does not
  mean VS30 is fed to the model.
- Existing realtime generator already uses random_bins and 3 draws/event/epoch;
  fixed 1/3/5/10/20/40/90 s are validation times. Its frozen encoder executes
  inside forward; RT55 DPK prior cache is disabled.
- `gemini_util_light.py` loads and centers the complete record, crops with a
  sampled P anchor, selects realtime cutoff as exclusive current+1, then adjust_mean
  uses `:cutout+1`. The integrated synthetic counterexample uses the actual crop
  offset and first forbidden original sample. A lightweight synthetic encoder in
  legacy FullModel propagates the leak to features/predictions. This does not
  quantify any historical checkpoint's metric impact.
- `gemini_models.FullModel` owns the common position/context/readout/MDN/auxiliary
  architecture and scale projection. The only legacy edit injects optional
  station_waveform_model/full_model_class into its factory. Legacy defaults,
  parameter keys/shapes, preprocessing, losses and all original configs remain
  untouched. Dedicated FE01FullModel owns the stricter preprocessing.
- Actual DiTing YAML is `diting_1200m_backbone_attnpool.yml`: MAE, 10000 samples,
  patch 50, depth 24, width 1792 parsed from hps. RT55's inherited model_params
  diting_frontend string says vit_adapter; it is stale relative to the explicit
  YAML/build_diting_args route. FE01 builds from that YAML and reuses its
  BackboneAttentionPoolAdapter before positional fusion, not the stale string.
  Encoder state is strictly checked after the existing MAE remapping; partial
  checkpoint loading fails. Full encoder is frozen/eval; adapter remains trainable.
- Original TEAM source was inspected locally at sibling TEAM/models.py. Legacy
  torch NormalizedScaleEmbedding has tuple/string activation/flatten/scale bugs.
  FE01 uses a separate corrected original CNN port, with common physical scale
  replacing original ln-scale concat. Both original TEAM and dtbench local
  LICENSE snapshots are GPL v3; the source package retains the dtbench license.
- Native PhaseNet bottleneck and EQT complete shared trunk were traced against
  installed SeisBench **0.10.1**, including explicit padding and transformer tuple
  outputs, and tested against official model hooks. STEAD **version 2** was
  advertised for both and downloaded from the official GFZ mirror; component
  metadata is ZNE. No runtime from_pretrained/annotate/classify is used. Weight
  metadata uses peak normalization; common FE01 masked/joint peak differs and is
  explicitly recorded. Original/old weight normalization cannot be silently used.

## Data path and pending facts

`split_catalog → read_event → native window + masks → causal geometry → normalized
frontend + physical scale → unchanged RT55 downstream → original loss/evaluation`.
Year shards resolve by basename from the frozen RT55 event manifest. Dev is named
validation; duplicate event/split identities fail. HDF5 expects `waveforms`,
`coords`, `p_picks`, `pga`, `record_start_sample`, `valid_n_samples`, source_network
and station_codes, metadata sampling_rate=100. knt is KNET; kik is excluded.
Observed zeros are valid, masks follow recorded support. Optional component/sample
masks tighten it conservatively. Naive explicitly JST-named timestamps are
localized Asia/Tokyo; UTC-offset strings are converted to UTC. Global record time
axes must agree within one sample. Origin replay requires recorded axis/origin.

Japan production 2000–2024 shards and actual DiTing 1200M checkpoint were not
available to this implementation run. Historical documented 9627/1383/2769
train/validation/test counts are **not** re-certified. Production units, metadata,
cohort, hashing, strict weight keys/shapes, causal forward, timings and device
memory must pass manual HPC audit/pilot. No test labels were read locally.
Synthetic HDF5 is a separate generated fixture, not Japan evidence. Upstream
offline resampling/filtering and real trigger latency remain uncertified.

## Launcher audit

Legacy train_light_slurm.sh has a hardcoded WORKDIR, unconditional DiTing weight
path, mainline partition/environment defaults, AUTO_SBATCH default 1 and posttrain
RUN_EVAL default 1; RESET_WEIGHT_PATH is 0 but can delete with flags. FE01 does not
call this launcher. New scripts keep its working Slurm-direct rank/LOCAL_RANK
convention and DCU GRES allocation, with user-filled modules/account/resources,
RESET_WEIGHT_PATH=0, AUTO_SBATCH=0, no automatic evaluation/resubmission/deletion.
print_submit_commands creates log directories before printing dynamic sbatch
arguments. Device torch API uses cuda for ROCm; no NVIDIA/CUDA install is assumed.

External dependency dtbench is not included in the baseline Git repository.
Source bundling includes an exact per-file hashed vendor snapshot and license;
the bundle and dependency identity are audited. A dedicated environment shares
working cluster torch/ROCm, pins SeisBench and checks dependency presence. A local
CPU environment cannot validate the cluster stack.

Official sources consulted:
[PhaseNet implementation](https://seisbench.readthedocs.io/en/stable/_modules/seisbench/models/phasenet.html),
[EQT implementation](https://seisbench.readthedocs.io/en/stable/_modules/seisbench/models/eqtransformer.html),
[weight catalog](https://seisbench.readthedocs.io/en/stable/pages/models/pretrained_models.html),
[original TEAM](https://github.com/yetinam/TEAM).
Execution uses locked installed source/weights, not changing stable documentation.
