# RT59-v3 dual-objective residual transport

RT59-v3 is one fixed development experiment starting from the RT58 epoch-8
`full_model_last.pth`. It preserves the RT57 gamma=1.66 predictor and the
complete RT58 head as frozen modules. Only
`pga_anchor_residual_transport_head.local.*` and
`pga_anchor_residual_transport_head.transport.*` are optimized.

## Forward definition

The frozen public-query MDN is the baseline. Input-station coordinates are
decoded together as one additional lightweight query batch from cached
station/event state; the DiTing encoder and station adapter are not rerun.
This produces the input-coordinate base mean and predictive sigma. The frozen
RT58 anchor supplies the station residual relative to that base.

A valid public query with exactly one coordinate match to a valid selected
input station (`atol=1e-6` per current model-coordinate component, `rtol=0`)
uses the local correction. Queries with no match or duplicate ambiguous
matches use transport. Routing uses only forward-visible coordinates and
masks. Labels, target type and random/normal protocol metadata never enter the
forward route. Invalid queries and rows without a valid station receive zero
correction.

Only every mixture component mean is shifted. Mixture logits, component
sigmas and predictive mixture variance are unchanged. Enabling RT59 disables
application of the old RT58 field delta, while the old module remains
registered so the RT58 checkpoint loads strictly. Disabling the RT59 flag
preserves the old RT58 path and parameter schema.

## Objective and schedule

Valid targets are grouped as random non-input (R), normal unique observed
(NI), and normal other (NO). The group weights are fixed to 0.50/0.25/0.25,
with global DDP denominators. Main per-target loss is MDN NLL plus 0.10
SmoothL1 and 0.10 MSE. The normal truth-regret term has weight 0.05 and is zero
when the candidate improves over the frozen base. Relative-transfer,
candidate, and within-event difference losses apply only to R/NO.

Fresh initialization copies only RT58 `set_input`, `set_blocks`,
`pair_encoder`, and `score_head`. Resume loads the RT59 checkpoint and refuses
to repeat warm-copy. The local and transport parameter groups use Adam with
the RT58 defaults (betas 0.9/0.999, eps 1e-8, zero weight decay). Both start at
5e-4. The fixed eight-epoch schedule is 5e-4 for epochs 1–4, 2.5e-4 for 5–6,
and 1.25e-4 for 7–8. Each branch is clipped independently at norm 1.0.

## Slurm submission

The launcher defaults to `DRY_RUN=1`, accepts only `train`, `eval`, or `all`,
and never evaluates the held-out test split. `ACTION=all` submits one training
job and two `afterok` validation jobs. The random validation forward already
contains the fixed-context valid-u/d roll control, so there is no extra roll
job.

For a manually uploaded source folder, first obtain the source and checkpoint
digests without submitting:

```bash
cd /path/to/uploaded/team_pytorch_query_geometry_diagnostics
SOURCE_IDENTITY_MODE=uploaded_sha256 \
WORKDIR="$PWD" \
ACTION=all DRY_RUN=1 \
bash tools/run_rt59_dual_objective_transport_v3_slurm.sh

sha256sum /absolute/path/to/rt58/full_model_last.pth
```

Then submit the fixed run using the exact values printed above:

```bash
SOURCE_IDENTITY_MODE=uploaded_sha256 \
EXPECTED_SOURCE_MANIFEST_SHA256=3e1164bc4fb0441fb33e5d8cd03aa1708416708d930b01e71c4a82fd3779715e \
JAPAN_FULL_DATA_ROOT=/absolute/path/to/origin_corrected_diting_vel_acc_vs30 \
RT58_BASE_CHECKPOINT=/absolute/path/to/rt58/full_model_last.pth \
RT58_BASE_CHECKPOINT_SHA256=<64-char-checkpoint-sha256> \
RT59_WEIGHT_PATH="$PWD/weights_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42" \
WORKDIR="$PWD" ACTION=all DRY_RUN=0 CONFIRM_RT59=1 \
bash tools/run_rt59_dual_objective_transport_v3_slurm.sh
```

If only validation must be retried after a completed epoch-8 training job,
use `ACTION=eval`; the source identity is still mandatory, but the RT58 base
checkpoint digest is not needed.

After both NPZ files are available:

```bash
python tools/analyze_rt59_dual_objective_npz.py \
  --random_npz logs/<rt59-run>/epoch8_random_validation/eval_validation_epoch8_random.npz \
  --normal_npz logs/<rt59-run>/epoch8_normal_validation/eval_validation_epoch8_normal.npz \
  --output_dir reports/rt59_dual_objective_transport_v3_validation_20260915
```

The analyzer writes `summary.json`, `gates.csv`, `group_metrics.csv`,
`paired_ci.csv`, `strata_counts.csv`, `truth_prediction_density.png` (when
matplotlib is available), and `README.md`. The strata file reports actual
event/realtime-row/target/field counts by elapsed time, actual station count,
formal target type, and observable route. Its decision is conjunctive; missing required
probability evidence yields `INCOMPLETE_EVIDENCE`, never an implicit pass.

## 2026-09-18 first-submission correction

Job `27589430` exposed a bookkeeping-scope bug after the first backward pass:
the RT59 epoch accumulators had been initialized in the single-station
pretraining loop instead of `train_model`. Commit
`7e00824b3aab7a15286dfd9bf9a264b03080c1cd` moves them into the full-model
epoch scope and adds a regression that executes one complete tiny RT59 train
and validation epoch. The failed job reached no optimizer step and is not a
resumable RT59 run. Re-upload the corrected source and use a new empty
`RT59_WEIGHT_PATH`; do not resume or overwrite the failed output directory.
