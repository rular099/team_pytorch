# RT61 wave-geometry residual conditioning

Status: implementation and local verification complete; formal HPC run not submitted.

## Fixed experiment

RT61 starts only from the exact RT59-v3 epoch-8 parent:

- checkpoint SHA-256: `5dbc15c6af5b8d341dfd4a377a68217b1e1aa3589550690997c02e81332aa1cd`
- immutable residual-readout SHA-256: `db11b3c6152ea8ea76d78441769da04e6f03bb313050d70c64e4d12e1259f021`
- seed 42; all inherited 25 annual Japan 2000–2024 training shards and the existing event split
- exactly eight new epochs; only final epoch 8 is formally evaluated
- no held-out test, extra epoch, seed/rank/loss sweep, smoke, roll or checkpoint selection

The adapter is opt-in and fixed at rank 16. Terminal-readable definition:

```text
a[i]    = tanh(Ww · LN_no_affine(z[i]))
b[q,i]  = tanh(Wg · phi[q,i])
h1[q,i] = h0[q,i] + Wo · (a[i] ⊙ b[q,i])
```

Raw LaTeX:

```latex
a_i=\tanh(W_w\operatorname{LN}(z_i)),\quad
b_{qi}=\tanh(W_g\phi_{qi}),\quad
h^1_{qi}=h^0_{qi}+W_o(a_i\odot b_{qi}).
```

`Ww` and `Wg` use Xavier initialization, all projections have no bias, and `Wo`
is zero initialized. The immutable RT59 teacher consumes `h0`; only the student
readout consumes `h1`. Frozen `score_head` weights are also computed from `h0`.

Terminal-readable candidate difference:

```text
remote Delta[q] = sum_i frozen_weight[q,i] * frozen_distance_gate[q,i]
                  * (student_readout(h1[q,i], scalars)
                     - RT59_readout(h0[q,i], scalars))

mu_RT61[q,m] = mu_RT59[q,m] + remote Delta[q]
input Delta[q] = 0
alpha_RT61 = alpha_RT59
sigma_RT61 = sigma_RT59
```

Raw LaTeX:

```latex
\Delta_q=\sum_i w^{59}_{qi}d_{qi}
\left[r_\theta(h^1_{qi},s_{qi})-r_{59}(h^0_{qi},s_{qi})\right],\quad
\mu^{61}_{qm}=\mu^{59}_{qm}+\Delta_q.
```

Production dimensions yield an exact runtime-checked whitelist of 9 tensors and
99,621 scalars: 6/67,077 in the existing residual readout and 3/32,544 in the
adapter. All other parameters and buffers are included in the immutable shared-state
fingerprint. Each RT61 checkpoint also produces a small `*.rt61_delta.pth` package;
it still requires the exact parent and validates the shared-state fingerprint before
reconstruction.

## Objective and evaluation

The objective, sampler, mask/cutoff, normalization and MDN shape are unchanged from
RT60. Terminal-readable loss:

```text
L = grouped_point + 0.05 * remote_parent_regret
                  + 0.40 * field_contrast
```

Raw LaTeX:

```latex
\mathcal L=\mathcal L_{\mathrm{grouped\ point}}
+0.05\,\mathcal L_{\mathrm{remote\ parent\ regret}}
+0.40\,\mathcal L_{\mathrm{field\ contrast}}.
```

The evaluator exports unambiguous `val_rt61_reference_*`, `val_rt61_increment` and
`val_rt61_adapter_off_*` fields. `val_rt59_base_*` retains its historical RT57 meaning.
The adapter-off output uses the trained RT61 readout with the adapter disabled; it is
neither RT59 nor an independently trained RT60 model.

The formal analyzer adds per-field error decomposition. Terminal-readable identities:

```text
e[q] = pred[q] - truth[q]
field_MSE = mean(e)^2 + mean((e - mean(e))^2)

p_c = pred - mean(pred)
y_c = truth - mean(truth)
a_star = dot(p_c, y_c) / dot(p_c, p_c)
orthogonal_error = mean((y_c - a_star * p_c)^2)
```

Raw LaTeX:

```latex
\operatorname{MSE}(e)=\bar e^2+\frac1Q\sum_q(e_q-\bar e)^2,\quad
a^*=\frac{\langle p_c,y_c\rangle}{\langle p_c,p_c\rangle}.
```

Near-constant predictions are marked unidentifiable instead of stabilized with an
arbitrary epsilon. The projection is an offline truth-oracle diagnostic and is never
applied to formal predictions.

## Manual HPC submission

Upload the complete repository snapshot, enter that directory, and identify the exact
RT59 epoch-8 `full_model_last.pth`. The launcher defaults to dry-run and can operate in
an uploaded folder without Git metadata.

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics

export RT59_PARENT_CHECKPOINT=/absolute/path/to/rt59-v3-epoch8/full_model_last.pth

WORKDIR="$PWD" \
SOURCE_IDENTITY_MODE=uploaded_sha256 \
DRY_RUN=1 ACTION=all \
bash tools/run_rt61_wave_geometry_slurm.sh
```

Copy the printed `source_manifest_sha256`, then submit the single preregistered graph:

```bash
WORKDIR="$PWD" \
SOURCE_IDENTITY_MODE=uploaded_sha256 \
EXPECTED_SOURCE_MANIFEST_SHA256=<SHA256_PRINTED_BY_DRY_RUN> \
DRY_RUN=0 CONFIRM_RT61=1 ACTION=all \
bash tools/run_rt61_wave_geometry_slurm.sh
```

The launcher independently verifies the exact RT59 checkpoint SHA, resolved parent
contract, all 25 data shards, clean output directory, and source identity. It submits
one 4-node/16-DCU training job and two epoch-8 validation jobs with `afterok` dependency.
Its default wall time is `23:50:00`, below the cluster's observed one-day enforcement.

For a Git checkout, use `SOURCE_IDENTITY_MODE=git` and additionally set the full pushed
commit as `EXPECTED_GIT_COMMIT`.

## Formal analysis after the run

After the random and normal NPZ/metrics files and scalar CSV directory exist, run:

```bash
python tools/analyze_rt61_wave_geometry_npz.py \
  --random_npz /path/eval_validation_epoch8_random.npz \
  --normal_npz /path/eval_validation_epoch8_normal.npz \
  --random_metrics /path/eval_validation_epoch8_random.metrics.json \
  --normal_metrics /path/eval_validation_epoch8_normal.metrics.json \
  --training_config /path/weights_rt61/config.json \
  --random_config pga_configs/transformer_japan_full_2000_2024_rt61_wave_geometry_residual_seed42_chaosuan.json \
  --normal_config pga_configs/transformer_japan_full_2000_2024_rt61_wave_geometry_residual_seed42_normal_validation_chaosuan.json \
  --training_log_dir logs/weights_rt61/scalars \
  --expected_source_manifest_sha256 <SUBMITTED_SOURCE_SHA256> \
  --implementation_commit <FULL_GIT_COMMIT_OR_OMIT_VALUE> \
  --output_dir reports/rt61_wave_geometry_validation_YYYYMMDD
```

The analyzer fails closed on identity/count/schema differences and emits the 14 mechanism
gates, 25 required legacy gates, 4 diagnostics, the separately named
`useful_joint_progress`, event-cluster paired CIs, strata, per-field decomposition,
reference invariance, decision, training summary and figures. Formal analysis has not
been run yet because no RT61 HPC result exists.
