# RT58 Waveform-Anchor Transfer Field

RT58 is an opt-in residual field head on the fixed RT57 epoch-6 model. It keeps
the entire loaded RT57 network frozen and trains only
`pga_anchor_transfer_head.*`. The official development run uses the existing
RT57 temporal residual at scale 1.66, one seed, and exactly eight new epochs.

## Model path

Terminal form:

```text
a_i = AnchorHead([u_i, d_i, z_e])

t_ji = tanh(||qcoord_j - x_i|| / s) * TransferMLP(...)

c_ji = a_i + t_ji

             exp(score_ji)
w_ji = ---------------------------
        sum_m exp(score_jm)

m_anchor_j = sum_i w_ji c_ji

delta_j = gate_j * (m_anchor_j - stopgrad(m_57_j))
m_final_j = m_57_j + delta_j
```

LaTeX:

```latex
a_i = \operatorname{AnchorHead}([u_i,d_i,z_e])

t_{ji}=\tanh\!\left(\frac{\lVert qcoord_j-x_i\rVert}{s}\right)
\operatorname{TransferMLP}(u_i,d_i,z_e,z_{new},q_j,g_{ji})

c_{ji}=a_i+t_{ji},\qquad
w_{ji}=\frac{\exp(score_{ji})}{\sum_m\exp(score_{jm})}

m^{anchor}_j=\sum_i w_{ji}c_{ji}

\Delta_j=gate_j\left(m^{anchor}_j-\operatorname{stopgrad}(m^{57}_j)\right),
\qquad m^{final}_j=m^{57}_j+\Delta_j
```

The set encoder contains two permutation-equivariant pre-norm self-attention
blocks followed by valid-station mean pooling. The pair decoder receives only
frozen waveform/event/query representations and observable station/query
coordinates. It never receives true source coordinates, source distance, or
input-station PGA labels.

The output gate's final linear layer is zero-initialized. Consequently, a newly
constructed RT58 model loaded from RT57 exactly preserves the RT57 gamma=1.66
prediction until optimization begins. Anchor and station-query candidate losses
still provide first-step gradients to the internal anchor, set, and transfer
branches.

The anchor correction is applied equally to every MDN component mean. Mixture
logits and component sigmas are unchanged.

## Training objective and coordinates

The existing final MDN NLL and predictive-mean loss are retained. RT58 adds:

- absolute input-station anchor Huber loss, weight 0.20;
- station-query candidate Huber loss, weight 0.05;
- final within-event query-difference Huber loss, weight 0.40;
- normal-replay distillation to the frozen RT57 mean, weight 0.05, only for
  samples on which random masking was not applied.

Input PGA and query PGA labels are raw `log10(m/s^2)` values supplied to loss
code. Anchor, candidate, field, and final model quantities use the configured
normalized PGA coordinate. A raw Huber transition `delta_dex` is converted as:

```text
delta_model = delta_dex / PGA_std
```

```latex
\delta_{model}=\frac{\delta_{dex}}{\sigma_{PGA}}
```

With the pinned training standard deviation `0.4312468402225416`, the configured
anchor/difference transition 0.15 dex is about 0.34783 model units, and the pair
transition 0.20 dex is about 0.46377 model units.

## Compatibility

- `use_pga_anchor_transfer=false` leaves the RT55/RT56/RT57 model path and
  evaluation schema unchanged.
- `pga_temporal_residual_scale` is a plain float with default 1.0 and adds no
  checkpoint state.
- RT57 checkpoints strict-load into RT58 only with the declared missing prefix
  `pga_anchor_transfer_head.`; all unrelated missing or unexpected keys remain
  fatal.
- `freeze_mode=anchor_transfer_only` freezes all prior parameters, asserts the
  sole trainable prefix, and prints every trainable tensor plus aggregate counts.
- The production head has 50 parameter tensors and 2,943,790 scalar parameters
  for station dimension 256, embedding dimension 1000, and hidden dimension 256.

## Evaluation artifacts

When RT58 is enabled, `eval_checkpoint.py` additionally exports:

- `pga_anchor_pred`
- `pga_anchor_transfer`
- `pga_anchor_candidate`
- `pga_anchor_station_weights`
- `pga_anchor_field_mean`
- `pga_anchor_applied_delta`

Absolute quantities are exported in raw `log10(m/s^2)`; transfer and applied
delta are scaled as raw dex differences. Old output files receive none of these
keys when RT58 is disabled.

## Slurm protocol

`tools/run_rt58_waveform_anchor_transfer_slurm.sh` defaults to `DRY_RUN=1` and
supports `ACTION=train|eval|all`. It submits one eight-epoch training job and two
dependent primary validation jobs (fixed random geometry and normal geometry).
The optional waveform-roll validation waits for both primary evaluations.
There is no held-out-test action and no epoch-selection workflow.

The default source check is an exact commit plus clean worktree. For a manually
uploaded source folder without `.git`, explicitly select `uploaded_sha256` and
provide the reviewed source-manifest digest. The launcher also checks the RT57
checkpoint SHA-256 before submission and verifies `epoch == 6` inside the
compute-node Python environment before training. Evaluation verifies the
official `full_model_last.pth` has `epoch == 8`.
