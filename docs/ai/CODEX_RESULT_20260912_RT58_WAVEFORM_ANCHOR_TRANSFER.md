# CODEX-RESULT: RT58 Waveform-Anchor Transfer Field

task_id: `20260912-rt58-waveform-anchor-transfer-field`

base_commit: `143b63af5d0a477ac388d310dc470eb7349b5c90`

result_commit: `none` (implementation is intentionally awaiting review and push approval)

branch: `rt58-waveform-anchor-transfer-field`

## Implemented

- Added opt-in `PGAAnchorTransferHead` under the sole new checkpoint prefix
  `pga_anchor_transfer_head.*`.
- Reused frozen RT57 station `u_i/d_i`, event/query representations, and
  observable station/query geometry.
- Added two-layer permutation-equivariant station set encoding, waveform anchor
  regression, distance-gated station-to-query transfer, learned masked station
  aggregation, and an exactly zero-initialized final correction gate.
- Added plain-float `pga_temporal_residual_scale`, default 1.0; RT58 uses 1.66.
- Shifted MDN component means only; logits and sigmas remain unchanged.
- Added all requested RT58 losses, including raw-dex Huber conversion and
  normal-only replay distillation.
- Added `anchor_transfer_only` freezing, exact prefix assertion, and full
  trainable tensor/count reporting.
- Added random and normal RT58 configs, evaluation exports, unit tests, and a
  dry-run-first Slurm launcher supporting strict Git identity or explicit
  uploaded-source manifest identity.
- Added a compute-node epoch guard to `train_light_slurm.sh`; it is opt-in and
  therefore does not alter existing launch behavior.

## Changed files

- `gemini_models.py`: RT58 module, scale, integration, zero-gated MDN mean shift.
- `train_light.py`: optimizer/freeze mode, raw-dex conversion, anchor/candidate/
  replay losses, diagnostics.
- `eval_checkpoint.py`: conditional RT58 diagnostic exports.
- `train_light_slurm.sh`: optional compute-node load-checkpoint epoch guard.
- `pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json`:
  fixed RT58 training protocol.
- `pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_normal_validation_chaosuan.json`:
  normal validation companion.
- `tools/run_rt58_waveform_anchor_transfer_slurm.sh`: guarded training and
  validation submission.
- `tests/test_rt58_waveform_anchor_transfer.py`: RT58 behavior/compatibility tests.
- `docs/rt58_waveform_anchor_transfer.md`: architecture and operating notes.

## Verification

- `python -m py_compile gemini_models.py train_light.py eval_checkpoint.py`: pass.
- `bash -n train_light_slurm.sh`: pass.
- `bash -n tools/run_rt58_waveform_anchor_transfer_slurm.sh`: pass.
- `pytest -q tests/test_rt58_waveform_anchor_transfer.py`: pass, 11 tests.
- `PYTHONPATH=. pytest -q tests/test_rt57_gtnp_v2_station_distinctive.py`: pass, 10 tests.
- `PYTHONPATH=. pytest -q tests/test_causal_random_geometry.py`: pass, 7 tests.
- `PYTHONPATH=. pytest -q tests/test_eval_checkpoint_formal.py`: pass, 6 tests.
- `git diff --check`: pass.
- Uploaded-source launcher dry run: pass; exactly one eight-epoch train job, two
  primary validation jobs, optional waveform-roll after both, no test job.

The local standalone `pytest` executable does not add the repository root to
`sys.path` for the three pre-existing test modules; setting `PYTHONPATH=.` (or
using `python -m pytest`) is required in this environment. This is a runner-path
issue, not a test failure.

## Compatibility evidence

- Original RT55 config SHA-256:
  `bb28ce3b66a6bd389e6ccd2cb53062c1e68cdb603535f9930c75ac99a33d1c8c`
- Original RT56 config SHA-256:
  `92fe9f0f0942ae7c7e75ae4351c9cd90cfd274dcbbbfd474b5a6648994fb8015`
- Original RT57 config SHA-256:
  `0b4cb2f0b1ebec6d81c59a64022bf688893a17ded3260a834633e03128dd9f47`
- All three files remain byte-for-byte unchanged.
- Zero-init RT58 matches a loaded RT57 gamma=1.66 base exactly in the tiny
  end-to-end model test.
- The default scale=1.0 follows the original RT57 arithmetic branch without an
  added multiply.
- Input-station PGA values occur only in loss/evaluation metadata code and are
  absent from the RT58 forward signature.

## HPC status

Not submitted. The implementation task explicitly forbids submitting an HPC job
before review. The launcher accepts the user's manually uploaded-folder workflow
through `SOURCE_IDENTITY_MODE=uploaded_sha256`; no GitHub connection is needed
on the supercomputer.

## Remaining risks

- The new head keeps RT57 mixture uncertainty fixed by design. Very early
  waveform windows can therefore improve or worsen the mean without adapting
  sigma.
- Pair supervision may overconstrain a weak individual station despite its low
  0.05 weight.
- The zero final gate means internal branches learn immediately from auxiliary
  losses, while the main NLL initially updates the gate before it can strongly
  update upstream field features.
- Only the fixed eight-epoch validation run can establish whether spatial range,
  one-station transfer, and calibration meet the handoff gates.

## Review request

Please review anchor-label isolation, raw/normalized PGA conversions,
station-to-query transfer semantics, exact zero transfer at identical
coordinates, mask-safe aggregation, frozen RT57 trainability, RT55 compatibility,
and the decision to keep logits/sigmas fixed. If accepted, authorize commit/push
and run the single prescribed RT58 seed-42 eight-epoch validation workflow.
