# AI-HANDOFF: RT58 waveform-anchor transfer field

[AI-HANDOFF]
task_id: 20260912-rt58-waveform-anchor-transfer-field
repo: rular099/team_pytorch
branch: rt58-waveform-anchor-transfer-field
base_commit: 143b63af5d0a477ac388d310dc470eb7349b5c90
goal: >
  Implement RT58 as a waveform-anchor transfer field on top of the fixed RT57
  epoch-6 checkpoint. Stop spending the next cycle on formal-test rigor or
  historical audit. Freeze the complete RT57 model, expose each input station's
  waveform-derived final-PGA anchor directly to a trainable station-to-query
  transfer decoder, and train only this new field head. The primary goal is to
  improve non-input PGA accuracy, slope and spatial dynamic range, especially
  with one actual input station, while preserving RT55/RT56/RT57 compatibility.

verified_facts:
  - Repository state for this handoff:
      repo: rular099/team_pytorch
      source branch: rt57-geo-temporal-neural-process
      base commit: 143b63af5d0a477ac388d310dc470eb7349b5c90
  - Effective RT57 implementation commit:
      49c6d30335c5e6df7cfbde1ef03caf784ae3aef6
  - RT57 was trained for one fixed six-epoch seed-42 run from RT56 and evaluated
    on validation only.
  - Random validation, 75,654 non-input targets, log10(m/s^2):
      RT56 MAE 0.257229
      RT57 MAE 0.247857
      RT57 RMSE 0.320040
      RT57 R2 0.360340
      RT57 slope 0.379428
      RT57 NLL 0.220260
      RT57 Brier 0.192239
  - RT57 improves all requested elapsed-time and requested station-count strata.
  - RT57 improves within-event field ordering and differences, but the mean
    predicted P95-P05 range remains 0.360140 versus true 0.831276.
  - With one actual selected input station and at least five query targets:
      RT56 mean per-field range ratio 0.005101
      RT57 mean per-field range ratio 0.125352
      RT57 pairwise-delta MAE 0.340582
      RT57 Pearson 0.323983
    The near-constant-output failure is substantially reduced but spatial
    amplitude remains severely compressed.
  - The RT57 station-distinctive local residual representation is strong:
      correct local residual MAE 0.094864, correlation 0.867133
      waveform-roll MAE 0.280022, correlation -0.043037
  - In contrast, replacing only the RT57 final residual delta with its rolled-
    waveform counterpart worsens full-model MAE by only 0.001647 overall and
    0.002234 on multi-station samples.
  - Therefore the current limiting factor is no longer simply "station-specific
    waveform information is absent". The new u/d representation contains useful
    station information, but the final query decoder uses it too weakly; geometry
    and event-common information dominate the residual.
  - Offline validation scaling suggests the RT57 residual is under-scaled:
      gamma=1.66 gives slope 0.400316, MAE 0.245578 and one-station range ratio
      0.207285. This can be used as a development initialization, without
      spending a separate cycle on a formal gamma-only rerun.
  - RT57 already exports/retains:
      station-distinctive absolute state u_i
      station residual d_i
      input-station PGA targets/validity during training/evaluation
      frozen event/query representations
      query/station coordinates
  - Original RT55 loading and inference compatibility remains mandatory.

inferences:
  - The highest-value next change is to route station-specific waveform amplitude
    into the query prediction explicitly, rather than asking a generic residual
    MLP to discover that route.
  - A local input-station PGA estimate is a physically meaningful waveform
    anchor. A learned transfer from that station to a query can represent source,
    path, direction, attenuation and regional/site priors without supplying true
    epicentral distance or source labels.
  - Since the local representation already predicts station PGA well, unfreezing
    DiTing is not yet justified. First test whether explicit anchor-to-query
    transfer uses the available information.
  - A stronger raw-dex within-event difference loss is justified because spatial
    amplitude compression is the dominant remaining failure.

files_to_inspect:
  - AGENTS.md
  - docs/ai/PROJECT_CONTEXT.md
  - docs/ai/README.md
  - SESSION_SUMMARY.md
  - docs/rt57_gtnp_v2_station_distinctive.md
  - docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md
  - gemini_models.py
  - train_light.py
  - gemini_util_light.py
  - eval_checkpoint.py
  - eval_checkpoint_slurm.sh
  - tests/test_rt57_gtnp_v2_station_distinctive.py
  - tests/test_causal_random_geometry.py
  - tests/test_eval_checkpoint_formal.py
  - pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json
  - pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_normal_validation_chaosuan.json
  - tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh

constraints:
  - Create branch rt58-waveform-anchor-transfer-field from the exact base commit.
  - Before editing, print pwd, remotes, branch, HEAD and git status.
  - Stop without reset/stash/clean if HEAD differs from base_commit or unrelated
    worktree changes exist.
  - This task supersedes the earlier gamma-only formal-validation handoff.
  - Do not spend this task on occurrence audits, extra smoke workflows, CI
    infrastructure or held-out test.
  - Preserve original RT55, RT56 and RT57 gamma=1 configs byte-for-byte.
  - All RT58 behavior must be opt-in through new parameters/config.
  - Disabled RT58 switches must reproduce the existing RT57 output.
  - Existing checkpoints must strict-load through an explicit allowed-missing
    prefix for the new RT58 head; unrelated missing/unexpected keys remain fatal.
  - Do not add true source coordinates, epicentral/hypocentral distance or
    propagation labels as model inputs.
  - Do not overwrite any existing experiment output.
  - Do not submit HPC jobs in the implementation commit.
  - First RT58 experiment is one fixed seed and fixed eight new epochs; do not
    inspect multiple epochs to choose a favorite checkpoint.
  - Freeze the complete loaded RT57 model. Only the new anchor-transfer head is
    trainable.
  - Leave MDN mixture logits and component sigmas unchanged in the first RT58
    experiment.

proposed_change:
  - Architecture name:
      RT58 Waveform-Anchor Transfer Field (RT58-WATF)

  - Add a new module under a new state-dict prefix:
      pga_anchor_transfer_head.*

  - Reuse, without unfreezing, the RT57 station-distinctive quantities:
      u_i = waveform-derived absolute station state
      d_i = event-centered station residual
      z_e = frozen RT57 event representation
      q_j = query representation
      x_i = input-station coordinates
      qcoord_j = query coordinates

  - Add a trainable input-station absolute-PGA anchor head:
      a_i = AnchorHead([u_i, d_i, z_e])

    Internal coordinate:
      normalized PGA coordinate used by the current model/loss.
    Exported diagnostics:
      raw log10(m/s^2).

    Train a_i on every valid input station, not only single-station events:
      L_anchor = Huber(a_i, y_input_i)

    This is a training label only. Input PGA labels must not enter inference.

  - Add a trainable event/source refinement set encoder over valid inputs:
      z_new = SetEncoder({u_i, d_i, x_i})
    Use two small pre-norm self-attention blocks or a permutation-invariant
    attention-pooling encoder. It is part of pga_anchor_transfer_head and is
    trainable. The frozen RT57 event embedding z_e remains an input.

  - For station i and query j, construct observable geometry:
      g_ji = Geometry(
        query absolute coordinates,
        station absolute coordinates,
        query - station,
        norm(query - station)
      )

  - Predict an anchor-to-query transfer:
      t_ji = TransferMLP([u_i, d_i, z_e, z_new, q_j, g_ji])

    Enforce transfer consistency at identical coordinates with a distance gate:
      t_ji = tanh(distance_ji / distance_scale) * raw_transfer_ji

    Use an explicit configurable distance_scale in existing model-coordinate
    units; default 0.05 unless current coordinate scaling implies a more
    appropriate documented value. Do not use true source distance.

  - Predict station/query reliability:
      score_ji = ScoreMLP([u_i, d_i, z_e, z_new, q_j, g_ji])

  - Form each station's query candidate:
      c_ji = a_i + t_ji

  - Aggregate:
                    exp(score_ji)
      w_ji = ----------------------------
             sum_m exp(score_jm)

      m_anchor_j = sum_i w_ji * c_ji

  - Fuse with the frozen RT57 predictive mean:
      m_57_j = frozen RT57 predictive mixture mean

      delta_anchor_j =
          gate_j * (m_anchor_j - stop_gradient(m_57_j))

      m_final_j = m_57_j + delta_anchor_j

    Gate:
      - query/event-conditioned or scalar-per-query;
      - final gate/output initialized exactly to zero, so initial RT58 output is
        identical to its RT57 base;
      - internal anchor/transfer branches still receive gradients from their
        auxiliary losses on the first step.

  - Apply delta_anchor_j equally to all existing MDN component means:
      logits unchanged
      sigma unchanged
      mu_new = mu_RT57 + delta_anchor_j

  - Add optional RT57 residual scale as a plain config float:
      pga_temporal_residual_scale=1.0 by default
    For RT58 config set it to 1.66. Do not create a trainable/state-dict scale.
    Do not run a separate gamma-only experiment.

  - Losses for RT58:
      main final MDN NLL: existing
      final predictive-mean loss: 0.1
      anchor absolute-PGA loss on all valid input stations: 0.20
      pair candidate loss:
        Huber(c_ji, y_query_j), all valid station-query pairs: 0.05
      within-event query difference loss: 0.40
      normal replay retention loss against frozen RT57 mean: 0.05 on normal
        branch only

    Replace ambiguous normalized Huber configuration for the difference loss
    with:
      huber_delta_dex = 0.15
    Convert this internally to normalized units using the configured PGA std.

    Pair candidate loss is an auxiliary; the final aggregate NLL remains the
    primary objective.

  - Trainability:
      Add freeze_mode=anchor_transfer_only.
      Freeze every pre-existing RT57 parameter, including its station-
      distinctive residual head.
      Only pga_anchor_transfer_head.* may require gradients.
      Print exact trainable names/counts and fail if another prefix is trainable.

  - New config:
      pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json

    It extends the RT57 config and sets at least:
      model_params:
        use_pga_anchor_transfer: true
        pga_temporal_residual_scale: 1.66
        anchor_transfer_station_dim: 256
        anchor_transfer_hidden_dim: 256
        anchor_transfer_set_layers: 2
        anchor_transfer_heads: 4
        anchor_transfer_distance_scale: 0.05
        anchor_transfer_zero_init: true

      training_params:
        load_model_path: ${RT57_BASE_CHECKPOINT}
        load_model_additional_allowed_missing_prefixes:
          - pga_anchor_transfer_head.
        transfer_model_path: null
        reinit_fpn: false
        freeze_mode: anchor_transfer_only
        lr_pga_anchor_transfer: 0.0005
        epochs_full_model: 8
        weight_path: ${RT58_WEIGHT_PATH}

        distribution_mean_loss:
          enabled: true
          weight: 0.1

        pga_anchor_absolute_loss:
          enabled: true
          weight: 0.20
          huber_delta_dex: 0.15

        pga_anchor_pair_candidate_loss:
          enabled: true
          weight: 0.05
          huber_delta_dex: 0.20

        pga_within_event_difference_loss:
          enabled: true
          weight: 0.40
          huber_delta_dex: 0.15
          max_pairs: 105

        normal_replay_distillation_loss:
          enabled: true
          weight: 0.05
          apply_to_random: false

        train_generator_overrides:
          causal_random_input_mask:
            enabled: true
            apply_probability: 0.80
            station_counts: [1,3,5,8,12,16]
            target_sampling:
              enabled: true
              input_ratio: 0.0
              triggered_noninput_ratio: 0.2
              untriggered_ratio: 0.5
              fill_missing: true
              exclude_inputs: true

        validation_generator_overrides:
          fixed deterministic 100% random geometry, unchanged from RT57.

  - New normal-validation companion config:
      pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_normal_validation_chaosuan.json

  - eval_checkpoint.py:
      Export, when enabled:
        pga_anchor_pred
        pga_anchor_transfer
        pga_anchor_candidate
        pga_anchor_station_weights
        pga_anchor_field_mean
        pga_anchor_applied_delta
      Export input PGA targets/validity as already supported.
      Preserve old output schema when RT58 is disabled.

  - New launcher:
      tools/run_rt58_waveform_anchor_transfer_slurm.sh

    Actions:
      train
      eval
      all

    Required behavior:
      DRY_RUN=1 by default
      explicit confirmation
      exact Git commit and clean worktree check
      exact RT57 epoch-6 checkpoint metadata and SHA-256 check
      fixed eight new epochs
      official checkpoint = full_model_last.pth at epoch 8
      validation only:
        fixed random geometry
        normal protocol
        optional waveform-roll control after the two primary evaluations
      no held-out test
      no checkpoint selection over epochs
      new output directories only

  - New tests:
      tests/test_rt58_waveform_anchor_transfer.py

    Cover:
      - disabled RT58 path exactly reproduces RT57
      - zero-init complete RT58 reproduces loaded RT57
      - old RT57 checkpoint loads with only declared new-head missing prefixes
      - only pga_anchor_transfer_head is trainable
      - station permutation with aligned waveform/coordinate permutation is
        invariant
      - waveform-only permutation changes anchor/candidate/final output
      - one station, same query, different waveform changes anchor and output
      - one station, same waveform, different query changes transfer/output
      - identical query/station coordinates produce zero transfer within
        numerical tolerance
      - input anchor loss uses all valid input stations
      - no input-PGA label leaks into forward inference inputs
      - pair candidate and field-difference masks are correct
      - huber_delta_dex conversion is correct
      - logits/sigmas remain unchanged
      - RT55/RT56/RT57 original config hashes remain unchanged

acceptance_checks:
  - Run and report:
      python -m py_compile gemini_models.py train_light.py eval_checkpoint.py
      bash -n tools/run_rt58_waveform_anchor_transfer_slurm.sh
      pytest -q tests/test_rt58_waveform_anchor_transfer.py
      pytest -q tests/test_rt57_gtnp_v2_station_distinctive.py
      pytest -q tests/test_causal_random_geometry.py
      pytest -q tests/test_eval_checkpoint_formal.py
      git diff --check
  - Show exact trainable parameter prefixes and counts.
  - Verify original RT55, RT56 and RT57 config hashes unchanged.
  - Verify initial RT58 output equals RT57 with gamma=1.66 base plus zero anchor
    correction.
  - Verify input labels are used only by loss code.
  - Show launcher dry-run with one eight-epoch training job and validation jobs,
    and no test job.
  - Present diff and tests before commit/push; push only after user approval.

hpc_followup:
  - One fixed seed-42 run, exactly eight new epochs.
  - Base checkpoint:
      exact RT57 epoch-6 full_model_last.pth, verified by metadata/hash.
  - Development evidence is validation only; do not spend this cycle on
    held-out-test formalism.

  - Primary random metrics:
      non-input MAE/RMSE/R2/slope/bias
      NLL/Brier/sigma/coverage
      >=5-target Pearson, pairwise-delta MAE and range ratio
      actual selected-station-count=1 range ratio and pairwise-delta MAE
      requested-1-s MAE
      waveform-roll penalty
      anchor absolute-PGA MAE/correlation
      pair candidate MAE
      learned anchor gate magnitude

  - Normal metrics:
      all-target and non-input MAE/RMSE/R2/NLL/Brier

  - Performance go criteria relative to the existing RT57 gamma=1.66 offline
    development baseline:
      1. random MAE <= 0.242
      2. random slope >= 0.45
      3. random R2 >= 0.39
      4. >=5-target mean range ratio >= 0.55
      5. actual-one-station mean range ratio >= 0.25
      6. actual-one-station pairwise-delta MAE <= 0.330
      7. requested-1-s MAE <= 0.253
      8. random NLL <= 0.22 and calibration not materially degraded
      9. normal all-target MAE <= 0.136
      10. normal non-input MAE <= 0.218

  - No-go:
      - If anchor prediction is accurate but final spatial metrics do not
        improve, the transfer/aggregation design is the bottleneck; redesign it
        before unfreezing DiTing.
      - If anchor prediction itself is poor, only then consider unfreezing the
        final DiTing interaction block in a subsequent task.
      - Do not claim success from overall MAE alone if one-station range and
        pairwise-delta criteria fail.

risks:
  - Input-station final PGA is available only as a training target, not an
    inference input. The anchor head must infer it from the causal waveform.
  - Very early windows may not yet contain the peak; anchor uncertainty is real.
    The pair scorer/event context should learn reliability, but the first RT58
    version keeps the old sigma fixed.
  - A per-pair auxiliary can overconstrain each station to predict every query.
    Its weight is intentionally small.
  - Validation is still one seed and may be optimistic; this task intentionally
    prioritizes performance iteration over final statistical certification.
  - Absolute site response at a completely unseen continuous location remains
    underdetermined without query-side geotechnical covariates.

open_questions:
  - none; implement the specified RT58 experiment and do not insert additional
    diagnostic prerequisites
[/AI-HANDOFF]

[CODEX-RESULT]
task_id: 20260912-rt58-waveform-anchor-transfer-field
base_commit: 143b63af5d0a477ac388d310dc470eb7349b5c90
result_commit: <full SHA or none>
branch: rt58-waveform-anchor-transfer-field
changed_files:
  - <path + summary>
verification:
  - <exact command>: <pass/fail/not run + output>
compatibility:
  - <RT55/RT56/RT57 unchanged behavior and config hashes>
hpc_status:
  - not submitted
remaining_risks:
  - <known implementation limitations>
review_request:
  - Review anchor-label isolation, normalized/raw PGA coordinates,
    station-to-query transfer semantics, zero-transfer-at-identical-coordinate
    behavior, aggregation masks, frozen RT57 trainability and RT55 compatibility.
[/CODEX-RESULT]
