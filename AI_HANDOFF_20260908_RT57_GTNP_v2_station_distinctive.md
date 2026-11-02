# RT57-GTNP v2: explicitly address station-feature collapse

This handoff supersedes `AI_HANDOFF_20260908_RT57_Geo_Temporal_Neural_Process.md`.

[AI-HANDOFF]
task_id: 20260908-rt57-gtnp-v2-station-distinctive
repo: rular099/team_pytorch
branch: rt57-geo-temporal-neural-process
base_commit: e74f977271b0bcde42c7ac7a18cb77ac60107e34
goal: >
  Implement RT57-GTNP v2 for arbitrary-input-station to arbitrary-query PGA
  probabilistic prediction. Preserve the useful RT56 event/common representation,
  but explicitly add a trainable station-distinctive waveform branch inside a
  zero-output-initialized residual field head. The new branch must consume
  pre-pooling DiTing temporal tokens plus explicit waveform amplitude/duration
  features, separate event-common and station-residual information, and combine
  them with query-minus-station geometry. This supersedes the earlier RT57 v1
  handoff, which only partially bypassed pooled-feature collapse.

verified_facts:
  - Source state:
      repo: rular099/team_pytorch
      branch: query-geometry-diagnostics
      commit: e74f977271b0bcde42c7ac7a18cb77ac60107e34
  - Prior experiments reported same-event station waveform embeddings with
    pairwise cosine similarity above 0.99 under multiple pooling methods.
  - High raw cosine alone does not prove zero information, because a large
    event-common component can dominate a smaller station-specific residual.
    However, it is a plausible performance bottleneck when downstream station
    keys/values cannot reliably associate waveform differences with station
    coordinates.
  - Current code's DitingStationAdapter.temporal_tokens(inputs) returns the final
    DiTing temporal sequence `x` before the adapter's final station-vector
    pooling.
  - TargetConditionedTemporalPool already consumes temporal tokens and constructs
    pair conditioning from query embedding, station embedding, event embedding,
    query coordinates, station coordinates, relative coordinates, and distance.
  - PGATemporalResidualHead can add a query-specific scalar shift to every MDN
    component mean while preserving RT56 mixture logits and sigmas.
  - The current RT57-v1 concept freezes the old encoder/adapter and therefore
    does not change the existing pooled station-vector collapse. It can bypass
    that pooling through temporal tokens, but if the temporal tokens are also
    event-dominated it may learn mostly geometry/site priors.
  - RT55 loading/inference compatibility is a hard constraint.

inferences:
  - The model should represent two different objects rather than forcing one
    embedding to do both:
      event-common representation: source/event information shared by stations
      station-distinctive representation: arrival, amplitude, spectrum, path and
      local-site information that differs between stations
  - The station-distinctive branch should be newly trainable while the complete
    existing RT56 model remains frozen. This directly attacks pooling collapse
    without destabilizing the RT56 checkpoint.
  - Raw cosine should be monitored, not directly minimized. A local-PGA-residual
    auxiliary target is more task-relevant than an arbitrary decorrelation loss.

files_to_inspect:
  - AGENTS.md
  - docs/ai/PROJECT_CONTEXT.md
  - docs/ai/README.md
  - SESSION_SUMMARY.md
  - docs/rt56_random_geometry.md
  - gemini_models.py
  - train_light.py
  - gemini_util_light.py
  - eval_checkpoint.py
  - pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
  - pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
  - existing tests for temporal residual, station-local PGA, checkpoint
    compatibility, causal random geometry and formal evaluation

constraints:
  - Create `rt57-geo-temporal-neural-process` from the exact base commit.
  - Print pwd, remote, branch, HEAD and git status before editing.
  - Stop on base mismatch or unrelated worktree changes; do not reset/stash/clean.
  - This v2 handoff supersedes the previous RT57 v1 handoff.
  - Preserve RT55 and RT56 configs byte-for-byte.
  - Preserve all old defaults, parameter names, tensor shapes and inference
    outputs when the new RT57 switches are disabled.
  - Do not add true hypocentral/epicentral distance, source labels or path labels
    as inference inputs.
  - Do not use held-out test.
  - Do not submit HPC jobs in the implementation commit.
  - Do not make further historical NPZ auditing a prerequisite.
  - Keep the existing RT56 model, including its encoder, adapter, readout and MDN,
    frozen during the first RT57 run.
  - All new trainable modules must live under a new residual-head prefix so
    trainability can be isolated and checkpoint initialization can explicitly
    allow only that missing prefix.

proposed_change:
  - Architecture:
      RT57 Geometry-conditioned Target-Temporal Neural Process v2.

  - Add a new `StationDistinctiveTokenAdapter` inside
    `pga_temporal_residual_head`.

  - Inputs for station i:
      T_i: pre-pooling final DiTing temporal tokens
      A_i: explicit log-amplitude statistics from the original masked waveform
           (per-channel std/RMS/peak and station summaries)
      D_i: explicit valid-duration features, at least valid fraction or valid
           sample count; use post-P duration only if already available without
           changing model inference inputs
      M_i: temporal validity mask

  - Suggested station adapter:
      tokens = LayerNorm(T_i)
      tokens = two lightweight masked depthwise-separable temporal residual
               blocks with dilations 1 and 4
      pooled = concatenate(
          masked mean(tokens),
          masked std(tokens),
          four-query attention pool(tokens)
      )
      u_i = LayerNorm(MLP([pooled, A_i, D_i]))  # absolute station waveform state

    Keep dimensions moderate:
      temporal internal dim: 256
      station-distinctive output dim: 256
      attention-pool queries: 4
      dropout: 0.0 for the first controlled run

  - Explicit common/residual decomposition:
      e_u = masked_mean_i(u_i)
      d_i = u_i - stop_gradient(e_u) by default

    For one input station, d_i is zero but u_i remains available. Pair messages
    must use both u_i and d_i, never d_i alone.

  - Geometry-conditioned pair representation:
      g_ji = Fourier/MLP encoding of:
        query absolute coordinates
        station absolute coordinates
        query - station
        horizontal/3-D norm as already represented in model coordinates

      h_ji = MLP([
        query embedding_j,
        frozen RT56 event embedding,
        u_i,
        d_i,
        g_ji
      ])

  - Query-conditioned temporal value:
      temporal_ji =
        TargetConditionedTemporalPool(T_i | h_ji, M_i)

      pair_ji = PairValueMLP(h_ji)

      value_ji = temporal_ji + pair_value_gate * pair_ji

      message_j =
        sum_i softmax_i(score(h_ji)) * value_ji

      delta_mu_j = zero-output-initialized head(message_j)

      The final MDN remains:
        logits_new = logits_RT56
        mu_new = mu_RT56 + delta_mu_j
        sigma_new = sigma_RT56

  - Important implementation detail:
      Do not let PairValueMLP depend only on the old pooled station embedding and
      geometry. It must directly include the new u_i/d_i waveform summaries.
      Otherwise the model can improve using coordinate priors while leaving the
      waveform-feature-collapse problem untouched.

  - Explicit amplitude path:
      Reuse or expose the same masked raw-waveform amplitude statistics already
      computed for waveform scale embedding. Pass the uncompressed feature vector
      to StationDistinctiveTokenAdapter in addition to any frozen 500-D station
      embedding. Do not rely on amplitude information remaining recoverable after
      addition and LayerNorm.

  - Task-relevant station auxiliary loss:
      Add a small head from d_i (and u_i for single-station cases) to predict the
      input-station final-PGA residual relative to the valid input-station event
      mean:

        target_local_i = PGA_input_i - mean_valid_inputs(PGA_input)

      For one input station, either skip the centered residual loss or use a
      separate low-weight absolute local-PGA target from u_i.

      Suggested weights:
        multi-station local residual loss: 0.10
        single-station absolute local PGA loss: 0.02

      Labels may be used only as training targets; they must not enter inference
      inputs.

  - Main training losses:
      final MDN NLL: existing
      predictive-mean auxiliary: existing weight 0.1
      temporal/pair residual auxiliary: 0.2
      within-event query PGA-difference Huber loss: 0.2
      station-local residual auxiliary: 0.1
      optional single-station local absolute auxiliary: 0.02

    Use all valid query pairs (at most 105 for 15 targets).

  - Do not add a raw-cosine decorrelation penalty in the first run.
    Instead log:
      old pooled station cosine
      new absolute u_i cosine
      new centered d_i cosine
      ||d_i|| / ||u_i||
      effective rank of u_i and d_i over valid stations where feasible
      local-PGA-residual auxiliary MAE/correlation
      pair-value norm
      temporal-value norm
      pair/temporal ratio
      delta_mu mean/std/P95-P05 range

  - Required waveform-dependence controls in tests/evaluation:
      1. Keep coordinates fixed and permute waveforms among input stations:
         output must change for synthetic nonidentical station tokens.
      2. Permute waveform, amplitude features and station coordinates together:
         set output must remain permutation invariant.
      3. Zero temporal tokens and amplitude features:
         the new waveform-conditioned contribution must disappear, while the
         pure geometry path can remain.
      4. Same query/station geometry with different station token sequences:
         residual output must differ.
      5. Exactly one valid station with two different query coordinates:
         residual output must differ.
      6. Exactly one valid station with identical query coordinates but different
         waveforms:
         residual output must differ.

  - Trainability:
      Extend `freeze_mode=temporal_residual_only` or add an explicit
      `station_distinctive_residual_only` mode.
      Only modules under the new residual head, including
      StationDistinctiveTokenAdapter, pair scorer/value, temporal pool and final
      residual output, may be trainable.
      All original RT56 parameters must remain frozen.

  - Initialization:
      Internal new modules may use normal/Xavier initialization.
      The final scalar PGA residual output layer must be exactly zero initialized,
      so before training the complete RT57 output equals the loaded RT56 output.
      Test exact/near-exact equality.
      Confirm gradients reach internal station-distinctive modules after the
      output layer begins updating; a two-optimizer-step synthetic test is
      acceptable.

  - New config:
      `pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json`

    It extends RT56 and enables:
      use_pga_temporal_residual: true
      station_distinctive_adapter: true
      station_distinctive_use_amplitude_features: true
      station_distinctive_use_duration_features: true
      station_distinctive_common_residual_decomposition: true
      temporal_pool_pair_value_enabled: true
      pga_temporal_residual_mode: residual
      pga_temporal_residual_use_event_context: true
      pga_temporal_residual_station_weighting: learned_pair
      freeze_mode: station_distinctive_residual_only
      load_model_path: ${RT56_BASE_CHECKPOINT}
      epochs_full_model: 6
      lr_pga_temporal_residual: 5e-4
      causal random training probability: 0.75
      normal replay probability: 0.25
      fixed random validation probability: 1.0
      station counts: [1,3,5,8,12,16]

  - New launcher:
      `tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh`
    It must train fixed six epochs from the verified selected RT56 checkpoint,
    then evaluate the fixed epoch-6 checkpoint on random validation and normal
    validation only. No held-out test.

  - New tests:
      `tests/test_rt57_gtnp_v2_station_distinctive.py`
    Cover every behavior listed above plus RT55/RT56 unchanged-default
    compatibility, strict checkpoint missing-prefix handling, masks and loss
    correctness.

acceptance_checks:
  - `python -m py_compile gemini_models.py train_light.py`
  - `bash -n tools/run_rt57_gtnp_v2_station_distinctive_slurm.sh`
  - focused RT57-v2 tests pass
  - existing causal-random, formal-eval and RT55 compatibility tests pass
  - original RT55/RT56 config hashes remain unchanged
  - disabled switches reproduce legacy output
  - initial RT57 output reproduces RT56
  - synthetic one-station query change changes output
  - synthetic one-station waveform change changes output
  - waveform-only station permutation changes output when coordinates are held
    fixed
  - paired waveform/coordinate station permutation leaves output invariant
  - only the new residual branch is trainable
  - `git diff --check` passes

hpc_followup:
  - Run one seed, fixed six epochs.
  - Do not add another architecture variant in the same run.
  - Primary go/no-go metrics:
      random non-input MAE/RMSE/R2/slope/bias
      NLL/Brier/sigma/coverage
      actual selected-station-count=1 spatial range ratio
      one-station pairwise-delta MAE
      requested-1-s MAE
      normal all/non-input retention
      waveform-permutation degradation for the new residual contribution
      station-local residual auxiliary accuracy
  - Go only if:
      random non-input MAE <= RT56 baseline
      random slope >= 0.40
      actual one-station spatial range ratio improves materially
      one-station pairwise-delta MAE does not worsen
      waveform-only permutation measurably degrades RT57
      normal retention stays within the predeclared tolerance
  - If spatial metrics improve but waveform-only permutation has negligible
    effect, classify the gain as geometry-prior-dominated rather than evidence
    that waveform station distinctiveness was solved.

risks:
  - Pre-pooling DiTing temporal tokens may themselves be strongly event-common.
    This design gives a trainable adapter a chance to amplify task-relevant
    station differences, but cannot manufacture information absent from the
    frozen encoder.
  - If the v2 branch remains waveform-insensitive, the next—not concurrent—step
    is to unfreeze only the final DiTing interaction/block at a much smaller
    learning rate with RT56-output distillation.
  - For unseen continuous locations, coordinates alone cannot identify
    micro-scale site response; Vs30/geology/topography can be added later as
    query covariates.
  - A local-PGA auxiliary can encourage amplitude shortcuts if overweighted;
    keep it secondary to final query NLL and spatial-difference loss.

open_questions:
  - none; implement v2 instead of the earlier v1
[/AI-HANDOFF]

[CODEX-RESULT]
task_id: 20260908-rt57-gtnp-v2-station-distinctive
base_commit: e74f977271b0bcde42c7ac7a18cb77ac60107e34
result_commit: <full SHA or none>
branch: rt57-geo-temporal-neural-process
changed_files:
  - <path + summary>
verification:
  - <exact command>: <pass/fail/not run + output>
compatibility:
  - <RT55/RT56 evidence and original config hashes>
hpc_status:
  - not submitted
remaining_risks:
  - <known limitations>
review_request:
  - Review station-token source, amplitude/duration path, common/residual
    decomposition, waveform dependence controls, one-station query/waveform
    sensitivity, trainability isolation, and RT55 compatibility.
[/CODEX-RESULT]
