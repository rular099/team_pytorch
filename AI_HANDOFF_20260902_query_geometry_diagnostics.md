# Codex execution handoff: RT55/RT56 query-geometry diagnostics

Copy the complete `[AI-HANDOFF]` block below into Codex from the root of a clean worktree.

[AI-HANDOFF]
task_id: 20260902-query-geometry-diagnostics
repo: rular099/team_pytorch
branch: zhangb/native-scale-adapter-scaling
base_commit: 8bde65cb64fd860adf552dfe56505b4b31a78b13
goal: >
  Implement a non-destructive, validation-only diagnostic workflow that quantifies
  RT55/RT56 query-coordinate sensitivity, single-input spatial-output compression,
  and relevant checkpoint/readout gate states. Produce provenance-rich JSON/NPZ
  outputs and focused tests. Do not change model architecture, training behavior,
  existing RT55/RT56 configs, or existing inference defaults in this task.

verified_facts:
  - The required base is commit 8bde65cb64fd860adf552dfe56505b4b31a78b13
    on zhangb/native-scale-adapter-scaling.
  - Repository collaboration rules are in AGENTS.md, docs/ai/PROJECT_CONTEXT.md,
    docs/ai/README.md, and SESSION_SUMMARY.md. Read them before editing.
  - RT55 config:
    pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
  - RT56 config:
    pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
  - tests/test_causal_random_geometry.py asserts that RT56 preserves the RT55
    model_params and changes the sampling/fine-tuning protocol.
  - gemini_models.py contains the FullModel PGA readout, coordinate embedding,
    station-context, event-context, and CrossAttentionReadout paths.
  - With exactly one valid attention key, softmax attention weight is identically
    one; an ordinary logit-only distance bias or RoPE cannot by itself make the
    attended value query-dependent.
  - eval_checkpoint.py already contains deterministic split construction, formal
    PGA metrics, station-count metadata, waveform/station permutation controls,
    feature diagnostics, and evaluation provenance that should be reused rather
    than independently reimplemented.
  - PGA is evaluated in log10(m/s^2), using predictive mixture mean as the default
    point estimate.
  - Train, validation, and held-out test must remain strictly separated. This task
    uses validation only.

inferences:
  - The near-constant spatial output in many single-input cases is likely caused
    by a weak query-dependent residual path combined with a query-independent
    single-key attended value.
  - RT55 may obtain good all-target MAE partly through local waveform/amplitude
    shortcuts on input-station targets, without learning a strong non-input
    spatial field.
  - These are hypotheses to test. Do not encode them as assumed conclusions in
    the diagnostic calculations.

files_to_inspect:
  - AGENTS.md
  - docs/ai/PROJECT_CONTEXT.md
  - docs/ai/README.md
  - SESSION_SUMMARY.md
  - docs/rt56_random_geometry.md
  - train_light.py
  - gemini_util_light.py
  - gemini_models.py
  - eval_checkpoint.py
  - pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
  - pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
  - tests/test_causal_random_geometry.py
  - tests/test_eval_checkpoint_formal.py
  - tools/eval_rt55_validation_normal_roll_slurm.sh
  - tools/run_rt56_random_geometry_slurm.sh

constraints:
  - Before editing, print repository path, branch, HEAD, and git status.
  - If HEAD differs from base_commit, or the worktree contains unrelated changes,
    stop and report the discrepancy. Do not reset, clean, stash, or overwrite
    user work.
  - Preserve RT55 config, checkpoint loading, state-dict compatibility, output
    layout, and inference behavior.
  - New diagnostics must be opt-in and must not alter existing CLI defaults.
  - Prefer a new tool under tools/ over invasive changes to production forward
    paths. Small reusable helpers in eval_checkpoint.py are acceptable only when
    covered by tests and behavior-preserving.
  - Do not add true hypocentral/epicentral distance or propagation-path labels as
    model inputs. Geometry calculations in this task are diagnostic interventions
    only.
  - Use validation only. Do not run, inspect, or tune against held-out test.
  - Do not submit HPC jobs.
  - Do not commit generated NPZ, checkpoints, HDF5 data, Slurm logs, or large
    outputs.
  - Use new output paths and refuse to overwrite an existing output unless an
    explicit force flag is supplied.
  - Preserve exact counts of events, realtime samples, valid targets, station-count
    groups, and target-type groups in exported summaries.
  - Record resolved config and checkpoint provenance. Do not infer checkpoint
    identity from filenames.
  - In the first pass, modify files and run tests locally, then show the diff and
    verification results. Do not push until the user explicitly approves the push.

proposed_change:
  - Add tools/diagnose_query_geometry_sensitivity.py, or an equivalently named
    focused tool, with a documented CLI.
  - Reuse the existing eval_checkpoint model/dataset/provenance functions.
  - Required CLI inputs:
      - --config
      - --checkpoint
      - --protocol normal|random
      - --split val
      - --output-prefix
      - --device
      - --seed
      - --max-events or equivalent deterministic sample limiter
      - --station-counts, defaulting to 1,3,5,8,12,16
      - --radial-scales, defaulting to 0,0.5,1,1.5
  - Reject any split other than val in this initial tool.
  - Run the unmodified baseline prediction first.
  - For each valid sample, calculate baseline event-internal field diagnostics:
      - valid target count
      - station count
      - true P95-P05 range
      - predicted-mean P95-P05 range
      - range ratio when the true range is nonzero
      - event-centered MAE/RMSE
      - pairwise-delta MAE/RMSE, using deterministic pair sampling if all pairs
        are too expensive
      - event-internal Pearson and Spearman when defined
      - predictive sigma mean/median and 1-sigma/2-sigma coverage
      - target-type breakdown when realtime_target_type exists
  - Add a query-geometry intervention centered on the valid input-station centroid:
      q'(s) = c_input + s * (q - c_input)
    for each requested radial scale s, while keeping waveforms, station coordinates,
    masks, and labels fixed.
  - Treat radial scaling only as a sensitivity intervention, not as an accuracy
    protocol or a source-distance proxy.
  - Export for every scale:
      - mean absolute prediction change from scale=1
      - per-sample and per-target prediction changes
      - predicted spatial range
      - predictive sigma changes
      - breakdown by station count, especially n=1
  - Include a paired query-order equivariance sanity check:
      - permute valid query coordinates and their validity slots together
      - run inference
      - inverse-permute predictions
      - verify agreement with baseline within an explicit tolerance
    This is a correctness check, separate from the radial intervention.
  - Inspect and export checkpoint values/norms for relevant parameters when present:
      - waveform_scale_gate
      - pga_event_context_gate
      - station_context_gate
      - pga/readout attn_gate parameters
      - pga/readout ffn_gate parameters
      - query-injection gates
      - pga_query_token norm
    Missing optional keys must be reported, not treated as zero.
  - Where safely accessible from existing diagnostics, export:
      - PGA cross-attention entropy/max weight
      - valid attention mass
      - station embedding pairwise cosine
      - coordinate/waveform embedding norm ratios
    Do not change the model forward solely to obtain a diagnostic unless the change
    is opt-in, behavior-preserving, and tested.
  - Write:
      - <output-prefix>.summary.json
      - <output-prefix>.samples.npz
      - <output-prefix>.resolved_config.json
  - JSON/NPZ provenance must include:
      - repository and git commit
      - dirty-worktree flag
      - config path and resolved configuration
      - checkpoint path, file size, checkpoint epoch/loss/format metadata
      - optional checkpoint SHA-256 controlled by a CLI flag
      - protocol, split, seed, station-count list, radial scales
      - event/realtime/target sample counts
      - metric definitions and PGA unit
  - Add concise usage documentation either in the script docstring or a new
    docs/ diagnostic note. Do not rewrite the project status documents.
  - Add focused unit tests, preferably tests/test_query_geometry_diagnostics.py.
  - Keep implementation modular enough that a later architecture experiment can
    reuse the field-range and sensitivity metrics without importing a CLI main.

acceptance_checks:
  - A synthetic coordinate-sensitive fake model is detected as having nonzero
    radial query sensitivity.
  - A synthetic query-invariant fake model is detected as having zero or
    tolerance-level radial sensitivity.
  - radial scale 1 reproduces the original valid query coordinates exactly.
  - Invalid/padded query slots remain invalid and cannot affect exported metrics.
  - The paired query-order equivariance check correctly inverse-permutes outputs.
  - The tool reports exact sample and target counts and refuses non-validation
    splits.
  - The tool refuses accidental output overwrite without an explicit force flag.
  - Missing optional checkpoint gates are represented as missing/null with a
    reason, not silently converted to zero.
  - Existing formal metric definitions remain unchanged.
  - Run at minimum:
      pytest -q tests/test_query_geometry_diagnostics.py
      pytest -q tests/test_eval_checkpoint_formal.py
      pytest -q tests/test_causal_random_geometry.py
  - Also run:
      python -m py_compile tools/diagnose_query_geometry_sensitivity.py
      git diff --check
  - Report any broader test failure honestly; do not claim unrun tests passed.
  - Show a final git diff summary and confirm that the two original RT55/RT56
    config files are byte-for-byte unchanged.

hpc_followup:
  - Do not submit jobs. Provide command templates only.
  - RT55 normal-validation template:
      python tools/diagnose_query_geometry_sensitivity.py \
        --config pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json \
        --checkpoint "$RT55_EP32_CHECKPOINT" \
        --protocol normal \
        --split val \
        --output-prefix "$OUT/rt55_ep32_normal_querydiag" \
        --station-counts 1,3,5,8,12,16 \
        --radial-scales 0,0.5,1,1.5 \
        --device cuda:0
  - RT55 fixed-random zero-shot template:
      python tools/diagnose_query_geometry_sensitivity.py \
        --config pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json \
        --checkpoint "$RT55_EP32_CHECKPOINT" \
        --protocol random \
        --split val \
        --output-prefix "$OUT/rt55_ep32_random_querydiag" \
        --station-counts 1,3,5,8,12,16 \
        --radial-scales 0,0.5,1,1.5 \
        --device cuda:0
  - RT56 fixed-random template:
      python tools/diagnose_query_geometry_sensitivity.py \
        --config pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json \
        --checkpoint "$RT56_EP6_CHECKPOINT" \
        --protocol random \
        --split val \
        --output-prefix "$OUT/rt56_ep6_random_querydiag" \
        --station-counts 1,3,5,8,12,16 \
        --radial-scales 0,0.5,1,1.5 \
        --device cuda:0
  - RT56 normal-retention template:
      python tools/diagnose_query_geometry_sensitivity.py \
        --config pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json \
        --checkpoint "$RT56_EP6_CHECKPOINT" \
        --protocol normal \
        --split val \
        --output-prefix "$OUT/rt56_ep6_normal_querydiag" \
        --station-counts 1,3,5,8,12,16 \
        --radial-scales 0,0.5,1,1.5 \
        --device cuda:0
  - Codex must verify the exact supported protocol/config override mechanism from
    current code before finalizing these templates. Do not invent an unsupported
    CLI or silently evaluate the wrong generator protocol.

risks:
  - A radial query intervention creates synthetic geometries and must not be
    interpreted as an accuracy benchmark.
  - Current model/data coordinates may be absolute or transformed model units;
    use the existing tensors consistently and document their coordinate convention.
  - Running the RT55 checkpoint with the RT56 config is valid only when model_params
    are verified identical and checkpoint loading provenance is explicit.
  - Event/realtime samples and target slots are correlated. Later confidence
    intervals must bootstrap by event rather than treating targets as independent.
  - Optional diagnostics may require forward hooks. Hooks must be removed reliably
    and must not retain tensors across samples or cause GPU-memory growth.
  - Hashing a large checkpoint can be expensive; make it explicit and optional.
  - Do not use the held-out test statistics or prior 0.03-dex flatness threshold
    to tune this diagnostic or future models.

open_questions:
  - none for this diagnostic implementation
[/AI-HANDOFF]

## Required Codex return

After implementation, local verification, user-approved commit, and push, return:

[CODEX-RESULT]
task_id: 20260902-query-geometry-diagnostics
base_commit: 8bde65cb64fd860adf552dfe56505b4b31a78b13
result_commit: <full SHA or none>
branch: <actual branch>
changed_files:
  - <path + summary>
verification:
  - <exact command>: <pass/fail/not run + reason>
compatibility:
  - <RT55 regression evidence>
hpc_status:
  - not submitted
remaining_risks:
  - <known limitations>
review_request:
  - Review the implementation and diff at result_commit, especially protocol
    construction, query interventions, sample counts, provenance, and RT55
    compatibility.
[/CODEX-RESULT]
