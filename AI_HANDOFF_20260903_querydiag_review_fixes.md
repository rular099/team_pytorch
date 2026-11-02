# AI handoff: harden query-geometry diagnostics before HPC

[AI-HANDOFF]
task_id: 20260903-querydiag-review-fixes
repo: rular099/team_pytorch
branch: query-geometry-diagnostics
base_commit: a47d878d4d1df186f932d2455f30366f44ff9e1e
goal: >
  Fix the provenance, protocol-validation, target-group reporting, and Slurm
  safety issues found in review of a47d878 before any full supercomputer run.
  Preserve all RT55/RT56 production code, configs, checkpoint-loading behavior,
  and inference defaults.

verified_facts:
  - Branch query-geometry-diagnostics resolves to
    a47d878d4d1df186f932d2455f30366f44ff9e1e.
  - a47d878 has exactly one parent,
    8bde65cb64fd860adf552dfe56505b4b31a78b13.
  - The diff adds only:
      tools/diagnose_query_geometry_sensitivity.py
      tests/test_query_geometry_diagnostics.py
      tools/run_query_geometry_diagnostics_slurm.sh
  - No production model, loader, eval default, RT55 config, or RT56 config was
    modified.
  - The Python tool already enforces validation-only split aliases, preserves
    invalid query slots, reuses scale=1 baseline output, records exact sample
    arrays, and refuses existing outputs unless --force is passed.
  - Current validate_dataset_protocol() checks random-mask enabled state and
    apply_probability, but does not reject random protocols whose target
    sampling includes input stations or whose realtime validation definition
    differs from the pinned RT56 protocol.
  - Current Slurm action/output names assert rt55 ep32 and rt56 ep6, but the
    loaded checkpoint epoch is only recorded after loading and is not asserted.
  - Current provenance omits the actual --diting-config and
    --diting-pretrained inputs, although non-encoder checkpoints can depend on
    those external encoder artifacts.
  - The current Slurm worker prints Git branch/commit but does not require the
    reviewed commit, and the submission side has no same-job active guard.
  - Current summary reports spatial-field metrics over all valid targets, while
    target-type breakdown covers point errors only. Normal-protocol input
    targets can therefore hide non-input spatial-field compression.
  - GitHub has no CI/check status attached to a47d878; prior pass claims are
    local Codex evidence and must be returned with exact commands.

inferences:
  - The current RT55/RT56 source configs are correct, so an intended run may
    happen to use the right protocol; the validator is nevertheless too weak
    to make the output self-auditing.
  - A wrong frozen DiTing encoder can pass full-model state loading when the
    checkpoint excludes encoder tensors, making encoder provenance a result
    identity requirement rather than optional metadata.
  - With five forwards per realtime sample, a full unsharded validation run may
    exceed the default 23:50 wall time. This must be measured with a smoke run
    before full submission.

files_to_inspect:
  - AGENTS.md
  - docs/ai/PROJECT_CONTEXT.md
  - docs/ai/README.md
  - SESSION_SUMMARY.md
  - docs/rt56_random_geometry.md
  - tools/diagnose_query_geometry_sensitivity.py
  - tests/test_query_geometry_diagnostics.py
  - tools/run_query_geometry_diagnostics_slurm.sh
  - eval_checkpoint.py
  - train_light.py
  - gemini_util_light.py
  - tools/eval_rt55_validation_normal_roll_slurm.sh
  - tools/run_rt56_random_geometry_slurm.sh
  - pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
  - pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json

constraints:
  - Start by printing pwd, remote, branch, HEAD, and git status.
  - Stop without reset/stash/clean if HEAD differs from base_commit or unrelated
    worktree changes exist.
  - Do not edit gemini_models.py, gemini_util_light.py, train_light.py,
    eval_checkpoint.py, or the original RT55/RT56 configs unless a focused test
    proves a helper cannot remain in the diagnostic tool. Prefer changes only to
    the three files introduced by a47d878.
  - No held-out test access or tuning.
  - No model architecture, training, sampler, mask, loss, or inference-default
    changes.
  - Do not submit HPC jobs in this task.
  - Do not overwrite prior outputs or commit generated outputs/logs.
  - Do not infer checkpoint epoch or encoder identity from filenames.
  - Before push, show diff and exact local verification. Push only after user
    approval.

proposed_change:
  - In tools/diagnose_query_geometry_sensitivity.py:
    - Add --expected-checkpoint-epoch with optional/required behavior.
    - After loading, compare the checkpoint metadata epoch to the expected value;
      fail before inference on mismatch.
    - Do not label a run ep32/ep6 unless the expected and observed epoch match.
    - Expand checkpoint provenance to include excluded_prefixes,
      excluded_tensor_count, saved_tensor_count, total_tensor_count, and the
      observed epoch/loss/format when available.
    - Add provenance arguments for the actual DiTing config and pretrained
      encoder:
        absolute path
        file size
        SHA-256 (at least config SHA always; encoder SHA optional but controlled
        by a clear flag)
        parsed architecture-critical DiTing arguments
    - When checkpoint_format is non_encoder_v1 or encoder prefixes are excluded,
      require an explicit encoder source and compare it with checkpoint
      encoder_source when that metadata exists. On mismatch, fail unless an
      explicitly named unsafe override is supplied and recorded.
    - Record separate diagnostic_seed and generator sampling provenance. For
      every leaf generator include deterministic_sampling_seed, realtime mode,
      val_times, oversample, shuffle, random-mask fields, and target-sampling
      fields.
    - Strengthen validate_dataset_protocol():
        random requires enabled=true, apply_probability=1,
        target_sampling.enabled=true, exclude_inputs=true,
        the expected station-count set, realtime mode=val, and the pinned
        validation times;
        normal requires random mask disabled/effectively zero and the pinned
        RT55 validation realtime definition.
      Make expected protocol values explicit and export the checked snapshot.
    - Add global point correlation, R2, slope, and intercept to the baseline
      summary.
    - Add point and spatial-field summaries for:
        all valid targets
        non_input = target_type in {1,2}
        triggered_noninput
        untriggered
        input
      For spatial-field summaries report at least valid-target-count >=2 and >=5.
    - Add radial sensitivity summaries by the same target populations and station
      count. Keep raw per-target arrays unchanged.
    - Document that spatial summaries are unweighted across realtime samples,
      while global point metrics are target-weighted.
    - Make multi-file publication robust:
        write all three outputs to temporary names first;
        publish final files only after all serialization succeeds;
        write a final completion marker or equivalent manifest last.
      A partial prior run must be detected and reported, not silently mixed with
      a rerun.
  - In tools/run_query_geometry_diagnostics_slurm.sh:
    - Add EXPECTED_GIT_COMMIT, defaulting to the reviewed result commit supplied
      by the caller; worker must fail if actual HEAD differs unless an explicit
      override is set and logged.
    - Add same-job active checks patterned after
      tools/run_rt56_random_geometry_slurm.sh, with ALLOW_ACTIVE_JOB override.
    - Prevent concurrent workers from targeting the same output prefix, using an
      atomic lock directory/file in addition to the output-exists check.
    - Prefer exact run-directory resolved configs:
        RT55 resolved config default: $RT55_WEIGHT_DIR/config.json
        RT56 resolved config default: $RT56_WEIGHT_DIR/config.json
      If source configs are intentionally used, require an explicit mode and
      record that choice.
    - Use neutral RT56 output naming until checkpoint metadata has verified the
      epoch, or require RT56_EXPECTED_EPOCH and pass it to the Python tool.
    - Pass 32 for RT55 only when checkpoint metadata is expected and verified as
      epoch 32.
    - Print all identity inputs before sbatch: commit, config, config hash,
      checkpoint, expected epoch, DiTing config, encoder path, protocol, output.
    - Keep DRY_RUN=1 as default and do not weaken confirmation guards.
    - Add a documented smoke action/recipe with MAX_EVENTS=8. State explicitly
      that prefix-event smoke samples are for runtime/correctness only, not model
      conclusions.
  - In tests/test_query_geometry_diagnostics.py:
    - Add rejection tests for checkpoint epoch mismatch.
    - Add protocol-negative tests for random target exclusion false, random
      target sampling disabled, wrong station-count set, wrong val mode/times,
      and normal protocol with active random masking.
    - Add encoder-provenance tests for non-encoder checkpoint missing/mismatched
      encoder source.
    - Add all/non-input/triggered/untriggered spatial and radial grouping tests.
    - Add >=2 and >=5 valid-target count conditioning tests.
    - Add partial-output publication failure/recovery tests.
    - Add expected-commit/active-job/lock launcher tests where practical through
      dry-run or shell helper isolation.

acceptance_checks:
  - Exact commit comparison shows only the intended diagnostic/test/launcher
    files changed.
  - Original RT55/RT56 config blobs remain byte-for-byte unchanged.
  - Wrong expected checkpoint epoch fails before dataset inference.
  - A non-encoder checkpoint cannot run with an unrecorded or mismatched encoder
    source without an explicit, recorded override.
  - Random protocol with exclude_inputs=false is rejected.
  - Random protocol with wrong val_times or non-val realtime mode is rejected.
  - Normal protocol with causal random masking active is rejected.
  - Summary contains exact event/realtime/target counts plus all, non-input,
    triggered-noninput, untriggered and input point/field groups.
  - Spatial ranges for samples with one valid target do not enter >=2 or >=5
    summaries.
  - Partial serialization never leaves a set of final files that can be mistaken
    for a completed run.
  - Worker rejects a Git commit mismatch.
  - Submission rejects a same-name active job by default.
  - Two workers cannot concurrently claim one output prefix.
  - Run exact commands and report each separately:
      python -m py_compile tools/diagnose_query_geometry_sensitivity.py
      bash -n tools/run_query_geometry_diagnostics_slurm.sh
      pytest -q tests/test_query_geometry_diagnostics.py
      pytest -q tests/test_eval_checkpoint_formal.py
      pytest -q tests/test_causal_random_geometry.py
      DRY_RUN=1 ACTION=all bash tools/run_query_geometry_diagnostics_slurm.sh
      git diff --check
  - Return full command lines, case counts, pass/fail, and any warnings. Do not
    summarize them only as "N unittest cases."

hpc_followup:
  - Do not submit in this task.
  - After a reviewed fix commit is pushed and synchronized to the cluster:
    1. Inspect actual RT55 and RT56 checkpoint metadata and encoder source.
    2. Run one MAX_EVENTS=8 action at a time, starting with rt55_normal.
    3. Verify output commit/config/checkpoint/encoder provenance and completion
       marker.
    4. Verify baseline RT55 normal point metrics are consistent with the existing
       formal validation artifact, allowing only explained population/aggregation
       differences.
    5. Measure realtime samples per second and peak memory.
    6. Estimate full runtime as:
         full_seconds ≈ smoke_seconds
                        * full_realtime_samples
                        / smoke_realtime_samples
       If the estimate approaches the wall-time limit, implement deterministic
       event sharding plus a provenance-checking merge tool before full runs.
    7. Only then submit rt55_random and rt56_random full validation; run normal
       retention diagnostics afterward.
  - Required returned artifacts:
      *.summary.json
      *.samples.npz
      *.resolved_config.json
      completion manifest
      Slurm stdout/stderr
      exact sbatch command/environment snapshot

risks:
  - Prefix-event smoke selection is not geographically, temporally, or
    magnitude-stratified and cannot support model-quality conclusions.
  - Full runs perform about five forwards per realtime sample and may exceed the
    current 23:50 default.
  - Source configs can drift from run-directory resolved configs even when model
    state shapes remain load-compatible.
  - Output-directory names containing an epoch are not evidence of checkpoint
    identity.
  - Non-input target groups may have too few valid targets within some realtime
    samples; every field metric must report finite sample counts.
  - Raw NPZ permits later regrouping, but the summary must not present all-target
    spatial metrics as evidence for arbitrary non-input query performance.

open_questions:
  - none; inspect actual checkpoint metadata on the cluster and use neutral names
    until verified
[/AI-HANDOFF]

Required return:

[CODEX-RESULT]
task_id: 20260903-querydiag-review-fixes
base_commit: a47d878d4d1df186f932d2455f30366f44ff9e1e
result_commit: <full SHA or none>
branch: query-geometry-diagnostics
changed_files:
  - <path + summary>
verification:
  - <exact command>: <pass/fail/not run + output summary>
compatibility:
  - <RT55/RT56 evidence>
hpc_status:
  - not submitted
remaining_risks:
  - <limitations>
review_request:
  - Review the exact result commit before any HPC submission.
[/CODEX-RESULT]
