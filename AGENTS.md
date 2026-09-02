# Repository agent guidance

These instructions apply to the whole repository. Human instructions in the
current task take precedence.

## Start here

Before analysing or changing the project, read these files in order:

1. `docs/ai/PROJECT_CONTEXT.md` for the compact, current RT55/RT56 snapshot;
2. `docs/ai/README.md` for the ChatGPT Project and Codex handoff protocol;
3. `SESSION_SUMMARY.md` for the longer engineering history;
4. the config, launcher, implementation, tests, and result provenance that are
   directly relevant to the task.

Always record the repository, branch, and commit used for an analysis. Do not
assume that the default branch contains the active experiments.

## Non-negotiable project constraints

- Preserve the original RT55 checkpoint-loading and inference behaviour. New
  RT56 or later functionality must remain opt-in through a new config or an
  explicit flag unless the user authorizes a migration.
- Treat train, validation, and held-out test as different evidence. Never use a
  test result to tune sampling, hyperparameters, checkpoint selection, or a
  reporting threshold.
- Do not rename validation results as test results. Check the split, resolved
  config, checkpoint metadata, and output provenance rather than trusting a
  directory name.
- Do not invent metrics, logs, checkpoints, data fields, attention maps, or HPC
  outcomes that are absent from accessible artifacts.
- Large datasets, model weights, raw NPZ exports, and Slurm outputs normally
  live outside Git. Request the exact artifact or provide a reproducible command
  when they are needed.
- Preserve existing outputs and unrelated working-tree changes. Never overwrite
  an experiment directory or include unrelated untracked artifacts in a commit.
- Keep paths, credentials, and cluster-specific values out of source. Use
  documented environment variables for local and supercomputer paths.

## Evidence and experiment reporting

- Separate **verified fact**, **inference**, and **proposal**.
- For every quantitative claim, report the protocol, split, checkpoint epoch,
  target population, sample count, metric definition/unit, and source artifact.
- PGA values in the current RT55/RT56 reports use `log10(m/s^2)`. State this
  coordinate explicitly when comparing values.
- Report all-target and non-input-target results separately whenever both are
  available. Easy input targets must not hide performance at unseen targets.
- For probabilistic output, interpret point accuracy and calibration together:
  MAE/RMSE/R2, bias/slope, NLL/Brier, predictive sigma, and interval coverage.
- Preserve seeds, deterministic-mask settings, split manifests, resolved
  configs, checkpoint metadata, and launch commands in experiment handoffs.

## Change and verification discipline

- Trace the complete data path before proposing an architectural fix:
  dataset/sampler -> masks and coordinates -> model inputs -> readout -> loss ->
  evaluation export.
- Prefer the smallest change that can answer one research question. State the
  compatibility risk, expected observable effect, ablation/control, and rollback
  path.
- Add or update focused tests for sampling, masking, config inheritance,
  checkpoint compatibility, and metric provenance when those behaviours change.
- Use `bash -n` for modified shell launchers, compile/import checks for modified
  Python, and focused unit tests before broad test runs.
- Do not claim a cluster job succeeded from a local dry-run. Report local checks
  and supercomputer execution as separate stages.

## Code Review Rules

### RT55 compatibility

- Flag any change that alters RT55 parameter names, tensor shapes, config
  defaults, sampler behaviour, checkpoint loading, or inference without an
  explicit compatibility path and regression test.

### Data leakage and protocol drift

- Flag checkpoint selection or hyperparameter decisions based on held-out test
  results, split changes without a new experiment identity, and comparisons that
  use different masks, target populations, or realtime cutoffs as if paired.

### Result provenance

- Flag metrics without source files and sample counts, conclusions drawn from
  filenames instead of checkpoint metadata, and generated figures whose plotted
  data cannot be reproduced.

## Communication conventions

- Respond in Chinese unless the user requests another language.
- When presenting mathematics, first show a terminal-readable Unicode/ASCII
  two-dimensional formula, then provide the raw LaTeX.
- End implementation or review work with the handoff fields defined in
  `docs/ai/README.md`.
