# FE01 local engineering evidence

Task `20261001-fe01-native-window-random-time-v2`; base
`9c95dbfaf92f36b2673d816026cd3f25b91eec66`, dedicated FE01 branch. These files are
local synthetic/interface evidence, **not production Japan results**.

- verification.json: 87 focused tests, no skips, CPU; compile/import/bash checks.
- workspace_protection.json: unchanged original worktree snapshots.
- pretrained_manifest.json: actual official STEAD v2 weight file/version/SHA and
  metadata; binaries external, DiTing still unregistered locally.
- downstream_initial_state_fingerprints.json: actual width1000/seed42 common
  initial states and parameter counts for five available families; DiTing not run.
- causal_boundary_audit.json: full-width synthetic future perturbation invariance
  and actual native encoder/feature shapes.
- sampler_distribution_audit.json: 100000 sampler-only draws per capability.
- capability_manifest.json: declared prefix capacity, endpoint/pre-P/features;
  its template verification flags remain conservative rather than certifying HPC.
- SMOKE_SCOPE.json: explicit tiny synthetic training/replay scope.
- implementation_files.sha256: reviewable implementation/test file snapshots.

Raw logs, generated HDF5, tiny checkpoints, mask/MDN exports, maps/GIFs and the
synthetic review archive live in `artifacts/fe01/` outside Git. The source/weight
bundle and SHA sidecar are generated from the clean implementation commit.
Use docs/ai/FE01_HPC_RUNBOOK.md for genuine manually submitted audits/runs and
docs/ai/FE01_ARTIFACT_SCHEMA.md for the future evidence checklist. Missing genuine
metrics, device results and test exposure are not synthesized here.
