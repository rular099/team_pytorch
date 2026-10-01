# FE01 V2 verification — 2026-10-01

Engineering delivery only; **HPC NOT_SUBMITTED**. No Japan production training,
formal validation/test, remote execution or full data conversion was run.

## Executed checks

| Command/check | Exit/result | Actual evidence |
|---|---|---|
| `.venv-fe01/bin/python -m pytest -q tests/test_fe01_*.py tests/test_causal_random_geometry.py tests/test_scheduler_checkpoint_resume.py tests/test_query_geometry_diagnostics.py tests/test_eval_checkpoint_formal.py` | exit 0; **87 passed, 0 skipped** | CPU; native offline STEAD v2 hooks + generated HDF5 + synthetic legacy encoder |
| Python parse/compile checks of FE01 modules/scripts/tests and modified gemini_models.py | exit 0; 33 files | no production data reads |
| `bash -n` FE01 .sh/.sbatch/env template | exit 0; 10 files | no scheduler invocation |
| print_submit_commands with fake sbatch/srun marker executables | pass inside pytest | printed commands/log directory; neither executable invoked |
| `scripts/fe01/check_environment.py` | exit 0; ENV_PASS | local Python3.11 / torch2.11 / SeisBench0.10.1 / CPU; not ROCm evidence |
| `scripts/fe01/local_smoke.py --output artifacts/fe01/local_smoke_final` | exit 0; LOCAL_SMOKE_PASS | 9 generated events; one width20 CPU TEAM optimizer update; actual original MDN loss; val export/reference/site/figures; 3 cases × seconds1..2 real grid/GIF |
| sampler `.audit()` | pass; 100000 draws per DiTing/PhaseNet/EQT capability | counts/expectations/tolerances/retained mass in local report; simulation only |
| production-width common model initialization/causality | pass | actual RT55 width1000, seed42: TEAM/PhaseNet/EQT/amplitude/coords; real picking weights; source report fingerprints and encoder shapes |
| small DiTing strict-loading mock | pass inside pytest | proves mapping/key guard/frozen/common initialization mechanics; **does not prove 1200M loading** |
| pack_results and collect_runs --allow-incomplete | exit 0 | local synthetic review archive/inventory; explicit incomplete single-run status |
| `git diff --check` | exit 0 | minimal default-preserving legacy factory injection |
| original workspace exact before/after snapshots | pass | both HEAD/status/unstaged/staged bit-for-bit identical; workspace_protection.json |

Final pytest emits one nonfatal timm deprecated-import warning. The environment
reports optional apex/xformers absence; CPU checks pass without them. No ROCm,
real DiTing, full physical unit provenance or real time arrival was certified.
The full original RT55 checkpoint was not loaded; regression uses unchanged
factory defaults, exact keys/state/output comparison with a small encoder and
the repository's existing focused tests. Historical metric equivalence is not claimed.

One full-width common initial state digest is
`98c3efb9e55a8c5b0afd3dc204b095d2cb3449519e44b78726adcc9b6da1d569`
for the five actually available seed42 families. True DiTing equality must be
verified by manual audit; the mock is separate. All factual local evidence is
tracked in reports/fe01_local_20261001, including implementation file hashes.
Private paths and raw waveforms/weights/checkpoints are outside Git.

The legacy counterexample changes the actual cropped record's first forbidden
sample. Input, features and prediction change in legacy FullModel with a small
synthetic encoder. FE01 integrated future-NaN tests, masked filler tests and
full-width causal audits remain invariant. This detects a defect but does not
measure its effect on previously reported RT55/RT59 results.

## Original workspace protection

- team_pytorch: HEAD `8bde65cb64fd860adf552dfe56505b4b31a78b13`; snapshot digest
  `0e866d291a217a87980a995eee4ee7895a1cc0a4cb623f7689c06c171190d3eb`.
- team_pytorch_query_geometry_diagnostics: HEAD
  `8f54477ed11bf2623f40121648f7fcc756ada1a2`; snapshot digest
  `ade7a3c9069bd446b214add386a2b81803444506d856a4f22670448e59d54cfe`.

Digests include exact status/untracked inventory and tracked staged/unstaged
diffs. Existing V01 changes and untracked presentations/archive were preserved.
No original output was overwritten.

## Required manual checks

Follow FE01_HPC_RUNBOOK.md: fill actual account/modules/paths/resources, register
the actual DiTing checkpoint/provenance, pass each model's audit, inspect four
seed42 pilots, then manually submit formal 12 runs. Production cohort counts,
train-only normalization, spatial cores, gradient/resource efficiency, genuine
validation/site effects and 1–90 s region latency remain pending. Test stays
locked until explicit manual flags, protocol hash and exposure ledger are supplied.

Source bundling uses the exact committed branch and a hashed dtbench snapshot,
plus relative PhaseNet/EQT assets. It excludes historical result binaries and
provides SHA sidecar/SOURCE_IDENTITY. Bundle extraction/import verification is
reported in the final handoff; its generation follows the implementation commit.
