#!/usr/bin/env bash
# LOGIN NODE: inspect and print. This file must NEVER execute sbatch/srun.
set -euo pipefail
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${A01_ENV_FILE:-$code/a01.private.env}"
: "${A01_CODE_ROOT:?}" "${A01_OUTPUT_ROOT:?}" "${FE01_TRAIN_OUTPUT_ROOT:?}"
[[ "$A01_CODE_ROOT" == "$code" ]] || { echo 'Code root differs' >&2; exit 2; }
[[ "$A01_OUTPUT_ROOT" != "$FE01_TRAIN_OUTPUT_ROOT" && "$A01_OUTPUT_ROOT" != "$FE01_TRAIN_OUTPUT_ROOT/"* ]] || { echo 'Output overlaps original results' >&2; exit 2; }
for name in formal__team_original_scratch__seed42 formal__phasenet_pretrained_frozen__seed42 formal__eqt_pretrained_frozen__seed42; do
  for file in resolved_config.json protocol.lock.json best.pth last.pth; do
    [[ -f "$FE01_TRAIN_OUTPUT_ROOT/$name/$file" ]] || { echo "Missing: $name/$file" >&2; exit 2; }
  done
  [[ -f "$FE01_TRAIN_OUTPUT_ROOT/$name/init.pth" ]] || printf 'Missing %s/init.pth: audit will require exact reconstruction fingerprint.\n' "$name"
done
[[ -f "$FE01_WEIGHTS_ROOT/pretrained_manifest.json" && -f "$FE01_SPLIT_MANIFEST" ]] || { echo 'Data/weight declarations missing' >&2; exit 2; }
printf 'Code: %s\nOriginal results (read only): %s\nNEW outputs: %s\n' "$code" "$FE01_TRAIN_OUTPUT_ROOT" "$A01_OUTPUT_ROOT"
printf '%s\n' 'No jobs submitted. Commands to execute MANUALLY, one stage at a time:'
printf '%s\n' 'bash scripts/fe01_a01/submit.sh environment' 'bash scripts/fe01_a01/submit.sh audit 0-4' 'bash scripts/fe01_a01/submit.sh diagnostics 0-4' 'bash scripts/fe01_a01/submit.sh pilot 0-4' 'bash scripts/fe01_a01/submit.sh train 0-4' 'bash scripts/fe01_a01/submit.sh eval 0-3' 'bash scripts/fe01_a01/submit.sh eval 4' 'bash scripts/fe01_a01/submit.sh pack'
printf '%s\n' 'Index 0=TEAM OFF, 1=PhaseNet OFF, 2=EQT OFF, 3=DiTing ON, 4=DiTing OFF.'
