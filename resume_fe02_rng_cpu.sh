#!/usr/bin/env bash
# Additive hotfix: run from the ORIGINAL code directory, never a replacement tree.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shim="$root/compat/fe02_rng_cpu/sitecustomize.py"
[[ -f "$root/SOURCE_IDENTITY.json" ]] || { echo 'Run this from the original uploaded FE02 source directory' >&2; exit 2; }
grep -Eq '"commit"[[:space:]]*:[[:space:]]*"9f70371838ec31044e559096a939f023d2924f37"' "$root/SOURCE_IDENTITY.json" || {
    echo 'This hotfix is for the original 9f70371 FE02 source; do not replace existing code or locks' >&2; exit 2;
}
expected=162644b91b93c34ebc9f97708ac8482637702f5e0af2799e341e4f539b872af4
[[ "$(sha256sum -- "$shim" | cut -d ' ' -f 1)" == "$expected" ]] || { echo 'RNG compatibility file checksum mismatch' >&2; exit 2; }
export FE02_RNG_CPU_COMPAT=1 FE02_RNG_COMPAT_SHA256="$expected"
export FE02_RNG_LAUNCHER_SHA256=$(sha256sum -- "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)
export PYTHONPATH="$root/compat/fe02_rng_cpu${PYTHONPATH:+:$PYTHONPATH}"
export FE02_RESUME=1 FE02_INDICES="${FE02_INDICES:-4}"
exec bash "$root/scripts/fe02/submit.sh" "$@"
