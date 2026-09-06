#!/usr/bin/env bash

# Submit a short, read-only compute-node job that verifies the RT55/RT56
# query-diagnostic checkpoints, resolved configs, and external DiTing encoder.
# No training or model inference is performed.
#
# Dry run (default):
#   DRY_RUN=1 bash tools/inspect_querydiag_inputs_slurm.sh
#
# Submit:
#   DRY_RUN=0 CONFIRM_QUERYDIAG_INSPECT=1 \
#     bash tools/inspect_querydiag_inputs_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
DIAGNOSTIC_SCRIPT=${DIAGNOSTIC_SCRIPT:-$WORKDIR/tools/diagnose_query_geometry_sensitivity.py}
QUERYDIAG_LAUNCHER=${QUERYDIAG_LAUNCHER:-$WORKDIR/tools/run_query_geometry_diagnostics_slurm.sh}

RT55_WEIGHT_DIR=${RT55_WEIGHT_DIR:-$WORKDIR/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
RT56_WEIGHT_DIR=${RT56_WEIGHT_DIR:-$WORKDIR/weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42}
RT55_CONFIG=${RT55_CONFIG:-$RT55_WEIGHT_DIR/config.json}
RT56_CONFIG=${RT56_CONFIG:-$RT56_WEIGHT_DIR/config.json}
RT55_CHECKPOINT=${RT55_CHECKPOINT:-$RT55_WEIGHT_DIR/full_model_best_ep32.pth}
RT56_CHECKPOINT=${RT56_CHECKPOINT:-$RT56_WEIGHT_DIR/full_model_best.pth}
RT55_EXPECTED_EPOCH=${RT55_EXPECTED_EPOCH:-32}
RT56_EXPECTED_EPOCH=${RT56_EXPECTED_EPOCH:-6}

DITING_CONFIG=${DITING_CONFIG:-$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml}
DITING_PRETRAINED=${DITING_PRETRAINED:-/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt}

# These default hashes identify the reviewed upload-mode runtime files. They
# work even when the cluster tree has no Git metadata or cannot contact GitHub.
EXPECTED_DIAGNOSTIC_SHA256=${EXPECTED_DIAGNOSTIC_SHA256:-4b49c74012d3c0aff7ffae07ff8127eba9ba2c52e659c21eae50fa742b5c9034}
EXPECTED_LAUNCHER_SHA256=${EXPECTED_LAUNCHER_SHA256:-45ff9d663a0d069f669a81334bb7b3717be0a25af9c2e3b1a6aa8fa85ad4106e}
ALLOW_SOURCE_HASH_MISMATCH=${ALLOW_SOURCE_HASH_MISMATCH:-0}

OUT=${OUT:-$WORKDIR/logs/query_geometry_diagnostics_input_check}
OUTPUT_JSON=${OUTPUT_JSON:-$OUT/querydiag_input_identity.json}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_QUERYDIAG_INSPECT=${CONFIRM_QUERYDIAG_INSPECT:-0}
ALLOW_ACTIVE_JOB=${ALLOW_ACTIVE_JOB:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-0}
ENCODER_SHA256=${ENCODER_SHA256:-0}

JOB_NAME=${JOB_NAME:-team-qdiag-inspect}
SLURM_PARTITION=${SLURM_PARTITION:-diting}
QUERYDIAG_GRES_RESOURCE=${QUERYDIAG_GRES_RESOURCE:-dcu}
QUERYDIAG_GRES_COUNT=${QUERYDIAG_GRES_COUNT:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-2}
SLURM_TIME=${SLURM_TIME:-00:20:00}
CONDA_ENV=${CONDA_ENV:-lsm_env}
MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}

require_file() {
    local path=$1
    local label=$2
    if [[ ! -s "$path" ]]; then
        echo "$label is missing or empty: $path" >&2
        exit 1
    fi
}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

verify_runtime_sources() {
    require_file "$DIAGNOSTIC_SCRIPT" "Query-diagnostic Python tool"
    require_file "$QUERYDIAG_LAUNCHER" "Query-diagnostic Slurm launcher"
    local diagnostic_sha launcher_sha
    diagnostic_sha=$(file_sha256 "$DIAGNOSTIC_SCRIPT")
    launcher_sha=$(file_sha256 "$QUERYDIAG_LAUNCHER")
    echo "[identity] diagnostic_script=$DIAGNOSTIC_SCRIPT"
    echo "[identity] diagnostic_sha256=$diagnostic_sha"
    echo "[identity] expected_diagnostic_sha256=$EXPECTED_DIAGNOSTIC_SHA256"
    echo "[identity] querydiag_launcher=$QUERYDIAG_LAUNCHER"
    echo "[identity] launcher_sha256=$launcher_sha"
    echo "[identity] expected_launcher_sha256=$EXPECTED_LAUNCHER_SHA256"
    if [[ "$diagnostic_sha" != "$EXPECTED_DIAGNOSTIC_SHA256" || \
          "$launcher_sha" != "$EXPECTED_LAUNCHER_SHA256" ]]; then
        if [[ "$ALLOW_SOURCE_HASH_MISMATCH" != "1" ]]; then
            echo "Uploaded query-diagnostic source hash mismatch; refusing inspection." >&2
            exit 1
        fi
        echo "[UNSAFE WARN] ALLOW_SOURCE_HASH_MISMATCH=1; source hash mismatch accepted." >&2
    fi
}

local_git_identity() {
    local head status
    head=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || true)
    status=$(git -C "$WORKDIR" status --porcelain 2>/dev/null || true)
    echo "[identity] git_head=${head:-unavailable}"
    if [[ -n "$head" ]]; then
        echo "[identity] git_worktree_dirty=$([[ -n "$status" ]] && echo true || echo false)"
    else
        echo "[identity] git_worktree_dirty=unavailable"
    fi
}

check_active_job() {
    if [[ "$ALLOW_ACTIVE_JOB" == "1" ]]; then
        echo "[UNSAFE WARN] ALLOW_ACTIVE_JOB=1; active-job guard bypassed." >&2
        return 0
    fi
    if ! command -v squeue >/dev/null 2>&1; then
        return 0
    fi
    local active_job
    active_job=$(squeue --noheader --user "$(id -un)" --name "$JOB_NAME" --format='%A %T' 2>/dev/null | awk 'NF {print; exit}' || true)
    if [[ -n "$active_job" ]]; then
        echo "A same-name inspection job is already active: $JOB_NAME $active_job" >&2
        exit 1
    fi
}

for integer_spec in \
    "RT55_EXPECTED_EPOCH:$RT55_EXPECTED_EPOCH" \
    "RT56_EXPECTED_EPOCH:$RT56_EXPECTED_EPOCH"; do
    integer_name=${integer_spec%%:*}
    integer_value=${integer_spec#*:}
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "$integer_name must be a non-negative integer; got: $integer_value" >&2
        exit 2
    fi
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "$DRY_RUN" != "1" && "$CONFIRM_QUERYDIAG_INSPECT" != "1" ]]; then
        echo "Submission requires CONFIRM_QUERYDIAG_INSPECT=1." >&2
        exit 2
    fi
    for file_spec in \
        "$RT55_CONFIG:RT55 resolved config" \
        "$RT56_CONFIG:RT56 resolved config" \
        "$RT55_CHECKPOINT:RT55 checkpoint" \
        "$RT56_CHECKPOINT:RT56 checkpoint" \
        "$DITING_CONFIG:DiTing config" \
        "$DITING_PRETRAINED:DiTing pretrained encoder"; do
        require_file "${file_spec%%:*}" "${file_spec#*:}"
    done
    verify_runtime_sources
    local_git_identity
    if [[ -e "$OUTPUT_JSON" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
        echo "Inspection result already exists; choose a new OUT or preserve it: $OUTPUT_JSON" >&2
        exit 1
    fi
    check_active_job

    SCRIPT_PATH=$(cd "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
    export WORKDIR DIAGNOSTIC_SCRIPT QUERYDIAG_LAUNCHER
    export RT55_CONFIG RT56_CONFIG RT55_CHECKPOINT RT56_CHECKPOINT
    export RT55_EXPECTED_EPOCH RT56_EXPECTED_EPOCH DITING_CONFIG DITING_PRETRAINED
    export EXPECTED_DIAGNOSTIC_SHA256 EXPECTED_LAUNCHER_SHA256
    export ALLOW_SOURCE_HASH_MISMATCH OUT OUTPUT_JSON ALLOW_EXISTING_OUTPUT
    export CHECKPOINT_SHA256 ENCODER_SHA256 CONDA_ENV MODULE_UNLOAD MODULE_LOADS

    echo "[identity] rt55_config=$RT55_CONFIG"
    echo "[identity] rt55_checkpoint=$RT55_CHECKPOINT expected_epoch=$RT55_EXPECTED_EPOCH"
    echo "[identity] rt56_config=$RT56_CONFIG"
    echo "[identity] rt56_checkpoint=$RT56_CHECKPOINT expected_epoch=$RT56_EXPECTED_EPOCH"
    echo "[identity] diting_config=$DITING_CONFIG"
    echo "[identity] diting_pretrained=$DITING_PRETRAINED"
    echo "[identity] output_json=$OUTPUT_JSON"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[DRY-RUN] sbatch --job-name=%q --partition=%q --nodes=1 --ntasks=1 --cpus-per-task=%q --gres=%q --time=%q --chdir=%q --output=%q --error=%q --export=ALL %q\n' \
            "$JOB_NAME" "$SLURM_PARTITION" "$SLURM_CPUS_PER_TASK" \
            "$QUERYDIAG_GRES_RESOURCE:$QUERYDIAG_GRES_COUNT" "$SLURM_TIME" "$WORKDIR" \
            "$OUT/inspect-%j.out" "$OUT/inspect-%j.err" "$SCRIPT_PATH"
        echo "[OK] inspection dry run passed; no job submitted."
        exit 0
    fi

    mkdir -p "$OUT"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$SLURM_PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gres="$QUERYDIAG_GRES_RESOURCE:$QUERYDIAG_GRES_COUNT" \
        --time="$SLURM_TIME" \
        --chdir="$WORKDIR" \
        --output="$OUT/inspect-%j.out" \
        --error="$OUT/inspect-%j.err" \
        --export=ALL \
        "$SCRIPT_PATH"
    exit 0
fi

cd "$WORKDIR"
verify_runtime_sources
local_git_identity

restore_nounset=0
if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
fi
source /etc/profile
if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if [[ "$restore_nounset" -eq 1 ]]; then
    set -u
fi

if command -v module >/dev/null 2>&1 || declare -F module >/dev/null 2>&1; then
    if [[ -n "$MODULE_UNLOAD" ]]; then
        module unload "$MODULE_UNLOAD" || true
    fi
    for module_name in $MODULE_LOADS; do
        module load "$module_name"
    done
else
    echo "module command is unavailable on the compute node." >&2
    exit 1
fi

restore_nounset=0
if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
fi
export PS1=${PS1:-}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [[ "$restore_nounset" -eq 1 ]]; then
    set -u
fi

echo "[runtime] host=$(hostname)"
echo "[runtime] python=$(command -v python)"
python -c 'import torch; print("[runtime] torch_version=" + torch.__version__)'

python - <<'PY'
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_sha256(path, enabled):
    return sha256_file(path) if enabled else None


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def git_value(*arguments):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=os.environ["WORKDIR"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


encoder_path = Path(os.environ["DITING_PRETRAINED"]).resolve()
diting_config = Path(os.environ["DITING_CONFIG"]).resolve()
output_path = Path(os.environ["OUTPUT_JSON"]).resolve()
checkpoint_sha = os.environ.get("CHECKPOINT_SHA256", "0") == "1"
encoder_sha = os.environ.get("ENCODER_SHA256", "0") == "1"

specifications = {
    "rt55": {
        "checkpoint": Path(os.environ["RT55_CHECKPOINT"]).resolve(),
        "config": Path(os.environ["RT55_CONFIG"]).resolve(),
        "expected_epoch": int(os.environ["RT55_EXPECTED_EPOCH"]),
    },
    "rt56": {
        "checkpoint": Path(os.environ["RT56_CHECKPOINT"]).resolve(),
        "config": Path(os.environ["RT56_CONFIG"]).resolve(),
        "expected_epoch": int(os.environ["RT56_EXPECTED_EPOCH"]),
    },
}

errors = []
report = {
    "schema_version": 1,
    "status": "checking",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "repository": {
        "root": str(Path(os.environ["WORKDIR"]).resolve()),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
    },
    "runtime_sources": {},
    "diting": {},
    "runs": {},
}

for key, env_name, expected_name in (
    ("diagnostic", "DIAGNOSTIC_SCRIPT", "EXPECTED_DIAGNOSTIC_SHA256"),
    ("launcher", "QUERYDIAG_LAUNCHER", "EXPECTED_LAUNCHER_SHA256"),
):
    path = Path(os.environ[env_name]).resolve()
    observed = sha256_file(path)
    expected = os.environ[expected_name]
    report["runtime_sources"][key] = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": observed,
        "expected_sha256": expected,
        "matched": observed == expected,
    }
    if observed != expected and os.environ.get("ALLOW_SOURCE_HASH_MISMATCH") != "1":
        errors.append(f"{key} source SHA-256 mismatch")

report["diting"] = {
    "config": {
        "path": str(diting_config),
        "file_size_bytes": diting_config.stat().st_size,
        "sha256": sha256_file(diting_config),
    },
    "pretrained_encoder": {
        "path": str(encoder_path),
        "exists": encoder_path.is_file(),
        "file_size_bytes": encoder_path.stat().st_size if encoder_path.is_file() else None,
        "sha256": optional_sha256(encoder_path, encoder_sha) if encoder_path.is_file() else None,
        "sha256_computed": encoder_sha,
    },
}

for run_name, specification in specifications.items():
    checkpoint_path = specification["checkpoint"]
    config_path = specification["config"]
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    excluded_prefixes = checkpoint.get("excluded_prefixes") or []
    excluded_count = checkpoint.get("excluded_tensor_count")
    external_encoder_required = bool(
        checkpoint.get("checkpoint_format") == "non_encoder_v1"
        or excluded_prefixes
        or int(excluded_count or 0) > 0
    )
    recorded_encoder = checkpoint.get("encoder_source")
    encoder_matched = (
        None
        if not recorded_encoder
        else os.path.realpath(recorded_encoder) == os.path.realpath(encoder_path)
    )
    run_report = {
        "resolved_config": {
            "path": str(config_path),
            "file_size_bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "file_size_bytes": checkpoint_path.stat().st_size,
            "sha256": optional_sha256(checkpoint_path, checkpoint_sha),
            "sha256_computed": checkpoint_sha,
            "epoch": checkpoint.get("epoch"),
            "expected_epoch": specification["expected_epoch"],
            "loss": checkpoint.get("loss"),
            "checkpoint_format": checkpoint.get("checkpoint_format"),
            "encoder_source": recorded_encoder,
            "excluded_prefixes": excluded_prefixes,
            "excluded_tensor_count": excluded_count,
            "saved_tensor_count": checkpoint.get(
                "saved_tensor_count",
                len(state_dict) if isinstance(state_dict, dict) else None,
            ),
            "total_tensor_count": checkpoint.get("total_tensor_count"),
            "external_encoder_required": external_encoder_required,
            "encoder_source_matches": encoder_matched,
        },
    }
    report["runs"][run_name] = run_report
    if checkpoint.get("epoch") != specification["expected_epoch"]:
        errors.append(
            f"{run_name} checkpoint epoch mismatch: expected "
            f"{specification['expected_epoch']}, observed {checkpoint.get('epoch')}"
        )
    if external_encoder_required and not encoder_path.is_file():
        errors.append(f"{run_name} requires a missing external encoder")
    if external_encoder_required and encoder_matched is False:
        errors.append(
            f"{run_name} encoder source mismatch: recorded={recorded_encoder}, "
            f"actual={encoder_path}"
        )
    del checkpoint

report["errors"] = errors
report["status"] = "passed" if not errors else "failed"
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
try:
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass

print(json.dumps(json_safe(report), indent=2, sort_keys=True))
print(f"[result] output_json={output_path}")
if errors:
    raise SystemExit("INPUT CHECK FAILED")
print("INPUT CHECK PASSED")
PY
