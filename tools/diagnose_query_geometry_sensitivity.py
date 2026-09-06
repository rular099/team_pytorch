#!/usr/bin/env python3
"""Validation-only query-geometry sensitivity diagnostics for RT55/RT56.

The tool leaves waveforms, input-station coordinates, masks, labels, model
parameters, and evaluation protocol unchanged.  It perturbs only valid PGA
query coordinates around the valid input-station centroid:

    q_scaled = input_centroid + scale * (q - input_centroid)

This is a sensitivity intervention, not an accuracy benchmark or a physical
source-distance transformation.

Example (run from the repository root):

    python tools/diagnose_query_geometry_sensitivity.py \
      --config /path/to/resolved/config.json \
      --checkpoint /path/to/full_model_best.pth \
      --protocol normal --split val \
      --output-prefix /path/to/querydiag/rt55_ep32_normal \
      --device cuda:0

Outputs are written only after inference succeeds:

    <output-prefix>.summary.json
    <output-prefix>.samples.npz
    <output-prefix>.resolved_config.json
    <output-prefix>.complete.json

The completion manifest is published last. Existing complete or partial output
sets are refused unless ``--force`` is supplied. Only validation is accepted by
design.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint  # noqa: E402
from train_light import (  # noqa: E402
    CHECKPOINT_ENCODER_PREFIXES,
    build_diting_args,
    load_config_file,
)


PGA_COORDINATE = "log10(m/s^2)"
PINNED_VALIDATION_TIMES = (1.0, 3.0, 5.0, 10.0, 20.0, 40.0, 90.0)
PINNED_RANDOM_STATION_COUNTS = (1, 3, 5, 8, 12, 16)
EVENT_SHARDING_ALGORITHM = "global_realtime_block_round_robin_v1"
TARGET_TYPE_NAMES = {
    0: "input",
    1: "triggered_noninput",
    2: "untriggered",
}
DITING_ARCHITECTURE_KEYS = (
    "base_width",
    "target_width",
    "model_depth",
    "in_samples",
    "patch_size",
    "num_interactions",
    "interaction_indexes",
    "out_channels",
    "diting_frontend",
    "attn_pool_hidden_dim",
    "attn_pool_temperature",
    "attn_pool_topk",
    "pale_size",
    "stem_convKs",
    "cpe_kernel_size",
    "ffn_convKS",
    "fpn_convKS",
    "aggregate_convKS",
    "head_convKS",
    "inter_mode",
    "add_vit_feature",
    "use_extra_extractor",
    "norm_layer",
    "xattn",
    "drop_path",
    "head_drop_rate",
    "init_std",
    "input_mult",
    "attn_mult",
    "output_mult",
    "eval_type",
    "pretrain_method",
    "pretrained_load_mode",
    "hps",
)


def parse_numeric_list(
    value: Any,
    cast,
    *,
    name: str,
    positive: bool = False,
) -> List[Any]:
    """Parse a comma-separated numeric list, preserving first occurrence."""
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_items = list(value)
    else:
        raw_items = str(value).split(",")
    parsed: List[Any] = []
    for raw in raw_items:
        text = str(raw).strip()
        if not text:
            continue
        item = cast(text)
        if positive and item <= 0:
            raise ValueError(f"{name} values must be positive, got {item!r}")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} values must be finite, got {item!r}")
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed


def require_validation_split(split: str) -> str:
    canonical = eval_checkpoint._canonical_eval_splits([split])
    if canonical != ["val"]:
        raise ValueError(
            "Query-geometry diagnostics are validation-only; --split must be val "
            "or a validation alias."
        )
    return "val"


def diagnostic_output_paths(
    output_prefix: Union[os.PathLike, str],
) -> Dict[str, Path]:
    prefix = Path(output_prefix).expanduser()
    return {
        "summary": Path(str(prefix) + ".summary.json"),
        "samples": Path(str(prefix) + ".samples.npz"),
        "resolved_config": Path(str(prefix) + ".resolved_config.json"),
        "completion": Path(str(prefix) + ".complete.json"),
    }


def refuse_existing_outputs(
    paths: Mapping[str, Path],
    *,
    force: bool = False,
) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not force:
        formatted = "\n  ".join(existing)
        core_names = {"summary", "samples", "resolved_config"}
        complete = (
            "completion" in paths
            and paths["completion"].is_file()
            and all(paths[name].is_file() for name in core_names)
        )
        state = "completed" if complete else "partial/incomplete"
        raise FileExistsError(
            f"Detected a {state} diagnostic output set; refusing to overwrite:\n  "
            f"{formatted}\nUse --force only after preserving prior results."
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _serialize_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _serialize_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _run_git(args: Sequence[str], repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_provenance(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    status = _run_git(["status", "--porcelain"], repo_root)
    return {
        "repository_root": str(repo_root.resolve()),
        "remote_origin": _run_git(["config", "--get", "remote.origin.url"], repo_root),
        "commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "branch": _run_git(["branch", "--show-current"], repo_root),
        "dirty_worktree": None if status is None else bool(status),
        "git_status_porcelain": None if status is None else status.splitlines(),
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_checkpoint(path: Path) -> Any:
    """Load checkpoint metadata compatibly with pre-2.0 PyTorch releases."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def inspect_checkpoint_file(checkpoint_path: Path) -> Dict[str, Any]:
    """Read result-identity metadata without trusting the checkpoint filename."""
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Query diagnostics require a full-model checkpoint mapping with "
            f"model_state_dict: {checkpoint_path}"
        )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint model_state_dict is not a mapping: {checkpoint_path}")

    checkpoint_format = checkpoint.get("checkpoint_format")
    excluded_prefixes = checkpoint.get("excluded_prefixes")
    excluded_prefixes_source = "checkpoint_metadata"
    if excluded_prefixes is None and checkpoint_format == "non_encoder_v1":
        excluded_prefixes = list(CHECKPOINT_ENCODER_PREFIXES)
        excluded_prefixes_source = "non_encoder_v1_compatibility_default"
    elif excluded_prefixes is None:
        excluded_prefixes = []
        excluded_prefixes_source = "not_recorded"
    elif isinstance(excluded_prefixes, str):
        excluded_prefixes = [excluded_prefixes]
    else:
        excluded_prefixes = [str(value) for value in excluded_prefixes]
    saved_tensor_count = checkpoint.get("saved_tensor_count")
    if saved_tensor_count is None:
        saved_tensor_count = len(state_dict)
    excluded_tensor_count = checkpoint.get("excluded_tensor_count")
    total_tensor_count = checkpoint.get("total_tensor_count")
    if total_tensor_count is None and excluded_tensor_count is not None:
        total_tensor_count = int(saved_tensor_count) + int(excluded_tensor_count)

    external_encoder_required = bool(
        checkpoint_format == "non_encoder_v1"
        or excluded_prefixes
        or (excluded_tensor_count is not None and int(excluded_tensor_count) > 0)
    )
    identity = {
        "path": str(checkpoint_path.resolve()),
        "file_size_bytes": int(checkpoint_path.stat().st_size),
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        "checkpoint_format": checkpoint_format,
        "encoder_source": checkpoint.get("encoder_source"),
        "excluded_prefixes": excluded_prefixes,
        "excluded_prefixes_source": excluded_prefixes_source,
        "excluded_tensor_count": (
            None if excluded_tensor_count is None else int(excluded_tensor_count)
        ),
        "saved_tensor_count": int(saved_tensor_count),
        "total_tensor_count": (
            None if total_tensor_count is None else int(total_tensor_count)
        ),
        "external_encoder_required": external_encoder_required,
    }
    del checkpoint
    return _json_safe(identity)


def validate_checkpoint_epoch(
    checkpoint_identity: Mapping[str, Any],
    expected_epoch: Optional[int],
) -> Dict[str, Any]:
    observed = checkpoint_identity.get("epoch")
    result = {
        "expected_epoch": expected_epoch,
        "observed_epoch": observed,
        "verified": False,
    }
    if expected_epoch is None:
        result["status"] = "observed_not_asserted"
        return result
    if observed is None:
        raise ValueError(
            f"Expected checkpoint epoch {expected_epoch}, but checkpoint metadata has no epoch."
        )
    try:
        observed_int = int(observed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint epoch is not an integer: {observed!r}") from exc
    if observed_int != int(expected_epoch):
        raise ValueError(
            f"Checkpoint epoch mismatch: expected {int(expected_epoch)}, observed {observed_int}."
        )
    result.update({
        "observed_epoch": observed_int,
        "verified": True,
        "status": "matched",
    })
    return result


def _normalized_identity_path(value: Union[os.PathLike, str]) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))


def validate_encoder_source(
    checkpoint_identity: Mapping[str, Any],
    encoder_path: Optional[Path],
    *,
    encoder_was_explicit: bool,
    allow_unsafe_mismatch: bool = False,
) -> Dict[str, Any]:
    """Require and compare the external encoder used by non-encoder checkpoints."""
    required = bool(checkpoint_identity.get("external_encoder_required", False))
    checkpoint_source = checkpoint_identity.get("encoder_source")
    if required and (not encoder_was_explicit or encoder_path is None):
        raise ValueError(
            "This checkpoint excludes encoder tensors and therefore requires an explicit "
            "--diting-pretrained encoder source."
        )
    if encoder_path is None:
        return {
            "required": required,
            "explicit": bool(encoder_was_explicit),
            "checkpoint_encoder_source": checkpoint_source,
            "actual_encoder_source": None,
            "matched": None,
            "unsafe_mismatch_override": bool(allow_unsafe_mismatch),
            "status": "not_used",
        }
    encoder_path = encoder_path.expanduser()
    if not encoder_path.is_file():
        raise FileNotFoundError(f"DiTing pretrained encoder not found: {encoder_path}")

    actual_normalized = _normalized_identity_path(encoder_path)
    checkpoint_normalized = (
        _normalized_identity_path(checkpoint_source) if checkpoint_source else None
    )
    matched = (
        None if checkpoint_normalized is None
        else actual_normalized == checkpoint_normalized
    )
    if required and matched is False and not allow_unsafe_mismatch:
        raise ValueError(
            "External encoder source mismatch for non-encoder checkpoint: "
            f"checkpoint={checkpoint_source!r}, actual={str(encoder_path.resolve())!r}. "
            "Use --allow-unsafe-encoder-source-mismatch only for an explicitly "
            "reviewed diagnostic and preserve that unsafe provenance."
        )
    if matched is True:
        status = "matched"
    elif matched is False:
        status = "mismatch_allowed_unsafe"
    elif required:
        status = "checkpoint_source_unavailable_explicit_encoder_recorded"
    else:
        status = "not_comparable"
    return {
        "required": required,
        "explicit": bool(encoder_was_explicit),
        "checkpoint_encoder_source": checkpoint_source,
        "checkpoint_encoder_source_normalized": checkpoint_normalized,
        "actual_encoder_source": str(encoder_path.resolve()),
        "actual_encoder_source_normalized": actual_normalized,
        "matched": matched,
        "unsafe_mismatch_override": bool(allow_unsafe_mismatch),
        "status": status,
    }


def file_provenance(path: Path, *, compute_sha256: bool) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Provenance input not found: {path}")
    return {
        "path": str(path.resolve()),
        "file_size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path) if compute_sha256 else None,
        "sha256_computed": bool(compute_sha256),
    }


def diting_architecture_provenance(diting_args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: _json_safe(getattr(diting_args, key))
        for key in DITING_ARCHITECTURE_KEYS
        if hasattr(diting_args, key)
    }


def clone_inputs(inputs: Sequence[Any]) -> List[Any]:
    return [value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for value in inputs]


def radial_scale_query_coordinates(
    query_coords: torch.Tensor,
    query_valid: torch.Tensor,
    station_coords: torch.Tensor,
    station_valid: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale valid query offsets around the valid input-station centroid."""
    if query_coords.ndim != 2 or station_coords.ndim != 2:
        raise ValueError("Expected unbatched [slots, coord_dims] coordinate tensors")
    if query_coords.shape[-1] != station_coords.shape[-1]:
        raise ValueError(
            "Query and station coordinate dimensions differ: "
            f"{query_coords.shape[-1]} != {station_coords.shape[-1]}"
        )
    query_valid = torch.as_tensor(query_valid, device=query_coords.device).bool().reshape(-1)
    station_valid = torch.as_tensor(station_valid, device=station_coords.device).bool().reshape(-1)
    finite_station = torch.isfinite(station_coords).all(dim=-1)
    centroid_mask = station_valid & finite_station
    if not torch.any(centroid_mask):
        raise ValueError("Cannot define input centroid without a finite valid input station")
    centroid = station_coords[centroid_mask].mean(dim=0)
    if float(scale) == 1.0:
        return query_coords.clone(), centroid
    scaled = query_coords.clone()
    finite_query = torch.isfinite(query_coords).all(dim=-1)
    active = query_valid & finite_query
    scaled[active] = centroid + float(scale) * (query_coords[active] - centroid)
    return scaled, centroid


def radial_intervention_inputs(inputs: Sequence[Any], scale: float) -> Tuple[List[Any], torch.Tensor]:
    if len(inputs) < 5:
        raise ValueError("Expected waveform, station coords/valid, and PGA query coords/valid")
    modified = clone_inputs(inputs)
    modified[3], centroid = radial_scale_query_coordinates(
        modified[3],
        modified[4],
        modified[1],
        modified[2],
        scale,
    )
    return modified, centroid


def deterministic_query_permutation(
    n_slots: int,
    *,
    seed: int,
    sample_index: int,
) -> np.ndarray:
    if n_slots <= 1:
        return np.arange(n_slots, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + int(sample_index) * 104729 + 7919)
    permutation = rng.permutation(n_slots).astype(np.int64, copy=False)
    if np.array_equal(permutation, np.arange(n_slots)):
        permutation = np.roll(permutation, 1)
    return permutation


def _has_vs30_target_inputs(inputs: Sequence[Any]) -> bool:
    if len(inputs) < 9:
        return False
    station_slots = int(inputs[2].numel()) if isinstance(inputs[2], torch.Tensor) else -1
    query_slots = int(inputs[4].numel()) if isinstance(inputs[4], torch.Tensor) else -1
    expected = (
        isinstance(inputs[5], torch.Tensor)
        and isinstance(inputs[6], torch.Tensor)
        and isinstance(inputs[7], torch.Tensor)
        and isinstance(inputs[8], torch.Tensor)
        and inputs[5].ndim >= 1
        and inputs[6].ndim >= 1
        and inputs[7].ndim >= 1
        and inputs[8].ndim >= 1
        and int(inputs[5].shape[0]) == station_slots
        and int(inputs[6].shape[0]) == station_slots
        and int(inputs[7].shape[0]) == query_slots
        and int(inputs[8].shape[0]) == query_slots
        and inputs[6].dtype == torch.bool
        and inputs[8].dtype == torch.bool
    )
    return bool(expected)


def permute_query_aligned_inputs(
    inputs: Sequence[Any],
    permutation: Sequence[int],
) -> List[Any]:
    """Permute query slots and known query-aligned optional VS30 tensors."""
    if len(inputs) < 5:
        raise ValueError("Expected PGA query coordinates and validity inputs")
    permutation_t = torch.as_tensor(permutation, dtype=torch.long)
    n_slots = int(inputs[4].numel())
    if permutation_t.numel() != n_slots:
        raise ValueError(f"Expected a {n_slots}-slot permutation, got {permutation_t.numel()}")
    if sorted(permutation_t.tolist()) != list(range(n_slots)):
        raise ValueError("Query permutation must contain every slot exactly once")

    modified = clone_inputs(inputs)
    for input_index in (3, 4):
        tensor = modified[input_index]
        modified[input_index] = tensor.index_select(0, permutation_t.to(tensor.device))
    if _has_vs30_target_inputs(modified):
        for input_index in (7, 8):
            tensor = modified[input_index]
            modified[input_index] = tensor.index_select(0, permutation_t.to(tensor.device))
    return modified


def inverse_permute(values: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    inverse = np.argsort(np.asarray(permutation, dtype=np.int64))
    return np.asarray(values)[inverse]


def _inputs_to_device(inputs: Sequence[Any], device: torch.device) -> List[Any]:
    return [
        value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    ]


def pga_prediction_from_outputs(
    model: torch.nn.Module,
    outputs: Sequence[torch.Tensor],
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    raw_model = model.module if hasattr(model, "module") else model
    layout = list(raw_model.output_layout)
    if "pga" not in layout:
        raise ValueError(f"Model output_layout has no PGA head: {layout}")
    output = outputs[layout.index("pga")].detach().cpu().numpy()
    if output.shape[0] != 1:
        raise ValueError(f"Expected diagnostic batch size 1, got output shape {output.shape}")
    output = output[0]
    if eval_checkpoint._is_point_output(output):
        mean = eval_checkpoint._point_mu_from_output("pga", output)
        mean = eval_checkpoint._maybe_unnormalize_pga("pga", mean, config)
        sigma = np.full(np.asarray(mean).shape, np.nan, dtype=np.float64)
    else:
        _weights, _mu, _component_sigma, mean, sigma = (
            eval_checkpoint._mixture_stats_from_output(output)
        )
        mean = eval_checkpoint._maybe_unnormalize_pga("pga", mean, config)
        sigma = eval_checkpoint._maybe_unnormalize_pga_sigma("pga", sigma, config)
        if np.asarray(mean).shape[-1:] == (1,):
            mean = np.asarray(mean)[..., 0]
            sigma = np.asarray(sigma)[..., 0]
    return (
        np.asarray(mean, dtype=np.float64).reshape(-1),
        np.asarray(sigma, dtype=np.float64).reshape(-1),
    )


@torch.no_grad()
def predict_pga(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    device: torch.device,
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    outputs = model(*_inputs_to_device(inputs, device))
    return pga_prediction_from_outputs(model, outputs, config)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for ties without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_spatial_field_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: Optional[np.ndarray] = None,
    pair_sample_limit: int = 4096,
    seed: int = 42,
) -> Dict[str, Union[float, int]]:
    """Compute within-sample spatial-field metrics over valid target slots."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    valid = valid & np.isfinite(truth) & np.isfinite(prediction)
    truth_v = truth[valid]
    prediction_v = prediction[valid]
    metrics: Dict[str, Union[float, int]] = {
        "valid_target_count": int(valid.sum()),
        "true_p95_p05_range": float("nan"),
        "predicted_p95_p05_range": float("nan"),
        "range_ratio": float("nan"),
        "event_centered_mae": float("nan"),
        "event_centered_rmse": float("nan"),
        "pair_count": 0,
        "pairwise_delta_mae": float("nan"),
        "pairwise_delta_rmse": float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
        "predictive_sigma_mean": float("nan"),
        "predictive_sigma_median": float("nan"),
        "coverage_1sigma": float("nan"),
        "coverage_2sigma": float("nan"),
    }
    if truth_v.size == 0:
        return metrics

    true_range = float(np.percentile(truth_v, 95) - np.percentile(truth_v, 5))
    predicted_range = float(
        np.percentile(prediction_v, 95) - np.percentile(prediction_v, 5)
    )
    centered_error = (
        prediction_v - np.mean(prediction_v)
        - (truth_v - np.mean(truth_v))
    )
    metrics.update({
        "true_p95_p05_range": true_range,
        "predicted_p95_p05_range": predicted_range,
        "range_ratio": predicted_range / true_range if true_range > 0 else float("nan"),
        "event_centered_mae": float(np.mean(np.abs(centered_error))),
        "event_centered_rmse": float(np.sqrt(np.mean(centered_error ** 2))),
        "pearson": _pearson(truth_v, prediction_v),
        "spearman": _pearson(_rankdata(truth_v), _rankdata(prediction_v)),
    })

    if truth_v.size >= 2 and pair_sample_limit != 0:
        pair_i, pair_j = np.triu_indices(truth_v.size, k=1)
        if pair_sample_limit > 0 and pair_i.size > pair_sample_limit:
            rng = np.random.default_rng(int(seed))
            selected = rng.choice(pair_i.size, size=pair_sample_limit, replace=False)
            pair_i = pair_i[selected]
            pair_j = pair_j[selected]
        delta_error = (
            (prediction_v[pair_i] - prediction_v[pair_j])
            - (truth_v[pair_i] - truth_v[pair_j])
        )
        metrics.update({
            "pair_count": int(delta_error.size),
            "pairwise_delta_mae": float(np.mean(np.abs(delta_error))),
            "pairwise_delta_rmse": float(np.sqrt(np.mean(delta_error ** 2))),
        })

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)[valid]
        sigma_valid = np.isfinite(sigma) & (sigma >= 0)
        if np.any(sigma_valid):
            sigma_v = sigma[sigma_valid]
            error_v = np.abs(prediction_v[sigma_valid] - truth_v[sigma_valid])
            metrics.update({
                "predictive_sigma_mean": float(np.mean(sigma_v)),
                "predictive_sigma_median": float(np.median(sigma_v)),
                "coverage_1sigma": float(np.mean(error_v <= sigma_v)),
                "coverage_2sigma": float(np.mean(error_v <= 2.0 * sigma_v)),
            })
    return metrics


def _extract_pga_labels(
    model: torch.nn.Module,
    labels: Sequence[Any],
) -> np.ndarray:
    raw_model = model.module if hasattr(model, "module") else model
    layout = list(raw_model.output_layout)
    pga_index = layout.index("pga")
    value = labels[pga_index]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _event_id(info: Any, fallback: int) -> str:
    if isinstance(info, Mapping) and "event_id" in info:
        value = info["event_id"]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.detach().cpu().numpy().tolist()
        return str(value)
    return f"event-index:{fallback}"


def _dataset_source_index(dataset: Any, sample_index: int) -> int:
    """Return the JointGenerator child index when it is deterministically available."""
    indexes = getattr(dataset, "indexes", None)
    if indexes is None:
        return 0
    try:
        entry = indexes[sample_index]
    except (IndexError, KeyError, TypeError):
        return 0
    if isinstance(entry, (tuple, list)) and entry:
        try:
            return int(entry[0])
        except (TypeError, ValueError):
            return 0
    return 0


def _info_array(
    info: Any,
    key: str,
    length: int,
    *,
    dtype,
    fill_value,
) -> np.ndarray:
    if not isinstance(info, Mapping) or key not in info:
        return np.full(length, fill_value, dtype=dtype)
    value = _to_numpy(info[key], dtype=dtype).reshape(-1)
    if value.size != length:
        raise ValueError(f"{key} has {value.size} slots; expected {length}")
    return value


def _info_scalar(info: Any, key: str) -> float:
    if not isinstance(info, Mapping) or key not in info:
        return float("nan")
    value = _to_numpy(info[key], dtype=np.float64).reshape(-1)
    return float(value[0]) if value.size else float("nan")


def collect_scalar_model_diagnostics(model: torch.nn.Module) -> Dict[str, float]:
    raw_model = model.module if hasattr(model, "module") else model
    source = getattr(raw_model, "_last_diag", None)
    diagnostics: Dict[str, float] = {}
    if not isinstance(source, Mapping):
        return diagnostics
    for key, value in source.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size == 1:
            diagnostics[str(key)] = float(array.reshape(-1)[0])
    coords_norm = diagnostics.get("coords_emb_norm")
    wave_norm = diagnostics.get("wave_emb_norm")
    if (
        coords_norm is not None
        and wave_norm is not None
        and math.isfinite(coords_norm)
        and math.isfinite(wave_norm)
        and abs(wave_norm) > 1e-12
    ):
        diagnostics["coordinate_to_wave_embedding_norm_ratio"] = coords_norm / wave_norm
    return diagnostics


def _tensor_statistics(tensor: torch.Tensor) -> Dict[str, Any]:
    array = tensor.detach().cpu().to(torch.float64).reshape(-1)
    result: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(array.numel()),
        "l2_norm": float(torch.linalg.vector_norm(array)) if array.numel() else 0.0,
        "mean": float(array.mean()) if array.numel() else None,
        "std": float(array.std(unbiased=False)) if array.numel() else None,
        "min": float(array.min()) if array.numel() else None,
        "max": float(array.max()) if array.numel() else None,
    }
    if array.numel() <= 16:
        result["values"] = array.tolist()
    return result


def inspect_checkpoint_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """Inspect loaded checkpoint parameters without assuming optional gates exist."""
    raw_model = model.module if hasattr(model, "module") else model
    categories = {
        "waveform_scale_gate": lambda name: name.endswith("waveform_scale_gate"),
        "pga_event_context_gate": lambda name: name.endswith("pga_event_context_gate"),
        "station_context_gate": lambda name: name.endswith("station_context_gate"),
        "pga_readout_attn_gates": lambda name: (
            ("pga_cross_attention" in name or "pga_station_target_readout" in name)
            and name.endswith("attn_gate")
        ),
        "pga_readout_ffn_gates": lambda name: (
            ("pga_cross_attention" in name or "pga_station_target_readout" in name)
            and name.endswith("ffn_gate")
        ),
        "pga_readout_query_injection_gates": lambda name: (
            ("pga_cross_attention" in name or "pga_station_target_readout" in name)
            and name.endswith("query_injection_gate")
        ),
        "pga_readout_first_residual_gates": lambda name: (
            ("pga_cross_attention" in name or "pga_station_target_readout" in name)
            and name.endswith("first_residual_gate")
        ),
        "pga_query_token": lambda name: name.endswith("pga_query_token"),
    }
    named_parameters = list(raw_model.named_parameters())
    report: Dict[str, Any] = {}
    for category, predicate in categories.items():
        matches = [
            {"name": name, **_tensor_statistics(parameter)}
            for name, parameter in named_parameters
            if predicate(name)
        ]
        if matches:
            report[category] = {
                "status": "present",
                "parameters": matches,
            }
        else:
            report[category] = {
                "status": "missing",
                "parameters": None,
                "reason": "No matching parameter is present in the loaded model/checkpoint schema.",
            }
    return report


def _leaf_generators(dataset: Any) -> List[Any]:
    generators = getattr(dataset, "generators", None)
    if generators is None:
        return [dataset]
    leaves: List[Any] = []
    for generator in generators:
        leaves.extend(_leaf_generators(generator))
    return leaves


def _generator_sampling_snapshot(generator: Any, index: int) -> Dict[str, Any]:
    realtime = copy.deepcopy(getattr(generator, "realtime_training", None) or {})
    realtime_target = copy.deepcopy(
        getattr(generator, "realtime_target_sampling", None) or {}
    )
    mask = copy.deepcopy(
        getattr(generator, "causal_random_input_mask", None) or {"enabled": False}
    )
    target_sampling = copy.deepcopy(mask.get("target_sampling") or {})
    return _json_safe({
        "generator_index": int(index),
        "generator_class": type(generator).__name__,
        "deterministic_sampling_seed": getattr(
            generator, "deterministic_sampling_seed", None
        ),
        "oversample": getattr(generator, "oversample", None),
        "shuffle": getattr(generator, "shuffle", None),
        "realtime_training": {
            "enabled": bool(realtime.get("enabled", False)),
            "mode": realtime.get("mode"),
            "reference": realtime.get("reference"),
            "val_times": list(realtime.get("val_times") or []),
            "train_times": list(realtime.get("train_times") or []),
            "train_time_bins": copy.deepcopy(realtime.get("train_time_bins") or []),
            "bins_per_event_per_epoch": realtime.get("bins_per_event_per_epoch"),
            "bin_sampling": realtime.get("bin_sampling"),
        },
        "realtime_target_sampling": {
            "enabled": bool(realtime_target.get("enabled", False)),
            "input_ratio": realtime_target.get("input_ratio"),
            "triggered_noninput_ratio": realtime_target.get(
                "triggered_noninput_ratio"
            ),
            "untriggered_ratio": realtime_target.get("untriggered_ratio"),
            "fill_missing": realtime_target.get("fill_missing"),
            "exclude_inputs": bool(realtime_target.get("exclude_inputs", False)),
        },
        "causal_random_input_mask": {
            "enabled": bool(mask.get("enabled", False)),
            "apply_probability": float(
                mask.get(
                    "apply_probability",
                    1.0 if mask.get("enabled", False) else 0.0,
                )
            ),
            "station_counts": list(mask.get("station_counts", [])),
            "order_selected_by_pick": mask.get("order_selected_by_pick"),
            "target_sampling": {
                "enabled": bool(target_sampling.get("enabled", False)),
                "input_ratio": target_sampling.get("input_ratio"),
                "triggered_noninput_ratio": target_sampling.get(
                    "triggered_noninput_ratio"
                ),
                "untriggered_ratio": target_sampling.get("untriggered_ratio"),
                "fill_missing": target_sampling.get("fill_missing"),
                "exclude_inputs": bool(target_sampling.get("exclude_inputs", False)),
            },
        },
    })


def _same_numeric_sequence(actual: Sequence[Any], expected: Sequence[Any]) -> bool:
    try:
        actual_values = [float(value) for value in actual]
        expected_values = [float(value) for value in expected]
    except (TypeError, ValueError):
        return False
    return len(actual_values) == len(expected_values) and all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(actual_values, expected_values)
    )


def validate_dataset_protocol(
    dataset: Any,
    requested_protocol: str,
    *,
    expected_station_counts: Sequence[int] = PINNED_RANDOM_STATION_COUNTS,
    expected_val_times: Sequence[float] = PINNED_VALIDATION_TIMES,
) -> Dict[str, Any]:
    requested_protocol = str(requested_protocol).strip().lower()
    if requested_protocol not in {"normal", "random"}:
        raise ValueError(f"Unknown protocol {requested_protocol!r}; expected normal or random")
    expected_station_set = sorted({int(value) for value in expected_station_counts})
    expected_times = [float(value) for value in expected_val_times]
    details: List[Dict[str, Any]] = []
    for index, generator in enumerate(_leaf_generators(dataset)):
        detail = _generator_sampling_snapshot(generator, index)
        details.append(detail)
        mask = detail["causal_random_input_mask"]
        realtime = detail["realtime_training"]
        enabled = bool(mask["enabled"])
        probability = float(mask["apply_probability"])
        target_sampling = mask["target_sampling"]
        if not realtime["enabled"] or realtime["mode"] != "val":
            raise ValueError(
                f"--protocol {requested_protocol} requires realtime_training.enabled=true "
                f"and mode='val'; generator {index} has enabled={realtime['enabled']}, "
                f"mode={realtime['mode']!r}."
            )
        if not _same_numeric_sequence(realtime["val_times"], expected_times):
            raise ValueError(
                f"--protocol {requested_protocol} requires pinned validation val_times "
                f"{expected_times}; generator {index} has {realtime['val_times']}."
            )
        if detail["oversample"] is None or float(detail["oversample"]) != 1.0:
            raise ValueError(
                f"Validation generator {index} must use oversample=1; "
                f"got {detail['oversample']!r}."
            )
        if detail["shuffle"] is not False:
            raise ValueError(
                f"Validation generator {index} must use shuffle=false; "
                f"got {detail['shuffle']!r}."
            )
        if requested_protocol == "normal" and enabled and probability > 0.0:
            raise ValueError(
                "--protocol normal does not match the resolved validation generator: "
                f"causal random masking is enabled with probability {probability}."
            )
        if requested_protocol == "random":
            if not enabled or not math.isclose(
                probability, 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "--protocol random requires the resolved validation generator to use "
                    "causal random masking with apply_probability=1.0; got "
                    f"enabled={enabled}, probability={probability}."
                )
            actual_station_set = sorted({int(value) for value in mask["station_counts"]})
            if actual_station_set != expected_station_set:
                raise ValueError(
                    "--protocol random requires station_counts set "
                    f"{expected_station_set}; generator {index} has {actual_station_set}."
                )
            if not target_sampling["enabled"]:
                raise ValueError(
                    "--protocol random requires target_sampling.enabled=true; "
                    f"generator {index} has false."
                )
            if not target_sampling["exclude_inputs"]:
                raise ValueError(
                    "--protocol random requires target_sampling.exclude_inputs=true; "
                    f"generator {index} has false."
                )
    return {
        "status": "passed",
        "requested_protocol": requested_protocol,
        "expected": {
            "validation_realtime_mode": "val",
            "validation_times": expected_times,
            "oversample": 1,
            "shuffle": False,
            "random_station_count_set": expected_station_set,
            "random_apply_probability": 1.0,
            "random_target_sampling_enabled": True,
            "random_targets_exclude_inputs": True,
            "normal_random_mask_effective_probability": 0.0,
        },
        "generators": details,
    }


def _point_error_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(truth) & np.isfinite(prediction)
    result: Dict[str, Any] = {
        "targets": int(valid.sum()),
        "mae": None,
        "rmse": None,
        "bias": None,
        "correlation": None,
        "r2": None,
        "slope": None,
        "intercept": None,
        "predictive_sigma_mean": None,
        "predictive_sigma_median": None,
        "coverage_1sigma": None,
        "coverage_2sigma": None,
    }
    if not np.any(valid):
        return result
    residual = prediction[valid] - truth[valid]
    result.update({
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
    })
    truth_v = truth[valid]
    prediction_v = prediction[valid]
    if truth_v.size > 1:
        truth_std = float(np.std(truth_v))
        prediction_std = float(np.std(prediction_v))
        if truth_std > 0.0 and prediction_std > 0.0:
            result["correlation"] = float(np.corrcoef(truth_v, prediction_v)[0, 1])
        ss_tot = float(np.sum((truth_v - np.mean(truth_v)) ** 2))
        if ss_tot > 0.0:
            result["r2"] = float(1.0 - np.sum(residual ** 2) / ss_tot)
            slope, intercept = np.polyfit(truth_v, prediction_v, 1)
            result["slope"] = float(slope)
            result["intercept"] = float(intercept)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_valid = valid & np.isfinite(sigma) & (sigma >= 0)
        if np.any(sigma_valid):
            sigma_v = sigma[sigma_valid]
            error_v = np.abs(prediction[sigma_valid] - truth[sigma_valid])
            result.update({
                "predictive_sigma_mean": float(np.mean(sigma_v)),
                "predictive_sigma_median": float(np.median(sigma_v)),
                "coverage_1sigma": float(np.mean(error_v <= sigma_v)),
                "coverage_2sigma": float(np.mean(error_v <= 2.0 * sigma_v)),
            })
    return result


def _finite_summary(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "finite_samples": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p05": float(np.percentile(finite, 5)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
    }


def _group_point_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    sigma: np.ndarray,
    sample_mask: np.ndarray,
) -> Dict[str, Any]:
    group_mask = np.asarray(valid, dtype=bool) & np.asarray(sample_mask, dtype=bool)[:, None]
    result = _point_error_summary(truth, prediction, group_mask, sigma)
    result["realtime_samples"] = int(np.asarray(sample_mask, dtype=bool).sum())
    return result


def _target_population_masks(
    valid: np.ndarray,
    target_type: np.ndarray,
) -> Dict[str, np.ndarray]:
    valid = np.asarray(valid, dtype=bool)
    target_type = np.asarray(target_type)
    return {
        "all": valid,
        "non_input": valid & np.isin(target_type, (1, 2)),
        "triggered_noninput": valid & (target_type == 1),
        "untriggered": valid & (target_type == 2),
        "input": valid & (target_type == 0),
    }


def _compute_spatial_metric_arrays(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    sigma: np.ndarray,
    *,
    pair_sample_limit: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    records: Dict[str, List[float]] = defaultdict(list)
    for sample_index in range(int(truth.shape[0])):
        metrics = compute_spatial_field_metrics(
            truth[sample_index],
            prediction[sample_index],
            valid[sample_index],
            sigma=sigma[sample_index],
            pair_sample_limit=pair_sample_limit,
            seed=int(seed) + sample_index * 1009,
        )
        for key, value in metrics.items():
            records[key].append(value)
    return {key: np.asarray(values) for key, values in records.items()}


def _summarize_spatial_metric_arrays(
    metrics: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    counts = np.asarray(metrics["valid_target_count"], dtype=np.int64)
    result: Dict[str, Any] = {
        "aggregation": "unweighted_across_realtime_samples",
        "realtime_samples_total": int(counts.size),
        "realtime_samples_with_valid_targets": int((counts >= 1).sum()),
    }
    for threshold in (1, 2, 5):
        sample_mask = counts >= threshold
        key = f"valid_target_count_at_least_{threshold}"
        result[key] = {
            "realtime_samples": int(sample_mask.sum()),
            "metrics": {
                metric_name: _finite_summary(np.asarray(values)[sample_mask])
                for metric_name, values in metrics.items()
            },
        }
    return result


def _per_sample_prediction_range(
    prediction: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(prediction)
    counts = valid.sum(axis=1).astype(np.int64, copy=False)
    ranges = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    for sample_index in range(prediction.shape[0]):
        values = prediction[sample_index, valid[sample_index]]
        if values.size:
            ranges[sample_index] = float(
                np.percentile(values, 95) - np.percentile(values, 5)
            )
    return ranges, counts


def _radial_population_summary(
    prediction: np.ndarray,
    baseline: np.ndarray,
    sigma_scaled: np.ndarray,
    sigma_baseline: np.ndarray,
    group_valid: np.ndarray,
    actual_station_count: np.ndarray,
    ordered_station_counts: Sequence[int],
) -> Dict[str, Any]:
    prediction_change = np.asarray(prediction) - np.asarray(baseline)
    sigma_change = np.asarray(sigma_scaled) - np.asarray(sigma_baseline)
    group_valid = np.asarray(group_valid, dtype=bool)
    valid_change = group_valid & np.isfinite(prediction_change)
    valid_sigma_change = group_valid & np.isfinite(sigma_change)
    spatial_ranges, spatial_counts = _per_sample_prediction_range(
        prediction, group_valid
    )

    def summarize(sample_mask: np.ndarray) -> Dict[str, Any]:
        sample_mask = np.asarray(sample_mask, dtype=bool)
        target_mask = valid_change & sample_mask[:, None]
        sigma_mask = valid_sigma_change & sample_mask[:, None]
        result = {
            "realtime_samples": int(sample_mask.sum()),
            "targets": int(target_mask.sum()),
            "mean_abs_prediction_change_from_scale_1": (
                float(np.mean(np.abs(prediction_change[target_mask])))
                if np.any(target_mask) else None
            ),
            "median_abs_prediction_change_from_scale_1": (
                float(np.median(np.abs(prediction_change[target_mask])))
                if np.any(target_mask) else None
            ),
            "mean_abs_predictive_sigma_change_from_scale_1": (
                float(np.mean(np.abs(sigma_change[sigma_mask])))
                if np.any(sigma_mask) else None
            ),
            "predicted_p95_p05_range": {},
        }
        for threshold in (1, 2, 5):
            field_mask = sample_mask & (spatial_counts >= threshold)
            result["predicted_p95_p05_range"][
                f"valid_target_count_at_least_{threshold}"
            ] = {
                "realtime_samples": int(field_mask.sum()),
                **_finite_summary(spatial_ranges[field_mask]),
            }
        return result

    result = summarize(np.ones(group_valid.shape[0], dtype=bool))
    result["aggregation"] = {
        "point_changes": "target_weighted",
        "predicted_p95_p05_range": "unweighted_across_realtime_samples",
    }
    result["by_station_count"] = {
        str(int(count)): summarize(actual_station_count == int(count))
        for count in ordered_station_counts
    }
    return result


def _summarize_run(
    arrays: Mapping[str, np.ndarray],
    radial_scales: Sequence[float],
    station_counts: Sequence[int],
    equivariance_tolerance: float,
    *,
    pair_sample_limit: int,
    seed: int,
) -> Dict[str, Any]:
    truth = arrays["pga_truth"]
    baseline = arrays["baseline_prediction"]
    sigma = arrays["baseline_sigma"]
    valid = arrays["target_valid"].astype(bool)
    actual_station_count = arrays["station_count"]
    target_type = arrays["target_type"]
    sample_count = int(truth.shape[0])

    baseline_summary: Dict[str, Any] = {
        "point_metrics": _point_error_summary(truth, baseline, valid, sigma),
        "spatial_field_metrics": {
            key[len("field_"):]: _finite_summary(value)
            for key, value in arrays.items()
            if key.startswith("field_")
        },
        "aggregation": {
            "point_metrics": "target_weighted",
            "spatial_field_metrics": "unweighted_across_realtime_samples",
        },
        "target_groups": {},
        "by_station_count": {},
        "by_target_type": {},
    }
    observed_counts = sorted({int(value) for value in actual_station_count.tolist()})
    ordered_counts = list(dict.fromkeys([*station_counts, *observed_counts]))
    field_arrays = {
        key[len("field_"):]: value
        for key, value in arrays.items()
        if key.startswith("field_")
    }
    target_population_masks = _target_population_masks(valid, target_type)
    for group_name, group_valid in target_population_masks.items():
        group_fields = _compute_spatial_metric_arrays(
            truth,
            baseline,
            group_valid,
            sigma,
            pair_sample_limit=pair_sample_limit,
            seed=seed,
        )
        group_summary: Dict[str, Any] = {
            "point_metrics": _point_error_summary(
                truth, baseline, group_valid, sigma
            ),
            "spatial_field_metrics": _summarize_spatial_metric_arrays(group_fields),
            "by_station_count": {},
        }
        for count in ordered_counts:
            sample_mask = actual_station_count == int(count)
            group_summary["by_station_count"][str(int(count))] = {
                "point_metrics": _point_error_summary(
                    truth,
                    baseline,
                    group_valid & sample_mask[:, None],
                    sigma,
                ),
                "spatial_field_metrics": _summarize_spatial_metric_arrays({
                    key: value[sample_mask]
                    for key, value in group_fields.items()
                }),
            }
        baseline_summary["target_groups"][group_name] = group_summary
    for count in ordered_counts:
        mask = actual_station_count == int(count)
        group_summary = _group_point_summary(truth, baseline, valid, sigma, mask)
        group_summary["spatial_field_metrics"] = {
            key: _finite_summary(value[mask])
            for key, value in field_arrays.items()
        }
        baseline_summary["by_station_count"][str(int(count))] = group_summary
    for type_id, type_name in TARGET_TYPE_NAMES.items():
        type_valid = valid & (target_type == type_id)
        baseline_summary["by_target_type"][type_name] = _point_error_summary(
            truth, baseline, type_valid, sigma
        )
    known_target_type = np.isin(target_type, list(TARGET_TYPE_NAMES))
    baseline_summary["by_target_type"]["unknown_or_unavailable"] = _point_error_summary(
        truth,
        baseline,
        valid & ~known_target_type,
        sigma,
    )

    interventions: Dict[str, Any] = {}
    radial_predictions = arrays["radial_prediction"]
    radial_sigmas = arrays["radial_sigma"]
    for scale_index, scale in enumerate(radial_scales):
        prediction = radial_predictions[:, scale_index, :]
        sigma_scaled = radial_sigmas[:, scale_index, :]
        prediction_change = prediction - baseline
        sigma_change = sigma_scaled - sigma
        valid_change = valid & np.isfinite(prediction_change)
        valid_sigma_change = valid & np.isfinite(sigma_change)
        scale_summary: Dict[str, Any] = {
            "targets": int(valid_change.sum()),
            "mean_abs_prediction_change_from_scale_1": (
                float(np.mean(np.abs(prediction_change[valid_change])))
                if np.any(valid_change) else None
            ),
            "median_abs_prediction_change_from_scale_1": (
                float(np.median(np.abs(prediction_change[valid_change])))
                if np.any(valid_change) else None
            ),
            "mean_abs_predictive_sigma_change_from_scale_1": (
                float(np.mean(np.abs(sigma_change[valid_sigma_change])))
                if np.any(valid_sigma_change) else None
            ),
            "predicted_p95_p05_range": _finite_summary(
                arrays["radial_predicted_p95_p05_range"][:, scale_index]
            ),
            "aggregation": {
                "point_changes": "target_weighted",
                "predicted_p95_p05_range": "unweighted_across_realtime_samples",
            },
            "target_groups": {},
            "by_station_count": {},
        }
        for count in ordered_counts:
            sample_mask = actual_station_count == int(count)
            group_valid = valid_change & sample_mask[:, None]
            group_sigma_valid = valid_sigma_change & sample_mask[:, None]
            scale_summary["by_station_count"][str(int(count))] = {
                "realtime_samples": int(sample_mask.sum()),
                "targets": int(group_valid.sum()),
                "mean_abs_prediction_change_from_scale_1": (
                    float(np.mean(np.abs(prediction_change[group_valid])))
                    if np.any(group_valid) else None
                ),
                "mean_abs_predictive_sigma_change_from_scale_1": (
                    float(np.mean(np.abs(sigma_change[group_sigma_valid])))
                    if np.any(group_sigma_valid) else None
                ),
                "predicted_p95_p05_range": _finite_summary(
                    arrays["radial_predicted_p95_p05_range"][sample_mask, scale_index]
                ),
            }
        for group_name, group_valid in target_population_masks.items():
            scale_summary["target_groups"][group_name] = _radial_population_summary(
                prediction,
                baseline,
                sigma_scaled,
                sigma,
                group_valid,
                actual_station_count,
                ordered_counts,
            )
        interventions[str(float(scale))] = scale_summary

    diagnostic_summary = {
        key[len("diag_"):]: _finite_summary(value)
        for key, value in arrays.items()
        if key.startswith("diag_")
    }
    equivariance_error = arrays["query_equivariance_max_abs_prediction_error"]
    finite_equivariance = np.isfinite(equivariance_error)
    failed = finite_equivariance & (equivariance_error > equivariance_tolerance)
    equivariance = {
        "tolerance": float(equivariance_tolerance),
        "checked_samples": int(finite_equivariance.sum()),
        "passed_samples": int((finite_equivariance & ~failed).sum()),
        "failed_samples": int(failed.sum()),
        "maximum_abs_prediction_error": (
            float(np.max(equivariance_error[finite_equivariance]))
            if np.any(finite_equivariance) else None
        ),
        "maximum_abs_sigma_error": _finite_summary(
            arrays["query_equivariance_max_abs_sigma_error"]
        ),
    }
    return {
        "counts": {
            "events": int(np.unique(arrays["event_key"]).size),
            "realtime_samples": sample_count,
            "target_slots": int(valid.size),
            "valid_targets": int(valid.sum()),
            "station_count_histogram": {
                str(int(key)): int(value)
                for key, value in sorted(Counter(actual_station_count.tolist()).items())
            },
            "target_type_counts": {
                **{
                    name: int((valid & (target_type == type_id)).sum())
                    for type_id, name in TARGET_TYPE_NAMES.items()
                },
                "unknown_or_unavailable": int((valid & ~known_target_type).sum()),
            },
        },
        "baseline": baseline_summary,
        "radial_interventions": interventions,
        "query_order_equivariance": equivariance,
        "model_internal_diagnostics": diagnostic_summary,
    }


@torch.no_grad()
def run_query_geometry_diagnostics(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    config: Mapping[str, Any],
    *,
    protocol: str,
    station_counts: Sequence[int],
    radial_scales: Sequence[float],
    seed: int = 42,
    max_events: int = 0,
    num_event_shards: int = 1,
    event_shard_id: int = 0,
    pair_sample_limit: int = 4096,
    equivariance_tolerance: float = 1e-5,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Run baseline, radial interventions, and query-order sanity checks."""
    protocol_details = validate_dataset_protocol(dataset, protocol)
    model.eval()
    if max_events < 0:
        raise ValueError("max_events must be zero (all) or a positive integer")
    if num_event_shards < 1:
        raise ValueError("num_event_shards must be a positive integer")
    if event_shard_id < 0 or event_shard_id >= num_event_shards:
        raise ValueError(
            "event_shard_id must satisfy 0 <= event_shard_id < num_event_shards"
        )
    if num_event_shards > 1 and max_events:
        raise ValueError("max_events must be 0 when event sharding is enabled")
    if pair_sample_limit < 0:
        raise ValueError("pair_sample_limit must be zero or positive")

    requested_scales = [float(value) for value in radial_scales]
    if 1.0 not in requested_scales:
        raise ValueError("radial_scales must include 1 so the unmodified baseline is explicit")

    records: Dict[str, List[Any]] = defaultdict(list)
    diag_records: Dict[str, List[float]] = defaultdict(list)
    known_diag_keys: Set[str] = set()
    selected_event_ids: List[str] = []
    selected_event_set: Set[str] = set()
    examined_samples = 0
    samples_per_event: Optional[int] = None
    dataset_event_count: Optional[int] = None
    if num_event_shards > 1:
        samples_per_event = len(PINNED_VALIDATION_TIMES)
        if len(dataset) % samples_per_event:
            raise ValueError(
                "Event sharding requires the pinned validation dataset length to be "
                f"divisible by {samples_per_event}; got {len(dataset)}"
            )
        dataset_event_count = len(dataset) // samples_per_event
        selected_event_ordinals = [
            event_ordinal
            for event_ordinal in range(dataset_event_count)
            if event_ordinal % num_event_shards == event_shard_id
        ]
        if not selected_event_ordinals:
            raise ValueError(
                f"Event shard {event_shard_id} of {num_event_shards} is empty for "
                f"{dataset_event_count} validation events"
            )
        sample_plan = [
            (event_ordinal * samples_per_event + offset, event_ordinal)
            for event_ordinal in selected_event_ordinals
            for offset in range(samples_per_event)
        ]
    else:
        selected_event_ordinals = []
        sample_plan = [(sample_index, None) for sample_index in range(len(dataset))]

    event_key_by_ordinal: Dict[int, str] = {}
    samples_by_ordinal: Counter = Counter()
    encountered_event_ordinal: Dict[str, int] = {}
    for sample_index, planned_event_ordinal in sample_plan:
        inputs, labels, info = dataset[sample_index]
        event_id = _event_id(info, sample_index)
        dataset_source_index = _dataset_source_index(dataset, sample_index)
        event_key = f"{dataset_source_index}|{event_id}"
        if event_key not in selected_event_set:
            if max_events and len(selected_event_ids) >= max_events:
                break
            selected_event_ids.append(event_key)
            selected_event_set.add(event_key)
            encountered_event_ordinal[event_key] = len(encountered_event_ordinal)
        if planned_event_ordinal is None:
            event_ordinal = encountered_event_ordinal[event_key]
        else:
            event_ordinal = int(planned_event_ordinal)
            previous_key = event_key_by_ordinal.setdefault(event_ordinal, event_key)
            if previous_key != event_key:
                raise ValueError(
                    "Pinned validation event block is not event-contiguous: "
                    f"ordinal={event_ordinal} first={previous_key!r} "
                    f"sample={sample_index} observed={event_key!r}"
                )
            samples_by_ordinal[event_ordinal] += 1
        examined_samples += 1

        if len(inputs) < 5:
            raise ValueError(f"Dataset sample {sample_index} has no PGA query inputs")
        station_valid = _to_numpy(inputs[2], dtype=bool).reshape(-1)
        query_valid = _to_numpy(inputs[4], dtype=bool).reshape(-1)
        query_coords = _to_numpy(inputs[3], dtype=np.float64)
        station_coords = _to_numpy(inputs[1], dtype=np.float64)
        n_query = int(query_valid.size)
        truth = _extract_pga_labels(model, labels)
        if truth.size != n_query:
            raise ValueError(
                f"PGA label/query slot mismatch at sample {sample_index}: "
                f"{truth.size} != {n_query}"
            )

        baseline_prediction, baseline_sigma = predict_pga(model, inputs, device, config)
        if baseline_prediction.size != n_query:
            raise ValueError(
                f"PGA prediction/query slot mismatch at sample {sample_index}: "
                f"{baseline_prediction.size} != {n_query}"
            )
        baseline_diag = collect_scalar_model_diagnostics(model)
        processed_samples = len(records["event_id"])
        new_diag_keys = set(baseline_diag) - known_diag_keys
        for key in new_diag_keys:
            diag_records[key].extend([float("nan")] * processed_samples)
        known_diag_keys.update(new_diag_keys)
        for key in known_diag_keys:
            diag_records[key].append(baseline_diag.get(key, float("nan")))

        field = compute_spatial_field_metrics(
            truth,
            baseline_prediction,
            query_valid,
            sigma=baseline_sigma,
            pair_sample_limit=pair_sample_limit,
            seed=int(seed) + int(sample_index) * 1009,
        )

        radial_predictions: List[np.ndarray] = []
        radial_sigmas: List[np.ndarray] = []
        radial_coords: List[np.ndarray] = []
        radial_ranges: List[float] = []
        centroid = None
        for scale in requested_scales:
            scaled_inputs, scale_centroid = radial_intervention_inputs(inputs, scale)
            centroid = scale_centroid if centroid is None else centroid
            if scale == 1.0:
                prediction = baseline_prediction.copy()
                sigma = baseline_sigma.copy()
            else:
                prediction, sigma = predict_pga(model, scaled_inputs, device, config)
            radial_predictions.append(prediction)
            radial_sigmas.append(sigma)
            radial_coords.append(_to_numpy(scaled_inputs[3], dtype=np.float64))
            valid_prediction = query_valid & np.isfinite(prediction)
            if np.any(valid_prediction):
                values = prediction[valid_prediction]
                radial_ranges.append(
                    float(np.percentile(values, 95) - np.percentile(values, 5))
                )
            else:
                radial_ranges.append(float("nan"))

        permutation = deterministic_query_permutation(
            n_query,
            seed=seed,
            sample_index=sample_index,
        )
        permuted_inputs = permute_query_aligned_inputs(inputs, permutation)
        permuted_prediction, permuted_sigma = predict_pga(
            model, permuted_inputs, device, config
        )
        restored_prediction = inverse_permute(permuted_prediction, permutation)
        restored_sigma = inverse_permute(permuted_sigma, permutation)
        valid_prediction = query_valid & np.isfinite(baseline_prediction) & np.isfinite(restored_prediction)
        if np.any(valid_prediction):
            equivariance_prediction_error = float(
                np.max(np.abs(restored_prediction[valid_prediction] - baseline_prediction[valid_prediction]))
            )
        else:
            equivariance_prediction_error = float("nan")
        valid_sigma = query_valid & np.isfinite(baseline_sigma) & np.isfinite(restored_sigma)
        if np.any(valid_sigma):
            equivariance_sigma_error = float(
                np.max(np.abs(restored_sigma[valid_sigma] - baseline_sigma[valid_sigma]))
            )
        else:
            equivariance_sigma_error = float("nan")

        target_type = _info_array(
            info,
            "realtime_target_type",
            n_query,
            dtype=np.int64,
            fill_value=-1,
        )
        records["event_id"].append(event_id)
        records["event_key"].append(event_key)
        records["dataset_source_index"].append(dataset_source_index)
        records["event_index"].append(int(sample_index))
        records["event_ordinal"].append(int(event_ordinal))
        records["realtime_elapsed_time"].append(
            _info_scalar(info, "realtime_elapsed_time")
        )
        records["station_count"].append(int(station_valid.sum()))
        records["station_valid"].append(station_valid)
        records["station_coords"].append(station_coords)
        records["input_station_centroid"].append(_to_numpy(centroid, dtype=np.float64))
        records["query_coords"].append(query_coords)
        records["target_valid"].append(query_valid)
        records["target_type"].append(target_type)
        records["pga_truth"].append(truth)
        records["baseline_prediction"].append(baseline_prediction)
        records["baseline_sigma"].append(baseline_sigma)
        records["radial_prediction"].append(np.stack(radial_predictions, axis=0))
        records["radial_sigma"].append(np.stack(radial_sigmas, axis=0))
        records["radial_prediction_change_from_scale_1"].append(
            np.stack(radial_predictions, axis=0) - baseline_prediction[None, :]
        )
        records["radial_sigma_change_from_scale_1"].append(
            np.stack(radial_sigmas, axis=0) - baseline_sigma[None, :]
        )
        records["radial_query_coords"].append(np.stack(radial_coords, axis=0))
        records["radial_predicted_p95_p05_range"].append(np.asarray(radial_ranges))
        records["query_permutation"].append(permutation)
        records["query_equivariance_max_abs_prediction_error"].append(
            equivariance_prediction_error
        )
        records["query_equivariance_max_abs_sigma_error"].append(
            equivariance_sigma_error
        )
        for key, value in field.items():
            records[f"field_{key}"].append(value)

    if not records["event_id"]:
        raise RuntimeError("No validation samples were processed")
    if num_event_shards > 1:
        incomplete = {
            int(event_ordinal): int(samples_by_ordinal[event_ordinal])
            for event_ordinal in selected_event_ordinals
            if samples_by_ordinal[event_ordinal] != samples_per_event
        }
        if incomplete:
            raise ValueError(
                "Event sharding did not retain every pinned realtime sample for "
                f"each selected event: {incomplete}"
            )

    sample_count = len(records["event_id"])
    for key in known_diag_keys:
        values = diag_records[key]
        if len(values) < sample_count:
            values.extend([float("nan")] * (sample_count - len(values)))
        records[f"diag_{key}"] = values

    arrays: Dict[str, np.ndarray] = {}
    for key, values in records.items():
        if key in {"event_id", "event_key"}:
            arrays[key] = np.asarray(values, dtype=str)
        else:
            arrays[key] = np.asarray(values)
    arrays["radial_scales"] = np.asarray(requested_scales, dtype=np.float64)
    arrays["requested_station_counts"] = np.asarray(station_counts, dtype=np.int64)

    summary = _summarize_run(
        arrays,
        requested_scales,
        station_counts,
        equivariance_tolerance,
        pair_sample_limit=pair_sample_limit,
        seed=seed,
    )
    summary["selection"] = {
        "dataset_realtime_samples": int(len(dataset)),
        "examined_realtime_samples": int(examined_samples),
        "max_events": int(max_events),
        "selected_events": int(len(selected_event_ids)),
        "event_sharding": {
            "mode": "shard" if num_event_shards > 1 else "single",
            "algorithm": EVENT_SHARDING_ALGORITHM,
            "num_event_shards": int(num_event_shards),
            "event_shard_id": int(event_shard_id),
            "samples_per_event": samples_per_event,
            "dataset_event_count": dataset_event_count,
            "selected_event_count": int(len(selected_event_ids)),
        },
        "requested_station_count_breakdown": [int(value) for value in station_counts],
        "note": (
            "All encountered station counts are evaluated. --station-counts controls "
            "the requested reporting order and records the planned groups; realized "
            "random-geometry counts outside that list are retained and reported."
        ),
    }
    summary["resolved_validation_generators"] = protocol_details
    summary["checkpoint_parameters"] = inspect_checkpoint_parameters(model)
    return summary, arrays


def build_provenance(
    *,
    config_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    checkpoint_epoch_validation: Mapping[str, Any],
    diting_config_identity: Mapping[str, Any],
    diting_encoder_identity: Optional[Mapping[str, Any]],
    encoder_source_validation: Mapping[str, Any],
    diting_args: argparse.Namespace,
    generator_protocol_validation: Mapping[str, Any],
    config_source_mode: str,
    deployment_source_identity: Mapping[str, Any],
    protocol: str,
    split: str,
    seed: int,
    station_counts: Sequence[int],
    radial_scales: Sequence[float],
    max_events: int,
    pair_sample_limit: int,
    equivariance_tolerance: float,
    checkpoint_sha256: bool,
    invocation_argv: Sequence[str],
    event_sharding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint = dict(checkpoint_identity)
    checkpoint_path = Path(str(checkpoint["path"]))
    checkpoint["sha256"] = (
        sha256_file(checkpoint_path) if checkpoint_sha256 else None
    )
    checkpoint["sha256_computed"] = bool(checkpoint_sha256)
    checkpoint["epoch_validation"] = dict(checkpoint_epoch_validation)
    return {
        "repository": git_provenance(REPO_ROOT),
        "invocation": {
            "argv": [str(value) for value in invocation_argv],
            "cwd": str(Path.cwd().resolve()),
        },
        "resolved_run_config": dict(config_identity),
        "config_source_mode": config_source_mode,
        "deployment_source_identity": dict(deployment_source_identity),
        "checkpoint": checkpoint,
        "diting": {
            "config": dict(diting_config_identity),
            "pretrained_encoder": (
                None if diting_encoder_identity is None
                else dict(diting_encoder_identity)
            ),
            "encoder_source_validation": dict(encoder_source_validation),
            "architecture_arguments": diting_architecture_provenance(diting_args),
        },
        "protocol": protocol,
        "split": split,
        "diagnostic_seed": int(seed),
        "generator_sampling": dict(generator_protocol_validation),
        "station_counts": [int(value) for value in station_counts],
        "radial_scales": [float(value) for value in radial_scales],
        "max_events": int(max_events),
        "event_sharding": dict(event_sharding or {
            "mode": "single",
            "algorithm": EVENT_SHARDING_ALGORITHM,
            "num_event_shards": 1,
            "event_shard_id": 0,
        }),
        "pair_sample_limit": int(pair_sample_limit),
        "equivariance_tolerance": float(equivariance_tolerance),
        "pga_coordinate": PGA_COORDINATE,
        "point_estimate": "predictive_mixture_mean",
        "geometry_coordinate_convention": (
            "Radial interventions operate directly on the existing query/station "
            "coordinate tensors emitted by the resolved validation generator. No "
            "hypocentral distance, propagation path, or physical-unit conversion is added."
        ),
    }


def write_outputs(
    paths: Mapping[str, Path],
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
    force: bool,
) -> None:
    refuse_existing_outputs(paths, force=force)
    payload = {
        "schema_version": 2,
        "provenance": provenance,
        "metric_definitions": {
            "radial_intervention": (
                "q_scaled = valid_input_station_centroid + scale * "
                "(q - valid_input_station_centroid); labels and all model inputs "
                "except valid query coordinates remain fixed"
            ),
            "true_or_predicted_p95_p05_range": (
                "Within-realtime-sample 95th minus 5th percentile over valid PGA targets"
            ),
            "event_centered_error": (
                "Error after independently subtracting the valid-target mean from "
                "truth and prediction within each realtime sample"
            ),
            "pairwise_delta_error": (
                "(prediction_i - prediction_j) - (truth_i - truth_j) within one realtime sample"
            ),
            "coverage": "abs(predictive_mean - truth) <= k * predictive_mixture_std",
            "query_order_equivariance": (
                "Permute query-aligned inputs, run inference, inverse-permute predictions, "
                "then compare with the unmodified baseline over valid targets"
            ),
            "aggregation": (
                "Point metrics and pointwise radial changes are target-weighted. "
                "Spatial-field metrics and within-sample predicted ranges are first "
                "computed per realtime sample, then summarized without sample weights."
            ),
            "target_groups": {
                "all": "all valid PGA targets",
                "non_input": "valid targets with realtime_target_type in {1, 2}",
                "triggered_noninput": "valid targets with realtime_target_type == 1",
                "untriggered": "valid targets with realtime_target_type == 2",
                "input": "valid targets with realtime_target_type == 0",
            },
        },
        **summary,
    }
    npz_arrays = dict(arrays)
    npz_arrays["provenance_json"] = np.asarray(
        json.dumps(_json_safe(provenance), sort_keys=True), dtype=str
    )
    npz_arrays["metric_definitions_json"] = np.asarray(
        json.dumps(payload["metric_definitions"], sort_keys=True), dtype=str
    )
    npz_arrays["resolved_config_json"] = np.asarray(
        json.dumps(_json_safe(config), sort_keys=True), dtype=str
    )
    core_names = ("resolved_config", "samples", "summary")
    required_names = {*core_names, "completion"}
    missing_names = required_names - set(paths)
    if missing_names:
        raise ValueError(f"Missing diagnostic output paths: {sorted(missing_names)}")

    run_id = uuid.uuid4().hex
    temp_paths: Dict[str, Path] = {}
    for name in required_names:
        final_path = paths[name]
        suffix = final_path.suffix
        temp_paths[name] = final_path.parent / (
            f".{final_path.stem}.{run_id}.tmp{suffix}"
        )

    try:
        # Serialization is isolated from all final names. If any serializer
        # fails, a previously complete result remains complete and a new run
        # leaves no misleading final output set.
        _serialize_json(temp_paths["resolved_config"], config)
        _serialize_npz(temp_paths["samples"], npz_arrays)
        _serialize_json(temp_paths["summary"], payload)
        completion = {
            "schema_version": 1,
            "status": "complete",
            "run_id": run_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                name: {
                    "path": str(paths[name].resolve()),
                    "file_size_bytes": int(temp_paths[name].stat().st_size),
                    "sha256": sha256_file(temp_paths[name]),
                }
                for name in core_names
            },
        }
        _serialize_json(temp_paths["completion"], completion)

        # Recheck immediately before publication. Under --force, remove an old
        # completion marker first so a crash cannot make mixed files look done.
        refuse_existing_outputs(paths, force=force)
        if force and paths["completion"].exists():
            paths["completion"].unlink()
        for name in core_names:
            os.replace(temp_paths[name], paths[name])
        os.replace(temp_paths["completion"], paths["completion"])
    finally:
        for temp_path in temp_paths.values():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only RT55/RT56 query-geometry sensitivity diagnostics"
    )
    parser.add_argument("--config", required=True, help="Resolved evaluation config JSON")
    parser.add_argument(
        "--config-source-mode",
        choices=("resolved", "source", "unspecified"),
        default="unspecified",
        help="Identity label for a run-directory resolved config or source config.",
    )
    parser.add_argument(
        "--deployment-source-mode",
        choices=("git", "uploaded_sha256", "unspecified"),
        default="unspecified",
        help="How the deployed diagnostic source was authenticated by its launcher.",
    )
    parser.add_argument("--checkpoint", required=True, help="Full-model checkpoint")
    parser.add_argument("--protocol", required=True, choices=("normal", "random"))
    parser.add_argument("--split", required=True, help="Must be val/validation/dev")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Deterministically stop after this many event IDs; 0 evaluates all validation events.",
    )
    parser.add_argument(
        "--num-event-shards",
        type=int,
        default=1,
        help=(
            "Split the complete pinned validation set into this many deterministic "
            "whole-event shards; requires --max-events=0."
        ),
    )
    parser.add_argument(
        "--event-shard-id",
        type=int,
        default=0,
        help="Zero-based deterministic event shard to evaluate.",
    )
    parser.add_argument("--station-counts", default="1,3,5,8,12,16")
    parser.add_argument("--radial-scales", default="0,0.5,1,1.5")
    parser.add_argument("--pair-sample-limit", type=int, default=4096)
    parser.add_argument("--equivariance-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--expected-checkpoint-epoch",
        type=int,
        default=None,
        help=(
            "Assert checkpoint metadata epoch before model/dataset inference. "
            "Omit only when using epoch-neutral result naming."
        ),
    )
    parser.add_argument("--checkpoint-sha256", action="store_true")
    parser.add_argument(
        "--encoder-sha256",
        action="store_true",
        help="Compute SHA-256 for the external DiTing pretrained encoder.",
    )
    parser.add_argument(
        "--allow-unsafe-encoder-source-mismatch",
        action="store_true",
        help=(
            "Allow a non-encoder checkpoint to use an encoder path different from "
            "checkpoint encoder_source metadata; the unsafe override is recorded."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--diting-config",
        "--diting_config",
        dest="diting_config",
        default="./diting/config/conf_reg.yml",
    )
    parser.add_argument(
        "--diting-pretrained",
        "--diting_pretrained",
        dest="diting_pretrained",
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *[str(value) for value in argv]]
    )
    args = parse_args(argv)
    split = require_validation_split(args.split)
    station_counts = parse_numeric_list(
        args.station_counts,
        int,
        name="--station-counts",
        positive=True,
    )
    radial_scales = parse_numeric_list(
        args.radial_scales,
        float,
        name="--radial-scales",
    )
    if 1.0 not in radial_scales:
        raise ValueError("--radial-scales must include 1")
    if not math.isfinite(args.equivariance_tolerance) or args.equivariance_tolerance < 0:
        raise ValueError("--equivariance-tolerance must be finite and non-negative")
    if args.num_event_shards < 1:
        raise ValueError("--num-event-shards must be a positive integer")
    if args.event_shard_id < 0 or args.event_shard_id >= args.num_event_shards:
        raise ValueError(
            "--event-shard-id must satisfy 0 <= id < --num-event-shards"
        )
    if args.num_event_shards > 1 and args.max_events:
        raise ValueError("--max-events must be 0 when event sharding is enabled")

    paths = diagnostic_output_paths(args.output_prefix)
    refuse_existing_outputs(paths, force=args.force)
    config_path = Path(args.config).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    diting_config_path = Path(args.diting_config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not diting_config_path.is_file():
        raise FileNotFoundError(f"DiTing config not found: {diting_config_path}")

    checkpoint_identity = inspect_checkpoint_file(checkpoint_path)
    checkpoint_epoch_validation = validate_checkpoint_epoch(
        checkpoint_identity,
        args.expected_checkpoint_epoch,
    )
    config_identity = file_provenance(config_path, compute_sha256=True)
    diting_config_identity = file_provenance(
        diting_config_path, compute_sha256=True
    )
    deployment_source_identity = {
        "mode": args.deployment_source_mode,
        **file_provenance(Path(__file__), compute_sha256=True),
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config_file(str(config_path))
    diting_args = build_diting_args(
        str(diting_config_path),
        device=str(device),
        pretrained_override=args.diting_pretrained,
    )
    encoder_value = getattr(diting_args, "pretrained", None)
    encoder_path = Path(encoder_value) if encoder_value else None
    encoder_source_validation = validate_encoder_source(
        checkpoint_identity,
        encoder_path,
        encoder_was_explicit=bool(args.diting_pretrained),
        allow_unsafe_mismatch=args.allow_unsafe_encoder_source_mismatch,
    )
    diting_encoder_identity = (
        file_provenance(encoder_path, compute_sha256=args.encoder_sha256)
        if encoder_path is not None
        else None
    )

    print(f"[querydiag] repository={REPO_ROOT}")
    print(
        "[querydiag] deployment_source="
        f"mode={deployment_source_identity['mode']} "
        f"sha256={deployment_source_identity['sha256']}"
    )
    print(f"[querydiag] split={split} protocol={args.protocol} device={device}")
    print(f"[querydiag] config={config_path.resolve()}")
    print(f"[querydiag] config_sha256={config_identity['sha256']}")
    print(f"[querydiag] checkpoint={checkpoint_path.resolve()}")
    print(
        "[querydiag] checkpoint_epoch="
        f"expected={args.expected_checkpoint_epoch} "
        f"observed={checkpoint_identity.get('epoch')} "
        f"status={checkpoint_epoch_validation['status']}"
    )
    print(f"[querydiag] checkpoint_format={checkpoint_identity.get('checkpoint_format')}")
    print(f"[querydiag] diting_config={diting_config_path.resolve()}")
    print(f"[querydiag] diting_config_sha256={diting_config_identity['sha256']}")
    print(f"[querydiag] diting_pretrained={encoder_path}")
    print(
        "[querydiag] diting_pretrained_sha256="
        f"{None if diting_encoder_identity is None else diting_encoder_identity['sha256']}"
    )
    print(
        "[querydiag] encoder_source_validation="
        f"{encoder_source_validation['status']}"
    )
    print(f"[querydiag] radial_scales={radial_scales}")
    print(f"[querydiag] requested_station_count_breakdown={station_counts}")
    print(
        "[querydiag] event_shard="
        f"{args.event_shard_id}/{args.num_event_shards} "
        f"algorithm={EVENT_SHARDING_ALGORITHM}"
    )
    model = eval_checkpoint.build_model_and_load(
        config,
        diting_args,
        str(checkpoint_path),
        device,
    )
    raw_model = model.module if hasattr(model, "module") else model
    loaded_metadata = getattr(raw_model, "_eval_checkpoint_metadata", {})
    loaded_epoch = loaded_metadata.get("epoch")
    if loaded_epoch != checkpoint_identity.get("epoch"):
        raise RuntimeError(
            "Checkpoint epoch metadata changed between identity inspection and model "
            f"loading: inspected={checkpoint_identity.get('epoch')!r}, "
            f"loaded={loaded_epoch!r}."
        )
    checkpoint_epoch_validation["loaded_model_metadata_epoch"] = loaded_epoch
    checkpoint_epoch_validation["loaded_model_metadata_matched"] = True
    datasets = eval_checkpoint.build_datasets(config, splits=[split])
    dataset = datasets[split]
    summary, arrays = run_query_geometry_diagnostics(
        model,
        dataset,
        device,
        config,
        protocol=args.protocol,
        station_counts=station_counts,
        radial_scales=radial_scales,
        seed=args.seed,
        max_events=args.max_events,
        num_event_shards=args.num_event_shards,
        event_shard_id=args.event_shard_id,
        pair_sample_limit=args.pair_sample_limit,
        equivariance_tolerance=args.equivariance_tolerance,
    )
    provenance = build_provenance(
        config_identity=config_identity,
        checkpoint_identity=checkpoint_identity,
        checkpoint_epoch_validation=checkpoint_epoch_validation,
        diting_config_identity=diting_config_identity,
        diting_encoder_identity=diting_encoder_identity,
        encoder_source_validation=encoder_source_validation,
        diting_args=diting_args,
        generator_protocol_validation=summary["resolved_validation_generators"],
        config_source_mode=args.config_source_mode,
        deployment_source_identity=deployment_source_identity,
        protocol=args.protocol,
        split=split,
        seed=args.seed,
        station_counts=station_counts,
        radial_scales=radial_scales,
        max_events=args.max_events,
        pair_sample_limit=args.pair_sample_limit,
        equivariance_tolerance=args.equivariance_tolerance,
        checkpoint_sha256=args.checkpoint_sha256,
        invocation_argv=invocation_argv,
        event_sharding=summary["selection"]["event_sharding"],
    )
    write_outputs(
        paths,
        config=config,
        summary=summary,
        arrays=arrays,
        provenance=provenance,
        force=args.force,
    )
    print("[querydiag] complete")
    for name, path in paths.items():
        print(f"[querydiag] {name}={path.resolve()}")
    print(json.dumps(_json_safe(summary["counts"]), sort_keys=True))


if __name__ == "__main__":
    main()
