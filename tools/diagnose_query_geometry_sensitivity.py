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

Existing outputs are refused unless ``--force`` is supplied.  Only validation
is accepted by design.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint  # noqa: E402
from train_light import build_diting_args, load_config_file  # noqa: E402


PGA_COORDINATE = "log10(m/s^2)"
TARGET_TYPE_NAMES = {
    0: "input",
    1: "triggered_noninput",
    2: "untriggered",
}


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
    }


def refuse_existing_outputs(
    paths: Mapping[str, Path],
    *,
    force: bool = False,
) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not force:
        formatted = "\n  ".join(existing)
        raise FileExistsError(
            "Refusing to overwrite existing diagnostic output(s):\n  "
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def validate_dataset_protocol(dataset: Any, requested_protocol: str) -> List[Dict[str, Any]]:
    requested_protocol = str(requested_protocol).strip().lower()
    if requested_protocol not in {"normal", "random"}:
        raise ValueError(f"Unknown protocol {requested_protocol!r}; expected normal or random")
    details: List[Dict[str, Any]] = []
    for index, generator in enumerate(_leaf_generators(dataset)):
        mask = getattr(generator, "causal_random_input_mask", None) or {"enabled": False}
        enabled = bool(mask.get("enabled", False))
        probability = float(mask.get("apply_probability", 1.0 if enabled else 0.0))
        target_sampling = mask.get("target_sampling") or {}
        detail = {
            "generator_index": index,
            "enabled": enabled,
            "apply_probability": probability,
            "station_counts": list(mask.get("station_counts", [])),
            "targets_exclude_inputs": bool(target_sampling.get("exclude_inputs", False)),
        }
        details.append(detail)
        if requested_protocol == "normal" and enabled and probability > 0.0:
            raise ValueError(
                "--protocol normal does not match the resolved validation generator: "
                f"causal random masking is enabled with probability {probability}."
            )
        if requested_protocol == "random" and (not enabled or probability != 1.0):
            raise ValueError(
                "--protocol random requires the resolved validation generator to use "
                "causal random masking with apply_probability=1.0; got "
                f"enabled={enabled}, probability={probability}."
            )
    return details


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


def _summarize_run(
    arrays: Mapping[str, np.ndarray],
    radial_scales: Sequence[float],
    station_counts: Sequence[int],
    equivariance_tolerance: float,
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
    pair_sample_limit: int = 4096,
    equivariance_tolerance: float = 1e-5,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Run baseline, radial interventions, and query-order sanity checks."""
    protocol_details = validate_dataset_protocol(dataset, protocol)
    model.eval()
    if max_events < 0:
        raise ValueError("max_events must be zero (all) or a positive integer")
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

    for sample_index in range(len(dataset)):
        inputs, labels, info = dataset[sample_index]
        event_id = _event_id(info, sample_index)
        dataset_source_index = _dataset_source_index(dataset, sample_index)
        event_key = f"{dataset_source_index}|{event_id}"
        if event_key not in selected_event_set:
            if max_events and len(selected_event_ids) >= max_events:
                break
            selected_event_ids.append(event_key)
            selected_event_set.add(event_key)
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
    )
    summary["selection"] = {
        "dataset_realtime_samples": int(len(dataset)),
        "examined_realtime_samples": int(examined_samples),
        "max_events": int(max_events),
        "selected_events": int(len(selected_event_ids)),
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
    config_path: Path,
    checkpoint_path: Path,
    model: torch.nn.Module,
    protocol: str,
    split: str,
    seed: int,
    station_counts: Sequence[int],
    radial_scales: Sequence[float],
    max_events: int,
    checkpoint_sha256: bool,
) -> Dict[str, Any]:
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint_stat = checkpoint_path.stat()
    loaded_metadata = getattr(raw_model, "_eval_checkpoint_metadata", {})
    checkpoint_metadata = {
        key: loaded_metadata.get(key)
        for key in ("epoch", "loss", "checkpoint_format", "encoder_source")
    }
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "file_size_bytes": int(checkpoint_stat.st_size),
        "metadata": checkpoint_metadata,
        "sha256": sha256_file(checkpoint_path) if checkpoint_sha256 else None,
        "sha256_computed": bool(checkpoint_sha256),
    }
    return {
        "repository": git_provenance(REPO_ROOT),
        "config_path": str(config_path.resolve()),
        "checkpoint": checkpoint,
        "protocol": protocol,
        "split": split,
        "seed": int(seed),
        "station_counts": [int(value) for value in station_counts],
        "radial_scales": [float(value) for value in radial_scales],
        "max_events": int(max_events),
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
        "schema_version": 1,
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
    # Write temporary files and publish final names only after all serialization
    # succeeds.  Existing final files were checked before the expensive run and
    # are checked again here to catch concurrent writers.
    refuse_existing_outputs(paths, force=force)
    _atomic_json(paths["resolved_config"], config)
    _atomic_npz(paths["samples"], npz_arrays)
    _atomic_json(paths["summary"], payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only RT55/RT56 query-geometry sensitivity diagnostics"
    )
    parser.add_argument("--config", required=True, help="Resolved evaluation config JSON")
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
    parser.add_argument("--station-counts", default="1,3,5,8,12,16")
    parser.add_argument("--radial-scales", default="0,0.5,1,1.5")
    parser.add_argument("--pair-sample-limit", type=int, default=4096)
    parser.add_argument("--equivariance-tolerance", type=float, default=1e-5)
    parser.add_argument("--checkpoint-sha256", action="store_true")
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

    paths = diagnostic_output_paths(args.output_prefix)
    refuse_existing_outputs(paths, force=args.force)
    config_path = Path(args.config).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config_file(str(config_path))
    diting_args = build_diting_args(
        args.diting_config,
        device=str(device),
        pretrained_override=args.diting_pretrained,
    )

    print(f"[querydiag] repository={REPO_ROOT}")
    print(f"[querydiag] split={split} protocol={args.protocol} device={device}")
    print(f"[querydiag] config={config_path.resolve()}")
    print(f"[querydiag] checkpoint={checkpoint_path.resolve()}")
    print(f"[querydiag] radial_scales={radial_scales}")
    print(f"[querydiag] requested_station_count_breakdown={station_counts}")
    model = eval_checkpoint.build_model_and_load(
        config,
        diting_args,
        str(checkpoint_path),
        device,
    )
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
        pair_sample_limit=args.pair_sample_limit,
        equivariance_tolerance=args.equivariance_tolerance,
    )
    provenance = build_provenance(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=model,
        protocol=args.protocol,
        split=split,
        seed=args.seed,
        station_counts=station_counts,
        radial_scales=radial_scales,
        max_events=args.max_events,
        checkpoint_sha256=args.checkpoint_sha256,
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
