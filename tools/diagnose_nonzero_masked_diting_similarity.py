#!/usr/bin/env python
"""Diagnose DiTing feature similarity after masking padded-zero waveform tokens.

This is a diagnostic-only script. It does not train. For each selected sample it
runs the DiTing encoder, resamples the raw-waveform nonzero mask to each DiTing
feature length, and compares inter-station cosine similarity before and after
masking.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "diting"))
sys.path.insert(0, os.path.join(REPO_ROOT, "..", "ditingbench"))

import gemini_models as models  # noqa: E402
from eval_checkpoint import build_datasets  # noqa: E402
from train_light import (  # noqa: E402
    CHECKPOINT_ENCODER_PREFIXES,
    build_diting_args,
    clean_state_dict_keys,
    load_config_file,
    load_model_state_dict_compatible,
)


FEATURE_NAMES = ("f2", "f3", "f4", "x")
ROW_METADATA_KEYS = {
    "split",
    "sample_order",
    "matched_sample_order",
    "sample_index",
    "event_id",
    "feature",
    "elapsed_window",
}


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _scalar_to_python(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() <= 0:
            return None
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size <= 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _scalar_to_python(value[0])
    return value


def _p_pick_float(p_picks, key):
    if not isinstance(p_picks, dict) or key not in p_picks:
        return None
    value = _scalar_to_python(p_picks.get(key))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _p_pick_int(p_picks, key):
    value = _p_pick_float(p_picks, key)
    if value is None:
        return None
    return int(round(value))


def _parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    values = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def _collect_elapsed_edges(obj):
    edges = []
    if isinstance(obj, dict):
        train_time_bins = obj.get("train_time_bins")
        if isinstance(train_time_bins, (list, tuple)):
            for item in train_time_bins:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    edges.extend([item[0], item[1]])
        for key in ("train_times", "val_times"):
            values = obj.get(key)
            if isinstance(values, (list, tuple)):
                edges.extend(values)
        for value in obj.values():
            edges.extend(_collect_elapsed_edges(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            edges.extend(_collect_elapsed_edges(value))
    return edges


def _infer_elapsed_bin_edges(config):
    edges = []
    for value in _collect_elapsed_edges(config):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            edges.append(value)
    edges = sorted(set(edges))
    return edges


def _format_elapsed_value(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _elapsed_window_label(elapsed, edges):
    if elapsed is None or not np.isfinite(float(elapsed)):
        return "unknown"
    elapsed = float(elapsed)
    if not edges:
        return "all"
    edges = sorted(float(x) for x in edges if np.isfinite(float(x)))
    if not edges:
        return "all"
    if elapsed < edges[0]:
        return f"<{_format_elapsed_value(edges[0])}"
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= elapsed < high:
            return f"[{_format_elapsed_value(low)},{_format_elapsed_value(high)})"
    return f">={_format_elapsed_value(edges[-1])}"


def _sample_realtime_fields(p_picks):
    fields = {
        "realtime_requested_elapsed_time": _p_pick_float(
            p_picks, "realtime_requested_elapsed_time"
        ),
        "realtime_elapsed_time": _p_pick_float(p_picks, "realtime_elapsed_time"),
        "realtime_current_sample": _p_pick_int(p_picks, "realtime_current_sample"),
        "realtime_first_p_pick_sample": _p_pick_int(
            p_picks, "realtime_first_p_pick_sample"
        ),
        "realtime_time_bin": _p_pick_int(p_picks, "realtime_time_bin"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _elapsed_filter_matches(p_picks, elapsed_times, tolerance):
    if not elapsed_times:
        return True
    candidates = [
        _p_pick_float(p_picks, "realtime_requested_elapsed_time"),
        _p_pick_float(p_picks, "realtime_elapsed_time"),
    ]
    candidates = [x for x in candidates if x is not None and np.isfinite(x)]
    if not candidates:
        return False
    return any(
        abs(candidate - target) <= tolerance
        for candidate in candidates
        for target in elapsed_times
    )


def _summary(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _pairwise_cosine_values(vectors):
    if vectors is None or vectors.ndim != 2 or vectors.shape[0] <= 1:
        return []
    vectors = vectors.float()
    norms = vectors.norm(dim=-1)
    keep = norms > 1e-8
    vectors = vectors[keep]
    if vectors.shape[0] <= 1:
        return []
    vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = vectors @ vectors.T
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    return cos[mask].detach().cpu().tolist()


def _pairwise_summary(vectors):
    return _summary(_pairwise_cosine_values(vectors))


def _feature_to_token_view(feature, encoder_dim=None):
    """Convert one station feature tensor to (T, C)."""
    feat = feature.detach().float()
    if feat.dim() == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    if feat.dim() != 2:
        return None
    if encoder_dim is not None:
        if feat.shape[-1] == encoder_dim:
            return feat
        if feat.shape[0] == encoder_dim:
            return feat.transpose(0, 1).contiguous()
    # DiTing f2/f3/f4 are usually C x T; x is usually T x C.
    if feat.shape[0] > feat.shape[1]:
        return feat.transpose(0, 1).contiguous()
    return feat


def _resample_mask(mask, token_len):
    mask = mask.float().reshape(1, 1, -1)
    if mask.shape[-1] == int(token_len):
        return mask.reshape(-1).bool()
    return F.adaptive_max_pool1d(mask, int(token_len)).reshape(-1).bool()


def _masked_mean(tokens, mask):
    if mask is None or int(mask.sum().item()) <= 0:
        return None
    weights = mask.to(tokens.device, dtype=tokens.dtype)
    return (tokens * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1.0)


def _pairwise_intersection_flat_cos(tokens_by_station, masks_by_station, min_tokens=1):
    values = []
    counts = []
    n = len(tokens_by_station)
    for i in range(n):
        for j in range(i + 1, n):
            common = masks_by_station[i] & masks_by_station[j]
            count = int(common.sum().item())
            if count < min_tokens:
                continue
            vi = tokens_by_station[i][common].reshape(-1).float()
            vj = tokens_by_station[j][common].reshape(-1).float()
            denom = vi.norm() * vj.norm()
            if float(denom) <= 1e-8:
                continue
            values.append(float(torch.dot(vi, vj) / denom.clamp_min(1e-8)))
            counts.append(count)
    return _summary(values), _summary(counts)


def _build_model(config, diting_args, checkpoint_path, device):
    model_params = dict(config["model_params"])
    # This diagnostic only needs waveform_model[0]. Avoid instantiating DPK
    # side-models or temporal heads when a DPK experiment config is reused.
    model_params["station_token_weight_mode"] = "none"
    model_params["temporal_token_weight_mode"] = "none"
    model_params["use_pga_temporal_residual"] = False
    model = models.build_transformer_model(
        **model_params,
        trace_length=10000,
        diting_args=diting_args,
    ).to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = clean_state_dict_keys(state_dict)
        missing, unexpected = load_model_state_dict_compatible(
            model,
            state_dict,
            strict=False,
            context=checkpoint_path,
            allowed_missing_prefixes=tuple(checkpoint.get("excluded_prefixes", CHECKPOINT_ENCODER_PREFIXES))
            if isinstance(checkpoint, dict)
            else CHECKPOINT_ENCODER_PREFIXES,
        )
        print(
            f"[INFO] loaded checkpoint {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.eval()
    return model


@torch.no_grad()
def _encode_station_features(raw_model, waveform_norm, valid_indices, station_batch_size):
    encoder = raw_model.waveform_model[0]
    features_by_name = {name: [] for name in FEATURE_NAMES}
    station_batch_size = max(1, int(station_batch_size))
    for start in range(0, len(valid_indices), station_batch_size):
        chunk = valid_indices[start:start + station_batch_size]
        batch = waveform_norm[:, chunk, :, :].squeeze(0).contiguous()
        outputs = encoder(batch)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        for feat_idx, feature in enumerate(outputs[:len(FEATURE_NAMES)]):
            name = FEATURE_NAMES[feat_idx]
            for local_idx in range(feature.shape[0]):
                features_by_name[name].append(feature[local_idx:local_idx + 1].detach().cpu())
    return features_by_name


def _sample_event_id(dataset, sample_index, p_picks):
    if isinstance(p_picks, dict):
        for key in ("event_id", "event_name", "source_id"):
            if key in p_picks:
                value = p_picks[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                return str(value)
    try:
        index_entry = dataset.indexes[sample_index]
        event = dataset.event_metadata.iloc[index_entry]
        for key in ("event", "event_id", "EVENT_ID", "origin_time", "Time"):
            if key in event:
                return str(event[key])
    except Exception:
        pass
    return ""


@torch.no_grad()
def diagnose_dataset(model, dataset, split, indices, args):
    raw_model = model.module if hasattr(model, "module") else model
    adapter = raw_model.waveform_model[1] if isinstance(raw_model.waveform_model, torch.nn.Sequential) else None
    encoder_dim = getattr(adapter, "encoder_dim", None)
    rows = []
    matched_sample_count = 0

    for sample_order, sample_index in enumerate(indices):
        inputs, _labels, p_picks = dataset[sample_index]
        if not _elapsed_filter_matches(p_picks, args.elapsed_times, args.elapsed_tolerance):
            continue
        if args.max_matching_samples is not None and args.max_matching_samples >= 0:
            if matched_sample_count >= int(args.max_matching_samples):
                break
        matched_sample_order = matched_sample_count
        matched_sample_count += 1

        waveform = inputs[0].unsqueeze(0).to(args.device)
        station_valid = inputs[2].bool().cpu()
        valid_indices = station_valid.nonzero(as_tuple=False).flatten().tolist()
        if len(valid_indices) < args.min_stations:
            continue

        raw_waveform_cpu = inputs[0].detach().cpu()
        raw_nonzero_masks = [
            (raw_waveform_cpu[station_idx].abs().amax(dim=0) > args.nonzero_eps)
            for station_idx in valid_indices
        ]
        nonzero_sample_counts = [int(mask.sum().item()) for mask in raw_nonzero_masks]
        if max(nonzero_sample_counts, default=0) <= 0:
            continue

        waveform_norm = raw_model._normalize(waveform.clone(), mode="std", axis=3)
        valid_mask = station_valid.to(args.device)[None, :, None, None].float()
        waveform_norm = waveform_norm * valid_mask
        features_by_name = _encode_station_features(
            raw_model,
            waveform_norm,
            valid_indices,
            args.station_batch_size,
        )

        event_id = _sample_event_id(dataset, sample_index, p_picks)
        realtime_fields = _sample_realtime_fields(p_picks)
        elapsed_for_group = realtime_fields.get(
            "realtime_requested_elapsed_time",
            realtime_fields.get("realtime_elapsed_time"),
        )
        elapsed_window = _elapsed_window_label(elapsed_for_group, args.elapsed_bin_edges)
        for name in FEATURE_NAMES:
            station_features = features_by_name.get(name) or []
            token_views = [
                _feature_to_token_view(feature, encoder_dim=encoder_dim)
                for feature in station_features
            ]
            token_views = [x for x in token_views if x is not None]
            if len(token_views) < args.min_stations:
                continue
            token_len = int(token_views[0].shape[0])
            token_masks = [_resample_mask(mask, token_len) for mask in raw_nonzero_masks[:len(token_views)]]
            token_counts = [int(mask.sum().item()) for mask in token_masks]
            invalid_counts = [int((~mask).sum().item()) for mask in token_masks]

            full_flat = torch.stack([tokens.reshape(-1) for tokens in token_views])
            full_gap = torch.stack([tokens.mean(dim=0) for tokens in token_views])
            valid_summaries = [
                _masked_mean(tokens, mask)
                for tokens, mask in zip(token_views, token_masks)
            ]
            valid_summaries = [x for x in valid_summaries if x is not None]
            invalid_summaries = [
                _masked_mean(tokens, ~mask)
                for tokens, mask in zip(token_views, token_masks)
            ]
            invalid_summaries = [x for x in invalid_summaries if x is not None]

            intersection_summary, intersection_count_summary = _pairwise_intersection_flat_cos(
                token_views,
                token_masks,
                min_tokens=args.min_pair_tokens,
            )
            row = {
                "split": split,
                "sample_order": sample_order,
                "matched_sample_order": matched_sample_order,
                "sample_index": int(sample_index),
                "event_id": event_id,
                "elapsed_window": elapsed_window,
                "feature": name,
                "station_count": int(len(token_views)),
                "token_len": token_len,
                "raw_nonzero_sample_count_mean": _summary(nonzero_sample_counts)["mean"],
                "raw_nonzero_sample_count_min": _summary(nonzero_sample_counts)["min"],
                "raw_nonzero_sample_count_max": _summary(nonzero_sample_counts)["max"],
                "valid_token_count_mean": _summary(token_counts)["mean"],
                "valid_token_count_min": _summary(token_counts)["min"],
                "valid_token_count_max": _summary(token_counts)["max"],
                "invalid_token_count_mean": _summary(invalid_counts)["mean"],
                "full_flat_cos_mean": _pairwise_summary(full_flat)["mean"],
                "full_gap_cos_mean": _pairwise_summary(full_gap)["mean"],
                "valid_summary_cos_mean": _pairwise_summary(torch.stack(valid_summaries))
                ["mean"] if len(valid_summaries) >= args.min_stations else None,
                "invalid_summary_cos_mean": _pairwise_summary(torch.stack(invalid_summaries))
                ["mean"] if len(invalid_summaries) >= args.min_stations else None,
                "valid_intersection_flat_cos_mean": intersection_summary["mean"],
                "valid_intersection_flat_pair_count": intersection_summary["count"],
                "valid_intersection_token_count_mean": intersection_count_summary["mean"],
                "valid_intersection_token_count_min": intersection_count_summary["min"],
                "valid_intersection_token_count_max": intersection_count_summary["max"],
            }
            row.update(realtime_fields)
            rows.append(row)

        if (sample_order + 1) % max(1, args.log_every) == 0:
            print(
                f"[INFO] {split}: scanned {sample_order + 1}/{len(indices)} "
                f"selected samples, matched={matched_sample_count}"
            )

    return rows


def _write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        feature = row["feature"]
        for key, value in row.items():
            if key in ROW_METADATA_KEYS:
                continue
            if isinstance(value, (int, float)) and value is not None:
                grouped[feature][key].append(value)
    summary = {}
    for feature, metrics in grouped.items():
        summary[feature] = {key: _summary(values) for key, values in metrics.items()}
    return summary


def _aggregate_rows_by_elapsed_window(rows):
    windows = defaultdict(list)
    for row in rows:
        windows[row.get("elapsed_window", "unknown")].append(row)
    summary = {}
    for window in sorted(windows):
        summary[window] = {
            "row_count": len(windows[window]),
            "metrics": _aggregate_rows(windows[window]),
        }
    return summary


def _select_indices(dataset, args):
    all_indices = list(range(int(args.start_index), len(dataset), max(1, int(args.stride))))
    if args.max_samples is not None and args.max_samples >= 0:
        all_indices = all_indices[:int(args.max_samples)]
    return all_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DiTing inter-station feature cosine after oracle nonzero masking."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--diting_config", default="./diting/config/conf_reg.yml")
    parser.add_argument("--diting_pretrained", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional TEAM checkpoint. Not required for frozen encoder diagnostics.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "both"))
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--input_station_selection", default="config",
                        choices=("config", "default", "random", "p_pick", "epidist"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--station_batch_size", type=int, default=1)
    parser.add_argument("--min_stations", type=int, default=2)
    parser.add_argument("--min_pair_tokens", type=int, default=1)
    parser.add_argument("--nonzero_eps", type=float, default=1e-8)
    parser.add_argument("--elapsed_times", default="",
                        help="Comma-separated realtime elapsed seconds to keep, e.g. '3' or '1,3,5'.")
    parser.add_argument("--elapsed_tolerance", type=float, default=1e-4)
    parser.add_argument("--elapsed_bin_edges", default="",
                        help="Comma-separated elapsed-time bin edges for grouped summary. "
                             "Default infers train_time_bins/val_times from config.")
    parser.add_argument("--max_matching_samples", type=int, default=-1,
                        help="Stop after this many samples matching elapsed_times. -1 disables.")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()
    args.elapsed_times = _parse_float_list(args.elapsed_times)
    args.elapsed_bin_edges = _parse_float_list(args.elapsed_bin_edges)
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config_file(args.config)
    args.overfit_n = args.overfit_n or int(config.get("training_params", {}).get("overfit_n", 0))
    args.device = torch.device(args.device)
    if not args.elapsed_bin_edges:
        args.elapsed_bin_edges = _infer_elapsed_bin_edges(config)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    diting_args = build_diting_args(
        args.diting_config,
        device=str(args.device),
        pretrained_override=args.diting_pretrained,
    )
    print("[INFO] building model")
    model = _build_model(config, diting_args, args.checkpoint, args.device)

    print("[INFO] building datasets")
    datasets = build_datasets(
        config,
        overfit_n=args.overfit_n,
        input_station_selection=args.input_station_selection,
    )

    selected_splits = ("train", "val") if args.split == "both" else (args.split,)
    all_rows = []
    run_info = {
        "config": os.path.abspath(args.config),
        "diting_config": os.path.abspath(args.diting_config),
        "diting_pretrained": args.diting_pretrained,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "overfit_n": args.overfit_n,
        "max_samples": args.max_samples,
        "max_matching_samples": args.max_matching_samples,
        "station_batch_size": args.station_batch_size,
        "nonzero_eps": args.nonzero_eps,
        "elapsed_times": args.elapsed_times,
        "elapsed_tolerance": args.elapsed_tolerance,
        "elapsed_bin_edges": args.elapsed_bin_edges,
    }
    for split in selected_splits:
        dataset = datasets[split]
        indices = _select_indices(dataset, args)
        print(f"[INFO] diagnosing split={split}, dataset_size={len(dataset)}, selected={len(indices)}")
        rows = diagnose_dataset(model, dataset, split, indices, args)
        print(f"[INFO] split={split}, feature rows={len(rows)}")
        all_rows.extend(rows)

    per_sample_path = os.path.join(args.output_dir, "per_sample.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    _write_csv(per_sample_path, all_rows)
    summary = {
        "run_info": run_info,
        "row_count": len(all_rows),
        "metrics": _aggregate_rows(all_rows),
        "metrics_by_elapsed_window": _aggregate_rows_by_elapsed_window(all_rows),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] wrote {per_sample_path}")
    print(f"[INFO] wrote {summary_path}")
    for feature in FEATURE_NAMES:
        metrics = summary["metrics"].get(feature, {})
        full = metrics.get("full_flat_cos_mean", {}).get("mean")
        valid = metrics.get("valid_summary_cos_mean", {}).get("mean")
        inter = metrics.get("valid_intersection_flat_cos_mean", {}).get("mean")
        counts = metrics.get("valid_token_count_mean", {}).get("mean")
        print(
            f"[SUMMARY] {feature}: "
            f"full_flat={full}, valid_summary={valid}, "
            f"valid_intersection_flat={inter}, valid_tokens_mean={counts}"
        )
    for window, window_summary in summary["metrics_by_elapsed_window"].items():
        print(f"[SUMMARY_BY_WINDOW] {window}: row_count={window_summary['row_count']}")
        for feature in FEATURE_NAMES:
            metrics = window_summary["metrics"].get(feature, {})
            full = metrics.get("full_flat_cos_mean", {}).get("mean")
            valid = metrics.get("valid_summary_cos_mean", {}).get("mean")
            inter = metrics.get("valid_intersection_flat_cos_mean", {}).get("mean")
            counts = metrics.get("valid_token_count_mean", {}).get("mean")
            print(
                f"[SUMMARY_BY_WINDOW] {window} {feature}: "
                f"full_flat={full}, valid_summary={valid}, "
                f"valid_intersection_flat={inter}, valid_tokens_mean={counts}"
            )


if __name__ == "__main__":
    main()
