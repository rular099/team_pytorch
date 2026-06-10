#!/usr/bin/env python3
"""Precompute DPK token priors for DiTing station pooling.

This script compares two prior sources:
1. current MAE/pretrained encoder + DPK head
2. DPK fine-tuned encoder + DPK head

It is intentionally independent from training. Each distributed rank processes a
strided shard of the configured dataset and writes an NPZ prior shard plus CSV
diagnostics. Rank 0 also writes model comparison and aggregate summaries.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "diting"))
sys.path.insert(0, str(REPO_ROOT.parent / "ditingbench"))

import gemini_models as models  # noqa: E402
import gemini_util_light as util  # noqa: E402
import loader_light as loader  # noqa: E402
from train_light import (  # noqa: E402
    build_diting_args,
    build_overfit_event_metadata_splits,
    indexed_config_override,
    load_config_file,
    read_overfit_event_ids,
)


DEFAULT_DPK_CKPT = (
    "/public/home/test_bigmodel/seismogram/mx/ckpt/1200m_dpk/"
    "mae_init/720w_sft/model-4-latest.pth"
)

LEVEL_NAMES = ("f2", "f3", "f4", "x")
SOURCE_NAMES = ("mae_head", "dpk_finetuned")
MODE_NAMES = ("event", "all")


def log_rank(rank, message):
    print(f"[rank {rank}] {message}", flush=True)


def normalize_waveform(waveform):
    """Match FullModel._normalize(..., mode='std', axis=3)."""
    data = waveform - torch.mean(waveform, dim=3, keepdim=True)
    std = torch.std(data, dim=3, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return data / std


def canonical_dpk_outputs(outputs):
    if isinstance(outputs, dict):
        result = {}
        for key, value in outputs.items():
            if value is None:
                continue
            if isinstance(value, list):
                value = value[0]
            if value.dim() == 3 and value.shape[1] == 1:
                value = value.squeeze(1)
            result[key] = value
        return result

    if torch.is_tensor(outputs):
        if outputs.dim() != 3:
            raise ValueError(f"Expected tensor DPK output with shape (B,C,T), got {tuple(outputs.shape)}")
        if outputs.shape[1] < 3:
            raise ValueError(f"Tensor DPK output needs at least 3 channels, got {tuple(outputs.shape)}")
        return {
            "det": outputs[:, 0, :],
            "ppk": outputs[:, 1, :],
            "spk": outputs[:, 2, :],
        }

    raise TypeError(f"Unsupported DPK output type: {type(outputs)!r}")


def dpk_signal(outputs, mode, floor=1e-4, temperature=1.0):
    outputs = canonical_dpk_outputs(outputs)
    if "det" not in outputs:
        raise ValueError("DPK outputs do not contain det/event channel")

    if mode == "event":
        signal = outputs["det"]
    elif mode == "all":
        parts = [outputs["det"]]
        for key in ("ppk", "spk"):
            if key in outputs:
                parts.append(outputs[key])
        signal = torch.stack(parts, dim=0).amax(dim=0)
    else:
        raise ValueError(f"Unsupported prior mode: {mode}")

    if signal.dim() != 2:
        raise ValueError(f"Expected DPK signal shape (B,T), got {tuple(signal.shape)}")
    if temperature != 1.0:
        signal = signal.clamp_min(floor).pow(1.0 / temperature)
    return signal


def resample_signal(signal, token_len, mode="max"):
    if signal.shape[-1] == token_len:
        return signal.float()
    x = signal.unsqueeze(1).float()
    if mode == "avg":
        out = F.adaptive_avg_pool1d(x, token_len)
    elif mode == "max":
        out = F.adaptive_max_pool1d(x, token_len)
    else:
        raise ValueError(f"Unsupported resample mode: {mode}")
    return out.squeeze(1)


def normalize_prior(signal, floor=1e-4):
    signal = signal.float().clamp_min(floor)
    signal = signal / signal.mean(dim=-1, keepdim=True).clamp_min(floor)
    return signal.clamp_min(floor)


def effective_token_count(prior):
    prob = prior.float().clamp_min(1e-12)
    prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(prob * prob.log()).sum(dim=-1)
    return entropy.exp()


def pearson_per_row(a, b):
    a = a.float()
    b = b.float()
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    denom = torch.sqrt((a * a).sum(dim=-1) * (b * b).sum(dim=-1)).clamp_min(1e-12)
    return (a * b).sum(dim=-1) / denom


def to_numpy_dtype(x, dtype_name):
    arr = x.detach().cpu().numpy()
    if dtype_name == "float16":
        return arr.astype(np.float16)
    if dtype_name == "float32":
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def apply_model_param_overrides(diting_args, model_params, station_emb_dim):
    diting_frontend = model_params.get("diting_frontend", None)
    if diting_frontend is not None:
        diting_args.diting_frontend = diting_frontend
    diting_args.diting_station_pool_queries = model_params.get("diting_station_pool_queries", 4)
    diting_args.diting_station_pool_temperature = model_params.get("diting_station_pool_temperature", 1.0)
    diting_args.diting_station_pool_dropout = model_params.get("diting_station_pool_dropout", 0.0)
    diting_args.diting_station_metadata_mode = model_params.get("diting_station_metadata_mode", "none")
    diting_args.diting_station_metadata_dim = (
        station_emb_dim if diting_args.diting_station_metadata_mode != "none" else None
    )
    diting_args.diting_station_metadata_hidden_dim = model_params.get("diting_station_metadata_hidden_dim", 128)
    diting_args.diting_station_metadata_scale = model_params.get("diting_station_metadata_scale", 0.1)


def load_split_generators(config, split, limit=None, rank=0):
    training_params = config["training_params"]
    model_params = config["model_params"]
    generator_params = training_params.get("generator_params", [training_params.copy()])
    generator_params = [copy.deepcopy(g) for g in generator_params]

    data_paths = training_params["data_path"]
    if not isinstance(data_paths, list):
        data_paths = [data_paths]
        training_params["data_path"] = data_paths
    if len(data_paths) != len(generator_params):
        raise ValueError("training_params.data_path and generator_params length mismatch")

    overwrite_sampling_rate = training_params.get("overwrite_sampling_rate", None)
    min_stalta_ratio_at_pick = training_params.get("min_stalta_ratio_at_pick", 0.1)

    def load_events_for(data_path, generator, metadata_name, parts):
        return loader.load_events(
            data_path,
            event_metadata_path=metadata_name,
            limit=limit,
            parts=parts,
            shuffle_train_dev=generator.get("shuffle_train_dev", False),
            custom_split=generator.get("custom_split", None),
            min_mag=generator.get("min_mag", None),
            mag_key=generator.get("key", "MA"),
            overwrite_sampling_rate=overwrite_sampling_rate,
            decimate_events=generator.get("decimate_events", None),
            min_stalta_ratio_at_pick=min_stalta_ratio_at_pick,
        )

    full_data_train = [
        load_events_for(data_path, generator, "train_ev.csv", (True, False, False))
        for data_path, generator in zip(data_paths, generator_params)
    ]
    full_data_dev = [
        load_events_for(data_path, generator, "test_ev.csv", (False, True, False))
        for data_path, generator in zip(data_paths, generator_params)
    ]
    full_data_test = [
        load_events_for(data_path, generator, "test_ev.csv", (False, False, True))
        for data_path, generator in zip(data_paths, generator_params)
    ]

    event_metadata = {
        "train": [d[0] for d in full_data_train],
        "dev": [d[0] for d in full_data_dev],
        "test": [d[0] for d in full_data_test],
    }
    metadata = {
        "train": [d[2] for d in full_data_train],
        "dev": [d[2] for d in full_data_dev],
        "test": [d[2] for d in full_data_test],
    }

    overfit_n = int(training_params.get("overfit_n", 0) or 0)
    if overfit_n > 0:
        full_data_all = [
            load_events_for(data_path, generator, "overfit_ev.csv", None)
            for data_path, generator in zip(data_paths, generator_params)
        ]
        fixed_overfit_ids = None
        if training_params.get("overfit_event_ids_path"):
            if len(full_data_all) != 1:
                raise ValueError("overfit_event_ids_path currently supports exactly one data_path")
            fixed_overfit_ids = [read_overfit_event_ids(training_params["overfit_event_ids_path"])]
        event_train, event_dev, event_test, _ = build_overfit_event_metadata_splits(
            full_data_all,
            generator_params,
            overfit_n,
            selected_event_ids=fixed_overfit_ids,
        )
        event_metadata = {"train": event_train, "dev": event_dev, "test": event_test}

        adjusted = []
        for generator in generator_params:
            generator = copy.deepcopy(generator)
            realtime_cfg = generator.get("realtime_training") or {}
            if realtime_cfg.get("enabled", False):
                generator["trigger_based"] = True
                generator["disable_station_foreshadowing"] = True
            else:
                fixed_cutout = generator.get("cutout_end", generator.get("cutout_start", 0))
                generator["trigger_based"] = False
                generator["disable_station_foreshadowing"] = False
                generator["cutout_start"] = fixed_cutout
                generator["cutout_end"] = fixed_cutout
            generator["shuffle_train_dev"] = False
            generator["oversample"] = 1
            adjusted.append(generator)
        generator_params = adjusted
        log_rank(rank, f"overfit mode enabled: overfit_n={overfit_n}, split={split}")

    sampling_rate = metadata["train"][0]["sampling_rate"]
    for split_meta in metadata.values():
        for item in split_meta:
            if item["sampling_rate"] != sampling_rate:
                raise ValueError("All datasets must use the same sampling rate")

    max_stations = model_params["max_stations"]
    n_pga_targets = model_params.get("n_pga_targets", 0)
    no_event_token = model_params.get("no_event_token", False)
    station_experiment_cfg = training_params.get("station_experiment", None)
    train_generator_overrides = training_params.get("train_generator_overrides", None)
    validation_generator_overrides = training_params.get("validation_generator_overrides", None)

    def make_generator(dataset_split, index, generator_src):
        generator = copy.deepcopy(generator_src)
        noise_seconds = generator.get("noise_seconds", 5)
        cutout = (
            int(round(sampling_rate * (noise_seconds + generator["cutout_start"]))),
            int(round(sampling_rate * (noise_seconds + generator["cutout_end"]))),
        )
        generator["transform_target_only"] = generator.get("transform_target_only", True)
        defaults = dict(
            coords_target=True,
            label_smoothing=False if overfit_n > 0 else True,
            station_blinding=False,
            cutout=cutout,
            pga_targets=n_pga_targets,
            max_stations=max_stations,
            sampling_rate=sampling_rate,
            no_event_token=no_event_token,
            dump_debug_snapshot=False,
            use_coords_rel=model_params.get("use_coords_rel", False),
            use_coords_abs=model_params.get("use_coords_abs", True),
            use_coords_rel_abs_fusion=model_params.get("use_coords_rel_abs_fusion", False),
            use_vs30=model_params.get("use_vs30", False),
            station_experiment=station_experiment_cfg,
        )
        if training_params.get("deterministic_sampling", False):
            defaults["deterministic_sampling_seed"] = int(config.get("seed", 42)) + index * 1000003

        if dataset_split == "train":
            override = indexed_config_override(train_generator_overrides, index)
        else:
            old_oversample = generator.get("oversample", 1)
            generator["oversample"] = 1 if overfit_n > 0 else 4
            override = indexed_config_override(validation_generator_overrides, index)
            _ = old_oversample

        merged = {**defaults, **generator, **override}
        return util.PreloadedEventGenerator(
            event_metadata=event_metadata[dataset_split][index],
            metadata=metadata[dataset_split][index],
            data_path=data_paths[index],
            generator_params=generator,
            **merged,
        )

    requested_splits = ("train", "dev", "test") if split == "all" else (split,)
    generators = []
    for dataset_split in requested_splits:
        for index, generator in enumerate(generator_params):
            generators.append((dataset_split, index, make_generator(dataset_split, index, generator)))
    return generators


def build_models(config, args, device, is_dist):
    diting_args = build_diting_args(
        args.diting_config,
        device=device,
        distributed=is_dist,
        pretrained_override=args.diting_pretrained,
    )
    station_emb_dim = config["model_params"]["waveform_model_dims"][-1]
    apply_model_param_overrides(diting_args, config["model_params"], station_emb_dim)

    current_model = models.get_diting_model(diting_args, station_emb_dim=station_emb_dim)
    current_model.eval()
    current_encoder = current_model[0].to(device).eval()

    dpk_model, dpk_load_msg = models._load_dpk_model_from_checkpoint(args.dpk_checkpoint)
    dpk_model.eval()
    dpk_encoder = dpk_model[0]
    dpk_head = dpk_model[1]
    compare = models._compare_state_dicts_exact(current_encoder, dpk_encoder)
    compare.update(
        {
            "diting_pretrained": getattr(diting_args, "pretrained", ""),
            "diting_config": args.diting_config,
            "dpk_checkpoint": args.dpk_checkpoint,
            "dpk_load_missing_keys": len(getattr(dpk_load_msg, "missing_keys", [])),
            "dpk_load_unexpected_keys": len(getattr(dpk_load_msg, "unexpected_keys", [])),
        }
    )

    if args.share_if_identical and compare["all_equal"]:
        dpk_head.to(device).eval()
        dpk_model_ref = None
        compare["runtime_policy"] = "shared_current_encoder"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        dpk_model.to(device).eval()
        dpk_model_ref = dpk_model
        dpk_head = dpk_model[1]
        compare["runtime_policy"] = "separate_dpk_encoder"

    for module in (current_encoder, dpk_head):
        for param in module.parameters():
            param.requires_grad = False
    if dpk_model_ref is not None:
        for param in dpk_model_ref.parameters():
            param.requires_grad = False

    return current_encoder, dpk_model_ref, dpk_head, compare


def append_prior_arrays(array_store, source, mode, level, tensor, dtype_name):
    key = f"{source}_{mode}_{level}"
    array_store.setdefault(key, []).append(to_numpy_dtype(tensor, dtype_name))


def stat_float(x, index):
    return float(x[index].detach().cpu().item())


def process_station_batch(
    wave_batch,
    current_encoder,
    dpk_model,
    dpk_head,
    token_floor,
    weight_temperature,
    resample_mode,
    save_dtype,
):
    with torch.no_grad():
        current_features = current_encoder(wave_batch)
        if not isinstance(current_features, (list, tuple)):
            raise ValueError("Expected ViTAdapter current encoder to return f2/f3/f4/x feature list")
        encoder_dim = current_features[-1].shape[-1]
        token_lengths = [
            feat.shape[1] if feat.shape[-1] == encoder_dim else feat.shape[-1]
            for feat in current_features
        ]
        current_outputs = dpk_head(list(current_features))
        if dpk_model is None:
            dpk_outputs = dpk_head(list(current_features))
        else:
            dpk_outputs = dpk_model(wave_batch)

        result = {"arrays": {}, "stats": {}, "token_lengths": token_lengths}
        raw_outputs = {"mae_head": current_outputs, "dpk_finetuned": dpk_outputs}

        priors = {}
        for source in SOURCE_NAMES:
            priors[source] = {}
            for mode in MODE_NAMES:
                signal = dpk_signal(
                    raw_outputs[source],
                    mode=mode,
                    floor=token_floor,
                    temperature=weight_temperature,
                )
                priors[source][mode] = {}
                for level, token_len in zip(LEVEL_NAMES, token_lengths):
                    prior = normalize_prior(
                        resample_signal(signal, token_len, mode=resample_mode),
                        floor=token_floor,
                    )
                    priors[source][mode][level] = prior
                    append_prior_arrays(result["arrays"], source, mode, level, prior, save_dtype)
                    result["stats"][f"{source}_{mode}_{level}_effective"] = effective_token_count(prior)
                    result["stats"][f"{source}_{mode}_{level}_mean"] = prior.mean(dim=-1)
                    result["stats"][f"{source}_{mode}_{level}_max"] = prior.max(dim=-1).values

        for mode in MODE_NAMES:
            for level in LEVEL_NAMES:
                a = priors["mae_head"][mode][level]
                b = priors["dpk_finetuned"][mode][level]
                result["stats"][f"compare_{mode}_{level}_corr"] = pearson_per_row(a, b)
                result["stats"][f"compare_{mode}_{level}_l1"] = torch.mean(torch.abs(a - b), dim=-1)
                result["stats"][f"compare_{mode}_{level}_max_abs"] = torch.max(torch.abs(a - b), dim=-1).values

        return result


def sample_scalar(info, key, default=np.nan):
    value = info.get(key)
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.detach().cpu().reshape(-1)[0].item()
    return value


def process_dataset(args, generators, current_encoder, dpk_model, dpk_head, device, rank, world_size):
    array_store = {}
    rows = []
    failures = []
    processed_samples = 0
    processed_stations = 0
    started = time.time()

    for split_name, dataset_id, dataset in generators:
        total = len(dataset)
        log_rank(rank, f"processing split={split_name} dataset_id={dataset_id} len={total}")
        for sample_index in range(rank, total, world_size):
            if args.max_samples is not None and processed_samples >= args.max_samples:
                break
            try:
                inputs, _, p_pick_info = dataset[sample_index]
            except Exception as exc:  # Keep long precompute jobs from dying on one bad cutout.
                failures.append(
                    {
                        "split": split_name,
                        "dataset_id": dataset_id,
                        "sample_index": sample_index,
                        "error": repr(exc),
                    }
                )
                continue

            waveforms = inputs[0].unsqueeze(0).to(device, non_blocking=True)
            station_valid = inputs[2].bool().to(device, non_blocking=True)
            waveforms_norm = normalize_waveform(waveforms)
            waveforms_norm = waveforms_norm * station_valid[None, :, None, None].float()
            valid_slots = torch.where(station_valid)[0]
            if valid_slots.numel() == 0:
                continue

            selected_input_indices = p_pick_info.get("selected_input_indices")
            event_id = str(p_pick_info.get("event_id", ""))
            realtime_current_sample = sample_scalar(p_pick_info, "realtime_current_sample")
            realtime_elapsed_time = sample_scalar(p_pick_info, "realtime_elapsed_time")
            realtime_time_bin = sample_scalar(p_pick_info, "realtime_time_bin")

            for start in range(0, valid_slots.numel(), args.station_batch_size):
                slots = valid_slots[start : start + args.station_batch_size]
                station_wave = waveforms_norm[0, slots, :, :]
                batch_result = process_station_batch(
                    station_wave,
                    current_encoder=current_encoder,
                    dpk_model=dpk_model,
                    dpk_head=dpk_head,
                    token_floor=args.token_floor,
                    weight_temperature=args.dpk_weight_temperature,
                    resample_mode=args.dpk_weight_resample,
                    save_dtype=args.save_dtype,
                )

                for key, values in batch_result["arrays"].items():
                    array_store.setdefault(key, []).extend(values)

                for local_i, slot_tensor in enumerate(slots):
                    slot = int(slot_tensor.detach().cpu().item())
                    if selected_input_indices is not None and slot < len(selected_input_indices):
                        original_station_index = int(selected_input_indices[slot].detach().cpu().item())
                    else:
                        original_station_index = slot
                    row = {
                        "split": split_name,
                        "dataset_id": dataset_id,
                        "sample_index": sample_index,
                        "event_id": event_id,
                        "station_slot": slot,
                        "original_station_index": original_station_index,
                        "realtime_current_sample": realtime_current_sample,
                        "realtime_elapsed_time": realtime_elapsed_time,
                        "realtime_time_bin": realtime_time_bin,
                    }
                    for name, tensor in batch_result["stats"].items():
                        row[name] = stat_float(tensor, local_i)
                    rows.append(row)
                processed_stations += int(slots.numel())

            processed_samples += 1
            if processed_samples % args.log_every == 0:
                elapsed = time.time() - started
                log_rank(
                    rank,
                    f"samples={processed_samples} stations={processed_stations} "
                    f"elapsed={elapsed / 60.0:.1f} min",
                )

    return array_store, rows, failures, processed_samples, processed_stations


def write_rank_outputs(out_dir, rank, array_store, rows, failures, processed_samples, processed_stations):
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"dpk_priors_rank{rank:03d}.npz"
    csv_path = out_dir / f"station_prior_stats_rank{rank:03d}.csv"
    failure_path = out_dir / f"failures_rank{rank:03d}.json"
    summary_path = out_dir / f"summary_rank{rank:03d}.json"

    for row_index, row in enumerate(rows):
        row["rank"] = rank
        row["row_in_shard"] = row_index
        row["sample_station_key"] = (
            f'{row["split"]}|{row["dataset_id"]}|'
            f'{row["sample_index"]}|{row["station_slot"]}'
        )
        row["event_station_time_key"] = (
            f'{row["split"]}|{row["dataset_id"]}|{row["event_id"]}|'
            f'{row["realtime_current_sample"]}|{row["original_station_index"]}'
        )

    npz_payload = {}
    for key, chunks in array_store.items():
        if chunks:
            npz_payload[key] = np.concatenate(chunks, axis=0)
    if rows:
        npz_payload["meta_split"] = np.asarray([row["split"] for row in rows], dtype="U16")
        npz_payload["meta_event_id"] = np.asarray([row["event_id"] for row in rows], dtype="U64")
        npz_payload["meta_sample_station_key"] = np.asarray(
            [row["sample_station_key"] for row in rows],
            dtype="U128",
        )
        npz_payload["meta_event_station_time_key"] = np.asarray(
            [row["event_station_time_key"] for row in rows],
            dtype="U160",
        )
        for key in (
            "rank",
            "row_in_shard",
            "dataset_id",
            "sample_index",
            "station_slot",
            "original_station_index",
            "realtime_current_sample",
            "realtime_elapsed_time",
            "realtime_time_bin",
        ):
            npz_payload[f"meta_{key}"] = np.asarray([row[key] for row in rows])
    np.savez_compressed(npz_path, **npz_payload)

    fieldnames = sorted({field for row in rows for field in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(failure_path, "w") as f:
        json.dump(failures, f, indent=2)

    summary = {
        "rank": rank,
        "processed_samples": processed_samples,
        "processed_stations": processed_stations,
        "failed_samples": len(failures),
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
        "failure_path": str(failure_path),
        "array_shapes": {key: list(value.shape) for key, value in npz_payload.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def aggregate_rank_outputs(out_dir, world_size, model_compare, args):
    summaries = []
    rows = []
    for rank in range(world_size):
        summary_path = out_dir / f"summary_rank{rank:03d}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries.append(json.load(f))
        csv_path = out_dir / f"station_prior_stats_rank{rank:03d}.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                shard_path = out_dir / f"dpk_priors_rank{rank:03d}.npz"
                for row in reader:
                    row = dict(row)
                    row["shard_path"] = str(shard_path)
                    rows.append(row)

    hdf5_path = None
    if not args.skip_hdf5_merge:
        hdf5_path = write_hdf5_cache(out_dir, summaries, rows, args, model_compare)

    def mean_of(field):
        values = []
        for row in rows:
            value = row.get(field)
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        return float(np.mean(values)) if values else None

    key_metrics = {}
    for source in SOURCE_NAMES:
        for mode in MODE_NAMES:
            for level in LEVEL_NAMES:
                for suffix in ("effective", "max"):
                    field = f"{source}_{mode}_{level}_{suffix}"
                    key_metrics[field] = mean_of(field)
    for mode in MODE_NAMES:
        for level in LEVEL_NAMES:
            for suffix in ("corr", "l1", "max_abs"):
                field = f"compare_{mode}_{level}_{suffix}"
                key_metrics[field] = mean_of(field)

    total_samples = int(sum(item.get("processed_samples", 0) for item in summaries))
    total_stations = int(sum(item.get("processed_stations", 0) for item in summaries))
    total_failures = int(sum(item.get("failed_samples", 0) for item in summaries))
    summary = {
        "config": args.config,
        "split": args.split,
        "world_size": world_size,
        "processed_samples": total_samples,
        "processed_stations": total_stations,
        "failed_samples": total_failures,
        "token_floor": args.token_floor,
        "dpk_weight_temperature": args.dpk_weight_temperature,
        "dpk_weight_resample": args.dpk_weight_resample,
        "rank_summaries": summaries,
        "model_compare": model_compare,
        "key_metric_means": key_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "model_compare.json", "w") as f:
        json.dump(model_compare, f, indent=2)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "prior_shards": [item["npz_path"] for item in summaries],
                "hdf5_cache": str(hdf5_path) if hdf5_path is not None else None,
                "stats_csv": [item["csv_path"] for item in summaries],
                "cache_index_csv": str(out_dir / "cache_index.csv"),
                "cache_index_npz": str(out_dir / "cache_index.npz"),
                "summary": str(out_dir / "summary.json"),
                "model_compare": str(out_dir / "model_compare.json"),
            },
            f,
            indent=2,
        )
    write_cache_index(out_dir, rows, hdf5_path=hdf5_path)


def hdf5_compression_kwargs(args):
    if args.hdf5_compression == "none":
        return {}
    if args.hdf5_compression == "gzip":
        return {"compression": "gzip", "compression_opts": args.hdf5_gzip_level}
    if args.hdf5_compression == "lzf":
        return {"compression": "lzf"}
    raise ValueError(f"Unsupported hdf5 compression: {args.hdf5_compression}")


def hdf5_string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def row_int(row, field, default=-1):
    value = row.get(field, "")
    if value in (None, ""):
        return default
    return int(float(value))


def row_float(row, field, default=np.nan):
    value = row.get(field, "")
    if value in (None, ""):
        return default
    return float(value)


def write_hdf5_cache(out_dir, summaries, rows, args, model_compare):
    hdf5_path = out_dir / args.hdf5_name
    prior_keys = [f"{source}_{mode}_{level}" for source in SOURCE_NAMES for mode in MODE_NAMES for level in LEVEL_NAMES]
    summaries = sorted(summaries, key=lambda item: item.get("rank", 0))
    total_rows = sum(int(item.get("array_shapes", {}).get("meta_event_id", [0])[0]) for item in summaries)
    string_dtype = hdf5_string_dtype()
    compression_kwargs = hdf5_compression_kwargs(args)
    prior_dtype = np.float32 if args.hdf5_prior_dtype == "float32" else np.float16

    with h5py.File(hdf5_path, "w") as h5:
        h5.attrs["format"] = "team_dpk_prior_cache_v1"
        h5.attrs["created_unix_time"] = float(time.time())
        h5.attrs["config"] = args.config
        h5.attrs["split"] = args.split
        h5.attrs["world_size"] = int(len(summaries))
        h5.attrs["token_floor"] = float(args.token_floor)
        h5.attrs["dpk_weight_temperature"] = float(args.dpk_weight_temperature)
        h5.attrs["dpk_weight_resample"] = args.dpk_weight_resample
        h5.attrs["model_compare_json"] = json.dumps(model_compare)
        h5.attrs["run_args_json"] = json.dumps(vars(args), default=str)

        priors_group = h5.create_group("priors")
        datasets = {}
        first_npz = None
        for item in summaries:
            npz_path = item.get("npz_path")
            if npz_path and Path(npz_path).exists():
                first_npz = np.load(npz_path)
                break
        if first_npz is None:
            raise ValueError("No prior shards found for HDF5 merge")
        try:
            for source in SOURCE_NAMES:
                source_group = priors_group.require_group(source)
                for mode in MODE_NAMES:
                    mode_group = source_group.require_group(mode)
                    for level in LEVEL_NAMES:
                        key = f"{source}_{mode}_{level}"
                        if key not in first_npz:
                            continue
                        _, token_len = first_npz[key].shape
                        chunks = (min(max(total_rows, 1), args.hdf5_chunk_rows), token_len)
                        datasets[key] = mode_group.create_dataset(
                            level,
                            shape=(total_rows, token_len),
                            dtype=prior_dtype,
                            chunks=chunks,
                            **compression_kwargs,
                        )
        finally:
            first_npz.close()

        meta_group = h5.create_group("meta")
        index_group = h5.create_group("index")
        stats_group = h5.create_group("stats")

        meta_specs = {
            "split": ("U16", string_dtype),
            "event_id": ("U64", string_dtype),
            "dataset_id": (np.int64, np.int64),
            "sample_index": (np.int64, np.int64),
            "station_slot": (np.int64, np.int64),
            "original_station_index": (np.int64, np.int64),
            "realtime_current_sample": (np.float64, np.float64),
            "realtime_elapsed_time": (np.float64, np.float64),
            "realtime_time_bin": (np.float64, np.float64),
            "rank": (np.int64, np.int64),
            "row_in_shard": (np.int64, np.int64),
        }
        meta_datasets = {
            name: meta_group.create_dataset(name, shape=(total_rows,), dtype=h5_dtype)
            for name, (_, h5_dtype) in meta_specs.items()
        }
        index_datasets = {
            "sample_station_key": index_group.create_dataset(
                "sample_station_key", shape=(total_rows,), dtype=string_dtype
            ),
            "event_station_time_key": index_group.create_dataset(
                "event_station_time_key", shape=(total_rows,), dtype=string_dtype
            ),
            "h5_row": index_group.create_dataset("h5_row", data=np.arange(total_rows, dtype=np.int64)),
            "shard_path": index_group.create_dataset("shard_path", shape=(total_rows,), dtype=string_dtype),
        }

        offset = 0
        rows_by_rank = {}
        for row in rows:
            rows_by_rank.setdefault(row_int(row, "rank"), []).append(row)

        stat_fields = sorted(
            field
            for field in ({key for row in rows for key in row.keys()})
            if (
                field.endswith("_effective")
                or field.endswith("_mean")
                or field.endswith("_max")
                or field.startswith("compare_")
            )
        )
        stat_datasets = {
            field: stats_group.create_dataset(field, shape=(total_rows,), dtype=np.float32)
            for field in stat_fields
        }

        for item in summaries:
            rank = int(item.get("rank", 0))
            npz_path = item.get("npz_path")
            if not npz_path:
                continue
            with np.load(npz_path) as shard:
                n_rows = int(shard["meta_event_id"].shape[0]) if "meta_event_id" in shard else 0
                if n_rows == 0:
                    continue
                end = offset + n_rows
                for key, dataset in datasets.items():
                    dataset[offset:end] = shard[key].astype(prior_dtype, copy=False)

                rank_rows = rows_by_rank.get(rank, [])
                if len(rank_rows) != n_rows:
                    raise ValueError(
                        f"Rank {rank} row count mismatch: csv rows={len(rank_rows)}, npz rows={n_rows}"
                    )
                for local_row, row in enumerate(rank_rows):
                    h5_row = offset + local_row
                    row["h5_row"] = h5_row
                    row["hdf5_path"] = str(hdf5_path)
                    for name in meta_datasets:
                        if name in ("split", "event_id"):
                            meta_datasets[name][h5_row] = row.get(name, "")
                        elif name in ("realtime_current_sample", "realtime_elapsed_time", "realtime_time_bin"):
                            meta_datasets[name][h5_row] = row_float(row, name)
                        else:
                            meta_datasets[name][h5_row] = row_int(row, name)
                    index_datasets["sample_station_key"][h5_row] = row.get("sample_station_key", "")
                    index_datasets["event_station_time_key"][h5_row] = row.get("event_station_time_key", "")
                    index_datasets["shard_path"][h5_row] = row.get("shard_path", "")
                    for field, dataset in stat_datasets.items():
                        dataset[h5_row] = row_float(row, field)
                offset = end

    return hdf5_path


def write_cache_index(out_dir, rows, hdf5_path=None):
    index_fields = [
        "sample_station_key",
        "event_station_time_key",
        "hdf5_path",
        "h5_row",
        "shard_path",
        "rank",
        "row_in_shard",
        "split",
        "dataset_id",
        "sample_index",
        "event_id",
        "station_slot",
        "original_station_index",
        "realtime_current_sample",
        "realtime_elapsed_time",
        "realtime_time_bin",
    ]
    csv_path = out_dir / "cache_index.csv"
    npz_path = out_dir / "cache_index.npz"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields)
        writer.writeheader()
        for row in rows:
            if hdf5_path is not None:
                row.setdefault("hdf5_path", str(hdf5_path))
            writer.writerow({field: row.get(field, "") for field in index_fields})

    def values(field, dtype=None):
        arr = np.asarray([row.get(field, "") for row in rows])
        return arr.astype(dtype) if dtype is not None else arr

    int_fields = ("rank", "row_in_shard", "dataset_id", "sample_index", "station_slot", "original_station_index")
    float_fields = ("realtime_current_sample", "realtime_elapsed_time", "realtime_time_bin")
    payload = {
        "sample_station_key": values("sample_station_key", "U128"),
        "event_station_time_key": values("event_station_time_key", "U160"),
        "hdf5_path": values("hdf5_path", "U512"),
        "shard_path": values("shard_path", "U512"),
        "split": values("split", "U16"),
        "event_id": values("event_id", "U64"),
    }
    for field in ("h5_row",) + int_fields:
        raw = [row.get(field, -1) for row in rows]
        payload[field] = np.asarray([int(float(x)) if str(x) != "" else -1 for x in raw], dtype=np.int64)
    for field in float_fields:
        raw = [row.get(field, np.nan) for row in rows]
        payload[field] = np.asarray([float(x) if str(x) != "" else np.nan for x in raw], dtype=np.float64)
    np.savez_compressed(npz_path, **payload)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PGA experiment JSON config")
    parser.add_argument("--diting_config", required=True, help="DiTing YAML config")
    parser.add_argument("--diting_pretrained", default=None, help="Override MAE/pretrained DiTing checkpoint")
    parser.add_argument("--dpk_checkpoint", default=DEFAULT_DPK_CKPT, help="DPK fine-tuned checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory for prior shards and diagnostics")
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="train")
    parser.add_argument("--station_batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional per-rank smoke-test cap")
    parser.add_argument("--test_run", action="store_true", help="Use train_light-style load_events limit=300")
    parser.add_argument("--token_floor", type=float, default=1e-4)
    parser.add_argument("--dpk_weight_temperature", type=float, default=1.0)
    parser.add_argument("--dpk_weight_resample", choices=("max", "avg"), default="max")
    parser.add_argument("--save_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--hdf5_name", default="dpk_priors.h5", help="Merged HDF5 cache filename")
    parser.add_argument("--hdf5_prior_dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--hdf5_compression", choices=("none", "gzip", "lzf"), default="none")
    parser.add_argument("--hdf5_gzip_level", type=int, default=4)
    parser.add_argument("--hdf5_chunk_rows", type=int, default=1024)
    parser.add_argument("--skip_hdf5_merge", action="store_true", help="Only write rank NPZ shards and CSV diagnostics")
    parser.add_argument(
        "--no_share_if_identical",
        dest="share_if_identical",
        action="store_false",
        help="Keep a separate DPK encoder even when encoder weights compare exactly equal",
    )
    parser.set_defaults(share_if_identical=True)
    parser.add_argument("--device", default="cuda", help="Fallback device when not distributed")
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    is_dist, rank, world_size, local_rank = util.setup_distributed()
    if is_dist:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        if args.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(args.device if ":" in args.device else "cuda:0")
        else:
            device = torch.device("cpu")

    config = load_config_file(args.config)
    limit = 300 if args.test_run else None
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "run_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    if is_dist:
        dist.barrier()

    log_rank(rank, f"device={device} world_size={world_size}")
    generators = load_split_generators(config, args.split, limit=limit, rank=rank)
    current_encoder, dpk_model, dpk_head, model_compare = build_models(config, args, device, is_dist)
    if rank == 0:
        log_rank(rank, f"model_compare={model_compare}")

    array_store, rows, failures, processed_samples, processed_stations = process_dataset(
        args,
        generators=generators,
        current_encoder=current_encoder,
        dpk_model=dpk_model,
        dpk_head=dpk_head,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    write_rank_outputs(out_dir, rank, array_store, rows, failures, processed_samples, processed_stations)
    log_rank(rank, f"wrote shard: samples={processed_samples} stations={processed_stations} failures={len(failures)}")

    if is_dist:
        dist.barrier()
    if rank == 0:
        aggregate_rank_outputs(out_dir, world_size, model_compare, args)
        log_rank(rank, f"aggregate summary written to {out_dir / 'summary.json'}")
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
