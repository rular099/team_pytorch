import argparse
import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import torch

import loader_light as loader
import gemini_util_light as current_util


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_events(event_metadata, n):
    if n is None or n <= 0:
        return event_metadata
    for event_key in ["KiK_File", "#EventID", "EVENT"]:
        if event_key in event_metadata.columns:
            keep = event_metadata[event_key].unique()[:n]
            return event_metadata[event_metadata[event_key].isin(keep)].copy()
    return event_metadata.iloc[:n].copy()


def load_module_from_path(module_path, module_name, extra_sys_path=None):
    old_path = list(sys.path)
    try:
        if extra_sys_path is not None:
            sys.path.insert(0, extra_sys_path)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def load_module_from_git_revision(repo_root, revision, rel_path, module_name):
    source = subprocess.check_output(
        ["git", "-C", repo_root, "show", f"{revision}:{rel_path}"],
        text=True,
    )
    tmpdir = tempfile.mkdtemp(prefix=f"{module_name}_")
    module_path = os.path.join(tmpdir, os.path.basename(rel_path))
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(source)
    return load_module_from_path(module_path, module_name, extra_sys_path=repo_root)


def resolve_data_path(repo_root, data_path):
    if os.path.isabs(data_path):
        return data_path
    return os.path.join(repo_root, data_path)


def resolve_config_path(repo_root, config_path):
    if os.path.isabs(config_path):
        return config_path
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
    return os.path.join(repo_root, config_path)


def load_split(config, repo_root, split_name, overfit_n):
    training_params = copy.deepcopy(config["training_params"])
    generator_params = copy.deepcopy(training_params.get("generator_params", [training_params.copy()]))
    if not isinstance(training_params["data_path"], list):
        training_params["data_path"] = [training_params["data_path"]]

    part_map = {
        "train": (True, False, False),
        "dev": (False, True, False),
        "test": (False, False, True),
    }
    cache_map = {
        "train": "train_ev.csv",
        "dev": "test_ev.csv",
        "test": "test_ev.csv",
    }
    parts = part_map[split_name]
    overwrite_sampling_rate = training_params.get("overwrite_sampling_rate")
    full_data = []
    for data_path, generator in zip(training_params["data_path"], generator_params):
        full_data.append(
            loader.load_events(
                resolve_data_path(repo_root, data_path),
                event_metadata_path=os.path.join(repo_root, cache_map[split_name]),
                parts=parts,
                shuffle_train_dev=generator.get("shuffle_train_dev", False),
                custom_split=generator.get("custom_split"),
                min_mag=generator.get("min_mag"),
                mag_key=generator.get("key", "MA"),
                overwrite_sampling_rate=overwrite_sampling_rate,
                decimate_events=generator.get("decimate_events"),
            )
        )

    event_metadata = [d[0] for d in full_data]
    metadata = [d[2] for d in full_data]
    if overfit_n > 0:
        event_metadata = [subset_events(meta, overfit_n) for meta in event_metadata]
        generator_params = [copy.deepcopy(g) for g in generator_params]
        for generator_param in generator_params:
            fixed_cutout = generator_param.get("cutout_end", generator_param.get("cutout_start", 0))
            generator_param["trigger_based"] = False
            generator_param["disable_station_foreshadowing"] = False
            generator_param["shuffle_train_dev"] = False
            generator_param["oversample"] = 1
            generator_param["select_first"] = True
            generator_param["cutout_start"] = fixed_cutout
            generator_param["cutout_end"] = fixed_cutout
    return training_params, generator_params, event_metadata, metadata


def build_generator(util_module, event_metadata, metadata, data_path, generator_param, model_params,
                    overfit_mode, force_no_shuffle):
    generator_param = copy.deepcopy(generator_param)
    noise_seconds = generator_param.get("noise_seconds", 5)
    sampling_rate = metadata["sampling_rate"]
    cutout = (
        sampling_rate * (noise_seconds + generator_param["cutout_start"]),
        sampling_rate * (noise_seconds + generator_param["cutout_end"]),
    )
    generator_param["transform_target_only"] = generator_param.get("transform_target_only", True)
    if force_no_shuffle:
        generator_param["shuffle"] = False
    return util_module.PreloadedEventGenerator(
        event_metadata=event_metadata,
        metadata=copy.deepcopy(metadata),
        data_path=data_path,
        generator_params=generator_param,
        coords_target=True,
        label_smoothing=False if overfit_mode else True,
        station_blinding=False,
        cutout=cutout,
        pga_targets=model_params.get("n_pga_targets", 0),
        max_stations=model_params["max_stations"],
        sampling_rate=sampling_rate,
        no_event_token=model_params.get("no_event_token", False),
        **generator_param,
    )


def requested_event_key(generator, sample_index):
    base_index = generator.indexes[sample_index]
    if isinstance(base_index, tuple):
        base_index = base_index[0]
    return str(generator.event_keys[base_index])


def round_list(values, digits=4):
    return [round(float(v), digits) for v in values]


def summarize_sample(generator, sample_index):
    summary = {
        "requested_index": int(sample_index),
        "requested_event": requested_event_key(generator, sample_index),
    }
    module = sys.modules.get(generator.__class__.__module__)
    empty_sample_exc = getattr(module, "_EmptySample", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            inputs, outputs, p_picks = generator._get_one(sample_index)
        except Exception as exc:
            if empty_sample_exc is not None and isinstance(exc, empty_sample_exc):
                summary["status"] = "empty_sample"
                summary["warnings"] = [str(w.message) for w in caught]
                return summary
            summary["status"] = "error"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["warnings"] = [str(w.message) for w in caught]
            return summary

    waveforms = inputs[0].cpu().numpy()
    metadata = inputs[1].cpu().numpy()
    station_valid = inputs[2].cpu().numpy().astype(bool)
    if isinstance(p_picks, dict):
        p_picks = np.asarray(p_picks["shifted"]).reshape(-1)
    else:
        p_picks = np.asarray(p_picks).reshape(-1)
    valid_idx = np.where(station_valid)[0]
    summary["status"] = "ok"
    summary["warnings"] = [str(w.message) for w in caught]
    summary["waveform_shape"] = list(waveforms.shape)
    summary["valid_station_count"] = int(station_valid.sum())
    summary["valid_station_slots"] = valid_idx.tolist()
    summary["all_zero_station_count"] = int((~np.any(waveforms != 0, axis=(1, 2))).sum())
    summary["station_pick_head"] = round_list(p_picks[valid_idx][:8]) if len(valid_idx) else []
    summary["station_coord_head"] = [
        round_list(row, digits=5) for row in metadata[valid_idx[:3]]
    ]
    summary["waveform_absmax_head"] = round_list(np.max(np.abs(waveforms[valid_idx[:3]]), axis=(1, 2))) if len(valid_idx) else []

    if len(inputs) >= 5:
        pga_targets = inputs[3].cpu().numpy()
        pga_target_valid = inputs[4].cpu().numpy().astype(bool)
        pga_values = outputs[-1].cpu().numpy().reshape(-1)
        target_idx = np.where(pga_target_valid)[0]
        summary["pga_target_valid_count"] = int(pga_target_valid.sum())
        summary["pga_values_head"] = round_list(pga_values[target_idx][:8]) if len(target_idx) else []
        summary["pga_target_coord_head"] = [
            round_list(row, digits=5) for row in pga_targets[target_idx[:3]]
        ]
    else:
        summary["pga_target_valid_count"] = 0
        summary["pga_values_head"] = []
        summary["pga_target_coord_head"] = []
    return summary


def diff_summaries(lhs, rhs):
    keys = sorted(set(lhs) | set(rhs))
    diffs = []
    for key in keys:
        if lhs.get(key) != rhs.get(key):
            diffs.append(key)
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--overfit-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--other-revision", default="0f9c2fb69b3ca8f5c42544e1e5c85d7e6a75c07c")
    parser.add_argument("--force-no-shuffle", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_config_path(repo_root, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = config.get("seed", 42) if args.seed is None else args.seed
    training_params, generator_params, event_metadata, metadata = load_split(
        config=config,
        repo_root=repo_root,
        split_name=args.split,
        overfit_n=args.overfit_n,
    )
    idx = args.dataset_index
    data_path = resolve_data_path(repo_root, training_params["data_path"][idx])
    generator_param = generator_params[idx]
    meta = event_metadata[idx]
    meta_info = metadata[idx]
    overfit_mode = args.overfit_n > 0

    old_util = load_module_from_git_revision(
        repo_root=repo_root,
        revision=args.other_revision,
        rel_path="gemini_util_light.py",
        module_name="old_gemini_util_light",
    )

    set_seed(seed)
    current_gen = build_generator(
        util_module=current_util,
        event_metadata=meta,
        metadata=meta_info,
        data_path=data_path,
        generator_param=generator_param,
        model_params=config["model_params"],
        overfit_mode=overfit_mode,
        force_no_shuffle=args.force_no_shuffle,
    )
    set_seed(seed)
    old_gen = build_generator(
        util_module=old_util,
        event_metadata=meta,
        metadata=meta_info,
        data_path=data_path,
        generator_param=generator_param,
        model_params=config["model_params"],
        overfit_mode=overfit_mode,
        force_no_shuffle=args.force_no_shuffle,
    )

    print(json.dumps({
        "config": os.path.abspath(config_path),
        "split": args.split,
        "dataset_index": idx,
        "seed": seed,
        "other_revision": args.other_revision,
        "force_no_shuffle": args.force_no_shuffle,
        "dataset_length_current": len(current_gen),
        "dataset_length_other": len(old_gen),
    }, indent=2))

    n = min(args.num_samples, len(current_gen), len(old_gen))
    for sample_index in range(n):
        set_seed(seed + sample_index)
        lhs = summarize_sample(current_gen, sample_index)
        set_seed(seed + sample_index)
        rhs = summarize_sample(old_gen, sample_index)
        print(json.dumps({
            "sample_index": sample_index,
            "different_fields": diff_summaries(lhs, rhs),
            "current": lhs,
            "other": rhs,
        }, indent=2))


if __name__ == "__main__":
    main()
