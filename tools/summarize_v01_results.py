#!/usr/bin/env python3
"""Read-only V01 artifact audit; write a small, partial-results review package.

This is an offline reporting tool, not a replacement for the HPC evaluation
launcher. No data, checkpoint, evaluation configuration or runtime is modified.
Only load NPZ/checkpoint objects from trusted experiment artifacts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {"vfull": "weights_vfull_retry1", "vmissing": "weights_vmissing", "apair": "weights_apair"}
CELLS = (("vfull", "vfull"), ("vfull", "vmissing"),
         ("vmissing", "vfull"), ("vmissing", "vmissing"), ("apair", "apair"))
# A target index is a reordered sampler slot, not a physical station ID. Match
# exported query coordinates within the event and time; verify labels/cutoff.
PAIR_KEYS = ["event_id", "time_s", "query_lat", "query_lon", "query_depth"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stack(data, name, dtype=float):
    value = data["val_" + name]
    return np.asarray(value.tolist() if value.dtype == object else value, dtype=dtype)


def load_cell(path):
    with np.load(path, allow_pickle=True) as data:
        if any(key.startswith("test_") for key in data.files):
            raise ValueError("Reporting input must contain validation only")
        arrays = {key[4:]: stack(data, key[4:], str if key == "val_event_id" else float)
                  for key in data.files if key.startswith("val_")}
    valid = arrays["pga_target_valid"].astype(bool)
    truth = arrays["pga_label"].squeeze(-1)
    pred = arrays["pga_mu_best"]  # evaluator exports mixture mean under this legacy name
    if np.any(valid & (~np.isfinite(truth) | ~np.isfinite(pred))):
        raise ValueError("Non-finite prediction/label in a valid target")
    row, slot = np.where(valid)
    coords = arrays["pga_target_abs"][row, slot]
    frame = pd.DataFrame({
        "event_id": arrays["event_id"][row],
        "time_s": arrays["realtime_requested_elapsed_time"][row],
        "current_sample": arrays["realtime_current_sample"][row],
        "first_pick_sample": arrays["realtime_first_p_pick_sample"][row],
        "query_lat": coords[:, 0], "query_lon": coords[:, 1], "query_depth": coords[:, 2],
        "target_index": arrays["pga_target_indices"][row, slot].astype(int),
        "truth": truth[row, slot], "prediction": pred[row, slot],
        "target_type": arrays["realtime_target_type"][row, slot].astype(int),
        "input_count": arrays["station_valid"].astype(bool).sum(axis=1)[row],
        "sigma": arrays["pga_sigma"][row, slot],
        "nll": arrays["pga_nll_log10_mps2"][row, slot],
        "prob": arrays["pga_prob_ge_threshold"][row, slot],
    })
    frame["abs_error"] = abs(frame.prediction - frame.truth)
    return frame, arrays


def point_metrics(frame, threshold=-1.2):
    if frame.empty:
        return {"targets": 0, "events": 0}
    y = frame.truth.to_numpy(); p = frame.prediction.to_numpy(); e = p - y
    variance = np.sum((y - y.mean()) ** 2)
    return {
        "targets": len(frame), "events": frame.event_id.nunique(),
        "mae": float(abs(e).mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(e.mean()), "r2": float(1 - np.sum(e ** 2) / variance) if variance else None,
        "slope": float(np.sum((y-y.mean())*(p-p.mean())) / variance) if variance else None,
        "p90_abs_error": float(np.quantile(abs(e), .9)),
        "p95_abs_error": float(np.quantile(abs(e), .95)),
        "within_0p1": float(np.mean(abs(e) <= .1)), "within_0p2": float(np.mean(abs(e) <= .2)),
        "nll": float(frame.nll.mean()),
        "brier": float(np.mean((frame.prob.to_numpy() - (y >= threshold)) ** 2)),
        "positive_rate": float(np.mean(y >= threshold)), "probability_mean": float(frame.prob.mean()),
        "predictive_sigma_mean": float(frame.sigma.mean()),
        "coverage_1sigma": float(np.mean(abs(e) <= frame.sigma)),
        "coverage_2sigma": float(np.mean(abs(e) <= 2 * frame.sigma)),
        "event_macro_mae": float(frame.groupby("event_id").abs_error.mean().mean()),
    }


def pair_rows(left, right):
    for name, frame in (("left", left), ("right", right)):
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{name} has duplicate physical query/time keys; cannot pair")
    paired = left.merge(right, on=PAIR_KEYS, suffixes=("_left", "_right"), validate="one_to_one")
    for column in ("truth", "current_sample", "first_pick_sample", "target_type"):
        if not np.array_equal(paired[column + "_left"], paired[column + "_right"]):
            raise ValueError(f"Paired targets disagree on {column}")
    return paired


def cluster_ci(paired, draws, seed):
    if paired.empty or draws < 1:
        raise ValueError("Need nonempty pairs and positive draws")
    delta = paired.abs_error_right - paired.abs_error_left
    sufficient = pd.DataFrame({"event_id": paired.event_id, "delta": delta}).groupby("event_id").delta.agg(["sum", "count", "mean"])
    rng = np.random.default_rng(seed)
    totals = sufficient["sum"].to_numpy(); counts = sufficient["count"].to_numpy()
    means = sufficient["mean"].to_numpy(); n = len(sufficient)
    weighted = np.empty(draws); macro = np.empty(draws)
    for start in range(0, draws, 256):
        sample = rng.integers(0, n, size=(min(256, draws-start), n))
        weighted[start:start+len(sample)] = totals[sample].sum(axis=1) / counts[sample].sum(axis=1)
        macro[start:start+len(sample)] = means[sample].mean(axis=1)
    return {
        "paired_targets": len(paired), "paired_events": n,
        "delta_mae": float(delta.mean()), "ci_low": float(np.quantile(weighted, .025)),
        "ci_high": float(np.quantile(weighted, .975)),
        "event_macro_delta_mae": float(means.mean()),
        "event_macro_ci_low": float(np.quantile(macro, .025)),
        "event_macro_ci_high": float(np.quantile(macro, .975)),
        "delta_nll": float((paired.nll_right-paired.nll_left).mean()),
    }, sufficient.reset_index()


def field_metrics(frame):
    rows = []
    # Within each event/time, give each eligible sample equal weight. This is
    # descriptive field contrast, not cross-arm target-independent inference.
    for (event, time), group in frame.groupby(["event_id", "time_s"]):
        if len(group) < 5:
            continue
        y = group.truth.to_numpy(); p = group.prediction.to_numpy()
        truth_range = np.ptp(y)
        i, j = np.triu_indices(len(group), 1)
        rows.append({"event_id": event, "time_s": time,
                     "input_count": int(group.input_count.iloc[0]),
                     "targets": len(group), "truth_range": float(truth_range),
                     "prediction_range": float(np.ptp(p)),
                     "range_ratio": float(np.ptp(p)/truth_range) if truth_range > 1e-8 else np.nan,
                     "pairwise_difference_mae": float(np.mean(abs((p[i]-p[j])-(y[i]-y[j]))))})
    fields = pd.DataFrame(rows)
    result = []
    for name, subset in (("all_inputs", fields), ("single_input", fields[fields.input_count == 1])):
        for time in ("all7", "early135"):
            selected = subset if time == "all7" else subset[subset.time_s.isin([1,3,5])]
            result.append({"group": name, "time_window": time, "eligible_samples": len(selected),
                           "events": selected.event_id.nunique(),
                           "range_ratio_median": float(selected.range_ratio.median()) if len(selected) else None,
                           "pairwise_difference_mae": float(selected.pairwise_difference_mae.mean()) if len(selected) else None})
    return result


def audit_arrays(arrays):
    events = arrays["event_id"]; times = arrays["realtime_requested_elapsed_time"]
    station = arrays["station_valid"].astype(bool); role = arrays["v01_source_role"].astype(bool)
    valid = arrays["pga_target_valid"].astype(bool)
    samples = pd.DataFrame({"event_id": events, "time_s": times})
    duplicate = samples[samples.duplicated(["event_id", "time_s"], keep=False)]
    pre = arrays["waveform_pre_p_valid_seconds"][station]
    return {
        "samples": len(events), "unique_events_in_samples": len(set(events)),
        "unique_event_time_pairs": len(samples.drop_duplicates()),
        "extra_duplicate_samples": len(samples)-len(samples.drop_duplicates()),
        "duplicate_event_time_examples": duplicate.head(12).to_dict("records"),
        "valid_targets": int(valid.sum()),
        "valid_target_type_counts": {str(int(k)): int(v) for k,v in zip(*np.unique(arrays["realtime_target_type"][valid], return_counts=True))},
        "query_only_in_valid_input_slots": int((station & ~role).sum()),
        "selected_source_observations": int(station.sum()),
        "pre_p_valid_seconds_mean": float(pre.mean()), "pre_p_valid_seconds_median": float(np.median(pre)),
        "pre_p_valid_seconds_p90": float(np.quantile(pre,.9)),
        "zero_pre_p_source_observations": int(np.sum(pre == 0)),
        "post_p_valid_seconds_mean": float(arrays["waveform_post_p_valid_seconds"][station].mean()),
        "physical_sensor_id_exported": False,
        "pair_key_policy": "event + requested time + exact exported query coordinates; labels and cutoff checked",
    }


def tensor_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little")); digest.update(header)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_audit(root):
    import torch  # not needed for NPZ/statistical analysis
    rows = []
    for arm, directory in ARMS.items():
        for name in ("init", "last", "best"):
            path = root / directory / ("full_model_" + name + ".pth")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint["model_state_dict"]
            row = {"arm": arm, "kind": name, "file": str(path.relative_to(root)),
                   "bytes": path.stat().st_size, "file_sha256": file_sha(path),
                   "saved_state_tensor_sha256": tensor_sha(state),
                   "epoch": int(checkpoint["epoch"]), "loss": checkpoint.get("loss"),
                   "format": checkpoint.get("checkpoint_format"),
                   "saved_tensor_count": len(state), "excluded_tensor_count": checkpoint.get("excluded_tensor_count"),
                   "encoder_source": checkpoint.get("encoder_source"),
                   "optimizer_present": "optimizer_state_dict" in checkpoint,
                   "scheduler_present": "scheduler_state_dict" in checkpoint}
            if row["optimizer_present"]:
                optimizer = checkpoint["optimizer_state_dict"]
                steps = [float(value["step"]) for value in optimizer["state"].values() if "step" in value]
                row["optimizer_step_min"] = min(steps) if steps else None
                row["optimizer_step_max"] = max(steps) if steps else None
                row["optimizer_group_lrs"] = [float(group["lr"]) for group in optimizer["param_groups"]]
            rows.append(row)
            print(f"Audited {arm}/{name} epoch {row['epoch']}", flush=True)
            del checkpoint, state
            gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--audit-checkpoints", action="store_true")
    args = parser.parse_args()
    root = args.input_root; output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review package: {output}")
    output.mkdir(parents=True)
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            inventory.append({"file": str(path.relative_to(root)), "bytes": path.stat().st_size,
                              "sha256": file_sha(path)})
    pd.DataFrame(inventory).to_csv(output / "input_artifact_inventory.csv", index=False)
    configs = {}; curves = []
    light = output / "source_evidence"; light.mkdir()
    for arm, directory in ARMS.items():
        source = root / directory; config = json.loads((source/"config.json").read_text())
        configs[arm] = config; destination = light / directory; destination.mkdir()
        shutil.copy2(source/"config.json", destination/"config.json")
        for name in ("train_epoch_loss", "val_epoch_loss", "train_lr", "train_lr_adapter", "train_lr_team"):
            path = source/(name+".csv")
            shutil.copy2(path, destination/path.name)
            frame = pd.read_csv(path)
            for record in frame.to_dict("records"):
                curves.append({"arm": arm, "scalar": name, **record})
    pd.DataFrame(curves).to_csv(output/"training_scalars.csv", index=False)
    training_contract = {
        "model_params_identical_across_arms": all(c["model_params"] == configs["vfull"]["model_params"] for c in configs.values()),
        "seed_values": {a:c["seed"] for a,c in configs.items()},
        "epochs": {a:c["training_params"]["epochs_full_model"] for a,c in configs.items()},
        "learning_rates": {a:{k:c["training_params"][k] for k in ("lr","lr_adapter","lr_team")} for a,c in configs.items()},
        "parent_paths": {a:c["training_params"]["load_model_path"] for a,c in configs.items()},
        "frozen_split_paths": {a:c["training_params"]["frozen_split_manifest"] for a,c in configs.items()},
        "data_shard_counts": {a:len(c["training_params"]["data_path"]) for a,c in configs.items()},
        "inherited_full_data_manifest_is_not_actual_v01_counts": True,
        "preflight_cohort_counts_available": bool(list(root.rglob("preflight_summary.json"))),
        "frozen_split_file_available": bool(list(root.rglob("split_events.csv"))),
        "hpc_source_manifest_available": False,
    }
    if args.audit_checkpoints:
        rows = checkpoint_audit(root); write_json(output/"checkpoint_audit.json", rows)
        initial = [r for r in rows if r["kind"] == "init"]
        training_contract["initial_saved_tensors_identical"] = len({r["saved_state_tensor_sha256"] for r in initial}) == 1
        training_contract["all_last_checkpoint_epoch8"] = all(r["epoch"] == 8 for r in rows if r["kind"] == "last")
    write_json(output/"training_contract.json", training_contract)

    frames = {}; array_sets = {}; audits = {}; status = []; metric_rows = []; field_rows = []
    for checkpoint, view in CELLS:
        for protocol in ("normal", "random"):
            cell = "__".join((checkpoint, view, protocol)); base = root/"eval_retry1"/cell
            config_path = base.with_suffix(".config.json"); npz = base.with_suffix(".npz")
            metrics_path = base.with_suffix(".metrics.json"); log = base.with_suffix(".txt")
            shutil.copy2(config_path, light/config_path.name)
            text = log.read_text(errors="replace") if log.exists() else ""
            errors = re.findall(r"^(?:ValueError|RuntimeError|KeyError|IndexError):.*$", text, re.M)
            item = {"cell": cell, "protocol": protocol, "npz_present": npz.exists(),
                    "metrics_present": metrics_path.exists(), "log_present": log.exists(),
                    "terminal_error": errors[-1] if errors else None,
                    "status": "failed" if errors else ("exported" if npz.exists() and metrics_path.exists() else "incomplete")}
            status.append(item)
            if errors:
                excerpt = text[text.rfind("Traceback (most recent call last):"):]
                (light/(cell+".failure.txt")).write_text(excerpt)
                continue
            if not npz.exists() or not metrics_path.exists():
                continue
            formal = json.loads(metrics_path.read_text())
            if formal["splits"] != ["val"] or formal["checkpoint_metadata"]["epoch"] != 8:
                raise ValueError(f"{cell} is not fixed-epoch-8 validation")
            if formal["waveform_station_permutation"] != "none":
                raise ValueError("Not an unperturbed V01 validation")
            shutil.copy2(metrics_path, light/metrics_path.name)
            frame, arrays = load_cell(npz); frames[cell] = frame; array_sets[cell] = arrays
            audits[cell] = audit_arrays(arrays)
            recomputed = point_metrics(frame)
            official = formal["metrics"]["val"]["pga"]
            for key in ("targets", "events", "mae", "rmse", "bias", "r2", "slope", "nll", "brier", "coverage_1sigma", "coverage_2sigma"):
                if not np.isclose(recomputed[key], official[key], atol=1e-7, rtol=1e-7):
                    raise ValueError(f"{cell}: recomputed {key} disagrees with metrics.json")
            time_masks = [("all7", np.ones(len(frame), bool)), ("early135", frame.time_s.isin([1,3,5]))]
            time_masks += [(str(time), frame.time_s == time) for time in (1,3,5,10,20,40,90)]
            groups = [("all_targets", np.ones(len(frame), bool)),
                      ("non_input", frame.target_type != 0), ("input", frame.target_type == 0),
                      ("triggered_noninput", frame.target_type == 1), ("untriggered", frame.target_type == 2),
                      ("single_input", frame.input_count == 1),
                      ("strong_ge_minus1p2", frame.truth >= -1.2), ("weak_lt_minus1p2", frame.truth < -1.2)]
            for time_name, time_mask in time_masks:
                for group_name, mask in groups:
                    metric_rows.append({"cell": cell, "protocol": protocol, "time_window": time_name,
                                        "group": group_name, **point_metrics(frame[time_mask & mask])})
            field_rows += [{"cell": cell, **row} for row in field_metrics(frame)]
            print(f"Summarized {cell}: {len(frame)} valid targets", flush=True)

    write_json(output/"evaluation_status.json", status)
    write_json(output/"npz_contract_audit.json", audits)
    pd.DataFrame(metric_rows).to_csv(output/"metrics_by_time_and_group.csv", index=False)
    pd.DataFrame(field_rows).to_csv(output/"field_contrast_metrics.csv", index=False)
    paired_rows = []; event_stats = []; pairing_audit = []
    ff = "vfull__vfull__random"; fm = "vfull__vmissing__random"
    mf = "vmissing__vfull__random"; mm = "vmissing__vmissing__random"
    for name, left, right in (("R_MM_minus_R_FF",ff,mm), ("R_FM_minus_R_FF",ff,fm),
                              ("R_MF_minus_R_FF",ff,mf), ("R_MM_minus_R_MF",mf,mm),
                              ("R_MM_minus_R_FM",fm,mm)):
        if left not in frames or right not in frames:
            continue
        paired = pair_rows(frames[left], frames[right])
        pairing_audit.append({"comparison": name, "left_targets": len(frames[left]),
                              "right_targets": len(frames[right]), "matched_targets": len(paired),
                              "same_target_index": bool(np.array_equal(paired.target_index_left, paired.target_index_right)),
                              "truth_and_time_equal": True})
        for label, times in (("1",[1]), ("3",[3]), ("5",[5]), ("early135",[1,3,5]),
                             ("all7",[1,3,5,10,20,40,90])):
            selected = paired[paired.time_s.isin(times)]
            ci, sufficient = cluster_ci(selected, args.bootstrap_draws, args.bootstrap_seed)
            paired_rows.append({"comparison": name, "protocol": "random", "time_window": label,
                                "left": left, "right": right, "bootstrap_draws": args.bootstrap_draws,
                                "bootstrap_seed": args.bootstrap_seed, **ci})
            sufficient.insert(0, "time_window", label); sufficient.insert(0, "comparison", name)
            event_stats.append(sufficient)
    write_json(output/"pairing_audit.json", pairing_audit)
    pd.DataFrame(paired_rows).to_csv(output/"paired_event_cluster_ci.csv", index=False)
    if event_stats:
        pd.concat(event_stats, ignore_index=True).to_csv(output/"paired_event_sufficient_statistics.csv", index=False)
    dose = {}
    if ff in array_sets and fm in array_sets:
        a = array_sets[ff]; b = array_sets[fm]
        keys = ("event_id", "realtime_requested_elapsed_time", "realtime_current_sample", "realtime_first_p_pick_sample",
                "station_valid", "station_coords_abs", "selected_original_input_indices", "p_picks",
                "v01_source_role", "v01_template_id", "v01_retained_prep_samples", "waveform_post_p_valid_sample_count")
        dose["same_exported_inputs_cutoff_and_post_p_support"] = {k:bool(np.array_equal(a[k],b[k],equal_nan=True)) if a[k].dtype.kind != "U" else bool(np.array_equal(a[k],b[k])) for k in keys}
        selected = a["station_valid"].astype(bool)
        removed = (a["waveform_pre_p_valid_seconds"] - b["waveform_pre_p_valid_seconds"])[selected]
        dose.update({"selected_source_observations": int(selected.sum()), "removed_mean_seconds": float(removed.mean()),
                     "removed_median_seconds": float(np.median(removed)), "removed_p90_seconds": float(np.quantile(removed,.9)),
                     "changed_observations": int(np.sum(removed > 1e-6)), "negative_removed_observations": int(np.sum(removed < -1e-6)),
                     "raw_post_p_values_compared": False,
                     "note": "NPZ verifies post-P support counts, not raw waveform sample equality; sensor IDs and source hashes are not exported."})
    write_json(output/"intervention_support_audit.json", dose)
    summary = {
        "status": "partial_validation_results_normal_failed",
        "input_root": str(root), "date": "2026-10-03", "split": "val", "checkpoint_epoch": 8,
        "coordinate": "log10(m/s^2)", "point_estimate": "predictive_mixture_mean",
        "successful_exported_cells": len(frames), "failed_cells": sum(x["status"] == "failed" for x in status),
        "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": args.bootstrap_seed,
        "pair_key": PAIR_KEYS, "positive_delta": "right condition has larger MAE",
        "limitations": ["All normal cells failed; no normal metrics are available.",
                        "A-pair has duplicate event/time samples; aggregate scores are descriptive, not strict paired evidence.",
                        "Exported coordinates stand in for query identity; sensor IDs and frozen/cohort manifests are missing.",
                        "Single training seed; bootstrap does not quantify training-seed variation.",
                        "Sensitivity-only velocity, not full instrument deconvolution; cross-domain comparison is confounded.",
                        "Inherited full_data_manifest is historical and cannot prove actual V01 training counts.",
                        "Only exported post-P support checked; raw post-P waveforms not supplied.",
                        "No held-out test, Slurm sacct or uploaded runtime source manifest supplied."],
    }
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
