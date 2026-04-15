import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from japan_dataset_builder import load_converted_station, load_station_traces_from_event_archive


def normalize_sensor_suffix(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a converted Japan training sample against its raw archive source."
    )
    parser.add_argument("--waveform_root", required=True, help="Root directory containing yearly waveform subdirectories.")
    parser.add_argument("--hdf5", required=True, help="Converted HDF5 file to validate.")
    parser.add_argument("--stations_csv", required=True, help="Station metadata CSV emitted by the converter.")
    parser.add_argument("--event", help="Event id to validate. Defaults to the first CSV row.")
    parser.add_argument("--wave_idx", type=int, default=None, help="Station wave_idx within the event. Defaults to the first matching row.")
    parser.add_argument("--row_index", type=int, default=None, help="Direct row index into stations.csv. Overrides --event/--wave_idx.")
    parser.add_argument("--plot", help="Optional output path for a comparison plot.")
    parser.add_argument("--target_sampling_rate", type=float, default=100.0, help="Target sampling rate in Hz used during conversion.")
    return parser.parse_args()


def pick_row(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.row_index is not None:
        return df.iloc[args.row_index]

    if args.event is None:
        return df.iloc[0]

    subset = df[df["EVENT"].astype(str) == str(args.event)]
    if subset.empty:
        raise SystemExit(f"Event not found in stations CSV: {args.event}")
    if args.wave_idx is None:
        return subset.iloc[0]

    subset = subset[subset["wave_idx"] == args.wave_idx]
    if subset.empty:
        raise SystemExit(f"wave_idx={args.wave_idx} not found for EVENT={args.event}")
    return subset.iloc[0]


def build_plot(raw_resampled: np.ndarray, aligned: np.ndarray, start_sample: int, trigger_pick: int, refined_pick: int, pga_loc: int, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    labels = ["NS", "EW", "UD"]
    x_raw = np.arange(raw_resampled.shape[0])
    x_aligned = np.arange(aligned.shape[0])

    for i, ax in enumerate(axes):
        ax.plot(x_aligned, aligned[:, i], color="0.75", linewidth=0.9, label="aligned_full")
        ax.plot(start_sample + x_raw, raw_resampled[:, i], color="C0", linewidth=1.0, label="raw_resampled_inserted")
        ax.axvline(trigger_pick, color="C3", linestyle="--", linewidth=1.0, label="trigger_pick" if i == 0 else None)
        ax.axvline(refined_pick, color="C4", linestyle="-.", linewidth=1.0, label="refined_pick" if i == 0 else None)
        ax.axvline(pga_loc, color="C2", linestyle=":", linewidth=1.0, label="pga_loc" if i == 0 else None)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Aligned Sample Index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    waveform_root = Path(args.waveform_root).expanduser().resolve()
    hdf5_path = Path(args.hdf5).expanduser().resolve()
    stations_csv = Path(args.stations_csv).expanduser().resolve()

    df = pd.read_csv(stations_csv)
    row = pick_row(df, args)
    event_id = str(row["EVENT"])
    component_base = str(row["component_base"])
    inner_archive = str(row["inner_archive"])
    sensor_suffix = normalize_sensor_suffix(row["sensor_suffix"])

    archive_path = waveform_root / row["archive_relpath"]
    stations = load_station_traces_from_event_archive(
        outer_tar_path=archive_path,
        archive_relpath=str(row["archive_relpath"]),
        target_sampling_rate_hz=args.target_sampling_rate,
    )

    raw_station = None
    for station in stations:
        if (
            station.event_id == event_id
            and station.component_base == component_base
            and station.inner_archive_name == inner_archive
            and station.sensor_suffix == sensor_suffix
        ):
            raw_station = station
            break

    if raw_station is None:
        raise SystemExit("Failed to locate the raw station trace described by the selected CSV row.")

    converted = load_converted_station(hdf5_path=hdf5_path, event_id=event_id, wave_idx=int(row["wave_idx"]))
    aligned_waveform = converted["waveform"]
    start_sample = converted["record_start_sample"]
    valid_n_samples = converted["valid_n_samples"]
    p_pick = converted["p_pick"]
    p_pick_trigger = converted["p_pick_trigger_aligned"]
    p_pick_repaired = converted["p_pick_repaired_aligned"]
    p_pick_refined = converted["p_pick_refined_aligned"]
    pga_loc = converted["pga_norm_aligned_loc"]

    expected = np.zeros_like(aligned_waveform)
    expected[start_sample:start_sample + raw_station.waveform_resampled.shape[0], :] = raw_station.waveform_resampled

    inserted_slice = aligned_waveform[start_sample:start_sample + valid_n_samples]
    max_abs_diff_inserted = float(np.max(np.abs(inserted_slice - raw_station.waveform_resampled)))
    rms_diff_inserted = float(np.sqrt(np.mean((inserted_slice - raw_station.waveform_resampled) ** 2)))
    max_abs_diff_full = float(np.max(np.abs(aligned_waveform - expected)))

    print("Validation target")
    print(f"  EVENT: {event_id}")
    print(f"  wave_idx: {int(row['wave_idx'])}")
    print(f"  station_code: {row['station_code']}")
    print(f"  source_network: {row['source_network']}")
    print(f"  component_base: {row['component_base']}")
    print(f"  archive_relpath: {row['archive_relpath']}")
    print("")
    print("Numeric checks")
    print(f"  record_start_sample csv/hdf5: {int(row['record_start_sample'])} / {start_sample}")
    print(f"  valid_n_samples csv/hdf5: {int(row['valid_n_samples'])} / {valid_n_samples}")
    if "p_pick_trigger_aligned" in row.index:
        print(f"  p_pick_trigger_aligned csv/hdf5/raw-trigger: {int(row['p_pick_trigger_aligned'])} / {p_pick_trigger} / {start_sample + raw_station.p_pick_resampled}")
    print(f"  p_pick_aligned(final refined) csv/hdf5: {int(row['p_pick_aligned'])} / {p_pick}")
    if "p_pick_repaired_aligned" in row.index:
        print(f"  p_pick_repaired_aligned csv/hdf5: {int(row['p_pick_repaired_aligned'])} / {p_pick_repaired}")
    if "p_pick_refined_aligned" in row.index:
        print(f"  p_pick_refined_aligned csv/hdf5: {int(row['p_pick_refined_aligned'])} / {p_pick_refined}")
    print(f"  pga_norm_resampled_mps2 csv/hdf5/raw: {row['pga_norm_resampled_mps2']:.10f} / {converted['pga_norm_resampled_mps2']:.10f} / {raw_station.pga_norm_resampled_mps2:.10f}")
    print(f"  pga_norm_native_mps2 csv/hdf5/raw: {row['pga_norm_native_mps2']:.10f} / {converted['pga_norm_native_mps2']:.10f} / {raw_station.pga_norm_native_mps2:.10f}")
    print(f"  pga_norm_aligned_loc csv/hdf5: {int(row['pga_norm_aligned_loc'])} / {pga_loc}")
    print(f"  max_acc_header_gal raw: {raw_station.max_acc_header_gal}")
    print(f"  max_acc_header_gal hdf5: {converted['max_acc_header_gal']}")
    print(f"  inserted max_abs_diff: {max_abs_diff_inserted:.12e}")
    print(f"  inserted rms_diff: {rms_diff_inserted:.12e}")
    print(f"  full max_abs_diff: {max_abs_diff_full:.12e}")

    if args.plot:
        output_path = Path(args.plot).expanduser().resolve()
        build_plot(
            raw_resampled=raw_station.waveform_resampled,
            aligned=aligned_waveform,
            start_sample=start_sample,
            trigger_pick=p_pick_trigger,
            refined_pick=p_pick_refined,
            pga_loc=pga_loc,
            output_path=output_path,
        )
        print(f"  plot: {output_path}")


if __name__ == "__main__":
    main()
