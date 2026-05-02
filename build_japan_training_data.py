import argparse
from pathlib import Path

from japan_dataset_builder import build_year_dataset, discover_years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-ready Japan HDF5 + CSV metadata directly from raw waveform tar archives."
    )
    parser.add_argument("--waveform_root", required=True, help="Root directory containing yearly waveform subdirectories.")
    parser.add_argument("--output_dir", required=True, help="Directory to write HDF5 and CSV outputs.")
    parser.add_argument("--year", type=int, action="append", help="Year to process. Repeat to process multiple years. Defaults to all years found.")
    parser.add_argument("--min_stations", type=int, default=3, help="Minimum valid stations required to keep an event.")
    parser.add_argument("--target_sampling_rate", type=float, default=100.0, help="Target sampling rate in Hz.")
    parser.add_argument("--limit_events", type=int, default=None, help="Optional cap on the number of outer event archives per year.")
    parser.add_argument("--compression_level", type=int, default=4, help="gzip compression level for output HDF5.")
    parser.add_argument("--pick_mode", choices=["trigger_repair", "travel_time"], default="travel_time", help="Coarse P-pick source before STA/LTA refinement.")
    parser.add_argument("--p_velocity_km_s", type=float, default=6.0, help="P-wave velocity for --pick_mode travel_time.")
    parser.add_argument("--travel_time_intercept_s", type=float, default=0.0, help="Constant offset added to distance / velocity.")
    parser.add_argument("--stalta_pre_seconds", type=float, default=10.0, help="Seconds before coarse pick searched by STA/LTA.")
    parser.add_argument("--stalta_post_seconds", type=float, default=10.0, help="Seconds after coarse pick searched by STA/LTA.")
    parser.add_argument("--stalta_sta_seconds", type=float, default=0.2, help="STA window length in seconds.")
    parser.add_argument("--stalta_lta_seconds", type=float, default=1.0, help="LTA window length in seconds.")
    parser.add_argument("--stalta_threshold_ratio", type=float, default=2.5, help="STA/LTA threshold ratio.")
    parser.add_argument("--stalta_feature", choices=["vertical", "norm"], default="vertical", help="STA/LTA characteristic function.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    waveform_root = Path(args.waveform_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    years = args.year if args.year else discover_years(waveform_root)
    if not years:
        raise SystemExit(f"No year directories found under {waveform_root}")

    for year in years:
        output_hdf5 = output_dir / f"japan_{year}.hdf5"
        output_events_csv = output_dir / f"japan_{year}_events.csv"
        output_stations_csv = output_dir / f"japan_{year}_stations.csv"

        for path in (output_hdf5, output_events_csv, output_stations_csv):
            if path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite existing file without --overwrite: {path}")
            if path.exists() and args.overwrite:
                path.unlink()

        stats = build_year_dataset(
            waveform_root=waveform_root,
            output_hdf5=output_hdf5,
            output_events_csv=output_events_csv,
            output_stations_csv=output_stations_csv,
            year=year,
            target_sampling_rate_hz=args.target_sampling_rate,
            min_stations=args.min_stations,
            limit_events=args.limit_events,
            compression_level=args.compression_level,
            pick_mode=args.pick_mode,
            p_velocity_km_s=args.p_velocity_km_s,
            travel_time_intercept_s=args.travel_time_intercept_s,
            stalta_pre_seconds=args.stalta_pre_seconds,
            stalta_post_seconds=args.stalta_post_seconds,
            stalta_sta_seconds=args.stalta_sta_seconds,
            stalta_lta_seconds=args.stalta_lta_seconds,
            stalta_threshold_ratio=args.stalta_threshold_ratio,
            stalta_feature=args.stalta_feature,
        )

        print(f"\nYear {year} complete")
        print(f"  HDF5: {output_hdf5}")
        print(f"  Events CSV: {output_events_csv}")
        print(f"  Stations CSV: {output_stations_csv}")
        for key in sorted(stats):
            print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
