import argparse
import json
import os

import loader_light as loader
from train_light import select_diverse_event_ids, write_overfit_event_ids


def main():
    parser = argparse.ArgumentParser(description='Export fixed overfit event ids from a training config.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--overfit_n', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])
    data_paths = training_params['data_path']
    if not isinstance(data_paths, list):
        data_paths = [data_paths]
    if len(data_paths) != 1:
        raise ValueError('export_overfit_event_ids.py currently supports exactly one data_path')

    overfit_n = args.overfit_n or int(training_params['overfit_n'])
    output = args.output or training_params['overfit_event_ids_path']

    all_data = loader.load_events(
        data_paths[0],
        event_metadata_path='overfit_ev.csv',
        parts=None,
        shuffle_train_dev=generator_params[0].get('shuffle_train_dev', False),
        custom_split=generator_params[0].get('custom_split', None),
        min_mag=generator_params[0].get('min_mag', None),
        mag_key=generator_params[0].get('key', 'MA'),
        overwrite_sampling_rate=training_params.get('overwrite_sampling_rate', None),
        decimate_events=generator_params[0].get('decimate_events', None),
        min_stalta_ratio_at_pick=training_params.get('min_stalta_ratio_at_pick', 0.1),
        station_filter=generator_params[0].get('station_filter', training_params.get('station_filter', None)),
    )
    event_ids = select_diverse_event_ids(
        all_data[0],
        overfit_n,
        mag_key=generator_params[0].get('key', 'MA'),
    )

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_overfit_event_ids(output, event_ids)
    print(f'Exported {len(event_ids)} event ids to {output}')


if __name__ == '__main__':
    main()
