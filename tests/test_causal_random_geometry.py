import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import torch

from gemini_util_light import PreloadedEventGenerator
from train_light import load_config_file


def _bare_generator(mask_config):
    generator = PreloadedEventGenerator.__new__(PreloadedEventGenerator)
    generator.trace_length = 100
    generator.max_stations = 4
    generator.wave_eps = 1e-5
    generator.causal_random_input_mask = mask_config
    generator.realtime_target_sampling = {
        'enabled': True,
        'input_ratio': 0.3,
        'triggered_noninput_ratio': 0.2,
        'untriggered_ratio': 0.5,
        'fill_missing': True,
        'exclude_inputs': False,
    }
    return generator


class CausalRandomGeometryTests(unittest.TestCase):
    def test_mask_normalization_is_disabled_by_default_and_requires_realtime(self):
        self.assertEqual(
            PreloadedEventGenerator._normalize_causal_random_input_mask(None),
            {'enabled': False},
        )
        with self.assertRaisesRegex(ValueError, 'requires realtime_training'):
            PreloadedEventGenerator._normalize_causal_random_input_mask(
                {'enabled': True},
                realtime_enabled=False,
            )

        normalized = PreloadedEventGenerator._normalize_causal_random_input_mask(
            {
                'enabled': True,
                'apply_probability': 0.5,
                'station_counts': [8, 1, 3, 3],
                'target_sampling': {
                    'enabled': True,
                    'input_ratio': 0,
                    'triggered_noninput_ratio': 0.2,
                    'untriggered_ratio': 0.5,
                    'exclude_inputs': True,
                },
            },
            realtime_enabled=True,
        )
        self.assertEqual(normalized['station_counts'], [1, 3, 8])
        self.assertEqual(normalized['apply_probability'], 0.5)
        self.assertTrue(normalized['target_sampling']['exclude_inputs'])

    def test_full_event_selection_is_causal_random_and_pick_ordered(self):
        mask_config = {
            'enabled': True,
            'apply_probability': 1.0,
            'station_counts': [2],
            'order_selected_by_pick': True,
        }
        generator = _bare_generator(mask_config)
        picks = np.array([10, 20, 40, 80, 0], dtype=float)
        coords = np.array([
            [35.0, 139.0, 0.0],
            [35.1, 139.1, 0.0],
            [35.2, 139.2, 0.0],
            [35.3, 139.3, 0.0],
            [35.4, 139.4, 0.0],
        ])
        waveforms = np.ones((5, 100, 3), dtype=np.float32)

        sample = generator._sample_causal_random_input_indices(
            picks,
            coords,
            waveforms,
            current_sample=50,
            rng=np.random.default_rng(123),
        )

        self.assertTrue(sample['applied'])
        self.assertEqual(sample['available_count'], 3)
        self.assertEqual(sample['requested_count'], 2)
        self.assertEqual(len(sample['selected_indices']), 2)
        self.assertTrue(set(sample['selected_indices']).issubset({0, 1, 2}))
        selected_picks = picks[sample['selected_indices']]
        self.assertTrue(np.all(selected_picks[:-1] <= selected_picks[1:]))

    def test_noncausal_and_zero_signal_stations_are_never_selected(self):
        mask_config = {
            'enabled': True,
            'apply_probability': 1.0,
            'station_counts': [4],
            'order_selected_by_pick': True,
        }
        generator = _bare_generator(mask_config)
        picks = np.array([10, 20, 60, 30], dtype=float)
        coords = np.ones((4, 3), dtype=float)
        waveforms = np.ones((4, 100, 3), dtype=np.float32)
        waveforms[1] = 0.0

        sample = generator._sample_causal_random_input_indices(
            picks,
            coords,
            waveforms,
            current_sample=50,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_array_equal(sample['selected_indices'], np.array([0, 3]))

    def test_random_inputs_reserve_a_distinct_finite_pga_target(self):
        generator = _bare_generator({
            'enabled': True,
            'apply_probability': 1.0,
            'station_counts': [4],
            'order_selected_by_pick': True,
        })
        picks = np.full(4, 10, dtype=float)
        coords = np.ones((4, 3), dtype=float)
        waveforms = np.ones((4, 100, 3), dtype=np.float32)

        sample = generator._sample_causal_random_input_indices(
            picks,
            coords,
            waveforms,
            current_sample=50,
            target_values=np.arange(4, dtype=float),
            rng=np.random.default_rng(17),
        )

        self.assertEqual(sample['available_count'], 4)
        self.assertEqual(sample['requested_count'], 4)
        self.assertEqual(len(sample['selected_indices']), 3)
        self.assertEqual(len(set(range(4)) - set(sample['selected_indices'])), 1)

    def test_random_geometry_target_sampling_excludes_input_stations(self):
        generator = _bare_generator({'enabled': False})
        sampling = PreloadedEventGenerator._normalize_realtime_target_sampling(
            {
                'enabled': True,
                'input_ratio': 0.0,
                'triggered_noninput_ratio': 0.2,
                'untriggered_ratio': 0.5,
                'fill_missing': True,
                'exclude_inputs': True,
            },
            realtime_enabled=True,
        )
        active = np.arange(6)
        input_valid = np.array([True, True, False, False])
        picks = np.array([10, 20, 30, 60, 80, 0], dtype=float)

        selected, target_types = generator._sample_realtime_pga_targets(
            active,
            input_valid,
            picks,
            current_sample=40,
            n_targets=4,
            rng=np.random.default_rng(11),
            sampling_config=sampling,
        )

        self.assertFalse(set(selected).intersection({0, 1}))
        self.assertTrue(np.all(target_types != 0))

    def test_end_to_end_random_mask_zeros_unselected_inputs_and_excludes_targets(self):
        station_count = 6
        trace_length = 100
        event_metadata = pd.DataFrame({
            'EVENT': ['E1'] * station_count,
            'wave_idx': np.arange(station_count),
            'Magnitude': [5.0] * station_count,
            'Latitude': [35.0] * station_count,
            'Longitude': [139.0] * station_count,
            'DEPTH': [10.0] * station_count,
        })
        waveforms = np.zeros((station_count, trace_length, 3), dtype=np.float32)
        for station_idx in range(station_count):
            waveforms[station_idx, :, :] = np.linspace(
                0.0,
                float(station_idx + 1),
                trace_length,
                dtype=np.float32,
            )[:, None]
        picks = np.array([10, 20, 30, 40, 80, 0], dtype=np.int64)
        coords = np.column_stack((
            35.0 + np.arange(station_count) * 0.01,
            139.0 + np.arange(station_count) * 0.01,
            np.zeros(station_count),
        )).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, 'event.hdf5')
            with h5py.File(data_path, 'w') as h5:
                event = h5.create_group('data').create_group('E1')
                event.create_dataset('waveforms', data=waveforms)
                event.create_dataset('coords', data=coords)
                event.create_dataset('p_picks', data=picks)
                event.create_dataset('pga', data=np.arange(station_count, dtype=np.float32))

            generator = PreloadedEventGenerator(
                event_metadata,
                {'sampling_rate': 10},
                data_path,
                {'key': 'Magnitude', 'noise_seconds': 1},
                key='Magnitude',
                windowlen=trace_length,
                shuffle=False,
                max_stations=4,
                pga_targets=3,
                pga_from_inactive=True,
                sampling_rate=10,
                trigger_based=True,
                magnitude_resampling=1,
                scale_metadata=False,
                deterministic_sampling_seed=42,
                realtime_training={
                    'enabled': True,
                    'mode': 'val',
                    'val_times': [3],
                },
                realtime_target_sampling={
                    'enabled': True,
                    'input_ratio': 0.3,
                    'triggered_noninput_ratio': 0.2,
                    'untriggered_ratio': 0.5,
                },
                causal_random_input_mask={
                    'enabled': True,
                    'apply_probability': 1.0,
                    'station_counts': [2],
                    'target_sampling': {
                        'enabled': True,
                        'input_ratio': 0.0,
                        'triggered_noninput_ratio': 0.2,
                        'untriggered_ratio': 0.5,
                        'fill_missing': True,
                        'exclude_inputs': True,
                    },
                },
            )
            inputs, _, pick_info = generator[0]

        model_waveforms, _, station_valid = inputs[:3]
        self.assertEqual(int(station_valid.sum()), 2)
        self.assertTrue(torch.all(model_waveforms[~station_valid] == 0))
        selected = pick_info['selected_input_indices'][station_valid]
        self.assertTrue(set(selected.tolist()).issubset({0, 1, 2, 3}))
        target_indices = pick_info['pga_target_indices']
        target_indices = target_indices[target_indices >= 0]
        self.assertFalse(set(target_indices.tolist()).intersection({0, 1}))
        self.assertTrue(bool(pick_info['causal_random_mask_applied']))

    def test_rt56_preserves_rt55_model_architecture_and_uses_weight_only_init(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rt55_path = os.path.join(
            repo_root,
            'pga_configs',
            'transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json',
        )
        rt56_path = os.path.join(
            repo_root,
            'pga_configs',
            'transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json',
        )
        rt55_env = {
            'JAPAN_FULL_DATA_ROOT': '/tmp/japan-data',
            'JAPAN_FULL_WEIGHT_PATH': 'weights-rt55',
        }
        rt56_env = {
            'JAPAN_FULL_DATA_ROOT': '/tmp/japan-data',
            'RT55_EP32_CHECKPOINT': '/tmp/full_model_best_ep32.pth',
            'RT56_WEIGHT_PATH': 'weights-rt56',
        }
        with mock.patch.dict(os.environ, rt55_env, clear=True):
            rt55 = load_config_file(rt55_path)
        # JAPAN_FULL_WEIGHT_PATH appears only in the rt55 parent and is
        # replaced by rt56.  Loading the child must not require that obsolete
        # parent placeholder to be exported.
        with mock.patch.dict(os.environ, rt56_env, clear=True):
            rt56 = load_config_file(rt56_path)
        unresolved_rt56_env = dict(rt56_env)
        unresolved_rt56_env.pop('RT56_WEIGHT_PATH')
        with mock.patch.dict(os.environ, unresolved_rt56_env, clear=True):
            with self.assertRaisesRegex(ValueError, 'RT56_WEIGHT_PATH'):
                load_config_file(rt56_path)

        self.assertEqual(rt56['model_params'], rt55['model_params'])
        self.assertNotIn('causal_random_input_mask', rt55['training_params']['generator_params'][0])
        self.assertTrue(rt55['training_params']['transfer_model_path'])
        self.assertEqual(
            rt56['training_params']['load_model_path'],
            rt56_env['RT55_EP32_CHECKPOINT'],
        )
        self.assertIsNone(rt56['training_params']['transfer_model_path'])
        self.assertFalse(rt56['training_params']['reinit_fpn'])
        self.assertEqual(
            rt56['training_params']['train_generator_overrides']['causal_random_input_mask']['apply_probability'],
            0.5,
        )
        self.assertEqual(
            rt56['training_params']['validation_generator_overrides']['causal_random_input_mask']['apply_probability'],
            1.0,
        )
        self.assertEqual(
            rt56['training_params']['validation_generator_overrides']['realtime_training']['mode'],
            'val',
        )
        self.assertEqual(
            rt56['training_params']['validation_generator_overrides']['realtime_training']['val_times'],
            [1, 3, 5, 10, 20, 40, 90],
        )


if __name__ == '__main__':
    unittest.main()
