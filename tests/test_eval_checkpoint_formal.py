import math
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import eval_checkpoint


class FormalPgaMetricTests(unittest.TestCase):
    def test_split_aliases_are_canonical_and_deduplicated(self):
        self.assertEqual(
            eval_checkpoint._canonical_eval_splits('validation,test,dev,train'),
            ['val', 'test', 'train'],
        )
        with self.assertRaisesRegex(ValueError, 'Unknown eval split'):
            eval_checkpoint._canonical_eval_splits('holdout')

    def test_single_standard_normal_nll(self):
        nll = eval_checkpoint._mixture_nll_1d(
            weights=np.array([[1.0]]),
            mu=np.array([[[0.0]]]),
            sigma=np.array([[[1.0]]]),
            target=np.array([0.0]),
        )
        self.assertAlmostEqual(float(nll[0]), 0.5 * math.log(2.0 * math.pi))

    def test_nll_uses_logits_without_softmax_underflow(self):
        logits = np.array([[0.0, -1000.0]])
        nll = eval_checkpoint._mixture_nll_1d(
            weights=np.array([[1.0, 0.0]]),
            mu=np.array([[[100.0], [0.0]]]),
            sigma=np.array([[[1.0], [1.0]]]),
            target=np.array([0.0]),
            alpha_logits=logits,
        )
        self.assertAlmostEqual(
            float(nll[0]),
            1000.0 + 0.5 * math.log(2.0 * math.pi),
            places=6,
        )

    def test_formal_metrics_use_valid_targets_and_requested_definitions(self):
        results = {
            'pga_label': [np.array([0.0, 1.0])],
            'pga_mu_best': [np.array([0.1, 0.8])],
            'pga_target_valid': [np.array([True, True])],
            'pga_sigma': [np.array([0.2, 0.1])],
            'pga_nll_log10_mps2': [np.array([1.0, 3.0])],
            'pga_nll_model_space': [np.array([2.0, 4.0])],
            'pga_prob_ge_threshold': [np.array([0.2, 0.7])],
        }
        config = {
            'training_params': {
                'pga_loss_weighting': {'threshold': 0.5},
            },
        }

        metrics = eval_checkpoint.compute_formal_pga_metrics(results, config)

        self.assertEqual(metrics['targets'], 2)
        self.assertAlmostEqual(metrics['mae'], 0.15)
        self.assertAlmostEqual(metrics['rmse'], math.sqrt(0.025))
        self.assertAlmostEqual(metrics['r2'], 0.9)
        self.assertAlmostEqual(metrics['slope'], 0.7)
        self.assertAlmostEqual(metrics['intercept'], 0.1)
        self.assertAlmostEqual(metrics['nll'], 2.0)
        self.assertAlmostEqual(metrics['nll_model_space'], 3.0)
        self.assertAlmostEqual(metrics['brier'], 0.065)
        self.assertAlmostEqual(metrics['coverage_1sigma'], 0.5)
        self.assertAlmostEqual(metrics['coverage_2sigma'], 1.0)


class TestSplitDatasetTests(unittest.TestCase):
    def test_test_split_uses_test_partition_and_validation_protocol(self):
        class FakeGenerator:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with tempfile.TemporaryDirectory() as cache_dir:
            config = {
                'seed': 42,
                'model_params': {
                    'max_stations': 25,
                    'n_pga_targets': 15,
                    'no_event_token': False,
                },
                'training_params': {
                    'data_path': ['/data/year.hdf5'],
                    'metadata_cache_dir': cache_dir,
                    'generator_params': [{
                        'batch_size': 8,
                        'cutout_start': 0,
                        'cutout_end': 90,
                        'noise_seconds': 5,
                        'oversample': 4,
                        'shuffle_train_dev': False,
                    }],
                    'validation_generator_overrides': {
                        'oversample': 1,
                        'realtime_training': {
                            'enabled': True,
                            'mode': 'val',
                            'val_times': [1, 3, 5, 10, 20, 40, 90],
                        },
                    },
                },
            }
            loaded = ('test-events', {}, {'sampling_rate': 100})
            with mock.patch.object(
                eval_checkpoint.loader,
                'load_events',
                return_value=loaded,
            ) as load_events, mock.patch.object(
                eval_checkpoint.util,
                'PreloadedEventGenerator',
                FakeGenerator,
            ):
                datasets = eval_checkpoint.build_datasets(
                    config,
                    splits=['test'],
                )

        self.assertEqual(set(datasets), {'test'})
        self.assertEqual(load_events.call_args.kwargs['parts'], (False, False, True))
        test_dataset = datasets['test']
        self.assertEqual(test_dataset.kwargs['oversample'], 1)
        self.assertFalse(test_dataset.kwargs['shuffle'])
        self.assertEqual(test_dataset.kwargs['dpk_prior_cache_split'], 'test')
        self.assertEqual(
            test_dataset.kwargs['realtime_training']['mode'],
            'val',
        )
        self.assertEqual(
            test_dataset.kwargs['realtime_training']['val_times'],
            [1, 3, 5, 10, 20, 40, 90],
        )


class WaveformStationRollTests(unittest.TestCase):
    def test_roll_moves_waveform_mask_and_cached_weights_only(self):
        waveform = torch.arange(18).reshape(3, 2, 3)
        metadata = torch.tensor([[10.0], [20.0], [30.0]])
        station_valid = torch.tensor([True, True, False])
        pga_targets = torch.tensor([[1.0], [2.0]])
        pga_valid = torch.tensor([True, True])
        padding_mask = torch.tensor([
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ])
        cached_weights = torch.arange(12.0).reshape(3, 2, 2)
        inputs = [
            waveform,
            metadata,
            station_valid,
            pga_targets,
            pga_valid,
            padding_mask,
            cached_weights,
        ]

        permuted, source_slots = eval_checkpoint.apply_waveform_station_permutation(
            inputs,
            mode='roll',
        )

        np.testing.assert_array_equal(source_slots, np.array([1, 0, 2]))
        torch.testing.assert_close(permuted[0], waveform[[1, 0, 2]])
        torch.testing.assert_close(permuted[5], padding_mask[[1, 0, 2]])
        torch.testing.assert_close(permuted[6], cached_weights[[1, 0, 2]])
        torch.testing.assert_close(permuted[1], metadata)
        torch.testing.assert_close(permuted[2], station_valid)
        torch.testing.assert_close(permuted[3], pga_targets)
        torch.testing.assert_close(permuted[4], pga_valid)


if __name__ == '__main__':
    unittest.main()
