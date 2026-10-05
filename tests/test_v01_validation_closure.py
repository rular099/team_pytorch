import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from gemini_util_light import PreloadedEventGenerator, JointGenerator
from tools.v01_validation_contract import V01NoPrediction, read_reference_pick, run_closure_inference


class V01ValidationClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'cache.hdf5'
        self.meta = pd.DataFrame({
            'EVENT': ['E0'] * 4 + ['E1'] * 4, 'wave_idx': list(range(4)) * 2,
            'Magnitude': [5.] * 8, 'Latitude': [35.] * 8,
            'Longitude': [139.] * 8, 'DEPTH': [10.] * 8})
        with h5py.File(self.path, 'w') as h5:
            for name in ('E0', 'E1'):
                g = h5.require_group('data/' + name)
                wave = np.zeros((4, 100, 3), np.float32)
                if name == 'E1':
                    wave[0] = np.linspace(0, 1, 100)[:, None]
                g['waveforms'] = wave
                g['waveform_valid_mask'] = np.zeros((4, 100), bool)
                if name == 'E1':
                    g['waveform_valid_mask'][0] = True
                g['coords'] = np.column_stack((35 + np.arange(4)*.01,
                                               139 + np.arange(4)*.01, np.zeros(4)))
                g['p_picks'] = np.array([10, -20, 200, 0])
                g['pga'] = np.array([np.nan, -2., -1., -.5])
                g['v01_source_role'] = np.array([1, 0, 0, 0])
                # Match build_v01_paired_manifest: one scalar per event, not (1,).
                g['v01_reference_p_pick'] = np.asarray(10, dtype=np.int64)
                g['station_codes'] = np.array(['S', 'Qneg', 'Qlate', 'Qunknown'], dtype='S16')
                g['v01_source_sensor_id'] = np.array(['S', '', '', ''], dtype='S16')
                g['v01_paired_acc_sensor_id'] = np.array(['AS', '', '', ''], dtype='S16')
                g.attrs.update(absolute_window_start_timestamp=1700000000., plan_hash=name, split='dev')

    def generator(self, closure=True, random=False, probability=1., mode='val'):
        return PreloadedEventGenerator(
            self.meta, {'sampling_rate': 10}, str(self.path), {'key': 'Magnitude', 'noise_seconds': 1},
            key='Magnitude', windowlen=100, shuffle=False, max_stations=4, pga_targets=3,
            pga_from_inactive=True, sampling_rate=10, trigger_based=True,
            magnitude_resampling=1, scale_metadata=False, deterministic_sampling_seed=42,
            emit_waveform_padding_mask=True, metadata_support_station_valid=True,
            v01_validation_closure=closure,
            realtime_training={'enabled': True, 'mode': mode, 'val_times': [1, 3, 5]},
            realtime_target_sampling={'enabled': True, 'input_ratio': 0.,
                                     'triggered_noninput_ratio': .2, 'untriggered_ratio': .5},
            causal_random_input_mask={'enabled': random, 'apply_probability': probability,
                                     'station_counts': [1]})

    def test_scalar_reference_matches_actual_generator_cutoff(self):
        with h5py.File(self.path, 'r') as h5:
            self.assertEqual(h5['data/E1/v01_reference_p_pick'].shape, ())
            self.assertEqual(read_reference_pick(h5['data/E1']), 10)
        gen = self.generator()
        for index in (3, 4, 5):
            request = gen.describe_request(index)
            info = gen[index][2]
            self.assertEqual(request['absolute_cutoff_utc'], info['v01_absolute_cutoff_utc'])

    def test_singleton_reference_matches_scalar_request_and_tensors(self):
        scalar = self.generator()
        request = scalar.describe_request(3)
        sample = scalar[3]
        with h5py.File(self.path, 'r+') as h5:
            group = h5['data/E1']
            del group['v01_reference_p_pick']
            group['v01_reference_p_pick'] = np.array([10], dtype=np.int64)
            self.assertEqual(read_reference_pick(group), 10)
        singleton = self.generator()
        self.assertEqual(singleton.describe_request(3), request)
        other = singleton[3]
        for left, right in zip(sample[0] + sample[1], other[0] + other[1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)

    def test_malformed_reference_fails_instead_of_changing_clock(self):
        for value in (np.array([], dtype=np.int64), np.array([10, 20]),
                      np.asarray(np.nan), np.asarray(10.5)):
            with self.subTest(value=value), h5py.File(self.path, 'r+') as h5:
                group = h5['data/E1']
                del group['v01_reference_p_pick']
                group['v01_reference_p_pick'] = value
                with self.assertRaisesRegex(ValueError, 'v01_reference_p_pick must be'):
                    read_reference_pick(group)
            with self.assertRaisesRegex(ValueError, 'v01_reference_p_pick must be'):
                self.generator().describe_request(3)

    def test_clock_copy_preserves_caller_and_legacy_alias_is_opt_in(self):
        gen = self.generator()
        before = np.ones((1, 4), bool)
        gen._select_realtime_cutout(np.array([[10, -20, 200, 0]]), before,
                                   np.random.default_rng(1), {'mode': 'val', 'time_index': 0}, 100)
        self.assertTrue(before.all())
        gen.v01_validation_closure = False
        gen._select_realtime_cutout(np.array([[10, -20, 200, 0]]), before,
                                   np.random.default_rng(1), {'mode': 'val', 'time_index': 0}, 100)
        np.testing.assert_array_equal(before, [[True, False, False, False]])

    def test_production_normal_retains_negative_late_unknown_query_labels(self):
        gen = self.generator()
        inputs, outputs, info = gen[3]
        self.assertEqual(int(inputs[4].sum()), 3)
        self.assertEqual(set(info['v01_query_sensor_id']), {'Qneg', 'Qlate', 'Qunknown'})
        self.assertEqual(int(inputs[2].sum()), 1)
        self.assertTrue(torch.all(inputs[0][~inputs[2]] == 0))
        self.assertEqual(info['v01_absolute_cutoff_utc'], 1700000002.)
        legacy = self.generator(False)
        with self.assertRaisesRegex(ValueError, 'Found event without PGA'):
            legacy._get_one(3)

    def test_no_neighbour_substitution_for_empty_source(self):
        gen = self.generator()
        with self.assertRaises(V01NoPrediction) as ctx:
            gen[0]
        self.assertEqual(ctx.exception.outcome, 'no_available_source')
        self.assertEqual(gen.describe_request(0)['event_id'], 'E0')
        self.assertEqual(self.generator(False, True)[0][2]['event_id'], 'E1')

    def test_random_and_mixed_enabled_numeric_inputs_are_unchanged(self):
        for probability in (0., 1.):
            before = self.generator(False, True, probability)._get_one(3)
            after = self.generator(True, True, probability)._get_one(3)
            for a, b in zip(before[0] + before[1], after[0] + after[1]):
                torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
            self.assertTrue(torch.equal(after[2]['v01_source_role'][after[0][2]],
                                        torch.ones(int(after[0][2].sum()), dtype=torch.bool)))

    def test_closure_rejects_training_mode(self):
        with self.assertRaisesRegex(ValueError, 'restricted to realtime validation'):
            self.generator(mode='train')

    def test_ledger_counts_and_joint_request_identity(self):
        dataset = JointGenerator([self.generator()], shuffle=False)
        def inference(_model, holder, _device, _config, indices):
            sample = holder[indices[0]]
            return {'event_id': [sample[2]['event_id']]}
        results, summary = run_closure_inference(
            None, dataset, None, {}, range(len(dataset)),
            Path(self.tmp.name)/'requests.jsonl', inference)
        self.assertEqual(summary['requested'], 6)
        self.assertEqual(summary['outcomes']['predicted'], 3)
        self.assertEqual(summary['outcomes']['no_available_source'], 3)
        self.assertEqual(results['event_id'], ['E1'] * 3)
        self.assertEqual(sum(summary['outcomes'].values()), summary['requested'])

    def test_implementation_errors_abort_and_are_not_skipped(self):
        gen = self.generator()
        def broken(*args, **kwargs):
            raise ValueError('broken evaluator')
        with self.assertRaisesRegex(ValueError, 'broken evaluator'):
            run_closure_inference(None, gen, None, {}, [3], Path(self.tmp.name)/'failed.jsonl', broken)

    def test_training_mixed_path_is_not_migrated_by_default(self):
        gen = self.generator(False,True,.5,mode='train')
        self.assertFalse(gen.v01_validation_closure)
        context = {'mode':'train','bin_index':0}
        mask = np.ones((1,4),bool)
        clock = gen._select_realtime_cutout(np.array([[10,-20,200,0]]),mask,
                                           np.random.default_rng(42),context,100)
        self.assertGreaterEqual(clock['cutout'],1)
        np.testing.assert_array_equal(mask,[[True,False,False,False]])


if __name__ == '__main__':
    unittest.main()
