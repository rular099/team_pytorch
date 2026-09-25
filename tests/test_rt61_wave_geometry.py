import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_light
from tools.analyze_rt61_wave_geometry_npz import (
    _field_error_decomposition,
    _field_summary,
    _paired_field_ci,
)
from tools.rt60_contrast_objective import tensor_mapping_sha256
from tools.rt61_wave_geometry import (
    ADAPTER_PREFIX,
    build_reference_payload,
    build_trainable_delta_payload,
    model_shared_fingerprint,
    restore_trainable_delta,
    trainable_manifest,
)

models = train_light.models


class _Wrapper(nn.Module):
    def __init__(self, rt61=True):
        super().__init__()
        self.waveform_model = nn.ModuleList([
            nn.Linear(1, 1), nn.Linear(1, 1),
        ])
        self.pga_anchor_residual_transport_head = (
            models.PGAAnchorResidualTransportHead(
                station_dim=4,
                emb_dim=6,
                local_hidden_dim=8,
                transport_hidden_dim=256,
                set_layers=1,
                heads=2,
                enable_rt60_reference=rt61,
                use_rt61_wave_geometry_adapter=rt61,
                rt61_wave_geometry_rank=16,
            )
        )


def _inputs(query_coords, station_valid=None, query_valid=None, nan_padding=False):
    torch.manual_seed(61)
    station_coords = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    station_valid = (
        torch.ones(1, 2, dtype=torch.bool)
        if station_valid is None else station_valid
    )
    query_valid = (
        torch.ones(1, query_coords.shape[1], dtype=torch.bool)
        if query_valid is None else query_valid
    )
    station_u = torch.randn(1, 2, 4)
    station_d = torch.randn(1, 2, 4)
    query_emb = torch.randn(1, query_coords.shape[1], 6)
    if nan_padding:
        station_u[~station_valid] = float('nan')
        station_d[~station_valid] = float('nan')
        station_coords[~station_valid] = float('nan')
        query_emb[~query_valid] = float('nan')
        query_coords = query_coords.clone()
        query_coords[~query_valid] = float('nan')
    return (
        station_u,
        station_d,
        torch.randn(1, 6),
        query_emb,
        station_coords,
        query_coords,
        station_valid,
        query_valid,
        torch.randn(1, 2),
        torch.randn(1, 2, 1),
        torch.ones(1, 2, 1),
        torch.randn(1, query_coords.shape[1], 1),
    )


def _set_nonzero_parent_readout(head):
    with torch.no_grad():
        nn.init.xavier_uniform_(head.transport.residual_head[-1].weight)
        head.transport.residual_head[-1].bias.fill_(0.03)


def _configure_reference(head):
    head.transport.set_rt60_reference_state({
        key: value.detach().clone()
        for key, value in head.transport.residual_head.state_dict().items()
    })


class RT61WaveGeometryTests(unittest.TestCase):
    def test_config_contract_and_legacy_config_bytes(self):
        rt60 = REPO_ROOT / 'pga_configs' / (
            'transformer_japan_full_2000_2024_rt60_'
            'contrast_readout_seed42_chaosuan.json'
        )
        self.assertEqual(
            hashlib.sha256(rt60.read_bytes()).hexdigest(),
            '43f5a149adcba305348d84eee4cd4e9b949dba2ae8681d23ebd727429333f7fd',
        )
        env = {
            'JAPAN_FULL_DATA_ROOT': '/tmp/japan',
            'JAPAN_FULL_WEIGHT_PATH': '/tmp/rt55',
            'RT55_EP32_CHECKPOINT': '/tmp/rt55.pth',
            'RT56_WEIGHT_PATH': '/tmp/rt56',
            'RT56_BASE_CHECKPOINT': '/tmp/rt56.pth',
            'RT57_WEIGHT_PATH': '/tmp/rt57',
            'RT57_BASE_CHECKPOINT': '/tmp/rt57.pth',
            'RT58_WEIGHT_PATH': '/tmp/rt58',
            'RT58_BASE_CHECKPOINT': '/tmp/rt58.pth',
            'RT59_PARENT_CHECKPOINT': '/tmp/rt59.pth',
            'RT61_WEIGHT_PATH': '/tmp/rt61',
            'RT61_SOURCE_MANIFEST_SHA256': 'a' * 64,
        }
        path = REPO_ROOT / 'pga_configs' / (
            'transformer_japan_full_2000_2024_rt61_'
            'wave_geometry_residual_seed42_chaosuan.json'
        )
        with mock.patch.dict(os.environ, env, clear=False):
            config = train_light.load_config_file(str(path))
        self.assertTrue(config['model_params']['use_rt61_wave_geometry_adapter'])
        self.assertEqual(config['model_params']['rt61_wave_geometry_rank'], 16)
        training = config['training_params']
        self.assertEqual(
            training['freeze_mode'], 'rt61_wave_geometry_residual_only'
        )
        self.assertEqual(len(training['data_path']), 25)
        self.assertEqual(
            training['rt61_parent']['checkpoint_sha256'],
            '5dbc15c6af5b8d341dfd4a377a68217b1e1aa3589550690997c02e81332aa1cd',
        )

    def test_opt_in_state_dict_and_identity_reference_isolation(self):
        torch.manual_seed(10)
        legacy = _Wrapper(rt61=False)
        torch.manual_seed(10)
        model = _Wrapper(rt61=True)
        missing, unexpected = model.load_state_dict(legacy.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith(ADAPTER_PREFIX) for name in missing))
        _set_nonzero_parent_readout(model.pga_anchor_residual_transport_head)
        _configure_reference(model.pga_anchor_residual_transport_head)
        query = torch.tensor([[[0., 0., 0.], [3., 0., 0.]]])
        inputs = _inputs(query)
        initial = model.pga_anchor_residual_transport_head(*inputs)
        torch.testing.assert_close(
            initial['applied_delta'], initial['reference_applied_delta'],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            initial['reference_applied_delta'] + initial['rt61_increment'],
            initial['applied_delta'],
        )
        self.assertEqual(initial['rt61_increment'][0, 0].item(), 0.0)
        reference = initial['reference_applied_delta'].detach().clone()
        weights = initial['weights'].detach().clone()
        level = initial['level'].detach().clone()
        with torch.no_grad():
            transport = model.pga_anchor_residual_transport_head.transport
            transport.rt61_wave_geometry_adapter.output_projection.weight.normal_()
            transport.residual_head[-1].bias.add_(0.4)
        changed = model.pga_anchor_residual_transport_head(*inputs)
        torch.testing.assert_close(reference, changed['reference_applied_delta'])
        torch.testing.assert_close(weights, changed['weights'])
        torch.testing.assert_close(level, changed['level'])
        self.assertFalse(torch.equal(
            changed['applied_delta'][0, 1], reference[0, 1]
        ))

    def test_gradient_reach_permutations_chunk_repeat_and_nan_masks(self):
        model = _Wrapper(rt61=True)
        head = model.pga_anchor_residual_transport_head
        _set_nonzero_parent_readout(head)
        _configure_reference(head)
        train_light.apply_full_model_trainability(
            model, {'freeze_mode': 'rt61_wave_geometry_residual_only'}
        )
        optimizer, groups = train_light.build_optimizer_with_groups(model, {
            'lr': 1e-4, 'lr_rt61_trainable': 1e-4, 'optimizer': 'adam',
        })
        manifest = trainable_manifest(model)
        self.assertEqual(groups['rt61_trainable']['n_params'], manifest['scalar_count'])
        self.assertEqual(manifest['tensor_count'], 9)
        query = torch.tensor([[[2., 0., 0.], [4., 1., 0.]]])
        inputs = _inputs(query)
        before_step = head(*inputs)
        frozen_reference = before_step['reference_applied_delta'].detach().clone()
        frozen_weights = before_step['weights'].detach().clone()
        frozen_level = before_step['level'].detach().clone()
        first = before_step['applied_delta'].sum()
        first.backward()
        adapter = head.transport.rt61_wave_geometry_adapter
        self.assertGreater(adapter.output_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(adapter.wave_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(adapter.geometry_projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        second = head(*inputs)['applied_delta'].sum()
        second.backward()
        self.assertGreater(adapter.wave_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.geometry_projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        after_steps = head(*inputs)
        torch.testing.assert_close(
            after_steps['reference_applied_delta'], frozen_reference
        )
        torch.testing.assert_close(after_steps['weights'], frozen_weights)
        torch.testing.assert_close(after_steps['level'], frozen_level)

        head.eval()
        whole = head(*inputs)['applied_delta'].detach()
        station_perm = torch.tensor([1, 0])
        station_permuted_inputs = list(inputs)
        for position in (0, 1, 4, 6, 8, 9, 10):
            station_permuted_inputs[position] = inputs[position][:, station_perm]
        station_permuted = head(*station_permuted_inputs)[
            'applied_delta'
        ].detach()
        torch.testing.assert_close(station_permuted, whole)
        perm = torch.tensor([1, 0])
        permuted_inputs = list(inputs)
        permuted_inputs[5] = inputs[5][:, perm]
        permuted_inputs[3] = inputs[3][:, perm]
        permuted_inputs[7] = inputs[7][:, perm]
        permuted_inputs[11] = inputs[11][:, perm]
        permuted = head(*permuted_inputs)['applied_delta'].detach()
        torch.testing.assert_close(permuted, whole[:, perm])
        chunks = []
        for index in range(query.shape[1]):
            chunk_inputs = list(inputs)
            for position in (3, 5, 7, 11):
                chunk_inputs[position] = inputs[position][:, index:index + 1]
            chunks.append(head(*chunk_inputs)['applied_delta'].detach())
        torch.testing.assert_close(torch.cat(chunks, dim=1), whole)
        repeat_inputs = list(inputs)
        for position in (3, 5, 7, 11):
            repeat_inputs[position] = torch.cat([
                inputs[position], inputs[position][:, :1]
            ], dim=1)
        repeated = head(*repeat_inputs)['applied_delta'].detach()
        torch.testing.assert_close(repeated[:, :2], whole)
        torch.testing.assert_close(repeated[:, 2], whole[:, 0])

        masked = _inputs(
            torch.tensor([[[2., 0., 0.], [0., 0., 0.]]]),
            station_valid=torch.tensor([[True, False]]),
            query_valid=torch.tensor([[True, False]]),
            nan_padding=True,
        )
        masked_output = head(*masked)
        self.assertTrue(torch.isfinite(masked_output['applied_delta']).all())
        self.assertEqual(masked_output['applied_delta'][0, 1].item(), 0.0)

    def test_shared_fingerprint_checkpoint_and_delta_reconstruction(self):
        torch.manual_seed(21)
        parent = _Wrapper(rt61=False)
        parent.pga_anchor_residual_transport_head._warm_copy_complete.fill_(True)
        checkpoint = {
            'epoch': 8,
            'task_id': '20260915-rt59-dual-objective-transport-v3',
            'model_state_dict': parent.state_dict(),
        }
        torch.manual_seed(22)
        model = _Wrapper(rt61=True)
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        self.assertFalse(unexpected)
        self.assertTrue(all(name.startswith(ADAPTER_PREFIX) for name in missing))
        expected_readout = tensor_mapping_sha256({
            key: value.detach().clone()
            for key, value in model.pga_anchor_residual_transport_head.transport
            .residual_head.state_dict().items()
        })
        payload = build_reference_payload(
            model,
            checkpoint,
            'a' * 64,
            {'enabled': True, 'mean': -1.0, 'std': 0.5},
            {'manifest': 'b' * 64},
            '20260915-rt59-dual-objective-transport-v3',
            8,
            expected_readout_sha256=expected_readout,
        )
        shared_before = model_shared_fingerprint(model)
        train_light.apply_full_model_trainability(
            model, {'freeze_mode': 'rt61_wave_geometry_residual_only'}
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(torch.randn_like(parameter) * 0.01)
        self.assertEqual(shared_before, model_shared_fingerprint(model))
        delta = build_trainable_delta_payload(model, payload)

        reconstructed = _Wrapper(rt61=True)
        reconstructed.load_state_dict(checkpoint['model_state_dict'], strict=False)
        restore_trainable_delta(reconstructed, delta)
        self.assertEqual(
            delta['trainable_state_sha256'],
            trainable_manifest(reconstructed)['state_sha256'],
        )
        query = torch.tensor([[[2., 0., 0.], [3., 0., 0.]]])
        inputs = _inputs(query)
        torch.testing.assert_close(
            model.pga_anchor_residual_transport_head(*inputs)['applied_delta'],
            reconstructed.pga_anchor_residual_transport_head(*inputs)[
                'applied_delta'
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rt61.pth')
            object.__setattr__(model, '_rt61_reference_identity', payload)
            object.__setattr__(
                model, '_rt61_task_id',
                '20260925-rt61-wave-geometry-residual-conditioning',
            )
            train_light.save_model_checkpoint(
                path, model, epoch=1,
                training_params={'checkpoint': {'save_optimizer_state': False}},
                extra={'rt61_schedule_completed_epochs': 1},
            )
            saved = torch.load(path, map_location='cpu')
            self.assertIn('rt61_reference', saved)
            self.assertTrue(os.path.isfile(
                os.path.splitext(path)[0] + '.rt61_delta.pth'
            ))
            resumed = _Wrapper(rt61=True)
            resumed.pga_anchor_residual_transport_head._warm_copy_complete.fill_(True)
            train_light.apply_full_model_trainability(
                resumed, {'freeze_mode': 'rt61_wave_geometry_residual_only'}
            )
            resumed_optimizer, _ = train_light.build_optimizer_with_groups(
                resumed,
                {'lr': 1e-4, 'lr_rt61_trainable': 1e-4, 'optimizer': 'adam'},
            )
            _, _, _, start_epoch, _ = train_light.load_checkpoint(
                resumed,
                resumed_optimizer,
                None,
                path,
                torch.device('cpu'),
            )
            self.assertEqual(start_epoch, 1)
            self.assertEqual(
                tensor_mapping_sha256(
                    resumed.pga_anchor_residual_transport_head.transport
                    .export_rt60_reference_state()
                ),
                payload['readout_sha256'],
            )

    def test_field_error_decomposition_and_constant_prediction(self):
        truth = np.array([[0., 1., 2., 3., 100.]])
        prediction = np.array([[0., 1., 2., 3., 4.]])
        constant = np.ones_like(prediction)
        data = {
            'valid': np.ones_like(truth, dtype=bool),
            'target': truth,
            'event_ids': np.array(['2020-test']),
            'requested_time': np.array([1.0]),
            'station_count': np.array([1]),
            'models': {
                'candidate_rt61': {'mean': prediction},
                'constant': {'mean': constant},
            },
        }
        records = _field_error_decomposition(
            'random', data, np.ones_like(truth, dtype=bool)
        )
        candidate, flat = records
        self.assertLess(candidate['mse_decomposition_abs_error'], 1e-10)
        canonical = np.percentile(prediction[0], 95) - np.percentile(
            prediction[0], 5
        )
        self.assertAlmostEqual(candidate['prediction_p95_p05_range'], canonical)
        self.assertNotEqual(candidate['prediction_p95_p05_range'], np.ptp(prediction[0]))
        self.assertTrue(candidate['projection_identifiable'])
        self.assertFalse(flat['projection_identifiable'])
        self.assertIsNone(flat['a_star_signed_oracle'])
        field_records = [
            {
                'eligible_ge5': True,
                'actual_station_count': 1,
                **{
                    f'{name}_{metric}': value
                    for name in ('historical', 'reference', 'candidate')
                    for metric, value in (
                        ('range_ratio', 1.0),
                        ('range_abs_error', 0.0),
                        ('pairwise_delta_mae', 0.0),
                        ('pairwise_delta_rmse', 0.0),
                    )
                },
                'event_id': '2020-a',
            },
            {
                'eligible_ge5': True,
                'actual_station_count': 1,
                **{
                    f'{name}_{metric}': value
                    for name in ('historical', 'reference', 'candidate')
                    for metric, value in (
                        ('range_ratio', 3.0),
                        ('range_abs_error', 10.0),
                        ('pairwise_delta_mae', 10.0),
                        ('pairwise_delta_rmse', 10.0),
                    )
                },
                'event_id': '2020-a',
            },
            {
                'eligible_ge5': True,
                'actual_station_count': 1,
                **{
                    f'{name}_{metric}': value
                    for name in ('historical', 'reference', 'candidate')
                    for metric, value in (
                        ('range_ratio', 2.0),
                        ('range_abs_error', 5.0),
                        ('pairwise_delta_mae', 5.0),
                        ('pairwise_delta_rmse', 5.0),
                    )
                },
                'event_id': '2021-b',
            },
        ]
        summary = _field_summary(field_records)
        self.assertEqual(summary['fields_ge5'], 3)
        self.assertEqual(summary['one_station_fields_ge5'], 3)
        self.assertAlmostEqual(
            summary['one_station']['candidate']['pairwise_delta_mae'], 5.0
        )
        for index, record in enumerate(field_records):
            record['reference_pairwise_delta_mae'] = float(index)
            record['candidate_pairwise_delta_mae'] = float(index) - 0.1
        ci = _paired_field_ci(
            field_records,
            'reference_pairwise_delta_mae',
            'candidate_pairwise_delta_mae',
            seed=20260915,
            draws=100,
        )
        self.assertEqual(ci['events'], 2)
        self.assertEqual(ci['fields'], 3)
        self.assertAlmostEqual(ci['delta'], -0.1)

    def test_complete_tiny_rt61_train_and_validation_branch(self):
        from tests.test_rt59_dual_objective_transport import (
            _OneBatchLoader,
            _TinyEncoder,
            _TinyStationAdapter,
            _tiny_inputs,
            _tiny_model,
        )
        with mock.patch.object(
            models,
            'get_diting_model',
            return_value=nn.Sequential(_TinyEncoder(), _TinyStationAdapter()),
        ):
            model = _tiny_model(rt59=True)
        train_light.initialize_rt59_from_rt58(model)
        transport = model.pga_anchor_residual_transport_head.transport
        transport.enable_rt60_reference = True
        transport.use_rt61_wave_geometry_adapter = True
        transport.rt61_wave_geometry_adapter = models.RT61WaveGeometryAdapter(
            wave_dim=2 * transport.station_dim + transport.emb_dim + transport.hidden_dim,
            geometry_dim=10,
            hidden_dim=transport.hidden_dim,
            rank=16,
        )
        transport.set_rt60_reference_state({
            key: value.detach().clone()
            for key, value in transport.residual_head.state_dict().items()
        })
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in transport.residual_head.parameters():
            parameter.requires_grad = True
        for parameter in transport.rt61_wave_geometry_adapter.parameters():
            parameter.requires_grad = True
        optimizer, _ = train_light.build_optimizer_with_groups(model, {
            'lr': 1e-4, 'lr_rt61_trainable': 1e-4, 'optimizer': 'adam',
        })
        inputs = list(_tiny_inputs(observed=True))
        labels = [
            torch.zeros(2, 1, 1),
            torch.zeros(2, 1, 3),
            torch.tensor([[[[-1.0]], [[-0.5]]], [[[-1.2]], [[-0.8]]]]),
        ]
        picks = {
            'shifted': torch.tensor([[100., 200., 300.], [150., 0., 0.]]),
            'raw': torch.tensor([[100., 200., 300.], [150., 0., 0.]]),
            'shift': torch.zeros(2),
            'causal_random_mask_applied': torch.tensor([False, False]),
        }
        loader = _OneBatchLoader((inputs, labels, picks))
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            train_light, 'training_params', {'weight_path': tmpdir}, create=True,
        ), mock.patch.object(
            train_light, 'SummaryWriter', return_value=mock.MagicMock(),
        ), mock.patch.object(
            train_light, 'save_model_checkpoint',
        ) as save_checkpoint, mock.patch.object(
            train_light, 'export_scalar_history',
            return_value=(tmpdir, os.path.join(tmpdir, 'manifest.json')),
        ):
            train_light.train_model(
                model, loader, loader, optimizer, scheduler=None,
                num_epochs=1, save_name='rt61-tiny-loop',
                res_comps=['pga'], res_weight=[1.0], loss_type='mdn',
                pga_target_normalization={
                    'enabled': True, 'mean': -1.0, 'std': 0.5,
                },
                rt61_contrast_objective_cfg={'enabled': True},
                rt61_fixed_lr_schedule={
                    'completed_epoch_boundaries': [0, 4, 6],
                    'lr_values': [1e-4, 5e-5, 2.5e-5],
                },
                freeze_mode='rt61_wave_geometry_residual_only',
            )
        self.assertTrue(save_checkpoint.called)


if __name__ == '__main__':
    unittest.main()
