import hashlib
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint
import train_light
from tools.analyze_rt59_dual_objective_npz import _event_ids, _field_metrics
from tools.rt60_contrast_objective import (
    build_reference_payload,
    field_bucket_counts,
    field_contrast_values,
    local_field_objective,
    readout_state_dict,
    restore_reference_payload,
    rt60_contrast_objective,
    tensor_mapping_sha256,
)

models = train_light.models


class _HeadWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform_model = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
        self.pga_anchor_residual_transport_head = models.PGAAnchorResidualTransportHead(
            station_dim=4,
            emb_dim=6,
            local_hidden_dim=8,
            transport_hidden_dim=256,
            set_layers=1,
            heads=2,
            enable_rt60_reference=True,
        )


def _head_inputs(query_coords):
    torch.manual_seed(60)
    station_coords = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    return (
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 4),
        torch.randn(1, 6),
        torch.randn(1, query_coords.shape[1], 6),
        station_coords,
        query_coords,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, query_coords.shape[1], dtype=torch.bool),
        torch.randn(1, 2),
        torch.randn(1, 2, 1),
        torch.ones(1, 2, 1),
        torch.randn(1, query_coords.shape[1], 1),
    )


class _OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = self

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


class RT60ContrastReadoutTests(unittest.TestCase):
    def test_legacy_files_unchanged_and_new_config_contract(self):
        expected = {
            'pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json': 'bb28ce3b66a6bd389e6ccd2cb53062c1e68cdb603535f9930c75ac99a33d1c8c',
            'pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json': '92fe9f0f0942ae7c7e75ae4351c9cd90cfd274dcbbbfd474b5a6648994fb8015',
            'pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json': '0b4cb2f0b1ebec6d81c59a64022bf688893a17ded3260a834633e03128dd9f47',
            'pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json': '7108b4734752307cdeb6b14a68edb579f055a2a3a448edaae836707afdef2630',
            'pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json': '923469e285a5f1bc3b2380b0bb5b3da033f25a76810b47e9ce6bdd1fc6d53491',
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(), digest
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
            'RT59_PARENT_CHECKPOINT_SHA256': 'a' * 64,
            'RT60_WEIGHT_PATH': '/tmp/rt60',
            'RT60_SOURCE_MANIFEST_SHA256': 'b' * 64,
        }
        path = REPO_ROOT / 'pga_configs' / 'transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_chaosuan.json'
        with mock.patch.dict(os.environ, env, clear=False):
            config = train_light.load_config_file(str(path))
        self.assertTrue(config['model_params']['use_rt60_contrast_readout'])
        training = config['training_params']
        self.assertEqual(training['freeze_mode'], 'rt60_contrast_readout_only')
        self.assertFalse(training['rt59_dual_objective']['enabled'])
        self.assertEqual(training['rt60_fixed_lr_schedule']['lr_values'], [1e-4, 5e-5, 2.5e-5])

    def test_phase_a_uses_p95_p05_not_ptp_and_event_id_fails_closed(self):
        truth = np.array([[0., 1., 2., 3., 100.]])
        pred = np.array([[0., 1., 2., 3., 4.]])
        mask = np.ones_like(truth, dtype=bool)
        result = _field_metrics(truth, pred, mask, np.array([1]))
        canonical_truth = np.percentile(truth[0], 95) - np.percentile(truth[0], 5)
        canonical_pred = np.percentile(pred[0], 95) - np.percentile(pred[0], 5)
        self.assertAlmostEqual(result['range_ratio_mean'], canonical_pred / canonical_truth)
        self.assertAlmostEqual(result['ptp_range_ratio_mean_diagnostic'], 4.0 / 100.0)
        self.assertNotAlmostEqual(
            result['range_ratio_mean'], result['ptp_range_ratio_mean_diagnostic']
        )
        with self.assertRaises(KeyError):
            _event_ids({}, 1)

    def test_contrast_identity_translation_mask_and_ddp_algebra(self):
        prediction = torch.tensor([[1., 3., 8., float('nan')]], dtype=torch.float64)
        target = torch.tensor([[0., 2., 4., float('nan')]], dtype=torch.float64)
        groups = torch.tensor([[0, 0, 0, -1]])
        values, buckets = field_contrast_values(
            prediction, target, groups, torch.tensor([1])
        )
        error = torch.tensor([1., 1., 4.], dtype=torch.float64)
        pairs = torch.triu_indices(3, 3, 1)
        pair_form = 0.5 * (
            error[pairs[0]] - error[pairs[1]]
        ).square().mean()
        torch.testing.assert_close(values[0], pair_form)
        shifted, _ = field_contrast_values(
            prediction + torch.tensor([[7., 7., 7., 0.]], dtype=torch.float64),
            target,
            groups,
            torch.tensor([1]),
        )
        torch.testing.assert_close(values, shifted)
        theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
        y = torch.tensor([-.2, .5, .1, -.4], dtype=torch.float64)
        code = torch.tensor([0, 0, 2, 3])
        counts = field_bucket_counts(code)
        whole = local_field_objective((theta * x - y).square(), code, counts, 1)
        whole.backward()
        expected = theta.grad.clone()
        gradients = []
        for indices in (torch.tensor([0]), torch.tensor([1, 2, 3])):
            shard_theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
            loss = local_field_objective(
                (shard_theta * x[indices] - y[indices]).square(),
                code[indices], counts, 2,
            )
            loss.backward()
            gradients.append(shard_theta.grad)
        torch.testing.assert_close(torch.stack(gradients).mean(), expected)

    def test_reference_same_forward_routing_and_state_dict_compatibility(self):
        disabled = models.PGAAnchorResidualTransportHead(
            4, 6, transport_hidden_dim=256, set_layers=1, heads=2
        )
        enabled = models.PGAAnchorResidualTransportHead(
            4, 6, transport_hidden_dim=256, set_layers=1, heads=2,
            enable_rt60_reference=True,
        )
        enabled.load_state_dict(disabled.state_dict(), strict=True)
        self.assertEqual(tuple(disabled.state_dict()), tuple(enabled.state_dict()))
        reference = {
            key: value.detach().clone()
            for key, value in enabled.transport.residual_head.state_dict().items()
        }
        enabled.transport.set_rt60_reference_state(reference)
        query = torch.tensor([[[0., 0., 0.], [3., 0., 0.]]])
        first = enabled(*_head_inputs(query))
        torch.testing.assert_close(
            first['applied_delta'], first['reference_applied_delta'], rtol=0, atol=0
        )
        weights = first['weights'].detach().clone()
        level = first['level'].detach().clone()
        with torch.no_grad():
            enabled.transport.residual_head[-1].bias.add_(0.5)
        second = enabled(*_head_inputs(query))
        self.assertEqual(second['rt60_increment'][0, 0].item(), 0.0)
        torch.testing.assert_close(
            second['reference_applied_delta'] + second['rt60_increment'],
            second['applied_delta'],
        )
        torch.testing.assert_close(weights, second['weights'])
        torch.testing.assert_close(level, second['level'])
        self.assertFalse(torch.equal(
            second['applied_delta'][0, 1], second['reference_applied_delta'][0, 1]
        ))

    def test_exact_trainability_optimizer_step_and_reference_resume(self):
        model = _HeadWrapper()
        model.pga_anchor_residual_transport_head._warm_copy_complete.fill_(True)
        parent = {
            'epoch': 8,
            'model_state_dict': model.state_dict(),
            'checkpoint_format': 'full_v1',
        }
        payload = build_reference_payload(
            model, parent, 'a' * 64,
            {'enabled': True, 'mean': -1.0, 'std': 0.5},
            {'manifest': 'b' * 64}, 'rt59-v3', 8,
        )
        object.__setattr__(model, '_rt60_reference_identity', payload)
        object.__setattr__(model, '_rt60_task_id', '20260920-rt60-final-contrast-readout')
        teacher_hash = payload['readout_sha256']
        train_light.apply_full_model_trainability(
            model, {'freeze_mode': 'rt60_contrast_readout_only'}
        )
        train_light.set_rt60_readout_only_train_mode(model)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertEqual(len(trainable), 6)
        self.assertEqual(
            sum(dict(model.named_parameters())[name].numel() for name in trainable),
            67077,
        )
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer, groups = train_light.build_optimizer_with_groups(model, {
            'lr': 1e-4, 'lr_rt60_readout': 1e-4, 'optimizer': 'adam',
        })
        self.assertEqual(groups['rt60_readout']['n_params'], 67077)
        loss = sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.pga_anchor_residual_transport_head.transport.residual_head.parameters(), 1.0
        )
        optimizer.step()
        changed = [
            key for key, value in model.state_dict().items()
            if not torch.equal(value, before[key])
        ]
        self.assertTrue(changed)
        self.assertTrue(all(
            key.startswith('pga_anchor_residual_transport_head.transport.residual_head.')
            for key in changed
        ))
        self.assertEqual(
            tensor_mapping_sha256(
                model.pga_anchor_residual_transport_head.transport.export_rt60_reference_state()
            ),
            teacher_hash,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rt60.pth')
            train_light.save_model_checkpoint(
                path, model, epoch=1, optimizer=optimizer,
                training_params={'checkpoint': {'save_optimizer_state': True}},
                extra={'rt60_schedule_completed_epochs': 1},
            )
            restored = _HeadWrapper()
            restored.pga_anchor_residual_transport_head._warm_copy_complete.fill_(True)
            train_light.apply_full_model_trainability(
                restored, {'freeze_mode': 'rt60_contrast_readout_only'}
            )
            restored_optimizer, _ = train_light.build_optimizer_with_groups(restored, {
                'lr': 1e-4, 'lr_rt60_readout': 1e-4, 'optimizer': 'adam',
            })
            train_light.load_checkpoint(
                restored, restored_optimizer, None, path, torch.device('cpu')
            )
            restored_teacher = (
                restored.pga_anchor_residual_transport_head.transport
                .export_rt60_reference_state()
            )
            self.assertEqual(tensor_mapping_sha256(restored_teacher), teacher_hash)
            student_hash = tensor_mapping_sha256(readout_state_dict(restored))
            self.assertNotEqual(student_hash, teacher_hash)

    def test_objective_and_conditional_eval_export(self):
        model = _HeadWrapper()
        model._last_rt59_route_observed = torch.tensor([[False, False, False]])
        model._last_station_valid = torch.tensor([[True, False]])
        reference = torch.zeros(1, 3, 1, 3)
        reference[..., 2] = 1.0
        model._last_rt60_reference_mdn = reference
        final = reference.clone()
        final[..., 1] = torch.tensor([[[0.1], [0.3], [-0.2]]])
        final.requires_grad_()
        labels = [torch.tensor([[[[0.0]], [[0.1]], [[-0.1]]]])]
        total, stats = rt60_contrast_objective(
            model, [final], labels, ['pga'],
            torch.ones(1, 3, dtype=torch.bool),
            {'causal_random_mask_applied': torch.tensor([True])},
            {'enabled': True},
        )
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(stats['target_counts'].tolist(), [3, 0, 0])
        self.assertEqual(stats['field_counts'].tolist(), [1, 0, 0, 0])
        total.backward()
        self.assertTrue(torch.isfinite(final.grad).all())

        raw = type('Raw', (), {})()
        for name in (
            '_last_pga_temporal_base', '_last_pga_temporal_delta',
            '_last_pga_temporal_pred', '_last_pga_temporal_final',
            '_last_station_distinctive_local_residual_pred',
            '_last_station_distinctive_local_absolute_pred', '_last_pga_anchor_pred',
            '_last_pga_anchor_transfer', '_last_pga_anchor_candidate',
            '_last_pga_anchor_station_weights', '_last_pga_anchor_field_mean',
            '_last_pga_anchor_applied_delta', '_last_rt59_base_mdn',
            '_last_rt59_base_mean', '_last_rt59_input_base_mean',
            '_last_rt59_input_base_sigma', '_last_rt59_anchor',
            '_last_rt59_local_delta', '_last_rt59_level',
            '_last_rt59_relative_transfer', '_last_rt59_candidate',
            '_last_rt59_station_weights', '_last_rt59_applied_delta',
            '_last_rt59_route_observed', '_last_rt59_route_ambiguous',
            '_last_rt59_fixed_context_rolled_delta',
        ):
            setattr(raw, name, None)
        raw._last_rt60_reference_mdn = torch.tensor([[[[0., 0., 1.]]]])
        raw._last_rt60_reference_mean = torch.tensor([[[0.]]])
        raw._last_rt60_increment = torch.tensor([[2.]])
        results = defaultdict(list)
        config = {'training_params': {'pga_target_normalization': {
            'enabled': True, 'mean': -1.0, 'std': 0.5,
        }}}
        eval_checkpoint.append_pga_temporal_residual_outputs(results, raw, config)
        self.assertEqual(results['rt60_reference_mean'][0].item(), -1.0)
        self.assertEqual(results['rt60_increment'][0].item(), 1.0)

    def test_complete_tiny_train_and_validation_branch(self):
        from tests.test_rt59_dual_objective_transport import (
            _TinyEncoder, _TinyStationAdapter, _tiny_inputs, _tiny_model,
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
        transport.set_rt60_reference_state({
            key: value.detach().clone()
            for key, value in transport.residual_head.state_dict().items()
        })
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in transport.residual_head.parameters():
            parameter.requires_grad = True
        optimizer, _ = train_light.build_optimizer_with_groups(model, {
            'lr': 1e-4, 'lr_rt60_readout': 1e-4, 'optimizer': 'adam',
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
                num_epochs=1, save_name='rt60-tiny-loop',
                res_comps=['pga'], res_weight=[1.0], loss_type='mdn',
                pga_target_normalization={
                    'enabled': True, 'mean': -1.0, 'std': 0.5,
                },
                rt60_contrast_objective_cfg={'enabled': True},
                rt60_fixed_lr_schedule={
                    'completed_epoch_boundaries': [0, 4, 6],
                    'lr_values': [1e-4, 5e-5, 2.5e-5],
                },
                freeze_mode='rt60_contrast_readout_only',
            )
        self.assertTrue(save_checkpoint.called)


if __name__ == '__main__':
    unittest.main()
