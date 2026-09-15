import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint
import train_light
from tools.rt59_dual_objective import (
    group_codes,
    group_counts,
    local_group_objective,
    point_losses,
    route_correction,
    shift_mdn,
    unique_observed_match,
)

models = train_light.models


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 16)

    def forward(self, waveform):
        return self.projection(waveform.transpose(1, 2))


class _TinyStationAdapter(nn.Module):
    encoder_dim = 16
    uses_station_metadata = False

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 20)
        self.forward_calls = 0

    def forward(self, tokens, token_mask=None, **_kwargs):
        self.forward_calls += 1
        if token_mask is None:
            return self.projection(tokens.mean(dim=1))
        mask = models.AttentionPool1d._coerce_token_mask(
            token_mask, tokens.shape[1], tokens.device
        ).to(tokens.dtype)
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return self.projection(pooled)

    @staticmethod
    def temporal_tokens(tokens):
        return tokens


def _tiny_model(rt59=True):
    return models.build_transformer_model(
        max_stations=3,
        waveform_model_dims=(20,),
        output_mlp_dims=(8,),
        output_location_dims=(8,),
        mad_params={'n_heads': 2, 'att_dropout': 0.0, 'initializer_range': 0.02},
        ffn_params={'hidden_dim': 40},
        transformer_layers=1,
        n_pga_targets=2,
        pga_mixture=2,
        location_mixture=2,
        magnitude_mixture=1,
        pga_readout_mode='target_cross_attention',
        event_readout_mode='event_cross_attention',
        station_context_mode='off',
        readout_n_heads=2,
        output_distribution='mdn',
        use_amplitude_info=False,
        use_vs30=False,
        dataset_bias=False,
        use_pga_temporal_residual=True,
        pga_temporal_residual_zero_init=False,
        pga_temporal_residual_scale=1.66,
        station_distinctive_adapter=True,
        pga_temporal_residual_station_weighting='learned_pair',
        pga_temporal_residual_use_event_context=True,
        temporal_token_dim=16,
        temporal_pool_dim=8,
        temporal_pool_geom_hidden_dim=6,
        pga_temporal_residual_station_distinctive_dim=10,
        pga_temporal_residual_station_distinctive_hidden_dim=20,
        pga_temporal_residual_station_distinctive_amplitude_feature_dim=11,
        anchor_transfer_station_dim=10,
        anchor_transfer_hidden_dim=12,
        anchor_transfer_set_layers=2,
        anchor_transfer_heads=3,
        use_pga_anchor_transfer=True,
        use_pga_anchor_residual_transport=rt59,
        anchor_residual_local_hidden_dim=8,
        anchor_residual_transport_hidden_dim=12,
        anchor_residual_transport_set_layers=2,
        anchor_residual_transport_heads=3,
    )


def _tiny_inputs(observed=True):
    torch.manual_seed(3)
    waveforms = torch.randn(2, 3, 3, 20)
    station_coords = torch.randn(2, 3, 3)
    station_valid = torch.tensor([[True, True, True], [True, False, False]])
    query_coords = torch.randn(2, 2, 3)
    if observed:
        query_coords[:, 0] = station_coords[:, 0]
    query_valid = torch.ones(2, 2, dtype=torch.bool)
    sample_mask = torch.ones(2, 3, 20, dtype=torch.bool)
    sample_mask[1, 1:] = False
    return (
        waveforms,
        station_coords,
        station_valid,
        query_coords,
        query_valid,
        sample_mask,
    )


class RT59DualObjectiveTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260915)
        self.xs = torch.tensor([[[0., 0., 0.], [1., 2., 0.], [9., 9., 0.]]])
        self.xq = torch.tensor([[[0., 0., 0.], [0.1, 0., 0.], [1., 2., 0.]]])
        self.sv = torch.tensor([[True, True, False]])
        self.qv = torch.ones((1, 3), dtype=torch.bool)

    def test_legacy_config_hashes_and_rt59_protocol(self):
        expected = {
            'pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json': 'bb28ce3b66a6bd389e6ccd2cb53062c1e68cdb603535f9930c75ac99a33d1c8c',
            'pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json': '92fe9f0f0942ae7c7e75ae4351c9cd90cfd274dcbbbfd474b5a6648994fb8015',
            'pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json': '0b4cb2f0b1ebec6d81c59a64022bf688893a17ded3260a834633e03128dd9f47',
            'pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json': '7108b4734752307cdeb6b14a68edb579f055a2a3a448edaae836707afdef2630',
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(), digest)
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
            'RT59_WEIGHT_PATH': '/tmp/rt59',
        }
        config_path = REPO_ROOT / 'pga_configs' / 'transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json'
        with mock.patch.dict(os.environ, env, clear=False):
            config = train_light.load_config_file(str(config_path))
        self.assertTrue(config['model_params']['use_pga_anchor_residual_transport'])
        self.assertEqual(config['training_params']['freeze_mode'], 'dual_anchor_residual_only')
        self.assertEqual(config['training_params']['epochs_full_model'], 8)
        self.assertEqual(config['training_params']['rt59_dual_objective']['group_weights'], [0.5, 0.25, 0.25])
        self.assertFalse(config['training_params']['normal_replay_distillation_loss']['enabled'])

    def test_unique_route_duplicate_padding_and_permutation(self):
        match, observed, ambiguous = unique_observed_match(self.xs, self.xq, self.sv, self.qv)
        self.assertEqual(observed.tolist(), [[True, False, True]])
        self.assertFalse(ambiguous.any())
        local = torch.tensor([[1., 3., 8.]])
        remote = torch.tensor([[4., 5., 6.]])
        baseline = route_correction(local, remote, match, observed, self.qv)
        permutation = torch.tensor([1, 2, 0])
        p_match, p_observed, _ = unique_observed_match(
            self.xs[:, permutation], self.xq, self.sv[:, permutation], self.qv
        )
        permuted = route_correction(
            local[:, permutation], remote, p_match, p_observed, self.qv
        )
        torch.testing.assert_close(baseline, permuted, rtol=0, atol=0)
        duplicate = self.xs.clone()
        duplicate[:, 1] = duplicate[:, 0]
        _, duplicate_observed, duplicate_ambiguous = unique_observed_match(
            duplicate, self.xq, self.sv, self.qv
        )
        self.assertFalse(duplicate_observed[0, 0])
        self.assertTrue(duplicate_ambiguous[0, 0])
        padded_nan = self.xs.clone()
        padded_nan[:, 2] = float('nan')
        unique_observed_match(padded_nan, self.xq, self.sv, self.qv)
        padded_nan[:, 0] = float('nan')
        with self.assertRaises(ValueError):
            unique_observed_match(padded_nan, self.xq, self.sv, self.qv)

    def test_only_mdn_means_shift_and_variance_is_preserved(self):
        mixture = torch.randn(1, 3, 4, 3, dtype=torch.float64)
        mixture[..., 2] = mixture[..., 2].abs() + 0.2
        delta = torch.randn(1, 3, dtype=torch.float64)
        shifted = shift_mdn(mixture, delta)
        self.assertTrue(torch.equal(mixture[..., 0], shifted[..., 0]))
        self.assertTrue(torch.equal(mixture[..., 2], shifted[..., 2]))
        def stats(value):
            weight = value[..., 0].softmax(-1)
            mean = (weight * value[..., 1]).sum(-1)
            var = (weight * (value[..., 2].square() + (value[..., 1] - mean[..., None]).square())).sum(-1)
            return mean, var
        old_mean, old_var = stats(mixture)
        new_mean, new_var = stats(shifted)
        torch.testing.assert_close(new_mean, old_mean + delta)
        torch.testing.assert_close(new_var, old_var)

    def test_group_loss_regret_invalid_mask_and_ddp_algebra(self):
        base = torch.zeros(1, 3, 1, 3, dtype=torch.float64)
        base[..., 2] = 1.0
        final = shift_mdn(base, torch.tensor([[1., -1., 1.]], dtype=torch.float64))
        target = torch.ones(1, 3, dtype=torch.float64)
        groups = torch.tensor([[1, 2, 0]])
        losses = point_losses(base, final, target, groups)
        self.assertEqual(losses['regret'][0, 0].item(), 0.0)
        self.assertGreater(losses['regret'][0, 1].item(), 0.0)
        self.assertEqual(losses['regret'][0, 2].item(), 0.0)
        invalid_base = base.clone()
        invalid_final = final.clone().requires_grad_()
        invalid_target = target.clone()
        invalid_groups = torch.tensor([[0, 1, -1]])
        invalid_base[:, 2] = float('nan')
        invalid_target[:, 2] = float('nan')
        with torch.no_grad():
            invalid_final[:, 2] = float('nan')
        value = point_losses(invalid_base, invalid_final, invalid_target, invalid_groups)['main']
        objective = local_group_objective(value, invalid_groups, group_counts(invalid_groups))
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        self.assertTrue(torch.isfinite(invalid_final.grad).all())

        x = torch.tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
        y = torch.tensor([.3, .2, -.1, .8, 1., -.4, .2], dtype=torch.float64)
        code = torch.tensor([0, 0, 0, 0, 1, 2, 2])
        theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        counts = group_counts(code)
        whole = local_group_objective((theta * x - y).square(), code, counts)
        whole.backward()
        expected_grad = theta.grad.clone()
        shard_grads = []
        for indices in (torch.tensor([0, 1, 4]), torch.tensor([2, 3, 5, 6])):
            shard_theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
            shard = local_group_objective(
                (shard_theta * x[indices] - y[indices]).square(),
                code[indices],
                counts,
                world_size=2,
            )
            shard.backward()
            shard_grads.append(shard_theta.grad)
        torch.testing.assert_close(torch.stack(shard_grads).mean(), expected_grad)

    def test_tiny_full_model_cached_decode_zero_init_and_no_encoder_rerun(self):
        with mock.patch.object(models, 'get_diting_model', return_value=nn.Sequential(_TinyEncoder(), _TinyStationAdapter())):
            model = _tiny_model(rt59=True)
        train_light.initialize_rt59_from_rt58(model)
        train_light.apply_full_model_trainability(
            model, {'freeze_mode': 'dual_anchor_residual_only'}
        )
        train_light.set_rt59_only_train_mode(model)
        inputs = _tiny_inputs(observed=True)
        adapter = model.waveform_model[1]
        outputs = model(*inputs)
        self.assertEqual(adapter.forward_calls, inputs[0].shape[1])
        self.assertEqual(model.pga_cross_attention._last_layer_outputs[-1].shape[1], 2)
        self.assertEqual(model.pga_temporal_residual_head._last_delta.shape[1], 2)
        pga = outputs[model.output_layout.index('pga')]
        self.assertTrue(torch.equal(pga, model._last_rt59_base_mdn))
        self.assertTrue(torch.equal(
            model._last_rt59_applied_delta,
            torch.zeros_like(model._last_rt59_applied_delta),
        ))
        self.assertTrue(model._last_rt59_route_observed[:, 0].all())
        torch.testing.assert_close(
            model._last_rt59_base_mean[:, 0],
            model._last_rt59_input_base_mean[:, 0],
            rtol=1e-6,
            atol=1e-6,
        )
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('pga_anchor_residual_transport_head.') for name in trainable))
        local_ids = {id(value) for value in model.pga_anchor_residual_transport_head.local.parameters()}
        transport_ids = {id(value) for value in model.pga_anchor_residual_transport_head.transport.parameters()}
        self.assertFalse(local_ids & transport_ids)
        labels = [
            torch.zeros(2, 1, 1),
            torch.zeros(2, 1, 3),
            torch.tensor([[[[-1.0]], [[-0.5]]], [[[-1.2]], [[-0.8]]]]),
        ]
        p_picks = {
            'causal_random_mask_applied': torch.tensor([False, False]),
            'input_pga_values': torch.tensor([
                [-1.0, -0.8, -0.6], [-1.2, 0.0, 0.0],
            ]),
            'input_pga_valid': inputs[2],
        }
        objective, stats = train_light.rt59_grouped_objective(
            model,
            outputs,
            labels,
            model.output_layout,
            inputs[4],
            p_picks,
            {
                'enabled': True,
                'group_weights': [0.5, 0.25, 0.25],
                'relative_weight': 0.2,
                'candidate_weight': 0.05,
                'difference_weight': 0.4,
            },
            pga_target_normalization={'enabled': True, 'mean': -1.0, 'std': 0.5},
        )
        self.assertTrue(torch.isfinite(objective))
        self.assertEqual(stats['target_counts'].tolist(), [0, 2, 2])
        objective.backward()
        self.assertGreater(
            sum(
                (parameter.grad.abs().sum() for parameter in model.pga_anchor_residual_transport_head.parameters()
                 if parameter.grad is not None),
                torch.tensor(0.0),
            ).item(),
            0.0,
        )

    def test_checkpoint_warm_copy_and_resume_identity(self):
        def make_model(rt59):
            with mock.patch.object(models, 'get_diting_model', return_value=nn.Sequential(_TinyEncoder(), _TinyStationAdapter())):
                return _tiny_model(rt59=rt59)
        rt58 = make_model(False)
        rt59 = make_model(True)
        missing, unexpected = train_light.load_model_state_dict_compatible(
            rt59,
            rt58.state_dict(),
            strict=True,
            context='tiny RT58',
            allowed_missing_prefixes=('pga_anchor_residual_transport_head.',),
        )
        self.assertTrue(missing)
        self.assertFalse(unexpected)
        train_light.initialize_rt59_from_rt58(rt59)
        for name in ('set_input', 'set_blocks', 'pair_encoder', 'score_head'):
            old_state = getattr(rt58.pga_anchor_transfer_head, name).state_dict()
            new_state = getattr(rt59.pga_anchor_residual_transport_head.transport, name).state_dict()
            for key in old_state:
                self.assertTrue(torch.equal(old_state[key], new_state[key]))
        resumed = make_model(True)
        resumed.load_state_dict(rt59.state_dict(), strict=True)
        self.assertTrue(resumed.pga_anchor_residual_transport_head._warm_copy_complete.item())
        with self.assertRaises(RuntimeError):
            train_light.initialize_rt59_from_rt58(resumed)

    def test_branch_routing_gradient_isolation(self):
        head = models.PGAAnchorResidualTransportHead(
            station_dim=4,
            emb_dim=6,
            local_hidden_dim=8,
            transport_hidden_dim=8,
            set_layers=1,
            heads=2,
        )
        station_u = torch.randn(1, 2, 4)
        station_d = torch.randn(1, 2, 4)
        event = torch.randn(1, 6)
        query_emb = torch.randn(1, 1, 6)
        station_coords = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        station_valid = torch.ones(1, 2, dtype=torch.bool)
        query_valid = torch.ones(1, 1, dtype=torch.bool)
        anchor = torch.randn(1, 2)
        input_base = torch.randn(1, 2, 1)
        input_sigma = torch.ones(1, 2, 1)
        public_base = torch.randn(1, 1, 1)
        common = (station_u, station_d, event, query_emb, station_coords)
        observed = head(
            *common, station_coords[:, :1], station_valid, query_valid,
            anchor, input_base, input_sigma, public_base,
        )['applied_delta']
        (observed - 1).square().sum().backward()
        local_grad = sum((p.grad.abs().sum() for p in head.local.parameters() if p.grad is not None), torch.tensor(0.))
        transport_grad = sum((p.grad.abs().sum() for p in head.transport.parameters() if p.grad is not None), torch.tensor(0.))
        self.assertGreater(local_grad.item(), 0.0)
        self.assertEqual(transport_grad.item(), 0.0)
        head.zero_grad(set_to_none=True)
        remote_coords = torch.tensor([[[3., 0., 0.]]])
        remote = head(
            *common, remote_coords, station_valid, query_valid,
            anchor, input_base, input_sigma, public_base,
        )['applied_delta']
        (remote - 1).square().sum().backward()
        local_grad = sum((p.grad.abs().sum() for p in head.local.parameters() if p.grad is not None), torch.tensor(0.))
        transport_grad = sum((p.grad.abs().sum() for p in head.transport.parameters() if p.grad is not None), torch.tensor(0.))
        self.assertEqual(local_grad.item(), 0.0)
        self.assertGreater(transport_grad.item(), 0.0)

    def test_eval_export_is_conditional_and_scaled(self):
        raw = type('Raw', (), {})()
        raw._last_rt59_base_mdn = torch.tensor([[[[0., 0., 1.]]]])
        raw._last_rt59_base_mean = torch.tensor([[[0.]]])
        raw._last_rt59_input_base_mean = torch.tensor([[[0.]]])
        raw._last_rt59_input_base_sigma = torch.tensor([[[1.]]])
        raw._last_rt59_anchor = torch.tensor([[0.]])
        raw._last_rt59_local_delta = torch.tensor([[1.]])
        raw._last_rt59_level = torch.tensor([[0.5]])
        raw._last_rt59_relative_transfer = torch.tensor([[[1.]]])
        raw._last_rt59_candidate = torch.tensor([[[0.]]])
        raw._last_rt59_station_weights = torch.tensor([[[1.]]])
        raw._last_rt59_applied_delta = torch.tensor([[1.]])
        raw._last_rt59_route_observed = torch.tensor([[True]])
        raw._last_rt59_route_ambiguous = torch.tensor([[False]])
        raw._last_rt59_fixed_context_rolled_delta = torch.tensor([[2.]])
        for name in (
            '_last_pga_temporal_base', '_last_pga_temporal_delta',
            '_last_pga_temporal_pred', '_last_pga_temporal_final',
            '_last_station_distinctive_local_residual_pred',
            '_last_station_distinctive_local_absolute_pred', '_last_pga_anchor_pred',
            '_last_pga_anchor_transfer', '_last_pga_anchor_candidate',
            '_last_pga_anchor_station_weights', '_last_pga_anchor_field_mean',
            '_last_pga_anchor_applied_delta',
        ):
            setattr(raw, name, None)
        from collections import defaultdict
        results = defaultdict(list)
        config = {'training_params': {'pga_target_normalization': {
            'enabled': True, 'mean': -1.0, 'std': 0.5,
        }}}
        eval_checkpoint.append_pga_temporal_residual_outputs(results, raw, config)
        self.assertEqual(results['rt59_base_mean'][0].item(), -1.0)
        self.assertEqual(results['rt59_local_delta'][0].item(), 0.5)
        self.assertEqual(results['rt59_level'][0].item(), 0.5)
        self.assertEqual(results['rt59_fixed_context_rolled_delta'][0].item(), 1.0)

    def test_random_group_cannot_be_observed(self):
        observed = torch.tensor([[False, False], [True, False]])
        codes = group_codes(torch.ones_like(observed), observed, torch.tensor([True, False]))
        self.assertEqual(codes.tolist(), [[0, 0], [1, 2]])
        with self.assertRaises(ValueError):
            group_codes(torch.ones_like(observed), observed, torch.tensor([True, True]))


if __name__ == '__main__':
    unittest.main()
