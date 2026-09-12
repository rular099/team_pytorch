import hashlib
import inspect
import os
import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint
import train_light


models = train_light.models
RT58_CONFIG = (
    REPO_ROOT
    / 'pga_configs'
    / 'transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json'
)


def _head(zero_init=False):
    torch.manual_seed(20260912)
    return models.PGAAnchorTransferHead(
        station_dim=6,
        emb_dim=8,
        hidden_dim=12,
        set_layers=2,
        heads=3,
        distance_scale=0.05,
        zero_init=zero_init,
    )


def _head_inputs(batch=2):
    torch.manual_seed(57)
    stations, queries = 3, 4
    return {
        'station_u': torch.randn(batch, stations, 6),
        'station_d': torch.randn(batch, stations, 6),
        'event_emb': torch.randn(batch, 8),
        'query_emb': torch.randn(batch, queries, 8),
        'station_coords': torch.randn(batch, stations, 3),
        'query_coords': torch.randn(batch, queries, 3),
        'station_valid': torch.tensor(
            [[True, True, True], [True, False, False]], dtype=torch.bool
        )[:batch],
        'query_valid': torch.tensor(
            [[True, True, True, True], [True, True, False, False]], dtype=torch.bool
        )[:batch],
        'frozen_rt57_mean': torch.randn(batch, queries, 1),
    }


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

    def forward(self, tokens, token_mask=None, **_kwargs):
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


def _tiny_model(anchor=None, anchor_zero_init=True):
    common = {
        'max_stations': 3,
        'waveform_model_dims': (20,),
        'output_mlp_dims': (8,),
        'output_location_dims': (8,),
        'mad_params': {
            'n_heads': 2,
            'att_dropout': 0.0,
            'initializer_range': 0.02,
        },
        'ffn_params': {'hidden_dim': 40},
        'transformer_layers': 1,
        'n_pga_targets': 2,
        'pga_mixture': 2,
        'location_mixture': 2,
        'magnitude_mixture': 1,
        'pga_readout_mode': 'target_cross_attention',
        'event_readout_mode': 'event_cross_attention',
        'readout_n_heads': 2,
        'output_distribution': 'mdn',
        'use_amplitude_info': False,
        'use_pga_temporal_residual': True,
        'pga_temporal_residual_zero_init': False,
        'pga_temporal_residual_scale': 1.66,
        'station_distinctive_adapter': True,
        'pga_temporal_residual_station_weighting': 'learned_pair',
        'pga_temporal_residual_use_event_context': True,
        'temporal_token_dim': 16,
        'temporal_pool_dim': 8,
        'temporal_pool_geom_hidden_dim': 6,
        'pga_temporal_residual_station_distinctive_dim': 10,
        'pga_temporal_residual_station_distinctive_hidden_dim': 20,
        'pga_temporal_residual_station_distinctive_amplitude_feature_dim': 11,
        'anchor_transfer_station_dim': 10,
        'anchor_transfer_hidden_dim': 12,
        'anchor_transfer_set_layers': 2,
        'anchor_transfer_heads': 3,
        'anchor_transfer_zero_init': anchor_zero_init,
    }
    if anchor is not None:
        common['use_pga_anchor_transfer'] = bool(anchor)
    return models.build_transformer_model(**common)


def _tiny_inputs():
    torch.manual_seed(3)
    waveforms = torch.randn(2, 3, 3, 20)
    station_coords = torch.randn(2, 3, 3)
    station_valid = torch.tensor([[True, True, True], [True, False, False]])
    query_coords = torch.randn(2, 2, 3)
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


class RT58WaveformAnchorTransferTests(unittest.TestCase):
    def test_config_protocol_and_legacy_hashes(self):
        expected_hashes = {
            'pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json': (
                'bb28ce3b66a6bd389e6ccd2cb53062c1e68cdb603535f9930c75ac99a33d1c8c'
            ),
            'pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json': (
                '92fe9f0f0942ae7c7e75ae4351c9cd90cfd274dcbbbfd474b5a6648994fb8015'
            ),
            'pga_configs/transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json': (
                '0b4cb2f0b1ebec6d81c59a64022bf688893a17ded3260a834633e03128dd9f47'
            ),
        }
        for relative, expected in expected_hashes.items():
            self.assertEqual(
                hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(),
                expected,
            )
        env = {
            'JAPAN_FULL_DATA_ROOT': '/tmp/japan',
            'JAPAN_FULL_WEIGHT_PATH': 'rt55',
            'RT55_EP32_CHECKPOINT': '/tmp/rt55.pth',
            'RT56_WEIGHT_PATH': 'rt56',
            'RT56_BASE_CHECKPOINT': '/tmp/rt56.pth',
            'RT57_WEIGHT_PATH': 'rt57',
            'RT57_BASE_CHECKPOINT': '/tmp/rt57.pth',
            'RT58_WEIGHT_PATH': 'rt58',
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = train_light.load_config_file(str(RT58_CONFIG))
        model_cfg = config['model_params']
        train_cfg = config['training_params']
        self.assertTrue(model_cfg['use_pga_anchor_transfer'])
        self.assertEqual(model_cfg['pga_temporal_residual_scale'], 1.66)
        self.assertEqual(model_cfg['anchor_transfer_set_layers'], 2)
        self.assertEqual(train_cfg['freeze_mode'], 'anchor_transfer_only')
        self.assertEqual(train_cfg['epochs_full_model'], 8)
        self.assertEqual(train_cfg['lr_pga_anchor_transfer'], 5e-4)
        self.assertEqual(
            train_cfg['train_generator_overrides']['causal_random_input_mask']['apply_probability'],
            0.8,
        )
        production_head = models.PGAAnchorTransferHead(
            station_dim=256,
            emb_dim=1000,
            hidden_dim=256,
            set_layers=2,
            heads=4,
            distance_scale=0.05,
            zero_init=True,
        )
        self.assertEqual(sum(1 for _ in production_head.parameters()), 50)
        self.assertEqual(sum(p.numel() for p in production_head.parameters()), 2943790)

    def test_zero_init_and_identical_coordinate_transfer(self):
        head = _head(zero_init=True).eval()
        args = _head_inputs()
        args['query_coords'][:, 0] = args['station_coords'][:, 0]
        with torch.no_grad():
            delta = head(**args)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.allclose(
            head._last_transfer[:, 0, 0],
            torch.zeros_like(head._last_transfer[:, 0, 0]),
            atol=1e-8,
            rtol=0.0,
        ))
        self.assertTrue(torch.isfinite(head._last_station_weights).all())
        self.assertTrue(torch.equal(
            head._last_station_weights[1, :, 1:],
            torch.zeros_like(head._last_station_weights[1, :, 1:]),
        ))

    def test_zero_gate_auxiliaries_reach_internal_branches_on_first_step(self):
        head = _head(zero_init=True).train()
        args = _head_inputs()
        delta = head(**args)
        main = F.mse_loss(delta, torch.ones_like(delta))
        auxiliary = (
            head._last_anchor_pred.square().mean()
            + head._last_candidate.square().mean()
        )
        (main + auxiliary).backward()
        for parameter in (
            head.anchor_head[1].weight,
            head.set_input[1].weight,
            head.set_blocks[0].attn.in_proj_weight,
            head.pair_encoder[1].weight,
            head.transfer_head.weight,
            head.output_gate[-1].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().max().item(), 0.0)

    def test_station_permutation_invariance_and_waveform_identity(self):
        head = _head(zero_init=False).eval()
        args = {key: value[:1].clone() for key, value in _head_inputs().items()}
        args['station_valid'][:] = True
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            baseline = head(**args).clone()
            baseline_field = head._last_field_mean.clone()
            baseline_anchor = head._last_anchor_pred.clone()
            paired = dict(args)
            for key in ('station_u', 'station_d', 'station_coords', 'station_valid'):
                paired[key] = args[key][:, permutation]
            paired_output = head(**paired).clone()
            paired_field = head._last_field_mean.clone()
            paired_anchor = head._last_anchor_pred.clone()
            waveform_only = dict(args)
            waveform_only['station_u'] = args['station_u'][:, permutation]
            waveform_only['station_d'] = args['station_d'][:, permutation]
            waveform_output = head(**waveform_only).clone()
            waveform_candidate = head._last_candidate.clone()
        self.assertTrue(torch.allclose(paired_output, baseline, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(paired_field, baseline_field, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(
            paired_anchor,
            baseline_anchor[:, permutation],
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertGreater((waveform_output - baseline).abs().max().item(), 1e-8)
        self.assertGreater(
            (waveform_candidate - baseline_anchor[:, None, :]).abs().max().item(),
            1e-8,
        )

    def test_one_station_waveform_and_query_geometry_are_effective(self):
        head = _head(zero_init=False).eval()
        args = {key: value[:1].clone() for key, value in _head_inputs().items()}
        args['station_valid'][:] = torch.tensor([True, False, False])
        args['query_emb'][:, 1] = args['query_emb'][:, 0]
        args['query_coords'][:, 0] = args['station_coords'][:, 0]
        args['query_coords'][:, 1] = args['station_coords'][:, 0]
        args['query_coords'][:, 1, 0] += 0.4
        args['frozen_rt57_mean'][:, 1] = args['frozen_rt57_mean'][:, 0]
        with torch.no_grad():
            first = head(**args).clone()
            first_anchor = head._last_anchor_pred.clone()
            first_transfer = head._last_transfer.clone()
            changed = dict(args)
            changed['station_u'] = args['station_u'].clone()
            changed['station_d'] = args['station_d'].clone()
            changed['station_u'][:, 0] += 2.0
            changed['station_d'][:, 0, 0] -= 1.5
            second = head(**changed).clone()
            second_anchor = head._last_anchor_pred.clone()
        self.assertGreater((second_anchor[:, 0] - first_anchor[:, 0]).abs().max().item(), 1e-8)
        self.assertGreater((second - first).abs().max().item(), 1e-8)
        self.assertTrue(torch.allclose(
            first_transfer[:, 0, 0], torch.zeros_like(first_transfer[:, 0, 0]), atol=1e-8
        ))
        self.assertGreater(
            (first_transfer[:, 1, 0] - first_transfer[:, 0, 0]).abs().max().item(),
            1e-8,
        )
        self.assertGreater((first[:, 1] - first[:, 0]).abs().max().item(), 1e-8)

    def test_loaded_rt57_zero_init_rt58_matches_gamma_166_base(self):
        def tiny_diting(*_args, **_kwargs):
            return nn.Sequential(_TinyEncoder(), _TinyStationAdapter())

        with mock.patch.object(models, 'get_diting_model', side_effect=tiny_diting):
            torch.manual_seed(1)
            rt57_gamma = _tiny_model(anchor=False)
            torch.manual_seed(2)
            rt58 = _tiny_model(anchor=True, anchor_zero_init=True)
        train_light.load_model_state_dict_compatible(
            rt58,
            rt57_gamma.state_dict(),
            strict=True,
            context='tiny RT57 epoch-6 checkpoint',
            allowed_missing_prefixes=('pga_anchor_transfer_head.',),
        )
        rt57_gamma.eval()
        rt58.eval()
        with torch.no_grad():
            base_outputs = rt57_gamma(*_tiny_inputs())
            rt58_outputs = rt58(*_tiny_inputs())
        for base, candidate in zip(base_outputs[:3], rt58_outputs[:3]):
            self.assertTrue(torch.equal(base, candidate))
        self.assertTrue(torch.equal(
            rt58._last_pga_anchor_applied_delta,
            torch.zeros_like(rt58._last_pga_anchor_applied_delta),
        ))

    def test_full_model_anchor_losses_backward_with_runtime_shapes(self):
        def tiny_diting(*_args, **_kwargs):
            return nn.Sequential(_TinyEncoder(), _TinyStationAdapter())

        with mock.patch.object(models, 'get_diting_model', side_effect=tiny_diting):
            rt58 = _tiny_model(anchor=True, anchor_zero_init=True)
        train_light.apply_full_model_trainability(
            rt58,
            {'freeze_mode': 'anchor_transfer_only'},
        )
        inputs = _tiny_inputs()
        outputs = rt58(*inputs)
        pga_labels = torch.tensor([
            [[[0.0]], [[0.5]]],
            [[[-0.4]], [[0.2]]],
        ])
        labels = [torch.zeros(2, 1, 1), torch.zeros(2, 1, 3), pga_labels]
        p_picks = {
            'input_pga_values': torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.0, 0.0]]),
            'input_pga_valid': inputs[2],
            'causal_random_mask_applied': torch.tensor([True, False]),
        }
        norm = {'enabled': True, 'mean': -1.0, 'std': 0.5}
        losses = train_light.pga_anchor_transfer_aux_losses(
            rt58,
            outputs,
            labels,
            rt58.output_layout,
            inputs[4],
            p_picks,
            anchor_cfg={'enabled': True, 'weight': 0.2, 'huber_delta_dex': 0.15},
            pair_cfg={'enabled': True, 'weight': 0.05, 'huber_delta_dex': 0.2},
            distillation_cfg={
                'enabled': True,
                'weight': 0.05,
                'apply_to_random': False,
            },
            pga_target_normalization=norm,
        )
        self.assertEqual(set(losses), {
            'pga_anchor_absolute_loss',
            'pga_anchor_pair_candidate_loss',
            'normal_replay_distillation_loss',
        })
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(
            rt58.pga_anchor_transfer_head.pair_encoder[1].weight.grad.abs().max().item(),
            0.0,
        )

    def test_disabled_switch_and_anchor_only_trainability(self):
        def tiny_diting(*_args, **_kwargs):
            return nn.Sequential(_TinyEncoder(), _TinyStationAdapter())

        with mock.patch.object(models, 'get_diting_model', side_effect=tiny_diting):
            torch.manual_seed(4)
            default_rt57 = _tiny_model(anchor=None)
            torch.manual_seed(4)
            explicit_disabled = _tiny_model(anchor=False)
            rt58 = _tiny_model(anchor=True)
        self.assertEqual(default_rt57.state_dict().keys(), explicit_disabled.state_dict().keys())
        for left, right in zip(default_rt57(*_tiny_inputs()), explicit_disabled(*_tiny_inputs())):
            self.assertTrue(torch.equal(left, right))
        train_light.apply_full_model_trainability(
            rt58,
            {'freeze_mode': 'anchor_transfer_only'},
        )
        trainable = [name for name, param in rt58.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('pga_anchor_transfer_head.') for name in trainable))
        self.assertTrue(all(
            not param.requires_grad
            for name, param in rt58.named_parameters()
            if not name.startswith('pga_anchor_transfer_head.')
        ))
        train_light.set_anchor_transfer_only_train_mode(rt58)
        self.assertTrue(rt58.pga_anchor_transfer_head.training)
        self.assertFalse(rt58.pga_temporal_residual_head.training)

    def test_anchor_pair_and_difference_losses_honor_masks_and_dex(self):
        prediction = torch.zeros(2, 3)
        station_valid = torch.tensor([[True, True, True], [True, False, False]])
        dummy = types.SimpleNamespace(
            _last_pga_anchor_pred=prediction,
            _last_pga_anchor_candidate=torch.zeros(2, 2, 3),
            _last_station_valid=station_valid,
        )
        p_picks = {
            'input_pga_values': torch.tensor([[0.0, 0.5, 1.0], [-0.5, 99.0, 99.0]]),
            'input_pga_valid': station_valid.clone(),
        }
        norm = {'enabled': True, 'mean': -1.0, 'std': 0.5}
        anchor_cfg = {'enabled': True, 'weight': 0.2, 'huber_delta_dex': 0.15}
        anchor_loss = train_light.pga_anchor_absolute_aux_loss(
            dummy, p_picks, anchor_cfg, norm
        )
        anchor_target = (p_picks['input_pga_values'] + 1.0) / 0.5
        expected_anchor = 0.2 * F.smooth_l1_loss(
            prediction[station_valid],
            anchor_target[station_valid],
            beta=0.3,
        )
        self.assertTrue(torch.allclose(anchor_loss, expected_anchor))
        changed_picks = {key: value.clone() for key, value in p_picks.items()}
        changed_picks['input_pga_values'][~station_valid] = -9999.0
        self.assertTrue(torch.allclose(
            anchor_loss,
            train_light.pga_anchor_absolute_aux_loss(dummy, changed_picks, anchor_cfg, norm),
        ))

        labels = [torch.tensor([
            [[[0.0]], [[0.5]]],
            [[[-0.5]], [[99.0]]],
        ])]
        query_valid = torch.tensor([[True, True], [True, False]])
        pair_cfg = {'enabled': True, 'weight': 0.05, 'huber_delta_dex': 0.2}
        pair_loss = train_light.pga_anchor_pair_candidate_aux_loss(
            dummy, labels, ['pga'], query_valid, pair_cfg, norm
        )
        target = ((labels[0].squeeze(-1).squeeze(-1) + 1.0) / 0.5)
        pair_mask = query_valid[:, :, None] & station_valid[:, None, :]
        expected_pair = 0.05 * F.smooth_l1_loss(
            dummy._last_pga_anchor_candidate[pair_mask],
            target[:, :, None].expand_as(dummy._last_pga_anchor_candidate)[pair_mask],
            beta=0.4,
        )
        self.assertTrue(torch.allclose(pair_loss, expected_pair))

        mdn = torch.zeros(2, 2, 2, 3)
        mdn[..., 2] = 1.0
        difference_cfg = {
            'enabled': True,
            'weight': 0.4,
            'huber_delta_dex': 0.15,
            'max_pairs': 105,
        }
        difference = train_light.pga_within_event_difference_aux_loss(
            [mdn], labels, ['pga'], query_valid, difference_cfg, norm
        )
        labels_changed = [labels[0].clone()]
        labels_changed[0][~query_valid] = -12345.0
        difference_changed = train_light.pga_within_event_difference_aux_loss(
            [mdn], labels_changed, ['pga'], query_valid, difference_cfg, norm
        )
        self.assertTrue(torch.allclose(difference, difference_changed))
        self.assertEqual(train_light.pga_huber_delta_model_units(difference_cfg, norm), 0.3)

    def test_normal_replay_mask_and_no_label_forward_input(self):
        self.assertNotIn(
            'input_pga',
            inspect.signature(models.PGAAnchorTransferHead.forward).parameters,
        )
        base = torch.zeros(2, 2, 1)
        final = torch.tensor([[[10.0], [10.0]], [[1.0], [2.0]]])
        dummy = types.SimpleNamespace(_last_pga_anchor_base_mean=base)
        dummy._pga_point_mean_from_output = lambda value: value
        valid = torch.ones(2, 2, dtype=torch.bool)
        p_picks = {'causal_random_mask_applied': torch.tensor([True, False])}
        cfg = {'enabled': True, 'weight': 0.05, 'apply_to_random': False}
        loss = train_light.normal_replay_distillation_aux_loss(
            dummy,
            [final],
            ['pga'],
            valid,
            p_picks,
            cfg,
        )
        expected = 0.05 * F.smooth_l1_loss(
            final[1], base[1], beta=0.15
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_mdn_shift_and_eval_exports(self):
        torch.manual_seed(8)
        mdn = torch.randn(2, 4, 3, 3)
        mdn[..., 2] = mdn[..., 2].abs() + 0.1
        delta = torch.full((2, 4, 1), 0.25)
        shifted = models.FullModel._shift_pga_output_by_delta(
            types.SimpleNamespace(output_distribution='mdn'), mdn, delta
        )
        self.assertTrue(torch.equal(shifted[..., 0], mdn[..., 0]))
        self.assertTrue(torch.equal(shifted[..., 2], mdn[..., 2]))
        self.assertTrue(torch.allclose(shifted[..., 1], mdn[..., 1] + 0.25))

        raw_model = types.SimpleNamespace(
            _last_pga_temporal_base=None,
            _last_pga_temporal_delta=None,
            _last_pga_temporal_pred=None,
            _last_pga_temporal_final=None,
            _last_station_distinctive_local_residual_pred=None,
            _last_station_distinctive_local_absolute_pred=None,
            _last_pga_anchor_pred=torch.tensor([[1.0, 2.0]]),
            _last_pga_anchor_transfer=torch.tensor([[[0.5, -0.5]]]),
            _last_pga_anchor_candidate=torch.tensor([[[1.5, 1.5]]]),
            _last_pga_anchor_station_weights=torch.tensor([[[0.25, 0.75]]]),
            _last_pga_anchor_field_mean=torch.tensor([[[1.5]]]),
            _last_pga_anchor_applied_delta=torch.tensor([[[0.2]]]),
        )
        results = defaultdict(list)
        config = {
            'training_params': {
                'pga_target_normalization': {
                    'enabled': True,
                    'mean': -1.0,
                    'std': 0.5,
                }
            }
        }
        eval_checkpoint.append_pga_temporal_residual_outputs(results, raw_model, config)
        self.assertEqual(set(results), {
            'pga_anchor_pred',
            'pga_anchor_transfer',
            'pga_anchor_candidate',
            'pga_anchor_station_weights',
            'pga_anchor_field_mean',
            'pga_anchor_applied_delta',
        })
        self.assertAlmostEqual(float(results['pga_anchor_applied_delta'][0][0, 0]), 0.1)
        self.assertAlmostEqual(float(results['pga_anchor_pred'][0][0]), -0.5)


if __name__ == '__main__':
    unittest.main()
