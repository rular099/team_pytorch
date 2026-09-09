import hashlib
import os
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

import train_light


models = train_light.models
REPO_ROOT = Path(__file__).resolve().parents[1]
RT57_CONFIG = (
    REPO_ROOT
    / 'pga_configs'
    / 'transformer_japan_full_2000_2024_rt57_gtnp_v2_station_distinctive_seed42_chaosuan.json'
)


def _head(zero_init=False):
    torch.manual_seed(20260908)
    return models.PGATemporalResidualHead(
        token_dim=16,
        emb_dim=12,
        output_mlp_dims=(8,),
        hidden_dim=8,
        geom_hidden_dim=6,
        temporal_token_dim=16,
        zero_init=zero_init,
        station_distinctive_adapter=True,
        station_distinctive_dim=10,
        station_distinctive_attention_queries=4,
        station_distinctive_temporal_dilations=(1, 4),
        station_distinctive_amplitude_feature_dim=11,
        station_distinctive_duration_feature_dim=2,
        station_distinctive_hidden_dim=20,
    )


def _head_inputs():
    torch.manual_seed(17)
    batch, stations, tokens, targets = 2, 3, 9, 4
    return {
        'query': torch.randn(batch, targets, 12),
        'station_tokens': torch.randn(batch, stations, tokens, 16),
        'station_emb': torch.randn(batch, stations, 12),
        'station_valid': torch.tensor(
            [[True, True, True], [True, False, False]], dtype=torch.bool
        ),
        'query_coords': torch.randn(batch, targets, 3),
        'station_coords': torch.randn(batch, stations, 3),
        'event_emb': torch.randn(batch, 12),
        'station_token_mask': torch.tensor([
            [[True] * 9, [True] * 6 + [False] * 3, [True] * 4 + [False] * 5],
            [[True] * 7 + [False] * 2, [False] * 9, [False] * 9],
        ]),
        'amplitude_features': torch.randn(batch, stations, 11),
        'duration_features': torch.rand(batch, stations, 2),
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


class RT57StationDistinctiveTests(unittest.TestCase):
    def test_config_protocol_and_legacy_configs_are_unchanged(self):
        expected_hashes = {
            'pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json': (
                'bb28ce3b66a6bd389e6ccd2cb53062c1e68cdb603535f9930c75ac99a33d1c8c'
            ),
            'pga_configs/transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json': (
                '92fe9f0f0942ae7c7e75ae4351c9cd90cfd274dcbbbfd474b5a6648994fb8015'
            ),
        }
        for relative, expected in expected_hashes.items():
            digest = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)

        env = {
            'JAPAN_FULL_DATA_ROOT': '/tmp/japan',
            'JAPAN_FULL_WEIGHT_PATH': 'rt55',
            'RT55_EP32_CHECKPOINT': '/tmp/rt55.pth',
            'RT56_WEIGHT_PATH': 'rt56',
            'RT56_BASE_CHECKPOINT': '/tmp/rt56.pth',
            'RT57_WEIGHT_PATH': 'rt57',
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = train_light.load_config_file(str(RT57_CONFIG))
        model_cfg = config['model_params']
        train_cfg = config['training_params']
        self.assertTrue(model_cfg['station_distinctive_adapter'])
        self.assertTrue(model_cfg['station_distinctive_use_amplitude_features'])
        self.assertTrue(model_cfg['station_distinctive_use_duration_features'])
        self.assertTrue(model_cfg['station_distinctive_common_residual_decomposition'])
        self.assertTrue(model_cfg['temporal_pool_pair_value_enabled'])
        self.assertEqual(model_cfg['pga_temporal_residual_station_weighting'], 'learned_pair')
        self.assertEqual(train_cfg['freeze_mode'], 'station_distinctive_residual_only')
        self.assertEqual(train_cfg['epochs_full_model'], 6)
        self.assertEqual(train_cfg['lr_pga_temporal_residual'], 5e-4)
        self.assertEqual(
            train_cfg['train_generator_overrides']['causal_random_input_mask']['apply_probability'],
            0.75,
        )
        self.assertEqual(
            train_cfg['validation_generator_overrides']['causal_random_input_mask']['station_counts'],
            [1, 3, 5, 8, 12, 16],
        )

    def test_adapter_masks_and_single_station_residual(self):
        head = _head()
        args = _head_inputs()
        output = head(**args)
        self.assertEqual(output.shape, (2, 4, 1))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(head._last_station_u.shape, (2, 3, 10))
        self.assertTrue(torch.equal(
            head._last_station_u[1, 1:], torch.zeros_like(head._last_station_u[1, 1:])
        ))
        self.assertTrue(torch.allclose(
            head._last_station_d[1, 0], torch.zeros_like(head._last_station_d[1, 0]), atol=1e-6
        ))

    def test_empty_token_rows_are_safe_with_many_stations(self):
        """Regression for PyTorch 1.13 batch/station boolean indexing."""
        head = _head().eval()
        torch.manual_seed(23)
        batch, stations, tokens, targets = 8, 25, 200, 3
        station_valid = torch.zeros(batch, stations, dtype=torch.bool)
        station_valid[:, :4] = True
        station_token_mask = torch.zeros(batch, stations, tokens, dtype=torch.bool)
        station_token_mask[:, 0, :37] = True
        station_token_mask[:, 1, :91] = True
        station_token_mask[:, 2, :1] = True
        # Station 3 is marked valid but deliberately has an empty token row;
        # stations 4: are padding and therefore empty as well.
        args = {
            'query': torch.randn(batch, targets, 12),
            'station_tokens': torch.randn(batch, stations, tokens, 16),
            'station_emb': torch.randn(batch, stations, 12),
            'station_valid': station_valid,
            'query_coords': torch.randn(batch, targets, 3),
            'station_coords': torch.randn(batch, stations, 3),
            'event_emb': torch.randn(batch, 12),
            'station_token_mask': station_token_mask,
            'amplitude_features': torch.randn(batch, stations, 11),
            'duration_features': torch.rand(batch, stations, 2),
        }
        with torch.no_grad():
            output = head(**args)
        self.assertEqual(output.shape, (batch, targets, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_six_required_controls_are_functional_and_finite(self):
        head = _head(zero_init=False).eval()
        full_args = _head_inputs()
        args = {key: value[:1].clone() for key, value in full_args.items()}
        args['station_valid'][:] = True
        permutation = torch.tensor([1, 2, 0])
        with torch.no_grad():
            baseline = head(**args)

            # 1. Waveform-only station permutation, coordinates held fixed.
            waveform_only = dict(args)
            for key in (
                'station_tokens', 'station_token_mask',
                'amplitude_features', 'duration_features',
            ):
                waveform_only[key] = args[key][:, permutation]
            waveform_only_changed = head(**waveform_only)

            # 2. Waveform and station coordinates permuted together: set invariant.
            paired = dict(waveform_only)
            paired['station_emb'] = args['station_emb'][:, permutation]
            paired['station_coords'] = args['station_coords'][:, permutation]
            paired_output = head(**paired)

            # 3. Zero temporal/amplitude state removes waveform identity, but
            # the query/station geometry path remains active.
            zero_args = dict(args)
            zero_args['station_tokens'] = torch.zeros_like(args['station_tokens'])
            zero_args['station_token_mask'] = torch.zeros_like(args['station_token_mask'])
            zero_args['amplitude_features'] = torch.zeros_like(args['amplitude_features'])
            zero_args['duration_features'] = torch.ones_like(args['duration_features'])
            zero_output = head(**zero_args)
            zero_permuted = dict(zero_args)
            zero_permuted['station_coords'] = args['station_coords'][:, permutation]
            zero_permuted_output = head(**zero_permuted)
            zero_query_changed = dict(zero_args)
            zero_query_changed['query_coords'] = args['query_coords'].clone()
            zero_query_changed['query_coords'][:, :, 0] += 3.0
            zero_geometry_output = head(**zero_query_changed)

            # 4. Same geometry with a different station token sequence.
            sequence_changed = dict(args)
            sequence_changed['station_tokens'] = args['station_tokens'].clone()
            sequence_changed['station_tokens'][:, 0, 2:5, 0] += 1.5
            sequence_output = head(**sequence_changed)

            # 5. One station, identical query embeddings, distinct coordinates.
            one_station_query = {key: value.clone() for key, value in args.items()}
            one_station_query['station_valid'][:] = torch.tensor([True, False, False])
            one_station_query['query'][:, 1] = one_station_query['query'][:, 0]
            one_station_query['query_coords'][:, 1] = one_station_query['query_coords'][:, 0]
            one_station_query['query_coords'][:, 1, 0] += 4.0
            one_station_query_output = head(**one_station_query)

            # 6. One station, identical query geometry, distinct waveforms.
            one_station_wave = {key: value[:1].repeat(2, *([1] * (value.ndim - 1)))
                                for key, value in one_station_query.items()}
            one_station_wave['station_tokens'][1, 0, :, 0] += torch.linspace(0.0, 2.0, 9)
            one_station_wave_output = head(**one_station_wave)

        self.assertGreater((waveform_only_changed - baseline).abs().max().item(), 1e-8)
        self.assertTrue(torch.allclose(paired_output, baseline, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.isfinite(zero_output).all())
        self.assertTrue(torch.allclose(zero_permuted_output, zero_output, atol=1e-6, rtol=1e-6))
        self.assertGreater((zero_geometry_output - zero_output).abs().max().item(), 1e-8)
        self.assertGreater((sequence_output - baseline).abs().max().item(), 1e-8)
        self.assertGreater(
            (one_station_query_output[:, 1] - one_station_query_output[:, 0]).abs().max().item(),
            1e-8,
        )
        self.assertGreater(
            (one_station_wave_output[1] - one_station_wave_output[0]).abs().max().item(),
            1e-8,
        )

    def test_zero_init_preserves_mdn_and_only_shifts_component_means(self):
        head = _head(zero_init=True).eval()
        args = _head_inputs()
        with torch.no_grad():
            delta = head(**args)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

        torch.manual_seed(8)
        mdn = torch.randn(2, 4, 3, 3)
        mdn[..., 2] = mdn[..., 2].abs() + 0.1
        shifted = models.FullModel._shift_pga_output_by_delta(
            types.SimpleNamespace(output_distribution='mdn'),
            mdn,
            torch.full((2, 4, 1), 0.25),
        )
        self.assertTrue(torch.equal(shifted[..., 0], mdn[..., 0]))
        self.assertTrue(torch.equal(shifted[..., 2], mdn[..., 2]))
        self.assertTrue(torch.allclose(shifted[..., 1], mdn[..., 1] + 0.25))
        unchanged = models.FullModel._shift_pga_output_by_delta(
            types.SimpleNamespace(output_distribution='mdn'), mdn, delta
        )
        self.assertTrue(torch.equal(unchanged, mdn))

    def test_complete_zero_init_rt57_forward_matches_loaded_rt56_forward(self):
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
        }

        def tiny_diting(*_args, **_kwargs):
            return nn.Sequential(_TinyEncoder(), _TinyStationAdapter())

        with mock.patch.object(models, 'get_diting_model', side_effect=tiny_diting):
            torch.manual_seed(1)
            rt56 = models.build_transformer_model(**common)
            torch.manual_seed(2)
            rt57 = models.build_transformer_model(
                **common,
                use_pga_temporal_residual=True,
                station_distinctive_adapter=True,
                pga_temporal_residual_station_weighting='learned_pair',
                pga_temporal_residual_use_event_context=True,
                temporal_token_dim=16,
                temporal_pool_dim=8,
                temporal_pool_geom_hidden_dim=6,
                pga_temporal_residual_station_distinctive_dim=10,
                pga_temporal_residual_station_distinctive_hidden_dim=20,
                pga_temporal_residual_station_distinctive_amplitude_feature_dim=11,
            )
        train_light.load_model_state_dict_compatible(
            rt57,
            rt56.state_dict(),
            strict=True,
            context='tiny RT56 checkpoint',
            allowed_missing_prefixes=('pga_temporal_residual_head.',),
        )
        torch.manual_seed(3)
        waveforms = torch.randn(2, 3, 3, 20)
        station_coords = torch.randn(2, 3, 3)
        station_valid = torch.tensor([[True, True, True], [True, False, False]])
        query_coords = torch.randn(2, 2, 3)
        query_valid = torch.ones(2, 2, dtype=torch.bool)
        sample_mask = torch.ones(2, 3, 20, dtype=torch.bool)
        sample_mask[1, 1:] = False
        rt56.eval()
        rt57.eval()
        with torch.no_grad():
            rt56_outputs = rt56(
                waveforms, station_coords, station_valid,
                query_coords, query_valid, sample_mask,
            )
            rt57_outputs = rt57(
                waveforms, station_coords, station_valid,
                query_coords, query_valid, sample_mask,
            )
        self.assertEqual(len(rt56_outputs), 3)
        self.assertEqual(len(rt57_outputs), 5)
        for old_output, new_output in zip(rt56_outputs, rt57_outputs[:3]):
            self.assertTrue(torch.equal(old_output, new_output))
        self.assertTrue(torch.equal(
            rt57._last_pga_temporal_delta,
            torch.zeros_like(rt57._last_pga_temporal_delta),
        ))

    def test_checkpoint_compatibility_and_residual_only_trainability(self):
        class Legacy(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(3, 3)

        class RT57(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(3, 3)
                self.pga_temporal_residual_head = _head(zero_init=True)

        legacy = Legacy()
        rt57 = RT57()
        train_light.load_model_state_dict_compatible(
            rt57,
            legacy.state_dict(),
            strict=True,
            context='synthetic RT55/RT56 checkpoint',
            allowed_missing_prefixes=('pga_temporal_residual_head.',),
        )
        self.assertTrue(torch.equal(rt57.base.weight, legacy.base.weight))
        incomplete_legacy_state = dict(legacy.state_dict())
        incomplete_legacy_state.pop('base.bias')
        with self.assertRaisesRegex(RuntimeError, 'base.bias'):
            train_light.load_model_state_dict_compatible(
                RT57(),
                incomplete_legacy_state,
                strict=True,
                context='incomplete synthetic checkpoint',
                allowed_missing_prefixes=('pga_temporal_residual_head.',),
            )
        train_light.apply_full_model_trainability(
            rt57,
            {'freeze_mode': 'station_distinctive_residual_only'},
        )
        self.assertTrue(all(not p.requires_grad for p in rt57.base.parameters()))
        self.assertTrue(all(
            p.requires_grad for p in rt57.pga_temporal_residual_head.parameters()
        ))
        train_light.set_temporal_residual_only_train_mode(rt57)
        self.assertFalse(rt57.base.training)
        self.assertTrue(rt57.pga_temporal_residual_head.training)

    def test_disabled_switch_uses_the_legacy_temporal_head_path(self):
        torch.manual_seed(12)
        legacy_head = models.PGATemporalResidualHead(
            token_dim=16,
            emb_dim=12,
            output_mlp_dims=(8,),
            hidden_dim=8,
            geom_hidden_dim=6,
            temporal_token_dim=16,
            zero_init=False,
        ).eval()
        args = _head_inputs()
        legacy_args = {
            key: value for key, value in args.items()
            if key not in ('station_token_mask', 'amplitude_features', 'duration_features')
        }
        with torch.no_grad():
            expected = legacy_head(**legacy_args)
            actual = legacy_head(**args)
        self.assertFalse(legacy_head.station_distinctive_enabled)
        self.assertTrue(torch.equal(actual, expected))

    def test_auxiliary_losses_honor_masks_and_dual_station_cases(self):
        torch.manual_seed(9)
        mdn = torch.randn(2, 4, 3, 3)
        mdn[..., 2] = mdn[..., 2].abs() + 0.2
        labels = [torch.tensor([
            [[0.0], [0.5], [99.0], [99.0]],
            [[-0.2], [0.3], [0.8], [99.0]],
        ])]
        valid = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ])
        cfg = {'enabled': True, 'weight': 0.2, 'max_pairs': 105, 'huber_delta': 1.0}
        first = train_light.pga_within_event_difference_aux_loss(
            [mdn], labels, ['pga'], valid, cfg
        )
        labels_changed = [labels[0].clone()]
        labels_changed[0][~valid] = -9999.0
        second = train_light.pga_within_event_difference_aux_loss(
            [mdn], labels_changed, ['pga'], valid, cfg
        )
        self.assertTrue(torch.allclose(first, second))

        dummy = types.SimpleNamespace(
            _last_station_distinctive_local_residual_pred=torch.zeros(2, 3),
            _last_station_distinctive_local_absolute_pred=torch.zeros(2, 3),
            _last_station_valid=torch.tensor([[True, True, True], [True, False, False]]),
            _last_diag={},
        )
        p_picks = {
            'input_pga_values': torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.0, 0.0]]),
            'input_pga_valid': torch.tensor([[True, True, True], [True, False, False]]),
        }
        local = train_light.station_local_pga_aux_loss(
            dummy,
            labels,
            ['pga'],
            valid,
            p_picks,
            {
                'enabled': True,
                'multi_station_weight': 0.1,
                'single_station_weight': 0.02,
            },
            pga_target_normalization={'enabled': True, 'mean': -1.0, 'std': 0.5},
        )
        self.assertIsNotNone(local)
        self.assertTrue(torch.isfinite(local))
        self.assertIn('station_distinctive_local_residual_mae', dummy._last_diag)
        self.assertIn('station_distinctive_local_absolute_mae', dummy._last_diag)

    def test_two_optimizer_steps_reach_upstream_pair_modules(self):
        head = _head(zero_init=True).train()
        args = _head_inputs()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
        pair_before = head.distinctive_pair_mlp[0].weight.detach().clone()
        adapter_before = (
            head.station_distinctive_adapter.summary_mlp[0].weight.detach().clone()
        )
        target = torch.full((2, 4, 1), 0.75)
        for _ in range(2):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(head(**args), target)
            loss.backward()
            optimizer.step()
        pair_after = head.distinctive_pair_mlp[0].weight.detach()
        adapter_after = head.station_distinctive_adapter.summary_mlp[0].weight.detach()
        self.assertGreater((pair_after - pair_before).abs().max().item(), 0.0)
        self.assertGreater((adapter_after - adapter_before).abs().max().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
