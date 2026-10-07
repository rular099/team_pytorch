"""Reproduce unequal stochastic executions; reject real derivative changes."""
import json

import pytest
import torch

from fe01.model import build_model as legacy_build
from fe01_a01.identity import reuse_equivalence
from fe01_a01.equivalence import compare_tensors, paired_execution
from test_fe01_a01_helpers import config, sample


def test_paired_dropout_update_and_rng_backend_restoration(tmp_path):
    cfg = config(tmp_path)
    cfg['model_params']['hidden_dropout'] = .2
    cfg['model_params']['mad_params']['att_dropout'] = .2
    _, _, inputs = sample(tmp_path, cfg)
    initial = legacy_build(cfg).state_dict()
    rng = torch.get_rng_state().clone()
    settings = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    result = reuse_equivalence(cfg, cfg, initial, inputs, tmp_path/'ON_equivalence.json')
    assert result['status'] == 'PASS' and result['initial_equal']
    assert result['execution']['rng_replayed'] and result['update_numerically_equivalent']
    assert result['checks']['gradients']['passed'] and result['checks']['train_forward']['passed']
    assert torch.equal(rng, torch.get_rng_state())
    assert settings == (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    assert json.loads((tmp_path/'ON_equivalence.json').read_text())['status'] == 'PASS'


def test_changed_gradient_fails_and_saves_specific_evidence(tmp_path, monkeypatch):
    import fe01_a01.model as a01_model
    cfg = config(tmp_path); _, _, inputs = sample(tmp_path, cfg)
    initial = legacy_build(cfg).state_dict()
    original = a01_model.build_model

    def wrong_derivative(*args, **kwargs):
        model = original(*args, **kwargs)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda gradient: -gradient)
        return model

    monkeypatch.setattr(a01_model, 'build_model', wrong_derivative)
    path = tmp_path/'ON_equivalence.json'
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='gradients.*updated_state.*ON_equivalence.json'):
        reuse_equivalence(cfg, cfg, initial, inputs, path)
    report = json.loads(path.read_text())
    assert report['initial_equal'] and report['checks']['eval_forward']['passed']
    assert report['checks']['loss']['passed']
    assert report['checks']['gradients']['failed_tensors']
    assert report['checks']['updated_state']['failed_tensors']
    assert report['status'] == 'FAIL' and torch.equal(rng, torch.get_rng_state())


def test_last_bit_differences_and_invalid_numeric_states():
    baseline = torch.tensor([1., 0., -2.])
    last_bit = torch.nextafter(baseline, torch.full_like(baseline, float('inf')))
    assert not torch.equal(baseline, last_bit)
    assert compare_tensors({'weight':baseline}, {'weight':last_bit})['passed']
    for wrong in (baseline + .001, torch.full_like(baseline, float('nan')),
                  baseline[:2], baseline.double()):
        report = compare_tensors({'weight':baseline}, {'weight':wrong})
        assert not report['passed'] and report['failed_tensors'] == ['weight']
    assert not compare_tensors({'gradient':None}, {'gradient':torch.zeros(1)})['passed']
    assert not compare_tensors({'weight':baseline}, {})['passed']
    assert not compare_tensors({'counter':torch.tensor([1])}, {'counter':torch.tensor([2])})['passed']


def test_backend_settings_restored_even_when_comparison_raises():
    before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
              torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='sentinel'):
        with paired_execution('cpu'):
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            torch.rand(10)
            raise RuntimeError('sentinel')
    assert torch.equal(rng, torch.get_rng_state())
    assert before == (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                      torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
