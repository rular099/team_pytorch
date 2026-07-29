#!/usr/bin/env python3
"""Fast structural checks for the rt46/rt52-rt54 stage-1 matrix."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import gemini_models as models
from gemini_util_light import (
    _center_waveforms_with_sample_mask,
    _contiguous_waveform_support_mask,
    _crop_aligned_sample_mask,
)


CONFIG_DIR = ROOT / "pga_configs"
CONFIGS = {
    46: "transformer_japan_overfit_pga15_stage2_512_rt46_knet_cached_dpk_event_temporal_residual_scale4_chaosuan.json",
    52: "transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json",
    53: "transformer_japan_overfit_pga15_stage2_512_rt53_knet_nlta_s_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json",
    54: "transformer_japan_overfit_pga15_stage2_512_rt54_knet_nlta_m_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json",
}

ALLOWED_MODEL_KEYS = {
    "diting_station_adapter",
    "diting_nlta_x_channels",
    "diting_nlta_side_channels",
    "diting_nlta_x_pool_queries",
    "diting_nlta_side_pool_queries",
    "diting_nlta_attention_heads",
    "diting_nlta_tcn_dilations",
    "diting_nlta_ffn_multiplier",
    "diting_nlta_dropout",
}
ALLOWED_GENERATOR_KEYS = {
    "emit_waveform_padding_mask",
    "waveform_padding_mask_eps",
}


def load_configs():
    return {
        rt: json.loads((CONFIG_DIR / name).read_text())
        for rt, name in CONFIGS.items()
    }


def changed_keys(left, right):
    keys = set(left) | set(right)
    return {key for key in keys if left.get(key) != right.get(key)}


def validate_config_matrix(configs):
    baseline = configs[46]
    for rt in (52, 53, 54):
        config = configs[rt]
        top_level_changes = changed_keys(baseline, config)
        assert top_level_changes <= {
            "model_params",
            "training_params",
            "experiment_note",
            "station_filter_note",
        }, (rt, top_level_changes)

        model_changes = changed_keys(
            baseline["model_params"],
            config["model_params"],
        )
        assert model_changes <= ALLOWED_MODEL_KEYS, (rt, model_changes)

        train_base = dict(baseline["training_params"])
        train_new = dict(config["training_params"])
        base_generators = train_base.pop("generator_params")
        new_generators = train_new.pop("generator_params")
        train_changes = changed_keys(train_base, train_new)
        assert train_changes == {"weight_path"}, (rt, train_changes)
        assert len(base_generators) == len(new_generators) == 1
        generator_changes = changed_keys(base_generators[0], new_generators[0])
        assert generator_changes == ALLOWED_GENERATOR_KEYS, (rt, generator_changes)

        model_params = config["model_params"]
        generator = new_generators[0]
        assert generator["emit_waveform_padding_mask"] is True
        assert config["training_params"]["station_filter"] == "knet"
        assert model_params["station_token_weight_mode"] == "none"
        assert model_params["temporal_token_weight_mode"] == "cached_dpk_event"
        assert (
            config["training_params"]["dpk_prior_cache"]["paths"]
            == baseline["training_params"]["dpk_prior_cache"]["paths"]
        )

    assert configs[52]["model_params"]["diting_station_adapter"] == "legacy"
    assert configs[53]["model_params"]["diting_station_adapter"] == "nlta"
    assert configs[54]["model_params"]["diting_station_adapter"] == "nlta"
    assert configs[53]["model_params"]["diting_nlta_x_channels"] == 256
    assert configs[53]["model_params"]["diting_nlta_side_channels"] == 96
    assert configs[54]["model_params"]["diting_nlta_x_channels"] == 384
    assert configs[54]["model_params"]["diting_nlta_side_channels"] == 128


def validate_storage_mask_helpers():
    waveform = np.zeros((2, 12, 3), dtype=np.float32)
    waveform[0, 3:10] = 1.0
    waveform[0, 6] = 0.0
    waveform[1, 5:8] = np.arange(9, dtype=np.float32).reshape(3, 3)
    mask = _contiguous_waveform_support_mask(waveform)
    assert mask[0].tolist() == [False] * 3 + [True] * 7 + [False] * 2
    assert mask[0, 6], "internal exact zero must remain valid"
    centered = _center_waveforms_with_sample_mask(waveform, mask)
    assert np.all(centered[~mask] == 0)
    cropped = _crop_aligned_sample_mask(mask, target_length=8, crop_start=2)
    assert cropped.shape == (2, 8)
    padded = _crop_aligned_sample_mask(mask[:, :6], target_length=8, crop_start=0)
    assert padded.shape == (2, 8)
    assert not padded[:, 6:].any()


def validate_model_mask_plumbing():
    full_model = models.FullModel.__new__(models.FullModel)
    waveform = torch.randn(2, 3, 3, 12)
    sample_mask = torch.zeros(2, 3, 12, dtype=torch.bool)
    sample_mask[:, :, 4:] = True
    normalized = models.FullModel._normalize(
        full_model,
        waveform,
        mode="std",
        axis=3,
        sample_mask=sample_mask,
    )
    assert torch.equal(
        normalized[:, :, :, :4],
        torch.zeros_like(normalized[:, :, :, :4]),
    )

    cached = torch.rand(2, 3, 4, 40)
    parsed = models.FullModel._parse_extra_inputs(
        full_model,
        (sample_mask, cached),
        dataset=None,
    )
    assert parsed[4] is sample_mask
    assert parsed[5] is cached

    pool = models.AttentionPool1d(8, num_queries=2)
    features = torch.randn(2, 8, 12)
    pool_mask = torch.zeros(2, 12, dtype=torch.bool)
    pool_mask[0, 4:] = True
    output = pool(
        features,
        token_weight=torch.ones(2, 12),
        token_weight_scale=0.0,
        token_mask=pool_mask,
    )
    assert torch.isfinite(output).all()
    changed = features.clone()
    changed[0, :, :4] += 1000
    changed_output = pool(
        changed,
        token_weight=torch.ones(2, 12),
        token_weight_scale=0.0,
        token_mask=pool_mask,
    )
    assert torch.equal(output[0], changed_output[0])


def adapter_inputs(batch=2, encoder_dim=1792):
    return [
        torch.randn(batch, encoder_dim, 40),
        torch.randn(batch, encoder_dim, 20),
        torch.randn(batch, encoder_dim, 10),
        torch.randn(batch, 20, encoder_dim),
    ]


def validate_adapter(name, adapter, expected_params):
    adapter.eval()
    features = adapter_inputs(encoder_dim=adapter.encoder_dim)
    sample_mask = torch.zeros(2, 100, dtype=torch.bool)
    sample_mask[0, 20:] = True
    with torch.no_grad():
        output = adapter(features, token_mask=sample_mask)
    assert output.shape == (2, 1000)
    assert torch.isfinite(output).all()
    parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
    assert parameter_count == expected_params, (name, parameter_count, expected_params)

    changed = [feature.clone() for feature in features]
    changed[0][0, :, :8] += 1000
    changed[1][0, :, :4] += 1000
    changed[2][0, :, :2] += 1000
    changed[3][0, :4, :] += 1000
    with torch.no_grad():
        changed_output = adapter(changed, token_mask=sample_mask)
    assert torch.equal(output[0], changed_output[0]), (
        f"{name}: masked prefix changed the station embedding"
    )
    print(f"[OK] {name}: params={parameter_count:,}, finite output, masked-prefix invariant")


def main():
    torch.manual_seed(42)
    configs = load_configs()
    validate_config_matrix(configs)
    print("[OK] configs: only the stage-1 allowlisted variables differ from rt46")
    validate_storage_mask_helpers()
    print("[OK] storage mask: edge padding removed, internal zeros retained")
    validate_model_mask_plumbing()
    print("[OK] model mask: parsing, normalization, legacy pooling, all-invalid rows")

    validate_adapter(
        "NLTA-S",
        models.NativeScaleLateFusionAdapter(
            1792,
            1000,
            x_channels=256,
            side_channels=96,
            attention_heads=4,
        ),
        expected_params=4_698_744,
    )
    validate_adapter(
        "NLTA-M",
        models.NativeScaleLateFusionAdapter(
            1792,
            1000,
            x_channels=384,
            side_channels=128,
            attention_heads=6,
        ),
        expected_params=7_512_888,
    )
    legacy_params = sum(
        parameter.numel()
        for parameter in models.DitingStationAdapter(1792, 256, 1000).parameters()
    )
    assert legacy_params == 13_913_017
    print(f"[OK] legacy parameter reference: {legacy_params:,}")
    print("[PASS] native-scale adapter stage-1 validation complete")


if __name__ == "__main__":
    main()
