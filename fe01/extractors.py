"""Native intermediate features; picker output heads are never used as features."""
import importlib.metadata
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import sha256

SEISBENCH_VERSION = '0.10.1'


def checked_asset(manifest_path, family):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if family not in manifest['models']:
        raise ValueError('Offline '+family+' is unregistered; provide actual checkpoint/provenance before audit')
    entry = manifest['models'][family]
    paths = {}
    for key, record in entry['files'].items():
        path = manifest_path.parent / record['path']
        if not path.is_file():
            raise FileNotFoundError(f'Offline {family} asset missing: {path}')
        if sha256(path) != record['sha256']:
            raise ValueError(f'Offline {family} SHA mismatch: {path}')
        paths[key] = path
    return entry, paths


def load_seisbench(manifest_path, family):
    if importlib.metadata.version('seisbench') != SEISBENCH_VERSION:
        raise ValueError(f'FE01 locks SeisBench {SEISBENCH_VERSION}')
    import seisbench.models as sbm
    entry, files = checked_asset(manifest_path, family)
    if entry['weight_version'] == 'latest' or entry['seisbench_version'] != SEISBENCH_VERSION:
        raise ValueError('Weights/version must be pinned')
    cls = {'phasenet': sbm.PhaseNet, 'eqtransformer': sbm.EQTransformer}[family]
    # Construct and load directly: no annotate(), repository cleanup or network.
    metadata = json.loads(files['metadata'].read_text())
    model = cls(**metadata.get('model_args', {}))
    state = torch.load(files['weights'], map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    if model.sampling_rate != 100 or model.in_samples != entry['native_n_samples']:
        raise ValueError('Offline model capability mismatch')
    return model, entry


class PhaseNetBottleneck(nn.Module):
    output_channels = 128

    def __init__(self, picker):
        super().__init__()
        self.picker = picker
        self.output_channels = 128 * picker.filter_factor

    def forward(self, x):
        m = self.picker
        if x.shape[1:] != (3, 3001):
            raise ValueError('PhaseNet FE01 is native 3001 samples only')
        x = m.activation(m.in_bn(m.inc(x)))
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(m.down_branch):
            x = m.activation(bn1(conv_same(x)))
            if conv_down is not None:
                if i in (1, 3):
                    x = F.pad(x, (2, 3))
                elif i == 2:
                    x = F.pad(x, (1, 3))
                x = m.activation(bn2(conv_down(x)))
        return x


class EQTSharedTrunk(nn.Module):
    output_channels = 16

    def __init__(self, picker):
        super().__init__()
        self.picker = picker

    def forward(self, x):
        m = self.picker
        if x.shape[1:] != (m.in_channels, m.in_samples):
            raise ValueError('EQT native shape mismatch')
        x = m.bi_lstm_stack(m.res_cnn_stack(m.encoder(x)))
        x, _ = m.transformer_d0(x)
        x, _ = m.transformer_d(x)
        return x


class OriginalTEAM(nn.Module):
    """PyTorch port of yetinam/TEAM Conv2D→Conv1D→flatten/MLP.

    Internal raw-ln-scale concatenation is removed in favor of the common
    physical scale side channel. Flatten width is computed, never hardcoded.
    Time/channel axes follow the original Keras channels-last reshape.
    Architecture port from https://github.com/yetinam/TEAM, GPL-3.0;
    original license retained in fe01/TEAM_LICENSE.
    """
    def __init__(self, native_samples=10000, dims=(500, 500, 500), downsample=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (downsample, 1), stride=(downsample, 1))
        self.conv2 = nn.Conv2d(8, 32, (16, 3), stride=(1, 3))
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, 16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 32, 8), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 8), nn.ReLU(), nn.Conv1d(32, 16, 4), nn.ReLU())
        with torch.no_grad():
            flat_dim = self.convolve(torch.zeros(1, 3, native_samples)).numel()
        layers = []
        for width in dims:
            layers.extend([nn.Linear(flat_dim, width), nn.ReLU()])
            flat_dim = width
        self.mlp = nn.Sequential(*layers)
        self.native_samples = native_samples

    def convolve(self, x):
        x = x.transpose(1, 2).unsqueeze(1)  # N,1,time,component
        x = F.relu(self.conv2(F.relu(self.conv1(x)))).squeeze(-1)
        return self.temporal(x)

    def forward(self, x):
        if x.shape[1:] != (3, self.native_samples):
            raise ValueError('TEAM native shape mismatch')
        x = self.convolve(x)
        # Keras Flatten on (time,features) rather than PyTorch (features,time).
        return self.mlp(x.transpose(1, 2).contiguous().flatten(1))


def masked_mean(features, sample_mask):
    if features.ndim != 3:
        raise ValueError('Expected intermediate features N,C,time')
    support = F.adaptive_avg_pool1d(sample_mask[:, None].float(), features.shape[-1])
    return (features * support).sum(-1) / support.sum(-1).clamp_min(1)


class StationFrontend(nn.Module):
    def __init__(self, encoder, output_dim, native_dim=None, frozen=False, component_order='NEZ',
                 diting_adapter=None, constant=False):
        super().__init__()
        self.encoder = encoder
        self.diting_adapter = diting_adapter
        self.frozen = frozen
        self.constant = constant
        self.output_dim = output_dim
        self.permutation = ['NEZ'.index(component) for component in component_order]
        self.projection = nn.Linear(native_dim, output_dim) if native_dim is not None else nn.Identity()
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x, sample_mask):
        if self.constant:
            return x.new_zeros((len(x), self.output_dim))
        x = x[:, self.permutation]
        if self.frozen:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        if self.diting_adapter is not None:
            result = self.diting_adapter(features, token_mask=sample_mask)
        elif features.ndim == 3:
            result = self.projection(masked_mean(features, sample_mask))
        else:
            result = self.projection(features)
        return result * sample_mask.any(-1, keepdim=True).to(result.dtype)
