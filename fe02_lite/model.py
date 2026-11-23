"""Small task network over the real, frozen [1792,200] DiTing features."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(*dims):
    layers = []
    for i, (left, right) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def visible_waveforms(raw, mask):
    """The mask already combines storage validity and the exclusive cutoff."""
    support = mask.bool().unsqueeze(-2)
    clean = torch.where(support, raw, torch.zeros_like(raw))
    if not torch.isfinite(clean).all():
        raise ValueError('Non-finite visible waveform')
    mean = clean.sum(-1, keepdim=True) / support.sum(-1, keepdim=True).clamp_min(1)
    centered = torch.where(support, clean - mean, torch.zeros_like(clean))
    peak = centered.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return centered / peak, centered


def mdn(raw, components, dimensions):
    # Match legacy MixtureOutput semantics: logits, D means, D positive scales.
    # Packed head layout is [K logits, K*D means, K*D pre-scales].
    k, d = components, dimensions
    logits = raw[..., :k, None]
    means = raw[..., k:k + k*d].reshape(*raw.shape[:-1], k, d)
    scales = F.softplus(raw[..., k + k*d:].reshape(*raw.shape[:-1], k, d)) + 1e-4
    return torch.cat((logits, means, scales), -1)


class EventReadout(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = mlp(dim, 2*dim, dim)

    def forward(self, memory, valid):
        q = self.query.expand(len(memory), -1, -1)
        h, weights = self.attention(self.query_norm(q), self.memory_norm(memory), memory,
                                    key_padding_mask=~valid, need_weights=True)
        h = q + h
        return (h + self.ffn(self.ffn_norm(h)))[:, 0], weights


class LiteModel(nn.Module):
    output_layout = ['mag', 'loc', 'pga']

    def __init__(self, encoder, architecture, seed=42, amplitude_fixed_scale=1.0, encoder_batch=8):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.encoder_batch = encoder_batch
        self.amplitude_fixed_scale = float(amplitude_fixed_scale)
        d, a, r, heads = (architecture[k] for k in
                          ('model_dim', 'amplitude_dim', 'station_pool_queries', 'attention_heads'))
        self.dim, self.slots = d, r
        self.projection = nn.Linear(1792, d-a)
        self.amplitude = nn.Conv1d(3, a, 50, stride=50)
        self.pool_queries = nn.Parameter(torch.empty(r, d))
        self.pool_key_norm = nn.LayerNorm(d)
        self.coordinates = mlp(3, d, d)
        self.joint = mlp(2*d, 2*d, d)
        self.event = EventReadout(d, heads)
        self.location = mlp(d, 32, 21)
        # Magnitude is one Gaussian (mu, pre-sigma), no redundant mixing logit.
        self.magnitude = mlp(d, 32, 2)
        self.spatial = mlp(2*d, 2*d, d)
        self.target_query_norm = nn.LayerNorm(d)
        self.target_key_norm = nn.LayerNorm(d)
        self.target_attention = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.pga = mlp(3*d, 2*d, 9)
        self.initialize(seed)

    def initialize(self, seed):
        # Parameter-name streams pair all same-shaped shared modules across variants.
        with torch.random.fork_rng(devices=[]):
            for name, p in self.named_parameters():
                if name.startswith('encoder.'):
                    continue
                torch.manual_seed(int(hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest()[:12], 16))
                if p.ndim >= 2:
                    if 'queries' in name or name == 'event.query':
                        nn.init.normal_(p, std=.02)
                    else:
                        nn.init.xavier_uniform_(p)
                elif name.endswith('weight') and 'norm' in name:
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def task_state(self):
        return {n: t for n, t in self.state_dict().items() if not n.startswith('encoder.')}

    def load_task_state(self, state):
        expected = self.task_state()
        if set(state) != set(expected) or any(state[n].shape != expected[n].shape for n in expected):
            raise ValueError('Task checkpoint keys/shapes mismatch')
        # Do not recurse through encoder load hooks: e.g. BatchNorm can synthesize
        # missing num_batches_tracked even under strict=False, mutating buffers.
        with torch.no_grad():
            for name,tensor in expected.items():
                if state[name].dtype != tensor.dtype:
                    raise ValueError('Task checkpoint dtype mismatch: '+name)
                tensor.copy_(state[name])

    def encoder_state(self):
        return {'encoder.'+n: t for n, t in self.encoder.state_dict().items()}

    def encode_observations(self, raw, coords, stations, sample_mask, native_debug=False):
        if raw.shape[-2:] != (3, 10000) or sample_mask.shape != raw.shape[:2] + (10000,):
            raise ValueError('Expected [B,N,3,10000] plus [B,N,10000] mask')
        mask = sample_mask.bool() & stations.bool()[..., None]
        valid = mask.any(-1)
        if not valid.any(-1).all():
            raise ValueError('unsupported:no_valid_input_station')
        normalized, physical = visible_waveforms(raw, mask)
        b, n = raw.shape[:2]
        flattened = normalized.flatten(0, 1)
        active = valid.flatten().nonzero().flatten()
        native = raw.new_zeros((b*n, 1792, 200))
        # No autograd graph / train-mode buffer mutations in the frozen encoder.
        with torch.no_grad():
            for ids in active.split(self.encoder_batch):
                features = self.encoder(flattened[ids])
                if features.shape != (len(ids), 1792, 200):
                    raise ValueError('DiTing native feature interface differs from [1792,200]')
                native[ids] = features
        projected = self.projection(native.transpose(1, 2)).reshape(b, n, 200, -1)
        amp = F.gelu(self.amplitude(physical.flatten(0, 1) * self.amplitude_fixed_scale))
        amp = amp.transpose(1, 2).reshape(b, n, 200, -1)
        token_valid = mask.reshape(b, n, 200, 50).any(-1)
        amp = amp * token_valid[..., None]
        tokens = torch.cat((projected, amp), -1)
        keys = self.pool_key_norm(tokens)
        scores = torch.einsum('rd,bntd->bnrt', self.pool_queries, keys) / math.sqrt(self.dim)
        # Avoid softmax(all -inf) on padded stations; their weights are then zeroed.
        scores = scores.masked_fill(~token_valid[:, :, None, :], -1e9)
        weights = scores.softmax(-1) * token_valid[:, :, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        # Values retain raw amplitude information; no value LayerNorm.
        u = torch.einsum('bnrt,bntd->bnrd', weights, tokens)
        ec = self.coordinates(coords)
        m0 = self.joint(torch.cat((u, ec[:, :, None].expand(-1, -1, self.slots, -1)), -1))
        memory_valid = valid[:, :, None].expand(-1, -1, self.slots).flatten(1)
        e, event_attention = self.event(m0.flatten(1, 2), memory_valid)
        source_loc = mdn(self.location(e), 3, 3)
        mag_raw = self.magnitude(e)
        source_mag = torch.stack((torch.zeros_like(mag_raw[:, 0]), mag_raw[:, 0],
                                  F.softplus(mag_raw[:, 1]) + 1e-4), -1)[:, None]
        rs = self.spatial(torch.cat((e[:, None].expand(-1, n, -1), ec), -1))
        memory = m0 + rs[:, :, None]
        encoded = dict(memory=memory.flatten(1, 2), memory_valid=memory_valid, e=e,
                    source_loc=source_loc, source_mag=source_mag, amplitude=amp,
                    token_mask=token_valid, pool_attention=weights, u=u, m0=m0, m=memory,
                    station_spatial=rs, event_attention=event_attention,
                    token_visible_fraction=mask.reshape(b,n,200,50).float().mean(-1))
        if native_debug:
            encoded['native_features_debug'] = native.reshape(b,n,1792,200)
        return encoded

    def read_targets(self, encoded, targets):
        e, memory = encoded['e'], encoded['memory']
        q = self.spatial(torch.cat((e[:, None].expand(-1, targets.shape[1], -1), self.coordinates(targets)), -1))
        delta, attention = self.target_attention(self.target_query_norm(q), self.target_key_norm(memory), memory,
            key_padding_mask=~encoded['memory_valid'], need_weights=True)
        h = q + delta
        prediction = mdn(self.pga(torch.cat((h, e[:, None].expand_as(q), q), -1)), 3, 1)
        return prediction, dict(query=q, h=h, target_attention=attention, target_spatial=q)

    def forward(self, raw, coords, stations, targets, target_valid, sample_mask, diagnostics=False, native_debug=False):
        if native_debug and not diagnostics:
            raise ValueError('Native features are only available in explicit diagnostic debug mode')
        encoded = self.encode_observations(raw, coords, stations, sample_mask,native_debug)
        prediction, target = self.read_targets(encoded, targets)
        outputs = [encoded['source_mag'], encoded['source_loc'], prediction]
        if diagnostics:
            # Exporting is done by the caller; collecting has no RNG or output side effects.
            return outputs, {**encoded, **target, 'pga': prediction}
        return outputs


def build_model(cfg, device='cpu'):
    from fe01.extractors import checked_asset
    from fe01.config import sha256
    from train_light import build_diting_args
    from dtbench.training.modeling import _extract_pretrained_state_dict, _filter_backbone_state_dict
    import gemini_models as legacy
    entry, files = checked_asset(cfg['pretrained_manifest'], 'diting')
    args = build_diting_args(cfg['diting_config'], device='cpu', pretrained_override='')
    if (args.diting_frontend, args.target_width, args.model_depth, args.in_samples, args.patch_size) != (
            'backbone_attn_pool', 1792, 24, 10000, 50):
        raise ValueError('Expected fixed MAE1200M backbone YAML')
    if (entry['component_order'], entry['sampling_rate'], entry['native_n_samples']) != ('NEZ', 100, 10000):
        raise ValueError('Encoder asset declaration mismatch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'] + 19001)
        sequence = legacy.get_diting_model(args, cfg['architecture']['model_dim'])
        checkpoint = torch.load(files['weights'], map_location='cpu', weights_only=False)
        state = _filter_backbone_state_dict(_extract_pretrained_state_dict(checkpoint, args))
        expected = {'0.'+k: v for k, v in sequence[0].state_dict().items()}
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError('Pretrained encoder keys/shapes mismatch')
        sequence[0].load_state_dict({k[2:]: v for k, v in state.items()}, strict=True)
        encoder = sequence[0]
        patch = encoder.backbone.W_p
        if not (isinstance(patch, nn.Conv1d) and patch.kernel_size == (50,) and patch.stride == (50,)
                and patch.padding == (0,) and patch.dilation == (1,) and patch.in_channels == 3):
            raise ValueError('DiTing patch start/stride/padding differs; token alignment refused')
        if encoder.backbone.mask_ratio != 0:
            raise ValueError('Pretrained masking must be disabled for ordered 200-token inference')
        del sequence, checkpoint, state, expected
        model = LiteModel(encoder, cfg['architecture'], cfg['seed'], cfg['amplitude_fixed_scale'],
                          cfg['sampling']['encoder_batch'])
    model.encoder_identity = dict(checkpoint_sha256=entry['files']['weights']['sha256'],
        checkpoint_bytes=files['weights'].stat().st_size,
        checkpoint_path=str(files['weights'].resolve()), yaml_sha256=sha256(cfg['diting_config']),
        pretrained_manifest_sha256=sha256(cfg['pretrained_manifest']),pretrained_metadata=entry,
        encoder_class=type(encoder).__module__+'.'+type(encoder).__name__, native_shape=[1792,200],
        patch=dict(start=0, stride=50, kernel=50, padding=0, order='left to right; channel-major within patch'),
        effective_args={k: str(v) for k,v in vars(args).items()})
    return model.to(device)


def parameter_audit(model):
    groups = {}
    for name, value in model.named_parameters():
        key = name.split('.')[0]
        group = groups.setdefault(key, dict(registered=0, trainable=0))
        group['registered'] += value.numel()
        group['trainable'] += value.numel() if value.requires_grad else 0
    return dict(registered=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), modules=groups,
                optimizer_parameters=[dict(name=n, shape=list(p.shape), numel=p.numel())
                    for n,p in model.named_parameters() if p.requires_grad])
