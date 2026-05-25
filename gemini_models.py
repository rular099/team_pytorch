import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mup import MuReadout, set_base_shapes

from dtbench.models.diting.models.vit_adapter import ViTAdapter
from dtbench.models.diting.models.backbone_ablation import (
    Encoder_baseline_llama,
    get_encoder_size_dict,
)
from dtbench.training.modeling import (
    _extract_pretrained_state_dict,
    _filter_backbone_state_dict,
)

# Helper Functions
def gelu(x):
    """GELU activation function."""
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def normalize(s):
    return s.strip().lower().replace("-", "_")

def get_activation(activation: str):
    act = normalize(activation)
    table = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "prelu": nn.PReLU(),
        "none": nn.Identity(),
    }
    if act not in table:
        raise ValueError(f"Unknown activation: {activation}")
    return table[act]


def _rope_rotate_pairs(x, angles):
    pair_count = angles.shape[-1]
    x_pair = x.reshape(*x.shape[:-1], pair_count, 2)
    cos = torch.cos(angles)[:, None, :, :, None]
    sin = torch.sin(angles)[:, None, :, :, None]
    x0 = x_pair[..., 0:1]
    x1 = x_pair[..., 1:2]
    rotated = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rotated.reshape(*x.shape[:-1], pair_count * 2)


def _rope_center(coords, valid_mask=None):
    coords2 = coords[:, :, :2].to(dtype=torch.float32)
    if valid_mask is None:
        valid = torch.ones(coords2.shape[:2], device=coords2.device, dtype=coords2.dtype)
    else:
        valid = valid_mask.to(device=coords2.device, dtype=coords2.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (coords2 * valid[:, :, None]).sum(dim=1, keepdim=True) / denom[:, :, None]


def _rope_relative_xy(coords, center, coord_mode='relative_xy_km', lat_origin=37.0):
    coords2 = coords[:, :, :2].to(dtype=torch.float32)
    if coord_mode == 'relative_xy_model_units':
        xy = coords2 - center
        return xy[:, :, 1], xy[:, :, 0]
    if coord_mode != 'relative_xy_km':
        raise ValueError(
            "rope_coord_mode must be 'relative_xy_km' or 'relative_xy_model_units', "
            f"got {coord_mode!r}."
        )

    rel_deg = coords2 - center
    center_lat_deg = center[:, :, 0] + float(lat_origin)
    lon_scale = torch.cos(center_lat_deg * math.pi / 180.0).clamp_min(0.1)
    y_km = rel_deg[:, :, 0] * 111.19492664455874
    x_km = rel_deg[:, :, 1] * 111.19492664455874 * lon_scale.squeeze(1)[:, None]
    return x_km, y_km


def _apply_continuous_rope_qk(q, k, query_coords, key_coords, key_valid=None,
                              coord_mode='relative_xy_km', coord_scale=100.0,
                              rope_base=10000.0, lat_origin=37.0):
    if query_coords is None or key_coords is None:
        return q, k
    head_dim = q.shape[-1]
    rotary_dim = (head_dim // 4) * 4
    if rotary_dim < 4:
        return q, k

    pair_count_per_axis = rotary_dim // 4
    freq_idx = torch.arange(pair_count_per_axis, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (float(rope_base) ** (freq_idx / max(1, pair_count_per_axis)))

    center = _rope_center(key_coords.to(q.device), valid_mask=key_valid)
    q_x, q_y = _rope_relative_xy(
        query_coords.to(q.device),
        center,
        coord_mode=coord_mode,
        lat_origin=lat_origin,
    )
    k_x, k_y = _rope_relative_xy(
        key_coords.to(q.device),
        center,
        coord_mode=coord_mode,
        lat_origin=lat_origin,
    )

    q_x_angles = (q_x.to(dtype=q.dtype) / float(coord_scale))[:, :, None] * inv_freq[None, None, :]
    q_y_angles = (q_y.to(dtype=q.dtype) / float(coord_scale))[:, :, None] * inv_freq[None, None, :]
    k_x_angles = (k_x.to(dtype=k.dtype) / float(coord_scale))[:, :, None] * inv_freq.to(dtype=k.dtype)[None, None, :]
    k_y_angles = (k_y.to(dtype=k.dtype) / float(coord_scale))[:, :, None] * inv_freq.to(dtype=k.dtype)[None, None, :]

    x_slice = slice(0, 2 * pair_count_per_axis)
    y_slice = slice(2 * pair_count_per_axis, 4 * pair_count_per_axis)

    q_out = q.clone()
    k_out = k.clone()
    q_out[..., x_slice] = _rope_rotate_pairs(q_out[..., x_slice], q_x_angles)
    q_out[..., y_slice] = _rope_rotate_pairs(q_out[..., y_slice], q_y_angles)
    k_out[..., x_slice] = _rope_rotate_pairs(k_out[..., x_slice], k_x_angles)
    k_out[..., y_slice] = _rope_rotate_pairs(k_out[..., y_slice], k_y_angles)
    return q_out, k_out


def _cross_attention_with_rope(attn_module, query, key_value, key_valid,
                               attn_mask=None, query_coords=None, key_coords=None,
                               rope_coord_mode='relative_xy_km', rope_coord_scale=100.0,
                               rope_base=10000.0, rope_lat_origin=37.0):
    batch, query_len, embed_dim = query.shape
    key_len = key_value.shape[1]
    n_heads = attn_module.num_heads
    head_dim = embed_dim // n_heads
    if head_dim * n_heads != embed_dim:
        raise ValueError(f'embed_dim={embed_dim} must be divisible by n_heads={n_heads}.')

    q_weight, k_weight, v_weight = attn_module.in_proj_weight.chunk(3, dim=0)
    if attn_module.in_proj_bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = attn_module.in_proj_bias.chunk(3, dim=0)

    q = F.linear(query, q_weight, q_bias)
    k = F.linear(key_value, k_weight, k_bias)
    v = F.linear(key_value, v_weight, v_bias)

    q = q.reshape(batch, query_len, n_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(batch, key_len, n_heads, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(batch, key_len, n_heads, head_dim).permute(0, 2, 1, 3)

    q, k = _apply_continuous_rope_qk(
        q,
        k,
        query_coords=query_coords,
        key_coords=key_coords,
        key_valid=key_valid,
        coord_mode=rope_coord_mode,
        coord_scale=rope_coord_scale,
        rope_base=rope_base,
        lat_origin=rope_lat_origin,
    )

    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    if key_valid is not None:
        inv_mask = (~key_valid.bool()).to(score.dtype)[:, None, None, :]
        score = score - inv_mask * 1e6
    if attn_mask is not None:
        if attn_mask.dim() == 3:
            score = score + attn_mask.reshape(batch, n_heads, query_len, key_len).to(score.dtype)
        elif attn_mask.dim() == 2:
            score = score + attn_mask.to(score.dtype)[None, None, :, :]
        else:
            raise ValueError(f'Unsupported attn_mask shape for RoPE cross-attention: {attn_mask.shape}')

    weights = torch.softmax(score, dim=-1)
    if attn_module.dropout > 0:
        weights = F.dropout(weights, p=attn_module.dropout, training=attn_module.training)
    out = torch.matmul(weights, v)
    out = out.permute(0, 2, 1, 3).reshape(batch, query_len, embed_dim)
    out = attn_module.out_proj(out)
    return out, weights.mean(dim=1)

#class MLP(nn.Module):
#    def __init__(self, input_shape, dims=(100, 50), activation=F.relu, last_activation=None):
#        super().__init__()
#        if last_activation is None:
#            last_activation = activation
#
#        layers = [nn.Linear(input_shape[-1], dims[0]),
#                  nn.ReLU()]  # Assuming input_shape is a tuple (batch_size, feature_dim)
#
#        for i in range(len(dims) - 1):
#            layers.append(nn.Linear(dims[i], dims[i+1]))
#            if i < len(dims) - 2: # Add activation except for the last layer
#                layers.append(nn.ReLU())
#
#        self.mlp = nn.Sequential(*layers)
#
#    def forward(self, x):
#        return self.mlp(x)

class MLP(nn.Module):
    """
    通用全连接 MLP：
    - in_dim: 输入特征维度
    - dims:  每层输出维度列表（包含最后一层的输出维度）
    - activation: 中间层激活函数（默认 ReLU）
    - last_activation: 最后一层激活函数（默认 None，即不加）
    - dropout: 每个激活后可选 dropout
    - use_layernorm: 是否在最后一层前加 LayerNorm
    - use_batchnorm: 是否在每个线性层后加 BatchNorm1d（和 LayerNorm 二选一）
    """
    def __init__(
        self,
        input_shape: int,
        dims,
        activation='relu',
        last_activation=None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        assert not (use_layernorm and use_batchnorm), "LayerNorm 和 BatchNorm 不要同时开"
        in_dim = input_shape[-1]
        if isinstance(activation, str):
            activation_layer = get_activation(activation)
        if isinstance(last_activation, str):
            last_activation_layer = get_activation(last_activation)

        if isinstance(dims, int):
            dims = [dims]

        layers = []
        prev_dim = in_dim

        for i, dim in enumerate(dims):
            tmp_layer = nn.Linear(prev_dim, dim)
            nn.init.zeros_(tmp_layer.bias)
            layers.append(tmp_layer)

            is_last = (i == len(dims) - 1)

            # norm 一般只加在中间层或最后一层前，可以按需调整策略
            if use_batchnorm and not is_last:
                layers.append(nn.BatchNorm1d(dim))

            # 激活
            if not is_last:
                if activation is not None:
                    layers.append(activation_layer)
                    nn.init.kaiming_normal_(tmp_layer.weight, nonlinearity=activation)
            else:
                if last_activation is not None:
                    layers.append(last_activation_layer)
                    nn.init.kaiming_normal_(tmp_layer.weight, nonlinearity=last_activation)

            # Dropout
            if dropout > 0 and not is_last:
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        # 可选最后一层 LayerNorm（一般用于做 embedding）
        if use_layernorm:
            layers.append(nn.LayerNorm(prev_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (..., in_dim)
        # 支持任意前导维度（batch / 序列），只要最后一维是特征
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])   # 展平成 (N, in_dim)
        x = self.mlp(x)
        x = x.reshape(*orig_shape[:-1], -1)
        return x

class MixtureOutput(nn.Module):
    def __init__(self, input_shape, n=5, d=1, activation=None, eps=1e-4, bias_mu=1.8, bias_sigma=0.2, name=None):
        super().__init__()
        self.n = n
        self.d = d
        self.activation = activation
        if activation is not None:
            self.activation_fun = get_activation(activation)
        self.eps = eps

        self.alpha = nn.Linear(input_shape[-1], n)
        nn.init.normal_(self.alpha.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.alpha.bias)

        self.mu = nn.Linear(input_shape[-1], n * d)
        nn.init.normal_(self.mu.weight, mean=0.0, std=1e-3)
        mu_bias = torch.full((n * d,), float(bias_mu))
        mu_bias += 1e-3 * torch.randn(n * d)
        with torch.no_grad():
            self.mu.bias.copy_(mu_bias)

        self.sigma = nn.Linear(input_shape[-1], n * d)
        nn.init.normal_(self.sigma.weight, mean=0.0, std=1e-3)
        # softplus_inverse(bias_sigma) so that softplus(bias) ≈ bias_sigma at init
        sigma_bias = torch.full((n * d,), float(math.log(math.expm1(bias_sigma))))
        sigma_bias += 1e-3 * torch.randn(n * d)
        with torch.no_grad():
            self.sigma.bias.copy_(sigma_bias)

    def forward(self, x):
        #alpha = torch.softmax(self.alpha(x), dim=-1).unsqueeze(-1) # (batch, n, 1)
        alpha_logits = self.alpha(x).unsqueeze(-1) # (batch, n, 1)
        mu = self.mu(x).reshape(-1, self.n, self.d)  # (batch, n, d)
        if self.activation is not None:
            mu = self.activation_fun(mu) # (batch, n, d)
        sigma = F.softplus(self.sigma(x)).reshape(-1, self.n, self.d) + self.eps  # (batch, n, d)
        #out = torch.cat([alpha_logits, mu, sigma], dim=-1)  # (batch, n, 1+d+d)
        #return out
        return alpha_logits, mu, sigma



class PointOutput(nn.Module):
    """Deterministic regression output head for full-model point targets."""

    def __init__(self, input_shape, d=1, bias_mu=0.0, activation=None):
        super().__init__()
        self.d = d
        self.activation = activation
        if activation is not None:
            self.activation_fun = get_activation(activation)
        self.mu = nn.Linear(input_shape[-1], d)
        nn.init.normal_(self.mu.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.mu.bias, float(bias_mu))

    def forward(self, x):
        mu = self.mu(x)
        if self.activation is not None:
            mu = self.activation_fun(mu)
        return mu


class NormalizedScaleEmbedding(nn.Module):
    def __init__(self, input_shape, activation='relu', downsample=1, mlp_dims=(500, 300, 200, 150), eps=1e-8):
        super().__init__()
        self.activation = activation
        self.inp_shape = input_shape
        self.downsample = downsample
        self.mlp_dims = mlp_dims
        self.eps = eps

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(downsample, 1), stride=(downsample, 1))
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(16, 3), stride=(1, 3))
        self.conv3 = nn.Conv1d(32 * (input_shape[-1] // 3), 64, kernel_size=16)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=16)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(128, 32, kernel_size=8)
        self.maxpool3 = nn.MaxPool1d(2)
        self.conv6 = nn.Conv1d(32, 32, kernel_size=8)
        self.conv7 = nn.Conv1d(32, 16, kernel_size=4)
        self.flatten = nn.Flatten()
        self.mlp = MLP(input_shape=(865,), dims=self.mlp_dims, activation=activation) #Check this input_shape

    def forward(self, x):
        # Normalization
        x = x / (torch.max(torch.abs(x), dim=(1, 2), keepdim=True)[0] + self.eps)

        # Scale Embedding (Log of max absolute value)
        scale = torch.log(torch.max(torch.abs(x), dim=(1, 2))[0] + self.eps) / 100
        scale = scale.unsqueeze(-1)

        # Convolutional layers
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.reshape(x.size(0), -1, 32 * self.inp_shape[-1] // 3) # Flatten the spatial dims and channels
        x = self.activation(self.conv3(x))
        x = self.maxpool1(x)
        x = self.activation(self.conv4(x))
        x = self.maxpool2(x)
        x = self.activation(self.conv5(x))
        x = self.maxpool3(x)
        x = self.activation(self.conv6(x))
        x = self.activation(self.conv7(x))

        x = self.flatten(x)
        x = torch.cat([x, scale], dim=-1)

        x = self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, max_stations=32, emb_dim=500, layers=6, att_masking=False, hidden_dropout=0.0,
                 mad_params={}, ffn_params={}, norm_params={}):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim=emb_dim, **mad_params, **ffn_params, **norm_params) for _ in range(layers)])
        self.att_masking = att_masking
        self.hidden_dropout = hidden_dropout

    def forward(self, x, att_mask=None, padding_mask=None, coords=None):
        # The inputs are already handled by the calling function.
        self._last_attentions = []
        for block in self.blocks:
            x = block(x, att_mask, padding_mask, coords=coords)
            attn = getattr(block.attention, '_last_attention', None)
            if attn is not None:
                self._last_attentions.append(attn)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, emb_dim, hidden_dim, att_dropout=0.0, initializer_range=0.02, eps=1e-5,
                 use_team_rope=False, rope_coord_mode='relative_xy_km', rope_coord_scale=100.0,
                 rope_base=10000.0, rope_lat_origin=37.0):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            n_heads=n_heads,
            emb_dim=emb_dim,
            att_dropout=att_dropout,
            initializer_range=initializer_range,
            use_team_rope=use_team_rope,
            rope_coord_mode=rope_coord_mode,
            rope_coord_scale=rope_coord_scale,
            rope_base=rope_base,
            rope_lat_origin=rope_lat_origin,
        )
#        self.ffn = PointwiseFeedForward(hidden_dim=hidden_dim)
#        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
                      nn.Linear(emb_dim, hidden_dim),
                      nn.GELU(),
                      nn.Linear(hidden_dim, emb_dim),
        )
#        self.norm1 = LayerNormalization(eps=eps)
#        self.norm2 = LayerNormalization(eps=eps)
        self.norm1 =nn.LayerNorm(emb_dim)
        self.norm2 =nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(0.0) #Fixed dropout
        self.dropout2 = nn.Dropout(0.0) #Fixed dropout

    def forward(self, x, att_mask=None, padding_mask=None, coords=None):
        modified_x, _ = self.attention(x, attn_mask=att_mask, padding_mask=padding_mask, coords=coords)
        modified_x = self.dropout1(modified_x)
        x = self.norm1(x + modified_x)
        modified_x = self.ffn(x)
        modified_x = self.dropout2(modified_x)
        x = self.norm2(x + modified_x)
        return x


class GatedTransformerBlock(nn.Module):
    """Identity-initialized station self-attention block for context refinement."""

    def __init__(self, n_heads, emb_dim, hidden_dim, att_dropout=0.0, initializer_range=0.02, eps=1e-5,
                 use_team_rope=False, rope_coord_mode='relative_xy_km', rope_coord_scale=100.0,
                 rope_base=10000.0, rope_lat_origin=37.0,
                 residual_gate_init=0.0, ffn_gate_init=None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            n_heads=n_heads,
            emb_dim=emb_dim,
            att_dropout=att_dropout,
            initializer_range=initializer_range,
            use_team_rope=use_team_rope,
            rope_coord_mode=rope_coord_mode,
            rope_coord_scale=rope_coord_scale,
            rope_base=rope_base,
            rope_lat_origin=rope_lat_origin,
        )
        self.attn_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        if ffn_gate_init is None:
            ffn_gate_init = residual_gate_init
        self.ffn_gate = nn.Parameter(torch.tensor(float(ffn_gate_init)))
        self.ffn_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x, att_mask=None, padding_mask=None, coords=None):
        modified_x, _ = self.attention(x, attn_mask=att_mask, padding_mask=padding_mask, coords=coords)
        x = x + self.attn_gate * modified_x
        x = x + self.ffn_gate * self.ffn(self.ffn_norm(x))
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)
        return x


class StationContextEncoder(nn.Module):
    """Station-only context encoder with first residual then gated refinements."""

    def __init__(self, emb_dim, layers=2, mad_params=None, ffn_params=None,
                 residual_gate_init=0.0, ffn_gate_init=0.0):
        super().__init__()
        mad_params = dict(mad_params or {})
        ffn_params = dict(ffn_params or {})
        layers = int(layers)
        if layers < 1:
            raise ValueError(f'station context layers must be >= 1, got {layers}')
        self.first_block = TransformerBlock(emb_dim=emb_dim, **mad_params, **ffn_params)
        self.extra_blocks = nn.ModuleList([
            GatedTransformerBlock(
                emb_dim=emb_dim,
                residual_gate_init=residual_gate_init,
                ffn_gate_init=ffn_gate_init,
                **mad_params,
                **ffn_params,
            )
            for _ in range(layers - 1)
        ])
        self._last_attentions = []

    def forward(self, x, att_mask=None, padding_mask=None, coords=None):
        self._last_attentions = []
        x = self.first_block(x, att_mask=att_mask, padding_mask=padding_mask, coords=coords)
        attn = getattr(self.first_block.attention, '_last_attention', None)
        if attn is not None:
            self._last_attentions.append(attn)
        for block in self.extra_blocks:
            x = block(x, att_mask=att_mask, padding_mask=padding_mask, coords=coords)
            attn = getattr(block.attention, '_last_attention', None)
            if attn is not None:
                self._last_attentions.append(attn)
        return x


class CrossAttentionRefinementBlock(nn.Module):
    """Additional cross-attention refinement layer for query readouts."""

    def __init__(self, emb_dim, n_heads, att_dropout=0.0, distance_bias=False,
                 distance_hidden_dim=64, ffn_hidden_dim=None,
                 residual_gates=False, residual_gate_init=0.0,
                 ffn_gate_init=None, inject_base_query=False,
                 query_injection_gate_init=1.0, use_ffn=True,
                 use_rope=False, rope_coord_mode='relative_xy_km',
                 rope_coord_scale=100.0, rope_base=10000.0,
                 rope_lat_origin=37.0):
        super().__init__()
        self.residual_gates = bool(residual_gates)
        self.inject_base_query = bool(inject_base_query)
        self.use_ffn = bool(use_ffn)
        self.use_rope = bool(use_rope)
        self.rope_coord_mode = rope_coord_mode
        self.rope_coord_scale = float(rope_coord_scale)
        self.rope_base = float(rope_base)
        self.rope_lat_origin = float(rope_lat_origin)
        self.query_norm = nn.LayerNorm(emb_dim)
        self.kv_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=att_dropout,
            batch_first=True,
        )
        self.attn_out_norm = nn.LayerNorm(emb_dim)
        if self.residual_gates:
            self.attn_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
            if ffn_gate_init is None:
                ffn_gate_init = residual_gate_init
            self.ffn_gate = nn.Parameter(torch.tensor(float(ffn_gate_init)))
        else:
            self.attn_gate = None
            self.ffn_gate = None
        if self.inject_base_query:
            self.query_injection_gate = nn.Parameter(torch.tensor(float(query_injection_gate_init)))
        else:
            self.query_injection_gate = None
        if ffn_hidden_dim is None:
            ffn_hidden_dim = emb_dim
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, int(ffn_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(ffn_hidden_dim), emb_dim),
            )
        else:
            self.ffn = None
        self.ffn_norm = nn.LayerNorm(emb_dim)
        self.distance_bias = bool(distance_bias)
        if self.distance_bias:
            self.distance_mlp = nn.Sequential(
                nn.Linear(4, distance_hidden_dim),
                nn.GELU(),
                nn.Linear(distance_hidden_dim, 1),
            )
            nn.init.zeros_(self.distance_mlp[-1].weight)
            nn.init.zeros_(self.distance_mlp[-1].bias)
        else:
            self.distance_mlp = None

    def _distance_attn_mask(self, query_coords, station_coords):
        if self.distance_mlp is None:
            return None
        if query_coords is None or station_coords is None:
            raise ValueError('query_coords and station_coords are required when distance_bias=True.')
        rel = query_coords[:, :, None, :] - station_coords[:, None, :, :]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        geom = torch.cat([rel, dist], dim=-1)
        bias = self.distance_mlp(geom).squeeze(-1)
        return bias.repeat_interleave(self.attn.num_heads, dim=0)

    def _cross_attention(self, q, kv, station_valid, attn_mask, query_coords, station_coords):
        if self.use_rope:
            return _cross_attention_with_rope(
                self.attn,
                q,
                kv,
                station_valid,
                attn_mask=attn_mask,
                query_coords=query_coords,
                key_coords=station_coords,
                rope_coord_mode=self.rope_coord_mode,
                rope_coord_scale=self.rope_coord_scale,
                rope_base=self.rope_base,
                rope_lat_origin=self.rope_lat_origin,
            )
        return self.attn(
            q,
            kv,
            kv,
            key_padding_mask=~station_valid.bool(),
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )

    def forward(self, query, station_emb, station_valid, query_coords=None, station_coords=None,
                base_query=None):
        query_for_attention = query
        if self.query_injection_gate is not None and base_query is not None:
            query_for_attention = query_for_attention + self.query_injection_gate * base_query
        q = self.query_norm(query_for_attention)
        kv = self.kv_norm(station_emb)
        attn_mask = self._distance_attn_mask(query_coords, station_coords)
        out, attn = self._cross_attention(q, kv, station_valid, attn_mask, query_coords, station_coords)
        if self.attn_gate is not None:
            query = query + self.attn_gate * out
            if self.ffn is not None:
                query = query + self.ffn_gate * self.ffn(self.ffn_norm(query))
        else:
            query = self.attn_out_norm(query + out)
            if self.ffn is not None:
                query = self.ffn_norm(query + self.ffn(query))
        return query, attn


class CrossAttentionReadout(nn.Module):
    """Single-direction query-to-station readout.

    The output values come from station tokens only. Query embeddings control
    the attention weights but are not added back as a residual, which makes this
    readout a cleaner diagnostic for whether station information is usable.
    """

    def __init__(self, emb_dim, n_heads, att_dropout=0.0, distance_bias=False,
                 distance_hidden_dim=64, readout_layers=1, ffn_hidden_dim=None,
                 first_residual=False, first_residual_gate_init=None,
                 residual_gates=False, residual_gate_init=0.0,
                 ffn_gate_init=None, inject_base_query=False,
                 query_injection_gate_init=1.0, use_ffn=True,
                 use_rope=False, rope_coord_mode='relative_xy_km',
                 rope_coord_scale=100.0, rope_base=10000.0,
                 rope_lat_origin=37.0):
        super().__init__()
        readout_layers = int(readout_layers)
        if readout_layers < 1:
            raise ValueError(f'readout_layers must be >= 1, got {readout_layers}')
        self.readout_layers = readout_layers
        self.first_residual = bool(first_residual)
        self.use_rope = bool(use_rope)
        self.rope_coord_mode = rope_coord_mode
        self.rope_coord_scale = float(rope_coord_scale)
        self.rope_base = float(rope_base)
        self.rope_lat_origin = float(rope_lat_origin)
        self.query_norm = nn.LayerNorm(emb_dim)
        self.kv_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=att_dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(emb_dim)
        if first_residual_gate_init is None:
            self.first_residual_gate = None
        else:
            self.first_residual_gate = nn.Parameter(torch.tensor(float(first_residual_gate_init)))
        self.distance_bias = bool(distance_bias)
        if self.distance_bias:
            self.distance_mlp = nn.Sequential(
                nn.Linear(4, distance_hidden_dim),
                nn.GELU(),
                nn.Linear(distance_hidden_dim, 1),
            )
            nn.init.zeros_(self.distance_mlp[-1].weight)
            nn.init.zeros_(self.distance_mlp[-1].bias)
        else:
            self.distance_mlp = None
        self.extra_layers = nn.ModuleList([
            CrossAttentionRefinementBlock(
                emb_dim,
                n_heads,
                att_dropout=att_dropout,
                distance_bias=distance_bias,
                distance_hidden_dim=distance_hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                residual_gates=residual_gates,
                residual_gate_init=residual_gate_init,
                ffn_gate_init=ffn_gate_init,
                inject_base_query=inject_base_query,
                query_injection_gate_init=query_injection_gate_init,
                use_ffn=use_ffn,
                use_rope=use_rope,
                rope_coord_mode=rope_coord_mode,
                rope_coord_scale=rope_coord_scale,
                rope_base=rope_base,
                rope_lat_origin=rope_lat_origin,
            )
            for _ in range(readout_layers - 1)
        ])
        self._last_attention = None
        self._last_attentions = []

    def _distance_attn_mask(self, query_coords, station_coords):
        if self.distance_mlp is not None:
            if query_coords is None or station_coords is None:
                raise ValueError('query_coords and station_coords are required when distance_bias=True.')
            rel = query_coords[:, :, None, :] - station_coords[:, None, :, :]
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            geom = torch.cat([rel, dist], dim=-1)
            bias = self.distance_mlp(geom).squeeze(-1)
            return bias.repeat_interleave(self.attn.num_heads, dim=0)
        return None

    def _cross_attention(self, q, kv, station_valid, attn_mask, query_coords, station_coords):
        if self.use_rope:
            return _cross_attention_with_rope(
                self.attn,
                q,
                kv,
                station_valid,
                attn_mask=attn_mask,
                query_coords=query_coords,
                key_coords=station_coords,
                rope_coord_mode=self.rope_coord_mode,
                rope_coord_scale=self.rope_coord_scale,
                rope_base=self.rope_base,
                rope_lat_origin=self.rope_lat_origin,
            )
        return self.attn(
            q,
            kv,
            kv,
            key_padding_mask=~station_valid.bool(),
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )

    def forward(self, query, station_emb, station_valid, query_coords=None, station_coords=None):
        q = self.query_norm(query)
        kv = self.kv_norm(station_emb)
        attn_mask = self._distance_attn_mask(query_coords, station_coords)
        out, attn = self._cross_attention(q, kv, station_valid, attn_mask, query_coords, station_coords)
        base_query = query
        if self.first_residual:
            if self.first_residual_gate is None:
                query = self.out_norm(query + out)
            else:
                query = self.out_norm(query + self.first_residual_gate * out)
        else:
            query = self.out_norm(out)
        attentions = [attn.detach()]
        for layer in self.extra_layers:
            query, attn = layer(
                query,
                station_emb,
                station_valid,
                query_coords=query_coords,
                station_coords=station_coords,
                base_query=base_query,
            )
            attentions.append(attn.detach())
        self._last_attention = attentions[-1]
        self._last_attentions = attentions
        return query


class SynchronousStationTargetReadout(nn.Module):
    """Evolve station memory and target query in parallel without target-target attention."""

    def __init__(self, emb_dim, n_heads, station_layers=1, mad_params=None, ffn_params=None,
                 att_dropout=0.0, distance_bias=False, distance_hidden_dim=64,
                 ffn_hidden_dim=None, first_residual=False, first_residual_gate_init=None,
                 residual_gates=False, residual_gate_init=0.0, ffn_gate_init=None,
                 inject_base_query=False, query_injection_gate_init=1.0, use_ffn=True,
                 use_rope=False, rope_coord_mode='relative_xy_km',
                 rope_coord_scale=100.0, rope_base=10000.0, rope_lat_origin=37.0):
        super().__init__()
        station_layers = int(station_layers)
        if station_layers < 1:
            raise ValueError(f'station_layers must be >= 1, got {station_layers}')
        mad_params = dict(mad_params or {})
        ffn_params = dict(ffn_params or {})
        self.readout_layers = station_layers
        self.first_residual = bool(first_residual)
        self.station_first_block = TransformerBlock(emb_dim=emb_dim, **mad_params, **ffn_params)
        self.station_extra_blocks = nn.ModuleList([
            GatedTransformerBlock(
                emb_dim=emb_dim,
                residual_gate_init=residual_gate_init,
                ffn_gate_init=ffn_gate_init,
                **mad_params,
                **ffn_params,
            )
            for _ in range(station_layers - 1)
        ])
        self.target_first_readout = CrossAttentionReadout(
            emb_dim,
            n_heads,
            att_dropout=att_dropout,
            distance_bias=distance_bias,
            distance_hidden_dim=distance_hidden_dim,
            readout_layers=1,
            ffn_hidden_dim=ffn_hidden_dim,
            first_residual=first_residual,
            first_residual_gate_init=first_residual_gate_init,
            residual_gates=residual_gates,
            residual_gate_init=residual_gate_init,
            ffn_gate_init=ffn_gate_init,
            inject_base_query=inject_base_query,
            query_injection_gate_init=query_injection_gate_init,
            use_ffn=use_ffn,
            use_rope=use_rope,
            rope_coord_mode=rope_coord_mode,
            rope_coord_scale=rope_coord_scale,
            rope_base=rope_base,
            rope_lat_origin=rope_lat_origin,
        )
        self.extra_layers = nn.ModuleList([
            CrossAttentionRefinementBlock(
                emb_dim,
                n_heads,
                att_dropout=att_dropout,
                distance_bias=distance_bias,
                distance_hidden_dim=distance_hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                residual_gates=residual_gates,
                residual_gate_init=residual_gate_init,
                ffn_gate_init=ffn_gate_init,
                inject_base_query=inject_base_query,
                query_injection_gate_init=query_injection_gate_init,
                use_ffn=use_ffn,
                use_rope=use_rope,
                rope_coord_mode=rope_coord_mode,
                rope_coord_scale=rope_coord_scale,
                rope_base=rope_base,
                rope_lat_origin=rope_lat_origin,
            )
            for _ in range(station_layers - 1)
        ])
        self.first_residual_gate = self.target_first_readout.first_residual_gate
        self._last_attention = None
        self._last_attentions = []
        self._last_station_memory = None
        self._last_station_attentions = []

    def forward(self, query, station_emb, station_valid, query_coords=None, station_coords=None):
        station_state = station_emb
        base_query = query
        target_attentions = []
        station_attentions = []

        station_state = self.station_first_block(
            station_state,
            padding_mask=station_valid,
            coords=station_coords,
        )
        attn = getattr(self.station_first_block.attention, '_last_attention', None)
        if attn is not None:
            station_attentions.append(attn)
        query = self.target_first_readout(
            query,
            station_state,
            station_valid,
            query_coords=query_coords,
            station_coords=station_coords,
        )
        target_attentions.extend(getattr(self.target_first_readout, '_last_attentions', []))

        for station_block, target_block in zip(self.station_extra_blocks, self.extra_layers):
            station_state = station_block(
                station_state,
                padding_mask=station_valid,
                coords=station_coords,
            )
            attn = getattr(station_block.attention, '_last_attention', None)
            if attn is not None:
                station_attentions.append(attn)
            query, attn = target_block(
                query,
                station_state,
                station_valid,
                query_coords=query_coords,
                station_coords=station_coords,
                base_query=base_query,
            )
            target_attentions.append(attn.detach())

        self._last_station_memory = station_state
        self._last_station_attentions = station_attentions
        self._last_attentions = target_attentions
        self._last_attention = target_attentions[-1] if target_attentions else None
        return query


# Calculates and concatenates sinusoidal embeddings for lat, lon and depth
# Note: Permutation is completely unnecessary, but kept for compatibility reasons
# WARNING: Does not take into account curvature of the earth!
class PositionEmbedding(nn.Module):
    def __init__(self, wavelengths, emb_dim, borehole=False, rotation=None, rotation_anchor=None):
        super().__init__()
        self.wavelengths = wavelengths  # Format: [(min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth)]
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError('Rotations in the positional embedding require a rotation anchor')

        if rotation is not None:
            # print(f'Rotating by {np.rad2deg(rotation)} degrees')
            c, s = np.cos(rotation), np.sin(rotation)
#            self.rotation_matrix = torch.tensor(((c, -s), (s, c)), dtype=torch.float32) #K.variable replaced with tensor
            self.register_buffer('rotation_matrix', torch.tensor(((c, -s), (s, c)), dtype=torch.float32)) #K.variable replaced with tensor
        else:
            self.rotation_matrix = None

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]
        assert emb_dim % 10 == 0
        if borehole:
            assert emb_dim % 20 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        if borehole:
            depth_dim = emb_dim // 20

        lat_coeff = 2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        lon_coeff = 2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        depth_coeff = 2 * np.pi * 1. / min_depth * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        self.register_buffer('lat_coeff', torch.tensor(lat_coeff, dtype=torch.float32))
        self.register_buffer('lon_coeff', torch.tensor(lon_coeff, dtype=torch.float32))
        self.register_buffer('depth_coeff', torch.tensor(depth_coeff, dtype=torch.float32))

        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9
        self.mask = np.zeros(emb_dim)
        #self.register_buffer('mask', np.zeros(emb_dim))
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim *= 2
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = 2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        self.mask = torch.tensor(self.mask.astype('int32'), dtype=torch.long)  # Crucial: Convert to tensor

        self.fake_borehole = False

    def forward(self, x, mask=None):
        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1]
            lon_base *= torch.cos(lat_base * np.pi / 180)

            lat_base -= self.rotation_anchor[0]
            lon_base -= self.rotation_anchor[1] * np.cos(self.rotation_anchor[0] * np.pi / 180)

            latlon = torch.stack([lat_base, lon_base], dim=-1)
            rotated = latlon @ self.rotation_matrix

            lat_coeff = self.lat_coeff.to(device=x.device, dtype=x.dtype)
            lon_coeff = self.lon_coeff.to(device=x.device, dtype=x.dtype)
            depth_coeff = self.depth_coeff.to(device=x.device, dtype=x.dtype)
            lat_base = rotated[:, :, 0:1] * lat_coeff
            lon_base = rotated[:, :, 1:2] * lon_coeff
            depth_base = x[:, :, 2:3] * depth_coeff
        else:
            lat_coeff = self.lat_coeff.to(device=x.device, dtype=x.dtype)
            lon_coeff = self.lon_coeff.to(device=x.device, dtype=x.dtype)
            depth_coeff = self.depth_coeff.to(device=x.device, dtype=x.dtype)
            lat_base = x[:, :, 0:1] * lat_coeff
            lon_base = x[:, :, 1:2] * lon_coeff
            depth_base = x[:, :, 2:3] * depth_coeff

        if self.borehole:
            if self.fake_borehole:
                # Use third value for the depth of the top station and 0 for the borehole depth
                depth_base = x[:, :, 2:3] * depth_coeff * 0
                depth2_base = x[:, :, 2:3] * depth_coeff
            else:
                depth2_base = x[:, :, 3:4] * depth_coeff

            output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                                torch.sin(lon_base), torch.cos(lon_base),
                                torch.sin(depth_base), torch.cos(depth_base),
                                torch.sin(depth2_base), torch.cos(depth2_base)], dim=-1)
        else:
            output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                                torch.sin(lon_base), torch.cos(lon_base),
                                torch.sin(depth_base), torch.cos(depth_base)], dim=-1)

        output = torch.gather(output, dim=-1, index=self.mask.to(x.device).expand(x.size(0), x.size(1), -1))  # Move mask to the device

        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # Ensure float type for multiplication
            output *= mask  # Zero out all masked elements

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, initializer_range=0.02, att_dropout=0.0, infinity=1e6,
                 use_team_rope=False, rope_coord_mode='relative_xy_km', rope_coord_scale=100.0,
                 rope_base=10000.0, rope_lat_origin=37.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = int(emb_dim // n_heads)
        self.infinity = infinity
        self.att_dropout = att_dropout
        self.initializer_range = initializer_range #Added
        self.use_team_rope = bool(use_team_rope)
        self.rope_coord_mode = rope_coord_mode
        self.rope_coord_scale = float(rope_coord_scale)
        self.rope_base = float(rope_base)
        self.rope_lat_origin = float(rope_lat_origin)
        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)
        self.WO = nn.Linear(emb_dim, emb_dim)

    @staticmethod
    def _rotate_pairs(x, angles):
        pair_count = angles.shape[-1]
        x_pair = x.reshape(*x.shape[:-1], pair_count, 2)
        cos = torch.cos(angles)[:, None, :, :, None]
        sin = torch.sin(angles)[:, None, :, :, None]
        x0 = x_pair[..., 0:1]
        x1 = x_pair[..., 1:2]
        rotated = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return rotated.reshape(*x.shape[:-1], pair_count * 2)

    def _relative_xy_for_rope(self, coords, padding_mask=None):
        coords2 = coords[:, :, :2].to(dtype=torch.float32)
        if padding_mask is None:
            valid = torch.ones(coords2.shape[:2], device=coords2.device, dtype=coords2.dtype)
        else:
            valid = padding_mask.to(device=coords2.device, dtype=coords2.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        center = (coords2 * valid[:, :, None]).sum(dim=1, keepdim=True) / denom[:, :, None]

        if self.rope_coord_mode == 'relative_xy_model_units':
            xy = coords2 - center
            return xy[:, :, 1], xy[:, :, 0]
        if self.rope_coord_mode != 'relative_xy_km':
            raise ValueError(
                "rope_coord_mode must be 'relative_xy_km' or 'relative_xy_model_units', "
                f"got {self.rope_coord_mode!r}."
            )

        rel_deg = coords2 - center
        center_lat_deg = center[:, :, 0] + self.rope_lat_origin
        lon_scale = torch.cos(center_lat_deg * math.pi / 180.0).clamp_min(0.1)
        y_km = rel_deg[:, :, 0] * 111.19492664455874
        x_km = rel_deg[:, :, 1] * 111.19492664455874 * lon_scale.squeeze(1)[:, None]
        return x_km, y_km

    def _apply_team_rope(self, q, k, coords, padding_mask=None):
        if not self.use_team_rope or coords is None:
            return q, k
        rotary_dim = (self.d_key // 4) * 4
        if rotary_dim < 4:
            return q, k
        pair_count_per_axis = rotary_dim // 4
        freq_idx = torch.arange(pair_count_per_axis, device=q.device, dtype=q.dtype)
        inv_freq = 1.0 / (self.rope_base ** (freq_idx / max(1, pair_count_per_axis)))
        x_coord, y_coord = self._relative_xy_for_rope(coords.to(q.device), padding_mask=padding_mask)
        x_angles = (x_coord.to(dtype=q.dtype) / self.rope_coord_scale)[:, :, None] * inv_freq[None, None, :]
        y_angles = (y_coord.to(dtype=q.dtype) / self.rope_coord_scale)[:, :, None] * inv_freq[None, None, :]

        x_slice = slice(0, 2 * pair_count_per_axis)
        y_slice = slice(2 * pair_count_per_axis, 4 * pair_count_per_axis)

        def apply_one(tensor):
            out = tensor.clone()
            out[..., x_slice] = self._rotate_pairs(out[..., x_slice], x_angles)
            out[..., y_slice] = self._rotate_pairs(out[..., y_slice], y_angles)
            return out

        return apply_one(q), apply_one(k)

    def forward(self, x, attn_mask=None, padding_mask=None, coords=None):
        self.stations = x.shape[1]

        d_key = self.d_key
        n_heads = self.n_heads
        stations = self.stations

        q = self.WQ(x)  # (batch, stations, key*n_heads)
        q = q.reshape(-1, stations, d_key, n_heads)
        q = q.permute(0, 3, 1, 2)  # (batch, n_heads, stations, key)

        k = self.WK(x)  # (batch, stations, key*n_heads)
        k = k.reshape(-1, stations, d_key, n_heads)
        k = k.permute(0, 3, 1, 2)  # (batch, n_heads, stations, key)
        q, k = self._apply_team_rope(q, k, coords=coords, padding_mask=padding_mask)

        score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_key)  # (batch, n_heads, stations, stations)

        # Key masking: prevent attending to padding stations (from padding_mask)
        # Matches TF lines 268-271: score -= ~mask * infinity
        if padding_mask is not None:
            inv_mask = (~padding_mask).float()[:, None, None, :]  # (batch, 1, 1, stations)
            score = score - inv_mask * self.infinity

        # Additional key masking for PGA-only positions (from attn_mask)
        # Matches TF lines 272-276: score -= ~att_mask * infinity
        if attn_mask is not None:
            inv_mask = (~attn_mask).float()[:, None, None, :]  # (batch, 1, 1, stations)
            score = score - inv_mask * self.infinity

        score = torch.softmax(score, dim=-1) #Softmax on the last dimension
        self._last_attention = score.detach()
        if self.att_dropout > 0:
            score = F.dropout(score, p=self.att_dropout, training=self.training)

        v = self.WV(x)  # (batch, stations, key*n_heads)
        v = v.reshape(-1, stations, d_key, n_heads)
        v = v.permute(0, 3, 1, 2)  # (batch, n_heads, stations, key)

        o = torch.matmul(score, v)  # (batch, n_heads, stations, key)
        o = o.permute(0, 2, 1, 3)  # (batch, stations, n_heads, key)
        o = o.reshape(-1, stations, n_heads * d_key)
        o = self.WO(o)

        # Output masking: zero padding positions
        if padding_mask is not None:
            mask_float = padding_mask.unsqueeze(-1).float()  # (batch, stations, 1)
            o = o * mask_float

        return o, None


class PointwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, initializer='glorot_uniform', bias_initializer='zeros'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel1 = None
        self.bias1 = None
        self.kernel2 = None
        self.bias2 = None

    def build(self, input_shape):
        self.kernel1 = nn.Linear(input_shape[-1], self.hidden_dim)
        self.bias1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.kernel2 = nn.Linear(self.hidden_dim, input_shape[-1])
        self.bias2 = nn.Parameter(torch.zeros(input_shape[-1]))

        # Use torch.nn.init for initialization
        nn.init.xavier_uniform_(self.kernel1.weight)  # Or other initialization methods
        nn.init.zeros_(self.kernel1.bias)  # Or other initialization methods
        nn.init.xavier_uniform_(self.kernel2.weight)  # Or other initialization methods
        nn.init.zeros_(self.kernel2.bias)  # Or other initialization methods

    def forward(self, x):
        if self.kernel1 is None:
             self.build(x.shape)

        x = gelu(self.kernel1(x) + self.bias1)
        x = self.kernel2(x) + self.bias2
        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
         self.beta = nn.Parameter(torch.zeros(input_shape[-1]))
         self.gamma = nn.Parameter(torch.ones(input_shape[-1]))

    def forward(self, x):
        if self.beta is None:
             self.build(x.shape)
             self.beta.to(x.device)
             self.gamma.to(x.device)

        m = torch.mean(x, dim=-1, keepdim=True)
        s = torch.mean((x - m)**2, dim=-1, keepdim=True)
        z = (x - m) / torch.sqrt(s + self.eps)
        output = self.gamma * z + self.beta
        return output


class AddEventToken(nn.Module):
    def __init__(self, fixed=False, init_range=None):
        super().__init__()
        self.fixed = fixed
        self.init_range = init_range
        self.emb = None

    def build(self, x):
        input_shape = x.shape
        device = x.device
        if not self.fixed:
            if self.init_range is None:
                self.emb = nn.Parameter(torch.ones(input_shape[-1])).to(device)
            else:
                self.emb = nn.Parameter(torch.rand(input_shape[-1]) * 2 * self.init_range - self.init_range).to(device)

    def forward(self, x, mask=None):
        if self.emb is None and not self.fixed:
             self.build(x)

        pad = torch.ones_like(x[:, :1, :]).to(x.device)
        if self.emb is not None:
            pad = pad * self.emb
        x = torch.cat([pad, x], dim=1)
        return x

class AddConstantToMixture(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mix, const, mask=None):
        const = const.unsqueeze(-1)
        alpha = mix[:, :, 0]
        mu = mix[:, :, 1] + const
        sigma = mix[:, :, 2]
        output = torch.stack([alpha, mu, sigma], dim=-1)
        if mask is not None:
            while mask.ndim < output.ndim:
                mask = mask.unsqueeze(-1)
            output *= mask.float()  # Ensure mask is float
        return output


class Masking_nd(nn.Module):
    def __init__(self, mask_value=0., axis=-1, nodim=False, eps=1e-7):
        super().__init__()
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim
        self.eps = eps

    def forward(self, inputs, mask=None):
        if self.nodim:
            boolean_mask = (torch.abs(inputs - self.mask_value) > self.eps)
        else:
            boolean_mask = torch.any((torch.abs(inputs - self.mask_value) > self.eps), dim=self.axis, keepdim=True)

        return inputs * boolean_mask.float()

    def compute_mask(self, inputs, mask=None):
        if self.nodim:
            output_mask = (torch.abs(inputs - self.mask_value) > self.eps)
        else:
            output_mask = torch.any((torch.abs(inputs - self.mask_value) > self.eps), dim=self.axis)
        return output_mask


class StripMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x



class AttentionPool1d(nn.Module):
    """Learned multi-query attention pooling over temporal/token features."""

    def __init__(self, channels, num_queries=4, temperature=1.0, dropout=0.0):
        super().__init__()
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        self.channels = channels
        self.num_queries = num_queries
        self.temperature = float(temperature)
        self.dropout = float(dropout)
        self.query = nn.Parameter(torch.empty(num_queries, channels))
        self.key_norm = nn.LayerNorm(channels)
        self.out_norm = nn.LayerNorm(channels)
        self._last_attention = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.ones_(self.key_norm.weight)
        nn.init.zeros_(self.key_norm.bias)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)

    def forward(self, x):
        # x: (B, C, T). Return Q pooled vectors flattened to (B, Q*C).
        tokens = x.transpose(1, 2).contiguous().float()  # (B, T, C)
        keys = self.key_norm(tokens)
        scores = torch.einsum('btc,qc->btq', keys, self.query)
        scores = scores / max(self.temperature, 1e-6)
        weights = torch.softmax(scores, dim=1)
        if self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.einsum('btq,btc->bqc', weights, tokens)
        context = self.out_norm(context)
        self._last_attention = weights.detach()
        return context.reshape(context.shape[0], self.num_queries * self.channels)


class WaveformScaleEmbedding(nn.Module):
    """Embed per-station amplitude statistics without erasing absolute scale."""

    def __init__(self, input_dim, emb_dim, hidden_dim=None, log_divisor=10.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = min(emb_dim, max(32, input_dim * 4))
        self.log_divisor = float(log_divisor)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features / self.log_divisor)


def extract_waveform_scale_features(waveform):
    """Return per-station log-amplitude statistics for (B,C,T) or (B,S,C,T)."""
    eps = 1e-6
    squeeze_station_dim = False
    if waveform.dim() == 3:
        waveform = waveform.unsqueeze(1)
        squeeze_station_dim = True
    elif waveform.dim() != 4:
        raise ValueError(
            f"Expected waveform shape (B,C,T) or (B,S,C,T), got {tuple(waveform.shape)}"
        )

    waveform_f = waveform.float()
    abs_waveform = torch.abs(waveform_f)
    log_std_ch = torch.log(torch.std(waveform_f, dim=3, unbiased=False).clamp_min(eps))
    log_rms_ch = torch.log(torch.sqrt(torch.mean(waveform_f ** 2, dim=3).clamp_min(eps ** 2)))
    log_peak_ch = torch.log(torch.amax(abs_waveform, dim=3).clamp_min(eps))
    log_rms_all = torch.log(torch.sqrt(torch.mean(waveform_f ** 2, dim=(2, 3)).clamp_min(eps ** 2)))
    log_peak_all = torch.log(torch.amax(abs_waveform, dim=(2, 3)).clamp_min(eps))
    features = torch.cat(
        [log_std_ch, log_rms_ch, log_peak_ch, log_rms_all.unsqueeze(-1), log_peak_all.unsqueeze(-1)],
        dim=-1,
    )
    if squeeze_station_dim:
        features = features[:, 0, :]
    return features


class AuxiliaryMagnitudeHead(nn.Module):
    """Few-shot-style event magnitude head wrapped as a one-component mixture."""

    def __init__(self, emb_dim, hidden_dim=None, sigma_init=0.3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_sigma = nn.Parameter(torch.tensor(float(math.log(math.expm1(sigma_init)))))

    def forward(self, station_emb, station_valid):
        weights = station_valid.unsqueeze(-1).to(station_emb.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (station_emb * weights).sum(dim=1) / denom
        mu = torch.sigmoid(self.net(pooled)).unsqueeze(1) * 9.0
        alpha_logits = torch.zeros_like(mu)
        sigma = F.softplus(self.log_sigma).to(mu.dtype).expand_as(mu) + 1e-4
        return torch.cat([alpha_logits, mu, sigma], dim=-1)


class SingleStationRegressionHead(nn.Module):
    """Small point-regression head used only for single-station pretraining."""

    def __init__(self, emb_dim, hidden_dim=None, output_init=0.0, sigma_init=1.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, float(output_init))
        self.log_sigma = nn.Parameter(torch.tensor(float(math.log(math.expm1(sigma_init)))))

    def forward(self, x):
        mu = self.net(x).squeeze(-1)
        sigma = F.softplus(self.log_sigma).to(mu.dtype).expand_as(mu) + 1e-4
        return torch.stack([mu, sigma], dim=-1)


class SingleStationMultiTaskModel(nn.Module):
    """Pretrain station representations with station-level auxiliary tasks."""

    def __init__(self, waveform_model, emb_dim, input_channels=3, tasks=('mag', 'epidist', 'pga'),
                 waveform_scale_proj=None, waveform_scale_gain=1.0, waveform_scale_init_gate=0.1,
                 disable_waveform_scale=False, use_amplitude_info=None, task_hidden_dim=None,
                 task_output_init=None, task_sigma_init=None):
        super().__init__()
        self.waveform_model = waveform_model
        self.emb_dim = emb_dim
        self.input_channels = input_channels
        self.tasks = list(tasks)
        self.waveform_scale_proj = waveform_scale_proj
        self.waveform_scale_gain = waveform_scale_gain
        self.waveform_scale_gate = nn.Parameter(torch.tensor(float(waveform_scale_init_gate)))
        if use_amplitude_info is None:
            use_amplitude_info = not disable_waveform_scale
        self.use_amplitude_info = bool(use_amplitude_info)
        self.disable_waveform_scale = not self.use_amplitude_info
        self.layernorm = nn.LayerNorm(emb_dim)

        default_output_init = {
            'mag': 5.0,
            'epidist': 4.0,
            'pga': 0.0,
        }
        default_sigma_init = {
            'mag': 0.5,
            'epidist': 1.0,
            'pga': 1.0,
        }
        task_output_init = {**default_output_init, **(task_output_init or {})}
        task_sigma_init = {**default_sigma_init, **(task_sigma_init or {})}
        self.heads = nn.ModuleDict({
            task: SingleStationRegressionHead(
                emb_dim,
                hidden_dim=task_hidden_dim,
                output_init=task_output_init.get(task, 0.0),
                sigma_init=task_sigma_init.get(task, 1.0),
            )
            for task in self.tasks
        })
        self._last_diag = {}

    def _normalize(self, data, mode='std', axis=2):
        data = data - torch.mean(data, axis=axis, keepdims=True)
        if mode == 'std':
            std_data = torch.std(data, axis=axis, keepdims=True)
            std_data[std_data == 0] = 1
            return data / std_data
        if mode == 'max':
            max_data = torch.max(torch.abs(data), axis=axis, keepdims=True)[0]
            max_data[max_data == 0] = 1
            return data / max_data
        if mode == '':
            return data
        raise ValueError(f"Supported mode: 'max','std', got '{mode}'")

    @staticmethod
    def _mean_token_norm(x):
        return x.norm(dim=-1).mean()

    def encode_station(self, waveform):
        raw_waveform = waveform.clone()
        waveform = self._normalize(waveform, mode='std', axis=2)
        raw_station_emb = self.waveform_model(waveform)
        station_emb = self.layernorm(raw_station_emb)
        base_station_emb = station_emb
        if self.waveform_scale_proj is not None and self.use_amplitude_info:
            scale_features = extract_waveform_scale_features(raw_waveform)
            scale_emb = self.waveform_scale_proj(scale_features)
            gain_scale_emb = self.waveform_scale_gain * self.waveform_scale_gate * scale_emb
            station_emb = station_emb + gain_scale_emb
        else:
            scale_features = None
            scale_emb = None
            gain_scale_emb = None

        self._last_diag = {
            'station_adapter_raw_norm': self._mean_token_norm(raw_station_emb).detach(),
            'wave_emb_norm': self._mean_token_norm(station_emb).detach(),
        }
        if scale_emb is not None:
            scale_norm = self._mean_token_norm(scale_emb)
            gain_scale_norm = self._mean_token_norm(gain_scale_emb)
            raw_trunk_norm = self._mean_token_norm(base_station_emb)
            self._last_diag.update({
                'scale_emb_norm': scale_norm.detach(),
                'gain_scale_emb_norm': gain_scale_norm.detach(),
                'scale_trunk_ratio': (gain_scale_norm / (raw_trunk_norm + 1e-8)).detach(),
                'scale_gate': self.waveform_scale_gate.detach(),
                'scale_feature_mean': scale_features.mean().detach(),
                'scale_feature_std': scale_features.std(unbiased=False).detach(),
            })
        return station_emb

    def forward(self, waveform):
        station_emb = self.encode_station(waveform)
        outputs = {'embedding': station_emb}
        for task, head in self.heads.items():
            outputs[task] = head(station_emb)
        return outputs


class DitingStationAdapter(nn.Module):
    """Light multi-scale adapter from ViTAdapter features to station embeddings."""
    def __init__(self, encoder_dim, hidden_channels, output_dim,
                 pool_queries=4, pool_temperature=1.0, pool_dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.pool_queries = pool_queries
        self.base_pool = AttentionPool1d(
            encoder_dim,
            num_queries=pool_queries,
            temperature=pool_temperature,
            dropout=pool_dropout,
        )
        self.base_proj = MuReadout(encoder_dim * pool_queries, output_dim, bias=False)
        self.proj_f2 = nn.Conv1d(encoder_dim, hidden_channels, kernel_size=1)
        self.proj_f3 = nn.Conv1d(encoder_dim, hidden_channels, kernel_size=1)
        self.proj_f4 = nn.Conv1d(encoder_dim, hidden_channels, kernel_size=1)
        self.proj_x = nn.Conv1d(encoder_dim, hidden_channels, kernel_size=1)
        self.refine_f2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine_f3 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine_f4 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine_x = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool_f2 = AttentionPool1d(hidden_channels, pool_queries, pool_temperature, pool_dropout)
        self.pool_f3 = AttentionPool1d(hidden_channels, pool_queries, pool_temperature, pool_dropout)
        self.pool_f4 = AttentionPool1d(hidden_channels, pool_queries, pool_temperature, pool_dropout)
        self.pool_x = AttentionPool1d(hidden_channels, pool_queries, pool_temperature, pool_dropout)
        self.proj_out = nn.Linear(hidden_channels * pool_queries * 4, output_dim)
        self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.norm = nn.LayerNorm(output_dim)
        self._init_preserving_path()

    def _init_preserving_path(self):
        nn.init.zeros_(self.base_proj.weight)
        diag = min(self.output_dim, self.base_proj.in_features)
        with torch.no_grad():
            self.base_proj.weight[:diag, :diag] = torch.eye(diag)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj_out.bias)

    def reset_parameters(self):
        self._init_preserving_path()
        self.base_pool.reset_parameters()
        for module in (self.proj_f2, self.proj_f3, self.proj_f4, self.proj_x):
            nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for block in (self.refine_f2, self.refine_f3, self.refine_f4, self.refine_x):
            conv = block[0]
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for pool in (self.pool_f2, self.pool_f3, self.pool_f4, self.pool_x):
            pool.reset_parameters()
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, inputs):
        f2, f3, f4, x = inputs
        x = x.transpose(1, 2).contiguous()
        base = self.base_proj(self.base_pool(x))

        branch_f2 = self.pool_f2(self.refine_f2(self.proj_f2(f2)))
        branch_f3 = self.pool_f3(self.refine_f3(self.proj_f3(f3)))
        branch_f4 = self.pool_f4(self.refine_f4(self.proj_f4(f4)))
        branch_x = self.pool_x(self.refine_x(self.proj_x(x)))

        pooled = torch.cat([branch_f2, branch_f3, branch_f4, branch_x], dim=-1)
        delta = self.proj_out(pooled.float())
        return self.norm(base + self.delta_scale * delta)


class BackboneAttentionPoolAdapter(nn.Module):
    """Project plain backbone patch tokens into a station embedding."""

    def __init__(self, encoder_dim, output_dim, attn_temperature=0.5, attn_topk=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        self.attn_temperature = attn_temperature
        self.attn_topk = attn_topk

        self.base_proj = MuReadout(encoder_dim, output_dim, bias=False)
        self.attn_norm = nn.LayerNorm(encoder_dim)
        self.attn_query = nn.Parameter(torch.zeros(encoder_dim))
        self.focus_proj = MuReadout(encoder_dim, output_dim, bias=False)
        self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.norm = nn.LayerNorm(output_dim)
        self._last_attention = None
        self.reset_parameters()

    def _init_preserving_path(self):
        nn.init.zeros_(self.base_proj.weight)
        diag = min(self.output_dim, self.base_proj.in_features)
        with torch.no_grad():
            self.base_proj.weight[:diag, :diag] = torch.eye(diag)
        nn.init.normal_(self.focus_proj.weight, mean=0.0, std=1e-3)

    def reset_parameters(self):
        self._init_preserving_path()
        nn.init.normal_(self.attn_query, mean=0.0, std=1e-3)
        nn.init.ones_(self.attn_norm.weight)
        nn.init.zeros_(self.attn_norm.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def _masked_attention_weights(self, scores):
        if self.attn_topk and 0 < self.attn_topk < scores.shape[1]:
            topk_idx = torch.topk(scores, k=self.attn_topk, dim=1).indices
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(1, topk_idx, 0.0)
            scores = scores + mask
        temperature = max(float(self.attn_temperature), 1e-4)
        return torch.softmax(scores / temperature, dim=1)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected backbone features with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] == self.encoder_dim:
            tokens = x.transpose(1, 2).contiguous()
        elif x.shape[2] == self.encoder_dim:
            tokens = x
        else:
            raise ValueError(
                f"Expected encoder dim {self.encoder_dim} in shape {tuple(x.shape)}"
            )

        base = self.base_proj(tokens.mean(dim=1).float())
        scores = torch.matmul(self.attn_norm(tokens).float(), self.attn_query.float())
        weights = self._masked_attention_weights(scores)
        self._last_attention = weights.detach()
        focus = torch.sum(tokens * weights.unsqueeze(-1).type_as(tokens), dim=1)
        delta = self.focus_proj(focus.float())
        return self.norm(base + self.delta_scale * delta)


class GlobalMaxPooling1DMasked(nn.Module):
    def forward(self, x, mask=None):
        pseudo_infty = 1000.
        if mask is not None:
            # Subtract infty from padding positions so they never win max
            inv_mask = (~mask).unsqueeze(-1).float()
            x = x - inv_mask * pseudo_infty
        return torch.max(x, dim=1)[0]


def mixture_density_loss(y_pred, y_true, eps=1e-6, d=1, mean=True, print_shapes=False):
    if isinstance(y_pred, list):
        y_pred = y_pred[0]
        y_true = y_true[0]
    if print_shapes:
        print(f'True: {y_true.shape}')
        print(f'Pred: {y_pred.shape}')

    alpha = y_pred[:, :, 0]
    density = torch.ones_like(y_pred[:, :, 0]).to(y_pred.device)  # Move to device

    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = torch.maximum(sigma, torch.tensor(eps).to(y_pred.device)) #Move to device

        y_true_tmp = y_true[:, j].clone()
        while y_true_tmp.dim() < sigma.dim():
            y_true_tmp = y_true_tmp.unsqueeze(-1)
        density *= 1 / (np.sqrt(2 * np.pi) * sigma) * torch.exp(-(y_true_tmp - mu) ** 2 / (2 * sigma ** 2))

    density *= alpha
    density = torch.sum(density, dim=1)
    density += eps
    loss = - torch.log(density)

    if mean:
        return torch.mean(loss)
    else:
        return loss

def select_loss_components(outputs, labels, output_layout, res_comps):
    """Pick model outputs / labels for the requested task components.

    `output_layout` is the ordered list of task heads the model and generator
    actually produce (e.g. ['mag','loc','pga']). `res_comps` is the subset of
    tasks to train on this step; the returned lists are in res_comps order so
    the loss function can iterate positionally.

    Raises ValueError if a requested component is not in the layout — e.g.
    user asks to train 'loc' but the model was built with no_event_token=True.
    """
    name_to_idx = {n: i for i, n in enumerate(output_layout)}
    label_layout = [n for n in output_layout if n in ('mag', 'loc', 'pga')]
    label_name_alias = {'mag_aux': 'mag'}
    missing = [c for c in res_comps if c not in name_to_idx]
    missing_labels = [
        label_name_alias.get(c, c)
        for c in res_comps
        if label_name_alias.get(c, c) not in label_layout
    ]
    if missing:
        raise ValueError(
            f'res_comps {missing} not in model output_layout {output_layout}. '
            f'Enable the corresponding heads in the model config or remove '
            f'them from res_comps.'
        )
    if missing_labels:
        raise ValueError(
            f'res_comps require labels {missing_labels}, but available label layout is {label_layout}.'
        )
    sel_pred = [outputs[name_to_idx[c]] for c in res_comps]
    sel_true = [labels[label_layout.index(label_name_alias.get(c, c))] for c in res_comps]
    return sel_pred, sel_true


def mixture_density_loss_full(y_pred, y_true, eps=1e-6, d=1, mean=True, print_shapes=False, res_comps=None, res_weight=None, pga_target_valid=None):
    if res_comps is None:
        res_comps = ['mag', 'loc', 'pga']
        res_weight = np.array([1.,1.,1.])
    res_weight = res_weight / np.sum(res_weight)
    loss = 0.
    for i, res_comp in enumerate(res_comps):
        alpha_logits = y_pred[i][..., 0]
        log_density = torch.zeros_like(y_pred[i][..., 0]).to(y_pred[i].device)  # Move to device

        if res_comp == 'loc':
            d = 3
        else:
            d = 1

        for j in range(d):
            mu = y_pred[i][..., j + 1]
            sigma = y_pred[i][..., j + 1 + d]

            y_true_tmp = y_true[i][..., j].clone()
            while y_true_tmp.dim() < sigma.dim():
                y_true_tmp = y_true_tmp.unsqueeze(-1)
            log_density = log_density - torch.log(np.sqrt(2 * np.pi) * sigma) - (y_true_tmp - mu) ** 2 / (2 * sigma ** 2)

        log_density = log_density + torch.log_softmax(alpha_logits,dim=-1)
        log_density = torch.logsumexp(log_density, dim=-1)  # shape (B, n_pga) for pga, (B,) for mag/loc
        if res_comp == 'pga' and pga_target_valid is not None:
            mask = pga_target_valid.to(log_density.dtype)
            denom = mask.sum().clamp_min(1.0)
            comp_loss = -(log_density * mask).sum() / denom
        else:
            comp_loss = -torch.mean(log_density)
        loss = loss + res_weight[i] * comp_loss
    return loss


def _point_target_for_loss(y_true, y_pred, d):
    if d == 1:
        if y_pred.ndim >= 3:
            return y_true[..., :1].reshape_as(y_pred)
        return y_true.reshape(y_true.shape[0], -1)[:, :1].reshape_as(y_pred)
    return y_true.reshape(y_true.shape[0], -1, d)[:, 0, :].reshape_as(y_pred)


def _pga_loss_weight_tensor(target_raw, valid_mask, per_elem, pga_loss_weighting):
    if not pga_loss_weighting or not pga_loss_weighting.get('enabled', False):
        return None, None

    mode = normalize(pga_loss_weighting.get('mode', 'threshold'))
    base_weight = float(pga_loss_weighting.get('base_weight', 1.0))
    strong_weight = float(pga_loss_weighting.get('strong_weight', 1.0))
    if base_weight <= 0 or strong_weight <= 0:
        raise ValueError('pga_loss_weighting base_weight and strong_weight must be positive.')

    weights = torch.full_like(per_elem, base_weight)
    if mode == 'threshold':
        threshold = float(pga_loss_weighting.get('threshold', -1.2))
        strong = target_raw.to(per_elem.device, dtype=per_elem.dtype) >= threshold
        while strong.ndim < per_elem.ndim:
            strong = strong.unsqueeze(-1)
        weights = torch.where(strong, torch.full_like(weights, strong_weight), weights)
    else:
        raise ValueError(f"Unsupported pga_loss_weighting mode {mode!r}")

    if valid_mask is None:
        valid_f = torch.ones_like(per_elem)
    else:
        valid_f = valid_mask.to(per_elem.device).bool()
        while valid_f.ndim < per_elem.ndim:
            valid_f = valid_f.unsqueeze(-1)
        valid_f = valid_f.to(per_elem.dtype)

    if pga_loss_weighting.get('normalize_mean', True):
        denom = (weights * valid_f).sum().clamp_min(1.0)
    else:
        denom = valid_f.sum().clamp_min(1.0)
    return weights, denom


def point_regression_loss_full(y_pred, y_true, res_comps=None, res_weight=None,
                               pga_target_valid=None, loss_type='huber',
                               huber_delta=1.0, pga_target_normalization=None,
                               pga_loss_weighting=None):
    if res_comps is None:
        res_comps = ['mag', 'loc', 'pga']
        res_weight = np.array([1., 1., 1.])
    res_weight = np.asarray(res_weight, dtype=float)
    res_weight = res_weight / np.sum(res_weight)
    total_loss = 0.0
    for i, res_comp in enumerate(res_comps):
        pred = y_pred[i]
        if res_comp == 'loc':
            d = 3
        else:
            d = 1
        target = _point_target_for_loss(y_true[i], pred, d).to(pred.device, dtype=pred.dtype)
        target_raw = target
        if res_comp == 'pga' and pga_target_normalization is not None:
            mean = float(pga_target_normalization.get('mean', 0.0))
            std = max(float(pga_target_normalization.get('std', 1.0)), 1e-8)
            target = (target - mean) / std
        if loss_type == 'huber':
            per_elem = F.smooth_l1_loss(pred, target, beta=huber_delta, reduction='none')
        elif loss_type == 'l1':
            per_elem = torch.abs(pred - target)
        elif loss_type == 'mse':
            per_elem = (pred - target) ** 2
        else:
            raise ValueError(f"Unsupported point regression loss {loss_type!r}")

        if res_comp == 'pga' and pga_target_valid is not None:
            mask = pga_target_valid.to(pred.device).bool()
            while mask.ndim < per_elem.ndim:
                mask = mask.unsqueeze(-1)
            mask_f = mask.to(per_elem.dtype)
            pga_weights, weighted_denom = _pga_loss_weight_tensor(
                target_raw, pga_target_valid, per_elem, pga_loss_weighting
            )
            if pga_weights is not None:
                comp_loss = (per_elem * mask_f * pga_weights).sum() / weighted_denom
            else:
                denom = mask_f.sum().clamp_min(1.0)
                comp_loss = (per_elem * mask_f).sum() / denom
        else:
            comp_loss = per_elem.mean()
        total_loss = total_loss + float(res_weight[i]) * comp_loss
    return total_loss


def station_embedding_decorrelation_loss(station_emb, station_valid, eps=1e-8):
    """Penalize same-event station tokens becoming nearly identical directions."""
    if station_emb is None or station_valid is None:
        return None
    losses = []
    normed = F.normalize(station_emb, p=2, dim=-1, eps=eps)
    for sample_emb, valid in zip(normed, station_valid.bool()):
        x = sample_emb[valid]
        if x.shape[0] < 2:
            continue
        sim = torch.matmul(x, x.transpose(0, 1))
        off_diag = sim[~torch.eye(x.shape[0], device=x.device, dtype=torch.bool)]
        losses.append((off_diag ** 2).mean())
    if not losses:
        return station_emb.new_tensor(0.0)
    return torch.stack(losses).mean()


def clip_magnitude_mixture(mixture_output, mag_min=-2.0, mag_max=10.0):
    if isinstance(mixture_output, torch.Tensor):
        clipped = mixture_output.clone()
        clipped[..., 1] = torch.clamp(clipped[..., 1], min=mag_min, max=mag_max)
        return clipped
    clipped = np.array(mixture_output, copy=True)
    clipped[..., 1] = np.clip(clipped[..., 1], mag_min, mag_max)
    return clipped

def time_distributed_loss(y_true, y_pred, loss_func, norm=1, mean=True, summation=True, kwloss={}):
    seq_length = y_pred.shape[1]
    y_true = y_true.reshape(-1, (y_pred.shape[-1] - 1) // 2, 1)
    y_pred = y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1])
    loss = loss_func(y_true, y_pred, **kwloss)
    loss = loss.reshape(-1, seq_length)

    if mean:
        return torch.mean(loss)

    loss /= norm
    if summation:
        loss = torch.sum(loss)

    return loss

class FullModel(nn.Module):
    def __init__(self, waveform_model, position_embedding, transformer, mlp_mag, output_model_mag, mlp_loc,
                 output_model_loc, mlp_pga, output_model_pga, skip_transformer, alternative_coords_embedding,
                 metadata_shape, emb_dim, no_event_token, add_event_token, n_pga_targets, dataset_bias,
                 add_constant_to_mixture, n_datasets, waveform_scale_proj=None, waveform_scale_gain=1.0,
                 waveform_scale_init_gate=0.1, disable_waveform_scale=False, use_amplitude_info=None,
                 use_coords_rel=False, use_coords_abs=True, use_coords_rel_abs_fusion=False,
                 coords_abs_weight=0.1, coord_fusion_mode='add', aux_mag_head=None,
                 output_distribution='mdn', pga_readout_mode='query_transformer',
                 event_readout_mode='event_transformer',
                 pga_attention_diagnostics=False, pga_mask_sanity_check=False,
                 readout_n_heads=1, readout_dropout=0.0, query_token_init_range=0.02,
                 pga_distance_bias=False, pga_distance_bias_hidden_dim=64,
                 readout_layers=1, pga_readout_layers=None, event_readout_layers=None,
                 readout_ffn_hidden_dim=None,
                 readout_first_residual=False, readout_first_residual_gate_init=None,
                 readout_residual_gates=False, readout_residual_gate_init=0.0,
                 readout_ffn_gate_init=None, readout_inject_base_query=False,
                 readout_query_injection_gate_init=1.0, readout_use_ffn=True,
                 station_context_mode='off',
                 station_context_gate_init=0.0,
                 pga_use_event_context=False, pga_event_context_init_gate=0.0,
                 use_vs30=False, vs30_reference_mps=760.0, vs30_init_gate=0.0,
                 use_rope=False, rope_coord_mode='relative_xy_km', rope_coord_scale=100.0,
                 rope_base=10000.0, rope_lat_origin=37.0,
                 station_context_encoder=None, pga_station_target_readout=None):
        super().__init__()
        self.waveform_model = waveform_model
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.mlp_mag = mlp_mag
        self.output_model_mag = output_model_mag
        self.mlp_loc = mlp_loc
        self.output_model_loc = output_model_loc
        self.mlp_pga = mlp_pga
        self.output_model_pga = output_model_pga
        self.skip_transformer = skip_transformer
        self.alternative_coords_embedding = alternative_coords_embedding
        self.metadata_shape = metadata_shape
        self.emb_dim = emb_dim
        self.no_event_token = no_event_token
        self.add_event_token = add_event_token
        self.n_pga_targets = n_pga_targets
        self.dataset_bias = dataset_bias
        self.add_constant_to_mixture = add_constant_to_mixture
        self.n_datasets = n_datasets
        self.waveform_scale_proj = waveform_scale_proj
        self.waveform_scale_gain = waveform_scale_gain
        self.waveform_scale_gate = nn.Parameter(torch.tensor(float(waveform_scale_init_gate)))
        if use_amplitude_info is None:
            use_amplitude_info = not disable_waveform_scale
        self.use_amplitude_info = bool(use_amplitude_info)
        self.disable_waveform_scale = not self.use_amplitude_info
        self.use_coords_rel = use_coords_rel
        self.use_coords_abs = use_coords_abs
        self.use_coords_rel_abs_fusion = use_coords_rel_abs_fusion
        self.coords_abs_weight = coords_abs_weight
        self.coord_fusion_mode = coord_fusion_mode
        self.aux_mag_head = aux_mag_head
        self.output_distribution = output_distribution
        self.pga_readout_mode = pga_readout_mode
        self.event_readout_mode = event_readout_mode
        self.station_context_mode = station_context_mode or 'off'
        self.use_vs30 = bool(use_vs30)
        self.vs30_reference_mps = float(vs30_reference_mps)
        self.use_rope = bool(use_rope)
        self.station_context_encoder = station_context_encoder
        self.pga_station_target_readout = pga_station_target_readout
        self.pga_use_event_context = bool(pga_use_event_context)
        self.pga_attention_diagnostics = bool(pga_attention_diagnostics)
        self.pga_mask_sanity_check = bool(pga_mask_sanity_check)
        if pga_readout_layers is None:
            pga_readout_layers = readout_layers
        if event_readout_layers is None:
            event_readout_layers = readout_layers
        if self.output_distribution not in ('mdn', 'point'):
            raise ValueError(
                f"output_distribution must be 'mdn' or 'point', got {self.output_distribution!r}."
            )
        if self.pga_readout_mode not in (
            'query_transformer',
            'query_no_transformer',
            'direct_station',
            'target_cross_attention',
        ):
            raise ValueError(
                "pga_readout_mode must be one of 'query_transformer', "
                "'query_no_transformer', 'direct_station', 'target_cross_attention', "
                f"got {self.pga_readout_mode!r}."
            )
        if self.pga_readout_mode != 'query_transformer' and self.output_distribution != 'point':
            raise ValueError('pga_readout_mode ablations are implemented for point output_distribution.')
        if self.event_readout_mode not in (
            'event_transformer',
            'event_cross_attention',
            'direct_station_pool',
        ):
            raise ValueError(
                "event_readout_mode must be one of 'event_transformer', "
                "'event_cross_attention', 'direct_station_pool', "
                f"got {self.event_readout_mode!r}."
            )
        firstres_station_modes = ('firstres_transformer_pre_readout', 'gated_transformer_pre_readout_firstres')
        if self.station_context_mode not in (
            'off',
            'transformer_pre_readout',
            'gated_transformer_pre_readout',
            *firstres_station_modes,
            'synchronous_station_target',
        ):
            raise ValueError(
                "station_context_mode must be one of 'off', 'transformer_pre_readout', "
                "'gated_transformer_pre_readout', 'firstres_transformer_pre_readout', "
                "'gated_transformer_pre_readout_firstres', 'synchronous_station_target', "
                f"got {self.station_context_mode!r}."
            )
        if self.station_context_mode != 'off' and self.transformer is None:
            raise ValueError('station_context_mode requires skip_transformer=False.')
        if self.station_context_mode in firstres_station_modes and self.station_context_encoder is None:
            raise ValueError(f'{self.station_context_mode} requires station_context_encoder.')
        if self.station_context_mode == 'synchronous_station_target':
            if self.pga_readout_mode != 'target_cross_attention':
                raise ValueError('synchronous_station_target requires pga_readout_mode=target_cross_attention.')
            if self.pga_station_target_readout is None:
                raise ValueError('synchronous_station_target requires pga_station_target_readout.')
        if self.station_context_mode == 'gated_transformer_pre_readout':
            self.station_context_gate = nn.Parameter(torch.tensor(float(station_context_gate_init)))
        else:
            self.station_context_gate = None

        active_coord_modes = sum(bool(flag) for flag in (
            self.use_coords_rel, self.use_coords_abs, self.use_coords_rel_abs_fusion
        ))
        if active_coord_modes != 1:
            raise ValueError(
                'Exactly one of use_coords_rel / use_coords_abs / '
                'use_coords_rel_abs_fusion must be True.'
            )
        if self.alternative_coords_embedding and self.use_coords_rel_abs_fusion:
            raise ValueError(
                'use_coords_rel_abs_fusion requires positional coordinate embeddings; '
                'it is incompatible with alternative_coords_embedding=True.'
            )
        if self.coord_fusion_mode not in ('add', 'concat'):
            raise ValueError(
                f"coord_fusion_mode must be 'add' or 'concat', got {self.coord_fusion_mode!r}."
            )
        if self.alternative_coords_embedding and self.coord_fusion_mode != 'add':
            raise ValueError(
                'coord_fusion_mode=concat is only supported with positional coordinate '
                'embeddings (alternative_coords_embedding=False).'
            )
        self.loc_target_mode = 'abs' if self.use_coords_abs else 'rel'

        self.pga_query_token = nn.Parameter(torch.empty(1, 1, emb_dim))
        nn.init.normal_(self.pga_query_token, mean=0.0, std=float(query_token_init_range))
        self.event_query_token = nn.Parameter(torch.empty(1, 1, emb_dim))
        nn.init.normal_(self.event_query_token, mean=0.0, std=float(query_token_init_range))
        self.pga_cross_attention = CrossAttentionReadout(
            emb_dim,
            readout_n_heads,
            att_dropout=readout_dropout,
            distance_bias=pga_distance_bias,
            distance_hidden_dim=pga_distance_bias_hidden_dim,
            readout_layers=pga_readout_layers,
            ffn_hidden_dim=readout_ffn_hidden_dim,
            first_residual=readout_first_residual,
            first_residual_gate_init=readout_first_residual_gate_init,
            residual_gates=readout_residual_gates,
            residual_gate_init=readout_residual_gate_init,
            ffn_gate_init=readout_ffn_gate_init,
            inject_base_query=readout_inject_base_query,
            query_injection_gate_init=readout_query_injection_gate_init,
            use_ffn=readout_use_ffn,
            use_rope=use_rope,
            rope_coord_mode=rope_coord_mode,
            rope_coord_scale=rope_coord_scale,
            rope_base=rope_base,
            rope_lat_origin=rope_lat_origin,
        )
        self.event_cross_attention = CrossAttentionReadout(
            emb_dim,
            readout_n_heads,
            att_dropout=readout_dropout,
            readout_layers=event_readout_layers,
            ffn_hidden_dim=readout_ffn_hidden_dim,
            first_residual=readout_first_residual,
            first_residual_gate_init=readout_first_residual_gate_init,
            residual_gates=readout_residual_gates,
            residual_gate_init=readout_residual_gate_init,
            ffn_gate_init=readout_ffn_gate_init,
            inject_base_query=readout_inject_base_query,
            query_injection_gate_init=readout_query_injection_gate_init,
            use_ffn=readout_use_ffn,
        )
        if self.pga_use_event_context:
            self.pga_event_context_proj = nn.Linear(emb_dim, emb_dim)
            self.pga_event_context_gate = nn.Parameter(torch.tensor(float(pga_event_context_init_gate)))
        else:
            self.pga_event_context_proj = None
            self.pga_event_context_gate = None
        if self.use_vs30:
            self.station_vs30_proj = nn.Linear(2, emb_dim)
            self.target_vs30_proj = nn.Linear(2, emb_dim)
            self.station_vs30_gate = nn.Parameter(torch.tensor(float(vs30_init_gate)))
            self.target_vs30_gate = nn.Parameter(torch.tensor(float(vs30_init_gate)))
        else:
            self.station_vs30_proj = None
            self.target_vs30_proj = None
            self.station_vs30_gate = None
            self.target_vs30_gate = None

        if self.n_pga_targets > 0:
            self.att_masking = True
        else:
            self.att_masking = False

        # Ordered list of task heads actually produced by forward(), used to
        # map res_comps (e.g. ['pga']) to the correct index in outputs/labels.
        # mag and loc are tied together under `no_event_token`; pga is gated
        # independently by n_pga_targets.
        self.output_layout = []
        if not self.no_event_token:
            self.output_layout += ['mag', 'loc']
        if self.aux_mag_head is not None:
            self.output_layout += ['mag_aux']
        if self.n_pga_targets > 0:
            self.output_layout += ['pga']

        if dataset_bias:
            self.dataset_embedding = nn.Embedding(n_datasets, 1)
        self.Masking_nd_0_23 = Masking_nd(0, (2, 3))
        self.Masking_nd_0_2 = Masking_nd(0, axis=2, nodim=True)
#        self.layernorm = LayerNormalization()
        self.layernorm = nn.LayerNorm(emb_dim)
        if not self.alternative_coords_embedding and self.coord_fusion_mode == 'concat':
            self.coord_fusion_proj = nn.Linear(2 * emb_dim, emb_dim)
            self.coord_fusion_norm = nn.LayerNorm(emb_dim)
        else:
            self.coord_fusion_proj = None
            self.coord_fusion_norm = None
        if self.skip_transformer:
            mlp_input_length = self.emb_dim
            if self.alternative_coords_embedding:
                mlp_input_length += self.metadata_shape[0]
            self.mlp_layer = MLP((mlp_input_length,), [self.emb_dim, self.emb_dim], activation='relu')
            self.maxpool = GlobalMaxPooling1DMasked()
        self._last_diag = {}

    @staticmethod
    def _mean_token_norm(x):
        return x.norm(dim=-1).mean()

    @staticmethod
    def _masked_pairwise_cosine_between(x, y, mask):
        vals = []
        for sample_x, sample_y, valid in zip(x, y, mask):
            vx = sample_x[valid]
            vy = sample_y[valid]
            if vx.shape[0] == 0:
                continue
            cos = F.cosine_similarity(vx, vy, dim=-1)
            vals.append(cos.mean())
        if not vals:
            return x.new_tensor(float('nan'))
        return torch.stack(vals).mean()

    @staticmethod
    def _masked_pairwise_cosine_mean(x, mask):
        vals = []
        for sample, valid in zip(x, mask):
            valid_sample = sample[valid]
            if valid_sample.shape[0] < 2:
                continue
            normed = valid_sample / (valid_sample.norm(dim=-1, keepdim=True) + 1e-8)
            cos = normed @ normed.T
            n = cos.shape[0]
            off_diag = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
            vals.append(off_diag.mean())
        if not vals:
            return x.new_tensor(float('nan'))
        return torch.stack(vals).mean()

    def _normalize(self, data, mode, axis=1):
        """
        Normalize waveform of each sample. (inplace)
        """
        data = data - torch.mean(data, axis=axis, keepdims=True)
        if mode == "max":
            max_data = torch.max(data, axis=axis, keepdims=True)
            max_data[max_data == 0] = 1
            data = data/max_data # + 1e-6

        elif mode == "std":
            std_data = torch.std(data, axis=axis, keepdims=True)
            std_data[std_data == 0] = 1
            data = data/std_data # + 1e-6
        elif mode == "":
            return data
        else:
            raise ValueError(f"Supported mode: 'max','std', got '{mode}'")
        return data

    @staticmethod
    def _masked_coord_center(coords, valid_mask):
        weights = valid_mask.unsqueeze(-1).to(coords.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (coords * weights).sum(dim=1, keepdim=True) / denom

    def _make_relative_coords(self, coords, valid_mask):
        center = self._masked_coord_center(coords, valid_mask)
        rel = coords - center
        return rel * valid_mask.unsqueeze(-1).to(coords.dtype), center

    def _station_coord_features(self, coords_abs, coords_rel, valid_mask):
        if self.alternative_coords_embedding:
            coords_used = coords_rel if self.use_coords_rel else coords_abs
            return coords_used, None

        coords_abs_emb = self.position_embedding(coords_abs, mask=valid_mask)
        if self.use_coords_abs:
            return coords_abs_emb, coords_abs_emb

        coords_rel_emb = self.position_embedding(coords_rel, mask=valid_mask)
        if self.use_coords_rel:
            return coords_rel_emb, coords_rel_emb

        coords_emb = coords_rel_emb + self.coords_abs_weight * coords_abs_emb
        return coords_emb, coords_emb

    def _pga_coord_embedding(self, coords_abs, coords_rel, valid_mask):
        coords_abs_emb = self.position_embedding(coords_abs, mask=valid_mask)
        if self.use_coords_abs:
            return coords_abs_emb

        coords_rel_emb = self.position_embedding(coords_rel, mask=valid_mask)
        if self.use_coords_rel:
            return coords_rel_emb

        return coords_rel_emb + self.coords_abs_weight * coords_abs_emb

    def _direct_station_pga_embedding(self, station_emb, station_valid, n_pga):
        weights = station_valid.unsqueeze(-1).to(station_emb.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (station_emb * weights).sum(dim=1) / denom
        return pooled.unsqueeze(1).expand(-1, n_pga, -1)

    @staticmethod
    def _looks_like_dataset_tensor(value):
        if not torch.is_tensor(value):
            return False
        return value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long) and value.dim() <= 1

    def _parse_extra_inputs(self, extra_inputs, dataset):
        extras = list(extra_inputs)
        if extras and self._looks_like_dataset_tensor(extras[-1]):
            if dataset is None:
                dataset = extras[-1]
            extras = extras[:-1]
        station_vs30 = station_vs30_valid = pga_target_vs30 = pga_target_vs30_valid = None
        if extras:
            if len(extras) != 4:
                raise ValueError(
                    'Expected four VS30 tensors after pga_target_valid '
                    '(station_vs30, station_vs30_valid, pga_target_vs30, pga_target_vs30_valid); '
                    f'got {len(extras)} extra positional inputs.'
                )
            station_vs30, station_vs30_valid, pga_target_vs30, pga_target_vs30_valid = extras
        return station_vs30, station_vs30_valid, pga_target_vs30, pga_target_vs30_valid, dataset

    def _vs30_feature_tensor(self, values, valid, reference_mask, ref_tensor):
        if values is None:
            values = ref_tensor.new_zeros(reference_mask.shape + (1,))
            valid_mask = torch.zeros(reference_mask.shape, device=ref_tensor.device, dtype=torch.bool)
        else:
            values = values.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            if values.dim() == 2:
                values = values.unsqueeze(-1)
            if valid is None:
                valid_mask = torch.isfinite(values.squeeze(-1)) & (values.squeeze(-1) > 0)
            else:
                valid_mask = valid.to(device=ref_tensor.device).bool()
            valid_mask = valid_mask & reference_mask.bool()
        safe_values = torch.clamp(values.squeeze(-1), min=1.0)
        log_vs30 = torch.log(safe_values / self.vs30_reference_mps)
        log_vs30 = torch.where(valid_mask, log_vs30, torch.zeros_like(log_vs30))
        return torch.stack([log_vs30, valid_mask.to(ref_tensor.dtype)], dim=-1), valid_mask

    def _add_station_vs30_embedding(self, station_emb, station_vs30, station_vs30_valid, station_mask):
        if not self.use_vs30:
            return station_emb, station_mask.new_zeros(station_mask.shape, dtype=torch.bool)
        features, valid = self._vs30_feature_tensor(station_vs30, station_vs30_valid, station_mask, station_emb)
        site_emb = self.station_vs30_proj(features)
        station_emb = station_emb + self.station_vs30_gate * site_emb * valid.unsqueeze(-1).to(station_emb.dtype)
        return station_emb, valid

    def _add_target_vs30_embedding(self, query_emb, target_vs30, target_vs30_valid, target_mask):
        if not self.use_vs30:
            return query_emb, target_mask.new_zeros(target_mask.shape, dtype=torch.bool)
        features, valid = self._vs30_feature_tensor(target_vs30, target_vs30_valid, target_mask, query_emb)
        site_emb = self.target_vs30_proj(features)
        query_emb = query_emb + self.target_vs30_gate * site_emb * valid.unsqueeze(-1).to(query_emb.dtype)
        return query_emb, valid

    def _record_station_context_diag(self, station_context_emb, station_feature_emb, station_mask,
                                     mode_code=1.0, raw_context_emb=None):
        self._last_station_context_emb = station_context_emb
        self._last_diag['station_context_mode'] = station_context_emb.new_tensor(float(mode_code)).detach()
        if raw_context_emb is not None:
            self._last_diag['station_context_raw_emb_norm'] = self._mean_token_norm(raw_context_emb).detach()
        self._last_diag['station_context_emb_norm'] = self._mean_token_norm(station_context_emb).detach()
        self._last_diag['station_context_delta_norm'] = (
            self._mean_token_norm(station_context_emb - station_feature_emb)
        ).detach()
        self._last_diag['station_context_cosine_mean'] = self._masked_pairwise_cosine_mean(
            station_context_emb,
            station_mask,
        ).detach()

    def _station_memory_for_readout(self, station_feature_emb, station_mask, coords_abs=None):
        if self.station_context_mode == 'off':
            self._last_station_context_emb = station_feature_emb
            return station_feature_emb
        if self.station_context_mode == 'synchronous_station_target':
            self._last_station_context_emb = station_feature_emb
            return station_feature_emb

        if self.station_context_mode in (
            'firstres_transformer_pre_readout',
            'gated_transformer_pre_readout_firstres',
        ):
            station_context_emb = self.station_context_encoder(
                station_feature_emb.float(),
                padding_mask=station_mask,
                coords=coords_abs,
            )
            station_context_emb = station_context_emb * station_mask.unsqueeze(-1).to(station_context_emb.dtype)
            self._record_station_context_diag(
                station_context_emb,
                station_feature_emb,
                station_mask,
                mode_code=2.0,
                raw_context_emb=station_context_emb,
            )
            return station_context_emb

        transformer_context_emb = self.transformer(
            station_feature_emb.float(),
            padding_mask=station_mask,
            coords=coords_abs,
        )
        transformer_context_emb = transformer_context_emb * station_mask.unsqueeze(-1).to(transformer_context_emb.dtype)
        if self.station_context_mode == 'gated_transformer_pre_readout':
            station_context_emb = station_feature_emb + self.station_context_gate * (
                transformer_context_emb - station_feature_emb
            )
            station_context_emb = station_context_emb * station_mask.unsqueeze(-1).to(station_context_emb.dtype)
        else:
            station_context_emb = transformer_context_emb
        if self.station_context_gate is not None:
            self._last_diag['station_context_gate'] = self.station_context_gate.detach()
        self._record_station_context_diag(
            station_context_emb,
            station_feature_emb,
            station_mask,
            mode_code=1.0,
            raw_context_emb=transformer_context_emb,
        )
        return station_context_emb

    def _record_pga_mask_diag(self, att_mask, padding_mask, station_mask, pga_target_valid):
        if not self.pga_mask_sanity_check:
            return
        has_event = not (self.skip_transformer or self.no_event_token)
        station_offset = 1 if has_event else 0
        pga_offset = station_offset + station_mask.shape[1]
        ptv = pga_target_valid.bool()
        # att_mask and padding_mask are deliberately two-level masks:
        # - att_mask marks key types that may be attended at all. All station
        #   slots are allowed here, and padding_mask removes invalid stations.
        # - padding_mask marks valid sequence positions.
        expected_key_mask = torch.cat([torch.ones_like(station_mask), torch.zeros_like(ptv)], dim=1)
        if has_event:
            expected_key_mask = torch.cat(
                [torch.ones(station_mask.shape[0], 1, device=station_mask.device, dtype=torch.bool), expected_key_mask],
                dim=1,
            )
        expected_padding = torch.cat([station_mask, ptv], dim=1)
        if has_event:
            expected_padding = torch.cat(
                [torch.ones(station_mask.shape[0], 1, device=station_mask.device, dtype=torch.bool), expected_padding],
                dim=1,
            )

        self._last_diag['pga_mask_key_matches_expected'] = (
            (att_mask == expected_key_mask).float().mean().detach()
        )
        effective_key_mask = att_mask & padding_mask
        expected_effective_key_mask = torch.cat([station_mask, torch.zeros_like(ptv)], dim=1)
        if has_event:
            expected_effective_key_mask = torch.cat(
                [torch.ones(station_mask.shape[0], 1, device=station_mask.device, dtype=torch.bool),
                 expected_effective_key_mask],
                dim=1,
            )
        self._last_diag['pga_effective_key_matches_expected'] = (
            (effective_key_mask == expected_effective_key_mask).float().mean().detach()
        )
        self._last_diag['pga_padding_matches_expected'] = (
            (padding_mask == expected_padding).float().mean().detach()
        )
        self._last_diag['pga_mask_station_key_ratio'] = (
            att_mask[:, station_offset:pga_offset].float().mean().detach()
        )
        if has_event:
            self._last_diag['pga_mask_event_key_ratio'] = att_mask[:, :1].float().mean().detach()
        self._last_diag['pga_mask_pga_key_ratio'] = att_mask[:, pga_offset:].float().mean().detach()

    def _record_pga_attention_diag(self, seq_len, station_count, n_pga, has_event):
        if not self.pga_attention_diagnostics or self.transformer is None:
            return
        attentions = getattr(self.transformer, '_last_attentions', [])
        if not attentions:
            return
        station_offset = 1 if has_event else 0
        station_slice = slice(station_offset, station_offset + station_count)
        pga_slice = slice(seq_len - n_pga, seq_len)

        for layer_idx, attn in enumerate((attentions[0], attentions[-1])):
            prefix = 'pga_attn_first' if layer_idx == 0 else 'pga_attn_last'
            pga_query_attn = attn[:, :, pga_slice, :]
            station_mass = pga_query_attn[:, :, :, station_slice].sum(dim=-1).mean()
            pga_mass = pga_query_attn[:, :, :, pga_slice].sum(dim=-1).mean()
            self._last_diag[f'{prefix}_to_station'] = station_mass.detach()
            self._last_diag[f'{prefix}_to_pga'] = pga_mass.detach()
            if has_event:
                event_mass = pga_query_attn[:, :, :, 0].mean()
                self._last_diag[f'{prefix}_to_event'] = event_mass.detach()

    def _record_cross_attention_diag(self, prefix, readout, valid_mask):
        attn = getattr(readout, '_last_attention', None)
        if attn is None:
            return
        valid = valid_mask.bool().unsqueeze(1).to(attn.device)
        valid_mass = (attn * valid.to(attn.dtype)).sum(dim=-1).mean()
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        valid_count = valid_mask.float().sum(dim=-1).clamp_min(1.0).mean()
        max_entropy = torch.log(valid_count.to(attn.device))
        self._last_diag[f'{prefix}_attn_valid_mass'] = valid_mass.detach()
        self._last_diag[f'{prefix}_attn_entropy'] = entropy.detach()
        self._last_diag[f'{prefix}_attn_entropy_ratio'] = (entropy / (max_entropy + 1e-8)).detach()
        self._last_diag[f'{prefix}_attn_max_weight'] = attn.max(dim=-1).values.mean().detach()

    def _record_readout_gate_diag(self, prefix, readout):
        try:
            ref = next(readout.parameters())
        except StopIteration:
            return

        def add_scalar(name, value):
            if torch.is_tensor(value):
                self._last_diag[name] = value.detach()
            else:
                self._last_diag[name] = ref.new_tensor(float(value)).detach()

        add_scalar(f'{prefix}_readout_layers', getattr(readout, 'readout_layers', 1))
        add_scalar(f'{prefix}_first_residual_enabled', getattr(readout, 'first_residual', False))
        first_gate = getattr(readout, 'first_residual_gate', None)
        if first_gate is not None:
            add_scalar(f'{prefix}_first_residual_gate', first_gate)

        attn_gates = []
        ffn_gates = []
        injection_gates = []
        for idx, layer in enumerate(getattr(readout, 'extra_layers', []), start=1):
            attn_gate = getattr(layer, 'attn_gate', None)
            if attn_gate is not None:
                add_scalar(f'{prefix}_extra_layer{idx}_attn_gate', attn_gate)
                attn_gates.append(attn_gate.detach())
            ffn_gate = getattr(layer, 'ffn_gate', None)
            if ffn_gate is not None:
                add_scalar(f'{prefix}_extra_layer{idx}_ffn_gate', ffn_gate)
                ffn_gates.append(ffn_gate.detach())
            injection_gate = getattr(layer, 'query_injection_gate', None)
            if injection_gate is not None:
                add_scalar(f'{prefix}_extra_layer{idx}_query_injection_gate', injection_gate)
                injection_gates.append(injection_gate.detach())

        if attn_gates:
            attn_stack = torch.stack(attn_gates)
            add_scalar(f'{prefix}_extra_attn_gate_mean', attn_stack.mean())
            add_scalar(f'{prefix}_extra_attn_gate_max_abs', attn_stack.abs().max())
        if ffn_gates:
            ffn_stack = torch.stack(ffn_gates)
            add_scalar(f'{prefix}_extra_ffn_gate_mean', ffn_stack.mean())
            add_scalar(f'{prefix}_extra_ffn_gate_max_abs', ffn_stack.abs().max())
        if injection_gates:
            injection_stack = torch.stack(injection_gates)
            add_scalar(f'{prefix}_extra_query_injection_gate_mean', injection_stack.mean())
            add_scalar(f'{prefix}_extra_query_injection_gate_max_abs', injection_stack.abs().max())


    def _extract_scale_features(self, waveform):
        # waveform: (B, S, C, T). Match the per-channel time-axis normalization
        # by retaining channel-wise std/rms/peak plus station-level summaries.
        return extract_waveform_scale_features(waveform)

    def forward(self, waveform_inp, metadata_inp, station_valid,
                pga_targets_inp=None, pga_target_valid=None,
                *extra_inputs, dataset=None, att_mask=None):
        station_vs30, station_vs30_valid, pga_target_vs30, pga_target_vs30_valid, dataset = self._parse_extra_inputs(
            extra_inputs,
            dataset,
        )
        raw_waveform = waveform_inp.clone()
        waveform_inp = self._normalize(waveform_inp, mode='std', axis=3)
        # Apply explicit masks instead of inferring "validity == nonzero".
        # station_valid: (B, S) bool. pga_target_valid: (B, n_pga) bool.
        sv = station_valid.bool()
        waveforms_masked = waveform_inp * sv[:, :, None, None].float()
        coords_abs = metadata_inp * sv[:, :, None].float()
        coords_rel, coords_center = self._make_relative_coords(coords_abs, sv)

        raw_station_emb = torch.stack([self.waveform_model(waveforms_masked[:, i, :, :]) for i in range(waveforms_masked.shape[1])] , dim=1)
        scale_emb = None
        preln_wave_emb = raw_station_emb
        waveforms_emb = self.layernorm(preln_wave_emb)
        base_waveforms_emb = waveforms_emb
        if self.waveform_scale_proj is not None and self.use_amplitude_info:
            scale_features = self._extract_scale_features(raw_waveform)
            scale_emb = self.waveform_scale_proj(scale_features)
            gain_scale_emb = (
                self.waveform_scale_gain
                * self.waveform_scale_gate
                * scale_emb
                * sv[:, :, None].float()
            )
            waveforms_emb = waveforms_emb + gain_scale_emb
        else:
            scale_features = None
            gain_scale_emb = None

        coords_feat, coords_emb = self._station_coord_features(coords_abs, coords_rel, sv)
        if not self.alternative_coords_embedding:
            if self.coord_fusion_mode == 'add':
                emb = waveforms_emb + coords_feat
            else:
                fused = torch.cat([waveforms_emb, coords_feat], dim=-1)
                emb = self.coord_fusion_proj(fused)
                emb = self.coord_fusion_norm(emb)
        else:
            emb = torch.cat([waveforms_emb, coords_feat], dim=-1)
        station_vs30_valid_mask = None
        if self.use_vs30:
            emb, station_vs30_valid_mask = self._add_station_vs30_embedding(
                emb,
                station_vs30,
                station_vs30_valid,
                sv,
            )
        station_feature_emb = emb
        self._last_station_feature_emb = station_feature_emb
        self._last_station_valid = sv
        station_mask = sv  # (batch, n_stations) — comes from explicit station_valid input

        self._last_diag = {
            'station_adapter_raw_norm': self._mean_token_norm(raw_station_emb).detach(),
            'preln_wave_emb_norm': self._mean_token_norm(preln_wave_emb).detach(),
            'wave_emb_norm': self._mean_token_norm(waveforms_emb).detach(),
            'station_emb_norm': self._mean_token_norm(emb).detach(),
            'station_emb_cosine_mean': self._masked_pairwise_cosine_mean(emb, sv).detach(),
            'coords_center_abs_mean': coords_center.abs().mean().detach(),
            'coords_abs_mean': coords_abs.abs().mean().detach(),
            'coords_rel_mean': coords_rel.abs().mean().detach(),
            'coord_fusion_mode': 0.0 if self.coord_fusion_mode == 'add' else 1.0,
            'station_context_mode': emb.new_tensor(0.0).detach(),
        }
        if self.use_vs30:
            station_valid_count = sv.float().sum().clamp_min(1.0)
            self._last_diag['vs30_station_valid_ratio'] = (
                station_vs30_valid_mask.float().sum() / station_valid_count
            ).detach()
            self._last_diag['vs30_station_gate'] = self.station_vs30_gate.detach()
        if scale_emb is not None:
            scale_norm = self._mean_token_norm(scale_emb)
            self._last_diag['scale_emb_norm'] = scale_norm.detach()
            gain_scale_norm = self._mean_token_norm(gain_scale_emb)
            raw_trunk_norm = self._mean_token_norm(base_waveforms_emb)
            self._last_diag['gain_scale_emb_norm'] = gain_scale_norm.detach()
            self._last_diag['scale_trunk_ratio'] = (gain_scale_norm / (raw_trunk_norm + 1e-8)).detach()
            self._last_diag['scale_gate'] = self.waveform_scale_gate.detach()
            self._last_diag['scale_feature_mean'] = scale_features[sv].mean().detach() if sv.any() else scale_features.mean().detach()
            self._last_diag['scale_feature_std'] = scale_features[sv].std(unbiased=False).detach() if sv.any() else scale_features.std(unbiased=False).detach()
            self._last_diag['raw_trunk_scale_cosine'] = self._masked_pairwise_cosine_between(
                base_waveforms_emb, gain_scale_emb, sv
            ).detach()
        if not self.alternative_coords_embedding:
            self._last_diag['coords_emb_norm'] = self._mean_token_norm(coords_emb).detach()
        station_memory_emb = self._station_memory_for_readout(station_feature_emb, station_mask, coords_abs=coords_abs)

        # padding_mask: True=valid position, used for key masking + output zeroing (matches TF `mask`)
        # att_mask: True=attendable as key, used only for additional key masking (matches TF `att_mask`)
        transformer_emb = None
        pga_readout_emb = None

        if self.n_pga_targets:
            assert pga_target_valid is not None, \
                'pga_target_valid must be provided when n_pga_targets > 0'
            ptv = pga_target_valid.bool()
            pga_targets_abs = pga_targets_inp * ptv[:, :, None].float()
            pga_targets_rel = (pga_targets_abs - coords_center) * ptv[:, :, None].float()
            pga_emb = self._pga_coord_embedding(pga_targets_abs, pga_targets_rel, ptv)
            pga_query_emb = pga_emb + self.pga_query_token
            if self.use_vs30:
                pga_query_emb, pga_vs30_valid_mask = self._add_target_vs30_embedding(
                    pga_query_emb,
                    pga_target_vs30,
                    pga_target_vs30_valid,
                    ptv,
                )
                target_valid_count = ptv.float().sum().clamp_min(1.0)
                self._last_diag['vs30_target_valid_ratio'] = (
                    pga_vs30_valid_mask.float().sum() / target_valid_count
                ).detach()
                self._last_diag['vs30_target_gate'] = self.target_vs30_gate.detach()

            n_pga = pga_emb.shape[1]
            if self.pga_readout_mode == 'direct_station':
                pga_readout_emb = self._direct_station_pga_embedding(station_memory_emb, sv, n_pga)
                self._last_diag['pga_readout_mode'] = pga_readout_emb.new_tensor(2.0).detach()
            elif self.pga_readout_mode == 'target_cross_attention':
                if self.station_context_mode == 'synchronous_station_target':
                    pga_readout_emb = self.pga_station_target_readout(
                        pga_query_emb,
                        station_feature_emb,
                        sv,
                        query_coords=pga_targets_abs,
                        station_coords=coords_abs,
                    )
                    sync_station_memory = getattr(self.pga_station_target_readout, '_last_station_memory', None)
                    if sync_station_memory is not None:
                        self._record_station_context_diag(
                            sync_station_memory,
                            station_feature_emb,
                            sv,
                            mode_code=3.0,
                            raw_context_emb=sync_station_memory,
                        )
                        station_memory_emb = sync_station_memory
                    self._record_cross_attention_diag('pga_cross', self.pga_station_target_readout, sv)
                    self._record_readout_gate_diag('pga_cross', self.pga_station_target_readout)
                    self._last_diag['pga_readout_mode'] = pga_readout_emb.new_tensor(4.0).detach()
                else:
                    pga_readout_emb = self.pga_cross_attention(
                        pga_query_emb,
                        station_memory_emb,
                        sv,
                        query_coords=pga_targets_abs,
                        station_coords=coords_abs,
                    )
                    self._record_cross_attention_diag('pga_cross', self.pga_cross_attention, sv)
                    self._record_readout_gate_diag('pga_cross', self.pga_cross_attention)
                    self._last_diag['pga_readout_mode'] = pga_readout_emb.new_tensor(3.0).detach()
            elif self.pga_readout_mode == 'query_no_transformer':
                pga_readout_emb = pga_query_emb
                self._last_diag['pga_readout_mode'] = pga_readout_emb.new_tensor(1.0).detach()
            else:
                transformer_input = station_feature_emb
                transformer_coords = coords_abs
                if not (self.skip_transformer or self.no_event_token):
                    transformer_input = self.add_event_token(transformer_input)
                    transformer_coords = torch.cat([coords_center, transformer_coords], dim=1)
                transformer_input = torch.cat([transformer_input, pga_query_emb], dim=1)
                transformer_coords = torch.cat([transformer_coords, pga_targets_abs], dim=1)
                ones_1 = torch.ones(station_mask.shape[0], 1, device=station_mask.device, dtype=torch.bool)
                pga_false = torch.zeros(station_mask.shape[0], n_pga, device=station_mask.device, dtype=torch.bool)

                # padding_mask: [event_token=True, station_mask, pga_target_valid]
                padding_mask = torch.cat([station_mask, ptv], dim=1)
                if not (self.skip_transformer or self.no_event_token):
                    padding_mask = torch.cat([ones_1, padding_mask], dim=1)

                # att_mask: [event_token=True, station_mask=True(all), pga=False] — PGA positions are query-only
                if att_mask is None:
                    att_mask = torch.cat([pga_false], dim=1)  # only PGA part differs
                    if not (self.skip_transformer or self.no_event_token):
                        att_mask = torch.cat([ones_1, torch.ones_like(station_mask), att_mask], dim=1)
                    else:
                        att_mask = torch.cat([torch.ones_like(station_mask), att_mask], dim=1)

                self._record_pga_mask_diag(att_mask, padding_mask, station_mask, ptv)
                transformer_emb = self.transformer(
                    transformer_input.float(),
                    att_mask,
                    padding_mask,
                    coords=transformer_coords,
                )
                self._record_pga_attention_diag(
                    seq_len=transformer_emb.shape[1],
                    station_count=station_mask.shape[1],
                    n_pga=n_pga,
                    has_event=not (self.skip_transformer or self.no_event_token),
                )
                pga_readout_emb = transformer_emb[:, -self.n_pga_targets:, :]
                self._last_diag['pga_readout_mode'] = pga_readout_emb.new_tensor(0.0).detach()

        outputs = []
        if not self.no_event_token:
            if self.event_readout_mode == 'event_cross_attention':
                event_query = self.event_query_token.expand(station_memory_emb.shape[0], 1, -1)
                event_emb = self.event_cross_attention(event_query, station_memory_emb, sv).squeeze(1)
                self._record_cross_attention_diag('event_cross', self.event_cross_attention, sv)
                self._record_readout_gate_diag('event_cross', self.event_cross_attention)
                self._last_diag['event_readout_mode'] = event_emb.new_tensor(1.0).detach()
            elif self.event_readout_mode == 'direct_station_pool':
                event_emb = self._direct_station_pga_embedding(station_memory_emb, sv, 1).squeeze(1)
                self._last_diag['event_readout_mode'] = event_emb.new_tensor(2.0).detach()
            else:
                if self.skip_transformer:
                    event_tokens = torch.stack(
                        [self.mlp_layer(station_feature_emb[:, i, :]) for i in range(station_feature_emb.shape[1])],
                        dim=1,
                    )
                    event_emb = self.maxpool(event_tokens, mask=station_mask)
                else:
                    if transformer_emb is None:
                        transformer_input = self.add_event_token(station_feature_emb)
                        ones_1 = torch.ones(station_mask.shape[0], 1, device=station_mask.device, dtype=torch.bool)
                        padding_mask = torch.cat([ones_1, station_mask], dim=1)
                        transformer_coords = torch.cat([coords_center, coords_abs], dim=1)
                        transformer_emb = self.transformer(
                            transformer_input.float(),
                            padding_mask=padding_mask,
                            coords=transformer_coords,
                        )
                    event_emb = transformer_emb[:, 0, :]  # Select event embedding
                self._last_diag['event_readout_mode'] = event_emb.new_tensor(0.0).detach()

            if (
                pga_readout_emb is not None
                and self.pga_event_context_proj is not None
                and self.pga_event_context_gate is not None
            ):
                event_for_pga = self.pga_event_context_proj(event_emb).unsqueeze(1)
                pga_readout_emb = pga_readout_emb + self.pga_event_context_gate * event_for_pga
                self._last_diag['pga_event_context_gate'] = self.pga_event_context_gate.detach()

            mag_embedding = self.mlp_mag(event_emb)
            if self.output_distribution == 'point':
                out_mag = self.output_model_mag(mag_embedding)
            else:
                alpha_logits, mu, sigma = self.output_model_mag(mag_embedding)
                out_mag = torch.cat([alpha_logits, mu, sigma], dim=-1)  # (batch, n, 1+d+d)

            loc_embedding = self.mlp_loc(event_emb)
            if self.output_distribution == 'point':
                out_loc = self.output_model_loc(loc_embedding)
            else:
                alpha_logits, mu, sigma = self.output_model_loc(loc_embedding)
                out_loc = torch.cat([alpha_logits, mu, sigma], dim=-1)  # (batch, n, 1+d+d)

            outputs.append(out_mag)
            outputs.append(out_loc)

        if self.aux_mag_head is not None:
            outputs.append(self.aux_mag_head(base_waveforms_emb, sv))

        if self.n_pga_targets:
            pga_emb = pga_readout_emb
            pga_emb = torch.stack([self.mlp_pga(pga_emb[:, i, :]) for i in range(pga_emb.shape[1])], dim=1)
            pga_out_tmp = []
            for i in range(pga_emb.shape[1]):
                if self.output_distribution == 'point':
                    pga_out_tmp.append(self.output_model_pga(pga_emb[:, i, :]))
                else:
                    alpha_logits, mu, sigma = self.output_model_pga(pga_emb[:, i, :])
                    pga_out_tmp.append(torch.cat([alpha_logits, mu, sigma], dim=-1))
            output_pga = torch.stack(pga_out_tmp, dim=1)
            outputs.append(output_pga)

        if self.dataset_bias:
            if self.output_distribution == 'point':
                raise NotImplementedError('dataset_bias is only implemented for mdn outputs.')
            assert self.n_datasets is not None
            dataset_bias_term = self.dataset_embedding(dataset).squeeze(-1)
            outputs[0] = self.add_constant_to_mixture(outputs[0], dataset_bias_term)

        return outputs

def get_diting_model(args, station_emb_dim):
    """Build diting model with muP and pretrained weights.

    Uses the ditingbench approach for model construction and weight loading.
    The model is nn.Sequential([0]=encoder, [1]=station adapter).
    """
    depth = args.model_depth if hasattr(args, 'model_depth') else 24
    base_encoder_size = get_encoder_size_dict(width=args.base_width, depth=depth)
    target_encoder_size = get_encoder_size_dict(width=args.target_width, depth=depth)
    frontend = getattr(args, 'diting_frontend', 'vit_adapter')

    add_vit_feature = getattr(args, 'add_vit_feature', True)
    use_extra_extractor = getattr(args, 'use_extra_extractor', False)

    def build_model_pair(encoder_size):
        if frontend == 'vit_adapter':
            encoder = ViTAdapter(
                encoder_size=encoder_size,
                input_length=args.in_samples,
                args=args,
                add_vit_feature=add_vit_feature,
                use_extra_extractor=use_extra_extractor,
                out_x=True,
            )
            head = DitingStationAdapter(
                encoder_dim=encoder.backbone.d_model,
                hidden_channels=args.out_channels,
                output_dim=station_emb_dim,
                pool_queries=getattr(args, 'diting_station_pool_queries', 4),
                pool_temperature=getattr(args, 'diting_station_pool_temperature', 1.0),
                pool_dropout=getattr(args, 'diting_station_pool_dropout', 0.0),
            )
            backbone_module = encoder.backbone
        elif frontend == 'backbone_attn_pool':
            encoder = Encoder_baseline_llama(
                encoder_size=encoder_size,
                input_length=args.in_samples,
                args=args,
            )
            if not hasattr(encoder.backbone, 'mask_ratio'):
                encoder.backbone.set_mask(0.0, 'random')
            head = BackboneAttentionPoolAdapter(
                encoder_dim=encoder.backbone.d_model,
                output_dim=station_emb_dim,
                attn_temperature=getattr(args, 'attn_pool_temperature', 0.5),
                attn_topk=getattr(args, 'attn_pool_topk', 0),
            )
            backbone_module = encoder.backbone
        else:
            raise ValueError(f"Unsupported diting_frontend: {frontend}")
        return nn.Sequential(encoder, head), backbone_module

    # Build base model for muP
    base_model, _ = build_model_pair(base_encoder_size)

    # Build target model
    model, target_backbone = build_model_pair(target_encoder_size)

    # muP: set base shapes
    set_base_shapes(model, base_model)

    # Freeze encoder for linear_probe
    if args.eval_type == 'linear_probe':
        for param in target_backbone.parameters():
            param.requires_grad = False

    # Load pretrained weights
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"=> loading pretrained checkpoint '{args.pretrained}'")
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        state_dict = _extract_pretrained_state_dict(checkpoint, args)
        pretrained_load_mode = getattr(args, 'pretrained_load_mode', 'backbone')
        if pretrained_load_mode == 'backbone':
            state_dict = _filter_backbone_state_dict(state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"pretrained_load_mode: {pretrained_load_mode}")
        print(f"load_state_dict result: {msg}")
    elif args.pretrained:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")

    model = model.to(args.device)
    return model


def build_single_station_model(waveform_model_dims=(500, 500, 500),
                               borehole=False,
                               trace_length=3000,
                               diting_station_pool_queries=4,
                               diting_station_pool_temperature=1.0,
                               diting_station_pool_dropout=0.0,
                               waveform_scale_gain=1.0,
                               waveform_scale_hidden_dim=None,
                               waveform_scale_log_divisor=10.0,
                               waveform_scale_init_gate=0.1,
                               disable_waveform_scale=False,
                               use_amplitude_info=None,
                               single_station_tasks=('mag', 'epidist', 'pga'),
                               single_station_hidden_dim=None,
                               single_station_task_output_init=None,
                               single_station_task_sigma_init=None,
                               diting_args=None,
                               **kwargs):
    """Build a single-station pretraining model sharing the full model adapter path."""
    emb_dim = waveform_model_dims[-1]
    input_channels = 6 if borehole else 3
    if diting_args is not None:
        diting_args.diting_station_pool_queries = diting_station_pool_queries
        diting_args.diting_station_pool_temperature = diting_station_pool_temperature
        diting_args.diting_station_pool_dropout = diting_station_pool_dropout
    waveform_model = get_diting_model(diting_args, station_emb_dim=emb_dim)
    scale_feature_dim = 3 * input_channels + 2
    waveform_scale_proj = WaveformScaleEmbedding(
        scale_feature_dim,
        emb_dim,
        hidden_dim=waveform_scale_hidden_dim,
        log_divisor=waveform_scale_log_divisor,
    )
    return SingleStationMultiTaskModel(
        waveform_model,
        emb_dim,
        input_channels=input_channels,
        tasks=single_station_tasks,
        waveform_scale_proj=waveform_scale_proj,
        waveform_scale_gain=waveform_scale_gain,
        waveform_scale_init_gate=waveform_scale_init_gate,
        disable_waveform_scale=disable_waveform_scale,
        use_amplitude_info=use_amplitude_info,
        task_hidden_dim=single_station_hidden_dim,
        task_output_init=single_station_task_output_init,
        task_sigma_init=single_station_task_sigma_init,
    )


def build_transformer_model(max_stations,
                            waveform_model_dims=(500, 500, 500),
                            output_mlp_dims=(150, 100, 50, 30, 10),
                            output_location_dims=(150, 100, 50, 50, 50),
                            wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                            mad_params={"n_heads": 10,
                                        "att_dropout": 0.0,
                                        },
                            ffn_params={'hidden_dim': 1000},
                            transformer_layers=6,
                            hidden_dropout=0.0,
                            activation='relu',
                            n_pga_targets=0,
                            location_mixture=5,
                            pga_mixture=5,
                            magnitude_mixture=5,
                            borehole=False,
                            bias_mag_mu=1.8,
                            bias_mag_sigma=0.2,
                            bias_loc_mu=0,
                            bias_loc_sigma=1,
                            event_token_init_range=None,
                            dataset_bias=False,
                            n_datasets=None,
                            no_event_token=False,
                            trace_length=3000,
                            downsample=5,
                            rotation=None,
                            rotation_anchor=None,
                            skip_transformer=False,
                            alternative_coords_embedding=False,
                            diting_station_pool_queries=4,
                            diting_station_pool_temperature=1.0,
                            diting_station_pool_dropout=0.0,
                            waveform_scale_gain=1.0,
                            waveform_scale_hidden_dim=None,
                            waveform_scale_log_divisor=10.0,
                            waveform_scale_init_gate=0.1,
                            disable_waveform_scale=False,
                            use_amplitude_info=None,
                            use_coords_rel=False,
                            use_coords_abs=True,
                            use_coords_rel_abs_fusion=False,
                            coords_abs_weight=0.1,
                            coord_fusion_mode='add',
                            aux_mag_head=False,
                            aux_mag_hidden_dim=None,
                            aux_mag_sigma=0.3,
                            output_distribution='mdn',
                            pga_readout_mode='query_transformer',
                            event_readout_mode='event_transformer',
                            readout_n_heads=None,
                            readout_dropout=None,
                            query_token_init_range=0.02,
                            pga_distance_bias=False,
                            pga_distance_bias_hidden_dim=64,
                            readout_layers=1,
                            pga_readout_layers=None,
                            event_readout_layers=None,
                            readout_ffn_hidden_dim=None,
                            readout_first_residual=False,
                            readout_first_residual_gate_init=None,
                            readout_residual_gates=False,
                            readout_residual_gate_init=0.0,
                            readout_ffn_gate_init=None,
                            readout_inject_base_query=False,
                            readout_query_injection_gate_init=1.0,
                            readout_use_ffn=True,
                            station_context_mode='off',
                            station_context_gate_init=0.0,
                            pga_use_event_context=False,
                            pga_event_context_init_gate=0.0,
                            use_vs30=False,
                            vs30_reference_mps=760.0,
                            vs30_init_gate=0.0,
                            use_rope=None,
                            use_team_rope=False,
                            rope_coord_mode='relative_xy_km',
                            rope_coord_scale=100.0,
                            rope_base=10000.0,
                            rope_lat_origin=37.0,
                            pga_attention_diagnostics=False,
                            pga_mask_sanity_check=False,
                            diting_args=None,
                            **kwargs):
    if kwargs:
        print(f'Warning: Unused model parameters: {", ".join(kwargs.keys())}')

    emb_dim = waveform_model_dims[-1]
#    emb_dim = diting_args.target_width
    mad_params = mad_params.copy()  # Avoid modifying the input dicts
    ffn_params = ffn_params.copy()
    legacy_rope_requested = bool(use_team_rope or mad_params.get('use_team_rope', False))
    if use_rope is None:
        station_self_rope = legacy_rope_requested
        pga_readout_rope = False
    else:
        station_self_rope = bool(use_rope)
        pga_readout_rope = bool(use_rope)
    mad_params['use_team_rope'] = station_self_rope
    if use_rope is None:
        mad_params.setdefault('rope_coord_mode', rope_coord_mode)
        mad_params.setdefault('rope_coord_scale', rope_coord_scale)
        mad_params.setdefault('rope_base', rope_base)
        mad_params.setdefault('rope_lat_origin', rope_lat_origin)
    else:
        mad_params['rope_coord_mode'] = rope_coord_mode
        mad_params['rope_coord_scale'] = rope_coord_scale
        mad_params['rope_base'] = rope_base
        mad_params['rope_lat_origin'] = rope_lat_origin
    if readout_n_heads is None:
        readout_n_heads = mad_params.get('n_heads', 1)
    if readout_dropout is None:
        readout_dropout = mad_params.get('att_dropout', 0.0)

    #   Single station model
    if borehole:
        input_shape = (trace_length, 6)
        metadata_shape = (4,)
    else:
        input_shape = (trace_length, 3)
        metadata_shape = (3,)

#    waveform_model = NormalizedScaleEmbedding(input_shape, downsample=downsample, activation=activation,
#                                              mlp_dims=waveform_model_dims)
#    mlp_mag_single_station = MLP((waveform_model.mlp.mlp[-1].out_features,), output_mlp_dims, activation=activation) #Modified line
    if diting_args is not None:
        diting_args.diting_station_pool_queries = diting_station_pool_queries
        diting_args.diting_station_pool_temperature = diting_station_pool_temperature
        diting_args.diting_station_pool_dropout = diting_station_pool_dropout
    waveform_model = get_diting_model(diting_args, station_emb_dim=waveform_model_dims[-1])

    #   Event model

    if n_pga_targets:
        att_masking = True
    else:
        att_masking = False

    if not no_event_token:
        transformer_max_stations = max_stations + 1 + n_pga_targets
    else:
        transformer_max_stations = max_stations + n_pga_targets

    if not skip_transformer:
        transformer = Transformer(max_stations=transformer_max_stations, emb_dim=emb_dim, att_masking=att_masking,
                                  layers=transformer_layers, hidden_dropout=hidden_dropout, mad_params=mad_params,
                                  ffn_params=ffn_params)
    else:
        transformer = None

    if output_distribution not in ('mdn', 'point'):
        raise ValueError(f"output_distribution must be 'mdn' or 'point', got {output_distribution!r}.")

    mlp_mag = MLP((emb_dim,), output_mlp_dims, activation=activation)
    if output_distribution == 'point':
        output_model_mag = PointOutput((output_mlp_dims[-1],), d=1, bias_mu=bias_mag_mu, activation=None)
    else:
        output_model_mag = MixtureOutput((output_mlp_dims[-1],), n=magnitude_mixture, bias_mu=bias_mag_mu, activation='relu',
                                     bias_sigma=bias_mag_sigma)

    mlp_loc = MLP((emb_dim,), output_location_dims, activation=activation)
    if output_distribution == 'point':
        output_model_loc = PointOutput((output_location_dims[-1],), d=3, bias_mu=bias_loc_mu, activation=None)
    else:
        output_model_loc = MixtureOutput((output_location_dims[-1],), n=location_mixture, d=3, bias_mu=bias_loc_mu,activation=None,
                                         bias_sigma=bias_loc_sigma)

    mlp_pga = MLP((emb_dim,), output_mlp_dims, activation=activation)
    if output_distribution == 'point':
        output_model_pga = PointOutput((output_mlp_dims[-1],), d=1, bias_mu=0, activation=None)
    else:
        output_model_pga = MixtureOutput((output_mlp_dims[-1],), n=pga_mixture, bias_mu=0, bias_sigma=1, activation=None)

    # Module instantiation
    position_embedding = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole, rotation=rotation, rotation_anchor=rotation_anchor)

    if not no_event_token:
        add_event_token = AddEventToken(fixed=False, init_range=event_token_init_range)
    else:
        add_event_token = None

    if dataset_bias:
        add_constant_to_mixture = AddConstantToMixture()
    else:
        add_constant_to_mixture = None

    scale_feature_dim = 3 * input_shape[-1] + 2
    full_waveform_scale_proj = WaveformScaleEmbedding(
        scale_feature_dim,
        emb_dim,
        hidden_dim=waveform_scale_hidden_dim,
        log_divisor=waveform_scale_log_divisor,
    )
    aux_mag_module = AuxiliaryMagnitudeHead(
        emb_dim,
        hidden_dim=aux_mag_hidden_dim,
        sigma_init=aux_mag_sigma,
    ) if aux_mag_head else None
    effective_pga_readout_layers = pga_readout_layers if pga_readout_layers is not None else readout_layers
    station_context_encoder = None
    if station_context_mode in (
        'firstres_transformer_pre_readout',
        'gated_transformer_pre_readout_firstres',
    ):
        station_context_encoder = StationContextEncoder(
            emb_dim,
            layers=effective_pga_readout_layers,
            mad_params=mad_params,
            ffn_params=ffn_params,
            residual_gate_init=readout_residual_gate_init,
            ffn_gate_init=readout_ffn_gate_init,
        )
    pga_station_target_readout = None
    if station_context_mode == 'synchronous_station_target':
        pga_station_target_readout = SynchronousStationTargetReadout(
            emb_dim,
            readout_n_heads,
            station_layers=effective_pga_readout_layers,
            mad_params=mad_params,
            ffn_params=ffn_params,
            att_dropout=readout_dropout,
            distance_bias=pga_distance_bias,
            distance_hidden_dim=pga_distance_bias_hidden_dim,
            ffn_hidden_dim=readout_ffn_hidden_dim,
            first_residual=readout_first_residual,
            first_residual_gate_init=readout_first_residual_gate_init,
            residual_gates=readout_residual_gates,
            residual_gate_init=readout_residual_gate_init,
            ffn_gate_init=readout_ffn_gate_init,
            inject_base_query=readout_inject_base_query,
            query_injection_gate_init=readout_query_injection_gate_init,
            use_ffn=readout_use_ffn,
            use_rope=pga_readout_rope,
            rope_coord_mode=rope_coord_mode,
            rope_coord_scale=rope_coord_scale,
            rope_base=rope_base,
            rope_lat_origin=rope_lat_origin,
        )
    full_model = FullModel(waveform_model, position_embedding, transformer, mlp_mag, output_model_mag, mlp_loc,
                             output_model_loc, mlp_pga, output_model_pga, skip_transformer, alternative_coords_embedding,
                             metadata_shape, emb_dim, no_event_token, add_event_token, n_pga_targets, dataset_bias,
                             add_constant_to_mixture, n_datasets, waveform_scale_proj=full_waveform_scale_proj,
                             waveform_scale_gain=waveform_scale_gain,
                             waveform_scale_init_gate=waveform_scale_init_gate,
                             disable_waveform_scale=disable_waveform_scale,
                             use_amplitude_info=use_amplitude_info,
                             use_coords_rel=use_coords_rel,
                             use_coords_abs=use_coords_abs,
                             use_coords_rel_abs_fusion=use_coords_rel_abs_fusion,
                             coords_abs_weight=coords_abs_weight,
                             coord_fusion_mode=coord_fusion_mode,
                             aux_mag_head=aux_mag_module,
                             output_distribution=output_distribution,
                             pga_readout_mode=pga_readout_mode,
                             event_readout_mode=event_readout_mode,
                             readout_n_heads=readout_n_heads,
                             readout_dropout=readout_dropout,
                             query_token_init_range=query_token_init_range,
                             pga_distance_bias=pga_distance_bias,
                             pga_distance_bias_hidden_dim=pga_distance_bias_hidden_dim,
                             readout_layers=readout_layers,
                             pga_readout_layers=pga_readout_layers,
                             event_readout_layers=event_readout_layers,
                             readout_ffn_hidden_dim=readout_ffn_hidden_dim,
                             readout_first_residual=readout_first_residual,
                             readout_first_residual_gate_init=readout_first_residual_gate_init,
                             readout_residual_gates=readout_residual_gates,
                             readout_residual_gate_init=readout_residual_gate_init,
                             readout_ffn_gate_init=readout_ffn_gate_init,
                             readout_inject_base_query=readout_inject_base_query,
                             readout_query_injection_gate_init=readout_query_injection_gate_init,
                             readout_use_ffn=readout_use_ffn,
                             station_context_mode=station_context_mode,
                             station_context_gate_init=station_context_gate_init,
                             pga_use_event_context=pga_use_event_context,
                             pga_event_context_init_gate=pga_event_context_init_gate,
                             use_vs30=use_vs30,
                             vs30_reference_mps=vs30_reference_mps,
                             vs30_init_gate=vs30_init_gate,
                             use_rope=pga_readout_rope,
                             rope_coord_mode=rope_coord_mode,
                             rope_coord_scale=rope_coord_scale,
                             rope_base=rope_base,
                             rope_lat_origin=rope_lat_origin,
                             station_context_encoder=station_context_encoder,
                             pga_station_target_readout=pga_station_target_readout,
                             pga_attention_diagnostics=pga_attention_diagnostics,
                             pga_mask_sanity_check=pga_mask_sanity_check)
    return full_model


class EnsembleEvaluateModel:
    def __init__(self, config, max_ensemble_size=None, loss_limit=None, device='cpu', diting_args=None):
        self.config = config
        self.ensemble = config.get('ensemble', 1)
        true_ensemble_size = self.ensemble
        if max_ensemble_size is not None:
            self.ensemble = min(self.ensemble, max_ensemble_size)
        self.models = []
        self.device = device  # Store the device

        for ens_id in range(self.ensemble):
            model_params = config['model_params'].copy()
            if config['training_params'].get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                model_params['rotation'] = np.pi / 4 * ens_id / (true_ensemble_size - 1)
            full_model = build_transformer_model(**model_params, diting_args=diting_args)

            # Move models to the specified device
            self.models.append(full_model.to(self.device))

        self.loss_limit = loss_limit

    def predict_generator(self, generator, **kwargs):
        #Note: Implement generator if really needed, requires DataLoader object from torch

        #Create a torch DataLoader from the generator if needed, then call predict with the data

        raise NotImplementedError("Predict Generator not supported")

    def predict(self, inputs):
        # Ensure inputs are tensors and on the correct device
        inputs = [torch.as_tensor(i, device=self.device) for i in inputs]

        with torch.no_grad():  # Disable gradient calculation
            preds = [model(*inputs) for model in self.models]  # Call model with inputs, unpack if list

        merged = self.merge_preds(preds)
        if isinstance(merged, list) and len(merged) > 0:
            merged[0] = clip_magnitude_mixture(merged[0])
        else:
            merged = clip_magnitude_mixture(merged)
        return merged

    @staticmethod
    def merge_preds(preds):
        merged_preds = []

        if isinstance(preds[0], list):
            iter = range(len(preds[0]))
        else:
            iter = [-1]

        for i in iter:  # Iterate over mag, loc, pga, ...
            if i != -1:
                pred_item = torch.cat([x[i] for x in preds], dim=-2)
            else:
                pred_item = torch.cat(preds, dim=-2)

            if len(pred_item.shape) == 3:
                pred_item[:, :, 0] /= torch.sum(pred_item[:, :, 0], dim=-1, keepdim=True)
            elif len(pred_item.shape) == 4:
                pred_item[:, :, :, 0] /= torch.sum(pred_item[:, :, :, 0], dim=-1, keepdim=True)
            else:
                raise ValueError("Encountered prediction of unexpected shape")
            merged_preds.append(pred_item)

        if len(merged_preds) == 1:
            return merged_preds[0].cpu().numpy()  # Move to CPU and convert to NumPy
        else:
            return [p.cpu().numpy() for p in merged_preds] # Move to CPU and convert to NumPy

    def load_weights(self, weights_path):
        tmp_models = self.models
        self.models = []
        removed_models = 0
        for ens_id, model in enumerate(tmp_models):
            tmp_weights_path = os.path.join(weights_path, f'{ens_id}')

            # Find latest .pth checkpoint
            pth_files = sorted([x for x in os.listdir(tmp_weights_path) if x.endswith('.pth')])
            if not pth_files:
                raise FileNotFoundError(f'No .pth checkpoints found in {tmp_weights_path}')

            if self.loss_limit is not None:
                # Check val_loss from the latest checkpoint
                ckpt = torch.load(os.path.join(tmp_weights_path, pth_files[-1]), map_location='cpu')
                if isinstance(ckpt, dict) and 'loss' in ckpt and ckpt['loss'] > self.loss_limit:
                    removed_models += 1
                    continue

            weight_file = os.path.join(tmp_weights_path, pth_files[-1])
            ckpt = torch.load(weight_file, map_location='cpu')
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)

            self.models.append(model.to(self.device))

        if removed_models > 0:
            print(f'Removed {removed_models} models not fulfilling loss limit')
