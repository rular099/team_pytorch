import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mup import set_base_shapes

from dtbench.models.diting.models.vit_adapter import ViTAdapter
from dtbench.models.diting.models.backbone_ablation import get_encoder_size_dict
from dtbench.training.modeling import (
    build_interaction_indexes,
    parse_hps,
    _extract_pretrained_state_dict,
    _filter_backbone_state_dict,
)
from LSD.models.head import EncoderFeatures

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

    def forward(self, x, att_mask=None, padding_mask=None):
        # The inputs are already handled by the calling function.
        for block in self.blocks:
            x = block(x, att_mask, padding_mask)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, emb_dim, hidden_dim, att_dropout=0.0, initializer_range=0.02, eps=1e-5):
        super().__init__()
        self.attention = MultiHeadSelfAttention(n_heads=n_heads, emb_dim=emb_dim, att_dropout=att_dropout, initializer_range=initializer_range)
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

    def forward(self, x, att_mask=None, padding_mask=None):
        modified_x, _ = self.attention(x, attn_mask=att_mask, padding_mask=padding_mask)
        modified_x = self.dropout1(modified_x)
        x = self.norm1(x + modified_x)
        modified_x = self.ffn(x)
        modified_x = self.dropout2(modified_x)
        x = self.norm2(x + modified_x)
        return x


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

        self.lat_coeff = 2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        self.lon_coeff = 2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        self.depth_coeff = 2 * np.pi * 1. / min_depth * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))

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

            lat_base = rotated[:, :, 0:1] * torch.tensor(self.lat_coeff, device=x.device)  # Move coefficients to device
            lon_base = rotated[:, :, 1:2] * torch.tensor(self.lon_coeff, device=x.device)  # Move coefficients to device
            depth_base = x[:, :, 2:3] * torch.tensor(self.depth_coeff, device=x.device)  # Move coefficients to device
        else:
            lat_base = x[:, :, 0:1] * torch.tensor(self.lat_coeff, device=x.device)  # Move coefficients to device
            lon_base = x[:, :, 1:2] * torch.tensor(self.lon_coeff, device=x.device)  # Move coefficients to device
            depth_base = x[:, :, 2:3] * torch.tensor(self.depth_coeff, device=x.device)  # Move coefficients to device

        if self.borehole:
            if self.fake_borehole:
                # Use third value for the depth of the top station and 0 for the borehole depth
                depth_base = x[:, :, 2:3] * torch.tensor(self.depth_coeff, device=x.device) * 0
                depth2_base = x[:, :, 2:3] * torch.tensor(self.depth_coeff, device=x.device)  # Move coefficients to device
            else:
                depth2_base = x[:, :, 3:4] * torch.tensor(self.depth_coeff, device=x.device)  # Move coefficients to device

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
    def __init__(self, n_heads, emb_dim, initializer_range=0.02, att_dropout=0.0, infinity=1e6):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = int(emb_dim // n_heads)
        self.infinity = infinity
        self.att_dropout = att_dropout
        self.initializer_range = initializer_range #Added
        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)
        self.WO = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, attn_mask=None, padding_mask=None):
        self.stations = x.shape[1]

        d_key = self.d_key
        n_heads = self.n_heads
        stations = self.stations

        q = self.WQ(x)  # (batch, stations, key*n_heads)
        q = q.reshape(-1, stations, d_key, n_heads)
        q = q.permute(0, 3, 1, 2)  # (batch, n_heads, stations, key)

        k = self.WK(x)  # (batch, stations, key*n_heads)
        k = k.reshape(-1, stations, d_key, n_heads)
        k = k.permute(0, 3, 2, 1)  # (batch, n_heads, key, stations)

        score = torch.matmul(q, k) / np.sqrt(d_key)  # (batch, n_heads, stations, stations)

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



class StatPool1d(nn.Module):
    """Replace GAP with mean+max+std pooling, then project back to original dim."""
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels * 3, channels)

    def forward(self, x):
        # x: (B, C, T)
        mean_pool = x.mean(dim=-1)              # (B, C)
        max_pool = x.max(dim=-1).values         # (B, C)
        std_pool = x.std(dim=-1)                # (B, C)
        out = torch.cat([mean_pool, max_pool, std_pool], dim=-1)  # (B, 3C)
        return self.proj(out).unsqueeze(-1)      # (B, C, 1) — same shape as GAP output


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
    missing = [c for c in res_comps if c not in name_to_idx]
    if missing:
        raise ValueError(
            f'res_comps {missing} not in model output_layout {output_layout}. '
            f'Enable the corresponding heads in the model config or remove '
            f'them from res_comps.'
        )
    sel_pred = [outputs[name_to_idx[c]] for c in res_comps]
    sel_true = [labels[name_to_idx[c]] for c in res_comps]
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
                 add_constant_to_mixture, n_datasets, waveform_scale_proj=None, waveform_scale_gain=1.0):
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
        if self.n_pga_targets > 0:
            self.output_layout += ['pga']

        if dataset_bias:
            self.dataset_embedding = nn.Embedding(n_datasets, 1)
        self.Masking_nd_0_23 = Masking_nd(0, (2, 3))
        self.Masking_nd_0_2 = Masking_nd(0, axis=2, nodim=True)
#        self.layernorm = LayerNormalization()
        self.layernorm = nn.LayerNorm(emb_dim)
        if self.skip_transformer:
            mlp_input_length = self.emb_dim
            if self.alternative_coords_embedding:
                mlp_input_length += self.metadata_shape[0]
            self.mlp_layer = MLP((mlp_input_length,), [self.emb_dim, self.emb_dim], activation='relu')
            self.maxpool = GlobalMaxPooling1DMasked()

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


    def _extract_scale(self, waveform):
        scale = torch.log(torch.amax(torch.abs(waveform), dim=(2, 3)) + 1e-6)
        return scale.unsqueeze(-1)

    def forward(self, waveform_inp, metadata_inp, station_valid,
                pga_targets_inp=None, pga_target_valid=None,
                dataset=None, att_mask=None):
        raw_waveform = waveform_inp.clone()
        waveform_inp = self._normalize(waveform_inp, mode='std', axis=3)
        # Apply explicit masks instead of inferring "validity == nonzero".
        # station_valid: (B, S) bool. pga_target_valid: (B, n_pga) bool.
        sv = station_valid.bool()
        waveforms_masked = waveform_inp * sv[:, :, None, None].float()
        coords_masked = metadata_inp * sv[:, :, None].float()

        waveforms_emb = torch.stack([self.waveform_model(waveforms_masked[:, i, :, :]) for i in range(waveforms_masked.shape[1])] , dim=1)
        if self.waveform_scale_proj is not None:
            scale_emb = self.waveform_scale_proj(self._extract_scale(raw_waveform))
            waveforms_emb = waveforms_emb + self.waveform_scale_gain * scale_emb
        waveforms_emb = self.layernorm(waveforms_emb)

        if not self.alternative_coords_embedding:
            coords_emb = self.position_embedding(coords_masked)
            emb = waveforms_emb + coords_emb
        else:
            emb = torch.cat([waveforms_emb, coords_masked], dim=-1)

        if not (self.skip_transformer or self.no_event_token):
            emb = self.add_event_token(emb)

        # padding_mask: True=valid position, used for key masking + output zeroing (matches TF `mask`)
        # att_mask: True=attendable as key, used only for additional key masking (matches TF `att_mask`)
        station_mask = sv  # (batch, n_stations) — comes from explicit station_valid input

        if self.n_pga_targets:
            assert pga_target_valid is not None, \
                'pga_target_valid must be provided when n_pga_targets > 0'
            ptv = pga_target_valid.bool()
            pga_targets_masked = pga_targets_inp * ptv[:, :, None].float()
            pga_emb = self.position_embedding(pga_targets_masked)
            emb = torch.cat([emb, pga_emb], dim=1)

            n_pga = pga_emb.shape[1]
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

            emb = self.transformer(emb.float(), att_mask, padding_mask)
        else:
            if self.skip_transformer:
                emb = torch.stack([self.mlp_layer(emb[:, i, :]) for i in range(emb.shape[1])], dim=1)
                emb = self.maxpool(emb, mask=station_mask)
            else:
                # padding_mask: [event_token=True, station_mask]
                padding_mask = station_mask.clone()
                if not (self.skip_transformer or self.no_event_token):
                    ones_1 = torch.ones(station_mask.shape[0], 1, device=station_mask.device, dtype=torch.bool)
                    padding_mask = torch.cat([ones_1, padding_mask], dim=1)
                emb = self.transformer(emb.float(), padding_mask=padding_mask)

        outputs = []
        if not self.no_event_token:
            if self.skip_transformer:
                event_emb = emb
            else:
                event_emb = emb[:, 0, :]  # Select event embedding

            mag_embedding = self.mlp_mag(event_emb)
            #out_mag = self.output_model_mag(mag_embedding)
            alpha_logits, mu, sigma = self.output_model_mag(mag_embedding)
            out_mag = torch.cat([alpha_logits, mu, sigma], dim=-1)  # (batch, n, 1+d+d)

            loc_embedding = self.mlp_loc(event_emb)
            #out_loc = self.output_model_loc(loc_embedding)
            alpha_logits, mu, sigma = self.output_model_loc(loc_embedding)
            out_loc = torch.cat([alpha_logits, mu, sigma], dim=-1)  # (batch, n, 1+d+d)

            outputs.append(out_mag)
            outputs.append(out_loc)

        if self.n_pga_targets:
            pga_emb = emb[:, -self.n_pga_targets:, :]  # Select embeddings for pga
            pga_emb = torch.stack([self.mlp_pga(pga_emb[:, i, :]) for i in range(pga_emb.shape[1])], dim=1)
            #output_pga = torch.stack([self.output_model_pga(pga_emb[:, i, :]) for i in range(pga_emb.shape[1])], dim=1)
            pga_out_tmp = []
            for i in range(pga_emb.shape[1]):
                alpha_logits, mu, sigma = self.output_model_pga(pga_emb[:, i, :])
                pga_out_tmp.append(torch.cat([alpha_logits, mu, sigma], dim=-1))
            output_pga = torch.stack(pga_out_tmp, dim=1)
            outputs.append(output_pga)

        if self.dataset_bias:
            assert self.n_datasets is not None
            dataset_bias_term = self.dataset_embedding(dataset).squeeze(-1)
            outputs[0] = self.add_constant_to_mixture(outputs[0], dataset_bias_term)

        return outputs

def get_diting_model(args):
    """Build diting model (ViTAdapter + EncoderFeatures) with muP and pretrained weights.

    Uses the ditingbench approach for model construction and weight loading.
    The model is nn.Sequential([0]=ViTAdapter encoder, [1]=EncoderFeatures head).
    """
    depth = args.model_depth if hasattr(args, 'model_depth') else 24
    base_encoder_size = get_encoder_size_dict(width=args.base_width, depth=depth)
    target_encoder_size = get_encoder_size_dict(width=args.target_width, depth=depth)

    add_vit_feature = getattr(args, 'add_vit_feature', True)
    use_extra_extractor = getattr(args, 'use_extra_extractor', False)

    # Build base model for muP
    base_encoder = ViTAdapter(
        encoder_size=base_encoder_size,
        input_length=args.in_samples,
        args=args,
        add_vit_feature=add_vit_feature,
        use_extra_extractor=use_extra_extractor,
        out_x=True,
    )
    base_head = EncoderFeatures(encoder_dim=base_encoder.backbone.d_model, args=args)
    base_model = nn.Sequential(base_encoder, base_head)

    # Build target model
    target_encoder = ViTAdapter(
        encoder_size=target_encoder_size,
        input_length=args.in_samples,
        args=args,
        add_vit_feature=add_vit_feature,
        use_extra_extractor=use_extra_extractor,
        out_x=True,
    )
    target_head = EncoderFeatures(encoder_dim=target_encoder.backbone.d_model, args=args)
    model = nn.Sequential(target_encoder, target_head)

    # muP: set base shapes
    set_base_shapes(model, base_model)

    # Freeze encoder for linear_probe
    if args.eval_type == 'linear_probe':
        for param in model[0].backbone.parameters():
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
                            waveform_scale_gain=1.0,
                            diting_args=None,
                            **kwargs):
    if kwargs:
        print(f'Warning: Unused model parameters: {", ".join(kwargs.keys())}')

    emb_dim = waveform_model_dims[-1]
#    emb_dim = diting_args.target_width
    mad_params = mad_params.copy()  # Avoid modifying the input dicts
    ffn_params = ffn_params.copy()

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
    waveform_model = get_diting_model(diting_args)
    # Replace GAP in EncoderFeatures with StatPool1d (mean+max+std)
    encoder_features = waveform_model[1]
    encoder_features.gap = StatPool1d(diting_args.out_channels)
    dt2team = MLP((diting_args.out_channels,), [diting_args.out_channels, waveform_model_dims[-1]], activation=activation)
    waveform_model.add_module('dt2team', dt2team)

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

    mlp_mag = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model_mag = MixtureOutput((output_mlp_dims[-1],), n=magnitude_mixture, bias_mu=bias_mag_mu, activation='relu',
                                 bias_sigma=bias_mag_sigma)

    mlp_loc = MLP((emb_dim,), output_location_dims, activation=activation)
    output_model_loc = MixtureOutput((output_location_dims[-1],), n=location_mixture, d=3, bias_mu=bias_loc_mu,activation=None,
                                     bias_sigma=bias_loc_sigma)

    mlp_pga = MLP((emb_dim,), output_mlp_dims, activation=activation)
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

    full_waveform_scale_proj = nn.Linear(1, emb_dim)
    full_model = FullModel(waveform_model, position_embedding, transformer, mlp_mag, output_model_mag, mlp_loc,
                             output_model_loc, mlp_pga, output_model_pga, skip_transformer, alternative_coords_embedding,
                             metadata_shape, emb_dim, no_event_token, add_event_token, n_pga_targets, dataset_bias,
                             add_constant_to_mixture, n_datasets, waveform_scale_proj=full_waveform_scale_proj,
                             waveform_scale_gain=waveform_scale_gain)
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
