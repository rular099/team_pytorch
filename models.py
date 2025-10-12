import os
import numpy as np
import pickle
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from diting.finetuneing.utils import create_model

class StripMask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        # Assuming the mask is the second element in a tuple (data, mask)
        return x[0]

class MLP(nn.Module):
    def __init__(self, input_shape, dims=(100, 50), activation='ReLU', last_activation=None):
        super().__init__()
        if last_activation is None:
            last_activation = activation
        all_dims = input_shape + dims
        layers = []
        for i,d in enumerate(all_dims[:-1]):
            layers += [nn.Linear(all_dims[i], all_dims[i+1]), getattr(nn,activation)]
        layers += [nn.Linear(all_dims[-2], all_dims[-1]), getattr(nn,last_activation)]
        self.layers = nn.Sequential(*layers)

    def forward(self,x)
        return self.layers(x)

class MixtureOutput(nn.Module):
    def __init__(self, input_shape, n, d=1, activation='relu', eps=1e-4, bias_mu=1.8, bias_sigma=0.2, name=None):
        super().__init__()
        
        self.dense_alpha = nn.Linear(input_shape[0], n)
        self.dense_mu = nn.Linear(input_shape[0], n * d)
        self.dense_sigma = nn.Linear(input_shape[0], n * d)
        
        self.activation = getattr(F, activation)
        self.eps = eps
        
        # Initialize biases
        nn.init.constant_(self.dense_mu.bias, bias_mu)
        nn.init.constant_(self.dense_sigma.bias, bias_sigma)

    def forward(self, inp):
        alpha = self.dense_alpha(inp)
        alpha = F.softmax(alpha, dim=1).unsqueeze(2)  # Softmax and reshape to (n, 1)
        
        mu = self.dense_mu(inp)
        mu = self.activation(mu).view(-1, alpha.size(1), mu.size(1) // alpha.size(1))  # Reshape to (n, d)
        
        sigma = self.dense_sigma(inp)
        sigma = self.activation(sigma) + self.eps  # Add epsilon to avoid division by 0
        sigma = sigma.view(-1, alpha.size(1), sigma.size(1) // alpha.size(1))  # Reshape to (n, d)
        
        out = torch.cat([alpha, mu, sigma], dim=2)
        return out

class NormalizedScaleEmbedding(nn.Module):
    def __init__(self, input_shape, activation='relu', downsample=1, mlp_dims=(500, 300, 200, 150), eps=1e-8):
        super().__init__()
        self.act = getattr(F, activation)
        self.inp_shape = input_shape
        self.downsample = downsample
        self.mlp_dims = mlp_dims
        self.eps = eps
        
        # Define the layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(downsample, 1), stride=(downsample, 1))
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(16, 3), stride=(1, 3))
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=16)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(128, 32, kernel_size=8)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv6 = nn.Conv1d(32, 32, kernel_size=8)
        self.conv7 = nn.Conv1d(32, 16, kernel_size=4)
        self.flatten = nn.Flatten()
        self.mlp = MLP(input_shape=(865,), dims=self.mlp_dims, activation=self.activation)

    def forward(self, inp):
        # Normalize the input
        max_abs = torch.amax(torch.abs(inp), dim=(1,2), keepdim=True) + self.eps
        x = inp / max_abs
        
        # Expand dimensions
        x = x.unsqueeze(1)
        
        # Compute scale
        scale = torch.log(torch.amax(torch.abs(inp), dim=(1,2) keepdim=False) + self.eps) / 100
        scale = scale.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        
        # Reshape
        x = x.view(x.size(0), -1, 32 * self.inp_shape[-1] // 3)
        
        # Apply more convolutional and pooling layers
        x = self.conv3(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.act(x)
        x = self.pool3(x)
        x = self.conv6(x)
        x = self.act(x)
        x = self.conv7(x)
        x = self.act(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Concatenate with scale
        x = torch.cat([x, scale.squeeze()], dim=1)
        
        # Pass through MLP
        x = self.mlp(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, max_stations=32, emb_dim=500, layers=6, att_masking=False, hidden_dropout=0.0,
                 mad_params={}, ffn_params={}, norm_params={}):
        super().__init__()
        self.att_masking = att_masking
        self.hidden_dropout = hidden_dropout
        
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MultiHeadSelfAttention(**mad_params),
                PointwiseFeedForward(**ffn_params),
                LayerNormalization(**norm_params),
                LayerNormalization(**norm_params)
            ])
            for _ in range(layers)
        ])
    
    def forward(self, *inputs):
        if self.att_masking:
            inp, att_mask = inputs
        else:
            inp = inputs[0]
            att_mask = None
        
        x = inp
        for attention_layer, ffn_layer, norm1_layer, norm2_layer in self.blocks:
            if att_mask is not None:
                modified_x = attention_layer(x, att_mask)
            else:
                modified_x = attention_layer(x)
            if self.hidden_dropout > 0:
                modified_x = Dropout(self.hidden_dropout)(modified_x)
            x = norm1_layer(x + modified_x)
            modified_x = ffn_layer(x)
            if self.hidden_dropout > 0:
                modified_x = Dropout(self.hidden_dropout)(modified_x)
            x = norm2_layer(x + modified_x)
        
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, wavelengths, emb_dim, borehole=False, rotation=None, rotation_anchor=None, **kwargs):
        super().__init__()
        self.wavelengths = wavelengths  # Format: [(min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth)]
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError('Rotations in the positional embedding require a rotation anchor')

        if rotation is not None:
            c, s = np.cos(rotation), np.sin(rotation)
            self.rotation_matrix = torch.tensor(((c, -s), (s, c)), dtype=torch.float32)
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
        self.lat_coeff = torch.tensor(2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim)), dtype=torch.float32)
        self.lon_coeff = torch.tensor(2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim)), dtype=torch.float32)
        self.depth_coeff = torch.tensor(2 * np.pi * 1. / min_depth * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim)), dtype=torch.float32)
        
        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9
        
        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim *= 2
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = 2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        self.mask = torch.tensor(self.mask, dtype=torch.int64)
        self.fake_borehole = False

    def forward(self, x, mask=None):
        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1]
            lon_base *= torch.cos(lat_base * np.pi / 180)

            lat_base -= self.rotation_anchor[0]
            lon_base -= self.rotation_anchor[1] * torch.cos(self.rotation_anchor[0] * np.pi / 180)

            latlon = torch.stack([lat_base, lon_base], dim=-1)
            rotated = torch.matmul(latlon, self.rotation_matrix)

            lat_base = rotated[:, :, 0:1] * self.lat_coeff
            lon_base = rotated[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        else:
            lat_base = x[:, :, 0:1] * self.lat_coeff
            lon_base = x[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        
        if self.borehole:
            if self.fake_borehole:
                # Use third value for the depth of the top station and 0 for the borehole depth
                depth_base = x[:, :, 2:3] * self.depth_coeff * 0
                depth2_base = x[:, :, 2:3] * self.depth_coeff
            else:
                depth2_base = x[:, :, 3:4] * self.depth_coeff
            output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                                torch.sin(lon_base), torch.cos(lon_base),
                                torch.sin(depth_base), torch.cos(depth_base),
                                torch.sin(depth2_base), torch.cos(depth2_base)], dim=-1)
        else:
            output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                                torch.sin(lon_base), torch.cos(lon_base),
                                torch.sin(depth_base), torch.cos(depth_base)], dim=-1)
        
        output = torch.gather(output, -1, self.mask.unsqueeze(0).unsqueeze(0).expand_as(output))
        
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            output *= mask  # Zero out all masked elements
        
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, infinity=1e6,
                 att_masking=False,
                 kernel_initializer=None,
                 att_dropout=0.0,
                 input_shape,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.infinity = infinity
        self.att_masking = att_masking
        self.kernel_initializer = kernel_initializer if kernel_initializer is not None else nn.init.uniform_
        self.att_dropout = att_dropout
        self.build(input_shape)

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                self.kernel_initializer(m.weight, a=-1.2, b=1.2)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # 应用初始化到所有子模块
        self.apply(init_func)

    def build(self, input_shape):
        d_model = input_shape[-1]  # Embedding dim
        self.stations = input_shape[1]
        assert d_model % self.n_heads == 0
        d_key = d_model // self.n_heads  # = d_query = d_val
        self.d_key = d_key
        
        self.WQ = nn.Linear(d_model, d_key * self.n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_key * self.n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_key * self.n_heads, bias=False)
        self.WO = nn.Linear(d_key * self.n_heads, d_model, bias=False)
        
        self._init_weights()

    def forward(self, x, mask=None):
        d_key = self.d_key
        n_heads = self.n_heads
        
        if self.att_masking:
            att_mask = x[1]
            x = x[0]
            if mask is not None:
                mask = mask[0]
        else:
            att_mask = None
        
        batch_size, stations, _ = x.size()
        
        q = self.WQ(x)  # (batch, stations, key*n_heads)
        q = q.view(batch_size, stations, n_heads, d_key).transpose(1, 2)  # (batch, n_heads, stations, key)
        
        k = self.WK(x)  # (batch, stations, key*n_heads)
        k = k.view(batch_size, stations, n_heads, d_key).transpose(1, 2)  # (batch, n_heads, stations, key)
        
        v = self.WV(x)  # (batch, stations, key*n_heads)
        v = v.view(batch_size, stations, n_heads, d_key).transpose(1, 2)  # (batch, n_heads, stations, key)
        
        score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_key)  # (batch, n_heads, stations, stations)
        
        if mask is not None:
            inv_mask = (~mask).unsqueeze(1).unsqueeze(2).float()  # (batch, 1, 1, stations)
            score = score - inv_mask * self.infinity
        
        if att_mask is not None:
            inv_mask = (~att_mask).unsqueeze(1).unsqueeze(2).float()  # (batch, 1, 1, stations)
            score = score - inv_mask * self.infinity
        
        score = F.softmax(score, dim=-1)
        
        if self.att_dropout > 0:
            score = F.dropout(score, p=self.att_dropout, training=self.training)
        
        o = torch.matmul(score, v)  # (batch, n_heads, stations, key)
        o = o.transpose(1, 2).contiguous().view(batch_size, stations, n_heads * d_key)  # (batch, stations, n_heads*key)
        
        o = self.WO(o)  # (batch, stations, d_model)
        
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            o = o * mask
        
        return o

class PointwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, input_shape, kernel_initializer=None, bias_initializer=None, **kwargs):
        super(PointwiseFeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_initializer = kernel_initializer if kernel_initializer is not None else nn.init.xavier_uniform_
        self.bias_initializer = bias_initializer if bias_initializer is not None else nn.init.zeros_
        self.build(input_shape)

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                self.kernel_initializer(m.weight)
                if m.bias is not None:
                    self.bias_initializer(m.bias)
        # 应用初始化到所有子模块
        self.apply(init_func)

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.linear1 = nn.Linear(d_model, self.hidden_dim, bias=True)
        self.linear2 = nn.Linear(self.hidden_dim, d_model, bias=True)
        self._init_weights()

    def forward(self, x, mask=None):
        if self.linear1 is None or self.linear2 is None:
            self.build(x.size())
        
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            x *= mask  # Zero out all masked elements
        
        return x


class LayerNormalization(nn.Module):
    def __init__(self, input_shape, eps=1e-5, **kwargs):
        super(LayerNormalization, self).__init__()
        self.eps = eps

        d_model = input_shape[-1]
        self.beta = nn.Parameter(torch.Tensor(d_model))
        self.gamma = nn.Parameter(torch.Tensor(d_model))
	nn.init.zeros_(self.beta)
	nn.init.ones_(self.gamma)

    def forward(self, x, mask=None):
        
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        z = (x - mean) / std
        output = self.gamma * z + self.beta
        
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            output *= mask  # Zero out all masked elements
        
        return output

class AddEventToken(nn.Module):
    def __init__(self, input_shape, fixed=True, init_range=None, **kwargs):
        super().__init__()
        self.fixed = fixed
        self.init_range = init_range
        self.build(input_shape)

    def build(self, input_shape):
        self.emb = nn.Parameter(torch.Tensor(input_shape[-1]))
        if not self.fixed:
            if self.init_range is None:
                nn.init.ones_(self.emb)
            else:
                nn.init.uniform_(self.emb,-self.init_range, self.init_range)

    def forward(self, x, mask=None):
        pad = torch.ones_like(x[:, :1, :])
        if self.emb is not None:
            pad *= self.emb
        
        x = torch.cat([pad, x], dim=1)
        
        if mask is not None:
            mask_pad = torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask_pad, mask], dim=1)
        
        return x, mask


class AddConstantToMixture(nn.Module):
    def __init__(self, **kwargs):
        super(AddConstantToMixture, self).__init__()

    def forward(self, x, mask=None):
        mix, const = x
        const = const.unsqueeze(-1)
        
        alpha = mix[..., 0]
        mu = mix[..., 1] + const
        sigma = mix[..., 2]
        
        output = torch.stack([alpha, mu, sigma], dim=-1)
        
        mask = self.compute_mask(mask)
        if mask is not None:
            mask = mask.to(dtype=output.dtype)
            while mask.ndim < output.ndim:
                mask = mask.unsqueeze(-1)
            output *= mask
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, mask=None):
        if mask is None:
            return mask
        else:
            mask1 = mask[0]
            mask2 = mask[1]
            if mask1 is None:
                return mask2
            elif mask2 is None:
                return mask1
            else:
                return torch.logical_and(mask1, mask2)

class Masking_nd(nn.Module):
    def __init__(self, mask_value=0., axis=-1, nodim=False, **kwargs):
        super(Masking_nd, self).__init__()
        self.supports_masking = True
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim

    def compute_mask(self, inputs, mask=None):
        if self.nodim:
            output_mask = inputs != self.mask_value
        else:
            output_mask = (inputs != self.mask_value).any(dim=self.axis)
        return output_mask

    def forward(self, inputs):
        boolean_mask = (inputs != self.mask_value).any(dim=self.axis, keepdim=True)
        return inputs * boolean_mask.to(inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

class GetMask(nn.Module):
    def __init__(self, **kwargs):
        super(GetMask, self).__init__()

    def forward(self, x, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def compute_mask(self, inputs, mask=None):
        return mask

class StripMask(nn.Module):
    def __init__(self, **kwargs):
        super(StripMask, self).__init__()

    def forward(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def mixture_density_loss(y_true, y_pred, eps=1e-6, d=1, mean=True, print_shapes=True):
    if print_shapes:
        print(f'True: {y_true.shape}')
        print(f'Pred: {y_pred.shape}')
    
    alpha = y_pred[:, :, 0]
    density = torch.ones_like(y_pred[:, :, 0])  # Create an array of ones of correct size
    
    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = torch.max(sigma, torch.tensor(eps))
        density *= 1 / (np.sqrt(2 * np.pi) * sigma) * torch.exp(-(y_true[:, :, j] - mu) ** 2 / (2 * sigma ** 2))
    
    density *= alpha
    density = torch.sum(density, dim=1)
    density += eps
    loss = -torch.log(density)

    if mean:
        return torch.mean(loss)
    else:
        return loss

def time_distributed_loss(y_true, y_pred, loss_func, norm=1, mean=True, summation=True, kwloss={}):
    seq_length = y_pred.shape[1]
    y_true = y_true.view(-1, (y_pred.shape[-1] - 1) // 2, 1)
    y_pred = y_pred.view(-1, y_pred.shape[-2], y_pred.shape[-1])
    loss = loss_func(y_true, y_pred, **kwloss)
    loss = loss.view(-1, seq_length)

    if mean:
        return torch.mean(loss)

    loss /= norm
    if summation:
        loss = torch.sum(loss)

    return loss

class GlobalMaxPooling1DMasked(nn.Module):
    def __init__(self, **kwargs):
        super(GlobalMaxPooling1DMasked, self).__init__()

    def forward(self, x, mask=None):
        pseudo_infty = 1000.0
        
        if mask is not None:
            # Expand mask to match the dimensions of x
            mask = mask.unsqueeze(-1).expand_as(x)
            # Set masked positions to a very low value
            x = x - mask * pseudo_infty
        return x.max(dim=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, inputs, mask=None):
        return None

class NormalizedScaleEmbedding(nn.Module):
    def __init__(self, input_shape, downsample=5, activation='relu', mlp_dims=(500, 500, 500)):
        super(NormalizedScaleEmbedding, self).__init__()
        self.downsample = downsample
        self.activation = getattr(F, activation)
        layers = []
        in_channels = input_shape[-1]
        for out_channels in mlp_dims:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(getattr(F, activation)())
            layers.append(nn.Dropout(0.1))  # Optional dropout
            in_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, channels, length)
        x = F.avg_pool1d(x, kernel_size=self.downsample)
        return self.mlp(x).permute(0, 2, 1)  # Change back to (batch, length, channels)

class MLP(nn.Module):
    def __init__(self, input_shape, output_dims, activation='relu'):
        super(MLP, self).__init__()
        self.activation = getattr(F, activation)
        layers = []
        in_features = input_shape[0]
        for out_features in output_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(getattr(F, activation)())
            layers.append(nn.Dropout(0.1))  # Optional dropout
            in_features = out_features
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MixtureOutput(nn.Module):
    def __init__(self, input_shape, mixture_components, d=1, bias_mu=0, bias_sigma=1, activation='linear'):
        super(MixtureOutput, self).__init__()
        self.d = d
        self.bias_mu = bias_mu
        self.bias_sigma = bias_sigma
        self.activation = getattr(F, activation)
        self.alpha = nn.Linear(input_shape[0], mixture_components)
        self.mu = nn.Linear(input_shape[0], mixture_components * d)
        self.sigma = nn.Linear(input_shape[0], mixture_components * d)

    def forward(self, x):
        alpha = F.softmax(self.alpha(x), dim=-1)
        mu = self.mu(x).view(x.size(0), -1, self.d) + self.bias_mu
        sigma = F.softplus(self.sigma(x)).view(x.size(0), -1, self.d) + self.bias_sigma
        return torch.cat([alpha.unsqueeze(-1), mu, sigma], dim=-1)

class Masking_nd(nn.Module):
    def __init__(self, mask_value=0., axis=-1, nodim=False, **kwargs):
        super(Masking_nd, self).__init__()
        self.supports_masking = True
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim

    def compute_mask(self, inputs, mask=None):
        if self.nodim:
            output_mask = inputs != self.mask_value
        else:
            output_mask = (inputs != self.mask_value).any(dim=self.axis)
        return output_mask

    def forward(self, inputs):
        boolean_mask = (inputs != self.mask_value).any(dim=self.axis, keepdim=True)
        return inputs * boolean_mask.to(inputs.dtype)

class AddEventToken(nn.Module):
    def __init__(self, fixed=True, init_range=None, **kwargs):
        super(AddEventToken, self).__init__()
        self.fixed = fixed
        self.init_range = init_range
        self.emb = None

    def reset_parameters(self):
        if self.emb is not None:
            if self.init_range is None:
                nn.init.ones_(self.emb)
            else:
                nn.init.uniform_(self.emb, -self.init_range, self.init_range)

    def build(self, input_shape):
        if not self.fixed:
            if self.init_range is None:
                initializer = nn.init.ones_
            else:
                initializer = nn.init.uniform_
            self.emb = nn.Parameter(torch.Tensor(input_shape[-1]))
            self.reset_parameters()
        
        # Register parameter with the module
        if self.emb is not None:
            self.register_parameter('emb', self.emb)

    def forward(self, x):
        if self.emb is None:
            self.build(x.size())
        
        pad = torch.ones_like(x[:, :1, :])
        if self.emb is not None:
            pad *= self.emb
        
        x = torch.cat([pad, x], dim=1)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, wavelengths=((0.01, 10), (0.01, 10), (0.01, 10)), emb_dim=500, borehole=False, rotation=None, rotation_anchor=None):
        super(PositionEmbedding, self).__init__()
        self.wavelengths = wavelengths
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

    def forward(self, x):
        batch_size, seq_len, num_coords = x.size()
        pos_encoding = torch.zeros(batch_size, seq_len, self.emb_dim, device=x.device)
        position = x
        angle_rads = self.get_angles(position, self.wavelengths, self.emb_dim // (num_coords * 2))
        pos_encoding = torch.cat((torch.sin(angle_rads[:, :, 0::2]),
                                  torch.cos(angle_rads[:, :, 1::2])), dim=-1)
        return pos_encoding

    def get_angles(self, pos, wavelength, d_model):
        angle_rates = 1 / torch.pow(wavelength, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        angle_rads = pos.unsqueeze(-1) * angle_rates
        return angle_rads

class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-5, **kwargs):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.beta = None
        self.gamma = None

    def reset_parameters(self):
        if self.beta is not None:
            nn.init.zeros_(self.beta)
        if self.gamma is not None:
            nn.init.ones_(self.gamma)

    def build(self, input_shape):
        d_model = input_shape[-1]
        
        self.beta = nn.Parameter(torch.Tensor(d_model))
        self.gamma = nn.Parameter(torch.Tensor(d_model))
        
        self.reset_parameters()
        
        # Register parameters with the module
        self.register_parameter('beta', self.beta)
        self.register_parameter('gamma', self.gamma)

    def forward(self, x):
        if self.beta is None or self.gamma is None:
            self.build(x.size())
        
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        z = (x - mean) / std
        output = self.gamma * z + self.beta
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, att_dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.att_dropout = att_dropout
        
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.out_linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(att_dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
        output = self.out_linear(output)
        
        return output, attention_weights

class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, ffn_hidden_dim, att_dropout=0.0, hidden_dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.attn = MultiHeadAttention(emb_dim, n_heads, att_dropout)
        self.ffn = FeedForwardNetwork(emb_dim, ffn_hidden_dim, hidden_dropout)
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dropout2 = nn.Dropout(hidden_dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, max_stations, emb_dim, att_masking=False, layers=6, hidden_dropout=0.0, mad_params={}, ffn_params={}):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(emb_dim, mad_params['n_heads'], ffn_params['hidden_dim'], 
                             mad_params['att_dropout'], hidden_dropout) for _ in range(layers)
        ])
        self.att_masking = att_masking

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GlobalMaxPooling1DMasked(nn.Module):
    def __init__(self, **kwargs):
        super(GlobalMaxPooling1DMasked, self).__init__()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, mask=None):
        pseudo_infty = 1000.0
        
        if mask is not None:
            # Expand mask to match the dimensions of x
            mask = mask.unsqueeze(-1).expand_as(x)
            # Set masked positions to a very low value
            x = x - mask * pseudo_infty
        
        # Apply global max pooling
        output, _ = self.global_max_pool(x.permute(0, 2, 1))
        return output.squeeze(-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, inputs, mask=None):
        return None

class AddConstantToMixture(nn.Module):
    def __init__(self, **kwargs):
        super(AddConstantToMixture, self).__init__()

    def forward(self, x, mask=None):
        mix, const = x
        const = const.unsqueeze(-1)
        
        alpha = mix[..., 0]
        mu = mix[..., 1] + const
        sigma = mix[..., 2]
        
        output = torch.stack([alpha, mu, sigma], dim=-1)
        
        mask = self.compute_mask(mask)
        if mask is not None:
            mask = mask.to(dtype=output.dtype)
            while mask.ndim < output.ndim:
                mask = mask.unsqueeze(-1)
            output *= mask
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, mask=None):
        if mask is None:
            return mask
        else:
            mask1 = mask[0]
            mask2 = mask[1]
            if mask1 is None:
                return mask2
            elif mask2 is None:
                return mask1
            else:
                return torch.logical_and(mask1, mask2)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def mixture_density_loss(y_true, y_pred, eps=1e-6, d=1, mean=True, print_shapes=True):
    if print_shapes:
        print(f'True: {y_true.shape}')
        print(f'Pred: {y_pred.shape}')
    
    alpha = y_pred[:, :, 0]
    density = torch.ones_like(y_pred[:, :, 0])  # Create an array of ones of correct size
    
    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = torch.max(sigma, torch.tensor(eps))
        density *= 1 / (np.sqrt(2 * np.pi) * sigma) * torch.exp(-(y_true[:, :, j] - mu) ** 2 / (2 * sigma ** 2))
    
    density *= alpha
    density = torch.sum(density, dim=1)
    density += eps
    loss = -torch.log(density)

    if mean:
        return torch.mean(loss)
    else:
        return loss

def time_distributed_loss(y_true, y_pred, loss_func, norm=1, mean=True, summation=True, kwloss={}):
    seq_length = y_pred.shape[1]
    y_true = y_true.view(-1, (y_pred.shape[-1] - 1) // 2, 1)
    y_pred = y_pred.view(-1, y_pred.shape[-2], y_pred.shape[-1])
    loss = loss_func(y_true, y_pred, **kwloss)
    loss = loss.view(-1, seq_length)

    if mean:
        return torch.mean(loss)

    loss /= norm
    if summation:
        loss = torch.sum(loss)

    return loss

class MultiStaModel(nn.module):
    def __init__(self,
                 max_stations,
                 waveform_model_dims=(500, 500, 500),
                 output_mlp_dims=(150, 100, 50, 30, 10),
                 output_location_dims=(150, 100, 50, 50, 50),
                 wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                 mad_params={"n_heads": 10,
                             "att_dropout": 0.0,
                             "initializer_range": 0.02
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
		 **kwargs,
                ):
        super().__init__()
        emb_dim = waveform_model_dims[-1]
        # Event model
        if n_pga_targets:
            att_masking = True
            mad_params['att_masking'] = True
        else:
            att_masking = False
            mad_params['att_masking'] = False
    
        if not no_event_token:
            transformer_max_stations = max_stations + 1 + n_pga_targets
        else:
            transformer_max_stations = max_stations + n_pga_targets
    
        if not skip_transformer:
            self.transformer = Transformer(max_stations=transformer_max_stations, emb_dim=emb_dim, att_masking=att_masking,
                                      layers=transformer_layers, hidden_dropout=hidden_dropout, mad_params=mad_params,
                                      ffn_params=ffn_params)
    
        self.mlp_mag = MLP((emb_dim,), output_mlp_dims, activation=activation)
        self.output_model_mag = MixtureOutput((output_mlp_dims[-1],), magnitude_mixture, bias_mu=bias_mag_mu,
                                     bias_sigma=bias_mag_sigma)
    
        self.mlp_loc = MLP((emb_dim,), output_location_dims, activation=activation)
        self.output_model_loc = MixtureOutput((output_location_dims[-1],), location_mixture, d=3, bias_mu=bias_loc_mu,
                                         bias_sigma=bias_loc_sigma, activation='linear')
    
        self.mlp_pga = MLP((emb_dim,), output_mlp_dims, activation=activation)
        self.output_model_pga = MixtureOutput((output_mlp_dims[-1],), pga_mixture, activation='linear', bias_mu=-5, bias_sigma=1)
    
        if not alternative_coords_embedding:
            self.sta_pos_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole,
                                           rotation=rotation, rotation_anchor=rotation_anchor)
    
        if not  no_event_token:
            self.add_event = AddEventToken(fixed=False, init_range=event_token_init_range)
    
        if n_pga_targets:
            self.pga_pos_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole,
                                        rotation=rotation, rotation_anchor=rotation_anchor)
        if dataset_bias:
            self.dat_emb = nn.Embedding(n_datasets, 1)

        self.masking_wav = Masking_nd(0,(2,3))
        self.masking_coor = Masking_nd(0)
        self.masking_pga = Masking_nd(0)
        self.layer_norm = LayerNormalization()
        self.max_pool = GlobalMaxPooling1DMasked()
        self.add_const = AddConstantToMixture()

        self.alternative_coords_embedding = alternative_coords_embedding
        self.skip_transformer = skip_transformer
        self.no_event_token = no_event_token
        self.n_pga_targets = n_pga_targets
        self.dataset_bias = dataset_bias
        self.n_datasets = n_datasets

    def forward(self, x):
        waveform_inp,metadata_inp,pga_targets_inp,dataset = x
        waveforms_masked = self.masking_wav(waveform_inp)
        coords_masked = self.masking_coor(metadata_inp)

        n_stations = waveforms_masked.shape[1]
        waveforms_emb = rearrange(waveforms_masked, "b sta l c -> (b sta) () l c")
        waveforms_emb = waveform_model(waveforms_emb)
        waveforms_emb = rearrange(waveforms_emb, "(b sta) d -> b sta d", sta=n_stations)
        waveforms_emb = self.layer_norm(waveforms_emb)

        if not self.alternative_coords_embedding:
            coords_emb = self.sta_pos_emb(coords_masked)
            emb = waveforms_emb + coords_emb
        else:
            emb = torch.cat([waveforms_emb, coords_masked], dim=-1)

        if not (self.skip_transformer or self.no_event_token):
            emb = self.add_event(emb)

        if self.n_pga_targets:
            pga_targets_masked = self.mask_pag(pga_targets_inp)
            pga_emb = self.pga_pos_emb(pga_targets_masked)
            att_mask = torch.cat([torch.ones_like(emb[:, :, 0], dtype=torch.bool),
                                  torch.zeros_like(pga_emb[:, :, 0], dtype=torch.bool)], dim=1)
            emb = torch.cat([emb, pga_emb], dim=1)
            emb = transformer(emb, att_mask)
        else:
            if self.skip_transformer:
                mlp_input_length = emb_dim
                if self.alternative_coords_embedding:
                    mlp_input_length += metadata_shape[0]
                emb = self.max_pool(emb)
            else:
                emb = transformer(emb)

        if not self.no_event_token:
            if self.skip_transformer:
                event_emb = emb
            else:
                event_emb = emb[:, 0, :]  # Select event embedding

            mag_embedding = self.mlp_mag(event_emb)
            out = self.output_model_mag(mag_embedding)

            loc_embedding = self.mlp_loc(event_emb)
            out_loc = self.output_model_loc(loc_embedding)

        if self.n_pga_targets:
            pga_emb = emb[:, -self.n_pga_targets:, :]  # Select embeddings for pga
            n_pga = pga_emb.shape[1]
            pga_emb = rearrange(pga_emb, "b pga d -> (b pga) d")
            output_pga = self.output_model_pga(pga_emb)
            output_pga = rearrange(output_pga, "(b pga) d -> b pga d",pga=n_pga)

        if self.dataset_bias:
            assert self.n_datasets is not None
            dataset_bias_term = self.dat_emb(dataset)
            dataset_bias_term = dataset_bias_term.reshape(dataset_bias_term.shape[0],-1).squeeze(-1)
            out = self.add_const([out, dataset_bias_term])

        outputs = []
        if not self.no_event_token:
            outputs += [out, out_loc]

        if self.n_pga_targets:
            outputs += [output_pga]

        return outputs

def build_transformer_model(max_stations,
                            waveform_model_dims=(500, 500, 500),
                            output_mlp_dims=(150, 100, 50, 30, 10),
                            output_location_dims=(150, 100, 50, 50, 50),
                            wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                            mad_params={"n_heads": 10,
                                        "att_dropout": 0.0,
                                        "initializer_range": 0.02
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
                            **kwargs):
    if kwargs:
        print(f'Warning: Unused model parameters: {", ".join(kwargs.keys())}')

    emb_dim = waveform_model_dims[-1]
    mad_params = mad_params.copy()  # Avoid modifying the input dicts
    ffn_params = ffn_params.copy()

    if 'initializer_range' in mad_params:
        r = mad_params['initializer_range']
        del mad_params['initializer_range']

    # Single station model
    if borehole:
        input_shape = (trace_length, 6)
        metadata_shape = (4,)
    else:
        input_shape = (trace_length, 3)
        metadata_shape = (3,)
    
    waveform_model = NormalizedScaleEmbedding(input_shape, downsample=downsample, activation=activation,
                                              mlp_dims=waveform_model_dims)
    mlp_mag_single_station = MLP((waveform_model.output_shape[1],), output_mlp_dims, activation=activation)
    output_model_single_station = MixtureOutput((output_mlp_dims[-1],), 5, name='magnitude',
                                                bias_mu=bias_mag_mu, bias_sigma=bias_mag_sigma)

    single_station_model = nn.Sequential(waveform_model, mlp_mag_single_station, output_model_single_station)

    full_model = MultiStaModel(max_stations=max_stations,
                            waveform_model_dims=waveform_model_dims,
                            output_mlp_dims=output_mlp_dims,
                            output_location_dims=output_location_dims,
                            wavelength=wavelength,
                            mad_params=mad_params,
                            ffn_params=ffn_params,
                            transformer_layers=transformer_layers,
                            hidden_dropout=hidden_dropout,
                            activation=activation,
                            n_pga_targets=n_pga_targets,
                            location_mixture=location_mixture,
                            pga_mixture=pga_mixture,
                            magnitude_mixture=magnitude_mixture,
                            borehole=borehole,
                            bias_mag_mu=bias_mag_mu,
                            bias_mag_sigma=bias_mag_sigma,
                            bias_loc_mu=bias_loc_mu,
                            bias_loc_sigma=bias_loc_sigma,
                            event_token_init_range=event_token_init_range,
                            dataset_bias=dataset_bias,
                            n_datasets=n_datasets,
                            no_event_token=no_event_token,
                            trace_length=trace_length,
                            downsample=downsample,
                            rotation=rotation,
                            rotation_anchor=rotation_anchor,
                            skip_transformer=skip_transformer,
                            alternative_coords_embedding=alternative_coords_embedding,
                            **kwargs)
    return single_station_model, full_model

class EnsembleEvaluateModel:
    def __init__(self, config, max_ensemble_size=None, loss_limit=None):
        self.config = config
        self.ensemble = config.get('ensemble', 1)
        true_ensemble_size = self.ensemble
        if max_ensemble_size is not None:
            self.ensemble = min(self.ensemble, max_ensemble_size)
        self.models = []
        for ens_id in range(self.ensemble):
            model_params = config['model_params'].copy()
            if config['training_params'].get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                model_params['rotation'] = np.pi / 4 * ens_id / (true_ensemble_size - 1)
            _, model = build_transformer_model(**model_params)
            self.models.append(model)
        self.loss_limit = loss_limit

    def predict_generator(self, generator, **kwargs):
        preds = [self._predict_batchwise(model, generator) for model in self.models]
        return self.merge_preds(preds)

    def predict(self, inputs):
        preds = [model(inputs) for model in self.models]
        return self.merge_preds(preds)

    @staticmethod
    def merge_preds(preds):
        merged_preds = []

        if isinstance(preds[0], list):
            iter_range = range(len(preds[0]))
        else:
            iter_range = [-1]

        for i in iter_range:  # Iterate over mag, loc, pga, ...
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
            return merged_preds[0]
        else:
            return merged_preds

    def load_weights(self, weights_path):
        tmp_models = self.models
        self.models = []
        removed_models = 0
        for ens_id, model in enumerate(tmp_models):
            if self.loss_limit is not None:
                hist_path = os.path.join(weights_path, f'{ens_id}', 'hist.pkl')
                with open(hist_path, 'rb') as f:
                    hist = pickle.load(f)
                if np.min(hist['val_loss']) > self.loss_limit:
                    removed_models += 1
                    continue

            tmp_weights_path = os.path.join(weights_path, f'{ens_id}')
            weight_files = sorted([x for x in os.listdir(tmp_weights_path) if x[:5] == 'event'])
            if not weight_files:
                print(f"No valid weight files found for ensemble {ens_id}")
                continue
            
            weight_file = os.path.join(tmp_weights_path, weight_files[-1])
            state_dict = torch.load(weight_file)
            model.load_state_dict(state_dict)
            self.models.append(model)

        if removed_models > 0:
            print(f'Removed {removed_models} models not fulfilling loss limit')

    def _predict_batchwise(self, model, generator):
        all_preds = []
        with torch.no_grad():
            for batch_inputs in generator:
                if isinstance(batch_inputs, tuple):
                    inputs = batch_inputs[:-1]  # Assuming last element is the target
                else:
                    inputs = (batch_inputs,)
                
                outputs = model(*inputs)
                if isinstance(outputs, tuple):
                    outputs = list(outputs)
                else:
                    outputs = [outputs]
                
                all_preds.append(outputs)
        
        # Merge predictions from each batch
        final_preds = []
        for i in range(len(all_preds[0])):
            combined_output = torch.cat([pred[i] for pred in all_preds], dim=0)
            final_preds.append(combined_output)
        
        return final_preds
