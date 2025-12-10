import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from diting.finetuneing.utils import create_model
import diting.finetuneing.utils.help_builder as help_builder
import diting.LSD.models.backbone_ablation as backbone_ablation

from mup import set_base_shapes

# Helper Functions
def gelu(x):
    """GELU activation function."""
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class MLP(nn.Module):
    def __init__(self, input_shape, dims=(100, 50), activation=F.relu, last_activation=None):
        super().__init__()
        if last_activation is None:
            last_activation = activation

        layers = [nn.Linear(input_shape[-1], dims[0]),
                  nn.ReLU()]  # Assuming input_shape is a tuple (batch_size, feature_dim)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2: # Add activation except for the last layer
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MixtureOutput(nn.Module):
    def __init__(self, input_shape, n=5, d=1, activation=F.relu, eps=1e-4, bias_mu=1.8, bias_sigma=0.2, name=None):
        super().__init__()
        self.n = n
        self.d = d
        self.activation = activation
        self.eps = eps

        self.alpha = nn.Linear(input_shape[-1], n)
        self.mu = nn.Linear(input_shape[-1], n * d)
        nn.init.constant_(self.mu.bias, bias_mu) # Bias Init
        self.sigma = nn.Linear(input_shape[-1], n * d)
        nn.init.constant_(self.sigma.bias, bias_sigma)

    def forward(self, x):
        alpha = torch.softmax(self.alpha(x), dim=-1).unsqueeze(-1) # (batch, n, 1)
        mu = self.activation(self.mu(x)).reshape(-1, self.n, self.d)  # (batch, n, d)
        sigma = F.relu(self.sigma(x)).reshape(-1, self.n, self.d) + self.eps  # (batch, n, d)
        out = torch.cat([alpha, mu, sigma], dim=-1)  # (batch, n, 1+d+d)
        return out



class NormalizedScaleEmbedding(nn.Module):
    def __init__(self, input_shape, activation=F.relu, downsample=1, mlp_dims=(500, 300, 200, 150), eps=1e-8):
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

    def forward(self, x, att_mask=None):
        # The inputs are already handled by the calling function.
        for block in self.blocks:
            x = block(x, att_mask)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, emb_dim, hidden_dim, att_dropout=0.0, initializer_range=0.02, eps=1e-5):
        super().__init__()
        self.attention = MultiHeadSelfAttention(n_heads=n_heads, emb_dim=emb_dim, att_dropout=att_dropout, initializer_range=initializer_range)
#        self.ffn = PointwiseFeedForward(hidden_dim=hidden_dim)
#        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
                      nn.Linear(emb_dim, hidden_dim),
                      nn.ReLU(),
                      nn.Linear(hidden_dim, emb_dim),
        )
#        self.norm1 = LayerNormalization(eps=eps)
#        self.norm2 = LayerNormalization(eps=eps)
        self.norm1 =nn.LayerNorm(emb_dim) 
        self.norm2 =nn.LayerNorm(emb_dim) 
        self.dropout1 = nn.Dropout(0.0) #Fixed dropout
        self.dropout2 = nn.Dropout(0.0) #Fixed dropout

    def forward(self, x, att_mask=None, src_key_padding_mask=None):
        modified_x, _ = self.attention(x, attn_mask=att_mask)#, key_padding_mask=src_key_padding_mask)
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

    def forward(self, x, attn_mask=None):
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
    def __init__(self, mask_value=0., axis=-1, nodim=False):
        super().__init__()
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim

    def forward(self, inputs, mask=None):
        if self.nodim:
            boolean_mask = (inputs != self.mask_value)
        else:
            boolean_mask = torch.any((inputs != self.mask_value), dim=self.axis, keepdim=True)

        return inputs * boolean_mask.float()  # Ensure boolean_mask is float

    def compute_mask(self, inputs, mask=None):  # Add this for mask propagation
        if self.nodim:
            output_mask = (inputs != self.mask_value)
        else:
            output_mask = torch.any((inputs != self.mask_value), dim=self.axis)
        return output_mask


class StripMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x



class GlobalMaxPooling1DMasked(nn.Module):
    def forward(self, x, mask=None):
        pseudo_infty = 1000.
        if mask is None:
            # Ensure that the mask is not the maximum value any more
            mask = torch.ones_like(x, dtype=torch.bool).to(x.device) # Changed from None to ones
        mask = mask.unsqueeze(-1)
        x = x - mask.float() * pseudo_infty
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

class SingleStationModel(nn.Module):
    def __init__(self, waveform_model, mlp_mag_single_station, output_model_single_station):
        super().__init__()
        self.waveform_model = waveform_model
        self.mlp_mag_single_station = mlp_mag_single_station
        self.output_model_single_station = output_model_single_station

    def forward(self, waveform_inp_single_station):
        emb = self.waveform_model(waveform_inp_single_station)
        emb = self.mlp_mag_single_station(emb)
        out = self.output_model_single_station(emb)
        return out

class FullModel(nn.Module):
    def __init__(self, waveform_model, position_embedding, transformer, mlp_mag, output_model_mag, mlp_loc,
                 output_model_loc, mlp_pga, output_model_pga, skip_transformer, alternative_coords_embedding,
                 metadata_shape, emb_dim, no_event_token, add_event_token, n_pga_targets, dataset_bias,
                 AddConstantToMixture, n_datasets):
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
        self.add_constant_to_mixture = AddConstantToMixture
        self.n_datasets = n_datasets

        if self.n_pga_targets > 0:
            self.att_masking = True
        else:
            self.att_masking = False

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

    def forward(self, waveform_inp, metadata_inp, pga_targets_inp=None, dataset=None, att_mask = None):
        waveforms_masked = self.Masking_nd_0_23(waveform_inp)
        coords_masked = self.Masking_nd_0_2(metadata_inp)

        waveforms_emb = torch.stack([self.waveform_model(waveforms_masked[:, i, :, :]) for i in range(waveforms_masked.shape[1])] , dim=1)
        waveforms_emb = self.layernorm(waveforms_emb)

        if not self.alternative_coords_embedding:
            coords_emb = self.position_embedding(coords_masked)
            emb = waveforms_emb + coords_emb
        else:
            emb = torch.cat([waveforms_emb, coords_masked], dim=-1)

        if not (self.skip_transformer or self.no_event_token):
            emb = self.add_event_token(emb)

        if self.n_pga_targets:
            pga_targets_masked = self.Masking_nd_0_2(pga_targets_inp)
            pga_emb = self.position_embedding(pga_targets_masked)
            emb = torch.cat([emb, pga_emb], dim=1)

            if att_mask is None:
                # Create an attention mask based on the shape of emb
#                att_mask = torch.cat([torch.ones_like(emb[:, :emb.shape[1] - pga_emb.shape[1], 0], dtype=torch.bool),
#                                       torch.zeros_like(emb[:, emb.shape[1] - pga_emb.shape[1]:, 0], dtype=torch.bool)], dim=1)
                att_mask = self.Masking_nd_0_23.compute_mask(waveform_inp)
                att_mask = torch.cat([att_mask,
                                      torch.zeros_like(emb[:, emb.shape[1] - pga_emb.shape[1]:, 0], dtype=torch.bool)], dim=1)
                if not (self.skip_transformer or self.no_event_token):
                    att_mask = torch.cat([torch.ones_like(emb[:, :1, 0], dtype=torch.bool), att_mask], dim=1)
            emb = self.transformer(emb.float(), att_mask) # Modified line
        else:
            if self.skip_transformer:
                emb = torch.stack([self.mlp_layer(emb[:, i, :]) for i in range(emb.shape[1])], dim=1)
                emb = self.maxpool(emb, mask=coords_masked[:,:,0] != 0)
            else:
                emb = self.transformer(emb.float())

        outputs = []
        if not self.no_event_token:
            if self.skip_transformer:
                event_emb = emb
            else:
                event_emb = emb[:, 0, :]  # Select event embedding

            mag_embedding = self.mlp_mag(event_emb)
            out_mag = self.output_model_mag(mag_embedding)

            loc_embedding = self.mlp_loc(event_emb)
            out_loc = self.output_model_loc(loc_embedding)

            outputs.append(out_mag)
            outputs.append(out_loc)

        if self.n_pga_targets:
            pga_emb = emb[:, -self.n_pga_targets:, :]  # Select embeddings for pga
            pga_emb = torch.stack([self.mlp_pga(pga_emb[:, i, :]) for i in range(pga_emb.shape[1])], dim=1)
            output_pga = torch.stack([self.output_model_pga(pga_emb[:, i, :]) for i in range(pga_emb.shape[1])], dim=1)
            outputs.append(output_pga)

        if self.dataset_bias:
            assert self.n_datasets is not None
            dataset_bias_term = self.dataset_embedding(dataset).squeeze(-1)
            out = self.add_constant_to_mixture()(out, dataset_bias_term)

        return outputs

def get_diting_model(args):
    # Model (todo) enable mup init
    base_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.base_width, depth=24) # args.encoder_size
    base_model = help_builder.create_model(
        model_name=args.model_name,
        downstream_tasks=args.downstream_task,
        in_samples=args.in_samples,
        encoder_size=base_encoder_size_dict,
        eval_type=args.eval_type,
        pool_type=args.pool_type,
        args=args,
    )
    target_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.target_width, depth=24)
    model = help_builder.create_model(
        model_name=args.model_name,
        downstream_tasks=args.downstream_task,
        in_samples=args.in_samples,
        encoder_size=target_encoder_size_dict,
        eval_type=args.eval_type,
        pool_type=args.pool_type,
        args=args,
    )
    print(model)
    ### muP: set base_shapes
    set_base_shapes(model, base_model) # do_assert=False

    if not os.path.exists(args.resume) and args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained,weights_only=False, map_location="cpu")

            # rename moco pre-trained keys
            print('############# ckpt keys', checkpoint.keys())
            if args.pretrain_method == "mae":
                if args.pretrained.endswith('.pt'):
                    key = 'module'
                else:
                    key = 'state_dict'
                state_dict = checkpoint[key]
                # print(f'############# {key} has ', state_dict.keys())
                for k in list(state_dict.keys()):
                    # from deepspeed
                    if k.startswith("base_encoder."):
                        # remove prefix
                        state_dict['0.'+k[len("base_encoder.") :]] = state_dict[k] # for MAE:0. is mapped to encoder(in func:help_builder.create_model)
                    elif k.startswith("module.base_encoder."):
                        # remove prefix
                        state_dict['0.'+k[len("module.base_encoder.") :]] = state_dict[k] # for MAE:0. is mapped to encoder(in func:help_builder.create_model)
                    if args.dpk_head == 'vit_adapter_decoder_new':
                        if k.startswith("base_decoder."):
                            state_dict['1.decoder.'+k[len("base_decoder.") :]] = state_dict[k]
                        elif k.startswith("module.base_decoder."): 
                            state_dict['1.decoder.'+k[len("module.base_decoder.") :]] = state_dict[k]
                    elif 'decoder' in args.dpk_head or args.dpk_head == 'vit_adapter_TaskSeparatedUPerHead':
                        if k.startswith("base_decoder."):
                            state_dict['0.decoder.'+k[len("base_decoder.") :]] = state_dict[k]
                        elif k.startswith("module.base_decoder."): 
                            state_dict['0.decoder.'+k[len("module.base_decoder.") :]] = state_dict[k]
                    
                    del state_dict[k]
            elif args.pretrain_method == "lp":
                if args.pretrained.endswith('.pt'):
                    key = 'module'
                else:
                    key = 'model_dict'
                state_dict = checkpoint[key]
                del checkpoint["optimizer_dict"]
            else:
                raise NotImplementedError(f"Unsupported pretrain method:'{args.pretrain_method}'")
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            
            if args.pool_type == 'cls':
                assert msg.missing_keys == ['2.fc.weight', '2.fc.bias'],"load pretrain model fail!"
            elif args.pool_type == 'avg' or args.pool_type == 'attentive':
                missing_keys_except_attentive = [k for k in msg.missing_keys if not k.startswith('1.')]
                assert missing_keys_except_attentive == ['3.fc.weight', '3.fc.bias'],"load pretrain model fail!"
            
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            assert os.path.isfile(args.pretrained),"no checkpoint found at '{}'".format(args.pretrained)

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
                            activation=F.relu,
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
    dt2team = MLP((diting_args.out_channels,), waveform_model_dims[-1:], activation=activation)
    waveform_model.add_module('dt2team', dt2team)
    mlp_mag_single_station = MLP((waveform_model_dims[-1],), output_mlp_dims, activation=activation) #Modified line
    output_model_single_station = MixtureOutput((output_mlp_dims[-1],), n=5, name='magnitude',
                                                bias_mu=bias_mag_mu, bias_sigma=bias_mag_sigma)

    single_station_model = SingleStationModel(waveform_model, mlp_mag_single_station, output_model_single_station)

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

    mlp_mag = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model_mag = MixtureOutput((output_mlp_dims[-1],), n=magnitude_mixture, bias_mu=bias_mag_mu,
                                 bias_sigma=bias_mag_sigma)

    mlp_loc = MLP((emb_dim,), output_location_dims, activation=activation)
    output_model_loc = MixtureOutput((output_location_dims[-1],), n=location_mixture, d=3, bias_mu=bias_loc_mu,
                                     bias_sigma=bias_loc_sigma, activation=F.relu)

    mlp_pga = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model_pga = MixtureOutput((output_mlp_dims[-1],), n=pga_mixture, activation=F.relu, bias_mu=-5, bias_sigma=1)

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

    full_model = FullModel(waveform_model, position_embedding, transformer, mlp_mag, output_model_mag, mlp_loc,
                             output_model_loc, mlp_pga, output_model_pga, skip_transformer, alternative_coords_embedding,
                             metadata_shape, emb_dim, no_event_token, add_event_token, n_pga_targets, dataset_bias,
                             add_constant_to_mixture, n_datasets)
    return single_station_model, full_model


class EnsembleEvaluateModel:
    def __init__(self, config, max_ensemble_size=None, loss_limit=None, device='cpu'):
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
            single_station_model, full_model = build_transformer_model(**model_params)

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

         return self.merge_preds(preds)

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
            if self.loss_limit is not None:
                hist_path = os.path.join(weights_path, f'{ens_id}', 'hist.pkl')
                with open(hist_path, 'rb') as f:
                    hist = pickle.load(f)
                if np.min(hist['val_loss']) > self.loss_limit:
                    removed_models += 1
                    continue

            tmp_weights_path = os.path.join(weights_path, f'{ens_id}')
            weight_file = sorted([x for x in os.listdir(tmp_weights_path) if x[:5] == 'event'])[-1]
            weight_file = os.path.join(tmp_weights_path, weight_file)

            # Load weights to CPU first, then move to the device
            state_dict = torch.load(weight_file, map_location='cpu')
            model.load_state_dict(state_dict)

            self.models.append(model.to(self.device)) # Then move model to device after loading weights

        if removed_models > 0:
            print(f'Removed {removed_models} models not fulfilling loss limit')
