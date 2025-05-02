import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    waveform_inp_single_station = torch.randn(1, *input_shape)  # Dummy input for demonstration
    emb = waveform_model(waveform_inp_single_station)
    emb = mlp_mag_single_station(emb)
    out = output_model_single_station(emb)

    single_station_model = nn.Sequential(waveform_model, mlp_mag_single_station, output_model_single_station)

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
        transformer = Transformer(max_stations=transformer_max_stations, emb_dim=emb_dim, att_masking=att_masking,
                                  layers=transformer_layers, hidden_dropout=hidden_dropout, mad_params=mad_params,
                                  ffn_params=ffn_params)

    mlp_mag = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model = MixtureOutput((output_mlp_dims[-1],), magnitude_mixture, bias_mu=bias_mag_mu,
                                 bias_sigma=bias_mag_sigma)

    mlp_loc = MLP((emb_dim,), output_location_dims, activation=activation)
    output_model_loc = MixtureOutput((output_location_dims[-1],), location_mixture, d=3, bias_mu=bias_loc_mu,
                                     bias_sigma=bias_loc_sigma, activation='linear')

    mlp_pga = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model_pga = MixtureOutput((output_mlp_dims[-1],), pga_mixture, activation='linear', bias_mu=-5, bias_sigma=1)

    waveform_inp = torch.randn(1, max_stations, *input_shape)  # Dummy input for demonstration
    metadata_inp = torch.randn(1, max_stations, *metadata_shape)  # Dummy input for demonstration

    waveforms_masked = Masking_nd(0, (2, 3))(waveform_inp)
    coords_masked = Masking_nd(0)(metadata_inp)

    waveforms_emb = nn.utils.rnn.PackedSequence(waveforms_masked)  # Simulating TimeDistributed
    waveforms_emb = waveform_model(waveforms_emb.data)
    waveforms_emb = LayerNormalization()(waveforms_emb)

    if not alternative_coords_embedding:
        coords_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole,
                                       rotation=rotation, rotation_anchor=rotation_anchor)(coords_masked)

        emb = waveforms_emb + coords_emb
    else:
        emb = torch.cat([waveforms_emb, coords_masked], dim=-1)

    if not (skip_transformer or no_event_token):
        emb = AddEventToken(fixed=False, init_range=event_token_init_range)(emb)

    if n_pga_targets:
        pga_targets_inp = torch.randn(1, n_pga_targets, 3)  # Dummy input for demonstration
        pga_targets_masked = Masking_nd(0)(pga_targets_inp)
        pga_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole,
                                    rotation=rotation, rotation_anchor=rotation_anchor)(pga_targets_masked)
        att_mask = torch.cat([torch.ones_like(emb[:, :, 0], dtype=torch.bool),
                              torch.zeros_like(pga_emb[:, :, 0], dtype=torch.bool)], dim=1)
        emb = torch.cat([emb, pga_emb], dim=1)
        emb = transformer(emb, att_mask)
    else:
        if skip_transformer:
            mlp_input_length = emb_dim
            if alternative_coords_embedding:
                mlp_input_length += metadata_shape[0]

            emb = nn.utils.rnn.PackedSequence(MLP((mlp_input_length,), [emb_dim, emb_dim], activation=activation)(emb))
            emb = GlobalMaxPooling1DMasked()(emb.data)
        else:
            emb = transformer(emb)

    if not no_event_token:
        if skip_transformer:
            event_emb = emb
        else:
            event_emb = emb[:, 0, :]  # Select event embedding

        mag_embedding = mlp_mag(event_emb)
        out = output_model(mag_embedding)

        loc_embedding = mlp_loc(event_emb)
        out_loc = output_model_loc(loc_embedding)

    if n_pga_targets:
        pga_emb = emb[:, -n_pga_targets:, :]  # Select embeddings for pga
        pga_emb = nn.utils.rnn.PackedSequence(mlp_pga(pga_emb))
        output_pga = nn.utils.rnn.PackedSequence(output_model_pga(pga_emb.data))

    if dataset_bias:
        assert n_datasets is not None
        dataset = torch.randint(0, n_datasets, (1, 1))  # Dummy input for demonstration
        dataset_embedding = nn.Embedding(n_datasets, 1)
        dataset_bias_term = dataset_embedding(dataset).squeeze(-1)
        out = AddConstantToMixture()([out, dataset_bias_term])

    # Name output
    if not no_event_token:
        out = out
        out_loc = out_loc

    inputs = [waveform_inp, metadata_inp]
    outputs = []
    if not no_event_token:
        outputs += [out, out_loc]

    if n_pga_targets:
        inputs += [pga_targets_inp, att_mask]
        outputs += [output_pga]

    if dataset_bias:
        inputs += [dataset]

    full_model = nn.ModuleList(outputs)

    return single_station_model, full_model

# Example usage:
single_station_model, full_model = build_transformer_model(max_stations=10)

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

# Example usage:
if __name__ == "__main__":
    config = {
        'ensemble': 2,
        'model_params': {
            'max_stations': 10,
            'waveform_model_dims': (500, 500, 500),
            'output_mlp_dims': (150, 100, 50, 30, 10),
            'output_location_dims': (150, 100, 50, 50, 50),
            'wavelength': ((0.01, 10), (0.01, 10), (0.01, 10)),
            'mad_params': {"n_heads": 10, "att_dropout": 0.0, "initializer_range": 0.02},
            'ffn_params': {'hidden_dim': 1000},
            'transformer_layers': 6,
            'hidden_dropout': 0.0,
            'activation': 'relu',
            'n_pga_targets': 0,
            'location_mixture': 5,
            'pga_mixture': 5,
            'magnitude_mixture': 5,
            'borehole': False,
            'bias_mag_mu': 1.8,
            'bias_mag_sigma': 0.2,
            'bias_loc_mu': 0,
            'bias_loc_sigma': 1,
            'event_token_init_range': None,
            'dataset_bias': False,
            'n_datasets': None,
            'no_event_token': False,
            'trace_length': 3000,
            'downsample': 5,
            'rotation': None,
            'rotation_anchor': None,
            'skip_transformer': False,
            'alternative_coords_embedding': False,
        },
        'training_params': {
            'ensemble_rotation': False,
        }
    }

    ensemble_model = EnsembleEvaluateModel(config, max_ensemble_size=2, loss_limit=None)

