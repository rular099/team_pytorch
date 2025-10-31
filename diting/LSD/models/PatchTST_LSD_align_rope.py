"""
ref: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/src/models/patchTST.py
"""


__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from einops import repeat
from timm.models.vision_transformer import Block

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        # TODO
        # self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        # self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        # self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    #     self.init_layer()
        
        
    # def init_layer(self):
    #     torch.manual_seed(42)
    #     torch.nn.init.xavier_uniform_(self.W_Q.weight.data)
    #     torch.nn.init.xavier_uniform_(self.W_K.weight.data)
    #     torch.nn.init.xavier_uniform_(self.W_V.weight.data)
    #     torch.nn.init.constant_(self.W_Q.bias.data, 0.0)
    #     torch.nn.init.constant_(self.W_K.bias.data, 0.0)
    #     torch.nn.init.constant_(self.W_V.bias.data, 0.0)
        
    #     # torch.nn.init.xavier_uniform_(self.qkv.weight.data[:512])
    #     # torch.nn.init.xavier_uniform_(self.qkv.weight.data[512:1024])
    #     # torch.nn.init.xavier_uniform_(self.qkv.weight.data[1024:])
    #     # torch.nn.init.constant_(self.qkv.bias.data, 0.0)
        
    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # TODO
        # Linear (+ split in multiple heads)
        # q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
        #                                                                  2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        # k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
        #                                                                1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]
        
        qkv = self.qkv(Q).reshape(bs, -1, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = qkv.unbind(0)
        k_s = k_s.transpose(-2, -1)
        
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high
        # self.low, self.high = ranges
    def forward(self, x):
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm2d(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')
            
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x d_model x num_patch]
    """
    def __init__(self, c_in:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "cls", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False,mask_ratio=False,ff_norm=False, **kwargs):

        super().__init__()

        assert head_type in ['cls', 'avg pooling', 'feature','all'], 'head type should be either pretrain, prediction, or regression'
        self.head_type = head_type
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose,mask_ratio=mask_ratio,ff_norm=ff_norm, **kwargs)

        # # Head
        # self.n_vars = c_in
        # self.head_type = head_type
        #
        # if head_type == "pretrain":
        #     self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        # elif head_type == "prediction":
        #     self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        # elif head_type == "regression":
        #     self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        # elif head_type == "classification":
        #     self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)


    def forward(self, z):                             
        """
        z: tensor [bs x nvars * patch_len]
        """   
        res = self.backbone(z) # z: [bs x d_model x num_patch]
        if self.head_type == 'cls':
            res = res[:, :, 0]
        elif self.head_type == 'feature':
            res = res[:, :, 1:]
        elif self.head_type == 'all':
            res = res # x: [bs x (kept_num + 1) x encoder_dim], mask=ids_restore: [bs x patch_num]                                                     
        # z = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return res


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='LayerNrom', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=False, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False,mask_ratio=False,ff_norm=False,encoder_type='PatchTST',**kwargs):

        super().__init__()
        torch.manual_seed(42)
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        self.mask_ratio = mask_ratio  
        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_p = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_len,
            bias=True, # v2-5
        )

        # modify by lhl,from ori VIT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional encoding
        # TODO v2-6
        # self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)
        self.W_pos = nn.Parameter(torch.zeros(1, num_patch + 1, d_model), requires_grad=False)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder_type = encoder_type
        if encoder_type == 'MAE':
            self.encoder = nn.ModuleList([
            Block(d_model, n_heads, d_ff/d_model, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(n_layers)])
            self.norm = nn.LayerNorm(d_model,eps=1e-6)
            
        elif encoder_type == 'PatchTST':
            self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                        store_attn=store_attn,ff_norm=ff_norm)
            self.pre_norm = pre_norm
            if pre_norm:
                self.norm = nn.LayerNorm(d_model,eps=1e-6)
        else:
            raise NotImplementedError
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.manual_seed(42)
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        W_pos = positional_encoding('sincos', False, self.num_patch + 1, self.W_pos.shape[-1])
        self.W_pos.data.copy_(W_pos.float().unsqueeze(0))

        # v2-5
        w = self.W_p.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        torch.manual_seed(42)
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    # add by lhl 2024/3/7 , for MAE
    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        output:
            x_masked: [N x kept_num x encoder_dim]
            mask=ids_restore: [N x patch_num]
        """
        torch.manual_seed(42)
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x nvars x patch_len]
        """
        b = x.size(0)
        # Input encoding
        u = self.W_p(x)                                                          # u: [bs x d_model x num_patch]
        u = u.transpose(1, 2)                                                    # u: [bs x num_patch x d_model]
        # TODO v2-6
        # u = self.dropout(u + self.W_pos)
        u = u + self.W_pos[:, 1:, :]
        if self.mask_ratio:
            # TODO from MAE u: [bs x patch_num x encoder_dim]
            u, mask, ids_restore = self.random_masking(u)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)                  # cls_token: [bs x 1 x d_model]
        
        # TODO v2-6
        cls_tokens = cls_tokens + self.W_pos[:, :1, :]
        
        u = torch.cat([cls_tokens, u], dim=1)                                        # u: [bs x (num_patch + 1) x d_model]

        if self.encoder_type == 'MAE':
            for blk in self.encoder:
                u = blk(u)
            z = self.norm(u)
        else:
            # Encoder
            z = self.encoder(u)                                                      # z: [bs x (num_patch + 1) x d_model]
            
        if self.mask_ratio:
            if self.encoder_type == 'PatchTST':
                if self.pre_norm:
                    z = self.norm(z)
            return z, mask, ids_restore # TODO from MAE: z: [bs x (kept_num + 1) x encoder_dim], mask=ids_restore: [bs x patch_num]
        else:
            z = z.transpose(1, 2)                                                    # z: [bs x d_model x (num_patch + 1)]
            return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='LayerNrom', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,ff_norm=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn,ff_norm=ff_norm) for i in range(n_layers)])
        self.res_attention = res_attention

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        torch.manual_seed(42)
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: 
                output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: 
                output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='LayerNrom', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False,ff_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model,eps=1e-6)

        # Position-wise Feed-Forward
        # self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
        #                         get_activation_fn(activation),
        #                         nn.Dropout(dropout),
        #                         nn.Linear(d_ff, d_model, bias=bias))
        ff = [nn.Linear(d_model, d_ff, bias=bias),get_activation_fn(activation),nn.Dropout(dropout),]
        if ff_norm:
            ff += [nn.LayerNorm(d_ff,eps=1e-6)]
        ff += [nn.Linear(d_ff, d_model, bias=bias)]
        self.ff = nn.Sequential(*ff)

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model,eps=1e-6)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src, prev=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        # TODO v2-7
        src_ori = src.clone()
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src_ori + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        # TODO v2-7
        src_ori = src.clone()
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src_ori + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class PatchTSTDecoder(nn.Module):
    def __init__(self,
                # base
                num_patches=200,
                patch_size=50,
                # encoder
                encoder_dim=192, 
                # decoder
                decoder_dim=512, 
                decoder_depth=8, 
                decoder_num_heads=16,
                mlp_ratio=4,
                decoder_type = 'PatchTST',
                norm='LayerNorm',
                attn_dropout=0,
                dropout=0., # v2-4
                res_attention=False,
                store_attn=False,
                # act
                act='relu',
                # norm
                ff_norm=False,
                pre_norm=False,
                **kwargs
                ):
        super(PatchTSTDecoder, self).__init__()
        torch.manual_seed(42)
        self.num_patches = num_patches
        self.decoder_dim = decoder_dim
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # decoder pos
        # TODO v2-6
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.decoder_type = decoder_type
        if decoder_type == 'MAE':
            if norm == 'LayerNorm':
                norm_layer = nn.LayerNorm
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_dim,eps=1e-6)
        elif decoder_type == 'PatchTST':
            self.decoder = TSTEncoder(decoder_dim, decoder_num_heads, d_ff=mlp_ratio*decoder_dim, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=decoder_depth, 
                                    store_attn=store_attn,ff_norm=ff_norm)
            self.pre_norm = pre_norm
            if pre_norm:
                self.decoder_norm = nn.LayerNorm(decoder_dim,eps=1e-6)
        else:
            raise NotImplementedError()
            

        self.decoder_pred = nn.Linear(decoder_dim, patch_size * 3, bias=True) # decoder to patch
        
        self.initialize_weights()

    def initialize_weights(self):
        # TODO v2-6
        decoder_pos_embed = positional_encoding('sincos', False, self.num_patches + 1, self.decoder_pos_embed.shape[-1])
        # decoder_pos_embed = positional_encoding('sincos', False, self.num_patches, self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        torch.manual_seed(42)
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x, ids_restore):
        '''
        input:
            x: [bs x (kept_num + 1) x encoder_dim]
        output:
            x: [bs x patch_num x nvars * patch_len]
        '''
        # embed tokens
        x = self.decoder_embed(x) # x: [bs x (kept_num + 1) x decoder_dim]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # mask_tokens: [bs x drop_num x decoder_dim]
        # mask_tokens = torch.load('/public/home/lhl725/encoder/MAE-PatchTST/mask_tokens.pt',map_location=x.device)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        # TODO v2-6
        # x_ = x_ + self.decoder_pos_embed
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # TODO v2-6
        x = x + self.decoder_pos_embed # x: [bs x (1 + patch_num) x decoder_dim]
        

        # apply Transformer blocks
        if self.decoder_type == 'MAE':
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
        elif self.decoder_type == 'PatchTST':
            x = self.decoder(x)
            if self.pre_norm:
                x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


def PatchTST_samll(input_length=10000, head_type='cls',mask_ratio=0):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST(c_in=3, 
                    patch_len=pacth_len, 
                    stride=stride, 
                    num_patch=num_patch, 
                    n_layers=12, 
                    d_model=192, 
                    n_heads=6, 
                    d_ff=768, 
                    pe='sincos', 
                    # MAE
                    head_type=head_type,
                    mask_ratio = mask_ratio,
                    )

def PatchTSTEncoder_small(mask_ratio,
                          input_length=10000, 
                          head_type='all',
                          act = 'gelu',
                          pre_norm=False,
                          ff_norm=False):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST(c_in=3, 
                    patch_len=pacth_len, 
                    stride=stride, 
                    num_patch=num_patch, 
                    n_layers=12, 
                    d_model=192, 
                    n_heads=6, 
                    d_ff=768, 
                    pe='sincos', 
                    # MAE
                    head_type=head_type,
                    mask_ratio = mask_ratio,
                    # act
                    act=act,
                    # norm
                    pre_norm=pre_norm,
                    ff_norm=ff_norm
                )
    
def PatchTSTEncoder_base(mask_ratio,
                         input_length=10000, 
                         head_type='all',
                         act = 'gelu',
                         pre_norm=False,
                         ff_norm=False):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST(c_in=3, 
                    patch_len=pacth_len, 
                    stride=stride, 
                    num_patch=num_patch, 
                    n_layers=12, 
                    d_model=192, 
                    n_heads=12, 
                    d_ff=3072, 
                    pe='sincos', 
                    # MAE
                    head_type=head_type,
                    mask_ratio = mask_ratio,
                    # act
                    act=act,
                    # norm
                    pre_norm=pre_norm,
                    ff_norm=ff_norm
                    )
def PatchTSTEncoder_base_dmodel768(mask_ratio,
                         input_length=10000, 
                         head_type='all',
                         act = 'gelu',
                         pre_norm=False,
                         ff_norm=False):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST(c_in=3, 
                    patch_len=pacth_len, 
                    stride=stride, 
                    num_patch=num_patch, 
                    n_layers=12, 
                    d_model=768, 
                    n_heads=12, 
                    d_ff=3072, 
                    pe='sincos', 
                    # MAE
                    head_type=head_type,
                    mask_ratio = mask_ratio,
                    # act
                    act=act,
                    # norm
                    pre_norm=pre_norm,
                    ff_norm=ff_norm
                    )

def PatchTSTEncoder_base_vit(mask_ratio,
                         input_length=10000, 
                         head_type='all',
                         act = 'gelu',
                         pre_norm=True,
                         ff_norm=False,
                         learn_pe=False,
                         res_attention=False):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST(c_in=3, 
                    patch_len=pacth_len, 
                    stride=stride, 
                    num_patch=num_patch, 
                    n_layers=12, 
                    d_model=768, 
                    n_heads=12, 
                    d_ff=3072, 
                    pe='sincos', 
                    # MAE
                    head_type=head_type,
                    mask_ratio = mask_ratio,
                    # act
                    act=act,
                    # norm
                    pre_norm=pre_norm,
                    ff_norm=ff_norm,
                    learn_pe=learn_pe,
                    res_attention=res_attention
                    )

def PatchTSTDecoder_common_vit(encoder_dim,
                           input_length=10000,
                           pre_norm=True,
                           ff_norm=False,
                           act='gelu',
                           res_attention=False,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder(
                # base
                num_patch=num_patch,
                patch_size=pacth_len,
                # encoder
                encoder_dim=encoder_dim, 
                # decoder
                decoder_dim=512, 
                decoder_depth=8, 
                decoder_num_heads=16,
                mlp_ratio=4,
                # act
                act=act,
                # norm
                pre_norm=pre_norm,
                ff_norm=ff_norm,
                res_attention=res_attention,
            ) 


if __name__ == '__main__':
    from transformers import LlamaModel, LlamaConfig,LlamaTokenizerFast

    # Initializing a LLaMA llama-7b style configuration
    configuration = LlamaConfig()

    # Initializing a model from the llama-7b style configuration
    model = LlamaModel(configuration)

    # Accessing the model configuration
    configuration = model.config
    
    print(configuration)

    tokenizer = LlamaTokenizerFast()
    tokenizer.encode("Hello this is a test")
    
    generate_ids = model.generate(inputs.input_ids, max_length=30)