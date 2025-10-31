# Cell
import math
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = nn.LayerNorm
    print("Please 'pip install apex'")

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

from mup import MuReadout
from timm.layers import DropPath

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)
    
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
    elif pe == 'sincos': 
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    #     # For BC we register cos and sin cached
    #     self.max_seq_len_cached = max_position_embeddings
    #     t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
    #     t = t / self.scaling_factor
    #     freqs = torch.outer(t, self.inv_freq)
    #     # Different from paper, but it uses a different permutation in order to obtain the same calculation
    #     emb = torch.cat((freqs, freqs), dim=-1)
    #     self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
    #     self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    # @property
    # def sin_cached(self):
    #     # logger.warning_once(
    #     #     "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
    #     #     "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
    #     # )
    #     return self._sin_cached

    # @property
    # def cos_cached(self):
    #     # logger.warning_once(
    #     #     "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
    #     #     "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
    #     # )
    #     return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # if seq_len is not None:
            # logger.warning_once("The `seq_len` argument is deprecated and unused. It will be removed in v4.39.")

        # x: [bs, num_attention_heads, seq_len, head_size]
        # inv_freq_expanded: [bs , self.dim //2 , 1]
        # position_ids: [bs, seq_len] , position_ids_expanded: [bs, 1, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # @: [bs, self.dim // 2 , seq_len]
            # freqs: [bs, seq_len, self.dim // 2]
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float().to(x.device)).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.[bs,head,seq_len,embed_dim]
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding. [bs,seq_len,embed_dim]
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim).to(q.device)
    sin = sin.unsqueeze(unsqueeze_dim).to(q.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed # [bs,head,seq_len,embed_dim],[bs,head,seq_len,embed_dim]

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_mult=1., attn_dropout=0., res_attention=False, lsa=False, xattn=False, dec_module=False):
        super().__init__()
        self.attn_dropout = attn_dropout if xattn else nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        if dec_module:
            self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        else:
            self.scale = nn.Parameter(torch.tensor(head_dim ** -1 * math.sqrt(attn_mult)), requires_grad=lsa)  # Attention Logits Scaling
        self.lsa = lsa
        self.xattn = xattn

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
        if self.xattn:
            output = xops.memory_efficient_attention(
                q, k.transpose(2, 3), v,
                p=self.attn_dropout,
                scale=self.scale,
                attn_bias=xops.LowerTriangularMask() if attn_mask is not None else None,
            ) # output: [bs x n_heads x max_q_len x d_v]
            if self.res_attention:
                return output, None, None
            else:
                return output, None
        else:
            # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
            attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

            # Add pre-softmax attention scores from the previous layer (optional)
            if prev is not None: attn_scores = attn_scores + prev

            # Attention mask (optional)
            # TODO:check attn_mask is useless??
            if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
                if attn_mask.dtype == torch.bool:
                    attn_scores.masked_fill_(attn_mask, -np.inf)
                else:
                    attn_scores += attn_mask

            # Key padding mask (optional)
            if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
                attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

            # normalize the attention weights
            attn_weights = F.softmax(attn_scores, dim=-1).to(q.dtype)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
            attn_weights = self.attn_dropout(attn_weights)

            # compute the new values given the attention weights
            output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    elif activation.lower() == "quick_gelu": return QuickGELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

def get_norm_fn(norm, d_model=None):
    if callable(norm): return norm()
    if norm.lower() == 'layernorm':
        norm_fn = nn.LayerNorm(d_model)
    elif norm.lower() == 'fusedln':
        norm_fn = FusedLayerNorm(d_model)
    elif norm.lower() == 'rmsnorm':
        norm_fn = LlamaRMSNorm(d_model)
    elif norm.lower() == 'batchnorm':
        norm_fn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    else:
        raise NotImplementedError("norm function wrong")
    return norm_fn

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers, d_model, n_heads,
                 d_ff, norm, attn_dropout, dropout, store_attn,
                 pre_norm, bias, xattn, config):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.config = config

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_p = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_len,
            bias=bias,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            d_model, n_heads, d_ff=d_ff, norm=norm, 
            attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, n_layers=n_layers, store_attn=store_attn, 
            bias=bias, xattn=xattn, config=config, dec_module=False
        )
        
        self.encoder_post_norm = pre_norm
        if self.encoder_post_norm:
            self.encoder_norm = get_norm_fn(norm, d_model=d_model)

        self.initialize_weights()

    def initialize_weights(self):
        # (todo) Conv1D的参数如何初始化-是否可以视为visual token embedding initialize patch_embed like nn.Linear (instead of nn.Conv1d)
        w = self.W_p.weight.data
        w.normal_(mean=0.0, std=self.config.init_std) # the shape of W_p is torch.Size([128, 3, 50])
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def set_mask(self, mask_ratio, mask_way):
        self.mask_ratio = mask_ratio
        self.mask_way = mask_way

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

        return x_masked, mask, ids_restore,ids_keep # ids_keep for rope
    
    def block_masking(self,x):
        '''x:[bs x patch_num x encoder_dim]'''

        mask_ratio = self.mask_ratio
        block_size = 10
        B, N, D = x.size()
        mask_num = int(N * mask_ratio)
        keep_num = N - mask_num
        num_mask_block = mask_num // block_size
        
        def generate_unique_random(target_shape,max_num):
            if len(target_shape) == 1:
                B = target_shape[0]
                target = torch.randint(0, max_num, (B,1))
            else:
                B = target_shape[0]
                mask_num = target_shape[-1]
                target = torch.zeros(target_shape)
                for i in range(B):
                    target[i] = torch.randperm(max_num)[:mask_num] # among a sample, no overlap index
            return target
        
        # get mask block idx
        offset = generate_unique_random((B,), block_size)
        offset = offset.repeat(1,num_mask_block) # [B,1]
        k = generate_unique_random((B, num_mask_block),  N // block_size) # [B,num_mask_block]
        # k.shape: torch.Size([16, 10]) offset.shape: torch.Size([10, 1])
        block_idx = k * block_size + offset

        # mask的范围是block_idx到block_idx + block_size - 1，若越界，则取开头
        # 0 is keep, 1 is remove
        mask = torch.zeros(B, N).to(x.device)
        for i in range(num_mask_block):
            start_index1 = block_idx[:, i]
            end_index1 = torch.min(torch.tensor(N).repeat(B), block_idx[:, i] + block_size)
            mask1 = torch.arange(mask.size(1))[None, :] >= start_index1[:, None]
            mask1 &= torch.arange(mask.size(1))[None, :] < end_index1[:, None]

            assert (mask[mask1] == 1).sum() == 0
            mask[mask1] = 1
            
            start_index2 = torch.tensor(0).repeat(B)
            end_index2 = torch.max(torch.tensor(0).repeat(B),block_idx[:, i] + block_size - N)
            # print(start_index1,end_index1,start_index2,end_index2)
            mask2 = torch.arange(mask.size(1))[None, :] >= start_index2[:, None]
            mask2 &= torch.arange(mask.size(1))[None, :] < end_index2[:, None]
            assert (mask[mask2] == 1).sum() == 0
            mask[mask2] = 1

        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:,:keep_num]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked,mask,ids_restore,ids_keep

    def grid_masking(self,x):
        '''x:[bs x patch_num x encoder_dim]'''
        B, N, D = x.shape

        mask_ratio = self.mask_ratio
        block_size = 8
        num_patch = N // block_size
        keep_num = int(N * (1 - mask_ratio))
        
        # 0 is keep, 1 is remove
        mask = torch.zeros(B*num_patch, block_size).to(x.device)
        mask[:,:int(mask_ratio*block_size)] = 1
        mask = mask.reshape(B, -1)

        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:,:keep_num]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked,mask,ids_restore,ids_keep
    
    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x nvars x patch_len]
        """
        # Input encoding (Embedding output Scaling)
        u = self.W_p(x) * self.config.input_mult                                 # u: [bs x d_model x num_patch]
        u = u.transpose(1, 2)                                                    # u: [bs x num_patch x d_model]

        if self.mask_ratio:
            assert self.mask_way in ['random','block','grid']
            # TODO from MAE u: [bs x patch_num x encoder_dim]
            if self.mask_way == 'random':
                u, mask, ids_restore,ids_keep = self.random_masking(u)
            elif self.mask_way == 'block':
                u, mask, ids_restore, ids_keep  = self.block_masking(u)
            elif self.mask_way == 'grid':
                u, mask, ids_restore, ids_keep  = self.grid_masking(u)
        else: # TODO
            ids_keep = torch.arange(u.shape[1]).unsqueeze(0).repeat(u.shape[0],1).to(u.device)
            mask = None
            ids_restore = None

        # Encoder
        position_ids = ids_keep # [bs, keep_num]
        z = self.encoder(u, position_ids)                                        # z: [bs x (num_patch) x d_model]

        if self.encoder_post_norm:
            z = self.encoder_norm(z)

        if self.mask_ratio:
            return z, mask, ids_restore # TODO from MAE: z: [bs x (kept_num) x encoder_dim], mask=ids_restore: [bs x patch_num]
        # self.config存在ft_with_decoder这个参数
        elif self.config.pool_type == 'decoder':
            return z
        else:
            return z.transpose(1, 2) # for finetune and BYOL
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, norm, attn_dropout, dropout,
                 n_layers, pre_norm, store_attn, bias, xattn, config, dec_module):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, n_layers)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout,drop_path=dpr[i],
                pre_norm=pre_norm, store_attn=store_attn,
                bias=bias, xattn=xattn, config=config, n_layers=n_layers, dec_module=dec_module
            ) for i in range(n_layers)
        ])

    def forward(self, src:Tensor, position_ids=None, hierarchical=False):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        feat_list = []
        for mod in self.layers:
            output = mod(output, position_ids=position_ids)
            feat_list.append(output)
        
        if hierarchical:
            return feat_list
        else:
            return output


class MultiheadAttention_ROPE(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False, xattn=False, attn_mult=1., dec_module=False): # xattn=xattn
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        # rope
        self.hidden_size = d_model
        self.num_heads = n_heads
        self.num_key_value_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_position_embeddings = 400 # max_patch_num
        self.rope_theta = 10000.0
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)

        self.rotary_emb = LlamaRotaryEmbedding(
            d_k,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(
            d_model, n_heads, attn_mult=attn_mult, attn_dropout=attn_dropout,res_attention=res_attention, lsa=lsa, xattn=xattn, dec_module=dec_module
        )

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,position_ids=None):

        bsz,q_len,_ = Q.shape # TODO check
        if K is None: K = Q
        if V is None: V = Q

        # rope
        query_states = self.q_proj(Q)
        key_states = self.k_proj(K)
        value_states = self.v_proj(V)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids) # value_states:[bs,heads,q_len,embed_dim] -> cos:[bs,seq_len,embed_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                query_states, key_states.transpose(2, 3), value_states, prev=prev,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                query_states, key_states.transpose(2, 3), value_states, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bsz, -1,
                                                          self.num_heads * self.head_dim)  # output: [bs x q_len x n_heads * d_v]
        output = self.o_proj(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, store_attn,
                norm, attn_dropout, dropout,drop_path, bias, pre_norm, xattn, config, n_layers, dec_module):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.self_attn = MultiheadAttention_ROPE(
            d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, qkv_bias=bias, xattn=xattn, attn_mult=config.attn_mult, dec_module=dec_module,
        )
        # self.dropout_attn = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = get_norm_fn(norm,d_model)
        
        # ffn
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias) # default bias is `false`
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.act_fn = nn.SiLU()

        # Add & Norm
        # self.dropout_ffn = nn.Dropout(dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_ffn = get_norm_fn(norm, d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.config = config
        self.n_layers = n_layers
        self.dec_module = dec_module

    def forward(self, hidden_states:Tensor, position_ids=None):
        """
        x: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        # fix bug
        residual = hidden_states.clone()
        if self.pre_norm:
            hidden_states = self.norm_attn(hidden_states)
        
        ## Multi-Head attention
        assert position_ids is not None,"rope must have position_ids"
        hidden_states, attn = self.self_attn(hidden_states, hidden_states, hidden_states, position_ids=position_ids)

        if self.store_attn:
            self.attn = attn
        ## Add & Norm (Residual Connection Scaling)
        hidden_states = residual + self.drop_path1(hidden_states)
        
        if not self.pre_norm:
            hidden_states = self.norm_attn(hidden_states)

        # Feed-forward sublayer
        # fix bug
        residual = hidden_states.clone()
        if self.pre_norm:
            hidden_states = self.norm_ffn(hidden_states)
        ## Position-wise Feed-Forward
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        ## Add & Norm (Residual Connection Scaling)
        hidden_states = residual + self.drop_path2(hidden_states)
        
        if not self.pre_norm:
            hidden_states = self.norm_ffn(hidden_states)

        return hidden_states


class PatchTST_base(nn.Module):
    """
    Output dimension: 
         [bs x d_model x num_patch]
    """
    def __init__(self, num_patch:int, n_layers:int,d_model,n_heads,d_ff:int,
                 c_in:int=3, patch_len:int=50,
                 norm:str='LayerNorm', attn_dropout:float=0., dropout:float=0., 
                 pre_norm:bool=True, store_attn:bool=False,
                 head_type = "all", bias=True, 
                 xattn=False, config=None, **kwargs):

        super().__init__()

        assert head_type in ['avg pooling', 'feature','all'], 'head type should be either pretrain, prediction, or regression'
        self.head_type = head_type
        self.n_layers = n_layers
        # Backbone
        self.backbone = PatchTSTEncoder(
            c_in, num_patch=num_patch, patch_len=patch_len, 
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
            d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout, 
            pre_norm=pre_norm, store_attn=store_attn,
            norm=norm, bias=bias, xattn=xattn, config=config, **kwargs
        )
        
        self.config = config
        self.apply(self._init_weights)
        
    def _init_weights(self, module, readout_zero_init=False, query_zero_init=False):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            ### muP: swap constant std normal init with normal_ from `mup.init`.
            ### Because `_init_weights` is called in `__init__`, before `infshape` is set,
            ### we need to manually call `self.apply(self._init_weights)` after calling
            ### `set_base_shape(model, base)`
            if isinstance(module, MuReadout) and readout_zero_init:
                module.weight.data.zero_()
            else:
                if hasattr(module.weight, 'infshape'):
                    # mup.init.normal_(module.weight, std=self.args.init_std)
                    mup.init.xavier_uniform_(module.weight)
                else:
                    # module.weight.data.normal_(mean=0.0, std=self.args.init_std)
                    nn.init.xavier_uniform_(module.weight)
            ### End muP
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias'):
                module.bias.data.zero_()
        ### muP
        if isinstance(module, MultiheadAttention_ROPE):
            if query_zero_init:
                module.q_proj.weight.data[:] = 0
    
    def get_num_layers(self):
        return self.n_layers

    def forward(self, z):                             
        """
        z: [bs x c x seq_len]
        """   
        res = self.backbone(z) # z: [bs x c x seq_len]
            
        if self.head_type == 'feature':
            res = res # attentive pooling need [bs,patch_num,dim] [bs,dim,patch_num] 
        elif self.head_type == 'all': # MAE train
            res = res # x: [bs x (kept_num + 1) x encoder_dim], mask=ids_restore: [bs x patch_num]                                                     
        
        return res


class PatchTSTDecoder_base(nn.Module):
    def __init__(self,
                # encoder
                encoder_dim,
                # decoder
                decoder_dim,
                decoder_depth,
                decoder_num_heads,
                mlp_ratio,
                norm,
                patch_size=50,
                c_in=3,
                attn_dropout=0,
                dropout=0.,
                store_attn=False,
                pre_norm=True,
                bias=True,
                xattn=False,
                config=None,
                ):
        super(PatchTSTDecoder_base, self).__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.patch_size = patch_size
        self.bias = bias
        self.config = config
        # MAE decoder specifics
        self.decoder_embed = MuReadout(encoder_dim, decoder_dim, bias=bias, output_mult=config.output_mult)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder = TSTEncoder(decoder_dim, decoder_num_heads, d_ff=mlp_ratio*decoder_dim, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                pre_norm=pre_norm, n_layers=decoder_depth, store_attn=store_attn, bias=bias, xattn=xattn, config=config, dec_module=True)
        
        self.decoder_post_norm = pre_norm
        if self.decoder_post_norm:
            self.decoder_norm = get_norm_fn(norm,decoder_dim)
        
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * c_in, bias=bias) # decoder to patch
        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=self.config.init_std)
            
    def forward(self, x, ids_restore, hierarchical=False):
        '''
        input:
            x: [bs x kept_num x encoder_dim]
        output:
            x: [bs x patch_num x nvars * patch_len]
        '''
        # embed tokens (similar to LM head scaling)
        x = self.decoder_embed(x) # x: [bs x kept_num x decoder_dim]

        if ids_restore:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1) # mask_tokens: [bs x drop_num x decoder_dim]
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle

        # apply Transformer blocks
        position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1) # [bs, patch_num]
        x = self.decoder(x, position_ids, hierarchical)
        
        if hierarchical:
            return x
        
        if self.decoder_post_norm:
            x = self.decoder_norm(x)
        
        # if not self.config.pool_type == 'decoder':
        if hasattr(self, 'decoder_pred'):
            # project back to input space
            x = self.decoder_pred(x)

        return x


if __name__ == "__main__":
    pass