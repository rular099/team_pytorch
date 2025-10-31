import torch
import torch.nn as nn

from .conv_module import LayerNorm


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_features, kernel_size, qkv_bias=False):
        super().__init__()
        self.dw = nn.Conv1d(in_features, in_features, kernel_size, 1, kernel_size//2, bias=qkv_bias, groups=in_features)
        self.norm = LayerNorm(in_features)
        self.pw = nn.Conv1d(in_features, in_features, 1, bias=qkv_bias)
        
    def forward(self, x):
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw(x)
        return x


class PS_Attention(nn.Module):
    def __init__(self, dim, kernel_size=3, pale_size=7, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % 2 == 0, 'dim must be even'
        assert (dim//2) % num_heads == 0, 'dim should be divisible by num_heads'
        self.pale_size = pale_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.ModuleList([Depthwise_Separable_Conv(dim, kernel_size, qkv_bias),
                                  Depthwise_Separable_Conv(dim, kernel_size, qkv_bias),
                                  Depthwise_Separable_Conv(dim, kernel_size, qkv_bias)])
        self.proj = nn.Conv1d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
    
    def forward(self, q, k, v):
        bs, len_q, len_kv = q.shape[0], q.shape[2], k.shape[2]
        assert len_q%self.pale_size == 0 & len_kv%self.pale_size == 0, 'seq length should be divisible by pale_size'
        
        q,k,v = self.qkv[0](q),self.qkv[1](k),self.qkv[2](v)
        q,k,v = self.seq2pale(q,len_q,2),self.seq2pale(k,len_kv,2),self.seq2pale(v,len_kv,2)
        
        x = self.axis_attention(q, k, v, bs)
        x = x.permute(0, 2, 1, 3).contiguous().flatten(2) # --> (B, C, L)
        x = self.pale2seq(x, len_q, 2)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def axis_attention(self, q, k, v, B):
        B_,C,L_ = q.shape

        q = q.reshape(B_, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2).contiguous()
        k = k.reshape(B_, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(B_, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2).contiguous()
        
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).contiguous().reshape([B_,C,-1]).reshape([B,-1,C,L_]) # --> (B, pale_size, C, L//pale_size)
        return x
    
    def seq2pale(self, x, length, axis):
        x_idx = [torch.tensor(range(i, length, self.pale_size), device=x.device) for i in range(self.pale_size)]
        x = torch.cat([x.index_select(axis, idx).unsqueeze(1) for idx in x_idx], dim = 1) # --> (B, pale_size, C, L//pale_size)
        x = x.reshape([-1]+list(x.shape[2:])) # --> (B*pale_size, C, L//pale_size)
        return x
    
    def pale2seq(self, x, length, axis):
        idx = torch.cat([torch.tensor(range(i, length, self.pale_size), device=x.device) for i in range(self.pale_size)])
        idx = torch.argsort(idx)
        x = x.index_select(axis, idx) # --> (B, C, L)
        return x