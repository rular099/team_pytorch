from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp

from .conv_module import LayerNorm
from .pale_transformer import PS_Attention


class ConvFFN(nn.Module):
    def __init__(self, in_features, ffn_convKS=3, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(ffn_convKS, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x, L):
        x = x.transpose(1, 2) # [bs x num_patch x d_model]
        x = self.fc1(x)
        x = self.dwconv(x, L)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class DWConv(nn.Module):
    def __init__(self, ffn_convKS=3, dim=768):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=ffn_convKS, stride=1, padding=ffn_convKS//2, bias=True, groups=dim)

    def forward(self, x, L):
        B, N, C = x.shape
        n = N // 7
        x1 = x[:, 0:4 * n, :].transpose(1, 2).view(B, C, L * 2).contiguous()
        x2 = x[:, 4 * n:6 * n, :].transpose(1, 2).view(B, C, L).contiguous()
        x3 = x[:, 6 * n:, :].transpose(1, 2).view(B, C, L // 2).contiguous()
        x1 = self.dwconv(x1).transpose(1, 2)
        x2 = self.dwconv(x2).transpose(1, 2)
        x3 = self.dwconv(x3).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    

class Extractor(nn.Module):
    def __init__(self, dim, pale_size=5, ffn_convKS=3, cpe_convKS=3,
                 num_heads=6, drop=0., drop_path=0., 
                 with_cp=False, with_cffn=True, cffn_ratio=0.25, 
                 norm_layer=LayerNorm):
        super().__init__()
        self.cpe_q = nn.Conv1d(dim, dim, cpe_convKS, 1, cpe_convKS//2, groups=dim)
        self.cpe_k = nn.Conv1d(dim, dim, cpe_convKS, 1, cpe_convKS//2, groups=dim)
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = PS_Attention(dim, kernel_size=ffn_convKS, pale_size=pale_size, num_heads=num_heads)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), ffn_convKS=ffn_convKS, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat, L):
        '''
        Input:
            query: [bs x num_patch' x d_model], feat: [bs x num_patch x d_model]
        Output:
            query: [bs x num_patch' x d_model]
        '''
        def _inner_forward(query, feat):
            q = query.transpose(1, 2) # [bs x d_model x num_patch]
            B, C, N = q.shape
            n = N // 7
            q1 = q[:, :, 0:4 * n].view(B, C, L * 2).contiguous()
            q2 = q[:, :, 4 * n:6 * n].view(B, C, L).contiguous()
            q3 = q[:, :, 6 * n:].view(B, C, L // 2).contiguous()
            q1 = self.cpe_q(q1)
            q2 = self.cpe_q(q2)
            q3 = self.cpe_q(q3)
            cpe_query = torch.cat([q1, q2, q3], dim=2)
            q = cpe_query + q
            residual = q.clone()

            feat = feat.transpose(1, 2) # [bs x d_model x num_patch]
            feat = self.cpe_k(feat) + feat

            q = self.query_norm(q)
            k = v = self.feat_norm(feat)
            attn = self.attn(q, k, v)
            q = residual + attn

            if self.with_cffn:
                q = q + self.drop_path(self.ffn(self.ffn_norm(q), L))
            return q.transpose(1, 2)

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, pale_size=5, cpe_convKS=3, ffn_convKS=3,
                 norm_layer=LayerNorm, init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.cpe_q = nn.Conv1d(dim, dim, cpe_convKS, 1, cpe_convKS//2, groups=dim)
        self.cpe_k = nn.Conv1d(dim, dim, cpe_convKS, 1, cpe_convKS//2, groups=dim)
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = PS_Attention(dim, kernel_size=ffn_convKS, pale_size=pale_size, num_heads=num_heads)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat, L):
        '''
        Input:
            query: [bs x num_patch x d_model], feat: [bs x num_patch' x d_model]
        Output:
            query: [bs x num_patch x d_model]
        '''
        def _inner_forward(query, feat):
            q = query.transpose(1, 2) # [bs x d_model x num_patch]
            residual = q.clone()
            q = self.cpe_q(q) + q

            feat = feat.transpose(1, 2) # [bs x d_model x num_patch]
            B, C, N = feat.shape
            n = N // 7
            f1 = feat[:, :, 0:4 * n].view(B, C, L * 2).contiguous()
            f2 = feat[:, :, 4 * n:6 * n].view(B, C, L).contiguous()
            f3 = feat[:, :, 6 * n:].view(B, C, L // 2).contiguous()
            f1 = self.cpe_k(f1)
            f2 = self.cpe_k(f2)
            f3 = self.cpe_k(f3)
            cpe_feat = torch.cat([f1, f2, f3], dim=2)
            feat = cpe_feat + feat

            q = self.query_norm(q)
            k = v = self.feat_norm(feat)
            attn = self.attn(q, k, v)
            query = residual.transpose(1, 2) + self.gamma * attn.transpose(1, 2)
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, drop_path=0., cffn_ratio=0.25, 
                 extra_extractor=False, with_cp=False, pale_size=5, cpe_convKS=3, ffn_convKS=3):
        super().__init__()

        self.injector = Injector(
            dim=dim, num_heads=num_heads, with_cp=with_cp, 
            pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
        )
        self.extractor = Extractor(
            dim=dim, num_heads=num_heads, drop_path=drop_path, 
            cffn_ratio=cffn_ratio, with_cp=with_cp, 
            pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, drop_path=drop_path, 
                          cffn_ratio=cffn_ratio, with_cp=with_cp,
                          pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, L, position_ids=None):
        x = self.injector(query=x, feat=c, L=L)
        for idx, blk in enumerate(blocks):
            x = blk(x, position_ids)
        c = self.extractor(query=c, feat=x, L=L)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=x, L=L)
        return x, c

class InteractionBlockV2(nn.Module):
    def __init__(self, enc_dim, dec_dim, num_heads=6, drop_path=0., cffn_ratio=0.25, 
                 extra_extractor=False, with_cp=False, pale_size=5, cpe_convKS=3, ffn_convKS=3):
        super().__init__()
        if enc_dim is None:
            self.injector = Injector(
                dim=dec_dim, num_heads=num_heads, with_cp=with_cp, 
                pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
            )
            self.extractor = Extractor(
                dim=dec_dim, num_heads=num_heads, drop_path=drop_path, 
                cffn_ratio=cffn_ratio, with_cp=with_cp, 
                pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
            )
            if extra_extractor:
                self.extra_extractors = nn.Sequential(*[
                    Extractor(dim=dec_dim, num_heads=num_heads, drop_path=drop_path, 
                            cffn_ratio=cffn_ratio, with_cp=with_cp,
                            pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS)
                    for _ in range(2)
                ])
            else:
                self.extra_extractors = None
        elif dec_dim is None:
            self.injector = Injector(
                dim=enc_dim, num_heads=num_heads, with_cp=with_cp, 
                pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
            )
            self.extractor = Extractor(
                dim=enc_dim, num_heads=num_heads, drop_path=drop_path, 
                cffn_ratio=cffn_ratio, with_cp=with_cp, 
                pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
            )
            if extra_extractor:
                self.extra_extractors = nn.Sequential(*[
                    Extractor(dim=enc_dim, num_heads=num_heads, drop_path=drop_path, 
                            cffn_ratio=cffn_ratio, with_cp=with_cp,
                            pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS)
                    for _ in range(2)
                ])
            else:
                self.extra_extractors = None
        else:
            self.injector = Injector(
                dim=enc_dim, num_heads=num_heads, with_cp=with_cp, 
                pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
            )
            self.extractor = Extractor(
                dim=dec_dim, num_heads=num_heads, drop_path=drop_path, 
                cffn_ratio=cffn_ratio, with_cp=with_cp, 
                pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS
            )
            if extra_extractor:
                self.extra_extractors = nn.Sequential(*[
                    Extractor(dim=dec_dim, num_heads=num_heads, drop_path=drop_path, 
                            cffn_ratio=cffn_ratio, with_cp=with_cp,
                            pale_size=pale_size, cpe_convKS=cpe_convKS, ffn_convKS=ffn_convKS)
                    for _ in range(2)
                ])
            else:
                self.extra_extractors = None 

    def forward(self, x, c, enc_blocks, dec_blocks, dec_embd, c_trans, L, position_ids=None):
        if enc_blocks is None and dec_embd is not None:
            c = c_trans(c)
            x = dec_embd(x)
        x = self.injector(query=x, feat=c, L=L)
        if enc_blocks is not None:
            for idx, blk in enumerate(enc_blocks):
                x = blk(x, position_ids)
        if dec_embd is not None and enc_blocks is not None:
            x = dec_embd(x)
            c = c_trans(c)
        if dec_blocks is not None:
            for idx, blk in enumerate(dec_blocks):
                x = blk(x, position_ids)
        c = self.extractor(query=c, feat=x, L=L)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=x, L=L)
        return x, c


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, out_indices=(0, 1, 2, 3), stem_convKs=3):
        super().__init__()
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(*[
            nn.Conv1d(3, inplanes, kernel_size=stem_convKs, stride=2, padding=(stem_convKs-1)//2, bias=False),
            LayerNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, inplanes, kernel_size=stem_convKs, stride=2, padding=(stem_convKs-1)//2, bias=False),
            LayerNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, inplanes, kernel_size=stem_convKs, stride=1, padding=(stem_convKs-1)//2, bias=False),
            LayerNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(800)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv1d(inplanes, 2 * inplanes, kernel_size=stem_convKs, stride=2, padding=(stem_convKs-1)//2, bias=False),
            LayerNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv1d(2 * inplanes, 4 * inplanes, kernel_size=stem_convKs, stride=2, padding=(stem_convKs-1)//2, bias=False),
            LayerNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv1d(4 * inplanes, 4 * inplanes, kernel_size=stem_convKs, stride=2, padding=(stem_convKs-1)//2, bias=False),
            LayerNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        if len(out_indices) == 4:
            self.fc1 = nn.Conv1d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv1d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv1d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv1d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        if len(self.out_indices) == 4:
            c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        c2 = c2.transpose(1, 2)  # 8s
        c3 = c3.transpose(1, 2)  # 16s
        c4 = c4.transpose(1, 2)  # 32s

        if len(self.out_indices) == 4:
            return c1, c2, c3, c4
        else:
            return c2, c3, c4