# Copyright (c) Shanghai AI Lab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .adapter_modules import SpatialPriorModule, InteractionBlock, InteractionBlockV2
from .conv_module import LayerNorm
from .backbone_abla_models import PatchTST_base
import LSD.models.backbone_ablation as backbone_ablation
from mup import MuReadout
def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


class ViTAdapter(PatchTST_base):
    def __init__(self, encoder_size, input_length=10000, c_in=3, args=None, 
                 conv_inplane=64, num_heads=16, cffn_ratio=0.25,
                 add_vit_feature=True, use_extra_extractor=False, with_cp=False,
                 out_indices=(0, 1, 2), use_final_norm=True, out_x=True):

        patch_size = args.patch_size
        assert(input_length % patch_size == 0)
        num_patch = input_length // patch_size
        super().__init__(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_size,
            **encoder_size,
            pre_norm=True,
            norm=args.norm_layer,
            xattn=args.xattn,
            config=args,
        )
        self.args = args

        self.interaction_indexes = args.interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.out_indices = out_indices
        self.use_final_norm = use_final_norm
        self.num_block = len(self.backbone.encoder.layers)
        self.out_x = out_x
        embed_dim = self.backbone.d_model

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim,
                                      out_indices=out_indices, stem_convKs=args.stem_convKs)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=num_heads, drop_path=args.drop_path,
                             cffn_ratio=cffn_ratio, with_cp=with_cp, pale_size=args.pale_size, 
                             cpe_convKS=args.cpe_kernel_size, ffn_convKS=args.ffn_convKS,
                             extra_extractor=((True if i == len(
                                 self.interaction_indexes) - 1 else False) and use_extra_extractor))
            for i in range(len(self.interaction_indexes))
        ])
        if len(out_indices) == 4:
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            if self.use_final_norm:
                self.norm1 = LayerNorm(embed_dim)
            self.up.apply(self._init_weights)
        
        self.up_ = nn.ConvTranspose1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, stride=2)
        self.down_ = nn.MaxPool1d(kernel_size=2, stride=2)
        
        if self.use_final_norm:
            self.norm2 = LayerNorm(embed_dim)
            self.norm3 = LayerNorm(embed_dim)
            self.norm4 = LayerNorm(embed_dim)

        # parameter init
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        normal_(self.level_embed)
        
        # freeze the backbone (for LP)
        if "linear_probe" in args.eval_type:
            _freeze_params(self.backbone)
        else:
            pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # SPM forward
        if len(self.out_indices) == 4:
            c1, c2, c3, c4 = self.spm(x)
        else:
            c2, c3, c4 = self.spm(x)

        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)                        # c: [bs x num_patch x d_model]

        ##### interact with backbone #####
        # Input encoding (Embedding output Scaling)
        x = self.backbone.W_p(x) * self.args.input_mult           # x: [bs x d_model x num_patch]
        x = x.transpose(1, 2)                                     # x: [bs x num_patch x d_model]
        position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
        _, L, _ = x.shape

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # x: [bs x num_patch x d_model], c: [bs x num_patch x d_model]
            x, c = layer(x, c, self.backbone.encoder.layers[indexes[0]:indexes[-1]], L, position_ids)
        ##### END #####

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).contiguous()
        c3 = c3.transpose(1, 2).contiguous()
        c4 = c4.transpose(1, 2).contiguous()
        if len(self.out_indices) == 4:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            if len(self.out_indices) == 4:
                x3 = x.transpose(1, 2).contiguous()
                x1 = F.interpolate(x3, scale_factor=4, mode='linear', align_corners=False)
                x2 = F.interpolate(x3, scale_factor=2, mode='linear', align_corners=False)
                x4 = F.interpolate(x3, scale_factor=0.5, mode='linear', align_corners=False)
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            else:
                x3 = x.transpose(1, 2).contiguous()
                # x2 = F.interpolate(x3, scale_factor=2, mode='linear', align_corners=False)
                # x4 = F.interpolate(x3, scale_factor=0.5, mode='linear', align_corners=False)
                x2 = self.up_(x3)
                x4 = self.down_(x3)
                c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        if self.use_final_norm:
            if len(self.out_indices) == 4:
                f1 = self.norm1(c1.float()).contiguous()
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                return [f1, f2, f3, f4]
            else:
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                if not self.out_x:
                    return [f2, f3, f4]
                else:
                    x = self.backbone.encoder_norm(x)
                    return [f2, f3, f4, x]
        else:
            return [c1.float().contiguous(),
                    c2.float().contiguous(),
                    c3.float().contiguous(),
                    c4.float().contiguous()]

# vit_adapter_decoder
class ViTAdapterDecoder(PatchTST_base):
    def __init__(self, encoder_size, input_length=10000, c_in=3, args=None, 
                 conv_inplane=64, num_heads=16, cffn_ratio=0.25,
                 add_vit_feature=True, use_extra_extractor=False, with_cp=False,
                 out_indices=(0, 1, 2), freeze_type='enc', use_final_norm=True):

        patch_size = args.patch_size
        assert(input_length % patch_size == 0)
        num_patch = input_length // patch_size
        super().__init__(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_size,
            **encoder_size,
            pre_norm=True,
            norm=args.norm_layer,
            xattn=args.xattn,
            config=args,
        )
        self.args = args
        self.decoder = backbone_ablation.Decoder_baseline_llama(
            encoder_dim=self.backbone.d_model,
            decoder_size=backbone_ablation.get_decoder_size_dict(),
            args=args
        )
        del self.decoder.mask_token
        del self.decoder.decoder_pred
        
        self.interaction_indexes = args.interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.out_indices = out_indices
        self.use_final_norm = use_final_norm
        self.num_block = len(self.backbone.encoder.layers)
        enc_embed_dim = self.backbone.d_model
        dec_embed_dim = self.decoder.decoder_dim
        self.c_trans = MuReadout(enc_embed_dim, dec_embed_dim, bias=True)

        self.level_embed = nn.Parameter(torch.zeros(3, enc_embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=enc_embed_dim,
                                      out_indices=out_indices, stem_convKs=args.stem_convKs)
        interaction_list = []
        for i in range(len(self.interaction_indexes)):
            enc_dim = None
            dec_dim = None
            if self.interaction_indexes[i][-1] <= 24:
                enc_dim = enc_embed_dim
            elif self.interaction_indexes[i][0] >= 24:
                dec_dim = dec_embed_dim
            else:
                enc_dim, dec_dim = enc_embed_dim, dec_embed_dim
            interaction_block = InteractionBlockV2(enc_dim=enc_dim, dec_dim=dec_dim, num_heads=num_heads, drop_path=args.drop_path,
                             cffn_ratio=cffn_ratio, with_cp=with_cp, pale_size=args.pale_size, 
                             cpe_convKS=args.cpe_kernel_size, ffn_convKS=args.ffn_convKS,
                             extra_extractor=((True if i == len(
                                 self.interaction_indexes) - 1 else False) and use_extra_extractor))
            interaction_list.append(interaction_block)
            
        self.interactions = nn.Sequential(*interaction_list)
        if len(out_indices) == 4:
            self.up = nn.ConvTranspose2d(dec_embed_dim, dec_embed_dim, 2, 2)
            if self.use_final_norm:
                self.norm1 = LayerNorm(dec_embed_dim)
            self.up.apply(self._init_weights)
        
        self.up_ = nn.ConvTranspose1d(in_channels=dec_embed_dim, out_channels=dec_embed_dim, kernel_size=2, stride=2)
        self.down_ = nn.MaxPool1d(kernel_size=2, stride=2)
        if self.use_final_norm:
            self.norm2 = LayerNorm(dec_embed_dim)
            self.norm3 = LayerNorm(dec_embed_dim)
            self.norm4 = LayerNorm(dec_embed_dim)

        # parameter init
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        normal_(self.level_embed)
        
        # freeze the backbone (for LP)
        if freeze_type=='enc':
            _freeze_params(self.backbone)
        elif freeze_type=='enc_dec':
            _freeze_params(self.backbone)
            _freeze_params(self.decoder)
        else:
            pass
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def forward(self, x):
        # SPM forward
        if len(self.out_indices) == 4:
            c1, c2, c3, c4 = self.spm(x)
        else:
            c2, c3, c4 = self.spm(x)

        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)                        # c: [bs x num_patch x d_model]

        ##### interact with backbone #####
        # Input encoding (Embedding output Scaling)
        x = self.backbone.W_p(x) * self.args.input_mult           # x: [bs x d_model x num_patch]
        x = x.transpose(1, 2)                                     # x: [bs x num_patch x d_model]
        position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
        _, L, _ = x.shape
        
        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # x: [bs x num_patch x d_model], c: [bs x num_patch x d_model]
            c_trans, enc, dec, dec_embed = None, None, None, None
            if indexes[-1] <= 24:
                enc = self.backbone.encoder.layers[indexes[0]: indexes[-1]]
            elif indexes[0] <= 23:
                enc = self.backbone.encoder.layers[indexes[0]: 24]
                dec_embed = self.decoder.decoder_embed
                c_trans = self.c_trans
                dec = self.decoder.decoder.layers[0: indexes[-1] - 24]
            elif indexes[0] == 24:
                dec_embed = self.decoder.decoder_embed
                c_trans = self.c_trans
                dec = self.decoder.decoder.layers[0: indexes[-1] - 24] 
            else:
                dec = self.decoder.decoder.layers[indexes[0] - 24: indexes[-1] - 24]
            x, c = layer(x, c, enc, dec, dec_embed, c_trans, L, position_ids)
        ##### END #####

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).contiguous()
        c3 = c3.transpose(1, 2).contiguous()
        c4 = c4.transpose(1, 2).contiguous()
        if len(self.out_indices) == 4:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            if len(self.out_indices) == 4:
                x3 = x.transpose(1, 2).contiguous()
                x1 = F.interpolate(x3, scale_factor=4, mode='linear', align_corners=False)
                x2 = F.interpolate(x3, scale_factor=2, mode='linear', align_corners=False)
                x4 = F.interpolate(x3, scale_factor=0.5, mode='linear', align_corners=False)
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            else:
                x3 = x.transpose(1, 2).contiguous()
                # x2 = F.interpolate(x3, scale_factor=2, mode='linear', align_corners=False)
                # x4 = F.interpolate(x3, scale_factor=0.5, mode='linear', align_corners=False)
                x2 = self.up_(x3)
                x4 = self.down_(x3)
                c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        if self.use_final_norm:
            if len(self.out_indices) == 4:
                f1 = self.norm1(c1.float()).contiguous()
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                return [f1, f2, f3, f4]
            else:
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                return [f2, f3, f4]
        else:
            return [c1.float().contiguous(),
                    c2.float().contiguous(),
                    c3.float().contiguous(),
                    c4.float().contiguous()] 
        