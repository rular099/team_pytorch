
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from .PatchTST_LSD_MAE import TSTEncoder,positional_encoding
from .MAE_LSD import PatchEmbed1d

def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch

class MAE_PatchTST_independentChannel(nn.Module):
    def __init__(self,
                patch_len=50,
                num_patch=200,
                mask_ratio=None,
                # encoder
                encoder_dim=128, # TODO:MAE is 768
                # decoder
                decoder_dim=64, # TODO:PatchTST is None,MAE is 512
                # transformer
                n_heads=16, # MAE is 12
                d_ff=512, # TODO:Tranformer MLP dimension
                norm='LayerNorm',
                attn_dropout=0,
                dropout=0.2,
                pre_norm=False,
                act='relu',
                res_attention=False,
                n_layers=3, # TODO:number of Transformer layers
                store_attn=False,
                # position embed
                pe='zeros',
                learn_pe=True,  # MAE is False
                # train
                pred_norm = False,
                **kwargs):

        super().__init__()
        # base
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        # encoder
        self.encoder_dim = encoder_dim
        self.encoder_embedding = nn.Linear(patch_len, encoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, encoder_dim)
        self.encoder = TSTEncoder(encoder_dim, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                            pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                            store_attn=store_attn)
        
        
        # decoder
        self.decoder_embedding = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_pos_embed = positional_encoding(pe, learn_pe, num_patch, decoder_dim)
        self.decoder = TSTEncoder(decoder_dim, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)
        '''
        MAE application:
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        '''
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.mask_token = torch.nn.init.normal_(self.mask_token, std=.02)
        self.decoder_pred = nn.Linear(decoder_dim, patch_len, bias=True)
        
        self.norm_pix_loss = pred_norm

    def forward_encoder(self,x):
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        output:
            u: [bs * nvars x kept_len x encoder_dim]
            x_masked: [B x kept_len x encoder_dim]
            mask=ids_restore: [B x patch_num]
        """
        # TODO:Independent channel from PatchTST
        x = self.nvars2bs(x)                                                     # x->[bs * nvars x num_patch x patch_len]

        # Input encoding
        u = self.encoder_embedding(x)                                            # u: [bs * nvars x num_patch x encoder_dim]

        # Position embedding
        # TODO:MAE is without dropout
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x encoder_dim]
        
        # TODO:Random masking
        u, mask, ids_restore = self.random_masking(u)                            # u -> [bs * nvars x kept_len x encoder_dim] , mask=[B,patch_num]

        # TODO:append cls token?
        
        # Encoder: TSTEncoder
        u = self.encoder(u)                                                      # u: [bs * nvars x kept_len x encoder_dim]

        return u, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore) -> Tensor:          
        """
        # x: [bs * nvars x kept_len x encoder_dim]
        """
        # Input encoding
        x = self.decoder_embedding(x)                                            # x: [bs * nvars x kept_len x encoder_dim] -> [bs * nvars x kept_len x decoder_dim]
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1) # mask_tokens: [bs * nvars,drop_len,decoder_dim]
        x_ = torch.cat([x, mask_tokens], dim=1)  # x_ = [B,kept_len+drop_len,decoder_dim] = [bs * nvars,num_patch,decoder_dim]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # add pos embed
        x_ = x_ + self.decoder_pos_embed

        # TODO:decoder is TSTEncoder?
        x_ = self.decoder(x_) # x_: [bs * nvars,num_patch,decoder_dim]
        '''
        # decoder application from MAE:
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        '''

        # predictor projection
        x_ = self.decoder_pred(x_) # x_ -> [bs * nvars,num_patch,patch_len]
        
        return x_
    
    def nvars2bs(self,x):
        '''
        input:  [bs x num_patch x nvars x patch_len]
        output: [bs * nvars x num_patch x patch_len]
        '''
        bs, num_patch, n_vars, patch_len = x.shape
        x = x.transpose(1,2)                                              # x: [bs x nvars x num_patch x patch_len]
        x = torch.reshape(x, (bs*n_vars, num_patch, patch_len) )          # x: [bs * nvars x num_patch x patch_len]
        return x
    
    def forward_loss(self, z, pred, mask):
        """
        z: [bs x num_patch x n_vars x patch_len]
        pred: [bs * nvars x num_patch x patch_len]
        """
        target = self.nvars2bs(z) # target: [bs * nvars x num_patch x patch_len]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [bs * nvars, num_patch], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, x):
        """
        x: [bs x n_vars x seq_len]
        """
        z,num_patch = create_patch(torch.transpose(x,2,1),self.patch_len,self.patch_len) # z: [bs x num_patch x n_vars x patch_len]
        assert num_patch == self.num_patch
        latent, mask, ids_restore = self.forward_encoder(z)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(z, pred, mask)
        return loss, pred, mask


    
    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [bs * nvars x num_patch x encoder_dim]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # x_masked: [bs x len_kept x encoder_dim]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore # x_masked=[B x kept_len x encoder_dim], mask=ids_restore=[B x patch_num]

class MAE_PatchTST(nn.Module):
    def __init__(self,
                patch_len=50,
                mask_ratio=None,
                num_patch=200,
                # encoder
                encoder_dim=192, # TODO:MAE is 768
                encoder_depth=12, # TODO
                encoder_num_heads=6,
                encoder_mlp=768, # TODO:MAE is encoder_dim*4
                # decoder
                decoder_dim=512, # TODO
                decoder_depth=8, # TODO
                decoder_num_heads=16,
                decoder_mlp=768,
                # transformer
                norm='LayerNorm',
                attn_dropout=0,
                dropout=0.2,
                pre_norm=False,
                act='relu',
                res_attention=False,
                store_attn=False,
                # position embed
                pe='sincos',
                learn_pe=True,  # MAE is False
                # train
                pred_norm = False,
                **kwargs):
        
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = PatchEmbed1d(
            img_size = num_patch*patch_len,
            patch_size = patch_len,
            in_chans = 3,
            embed_dim = encoder_dim,
        )
        self.mask_ratio = mask_ratio
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_dim), requires_grad=False)  # fixed sin-cos embedding

        self.encoder = TSTEncoder(encoder_dim, encoder_num_heads, d_ff=encoder_mlp, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                            pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=encoder_depth, 
                            store_attn=store_attn)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder = TSTEncoder(decoder_dim, decoder_num_heads, d_ff=decoder_mlp, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=decoder_depth, 
                                    store_attn=store_attn)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred = nn.Linear(decoder_dim, patch_len * 3, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = pred_norm

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = positional_encoding('sincos', False, self.patch_embed.num_patches + 1, self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))


        decoder_pos_embed = positional_encoding('sincos', False, self.patch_embed.num_patches + 1, self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, L)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size

        l = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(imgs.shape[0], l, p * imgs.shape[1]))
        return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]
        
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs
    def unpatchify(self, x):
        """
        x: (N, L, p*3)
        imgs: (N, 3, L)
        """
        p = self.patch_embed.patch_size
        l = x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], l, p, 3))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        return imgs
    
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

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        '''
        input:
            x: [bs x patch_num x encoder_dim]
        output:
            x: [bs x (kept_num + 1) x encoder_dim]
            mask=ids_restore: [bs x patch_num]
        '''
        # embed patches
        x = self.patch_embed(x) # x: [bs x patch_num x encoder_dim]

        # add pos embed w/o cls token
        # TODO: in patchTST, cls do not have pos embedding
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # cls_tokens: [bs x 1 x encoder_dim]
        x = torch.cat((cls_tokens, x), dim=1) # x: [bs x (kept_num + 1) x encoder_dim]

        # apply Transformer blocks
        x = self.encoder(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
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
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed # x: [bs x (1 + patch_num) x decoder_dim]

        # apply Transformer blocks
        x = self.decoder(x) # x: [bs x (1 + patch_num) x decoder_dim]

        # predictor projection
        x = self.decoder_pred(x) # x: [bs x (1 + patch_num) x nvars * patch_len]

        # remove cls token
        x = x[:, 1:, :] # x: [bs x patch_num x nvars * patch_len]

        return x

    def forward_loss(self, x, pred, mask):
        """
        x: [bs x n_vars x seq_num]
        pred: [bs x patch_num x nvars * patch_len]
        mask: [bs x patch_num], 0 is keep, 1 is remove, 
        """
        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x,):
        '''
        x: [bs x nvars x seq_len]
        '''
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

def mae_TSTencoder_small(**kwargs):
    model = MAE_PatchTST(
            # encoder
            encoder_dim=192,
            encoder_depth=12,
            encoder_num_heads=6,
            encoder_mlp=768,
            # decoder
            decoder_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            decoder_mlp=2048, 
            **kwargs)
    return model

def mae_TSTencoder_base(**kwargs):
    model = MAE_PatchTST(
            # encoder
            encoder_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_mlp=3072,
            # decoder
            decoder_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            decoder_mlp=2048, 
            **kwargs)
    return model

if __name__ == '__main__':
    torch.manual_seed(42)
    input = torch.randn((32,3,10000))
    # model = MAE_PatchTST_independentChannel()
    model = MAE_PatchTST()
    loss,pred,mask = model(input)
    print(loss)
    '''
    def mae_vit_base_patch30_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=30, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    '''