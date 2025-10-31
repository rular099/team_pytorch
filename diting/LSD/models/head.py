"""
Implementation of the task head of the model.
ref: https://github.com/senli1073/SeisT/blob/main/models/seist.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from mup import MuReadout

import LSD.models.backbone_ablation as backbone_ablation
import LSD.models.backbone_abla_models as backbone_ablation_models
from .uper_head import UPerHead,TaskSeparatedUPerHead, TaskSeparatedUPerHead_new
from .conv_module import ConvModule


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.

    The output of conv-layer has the shape as `ceil(x.size(dim)/stride)`.

    Use this function to replace `padding='same'` which `torch.jit` and `torch.onnx` do not support.

    Args:
        x (torch.Tensor): N-dimensional tensor.
        input (Tensor): N-dimensional tensor
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.
        dim (int): Dimension to pad.
        padding_value (float): fill value.

    Raises:
        AssertionError: `kernel_size` is less than `stride`.

    Returns:
        torch.Tensor : padded tensor.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x

class EncoderFeatures(UPerHead):
    def __init__(self, 
                 encoder_dim,
                 args,
                 num_scales=3,
                 ):
        super(EncoderFeatures, self).__init__(in_channels=[encoder_dim]*num_scales, 
                 out_channels=args.out_channels,
                 num_classes=25*16,
                 fpn_convKS=args.fpn_convKS,
                 aggregate_convKS=args.aggregate_convKS,
                 head_convKS=args.head_convKS,
                 pool_scales=(1, 2, 4, 10), 
                 dropout_ratio=args.head_drop_rate,
                 args=args)
        self.mode = args.inter_mode
        in_channels=[encoder_dim]*num_scales
        out_channels=args.out_channels
        if self.mode == 'fpn_deep_5':
            self.fpn_bottleneck = ConvModule(
                len(in_channels) * out_channels,
                out_channels * 5,
                kernel_size=args.aggregate_convKS,
                stride=2,
                inplace=False
            )

    def forward(self, inputs):
        """Forward function."""
        x = inputs.pop()
        if self.mode == 'max_fpn':
            fpn_out, psp_out, fpn_feature = self._forward_feature(inputs)
            inter_input = fpn_feature
        elif self.mode == 'max_multi_scale':
            fpn_out, psp_out, fpn_feature = self._forward_feature(inputs)
            inter_input = inputs
        else:
            fpn_out, psp_out = self._forward_feature(inputs)
            inter_input = inputs
        
        task_head_inp = psp_out if self.args.reuse_ppm else fpn_out
        if 'det' in self.task_list:
            out = self.projlast_dpk(task_head_inp)
        elif 'emg' in self.task_list:
            out = self.projlast_emg(task_head_inp)
        elif 'dis' in self.task_list:
            out = self.projlast_dis(task_head_inp)
        out = self.gap(out).squeeze(-1)  # --> (B,C)

        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class HeadDetectionPicking(nn.Module):
    """Head of detection and phase-picking."""

    def __init__(
        self,
        feature_channels,
        output_length=10000,
        layer_channels=[64, 32, 32, 16, 16, 3],
        layer_kernel_sizes=[11, 11, 11, 11, 7, 11],
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        out_act_layer=nn.Sigmoid,
        out_channels=3,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)
        self.output_length = output_length
        self.depth = len(layer_channels)

        self.up_layers = nn.ModuleList()

        for i, (inc, outc, kers) in enumerate(
            zip(
                [feature_channels] + layer_channels[:-1],
                layer_channels[:-1] + [out_channels * 2],
                layer_kernel_sizes,
            )
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()

            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x):
        N, C, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=self.output_length)
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


# head_v1
class MuHeadDetectionPicking(HeadDetectionPicking):
    """Head of detection and phase-picking."""

    def __init__(
        self,
        feature_channels,
        hidden_dim=128,
        output_length=10000,
        out_channels=3,
    ):
        super().__init__(feature_channels=hidden_dim, output_length=output_length,out_channels=out_channels)
        self.fc = MuReadout(feature_channels, hidden_dim)

    def forward(self, x):
        x = self.fc(x.transpose(1, 2)).transpose(1, 2)
        _, _, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=self.output_length)
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear") #linear
            # x = torch.nn.Upsample(size=upsize, mode='nearest')(x)
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


# head_v2
class MuHeadDetectionPicking_decoder(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_size,
        args,
    ):
        super().__init__()
        self.decoder = backbone_ablation.Decoder_baseline_llama(
            encoder_dim=encoder_dim,
            decoder_size=decoder_size,
            args=args
        )
        del self.decoder.mask_token
        
        self.patch_len = args.patch_size
        self.decoder_projlast = args.decoder_projlast
        if self.decoder_projlast:
            self.projlast = nn.Conv1d(3, 3, args.convKS, 1, (args.convKS-1)//2)
        self.act_out = nn.Sigmoid()

    def unpatchify(self, x):
        """
        x: (N, L, p*3)
        imgs: (N, 3, L)
        """
        p = self.patch_len
        l = x.shape[1]
        x = x.reshape(shape=(x.shape[0], l, p, 3))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        return imgs
        
    def forward(self, x, return_dict=False):
        x = x.transpose(1, 2)
        ft_enc = x
        h = self.decoder(x, ids_restore=None)
        ft_dec = h
        h = self.unpatchify(h)
        if self.decoder_projlast:
            h = self.projlast(h)
        h = self.act_out(h)
        
        if return_dict:
            output_dict = {"pred":h, "after_enc":ft_enc, "after_dec":ft_dec}
            return output_dict
        else:
            return h
    

class Norm1d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.ln = backbone_ablation_models.LlamaRMSNorm(embed_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1).contiguous()
        return x
    

# head_v3
class MuHeadDetectionPicking_decoder_fpn_v1(nn.Module):
    def __init__(
        self, 
        encoder_dim,
        decoder_size,
        args,
        mla_channels=256, 
    ):
        super().__init__()
        # decoder
        self.decoder = backbone_ablation.Decoder_baseline_llama(
            encoder_dim=encoder_dim,
            decoder_size=decoder_size,
            args=args
        )
        del self.decoder.mask_token
        decoder_size = decoder_size['decoder_dim']

        # fpn
        self.enable_fpn = args.enable_fpn
        self.mla_num = args.num_scales # 1
        self.fpn_layer_idx = args.fpn_layer_idx # 4
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
            Norm1d(decoder_size),
            nn.GELU(),
            nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # refine net
        self.output_length = args.in_samples
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsampling_projs = nn.ModuleList()
        self.upsample_scales = [25, 100, 10] # order-2
        for i in range(self.mla_num):
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv1d(decoder_size, mla_channels, 1, bias=False),
                    Norm1d(mla_channels), 
                    nn.GELU()
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv1d(mla_channels, mla_channels, args.fpn_convKS, padding=(args.fpn_convKS-1)//2, bias=False), 
                    Norm1d(mla_channels), 
                    nn.GELU()
                )
            )
            self.upsampling_projs.append(
                nn.Sequential(
                    nn.Linear(mla_channels, self.upsample_scales[i] * 3, bias=True) # decoder to patch
                )
            )

        self.aggregate_tyee = args.aggregate_tyee
        if self.aggregate_tyee == 'concat':
            self.projlast = nn.Conv1d((self.mla_num+1) * 3, 3, args.convKS, 1, (args.convKS-1)//2)
        else:
            self.projlast = nn.Conv1d(3, 3, args.convKS, 1, (args.convKS-1)//2)
        self.act_out = nn.Sigmoid()

    def unpatchify(self, x, p):
        """
        p: patch_len
        """
        l = x.shape[1]
        x = x.reshape(shape=(x.shape[0], l, p, 3))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        return imgs
    
    def forward(self, x, return_dict=False):
        x = x.transpose(1, 2)
        ft_enc = x
        outs = self.decoder(x, ids_restore=None, hierarchical=True)
        outs[-1] = self.decoder.decoder_norm(outs[-1])
        
        ft_dec = outs[-1]
        ### build fpn inputs
        # 1/16
        x = self.decoder.decoder_pred(outs[-1])
        x = self.unpatchify(x, 50) # --> (B,3,10000)
        # 1/8
        ops = [self.fpn2, self.fpn4, self.fpn1] # order-2
        ops = ops[:self.mla_num]
        # outs = [outs[self.fpn_layer_idx-1]] * len(ops)
        outs = [outs[-1]] * len(ops)
        assert len(outs) == len(ops), "the length of features are not equal to fpn scales"
        z = []
        for i in range(len(ops)):
            z.append(ops[i](outs[i].permute(0, 2, 1))) # --> (B,C,L)
        
        ft_fpn_inp = z[0]
        # build fpn outputs
        outs = []
        for i in range(len(ops)):
            out = self.lateral_convs[i](z[i]) # build laterals
            out = self.fpn_convs[i](out) # --> (B,C,L)
            out = self.upsampling_projs[i](out.permute(0, 2, 1)) # --> (B,L,C)
            out = self.unpatchify(out, self.upsample_scales[i]) # --> (B,3,L')
            outs.append(F.interpolate(out, size=self.output_length, mode='linear')) # --> (B,3,10000)
        
        outs.append(F.interpolate(x, size=self.output_length, mode='linear')) # --> (B,3,10000)
        
        # aggregate features
        if self.aggregate_tyee == 'concat':
            out = torch.cat(outs, dim=1) # --> (B,3*4,10000)
        else:
            out = torch.mean(torch.stack(outs), dim=0) # --> (B,3,10000)
        
        ft_fpn_out = out
        out = self.projlast(out)
        out = self.act_out(out)

        if return_dict:
            output_dict = {"pred":out, "after_enc":ft_enc, "after_dec":ft_dec, "after_fpn_inp":ft_fpn_inp, "after_fpn_out":ft_fpn_out}
            return output_dict
        else:
            return out


# head_v3
class MuHeadDetectionPicking_decoder_fpn_v2(nn.Module):
    def __init__(
        self, 
        encoder_dim,
        decoder_size,
        args,
        mla_channels=256, 
    ):
        super().__init__()
        # decoder
        self.decoder = backbone_ablation.Decoder_baseline_llama(
            encoder_dim=encoder_dim,
            decoder_size=decoder_size,
            args=args
        )
        del self.decoder.mask_token
        del self.decoder.decoder_pred
        decoder_size = decoder_size['decoder_dim']

        # fpn
        self.enable_fpn = args.enable_fpn
        self.mla_num = args.num_scales # 2
        self.fpn_layer_idx = args.fpn_layer_idx # 4
        
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Identity()
        # refine net
        self.output_length = args.in_samples
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_scales = [25, 50]
        for i in range(2):
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv1d(decoder_size, mla_channels, 1, bias=False),
                    Norm1d(mla_channels), 
                    nn.ReLU()
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv1d(mla_channels, mla_channels, args.fpn_convKS, padding=(args.fpn_convKS-1)//2, bias=False), 
                    Norm1d(mla_channels), 
                    nn.ReLU()
                )
            )
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv1d(mla_channels, mla_channels, args.fpn_convKS, padding=(args.fpn_convKS-1)//2, bias=False), 
                    Norm1d(mla_channels), 
                    nn.ReLU()
                )
            )

        self.aggregate_tyee = args.aggregate_tyee
        self.decoder_pred = nn.Linear(mla_channels, 25*3, bias=True)
        self.projlast = nn.Conv1d(3, 3, args.convKS, 1, (args.convKS-1)//2)
        self.act_out = nn.Sigmoid()

    #     self.init_weights()
    
    # def init_weights(self):
    #     # init conv1d with kaiming_init
    #     if not hasattr(self.conv, 'init_weights'):
    #         kaiming_init(self.conv, a=0, nonlinearity='relu')
    #     # init norm with constant value
    #     if self.with_norm:
    #         constant_init(self.norm, 1, bias=0)

    def unpatchify(self, x, p):
        """
        p: patch_len
        """
        l = x.shape[1]
        x = x.reshape(shape=(x.shape[0], l, p, 3))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        return imgs
    
    def forward(self, x, return_dict=False):
        x = x.transpose(1, 2)
        outs = self.decoder(x, ids_restore=None, hierarchical=True)
        fpn_input = outs[self.fpn_layer_idx-1]
        fpn_input = self.decoder.decoder_norm(fpn_input)

        ### build fpn inputs
        ops = [self.fpn1, self.fpn2]
        outs = [fpn_input] * len(ops)
        assert len(outs) == len(ops), "the length of features are not equal to fpn scales"
        z = []
        for i in range(len(ops)):
            z.append(ops[i](outs[i].permute(0, 2, 1))) # --> (B,C,L)
        
        ### build fpn outputs
        outs = []
        for i in range(len(ops)):
            out = self.lateral_convs[i](z[i]) # build laterals
            out = self.fpn_convs[i](out) # --> (B,C,L)
            out = self.up_convs[i](out) # --> (B,C,L)
            if i == 1:
                out = F.interpolate(out, scale_factor=2, mode='linear', align_corners=False)
            outs.append(out)
        
        # aggregate features
        out = torch.sum(torch.stack(outs), dim=0) # --> (B,C,L)
        out = self.decoder_pred(out.permute(0, 2, 1)) # --> (B,L,C)
        out = self.unpatchify(out, 25) # --> (B,3,10000)
        out = self.projlast(out)
        out = self.act_out(out)
        return out
    

# head_v4
class MuHeadDetectionPicking_upernet(UPerHead):
    def __init__(
        self, 
        encoder_dim,
        args,
        num_scales=3,
    ):
        super().__init__(in_channels=[encoder_dim]*num_scales, 
                 out_channels=args.out_channels,
                 fpn_convKS=args.fpn_convKS,
                 aggregate_convKS=args.aggregate_convKS,
                 head_convKS=args.head_convKS,
                 dropout_ratio=args.head_drop_rate,
                 args=args)
        
class MuHead_TaskSeparatedUPerHead(TaskSeparatedUPerHead):
    def __init__(
        self, 
        encoder_dim,
        args,
        num_scales=3,
    ):
        super().__init__(in_channels=[encoder_dim]*num_scales, 
                 out_channels=args.out_channels,
                 fpn_convKS=args.fpn_convKS,
                 aggregate_convKS=args.aggregate_convKS,
                 head_convKS=args.head_convKS,
                 dropout_ratio=args.head_drop_rate,
                 args=args)
        
class MuHead_TaskSeparatedUPerHead_new(TaskSeparatedUPerHead_new):
    def __init__(
        self, 
        encoder_dim,
        args,
        num_scales=3,
    ):
        super().__init__(in_channels=[encoder_dim]*num_scales, 
                 out_channels=args.out_channels,
                 fpn_convKS=args.fpn_convKS,
                 aggregate_convKS=args.aggregate_convKS,
                 head_convKS=args.head_convKS,
                 dropout_ratio=args.head_drop_rate,
                 args=args)

class HeadClassification(nn.Module):
    """
    Head of the model for classification task.
    """

    def __init__(self, feature_dim, num_classes):
        """
        Args:
            feature_dim (int): feature dimension of the model.
            num_classes (int): number of classes.
        """
        super(HeadClassification, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        ### muP: replace nn.Linear with MuReadout
        self.fc = MuReadout(self.feature_dim, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.fc(x)
        x = self.softmax(x)
        return x


class HeadRegression(nn.Module):
    """
    Head of the model for regression task.
    """

    def __init__(self, feature_dim, downstream_task):
        """
        Args:
            feature_dim (int): feature dimension of the model.
        """
        super(HeadRegression, self).__init__()
        self.feature_dim = feature_dim
        self.downstream_task = downstream_task
        ### muP: replace nn.Linear with MuReadout
        if downstream_task == 'emg':
            self.fc = MuReadout(self.feature_dim, 1)
            self.act = nn.Sigmoid()
            self.scale_factor = 9.0
        elif downstream_task == 'baz':
            self.fc1 = MuReadout(self.feature_dim, 1)
            self.fc2 = MuReadout(self.feature_dim, 1)
            self.act = nn.Tanh()
            self.scale_factor = 360.0
        elif downstream_task == 'dis':
            self.fc = MuReadout(self.feature_dim, 1)
            self.act = nn.Sigmoid()
            self.scale_factor = 500.0
        else:
            self.scale_factor = 1.0

    def forward(self, x):

        if self.downstream_task == 'baz':
            cos_x = self.act(self.fc1(x))
            sin_x = self.act(self.fc2(x))
            return [cos_x, sin_x]
        else:
            x = self.fc(x)
            x = self.act(x) * self.scale_factor
            return x
        


# head_v3
class MuHeadDetectionPicking_decoder_fpn_v0(nn.Module):
    def __init__(
        self, 
        encoder_dim,
        decoder_size,
        args,
        mla_channels=256, 
    ):
        super().__init__()
        # decoder
        self.decoder = backbone_ablation.Decoder_baseline_llama(
            encoder_dim=encoder_dim,
            decoder_size=decoder_size,
            args=args
        )
        del self.decoder.mask_token
        del self.decoder.decoder_norm
        del self.decoder.decoder_pred
        decoder_size = decoder_size['decoder_dim']

        self.enable_fpn = args.enable_fpn
        self.mla_num = args.num_scales
        if self.enable_fpn:
            # fpn
            self.fpn_layer_idx = args.fpn_layer_idx
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
                Norm1d(decoder_size),
                nn.GELU(),
                nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose1d(decoder_size, decoder_size, kernel_size=2, stride=2),
            )
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool1d(kernel_size=2, stride=2)
            
            # refine net
            self.output_length = args.in_samples
            self.lateral_convs = nn.ModuleList()
            self.fpn_convs = nn.ModuleList()
            self.upsampling_projs = nn.ModuleList()
            if self.mla_num < 4:
                self.upsample_scales = [50, 25, 100, 10] # order-2
            else:
                self.upsample_scales = [10, 25, 50, 100] # order-1
            for i in range(self.mla_num):
                self.lateral_convs.append(
                    nn.Sequential(
                        nn.Conv1d(decoder_size, mla_channels, 1, bias=False),
                        Norm1d(mla_channels), 
                        nn.ReLU()
                    )
                )
                self.fpn_convs.append(
                    nn.Sequential(
                        nn.Conv1d(mla_channels, mla_channels, args.fpn_convKS, padding=(args.fpn_convKS-1)//2, bias=False), 
                        Norm1d(mla_channels), 
                        nn.ReLU()
                    )
                )
                self.upsampling_projs.append(
                    nn.Sequential(
                        nn.Linear(mla_channels, self.upsample_scales[i] * 3, bias=True) # decoder to patch
                    )
                )
        else:
            # norm and pred
            self.upsampling_projs = nn.ModuleList()
            for i in range(self.mla_num):
                self.upsampling_projs.append(
                    nn.Sequential(
                        backbone_ablation_models.get_norm_fn('rmsnorm', decoder_size),
                        nn.Linear(decoder_size, 50 * 3, bias=True) # decoder to patch
                    )
                )

        self.aggregate_tyee = args.aggregate_tyee
        if self.aggregate_tyee == 'concat':
            self.projlast = nn.Conv1d(self.mla_num * 3, 3, args.convKS, 1, (args.convKS-1)//2)
        else:
            self.projlast = nn.Conv1d(3, 3, args.convKS, 1, (args.convKS-1)//2)
        self.act_out = nn.Sigmoid()

        # self.init_weights()
    
    '''
    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            kaiming_init(self.conv, a=0, nonlinearity='relu')
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)
    '''

    def unpatchify(self, x, p):
        """
        p: patch_len
        """
        l = x.shape[1]
        x = x.reshape(shape=(x.shape[0], l, p, 3))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        return imgs
    
    def forward(self, x):
        x = x.transpose(1, 2)
        outs = self.decoder(x, ids_restore=None, hierarchical=True)

        ### build fpn inputs
        if self.enable_fpn:
            if self.mla_num < 4:
                ops = [self.fpn3, self.fpn2, self.fpn4, self.fpn1] # order-2
                ops = ops[:self.mla_num]
                assert self.fpn_layer_idx == 4
            else:
                ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4] # fine to coarse

            if self.fpn_layer_idx == 0: # all decoder layers
                outs = list(reversed(outs)) # fine to coarse
            else:
                outs = [outs[self.fpn_layer_idx-1]] * len(ops)
            assert len(outs) == len(ops), "the length of features are not equal to fpn scales"
            z = []
            for i in range(len(ops)):
                z.append(ops[i](outs[i].permute(0, 2, 1))) # --> (B,C,L)
            
            # build fpn outputs
            outs = []
            for i in range(len(ops)):
                out = self.lateral_convs[i](z[i]) # build laterals
                out = self.fpn_convs[i](out) # --> (B,C,L)
                out = self.upsampling_projs[i](out.permute(0, 2, 1)) # --> (B,L,C)
                out = self.unpatchify(out, self.upsample_scales[i]) # --> (B,3,L')
                outs.append(F.interpolate(out, size=self.output_length, mode='linear')) # --> (B,3,10000)
        else:
            outs = [self.unpatchify(self.upsampling_projs[i](outs[i]), 50) for i in range(len(outs))]
        
        # aggregate features
        if self.aggregate_tyee == 'concat':
            out = torch.cat(outs, dim=1) # --> (B,3*4,10000)
        else:
            out = torch.mean(torch.stack(outs), dim=0) # --> (B,3,10000)
        out = self.projlast(out)
        out = self.act_out(out)
        return out
