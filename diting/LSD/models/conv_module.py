# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_feature_map(x, size):
    return F.interpolate(x, size=size, mode='linear', align_corners=False)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, seq_length, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, seq_length).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.to(torch.float32)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None].to(torch.float32) * x + self.bias[:, None].to(torch.float32)
            x = x.to(input_dtype)
            return x


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias='auto',
                 norm_cfg='ln',
                 act_cfg='relu',
                 inplace=True):
        super().__init__()
        self.act_cfg = act_cfg
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm

        # build convolution layer
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=bias
        )

        # build normalization layers (todo: more normalization methods)
        if self.with_norm:
            norm_channels = out_channels
            assert norm_cfg == 'ln'
            self.norm = LayerNorm(norm_channels)

        # build activation layer (todo: more activation functions)
        if self.with_activation:
            assert act_cfg == 'relu'
            self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        # Initialize the convolution layer using kaiming initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        
        # Initialize the normalization layer with constant initialization
        if self.with_norm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)
        return x