# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from .conv_module import ConvModule, resize_feature_map


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        out_channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, pool_scales, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(pool_scale),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        inplace=False)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        target_len = x.size()[-1]
        for i, ppm in enumerate(self):
            ppm_out = ppm(x)
            scale = self.pool_scales[i]
            upsampled_ppm_out = ppm_out.repeat_interleave(target_len//scale, dim=-1)[:, :, :target_len]
            # upsampled_ppm_out = resize_feature_map(
            #     ppm_out,
            #     size=x.size()[-1])
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs