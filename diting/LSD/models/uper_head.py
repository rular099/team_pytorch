# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from .conv_module import ConvModule, LayerNorm, resize_feature_map
from .psp_head import PPM
from .backbone_ablation import *

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, 
                 in_channels=[1792, 1792, 1792], 
                 out_channels=256,
                 num_classes=25*3,
                 fpn_convKS=3,
                 aggregate_convKS=3,
                 head_convKS=3,
                 pool_scales=(1, 2, 4, 10), 
                 dropout_ratio=0,
                 args=None):
        
        super(UPerHead, self).__init__()
        self.inter_mode = args.inter_mode
        self.in_channels = in_channels
        self.num_classes = num_classes
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            in_channels[-1], # 1792
            out_channels)
        
        self.bottleneck = ConvModule(
            in_channels[-1] + len(pool_scales) * out_channels,
            out_channels, # 256
            kernel_size=aggregate_convKS)
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                out_channels, # 256
                kernel_size=1,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=fpn_convKS,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.up_convs.append(nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2))
        self.up_fpn_convs = nn.ModuleList()
        self.up_fpn_convs.append(nn.Identity())
        self.up_fpn_convs.append(nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2))
        self.up_fpn_convs.append(nn.Sequential(nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2),
                                LayerNorm(out_channels),
                                nn.GELU(),
                                nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2)))
        # self.up_fpn_convs.append(nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=4))
        
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * out_channels,
            out_channels,
            kernel_size=aggregate_convKS,
            inplace=False)
        
        self.act_out = nn.Sigmoid()
        self.args = args
        ### Task Head
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        # dpk head
        self.decoder_pred = nn.Linear(out_channels, num_classes, bias=True)
        self.projlast_dpk = nn.Conv1d(3, 3, head_convKS, 1, (head_convKS-1)//2)
        # other task head
        self.task_list = args.downstream_task.split('_')
        self.gap = nn.AdaptiveAvgPool1d(1)
        if 'dis' in self.task_list:
            self.projlast_dis = nn.Conv1d(out_channels, out_channels, head_convKS, 1, (head_convKS-1)//2)
            self.cls_head_dis = nn.Linear(out_channels, 1, bias=True)
            self.scale_factor_dis = 500.0
        if 'emg' in self.task_list:
            self.projlast_emg = nn.Conv1d(out_channels, out_channels, head_convKS, 1, (head_convKS-1)//2)
            self.cls_head_emg = nn.Linear(out_channels, 1, bias=True)
            self.scale_factor_emg = 9.0
        if 'fmp' in self.task_list:
            self.projlast_fmp = nn.Conv1d(out_channels, out_channels, head_convKS, 1, (head_convKS-1)//2)
            self.cls_head_fmp = nn.Linear(out_channels, 1, bias=True)
            self.scale_factor_fmp = 1.0

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, out_channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # convert to float32
        inputs = [x.float() for x in inputs]
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))
        psp_ft = laterals[-1]

        # build top-down path
        used_backbone_levels = len(laterals) # scale=3
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[-1]
            # laterals[i - 1] = laterals[i - 1] + resize_feature_map(laterals[i], size=prev_shape)
            laterals[i - 1] = laterals[i - 1] + self.up_convs[i - 1](laterals[i])
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            # fpn_outs[i] = resize_feature_map(
            #     fpn_outs[i],
            #     size=fpn_outs[0].shape[-1]
            # )
            fpn_outs[i] = self.up_fpn_convs[i](fpn_outs[i])
        fpn_feature = fpn_outs
        # print(f'---------fpn_outs_len:{len(fpn_outs)}-------------')
        # print(f'---------fpn_outs[0]_shape:{fpn_outs[0].shape}-------------')
        # print(f'---------fpn_outs[1]_shape:{fpn_outs[1].shape}-------------')
        # print(f'---------fpn_outs[2]_shape:{fpn_outs[2].shape}-------------')
        # print(f'---------fpn_outs[3]_shape:{fpn_outs[3].shape}-------------')   
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        if self.inter_mode == 'max_fpn' or self.inter_mode == 'max_multi_scale':
            return feats, psp_ft, fpn_feature
        return feats, psp_ft

    def unpatchify(self, x, p, c=3):
        """
        p: patch_len
        """
        l = x.shape[1]
        x = x.reshape(shape=(x.shape[0], l, p, c))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], c, l * p))
        return imgs
    
    def cls_pixel(self, feat):
        """Classify each pixel."""
        out = self.decoder_pred(feat.permute(0, 2, 1)) # --> (B,L,C) [2, 256, 400] -> [2, 400, 75] , self.decoder_pred:Linear(in_features=256, out_features=75, bias=True)
        out = self.unpatchify(out, self.num_classes//3) # --> (B,3,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        out = self.projlast_dpk(out)
        if self.args.loss_type == 'bce':
            out = self.act_out(out)
        return out
    
    def cls_waveform(self, feat, projlast, cls_head, scale_factor):
        """Classify each waveform."""
        out = projlast(feat)
        out = self.gap(out).squeeze(-1)  # --> (B,C)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = cls_head(out)
        out = self.act_out(out) * scale_factor
        return out
    
    def forward(self, inputs):
        """Forward function."""
        outputs = []
        fpn_out, psp_out = self._forward_feature(inputs)
        outputs.append(self.cls_pixel(fpn_out))
        if 'dis' in self.task_list:
            task_head_inp = psp_out if self.args.reuse_ppm else fpn_out
            outputs.append(
                self.cls_waveform(task_head_inp, self.projlast_dis, self.cls_head_dis, self.scale_factor_dis)
            )
        if 'emg' in self.task_list:
            task_head_inp = psp_out if self.args.reuse_ppm else fpn_out
            outputs.append(
                self.cls_waveform(task_head_inp, self.projlast_emg, self.cls_head_emg, self.scale_factor_emg)
            )

        return outputs
    
class TaskSeparatedUPerHead(UPerHead):
    """Task-separated UPerHead with independent heads for each task.
    
    This class extends UPerHead to support task-specific outputs, where each 
    task is handled by its own dedicated head. The forward method takes an 
    additional argument to specify the task being executed.
    """
    def __init__(self, 
                 in_channels=[1792, 1792, 1792], 
                 out_channels=256,
                 num_classes=25*16, # for task dimension
                 fpn_convKS=3,
                 aggregate_convKS=3,
                 head_convKS=3,
                 pool_scales=(1, 2, 4, 10), 
                 dropout_ratio=0,
                 args=None):
        super(TaskSeparatedUPerHead, self).__init__(
                 in_channels=in_channels, 
                 out_channels=out_channels,
                 num_classes=num_classes,
                 fpn_convKS=fpn_convKS,
                 aggregate_convKS=aggregate_convKS,
                 head_convKS=head_convKS,
                 pool_scales=pool_scales, 
                 dropout_ratio=dropout_ratio,
                 args=args)
        
        if 'det' in self.task_list:
            self.projlast_det = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
        if 'ppk' in self.task_list:
            self.projlast_ppk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
        if 'spk' in self.task_list:
            self.projlast_spk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
        
    def cls_pixel(self, feat, task):
        """Classify each pixel."""
        out = self.decoder_pred(feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        
        if task == 'det':
            out = self.projlast_det(out) # (B,16,10000) --> (B,1,10000)
        elif task == 'ppk':
            out = self.projlast_ppk(out) # (B,16,10000) --> (B,1,10000)
        elif task == 'spk':
            out = self.projlast_spk(out) # (B,16,10000) --> (B,1,10000)
        else:
            raise ValueError(f'Invalid task: {task} for cls_pixel')
        
        if self.args.loss_type == 'bce':
            out = self.act_out(out)
        return out
        
    def forward(self, inputs):
        """Forward function."""
        outputs = {}
        fpn_out, psp_out = self._forward_feature(inputs)
        
        for task in ['det','ppk','spk']:
            if task in self.task_list:
                outputs[task] = self.cls_pixel(fpn_out,task).squeeze(1)
        
        task_head_inp = psp_out if self.args.reuse_ppm else fpn_out
        if 'dis' in self.task_list:
            outputs['dis'] = self.cls_waveform(task_head_inp, self.projlast_dis, self.cls_head_dis, self.scale_factor_dis)
        if 'emg' in self.task_list:
            outputs['emg'] = self.cls_waveform(task_head_inp, self.projlast_emg, self.cls_head_emg, self.scale_factor_emg)
        
        return outputs

class TaskSeparatedUPerHead_new(UPerHead):
    """Task-separated UPerHead with independent heads for each task.
    
    This class extends UPerHead to support task-specific outputs, where each 
    task is handled by its own dedicated head. The forward method takes an 
    additional argument to specify the task being executed.
    """

    def __init__(self, 
                 in_channels=[1792, 1792, 1792], 
                 out_channels=256,
                 num_classes=25*16, # for task dimension
                 fpn_convKS=3,
                 aggregate_convKS=3,
                 head_convKS=3,
                 pool_scales=(1, 2, 4, 10), 
                 dropout_ratio=0,
                 args=None):
        super(TaskSeparatedUPerHead_new, self).__init__(
                 in_channels=in_channels, 
                 out_channels=out_channels,
                 num_classes=num_classes,
                 fpn_convKS=fpn_convKS,
                 aggregate_convKS=aggregate_convKS,
                 head_convKS=head_convKS,
                 pool_scales=pool_scales, 
                 dropout_ratio=dropout_ratio,
                 args=args)
        self.mode = args.inter_mode
        self.decoder = Decoder_baseline_llama(
                encoder_dim=in_channels[0],
                decoder_size=get_decoder_size_dict(),
                args=args
            )
        del self.decoder.mask_token
        del self.decoder.decoder_pred
        self.projlast = nn.Conv1d(3, 3, args.convKS, 1, (args.convKS-1)//2)   
        if self.mode == 'fixed':
            self.connect_convs = nn.ModuleList()
            self.connect_convs.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_channels[0], out_channels=256, kernel_size=2, stride=2), 
                ConvModule(in_channels=256, out_channels=256, kernel_size=3)
                )
            )
            self.connect_convs.append(nn.Sequential(
                nn.Conv1d(in_channels=in_channels[0], out_channels=256, kernel_size=1),
                ConvModule(in_channels=256, out_channels=256, kernel_size=3)
                )
            )
            self.connect_convs.append(nn.Sequential(
                nn.Conv1d(in_channels=in_channels[0], out_channels=256, kernel_size=2, stride=2),
                ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                )   
            ) 
            self.connect_out = nn.Sequential(
                nn.ConvTranspose1d(in_channels=256, out_channels=out_channels, kernel_size=2, stride=2),
                ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3)  
            )
        elif self.mode.startswith('max'):
            if self.mode =='max_fpn':
                in_dim = out_channels
                self.connect_convs = nn.ModuleList()
                self.connect_convs.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                    )
                )
                self.connect_convs.append(nn.Sequential(
                    nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=1),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                    )
                )
                self.connect_convs.append(nn.Sequential(
                    nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=1),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                    )
                )
                self.connect_x = nn.Sequential(
                    nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=1),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                )
                self.connect_out = nn.Sequential(
                    nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1),
                    ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3) 
                ) 
            else:
                in_dim = in_channels[0]
                self.connect_convs = nn.ModuleList()
                self.connect_convs.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=in_dim, out_channels=256, kernel_size=2, stride=2),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                    )
                )
                self.connect_convs.append(nn.Sequential(
                    nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=1),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                    )    
                )
                self.connect_convs.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                    )
                )
                self.connect_x = nn.Sequential(
                    nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=1),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                )
                self.connect_out = nn.Sequential(
                    nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1),
                    ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3) 
                )
            if self.mode == 'max_multi_scale':
                
                self.connect_head = nn.ModuleList()
                self.connect_head.append(
                    ConvModule(out_channels, out_channels, kernel_size=3) 
                )
                self.connect_head.append(
                    ConvModule(out_channels, out_channels, kernel_size=3)
                )
                self.connect_head.append(
                    ConvModule(out_channels, out_channels, kernel_size=3)
                )
                self.projlast_fpn = nn.ModuleList()
                for i in range(3):
                    projlast_fpn_item = nn.ModuleList()
                    projlast_fpn_item.append(nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2))
                    # self.projlast_ppk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
                    projlast_fpn_item.append(nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)) 
                    projlast_fpn_item.append(nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2))
                    self.projlast_fpn.append(projlast_fpn_item)
                # self.projlast_spk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
                self.decoder_pred_fpn = nn.ModuleList()
                self.decoder_pred_fpn.append(nn.Linear(out_channels, num_classes, bias=True))
                self.decoder_pred_fpn.append(nn.Linear(out_channels, num_classes, bias=True))
                self.decoder_pred_fpn.append(nn.Linear(out_channels, num_classes, bias=True))
            elif self.mode == 'max_multi_scale_new':
                self.decoder_pred_fpn = nn.Linear(out_channels, num_classes, bias=True)
                self.projlast_fpn = nn.ModuleList()
                self.projlast_fpn.append(nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2))
                # self.projlast_ppk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
                self.projlast_fpn.append(nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)) 
                self.projlast_fpn.append(nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2))

        elif self.mode.startswith('fpn'):
            if self.mode == 'fpn_deep_3':
                self.connect_x = nn.Sequential(
                    nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
                    ConvModule(in_channels=out_channels, out_channels=256 * 3, kernel_size=3) 
                )
            elif self.mode == 'fpn_deep_5':
                # self.connect_x = nn.Sequential(
                #     nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
                #     ConvModule(in_channels=out_channels, out_channels=256 * 5, kernel_size=3) 
                # )
                self.fpn_bottleneck = ConvModule(
                    len(self.in_channels) * out_channels,
                    out_channels * 5,
                    kernel_size=aggregate_convKS,
                    stride=2,
                    inplace=False
                )
            else:
                self.connect_x = nn.Sequential(
                    nn.Conv1d(in_channels=out_channels, out_channels=256, kernel_size=2, stride=2),
                    ConvModule(in_channels=256, out_channels=256, kernel_size=3) 
                )
            self.connect_out = nn.Sequential(
                nn.ConvTranspose1d(in_channels=256, out_channels=out_channels, kernel_size=2, stride=2),
                ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            )
        self.fused_feature = args.fused_feature
        self.fused_weight = args.fused_weight
        if 'det' in self.task_list:
            self.projlast_det = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
        if 'ppk' in self.task_list:
            self.projlast_ppk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
        if 'spk' in self.task_list:
            self.projlast_spk = nn.Conv1d(16, 1, head_convKS, 1, (head_convKS-1)//2)
        
        self.custom_heads = []
        
    def cls_pixel(self, feat, task):
        """Classify each pixel."""
        out = self.decoder_pred(feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        
        if task == 'det':
            out = self.projlast_det(out) # (B,16,10000) --> (B,1,10000)
        elif task == 'ppk':
            out = self.projlast_ppk(out) # (B,16,10000) --> (B,1,10000)
        elif task == 'spk':
            out = self.projlast_spk(out) # (B,16,10000) --> (B,1,10000)
        else:
            raise ValueError(f'Invalid task: {task} for cls_pixel')
        
        if self.args.loss_type == 'bce':
            fpn_out = self.act_out(out)
        return fpn_out

    def fpn_cls_pixel(self, feat, task, idx):
        """Classify each pixel."""
        
        feat = self.connect_head[idx](feat)
        
        out = self.decoder_pred_fpn[idx](feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        
        if task == 'det':
            out = self.projlast_fpn[idx][0](out) # (B,16,10000) --> (B,1,10000)
        elif task == 'ppk':
            out = self.projlast_fpn[idx][1](out) # (B,16,10000) --> (B,1,10000)
        elif task == 'spk':
            out = self.projlast_fpn[idx][2](out) # (B,16,10000) --> (B,1,10000)
        else:
            raise ValueError(f'Invalid task: {task} for cls_pixel')
        
        if self.args.loss_type == 'bce':
            fpn_out = self.act_out(out)
        return fpn_out
    
    def fpn_cls_pixel_new(self, feat, task):
        """Classify each pixel."""
        
        # feat = self.connect_head[idx](feat)
        
        out = self.decoder_pred_fpn(feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        
        if task == 'det':
            out = self.projlast_fpn[0](out) # (B,16,10000) --> (B,1,10000)
        elif task == 'ppk':
            out = self.projlast_fpn[1](out) # (B,16,10000) --> (B,1,10000)
        elif task == 'spk':
            out = self.projlast_fpn[2](out) # (B,16,10000) --> (B,1,10000)
        else:
            raise ValueError(f'Invalid task: {task} for cls_pixel')
        
        if self.args.loss_type == 'bce':
            fpn_out = self.act_out(out)
        return fpn_out
       
    def forward(self, inputs):
        """Forward function."""
        outputs = {}
        
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
        # f2, f3, f4, x = inputs[0], inputs[1], inputs[2], inputs[3]
        position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
        h = self.decoder.decoder_embed(x)
        if self.mode.startswith('fpn'):
            fpn_fused_feat = [None] * 5
            digit = [int(self.mode.split('_')[-1])]
            if self.mode.startswith('fpn_deep'):
                if digit[0] == 3:
                    digit = [1, 2, 3]
                    parts = self.connect_x(fpn_out).transpose(1, 2).split([256] * 3, dim=2)
                    for idx, part in zip(digit, parts):
                        fpn_fused_feat[idx] = part
                elif digit[0] == 5:
                    digit = [0, 1, 2, 3, 4]
                    # parts = self.connect_x(fpn_out).transpose(1, 2).split([256] * 5, dim=2)
                    parts = fpn_out.transpose(1, 2).split([256] * 5, dim=2) # TODO: fixed channel split
                    for idx, part in zip(digit, parts):
                        fpn_fused_feat[idx] = part 
            else:
                fpn_fused_feat[digit[0]] = self.connect_x(fpn_out).transpose(1, 2)
            
        for i, layer in enumerate(self.decoder.decoder.layers):
            if self.mode.startswith('fpn') and i in digit:
                h = h + fpn_fused_feat[i]
            h = layer(h, position_ids)
            if i < 3:
                if self.mode == 'max_fpn' and i == 0:
                    h = self.connect_convs[i](h.transpose(1, 2)).transpose(1, 2) + self.connect_x(inter_input[2 - i]).transpose(1, 2)
                    position_ids = torch.arange(h.shape[1]).unsqueeze(0).repeat(h.shape[0], 1).to(h.device) 
                elif (self.mode == 'max' or self.mode.startswith('max_multi_scale')) and i == 2:
                    h = self.connect_convs[i](h.transpose(1, 2)).transpose(1, 2) + self.connect_x(inter_input[2 - i]).transpose(1, 2)
                    position_ids = torch.arange(h.shape[1]).unsqueeze(0).repeat(h.shape[0], 1).to(h.device)
                elif self.mode.startswith('max') or self.mode == 'fixed':
                    h = h + self.connect_convs[i](inter_input[2 - i]).transpose(1, 2)                    
        if self.mode.startswith('fpn') and 4 in digit:
            h = h + fpn_fused_feat[4]
        h = h.transpose(1, 2)
        # decoder_norm
        h = self.connect_out(h)
        for task in ['det','ppk','spk']:
            if task in self.task_list:
                if self.mode == 'max_multi_scale':
                    outputs[task] = []
                    outputs[task].append(self.cls_pixel(h, task).squeeze(1))
                    outputs[task].append(self.fpn_cls_pixel(fpn_feature[0], task, 0))
                    outputs[task].append(self.fpn_cls_pixel(fpn_feature[1], task, 1))
                    outputs[task].append(self.fpn_cls_pixel(fpn_feature[2], task, 2))
                elif self.mode == 'max_multi_scale_new':
                    outputs[task] = []
                    outputs[task].append(self.cls_pixel(h, task).squeeze(1))
                    outputs[task].append(self.fpn_cls_pixel_new(fpn_out, task)) 
                else:
                    outputs[task] = self.cls_pixel(h, task).squeeze(1)
        
        task_head_inp = psp_out if self.args.reuse_ppm else fpn_out
        if 'dis' in self.task_list:
            outputs['dis'] = self.cls_waveform(task_head_inp, self.projlast_dis, self.cls_head_dis, self.scale_factor_dis)
        if 'emg' in self.task_list:
            outputs['emg'] = self.cls_waveform(task_head_inp, self.projlast_emg, self.cls_head_emg, self.scale_factor_emg)
        
        if self.mode == 'max_multi_scale' or self.mode == 'max_fpn':
            head_input = {'task_head_inp':task_head_inp,'fpn_feature':[h,fpn_feature],'fpn_out':[h,fpn_out]}
        else:
            head_input = {'task_head_inp':task_head_inp,'fpn_out':[h,fpn_out]}

        for head in self.custom_heads:
            if head['task'] in self.task_list:
                outputs[head['task']] = head['func'](head_input[head['input']],*head['args'])
        
        return outputs 
