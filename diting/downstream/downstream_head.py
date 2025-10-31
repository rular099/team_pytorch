from LSD.models.uper_head import TaskSeparatedUPerHead_new
import torch.nn as nn
class ClassificationHead(TaskSeparatedUPerHead_new):
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
        self.projlast_cls = nn.Conv1d(args.out_channels, args.out_channels, args.head_convKS, 1, (args.head_convKS-1)//2)
        self.cls_head_cls = nn.Linear(args.out_channels, args.num_classes, bias=True)
        self.act_out_cls = nn.Softmax(dim=1)
        self.custom_heads.append({'task':args.downstream_task,'func':self.cls_waveform,'input':'task_head_inp','args':[]})
    
    def cls_waveform(self,feat):
        out = self.projlast_cls(feat)
        out = self.gap(out).squeeze(-1) # --> (B,C)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.cls_head_cls(out)
        out = self.act_out_cls(out)
        return out

class RegressionHead(TaskSeparatedUPerHead_new):
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
        self.projlast_reg = nn.Conv1d(args.out_channels, args.out_channels, args.head_convKS, 1, (args.head_convKS-1)//2)
        self.cls_head_reg = nn.Linear(args.out_channels, 1, bias=True)
        self.scale_factor_reg = args.head_scale_factor
        reg_head_args = [self.projlast_reg,self.cls_head_reg,self.scale_factor_reg]
        self.custom_heads.append({'task':args.downstream_task,'func':self.cls_waveform,'input':'task_head_inp','args':reg_head_args})

class SegmentationHead(TaskSeparatedUPerHead_new):
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
        self.projlast_seg = nn.Conv1d(16, 1, args.head_convKS, 1, (args.head_convKS-1)//2)
        if self.mode == 'max_multi_scale':
            self.custom_heads.append({'task':args.downstream_task,'func':self.custom_output,'input':'fpn_feature','args':[]})
        else:
            self.custom_heads.append({'task':args.downstream_task,'func':self.custom_output,'input':'fpn_out','args':[]})

    def cls_pixel(self, feat):
        """Classify each pixel."""
        out = self.decoder_pred(feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.projlast_seg(out) # (B,16,10000) --> (B,1,10000)
        
        if self.args.loss_type == 'bce':
            fpn_out = self.act_out(out)
        return fpn_out

    def fpn_cls_pixel(self, feat, idx=0):
        """Classify each pixel."""

        feat = self.connect_head[idx](feat)
        
        out = self.decoder_pred_fpn[idx](feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.projlast_fpn[idx][0](out) # (B,16,10000) --> (B,1,10000)
        
        if self.args.loss_type == 'bce':
            fpn_out = self.act_out(out)
        return fpn_out
    
    def fpn_cls_pixel_new(self, feat):
        """Classify each pixel."""
        
        # feat = self.connect_head[idx](feat)
        
        out = self.decoder_pred_fpn(feat.permute(0, 2, 1)) # -->(B,L,C) feat:(2,256,400)->out:(2,400,256)
        out = self.unpatchify(out, self.num_classes//16, c=16) # -->(B,16,10000)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.projlast_fpn[0](out) # (B,16,10000) --> (B,1,10000)
        
        if self.args.loss_type == 'bce':
            fpn_out = self.act_out(out)
        return fpn_out

    def custom_output(self,feat,projlast):
        if self.mode == 'max_multi_scale':
            h, fpn_feature = feat
            output = []
            output.append(self.cls_pixel(h).squeeze(1))
            output.append(self.fpn_cls_pixel(fpn_feature[0], 0))
            output.append(self.fpn_cls_pixel(fpn_feature[1], 1))
            output.append(self.fpn_cls_pixel(fpn_feature[2], 2))
        elif self.mode == 'max_multi_scale_new':
            h, fpn_out = feat
            output = []
            output.append(self.cls_pixel(h).squeeze(1))
            output.append(self.fpn_cls_pixel_new(fpn_out)) 
        else:
            h = feat[0]
            output = self.cls_pixel(h).squeeze(1)
        return output
heads = {'cls': ClassificationHead,
         'reg': RegressionHead,
         'seg': SegmentationHead}
