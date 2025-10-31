import torch.nn as nn
import torch

from LSD.models import head
from LSD.models.PatchTST_LSD_BYOL import PatchTST_samll
from finetuneing.models.loss import HuberLoss, CELoss, BCELoss, BazLoss, BinaryFocalLoss, CircleLoss
import LSD.models.backbone_ablation as backbone_ablation
import LSD.models.vit_adapter as vit_adapter
from finetuneing.utils.pooler import Attentive, AVGPool
from finetuneing.datasets.SFTData import get_id_type


def get_loss(downstream_task, det_weight, p_weight, s_weight, loss_type="bce", focal_loss_gamma=1):
    if downstream_task == "det_ppk_spk_dis_emg":
        return [BCELoss(),BCELoss(),BCELoss(), HuberLoss(), HuberLoss()]
    elif downstream_task == "det_ppk_spk": # dpk
        return [BCELoss(),BCELoss(),BCELoss()]
    else: # dis or emg
        return HuberLoss()


class DPK_decoder(nn.Module):
    def __init__(self, encoder,downstream_head,decoder_size,args):
        super(DPK_decoder, self).__init__()
        self.encoder = encoder
        self.decoder = backbone_ablation.Decoder_baseline_llama(
            encoder_dim=encoder.backbone.d_model,
            decoder_size=decoder_size,
            args = args
        )
        
        self.downstream_head = downstream_head
        
    def __getitem__(self, key):
        if key == 0:
            return self.encoder
        elif key == 1:
            return self.decoder
        elif key == 2:
            return self.downstream_head
    
    def forward(self, x):
        h = self.encoder(x) # h:[bs,patch_num,feature_dim]，ori:[bs,feature_dim,patch_num]
        y = self.decoder(h,ids_restore=None).transpose(1,2) # y: [bs,feature_dim,patch_num]
        return self.downstream_head(y) # [bs,c,seq_len]
    
    
class DPK_splitPS(nn.Module):
    def __init__(self, encoder,feature_dim,in_samples,args):
        super(DPK_splitPS, self).__init__()
        self.encoder = encoder
        if args.hps != '': 
            downstream_head_class = head.MuHeadDetectionPicking
        else:
            downstream_head_class = head.HeadDetectionPicking
        
        heads = ['Pn_index','Pg_index'] + ['Sn_index','Sg_index'] + ['det']

        self.downstream_heads = nn.ModuleDict(
            {head:downstream_head_class(feature_channels=feature_dim, output_length=in_samples,out_channels=1) for head in heads}
        )
        
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder.to(*args, **kwargs)
        for head in self.downstream_heads.values():
            head.to(*args, **kwargs)
        return self
    
    def forward(self, x,ppks_type,spks_type):
        h = self.encoder(x)
        output_det = self.downstream_heads['det'](h)
        
        # 创建output_p和output_s为0
        output_p = torch.zeros_like(output_det)
        output_s = torch.zeros_like(output_det)
        for type_id in [0,2]:
            # 1. 从ppks_type获取索引
            index = torch.where(ppks_type.flatten() == type_id)
            # 2. 从h中获取对应的特征
            h_index = h[index]
            # 3. 送入对应的head中
            output_p[index] = self.downstream_heads[get_id_type[type_id]](h_index)
            
        for type_id in [1,3]:
            # 1. 从ppks_type获取索引
            index = torch.where(spks_type.flatten() == type_id)
            # 2. 从h中获取对应的特征
            h_index = h[index]
            # 3. 送入对应的head中
            output_s[index] = self.downstream_heads[get_id_type[type_id]](h_index)
        
        output = torch.cat([output_p,output_s,output_det],dim=1)
        
        return output


def create_model(model_name, downstream_tasks, in_samples, encoder_size, eval_type, pool_type, args):
    if model_name == 'llama':
        if args.dpk_head == 'vit_adapter':
            encoder = vit_adapter.ViTAdapter(
                encoder_size, input_length=in_samples, args=args,
            )
        elif args.dpk_head == 'vit_adapter_decoder_new':
            encoder = vit_adapter.ViTAdapter(
                encoder_size, input_length=in_samples, args=args, out_x=True
            )
        else:
            raise NotImplementedError
    elif model_name == "PatchTST":
        encoder = PatchTST_samll(input_length=in_samples)
    elif model_name == 'jepa':
        raise NotImplementedError
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    feature_dim = encoder.backbone.d_model
    if model_name != 'PatchTST' and model_name != 'PatchTST_speed':
        encoder.backbone.set_mask(mask_ratio=0, mask_way=None)

    downstream_head = head.EncoderFeatures(                    
            encoder_dim=encoder.backbone.d_model,                               
            args=args                                                           
        ) 
    
    if eval_type == 'linear_probe':
        def freeze_encoder(encoder):
            """
            Freeze the parameters of the encoder.
            """
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
#        if not args.dpk_head.startswith("vit_adapter"):
#            freeze_encoder(encoder)
        freeze_encoder(encoder)
    elif eval_type == 'partial_finetune':
        # currently not support head_v4
        def freeze_encoder_layers(encoder, num_layers_to_freeze):
            """
            Freeze the specific layers of the encoder.
            """
            encoder.eval()
            assert num_layers_to_freeze <= 24, f"The number of layers to freeze should be less than 24, but got {num_layers_to_freeze}"
            # freeze stem layer
            for param in encoder.backbone.W_p.parameters():
                param.requires_grad = False
            # freeze the first n layers
            for i, layer in enumerate(encoder.backbone.encoder.layers):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        freeze_encoder_layers(encoder, args.freeze_layers)
    elif eval_type == 'finetune':
        pass
    else:
        raise NotImplementedError

    model_list = []
    model_list.append(encoder)
    
    if pool_type == 'attentive':
        encoder.head_type = 'feature'
        attentive_pooler = Attentive(embed_dim=feature_dim)
        model_list.append(attentive_pooler)
    elif pool_type == 'avg':
        encoder.head_type = 'feature'
        avg_pooler = AVGPool()
        model_list.append(avg_pooler)
    elif pool_type == 'cls':
        encoder.head_type = 'cls'
    elif pool_type == 'none':
        pass
    else:
        raise NotImplementedError
    
    model_list.append(downstream_head)
    model = nn.Sequential(*model_list)
    print(f"[{eval_type}/{pool_type}]Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model

def get_labels_tasks(downstream_task):
    """
    downstream task : str
        "dis": seismic epicenter distance regression
        "dpk": seismic detection and picking
        "mag_full": seismic magnitude regression using full waveforms
        "mag_P_only": seismic magnitude regression using P-wave only
        "baz": seismic back-azimuth regression
        "fmp": seismic first motion polarity classification
        "dep": seismic depth regression
        "cls": seismic event classification

    label : list
        List of labels for the downstream task
        "dis": epicenter distance label
        "det": detection label
        "ppk": P-wave picking label
        "spk": S-wave picking label
        "mag_full": seismic magnitude regression using full waveforms label
        "mag_P_only": seismic magnitude regression using P-wave only label
        "baz": back-azimuth label
        "fmp": first motion polarity label
        "dep": depth label
        "cls": event classification label
    
    return : label_list, task_list
    """
    if downstream_task == "det_ppk_spk_dis_emg":
        return [["det", "ppk", "spk", "dis", "emg"]], ["det", "ppk", "spk", "dis", "emg"]
    elif downstream_task == "det_ppk_spk":
        return [["det", "ppk", "spk"]], ["det", "ppk", "spk"]
    elif downstream_task == "dis":
        return ["dis"], ["dis"]
    elif downstream_task == "emg":
        return ["emg"], ["emg"]
    elif downstream_task == "fmp":
        return ["fmp"], ["fmp"]
    else:
        return [downstream_task], [downstream_task]
#        raise NotImplementedError(f"Downstream task {downstream_task} not implemented")
