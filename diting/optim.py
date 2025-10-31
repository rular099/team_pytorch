import json
import re

from distributed import is_master


def get_num_layer_for_transformer(param_name, num_max_layer):
    layer_0 = {
        "W_p",
        "patch_embed",
        "conv1",
        "pos_embed", 
        "positional_embedding",
        "cls_token", 
        "mask_token", 
    }

    if any(l in param_name for l in layer_0):
        return 0

    block_regex = re.compile(r"blocks\.([0-9]+)\.")
    match_block = block_regex.search(param_name)

    layer_regex = re.compile(r"encoder.layers\.([0-9]+)\.") 
    match_layer = layer_regex.search(param_name)
    if match_block is not None:
        return int(match_block.group(1)) + 1
    elif match_layer is not None:
        return int(match_layer.group(1)) + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))
    

def get_layer_id_for_transformer(param_name):
    layer_backbone = {
        "0.backbone",
    }

    if any(l in param_name for l in layer_backbone):
        return 0
    else:
        return 1

    
class BlockDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_layer_id_for_transformer(var_name)
    

def get_parameters(args, model, assigner):
    skip = set(['0.level_embed'])

    lr = args.lr
    weight_decay = args.weight_decay
    if hasattr(model, 'no_weight_decay'):
        skip = set.union(skip, model.no_weight_decay())

    get_num_layer  = assigner.get_layer_id if assigner is not None else None
    get_layer_scale = assigner.get_scale if assigner is not None else None

    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
            
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            if layer_id is not None:
                group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    
    if is_master(args):
        print(f"Parameters: ")
        print(f"Skip weight decay name: {skip}")
        print(f"Num of parameters group: {len(parameter_group_vars.values())}")
        print(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")

    return list(parameter_group_vars.values())


def get_assigner(args, model):
    ld = args.layer_decay if hasattr(args, 'layer_decay') else 1.0
    
    if ld < 1.0:
        num_layers = model[0].get_num_layers()
        assigner = LayerDecayValueAssigner(list(ld ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    # backbone has a different lr with task head, default is [1, 1]
    if args.block_decay[0] != args.block_decay[1]:
        assigner = BlockDecayValueAssigner(args.block_decay)

    if assigner is not None:
        print("Assigned layer decay values = %s" % str(assigner.values))
    return assigner


def get_all_parameters(args, model):
    assigner = get_assigner(args, model)
    parameters = get_parameters(args, model, assigner)
    return parameters


def get_all_parameters_mup_wd(model, decoupled_wd=False, skip='mask_token', lr_scale=1., **kwargs):
    args = kwargs['args']
    lr = kwargs['lr']
    weight_decay = kwargs.get('weight_decay', 0.)
    parameter_group_names = {}
    parameter_group_vars = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert hasattr(p, 'infshape'), (
            f'A parameter with shape {p.shape} does not have `infshape` attribute. '
            'Did you forget to call `mup.set_base_shapes` on the model?')
        
        if p.infshape.ninf() == 2:
            group_name = "matrix_like_p_decay"
            width_mult = p.infshape.width_mult()
            
            this_lr = lr / width_mult
            this_weight_decay = weight_decay
            if not decoupled_wd:
                this_weight_decay *= width_mult
        elif p.infshape.ninf() > 2:
            raise NotImplementedError('more than 2 inf dimensions')
        else:
            this_lr = lr
            if p.ndim <= 1 or name.endswith(".bias") or skip in name:
                group_name = "vector_like_p_no_decay"
                this_weight_decay = 0.
            else:
                group_name = "vector_like_p_decay"
                this_weight_decay = weight_decay
            
        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": lr_scale,
                "lr": this_lr,
                "base_lr": this_lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": lr_scale,
                "lr": this_lr,
                "base_lr": this_lr,
            }
        parameter_group_vars[group_name]["params"].append(p)
        parameter_group_names[group_name]["params"].append(name)
    
    if is_master(args):
        print(f"Parameters: ")
        print(f"Skip weight decay name: {skip}")
        print(f"Num of parameters group: {len(parameter_group_vars.values())}")
        print(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    
    return list(parameter_group_vars.values())