from  .backbone_abla_models_maepretrain import PatchTST_base
from .backbone_abla_models_maepretrain import PatchTSTDecoder_base


def get_encoder_size_dict(width=256, depth=4):
    encoder_size_dict = {
        'n_layers':depth, 
        "d_model":width, 
        "n_heads":width//64, 
        "d_ff":width*4,
    }
    return encoder_size_dict

def get_decoder_size_dict(width=256, depth=4):
    decoder_size_dict = {
        'decoder_dim':width, 
        "decoder_depth":depth, 
        "decoder_num_heads":width//32, 
        "mlp_ratio":4,
    }
    return decoder_size_dict

decoder_base = {
    "decoder_dim":256,
    "decoder_depth":8,
    "decoder_num_heads":8,
    "mlp_ratio":4,
}

encoder_tiny = {
    'n_layers':12, 
    "d_model":192, 
    "n_heads":6, 
    "d_ff":768,
}
encoder_base = {
    'n_layers':12, 
    "d_model":768, 
    "n_heads":12, 
    "d_ff":3072,
}
encoder_large = {
    'n_layers':22, 
    "d_model":1024, 
    "n_heads":16, 
    "d_ff":4096,
}
encoder_giant = {
    'n_layers':22, 
    "d_model":2048, 
    "n_heads":32, 
    "d_ff":8192,
}

encoder_9m = {
    'n_layers':8, 
    "d_model":320, 
    "n_heads":5, 
    "d_ff":1280,
}
encoder_30m = {
    'n_layers':12, 
    "d_model":512, 
    "n_heads":8, 
    "d_ff":2048,
}
encoder_100m = {
    'n_layers':16, 
    "d_model":768, 
    "n_heads":12, 
    "d_ff":3072,
}
encoder_300m = {
    'n_layers':20, 
    "d_model":1024, 
    "n_heads":16, 
    "d_ff":4096,
}
encoder_1b = {
    'n_layers':24, 
    "d_model":1536, 
    "n_heads":24, 
    "d_ff":6144,
}

decoder_size_depth1_width144 = {
    "decoder_dim":144,
    "decoder_depth":1,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth2_width144 = {
    "decoder_dim":144,
    "decoder_depth":2,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth4_width144 = {
    "decoder_dim":144,
    "decoder_depth":4,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth8_width144 = {
    "decoder_dim":144,
    "decoder_depth":8,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth12_width144 = {
    "decoder_dim":144,
    "decoder_depth":12,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth8_width96 = {
    "decoder_dim":96,
    "decoder_depth":8,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth8_width192 = {
    "decoder_dim":192,
    "decoder_depth":8,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth8_width384 = {
    "decoder_dim":384,
    "decoder_depth":8,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}
decoder_size_depth8_width512 = {
    "decoder_dim":512,
    "decoder_depth":8,
    "decoder_num_heads":6,
    "mlp_ratio":4,
}

decoder_size_depth8_width512_base = {
    "decoder_dim":512,
    "decoder_depth":8,
    "decoder_num_heads":16,
    "mlp_ratio":4,
}

# add
decoder_size_depth1_width96 = {
    "decoder_dim":96,
    "decoder_depth":1,
    "decoder_num_heads":16,
    "mlp_ratio":4,
}
decoder_size_depth1_width192 = {
    "decoder_dim":192,
    "decoder_depth":1,
    "decoder_num_heads":16,
    "mlp_ratio":4,
}
decoder_size_depth4_width96 = {
    "decoder_dim":96,
    "decoder_depth":4,
    "decoder_num_heads":16,
    "mlp_ratio":4,
}
decoder_size_depth4_width192 = {
    "decoder_dim":192,
    "decoder_depth":4,
    "decoder_num_heads":16,
    "mlp_ratio":4,
}

# llama
def Encoder_baseline_llama(
        encoder_size,
        input_length=10000,
        c_in=3,
        args=None):
    patch_size = args.patch_size
    assert(input_length % patch_size == 0)
    num_patch = input_length // patch_size

    return PatchTST_base(
        c_in=c_in,
        num_patch=num_patch,
        patch_len=patch_size,
        **encoder_size,
        # norm
        pre_norm=True,
        norm=args.norm_layer, # 'rmsnorm'
        xattn=args.xattn,
        config=args,
    )
    
def Decoder_baseline_llama(
        encoder_dim,
        decoder_size,
        input_length=10000,
        c_in=3,
        args=None):
    patch_size = args.patch_size
    assert(input_length % patch_size == 0)

    return PatchTSTDecoder_base(
        c_in=c_in,
        patch_size=patch_size,
        encoder_dim=encoder_dim,
        **decoder_size,
        # norm
        norm=args.norm_layer,
        pre_norm=True,
        xattn=args.xattn,
        config=args,
    )

# baseline
def Encoder_baseline(encoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False
                )
    
def Decoder_baseline(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False
            )
# baseline_resAttention
def Encoder_baseline_resAttention(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=True
                )
    
def Decoder_baseline_resAttention(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=True
            )
    
# baseline_gelu
def Encoder_baseline_relu(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    act='relu'
                )
    
def Decoder_baseline_relu(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                act = 'relu'
            )
    
# baseline_postnorm
def Encoder_baseline_postnorm(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=False,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    # postnorm
                )
    
def Decoder_baseline_postnorm(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=False,
                # res-attention
                res_attention=False,
                # postnorm
            )
    
# baseline_qkvUnmerge
def Encoder_baseline_qkvUnmerge(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    # qkvUnmerge
                    qkvUnmerge=True
                )
    
def Decoder_baseline_qkvUnmerge(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                # qkvUnmerge
                qkvUnmerge=True
            )
    
# bias
def Encoder_baseline_bias(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    # attn&ff:bias
                    bias=False,
                )
    
def Decoder_baseline_bias(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                # attn&ff:bias
                bias=False,
            )

# learnPe
def Encoder_baseline_learnPe(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=True,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                )
    
def Decoder_baseline_learnPe(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                learn_pe=True
            )
    
# clsWithPos
def Encoder_baseline_clsWithPos(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    clsWithPos=True,
                )
    
def Decoder_baseline_clsWithPos(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                clsWithPos=True,
            )
    
# rope
def Encoder_baseline_rope(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    rope=True
                )
    
def Decoder_baseline_rope(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                rope=True
            )
# rmsnorm
def Encoder_baseline_rmsnorm(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=False
                )
    
def Decoder_baseline_rmsnorm(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=True,
                # res-attention
                res_attention=False
            )
    
# swiglu
def Encoder_baseline_swiglu(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='LayerNorm',
                    # res-attention
                    res_attention=False,
                    swiglu=True
                )
    
def Decoder_baseline_swiglu(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='LayerNorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                swiglu=True
            )
    
# batchnorm
def Encoder_baseline_batchnorm(encoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='batchnorm',
                    # res-attention
                    res_attention=False
                )
    
def Decoder_baseline_batchnorm(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='batchnorm',
                pre_norm=True,
                # res-attention
                res_attention=False
            )
    
# llama as new_baseline
# llama_bias
def Encoder_llama_bias(
        encoder_size,
        input_length=10000,
        c_in=3,
        args=None):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(
        c_in=c_in,
        num_patch=num_patch,**encoder_size,
        # position embedding
        learn_pe=False,
        pe='sincos',
        # norm
        pre_norm=True,
        norm=args.norm_layer,
        # res-attention
        res_attention=False,
        rope=True,
        swiglu=True,
        # attn&ff:bias
        bias=False,
        xattn=args.xattn,
    )
    
def Decoder_llama_bias(
        encoder_dim,
        decoder_size,
        input_length=10000,
        c_in=3,
        args=None):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(
        num_patches=num_patch,
        c_in=c_in,
        patch_size=pacth_len,
        encoder_dim=encoder_dim,
        **decoder_size,
        # norm
        norm=args.norm_layer,
        pre_norm=True,
        # res-attention
        res_attention=False,
        rope=True,
        swiglu=True,
        # attn&ff:bias
        bias=False,
        xattn=args.xattn,
    )

def Encoder_llama_learnpe(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=True,
                    pe='sincos',
                    rope=False,
                    # norm
                    pre_norm=True,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=False,
                    swiglu=True,
                )
    
def Decoder_llama_learnpe(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                swiglu=True,
                learn_pe=True,
            )
    
def Encoder_llama_sincospe(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=False,
                    rope=False,
                    swiglu=True,
                )
    
def Decoder_llama_sincospe(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                rope=False,
                swiglu=True,
            )

def Encoder_llama_postnorm(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=False,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=False,
                    rope=True,
                    swiglu=True,
                )
    
def Decoder_llama_postnorm(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=False,
                # res-attention
                res_attention=False,
                rope=True,
                swiglu=True,
            )
    
# llama
def Encoder_llama_layernorm(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='layernorm',
                    # res-attention
                    res_attention=False,
                    rope=True,
                    swiglu=True,
                )
    
def Decoder_llama_layernorm(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='layernorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                rope=True,
                swiglu=True,
            )
    
# llama
def Encoder_llama_batchnorm(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='batchnorm',
                    # res-attention
                    res_attention=False,
                    rope=True,
                    swiglu=True,
                )
    
def Decoder_llama_batchnorm(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='batchnorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                rope=True,
                swiglu=True,
            )
    
# llama
def Encoder_llama_resattention(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=True,
                    rope=True,
                    swiglu=True,
                )
    
def Decoder_llama_resattention(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=True,
                # res-attention
                res_attention=True,
                rope=True,
                swiglu=True,
            )
    
# llama
def Encoder_llama_relu(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=False,
                    rope=True,
                    act='relu',
                )
    
def Decoder_llama_relu(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                rope=True,
                act = 'relu'
            )
    
    
# llama
def Encoder_llama_gelu(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='rmsnorm',
                    # res-attention
                    res_attention=False,
                    rope=True,
                    act='gelu',
                )
    
def Decoder_llama_gelu(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='rmsnorm',
                pre_norm=True,
                # res-attention
                res_attention=False,
                rope=True,
                act = 'gelu'
            )
    
# llama
def Encoder_llama_layernorm_resattn(encoder_size,
                    input_length=10000,c_in=3):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTST_base(c_in=c_in,num_patch=num_patch,**encoder_size,
                    # position embedding
                    learn_pe=False,
                    pe='sincos',
                    # norm
                    pre_norm=True,
                    norm='layernorm',
                    # res-attention
                    res_attention=True,
                    rope=True,
                    swiglu=True,
                )
    
def Decoder_llama_layernorm_resattn(encoder_dim,decoder_size,
                    input_length=10000,c_in=3,):
    stride = 50
    pacth_len = 50
    assert(input_length % stride == 0)
    num_patch = input_length // stride

    return PatchTSTDecoder_base(num_patches=num_patch,c_in=c_in,patch_size=pacth_len,encoder_dim=encoder_dim,**decoder_size,
                # norm
                norm='layernorm',
                pre_norm=True,
                # res-attention
                res_attention=True,
                rope=True,
                swiglu=True,
            )