#!/usr/bin/env python
"""Render FE01 configurations/manifests; never submits jobs or starts training."""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import fingerprint, load, write_json
from fe01.windows import CAPABILITIES

MAIN=['diting_pretrained_frozen','team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen']


def base_config():
    root=Path(__file__).resolve().parents[2]
    inherited=load(root/'pga_configs/transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json',expand=False)
    return dict(fe01=dict(enabled=True,version=2),seed=42,sampling_seed=42,
        model_family='team_original_scratch',native_n_samples=10000,
        query_chunk_size=15,query_independence_verified=False,
        pretrained_manifest='${FE01_WEIGHTS_ROOT}/pretrained_manifest.json',
        diting_config='${FE01_CODE_ROOT}/diting/config/diting_1200m_backbone_attnpool.yml',
        output_root='${FE01_OUTPUT_ROOT}', model_params=inherited['model_params'],
        data=dict(root='${DATA_ROOT}',split_manifest='${FE01_SPLIT_MANIFEST}',station_filter='knet',
                  sampling_rate=100,component_order='NEZ',units='m/s^2',
                  upstream_processing='legacy offline resampling/filtering; online causality not certified'),
        window=dict(protocol='native_prefix_v2',pre_p_seconds=5,strict_causal=True,rolling_trained=False),
        realtime=dict(bins=[[1,3],[3,5],[5,10],[10,20],[20,40],[40,90]],
                      probabilities=[.20,.20,.20,.15,.15,.10],draws_per_event=3,
                      fixed_times=[1,3,5,10,20,40,90],common_times=[1,3,5,10,20]),
        geometry=dict(random_station_counts=[1,3,5,8,12,16],train_random_probability=.5),
        training=dict(epochs=12,max_updates=None,global_batch=128,microbatch=8,lr=.001,
                      weight_decay=0,gradient_clip=5,scheduler='cosine_fixed_updates',
                      validation_frequency_epochs=1,workers=0,early_stopping=False,
                      res_comps=inherited['training_params']['res_comps'],
                      res_weight=inherited['training_params']['res_weight'],
                      distribution_mean_loss=inherited['training_params']['distribution_mean_loss']),
        target_normalization=inherited['training_params']['pga_target_normalization'],
        selection=dict(rule='equal protocols / equal common fixed times / noninput MAE; tie earlier epoch',
                       threshold_log10_pga=-1.2),
        spatial=dict(enabled=False,manifest='${FE01_SPATIAL_MANIFEST}',buffer_km=20),
        transfer_model_path=None,load_model_path=None)


def render(destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    base=base_config()
    write_json(destination/'base.json',base)
    rows=[]
    for stage in ('pilot','formal','spatial'):
        families=MAIN if stage!='formal' else MAIN+['amplitude_only','coords_only','diting_random_frozen']
        for family in families:
            for seed in ([42] if stage=='pilot' else [42,43,44]):
                cfg=copy.deepcopy(base)
                cfg.update(seed=seed,model_family=family,native_n_samples=CAPABILITIES[family].native_n_samples,
                           run_id=f'{stage}__{family}__seed{seed}',stage=stage)
                # Sampling stream is paired across optimization seeds and encoders.
                if stage=='pilot':
                    cfg['training'].update(max_updates=4,epochs=1,global_batch=16,microbatch=1)
                    cfg['audit_limits']=dict(train_events=16,val_events=8)
                if stage=='spatial':
                    cfg['spatial']['enabled']=True
                path=destination/stage/(cfg['run_id']+'.json')
                write_json(path,cfg)
                rows.append(dict(stage=stage,model_family=family,seed=seed,run_id=cfg['run_id'],
                                 config=str(path),main=family in MAIN,protocol_sha256=fingerprint(cfg)))
    write_json(destination/'native_rolling_template.json',dict(extends='base.json',
        window=dict(protocol='native_rolling_v2',rolling_trained=True),
        note='Explicit follow-up only; not included in default run matrices'))
    header=['stage','model_family','seed','run_id','config','main','protocol_sha256']
    (destination/'runs.tsv').write_text('\t'.join(header)+'\n'+''.join('\t'.join(str(r[k]) for k in header)+'\n' for r in rows))
    write_json(destination/'capability_manifest.json',{f:c.manifest() for f,c in CAPABILITIES.items()})
    print(f'Rendered {len(rows)} configurations. Formal primary indices: 0..11. No jobs submitted.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='configs/fe01')
    args=parser.parse_args()
    render(args.output)

if __name__=='__main__':
    main()
