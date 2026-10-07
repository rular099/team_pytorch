"""Small synthetic train/validation fixtures; no held-out production access."""
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch

from scripts.fe01.prepare_runs import base_config
from fe01.windows import CAPABILITIES
from fe01.data import read_event,prepare_sample,split_catalog
from fe01.engine import inputs_to
from fe01_a01 import ON

torch.set_num_threads(1)


def config(tmp,family='team_original_scratch',mode=ON):
    cfg=base_config()
    cfg.update(model_family=family,native_n_samples=CAPABILITIES[family].native_n_samples,
        absolute_amplitude_mode=mode,output_root=str(tmp),run_id='synthetic_a01',
        pretrained_manifest=os.environ.get('FE01_A01_TEST_WEIGHTS',str(Path(tmp)/'no_weights.json')))
    cfg['model_params'].update(waveform_model_dims=[20,20,20],output_mlp_dims=[8],output_location_dims=[8],
        mad_params=dict(n_heads=2,att_dropout=0.),ffn_params=dict(hidden_dim=40),transformer_layers=1,
        max_stations=8,n_pga_targets=6,readout_n_heads=2,readout_ffn_hidden_dim=40,pga_readout_layers=4)
    cfg['spatial']['manifest']=str(Path(tmp)/'spatial.csv')
    cfg['data'].update(root=str(Path(tmp)/'data'),split_manifest=str(Path(tmp)/'split_events.csv'))
    return cfg


def archive(tmp):
    path=Path(tmp)/'data/2020/japan_2020.hdf5';path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261007);rows=[]
    with h5py.File(path,'w') as f:
        f.create_dataset('metadata/sampling_rate',data=100)
        for i in range(4):
            event_id=f'2020010{i+1}000000';g=f.create_group('data/'+event_id)
            n,length=12,11500
            wave=rng.normal(0,.01,(n,length,3)).astype(np.float32);wave[:,550:565]=0
            coords=np.column_stack([35+np.arange(n)*.06,139+np.arange(n)*.08,np.zeros(n)]).astype(np.float32)
            fields=dict(waveforms=wave,coords=coords,p_picks=500+np.arange(n)*40,
                pga=np.linspace(-2,-.5,n,dtype=np.float32),record_start_sample=np.zeros(n,dtype=np.int64),
                valid_n_samples=np.full(n,length),station_codes=np.array([f'S{j:02d}' for j in range(n)],dtype='S'),
                source_network=np.array(['knt']*n,dtype='S'),
                record_start_time_jst=np.array(['2020-01-01T00:00:00+09:00']*n,dtype='S'))
            for k,v in fields.items():g.create_dataset(k,data=v)
            rows.append(dict(dataset_index=20,EVENT=event_id,split='train' if i<2 else 'val',
                source_data_path=str(path),Latitude=35,Longitude=139,DEPTH=10,Magnitude=4+i*.2,
                **{'Origin_Time(JST)':'2020-01-01T00:00:01+09:00'},n_station_rows=n))
    pd.DataFrame(rows).to_csv(Path(tmp)/'split_events.csv',index=False)
    return path


def sample(tmp,cfg,elapsed=3):
    archive(tmp)
    row=split_catalog(cfg,'val').iloc[0]
    event=read_event(row,cfg,elapsed,full_for_audit=True)
    result=prepare_sample(event,cfg,elapsed,'normal',queries=np.array([8,9,10,11]))
    return event,result,inputs_to(result,'cpu')
