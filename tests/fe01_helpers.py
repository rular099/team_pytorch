import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/fe01-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/fe01-xdg')
os.environ.setdefault('SEISBENCH_CACHE_ROOT','/tmp/fe01-seisbench-test')

import h5py
import numpy as np
import pandas as pd
import torch

from scripts.fe01.prepare_runs import base_config

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def config(tmp,family='team_original_scratch'):
    cfg=base_config()
    cfg.update(model_family=family,output_root=str(tmp),run_id='synthetic_test',
               pretrained_manifest=str(ROOT/'offline_weights/stead_v2/pretrained_manifest.json'))
    from fe01.windows import CAPABILITIES
    cfg['native_n_samples']=CAPABILITIES[family].native_n_samples
    cfg['spatial']['manifest']=str(Path(tmp)/'spatial.csv')
    cfg['model_params'].update(waveform_model_dims=[20,20,20],output_mlp_dims=[8],output_location_dims=[8],
        mad_params={'n_heads':2,'att_dropout':0.0},ffn_params={'hidden_dim':40},
        transformer_layers=1,max_stations=4,n_pga_targets=3,readout_n_heads=2,
        readout_ffn_hidden_dim=40,pga_readout_layers=1,event_readout_layers=1)
    cfg['data']['root']=str(Path(tmp)/'data')
    cfg['data']['split_manifest']=str(Path(tmp)/'split_events.csv')
    return cfg


def synthetic_archive(tmp,events=3,stations=6,length=11000):
    root=Path(tmp)/'data/2020';root.mkdir(parents=True,exist_ok=True)
    path=root/'japan_2020.hdf5'
    rng=np.random.default_rng(20)
    rows=[]
    with h5py.File(path,'w') as f:
        f.create_dataset('metadata/sampling_rate',data=100)
        for i in range(events):
            event_id=f'2020010{i+1}000000'
            g=f.create_group('data/'+event_id)
            wave=rng.normal(size=(stations,length,3)).astype(np.float32)*.01
            coords=np.column_stack([35+np.arange(stations)*.05,139+np.arange(stations)*.05,np.zeros(stations)]).astype(np.float32)
            picks=np.arange(stations)*200+500
            fields=dict(waveforms=wave,coords=coords,p_picks=picks,pga=np.linspace(-2,-.5,stations,dtype=np.float32),
                record_start_sample=np.zeros(stations,dtype=np.int64),valid_n_samples=np.full(stations,length),
                station_codes=np.asarray([f'S{j}' for j in range(stations)],dtype='S'),
                source_network=np.asarray(['knt']*stations,dtype='S'),
                record_start_time_jst=np.asarray(['2020-01-01T00:00:00+09:00']*stations,dtype='S'))
            for k,v in fields.items():
                g.create_dataset(k,data=v)
            rows.append(dict(dataset_index=20,EVENT=event_id,split=['train','val','test'][i%3],
                source_data_path=str(path),Latitude=35,Longitude=139,DEPTH=10,Magnitude=4+i*.2,
                **{'Origin_Time(JST)':'2020-01-01T00:00:01+09:00'},n_station_rows=stations))
    pd.DataFrame(rows).to_csv(Path(tmp)/'split_events.csv',index=False)
    return path
