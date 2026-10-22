import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from fe01_a01_compare.covariates import physical_event,hydrate
from fe01_a01_compare.crossings import extract
from fe01_a01_compare.io import csv_read,read_json
from fe01_a01_compare.contracts import contract


def tiny_event(tmp_path):
    path=tmp_path/'japan_2020.hdf5'
    rng=np.random.default_rng(3)
    wave=(rng.normal(size=(6,3500,3))*.001).astype(np.float32)
    wave[:,800:1400]*=50
    with h5py.File(path,'w') as f:
        f.create_dataset('metadata/sampling_rate',data=100)
        g=f.create_group('data/20200101000000')
        for name,data in [('waveforms',wave),('source_network',np.array([b'knt']*6)),
                          ('station_codes',np.array([f'S{i:02d}'.encode() for i in range(6)])),
                          ('coords',np.array([[35+i//3*.1,139+i%3*.1,.02] for i in range(6)],np.float32)),
                          ('p_picks',np.arange(6)*100+600),('record_start_sample',np.zeros(6,int)),
                          ('valid_n_samples',np.full(6,3500)),
                          ('pga',np.log10(np.linalg.norm(wave,axis=2).max(1)+1e-10)),
                          ('record_start_time_jst',np.array([b'2020-01-01 09:00:00']*6))]:
            g.create_dataset(name,data=data)
    row=pd.Series(dict(dataset_id=path.name,event_id='20200101000000',hdf5_path=str(path),Magnitude=4.2,DEPTH=10,
                       Latitude=35.,Longitude=139.,**{'Origin_Time(JST)':'2020-01-01 08:59:56'}))
    return row


def test_real_hdf_reader_shared_prefix_and_quality(tmp_path):
    row=tiny_event(tmp_path)
    e=physical_event(row,max_elapsed=1)
    assert e['waveform'].shape[-1]==701
    assert e['clock']['first_p_minus_origin_seconds']==10
    out=tmp_path/'cov';out.mkdir()
    hydrate(pd.DataFrame([row]),None,contract(),out)
    f=csv_read(out/'request_covariates.csv.gz')
    assert len(f)==6*20*2
    assert f.groupby(['geometry_protocol','elapsed_time']).station_id.nunique().eq(6).all()
    assert f.protocol.eq('fixed_query_replay_v4').all()
    assert f.input_count.max()<=6
    grids=read_json(out/'grid_manifest.json')
    assert 'km' in grids[0]['third_coordinate']
    assert f.origin_elapsed_seconds.notna().all()
    out2=tmp_path/'crossing';out2.mkdir()
    extract(pd.DataFrame([row]),[-1.2,0.],out2)
    truth=csv_read(out2/'crossing_truth.csv.gz')
    assert len(truth)==12
    assert set(truth.status).issubset({'not_exceeded','exact_within_record'})
