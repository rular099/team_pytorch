import json
from pathlib import Path
import numpy as np
import pandas as pd
from tests.test_fe01_review_runners import fixture
from fe01_review.requests import lock_new_requests,read_frame
from fe01_review.provenance import write_json,fingerprint
from fe01_review.runners import pack_inputs


def test_shared_physical_prefix_identity_across_native_shapes(tmp_path):
    cfg,event,requests=fixture(tmp_path)
    a,one=pack_inputs(event,requests.iloc[0],cfg,10000,queries=requests.station_id[:2].tolist())
    b,two=pack_inputs(event,requests.iloc[0],cfg,3001,queries=requests.station_id[:2].tolist())
    assert a[0].shape!=b[0].shape
    assert one['physical_received_history_sha256']==two['physical_received_history_sha256']
    assert one['model_input_sha256']!=two['model_input_sha256']


def test_cpu_request_plan_locks_real_elevations_duplicate_draws_and_fixed_s0(tmp_path):
    cfg,event,requests=fixture(tmp_path);plan=tmp_path/'plan';plan.mkdir();out=tmp_path/'requests';out.mkdir()
    requests.to_csv(plan/'fixed_requests.csv.gz',index=False)
    mapping=pd.DataFrame([dict(dataset_id=event['dataset_id'],event_id=event['event_id'],elapsed_time=t,draw=i,snapshot_id=fingerprint(t)) for i,t in enumerate([2.34,2.34,3.45])])
    mapping.to_csv(plan/'random_draw_mapping.csv',index=False);mapping.to_csv(plan/'random_time_draws.csv',index=False)
    write_json(plan/'case_manifest.json',dict(cases=[dict(dataset_id=event['dataset_id'],event_id=event['event_id'])]))
    lock_new_requests(cfg,plan,out)
    random=read_frame(out/'random_requests.csv.gz');assert random.elapsed_time.nunique()==2
    assert not random.duplicated(['dataset_id','event_id','elapsed_time','geometry_protocol','station_id']).any()
    metadata=read_frame(out/'fixed_station_metadata.csv.gz');assert 'query_elevation' in metadata
    replay=read_frame(out/'replay_requests.csv.gz');assert set(replay.elapsed_time)==set(range(1,21))
    fixed=replay[replay.input_mode=='fixed_s0']
    assert fixed.groupby('geometry_protocol').input_ids.nunique().eq(1).all()
    assert fixed.target_role.ne('observed_input').all()
    assert replay.groupby(['geometry_protocol','input_mode']).apply(lambda g:g.groupby('elapsed_time').station_id.apply(frozenset).nunique(),include_groups=False).eq(1).all()
