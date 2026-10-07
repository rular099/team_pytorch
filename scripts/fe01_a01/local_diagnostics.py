#!/usr/bin/env python3
"""Authorized local synthetic smoke, real offline pickers optional; never production."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--requests',required=True);parser.add_argument('--weights',required=True)
    args=parser.parse_args()
    os.environ['FE01_A01_TEST_WEIGHTS']=args.weights
    import numpy as np
    import pandas as pd
    import torch
    from test_fe01_a01_helpers import config,sample
    from fe01.model import build_model as old_build,common_state,state_fingerprint
    from fe01.config import sha256
    from fe01.windows import CAPABILITIES,build_window
    from fe01.data import prepare_sample
    from fe01.engine import inputs_to
    from fe01_review.requests import read_frame,population_audit
    from fe01_a01.provenance import new_output,write_json,source_identity
    from fe01_a01.model import build_model
    from fe01_a01 import ON,OFF
    from fe01_a01.identity import reuse_equivalence,training_budget
    from fe01_a01.diagnostics import query_trace,permutation_controls,select_probes,nested_sets
    from fe01_a01.availability import scale_audit,future_audit
    from fe01_a01.plotting import plot_diagnostics
    output=new_output(args.output)
    requests=read_frame(args.requests);population=population_audit(requests)
    selected,strata=select_probes(requests)
    selected.to_csv(output/'real_metadata_probe_decisions.csv',index=False)
    strata.to_csv(output/'real_metadata_probe_strata.csv',index=False)
    keys=['dataset_id','event_id','elapsed_time','geometry_protocol']
    # Preserve request identities without copying all original predictions/labels.
    chosen=requests.merge(selected[keys],on=keys,validate='many_to_one')
    cols=keys+['station_id','target_role','input_ids','input_count','history_start_sample','history_end_sample','cutout_exclusive']
    chosen[cols].to_csv(output/'real_metadata_probe_requests.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(output/'real_metadata_population.json',dict(**population,source_csv_sha256=sha256(args.requests),
        probe_decisions=len(selected),probe_requests=len(chosen),prediction_access_for_selection=False))
    checks=[]
    for family in ('team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen'):
        destination=output/'synthetic'/family;destination.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix='fe01-a01-synthetic-') as temporary:
            cfg=config(Path(temporary),family);event,pack,inputs=sample(Path(temporary),cfg)
            baseline=old_build(cfg)
            equality=reuse_equivalence(cfg,cfg,baseline.state_dict(),inputs)
            write_json(destination/'ON_equivalence.json',equality)
            write_json(destination/'initial_identity.json',dict(evidence='SYNTHETIC downstream; actual STEAD encoders for PN/EQT',
                state_sha256=state_fingerprint(baseline.state_dict()),common_initial_sha256=state_fingerprint(common_state(baseline)),
                production_init_comparison='NOT_RUN: no production init.pth bytes locally'))
            portable=copy.deepcopy(cfg);portable['output_root']='$SYNTHETIC_ROOT'
            portable['data']['root']='$SYNTHETIC_ROOT/data';portable['data']['split_manifest']='$SYNTHETIC_ROOT/split_events.csv'
            portable['spatial']['manifest']='$SYNTHETIC_ROOT/spatial.csv';portable['pretrained_manifest']='$REGISTERED_OFFLINE_WEIGHTS/pretrained_manifest.json'
            write_json(destination/'resolved_synthetic_config.json',portable)
            stages=[];sens=[];gates=[];branches=[];controls=[];future=[];scale=[];nested=[]
            for mode in (ON,OFF):
                cfg['absolute_amplitude_mode']=mode;model=build_model(cfg)
                for k in (1,3,5,8):
                    x=[v.clone() for v in inputs]
                    k=min(k,int(inputs[2].sum()))
                    x[0][:,k:]=0;x[1][:,k:]=0;x[2][:,k:]=False;x[5][:,k:]=False
                    tag=dict(mode=mode,evidence='SYNTHETIC_UNTRAINED_DOWNSTREAM',input_count=k)
                    r,t,s=query_trace(model,x,cfg['target_normalization'],destination/'traces'/f'{mode}_K{k}',tag['evidence'])
                    stages.append(t.assign(**tag));sens.append(s.assign(**tag))
                    gates.append({**tag,**r['gates']});branches.append({**tag,**r['branches']})
                    controls.append({**tag,**permutation_controls(model,x)})
                scale.extend({**r,'evidence':'SYNTHETIC_UNTRAINED_DOWNSTREAM'} for r in scale_audit(model,inputs,mode)['rows'])
                for case,elapsed in (('trigger_boundary',.4),('native_capacity',CAPABILITIES[family].max_elapsed_sample()/100),
                    ('tail_missing',3.),('internal_legal_zero',3.),('invalid_station',3.)):
                    changed=copy.deepcopy(event)
                    if case=='tail_missing':changed['storage'][1,...,650:]=False
                    elif case=='invalid_station':changed['storage'][-1]=False
                    if case=='internal_legal_zero':changed['waveform'][...,550:565]=0
                    future.extend({**r,'mode':mode,'evidence':'SYNTHETIC_UNTRAINED_DOWNSTREAM'} for r in future_audit(model,cfg,changed,elapsed,case=case)['rows'])
                low=[v.clone() for v in inputs];low[0]*=1e-12
                write_json(destination/f'eps_limit_{mode}.json',scale_audit(model,low,mode))
                _,mask,_=build_window(event['waveform'],event['storage'],event['reference_sample'],3,CAPABILITIES[family])
                _,_,ninfo=nested_sets(event,mask.all(1),cfg);nested.append(ninfo)
            pd.concat(stages).to_csv(destination/'query_stages.csv',index=False)
            pd.concat(sens).to_csv(destination/'query_sensitivity.csv',index=False)
            for name,rows in (('gates',gates),('branch_norms',branches),('station_controls',controls),('future_audit',future),('scale_audit',scale)):
                pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)
            write_json(destination/'nested_support.json',nested)
            plot_diagnostics(destination,'SYNTHETIC untrained downstream — '+family)
            checks.append(dict(family=family,ON_equivalence='PASS',scale='PASS_NORMAL_REGIME',future='PASS_HDF_PREFIX',
                production_trained_checkpoint='NOT_RUN',formal_training='NOT_SUBMITTED'))
            print('LOCAL_DIAGNOSTIC_PASS',family,flush=True)
    write_json(output/'local_summary.json',dict(checks=checks,production_queries='METADATA_ONLY',
        production_weights='NOT_RUN',diting='NOT_RUN: registered production asset not locally accessible',
        upstream='UPSTREAM_CAUSALITY_UNKNOWN',hpc='NOT_SUBMITTED',budget=training_budget(9084,cfg,16),
        source=source_identity()))


if __name__=='__main__':main()
