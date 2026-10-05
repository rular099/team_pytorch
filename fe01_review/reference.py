"""One common train-only final-label reference; no waveform or held-out fitting."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from fe01.data import split_catalog,strings
from fe01.spatial import fit_reference
from .provenance import require,read_json,write_json,sha256,fingerprint,provenance,seal,expand
from .runners import relocate,data_gate
from .checkpoint_inventory import inspect_checkpoint,load_cpu


def training_labels(cfg):
    catalog=split_catalog(cfg,'train');rows=[]
    require(catalog.event_id.nunique()==len(catalog),'Legacy reference fit requires globally unique train event IDs')
    require(len(catalog)==9084,'Frozen common training cohort differs')
    for path,events in catalog.groupby('hdf5_path'):
        with h5py.File(path,'r') as handle:
            for e in events.itertuples():
                g=handle['data'][str(e.event_id)]
                ids=strings(g['station_codes'][()]);network=strings(g['source_network'][()])
                pga=g['pga'][()].astype(np.float32);coords=g['coords'][()].astype(np.float32)
                for i in np.flatnonzero((network=='knt')&np.isfinite(pga)):
                    rows.append(dict(dataset_id=e.dataset_id,event_id=str(e.event_id),station_id=ids[i],split='train',truth=float(pga[i]),
                        latitude=float(coords[i,0]),longitude=float(coords[i,1]),magnitude=float(e.Magnitude),
                        depth=float(np.float32(e.DEPTH)),event_latitude=float(np.float32(e.Latitude)),event_longitude=float(np.float32(e.Longitude))))
    table=pd.DataFrame(rows)
    require(len(table)==146299,'Train final-label population differs from frozen normalization cohort')
    require(not table.duplicated(['dataset_id','event_id','station_id']).any(),'Train labels duplicated across decisions')
    require(np.isfinite(table.select_dtypes('number')).all().all(),'Invalid training reference metadata')
    norm=cfg['target_normalization']
    require(abs(table.truth.mean()-norm['mean'])<1e-10 and abs(table.truth.std(ddof=0)-norm['std'])<1e-10,'Train label normalization identity differs')
    return table


def actual_exposure(model):
    status=inspect_checkpoint(model['checkpoint'],model['epoch'],'fe01',model['config'],model['run_dir'])
    if status['status']!='IDENTITY_PASS':return [],dict(run_id=model['run_id'],status='UNKNOWN',reason=status['reason'])
    checkpoint=load_cpu(model['checkpoint']);journals=checkpoint.get('committed_journals',[]);del checkpoint
    require(journals,'Checkpoint has no committed journals for exposure identity')
    stations={};hashes={};cohort=set()
    for name in journals:
        path=Path(model['run_dir'])/name
        if not path.is_file():return [],dict(run_id=model['run_id'],status='UNKNOWN',reason='committed journal missing')
        hashes[name]=sha256(path)
        for line in path.open():
            r=json.loads(line);require(r['split']=='train' and r['epoch']<=model['epoch'],'Exposure journal outside committed train epochs')
            event=(r['dataset_id'],r['event_id']);cohort.add(event)
            for key in ('input_ids','query_ids'):
                for station in r[key]:stations.setdefault(station,{'input_ids':set(),'query_ids':set()})[key].add(event)
    rows=[dict(run_id=model['run_id'],station_id=s,input_training_events=len(v['input_ids']),query_training_events=len(v['query_ids']),
        station_seen_as_input=bool(v['input_ids']),station_seen_as_query=bool(v['query_ids']),identity_status='AUTHENTICATED_COMMITTED_JOURNALS') for s,v in stations.items()]
    return rows,dict(run_id=model['run_id'],status='PASS',checkpoint_sha256=status['checkpoint_sha256'],committed_journals_sha256=hashes,unique_train_events=len(cohort))


def build_reference(source_root,output,manifest=None):
    root=Path(source_root)/'formal__team_original_scratch__seed42'
    cfg=relocate(read_json(root/'resolved_config.json'));lock=read_json(root/'protocol.lock.json')
    data_identity=data_gate(cfg,lock);labels=training_labels(cfg)
    result=fit_reference(labels,penalty=.1,shrinkage=10)
    result.update(label_population_sha256=fingerprint(labels.sort_values(['dataset_id','event_id','station_id']).to_dict('records')),
        train_cohort_sha256=fingerprint(cfg['audited_cohorts']['train']),final_labels=len(labels),iterations=20,
        parent_lock_sha256=sha256(root/'protocol.lock.json'),fit_split='train',source_config_sha256=sha256(root/'resolved_config.json'))
    labels.to_csv(output/'training_final_labels.csv.gz',index=False)
    write_json(output/'train_only_reference.json',result)
    exposure=[];statuses=[]
    if manifest:
        for model in expand(read_json(manifest))['models']:
            if model['kind']=='fe01':
                rows,status=actual_exposure(model);exposure+=rows;statuses.append(status)
            else:statuses.append(dict(run_id=model['run_id'],status='UNKNOWN',reason='Legacy original sampler exposure not inferred from FE01 label cohort'))
    pd.DataFrame(exposure).to_csv(output/'actual_training_station_exposure.csv',index=False)
    write_json(output/'training_exposure_identity.json',statuses)
    write_json(output/'provenance.json',provenance(action='reference',data_identity=data_identity,reference_sha256=sha256(output/'train_only_reference.json'),
        interpretation='posthoc oracle; same reference for all systems; no validation/test fitting'))
    seal(output)
