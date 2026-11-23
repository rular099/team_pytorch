"""Focused synthetic contracts; none of these fixtures are Japan/HPC evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fe02_lite.config import resolve, VARIANTS, fingerprint, code_identity, write_json
from fe02_lite.model import LiteModel, visible_waveforms, parameter_audit, mdn
from fe02_lite.data import (catalog, read_event, prepare, metadata, audit_data, transform_coordinates)
from fe02_lite.artifacts import (state_digest, rng_state, restore_rng, read_table, TableWriter,save_diagnostic)
from fe02_lite.losses import loss_terms
from fe02_lite.evaluation import evaluate, export_diagnostics
from fe02_lite import engine, analysis
from fe01.metrics import score_rows, decode

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch=nn.Conv1d(3,4,50,stride=50)
        self.norm=nn.BatchNorm1d(4)
    def forward(self,x):
        return self.norm(self.patch(x)).repeat(1,448,1)


def make_model(cfg,device='cpu'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        model=LiteModel(SyntheticEncoder(),cfg['architecture'],cfg['seed'],encoder_batch=2)
    model.encoder_identity={'synthetic':True}
    return model.to(device)


@pytest.fixture
def cfg(tmp_path,monkeypatch):
    monkeypatch.setenv('FE02_LITE_OUTPUT_ROOT',str(tmp_path/'outputs'))
    monkeypatch.setenv('FE02_LITE_DATA_ROOT',str(tmp_path/'data'))
    monkeypatch.setenv('FE02_LITE_SPLIT_MANIFEST',str(tmp_path/'split.csv'))
    monkeypatch.setenv('FE02_LITE_SHARD_IDENTITY_MANIFEST','')
    result=resolve(ROOT/'configs/fe02_lite/knet.json')
    result['sampling'].update(max_stations=3,targets_per_instance=3,query_chunk_size=2,encoder_batch=2)
    result['diagnostics'].update(max_events=1,max_bytes=16*1024*1024)
    return result


@pytest.fixture
def dataset(cfg):
    path=Path(cfg['data']['root'])/'2001/japan_2001.hdf5'
    path.parent.mkdir(parents=True)
    rows=[]
    with h5py.File(path,'w') as handle:
        handle.create_dataset('metadata/sampling_rate',data=100)
        for e in range(4):
            event_id=f'200101010000{e:02d}'
            group=handle.create_group('data/'+event_id)
            networks=['knt']*3+['kik']*6
            layers=['single_surface']*3+['borehole','surface']*3
            suffixes=['']*3+['1','2']*3
            codes=['K0','K1','K2','I0','I0','I1','I1','I2','I2']
            rng=np.random.default_rng(e)
            wave=rng.normal(size=(9,12000,3)).astype(np.float32)*np.arange(1,10)[:,None,None]
            wave+=np.arange(12000,dtype=np.float32)[None,:,None]*.0001
            picks=np.array([500,900,8000,520,540,920,940,8020,8040],np.float64)
            starts=np.arange(9)*10
            labels=np.asarray([np.linalg.norm(wave[i,starts[i]:starts[i]+11000],axis=-1).max() for i in range(9)])
            values=dict(waveforms=wave,coords=np.column_stack((37+np.arange(9)*.01,140+np.arange(9)*.02,np.arange(9)*.005)),
                p_picks=picks,record_start_sample=starts,valid_n_samples=np.full(9,11000),
                pga=np.log10(labels+1e-10),pga_norm_resampled_mps2=labels,
                source_network=np.asarray(networks,'S'),sensor_class=np.asarray(layers,'S'),
                sensor_suffix=np.asarray(suffixes,'S'),station_codes=np.asarray(codes,'S'),
                component_valid=np.ones((9,3),bool),waveform_sample_valid=np.ones((9,12000,3),bool),
                record_start_time_jst=np.asarray([pd.Timestamp(1700000000+float(s)/100,unit='s',tz='UTC').tz_convert('Asia/Tokyo').isoformat() for s in starts],'S'))
            for key,value in values.items():group.create_dataset(key,data=value)
            # A different valid-interval hole for surface and borehole.
            group['waveform_sample_valid'][3,550:560,:]=False
            group['waveform_sample_valid'][4,570:590,:]=False
            rows.append(dict(EVENT=event_id,split='train' if e<2 else 'val',source_data_path=str(path),
                Magnitude=4+e*.1,Latitude=37.2,Longitude=140.3,DEPTH=12.0))
    pd.DataFrame(rows).to_csv(cfg['data']['split_manifest'],index=False)
    return path


def effective(cfg,tmp_path):
    lock=audit_data(cfg,tmp_path/'audit_data')
    result=copy.deepcopy(cfg);result.update(lock['effective_config'])
    return result,lock


def test_three_sensor_views_and_masks(cfg,dataset,tmp_path):
    selected={}
    norms=[]
    for view in ['knet','kiknet_borehole','kiknet_surface']:
        current=copy.deepcopy(cfg);current['dataset_view']=view
        row=catalog(current,'train').iloc[0]
        event=read_event(row,current,1)
        selected[view]=event
        with h5py.File(dataset,'r') as h:
            meta=metadata(h['data'][row.event_id],view)
            keep=meta['indices']
            np.testing.assert_array_equal(event['waveform'],h['data'][row.event_id]['waveforms'][keep,:event['cutout_exclusive'],:].astype(np.float32).transpose(0,2,1))
            np.testing.assert_allclose(event['pga'],h['data'][row.event_id]['pga'][keep],rtol=1e-6)
            assert len(set(event['ids']))==3
        current,lock=effective(current,tmp_path/view)
        norms.append(current['target_normalization']['mean'])
        assert lock['split_manifest']['rule'].startswith('builder EVENT')
        assert (tmp_path/view/'audit_data/view_records.csv.gz').is_file()
    assert len(set(norms))==3
    bore,surface=selected['kiknet_borehole'],selected['kiknet_surface']
    assert not bore['storage'][0,550:560].any()
    assert surface['storage'][0,550:560].all()
    assert not surface['storage'][0,570:590].any()
    np.testing.assert_array_equal(bore['physical_ids'],surface['physical_ids'])
    assert bore['reference_sample']!=surface['reference_sample']


def test_refuse_ambiguous_data_and_split_leak(cfg,dataset):
    with h5py.File(dataset,'r+') as h:
        g=h['data'][catalog(cfg,'train').iloc[0].event_id]
        g['sensor_suffix'][3]=b'2'
        with pytest.raises(ValueError,match='suffix'): metadata(g,'kiknet_borehole')
        g['sensor_suffix'][3]=b'1'
        g.create_dataset('p_pick_refined_aligned',data=g['p_picks'][()])
        g['p_pick_refined_aligned'][0]+=1
        with pytest.raises(ValueError,match='ambiguous timing'):metadata(g,'knet')
        del g['p_pick_refined_aligned']
        g['pga'][4]=8
        with pytest.raises(ValueError,match='pga is not'): metadata(g,'kiknet_surface')
        del g['waveforms']
        g.create_dataset('waveforms',shape=(9,10000,6),dtype='f4')
        with pytest.raises(ValueError,match='six-channel'):metadata(g,'knet')
    table=pd.read_csv(cfg['data']['split_manifest'],dtype={'EVENT':str})
    duplicate=table.iloc[[0]].copy();duplicate['split']='val'
    pd.concat([table,duplicate]).to_csv(cfg['data']['split_manifest'],index=False)
    with pytest.raises(ValueError,match='Repeated|crosses'):catalog(cfg,'train')


def test_audit_exclusions_suffix_zero_and_reader_future_boundary(cfg,dataset,tmp_path):
    row=catalog(cfg,'train').iloc[0]
    before=read_event(row,cfg,1)
    with h5py.File(dataset,'r+') as handle:
        group=handle['data'][row.event_id]
        group['sensor_suffix'][:3]=np.asarray(['0']*3,'S')
        group['waveforms'][:3,before['cutout_exclusive']:,:]=np.nan
    after=read_event(row,cfg,1)
    np.testing.assert_array_equal(before['waveform'],after['waveform'])
    # Missing all components excludes the event and still produces all sensor reasons.
    with h5py.File(dataset,'r+') as handle:
        handle['data'][row.event_id]['component_valid'][:3]=False
    effective_cfg,lock=effective(cfg,tmp_path)
    assert len(effective_cfg['audited_cohorts']['train'])==1
    assert lock['sensor_exclusion_reason_counts']['missing_component']==3
    assert len(lock['waveform_label_checks'])==3
    assert effective_cfg['target_normalization']['count']==3
    records=pd.read_csv(tmp_path/'audit_data/view_records.csv.gz')
    assert len(records)==12 and (~records.cohort_included).sum()==3


def test_audit_rejects_wrong_waveform_units(cfg,dataset,tmp_path):
    row=catalog(cfg,'train').iloc[0]
    with h5py.File(dataset,'r+') as handle:
        group=handle['data'][row.event_id]
        group['waveforms'][0]=group['waveforms'][0][()]*100
    with pytest.raises(ValueError,match='waveform/PGA magnitude mismatch'):
        audit_data(cfg,tmp_path/'bad_units')


def test_model_causality_amplitude_alignment_and_invariances(cfg):
    model=make_model(cfg).eval()
    raw=torch.randn(1,3,3,10000)
    mask=torch.zeros(1,3,10000,dtype=torch.bool);mask[:,:,:736]=True
    coords=torch.randn(1,3,3);targets=torch.randn(1,5,3)
    stations=torch.ones(1,3,dtype=torch.bool);valid=torch.ones(1,5,dtype=torch.bool)
    args=[raw,coords,stations,targets,valid,mask]
    before=rng_state()
    with torch.no_grad():
        out,tensors=model(*args,diagnostics=True)
        assert torch.equal(before['torch'],torch.get_rng_state())
        assert tensors['amplitude'].shape==(1,3,200,8)
        assert tensors['token_mask'][0,0,14] and not tensors['token_mask'][0,0,15]
        normalized,centered=visible_waveforms(raw,mask)
        blocks=centered[0].reshape(3,3,200,50).permute(0,2,1,3).flatten(-2)
        linear=torch.nn.functional.linear(blocks,model.amplitude.weight.flatten(1),model.amplitude.bias)
        expected=torch.nn.functional.gelu(linear)*tensors['token_mask'][0,:,:,None]
        torch.testing.assert_close(tensors['amplitude'][0],expected,atol=2e-6,rtol=2e-6)
        other=raw.clone();other[~mask[:,:,None].expand_as(raw)]=float('nan')
        changed=model(other,*args[1:])
        for a,b in zip(out,changed):torch.testing.assert_close(a,b,atol=0,rtol=0)
        norm_scaled,_=visible_waveforms(raw*3,mask)
        torch.testing.assert_close(normalized,norm_scaled,atol=1e-6,rtol=1e-6)
        _,scaled=model(raw*3,*args[1:],diagnostics=True)
        assert not torch.allclose(tensors['amplitude'],scaled['amplitude'])
        encoded=model.encode_observations(raw,coords,stations,mask)
        individual=torch.cat([model.read_targets(encoded,targets[:,i:i+1])[0] for i in range(5)],1)
        torch.testing.assert_close(out[2],individual,atol=2e-6,rtol=2e-6)
        reverse=model.read_targets(encoded,targets.flip(1))[0].flip(1)
        torch.testing.assert_close(out[2],reverse,atol=2e-6,rtol=2e-6)
        extra=model.read_targets(encoded,torch.cat((targets,torch.randn(1,4,3)),1))[0][:,:5]
        torch.testing.assert_close(out[2],extra,atol=2e-6,rtol=2e-6)
        perm=torch.tensor([2,0,1])
        perm_out=model(raw[:,perm],coords[:,perm],stations[:,perm],targets,valid,mask[:,perm])
        for a,b in zip(out,perm_out):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        no_data=torch.zeros_like(stations)
        with pytest.raises(ValueError,match='no_valid'):model(raw,coords,no_data,targets,valid,mask)


def test_real_audit_contract_with_synthetic_encoder_restores_state(cfg,dataset,tmp_path):
    cfg,lock=effective(cfg,tmp_path)
    row=catalog(cfg,'train').iloc[0]
    sample=prepare(read_event(row,cfg,3),cfg,3)
    model=make_model(cfg)
    model.encoder.norm.num_batches_tracked.fill_(23)
    initial=state_digest(model.state_dict());rng=rng_state()
    result=engine.audit_model(model,cfg,sample,'cpu')
    assert result['module_gradient_norms']['amplitude']>0
    assert state_digest(model.state_dict())==initial
    assert torch.equal(rng['torch'],torch.get_rng_state())
    model.train();assert not model.encoder.training
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert parameter_audit(model)['trainable']==740592


def test_source_loss_once_and_mdn_export(cfg):
    cfg['target_normalization']=dict(mean=-1.5,std=.4)
    mag=mdn(torch.randn(2,3,requires_grad=True),1,1)
    loc=mdn(torch.randn(2,21,requires_grad=True),3,3)
    pga=mdn(torch.randn(2,5,9,requires_grad=True),3,1)
    labels=[torch.randn(2,1),torch.randn(2,3),torch.randn(2,5,1)]
    valid=torch.ones(2,5,dtype=torch.bool)
    value,terms=loss_terms([mag,loc,pga],labels,valid,cfg)
    # Repeating the targets leaves event-level source weighting unchanged.
    repeated=loss_terms([mag,loc,pga.repeat(1,3,1,1)],labels[:2]+[labels[2].repeat(1,3,1)],valid.repeat(1,3),cfg)[1]
    for name in ['location_raw','magnitude_raw','location_weighted','magnitude_weighted']:
        assert torch.equal(terms[name],repeated[name])
    value.backward();assert pga.grad_fn is not None
    w,mu,sigma=decode(pga.detach().numpy(),cfg['target_normalization'])
    score=score_rows(labels[2][...,0].numpy(),w,mu,sigma)
    # Change of variable of density adds log(std) in SI log10 coordinates.
    from fe02_lite.losses import nll
    direct=nll(pga,(labels[2]+1.5)/.4,1).detach().numpy()+np.log(.4)
    np.testing.assert_allclose(score['nll'],direct,atol=1e-5,rtol=1e-5)
    assert np.all(score['q025']<score['q975'])


def test_evaluation_analysis_and_bounded_diagnostics(cfg,dataset,tmp_path):
    cfg,lock=effective(cfg,tmp_path)
    model=make_model(cfg)
    identity=dict(checkpoint_epoch=0,checkpoint_step=0,checkpoint_sha256='synthetic',
        **{k:'SYNTHETIC' for k in ['data_shards_sha256','split_manifest_sha256','view_manifest_sha256',
                                 'cohort_sha256','encoder_checkpoint_sha256','encoder_yaml_sha256']})
    selection,frame=evaluate(model,cfg,tmp_path/'eval',identity,'cpu')
    assert np.isfinite(selection)
    sources=read_table(tmp_path/'eval/sources')
    assert not sources.observation_id.duplicated().any()
    assert len(frame)>len(sources)
    recomputed=analysis.scored(tmp_path/'eval')
    np.testing.assert_allclose(recomputed.prediction,frame.prediction)
    analysis.summarize_run(tmp_path/'eval',tmp_path/'analysis',draws=10)
    assert (tmp_path/'analysis/spatial_fields.csv').exists()
    paired=frame.copy();paired['variant']='amp16'
    writer=TableWriter(tmp_path/'paired/predictions')
    for record in paired.to_dict('records'):writer.append(record)
    writer.close()
    analysis.compare([tmp_path/'eval',tmp_path/'paired'],tmp_path/'comparison',draws=10)
    comparison=pd.read_csv(tmp_path/'comparison/paired_event_bootstrap.csv')
    assert len(comparison)==1 and comparison.delta_mae.iloc[0]==0
    changed=paired.copy();changed['absolute_cutoff_utc']+=.01
    writer=TableWriter(tmp_path/'different_time/predictions')
    for record in changed.to_dict('records'):writer.append(record)
    writer.close()
    analysis.compare([tmp_path/'eval',tmp_path/'different_time'],tmp_path/'unpaired',draws=10)
    assert 'refused' in pd.read_csv(tmp_path/'unpaired/pairing_support.csv').kind.iloc[0]
    state=rng_state();initial=state_digest(model.state_dict())
    export_diagnostics(model,cfg,tmp_path/'audit_data',tmp_path/'diag','init',dict(run_id='synthetic'),'cpu')
    assert state_digest(model.state_dict())==initial
    assert torch.equal(state['torch'],torch.get_rng_state())
    saved=list((tmp_path/'diag').glob('*.npz'));assert saved
    skipped=save_diagnostic(tmp_path/'tiny','x',{'x':torch.ones(100)},dict(test=True),cap_bytes=1)
    assert skipped['status']=='skipped'
    assert not list((tmp_path/'tiny').glob('*.npz'))
    physical=np.array([[36,139,23]],np.float32)
    transformed=transform_coordinates(physical,cfg,source=True)
    np.testing.assert_allclose(transform_coordinates(transformed,cfg,source=True,inverse=True),physical,atol=1e-6)


def test_variant_init_and_parameter_counts(cfg,tmp_path):
    baseline=make_model(cfg)
    for variant,changes in VARIANTS.items():
        current=copy.deepcopy(cfg);current['architecture'].update(changes)
        model=make_model(current)
        assert model.projection.out_features+model.amplitude.out_channels==current['architecture']['model_dim']
        for name,value in model.task_state().items():
            ref=baseline.task_state()[name]
            if value.shape==ref.shape:assert torch.equal(value,ref),name
    custom=tmp_path/'independent_loss.json'
    custom.write_text(json.dumps(dict(extends=str(ROOT/'configs/fe02_lite/knet.json'),loss=dict(location=.03,magnitude=.07))))
    independent=resolve(custom)
    from fe02_lite.config import validate
    validate(independent)
    assert independent['loss']['location']!=independent['loss']['magnitude']
    disabled=resolve(ROOT/'configs/fe02_lite/ablations/no_source_aux.json')
    assert disabled['loss']['location']==disabled['loss']['magnitude']==0


def test_table_integrity_and_cpu_rng(tmp_path,monkeypatch):
    writer=TableWriter(tmp_path/'table',rows_per_shard=2)
    for i in range(5):writer.append(dict(event_id=str(i),truth=float(i),valid=True))
    writer.close();frame=read_table(tmp_path/'table')
    assert frame.event_id.tolist()==list(map(str,range(5)))
    state=rng_state();expected=torch.rand(5);restore_rng(state)
    assert torch.equal(expected,torch.rand(5))
    class Relocated(torch.Tensor):
        @property
        def device(self):return torch.device('cuda:0')
        def cpu(self):return self.as_subclass(torch.Tensor).clone()
    byte_states=[torch.Generator().manual_seed(seed).get_state() for seed in (42,43)]
    state['device']=[s.as_subclass(Relocated) for s in byte_states]
    observed=[]
    def fake_device_restore(states):
        assert all(s.device.type=='cpu' and s.dtype==torch.uint8 for s in states)
        observed.extend(states)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',fake_device_restore)
    restore_rng(state)
    assert all(torch.equal(a,b) for a,b in zip(byte_states,observed))
    path=tmp_path/'table/part-00000.npz'
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash'):read_table(tmp_path/'table')


def test_submit_mapping_dry_run_and_shells(cfg,tmp_path):
    env=dict(os.environ,FE02_LITE_LOGIN_PYTHON=os.sys.executable)
    command=['bash',str(ROOT/'scripts/fe02_lite/submit_all.sh'),'--dry-run']
    result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    plan=json.JSONDecoder().raw_decode(result.stdout)[0]
    assert [r['dataset_view'] for r in plan['tasks']]==['knet','kiknet_borehole','kiknet_surface']
    assert all(r['seed']==42 and r['variant']=='main_d128' for r in plan['tasks'])
    assert not Path(cfg['output_root']).exists()
    for view in ['knet','kiknet_borehole','kiknet_surface']:
        result=subprocess.run(['bash',str(ROOT/f'scripts/fe02_lite/submit_{view}.sh'),'--dry-run'],env=env,text=True,capture_output=True,check=True)
        assert json.JSONDecoder().raw_decode(result.stdout)[0]['tasks'][0]['dataset_view']==view
    for script in (ROOT/'scripts/fe02_lite').glob('*.sh'):
        subprocess.run(['bash','-n',str(script)],check=True)
    subprocess.run(['bash','-n',str(ROOT/'fe02_lite_cluster.env.example')],check=True)


def test_legacy_files_unchanged():
    paths=['fe01','fe02','gemini_models.py','train_light.py','loader_light.py','gemini_util_light.py','configs/fe02','scripts/fe02','pga_configs']
    result=subprocess.run(['git','diff','618661f','--',*paths],cwd=ROOT,text=True,capture_output=True,check=True)
    assert not result.stdout


def test_training_resume_exact_state_and_committed_journals(cfg,dataset,tmp_path,monkeypatch):
    cfg['diagnostics']['enabled']=False
    cfg['training'].update(epochs=2,global_batch=2,microbatch=1,max_updates=None)
    cfg['realtime']['draws_per_event']=1
    cfg['logging']['module_frequency']=1
    audit_dir=tmp_path/'protocol'
    data=audit_data(cfg,audit_dir/'data')
    effective_cfg=copy.deepcopy(cfg);effective_cfg.update(data['effective_config'])
    model=make_model(effective_cfg)
    lock=dict(status='AUDIT_PASS',config_sha256=fingerprint(cfg),code_sha256=code_identity(),
        data=data,encoder_identity=model.encoder_identity,task_initial_sha256=state_digest(model.task_state()),
        effective_config=data['effective_config'])
    write_json(audit_dir/'protocol.lock.json',lock)
    actual_check=engine.check_lock
    monkeypatch.setattr(engine,'check_lock',lambda config,path: lock)
    planned=pd.read_csv(audit_dir/'data/validation_population.csv.gz',dtype={'event_id':str,'station_id':str})
    def cheap_validation(model,config,directory,identity,device,times):
        Path(directory).mkdir(parents=True,exist_ok=False)
        return float(sum(p.detach().sum() for p in model.parameters() if p.requires_grad)),planned.copy()
    monkeypatch.setattr(engine,'evaluate',cheap_validation)
    cfg_a=copy.deepcopy(cfg);cfg_a['output_root']=str(tmp_path/'straight')
    straight=engine.train(cfg_a,audit_dir,model_factory=make_model)
    expected=torch.load(straight/'last.pth',map_location='cpu',weights_only=False)
    cfg_b=copy.deepcopy(cfg);cfg_b['output_root']=str(tmp_path/'resumed')
    calls=[]
    def interrupt_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('synthetic interruption before second epoch publication')
        return cheap_validation(*args,**kwargs)
    monkeypatch.setattr(engine,'evaluate',interrupt_second)
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        engine.train(cfg_b,audit_dir,model_factory=make_model)
    partial=Path(cfg_b['output_root'])/cfg_b['run_id']
    checkpoint=torch.load(partial/'last.pth',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==1
    assert checkpoint['sampling_position']['next_epoch']==1
    assert not any(n.startswith('encoder.') for n in checkpoint['task_state'])
    monkeypatch.setattr(engine,'evaluate',cheap_validation)
    resumed=engine.train(cfg_b,audit_dir,resume=True,model_factory=make_model)
    got=torch.load(resumed/'last.pth',map_location='cpu',weights_only=False)
    assert got['epoch']==expected['epoch']==2 and got['step']==expected['step']==2
    assert got['scheduler']==expected['scheduler']
    for name,value in got['task_state'].items():torch.testing.assert_close(value,expected['task_state'][name],atol=0,rtol=0)
    for key,value in got['optimizer']['state'].items():
        for name,tensor in value.items():torch.testing.assert_close(tensor,expected['optimizer']['state'][key][name],atol=0,rtol=0)
    assert torch.equal(got['rank_rng'][0]['torch'],expected['rank_rng'][0]['torch'])
    # Interrupted second-epoch files remain; last references only committed attempts.
    assert len(list((resumed/'sample_journals').glob('*.gz')))==3
    assert len([r for r in got['committed_files'] if 'sample_journals' in r['path']])==2
    wrong=copy.deepcopy(cfg_b);wrong['training']['lr']=.02
    with pytest.raises(ValueError,match='identity changed'):
        engine.train(wrong,audit_dir,resume=True,model_factory=make_model)


def test_submit_snapshot_and_fake_sbatch_only(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('lite_submit',ROOT/'scripts/fe02_lite/submit.py')
    submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
    source=tmp_path/'source with spaces';source.mkdir()
    members={'scripts/fe02_lite/worker.sh':'#!/bin/sh\nexit 0\n',
             'vendor/dtbench/training/modeling.py':'# synthetic dependency only\n',
             'configs/fe02_lite/knet.json':'{}\n'}
    for name,content in members.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    hashes={name:submit.digest(source/name) for name in members}
    (source/'SOURCE_IDENTITY.json').write_text(json.dumps(dict(commit='f'*40,files_sha256=hashes)))
    monkeypatch.setattr(submit,'ROOT',source)
    bindir=tmp_path/'bin';bindir.mkdir();stub=bindir/'sbatch'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$FE02_LITE_TEST_CAPTURE"\nprintf "99123\\n"\n')
    stub.chmod(0o755)
    monkeypatch.setenv('PATH',str(bindir)+':'+os.environ['PATH'])
    values=dict(OUTPUT_ROOT=str(tmp_path/'out with spaces'),DATA_ROOT='/synthetic data',SPLIT_MANIFEST='/synthetic split',
        DITING_CHECKPOINT='/synthetic checkpoint',NODES='1',DEVICES_PER_NODE='1',CONCURRENCY='1',
        CPUS_PER_TASK='2',PARTITION='synthetic',GRES_KIND='dcu',WALLTIME='00:05:00',TEST_CAPTURE=str(tmp_path/'captured'))
    for key,value in values.items():monkeypatch.setenv('FE02_LITE_'+key,value)
    monkeypatch.setattr(os.sys,'argv',['submit.py','--dataset-view','all'])
    submit.main()
    launch=next((Path(values['OUTPUT_ROOT'])/'launches').iterdir())
    assert (launch/'sbatch_receipt.txt').read_text().splitlines()==['99123']*3
    tasks=json.loads((launch/'tasks.json').read_text())
    assert len(tasks)==3 and tasks[2]['dataset_view']=='kiknet_surface'
    capture=(tmp_path/'captured').read_text()
    assert all('--array='+str(i)+'-'+str(i)+'%1' in capture for i in range(3))
    assert capture.count('--dependency=afterany:99123')==2
    assert '--job-name=fe02-lite-kiknet_surface-formal-main_d128-s42' in capture
    original=(launch/'source/configs/fe02_lite/knet.json').read_text()
    (source/'configs/fe02_lite/knet.json').write_text('changed after queueing')
    assert (launch/'source/configs/fe02_lite/knet.json').read_text()==original
    assert 'out with spaces' in (launch/'launch.env').read_text()
    # Missing/wrong launch snapshot hashes fail before environment/model/data loading.
    failure=subprocess.run(['bash',str(ROOT/'scripts/fe02_lite/worker.sh'),str(launch/'launch.env'),'wrong','wrong'],
        env=dict(os.environ,SLURM_JOB_ID='99123'),capture_output=True,text=True)
    assert failure.returncode!=0 and 'environment changed' in failure.stderr
