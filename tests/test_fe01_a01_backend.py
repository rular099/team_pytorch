"""CPU canonical control, causal input identity, rank caches and submission gates."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fe01.model import build_model as legacy_build
from fe01_a01 import OFF
from fe01_a01.identity import canonical_reuse_equivalence
from test_fe01_a01_helpers import config,sample
from test_fe01_a01_scheduler import settings,gates,ROOT


@pytest.mark.parametrize('family',['team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen'])
def test_actual_registered_encoder_canonical_CPU_exact_update(tmp_path,family):
    cfg=config(tmp_path,family,OFF)
    if family!='team_original_scratch' and not Path(cfg['pretrained_manifest']).exists():
        pytest.skip('Actual offline registered weights unavailable')
    _,_,inputs=sample(tmp_path,cfg);initial=legacy_build(cfg).state_dict()
    threads=torch.get_num_threads();rng=torch.get_rng_state().clone()
    report=canonical_reuse_equivalence(cfg,cfg,initial,inputs,tmp_path/'exact.json')
    assert report['canonical_reference']['exact_state_update'] and report['update_state_equal']
    assert report['execution']['device']=='cpu'
    assert all(check['max_abs']==0 for check in report['checks'].values())
    assert torch.get_num_threads()==threads and torch.equal(rng,torch.get_rng_state())


def test_canonical_CPU_does_not_accept_even_one_ulp_update_mutation(tmp_path,monkeypatch):
    cfg=config(tmp_path);_,_,inputs=sample(tmp_path,cfg);initial=legacy_build(cfg).state_dict()
    original=torch.optim.Adam
    count=[0]
    class WrongUpdate(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);count[0]+=1;self.sequence=count[0]
        def step(self,*args,**kwargs):
            result=super().step(*args,**kwargs)
            if self.sequence==2:
                with torch.no_grad():
                    value=self.param_groups[0]['params'][0].view(-1)[0]
                    value.copy_(torch.nextafter(value,torch.full_like(value,float('inf'))))
            return result
    monkeypatch.setattr(torch.optim,'Adam',WrongUpdate)
    path=tmp_path/'exact_fail.json';threads=torch.get_num_threads()
    with pytest.raises(ValueError,match='Canonical CPU'):
        canonical_reuse_equivalence(cfg,cfg,initial,inputs,path)
    result=json.loads(path.read_text())
    assert not result['update_state_equal'] and result['status']=='FAIL'
    assert 'canonical_CPU_exact_equivalence' in result['failed_checks']
    assert torch.get_num_threads()==threads


def test_future_exact_prefix_with_small_relative_encoder_roundoff(tmp_path,monkeypatch):
    import fe01_a01.availability as module
    from fe01_a01.model import build_model
    cfg=config(tmp_path,mode=OFF);event,_,_=sample(tmp_path,cfg);original=module.snapshot;calls=[0]
    def roundoff(model,inputs):
        result=original(model,inputs);calls[0]+=1
        result['encoder_adapter']=result['encoder_adapter'].clone()
        result['encoder_adapter'].view(-1)[0]=10.+(0. if calls[0]==1 else 3.8e-6)
        return result
    monkeypatch.setattr(module,'snapshot',roundoff)
    report=module.future_audit(build_model(cfg),cfg,event,3,report_path=tmp_path/'future.json')
    for comparison in report['comparisons']:
        assert comparison['passed'] and comparison['complete_inputs_exact'] and comparison['prefix_statistics_exact']
    assert report['rows'][0]['encoder_adapter']>1e-6


def test_future_rejects_changed_complete_input_and_records_failure(tmp_path,monkeypatch):
    import fe01_a01.availability as module
    from fe01_a01.model import build_model
    cfg=config(tmp_path,mode=OFF);event,_,_=sample(tmp_path,cfg);original=module.prepare_sample;calls=[0]
    def leak(*args,**kwargs):
        result=original(*args,**kwargs);calls[0]+=1
        if calls[0]>1:
            result=copy.deepcopy(result);result['inputs'][0].view(-1)[-1]+=.01
        return result
    monkeypatch.setattr(module,'prepare_sample',leak)
    path=tmp_path/'future_failure.json'
    with pytest.raises(ValueError,match='Prefix input equality'):
        module.future_audit(build_model(cfg),cfg,event,3,report_path=path)
    assert not json.loads(path.read_text())['checks'][0]['complete_inputs_exact']


def test_sixteen_rank_import_config_caches_are_separate(tmp_path):
    script=ROOT/'scripts/fe01_a01/rank_exec.sh'
    program='import os,json,pathlib; p=pathlib.Path(os.environ["SEISBENCH_CACHE_ROOT"])/"config.json"; p.write_text(json.dumps({"rank":os.environ["RANK"]})); print(json.dumps({"rank":os.environ["RANK"],"cache":str(p.parent),"config":json.loads(p.read_text())}))'
    def run(rank):
        env={**os.environ,'A01_OUTPUT_ROOT':str(tmp_path),'SLURM_JOB_ID':'synthetic_job','SLURM_PROCID':str(rank),
             'SLURM_LOCALID':str(rank%4),'A01_WORLD_SIZE':'16'}
        result=subprocess.run(['bash',str(script),sys.executable,'-c',program],env=env,text=True,capture_output=True,check=True)
        return json.loads(result.stdout)
    with ThreadPoolExecutor(max_workers=16) as executor:results=list(executor.map(run,range(16)))
    assert len(set(value['cache'] for value in results))==16
    assert all(value['config']['rank']==value['rank'] for value in results)


def test_incomplete_stale_or_unmatched_gates_never_submit(tmp_path):
    env,log=settings(tmp_path)
    command=['bash','scripts/fe01_a01/submit.sh','train','0-2']
    result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'A01_PREFLIGHT_BLOCKED' in result.stderr and not log.exists()
    gates(env,[1])
    result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and not log.exists()
    result=subprocess.run(command[:-1]+['1'],cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert len(log.read_text().splitlines())==1
    path=Path(env['A01_OUTPUT_ROOT'])/'diagnostics/a01__phasenet_pretrained_frozen__off__seed42/diagnostics.json'
    data=json.loads(path.read_text());data['a01_code_sha256']='stale';path.write_text(json.dumps(data))
    result=subprocess.run(command[:-1]+['1'],cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and len(log.read_text().splitlines())==1
    gates(env,[3,4])
    path=Path(env['A01_OUTPUT_ROOT'])/'audits/a01__diting_pretrained_frozen__off__seed42/protocol.lock.json'
    data=json.loads(path.read_text());data['initial_state']['encoder_sha256']='wrong';path.write_text(json.dumps(data))
    result=subprocess.run(command[:-1]+['3'],cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and '初始化' in result.stderr and len(log.read_text().splitlines())==1


def test_submission_preflight_does_not_import_torch(tmp_path):
    from scripts.fe01_a01.preflight import current_code_sha
    from fe01_a01.provenance import source_identity
    assert current_code_sha()==source_identity()['a01_code_sha256']
    env,_=settings(tmp_path);gates(env,[0])
    sentinel=tmp_path/'sentinel';sentinel.mkdir();(sentinel/'torch.py').write_text('raise RuntimeError("TORCH_MUST_NOT_BE_IMPORTED")\n')
    env['PYTHONPATH']=str(sentinel)
    result=subprocess.run(['python3','scripts/fe01_a01/preflight.py','train','--indices','0','--output',env['A01_OUTPUT_ROOT']],
                          cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'A01_PREFLIGHT_PASS' in result.stdout
