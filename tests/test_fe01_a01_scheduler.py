import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def settings(tmp):
    original=tmp/'original';original.mkdir()
    for family in ('team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen'):
        run=original/f'formal__{family}__seed42';run.mkdir()
        for name in ('resolved_config.json','protocol.lock.json','init.pth','best.pth','last.pth'):(run/name).touch()
    weights=tmp/'weights';weights.mkdir();(weights/'pretrained_manifest.json').touch()
    split=tmp/'split.csv';split.touch()
    private=tmp/'private.env'
    values=dict(A01_CODE_ROOT=ROOT,A01_OUTPUT_ROOT=tmp/'new',FE01_TRAIN_OUTPUT_ROOT=original,
        FE01_WEIGHTS_ROOT=weights,FE01_SPLIT_MANIFEST=split,A01_PARTITION='',A01_TRAIN_NODES=4,A01_DEVICES_PER_NODE=4,
        A01_CPUS_PER_TASK=2,A01_GRES='dcu:4',A01_SINGLE_GRES='dcu:1',A01_TRAIN_TIME='24:00:00',A01_GATE_TIME='08:00:00',
        FE01_CODE_ROOT=ROOT,FE01_OUTPUT_ROOT=original,DATA_ROOT=tmp/'data',FE01_SPATIAL_MANIFEST=tmp/'spatial.csv',
        A01_DITING_MANIFEST=weights/'pretrained_manifest.json')
    import shlex
    private.write_text(''.join('export '+k+'='+shlex.quote(str(v))+'\n' for k,v in values.items()))
    fake=tmp/'bin';fake.mkdir();log=tmp/'scheduler.log'
    for name in ('sbatch','srun'):
        script=fake/name
        script.write_text('#!/bin/bash\nprintf "%s\\n" "'+name+' $*" >> "$FAKE_SCHEDULER_LOG"\nprintf "12345\\n"\n')
        script.chmod(0o755)
    env={**os.environ,**{k:str(v) for k,v in values.items()},'PATH':str(fake)+':'+os.environ['PATH'],
         'A01_ENV_FILE':str(private),'FAKE_SCHEDULER_LOG':str(log)}
    return env,log


def gates(env,indices):
    from unittest.mock import patch
    from fe01.config import fingerprint,sha256,load,write_json
    from scripts.fe01_a01.preflight import current_code_sha
    root=Path(env['A01_OUTPUT_ROOT']);code=current_code_sha()
    write_json(root/'environment/environment.json',dict(status='PASS',source=dict(a01_code_sha256=code)))
    matrix=json.loads((ROOT/'configs/fe01_a01/matrix.json').read_text())['new_training']
    with patch.dict(os.environ,env):
        for index in indices:
            cfg=load(ROOT/matrix[index]['config']);audit=root/'audits'/cfg['run_id']/'protocol.lock.json'
            lock=dict(status='AUDIT_PASS',a01_code_sha256=code,config_sha256=fingerprint(cfg),
                initial_state=dict(state_sha256='synthetic',common_sha256='synthetic',encoder_sha256='synthetic'),
                sampling=dict(budget=dict(total_updates=2544)))
            write_json(audit,lock)
            for stage,file in (('diagnostics','diagnostics.json'),('pilot','pilot.json')):
                write_json(root/stage/cfg['run_id']/file,dict(status='PASS',audit_lock_sha256=sha256(audit),a01_code_sha256=code))


def test_prepare_never_submits_and_explicit_entry_submits_one_stage(tmp_path):
    env,log=settings(tmp_path)
    result=subprocess.run(['bash','scripts/fe01_a01/prepare.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not log.exists()
    gates(env,[0,1,2])
    result=subprocess.run(['bash','scripts/fe01_a01/submit.sh','train','0-2'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    lines=log.read_text().splitlines()
    assert len(lines)==1 and lines[0].startswith('sbatch ') and '--array=0-2%3' in lines[0]
    assert lines[0].endswith('job.sbatch train')
    for bad in ('test','all','resume'):
        result=subprocess.run(['bash','scripts/fe01_a01/submit.sh',bad],cwd=ROOT,env=env,capture_output=True)
        assert result.returncode!=0
    assert len(log.read_text().splitlines())==1


def test_matrix_has_exactly_five_new_seed42_runs():
    matrix=json.loads((ROOT/'configs/fe01_a01/matrix.json').read_text())
    assert [(r['family'],r['mode']) for r in matrix['new_training']]==[
        ('TEAM','OFF'),('PhaseNet','OFF'),('EQT','OFF'),('DiTing','ON'),('DiTing','OFF')]
