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
        A01_CPUS_PER_TASK=2,A01_GRES='dcu:4',A01_SINGLE_GRES='dcu:1',A01_TRAIN_TIME='24:00:00',A01_GATE_TIME='08:00:00')
    import shlex
    private.write_text(''.join('export '+k+'='+shlex.quote(str(v))+'\n' for k,v in values.items()))
    fake=tmp/'bin';fake.mkdir();log=tmp/'scheduler.log'
    for name in ('sbatch','srun'):
        script=fake/name
        script.write_text('#!/bin/bash\nprintf "%s\\n" "'+name+' $*" >> "$FAKE_SCHEDULER_LOG"\nprintf "12345\\n"\n')
        script.chmod(0o755)
    env={**os.environ,'PATH':str(fake)+':'+os.environ['PATH'],'A01_ENV_FILE':str(private),'FAKE_SCHEDULER_LOG':str(log)}
    return env,log


def test_prepare_never_submits_and_explicit_entry_submits_one_stage(tmp_path):
    env,log=settings(tmp_path)
    result=subprocess.run(['bash','scripts/fe01_a01/prepare.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not log.exists()
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
