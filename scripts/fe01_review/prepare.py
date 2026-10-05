#!/usr/bin/env python3
"""Create directories/job metadata and PRINT submission commands, never execute."""
import os
import sys
import shlex
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01_review.provenance import read_json,write_json,require,expand,sha256


def prepare():
    env=os.environ
    require(env.get('DRY_RUN','1')=='1','Only DRY_RUN=1 allowed')
    code=Path(env['FE01_CODE_ROOT']).resolve();root=Path(env['FE01_REVIEW_OUTPUT_ROOT']).resolve()/env['REVIEW_RUN_ID']
    protected=Path(env['FE01_TRAIN_OUTPUT_ROOT']).resolve()
    require(root!=protected and protected not in root.parents,'New output must not overlap training directory')
    require(not root.exists() or not any(root.iterdir()),'Review run ID already nonempty; choose a NEW run ID')
    models=expand(read_json(env['FE01_REVIEW_MANIFEST']))['models'];plan=Path(env['FE01_REVIEW_PLAN'])
    requests_sha=sha256(plan/'fixed_requests.csv.gz');random_sha=sha256(plan/'random_time_draws.csv')
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir()
    common=['sbatch','--partition='+env['FE01_PARTITION'],'--export=ALL,FE01_REVIEW_ENV='+str(Path(env['FE01_REVIEW_ENV']).resolve()),
        '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
    if env.get('FE01_ACCOUNT'):common.append('--account='+env['FE01_ACCOUNT'])
    cap=int(env.get('FE01_ARRAY_CONCURRENCY','8'));require(1<=cap<=16,'Concurrency must be1..16')
    jobs=[]
    def printed(stage,script,array=None,args=(),dependency='',scope='',request_sha='pending CPU HDF planning'):
        command=common+([] if array is None else ['--array='+array])+[str(code/'scripts/fe01_review'/script),*args]
        jobs.append(dict(stage=stage,script=script,array=array,split='val',scope=scope,depends_on=dependency,
            request_manifest_sha256=request_sha,device_required=script in ('evaluate_job.sbatch','replay_job.sbatch'),
            expected_fixed_rows=194265 if scope=='fixed' else None,expected_noninput_rows=139440 if scope=='fixed' else None,
            denominator='actual planned count; never copied from fixed-time domain' if scope!='fixed' else 'locked fixed population',
            command=shlex.join(command),execution_status='NOT_SUBMITTED_THIS_TASK'))
        print(f'\n{stage}: {dependency}\n{shlex.join(command)}')
    print('DRY_RUN=1: only commands printed. Run the stages in order AFTER inspecting the previous gates.')
    print('Model array IDs (this table is only for the NEW EVAL1 scripts):')
    for i,m in enumerate(models):print(i,m['run_id'],'fixed epoch='+str(m['epoch']))
    print('Submission working directory:',code)
    printed('identity','identity_job.sbatch',scope='CPU checkpoint inventory + shared HDF request plan',request_sha=requests_sha)
    printed('verify','evaluate_job.sbatch',f'0-{len(models)-1}%{cap}',('verify',),dependency='identity CPU job completed; source/HDF pins valid; each model independently gated',scope='<=40 metadata selected decisions',request_sha=requests_sha)
    printed('legacy_fixed','evaluate_job.sbatch',f'9,10%{cap}',('evaluate','fixed'),dependency='only legacy models with matching selected epoch AND verification PASS',scope='fixed',request_sha=requests_sha)
    printed('random','evaluate_job.sbatch',f'0-{len(models)-1}%{cap}',('evaluate','random'),dependency='each model verification PASS',scope='random 3930 original draws,3926 snapshots',request_sha=random_sha)
    printed('long','evaluate_job.sbatch',f'0,1,2,6,7,8,9,10%{cap}',('evaluate','long'),dependency='each model verification PASS; PhaseNet N/A; EQT90 N/A',scope='TEAM/legacy40,90; EQT40')
    printed('reference','reference_job.sbatch',dependency='identity data pins valid',scope='9084 train events,146299 labels once')
    printed('analyze','analyze_job.sbatch',dependency='reference completed; verification reports present',scope='existing nine CSV diagnostics + common train reference')
    printed('replay','replay_job.sbatch',f'0,3,6,9,10%{cap}',dependency='selected seed42 and legacy verification PASS; common request plan ready',scope='3 fixed cases,1..20s,normal/random,natural/fixed_S0')
    printed('pack','pack_job.sbatch',dependency='all intended evidence jobs finished or explicitly BLOCKED',scope='validate manifests,compare systems,light archive')
    write_json(root/'jobs_manifest.json',dict(models=models,fixed_request_sha256=requests_sha,random_draw_sha256=random_sha,jobs=jobs,
        scheduler_called=False,split='val',submission='manual only',account_declared=bool(env.get('FE01_ACCOUNT'))))
    print('\nRead-only job status command to run MANUALLY after submission:')
    print('sacct -j <YOUR_COMMA_SEPARATED_JOB_IDS> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P > '+shlex.quote(str(root/'sacct_status.txt')))


if __name__=='__main__':prepare()
