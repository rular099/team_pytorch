#!/usr/bin/env python3
"""User-invoked, one-stage sbatch submission; no model/data processing here."""
import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fe01_review.provenance import read_json, write_json, require, expand, sha256, evaluation_identity

STAGES = {
    'identity': ('identity_job.sbatch', ()),
    'verify': ('evaluate_job.sbatch', ('verify',)),
    'legacy_fixed': ('evaluate_job.sbatch', ('evaluate', 'fixed')),
    'random': ('evaluate_job.sbatch', ('evaluate', 'random')),
    'long': ('evaluate_job.sbatch', ('evaluate', 'long')),
    'reference': ('reference_job.sbatch', ()),
    'analyze': ('analyze_job.sbatch', ()),
    'replay': ('replay_job.sbatch', ()),
    'pack': ('pack_job.sbatch', ()),
}
GPU_STAGES = {'verify', 'legacy_fixed', 'random', 'long', 'replay'}


def completed(directory, filename):
    require(not (directory / 'failure.json').exists(), 'Previous stage failed: ' + str(directory))
    require((directory / filename).is_file() and (directory / 'artifact_manifest.sha256').is_file(),
            'Previous stage is unfinished: ' + str(directory))
    return read_json(directory / filename)


def eligible(stage, model):
    if stage == 'legacy_fixed':
        return model['kind'] in ('rt55', 'rt61')
    if stage == 'long':
        return model['kind'] in ('rt55', 'rt61') or model.get('model_family') in ('team_original_scratch', 'eqt_pretrained_frozen')
    if stage == 'replay':
        return model['kind'] in ('rt55', 'rt61') or model.get('seed') == 42
    return True


def selected_models(stage, root, models, plan):
    selected = []
    inventory = {row['run_id']: row for row in read_json(root / 'identity/checkpoint_inventory.json')}
    module_sha = evaluation_identity()['evaluation_module_sha']
    for index, model in enumerate(models):
        if not eligible(stage, model):
            continue
        reason = None
        if model.get('inventory_status') != 'IDENTITY_PASS':
            reason = inventory.get(model['run_id'], {}).get('selected', {}).get('reason', 'checkpoint identity is BLOCKED')
        elif stage != 'verify':
            directory = root / 'verification' / model['run_id']
            try:
                gate = completed(directory, 'verification.json')
                require(gate.get('status') == 'PASS', 'forward verification is not PASS')
                require(gate.get('run_id') == model['run_id'] and gate.get('checkpoint_epoch') == model['epoch'],
                        'verification model/epoch differs')
                require(gate.get('checkpoint_sha256') == model['checkpoint_sha256'], 'verification checkpoint differs')
                require(gate.get('evaluation_module_sha') == module_sha, 'verification source is stale; use a new run ID')
                require(gate.get('request_manifest_sha256') == sha256(plan / 'fixed_requests.csv.gz'), 'verification requests differ')
            except (ValueError, OSError) as exc:
                reason = str(exc)
        if reason:
            print('SKIP', model['run_id'], reason, flush=True)
        else:
            selected.append(index)
            print('MODEL', model['run_id'], 'epoch=' + str(model['epoch']), flush=True)
    require(selected, 'No eligible models passed the required gates; no job submitted')
    return selected


def output_paths(stage, root, models, indices):
    if stage == 'verify':
        return [root / 'verification' / models[i]['run_id'] for i in indices]
    if stage == 'replay':
        return [root / 'replay' / models[i]['run_id'] for i in indices]
    if stage in ('legacy_fixed', 'random', 'long'):
        scope = 'fixed' if stage == 'legacy_fixed' else stage
        return [root / 'evaluations' / models[i]['run_id'] / scope for i in indices]
    return [root / {'pack': 'return_package', 'analyze': 'analysis'}.get(stage, stage)]


def submit(stage):
    env = os.environ
    code = Path(env['FE01_CODE_ROOT']).resolve()
    root = (Path(env['FE01_REVIEW_OUTPUT_ROOT']) / env['REVIEW_RUN_ID']).resolve()
    protected = Path(env['FE01_TRAIN_OUTPUT_ROOT']).resolve()
    plan = Path(env['FE01_REVIEW_PLAN']).resolve()
    env_file = Path(env['FE01_REVIEW_ENV']).resolve()
    require(root != protected and protected not in root.parents and root not in protected.parents,
            'New output overlaps original training directory')
    require(shutil.which('sbatch'), 'sbatch unavailable; run submit.sh on the supercomputer login node')
    source_models = expand(read_json(env['FE01_REVIEW_MANIFEST']))['models']
    settings = dict(env_sha256=sha256(env_file), models=source_models,
                    fixed_request_sha256=sha256(plan / 'fixed_requests.csv.gz'),
                    random_draw_sha256=sha256(plan / 'random_time_draws.csv'),
                    evaluation_module_sha=evaluation_identity()['evaluation_module_sha'])
    manifest_path = root / 'jobs_manifest.json'
    if not manifest_path.exists():
        require(stage == 'identity', 'First submit identity')
        require(not root.exists() or not any(root.iterdir()), 'Nonempty review directory; use a new REVIEW_RUN_ID')
        root.mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(exist_ok=True)
    # Serialize invocations and retain submission intent before contacting Slurm.
    with (root / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = read_json(manifest_path) if manifest_path.exists() else dict(
            **settings, split='val', submission='user-invoked submit.sh', scheduler_called=False, jobs=[])
        require(all(record.get(k) == v for k, v in settings.items()), 'Submission settings changed; use a new REVIEW_RUN_ID')
        require(not any(j['stage'] == stage for j in record['jobs']), 'Stage already requested: ' + stage + '; inspect jobs_manifest.json')
        require(all(j['execution_status'] == 'SUBMITTED' for j in record['jobs']),
                'Previous submission is unconfirmed; inspect jobs_manifest.json and Slurm before continuing')
        require(not any(j['stage'] == 'pack' for j in record['jobs']), 'Pack already requested; this run accepts no more submissions')
        models, indices = source_models, []
        if stage != 'identity':
            frozen = completed(root / 'identity', 'provenance.json')
            requests = completed(root / 'requests', 'provenance.json')
            require(requests.get('hydrated') is True, 'CPU request planning is incomplete')
            require(frozen.get('evaluation_module_sha') == settings['evaluation_module_sha']
                    and requests.get('evaluation_module_sha') == settings['evaluation_module_sha'], 'CPU evidence source is stale')
            models = read_json(root / 'identity/frozen_checkpoint_manifest.json')['models']
            require([m['run_id'] for m in models] == [m['run_id'] for m in source_models], 'Frozen model order differs')
        if stage in GPU_STAGES:
            indices = selected_models(stage, root, models, plan)
        if stage == 'analyze':
            completed(root / 'reference', 'train_only_reference.json')
        for path in output_paths(stage, root, models, indices):
            require(not path.exists(), 'Existing output will not be overwritten: ' + str(path))
        cap = int(env.get('FE01_ARRAY_CONCURRENCY', '8'))
        require(1 <= cap <= 16, 'Concurrency must be 1..16')
        command = ['sbatch', '--parsable', '--chdir=' + str(code), '--partition=' + env['FE01_PARTITION'],
                   '--job-name=fe01e1_' + stage, '--export=ALL,FE01_REVIEW_ENV=' + str(env_file),
                   '--output=' + str(root / 'logs/%x_%A_%a.out'), '--error=' + str(root / 'logs/%x_%A_%a.err')]
        if env.get('FE01_ACCOUNT'):
            command.append('--account=' + env['FE01_ACCOUNT'])
        if indices:
            command.append('--array=' + ','.join(map(str, indices)) + '%' + str(min(cap, len(indices))))
        parents = {'verify': ['identity'], 'reference': ['identity'], 'analyze': ['reference', 'verify'],
                   'legacy_fixed': ['verify'], 'random': ['verify'], 'long': ['verify'], 'replay': ['verify', 'reference']}
        upstream = [j['job_id'] for j in record['jobs'] if j.get('job_id') and
                    (stage == 'pack' or j['stage'] in parents.get(stage, []))]
        if upstream:
            # Gate files above establish success per model. An array's failed
            # member must not block the valid members' downstream jobs.
            command.append('--dependency=afterany:' + ':'.join(upstream))
        script, args = STAGES[stage]
        command.extend([str(code / 'scripts/fe01_review' / script), *args])
        job = dict(stage=stage, command=command, model_ids=indices, run_ids=[models[i]['run_id'] for i in indices],
                   execution_status='SUBMISSION_REQUESTED', dependency_job_ids=upstream)
        record['jobs'].append(job)
        record['scheduler_called'] = True
        write_json(manifest_path, record)
        result = subprocess.run(command, cwd=code, text=True, capture_output=True)
        job.update(scheduler_stdout=result.stdout.strip(), scheduler_stderr=result.stderr.strip(), returncode=result.returncode)
        match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', result.stdout.strip()) if result.returncode == 0 else None
        job['execution_status'] = 'SUBMITTED' if match else 'SUBMISSION_UNCONFIRMED'
        if match:
            job['job_id'] = match.group(1)
        write_json(manifest_path, record)
        require(match, 'Submission not confirmed; inspect jobs_manifest.json before retrying: ' + result.stderr.strip())
        print('SUBMITTED', stage, 'JobID=' + job['job_id'], flush=True)
        print('Logs:', root / 'logs')
        print('Status: squeue -j ' + job['job_id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    args = parser.parse_args()
    try:
        submit(args.stage)
    except (ValueError, OSError, KeyError) as exc:
        print('SUBMISSION_STOPPED:', str(exc), file=sys.stderr)
        sys.exit(2)
