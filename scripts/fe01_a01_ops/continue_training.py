#!/usr/bin/env python3
"""Operations only: stdlib login checks; allocated-rank communication/resume.

This file is deliberately outside the authenticated A01 training source.
It calls the existing engine.train(resume=True); it does not change that loop.
"""
import argparse
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def core():
    root = Path(os.environ['A01_CODE_ROOT']).resolve()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('a01_original_preflight', root/'scripts/fe01_a01/preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


def config(index):
    root, pf = core()
    matrix = pf.read(root/'configs/fe01_a01/matrix.json')['new_training']
    pf.require(0 <= index < len(matrix), 'Invalid A01 index')
    return pf.load(root/matrix[index]['config'])


def plan(indices):
    """Retain every original prerequisite; replace only the empty-run rule."""
    root, pf = core()
    output = Path(os.environ['A01_OUTPUT_ROOT']).resolve()
    code_sha = pf.current_code_sha()
    env = pf.read(output/'environment/environment.json')
    pf.require(env['status'] == 'PASS' and env['source']['a01_code_sha256'] == code_sha,
               'Matching environment gate required')
    rows = []
    for index in indices:
        cfg = config(index)
        pf.require(Path(cfg['output_root']).resolve() == output, 'Config/output root mismatch')
        audit = output/'audits'/cfg['run_id']/'protocol.lock.json'
        lock = pf.read(audit)
        pf.require(lock['status'] == 'AUDIT_PASS' and lock['a01_code_sha256'] == code_sha
                   and lock['config_sha256'] == pf.fingerprint(cfg), 'Matching audit gate required')
        if cfg['model_family'] == 'diting_pretrained_frozen':
            other_cfg = config(4 if index == 3 else 3)
            other = pf.read(output/'audits'/other_cfg['run_id']/'protocol.lock.json')
            pf.require(other['status'] == 'AUDIT_PASS' and other['a01_code_sha256'] == code_sha
                       and other['config_sha256'] == pf.fingerprint(other_cfg), 'Matched DiTing audit required')
            pf.require(other['sampling'] == lock['sampling'], 'DiTing sampling differs')
            for key in ('state_sha256', 'common_sha256', 'encoder_sha256'):
                pf.require(other['initial_state'][key] == lock['initial_state'][key], 'DiTing init differs: '+key)
        for stage, name in (('diagnostics', 'diagnostics.json'), ('pilot', 'pilot.json')):
            gate = pf.read(output/stage/cfg['run_id']/name)
            pf.require(gate['status'] == 'PASS' and gate['audit_lock_sha256'] == pf.sha256(audit)
                       and gate['a01_code_sha256'] == code_sha, 'Matching '+stage+' gate required')
        run = output/cfg['run_id']
        curves = []
        if (run/'training_curves.csv').is_file():
            with (run/'training_curves.csv').open() as stream:
                curves = list(csv.DictReader(stream))
        if curves:
            pf.require([int(r['epoch']) for r in curves] == list(range(1, len(curves)+1)), 'Noncontiguous curves')
            pf.require(int(curves[-1]['epoch']) < cfg['training']['epochs'], 'Completed run; will not retrain: '+cfg['run_id'])
        last = run/'last.pth'
        nonempty = run.exists() and any(run.iterdir())
        pf.require(last.is_file() or not nonempty,
                   'Nonempty run without last.pth; preserve it for inspection: '+str(run))
        rows.append(dict(index=index, run_id=cfg['run_id'], action='resume' if last.is_file() else 'fresh',
                         curve_epoch=int(curves[-1]['epoch']) if curves else None,
                         checkpoint=str(last) if last.is_file() else None))
    return rows


def seconds(value):
    if value in ('UNLIMITED', 'INFINITE'):
        return float('inf')
    days, sep, rest = value.partition('-')
    days, rest = (int(days), rest) if sep else (0, days)
    parts = [int(x) for x in rest.split(':')]
    if len(parts) == 3:
        hours, minutes, secs = parts
    elif len(parts) == 2:
        hours, minutes, secs = (parts[0], parts[1], 0) if sep else (0, parts[0], parts[1])
    elif len(parts) == 1:
        hours, minutes, secs = (parts[0], 0, 0) if sep else (0, parts[0], 0)
    else:
        raise ValueError('Invalid Slurm time')
    result = days*86400 + hours*3600 + minutes*60 + secs
    if result <= 0 or minutes >= 60 or secs >= 60:
        raise ValueError('Invalid Slurm time')
    return result


def walltime():
    """Aim for 72h, cap by the visible partition limit; unknown -> old limit."""
    explicit = os.environ.get('A01_RECOVERY_TIME')
    if explicit:
        seconds(explicit)
        return explicit, 'explicit A01_RECOVERY_TIME; scheduler/account limits still apply'
    try:
        result = subprocess.run(['scontrol', 'show', 'partition', '-o'], text=True, capture_output=True, timeout=10, check=True)
        partitions = [dict(token.split('=', 1) for token in line.split() if '=' in token) for line in result.stdout.splitlines()]
        chosen = os.environ.get('A01_PARTITION', '')
        matches = [p for p in partitions if p.get('PartitionName') == chosen] if chosen else [p for p in partitions if p.get('Default') == 'YES']
        if len(matches) != 1:
            raise ValueError('Cannot identify one partition')
        p = matches[0]
        limit = min(72*3600, seconds(p['MaxTime']))
        hours, rem = divmod(int(limit), 3600)
        minutes, secs = divmod(rem, 60)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}', 'partition='+p['PartitionName']+' MaxTime='+p['MaxTime']+'; account/QOS limits still apply'
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        fallback = os.environ['A01_TRAIN_TIME']
        seconds(fallback)
        return fallback, 'partition limit unavailable; retaining original A01_TRAIN_TIME'


def collective():
    import datetime
    import torch
    import torch.distributed as dist
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 16 or not torch.cuda.is_available():
        raise ValueError('Audited world16 and DCU required')
    print('A01_COLLECTIVE_START '+json.dumps(dict(host=socket.gethostname(), rank=rank, local=local)), flush=True)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(seconds=120))
    try:
        print('A01_COLLECTIVE_REDUCE '+str(rank), flush=True)
        tensor = torch.tensor([rank+1.], device='cuda:'+str(local))
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        actual = tensor.item()
        if actual != world*(world+1)/2:
            raise ValueError('Collective sum mismatch')
        dist.barrier(device_ids=[local])
        print('A01_COLLECTIVE_PASS '+json.dumps(dict(host=socket.gethostname(), rank=rank, sum=actual)), flush=True)
    finally:
        dist.destroy_process_group()


def checkpoint_gate(payload, cfg, lock, world):
    """CPU-loaded checkpoint metadata; tensor load and initial/data gates are core's."""
    from fe01.config import fingerprint
    effective = copy.deepcopy(cfg)
    effective.update(lock['effective_config'])
    def require(ok, message):
        if not ok:
            raise ValueError('Resume checkpoint: '+message)
    require(payload['identity'] == fingerprint(dict(config=effective, audit_lock=lock, world_size=world)), 'identity differs')
    require(payload['config'] == effective and payload['source_config_sha256'] == fingerprint(cfg), 'config differs')
    require(payload['a01_code_sha256'] == lock['a01_code_sha256'], 'source differs')
    require(payload['world_size'] == world == lock['world_size'], 'world differs')
    require(payload['absolute_amplitude_mode'] == effective['absolute_amplitude_mode'], 'amplitude differs')
    require(payload['encoder_state_sha256'] == lock['initial_state']['encoder_sha256'], 'encoder differs')
    epoch, updates = payload['epoch'], payload['updates']
    budget = lock['sampling']['budget']
    require(isinstance(epoch, int) and 0 < epoch < effective['training']['epochs'], 'completed/invalid epoch')
    require(updates == epoch*budget['updates_per_epoch'], 'update budget differs')
    curves = payload['curve_records']
    require([r['epoch'] for r in curves] == list(range(1, epoch+1)) and curves[-1]['updates'] == updates, 'curves differ')
    require(len(payload['rank_rng']) == world and all(all(k in r for k in ('rng', 'python_rng', 'numpy_rng', 'cuda_rng')) for r in payload['rank_rng']), 'per-rank RNG missing')
    require(all(k in payload for k in ('optimizer', 'scheduler', 'committed_journals', 'common_initial_state_sha256', 'model_state_dict', 'best_epoch', 'best_selection_score')), 'training state missing')
    return epoch


def train_one(index):
    import torch
    from fe01_a01 import engine
    from fe01.config import sha256, write_json
    cfg = config(index)
    row = plan([index])[0]
    root = Path(cfg['output_root'])
    audit = root/'audits'/cfg['run_id']
    lock = engine.check_audit(cfg, audit)
    engine.check_matched_pair(cfg, lock)
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != lock['world_size']:
        raise ValueError('Training world differs from audit')
    if rank == 0:
        operation = Path(os.environ['A01_OPERATION_DIRECTORY'])
        if row['action'] == 'resume':
            last = Path(row['checkpoint'])
            payload = torch.load(last, map_location='cpu', weights_only=False)
            epoch = checkpoint_gate(payload, cfg, lock, int(os.environ['WORLD_SIZE']))
            backup = operation/('checkpoint_before_'+str(index))
            backup.mkdir()
            records = {}
            for name in ('last.pth', 'best.pth', 'training_curves.csv', 'resolved_config.json', 'protocol.lock.json'):
                src = last.parent/name
                if not src.is_file():
                    continue
                target = backup/name
                if src.suffix == '.pth':
                    try:
                        os.link(src, target)  # core publishes new files via atomic replace
                    except OSError:
                        shutil.copy2(src, target)
                else:
                    shutil.copy2(src, target)
                records[name] = dict(sha256=sha256(target), bytes=target.stat().st_size)
            write_json(backup/'manifest.json', dict(index=index, checkpoint_epoch=epoch, files=records))
            del payload
            print('A01_RESUME_FROM_EPOCH='+str(epoch)+' index='+str(index), flush=True)
        else:
            print('A01_FRESH_FROM_AUDITED_INIT index='+str(index), flush=True)
        write_json(operation/('train_action_'+str(index)+'.json'), row)
    # The original engine restores model/Adam/scheduler/rank RNG and journals.
    engine.train(cfg, audit, resume=row['action'] == 'resume')


def supervise(index):
    """Bound startup of the *training* group too, using its first new journal."""
    cfg = config(index)
    rank = int(os.environ['RANK'])
    journal = Path(cfg['output_root'])/cfg['run_id']/'sample_journals'
    pattern = '*_rank'+str(rank)+'.jsonl'
    previous = set(journal.glob(pattern))
    limit = float(os.environ.get('A01_STARTUP_TIMEOUT_SECONDS', '1200'))
    if not 0 < limit <= 3600:
        raise ValueError('Startup timeout must be positive and at most one hour')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'train', '--index', str(index)])
    started = time.monotonic()
    old_handlers = {}
    def forward(signum, frame):
        child.send_signal(signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, forward)
    try:
        while child.poll() is None:
            current = set(journal.glob(pattern))-previous
            if any(p.stat().st_size > 0 for p in current):
                print(f'A01_TRAIN_STARTED index={index} rank={rank}', flush=True)
                return child.wait()
            if time.monotonic()-started >= limit:
                print(f'A01_TRAIN_STARTUP_TIMEOUT index={index} rank={rank} seconds={limit}; preserved all outputs', file=sys.stderr, flush=True)
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()
                return 124
            time.sleep(min(5., limit/4))
        return child.returncode
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['plan', 'walltime', 'collective', 'train', 'supervise'])
    parser.add_argument('--indices', default='3,4')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.mode == 'plan':
        values = args.indices.split(',')
        if not values or any(v not in ('0', '1', '2', '3', '4') for v in values) or len(set(values)) != len(values):
            raise ValueError('Invalid/duplicate indices')
        print(json.dumps(plan([int(v) for v in values]), ensure_ascii=False, indent=2))
    elif args.mode == 'walltime':
        value, reason = walltime()
        print('A01_CONTINUE_TIME='+value+' '+reason, file=sys.stderr)
        print(value)
    elif args.mode == 'collective':
        collective()
    else:
        if args.index is None:
            raise ValueError('index required')
        core()
        if args.mode == 'supervise':
            raise SystemExit(supervise(args.index))
        train_one(args.index)


if __name__ == '__main__':
    main()
