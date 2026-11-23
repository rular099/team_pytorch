"""Portable, bounded diagnostics and unbounded lossless evaluation exports."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .config import sha256, write_json


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                device=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    # torch.load(map_location=device) used to break ROCm ByteTensor states.
    # Always load checkpoints on CPU AND explicitly normalize every RNG tensor.
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if state['device'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['device']])


def atomic_save(path, value):
    path = Path(path)
    torch.save(value, str(path)+'.tmp')
    os.replace(str(path)+'.tmp', path)


def runtime():
    root = Path(__file__).resolve().parents[1]
    identity = root/'SOURCE_IDENTITY.json'
    if identity.exists():
        source = json.loads(identity.read_text())
        commit = source['commit']
    else:
        commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip()
    versions = {}
    for name in ['torch','numpy','scipy','pandas','h5py','PyYAML','mup','einops','deepspeed']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
    return dict(git_commit=commit, command=sys.argv, python=sys.version, pytorch=torch.__version__,versions=versions,
        numpy=np.__version__, pandas=pd.__version__, hip=torch.version.hip,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_package_sha256=sha256(identity) if identity.exists() else None,
        world_size=int(os.environ.get('WORLD_SIZE','1')),precision='fp32',amp_state=None)


class TableWriter:
    """NPZ columns contain numeric or Unicode arrays; never require pickle."""
    def __init__(self, directory, rows_per_shard=4096):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows, self.files, self.limit = [], [], rows_per_shard
        self.columns = {}

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        frame = pd.DataFrame(self.rows)
        data = {}
        for name in frame:
            series = frame[name]
            if series.dtype.kind in 'OUS':
                data[name] = series.fillna('').astype(str).to_numpy(dtype=str)
            else:
                data[name] = series.to_numpy()
        path = self.path/f'part-{len(self.files):05d}.npz'
        columns = {k:('unicode' if v.dtype.kind in 'US' else str(v.dtype)) for k,v in data.items()}
        if self.columns and columns != self.columns:
            raise ValueError('Evaluation shard schema changed')
        self.columns = columns
        np.savez_compressed(path, **data)
        self.files.append(dict(path=path.name, rows=len(frame), bytes=path.stat().st_size,sha256=sha256(path)))
        self.rows.clear()

    def close(self, metadata=None):
        self.flush()
        write_json(self.path/'manifest.json',dict(schema='fe02-lite-column-table-v1',
            files=self.files,rows=sum(f['rows'] for f in self.files),columns=self.columns,metadata=metadata or {}))


def read_table(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    frames = []
    for record in manifest['files']:
        file = path/record['path']
        if sha256(file) != record['sha256']:
            raise ValueError('Artifact hash mismatch: '+str(file))
        with np.load(file, allow_pickle=False) as data:
            frame = pd.DataFrame({k:data[k] for k in data.files})
        if len(frame) != record['rows']:
            raise ValueError('Artifact row count mismatch')
        frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def save_diagnostic(directory, tag, tensors, metadata, cap_bytes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    arrays = {k:v.detach().cpu().numpy() for k,v in tensors.items() if torch.is_tensor(v)}
    arrays = {k:(v.astype(np.float16) if v.dtype.kind=='f' else v) for k,v in arrays.items()}
    # Conservative bound, no giant in-memory compressed buffer. No native features by default.
    needed = sum(v.nbytes for v in arrays.values()) + len(json.dumps(metadata).encode()) + 65536
    entry = dict(tag=tag, precision='float16 floats / original masks',estimated_bytes=needed,
                 status='saved' if used+needed <= cap_bytes else 'skipped', reason='')
    if used+needed <= cap_bytes:
        path = directory/(tag+'.npz')
        if path.exists():
            existing = json.loads((directory/(tag+'.json')).read_text())
            if existing != metadata:
                raise ValueError('Diagnostic checkpoint/sample identity mismatch: '+tag)
            return dict(tag=tag,status='already_saved',sha256=sha256(path))
        np.savez_compressed(path, **arrays)
        write_json(directory/(tag+'.json'), metadata)
        entry.update(bytes=path.stat().st_size,sha256=sha256(path))
    else:
        entry['reason'] = 'diagnostic_storage_cap'
    with (directory/'index.jsonl').open('a') as stream:
        stream.write(json.dumps(entry)+'\n')
    return entry
