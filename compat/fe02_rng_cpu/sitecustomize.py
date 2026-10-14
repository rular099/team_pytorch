"""Opt-in, hash-pinned CPU ByteTensor restore for the existing FE02 checkpoint.

Only training ranks opt in. Original source, protocol locks, checkpoints, and
random-state bytes are not rewritten. No import of torch on login/helper paths.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import sys


def cpu_byte_states(states):
    import torch
    result = []
    for state in states:
        if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
            raise TypeError('CUDA RNG restore requires one-dimensional uint8 tensors; refusing to reseed/cast')
        result.append(state.detach().cpu().contiguous())
    return result


def install():
    # Fail closed before touching torch if a queued task sees changed shim bytes.
    shim_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if shim_sha != os.environ.get('FE02_RNG_COMPAT_SHA256'):
        raise SystemExit('FE02 RNG compatibility file changed or hash was not pinned')
    import torch
    original = torch.cuda.set_rng_state_all
    if getattr(original, '_fe02_cpu_rng_restore', False):
        return
    calls = 0

    @functools.wraps(original)
    def restore(states):
        nonlocal calls
        normalized = cpu_byte_states(states)
        result = original(normalized)
        calls += 1
        # Separate provenance from the unchanged original training/audit locks.
        receipt = dict(fix='fe02-cuda-rng-cpu-byte-v1', shim_sha256=shim_sha,
            launcher_sha256=os.environ.get('FE02_RNG_LAUNCHER_SHA256'),
            original_source_commit='9f70371838ec31044e559096a939f023d2924f37',
            torch_version=torch.__version__, rank=int(os.environ['RANK']),
            job_id=os.environ.get('SLURM_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
            state_bytes_preserved=True, restore_device='cpu',
            states=[dict(bytes=s.numel(), sha256=hashlib.sha256(bytes(s.tolist())).hexdigest())
                    for s in normalized])
        folder = Path(os.environ['FE02_OUTPUT_ROOT']) / 'compatibility_receipts'
        folder.mkdir(parents=True, exist_ok=True)
        name = 'rng_cpu_{}_{}_rank{}_call{}.json'.format(
            receipt['job_id'], receipt['array_index'], receipt['rank'], calls)
        with (folder / name).open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        print('[FE02-RNG-CPU] restored unchanged RNG bytes on CPU; rank={} shim={}'.format(
            receipt['rank'], shim_sha), file=sys.stderr, flush=True)
        return result

    restore._fe02_cpu_rng_restore = True
    torch.cuda.set_rng_state_all = restore


if os.environ.get('FE02_RNG_CPU_COMPAT') == '1' and 'RANK' in os.environ:
    install()
