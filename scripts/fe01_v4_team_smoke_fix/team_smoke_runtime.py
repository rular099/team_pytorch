"""Allocated DCU worker: original runner/gates plus scoped repeatability controls."""
import argparse
import importlib.metadata
import os
from pathlib import Path
from fix_support import (TASK, operational_identity, require, validate_plan, write,
                         bind_smoke_callback, controlled_smoke, digest)


def run(session):
    import torch
    from fe01_a01_compare import orchestration as ops
    from fe01_a01_compare.io import attempt
    from fe01_a01_compare import replay
    from fe01_a01.equivalence import paired_execution
    require(bool(os.environ.get('SLURM_STEP_ID')) and bool(os.environ.get('SLURM_JOB_ID')),
            'Run only inside the allocated srun step')
    require(os.environ.get('SLURM_ARRAY_TASK_ID') == str(TASK), 'Only task 3 is allowed')
    print('ON_V4_TEAM_SMOKE_DEVICE', torch.__version__, torch.version.hip, torch.cuda.is_available(), flush=True)
    require(torch.cuda.is_available(), 'DCU unavailable inside allocated srun step')
    plan = ops.verify_plan(session)
    validate_plan(plan, os.environ['ON_EVAL_RELEASE_ROOT'])
    revision = operational_identity()
    ops.prerequisites(session, 'model_smoke')
    if ops.completed(session, 'model_smoke', TASK):
        print('ALREADY_COMPLETE')
        return
    identity = dict(ops.stage_identity(session, 'model_smoke', TASK),
                    runtime_revision_sha256=revision['revision_sha256'], runtime_revision_commit=revision['commit'])
    with attempt(session, 'model_smoke', TASK, identity) as out:
        if out is None:
            return
        print('TEAM_SMOKE_ATTEMPT=' + str(out), flush=True)
        versions = {}
        for name in ('numpy', 'pandas', 'scipy', 'h5py', 'torch', 'seisbench'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'NOT_INSTALLED'
        write(out / 'runtime_environment.json', dict(python=__import__('sys').version, versions=versions,
              private_env_file_sha256=digest(plan['env_file']), loaded_modules=os.environ.get('LOADEDMODULES', 'UNKNOWN'),
              resources={k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_ARRAY_TASK_ID',
                                                        'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE')}))
        write(out / 'operational_execution.json', dict(revision=revision, device='cuda',
              original_runner_bytecode_unchanged=True, private_namespace_override=['verify_smoke'],
              backend_policy='original A01 paired_execution plus strict deterministic algorithms (warn_only=False), scoped to TEAM smoke gates; all settings restored after each audit',
              tolerance='unchanged original absolute 1e-5; no relative pass tolerance',
              production_inference_and_latency='unchanged original implementation/defaults',
              python_executable=__import__('sys').executable, torch=torch.__version__, rocm=torch.version.hip,
              device_name=torch.cuda.get_device_name(0)))
        callback = controlled_smoke(replay.verify_smoke, paired_execution, torch, out / 'smoke_diagnostics')
        runner = bind_smoke_callback(replay.runner_task, callback)
        runner(session, plan['tasks']['model_smoke'][TASK], 'model_smoke', out, 'cuda')
        print('TEAM_MODEL_SMOKE_PASS_WITH_OPERATIONAL_REVISION', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True)
    a = p.parse_args()
    run(Path(a.session))


if __name__ == '__main__':
    main()
