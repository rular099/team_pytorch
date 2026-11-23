#!/usr/bin/env python
"""Reproduce focused local checks and save explicitly synthetic evidence."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TESTS=['tests/test_fe02_lite.py','tests/test_fe01_native_windows.py','tests/test_fe01_time_sampling.py',
       'tests/test_fe01_statistics.py','tests/test_fe02_analysis.py','tests/test_fe02_simple_submit.py',
       'tests/test_fe02_cluster_env.py']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    checks=[]
    commands=[[sys.executable,'-m','compileall','-q','fe02_lite','scripts/fe02_lite'],
              [sys.executable,'-m','pytest','-q',*TESTS],
              ['git','diff','--check']]
    commands += [['bash','-n',str(p)] for p in sorted((ROOT/'scripts/fe02_lite').glob('*.sh'))]
    commands += [['bash','-n','fe02_lite_cluster.env.example']]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.setdefault('MPLCONFIGDIR','/tmp/fe02-lite-local-matplotlib')
    for i,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        path=out/f'check_{i:02d}.txt'
        path.write_text(result.stdout+result.stderr)
        checks.append(dict(command=command,status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,log=path.name))
        print(json.dumps(checks[-1]),flush=True)
        if result.returncode: break
    import torch
    sys.path.insert(0,str(ROOT))
    from fe02_lite.model import LiteModel,parameter_audit
    from fe02_lite.config import VARIANTS,code_identity
    counts={}
    main_report=None
    for variant,change in VARIANTS.items():
        arch=dict(model_dim=128,amplitude_dim=8,station_pool_queries=2,attention_heads=4)
        arch.update(change)
        model=LiteModel(torch.nn.Identity(),arch)
        report=parameter_audit(model)
        counts[variant]=report['trainable']
        if variant=='main_d128': main_report=report
    (out/'parameters.json').write_text(json.dumps(dict(task_parameters=main_report,variant_counts=counts,
        encoder='NOT_LOADED; real registered/total count is a compute-node gate'),indent=2)+'\n')
    identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    legacy=['fe01','fe02','gemini_models.py','train_light.py','loader_light.py','gemini_util_light.py','configs/fe02','scripts/fe02','pga_configs']
    unchanged=not subprocess.check_output(['git','-C',str(ROOT),'diff','618661f','--',*legacy]).strip()
    report=dict(task_id='20261123-fe02-lite-event-spatial',status='PASS' if all(c['exit_code']==0 for c in checks) and unchanged else 'FAIL',
        evidence='LOCAL SYNTHETIC / MOCK ONLY; no real Japan/DiTing/ROCm performance',
        source_commit_at_check=identity,source_code_sha256=code_identity(),
        source_may_be_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']).strip()),
        python=platform.python_version(),torch=torch.__version__,
        checks=checks,legacy_source_unchanged_from_base=unchanged,
        real_data_view_audits='NOT_RUN; compute worker audit gate',real_diting_forward='NOT_RUN; compute worker gate',
        rocm_rng_restore='NOT_RUN on hardware; explicit CPU state conversion implemented and CPU replay tested',
        hpc_training='NOT_SUBMITTED',held_out_test='NOT_RUN',
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='PASS': raise SystemExit(1)


if __name__=='__main__':main()
