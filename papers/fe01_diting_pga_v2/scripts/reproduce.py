"""Full CPU-only asset pipeline. Browser printing is an explicit separate action."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
from common import *

env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
records=[]
for script in ['audit_inputs.py','build_assets.py','claim_trace.py','finalize_manifest.py','compose.py','verify.py']:
    cmd=[sys.executable,'-B',str(PAPER/'scripts'/script)]
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    r=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True)
    log=PAPER/'qa'/('reproduce_'+script.replace('.py','.log'))
    log.write_text(r.stdout+r.stderr)
    records.append(dict(command=cmd,started_utc=start,returncode=r.returncode,log=str(log.relative_to(PAPER))))
    write_json(PAPER/'qa/reproduction_commands.json',dict(commands=records,status='RUNNING' if r.returncode==0 else 'FAILED'))
    print(script,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
write_json(PAPER/'qa/reproduction_commands.json',dict(commands=records,status='PASS',browser_pdf='separate explicit step'))
