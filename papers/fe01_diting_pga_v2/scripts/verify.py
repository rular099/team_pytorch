"""Execute focused QA, re-read all pinned input hashes, and inspect paper-only diff."""
import datetime
import json
import subprocess
import sys
from common import *

commands=[([sys.executable,'-B','-m','unittest','discover','-s',str(PAPER/'tests'),'-v'],'focused_tests')]
logs=[]
for cmd,name in commands:
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=subprocess.run(cmd,capture_output=True,text=True,cwd=REPO)
    (PAPER/f'qa/{name}.log').write_text(p.stdout+p.stderr)
    logs.append(dict(command=cmd,started_utc=start,returncode=p.returncode,log=f'qa/{name}.log'))
    if p.returncode:raise SystemExit(p.stderr)

manifest=json.loads((PAPER/'evidence/input_manifest.json').read_text())
hash_checks=[]
for row in manifest['files']:
    if row['full_commit'] is None:continue
    actual=sha(git_bytes(row['full_commit'],row['path']))
    hash_checks.append(dict(commit=row['full_commit'],path=row['path'],sha256=actual,unchanged=actual==row['actual_sha256']))
assert all(r['unchanged'] for r in hash_checks)
diff=subprocess.check_output(['git','-C',str(REPO),'diff','--name-only',BASE],text=True).splitlines()
assert all(p.startswith('papers/fe01_diting_pga_v2/') for p in diff)
write_json(PAPER/'qa/input_hashes_after.json',dict(status='PASS',files=hash_checks))
ms=(PAPER/'manuscript.md').read_text();import re
body=ms.split('## 1 引言')[1].split('## 数据与代码可用性')[0]
abstract=ms.split('## 摘要')[1].split('关键词')[0]
write_json(PAPER/'qa/verification.json',dict(status='PASS',commands=logs,source_files_rehashed=len(hash_checks),
    production_diff_paths=[p for p in diff if not p.startswith('papers/')],
    manuscript_han=len(re.findall('[\u3400-\u9fff]',body)),abstract_han=len(re.findall('[\u3400-\u9fff]',abstract)),
    test_set_accessed=False,hpc_jobs_submitted=False,inference_run=False,new_bootstrap_run=False))
print(f'PASS: focused tests; {len(hash_checks)} pinned inputs unchanged; paper-only diff')
