"""Read-only comparison with the startup snapshot; never modify other worktrees."""
import hashlib
import json
import subprocess
from common import *

before=json.loads((PAPER/'evidence/worktree_protection_before.json').read_text())
blocks=subprocess.check_output(['git','-C',str(REPO),'worktree','list','--porcelain'],text=True).strip().split('\n\n')
entries=[]
for block in blocks:
    values=dict(line.split(' ',1) for line in block.splitlines() if ' ' in line)
    entries.append(values)
result={}
for label,old in before.items():
    matches=[w for w in entries if w.get('branch')=='refs/heads/'+old['branch']]
    assert len(matches)==1,(label,len(matches))
    path=matches[0]['worktree']
    def git(*args):return subprocess.check_output(['git','-C',path,*args])
    current=dict(head=git('rev-parse','HEAD').decode().strip(),branch=git('branch','--show-current').decode().strip(),
        status=git('status','--short').decode(),tracked_diff_sha256=sha(git('diff','--binary')),
        staged_diff_sha256=sha(git('diff','--cached','--binary')),
        untracked=git('ls-files','--others','--exclude-standard').decode().splitlines())
    checks={k:current[k]==old[k] for k in ['head','branch','status','tracked_diff_sha256','staged_diff_sha256']}
    checks['existing_untracked_paths_preserved']=set(old['untracked']).issubset(current['untracked'])
    added=sorted(set(current['untracked'])-set(old['untracked']))
    checks['new_untracked_only_delivery']=all(p.startswith('artifacts/fe01_paper_v2_20261116/') for p in added)
    assert all(checks.values()),(label,checks)
    result[label]=dict(checks=checks,current=current,new_delivery_files=added)
write_json(PAPER/'qa/worktree_protection_after.json',dict(status='PASS',worktrees=result,
    scope='HEAD/branch/tracked diff/staged diff/status and file-list preservation; no claim that all preexisting untracked bytes were independently hashed at startup'))
print('PASS: four scientific worktrees unchanged; preexisting untracked paths preserved')
