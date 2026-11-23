#!/usr/bin/env python
"""Collect the latest fully sealed best/last evaluation of explicitly found runs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe02_lite.analysis import compare

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',required=True)
p.add_argument('--output',required=True)
p.add_argument('--checkpoint',choices=['best','last'],default='best')
p.add_argument('--bootstrap-draws',type=int,default=1000)
a=p.parse_args()
paths=[]
for run in sorted(Path(a.root).glob('fe02_lite__*')):
    found=[d for d in (run/'evaluations').glob(a.checkpoint+'_*') if
           (d/'predictions/manifest.json').is_file() and (d/'analysis/summary.json').is_file()]
    if found: paths.append(sorted(found)[-1])
if not paths: raise ValueError('No complete sealed evaluations; no invented summary')
compare(paths,a.output,a.bootstrap_draws)
print(json.dumps(dict(evaluations=list(map(str,paths)),output=a.output)))
