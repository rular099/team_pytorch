#!/usr/bin/env python
"""Independent FE02 index: seed-major, R0/A/B/C/M. No torch imports."""
import argparse
import csv
from pathlib import Path


def select(root, stage, index):
    with (Path(root)/'configs/fe02/runs.tsv').open() as stream:
        rows=[r for r in csv.DictReader(stream,delimiter='\t') if r['stage']==stage]
    if not 0<=index<len(rows): raise ValueError('FE02 index outside stage matrix')
    return rows[index]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--stage',choices=['formal','pilot'],default='formal')
    p.add_argument('--index',type=int,required=True);a=p.parse_args()
    r=select(a.root,a.stage,a.index)
    print(str(Path(a.root)/r['config']));print(r['run_id'])
