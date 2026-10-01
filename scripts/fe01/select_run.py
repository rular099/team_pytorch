#!/usr/bin/env python
"""0-based stage selection; primary groups only unless explicitly requested."""
import argparse
import csv
from pathlib import Path

def select(root,stage,index,controls=False):
    with (Path(root)/'configs/fe01/runs.tsv').open() as stream:
        rows=[r for r in csv.DictReader(stream,delimiter='\t') if r['stage']==stage and (controls or r['main']=='True')]
    if index<0 or index>=len(rows):
        raise ValueError(f'Array index {index} outside 0..{len(rows)-1}')
    return rows[index]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--stage',choices=['pilot','formal','spatial'],required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--controls',action='store_true')
    a=p.parse_args();r=select(a.root,a.stage,a.index,a.controls)
    print(str(Path(a.root)/r['config']));print(r['run_id'])
