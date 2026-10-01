#!/usr/bin/env python
"""Freeze non-standard validation times and geographic holdouts before predictions."""
import argparse
from pathlib import Path
import sys

import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import load
from fe01.data import split_catalog
from fe01.sampling import TimeSampler
from fe01.spatial import make_holdout

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['random-times','spatial'])
    p.add_argument('--config');p.add_argument('--station-catalog');p.add_argument('--output',required=True)
    a=p.parse_args()
    path=Path(a.output)
    if path.exists():
        raise FileExistsError('Manifest already frozen')
    path.parent.mkdir(parents=True,exist_ok=True)
    if a.action=='spatial':
        frame=make_holdout(pd.read_csv(a.station_catalog,dtype={'station_id':str}))
    else:
        cfg=load(a.config);sampler=TimeSampler();rows=[]
        for _,event in split_catalog(cfg,'val').iterrows():
            for draw in range(3):
                attempt=0
                while True:
                    t=sampler.sample(event.dataset_id,event.event_id,0,draw*1000+attempt,42,'random-validation')
                    if t not in cfg['realtime']['fixed_times']:
                        break
                    attempt+=1
                rows.append(dict(dataset_id=event.dataset_id,event_id=event.event_id,draw=draw,elapsed_time=t))
        frame=pd.DataFrame(rows)
    frame.to_csv(path,index=False)
if __name__=='__main__':
    main()
