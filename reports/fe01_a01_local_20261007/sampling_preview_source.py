"""Reproduce a metadata-only preview from frozen actual train cohorts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from fe01.config import fingerprint,sha256
from fe01.sampling import TimeSampler
from fe01_a01.identity import training_budget
from fe01.windows import CAPABILITIES
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
records=[]
for family in ('team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen','diting_pretrained_frozen'):
    original=family if family!='diting_pretrained_frozen' else 'team_original_scratch'
    path=Path(a.metadata)/f'formal__{original}__seed42/resolved_config.json'
    cfg=json.loads(path.read_text());cap=CAPABILITIES[family]
    sampler=TimeSampler(max_sample=cap.max_elapsed_sample(),bins=cfg['realtime']['bins'],probabilities=cfg['realtime']['probabilities'])
    cohort=sorted(cfg['audited_cohorts']['train'])[:32];rows=[]
    for dataset,event in cohort:
        for epoch in range(12):
            for draw in range(3):
                rows.append(dict(dataset_id=dataset,event_id=event,epoch=epoch,draw=draw,
                    elapsed_time=sampler.sample(dataset,event,epoch,draw,42),geometry_stream='mixed',sampling_seed=42))
    pd.DataFrame(rows).to_csv(out/(family+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
    records.append(dict(family=family,scope='first32 frozen train-event metadata; no HDF/label access; not full training journal',
        preview_events=len(cohort),preview_draws=len(rows),plan_sha256=fingerprint(rows),
        resolved_config_source_sha256=sha256(path),native_n_samples=cap.native_n_samples,
        full_budget=training_budget(len(cfg['audited_cohorts']['train']),cfg,16),sampler_audit=sampler.audit(),
        geometry_selected_ids='NOT_RUN: production HDF unavailable locally'))
(out/'preview_identity.json').write_text(json.dumps(records,indent=2)+'\n')
print('SAMPLING_METADATA_PREVIEW_PASS',len(records),'families')
