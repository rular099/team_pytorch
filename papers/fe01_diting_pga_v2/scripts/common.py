"""Paper-only utilities. Read pinned Git blobs; never import scientific model code."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import warnings

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
BASE = '560a29d8c82ba634549ed886840b2a2a539e0114'
RESULT = 'b366434d2cceb55d188576ce1084f571e09da097'
MAIN = '50672e131d876a604432b283162d5b0189d40cf0'
HIST = '29ffb20ba0effc0c345a0329ba3f8c658dc8cef0'
ROOT = 'reports/fe01_v4_hpc_results_20261115/'
FAMILIES = ['DiTing','PhaseNet','EQT','TEAM']
EPOCHS = dict(DiTing=12,PhaseNet=12,EQT=10,TEAM=11)
COLORS = dict(DiTing='#2166ac',PhaseNet='#d97706',EQT='#238443',TEAM='#8c51a5')
MARKERS = dict(DiTing='o',PhaseNet='s',EQT='^',TEAM='D')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def git_bytes(commit, path):
    return subprocess.check_output(['git','-C',str(REPO),'show',commit+':'+path])


class Evidence:
    def __init__(self):
        self.path = PAPER/'evidence/input_manifest.json'
        self.records = json.loads(self.path.read_text())['files'] if self.path.exists() else []

    def data(self, path, commit=RESULT, use='scientific_input', parsing='bytes'):
        data = git_bytes(commit,path)
        blob = subprocess.check_output(['git','-C',str(REPO),'rev-parse',commit+':'+path],text=True).strip()
        row = dict(repository='rular099/team_pytorch',full_commit=commit,path=path,git_blob_sha=blob,
            actual_sha256=sha(data),bytes=len(data),read_method='git show pinned commit:path',
            representation='repository bytes; archived metadata may be explicitly redacted',use=use,parsing=parsing)
        for existing in self.records:
            if existing['full_commit']==commit and existing['path']==path:
                existing['parsing']=','.join(sorted(set(existing['parsing'].split(','))|{parsing}))
                return data
        self.records.append(row)
        return data

    def local(self,path,label,use='user_instruction'):
        data=Path(path).read_bytes()
        row=dict(repository=None,full_commit=None,path=label,git_blob_sha=None,actual_sha256=sha(data),bytes=len(data),
            read_method='explicitly supplied local file',representation='original supplied bytes',use=use,parsing='UTF-8 or JSON')
        if not any(v['path']==label and v['full_commit'] is None for v in self.records):self.records.append(row)
        return data

    def text(self,path,commit=RESULT,use='method_or_context'):
        return self.data(path,commit,use,'UTF-8').decode()

    def json(self,path,commit=RESULT,use='scientific_metadata'):
        return json.loads(self.data(path,commit,use,'JSON'))

    def csv(self,path,commit=RESULT,**kwargs):
        data=self.data(path,commit,'saved_numeric_table','CSV'+('/gzip' if path.endswith('.gz') else ''))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message="The 'float_precision' argument is deprecated.*")
            return pd.read_csv(io.BytesIO(data),compression='gzip' if path.endswith('.gz') else None,
                float_precision='round_trip',dtype={'dataset_id':str,'event_id':str},**kwargs)

    def eid(self,eid):
        index=json.loads((PAPER/'evidence/blueprint_v2/05_EVIDENCE_INDEX.json').read_text())
        row=next(v for v in index['evidence'] if v['id']==eid)
        path,commit=row['path'],row['commit']
        if path.endswith(('.csv','.csv.gz')):return self.csv(path,commit)
        if path.endswith('.json'):return self.json(path,commit)
        return self.text(path,commit)

    def save(self):
        write_json(self.path,dict(repository='rular099/team_pytorch',paper_base_commit=BASE,
            fixed_result_commit=RESULT,files=sorted(self.records,key=lambda v:(v['full_commit'] or '',v['path']))))


def unique(frame, **keys):
    part=frame
    for key,value in keys.items():part=part[part[key].eq(value)]
    if len(part)!=1:raise ValueError(('Expected unique source row',keys,len(part)))
    return part.iloc[0]


def md_table(frame, digits=4):
    def cell(x):
        if pd.isna(x):return '未提供'
        if isinstance(x,float):return f'{x:.{digits}f}'
        return str(x).replace('|','/')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)
