#!/usr/bin/env python
"""Archive the exact clean implementation commit plus the dtbench dependency."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import datetime

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--dtbench-root',required=True,help='Directory containing dtbench/ and LICENSE')
    p.add_argument('--weights-root',help='Optionally include registered offline assets; DiTing may be large')
    a=p.parse_args();output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']).strip():
        raise ValueError('Commit FE01 changes before creating a source bundle')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only','-z',commit]).decode().split('\0')
    excluded=[];selected=[]
    for name in filter(None,names):
        path=Path(name)
        if (name.startswith('reports/') and not name.startswith('reports/fe01_local_20261001/')) or name.startswith('ppt_figures/') or path.suffix in {'.npz','.pt','.pth','.hdf5','.h5','.zip','.gz','.pptx','.png','.pdf'}:
            excluded.append(name)
        else: selected.append(name)
    archive=subprocess.check_output(['git','-C',str(ROOT),'archive','--format=tar',commit,'--',*selected])
    files={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isfile(): files[item.name]=(source.extractfile(item).read(),item.mode)
    dependency=Path(a.dtbench_root).resolve()
    if not (dependency/'dtbench/training/modeling.py').is_file() or not (dependency/'LICENSE').is_file():
        raise FileNotFoundError('dtbench package and license required')
    for path in sorted((dependency/'dtbench').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.yaml','.yml','.json'):
            files['vendor/'+str(path.relative_to(dependency))]=(path.read_bytes(),0o644)
    files['vendor/DITINGBENCH_LICENSE']=( (dependency/'LICENSE').read_bytes(),0o644)
    if a.weights_root:
        weights=Path(a.weights_root)
        manifest=json.loads((weights/'pretrained_manifest.json').read_text())
        files['offline_weights/pretrained_manifest.json']=((weights/'pretrained_manifest.json').read_bytes(),0o644)
        for model in manifest['models'].values():
            for record in model['files'].values():
                relative=Path(record['path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Use a portable relative registered weight path before bundling')
                data=(weights/relative).read_bytes()
                if hashlib.sha256(data).hexdigest()!=record['sha256']: raise ValueError('Weight hash mismatch')
                files['offline_weights/'+str(relative)]=(data,0o644)
    hashes={name:hashlib.sha256(data).hexdigest() for name,(data,_) in files.items()}
    identity=dict(commit=commit,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files_sha256=hashes,excluded_historical_artifacts=excluded,
        dependency='dtbench source snapshot, hashed per file; not assumed to be the historical RT55 installation')
    files['SOURCE_IDENTITY.json']=(json.dumps(identity,indent=2).encode(),0o644)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as destination:
        for name,(data,mode) in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=mode
            destination.addfile(item,io.BytesIO(data))
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(json.dumps(dict(commit=commit,archive=str(output),sha256=digest,files=len(files))))

if __name__=='__main__': main()
