#!/usr/bin/env python3
"""Local source bundle from a clean committed A01 tree plus frozen dtbench."""
import argparse
import io
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def dependency_root():
    # dtbench is a PEP420 namespace package and can have __file__ = None.
    spec=importlib.util.find_spec('dtbench')
    if spec is None or not spec.submodule_search_locations:
        raise ValueError('Frozen dtbench source missing')
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def main():
    from fe01.config import sha256
    from fe01_a01.provenance import source_identity,require
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();target=Path(args.output)
    require(not target.exists(),'Source bundle exists; choose a new name')
    status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--','fe01_a01','scripts/fe01_a01','configs/fe01_a01','docs/ai/FE01_A01*','tests/test_fe01_a01*'])
    require(not status.strip(),'Commit A01 changes before packaging')
    identity=source_identity();identity['commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
    raw=subprocess.check_output(['git','-C',str(ROOT),'archive','HEAD'])
    dependency=dependency_root()
    target.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target,'w:gz') as archive:
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as source:
            for entry in source.getmembers():
                if not entry.isfile() or entry.name.startswith(('reports/','docs/research/')):continue
                if entry.name.endswith(('.pt','.pth','.hdf5','.zip','.tar.gz','.pdf','.png','.jpg')):continue
                data=source.extractfile(entry).read()
                entry.name='fe01_a01_source/'+entry.name;entry.mtime=0
                archive.addfile(entry,io.BytesIO(data))
        for path in sorted(dependency.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.yml','.yaml','.json'):continue
            data=path.read_bytes();name='fe01_a01_source/vendor/dtbench/dtbench/'+str(path.relative_to(dependency))
            entry=tarfile.TarInfo(name);entry.size=len(data);entry.mtime=0;archive.addfile(entry,io.BytesIO(data))
        for path in dependency.parent.glob('LICENSE*'):
            data=path.read_bytes();entry=tarfile.TarInfo('fe01_a01_source/vendor/dtbench/'+path.name)
            entry.size=len(data);entry.mtime=0;archive.addfile(entry,io.BytesIO(data))
        data=(json.dumps(identity,indent=2)+'\n').encode()
        entry=tarfile.TarInfo('fe01_a01_source/FE01_A01_SOURCE_IDENTITY.json');entry.size=len(data);entry.mtime=0
        archive.addfile(entry,io.BytesIO(data))
    digest=sha256(target);target.with_name(target.name+'.sha256').write_text(digest+'  '+target.name+'\n')
    print(json.dumps(dict(commit=identity['commit'],source_package=str(target),sha256=digest),indent=2))


if __name__=='__main__':main()
