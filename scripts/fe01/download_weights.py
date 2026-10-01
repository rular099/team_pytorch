#!/usr/bin/env python
"""Explicit local-only online export, or import verified existing SeisBench cache.

Runtime train/eval never imports this downloader. The downloader records exact
versions, component order, source metadata, hashes, and unresolved data overlap.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import sha256,write_json
from fe01.extractors import SEISBENCH_VERSION


def export(output,name,version,cache=None,remote=None,diting=None):
    output=Path(output).resolve()
    if (output/'pretrained_manifest.json').exists():
        raise FileExistsError('Offline package exists; choose a new directory')
    output.mkdir(parents=True,exist_ok=True)
    os.environ['SEISBENCH_CACHE_ROOT']=str(output/'.download-cache')
    import seisbench
    import seisbench.models as sbm
    if importlib.metadata.version('seisbench')!=SEISBENCH_VERSION:
        raise ValueError('Unexpected SeisBench version')
    if remote:
        # Explicit repository choice, recorded; never silently switch weights.
        seisbench.use_backup_repository(remote)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  seisbench_version=SEISBENCH_VERSION,models={},available_pretrained={},
                  overlap_status='unknown; STEAD contains global stations; Japan overlap requires corpus audit',
                  repository=seisbench.remote_root)
    for family,cls,n in [('phasenet',sbm.PhaseNet,3001),('eqtransformer',sbm.EQTransformer,6000)]:
        if cache is not None:
            source=Path(cache)/'models/v3'/family
            manifest['available_pretrained'][family]=sorted({p.name.split('.')[0] for p in source.glob('*.json.v*')})
        else:
            available=cls.list_pretrained(remote=True)
            manifest['available_pretrained'][family]=available
            if name not in available:
                raise ValueError(f'{name} not advertised for {family}: {available}')
            if version not in cls.list_versions(name,remote=True):
                raise ValueError(f'Explicit version {version} unavailable for {family}')
            cls.from_pretrained(name,version_str=version,update=False)
            source=cls._model_path()
        target=output/family
        target.mkdir()
        paths={}
        for field,ext in [('weights','pt'),('metadata','json')]:
            original=source/f'{name}.{ext}.v{version}'
            if not original.is_file():
                raise FileNotFoundError(original)
            destination=target/f'{name}.{ext}.v{version}'
            shutil.copyfile(original,destination)
            paths[field]=dict(path=str(destination.relative_to(output)),sha256=sha256(destination),bytes=destination.stat().st_size)
        metadata=json.loads((output/paths['metadata']['path']).read_text())
        manifest['models'][family]=dict(files=paths,weight_name=name,weight_version=version,
            seisbench_version=SEISBENCH_VERSION,native_n_samples=n,sampling_rate=100,
            component_order=metadata.get('model_args',{}).get('component_order','ZNE'),
            pretrained_metadata=metadata,license='SeisBench MIT; individual weight/data terms require metadata audit',
            normalized_input_difference='FE01 common masked joint-peak; original preprocessing recorded in metadata',
            source_dataset=name,overlap_status='unknown; no no-leakage claim')
    if diting:
        path=Path(diting).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest['models']['diting']=dict(files={'weights':dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)},
                                        source_dataset='requires user provenance audit',overlap_status='unknown')
    else:
        manifest['missing_diting']='User must register actual MAE 1200M checkpoint before DiTing audit/train'
    write_json(output/'pretrained_manifest.json',manifest)
    print(output/'pretrained_manifest.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--name',default='stead')
    p.add_argument('--version',default='2')
    p.add_argument('--existing-cache')
    p.add_argument('--remote',help='Explicit official repository URL; optional documented GFZ mirror')
    p.add_argument('--diting-checkpoint')
    a=p.parse_args()
    if a.version=='latest':
        p.error('An exact weight version is required')
    export(a.output,a.name,a.version,a.existing_cache,a.remote,a.diting_checkpoint)

if __name__=='__main__':
    main()
