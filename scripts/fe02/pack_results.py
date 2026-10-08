#!/usr/bin/env python
"""Package FE02 evidence and Slurm logs; no checkpoint/waveform payloads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import sha256


def pack(roots, output, max_file_mb=20):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    files={};records=[]
    allowed={'.json','.csv','.gz','.npz','.tsv','.md','.txt','.out','.err','.sha256'}
    for index,root in enumerate(map(Path,roots)):
        if not root.is_dir(): raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix not in allowed or 'sample_journals' in path.parts: continue
            name=f'{index:02d}_{root.name}/'+str(path.relative_to(root))
            included=path.stat().st_size<=max_file_mb*1024**2
            records.append(dict(path=name,bytes=path.stat().st_size,sha256=sha256(path),included=included,
                external_source=str(path) if not included else None))
            if included:files[name]=path.read_bytes()
    required=['protocol.lock.json','resolved_config.json','pretrained_manifest.json','model_interface_audit.json',
              'freeze_gradient_audit.json','query_independence_audit.json','causal_boundary_audit.json',
              'training_curves.csv','provenance.json','runs_manifest.csv','headline_primary.csv',
              'metrics_time_geometry_count_target.csv','spatial_fields.csv','paired_event_bootstrap.csv']
    files['evidence_inventory.json']=json.dumps(dict(artifacts=records,
        required={name:'present' if any(Path(r['path']).name==name for r in records) else 'NOT_PROVIDED' for name in required},
        note='No weights/waves/journals included. Missing outcomes stay missing. Source logs do not replace sacct.'),indent=2).encode()
    files['artifact_manifest.sha256']=''.join(hashlib.sha256(v).hexdigest()+'  '+n+'\n' for n,v in sorted(files.items())).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as archive:
        for name,data in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    output.with_suffix(output.suffix+'.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-file-mb',type=float,default=20)
    a=p.parse_args();pack(a.roots,a.output,a.max_file_mb)
