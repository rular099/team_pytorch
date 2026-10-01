#!/usr/bin/env python
"""Package review evidence, hash external large outputs, exclude checkpoints/raw data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-file-mb',type=float,default=10)
    a=p.parse_args();output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    files={};records=[]
    allowed={'.json','.jsonl','.csv','.gz','.npz','.tsv','.md','.png','.pdf','.gif','.txt','.sha256'}
    for index,root in enumerate(map(Path,a.roots)):
        if not root.is_dir(): raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix in {'.pth','.pt','.hdf5','.h5'}: continue
            if 'sample_journals' in path.parts or '.cache' in path.parts: continue
            name=f'{index:02d}_{root.name}/'+str(path.relative_to(root))
            size=path.stat().st_size
            included=path.suffix in allowed and size<=a.max_file_mb*1024**2
            records.append(dict(path=name,bytes=size,sha256=sha256(path),included=included,
                                external_source=str(path) if not included else None))
            if included: files[name]=path.read_bytes()
    missing={}
    for name in ['protocol.lock.json','resolved_config.json','data_split_audit.json','pretrained_manifest.json',
        'model_interface_audit.json','model_time_support.csv','sampler_distribution_audit.json',
        'causal_boundary_audit.json','random_validation_times_manifest.csv','test_exposure_ledger.json',
        'actual_training_time_histograms.csv','training_curves.csv','paired_event_bootstrap.csv','efficiency.csv',
        'site_residuals.csv','station_counts.csv','spatial_holdout_manifest.csv','case_selection_manifest.json',
        'replay_latency.csv','replay_metrics_by_time.csv']:
        missing[name]='present' if any(Path(r['path']).name==name for r in records) else 'NOT_PROVIDED'
    files['evidence_inventory.json']=json.dumps(dict(artifacts=records,required_evidence=missing,
        note='Missing files stay missing; no inferred HPC success. Large outputs retain external hashes.'),indent=2).encode()
    files['artifact_manifest.sha256']=''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name,data in sorted(files.items())).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as archive:
        for name,data in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    output.with_suffix(output.suffix+'.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    print(output)

if __name__=='__main__': main()
