"""Validate produced manifests and assemble light, reviewable return evidence."""
import os
import shutil
import tarfile
from pathlib import Path
from .provenance import require,sha256,write_json,provenance,seal


def validate_manifest(path):
    for line in path.read_text().splitlines():
        digest,name=line.split('  ',1);p=path.parent/name
        require(p.resolve().is_relative_to(path.parent.resolve()),'Artifact path escapes result root')
        require(p.is_file() and sha256(p)==digest,'Artifact SHA mismatch: '+name)


def pack(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    for path in root.rglob('artifact_manifest.sha256'):
        if output not in path.parents:validate_manifest(path)
    evidence=output/'evidence';evidence.mkdir()
    for p in sorted(root.rglob('*')):
        if not p.is_file() or output in p.parents:continue
        if any(part in ('cache','frames') for part in p.relative_to(root).parts):continue
        if p.name in ('predictions.csv.gz','station_predictions.csv.gz','grid_predictions.csv.gz','training_final_labels.csv.gz'):continue
        if p.suffix in ('.pth','.pt','.hdf5','.zip'):continue
        if p.stat().st_size>80*1024**2:continue
        target=evidence/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    if os.environ.get('FE01_TRAIN_OUTPUT_ROOT') and os.environ.get('FE01_REVIEW_PLAN'):
        from .comparison import compare_available
        compare_available(root,os.environ['FE01_TRAIN_OUTPUT_ROOT'],os.environ['FE01_REVIEW_PLAN'],evidence)
    metadata_name='package_provenance.json' if (evidence/'summary.json').exists() else 'summary.json'
    write_json(evidence/metadata_name,provenance(hpc_status='USER_PRODUCED_ARTIFACTS_ONLY',
        scheduler_status='unknown unless actual sacct_status.txt included',production_files_excluded=['weights','HDF5','full prediction CSVs','frame CSVs'],
        missing_evidence='inspect per-stage failures and comparison gates; absent files never imply PASS'))
    seal(evidence)
    archive=output/'fe01_eval1_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:handle.add(evidence,arcname='fe01_eval1_evidence')
    write_json(output/'package.json',dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive)))
    seal(output)
