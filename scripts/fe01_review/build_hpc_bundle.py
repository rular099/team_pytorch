#!/usr/bin/env python3
"""Local overlay packager; never includes frozen training modules or weights."""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01_review.provenance import ROOT,sha256,training_identity,evaluation_identity,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--env',required=True)
    p.add_argument('--legacy-inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);require(not output.exists(),'Bundle already exists; use fresh name')
    files={}
    for directory in ('fe01_review','scripts/fe01_review','configs/fe01_review'):
        for f in (ROOT/directory).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:files[str(f.relative_to(ROOT))]=f
    for name in ('FE01_EVAL1_PROTOCOL.md','FE01_EVAL1_HPC_RUNBOOK.md','CODEX_RESULT_20261004_FE01_EVAL1_PREP.md'):
        f=ROOT/'docs/ai'/name
        if f.is_file():files[str(f.relative_to(ROOT))]=f
    f=ROOT/'reports/fe01_hpc_results_20261004/run_summary.csv';files[str(f.relative_to(ROOT))]=f
    for f in Path(a.plan).iterdir():
        if f.is_file():files['artifacts/fe01/eval1_frozen_plan/'+f.name]=f
    files['scripts/fe01_review/cluster_zb.env']=Path(a.env)
    for f in Path(a.legacy_inputs).iterdir():
        if f.is_file():files['artifacts/fe01/eval1_inputs/'+f.name]=f
    old=training_identity();new=evaluation_identity()
    manifest=''.join(f'{sha256(f)}  {name}\n' for name,f in sorted(files.items()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as t:
        for name,f in sorted(files.items()):t.add(f,arcname=name)
        for name,data in [('FE01_REVIEW_SOURCE_IDENTITY.json',json.dumps(dict(training=old,evaluation=new),indent=2).encode()),
            ('artifacts/fe01/eval1_overlay.sha256',manifest.encode())]:
            info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    Path(str(output)+'.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    print(output,output.stat().st_size,sha256(output))


if __name__=='__main__':main()
