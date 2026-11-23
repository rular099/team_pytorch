#!/usr/bin/env python3
"""Light results plus complete external file/hash index; never include encoder/data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--max-bytes',type=int,default=128*1024*1024)
    p.add_argument('--include-predictions',action='store_true')
    a=p.parse_args()
    root,output=Path(a.root).resolve(),Path(a.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if root==output or root in output.parents: raise ValueError('Place result archive outside the source result tree')
    files=[];selected=[];used=0
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        name=str(path.relative_to(root));size=path.stat().st_size
        forbidden=any(s in path.parts for s in ('offline_weights','source','cache')) or path.suffix in ('.h5','.hdf5')
        eligible=path.suffix in ('.json','.csv','.md','.tsv','.txt') or path.name.endswith('.csv.gz')
        if a.include_predictions and any(x in path.parts for x in ('predictions','sources','statuses')): eligible=True
        include=eligible and not forbidden and used+size<=a.max_bytes
        reason='included' if include else 'external binary/diagnostic or size cap; see original absolute path'
        if forbidden: reason='excluded data/weights/source/cache'
        files.append(dict(path=name,absolute_path=str(path),bytes=size,sha256=digest(path),included=include,reason=reason))
        if include: selected.append(path);used+=size
    index=json.dumps(dict(files=files,included_bytes=used,max_bytes=a.max_bytes,
        rule='Evaluation source tables are preserved externally even when excluded from this light package'),indent=2).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as archive:
        for path in selected: archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
        item=tarfile.TarInfo('EXTERNAL_ARTIFACT_INDEX.json');item.size=len(index)
        archive.addfile(item,io.BytesIO(index))
    output.with_suffix(output.suffix+'.sha256').write_text(digest(output)+'  '+output.name+'\n')
    print(json.dumps(dict(archive=str(output),files=len(selected),external_index=len(files),bytes=output.stat().st_size)))


if __name__=='__main__': main()
