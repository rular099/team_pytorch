"""Lightweight review exports; never include models or raw waveforms."""
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from fe01.config import sha256
from .provenance import source_identity, require, write_json


def redact(value):
    if isinstance(value,dict):
        return {redact(k):redact(v) for k,v in value.items()}
    if isinstance(value,list): return [redact(v) for v in value]
    if isinstance(value,str) and (value.startswith('/public/') or value.startswith('/home/')):
        return 'PRIVATE_PATH/'+Path(value).name
    return value


def review_package(root):
    root=Path(root)
    target=root/'review.tar.gz'
    require(not target.exists(),'Review package exists; preserve it, choose a new batch ID')
    files=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(p in ('.cache','traces','sample_journals') for p in path.parts):continue
        if path.suffix in ('.json','.csv','.gz','.png','.md','.tsv') and not path.name.endswith('.tar.gz'):
            files.append(path)
    manifest={}
    with tarfile.open(target,'w:gz') as archive:
        for path in files:
            data=path.read_bytes()
            if path.suffix=='.json':data=(json.dumps(redact(json.loads(data)),ensure_ascii=False,indent=2)+'\n').encode()
            name=str(path.relative_to(root))
            entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644;entry.mtime=0
            archive.addfile(entry,io.BytesIO(data));manifest[name]=hashlib.sha256(data).hexdigest()
        data=(json.dumps(dict(source=source_identity(),files_sha256=manifest),indent=2)+'\n').encode()
        entry=tarfile.TarInfo('A01_REVIEW_MANIFEST.json');entry.size=len(data);entry.mtime=0
        archive.addfile(entry,io.BytesIO(data))
    digest=sha256(target)
    (root/'review.tar.gz.sha256').write_text(digest+'  review.tar.gz\n')
    print('REVIEW_PACKAGE',target,digest,flush=True)
