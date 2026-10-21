#!/usr/bin/env python3
"""Verify transport and every manifest member; extract regular files safely."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify_extract(archive, destination):
    archive=Path(archive).resolve();destination=Path(destination).resolve()
    expected=Path(str(archive)+'.sha256').read_text().split()
    if len(expected)!=2 or expected[1].lstrip('*')!=archive.name:
        raise ValueError('Invalid archive checksum sidecar')
    actual=sha256(archive)
    if actual!=expected[0]:raise ValueError('Archive SHA256 mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    records={}
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            name=PurePosixPath(member.name)
            if (not member.isreg() or name.is_absolute() or '..' in name.parts or
                    str(name)!=member.name or member.name in records):
                raise ValueError('Unsafe/duplicate/nonregular archive member: '+member.name)
            target=destination.joinpath(*name.parts);target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256();size=0
            with stream.extractfile(member) as source,target.open('xb') as output:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    digest.update(block);size+=len(block);output.write(block)
            if size!=member.size:raise ValueError('Truncated archive member: '+member.name)
            records[member.name]=dict(bytes=size,sha256=digest.hexdigest())
    manifest=json.loads((destination/'A01_REVIEW_MANIFEST.json').read_text())
    expected_members=manifest['files_sha256']
    if set(records)-{'A01_REVIEW_MANIFEST.json'}!=set(expected_members):
        raise ValueError('Manifest member set mismatch')
    for name,digest in expected_members.items():
        if records[name]['sha256']!=digest:raise ValueError('Manifest SHA256 mismatch: '+name)
    result=dict(status='PASS',archive_sha256=actual,archive_bytes=archive.stat().st_size,
                regular_members=len(records),manifest_members=len(expected_members),
                extracted_bytes=sum(row['bytes'] for row in records.values()),files=records,
                source=manifest['source'],
                seal_scope='Archive manifest authenticates exported bytes. Original eval seals refer to original unredacted bytes.')
    path=destination.parent/'archive_verification.json'
    with path.open('x') as output:json.dump(result,output,ensure_ascii=False,indent=2);output.write('\n')
    print('ARCHIVE_VERIFIED members='+str(len(records))+' SHA256='+actual,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();verify_extract(args.source,args.destination)
