#!/usr/bin/env python
"""Verify the uploaded source without git or torch; refuse changed/missing bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    identity=json.loads((root/'SOURCE_IDENTITY.json').read_text())
    for name,expected in identity['files_sha256'].items():
        path=root/name
        if root not in path.resolve().parents: raise ValueError('Unsafe manifest member')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Uploaded source mismatch: '+name)
    print('FE02 uploaded source verified:',identity['commit'],len(identity['files_sha256']),'files')
    return identity

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    verify(p.parse_args().root)
