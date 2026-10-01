#!/usr/bin/env python
"""Read-only dependency/backend check; no dependency upgrades or model download."""
import argparse
import contextlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import fe01

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-rocm',action='store_true');a=p.parse_args()
    if sys.version_info<(3,9): raise RuntimeError('SeisBench 0.10.1 needs Python >=3.9')
    packages=['torch','seisbench','numpy','scipy','h5py','pandas','yaml','matplotlib','timm','mup','einops','PIL']
    versions={}
    for name in packages:
        with contextlib.redirect_stdout(sys.stderr): importlib.import_module(name)
        distribution={'yaml':'PyYAML','PIL':'pillow'}.get(name,name)
        versions[name]=importlib.metadata.version(distribution)
    if versions['seisbench']!='0.10.1' or versions['mup']!='1.0.0': raise ValueError('Pinned SeisBench/mup version mismatch')
    with contextlib.redirect_stdout(sys.stderr): import dtbench.training.modeling
    import torch
    if a.require_rocm and (not torch.cuda.is_available() or not torch.version.hip):
        raise RuntimeError('Requested DCU device requires the cluster working ROCm torch build')
    print(json.dumps(dict(python=sys.version,versions=versions,rocm=torch.version.hip,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',status='ENV_PASS'),indent=2))

if __name__=='__main__': main()
