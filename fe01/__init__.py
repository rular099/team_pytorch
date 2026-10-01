"""Explicit FE01 v2 experiment; importing this package does not run jobs."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
dependency=Path(os.environ['DTBENCH_ROOT']) if os.environ.get('DTBENCH_ROOT') else (
    ROOT/'vendor' if (ROOT/'vendor/dtbench').is_dir() else ROOT.parent/'ditingbench')
for candidate in (dependency, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

BASE_COMMIT = '9c95dbfaf92f36b2673d816026cd3f25b91eec66'
